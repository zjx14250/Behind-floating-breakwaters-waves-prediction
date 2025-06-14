import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
from torch.nn import ModuleList
import math


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out

# 相对位置编码
class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_relative_position=16):
        super(RelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        
        self.dropout = nn.Dropout(p=dropout)
        self.relative_position_k = RelativePosition(self.head_dim, max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, max_relative_position)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # queries, keys, values shape: [B, L, H, head_dim]
        B, L, H, head_dim = queries.shape
        S = keys.shape[1]
        
        # Self-Attention
        # reshape to [B, H, L, head_dim]
        r_q1 = queries.permute(0, 2, 1, 3)
        r_k1 = keys.permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))
        
        # Relative Position Attention
        r_q2 = queries.permute(1, 0, 2, 3).contiguous().view(L, B * H, head_dim)
        r_k2 = self.relative_position_k(L, S)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(B, H, L, S)
        
        # Combine attentions
        attn = (attn1 + attn2) / self.scale
        attn = self.dropout(torch.softmax(attn, dim=-1))
        
        # Value attention
        r_v1 = values.permute(0, 2, 1, 3)  # [B, H, S, head_dim]
        weight1 = torch.matmul(attn, r_v1)
        
        r_v2 = self.relative_position_v(L, S)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(L, B * H, S)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(B, H, L, head_dim)
        
        # Combine and reshape
        x = weight1 + weight2  # [B, H, L, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, H, head_dim]
        
        return x

# 使用示例
class RelativeAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(RelativeAttentionLayer, self).__init__()
        
        self.n_heads = n_heads
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        
        self.inner_attention = RelativeMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # 投影
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # 相对位置注意力
        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        
        # 重塑并投影回原始维度
        out = out.view(B, L, -1)
        
        return self.out_projection(out), None
    

# Activation functions for Q_MA and K_MA
def ma_q_activation(x):
    x = - x / math.sqrt(x.shape[-1])
    x = - F.leaky_relu(x, negative_slope=0.02)
    return x

def ma_k_activation(x, k=0.02):
    x = x / math.sqrt(x.shape[-1])
    x = 1 / (1 + torch.exp(-x * k))
    return x

def ma_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, return_weight=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    query = ma_q_activation(query)
    key = ma_k_activation(torch.dropout(key, dropout_p, train=True))
    attn_weight = query @ key.transpose(-2, -1)
    attn_weight = attn_weight.tril(diagonal=0)
    attn_weight = attn_weight
    if return_weight:
        return attn_weight
    return attn_weight @ value

def generate_attn_weight_from_qk(q, k, scale=True, softmax=True, diagonal=0):
    L, S = q.size(-2), k.size(-2)
    scale_factor = 1 / math.sqrt(q.size(-1)) if scale else 1
    attn_bias = torch.zeros(L, S, dtype=q.dtype)
    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=diagonal)
    attn_bias.masked_fill_(temp_mask.logical_not(), -1000)
    attn_bias.to(q.dtype)
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    if softmax:
        attn_weight += attn_bias
    else:
        attn_weight *= temp_mask.float()
    attn_weight = torch.softmax(attn_weight, dim=-1) if softmax else attn_weight
    if diagonal == -1:
        attn_weight[:, :, 0, :] = 0
    return attn_weight

class CausalSelfAttentionARMA(nn.Module):
    def __init__(self, n_embd, n_head, block_size, bias=True, dropout=0.1, ma_dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0

        self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)
        self.k2 = nn.Linear(n_embd, n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout_ma = nn.Dropout(ma_dropout)
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.ma_dropout_rate = ma_dropout
        
        # 注册因果mask并确保它在正确的设备上
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                  .view(1, 1, block_size, block_size))

    def scaled_dot_product_attention(self, q, k, v, mask=None, dropout_p=0.0):
        # 确保所有输入在同一设备上
        device = q.device
        
        # 计算注意力分数
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用因果mask
        if mask is not None:
            mask = mask.to(device)  # 确保mask在正确的设备上
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 应用softmax
        attn = torch.softmax(scores, dim=-1)
        
        # 应用dropout
        if dropout_p > 0.0:
            attn = self.attn_dropout(attn)
            
        # 计算输出
        return torch.matmul(attn, v)

    def forward(self, x):
        device = x.device  # 获取输入张量的设备
        
        # 确保所有模块都在正确的设备上
        self.c_attn = self.c_attn.to(device)
        self.k2 = self.k2.to(device)
        self.c_proj = self.c_proj.to(device)
        
        B, T, C = x.size()
        q, k = self.c_attn(x).split(self.n_embd, dim=2)
        v = x
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 使用自定义的scaled_dot_product_attention
        causal_mask = self.bias[:, :, :T, :T].to(device)  # 确保mask在正确的设备上
        y = self.scaled_dot_product_attention(
            q, k, v,
            mask=causal_mask,
            dropout_p=self.dropout if self.training else 0
        )

        e = v[:, :, 1:, :] - y[:, :, :-1, :]
        k2 = self.k2(x[:, :-1]).view(B, T-1, self.n_head, -1).transpose(1, 2)
        q2 = q[:, :, 1:, :]

        # 对MA部分也使用自定义的attention
        causal_mask_ma = self.bias[:, :, :T-1, :T-1].to(device)  # 确保mask在正确的设备上
        y2 = self.scaled_dot_product_attention(
            q2, k2, e,
            mask=causal_mask_ma,
            dropout_p=self.ma_dropout_rate if self.training else 0
        )
        
        y2 = torch.cat([torch.zeros_like(y2[:, :, :1, :]), y2], dim=2)
        
        # 重组输出
        y = self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))) + \
            self.dropout_ma(self.c_proj(y2.transpose(1, 2).contiguous().view(B, T, C)))
        
        return y

    def calculate_arma_weights(self, x, channel=None, channel_average=False, normalize=True, output_sequence=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        weights_ar = generate_attn_weight_from_qk(q, k, scale=True, softmax=normalize, diagonal=0)
        if channel is not None:
            weights_ar = weights_ar[:, channel]
            
        k2 = self.k2(x[:, :-1]).view(B, T-1, self.n_head, -1).transpose(1, 2) # (B, nh, T, hs)
        q2 = q[:, :, 1:, :]

        Beta_output = ma_scaled_dot_product_attention(q2, k2, None, attn_mask=None, dropout_p=self.ma_dropout_rate if self.training else 0, is_causal=True, return_weight=True)
        Beta = torch.zeros(Beta_output.shape[0], Beta_output.shape[1], T, T, device=x.device)
        Beta[:, :, 1:, 0:-1] = Beta_output
        if channel is not None:
            Beta = Beta[:, channel]

        if channel is not None:
            I = torch.eye(Beta.shape[-1], device=x.device).unsqueeze(0).expand(B, -1, -1)
            Beta_inverse = torch.inverse(I - Beta) # torch.linalg.pinv(I - Beta)
            weights_ma = Beta @ Beta_inverse   # Omega
        else:
            I = torch.eye(Beta.shape[-1], device=x.device).unsqueeze(0).unsqueeze(0).expand(B, Beta.shape[1], -1, -1)
            Beta_inverse = torch.inverse(I - Beta) # torch.linalg.pinv(I - Beta)
            weights_ma = torch.einsum('bdli,bdij->bdlj', Beta, Beta_inverse) # Beta @ Beta_inverse   # Omega
        if channel is None and channel_average:
            weights_ar = weights_ar.mean(dim=1)
            weights_ma = weights_ma.mean(dim=1)
            Beta = Beta.mean(dim=1)

        if not output_sequence:
            return weights_ar, weights_ma, Beta
        else:
            v = x
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            e = v[:, :, 1:, :] - y[:, :, :-1, :]
            y2 = ma_scaled_dot_product_attention(q2, k2, e, attn_mask=None, dropout_p=self.ma_dropout_rate if self.training else 0, is_causal=True)
            y2 = torch.cat([torch.zeros_like(y2[:, :, :1, :]), y2], dim=2)
            yf = self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))) + self.dropout_ma(self.c_proj(y2.transpose(1, 2).contiguous().view(B, T, C)))
            return weights_ar, weights_ma, Beta, y.transpose(1, 2).contiguous().view(B, T, C), yf    

class CausalARMAAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, attention_dropout=0.1, output_attention=False, d_model=512, n_heads=8, max_seq_len=1000):
        super().__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # 直接传递参数初始化attention层
        self.attention = CausalSelfAttentionARMA(
            n_embd=d_model,
            n_head=n_heads,
            block_size=max_seq_len,
            bias=True,
            dropout=attention_dropout,
            ma_dropout=attention_dropout
        )
        
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        
        # 重塑输入张量
        x = queries.view(B, L, H * E)
        
        # 使用CausalSelfAttentionARMA
        output = self.attention(x)
        
        # 重塑输出以匹配期望的格式
        output = output.view(B, L, H, E)
        
        if self.output_attention:
            # 如果需要注意力权重，获取它们
            _, weights_ma, _, _, _ = self.attention.calculate_arma_weights(
                x, 
                channel_average=False,
                normalize=True,
                output_sequence=True
            )
            return output.contiguous(), weights_ma
        else:
            return output.contiguous(), None