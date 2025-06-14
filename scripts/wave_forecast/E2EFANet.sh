export CUDA_VISIBLE_DEVICES=0

model_name=E2EFANet
seq_len=48
pred_len=48
model_id=waveheight_experiment

# 1. 训练
log_file="./logs/WaveForecasting/${model_name}_${model_id}_e${e_layers}_d${d_layers}.log"
python -u run.py \
  --task_name forecasting \
  --is_training 1 \
  --root_path ./dataset/wave/ \
  --data_path H=0.18_T=2_new.csv \
  --model_id $model_id \
  --model $model_name \
  --data WAVE \
  --features M \
  --target Elevation_10 \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --des 'Exp' \
  --itr 1  2>&1 | tee -a $log_file

# 2. 泛化测试
log_file="./logs/WaveForecasting/FH/${model_name}_${model_id}_e${e_layers}_d${d_layers}.log"
for data in 'H=0.09_T=2_new.csv' 'H=0.12_T=2_new.csv' 'H=0.15_T=1.5_new.csv' 'H=0.15_T=2_new.csv' 'H=0.15_T=2.5_new.csv' 'H=0.18_T=1.5_new.csv' 'H=0.21_T=2_new.csv'
do
python -u run.py \
  --task_name forecasting \
  --is_training 0 \
  --root_path ./dataset/wave/ \
  --data_path $data \
  --model_id $model_id \
  --model $model_name \
  --data WAVE \
  --features M \
  --target Elevation_10 \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --des 'Exp' \
  --itr 1 \
  --fanhua \
  --batch_size 1 2>&1 | tee -a $log_file
done