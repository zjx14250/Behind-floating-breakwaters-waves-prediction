## Introduction

This research aims to develop a novel deep learning framework called the Exogenous-to-Endogenous Frequency-Aware Network (E2E-FANet), with the direct objective of achieving high-precision prediction of free surface elevation behind floating breakwaters. 

The core innovations of this model lie in:
1) Designing an exogenous-to-endogenous cross-attention mechanism that explicitly models the driving influence of structural motion on water surface variations by using breakwater motion as the "keys" and "values" in the attention mechanism, while treating free surface elevation as the "query," thereby overcoming the causal relationship modeling deficiencies of traditional models; 
2) Introducing a dual-basis frequency mapping module that utilizes orthogonal sine and cosine basis functions to enhance the model's capability in capturing frequency and periodicity characteristics within free surface elevation sequences.

## E2E-FANet
![Framework of the proposed E2E-FANet](figure/image.png)

## Install requirements
```bash
pip install -r requirements.txt
```

## Run
```bash
bash ./scripts/wave_forecast/E2EFANet.sh
```

## Result
![图片描述](figure/result1.png)
![图片描述](figure/result2.png)
![图片描述](figure/result3.png)
