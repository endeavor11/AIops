# Time-series-Anomaly-Detection-transformer

Transformer based Time-series Anomaly detector model implemented in Pytorch.

This is an implementation of transformer based time-series anomaly detector, which consists of two-stage strategy of time-series prediction and anomaly score calculation.

## Requirements

* Ubuntu 16.04+ 
* Python 3.5+
* Pytorch 0.4.0+
* Numpy
* Matplotlib
* Scikit-learn
* argparse
* visdom
* pickle

## Dataset

 [OmniAnomaly/ServerMachineDataset at master · hdjkfhkj/OmniAnomaly (github.com)](https://github.com/hdjkfhkj/OmniAnomaly/tree/master/ServerMachineDataset) 

We build on this public dataset with anomaly enhancements

## Transformer-based Multi-Step Prediction Model

__0. Architecture__

![model_architecture](E:\_文档\_研究生_实验文档\gitee_File\第一篇论文-纵向联邦学习\纵向联邦学习\上传到github的代码\fig\model_architecture.png)



## Example of usage



__0. File structure__

* **AlignmentTimeSeries**: This fold includes the process of our exploration of time series alignment methods and the experimental results of time series alignment.

* **Anomaly_detection**:  This folder contains the anomaly detection models for the general structure, including models based on LSTM and models based on Transformer.

* **Anomaly_detection_VFL**: This folder contains the anomaly detection models for vertical federated learning, both LSTM based models and Transformer based models.

* **DistriTailNoise2Th**: This folder contains the implementation of the anomaly threshold detection algorithm and related experimental explorations.

  

__1. Time-series prediction:__
Train and save prediction model on a time-series trainset.

```python
CUDA_VISIBLE_DEVICES=1  python 1_train_predictor_FL.py --data SMD_1_3_10dim_E   --filename  machine-1-3_10dim.pkl   --epochs  1100 --emsize 32   --lr 0.002 --bptt 200 --prediction_window_size 50   --log_interval 5 --batch_size 128 --weight_decay 0.0 --model transformer --index 1st
```


__2. Anomaly detection:__
Fit multivariate gaussian distribution and
calculate anomaly scores on a time-series testset

```
CUDA_VISIBLE_DEVICES=1  python 2_2_anomaly_detection_FL.py   --data SMD_1_3_10dim_E   --filename machine-1-3_10dim.pkl   --prediction_window_size 50 --index 1st
```


## Result

![result](E:\_文档\_研究生_实验文档\gitee_File\第一篇论文-纵向联邦学习\纵向联邦学习\上传到github的代码\fig\result.png)



## Contact

If you have any questions, please open an issue.

