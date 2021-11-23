# Integrating Knowledge Distillation With Multi-View Neural Network for Multi-Label ECG Classification

This is the code for the paper "Integrating Knowledge Distillation With Multi-View Neural Network for Multi-Label ECG Classification"

# Dependency

- python>=3.7
- pytorch>=1.7.0
- torchvision>=0.8.1
- numpy>=1.19.5
- tqdm>=4.62.0
- scipy>=1.5.4
- wfdb>=3.2.0
- scikit-learn>=0.24.2

# Usage

## Configuration

There is a configuration file "config.py", where one can edit both the training and test options.

## Stage 1: Training 

After setting the configuration, to start training, simply run

> python main_train.py

Since MiniRocket's training strategy is slightly different from the others, to start training in MiniRocket, run

> python minirocket_train.py

## Stage 2: Knowledge Distillation

# Dataset

PTB-XL dataset can be downloaded from [PTB-XL, a large publicly available electrocardiography dataset v1.0.1 (physionet.org)](https://www.physionet.org/content/ptb-xl/1.0.1/).

CPSC2018 dataset can be downloaded from [The China Physiological Signal Challenge 2018 (icbeb.org)](http://2018.icbeb.org/Challenge.html)

HFHC dataset can be downloaded from https://tianchi.aliyun.com/competition/entrance/231754/information

# Citation

If you find this idea useful in your research, please consider citing:

```
@article{
  title={Integrating Knowledge Distillation With Multi-View Neural Network for Multi-Label ECG Classification},
  author={Shunxiang Yang, Cheng Lian, Zhigang Zeng, Bingrong Xu, Yixin Su},
  journal={},
  year={2021}
}
```
