# A Multi-View Multi-Scale Neural Network for Multi-Label ECG Classification

This is the code for the paper "A Multi-View Multi-Scale Neural Network for Multi-Label ECG Classification"

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

The multi-view network trained in the first stage is used to train the single-view network, run

> python main_distillation.py

# Dataset

PTB-XL dataset can be downloaded from [PTB-XL, a large publicly available electrocardiography dataset v1.0.1 (physionet.org)](https://www.physionet.org/content/ptb-xl/1.0.1/).

CPSC2018 dataset can be downloaded from [The China Physiological Signal Challenge 2018 (icbeb.org)](http://2018.icbeb.org/Challenge.html)

HFHC dataset can be downloaded from https://tianchi.aliyun.com/competition/entrance/231754/information

# Citation

If you find this idea useful in your research, please consider citing:

```
@article{
  title={A Multi-View Multi-Scale Neural Network for Multi-Label ECG Classification},
}
```
