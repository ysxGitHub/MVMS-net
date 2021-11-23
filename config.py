# -*- coding: utf-8 -*-
'''
@time: 2021/4/16 18:45

@ author:
'''

class Config:

    seed = 10

    # path
    datafolder = '../data/ptbxl/'
    # datafolder = '../data/CPSC/'
    # datafolder = '../data/hf/'

    #
    '''
    experiment = exp0, exp1, exp1.1, exp1.1.1, exp2, exp3
    '''
    experiment = 'exp0'

    # for train
    '''
    MyNet6View, resnet1d_wang, xresnet1d101, inceptiontime, fcn_wang, lstm, lstm_bidir, vit, mobilenetv3_small
    '''
    model_name = 'MyNet6View'

    model_name2 = 'MyNet'

    batch_size = 64

    max_epoch = 100

    lr = 0.001

    device_num = 1

    # eg: MyNet6View_all_checkpoint_best_tpr.pth
    checkpoints = 'MyNet6View_exp0_checkpoint_best_auc.pth'

    # knowledge distillation param
    alpha = 0.5
    temperature = 2


config = Config()
