# -*- coding: utf-8 -*-
"""
@time: 2021/4/15 15:40

@ author:
"""
import torch, time, os
import models, utils
from torch import optim
from dataset import load_datasets
from config import config

from sklearn.metrics import roc_auc_score

import numpy as np
import random
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    print('seed: ', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model_large, model_small, optimizer, criterion, train_dataloader):
    model_large.train(), model_small.train()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []
    for inputs, target in train_dataloader:

        inputs = inputs + torch.randn_like(inputs) * 0.1

        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        with torch.no_grad():
            target1 = model_large(inputs)

        output = model_small(inputs)

        loss = criterion(output, target, target1)

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1

        output = torch.sigmoid(output)
        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())
    auc = roc_auc_score(targets, outputs)
    TPR = utils.compute_TPR(targets, outputs)
    print('train_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f' % (loss_meter / it_count, auc, TPR))
    return loss_meter / it_count, auc, TPR


def test_epoch(model_large, model_small, criterion, val_dataloader):
    model_large.eval(), model_small.eval()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []
    with torch.no_grad():
        for inputs, target in val_dataloader:

            inputs = inputs + torch.randn_like(inputs) * 0.1

            inputs = inputs.to(device)
            target = target.to(device)

            target1 = model_large(inputs)

            output = model_small(inputs)

            loss = criterion(output, target, target1)

            loss_meter += loss.item()
            it_count += 1

            output = torch.sigmoid(output)
            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

        auc = roc_auc_score(targets, outputs)
        TPR = utils.compute_TPR(targets, outputs)

    print('test_loss: %.4f,   macro_auc: %.4f,   TPR: %.4f' % (loss_meter / it_count, auc, TPR))
    return loss_meter / it_count, auc, TPR


def train(config=config):
    # seed
    setup_seed(config.seed)
    print('torch.cuda.is_available:', torch.cuda.is_available())

    # datasets
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
        datafolder=config.datafolder,
        experiment=config.experiment,
    )

    # mode
    print('model_name:{}, num_classes={}'.format(config.model_name, num_classes))
    model_large = getattr(models, config.model_name)(num_classes=num_classes)
    model_small = getattr(models, config.model_name2)(num_classes=num_classes)

    model_large = model_large.to(device)
    model_small = model_small.to(device)

    # optimizer and loss
    optimizer = optim.Adam(model_small.parameters(), lr=config.lr)
    criterion = utils.KdLoss(config.alpha, config.temperature)

    if config.checkpoints is not None:
        checkpoints = torch.load(os.path.join('checkpoints', config.checkpoints))
        model_dict = model_large.state_dict()
        state_dict = {k: v for k, v in checkpoints['model_state_dict'].items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model_large.load_state_dict(model_dict)
        print('best_acc: ',checkpoints['best_acc'])

    # =========>train<=========
    for epoch in range(1, config.max_epoch + 1):
        print('#epoch: {}  batch_size: {}  Current Learning Rate: {}'.format(epoch, config.batch_size,
                                                                             config.lr))

        since = time.time()
        train_loss, train_auc, train_TPR = train_epoch(model_large, model_small, optimizer, criterion, train_dataloader)

        val_loss, val_auc, val_TPR = test_epoch(model_large, model_small, criterion, val_dataloader)

        test_loss, test_auc, test_TPR = test_epoch(model_large, model_small, criterion, test_dataloader)


        result_list = [
            [epoch, train_loss, train_auc, train_TPR,
             val_loss, val_auc, val_TPR,
             test_loss, test_auc, test_TPR]]
        if epoch == 1:
            columns = ['epoch', 'train_loss', 'train_auc', 'train_TPR',
                       'val_loss', 'val_auc', 'val_TPR',
                       'test_loss', 'test_auc', 'test_TPR']
        else:
            columns = ['', '', '', '', '', '', '', '', '', '']
        dt = pd.DataFrame(result_list, columns=columns)
        dt.to_csv(config.model_name + config.experiment + 'result.csv', mode='a')
        print('time:%s\n' % (utils.print_time_cost(since)))


if __name__ == '__main__':
    train(config)

