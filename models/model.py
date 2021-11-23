# -*- coding: utf-8 -*-
'''
@time: 2021/4/17 20:14

@ author: ysx
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import CoordAtt


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class Res2Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=5, stride=1, downsample=None, groups=1, base_width=26,
                 dilation=1, scale=4, first_block=True, norm_layer=nn.BatchNorm1d,
                 atten=True):

        super(Res2Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        width = int(planes * (base_width / 64.)) * groups
        # print(width)

        self.atten = atten

        self.conv1 = conv1x1(inplanes, width * scale)
        self.bn1 = norm_layer(width * scale)

        # If scale == 1, single conv else identity & (scale - 1) convs
        nb_branches = max(scale, 2) - 1
        if first_block:
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride,
                                              padding=kernel_size // 2, groups=1, bias=False, dilation=1)
                                    for _ in range(nb_branches)])
        self.bns = nn.ModuleList([norm_layer(width) for _ in range(nb_branches)])
        self.first_block = first_block
        self.scale = scale

        self.conv3 = conv1x1(width * scale, planes * self.expansion)

        self.relu = Mish()
        self.bn3 = norm_layer(planes * self.expansion)  # bn reverse

        # self.dropout = nn.Dropout(.1)

        if self.atten is True:
            # self.attention = SELayer(planes * self.expansion)
            # self.attention = CBAM(planes * self.expansion)
            self.attention = CoordAtt(planes * self.expansion, planes * self.expansion)
        else:
            self.attention = None

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):

        out = self.conv1(x)

        out = self.relu(out)
        out = self.bn1(out)  # bn reverse
        # Chunk the feature map
        xs = torch.chunk(out, self.scale, dim=1)
        # Initialize output as empty tensor for proper concatenation
        y = 0
        for idx, conv in enumerate(self.convs):
            # Add previous y-value
            if self.first_block:
                y = xs[idx]
            else:
                y += xs[idx]
            y = conv(y)
            y = self.relu(self.bns[idx](y))
            # Concatenate with previously computed values
            out = torch.cat((out, y), 1) if idx > 0 else y
        # Use last chunk as x1
        if self.scale > 1:
            if self.first_block:
                out = torch.cat((out, self.pool(xs[len(self.convs)])), 1)
            else:
                out = torch.cat((out, xs[len(self.convs)]), 1)

        # out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.atten:
            out = self.attention(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class MyNet(nn.Module):

    def __init__(self, num_classes=5, input_channels=12, single_view=False):
        super(MyNet, self).__init__()

        self.single_view = single_view

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=25, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = Mish()

        self.layer1 = Res2Block(inplanes=64, planes=128, kernel_size=15, stride=2, atten=True)

        self.layer2 = Res2Block(inplanes=128, planes=128, kernel_size=15, stride=2, atten=True)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if not self.single_view:
            self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.layer1(output)

        output = self.layer2(output)

        output = self.avgpool(output)

        output = output.view(output.size(0), -1)

        if not self.single_view:
            output = self.fc(output)

        return output


class AdaptiveWeight(nn.Module):
    def __init__(self, plances=32):
        super(AdaptiveWeight, self).__init__()

        self.fc = nn.Linear(plances, 1)
        # self.bn = nn.BatchNorm1d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        # out = self.bn(out)
        out = self.sig(out)

        return out


class MyNet6View(nn.Module):

    def __init__(self, num_classes=5):
        super(MyNet6View, self).__init__()

        self.MyNet1 = MyNet(input_channels=1, single_view=True)
        self.MyNet2 = MyNet(input_channels=2, single_view=True)
        self.MyNet3 = MyNet(input_channels=2, single_view=True)
        self.MyNet4 = MyNet(input_channels=2, single_view=True)
        self.MyNet5 = MyNet(input_channels=2, single_view=True)
        self.MyNet6 = MyNet(input_channels=3, single_view=True)

        self.fuse_weight_1 = AdaptiveWeight(128)
        self.fuse_weight_2 = AdaptiveWeight(128)
        self.fuse_weight_3 = AdaptiveWeight(128)
        self.fuse_weight_4 = AdaptiveWeight(128)
        self.fuse_weight_5 = AdaptiveWeight(128)
        self.fuse_weight_6 = AdaptiveWeight(128)

        self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        outputs_view = [self.MyNet1(x[:, 3, :].unsqueeze(1)),
                        self.MyNet2(torch.cat((x[:, 0, :].unsqueeze(1), x[:, 4, :].unsqueeze(1)), dim=1)),
                        self.MyNet3(x[:, 6:8, :]),
                        self.MyNet4(x[:, 8:10, :]),
                        self.MyNet5(x[:, 10:12, :]),
                        self.MyNet6(torch.cat((x[:, 1:3, :], x[:, 5, :].unsqueeze(1)), dim=1))]

        fuse_weight_1 = self.fuse_weight_1(outputs_view[0])
        fuse_weight_2 = self.fuse_weight_2(outputs_view[1])
        fuse_weight_3 = self.fuse_weight_3(outputs_view[2])
        fuse_weight_4 = self.fuse_weight_4(outputs_view[3])
        fuse_weight_5 = self.fuse_weight_5(outputs_view[4])
        fuse_weight_6 = self.fuse_weight_6(outputs_view[5])

        output = fuse_weight_1 * outputs_view[0] + fuse_weight_2 * outputs_view[1] + fuse_weight_3 * \
                 outputs_view[2] + fuse_weight_4 * outputs_view[3] + fuse_weight_5 * outputs_view[
                     4] + fuse_weight_6 * outputs_view[5]

        x_out = self.fc(output)

        return x_out

