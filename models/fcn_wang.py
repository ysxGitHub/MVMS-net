import torch.nn as nn


def conv1d(in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    lst = []
    lst.append(nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                         dilation=dilation, bias=False))
    lst.append(nn.BatchNorm1d(out_planes))
    lst.append(nn.ReLU(True))
    return nn.Sequential(*lst)


class FCN(nn.Sequential):
    def __init__(self, input_channels=12, num_classes=5):
        super(FCN, self).__init__()
        self.conv1 = conv1d(input_channels, 128, kernel_size=8, stride=1)
        self.conv2 = conv1d(128, 256, kernel_size=5, stride=1)
        self.conv3 = conv1d(256, 128, kernel_size=3, stride=1)
        self.Avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.Avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def fcn_wang(**kwargs):
    return FCN(input_channels=12, **kwargs)

