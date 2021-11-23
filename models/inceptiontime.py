import torch
import torch.nn as nn



class BaseBlock(nn.Module):
    def __init__(self, in_planes):
        super(BaseBlock, self).__init__()

        self.bottleneck = nn.Conv1d(in_planes, 32, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=39, stride=1, padding=19, bias=False)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=19, stride=1, padding=9, bias=False)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4, bias=False)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.conv1 = nn.Conv1d(in_planes, 32, kernel_size=1, stride=1, bias=False)

        self.bn = nn.BatchNorm1d(32 * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.bottleneck(x)
        output4 = self.conv4(output)
        output3 = self.conv3(output)
        output2 = self.conv2(output)

        output1 = self.maxpool(x)
        output1 = self.conv1(output1)

        x_out = self.relu(self.bn(torch.cat((output1, output2, output3, output4), dim=1)))
        return x_out


class InceptionTime(nn.Module):
    def __init__(self, in_channel=12, num_classes=10):
        super(InceptionTime, self).__init__()

        self.BaseBlock1 = BaseBlock(in_channel)
        self.BaseBlock2 = BaseBlock(128)
        self.BaseBlock3 = BaseBlock(128)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channel, 128, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)

        self.BaseBlock4 = BaseBlock(128)
        self.BaseBlock5 = BaseBlock(128)
        self.BaseBlock6 = BaseBlock(128)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)

        self.Avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        shortcut1 = self.bn1(self.conv1(x))

        output1 = self.BaseBlock1(x)
        output1 = self.BaseBlock2(output1)
        output1 = self.BaseBlock3(output1)
        output1 = self.relu(output1 + shortcut1)

        shortcut2 = self.bn2(self.conv2(output1))

        output2 = self.BaseBlock4(output1)
        output2 = self.BaseBlock5(output2)
        output2 = self.BaseBlock6(output2)
        output2 = self.relu(output2 + shortcut2)

        output = self.Avgpool(output2 + shortcut2)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def inceptiontime(**kwargs):
    return InceptionTime(in_channel=12, **kwargs)


