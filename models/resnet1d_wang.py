import torch.nn as nn

def conv(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=[3, 3], downsample=None):
        super().__init__()
        if (isinstance(kernel_size, int)): kernel_size = [kernel_size, kernel_size // 2 + 1]

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Sequential):
    def __init__(self, block, layers, kernel_size=5, num_classes=2, input_channels=12):
        super(ResNet1d, self).__init__()
        self.inplanes = 64
        self.conv = nn.Conv1d(input_channels, 64, kernel_size=8, stride=2, padding=7 // 2, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block1 = self._make_layer(block, 64, layers[0], kernel_size=kernel_size)
        self.block2 = self._make_layer(block, 128, layers[1], stride=2, kernel_size=kernel_size)
        self.block3 = self._make_layer(block, 128, layers[1], stride=2, kernel_size=kernel_size)

        self.Avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, kernel_size, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.relu(self.bn(self.conv(x)))
        output = self.maxpool(output)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.Avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


def resnet1d_wang(**kwargs):
    return ResNet1d(BasicBlock1d, [1, 1, 1], **kwargs)

