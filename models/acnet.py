import torch
import torch.nn as nn
import torch.nn.functional as F


class Dccacblock(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Dccacblock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.3)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.3)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2-1)
        self.relu3 = nn.LeakyReLU(0.3)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        return self.dp(self.relu3(self.conv3(self.relu2(self.conv2(self.relu1(self.conv1(x)))))))


class Dccacb(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Dccacb, self).__init__()
        self.block1 = Dccacblock(12, 24)
        self.block2 = Dccacblock(12, 24)
        self.block3 = Dccacblock(12, 24)
        self.block4 = Dccacblock(12, 24)
        self.block5 = Dccacblock(12, 48)
        self.rnn = nn.GRU(12, 12, bidirectional=True)
        self.relu = nn.LeakyReLU(0.3)
        self.dp = nn.Dropout(0.2)

        self.attention_layer = nn.Sequential(
            nn.Linear(12, 12),
            nn.ReLU(inplace=True)
        )
        self.bn = nn.BatchNorm1d(12)
        self.relu1 = nn.LeakyReLU(0.3)

        self.fc = nn.Linear(12, num_classes)

    def forward(self, x):
        x = self.block5(self.block4(self.block3(self.block2(self.block1(x)))))
        x = x.transpose(0, 2).transpose(1, 2)

        x, _ = self.rnn(x)
        x = self.dp(self.relu(x))

        x = x.transpose(0, 1)
        x = self.dp(self.relu1(self.bn(self.attention_net_with_w(x))))

        x = self.fc(x)

        return x

    def attention_net_with_w(self, lstm_out):
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        atten_w = self.attention_layer(h)
        m = nn.Tanh()(h)
        atten_context = torch.bmm(m, atten_w.transpose(1, 2))

        softmax_w = F.softmax(atten_context, dim=-1)

        context = torch.bmm(h.transpose(1, 2), softmax_w)
        result = torch.sum(context, dim=-1)
        result = nn.Dropout(0.)(result)
        return result



def dccacb(**kwargs):
    return Dccacb(input_channels=12, **kwargs)


# if __name__ == '__main__':
#     model = Dccacb(10, 12)
#     x = torch.randn(64, 12, 1000)

