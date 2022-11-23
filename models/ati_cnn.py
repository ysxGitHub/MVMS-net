import torch
import torch.nn as nn
import torch.nn.functional as F

class ATICNN(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(ATICNN, self).__init__()
        self.L = 32
        self.D = 128
        self.K = num_classes

        self.features = features 
        self.lstm = nn.LSTM(num_layers=2, input_size=512, hidden_size=32, dropout=0.2) 

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), # 32 * 32
            nn.Tanh(),
            nn.Linear(self.D, self.K)  # 32 * 9
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x) # N*3*224*224
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x = self.lstm(x)
        x = x[0].transpose(0, 1) # [batch, num_directions * num_layers, hidden_size]
        # x = x.transpose(1, 2)
        A = self.attention(x)  # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.bmm(x.transpose(1, 2), A)  # KxL
        # print(x.shape, A.shape)
        M = torch.sum(M, dim=1)
        return M

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) 
                nn.init.constant_(m.bias, 0) 



def make_features(input_channels: int, cfg: list):
    layers = []
    in_channels = input_channels
    for v in cfg: 
        if v == "M":  
            layers += [nn.MaxPool1d(kernel_size=1, stride=2)]
        else:  
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(True)]
            in_channels = v  
    return nn.Sequential(*layers)


def ATI_CNN(input_channels=12, num_classes=9, **kwargs):
    config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = ATICNN(make_features(input_channels, config), num_classes = num_classes, **kwargs)
    return model
  
