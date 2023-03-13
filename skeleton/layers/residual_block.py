import torch as t
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, bot_mul, use_1x1conv=False, strides=1, kernel_size=3, padding=1):
        super().__init__()
        bot_channels = int(round(out_channels * bot_mul))

        self.conv1 = nn.Conv1d(in_channels, bot_channels, kernel_size=kernel_size, padding=2,
                               stride=1)
        self.conv2 = nn.Conv1d(out_channels, bot_channels, kernel_size=kernel_size, stride=strides, padding=padding, groups=bot_channels//groups)
        self.conv3 = nn.LazyConv1d(out_channels, kernel_size=kernel_size, stride=1)

        if use_1x1conv:
            self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=strides)
            self.bn4 = nn.BatchNorm1d(out_channels)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))

        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        Y += X
        return F.relu(Y)