import torch as t
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self,out_channels, use_1x1conv=False, strides=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.LazyConv1d(out_channels, kernel_size=kernel_size, padding=padding,
                               stride=strides)
        self.conv2 = nn.LazyConv1d(out_channels, kernel_size=kernel_size, padding=padding)
        if use_1x1conv:
            self.conv3 = nn.LazyConv1d(out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)