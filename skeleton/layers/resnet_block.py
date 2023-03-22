import torch as t
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size = 3, padding = 1),
                        nn.BatchNorm1d(out_channels))
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, X):
        Y = self.conv1(X)
        Y = self.conv2(Y)
        residual = self.conv3(X)
        Y += residual
        Y = F.relu(Y)
        return Y
