import torch as t
import torch.nn as nn

from skeleton.layers.resnext_block import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, triples):
        super(ResNet, self).__init__()
        modules = []
        modules.append(self.starting_block(128))
        for _, triple in enumerate(triples):
            in_channels, num_residuals, out_channels = triple[0], triple[1], triple[2]
            block = self.block(in_channels, num_residuals,out_channels)
            modules.append(block)

        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool1d(3),
            nn.Flatten(),
            nn.LazyLinear(128)
        ))
        
        self.net = nn.Sequential(*modules)


    def starting_block(self, input_channels):
        return nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

    def block(self, in_channels,num_residuals, out_channels):
        blk = []
        for i in range(num_residuals):
            if i == 0:
                blk.append(ResidualBlock(in_channels, out_channels, 16 , 1, use_1x1conv=True))
            else:
                blk.append(ResidualBlock(in_channels * 2, out_channels, 16, 1))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.net(x)
        return x
