import torch as t
import torch.nn as nn

from skeleton.layers.residual_block import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, triples):
        super(ResNet, self).__init__()
        modules = []
        modules.append(self.starting_block(128))
        for i, triple in enumerate(triples):
            num_residuals, out_channels = triple[0], triple[1]
            block = self.block(num_residuals, out_channels, first_block=(i==0))
            modules.append(block)
            modules.append(nn.Sequential(nn.ReLU(), nn.AdaptiveAvgPool1d(3)))

        modules.append(nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(128)
        ))
        
        self.net = nn.Sequential(*modules)


    def starting_block(self, input_channels):
        return nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

    def block(self, num_residuals, out_channels, first_block = False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResidualBlock(out_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(ResidualBlock(out_channels))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.net(x)
        return x
