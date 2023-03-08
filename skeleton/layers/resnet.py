import torch as t
import torch.nn as nn

from skeleton.layers.residual_block import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, triples):
        super(ResNet, self).__init__()
        modules = []
        modules.append(self.starting_block(128))
        for _, triple in enumerate(triples):
            num_residuals, out_channels = triple[0], triple[1]
            block = self.block(num_residuals,out_channels)
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

    def block(self, num_residuals, out_channels):
        blk = []
        for _ in range(num_residuals):
            blk.append(ResidualBlock(out_channels, use_1x1conv=True))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.net(x)
        return x
