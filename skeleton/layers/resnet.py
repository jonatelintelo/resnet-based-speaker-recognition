import torch as t
import torch.nn as nn

from skeleton.layers.residual_block import ResidualBlock

class ResNet(nn.Module):
    def __init__(self, triples):
        super(ResNet, self).__init__()
        modules = []
        for i, triple in enumerate(triples):
            block = self.block(*triple)
            modules.append(block)
        
        self.sb = self.starting_block(40, 32)

        self.net = nn.Sequential(*modules)

    def starting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(out_channels), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

    def block(self, in_channels, num_residuals, out_channels):
        blk = []
        for i in range(num_residuals):
            if i == 0:
                blk.append(ResidualBlock(in_channels, out_channels))
            else:
                blk.append(ResidualBlock(in_channels*2, out_channels))
        return nn.Sequential(*blk)
    
    def forward(self, x):
        x = self.sb(x)
        x = self.net(x)
        return x
