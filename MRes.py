import torch
import torch.nn as nn
import torch.nn.functional as F

from SeparableConv2d import SeparableConv2d

class MRes(nn.Module):
    def __init__(self, in_channels):

        super(MRes, self).__init__()
        self.conv1 = SeparableConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = SeparableConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.convd3 = SeparableConv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.convd5 = SeparableConv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        b1 = self.convd3(x)
        b2 = self.convd5(x)
        x = b1 + b2
        x = self.conv2(x)
        return x + residual
