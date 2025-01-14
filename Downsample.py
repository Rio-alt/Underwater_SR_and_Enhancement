import torch
import torch.nn as nn
import torch.nn.functional as F

from SeparableConv2d import SeparableConv2d

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = SeparableConv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)