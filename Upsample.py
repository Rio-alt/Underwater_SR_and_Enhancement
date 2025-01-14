import torch
import torch.nn as nn
import torch.nn.functional as F

from SeparableConv2d import SeparableConv2d

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        self.gn = nn.GroupNorm(8, in_channels)
        self.relu = nn.ReLU()
        self.conv = SeparableConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.relu(self.gn(x))
        upsampled = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(upsampled)