
import torch
import torch.nn as nn
import torch.nn.functional as F

class GSA(nn.Module):
    def __init__(self, in_channels):
        super(GSA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gn = nn.GroupNorm(8, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.gn(y)
        y = self.sigmoid(self.conv2(y))
        return x * y

