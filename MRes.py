import torch
import torch.nn as nn
import torch.nn.functional as F


class MRes(nn.Module):
    def __init__(self, in_channels):

        super(MRes, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.convd3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.convd5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)

    def forward(self, x):
        residual = x
        print("Shape of x:", x.shape)
        x = self.relu(self.conv1(x))
        print("Shape of x(relu):", x.shape)
        b1 = self.convd3(x)
        print("Shape of b1:", b1.shape)
        b2 = self.convd5(x)
        print("Shape of b2:", b2.shape)
        x = b1 + b2
        print("Shape of b1+b2:", x.shape)
        x = self.conv2(x)
        print("Shape of conv(b1+b2):", x.shape)
        return x + residual
