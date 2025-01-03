import torch
import torch.nn as nn
import torch.nn.functional as F

from Upsample import Upsample
from Downsample import Downsample

class CGM(nn.Module):
    def __init__(self, in_channels):
        super(CGM, self).__init__()
        self.outArray = [None] * 4

        self.down1 = Downsample(in_channels)
        self.down2 = Downsample(in_channels)
        self.down3 = Downsample(in_channels)
        self.down4 = Downsample(in_channels)

        self.up3 = Upsample(in_channels)
        self.up2 = Upsample(in_channels)
        self.up1 = Upsample(in_channels)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        self.outArray[0] = self.conv2(self.relu(self.conv1(x)))

        x = self.up3(self.outArray[0])

        self.outArray[1] = self.conv4(self.relu(self.conv3(x)))

        x = self.up2(self.outArray[1])

        self.outArray[2] = self.conv6(self.relu(self.conv5(x)))

        x = self.up1(self.outArray[2])

        self.outArray[3] = self.conv8(self.relu(self.conv7(x)))

        return self.outArray