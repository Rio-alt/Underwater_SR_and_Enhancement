import torch
import torch.nn as nn
import torch.nn.functional as F

import GSA
import FIM
import MRes

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, skip_connection):
        upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(upsampled + skip_connection)

class PFDM(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(PFDM, self).__init__()
        # Downsampling layers
        self.down1 = DownsampleBlock(in_channels, base_channels)
        self.down2 = DownsampleBlock(base_channels, base_channels * 2)
        self.down3 = DownsampleBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownsampleBlock(base_channels * 4, base_channels * 8)

        # MRes and FIM
        self.mres4 = MRes(base_channels * 8)
        self.fim4 = FIM(base_channels * 8)

        # Upsampling layers
        self.up3 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.up1 = UpsampleBlock(base_channels * 2, base_channels)

        # GSA
        self.gsa3 = GSA(base_channels * 4)
        self.gsa2 = GSA(base_channels * 2)
        self.gsa1 = GSA(base_channels)

        # Final convolution
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Downsampling
        f1 = self.down1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f4 = self.down4(f3)

        # Lowest frequency processing
        f4_processed = self.fim4(self.mres4(f4))

        # Upsampling and fusion
        f3 = self.gsa3(f3 + self.up3(f4_processed, f3))
        f2 = self.gsa2(f2 + self.up2(f3, f2))
        f1 = self.gsa1(f1 + self.up1(f2, f1))

        # Final output
        output = self.final_conv(f1)
        return output


# Example usage
input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 resolution
model = PFDM(in_channels=3)
output = model(input_tensor)
print(output.shape)  # Should be [1, 3, 256, 256]
