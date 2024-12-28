import torch
import torch.nn as nn

class HUP1(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(HUP1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=5, padding=2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
