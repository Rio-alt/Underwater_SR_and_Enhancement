import torch
import torch.nn as nn
import torch.nn.functional as F

from PFDM import PFDM 
from CGM import CGM
from HUP0 import HUP0
from HUP1 import HUP1

class PFIN(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(PFIN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        self.cgm_module = CGM(base_channels)
        self.pfdm_module = PFDM(base_channels)


    def forward(self, x, upscale_factor):
        input = self.conv1(x)
        cgmOut = self.cgm_module(input)
        pfdmOut = self.pfdm_module(input, cgmOut)

        hup0 = HUP0(in_channels=64, out_channels=3, upscale_factor=2*upscale_factor)
        hup1 = HUP1(in_channels=3, out_channels=3, upscale_factor=upscale_factor)
        S = hup0(pfdmOut) + hup1(x)

        E = self.conv2(pfdmOut)


        return [S , E]