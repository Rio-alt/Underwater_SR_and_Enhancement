import torch
import torch.nn as nn
import torch.nn.functional as F

from Upsample import Upsample
from Downsample import Downsample
from GSA import GSA
from FIM import FIM
from MRes import MRes


class PFDM(nn.Module):
    def __init__(self, in_channels):
        super(PFDM, self).__init__()

        # Initial layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Downsampling layers
        self.down1 = Downsample(in_channels)
        self.down2 = Downsample(in_channels)
        self.down3 = Downsample(in_channels)

        # Upsampling layers
        self.up3 = Upsample(in_channels)
        self.up2 = Upsample(in_channels)
        self.up1 = Upsample(in_channels)

        # MRes layers
        self.mres1 = MRes(in_channels)
        self.mres2 = MRes(in_channels)
        self.mres3 = MRes(in_channels)
        self.mres4 = MRes(in_channels)
        self.mres5 = MRes(in_channels)
        self.mres6 = MRes(in_channels)
        self.mres7 = MRes(in_channels)
        self.mres8 = MRes(in_channels)

        # FIM layers
        self.fim1 = FIM(in_channels)
        self.fim2 = FIM(in_channels)
        self.fim3 = FIM(in_channels)
        self.fim4 = FIM(in_channels)

        # GSA layers
        self.gsa1 = GSA(in_channels)
        self.gsa2 = GSA(in_channels)
        self.gsa3 = GSA(in_channels)
        self.gsa4 = GSA(in_channels)
        self.gsa5 = GSA(in_channels)
        self.gsa6 = GSA(in_channels)
        self.gsa7 = GSA(in_channels)

        

    def forward(self, x, cgmOut):
        b1 = self.maxpool(x)
        b2 = self.avgpool(x)
        x = b1 + b2
        x = self.relu(self.conv(x))

        # Downsampling
        f1 = x
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)

        f11 = f1 + self.up1(f2)
        f21 = f2 + self.up2(f3)
        f31 = f3 + self.up3(f4)



        f4_mres = self.mres1(f4)
        f4_fim = self.fim1(f4_mres, cgmOut[0]) 



        f3_gsa = self.gsa1(f31, self.up3(f4_fim))
        f3_mres = self.mres2(f3_gsa)
        f3_fim = self.fim2(f3_mres, cgmOut[1])



        f2_gsa1 = self.gsa2(f21, self.up2(f3_gsa))
        f2_mres1 = self.mres3(f2_gsa1)

        f2_gsa2 = self.gsa3(f2_mres1, self.up2(f3_fim))
        f2_mres2 = self.mres4(f2_gsa2)

        f2_fim = self.fim3(f2_mres2, cgmOut[2])



        f1_gsa1 = self.gsa4(f11, self.up1(f2_gsa1))
        f1_mres1 = self.mres5(f1_gsa1)

        f1_gsa2 = self.gsa5(f1_mres1, self.up1(f2_mres1))
        f1_mres2 = self.mres6(f1_gsa2)

        f1_gsa3 = self.gsa6(f1_mres2, self.up1(f2_gsa2))
        f1_mres3 = self.mres7(f1_gsa3)

        f1_gsa4 = self.gsa7(f1_mres3, self.up1(f2_fim))
        f1_mres4 = self.mres8(f1_gsa4)

        f1_fim = self.fim4(f1_mres4, cgmOut[3])


        output = f1_fim
       
        return output
