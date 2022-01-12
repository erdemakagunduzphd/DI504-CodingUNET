from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
# This is the "dropout modified" UNET code with; 
# for Conv2d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
# for BatchNorm2d, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
# for MaxPool2d, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
# for ConvTranspose2d, kernel_size=(2, 2), stride=(2, 2)
# Dropout(p=0.5) layers are added after MaxPool2d layers for each down path
# Dropout(p=0.5) layers are added after ConvTranspose2d layers and after concatenate for each up path
# 4 down paths
# 4 up paths
# sigmoid as the final layer


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        fsize= 3;
        psize = int((fsize-1)/2)
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, fsize,padding=(psize, psize)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, fsize,padding=(psize, psize)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2), nn.Dropout(0.5),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.bilinear = bilinear

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        
        self.drop = nn.Dropout(0.5)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        x = torch.cat([x2, x1], dim=1)
        x = self.drop(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class sigmoidOut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(sigmoidOut, self).__init__()
        self.outsig= nn.Sigmoid()

    def forward(self, x):
        x = self.outsig(x)
        return x


class UNet(nn.Module):
    def __init__(self, classes):
        super(UNet, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, classes)
        self.outsig = sigmoidOut(1,1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.outsig(x)
        #return F.sigmoid(x)
        return x

##############################################################################