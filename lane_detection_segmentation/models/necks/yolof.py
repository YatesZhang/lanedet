import torch
from torch import nn


class BaseBlock(nn.Module):
    def __init__(self, scale, size=(36,100), proj_dim=64, batch_norm=True):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=proj_dim, out_channels=proj_dim, kernel_size=(1, 1))
        self.dilated_conv = nn.Conv2d(in_channels=proj_dim, out_channels=proj_dim,
                                      kernel_size=(3, 3), dilation=scale, padding=(scale, scale))
        self.conv2 = nn.Conv2d(in_channels=proj_dim, out_channels=proj_dim, kernel_size=(1, 1))
        if batch_norm:
            self.bn = nn.BatchNorm2d(proj_dim, affine=True)
        else:
            self.bn = nn.Identity()
        self.relu = nn.ReLU(inplace=False)      # to ensure the backward
        self.h, self.w = size

    def forward(self, x):
        identity = x
        x = torch.roll(x, shifts=self.h//4, dims=2)
        x = torch.roll(x, shifts=self.w//4, dims=3)
        x = self.conv1(x)   # conv1x1
        x = self.dilated_conv(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x += identity
        return x


# [b, 128, 36, 100] as defualt input
class YOLOFplus(nn.Module):
    def __init__(self, c_in=128, fea_size=(36, 100), proj_dim=128, batch_norm=True):
        super(YOLOFplus, self).__init__()
        self.c = c_in
        self.h, self.w = fea_size
        self.proj_down = nn.Conv2d(in_channels=c_in, out_channels=proj_dim, kernel_size=(1, 1))
        self.blk1 = BaseBlock(scale=1, size=fea_size, proj_dim=proj_dim, batch_norm=batch_norm)
        self.blk2 = BaseBlock(scale=2, size=fea_size, proj_dim=proj_dim, batch_norm=batch_norm)
        self.blk3 = BaseBlock(scale=3, size=fea_size, proj_dim=proj_dim, batch_norm=batch_norm)
        self.blk4 = BaseBlock(scale=4, size=fea_size, proj_dim=proj_dim, batch_norm=batch_norm)
        self.proj_up = nn.Conv2d(in_channels=proj_dim, out_channels=c_in, kernel_size=(1, 1))

    def forward(self, x):
        x = self.proj_down(x)
        x = self.blk4(x)
        x = self.blk3(x)
        x = self.blk2(x)
        x = self.blk1(x)
        x = self.proj_up(x)
        return x


class CustomBlock(nn.Module):
    def __init__(self, custom_conv, size=(36,100), proj_dim=64, batch_norm=True):
        super(CustomBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=proj_dim, out_channels=proj_dim, kernel_size=(1, 1))
        self.custom_conv = custom_conv
        self.conv2 = nn.Conv2d(in_channels=proj_dim, out_channels=proj_dim, kernel_size=(1, 1))
        if batch_norm:
            self.bn = nn.BatchNorm2d(proj_dim, affine=True)
        else:
            self.bn = nn.Identity()
        self.relu = nn.ReLU(inplace=False)      # to ensure the backward
        self.h, self.w = size

    def forward(self, x):
        identity = x
        x = torch.roll(x, shifts=self.h//4, dims=2)
        x = torch.roll(x, shifts=self.w//4, dims=3)
        x = self.conv1(x)   # conv1x1
        x = self.custom_conv(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x += identity
        return x
        
# [b, 128, 36, 100] as defualt input
class YOLOFpro(nn.Module):
    def __init__(self, c_in=128, fea_size=(36, 100), proj_dim=64, batch_norm=True):
        super(YOLOFpro, self).__init__()
        self.c = c_in
        self.h, self.w = fea_size
        self.proj_down = nn.Conv2d(in_channels=c_in, out_channels=proj_dim, kernel_size=(1, 1))
        self.conv7x7 = nn.Conv2d(in_channels=proj_dim, out_channels=proj_dim,
         kernel_size=(7, 7), padding=(3, 3), bias=False)
        self.blk9x1 = CustomBlock(
            custom_conv=nn.Conv2d(in_channels=proj_dim, out_channels=proj_dim, kernel_size=(9, 1), 
            padding=(4, 0), bias=False)
            )
        self.blk1x9 = CustomBlock(
            custom_conv=nn.Conv2d(in_channels=proj_dim, out_channels=proj_dim, kernel_size=(1, 9), 
            padding=(0, 4), bias=False)
            )
        self.blk3 = BaseBlock(scale=3, size=fea_size, proj_dim=proj_dim, batch_norm=batch_norm)
        self.blk4 = BaseBlock(scale=4, size=fea_size, proj_dim=proj_dim, batch_norm=batch_norm)
        self.proj_up = nn.Conv2d(in_channels=proj_dim, out_channels=c_in, kernel_size=(1, 1))

    def forward(self, x):
        x = self.proj_down(x)
        x = self.conv7x7(x)
        x = self.blk1x9(x)
        x = self.blk9x1(x)
        x = self.blk4(x)
        x = self.blk3(x)
        x = self.proj_up(x)
        return x