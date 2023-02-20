import torch.nn as nn
import torch
import torch.nn.functional as F
from .psa import BasicBlock as psa_blk


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp=128, oup=128, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class RESA(nn.Module):
    def __init__(self, cfg):
        super(RESA, self).__init__()
        self.iter = cfg.iter
        chan = cfg.input_channel
        fea_stride = cfg.fea_stride
        self.height = cfg.img_height // fea_stride
        self.width = cfg.img_width // fea_stride
        self.alpha = cfg.alpha
        conv_stride = cfg.conv_stride

        # i in range(4) as defualt
        # conv_stride = 9 as defualt
        for i in range(self.iter):
            conv_vert1 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride // 2), groups=1, bias=False)
            conv_vert2 = nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride // 2), groups=1, bias=False)

            setattr(self, 'conv_d' + str(i), conv_vert1)
            setattr(self, 'conv_u' + str(i), conv_vert2)

            conv_hori1 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride // 2, 0), groups=1, bias=False)
            conv_hori2 = nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride // 2, 0), groups=1, bias=False)

            setattr(self, 'conv_r' + str(i), conv_hori1)
            setattr(self, 'conv_l' + str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //
                     2 ** (self.iter - i)) % self.height
            setattr(self, 'idx_d' + str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //
                     2 ** (self.iter - i)) % self.height
            setattr(self, 'idx_u' + str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_r' + str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_l' + str(i), idx_l)

    def forward(self, x):
        x = x.clone()

        for direction in ['d', 'u']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))  # x[...,idx,:]

        for direction in ['r', 'l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))  # 做了加法
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x

    def info(self):
        print("resa at models/resa.py")


class RESAX(nn.Module):
    def __init__(self, cfg):
        super(RESAX, self).__init__()
        self.iter = cfg.iter
        chan = cfg.input_channel
        fea_stride = cfg.fea_stride
        self.height = cfg.img_height // fea_stride
        self.width = cfg.img_width // fea_stride
        self.alpha = cfg.alpha
        conv_stride = cfg.conv_stride

        
        # i in range(4) as defualt
        # conv_stride = 9 as defualt
        for i in range(self.iter):
            conv_vert1 = nn.Sequential(nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride // 2), groups=1, bias=False)
                ,nn.BatchNorm2d(num_features=chan, affine=True))
            conv_vert2 = nn.Sequential(nn.Conv2d(
                chan, chan, (1, conv_stride),
                padding=(0, conv_stride // 2), groups=1, bias=False)
                ,nn.BatchNorm2d(num_features=chan, affine=True))

            setattr(self, 'conv_d' + str(i), conv_vert1)
            setattr(self, 'conv_u' + str(i), conv_vert2)

            conv_hori1 = nn.Sequential(nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride // 2, 0), groups=1, bias=False)
                ,nn.BatchNorm2d(num_features=chan, affine=True))

            conv_hori2 = nn.Sequential(nn.Conv2d(
                chan, chan, (conv_stride, 1),
                padding=(conv_stride // 2, 0), groups=1, bias=False)
                ,nn.BatchNorm2d(num_features=chan, affine=True))

            setattr(self, 'conv_r' + str(i), conv_hori1)
            setattr(self, 'conv_l' + str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //
                     2 ** (self.iter - i)) % self.height
            setattr(self, 'idx_d' + str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //
                     2 ** (self.iter - i)) % self.height
            setattr(self, 'idx_u' + str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_r' + str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_l' + str(i), idx_l)

    def forward(self, x):
        x = x.clone()

        for direction in ['d', 'u']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))  # x[...,idx,:]

        for direction in ['r', 'l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))  # 做了加法
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x

    def info(self):
        print("resa at models/resa.py")



class PSA_neck(nn.Module):
    def __init__(self, cfg):
        super(PSA_neck, self).__init__()
        self.iter = cfg.iter
        chan = cfg.input_channel
        fea_stride = cfg.fea_stride
        self.height = cfg.img_height // fea_stride
        self.width = cfg.img_width // fea_stride
        self.alpha = cfg.alpha
        conv_stride = cfg.conv_stride

        # i in range(4) as defualt
        # conv_stride = 9 as defualt
        for i in range(self.iter):
            conv_vert1 = psa_blk(inplanes=128,planes=128,kernel=9)
            conv_vert2 = psa_blk(inplanes=128,planes=128,kernel=9)

            setattr(self, 'conv_d' + str(i), conv_vert1)
            setattr(self, 'conv_u' + str(i), conv_vert2)

            conv_hori1 = psa_blk(inplanes=128,planes=128,kernel=9)
            conv_hori2 = psa_blk(inplanes=128,planes=128,kernel=9)

            setattr(self, 'conv_r' + str(i), conv_hori1)
            setattr(self, 'conv_l' + str(i), conv_hori2)

            idx_d = (torch.arange(self.height) + self.height //
                     2 ** (self.iter - i)) % self.height
            setattr(self, 'idx_d' + str(i), idx_d)

            idx_u = (torch.arange(self.height) - self.height //
                     2 ** (self.iter - i)) % self.height
            setattr(self, 'idx_u' + str(i), idx_u)

            idx_r = (torch.arange(self.width) + self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_r' + str(i), idx_r)

            idx_l = (torch.arange(self.width) - self.width //
                     2 ** (self.iter - i)) % self.width
            setattr(self, 'idx_l' + str(i), idx_l)

    def forward(self, x):
        x = x.clone()

        for direction in ['d', 'u']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx, :])))  # x[...,idx,:]

        for direction in ['r', 'l']:
            for i in range(self.iter):
                conv = getattr(self, 'conv_' + direction + str(i))  # 做了加法
                idx = getattr(self, 'idx_' + direction + str(i))
                x.add_(self.alpha * F.relu(conv(x[..., idx])))

        return x

    def info(self):
        print("resa at models/resa.py")




