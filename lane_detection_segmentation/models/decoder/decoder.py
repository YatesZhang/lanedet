from os import stat
from turtle import forward
from grpc import insecure_channel
from torch import nn
import torch.nn.functional as F
from ..necks.psa import BasicBlock as psa_blk
from mmdet.ops.dcn import ModulatedDeformConvPack as Dcn2d
import torch
from config.culane import batch_size


class SegDecoder(nn.Module):
    def __init__(self, cfg):
        super(SegDecoder,self).__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(0.1)
        self.conv1x1 = nn.Conv2d(128,cfg.num_classes,kernel_size=1)
        
        # softmax for channel dim:
        self.softmax = nn.Softmax(dim=1)

    def info(self):
        print("PlainDecoder at models/decoder.py")
    
    def set_state(self, state):
        assert state in ['train', 'val','finetuning','pretraining']
        pass
    
    def forward(self,x):
        x = self.dropout(x)
        x = self.conv1x1(x)
        x = self._nms(x)
        x = self.softmax(x)
        x = F.interpolate(x, size=[self.cfg.img_height,  self.cfg.img_width],
                    mode='bilinear', align_corners=False)
        return x

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
        keep = (hmax == heat).float()  # false:0 true:1
        return heat * keep  # type: tensor

class DcnDecoder(nn.Module):
    def __init__(self, cfg):
        super(DcnDecoder, self).__init__()
        self.cfg = cfg
        num_classes = cfg.num_classes
        self.proj = nn.Conv2d(128, 64, kernel_size=(1, 1))
        self.dropout = nn.Dropout2d(0.1)
        self.dcn3x3 = Dcn2d(in_channels=64, out_channels= 64, kernel_size=3, padding=1)
        self.dcn1x1 = Dcn2d(in_channels=64, out_channels= num_classes, kernel_size=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=num_classes, out_channels= num_classes, kernel_size=1)
        
    def info(self):
        print("PlainDecoder at models/decoder.py")
    
    def set_state(self, state):
        assert state in ['train', 'val','finetuning','pretraining']
        pass
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.proj(x)
        x = self.dcn3x3(x)
        x = self.dcn1x1(x)
        x = self.conv1x1(x)
        x = F.interpolate(x, size=[self.cfg.img_height,  self.cfg.img_width],
                           mode='bilinear', align_corners=False)
        return x


class MemDecoder(nn.Module):
    def __init__(self, cfg):
        super(MemDecoder, self).__init__()
        self.cfg = cfg
        num_classes = cfg.num_classes
        self.register_buffer('label_memo'
        , torch.zeros((batch_size, num_classes, 36, 100)))

        self.proj = nn.Conv2d(128, 64, kernel_size=(1, 1))
        self.dropout = nn.Dropout2d(0.1)
        self.dcn3x3 = Dcn2d(in_channels=64, out_channels= 64, kernel_size=3, padding=1)
        self.dcn1x1 = Dcn2d(in_channels=64, out_channels= num_classes, kernel_size=1, padding=0)

        self.memo_dcn1x1 = Dcn2d(in_channels=num_classes*2, out_channels= num_classes, kernel_size=1)
        self.conv1x1 = nn.Conv2d(in_channels=num_classes, out_channels= num_classes, kernel_size=1)
        
    def info(self):
        print("PlainDecoder at models/decoder.py")
    
    def set_state(self, state):
        assert state in ['train', 'val','finetuning','pretraining']
        pass
    
    def forward(self, x):
        
        x = self.dropout(x)    # x: [b, 128, 36, 100]
        x = self.proj(x)       # -> [b, 64, 36, 100]
        x = self.dcn3x3(x)     
        x = self.dcn1x1(x)     # -> [b, 5, 36, 100]

        # memo : [b, 5, 36, 100] + x: [b, 5, 36, 100] 
        memo = self.label_memo.detach()
        x = torch.cat([x, memo], dim=1)
        x = self.memo_dcn1x1(x)
        x = self.conv1x1(x)
        
        # update the memory:
        self.label_memo = x

        # upsampling:
        x = F.interpolate(x, size=[self.cfg.img_height,  self.cfg.img_width],
                           mode='bilinear', align_corners=False)
        return x

class PlainDecoder(nn.Module):
    def __init__(self, cfg):
        super(PlainDecoder, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2d(0.1)
        self.conv8 = nn.Conv2d(128, cfg.num_classes, 1)

    def info(self):
        print("PlainDecoder at models/decoder.py")
    
    def set_state(self, state):
        assert state in ['train', 'val','finetuning','pretraining']
        pass
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.conv8(x)
        x = F.interpolate(x, size=[self.cfg.img_height,  self.cfg.img_width],
                           mode='bilinear', align_corners=False)
        return x



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        output = self.conv3x1_1(x)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        # +input = identity (residual connection)
        return F.relu(output + x)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, up_width, up_height):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)

        self.bn = nn.BatchNorm2d(noutput, eps=1e-3, track_running_stats=True)

        self.follows = nn.ModuleList()
        self.follows.append(non_bottleneck_1d(noutput, 0, 1))
        self.follows.append(non_bottleneck_1d(noutput, 0, 1))

        # interpolate
        self.up_width = up_width
        self.up_height = up_height
        self.interpolate_conv = conv1x1(ninput, noutput)
        self.interpolate_bn = nn.BatchNorm2d(
            noutput, eps=1e-3, track_running_stats=True)

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        out = F.relu(output)
        for follow in self.follows:
            out = follow(out)

        interpolate_output = self.interpolate_conv(x)
        interpolate_output = self.interpolate_bn(interpolate_output)
        interpolate_output = F.relu(interpolate_output)

        interpolate = F.interpolate(interpolate_output, size=[self.up_height,  self.up_width],
                                    mode='bilinear', align_corners=False)

        return out + interpolate


class BUSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        img_height = cfg.img_height
        img_width = cfg.img_width
        num_classes = cfg.num_classes

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(ninput=128, noutput=64,
                                          up_height=int(img_height)//4, up_width=int(img_width)//4))
        self.layers.append(UpsamplerBlock(ninput=64, noutput=32,
                                          up_height=int(img_height)//2, up_width=int(img_width)//2))
        self.layers.append(UpsamplerBlock(ninput=32, noutput=16,
                                          up_height=int(img_height)//1, up_width=int(img_width)//1))

        self.output_conv = conv1x1(16, num_classes)

    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output
