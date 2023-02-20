import torch
from mmcv.utils import ConfigDict
from .resnet import ResNetWrapper
config = dict(
    resnet='resnet18',
    pretrained=True,
    progress=True,
    in_channels=[64, 128, 256, 512],
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    psa=False
)

cfg = ConfigDict(config)
backbone = ResNetWrapper(cfg)


# test code to observe the output shape of the backbone
if __name__ == '__main__':
    img = torch.randn((1, 3, 288, 800))
    print(backbone)
    print(backbone(img).shape)

