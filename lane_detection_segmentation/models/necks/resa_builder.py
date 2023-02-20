from .resa import RESA, PSA_neck, RESAX, CoordAtt
from mmcv.utils import ConfigDict
import torch
from .psa import GAM_Attention
from torch import nn
from .yolof import YOLOFplus


config = dict(
    iter=4,
    input_channel=128,
    fea_stride=8,
    img_height=288,
    img_width=800,
    alpha=2.0,
    conv_stride=9
)
cfg = ConfigDict(config)
# resa = nn.Sequential(GAM_Attention(in_channels=128, out_channels=128), RESA(cfg=cfg))
resa = CoordAtt()
# psa = PSA_neck(cfg=cfg)


yolof = nn.Sequential(GAM_Attention(in_channels=128, out_channels=128)
,*[YOLOFplus() for i in range(4)])

# test code:
if __name__ == '__main__':
    feature_map = torch.randn((1, 128, 36, 100))
    print(resa(feature_map).shape)






