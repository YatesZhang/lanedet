from .head import ExistHead
from mmcv.utils import ConfigDict
import torch


config = dict(
    num_classes=4+1,
    img_height=288,
    img_width=800,
    fea_stride=8,
    conv_stride=8
)
cfg = ConfigDict(config)

head = ExistHead(cfg)

# test code:
if __name__ == '__main__':
    feature_map = torch.randn((1, 128, 36, 100))
    print(head(feature_map).shape)