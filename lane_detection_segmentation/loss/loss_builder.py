from mmcv.utils import ConfigDict
from loss.indice_loss import DiceLoss
from loss.crossentropy_loss import CEWithLogitLoss
from loss.indice_loss import PoolingLoss
from .focal_loss import FocalLoss
config = dict(
    seg_weight=2.,
    exist_weight=0.1,
    pooling_weight=2.,
    inplace=False
)
cfg = ConfigDict(config)

seg_loss = DiceLoss(weight=cfg.seg_weight, inplace=cfg.inplace).cuda()
exist_loss = CEWithLogitLoss(weight=cfg.exist_weight).cuda()
pooling_loss = PoolingLoss(weight=cfg.pooling_weight, inplace=cfg.inplace).cuda()
# focal_loss = FocalLoss(alpha=0.5,gamma=2.0,reduction='mean').cuda()


