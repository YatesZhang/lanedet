import torch
import sys
sys.path.append('..')

from configs.culane.final_exp_res18_s8 import *
from resnet import ResNet
from lanepoints_conv import LanePointsConv
from dcn_fpn import DeformFPN
from mmdet.ops import DeformConv1D
from ganet_head import GANetHeadFast
import pdb 

def get(cfg: dict):
    _cfg = cfg.copy()
    if 'type' in _cfg:
        _cfg.pop('type')
    return _cfg

device = 0
img = torch.randn((1,3,320,800)).cuda(device)
backbone = ResNet(**get(model['backbone'])).cuda(device)
fea = backbone(img)

neck = DeformFPN(**get(model['neck'])).cuda(device)
agg = neck(fea)
# [k for k in agg]: 
# ['features', 'aux_feat', 'deform_points']

head = GANetHeadFast(**get(model['head'])).cuda(device)

out_train = head.forward_train(inputs=agg['features'], aux_feat=agg['aux_feat'])
out_test  = head.forward_test(inputs=agg['features'], aux_feat=agg['aux_feat'])
