from mmdet.ops import DeformConv1D
import numpy as np
import torch
import pdb


if __name__ == '__main__':
    feat_channels = 1
    point_feat_channels = 14
    dcn_kernel = 7
    dcn_pad = 3
    dcn = DeformConv1D(feat_channels,point_feat_channels,dcn_kernel, 1, dcn_pad)
    dcn = torch.nn.parallel.DataParallel(dcn).cuda()
    offset = np.array([0, -3, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 0, 3]).reshape((1, -1, 1, 1))
    offset = torch.from_numpy(offset).cuda()
    x = torch.ones((1, feat_channels, 10, 10)).cuda()
    
    out = dcn(x, x+offset)
    pdb.set_trace()
    
    