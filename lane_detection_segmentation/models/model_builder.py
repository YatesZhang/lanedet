from models.backbone.backbone_builder import backbone
from models.decoder.decoder_builder import decoder, dcn_decoder
from models.heads.head_builder import head
from models.necks.resa_builder import resa, yolof
from models.necks.shift_cnn_builder import shift_conv
from models.necks.yolof import YOLOFplus,YOLOFpro
from models.necks.psa import PSA_s
from models.model import RESANet
import torch
from torch import nn
# output = {'seg': seg, 'exist': exist}

net = RESANet(backbone=backbone, cascade=resa
, decoder=decoder
, head=head)

# net = RESANet(backbone=backbone, 
# cascade=nn.Sequential(PSA_s(inplanes=128,planes=128),
#                 YOLOFpro(batch_norm=True))
#               , decoder=decoder, head=head)
# net = MutiHeads(backbone=backbone, seg_head=seg_head
# , heat_head=heat_map_head,exist_head=head)


# test code:
if __name__ == '__main__':
    img = torch.randn((1, 3, 288, 800))
    out = net(img)
    # print("net output dict(seg, exist): ", out['seg'].shape, out['exist'].shape)
    print("seg:", out['seg'][0].shape, out['seg'][1].shape)
    # flops, params = profile(net, inputs=(img,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print("(flops, params):", flops, params)