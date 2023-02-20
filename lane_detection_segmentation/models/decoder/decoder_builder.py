from models.decoder.decoder import PlainDecoder,DcnDecoder, MemDecoder
from mmcv.utils import ConfigDict
import torch


config = dict(
    num_classes=4+1,
    img_height=288,
    img_width=800
)
cfg = ConfigDict(config)

decoder = PlainDecoder(cfg)
# dcn_decoder = DcnDecoder(cfg)
dcn_decoder = MemDecoder(cfg)
# decoder_busd = BUSD(cfg)
# decoder_c2f = Coarse2fine(cfg)
# decoder_sa = SADecoder(cfg)
# test code:
if __name__ == '__main__':
    feature_map = torch.randn((1, 128, 36, 100))
    # print(decoder_c2f(feature_map))