from torch import nn
import torch


class RESANet(nn.Module):
    def __init__(self, backbone, cascade, decoder, head):
        super(RESANet, self).__init__()
        self.backbone = backbone
        self.resa = cascade
        self.decoder = decoder
        self.heads = head

    def info(self):
        self.backbone.info()
        self.resa.info()
        self.decoder.info()
        self.heads.info()

    def forward(self, batch):
        fea = self.backbone(batch)
        fea = self.resa(fea)

        seg = self.decoder(fea)
        exist = self.heads(fea)

        output = {'seg': seg, 'exist': exist}
        return output
    
    def set_state(self, state):
        # assert state in ['train', 'val']
        self.decoder.set_state(state)
        return 


class MutiHeads(nn.Module):
    def __init__(self, backbone, seg_head, heat_head, exist_head, cascade=nn.Identity()):
        super(MutiHeads, self).__init__()
        self.backbone = backbone
        self.resa = cascade

        self.seg_head = seg_head
        self.heat_head = heat_head
        self.exist_head = exist_head

    def info(self):
        self.backbone.info()

    def forward(self, batch):
        fea = self.backbone(batch)
        fea = self.resa(fea)

        seg = self.seg_head(fea)        # channels : 5        
        heat_map = self.heat_head(fea)    # channels: 1
        exist = self.exist_head(fea)

        output = {'seg': seg, 'exist': exist, 'heat_map':heat_map}
        return output
    
    def set_state(self, state):
        # assert state in ['train', 'val']
        self.seg_head.set_state(state)
        return 
