import torch
from loss.loss_builder import seg_loss, exist_loss, pooling_loss
from torch import nn


def _neg_loss(pred, gt, channel_weights=None,eps=1e-6):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if channel_weights is None:
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
    else:
        pos_loss_sum = 0
        neg_loss_sum = 0
        for i in range(len(channel_weights)):
            p = pos_loss[:, i, :, :].sum() * channel_weights[i]
            n = neg_loss[:, i, :, :].sum() * channel_weights[i]
            pos_loss_sum += p
            neg_loss_sum += n
        pos_loss = pos_loss_sum
        neg_loss = neg_loss_sum
    if num_pos > 2:
        loss = loss - (pos_loss + neg_loss) / num_pos
    else:
        loss = loss - (pos_loss + neg_loss) / 256
        loss = torch.tensor(0, dtype=torch.float32).to(pred.device)
    return loss

class Trainer(object):
    def __init__(self):
        # loss function are on cuda
        self.seg_loss = seg_loss
        self.exist_loss = exist_loss

    def forward(self, net, *args):
        img, label, exist = args

        # output = {'seg': seg, 'exist': exist}
        output = net(img)
        l1 = self.exist_loss(output['exist'], exist)
        l2 = self.seg_loss(output['seg'], label)
        loss = l1 + l2

        return {'exist_loss': l1, 'seg_loss': l2, 'loss': loss}
    
    def __str__(self) -> str:
        return "\nseg_loss: " + self.seg_loss.__str__() \
            + "\nexist_loss:" + self.exist_loss.__str__()


class PreTrainer(object):
    def __init__(self):
        # loss function are on cuda
        self.exist_loss = exist_loss
        self.pooling_loss = pooling_loss

    def forward(self, net, *args):
        img, label, exist = args

        # output = {'seg': seg, 'exist': exist}
        output = net(img)

        # seg[0]: pooling mask, seg[1]: true_mask
        l1 = self.exist_loss(output['exist'], exist)
        l2 = self.pooling_loss(output['seg'], label)
        loss = l1 + l2

        return {'exist_loss': l1, 'pool_loss':l2, 'loss': loss}
    
    def __str__(self) -> str:
        return "\nexist_loss: " + self.exist_loss.__str__() \
            + "\npooling_loss: " + str(self.pooling_loss) \
            + "\nweight: " \
            + ", " + str(self.exist_loss.weight) \
            + ", " + str(self.pooling_loss.weight)