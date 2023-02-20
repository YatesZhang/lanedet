import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(tensor_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tensor_in = tensor_in.contiguous().view(tensor_in.size()[0], -1)    # flatten after the batch dim
    target = target.contiguous().view(target.size()[0], -1).float()     # target.flatten(1).contiguous()
    
    a = torch.sum(tensor_in * target, 1)
    b = torch.sum(tensor_in * tensor_in, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return (1-d).mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=1., num_classes=5, inplace=False):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.inplace = inplace

    def forward(self, tensor_in, target):
        if self.inplace:
            target = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)
            tensor_in = F.softmax(tensor_in, dim=1)[:, 1:]
            return self.weight * dice_loss(tensor_in, target)
        else:
            # to avoid calculation inplace:
            _target = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)
            _tensor_in = F.softmax(tensor_in, dim=1)[:, 1:]
            return self.weight * dice_loss(_tensor_in, _target[:, 1:])


class PoolingLoss(nn.Module):
    def __init__(self, weight=1., num_classes=5, inplace=False):
        super(PoolingLoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.inplace = inplace
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, tensor_in, target):
        if self.inplace:
            target = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)
            target = self.avgpool(target.float())[:, 1:]
            tensor_in = F.softmax(tensor_in, dim=1)[:, 1:]
            return self.weight * dice_loss(tensor_in, target)
        else:
            # to avoid calculation inplace:
            _target = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)
            _target = self.avgpool(_target.float()[:, 1:])
            _tensor_in = F.softmax(tensor_in, dim=1)[:, 1:]
            return self.weight * dice_loss(_tensor_in, _target)

