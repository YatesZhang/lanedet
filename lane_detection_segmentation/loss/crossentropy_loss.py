import torch
from torch import nn


class CELoss(nn.Module):
    def __init__(self, weight=1., ignore_label=255):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')
        self.wight = weight

    # in_tensor: float(), target: long()
    def forward(self, in_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.wight * self.loss(in_tensor, target.long())


class CEWithLogitLoss(nn.Module):
    def __init__(self, weight=1.):
        super(CEWithLogitLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.weight = weight

    # in_tensor: float(), target: float()
    def forward(self, in_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * self.loss(in_tensor, target.float())
