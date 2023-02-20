import torch
from torch import nn
import torch.nn.functional as F


# shift first and use bi-linear interpolation to handle backward pass
# the floor score \in [0, 1] will recurrently shift the feature map

class HorizontalShiftConv2d(nn.Module):
    def __init__(self, in_c: int, kernel_len: int, stride=(1, 1), bias=False, feature_size=(36, 100)):
        super(HorizontalShiftConv2d, self).__init__()

        # get the size of feature map:
        self.h, self.w = feature_size

        # kernels:
        self.kernel_size = (1, kernel_len)   # 9 as defualt

        # feature aggregators:
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=self.kernel_size,
                              stride=stride, padding=(0, kernel_len//2), bias=bias)

        # to predict shifted index:
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=in_c, out_features=in_c)

    def get_shift_idx(self, scores: torch.Tensor):
        # to get the shifted index for each channel:
        b, c = scores.size()

        # idx_bottom: [b, c] -> [b, c, 1]
        idx_bottom = (torch.floor(self.w * scores.detach()).long() + self.w).view(b, c, 1)

        # repeats channel times:
        # [w, ] -> [c, w] -> [b, c, w] + [b, c, 1] -> [b, c, w]
        idx_bottom = (torch.arange(self.w).repeat(c, 1).repeat(b, 1, 1).to(idx_bottom.device) + idx_bottom) % self.w
        # [b, c, w] -> [h, b, c, w] -> [b, c, h, w]
        idx_bottom = idx_bottom.repeat(self.h, 1, 1, 1).permute(1, 2, 0, 3)
        return idx_bottom

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # [b, c, h, w] -> [b, c, 1, 1] -> [b, c]
        squeezed_feature = self.pooling(x).flatten(1)
        # get the shift scores for horizontal: [b, c]
        horizontal = self.fc(squeezed_feature).contiguous()
        del squeezed_feature

        horizontal = horizontal / torch.max(torch.abs(horizontal), dim=1)[0].view(b, 1)

        # idx_width: [b, c, h, w]
        idx_width = self.get_shift_idx(horizontal)

        # get the shifted feature along horizon (w: dim==3)
        # x.clone(): to forbid modify the gradient inplace
        shifted_feature = x.clone().gather(dim=3, index=idx_width).contiguous()
        del idx_width
        
        # x = x.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()
        horizontal = F.softmax(torch.abs(horizontal), dim=1)

        # suppress the grad if shifted a large step
        x.add_((1 - horizontal).view(b, c, 1, 1) * F.relu(self.conv(shifted_feature)))
        
        return x


class VerticalShiftConv2d(nn.Module):
    def __init__(self, in_c: int, kernel_len: int, stride=(1, 1), bias=False, feature_size=(36, 100)):
        super(VerticalShiftConv2d, self).__init__()

        # get the size of feature map:
        self.h, self.w = feature_size

        # kernels:
        self.kernel_size = (kernel_len, 1)   # 9 as defualt

        # feature aggregators:
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=self.kernel_size,
                              stride=stride, padding=(kernel_len//2, 0), bias=bias)

        # to predict shifted index:
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=in_c, out_features=in_c)

    def get_shift_idx(self, scores: torch.Tensor):
        # to get the shifted index for each channel:
        b, c = scores.size()

        # idx_bottom: [b, c] -> [b, c, 1]
        idx_bottom = (torch.floor(self.h * scores.detach()).long() + self.h).view(b, c, 1)

        # repeats channel times:
        # [h, ] -> [c, h] -> [b, c, h] + [b, c, 1] -> [b, c, h]
        idx_bottom = (torch.arange(self.h).repeat(c, 1).repeat(b, 1, 1).to(idx_bottom.device) + idx_bottom) % self.h
        # [b, c, h] -> [w, b, c, h] -> [b, c, h, w]
        idx_bottom = idx_bottom.repeat(self.w, 1, 1, 1).permute(1, 2, 3, 0)
        return idx_bottom

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # [b, c, h, w] -> [b, c, 1, 1] -> [b, c]
        squeezed_feature = self.pooling(x).flatten(1)
        # get the shift scores for horizontal: [b, c]
        horizontal = self.fc(squeezed_feature).contiguous()
        del squeezed_feature    # to save memory

        horizontal = horizontal / torch.max(torch.abs(horizontal), dim=1)[0].view(b, 1)

        # idx_height: [b, c, h, w]
        idx_height = self.get_shift_idx(horizontal)

        # get the shifted feature along horizon (h: dim==2)
        # x.clone(): to forbid modify the gradient inplace
        shifted_feature = x.clone().gather(dim=2, index=idx_height).contiguous()
        del idx_height      # to save memory

        horizontal = F.softmax(torch.abs(horizontal), dim=1)

        # suppress the grad if shifted a large step
        x.add_((1 - horizontal).view(b, c, 1, 1) * F.relu(self.conv(shifted_feature)))
        
        return x