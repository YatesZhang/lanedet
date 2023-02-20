from torch import nn
from .shift_cnn import HorizontalShiftConv2d, VerticalShiftConv2d

module_list = []
for i in range(4):
    module_list.append(nn.Sequential(
        HorizontalShiftConv2d(in_c=128, kernel_len=9, feature_size=(36, 100)),
        VerticalShiftConv2d(in_c=128, kernel_len=9, feature_size=(36, 100)))
    )
shift_conv = nn.Sequential(*module_list)