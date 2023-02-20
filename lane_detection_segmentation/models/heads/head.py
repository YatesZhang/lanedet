import torch
from torch import nn
import torch.nn.functional as F


class ExistHead(nn.Module):
    def __init__(self, cfg=None):
        super(ExistHead, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2d(0.1)  # ???
        self.conv8 = nn.Conv2d(128, cfg.num_classes, 1)

        stride = cfg.fea_stride * 2
        self.fc9 = nn.Linear(
            int(cfg.num_classes * cfg.img_width / stride * cfg.img_height / stride), 128)
        self.fc10 = nn.Linear(128, cfg.num_classes - 1)

    def info(self):
        print("ExistHead")

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv8(x)

        x = F.softmax(x, dim=1)
        x = F.avg_pool2d(x, 2, stride=2, padding=0)
        x = x.view(-1, x.numel() // x.shape[0])
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)
        x = torch.sigmoid(x)

        return x


class DCNHead(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(DCNHead, self).__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv3x3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3),
                                 padding=(1, 1))
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=cfg.num_classes, kernel_size=(1, 1))

    def set_state(self, state):
        # assert state in ['train', 'val']
        pass

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv3x3(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x = self._nms(x)
        x = F.interpolate(x, size=[self.cfg.img_height, self.cfg.img_width],
                          mode='bilinear', align_corners=False)
        return x
