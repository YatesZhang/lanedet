# from enum import EnumMeta
# import torchvision
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from transform.transforms_builder import train_transforms, test_transforms
import cv2
# from transform.pipeline import Pipeline
from .key_point_map import CollectLanePoints

"""
line item like this:
    ['/driver_23_30frame/05151649_0422.MP4/00000.jpg',
      '/laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png',
      '1',
      '1',
      '1',
      '1']
"""

class TrainPhase(object):
    def __init__(self, cfg):
        self.cut_height = cfg.cut_height
        
        # read list/train_gt.txt in CULaneEval data root
        self.data_root = cfg.data_root
        self.list_path = cfg.train_gt_txt
        
        with open(self.list_path, 'r') as f:
            self.list = f.readlines()

        # build img, label, exist list for train set:
        self.img_list = []      # string: path
        self.label_list = []    # string: path
        self.exist_list = []    # np.array.int: lane instance mask
        # self.k_points_list = [] # string: path

        for line in self.list:
            line = line.split()    # not inplace, split the 'space' and '\n'
            self.img_list.append(self.data_root + line[0])
            self.label_list.append(self.data_root + line[1])
            self.exist_list.append(np.array(line[2:]).astype('int'))
            # self.k_points_list.append(self.data_root + line[0][:-3] +'lines.txt')
        
        # train transform:
        self.transform = train_transforms

    def __getitem__(self, item):
        # get the img:
        img = cv2.imread(self.img_list[item]).astype('float32')     # cv2.IMREAD_UNCHANGED
        img = img[self.cut_height:, ...]                            # cut height 

        # get the label: 
        label = cv2.imread(self.label_list[item])    # label (590, 1640, 3) with the same 3 channels
        label = label[..., 0]                        # to (590, 1640)
        label = label[self.cut_height:, :]           # to (350, 1640)
        
        # get the mask:
        exist = self.exist_list[item]   
        
        # let the img and label in the same transformation: 
        img, label = self.transform((img, label))
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        label = torch.from_numpy(label).contiguous().long()

        # exist transformation:
        exist = torch.from_numpy(exist).float()

        return img, label, exist

    def __len__(self):
        return len(self.list)

class TrainPhaseGaussian(object):
    def __init__(self, cfg):
        # function: input key point in list
        self.get_heat_map = CollectLanePoints(hm_down_scale=8, radius=2, root_radius=6)

        self.cut_height = cfg.cut_height
        
        # read list/train_gt.txt in CULaneEval data root
        self.data_root = cfg.data_root
        self.list_path = cfg.train_gt_txt
        
        with open(self.list_path, 'r') as f:
            self.list = f.readlines()

        # build img, label, exist list for train set:
        self.img_list = []      # string: path
        # self.label_list = []    # string: path
        self.exist_list = []    # np.array.int: lane instance mask
        self.k_points_list = [] # string: path

        for line in self.list:
            line = line.split()    # not inplace, split the 'space' and '\n'
            self.img_list.append(self.data_root + line[0])
            # self.label_list.append(self.data_root + line[1])
            self.exist_list.append(np.array(line[2:]).astype('int'))
            self.k_points_list.append(self.data_root + line[0][:-3] +'lines.txt')
        
        # train transform:
        self.transform = train_transforms

    def __getitem__(self, item):
        # get the img:
        img = cv2.imread(self.img_list[item]).astype('float32')     # cv2.IMREAD_UNCHANGED
        img = img[self.cut_height:, ...]                            # cut height 

        # get the label: 
        label = self.get_heat_map()
        label = cv2.imread(self.label_list[item])    # label (590, 1640, 3) with the same 3 channels
        label = label[..., 0]                        # to (590, 1640)
        label = label[self.cut_height:, :]           # to (350, 1640)
        
        # get the mask:
        exist = self.exist_list[item]   
        
        # let the img and label in the same transformation: 
        img, label = self.transform((img, label))
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        label = torch.from_numpy(label).contiguous().long()

        # exist transformation:
        exist = torch.from_numpy(exist).float()

        return img, label, exist

    def __len__(self):
        return len(self.list)

class TestPhase(object):
    def __init__(self, cfg):
        self.cut_height = cfg.cut_height
        
        # read list/test.txt in CULaneEval data root
        self.data_root = cfg.data_root
        self.list_path = cfg.test_txt

        with open(self.list_path, 'r') as f:
            self.list = f.readlines()

        # build img_list for the test set:
        self.img_list = []
        for idx, line in enumerate(self.list):
            line = line.split()[0]       # no '/n'
            self.list[idx] = line
            self.img_list.append(self.data_root + line)

        # test transform:
        self.transform = test_transforms

    def __getitem__(self, item):
        # get the img:
        img = cv2.imread(self.img_list[item]).astype('float32')
        img = img[self.cut_height:, ...]
        
        img = self.transform((img,))[0]
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        return img, self.list[item]
    
    def __len__(self):
        return len(self.list)


class CULane(Dataset):
    def __init__(self, phase):
        super(CULane, self).__init__()
        self.phase = phase
        self.img_height = 288
        self.img_width = 800
        self.ori_imgh = 590
        self.ori_imgw = 1640
        self.cut_height = 240         # defualt as 0, RESA: 240

    def __getitem__(self, index):
        return self.phase[index]

    def __len__(self):
        return len(self.phase)

