import torch
import torchvision
import transform.transforms as T
import numpy as np
import albumentations as A
import mmcv
import torch.nn.functional as F


def resize_lable(img,size=(288,800)):
    assert img.shape[0]==350 and img.shape[1]==820
    # 350, 820 as defualt
    src_h, src_w = img.shape[0], img.shape[1]
    h, w = size
    x = np.arange(h)    # 288
    y = np.arange(w)    # 800
    
    x = np.round(x * (src_h/h)).astype('int')
    y = np.round(y * (src_w/w)).astype('int')

    img = img[x,:]
    img = img[:,y]
    return img

def one_hot(label):
    buffer = np.zeros((5, *label.shape))
    buffer[0,:,:] = (label)<=1e-6
    buffer[1,:,:] = (label-1)<=1e-6
    buffer[2,:,:] = (label-2)<=1e-6
    buffer[3,:,:] = (label-3)<=1e-6
    buffer[4,:,:] = (label-4)<=1e-6
    return buffer.astype('int')

class Pipeline(object):
    def __init__(self, train=True) -> None:
        if train:
            self.img_transforms = A.Compose([
                A.Resize(height=288, width=800,p=1),
                A.RandomBrightness(limit=0.2, p=0.3),
                A.Blur(blur_limit=3, p=0.2),
            ])
        else:
            self.img_transforms = A.Compose([
                A.Resize(height=288, width=800,p=1)
            ])
        # img normalize:
        self.mean = np.array([103.939, 116.779, 123.68])
        self.std = np.array([1., 1., 1.])

        # self.mean = np.array([75.3, 76.6, 77.6])
        # self.std = np.array([50.5, 53.8, 54.3])
        
        self.to_rgb = False
        self.train = train

    def __call__(self,img: np.array, label=None):

        # img: (350, 1640, 3) -> (288, 800, 3) -> (3, 288, 800)
        img = self.img_transforms(image=img)['image']
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        
        if self.train:
            # label: (350, 1640) -> (350, 820)
            label = label[:,::2]
            label = resize_lable(label)               
            label = torch.from_numpy(label).contiguous().long()
            return img, label
        else:
            return img
