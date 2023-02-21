import torch
import pdb

fpn_layer_num = 3                            # check
fpn_down_scale = [8,16,32]                   # check
mask_down_scale = 8                          # check
hm_down_scale = 8                            # check
line_width = 3
radius = 2  # gaussian circle radius
root_radius = 4
vaniehsd_radius = 8
joint_nums = 1                               # check
joint_weights = [1, 0.4, 0.2]                # check
sample_per_lane = [41, 21, 11]               # check
dcn_point_num = [7, 5, 3]                    # check
sample_gt_points = [41, 21, 11]              # check

crop_bbox = [0, 270, 1640, 590]
train_al_pipeline = [
    dict(type='Compose', params=dict(bboxes=False, keypoints=True, masks=False)),
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='Resize', height=320, width=800, p=1),
    dict(type='RandomBrightness', limit=0.2, p=0.6),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6),
    dict(
        type='RandomResizedCrop',
        height=320,
        width=800,
        scale=(0.8, 1.2),
        ratio=(1.7, 2.7),
        p=0.6),
    dict(type='Resize', height=320, width=800, p=1),
]

train_pipeline = [
    dict(type='albumentation', pipelines=train_al_pipeline),
    dict(type='Normalize', **dict(mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3])),
    dict(type='DefaultFormatBundle'),       # to DataContainer to_tensor
    dict(
        type='CollectLanePoints',
        fpn_layer_num=fpn_layer_num,     # 3
        down_scale=mask_down_scale,      # 8
        hm_down_scale=hm_down_scale,     # 8
        max_mask_sample=5,
        line_width=line_width,           # 3
        radius=radius,                   # 2: gaussian circle radius
        root_radius=root_radius,         # 4
        vanished_radius=vaniehsd_radius, # 8
        joint_nums=joint_nums,           # 1
        joint_weights=joint_weights,     # [1, 0.4, 0.2]
        sample_per_lane=sample_per_lane, # [41, 21, 11]
        fpn_down_scale=fpn_down_scale,   # [8,16,32]
        keys=['img', 'gt_cpts_hm', 'gt_kpts_hm', 'int_offset', 'pts_offset',
              'gt_masks', *[f'lane_points_l{i}' for i in range(fpn_layer_num)],
              'offset_mask', 'offset_mask_weight', 'gt_vp_hm'],
        meta_keys=[
            'filename', 'sub_img_name',
            'ori_shape', 'img_shape', 'down_scale', 'hm_down_scale',
            'img_norm_cfg', 'gt_points'
        ]),
]
import sys

sys.path.append('..')
from mmcv.parallel import DataContainer as DC

from configs.culane.final_exp_res18_s8 import model
from mmdet.datasets.culane_dataset import CulaneDataset
from mmdet.models.builder import build_detector

net = build_detector(
   model, train_cfg=model['train_cfg'], test_cfg=model['test_cfg'])
net = net.cuda()
test_mode = False
print("TEST MODE : ", test_mode)

# pdb.set_trace()
culane = CulaneDataset(data_root="/disk/gaoyao/dataset/culane"
, data_list="/disk/gaoyao/dataset/culane/list/train.txt"
, pipeline=train_pipeline, test_mode=test_mode)


for i,data in enumerate(culane):
    if i==0:
        for key in data.keys():
            if isinstance(data[key],DC):
                data[key] = data[key].data
                if isinstance(data[key],torch.Tensor) and len(data[key].shape)<=3:
                    data[key] = data[key].unsqueeze(0)
            print(key,type(data[key]))
            if isinstance(data[key],torch.Tensor):
                data[key] = data[key].cuda()
                # print(data[key].shape)
        img = data['img']
        if test_mode:
            img_metas = None
        else:
            img_metas = data['img_metas']
        data['thr'] = 0.3
        data['kpt_thr'] = 0.3
        data['cpt_thr'] = 0.3
        data.pop('img')
        data.pop('img_metas')
        # pdb.set_trace()
        net(img=img, img_metas=img_metas,**data)
        # print(net)
        break


