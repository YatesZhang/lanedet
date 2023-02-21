# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Yinchao Ma
# @Email   : imyc@mail.ustc.edu.cn
# --------------------------------------------------------


import math
import os

import torch

from mmdet.core import build_assigner

from .builder import DETECTORS, build_loss
from .single_stage import SingleStageDetector
import pdb 

@DETECTORS.register_module
class GANet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss='LaneLossAggress',
                 loss_weights={},
                 output_scale=4,
                 num_classes=1,
                 point_scale=True,
                 sample_gt_points=[11, 11, 11, 11],
                 assigner_cfg=dict(type='LaneAssigner'),
                 use_smooth=False):
        super(GANet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=head,
            train_cfg=None,
            test_cfg=None,
            pretrained=pretrained)
        self.sample_gt_points = sample_gt_points
        self.num_classes = num_classes
        self.head = head
        self.use_smooth = use_smooth
        self.assigner_cfg = assigner_cfg
        self.loss_weights = loss_weights
        self.point_scale = point_scale
        if test_cfg is not None and 'out_scale' in test_cfg.keys():
            self.output_scale = test_cfg['out_scale']
        else:
            self.output_scale = output_scale
        self.loss = build_loss(loss)
        # if self.assigner_cfg:
        #     self.assigner = build_assigner(self.assigner_cfg)

    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        # pdb.set_trace()
        if kwargs.get('inference', False):
            return self.inference(img, img_metas, **kwargs)
        elif return_loss:
            return self.forward_train(img, img_metas, **kwargs)  # img shape [8 3 320 800]
        else:  # test:
            return self.forward_test(img, img_metas, **kwargs)
        
    def set_loss_dict(self, loss_dict:dict):
        for name in loss_dict:
            self.set_loss_weight(name, loss_dict[name])
        return 
    
    def set_loss_weight(self,name: str, val: float):
        assert name in self.loss_weights
        self.loss_weights[name] = val
        return 
    
    def forward_train(self, img, img_metas, **kwargs):
        # kwargs -> ['exist_mask', 'instance_mask', 'gauss_mask', 'hm_down_scale', 'lane_points']
        output = self.backbone(img.type(torch.cuda.FloatTensor))  # shape [8 128 80 200]  swin [B C 40 100]
        
        output = self.neck(output)  # train: (features, aux_feat)

        # aux_feat不存在时候默认返回None
        [cpts_hm, kpts_hm, pts_offset, int_offset] = self.bbox_head.forward_train(output['features'],
                                                                                    output.get("aux_feat", None))
        cpts_hm = torch.clamp(torch.sigmoid(cpts_hm), min=1e-4, max=1 - 1e-4)
        kpts_hm = torch.clamp(torch.sigmoid(kpts_hm), min=1e-4, max=1 - 1e-4)
        
        # calculate loss:
        # loss_weights=dict(center=0, point=1.0, error=1.0, offset=0.5, aux=0)
        loss_items = []
        if self.loss_weights["center"] > 0:    # 0 as default    
            loss_items.append({"type": "focalloss"
                               , "gt": kwargs['gt_cpts_hm'], "pred": cpts_hm
                               , "weight": self.loss_weights["center"]})
            
        if self.loss_weights["point"] > 0:
            loss_items.append({"type": "focalloss", "gt": kwargs['gt_kpts_hm'], "pred": kpts_hm
                               , "weight": self.loss_weights["point"]})
        if self.loss_weights["error"] > 0:
            loss_items.append({"type": "regl1kploss", "gt": kwargs['int_offset'], "pred": int_offset,
                                "mask": kwargs['offset_mask'], "weight": self.loss_weights["error"]})
        if self.loss_weights["offset"] > 0:
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_offset'], "pred": pts_offset,
                                "mask": kwargs['offset_mask_weight'], "weight": self.loss_weights["offset"]})
        losses = self.loss(loss_items)
        return losses

    def test_inference(self, img, hack_seeds=None, **kwargs):
        """Test without augmentation."""
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)  # shape [8 64 80 200]
        [cpts_hm, kpts_hm, pts_offset, int_offset] = self.bbox_head.forward_train(output['features'],
                                                                                    output.get("aux_feat", None))
        seeds, hm = self.bbox_head.forward_test(output['features'], output.get("aux_feat", None), hack_seeds,
                                                kwargs['thr'], kwargs['kpt_thr'],
                                                kwargs['cpt_thr'])
        output['cpts_hm'] = cpts_hm
        output['kpts_hm'] = kpts_hm
        output['pts_offset'] = pts_offset
        output['int_offset'] = int_offset
        output['deform_points'] = output['deform_points']
        output['seeds'] = seeds
        output['hm'] = hm
        return output

    def forward_test(self, img, img_metas,
                     hack_seeds=None,
                     **kwargs):
        """Test without augmentation."""
        """
        output in backbone: tuple
        ([1,64,80,200],[1,128,40,100],[1,256,20,50],[1,512,10,25])
        """
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        """
        output in neck: dict{
            'features': tuple(3){[1,64,40,100],[1,64,20,50],[1,64,10,25]}
            'aux_feat': [1,64,40,100]
            'deform_points': tuple(3){[1,14,40,100],None,None},
        }
        ([1,64,80,200],[1,128,40,100],[1,256,20,50],[1,512,10,25])
        """        
        output = self.neck(output)
        if self.head:
            seeds, hm = self.bbox_head.forward_test(output['features'], output.get("aux_feat", None), hack_seeds,
                                                    kwargs['thr'], kwargs['kpt_thr'],
                                                    kwargs['cpt_thr'])
        # seeds: start point
        # hm: [ ([x_error, y_error], [x_loc, y_loc]), ..., ...] 

        return [seeds, hm]
    def forward_fune_tune(self, x):
        
        pass
    
    def forward_dummy(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        x = self.bbox_head.forward_train(x['features'], x.get("aux_feat", None))
        return x
    
    def inference(self, img, img_metas, **kwargs):
        hm_thr = kwargs['hm_thr']
        kpt_thr = kwargs['kpt_thr']
        cpt_thr = kwargs['cpt_thr']
        decoder = kwargs['decoder']
        
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)
        # @test_code/ganet_head.py in class: GANetHeadFast
        output = self.bbox_head.inference(output, aux_feat=output.get("aux_feat", None)
                                          , hm_thr=hm_thr, kpt_thr=kpt_thr, cpt_thr=cpt_thr, decoder=decoder)
        return output 


@DETECTORS.register_module
class HyperLaneNet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss='LaneLossAggress',
                 loss_weights={},
                 output_scale=4,
                 num_classes=1,
                 point_scale=True,
                 sample_gt_points=[11, 11, 11, 11],
                 assigner_cfg=dict(type='LaneAssigner'),
                 use_smooth=False):
        super(HyperLaneNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=head,
            train_cfg=None,
            test_cfg=None,
            pretrained=pretrained)
        self.sample_gt_points = sample_gt_points
        self.num_classes = num_classes
        self.head = head
        self.use_smooth = use_smooth
        self.assigner_cfg = assigner_cfg
        self.loss_weights = loss_weights
        self.point_scale = point_scale
        if test_cfg is not None and 'out_scale' in test_cfg.keys():
            self.output_scale = test_cfg['out_scale']
        else:
            self.output_scale = output_scale
        self.loss = build_loss(loss)
        # if self.assigner_cfg:
        #     self.assigner = build_assigner(self.assigner_cfg)

    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        # pdb.set_trace()
        if kwargs.get('inference', False):
            return self.inference(img, img_metas, **kwargs)
        elif return_loss:
            return self.forward_train(img, img_metas, **kwargs)  # img shape [8 3 320 800]
        else:  # test:
            return self.forward_test(img, img_metas, **kwargs)
        
    def set_loss_dict(self, loss_dict:dict):
        for name in loss_dict:
            self.set_loss_weight(name, loss_dict[name])
        return 
    
    def set_loss_weight(self,name: str, val: float):
        assert name in self.loss_weights
        self.loss_weights[name] = val
        return 
    
    def forward_train(self, img, img_metas, **kwargs):
        # kwargs -> ['exist_mask', 'instance_mask', 'gauss_mask', 'hm_down_scale', 'lane_points']
        output = self.backbone(img.type(torch.cuda.FloatTensor))  # shape [8 128 80 200]  swin [B C 40 100]
        
        output = self.neck(output)  # train: (features, aux_feat)

        # aux_feat不存在时候默认返回None
        [kpts_hm, int_offset, hyper_plane] = self.bbox_head.forward_train(output['features'],
                                                                                    output.get("aux_feat", None))
        
        kpts_hm = torch.clamp(torch.sigmoid(kpts_hm), min=1e-4, max=1 - 1e-4)
        
        # calculate loss:
        # loss_weights=dict(center=0, point=1.0, error=1.0, offset=0.5, aux=0)
        loss_items = []
            
        if self.loss_weights["point"] > 0:
            loss_items.append({"type": "focalloss", "gt": kwargs['gt_kpts_hm'], "pred": kpts_hm
                               , "weight": self.loss_weights["point"]})
        if self.loss_weights["error"] > 0:
            loss_items.append({"type": "regl1kploss", "gt": kwargs['int_offset'], "pred": int_offset,
                                "mask": kwargs['offset_mask'], "weight": self.loss_weights["error"]})
        if self.loss_weights["offset"] > 0:
            loss_items.append({"type": "regl1kploss", "gt": kwargs['hyper_plane'], "pred": hyper_plane,
                                "mask": kwargs['offset_mask_weight'], "weight": self.loss_weights["offset"]})
        losses = self.loss(loss_items)
        return losses

    def test_inference(self, img, hack_seeds=None, **kwargs):
        """Test without augmentation."""
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)  # shape [8 64 80 200]
        [cpts_hm, kpts_hm, pts_offset, int_offset] = self.bbox_head.forward_train(output['features'],
                                                                                    output.get("aux_feat", None))
        seeds, hm = self.bbox_head.forward_test(output['features'], output.get("aux_feat", None), hack_seeds,
                                                kwargs['thr'], kwargs['kpt_thr'],
                                                kwargs['cpt_thr'])
        output['cpts_hm'] = cpts_hm
        output['kpts_hm'] = kpts_hm
        output['pts_offset'] = pts_offset
        output['int_offset'] = int_offset
        output['deform_points'] = output['deform_points']
        output['seeds'] = seeds
        output['hm'] = hm
        return output

    def forward_test(self, img, img_metas,
                     hack_seeds=None,
                     **kwargs):
        """Test without augmentation."""
        """
        output in backbone: tuple
        ([1,64,80,200],[1,128,40,100],[1,256,20,50],[1,512,10,25])
        """
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        """
        output in neck: dict{
            'features': tuple(3){[1,64,40,100],[1,64,20,50],[1,64,10,25]}
            'aux_feat': [1,64,40,100]
            'deform_points': tuple(3){[1,14,40,100],None,None},
        }
        ([1,64,80,200],[1,128,40,100],[1,256,20,50],[1,512,10,25])
        """        
        output = self.neck(output)
        if self.head:
            seeds, hm = self.bbox_head.forward_test(output['features'], output.get("aux_feat", None), hack_seeds,
                                                    kwargs['thr'], kwargs['kpt_thr'],
                                                    kwargs['cpt_thr'])
        # seeds: start point
        # hm: [ ([x_error, y_error], [x_loc, y_loc]), ..., ...] 

        return [seeds, hm]
    def forward_fune_tune(self, x):
        
        pass
    
    def forward_dummy(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        x = self.bbox_head.forward_train(x['features'], x.get("aux_feat", None))
        return x
    
    def inference(self, img, img_metas, **kwargs):
        hm_thr = kwargs['hm_thr']
        kpt_thr = kwargs['kpt_thr']
        cpt_thr = kwargs['cpt_thr']
        decoder = kwargs['decoder']
        
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)
        # @test_code/ganet_head.py in class: GANetHeadFast
        output = self.bbox_head.inference(output, aux_feat=output.get("aux_feat", None)
                                          , hm_thr=hm_thr, kpt_thr=kpt_thr, cpt_thr=cpt_thr, decoder=decoder)
        return output 
