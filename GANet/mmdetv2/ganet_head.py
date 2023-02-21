# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Jinsheng Wang
# @Email   : jswang@stu.pku.edu.cn
# --------------------------------------------------------

import pdb

import numpy as np
import torch
import torch.functional as F
import torch.nn.functional as F
from torch import nn
from ctnet_head import CtnetHead
from collections import Iterable
from mmdetv2.models.builder import HEADS

def compute_locations(shape, device):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], 1)
    return pos


def make_mask(shape=(1, 80, 200), device=torch.device('cuda')):
    x_coord = torch.arange(0, shape[-1], step=1, dtype=torch.float32, device=device)
    x_coord = x_coord.reshape(1, 1, -1)
    # x_coord = np.repeat(x_coord, shape[1], 1)
    x_coord = x_coord.repeat(1, shape[1], 1)
    y_coord = torch.arange(0, shape[-2], step=1, dtype=torch.float32, device=device)
    y_coord = y_coord.reshape(1, -1, 1)
    y_coord = y_coord.repeat(1, 1, shape[-1])
    coord_mat = torch.cat((x_coord, y_coord), axis=0)
    # print('coord_mat shape{}'.format(coord_mat.shape))
    return coord_mat


def make_coordmat(shape=(1, 80, 200), device=torch.device('cuda')):
    x_coord = torch.arange(0, shape[-1], step=1, dtype=torch.float32, device=device)
    x_coord = x_coord.reshape(1, 1, -1)
    # x_coord = np.repeat(x_coord, shape[1], 1)
    x_coord = x_coord.repeat(1, shape[1], 1)
    y_coord = torch.arange(0, shape[-2], step=1, dtype=torch.float32, device=device)
    y_coord = y_coord.reshape(1, -1, 1)
    y_coord = y_coord.repeat(1, 1, shape[-1])
    coord_mat = torch.cat((x_coord, y_coord), axis=0)
    # print('coord_mat shape{}'.format(coord_mat.shape))
    return coord_mat


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
        #                        output_padding=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU()
        # )

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        # out = self.upsample(out)
        out = F.interpolate(input=out, scale_factor=2, mode='bilinear')
        return out

@HEADS.register_module()
class GANetHeadFast(nn.Module):

    def __init__(self,
                 heads,
                 in_channels,
                 num_classes,
                 branch_in_channels=288,
                 hm_idx=0,  # input id for heatmap
                 joint_nums=1,
                 regression=True,
                 upsample_num=0,
                 root_thr=1,
                 train_cfg=None,
                 test_cfg=None):
        super(GANetHeadFast, self).__init__()
        self.root_thr = root_thr
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.joint_nums = joint_nums
        if upsample_num > 0:
            self.upsample_module = nn.ModuleList([UpSampleLayer(in_ch=branch_in_channels, out_ch=branch_in_channels)
                                                  for i in range(upsample_num)])
        else:
            self.upsample_module = None

        self.centerpts_head = CtnetHead(
            heads,
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

        self.keypts_head = CtnetHead(
            heads,
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

        self.offset_head = CtnetHead(
            heads=dict(offset_map=self.joint_nums * 2),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

        self.reg_head = CtnetHead(
            heads=dict(offset_map=2),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

    def ktdet_decode_fast(self, heat, offset, error, thr=0.1, root_thr=1):
        
        def _nms(heat, kernel=3):
            hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
            keep = (hmax == heat).float()  # false:0 true:1
            return heat * keep  # type: tensor
        heat_nms = _nms(heat)    # heat_nms: [1, 1, 40, 100]

        
        # generate root centers array from offset map parallel
        # offset_split: tuple: ([1, 1, 40, 100], [1, 1, 40, 100])
        # pdb.set_trace()
        offset_split = torch.split(offset, 1, dim=1)
        mask = torch.lt(offset_split[1], root_thr)  # offset < 1 is start_point  should be abs(offset) < 1!!!!
        mask_nms = torch.gt(heat_nms, thr)          # key point score > 0.3
        mask_low = mask * mask_nms                  # mask_low: start point map
        mask_low = mask_low[0, 0].transpose(1, 0).detach().cpu().numpy()
        
        # e.g.:
        # hm = np.array([[1,1,1,1],[0,1,0,1],[0,0,0,1]])
        # np.where(hm):
        # whose index is 1: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 1), (2, 3) ]
        # out: (array([0, 0, 0, 0, 1, 1, 2]), array([0, 1, 2, 3, 1, 3, 3]))
        idx = np.where(mask_low)    # idx: start point
        # return start points:
        root_center_arr = np.array(idx, dtype=int).transpose()    # root_center_arr: start points' indice

        # -----------------------------------------------------------------
        # generate roots by coord add offset parallel
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach()
        offset = offset.squeeze(0).permute(1, 2, 0).detach()
        error = error.squeeze(0).permute(1, 2, 0).detach()

        # torch.meshgrid:
        coord_mat = make_coordmat(shape=heat.shape[1:])  # 0.2ms  
        coord_mat = coord_mat.permute(1, 2, 0)
        
        # print('\nkpt thr:', thr)
        heat_mat = heat_nms.repeat(1, 1, 2)
        root_mat = coord_mat + offset   # dcn的偏移量
        align_mat = coord_mat + error    # 修正小数到整数偏移整数

        inds_mat = torch.where(heat_mat > thr)    # 大于thr的关键点本来的位置


        root_arr = root_mat[inds_mat].reshape(-1, 2).cpu().numpy()    # 关键点在修正的位置
        align_arr = align_mat[inds_mat].reshape(-1, 2).cpu().numpy()    # 关键点在误差修正矩阵的位置

        kpt_seeds = []
        for (align, root) in (zip(align_arr, root_arr)):
            # align: np.array([x, y]) root: np.array([x, y])
            kpt_seeds.append((align, np.array(root, dtype=float)))

        # root_center_arr: start_point_index
        # kpts_seeds: [ ([x_error, y_error], [x_loc, y_loc]), ..., ...] 
        return root_center_arr, kpt_seeds

    def forward_train(self, inputs, aux_feat=None):
        # import pdb 
        # pdb.set_trace()
        
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        if self.upsample_module is not None:
            for upsample in self.upsample_module:
                f_hm = upsample(f_hm)
                if aux_feat is not None:
                    aux_feat = upsample(aux_feat)

        z = self.centerpts_head(f_hm)
        cpts_hm = z['hm']    # cpts_hm.shape: [1, 1, 40, 100]

        z_ = self.keypts_head(f_hm)
        kpts_hm = z_['hm']    # kpts_hm.shape: [1, 1, 40, 100]

        if aux_feat is not None:
            f_hm = aux_feat
        o = self.offset_head(f_hm)
        pts_offset = o['offset_map']

        o_ = self.reg_head(f_hm)
        int_offset = o_['offset_map']

        return [cpts_hm, kpts_hm, pts_offset, int_offset]

    # test phase: 
    def forward_test(
            self,
            inputs,
            aux_feat=None,
            hack_seeds=None,
            hm_thr=0.3,
            kpt_thr=0.4,
            cpt_thr=0.4,
    ):  
        
        
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]    # f_hm: [1, 64, 40, 100]
            
        # center points hm
        z = self.centerpts_head(f_hm)
        hm = z['hm']
        hm = torch.clamp(hm.sigmoid(), min=1e-4, max=1 - 1e-4)    # hm.shape: [1, 1, 40, 100]
        
        # key points hm
        z_ = self.keypts_head(f_hm)
        kpts_hm = z_['hm']                                   # kpts_hm.shape: [1, 1, 40, 100]
        kpts_hm = torch.clamp(kpts_hm.sigmoid(), min=1e-4, max=1 - 1e-4)

        # offset map
        if aux_feat is not None:
            f_hm = aux_feat
        o = self.offset_head(f_hm)
        pts_offset = o['offset_map']

        o_ = self.reg_head(f_hm)
        int_offset = o_['offset_map']
        
        # -------------------------------------------------------------------------------------
        if pts_offset.shape[1] > 2:    # False
            def _nms(heat, kernel=3):
                hmax = nn.functional.max_pool2d(heat, (1, 3), stride=(1, 1), padding=(0, 1))
                keep = (hmax == heat).float()  # false:0 true:1
                return heat * keep  # type: tensor

            heat_nms = _nms(kpts_hm)
            offset_split = torch.split(pts_offset, 1, dim=1)
            mask = torch.lt(offset_split[1], self.root_thr)  # offset < 1
            mask_nms = torch.gt(heat_nms, kpt_thr)  # key point score > 0.3
            mask_low = mask * mask_nms
            mask_low = torch.squeeze(mask_low).permute(1, 0).detach().cpu().numpy()
            idx = np.where(mask_low)
            cpt_seeds = np.array(idx, dtype=int).transpose()
            kpt_seeds = self.ktdet_decode(kpts_hm, pts_offset, int_offset,
                                          thr=kpt_thr)  # key point position list[dict{} ]
        # -------------------------------------------------------------------------------------
        else:
            # cpt_seeds: start point
            # kpt_seeds: [ ([x_error, y_error], [x_loc, y_loc]), ..., ...] 
            cpt_seeds, kpt_seeds = self.ktdet_decode_fast(kpts_hm, pts_offset, int_offset, thr=kpt_thr,
                                                          root_thr=self.root_thr)
        return [cpt_seeds, kpt_seeds]

    def inference_mask(self, pos):
        pass

    def forward(
            self,
            x_list,
            hm_thr=0.3,
            kpt_thr=0.4,
            cpt_thr=0.4,
    ):  
        return self.forward_test(x_list, hm_thr, kpt_thr, cpt_thr)

    def init_weights(self):
        # ctnet_head will init weights during building
        pass
    
    def set_grad(self, head_name: str, requires_grad: bool, logger):
        assert hasattr(self, head_name)
        head = getattr(self, head_name)    # e.g. self.keypts_head: CtnetHead
        
        if not isinstance(head, Iterable):
            assert isinstance(head, CtnetHead)
            head_names = head.heads    # dict
            for head_name in head_names:
                logger.info("head: "+ head_name + " 's grad is set to: " + str(requires_grad))
                for params in getattr(head, head_name).parameters():    # head: nn.Conv2d
                    params.requires_grad = requires_grad
        else:
            raise NotImplementedError
        return 
    
    def inference(self, inputs, aux_feat=None, hm_thr=0.3, kpt_thr=0.4, cpt_thr=0.4, decoder=None):
        """_summary_
        input = {
            "features": tuple(outs),
            "aux_feat": aux_feat,    # [1, 64, 40, 100] aux_feat: 自顶向下的FPN
        }
        Args:
            inputs (_type_): _description_
            aux_feat (_type_, optional): _description_. Defaults to None.
            hm_thr (float, optional): _description_. Defaults to 0.3.
            kpt_thr (float, optional): _description_. Defaults to 0.4.
            cpt_thr (float, optional): _description_. Defaults to 0.4.
            decoder (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        x_list = list(inputs['features'])
        f_hm = x_list[self.hm_idx]    # f_hm: [1, 64, 40, 100]
        
        # center points hm
        z = self.centerpts_head(f_hm)
        hm = z['hm']
        hm = torch.clamp(hm.sigmoid(), min=1e-4, max=1 - 1e-4)    # hm.shape: [1, 1, 40, 100]
        
        # key points hm
        z_ = self.keypts_head(f_hm)
        kpts_hm = z_['hm']                                   # kpts_hm.shape: [1, 1, 40, 100]
        kpts_hm = torch.clamp(kpts_hm.sigmoid(), min=1e-4, max=1 - 1e-4)

        # offset map
        if inputs['aux_feat'] is not None:
            f_hm = aux_feat
        
        o = self.offset_head(f_hm)
        pts_offset = o['offset_map']

        o_ = self.reg_head(f_hm)
        int_offset = o_['offset_map']
        
        result = dict(
            start_point_hm=hm,
            kpts_hm=kpts_hm,
            offset=pts_offset,
            error=int_offset
        )
        if decoder is None:
            cpt_seeds, kpt_seeds = self.ktdet_decode_fast(kpts_hm, pts_offset, int_offset, thr=kpt_thr,
                                                                root_thr=self.root_thr)
            result.update({'start_points': cpt_seeds})
            result.update({'kpts_points': kpt_seeds})
        else:
            assert callable(decoder)
            decoder(result, hm_thr, kpt_thr, cpt_thr)
        return result 


@HEADS.register_module()
class Head(nn.Module):
    def __init__(self,
                 heads,
                 in_channels,
                 num_classes,
                 branch_in_channels=288,
                 hm_idx=0,  # input id for heatmap
                 joint_nums=1,
                 regression=True,
                 upsample_num=0,
                 root_thr=1,
                 train_cfg=None,
                 test_cfg=None):
        super(Head, self).__init__()
        self.root_thr = root_thr
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.joint_nums = joint_nums
        if upsample_num > 0:
            self.upsample_module = nn.ModuleList([UpSampleLayer(in_ch=branch_in_channels, out_ch=branch_in_channels)
                                                  for i in range(upsample_num)])
        else:
            self.upsample_module = None

        self.keypts_head = CtnetHead(
            heads,
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

        self.height_head = CtnetHead(
            heads=dict(height=1),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels
        )
        
        self.reg_head = CtnetHead(
            heads=dict(offset_map=2),
            channels_in=branch_in_channels,
            final_kernel=1,
            head_conv=branch_in_channels)

    def forward_train(self, inputs, aux_feat=None):
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        if self.upsample_module is not None:
            for upsample in self.upsample_module:
                f_hm = upsample(f_hm)
                if aux_feat is not None:
                    aux_feat = upsample(aux_feat)

        
        
        z_ = self.keypts_head(f_hm)
        kpts_hm = z_['hm']    # kpts_hm.shape: [1, 1, 40, 100]

        if aux_feat is not None:
            f_hm = aux_feat
        
        z = self.height_head(f_hm)
        hyper_plane = z['height']
        
        o_ = self.reg_head(f_hm)
        int_offset = o_['offset_map']

        return [kpts_hm, int_offset, hyper_plane]


    def inference_mask(self, pos):
        pass

    def forward(
            self,
            x_list,
            hm_thr=0.3,
            kpt_thr=0.4,
            cpt_thr=0.4,
    ):  
        return self.forward_test(x_list, hm_thr, kpt_thr, cpt_thr)

    def init_weights(self):
        # ctnet_head will init weights during building
        pass
    
    def set_grad(self, head_name: str, requires_grad: bool, logger):
        assert hasattr(self, head_name)
        head = getattr(self, head_name)    # e.g. self.keypts_head: CtnetHead
        
        if not isinstance(head, Iterable):
            assert isinstance(head, CtnetHead)
            head_names = head.heads    # dict
            for head_name in head_names:
                logger.info("head: "+ head_name + " 's grad is set to: " + str(requires_grad))
                for params in getattr(head, head_name).parameters():    # head: nn.Conv2d
                    params.requires_grad = requires_grad
        else:
            raise NotImplementedError
        return 
    
    def inference(self, inputs, aux_feat=None, hm_thr=0.3, kpt_thr=0.4, cpt_thr=0.4, decoder=None):
        """_summary_
        input = {
            "features": tuple(outs),
            "aux_feat": aux_feat,    # [1, 64, 40, 100] aux_feat: 自顶向下的FPN
        }
        Args:
            inputs (_type_): _description_
            aux_feat (_type_, optional): _description_. Defaults to None.
            hm_thr (float, optional): _description_. Defaults to 0.3.
            kpt_thr (float, optional): _description_. Defaults to 0.4.
            cpt_thr (float, optional): _description_. Defaults to 0.4.
            decoder (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        x_list = list(inputs['features'])
        f_hm = x_list[self.hm_idx]    # f_hm: [1, 64, 40, 100]
        
        # key points hm
        z_ = self.keypts_head(f_hm)
        kpts_hm = z_['hm']                                   # kpts_hm.shape: [1, 1, 40, 100]
        kpts_hm = torch.clamp(kpts_hm.sigmoid(), min=1e-4, max=1 - 1e-4)

        # offset map
        if inputs['aux_feat'] is not None:
            f_hm = aux_feat
        

        o_ = self.reg_head(f_hm)
        int_offset = o_['offset_map']
        
        z = self.height_head(f_hm)
        hyper_plane = z['height']
        
        result = dict(
            kpts_hm=kpts_hm,
            error=int_offset,
            hyper_plane=hyper_plane
        )

        return result 

