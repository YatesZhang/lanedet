# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Jinsheng Wang
# @Email   : jswang@stu.pku.edu.cn
# --------------------------------------------------------

import os
import math
import copy
import random

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from mmcv import Timer


def compute_locations(shape, device):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], 1)
    return pos


def choose_highest_score(group):
    highest_score = -1
    highest_idx = -1
    for idx, _, score in group:
        if score > highest_score:
            highest_idx = idx
    return highest_idx


def choose_mean_point(group):
    group_ = np.array(group).reshape(-1, 2)
    mean_point = np.mean(group_, axis=0, dtype=int)
    return mean_point


def cal_dis(p1, p2):
    result = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return result


def search_groups(coord, groups, thr):
    # coord : A start point: (x, y)
    # groups: list 
    for idx_group, group in enumerate(groups):
        for group_point in group:
            if isinstance(group_point, tuple):    # False
                group_point_coord = group_point[-1]  # center
            else:
                group_point_coord = group_point
            if cal_dis(coord, group_point_coord) <= thr:    # thr <= 5 
                return idx_group
    return -1


def search_groups_by_centers(coord, cluster_centers, cluster_thr):
    for idx_group, cluster_center in enumerate(cluster_centers):
        dis = cal_dis(coord, cluster_center)
        if dis <= cluster_thr:
            return idx_group
    return -1


def group_points(seeds, center_seeds, thr, by_center_thr=None):
    """ False """
    def update_coords(points, cluster_centers, thr=5):
        groups = []
        groups_centers = []
        groups_centers_mean = []

        # group centers first: 聚类起始点
        for cluster_center in cluster_centers:
            idx_group = search_groups(cluster_center, groups_centers, thr)
            if idx_group < 0:
                groups_centers.append([cluster_center])
            else:
                groups_centers[idx_group].append(cluster_center)

        # choose mean center: 计算起始点cluster的均值
        for group_center in groups_centers:
            group_center_new = choose_mean_point(group_center)
            groups_centers_mean.append(group_center_new)

        for idx, (coord, score, center) in enumerate(points):
            idx_group = search_groups(center, groups, thr)
            if idx_group < 0:
                groups.append([(idx, coord, score, center)])
            else:
                groups[idx_group].append((idx, coord, score, center))  # belong to one line or nearby points
        # TODO
        # print('group size: {}   cluster thr: {}'.format(len(groups), thr))
        return groups, groups_centers_mean
    # -----------------------------------------------------------------------
    # TODO group by center
    def update_coords_by_center(points, cluster_centers, thr=5):
        groups = []
        groups_centers = []
        groups_centers_mean = []
        
        # -----------------------part 1: 给候选起始点打标签,以聚类起始点 ------------------------
        # group centers first
        # 使用聚类算法仅仅输入起始点的候选坐标
        for cluster_center in cluster_centers:
            idx_group = search_groups(cluster_center, groups_centers, thr)
            if idx_group < 0:
                groups_centers.append([cluster_center])
            else:
                groups_centers[idx_group].append(cluster_center)
        # group_centers[group_idx] : 第group_idx个起始点簇 
        # -----------------------part 1: 给候选起始点打标签 ------------------------
        
        
        # choose mean center
        # ----------------------计算簇的中心坐标-------------------------------------
        for group_center in groups_centers:
            group_center_new = choose_mean_point(group_center)
            groups_centers_mean.append(group_center_new)
            groups.append([])
        # ----------------------计算簇的中心坐标-------------------------------------
        
        
        # group key points by center
        # -------------------- 算法好处: 自动聚类则不会被车道线条数限制---------------
        for idx, (coord, score, center) in enumerate(points):
            idx_group = search_groups_by_centers(coord=center,
                                                 cluster_centers=groups_centers_mean,
                                                 cluster_thr=thr)
            if idx_group == -1:
                continue
            groups[idx_group].append((idx, coord, score, center))
        return groups, groups_centers_mean

    # -----------------------------------------------------------------------
    points = [(item['align'], item['score'], item['center']) for item in seeds]
    # centers = [(item['coord'], item['score']) for item in center_seeds]
    if by_center_thr is None:
        groups, groups_centers = update_coords(points=points,
                                               cluster_centers=center_seeds,
                                               thr=thr)
    else:
        groups, groups_centers = update_coords_by_center(points=points,
                                                         cluster_centers=center_seeds,
                                                         thr=by_center_thr)
    return groups, groups_centers


def group_points_fast(seeds, center_seeds, thr, by_center_thr=None):
    """ True """
    # import pdb
    # pdb.set_trace()
    def update_coords_fast(points, thr=5):
        """ False """
        groups = []
        groups_centers = []
        for idx, (align, center) in enumerate(points):
            idx_group = search_groups(center, groups, thr)
            if idx_group < 0:
                groups.append([(idx, align, center)])
            else:
                groups[idx_group].append((idx, align, center))  # belong to one line or nearby points
        # TODO
        # print('group size: {}   cluster thr: {}'.format(len(groups), thr))
        return groups, groups_centers

    def update_coords_fast_by_center(points, cluster_centers, thr=5, by_center_thr=5):
        """ True """
        # points: 
        # cluster_centers == cpt_seeds: start point: [[x,y], ..., ...]
        # points == kpt_seeds: [ ([x_error, y_error], [x_loc, y_loc]), ..., ...] 
        groups = []
        groups_centers = []
        groups_centers_mean = []
        # group centers first

        for cluster_center in cluster_centers:
            idx_group = search_groups(cluster_center, groups_centers, thr)

            # idx_group == -1 if we firstly get start in this for loop (i.e. groups_centers is empty)
            if idx_group < 0:
                groups_centers.append([cluster_center])
            else:
                groups_centers[idx_group].append(cluster_center)

        # choose mean center
        for group_center in groups_centers:
            group_center_new = choose_mean_point(group_center)
            groups_centers_mean.append(group_center_new)
            groups.append([])

        # group key points by center
        # import pdb
        # pdb.set_trace()
        for idx, (align, center) in enumerate(points):

            # 距离小于5则可以
            idx_group = search_groups_by_centers(center, groups_centers_mean, by_center_thr)
            if idx_group < 0:
                # groups.append([(idx, align, center)])
                continue
            else:
                groups[idx_group].append((idx, align, center))  # belong to one line or nearby points
        # TODO
        # print('group size: {}  cluster thr: {} cluster by center thr {}'.format(len(groups),
        #                                                                         thr,
        #                                                                         by_center_thr))
        return groups, groups_centers_mean


    if by_center_thr is None:    # False
        groups, groups_centers = update_coords_fast(points=seeds, thr=thr)
    else:    # by_center_thr = 5
        groups, groups_centers = update_coords_fast_by_center(points=seeds,
                                                              cluster_centers=center_seeds,
                                                              thr=by_center_thr,
                                                              by_center_thr=by_center_thr)

    # groups: (idx, align, center)
    # groups_centers:
    return groups, groups_centers


class PostProcessor(object):

    def __init__(self,
                 min_points=5,
                 hm_downscale=16,
                 mask_downscale=8,
                 use_offset=True,
                 cluster_thr=4,
                 cluster_by_center_thr=None,
                 group_fast=False,
                 **kwargs):
        self.min_points = min_points
        self.hm_downscale = hm_downscale
        self.mask_downscale = mask_downscale
        # self.use_offset = use_offset
        self.cluster_thr = cluster_thr
        self.cluster_by_center_thr = cluster_by_center_thr
        self.group_fast = group_fast

    def lane_post_process(self, kpt_groups, cpt_groups, downscale):
        # kpt_groups: [(idx, align, center), ..., ...]
        lanes = []
        cluster_centers = []
        for lane_idx, group in enumerate(kpt_groups):
            points = []
            centers = []
            if len(group) > 1:
                for point in group:
                    points.append([point[1][0] * downscale, point[1][1] * downscale])
                    centers.append([point[-1][0] * downscale, point[-1][1] * downscale])
                # points = ploy_fitting_cube(points, h=320, w=800, sample_num=150)
                lanes.append(
                    dict(
                        id_class=lane_idx,
                        points=points,
                        centers=centers,
                    )
                )
        for center_idx, center in enumerate(cpt_groups):
            cluster_center = [center[0] * downscale, center[1] * downscale]
            cluster_centers.append(
                dict(
                    id_class=center_idx,
                    center=cluster_center,
                )
            )
        return lanes, cluster_centers

    def __call__(self, output, downscale):
        output = list(output)
        cpt_seeds, kpt_seeds = output[0], output[1]
        # print('cpt nums:{} kpt nums:{}'.format(len(cpt_seeds), len(kpt_seeds)))
        # output: __len__ == 2, output is [cpt_seeds, kpt_seeds]
        # cpt_seeds: start point
        # kpt_seeds: [ ([x_error, y_error], [x_offset, y_offset]), ..., ...] 

        if self.group_fast is True:    # True:
            # kpt_groups: (idx, align, center)
            kpt_groups, cpt_groups = group_points_fast(kpt_seeds,
                                                       cpt_seeds,
                                                       self.cluster_thr,
                                                       self.cluster_by_center_thr)
        else:
            kpt_groups, cpt_groups = group_points(kpt_seeds,
                                                  cpt_seeds,
                                                  self.cluster_thr,
                                                  self.cluster_by_center_thr)

        # import pdb
        # pdb.set_trace()                                          
        # kpt_groups: [(idx, align, center), ..., ...]
        lanes, cluster_centers = self.lane_post_process(kpt_groups, cpt_groups, downscale=downscale)

        return lanes, cluster_centers
