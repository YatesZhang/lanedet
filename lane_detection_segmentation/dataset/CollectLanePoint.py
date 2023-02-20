from functools import cmp_to_key
import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString
import scipy.interpolate as spi
import torch
import torch.nn.functional as F

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def convert_list(p, downscale=None):
    xy = list()
    if downscale is None:
        for i in range(len(p) // 2):
            xy.append((p[2 * i], p[2 * i + 1]))
    else:
        for i in range(len(p) // 2):
            xy.append((p[2 * i] / downscale, p[2 * i + 1] / downscale))
    return xy


def ploy_fitting_cube(line, h, w, sample_num=100):
    line_coords = np.array(line).reshape((-1, 2))
    line_coords = np.array(sorted(line_coords, key=lambda x: x[1]))
    line_coords = line_coords[line_coords[:, 0] > 0, :]
    line_coords = line_coords[line_coords[:, 0] < w, :]

    X = line_coords[:, 1]
    # print('line coords X : {}'.format(X.shape))
    Y = line_coords[:, 0]
    # print('line coords Y : {}'.format(Y.shape))
    if len(X) < 2:
        return None
    new_x = np.linspace(max(X[0], 0), min(X[-1], h), sample_num)

    if len(X) > 3:
        ipo3 = spi.splrep(X, Y, k=3)
        iy3 = spi.splev(new_x, ipo3)
    else:
        ipo3 = spi.splrep(X, Y, k=1)
        iy3 = spi.splev(new_x, ipo3)
    return np.concatenate([iy3[:, None], new_x[:, None]], axis=1)


def clamp_line(line, box, min_length=0):
    left, top, right, bottom = box
    loss_box = Polygon([[left, top], [right, top], [right, bottom],
                        [left, bottom]])
    line_coords = np.array(line).reshape((-1, 2))  # -1 points nums
    if line_coords.shape[0] < 2:
        return None
    try:
        line_string = LineString(line_coords)
        I = line_string.intersection(loss_box)  # line intersection with box
        if I.is_empty:
            return None
        if I.length < min_length:
            return None
        if isinstance(I, LineString):
            pts = list(I.coords)
            return pts
        elif isinstance(I, MultiLineString):
            pts = []
            Istrings = list(I)
            for Istring in Istrings:
                pts += list(Istring.coords)
            return pts
    except:
        return None


class CollectLanePoints(object):
    def __init__(
            self,
            cut_height=240,
            size=(590, 1640)
    ):
        super(CollectLanePoints, self).__init__()
        self.hm_down_scale = 8
        self.radius = 2
        self.root_radius = 6
        self.cut_height = cut_height
        self.h, self.w = size

    def __call__(self, id_classes: np.ndarray, gt_points):
        hm_h = 590 // 8
        hm_w = 1640 // 8
        cut_height = 240 // 8

        id_classes = id_classes.astype('int')
        id_instances = np.arange(4)[id_classes > 0]

        # gt init
        gt_hm = np.zeros((4, hm_h, hm_w), np.float32)
        gt_kpts_hm = np.zeros((4, hm_h, hm_w), np.float32)

        # gt heatmap and ins of bank
        end_points = []
        start_points = []

        for i, pts in zip(id_instances, gt_points):  # per lane
            pts = convert_list(pts, self.hm_down_scale)
            if len(pts) < 2:
                continue
            pts = ploy_fitting_cube(pts, hm_h, hm_w, int(360 / self.hm_down_scale))
            if pts is None:
                continue

            pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))  # down sort by y
            pts = clamp_line(pts, box=[0, 0, hm_w - 1, hm_h - 1], min_length=1)

            # TODO: to draw key points:
            if pts is not None and len(pts) > 1:
                joint_points = []
                start_point, end_point = pts[0], pts[-1]
                end_points.append(end_point)
                start_points.append(start_point)

                joint_points.append(pts[0])

                # TODO: draw gt gaussian heatmap
                for pt in pts:
                    pt_int = (int(pt[0]), int(pt[1]))
                    gt_kpts_hm[i] = draw_umich_gaussian(gt_kpts_hm[i], pt_int, radius=self.radius)  # key points

        # # draw start points
        # if len(start_points) > 0:
        #     for start_point in start_points:
        #         gt_hm[i] = draw_umich_gaussian(gt_hm[i], start_point, radius=self.root_radius)  # start points
        
        # convert to pytorch tensor:
        # gt_hm = torch.from_numpy(gt_hm[:,cut_height, :]).unsqueeze(0)
        gt_kpts_hm = torch.from_numpy(gt_kpts_hm[:, cut_height, :]).unsqueeze(0)
        # gt_hm = F.interpolate(gt_hm, size=[288, 800], mode='nearest')
        gt_kpts_hm = F.interpolate(gt_kpts_hm, size=[288, 800], mode='nearest')
        
        return gt_kpts_hm




