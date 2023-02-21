import pdb
import math
from random import randint
import scipy.interpolate as spi
import bezier
from matplotlib import pyplot as plt
import numpy as np


def get_points_from_hm(hm: np.ndarray, thr=0.9):
    hm_flatten = hm.flatten()
    hm_flatten = np.sort(hm_flatten)
    thr = round(hm_flatten.shape[0] * thr)

    inf = hm > hm_flatten[thr]    # inference mask
    g = dict()                    # graph point
    points = np.array(list(np.where(inf))).T
    for y,x in points:
        if g.get(y, None) is None:
            g[y] = []
        else:
            g[y].append(x)
    for y in g:
        g[y] = np.array(sorted(g[y]))
    return g 

def y_intersect(lane1, lane2):
    """
    return True if the projections of two lanes on y-axis intersect with each other
    :param lane1: s1: vanishing point, e1: ending point
    :param lane2:
    :return:
    """
    s1, s2 = lane1[0][0], lane2[0][0]
    e1, e2 = lane1[-1][0], lane2[-1][0]
    if s2 > e1 or s1 > e2:
        return False
    return True

# return cosine similarity of two vectors
def cos_similarity(r_vanishing, c_vanishing, r1, c1, r2, c2):
    v1 = (r1 - r_vanishing, c1 - c_vanishing)
    v2 = (r2 - r_vanishing, c2 - c_vanishing)
    V1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    V2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
    if V1 == 0 or V2 == 0:
        return 0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (V1 * V2)


def distance(r1, c1, r2, c2):
    return ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5


def in_same_lane(start_pt_r, start_pt_c, r1, c1, r2, c2, distance_thr=5, theta_thr=math.cos(30)):
    if distance(r1, c1, r2, c2) < distance_thr:
        return True
    # if cos_similarity(start_pt_r, start_pt_c, r1, c1, r2, c2) <= theta_thr:
    #     return True
    return False


def denoise(g):
    clusters = dict()
    for row in g:
        # clusters = []
        points = list(g[row]) 
        if len(points) == 0: # points list is empty
            continue  
        
        # cumsum calculation:
        pre = [0] * len(points) 
        pre[0] = points[0] 
        for i in range(len(points) - 1):
            pre[i+1] = points[i+1] + pre[i] 
        
        # get clusters boundary in ordered list: 
        boundarys = []
        for i in range(len(points) - 1):
            if points[i+1] - points[i] > 5:
                boundarys.append(i)    # len(points) - 1 is always unavailable for i 
        boundarys.append(len(points) - 1)
        
        # clusters centers calculation: 
        clusters[row] = []  
        left = -1 
        for right in boundarys: 
            if left < 0:    # the first cluster
                clusters[row].append(pre[right] / (right + 1))     
            else:    # not the 1st cluster
                # assert right > left
                clusters[row].append((pre[right] - pre[left]) / (right - left))    
            left = right  
    return clusters


def lane_assignment(clusters):
    start_pt_r = -1  # row index of start point
    start_pt_c = -1  # column index of start point

    # vanishing point calculation:
    for row in clusters:
        top = clusters.pop(row)  # type: list
        start_pt_c = sum(top) / len(top)
        start_pt_r = row
        break
    start_points = (start_pt_r, start_pt_c)

    # init the lanes:
    # every point is in different lanes
    lanes = []
    for row in clusters:
        for col in clusters[row]:
            lanes.append([(row, col)])
        clusters.pop(row)
        break

    for row in clusters:
        # if row == 26:
        #     pdb.set_trace()
        for i, col in enumerate(clusters[row]):  # every col is in differnet lanes
            # current point: (row, col)
            # binary search the lanes:
            left, right = -1, len(lanes)
            while left + 1 != right:
                mid = (left + right) >> 1
                if lanes[mid][-1][1] <= col:
                    left = mid    # search the last lane that <= curr point
                else:
                    right = mid    # search the first lane that > curr point
            if left == -1:    # left: out of index
                r_dist = distance(row, col, *lanes[right][-1])
                if r_dist <= 5:
                    lanes[right].append((row, col))
                else:
                    lanes.insert(0, [(row, col)])
            elif right == len(lanes):    # right: out of index
                l_dist = distance(row, col, *lanes[left][-1])
                if l_dist <= 5:
                    lanes[left].append((row, col))
                else:
                    lanes.append([(row, col)])
            else:    # there's no out of index
                l_dist = distance(row, col, *lanes[left][-1])
                r_dist = distance(row, col, *lanes[right][-1])
                if l_dist <= r_dist:
                    if l_dist < 5:
                        lanes[left].append((row, col))
                    else:
                        continue    # ignore the point
                else:
                    if r_dist < 5:
                        lanes[right].append((row, col))
                    else:
                        continue    # ignore the point
            lanes = sorted(lanes, key=lambda _lane: _lane[-1][1])

    return dict(
        lanes=lanes,
        vanishing_point=start_points
    )
    

def curve_matching(_lanes: list):
    concated = {
        k: False for k in range(len(_lanes))
    }
    lanes_concat = []
    for i in range(len(_lanes)):
        p1_curve = None
        for j in range(i+1, len(_lanes)):
            if not concated.get(j, False):    # if not
                l1, l2 = np.array(_lanes[i]), np.array(_lanes[j])
                if y_intersect(l1, l2):
                    continue
                # if not intersect with each other:
                if p1_curve is None:
                    p1_curve = np.polyfit(x=l1[:, 0], y=l1[:, 1], deg=1)

                # get the start point and end point:
                col_s2, col_e2 = l2[0][1], l2[-1][1]
                row_s2, row_e2 = l2[0][0], l2[-1][0]

                # evaluate the curve on target row:
                col_s1 = np.polyval(p1_curve, row_s2)
                col_e1 = np.polyval(p1_curve, row_e2)

                if abs(col_e1 - col_e2) <= 5 or abs(col_s1 - col_s2) <= 5:
                    concated[j] = True
                    concated[i] = True
                    if row_s2 > l1[-1][0]:    # s2 > e1
                        lanes_concat.append(_lanes[i] + _lanes[j])
                    else:
                        lanes_concat.append(_lanes[j] + _lanes[i])
        if not concated.get(i, False):
            lanes_concat.append(_lanes[i])
    return lanes_concat


def lanes_throwing(_lanes: list, max_lanes=4):
    if len(_lanes) <= 4:
        return _lanes
    _lanes = [len(_lane) for _lane in sorted(_lanes, key=lambda _lane: len(_lane), reverse=True)]
    return _lanes[:max_lanes]


def B_spline_curve_fitting(_lane: list):
    _lane = np.array(_lane)
    ipo3 = spi.splrep(_lane[:, 0], _lane[:, 1], k=3)
    vanishing_point = _lane[0]
    x = np.arange(vanishing_point[0], 40)
    y = spi.splev(x, ipo3)
    return np.array([x,y]).T

def linear_curve_fitting(_lane: list):
    _lane = np.array(_lane)
    p1 = np.polyfit(_lane[:, 0],_lane[:, 1], 1)
    vanishing_point = _lane[0]
    x = np.arange(vanishing_point[0], 40)
    y = np.polyval(p1, x)
    return np.array([x, y]).T

def bezier_curve_fitting(_lane: list, num_pts=20):
    degree = len(_lane) - 1
    _lane = np.array(_lane).T    #[h, w] -> [h.T; w.T]
    return bezier.Curve(_lane, degree).evaluate_multi(np.linspace(0, 1, num_pts)).T

def lanes_interpolation(_lanes: list, vanishing_point: tuple):
    bezier_lanes = []
    v_row ,v_col = vanishing_point
    for _lane in _lanes:
        s_row = _lane[0][0]    # start row of the lane
        if abs(s_row - v_row) >= 9:
            bezier_lanes.append(bezier_curve_fitting([vanishing_point] + _lane))
        else:
            bezier_lanes.append(bezier_curve_fitting(_lane))
    return bezier_lanes

def scale_up(_lanes: list):
    """
        _lane: point list: [(row1, col1), (row2, col2),...] is in _lanes
        row in [0, 40) and col in [0, 100)
        we scale the point to r in [0, 590 -270) and c in [0, 1640)
    """
    lanes = []
    for _lane in _lanes:
        _lane = np.array(_lane)    # shape: [n_points, 2]
        
        # row:
        _lane[:, 0] *= (32 / 4)    # (590-270) / 40
        _lane[:, 0] += 270 
        
        # col:
        _lane[:, 1] *= 16.4    # 1640 / 100 
        
        lanes.append(_lane) 
    return lanes 

def inference(hm: np.ndarray):
    """_summary_

    Args:
        hm (np.ndarray): _description_
    """
    points_graph = get_points_from_hm(hm, thr=0.9)
    points_graph = denoise(points_graph)
    assignment = lane_assignment(points_graph)
    
    vanishing_point = assignment['vanishing_point']
    lanes = assignment['lanes']
    lanes = curve_matching(lanes)
    lanes = lanes_throwing(lanes, max_lanes=4) 
    lanes = lanes_interpolation(lanes, vanishing_point)
    lanes = scale_up(lanes)
    return lanes

