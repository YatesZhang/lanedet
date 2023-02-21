from .heatmap import read_lane_lines
import numpy as np 
from matplotlib import pyplot as plt 
import cv2 
import torch 
from .converters import get_hm, get_img_path
from random import randint


def BGR2RGB(img: np.ndarray):
    if img.shape[0] == 3:
        img = img.transpose((1,2,0))
    return img[:,:,[2,1,0]]

def scale(lane: np.ndarray, scale_factor=(1, 1), y_cut=270):
    """
        to scale the key points (inplace)
        lane: {[x, y] | x in [0, 1640], y in [0, 590]}
    """
    x_scale, y_scale = scale_factor
    lane[:, 0] *= x_scale
    lane[:, 1] -= y_cut
    lane[:, 1] *= y_scale    
    return lane 

def draw_kpts(_lanes: list, y_cut=270, sz=(40, 100)):
    """to draw ground truth of key points
        1) for each lane in _lanes: lane.shape == (n, 2),  n (x, y) points, x in [0, 1640], y in [0, 590] 
        2) cut the image first, and down scale to sz
    Args:
        _lanes (list): read_lane_lines(txt_path)
        y_cut (int, optional): cut the image e.g.: img[cut:, :]. Defaults to 270.
        sz (tuple, optional): down scale to sz. Defaults to (40, 100).
        
    """
    lanes = _lanes.copy()
    scale_factor = (sz[1] / 1640, sz[0] / (590 - y_cut))
    lanes = [scale(lane, scale_factor=scale_factor, y_cut=y_cut) for lane in lanes]
    
    kpts_mask = np.zeros(sz)
    # draw key points:
    for lane in lanes:
        for point in lane:
            # x in [0, sz[1]], y in [0, sz[0]] defaults to x in [0, 100], y in [0, 40]
            point = (int(point[0]), int(point[1]))
            if point[0] < 0 or point[1] < 0 or point[0] >= sz[1] or point[1] >= sz[0]:
                continue    # if out of boundary
            else:
                if sz[0] <= 160 and sz[1] <= 400:     # if image is small 
                    kpts_mask[point[1], point[0]] = 1    # mask
                else:    # if image is large
                    radius = sz[0] // 160 + 1
                    cv2.circle(kpts_mask, point, radius=radius, color=(255,255,255), thickness=-1) 
    return kpts_mask

def draw_lanes_on_img(_lanes: list, img: np.ndarray, y_cut=270, sz=(40, 100)):
    """to draw ground truth of key points
        1) for each lane in _lanes: lane.shape == (n, 2),  n (x, y) points, x in [0, 1640], y in [0, 590] 
        2) cut the image first, and down scale to sz
    Args:
        _lanes (list): read_lane_lines(txt_path)
        y_cut (int, optional): cut the image e.g.: img[cut:, :]. Defaults to 270.
        sz (tuple, optional): down scale to sz. Defaults to (40, 100).
        
    """
    lanes = _lanes.copy()
    scale_factor = (sz[1] / 1640, sz[0] / (590 - y_cut))
    lanes = [scale(lane, scale_factor=scale_factor, y_cut=y_cut) for lane in lanes]
    
    kpts_mask = cv2.resize(img, (sz[1], sz[0]))    # cv2.resize维度是反过来的
    # draw key points:
    for lane in lanes:
        color = (randint(0, 255),randint(0, 255),randint(0, 255))
        for point in lane:
            # x in [0, sz[1]], y in [0, sz[0]] defaults to x in [0, 100], y in [0, 40]
            point = (int(point[0]), int(point[1]))
            if point[0] < 0 or point[1] < 0 or point[0] >= sz[1] or point[1] >= sz[0]:
                continue    # if out of boundary
            else:
                radius = sz[0] // 160 + 1
                cv2.circle(kpts_mask, point, radius=radius, color=color, thickness=-1) 
    return kpts_mask

def show_heatmap(data):
    """_summary_
        1) get numpy array like: (height, width) from shape (b, c, h, w)
        2) plot the heat map
    Args:
        hm (_type_): _description_
    """
    # 1) get numpy array:
    hm = get_hm(data)
    
    # 2) show image:
    plt.imshow(hm) 
    plt.show()     
    

def show_hist(hm: np.ndarray):
    """_summary_

    Args:
        hm (_type_): numpy array, hm.max() <= 1
    Returns:
        numpy array: histogram of an image
    """
    img = hm.ravel()
    if img.max() <= 1 + 1e-6:
        img = (img * 255)
    hist = plt.hist(img.astype('int'), 256)[0]
    plt.show()
    return hist

def show_thr_mask(result_list: list, idx, thr=0.4):
    """_summary_

    Args:
        result_list (list): a list
        idx (_type_): _description_
        thr (float, optional): _description_. Defaults to 0.4.
    """
    data_item = result_list[idx]['out']
    hm = get_hm(data=data_item)
    plt.imshow(hm > thr)
    
    
def add_mask(hm: np.ndarray, mask: np.ndarray):
    """_summary_

    Args:
        hm (np.ndarray): _description_
        mask (np.ndarray): _description_
    """
    pass 

def get_color():
    return np.array([randint(0, 255), randint(0, 255), randint(0, 255)]) / 255


def draw_a_lane(_lane: list, color=None):
    """_summary_
    draw a lane from a point list
    Args:
        _lane (list): element: (row, col) where row in [0, 40), and col in [0, 100)
        color (_type_, optional): _description_. Defaults to None.
    """
    if color is None:
        color = get_color()
    hm = np.zeros((40,100,3))
    for h, w in _lane:
        if h < 0 or w < 0 or h >=40 or w >= 100:
            continue
        hm[round(h),round(w)] = color
    plt.imshow(hm)
    plt.show()


def draw_lanes(_lanes: list):
    """_summary_
    draw a '_lane' (in _lanes) with different random color
    Args:
        _lanes (list): _description_
    """
    colors = []
    for i in range(len(_lanes)):
        colors.append(get_color())
    hm = np.zeros((40,100,3))
    for color, _lane in zip(colors, _lanes):
        for h, w in _lane:
            hm[round(h),round(w)] = color
    plt.imshow(hm)
    plt.show()
    
    
    