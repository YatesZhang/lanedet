import os 
import numpy as np 
from .lane_formatting import *


def gt_lines_txt(img_path: str):
    path = img_path[:-3] + 'lines.txt'
    assert os.path.exists(path)
    return path 

def read_lane_lines(path: str, as_tuple=False):
    lines = []
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    if as_tuple:
        res = []
        for i, line in enumerate(lines):
            res.append([])
            for i in range(len(line)//2):
                res[i].append((line[2 * i], line[2 * i + 1]))
        return res
    else:
        lines = [np.array([float(s) for s in line]).reshape((-1, 2)) for line in lines]
        return lines 


def align(line: np.ndarray):    
    
    pass 

def draw_heatmap(lines: list):
    
    pass 