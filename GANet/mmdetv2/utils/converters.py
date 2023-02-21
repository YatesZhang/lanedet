import os 
import torch 
import numpy as np 

def get_sub_name(data):
    return data['img_metas'].data[0][0]['sub_img_name']

def get_img_path(data):
    path = data['img_metas'].data[0][0]['filename']
    assert os.path.exists(path)
    return path 

def get_txt_path(data):
    path = data['img_metas'].data[0][0]['filename'][:-3] + 'lines.txt'
    assert os.path.exists(path)
    return path 

def get_hm(data):
    """_summary_
        get gaussian heat map from data
    Args:
        data (_type_): dict data from data loader
    Returns:
        numpy heat map, shape: (height, width) 
    """
    hm = data['kpts_hm']
    if hasattr(hm, 'data'):    # DataContainer
        t = hm.data
    if isinstance(t, torch.Tensor):    # Tensor
        t = t.cpu().numpy()
    assert isinstance(t, np.ndarray)   # ndarray
    if len(t.shape) == 4:
        t = t[0, 0]
    if len(t.shape) == 3:
        t = t[0]
    return t 


def get(cfg: dict):
    _cfg = cfg.copy()
    if 'type' in _cfg:
        _cfg.pop('type')
    return _cfg