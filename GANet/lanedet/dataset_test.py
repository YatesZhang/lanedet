import sys 
sys.path.append('..')
import pdb 
from configs.culane.final_exp_res18_s8 import *
from registry import get 
from mmdet.datasets.culane_dataset import CulaneDataset

dataset_cfg = data['train']
culane = CulaneDataset(**get(dataset_cfg))
pdb.set_trace()