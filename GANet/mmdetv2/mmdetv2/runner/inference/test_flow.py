from .inference_tool import inference, inference_hyper_plane
from .converters import get_hm,get_sub_name
from mmcv.parallel import MMDataParallel
import torch
from tqdm import tqdm 
import os 
from mmdet.utils.general_utils import mkdir


def out_result(results: list, out_file: str):
    # img = np.zeros((590, 1640))
    with open(out_file,'w') as f:  
        for _lane in results: 
            for i in range(len(_lane)):
                col = _lane[i][1]
                row = _lane[i][0]
                print('%.2f %.2f '% (col, row), end='',file=f)
            print('', file=f)  

def infer_from_hyper_plane(model, data_loader, result_dst, logger=None):
    assert isinstance(model, MMDataParallel)
    model.eval()
    if logger is not None:
        logger.info("infer_from_hyper_plane! ")
    with torch.no_grad(): 
        for i, data in tqdm(enumerate(data_loader)): 
            output = model(**data, return_loss=False, inference=True, hm_thr=0.3, kpt_thr=0.4, cpt_thr=0.4, decoder=None)
            
            # infer from heat map only:
            results = inference_hyper_plane(data=output)    
            
            # get output path from data and make directory:
            jpg_relative_path = get_sub_name(data) 
            out_file_path = result_dst + jpg_relative_path[:-3] + 'lines.txt'
            dst_folder = os.path.split(out_file_path)[0]
            mkdir(dst_folder)
            out_result(results=results, out_file=out_file_path)
    pass 

def infer_from_hm(model, data_loader, result_dst, logger=None):
    assert isinstance(model, MMDataParallel)
    model.eval()
    if logger is not None:
        logger.info("do inference from heat map only! ")
    with torch.no_grad(): 
        for i, data in tqdm(enumerate(data_loader)): 
            output = model(**data, return_loss=False, inference=True, hm_thr=0.3, kpt_thr=0.4, cpt_thr=0.4, decoder=None)
            
            # infer from heat map only:
            hm = get_hm(output) 
            results = inference(hm=hm, index=i)    
            
            # get output path from data and make directory:
            jpg_relative_path = get_sub_name(data) 
            out_file_path = result_dst + jpg_relative_path[:-3] + 'lines.txt'
            dst_folder = os.path.split(out_file_path)[0]
            mkdir(dst_folder)
            out_result(results=results, out_file=out_file_path)

            