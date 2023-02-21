import random
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook
from mmdetv2.runner.runner import Runner

from mmdetv2.core.evaluation.eval_hooks import (DistEvalHook, EvalHook)
from mmdetv2.core.optimizer.builder import build_optimizer
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger
import pdb

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars

class BatchProcessor(object):
    def __init__(self,
                 show=None,
                 hm_thr=0.3,
                 kpt_thr=0.4,
                 cpt_thr=0.4,
                 points_thr=4,
                 result_dst=None,
                 cluster_thr=4,
                 cluster_by_center_thr=None,
                 group_fast=False,
                 crop_bbox=(0, 270, 1640, 590)
                ):
        self.show = show,
        self.hm_thr = hm_thr,
        self.kpt_thr= kpt_thr,
        self.cpt_thr= cpt_thr,
        self.points_thr= points_thr,
        self.result_dst= result_dst,
        self.cluster_thr = cluster_thr,
        self.cluster_by_center_thr = cluster_by_center_thr,
        self.group_fast= group_fast,
        self.crop_bbox= crop_bbox
    
    def __call__(model, data, train_mode, **kwargs):
        if train_mode:
            losses = model(**data)
            loss, log_vars = parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
            return outputs
        else:
            pass
    pass 

def batch_processor(model, data, train_mode,**kwargs):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)
    
    assert isinstance(dataset, dict)
    
    # 1) prepare data loaders
    data_loaders = dict()
    for k in dataset:
        if k == 'train' or k == 'fine_tune':
            data_loaders[k] = build_dataloader(
                dataset[k],
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed
            )
        elif k == 'test' or k == 'val':
            data_loaders[k] = build_dataloader(
                dataset[k],
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False
            )
        else:
            raise KeyError(f'{k} is not in mode : train, val, test, fine_tune')
    
    # 2) put model on gpus
    if distributed:
        # find_unused_parameters = cfg.get('find_unused_parameters', False)
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    print(model)
    # 3) build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta,
        cfg=cfg)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    optimizer_config = cfg.optimizer_config

    # 4) register hooks
    # register training hooks: 
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    # register val hooks:
    # if 'val' in dataset or 'test' in dataset:
    #     eval_cfg = cfg.get('evaluation', {})
    #     eval_hook = DistEvalHook if distributed else EvalHook
    #     # data_loaders['test'] use the same settings as data_loaders['val']
    #     runner.register_hook(eval_hook(data_loaders['test'], **eval_cfg))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
