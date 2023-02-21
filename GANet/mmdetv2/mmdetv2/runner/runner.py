# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os.path as osp
import time

import torch

import mmcv
from .checkpoint import load_checkpoint, save_checkpoint
from .dist_utils import get_dist_info
from .hooks import HOOKS, IterTimerHook

from mmcv.runner.hooks import Hook
from mmcv.parallel import MMDataParallel,MMDistributedDataParallel
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import get_host_info, get_time_str, obj_from_dict
from .test_dataset import single_gpu_test
from .inference.test_flow import infer_from_hm, infer_from_hyper_plane


from .eval import summarize
import os 
import pdb 
from torch.nn.parallel import DataParallel,DistributedDataParallel
from collections import Iterable



def is_module_wrapper(model):
    # DistributedDataParalled is not Implemented
    if isinstance(model, MMDataParallel) or isinstance(model, DataParallel):
        return True
    elif isinstance(model, DistributedDataParallel) or isinstance(model, MMDistributedDataParallel):
        raise NotImplementedError
    else:
        return False

class Runner(object):
    """A training helper for PyTorch.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 meta=None,
                 cfg=None):
        assert callable(batch_processor)
        self.model = model
        self.cfg = cfg
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.batch_processor = batch_processor

        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        if meta is not None:
            assert isinstance(meta, dict), '"meta" must be a dict or None'
        self.meta = meta

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self.start_epoch = self._epoch
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(optimizer, torch.optim,
                                      dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                f'but got {type(optimizer)}')
        return optimizer

    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = f'{self.timestamp}.log'
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list: Current momentum of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        momentums = []
        for group in self.optimizer.param_groups:
            if 'momentum' in group.keys():
                momentums.append(group['momentum'])
            elif 'betas' in group.keys():
                momentums.append(group['betas'][0])
            else:
                momentums.append(0)
        return momentums

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        if not isinstance(hook, Hook):
            pdb.set_trace()
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader

        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1    # @property self.epoch = self._epoch
    
    def after_test_epoch(self):
        assert hasattr(self, 'result_dst')
        summarize(result_dst=self.result_dst, data_root=self.cfg.data_root,logger=self.logger)
        
    def set_model_grad(self, requires_grad):
        model = self.model
        
        # get the real model from wrapper:
        if is_module_wrapper(self.model):
            model = model.module
        
        for name, param in model.named_parameters():
            param.requires_grad = requires_grad
            
            # update logger:
            if requires_grad == False:
                self.logger.info('module: ' + name + "is frozen")
            else:
                self.logger.info('module: ' + name + "is unfrozen")
        return 
    
    def fine_tune(self, data_loaders, **kwargs):
        """
        inner function: to_fine_tune_head
        self.cfg.fine_tuing_work_flow[head_name]: 
            [(train,5),(test,1),(train,5),(test,1),(train,5),(test,1)] 
        
        progress:
        1) fine-tuning model from path: (if not given, raise ERROR) 
        2) freeze the backbone, neck, head in GANet:  reference:
            https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/how_to.html?highlight=hook#id3
        3) for head in GANetHeadFast:
                unfreeze this head
                head.forward_loss = self.cfg.loss[head_name]
                logger update info
                for epochs, flow in self.cfg.fine_tuing_work_flow[head_name]:
                    train(model)
                freeze this head
        4) unfreeze the model
        """
        
        # fine-tune from the the checkpoint
        checkpoint_path = self.cfg.resume_from
        if checkpoint_path is None:
            checkpoint_path = self.cfg.work_dir + '/epoch_60.pth'
        self.logger.info("resume from the model: " + checkpoint_path)
        
        # load model and freeze the grad, resume_optimizer if False !!!
        self.resume(checkpoint=checkpoint_path, resume_optimizer=False)
        self.set_model_grad(False)
        
        # funtuning: 
        
        # loss_weights = dict(center=0.0,
        #             point=1.0,
        #             error=1.0,
        #             offset=0.5,
        #             aux=0.2                    
        #         )
        
        # get cfg:
        
        cfg = self.cfg.fine_tune
        heads = cfg.head_names
        loss_cfg = cfg.loss_cfg
        workflows = cfg.workflows
        
        for head in heads:
            # --------------------before fine-tuning------------------------------
            self.logger.info("fine-tuning phase on head: " + head)
            
            # get workflow:
            workflow = workflows[head]
            self.logger.info("work flow for fine-tuning phase :" + str(workflow))
            
            # set loss weight and freeze the head:
            loss_dict = loss_cfg[head]
            self.logger.info("loss weight is set to: " + str(loss_dict))
            self.logger.info("the head: " + head + " will be unfrozen")
            
            is_wrapper = is_module_wrapper(self.model)
            if is_wrapper:
                self.model.module.set_loss_dict(loss_dict)
                # @GANetheadFast.freeze(self, head_name: str)
                self.model.module.bbox_head.set_grad(head, True, self.logger)
            else:
                self.model.set_loss_dict(loss_dict)
                self.model.bbox_head.set_grad(head, True, self.logger)
            # --------------------before fine-tuning------------------------------
            
            # get runner and run fine-tuning workflow:
            for flow, epochs in workflow:
                if not hasattr(self, flow):
                    raise ValueError(f"runner doesn't has mode: {flow}")
                epoch_runner = getattr(self, flow)
                for epoch in range(epochs):
                    epoch_runner(data_loaders[flow], **kwargs)
                pass
            # fine-tuning workflow finished
            
            # --------------------after fine-tuning--------------------------------
            # loss_dict = self.cfg.loss_weights
            # self.logger.info("fine-tuning finished, loss weight is set to: " + str(loss_dict))
            self.logger.info("fine-tuning finished, the head: " + head + " will be frozen")
            if is_wrapper:
                self.model.module.set_loss_dict(loss_dict)
                # @GANetheadFast.freeze(self, head_name: str)
                self.model.module.bbox_head.set_grad(head, False, self.logger)
            else:
                self.model.set_loss_dict(loss_dict)
                self.model.bbox_head.set_grad(head, False, self.logger)
            # --------------------after fine-tuning--------------------------------  
        
        self.set_model_grad(True)       
        return 
    
    def test(self, data_loader, **kwargs):
        # get path: 
        self.result_dst = self.cfg.work_dir + '/result'
        self.model_dst = self.cfg.work_dir + '/latest.pth'
        
        self.logger.info("OUTPUT reslut txt files to: " + self.result_dst)
        
        # freeze the params:
        self.model.eval()
        self.mode = 'test'
        
        # load params from the latest check point: 
        if self.epoch == self.start_epoch:
            latest_model = self.model_dst
            if self.start_epoch == 0:
                self.model_dst = self.cfg.work_dir + '/epoch_60.pth'
            # else: model_dst = latest.pth
            if os.path.exists(self.model_dst):
                self.load_checkpoint(self.model_dst)
                self.logger.info("load model from checkpoint: " + self.model_dst)
            elif os.path.exists(latest_model):
                self.load_checkpoint(latest_model)
                self.logger.info("load model from checkpoint: " + latest_model)
            else:
                self.logger.info("Model Path: " + self.model_dst + 
                                " NOT FOUND, a new un-trained model will be created only for DEBUG!")

        self.data_loader = data_loader
        
        
        # single_gpu_test(seg_model=self.model,
        #     data_loader=self.data_loader,
        #     show=self.cfg.show,
        #     show_dst=self.cfg.show_dst,
        #     hm_thr=self.cfg.hm_thr,
        #     kpt_thr=self.cfg.kpt_thr,
        #     cpt_thr=self.cfg.cpt_thr,
        #     points_thr=self.cfg.points_thr,
        #     result_dst=self.result_dst,
        #     cluster_thr=self.cfg.cluster_thr,
        #     cluster_by_center_thr=self.cfg.cluster_by_center_thr,
        #     group_fast=self.cfg.group_fast,
        #     crop_bbox=self.cfg.crop_bbox)  
        # self.after_test_epoch()    # summarize result
        
        infer_from_hyper_plane(
            model=self.model,
            data_loader=self.data_loader,
            result_dst=self.result_dst,
            logger=self.logger 
        )
        
        self.after_test_epoch()    # summarize result

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self.start_epoch = self._epoch
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs=60, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, dict)
        assert mmcv.is_list_of(workflow, tuple)
        # assert len(data_loaders) <= len(workflow)

        self._max_epochs = max_epochs    # 60 as default
        for flow in enumerate(workflow):    # [('train',18), ('val',1)]
            mode, epochs = flow
            if mode == 'train':
                # 60 * len(datasets) as default:
                self._max_iters = self._max_epochs * len(data_loaders['train'])    
                break
        
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for flow in workflow:
                mode, epochs = flow
                
                # 1) get epoch runner from runner:
                if isinstance(mode, str):  
                    if not hasattr(self, mode):    # e.g.: self.train()
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    f'callable function, not {type(mode)}')
                
                # 2) run epoch runner:
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:    # @property self.epoch = self._epoch
                        return
                    elif mode == 'fine_tune':
                        epoch_runner(data_loaders, **kwargs)    # fine-tuning phase need train loader & test loader
                    else:   
                        epoch_runner(data_loaders[mode], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hook(self, lr_config):
        if isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater updater.
            # Since this is not applicable for `CosineAnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = mmcv.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook)

    def register_optimizer_hook(self, optimizer_config):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = mmcv.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook)

    def register_checkpoint_hook(self, checkpoint_config):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook)

    def register_momentum_hook(self, momentum_config):
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert 'policy' in momentum_config
            policy_type = momentum_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of momentum updater.
            # Since this is not applicable for `CosineAnealingMomentumUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'MomentumUpdaterHook'
            momentum_config['type'] = hook_type
            hook = mmcv.build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook)

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = mmcv.build_from_cfg(
                info, HOOKS, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - MomentumUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)
