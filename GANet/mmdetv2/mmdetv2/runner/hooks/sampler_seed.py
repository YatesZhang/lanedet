# Copyright (c) Open-MMLab. All rights reserved.
from .hook import HOOKS
from mmcv.runner.hooks import Hook

@HOOKS.register_module()
class DistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        runner.data_loader.sampler.set_epoch(runner.epoch)
