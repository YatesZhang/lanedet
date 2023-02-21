# Copyright (c) Open-MMLab. All rights reserved.
from .hook import HOOKS
from mmcv.runner.hooks import Hook

@HOOKS.register_module()
class ClosureHook(Hook):

    def __init__(self, fn_name, fn):
        assert hasattr(self, fn_name)
        assert callable(fn)
        setattr(self, fn_name, fn)
