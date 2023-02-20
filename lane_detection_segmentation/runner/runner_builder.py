from runner.train import Runner
from mmcv.utils import ConfigDict
from config.culane import env

cfg = ConfigDict(env)

runner = Runner(cfg)
