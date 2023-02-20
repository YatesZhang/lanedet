from evaluator.culane_eval import CULaneEval
from mmcv.utils import ConfigDict
from config.culane import env


cfg = ConfigDict(env)
culane_eval = CULaneEval(cfg)
