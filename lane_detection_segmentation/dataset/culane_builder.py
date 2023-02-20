from mmcv.utils import ConfigDict
from dataset.culane import TrainPhase, TestPhase, CULane
from config.culane import env


cfg = ConfigDict(env)


train_set = CULane(TrainPhase(cfg))
test_set = CULane(TestPhase(cfg))

