import math
from torch.optim.lr_scheduler import LambdaLR
from config.culane import total_iter


def get_scheduler(optimizer):
    fn = lambda _iter: math.pow(1 - _iter / total_iter, 0.9)
    return LambdaLR(optimizer=optimizer, lr_lambda=fn)
