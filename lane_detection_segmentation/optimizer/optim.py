import torch
from mmcv.utils import ConfigDict


config = dict(
    type='SGD',
    lr=0.025,
    weight_decay=1e-4,
    momentum=0.9
)
config = ConfigDict(config)


def get_optimizer(net, cfg=config):
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    if cfg.type == 'Adam':
        optimizer = torch.optim.Adam(training_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.type == 'SGD':
        optimizer = torch.optim.SGD(training_params, lr=cfg.lr, momentum=cfg.momentum
                                    , weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer
