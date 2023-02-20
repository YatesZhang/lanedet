import torch
from dataset.culane_builder import train_set, test_set
from config.culane import batch_size


def get_trainloader(cfg):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=cfg.pin_mem, drop_last=False)
    return train_loader


def get_testloader(cfg):
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=cfg.pin_mem, drop_last=False)
    return test_loader



