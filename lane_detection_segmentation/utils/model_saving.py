import torch
import os
import torch.nn.functional
from termcolor import cprint


def save_model(net, optim, scheduler, work_dir, epoch):
    model_dir = os.path.join(work_dir, 'ckpt')
    # mkdir [-p] dirname : to ensure the existence of the path
    os.system('mkdir -p {}'.format(model_dir))
    saving_path = os.path.join(model_dir, '{}.pth'.format(epoch))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, saving_path)
    cprint('model saved at: ' + saving_path, on_color='on_red')


def load_network_specified(net, model_dir, logger=None):
    pretrained_net = torch.load(model_dir)['net']
    net_state = net.state_dict()
    state = {}
    for k, v in pretrained_net.items():
        if k not in net_state.keys() or v.size() != net_state[k].size():
            if logger:
                logger.info('skip weights: ' + k)
            continue
        state[k] = v
    net.load_state_dict(state, strict=False)


def load_network(net, model_dir, finetune_from=None, logger=None):
    if finetune_from:
        if logger:
            logger.info('Finetune model from: ' + finetune_from)
        load_network_specified(net, finetune_from, logger)
        return
    if logger:
        logger.info('load pretrained model from: '+ model_dir)
    pretrained_model = torch.load(model_dir)
    net.load_state_dict(pretrained_model['net'], strict=True)
