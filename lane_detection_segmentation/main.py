import os
import torch
from runner.runner_builder import runner
from mmcv import ConfigDict
from config.culane import env
import torch.backends.cudnn as cudnn
import argparse

cfg = ConfigDict(env)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device
    # torch.multiprocessing.set_start_method('spawn')
    cudnn.benchmark = True
    cudnn.fastest = True

    runner.run()


if __name__ == '__main__':
    main()
    # print(os.path.join(cfg.work_dir, cfg.exp_name + '.log'))


