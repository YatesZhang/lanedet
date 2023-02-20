from mmcv.utils import ConfigDict
from dataset.dataloader import get_testloader, get_trainloader


mem_save = dict(
    workers=4,
    pin_mem=False
)
mem_save = ConfigDict(mem_save)


speed = dict(
    workers=12,
    pin_mem=False
)
speed = ConfigDict(speed)


train_loader = get_trainloader(speed)
test_loader = get_testloader(speed)



