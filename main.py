from datasets.builder import DATASET
from datasets.CUlaneDataset import CULaneDataset    # important!
from objprint import objprint


from mmcv import load
def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    # dataset = DATASET.build(dict(type='CULaneDataset', cfg='1'))
    # objprint(dataset)
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    lists = load(r'D:\dataset\CULane\list\train_gt.txt')
    print(lists)
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
