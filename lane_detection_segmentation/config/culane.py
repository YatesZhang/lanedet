epochs = 12
workflow = [
  ('train', 3), ('val', 1), ('train', 3), ('val', 1)
, ('train', 6), ('val', 1), ('train', 6), ('val', 1)
, ('train', 6), ('val', 1), ('train', 6), ('val', 1)
, ('train', 6), ('val', 1), ('train', 6), ('val', 1)
, ('train', 6), ('val', 1), ('train', 6), ('val', 1)
]

batch_size = 16
total_iter = (88880 // batch_size) * epochs
totol_steps = 88880 // batch_size
culane_evaluator_path = '/disk/zhangyunzhi/py/Ultra-Fast-Lane-Detection/evaluation/culane/culane_evaluator'

# the working environment:
env = dict(
    work_dir='work_dir/culane',
    device="0",                     # gpu 1 as defualt
    exp_name='cooatt',
    log_name='cooatt',
    data_root='/disk/gaoyao/dataset/culane',
    train_gt_txt='/disk/gaoyao/dataset/culane/list/train_gt.txt',
    test_txt='/disk/gaoyao/dataset/culane/list/test.txt',
    # resume_from='/disk/zhangyunzhi/py/videoLaneDetection/work_dir/culane/20220526_150859/ckpt/18.pth',
    resume_from=None,
    view=False,
    cut_height=240,
    parallel=False,
    with_heat_map=False              # defualt: False
)