train_set = dict(
    type='CULane',
    data_root=r'D:\dataset\CULane',
    mode='train',
    train_data=['img', 'seg', 'exist', 'kpts', 'splines',
                'spt_hm', 'hm', 'error', 'mask', 'weight', 'ofstm'],
    diameter=5,
)