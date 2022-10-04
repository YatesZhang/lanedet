from torch.utils.data.dataset import Dataset, T_co
from .builder import DATASET
import numpy as np
from PIL import Image
from .lane_formatting import gaussian2D, draw_gaussian_mask, get_splines, get_weight

down_sample = dict(
    img=(800, 320),  # Pillow style
    seg=(800, 320),  # Pillow style
    kpts=8,  # scale the numpy array
    hm=(590 // 8, 1640 // 8),  # pytorch style
    fpn=[4, 8, 16, 32]
)

train_data = ['img', 'seg', 'kpts', 'hm']


# to get img data on cpu
@DATASET.register_module(name='CULane')
class CULaneDataset(Dataset):
    # __slots__ = ['data_root', ]

    def __init__(self, data_root: str, mode: str, train_data:dict, diameter=5
                 , down_sample=down_sample):
        super(CULaneDataset, self).__init__()

        # get the mode:
        assert mode in ['train', 'test', 'val'], "given mode is not in [train, test, val]! "
        self.mode = mode

        assert len(train_data) >= 2 and 'img' in train_data, "img is necessary during training phase!"
        self.train_data = train_data
        self.io = ['img', 'seg', 'exist', 'kpts']

        # get the data root path:
        self.data_root = data_root

        # get the data list:
        self.data_list_path = ''
        if self.mode == 'train':
            self.data_list_path = self.data_root + '/list/train_gt.txt'
        elif self.mode == 'test':
            self.data_list_path = self.data_root + '/list/test.txt'
        elif self.mode == 'val':
            self.data_list_path = self.data_root + '/list/val.txt'
        else:
            raise NotImplementedError(self.mode + 'is not implemented !')
        self.data_list = self._get_data_list(self.data_list_path)

        # get down sample rate:
        self.down_sample = down_sample
        self.img_size = down_sample['img']
        self.seg_size = down_sample['seg']

        # gaussian 2D heat map:
        self.diameter = diameter
        self.gaussian2d = gaussian2D((diameter, diameter), diameter / 6)

        #
        self.train_dict = {
            'img': None,  # IMaGe
            'seg': None,  # SEGmentation
            'exist': None,  # EXIST mask for lane instance
            'kpts': None,  # Key PoinTS
            'splines': None,  # B-SPLINES
            'spt_hm': None,  # Start PoinT Heat Map
            'hm': None,  # Heat Map
            'error': None,  # regression ERROR between float and integer in rasterization of coordination
            'mask': None,  # binary MASK of the key points
            'weight': None,  # to WEIGHT the offset
            'ofstm': None  # OFfSeT Map from the start point
        }

        # data name to train:
        self.img = None
        self.seg = None
        self.exist = None
        self.kpts = None
        self.splines = None
        self.spt_hm = None
        self.hm = None
        self.error = None
        self.mask = None
        self.weight = None
        self.ofstm = None

        for Integer, var_name in enumerate(self.train_data):
            assert not self.train_dict[var_name]
            self.train_dict[var_name] = True
            setattr(self, var_name, Integer)

        sz = (590//8, 1640 // 8)
        data = dict()
        if self.img is not None:
            data['img'] = np.array([])

        # to create buffer only in training phase:
        if self.mode == 'train':
            if self.seg is not None:
                data['seg'] = np.array([])
            if self.exist is not None:
                data['exist'] = np.array([])
            if self.kpts is not None:
                data['kpts'] = []
            if self.splines is not None:
                data['splines'] = []
            if self.spt_hm is not None:
                data['spt_hm'] = np.zeros((1, *sz))
            if self.hm is not None:
                data['hm'] = np.zeros((1, *sz))
            if self.error is not None:
                data['error'] = np.zeros((2, *sz))
            if self.mask is not None:
                data['mask'] = np.zeros((1, *sz))
            if self.weight is not None:
                data['weight'] = np.zeros((2, *sz))
            if self.ofstm is not None:
                data['ofstm'] = np.zeros((2, *sz))
        self.data = data

    def __getitem__(self, index) -> T_co:
        if self.mode == 'train':
            return self.train_phase(index)
        else:  # self.mode is test or val:
            return self.test_phase(index)

    def __len__(self):
        return len(self.data_list['img'])

    def _get_data_list(self, data_list_path) -> dict:
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
        data_list = dict()

        if self.mode == 'train':
            data_list['img'] = [line.split()[0] for line in lines]  # image list
            if 'seg' in self.io:  # label list: segmentation, lanes' masks
                data_list['seg'] = [line.split()[1] for line in lines]
            if 'exist' in self.io:  # label list: lane instance's exist masks
                data_list['exist'] = [np.array(line.split()[2:]).astype('int') for line in lines]
            if 'kpts' in self.io:  # label list: lanes' key points, annotated in txt files
                data_list['kpts'] = [line[:-3] + 'lines.txt' for line in data_list['img']]
            return data_list
        elif self.mode == 'test' or self.mode == 'val':
            data_list['img'] = [line[:-1] for line in lines]
            return data_list
        else:
            raise NotImplementedError()

    def test_phase(self, idx) -> dict:
        data = self.data.copy()
        self.get_img(data, idx)
        return data

    def train_phase(self, idx) -> dict:
        data = self.data.copy()

        # get image for training:
        self.get_img(data, idx)

        # get segmentation mask:
        if self.seg is not None:
            self.get_seg(data, idx)

        # get lane-instance-wise exist mask:
        if self.exist is not None:
            self.get_exist(data, idx)

        # get key points:
        if self.kpts is not None:
            self.get_kpts(data, idx)

        # using B-splines to fit the key points :
        if self.splines is not None:
            get_splines(data)

        scale = 8    # 8
        sz = (590//8, 1640//8)

        # get the pointers in data:
        splines = data['splines']

        # get the heat map
        # loss function: focal loss
        spt_hm = data['spt_hm']
        hm = data['hm']

        # get the regression error map
        # loss function: L1 loss for regression with mask
        error = data['error']
        mask = data['mask']

        # get the offset map (distance from start points)
        # loss function: weight the offset map and calculate L1 loss for regression
        offset_hm = data['ofstm']
        offset_mask_weight = data['weight']

        # lane: [x^T, y^T], x(in [0, 1640]), y(in [0, 590]) are 1-D vectors
        for lane in splines:
            x = lane[0]    # in [0, 1640]
            y = lane[1]    # in [0, 590]
            start_point = (x[-1], y[-1])
            end_point = (x[0], y[0])

            if self.spt_hm is not None:
                draw_gaussian_mask(hm=spt_hm, point=start_point, gaussian2d=self.gaussian2d)

            for point in zip(x, y):
                # get regression heat map for int error:
                pt_int = (int(point[0]), int(point[1]))

                if self.hm is not None:
                    draw_gaussian_mask(hm=hm, point=pt_int, gaussian2d=self.gaussian2d)

                # down scale the points by default 8:
                start_point = start_point[0] // scale, start_point[1] // scale
                end_point = end_point[0] // scale, end_point[1] // scale
                point = point[0] / scale, point[1] / scale
                pt_int = pt_int[0] // scale, pt_int[1] // scale

                if self.error is not None:
                    dx = point[0] - pt_int[0]
                    dy = point[1] - pt_int[1]
                    try:
                        error[0, pt_int[1], pt_int[0]] = dx
                        error[1, pt_int[1], pt_int[0]] = dy
                    except IndexError:
                        import pdb
                        pdb.set_trace()
                    assert dx < 2 and dy < 2, "error should in [0, 1]"

                if self.mask is not None:
                    mask[0, pt_int[1], pt_int[0]] = 1

                # get offset heat map from start point:
                delta_y = start_point[1] - point[1]
                if self.ofstm is not None:
                    delta_x = start_point[0] - point[0]
                    offset_hm[0, pt_int[1], pt_int[0]] = delta_x
                    offset_hm[1, pt_int[1], pt_int[0]] = delta_y

                # get weight from y-offset:
                if self.weight is not None:
                    weight = get_weight(distance=delta_y, y_length=abs(start_point[1] - end_point[1]))
                    offset_mask_weight[0, pt_int[1], pt_int[0]] = weight
                    offset_mask_weight[1, pt_int[1], pt_int[0]] = weight
        return data

    # to get resized image:
    def get_img(self, data, idx):
        img_path = self.data_root + self.data_list['img'][idx]
        img = Image.open(img_path).resize(self.img_size, resample=Image.BILINEAR)
        data['img'] = np.array(img)

    # to get resized segmentation
    def get_seg(self, data, idx):
        seg_path = self.data_root + self.data_list['seg'][idx]
        seg = Image.open(seg_path).resize(self.seg_size, resample=Image.NEAREST)
        data['seg'] = np.array(seg)

    # to get lane-instance-wise existence mask
    def get_exist(self, data, idx):
        data['exist'] = self.data_list['exist'][idx]

    # in GANet: if len(lines[i]) <= 3: skip the lane, but we use all
    def get_kpts(self, data, idx):
        """
        kpts: list, len(list)==num_lanes
        elements in list: N-D array, column vector
        data['kpts']:
        [lane(x^T, y^T) for lane in lanes], x in [0, 1640], y in [0, 590]
        :param data: data_dict to modify
        :param idx:
        :return:
        """
        annotation_path = self.data_root + self.data_list['kpts'][idx]
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        # convert to float:
        lines = [line.strip().split() for line in lines]
        lines = [np.array([float(s) for s in line]).reshape((-1, 2)) for line in lines]
        data['kpts'] = lines
