import numpy as np
from scipy import interpolate


def get_weight(distance, y_length):
    """
    to assign weight at data['weight'] in train phase
    :param distance: y-offset between current point and start point
    :param y_length: absolute y-distance between start point and end point
    :return:
    """
    if distance < 0:
        return 0.2
    elif distance < 2 * y_length:
        return 1
    else:
        return 0.4


def get_splines(data):
    """
    splines: list, use cubic B-spline to fit the key points in data
    :param data: modify the data inplace, append key 'splines'
    :return: data['splines'] = [ [x (in [0, 1640]), y (in [0, 590])] ]
    """
    data['splines'] = [findBSpline(lane=lane) for lane in data['kpts']]


def gaussian2D(shape=(5, 5), sigma=1.):
    """
    draw 2-D guassian map in a matrix, whose shape is (5, 5) as default
    :param shape: 5x5 matrix as default
    :param sigma: sigma of gaussian distribution
    :return: 5x5 N-D array
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # eps = 2^-16
    return h


def findBSpline(lane: np.array, sample_pts=45, h=590, w=1640) -> list:
    """
    fit the curve in cubic B-spline, with scale: [h, w] = [590, 1640]
    :param lane: type: N-D array, len(key points) is the number of points in a lane
    and kpts.shape = (-1, 2)
    :param sample_pts: uniformly resample 45 points
    :param h: image height in CULane
    :param w: image width in CULane
    :return: list: [y (in [0, 1640]), x (in [0, 590])], x and y are 1-D vectors
    """

    # clip the raw data in xxx.lines.txt: key points
    lane = lane[lane[:, 0] >= 0, :]
    lane = lane[lane[:, 0] < w, :]
    lane = lane[lane[:, 1] >= 0, :]
    lane = lane[lane[:, 1] < h]

    lane = lane[::-1]
    if len(lane) < 2:
        return []

    X = lane[:, 1]  # X in [0, 590]
    Y = lane[:, 0]  # Y in [0, 1640]

    x = np.linspace(start=max(X[0], 0), stop=min(X[-1], h), num=sample_pts)
    try:
        if len(X) > 3:
            # to find B-spline representation of an N-D curve:
            curve_func = interpolate.splrep(X, Y, k=3)
            # to evaluate the B-spline
            y = interpolate.splev(x, curve_func)
        else:
            # use linear mode:
            curve_func = interpolate.splrep(X, Y, k=1)
            y = interpolate.splev(x, curve_func)
    except ValueError:
        print("Value Error! input array should be ascending order!, but we get X:")
        print(X)
        print("and Y:")
        print(Y)
        print()
    # y in [0, 1640], x in [0, 590]
    return [y, x]


def draw_gaussian_mask(hm: np.array, point, gaussian2d, radius=2, down_scale=8):
    """
    we do down sampling in this function.
    Input point should in [1640, 590]
    :param down_scale: 8 as default: point = (x, y) / 8
    :param radius: 2 as default
    :param gaussian2d: cached 2-D gaussian heat map
    :param hm: heat map, type: np.array, shape: [1, 590//8, 1640//8]
    :param point: Iterable, len==2, x in [0, 1640], y in [0, 590]
    :return: void, modify the heat map, 'hm', inplace
    """
    # x in [0, 1640/8], y in [0, 590/8]

    x, y = int(point[0] / down_scale), int(point[1] / down_scale)
    h, w = hm.shape[1], hm.shape[2]  # (1, 590//8, 1640//8)
    t = min(h - 1, y + radius)  # top of the target region in heat map
    r = min(w - 1, x + radius)  # right of the target region in heat map
    b = max(0, y - radius)  # bottom of the target region in heat map
    l = max(0, x - radius)  # left of the target region in heat map

    dt = y + radius - t  # delete the row on top in gaussian-2d
    dr = x + radius - r  # delete the row on right in gaussian-2d
    db = b - y + radius  # delete the row on bottom in gaussian-2d
    dl = l - x + radius  # delete the row on left in gaussian-2d

    diameter = radius * 2 + 1
    # import pdb
    # pdb.set_trace()
    hm[0, b:t + 1, l:r + 1] = np.maximum(hm[0, b:t + 1, l:r + 1], gaussian2d[db:diameter - dt, dl:diameter - dr])
