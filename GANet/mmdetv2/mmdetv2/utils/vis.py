import numpy
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from objprint import objprint
import warnings
import torch 

def print0(data: numpy.ndarray):
    """
    print empty numpy array
    :param data:
    :return:
    """
    if len(data.shape) == 0:
        print("empty array!")
    return


def print1(data: numpy.ndarray):
    """
    print 1-D numpy array
    :param data:
    :return:
    """
    if len(data.shape) != 1:
        return
    print("array: ", len(data))
    print(data)
    return


def print2(data: numpy.ndarray):
    """
    print 2-D numpy array
    :param data:
    :return:
    """
    if len(data.shape) != 2:
        return
    h, w = data.shape
    if h * w >= 100 * 100 or h < 30 or w < 30:
        plt.imshow(data)
        plt.show()
    else:
        warnings.warn("2-D array's size is less than 100 * 100, we'll print the matrix:")
        print("matrix: ", data.shape)
        print(data)
    return


class DataVis(object):
    """
    data visualization:
    data type: numpy.ndarray, PIL.Image.Image are supported
    self.inline: on jupyter notebook
    """
    def __init__(self, inline=True, mode='PILLOW'):
        """
        :param inline: on jupyter notebook
        :param mode: which function will be used to read image
        """
        assert mode in ['PILLOW', 'OPENCV'], "only PILLOW and OPENCV are supported!"
        self.PILLOW = False
        self.OPENCV = False
        setattr(self, mode, True)

        self.ndarray_showing_func = [print0, print1, print2, self.show3, self.show4]
        self.inline = inline

    def __call__(self, data):
        self.data_vis(data)

    def show_numpy(self, data: numpy.ndarray):
        if data is None:
            n = 0
        else:
            n = len(data.shape)

        if n >= 5:
            raise ValueError("ndarray's dimension should no more than 5! ")
        self.ndarray_showing_func[n](data)

    def show_dict(self, data: dict):
        keys = "keys: "
        n = len(keys)
        for i, k in enumerate(data.keys()):
            if i == 0:
                keys += k
            else:
                keys += (" ," + k)
            if n > 50:
                n -= 50
                keys += '\n'
        print(keys)

        for k in data.keys():
            print("------------------------")
            print(k)
            self.data_vis(data[k])

    def data_vis(self, data):
        """
        show data:
        :param data:
        :return:
        """
        
        
        if isinstance(data, Image.Image):
            data.show()
            
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data, numpy.ndarray):
            self.show_numpy(data)
        elif isinstance(data, str):
            print(data)
        elif isinstance(data, list):
            print(data)
        elif isinstance(data, tuple):
            print(data)
        elif isinstance(data, dict):
            self.show_dict(data)
        else:
            print("type: " + str(type(data)))
            objprint(data)
        return

    def show4(self, data: numpy.ndarray):
        """
        inline: run on jupyter, if we have statement: %matplotlib inline
        show a batch of images in a window
        :param data: whose dimension is (batch, channel, h, w)
        :return:
        """
        if len(data.shape) > 4:
            raise NotImplementedError("data dimension is: " + str(len(data.shape))
                                      + " ,more than 4, which is not supported!")
        else:
            batch_sz, c, h, w = data.shape
            # max h, w: (1080, 1920)
            if c != 1 or c != 3:
                raise ValueError("images' channel should be 1 or 3!")
            if batch_sz * h * w > 1080 * 1920 or h > 1080 or w > 1920:
                raise ValueError("images are too large!")

            H = h  # current height of figure's window with muti-images
            W = w  # current width of figure's window  with muti-images
            row = 1  # rows of the figure's window
            col = 1  # columns of the figure's window

            # fit the window size by greedy policy:
            for index in range(batch_sz):
                if H < W:
                    H += H
                    row += 1
                else:
                    W += W
                    col += 1
                N = row * col  # current capacity of figures' window
                if N >= batch_sz:
                    break
                if H > 1080 or W > 1920:
                    raise ValueError("images are too large!")

            # show images:
            for img_idx in range(batch_sz):
                plt.subplot(row, col, img_idx), plt.title(str(img_idx))
                self.show3(data[img_idx])
            plt.show()
        return

    def show3(self, data: numpy.ndarray):
        """
        show image who has 3 dimensions
        inline: run on jupyter, if we have statement: %matplotlib inline
        :param data: numpy image
        :return:
        """
        if len(data.shape) != 3:
            return

        channel = data.shape[0]
        if channel > 3 and data.shape[2] == 3:
            if self.inline:
                if self.OPENCV:
                    # B, G, R -> R, G, B
                    plt.imshow(data[:, :, [2, 1, 0]])
                elif self.PILLOW:
                    # show R, G, B
                    plt.imshow(data)
                else:
                    raise NotImplementedError("only OPENCV and PILLOW are supported!")
                plt.show()
            else:
                img_show = Image.fromarray(data)
                img_show.show()
            return

        # to show mask, confidence map or heat map:
        if channel == 1:
            plt.imshow(data[0])
            plt.show()
            return

        # data : (2, h, w), show 2 channels:
        elif channel == 2:
            # h < w:
            if data.shape[1] < data.shape[2]:
                plt.subplot(2, 1, 1)
                plt.imshow(data[0])
                plt.subplot(2, 1, 2)
                plt.imshow(data[1])
                plt.show()
            else:
                plt.subplot(1, 2, 1)
                plt.imshow(data[0])
                plt.subplot(1, 2, 2)
                plt.imshow(data[1])
                plt.show()
            return
        elif channel == 3:
            plt.imshow(data)
            plt.show()
            return
        else:
            msg = "only 3-channel or arrays with lower dimensions are supported! But we got channel: " + str(channel)
            raise NotImplementedError(msg)
    
    def __repr__(self):
        msg = "show opencv img: " + str(self.OPENCV) + "\n" \
            + "show Pillow img: " + str(self.PILLOW) + "\n" \
                "show img on jupyter notebook: " + str(self.inline) + "\n"
        return msg
