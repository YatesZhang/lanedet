import time
import logging
from utils.logger import get_logger


class Timer(object):
    def __init__(self, log_name):
        # self.func = func
        self.logger = logging.getLogger(log_name)

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start = time.time()
            ret = func(*args, **kwargs)
            time_consumed = int(time.time() - start)
            h = time_consumed // 3600
            time_consumed %= 3600
            m = time_consumed // 60
            s = time_consumed % 60
            self.logger.info("time consumed: %dh%dm%ds" % (h, m, s))
            return ret
        return wrapper


if __name__ == '__main__':
    logger1 = get_logger(name='log_nameless', log_path='./timer.log')

    @Timer(log_name='log_nameless')
    def add(a, b):
        i = 100000000
        while i > 0:
            i -= 1
        return a + b

    # equals to:
    add = Timer(log_name='log_nameless')(add)
    print(add(2, 3))


