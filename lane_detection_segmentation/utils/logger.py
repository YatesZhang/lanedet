import logging
import datetime
import os
from objprint import objprint


def get_logger(name, log_path=None, log_level=logging.INFO):
    logging.basicConfig(filename=log_path
                        , level=log_level
                        , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                        , filemode='w')

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # StreamHandler:  output on screen
    # FileHandler  :  output on disk
    logger = logging.getLogger(name)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt=formatter)
    # logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = os.path.join(cfg.work_dir, now)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    return work_dir


# test code:
if __name__ == '__main__':
    logger1 = get_logger(name='logger1', log_path='./logger1.log')
    logger1.debug("This is  DEBUG of logger1 !!")
    logger1.info("This is  INFO of logger1 !!")
    logger1.warning("This is  WARNING of logger1 !!")
    logger1.error("This is  ERROR of logger1 !!")
    logger1.critical("This is  CRITICAL of logger1 !!")

    objprint(logger1)
    logger2 = logging.getLogger("logger1")
    objprint(logger2)
    logger2.info("In fact, I'm logger1!")