# encoding: UTF-8

import logging
import datetime
import os

# Declare Log system
logger = logging.getLogger('[Learn Tensorflow] Chinese hand writing')
logger.setLevel(logging.DEBUG)

logfile = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + '.log'

fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Declare log formatter
formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d][%(levelname)s]: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)


# Declare const value
base_dir = '../'
training_data_dir = base_dir + 'data/HWDB1.1trn_gnt'
test_data_dir = base_dir + 'data/HWDB1.1tst_gnt'


def get_training_file_list(train_dir='data',
                           target_data_amount=1):
    """
    Get training file list
    :param train_dir: training data store path
    :param target_data_amount: how many group of data would be read from original data file
    :return: file list
    """
    training_files = []

    # 遍历给定路径下的所有文件
    for file_name in os.listdir(train_dir):
        # 按照扩展名过滤
        if file_name.endswith('.gnt'):
            training_files += [os.path.join(train_dir, file_name)]

    logger.debug("Find " + str(len(training_files)) + " files.")

    # 仅返回适当数量的文件列表
    if not 0 <= target_data_amount <= len(training_files):
        msg = 'Do not have enough data. Target value should between 0 and {}. Received: {}.'.format(
            len(training_files), target_data_amount)
        logger.error(msg)
        raise ValueError(msg)

    training_files.sort()

    return training_files[:target_data_amount]


if __name__ == '__main__':
    logger.info("== Start ==")
    training_data_files = get_training_file_list(training_data_dir, target_data_amount=1)
    test_data_files = get_training_file_list(test_data_dir, target_data_amount=1)

