# encoding: UTF-8

import datetime
import logging
import os
import struct

import cv2
import numpy as np

from helpTool import HanZi, ImFilterPipeline

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
# TODO change to your custom path
base_dir = '/myWork/999_tmp/ai/data/handwrite'
training_data_dir = base_dir + '/HWDB1.1trn_gnt'
test_data_dir = base_dir + '/HWDB1.1tst_gnt'
load_data_amount = 3
# 汉字字典 (key : val) = (汉字 : Hanzi Class)
dc = dict()

# Declare image improve
cim = ImFilterPipeline()
pip_settings = cim.pipeline
pip_settings["rotated"] = 1
# pip_settings["gaussianBlur"] = 1
pip_settings["resize"] = 1

# Declare image improve
cim_resize = ImFilterPipeline()
pip_resize_settings = cim_resize.pipeline
pip_resize_settings["resize"] = 1


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


def read_binary_image_from_file_as_data_source(file_handle):
    """
    read binary image from file as data source
    :param file_handle: file
    :return: image array, tagcode, word
    """
    header_size = 10
    cOunt = 0
    while True:
        header = np.fromfile(file_handle, dtype=np.uint8, count=header_size)

        # read to the end
        if not header.size:
            break

        cOunt += 1

        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tagcode = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        word = struct.pack('>H', tagcode).decode('gb2312')

        if header_size + width * height != sample_size:
            break

        image = np.fromfile(file_handle, dtype=np.uint8, count=width * height).reshape((height, width))

        yield image, tagcode, word


def image_resize(img):
    """
    Resize image
    :param img: image array
    :return: resized image array
    """
    processed_image = cim_resize.filter(img)

    return processed_image


def image_improve(img):
    """
    Improve image
    :param img: image array
    :return: resized image array
    """
    processed_image = cim.filter(img)

    return processed_image


def get_data(file_list, folder='png/', is_improve_image=False):
    """
    Get data
    :param file_list: 数据文件列表
    :param folder: 文件保存路径
    :param is_improve_image: if improve image
    :return: image_list, label_list
    """
    image_list = []
    label_list = []
    load_process = 0

    for __fls in file_list:

        load_process += 1
        logger.debug("Processing: {:.2f}%".format(load_process / len(file_list) * 100))

        with open(__fls, 'rb') as f:
            for image, tagcode, word in read_binary_image_from_file_as_data_source(f):

                label_list += [word]

                # 检查当前读取的汉字是否已经存在于汉字字典中
                if word not in dc:
                    counter = "{:0>5}".format(str(len(dc)))
                    # declare new Hanzi instance
                    _Hanzi_object = HanZi(counter, 1)
                    dc[word] = _Hanzi_object
                else:
                    _Hanzi_object = dc.get(word)

                if not is_improve_image:
                    # without image improve
                    image_list = [image_resize(image)]
                else:
                    # improve image
                    image_list = [image_resize(image), image_improve(image)]

                # create folder to write image
                os.makedirs(folder + _Hanzi_object.folder_name, exist_ok=True)

                for im in image_list:
                    cv2.imwrite(folder + _Hanzi_object.folder_name + "/"
                                       + str(_Hanzi_object.get_and_increase_index()) + '.png', im)

                # debug
                # if len(label_list) % 10 == 0 and len(label_list) >= 10:
                #     logger.debug("Current load: {}".format(label_list[len(label_list) - 10:]))

    # debug
    # if len(label_list) % 10 != 0 and len(label_list) >= 10:
    #     logger.debug("Current load: {}".format(label_list[int(len(label_list) / 10) * 10:]))

    return image_list, label_list


if __name__ == '__main__':
    logger.info("== Start ==")
    training_data_files = get_training_file_list(training_data_dir, target_data_amount=load_data_amount)
    test_data_files = get_training_file_list(test_data_dir, target_data_amount=load_data_amount)

    training_img_list, training_label_list = get_data(training_data_files, base_dir + '/data/train/', is_improve_image=True)
    logger.info("Load {} images for training.".format(len(training_label_list)))

    test_img_list, test_label_list = get_data(test_data_files, base_dir + '/data/test/')
    logger.info("Load {} images for test.".format(len(test_label_list)))

    logger.info("== Job finished ==")
