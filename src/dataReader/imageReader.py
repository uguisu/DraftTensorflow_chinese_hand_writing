# encoding: UTF-8
# Image Reader

import os
import numpy as np
import struct
import PIL.Image as pilI
import dataReader.toolkits as tk

# Debug flag
DEBUG_FLG = True

dc = dict()


# 字典内部对象
class HanZi:
    def __init__(self,
                 folder_name,
                 index):
        self._folder_name = folder_name
        self._index = index

    @property
    def folder_name(self):
        return self._folder_name

    @property
    def index(self):
        return self._index

    def get_and_increase_index(self):
        rtn = self._index
        self._index += 1
        return rtn


# Get training file list
def get_training_file_list(train_dir='data',
                           validation_size=1):

    training_files = []

    # 遍历给定路径下的所有文件
    for file_name in os.listdir(train_dir):
        # 按照扩展名过滤
        if file_name.endswith('.gnt'):
            training_files += [os.path.join(train_dir, file_name)]

    # Debug
    if DEBUG_FLG:
        print("[DEBUG][get_training_file_list()] Find " + str(len(training_files)) + " files.")

    # 仅返回适当数量的文件列表
    if not 0 <= validation_size <= len(training_files):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(training_files), validation_size))

    training_files.sort()

    return training_files[:validation_size]


# Read data from a single file
def one_file(f):
    header_size = 10
    while True:
        header = np.fromfile(f, dtype=np.uint8, count=header_size)
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tagcode = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        if header_size + width * height != sample_size:
            break
        image = np.fromfile(f, dtype=np.uint8, count=width * height).reshape((height, width))
        yield image, tagcode


# Get training data
def get_training_data(file_list,
                      base_dir='',
                      sub_folder='png/',
                      char_set=None):
    image_list = []
    label_list = []
    __debug_label_list = []

    train_counter = 0

    # 读取数据,或按照指定字符集读取数据
    for __fls in file_list:
        with open(__fls, 'rb') as f:
            for image, tagcode in one_file(f):
                # Decode Label
                __label_gb2312 = struct.pack('>H', tagcode).decode('gb2312')
                if char_set is None:
                    # 统一Label编码
                    label_list += [__label_gb2312]
                    __debug_label_list += [__label_gb2312]

                    # TODO 输出图片
                    im = pilI.fromarray(image)

                    _Hanzi_object = dc.get(__label_gb2312)
                    if __label_gb2312 not in dc:
                        counter = "{:0>4}".format(str(len(dc)))
                        _Hanzi_object = HanZi(counter, 1)
                        dc[__label_gb2312] = _Hanzi_object

                    os.makedirs(base_dir + sub_folder + _Hanzi_object.folder_name, exist_ok=True)
                    im.convert('RGB').save(base_dir
                                           + sub_folder
                                           + _Hanzi_object.folder_name + "/"
                                           + str(_Hanzi_object.get_and_increase_index()) + '.png')

                    # 统一图像大小
                    image_list += [tk.resize_and_normalize_image(image)]
                    # image_list += [image]
                    # image_list.append(image)
                    # image_list = np.append(image_list, image)
                elif char_set is not None and __label_gb2312 in char_set:
                    # 统一Label编码
                    # label_list += [__label_gb2312]
                    label_list += [tk.convert_to_one_hot(__label_gb2312, char_set=char_set)]
                    __debug_label_list += [__label_gb2312]

                    # TODO 输出图片
                    im = pilI.fromarray(image)

                    _Hanzi_object = dc.get(__label_gb2312)
                    if __label_gb2312 not in dc:
                        counter = "{:0>4}".format(str(len(dc)))
                        _Hanzi_object = HanZi(counter, 1)
                        dc[__label_gb2312] = _Hanzi_object

                    os.makedirs(base_dir + sub_folder + _Hanzi_object.folder_name, exist_ok=True)
                    im.convert('RGB').save(base_dir
                                           + sub_folder
                                           + _Hanzi_object.folder_name + "/"
                                           + str(_Hanzi_object.get_and_increase_index()) + '.png')

                    # 统一图像大小
                    image_list += [tk.resize_and_normalize_image(image)]

            train_counter += 1

    # Debug
    if DEBUG_FLG:
        print("[DEBUG][get_training_data()] Dump data.")
        __debugLabel = ""
        for i in range(len(__debug_label_list)):
            # 输出标签
            __debugLabel += __debug_label_list[i]
            if (i + 1) % 10 == 0:
                print("\t%03d : %s" % ((i + 1) // 10, __debugLabel))
                __debugLabel = ""
        # Output remained
        print(__debugLabel)

    return image_list, label_list
