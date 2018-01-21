# encoding: UTF-8
import random
import numpy as np
import scipy.misc


# 生成随机样本
def get_random_char_set(char_set,
                        size):
    return ''.join(random.sample(char_set, size))


# one hot
def convert_to_one_hot(char,
                       char_set):

    vector = np.zeros(len(char_set))
    vector[char_set.index(char)] = 1
    return vector


# Resize image
def resize_and_normalize_image(img):
    # 补方
    pad_size = abs(img.shape[0]-img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
    # 缩放
    img = scipy.misc.imresize(img, (64 - 4*2, 64 - 4*2))
    img = np.lib.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=255)
    assert img.shape == (64, 64)

    img = img.flatten()
    # 像素值范围-1到1
    img = (img - 128) / 128
    return img
