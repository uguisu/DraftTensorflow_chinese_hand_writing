# encoding: UTF-8
import os
import argparse
import yaml
import tensorflow as tf

# ====================================
# ======   Define Log system    ======
# ====================================
# TODO logger

# ====================================
# ======   Define usage         ======
# ====================================
# Refer: https://docs.python.org/3.6/library/argparse.html
parser = argparse.ArgumentParser(prog='PROG',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='[Learn Tensorflow] Chinese hand writing.\n'
                                             'Train and test a CNN network to identify hand writing.')
parser.add_argument('-s', dest='yaml_file', default='cnn.yml',
                    help='path of YAML setting file stored. The default file will be cnn.yml')
parser.add_argument('-d', dest='is_debug', action='store_true',
                    help='debug flag')
args = parser.parse_args()

# ====================================
# ======   Define variables     ======
# ====================================
try:
    _f = open(args.yaml_file, 'r', encoding='UTF-8')
    yml_settings = yaml.load(_f.read())
except FileNotFoundError:
    # TODO logger
    # file not found
    exit(-1)

# ====================================
# ======   Define const value   ======
# ====================================
tf.app.flags.DEFINE_string('checkpoint_dir', yml_settings['checkpoint_dir'], 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', yml_settings['train_data_dir'], 'the train data dir')
tf.app.flags.DEFINE_string('test_data_dir',  yml_settings['test_data_dir'],  'the test data dir')
tf.app.flags.DEFINE_string('model_dir',      yml_settings['model_dir'],      'the model dir')

tf.app.flags.DEFINE_integer('total_characters', yml_settings['total_characters'], 'total characters')
tf.app.flags.DEFINE_integer('epoch',            yml_settings['epoch'],            'epoch')
tf.app.flags.DEFINE_integer('batch_size',       yml_settings['batch_size'],       'batch size')
tf.app.flags.DEFINE_integer('dataset_buffer',   yml_settings['dataset_buffer'],   'dataset buffer size')
tf.app.flags.DEFINE_integer('channels',         yml_settings['channels'],         'channels')
tf.app.flags.DEFINE_integer('image_size',       yml_settings['image_size'],       'image size')

tf.app.flags.DEFINE_float('learning_rate', yml_settings['learning_rate'], 'learning rate')
tf.app.flags.DEFINE_float('drop_keep',     yml_settings['drop_keep'],     'drop keep')

FLAGS = tf.app.flags.FLAGS


def get_train_inputs(batch_size, training_data):
    """
    Define the training input
    :param batch_size: batch size
    :param training_data: training data
    :return: callable function
    """

    def train_inputs():
        images, labels = get_data_file_list(training_data)

        # Create a Dataset object whose elements are slices of the given tensors.
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Infinite iterations
        # 1) This transformation applies parse_file_to_image_tensor() to each element of this dataset,
        # and returns a new dataset containing the transformed elements, in the same order as they
        # appeared in the input.
        dataset = dataset.map(parse_file_to_image_tensor, num_parallel_calls=4)
        # 2) Repeates this dataset count times
        dataset = dataset.repeat(None)
        # 3) Randomly shuffles the elements of this dataset.
        dataset = dataset.shuffle(buffer_size=10000)
        # 4) Combines consecutive elements of this dataset into batches.
        dataset = dataset.batch(batch_size)

        return dataset

    return train_inputs


def get_test_inputs(batch_size, test_data):
    """
    Define the test input
    :param batch_size: batch size
    :param test_data: test data
    :return: callable function
    """

    def test_inputs():
        images, labels = get_data_file_list(test_data)

        # Create a Dataset object whose elements are slices of the given tensors.
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # Infinite iterations
        # 1) This transformation applies parse_file_to_image_tensor() to each element of this dataset,
        # and returns a new dataset containing the transformed elements, in the same order as they
        # appeared in the input.
        dataset = dataset.map(parse_file_to_image_tensor, num_parallel_calls=4)
        # 2) Repeates this dataset count times
        dataset = dataset.repeat(None)
        # 3) Randomly shuffles the elements of this dataset.
        dataset = dataset.shuffle(buffer_size=10000)
        # 4) Combines consecutive elements of this dataset into batches.
        dataset = dataset.batch(batch_size)

        return dataset

    return test_inputs


def get_data_file_list(data_dir):
    """
    Get data file list
    :param data_dir: data full path
    :return: images_list, labels_list
    """
    # image file list
    images_list = []
    # label list
    labels_list = []

    for _tmp_random in range(FLAGS.total_characters):
        sub_folder_name = "{:0>5}".format(str(_tmp_random))
        for file_list in os.listdir(data_dir + sub_folder_name):
            images_list.append(data_dir + sub_folder_name + "/" + file_list)
            labels_list.append(_tmp_random)

    return images_list, labels_list


def parse_file_to_image_tensor(filename, label):
    """
    Open image file and decode it to a JPEG-encoded image to a uint8 tensor.
    :param filename: image file name
    :param label: label
    :return: image, label
    """
    label_t = tf.convert_to_tensor(label, dtype=tf.int32)

    # tf.read_file() needs a scalar input (i.e., just one string), but the results of tf.decode_csv are coming
    # back in a "rank 1" context, i.e, a 1-D list. You need to dereference the results
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_t = tf.image.decode_png(image_string, channels=FLAGS.channels)

    # This will convert to float values in [0, 1]
    image_t = tf.image.convert_image_dtype(image_t, tf.float32)

    # resize
    # [NOTICE] If dtype=tf.float32, you will got following error:
    #     ValueError: 'size' must be a 1-D int32 Tensor
    # TODO following command an be skipped?
    new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
    image_t = tf.image.resize_images(image_t, new_size)

    return image_t, label_t


if __name__ == '__main__':
    print(tf.__version__)
