# encoding: UTF-8
# For study tensorflow
#     Required:
#     - Python 3.6+
#     - Tensorflow 1.4
# Refer:
#     http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/
#     http://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/
import os
import logging
import datetime
import random
import tensorflow as tf

# ====================================
# ======   Declare Log system   ======
# ====================================
logger = logging.getLogger('[Learn Tensorflow] Chinese hand writing')
logger.setLevel(logging.DEBUG)

logfile = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.log'

fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Declare lof formatter
formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d][%(levelname)s]: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# =====================================
# ======   Declare const value   ======
# =====================================
tf.app.flags.DEFINE_boolean('isDebug',  True, 'debug flag')

tf.app.flags.DEFINE_string('checkpoint_dir', '../data/checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', '../data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir',  '../data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_integer('epoch',               1, 'epoch')
tf.app.flags.DEFINE_integer('channels',            1, 'color channels')
tf.app.flags.DEFINE_integer('image_size',         64, 'image size')
tf.app.flags.DEFINE_integer('batch_size',        128, 'batch size')
tf.app.flags.DEFINE_integer('gpu_model',           0, 'gpu model')
# tf.app.flags.DEFINE_integer('total_characters', 3754, 'total characters')
tf.app.flags.DEFINE_integer('total_characters',    3, 'total characters')

FLAGS = tf.app.flags.FLAGS

# =====================================
# ======   Declare variables     ======
# =====================================
global_random_range = []

X = tf.placeholder(tf.int32, [FLAGS.total_characters, FLAGS.image_size, FLAGS.image_size, FLAGS.channels])


# Get data file list
# Image file and label will be returned as list
#     train_dir: data storage path
#     validation_size: "-1": unlimited, any value grater than zero: limitation size
def get_data_file_list(train_dir='../data',
                       validation_size=-1):
    # image file list
    images_list = []
    # label list
    label_list = []

    # verify limitation
    #     If user setup 'validation_size' property, the following program will choice some data from 'train_dir'
    #     randomly.
    #     Otherwise, the whole dataset, declared by FLAGS.total_characters will be target.
    if validation_size > 0:

        global global_random_range

        # random range
        random_range = []

        if len(global_random_range) == 0:

            for i in range(validation_size):
                # got a random value
                _tmp_random = random.SystemRandom().randrange(0, FLAGS.total_characters)
                while _tmp_random in random_range:
                    # already exists, got a new one
                    _tmp_random = random.SystemRandom().randrange(0, FLAGS.total_characters)

                # attach it to the list
                random_range.append(_tmp_random)

            # update global variable
            global_random_range = random_range
        else:
            random_range = global_random_range

        for _tmp_random in random_range:
            sub_folder_name = "{:0>4}".format(str(_tmp_random))
            for file_list in os.listdir(train_dir + sub_folder_name):
                images_list.append(train_dir + sub_folder_name + '/' + file_list)
                label_list.append(_tmp_random)
    else:
        # use all data
        for _tmp_random in range(FLAGS.total_characters):
            sub_folder_name = "{:0>4}".format(str(_tmp_random))
            for file_list in os.listdir(train_dir + sub_folder_name):
                images_list.append(train_dir + sub_folder_name + '/' + file_list)
                label_list.append(_tmp_random)

    return images_list, label_list


# Read data from file
def read_data_from_file(image_file_list,
                        label_list):
    image_tensor = tf.convert_to_tensor(image_file_list, dtype=tf.string)
    label_tensor = tf.convert_to_tensor(label_list, tf.int64)

    # Refer: https://tensorflow.google.cn/api_docs/python/tf/train/slice_input_producer
    all_matrix_tensor = tf.train.slice_input_producer([image_tensor, label_tensor],
                                                      num_epochs=FLAGS.epoch,
                                                      shuffle=False)

    # read file as binary
    image_tensor = tf.read_file(all_matrix_tensor[0])
    # decode png images
    image_tensor = tf.image.decode_png(image_tensor,
                                       channels=FLAGS.channels)

    # resize to same size
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    # [NOTICE] If dtype=tf.float32, you will got following error:
    #     ValueError: 'size' must be a 1-D int32 Tensor
    new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
    image_tensor = tf.image.resize_images(image_tensor, new_size)

    label_tensor = all_matrix_tensor[1]

    image_batch, label_batch = tf.train.shuffle_batch([image_tensor, label_tensor],
                                                      batch_size=FLAGS.batch_size,
                                                      capacity=50000,
                                                      min_after_dequeue=10000)
    # Debug
    if FLAGS.isDebug:
        logger.debug(image_batch)

    return image_batch, label_batch


# Train
def train():

    # Debug
    if FLAGS.isDebug:
        train_image, train_label = get_data_file_list(FLAGS.train_data_dir, validation_size=2)
        test_image, test_label = get_data_file_list(FLAGS.test_data_dir, validation_size=2)
    else:
        train_image, train_label = get_data_file_list(FLAGS.train_data_dir)
        test_image, test_label = get_data_file_list(FLAGS.test_data_dir)

    # Debug
    if FLAGS.isDebug:
        for i in range(len(train_label)):
            logger.debug(str.format("Train data: %d : %s") % (train_label[i], train_image[i]))
        for i in range(len(test_label)):
            logger.debug(str.format("Test  data: %d : %s") % (test_label[i], test_image[i]))

    train_image_batch, train_label_batch = read_data_from_file(train_image, train_label)
    test_image_batch, test_label_batch = read_data_from_file(test_image, test_label)

    # GPU settings
    # Refer: https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    config = tf.ConfigProto()
    if FLAGS.gpu_model == 1:
        # 1) Allow growth: (more flexible)
        config.gpu_options.allow_growth = True
    elif FLAGS.gpu_model == 2:
        # 2) Allocate fixed memory:
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
    else:
        # 3) Do not use GPU
        config = None

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())


# Output default information
def output_default_info():
    logger.info("============================ Start ============================")
    logger.info("===============================================================")
    logger.info("Basic information")
    logger.info("\tTensorflow version: " + tf.__version__)
    logger.info("\tepoch             : " + str(FLAGS.epoch))
    logger.info("\tchannels          : " + str(FLAGS.channels))
    logger.info("\timage size        : " + str(FLAGS.image_size))
    logger.info("\tbatch size        : " + str(FLAGS.batch_size))
    logger.info("\tgpu model         : " + str(FLAGS.gpu_model))
    logger.info("\ttotal characters  : " + str(FLAGS.total_characters))


if __name__ == "__main__":
    # execute only if run as a script
    # output default information
    output_default_info()
    # start training
    train()
