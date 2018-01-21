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

tf.app.flags.DEFINE_string('checkpoint_dir', '../../data/checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', '../../data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir',  '../../data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_integer('epoch', 1, 'epoch')
tf.app.flags.DEFINE_integer('channels', 1, 'color channels')
tf.app.flags.DEFINE_integer('image_size', 64, 'image size')


FLAGS = tf.app.flags.FLAGS

# =====================================
# ======   Declare variables     ======
# =====================================
a = 1  # TODO


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

    # loop all files under target path
    for root, sub_folder, file_list in os.walk(train_dir):
        images_list += [os.path.join(root, file_path) for file_path in file_list]
    label_list += [int(file_name[len(train_dir):].split(os.sep)[0]) for file_name in images_list]

    # verify limitation
    if validation_size > 0:
        images_list = images_list[:validation_size]
        label_list = label_list[:validation_size]

    return images_list, label_list


# Read data from file
def read_data_from_file(image_file_list,
                        label_list):
    image_tensor = tf.convert_to_tensor(image_file_list, dtype=tf.string)
    label_tensor = tf.convert_to_tensor(label_list, tf.int64)

    # read file as binary
    image_tensor = tf.read_file(image_tensor)
    # decode png images
    image_tensor = tf.image.decode_png(image_tensor,
                                       channels=FLAGS.channels)

    # resize to same size
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.float32)
    image_tensor = tf.image.resize_images(image_tensor, new_size)

    # Refer: https://tensorflow.google.cn/api_docs/python/tf/train/slice_input_producer
    all_matrix_tensor = tf.train.slice_input_producer([image_tensor, label_tensor],
                                                      num_epochs=FLAGS.epoch,
                                                      shuffle=False)


# Train
def train():
    train_image, train_label = get_data_file_list(FLAGS.train_data_dir)
    test_image, test_label = get_data_file_list(FLAGS.test_data_dir)

    # Debug
    if FLAGS.isDebug:
        for i in range(len(train_label)):
            logger.debug(str.format("Train data: %d : %s") % (train_label[i], train_image[i]))
        for i in range(len(test_label)):
            logger.debug(str.format("Test  data: %d : %s") % (test_label[i], test_image[i]))


# Output default information
def output_default_info():
    logger.info("============================ Start ============================")
    logger.info("===============================================================")
    logger.info("Basic information")
    logger.info("\tTensorflow version: " + tf.__version__)
    logger.info("\tepoch             : " + str(FLAGS.epoch))
    logger.info("\tchannels          : " + str(FLAGS.channels))
    logger.info("\timage_size        : " + str(FLAGS.image_size))


if __name__ == "__main__":
    # execute only if run as a script
    # output default information
    output_default_info()
    # start training
    train()

