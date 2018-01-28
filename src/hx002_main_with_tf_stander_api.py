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
import time
import tensorflow as tf

# ====================================
# ======   Declare Log system   ======
# ====================================
logger = logging.getLogger('[Learn Tensorflow] Chinese hand writing')
logger.setLevel(logging.DEBUG)

logfile = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + '.log'

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
tf.app.flags.DEFINE_integer('max_steps',       16002, 'the max training steps ')
tf.app.flags.DEFINE_float('learn_rate',          0.1, 'learn rate')
tf.app.flags.DEFINE_float('drop_keep',          0.75, 'drop keep')

FLAGS = tf.app.flags.FLAGS

# =====================================
# ======   Declare variables     ======
# =====================================
global_random_range = []

K = 32    # 1 convolutional layer output depth
L = 64    # 2 convolutional layer output depth
M = 1024  # full connection layer
N = FLAGS.total_characters  # full connection layer

# input X: 64 x 64 grayscale images, the first dimension (None) will index the images in the mini-batch
#          [batch, in_height, in_width, in_channels]
X = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.channels])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# The probability that each element is kept.
pkeep = tf.placeholder(tf.float32)

# filter W1 : 2 x 2 patch, 1 input channel, K output channels
#             [filter_height, filter_width, in_channels, out_channels]
W1 = tf.Variable(tf.truncated_normal([2, 2, FLAGS.channels, K], stddev=0.1))
B1 = tf.Variable(tf.random_normal([K], stddev=0.1))
# filter W2 : 2 x 2 patch, K input channel, L output channels
#             [filter_height, filter_width, in_channels, out_channels]
W2 = tf.Variable(tf.truncated_normal([2, 2, K, L], stddev=0.1))
B2 = tf.Variable(tf.random_normal([L], stddev=0.1))

W3 = tf.Variable(tf.truncated_normal([32 * 32 * L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))


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


def model_network():

    # tf.nn.conv2d()        -> https://tensorflow.google.cn/api_docs/python/tf/nn/conv2d
    # tf.nn.bias_add()      -> https://tensorflow.google.cn/api_docs/python/tf/nn/bias_add
    # tf.truncated_normal() -> https://tensorflow.google.cn/api_docs/python/tf/truncated_normal
    # tf.random_normal()    -> https://tensorflow.google.cn/api_docs/python/tf/random_normal

    # Convolutional Layer #1
    stride = 1  # output is 64x64
    conv1 = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
    bias1 = tf.nn.bias_add(conv1, B1)
    relu1 = tf.nn.relu(bias1)

    # Pooling Layer #1
    # [NOTICE] ksize: A 1-D int Tensor of 4 elements.
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME')

    # Convolutional Layer #2
    stride = 1  # output is 32x32
    conv2 = tf.nn.conv2d(pool1, W2, strides=[1, stride, stride, 1], padding='SAME')
    bias2 = tf.nn.bias_add(conv2, B2)
    relu2 = tf.nn.relu(bias2)

    # Pooling Layer #2
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME')

    # Full Layer
    # reshape the output from the third convolution for the fully connected layer
    pool2_flat = tf.reshape(relu1, shape=[-1, 32 * 32 * L])
    fc1 = tf.matmul(pool2_flat, W3) + B3
    relu3 = tf.nn.relu(fc1)
    dropout1 = tf.nn.dropout(relu3, pkeep)

    # Logits Layer
    ylogits = tf.matmul(dropout1, W4) + B4
    y = tf.nn.softmax(ylogits)

    # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=Y_)
    loss = tf.reduce_mean(loss)*100

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training step, the learning rate is a placeholder
    train_step = tf.train.AdamOptimizer(FLAGS.learn_rate).minimize(loss)

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)

    return {
        "X": X,
        "Y_": Y_,
        "pkeep": pkeep,
        "loss": loss,
        "accuracy": accuracy,
        "train_step": train_step,
        "global_step": global_step
    }

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

    model_net = model_network()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                start_time = time.time()

                loss, accuracy, train_step, global_step = sess.run([model_net['loss'],
                                                                   model_net['accuracy'],
                                                                   model_net['train_step'],
                                                                   model_net['global_step']],
                                                                  feed_dict={model_net['X']: train_image_batch,
                                                                             model_net['Y_']: train_label_batch,
                                                                             model_net['pkeep']: FLAGS.drop_keep})
                end_time = time.time()
                if global_step > FLAGS.max_steps:
                    break
                else:
                    logger.info("the step {0} takes {1} loss {2}".format(global_step, end_time - start_time, loss))
        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
        finally:
            coord.request_stop()
        coord.join(threads)


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
    logger.info("\tmax steps         : " + str(FLAGS.max_steps))
    logger.info("\tlearn rate        : " + str(FLAGS.learn_rate))
    logger.info("\tdrop keep         : " + str(FLAGS.drop_keep))


if __name__ == "__main__":
    # execute only if run as a script
    # output default information
    output_default_info()
    # start training
    train()
