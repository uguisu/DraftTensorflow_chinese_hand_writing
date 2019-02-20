# encoding: UTF-8
import os
import argparse
import yaml
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys

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
# tf.app.flags.DEFINE_string('checkpoint_dir', yml_settings['checkpoint_dir'], 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', yml_settings['train_data_dir'], 'the train data dir')
tf.app.flags.DEFINE_string('test_data_dir',  yml_settings['test_data_dir'],  'the test data dir')
tf.app.flags.DEFINE_string('model_dir',      yml_settings['model_dir'],      'the model dir')

tf.app.flags.DEFINE_integer('total_characters', yml_settings['total_characters'], 'total characters')
# tf.app.flags.DEFINE_integer('epoch',            yml_settings['epoch'],            'epoch')
tf.app.flags.DEFINE_integer('batch_size',       yml_settings['batch_size'],       'batch size')
tf.app.flags.DEFINE_integer('dataset_buffer',   yml_settings['dataset_buffer'],   'dataset buffer size')
tf.app.flags.DEFINE_integer('channels',         yml_settings['channels'],         'channels')
tf.app.flags.DEFINE_integer('image_size',       yml_settings['image_size'],       'image size')
tf.app.flags.DEFINE_integer('max_steps',        yml_settings['max_steps'],        'training max step')

tf.app.flags.DEFINE_float('learning_rate', yml_settings['learning_rate'], 'learning rate')
# tf.app.flags.DEFINE_float('drop_keep',     yml_settings['drop_keep'],     'drop keep')

FLAGS = tf.app.flags.FLAGS


def run_experiment(argv=None):
    """
    Run the training experiment.
    """
    # Define model parameters
    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        # n_classes = 10
        # training_step=5000,
        min_eval_frequency=100
    )

    # Set the run_config and the directory to save the model and stats
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=params.min_eval_frequency
    )

    # Define the classifier
    estimator = get_estimator(run_config, params)

    # Setup data loaders
    train_input_fn = get_train_inputs(batch_size=FLAGS.batch_size, training_data=FLAGS.train_data_dir)
    eval_input_fn = get_test_inputs(batch_size=FLAGS.batch_size, test_data=FLAGS.test_data_dir)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=FLAGS.max_steps,
        hooks=None
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=100,
        hooks=None
    )

    # Run process
    tf.estimator.train_and_evaluate(
        estimator=estimator,
        # All training related specification is held in train_spec,
        # including training input_fn and training max steps, etc.
        train_spec=train_spec,
        # All evaluation and export related specification is held in eval_spec,
        # including evaluation input_fn, steps, etc.
        eval_spec=eval_spec
    )


def get_estimator(run_config, params):
    """
    Return the model as a Tensorflow Estimator object
    :param run_config: (RunConfig) Configuration for Estimator run.
    :param params: (HParams) hyperparameters.
    :return: (Estimator) Estimator object
    """
    return tf.estimator.Estimator(
        # first-class function
        model_fn=model_fn,
        # HParams
        params=params,
        # Run Config
        config=run_config
    )


def model_fn(feature, labels, mode, params):
    """
    Model function used in the estimator
    :param feature: (Tensor) Input features to the model.
    :param labels: (Tensor)  Labels tensor for training and evaluation.
    :param mode: (ModeKeys) Specifies if training, evaluation or prediction.
    :param params: (HParams) Hyperparameters.
    :return: (EstimatorSpec) Model to be run by Estimator.
    """
    is_training = mode == ModeKeys.TRAIN
    # Define model's architecture
    logits = architecture(feature, is_training=is_training)
    predictions = tf.argmax(logits, axis=-1)
    # Loss, training and eval operations are not needed during inference
    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.INFER:
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels, tf.int32),
            logits=logits
        )
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def get_train_op_fn(loss, params):
    """
    Get the training Op
    :param loss: (Tensor) Scalar Tensor that represents the loss function.
    :param params: (HParams) Hyperparameters (needs to have 'learning_rate').
    :return: Training Op
    """
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


def get_eval_metric_ops(labels, predictions):
    """
    Return a dict of the evaluation Ops.
    :param labels: (Tensor) Labels tensor for training and evaluation.
    :param predictions: (Tensor) Predictions Tensor.
    :return: Dict of metric results keyed by name.
    """
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy'
        )
    }


def architecture(inputs, is_training, scope='HandWritingConvNet'):
    """
    Return the output operation following the network architecture

    input => (?, 64, 64, 1)
    conv1 => (?, 64, 64, 64)
    pool2 => (?, 32, 32, 64)
    conv3 => (?, 32, 32, 128)
    pool4 => (?, 16, 16, 128)
    resharp => (?, 32768)   32768 = 16*16*128

    :param inputs: (Tensor) Input tensor
    :param is_training: (boolean) True, if in training mode
    :param scope: (str) Name of the scope of the architecture
    :return: Neural Network
    """
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer()):
            net = slim.conv2d(inputs, 64, [2, 2], padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
            net = slim.conv2d(net, 128, [2, 2], padding='SAME', scope='conv3')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')
            net = tf.reshape(net, [-1, 16 * 16 * 128])
            net = slim.fully_connected(net, 3755 * 2, scope='fn5')
            net = slim.dropout(net, is_training=is_training, scope='dropout6')
            net = slim.fully_connected(net, 3755 * 2, scope='fn7')
            net = slim.dropout(net, is_training=is_training, scope='dropout8')
            net = slim.fully_connected(net, 3755, scope='output', activation_fn=None)

    # Take care of how many classes of your result. In this case, the final result will be 3755.
    return net


def get_train_inputs(batch_size, training_data):
    """
    Define the training input
    :param batch_size: (int) batch size
    :param training_data: (str) training data
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
        dataset = dataset.shuffle(buffer_size=FLAGS.dataset_buffer)
        # 4) Combines consecutive elements of this dataset into batches.
        dataset = dataset.batch(batch_size)

        return dataset

    return train_inputs


def get_test_inputs(batch_size, test_data):
    """
    Define the test input
    :param batch_size: (int) batch size
    :param test_data: (str) test data
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
        dataset = dataset.shuffle(buffer_size=FLAGS.dataset_buffer)
        # 4) Combines consecutive elements of this dataset into batches.
        dataset = dataset.batch(batch_size)

        return dataset

    return test_inputs


def get_data_file_list(data_dir):
    """
    Get data file list
    :param data_dir: (str) data full path
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
    :param filename: (str) image file name
    :param label: (str) label
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

    tf.app.run(
        main=run_experiment
    )
