# encoding: UTF-8
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
tf.app.flags.DEFINE_integer('image_size',       yml_settings['image_size'],       'image size')

tf.app.flags.DEFINE_float('learning_rate', yml_settings['learning_rate'], 'learning rate')
tf.app.flags.DEFINE_float('drop_keep',     yml_settings['drop_keep'],     'drop keep')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    print(tf.__version__)
