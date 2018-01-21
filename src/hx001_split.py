# encoding: UTF-8
# This program will split gnt files to PNG.
# Each character will be stored in the same folder
import dataReader.imageReader as ir
import datetime

# Show Start time
print("== Start  == " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

base_dir = '../../'
training_data_dir = base_dir + 'data/HWDB1.1trn_gnt'
test_data_dir = base_dir + 'data/HWDB1.1tst_gnt'
# Debug flag
DEBUG_FLG = False
ir.DEBUG_FLG = DEBUG_FLG

# 数据文件列表
# training_data_files = ir.get_training_file_list(training_data_dir, validation_size=240)
# test_data_files = ir.get_training_file_list(test_data_dir, validation_size=60)
training_data_files = ir.get_training_file_list(training_data_dir, validation_size=2)
test_data_files = ir.get_training_file_list(test_data_dir, validation_size=1)

# Debug
if DEBUG_FLG:
    for __fls in training_data_files:
        print("[DEBUG][MAIN]Find file: " + __fls)

training_img_list, training_label_list = ir.get_training_data(training_data_files,
                                                              base_dir,
                                                              'png/train/',
                                                              char_set=None)
test_img_list, test_label_list = ir.get_training_data(test_data_files,
                                                      base_dir,
                                                      'png/test/',
                                                      char_set=None)

# Show Finished time
print("== Finish == " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
