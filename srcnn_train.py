import os
import numpy as np
import datetime
from srcnn_model2_model import SRCNNModel

from srcnn_utils import (
  read_data, 
  input_setup, 
  imsave,
  merge2
)
import tensorflow as tf
from srcnn_customize_tbcallback import customModelCheckpoint

class Flag():
    def __init__(self, is_train):
        self.epoch = 15000
        self.batch_size = 128
        self.image_size = 33
        self.label_size = 21
        self.learning_rate = 1e-4
        self.c_dim = 1
        self.scale = 3
        self.stride = 14
        self.checkpoint_dir = "checkpoint"
        self.data_dir = "data"
        self.sample_dir = "sample"
        self.is_train = is_train
        self.data_folder = "data"

train_flag = Flag(is_train=True)
input_setup(train_flag)
train_data_dir = os.path.join('./{}'.format(train_flag.data_folder), "train.h5")
train_data, train_label = read_data(train_data_dir)

test_data_dir = os.path.join('./{}'.format(train_flag.data_folder), "test.h5")
test_data, test_label = read_data(test_data_dir)

test_flag = Flag(is_train=False)
nx, ny, org_img, ground_tureth = input_setup(test_flag)

summary_folder=os.path.join('./logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

summary_writer_obj = tf.summary.create_file_writer(os.path.join(summary_folder, 'test'))

test_callback = customModelCheckpoint(summary_writer =  summary_writer_obj,
                                      feed_inputd_display=(test_data,test_label,nx,ny,test_flag.stride))


resume = False
is_train = True

Model = SRCNNModel(checkpoint_dir='ckpt', summary_folder=summary_folder)
if (resume and not is_train):
    Model.ResumeModel()
else:
    Model.BuildModel()
    Model.CustomizedCallback([test_callback])

if (is_train):
    Model.Train(train_data, train_label, 
                test_data, test_label, 
                train_flag.batch_size, train_flag.epoch)


predict_result = Model.Test(test_data)

result = merge2(predict_result, [nx, ny], test_data, test_flag.stride)
result = result.squeeze()
sx, sy = result.shape
image_path = os.path.join(os.getcwd(), test_flag.sample_dir)
image_path = os.path.join(image_path, "test_after_training.png")
imsave(result, image_path)



