import os
from sys import flags
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import datetime

from srcnn_utils import (
  read_data, 
  input_setup, 
  imsave,
  merge2
)

from srcnn_customize_tbcallback import customModelCheckpoint


class SRCNNModel():
    def __init__(self, checkpoint_dir='checkpoint', summary_folder='logs'):
        self.Model = None
        self.Checkpoint_dir = checkpoint_dir

        self.summary_folder = summary_folder

        self.summary_writer_obj = tf.summary.create_file_writer(os.path.join(self.summary_folder,'test'))

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.summary_folder, 
                                                                profile_batch=0, 
                                                                write_images=True)

        SaveCheckpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.Checkpoint_dir, 'model.h5'), 
                                                        monitor='loss', 
                                                        save_best_only=True, mode='min')

        self.callback = [tensorboard_callback, SaveCheckpoint]

    def CustomizedCallback(self, callback_list):
        self.callback.extend(callback_list)

    def ResumeModel(self):
        self.Model = keras.models.load_model('ckpt/model.h5', compile=True)

        return self.Model is not None

    def BuildModel(self, learning_rate=1e-4): # Model definition

        img_inputs = keras.Input(shape=(33, 33, 1))
        conv1 = Conv2D(filters=64, kernel_size=9, activation='relu')
        conv2 = Conv2D(filters=32, kernel_size=1, activation='relu')
        conv3 = Conv2D(filters=1, kernel_size=5)

        x = conv1(img_inputs)
        x = conv2(x)
        output = conv3(x)

        self.Model = keras.Model(inputs=img_inputs, outputs=output, name="SRCNN_model")
        self.Model.summary()
        keras.utils.plot_model(self.Model, "srcnn_model.png")
        keras.utils.plot_model(self.Model, "srcnn_model_with_shape_info.png", show_shapes=True)

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer_object = tf.keras.optimizers.SGD(learning_rate)

        train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.Model.compile(
            loss=loss_object,
            optimizer=optimizer_object
        )

    def Train(self, train_data, train_label, test_data, test_label, batch_size, epoch):
        print("Fit model on training data")

        history = self.Model.fit(
            train_data,
            train_label,
            batch_size=batch_size,
            epochs=epoch,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(test_data, test_label),
            callbacks=self.callback
        )

        return history

    def Test(self, test_data):
        predict_result = self.Model.predict(test_data)

        return predict_result

    def Predict(self, test_data):
        predict_result = self.Model.predict(test_data)

        return predict_result
