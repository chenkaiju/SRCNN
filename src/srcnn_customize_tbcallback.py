import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
from tensorflow.keras import backend as K

from srcnn_utils import (
  #read_data, 
  #input_setup, 
  #imsave,
  merge,
  merge2
)

class customModelCheckpoint(Callback):
    def __init__(self, summary_writer, feed_inputd_display = None):
        super(customModelCheckpoint, self).__init__()
        self.seen = 0
        self.feed_inputs_display = feed_inputd_display
        self.writer = summary_writer

    # this function will return the feeding data for TensorBoard visualization;
    # arguments:
    #  * feed_input_display : [(input_yourModelNeed, left_image, disparity_gt ), ..., (input_yourModelNeed, left_image, disparity_gt), ...], i.e., the list of tuples of Numpy Arrays what your model needs as input and what you want to display using TensorBoard. Note: you have to feed the input to the model with feed_dict, if you want to get and display the output of your model. 
    def custom_set_feed_input_to_display(self, feed_inputs_display):
        self.feed_inputs_display = feed_inputs_display

    # # copied from the above answers;
    # def make_image(self, numpy_img):
    #     from PIL import Image
    #     height, width, channel = numpy_img.shape
    #     image = Image.fromarray(numpy_img)
    #     import io
    #     output = io.BytesIO()
    #     image.save(output, format='PNG')
    #     image_string = output.getvalue()
    #     output.close()
    #     return tf.Summary.Image(height=height, width=width, colorspace= channel, encoded_image_string=image_string)


    # A callback has access to its associated model through the class property self.model.
    def on_epoch_end(self, epoch, logs = None):
        images, label, nx, ny, stride = self.feed_inputs_display

        predict_result = self.model.predict(images)

        #result = merge(predict_result, [nx, ny])
        result = merge2(predict_result, [nx, ny], images, stride)
        tbimages1 = np.expand_dims(result, axis=0)
        #result = result.squeeze()
        #sx, sy = result.shape
        #tbimages1 = np.reshape(result, (-1, sx, sy, 1))

        original = merge2(images, [nx, ny], images, stride)
        tbimages2 = np.expand_dims(original, axis=0)
        # tbimages2 = np.reshape(result, (-1, ssx, ssy, 1))

        tbImageGRoup = tf.concat([tbimages1, tbimages2], axis=2)

        with self.writer.as_default():
            tf.summary.image('test image', tbImageGRoup, step=epoch)
            #tf.summary.image('test image', tbimages1, step=epoch)
            #tf.summary.image('org image', tbimages2, step=epoch)

        #self.wandbRun.tensorflow.log(tf.summary.merge_all())