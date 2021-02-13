import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow as tf


class PositionalEncoding(layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model):
        angles = 1 / np.power(10000.0, (2 * (i / 2) / np.float32(d_model)))
        return pos * angles

    def call(self, inputs):
        inputs_shape_list = inputs.shape.as_list()
        seq_length = inputs_shape_list[-2]
        d_model = inputs_shape_list[-1]
        pos = np.arange(seq_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angles = self.get_angles(pos, i, d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return inputs + tf.cast(pos_encoding, tf.float32)
