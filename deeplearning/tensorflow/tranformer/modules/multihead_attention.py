import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

from .attention_utils import AttentionUtils


class MultiHeadAttention(layers.Layer):
    def __init__(self, number_of_projection):
        super(MultiHeadAttention, self).__init__()
        self.number_of_projection = number_of_projection
        self.attention_utils = AttentionUtils()

    # this method is called at the first time the object is use
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        # assert the integers result in zero
        assert self.d_model % self.number_of_projection == 0
        # assure integer result
        self.d_proj = self.d_model // self.number_of_projection
        # create the dense layers
        self.query_linear_func = layers.Dense(units=self.d_model)
        self.keys_linear_func = layers.Dense(units=self.d_model)
        self.value_linear_func = layers.Dense(units=self.d_model)
        self.final_linear_func = layers.Dense(units=self.d_model)

    def split_projections(self, inputs, batch_size):
        shape = (batch_size, -1, self.number_of_projection, self.d_proj)
        splited_inputs = tf.reshape(
            inputs, shape=shape
        )  # (batch_size, seq_length, number_of_projections, d_proj)
        # shifting the 2nd and 3rd args
        splited_inputs = tf.transpose(
            splited_inputs, perm=[0, 2, 1, 3]
        )  # (batch_size, number_of_projections, seq_length,  d_proj)
        return splited_inputs

    def call(self, queries, keys, values, mask):
        batch_size = tf.shape(queries)[0]
        # build the dense layers
        queries = self.query_linear_func(queries)
        keys = self.keys_linear_func(keys)
        values = self.value_linear_func(values)
        # split the inputs
        queries = self.split_projections(queries, batch_size)
        keys = self.split_projections(keys, batch_size)
        values = self.split_projections(values, batch_size)
        attention = self.attention_utils.scaled_dot_product_attention(
            queries, keys, values, mask
        )
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, shape=(batch_size, -1, self.d_model))
        outputs = self.final_linear_func(concat_attention)
        return outputs
