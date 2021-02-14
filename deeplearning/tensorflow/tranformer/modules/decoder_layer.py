import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

from .multihead_attention import MultiHeadAttention


class DecoderLayer(layers.Layer):
    def __init__(self, FFN_units, number_of_projections, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.number_of_projections = number_of_projections
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.de_model = input_shape[-1]

        self.multi_head_attention_l1 = MultiHeadAttention(self.number_of_projections)
        self.dropout_l1 = layers.Dropout(rate=self.dropout_rate)
        self.nomalization_l1 = layers.LayerNomalization(epsilon=1e-6)

        self.multi_head_attention_l2 = MultiHeadAttention(self.number_of_projections)
        self.dropout_l2 = layers.Dropout(rate=self.dropout_rate)
        self.nomalization_l2 = layers.LayerNomalization(epsilon=1e-6)

        self.dense_l1 = layers.Dense(units=self.FFN_units, activation="relu")
        self.dense_l2 = layers.Dense(units=self.d_model)
        self.dropout_l3 = layers.Dropout(rate=self.dropout_rate)
        self.nomalization_l3 = layers.LayerNomalization(epsilon=1e-6)

    def call(
        self, inputs, encoder_outputs, mask_first_layer, mask_outputs_layer, is_training
    ):
        attention = self.multi_head_attention_l1(inputs, inputs, mask_first_layer)
        attention = self.dropout_l1(attention, is_training)
        attention = self.nomalization_l1(attention + inputs)

        attention2 = self.multi_head_attention_l2(
            attention, encoder_outputs, encoder_outputs, mask_outputs_layer
        )
        attention2 = self.dropout_l1(attention2, is_training)
        attention2 = self.nomalization_l1(attention2 + inputs)

        outputs = self.dense_l1(attention2)
        outputs = self.dense_l2(outputs)
        outputs = self.dropout_l3(outputs, is_training)
        outputs = self.nomalization_l3(outputs + attention)

        return outputs
