import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

from .positional_enconding import PositionalEncoding
from .enconder_layer import EncoderLayer


class Encoder(layers.Layer):
    def __init__(
        self,
        number_of_layers,
        FFN_units,
        number_of_projections,
        dropout_rate,
        vocab_size,
        d_model,
        name="encoder",
    ):
        super(Encoder, self).__init__(name=name)
        self.number_of_layers = number_of_layers
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.enc_layers = [
            EncoderLayer(FFN_units, number_of_projections, dropout_rate)
            for _ in range(number_of_layers)
        ]

    def call(self, inputs, mask, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)

        for i in range(self.number_of_layers):
            outputs = self.enc_layers[i](outputs, mask, training)

        return outputs