import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

from .decoder_layer import DecoderLayer
from .positional_enconding import PositionalEncoding


class Decoder(layers.Layer):
    def __init__(
        self,
        number_of_layers,
        FFN_units,
        number_of_projections,
        dropout_rate,
        vocab_size,
        d_model,
        name="decoder",
    ):
        super(Decoder, self).__init__(name=name)
        self.number_of_layers = number_of_layers
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.decoding_layers = [
            DecoderLayer(FFN_units, number_of_projections, dropout_rate)
            for _ in range(number_of_layers)
        ]

    def call(
        self, inputs, encoder_outputs, mask_first_layer, mask_outputs_layer, is_training
    ):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, is_training)

        for i in range(self.number_of_layers):
            outputs = self.decoding_layers[i](
                outputs, mask_first_layer, mask_outputs_layer, is_training
            )

        return outputs
