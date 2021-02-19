import tensorflow as tf
from tensorflow.keras import layers

from .encoder import Encoder
from .decoder import Decoder


class Transformer(tf.keras.Model):
    def __init__(
        self,
        vocab_size_enc,
        vocab_size_dec,
        d_model,
        number_of_layers,
        FFN_units,
        number_of_projections,
        dropout_rate,
        name="transformer",
    ):
        super(Transformer, self).__init__(name=name)
        self.encoder = Encoder(
            number_of_layers,
            FFN_units,
            number_of_projections,
            dropout_rate,
            vocab_size_enc,
            d_model,
        )
        self.decoder = Decoder(
            number_of_layers,
            FFN_units,
            number_of_projections,
            dropout_rate,
            vocab_size_dec,
            d_model,
        )
        self.last_linear = layers.Dense(units=vocab_size_dec)

    def create_padding_mask(self, sequence):  # sequence (batch_size, sequence_length)
        mask = tf.cast(tf.math.equal(sequence, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, sequence):
        sequence_length = tf.shape(sequence)[1]
        look_ahead_mask = tf.linalg.band_part(
            tf.ones((sequence_length, sequence_length)), -1, 0
        )
        return look_ahead_mask

    def create_decoder_mask(self, inputs_dec):
        dec_padding_mask = self.create_padding_mask(inputs_dec)
        dec_look_ahead_mask = self.create_look_ahead_mask(inputs_dec)
        dec_mask = tf.maximum(dec_padding_mask, dec_look_ahead_mask)
        return dec_mask

    def call(self, inputs_enc, inputs_dec, is_training):
        enc_mask = self.create_padding_mask(inputs_enc)
        dec_mask = self.create_decoder_mask(inputs_dec)
        dec_padding_mask = self.create_padding_mask(inputs_enc)

        encoder_outputs = self.encoder(inputs_enc, enc_mask, is_training)
        decoder_outputs = self.decoder(
            inputs_dec, encoder_outputs, dec_mask, dec_padding_mask, is_training
        )
        outputs = self.last_linear(decoder_outputs)

        return outputs
