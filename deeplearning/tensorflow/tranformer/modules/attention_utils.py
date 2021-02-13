import tensorflow as tf


class AttentionUtils:
    def __init__(self):
        super(AttentionUtils, self).__init__()

    def scaled_dot_product_attention(self, queries, keys, values, mask):
        product = tf.matmul(queries, keys, transpose_b=True)
        keys_dimmention = tv.cast(tf.shape(keys)[-1], tf.float32)
        scaled_product = product / tf.math.sqrt(keys_dimmention)
        if mask is not None:
            scaled_product += mask * -1e9
        attention = tf.matmul(tf.nn.softmax(scaled_product, axis=1), values)
        return attention
