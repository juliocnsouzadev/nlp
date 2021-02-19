import tensorflow as tf


class Losses:
    def __init__(self):
        super(Losses, self).__init__()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )


def loss_function(self, target, predictions):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = self.loss_object(target, predictions)
    mask = tf.cast(mask, dtype=loss_.dtype)  # making sure they are same type
    loss_ *= mask
    return tf.reduce_mean(loss_)
