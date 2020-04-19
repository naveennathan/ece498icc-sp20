import tensorflow as tf


def function(loss, lr):
    optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)
    return optimizer

