import tensorflow as tf


def function(shape):
    X = tf.Variable(tf.random.uniform(shape=shape))
    return X
