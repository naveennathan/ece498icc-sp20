import tensorflow as tf


def function(a, X, b, y):
    Z = a * tf.matmul(tf.transpose(X), X) + tf.matmul(tf.transpose(b), X)
    loss = tf.math.squared_difference(Z, y)
    return loss
