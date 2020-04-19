from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import matplotlib.pyplot as plt


def get_one_hot(data):
    # Init numpy array
    if data[0] == 0:
        ret_batch2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif data[0] == 1:
        ret_batch2 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif data[0] == 2:
        ret_batch2 = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    elif data[0] == 3:
        ret_batch2 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    elif data[0] == 4:
        ret_batch2 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    elif data[0] == 5:
        ret_batch2 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    elif data[0] == 6:
        ret_batch2 = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    elif data[0] == 7:
        ret_batch2 = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
    elif data[0] == 8:
        ret_batch2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    else:
        ret_batch2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    for i in range(1, len(data)):
        if data[i] == 0:
            tmparr = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        elif data[i] == 1:
            tmparr = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        elif data[i] == 2:
            tmparr = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        elif data[i] == 3:
            tmparr = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
        elif data[i] == 4:
            tmparr = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
        elif data[i] == 5:
            tmparr = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
        elif data[i] == 6:
            tmparr = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        elif data[i] == 7:
            tmparr = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
        elif data[i] == 8:
            tmparr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        else:
            tmparr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        ret_batch2 = np.append(ret_batch2, tmparr, axis=0)

    return ret_batch2


# Load Data
fashion_mnist = keras.datasets.fashion_mnist
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
trainX, testX = trainX / 255.0, testX / 255.0

# Training Parameters
batch_size = 128
learning_rate = 0.001
num_steps = int(60000 / batch_size)
display_step = batch_size
epochs = 6

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

tf.reset_default_graph()

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1, padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer with 3 filters and a kernel size of 5
    # conv1 = tf.layers.conv2d(x, 3, 5, activation=tf.nn.relu)
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], padding="VALID")
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    # conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer with 3 filters and a kernel size of 3
    # conv2 = tf.layers.conv2d(conv1, 3, 3, activation=tf.nn.relu)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    # conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    conv2 = maxpool2d(conv1, k=2)

    # Flatten the data to a 1-D vector for the fully connected layer
    # fc1 = tf.contrib.layers.flatten(conv2)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    # Fully connected layer
    # fc1 = tf.layers.dense(fc1, 100)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # fc1 = tf.layers.dense(fc1, 50)
    fc1 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc1 = tf.nn.relu(fc1)

    # Output layer, class prediction
    # out = tf.layers.dense(fc1, 10)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.get_variable("W0", shape=[5, 5, 1, 3], initializer=tf.contrib.layers.xavier_initializer()),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.get_variable("W1", shape=[3, 3, 3, 3], initializer=tf.contrib.layers.xavier_initializer()),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.get_variable("W2", shape=[108, 100], initializer=tf.contrib.layers.xavier_initializer()),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd2': tf.get_variable("W3", shape=[100, 50], initializer=tf.contrib.layers.xavier_initializer()),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.get_variable("W4", shape=[50, num_classes], initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
    'bc1': tf.get_variable("B0", shape=[3], initializer=tf.compat.v1.zeros_initializer()),
    'bc2': tf.get_variable("B1", shape=[3], initializer=tf.compat.v1.zeros_initializer()),
    'bd1': tf.get_variable("B2", shape=[100], initializer=tf.compat.v1.zeros_initializer()),
    'bd2': tf.get_variable("B3", shape=[50], initializer=tf.compat.v1.zeros_initializer()),
    'out': tf.get_variable("B4", shape=[num_classes], initializer=tf.compat.v1.zeros_initializer())
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    # saver.restore(sess, "lowlevel_model.cpkt")
    # Run the initializer
    sess.run(init)
    loss_function_avg = []
    for i in range(0, epochs):
        total_loss = []
        for step in range(1, num_steps + 1):

            batch_x, batch_y = trainX[batch_size * (step - 1):batch_size * step], trainY[batch_size * (step - 1):batch_size * step]
            batch_y = get_one_hot(batch_y)

            batch_x = tf.reshape(batch_x, shape=[batch_size, 784]).eval()
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                total_loss.append(loss)
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
        loss_function_avg.append(sum(total_loss)/len(total_loss))
    print("Optimization Finished!")
    # saved_model = saver.save(sess, "lowlevel_model.cpkt")
    # Calculate accuracy for test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: tf.reshape(testX, shape=[10000, 784]).eval(),
                                        Y: get_one_hot(testY)}))
    plt.plot(loss_function_avg)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()