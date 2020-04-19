from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import h5py

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(3, (5, 5), strides=1, padding='valid', activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(strides=2),

    keras.layers.Conv2D(3, (3, 3), strides=1, padding='same', activation='relu', input_shape=(12, 12, 3)),
    keras.layers.MaxPooling2D(strides=2),

    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
train_image_reshape = train_images.reshape(60000, 28, 28, 1)
test_image_reshape = test_images.reshape(10000, 28, 28, 1)
history = model.fit(train_image_reshape, train_labels, validation_data=(test_image_reshape, test_labels), epochs=10, batch_size=32)

# model.save('keras_model')
# new_model = tf.keras.models.load_model('keras_model')
# new_model.summary()

test_loss, test_acc = model.evaluate(test_image_reshape, test_labels, verbose=2)
# test_loss, test_acc = new_model.evaluate(test_image_reshape, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()