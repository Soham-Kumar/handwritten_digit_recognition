import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

dataset = tf.keras.datasets.mnist
(image_train, number_train), (image_test, number_test) = dataset.load_data()

# Normalizing the pixel data (Scaling down data of range 0-255 to 0-1)
# image_train = image_train / 255.0

image_train = tf.keras.utils.normalize(image_train, axis=1)
image_test = tf.keras.utils.normalize(image_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(image_train, number_train, epochs = 3)

model.save('handwritten_digit_recognition.model')