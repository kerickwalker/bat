import tensorflow as tf
from tensorflow.keras import models, layers

conv_network = models.Sequential()
conv_network.add(layers.Conv2D(2, (11, 11), activation='relu', input_shape=(400, 100, 3)))
conv_network.add(layers.MaxPooling2D(5,5))
conv_network.add(layers.Conv2D(2, (4,4), activation='relu'))
conv_network.add(layers.MaxPooling2D(3,3))
conv_network.add(layers.Flatten())
conv_network.add(layers.Dense(2, activation='relu'))
conv_network.summary()