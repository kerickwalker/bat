import tensorflow as tf
from tensorflow.keras import models, layers

dumb_network = models.Sequential()
dumb_network.add(layers.Dense(100, activation='relu', input_shape=(400, 100, 3)))
dumb_network.add(layers.Dense(20, activation='relu'))
dumb_network.add(layers.Dense(2, activation='relu'))
dumb_network.summary()