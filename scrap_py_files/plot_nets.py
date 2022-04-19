import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Softmax, Dropout
import graphviz
import pydot

"""
Attempt to graphically plot a neural net. Actual code used in representative_net.txt.
"""

tuned1 = tf.keras.Sequential([
    Flatten(input_shape=(30, 30)),
    Dense(384, activation='relu'),
    Dense(265, activation='tanh'),
    Dense(2)])

tuned4 = tf.keras.Sequential([
    Flatten(input_shape=(30, 30)),
    Dense(288, activation='relu'),
    Dense(128, activation='relu'),
    Dense(320, activation='tanh'),
    Dropout(rate=0.25),
    Dense(2)])

tuned7 = tf.keras.Sequential([
    Flatten(input_shape=(30, 30)),
    Dense(224, activation='tanh'),
    Dense(384, activation='relu'),
    Dense(160, activation='relu'),
    Dropout(rate=0.5),
    Dense(2)])

tf.keras.utils.plot_model(tuned1, to_file='test.png', show_shapes=True, show_layer_activations=True)
tuned1_dot = tf.keras.utils.model_to_dot(tuned1, show_shapes=True, show_layer_activations=True)
