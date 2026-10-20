import numpy as np
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow Version: {tf.__version__}")

# Define single-layer model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Print model architecture
model.summary()

# Retrieve initial un-trained weights W and bias b
weights = model.get_weights()
w, b = weights[0], weights[1]
print("\nInitial Random Weights:")
print(f"W (shape {w.shape}):\n{w}")
print(f"b (shape {b.shape}):\n{b}")

# Demonstrating dynamic shape adjustment with 2 input features
model_2d = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2])
])
w_2d, b_2d = model_2d.get_weights()
print("\nWeights with input_shape=[2]:")
print(f"W_2d (shape {w_2d.shape}):\n{w_2d}")
print(f"b_2d (shape {b_2d.shape}):\n{b_2d}")
