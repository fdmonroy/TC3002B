import os
import shutil
import numpy as np
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate dataset: y = 2x + 1
X = np.arange(-10.0, 11.0, 1.0, dtype=np.float32)
Y = 2.0 * X + 1.0

# Shuffle dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Split dataset (80% train, 20% validation)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

# Define single dense layer model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.summary()

# Initial weights before training
weights = model.get_weights()
print(f"Initial W: {weights[0].flatten()}, b: {weights[1]}")

# Compile with SGD and MeanSquaredError
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanSquaredError()]
)

# Train model for 20 epochs
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, verbose=1)

# Evaluate predictions
test_inputs = np.array([0.0, 1.0, 4.0], dtype=np.float32)
predictions = model.predict(test_inputs)
print("\nPredictions:")
for x_val, pred in zip(test_inputs, predictions):
    print(f"Input: {x_val:.1f} -> Predicted: {float(pred[0]):.4f} (Expected: {2.0 * x_val + 1.0:.1f})")

# Export model for TensorFlow Serving
export_path = os.path.join(os.path.dirname(__file__), "linear_model", "1")
if os.path.exists(export_path):
    shutil.rmtree(export_path)

model.export(export_path)
print(f"\nModel exported to: {export_path}")
