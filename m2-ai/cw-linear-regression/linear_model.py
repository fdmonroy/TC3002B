import os
import shutil
import numpy as np
import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate dataset: y = 2x + 1
X = np.arange(-10.0, 11.0, 1.0, dtype=np.float32)
Y = 2.0 * X + 1.0

print(f"Generated {len(X)} samples.")
print(f"Sample X: {X[:5]}")
print(f"Sample Y: {Y[:5]}")

# Shuffle dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Split dataset into training and validation sets (80% / 20%)
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]

print(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples.")

# Define model architecture (Single Dense layer)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile model with SGD optimizer and MeanSquaredError loss/metric
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanSquaredError()]
)

# Train model for 20 epochs
print("\nStarting training (20 epochs)...")
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    verbose=1
)

# Predictions evaluation
test_inputs = np.array([0.0, 1.0, 4.0], dtype=np.float32)
predictions = model.predict(test_inputs)

print("\n--- Predictions ---")
for x_val, pred in zip(test_inputs, predictions):
    print(f"Input: {x_val:.1f} -> Predicted: {float(pred[0]):.4f} (Expected: {2.0 * x_val + 1.0:.1f})")

# Export model for TensorFlow Serving
export_path = os.path.join(os.path.dirname(__file__), "linear_model", "1")
if os.path.exists(export_path):
    shutil.rmtree(export_path)

print(f"\nSaving model to {export_path} for TensorFlow Serving...")
try:
    # Keras 3 / TF 2.x export
    model.export(export_path)
    print("Model exported via model.export() successfully.")
except Exception as e:
    # Fallback to tf.saved_model.save
    tf.saved_model.save(model, export_path)
    print(f"Model exported via tf.saved_model.save(). (Info: {e})")

# Verify export directory
if os.path.exists(os.path.join(export_path, "saved_model.pb")):
    print(f"SavedModel bundle confirmed at {export_path}/saved_model.pb")
