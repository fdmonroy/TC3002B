import os
import openpyxl
import numpy as np
import pandas as pd

# Activation functions and their derivatives
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def d_sigmoid(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def d_relu(a: np.ndarray) -> np.ndarray:
    return (a > 0.0).astype(np.float64)

def mse(pred: np.ndarray, true: np.ndarray) -> tuple[float, np.ndarray]:
    loss = float(np.mean((pred - true) ** 2))
    grad = pred - true
    return loss, grad

class Layer:
    def __init__(self, in_features: int, out_features: int, is_output: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.is_output = is_output
        # Scaled uniform initialization
        scale = 0.2 if is_output else 0.05
        self.W = np.random.uniform(-scale, scale, (in_features, out_features))
        self.b = np.zeros((1, out_features))

class NeuralNetwork:
    def __init__(self, architecture: list[int] = [2, 4, 8, 1]):
        self.layers = []
        for i in range(len(architecture) - 1):
            is_out = (i == len(architecture) - 2)
            self.layers.append(Layer(architecture[i], architecture[i + 1], is_output=is_out))

    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        activations = [X]
        for i, layer in enumerate(self.layers):
            z = activations[-1] @ layer.W + layer.b
            a = sigmoid(z) if layer.is_output else relu(z)
            activations.append(a)
        return activations

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 500, lr: float = 0.001) -> list[float]:
        history = []
        for epoch in range(1, epochs + 1):
            activations = self.forward(X)
            output = activations[-1]
            loss, d_loss = mse(output, Y)
            history.append(loss)

            # Backpropagation
            delta = d_loss * d_sigmoid(output)
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                prev_a = activations[i]

                grad_W = prev_a.T @ delta
                grad_b = np.sum(delta, axis=0, keepdims=True)

                if i > 0:
                    prev_layer = self.layers[i - 1]
                    delta = (delta @ layer.W.T) * d_relu(activations[i])

                layer.W -= grad_W * lr
                layer.b -= grad_b * lr

            if epoch % 100 == 0 or epoch == epochs:
                print(f"Epoch {epoch:4d}/{epochs} - Loss (MSE): {loss:.6f}")

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[-1]

def load_data(base_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load City 1 and City 2 datasets and partition into train and validation sets."""
    c1_path = os.path.join(base_dir, "data", "city1.csv")
    c2_path = os.path.join(base_dir, "data", "city2.csv")

    if not os.path.exists(c1_path) or not os.path.exists(c2_path):
        from dataset import save_datasets
        print("Datasets not found. Generating default city coordinates...")
        save_datasets(output_dir=os.path.join(base_dir, "data"), num_points=300)

    df1 = pd.read_csv(c1_path)
    df2 = pd.read_csv(c2_path)

    # 10 validation points from each city (20 total)
    val1 = df1.iloc[:10][["lat", "lon"]].values
    val2 = df2.iloc[:10][["lat", "lon"]].values
    X_val = np.vstack([val1, val2])
    Y_val = np.vstack([np.zeros((10, 1)), np.ones((10, 1))])

    # Remaining points for training
    train1 = df1.iloc[10:][["lat", "lon"]].values
    train2 = df2.iloc[10:][["lat", "lon"]].values
    X_train = np.vstack([train1, train2])
    Y_train = np.vstack([np.zeros((len(train1), 1)), np.ones((len(train2), 1))])

    return X_train, Y_train, X_val, Y_val

def update_excel_validation(base_dir: str, nn: NeuralNetwork, X_val: np.ndarray, Y_val: np.ndarray):
    """Populate template Excel validation sheet with test points and trained weights."""
    template_path = os.path.join(base_dir, "template_red_neuronal.xlsx")
    output_path = os.path.join(base_dir, "validation_weights.xlsx")

    if not os.path.exists(template_path):
        print(f"Template {template_path} not found. Skipping Excel export.")
        return

    wb = openpyxl.load_workbook(template_path)
    ws = wb["tensorflow validation N=20"]

    # Write 20 validation coordinates (Rows 3 to 22, Columns B and C)
    for i in range(20):
        row = 3 + i
        ws.cell(row=row, column=2, value=float(X_val[i, 0]))  # Latitud (Col B)
        ws.cell(row=row, column=3, value=float(X_val[i, 1]))  # Longitud (Col C)
        ws.cell(row=row, column=53, value=float(Y_val[i, 0])) # Target label (Col BA)

    # Layer 1: W1 (2x4) in E3:H4, b1 (1x4) in E8:H8
    W1 = nn.layers[0].W
    b1 = nn.layers[0].b
    for r in range(2):
        for c in range(4):
            ws.cell(row=3 + r, column=5 + c, value=float(W1[r, c]))
    for c in range(4):
        ws.cell(row=8, column=5 + c, value=float(b1[0, c]))

    # Layer 2: W2 (4x8) in T3:AA6, b2 (1x8) in T9:AA9
    W2 = nn.layers[1].W
    b2 = nn.layers[1].b
    for r in range(4):
        for c in range(8):
            ws.cell(row=3 + r, column=20 + c, value=float(W2[r, c]))
    for c in range(8):
        ws.cell(row=9, column=20 + c, value=float(b2[0, c]))

    # Layer 3: W3 (8x1) in AU3:AU10, b3 (1x1) in AU13
    W3 = nn.layers[2].W
    b3 = nn.layers[2].b
    for r in range(8):
        ws.cell(row=3 + r, column=47, value=float(W3[r, 0]))
    ws.cell(row=13, column=47, value=float(b3[0, 0]))

    wb.save(output_path)
    print(f"Validation sheet saved to: {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    np.random.seed(42)

    print("Loading city datasets...")
    X_train, Y_train, X_val, Y_val = load_data(base_dir)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    print("\nInitializing Neural Network [2, 4, 8, 1] (ReLU, ReLU, Sigmoid)...")
    nn = NeuralNetwork(architecture=[2, 4, 8, 1])

    print("Training network (500 epochs, lr=0.001)...")
    nn.train(X_train, Y_train, epochs=500, lr=0.001)

    print("\n--- Validation on 20 GPS Points ---")
    predictions = nn.predict(X_val)

    val_df = pd.DataFrame({
        "Lat": X_val[:, 0],
        "Lon": X_val[:, 1],
        "City": ["City 1 (Buenos Aires)"] * 10 + ["City 2 (Cairo)"] * 10,
        "Target": Y_val.flatten().astype(int),
        "Predicted": predictions.flatten().round(4),
        "Class": (predictions.flatten() >= 0.5).astype(int)
    })
    print(val_df.to_string(index=False))

    acc = np.mean(val_df["Target"] == val_df["Class"]) * 100.0
    print(f"\nValidation Accuracy: {acc:.1f}%")

    # Export to Excel sheet
    update_excel_validation(base_dir, nn, X_val, Y_val)

if __name__ == "__main__":
    main()
