import os
import sys

ROWS = 8
COLS = 5
SIZE = ROWS * COLS


def load_pattern(file_path: str) -> list:
    flat = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                val = float(token)
                flat.append(1.0 if val > 0 else -1.0)
    if len(flat) != SIZE:
        raise ValueError(f"El archivo {file_path} tiene {len(flat)} valores, se esperaban {SIZE}.")
    return flat


def outer_product(vector_a: list, vector_b: list) -> list:
    length = len(vector_a)
    result = []
    for i in range(length):
        row = []
        for j in range(length):
            row.append(vector_a[i] * vector_b[j])
        result.append(row)
    return result


def add_matrices(matrix_a: list, matrix_b: list) -> list:
    length = len(matrix_a)
    result = []
    for i in range(length):
        row = []
        for j in range(length):
            row.append(matrix_a[i][j] + matrix_b[i][j])
        result.append(row)
    return result


def zero_diagonal(matrix: list) -> list:
    length = len(matrix)
    result = []
    for i in range(length):
        row = []
        for j in range(length):
            if i == j:
                row.append(0.0)
            else:
                row.append(matrix[i][j])
        result.append(row)
    return result


def step_activation(activation_value: float, current_state_value: float) -> float:
    if activation_value > 0:
        return 1.0
    if activation_value < 0:
        return -1.0
    return current_state_value


def update_vector(state: list, weights: list) -> list:
    length = len(weights)
    next_state = []
    for j in range(length):
        activation = 0.0
        for i in range(len(state)):
            activation += state[i] * weights[i][j]
        next_state.append(step_activation(activation, state[j]))
    return next_state


def train_hopfield(patterns: dict) -> list:
    weights = [[0.0 for _ in range(SIZE)] for _ in range(SIZE)]
    for name, pattern in patterns.items():
        outer = outer_product(pattern, pattern)
        weights = add_matrices(weights, outer)
    weights = zero_diagonal(weights)
    return weights


def run_hopfield(test_pattern: list, weights: list, max_iterations: int = 20) -> tuple:
    current_state = list(test_pattern)
    for iteration in range(1, max_iterations + 1):
        next_state = update_vector(current_state, weights)
        if next_state == current_state:
            return next_state, iteration
        current_state = next_state
    return current_state, max_iterations


def classify(vector: list, patterns: dict) -> tuple:
    best_name = None
    best_score = -float("inf")
    for name, pattern in patterns.items():
        score = sum(vector[i] * pattern[i] for i in range(SIZE))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, int(best_score)


def render_grid_side_by_side(v1: list, v2: list, label1: str, label2: str):
    print(f"\n{label1:<15}  |  {label2}")
    print("-" * 35)
    for r in range(ROWS):
        row1 = "".join("#" if v1[r * COLS + c] == 1.0 else "." for c in range(COLS))
        row2 = "".join("#" if v2[r * COLS + c] == 1.0 else "." for c in range(COLS))
        print(f"  {row1:<13}  |    {row2}")


def test_file(file_path: str, weights: list, training_patterns: dict):
    print("=" * 60)
    print(f"Evaluando archivo: {file_path}")
    test_vec = load_pattern(file_path)
    recovered, iters = run_hopfield(test_vec, weights)
    predicted_label, score = classify(recovered, training_patterns)

    render_grid_side_by_side(test_vec, recovered, "Entrada", f"Recuperado ({predicted_label})")
    print(f"\nConvergencia en {iters} iteracion(es).")
    print(f"Patron reconocido: {predicted_label} (coincidencia: {score} de {SIZE})")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    test_dir = os.path.join(dataset_dir, "test")

    train_files = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".txt"))
    training_patterns = {}
    for f in train_files:
        name = os.path.splitext(f)[0]
        training_patterns[name] = load_pattern(os.path.join(dataset_dir, f))

    print("=" * 60)
    print("RED HOPFIELD - RECONOCIMIENTO DE FIGURAS 8x5")
    print("=" * 60)
    print(f"Patrones de entrenamiento cargados ({len(training_patterns)}):")
    for name in training_patterns:
        print(f" - {name}")

    weights = train_hopfield(training_patterns)
    print(f"\nMatriz de pesos construida: {SIZE}x{SIZE} con diagonal cero.")

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        if not os.path.isabs(target_path):
            target_path = os.path.join(os.getcwd(), target_path)
        test_file(target_path, weights, training_patterns)
    else:
        test_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".txt"))
        print(f"\nEjecutando pruebas sobre carpeta {test_dir} ({len(test_files)} archivos):\n")
        for f in test_files:
            test_file(os.path.join(test_dir, f), weights, training_patterns)


if __name__ == "__main__":
    main()
