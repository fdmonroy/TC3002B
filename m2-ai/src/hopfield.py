import os


def load_pattern_file(file_path: str, expected_rows: int = 8, expected_cols: int = 5) -> list:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    if len(raw_lines) != expected_rows:
        raise ValueError(
            f"Invalid row count in {file_path}. Expected {expected_rows}, received {len(raw_lines)}."
        )

    flat_vector = []
    for row_index, line in enumerate(raw_lines):
        tokens = line.split()
        if len(tokens) != expected_cols:
            raise ValueError(
                f"Invalid column count in {file_path} at row {row_index}. Expected {expected_cols}, received {len(tokens)}."
            )

        for token in tokens:
            if token == "1":
                flat_vector.append(1)
            elif token in ("-1", "0"):
                flat_vector.append(-1)
            else:
                raise ValueError(
                    f"Invalid token '{token}' in {file_path}. Only 1, -1, or 0 are allowed."
                )

    expected_total = expected_rows * expected_cols
    if len(flat_vector) != expected_total:
        raise ValueError(
            f"Corrupted grid vector in {file_path}. Expected {expected_total} units, got {len(flat_vector)}."
        )

    return flat_vector


def load_dataset_dir(directory_path: str, expected_rows: int = 8, expected_cols: int = 5) -> dict:
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Directory not found: {directory_path}")

    dataset = {}
    filenames = sorted(os.listdir(directory_path))
    for fname in filenames:
        if fname.endswith(".txt"):
            full_path = os.path.join(directory_path, fname)
            label = os.path.splitext(fname)[0]
            vector = load_pattern_file(full_path, expected_rows, expected_cols)
            dataset[label] = vector

    if not dataset:
        raise ValueError(f"No pattern text files found inside {directory_path}")

    return dataset


class HopfieldNetwork:
    def __init__(self, size: int):
        self.size = size
        self.weights = [[0 for _ in range(size)] for _ in range(size)]
        self.patterns = {}

    def train(self, patterns: dict):
        if not patterns:
            raise ValueError("Training dataset cannot be empty.")

        self.patterns = patterns
        self.weights = [[0 for _ in range(self.size)] for _ in range(self.size)]

        for label, vector in patterns.items():
            if len(vector) != self.size:
                raise ValueError(
                    f"Pattern {label} length {len(vector)} does not match network size {self.size}."
                )
            for i in range(self.size):
                for j in range(self.size):
                    if i != j:
                        self.weights[i][j] += vector[i] * vector[j]

    def predict(self, input_vector: list, max_iterations: int = 20, asynchronous: bool = False) -> tuple:
        if len(input_vector) != self.size:
            raise ValueError(
                f"Input vector length {len(input_vector)} does not match network size {self.size}."
            )

        current_state = list(input_vector)
        energy_history = [self.compute_energy(current_state)]

        for iteration in range(1, max_iterations + 1):
            if asynchronous:
                next_state = list(current_state)
                for j in range(self.size):
                    activation = 0
                    for i in range(self.size):
                        activation += self.weights[i][j] * next_state[i]
                    next_state[j] = self._activation_rule(activation, next_state[j])
            else:
                next_state = [0 for _ in range(self.size)]
                for j in range(self.size):
                    activation = 0
                    for i in range(self.size):
                        activation += current_state[i] * self.weights[i][j]
                    next_state[j] = self._activation_rule(activation, current_state[j])

            current_energy = self.compute_energy(next_state)
            energy_history.append(current_energy)

            if next_state == current_state:
                return next_state, iteration - 1, energy_history

            current_state = next_state

        return current_state, max_iterations, energy_history

    def classify(self, vector: list) -> tuple:
        if not self.patterns:
            raise ValueError("Network has not been trained with labeled patterns.")

        best_label = None
        best_similarity = -float("inf")
        scores = {}

        for label, target_vector in self.patterns.items():
            sim = self._calculate_dot_product(vector, target_vector)
            scores[label] = sim
            if sim > best_similarity:
                best_similarity = sim
                best_label = label

        return best_label, best_similarity, scores

    def compute_energy(self, state: list) -> float:
        total = 0
        for i in range(self.size):
            for j in range(self.size):
                total += self.weights[i][j] * state[i] * state[j]
        return -0.5 * total

    def format_grid(self, vector: list, rows: int = 8, cols: int = 5) -> str:
        if len(vector) != rows * cols:
            raise ValueError("Vector size does not match specified grid dimensions.")

        lines = []
        for r in range(rows):
            row_tokens = []
            for c in range(cols):
                val = vector[r * cols + c]
                row_tokens.append("#" if val == 1 else ".")
            lines.append(" ".join(row_tokens))
        return "\n".join(lines)

    def _activation_rule(self, activation_sum: int, current_value: int) -> int:
        if activation_sum > 0:
            return 1
        if activation_sum < 0:
            return -1
        return current_value

    def _calculate_dot_product(self, v1: list, v2: list) -> int:
        total = 0
        for i in range(len(v1)):
            total += v1[i] * v2[i]
        return total
