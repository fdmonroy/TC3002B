import sys


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


def run_hopfield(patterns: list, test_pattern: list, max_iterations: int = 50) -> list:
    size = len(patterns[0])
    weight_sum = [[0.0 for _ in range(size)] for _ in range(size)]

    for pattern_index, pattern in enumerate(patterns):
        outer = outer_product(pattern, pattern)
        print(f"\nMatriz X{pattern_index + 1}^T * X{pattern_index + 1}:")
        for row in outer:
            print([int(x) if x.is_integer() else x for x in row])
        weight_sum = add_matrices(weight_sum, outer)

    weights = zero_diagonal(weight_sum)
    print("\nMatriz con diagonal cero (T):")
    for row in weights:
        print([int(x) if x.is_integer() else x for x in row])

    current_state = [float(x) for x in test_pattern]
    iteration = 0
    converged = False

    print(f"\nPatron inicial U(0): {current_state}")

    while iteration < max_iterations:
        iteration += 1
        next_state = update_vector(current_state, weights)
        print(f"U({iteration}) = {next_state}")

        if next_state == current_state:
            converged = True
            print(f"\nConvergencia alcanzada en iteracion {iteration}.")
            print(f"Patron recuperado mas cercano: {next_state}")
            return next_state

        current_state = next_state

    if not converged:
        print(f"\nNo se encontro convergencia tras {max_iterations} iteraciones.")
    return current_state


def to_bipolar(matrix_rows: list) -> list:
    vector = []
    for row in matrix_rows:
        for val in row:
            vector.append(-1.0 if val == 0 else 1.0)
    return vector


def run_class_demo():
    print("=" * 60)
    print("RED HOPFIELD - DEMOSTRACION EJEMPLO DE CLASE")
    print("=" * 60)

    x1_matrix = [
        [1.0, 1.0],
        [1.0, 0.0]
    ]
    x2_matrix = [
        [0.0, 0.0],
        [0.0, 1.0]
    ]

    p1 = to_bipolar(x1_matrix)
    p2 = to_bipolar(x2_matrix)

    print(f"Patron X1 (bipolar): {p1}")
    print(f"Patron X2 (bipolar): {p2}")

    print("\n" + "-" * 50)
    print("PRUEBA 1 (Slide 12): Patron A = (1, 1, 1, -1)")
    print("-" * 50)
    run_hopfield([p1, p2], [1.0, 1.0, 1.0, -1.0])

    print("\n" + "-" * 50)
    print("PRUEBA 2 (Slide 13): Patron A = (-1, -1, -1, -1)")
    print("-" * 50)
    run_hopfield([p1, p2], [-1.0, -1.0, -1.0, -1.0])


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--demo", "-d"):
        run_class_demo()
        return

    try:
        dim_prompt = "Introduce la dimension de las matrices N (o presiona Enter para ejemplo de clase): "
        dim_input = input(dim_prompt).strip()
    except (EOFError, KeyboardInterrupt):
        dim_input = ""

    if not dim_input:
        run_class_demo()
        return

    try:
        n = int(dim_input)
    except ValueError:
        print("Dimension no numerica. Ejecutando ejemplo de clase...")
        run_class_demo()
        return

    x1_rows = []
    print(f"Introduce las {n} filas de x1:")
    for i in range(n):
        row = [float(x) for x in input(f"Fila {i + 1}: ").split()]
        x1_rows.append(row)

    x2_rows = []
    print(f"Introduce las {n} filas de x2:")
    for i in range(n):
        row = [float(x) for x in input(f"Fila {i + 1}: ").split()]
        x2_rows.append(row)

    m1 = to_bipolar(x1_rows)
    m2 = to_bipolar(x2_rows)

    try:
        eval_input = input("\nIntroduce el patron a evaluar: ").strip()
    except (EOFError, KeyboardInterrupt):
        eval_input = ""

    if not eval_input:
        test_pattern = list(m1)
    else:
        test_pattern = [float(x) for x in eval_input.split()]

    run_hopfield([m1, m2], test_pattern)


if __name__ == "__main__":
    main()
