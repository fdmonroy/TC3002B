import argparse
import os
import random
import sys

from src.hopfield import HopfieldNetwork, load_dataset_dir, load_pattern_file


def render_side_by_side(v1: list, v2: list, label1: str, label2: str, rows: int = 8, cols: int = 5) -> str:
    lines = []
    header = f"{label1:<{cols * 2 + 4}} |  {label2}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in range(rows):
        row1_str = " ".join("#" if v1[r * cols + c] == 1 else "." for c in range(cols))
        row2_str = " ".join("#" if v2[r * cols + c] == 1 else "." for c in range(cols))
        lines.append(f"{row1_str:<{cols * 2 + 4}} |  {row2_str}")

    return "\n".join(lines)


def inject_noise(vector: list, num_flips: int, seed: int = 42) -> list:
    noisy = list(vector)
    rng = random.Random(seed)
    chosen_indices = rng.sample(range(len(vector)), num_flips)
    for idx in chosen_indices:
        noisy[idx] *= -1
    return noisy


def run_single(network: HopfieldNetwork, input_path: str, rows: int, cols: int, async_mode: bool):
    print(f"\nProcessing pattern: {input_path}")
    raw_vector = load_pattern_file(input_path, rows, cols)
    converged_vector, iterations, energies = network.predict(
        raw_vector, asynchronous=async_mode
    )
    predicted_label, top_score, all_scores = network.classify(converged_vector)

    print("\nPattern visual representation (Input vs Recovered):")
    print(render_side_by_side(raw_vector, converged_vector, "Input Pattern", f"Recovered ({predicted_label})", rows, cols))

    print("\nConvergence analysis:")
    print(f"  Iterations to stable state: {iterations}")
    print(f"  Initial energy: {energies[0]:.2f}")
    print(f"  Final energy:   {energies[-1]:.2f}")
    print(f"  Predicted class: {predicted_label}")
    print(f"  Attractor similarity: {top_score} / {rows * cols}")

    print("\nClass similarity scores:")
    for label, score in sorted(all_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  {label:<15}: {score:>4}")


def run_batch(network: HopfieldNetwork, test_dir: str, rows: int, cols: int, async_mode: bool):
    print(f"\nBatch evaluation across directory: {test_dir}")
    test_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".txt"))
    if not test_files:
        print("No test text files discovered.")
        return

    print(f"{'Filename':<22} | {'Prediction':<12} | {'Iterations':<10} | {'Score':<6}")
    print("-" * 58)

    for fname in test_files:
        fpath = os.path.join(test_dir, fname)
        vector = load_pattern_file(fpath, rows, cols)
        converged, iterations, _ = network.predict(vector, asynchronous=async_mode)
        label, score, _ = network.classify(converged)
        print(f"{fname:<22} | {label:<12} | {iterations:<10} | {score:>3}/{rows * cols}")


def run_noise_test(network: HopfieldNetwork, input_path: str, flips: int, seed: int, rows: int, cols: int, async_mode: bool):
    print(f"\nNoise stress experiment on {input_path} (Flipping {flips} cells, seed={seed}):")
    clean_vector = load_pattern_file(input_path, rows, cols)
    ground_truth, _, _ = network.classify(clean_vector)

    noisy_vector = inject_noise(clean_vector, flips, seed)
    converged_vector, iterations, energies = network.predict(
        noisy_vector, asynchronous=async_mode
    )
    predicted_label, top_score, _ = network.classify(converged_vector)

    print("\nPattern visual representation (Corrupted vs Recovered):")
    print(render_side_by_side(noisy_vector, converged_vector, f"Corrupted ({flips} flips)", f"Recovered ({predicted_label})", rows, cols))

    is_recovered = (converged_vector == clean_vector)
    status = "SUCCESS" if is_recovered else "FAILED (Trapped in spurious attractor)"

    print("\nExperiment outcome:")
    print(f"  Original class:  {ground_truth}")
    print(f"  Recovered class: {predicted_label}")
    print(f"  Iterations:      {iterations}")
    print(f"  Match status:    {status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hopfield Network Pattern Recognizer from scratch."
    )
    parser.add_argument(
        "--train",
        type=str,
        default="dataset",
        help="Path to directory containing canonical training text files."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to single pattern text file for recognition."
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Path to directory of test files for batch recognition."
    )
    parser.add_argument(
        "--noise",
        type=int,
        help="Number of random bits to flip for noise tolerance test."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible noise generation."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=8,
        help="Grid rows (default: 8)."
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Grid columns (default: 5)."
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Use asynchronous neuron updates instead of synchronous updates."
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    total_units = args.rows * args.cols
    network = HopfieldNetwork(total_units)

    print(f"Loading training dataset from: {args.train}")
    training_patterns = load_dataset_dir(args.train, args.rows, args.cols)
    print(f"Training on {len(training_patterns)} canonical patterns: {list(training_patterns.keys())}")
    network.train(training_patterns)

    if args.noise is not None:
        target_file = args.input or os.path.join(args.train, f"{list(training_patterns.keys())[0]}.txt")
        run_noise_test(network, target_file, args.noise, args.seed, args.rows, args.cols, args.async_mode)
    elif args.input:
        run_single(network, args.input, args.rows, args.cols, args.async_mode)
    elif args.test:
        run_batch(network, args.test, args.rows, args.cols, args.async_mode)
    else:
        default_test_dir = os.path.join(args.train, "test")
        if os.path.isdir(default_test_dir):
            run_batch(network, default_test_dir, args.rows, args.cols, args.async_mode)
        else:
            print("No action specified. Run with --help for command line usage.")


if __name__ == "__main__":
    main()
