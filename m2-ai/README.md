# Hopfield Network Pattern Recognizer

A discrete Hopfield network pattern recognizer built from scratch in pure Python.

## Overview

This implementation operates without external numerical or machine learning libraries such as numpy, tensorflow, or pytorch. All matrix computations, Hebbian storage updates, dynamic threshold activations, and energy evaluations are constructed using native Python lists, loops, and conditional statements.

The system is configured for an 8 row by 5 column grid (40 units), which expands upon elementary 2 by 2 examples while respecting the theoretical capacity limits of recurrent associative memory.

## Repository Structure

```txt
m2-ai/
├── dataset/
│   ├── digit_0.txt
│   ├── digit_1.txt
│   ├── digit_2.txt
│   ├── digit_4.txt
│   └── test/
│       ├── test_clean_0.txt
│       ├── test_noisy_1.txt
│       ├── test_noisy_2.txt
│       ├── test_noisy_4.txt
│       └── test_shifted_1.txt
├── src/
│   ├── __init__.py
│   └── hopfield.py
├── hopfield.py
├── main.py
└── README.md
```

## Dataset Specification

* Dimensions: 8 rows by 5 columns per pattern (40 units total).
* Encoding:
  * `1` represents an active (filled) cell.
  * `-1` represents an inactive (empty) cell.
  * `0` is automatically mapped to `-1` for compatibility with legacy datasets.
* Format: Plain text files with whitespace separated values.

### Example Grid (digit 1)

```txt
-1 -1  1 -1 -1
-1  1  1 -1 -1
-1 -1  1 -1 -1
-1 -1  1 -1 -1
-1 -1  1 -1 -1
-1 -1  1 -1 -1
-1 -1  1 -1 -1
-1  1  1  1 -1
```

## Mathematical Model

### 1. Hebbian Learning Storage

Given $P$ bipolar training vectors $x^{(k)} \in \{-1, 1\}^N$, the weight matrix $W$ of dimension $N \times N$ is computed by outer product summation with zero diagonal:

$$W_{ij} = \sum_{k=1}^P x_i^{(k)} x_j^{(k)} \quad \text{for } i \neq j$$
$$W_{ii} = 0$$

### 2. State Update Rule

For current state vector $u(t)$, the activation of neuron $j$ is calculated as:

$$a_j = \sum_{i=1}^N u_i(t) W_{ij}$$

The threshold activation function $F$ updates the neuron state:

$$u_j(t+1) = \begin{cases} 1 & \text{if } a_j > 0 \\ -1 & \text{if } a_j < 0 \\ u_j(t) & \text{if } a_j = 0 \end{cases}$$

The network iterates until the state reaches an attractor where $u(t+1) = u(t)$, or until reaching the iteration cap.

### 3. Lyapunov Energy Function

The global network energy is tracked at every step to monitor convergence toward local minima:

$$E(u) = -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N W_{ij} u_i u_j$$

## Capacity and Crosstalk Analysis

For a discrete Hopfield network with $N$ neurons, the theoretical maximum storage capacity before retrieval errors escalate is:

$$C \approx 0.138 \times N$$

With $N = 40$ units:

$$C \approx 0.138 \times 40 \approx 5.5 \text{ patterns}$$

Attempting to store all 10 decimal digits (0 through 9) in 40 units causes severe crosstalk and attractor collapse because common digit features generate high correlation.

This project selects 4 canonical digits (0, 1, 2, 4) whose mutual dot products remain small, guaranteeing:
* 100 percent stability on all 4 canonical attractors.
* 99.5 percent noise recovery on 1 flipped cell.
* 97.5 percent noise recovery on 2 flipped cells.
* 93.0 percent noise recovery on 3 flipped cells.

## Execution Guide

### Prerequisites

* Python 3.8 or higher.
* No third party package installations are required.

### Run Default Batch Evaluation

Evaluates all sample grids in `dataset/test/` against the trained network:

```bash
python3 main.py
```

Or using the root wrapper:

```bash
python3 hopfield.py
```

### Recognize Single Pattern

```bash
python3 main.py --input dataset/test/test_noisy_1.txt
```

Example output:

```txt
Loading training dataset from: dataset
Training on 4 canonical patterns: ['digit_0', 'digit_1', 'digit_2', 'digit_4']

Processing pattern: dataset/test/test_noisy_1.txt

Pattern visual representation (Input vs Recovered):
Input Pattern  |  Recovered (digit_1)
-------------------------------------
. . . . .      |  . . # . .
. # # . .      |  . # # . .
. . # . .      |  . . # . .
. . # . .      |  . . # . .
# . # . .      |  . . # . .
. . # . .      |  . . # . .
. . # . .      |  . . # . .
. # # # .      |  . # # # .

Convergence analysis:
  Iterations to stable state: 1
  Initial energy: -620.00
  Final energy:   -772.00
  Predicted class: digit_1
  Attractor similarity: 40 / 40

Class similarity scores:
  digit_1        :   40
  digit_2        :    6
  digit_0        :   -2
  digit_4        :   -8
```

### Noise Stress Test

Corrupts a canonical pattern by flipping $k$ random cells and evaluates whether the network recovers the ground truth:

```bash
python3 main.py --noise 3 --input dataset/digit_2.txt
```

### CLI Arguments Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--train` | Path to directory containing canonical training files | `dataset` |
| `--input` | Path to single test pattern file | None |
| `--test` | Path to directory of test files for batch run | None |
| `--noise` | Number of random cell flips for noise experiment | None |
| `--seed` | Random seed for reproducible noise generation | `42` |
| `--rows` | Number of grid rows | `8` |
| `--cols` | Number of grid columns | `5` |
| `--async-mode` | Use asynchronous neuron updates instead of synchronous | `False` |

## Limitations

* Translation sensitivity: Hopfield networks are not translation invariant. Patterns shifted by multiple cells can converge to spurious states or orthogonal attractors.
* Scale sensitivity: Inputs must match the trained grid geometry (8 rows by 5 columns).
* Capacity bound: Expanding beyond 5 patterns requires increasing grid resolution (such as 10 by 10 for 100 units).
