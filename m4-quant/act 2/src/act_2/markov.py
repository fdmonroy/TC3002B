"""Markov chain engine for stochastic musical transitions and stationary distribution analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

class MarkovChain:
    """Discrete time Markov chain modeling transitions between discrete musical states."""

    def __init__(self, states: list[str], transition_matrix: np.ndarray | list[list[float]]):
        self.states = list(states)
        self.n_states = len(states)
        self.transition_matrix = np.array(transition_matrix, dtype=np.float64)
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self._validate()

    def _validate(self) -> None:
        """Verify dimensional, nonnegativity, and stochastic row constraints."""
        if self.transition_matrix.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"Matrix shape {self.transition_matrix.shape} does not match states count {self.n_states}"
            )
        if np.any(self.transition_matrix < 0.0):
            raise ValueError("All transition matrix entries must be nonnegative.")
        
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(f"Every row in transition matrix must sum to 1.0. Row sums: {row_sums}")

    def next_state(self, current_state: str, rng: np.random.Generator | None = None) -> str:
        """Sample the subsequent state using random number generator and transition probabilities."""
        if current_state not in self.state_to_idx:
            raise KeyError(f"State '{current_state}' not found in state space.")
        
        curr_idx = self.state_to_idx[current_state]
        probs = self.transition_matrix[curr_idx]
        
        if rng is not None:
            next_idx = rng.choice(self.n_states, p=probs)
        else:
            next_idx = np.random.choice(self.n_states, p=probs)
            
        return self.states[next_idx]

    def simulate(
        self,
        initial_state: str,
        n_steps: int,
        seed: int | None = None
    ) -> list[str]:
        """Perform an n step random walk across the Markov chain."""
        rng = np.random.default_rng(seed)
        sequence = [initial_state]
        curr = initial_state
        
        for _ in range(n_steps):
            curr = self.next_state(curr, rng=rng)
            sequence.append(curr)
            
        return sequence

    def compute_n_step_distribution(self, initial_vector: np.ndarray, steps: int) -> np.ndarray:
        """Calculate state probability vector after n transitions: pi^(n) = pi^(0) * P^n."""
        matrix_power = np.linalg.matrix_power(self.transition_matrix, steps)
        return np.dot(initial_vector, matrix_power)

    def compute_stationary_distribution(self) -> np.ndarray:
        """Solve for stationary distribution vector pi such that pi * P = pi and sum(pi) = 1."""
        # (P^T - I) * pi^T = 0 subject to sum(pi) = 1
        A = np.transpose(self.transition_matrix) - np.eye(self.n_states)
        A[-1, :] = 1.0  # Replace last equation with normalization condition
        b = np.zeros(self.n_states)
        b[-1] = 1.0
        
        stationary = np.linalg.lstsq(A, b, rcond=None)[0]
        # Clean any small numerical negatives
        stationary = np.clip(stationary, 0.0, 1.0)
        return stationary / np.sum(stationary)

    def to_dataframe(self) -> pd.DataFrame:
        """Represent the transition matrix as a labeled Pandas DataFrame."""
        return pd.DataFrame(
            self.transition_matrix,
            index=self.states,
            columns=self.states
        )
