"""Plotting and visualization utilities for transition matrices and musical trajectories."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from act_2.markov import MarkovChain

def plot_transition_heatmap(
    chain: MarkovChain,
    title: str = "Matriz de Transición de la Cadena de Markov",
    save_path: str | Path | None = None
) -> plt.Figure:
    """Plot an annotated heatmap of the transition matrix probabilities."""
    df = chain.to_dataframe()
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Probabilidad de transición"},
        linewidths=0.8,
        linecolor="#CBD5E0",
        ax=ax
    )
    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Estado siguiente (t + 1)", fontsize=11, labelpad=8)
    ax.set_ylabel("Estado actual (t)", fontsize=11, labelpad=8)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300)
    return fig

def plot_stationary_distribution(
    chain: MarkovChain,
    title: str = "Distribución Estacionaria de Estados",
    save_path: str | Path | None = None
) -> plt.Figure:
    """Plot bar chart showing the long term asymptotic state occupancy probabilities."""
    pi = chain.compute_stationary_distribution()
    states = chain.states
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(states, pi, color="#3182CE", edgecolor="#2B6CB0", width=0.55)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )
        
    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Estado musical", fontsize=11, labelpad=8)
    ax.set_ylabel("Probabilidad estacionaria", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(pi) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300)
    return fig

def plot_melody_roll(
    sequence: list[str],
    durations: list[float] | None = None,
    title: str = "Trayectoria Melódica (Piano Roll)",
    save_path: str | Path | None = None
) -> plt.Figure:
    """Plot a step by step piano roll representation of the generated melody sequence."""
    unique_notes = sorted(list(set(sequence)), key=lambda n: sequence.index(n))
    note_to_y = {note: idx for idx, note in enumerate(unique_notes)}
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    if durations is None:
        durations = [1.0] * len(sequence)
        
    times = [0.0]
    for d in durations[:-1]:
        times.append(times[-1] + d)
        
    y_vals = [note_to_y[n] for n in sequence]
    
    ax.step(times, y_vals, where="post", color="#4299E1", linewidth=1.5, alpha=0.8)
    ax.scatter(times, y_vals, color="#2B6CB0", s=45, zorder=3)
    
    for t, y, note in zip(times, y_vals, sequence):
        ax.annotate(
            note,
            (t, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            fontweight="bold"
        )
        
    ax.set_yticks(range(len(unique_notes)))
    ax.set_yticklabels(unique_notes, fontweight="bold")
    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Tiempo transcurrido (segundos)", fontsize=11, labelpad=8)
    ax.set_ylabel("Nota musical", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300)
    return fig
