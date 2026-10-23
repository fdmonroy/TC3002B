"""Publication quality visualization utilities for M/M/1 and M/M/c queue simulations."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from act_3.analytics import compute_mm1_state_probabilities, compute_mm1_theoretical
from act_3.simulation import SimulationResult


def plot_input_distributions(
    result: SimulationResult,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot empirical interarrival and service time histograms with theoretical exponential PDFs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    interarrivals = result.interarrival_samples_mins
    services = result.service_samples_mins
    lam_min = result.arrival_rate / 60.0
    mu_min = result.service_rate / 60.0

    panels = [
        (
            axes[0],
            interarrivals,
            lam_min,
            "#2B6CB0",
            "Tiempo entre llegadas (Exponencial, media 12.0 min)",
            "Tiempo entre llegadas (minutos)",
        ),
        (
            axes[1],
            services,
            mu_min,
            "#2F855A",
            "Tiempo de servicio (Exponencial, media 8.0 min)",
            "Duración de atención (minutos)",
        ),
    ]

    for ax, samples, rate_min, color, title, xlabel in panels:
        upper_x = float(np.percentile(samples, 99))
        clipped = samples[samples <= upper_x]

        sns.histplot(
            clipped,
            stat="density",
            bins=32,
            ax=ax,
            color=color,
            alpha=0.32,
            edgecolor="#FFFFFF",
            linewidth=0.5,
            label="Densidad empírica",
        )

        x_grid = np.linspace(0.0, upper_x, 250)
        pdf_vals = rate_min * np.exp(-rate_min * x_grid)
        ax.plot(
            x_grid,
            pdf_vals,
            color=color,
            lw=2.0,
            label="PDF teórica exponencial",
        )

        mean_val = float(np.mean(samples))
        median_val = float(np.median(samples))

        ax.axvline(
            mean_val,
            color="#E53E3E",
            linestyle="--",
            lw=1.5,
            label=f"Media: {mean_val:.2f} min",
        )
        ax.axvline(
            median_val,
            color="#1A202C",
            linestyle=":",
            lw=1.5,
            label=f"Mediana: {median_val:.2f} min",
        )

        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylabel("Densidad de probabilidad", fontsize=9.5)
        ax.set_xlim(0.0, upper_x)
        ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)
        sns.despine(ax=ax)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_temporal_dynamics_and_occupancy(
    result: SimulationResult,
    window_mins: float = 360.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot discrete event sample path N(t) and stationary geometric state distribution P_k."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    tl = result.timeline_df
    tl_win = tl[tl["time_mins"] <= window_mins].copy()

    ax0 = axes[0]
    ax0.step(
        tl_win["time_mins"],
        tl_win["n_system"],
        where="post",
        color="#2B6CB0",
        lw=1.5,
        label="Clientes en sistema N(t)",
    )
    ax0.fill_between(
        tl_win["time_mins"],
        tl_win["n_queue"],
        step="post",
        color="#63B3ED",
        alpha=0.35,
        label="Clientes en fila Q(t)",
    )

    theo_l = result.arrival_rate / (result.service_rate - result.arrival_rate)
    ax0.axhline(
        theo_l,
        color="#E53E3E",
        linestyle="--",
        lw=1.5,
        label=f"Media estacionaria L = {theo_l:.2f}",
    )

    ax0.set_title(
        f"Trayectoria por eventos discretos (Primeros {int(window_mins)} min)",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax0.set_xlabel("Tiempo de simulación (minutos)", fontsize=9.5)
    ax0.set_ylabel("Número de clientes", fontsize=9.5)
    ax0.set_xlim(0.0, window_mins)
    ax0.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)
    sns.despine(ax=ax0)

    ax1 = axes[1]
    max_k = 10
    rho = result.arrival_rate / result.service_rate
    theo_probs = compute_mm1_state_probabilities(rho, max_k=max_k)
    emp_probs = np.array(
        [result.state_occupancy.get(k, 0.0) for k in range(max_k + 1)],
        dtype=np.float64,
    )

    k_idx = np.arange(max_k + 1)
    width = 0.38
    ax1.bar(
        k_idx - width / 2,
        theo_probs,
        width=width,
        color="#2B6CB0",
        alpha=0.85,
        label="Teórica geométrica (1 - ρ)ρ^k",
    )
    ax1.bar(
        k_idx + width / 2,
        emp_probs,
        width=width,
        color="#38A169",
        alpha=0.85,
        label="Empírica SimPy",
    )

    ax1.set_title(
        "Distribución estacionaria de estados P(N = k)",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax1.set_xlabel("Estado k (clientes en el sistema)", fontsize=9.5)
    ax1.set_ylabel("Probabilidad estacionaria", fontsize=9.5)
    ax1.set_xticks(k_idx)
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)
    sns.despine(ax=ax1)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_congestion_explosion(
    df_sweep: pd.DataFrame,
    service_rate: float = 7.5,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the non linear explosion of system time W as utilization rho approaches 1."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax0 = axes[0]
    labels = [
        f"ρ={r:.2f}\n(λ={lam:g}/h)"
        for r, lam in zip(df_sweep["Utilización ρ"], df_sweep["Tasa λ (/h)"])
    ]
    x_pos = np.arange(len(labels))
    w_theo = df_sweep["W teórico (min)"].values
    w_sim = df_sweep["W SimPy (min)"].values

    bars = ax0.bar(
        x_pos - 0.18,
        w_theo,
        width=0.35,
        color="#2B6CB0",
        label="W teórico (min)",
    )
    ax0.bar(
        x_pos + 0.18,
        w_sim,
        width=0.35,
        color="#DD6B20",
        alpha=0.88,
        label="W SimPy (min)",
    )

    for bar in bars:
        h = bar.get_height()
        ax0.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax0.set_title(
        "Escenarios de carga con μ = 7.5 por hora fija",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax0.set_xlabel("Intensidad de tráfico ρ y tasa de llegada λ", fontsize=9.5)
    ax0.set_ylabel("Tiempo total en sistema W (minutos)", fontsize=9.5)
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(labels, fontsize=8.5)
    ax0.set_ylim(0, max(w_theo) * 1.18)
    ax0.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)
    sns.despine(ax=ax0)

    ax1 = axes[1]
    rho_grid = np.linspace(0.10, 0.96, 200)
    w_curve = [compute_mm1_theoretical(r * service_rate, service_rate).w_system_mins for r in rho_grid]
    wq_curve = [compute_mm1_theoretical(r * service_rate, service_rate).w_queue_mins for r in rho_grid]

    ax1.plot(rho_grid, w_curve, color="#2B6CB0", lw=2.0, label="W(ρ) total teórico")
    ax1.plot(rho_grid, wq_curve, color="#319795", lw=1.8, linestyle="--", label="Wq(ρ) en fila teórico")
    ax1.scatter(
        df_sweep["Utilización ρ"],
        df_sweep["W SimPy (min)"],
        color="#DD6B20",
        s=45,
        zorder=4,
        label="Muestras SimPy W",
    )
    ax1.axvline(1.0, color="#E53E3E", linestyle=":", lw=1.5, label="Asíntota inestable (ρ = 1)")

    ax1.set_title(
        "Divergencia hiperbólica de espera cuando ρ se acerca a 1",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax1.set_xlabel("Factor de utilización del servidor (ρ = λ / μ)", fontsize=9.5)
    ax1.set_ylabel("Tiempo promedio (minutos)", fontsize=9.5)
    ax1.set_xlim(0.10, 1.02)
    ax1.set_ylim(0, 210)
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)
    sns.despine(ax=ax1)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig


def plot_capacity_comparison(
    df_capacity: pd.DataFrame,
    wait_m1: np.ndarray,
    wait_m2: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot operational comparison between M/M/1 (1 server) and M/M/2 (2 servers)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax0 = axes[0]
    configs = df_capacity["Configuración"].tolist()
    x_idx = np.arange(len(configs))
    wq_vals = df_capacity["Espera en fila Wq (min)"].values
    util_vals = df_capacity["Utilización por servidor (%)"].values

    b1 = ax0.bar(
        x_idx - 0.18,
        wq_vals,
        width=0.34,
        color="#C05621",
        label="Espera en fila Wq (min)",
    )
    b2 = ax0.bar(
        x_idx + 0.18,
        util_vals,
        width=0.34,
        color="#2B6CB0",
        label="Utilización por caja (%)",
    )

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax0.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax0.set_title(
        "Compromiso operativo: espera vs utilización de cajas",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax0.set_xlabel("Configuración del sistema (λ = 5.0/h, μ = 7.5/h)", fontsize=9.5)
    ax0.set_ylabel("Magnitud (minutos / porcentaje)", fontsize=9.5)
    ax0.set_xticks(x_idx)
    ax0.set_xticklabels(configs, fontsize=9.5)
    ax0.set_ylim(0, max(max(wq_vals), max(util_vals)) * 1.22)
    ax0.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)
    sns.despine(ax=ax0)

    ax1 = axes[1]
    upper_w = float(np.percentile(wait_m1, 96))
    w1_clip = wait_m1[wait_m1 <= upper_w]
    w2_clip = wait_m2[wait_m2 <= upper_w]

    sns.histplot(
        w1_clip,
        stat="density",
        bins=28,
        ax=ax1,
        color="#C05621",
        alpha=0.30,
        edgecolor="#FFFFFF",
        label="Distribución Wq en M/M/1",
    )
    sns.histplot(
        w2_clip,
        stat="density",
        bins=28,
        ax=ax1,
        color="#2B6CB0",
        alpha=0.35,
        edgecolor="#FFFFFF",
        label="Distribución Wq en M/M/2",
    )

    m1_mean = float(np.mean(wait_m1))
    m1_med = float(np.median(wait_m1))
    m2_mean = float(np.mean(wait_m2))
    m2_med = float(np.median(wait_m2))

    ax1.axvline(
        m1_mean,
        color="#C05621",
        linestyle="--",
        lw=1.6,
        label=f"Media M/M/1: {m1_mean:.2f} min",
    )
    ax1.axvline(
        m1_med,
        color="#7B341E",
        linestyle=":",
        lw=1.5,
        label=f"Mediana M/M/1: {m1_med:.2f} min",
    )
    ax1.axvline(
        m2_mean,
        color="#2B6CB0",
        linestyle="--",
        lw=1.6,
        label=f"Media M/M/2: {m2_mean:.2f} min",
    )

    ax1.set_title(
        f"Distribución de tiempos de espera (Mediana M/M/2: {m2_med:.2f} min)",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax1.set_xlabel("Tiempo de espera en fila Wq (minutos)", fontsize=9.5)
    ax1.set_ylabel("Densidad empírica", fontsize=9.5)
    ax1.set_xlim(0.0, upper_w)
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.0)
    sns.despine(ax=ax1)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    return fig
