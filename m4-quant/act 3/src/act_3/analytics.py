"""Closed form analytical formulas for M/M/1 and M/M/c Erlang C queuing systems."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MM1TheoreticalMetrics:
    """Steady state analytical performance measures for an M/M/1 queue."""

    arrival_rate: float
    service_rate: float
    rho: float
    p0: float
    l_system: float
    l_queue: float
    w_system_hours: float
    w_queue_hours: float
    w_system_mins: float
    w_queue_mins: float


def compute_mm1_theoretical(arrival_rate: float, service_rate: float) -> MM1TheoreticalMetrics:
    """Compute exact steady state metrics for an M/M/1 queue given hourly rates."""
    if arrival_rate <= 0.0:
        raise ValueError("Arrival rate lambda must be strictly positive.")
    if service_rate <= 0.0:
        raise ValueError("Service rate mu must be strictly positive.")
    if arrival_rate >= service_rate:
        raise ValueError(
            f"Unstable M/M/1 queue: arrival_rate ({arrival_rate}) must be strictly less than service_rate ({service_rate})."
        )

    rho = arrival_rate / service_rate
    p0 = 1.0 - rho
    l_system = rho / (1.0 - rho)
    l_queue = (rho**2) / (1.0 - rho)
    w_system_hours = 1.0 / (service_rate - arrival_rate)
    w_queue_hours = arrival_rate / (service_rate * (service_rate - arrival_rate))

    return MM1TheoreticalMetrics(
        arrival_rate=float(arrival_rate),
        service_rate=float(service_rate),
        rho=float(rho),
        p0=float(p0),
        l_system=float(l_system),
        l_queue=float(l_queue),
        w_system_hours=float(w_system_hours),
        w_queue_hours=float(w_queue_hours),
        w_system_mins=float(w_system_hours * 60.0),
        w_queue_mins=float(w_queue_hours * 60.0),
    )


def compute_mmc_theoretical(
    arrival_rate: float,
    service_rate: float,
    n_servers: int = 1,
) -> dict[str, float]:
    """Compute steady state analytical metrics for an M/M/c queue using Erlang C."""
    if arrival_rate <= 0.0 or service_rate <= 0.0:
        raise ValueError("Rates lambda and mu must be strictly positive.")
    if n_servers < 1:
        raise ValueError("Number of servers c must be at least 1.")
    if arrival_rate >= n_servers * service_rate:
        raise ValueError(
            f"Unstable M/M/{n_servers} queue: arrival_rate ({arrival_rate}) >= c * service_rate ({n_servers * service_rate})."
        )

    a = arrival_rate / service_rate
    rho_per_server = arrival_rate / (n_servers * service_rate)

    sum_terms = sum((a**k) / math.factorial(k) for k in range(n_servers))
    last_term = ((a**n_servers) / math.factorial(n_servers)) * (1.0 / (1.0 - rho_per_server))
    p0 = 1.0 / (sum_terms + last_term)

    prob_wait_erlang_c = last_term * p0
    l_queue = prob_wait_erlang_c * rho_per_server / (1.0 - rho_per_server)
    w_queue_hours = l_queue / arrival_rate
    w_system_hours = w_queue_hours + (1.0 / service_rate)
    l_system = arrival_rate * w_system_hours

    return {
        "arrival_rate": float(arrival_rate),
        "service_rate": float(service_rate),
        "n_servers": int(n_servers),
        "rho_per_server": float(rho_per_server),
        "p0": float(p0),
        "prob_wait": float(prob_wait_erlang_c),
        "l_system": float(l_system),
        "l_queue": float(l_queue),
        "w_system_hours": float(w_system_hours),
        "w_queue_hours": float(w_queue_hours),
        "w_system_mins": float(w_system_hours * 60.0),
        "w_queue_mins": float(w_queue_hours * 60.0),
    }


def compute_mm1_state_probabilities(rho: float, max_k: int = 15) -> np.ndarray:
    """Compute geometric steady state occupancy probabilities P(N=k) = (1 - rho) * rho^k."""
    if not (0.0 <= rho < 1.0):
        raise ValueError("Traffic intensity rho must satisfy 0 <= rho < 1.")
    k_vals = np.arange(max_k + 1, dtype=np.float64)
    return (1.0 - rho) * np.power(rho, k_vals)


def build_rho_sweep_dataframe(service_rate: float, rho_values: list[float]) -> pd.DataFrame:
    """Build table of M/M/1 congestion metrics across a sequence of utilization levels."""
    rows = []
    for rho in rho_values:
        lam = rho * service_rate
        metrics = compute_mm1_theoretical(lam, service_rate)
        rows.append(
            {
                "Utilización ρ": metrics.rho,
                "Tasa llegada λ (/h)": metrics.arrival_rate,
                "Tasa servicio μ (/h)": metrics.service_rate,
                "Clientes en fila Lq": metrics.l_queue,
                "Clientes en sistema L": metrics.l_system,
                "Espera en fila Wq (min)": metrics.w_queue_mins,
                "W (min)": metrics.w_system_mins,
            }
        )
    return pd.DataFrame(rows)
