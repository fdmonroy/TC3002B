"""Analytical formulas for M/M/1 and M/M/c queues."""

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MM1TheoreticalMetrics:
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
    interarrival_mean_mins: float
    service_mean_mins: float
    interarrival_median_mins: float
    service_median_mins: float


def compute_mm1_theoretical(
    arrival_rate: float,
    service_rate: float,
) -> MM1TheoreticalMetrics:
    if arrival_rate <= 0.0:
        raise ValueError("arrival_rate must be > 0.")
    if service_rate <= 0.0:
        raise ValueError("service_rate must be > 0.")
    if arrival_rate >= service_rate:
        raise ValueError("Unstable queue: arrival_rate must be < service_rate.")

    rho = arrival_rate / service_rate
    p0 = 1.0 - rho
    l_system = rho / (1.0 - rho)
    l_queue = (rho**2) / (1.0 - rho)
    w_system_hours = 1.0 / (service_rate - arrival_rate)
    w_queue_hours = arrival_rate / (service_rate * (service_rate - arrival_rate))

    interarrival_mean_mins = 60.0 / arrival_rate
    service_mean_mins = 60.0 / service_rate

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
        interarrival_mean_mins=float(interarrival_mean_mins),
        service_mean_mins=float(service_mean_mins),
        interarrival_median_mins=float(interarrival_mean_mins * math.log(2.0)),
        service_median_mins=float(service_mean_mins * math.log(2.0)),
    )


def compute_mmc_theoretical(
    arrival_rate: float,
    service_rate: float,
    n_servers: int = 1,
) -> dict[str, float]:
    if arrival_rate <= 0.0 or service_rate <= 0.0:
        raise ValueError("Rates must be > 0.")
    if n_servers < 1:
        raise ValueError("n_servers must be >= 1.")

    a = arrival_rate / service_rate
    rho = arrival_rate / (n_servers * service_rate)
    if rho >= 1.0:
        raise ValueError("Unstable queue: rho must be < 1.")

    sum_terms = sum((a**k) / math.factorial(k) for k in range(n_servers))
    last_term = ((a**n_servers) / math.factorial(n_servers)) * (1.0 / (1.0 - rho))
    p0 = 1.0 / (sum_terms + last_term)

    prob_wait = last_term * p0
    l_queue = prob_wait * rho / (1.0 - rho)
    w_queue_hours = l_queue / arrival_rate
    w_system_hours = w_queue_hours + (1.0 / service_rate)
    l_system = arrival_rate * w_system_hours

    return {
        "arrival_rate": float(arrival_rate),
        "service_rate": float(service_rate),
        "n_servers": float(n_servers),
        "offered_load": float(a),
        "rho_per_server": float(rho),
        "p0": float(p0),
        "prob_wait": float(prob_wait),
        "l_queue": float(l_queue),
        "l_system": float(l_system),
        "w_queue_hours": float(w_queue_hours),
        "w_system_hours": float(w_system_hours),
        "w_queue_mins": float(w_queue_hours * 60.0),
        "w_system_mins": float(w_system_hours * 60.0),
    }


def compute_mm1_state_probabilities(
    rho: float,
    max_k: int = 15,
) -> np.ndarray:
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0, 1).")
    k_vals = np.arange(max_k + 1, dtype=np.float64)
    return (1.0 - rho) * np.power(rho, k_vals)


def build_rho_sweep_dataframe(
    service_rate: float,
    rho_values: list[float],
) -> pd.DataFrame:
    rows = []
    for rho in rho_values:
        lam = rho * service_rate
        m = compute_mm1_theoretical(lam, service_rate)
        rows.append(
            {
                "rho": m.rho,
                "arrival_rate": m.arrival_rate,
                "service_rate": m.service_rate,
                "l_queue": m.l_queue,
                "l_system": m.l_system,
                "w_queue_mins": m.w_queue_mins,
                "w_system_mins": m.w_system_mins,
            }
        )
    return pd.DataFrame(rows)
