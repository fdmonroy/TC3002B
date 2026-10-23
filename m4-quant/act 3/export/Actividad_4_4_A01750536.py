"""Actividad 4.4: Simulación M/M/1 con eventos discretos en SimPy.

Alumno: Fernando Daniel Monroy Sánchez (A01750536)
Profesor: Luis Ricardo Peña Llamas
Curso: Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 570)
"""

from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import simpy


@dataclass(frozen=True)
class MM1TheoreticalMetrics:
    arrival_rate: float
    service_rate: float
    rho: float
    p0: float
    l_system: float
    l_queue: float
    w_system_mins: float
    w_queue_mins: float
    interarrival_mean_mins: float
    service_mean_mins: float
    interarrival_median_mins: float
    service_median_mins: float


@dataclass(frozen=True)
class CustomerRecord:
    customer_id: int
    arrival_time: float
    service_start_time: float
    departure_time: float
    wait_time: float
    service_duration: float
    system_time: float
    queue_length_on_arrival: int


@dataclass
class SimulationResult:
    arrival_rate: float
    service_rate: float
    n_servers: int
    sim_time: float
    warmup_time: float
    customers_served: int
    rho_empirical: float
    l_system_empirical: float
    l_queue_empirical: float
    w_system_mins_empirical: float
    w_queue_mins_empirical: float
    interarrival_samples_mins: np.ndarray
    service_samples_mins: np.ndarray
    records_df: pd.DataFrame
    timeline_df: pd.DataFrame
    state_occupancy: dict[int, float]


def compute_mm1_theoretical(
    arrival_rate: float,
    service_rate: float,
) -> MM1TheoreticalMetrics:
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
    a = arrival_rate / service_rate
    rho = arrival_rate / (n_servers * service_rate)
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
        "rho_per_server": float(rho),
        "p0": float(p0),
        "prob_wait": float(prob_wait),
        "l_queue": float(l_queue),
        "l_system": float(l_system),
        "w_queue_mins": float(w_queue_hours * 60.0),
        "w_system_mins": float(w_system_hours * 60.0),
    }


def compute_mm1_state_probabilities(
    rho: float,
    max_k: int = 10,
) -> np.ndarray:
    k_vals = np.arange(max_k + 1, dtype=np.float64)
    return (1.0 - rho) * np.power(rho, k_vals)


def inverse_transform_exponential(
    rate: float,
    rng: np.random.Generator,
) -> float:
    u = float(rng.uniform(0.0, 1.0))
    return -math.log(1.0 - u) / rate


def run_queue_simulation(
    arrival_rate: float,
    service_rate: float,
    sim_time: float = 180000.0,
    n_servers: int = 1,
    warmup_time: float = 3000.0,
    seed: int = 2026,
    max_timeline_points: int = 2000,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    lambda_per_min = arrival_rate / 60.0
    mu_per_min = service_rate / 60.0

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=n_servers)

    records: list[CustomerRecord] = []
    interarrival_samples: list[float] = []
    service_samples: list[float] = []

    state = {
        "n_system": 0,
        "last_event_time": 0.0,
        "area_system": 0.0,
        "area_queue": 0.0,
        "area_busy": 0.0,
    }
    state_time_by_k: dict[int, float] = {k: 0.0 for k in range(16)}
    timeline_events: list[dict[str, float | int | str]] = [
        {"time_mins": 0.0, "n_system": 0, "n_queue": 0, "event": "init"}
    ]

    def _accumulate_areas(now: float) -> None:
        prev_t = state["last_event_time"]
        if now > prev_t:
            start_eff = max(prev_t, warmup_time)
            if now > start_eff:
                dt = now - start_eff
                n_sys = state["n_system"]
                n_q = max(0, n_sys - n_servers)
                n_busy = min(n_sys, n_servers)
                state["area_system"] += n_sys * dt
                state["area_queue"] += n_q * dt
                state["area_busy"] += n_busy * dt
                k_bucket = min(n_sys, 15)
                state_time_by_k[k_bucket] = state_time_by_k.get(k_bucket, 0.0) + dt
        state["last_event_time"] = now

    def _record_transition(delta: int, event_type: str) -> None:
        now = float(env.now)
        _accumulate_areas(now)
        state["n_system"] += delta
        if len(timeline_events) < max_timeline_points:
            n_sys = state["n_system"]
            timeline_events.append(
                {
                    "time_mins": now,
                    "n_system": n_sys,
                    "n_queue": max(0, n_sys - n_servers),
                    "event": event_type,
                }
            )

    def _customer_process(customer_id: int, arrival_t: float) -> simpy.events.Event:
        q_on_arrival = max(0, state["n_system"] - n_servers)
        _record_transition(+1, "arrival")

        with server.request() as req:
            yield req
            service_start_t = float(env.now)
            wait_t = service_start_t - arrival_t
            service_dur = inverse_transform_exponential(mu_per_min, rng)
            service_samples.append(service_dur)
            yield env.timeout(service_dur)
            departure_t = float(env.now)
            _record_transition(-1, "departure")

            if departure_t >= warmup_time:
                records.append(
                    CustomerRecord(
                        customer_id=customer_id,
                        arrival_time=arrival_t,
                        service_start_time=service_start_t,
                        departure_time=departure_t,
                        wait_time=wait_t,
                        service_duration=service_dur,
                        system_time=departure_t - arrival_t,
                        queue_length_on_arrival=q_on_arrival,
                    )
                )

    def _arrival_generator() -> simpy.events.Event:
        cid = 0
        while True:
            dt = inverse_transform_exponential(lambda_per_min, rng)
            interarrival_samples.append(dt)
            yield env.timeout(dt)
            cid += 1
            env.process(_customer_process(cid, float(env.now)))

    env.process(_arrival_generator())
    env.run(until=sim_time)
    _accumulate_areas(sim_time)

    effective_horizon = sim_time - warmup_time
    l_sys_emp = state["area_system"] / effective_horizon
    l_q_emp = state["area_queue"] / effective_horizon
    rho_emp = state["area_busy"] / (n_servers * effective_horizon)
    occupancy_probs = {
        k: (t_val / effective_horizon) for k, t_val in sorted(state_time_by_k.items())
    }

    records_df = pd.DataFrame([r.__dict__ for r in records])
    w_sys_emp = float(records_df["system_time"].mean())
    w_q_emp = float(records_df["wait_time"].mean())

    return SimulationResult(
        arrival_rate=float(arrival_rate),
        service_rate=float(service_rate),
        n_servers=int(n_servers),
        sim_time=float(sim_time),
        warmup_time=float(warmup_time),
        customers_served=len(records),
        rho_empirical=float(rho_emp),
        l_system_empirical=float(l_sys_emp),
        l_queue_empirical=float(l_q_emp),
        w_system_mins_empirical=float(w_sys_emp),
        w_queue_mins_empirical=float(w_q_emp),
        interarrival_samples_mins=np.array(interarrival_samples, dtype=np.float64),
        service_samples_mins=np.array(service_samples, dtype=np.float64),
        records_df=records_df,
        timeline_df=pd.DataFrame(timeline_events),
        state_occupancy=occupancy_probs,
    )


def compare_empirical_vs_theoretical(result: SimulationResult) -> pd.DataFrame:
    theo = compute_mmc_theoretical(
        result.arrival_rate,
        result.service_rate,
        n_servers=result.n_servers,
    )
    emp_p0 = result.state_occupancy.get(0, 0.0)
    metrics_spec = [
        ("Utilización (rho)", theo["rho_per_server"], result.rho_empirical),
        ("Sistema vacío (P0)", theo["p0"], emp_p0),
        ("Clientes en sistema (L)", theo["l_system"], result.l_system_empirical),
        ("Clientes en fila (Lq)", theo["l_queue"], result.l_queue_empirical),
        ("Tiempo total W (min)", theo["w_system_mins"], result.w_system_mins_empirical),
        ("Espera en fila Wq (min)", theo["w_queue_mins"], result.w_queue_mins_empirical),
    ]

    rows = []
    for label, t_val, e_val in metrics_spec:
        rel_err = abs(e_val - t_val) / t_val * 100.0
        rows.append(
            {
                "Métrica": label,
                "Teórico": float(t_val),
                "SimPy": float(e_val),
                "Error (%)": float(rel_err),
            }
        )
    return pd.DataFrame(rows)


def run_rho_sweep_simulation(
    service_rate: float,
    rho_values: list[float],
    sim_time: float = 120000.0,
    warmup_time: float = 2000.0,
    seed: int = 101,
) -> pd.DataFrame:
    rows = []
    for idx, rho in enumerate(rho_values):
        lam = rho * service_rate
        theo = compute_mm1_theoretical(lam, service_rate)
        sim_res = run_queue_simulation(
            arrival_rate=lam,
            service_rate=service_rate,
            sim_time=sim_time,
            n_servers=1,
            warmup_time=warmup_time,
            seed=seed + idx * 17,
            max_timeline_points=10,
        )
        rows.append(
            {
                "rho": theo.rho,
                "Tasa": theo.arrival_rate,
                "Lq": theo.l_queue,
                "L": theo.l_system,
                "Wq teo": theo.w_queue_mins,
                "W teo": theo.w_system_mins,
                "W SimPy": sim_res.w_system_mins_empirical,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams["axes.edgecolor"] = "#CBD5E0"
    plt.rcParams["axes.linewidth"] = 0.8

    sim_mm1 = run_queue_simulation(
        arrival_rate=5.0,
        service_rate=7.5,
        sim_time=180000.0,
        n_servers=1,
        warmup_time=3000.0,
        seed=2026,
    )
    print("=== Convergencia M/M/1 (lambda=5.0/h, mu=7.5/h) ===")
    print(compare_empirical_vs_theoretical(sim_mm1).to_string(index=False))

    rho_levels = [0.40, 2.0 / 3.0, 0.80, 0.90, 0.95]
    df_sweep = run_rho_sweep_simulation(service_rate=7.5, rho_values=rho_levels)
    print("\n=== Barrido de Congestión (mu=7.5/h fija) ===")
    print(df_sweep.round(3).to_string(index=False))

    sim_mm2 = run_queue_simulation(
        arrival_rate=5.0,
        service_rate=7.5,
        sim_time=180000.0,
        n_servers=2,
        warmup_time=3000.0,
        seed=2026,
    )
    theo_m1 = compute_mmc_theoretical(5.0, 7.5, n_servers=1)
    theo_m2 = compute_mmc_theoretical(5.0, 7.5, n_servers=2)
    df_cap = pd.DataFrame(
        [
            {
                "Modelo": "M/M/1 (1 caja)",
                "Uso (%)": theo_m1["rho_per_server"] * 100.0,
                "Espera (%)": theo_m1["prob_wait"] * 100.0,
                "Lq": theo_m1["l_queue"],
                "Wq teo": theo_m1["w_queue_mins"],
                "Wq SimPy": sim_mm1.w_queue_mins_empirical,
                "W teo": theo_m1["w_system_mins"],
            },
            {
                "Modelo": "M/M/2 (2 cajas)",
                "Uso (%)": theo_m2["rho_per_server"] * 100.0,
                "Espera (%)": theo_m2["prob_wait"] * 100.0,
                "Lq": theo_m2["l_queue"],
                "Wq teo": theo_m2["w_queue_mins"],
                "Wq SimPy": sim_mm2.w_queue_mins_empirical,
                "W teo": theo_m2["w_system_mins"],
            },
        ]
    )
    print("\n=== Comparativa M/M/1 vs M/M/2 ===")
    print(df_cap.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
