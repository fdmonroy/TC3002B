"""Discrete event queue simulator in SimPy."""

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import simpy

from act_3.analytics import compute_mm1_theoretical, compute_mmc_theoretical


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


def inverse_transform_exponential(
    rate: float,
    rng: np.random.Generator,
) -> float:
    if rate <= 0.0:
        raise ValueError("rate must be > 0.")
    u = float(rng.uniform(0.0, 1.0))
    return -math.log(1.0 - u) / rate


def run_queue_simulation(
    arrival_rate: float,
    service_rate: float,
    sim_time: float = 120000.0,
    n_servers: int = 1,
    warmup_time: float = 2000.0,
    seed: int = 42,
    max_timeline_points: int = 2000,
) -> SimulationResult:
    if arrival_rate <= 0.0 or service_rate <= 0.0:
        raise ValueError("Rates must be > 0.")
    if n_servers < 1:
        raise ValueError("n_servers must be >= 1.")
    if sim_time <= warmup_time:
        raise ValueError("sim_time must be > warmup_time.")

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
    l_sys_emp = state["area_system"] / effective_horizon if effective_horizon > 0 else 0.0
    l_q_emp = state["area_queue"] / effective_horizon if effective_horizon > 0 else 0.0
    rho_emp = state["area_busy"] / (n_servers * effective_horizon) if effective_horizon > 0 else 0.0

    occupancy_probs = {
        k: (t_val / effective_horizon) for k, t_val in sorted(state_time_by_k.items())
    }

    if records:
        records_df = pd.DataFrame([r.__dict__ for r in records])
        w_sys_emp = float(records_df["system_time"].mean())
        w_q_emp = float(records_df["wait_time"].mean())
    else:
        records_df = pd.DataFrame(
            columns=[
                "customer_id",
                "arrival_time",
                "service_start_time",
                "departure_time",
                "wait_time",
                "service_duration",
                "system_time",
                "queue_length_on_arrival",
            ]
        )
        w_sys_emp = 0.0
        w_q_emp = 0.0

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
        rel_err = abs(e_val - t_val) / t_val * 100.0 if t_val > 0 else 0.0
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
    sim_time: float = 90000.0,
    warmup_time: float = 2000.0,
    seed: int = 42,
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
