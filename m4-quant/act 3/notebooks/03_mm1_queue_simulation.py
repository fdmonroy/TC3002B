# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Actividad 4.4
#
# **Curso:** Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 570)  
# **Profesor:** Luis Ricardo Peña Llamas  
# **Alumno:** Fernando Daniel Monroy Sánchez (A01750536)  
# **Fecha:** 22 de octubre de 2026  
#
# ---
#
# ## 1. Introducción
#
# En este trabajo se modela la caja de cobro única de una zapatería como una línea de espera $M/M/1$. Los clientes llegan a una tasa media de $\lambda = 5.0\text{ clientes por hora}$ (uno cada $12.0\text{ minutos}$). El cajero atiende a una razón media de $\mu = 7.5\text{ clientes por hora}$ ($8.0\text{ minutos}$ por cliente).
#
# El reporte cubre cuatro puntos:
# 1. Clasificación del sistema bajo la notación de Kendall ($M/M/1/\infty/\infty/\text{FIFO}$).
# 2. Simulación por eventos discretos en SimPy mediante transformada inversa.
# 3. Validación de las métricas empíricas contra las ecuaciones analíticas y la Ley de Little ($L = \lambda W$).
# 4. Análisis de congestión cuando $\rho \to 1$ y comparación contra una segunda caja ($M/M/2$).

# %%
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from great_tables import GT

from act_3.analytics import compute_mmc_theoretical
from act_3.simulation import (
    run_queue_simulation,
    compare_empirical_vs_theoretical,
    run_rho_sweep_simulation,
)
from act_3.visualizer import (
    plot_input_distributions,
    plot_temporal_dynamics_and_occupancy,
    plot_congestion_explosion,
    plot_capacity_comparison,
)

assets_dir = Path("assets") if Path("assets").is_dir() else Path("../assets")
assets_dir.mkdir(parents=True, exist_ok=True)


def _gt_to_latex(self):
    latex_str = self.as_latex()
    latex_str = re.sub(
        r"\\begin\{table\}(\[[^\]]*\])?",
        r"\\begin{table}[H]\n\\centering",
        latex_str,
    )
    latex_str = latex_str.replace(
        r"\begin{tabular*}{\linewidth}",
        r"\begin{tabular*}{0.75\linewidth}",
    )
    latex_str = re.sub(
        r"\\fontsize\{[^}]+\}\{[^}]+\}\\selectfont",
        r"\\normalsize",
        latex_str,
    )
    latex_str = re.sub(
        r"\\caption\*\{\s*\{\\large\s+([^\}]+)\}\s*\}",
        r"\\caption*{\\textbf{\\normalsize \1}}",
        latex_str,
    )
    return latex_str


GT._repr_latex_ = _gt_to_latex

sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams["axes.edgecolor"] = "#CBD5E0"
plt.rcParams["axes.linewidth"] = 0.8

kendall_data = pd.DataFrame(
    [
        {
            "Componente": "A: Proceso de llegadas",
            "Símbolo": "M (Markoviano)",
            "Especificación": "Poisson (5.0/h; media 12.0 min)",
        },
        {
            "Componente": "B: Tiempo de servicio",
            "Símbolo": "M (Markoviano)",
            "Especificación": "Exponencial (7.5/h; media 8.0 min)",
        },
        {
            "Componente": "c: Servidores activos",
            "Símbolo": "1",
            "Especificación": "1 caja de cobro activa",
        },
        {
            "Componente": "K: Capacidad del sistema",
            "Símbolo": "Infinito",
            "Especificación": "Sin truncamiento de fila",
        },
        {
            "Componente": "N: Población fuente",
            "Símbolo": "Infinito",
            "Especificación": "Población abierta de clientes",
        },
        {
            "Componente": "D: Disciplina de atención",
            "Símbolo": "FIFO",
            "Especificación": "Primero en llegar, primero atendido",
        },
        {
            "Componente": "rho: Intensidad de tráfico",
            "Símbolo": "2/3 (0.6667)",
            "Especificación": "66.67% de utilización media",
        },
    ]
)

(
    GT(kendall_data)
    .tab_header(title="Especificación del Sistema según la Notación de Kendall")
    .cols_align(align="left", columns=["Componente", "Símbolo", "Especificación"])
)

# %% [markdown]
# ```{=latex}
# \newpage
# ```
#
# ## 2. Desarrollo
#
# ### 2.1. Propiedades de Kendall
#
# En la notación $A/B/c/K/N/D$, el modelo $M/M/1$ indica llegadas de Poisson y tiempos de servicio exponenciales con un servidor. El número de llegadas en un intervalo $t$ sigue la distribución de Poisson:
# $$P(N(t) = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}, \quad k = 0, 1, 2, \dots$$
#
# El tiempo entre llegadas $T_a$ y el tiempo de servicio $T_s$ siguen distribuciones exponenciales sin memoria con medias $1/\lambda = 12.0\text{ min}$ y $1/\mu = 8.0\text{ min}$:
# $$f_{T_a}(t) = \lambda e^{-\lambda t}, \qquad f_{T_s}(t) = \mu e^{-\mu t}, \qquad t \ge 0$$
#
# Para simular estas variables a partir de números uniformes $U \sim \text{Uniforme}(0, 1)$, se aplica la transformada inversa:
# $$X = -\frac{1}{\theta} \ln(1 - U)$$

# %%
sim_mm1 = run_queue_simulation(
    arrival_rate=5.0,
    service_rate=7.5,
    sim_time=180000.0,
    n_servers=1,
    warmup_time=3000.0,
    seed=2026,
)

plot_input_distributions(
    sim_mm1,
    save_path=assets_dir / "distribuciones_entrada.png",
)
plt.show()

# %% [markdown]
# ### 2.2. Simulación por eventos
#
# En SimPy el reloj salta directamente entre eventos discretos: llegada del cliente (`arrival`), inicio de atención en el recurso `Resource(capacity=1)` y salida (`departure`). Cada cliente solicita la caja con `server.request()`, espera en fila el tiempo $W_{q,i}$ y ocupa el servidor durante `env.timeout(service_duration)`.

# %%
sample_records = sim_mm1.records_df.head(10).copy()
sample_table_df = pd.DataFrame(
    [
        {
            "ID": int(row["customer_id"]),
            "Llegada": f"{row['arrival_time']:.2f}",
            "Inicio": f"{row['service_start_time']:.2f}",
            "Espera Wq": f"{row['wait_time']:.2f}",
            "Servicio": f"{row['service_duration']:.2f}",
            "Salida": f"{row['departure_time']:.2f}",
            "Fila": int(row["queue_length_on_arrival"]),
        }
        for _, row in sample_records.iterrows()
    ]
)

(
    GT(sample_table_df)
    .tab_header(title="Muestra de Eventos Discretos Registrados (Minutos)")
    .cols_align(align="center", columns=["ID", "Fila"])
    .cols_align(
        align="right",
        columns=["Llegada", "Inicio", "Espera Wq", "Servicio", "Salida"],
    )
)

# %%
plot_temporal_dynamics_and_occupancy(
    sim_mm1,
    window_mins=360.0,
    save_path=assets_dir / "dinamica_temporal_cola.png",
)
plt.show()

# %% [markdown]
# ### 2.3. Desempeño estacionario
#
# Cuando $\rho = \lambda / \mu < 1$, la probabilidad de tener $k$ clientes en el sistema es $P_k = (1 - \rho)\rho^k$, con $P_0 = 1/3 \approx 0.3333$. Por la Ley de Little ($L = \lambda W$ y $L_q = \lambda W_q$), las métricas teóricas son:
# $$L = \frac{\rho}{1 - \rho} = 2.0, \qquad L_q = \frac{\rho^2}{1 - \rho} = 1.3333$$
# $$W = \frac{1}{\mu - \lambda} = 24.0\text{ min}, \qquad W_q = \frac{\lambda}{\mu(\mu - \lambda)} = 16.0\text{ min}$$

# %%
df_comparison = compare_empirical_vs_theoretical(sim_mm1)

(
    GT(df_comparison)
    .tab_header(title="Convergencia Empírica en SimPy frente a Solución Analítica M/M/1")
    .fmt_number(columns=["Teórico", "SimPy"], decimals=4)
    .fmt_number(columns=["Error (%)"], decimals=2)
    .cols_align(align="left", columns=["Métrica"])
    .cols_align(align="right", columns=["Teórico", "SimPy", "Error (%)"])
)

# %% [markdown]
# ### 2.4. Congestión no lineal
#
# Con capacidad fija $\mu = 7.5\text{ clientes por hora}$, el tiempo en el sistema $W = \frac{1}{\mu(1 - \rho)}$ crece de forma hiperbólica cuando $\rho \to 1$. Al subir la tasa de llegada un 35% (de $\lambda = 5.0\text{/h}$ con $\rho = 0.67$ a $\lambda = 6.75\text{/h}$ con $\rho = 0.90$), el tiempo total $W$ pasa de $24.0\text{ minutos}$ a $80.0\text{ minutos}$.

# %%
rho_levels = [0.40, 2.0 / 3.0, 0.80, 0.90, 0.95]
df_sweep = run_rho_sweep_simulation(
    service_rate=7.5,
    rho_values=rho_levels,
    sim_time=120000.0,
    warmup_time=2000.0,
    seed=101,
)

(
    GT(df_sweep)
    .tab_header(title="Crecimiento No Lineal de la Espera con Capacidad Fija (7.5/h)")
    .fmt_number(columns=["rho", "Lq", "L"], decimals=2)
    .fmt_number(columns=["Tasa"], decimals=3)
    .fmt_number(columns=["Wq teo", "W teo", "W SimPy"], decimals=1)
    .cols_align(align="right", columns=list(df_sweep.columns))
)

# %%
plot_congestion_explosion(
    df_sweep,
    service_rate=7.5,
    save_path=assets_dir / "explosion_congestion_rho.png",
)
plt.show()

# %% [markdown]
# ### 2.5. Dimensionamiento operativo
#
# Si se abre una segunda caja ($M/M/2$) bajo la misma demanda ($\lambda = 5.0\text{/h}$, $\mu = 7.5\text{/h}$), la utilización por cajero baja a $\rho_2 = \lambda / (2\mu) = 33.33\%$. Con ello, la espera promedio en fila $W_q$ cae de $16.00\text{ minutos}$ a $1.00\text{ minuto}$, y el tiempo total $W$ baja de $24.00\text{ minutos}$ a $9.00\text{ minutos}$.

# %%
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

df_capacity = pd.DataFrame(
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

(
    GT(df_capacity)
    .tab_header(title="Impacto Operativo de Apertura de Segunda Caja (5.0/h vs 7.5/h)")
    .fmt_number(
        columns=["Uso (%)", "Espera (%)", "Wq teo", "Wq SimPy", "W teo"],
        decimals=2,
    )
    .fmt_number(columns=["Lq"], decimals=4)
    .cols_align(align="left", columns=["Modelo"])
    .cols_align(
        align="right",
        columns=[c for c in df_capacity.columns if c != "Modelo"],
    )
)

# %%
plot_capacity_comparison(
    df_capacity,
    sim_mm1.records_df["wait_time"].values,
    sim_mm2.records_df["wait_time"].values,
    save_path=assets_dir / "comparativa_capacidad_mmc.png",
)
plt.show()

# %% [markdown]
# ```{=latex}
# \newpage
# ```
#
# ## 3. Reflexión
#
# Simular con eventos discretos en SimPy ofrece dos ventajas directas. Primero, el reloj salta de un evento al siguiente usando tiempos generados con la transformada inversa $X = -(1/\lambda)\ln(1 - U)$. No pierde tiempo evaluando segundos vacíos y simula miles de horas en pocos segundos. Segundo, permite probar reglas reales como cambios de turno, abandonos de fila o capacidad limitada sin resolver ecuaciones complejas.
#
# Modelar un proceso como una fila muestra que los promedios simples engañan. Si llega un cliente cada 12 minutos y el cajero tarda 8 minutos, parece que sobran 4 minutos por persona. Sin embargo, como las llegadas son aleatorias, se forman rachas que producen una espera promedio de 16 minutos en fila. Además, la espera crece de forma no lineal cuando la ocupación $\rho$ se acerca a 1. Subir las llegadas un 35% (de $\rho = 0.67$ a $\rho = 0.90$) triplica el tiempo total de 24 a 80 minutos porque el cajero se queda sin margen para absorber picos.
#
# Con las métricas de la fila ($L$, $L_q$, $W$, $W_q$, $\rho$ y las probabilidades $P_k$), un administrador puede decidir cuántas cajas abrir según la hora, tal como hacen tiendas como Walmart. Al pasar de una a dos cajas, la espera en fila baja de 16 minutos a 1 minuto, pero el uso de cada cajero cae de 66.7% a 33.3%. La simulación permite medir este balance entre tiempo de espera del cliente y costo de personal antes de hacer cambios físicos.
