# Actividad 4.4: Modelado y simulacion de lineas de espera M/M/1 mediante eventos discretos

Este proyecto implementa el modelado matematico y la simulacion por eventos discretos de sistemas de colas M/M/1 y M/M/c utilizando SimPy.

## Estructura del proyecto

* `notebooks/`: Cuadernos interactivos Jupyter en formato percent script (.py) y notebook (.ipynb) sincronizados mediante jupytext.
* `src/act_3/`: Modulos de codigo Python reutilizables (formulacion analitica de sistemas M/M/1 y Erlang C, motor de eventos discretos en SimPy, generacion por transformada inversa y visualizacion).
* `scripts/`: Filtros Lua para Pandoc y plantillas TeX para supresion de celdas de codigo y tipografia academica.
* `assets/`: Graficas de distribuciones de entrada, trayectorias temporales, congestion no lineal y dimensionamiento de capacidad.
* `export/`: Entregables compilados en formatos Word (.docx) y PDF (.pdf).
* `pyproject.toml`: Especificacion del proyecto y dependencias manejadas con uv.

## Fundamentos matematicos y operativos

* Notacion de Kendall ($M/M/1/\infty/\infty/\text{FIFO}$): llegadas de Poisson a tasa $\lambda$, tiempos de servicio exponenciales a tasa $\mu$ y servidor unico con disciplina primero en entrar, primero en salir.
* Ley de Little y estado estacionario: validacion empirica de $L = \lambda W$, $L_q = \lambda W_q$ y distribucion geometrica de ocupacion $P_k = (1 - \rho)\rho^k$ bajo condicion de estabilidad $\rho = \lambda/\mu < 1$.
* Simulacion por eventos discretos: muestreo exponencial mediante transformada inversa $X = -(1/\lambda)\ln(1 - U)$ e integracion temporal exacta de longitud de fila y utilizacion en SimPy.

## Ejecucion

Para sincronizar dependencias y abrir el cuaderno interactivo en Jupyter:

```bash
uv sync
uv run jupyter lab
```
