# Actividad 4.1: Procesos de la vida real como distribuciones de probabilidad

Este proyecto implementa el analisis estadistico y ajuste de distribuciones de probabilidad sobre datos reales de visitas urbanas en Nueva York recolectados por Foursquare (dataset TSMC2014).

## Estructura del proyecto

* `data/raw/`: Archivos de datos originales sin procesar.
* `data/processed/`: Tablas transformadas y agregaciones.
* `notebooks/`: Cuadernos Jupyter interactivos para analisis y graficacion.
* `src/act_1/`: Modulos de codigo Python reutilizables (carga de datos, utilidades).
* `pyproject.toml`: Especificacion del proyecto y dependencias de uv.

## Datos

* Fuente: Foursquare NYC Checkins (Dingqi Yang / Kaggle).
* Archivo principal: `data/raw/dataset_TSMC2014_NYC.txt`.
* Numero de observaciones: 227428 registros.

## Ejecucion

Para abrir el cuaderno interactivo en Jupyter:

```bash
uv run jupyter lab
```
