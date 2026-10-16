# Actividad 4.2: Procesos estocasticos y generador de melodias mediante cadenas de Markov

Este proyecto implementa el modelado y simulacion de procesos estocasticos a traves de cadenas de Markov discretas en el tiempo para la generacion armonica y melodica de musica.

## Estructura del proyecto

* `notebooks/`: Cuadernos interactivos Jupyter en formato percent script (.py) y notebook (.ipynb) sincronizados mediante jupytext.
* `src/act_2/`: Modulos de codigo Python reutilizables (escalas musicales, sintesis y envolventes de audio, algebra de matrices de transicion, simulador de cadenas).
* `scripts/`: Filtros Lua para Pandoc y plantillas TeX para supresion de celdas de codigo y tipografia academica.
* `export/`: Entregables en formatos Word (.docx) y PDF (.pdf).
* `pyproject.toml`: Especificacion del proyecto y dependencias manejadas con uv.

## Fundamentos matematicos y musicales

* Cadena de Markov discreta en el tiempo: modelado de la propiedad de Markov donde la probabilidad condicional de la transicion depende exclusivamente del estado actual.
* Matriz estocastica: verificacion formal de entradas no negativas y suma de filas unitaria mediante algebra matricial.
* Sintesis acustica: generacion senoidal en temperamento igual de 12 tonos con suavizado de envolvente ADSR y duraciones ritmicas variables.

## Ejecucion

Para sincronizar dependencias y abrir el cuaderno interactivo en Jupyter:

```bash
uv sync
uv run jupyter lab
```
