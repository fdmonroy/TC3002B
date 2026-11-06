# Actividad 4.5: Similitud en textos

Este proyecto compara la similitud entre guiones de peliculas y un texto literario mediante representaciones vectoriales TF-IDF y similitud coseno, e incluye la generacion de pseudo dialogos con cadenas de Markov.

## Estructura del proyecto

* `notebooks/`: Cuadernos interactivos Jupyter en formato percent script (.py) y notebook (.ipynb) sincronizados mediante jupytext.
* `src/act_4/`: Modulos de codigo Python reutilizables (descarga y limpieza de corpus de guiones y texto literario, vectorizacion TF-IDF, calculo de similitud coseno y generacion de texto con cadenas de Markov).
* `scripts/`: Filtros Lua para Pandoc y plantillas TeX para supresion de celdas de codigo y tipografia academica.
* `data/raw/`: Corpus originales descargados (guiones de Shrek, Shrek 2 y Toy Story, y Alicia en el pais de las maravillas) junto con sus versiones limpias.
* `assets/`: Graficas de matrices de similitud y longitudes de corpus.
* `export/`: Entregables compilados en formatos Word (.docx) y PDF (.pdf).
* `pyproject.toml`: Especificacion del proyecto y dependencias manejadas con uv.

## Fundamentos matematicos

* Bolsa de palabras (BOW): cada documento se representa como un vector de conteos de terminos $w_{t,d}$ sobre el vocabulario global, ignorando el orden de las palabras pero conservando la frecuencia de aparicion de cada termino.
* Ponderacion TF-IDF: cada conteo se reescala con el producto $TF \times IDF$, donde la frecuencia inversa de documento se define como $IDF = \log(N/df(t))$, con $N$ el numero de documentos del corpus y $df(t)$ el numero de documentos donde aparece el termino $t$; los terminos frecuentes en un documento pero raros en el corpus reciben mayor peso.
* Similitud coseno: la cercania entre dos vectores $u$ y $v$ se mide con
$$\cos(\theta) = \frac{u \cdot v}{\|u\|\,\|v\|},$$
que captura la coincidencia de orientacion entre documentos independientemente de su longitud y toma valores entre $-1$ y $1$.
* Cadenas de Markov para generacion: un modelo de orden $k$ estima la probabilidad de la siguiente palabra dado el estado de las $k$ palabras previas, lo que permite sintetizar pseudo dialogos que imitan el estilo y el vocabulario de un corpus sin copiar sus oraciones exactas.

## Ejecucion

Para sincronizar dependencias y abrir el cuaderno interactivo en Jupyter:

```bash
uv sync
uv run jupyter lab
```
