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
# # Actividad 4.5
#
# **Curso:** Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 570)  
# **Profesor:** Luis Ricardo Peña Llamas  
# **Alumno:** Fernando Daniel Monroy Sánchez (A01750536)  
# **Fecha:** 5 de noviembre de 2026  
#
# ---
#
# ## 1. Introducción
#
# Esta actividad mide qué tanto se parecen dos cuerpos de texto. La idea es simple: cada texto se convierte en un vector de pesos TF-IDF y la cercanía entre dos textos se mide con la similitud coseno.
#
# El corpus base es el guion de Shrek (2001). Con cadenas de Markov generamos un texto sintético del tamaño de la mitad del original, y luego comparamos el guion contra ese texto, contra Shrek 2, contra Toy Story y contra Alicia en el país de las maravillas.
#
# ¿Por qué Shrek? Tiene mucho diálogo, vocabulario propio (ogre, swamp, Farquaad), una transcripción limpia disponible y sus 9110 palabras alcanzan para sostener una cadena de Markov de orden 2. Los otros tres textos dan un gradiente natural: misma franquicia, mismo género y un registro literario distinto.
#
# El reporte sigue el pipeline de los módulos de `act_4`:
#
# 1. Descarga y limpieza de los corpus en `data/raw`, con el módulo `corpus`.
# 2. Generación reproducible del texto sintético con `markovify`, orden 2 y semilla 2026, con el módulo `generator`.
# 3. Vectorización TF-IDF, similitud coseno y gráficas, con el módulo `similarity`.
# 4. Reflexión final.

# %%
from pathlib import Path
import re

import numpy as np
import pandas as pd
from great_tables import GT

from act_4.corpus import CORPUS_FILES, load_corpus

# Raíz de la actividad: contiene data/ y assets/. El cuaderno se ejecuta
# desde notebooks/, así que fuera de la raíz se usa el directorio padre.
root_dir = Path(".") if Path("data").is_dir() else Path("..")
assets_dir = Path("assets") if Path("assets").is_dir() else Path("../assets")
raw_dir = root_dir / "data" / "raw"

# %%
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

# %% [markdown]
# ```{=latex}
# \newpage
# ```
#
# ## 2. Cuerpo de texto
#
# El corpus base es la transcripción del diálogo de Shrek (2001), tomada de la página canónica del wiki de la franquicia. La URL que daba la actividad ya no existe y la sustitución se documenta en la sección 7. Los textos de contraste son:
#
# * **Shrek 2 (2004):** transcripción del mismo wiki de Fandom. Comparte personajes y expresiones con el original, aunque el guion es de otro equipo.
# * **Toy Story (1995):** guion de IMSDb, porque la transcripción de fandom fue retirada. Trae algunas acotaciones de escena además del diálogo.
# * **Alicia en el país de las maravillas (1865):** ebook 11 de Project Gutenberg sin la licencia. Es narrativa literaria británica, no diálogo de película.
#
# La limpieza del módulo `corpus` deja cada página en texto corrido: quita navegación, tablas y leyendas del wiki, tira etiquetas de personaje y descripciones de escena, y se queda solo con las líneas habladas. Alicia pasa por `strip_gutenberg_boilerplate` para quitarle la licencia del inicio y del final.

# %%
CORPUS_LABELS = {
    "shrek": "Shrek (2001)",
    "shrek_2": "Shrek 2 (2004)",
    "toy_story": "Toy Story (1995)",
    "alicia": "Alicia (1865)",
}

corpora = {name: load_corpus(name, raw_dir) for name in CORPUS_FILES}

corpus_table = pd.DataFrame(
    [
        {
            "Corpus": CORPUS_LABELS[name],
            "Archivo": CORPUS_FILES[name],
            "Palabras": len(text.split()),
            "Caracteres": len(text),
        }
        for name, text in corpora.items()
    ]
)

(
    GT(corpus_table)
    .tab_header(title="Corpus limpios almacenados en data/raw")
    .fmt_number(columns=["Palabras", "Caracteres"], decimals=0)
    .cols_align(align="left", columns=["Corpus", "Archivo"])
    .cols_align(align="right", columns=["Palabras", "Caracteres"])
)

# %%
import textwrap

print(textwrap.fill(corpora["shrek"][:400] + " ...", width=100))

# %% [markdown]
# ## 3. Texto sintético
#
# El texto sintético se generó con `markovify` usando una cadena de Markov de orden 2: la probabilidad de la siguiente palabra depende del par de palabras previas. Cada oración sale de caminar la cadena hasta un fin de oración, con hasta 100 reintentos.
#
# Las decisiones quedaron fijadas en el módulo `generator`:
#
# * **Tamaño objetivo:** la mitad de las palabras del original, o sea 4555 de 9110, con margen de 10%. El texto final tiene 4558 palabras.
# * **Semilla 2026:** `markovify` 0.9.4 no acepta un generador aleatorio en `make_sentence`, así que se resembra el módulo global `random` con 2026 antes de construir el modelo y antes de generar. Así dos corridas sobre el mismo corpus producen un texto idéntico por bytes.
# * **Sonda de viabilidad:** antes de generar se prueban 500 oraciones con el modelo de orden 2. Si menos de 100 salieran utilizables, se reconstruiría el modelo con orden 1. Nunca hizo falta.
#
# La celda de abajo reconstruye el modelo, regenera el texto en memoria y lo compara contra el archivo en disco para demostrar que la generación es reproducible.

# %%
from collections import Counter

from act_4.generator import (
    SYNTHETIC_TARGET_FRACTION,
    build_model,
    generate_synthetic,
    target_word_count,
)

shrek_text = corpora["shrek"]
model, state_size_used = build_model(shrek_text, state_size=2, seed=2026)
synthetic_regenerated = generate_synthetic(
    model, target_word_count(shrek_text), seed=2026
)

synthetic_path = root_dir / "data" / "synthetic" / "shrek_sintetico.txt"
synthetic_disk = synthetic_path.read_text(encoding="utf-8")
reproducible = synthetic_regenerated == synthetic_disk

synthetic_sentences = [line for line in synthetic_disk.splitlines() if line.strip()]
original_sentences = {
    s.strip() for s in re.split(r"(?<=[.!?])\s+", shrek_text) if s.strip()
}
verbatim_duplicates = sum(1 for s in synthetic_sentences if s in original_sentences)
repeated_within = len(synthetic_sentences) - len(set(synthetic_sentences))
prefix_counts = Counter(" ".join(s.split()[:4]) for s in synthetic_sentences)
shared_prefixes = sum(c - 1 for c in prefix_counts.values() if c > 1)
loop_example = next(
    (s for s in synthetic_sentences if "that you that you" in s), None
)

generation_table = pd.DataFrame(
    [
        {"Indicador": "Palabras del corpus original", "Valor": f"{len(shrek_text.split())}"},
        {"Indicador": "Fracción objetivo", "Valor": f"{SYNTHETIC_TARGET_FRACTION}"},
        {"Indicador": "Palabras objetivo", "Valor": f"{target_word_count(shrek_text)}"},
        {"Indicador": "Palabras generadas", "Valor": f"{len(synthetic_disk.split())}"},
        {"Indicador": "Oraciones generadas", "Valor": f"{len(synthetic_sentences)}"},
        {"Indicador": "Tamaño de estado usado", "Valor": f"{state_size_used} (respaldo a 1 no activado)"},
        {"Indicador": "Oraciones duplicadas literales del original", "Valor": f"{verbatim_duplicates}"},
        {"Indicador": "Oraciones repetidas dentro del texto sintético", "Valor": f"{repeated_within}"},
        {"Indicador": "Repeticiones de línea por prefijo de 4 palabras", "Valor": f"{shared_prefixes}"},
        {"Indicador": "Reproducible contra el archivo en disco", "Valor": f"{reproducible}"},
    ]
)

(
    GT(generation_table)
    .tab_header(title="Generación del corpus sintético con cadenas de Markov")
    .cols_align(align="left", columns=["Indicador", "Valor"])
)

# %%
for sentence in synthetic_sentences[:3]:
    print("-", sentence)
print("Ejemplo de bucle de repetición:", loop_example)

# %% [markdown]
# El texto sintético suena parecido al original, con oraciones en su mayoría gramaticales o casi gramaticales. Sus 497 líneas son todas nuevas y ninguna es copia literal de una oración del guion: hay 0 duplicados. Las rarezas del texto se comentan en la sección 7.

# %% [markdown]
# ## 4. Representación vectorial
#
# **Bolsa de palabras (BOW).** Cada documento es un vector de conteos $w_{t,d}$ sobre el vocabulario global: cuántas veces aparece el término $t$ en el documento $d$. La representación guarda las frecuencias pero tira el orden de las palabras: "Shrek rescata a Fiona" y "Fiona rescata a Shrek" dan el mismo vector.
#
# **Ponderación TF-IDF.** Cada conteo se reescala con el producto $TF \times IDF$, de modo que los términos frecuentes en un documento pero raros en el corpus reciben mayor peso. Con los parámetros usados:
#
# * **TF sublineal** (`sublinear_tf=True`): la frecuencia bruta se sustituye por
# $$tf'(t,d) = 1 + \ln tf(t,d)$$
# que amortigua la contribución de un término muy repetido dentro de un mismo documento.
# * **IDF suavizado** (el esquema de scikit-learn con `smooth_idf=True`):
# $$idf(t) = \ln\!\left(\frac{1 + N}{1 + df(t)}\right) + 1$$
# donde $N$ es el número de documentos y $df(t)$ el número de documentos donde aparece el término $t$.
# * El peso final es $w_{t,d} = tf'(t,d) \cdot idf(t)$ y cada vector se normaliza con norma L2 (`norm="l2"`), de modo que $\|d\|_2 = 1$ y la similitud coseno se reduce al producto punto.
#
# Con `ngram_range=(1, 3)` el vocabulario incluye bigramas y trigramas. La lista de palabras vacías del inglés elimina términos gramaticales de alto uso que no distinguen documentos.

# %%
from act_4.similarity import VECTORIZER_PARAMS

PARAM_EFFECT = {
    "lowercase": "Convertir todo a minúsculas antes de tokenizar",
    "stop_words": "Lista de palabras vacías del inglés eliminada del vocabulario",
    "ngram_range": "El vocabulario incluye uni, bi y trigramas",
    "max_df": "Descartar términos presentes en más del 90% de los documentos",
    "min_df": "Conservar todo término que aparezca al menos una vez",
    "sublinear_tf": "Frecuencia sublineal 1 + ln(tf) que amortigua términos dominantes",
    "norm": "Normalizar cada vector con norma L2",
}

params_table = pd.DataFrame(
    [
        {
            "Parámetro": name,
            "Valor": repr(value),
            "Efecto": PARAM_EFFECT[name],
        }
        for name, value in VECTORIZER_PARAMS.items()
    ]
)

(
    GT(params_table)
    .tab_header(title="Parámetros de TfidfVectorizer usados en la actividad")
    .cols_align(align="left", columns=["Parámetro", "Valor", "Efecto"])
)

# %% [markdown]
# ## 5. Similitud coseno
#
# La cercanía entre dos vectores $u$ y $v$ se mide con el coseno del ángulo entre ellos:
# $$\text{similitud}(u, v) = \cos(\theta) = \frac{u \cdot v}{\|u\|_2 \, \|v\|_2}$$
#
# La métrica compara la dirección de los vectores, no su longitud. Con vectores TF-IDF no negativos da valores entre 0 y 1. La diagonal de la matriz vale 1.0, la matriz es simétrica y el resto mide el traslape de vocabulario y n-gramas entre cada par.
#
# Los cinco textos se vectorizan con un solo ajuste del `TfidfVectorizer`, así todas las comparaciones usan el mismo vocabulario. La matriz resultante es dispersa, como en la demo de clase, y las celdas de abajo reportan su forma y su densidad.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

from act_4.similarity import (
    DOCUMENT_LABELS,
    LABEL_SHREK,
    build_document_set,
    compute_similarity_matrix,
    plot_heatmap,
    plot_ranked_bar,
    rank_against,
)

documents = build_document_set(root_dir)
matrix = compute_similarity_matrix(documents)

docs_in_order = [documents[label] for label in DOCUMENT_LABELS]
tfidf = TfidfVectorizer(**VECTORIZER_PARAMS).fit_transform(docs_in_order)
total_entries = tfidf.shape[0] * tfidf.shape[1]

print("Orden de los documentos:", list(DOCUMENT_LABELS))
print("Diagonal igual a 1.0:", bool(np.allclose(np.diag(matrix.values), 1.0)))
print("Matriz simétrica:", bool(np.allclose(matrix.values, matrix.values.T)))
print("Forma de la matriz TF-IDF:", tfidf.shape)
print("Entradas distintas de cero:", tfidf.nnz, "de", total_entries)
print("Densidad: {:.1%}".format(tfidf.nnz / total_entries))

# %%
matrix_display = matrix.round(4).rename_axis("Documento").reset_index()

(
    GT(matrix_display)
    .tab_header(title="Matriz de similitud coseno entre los cinco textos (TF-IDF)")
    .fmt_number(columns=list(matrix.columns), decimals=4)
    .cols_align(align="left", columns=["Documento"])
    .cols_align(align="right", columns=list(matrix.columns))
)

# %%
plot_heatmap(matrix, assets_dir / "similitud_heatmap.png")

# %% [markdown]
# ![Mapa de calor de la matriz de similitud coseno](../assets/similitud_heatmap.png)

# %%
ranking = rank_against(LABEL_SHREK, matrix)

ranking_display = (
    ranking.round(4)
    .rename("Similitud")
    .rename_axis("Texto")
    .reset_index()
)

(
    GT(ranking_display)
    .tab_header(title="Similitud de cada texto contra Shrek (2001), en orden descendente")
    .fmt_number(columns=["Similitud"], decimals=4)
    .cols_align(align="left", columns=["Texto"])
    .cols_align(align="right", columns=["Similitud"])
)

# %% [markdown]
# ```{=latex}
# \newpage
# ```
#
# ## 6. Comparación contra otros cuerpos
#
# El ranking contra Shrek (2001) ordena los textos de mayor a menor traslape léxico y de n-gramas:
#
# | Texto | Similitud |
# |---|---|
# | Shrek Sintético | 0.2144 |
# | Shrek 2 | 0.0827 |
# | Toy Story | 0.0474 |
# | Alicia | 0.0393 |
#
# El orden es el que uno esperaría. Primero el texto sintético, construido con el propio vocabulario de Shrek. Luego la secuela, que hereda personajes y expresiones. Después Toy Story, del mismo tipo de humor pero sin vocabulario compartido. Al final Alicia, una novela de 1865 narrada, no dialogada.
#
# El rango de n-gramas cambia las magnitudes pero no el orden. La celda de abajo repite la medición con solo unigramas, y con los demás parámetros iguales, para verificarlo.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

params_unigram = dict(VECTORIZER_PARAMS) | {"ngram_range": (1, 1)}
vectorizer = TfidfVectorizer(**params_unigram)
tfidf_unigram = vectorizer.fit_transform(
    [documents[label] for label in DOCUMENT_LABELS]
)
matrix_unigram = pd.DataFrame(
    cosine_similarity(tfidf_unigram),
    index=DOCUMENT_LABELS,
    columns=DOCUMENT_LABELS,
)
ranking_unigram = matrix_unigram[LABEL_SHREK].sort_values(ascending=False).drop(LABEL_SHREK)

comparison = pd.DataFrame(
    {
        "Similitud (n-gramas 1 a 3)": ranking.drop(LABEL_SHREK),
        "Similitud (solo unigramas)": ranking_unigram,
    }
).rename_axis("Texto").reset_index()

(
    GT(comparison)
    .tab_header(title="Ranking contra Shrek (2001) bajo dos configuraciones de vectorización")
    .fmt_number(
        columns=["Similitud (n-gramas 1 a 3)", "Similitud (solo unigramas)"],
        decimals=4,
    )
    .cols_align(align="left", columns=["Texto"])
    .cols_align(
        align="right",
        columns=["Similitud (n-gramas 1 a 3)", "Similitud (solo unigramas)"],
    )
)

# %%
plot_ranked_bar(ranking, assets_dir / "similitud_ordenada.png")

# %% [markdown]
# ![Ranking de similitud coseno contra Shrek (2001)](../assets/similitud_ordenada.png)

# %% [markdown]
# ## 7. Reflexión
#
# ### 7.1 ¿Cómo varía la medida de similitud en función del texto que se compara?
#
# Varía bastante y de forma predecible. La similitud crece con el vocabulario compartido y el ranking lo demuestra. El texto sintético va primero porque todas sus palabras vienen del original. Le sigue Shrek 2, que comparte personajes y expresiones. Toy Story solo comparte el tipo de humor. Alicia queda al final porque con Shrek solo comparte el inglés más común, justo el que las palabras vacías eliminan. Y como en la demo de clase, donde dos reseñas sin nada que ver quedaban encima solo por compartir la expresión grab and go, aquí hasta los pares menos parecidos quedan por encima de cero: compartir unas cuantas palabras ya mueve el número.
#
# Lo sorprendente fue la escala. En clase vimos una referencia: 1.0 para una copia, cerca de 0.89 para una paráfrasis y 0 para textos sin nada en común. Aquí el par más parecido apenas llega a 0.2144. La culpa es de la dimensionalidad: el vocabulario se dispara a casi 49 mil columnas, como se ve en la sección 5, un n-grama compartido ya es raro y los cosenos se aplastan hacia el cero. Con solo unigramas el orden queda igual y los valores suben mucho, como se ve en la sección 6. La lección que me llevo es leer un ranking de forma relativa, comparando sus valores entre sí y no contra un umbral absoluto.

# %% [markdown]
# ### 7.2 ¿Una similitud alta garantiza que dos textos son del mismo autor?
#
# No, y esta actividad trae la prueba. El texto más parecido a Shrek en toda la matriz es el sintético, con 0.2144, y no lo escribió nadie: lo escribió una cadena de Markov, sin estilo ni intención, solo reusando el vocabulario. La métrica compara frecuencias de términos y ni siquiera ve el orden de las palabras: "Shrek rescata a Fiona" y "Fiona rescata a Shrek" dan el mismo vector.
#
# Dicho de otro modo, TF-IDF mide parentesco de tema y vocabulario, no de autoría. Dos autores distintos quedan cerca cuando comparten franquicia o tema, como pasa entre Shrek y su secuela. Para atribuir autoría hacen falta otras señales, como las palabras función o el ritmo de las oraciones, y modelos hechos para eso, como el de Khmelev que viene en los recursos de la actividad.

# %% [markdown]
# ### 7.3 Decisiones y limitaciones
#
# Elegimos TF-IDF porque es interpretable y porque la pregunta de la actividad es una pregunta de frecuencias. Cada peso del vector es un término identificable y cada resultado se explica por los términos compartidos. Una métrica semántica costaría más y explicaría menos aquí. En clase también salió la idea de mezclar varias métricas de similitud, pero aquí una bastó. Es el KISS que se vio en clase.
#
# Tampoco el camino fue limpio. La sección 2 documenta las tres fuentes que hubo que sustituir, el texto sintético repite palabras y oraciones con ejemplos en la sección 3, y las palabras vacías son una lista fija, así que parte del estilo de los guiones se queda fuera de la métrica. Nada de esto cambia las conclusiones de arriba.
