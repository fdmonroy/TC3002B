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
# Esta actividad mide la semejanza entre textos. Cada texto se convierte en un vector de pesos TF-IDF y la cercanía entre dos textos se cuantifica con la similitud coseno.
#
# El corpus base es el guion de Shrek (2001). Se generó un texto sintético con cadenas de Markov, del tamaño de la mitad del original. Luego se comparó el guion contra ese texto sintético, contra la secuela Shrek 2, contra Toy Story y contra Alicia en el país de las maravillas.
#
# Se eligió Shrek porque su diálogo es abundante y con vocabulario propio (ogre, swamp, Farquaad), la transcripción está disponible como texto limpio y sus 9110 palabras sostienen una cadena de Markov de orden 2. Los tres textos de contraste forman un gradiente natural: misma franquicia, mismo género y registro literario distinto.
#
# El reporte sigue el pipeline de los módulos de `act_4`:
#
# 1. Descarga y limpieza de los cuatro corpus reales en `data/raw`, de forma idempotente, con el módulo `corpus`.
# 2. Generación reproducible del texto sintético con `markovify`, orden de estado 2 y semilla 2026, con el módulo `generator`.
# 3. Vectorización TF-IDF de los cinco textos, similitud coseno, matriz, mapa de calor y ranking, con el módulo `similarity`.
# 4. Reflexión sobre qué mide la métrica, qué no mide y qué fallos hubo.

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
# El corpus base es la transcripción del diálogo de Shrek (2001), tomada de la página canónica de transcripción del wiki de la franquicia. La URL indicada en la actividad ya no existe y la sustitución se documenta en la sección 7. Los textos de contraste son:
#
# * **Shrek 2 (2004):** transcripción del mismo wiki de Fandom. Comparte personajes y expresiones con el original, aunque su guion es de otro equipo de guionistas.
# * **Toy Story (1995):** guion publicado en IMSDb, porque la transcripción de fandom fue retirada. Conserva algunas acotaciones de escena además del diálogo.
# * **Alicia en el país de las maravillas (1865):** ebook 11 de Project Gutenberg con el encabezado y la licencia eliminados. Es narrativa literaria británica, no diálogo de película.
#
# La limpieza del módulo `corpus` reduce cada página a texto corrido: elimina la navegación, las tablas y las leyendas del wiki, descarta etiquetas de personaje y descripciones de escena, y conserva solo las líneas habladas. Alicia pasa por `strip_gutenberg_boilerplate`, que corta la licencia del inicio y del final.

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
# El texto sintético se generó con `markovify` mediante una cadena de Markov de orden 2: la probabilidad de la siguiente palabra depende del par de palabras previas. Cada oración se produce caminando la cadena hasta un marcador de fin de oración, con hasta 100 reintentos.
#
# Decisiones fijadas en el módulo `generator`:
#
# * **Tamaño objetivo:** la mitad de las palabras del original, es decir 4555 de 9110, con margen de 10%. El texto final tiene 4558 palabras.
# * **Semilla 2026:** `markovify` 0.9.4 no acepta un generador aleatorio en `make_sentence`, así que se resembra el módulo global `random` con 2026 antes de construir el modelo y antes de generar. Dos ejecuciones sobre el mismo corpus producen un texto idéntico por bytes.
# * **Sonda de viabilidad:** antes de generar se prueban 500 oraciones con el modelo de orden 2. Si menos de 100 salieran utilizables, el modelo se reconstruiría con orden 1. No fue necesario.
#
# La siguiente celda reconstruye el modelo, regenera el texto sintético en memoria y lo compara contra el archivo en disco para demostrar la reproducibilidad.

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
# El texto sintético imita el vocabulario y la cadencia del original con oraciones en su mayoría gramaticales o casi gramaticales. Sus 497 líneas son todas nuevas y ninguna es copia literal de una oración del guion original: hay 0 duplicados literales. Las patologías del texto se discuten en la sección 7.

# %% [markdown]
# ## 4. Representación vectorial
#
# **Bolsa de palabras (BOW).** Cada documento es un vector de conteos $w_{t,d}$ sobre el vocabulario global: cuántas veces aparece el término $t$ en el documento $d$. La representación conserva las frecuencias pero pierde el orden de las palabras: "Shrek rescata a Fiona" y "Fiona rescata a Shrek" producen el mismo vector.
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
# La métrica captura la coincidencia de orientación entre documentos, sin importar su longitud. Con vectores TF-IDF no negativos toma valores entre 0 y 1. La diagonal de la matriz vale 1.0, la matriz es simétrica y el resto de los valores mide el traslape de vocabulario y n-gramas entre cada par.
#
# Los cinco textos se vectorizan con un solo ajuste del `TfidfVectorizer` sobre una matriz compartida, de modo que todas las comparaciones usan el mismo vocabulario.

# %%
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

print("Orden de los documentos:", list(DOCUMENT_LABELS))
print("Diagonal igual a 1.0:", bool(np.allclose(np.diag(matrix.values), 1.0)))
print("Matriz simétrica:", bool(np.allclose(matrix.values, matrix.values.T)))

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
# El orden es el que la intuición sugiere. El texto más parecido a Shrek es el texto sintético, construido con su propio vocabulario. Sigue la secuela de la misma franquicia, con sus personajes y expresiones recurrentes. Después Toy Story, del mismo género y tipo de humor pero sin vocabulario compartido. Al final Alicia, una novela británica de 1865 con registro literario y narrado en lugar de dialogado.
#
# La elección del rango de n-gramas cambia las magnitudes pero no el orden. La siguiente celda repite la medición con solo unigramas como verificación de robustez y con el resto de los parámetros idéntico.

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
# La similitud crece con el vocabulario compartido y el ranking lo mostró. El texto sintético encabeza porque toma todo su vocabulario del original. Le sigue Shrek 2, que comparte personajes y expresiones. Toy Story comparte género pero casi no comparte vocabulario. Alicia queda última porque solo comparte el inglés común, que la lista de palabras vacías elimina.
#
# Las magnitudes, en cambio, dependen de la configuración de vectorización. En clase se vio una escala de referencia calibrada: 1.0 para una copia, cerca de 0.89 para una paráfrasis y 0 para textos sin términos comunes. Aquí el par más parecido apenas alcanza 0.2144. La causa es la dimensionalidad: con n-gramas de 1 a 3, un n-grama compartido es raro y los cosenos se comprimen hacia cero. La verificación con solo unigramas da el mismo orden con valores mucho mayores, como muestra la sección 6. La lectura correcta es comparar entre sí los valores de un mismo ranking y no contra un umbral absoluto.

# %% [markdown]
# ### 7.2 ¿Una similitud alta garantiza que dos textos son del mismo autor?
#
# No. El texto más parecido a Shrek en toda la matriz es el sintético, con 0.2144, y no lo escribió ningún autor. Lo escribió una cadena de Markov que remuestrea el vocabulario del original sin estilo ni intención. La métrica compara frecuencias de términos y pierde el orden de las palabras: "Shrek rescata a Fiona" y "Fiona rescata a Shrek" producen el mismo vector.
#
# Por eso la similitud mide parentesco temático y léxico, no autoría. Dos textos de autores distintos pueden quedar cerca cuando comparten tema o franquicia, como Shrek y su secuela. La atribución de autoría necesita señales distintas, como las palabras función o el ritmo de las oraciones, y modelos diseñados para eso, como el de Khmelev citado en los recursos de la actividad.

# %% [markdown]
# ### 7.3 Decisiones y limitaciones
#
# Se eligió TF-IDF porque es interpretable y responde una pregunta de frecuencias. Cada peso del vector es un término identificable y cada resultado se explica por los términos compartidos. Una métrica semántica añadiría costo y opacidad sin cambiar la pregunta. Es el principio de KISS que se validó en clase.
#
# El camino no fue limpio y conviene decirlo. La sección 2 documenta las tres sustituciones de fuente que la descarga requirió. El texto sintético repite palabras y oraciones, con los ejemplos en la sección 3. Y la lista de palabras vacías es fija, así que parte del estilo de los guiones queda fuera de la métrica. Ninguno de estos puntos cambia las conclusiones de la reflexión.
