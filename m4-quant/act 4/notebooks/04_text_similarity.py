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
# Esta actividad estudia la semejanza entre textos mediante representaciones vectoriales. Cada texto se convierte en un vector de pesos TF-IDF y la cercanía entre dos textos se cuantifica con la similitud coseno. Sobre esa métrica se construyen dos productos:
#
# 1. La generación de un pseudo diálogo sintético con cadenas de Markov, del tamaño de la mitad del corpus original.
# 2. Una matriz de similitud que compara el guion de Shrek (2001) contra ese texto sintético, contra la secuela Shrek 2, contra otra película de animación del mismo género (Toy Story) y contra una novela clásica de registro literario (Alicia en el país de las maravillas).
#
# Se eligió Shrek como corpus base porque es una película de animación con diálogo abundante y expresiones propias (ogre, swamp, Farquaad), con transcripción disponible como texto limpio, y porque su tamaño (9110 palabras) sostiene sin problema una cadena de Markov de orden 2. Además, las tres obras de contraste forman un gradiente natural de comparación: misma franquicia, mismo género y registro literario distinto.
#
# El reporte sigue el pipeline completo de los módulos de `act_4`:
#
# 1. Descarga y limpieza de los cuatro corpus reales en `data/raw` (módulo `corpus`), de forma idempotente: si el archivo limpio ya existe, no se descarga nada.
# 2. Generación reproducible del corpus sintético con `markovify`, orden de estado 2 y semilla 2026 (módulo `generator`).
# 3. Vectorización TF-IDF de los cinco textos, similitud coseno, matriz completa, mapa de calor y ranking (módulo `similarity`).
# 4. Reflexión sobre qué mide la métrica, qué no mide y qué decisiones, sustituciones y fallos hubo en el camino.

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
# El corpus base es la transcripción del diálogo de Shrek (2001), obtenida de la página canónica de transcripción del wiki de la franquicia (la URL de transcripción indicada originalmente en la actividad ya no existe; se documenta la sustitución en la sección 7). Los textos de contraste son:
#
# * **Shrek 2 (2004):** transcripción del mismo wiki de Fandom; comparte personajes y expresiones con el original aunque su guion es de otro equipo de guionistas.
# * **Toy Story (1995):** guion publicado en IMSDb (la transcripción de fandom fue retirada); al venir de un guion cinematográfico conserva algunas acotaciones de escena además del diálogo.
# * **Alicia en el país de las maravillas (1865):** ebook 11 de Project Gutenberg, con el encabezado y la licencia eliminados; es narrativa literaria británica, no diálogo de película.
#
# La limpieza del módulo `corpus` reduce cada página a texto corrido: elimina navegación, tablas, leyendas e iconos del wiki, descarta etiquetas de personaje, descripciones de escena y encabezados, y conserva solo las líneas habladas. Alicia pasa por `strip_gutenberg_boilerplate`, que corta la licencia del inicio y del final.

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
print(corpora["shrek"][:400], "...")

# %% [markdown]
# ## 3. Texto sintético
#
# El pseudo diálogo sintético se genera con `markovify` mediante una cadena de Markov de orden 2: la probabilidad de la siguiente palabra depende del par de palabras previas. Cada oración se produce caminando la cadena desde un estado inicial hasta un marcador de fin de oración, con hasta 100 reintentos por oración.
#
# Decisiones de generación fijadas en el módulo `generator`:
#
# * **Tamaño objetivo:** la mitad de las palabras del original (`SYNTHETIC_TARGET_FRACTION = 0.5`, es decir, 4555 de 9110 palabras), con un margen de desviación de 10%: el texto final tiene 4558 palabras (desviación de 0.07%).
# * **Semilla 2026:** `markovify` 0.9.4 no acepta un generador aleatorio como argumento de `make_sentence`, así que se resembrica el módulo global `random` con 2026 antes de construir el modelo y antes de generar. Dos ejecuciones sobre el mismo corpus producen un texto idéntico por bytes.
# * **Sonda de viabilidad:** antes de generar, se intentan 500 oraciones con el modelo de orden 2; si menos de 100 salieran utilizables se reconstruiría el modelo con orden 1. No fue necesario: las 500 llamadas devolvieron oración.
#
# La siguiente celda reconstruye el modelo, regenera el texto sintético en memoria y lo compara con el archivo en disco (`data/synthetic/shrek_sintetico.txt`) para demostrar la reproducibilidad.

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
        {"Indicador": "Líneas que comparten sus primeras 4 palabras con otra línea", "Valor": f"{shared_prefixes}"},
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
# El texto sintético imita el vocabulario y la cadencia del original con oraciones en su mayoría gramaticales o casi gramaticales, pero sus 497 líneas son todas nuevas: ninguna es copia literal de una oración del guion original (0 duplicados literales). Sus patologías cualitativas se discuten en la sección 7.

# %% [markdown]
# ## 4. Representación vectorial
#
# **Bolsa de palabras (BOW).** Cada documento se representa como un vector de conteos $w_{t,d}$ sobre el vocabulario global: cuántas veces aparece el término $t$ en el documento $d$. La representación conserva las frecuencias pero pierde el orden de las palabras: "Shrek rescata a Fiona" y "Fiona rescata a Shrek" producen el mismo vector.
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
# El vocabulario no se limita a palabras sueltas: `ngram_range=(1, 3)` agrega bigramas y trigramas, y la lista de palabras vacías del inglés elimina términos gramaticales de alto uso que no distinguen documentos.

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
# La métrica captura la coincidencia de orientación entre documentos, independientemente de su longitud, y con vectores TF-IDF no negativos toma valores entre 0 y 1. La diagonal de la matriz vale 1.0 (cada texto consigo mismo, dentro de la precisión de punto flotante), la matriz es simétrica y el resto de los valores mide el traslape de vocabulario y n-gramas entre cada par.
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
# El orden es el que la intuición sugiere: el texto más parecido a Shrek es el pseudo diálogo construido con su propio vocabulario; sigue la secuela de la misma franquicia, con sus personajes y expresiones recurrentes; después otra película de animación del mismo tipo de humor pero sin vocabulario compartido; y al final una novela británica de 1865, con registro literario y narrado en lugar de dialogado.
#
# La elección del rango de n-gramas cambia las magnitudes pero no el orden. La siguiente celda repite la medición con solo unigramas (el resto de los parámetros idénticos) como verificación de robustez.

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
# La similitud coseno TF-IDF crece con el traslape de vocabulario y de n-gramas entre los textos, y el ranking medido lo refleja con claridad. Contra Shrek (2001), el texto sintético encabeza con 0.2144: no es sorprendente, porque las 4558 palabras del pseudo diálogo provienen en su totalidad del propio Shrek. Sus términos distintivos (ogre, swamp, Lord Farquaad) y sus bigramas y trigramas característicos reaparecen con pesos TF-IDF altos en ambos vectores, de modo que apuntan en direcciones parecidas. Y esto ocurre aunque ninguna de sus 497 oraciones sea copia literal de una oración del original: el traslape es léxico, no de frases copiadas.
#
# Shrek 2 ocupa el segundo lugar con 0.0827. Es una secuela de la misma franquicia, con los mismos personajes, sus nombres propios y sus expresiones recurrentes, así que comparte con el original mucho vocabulario específico que no aparece en ningún otro texto del corpus. Toy Story baja a 0.0474: comparte género (animación, comedia, público infantil) y registro de diálogo cinematográfico, incluidas palabras propias de comedia rápida como yes, hey, come on, pero casi no comparte vocabulario distintivo; además, al venir de un guion de IMSDb conserva algunas acotaciones de escena. Alicia queda última con 0.0393: es narrativa literaria británica del siglo 19, con un registro sin diálogo moderno y un vocabulario que solo se cruza con el de Shrek en el inglés más común, precisamente el que la lista de palabras vacías elimina.
#
# La magnitud, en cambio, depende fuertemente de la configuración de vectorización. En clase se revisó una escala de referencia calibrada: 1.0 para una copia idéntica, cerca de 0.89 para una paráfrasis, cerca de 0.55 para un traslape parcial y 0 para dos textos sin términos comunes. Los valores medidos aquí quedan muy por debajo de esos valores de referencia: el par más parecido de toda la actividad alcanza apenas 0.2144. La razón es la dimensionalidad. Con n-gramas de 1 a 3 y cinco documentos, el vocabulario se multiplica por los bigramas y trigramas que solo aparecen en un documento; un n-grama compartido por dos textos distintos se vuelve un evento raro y las normas L2 se reparten entre muchas más dimensiones, así que los cosenos se comprimen hacia 0. La verificación de robustez con solo unigramas lo confirma: las similitudes contra Shrek suben a 0.5217, 0.3029, 0.1912 y 0.1763, es decir, entre 2.4 y 4.5 veces más, y la escala calibrada de la clase corresponde a un escenario de dimensionalidad baja como ese. El orden del ranking es idéntico en ambas configuraciones, pero las magnitudes no son comparables entre configuraciones distintas.
#
# La consecuencia práctica es que un umbral absoluto (por ejemplo, "similitud mayor a 0.5 indica paráfrasis") solo tiene sentido calibrado dentro de una misma configuración. El fenómeno de compresión que se discutió en clase, la tendencia de las magnitudes a desvanecerse hacia cero en espacios de alta dimensionalidad, es visible aquí: bajo la configuración enviada, todos los pares que no involucran al corpus base Shrek quedan entre 0.018 y 0.047, y contra Shrek solo el texto sintético (0.2144) y Shrek 2 (0.0827) superan 0.05. Por eso los parámetros incluyen `sublinear_tf=True` y `max_df=0.9`, que amortiguan las dos causas principales: términos con frecuencias muy desbalanceadas dentro de un documento y términos omnipresentes que engordan el vector sin distinguir nada. La lectura correcta de estos resultados es comparar entre sí los valores de un mismo ranking, con cortes calibrados a partir de las propias mediciones, y no aplicar la escala de referencia de forma absoluta.

# %% [markdown]
# ### 7.2 ¿Una similitud alta garantiza que dos textos son del mismo autor?
#
# No, y la propia actividad contiene el contraejemplo. El texto más parecido a Shrek en toda la matriz es el texto sintético, con 0.2144, delante de la secuela oficial de la misma franquicia. Y no lo escribió ningún autor: lo escribió Markovify, una cadena de Markov de orden 2 que remuestrea el vocabulario del original sin intención, estilo ni creatividad. Con compartir el vocabulario basta para encabezar el ranking.
#
# La razón es que la similitud coseno sobre TF-IDF mide la dirección de un vector de frecuencias, no el estilo ni la intención. Dos textos son similares si usan términos semejantes en proporciones semejantes, y nada más. La bolsa de palabras pierde el orden: "Shrek rescata a Fiona" y "Fiona rescata a Shrek" producen exactamente el mismo vector, pese a significar lo contrario. La métrica es sorda a la sintaxis, a la estructura narrativa, al humor y a cualquier decisión autoral que no deje huella en las frecuencias de términos.
#
# En la otra dirección, dos textos del mismo autor pueden quedar lejos en este ranking, y dos textos de autores distintos pueden quedar cerca. Shrek 2 (0.0827) fue escrita por un equipo de guionistas distinto al de la película original, y aun así supera a Toy Story (0.0474) y a Alicia (0.0393) contra el primer Shrek: su cercanía proviene de compartir franquicia, personajes y expresiones, no de compartir autor. La similitud TF-IDF mide parentesco temático y léxico, no autoría.
#
# La atribución de autoría necesita señales distintas: la distribución de frecuencias de las palabras función (the, of, and, las muletillas), el ritmo de las oraciones, las marcas de puntuación y los hábitos tipográficos, y modelos diseñados para eso, como el de Khmelev citado en los recursos de la actividad, que clasifica autoría a partir de la distribución de frecuencias de palabras. Esas señales son, además, notablemente estables frente a que un autor cambie de tema, justo lo contrario de TF-IDF, que premia el tema común.
#
# Vale una precisión sobre el texto sintético: que 0 de sus 497 oraciones sean duplicados literales de oraciones del original no significa que el texto no repita. Markovify sí repite sus propias oraciones: 80 líneas del archivo sintético aparecen más de una vez y 192 comparten las primeras cuatro palabras con otra línea del mismo archivo. No hay copia del original, pero sí repetición interna; otra señal de que se trata de un fenómeno estadístico y no de una obra.

# %% [markdown]
# ### 7.3 Limitaciones y decisiones
#
# **La compresión de magnitudes es la principal limitación de la métrica tal como está configurada y se discutió en el punto 7.1.** En espacios de alta dimensionalidad los valores de similitud coseno se desvanecen hacia cero, y aquí el efecto es fuerte: bajo n-gramas de 1 a 3, todos los pares que no involucran al corpus Shrek quedan entre 0.018 y 0.047. Además de las mitigaciones de parámetros ya citadas (`sublinear_tf=True` para amortiguar frecuencias dominantes y `max_df=0.9` para eliminar términos omnipresentes), la decisión recomendada es leer los rankings en términos relativos: en un ranking de candidatos, el orden y las distancias entre posiciones aportan más información que el valor absoluto contra un umbral fijo. La verificación con solo unigramas complementa esa lectura al mostrar que el orden es un resultado estable mientras que las magnitudes son un artefacto de la configuración.
#
# **Por qué TF-IDF y no Word2Vec o embeddings neuronales.** TF-IDF es interpretable: cada peso del vector corresponde a un término identificable y cada resultado se puede explicar diciendo qué términos comparten dos textos. Corre en CPU con cinco documentos en fracciones de segundo. La pregunta de la actividad (qué tan parecidos son dos textos por su vocabulario y sus n-gramas) es una pregunta de frecuencias, y para responderla una métrica de frecuencias es suficiente y directa: un embedding semántico añade costo, capacidad de error y opacidad sin cambiar la pregunta, y con 9110 palabras de corpus de entrenamiento disponible en el proyecto sería además inviable entrenar un modelo de calidad. Aplicar aquí el principio de KISS validado en clase: la solución más sencilla que responde la pregunta es la correcta.
#
# **Transparencia metodológica: tres sustituciones de fuente.** El pipeline requirió sustituir tres fuentes respecto a lo que pedía la actividad, y todas están documentadas en el módulo `corpus`:
#
# 1. La URL de transcripción de Shrek indicada en la actividad ya no existe; se usó la página canónica de transcripción del mismo wiki de la franquicia.
# 2. La transcripción de Toy Story de transcripts.fandom fue retirada; se usó el guion canónico de IMSDb, que conserva algunas acotaciones de escena junto al diálogo.
# 3. Las peticiones directas a páginas /wiki/ del wiki de Fandom devuelven un error 403 por la protección de Cloudflare; se consultó la action API de MediaWiki del mismo wiki, que entrega el HTML parseado sin bloquear.
#
# **Reproducibilidad de la generación.** `markovify` 0.9.4 no acepta un argumento de generador aleatorio en `make_sentence` (su cadena obtiene los números del módulo global `random`), así que la reproducibilidad se logró resembrando `random.seed(2026)` al inicio tanto de la construcción del modelo como de la generación. La sección 3 lo demuestra: una regeneración completa en memoria coincide por bytes con el archivo en disco. La regla de respaldo definida, reconstruir con tamaño de estado 1 si la sonda de 500 oraciones devolviera menos de 100 utilizables, nunca se activó: el corpus sostiene el estado 2 sin dificultad (500 de 500 oraciones en la sonda).
#
# **Patologías del texto sintético, documentadas con honestidad.** El texto generado no es impecable y conviene decirlo. Presenta bucles raros de repetición de palabras propios de las cadenas de orden 2 (en la sección 3 se imprime el ejemplo "I tell you that you that you can find any stairs.", donde la cadena entra en el ciclo "that you / you that"), deriva interrogativa (oraciones que empiezan como enunciado y terminan en signo de pregunta, porque la cadena une el inicio de una oración del original con el final de otra) y repetición interna de oraciones completas (80 líneas duplicadas de 497, 192 de ellas compartiendo sus primeras cuatro palabras con otra línea, medidos en la tabla de la sección 3). Ninguno de estos defectos invalida la actividad, pero muestran los límites reales de un modelo generativo tan simple y explican por qué el texto sintético imita el léxico del original pero no su coherencia narrativa.
#
# **Otras limitaciones menores.** La lista de palabras vacías del inglés es fija y elimina términos que en un guion de comedia sí cargan estilo (well, hey), de modo que parte del material del estilo de los guiones queda fuera de la métrica, y el conteo de oraciones del corpus original, usado en la sección 3 para la comparación de duplicados literales, depende de una separación por signos de puntuación que no es perfecta. La métrica también es insensible a la longitud: la normalización L2 hace que Alicia (26524 palabras) y Shrek (9110 palabras) sean comparables, a costa de ignorar la información de tamaño.
