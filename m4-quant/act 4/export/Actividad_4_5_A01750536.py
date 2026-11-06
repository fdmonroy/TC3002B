"""Actividad 4.5: Similitud en textos mediante TF-IDF y similitud coseno.

Alumno: Fernando Daniel Monroy Sánchez (A01750536)
Profesor: Luis Ricardo Peña Llamas
Curso: Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 570)
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#: Activity root, resolved from this script's location in ``export/`` via
#: ``Path(__file__).parents[1]`` (the directory that holds ``data/``).
ACTIVITY_DIR = Path(__file__).resolve().parents[1]

#: Subdirectory holding the cleaned corpora, relative to the activity root.
RAW_DIR = ACTIVITY_DIR / "data" / "raw"

#: Subdirectory holding the synthetic corpus, relative to the activity root.
SYNTHETIC_DIR = ACTIVITY_DIR / "data" / "synthetic"

#: File holding the synthetic corpus, relative to ``data/synthetic``.
SYNTHETIC_FILE = "shrek_sintetico.txt"

#: Local output directory for the reproduced figures, next to this script.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

#: Viewer-facing Spanish labels, in canonical analysis order.
DOCUMENT_LABELS = (
    "Shrek",
    "Shrek Sintético",
    "Shrek 2",
    "Toy Story",
    "Alicia",
)

#: Cleaned corpus file behind each raw label, in the same order.
CORPUS_FILES = {
    "Shrek": "shrek_original.txt",
    "Shrek 2": "shrek_2.txt",
    "Toy Story": "toy_story.txt",
    "Alicia": "alicia.txt",
}

#: TfidfVectorizer configuration, copied verbatim from the activity (task 4).
VECTORIZER_PARAMS = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 3),
    "max_df": 0.9,
    "min_df": 1,
    "sublinear_tf": True,
    "norm": "l2",
}

#: Figure conventions shared with the activity report: theme, spine, dpi.
_THEME_KWARGS = {"style": "whitegrid"}
_FONT_SCALE = 0.95
_SPINE_COLOR = "#CBD5E0"
_SPINE_LINEWIDTH = 0.8
_DPI = 300


def _apply_style() -> None:
    """Set the seaborn/matplotlib style used by the activity figures."""
    sns.set_theme(**_THEME_KWARGS, font_scale=_FONT_SCALE)
    plt.rcParams["axes.edgecolor"] = _SPINE_COLOR
    plt.rcParams["axes.linewidth"] = _SPINE_LINEWIDTH


def build_document_set() -> dict[str, str]:
    """Return the five analysis documents keyed by display label.

    Loads the four cleaned raw corpora from ``data/raw`` and the
    Markov-generated synthetic Shrek text from ``data/synthetic``,
    preserving the canonical order of ``DOCUMENT_LABELS``.
    """
    documents: dict[str, str] = {}
    for label in DOCUMENT_LABELS:
        if label == "Shrek Sintético":
            path = SYNTHETIC_DIR / SYNTHETIC_FILE
        else:
            path = RAW_DIR / CORPUS_FILES[label]
        documents[label] = path.read_text(encoding="utf-8")
    return documents


def compute_similarity_matrix(documents: dict[str, str]) -> pd.DataFrame:
    """Compute pairwise cosine similarities between the TF-IDF vectors.

    A single ``fit_transform`` builds one shared vocabulary over all five
    documents; cosine similarity over the rows of the resulting matrix
    yields the square similarity DataFrame.
    """
    labels = list(documents)
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    tfidf = vectorizer.fit_transform([documents[label] for label in labels])
    similarities = cosine_similarity(tfidf)
    return pd.DataFrame(similarities, index=labels, columns=labels)


def rank_against(reference: str, matrix: pd.DataFrame) -> pd.Series:
    """Return similarities against ``reference`` in descending order."""
    if reference not in matrix.columns:
        raise ValueError(
            f"Unknown reference: {reference!r}. Known: {list(matrix.columns)}"
        )
    series = matrix[reference].copy().sort_values(ascending=False)
    series.name = reference
    return series


def plot_heatmap(matrix: pd.DataFrame, dest: Path) -> None:
    """Save the annotated similarity heatmap to ``dest`` at 300 dpi."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Similitud coseno"},
        ax=ax,
    )
    ax.set_title("Matriz de similitud coseno (TF-IDF)", pad=12)
    ax.set_xlabel("Documento")
    ax.set_ylabel("Documento")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(dest), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ranked_bar(series: pd.Series, dest: Path) -> None:
    """Save the ranking against one reference as an ordered horizontal
    bar chart, excluding the reference's self-similarity.
    """
    reference = str(series.name)
    if reference in series.index:
        series = series.drop(reference)
    series = series.sort_values(ascending=True)

    _apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(
        series.index.astype(str),
        series.values,
        color="#2B6CB0",
        edgecolor=_SPINE_COLOR,
        linewidth=_SPINE_LINEWIDTH,
    )
    for y, value in enumerate(series.values):
        ax.text(value + 0.002, y, f"{value:.3f}", va="center", fontsize=9)
    ax.set_title(f"Similitud coseno con «{reference}» (TF-IDF)", pad=12)
    ax.set_xlabel("Similitud coseno")
    ax.set_ylabel("Documento")
    ax.set_xlim(0.0, max(0.05, float(series.max()) * 1.2))
    sns.despine(ax=ax, left=True)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
        spine.set_linewidth(_SPINE_LINEWIDTH)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(dest), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    documents = build_document_set()
    matrix = compute_similarity_matrix(documents)
    ranking = rank_against("Shrek", matrix)

    print("Matriz de similitud coseno (TF-IDF, n-gramas de 1 a 3):")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(matrix)

    print("\nRanking de similitud contra el corpus Shrek:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(ranking)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_dest = OUTPUT_DIR / "similitud_heatmap.png"
    ranked_bar_dest = OUTPUT_DIR / "similitud_ordenada.png"
    plot_heatmap(matrix, heatmap_dest)
    plot_ranked_bar(ranking, ranked_bar_dest)
    print(f"\nFiguras guardadas en {heatmap_dest} y {ranked_bar_dest}")


if __name__ == "__main__":
    main()
