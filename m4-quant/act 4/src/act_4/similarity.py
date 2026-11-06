"""TF-IDF vectorization and cosine similarity for the text similarity activity.

Turns the five corpora of the activity (the cleaned film dialogues, the
Markov-generated synthetic Shrek text and the Project Gutenberg edition of
"Alice's Adventures in Wonderland") into a single TF-IDF matrix with one
fit of :class:`sklearn.feature_extraction.text.TfidfVectorizer` and scores
every pair of documents with cosine similarity, producing the matrix and
figures consumed by the activity report.

Deterministic: TF-IDF fitting and cosine similarity involve no randomness,
so repeated runs over the same on-disk corpora yield identical results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from act_4.corpus import load_corpus

#: Viewer-facing Spanish label of the original Shrek dialogue corpus.
LABEL_SHREK: str = "Shrek"

#: Viewer-facing Spanish label of the Markov-generated synthetic corpus.
LABEL_SYNTHETIC: str = "Shrek Sintético"

#: Viewer-facing Spanish labels, in canonical analysis order.
DOCUMENT_LABELS: tuple[str, ...] = (
    LABEL_SHREK,
    LABEL_SYNTHETIC,
    "Shrek 2",
    "Toy Story",
    "Alicia",
)

#: Cleaned corpus name (for :func:`act_4.corpus.load_corpus`) behind each
#: raw label, in the same order as :data:`DOCUMENT_LABELS`.
_RAW_CORPORA: dict[str, str] = {
    LABEL_SHREK: "shrek",
    "Shrek 2": "shrek_2",
    "Toy Story": "toy_story",
    "Alicia": "alicia",
}

#: File holding the synthetic corpus, relative to ``data/synthetic``.
SYNTHETIC_FILE: Path = Path("shrek_sintetico.txt")

#: Subdirectory holding the cleaned corpora, relative to the activity root.
RAW_SUBDIR: Path = Path("data") / "raw"

#: Subdirectory holding the synthetic corpus, relative to the activity root.
SYNTHETIC_SUBDIR: Path = Path("data") / "synthetic"

#: TfidfVectorizer configuration for the similarity analysis. ``ngram_range``
#: (1, 3) follows the class guidance to include 2-3-token n-grams for
#: similarity tasks, capturing multi-word expressions while keeping unigram
#: coverage; ``stop_words="english"`` is a parameter documented in class.
VECTORIZER_PARAMS: dict = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 3),
    "max_df": 0.9,
    "min_df": 1,
    "sublinear_tf": True,
    "norm": "l2",
}

#: Act 3 figure conventions: theme, font scale and spine color.
_THEME_KWARGS: dict = {"style": "whitegrid"}
_FONT_SCALE: float = 0.95
_SPINE_COLOR: str = "#CBD5E0"
_SPINE_LINEWIDTH: float = 0.8
_DPI: int = 300


def _apply_style() -> None:
    """Set the act 3 seaborn/matplotlib style for similarity figures."""
    sns.set_theme(**_THEME_KWARGS, font_scale=_FONT_SCALE)
    plt.rcParams["axes.edgecolor"] = _SPINE_COLOR
    plt.rcParams["axes.linewidth"] = _SPINE_LINEWIDTH


def build_document_set(data_dir: Path = Path(".")) -> dict[str, str]:
    """Return the five analysis documents keyed by Spanish display label.

    ``data_dir`` is the activity root containing ``data/raw`` (cleaned
    corpora) and ``data/synthetic`` (the synthetic corpus). The returned
    dict preserves the canonical order of :data:`DOCUMENT_LABELS`.
    """
    documents: dict[str, str] = {}
    for label in DOCUMENT_LABELS:
        if label == LABEL_SYNTHETIC:
            synthetic_path = data_dir / SYNTHETIC_SUBDIR / SYNTHETIC_FILE
            documents[label] = synthetic_path.read_text(encoding="utf-8")
        else:
            documents[label] = load_corpus(
                _RAW_CORPORA[label], data_dir / RAW_SUBDIR
            )
    return documents


def compute_similarity_matrix(documents: dict[str, str]) -> pd.DataFrame:
    """Compute pairwise cosine similarities between the TF-IDF vectors.

    A single :class:`TfidfVectorizer` fit (``fit_transform``) builds one
    shared vocabulary over all documents; cosine similarity over the rows
    of the resulting matrix yields the square similarity DataFrame indexed
    and columned by document label.
    """
    labels = list(documents)
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    tfidf = vectorizer.fit_transform([documents[label] for label in labels])
    similarities = cosine_similarity(tfidf)
    return pd.DataFrame(similarities, index=labels, columns=labels)


def rank_against(reference: str, matrix: pd.DataFrame) -> pd.Series:
    """Return similarities against ``reference`` in descending order.

    The reference document itself is included with its (exact) similarity
    of 1.0; callers drop it when plotting comparisons.
    """
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


def plot_ranked_bar(series: pd.Series | pd.DataFrame, dest: Path) -> None:
    """Save the similarity ranking against one reference as an ordered
    horizontal bar chart, excluding the reference's self-similarity.
    """
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
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
    _style_axes(ax)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(dest), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax) -> None:
    """Re-apply the act 3 spine style that seaborn theming may override."""
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
        spine.set_linewidth(_SPINE_LINEWIDTH)
