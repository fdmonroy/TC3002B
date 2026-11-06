"""Actividad 4.5: Similitud en textos mediante TF-IDF y similitud coseno."""

from pathlib import Path

import pandas as pd

from act_4 import generator
from act_4.corpus import load_corpus
from act_4.similarity import (
    LABEL_SHREK,
    RAW_SUBDIR,
    SYNTHETIC_FILE,
    SYNTHETIC_SUBDIR,
    build_document_set,
    compute_similarity_matrix,
    plot_heatmap,
    plot_ranked_bar,
    rank_against,
)

#: Subdirectory holding the report figures, relative to the activity root.
ASSETS_SUBDIR = Path("assets")

#: Heatmap figure written by the pipeline, relative to :data:`ASSETS_SUBDIR`.
HEATMAP_FIG = Path("similitud_heatmap.png")

#: Ranked bar figure written by the pipeline, relative to :data:`ASSETS_SUBDIR`.
RANKED_BAR_FIG = Path("similitud_ordenada.png")


def _activity_root() -> Path:
    """Return the activity root, tolerating invocation one directory below it.

    The activity root is the directory that contains ``data/`` and
    ``assets/``. All runner paths (corpora, synthetic corpus, figures)
    are derived from this single root so a run from a subdirectory reads
    and writes the same files as a run from the root itself.
    """
    cwd = Path(".")
    if (cwd / RAW_SUBDIR).is_dir():
        return cwd
    parent = Path("..")
    if (parent / RAW_SUBDIR).is_dir():
        return parent
    raise FileNotFoundError(
        f"No se encontró el directorio {RAW_SUBDIR} ni en el directorio "
        f"actual ni en su padre; ejecute act-4 desde la raíz de la actividad."
    )


def main() -> None:
    """Run the full activity pipeline: corpora, synthetic text, similarity."""
    print("Actividad 4.5: Similitud en textos mediante TF-IDF y similitud coseno")

    root = _activity_root()
    raw_dir = root / RAW_SUBDIR
    assets_dir = root / ASSETS_SUBDIR
    synthetic_path = root / SYNTHETIC_SUBDIR / SYNTHETIC_FILE

    # 1. Load the cleaned corpora (idempotent: fetches only when missing).
    for name in ("shrek", "shrek_2", "toy_story", "alicia"):
        load_corpus(name, raw_dir)

    # 2. Build the synthetic corpus if it is missing on disk.
    if not synthetic_path.exists() or synthetic_path.stat().st_size == 0:
        shrek_text = load_corpus("shrek", raw_dir)
        model, state_size_used = generator.build_model(shrek_text)
        synthetic = generator.generate_synthetic(
            model, generator.target_word_count(shrek_text)
        )
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_path.write_text(synthetic, encoding="utf-8")
        print(
            f"Texto sintético generado (estado {state_size_used}, "
            f"{len(synthetic.split())} palabras) en {synthetic_path}"
        )
    else:
        words = len(synthetic_path.read_text(encoding="utf-8").split())
        print(f"Texto sintético en disco: {synthetic_path} ({words} palabras)")

    # 3. Compute the similarity matrix over the five documents.
    documents = build_document_set(root)
    matrix = compute_similarity_matrix(documents)
    ranking = rank_against(LABEL_SHREK, matrix)

    print("\nMatriz de similitud coseno (TF-IDF, n-gramas de 1 a 3):")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(matrix)

    print(f"\nRanking de similitud contra el corpus {LABEL_SHREK}:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(ranking)

    # 4. Save both figures.
    assets_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dest = assets_dir / HEATMAP_FIG
    ranked_bar_dest = assets_dir / RANKED_BAR_FIG
    plot_heatmap(matrix, heatmap_dest)
    plot_ranked_bar(ranking, ranked_bar_dest)
    print(f"\nFiguras guardadas en {heatmap_dest} y {ranked_bar_dest}")


if __name__ == "__main__":
    main()
