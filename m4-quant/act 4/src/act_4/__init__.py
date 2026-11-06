"""Actividad 4.5: Similitud en textos mediante TF-IDF y similitud coseno."""

from pathlib import Path

import pandas as pd

from act_4 import generator
from act_4.corpus import load_corpus
from act_4.similarity import (
    LABEL_SHREK,
    SYNTHETIC_FILE,
    SYNTHETIC_SUBDIR,
    build_document_set,
    compute_similarity_matrix,
    plot_heatmap,
    plot_ranked_bar,
    rank_against,
)

#: Directory holding the report figures, relative to the activity root.
ASSETS_DIR = Path("assets")

#: Directory holding the cleaned corpora, relative to the activity root.
RAW_DIR = Path("data") / "raw"

#: Heatmap figure written by the pipeline.
HEATMAP_FIG = ASSETS_DIR / "similitud_heatmap.png"

#: Ranked bar figure written by the pipeline.
RANKED_BAR_FIG = ASSETS_DIR / "similitud_ordenada.png"


def main() -> None:
    """Run the full activity pipeline: corpora, synthetic text, similarity."""
    print("Actividad 4.5: Similitud en textos mediante TF-IDF y similitud coseno")

    # 1. Load the cleaned corpora (idempotent: fetches only when missing).
    raw_dir = RAW_DIR if RAW_DIR.is_dir() else Path("..") / RAW_DIR
    for name in ("shrek", "shrek_2", "toy_story", "alicia"):
        load_corpus(name, raw_dir)

    # 2. Build the synthetic corpus if it is missing on disk.
    synthetic_path = SYNTHETIC_SUBDIR / SYNTHETIC_FILE
    if not synthetic_path.exists() or synthetic_path.stat().st_size == 0:
        shrek_text = load_corpus("shrek", raw_dir)
        model, state_size_used = generator.build_model(shrek_text)
        synthetic = generator.generate_synthetic(
            model, generator.target_word_count(shrek_text)
        )
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_path.write_text(synthetic, encoding="utf-8")
        print(
            f"Texto sintetico generado (estado {state_size_used}, "
            f"{len(synthetic.split())} palabras) en {synthetic_path}"
        )
    else:
        words = len(synthetic_path.read_text(encoding="utf-8").split())
        print(f"Texto sintetico en disco: {synthetic_path} ({words} palabras)")

    # 3. Compute the similarity matrix over the five documents.
    data_dir = Path(".") if RAW_DIR.is_dir() else Path("..")
    documents = build_document_set(data_dir)
    matrix = compute_similarity_matrix(documents)
    ranking = rank_against(LABEL_SHREK, matrix)

    print(f"\nMatriz de similitud coseno (TF-IDF, n-gramas de 1 a 3):")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(matrix)

    print(f"\nRanking de similitud contra el corpus {LABEL_SHREK}:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(ranking)

    # 4. Save both figures.
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plot_heatmap(matrix, HEATMAP_FIG)
    plot_ranked_bar(ranking, RANKED_BAR_FIG)
    print(f"\nFiguras guardadas en {HEATMAP_FIG} y {RANKED_BAR_FIG}")


if __name__ == "__main__":
    main()
