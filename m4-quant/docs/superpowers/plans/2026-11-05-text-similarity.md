# Actividad 4.5: Similitud en textos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure TF-IDF cosine similarity between an original film-script corpus and a Markovify-generated synthetic corpus, compare both against three other text bodies, and deliver the reflection as a zip on Canvas.

**Architecture:** A new `act 4/` workspace mirrors `act 1`–`act 3`: a docx/pdf report authored as a jupytext percent-format notebook in Spanish, reusable logic in `src/act_4/` (corpus fetching and cleaning, Markovify synthesis, TF-IDF vectorization and cosine similarity), figures in `assets/`, and deliverables in `export/` (docx, pdf, standalone script, and the submission zip).

**Tech Stack:** Python 3.13 (uv), markovify, scikit-learn (`TfidfVectorizer`, `cosine_similarity`), pandas, numpy, matplotlib, seaborn, jupytext/jupyter, pandoc with Lua filter for export.

**Spec:** Actividad 4.5 of TC3002B module 4, due 5 Nov 2026, delivered as a zip containing: (1) cuerpo de interés, (2) texto sintético, (3) documento en Word con la reflexión, and per the professor's clarification, the code as well. Rubric: cuerpo 10%, sintético 10%, reflexión 80%. Any methodology is valid if justified; the report must argue why the method was chosen, and must document any failure or fallback tried. The evaluation centers on the analysis, not the implementation.

## Global Constraints

- Follow act 1–3 layout exactly: `pyproject.toml` (name `act-4`, `requires-python = ">=3.13"`, `[project.scripts] act-4 = "act_4:main"`, uv_build backend), `src/act_4/`, `notebooks/`, `scripts/`, `assets/`, `data/raw/`, `export/`, `README.md`, `.python-version`.
- No test suite: consistent with act 1–3 and the author's workflow. Verification is by execution with printed/inspected output at each step.
- The report is written in Spanish. Header per act 3: Curso "Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 570)", Profesor "Luis Ricardo Peña Llamas", Alumno "Fernando Daniel Monroy Sánchez (A01750536)", date 5 de noviembre de 2026.
- No hyphenated wording in the report or code comments beyond established library names (use "TF-IDF" for the exact technique name, otherwise prefer plain wording).
- Random seeds pinned to `2026` everywhere; TF-IDF parameters fixed per task 4 so results reproduce.
- Filenames contain no accented characters.
- Notebook is the single source of truth for the report; the standalone export script mirrors it without importing `act_4`.
- Corpus is the author's choice and it is deliberately playful but justified: the original Shrek script. The justification: a single-writer-team film script with a distinct vocabulary register (fairy-tale parody), where the Markovify synthetic text provides a built-in counterexample for the authorship question because it scores high while having no human author.

## Review Focus

- **Thin corpus for Markovify.** The Shrek transcript is ~6–8k words; `state_size=2` may yield few complete sentences. Mitigation pinned in task 3: repeated `make_sentence` calls with a seeded model, fallback to `state_size=1` if fewer than 100 sentences are produced, and the fallback documented in the report. Verify: synthetic txt has at least the pinned word count target.
- **Polluted download.** Fandom transcript pages embed navigation, image captions, and wiki chrome. Task 2 must strip everything outside the transcript body and record word counts before/after cleaning for the report table. Verify: cleaned text starts and ends with dialogue, spot-check first and last 200 characters.
- **Vocabulary leakage between fit and transform.** The similarity matrix must come from a single `fit_transform` over all five documents; fitting only on the original would make cross-document similarities meaningless. Verify: matrix diagonal is exactly 1.0 and matrix is symmetric.
- **High-dimensionality distortion.** With tens of thousands of n-gram features, cosine values can crowd toward zero, as discussed in class. Task 4 pins `sublinear_tf=True` (log-scaled term frequencies) and `max_df=0.9`; the reflection must discuss this limitation. Verify: off-diagonal similarities of the synthetic text still clearly exceed those of unrelated corpora.
- **Zip completeness.** The grader opens only the zip. Task 6 lists its single source of contents. Verify: `unzip -l` output shows only files needed: original corpus, synthetic text, the standalone script, and the Word document.

---

### Task 1: Workspace

**Files:**
- Create: `act 4/pyproject.toml`
- Create: `act 4/.python-version`
- Create: `act 4/README.md`
- Create: `act 4/src/act_4/__init__.py`
- Create: `act 4/scripts/preamble.tex`
- Create: `act 4/scripts/hide_code_cells.lua`

**Interfaces:**
- Consumes: nothing.
- Produces: a uv-syncable package `act-4` importing as `act_4`, with a `main` entry point in `src/act_4/__init__.py` that prints a one-line placeholder message (replaced in task 5).

- [ ] **Step 1: Copy `act 3/pyproject.toml` to `act 4/pyproject.toml` and adjust**

Name `act-4`, description "Actividad 4.5: Similitud en textos mediante TF-IDF y similitud coseno". Dependencies: keep `ipykernel`, `jupyter`, `jupytext`, `matplotlib`, `numpy`, `pandas`, `seaborn`, `tabulate`; add `markovify>=0.9.3`, `scikit-learn>=1.7.0` (`beautifulsoup4>=4.13.0` is needed for the fandom HTML cleaning in task 2, add it here); drop `simpy`, `scipy`, `great-tables` unless table output needs them (act 3 used `great-tables` for report tables, so keep `great-tables`).

- [ ] **Step 2: Copy `scripts/preamble.tex` and `scripts/hide_code_cells.lua` unchanged from `act 3/scripts/`**

- [ ] **Step 3: Copy `act 3/.python-version` and write `README.md` following act 3's structure**

Same section order as act 3: title `# Actividad 4.5: Similitud en textos`, project structure, mathematical foundations (BOW, TF-IDF with the $TF \times IDF$ formula and $IDF = \log(N/df(t))$, $\cos(\theta) = \frac{u \cdot v}{\|u\|\|v\|}$, Markov chains for generation), and a `uv sync` / `uv run jupyter lab` execution section.

- [ ] **Step 4: Write `src/act_4/__init__.py` with a placeholder `main()` that prints a one-line placeholder message**

- [ ] **Step 5: Sync and verify**

Run: `cd "act 4" && uv sync && uv run act-4`
Expected: dependencies resolve and the placeholder message prints.

- [ ] **Step 6: Commit**

```bash
git add "act 4"
git commit
```

Commit message: `feat(quant): initialize act 4 workspace for text similarity`. Follow the repository convention: plain `git commit` is replaced by a heredoc commit message ending with `Generated by Mistral Vibe.` and `Co-Authored-By: Mistral Vibe <vibe@mistral.ai>`.

---

### Task 2: Corpora

**Files:**
- Create: `act 4/src/act_4/corpus.py`
- Create: `act 4/data/raw/shrek_original.txt`
- Create: `act 4/data/raw/shrek_2.txt`
- Create: `act 4/data/raw/toy_story.txt`
- Create: `act 4/data/raw/alicia.txt`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SCRIPT_SOURCES: dict[str, str]` mapping corpus name to URL (Shrek transcript from `https://shrek.fandom.com/wiki/Transcript_(Shrek)`, Shrek 2 from the same wiki family, Toy Story from `https://transcripts.fandom.com/wiki/Toy_Story`; the executor verifies these URLs and swaps in the canonical transcript pages if they redirect).
  - `ALICE_URL: str = "https://www.gutenberg.org/files/11/11-0.txt"` (Project Gutenberg, Alice's Adventures in Wonderland, chosen as the fourth body: a literary fairy tale that shares Shrek's fairy-tale vocabulary while differing in register, author, and medium; it strengthens the reflection on what similarity does and does not capture).
  - `fetch_corpus(name: str) -> Path` — idempotent download to `data/raw/`, skips if the file already exists and is non-empty.
  - `clean_transcript(raw_html: str) -> str` — strips wiki chrome (nav, tables, image captions), extracts dialogue lines only, joins them into flowing sentence text.
  - `strip_gutenberg_boilerplate(raw_text: str) -> str` — removes the license header and footer.
  - `load_corpus(name: str, data_dir: Path) -> str` — returns cleaned final text.

- [ ] **Step 1: Implement `corpus.py` with the interfaces above**

Fetch via `python` standard `urllib.request` (network runs under `uv run`, not in the sandbox). HTML parsing with `BeautifulSoup`. The key cleaning decision: the fandom transcripts render dialogue as a sequence of table rows or list lines tagged with the character name; extract the quoted text and join with spaces so Markovify and TF-IDF see continuous dialogue. Word-count check happens in step 2.

- [ ] **Step 2: Fetch all four corpora and verify content quality**

Run: `cd "act 4" && uv run python -c "from pathlib import Path; from act_4 import corpus; names = ['shrek', 'shrek_2', 'toy_story', 'alicia']; print({n: len(corpus.load_corpus(n, Path('data/raw')).split()) for n in names})"`
Expected: `shrek_original` cleans to roughly 5–8k words, the other two scripts comparable, and `alicia` ~26k words after boilerplate stripping. The file size for the raw Shrek HTML must drop substantially after cleaning.
If a chosen URL does not exist or the page layout defeats extraction, fall back to the sibling canonical transcript source and note the substitution in the report's methodology section.

- [ ] **Step 3: Spot-check the cleaned Shrek text**

Run: `head -c 400 "act 4/data/raw/shrek_original.txt" && tail -c 400 "act 4/data/raw/shrek_original.txt"`
Expected: both ends show dialogue text, no HTML tags, no wiki menus, no character-name clutter beyond the transcript's own formatting.

- [ ] **Step 4: Commit**

Commit message: `feat(quant): fetch and clean film script and literary corpora`.

---

### Task 3: Synthetic Text

**Files:**
- Create: `act 4/src/act_4/generator.py`
- Create: `act 4/data/synthetic/shrek_sintetico.txt`

**Interfaces:**
- Consumes: `load_corpus` from task 2.
- Produces:
  - `build_model(text: str, state_size: int = 2, seed: int = 2026) -> tuple[markovify.text.Text, int]` — returns the fitted model and the state size actually used, so callers can report the fallback; falls back internally to `state_size=1` when fewer than 100 sentences can be drawn at state size 2.
  - `generate_synthetic(model: markovify.text.Text, target_words: int, seed: int = 2026) -> str` — loop `make_sentence` until the target word count is reached, retrying with `tries` above default.
  - `SYNTHETIC_TARGET_FRACTION: float = 0.5`

- [ ] **Step 1: Implement `generator.py` with the interfaces above**

- [ ] **Step 2: Generate `shrek_sintetico.txt` and verify size**

Run: `cd "act 4" && uv run python -c "from pathlib import Path; from act_4 import corpus, generator; original = corpus.load_corpus('shrek', Path('data/raw')); model, size = generator.build_model(original); synth = generator.generate_synthetic(model, len(original.split()) // 2); Path('data/synthetic/shrek_sintetico.txt').write_text(synth); print(len(synth.split()))"`
Expected: word count close to half the original (`target_words`), file present, content visibly dialogue-like.

- [ ] **Step 3: Inspect 30 random sentences qualitatively**

Expected: mostly grammatical or near-grammatical English lines drawing on Shrek's vocabulary. Note the qualitative character of the output and any pathological short-sentence behavior for the report. The synthetic text is short (roughly 4k words), which is fine: comparing a ~4k-word synthetic body to a ~7k-word original is the correct asymmetrical scenario for the reflection.

- [ ] **Step 4: Commit**

Commit message: `feat(quant): generate synthetic shrek corpus via markovify`.

---

### Task 4: Similarity

**Files:**
- Create: `act 4/src/act_4/similarity.py`
- Create: `act 4/assets/similitud_heatmap.png`
- Create: `act 4/assets/similitud_ordenada.png`

**Interfaces:**
- Consumes: `load_corpus` (task 2), synthetic text file (task 3).
- Produces:
  - `VECTORIZER_PARAMS: dict = {"lowercase": True, "stop_words": "english", "ngram_range": (1, 3), "max_df": 0.9, "min_df": 1, "sublinear_tf": True, "norm": "l2"}` — `ngram_range` (1,3) per the class guidance to include 2–3-token n-grams for similarity tasks, capturing multi-word expressions while keeping unigram coverage; `stop_words="english"` is a parameter documented in class.
  - `build_document_set(data_dir: Path) -> dict[str, str]` with Spanish display labels, ordered: "Shrek", "Shrek Sintético", "Shrek 2", "Toy Story", "Alicia".
  - `compute_similarity_matrix(documents: dict[str, str]) -> pandas.DataFrame` — single `fit_transform` over all documents, `sklearn.metrics.pairwise.cosine_similarity` on the result, DataFrame indexed by label.
  - `rank_against(reference: str, matrix: pandas.DataFrame) -> pandas.Series` — descending similarity against one reference.
  - `plot_heatmap(matrix: pandas.DataFrame, dest: Path) -> None` and `plot_ranked_bar(series: pandas.DataFrame, dest: Path) -> None` — seaborn figure saved at 300 dpi to `assets/`, styled per `act 3` conventions (whitegrid, edge color `#CBD5E0`).

- [ ] **Step 1: Implement `similarity.py` with the interfaces above**

All labels Spanish; both figures rendered by these functions, not ad hoc notebook code.

- [ ] **Step 2: Compute the matrix and verify invariants**

Run: `cd "act 4" && uv run python -c "...print(compute_similarity_matrix(build_document_set(Path('.'))))"`
Expected: diagonal exactly 1.0, symmetric matrix, synthetic opening high but below 1.0, "Alicia" lowest against Shrek. Shrek 2 vs Shrek similar to or below synthetic-vs-original; if the synthetic text does not rank in the top 2 against Shrek, note it for discussion in the reflection rather than adjusting parameters. If all off-diagonal values crowd below 0.05, report the high-dimensionality distortion hypothesis and test the unigram-only `ngram_range=(1,1)` variant as a robustness check, printing both matrices side by side.

- [ ] **Step 3: Verify figure assets exist and render correctly**

Run: `ls "act 4/assets"`
Expected: `similitud_heatmap.png` and `similitud_ordenada.png` non-trivial in size and readable when opened.

- [ ] **Step 4: Commit**

Commit message: `feat(quant): compute tf-idf cosine similarity matrix across corpora`.

---

### Task 5: Report

**Files:**
- Modify: `act 4/src/act_4/__init__.py` (replace placeholder `main`)
- Create: `act 4/notebooks/04_text_similarity.py`
- Create: `act 4/notebooks/04_text_similarity.ipynb` (generated)

**Interfaces:**
- Consumes: `corpus.py`, `generator.py`, `similarity.py` (tasks 2–4), `assets/` figures.
- Produces: act-graded reflection as a paired jupytext notebook mirroring `act 3/notebooks/03_mm1_queue_simulation.py` in structure and format.

- [ ] **Step 1: Write the notebook percent script with the following section skeleton**

```
1. Introducción — propósito de la actividad y justificación del corpus elegido
2. Cuerpo de texto — Shrek (2001) transcript, limpieza, conteo de palabras por corpus
3. Texto sintético — Markovify con cadenas de Markov, estado 2, semilla 2026
4. Representación vectorial — BOW, TF-IDF con la formula y parámetros exactos usados
5. Similitud coseno — definición de la métrica, matriz completa, mapa de calor, ranking
6. Comparación contra otros cuerpos — Shrek 2, Toy Story, Alicia
7. Reflexión — developed below
```

Section 7 (80% of the grade) must answer, in order, with the actual computed values cited:
1. ¿Cómo varía la medida de similitud en función del texto que se compara? — explain the observed ranking: same-franchise script scores highest, same-genre animated script scores mid, literary fairy tale scores lowest; argue from shared vocabulary, register, and n-gram overlap; state how the choice of n-gram range and stop word list changes the measurement; cite the calibrated scale from class (1.0 copia, ~0.89 paráfrasis, ~0.55 traslape parcial, 0 textos sin términos comunes).
2. ¿Una similitud alta garantiza que dos textos son del mismo autor? — No, and the synthetic corpus is the counterexample: it ranks near the top against the original though Markovify, not an author, wrote it. Expand: TF-IDF coseno mide dirección de un vector de frecuencias, no estilo ni intención (Bag of Words pierde el orden); texts by different authors in the same franchise can score as high as same-author texts; authorship attribution requires different signals (palabras función, ritmo, marcas de puntuación) and modelos como los de Khmelev citados en los recursos de la actividad.
3. Limitaciones y decisiones — high dimensionality can crowd cosine values toward zero (the class discussion of vanishing magnitudes; why `sublinear_tf=True` and `max_df=0.9` mitigate it); why TF-IDF and not Word2Vec or a neural embedding (interpretable, CPU-only, sufficient for a frequency-based metric, KISS as validated in class); what was tried and what failed, honestly reported (any fallbacks or URL substitutions from tasks 2–3 belong here).

- [ ] **Step 2: Pair and render the notebook**

Run: `cd "act 4" && uv run jupytext --set-formats ipynb,py:percent notebooks/04_text_similarity.py && uv run jupytext --sync notebooks/04_text_similarity.py && uv run jupyter nbconvert --execute --inplace notebooks/04_text_similarity.ipynb`
Expected: ipynb generated and executed without cell errors, figures embedded, tables printed.

- [ ] **Step 3: Replace `main()` in `src/act_4/__init__.py` with a runner that executes the full pipeline**

`main()` should: load corpora, build the synthetic text if missing, compute the matrix, print the similarity DataFrame and ranking, and save both figures. The standalone export in task 6 will mirror these outputs for the code submission.

- [ ] **Step 4: Verify runner end to end**

Run: `cd "act 4" && uv run act-4`
Expected: full pipeline prints the matrix table to stdout without errors.

- [ ] **Step 5: Commit**

Commit message: `feat(quant): write text similarity report notebook`.

---

### Task 6: Deliverables

**Files:**
- Create: `act 4/export/Actividad_4_5_A01750536.py` (standalone script)
- Create: `act 4/export/Actividad_4_5_A01750536.docx`
- Create: `act 4/export/Actividad_4_5_A01750536.pdf`
- Create: `act 4/export/Actividad_4_5_A01750536.zip`

**Interfaces:**
- Consumes: the executed notebook (task 5), `data/raw/*.txt`, `data/synthetic/shrek_sintetico.txt`.
- Produces: the Canvas submission zip.

- [ ] **Step 1: Write the standalone script `export/Actividad_4_5_A01750536.py` mirroring `act 3`'s export script style**

Docstring header (activity 4.5, student, professor, course ordered as in act 3 standalone), self-contained logic (no `act_4` imports, paths resolved relative to its own location via `Path(__file__).parents[1]`), loading the corpus and synthetic files from their on-disk locations, recomputing the vectorization parameters imported verbatim from task 4, reproducing the cosine matrix, ranked series, and both figures, and printing all tables inside `main()`. A grader must be able to run it against the shipped data files.

- [ ] **Step 2: Verify the standalone script runs outside the package**

Run: `cd "act 4/export" && uv run --project .. python Actividad_4_5_A01750536.py`
Expected: same similarity matrix and ranking values as task 5 printed, figures written to a local output directory.

- [ ] **Step 3: Export the report to docx and pdf**

Run the same pandoc pipeline used for act 3: from `act 4/`, first strip notebook cell metadata if pandoc fails on it, then
`pandoc notebooks/04_text_similarity.ipynb -o export/Actividad_4_5_A01750536.docx --lua-filter=scripts/hide_code_cells.lua` and the same input with `-o export/Actividad_4_5_A01750536.pdf --pdf-engine=xelatex -H scripts/preamble.tex --lua-filter=scripts/hide_code_cells.lua`
Expected: both files build; code cell inputs suppressed in both outputs (lua filter), figures and tables preserved.

- [ ] **Step 4: Assemble the zip**

Contents exactly:
```
Actividad_4_5_A01750536.zip
├── shrek_original.txt        (copy of data/raw/shrek_original.txt, cleaned corpus)
├── shrek_sintetico.txt       (copy of data/synthetic/shrek_sintetico.txt)
├── Actividad_4_5_A01750536.docx
└── Actividad_4_5_A01750536.py (standalone script)
```
Wrapped in a top folder `Actividad_4_5_A01750536/` if previous activities' zips used one; match the convention found in `act 1`–`act 3` git history (`git show 2bd3d00 --stat`). English filenames, no accents.

- [ ] **Step 5: Verify the zip**

Run: `unzip -l "act 4/export/Actividad_4_5_A01750536.zip"`
Expected: only the four files from step 4, each non-zero in size.

- [ ] **Step 6: Commit**

Commit message: `feat(quant): export activity 4.5 docx, standalone script, and delivery zip`.

---

## Test Ideas (post-implementation, author's choice)

Per the author's workflow these are not implemented; offered for consideration only:
- P0: `compute_similarity_matrix` invariants — diagonal 1.0, symmetry, range [0, 1] — one assertion block.
- P0: `generate_synthetic` respects `target_words` within a tolerance and always returns non-empty text.
- P1: `clean_transcript` drops every HTML tag and wiki navigation class on a fixture of the real page.
- P2: full pipeline reproduction — fixed seed produces byte-identical similarity matrix across two runs.
