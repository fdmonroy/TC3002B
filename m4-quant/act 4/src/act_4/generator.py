"""Synthetic text generation for the text similarity activity.

Builds a Markov chain model over the cleaned Shrek dialogue corpus with
markovify and generates a synthetic corpus whose size is
:data:`SYNTHETIC_TARGET_FRACTION` of the original word count.

Reproducibility: markovify 0.9.4's ``Text.make_sentence`` has no ``rng``
argument and draws from the global :mod:`random` stream, so the global
generator is re-seeded with the same seed at the start of
:func:`build_model` and :func:`generate_synthetic`. Two runs over the same
on-disk corpus therefore produce byte-identical synthetic text.
"""

import random

import markovify

#: Fraction of the original corpus word count used as the synthetic target.
SYNTHETIC_TARGET_FRACTION: float = 0.5

#: Seed used for both model construction and generation.
DEFAULT_SEED: int = 2026

#: ``make_sentence`` attempts per sentence (markovify's default is 10).
GENERATION_TRIES: int = 100

#: Minimum number of usable sentences required to keep ``state_size=2``.
MIN_USABLE_SENTENCES: int = 100

#: ``make_sentence`` calls used to probe whether a state size is usable.
PROBE_ATTEMPTS: int = 500

#: Maximum word-count overshoot above ``target_words``, as a fraction of it.
MAX_OVERSHOOT_FRACTION: float = 0.10

#: Consecutive failed sentence draws tolerated before giving up early.
_MAX_NONE_STREAK: int = 100


def _count_usable_sentences(model: markovify.text.Text, attempts: int) -> int:
    """Count the sentences ``model`` can actually produce in ``attempts`` tries.

    ``make_sentence`` returns ``None`` when it cannot assemble a sentence
    that passes its output checks within its retry budget; a state size that
    mostly fails this signals a corpus too small for that state size.
    """
    usable = 0
    for _ in range(attempts):
        if model.make_sentence(tries=GENERATION_TRIES) is not None:
            usable += 1
    return usable


def build_model(
    text: str, state_size: int = 2, seed: int = DEFAULT_SEED
) -> tuple[markovify.text.Text, int]:
    """Fit a Markov model on ``text``, falling back to a smaller state size.

    Marks the global :mod:`random` generator with ``seed`` (model
    construction itself is deterministic, but the usability probe that
    follows draws random sentences). The model is first built at
    ``state_size``; if fewer than :data:`MIN_USABLE_SENTENCES` sentences can
    be drawn from it (checked by attempting :data:`PROBE_ATTEMPTS`
    ``make_sentence`` calls), the model is rebuilt at ``state_size=1``.

    Returns the fitted model and the state size actually used, so callers
    can report the fallback.
    """
    random.seed(seed)
    model = markovify.Text(text, state_size=state_size)
    if _count_usable_sentences(model, PROBE_ATTEMPTS) >= MIN_USABLE_SENTENCES:
        return model, state_size
    fallback = 1
    model = markovify.Text(text, state_size=fallback)
    return model, fallback


def generate_synthetic(
    model: markovify.text.Text, target_words: int, seed: int = DEFAULT_SEED
) -> str:
    """Draw sentences from ``model`` until ``target_words`` words are reached.

    The loop repeatedly calls ``model.make_sentence(tries=100)`` (a retry
    budget above markovify's default of 10) and accumulates the returned
    sentences. Two stopping rules apply:

    1. The target is reached: the loop stops once the accumulated word
       count is at least ``target_words``. The final sentence may push the
       total above the target, but the loop is abandoned first if the next
       sentence would overshoot by more than 10% of ``target_words``, so the
       total stays within ``[target_words * 0.9, target_words * 1.1)``.
    2. Drawing stalls: after 100 consecutive ``make_sentence`` failures the
       loop returns what it has instead of spinning forever.

    Marks the global :mod:`random` generator with ``seed`` first, so the
    generated text is reproducible for a given model. Sentences are returned
    joined by single newlines, one sentence per line.
    """
    if target_words <= 0:
        raise ValueError(f"target_words must be positive, got {target_words}")
    random.seed(seed)
    max_overshoot = max(1, round(MAX_OVERSHOOT_FRACTION * target_words))
    sentences: list[str] = []
    word_count = 0
    none_streak = 0
    while word_count < target_words:
        sentence = model.make_sentence(tries=GENERATION_TRIES)
        if sentence is None:
            none_streak += 1
            if none_streak >= _MAX_NONE_STREAK:
                break
            continue
        none_streak = 0
        words = len(sentence.split())
        if word_count + words > target_words + max_overshoot:
            break
        sentences.append(sentence)
        word_count += words
    return "\n".join(sentences)


def target_word_count(text: str) -> int:
    """Return the synthetic generation target for ``text``.

    :data:`SYNTHETIC_TARGET_FRACTION` of the corpus word count, truncated.
    """
    return int(len(text.split()) * SYNTHETIC_TARGET_FRACTION)
