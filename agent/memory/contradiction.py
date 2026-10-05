"""Lexical contradiction detection between candidate memories.

Ported verbatim from the OpenCode/MemPalace Hindsight primitives (upstream
``plugins/memory/rekal/mempalace_hindsight.py``).  Stdlib only and pure.
"""

from __future__ import annotations

from agent.memory.dedup import _TRGM_WORD, trigram_similarity

# Surface markers of negation/reversal. Lexical detection cannot prove a
# contradiction, so callers treat a hit as a candidate for review (a
# `contradicts` link) rather than as grounds to delete or rewrite anything.
NEGATION_MARKERS = (
    "not ", "no longer ", "doesn't ", "does not ", "isn't ", "is not ",
    "aren't ", "are not ", "won't ", "will not ", "never ", "cannot ", "can't ",
    "removed ", "stopped ", "dropped ", "disabled ", "deprecated ", "reverted ",
    "instead of ", "rather than ", "switched from ", "moved away from ",
)

# Antonym pairs that flip a statement's meaning without any negation word.
POLARITY_PAIRS = (
    ("enable", "disable"), ("enabled", "disabled"), ("allow", "deny"),
    ("allowed", "denied"), ("add", "remove"), ("added", "removed"),
    ("include", "exclude"), ("start", "stop"), ("started", "stopped"),
    ("on", "off"), ("true", "false"), ("always", "never"),
    ("support", "unsupported"), ("prefer", "avoid"),
)

CONTRADICTION_MIN_SIMILARITY = 0.35
CONTRADICTION_MAX_SIMILARITY = 0.97


def _has_negation(text):
    padded = f" {text.lower()} "
    return any(marker in padded for marker in NEGATION_MARKERS)


def _polarity_conflict(a_low, b_low):
    a_words = set(_TRGM_WORD.findall(a_low))
    b_words = set(_TRGM_WORD.findall(b_low))
    for left, right in POLARITY_PAIRS:
        if (left in a_words and right in b_words) or (right in a_words and left in b_words):
            return True
    return False


def detect_contradiction(new_content, existing_content,
                         min_similarity=CONTRADICTION_MIN_SIMILARITY,
                         max_similarity=CONTRADICTION_MAX_SIMILARITY):
    """Decide whether two statements about the same subject disagree.

    Ported from the OpenCode/MemPalace Hindsight primitives.

    Returns `(is_contradiction, confidence)`.

    The gate is deliberately two-sided.  Texts must be similar enough to be
    about the same thing, but not so similar that they are effectively the same
    sentence -- near-identical text is a duplicate, which is handled by content
    hashing, not a contradiction.  Within that band, either an explicit negation
    marker on exactly one side or an antonym pair across the two is treated as
    a candidate conflict.
    """
    if not new_content or not existing_content:
        return False, 0.0

    similarity = trigram_similarity(new_content, existing_content)
    if similarity < min_similarity or similarity > max_similarity:
        return False, 0.0

    new_low = new_content.lower()
    existing_low = existing_content.lower()

    negation_differs = _has_negation(new_low) != _has_negation(existing_low)
    polarity_differs = _polarity_conflict(new_low, existing_low)

    if not negation_differs and not polarity_differs:
        return False, 0.0

    # Confidence rises with topical overlap and with agreement between the two
    # independent detectors.
    confidence = 0.5 + 0.3 * similarity
    if negation_differs and polarity_differs:
        confidence += 0.15
    return True, round(min(0.95, confidence), 3)
