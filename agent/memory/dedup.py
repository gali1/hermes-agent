"""Trigram similarity, degeneracy filtering, and content deduplication.

Ported from the OpenCode/MemPalace Hindsight primitives (upstream
``plugins/memory/rekal/mempalace_hindsight.py``).  Stdlib only and pure.
"""

from __future__ import annotations

import re

_TRGM_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def trigram_set(text):
    """Padded per-word trigram set, matching PostgreSQL pg_trgm semantics."""
    trigrams = set()
    for word in _TRGM_WORD.findall((text or "").lower()):
        padded = f"  {word} "
        for i in range(len(padded) - 2):
            trigrams.add(padded[i : i + 3])
    return trigrams


def trigram_similarity(a, b):
    """Jaccard similarity over trigram sets, in [0,1]."""
    set_a = a if isinstance(a, set) else trigram_set(a)
    set_b = b if isinstance(b, set) else trigram_set(b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a) + len(set_b) - intersection
    return intersection / union if union else 0.0


_DEGENERATE = {"...", "…", "-", "--", "---", ".", "..", "•", "·", "*", "**", "***", "n/a", "na", "none", "null"}
_PUNCT_ONLY = set(".,;:!?-–—…\"'`´ \t\n\r")


def is_degenerate(text):
    """True for content that carries no recoverable meaning.

    Filters extraction artefacts before they occupy a memory slot and, worse,
    match every future query weakly through FTS prefix expansion.
    """
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.lower() in _DEGENERATE:
        return True
    if all(ch in _PUNCT_ONLY for ch in stripped):
        return True
    if len(stripped) <= 2 and not any(ch.isalnum() for ch in stripped):
        return True
    return False


def normalize_content(text):
    """Strip, collapse internal whitespace, and lowercase for comparison."""
    if not isinstance(text, str) or not text:
        return ""
    return " ".join(text.split()).lower()


def is_duplicate(a, b, threshold=0.97):
    """True when two texts are the same normalized content or near-identical."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    norm_a = normalize_content(a)
    norm_b = normalize_content(b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    return trigram_similarity(norm_a, norm_b) >= threshold
