"""Adaptive keep-count sizing via information saturation (Headroom-derived).

Pure, stdlib-only (``zlib`` + ``hashlib``), deterministic. Ports Headroom's
``transforms/adaptive_sizer.py``: items are ranked, a cumulative unique-bigram
coverage curve is built, and a Kneedle-style knee gives the point where more
items stop adding new information. A fast path covers tiny inputs and a zlib
ratio check validates the result.

Input items may be plain strings, ``(id, score)`` pairs, or dicts carrying
``text``/``content``/``id`` plus an optional ``score``. Items are processed in
score order (descending, stable), so callers can pass selection candidates in
any order.
"""

from __future__ import annotations

import hashlib
import json
import zlib

__all__ = [
    "compute_optimal_k",
    "compute_unique_bigram_curve",
    "count_unique_simhash",
    "find_knee",
]

_FAST_PATH_N = 8
_SIMHASH_HAMMING_THRESHOLD = 3
_ZLIB_MIN_BYTES = 200
_ZLIB_TOLERANCE = 0.15


def _item_text(item) -> tuple[str, float]:
    """Return ``(text, score)`` for a string, pair, or dict item."""
    if isinstance(item, str):
        return item, 0.0
    if isinstance(item, dict):
        for key in ("text", "content", "id", "name"):
            value = item.get(key)
            if isinstance(value, str):
                break
        else:
            try:
                value = json.dumps(item, sort_keys=True, default=str)
            except (TypeError, ValueError):
                value = str(item)
        score = item.get("score", 0.0)
        return value, float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        score = item[1]
        return str(item[0]), (
            float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0
        )
    return str(item), 0.0


def compute_optimal_k(items, min_k: int = 5, max_k: int = 30) -> int:
    """Return how many of ``items`` to keep, in ``[min_k, max_k]``.

    ``items`` is a sequence of strings, ``(id, score)`` pairs, or dicts; it is
    re-ordered by descending score (stable) before analysis. Deterministic and
    never raises: unparseable items fall back to their ``str()`` form.
    """
    try:
        return _compute_optimal_k(items, int(min_k), int(max_k))
    except Exception:
        return max(0, min(len(items), int(max_k))) if items else 0


def _compute_optimal_k(items, min_k: int, max_k: int) -> int:
    scored = [_item_text(item) for item in items]
    scored = [(text, score, pos) for pos, (text, score) in enumerate(scored)]
    scored.sort(key=lambda row: (-row[1], row[2]))
    texts = [row[0] for row in scored]

    n = len(texts)
    effective_max = max_k if max_k is not None else n
    if n <= _FAST_PATH_N:
        return min(n, effective_max)

    unique_count = count_unique_simhash(texts)
    if unique_count <= 3:
        return min(max(min_k, unique_count), effective_max)

    curve = compute_unique_bigram_curve(texts)
    knee = find_knee(curve)
    diversity_ratio = unique_count / n

    if knee is None:
        keep_fraction = 0.3 + 0.7 * diversity_ratio
        knee = max(min_k, int(n * keep_fraction))
    elif diversity_ratio > 0.7:
        diversity_floor = max(min_k, int(n * (0.3 + 0.7 * diversity_ratio)))
        knee = max(knee, diversity_floor)

    k = max(min_k, int(knee))
    k = min(k, effective_max)
    k = _validate_with_zlib(texts, k, effective_max)
    return max(min_k, min(k, effective_max))


def find_knee(curve: list[int]) -> int | None:
    """Index count at the knee of a monotonically increasing curve, or None.

    Kneedle: normalize to [0, 1] and take the point of maximum deviation above
    the ``y = x`` diagonal. Returns a 1-based count (the knee index + 1), or
    None when the curve hugs the diagonal (no saturation).
    """
    n = len(curve)
    if n < 3:
        return None
    x_min, x_max = 0, n - 1
    y_min, y_max = curve[0], curve[-1]
    if y_max == y_min:
        return 1
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_diff = -1.0
    knee_idx = None
    for i in range(n):
        x_norm = (i - x_min) / x_range
        y_norm = (curve[i] - y_min) / y_range
        diff = y_norm - x_norm
        if diff > max_diff:
            max_diff = diff
            knee_idx = i
    if max_diff < 0.05:
        return None
    return knee_idx + 1 if knee_idx is not None else None


def _is_cjk_char(c: str) -> bool:
    o = ord(c)
    return (
        0x3040 <= o <= 0x30FF
        or 0x3400 <= o <= 0x4DBF
        or 0x4E00 <= o <= 0x9FFF
        or 0xAC00 <= o <= 0xD7AF
        or 0xF900 <= o <= 0xFAFF
    )


def compute_unique_bigram_curve(items) -> list[int]:
    """Cumulative unique word-bigram counts as items are seen in order."""
    seen: set[tuple[str, str]] = set()
    curve: list[int] = []
    for item in items:
        words = str(item).lower().split()
        if len(words) >= 2:
            for j in range(len(words) - 1):
                seen.add((words[j], words[j + 1]))
        elif words and len(words[0]) >= 2 and any(_is_cjk_char(c) for c in words[0]):
            w = words[0]
            for j in range(len(w) - 1):
                seen.add((w[j], w[j + 1]))
        else:
            seen.add((words[0] if words else "", ""))
        curve.append(len(seen))
    return curve


def _simhash(text: str) -> int:
    v = [0] * 64
    lowered = text.lower()
    for i in range(max(1, len(lowered) - 3)):
        gram = lowered[i : i + 4]
        h = int(
            hashlib.md5(gram.encode(), usedforsecurity=False).hexdigest()[:16], 16
        )
        for j in range(64):
            if h & (1 << j):
                v[j] += 1
            else:
                v[j] -= 1
    fingerprint = 0
    for j in range(64):
        if v[j] > 0:
            fingerprint |= 1 << j
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def count_unique_simhash(items, threshold: int = _SIMHASH_HAMMING_THRESHOLD) -> int:
    """Number of distinct content groups (greedy SimHash clustering)."""
    if not items:
        return 0
    fingerprints = [_simhash(str(item)) for item in items]
    clusters: list[int] = []
    for fp in fingerprints:
        if not any(_hamming_distance(fp, rep) <= threshold for rep in clusters):
            clusters.append(fp)
    return len(clusters)


def _validate_with_zlib(items, k: int, max_k: int, tolerance: float = _ZLIB_TOLERANCE) -> int:
    if k >= len(items) or k >= max_k:
        return k
    full_text = "\n".join(str(item) for item in items).encode()
    subset_text = "\n".join(str(item) for item in items[:k]).encode()
    if len(full_text) < _ZLIB_MIN_BYTES:
        return k
    full_compressed = len(zlib.compress(full_text, level=1))
    subset_compressed = len(zlib.compress(subset_text, level=1))
    full_ratio = full_compressed / len(full_text) if full_text else 1.0
    subset_ratio = subset_compressed / len(subset_text) if subset_text else 1.0
    if abs(full_ratio - subset_ratio) > tolerance:
        return min(int(k * 1.2), max_k)
    return k
