"""Grep/ripgrep search-result compressor (Headroom-derived, stdlib-only).

Parses ``path:line:content`` and ``path:line:col:content`` rows, groups them by
file, scores each row (error keywords +0.5/+0.4/+0.3, context words +0.3,
long/rare tokens +0.2, capped at 1.0), and keeps the top rows under a per-file
cap and a total cap chosen by :func:`compute_optimal_k` (clamped to
``max_total``). The first and last matches overall are always kept.

Compression is lossy and gated: without ``allow_lossy`` AND a
``retrieval_hint`` the input is returned byte-identical. When it runs, each
file's omitted rows become ``... N more matches in <path>`` and a single
trailing ``... N matches omitted (retrieve: <hint>)`` line carries the
retrieval path; the number of dropped matches is returned.
"""

from __future__ import annotations

import re

from .adaptive_sizer import compute_optimal_k

__all__ = ["compress_search", "parse_search_lines", "score_line"]

_SEARCH_LINE_RE = re.compile(
    r"^(?P<path>[^\s:]+):(?P<line>\d+):(?:(?P<col>\d+):)?(?P<content>.*)$"
)
_TIMESTAMP_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[T ]")

_ERROR_TIERS = (
    (
        re.compile(
            r"\b(?:error|fail|failed|failure|fatal|exception|panic|denied|timeout|refused)\b",
            re.IGNORECASE,
        ),
        0.5,
    ),
    (re.compile(r"\bwarn(?:ing)?\b", re.IGNORECASE), 0.4),
    (re.compile(r"\b(?:critical|abort|rejected)\b", re.IGNORECASE), 0.3),
)
_LONG_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{11,}")


def parse_search_lines(text: str) -> list[tuple[str, int, str, str]]:
    """Return ``(path, line_number, content, original_line)`` for match rows."""
    matches = []
    for line in text.split("\n"):
        if not line or _TIMESTAMP_RE.match(line):
            continue
        match = _SEARCH_LINE_RE.match(line)
        if not match:
            continue
        matches.append((match.group("path"), int(match.group("line")), match.group("content"), line))
    return matches


def score_line(content: str, context_words: frozenset[str] | set[str] = frozenset()) -> float:
    """Score a match line in ``[0, 1]``; deterministic, never raises."""
    score = 0.0
    for pattern, boost in _ERROR_TIERS:
        if pattern.search(content):
            score += boost
            break
    lowered = content.lower()
    for word in context_words:
        if word in lowered:
            score += 0.3
    if _LONG_TOKEN_RE.search(content):
        score += 0.2
    return min(1.0, score)


def compress_search(
    text,
    *,
    retrieval_hint: str | None = None,
    allow_lossy: bool = False,
    max_total: int = 30,
    max_per_file: int = 5,
    max_files: int = 15,
    context: str = "",
) -> tuple[str, int]:
    """Return ``(text, dropped_matches)``; never larger than the input."""
    if not isinstance(text, str) or not text:
        return text, 0
    if not (allow_lossy and retrieval_hint):
        return text, 0

    matches = parse_search_lines(text)
    if len(matches) < 2:
        return text, 0

    context_words = {w for w in (context or "").lower().split() if len(w) > 2}
    entries = [
        (idx, path, line_no, content, line, score_line(content, context_words))
        for idx, (path, line_no, content, line) in enumerate(matches)
    ]

    files: dict[str, list[tuple]] = {}
    for entry in entries:
        files.setdefault(entry[1], []).append(entry)

    first_index = entries[0][0]
    last_index = entries[-1][0]

    file_order = list(files)
    ranked_files = sorted(
        file_order,
        key=lambda path: (
            -sum(entry[5] for entry in files[path]),
            files[path][0][0],
        ),
    )
    kept_files = ranked_files[:max_files]

    adaptive_total = compute_optimal_k(
        [(entry[4], entry[5]) for entry in entries], min_k=5, max_k=max_total
    )

    selected: set[int] = set()
    for path in kept_files:
        rows = sorted(files[path], key=lambda entry: (-entry[5], entry[0]))
        for entry in rows[:max_per_file]:
            selected.add(entry[0])

    if len(selected) > adaptive_total:
        removable = sorted(
            (i for i in selected if i not in (first_index, last_index)),
            key=lambda i: (entries[i][5], i),
        )
        while len(selected) > adaptive_total and removable:
            selected.discard(removable.pop(0))
    selected.add(first_index)
    selected.add(last_index)

    # The forced first/last rows may push their file over the per-file cap;
    # evict that file's lowest-scoring non-forced rows to honor the cap.
    for path in file_order:
        rows = files[path]
        while True:
            chosen = [entry for entry in rows if entry[0] in selected]
            if len(chosen) <= max_per_file:
                break
            removable = sorted(
                (entry for entry in chosen if entry[0] not in (first_index, last_index)),
                key=lambda entry: (entry[5], entry[0]),
            )
            if not removable:
                break
            selected.discard(removable[0][0])

    if len(selected) >= len(entries):
        return text, 0

    out: list[str] = []
    dropped_total = 0
    for path in file_order:
        rows = files[path]
        kept = [entry for entry in rows if entry[0] in selected]
        dropped = [entry for entry in rows if entry[0] not in selected]
        out.extend(entry[4] for entry in kept)
        if dropped:
            out.append(f"... {len(dropped)} more matches in {path}")
            dropped_total += len(dropped)
    if dropped_total == 0:
        return text, 0
    out.append(f"... {dropped_total} matches omitted (retrieve: {retrieval_hint})")

    trailing = text.endswith("\n")
    candidate = "\n".join(out) + ("\n" if trailing else "")
    if len(candidate) >= len(text):
        return text, 0
    return candidate, dropped_total
