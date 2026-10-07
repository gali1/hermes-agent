"""Format-native, information-preserving folds for tool output.

Every fold here is a pure function of its input and keeps the output looking
like its own type (logs stay logs, listings stay listings). Each fold ships
with an exact inverse and :func:`fold_lossless` verifies the round-trip at
runtime: if the inverse does not reproduce the pre-fold text, or the result
is not strictly smaller, the pre-fold text is kept. Nothing here raises --
:func:`fold_lossless` returns its input unchanged on any error.

The folds are information-preserving by construction, not by heuristic:

* :func:`collapse_runs` replaces a run of N (N >= 2) identical consecutive
  lines with the line once plus ``... (repeated N times)``. The count is in
  the marker, so the original is exactly reconstructible.
* :func:`fold_repeated_blocks` replaces a block of K >= 2 consecutive lines
  that exactly reproduces an earlier region with
  ``... (repeats K lines from D lines back)``. Both coordinates are in
  original lines and the block never overlaps its anchor (K <= D), so the
  referenced region is already reconstructed when the marker is expanded.
* :func:`fold_path_listing` hoists a shared parent directory out of a pure
  path listing into a ``dir/`` heading line; the basenames beneath it
  re-prefix losslessly.

ANSI SGR (color) escape sequences are non-semantic and stripped one-way;
they are the only intentionally dropped bits.
"""

from __future__ import annotations

import re

__all__ = [
    "strip_ansi",
    "collapse_runs",
    "expand_runs",
    "fold_repeated_blocks",
    "unfold_repeated_blocks",
    "fold_path_listing",
    "path_unheading",
    "fold_lossless",
]

# ANSI CSI SGR (color/style) escape sequences: ESC [ ... m.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Run-collapse marker. The count is captured for exact inversion.
_RUN_MARKER_RE = re.compile(r"^\.\.\. \(repeated (\d+) times\)$")

# Multi-line block back-reference marker. Length and distance are in ORIGINAL
# lines; everything before a marker expands to the exact original prefix, so
# `distance` lines back in the expanded output is the block's first
# occurrence.
_BLOCK_MARKER_RE = re.compile(r"^\.\.\. \(repeats (\d+) lines from (\d+) lines back\)$")

# fold_repeated_blocks search bounds: minimum/maximum block length worth a
# marker, candidate anchors per line, and an input size cap so the scan stays
# negligible on huge payloads.
_FOLD_MIN_BLOCK = 2
_FOLD_MAX_BLOCK = 64
_FOLD_MAX_CANDIDATES = 8
_FOLD_MAX_LINES = 20_000

# A whole-line file path: optional ``./``/``../`` root, >= 1 directory
# segment, then a basename. No whitespace or ':' so mixed/log lines are not
# mistaken for listing rows. Directory-only lines (trailing '/') do not match
# (empty basename), which keeps the fold unambiguous.
_PATH_ROW_RE = re.compile(r"^(?P<dir>(?:\.{0,2}/)?(?:[^/\s:]+/)+)(?P<base>[^/\s:]+)$")


def strip_ansi(text: str) -> str:
    """Remove ANSI CSI/SGR (color) escape sequences. Color is non-semantic."""
    if not isinstance(text, str):
        return text
    return _ANSI_RE.sub("", text)


def _split_keep_trailing(text: str) -> tuple[list[str], bool]:
    """Split into lines, remembering whether a trailing newline was present."""
    if text == "":
        return [], False
    had_trailing = text.endswith("\n")
    body = text[:-1] if had_trailing else text
    return body.split("\n"), had_trailing


def _join(lines: list[str], had_trailing: bool) -> str:
    out = "\n".join(lines)
    if had_trailing:
        out += "\n"
    return out


def collapse_runs(text: str) -> str:
    """Collapse runs of >= 2 identical consecutive lines.

    A run of N (N >= 2) identical lines becomes the line once followed by
    ``... (repeated N times)``. Exact inverse: :func:`expand_runs`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i
        while j + 1 < n and lines[j + 1] == lines[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= 2:
            out.append(lines[i])
            out.append(f"... (repeated {run_len} times)")
        else:
            out.append(lines[i])
        i = j + 1
    return _join(out, had_trailing)


def expand_runs(text: str) -> str:
    """Exact inverse of :func:`collapse_runs`."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if i + 1 < n:
            m = _RUN_MARKER_RE.match(lines[i + 1])
            if m:
                count = int(m.group(1))
                out.extend([line] * count)
                i += 2
                continue
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def _remember(positions: dict[str, list[int]], line: str, index: int) -> None:
    """Track recent original positions of ``line``, bounded per distinct line."""
    bucket = positions.setdefault(line, [])
    bucket.append(index)
    if len(bucket) > _FOLD_MAX_CANDIDATES:
        del bucket[0]


def fold_repeated_blocks(text: str) -> str:
    """Collapse multi-line blocks that repeat earlier content into back-refs.

    A run of K (K >= 2) consecutive lines that exactly reproduces K lines seen
    D lines earlier becomes ``... (repeats K lines from D lines back)``.
    Coordinates are in original lines and the fold is only taken when the
    block does not overlap its anchor (K <= D), so on expansion the referenced
    region is always already reconstructed. Exact inverse:
    :func:`unfold_repeated_blocks`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    n = len(lines)
    if n < _FOLD_MIN_BLOCK * 2 or n > _FOLD_MAX_LINES:
        return text
    positions: dict[str, list[int]] = {}
    out: list[str] = []
    i = 0
    while i < n:
        best_len = 0
        best_dist = 0
        for q in reversed(positions.get(lines[i], ())):
            max_len = min(_FOLD_MAX_BLOCK, n - i, i - q)
            length = 0
            while length < max_len and lines[q + length] == lines[i + length]:
                length += 1
            if length > best_len:
                best_len = length
                best_dist = i - q
        if best_len >= _FOLD_MIN_BLOCK:
            marker = f"... (repeats {best_len} lines from {best_dist} lines back)"
            block_chars = sum(len(lines[i + k]) + 1 for k in range(best_len))
            if block_chars > len(marker) + 1:
                out.append(marker)
                for k in range(best_len):
                    _remember(positions, lines[i + k], i + k)
                i += best_len
                continue
        _remember(positions, lines[i], i)
        out.append(lines[i])
        i += 1
    return _join(out, had_trailing)


def unfold_repeated_blocks(text: str) -> str:
    """Exact inverse of :func:`fold_repeated_blocks`."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    for line in lines:
        m = _BLOCK_MARKER_RE.match(line)
        if m:
            length, dist = int(m.group(1)), int(m.group(2))
            start = len(out) - dist
            if start >= 0 and length <= dist:
                out.extend(out[start : start + length])
                continue
        out.append(line)
    return _join(out, had_trailing)


def fold_path_listing(text: str) -> str:
    """Fold a pure file-path listing (``find`` / ``ls -1`` / ``rg -l`` output).

    Each parent directory is printed once on its own line (ending in ``/``),
    then the bare basenames beneath it. Requires >= 2 path rows or there is
    nothing to group. Non-matching lines pass through verbatim. The exact
    inverse is :func:`path_unheading`; :func:`fold_lossless` verifies the
    round-trip and discards the fold on any mismatch.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if sum(1 for ln in lines if _PATH_ROW_RE.match(ln)) < 2:
        return text
    out: list[str] = []
    current: str | None = None
    for line in lines:
        m = _PATH_ROW_RE.match(line)
        if m:
            d = m.group("dir")
            if d != current:
                out.append(d)
                current = d
            out.append(m.group("base"))
        else:
            out.append(line)
            current = None
    return _join(out, had_trailing)


def path_unheading(text: str) -> str:
    """Exact inverse of :func:`fold_path_listing`.

    A *heading* is a line ending in ``/`` immediately followed by a basename
    row (a non-empty line with no ``/``); it is consumed and re-prefixed onto
    each following basename row until a blank line or another heading.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        is_base = line != "" and "/" not in line
        if current is not None and is_base:
            out.append(current + line)
            i += 1
            continue
        if line.endswith("/") and i + 1 < n and lines[i + 1] != "" and "/" not in lines[i + 1]:
            current = line
            i += 1
            continue
        current = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def fold_lossless(text: str) -> str:
    """Apply every lossless fold, self-verified at each step.

    Order: ANSI strip, :func:`collapse_runs`, :func:`fold_repeated_blocks`,
    :func:`fold_path_listing`. Each fold is adopted only when its exact
    inverse reproduces the pre-fold text and the candidate is strictly
    smaller. Deterministic; returns ``text`` unchanged on any error.
    """
    if not isinstance(text, str) or not text:
        return text
    try:
        result = strip_ansi(text)
        for fold, inverse in (
            (collapse_runs, expand_runs),
            (fold_repeated_blocks, unfold_repeated_blocks),
            (fold_path_listing, path_unheading),
        ):
            candidate = fold(result)
            if candidate == result:
                continue
            try:
                if inverse(candidate) != result:
                    continue
            except Exception:
                continue
            if len(candidate) < len(result):
                result = candidate
        return result
    except Exception:
        return text
