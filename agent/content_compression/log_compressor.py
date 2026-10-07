"""Log/build-output compressor (Headroom-derived, stdlib-only).

Two tiers:

* **Always lossless** — repeated consecutive runs are collapsed with an exact
  count marker via :func:`agent.content_compression.lossless.collapse_runs`.
  Identical *adjacent* warnings are therefore collapsed losslessly; identical
  warnings that are not adjacent are only deduped in the lossy tier, because
  dropping a non-adjacent duplicate without a count is not information-
  preserving.
* **Lossy, gated** — only with ``allow_lossy`` AND a ``retrieval_hint``.
  Errors (max 10, first/last guaranteed) with +/- 3 context lines, stack
  traces (max 3; head 3 frames + up to 5 app frames), warnings (max 5,
  deduped), and pytest/npm/cargo summary lines are kept; the rest fill up to
  ``max_lines`` in order. A trailing ``... N lines omitted (retrieve: hint)``
  marker carries the retrieval path and the dropped count is returned.

Never raises; returns the input unchanged when nothing is smaller.
"""

from __future__ import annotations

import re

from .lossless import collapse_runs

__all__ = ["compress_log", "MAX_ERRORS", "ERROR_CONTEXT_LINES", "MAX_STACK_TRACES", "MAX_WARNINGS"]

MAX_ERRORS = 10
ERROR_CONTEXT_LINES = 3
MAX_STACK_TRACES = 3
MAX_WARNINGS = 5

_ERROR_RE = re.compile(
    r"\b(?:ERROR|FAIL(?:ED|URE)?|FATAL|CRITICAL)\b"
    r"|\b\w*(?:Error|Exception)\b"
    r"|^panic:|^fatal error:"
    r"|Traceback \(most recent call last\)",
    re.IGNORECASE,
)
_WARN_RE = re.compile(r"\bWARN(?:ING)?\b", re.IGNORECASE)
_SUMMARY_RE = re.compile(
    r"^\d+\s+(?:passed|failed|skipped|error|warning)s?\b"
    r"|^={3,}"
    r"|^-{3,}"
    r"|^(?:PASSED|FAILED|SKIPPED)\b",
    re.IGNORECASE,
)
_TRACE_START_RE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|^panic:"
    r"|^fatal error: "
    r"|^goroutine \d+ \["
    r"|^thread '[^']*' panicked at"
    r"|^stack backtrace:"
    r"|^Unhandled exception\."
    r"|^Caused by: "
)
_TRACE_LINE_RE = re.compile(
    r'^\s+File "'
    r"|^\s+at "
    r"|^\s+\S+\.(?:go|py|rs|java|cs):\d+"
    r"|^\s+\.\.\. \d+ more$"
    r"|^\s+\d+: "
    r"|^\t\S+\.go:\d+"
)
_FRAME_RE = re.compile(r'^\s+File "|^\s+at |^\s+\S+\.(?:go|py|rs|java|cs):\d+')
_RUNTIME_RE = re.compile(
    r"site-packages|/lib/python|/usr/lib/|node_modules|/rustlib/|\.rustup/|/go/src/"
)
_TRACE_HEAD_FRAMES = 3
_TRACE_APP_FRAMES = 5
_TRACE_MIN_LINES = _TRACE_HEAD_FRAMES + _TRACE_APP_FRAMES


def compress_log(
    text,
    *,
    retrieval_hint: str | None = None,
    allow_lossy: bool = False,
    max_lines: int = 100,
) -> tuple[str, int]:
    """Return ``(text, dropped_lines)``; never larger than the input.

    The lossless run-collapse always applies. The lossy selection runs only
    with ``allow_lossy`` AND ``retrieval_hint``.
    """
    if not isinstance(text, str) or not text:
        return text, 0
    try:
        base = collapse_runs(text)
    except Exception:
        base = text
    if not base or len(base) >= len(text):
        base = text

    if not (allow_lossy and retrieval_hint):
        return base, 0

    lines, trailing = _split_lines(base)
    n = len(lines)
    if n <= max_lines:
        return base, 0

    selected = _select_indices(lines, max_lines)
    dropped = n - len(selected)
    if dropped <= 0:
        return base, 0

    out = [lines[i] for i in sorted(selected)]
    out.append(f"... {dropped} lines omitted (retrieve: {retrieval_hint})")
    candidate = "\n".join(out) + ("\n" if trailing else "")
    if len(candidate) >= len(text):
        return text, 0
    return candidate, dropped


def _split_lines(text: str) -> tuple[list[str], bool]:
    if text == "":
        return [], False
    trailing = text.endswith("\n")
    body = text[:-1] if trailing else text
    return body.split("\n"), trailing


def _select_indices(lines: list[str], max_lines: int) -> set[int]:
    n = len(lines)
    selected: set[int] = set()

    error_indices = [i for i, line in enumerate(lines) if _ERROR_RE.search(line)]
    chosen_errors = _choose_errors(error_indices)
    selected.update(chosen_errors)
    for i in chosen_errors:
        for j in range(max(0, i - ERROR_CONTEXT_LINES), min(n, i + ERROR_CONTEXT_LINES + 1)):
            selected.add(j)

    for block in _trace_blocks(lines)[:MAX_STACK_TRACES]:
        selected.update(block)

    warnings: list[int] = []
    seen: set[str] = set()
    for i, line in enumerate(lines):
        if not _WARN_RE.search(line) or _ERROR_RE.search(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        warnings.append(i)
        if len(warnings) >= MAX_WARNINGS:
            break
    selected.update(warnings)

    selected.update(i for i, line in enumerate(lines) if _SUMMARY_RE.search(line))

    if len(selected) < max_lines:
        # Fill with ordinary lines only: re-adding error/warning/trace/summary
        # lines beyond their caps would silently undo the guarantees above.
        for i in range(n):
            if i not in selected and not _is_special_line(lines[i]):
                selected.add(i)
                if len(selected) >= max_lines:
                    break
    return selected


def _is_special_line(line: str) -> bool:
    return bool(
        _ERROR_RE.search(line)
        or _WARN_RE.search(line)
        or _SUMMARY_RE.search(line)
        or _TRACE_START_RE.search(line)
        or _TRACE_LINE_RE.search(line)
    )


def _choose_errors(error_indices: list[int]) -> list[int]:
    if not error_indices:
        return []
    chosen = [error_indices[0]]
    if error_indices[-1] != error_indices[0]:
        chosen.append(error_indices[-1])
    for i in error_indices:
        if len(chosen) >= MAX_ERRORS:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen


def _trace_blocks(lines: list[str]) -> list[set[int]]:
    """Indices of each stack-trace block, trimmed to head + app frames."""
    blocks: list[set[int]] = []
    n = len(lines)
    i = 0
    while i < n:
        if not _TRACE_START_RE.search(lines[i]):
            i += 1
            continue
        j = i + 1
        while j < n and lines[j].strip() and (
            _TRACE_LINE_RE.search(lines[j]) or lines[j][:1] in (" ", "\t")
        ):
            j += 1
        blocks.append(_trim_trace(lines[i:j], i))
        i = j
    return blocks


def _trim_trace(block: list[str], offset: int) -> set[int]:
    length = len(block)
    if length <= _TRACE_MIN_LINES:
        return set(range(offset, offset + length))
    keep = [0, 1, 2]
    candidates = [i for i in range(3, length) if _FRAME_RE.search(block[i])]
    app = [i for i in candidates if not _RUNTIME_RE.search(block[i])]
    chosen = app[:_TRACE_APP_FRAMES]
    if len(chosen) < _TRACE_APP_FRAMES:
        for i in candidates:
            if i not in chosen:
                chosen.append(i)
                if len(chosen) >= _TRACE_APP_FRAMES:
                    break
    if not chosen:
        chosen = list(range(3, min(length, 3 + _TRACE_APP_FRAMES)))
    keep.extend(chosen)
    return {offset + i for i in sorted(set(keep))}
