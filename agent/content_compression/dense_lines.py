"""Elide dense machine-generated lines (minified JS/CSS, base64, blobs).

Tool outputs that dump a fetched web page, a bundled asset or an encoded blob
arrive as a few very long lines with almost no whitespace. No structural fold
can help with them. This module keeps a head and a tail of each such line and
replaces the middle with a one-line marker.

This transform is LOSSY. Callers must only enable it when the original is
retrievable (the marker embeds a retrieval hint pointing at the persisted
original); :func:`agent.content_compression.compress_tool_output` refuses to
elide when no retrieval path is available.

A line is "dense" when it is long AND has a tiny fraction of spaces. Prose,
code, logs, indented JSON, CSV and markdown all have far more than 6% spaces,
so they are never touched. Tab-containing lines (TSV / ``psql -A``) and
JSON-shaped lines are skipped, as are lines carrying error signals
(:func:`agent.content_compression.protection.is_error_line`). Must-keep token
protection is deliberately NOT applied here: base64 blobs and minified
assets contain ALLCAPS runs and path-like spans, and eliding them is safe
because the marker keeps the original retrievable.
"""

from __future__ import annotations

from .protection import is_error_line

__all__ = [
    "MIN_LINE_CHARS",
    "MAX_SPACE_RATIO",
    "MIN_DENSE_TOTAL_CHARS",
    "HEAD_CHARS",
    "TAIL_CHARS",
    "is_dense_line",
    "elide_dense_lines",
]

MIN_LINE_CHARS = 300
MAX_SPACE_RATIO = 0.06
# A single dense line (a JWT, a signed URL, a PATH) is a value the agent asked
# for, not a dump: a block is only elided when its dense lines add up to at
# least this many chars. Real bundle dumps are tens of KB.
MIN_DENSE_TOTAL_CHARS = 2000
HEAD_CHARS = 160
TAIL_CHARS = 80


def is_dense_line(
    line: str,
    *,
    min_line_chars: int = MIN_LINE_CHARS,
    max_space_ratio: float = MAX_SPACE_RATIO,
) -> bool:
    """True when ``line`` is long, nearly whitespace-free, and unprotected.

    Tab-containing lines pass the space ratio but are data the agent asked
    for; minified assets and encoded blobs never carry tabs. A compact JSON
    value on one line is also nearly whitespace-free, but it is structured
    data another compressor handles; the elider must not pre-empt it.
    """
    if not isinstance(line, str):
        return False
    n = len(line)
    if n < min_line_chars or "\t" in line:
        return False
    if (line.count(" ") / n) > max_space_ratio:
        return False
    stripped = line.strip()
    if stripped[:1] in "{[" and stripped[-1:] in "}]":
        return False
    return not is_error_line(line)


def elide_dense_lines(
    text: str,
    *,
    retrieval_hint: str | None = None,
    min_line_chars: int = MIN_LINE_CHARS,
    max_space_ratio: float = MAX_SPACE_RATIO,
    min_dense_total_chars: int = MIN_DENSE_TOTAL_CHARS,
    head_chars: int = HEAD_CHARS,
    tail_chars: int = TAIL_CHARS,
) -> tuple[str, int]:
    """Return ``(text_with_dense_lines_elided, lines_elided)``.

    Byte-identical to the input (and ``0``) when no line is dense or the
    rewrite does not save at least one character. When ``retrieval_hint`` is
    given, each elided line's marker gains `` (retrieve: <hint>)`` so the
    original remains reachable.
    """
    if not isinstance(text, str) or len(text) < min_line_chars:
        return text, 0
    lines = text.split("\n")
    dense = [
        is_dense_line(line, min_line_chars=min_line_chars, max_space_ratio=max_space_ratio)
        for line in lines
    ]
    if sum(len(line) for line, d in zip(lines, dense) if d) < min_dense_total_chars:
        return text, 0

    out: list[str] = []
    n_elided = 0
    for line, d in zip(lines, dense):
        if not d or len(line) <= head_chars + tail_chars:
            out.append(line)
            continue
        omitted = len(line) - head_chars - tail_chars
        replacement = (
            f"{line[:head_chars]} …[elided {omitted} chars]… {line[-tail_chars:]}"
        )
        if retrieval_hint:
            replacement += f" (retrieve: {retrieval_hint})"
        out.append(replacement)
        n_elided += 1

    if n_elided == 0:
        return text, 0
    candidate = "\n".join(out)
    if len(candidate) >= len(text):
        return text, 0
    return candidate, n_elided
