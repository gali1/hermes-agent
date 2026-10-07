"""Tabular-text compressor: CSV/TSV/markdown tables to the CSV-schema render.

Parses delimited text (comma/tab/semicolon) or a markdown table into records
and re-renders them through :func:`agent.content_compression.json_crusher.
render_csv_schema`, which folds the repeated column names into one
``[N]{k:type,...}`` schema header. The result is adopted only when it is
strictly smaller AND re-parses back to the exact same records (all cells are
strings, so the round-trip is exact).

Ragged rows, duplicate headers, and unparseable tables pass through
byte-identical. There is no lossy tabular tier, so ``allow_lossy`` and
``retrieval_hint`` are accepted for call-site symmetry and ignored. Never
raises.
"""

from __future__ import annotations

import csv
import io
import re

from .json_crusher import parse_csv_schema, render_csv_schema

__all__ = ["compress_tabular", "parse_tabular"]

_MD_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")


def compress_tabular(
    text,
    *,
    retrieval_hint: str | None = None,
    allow_lossy: bool = False,
) -> tuple[str, int]:
    """Return ``(text, 0)``; never larger than the input."""
    if not isinstance(text, str) or not text:
        return text, 0
    try:
        parsed = parse_tabular(text)
    except Exception:
        return text, 0
    if parsed is None:
        return text, 0
    headers, rows = parsed
    if len(set(headers)) != len(headers) or not headers:
        return text, 0
    records = [dict(zip(headers, row)) for row in rows]
    rendered = render_csv_schema(records)
    if rendered is None or len(rendered) >= len(text):
        return text, 0
    try:
        if parse_csv_schema(rendered) != records:
            return text, 0
    except Exception:
        return text, 0
    return rendered, 0


def parse_tabular(text: str) -> tuple[list[str], list[list[str]]] | None:
    """Detect markdown/CSV/TSV shape and return ``(headers, rows)``.

    Returns None for ragged tables, tables with fewer than two columns or two
    rows, and anything not confidently tabular.
    """
    markdown = _parse_markdown(text)
    if markdown is not None:
        return markdown
    return _parse_delimited(text)


def _parse_markdown(text: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [ln for ln in text.split("\n") if ln.strip() and "|" in ln]
    if len(lines) < 3:
        return None
    headers = _split_md_row(lines[0])
    if len(headers) < 2:
        return None
    if not _is_md_separator(lines[1]):
        return None
    rows = [_split_md_row(ln) for ln in lines[2:]]
    if not rows:
        return None
    width = len(headers)
    if any(len(row) != width for row in rows):
        return None
    return headers, rows


def _split_md_row(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _is_md_separator(row: str) -> bool:
    cells = [cell for cell in _split_md_row(row) if cell]
    return len(cells) >= 2 and all(_MD_SEP_CELL_RE.match(cell) for cell in cells)


def _parse_delimited(text: str) -> tuple[list[str], list[list[str]]] | None:
    sample = [ln for ln in text.split("\n") if ln.strip()]
    if len(sample) < 3:
        return None
    first = sample[0]
    delimiter = "\t" if "\t" in first else (";" if first.count(";") >= first.count(",") and ";" in first else ",")
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        parsed = [row for row in reader if any(cell.strip() for cell in row)]
    except csv.Error:
        return None
    if len(parsed) < 3:
        return None
    headers = [cell.strip() for cell in parsed[0]]
    rows = [[cell.strip() for cell in row] for row in parsed[1:]]
    if len(headers) < 2 or not rows:
        return None
    width = len(headers)
    if any(len(row) != width for row in rows):
        return None
    return headers, rows
