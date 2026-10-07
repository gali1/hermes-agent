"""JSON array crusher: lossless CSV-schema render first, gated lossy drop.

Ports the parts of Headroom's SmartCrusher that fit a stdlib-only, always
retrievable design:

* **Lossless render** — arrays of homogeneous objects (>= 5 items, >= 200
  chars) are rendered as a ``[N]{k1:type,k2:type?}`` schema header plus one
  CSV row per item, mirroring the wire format of Headroom's
  ``CsvSchemaFormatter`` (missing key -> empty cell, ``null`` -> bare ``null``,
  empty/literal-``null`` strings CSV-quoted, quotes doubled). The render is
  adopted only when it is byte-smaller than the input AND re-parsing it yields
  the exact same list of dicts.
* **Lossy drop** — only with ``allow_lossy`` AND a ``retrieval_hint``. First,
  last, error-keyword items, and numeric anomalies (> 2 sigma from a numeric
  field's mean) are always kept; the rest fill up to ``max_items`` in original
  order. Dropped items become a trailing
  ``... N items omitted (retrieve: <hint>)`` line and the count is returned.

Malformed JSON and non-array payloads fall back to compact separators when
that is strictly smaller, else the input is returned unchanged. Never raises.
"""

from __future__ import annotations

import json
import re

__all__ = [
    "compress_json",
    "render_csv_schema",
    "parse_csv_schema",
    "MIN_ITEMS",
    "MIN_CHARS",
    "VARIANCE_THRESHOLD",
    "ERROR_KEYWORD_RE",
]

MIN_ITEMS = 5
MIN_CHARS = 200
VARIANCE_THRESHOLD = 2.0

ERROR_KEYWORD_RE = re.compile(
    r"error|fail|fatal|exception|panic|denied|timeout|refused", re.IGNORECASE
)

_HEADER_RE = re.compile(r"^\[(\d+)\]\{(.*)\}$")
_QUOTE_TRIGGERS = (",", '"', "\n", "\r")


def compress_json(
    text,
    *,
    max_items: int = 15,
    retrieval_hint: str | None = None,
    allow_lossy: bool = False,
) -> tuple[str, int]:
    """Return ``(text, dropped_items)``; ``text`` is never larger than input.

    ``dropped_items`` is non-zero only when the lossy path ran and removed
    items. Lossy output embeds ``retrieval_hint`` in its omission marker; with
    no hint (or ``allow_lossy=False``) the function is strictly lossless.
    """
    if not isinstance(text, str) or not text:
        return text, 0
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return text, 0

    if (
        isinstance(value, list)
        and len(value) >= MIN_ITEMS
        and len(text) >= MIN_CHARS
        and all(isinstance(item, dict) for item in value)
    ):
        rendered = render_csv_schema(value)
        if rendered is not None and len(rendered) < len(text):
            try:
                if parse_csv_schema(rendered) == value:
                    return rendered, 0
            except Exception:
                pass

        if allow_lossy and retrieval_hint:
            kept = _select_lossy_indices(value, max_items)
            dropped = len(value) - len(kept)
            if dropped > 0:
                kept_items = [value[i] for i in sorted(kept)]
                body = render_csv_schema(kept_items)
                if body is None:
                    body = _compact(kept_items)
                try:
                    if parse_csv_schema(body) != kept_items:
                        body = _compact(kept_items)
                except Exception:
                    body = _compact(kept_items)
                marker = f"... {dropped} items omitted (retrieve: {retrieval_hint})"
                candidate = body.rstrip("\n") + "\n" + marker + "\n"
                if len(candidate) < len(text):
                    return candidate, dropped

    return _compact_or_unchanged(text, value)


def _compact(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        return ""


def _compact_or_unchanged(text: str, value) -> tuple[str, int]:
    compact = _compact(value)
    if compact and len(compact) < len(text):
        return compact, 0
    return text, 0


def _select_lossy_indices(items: list[dict], max_items: int) -> set[int]:
    n = len(items)
    kept: set[int] = {0, n - 1}
    kept |= _error_indices(items)
    kept |= _anomaly_indices(items)
    remaining = max_items - len(kept)
    if remaining > 0:
        for i in range(n):
            if i not in kept:
                kept.add(i)
                remaining -= 1
                if remaining <= 0:
                    break
    return kept


def _error_indices(items: list[dict]) -> set[int]:
    out: set[int] = set()
    for i, item in enumerate(items):
        try:
            blob = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError, RecursionError):
            blob = str(item)
        if ERROR_KEYWORD_RE.search(blob):
            out.add(i)
    return out


def _anomaly_indices(items: list[dict]) -> set[int]:
    numeric: dict[str, list[tuple[int, float]]] = {}
    for i, item in enumerate(items):
        for key, value in item.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric.setdefault(key, []).append((i, float(value)))
    anomalies: set[int] = set()
    for pairs in numeric.values():
        if len(pairs) < 2:
            continue
        values = [v for _, v in pairs]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        if variance <= 0.0:
            continue
        std = variance**0.5
        threshold = VARIANCE_THRESHOLD * std
        for i, value in pairs:
            if abs(value - mean) > threshold:
                anomalies.add(i)
    return anomalies


def render_csv_schema(items: list[dict]) -> str | None:
    """Render dicts as ``[N]{k:type,...}`` + CSV rows, or None if unencodable.

    Every row has exactly one cell per schema column; a missing key is an empty
    cell and ``None`` is bare ``null``. The output is round-trip verified by
    callers via :func:`parse_csv_schema`.
    """
    if not items or not all(isinstance(item, dict) for item in items):
        return None
    keys: list[str] = []
    for item in items:
        for key in item:
            if key not in keys:
                keys.append(key)
    if not keys:
        return None
    for key in keys:
        if not key or any(ch in key for ch in ',:{}"\n\r'):
            return None
    columns = [(key, *_classify(items, key)) for key in keys]
    decl = ",".join(
        f"{key}:{type_tag}{'?' if nullable else ''}" for key, type_tag, nullable in columns
    )
    lines = [f"[{len(items)}]{{{decl}}}"]
    for item in items:
        cells = []
        for key, type_tag, _nullable in columns:
            if key not in item:
                cells.append("")
            else:
                cells.append(_render_cell(item[key], type_tag))
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def _classify(items: list[dict], key: str) -> tuple[str, bool]:
    present = [item[key] for item in items if key in item]
    nullable = len(present) != len(items) or any(value is None for value in present)
    values = [value for value in present if value is not None]
    if not values:
        return "null", nullable
    if all(isinstance(value, bool) for value in values):
        return "bool", nullable
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "int", nullable
    if all(isinstance(value, float) for value in values):
        return "float", nullable
    if all(isinstance(value, str) for value in values):
        if any("\n" in value or "\r" in value for value in values):
            return "json", nullable
        return "string", nullable
    return "json", nullable


def _render_cell(value, type_tag: str) -> str:
    if value is None:
        return "null"
    if type_tag == "string":
        if value == "" or value == "null" or any(ch in value for ch in _QUOTE_TRIGGERS):
            return _csv_quote(value)
        return value
    if type_tag == "json":
        return _csv_quote(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _csv_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def parse_csv_schema(text: str) -> list[dict] | None:
    """Parse a :func:`render_csv_schema` block back to a list of dicts.

    Returns None when the text is not a schema block or does not decode
    exactly; callers treat None as "not losslessly reversible".
    """
    if not isinstance(text, str) or not text:
        return None
    lines = text.split("\n")
    match = _HEADER_RE.match(lines[0]) if lines else None
    if not match:
        return None
    declared = int(match.group(1))
    columns: list[tuple[str, str, bool]] = []
    if match.group(2):
        for decl in match.group(2).split(","):
            name, sep, type_tag = decl.rpartition(":")
            if not sep or not name:
                return None
            nullable = type_tag.endswith("?")
            if nullable:
                type_tag = type_tag[:-1]
            if type_tag not in ("int", "float", "bool", "string", "json", "null"):
                return None
            columns.append((name, type_tag, nullable))
    body = lines[1:]
    if body and body[-1] == "":
        body = body[:-1]
    if len(body) != declared:
        return None
    records: list[dict] = []
    for line in body:
        cells = _split_csv_line(line)
        if cells is None or len(cells) != len(columns):
            return None
        record: dict = {}
        for (name, type_tag, _nullable), (quoted, value) in zip(columns, cells):
            if not quoted and value == "":
                continue
            if not quoted and value == "null":
                record[name] = None
                continue
            try:
                record[name] = _decode_cell(quoted, value, type_tag)
            except (ValueError, TypeError):
                return None
        records.append(record)
    return records


def _decode_cell(quoted: bool, value: str, type_tag: str):
    if type_tag == "json":
        return json.loads(value)
    if type_tag == "string":
        return value
    if type_tag == "null":
        if value == "null":
            return None
        raise ValueError("non-null cell in null column")
    if type_tag == "int":
        return int(value)
    if type_tag == "float":
        return float(value)
    if type_tag == "bool":
        if value == "true":
            return True
        if value == "false":
            return False
        raise ValueError("invalid bool cell")
    raise ValueError("unknown type tag")


def _split_csv_line(line: str) -> list[tuple[bool, str]] | None:
    """Split one CSV line into ``(was_quoted, value)`` cells.

    Mirrors the wire format emitted by :func:`_render_cell`: quotes escape by
    doubling and cells never contain raw newlines (multi-line strings are
    forced into ``json`` columns and JSON-escaped).
    """
    cells: list[tuple[bool, str]] = []
    i, n = 0, len(line)
    while i <= n:
        if i < n and line[i] == '"':
            i += 1
            buf: list[str] = []
            closed = False
            while i < n:
                ch = line[i]
                if ch == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        buf.append('"')
                        i += 2
                        continue
                    i += 1
                    closed = True
                    break
                buf.append(ch)
                i += 1
            if not closed:
                return None
            cells.append((True, "".join(buf)))
        else:
            j = i
            while j < n and line[j] != ",":
                j += 1
            cells.append((False, line[i:j]))
            i = j
        if i >= n:
            break
        if line[i] != ",":
            return None
        i += 1
        if i > n:
            break
    return cells
