"""Rule-based entity extraction over free text.

Ported from the OpenCode/MemPalace Hindsight primitives.  Stdlib only,
bounded by `limit`, and safe on None/empty input.
"""

from __future__ import annotations

import re

ENTITY_PATTERNS = [
    ("file_path", re.compile(r"(?:/[\w.\-]+)+\.\w+")),
    ("url", re.compile(r"https?://[^\s<>\"')]+")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("version", re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")),
    ("constant", re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")),
    ("camel_case", re.compile(r"\b(?=[A-Za-z0-9]{3,}\b)[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b")),
    ("identifier", re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:[._][A-Za-z0-9_]+)+\b")),
    ("backtick", re.compile(r"`([^`\n]+)`")),
]


def extract_entities(text, limit=32) -> list[dict]:
    """Extract deduplicated entities in order of first appearance, capped at `limit`."""
    if not text or limit <= 0:
        return []
    if not isinstance(text, str):
        text = str(text)

    seen = set()
    found = []
    for kind, pattern in ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1) if match.groups() else match.group(0)
            value = value.strip()
            if not value:
                continue
            key = (kind, value.lower())
            if key in seen:
                continue
            seen.add(key)
            found.append({"text": value, "kind": kind})
            if len(found) >= limit:
                return found
    return found
