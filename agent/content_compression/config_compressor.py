"""Structured-config compressor (Headroom-derived, stdlib-only).

Two tiers, mirroring Headroom's ``ConfigCompressor``:

* **Lossless** — :func:`agent.content_compression.lossless.fold_lossless`
  (run/stanza folding, path listing) with per-step round-trip verification.
  Always on.
* **Lossy, gated** — whole-line comments (``#`` / ``;``) and blank lines are
  elided only with ``allow_lossy`` AND a ``retrieval_hint``; the original must
  be retrievable through the hint embedded in the marker. The elision is
  skipped entirely when the text may contain data that looks like a comment:
  YAML block scalars (``|`` / ``>``) or TOML multi-line strings (``\"\"\"`` /
  ``'''``).

Adopted only when strictly smaller; never raises.
"""

from __future__ import annotations

import re

from .lossless import fold_lossless

__all__ = ["compress_config"]

_YAML_BLOCK_SCALAR_RE = re.compile(r":\s*[|>][+-]?\d*\s*$", re.MULTILINE)
_TOML_MULTILINE_RE = re.compile(r'"""|\'\'\'')
_COMMENT_RE = re.compile(r"^\s*[#;]")


def compress_config(
    text,
    *,
    retrieval_hint: str | None = None,
    allow_lossy: bool = False,
) -> tuple[str, int]:
    """Return ``(text, dropped_lines)``; never larger than the input."""
    if not isinstance(text, str) or not text:
        return text, 0
    try:
        base = fold_lossless(text)
    except Exception:
        base = text
    if not base or len(base) >= len(text):
        base = text

    result = base
    dropped = 0
    if allow_lossy and retrieval_hint and not _has_block_scalar(base):
        stripped, elided = _strip_comment_blank_lines(base)
        if elided > 0:
            trailing = base.endswith("\n")
            marker = f"... {elided} comment/blank lines elided (retrieve: {retrieval_hint})"
            candidate = stripped.rstrip("\n") + "\n" + marker + ("\n" if trailing else "")
            if len(candidate) < len(base):
                result, dropped = candidate, elided

    if len(result) >= len(text):
        return text, 0
    return result, dropped


def _has_block_scalar(text: str) -> bool:
    return bool(_YAML_BLOCK_SCALAR_RE.search(text) or _TOML_MULTILINE_RE.search(text))


def _strip_comment_blank_lines(text: str) -> tuple[str, int]:
    trailing = text.endswith("\n")
    lines = (text[:-1] if trailing else text).split("\n")
    kept: list[str] = []
    elided = 0
    for line in lines:
        if _COMMENT_RE.match(line) or not line.strip():
            elided += 1
        else:
            kept.append(line)
    return "\n".join(kept) + ("\n" if trailing else ""), elided
