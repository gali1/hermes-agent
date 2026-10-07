"""Structured token estimation for content-aware compression (Phase 3b).

A pure, deterministic improvement over the ubiquitous ``len(text) // 4``
heuristic. JSON payloads are priced from their structure (object keys, string
words, numbers, literals, and punctuation); code-shaped text from its
identifiers and operators; everything else from words plus punctuation. Each
branch is clamped so the estimate stays in a sane band for its content type.

stdlib only (``json`` + ``re``), no tokenizer dependency. Empty input returns
0; any non-empty input returns at least 1.
"""

from __future__ import annotations

import json
import re

__all__ = ["estimate_tokens_structured", "estimate_messages_tokens_structured"]

_CODE_CHARS = "(){};=<>"
_CODE_CHAR_RATIO = 0.03
_CODE_MIN_CHARS = 4

_CODE_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"  # identifiers
    r"|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"  # numbers
    r"|[^\sA-Za-z0-9_]+"  # operator / punctuation runs
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

_NOT_JSON = object()


def _clamp(value: int, low: int, high: int) -> int:
    low = max(1, low)
    high = max(low, high)
    return max(low, min(value, high))


def _string_tokens(value: str) -> int:
    words = value.split()
    return max(1, len(words)) if value else 0


def _json_tokens(value) -> int:
    """Token estimate for a parsed JSON value from its structure."""
    if isinstance(value, dict):
        total = 2  # opening and closing brace
        for key, item in value.items():
            total += 1 + _string_tokens(str(key)) + 1  # key, colon
            total += _json_tokens(item) + 1  # value, comma
        return total
    if isinstance(value, list):
        total = 2  # opening and closing bracket
        for item in value:
            total += _json_tokens(item) + 1  # value, comma
        return total
    if isinstance(value, str):
        return _string_tokens(value)
    if value is True or value is False or value is None:
        return 1
    if isinstance(value, (int, float)):
        return 1
    return 1


def _looks_like_code(text: str) -> bool:
    code_chars = sum(text.count(ch) for ch in _CODE_CHARS)
    return code_chars >= _CODE_MIN_CHARS and (code_chars / len(text)) >= _CODE_CHAR_RATIO


def _code_tokens(text: str) -> int:
    total = 0
    for token in _CODE_TOKEN_RE.findall(text):
        first = token[0]
        if first.isalpha() or first == "_" or first.isdigit():
            total += 1
        else:
            total += (len(token) + 1) // 2
    return total


def _prose_tokens(text: str) -> int:
    return len(_WORD_RE.findall(text)) + len(_PUNCT_RE.findall(text))


def estimate_tokens_structured(text) -> int:
    """Estimate tokens for ``text`` from its shape rather than ``chars / 4``.

    JSON gets a structural count clamped to ``[len // 6, len // 3]``; code
    (enough ``(){};=<>`` characters) is split into identifiers/operators and
    capped at ``len``; anything else uses words plus punctuation clamped to
    ``[len // 5, len // 3]``. Deterministic, stdlib only, never raises for a
    string. Empty/non-string input returns 0.
    """
    if not isinstance(text, str) or not text:
        return 0

    length = len(text)
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        value = _NOT_JSON

    if value is not _NOT_JSON:
        return _clamp(_json_tokens(value), length // 6, length // 3)
    if _looks_like_code(text):
        return _clamp(_code_tokens(text), 1, length)
    return _clamp(_prose_tokens(text), length // 5, length // 3)


def estimate_messages_tokens_structured(messages) -> int:
    """Sum :func:`estimate_tokens_structured` over string message contents.

    List/multimodal payloads and non-dict messages are skipped, mirroring the
    existing rough estimator. Never raises for a list/None input.
    """
    if not messages:
        return 0
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            total += estimate_tokens_structured(content)
    return total
