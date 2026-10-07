"""Line-level protection regexes for content-aware compression.

Pure, no I/O. Two protection surfaces:

* :data:`CRITICAL_LINE_RE` matches lines that carry error/diagnostic
  semantics (ERROR / FATAL / CRITICAL / Traceback / panic / exception /
  assertion). These lines must never be elided or dropped.
* :data:`MUST_KEEP_RE` is ported from Headroom's Kompress must-keep token
  regexes: hex ids, standalone numbers, version strings, ALLCAPS
  identifiers, dotted and unix paths, file extensions, CLI flags,
  CamelCase names, negations/modals, and boolean connectives. Losing any
  of these tokens changes what the surrounding text means (a dropped
  ``not`` inverts the sentence; a dropped path is unreconstructable).

:func:`is_error_line` is the narrow "do not elide this" guard used by
dense-line elision: a dense blob containing an ALLCAPS run (base64) or a
path is still safely elidable when the original is retrievable, but a line
carrying an error signal must never be. :func:`is_critical_line` additionally
consults :data:`MUST_KEEP_RE` and is the guard for lossy token-level
compression (Phase 2), where dropping a ``not`` or a path changes meaning.
"""

from __future__ import annotations

import re

__all__ = ["CRITICAL_LINE_RE", "MUST_KEEP_RE", "is_critical_line", "is_error_line"]

# Error/diagnostic semantics. Case-insensitive so "Error", "error" and
# "ERROR" all match; suffixed forms ("errors", "AssertionError") match too.
CRITICAL_LINE_RE = re.compile(
    r"\b(?:error|fatal|critical|panic|exception|assert|traceback)\w*",
    re.IGNORECASE,
)

# Tokens whose loss degrades or inverts meaning. Ported (with a version
# alternative added) from headroom/transforms/kompress_compressor.py.
MUST_KEEP_RE = re.compile(
    r"\b0x[0-9A-Fa-f]+\b"  # hex addresses/IDs: 0x7fff2038
    r"|(?<![\w.])\d+(?:\.\d+)?(?![\w.])"  # standalone numbers: 42, 3.14
    r"|\bv?\d+(?:\.\d+){1,}(?:[-+][0-9A-Za-z.]+)?\b"  # versions: 1.2.3, v2.0.1-rc1
    r"|[A-Z_]{2,}"  # ALLCAPS: SIGILL, HTTP, EOF, ERROR
    r"|[a-z_][a-z0-9_]*\.[a-z0-9_]+"  # dotted.paths: libsystem_kernel.dylib
    r"|/[a-z0-9/._-]{2,}"  # unix paths: /usr/lib/python3.so
    r"|\.[a-z]{2,4}\b"  # extensions: .py .so .json
    r"|--?[a-z][\w-]*"  # flags: --verbose, -n
    r"|\b[A-Z][a-z]+[A-Z]\w*"  # CamelCase: IndexError, EXC_BAD_INSTRUCTION
    # Directive words: losing one does not degrade the sentence, it can
    # invert it ("do not guess" -> "do guess").
    r"|(?i:\b(?:not|never|none|cannot|can't|don't|doesn't|didn't|won't|shouldn't"
    r"|mustn't|isn't|aren't|avoid|refuse|prohibited|forbidden|disallow|unless"
    r"|except|without|must|should|shall|required|always|only|mandatory)\b)"
    # Boolean connectives decide which predicates must hold.
    r"|(?i:\b(?:and|or|nor|xor)\b)"
)


def is_error_line(line: str) -> bool:
    """Return True when ``line`` carries error/diagnostic semantics.

    The dense-line elision guard: error lines are never elided even when the
    original is retrievable, because the diagnostic is the reason the agent
    read the output at all. Non-string input returns False.
    """
    if not isinstance(line, str) or not line:
        return False
    return bool(CRITICAL_LINE_RE.search(line))


def is_critical_line(line: str) -> bool:
    """Return True when ``line`` must never be elided.

    A line is critical when it carries error/diagnostic semantics
    (:data:`CRITICAL_LINE_RE`) or contains any must-keep token
    (:data:`MUST_KEEP_RE`). Non-string input returns False (fail-open for
    the caller's purposes: the caller treats non-critical as elidable, so
    this guard is intentionally conservative only for strings).
    """
    if not isinstance(line, str) or not line:
        return False
    return bool(CRITICAL_LINE_RE.search(line) or MUST_KEEP_RE.search(line))
