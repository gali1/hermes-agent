"""Reliability-aware re-ranking for tool search (TASK-2 / AnyTool port).

A tool that keeps failing should not outrank an equally relevant tool that
works — but relevance stays dominant, and a bad record must never *hide* a
tool. So this module is deliberately narrow:

* :func:`reliability_multiplier` maps one tool's call outcomes to a multiplier
  in ``[0.2, 1.0]``. It is neutral (``1.0``) with fewer than
  :data:`MIN_CALLS_FOR_SIGNAL` calls and neutral at or above a
  :data:`NEUTRAL_SUCCESS_RATE` success rate — only a *demonstrably* unreliable
  tool is discounted at all.
* :func:`rerank_scored` re-orders the already-admitted, already-scored
  candidates. It is applied AFTER the admission gate, so the candidate set is
  identical with and without it: nothing is filtered, only reordered, and the
  caller still truncates to the same result count.
* :func:`load_outcome_counts` reads the newest :data:`WINDOW_ROWS` persisted
  message rows inside a :data:`WINDOW_DAYS`-day window. The window's "now" is a
  PARAMETER — no clock read happens inside scoring, so the same history always
  produces the same multipliers.

Fail-open everywhere: a missing/unreadable state DB, a schema older than the
``messages`` columns, a malformed row — all degrade to "no signal", which is
exactly the pre-TASK-2 behaviour.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

logger = logging.getLogger("tools.tool_search")

# Rolling window: the newest N message rows, and nothing older than WINDOW_DAYS.
WINDOW_ROWS = 8000
WINDOW_DAYS = 30.0
WINDOW_SECONDS = WINDOW_DAYS * 86400.0
# A rate computed from one or two calls is noise.
MIN_CALLS_FOR_SIGNAL = 3
# At/above this success rate a tool is simply not suspect.
NEUTRAL_SUCCESS_RATE = 0.40
# The band. 0.2 is a meaningful demotion that still cannot erase a hit: the
# multiplier scales the relevance score, it never zeroes or removes the entry.
MIN_MULTIPLIER = 0.2
MAX_MULTIPLIER = 1.0
# Read-only open budget for a state.db that another process is writing.
READ_TIMEOUT_S = 1.0
# Bytes of each tool result body inspected for the success/failure verdict. Tool
# outputs can be megabytes each, so the window reads a bounded prefix; a payload
# cut mid-envelope is treated as "no evidence" rather than guessed at.
CONTENT_PROBE_CHARS = 1500

# Keys a tool result payload uses to signal a refused (not failed) call.
_GUARDRAIL_REFUSAL_KEY = "guardrail_refusal"
# Leading text of a plain-string tool error (``tool_error`` emits "Error: ...").
_ERROR_PREFIXES = ("error:", "error ", "tool error", "failed:", "exception:")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _count(value: Any) -> int:
    """Non-negative int from anything (``0`` when unparseable)."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def reliability_multiplier(calls: Any, ok: Any) -> float:
    """Ranking multiplier for one tool from its windowed call counts.

    Neutral (``1.0``) under :data:`MIN_CALLS_FOR_SIGNAL` calls and at or above
    :data:`NEUTRAL_SUCCESS_RATE` success. Below that the multiplier falls
    linearly from ``1.0`` at the neutral boundary to ``MIN_MULTIPLIER`` at a 0%
    success rate, so the function is continuous and monotonic in the rate, and
    always inside ``[MIN_MULTIPLIER, MAX_MULTIPLIER]``.
    """
    calls = _count(calls)
    ok = min(_count(ok), calls)
    if calls < MIN_CALLS_FOR_SIGNAL:
        return MAX_MULTIPLIER
    rate = ok / calls
    if rate >= NEUTRAL_SUCCESS_RATE:
        return MAX_MULTIPLIER
    span = MAX_MULTIPLIER - MIN_MULTIPLIER
    return _clamp(MIN_MULTIPLIER + span * (rate / NEUTRAL_SUCCESS_RATE),
                  MIN_MULTIPLIER, MAX_MULTIPLIER)


def multiplier_from_record(record: Optional[Mapping[str, Any]]) -> float:
    """Convenience wrapper for a ``{"calls": n, "ok": k}`` mapping."""
    if not isinstance(record, Mapping):
        return MAX_MULTIPLIER
    return reliability_multiplier(record.get("calls", 0), record.get("ok", 0))


def _result_ok(content: Any) -> Optional[bool]:
    """Whether one persisted tool result counts as success (``None`` = no evidence).

    Mirrors the shapes the runtime emits: JSON objects carrying ``error``/``ok``/
    ``success`` (``tool_error`` writes ``{"error": ...}``), and plain-string
    errors that start with ``Error:``. Harness guardrail refusals are neutral —
    they refused a redundant call, nothing failed.
    """
    if content is None:
        return None
    if isinstance(content, (bytes, bytearray)):
        try:
            content = content.decode("utf-8", "replace")
        except Exception:
            return None
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    if text[0] in "{[":
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            # A truncated envelope (the probe prefix cut it) or malformed body:
            # no verdict rather than a guess.
            return None
        if isinstance(data, dict):
            if data.get(_GUARDRAIL_REFUSAL_KEY) is True:
                return None
            if data.get("error"):
                return False
            if data.get("ok") is False or data.get("success") is False:
                return False
            return True
        if isinstance(data, list):
            return True
    lowered = text[:64].lower()
    if lowered.startswith(_ERROR_PREFIXES):
        return False
    return True


def counts_from_rows(rows: Iterable[Sequence[Any]]) -> Dict[str, Dict[str, int]]:
    """Aggregate ``(tool_name, content)`` rows into ``{name: {calls, ok, fail}}``.

    Pure and order-independent (counts, not sequences), so a differently ordered
    read of the same window yields identical multipliers.
    """
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        try:
            name, content = row[0], row[1]
        except (TypeError, IndexError):
            continue
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name:
            continue
        verdict = _result_ok(content)
        if verdict is None:
            continue
        record = counts.setdefault(name, {"calls": 0, "ok": 0, "fail": 0})
        record["calls"] += 1
        if verdict:
            record["ok"] += 1
        else:
            record["fail"] += 1
    return counts


def multipliers_from_counts(counts: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """``{name: multiplier}`` for the tools that are NOT neutral.

    Tools absent from the mapping are neutral by definition, so the caller can
    treat a missing key as ``1.0`` and this mapping stays small.
    """
    out: Dict[str, float] = {}
    for name, record in counts.items():
        multiplier = multiplier_from_record(record)
        if multiplier != MAX_MULTIPLIER:
            out[str(name)] = multiplier
    return out


def default_state_db_path() -> Path:
    """``<hermes home>/state.db`` resolved at CALL time (HERMES_HOME redirects work)."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "state.db"
    except Exception:  # pragma: no cover - constants module is always present
        return Path.home() / ".hermes" / "state.db"


def load_outcome_counts(*, db_path: Optional[Any] = None, now: Optional[float] = None,
                        max_rows: int = WINDOW_ROWS,
                        window_seconds: float = WINDOW_SECONDS) -> Dict[str, Dict[str, int]]:
    """Windowed per-tool outcome counts from the session message store.

    Reads the newest ``max_rows`` ``role='tool'`` rows with a non-empty
    ``tool_name`` and ``timestamp >= now - window_seconds``. Strictly read-only
    (``mode=ro``) so it can never contend with the live gateway's writer.
    Returns ``{}`` on any failure — callers then behave exactly as before.
    """
    rows = load_outcome_rows(db_path=db_path, now=now, max_rows=max_rows,
                             window_seconds=window_seconds)
    return counts_from_rows(rows)


def load_outcome_rows(*, db_path: Optional[Any] = None, now: Optional[float] = None,
                      max_rows: int = WINDOW_ROWS,
                      window_seconds: float = WINDOW_SECONDS) -> List[Tuple[str, Any]]:
    """The raw ``(tool_name, content_head)`` rows of the window (empty on any failure).

    Only the first :data:`CONTENT_PROBE_CHARS` of each result body are read: the
    success/failure verdict lives in the envelope, and full tool outputs can be
    megabytes each — 8000 of them would make every search expensive.
    """
    path = Path(db_path) if db_path is not None else default_state_db_path()
    if not path.is_file():
        return []
    since = (time.time() if now is None else float(now)) - float(window_seconds)
    try:
        conn = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro", uri=True,
                               timeout=READ_TIMEOUT_S)
    except sqlite3.Error as exc:
        logger.debug("tool reliability: cannot open %s read-only: %s", path, exc)
        return []
    try:
        cursor = conn.execute(
            "SELECT tool_name, substr(content, 1, ?) FROM messages "
            "WHERE role = 'tool' AND tool_name IS NOT NULL AND tool_name != '' "
            "AND timestamp >= ? ORDER BY id DESC LIMIT ?",
            (CONTENT_PROBE_CHARS, since, max(1, int(max_rows))),
        )
        return [(row[0], row[1]) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        logger.debug("tool reliability: message window unreadable (%s): %s", path, exc)
        return []
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def load_reliability_multipliers(*, db_path: Optional[Any] = None,
                                 now: Optional[float] = None) -> Dict[str, float]:
    """``{tool_name: multiplier}`` for the current window (``{}`` when no signal).

    The clock is read exactly once, here, at the window boundary — never inside
    :func:`reliability_multiplier` or :func:`rerank_scored`.
    """
    return multipliers_from_counts(load_outcome_counts(db_path=db_path, now=now))


def rerank_scored(scored: Sequence[Tuple[float, Any]],
                  multipliers: Optional[Mapping[str, float]]) -> List[Tuple[float, Any]]:
    """Reorder ``(score, entry)`` candidates by ``score * multiplier``.

    Post-admission only: every input pair appears in the output exactly once
    (a pure permutation, no filtering), and ties keep their incoming order so
    the result is deterministic for a given candidate list and mapping. With no
    multipliers this returns the input order untouched.
    """
    items = list(scored)
    if not multipliers:
        return items

    def _adjusted(pair: Tuple[float, Any]) -> float:
        score, entry = pair
        name = getattr(entry, "name", "")
        multiplier = multipliers.get(name, MAX_MULTIPLIER)
        try:
            multiplier = _clamp(float(multiplier), MIN_MULTIPLIER, MAX_MULTIPLIER)
        except (TypeError, ValueError):
            multiplier = MAX_MULTIPLIER
        return score * multiplier

    return sorted(items, key=_adjusted, reverse=True)


__all__ = [
    "WINDOW_ROWS", "WINDOW_DAYS", "WINDOW_SECONDS", "MIN_CALLS_FOR_SIGNAL",
    "NEUTRAL_SUCCESS_RATE", "MIN_MULTIPLIER", "MAX_MULTIPLIER", "CONTENT_PROBE_CHARS",
    "reliability_multiplier", "multiplier_from_record", "counts_from_rows",
    "multipliers_from_counts", "load_outcome_rows", "load_outcome_counts",
    "load_reliability_multipliers", "rerank_scored", "default_state_db_path",
]
