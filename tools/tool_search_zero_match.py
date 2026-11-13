"""Bounded zero-match LLM fallback for tool search (TASK-2).

Lexical search (BM25 + rarest-token admission) is precise and cheap, and that is
its failure mode too: a query whose intent word no tool carries returns nothing
even when a fitting tool exists under a different vocabulary ("ping the team"
against ``slack_post_message``). This module adds ONE bounded recovery attempt:

* it runs only when the lexical pass returned no hits at all,
* it sends a hard-capped catalog listing (names + one-line descriptions only —
  never schemas),
* it makes ONE model call under a wall-clock cap,
* and it returns nothing at all on any failure (no answer, unparsable answer,
  provider error, timeout): the caller keeps its old empty result.

Proposals are candidates, not results: :func:`select_proposals` drops every name
that is not present in the caller's real corpus, so a hallucinated tool can
never be returned to the model.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("tools.tool_search")

# Hard caps on everything this pass can cost.
MAX_CATALOG_ENTRIES = 150
MAX_CATALOG_CHARS = 6000
MAX_PROPOSALS = 5
DEFAULT_TIMEOUT_S = 4.0
DESCRIPTION_CHARS = 90
MAX_TOKENS = 160

_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:\-]*")
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")

_SYSTEM_PROMPT = (
    "You map a capability request to tool names from a catalog. Reply with a "
    "JSON array of at most {max_proposals} tool names copied EXACTLY from the "
    "catalog, best first. Reply [] when no catalog tool plausibly fits. No "
    "prose, no explanation, no invented names."
)


def _one_line(text: str, cap: int = DESCRIPTION_CHARS) -> str:
    """Whitespace-collapsed, clipped description (no newlines in the prompt)."""
    line = " ".join(str(text or "").split())
    if len(line) <= cap:
        return line
    return line[:cap].rsplit(" ", 1)[0] + "…"


def catalog_listing(entries: Sequence[Any], *, max_entries: int = MAX_CATALOG_ENTRIES,
                    max_chars: int = MAX_CATALOG_CHARS) -> str:
    """``- name: one-line description`` lines, capped by entry count AND characters.

    Deterministic (input order preserved, so the caller's catalog order decides
    which entries fit) and cheap: the total payload can never exceed
    ``max_chars``, whatever the model later does with it.
    """
    lines: List[str] = []
    used = 0
    for entry in entries[:max(0, max_entries)]:
        name = str(getattr(entry, "name", "") or "").strip()
        if not name:
            continue
        description = _one_line(getattr(entry, "description", ""))
        line = f"- {name}: {description}" if description else f"- {name}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def build_messages(query: str, listing: str, *, max_proposals: int = MAX_PROPOSALS) -> List[Dict[str, str]]:
    """The two-message request body for the proposal pass."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT.format(max_proposals=max_proposals)},
        {"role": "user",
         "content": f"Catalog:\n{listing}\n\nRequest: {query}\n\nJSON array of tool names:"},
    ]


def parse_proposals(text: Any, *, max_proposals: int = MAX_PROPOSALS) -> List[str]:
    """Tolerant parse of a model answer into candidate names (never raises).

    Accepts a JSON array (of strings or of ``{"name": ...}`` objects), a bare
    JSON string, or a plain list of names one per line / comma separated.
    """
    if not isinstance(text, str):
        return []
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    if not cleaned:
        return []
    start, end = cleaned.find("["), cleaned.rfind("]")
    candidates: List[str] = []
    if 0 <= start < end:
        try:
            parsed = json.loads(cleaned[start:end + 1])
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    candidates.append(item)
                elif isinstance(item, dict):
                    value = item.get("name") or item.get("tool") or item.get("tool_name")
                    if isinstance(value, str):
                        candidates.append(value)
    if not candidates:
        for token in _TOKEN_RE.findall(cleaned):
            if token.lower() in {"json", "array", "tool", "names", "none", "null"}:
                continue
            candidates.append(token)
    out: List[str] = []
    for candidate in candidates:
        name = candidate.strip().strip("\"'`,;")
        if name and name not in out:
            out.append(name)
        if len(out) >= max_proposals:
            break
    return out


def select_proposals(names: Iterable[str], corpus: Sequence[Any], *,
                     limit: Optional[int] = None) -> List[Any]:
    """Map proposed names onto real catalog entries, dropping everything else.

    A name that is not in the corpus is a hallucination and is dropped, not
    returned. Order follows the model's ranking; duplicates collapse. The
    ordering is intentionally NOT relevance-sorted: the caller already knows the
    lexical arm found nothing, so the model's ranking is all the signal there is.
    """
    by_name: Dict[str, Any] = {}
    for entry in corpus:
        name = getattr(entry, "name", None)
        if isinstance(name, str) and name and name not in by_name:
            by_name[name] = entry
    selected: List[Any] = []
    seen: set = set()
    for raw in names:
        name = str(raw or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        entry = by_name.get(name)
        if entry is not None:
            selected.append(entry)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _default_call(messages: List[Dict[str, str]], timeout: float) -> Any:
    """One auxiliary-model request (imported lazily: the bridge path stays light)."""
    from agent.auxiliary_client import call_llm

    return call_llm(task="tool_search", temperature=0, max_tokens=MAX_TOKENS,
                    timeout=timeout, messages=messages)


def _response_text(response: Any) -> str:
    """Best-effort text of an auxiliary response (``""`` when it has none)."""
    try:
        return str(response.choices[0].message.content or "")
    except (AttributeError, IndexError, TypeError):
        pass
    content = getattr(response, "content", None)
    return content if isinstance(content, str) else ""


def propose_names(query: str, entries: Sequence[Any], *,
                  call: Optional[Callable[[List[Dict[str, str]], float], Any]] = None,
                  timeout: float = DEFAULT_TIMEOUT_S,
                  max_proposals: int = MAX_PROPOSALS) -> List[str]:
    """One bounded proposal pass; ``[]`` on any failure, silence included.

    The call runs on a daemon thread joined for at most ``timeout`` seconds, so
    a provider that ignores the request timeout cannot stall the tool-search
    call that triggered it. Nothing is retried: one attempt, then the old
    empty result.
    """
    listing = catalog_listing(entries)
    if not listing or not str(query or "").strip():
        return []
    messages = build_messages(query, listing, max_proposals=max_proposals)
    caller = call or _default_call
    box: Dict[str, Any] = {}

    def _run() -> None:
        try:
            box["response"] = caller(messages, timeout)
        except BaseException as exc:  # noqa: BLE001 - never propagate into the turn
            box["error"] = exc

    worker = threading.Thread(target=_run, name="tool-search-zero-match", daemon=True)
    try:
        worker.start()
        worker.join(max(0.0, float(timeout)))
    except Exception as exc:  # pragma: no cover - thread start can only fail hard
        logger.debug("tool search zero-match: cannot start proposal thread: %s", exc)
        return []
    if worker.is_alive():
        logger.debug("tool search zero-match: model pass exceeded %.2fs; using the "
                     "empty lexical result", timeout)
        return []
    if "error" in box:
        logger.debug("tool search zero-match: model pass failed (%s: %s); using the "
                     "empty lexical result", type(box["error"]).__name__, box["error"])
        return []
    return parse_proposals(_response_text(box.get("response")), max_proposals=max_proposals)


def propose_entries(query: str, entries: Sequence[Any], corpus: Sequence[Any], *,
                    call: Optional[Callable[[List[Dict[str, str]], float], Any]] = None,
                    timeout: float = DEFAULT_TIMEOUT_S,
                    max_proposals: int = MAX_PROPOSALS,
                    limit: Optional[int] = None) -> List[Any]:
    """Propose, then filter against the real corpus — the whole fallback in one call."""
    names = propose_names(query, entries, call=call, timeout=timeout,
                          max_proposals=max_proposals)
    if not names:
        return []
    return select_proposals(names, corpus, limit=limit)


__all__ = [
    "MAX_CATALOG_ENTRIES", "MAX_CATALOG_CHARS", "MAX_PROPOSALS", "DEFAULT_TIMEOUT_S",
    "catalog_listing", "build_messages", "parse_proposals", "select_proposals",
    "propose_names", "propose_entries",
]
