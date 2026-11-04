"""Deferred-tool catalog for tool search: BM25 retrieval over deferrable tool
defs plus the budgeted, byte-stable catalog listing embedded in the bridge.

Ranking is lexical only. Recorded call outcomes can re-order the hits a query already
admitted (:func:`load_tool_reliability` + ``search_catalog(reliability=...)``) but never
decide which tools are offered."""

from __future__ import annotations

import functools
import json
import logging
import math
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import snowballstemmer

from tools.delegate_tool_results import _looks_like_error_output

logger = logging.getLogger(__name__)

# Reserved bridge names: a user/plugin/MCP tool may not take them (registry override
# protection rejects such registrations).
TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_CALL_NAME = "tool_call"
BRIDGE_TOOL_NAMES = frozenset({TOOL_SEARCH_NAME, TOOL_DESCRIBE_NAME, TOOL_CALL_NAME})
# Chars-per-token rule of thumb; 4.0 slightly underestimates (fewer false activations).
CHARS_PER_TOKEN = 4.0


@dataclass
class CatalogEntry:
    """One deferrable tool, in a form the bridge tools can search and serve."""

    name: str
    description: str
    schema: Dict[str, Any]  # the full {"type":"function", "function": {...}} entry
    source: str  # "mcp" | "plugin" | "other"
    source_name: str  # toolset name, e.g. "mcp-github" or "kanban"
    _tokens: List[str] = field(default_factory=list)  # pre-tokenized for BM25


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_thread_local = threading.local()


@functools.lru_cache(maxsize=16384)
def _stem(token: str) -> str:
    """Stem one token, memoized across stateless catalog rebuilds. Snowball stemmers carry
    mutable parsing state and bridge dispatch runs on parallel tool-call threads, so the
    stemmer is one-per-thread, created lazily."""
    if getattr(_thread_local, "stemmer", None) is None:
        _thread_local.stemmer = snowballstemmer.stemmer("english")
    return _thread_local.stemmer.stemWord(token)


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, Snowball-stemmed (English); shared by the index and
    query paths so "issues" matches ``create_issue``."""
    return [_stem(token.lower()) for token in _TOKEN_RE.findall(text)] if text else []


def _fn(td: Dict[str, Any]) -> Dict[str, Any]:
    """The ``function`` block of a tool-def (``{}`` when absent/None)."""
    return td.get("function") or {}


def _registry_entry(name: str) -> Any:
    """Registry entry for ``name``; None when unregistered OR the registry raises (lookup
    failures must never fail a bridge call). Lazy import: tests patch the registry."""
    try:
        from tools.registry import registry
        return registry.get_entry(name)
    except Exception:
        return None


def _registry_toolset(name: str) -> Optional[str]:
    """Toolset of a registered tool; None when unregistered or malformed (no str toolset)."""
    toolset = getattr(_registry_entry(name), "toolset", None)
    return toolset if isinstance(toolset, str) else None


def _entry_search_text(td: Dict[str, Any], source_label: str = "") -> str:
    """Search-text blob: split name words + source label + description + top-level parameter
    names (schema bodies are noise with no recall gain). The ``mcp__`` prefix is dropped — it
    is in every MCP document, so its IDF is ~0. The source label lets a service-name query
    ("linear") reach a tool whose own name omits the vendor."""
    fn = _fn(td)
    name = fn.get("name", "")
    if name.startswith("mcp__"):
        name = name[len("mcp__"):]
    name_words = re.sub(r"[_.:-]", " ", name)
    extra = source_label if source_label and source_label not in name_words.split() else ""
    param_names = " ".join(((fn.get("parameters") or {}).get("properties") or {}).keys())
    return f"{name_words} {extra} {fn.get('description', '') or ''} {param_names}"


def _classify_source(name: str) -> Tuple[str, str]:
    """Return (source_kind, source_name) for a registered tool name."""
    toolset = _registry_toolset(name)
    if toolset is None:
        return ("other", "")
    return ("mcp" if toolset.startswith("mcp-") else "plugin", toolset)


def build_catalog(tool_defs: List[Dict[str, Any]]) -> List[CatalogEntry]:
    """Build the deferred-tool catalog from the deferrable subset of tool-defs."""
    catalog: List[CatalogEntry] = []
    for td in tool_defs:
        fn = _fn(td)
        name = fn.get("name", "")
        if not name:
            continue
        source, source_name = _classify_source(name)
        # Index the human-facing label ("linear", not "mcp-linear").
        source_label = _listing_group_label(source_name) if source_name else ""
        catalog.append(CatalogEntry(
            name=name, description=fn.get("description", "") or "", schema=td, source=source,
            source_name=source_name, _tokens=_tokenize(_entry_search_text(td, source_label))))
    return catalog


def _bm25_score(query_tokens: List[str], doc_tokens: List[str], doc_lengths: List[int],
                avg_dl: float, doc_freq: Dict[str, int], n_docs: int, k1: float = 1.5,
                b: float = 0.75) -> float:
    """Standard BM25 for one query against one document (inlined; the catalog is bounded —
    typically < 500 tools — so a dependency is not worth it)."""
    score = 0.0
    dl = len(doc_tokens)
    doc_tf = Counter(doc_tokens)
    for q in query_tokens:
        df, tf = doc_freq.get(q, 0), doc_tf.get(q, 0)
        if df and tf:
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0)))
    return score


_CorpusStats = Tuple[List[int], float, Dict[str, int], int]  # doc_lengths, avg_dl, df, n_docs


def _corpus_stats(catalog: List[CatalogEntry]) -> _CorpusStats:
    """Compute the BM25 statistics shared by every query over a catalog."""
    doc_lengths = [len(entry._tokens) for entry in catalog]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_freq = Counter(tok for entry in catalog for tok in set(entry._tokens))
    return doc_lengths, avg_dl, dict(doc_freq), len(catalog)


def _gate_token(query_tokens: List[str], doc_freq: Dict[str, int], n_docs: int) -> str:
    """The query token with the highest IDF: the word that names the intent. ``send``,
    ``read``, ``create`` sit in dozens of tool documents and separate nothing; ``gmail``,
    ``github``, ``incident`` sit in a few and separate everything. A document without this
    token answered a different question, however many common tokens it shares."""
    def _idf(token: str) -> float:
        df = doc_freq.get(token, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    return max(query_tokens, key=_idf)


# Relevance floor. The rarest-token gate stops queries whose intent word no tool carries; it
# does not stop a long hunt whose every word exists SOMEWHERE in the catalog while no single
# tool carries more than one of them (observed: "run shell command execute code python" -> a
# workflow-rerun tool sharing only "run"; the model read "results exist" as "it is in here"
# and re-searched 216 times). A document must also match MIN_QUERY_TERM_COVERAGE of the
# query's ANSWERABLE terms (present in at least one document) before it is offered. Coverage
# only engages from MIN_ANSWERABLE_TERMS_FOR_COVERAGE terms up: short queries ("list issues")
# legitimately differ from a tool by a word, and are where a coverage rule costs real recall.
MIN_QUERY_TERM_COVERAGE = 0.5
MIN_ANSWERABLE_TERMS_FOR_COVERAGE = 4


def _required_term_coverage(answerable_term_count: int) -> int:
    """How many of a query's ANSWERABLE unique terms a document must match to be offered:
    one below the engagement floor, else at least half, rounded up."""
    if answerable_term_count < MIN_ANSWERABLE_TERMS_FOR_COVERAGE:
        return 1
    return math.ceil(answerable_term_count * MIN_QUERY_TERM_COVERAGE)


# Recorded-outcome weighting. Hermes persists every tool result already (``messages``:
# ``role='tool'``, ``tool_name``, ``content``) and never reads it back into retrieval: a tool
# that fails most of the time is still the best lexical match for its query, so it keeps
# winning the slot and keeps failing. This is AnyTool's ``ToolQualityManager.adjust_ranking``
# in narrow form — a post-hoc multiplier over outcomes Hermes already recorded. It is applied
# AFTER admission, so it re-orders the hits lexical search admitted and can never hide one.
MIN_CALLS_FOR_RELIABILITY = 3      # no penalty without evidence: a new tool is not a bad tool
RELIABILITY_HEALTHY_RATE = 0.4     # at or above this success rate a tool is never penalised
RELIABILITY_FLOOR = 0.2            # hard clamp: the worst tool still ranks, just last
RELIABILITY_WINDOW_DAYS = 30       # older outcomes say nothing about today's tool
RELIABILITY_MAX_CALLS = 2000       # bounded read; the newest recorded calls are the evidence
# The read walks only the newest N message rows (a rowid range scan), so its cost tracks this
# number rather than the size of the state DB: 8ms over 8k rows, vs 1.6s cold-scanning a 154MB
# DB that has no index on `role`. A tool quieter than this window has no evidence yet.
RELIABILITY_SCAN_ROWS = 8000


def _clamp_reliability(value: Any) -> float:
    """Clamp a multiplier into ``[RELIABILITY_FLOOR, 1.0]``; junk is neutral.

    A caller-supplied mapping is untrusted input and no value it can hold may demote a tool
    out of the results, so the floor is enforced here rather than trusted from the caller.
    """
    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(multiplier):
        return 1.0
    return min(1.0, max(RELIABILITY_FLOOR, multiplier))


def tool_reliability_multiplier(successes: int, calls: int, consecutive_failures: int = 0) -> float:
    """Score multiplier in ``[RELIABILITY_FLOOR, 1.0]`` for one tool's recorded outcomes.

    Neutral below :data:`MIN_CALLS_FOR_RELIABILITY` calls (insufficient evidence) and at or
    above :data:`RELIABILITY_HEALTHY_RATE` successes; below that it slides from 1.0 at the
    threshold to 0.3 at a zero success rate, with a further penalty once a tool has failed its
    last three calls in a row (a run of failures is a stronger signal than a low average).
    """
    if calls < MIN_CALLS_FOR_RELIABILITY:
        return 1.0
    success_rate = successes / calls
    if success_rate >= RELIABILITY_HEALTHY_RATE:
        return 1.0
    multiplier = 0.3 + (success_rate / RELIABILITY_HEALTHY_RATE) * 0.7
    if consecutive_failures >= 3:  # the newest calls all failed: still broken, not just patchy
        multiplier -= min(0.3, (consecutive_failures - 2) * 0.1)
    return _clamp_reliability(multiplier)


def _result_failed(content: Any) -> bool:
    """Whether one recorded tool result is a failure.

    Structural signals first: the ``{"output", "exit_code", "error"}`` envelope the terminal
    and exec tools persist, where a non-zero exit code is the canonical failure signal. Results
    that are not an envelope fall back to the shared text heuristic (MCP tools return prose).
    Mirrors the signals in ``agent.display._detect_tool_failure``; that module (i18n, rich) is
    not imported here because this runs inside the retrieval path.
    """
    if not isinstance(content, str) or not content:
        return False
    try:
        data = json.loads(content)
    except ValueError:
        data = None
    if isinstance(data, dict):
        exit_code = data.get("exit_code")
        return bool(
            data.get("error")
            or data.get("success") is False
            or (isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0)
        )
    return _looks_like_error_output(content)


_RELIABILITY_SQL = (
    "SELECT tool_name, content FROM messages"
    " WHERE id > (SELECT MAX(id) FROM messages) - ?"
    " AND role = 'tool' AND tool_name IS NOT NULL AND timestamp >= ?"
    " ORDER BY id DESC LIMIT ?"
)


def _recent_tool_results(db_path: Optional[Path], *, days: int, max_calls: int,
                         now: Optional[float]) -> List[Tuple[Any, Any]]:
    """Newest recorded ``(tool_name, content)`` tool results, or ``[]``.

    Every failure mode returns empty history — no state DB, a schema without ``tool_name``, a
    locked or half-written file, a pytest process resolving the production DB (the state
    layer's own live-DB guard) — because the caller then ranks exactly as it did before, which
    is the only safe direction for this to fail. Read-only, no write lock, bounded row count.
    """
    try:
        from hermes_state import _default_db_path, _ensure_test_isolation
        from hermes_state_holders import read_only_db_uri

        path = Path(db_path) if db_path is not None else _default_db_path()
        _ensure_test_isolation(path)
        cutoff = (time.time() if now is None else now) - days * 86400
        conn = sqlite3.connect(read_only_db_uri(path), uri=True, timeout=1.0)
        try:
            return conn.execute(
                _RELIABILITY_SQL, (RELIABILITY_SCAN_ROWS, cutoff, max_calls)).fetchall()
        finally:
            conn.close()
    except Exception:  # a broken state DB is not a search failure: history stays empty
        logger.debug("tool reliability history unavailable; ranking stays neutral", exc_info=True)
        return []


def load_tool_reliability(db_path: Optional[Path] = None, *, days: int = RELIABILITY_WINDOW_DAYS,
                          max_calls: int = RELIABILITY_MAX_CALLS,
                          now: Optional[float] = None) -> Dict[str, float]:
    """Reliability multiplier per tool name, derived from the outcomes Hermes already recorded.

    ``{tool_name: multiplier}`` for every tool called inside the window; a tool with no
    history is absent and ranks neutrally. Call this ONCE per search dispatch and pass the
    result to :func:`search_catalog` — the ranking loop must never touch the database.
    Read-only, bounded, and never raises: no history means no change to the ranking.
    """
    calls: Dict[str, int] = {}
    successes: Dict[str, int] = {}
    failing_streak: Dict[str, int] = {}
    streak_over: set = set()  # a newer success ended this tool's run of failures
    for name, content in _recent_tool_results(db_path, days=days, max_calls=max_calls, now=now):
        if not isinstance(name, str) or not name:
            continue
        calls[name] = calls.get(name, 0) + 1
        if _result_failed(content):
            if name not in streak_over:  # rows arrive newest first
                failing_streak[name] = failing_streak.get(name, 0) + 1
        else:
            successes[name] = successes.get(name, 0) + 1
            streak_over.add(name)
    return {
        name: tool_reliability_multiplier(successes.get(name, 0), count, failing_streak.get(name, 0))
        for name, count in calls.items()
    }


def search_catalog(catalog: List[CatalogEntry], query: str, limit: int = 5, *,
                   corpus_stats: Optional[_CorpusStats] = None,
                   reliability: Optional[Dict[str, Any]] = None) -> List[CatalogEntry]:
    """Top-``limit`` catalog entries for ``query`` by BM25 (exact name match ranks first).

    Admission is by the query's rarest token (:func:`_gate_token`), not by ``score > 0``:
    BM25 is additive over the tokens a document shares with the query, so on a large catalog
    ``score > 0`` admits one-token matches and fills every slot with them (measured: "send
    gmail email" returned 5 incident tools that only shared ``email``). A token no document
    carries admits nothing; the caller's empty-group hint tells the model to retry without it.
    Long queries additionally need :func:`_required_term_coverage` of their answerable terms.

    ``reliability`` (:func:`load_tool_reliability`) re-orders the admitted hits by how often
    each tool has actually worked. It is applied AFTER the gate, so it can demote a hit but
    never remove one; an exact-name hit keeps its infinite score and therefore its first place.
    ``None`` — the default — ranks exactly as before."""
    query_tokens = _tokenize(query) if catalog and limit > 0 else []
    if not query_tokens:
        return []
    corpus_stats = corpus_stats or _corpus_stats(catalog)
    doc_freq = corpus_stats[2]
    gate = _gate_token(query_tokens, doc_freq, corpus_stats[3])
    answerable = {t for t in query_tokens if doc_freq.get(t, 0) > 0}
    required_terms = _required_term_coverage(len(answerable))
    exact_name = query.strip().lower()

    def _admitted(entry: CatalogEntry) -> bool:
        """Carries the intent word AND enough of the answerable terms (exact name is exempt)."""
        if entry.name.lower() == exact_name:
            return True
        tokens = set(entry._tokens)
        return gate in tokens and sum(1 for t in answerable if t in tokens) >= required_terms

    scored = [
        (float("inf") if entry.name.lower() == exact_name
         else _bm25_score(query_tokens, entry._tokens, *corpus_stats), entry)
        for entry in catalog if _admitted(entry)]
    if reliability:
        scored = [(score * _clamp_reliability(reliability.get(entry.name, 1.0)), entry)
                  for score, entry in scored]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


# Sentence end: ., !, ? followed by whitespace/EOS, not inside e.g./i.e./etc.
_SENTENCE_END_RE = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)[.!?](?=\s|$)")


def _short_desc(description: str, max_chars: int = 60) -> str:
    """First sentence of a tool description, clipped to ``max_chars`` on a word boundary.
    ``e.g.``/``i.e.``/``etc.`` do not end a sentence; whitespace normalization and the regex
    search stay linear-time on hostile input."""
    text = " ".join((description or "").split())
    m = _SENTENCE_END_RE.search(text)
    text = text[:m.end()] if m else text
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    clipped = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped
    return clipped.rstrip(",;: ") + "…"


def _listing_group_label(source_name: str) -> str:
    """Human-facing group heading for a toolset, e.g. ``mcp-github`` -> ``github``."""
    label = source_name or "other"
    return label[4:] if label.startswith("mcp-") else label


def hidden_declared_sources() -> List[Dict[str, Any]]:
    """Return deterministic summaries for declared MCP servers hidden by their check."""
    from hermes_platform import declaration
    from tools.mcp_liveness import unavailable_details
    from tools.registry import registry

    grouped: Dict[str, List[Any]] = {}
    for entry in registry.get_all_entries():
        if entry.toolset.startswith("mcp-"):
            grouped.setdefault(entry.toolset[4:], []).append(entry)
    rows: List[Dict[str, Any]] = []
    for server_name in sorted(grouped):
        if declaration.lookup(server_name) is None:
            continue
        entries = grouped[server_name]
        if any(entry.check_fn is None or bool(entry.check_fn()) for entry in entries):
            continue
        details = unavailable_details(server_name)
        if details is None:
            continue
        _decl, _current, sentence = details
        rows.append({"name": server_name, "tool_count": len(entries), "unavailable": sentence})
    return rows


def build_catalog_listing_with_form(
    deferrable: List[Dict[str, Any]], *, max_tokens: int = 4000) -> Tuple[Optional[str], str]:
    """Render the deferred-catalog manifest: ``- name: short desc`` lines grouped per source.
    Returns ``(text, form)``; form is ``"full"``, ``"names"``, ``"mixed"`` (oversized servers
    collapsed to a name + count line), ``"groups"`` (every server summarized) or ``"none"``
    (over budget even summarized -> text is None). Ordering is deterministic (sorted groups
    and tools) so the block is byte-stable — the request prefix stays cacheable. Degradation
    is PER SERVER, largest first: one huge server must not cost a small one its listing."""
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for td in deferrable:
        fn = _fn(td)
        name = fn.get("name", "")
        if name:
            # _classify_source gives ("other", "") when unregistered; the label of "" is "other".
            label = _listing_group_label(_classify_source(name)[1])
            groups.setdefault(label, []).append((name, _short_desc(fn.get("description", ""))))
    unavailable = hidden_declared_sources()
    if not groups and not unavailable:
        return None, "none"

    def render_group(label: str, mode: str) -> str:
        """Render one server's block. mode: 'full' | 'names' | 'summary'."""
        tools = sorted(groups[label])
        if mode == "summary":
            return (f"{label} ({len(tools)} tools — names not listed; "
                    f"discover via `{TOOL_SEARCH_NAME}`)")
        lines = [f"{label} tools ({len(tools)}):"]
        if mode == "full":
            lines.extend(f"- {name}: {desc}" if desc else f"- {name}" for name, desc in tools)
        else:
            lines.append(", ".join(name for name, _ in tools))
        return "\n".join(lines)

    header = ("Deferred tool catalog (call schemas via "
              f"`{TOOL_DESCRIBE_NAME}`, invoke via `{TOOL_CALL_NAME}`):")

    def assemble_if_fits(modes: Dict[str, str]) -> Optional[str]:
        available_blocks = {label: render_group(label, modes[label]) for label in groups}
        unavailable_blocks = {
            row["name"]: (
                f"{row['name']} ({row['tool_count']} tools unavailable: {row['unavailable']})"
                if row.get("tool_count") is not None
                else f"{row['name']} (tools unavailable: {row['unavailable']})"
            )
            for row in unavailable
        }
        blocks = [
            available_blocks[label] if label in available_blocks else unavailable_blocks[label]
            for label in sorted(available_blocks | unavailable_blocks)
        ]
        text = "\n".join([header] + blocks)
        return text if math.ceil(len(text) / CHARS_PER_TOKEN) <= max_tokens else None

    for mode in ("full", "names"):  # 1. everything full; 2. everything names-only
        modes = {lbl: mode for lbl in groups}
        text = assemble_if_fits(modes)
        if text is not None:
            return text, mode
    # 3. Collapse the LARGEST rendered groups first (deterministic: size then label).
    for lbl in sorted(groups, key=lambda lbl: (-len(render_group(lbl, "names")), lbl)):
        modes[lbl] = "summary"
        text = assemble_if_fits(modes)
        if text is not None:
            return text, "groups" if all(m == "summary" for m in modes.values()) else "mixed"
    return None, "none"
