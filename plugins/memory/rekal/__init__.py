"""Rekal memory plugin — Hermes host for the upstream MemPalace/Rekal engine.

The engine (``mempalace_rekal_engine.py``) and the retrieval primitives
(``mempalace_hindsight.py``) are the current upstream implementations from
OpenCode, kept verbatim apart from import/packaging shims.  This module is the
Hermes integration layer that replaces OpenCode's bridge subprocess, TypeScript
tool and session hooks:

* one ``mempalace`` tool carrying the upstream operation set and result
  formatting (``mempalace.ts`` / ``mempalace_bridge.py``),
* automatic recall through ``prefetch()`` using the upstream advanced retrieval
  path (rank fusion + graph expansion + temporal analysis),
* automatic capture and conversation offloading through ``sync_turn()``,
  ``on_session_end()`` and ``on_pre_compress()``,
* the upstream memory protocol as the tool description, plus the
  ``<memory_system>`` trust block in the system prompt.

MemPalace's own drawer/KG/diary operations are optional: when the ``mempalace``
pip package is not installed those operations return a clean install hint and
the zero-dependency Rekal engine keeps working.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from plugins.memory.rekal.mempalace_rekal_engine import RekalEngine
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 10

_TRIVIAL_RE = re.compile(
    r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np|done|perfect|great|nice)\.?$",
    re.IGNORECASE,
)

_CONTEXT_STRIP_RE = re.compile(
    r"<(?:mempalace_context|rekal-context)>[\s\S]*?</(?:mempalace_context|rekal-context)>\s*",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# MemPalace optional dependency (drawer / knowledge-graph / diary operations)
# ---------------------------------------------------------------------------

_MEMPALACE_IMPORT_ERROR: Optional[str] = None
_mcp_mod: Any = None
_kg: Any = None
_palace_config: Any = None
_search_memories: Any = None


def _load_mempalace() -> None:
    """Import the optional mempalace package once, mirroring the bridge.

    Lazy on purpose: importing mempalace pulls in ChromaDB and the embedding
    stack, which is too heavy for provider listing/setup paths.  A failure is
    recorded and surfaced only when a MemPalace-native operation is invoked.
    """
    global _MEMPALACE_IMPORT_ERROR, _mcp_mod, _kg, _palace_config, _search_memories
    if _mcp_mod is not None or _MEMPALACE_IMPORT_ERROR is not None:
        return
    try:
        from mempalace import mcp_server as mcp_mod
        from mempalace.searcher import search_memories as search_fn

        _mcp_mod = mcp_mod
        _search_memories = search_fn
        _kg = getattr(mcp_mod, "_kg", None) or getattr(mcp_mod, "KnowledgeGraph", None)
        cfg = getattr(mcp_mod, "_config", None) or getattr(mcp_mod, "MempalaceConfig", None)
        if callable(cfg) and not isinstance(cfg, dict):
            try:
                cfg = cfg()
            except Exception:
                pass
        _palace_config = cfg
    except Exception as exc:
        _MEMPALACE_IMPORT_ERROR = str(exc)


def _require_mempalace() -> None:
    if _mcp_mod is None:
        raise RuntimeError(
            f"mempalace import failed: {_MEMPALACE_IMPORT_ERROR or 'not installed'}. "
            "Install with: pip install mempalace"
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _default_config() -> dict:
    return {
        "data_dir": "rekal",
        "auto_recall": True,
        "auto_capture": True,
        "max_results": _DEFAULT_MAX_RESULTS,
    }


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _load_rekal_config(hermes_home: str) -> dict:
    config = _default_config()
    config_path = Path(hermes_home) / "rekal.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config.update({k: v for k, v in raw.items() if v is not None})
        except Exception:
            logger.debug("Failed to parse %s", config_path, exc_info=True)
    config["auto_recall"] = _as_bool(config.get("auto_recall"), True)
    config["auto_capture"] = _as_bool(config.get("auto_capture"), True)
    try:
        config["max_results"] = max(1, min(50, int(config.get("max_results", _DEFAULT_MAX_RESULTS))))
    except Exception:
        config["max_results"] = _DEFAULT_MAX_RESULTS
    return config


def _save_rekal_config(values: dict, hermes_home: str) -> None:
    config_path = Path(hermes_home) / "rekal.json"
    existing = {}
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    existing.update(values)
    from utils import atomic_json_write

    atomic_json_write(config_path, existing, mode=0o600, sort_keys=True)


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def _clean_text_for_capture(text: str) -> str:
    return _CONTEXT_STRIP_RE.sub("", text or "").strip()


def _is_trivial_message(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


def _format_relative_time(iso_timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        seconds = (now - dt).total_seconds()
        if seconds < 1800:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        if seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        if seconds < 604800:
            return f"{int(seconds / 86400)}d ago"
        if dt.year == now.year:
            return dt.strftime("%d %b")
        return dt.strftime("%d %b %Y")
    except Exception:
        return ""


def _format_prefetch_context(results: List[Dict[str, Any]], max_results: int) -> str:
    if not results:
        return ""
    lines = []
    for item in results[:max_results]:
        content = item.get("content", "")
        if not content:
            continue
        mtype = item.get("memory_type", "fact")
        rel = _format_relative_time(item.get("updated_at", ""))
        prefix_bits = []
        if rel:
            prefix_bits.append(f"[{rel}]")
        try:
            importance = float(item.get("importance", 0) or 0)
        except (TypeError, ValueError):
            importance = 0.0
        if importance >= 0.7:
            prefix_bits.append("[important]")
        prefix = " ".join(prefix_bits)
        lines.append(f"- {prefix} [{mtype}] {content}".strip())
    if not lines:
        return ""
    intro = (
        "The following is background context from long-term memory. Use it silently when relevant. "
        "Do not force memories into the conversation."
    )
    body = "\n".join(lines)
    return f"<mempalace_context>\n{intro}\n\n{body}\n</mempalace_context>"


# ---------------------------------------------------------------------------
# Tool description — upstream mempalace.txt, verbatim
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTION = """SINGLE TOOL called "mempalace". Do NOT call operations below as separate tools — always call mempalace with the "operation" parameter set to the desired operation name.

Persistent AI memory system. You have two memory layers: the conversation prompt (short-term) and mempalace (long-term). Use them together to minimize token usage while maintaining full recall.

## HYBRID MEMORY PROTOCOL — Follow This Every Session

### Phase 1: Session Bootstrap (MANDATORY — do first)
Call session_init with your current task. This returns prior memories, recent diary entries, conflicts, and timeline in one call. Read the result before doing anything else.

If session_init is unavailable, call build_context instead.

### Phase 2: Active Work (during the session)
While working, follow these rules:

SHORT-TERM (stays in prompt):
- Current user message and your response
- Active file contents being edited
- Tool outputs from the current task
- Errors being debugged right now

LONG-TERM (goes to mempalace immediately):
- User states a preference → memory_store(memory_type="preference")
- Architecture or convention discovered → memory_store(memory_type="fact")
- Decision made with reasoning → memory_store(memory_type="fact")
- Bug with non-obvious cause → memory_store(memory_type="episode")
- Procedure or workflow described → memory_store(memory_type="procedure")
- Task completed or milestone reached → diary_write

### Phase 3: Missing Context Detection
When you realize you're missing context:
1. Check: is it in the current prompt? → Use it directly
2. Not in prompt? → Call memory_recall (or build_context) to retrieve from mempalace
3. Still not found? → Ask the user or use file tools

NEVER guess when mempalace might have the answer. Query it.

### Phase 4: Conversation Offloading
When conversation grows long (many turns), call ingest_turns to compress and store older conversation content. This preserves the knowledge without keeping it in the prompt.

Call ingest_turns when:
- Conversation exceeds ~15 turns
- Switching to a different task within the same session
- Before a complex multi-step operation that needs prompt space
- User says "let's move on" or changes topic

### Phase 5: Session End
Before the session ends:
1. Call ingest_turns for any un-stored conversation content
2. Call diary_write summarizing what was accomplished
3. Store any final decisions or discoveries via memory_store

## Operations

### Session lifecycle
- session_init: Bootstrap a session. Returns relevant memories + diary + conflicts + timeline. CALL THIS FIRST.
- build_context: Load relevant memories for a query. Use mid-session when switching tasks.
- ingest_turns: Compress and store conversation turns into structured memories. Call periodically to offload prompt.
- diary_write / diary_read: Session journal for continuity.

### MemPalace native (verbatim drawers)
- search: Semantic search over stored drawers
- store: File verbatim content into a wing/room
- status, list_wings, list_rooms: Palace overview
- kg_add, kg_query, kg_invalidate: Knowledge graph entity relationships

### Rekal engine (structured memories with lifecycle)
- memory_store: Store a typed, tagged memory. Pass memory_id-free content plus
  memory_type, project, tags, importance (0-1).
- memory_search: Hybrid search (BM25 + vector + recency) with configurable weights
- memory_recall: PREFERRED for retrieval. Same store, stronger ranking — rank
  fusion across keyword/vector/graph/time, associative expansion through links,
  and natural-language time windows. Use this when you want the best answer and
  memory_search when you need the exact legacy scoring.
- memory_update: Edit an existing memory (requires memory_id)
- memory_supersede: Replace a memory while preserving history (requires old_id).
  ALWAYS prefer over delete+store.
- memory_delete: Remove a memory (requires memory_id)
- memory_link: Connect memories (from_id, to_id, link_relation)
- memory_conflicts: Find contradicting pairs
- memory_health: Database statistics
- memory_similar, memory_topics, memory_timeline, memory_related: Introspection
- set_config: Per-project scoring weights (project, key, value)

### Retrieval tuning (memory_recall / memory_search)
- fusion="rrf": combine retrieval strategies by rank instead of raw score.
- graph_expand=true: also return memories linked to the top hits, even when they
  match neither the text nor the vector query. This is how you recover context
  that uses different vocabulary than your question.
- temporal=true: read a time window out of the query ("yesterday", "last week",
  "March 2026", "3 days ago") and prefer memories from that period. The time
  words are removed from the keyword match, so they never suppress results.

### Evidence and contradictions (automatic)
- Storing content that already exists does not create a duplicate; it increments
  that memory's proof_count. Repeatedly observed facts rank higher. Do not try
  to avoid re-storing something you have seen again — the repetition is signal.
- Storing content that negates an existing memory automatically records a
  `contradicts` link and returns it under `contradicts`. Nothing is deleted:
  review the pair and call memory_supersede if the new statement wins.

### Verification
- contradiction_check: Check before storing new facts
- fact_check: Validate a claim against KG + memory
- multi_hop: Find indirect entity relationships

## Storing Rules

### ALWAYS search before storing
1. mempalace operation=memory_recall query="<topic>" limit=5
2. Duplicate exists → skip
3. Same topic, outdated → memory_supersede(old_id, content)
4. No match → memory_store(content, memory_type, tags)

### Distill before storing
NEVER store raw dialogue. Extract the durable fact, compress.
Drop: articles, filler, hedging. Keep: technical terms, proper nouns, reasons.
One memory = one fact, 1-2 sentences.

### memory_type: fact | preference | procedure | context | episode
### tags: 2-4 specific tags, never generic

## Token Optimization Strategy

As conversation grows:
1. Early turns (1-10): Everything fits in prompt. Store decisions/preferences to mempalace as they occur.
2. Mid conversation (10-20): Call ingest_turns on turns 1-10. Prompt keeps only recent turns + session_init context.
3. Long conversation (20+): Call ingest_turns again. Rely heavily on memory_search for historical context. Prompt holds only current task + last few turns.

The goal: a 100-turn conversation should use roughly the same prompt tokens as a 10-turn conversation, because the first 90 turns are in mempalace and retrievable on demand.

## What NOT to Store
- Transient state ("currently editing X")
- Trivially re-discoverable facts (line numbers)
- Vague platitudes ("user likes clean code")
- Secrets, API keys, passwords — never
- Content already in the current prompt (wasteful duplication)

## Validation
NEVER blindly trust retrieved memory. Cross-reference against current file state when uncertain. If memory contradicts current code, trust the code.
"""

_SYSTEM_PROMPT_BLOCK = """<memory_system>
You have access to MemPalace — a persistent, local memory system that survives across sessions.
Relevant memories from previous sessions may be automatically injected into your context inside <mempalace_context> tags.

Rules for using retrieved memories:
- NEVER blindly trust retrieved memory. Always validate against current file state and session context.
- If a memory contradicts current code or files, trust the code — the memory may be stale.
- Use the mempalace tool explicitly when you need to store important decisions, search for specific history, or manage the knowledge graph.
- Significant discoveries, architecture decisions, and resolved bugs are automatically stored after each response.
- You can proactively store or search memory using the mempalace tool when automatic retrieval is insufficient.
</memory_system>"""

# ---------------------------------------------------------------------------
# Tool schema — upstream mempalace.ts parameter contract
# ---------------------------------------------------------------------------

_OPERATIONS = [
    "search", "smart_search", "store", "status", "list_wings", "list_rooms",
    "kg_add", "kg_query", "kg_invalidate", "contradiction_check", "fact_check",
    "multi_hop", "diary_write", "diary_read", "build_context", "ingest_turns",
    "memory_conflicts", "memory_delete", "memory_health", "memory_link",
    "memory_recall", "memory_related", "memory_search", "memory_similar",
    "memory_store", "memory_supersede", "memory_timeline", "memory_topics",
    "memory_update", "session_init", "set_config",
]

_MEMORY_TYPES = ["fact", "preference", "procedure", "context", "episode"]
_LINK_RELATIONS = ["supersedes", "contradicts", "related_to"]

MEMPALACE_SCHEMA = {
    "name": "mempalace",
    "description": _TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(_OPERATIONS),
                "description": "The operation to perform.",
            },
            "query": {"type": "string"},
            "content": {"type": "string"},
            "wing": {"type": "string"},
            "room": {"type": "string"},
            "drawer": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "number"},
            "entity": {"type": "string"},
            "relation": {"type": "string"},
            "target": {"type": "string"},
            "confidence": {"type": "number"},
            "entry": {"type": "string"},
            "depth": {"type": "number"},
            "agent_name": {"type": "string"},
            "statement": {"type": "string"},
            "claim": {"type": "string"},
            "expand_with_kg": {"type": "boolean"},
            "memory_id": {"type": "string"},
            "old_id": {"type": "string"},
            "from_id": {"type": "string"},
            "to_id": {"type": "string"},
            "link_relation": {"type": "string", "enum": list(_LINK_RELATIONS)},
            "memory_type": {"type": "string", "enum": list(_MEMORY_TYPES)},
            "project": {"type": "string"},
            "importance": {"type": "number"},
            "fusion": {"type": "string", "enum": ["rrf"]},
            "graph_expand": {"type": "boolean"},
            "temporal": {"type": "boolean"},
            "strategy_boosts": {"type": "object"},
            "turns": {"type": "array", "items": {"type": "string"}},
            "topic": {"type": "string"},
            "task": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "as_of": {"type": "string"},
            "direction": {"type": "string", "enum": ["in", "out", "both"]},
            "valid_from": {"type": "string"},
            "ended": {"type": "string"},
            "key": {"type": "string"},
            "value": {"type": "number"},
            "w_fts": {"type": "number"},
            "w_vec": {"type": "number"},
            "w_recency": {"type": "number"},
            "w_access": {"type": "number"},
            "half_life": {"type": "number"},
        },
        "required": ["operation"],
    },
}


# ---------------------------------------------------------------------------
# Result formatting — upstream mempalace.ts switch
# ---------------------------------------------------------------------------

def _json_output(raw: Any) -> str:
    return json.dumps(raw, indent=2, default=str)


def _format_search_results(data: Any, adaptive: bool = False) -> str:
    envelope = data if isinstance(data, dict) else {}
    results = envelope.get("results") or []

    if not results:
        return "No memories found matching your query."

    header_parts: List[str] = []
    if adaptive:
        if envelope.get("total_drawers"):
            header_parts.append(f"Palace: {envelope['total_drawers']} drawers")
        if envelope.get("adaptive_k"):
            header_parts.append(f"Candidates scanned: {envelope['adaptive_k']}")
        if envelope.get("kg_expanded"):
            header_parts.append("Query expanded via KG")
    header = (" | ".join(header_parts) + "\n\n") if header_parts else ""

    formatted = []
    for i, r in enumerate(results):
        r = r if isinstance(r, dict) else {}
        similarity = f"{r['similarity']:.3f}" if isinstance(r.get("similarity"), (int, float)) else "?"
        bm25 = f"{r['bm25_score']:.2f}" if isinstance(r.get("bm25_score"), (int, float)) else ""
        wing = r.get("wing") or "unknown"
        room = r.get("room") or "unknown"
        source = r.get("source_file") or ""
        created = r.get("created_at") or ""
        text = r.get("text") or ""
        via = r.get("matched_via") or "drawer"

        line = f"[{i + 1}] sim={similarity}"
        if bm25:
            line += f" bm25={bm25}"
        if adaptive and isinstance(r.get("adaptive_score"), (int, float)):
            line += f" adaptive={r['adaptive_score']:.3f}"
        if adaptive and isinstance(r.get("recency_score"), (int, float)):
            line += f" recency={r['recency_score']:.2f}"
        line += f" | {wing}/{room}"
        if source and source != "?":
            line += f" | {source}"
        if created and created != "unknown":
            line += f" | {created}"
        line += f" ({via})"

        formatted.append(f"{line}\n    {str(text)[:2000]}")

    return header + "\n\n".join(formatted)


def _format_memory_recall(data: Any) -> str:
    envelope = data if isinstance(data, dict) else {}
    results = envelope.get("results") or []
    retrieval = envelope.get("retrieval") or {}

    arms = retrieval.get("arms")
    window = retrieval.get("temporal_window")
    header = " | ".join(
        part
        for part in (
            f"mode: {retrieval.get('mode') or 'unknown'}",
            f"arms: {', '.join(arms)}" if isinstance(arms, list) else "",
            f"graph-expanded: {retrieval['graph_expanded']}" if retrieval.get("graph_expanded") else "",
            f"window: {' .. '.join(window)}" if isinstance(window, list) else "",
        )
        if part
    )

    lines = [header, ""]
    for i, r in enumerate(results):
        r = r if isinstance(r, dict) else {}
        try:
            score = float(r.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        try:
            proof = float(r.get("proof_count") or 0)
        except (TypeError, ValueError):
            proof = 0.0
        signals = " ".join(
            signal
            for signal in (
                f"score={score:.4f}",
                f"rrf#{r['rrf_rank']}" if r.get("rrf_rank") else "",
                f"graph={r['graph_score']}" if r.get("graph_score") else "",
                f"temporal={r['temporal_score']}" if r.get("temporal_score") else "",
                f"proof={r['proof_count']}" if proof > 1 else "",
                f"via={r['via']}" if r.get("via") else "",
            )
            if signal
        )
        lines.append(f"[{i + 1}] {signals}\n{str(r.get('content') or '')[:2000]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class RekalMemoryProvider(MemoryProvider):
    def __init__(self):
        self._config = _default_config()
        self._engine: Optional[RekalEngine] = None
        self._hermes_home = ""
        self._session_id = ""
        self._turn_count = 0
        self._active = False
        self._auto_recall = True
        self._auto_capture = True
        self._max_results = _DEFAULT_MAX_RESULTS
        self._write_enabled = True
        self._session_turns: List[Dict[str, str]] = []
        self._turns_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "rekal"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        _save_rekal_config(values, hermes_home)

    def get_status_config(self, provider_config: dict) -> dict:
        if not self._engine:
            return {"summary": "Not initialized"}
        h = self._engine.health()
        status = "Connected"
        if h.get("active_memories", 0) == 0:
            status = "No memories yet"
        return {
            "summary": (
                f"{status} · {h.get('active_memories', 0)} memories · "
                f"{h.get('total_links', 0)} links · {h.get('total_conflicts', 0)} conflicts"
            )
        }

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from hermes_cli.config import save_config

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = self.name
        save_config(config)
        print(f"\n  Memory provider: rekal")
        print("  Activation saved to config.yaml")
        print("  Start a new session to activate.\n")

    # -- Lifecycle ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._session_id = session_id
        self._turn_count = 0
        self._config = _load_rekal_config(self._hermes_home)

        _load_mempalace()
        data_dir = str(Path(self._hermes_home) / str(self._config.get("data_dir", "rekal")))

        # Mirror the bridge's palace resolution: reuse MemPalace's configured
        # palace when present, otherwise keep the Rekal store self-contained.
        if _palace_config is not None and hasattr(_palace_config, "palace_path") and not _palace_config.palace_path:
            try:
                _palace_config.palace_path = data_dir
            except Exception:
                pass
        palace_path = getattr(_palace_config, "palace_path", None) if _palace_config else None
        palace_path = palace_path or data_dir

        self._engine = RekalEngine(
            data_dir=data_dir,
            palace_path=palace_path,
            kg=_kg,
            search_memories_fn=_search_memories,
        )

        self._auto_recall = self._config["auto_recall"]
        self._auto_capture = self._config["auto_capture"]
        self._max_results = self._config["max_results"]

        with self._turns_lock:
            self._session_turns = []

        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = agent_context not in {"cron", "flush", "subagent"}
        self._active = True

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        return _SYSTEM_PROMPT_BLOCK

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall or not self._engine or not query.strip():
            return ""
        try:
            envelope = self._engine.search(
                query,
                limit=self._max_results,
                fusion="rrf",
                graph_expand=True,
                temporal=True,
            )
            results = envelope.get("results", []) if isinstance(envelope, dict) else []
            return _format_prefetch_context(results, self._max_results)
        except Exception:
            logger.debug("Rekal prefetch failed", exc_info=True)
            return ""

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = max(turn_number, 0)

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Any = None) -> None:
        if not self._active or not self._auto_capture or not self._write_enabled or not self._engine:
            return

        clean_user = _clean_text_for_capture(user_content)
        clean_assistant = _clean_text_for_capture(assistant_content)
        if not clean_user and not clean_assistant:
            return

        with self._turns_lock:
            if clean_user and not _is_trivial_message(clean_user):
                self._session_turns.append({"role": "user", "content": clean_user})
            if clean_assistant:
                self._session_turns.append({"role": "assistant", "content": clean_assistant})

    def _ingest_turns(self, turns: List[Dict[str, str]], *, session_id: str = "") -> None:
        if not self._engine or not turns:
            return
        try:
            self._engine.ingest_turns(turns, wing="project", room="conversations")
        except Exception:
            logger.debug("Rekal turn ingest failed (session=%s)", session_id, exc_info=True)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._active or not self._write_enabled or not self._engine:
            return
        with self._turns_lock:
            turns = list(self._session_turns)
            self._session_turns = []
        self._ingest_turns(turns, session_id=self._session_id)

    def on_session_switch(self, new_session_id: str, *,
                          parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        if not self._active or not self._write_enabled or not self._engine:
            self._session_id = str(new_session_id or "").strip() or self._session_id
            with self._turns_lock:
                self._session_turns = []
            return

        old_session_id = self._session_id
        with self._turns_lock:
            old_turns = list(self._session_turns)
            self._session_turns = []
        self._ingest_turns(old_turns, session_id=old_session_id)

        self._session_id = str(new_session_id or "").strip() or old_session_id
        self._turn_count = 0

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Offload turns about to be compressed into long-term memory."""
        if not self._active or not self._write_enabled or not self._engine:
            return ""
        turns: List[Dict[str, str]] = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                )
            content = _clean_text_for_capture(str(content or ""))
            if content:
                turns.append({"role": role, "content": content})
        self._ingest_turns(turns, session_id=self._session_id)
        return ""

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self._active or not self._write_enabled or not self._engine:
            return
        if action != "add" or not (content or "").strip():
            return
        try:
            self._engine.store(
                content.strip(),
                memory_type="fact",
                wing="project",
                room="memory_tool",
                tags=["memory_tool", str(target or "memory")],
            )
        except Exception:
            logger.debug("Rekal on_memory_write failed", exc_info=True)

    def backup_paths(self) -> List[str]:
        if self._engine:
            return [self._engine._db_path]
        return []

    def shutdown(self) -> None:
        if self._active and self._write_enabled and self._engine:
            with self._turns_lock:
                turns = list(self._session_turns)
                self._session_turns = []
            self._ingest_turns(turns, session_id=self._session_id)
        if self._engine:
            try:
                self._engine.checkpoint()
            except Exception:
                logger.debug("Rekal checkpoint failed", exc_info=True)
            try:
                self._engine.close()
            except Exception:
                logger.debug("Rekal close failed", exc_info=True)
        self._active = False

    # -- Tool surface ------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MEMPALACE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "mempalace":
            return tool_error(f"Unknown tool: {tool_name}")
        if not self._active or not self._engine:
            return tool_error("Rekal is not initialized")

        operation = str(args.get("operation") or "").strip()
        handler = getattr(self, f"_op_{operation}", None) if operation else None
        if handler is None:
            valid = ", ".join(sorted(_OPERATIONS))
            return tool_error(f"Unknown operation: {operation}. Valid: {valid}")

        try:
            return handler(args)
        except Exception as exc:
            logger.debug("mempalace operation '%s' failed", operation, exc_info=True)
            return tool_error(f"MemPalace operation '{operation}' failed: {exc}")

    # -- Operation helpers -------------------------------------------------

    @staticmethod
    def _int_arg(args: Dict[str, Any], key: str, default: int) -> int:
        try:
            return int(args.get(key, default))
        except (TypeError, ValueError):
            return default

    # -- MemPalace native operations --------------------------------------

    def _op_search(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_search(
            query=args.get("query", ""),
            limit=self._int_arg(args, "limit", 5),
            wing=args.get("wing") or None,
            room=args.get("room") or None,
        )
        return _format_search_results(raw)

    def _op_smart_search(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        query = args.get("query", "") or ""
        requested = args.get("limit")
        if requested is not None:
            limit = self._int_arg(args, "limit", 5)
        else:
            terms = len([t for t in query.split() if t])
            limit = 12 if terms <= 2 else (8 if terms <= 5 else 5)
        raw = _mcp_mod.tool_search(
            query=query,
            limit=limit,
            wing=args.get("wing") or None,
            room=args.get("room") or None,
        )
        return _format_search_results(raw, adaptive=True)

    def _op_store(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        wing = args.get("wing", "project")
        room = args.get("room", "general")
        raw = _mcp_mod.tool_add_drawer(
            wing=wing,
            room=room,
            content=args.get("content", ""),
            source_file=args.get("drawer") or None,
            added_by="hermes-agent",
        )
        if isinstance(raw, dict) and raw.get("success") is False:
            return f"Store failed: {raw.get('error') or 'unknown error'}"
        drawer_id = raw.get("drawer_id", "unknown") if isinstance(raw, dict) else "unknown"
        return f"Memory stored successfully.\n  ID: {drawer_id}\n  Location: {wing}/{room}"

    def _op_status(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_status()
        raw = raw if isinstance(raw, dict) else {}
        lines = [f"Total drawers: {raw.get('total_drawers', 0)}"]
        wings = raw.get("wings")
        if isinstance(wings, dict) and wings:
            lines.append("Wings: " + ", ".join(f"{k}({v})" for k, v in wings.items()))
        rooms = raw.get("rooms")
        if isinstance(rooms, dict) and rooms:
            lines.append("Rooms: " + ", ".join(f"{k}({v})" for k, v in rooms.items()))
        if raw.get("palace_path"):
            lines.append(f"Path: {raw['palace_path']}")
        if raw.get("vector_disabled"):
            reason = raw.get("vector_disabled_reason") or "run mempalace repair"
            lines.append(f"WARNING: Vector search disabled — {reason}")
        return "\n".join(lines)

    def _op_list_wings(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_list_wings()
        wings = raw.get("wings") if isinstance(raw, dict) else None
        if isinstance(wings, dict) and wings:
            return "\n".join(f"{name}: {count} drawers" for name, count in wings.items())
        return "No wings found. The palace is empty."

    def _op_list_rooms(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_list_rooms(wing=args.get("wing") or None)
        raw = raw if isinstance(raw, dict) else {}
        rooms = raw.get("rooms")
        wing_name = raw.get("wing") or args.get("wing") or "all"
        if isinstance(rooms, dict) and rooms:
            return "\n".join(f"{name}: {count} drawers" for name, count in rooms.items())
        return "No rooms found."

    def _op_kg_add(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_kg_add(
            subject=args.get("entity", ""),
            predicate=args.get("relation", ""),
            object=args.get("target", ""),
            valid_from=args.get("valid_from"),
        )
        if isinstance(raw, dict) and raw.get("success") is False:
            return f"Failed: {raw.get('error') or 'unknown error'}"
        fact = raw.get("fact") if isinstance(raw, dict) else None
        if not fact:
            fact = f"{args.get('entity')} → {args.get('relation')} → {args.get('target')}"
        return f"Added: {fact}"

    def _op_kg_query(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_kg_query(
            entity=args.get("entity", ""),
            as_of=args.get("as_of"),
            direction=args.get("direction", "both"),
        )
        facts = raw.get("facts") if isinstance(raw, dict) else None
        if not facts:
            return f"No facts found for entity: {args.get('entity')}"
        lines = []
        for fact in facts:
            fact = fact if isinstance(fact, dict) else {}
            subject = fact.get("subject") or fact.get("entity") or "?"
            predicate = fact.get("predicate") or fact.get("relation") or "?"
            obj = fact.get("object") or fact.get("target") or "?"
            suffix = ""
            if fact.get("valid_from"):
                suffix += f" (from: {fact['valid_from']})"
            if fact.get("ended"):
                suffix += f" [ended: {fact['ended']}]"
            lines.append(f"{subject} —[{predicate}]→ {obj}{suffix}")
        return "\n".join(lines)

    def _op_kg_invalidate(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_kg_invalidate(
            subject=args.get("entity", ""),
            predicate=args.get("relation", ""),
            object=args.get("target", ""),
            ended=args.get("ended"),
        )
        if isinstance(raw, dict) and raw.get("success") is False:
            return f"Failed: {raw.get('error') or 'unknown error'}"
        fact = raw.get("fact") if isinstance(raw, dict) else None
        if not fact:
            fact = f"{args.get('entity')} → {args.get('relation')} → {args.get('target')}"
        return f"Invalidated: {fact}"

    def _op_diary_write(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_diary_write(
            agent_name=args.get("agent_name", "hermes"),
            entry=args.get("entry", ""),
            topic=args.get("topic", "general"),
            wing=args.get("wing", ""),
        )
        if isinstance(raw, dict) and raw.get("success") is False:
            return f"Failed: {raw.get('error') or 'unknown error'}"
        raw = raw if isinstance(raw, dict) else {}
        return (
            "Diary entry logged.\n"
            f"  ID: {raw.get('entry_id', '?')}\n"
            f"  Agent: {raw.get('agent', args.get('agent_name', 'hermes'))}\n"
            f"  Time: {raw.get('timestamp', '?')}"
        )

    def _op_diary_read(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_diary_read(
            agent_name=args.get("agent_name", "hermes"),
            last_n=self._int_arg(args, "limit", 10),
            wing=args.get("wing", ""),
        )
        raw = raw if isinstance(raw, dict) else {}
        entries = raw.get("entries") or []
        if not entries:
            return raw.get("message") or "No diary entries found."
        blocks = []
        for entry in entries:
            entry = entry if isinstance(entry, dict) else {}
            timestamp = entry.get("timestamp") or entry.get("date") or "?"
            blocks.append(f"[{timestamp}] ({entry.get('topic', 'general')})\n{entry.get('content', '')}")
        return "\n\n".join(blocks)

    # -- Rekal engine operations -------------------------------------------

    def _op_memory_store(self, args: Dict[str, Any]) -> str:
        raw = self._engine.store(
            content=args.get("content", ""),
            memory_type=args.get("memory_type", "fact"),
            project=args.get("project"),
            wing=args.get("wing"),
            room=args.get("room"),
            tags=args.get("tags"),
            importance=args.get("importance", 0.5),
        )
        return _json_output(raw)

    def _op_memory_search(self, args: Dict[str, Any]) -> str:
        raw = self._engine.search(
            query=args.get("query", ""),
            limit=self._int_arg(args, "limit", 10),
            project=args.get("project"),
            memory_type=args.get("memory_type"),
            wing=args.get("wing"),
            room=args.get("room"),
            w_fts=args.get("w_fts"),
            w_vec=args.get("w_vec"),
            w_recency=args.get("w_recency"),
            half_life=args.get("half_life"),
            fusion=args.get("fusion"),
            graph_expand=args.get("graph_expand"),
            temporal=args.get("temporal"),
            strategy_boosts=args.get("strategy_boosts"),
        )
        return _json_output(raw)

    def _op_memory_recall(self, args: Dict[str, Any]) -> str:
        raw = self._engine.search(
            query=args.get("query", ""),
            limit=self._int_arg(args, "limit", 10),
            project=args.get("project"),
            memory_type=args.get("memory_type"),
            wing=args.get("wing"),
            room=args.get("room"),
            fusion=args.get("fusion", "rrf"),
            graph_expand=args.get("graph_expand", True),
            temporal=args.get("temporal", True),
            strategy_boosts=args.get("strategy_boosts"),
        )
        return _format_memory_recall(raw)

    def _op_memory_update(self, args: Dict[str, Any]) -> str:
        raw = self._engine.update(
            memory_id=args.get("memory_id", ""),
            content=args.get("content"),
            tags=args.get("tags"),
            memory_type=args.get("memory_type"),
        )
        return _json_output(raw)

    def _op_memory_supersede(self, args: Dict[str, Any]) -> str:
        raw = self._engine.supersede(
            old_id=args.get("old_id", ""),
            new_content=args.get("content", ""),
            memory_type=args.get("memory_type"),
            project=args.get("project"),
            wing=args.get("wing"),
            room=args.get("room"),
            tags=args.get("tags"),
        )
        return _json_output(raw)

    def _op_memory_delete(self, args: Dict[str, Any]) -> str:
        raw = self._engine.delete(memory_id=args.get("memory_id", ""))
        return _json_output(raw)

    def _op_memory_link(self, args: Dict[str, Any]) -> str:
        raw = self._engine.link(
            from_id=args.get("from_id", ""),
            to_id=args.get("to_id", ""),
            relation=args.get("link_relation", "related_to"),
        )
        return _json_output(raw)

    def _op_build_context(self, args: Dict[str, Any]) -> str:
        raw = self._engine.build_context(
            query=args.get("query", ""),
            project=args.get("project"),
            limit=self._int_arg(args, "limit", 10),
            w_fts=args.get("w_fts"),
            w_vec=args.get("w_vec"),
            w_recency=args.get("w_recency"),
            half_life=args.get("half_life"),
        )
        return _json_output(raw)

    def _op_memory_conflicts(self, args: Dict[str, Any]) -> str:
        raw = self._engine.get_conflicts(project=args.get("project"))
        return _json_output(raw)

    def _op_memory_health(self, args: Dict[str, Any]) -> str:
        return _json_output(self._engine.health())

    def _op_memory_similar(self, args: Dict[str, Any]) -> str:
        raw = self._engine.similar(
            memory_id=args.get("memory_id", ""),
            limit=self._int_arg(args, "limit", 5),
        )
        return _json_output(raw)

    def _op_memory_topics(self, args: Dict[str, Any]) -> str:
        return _json_output(self._engine.topics(project=args.get("project")))

    def _op_memory_timeline(self, args: Dict[str, Any]) -> str:
        raw = self._engine.timeline(
            project=args.get("project"),
            start=args.get("start"),
            end=args.get("end"),
            limit=self._int_arg(args, "limit", 20),
        )
        return _json_output(raw)

    def _op_memory_related(self, args: Dict[str, Any]) -> str:
        return _json_output(self._engine.related(memory_id=args.get("memory_id", "")))

    def _op_contradiction_check(self, args: Dict[str, Any]) -> str:
        raw = self._engine.contradiction_check(
            statement=args.get("statement", args.get("content", "")),
            entity=args.get("entity"),
        )
        contradictions = raw.get("contradictions") or [] if isinstance(raw, dict) else []
        verdict = raw.get("verdict", "unknown") if isinstance(raw, dict) else "unknown"
        lines = [
            f"Statement: {raw.get('statement') or args.get('statement') or '?'}",
            f"Verdict: {verdict}",
            "Entities checked: "
            + (", ".join(raw.get("entities_checked") or []) or "none"),
        ]
        if contradictions:
            lines.append("")
            lines.append("Contradictions:")
            for c in contradictions:
                c = c if isinstance(c, dict) else {}
                lines.append(f"  [{c.get('type')}] confidence={c.get('confidence')} — {c.get('reason')}")
                lines.append(f"    Fact: {c.get('fact')}")
        return "\n".join(lines)

    def _op_fact_check(self, args: Dict[str, Any]) -> str:
        raw = self._engine.fact_check(claim=args.get("claim", args.get("content", "")))
        raw = raw if isinstance(raw, dict) else {}
        verdict = raw.get("verdict", "unknown")
        confidence = raw.get("confidence", 0)
        supporting = raw.get("supporting_evidence") or []
        contradicting = raw.get("contradicting_evidence") or []
        lines = [
            f"Claim: {raw.get('claim') or args.get('claim') or '?'}",
            f"Verdict: {verdict}",
            f"Confidence: {confidence}",
            "Entities: " + (", ".join(raw.get("entities") or []) or "none"),
        ]
        if supporting:
            lines.append("")
            lines.append("Supporting evidence:")
            for s in supporting:
                s = s if isinstance(s, dict) else {}
                if s.get("fact"):
                    lines.append(f"  [{s.get('type')}] {s['fact']}")
                elif s.get("text"):
                    lines.append(f"  [{s.get('type')}] sim={s.get('similarity')} — {s['text']}")
        if contradicting:
            lines.append("")
            lines.append("Contradicting evidence:")
            for c in contradicting:
                c = c if isinstance(c, dict) else {}
                lines.append(f"  [{c.get('type')}] confidence={c.get('confidence')} — {c.get('reason')}")
        return "\n".join(lines)

    def _op_multi_hop(self, args: Dict[str, Any]) -> str:
        raw = self._engine.multi_hop(
            start_entity=args.get("entity", ""),
            target_entity=args.get("target"),
            max_hops=self._int_arg(args, "depth", 3),
        )
        raw = raw if isinstance(raw, dict) else {}
        if raw.get("error"):
            return f"Failed: {raw['error']}"
        paths = raw.get("paths") or []
        lines = [
            f"Start: {raw.get('start') or args.get('entity')}",
        ]
        if raw.get("target"):
            lines.append(f"Target: {raw['target']}")
        lines.append(f"Max hops: {raw.get('max_hops', args.get('depth', 3))}")
        lines.append(f"Nodes explored: {raw.get('graph_explored', '?')}")
        if paths:
            lines.append("")
            lines.append("Paths found:")
            for p in paths:
                p = p if isinstance(p, dict) else {}
                path_arr = p.get("path") or []
                lines.append(f"  [{p.get('hops')} hops] {' '.join(str(x) for x in path_arr)}")
        elif raw.get("target"):
            lines.append("")
            lines.append(
                f"No path found from {raw.get('start')} to {raw['target']} "
                f"within {raw.get('max_hops')} hops."
            )
        reachable = raw.get("reachable") or []
        if reachable:
            lines.append("")
            lines.append(f"Reachable entities ({len(reachable)}): {', '.join(str(x) for x in reachable)}")
        return "\n".join(lines)

    def _op_set_config(self, args: Dict[str, Any]) -> str:
        project = args.get("project")
        if not project:
            raise ValueError("project is required for set_config")
        raw = self._engine.set_config(project, args.get("key", ""), args.get("value", ""))
        return _json_output(raw)

    def _op_session_init(self, args: Dict[str, Any]) -> str:
        raw = self._engine.session_init(
            task=args.get("query") or args.get("task") or "",
            project=args.get("project"),
            limit=self._int_arg(args, "limit", 10),
            w_fts=args.get("w_fts"),
            w_vec=args.get("w_vec"),
            w_recency=args.get("w_recency"),
            half_life=args.get("half_life"),
        )
        return _json_output(raw)

    def _op_ingest_turns(self, args: Dict[str, Any]) -> str:
        turns = args.get("turns", args.get("content", ""))
        if isinstance(turns, list):
            normalized: List[Any] = []
            for item in turns:
                if isinstance(item, dict):
                    normalized.append(item)
                else:
                    normalized.append({"role": "mixed", "content": str(item)})
            turns = normalized
        raw = self._engine.ingest_turns(
            turns=turns,
            project=args.get("project"),
            wing=args.get("wing"),
            room=args.get("room"),
        )
        return _json_output(raw)


def register(ctx):
    ctx.register_memory_provider(RekalMemoryProvider())
