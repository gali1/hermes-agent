"""Rekal memory plugin — Hermes host for the upstream MemPalace/Rekal engine.

The engine (``mempalace_rekal_engine.py``) and the retrieval primitives
(``mempalace_hindsight.py``) are the current upstream implementations from
OpenCode, kept verbatim apart from import/packaging shims.  This module is the
Hermes integration layer that replaces OpenCode's bridge subprocess, TypeScript
tool and session hooks:

* a small set of narrowly scoped tools (``rekal_search``, ``rekal_remember``,
  ``rekal_link``, ``rekal_overview``, ``rekal_ingest``, ``rekal_diary``) instead
  of one 32-operation mega-tool.  Every LLM-facing action is additive or
  read-only; maintenance (delete, deduplicate, FTS rebuild, migrations,
  scoring config) lives behind :meth:`RekalMemoryProvider.internal_op` and is
  never routable from ``handle_tool_call()``.
* automatic recall through ``prefetch()`` using the upstream advanced retrieval
  path (rank fusion + graph expansion + temporal analysis),
* automatic capture and conversation offloading through ``sync_turn()``,
  ``on_session_end()`` and ``on_pre_compress()``,
* the upstream memory protocol as tool guidance, plus the ``<memory_system>``
  trust block in the system prompt.

The diary is backed by the Rekal SQLite store (``rekal_diary`` table), so it
works with zero optional dependencies.  MemPalace's own drawer / KG operations
remain available but are internal: when the ``mempalace`` pip package is not
installed they return a clean install hint and the zero-dependency Rekal engine
keeps working.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# MemPalace optional dependency (drawer / knowledge-graph operations)
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _json_output(raw: Any) -> str:
    return json.dumps(raw, indent=2, default=str)


def _hit_payload(item: Dict[str, Any], max_content: int = 2000) -> Dict[str, Any]:
    """Result row for the LLM surface: keep every scoring signal, drop empties."""
    payload: Dict[str, Any] = {}
    for key, value in item.items():
        if value in (None, "", [], {}):
            continue
        if key == "content" and isinstance(value, str):
            value = value[:max_content]
        payload[key] = value
    return payload


# ---------------------------------------------------------------------------
# Store-backed diary (zero optional dependencies)
# ---------------------------------------------------------------------------
#
# Upstream kept the journal inside the optional mempalace package, so diary
# write/read were dead without it.  A diary entry is a journal line, not a
# recall candidate: storing it in ``memories`` would pollute search results and
# feed narrative text into contradiction detection.  It therefore gets its own
# table in the same SQLite store, created idempotently on first use.

_DIARY_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS rekal_diary (
        id TEXT PRIMARY KEY,
        agent_name TEXT NOT NULL,
        wing TEXT,
        topic TEXT,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_rekal_diary_agent
        ON rekal_diary(agent_name, created_at DESC)""",
)

_DIARY_COLUMNS = "id, agent_name, wing, topic, content, created_at"


class _StoreDiary:
    """Diary journal backed by the Rekal SQLite store."""

    def __init__(self, engine: RekalEngine):
        self._engine = engine
        self._ready = False

    # The engine owns the connection, its busy-retry helper and the commit
    # protocol; the diary reuses them rather than opening a second writer.
    def _execute(self, sql: str, params: Tuple[Any, ...] = ()) -> Any:
        execute = getattr(self._engine, "_execute", None)
        if callable(execute):
            return execute(sql, params)
        return self._engine.db.execute(sql, params)

    def _commit(self) -> None:
        self._engine.db.commit()

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        for statement in _DIARY_SCHEMA:
            self._execute(statement)
        self._commit()
        self._ready = True

    def write(self, agent_name: str, entry: str, topic: str = "general", wing: str = "") -> Dict[str, Any]:
        self._ensure_schema()
        entry_id = uuid.uuid4().hex[:16]
        timestamp = _utc_now_iso()
        self._execute(
            f"INSERT INTO rekal_diary ({_DIARY_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
            (entry_id, agent_name, wing, topic, entry, timestamp),
        )
        self._commit()
        return {
            "success": True,
            "entry_id": entry_id,
            "agent": agent_name,
            "topic": topic,
            "timestamp": timestamp,
        }

    def read(self, agent_name: str = "hermes", last_n: int = 10, wing: str = "") -> List[Dict[str, Any]]:
        self._ensure_schema()
        rows = self._execute(
            f"""SELECT {_DIARY_COLUMNS} FROM rekal_diary
                WHERE (? = '' OR agent_name = ?)
                  AND (? = '' OR wing = ?)
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?""",
            (agent_name, agent_name, wing, wing, last_n),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "agent": row["agent_name"],
                "wing": row["wing"],
                "topic": row["topic"],
                "content": row["content"],
                "timestamp": row["created_at"],
            }
            for row in rows
        ]


# ---------------------------------------------------------------------------
# LLM-facing tool surface
# ---------------------------------------------------------------------------
#
# Shape: several narrowly scoped tools, the same convention the other external
# memory providers use (``retaindb_*``, ``brv_*``), with an ``action`` enum only
# where a tool genuinely covers more than one verb.  Schemas are built from the
# dispatch tables below, so ``get_tool_schemas()`` cannot advertise an action
# that ``handle_tool_call()`` would refuse.
#
# INVARIANT: nothing in _TOOL_SPECS may be destructive or administrative.
# delete / deduplicate / FTS rebuild / migrations / scoring config are in
# _INTERNAL_OPERATIONS and are only reachable from internal_op() (CLI/tests).

_MEMORY_TYPES = ["fact", "preference", "procedure", "context", "episode"]
_LINK_RELATIONS = ["supersedes", "contradicts", "related_to"]

_PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    "query": {"type": "string", "description": "What to look for."},
    "memory_id": {"type": "string", "description": "Memory ID (from search/store output)."},
    "limit": {"type": "number", "description": "Max results (default: 10, max: 50)."},
    "project": {"type": "string", "description": "Optional project scope."},
    "memory_type": {"type": "string", "enum": list(_MEMORY_TYPES), "description": "Filter or classify by type."},
    "fusion": {"type": "string", "enum": ["rrf"], "description": "'rrf' combines retrieval arms by rank."},
    "graph_expand": {"type": "boolean", "description": "Also return memories linked to the top hits."},
    "temporal": {"type": "boolean", "description": "Read a time window out of the query and prefer that period."},
    "strategy_boosts": {"type": "object", "description": "Per-strategy weight overrides."},
    "content": {"type": "string", "description": "The fact/procedure/episode text to store (one self-contained statement)."},
    "tags": {"type": "array", "items": {"type": "string"}, "description": "2-4 specific tags."},
    "importance": {"type": "number", "description": "Importance 0-1 (default: 0.5)."},
    "wing": {"type": "string", "description": "Storage wing (default: project)."},
    "room": {"type": "string", "description": "Storage room (default: general)."},
    "from_id": {"type": "string", "description": "Source memory ID."},
    "to_id": {"type": "string", "description": "Target memory ID."},
    "link_relation": {"type": "string", "enum": list(_LINK_RELATIONS), "description": "Relation to record (default: related_to)."},
    "start": {"type": "string", "description": "ISO timestamp lower bound."},
    "end": {"type": "string", "description": "ISO timestamp upper bound."},
    "statement": {"type": "string", "description": "Optional claim to check against memory + knowledge graph."},
    "turns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Conversation turns to offload (strings, or objects with role/content).",
    },
    "entry": {"type": "string", "description": "Diary entry text."},
    "topic": {"type": "string", "description": "Diary topic (default: general)."},
    "agent_name": {"type": "string", "description": "Diary owner (default: hermes)."},
}

_TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "rekal_search": {
        "description": (
            "Search Rekal long-term memory. ACTIONS:\n"
            "• search — weighted keyword/vector/recency scoring; use when you want the plain ranking.\n"
            "• recall — PREFERRED. Rank fusion across keyword/vector/graph/time, associative expansion through "
            "links, and natural-language time windows ('yesterday', 'last week', 'March 2026'). Time words are "
            "removed from the keyword match, so they never suppress results.\n"
            "• get — fetch one memory by ID together with the memories it is linked to.\n\n"
            "Search before storing, and recall stored context before answering questions about past work."
        ),
        "properties": (
            "query", "memory_id", "limit", "project", "memory_type",
            "fusion", "graph_expand", "temporal", "strategy_boosts",
        ),
        "actions": {"search": "_act_search", "recall": "_act_recall", "get": "_act_get"},
        "required": ("action",),
    },
    "rekal_remember": {
        "description": (
            "Store a durable memory, or reinforce one you observed again. ACTIONS:\n"
            "• store — one self-contained fact/preference/procedure/context/episode. Distil: no raw dialogue, "
            "1-2 sentences, 2-4 specific tags. Re-storing identical content does not duplicate it; it raises that "
            "memory's proof_count (repetition is evidence). Storing content that negates an existing memory records "
            "a 'contradicts' link and returns it under `contradicts` — nothing is deleted.\n"
            "• reinforce — record another independent observation of an existing memory by ID (raises proof_count).\n\n"
            "Never store secrets, API keys or passwords."
        ),
        "properties": (
            "content", "memory_id", "memory_type", "tags", "importance", "project", "wing", "room",
        ),
        "actions": {"store": "_act_store", "reinforce": "_act_reinforce"},
        "required": ("action",),
    },
    "rekal_link": {
        "description": (
            "Connect two memories. Relations: related_to (association), contradicts (disagreement to review), "
            "supersedes (from_id replaces to_id while to_id is kept). Pick supersedes rather than deleting an "
            "outdated memory."
        ),
        "properties": ("from_id", "to_id", "link_relation"),
        "handler": "_act_link",
        "required": ("from_id", "to_id"),
    },
    "rekal_overview": {
        "description": (
            "Introspect the memory store. ACTIONS:\n"
            "• timeline — recent memories in time order (optionally bounded by start/end).\n"
            "• topics — memory-type distribution with counts.\n"
            "• contradictions — contradicting memory pairs, or (when `statement` is given) a verdict on whether a "
            "claim conflicts with what is stored. Check before storing a fact that might disagree with memory.\n"
            "• stats — store statistics: totals, links, conflicts, duplicate groups, per-type/per-project counts."
        ),
        "properties": ("project", "start", "end", "limit", "statement"),
        "actions": {
            "timeline": "_act_timeline",
            "topics": "_act_topics",
            "contradictions": "_act_contradictions",
            "stats": "_act_stats",
        },
        "required": ("action",),
    },
    "rekal_ingest": {
        "description": (
            "Offload conversation turns into long-term memory. Each turn is distilled into structured memories; "
            "purely mechanical turns are skipped. Call when the conversation grows long (~15+ turns), when switching "
            "tasks, or before a long multi-step operation that needs prompt space."
        ),
        "properties": ("turns", "content", "project", "wing", "room"),
        "handler": "_act_ingest",
        "required": (),
    },
    "rekal_diary": {
        "description": (
            "Session journal for continuity across sessions. ACTIONS:\n"
            "• write — log what was accomplished, decisions taken, and what comes next.\n"
            "• read — recent entries, newest first."
        ),
        "properties": ("entry", "topic", "limit", "wing", "agent_name"),
        "actions": {"write": "_act_diary_write", "read": "_act_diary_read"},
        "required": ("action",),
    },
}


def _build_tool_schema(tool_name: str, spec: Dict[str, Any], param_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    actions = spec.get("actions") or {}
    if actions:
        properties["action"] = {
            "type": "string",
            "enum": list(actions),
            "description": "Which action to perform.",
        }
    for name in spec["properties"]:
        properties[name] = copy.deepcopy(param_specs[name])
    required: List[str] = list(spec.get("required") or ())
    if actions and "action" not in required:
        required.insert(0, "action")
    return {
        "name": tool_name,
        "description": spec["description"],
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


TOOL_SCHEMAS: Tuple[Dict[str, Any], ...] = tuple(
    _build_tool_schema(name, spec, _PARAM_SPECS) for name, spec in _TOOL_SPECS.items()
)

# The complete LLM-facing action set, for tests and for the CLI to diff against.
# Actionless tools appear as (tool, None).
LLM_ACTIONS: Tuple[Tuple[str, Optional[str]], ...] = tuple(
    (tool, action)
    for tool, spec in _TOOL_SPECS.items()
    for action in ((tuple(spec["actions"]) if spec.get("actions") else (None,)))
)


# ---------------------------------------------------------------------------
# Internal / CLI-only operations (never reachable from handle_tool_call)
# ---------------------------------------------------------------------------
#
# Everything the old mega-tool exposed, including maintenance. These are
# destructive or administrative and must not be model-callable: deleting
# memories, hard-deleting duplicate rows, rebuilding FTS, running migrations and
# rewriting per-project scoring weights all belong to an operator.

_INTERNAL_OPERATIONS: Dict[str, str] = {
    # Retrieval / introspection (same capability as the LLM surface, kept for CLI parity)
    "build_context": "_op_build_context",
    "contradiction_check": "_op_contradiction_check",
    "fact_check": "_op_fact_check",
    "ingest_turns": "_op_ingest_turns",
    "memory_conflicts": "_op_memory_conflicts",
    "memory_health": "_op_memory_health",
    "memory_link": "_op_memory_link",
    "memory_recall": "_op_memory_recall",
    "memory_related": "_op_memory_related",
    "memory_search": "_op_memory_search",
    "memory_similar": "_op_memory_similar",
    "memory_store": "_op_memory_store",
    "memory_timeline": "_op_memory_timeline",
    "memory_topics": "_op_memory_topics",
    "multi_hop": "_op_multi_hop",
    "session_init": "_op_session_init",
    # Journal (store-backed since C1)
    "diary_write": "_op_diary_write",
    "diary_read": "_op_diary_read",
    # MemPalace-native drawers / knowledge graph (optional dependency)
    "search": "_op_search",
    "smart_search": "_op_smart_search",
    "store": "_op_store",
    "status": "_op_status",
    "list_wings": "_op_list_wings",
    "list_rooms": "_op_list_rooms",
    "kg_add": "_op_kg_add",
    "kg_query": "_op_kg_query",
    "kg_invalidate": "_op_kg_invalidate",
    # Mutation / maintenance — operator only
    "memory_update": "_op_memory_update",
    "memory_supersede": "_op_memory_supersede",
    "memory_delete": "_op_memory_delete",
    "deduplicate": "_op_deduplicate",
    "consolidate": "_op_deduplicate",   # no non-destructive consolidate exists yet; dry-run unless apply=true
    "rebuild_fts": "_op_rebuild_fts",
    "migrate": "_op_migrate",
    "set_config": "_op_set_config",
}

_SYSTEM_PROMPT_BLOCK = """<memory_system>
You have access to Rekal — a persistent, local memory system that survives across sessions.
Relevant memories from previous sessions may be automatically injected into your context inside <mempalace_context> tags.

Rules for using retrieved memories:
- NEVER blindly trust retrieved memory. Always validate against current file state and session context.
- If a memory contradicts current code or files, trust the code — the memory may be stale.
- Use rekal_search / rekal_remember / rekal_link / rekal_overview / rekal_ingest / rekal_diary when you need to store important decisions, search for specific history, or connect related knowledge.
- Significant discoveries, architecture decisions, and resolved bugs are automatically stored after each response.
</memory_system>"""


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class RekalMemoryProvider(MemoryProvider):
    def __init__(self):
        self._config = _default_config()
        self._engine: Optional[RekalEngine] = None
        self._diary: Optional[_StoreDiary] = None
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
        self._diary = _StoreDiary(self._engine)

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
        self._diary = None
        self._active = False

    # -- Tool surface ------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [dict(schema) for schema in TOOL_SCHEMAS]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        spec = _TOOL_SPECS.get(tool_name)
        if spec is None:
            known = ", ".join(_TOOL_SPECS)
            return tool_error(f"Unknown tool: {tool_name}. Rekal tools: {known}")
        if not self._active or not self._engine:
            return tool_error("Rekal is not initialized")

        args = args or {}
        actions = spec.get("actions") or {}
        if actions:
            action = str(args.get("action") or "").strip()
            if action not in actions:
                valid = ", ".join(actions)
                return tool_error(
                    f"Unknown action for {tool_name}: {action or '<missing>'}. Valid actions: {valid}"
                )
            handler_name = actions[action]
        else:
            action = tool_name
            handler_name = spec["handler"]

        # Dispatch ONLY through the spec table above: never getattr on a
        # caller-supplied operation name, or internal maintenance would be
        # reachable again.
        handler = getattr(self, handler_name, None)
        if handler is None:  # schema/handler drift must fail closed, not silently
            logger.error("rekal: %s advertises action '%s' with no handler %s", tool_name, action, handler_name)
            return tool_error(f"{tool_name} action '{action}' is unavailable in this build")

        try:
            return handler(args)
        except ValueError as exc:
            return tool_error(str(exc))
        except Exception as exc:
            logger.debug("rekal %s/%s failed", tool_name, action, exc_info=True)
            return tool_error(f"Rekal {tool_name} '{action}' failed: {exc}")

    def internal_op(self, operation: str, **args: Any) -> str:
        """Run a maintenance/administrative operation (CLI + tests only).

        Deliberately unreachable from ``handle_tool_call``: destructive and
        administrative operations must never be model-callable.
        """
        handler_name = _INTERNAL_OPERATIONS.get(operation)
        if handler_name is None:
            valid = ", ".join(sorted(_INTERNAL_OPERATIONS))
            return tool_error(f"Unknown internal operation: {operation}. Valid: {valid}")
        if not self._active or not self._engine:
            return tool_error("Rekal is not initialized")
        handler = getattr(self, handler_name, None)
        if handler is None:  # pragma: no cover - table is static
            return tool_error(f"Internal operation '{operation}' has no handler")
        try:
            return handler(args)
        except ValueError as exc:
            return tool_error(str(exc))
        except Exception as exc:
            logger.debug("rekal internal op %s failed", operation, exc_info=True)
            return tool_error(f"Internal operation '{operation}' failed: {exc}")

    @staticmethod
    def internal_operations() -> List[str]:
        return sorted(_INTERNAL_OPERATIONS)

    # -- Argument helpers --------------------------------------------------

    @staticmethod
    def _int_arg(args: Dict[str, Any], key: str, default: int) -> int:
        try:
            return int(args.get(key, default))
        except (TypeError, ValueError):
            return default

    def _limit_arg(self, args: Dict[str, Any], default: int = 10) -> int:
        return max(1, min(50, self._int_arg(args, "limit", default)))

    @staticmethod
    def _require_arg(args: Dict[str, Any], key: str) -> str:
        value = str(args.get(key) or "").strip()
        if not value:
            raise ValueError(f"{key} is required")
        return value

    # -- LLM actions: search / remember / link / overview / ingest / diary --

    def _search_envelope(self, args: Dict[str, Any], *, advanced: bool) -> Dict[str, Any]:
        query = self._require_arg(args, "query")
        kwargs: Dict[str, Any] = {
            "limit": self._limit_arg(args, 10),
            "project": args.get("project"),
            "memory_type": args.get("memory_type"),
            "strategy_boosts": args.get("strategy_boosts"),
        }
        if advanced:
            kwargs["fusion"] = args.get("fusion") or "rrf"
            kwargs["graph_expand"] = _as_bool(args.get("graph_expand"), True)
            kwargs["temporal"] = _as_bool(args.get("temporal"), True)
        else:
            kwargs["fusion"] = args.get("fusion")
            kwargs["graph_expand"] = args.get("graph_expand")
            kwargs["temporal"] = args.get("temporal")

        envelope = self._engine.search(query, **kwargs)
        results = envelope.get("results", []) if isinstance(envelope, dict) else []
        payload: Dict[str, Any] = {
            "results": [_hit_payload(hit) for hit in results],
            "count": len(results),
        }
        if isinstance(envelope, dict) and envelope.get("retrieval"):
            payload["retrieval"] = envelope["retrieval"]
        return payload

    def _act_search(self, args: Dict[str, Any]) -> str:
        return _json_output(self._search_envelope(args, advanced=False))

    def _act_recall(self, args: Dict[str, Any]) -> str:
        return _json_output(self._search_envelope(args, advanced=True))

    def _act_get(self, args: Dict[str, Any]) -> str:
        memory_id = self._require_arg(args, "memory_id")
        memory = self._engine.get(memory_id)
        if not memory:
            return tool_error(f"Memory not found: {memory_id}")
        payload = _hit_payload(memory)
        try:
            payload["links"] = self._engine.related(memory_id)
        except Exception:
            logger.debug("rekal get: related() failed for %s", memory_id, exc_info=True)
        return _json_output(payload)

    def _act_store(self, args: Dict[str, Any]) -> str:
        content = self._require_arg(args, "content")
        memory_type = str(args.get("memory_type") or "fact").strip()
        if memory_type not in _MEMORY_TYPES:
            raise ValueError(f"memory_type must be one of: {', '.join(_MEMORY_TYPES)}")
        tags = args.get("tags")
        if tags is not None and not isinstance(tags, list):
            tags = [str(tags)]
        try:
            importance = float(args.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        raw = self._engine.store(
            content=content,
            memory_type=memory_type,
            project=args.get("project"),
            wing=args.get("wing"),
            room=args.get("room"),
            tags=tags,
            importance=max(0.0, min(1.0, importance)),
        )
        return _json_output(raw)

    def _act_reinforce(self, args: Dict[str, Any]) -> str:
        memory_id = self._require_arg(args, "memory_id")
        reinforce = getattr(self._engine, "reinforce", None) or getattr(self._engine, "_reinforce", None)
        if not callable(reinforce):
            return tool_error("This Rekal store does not support reinforcement")
        proof_count = reinforce(memory_id)
        if proof_count is None:
            return tool_error(f"Memory not found (or store too old to reinforce): {memory_id}")
        return _json_output({
            "success": True,
            "memory_id": memory_id,
            "proof_count": proof_count,
        })

    def _act_link(self, args: Dict[str, Any]) -> str:
        from_id = self._require_arg(args, "from_id")
        to_id = self._require_arg(args, "to_id")
        relation = str(args.get("link_relation") or "related_to").strip()
        if relation not in _LINK_RELATIONS:
            raise ValueError(f"link_relation must be one of: {', '.join(_LINK_RELATIONS)}")
        raw = self._engine.link(from_id=from_id, to_id=to_id, relation=relation)
        if isinstance(raw, dict) and raw.get("error"):
            return tool_error(str(raw["error"]))
        return _json_output(raw)

    def _act_timeline(self, args: Dict[str, Any]) -> str:
        rows = self._engine.timeline(
            project=args.get("project"),
            start=args.get("start"),
            end=args.get("end"),
            limit=self._limit_arg(args, 20),
        )
        return _json_output({
            "results": [_hit_payload(row, max_content=400) for row in rows],
            "count": len(rows),
        })

    def _act_topics(self, args: Dict[str, Any]) -> str:
        topics = self._engine.topics(project=args.get("project"))
        return _json_output({"topics": topics, "count": len(topics)})

    def _act_contradictions(self, args: Dict[str, Any]) -> str:
        statement = str(args.get("statement") or "").strip()
        if statement:
            return _json_output(self._engine.contradiction_check(statement))
        conflicts = self._engine.get_conflicts(project=args.get("project"))
        return _json_output({"conflicts": conflicts, "count": len(conflicts)})

    def _act_stats(self, args: Dict[str, Any]) -> str:
        return _json_output(self._engine.health())

    def _act_ingest(self, args: Dict[str, Any]) -> str:
        turns = args.get("turns")
        if turns is None:
            turns = args.get("content", "")
        if isinstance(turns, list):
            normalized: List[Any] = []
            for item in turns:
                if isinstance(item, dict):
                    normalized.append(item)
                else:
                    normalized.append({"role": "mixed", "content": str(item)})
            turns = normalized
        if not turns:
            raise ValueError("turns is required")
        raw = self._engine.ingest_turns(
            turns=turns,
            project=args.get("project"),
            wing=args.get("wing"),
            room=args.get("room"),
        )
        return _json_output(raw)

    def _diary_store(self) -> _StoreDiary:
        if self._diary is None:
            if self._engine is None:
                raise RuntimeError("Rekal diary is not initialized")
            self._diary = _StoreDiary(self._engine)
        return self._diary

    def _act_diary_write(self, args: Dict[str, Any]) -> str:
        entry = self._require_arg(args, "entry")
        raw = self._diary_store().write(
            agent_name=str(args.get("agent_name") or "hermes"),
            entry=entry,
            topic=str(args.get("topic") or "general"),
            wing=str(args.get("wing") or ""),
        )
        return _json_output(raw)

    def _act_diary_read(self, args: Dict[str, Any]) -> str:
        entries = self._diary_store().read(
            agent_name=str(args.get("agent_name") or "hermes"),
            last_n=self._limit_arg(args, 10),
            wing=str(args.get("wing") or ""),
        )
        return _json_output({"entries": entries, "count": len(entries)})

    # -- Result formatting for MemPalace-native operations ------------------

    @staticmethod
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

    # -- Internal: MemPalace native operations ------------------------------

    def _op_search(self, args: Dict[str, Any]) -> str:
        _require_mempalace()
        raw = _mcp_mod.tool_search(
            query=args.get("query", ""),
            limit=self._int_arg(args, "limit", 5),
            wing=args.get("wing") or None,
            room=args.get("room") or None,
        )
        return self._format_search_results(raw)

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
        return self._format_search_results(raw, adaptive=True)

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
        """Journal write — store-backed, so mempalace is not required.

        Kept under the legacy operation name so CLI callers keep working; the
        result carries the same fields the bridge used (entry_id/agent/timestamp).
        """
        return self._act_diary_write(args)

    def _op_diary_read(self, args: Dict[str, Any]) -> str:
        """Journal read — store-backed, so mempalace is not required."""
        return self._act_diary_read(args)

    # -- Internal: Rekal engine operations ----------------------------------

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
        return _json_output(raw)

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
        raw = self._engine.delete(memory_id=self._require_arg(args, "memory_id"))
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
        return _json_output(raw)

    def _op_fact_check(self, args: Dict[str, Any]) -> str:
        return _json_output(self._engine.fact_check(claim=args.get("claim", args.get("content", ""))))

    def _op_multi_hop(self, args: Dict[str, Any]) -> str:
        raw = self._engine.multi_hop(
            start_entity=args.get("entity", ""),
            target_entity=args.get("target"),
            max_hops=self._int_arg(args, "depth", 3),
        )
        return _json_output(raw)

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

    # -- Internal: maintenance (destructive/administrative) -----------------

    def _op_deduplicate(self, args: Dict[str, Any]) -> str:
        """Hard-deletes duplicate rows. Dry-run unless ``apply`` is explicitly true."""
        apply = _as_bool(args.get("apply"), False)
        raw = self._engine.deduplicate(project=args.get("project"), dry_run=not apply)
        return _json_output(raw)

    def _op_rebuild_fts(self, args: Dict[str, Any]) -> str:
        try:
            self._engine._execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self._engine.db.commit()
            cache = getattr(self._engine, "_cache", None)
            if cache is not None and hasattr(cache, "clear"):
                cache.clear()
        except Exception as exc:
            return _json_output({"success": False, "error": str(exc)})
        return _json_output({"success": True, "rebuilt": True})

    def _op_migrate(self, args: Dict[str, Any]) -> str:
        self._engine._run_migrations()
        self._engine._repair_fts_if_needed()
        return _json_output({"success": True, "migrated": True})


def register(ctx):
    ctx.register_memory_provider(RekalMemoryProvider())
