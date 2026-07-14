"""Rekal local memory plugin using the MemoryProvider interface.

Provides persistent local memory via SQLite + FTS5 with hybrid search,
conflict detection, memory lifecycle management, and session ingest.
Zero external dependencies — uses only Python stdlib.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from plugins.memory.rekal.rekal_engine import RekalEngine
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 10
_DEFAULT_CAPTURE_MODE = "all"
_MIN_CAPTURE_LENGTH = 20

_TRIVIAL_RE = re.compile(
    r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np|done|perfect|great|nice)\.?$",
    re.IGNORECASE,
)

_CONTEXT_STRIP_RE = re.compile(
    r"<rekal-context>[\s\S]*?</rekal-context>\s*", re.DOTALL
)


def _default_config() -> dict:
    return {
        "data_dir": "rekal",
        "auto_recall": True,
        "auto_capture": True,
        "max_results": _DEFAULT_MAX_RESULTS,
        "capture_mode": _DEFAULT_CAPTURE_MODE,
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


def _format_relative_time(iso_timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
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
        updated = item.get("updated_at", "")
        rel = _format_relative_time(updated)
        prefix_bits = []
        if rel:
            prefix_bits.append(f"[{rel}]")
        importance = item.get("importance", 0)
        if importance >= 7:
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
    return f"<rekal-context>\n{intro}\n\n{body}\n</rekal-context>"


def _clean_text_for_capture(text: str) -> str:
    text = _CONTEXT_STRIP_RE.sub("", text or "")
    return text.strip()


def _is_trivial_message(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


def _detect_category(text: str) -> str:
    lowered = text.lower()
    if re.search(r"prefer|like|love|hate|want|favorite", lowered):
        return "preference"
    if re.search(r"decided|will use|going with|choos|select", lowered):
        return "decision"
    if re.search(r"\bis\b|\bare\b|\bhas\b|\bhave\b|\bwas\b|\bwere\b", lowered):
        return "fact"
    return "other"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

STORE_SCHEMA = {
    "name": "rekal_store",
    "description": "Store an explicit memory for future recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "memory_type": {"type": "string", "description": "Type: fact, preference, decision, context, episode, procedure.", "enum": ["fact", "preference", "decision", "context", "episode", "procedure"]},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization."},
            "importance": {"type": "number", "description": "Importance 1-10. Higher = more weight in search.", "minimum": 1, "maximum": 10},
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA = {
    "name": "rekal_search",
    "description": "Search long-term memory by semantic similarity (FTS5 BM25 + recency + importance).",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Maximum results to return, 1 to 50."},
            "memory_type": {"type": "string", "description": "Optional filter by memory type.", "enum": ["fact", "preference", "decision", "context", "episode", "procedure"]},
        },
        "required": ["query"],
    },
}

UPDATE_SCHEMA = {
    "name": "rekal_update",
    "description": "Update an existing memory's content, type, tags, or importance.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ID of the memory to update."},
            "content": {"type": "string", "description": "New content."},
            "memory_type": {"type": "string", "description": "New type.", "enum": ["fact", "preference", "decision", "context", "episode", "procedure"]},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags."},
            "importance": {"type": "number", "description": "New importance 1-10.", "minimum": 1, "maximum": 10},
        },
        "required": ["memory_id"],
    },
}

FORGET_SCHEMA = {
    "name": "rekal_forget",
    "description": "Delete a memory by its ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ID of the memory to delete."},
        },
        "required": ["memory_id"],
    },
}

LINK_SCHEMA = {
    "name": "rekal_link",
    "description": "Link two memories with a relationship (related_to, contradicts, supports, references).",
    "parameters": {
        "type": "object",
        "properties": {
            "source_id": {"type": "string", "description": "Source memory ID."},
            "target_id": {"type": "string", "description": "Target memory ID."},
            "relation": {"type": "string", "description": "Relationship type.", "enum": ["related_to", "contradicts", "supports", "references"]},
        },
        "required": ["source_id", "target_id"],
    },
}

CONFLICTS_SCHEMA = {
    "name": "rekal_conflicts",
    "description": "Find memories that contradict each other (linked with contradicts relation).",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Optional. If set, show conflicts for this specific memory."},
            "limit": {"type": "integer", "description": "Maximum results, 1 to 50."},
        },
    },
}

SIMILAR_SCHEMA = {
    "name": "rekal_similar",
    "description": "Find memories with similar content to a given memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "ID of the memory to find similar ones for."},
            "limit": {"type": "integer", "description": "Maximum results, 1 to 20."},
        },
    },
}

TOPICS_SCHEMA = {
    "name": "rekal_topics",
    "description": "Get tag-based topic clusters with memory counts.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum topics, 1 to 50."},
        },
    },
}

TIMELINE_SCHEMA = {
    "name": "rekal_timeline",
    "description": "Get memories sorted by creation date.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Maximum results, 1 to 50."},
            "memory_type": {"type": "string", "description": "Optional filter by type."},
        },
    },
}

HEALTH_SCHEMA = {
    "name": "rekal_health",
    "description": "Get database statistics: memory count, link count, session count, storage size.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

ALL_SCHEMAS = [
    STORE_SCHEMA, SEARCH_SCHEMA, UPDATE_SCHEMA, FORGET_SCHEMA, LINK_SCHEMA,
    CONFLICTS_SCHEMA, SIMILAR_SCHEMA, TOPICS_SCHEMA, TIMELINE_SCHEMA, HEALTH_SCHEMA,
]


def _with_kebab_aliases(schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aliases = {
        "rekal_store": "rekal-save",
        "rekal_search": "rekal-search",
        "rekal_forget": "rekal-forget",
        "rekal_update": "rekal-update",
        "rekal_link": "rekal-link",
        "rekal_conflicts": "rekal-conflicts",
        "rekal_similar": "rekal-similar",
        "rekal_topics": "rekal-topics",
        "rekal_timeline": "rekal-timeline",
        "rekal_health": "rekal-health",
    }
    expanded = list(schemas)
    for schema in schemas:
        kebab = aliases.get(schema.get("name", ""))
        if not kebab:
            continue
        copy = json.loads(json.dumps(schema))
        copy["name"] = kebab
        expanded.append(copy)
    return expanded


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
        self._capture_mode = _DEFAULT_CAPTURE_MODE
        self._write_enabled = True
        self._session_turns: List[Dict[str, str]] = []
        self._prefetch_lock = threading.Lock()
        self._last_prefetch = ""

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
        return {"summary": f"{status} · {h.get('active_memories', 0)} memories · {h.get('total_links', 0)} links · {h.get('total_sessions', 0)} sessions"}

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from hermes_cli.config import save_config

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = self.name
        save_config(config)
        print(f"\n  Memory provider: rekal")
        print("  Activation saved to config.yaml")
        print("  Start a new session to activate.\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._session_id = session_id
        self._turn_count = 0
        self._config = _load_rekal_config(self._hermes_home)

        data_dir = str(Path(self._hermes_home) / str(self._config.get("data_dir", "rekal")))
        self._engine = RekalEngine(data_dir)

        self._auto_recall = self._config["auto_recall"]
        self._auto_capture = self._config["auto_capture"]
        self._max_results = self._config["max_results"]

        self._session_turns = []

        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = agent_context not in {"cron", "flush", "subagent"}
        self._active = True

        self._engine.session_init(session_id)

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        return (
            "# Rekal Memory\n"
            "Active. Local SQLite memory with FTS5 search.\n"
            "Available tools: rekal-store, rekal-search, rekal-update, rekal-forget, "
            "rekal-link, rekal-conflicts, rekal-similar, rekal-topics, rekal-timeline, rekal-health "
            "(aliases: rekal_save, rekal_search, rekal_update, rekal_forget, rekal_link, "
            "rekal_conflicts, rekal_similar, rekal_topics, rekal_timeline, rekal_health)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall or not self._engine or not query.strip():
            return ""
        try:
            results = self._engine.search(query, limit=self._max_results)
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

        self._session_turns.append({"user": clean_user, "assistant": clean_assistant})

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._active or not self._write_enabled or not self._engine or not self._session_id:
            return
        if not self._session_turns:
            return
        try:
            self._engine.ingest_turns(
                self._session_id,
                self._session_turns,
                memory_type="context",
            )
        except Exception:
            logger.debug("Rekal session-end ingest failed", exc_info=True)
        self._session_turns = []

    def on_session_switch(self, new_session_id: str, *,
                          parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        if not self._active or not self._write_enabled or not self._engine:
            self._session_id = str(new_session_id or "").strip() or self._session_id
            self._session_turns = []
            return

        old_session_id = self._session_id
        old_turns = list(self._session_turns)

        if old_turns and old_session_id:
            try:
                self._engine.ingest_turns(
                    old_session_id,
                    old_turns,
                    memory_type="context",
                )
            except Exception:
                logger.debug("Rekal session-switch ingest failed", exc_info=True)

        self._session_id = str(new_session_id or "").strip() or old_session_id
        self._session_turns = []
        self._turn_count = 0

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
                metadata={"target": target, "source": "memory_tool", **(metadata or {})},
            )
        except Exception:
            logger.debug("Rekal on_memory_write failed", exc_info=True)

    def backup_paths(self) -> List[str]:
        if self._engine:
            return [self._engine._db_path]
        return []

    def shutdown(self) -> None:
        if self._active and self._write_enabled and self._engine and self._session_turns and self._session_id:
            logger.info("Rekal: Saving session via shutdown (session=%s, turns=%d)", self._session_id, len(self._session_turns))
            try:
                self._engine.ingest_turns(
                    self._session_id,
                    self._session_turns,
                    memory_type="context",
                )
            except Exception:
                logger.debug("Rekal shutdown ingest failed", exc_info=True)
        if self._engine:
            self._engine.checkpoint()
            self._engine.close()
        self._session_turns = []

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return _with_kebab_aliases(ALL_SCHEMAS)

    # -- Tool handlers -----------------------------------------------------

    def _tool_store(self, args: dict) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        memory_type = str(args.get("memory_type") or "fact").strip()
        tags = args.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        importance = max(1, min(10, int(args.get("importance", 5) or 5)))
        try:
            result = self._engine.store(
                content,
                memory_type=memory_type,
                tags=tags,
                importance=importance,
            )
            preview = content[:80] + ("..." if len(content) > 80 else "")
            return json.dumps({"saved": True, "id": result["id"], "preview": preview})
        except Exception as exc:
            return tool_error(f"Failed to store memory: {exc}")

    def _tool_search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        try:
            limit = max(1, min(50, int(args.get("limit", _DEFAULT_MAX_RESULTS) or _DEFAULT_MAX_RESULTS)))
        except Exception:
            limit = _DEFAULT_MAX_RESULTS
        memory_type = str(args.get("memory_type") or "").strip() or None
        try:
            results = self._engine.search(query, limit=limit, memory_type=memory_type)
            formatted = []
            for item in results:
                entry: Dict[str, Any] = {
                    "id": item.get("id", ""),
                    "content": item.get("content", ""),
                    "memory_type": item.get("memory_type", ""),
                    "importance": item.get("importance", 0),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                }
                tags = item.get("tags", [])
                if tags:
                    entry["tags"] = tags
                formatted.append(entry)
            return json.dumps({"results": formatted, "count": len(formatted)})
        except Exception as exc:
            return tool_error(f"Search failed: {exc}")

    def _tool_update(self, args: dict) -> str:
        memory_id = str(args.get("memory_id") or "").strip()
        if not memory_id:
            return tool_error("memory_id is required")
        try:
            kwargs: Dict[str, Any] = {}
            if "content" in args:
                kwargs["content"] = str(args["content"])
            if "memory_type" in args:
                kwargs["memory_type"] = str(args["memory_type"])
            if "tags" in args and isinstance(args["tags"], list):
                kwargs["tags"] = args["tags"]
            if "importance" in args:
                kwargs["importance"] = max(1, min(10, int(args["importance"])))
            result = self._engine.update(memory_id, **kwargs)
            if result is None:
                return tool_error(f"Memory not found: {memory_id}")
            return json.dumps({"updated": True, "id": memory_id})
        except Exception as exc:
            return tool_error(f"Update failed: {exc}")

    def _tool_forget(self, args: dict) -> str:
        memory_id = str(args.get("memory_id") or "").strip()
        if not memory_id:
            return tool_error("memory_id is required")
        try:
            self._engine.delete(memory_id)
            return json.dumps({"deleted": True, "id": memory_id})
        except Exception as exc:
            return tool_error(f"Forget failed: {exc}")

    def _tool_link(self, args: dict) -> str:
        source_id = str(args.get("source_id") or "").strip()
        target_id = str(args.get("target_id") or "").strip()
        if not source_id or not target_id:
            return tool_error("source_id and target_id are required")
        relation = str(args.get("relation") or "related_to").strip()
        try:
            success = self._engine.link(source_id, target_id, relation=relation)
            if not success:
                return tool_error("Link failed: one or both memories not found, or duplicate link")
            return json.dumps({"linked": True, "source_id": source_id, "target_id": target_id, "relation": relation})
        except Exception as exc:
            return tool_error(f"Link failed: {exc}")

    def _tool_conflicts(self, args: dict) -> str:
        memory_id = str(args.get("memory_id") or "").strip() or None
        try:
            limit = max(1, min(50, int(args.get("limit", 10) or 10)))
        except Exception:
            limit = 10
        try:
            results = self._engine.get_conflicts(memory_id=memory_id, limit=limit)
            return json.dumps({"conflicts": results, "count": len(results)})
        except Exception as exc:
            return tool_error(f"Conflict check failed: {exc}")

    def _tool_similar(self, args: dict) -> str:
        memory_id = str(args.get("memory_id") or "").strip()
        if not memory_id:
            return tool_error("memory_id is required")
        try:
            limit = max(1, min(20, int(args.get("limit", 5) or 5)))
        except Exception:
            limit = 5
        try:
            results = self._engine.similar(memory_id, limit=limit)
            formatted = [{"id": r.get("id"), "content": r.get("content", "")[:120]} for r in results]
            return json.dumps({"results": formatted, "count": len(formatted)})
        except Exception as exc:
            return tool_error(f"Similar query failed: {exc}")

    def _tool_topics(self, args: dict) -> str:
        try:
            limit = max(1, min(50, int(args.get("limit", 20) or 20)))
        except Exception:
            limit = 20
        try:
            results = self._engine.topics(limit=limit)
            return json.dumps({"topics": results, "count": len(results)})
        except Exception as exc:
            return tool_error(f"Topics failed: {exc}")

    def _tool_timeline(self, args: dict) -> str:
        try:
            limit = max(1, min(50, int(args.get("limit", 20) or 20)))
        except Exception:
            limit = 20
        memory_type = str(args.get("memory_type") or "").strip() or None
        try:
            results = self._engine.timeline(limit=limit, memory_type=memory_type)
            formatted = []
            for item in results:
                formatted.append({
                    "id": item.get("id"),
                    "content": item.get("content", "")[:120],
                    "memory_type": item.get("memory_type"),
                    "created_at": item.get("created_at"),
                    "importance": item.get("importance"),
                })
            return json.dumps({"results": formatted, "count": len(formatted)})
        except Exception as exc:
            return tool_error(f"Timeline failed: {exc}")

    def _tool_health(self, args: dict) -> str:
        try:
            h = self._engine.health()
            return json.dumps(h)
        except Exception as exc:
            return tool_error(f"Health check failed: {exc}")

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._active or not self._engine:
            return tool_error("Rekal is not initialized")
        aliases = {
            "rekal-save": "rekal_store",
            "rekal-search": "rekal_search",
            "rekal-forget": "rekal_forget",
            "rekal-update": "rekal_update",
            "rekal-link": "rekal_link",
            "rekal-conflicts": "rekal_conflicts",
            "rekal-similar": "rekal_similar",
            "rekal-topics": "rekal_topics",
            "rekal-timeline": "rekal_timeline",
            "rekal-health": "rekal_health",
        }
        tool_name = aliases.get(tool_name, tool_name)
        dispatch = {
            "rekal_store": self._tool_store,
            "rekal_search": self._tool_search,
            "rekal_update": self._tool_update,
            "rekal_forget": self._tool_forget,
            "rekal_link": self._tool_link,
            "rekal_conflicts": self._tool_conflicts,
            "rekal_similar": self._tool_similar,
            "rekal_topics": self._tool_topics,
            "rekal_timeline": self._tool_timeline,
            "rekal_health": self._tool_health,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return tool_error(f"Unknown tool: {tool_name}")
        return handler(args)


def register(ctx):
    ctx.register_memory_provider(RekalMemoryProvider())
