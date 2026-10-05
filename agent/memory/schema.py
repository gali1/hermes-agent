"""Typed records for the memory primitives package.

Ported from the OpenCode/MemPalace Hindsight primitives (upstream
``plugins/memory/rekal/mempalace_hindsight.py``).  Stdlib only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MEMORY_TYPES = (
    "fact",
    "preference",
    "decision",
    "procedure",
    "context",
    "episode",
    "observation",
    "relationship",
    "bug",
    "solution",
    "architecture",
    "configuration",
    "workflow",
)

MEMORY_SCOPES = ("global", "user", "project", "repository", "session", "task")

SOURCE_TYPES = (
    "conversation",
    "tool",
    "file",
    "git",
    "user",
    "agent",
    "memory",
    "inferred",
    "graph",
)


@dataclass
class MemoryScope:
    kind: str = "global"
    value: str = ""


@dataclass
class MemorySource:
    type: str = "conversation"
    session_id: str = ""
    turn_id: str = ""
    message_id: str = ""
    detail: str = ""


@dataclass
class MemoryEvidence:
    proof_count: int = 1
    confidence: float = 0.5
    last_reinforced_at: str | None = None
    sources: list = field(default_factory=list)


@dataclass
class MemoryRecord:
    id: str = ""
    content: str = ""
    memory_type: str = "fact"
    scope: str = "global"
    project: str | None = None
    session_id: str = ""
    user_id: str = ""
    tags: list = field(default_factory=list)
    importance: float = 0.5
    status: str = "active"
    valid_from: str | None = None
    valid_until: str | None = None
    created_at: str = ""
    updated_at: str = ""
    access_count: int = 0
    last_accessed_at: str | None = None
    proof_count: int = 1
    last_reinforced_at: str | None = None
    superseded_by: str | None = None
    source: MemorySource | dict | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryLink:
    from_id: str
    to_id: str
    relation: str = "related_to"
    created_at: str = ""
    metadata: dict | None = None


@dataclass
class MemoryConflict:
    memory_id: str
    related_id: str
    content: str = ""
    related_content: str = ""
    relation: str = "contradicts"
    confidence: float = 0.0
    created_at: str = ""


@dataclass
class MemoryQuery:
    query: str = ""
    limit: int = 10
    memory_type: str | None = None
    scope: str | None = None
    project: str | None = None
    session_id: str = ""
    tags: list | None = None
    min_importance: float = 0.0
    fusion: str | None = None
    graph_expand: bool = False
    temporal: bool = False
    strategy_boosts: dict | None = None


@dataclass
class MemoryResult(MemoryRecord):
    score: float = 0.0
    fts_score: float = 0.0
    vec_score: float = 0.0
    recency_score: float = 0.0
    access_score: float = 0.0
    rrf_score: float = 0.0
    rrf_rank: int = 0
    source_ranks: dict | None = None
    graph_score: float = 0.0
    temporal_score: float = 0.0
    via: str | None = None
    source_kind: str = "engine"


_RECORD_FIELDS = (
    "id",
    "content",
    "memory_type",
    "scope",
    "project",
    "session_id",
    "user_id",
    "tags",
    "importance",
    "status",
    "valid_from",
    "valid_until",
    "created_at",
    "updated_at",
    "access_count",
    "last_accessed_at",
    "proof_count",
    "last_reinforced_at",
    "superseded_by",
    "source",
    "metadata",
)

_SOURCE_FIELDS = ("type", "session_id", "turn_id", "message_id", "detail")


def _coerce_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return [text]
        if isinstance(parsed, list):
            return parsed
        return [text]
    return [value]


def _coerce_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_source(value: Any) -> Any:
    if isinstance(value, MemorySource):
        return value
    if isinstance(value, dict):
        return MemorySource(**{name: value[name] for name in _SOURCE_FIELDS if name in value})
    return value


def record_from_row(row: dict) -> MemoryRecord:
    """Map a DB row dict onto a MemoryRecord, tolerating missing keys."""
    if not isinstance(row, dict):
        return MemoryRecord()
    values = {name: row[name] for name in _RECORD_FIELDS if name in row}
    values["tags"] = _coerce_list(values.get("tags"))
    values["metadata"] = _coerce_dict(values.get("metadata"))
    if "source" in values:
        values["source"] = _coerce_source(values["source"])
    return MemoryRecord(**values)
