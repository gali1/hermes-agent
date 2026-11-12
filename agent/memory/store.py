"""Structured SQLite memory store for the enhanced Hermes memory layer.

Adapted from the OpenCode/MemPalace Rekal engine (the canonical upstream
implementation) and restructured as a native Hermes core module.  Provides:

  - typed memories with lifecycle (store / update / supersede / delete / link)
  - FTS5 BM25 lexical retrieval with self-healing index
  - optional vector arm (caller-supplied ``vector_search_fn``)
  - evidence accumulation (``proof_count`` / ``last_reinforced_at``)
  - automatic contradiction detection on store
  - Hindsight-derived advanced retrieval: RRF rank fusion, bounded
    multiplicative scoring, graph spreading activation, temporal windows
  - per-project scoring weights
  - scope / project / session filtering
  - failure containment: no operation ever raises into the agent turn

Stdlib only.  The database path is always injected by the caller so the store
stays profile-scoped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from array import array
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent.memory.contradiction import detect_contradiction
from agent.memory.dedup import (
    find_near_duplicate,
    is_degenerate,
    is_duplicate,
    normalize_content,
    trigram_similarity,
)
from agent.memory.evidence import confidence as evidence_confidence
from agent.memory.evidence import source_quality as evidence_source_quality
from agent.memory.graph import expand_links
from agent.memory.ranking import (
    LEXICAL_ANCHOR_MARGIN,
    LEXICAL_ANCHOR_MIN_FTS,
    MAX_VECTOR_WEIGHT,
    boosted_rrf_score,
    combined_score,
    infer_strategy_boosts,
    proof_norm,
    reciprocal_rank_fusion,
    recency_for_range,
)
from agent.memory.temporal import analyze_query, temporal_proximity

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2

_DEFAULT_SEARCH_LIMIT = 10
_DEFAULT_MAX_SEARCH = 50
_ACCESS_SATURATION = 50.0
_DEFAULT_W_FTS = 0.30
_DEFAULT_W_VEC = 0.30
_DEFAULT_W_RECENCY = 0.20
_DEFAULT_W_ACCESS = 0.20
_DEFAULT_HALF_LIFE = 30.0
_CONFLICT_SCAN_LIMIT = 40
_RETRY_ATTEMPTS = 4
_RETRY_BASE_DELAY = 0.04

# Fuzzy near-duplicate scan bounds.  Only reached after an exact content_hash
# miss, and never unbounded: the candidate set is one project's most recent N
# active rows, so insert cost stays flat as the store grows.
_DEFAULT_FUZZY_SCAN = 50
_DEFAULT_FUZZY_THRESHOLD = 0.97
_CONSOLIDATE_SCAN_LIMIT = 200
_TOPIC_SCAN_LIMIT = 500
_TOPIC_MIN_LEN = 3
_EVIDENCE_CONTRADICTION_STEP = 0.05
_EVIDENCE_CONTRADICTION_CAP = 0.25

_ACTIVE = "active"
_SUPERSEDED = "superseded"

#: Bounded stopword set for topic discovery (no external corpus, no model).
_TOPIC_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can could did do does doing down
during each few for from further had has have having he her here hers him his
how i if in into is it its itself just me more most my no nor not now of off on
once only or other our out over own same she should so some such than that the
their them then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your
user users memory memories note notes fact facts thing things something really
actually basically currently today yesterday tomorrow time times way ways
""".split())

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    content          TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    memory_type      TEXT NOT NULL DEFAULT 'fact',
    project          TEXT,
    scope            TEXT NOT NULL DEFAULT 'global',
    session_id       TEXT NOT NULL DEFAULT '',
    user_id          TEXT NOT NULL DEFAULT '',
    wing             TEXT,
    room             TEXT,
    tags             TEXT NOT NULL DEFAULT '[]',
    importance       REAL NOT NULL DEFAULT 0.5,
    status           TEXT NOT NULL DEFAULT 'active',
    valid_from       TEXT,
    valid_until      TEXT,
    source           TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}',
    proof_count      INTEGER NOT NULL DEFAULT 1,
    last_reinforced_at TEXT,
    confidence       REAL NOT NULL DEFAULT 0.5,
    occurred_start   TEXT,
    occurred_end     TEXT,
    superseded_by    TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT
);

CREATE TABLE IF NOT EXISTS memory_links (
    from_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_id       TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL DEFAULT 'related_to',
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id, relation)
);

CREATE TABLE IF NOT EXISTS memory_config (
    project TEXT NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    PRIMARY KEY (project, key)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags, project,
    content='memories',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags, project)
    VALUES (new.rowid, new.content, new.tags, new.project);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, project)
    VALUES ('delete', old.rowid, old.content, old.tags, old.project);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, project)
    VALUES ('delete', old.rowid, old.content, old.tags, old.project);
    INSERT INTO memories_fts(rowid, content, tags, project)
    VALUES (new.rowid, new.content, new.tags, new.project);
END;

CREATE INDEX IF NOT EXISTS idx_memories_project    ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_scope      ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_type       ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_hash       ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_created    ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_updated    ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_session    ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_links_from          ON memory_links(from_id);
CREATE INDEX IF NOT EXISTS idx_links_to            ON memory_links(to_id);
CREATE INDEX IF NOT EXISTS idx_links_relation      ON memory_links(relation);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON memory_embeddings(model);
"""

_MIGRATIONS = [
    "ALTER TABLE memories ADD COLUMN proof_count INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE memories ADD COLUMN last_reinforced_at TEXT",
    # D1: event time beside store time.  Nullable: legacy rows keep NULL and
    # every reader falls back to created_at.
    "ALTER TABLE memories ADD COLUMN occurred_start TEXT",
    "ALTER TABLE memories ADD COLUMN occurred_end TEXT",
    # D2: per-record confidence, neutral default so existing rows are unchanged.
    "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 0.5",
]

_VALID_WEIGHT_KEYS = {"w_fts", "w_vec", "w_recency", "w_access", "half_life"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _clamp_importance(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _parse_ts(timestamp_str: Any) -> Optional[datetime]:
    if not timestamp_str:
        return None
    try:
        dt = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _days_since(timestamp_str: Any) -> float:
    dt = _parse_ts(timestamp_str)
    if dt is None:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc).replace(tzinfo=None) - dt).total_seconds() / 86400.0)


def _recency_score(days: float, half_life: float = _DEFAULT_HALF_LIFE) -> float:
    return math.exp(-0.693 * max(0.0, days) / max(0.1, half_life))


def _access_score(access_count: Any) -> float:
    try:
        count = max(0, int(access_count or 0))
    except (TypeError, ValueError):
        count = 0
    return min(1.0, math.log1p(count) / math.log1p(_ACCESS_SATURATION))


def _normalize_fts(score: Any) -> float:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.0
    if score >= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(score))


def _normalize_vec(distance: Any) -> float:
    try:
        return max(0.0, 1.0 - float(distance))
    except (TypeError, ValueError):
        return 0.0


def _content_key(text: str) -> str:
    return hashlib.md5((text or "")[:200].encode("utf-8", errors="replace")).hexdigest()[:12]


def _quote_fts(query: str, match_any: bool = False) -> str:
    """Build an FTS5 query: prefix match for tokens >= 3 chars, exact below.

    ``match_any`` switches the implicit AND to OR — FTS5 ANDs bare terms, so a
    natural-language query of more than a few words otherwise requires every
    term to appear in one memory and the keyword arm returns nothing.
    """
    tokens = [t for t in re.split(r"\s+", (query or "").replace('"', " ").replace("\x00", "")) if t]
    parts = []
    for raw in tokens:
        safe = re.sub(r"[^\w\-']", "", raw)
        if not safe:
            continue
        if len(safe) >= 3:
            parts.append(f'"{safe}"*')
        else:
            parts.append(f'"{safe}"')
    return (" OR " if match_any else " ").join(parts)


def _list_field(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return [t.strip() for t in value.split(",") if t.strip()]
    return []


def _json_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _iso_ts(value: Any) -> Optional[str]:
    """Normalise an event timestamp to the store's ``YYYY-MM-DD HH:MM:SS`` form.

    Text comparison is what makes the temporal predicate cheap (no per-row
    parsing), so every write has to land in one lexicographically sortable
    format.  Datetimes are converted; strings are parsed and re-rendered when
    possible and otherwise kept as-is so a caller's opaque value is never lost.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    if not text:
        return None
    parsed = _parse_ts(text)
    if parsed is None:
        return text
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(0.95, float(value)))
    except (TypeError, ValueError):
        return default


def _source_type(source: Any) -> Optional[str]:
    """Extract a source *type* string from the stored ``source`` field.

    ``source`` is free-form: a plain string, or a dict (already decoded by
    ``_row_to_dict``) carrying ``type``/``source``/``kind``.  Only the type is
    needed for the quality signal; anything else stays untouched.
    """
    if source is None:
        return None
    if isinstance(source, dict):
        for key in ("type", "source", "kind"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    if isinstance(source, str):
        text = source.strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                return _source_type(parsed)
        return text
    return None


_TOPIC_TOKEN = re.compile(r"[a-z][a-z0-9_+#.\-]*", re.UNICODE)


def _topic_tokens(text: str) -> List[str]:
    """Deterministic, bounded tokenisation for topic discovery."""
    return [
        token for token in _TOPIC_TOKEN.findall((text or "").lower())
        if len(token) >= _TOPIC_MIN_LEN and token not in _TOPIC_STOPWORDS
    ]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MemoryStore:
    """SQLite + FTS5 structured memory store.

    Writes are serialized through ``_write_lock`` because Hermes drives the
    store from both the foreground turn thread (search) and the background
    memory-sync worker (store/reinforce).  Reads use the same connection; the
    sqlite3 module is compiled in serialized mode and ``_execute`` retries on
    transient BUSY errors.
    """

    def __init__(self, db_path: str, *, fuzzy_dedup_limit: int = _DEFAULT_FUZZY_SCAN):
        self._db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._write_lock = threading.RLock()
        self._vector_max_share = 0.5
        self._vector_weight: Optional[float] = None
        try:
            self._fuzzy_dedup_limit = max(0, int(fuzzy_dedup_limit))
        except (TypeError, ValueError):
            self._fuzzy_dedup_limit = _DEFAULT_FUZZY_SCAN
        self.db = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA cache_size=-8000")
        self.db.execute("PRAGMA temp_store=MEMORY")
        with self._write_lock:
            self.db.executescript(_SCHEMA_SQL)
            self._run_migrations()
            self._repair_fts_if_needed()
            self.db.execute("ANALYZE")
            self.db.commit()

    # -- Connection helpers -------------------------------------------------

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        last_exc: Optional[Exception] = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                return self.db.execute(sql, params)
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if "locked" in str(exc).lower() and attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                raise
        raise last_exc  # pragma: no cover

    def _commit(self) -> None:
        try:
            self.db.commit()
        except Exception:
            logger.debug("memory store commit failed", exc_info=True)

    def _run_migrations(self) -> None:
        existing = {row[1] for row in self._execute("PRAGMA table_info(memories)")}
        for stmt in _MIGRATIONS:
            column = stmt.split("ADD COLUMN")[1].strip().split()[0]
            if column not in existing:
                try:
                    self._execute(stmt)
                except sqlite3.OperationalError:
                    logger.debug("memory migration skipped: %s", stmt)
        self._commit()

    def _repair_fts_if_needed(self) -> None:
        """Rebuild the FTS index when it drifts out of sync with ``memories``."""
        try:
            memories = self._execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            indexed = self._execute("SELECT COUNT(*) FROM memories_fts_docsize").fetchone()[0]
            if memories == indexed:
                return
            logger.debug("memory FTS out of sync (%s rows, %s indexed); rebuilding", memories, indexed)
        except sqlite3.DatabaseError:
            logger.debug("memory FTS unreadable; rebuilding", exc_info=True)
        try:
            self._execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self._commit()
        except sqlite3.DatabaseError:
            try:
                self._execute("DROP TABLE IF EXISTS memories_fts")
                self._execute(
                    "CREATE VIRTUAL TABLE memories_fts USING fts5("
                    "content, tags, project, content='memories', content_rowid='rowid')"
                )
                self._execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                self._commit()
            except sqlite3.DatabaseError:
                logger.warning("memory FTS unavailable; keyword search degraded", exc_info=True)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        if row is None:
            return {}
        d = dict(row)
        for field in ("tags", "metadata"):
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        if isinstance(d.get("source"), str):
            try:
                parsed = json.loads(d["source"])
                if isinstance(parsed, (dict, list)):
                    d["source"] = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("proof_count") is None:
            d["proof_count"] = 1
        if d.get("confidence") is None:
            d["confidence"] = 0.5
        if d.get("importance") is None:
            d["importance"] = 0.5
        return d

    def _fetch_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = self._execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    # -- Filters ------------------------------------------------------------

    @staticmethod
    def _filter_sql(project=None, memory_type=None, scope=None, tags=None,
                    min_importance=0.0, alias: str = "memories") -> Tuple[List[str], list]:
        conditions = [
            f"{alias}.status = 'active'",
            f"{alias}.id NOT IN (SELECT to_id FROM memory_links WHERE relation = 'supersedes')",
        ]
        params: list = []
        if project:
            conditions.append(f"{alias}.project = ?")
            params.append(project)
        if memory_type:
            conditions.append(f"{alias}.memory_type = ?")
            params.append(memory_type)
        if scope:
            conditions.append(f"{alias}.scope = ?")
            params.append(scope)
        if min_importance and float(min_importance) > 0:
            conditions.append(f"{alias}.importance >= ?")
            params.append(float(min_importance))
        for tag in tags or []:
            conditions.append(
                f"EXISTS (SELECT 1 FROM json_each({alias}.tags) WHERE value = ?)"
            )
            params.append(str(tag))
        return conditions, params

    # -- Temporal predicates (D3/D4) ----------------------------------------

    @staticmethod
    def _window_bounds(window) -> Optional[Tuple[str, str]]:
        """Render a ``(start, end)`` datetime pair as sortable store timestamps."""
        if not window:
            return None
        try:
            start, end = window
        except (TypeError, ValueError):
            return None
        start_ts = _iso_ts(start)
        end_ts = _iso_ts(end)
        if not start_ts or not end_ts:
            return None
        return start_ts, end_ts

    def _time_predicate(self, alias: str, window) -> Tuple[str, list]:
        """SQL half of the hard temporal filter.

        Event time wins when present; ``created_at`` is the fallback, so legacy
        rows with no ``occurred_*`` keep their old meaning.  A memory occupies
        ``[occurred_start or created_at, occurred_end or occurred_start or
        created_at]`` and overlaps the window when ``start <= window_end AND
        end >= window_start``.  Values are normalised to one fixed-width text
        format on write, so plain string comparison is exact.
        """
        bounds = self._window_bounds(window)
        if not bounds:
            return "", []
        start_ts, end_ts = bounds
        sql = (
            f"COALESCE({alias}.occurred_start, {alias}.created_at) <= ?"
            f" AND COALESCE({alias}.occurred_end, {alias}.occurred_start, "
            f"{alias}.created_at) >= ?"
        )
        return sql, [end_ts, start_ts]

    def _memory_in_window(self, mem: Dict[str, Any], window) -> bool:
        """Python half of the hard temporal filter; must agree with SQL."""
        bounds = self._window_bounds(window)
        if not bounds:
            return True
        start_ts, end_ts = bounds
        start = mem.get("occurred_start") or mem.get("created_at")
        end = mem.get("occurred_end") or mem.get("occurred_start") or start
        if not start or not end:
            return False
        return str(start)[:19] <= end_ts and str(end)[:19] >= start_ts

    @staticmethod
    def _event_bounds(mem: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Event-time bounds for scoring, falling back to store time."""
        start = _parse_ts(mem.get("occurred_start"))
        end = _parse_ts(mem.get("occurred_end")) or start
        return start, end

    @staticmethod
    def _recency_reference(mem: Dict[str, Any]) -> Any:
        """Event time when present, else store time (D4)."""
        return mem.get("occurred_start") or mem.get("created_at")

    def _list_recent(self, limit: int, *, project=None, memory_type=None, scope=None,
                     tags=None, min_importance=0.0) -> List[Dict[str, Any]]:
        conditions, params = self._filter_sql(
            project, memory_type, scope, tags, min_importance
        )
        sql = (
            "SELECT * FROM memories WHERE " + " AND ".join(conditions)
            + " ORDER BY updated_at DESC LIMIT ?"
        )
        rows = self._execute(sql, tuple(params) + (limit,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- Lifecycle ----------------------------------------------------------

    def store(self, content, memory_type="fact", *, project=None, scope="global",
              session_id="", user_id="", tags=None, importance=0.5,
              source=None, metadata=None, allow_duplicate=False,
              occurred_start=None, occurred_end=None, confidence=None,
              fuzzy_dedup: bool = True) -> Dict[str, Any]:
        try:
            content = (content or "").strip()
            if is_degenerate(content):
                return {"success": False, "error": "content is required and must not be degenerate."}
            chash = _content_hash(content)
            occurred_start_ts = _iso_ts(occurred_start)
            occurred_end_ts = _iso_ts(occurred_end) or occurred_start_ts
            confidence_value = _clamp_confidence(confidence)
            with self._write_lock:
                if not allow_duplicate:
                    existing = self._execute(
                        "SELECT id FROM memories WHERE content_hash = ? AND status = 'active' LIMIT 1",
                        (chash,),
                    ).fetchone()
                    if existing:
                        proof = self._reinforce_locked(existing["id"])
                        result: Dict[str, Any] = {
                            "success": True,
                            "memory_id": existing["id"],
                            "duplicate": True,
                            "reinforced": True,
                            "message": "Memory already exists with identical content.",
                        }
                        if proof is not None:
                            result["proof_count"] = proof
                        return result

                    # B5: only after the exact-hash fast path misses do we pay
                    # for a fuzzy scan, and only over one project's most recent
                    # ``_fuzzy_dedup_limit`` active rows.  A near-duplicate is
                    # reinforced in place — never merged, never deleted.
                    if fuzzy_dedup and self._fuzzy_dedup_limit > 0:
                        fuzzy_hit = self._find_fuzzy_duplicate_locked(content, project, chash)
                        if fuzzy_hit:
                            dup_id, similarity = fuzzy_hit
                            proof = self._reinforce_locked(dup_id)
                            conf = self._bump_confidence_locked(dup_id)
                            fuzzy_result: Dict[str, Any] = {
                                "success": True,
                                "memory_id": dup_id,
                                "duplicate": True,
                                "fuzzy": True,
                                "similarity": round(similarity, 4),
                                "reinforced": True,
                                "message": "Memory is a near-duplicate of existing content.",
                            }
                            if proof is not None:
                                fuzzy_result["proof_count"] = proof
                            if conf is not None:
                                fuzzy_result["confidence"] = conf
                            return fuzzy_result

                memory_id = _new_id()
                ts = _now()
                imp = _clamp_importance(importance)
                self._execute(
                    """INSERT INTO memories
                       (id, content, content_hash, memory_type, project, scope,
                        session_id, user_id, tags, importance, source, metadata,
                        proof_count, confidence, occurred_start, occurred_end,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory_id, content, chash, str(memory_type or "fact"),
                        project, str(scope or "global"), str(session_id or ""),
                        str(user_id or ""),
                        json.dumps(_list_field(tags), ensure_ascii=False),
                        imp, _json_or_none(source),
                        json.dumps(metadata or {}, ensure_ascii=False),
                        1, confidence_value, occurred_start_ts, occurred_end_ts, ts, ts,
                    ),
                )
                self._commit()
                conflicts = self._detect_conflicts_locked(memory_id, content, project)

            result = {
                "success": True,
                "memory_id": memory_id,
                "memory_type": str(memory_type or "fact"),
                "project": project,
                "scope": str(scope or "global"),
                "importance": imp,
                "confidence": confidence_value,
            }
            if occurred_start_ts:
                result["occurred_start"] = occurred_start_ts
            if occurred_end_ts:
                result["occurred_end"] = occurred_end_ts
            if conflicts:
                result["contradicts"] = conflicts
            return result
        except Exception as exc:
            logger.debug("memory store failed", exc_info=True)
            return {"success": False, "error": str(exc)}

    def _find_fuzzy_duplicate_locked(self, content: str,
                                     project: Optional[str],
                                     chash: str) -> Optional[Tuple[str, float]]:
        """Nearest near-duplicate in the same project's most recent N rows.

        Returns ``(memory_id, similarity)`` or None.  The scan is bounded by
        ``_fuzzy_dedup_limit`` and runs only on the exact-hash miss path.
        Degenerate rows are skipped rather than compared: a punctuation-only
        row can otherwise look "similar" to short real content.
        """
        try:
            if project:
                rows = self._execute(
                    "SELECT id, content FROM memories WHERE status = 'active' "
                    "AND content_hash != ? AND project = ? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (chash, project, self._fuzzy_dedup_limit),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT id, content FROM memories WHERE status = 'active' "
                    "AND content_hash != ? AND project IS NULL "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (chash, self._fuzzy_dedup_limit),
                ).fetchall()
        except Exception:
            logger.debug("fuzzy duplicate scan failed", exc_info=True)
            return None
        return find_near_duplicate(
            content,
            [(row["id"], row["content"] or "") for row in rows],
            _DEFAULT_FUZZY_THRESHOLD,
        )

    def _bump_confidence_locked(self, memory_id: str,
                                bump: float = 0.03) -> Optional[float]:
        """Raise a row's stored confidence by ``bump``, capped at 0.95."""
        row = self._execute(
            "SELECT confidence FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        target = min(0.95, _clamp_confidence(row["confidence"]) + bump)
        self._execute(
            "UPDATE memories SET confidence = ? WHERE id = ?", (target, memory_id)
        )
        self._commit()
        return target

    def set_fuzzy_dedup(self, limit) -> Dict[str, Any]:
        """Bound (or disable with 0) the near-duplicate candidate scan."""
        try:
            self._fuzzy_dedup_limit = max(0, int(limit))
        except (TypeError, ValueError):
            return {"success": False, "error": "limit must be an integer"}
        return {"success": True, "fuzzy_dedup_limit": self._fuzzy_dedup_limit}

    def _reinforce_locked(self, memory_id: str) -> Optional[int]:
        cursor = self._execute(
            "UPDATE memories SET proof_count = COALESCE(proof_count, 1) + 1, "
            "last_reinforced_at = ? WHERE id = ?",
            (_now(), memory_id),
        )
        self._commit()
        if cursor.rowcount == 0:
            return None
        row = self._execute("SELECT proof_count FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return int(row["proof_count"]) if row else None

    def reinforce(self, memory_id: str) -> Dict[str, Any]:
        try:
            with self._write_lock:
                proof = self._reinforce_locked(memory_id)
            if proof is None:
                return {"success": False, "error": f"Memory {memory_id} not found"}
            return {"success": True, "memory_id": memory_id, "proof_count": proof}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def has_content(self, content: str) -> bool:
        """Return True when an active memory already holds this exact content."""
        try:
            chash = _content_hash((content or "").strip())
            row = self._execute(
                "SELECT 1 FROM memories WHERE content_hash = ? AND status = 'active' LIMIT 1",
                (chash,),
            ).fetchone()
            return row is not None
        except Exception:
            logger.debug("has_content check failed", exc_info=True)
            return False

    def _detect_conflicts_locked(self, memory_id: str, content: str,
                                 project: Optional[str]) -> List[Dict[str, Any]]:
        try:
            if project:
                rows = self._execute(
                    "SELECT id, content FROM memories WHERE id != ? AND status = 'active' "
                    "AND project = ? ORDER BY created_at DESC LIMIT ?",
                    (memory_id, project, _CONFLICT_SCAN_LIMIT),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT id, content FROM memories WHERE id != ? AND status = 'active' "
                    "ORDER BY created_at DESC LIMIT ?",
                    (memory_id, _CONFLICT_SCAN_LIMIT),
                ).fetchall()
            found: List[Dict[str, Any]] = []
            for row in rows:
                hit, confidence = detect_contradiction(content, row["content"])
                if not hit:
                    continue
                self._execute(
                    "INSERT OR IGNORE INTO memory_links "
                    "(from_id, to_id, relation, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (memory_id, row["id"], "contradicts",
                     json.dumps({"confidence": confidence}), _now()),
                )
                found.append({
                    "memory_id": row["id"],
                    "confidence": confidence,
                    "content": (row["content"] or "")[:160],
                })
            if found:
                self._commit()
            return found
        except Exception:
            logger.debug("contradiction scan skipped", exc_info=True)
            return []

    def get(self, memory_id: str, track_access: bool = True) -> Optional[Dict[str, Any]]:
        try:
            row = self._execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not row:
                return None
            result = self._row_to_dict(row)
            if track_access:
                with self._write_lock:
                    self._execute(
                        "UPDATE memories SET access_count = access_count + 1, "
                        "last_accessed_at = ? WHERE id = ?",
                        (_now(), memory_id),
                    )
                    self._commit()
                result["access_count"] = int(result.get("access_count") or 0) + 1
            return result
        except Exception:
            logger.debug("memory get failed", exc_info=True)
            return None

    def update(self, memory_id: str, *, content=None, memory_type=None, tags=None,
               importance=None, metadata=None, status=None,
               valid_until=None, confidence=None, occurred_start=None,
               occurred_end=None) -> Dict[str, Any]:
        try:
            if content is None and memory_type is None and tags is None and importance is None \
                    and metadata is None and status is None and valid_until is None \
                    and confidence is None and occurred_start is None and occurred_end is None:
                return {"success": True, "memory_id": memory_id, "noop": True}
            updates: Dict[str, Any] = {"updated_at": _now()}
            if content is not None:
                content = str(content).strip()
                if is_degenerate(content):
                    return {"success": False, "error": "content must not be degenerate."}
                updates["content"] = content
                updates["content_hash"] = _content_hash(content)
            if memory_type is not None:
                updates["memory_type"] = str(memory_type)
            if tags is not None:
                updates["tags"] = json.dumps(_list_field(tags), ensure_ascii=False)
            if importance is not None:
                updates["importance"] = _clamp_importance(importance)
            if metadata is not None:
                updates["metadata"] = json.dumps(metadata or {}, ensure_ascii=False)
            if status is not None:
                updates["status"] = str(status)
            if valid_until is not None:
                updates["valid_until"] = str(valid_until)
            if confidence is not None:
                updates["confidence"] = _clamp_confidence(confidence)
            if occurred_start is not None:
                updates["occurred_start"] = _iso_ts(occurred_start)
            if occurred_end is not None:
                updates["occurred_end"] = _iso_ts(occurred_end)
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            with self._write_lock:
                cursor = self._execute(
                    f"UPDATE memories SET {set_clause} WHERE id = ?",
                    tuple(updates.values()) + (memory_id,),
                )
                self._commit()
            if cursor.rowcount == 0:
                return {"success": False, "error": f"Memory {memory_id} not found"}
            return {"success": True, "memory_id": memory_id}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def supersede(self, old_id: str, new_content: str, *, memory_type=None,
                  project=None, scope=None, tags=None, importance=None,
                  source=None, metadata=None, confidence=None,
                  occurred_start=None, occurred_end=None) -> Dict[str, Any]:
        try:
            with self._write_lock:
                old = self._execute(
                    "SELECT * FROM memories WHERE id = ?", (old_id,)
                ).fetchone()
                if not old:
                    return {"success": False, "error": f"Memory {old_id} not found"}
                new_content = (new_content or "").strip()
                if is_degenerate(new_content):
                    return {"success": False, "error": "new_content is required and must not be degenerate."}
                old_dict = self._row_to_dict(old)
                new_id = _new_id()
                ts = _now()
                effective_tags = tags if tags is not None else old_dict.get("tags") or []
                effective_importance = (
                    _clamp_importance(importance) if importance is not None
                    else _clamp_importance(old_dict.get("importance", 0.5))
                )
                self._execute(
                    """INSERT INTO memories
                       (id, content, content_hash, memory_type, project, scope,
                        session_id, user_id, tags, importance, source, metadata,
                        confidence, occurred_start, occurred_end,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_id, new_content, _content_hash(new_content),
                        str(memory_type or old_dict.get("memory_type") or "fact"),
                        project if project is not None else old_dict.get("project"),
                        str(scope or old_dict.get("scope") or "global"),
                        old_dict.get("session_id") or "", old_dict.get("user_id") or "",
                        json.dumps(_list_field(effective_tags), ensure_ascii=False),
                        effective_importance,
                        _json_or_none(source) if source is not None else _json_or_none(old_dict.get("source")),
                        json.dumps(metadata if metadata is not None else old_dict.get("metadata") or {}, ensure_ascii=False),
                        _clamp_confidence(
                            confidence if confidence is not None else old_dict.get("confidence")
                        ),
                        _iso_ts(occurred_start) if occurred_start is not None
                        else old_dict.get("occurred_start"),
                        _iso_ts(occurred_end) if occurred_end is not None
                        else old_dict.get("occurred_end"),
                        ts, ts,
                    ),
                )
                self._execute(
                    "UPDATE memories SET status = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
                    (_SUPERSEDED, new_id, ts, old_id),
                )
                self._execute(
                    "INSERT OR IGNORE INTO memory_links (from_id, to_id, relation, metadata, created_at) "
                    "VALUES (?, ?, 'supersedes', '{}', ?)",
                    (new_id, old_id, ts),
                )
                self._commit()
            return {
                "success": True,
                "new_id": new_id,
                "old_id": old_id,
                "fact": f"Created {new_id} superseding {old_id}",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def delete(self, memory_id: str) -> Dict[str, Any]:
        try:
            with self._write_lock:
                cursor = self._execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                self._commit()
            if cursor.rowcount == 0:
                return {"success": False, "error": f"Memory {memory_id} not found"}
            return {"success": True, "memory_id": memory_id}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # -- Links --------------------------------------------------------------

    def link(self, from_id: str, to_id: str, relation: str = "related_to",
             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            with self._write_lock:
                for memory_id in (from_id, to_id):
                    if not self._execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone():
                        return {"success": False, "error": f"Memory {memory_id} not found"}
                self._execute(
                    "INSERT OR IGNORE INTO memory_links "
                    "(from_id, to_id, relation, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (from_id, to_id, str(relation or "related_to"),
                     json.dumps(metadata or {}, ensure_ascii=False), _now()),
                )
                self._commit()
            return {"success": True, "from": from_id, "to": to_id, "relation": relation}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def unlink(self, from_id: str, to_id: str, relation: Optional[str] = None) -> Dict[str, Any]:
        try:
            with self._write_lock:
                if relation:
                    cursor = self._execute(
                        "DELETE FROM memory_links WHERE from_id = ? AND to_id = ? AND relation = ?",
                        (from_id, to_id, relation),
                    )
                else:
                    cursor = self._execute(
                        "DELETE FROM memory_links WHERE from_id = ? AND to_id = ?",
                        (from_id, to_id),
                    )
                self._commit()
            if cursor.rowcount == 0:
                return {"success": False, "error": "No matching link found"}
            return {"success": True, "from": from_id, "to": to_id, "relation": relation}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # -- Consolidation (D9) --------------------------------------------------

    def consolidate(self, project: Optional[str] = None, *, threshold: float = 0.97,
                    scan_limit: int = _CONSOLIDATE_SCAN_LIMIT,
                    dry_run: bool = False) -> Dict[str, Any]:
        """Link near-duplicate memories to a canonical row.  Never deletes.

        Groups the project's oldest-first active rows (bounded by
        ``scan_limit``) with :func:`agent.memory.dedup.is_duplicate`, keeps the
        oldest row as canonical, and for every duplicate writes a
        ``duplicate_of`` link plus a ``supersedes`` link from the canonical,
        marks the duplicate ``status='superseded'`` with ``superseded_by`` set,
        and folds its proof count into the canonical row.

        Originals keep their content, their evidence counters and their rows;
        :meth:`delete` is never called and no evidence is discarded.  Comparison
        is capped at the first 600 characters so the pass stays bounded.
        """
        try:
            try:
                limit = max(1, min(500, int(scan_limit)))
            except (TypeError, ValueError):
                limit = _CONSOLIDATE_SCAN_LIMIT
            try:
                threshold_value = max(0.5, min(1.0, float(threshold)))
            except (TypeError, ValueError):
                threshold_value = 0.97

            sql = (
                "SELECT id, content, project, proof_count, confidence "
                "FROM memories WHERE status = 'active'"
            )
            params: list = []
            if project:
                sql += " AND project = ?"
                params.append(project)
            sql += " ORDER BY created_at ASC, rowid ASC LIMIT ?"
            params.append(limit)
            rows = self._execute(sql, tuple(params)).fetchall()

            canonicals: Dict[Optional[str], List[Dict[str, Any]]] = {}
            linked: List[Dict[str, Any]] = []
            skipped = 0
            ts = _now()
            with self._write_lock:
                for row in rows:
                    content = row["content"] or ""
                    if is_degenerate(content):
                        skipped += 1
                        continue
                    bucket = canonicals.setdefault(row["project"], [])
                    head = content[:600]
                    match = None
                    similarity = 0.0
                    for candidate in bucket:
                        if candidate["head"] == head or is_duplicate(
                            head, candidate["head"], threshold_value
                        ):
                            match = candidate
                            similarity = trigram_similarity(
                                normalize_content(head), normalize_content(candidate["head"])
                            )
                            break
                    if match is None:
                        bucket.append({
                            "id": row["id"],
                            "head": head,
                            "content": content,
                            "proof_count": int(row["proof_count"] or 1),
                            "confidence": _clamp_confidence(row["confidence"]),
                            "duplicates": [],
                        })
                        continue
                    match["duplicates"].append(row["id"])
                    match["proof_count"] += int(row["proof_count"] or 1)
                    match["confidence"] = max(
                        match["confidence"], _clamp_confidence(row["confidence"])
                    )
                    linked.append({
                        "duplicate_id": row["id"],
                        "canonical_id": match["id"],
                        "similarity": round(similarity, 4),
                        "proof_count": int(row["proof_count"] or 1),
                    })
                    if dry_run:
                        continue
                    metadata = json.dumps(
                        {"similarity": round(similarity, 4), "consolidated_at": ts},
                        ensure_ascii=False,
                    )
                    self._execute(
                        "INSERT OR IGNORE INTO memory_links "
                        "(from_id, to_id, relation, metadata, created_at) "
                        "VALUES (?, ?, 'duplicate_of', ?, ?)",
                        (row["id"], match["id"], metadata, ts),
                    )
                    self._execute(
                        "INSERT OR IGNORE INTO memory_links "
                        "(from_id, to_id, relation, metadata, created_at) "
                        "VALUES (?, ?, 'supersedes', '{}', ?)",
                        (match["id"], row["id"], ts),
                    )
                    self._execute(
                        "UPDATE memories SET status = ?, superseded_by = ?, updated_at = ? "
                        "WHERE id = ?",
                        (_SUPERSEDED, match["id"], ts, row["id"]),
                    )
                if not dry_run:
                    for bucket in canonicals.values():
                        for canonical in bucket:
                            if not canonical["duplicates"]:
                                continue
                            self._execute(
                                "UPDATE memories SET proof_count = ?, confidence = ?, "
                                "last_reinforced_at = ? WHERE id = ?",
                                (canonical["proof_count"], canonical["confidence"],
                                 ts, canonical["id"]),
                            )
                    self._commit()

            groups = [
                {
                    "canonical_id": canonical["id"],
                    "duplicate_ids": canonical["duplicates"],
                    "proof_count": canonical["proof_count"],
                }
                for bucket in canonicals.values() for canonical in bucket
                if canonical["duplicates"]
            ]
            return {
                "success": True,
                "dry_run": bool(dry_run),
                "threshold": threshold_value,
                "scanned": len(rows),
                "skipped_degenerate": skipped,
                "linked": linked,
                "groups": groups,
                "groups_count": len(groups),
                "deleted": 0,
            }
        except Exception as exc:
            logger.debug("memory consolidate failed", exc_info=True)
            return {"success": False, "error": str(exc), "deleted": 0}

    # -- Embeddings ---------------------------------------------------------

    def set_vector_max_share(self, value) -> None:
        """Bound the share of fused results that may be vector-only."""
        try:
            self._vector_max_share = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            self._vector_max_share = 0.5

    def set_vector_weight(self, value) -> None:
        """Set the default vector-arm weight, hard-capped at MAX_VECTOR_WEIGHT.

        Applied as a default only: per-project ``memory_config`` rows and
        per-call ``w_vec`` arguments still override it, and the final value is
        clamped so the vector arm can never dominate lexical retrieval.
        """
        try:
            self._vector_weight = max(0.0, min(MAX_VECTOR_WEIGHT, float(value)))
        except (TypeError, ValueError):
            self._vector_weight = None

    def set_embedding(self, memory_id: str, vector, model: str) -> bool:
        """Persist a float32 embedding for a memory. Failure-safe."""
        try:
            values = [float(x) for x in vector]
            if not values:
                return False
            blob = array("f", values).tobytes()
            with self._write_lock:
                self._execute(
                    "INSERT INTO memory_embeddings (memory_id, model, dim, vector, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(memory_id) DO UPDATE SET model = excluded.model, "
                    "dim = excluded.dim, vector = excluded.vector, updated_at = excluded.updated_at",
                    (memory_id, str(model), len(values), blob, _now()),
                )
                self._commit()
            return True
        except Exception:
            logger.debug("memory set_embedding failed", exc_info=True)
            return False

    def get_embedding(self, memory_id: str) -> Optional[Tuple[List[float], str, int]]:
        """Return ``(vector, model, dim)`` for a memory, or None."""
        try:
            row = self._execute(
                "SELECT model, dim, vector FROM memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if not row:
                return None
            values = array("f")
            values.frombytes(row["vector"])
            return list(values), row["model"], int(row["dim"])
        except Exception:
            logger.debug("memory get_embedding failed", exc_info=True)
            return None

    def delete_embedding(self, memory_id: str) -> bool:
        try:
            with self._write_lock:
                self._execute(
                    "DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
                )
                self._commit()
            return True
        except Exception:
            logger.debug("memory delete_embedding failed", exc_info=True)
            return False

    def load_embeddings(self, model: str) -> Dict[str, List[float]]:
        """All stored vectors for ``model`` as ``{memory_id: [floats]}``."""
        try:
            rows = self._execute(
                "SELECT memory_id, vector FROM memory_embeddings WHERE model = ?",
                (model,),
            ).fetchall()
            result: Dict[str, List[float]] = {}
            for row in rows:
                values = array("f")
                values.frombytes(row["vector"])
                result[row["memory_id"]] = list(values)
            return result
        except Exception:
            logger.debug("memory load_embeddings failed", exc_info=True)
            return {}

    def embedding_count(self, model: Optional[str] = None) -> int:
        try:
            if model is None:
                row = self._execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()
            else:
                row = self._execute(
                    "SELECT COUNT(*) FROM memory_embeddings WHERE model = ?", (model,)
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            logger.debug("memory embedding_count failed", exc_info=True)
            return 0

    def missing_embedding_count(self, model: Optional[str] = None) -> int:
        """Active memories with no embedding row.

        ``model=None`` means "no embedding row for any model".
        """
        try:
            if model is None:
                row = self._execute(
                    "SELECT COUNT(*) FROM memories m WHERE m.status = 'active' "
                    "AND NOT EXISTS (SELECT 1 FROM memory_embeddings e "
                    "WHERE e.memory_id = m.id)"
                ).fetchone()
            else:
                row = self._execute(
                    "SELECT COUNT(*) FROM memories m WHERE m.status = 'active' "
                    "AND NOT EXISTS (SELECT 1 FROM memory_embeddings e "
                    "WHERE e.memory_id = m.id AND e.model = ?)",
                    (model,),
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            logger.debug("memory missing_embedding_count failed", exc_info=True)
            return 0

    def missing_embedding_ids(self, model: str, limit: int = 64) -> List[Tuple[str, str]]:
        """``(id, content)`` for active memories lacking an embedding for ``model``.

        Oldest-created first so a background indexer makes steady forward
        progress without rescanning the same rows.
        """
        try:
            rows = self._execute(
                "SELECT m.id, m.content FROM memories m WHERE m.status = 'active' "
                "AND NOT EXISTS (SELECT 1 FROM memory_embeddings e "
                "WHERE e.memory_id = m.id AND e.model = ?) "
                "ORDER BY m.created_at ASC, m.rowid ASC LIMIT ?",
                (model, max(1, int(limit or 64))),
            ).fetchall()
            return [(row["id"], row["content"]) for row in rows]
        except Exception:
            logger.debug("memory missing_embedding_ids failed", exc_info=True)
            return []

    # -- Config -------------------------------------------------------------

    def _resolve_weights(self, project=None, w_fts=None, w_vec=None,
                         w_recency=None, w_access=None,
                         half_life=None) -> Dict[str, float]:
        result = {
            "w_fts": _DEFAULT_W_FTS,
            "w_vec": _DEFAULT_W_VEC,
            "w_recency": _DEFAULT_W_RECENCY,
            "w_access": _DEFAULT_W_ACCESS,
            "half_life": _DEFAULT_HALF_LIFE,
        }
        if self._vector_weight is not None:
            result["w_vec"] = self._vector_weight
        if project:
            try:
                for row in self._execute(
                    "SELECT key, value FROM memory_config WHERE project = ?", (project,)
                ):
                    if row["key"] in result:
                        try:
                            result[row["key"]] = float(row["value"])
                        except (TypeError, ValueError):
                            pass
            except Exception:
                logger.debug("weight resolution failed for project %s", project, exc_info=True)
        for key, value in (
            ("w_fts", w_fts), ("w_vec", w_vec), ("w_recency", w_recency),
            ("w_access", w_access), ("half_life", half_life),
        ):
            if value is not None:
                try:
                    result[key] = float(value)
                except (TypeError, ValueError):
                    pass
        try:
            result["w_vec"] = max(0.0, min(MAX_VECTOR_WEIGHT, float(result["w_vec"])))
        except (TypeError, ValueError):
            result["w_vec"] = _DEFAULT_W_VEC
        return result

    def set_config(self, project: str, key: str, value: Any) -> Dict[str, Any]:
        if key not in _VALID_WEIGHT_KEYS:
            return {"error": f"Invalid key '{key}'. Valid: {', '.join(sorted(_VALID_WEIGHT_KEYS))}"}
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return {"error": f"Value must be numeric, got: {value}"}
        try:
            with self._write_lock:
                self._execute(
                    "INSERT INTO memory_config (project, key, value) VALUES (?, ?, ?) "
                    "ON CONFLICT (project, key) DO UPDATE SET value = excluded.value",
                    (project, key, str(numeric)),
                )
                self._commit()
            return {"success": True, "key": key, "value": numeric, "project": project}
        except Exception as exc:
            return {"error": str(exc)}

    # -- Retrieval ----------------------------------------------------------

    def search(self, query: str, limit: int = _DEFAULT_SEARCH_LIMIT, *,
               project=None, memory_type=None, scope=None, tags=None,
               min_importance: float = 0.0, fusion=None, graph_expand=False,
               temporal=False, temporal_filter: bool = False,
               strategy_boosts=None, vector_search_fn=None,
               now=None, w_fts=None, w_vec=None, w_recency=None,
               w_access=None, half_life=None, track_access: bool = True) -> Dict[str, Any]:
        """Retrieve memories.

        ``temporal`` derives a time window from the query and feeds it to the
        temporal-proximity *score*.  ``temporal_filter`` additionally makes the
        window a hard predicate: rows whose event time (``occurred_start`` /
        ``occurred_end``) — or ``created_at`` when no event time is stored —
        does not overlap the window are excluded entirely.  It is opt-in so
        existing callers keep score-only behaviour.
        """
        try:
            limit = max(1, min(_DEFAULT_MAX_SEARCH, int(limit or _DEFAULT_SEARCH_LIMIT)))
        except (TypeError, ValueError):
            limit = _DEFAULT_SEARCH_LIMIT
        weights = self._resolve_weights(
            project, w_fts, w_vec, w_recency, w_access, half_life
        )

        if not query or not query.strip():
            results = self._list_recent(
                limit, project=project, memory_type=memory_type,
                scope=scope, tags=tags, min_importance=min_importance,
            )
            return {
                "query": query or "",
                "weights": weights,
                "results": results[:limit],
                "total_candidates": len(results),
            }

        temporal_window = None
        lexical_query = query
        if temporal:
            try:
                temporal_window, lexical_query = analyze_query(query)
            except Exception:
                temporal_window, lexical_query = None, query

        # Query-aware arm weighting: identifier/short/literal queries lean on
        # the FTS arm, natural-language questions on the vector/graph arms.
        # Only inferred when the caller did not supply explicit boosts.
        if strategy_boosts is None and (fusion or graph_expand or temporal):
            strategy_boosts = infer_strategy_boosts(query)

        try:
            return self._search_inner(
                query=query, lexical_query=lexical_query, temporal_window=temporal_window,
                limit=limit, project=project, memory_type=memory_type, scope=scope,
                tags=tags, min_importance=min_importance, fusion=fusion,
                graph_expand=graph_expand, temporal=temporal,
                temporal_filter=bool(temporal_filter),
                strategy_boosts=strategy_boosts, vector_search_fn=vector_search_fn,
                now=now, weights=weights, track_access=track_access,
            )
        except Exception as exc:
            logger.debug("memory search failed; returning recent rows", exc_info=True)
            results = self._list_recent(
                limit, project=project, memory_type=memory_type,
                scope=scope, tags=tags, min_importance=min_importance,
            )
            return {
                "query": query,
                "weights": weights,
                "results": results[:limit],
                "total_candidates": len(results),
                "retrieval": {"mode": "fallback", "error": str(exc)},
            }

    def _search_inner(self, *, query, lexical_query, temporal_window, limit,
                      project, memory_type, scope, tags, min_importance,
                      fusion, graph_expand, temporal, strategy_boosts,
                      vector_search_fn, now, weights, track_access=True,
                      temporal_filter=False) -> Dict[str, Any]:
        advanced = bool(fusion or graph_expand or temporal)
        hard_window = temporal_window if temporal_filter else None
        fts_scores: Dict[str, float] = {}
        fts_query = _quote_fts(lexical_query, match_any=advanced)
        if fts_query:
            try:
                conditions, params = self._filter_sql(
                    project, memory_type, scope, tags, min_importance, alias="m"
                )
                time_sql, time_params = self._time_predicate("m", hard_window)
                if time_sql:
                    conditions.append(time_sql)
                    params = list(params) + time_params
                sql = (
                    "SELECT m.id, memories_fts.rank AS fts_rank "
                    "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                    "WHERE memories_fts MATCH ? AND " + " AND ".join(conditions)
                    + " ORDER BY memories_fts.rank LIMIT ?"
                )
                for row in self._execute(sql, (fts_query,) + tuple(params) + (limit * 4,)):
                    fts_scores[row["id"]] = float(row["fts_rank"] or 0.0)
            except sqlite3.OperationalError:
                logger.debug("FTS query failed; falling back to recent rows", exc_info=True)

        vec_distances: Dict[str, float] = {}
        vec_rows: List[Dict[str, Any]] = []
        vector_ids: set = set()
        if vector_search_fn is not None:
            try:
                raw = vector_search_fn(query=query, limit=limit * 4)
                hits = [
                    hit for hit in ((raw or {}).get("results", []) or [])
                    if isinstance(hit, dict)
                ]
                # Vector hits from a real index carry the store id, so the
                # memory row can be scored with its actual metadata.  Resolve
                # them in one query; unresolvable hits keep the content-key
                # path and the synthetic fallback row below.
                hit_ids = [str(hit.get("id")) for hit in hits if hit.get("id")]
                if hit_ids:
                    placeholders = ",".join("?" for _ in hit_ids)
                    try:
                        for row in self._execute(
                            f"SELECT id FROM memories WHERE id IN ({placeholders})",
                            tuple(hit_ids),
                        ):
                            vector_ids.add(row["id"])
                    except Exception:
                        logger.debug("vector id resolution failed (non-fatal)", exc_info=True)
                for hit in hits:
                    text = hit.get("text") or ""
                    if not text:
                        continue
                    key = _content_key(text)
                    hit_id = str(hit["id"]) if hit.get("id") else None
                    distance = hit.get("distance")
                    if distance is not None:
                        try:
                            distance_val = float(distance)
                        except (TypeError, ValueError):
                            distance_val = None
                        if distance_val is not None:
                            vec_distances[key] = distance_val
                            if hit_id in vector_ids:
                                vec_distances[hit_id] = distance_val
                    if hit_id not in vector_ids:
                        vec_rows.append(hit)
            except Exception:
                logger.debug("vector arm failed (non-fatal)", exc_info=True)

        # Ordered candidate list, not a set: iteration order decides tie-breaks
        # in the scoring pass below, and set order varies with PYTHONHASHSEED
        # across processes, which made otherwise identical searches return
        # different orderings run to run.  FTS order first (best BM25 first),
        # then vector hits, then the recency/importance backfill.
        candidate_ids: List[str] = []
        seen_candidates: set = set()
        for memory_id in list(fts_scores) + sorted(vector_ids):
            if memory_id not in seen_candidates:
                seen_candidates.add(memory_id)
                candidate_ids.append(memory_id)
        if len(candidate_ids) < limit * 2:
            conditions, params = self._filter_sql(
                project, memory_type, scope, tags, min_importance
            )
            time_sql, time_params = self._time_predicate("memories", hard_window)
            if time_sql:
                conditions.append(time_sql)
                params = list(params) + time_params
            sql = (
                "SELECT id FROM memories WHERE " + " AND ".join(conditions)
                + " ORDER BY importance DESC, created_at DESC LIMIT ?"
            )
            for row in self._execute(sql, tuple(params) + (limit * 4,)):
                if row["id"] not in seen_candidates:
                    seen_candidates.add(row["id"])
                    candidate_ids.append(row["id"])

        scored: List[Dict[str, Any]] = []
        seen_keys = set()
        for memory_id in candidate_ids:
            mem = self._fetch_memory(memory_id)
            if not mem:
                continue
            if project and mem.get("project") != project:
                continue
            if memory_type and mem.get("memory_type") != memory_type:
                continue
            if scope and mem.get("scope") != scope:
                continue
            if min_importance and float(mem.get("importance") or 0) < float(min_importance):
                continue
            if tags and not set(str(t) for t in tags).issubset(
                set(str(t) for t in mem.get("tags") or [])
            ):
                continue
            if hard_window is not None and not self._memory_in_window(mem, hard_window):
                continue
            key = _content_key(mem.get("content") or "")
            seen_keys.add(key)
            fts_norm = _normalize_fts(fts_scores.get(memory_id, 0.0))
            vec_distance = vec_distances.get(memory_id)
            if vec_distance is None:
                vec_distance = vec_distances.get(key, 1.0)
            vec_norm = _normalize_vec(vec_distance)
            rec_norm = _recency_score(
                _days_since(self._recency_reference(mem)), weights["half_life"]
            )
            acc_norm = _access_score(mem.get("access_count"))
            importance = _clamp_importance(mem.get("importance"))
            raw_score = (
                weights["w_fts"] * fts_norm
                + weights["w_vec"] * vec_norm
                + weights["w_recency"] * rec_norm
                + weights["w_access"] * acc_norm
            )
            score = raw_score * 0.85 + importance * 0.15
            scored.append({
                **mem,
                "score": round(score, 6),
                "fts_score": round(fts_norm, 3),
                "vec_score": round(vec_norm, 3),
                "recency_score": round(rec_norm, 3),
                "access_score": round(acc_norm, 3),
                # ``source`` is the retrieval-arm label on results, so keep the
                # record's own provenance type under its own key for B6.
                "source_type": _source_type(mem.get("source")),
                "source": "engine",
            })

        for hit in vec_rows:
            text = hit.get("text") or ""
            key = _content_key(text)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            distance = hit.get("distance", 1.0)
            vec_norm = _normalize_vec(distance)
            rec_norm = _recency_score(_days_since(hit.get("created_at")), weights["half_life"])
            raw_score = (
                weights["w_fts"] * 0.0
                + weights["w_vec"] * vec_norm
                + weights["w_recency"] * rec_norm
            )
            scored.append({
                "id": hit.get("id") or key,
                "content": text[:2000],
                "memory_type": "fact",
                "project": None,
                "scope": "global",
                "session_id": "",
                "tags": [],
                "importance": 0.5,
                "created_at": hit.get("created_at") or _now(),
                "updated_at": hit.get("created_at") or _now(),
                "access_count": 0,
                "proof_count": 1,
                "last_reinforced_at": None,
                "score": round(raw_score * 0.85 + 0.5 * 0.15, 6),
                "fts_score": 0.0,
                "vec_score": round(vec_norm, 3),
                "recency_score": round(rec_norm, 3),
                "access_score": 0.0,
                "source": "vector",
            })

        fusion_meta = None
        if advanced:
            scored, fusion_meta = self._refine_candidates(
                query=query, scored=scored, limit=limit, fusion=fusion,
                graph_expand=graph_expand, temporal=temporal,
                strategy_boosts=strategy_boosts, project=project,
                memory_type=memory_type, scope=scope, tags=tags,
                min_importance=min_importance, temporal_window=temporal_window,
                weights=weights, now=now, hard_window=hard_window,
            )

        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        top = scored[:limit]
        # B6: evidence quality is surfaced for attribution only.  It is
        # deliberately attached after ranking and never read by the scorer, so
        # existing memories keep exactly the ranking they had before.
        self._attach_evidence_attribution(top)
        if track_access:
            try:
                with self._write_lock:
                    for item in top:
                        if item.get("source") == "engine":
                            self._execute(
                                "UPDATE memories SET access_count = access_count + 1, "
                                "last_accessed_at = ? WHERE id = ?",
                                (_now(), item["id"]),
                            )
                    self._commit()
            except Exception:
                logger.debug("access-count update failed (non-fatal)", exc_info=True)

        envelope: Dict[str, Any] = {
            "query": query,
            "weights": weights,
            "results": top,
            "total_candidates": len(scored),
        }
        if fusion_meta is not None:
            envelope["retrieval"] = fusion_meta
        return envelope

    def _attach_evidence_attribution(self, entries: List[Dict[str, Any]]) -> None:
        """Add ``evidence_confidence`` / ``source_quality`` to result rows.

        Attribution is write-free and score-free by design: these are the
        per-result evidence signals from :mod:`agent.memory.evidence`, computed
        from proof count, source type, temporal consistency and the number of
        open ``contradicts`` links.  ``confidence`` (the stored per-record
        column) is left untouched, as is every field the scorer reads.
        """
        contradiction_counts: Dict[str, int] = {}
        engine_ids = [
            str(entry["id"]) for entry in entries
            if entry.get("id") and entry.get("source") == "engine"
        ]
        if engine_ids:
            try:
                placeholders = ",".join("?" for _ in engine_ids)
                for row in self._execute(
                    f"SELECT from_id, COUNT(*) AS c FROM memory_links "
                    f"WHERE relation = 'contradicts' AND from_id IN ({placeholders}) "
                    f"GROUP BY from_id",
                    tuple(engine_ids),
                ):
                    contradiction_counts[row["from_id"]] = int(row["c"] or 0)
            except Exception:
                logger.debug("evidence attribution scan failed", exc_info=True)
        for entry in entries:
            quality = evidence_source_quality(entry.get("source_type"))
            penalty = min(
                _EVIDENCE_CONTRADICTION_CAP,
                _EVIDENCE_CONTRADICTION_STEP * contradiction_counts.get(str(entry.get("id")), 0),
            )
            try:
                temporal_consistency = float(entry.get("temporal_score", 0.5))
            except (TypeError, ValueError):
                temporal_consistency = 0.5
            entry["source_quality"] = round(quality, 3)
            entry["evidence_confidence"] = round(
                evidence_confidence(
                    entry.get("proof_count", 1), quality, penalty, temporal_consistency
                ),
                3,
            )

    def _link_adjacency(self, seed_ids: List[str], max_nodes: int = 2000) -> Dict[str, List[Tuple[str, str]]]:
        if not seed_ids:
            return {}
        adjacency: Dict[str, List[Tuple[str, str]]] = {}
        frontier = list(seed_ids)
        seen = set(seed_ids)
        for _ in range(2):
            if not frontier or len(seen) >= max_nodes:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = self._execute(
                f"SELECT from_id, to_id, relation FROM memory_links "
                f"WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})",
                tuple(frontier) * 2,
            ).fetchall()
            next_frontier = []
            for row in rows:
                for src, dst in ((row["from_id"], row["to_id"]), (row["to_id"], row["from_id"])):
                    edges = adjacency.setdefault(src, [])
                    if (dst, row["relation"]) not in edges:
                        edges.append((dst, row["relation"]))
                    if dst not in seen:
                        seen.add(dst)
                        next_frontier.append(dst)
            frontier = next_frontier
        return adjacency

    def _refine_candidates(self, *, query, scored, limit, fusion, graph_expand,
                           temporal, strategy_boosts, project, memory_type,
                           scope, tags, min_importance, temporal_window, weights,
                           now=None, hard_window=None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        try:
            meta: Dict[str, Any] = {}
            by_id = {entry["id"]: entry for entry in scored}

            activations: Dict[str, float] = {}
            if graph_expand:
                seeds = [
                    entry["id"]
                    for entry in sorted(scored, key=lambda item: item.get("score", 0.0), reverse=True)[:10]
                    if entry.get("source") == "engine"
                ]
                adjacency = self._link_adjacency(seeds)
                activations = expand_links(
                    seeds, adjacency, max_hops=2, budget=max(limit * 2, 20)
                )
                for memory_id, activation in activations.items():
                    if memory_id in by_id:
                        continue
                    mem = self._fetch_memory(memory_id)
                    if not mem:
                        continue
                    if project and mem.get("project") != project:
                        continue
                    if memory_type and mem.get("memory_type") != memory_type:
                        continue
                    if scope and mem.get("scope") != scope:
                        continue
                    if min_importance and float(mem.get("importance") or 0) < float(min_importance):
                        continue
                    if tags and not set(str(t) for t in tags).issubset(
                        set(str(t) for t in mem.get("tags") or [])
                    ):
                        continue
                    if hard_window is not None and not self._memory_in_window(mem, hard_window):
                        continue
                    entry = {
                        **mem,
                        "score": 0.0,
                        "fts_score": 0.0,
                        "vec_score": 0.0,
                        "recency_score": 0.0,
                        "access_score": 0.0,
                        "source_type": _source_type(mem.get("source")),
                        "source": "engine",
                        "via": "graph",
                    }
                    scored.append(entry)
                    by_id[memory_id] = entry
                meta["graph_expanded"] = len(activations)

            proximity: Dict[str, float] = {}
            if temporal:
                window = temporal_window
                if window:
                    start, end = window
                    meta["temporal_window"] = [start.isoformat(), end.isoformat()]
                    for entry in scored:
                        # D4: rank by when the event happened, not when the
                        # agent happened to write it down.  Rows without event
                        # time fall back to created_at, exactly as before.
                        event_start, _event_end = self._event_bounds(entry)
                        target = event_start or _parse_ts(entry.get("created_at"))
                        proximity[entry["id"]] = temporal_proximity(target, start, end)
                else:
                    meta["temporal_window"] = None

            if not fusion:
                for entry in scored:
                    entry["score"] = round(combined_score(
                        entry.get("score") or 0.0,
                        graph=activations.get(entry["id"], 0.5) if activations else 0.5,
                        recency=proximity.get(entry["id"], 0.5) if proximity else 0.5,
                    ), 6)
                    if entry["id"] in activations:
                        entry["graph_score"] = round(activations[entry["id"]], 3)
                    if entry["id"] in proximity:
                        entry["temporal_score"] = round(proximity[entry["id"]], 3)
                meta["mode"] = "boosted"
                return scored, meta

            arms: List[Tuple[str, List[str]]] = []
            fts_ranked = [
                entry["id"] for entry in sorted(
                    scored, key=lambda item: item.get("fts_score", 0.0), reverse=True
                ) if entry.get("fts_score", 0.0) > 0
            ]
            if fts_ranked:
                arms.append(("fts", fts_ranked))
            vec_ranked = [
                entry["id"] for entry in sorted(
                    scored, key=lambda item: item.get("vec_score", 0.0), reverse=True
                ) if entry.get("vec_score", 0.0) > 0
            ]
            if vec_ranked:
                arms.append(("vec", vec_ranked))
            if activations:
                arms.append(("graph", [
                    mid for mid, _ in sorted(activations.items(), key=lambda kv: kv[1], reverse=True)
                ]))
            if proximity:
                temporal_ranked = [
                    mid for mid, value in sorted(proximity.items(), key=lambda kv: kv[1], reverse=True)
                    if value > 0
                ]
                if temporal_ranked:
                    arms.append(("temporal", temporal_ranked))

            if not arms:
                meta["mode"] = "rrf-empty"
                return scored, meta

            fused = reciprocal_rank_fusion(arms)
            meta["mode"] = "rrf"
            meta["arms"] = [name for name, _ in arms]

            # Only boost arms that actually produced candidates.  When no
            # semantic arm is present, a question classification has nothing to
            # lean on, so drop its lexical damping entirely rather than
            # degrading an FTS-only install.  Lookup classifications keep their
            # lexical boost regardless.
            lexical_classified = (strategy_boosts or {}).get("fts") in ("high", "medium")
            effective_boosts = strategy_boosts
            if strategy_boosts:
                available = {name for name, _ in arms}
                effective_boosts = {
                    arm: level
                    for arm, level in strategy_boosts.items()
                    if arm in available
                }
                if not (available & {"vec", "graph"}) and not lexical_classified:
                    effective_boosts = None
                if not effective_boosts:
                    effective_boosts = None

            current = now or datetime.now(timezone.utc).replace(tzinfo=None)
            results: List[Dict[str, Any]] = []
            for item in fused:
                entry = by_id.get(item["id"])
                if entry is None:
                    continue
                base = boosted_rrf_score(
                    item["rrf_score"], item["source_ranks"], effective_boosts or {}
                )
                event_start, event_end = self._event_bounds(entry)
                created = _parse_ts(entry.get("created_at"))
                recency = recency_for_range(event_start, event_end, created, current)
                if entry["id"] in proximity:
                    recency = proximity[entry["id"]]
                entry["score"] = round(combined_score(
                    base,
                    recency=recency,
                    importance=_clamp_importance(entry.get("importance")),
                    proof=proof_norm(entry.get("proof_count", 1)),
                    graph=activations.get(entry["id"], 0.5) if activations else 0.5,
                ), 6)
                entry["rrf_score"] = round(item["rrf_score"], 6)
                entry["rrf_rank"] = item["rrf_rank"]
                entry["source_ranks"] = item["source_ranks"]
                if entry["id"] in activations:
                    entry["graph_score"] = round(activations[entry["id"]], 3)
                if entry["id"] in proximity:
                    entry["temporal_score"] = round(proximity[entry["id"]], 3)
                results.append(entry)

            fused_ids = {item["id"] for item in fused}
            for entry in scored:
                if entry["id"] not in fused_ids:
                    entry["score"] = round((entry.get("score") or 0.0) * 0.01, 6)
                    results.append(entry)

            # Bounded vector influence: RRF lets convergent evidence
            # accumulate across arms, but a memory surfaced by the vector arm
            # alone must never dominate the fused ranking.  Keep at most `cap`
            # vector-only entries in the ranked prefix and defer the rest to
            # the tail (still retrievable, just not promoted).  The caller
            # re-sorts by score afterwards, so deferred entries are also
            # pushed below the lowest kept score to keep the split stable.
            cap = max(1, int(len(results) * self._vector_max_share))
            kept, deferred, vec_only_seen = [], [], 0
            for entry in results:
                ranks = entry.get("source_ranks") or {}
                is_vec_only = bool(ranks) and set(ranks.keys()) == {"vec_rank"}
                if is_vec_only:
                    vec_only_seen += 1
                    if vec_only_seen > cap:
                        deferred.append(entry)
                        continue
                kept.append(entry)
            if deferred:
                floor = min((entry.get("score", 0.0) for entry in kept), default=0.0)
                for entry in deferred:
                    entry["score"] = round(min(entry.get("score", 0.0), floor * 0.5), 6)
            results = kept + deferred

            # Lexical anchoring: for lookup-shaped queries, pin the strongest
            # BM25 candidate at rank 1.  Plain RRF can tie it with a graph or
            # vector arm's favourite, and the multiplicative signals can then
            # flip the order; the anchor guarantees exact matches are never
            # displaced.  The other arms still order everything below it.
            # Question-like queries are excluded: their top lexical hit is
            # often not the intended memory at all.
            if lexical_classified and effective_boosts and results:
                fts_scores = sorted(
                    (entry.get("fts_score", 0.0) for entry in results), reverse=True
                )
                best_fts_score = fts_scores[0] if fts_scores else 0.0
                runner_up = fts_scores[1] if len(fts_scores) > 1 else 0.0
                clear_leader = (
                    best_fts_score > 0
                    and best_fts_score >= runner_up + LEXICAL_ANCHOR_MARGIN
                )
                if best_fts_score >= LEXICAL_ANCHOR_MIN_FTS or clear_leader:
                    best_fts = max(results, key=lambda entry: entry.get("fts_score", 0.0))
                    top_score = max(entry.get("score", 0.0) for entry in results)
                    if results[0] is not best_fts:
                        best_fts["score"] = round(
                            max(best_fts.get("score", 0.0), top_score * 1.02), 6
                        )
                        results = [best_fts] + [
                            entry for entry in results if entry is not best_fts
                        ]
                    best_fts["anchored"] = True
                    meta["anchored"] = best_fts.get("id")
            return results, meta
        except Exception as exc:
            logger.debug("advanced retrieval failed; falling back to base ranking", exc_info=True)
            return scored, {"mode": "fallback", "error": str(exc)}

    # -- Introspection ------------------------------------------------------

    def get_conflicts(self, project: Optional[str] = None, limit: int = 50,
                      include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Contradiction links, open by default.

        A link dismissed or resolved through :meth:`resolve_conflict` keeps its
        row and its history; it simply stops appearing here unless
        ``include_resolved`` is set.
        """
        try:
            sql = (
                "SELECT ml.from_id, m1.content AS from_content, "
                "ml.to_id, m2.content AS to_content, ml.relation, "
                "ml.metadata AS link_metadata, ml.created_at "
                "FROM memory_links ml "
                "JOIN memories m1 ON m1.id = ml.from_id "
                "JOIN memories m2 ON m2.id = ml.to_id "
                "WHERE ml.relation = 'contradicts'"
            )
            params: list = []
            if project:
                sql += " AND (m1.project = ? OR m2.project = ?)"
                params.extend([project, project])
            # Over-fetch so filtering resolved links in Python still fills the
            # caller's limit; the scan stays bounded at 4x.
            sql += " ORDER BY ml.created_at DESC LIMIT ?"
            params.append(int(limit) * 4)
            results = []
            for row in self._execute(sql, tuple(params)):
                metadata: Dict[str, Any] = {}
                try:
                    parsed = json.loads(row["link_metadata"] or "{}")
                    if isinstance(parsed, dict):
                        metadata = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
                state = str(metadata.get("state") or "open")
                if state != "open" and not include_resolved:
                    continue
                try:
                    confidence = float(metadata.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                results.append({
                    "memory_id": row["from_id"],
                    "content": row["from_content"],
                    "related_id": row["to_id"],
                    "related_content": row["to_content"],
                    "relation": row["relation"],
                    "confidence": confidence,
                    "state": state,
                    "resolved_at": metadata.get("resolved_at"),
                    "resolution_note": metadata.get("note"),
                    "created_at": row["created_at"],
                })
                if len(results) >= int(limit):
                    break
            return results
        except Exception:
            logger.debug("conflict query failed", exc_info=True)
            return []

    def resolve_conflict(self, from_id: str, to_id: str, *,
                         resolution: str = "dismissed",
                         note: Optional[str] = None) -> Dict[str, Any]:
        """Dismiss or resolve a contradiction link without deleting anything.

        The link row stays (and so does both memories); only its metadata gains
        ``state``/``resolved_at``/``note``.  History is preserved: pass
        ``resolution="open"`` to reopen a dismissed link.
        """
        allowed = {"open", "dismissed", "resolved"}
        state = str(resolution or "dismissed").lower()
        if state not in allowed:
            return {"success": False, "error": f"resolution must be one of {sorted(allowed)}"}
        try:
            with self._write_lock:
                row = self._execute(
                    "SELECT metadata FROM memory_links WHERE from_id = ? AND to_id = ? "
                    "AND relation = 'contradicts'",
                    (from_id, to_id),
                ).fetchone()
                if not row:
                    return {"success": False, "error": "No contradiction link found"}
                try:
                    metadata = json.loads(row["metadata"] or "{}")
                    if not isinstance(metadata, dict):
                        metadata = {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                metadata["state"] = state
                metadata["resolved_at"] = _now() if state != "open" else None
                if note:
                    metadata["note"] = str(note)
                self._execute(
                    "UPDATE memory_links SET metadata = ? WHERE from_id = ? AND to_id = ? "
                    "AND relation = 'contradicts'",
                    (json.dumps(metadata, ensure_ascii=False), from_id, to_id),
                )
                self._commit()
            return {
                "success": True,
                "from": from_id,
                "to": to_id,
                "relation": "contradicts",
                "state": state,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def related(self, memory_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            rows = self._execute(
                "SELECT m.* FROM memories m "
                "JOIN memory_links l ON (l.to_id = m.id OR l.from_id = m.id) "
                "WHERE (l.from_id = ? OR l.to_id = ?) AND m.id != ? AND m.status = 'active' "
                "ORDER BY l.created_at DESC LIMIT ?",
                (memory_id, memory_id, memory_id, int(limit)),
            ).fetchall()
            seen = set()
            results = []
            for row in rows:
                item = self._row_to_dict(row)
                if item["id"] in seen:
                    continue
                seen.add(item["id"])
                results.append(item)
            return results
        except Exception:
            logger.debug("related query failed", exc_info=True)
            return []

    def timeline(self, project: Optional[str] = None, start=None, end=None,
                 limit: int = 20) -> List[Dict[str, Any]]:
        try:
            sql = "SELECT * FROM memories WHERE 1 = 1"
            params: list = []
            if project:
                sql += " AND project = ?"
                params.append(project)
            if start:
                sql += " AND created_at >= ?"
                params.append(str(start))
            if end:
                sql += " AND created_at <= ?"
                params.append(str(end))
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(int(limit))
            return [self._row_to_dict(row) for row in self._execute(sql, tuple(params))]
        except Exception:
            logger.debug("timeline query failed", exc_info=True)
            return []

    def topics(self, project: Optional[str] = None, limit: int = 50,
               scan_limit: int = _TOPIC_SCAN_LIMIT) -> List[Dict[str, Any]]:
        """Discover topics from content, not from the ``memory_type`` enum.

        Bounded and deterministic: a fixed-size recent window of active rows,
        rule-based tokenisation, a local stopword list, and a total ordering on
        ``(-document_frequency, token)``.  No model calls, no external corpus.

        ``count`` is document frequency (how many memories mention the term),
        ``latest`` is the newest memory it appears in.  A term has to appear in
        at least two memories once the window holds four or more, so single
        one-off words do not masquerade as topics.
        """
        try:
            try:
                window = max(1, min(_TOPIC_SCAN_LIMIT, int(scan_limit)))
            except (TypeError, ValueError):
                window = _TOPIC_SCAN_LIMIT
            sql = (
                "SELECT content, created_at FROM memories WHERE status = 'active'"
            )
            params: list = []
            if project:
                sql += " AND project = ?"
                params.append(project)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(window)
            rows = self._execute(sql, tuple(params)).fetchall()

            counts: Dict[str, int] = {}
            latest: Dict[str, str] = {}
            for row in rows:
                created = row["created_at"]
                for token in set(_topic_tokens(row["content"])):
                    counts[token] = counts.get(token, 0) + 1
                    if created and (token not in latest or created > latest[token]):
                        latest[token] = created

            min_count = 2 if len(rows) >= 4 else 1
            ranked = sorted(
                ((token, count) for token, count in counts.items() if count >= min_count),
                key=lambda item: (-item[1], item[0]),
            )
            return [
                {"topic": token, "count": count, "latest": latest.get(token),
                 "kind": "term"}
                for token, count in ranked[:max(1, int(limit))]
            ]
        except Exception:
            logger.debug("topics query failed", exc_info=True)
            return []

    def diagnostics(self) -> Dict[str, Any]:
        """Read-only health diagnostics: integrity, links, FTS, growth.

        Unlike :meth:`health` this reports *problems* rather than totals, and it
        never repairs anything — no FTS rebuild, no link cleanup, no writes of
        any kind.  Safe to call from a status command on a live store.
        """
        def _scalar(sql: str, params: tuple = (), default: int = 0) -> int:
            try:
                row = self._execute(sql, params).fetchone()
                if row is None or row[0] is None:
                    return default
                return int(row[0])
            except Exception:
                return default

        try:
            result: Dict[str, Any] = {
                "ok": True,
                "read_only": True,
                "schema_version": _SCHEMA_VERSION,
                "generated_at": _now(),
            }

            integrity_ok = True
            integrity_text = "ok"
            try:
                row = self._execute("PRAGMA quick_check").fetchone()
                integrity_text = str(row[0]) if row else "unknown"
                integrity_ok = integrity_text.lower() == "ok"
            except sqlite3.DatabaseError as exc:
                integrity_ok = False
                integrity_text = str(exc)
            result["integrity"] = {"ok": integrity_ok, "result": integrity_text}

            orphan_links = _scalar(
                "SELECT COUNT(*) FROM memory_links l "
                "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = l.from_id) "
                "OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = l.to_id)"
            )
            orphan_samples: List[Dict[str, str]] = []
            if orphan_links:
                try:
                    for row in self._execute(
                        "SELECT from_id, to_id, relation FROM memory_links l "
                        "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = l.from_id) "
                        "OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = l.to_id) "
                        "LIMIT 10"
                    ):
                        orphan_samples.append({
                            "from_id": row["from_id"],
                            "to_id": row["to_id"],
                            "relation": row["relation"],
                        })
                except Exception:
                    logger.debug("orphan link sample failed", exc_info=True)
            result["orphan_links"] = {"count": orphan_links, "sample": orphan_samples}

            memory_rows = _scalar("SELECT COUNT(*) FROM memories")
            fts_rows = _scalar("SELECT COUNT(*) FROM memories_fts_docsize")
            orphan_fts = _scalar(
                "SELECT COUNT(*) FROM memories_fts_docsize d "
                "WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.rowid = d.id)"
            )
            unindexed = _scalar(
                "SELECT COUNT(*) FROM memories m "
                "WHERE NOT EXISTS (SELECT 1 FROM memories_fts_docsize d WHERE d.id = m.rowid)"
            )
            result["stale_fts"] = {
                "ok": memory_rows == fts_rows and orphan_fts == 0 and unindexed == 0,
                "memory_rows": memory_rows,
                "fts_rows": fts_rows,
                "orphan_fts_rows": orphan_fts,
                "unindexed_rows": unindexed,
            }

            now = datetime.now(timezone.utc).replace(tzinfo=None)
            week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            day_ago = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            growth: Dict[str, Dict[str, int]] = {}
            try:
                for row in self._execute(
                    "SELECT COALESCE(project, '<none>') AS project, COUNT(*) AS total, "
                    "SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS last_7d, "
                    "SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS last_24h "
                    "FROM memories GROUP BY project ORDER BY total DESC LIMIT 50",
                    (week_ago, day_ago),
                ):
                    growth[row["project"]] = {
                        "total": int(row["total"] or 0),
                        "last_7d": int(row["last_7d"] or 0),
                        "last_24h": int(row["last_24h"] or 0),
                    }
            except Exception:
                logger.debug("growth query failed", exc_info=True)
            result["per_project_growth"] = growth

            columns = {
                row[1] for row in self._execute("PRAGMA table_info(memories)")
            }
            result["schema"] = {
                "occurred_start": "occurred_start" in columns,
                "occurred_end": "occurred_end" in columns,
                "confidence": "confidence" in columns,
            }
            result["ok"] = bool(
                integrity_ok
                and orphan_links == 0
                and result["stale_fts"]["ok"]
            )
            return result
        except Exception as exc:
            logger.debug("memory diagnostics failed", exc_info=True)
            return {"ok": False, "read_only": True, "error": str(exc)}

    def health(self) -> Dict[str, Any]:
        def _count(sql: str, params: tuple = ()) -> int:
            try:
                row = self._execute(sql, params).fetchone()
                return int(row[0]) if row and row[0] is not None else 0
            except Exception:
                return 0

        try:
            by_type = {}
            for row in self._execute(
                "SELECT memory_type, COUNT(*) AS count FROM memories GROUP BY memory_type"
            ):
                by_type[row["memory_type"]] = row["count"]
            by_project = {}
            for row in self._execute(
                "SELECT COALESCE(project, '<none>') AS project, COUNT(*) AS count "
                "FROM memories GROUP BY project"
            ):
                by_project[row["project"]] = row["count"]
            fts_ok = True
            try:
                self._execute("SELECT COUNT(*) FROM memories_fts_docsize").fetchone()
            except sqlite3.DatabaseError:
                fts_ok = False
            total = _count("SELECT COUNT(*) FROM memories")
            superseded = _count("SELECT COUNT(*) FROM memories WHERE status != 'active'")
            db_size = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0
            wal_size = 0
            if os.path.exists(self._db_path + "-wal"):
                wal_size = os.path.getsize(self._db_path + "-wal")
            oldest = self._execute("SELECT MIN(created_at) FROM memories").fetchone()[0]
            newest = self._execute("SELECT MAX(created_at) FROM memories").fetchone()[0]
            return {
                "total_memories": total,
                "active_memories": total - superseded,
                "total_links": _count("SELECT COUNT(*) FROM memory_links"),
                "total_conflicts": _count(
                    "SELECT COUNT(*) FROM memory_links WHERE relation = 'contradicts'"
                ),
                "total_superseded": _count(
                    "SELECT COUNT(*) FROM memory_links WHERE relation = 'supersedes'"
                ),
                "duplicate_content_groups": _count(
                    "SELECT COUNT(*) FROM (SELECT content_hash FROM memories "
                    "WHERE content_hash IS NOT NULL GROUP BY content_hash HAVING COUNT(*) > 1)"
                ),
                "oldest_memory": oldest,
                "newest_memory": newest,
                "memories_by_type": by_type,
                "memories_by_project": by_project,
                "fts_ok": fts_ok,
                "embeddings_indexed": self.embedding_count(),
                "embeddings_missing": self.missing_embedding_count(None),
                "db_size_bytes": db_size,
                "wal_size_bytes": wal_size,
                "schema_version": _SCHEMA_VERSION,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # -- Teardown -----------------------------------------------------------

    def checkpoint(self) -> None:
        try:
            with self._write_lock:
                self._execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._commit()
        except Exception:
            logger.debug("memory checkpoint failed", exc_info=True)

    def close(self) -> None:
        self.checkpoint()
        try:
            with self._write_lock:
                self.db.close()
        except Exception:
            logger.debug("memory close failed", exc_info=True)
