"""Rekal — standalone local SQLite memory engine for Hermes Agent.

Adapted from OpenCode's MemPalace Rekal engine. Uses only Python stdlib
(sqlite3, hashlib, json, uuid). No external dependencies.

Provides:
  - FTS5 BM25 + recency + access-frequency + importance hybrid search
  - Memory lifecycle: store, update, supersede, delete, link
  - Content-dedup via SHA-256 hashing
  - Conflict detection
  - Session management: init, ingest_turns
  - Graph traversal: related, similar, timeline, topics
  - TTL-aware LRU cache for hot memories
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1

_DEFAULT_SEARCH_LIMIT = 10
_DEFAULT_MAX_SEARCH = 50
_DEFAULT_CACHE_TTL = 300
_DEFAULT_CACHE_MAXSIZE = 5000
_DEFAULT_RETRY_DELAY = 0.05
_DEFAULT_RETRY_ATTEMPTS = 3

_FTS_TABLE = "memories_fts"
_MEMORY_TABLE = "memories"
_LINKS_TABLE = "memory_links"
_SESSIONS_TABLE = "sessions"
_CONFIG_TABLE = "rekal_config"

_SEARCH_WEIGHTS = {
    "bm25": 0.5,
    "recency": 0.2,
    "access_frequency": 0.15,
    "importance": 0.15,
}

# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {_MEMORY_TABLE} (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    memory_type     TEXT NOT NULL DEFAULT 'fact',
    tags            TEXT NOT NULL DEFAULT '[]',
    importance      REAL NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    metadata        TEXT NOT NULL DEFAULT '{{}}',
    superseded_by   TEXT,
    superseded      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    accessed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS {_LINKS_TABLE} (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES {_MEMORY_TABLE}(id) ON DELETE CASCADE,
    target_id       TEXT NOT NULL REFERENCES {_MEMORY_TABLE}(id) ON DELETE CASCADE,
    relation        TEXT NOT NULL DEFAULT 'related_to',
    metadata        TEXT NOT NULL DEFAULT '{{}}',
    created_at      TEXT NOT NULL,
    UNIQUE(source_id, target_id, relation)
);

CREATE TABLE IF NOT EXISTS {_SESSIONS_TABLE} (
    id              TEXT PRIMARY KEY,
    metadata        TEXT NOT NULL DEFAULT '{{}}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS {_CONFIG_TABLE} (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);

-- FTS5 for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
    content,
    content={_MEMORY_TABLE},
    content_rowid='rowid',
    tokenize='porter unicode61 remove_diacritics 1'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON {_MEMORY_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON {_MEMORY_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, content) VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON {_MEMORY_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO {_FTS_TABLE}(rowid, content) VALUES (new.rowid, new.content);
END;

-- Indices
CREATE INDEX IF NOT EXISTS idx_memories_type ON {_MEMORY_TABLE}(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON {_MEMORY_TABLE}(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON {_MEMORY_TABLE}(updated_at);
CREATE INDEX IF NOT EXISTS idx_links_source ON {_LINKS_TABLE}(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON {_LINKS_TABLE}(target_id);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize_fts(text: str) -> str:
    """Prepare text for FTS5 query — escape special chars."""
    if not text:
        return ""
    result = []
    for ch in text:
        if ch in ('"', "'"):
            result.append(" ")
        elif ch in ('(', ')', '*', '^', '-', '+', '~', 'AND', 'OR', 'NOT'):
            result.append(" ")
        else:
            result.append(ch)
    return "".join(result).strip()


def _quote_fts(term: str) -> str:
    """Quote a single term for FTS5 exact matching."""
    escaped = term.replace('"', '""')
    return f'"{escaped}"'


def _now_ts() -> float:
    return time.time()


def _days_since(iso_timestamp: str) -> float:
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return max(0.0, (_now_ts() - dt.timestamp()) / 86400.0)
    except Exception:
        return 999.0


def _recency_score(iso_timestamp: str, half_life_days: float = 7.0) -> float:
    return math.exp(-_days_since(iso_timestamp) / half_life_days)


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


def _tagify(value: Any) -> str:
    tags = _list_field(value)
    return json.dumps(tags, ensure_ascii=False)


# ---------------------------------------------------------------------------
# TTL Cache
# ---------------------------------------------------------------------------

class _TTLCache:
    """Simple TTL-aware LRU cache."""

    def __init__(self, ttl: float = _DEFAULT_CACHE_TTL, maxsize: int = _DEFAULT_CACHE_MAXSIZE):
        self._ttl = ttl
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._data: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires, value = entry
            if _now_ts() > expires:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._data) >= self._maxsize:
                try:
                    oldest = min(self._data.keys(), key=lambda k: self._data[k][0])
                    del self._data[oldest]
                except (ValueError, KeyError):
                    pass
            self._data[key] = (_now_ts() + self._ttl, value)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# ---------------------------------------------------------------------------
# RekalEngine
# ---------------------------------------------------------------------------

class RekalEngine:
    """Standalone local memory engine backed by SQLite + FTS5.

    All data is stored under ``data_dir`` as ``rekal.db`` (with WAL).
    Thread-safe via per-connection thread-local storage + write lock.
    """

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(self._data_dir / "rekal.db")
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._cache = _TTLCache()
        self._closed = False
        self._weights = dict(_SEARCH_WEIGHTS)
        self._run_migrations()

    # -- Connection management ---------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self._get_conn()
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                for attempt in range(_DEFAULT_RETRY_ATTEMPTS):
                    time.sleep(_DEFAULT_RETRY_DELAY * (attempt + 1))
                    try:
                        return conn.execute(sql, params)
                    except sqlite3.OperationalError:
                        continue
            raise

    def _executemany(self, sql: str, params_list: List[tuple]) -> None:
        conn = self._get_conn()
        with self._write_lock:
            try:
                conn.executemany(sql, params_list)
                conn.commit()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    for attempt in range(_DEFAULT_RETRY_ATTEMPTS):
                        time.sleep(_DEFAULT_RETRY_DELAY * (attempt + 1))
                        try:
                            conn.executemany(sql, params_list)
                            conn.commit()
                            return
                        except sqlite3.OperationalError:
                            continue
                raise

    def _run_migrations(self) -> None:
        with self._write_lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            self._set_config("schema_version", str(_SCHEMA_VERSION))
            conn.commit()

    def _set_config(self, key: str, value: str) -> None:
        self._execute(
            f"INSERT OR REPLACE INTO {_CONFIG_TABLE}(key, value) VALUES (?, ?)",
            (key, value),
        )

    def close(self) -> None:
        self._closed = True
        self._cache.clear()
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def checkpoint(self) -> None:
        """Force WAL checkpoint to reclaim space."""
        try:
            conn = self._get_conn()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as e:
            logger.debug("Rekal checkpoint failed: %s", e)

    # -- Cache helpers -----------------------------------------------------

    def _cache_key(self, prefix: str, *parts: str) -> str:
        return f"{prefix}:{':'.join(parts)}"

    def _cached_search(self, query: str, limit: int) -> Optional[List[Dict[str, Any]]]:
        return self._cache.get(self._cache_key("search", query, str(limit)))

    def _store_search_cache(self, query: str, limit: int, results: List[Dict[str, Any]]) -> None:
        self._cache.set(self._cache_key("search", query, str(limit)), results)

    # -- CRUD operations ---------------------------------------------------

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        cached = self._cache.get(self._cache_key("mem", memory_id))
        if cached is not None:
            return cached
        row = self._execute(
            f"SELECT * FROM {_MEMORY_TABLE} WHERE id = ? AND superseded = 0",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        mem = self._row_to_dict(row)
        self._execute(
            f"UPDATE {_MEMORY_TABLE} SET access_count = access_count + 1, accessed_at = ? WHERE id = ?",
            (_now(), memory_id),
        )
        self._commit()
        self._cache.set(self._cache_key("mem", memory_id), mem)
        return mem

    def store(self, content: str, memory_type: str = "fact",
              tags: Optional[List[str]] = None,
              importance: float = 1.0,
              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("content is required")
        c_hash = _content_hash(content)
        now = _now()
        memory_id = _new_id()
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        self._execute(
            f"""INSERT OR IGNORE INTO {_MEMORY_TABLE}
                (id, content, content_hash, memory_type, tags, importance, metadata, created_at, updated_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, content, c_hash, memory_type, tags_json, importance, meta_json, now, now, now),
        )
        self._commit()
        mem = self._row_to_dict(
            self._execute(f"SELECT * FROM {_MEMORY_TABLE} WHERE id = ?", (memory_id,)).fetchone()
        )
        self._cache.set(self._cache_key("mem", memory_id), mem)
        return mem

    def batch_store(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now = _now()
        rows = []
        for item in items:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            c_hash = _content_hash(content)
            memory_id = _new_id()
            rows.append((
                memory_id, content, c_hash,
                item.get("memory_type", "fact"),
                _tagify(item.get("tags", [])),
                float(item.get("importance", 1.0)),
                json.dumps(item.get("metadata", {}), ensure_ascii=False),
                now, now, now,
            ))
        if not rows:
            return []
        self._executemany(
            f"""INSERT INTO {_MEMORY_TABLE}
                (id, content, content_hash, memory_type, tags, importance, metadata, created_at, updated_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        results = []
        for row in rows:
            results.append(self._row_to_dict(
                self._execute(f"SELECT * FROM {_MEMORY_TABLE} WHERE id = ?", (row[0],)).fetchone()
            ))
        for r in results:
            self._cache.set(self._cache_key("mem", r["id"]), r)
        return results

    def update(self, memory_id: str, content: Optional[str] = None,
               memory_type: Optional[str] = None, tags: Optional[List[str]] = None,
               importance: Optional[float] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        existing = self._execute(
            f"SELECT * FROM {_MEMORY_TABLE} WHERE id = ? AND superseded = 0",
            (memory_id,),
        ).fetchone()
        if existing is None:
            return None
        now = _now()
        updates = {"updated_at": now, "accessed_at": now}
        if content is not None:
            updates["content"] = content.strip()
            updates["content_hash"] = _content_hash(content.strip())
        if memory_type is not None:
            updates["memory_type"] = memory_type
        if tags is not None:
            updates["tags"] = json.dumps(tags, ensure_ascii=False)
        if importance is not None:
            updates["importance"] = importance
        if metadata is not None:
            updates["metadata"] = json.dumps(metadata, ensure_ascii=False)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [memory_id]
        self._execute(
            f"UPDATE {_MEMORY_TABLE} SET {set_clause} WHERE id = ?",
            tuple(values),
        )
        self._commit()
        self._cache.invalidate(self._cache_key("mem", memory_id))
        self._cache.invalidate(self._cache_key("search", "*", "*"))
        return self.get(memory_id)

    def supersede(self, memory_id: str, content: str,
                  memory_type: Optional[str] = None,
                  tags: Optional[List[str]] = None,
                  importance: Optional[float] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Mark memory as superseded and create a new version."""
        existing = self._execute(
            f"SELECT * FROM {_MEMORY_TABLE} WHERE id = ? AND superseded = 0",
            (memory_id,),
        ).fetchone()
        if existing is None:
            return None
        now = _now()
        new_id = _new_id()
        new_type = memory_type or existing["memory_type"]
        new_tags = json.dumps(tags or json.loads(existing["tags"]), ensure_ascii=False)
        new_importance = importance if importance is not None else existing["importance"]
        new_meta = json.dumps(metadata or json.loads(existing["metadata"]), ensure_ascii=False)
        c_hash = _content_hash(content.strip())

        with self._write_lock:
            self._execute(
                f"UPDATE {_MEMORY_TABLE} SET superseded = 1, superseded_by = ?, updated_at = ? WHERE id = ?",
                (new_id, now, memory_id),
            )
            self._execute(
                f"""INSERT INTO {_MEMORY_TABLE}
                    (id, content, content_hash, memory_type, tags, importance, metadata, created_at, updated_at, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_id, content.strip(), c_hash, new_type, new_tags, new_importance, new_meta, now, now, now),
            )
            self._commit()

        self._cache.invalidate(self._cache_key("mem", memory_id))
        new_mem = self.get(new_id)
        return new_mem

    def delete(self, memory_id: str) -> bool:
        self._execute(
            f"DELETE FROM {_MEMORY_TABLE} WHERE id = ?",
            (memory_id,),
        )
        self._commit()
        self._cache.invalidate(self._cache_key("mem", memory_id))
        self._cache.invalidate(self._cache_key("search", "*", "*"))
        return True

    # -- Linking -----------------------------------------------------------

    def link(self, source_id: str, target_id: str,
             relation: str = "related_to") -> bool:
        source = self._execute(
            f"SELECT 1 FROM {_MEMORY_TABLE} WHERE id = ?", (source_id,)
        ).fetchone()
        target = self._execute(
            f"SELECT 1 FROM {_MEMORY_TABLE} WHERE id = ?", (target_id,)
        ).fetchone()
        if not source or not target:
            return False
        now = _now()
        link_id = _new_id()
        try:
            self._execute(
                f"""INSERT OR IGNORE INTO {_LINKS_TABLE}
                    (id, source_id, target_id, relation, created_at)
                    VALUES (?, ?, ?, ?, ?)""",
                (link_id, source_id, target_id, relation, now),
            )
            self._commit()
            return True
        except sqlite3.IntegrityError:
            return False

    # -- Query helpers -----------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        if row is None:
            return {}
        d = dict(row)
        for field in ("tags", "metadata"):
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def _fetch_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        row = self._execute(
            f"SELECT * FROM {_MEMORY_TABLE} WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def _commit(self) -> None:
        try:
            self._get_conn().commit()
        except Exception:
            pass

    # -- Search ------------------------------------------------------------

    def search(self, query: str, limit: int = _DEFAULT_SEARCH_LIMIT,
               memory_type: Optional[str] = None,
               tags: Optional[List[str]] = None,
               min_importance: float = 0.0) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return self._list_recent(limit, memory_type)

        # Check cache
        cache_key = f"search:{query}:{limit}:{memory_type}:{min_importance}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        norm = _normalize_fts(query)
        if not norm:
            return self._list_recent(limit, memory_type)

        # Use OR disjunction with BM25 ranking for broader recall.
        # AND by default would miss partial matches, which is worse
        # for memory retrieval where we want ANY relevant memory.
        or_query = " OR ".join(_quote_fts(t) for t in norm.split() if t)

        fts_sql = f"""
            SELECT rowid, bm25({_FTS_TABLE}, 0, 1) AS bm25_score
            FROM {_FTS_TABLE}
            WHERE {_FTS_TABLE} MATCH ?
            ORDER BY bm25({_FTS_TABLE}, 0, 1)
            LIMIT ?
        """
        try:
            fts_rows = self._execute(fts_sql, (or_query, limit * 2)).fetchall()
        except sqlite3.OperationalError:
            return self._list_recent(limit, memory_type)

        fts_rowids = {r["rowid"] for r in fts_rows}
        if not fts_rowids:
            return self._list_recent(limit, memory_type)

        # Build bm25 score map (normalize to 0-1)
        bm25_values = [r["bm25_score"] for r in fts_rows if r["bm25_score"] is not None]
        min_bm25 = min(bm25_values) if bm25_values else 0
        max_bm25 = max(bm25_values) if bm25_values else 1
        bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        bm25_scores = {
            r["rowid"]: 1.0 - ((r["bm25_score"] - min_bm25) / bm25_range)
            for r in fts_rows if r["bm25_score"] is not None
        }

        # Fetch full memory records with conditions
        conditions = [f"rowid IN ({','.join(str(r) for r in fts_rowids)})", "superseded = 0"]
        params: list = []
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if min_importance > 0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        # Tag filtering (if tags specified)
        tag_joins = ""
        if tags:
            for i, tag in enumerate(tags):
                alias = f"t{i}"
                tag_joins += f"""
                    CROSS JOIN json_each({_MEMORY_TABLE}.tags) AS {alias}
                    ON {alias}.value = ?
                """
                conditions.append(f"{alias}.value IS NOT NULL")
                params.append(tag)

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT m.*, m.rowid as _rowid
            FROM {_MEMORY_TABLE} m {tag_joins}
            WHERE {where_clause}
        """
        rows = self._execute(sql, tuple(params)).fetchall()

        # Calculate combined score
        now_ts = _now_ts()
        scored = []
        for row in rows:
            mem = self._row_to_dict(row)
            rowid = row["_rowid"]
            bm25 = bm25_scores.get(rowid, 0.0)
            recency = _recency_score(mem.get("updated_at", ""))
            access_count = mem.get("access_count", 0)
            importance = mem.get("importance", 1.0)

            scored.append({
                **mem,
                "_score": (
                    self._weights["bm25"] * bm25
                    + self._weights["recency"] * recency
                    + self._weights["access_frequency"] * min(1.0, access_count / 10.0)
                    + self._weights["importance"] * min(1.0, importance / 10.0)
                ),
            })

        # Sort by combined score, then by recency tiebreaker
        scored.sort(key=lambda x: (-x["_score"], -_days_since(x.get("updated_at", ""))))

        # Update access counts
        for mem in scored[:limit]:
            try:
                self._execute(
                    f"UPDATE {_MEMORY_TABLE} SET access_count = access_count + 1, accessed_at = ? WHERE id = ?",
                    (_now(), mem["id"]),
                )
            except Exception:
                pass
        self._commit()

        result = scored[:limit]
        self._cache.set(cache_key, result)
        return result

    def _list_recent(self, limit: int, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        conditions = ["superseded = 0"]
        params: list = []
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        where = " AND ".join(conditions)
        rows = self._execute(
            f"SELECT * FROM {_MEMORY_TABLE} WHERE {where} ORDER BY updated_at DESC LIMIT ?",
            tuple(params) + (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- Build Context -----------------------------------------------------

    def build_context(self, query: str, limit: int = 5) -> str:
        """Build a formatted context block for LLM injection."""
        results = self.search(query, limit=limit)
        if not results:
            return ""
        lines = []
        for mem in results:
            content = mem.get("content", "")
            if not content:
                continue
            days = _days_since(mem.get("updated_at", ""))
            recency = "ago"
            if days < 1:
                recency = "today"
            elif days < 2:
                recency = "yesterday"
            elif days < 7:
                recency = f"{int(days)}d ago"
            elif days < 30:
                recency = f"{int(days)}d ago"
            else:
                recency = f"{int(days)}d ago"
            mtype = mem.get("memory_type", "fact")
            lines.append(f"- [{mtype}, {recency}] {content}")
        if not lines:
            return ""
        return (
            "<rekal-context>\n"
            "Relevant memories from previous sessions:\n"
            + "\n".join(lines)
            + "\n</rekal-context>"
        )

    # -- Conflict Detection ------------------------------------------------

    def get_conflicts(self, memory_id: Optional[str] = None,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Find memories that contradict each other.

        A conflict is defined as two active (non-superseded) memories linked
        with relation='contradicts'.
        """
        if memory_id:
            rows = self._execute(
                f"""SELECT m.*, l.relation
                    FROM {_MEMORY_TABLE} m
                    JOIN {_LINKS_TABLE} l ON (l.source_id = m.id OR l.target_id = m.id)
                    WHERE (l.source_id = ? OR l.target_id = ?)
                    AND l.relation = 'contradicts'
                    AND m.superseded = 0
                    LIMIT ?""",
                (memory_id, memory_id, limit),
            ).fetchall()
        else:
            rows = self._execute(
                f"""SELECT DISTINCT m.*, l.relation
                    FROM {_MEMORY_TABLE} m
                    JOIN {_LINKS_TABLE} l ON (l.source_id = m.id OR l.target_id = m.id)
                    WHERE l.relation = 'contradicts'
                    AND m.superseded = 0
                    ORDER BY m.updated_at DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- Similar -----------------------------------------------------------

    def similar(self, memory_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find memories with similar content via FTS5."""
        mem = self._fetch_memory(memory_id)
        if not mem:
            return []
        content = mem.get("content", "")
        if not content:
            return []

        words = _normalize_fts(content).split()[:20]
        if not words:
            return []

        query = " OR ".join(_quote_fts(w) for w in words)
        results = self.search(query, limit=limit + 1)
        return [r for r in results if r.get("id") != memory_id][:limit]

    # -- Topics ------------------------------------------------------------

    def topics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Aggregate tags into topic clusters with counts."""
        rows = self._execute(
            f"SELECT tags FROM {_MEMORY_TABLE} WHERE superseded = 0"
        ).fetchall()
        counter: Dict[str, int] = {}
        for row in rows:
            tags = _list_field(row["tags"])
            for tag in tags:
                counter[tag] = counter.get(tag, 0) + 1
        sorted_tags = sorted(counter.items(), key=lambda x: -x[1])
        return [{"tag": t, "count": c} for t, c in sorted_tags[:limit]]

    # -- Timeline ----------------------------------------------------------

    def timeline(self, limit: int = 20,
                 memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        conditions = ["superseded = 0"]
        params: list = []
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        where = " AND ".join(conditions)
        rows = self._execute(
            f"SELECT * FROM {_MEMORY_TABLE} WHERE {where} ORDER BY created_at DESC LIMIT ?",
            tuple(params) + (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- Related -----------------------------------------------------------

    def related(self, memory_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find memories linked via the memory_links table."""
        rows = self._execute(
            f"""SELECT m.*
                FROM {_MEMORY_TABLE} m
                JOIN {_LINKS_TABLE} l ON (l.target_id = m.id OR l.source_id = m.id)
                WHERE (l.source_id = ? OR l.target_id = ?)
                AND m.id != ?
                AND m.superseded = 0
                ORDER BY l.created_at DESC
                LIMIT ?""",
            (memory_id, memory_id, memory_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # -- Session Management ------------------------------------------------

    def session_init(self, session_id: str,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        now = _now()
        try:
            self._execute(
                f"""INSERT OR IGNORE INTO {_SESSIONS_TABLE}
                    (id, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?)""",
                (session_id, json.dumps(metadata or {}), now, now),
            )
            self._commit()
            return True
        except Exception:
            return False

    def ingest_turns(self, session_id: str, turns: List[Dict[str, Any]],
                     memory_type: str = "context",
                     max_tokens_hint: int = 0) -> Optional[Dict[str, Any]]:
        """Ingest conversation turns into memories, return summary."""
        if not turns:
            return None
        now = _now()
        ingested = 0

        for turn in turns:
            user_msg = (turn.get("user") or turn.get("user_message") or "").strip()
            asst_msg = (turn.get("assistant") or turn.get("assistant_message") or "").strip()
            combined = f"User: {user_msg}\nAssistant: {asst_msg}" if user_msg and asst_msg else (user_msg or asst_msg)
            if not combined or len(combined) < 20:
                continue
            c_hash = _content_hash(combined)

            # Dedup check
            exists = self._execute(
                f"SELECT 1 FROM {_MEMORY_TABLE} WHERE content_hash = ? AND memory_type = ?",
                (c_hash, memory_type),
            ).fetchone()
            if exists:
                continue

            mid = _new_id()
            self._execute(
                f"""INSERT INTO {_MEMORY_TABLE}
                    (id, content, content_hash, memory_type, tags, importance, metadata, created_at, updated_at, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mid, combined, c_hash, memory_type, "[]", 0.5,
                 json.dumps({"session_id": session_id, "ingested": True, "source": "turn"}),
                 now, now, now),
            )
            ingested += 1

        if ingested:
            self._commit()

        return {
            "session_id": session_id,
            "ingested": ingested,
            "memory_type": memory_type,
        }

    # -- Health / Stats ----------------------------------------------------

    def health(self) -> Dict[str, Any]:
        try:
            conn = self._get_conn()
            total = self._execute(
                f"SELECT COUNT(*) as c FROM {_MEMORY_TABLE}"
            ).fetchone()["c"]
            active = self._execute(
                f"SELECT COUNT(*) as c FROM {_MEMORY_TABLE} WHERE superseded = 0"
            ).fetchone()["c"]
            links = self._execute(
                f"SELECT COUNT(*) as c FROM {_LINKS_TABLE}"
            ).fetchone()["c"]
            sessions = self._execute(
                f"SELECT COUNT(*) as c FROM {_SESSIONS_TABLE}"
            ).fetchone()["c"]
            db_size = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0
            wal_size = 0
            wal_path = self._db_path + "-wal"
            if os.path.exists(wal_path):
                wal_size = os.path.getsize(wal_path)
            return {
                "ok": True,
                "total_memories": total,
                "active_memories": active,
                "total_links": links,
                "total_sessions": sessions,
                "db_size_bytes": db_size,
                "wal_size_bytes": wal_size,
                "schema_version": _SCHEMA_VERSION,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def deduplicate(self) -> Dict[str, Any]:
        """Remove exact duplicate memories (same content_hash), keeping oldest."""
        removed = 0
        rows = self._execute(
            f"""SELECT content_hash, MIN(created_at) as first_created
                FROM {_MEMORY_TABLE}
                WHERE superseded = 0
                GROUP BY content_hash
                HAVING COUNT(*) > 1"""
        ).fetchall()
        for row in rows:
            c_hash = row["content_hash"]
            first_created = row["first_created"]
            duplicates = self._execute(
                f"""SELECT id FROM {_MEMORY_TABLE}
                    WHERE content_hash = ? AND superseded = 0 AND created_at > ?""",
                (c_hash, first_created),
            ).fetchall()
            for dup in duplicates:
                self._execute(
                    f"DELETE FROM {_MEMORY_TABLE} WHERE id = ?",
                    (dup["id"],),
                )
                removed += 1
        if removed:
            self._commit()
        return {"removed": removed}

    # -- Config ------------------------------------------------------------

    def set_config(self, key: str, value: Any) -> None:
        if key == "weights":
            if isinstance(value, dict):
                self._weights.update(value)
        self._set_config(key, json.dumps(value) if not isinstance(value, str) else value)
