"""
mempalace_rekal_engine.py — Rekal memory intelligence natively embedded into MemPalace.

Place at: packages/opencode/src/tool/mempalace_rekal_engine.py

Provides:
  - Structured memory entries with types, tags, projects, access tracking
  - Memory lifecycle: store, update, supersede (with link preservation), delete
  - Memory links: supersedes, contradicts, related_to
  - Hybrid search: FTS5 BM25 + ChromaDB vector + recency decay + access frequency (configurable weights)
  - Conflict detection and resolution
  - Session context building (build_context)
  - Multi-hop graph traversal
  - Memory health and introspection
  - Content deduplication via SHA-256 hashing (store / batch_store)
  - TTL-aware in-memory LRU cache for hot memory rows
  - Batch ingestion via batch_store()
  - Direct ID lookup via get()
  - WAL checkpoint on close for full persistence
  - Locked-DB retry with exponential back-off

Architecture:
  - SQLite database (rekal_memories.db) for structured metadata, FTS5, links, config
  - MemPalace's ChromaDB for vector embeddings (reused, not duplicated)
  - MemPalace's KnowledgeGraph for entity relationships (reused)
  - All synchronous (matches bridge stdin loop)

Zero new pip dependencies. Uses only stdlib + existing MemPalace.
"""

import collections
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger("mempalace_rekal")

# Hindsight-derived retrieval primitives (rank fusion, graph expansion, temporal
# analysis, evidence accumulation, contradiction detection).  Imported
# defensively: every feature that uses them is opt-in, so a missing or broken
# module degrades to the original weighted-sum behaviour instead of taking the
# whole engine down.
try:
    from . import mempalace_hindsight as _hs

    HINDSIGHT_AVAILABLE = True
except Exception:  # pragma: no cover - direct-file execution fallback
    try:
        import mempalace_hindsight as _hs  # type: ignore[no-redef]

        HINDSIGHT_AVAILABLE = True
    except Exception as _hs_err:  # pragma: no cover - exercised only on broken installs
        _hs = None
        HINDSIGHT_AVAILABLE = False
        logger.debug("hindsight primitives unavailable (advanced retrieval disabled): %s", _hs_err)


# ══════════════════════════════════════════════════════════════════════════
#  TTL-Aware LRU Cache
# ══════════════════════════════════════════════════════════════════════════

class _TTLCache:
    """LRU cache with per-entry TTL. Not thread-safe — synchronous use only."""

    def __init__(self, maxsize=512, ttl=120):
        self._store: collections.OrderedDict = collections.OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, key, default=None):
        entry = self._store.get(key)
        if entry is None:
            return default
        value, expires = entry
        if time.monotonic() > expires:
            self._store.pop(key, None)
            return default
        self._store.move_to_end(key)
        return value

    def set(self, key, value):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.monotonic() + self._ttl)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def invalidate(self, key):
        self._store.pop(key, None)

    def clear(self):
        self._store.clear()


# ══════════════════════════════════════════════════════════════════════════
#  Schema
# ══════════════════════════════════════════════════════════════════════════

SCHEMA = """\
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT,
    memory_type TEXT NOT NULL DEFAULT 'fact'
        CHECK (memory_type IN ('fact','preference','procedure','context','episode')),
    project TEXT,
    wing TEXT,
    room TEXT,
    tags TEXT,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT
);

CREATE TABLE IF NOT EXISTS memory_links (
    from_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK (relation IN ('supersedes','contradicts','related_to')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id, relation)
);

CREATE TABLE IF NOT EXISTS rekal_config (
    project TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (project, key)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags, project,
    content='memories',
    content_rowid='rowid'
);

-- Indexes for fast filtered retrieval
CREATE INDEX IF NOT EXISTS idx_memories_project      ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_type         ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_wing_room    ON memories(wing, room);
CREATE INDEX IF NOT EXISTS idx_memories_created      ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_updated      ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_importance   ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_links_from            ON memory_links(from_id);
CREATE INDEX IF NOT EXISTS idx_links_to              ON memory_links(to_id);
CREATE INDEX IF NOT EXISTS idx_links_relation        ON memory_links(relation);

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
"""

# Migration: columns added after initial release
_MIGRATIONS = [
    "ALTER TABLE memories ADD COLUMN content_hash TEXT",
    "ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5",
    # Evidence accumulation: how many times this memory has been independently
    # observed.  Defaults to 1 so every pre-existing row reads as "seen once",
    # which is exactly neutral under proof_norm() and therefore cannot change
    # the ranking of any memory stored before this migration.
    "ALTER TABLE memories ADD COLUMN proof_count INTEGER NOT NULL DEFAULT 1",
    # Last time evidence for this memory was re-observed; lets recency reflect
    # reinforcement without overwriting the original created_at.
    "ALTER TABLE memories ADD COLUMN last_reinforced_at TEXT",
]


# ══════════════════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_W_FTS = 0.30
DEFAULT_W_VEC = 0.30
DEFAULT_W_RECENCY = 0.20
DEFAULT_W_ACCESS = 0.20          # access-frequency boost (new)
DEFAULT_HALF_LIFE = 30.0
ACCESS_SATURATION = 50.0         # access_count at which frequency score ≈ 1


def _normalize_fts(score):
    """FTS5 BM25: negative scores, lower = better. Sigmoid to [0,1]."""
    if score >= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(score))


def _normalize_vec(distance):
    """Cosine distance [0,2] → similarity [0,1]."""
    return max(0.0, 1.0 - distance)


def _recency_score(days, half_life=30.0):
    """Exponential decay. 0 days = 1.0, half_life days = 0.5."""
    return math.exp(-0.693 * max(0.0, days) / max(0.1, half_life))


def _access_score(access_count):
    """Log-saturating score: 0 → 0.0, ACCESS_SATURATION → ≈ 1.0."""
    return min(1.0, math.log1p(max(0, access_count)) / math.log1p(ACCESS_SATURATION))


def _importance_boost(importance):
    """Clamp user-set importance to [0, 1]."""
    return max(0.0, min(1.0, float(importance or 0.5)))


def _days_since(timestamp_str):
    """Parse ISO timestamp, return days since now. 0 on error."""
    try:
        if not timestamp_str:
            return 0
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return max(0.0, delta.total_seconds() / 86400)
    except (ValueError, TypeError):
        return 0


def _parse_ts(timestamp_str):
    """Parse a stored timestamp into a naive UTC datetime, or None."""
    if not timestamp_str:
        return None
    try:
        dt = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _quote_fts(query, match_any=False):
    """Build an FTS5 query that tries prefix matching for tokens ≥ 3 chars
    and falls back to exact phrase for short tokens.  Returns empty string
    when the query produces no usable tokens.

    `match_any` switches the implicit AND to an explicit OR.  FTS5 ANDs bare
    terms, so a natural-language query of more than two or three words
    effectively requires every one of them to appear in a single memory and
    the keyword arm returns nothing.  OR restores recall and leaves the
    ordering to bm25 and to rank fusion, which is where it belongs.  The
    default stays AND so existing callers see identical results.
    """
    tokens = [t for t in re.split(r"\s+", (query or "").replace('"', " ").replace("\x00", "")) if t]
    parts = []
    for raw in tokens:
        safe = re.sub(r"[^\w\-']", "", raw)
        if not safe:
            continue
        if len(safe) >= 3:
            parts.append(f'"{safe}"*')   # prefix match on last token of phrase
        else:
            parts.append(f'"{safe}"')    # exact match for very short tokens
    return (" OR " if match_any else " ").join(parts)


def _content_hash(content):
    """Stable SHA-256 fingerprint of memory content."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _new_id():
    return uuid.uuid4().hex[:16]


# ══════════════════════════════════════════════════════════════════════════
#  Engine
# ══════════════════════════════════════════════════════════════════════════

class RekalEngine:
    """Synchronous Rekal memory engine backed by SQLite + MemPalace ChromaDB.

    Initialized once per bridge subprocess lifetime. The SQLite database
    lives alongside MemPalace's ChromaDB in the same data directory.

    Recall improvements over baseline:
      - SQL indexes on every filter column → O(log n) filtered queries
      - content_hash deduplication → no redundant entries polluting results
      - access-frequency scoring component → hot memories float to the top
      - prefix-aware FTS5 queries → partial-token recall
      - TTL LRU cache for individual memory rows → near-zero re-read cost
      - WAL checkpoint on close → durable persistence after every session
      - Retry with back-off on SQLite BUSY → no silent write loss
    """

    def __init__(self, data_dir, palace_path=None, kg=None, search_memories_fn=None):
        self.data_dir = data_dir
        self.palace_path = palace_path or data_dir
        self.kg = kg
        self.search_memories_fn = search_memories_fn

        # In-memory row cache (id → dict); 2-minute TTL, 512 entries max
        self._cache = _TTLCache(maxsize=512, ttl=120)

        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "rekal_memories.db")
        # Exposed for Hermes's `hermes backup` provider contract, which
        # snapshots each provider's on-disk state via backup_paths().
        self._db_path = db_path
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")   # safe with WAL; faster
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA cache_size=-8000")     # 8 MB page cache
        self.db.execute("PRAGMA temp_store=MEMORY")
        # Must run before SCHEMA: the legacy link table lacks the from_id/to_id
        # columns the new index definitions reference, so executescript(SCHEMA)
        # would abort on "no such column".
        self._migrate_legacy_hermes_schema()
        self.db.executescript(SCHEMA)
        self._run_migrations()
        self._repair_fts_if_needed()
        self.db.execute("ANALYZE")                     # refresh query-planner stats
        self.db.commit()

    def _run_migrations(self):
        """Idempotently apply schema migrations for columns added post-release."""
        existing_cols = {row[1] for row in self.db.execute("PRAGMA table_info(memories)")}
        for stmt in _MIGRATIONS:
            col = stmt.split("ADD COLUMN")[1].strip().split()[0]
            if col not in existing_cols:
                try:
                    self.db.execute(stmt)
                    logger.debug("Migration applied: %s", stmt)
                except sqlite3.OperationalError as e:
                    logger.debug("Migration skipped (%s): %s", e, stmt)
        self.db.commit()

    def _migrate_legacy_hermes_schema(self):
        """Migrate the schema written by Hermes's pre-upstream Rekal port.

        The replaced port used ``source_id``/``target_id`` link columns, a
        project-less ``rekal_config(key, value)`` table, and a 1-10 importance
        scale.  The upstream engine reads ``from_id``/``to_id``, resolves
        weights with ``WHERE project = ?``, and scores importance in 0-1.
        Opening the old file unchanged would raise ``no such column`` on the
        first query.  This rebuilds the two tables in place, preserves legacy
        supersede state as ``supersedes`` links, and rescales importance.
        No-op on upstream-shaped or fresh databases; idempotent because the
        migration condition disappears once the tables are rebuilt.
        """
        try:
            link_cols = {row[1] for row in self.db.execute("PRAGMA table_info(memory_links)")}
        except sqlite3.DatabaseError:
            return
        if not link_cols or "from_id" in link_cols or "source_id" not in link_cols:
            return

        logger.info("Migrating legacy Hermes Rekal schema to upstream layout")
        self.db.execute("ALTER TABLE memory_links RENAME TO memory_links_legacy")
        self.db.execute(
            """CREATE TABLE memory_links (
                   from_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                   to_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                   relation TEXT NOT NULL CHECK (relation IN ('supersedes','contradicts','related_to')),
                   created_at TEXT NOT NULL,
                   PRIMARY KEY (from_id, to_id, relation))"""
        )
        self.db.execute(
            """INSERT OR IGNORE INTO memory_links (from_id, to_id, relation, created_at)
               SELECT l.source_id, l.target_id,
                      CASE WHEN l.relation IN ('supersedes','contradicts')
                           THEN l.relation ELSE 'related_to' END,
                      COALESCE(l.created_at, ?)
               FROM memory_links_legacy l
               WHERE EXISTS (SELECT 1 FROM memories m WHERE m.id = l.source_id)
                 AND EXISTS (SELECT 1 FROM memories m WHERE m.id = l.target_id)""",
            (_now(),),
        )
        self.db.execute("DROP TABLE memory_links_legacy")

        mem_cols = {row[1] for row in self.db.execute("PRAGMA table_info(memories)")}
        # The replaced port's memories table lacked the upstream project/wing/
        # room/last_accessed_at columns; SCHEMA's indexes and _row_to_dict read
        # them, so add them before executescript runs.
        for column, ddl in (
            ("project", "ALTER TABLE memories ADD COLUMN project TEXT"),
            ("wing", "ALTER TABLE memories ADD COLUMN wing TEXT"),
            ("room", "ALTER TABLE memories ADD COLUMN room TEXT"),
            ("last_accessed_at", "ALTER TABLE memories ADD COLUMN last_accessed_at TEXT"),
        ):
            if column not in mem_cols:
                self.db.execute(ddl)
        if {"superseded", "superseded_by"} <= mem_cols:
            self.db.execute(
                """INSERT OR IGNORE INTO memory_links (from_id, to_id, relation, created_at)
                   SELECT m.superseded_by, m.id, 'supersedes', COALESCE(m.updated_at, ?)
                   FROM memories m
                   WHERE m.superseded = 1 AND m.superseded_by IS NOT NULL
                     AND EXISTS (SELECT 1 FROM memories n WHERE n.id = m.superseded_by)""",
                (_now(),),
            )
        if "importance" in mem_cols:
            self.db.execute("UPDATE memories SET importance = importance / 10.0 WHERE importance > 1")

        try:
            cfg_cols = {row[1] for row in self.db.execute("PRAGMA table_info(rekal_config)")}
        except sqlite3.DatabaseError:
            cfg_cols = set()
        if cfg_cols and "project" not in cfg_cols:
            self.db.execute("ALTER TABLE rekal_config RENAME TO rekal_config_legacy")
            self.db.execute(
                """CREATE TABLE rekal_config (
                       project TEXT NOT NULL,
                       key TEXT NOT NULL,
                       value TEXT NOT NULL,
                       PRIMARY KEY (project, key))"""
            )
            self.db.execute(
                "INSERT OR REPLACE INTO rekal_config (project, key, value) "
                "SELECT '', key, value FROM rekal_config_legacy"
            )
            self.db.execute("DROP TABLE rekal_config_legacy")
        self.db.commit()

    def _repair_fts_if_needed(self):
        """Rebuild the FTS index when it has drifted out of sync with `memories`.

        `memories_fts` is an external-content FTS5 table kept current by
        triggers. If rows ever reach `memories` without the triggers firing --
        a database created by an older build, a restore that copied only the
        base table, an interrupted write -- the index and the table disagree.
        SQLite then reports `database disk image is malformed` on the next
        delete or update, because the delete trigger asks the index to remove a
        row it never indexed, and every subsequent operation fails.

        Counting both sides is cheap (FTS5 keeps its own docsize table), and a
        rebuild is far preferable to an engine that raises on every call.
        """
        try:
            # Neither obvious probe works here. `SELECT COUNT(*) FROM
            # memories_fts` is answered from the content table, so it reports
            # the right number even when the index holds nothing, and
            # 'integrity-check' only validates the index's internal structure,
            # which an empty index satisfies. The shadow `_docsize` table holds
            # exactly one row per *indexed* document, so comparing it against
            # `memories` is the one cheap check that actually detects drift.
            memories = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            indexed = self.db.execute("SELECT COUNT(*) FROM memories_fts_docsize").fetchone()[0]
            if memories == indexed:
                return
            logger.debug("FTS index out of sync (%s rows, %s indexed); rebuilding", memories, indexed)
        except sqlite3.DatabaseError as exc:
            logger.debug("FTS index unreadable (%s); rebuilding", exc)

        try:
            self.db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self.db.commit()
        except sqlite3.DatabaseError as exc:
            # A rebuild is best-effort: if the index is too damaged to count or
            # rebuild in place, drop and recreate it rather than leaving the
            # engine permanently unusable.
            logger.debug("FTS rebuild failed (%s); recreating the index", exc)
            try:
                self.db.execute("DROP TABLE IF EXISTS memories_fts")
                self.db.execute(
                    "CREATE VIRTUAL TABLE memories_fts USING fts5("
                    "content, tags, project, content='memories', content_rowid='rowid')"
                )
                self.db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                self.db.commit()
            except sqlite3.DatabaseError as inner:
                logger.warning("FTS index unavailable; keyword search degraded: %s", inner)

    def _execute(self, sql, params=(), retries=4, base_delay=0.04):
        """Execute with exponential back-off on SQLITE_BUSY / locked errors."""
        for attempt in range(retries):
            try:
                return self.db.execute(sql, params)
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise

    def close(self):
        """Flush WAL to the main DB file before closing for full persistence."""
        try:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.commit()
        except Exception:
            pass
        try:
            self.db.close()
        except Exception:
            pass

    def checkpoint(self):
        """Manually trigger a WAL checkpoint and return diagnostic info."""
        try:
            row = self.db.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            return {
                "success": True,
                "wal_pages": row[1] if row else None,
                "checkpointed": row[2] if row else None,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Config ────────────────────────────────────────────────────────

    def _resolve_weights(self, project=None, w_fts=None, w_vec=None,
                         w_recency=None, w_access=None, half_life=None):
        """Four-level weight resolution: per-call > DB config > defaults."""
        result = {
            "w_fts": DEFAULT_W_FTS,
            "w_vec": DEFAULT_W_VEC,
            "w_recency": DEFAULT_W_RECENCY,
            "w_access": DEFAULT_W_ACCESS,
            "half_life": DEFAULT_HALF_LIFE,
        }
        if project:
            cursor = self._execute(
                "SELECT key, value FROM rekal_config WHERE project = ?",
                (project,),
            )
            for row in cursor:
                if row["key"] in result:
                    try:
                        result[row["key"]] = float(row["value"])
                    except (ValueError, TypeError):
                        pass
        overrides = {
            "w_fts": w_fts, "w_vec": w_vec, "w_recency": w_recency,
            "w_access": w_access, "half_life": half_life,
        }
        for k, v in overrides.items():
            if v is not None:
                result[k] = float(v)
        return result

    def set_config(self, project, key, value):
        valid_keys = {"w_fts", "w_vec", "w_recency", "w_access", "half_life"}
        if key not in valid_keys:
            return {"error": f"Invalid key '{key}'. Valid: {', '.join(sorted(valid_keys))}"}
        try:
            float(value)
        except (ValueError, TypeError):
            return {"error": f"Value must be numeric, got: {value}"}
        self._execute(
            "INSERT INTO rekal_config (project, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT (project, key) DO UPDATE SET value = excluded.value",
            (project, key, str(value)),
        )
        self.db.commit()
        return {"success": True, "key": key, "value": value, "project": project}

    # ── Direct lookup ─────────────────────────────────────────────────

    def get(self, memory_id, track_access=True):
        """Retrieve a single memory by ID with optional access tracking.

        Checks the in-memory cache first; falls back to SQLite on miss.
        Returns None if the memory does not exist.
        """
        cached = self._cache.get(memory_id)
        if cached is not None:
            if track_access:
                ts = _now()
                self._execute(
                    "UPDATE memories SET access_count = access_count + 1, "
                    "last_accessed_at = ? WHERE id = ?",
                    (ts, memory_id),
                )
                self.db.commit()
                cached = dict(cached)
                cached["access_count"] = cached.get("access_count", 0) + 1
                self._cache.set(memory_id, cached)
            return cached

        row = self._execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        result = self._row_to_dict(row)
        if track_access:
            ts = _now()
            self._execute(
                "UPDATE memories SET access_count = access_count + 1, "
                "last_accessed_at = ? WHERE id = ?",
                (ts, memory_id),
            )
            self.db.commit()
            result["access_count"] += 1
        self._cache.set(memory_id, result)
        return result

    # ── Store ─────────────────────────────────────────────────────────

    def store(self, content, memory_type="fact", project=None, wing=None,
              room=None, tags=None, importance=0.5, allow_duplicate=False):
        """Store a memory, returning early (with duplicate=True) if identical
        content already exists and allow_duplicate is False.
        """
        chash = _content_hash(content)

        if not allow_duplicate:
            existing = self._execute(
                "SELECT id FROM memories WHERE content_hash = ?", (chash,)
            ).fetchone()
            if existing:
                # Re-observing the same fact is evidence, not noise. Counting it
                # turns repetition into a ranking signal (proof_norm) instead of
                # silently discarding the observation as the baseline did.
                proof = self._reinforce(existing["id"])
                result = {
                    "success": True,
                    "memory_id": existing["id"],
                    "duplicate": True,
                    "message": "Memory already exists with identical content.",
                }
                if proof is not None:
                    result["proof_count"] = proof
                    result["reinforced"] = True
                return result

        memory_id = _new_id()
        ts = _now()
        tags_json = json.dumps(tags) if tags else None
        importance = max(0.0, min(1.0, float(importance or 0.5)))

        self._execute(
            """INSERT INTO memories
               (id, content, content_hash, memory_type, project, wing, room,
                tags, importance, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, content, chash, memory_type,
             project, wing or "project", room or "general",
             tags_json, importance, ts, ts),
        )
        self.db.commit()

        # Also store in MemPalace ChromaDB for vector search
        try:
            from mempalace.mcp_server import tool_add_drawer
            tool_add_drawer(
                wing=wing or "project",
                room=room or "general",
                content=content,
                added_by="rekal",
            )
        except Exception as e:
            logger.debug("ChromaDB store failed (non-fatal): %s", e)

        result = {
            "success": True,
            "memory_id": memory_id,
            "memory_type": memory_type,
            "project": project,
            "wing": wing or "project",
            "room": room or "general",
            "importance": importance,
        }

        # Surface disagreements at write time. The baseline exposed
        # `memory_conflicts` but nothing ever created a `contradicts` link, so
        # it could only ever return an empty list.
        conflicts = self._detect_conflicts(memory_id, content, project)
        if conflicts:
            result["contradicts"] = conflicts

        return result

    def _reinforce(self, memory_id):
        """Record another independent observation of an existing memory.

        Returns the new proof count, or None if the column is not yet present
        (a database written by an older engine that has not been reopened).
        """
        try:
            self._execute(
                "UPDATE memories SET proof_count = COALESCE(proof_count, 1) + 1, "
                "last_reinforced_at = ? WHERE id = ?",
                (_now(), memory_id),
            )
            self.db.commit()
            self._cache.invalidate(memory_id)
            row = self._execute(
                "SELECT proof_count FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            return int(row["proof_count"]) if row else None
        except sqlite3.OperationalError as exc:
            logger.debug("reinforce skipped: %s", exc)
            return None

    def _detect_conflicts(self, memory_id, content, project, max_checked=40):
        """Link a newly stored memory to existing memories it contradicts.

        Lexical detection cannot prove a contradiction, so a hit records a
        `contradicts` link for review rather than superseding or deleting
        anything. Nothing is ever removed on the strength of this signal.
        """
        if not HINDSIGHT_AVAILABLE or not content:
            return []
        try:
            # Only same-project memories of comparable length are plausible
            # conflicts; this keeps the scan bounded on large stores.
            if project:
                rows = self._execute(
                    "SELECT id, content FROM memories WHERE id != ? AND project = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (memory_id, project, max_checked),
                ).fetchall()
            else:
                rows = self._execute(
                    "SELECT id, content FROM memories WHERE id != ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (memory_id, max_checked),
                ).fetchall()

            found = []
            for row in rows:
                is_conflict, confidence = _hs.detect_contradiction(content, row["content"])
                if not is_conflict:
                    continue
                self._execute(
                    "INSERT OR IGNORE INTO memory_links (from_id, to_id, relation, created_at) "
                    "VALUES (?, ?, 'contradicts', ?)",
                    (memory_id, row["id"], _now()),
                )
                found.append({
                    "memory_id": row["id"],
                    "confidence": confidence,
                    "content": (row["content"] or "")[:160],
                })
            if found:
                self.db.commit()
            return found
        except Exception as exc:
            logger.debug("conflict detection skipped: %s", exc)
            return []

    # ── Batch Store ───────────────────────────────────────────────────

    def batch_store(self, items, allow_duplicate=False):
        """Atomically ingest a list of memory dicts.

        Each item must have 'content' and may include 'memory_type',
        'project', 'wing', 'room', 'tags', and 'importance'.

        Returns a list of per-item results in the same order as input.
        """
        if not items:
            return {"success": True, "results": [], "stored": 0, "skipped": 0}

        ts = _now()
        results = []
        stored = 0
        skipped = 0

        try:
            self.db.execute("BEGIN")
            for item in items:
                content = item.get("content", "")
                if not content:
                    results.append({"success": False, "error": "Empty content"})
                    skipped += 1
                    continue

                chash = _content_hash(content)

                if not allow_duplicate:
                    existing = self.db.execute(
                        "SELECT id FROM memories WHERE content_hash = ?", (chash,)
                    ).fetchone()
                    if existing:
                        results.append({
                            "success": True,
                            "memory_id": existing["id"],
                            "duplicate": True,
                        })
                        skipped += 1
                        continue

                memory_id = _new_id()
                tags = item.get("tags")
                importance = max(0.0, min(1.0, float(item.get("importance", 0.5))))
                self.db.execute(
                    """INSERT INTO memories
                       (id, content, content_hash, memory_type, project, wing, room,
                        tags, importance, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (memory_id, content, chash,
                     item.get("memory_type", "fact"),
                     item.get("project"),
                     item.get("wing", "project"),
                     item.get("room", "general"),
                     json.dumps(tags) if tags else None,
                     importance, ts, ts),
                )
                results.append({"success": True, "memory_id": memory_id})
                stored += 1

            self.db.execute("COMMIT")
        except Exception as exc:
            self.db.execute("ROLLBACK")
            return {"success": False, "error": str(exc), "results": results}

        return {"success": True, "results": results, "stored": stored, "skipped": skipped}

    # ── Update ────────────────────────────────────────────────────────

    def update(self, memory_id, content=None, tags=None, memory_type=None,
               importance=None):
        if content is None and tags is None and memory_type is None and importance is None:
            return {"success": True, "memory_id": memory_id, "noop": True}

        tags_json = json.dumps(tags) if tags is not None else None
        chash = _content_hash(content) if content is not None else None
        imp = max(0.0, min(1.0, float(importance))) if importance is not None else None
        ts = _now()

        cursor = self._execute(
            """UPDATE memories SET
               content      = COALESCE(?, content),
               content_hash = COALESCE(?, content_hash),
               tags         = COALESCE(?, tags),
               memory_type  = COALESCE(?, memory_type),
               importance   = COALESCE(?, importance),
               updated_at   = ?
               WHERE id = ?""",
            (content, chash, tags_json, memory_type, imp, ts, memory_id),
        )
        self.db.commit()
        self._cache.invalidate(memory_id)

        if cursor.rowcount == 0:
            return {"success": False, "error": f"Memory {memory_id} not found"}
        return {"success": True, "memory_id": memory_id}

    # ── Supersede ─────────────────────────────────────────────────────

    def supersede(self, old_id, new_content, memory_type=None, project=None,
                  wing=None, room=None, tags=None, importance=None):
        old = self._execute("SELECT * FROM memories WHERE id = ?", (old_id,)).fetchone()
        if not old:
            return {"success": False, "error": f"Memory {old_id} not found"}

        new_id = _new_id()
        ts = _now()
        eff_tags = json.dumps(tags) if tags is not None else old["tags"]
        imp = max(0.0, min(1.0, float(importance))) if importance is not None \
            else float(old["importance"] if old["importance"] is not None else 0.5)
        chash = _content_hash(new_content)

        self._execute(
            """INSERT INTO memories
               (id, content, content_hash, memory_type, project, wing, room,
                tags, importance, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (new_id, new_content, chash,
             memory_type or old["memory_type"],
             project or old["project"],
             wing or old["wing"],
             room or old["room"],
             eff_tags, imp, ts, ts),
        )
        self._execute(
            "INSERT INTO memory_links (from_id, to_id, relation, created_at) "
            "VALUES (?, ?, 'supersedes', ?)",
            (new_id, old_id, ts),
        )
        self.db.commit()
        self._cache.invalidate(old_id)

        try:
            from mempalace.mcp_server import tool_add_drawer
            tool_add_drawer(
                wing=wing or old["wing"] or "project",
                room=room or old["room"] or "general",
                content=new_content,
                added_by="rekal",
            )
        except Exception:
            pass

        return {
            "success": True,
            "new_id": new_id,
            "old_id": old_id,
            "fact": f"Created {new_id} superseding {old_id}",
        }

    # ── Delete ────────────────────────────────────────────────────────

    def delete(self, memory_id):
        cursor = self._execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()
        self._cache.invalidate(memory_id)
        if cursor.rowcount == 0:
            return {"success": False, "error": f"Memory {memory_id} not found"}
        return {"success": True, "memory_id": memory_id}

    # ── Link ──────────────────────────────────────────────────────────

    def link(self, from_id, to_id, relation):
        if relation not in ("supersedes", "contradicts", "related_to"):
            return {"error": f"Invalid relation: {relation}"}
        try:
            self._execute(
                "INSERT OR IGNORE INTO memory_links "
                "(from_id, to_id, relation, created_at) VALUES (?, ?, ?, ?)",
                (from_id, to_id, relation, _now()),
            )
            self.db.commit()
            return {"success": True, "from": from_id, "to": to_id, "relation": relation}
        except sqlite3.IntegrityError as e:
            return {"error": str(e)}

    # ── Internal helpers ──────────────────────────────────────────────

    def _row_to_dict(self, row):
        """Convert a sqlite3.Row from the memories table to a plain dict."""
        tags = []
        try:
            tags = json.loads(row["tags"]) if row["tags"] else []
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "id": row["id"],
            "content": row["content"],
            "content_hash": row["content_hash"] if "content_hash" in row.keys() else None,
            "memory_type": row["memory_type"],
            "project": row["project"],
            "wing": row["wing"],
            "room": row["room"],
            "tags": tags,
            "importance": float(row["importance"]) if row["importance"] is not None else 0.5,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "access_count": row["access_count"],
            "last_accessed_at": row["last_accessed_at"],
            # Post-migration columns: tolerate rows read before the migration ran.
            "proof_count": (
                int(row["proof_count"])
                if "proof_count" in row.keys() and row["proof_count"] is not None
                else 1
            ),
            "last_reinforced_at": (
                row["last_reinforced_at"] if "last_reinforced_at" in row.keys() else None
            ),
        }

    def _fetch_memory(self, memory_id):
        """Fetch a memory row dict, using the cache when available."""
        cached = self._cache.get(memory_id)
        if cached is not None:
            return cached
        row = self._execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        result = self._row_to_dict(row)
        self._cache.set(memory_id, result)
        return result

    # ── Search (hybrid) ───────────────────────────────────────────────

    def search(self, query, limit=10, project=None, memory_type=None,
               wing=None, room=None, w_fts=None, w_vec=None,
               w_recency=None, w_access=None, half_life=None,
               fusion=None, graph_expand=None, temporal=None, strategy_boosts=None):
        """Hybrid search over the Rekal store.

        `fusion`, `graph_expand` and `temporal` are opt-in refinements layered
        on top of the original weighted-sum ranking.  When all three are unset
        (the default) this method behaves exactly as before, byte for byte, so
        existing callers and stored weights are unaffected.

        fusion="rrf"      rank-space Reciprocal Rank Fusion across the FTS,
                          vector, graph and temporal arms, followed by bounded
                          multiplicative boosts for recency / importance /
                          evidence / graph activation.
        graph_expand=True pull in memories linked to the top hits even when they
                          match neither the text nor the vector query.
        temporal=True     parse a time window out of the query and prefer
                          memories near its midpoint.
        """
        weights = self._resolve_weights(
            project, w_fts, w_vec, w_recency, w_access, half_life
        )

        # ── Phase 0: Query analysis (opt-in) ──────────────────────────
        # Separate "when" from "what" before the lexical arm runs. FTS5 ANDs
        # its terms, so leaving "yesterday" in the query forces every stored
        # memory to literally contain that word and collapses the arm to zero
        # hits. Only the keyword query is rewritten; the vector arm keeps the
        # full text, where the temporal wording is still useful signal.
        temporal_window = None
        lexical_query = query
        if temporal and HINDSIGHT_AVAILABLE:
            try:
                temporal_window, lexical_query = _hs.analyze_query(query)
            except Exception as exc:
                logger.debug("query analysis skipped: %s", exc)
                temporal_window, lexical_query = None, query

        # ── Phase 1: FTS5 candidates ──────────────────────────────────
        fts_scores = {}
        # Advanced retrieval ranks with RRF, which tolerates a wide, noisy
        # keyword arm; the baseline path has no such re-ranking and keeps the
        # stricter AND matching it was tuned for.
        fts_query = _quote_fts(lexical_query, match_any=bool(fusion or graph_expand or temporal))
        if fts_query:
            try:
                # Exclude memories that have been superseded
                cursor = self._execute(
                    """SELECT m.id, memories_fts.rank AS fts_rank
                       FROM memories_fts
                       JOIN memories m ON m.rowid = memories_fts.rowid
                       WHERE memories_fts MATCH ?
                         AND m.id NOT IN (
                             SELECT to_id FROM memory_links WHERE relation = 'supersedes'
                         )
                       ORDER BY memories_fts.rank
                       LIMIT ?""",
                    (fts_query, limit * 4),
                )
                for row in cursor:
                    fts_scores[row["id"]] = row["fts_rank"]
            except sqlite3.OperationalError as exc:
                logger.debug("FTS5 query failed (%s) — falling back to prefix-less form.", exc)
                # Re-try with simple quoted tokens (no prefix *)
                fallback = " ".join(f'"{t}"' for t in query.split() if t)
                try:
                    cursor = self._execute(
                        """SELECT m.id, memories_fts.rank AS fts_rank
                           FROM memories_fts
                           JOIN memories m ON m.rowid = memories_fts.rowid
                           WHERE memories_fts MATCH ?
                             AND m.id NOT IN (
                                 SELECT to_id FROM memory_links WHERE relation = 'supersedes'
                             )
                           ORDER BY memories_fts.rank
                           LIMIT ?""",
                        (fallback, limit * 4),
                    )
                    for row in cursor:
                        fts_scores[row["id"]] = row["fts_rank"]
                except sqlite3.OperationalError:
                    pass

        # ── Phase 2: ChromaDB vector candidates ───────────────────────
        chroma_hits = []
        vec_scores = {}
        if self.search_memories_fn:
            try:
                import inspect
                kwargs = {
                    "query": query,
                    "palace_path": self.palace_path,
                    "wing": wing,
                    "room": room,
                    "n_results": limit * 4,
                    "max_distance": 0.0,
                }
                sig = inspect.signature(self.search_memories_fn).parameters
                if "vector_disabled" in sig:
                    kwargs["vector_disabled"] = False
                chroma_results = self.search_memories_fn(**kwargs)
                for hit in chroma_results.get("results", []):
                    text = hit.get("text", "")
                    dist = hit.get("distance")
                    if text:
                        ck = hashlib.md5(text[:200].encode()).hexdigest()[:12]
                        if dist is not None:
                            vec_scores[ck] = float(dist)
                        chroma_hits.append(hit)
            except Exception as exc:
                logger.debug("ChromaDB vector search failed: %s", exc)

        # ── Phase 3: Build candidate set from SQLite ──────────────────
        # Always seed with recency-ordered rows so low-traffic entries aren't lost
        candidate_ids = set(fts_scores.keys())
        if len(candidate_ids) < limit * 2:
            extra_cursor = self._execute(
                """SELECT id FROM memories
                   WHERE id NOT IN (
                       SELECT to_id FROM memory_links WHERE relation = 'supersedes'
                   )
                   ORDER BY importance DESC, created_at DESC
                   LIMIT ?""",
                (limit * 4,),
            )
            for row in extra_cursor:
                candidate_ids.add(row["id"])

        # ── Phase 4: Score Rekal SQLite entries ───────────────────────
        scored = []
        seen_content_keys = set()

        for cid in candidate_ids:
            mem = self._fetch_memory(cid)
            if not mem:
                continue
            if project and mem["project"] != project:
                continue
            if memory_type and mem["memory_type"] != memory_type:
                continue
            if wing and mem["wing"] != wing:
                continue
            if room and mem["room"] != room:
                continue

            fts_norm = _normalize_fts(fts_scores.get(cid, 0.0))
            ck = hashlib.md5((mem["content"] or "")[:200].encode()).hexdigest()[:12]
            seen_content_keys.add(ck)
            vec_dist = vec_scores.get(ck, 1.0)
            vec_norm = _normalize_vec(vec_dist)
            days = _days_since(mem["created_at"])
            rec_norm = _recency_score(days, weights["half_life"])
            acc_norm = _access_score(mem["access_count"])
            imp_boost = _importance_boost(mem["importance"])

            # Combine weighted components; importance is an additive multiplier
            raw_score = (
                weights["w_fts"] * fts_norm +
                weights["w_vec"] * vec_norm +
                weights["w_recency"] * rec_norm +
                weights["w_access"] * acc_norm
            )
            # Importance shifts score toward 1 (high) or 0 (low) via lerp
            score = raw_score * 0.85 + imp_boost * 0.15

            scored.append({
                "id": mem["id"],
                "content": mem["content"],
                "memory_type": mem["memory_type"],
                "project": mem["project"],
                "wing": mem["wing"],
                "room": mem["room"],
                "tags": mem["tags"],
                "importance": mem["importance"],
                "created_at": mem["created_at"],
                "updated_at": mem["updated_at"],
                "access_count": mem["access_count"],
                "score": round(score, 4),
                "fts_score": round(fts_norm, 3),
                "vec_score": round(vec_norm, 3),
                "recency_score": round(rec_norm, 3),
                "access_score": round(acc_norm, 3),
                "source": "rekal",
            })

        # ── Phase 5: Augment with ChromaDB-only (MemPalace native) ────
        for hit in chroma_hits:
            text = hit.get("text", "")
            ck = hashlib.md5(text[:200].encode()).hexdigest()[:12]
            if ck in seen_content_keys:
                continue
            seen_content_keys.add(ck)

            hit_wing = hit.get("wing", "unknown")
            hit_room = hit.get("room", "unknown")
            if wing and hit_wing != wing:
                continue
            if room and hit_room != room:
                continue

            vec_dist = hit.get("distance", 1.0)
            vec_norm = _normalize_vec(float(vec_dist) if vec_dist is not None else 1.0)
            bm25 = hit.get("bm25_score", 0.0) or 0.0
            filed_at = hit.get("created_at", "unknown")
            days = _days_since(filed_at) if filed_at != "unknown" else 0
            rec_norm = _recency_score(days, weights["half_life"])
            fts_norm = min(bm25 / 10.0, 1.0) if bm25 > 0 else 0.0

            raw_score = (
                weights["w_fts"] * fts_norm +
                weights["w_vec"] * vec_norm +
                weights["w_recency"] * rec_norm
                # no access component — chroma-only entries have no counter
            )
            score = raw_score * 0.85 + 0.5 * 0.15  # neutral importance

            scored.append({
                "id": hit.get("source_file", ck),
                "content": text[:2000],
                "memory_type": "fact",
                "project": None,
                "wing": hit_wing,
                "room": hit_room,
                "tags": [],
                "importance": 0.5,
                "created_at": filed_at,
                "updated_at": filed_at,
                "access_count": 0,
                "score": round(score, 4),
                "fts_score": round(fts_norm, 3),
                "vec_score": round(vec_norm, 3),
                "recency_score": round(rec_norm, 3),
                "access_score": 0.0,
                "source": "mempalace_drawer",
            })

        # ── Phase 5.5: Advanced retrieval (opt-in, Hindsight-derived) ─
        fusion_meta = None
        if HINDSIGHT_AVAILABLE and (fusion or graph_expand or temporal):
            scored, fusion_meta = self._refine_candidates(
                query=query,
                scored=scored,
                limit=limit,
                fusion=fusion,
                graph_expand=graph_expand,
                temporal=temporal,
                strategy_boosts=strategy_boosts,
                project=project,
                memory_type=memory_type,
                wing=wing,
                room=room,
                temporal_window=temporal_window,
            )

        # ── Phase 6: Update access counters for Rekal entries ─────────
        ts = _now()
        for s in scored:
            if s.get("source") == "rekal":
                self._execute(
                    "UPDATE memories SET access_count = access_count + 1, "
                    "last_accessed_at = ? WHERE id = ?",
                    (ts, s["id"]),
                )
                # Invalidate stale cache entry so next fetch reflects new count
                self._cache.invalidate(s["id"])
        self.db.commit()

        scored.sort(key=lambda x: x["score"], reverse=True)
        if fusion_meta:
            return {
                "query": query,
                "weights": weights,
                "retrieval": fusion_meta,
                "results": scored[:limit],
                "total_candidates": len(scored),
            }
        return {
            "query": query,
            "weights": weights,
            "results": scored[:limit],
            "total_candidates": len(scored),
        }

    # ── Advanced retrieval (Hindsight-derived, opt-in) ────────────────

    def _link_adjacency(self, seed_ids, max_nodes=2000):
        """Undirected adjacency map over memory_links, restricted to the
        component reachable from `seed_ids` in a couple of hops.

        Links are traversed in both directions: `supersedes` and `contradicts`
        are stored one-way, but for relatedness purposes an edge is evidence in
        either direction.
        """
        if not seed_ids:
            return {}
        adjacency = {}
        frontier = list(seed_ids)
        seen = set(seed_ids)
        for _ in range(2):
            if not frontier or len(seen) >= max_nodes:
                break
            placeholders = ",".join("?" for _ in frontier)
            rows = self._execute(
                f"""SELECT from_id, to_id, relation FROM memory_links
                    WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})""",
                tuple(frontier) * 2,
            ).fetchall()
            next_frontier = []
            for row in rows:
                for src, dst in ((row["from_id"], row["to_id"]), (row["to_id"], row["from_id"])):
                    edges = adjacency.setdefault(src, [])
                    # A row matches both IN clauses when either endpoint is in
                    # the frontier, so skip the duplicate rather than letting
                    # the same edge be traversed twice.
                    if (dst, row["relation"]) not in edges:
                        edges.append((dst, row["relation"]))
                    if dst not in seen:
                        seen.add(dst)
                        next_frontier.append(dst)
            frontier = next_frontier
        return adjacency

    def _refine_candidates(self, query, scored, limit, fusion, graph_expand,
                           temporal, strategy_boosts, project, memory_type,
                           wing, room, temporal_window=None):
        """Re-rank `scored` using rank fusion, graph expansion and temporal
        analysis.  Returns `(results, metadata)`.

        Any failure here is contained: the original `scored` list is returned
        unchanged so an opt-in refinement can never make search worse than the
        baseline it layers on top of.
        """
        try:
            meta = {}
            by_id = {entry["id"]: entry for entry in scored}

            # Graph expansion: reach memories linked to the current hits even
            # when they match neither the text nor the vector query. Seeds are
            # the strongest current candidates only -- expanding from the whole
            # pool would pull in most of a densely linked store.
            activations = {}
            if graph_expand:
                seeds = [e["id"] for e in sorted(scored, key=lambda x: x["score"], reverse=True)[:10]
                         if e.get("source") == "rekal"]
                adjacency = self._link_adjacency(seeds)
                activations = _hs.expand_links(seeds, adjacency, max_hops=2, budget=max(limit * 2, 20))
                for mem_id, activation in activations.items():
                    if mem_id in by_id:
                        continue
                    mem = self._fetch_memory(mem_id)
                    if not mem:
                        continue
                    # Expanded memories still honour the caller's filters.
                    if project and mem["project"] != project:
                        continue
                    if memory_type and mem["memory_type"] != memory_type:
                        continue
                    if wing and mem["wing"] != wing:
                        continue
                    if room and mem["room"] != room:
                        continue
                    entry = {
                        "id": mem["id"],
                        "content": mem["content"],
                        "memory_type": mem["memory_type"],
                        "project": mem["project"],
                        "wing": mem["wing"],
                        "room": mem["room"],
                        "tags": mem["tags"],
                        "importance": mem["importance"],
                        "created_at": mem["created_at"],
                        "updated_at": mem["updated_at"],
                        "access_count": mem["access_count"],
                        "proof_count": mem.get("proof_count", 1),
                        "score": 0.0,
                        "fts_score": 0.0,
                        "vec_score": 0.0,
                        "recency_score": 0.0,
                        "access_score": 0.0,
                        "source": "rekal",
                        "via": "graph",
                    }
                    scored.append(entry)
                    by_id[mem_id] = entry
                meta["graph_expanded"] = len(activations)

            # Temporal analysis: derive a window from the query text and score
            # each candidate by how close it sits to that window's midpoint.
            proximity = {}
            if temporal:
                # Reuse the window computed during query analysis so the
                # lexical rewrite and the proximity scoring can never disagree.
                window = temporal_window if temporal_window is not None else _hs.extract_temporal_constraint(query)
                if window:
                    start, end = window
                    meta["temporal_window"] = [start.isoformat(), end.isoformat()]
                    for entry in scored:
                        created = _parse_ts(entry.get("created_at"))
                        proximity[entry["id"]] = _hs.temporal_proximity(created, start, end)
                else:
                    meta["temporal_window"] = None

            if not fusion:
                # Graph/temporal without fusion: fold the new signals into the
                # existing score multiplicatively, leaving its composition intact.
                for entry in scored:
                    entry["score"] = round(_hs.combined_score(
                        entry["score"] or 0.0,
                        graph=activations.get(entry["id"], 0.5) if activations else 0.5,
                        recency=proximity.get(entry["id"], 0.5) if proximity else 0.5,
                    ), 4)
                    if entry["id"] in activations:
                        entry["graph_score"] = round(activations[entry["id"]], 3)
                    if entry["id"] in proximity:
                        entry["temporal_score"] = round(proximity[entry["id"]], 3)
                meta["mode"] = "boosted"
                return scored, meta

            # Rank fusion. Each arm contributes an ordering, never a raw score:
            # bm25 rank, cosine distance, graph activation and temporal
            # proximity are not on comparable scales, and RRF is exactly the
            # tool for combining orderings whose scores cannot be compared.
            arms = []
            fts_ranked = [e["id"] for e in sorted(scored, key=lambda x: x.get("fts_score", 0.0), reverse=True)
                          if e.get("fts_score", 0.0) > 0]
            if fts_ranked:
                arms.append(("fts", fts_ranked))
            vec_ranked = [e["id"] for e in sorted(scored, key=lambda x: x.get("vec_score", 0.0), reverse=True)
                          if e.get("vec_score", 0.0) > 0]
            if vec_ranked:
                arms.append(("vec", vec_ranked))
            if activations:
                arms.append(("graph", [mid for mid, _ in sorted(
                    activations.items(), key=lambda kv: kv[1], reverse=True)]))
            if proximity:
                temporal_ranked = [mid for mid, prox in sorted(
                    proximity.items(), key=lambda kv: kv[1], reverse=True) if prox > 0]
                if temporal_ranked:
                    arms.append(("temporal", temporal_ranked))

            if not arms:
                meta["mode"] = "rrf-empty"
                return scored, meta

            fused = _hs.reciprocal_rank_fusion(arms)
            meta["mode"] = "rrf"
            meta["arms"] = [name for name, _ in arms]

            now = datetime.now(timezone.utc).replace(tzinfo=None)
            results = []
            for item in fused:
                entry = by_id.get(item["id"])
                if entry is None:
                    continue
                base = _hs.boosted_rrf_score(
                    item["rrf_score"], item["source_ranks"], strategy_boosts or {}
                )
                created = _parse_ts(entry.get("created_at"))
                recency = _hs.recency_for_range(created, None, None, now)
                if entry["id"] in proximity:
                    # An explicit temporal window in the query outranks generic
                    # freshness: the user asked about a period, not about "new".
                    recency = proximity[entry["id"]]
                entry["score"] = round(_hs.combined_score(
                    base,
                    recency=recency,
                    importance=_importance_boost(entry.get("importance")),
                    proof=_hs.proof_norm(entry.get("proof_count", 1)),
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

            # Anything the arms never ranked (e.g. backfill rows) keeps its
            # original score and sorts after the fused set.
            fused_ids = {item["id"] for item in fused}
            for entry in scored:
                if entry["id"] not in fused_ids:
                    entry["score"] = round((entry.get("score") or 0.0) * 0.01, 6)
                    results.append(entry)

            return results, meta
        except Exception as exc:
            logger.debug("advanced retrieval failed, falling back to base ranking: %s", exc)
            return scored, {"mode": "fallback", "error": str(exc)}

    # ── Build Context ─────────────────────────────────────────────────

    def build_context(self, query, project=None, limit=10, w_fts=None,
                      w_vec=None, w_recency=None, w_access=None, half_life=None):
        memories = self.search(
            query, limit=limit, project=project,
            w_fts=w_fts, w_vec=w_vec, w_recency=w_recency,
            w_access=w_access, half_life=half_life,
        )
        conflicts = self.get_conflicts(project=project)

        results = memories.get("results", [])

        # Include one hop of related memories for richer context
        related_ids = set()
        for r in results[:5]:  # limit graph expansion to top-5 for speed
            for link_row in self._execute(
                """SELECT to_id AS id FROM memory_links
                   WHERE from_id = ? AND relation = 'related_to'
                   UNION
                   SELECT from_id AS id FROM memory_links
                   WHERE to_id = ? AND relation = 'related_to'""",
                (r["id"], r["id"]),
            ):
                related_ids.add(link_row["id"])

        related_memories = []
        for rid in related_ids - {r["id"] for r in results}:
            mem = self._fetch_memory(rid)
            if mem:
                related_memories.append(mem)

        if results:
            oldest = min(r["created_at"] for r in results)
            newest = max(r["created_at"] for r in results)
            timeline = (
                f"{len(results)} memories from {oldest} to {newest}"
                + (f" (+{len(related_memories)} related)" if related_memories else "")
            )
        else:
            timeline = "No memories found"

        return {
            "query": query,
            "memories": results,
            "related_memories": related_memories,
            "conflicts": conflicts,
            "timeline_summary": timeline,
            "weights": memories.get("weights", {}),
        }

    # ── Conflicts ─────────────────────────────────────────────────────

    def get_conflicts(self, project=None):
        query = """
            SELECT ml.from_id, m1.content AS from_content,
                   ml.to_id, m2.content AS to_content,
                   ml.relation, ml.created_at
            FROM memory_links ml
            JOIN memories m1 ON m1.id = ml.from_id
            JOIN memories m2 ON m2.id = ml.to_id
            WHERE ml.relation = 'contradicts'
        """
        params = []
        if project:
            query += " AND (m1.project = ? OR m2.project = ?)"
            params.extend([project, project])

        results = []
        for row in self._execute(query, params):
            results.append({
                "memory_id": row["from_id"],
                "content": row["from_content"],
                "related_id": row["to_id"],
                "related_content": row["to_content"],
                "relation": row["relation"],
                "created_at": row["created_at"],
            })
        return results

    # ── Health ────────────────────────────────────────────────────────

    def health(self):
        def _count(sql, params=()):
            row = self._execute(sql, params).fetchone()
            return int(row[0]) if row else 0

        def _first(sql, params=()):
            row = self._execute(sql, params).fetchone()
            return str(row[0]) if row and row[0] else None

        total = _count("SELECT COUNT(*) FROM memories")
        total_links = _count("SELECT COUNT(*) FROM memory_links")
        total_conflicts = _count(
            "SELECT COUNT(*) FROM memory_links WHERE relation = 'contradicts'"
        )
        total_superseded = _count(
            "SELECT COUNT(*) FROM memory_links WHERE relation = 'supersedes'"
        )
        total_duplicates = _count(
            """SELECT COUNT(*) FROM (
                   SELECT content_hash FROM memories
                   WHERE content_hash IS NOT NULL
                   GROUP BY content_hash HAVING COUNT(*) > 1
               )"""
        )
        oldest = _first("SELECT MIN(created_at) FROM memories")
        newest = _first("SELECT MAX(created_at) FROM memories")

        by_type = {}
        for row in self._execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memories GROUP BY memory_type"
        ):
            by_type[row["memory_type"]] = row["cnt"]

        by_project = {}
        for row in self._execute(
            "SELECT COALESCE(project, '<none>') as p, COUNT(*) as cnt "
            "FROM memories GROUP BY project"
        ):
            by_project[row["p"]] = row["cnt"]

        cache_size = len(self._cache._store)

        return {
            "total_memories": total,
            "total_links": total_links,
            "total_conflicts": total_conflicts,
            "total_superseded": total_superseded,
            "active_memories": total - total_superseded,
            "duplicate_content_groups": total_duplicates,
            "oldest_memory": oldest,
            "newest_memory": newest,
            "memories_by_type": by_type,
            "memories_by_project": by_project,
            "cache_entries": cache_size,
        }

    # ── Deduplicate ───────────────────────────────────────────────────

    def deduplicate(self, project=None, dry_run=False):
        """Find memories with identical content_hash and soft-delete duplicates,
        keeping the oldest entry per hash group.

        Returns a summary dict with counts and affected IDs.
        """
        where = "WHERE content_hash IS NOT NULL"
        params: list = []
        if project:
            where += " AND project = ?"
            params.append(project)

        rows = self._execute(
            f"""SELECT content_hash, MIN(created_at) AS keep_ts
                FROM memories {where}
                GROUP BY content_hash HAVING COUNT(*) > 1""",
            params,
        ).fetchall()

        removed_ids = []
        for row in rows:
            chash = row["content_hash"]
            keep_ts = row["keep_ts"]
            dupes = self._execute(
                "SELECT id FROM memories WHERE content_hash = ? AND created_at != ?",
                (chash, keep_ts),
            ).fetchall()
            for d in dupes:
                if not dry_run:
                    self._execute("DELETE FROM memories WHERE id = ?", (d["id"],))
                    self._cache.invalidate(d["id"])
                removed_ids.append(d["id"])

        if not dry_run and removed_ids:
            self.db.commit()

        return {
            "dry_run": dry_run,
            "duplicate_groups": len(rows),
            "removed": len(removed_ids),
            "removed_ids": removed_ids,
        }

    # ── Similar ───────────────────────────────────────────────────────

    def similar(self, memory_id, limit=5):
        mem = self._fetch_memory(memory_id)
        if not mem:
            return {"error": f"Memory {memory_id} not found", "results": []}
        return self.search(mem["content"], limit=limit + 1)

    # ── Topics ────────────────────────────────────────────────────────

    def topics(self, project=None):
        query = """
            SELECT memory_type AS topic, COUNT(*) AS count, MAX(created_at) AS latest
            FROM memories
            WHERE (? IS NULL OR project = ?)
            GROUP BY memory_type ORDER BY count DESC
        """
        results = []
        for row in self._execute(query, (project, project)):
            results.append({
                "topic": row["topic"],
                "count": row["count"],
                "latest": row["latest"],
            })
        return results

    # ── Timeline ──────────────────────────────────────────────────────

    def timeline(self, project=None, start=None, end=None, limit=20):
        query = """
            SELECT * FROM memories
            WHERE (? IS NULL OR project = ?)
              AND (? IS NULL OR created_at >= ?)
              AND (? IS NULL OR created_at <= ?)
            ORDER BY created_at DESC LIMIT ?
        """
        results = []
        for row in self._execute(
            query, (project, project, start, start, end, end, limit)
        ):
            results.append(self._row_to_dict(row))
        return results

    # ── Related ───────────────────────────────────────────────────────

    def related(self, memory_id):
        results = []
        for row in self._execute(
            """SELECT ml.relation, ml.to_id AS id, m.content
               FROM memory_links ml JOIN memories m ON m.id = ml.to_id
               WHERE ml.from_id = ?""",
            (memory_id,),
        ):
            results.append(dict(row))
        for row in self._execute(
            """SELECT ml.relation, ml.from_id AS id, m.content
               FROM memory_links ml JOIN memories m ON m.id = ml.from_id
               WHERE ml.to_id = ?""",
            (memory_id,),
        ):
            results.append(dict(row))
        return results

    # ── Multi-hop Graph Traversal ─────────────────────────────────────

    def multi_hop(self, start_entity, target_entity=None, max_hops=3):
        if not self.kg:
            return {"error": "Knowledge graph not available", "paths": []}

        max_hops = max(1, min(max_hops, 6))
        queue = [(start_entity, [start_entity], [], [])]
        visited = {start_entity.lower()}
        found_paths = []
        all_reachable = {start_entity}
        explored = 0

        while queue and explored < 500:
            current, epath, ppath, fpath = queue.pop(0)
            explored += 1
            if len(epath) - 1 >= max_hops:
                continue

            try:
                facts = self.kg.query_entity(current, direction="both")
            except Exception:
                continue

            for fact in facts:
                subj = fact.get("subject", "")
                pred = fact.get("predicate", "")
                obj = fact.get("object", "")
                if fact.get("ended"):
                    continue

                if subj.lower() == current.lower():
                    neighbor, direction = obj, "→"
                elif obj.lower() == current.lower():
                    neighbor, direction = subj, "←"
                else:
                    continue

                if not neighbor:
                    continue
                all_reachable.add(neighbor)
                if neighbor.lower() in visited:
                    continue
                visited.add(neighbor.lower())

                new_ep = epath + [neighbor]
                new_pp = ppath + [pred]
                label = f"—[{pred}]{direction}" if direction == "→" else f"{direction}[{pred}]—"
                new_fp = fpath + [label, neighbor]

                if target_entity and neighbor.lower() == target_entity.lower():
                    found_paths.append({
                        "hops": len(new_ep) - 1,
                        "path": [epath[0]] + new_fp,
                        "entities": new_ep,
                        "predicates": new_pp,
                    })

                if len(new_ep) - 1 < max_hops:
                    queue.append((neighbor, new_ep, new_pp, new_fp))

        found_paths.sort(key=lambda p: p["hops"])
        result = {
            "start": start_entity,
            "target": target_entity,
            "max_hops": max_hops,
            "paths": found_paths[:20],
            "reachable_entities": len(all_reachable),
            "graph_explored": explored,
        }
        if not target_entity:
            result["reachable"] = sorted(all_reachable - {start_entity})
        return result

    # ── Contradiction Check ───────────────────────────────────────────

    def contradiction_check(self, statement, entity=None):
        entities = [entity] if entity else re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", statement)
        if not entities:
            return {
                "statement": statement,
                "entities_checked": [],
                "contradictions": [],
                "verdict": "no_contradictions",
            }

        contradictions = []
        if self.kg:
            stmt_lower = statement.lower()
            is_negated = any(
                neg in stmt_lower
                for neg in ["not ", "no longer ", "doesn't ", "does not ",
                            "isn't ", "is not ", "removed ", "stopped ",
                            "dropped ", "disabled ", "deprecated "]
            )
            for ent in entities[:10]:
                try:
                    facts = self.kg.query_entity(ent, direction="both")
                except Exception:
                    continue
                for fact in facts:
                    if fact.get("ended"):
                        continue
                    subj = fact.get("subject", "")
                    obj = fact.get("object", "")
                    pred = fact.get("predicate", "")
                    subj_in = subj.lower() in stmt_lower
                    obj_in = obj.lower() in stmt_lower
                    if subj_in and obj_in and is_negated:
                        contradictions.append({
                            "type": "kg",
                            "fact": f"{subj} → {pred} → {obj}",
                            "confidence": 0.85,
                            "reason": (
                                f"Statement negates existing: {subj} → {pred} → {obj}"
                            ),
                        })

        max_conf = max((c["confidence"] for c in contradictions), default=0.0)
        verdict = (
            "strong_contradictions" if max_conf >= 0.8
            else "possible_contradictions" if max_conf >= 0.5
            else "no_contradictions"
        )
        return {
            "statement": statement,
            "entities_checked": entities[:10],
            "contradictions": contradictions[:10],
            "verdict": verdict,
        }

    # ── Fact Check ────────────────────────────────────────────────────

    def fact_check(self, claim):
        entities = re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", claim)
        supporting = []
        confidence = 0.0

        if self.kg:
            claim_lower = claim.lower()
            for ent in entities[:10]:
                try:
                    facts = self.kg.query_entity(ent, direction="both")
                except Exception:
                    continue
                for fact in facts:
                    if fact.get("ended"):
                        continue
                    subj = fact.get("subject", "")
                    pred = fact.get("predicate", "")
                    obj = fact.get("object", "")
                    matches = sum([
                        subj.lower() in claim_lower,
                        pred.lower().replace("_", " ") in claim_lower,
                        obj.lower() in claim_lower,
                    ])
                    if matches >= 3:
                        supporting.append({
                            "type": "kg_exact",
                            "fact": f"{subj} → {pred} → {obj}",
                            "strength": 0.4,
                        })
                        confidence += 0.4
                    elif matches >= 2:
                        supporting.append({
                            "type": "kg_partial",
                            "fact": f"{subj} → {pred} → {obj}",
                            "strength": 0.2,
                        })
                        confidence += 0.2

        # Memory search evidence
        search_result = self.search(claim, limit=5)
        for hit in search_result.get("results", []):
            sim = hit.get("vec_score", 0)
            if sim > 0.7:
                supporting.append({
                    "type": "memory_strong",
                    "text": hit["content"][:200],
                    "similarity": sim,
                    "strength": 0.3,
                })
                confidence += 0.3
            elif sim > 0.4:
                supporting.append({
                    "type": "memory_partial",
                    "text": hit["content"][:200],
                    "similarity": sim,
                    "strength": 0.15,
                })
                confidence += 0.15

        # Check contradictions
        contra = self.contradiction_check(claim)
        contradicting = contra.get("contradictions", [])
        for c in contradicting:
            confidence -= 0.3 * c.get("confidence", 0.5)

        confidence = round(max(0.0, min(1.0, confidence)), 3)
        verdict = (
            "contradicted" if contradicting and confidence < 0.3
            else "supported" if confidence >= 0.6
            else "partially_supported" if confidence >= 0.3
            else "unverified"
        )
        return {
            "claim": claim,
            "confidence": confidence,
            "verdict": verdict,
            "supporting_evidence": supporting[:10],
            "contradicting_evidence": contradicting[:5],
            "entities": entities[:10],
        }

# ── Session Init ──────────────────────────────────────────────────

    def session_init(self, task, project=None, limit=10, w_fts=None,
                     w_vec=None, w_recency=None, half_life=None):
        """One-call session bootstrap: memories + diary + conflicts + timeline."""
        context = self.build_context(
            query=task, project=project, limit=limit,
            w_fts=w_fts, w_vec=w_vec, w_recency=w_recency, half_life=half_life,
        )

        # Recent diary entries
        diary = []
        try:
            from mempalace import mcp_server as mcp
            diary_result = mcp.tool_diary_read(agent_name="hermes", last_n=5)
            diary = diary_result.get("entries", [])
        except Exception:
            pass

        # Recent timeline (last 10 memories regardless of query relevance)
        recent = self.timeline(project=project, limit=10)

        # Health summary
        health = self.health()

        return {
            "task": task,
            "memories": context.get("memories", []),
            "conflicts": context.get("conflicts", []),
            "timeline_summary": context.get("timeline_summary", ""),
            "weights": context.get("weights", {}),
            "diary_entries": diary,
            "recent_memories": recent,
            "health": {
                "total": health.get("total_memories", 0),
                "active": health.get("active_memories", 0),
                "conflicts": health.get("total_conflicts", 0),
            },
        }

    # ── Ingest Turns ──────────────────────────────────────────────────

    def ingest_turns(self, turns, project=None, wing=None, room=None):
        """Compress and store conversation turns as structured memories.

        Each turn is analyzed for durable knowledge. Transient content
        (greetings, acknowledgments, tool invocations) is skipped.
        Discoveries, decisions, and preferences are stored as memories.

        Args:
            turns: list of dicts with 'role' and 'content' keys,
                   or a single string of conversation text.
            project: optional project scope
            wing: defaults to 'project'
            room: defaults to 'conversations'

        Returns summary of what was ingested.
        """
        if isinstance(turns, str):
            turns = [{"role": "mixed", "content": turns}]

        wing = wing or "project"
        room = room or "conversations"
        stored = 0
        skipped = 0

        # Transient content markers — skip turns that are purely mechanical
        skip_markers = [
            "certainly", "sure thing", "of course", "let me ",
            "i'll ", "here's what", "tool call", "tool output",
        ]

        for turn in turns:
            content = turn.get("content", "")
            if not content or len(content.strip()) < 20:
                skipped += 1
                continue

            # Skip purely transient turns
            content_lower = content.lower()[:100]
            if any(m in content_lower for m in skip_markers):
                # But still store if the turn is long (likely has substance)
                if len(content) < 200:
                    skipped += 1
                    continue

            # Check for duplicates before storing
            search_result = self.search(
                query=content[:200], limit=3, wing=wing, room=room,
            )
            existing = search_result.get("results", [])
            is_duplicate = any(
                r.get("score", 0) > 0.85 for r in existing
            )
            if is_duplicate:
                skipped += 1
                continue

            # Determine memory type from content
            mem_type = "fact"
            if turn.get("role") == "user":
                # User messages often contain preferences or instructions
                if any(w in content_lower for w in ["prefer", "always", "never", "don't", "use "]):
                    mem_type = "preference"
                elif any(w in content_lower for w in ["step", "first", "then", "process"]):
                    mem_type = "procedure"
            elif any(w in content_lower for w in ["decided", "chose", "conclusion", "found that"]):
                mem_type = "fact"
            elif any(w in content_lower for w in ["error", "bug", "fix", "issue", "crash"]):
                mem_type = "episode"

            # Truncate very long turns to essential content
            store_content = content[:2000]
            if len(content) > 2000:
                store_content += "\n[...truncated from {} chars]".format(len(content))

            self.store(
                content=store_content,
                memory_type=mem_type,
                project=project,
                wing=wing,
                room=room,
                tags=["auto-ingested", turn.get("role", "unknown")],
            )
            stored += 1

        return {
            "success": True,
            "stored": stored,
            "skipped": skipped,
            "total_turns": len(turns),
            "wing": wing,
            "room": room,
        }
