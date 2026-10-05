"""Recall@k evaluation harness for the enhanced memory layer (experimental).

Measures retrieval quality on real session data: it reads user/assistant
messages from a Hermes ``state.db``, mines durable candidates, stores them in a
throwaway memory store, derives queries from each memory, and reports recall@1,
recall@k and MRR per retrieval mode (lexical, hybrid, vector, hybrid+vector).

Privacy: memory contents are never printed — only aggregate metrics.  The
throwaway store is created under a caller-supplied temp directory.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent.memory.entities import extract_entities
from agent.memory.mining import mine_turns
from agent.memory.store import MemoryStore

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "about", "after", "again", "against", "because", "been", "before", "being",
    "between", "could", "does", "doing", "down", "during", "each", "from",
    "further", "have", "having", "here", "into", "itself", "more", "most",
    "only", "other", "over", "same", "should", "some", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "under", "until", "very", "were", "what", "when", "where",
    "which", "while", "with", "would", "your", "yours", "user", "users",
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]{2,}")


@dataclass
class EvalQuery:
    memory_id: str
    query: str
    kind: str


@dataclass
class ModeScore:
    mode: str
    kind: str
    n: int
    recall_at_1: float
    recall_at_k: float
    mrr: float


def load_session_messages(db_path: str, limit: int = 2000,
                          session_id: Optional[str] = None) -> List[Dict[str, str]]:
    """Read the most recent real user/assistant messages, oldest first."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sql = (
            "SELECT role, content FROM messages "
            "WHERE role IN ('user', 'assistant') AND content IS NOT NULL AND content != ''"
        )
        params: list = []
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(int(limit))
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


def _content_words(text: str, limit: int = 6) -> List[str]:
    words = [w for w in _TOKEN_RE.findall(text or "") if w.lower() not in _STOPWORDS]
    words.sort(key=len, reverse=True)
    seen: List[str] = []
    for word in words:
        if word.lower() not in {s.lower() for s in seen}:
            seen.append(word)
        if len(seen) >= limit:
            break
    return seen


def _topic(memory: Dict[str, Any]) -> str:
    content = memory.get("content") or ""
    entities = [e["text"] for e in extract_entities(content, limit=1)]
    if entities:
        return entities[0]
    words = _content_words(content, 3)
    return " ".join(words)


def _type_question(memory_type: Optional[str], topic: str) -> str:
    return {
        "preference": f"what are the user's preferences about {topic}?",
        "decision": f"what was decided about {topic}?",
        "procedure": f"how do we handle {topic}?",
        "bug": f"what issue involved {topic}?",
        "solution": f"how was {topic} fixed?",
    }.get(str(memory_type or ""), f"what do we know about {topic}?")


def build_queries(memories: Sequence[Dict[str, Any]]) -> List[EvalQuery]:
    """Derive deterministic query forms from each stored memory."""
    queries: List[EvalQuery] = []
    for memory in memories:
        memory_id = memory.get("id")
        content = (memory.get("content") or "").strip()
        if not memory_id or not content:
            continue

        exact = " ".join(content.split()[:8]).strip()
        if exact:
            queries.append(EvalQuery(memory_id, exact, "exact"))

        keywords = _content_words(content, 4)
        if keywords:
            queries.append(EvalQuery(memory_id, " ".join(keywords), "keywords"))

        entities = [e["text"] for e in extract_entities(content, limit=2)]
        if entities:
            queries.append(EvalQuery(
                memory_id, f"what do we know about {' and '.join(entities)}?", "entity"
            ))

        topic = _topic(memory)
        if topic:
            queries.append(EvalQuery(
                memory_id, _type_question(memory.get("memory_type"), topic), "concept"
            ))
    return queries


def _rank_of(hits: Sequence[Dict[str, Any]], memory_id: str) -> int:
    for index, hit in enumerate(hits or []):
        if hit.get("id") == memory_id:
            return index + 1
    return 0


def run_eval(store: MemoryStore, queries: Sequence[EvalQuery], k: int = 5,
             vector_search_fn: Optional[Callable[..., Any]] = None) -> List[ModeScore]:
    """Run every query in each available mode and aggregate recall/MRR."""
    buckets: Dict[Tuple[str, str], Dict[str, float]] = {}

    def _record(mode: str, kind: str, rank: int) -> None:
        bucket = buckets.setdefault((mode, kind), {"n": 0, "hit1": 0, "hitk": 0, "rr": 0.0})
        bucket["n"] += 1
        if rank == 1:
            bucket["hit1"] += 1
        if rank and rank <= k:
            bucket["hitk"] += 1
        if rank:
            bucket["rr"] += 1.0 / rank

    for query in queries:
        mode_hits: Dict[str, List[Dict[str, Any]]] = {
            "fts": store.search(query.query, limit=k, fusion=None).get("results", []),
            "hybrid": store.search(
                query.query, limit=k, fusion="rrf", graph_expand=True, temporal=True
            ).get("results", []),
        }
        if vector_search_fn is not None:
            mode_hits["hybrid+vector"] = store.search(
                query.query, limit=k, fusion="rrf", graph_expand=True, temporal=True,
                vector_search_fn=vector_search_fn,
            ).get("results", [])
            raw = vector_search_fn(query=query.query, limit=k) or {}
            mode_hits["vector"] = [
                {"id": hit.get("id"), "content": hit.get("text", "")}
                for hit in raw.get("results", [])
            ]
        for mode, hits in mode_hits.items():
            _record(mode, query.kind, _rank_of(hits, query.memory_id))

    scores: List[ModeScore] = []
    for (mode, kind), bucket in sorted(buckets.items()):
        n = max(1, int(bucket["n"]))
        scores.append(ModeScore(
            mode=mode,
            kind=kind,
            n=int(bucket["n"]),
            recall_at_1=bucket["hit1"] / n,
            recall_at_k=bucket["hitk"] / n,
            mrr=bucket["rr"] / n,
        ))
    return scores


def build_store_from_messages(messages: Sequence[Dict[str, str]], data_dir: str,
                              mining_limit: int = 200) -> Tuple[MemoryStore, List[Dict[str, Any]]]:
    """Mine durable candidates from messages and load them into a fresh store."""
    store = MemoryStore(os.path.join(data_dir, "eval_memories.db"))
    memories: List[Dict[str, Any]] = []
    for candidate in mine_turns(messages, limit=mining_limit):
        content = (candidate.get("content") or "").strip()
        if not content:
            continue
        result = store.store(
            content,
            memory_type=candidate.get("memory_type") or "fact",
            source={"type": "eval"},
        )
        if result.get("success"):
            memories.append({
                "id": result["memory_id"],
                "content": content,
                "memory_type": candidate.get("memory_type") or "fact",
            })
    return store, memories


def format_scores(scores: Sequence[ModeScore]) -> str:
    """Render an aggregate table; no memory contents."""
    if not scores:
        return "no queries evaluated"
    kinds = sorted({score.kind for score in scores})
    modes = sorted({score.mode for score in scores})
    by_key = {(score.mode, score.kind): score for score in scores}

    width = max(len(kind) for kind in kinds + ["overall"])
    header = f"{'query kind'.ljust(width)} | " + " | ".join(
        f"{mode} (n, R@1, R@k, MRR)".ljust(28) for mode in modes
    )
    lines = [header, "-" * len(header)]
    for kind in kinds + ["overall"]:
        cells = []
        for mode in modes:
            matching = [s for s in scores if s.mode == mode and (kind == "overall" or s.kind == kind)]
            total_n = sum(s.n for s in matching)
            if not total_n:
                cells.append("-".ljust(28))
                continue
            r1 = sum(s.recall_at_1 * s.n for s in matching) / total_n
            rk = sum(s.recall_at_k * s.n for s in matching) / total_n
            mrr = sum(s.mrr * s.n for s in matching) / total_n
            cells.append(f"{total_n:>4}  {r1:.3f}  {rk:.3f}  {mrr:.3f}".ljust(28))
        lines.append(f"{kind.ljust(width)} | " + " | ".join(cells))
    return "\n".join(lines)
