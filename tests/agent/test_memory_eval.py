"""Tests for the recall@k eval harness (agent/memory/eval.py)."""

import json
import math
import sqlite3

from agent.memory.eval import (
    build_queries,
    build_store_from_messages,
    format_scores,
    load_session_messages,
    run_eval,
)


class _BagEmbedder:
    """Deterministic bag-of-token embedder: near-duplicate text -> close vectors."""

    name = "fake"
    model = "fake-bag-32"

    def embed(self, texts):
        vectors = []
        for text in texts:
            vec = [0.0] * 32
            for token in (text or "").lower().split():
                vec[hash(token) % 32] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


def _make_db(tmp_path):
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, timestamp REAL)"
    )
    rows = [
        ("s1", "user", "The deployment pipeline uses blue-green releases for every service"),
        ("s1", "assistant", "Understood."),
        ("s1", "user", "We decided to use PostgreSQL 17 for the analytics database"),
        ("s2", "user", "The gateway requires mutual TLS on all inbound traffic"),
    ]
    for i, (sid, role, content) in enumerate(rows):
        con.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (sid, role, content, 1000.0 + i),
        )
    con.commit()
    con.close()
    return db


def test_load_session_messages(tmp_path):
    db = _make_db(tmp_path)
    messages = load_session_messages(str(db))
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "user"]
    assert messages[0]["content"].startswith("The deployment pipeline")

    only_s2 = load_session_messages(str(db), session_id="s2")
    assert len(only_s2) == 1


def test_build_queries_covers_kinds(tmp_path):
    store, memories = build_store_from_messages(
        load_session_messages(str(_make_db(tmp_path))), str(tmp_path / "work")
    )
    assert memories
    queries = build_queries(memories)
    kinds = {q.kind for q in queries}
    assert {"exact", "keywords", "concept"} <= kinds
    assert all(q.query.strip() for q in queries)
    assert all(q.memory_id for q in queries)
    store.close()


def test_run_eval_lexical_and_vector(tmp_path):
    db = _make_db(tmp_path)
    store, memories = build_store_from_messages(
        load_session_messages(str(db)), str(tmp_path / "work")
    )
    queries = build_queries(memories)

    lexical = run_eval(store, queries, k=5)
    assert lexical
    exact = [s for s in lexical if s.mode == "fts" and s.kind == "exact"]
    assert exact and exact[0].recall_at_1 == 1.0

    provider = _BagEmbedder()
    vectors = {}
    for memory in memories:
        vec = provider.embed([memory["content"]])[0]
        store.set_embedding(memory["id"], vec, provider.model)
        vectors[memory["id"]] = vec

    def fake_search(query=None, limit=10, **kwargs):
        query_vec = provider.embed([query or ""])[0]
        ranked = sorted(
            ({"id": mid, "text": "", "distance": 1.0 - _cos(query_vec, vec)}
             for mid, vec in vectors.items()),
            key=lambda hit: hit["distance"],
        )
        return {"results": ranked[:limit]}

    vector_scores = run_eval(store, queries, k=5, vector_search_fn=fake_search)
    modes = {s.mode for s in vector_scores}
    assert "vector" in modes and "hybrid+vector" in modes
    assert all(0.0 <= s.recall_at_k <= 1.0 and 0.0 <= s.mrr <= 1.0 for s in vector_scores)
    store.close()


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def test_format_scores_hides_content(tmp_path):
    db = _make_db(tmp_path)
    store, memories = build_store_from_messages(
        load_session_messages(str(db)), str(tmp_path / "work")
    )
    table = format_scores(run_eval(store, build_queries(memories), k=3))
    assert "fts" in table and "hybrid" in table
    assert "blue-green" not in table
    store.close()


def test_cli_smoke(tmp_path, capsys):
    import scripts.memory_eval as cli

    db = _make_db(tmp_path)
    code = cli.main(["--db", str(db), "--json", "--limit", "50"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload
    assert {row["mode"] for row in payload} >= {"fts", "hybrid"}
