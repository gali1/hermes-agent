"""Tests for the experimental vector/embedding memory support.

Hermetic: stdlib + pytest only.  The optional ``fastembed`` package is never
imported — its absence is simulated through ``sys.modules`` — and no test
touches the network.
"""

import hashlib
import math
import sys
import time

import pytest

from agent.memory.backend import EnhancedMemoryBackend
from agent.memory.embeddings import (
    EmbeddingIndexer,
    EmbeddingUnavailable,
    LocalVectorIndex,
    build_vector_support,
)
from agent.memory.store import MemoryStore

DIM = 8


def _hash_vector(text):
    """Deterministic bag-of-words signed-hash embedding (unit norm).

    Identical text yields identical vectors; near-duplicate text (one word
    changed) yields a close vector, so cosine distance is meaningful.
    """
    vector = [0.0] * DIM
    for word in str(text).lower().split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        index = digest[0] % DIM
        sign = 1.0 if digest[1] % 2 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class FakeProvider:
    """Deterministic fake embedding provider (no optional deps)."""

    name = "fake"
    model = "fake-v1"

    def __init__(self, fail=False):
        self.fail = fail
        self.embed_calls = []

    def embed(self, texts):
        texts = list(texts)
        self.embed_calls.append(texts)
        if self.fail:
            raise RuntimeError("provider exploded")
        return [_hash_vector(text) for text in texts]


class MappedProvider:
    """Provider returning caller-supplied vectors for exact texts."""

    name = "mapped"
    model = "mapped-v1"

    def __init__(self, mapping, default=None):
        self.mapping = dict(mapping)
        self.default = list(default) if default is not None else [0.0] * DIM

    def embed(self, texts):
        return [list(self.mapping.get(text, self.default)) for text in texts]


@pytest.fixture
def store(tmp_path):
    return MemoryStore(str(tmp_path / "memories.db"))


def _embed_all(store, provider, texts):
    ids = [store.store(text)["memory_id"] for text in texts]
    for memory_id, vector in zip(ids, provider.embed(texts)):
        assert store.set_embedding(memory_id, vector, provider.model) is True
    return ids


# ── Ladder selection ───────────────────────────────────────────────────────

def test_none_backend_returns_no_support(store):
    assert build_vector_support({"vector_backend": "none"}, store) == (None, None)
    assert build_vector_support({}, store) == (None, None)
    assert build_vector_support({"vector_backend": "bogus"}, store) == (None, None)


def test_fastembed_missing_raises(monkeypatch, store):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(EmbeddingUnavailable):
        build_vector_support({"vector_backend": "fastembed"}, store)


def test_remote_missing_key_raises(monkeypatch, store):
    monkeypatch.delenv("HERMES_TEST_MISSING_KEY", raising=False)
    with pytest.raises(EmbeddingUnavailable):
        build_vector_support(
            {
                "vector_backend": "remote",
                "vector_remote": {"api_key_env": "HERMES_TEST_MISSING_KEY"},
            },
            store,
        )


# ── Store embedding lifecycle ──────────────────────────────────────────────

def test_embedding_lifecycle(store):
    memory_id = store.store("The parser handles nested expressions")["memory_id"]
    vector = [0.1, 0.2, 0.3, 0.4]
    assert store.set_embedding(memory_id, vector, "m1") is True

    assert store.embedding_count() == 1
    assert store.embedding_count("m1") == 1
    assert store.embedding_count("other") == 0

    stored_vector, model, dim = store.get_embedding(memory_id)
    assert model == "m1"
    assert dim == 4
    assert stored_vector == pytest.approx(vector, abs=1e-6)

    loaded = store.load_embeddings("m1")
    assert list(loaded) == [memory_id]
    assert loaded[memory_id] == pytest.approx(vector, abs=1e-6)
    assert store.load_embeddings("other") == {}

    assert store.missing_embedding_count("m1") == 0
    assert store.missing_embedding_count(None) == 0
    assert store.missing_embedding_ids("m1") == []

    second = store.store("Another memory about indexing")["memory_id"]
    assert store.missing_embedding_count("m1") == 1
    assert store.missing_embedding_count(None) == 1
    assert [row[0] for row in store.missing_embedding_ids("m1")] == [second]

    assert store.delete_embedding(memory_id) is True
    assert store.get_embedding(memory_id) is None
    assert store.embedding_count() == 0
    assert store.missing_embedding_count(None) == 2


def test_embedding_cascade_and_supersede(store):
    first = store.store("Original content about caching")["memory_id"]
    assert store.set_embedding(first, [1.0, 0.0], "m1") is True
    assert store.embedding_count() == 1

    # Deleting the memory cascades the embedding row away.
    assert store.delete(first)["success"] is True
    assert store.get_embedding(first) is None
    assert store.embedding_count() == 0

    old = store.store("A fact that will be replaced")["memory_id"]
    assert store.set_embedding(old, [0.5, 0.5], "m1") is True
    result = store.supersede(old, "The replacement fact")
    assert result.get("new_id")
    new_id = result["new_id"]

    # The superseded row keeps its embedding; the new row starts missing one.
    assert store.get_embedding(old) is not None
    assert store.get_embedding(new_id) is None
    assert new_id in [row[0] for row in store.missing_embedding_ids("m1")]


# ── LocalVectorIndex ───────────────────────────────────────────────────────

def test_local_vector_index_search(store):
    provider = FakeProvider()
    texts = [
        "SQLite WAL mode keeps readers unblocked",
        "Exponential backoff retries transient failures",
        "OAuth tokens expire after thirty days",
    ]
    ids = _embed_all(store, provider, texts)
    index = LocalVectorIndex(store, provider, provider.model)
    assert index.indexed_count() == 3

    results = index.search(query=texts[1], limit=3)["results"]
    assert results
    assert results[0]["id"] == ids[1]
    assert results[0]["distance"] < 0.01
    assert results[0]["text"] == texts[1]

    assert index.search(query="", limit=3) == {"results": []}
    assert index.search(query=None, limit=3) == {"results": []}


def test_local_vector_index_broken_provider(store):
    provider = FakeProvider(fail=True)
    store.store("anything at all")
    index = LocalVectorIndex(store, provider, provider.model)
    assert index.search(query="anything", limit=3) == {"results": []}


# ── EmbeddingIndexer ───────────────────────────────────────────────────────

def test_embedding_indexer_indexes_missing(store):
    provider = FakeProvider()
    for i in range(3):
        store.store(f"Durable fact number {i} about indexing")

    indexer = EmbeddingIndexer(store, provider, provider.model, batch_size=2)
    indexer.start()
    try:
        indexer.enqueue()
        deadline = time.time() + 5
        while time.time() < deadline and indexer.stats()["indexed_total"] < 3:
            time.sleep(0.05)

        stats = indexer.stats()
        assert stats["indexed_total"] == 3
        assert stats["running"] is True
        assert stats["provider"] == "fake"
        assert store.embedding_count(provider.model) == 3
        assert store.missing_embedding_count(provider.model) == 0
    finally:
        indexer.stop()
    assert indexer.stats()["running"] is False


# ── Integration with MemoryStore.search ────────────────────────────────────

def test_vector_search_finds_non_lexical_memory(store):
    provider = MappedProvider({
        "orbital mechanics": [1.0, 0.0, 0.0],
        "satellite calibration procedure": [0.99, 0.05, 0.0],
    })
    memory_id = store.store("satellite calibration procedure")["memory_id"]
    store.set_embedding(
        memory_id,
        provider.embed(["satellite calibration procedure"])[0],
        provider.model,
    )
    index = LocalVectorIndex(store, provider, provider.model)

    envelope = store.search("orbital mechanics", limit=5, vector_search_fn=index.search)
    ids = [row["id"] for row in envelope["results"]]
    assert memory_id in ids
    hit = next(row for row in envelope["results"] if row["id"] == memory_id)
    assert hit["fts_score"] == 0.0
    assert hit["vec_score"] > 0


def test_vector_share_cap_limits_vec_only_promotion(store):
    store.set_vector_max_share(0.5)
    query = "zephyr"
    query_vector = [1.0] + [0.0] * (DIM - 1)
    near = [0.98] + [0.05] * (DIM - 1)

    lexical_texts = [
        "zephyr pipeline handles ingestion",
        "zephyr scheduler coordinates workers",
    ]
    vector_only_texts = [
        "quasar telemetry archive",
        "nebula archive rollup",
        "pulsar archive sharding",
        "comet archive retention",
    ]
    mapping = {query: query_vector}
    for text in lexical_texts + vector_only_texts:
        mapping[text] = near
    provider = MappedProvider(mapping)
    _embed_all(store, provider, lexical_texts + vector_only_texts)
    index = LocalVectorIndex(store, provider, provider.model)

    envelope = store.search(query, limit=10, fusion="rrf", vector_search_fn=index.search)
    results = envelope["results"]
    assert len(results) == len(lexical_texts) + len(vector_only_texts)

    def is_vec_only(entry):
        ranks = entry.get("source_ranks") or {}
        return bool(ranks) and set(ranks.keys()) == {"vec_rank"}

    flags = [is_vec_only(entry) for entry in results]
    assert sum(flags) >= len(vector_only_texts)

    cap = max(1, int(len(results) * 0.5))
    seen = 0
    deferred_index = None
    for index_pos, flag in enumerate(flags):
        if flag:
            seen += 1
            if seen > cap:
                deferred_index = index_pos
                break

    # There are more vector-only entries than the cap allows, so the excess
    # must have been deferred past every non-vector-only entry.
    assert deferred_index is not None
    last_non_vec = max((i for i, flag in enumerate(flags) if not flag), default=-1)
    assert last_non_vec < deferred_index
    assert sum(flags[: cap + 1]) <= cap


# ── Backend facade ─────────────────────────────────────────────────────────

def test_backend_degrades_when_fastembed_absent(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    backend = EnhancedMemoryBackend(
        str(tmp_path), {"enabled": True, "vector_backend": "fastembed"}
    )
    backend.initialize("s1", platform="cli")

    assert backend.available is True
    assert backend._embedding_indexer is None
    assert backend._vector_backend is None

    backend.observe_turn("The deployment uses blue-green releases", "ok")
    ctx = backend.prefetch("blue-green releases")
    assert "blue-green" in ctx

    health = backend.health()
    assert health["embedding"] == {"provider": "none"}
    backend.shutdown()


def test_no_vector_backend_keeps_lexical_behavior(tmp_path):
    backend = EnhancedMemoryBackend(str(tmp_path), {"enabled": True})
    backend.initialize("s1")
    backend.observe_turn("I prefer concise answers and absolute paths", "Noted.")

    ctx = backend.prefetch("concise answers")
    assert "concise" in ctx

    envelope = backend.search("concise answers")
    assert envelope["results"]
    assert all(row.get("vec_score", 0.0) == 0.0 for row in envelope["results"])

    health = backend.health()
    assert health["embedding"]["provider"] == "none"
    assert health["embeddings_indexed"] == 0
    assert health["embeddings_missing"] >= 1
    backend.shutdown()
