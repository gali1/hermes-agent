"""Experimental vector/embedding support for the enhanced memory layer.

Feature-flagged via ``memory.enhanced.vector_backend`` (default ``"none"``).
Every backend is optional and lazily imported; any failure raises
:class:`EmbeddingUnavailable`, which the backend facade catches so the agent
degrades to lexical-only retrieval.  The vector arm is additive: it can
surface memories the FTS index misses, but its influence is bounded — the
store clamps the vector weight (``MAX_VECTOR_WEIGHT``) and caps how many
vector-only memories may appear near the top of a fused result list
(``vector_max_share``).

Backends:
  - ``none``      — disabled (default)
  - ``fastembed`` — local ONNX embeddings via the optional ``fastembed`` extra
  - ``remote``    — OpenAI-compatible ``/embeddings`` endpoint via httpx
  - ``mempalace`` — reuse the installed mempalace searcher as a vector arm

Stdlib only at import time.  The heavy/optional dependencies (``fastembed``,
``httpx``, ``mempalace``) are imported inside the backend constructors so the
module can always be imported and missing packages degrade gracefully.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_REMOTE_MODEL = "text-embedding-3-small"
BACKENDS = ("none", "fastembed", "remote", "mempalace")


class EmbeddingUnavailable(RuntimeError):
    """Raised when a configured embedding backend cannot be constructed."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class _FastEmbedProvider:
    """Local embeddings via the optional ``fastembed`` package.

    The package import and model instantiation happen once, lazily and behind
    a lock, so concurrent indexer/search use is safe.  Any import or model
    failure surfaces as :class:`EmbeddingUnavailable`.
    """

    def __init__(self, model: str):
        self.name = "fastembed"
        self.model = model
        self._lock = threading.Lock()
        self._model = None
        self._ensure_model()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(model_name=self.model)
                except Exception as exc:
                    raise EmbeddingUnavailable(f"fastembed unavailable: {exc}") from exc
        return self._model

    def embed(self, texts) -> List[List[float]]:
        try:
            model = self._ensure_model()
            return [[float(value) for value in vector] for vector in model.embed(list(texts))]
        except EmbeddingUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingUnavailable(f"fastembed embedding failed: {exc}") from exc


class _RemoteEmbeddingProvider:
    """OpenAI-compatible embeddings endpoint via the core httpx dependency."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.name = "remote"
        self.model = model
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def embed(self, texts) -> List[List[float]]:
        texts = list(texts)
        try:
            import httpx
        except Exception as exc:
            raise EmbeddingUnavailable(f"httpx unavailable: {exc}") from exc
        try:
            response = httpx.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self.model, "input": texts},
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") or []
            vectors = [list(item["embedding"]) for item in data]
        except Exception as exc:
            raise EmbeddingUnavailable(f"remote embedding failed: {exc}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable("remote embedding response size mismatch")
        try:
            return [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as exc:
            raise EmbeddingUnavailable(f"remote embedding malformed: {exc}") from exc


# ---------------------------------------------------------------------------
# Vector index (search over stored embeddings)
# ---------------------------------------------------------------------------


class LocalVectorIndex:
    """Cosine-similarity search over embeddings persisted in a MemoryStore.

    Pure Python on purpose: the store is stdlib-only and the memory sizes here
    are small.  Every failure path returns an empty result envelope so the
    vector arm can never break a turn.
    """

    def __init__(self, store, provider, model: str):
        self._store = store
        self._provider = provider
        self._model = model

    def indexed_count(self) -> int:
        try:
            return int(self._store.embedding_count(self._model))
        except Exception:
            return 0

    def search(self, query=None, limit=10, **kwargs) -> Dict[str, Any]:
        if not query or not str(query).strip() or self._store is None:
            return {"results": []}
        try:
            vectors = self._provider.embed([str(query)])
            if not vectors:
                return {"results": []}
            query_vector = [float(value) for value in vectors[0]]
            query_norm = math.sqrt(sum(value * value for value in query_vector))
            if query_norm == 0.0:
                return {"results": []}

            stored = self._store.load_embeddings(self._model)
            scored: List[Tuple[float, str]] = []
            for memory_id, vector in stored.items():
                norm = math.sqrt(sum(value * value for value in vector))
                if norm == 0.0:
                    continue
                dot = sum(a * b for a, b in zip(query_vector, vector))
                similarity = max(-1.0, min(1.0, dot / (query_norm * norm)))
                scored.append((similarity, memory_id))
            scored.sort(key=lambda item: item[0], reverse=True)

            try:
                top_n = max(1, int(limit))
            except (TypeError, ValueError):
                top_n = 10
            results = []
            for similarity, memory_id in scored[:top_n]:
                memory = self._store.get(memory_id, track_access=False)
                if not memory:
                    continue
                results.append({
                    "id": memory_id,
                    "text": memory.get("content") or "",
                    "distance": 1.0 - similarity,
                })
            return {"results": results}
        except Exception:
            logger.debug("vector search failed (non-fatal)", exc_info=True)
            return {"results": []}


# ---------------------------------------------------------------------------
# Background indexer
# ---------------------------------------------------------------------------

_SENTINEL = object()


class EmbeddingIndexer:
    """Background worker that keeps stored embeddings in sync.

    The queue is only a wake-up hint: the worker always scans
    ``missing_embedding_ids`` so a dropped or stale id can never wedge the
    index.  All state is failure-contained — the worker never raises into the
    caller and a provider failure is recorded in ``last_error``.
    """

    def __init__(self, store, provider, model: str, batch_size: int = 16):
        self._store = store
        self._provider = provider
        self._model = model
        self._batch_size = max(1, int(batch_size or 16))
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._indexed_total = 0
        self._last_error: Optional[str] = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._running = True
            self._thread = threading.Thread(
                target=self._worker, name="memory-embed-indexer", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5) -> None:
        self._stop.set()
        try:
            self._queue.put(_SENTINEL)
        except Exception:
            logger.debug("embedding sentinel enqueue failed (non-fatal)", exc_info=True)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._running = False
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None

    def enqueue(self, memory_ids=None) -> None:
        try:
            if memory_ids is None:
                self._queue.put(None)
            else:
                self._queue.put(list(memory_ids))
        except Exception:
            logger.debug("embedding enqueue failed (non-fatal)", exc_info=True)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            running = bool(
                self._running and self._thread is not None and self._thread.is_alive()
            )
            return {
                "provider": getattr(self._provider, "name", "unknown"),
                "model": self._model,
                "running": running,
                "queued": self._queue.qsize(),
                "indexed_total": self._indexed_total,
                "last_error": self._last_error,
            }

    # -- Worker -------------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                item = None
            else:
                if item is _SENTINEL:
                    break
            self._drain_queue()
            self._scan_once()
        with self._lock:
            self._running = False

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _scan_once(self) -> None:
        if self._stop.is_set():
            return
        try:
            missing = self._store.missing_embedding_ids(
                self._model, limit=self._batch_size
            )
        except Exception as exc:
            self._record_error(exc)
            return
        if not missing:
            return
        try:
            vectors = self._provider.embed([content for _, content in missing])
        except Exception as exc:
            self._record_error(exc)
            return
        for (memory_id, _), vector in zip(missing, vectors):
            if self._stop.is_set():
                break
            try:
                if self._store.set_embedding(memory_id, vector, self._model):
                    with self._lock:
                        self._indexed_total += 1
            except Exception as exc:
                self._record_error(exc)

    def _record_error(self, exc: Exception) -> None:
        with self._lock:
            self._last_error = str(exc) or exc.__class__.__name__
        logger.debug("memory embedding indexing failed (non-fatal)", exc_info=True)


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


def _as_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def build_vector_support(
    config, store
) -> Tuple[Optional[Callable], Optional[EmbeddingIndexer]]:
    """Build the configured vector arm.

    Returns ``(search_fn, indexer)``.  ``search_fn`` matches the store's
    ``vector_search_fn(query=..., limit=...)`` contract; ``indexer`` is the
    background embedder (``None`` when the backend has no local index, e.g.
    ``mempalace``).  Raises :class:`EmbeddingUnavailable` when the selected
    backend cannot be constructed — the caller then degrades to lexical-only
    retrieval.
    """
    config = config or {}
    backend = str(config.get("vector_backend") or "none").strip().lower()

    if backend in ("", "none"):
        return None, None

    if backend == "fastembed":
        model = str(config.get("vector_model") or DEFAULT_FASTEMBED_MODEL)
        provider = _FastEmbedProvider(model)
        index = LocalVectorIndex(store, provider, model)
        indexer = EmbeddingIndexer(
            store, provider, model,
            batch_size=_as_int(config.get("vector_index_batch"), 16),
        )
        return index.search, indexer

    if backend == "remote":
        remote = config.get("vector_remote") or {}
        base_url = str(remote.get("base_url") or "https://api.openai.com/v1")
        api_key_env = str(remote.get("api_key_env") or "OPENAI_API_KEY")
        model = str(remote.get("model") or DEFAULT_REMOTE_MODEL)
        try:
            timeout = float(remote.get("timeout") or 30)
        except (TypeError, ValueError):
            timeout = 30.0
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EmbeddingUnavailable(f"missing embedding API key in ${api_key_env}")
        provider = _RemoteEmbeddingProvider(base_url, api_key, model, timeout)
        index = LocalVectorIndex(store, provider, model)
        indexer = EmbeddingIndexer(
            store, provider, model,
            batch_size=_as_int(config.get("vector_index_batch"), 16),
        )
        return index.search, indexer

    if backend == "mempalace":
        try:
            from mempalace.searcher import search_memories
        except Exception as exc:
            raise EmbeddingUnavailable(f"mempalace unavailable: {exc}") from exc

        def _search(query=None, limit=10, **kwargs):
            try:
                raw = search_memories(query=query, limit=limit)
            except Exception:
                logger.debug("mempalace vector search failed (non-fatal)", exc_info=True)
                return {"results": []}
            if isinstance(raw, dict):
                raw_results = raw.get("results") or []
            elif isinstance(raw, (list, tuple)):
                raw_results = raw
            else:
                raw_results = []
            results = []
            for hit in raw_results:
                if not isinstance(hit, dict):
                    continue
                text = hit.get("text") or hit.get("content") or ""
                try:
                    distance = 1.0 - float(hit.get("similarity", 0.0))
                except (TypeError, ValueError):
                    distance = 1.0
                results.append({"text": text, "distance": distance})
            return {"results": results}

        return _search, None

    logger.debug("unknown vector backend %r; vector support disabled", backend)
    return None, None
