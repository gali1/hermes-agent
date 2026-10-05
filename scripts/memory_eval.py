#!/usr/bin/env python3
"""Recall@k evaluation for the enhanced memory layer (experimental).

Reads real session messages from a Hermes state.db, mines durable candidates,
and measures recall@1 / recall@k / MRR for lexical, hybrid, vector and
hybrid+vector retrieval.  Memory contents are never printed.

Usage:
    python scripts/memory_eval.py                       # ~/.hermes/state.db
    python scripts/memory_eval.py --limit 800 --k 5
    python scripts/memory_eval.py --vector-backend fastembed
    python scripts/memory_eval.py --json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory.eval import (  # noqa: E402
    build_queries,
    build_store_from_messages,
    format_scores,
    load_session_messages,
    run_eval,
)


def _default_db_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state.db"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recall@k eval for enhanced memory")
    parser.add_argument("--db", default="", help="Path to a Hermes state.db (default: $HERMES_HOME/state.db)")
    parser.add_argument("--session", default="", help="Only evaluate messages from this session id")
    parser.add_argument("--limit", type=int, default=600, help="Max messages to sample (default 600)")
    parser.add_argument("--mining-limit", type=int, default=200, help="Max mined memories (default 200)")
    parser.add_argument("--k", type=int, default=5, help="Recall@k cutoff (default 5)")
    parser.add_argument("--vector-backend", default="none",
                        choices=["none", "fastembed", "remote", "mempalace"])
    parser.add_argument("--vector-model", default="", help="Override the provider's default model")
    parser.add_argument("--json", action="store_true", help="Emit raw scores as JSON")
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else _default_db_path()
    if not db_path.exists():
        print(f"✗ session database not found: {db_path}")
        print("  Pass --db /path/to/state.db or set HERMES_HOME.")
        return 1

    messages = load_session_messages(str(db_path), limit=args.limit, session_id=args.session or None)
    if not messages:
        print(f"✗ no user/assistant messages found in {db_path}")
        return 1

    workdir = tempfile.mkdtemp(prefix="hermes-memory-eval-")
    vector_search_fn = None
    indexer = None
    try:
        store, memories = build_store_from_messages(
            messages, workdir, mining_limit=args.mining_limit
        )
        if not memories:
            print("✗ no durable memories mined from the sampled messages")
            return 1

        if args.vector_backend != "none":
            try:
                from agent.memory.embeddings import EmbeddingUnavailable, build_vector_support

                config = {"vector_backend": args.vector_backend}
                if args.vector_model:
                    config["vector_model"] = args.vector_model
                vector_search_fn, indexer = build_vector_support(config, store)
                if vector_search_fn is None:
                    print(f"⚠ vector backend '{args.vector_backend}' unavailable; lexical modes only")
                elif indexer is not None:
                    indexer.start()
                    indexer.enqueue()
                    model = indexer.stats().get("model", "")
                    deadline = time.time() + 120
                    while time.time() < deadline:
                        if store.missing_embedding_count(model) == 0:
                            break
                        time.sleep(0.2)
                    missing = store.missing_embedding_count(model)
                    print(f"  indexed {store.embedding_count(model)}/{len(memories)} memories"
                          + (f" ({missing} pending)" if missing else ""))
                else:
                    print(f"  using external vector backend '{args.vector_backend}'")
            except EmbeddingUnavailable as exc:
                print(f"⚠ vector backend unavailable ({exc}); lexical modes only")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"⚠ vector backend failed ({exc}); lexical modes only")

        queries = build_queries(memories)
        if not queries:
            print("✗ could not derive any queries from the mined memories")
            return 1

        scores = run_eval(store, queries, k=args.k, vector_search_fn=vector_search_fn)
        if args.json:
            print(json.dumps([asdict(score) for score in scores], indent=2))
        else:
            print(f"messages={len(messages)}  memories={len(memories)}  queries={len(queries)}  k={args.k}")
            print()
            print(format_scores(scores))
            print()
            print("Note: metrics only — no memory contents are printed. "
                  "Treat this as a smoke eval, not a benchmark.")
        return 0
    finally:
        if indexer is not None:
            try:
                indexer.stop()
            except Exception:
                pass
        try:
            store.close()
        except Exception:
            pass
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
