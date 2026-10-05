# Changelog

All notable changes to this project are documented here. Entries follow
GitHub release-note style: feature, change, test, documentation and
installation categories with a compare link per release.

## 2026-10-05 — Enhanced memory layer, upstream Rekal plugin & install docs

**Full Changelog**:
[`4d7f8ade3...98ac7de2b`](https://github.com/gali1/hermes-agent/compare/4d7f8ade3...98ac7de2b)

Strictly additive: every existing memory capability (curated `MEMORY.md` /
`USER.md`, the `memory` tool add/replace/remove/batch, external memory
providers, prefetch/sync/session-end/pre-compress lifecycle, profile
isolation, prompt-cache layering, context-engine sanitization, skill
scaffolding stripping) is preserved and behaves identically unless the new
layer is explicitly enabled.

### ✨ Features

- **Additive enhanced memory layer** (`agent/memory/`, opt-in via
  `memory.enhanced.enabled`, data in the profile-scoped
  `$HERMES_HOME/<memory.enhanced.data_dir>/`):
  - Typed records: `MemoryRecord`, `MemoryQuery`, `MemoryResult`,
    `MemoryScope`, `MemorySource`, `MemoryEvidence`, `MemoryLink`,
    `MemoryConflict`, plus extensible type/scope/source constants.
  - Retrieval intelligence: Reciprocal Rank Fusion, rank-space strategy
    boosts, bounded multiplicative scoring, linear/exponential recency decay,
    coarse-date handling, `proof_norm`.
  - Temporal reasoning: rule-based query analysis (14 patterns), lexical
    rewrite that preserves the original query, temporal proximity/coverage.
  - Graph: bounded spreading activation with relation boosts, cycle
    protection, budget and activation threshold.
  - Quality: trigram/normalized dedup, degenerate filtering,
    negation/polarity contradiction detection, bounded evidence confidence,
    rule-based entity extraction, turn mining with secret/`<private>`/
    transient filtering, budgeted recall formatting.
  - Storage: stdlib SQLite + FTS5 with typed lifecycle
    (store/update/supersede/delete/link/unlink/reinforce), idempotent
    migrations, FTS self-heal, `proof_count`/`last_reinforced_at` evidence,
    automatic contradiction links, hybrid retrieval (BM25 + optional vector
    arm + RRF + graph + temporal), per-project weights,
    scope/project/session filters, superseded-row exclusion, full failure
    containment.
  - `EnhancedMemoryBackend` facade with auto-recall, per-turn observation,
    session-end and pre-compression mining, curated-write mirroring,
    diagnostics and graceful degradation to built-in memory.
- **Enhanced `memory` tool actions**: `search`, `recall`, `conflicts`,
  `timeline`, `topics`, `health`, `reinforce` with optional
  `query`/`limit`/`memory_type`/`graph_expand`/`temporal`/`memory_id`
  parameters; clear configuration hint when the layer is disabled.
- **Upstream Rekal plugin** (`plugins/memory/rekal/`): engine and Hindsight
  primitives replaced with the current OpenCode implementations, provider
  rewritten as the Hermes host of the single `mempalace` tool (31
  operations, upstream result formatting and protocol text), auto-recall,
  per-turn/session-end/pre-compression capture, and graceful degradation
  when the optional `mempalace` pip package is absent.

### 🔧 Changed

- `agent/memory_manager.py`: accepts `enhanced_config`; fans prefetch, sync,
  session-end, pre-compress, session-switch, memory-write and shutdown out
  to the backend only when enabled; new `enhanced_requested`,
  `enhanced_enabled`, `enhanced_backend` properties. Provider dispatch is
  untouched.
- `agent/agent_init.py`: creates the memory manager for enhanced-only
  configurations (previously it existed only with an external provider).
- `agent/system_prompt.py`: static memory protocol/trust block added to the
  stable prompt tier only when the layer is active (cache-safe).
- `hermes_cli/config.py`: `memory.enhanced` defaults (`enabled: false`,
  `data_dir: memory_engine`, `auto_recall`, `auto_capture`, `max_recall`,
  `recall_budget_chars`).
- `tools/memory_tool.py` and both dispatch paths
  (`agent/tool_executor.py`, `agent/agent_runtime_helpers.py`): enhanced
  actions wired through without changing add/replace/remove/batch semantics,
  budget checks, approval gate or error shapes.
- Rekal plugin legacy data migrates in place (`source_id`/`target_id` links,
  project-less config, 1–10 importance, missing project/wing/room columns,
  legacy supersede state); old `rekal_engine.py` removed.

### 🧪 Tests

- **347 tests pass** — 72 new core/tool tests plus the existing
  memory-provider, memory-tool and adjacent suites, and 45 plugin tests.
- New suites: `test_memory_ranking`, `test_memory_temporal`,
  `test_memory_graph`, `test_memory_dedup`, `test_memory_contradiction`,
  `test_memory_evidence`, `test_memory_mining`, `test_memory_entities`,
  `test_memory_context`, `test_memory_store`, `test_memory_backend`,
  `test_memory_manager_enhanced`, `test_memory_tool_enhanced`, plus ported
  `tests/plugins/memory/test_rekal_*`.
- Schema-contract tests pin that every parameter the handlers read is
  declared and that existing action enums/batch behavior are unchanged.

### 📚 Documentation

- **README**: reinstall-over-existing guidance (`pip install .` performs a
  code-only replace in the active environment; caveats for managed copies,
  PEP 668, running gateway, unsupported pip/PyPI policy) and system-wide
  installs (`/opt/hermes/venv` + `/usr/local/bin` symlink with per-user
  `HERMES_HOME`, plus Docker / pipx / uv tool / NixOS options).
- **CHANGELOG**: this GitHub-styled release-note entry.

### 📦 Install

- Verified `pip install .` from source in a clean venv: the wheel builds,
  ships `agent/memory/` (28 files) and the updated plugin, imports from
  site-packages, and reinstalls cleanly over an existing `hermes-agent`.

### 🚫 Not included (optional phases)

- Default vector/embedding backend (the optional vector hook exists and
  degrades to FTS5-only retrieval).
- Automatic indexing of existing `MEMORY.md`/`USER.md` into the structured
  store.
- MemPalace pip-package-dependent knowledge-graph/diary/fact-check
  operations (available through the optional plugin when `mempalace` is
  installed).
