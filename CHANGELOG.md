# Changelog

## 2026-10-05 — Additive enhanced memory layer + upstream Rekal plugin

Strictly additive: every existing memory capability (curated `MEMORY.md` /
`USER.md`, the `memory` tool add/replace/remove/batch, external memory
providers, prefetch/sync/session-end/pre-compress lifecycle, profile
isolation, prompt-cache layering, context-engine sanitization, skill
scaffolding stripping) is preserved and behaves identically unless the new
layer is explicitly enabled.

### Added — core enhanced memory layer (`agent/memory/`, opt-in)

Enable with `memory.enhanced.enabled: true` in `config.yaml`; data lives in
the profile-scoped `$HERMES_HOME/<memory.enhanced.data_dir>/` directory.

- `schema.py` — typed records: `MemoryRecord`, `MemoryQuery`, `MemoryResult`,
  `MemoryScope`, `MemorySource`, `MemoryEvidence`, `MemoryLink`,
  `MemoryConflict`, plus extensible type/scope/source constants.
- `ranking.py` — Reciprocal Rank Fusion, rank-space strategy boosts, bounded
  multiplicative combined scoring, linear/exponential recency decay,
  coarse-date handling, `proof_norm`.
- `temporal.py` — rule-based temporal query analysis (14 patterns), lexical
  rewrite that preserves the original query, temporal proximity and coverage.
- `graph.py` — bounded spreading activation with relation boosts, cycle
  protection, budget and activation threshold.
- `dedup.py` — trigram similarity, normalized-content equality, degenerate
  content filtering.
- `contradiction.py` — negation/polarity/similarity gates with confidence band;
  no false positives on unrelated or near-identical text.
- `evidence.py` — bounded confidence from proof count, source quality,
  temporal consistency and contradiction penalty.
- `entities.py` — bounded rule-based entity extraction.
- `mining.py` — conversation scanning/classification into durable candidates
  with secret, `<private>` block and transient filtering.
- `context.py` — budgeted recall selection and formatting.
- `store.py` — stdlib SQLite + FTS5 structured store: typed lifecycle
  (store/update/supersede/delete/link/unlink/reinforce), idempotent
  migrations, FTS self-heal, evidence accumulation
  (`proof_count`/`last_reinforced_at`), automatic contradiction links on
  store, hybrid retrieval (BM25 + optional vector arm + RRF + graph expansion
  + temporal windows), per-project scoring weights, scope/project/session
  filters, superseded-row exclusion, and full failure containment.
- `backend.py` — `EnhancedMemoryBackend` facade: auto-recall, per-turn
  observation, session-end and pre-compression mining, curated-write
  mirroring, diagnostics, optional vector-search hook, graceful degradation
  to built-in memory when the store cannot initialize.

### Changed — core integration (no-op when the layer is disabled)

- `agent/memory_manager.py` — accepts `enhanced_config`; fans prefetch, sync,
  session-end, pre-compress, session-switch, memory-write and shutdown out to
  the backend only when enabled; new `enhanced_requested` / `enhanced_enabled`
  / `enhanced_backend` properties. Provider dispatch is untouched.
- `agent/agent_init.py` — creates the memory manager for enhanced-only
  configurations (previously it existed only with an external provider).
- `agent/system_prompt.py` — static memory protocol/trust block added to the
  stable prompt tier only when the layer is active (cache-safe).
- `hermes_cli/config.py` — `memory.enhanced` defaults (`enabled: false`,
  `data_dir: memory_engine`, `auto_recall`, `auto_capture`, `max_recall`,
  `recall_budget_chars`).
- `tools/memory_tool.py` — additive actions `search`, `recall`, `conflicts`,
  `timeline`, `topics`, `health`, `reinforce` with optional
  `query`/`limit`/`memory_type`/`graph_expand`/`temporal`/`memory_id`
  parameters. Existing add/replace/remove/batch semantics, budget checks,
  approval gate and error shapes are unchanged; enhanced actions return a
  clear configuration hint when disabled.
- `agent/tool_executor.py`, `agent/agent_runtime_helpers.py` — both memory
  dispatch call paths pass the enhanced backend and new parameters.

### Replaced — `plugins/memory/rekal/` (optional external provider)

- `mempalace_rekal_engine.py` and `mempalace_hindsight.py` now carry the
  current upstream OpenCode implementations (import/packaging shim,
  `_db_path` for backups, idempotent migration of the earlier Hermes plugin
  schema: `source_id`/`target_id` links, project-less config, 1–10
  importance, missing project/wing/room columns, legacy supersede state).
- The provider is now the Hermes host of the upstream single `mempalace` tool
  (31 operations with upstream result formatting and protocol text),
  auto-recall via the advanced retrieval defaults, per-turn/session-end/
  pre-compression capture, and graceful degradation when the optional
  `mempalace` pip package is absent. The old `rekal_engine.py` is removed.

### Testing

- 347 tests pass: 72 new core/tool tests plus the existing memory-provider,
  memory-tool and adjacent suites, and 45 plugin tests.
- New suites: `test_memory_ranking`, `test_memory_temporal`,
  `test_memory_graph`, `test_memory_dedup`, `test_memory_contradiction`,
  `test_memory_evidence`, `test_memory_mining`, `test_memory_entities`,
  `test_memory_context`, `test_memory_store`, `test_memory_backend`,
  `test_memory_manager_enhanced`, `test_memory_tool_enhanced`, plus ported
  `tests/plugins/memory/test_rekal_*`.
- Schema-contract tests pin that every parameter the handlers read is
  declared and that existing action enums/batch behavior are unchanged.

### Install verification

- `pip install .` from source in a clean venv builds and installs
  successfully; the wheel ships `agent/memory/` (28 files) and the updated
  plugin, and `agent.memory` imports from site-packages.

### Not included (optional phases)

- Default vector/embedding backend (the optional vector hook exists and
  degrades to FTS5-only).
- Automatic indexing of existing `MEMORY.md`/`USER.md` into the structured
  store.
- MemPalace pip-package-dependent knowledge-graph/diary/fact-check
  operations (available through the optional plugin when `mempalace` is
  installed).
