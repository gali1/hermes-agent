# Memory enhancements — programme tracker

Authoritative status of the memory-enhancement programme (branch `memory-enhance-2026`,
base `81ba293f5f`). Written so every approved item can be checked without reading commits.

Source material (all four read in full):

| File | Covers |
|---|---|
| `3.1-preservation.md` | 21 pre-existing capabilities that must not regress |
| `3.2-3.8-engine.md` | rekal/MemPalace engine, 56 capability rows |
| `3.9-3.15-lifecycle.md` | enhanced layer (`agent/memory/`), 58 rows |
| `5-webui-binding.md` | Windows Desktop / dashboard binding |

Line references are to those audits unless marked as repo paths. Audits were taken against
`routing-release-20261103` @ `3272b009a2`; where this branch has moved on, §6 says so.

**Counting note.** The programme enumerates **40 approved IDs** — B1–B14 (14), C1–C8 (8),
D1–D14 (14), E1–E4 (4) — plus **D15**, which is tracked but HELD (§3). That is 41 tracked IDs,
of which 40 are approved for implementation. No 41st approved item exists; the "41 approved"
phrasing in the shard brief counts the held item.

Legend: `landed` = on branch with a commit; `W1`–`W5` = in flight in that shard's worktree.

---

## 1. Work items

| ID | What it fixes or adds | File(s) it touches | How it is verified |
|---|---|---|---|
| **B1** *landed* `e5aef6c2fc` | Evidence accumulation actually ranks: `proof_count` carried into the scored candidate dict (was read as default `1` ⇒ always neutral 0.5). | `plugins/memory/rekal/mempalace_rekal_engine.py` | `tests/plugins/memory/test_rekal_engine.py` — re-store N times, assert the corroborated row outranks an identical-score peer. Added in `e5aef6c2fc`. |
| **B2** W1 | `w_access` plumbed to `memory_search` / `build_context` — declared in the tool schema, never forwarded (stays at default). | `plugins/memory/rekal/__init__.py` | `test_rekal_provider.py`: non-default `w_access` changes recall order; absent param leaves order unchanged. |
| **B3** W1 | Event time on the record: `occurred_start` / `occurred_end` columns + migration, so `recency_for_range` and coarse-date handling stop being unreachable dead code. | `mempalace_rekal_engine.py` | Engine test: a memory about 2019 stored today ranks by the event window; a month-only date caps at 0.5, a point date does not. |
| **B4** W1 | Supersede exclusion applied to graph-expanded nodes — a row excluded from the FTS arm can currently be pulled back in through `supersedes`/`contradicts` (the largest multipliers). | `mempalace_rekal_engine.py` | Test: memory reachable **only** via a supersede edge must not appear in results; project/type filter on expanded nodes still holds. |
| **B5** W1 | `link_activation` wired as shared-connection activation (implemented + unit-tested, called nowhere). | `mempalace_rekal_engine.py`, `mempalace_hindsight.py` | Two memories sharing ≥3 entities outrank an equally scored non-sharing pair; tanh saturation bounds asserted (1 link→0.46, 4→0.96). |
| **B6** W2 | `select_with_temporal_coverage` called by `search` (8-bucket round-robin, defined + tested, never called). | `mempalace_rekal_engine.py` | 20 rows over 20 days, `limit=5` → 5 distinct days, not 5 rows from the 3 dense ones. |
| **B7** W2 | Hard temporal predicate in `search` — the window today only scores, so "in 2015" still returns 2026 rows. | `mempalace_rekal_engine.py` | "in 2015" returns only 2015 event rows; "last 2 weeks" bounded at both ends; purely temporal query with empty residual still filters. |
| **B8** *landed* `fbf3a5f9b2` | Enhanced prefetch bounded like every external provider (inline, untimed call on the turn's critical path). | `agent/memory_manager.py` | `tests/agent/test_memory_manager_enhanced.py` — stuck call is skipped on later turns, no turn exceeds the bound. Added in `fbf3a5f9b2`. |
| **B9** W2 | Contradiction thresholds configurable (`min_similarity` / `max_similarity` accepted but never passed; tuning needs a code edit). | `mempalace_rekal_engine.py`, `plugins/memory/rekal/__init__.py` | `set_config` round-trip; a pair at 0.30 is a conflict only when the band is widened; defaults unchanged. |
| **B10** W2 | Value / entity substitution conflicts: "TTL is 300s" vs "600s", "Postgres 15" vs "16", "Alice" vs "Bob" all currently return `(False, 0.0)`. | `mempalace_hindsight.py`, `mempalace_rekal_engine.py` | Table-driven test over the three verified misses; the 6 verified near-miss pairs must still return `(False, 0.0)`. |
| **B11** *landed* `81ba293f5f` | Recency decays from the last observation (`last_reinforced_at` was written and never read). | `mempalace_rekal_engine.py` | `test_rekal_engine.py` — re-observed old memory ranks as fresh; untouched peer keeps its decay. Added in `81ba293f5f`. |
| **B12** *landed* `dadc300da4` | `batch_store` records corroboration like `store()` (duplicates were returned but not reinforced). | `mempalace_rekal_engine.py` | `test_rekal_engine.py` — same fact via batch and via single store yields the same `proof_count`. Added in `dadc300da4`. |
| **B13** W3 | Near-duplicate detection at rekal store time (only a `score > 0.85` heuristic buried in `ingest_turns`). | `mempalace_rekal_engine.py` | Store twice with a one-word edit → flagged duplicate, evidence reinforced, one row; `allow_duplicate` still bypasses. |
| **B14** W3 | Conflict resolution workflow: a `contradicts` link can be reviewed, dismissed, or converted to a proposed `supersede`. Non-destructive — both rows kept. | `mempalace_rekal_engine.py`, `plugins/memory/rekal/__init__.py` | Resolve op leaves both rows readable, records the decision, and removes the pair from `get_conflicts`. |
| **C1** W3 | `evidence.confidence()` wired (source quality × proof × temporal − contradiction) — currently no caller outside its own module. | `agent/memory/evidence.py`, `agent/memory/store.py`, `agent/memory/ranking.py` | `tests/agent/test_memory_evidence.py` + ranking test: same content from a weaker source sorts lower. |
| **C2** W3 | `is_duplicate()` (trigram ≥ 0.97) wired into the store path — zero non-test callers today, so hash-only dedup misses 6 known pairs at ≥ 0.9. | `agent/memory/dedup.py`, `agent/memory/store.py` | `test_memory_dedup.py`: the 6 measured pairs collapse to one row, originals kept per §2; distinct facts are not merged. |
| **C3** W3 | Non-destructive consolidation pass (link + supersede, originals archived not deleted). | `agent/memory/store.py`, `agent/memory/backend.py` | After the pass: 0 rows deleted, originals still retrievable by id, `supersedes` links present, `retire_stale` does not fight it. |
| **C4** W4 | `agent_context` write scoping for the enhanced backend — `initialize(**kwargs)` ignores it, so cron/subagent turns are mined into the primary profile DB. | `agent/memory/backend.py`, `agent/memory_manager.py` | `test_memory_agent_context.py`: `agent_context="cron"` inserts nothing; `"cli"` still captures. Fails without the change. |
| **C5** W4 | `topics()` returns real topic discovery, not `GROUP BY memory_type` (a type histogram). | `agent/memory/store.py` | Two synthetic clusters → two named topics; type counts no longer presented as topics. |
| **C6** W4 | Atomic fact granularity at capture (live mean active row 1,541 chars, ceiling 8,000; 894/1,143 rows are assistant narration). | `agent/memory/mining.py` | One long turn yields ≤N atomic rows each under the cap; a multi-fact paragraph is split, not truncated. |
| **C7** W4 | Confirm `memory.enhanced.vector_backend` / `vector_model` are honoured on **this** branch; audit 3.15 called them unread (true at `3272b009a2`, false here — see §6). | `agent/memory/embeddings.py`, `agent/memory/backend.py` | `test_memory_vector.py`: `vector_backend=none` → lexical only; unknown backend → `EmbeddingUnavailable` → lexical only, no crash. |
| **C8** W4 | Embedding work kept off the inline prefetch path (indexer thread and vector query cost under the B8 bound). | `agent/memory/embeddings.py`, `agent/memory/backend.py` | Latency test with the arm enabled: prefetch stays inside the B8 bound; indexer never runs inline on a turn. |
| **D1** W4 | `hermes memory status` surfaces enhanced-store health (totals, by-type, conflicts, archived, DB+WAL size, FTS probe) — reachable in code, unreachable from the CLI. | `hermes_cli/memory_setup.py` | Command output on a populated profile lists the enhanced block; on a disabled layer it says so without erroring. |
| **D2** W4 | Operator maintenance commands for the enhanced store: purge / retire / backup. | `hermes_cli/memory_setup.py` | Each subcommand is dry-run by default, prints counts, and refuses when the layer is disabled. |
| **D3** W5 | Enhanced actions not advertised when the layer is off (all 7 sit in the schema enum unconditionally; they answer with an error string). | `tools/memory_tool.py` | Schema built with the layer disabled contains no enhanced actions; enabled layer contains all 7. |
| **D4** W5 | Trust stance reconciled: core fence says "authoritative reference data", enhanced block says "NEVER blindly trust" — same turn, same block. | `agent/memory_manager.py`, `agent/memory/backend.py` | Both texts asserted consistent by one test; the fence text is the single source. |
| **D5** W5 | Missing-context detection: a prefetch returning `""` is joined and returned with nothing inspecting it. | `agent/memory_manager.py`, `agent/turn_context.py` | Empty-recall turn emits the missing-context signal; non-empty turn does not. |
| **D6** W5 | Session-start recall parity, non-blocking (session-scoped recall exists only as a rekal tool op the model must choose). | `agent/memory_manager.py`, `agent/agent_init.py` | Turn 1 of a new session has recall available without a tool call; startup time not increased (see §2 risk). |
| **D7** W5 | Backup/export coverage: the enhanced DB is included in `hermes backup` under the `backup_paths()` contract. | `agent/memory/backend.py`, `hermes_cli/backup.py` | Backup archive contains `memory_engine/memories.db`; a profile with the layer disabled still backs up cleanly. |
| **D8** W5 | Retention/archive invariants written down and guarded: purge must not resurrect rows the miner re-adds. | `agent/memory/backend.py`, `agent/memory/store.py` | Test: purge → session end → miner run leaves the purged rows purged (`has_content`/`content_status` guard). |
| **D9** W5 | Operator views for timeline and topics. | `hermes_cli/memory_setup.py` | Timelines respect start/end/limit; topics print discovered topics, not type counts (pairs with C5). |
| **D10** W5 | Conflicts visible to the operator, not only to the model. | `hermes_cli/memory_setup.py` | Profile with one known conflict lists it; none → empty-state message. |
| **D11** W5 | Migration gating by `user_version` — `_SCHEMA_VERSION` is informational only today, so evolution relies on per-column presence checks. | `agent/memory/store.py` | Opening a v0 DB sets the version and runs once; re-opening is a no-op (idempotent), and a future-version DB is refused rather than half-migrated. |
| **D12** W5 | Junk-purge / origin-backfill counters surfaced in health. | `agent/memory/store.py`, `agent/memory/backend.py` | Health output includes both counts; they match a hand-counted run on a fixture DB. |
| **D13** W5 | FTS degrade telemetry: rebuild → drop/recreate → "no keyword arm" currently degrades silently. | `agent/memory/store.py` | Corrupt the FTS index: search still works, and health reports the degraded mode instead of looking healthy. |
| **D14** W5 | Architecture contract for both audiences (model prompt text and operator docs), consistent with §2. | `docs/MEMORY-ENHANCEMENTS.md`, `agent/memory/backend.py` | Document reviewed against the prompt block; no claim in one contradicts the other. |
| **E1** W5 | Governance: `memory.provider` stays `''`; rekal stays disabled as a provider; algorithms are ported, never activated as a layer. | config (`memory.provider`), docs | `hermes memory status` reports no external provider; `memory.provider` empty in every shipped config. |
| **E2** W5 | Test gate: the programme's baseline is enforced, not assumed. | `docs/MEMORY-ENHANCEMENTS.md` (§5) | The §5 command run at any shard head yields ≥374 passed, 0 failed. |
| **E3** W5 | Recall-quality measurement: `eval.py` harness (recall@1, recall@k, MRR per retrieval mode) runnable on real session data. | `agent/memory/eval.py`, `scripts/memory_eval.py` | `tests/agent/test_memory_eval.py`; a run prints aggregates only, never memory contents. |
| **E4** W5 | Operator verification pack: this tracker, kept current with the shards. | `docs/MEMORY-ENHANCEMENTS.md` | Every landed row names a commit; every in-flight row names a shard and a test. |

**D15** is not in the approved set — see §3.

Items are the shard-plan IDs; the four landed ones are confirmed by commit subject, and each row
cites the audit section it came from so a mapping mismatch can be spotted from the description.

---

## 2. Architecture — the decided shape

Three layers, and only three.

**Layer 0 — `MEMORY.md` / `USER.md`.** Curated, bounded in characters, always on. Injected as a
frozen system-prompt snapshot so mid-session writes never invalidate the prompt cache. Independent
of `memory.provider`: the builtin store path runs before provider activation
(`agent/agent_init.py:1315-1333`).

**Layer 1 — the enhanced store** (`agent/memory/`, `$HERMES_HOME/memory_engine/memories.db`).
The structured store: typed records, SQLite + FTS5, RRF fusion, graph expansion, contradiction
links, lifecycle (archive-not-delete, retention, junk purge, backup). Enabled by default
(`memory.enhanced.enabled: true`). This is where retrieval quality work happens.

**Layer 2 — the external provider slot, kept EMPTY.** `memory.provider: ''`. The slot holds at most
one external provider and is mutually exclusive by construction
(`agent/memory_manager.py:442-451` rejects a second one). It stays empty.

**rekal stays disabled as a provider.** Its retrieval algorithms are ported into layers 0/1 instead
(B1–B14). This is a deliberate inversion of the obvious move: the code is good, the deployment shape
is wrong.

**Why a third independent store was rejected.** Enabling rekal would add a second SQLite store
writing the same turns as layer 0 and layer 1 — three captures, three diverging truth sets, and
nothing in the manager to de-conflict them (a provider projecting its own user model gives the
model two competing profiles). The slot is mutually exclusive, so it would evict mem0/honcho rather
than compose with them, and swapping providers later orphans the previous store with no migration
path. Porting keeps one store of record, one migration surface, and one backup.

Retrieval logic is portable; storage is not. Only the algorithms cross the boundary.

**Known risk accepted with this shape:** layer 1 runs on the turn's critical path (bounded by B8)
and its writes share the single serialized worker with a 5 s shutdown drain. New session-start
hooks must not block — that is why D6 is specified as non-blocking rather than "call recall at
boot", and why C8 keeps indexing off the prefetch path.

---

## 3. Security — WebUI binding (D15, HELD)

**D15 is HELD pending explicit operator authorization.** No shard implements it. Nothing in this
programme changes a listener, a firewall rule, or an auth provider.

- **Loopback remains the default.** `hermes dashboard` defaults to `--host 127.0.0.1 --port 9119`
  (`hermes_cli/subcommands/dashboard.py:18-19`); `start_server` defaults identically
  (`hermes_cli/web_server.py:1591-1593`). The Desktop app spawns
  `hermes serve --host 127.0.0.1 --port 0` and never uses 9119.
- **A non-loopback bind requires an auth provider, or the server refuses to start.**
  `should_require_auth()` returns true for any host outside `{localhost, 127.0.0.1, ::1}` — RFC1918,
  CGNAT and link-local are deliberately treated as PUBLIC (`web_server.py:503, 522-529`). With the
  gate engaged and no provider registered, startup raises `SystemExit`
  (`web_server.py:1200-1204`); the message is at `:1150`. `--insecure` / `allow_public` is a no-op
  kept for old launch scripts (`:1189-1198`).
- **State on the Windows box today:** `dashboard.basic_auth.{username,password_hash,password,secret}`
  and `dashboard.oauth.{client_id,portal_url}` are empty, so a `0.0.0.0` bind exits non-zero at
  startup. The requirement cannot be met by adding `--host`; auth is a prerequisite.
- **The proposed firewall rule is documented, not created.** It must not be created until D15 is
  authorized *and* an auth provider is configured. Recorded here for review only — do not run:

  ```powershell
  # DO NOT RUN — proposed only, blocked on D15 authorization
  New-NetFirewallRule -DisplayName "Hermes WebUI (TCP 9119, LAN)" `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort 9119 `
    -Profile Private,Domain -RemoteAddress 192.168.1.0/24
  ```

  Private+Domain only, `Public` excluded (the box carries Hyper-V/VMware adapters), scoped to the
  LAN rather than any source. Scope to the venv python, not `hermes.exe` (a 115 KB shim), if
  `-Program` is used at all.
- **Do not rebind the Electron spawn.** Flipping `apps/desktop/electron/backend-command.ts:26` to
  `0.0.0.0` breaks the desktop loopback-auth exemption, the orphan reaper's cmdline match, and the
  port-0 token model. LAN access is an attach decision (`remoteBaseUrl`), not a rebind.

---

## 4. Exclusions

| ID | Excluded form | Why this form is excluded | What delivers the capability instead |
|---|---|---|---|
| **F1** | Enabling rekal as the external provider, tuned in place | Second store capturing the same turns; single mutually-exclusive slot; switching providers later orphans the store with no migration. Three layers would all be live on this host (`enhanced.enabled: true`). | Port the algorithms into layers 0/1 (B1–B14). The code is reused; the store is not duplicated. |
| **F2** | Binding the dashboard to `0.0.0.0:9119` as a default | Contradicts the governing guides ("bind to loopback by default", "never expose an unauthenticated administrative interface on all interfaces"); fails closed today because no auth provider is configured; the cited "§5.1–5.4 requires it" is **NOT SUPPORTED** — those are design-system sections and `9119` appears nowhere in the guides. | Loopback + authenticated tunnel for remote access; an opt-in, authenticated non-loopback bind only if D15 is authorized (§3). |
| **F3** | Rebinding the Desktop app's backend to `0.0.0.0` | Breaks three coupled invariants: the `HERMES_DESKTOP` loopback auth exemption, the orphan reaper's `--host 127.0.0.1 --port 0` match, and the port-0 session-token contract. Also makes every local tool call remotely invocable once a browser session is obtained. | Attach the desktop to a remote gateway via `remoteBaseUrl`; keep the bundled backend loopback + ephemeral port. |
| **F4** | Destructive consolidation — trigram merge that deletes near-duplicates | Violates the never-delete rule; provenance-carrying bullets are never deduped; 894/1,143 rows are narration the retention pass needs as evidence; a merge races the purge↔miner resurrection guard. rekal's own `deduplicate()` deletes, so it is not a model to copy. | C3: link + supersede, originals archived and still retrievable by id. C2 dedups at write time, before a row becomes evidence. |

---

## 5. Verification recipe

Run from the shard worktree, with the exact interpreter:

```bash
cd /home/bro/build/<w1..w6>
/home/bro/hermes-venv/bin/python -m pytest \
  tests/plugins/memory \
  tests/agent/test_memory*.py \
  tests/agent/test_builtin_memory*.py \
  -q --ignore=tests/agent/test_ssl_transport_contracts.py
```

**Baseline to beat** (measured at `81ba293f5f`, this recipe, 87.30 s):

```
374 passed, 1 skipped
```

Pass condition: **≥ 374 passed, 0 failed, ≤ 1 skipped.** Any other shape is a regression, not a
trade-off.

Additional suites for the ported engine work:

```bash
/home/bro/hermes-venv/bin/python -m pytest \
  tests/plugins/memory/test_rekal_engine.py \
  tests/plugins/memory/test_rekal_hindsight.py \
  tests/plugins/memory/test_rekal_provider.py -q
```

**Environment caveats.**

- Harness is python **3.12.3** (`/home/bro/hermes-venv`), pytest-asyncio 1.3.0.
- `tests/agent/test_ssl_transport_contracts.py` **cannot be collected here**: `truststore` is a
  python ≥ 3.14 dependency —

  ```
  ImportError while importing test module ... test_ssl_transport_contracts.py
  from truststore._ssl_constants import _original_SSLContext
  ModuleNotFoundError: No module named 'truststore'
  ```

  This is pre-existing and unrelated to the programme. The `--ignore=` flag is mandatory; without
  it the whole run reports a collection error and the baseline is unmeasurable.
- A test added for a behaviour change must fail without it:
  `git stash push -- <file>` → run the test → `git stash pop`. A test that passes both ways proves
  nothing.

---

## 6. Where the audits disagree

Recorded rather than silently resolved. Each row names both sources.

1. **Near-duplicate detection.** `3.7/#3` classifies it **NEW-SAFE** (the rekal store matches on
   exact SHA-256 only). `3.9` classifies "near-duplicate dedup (trigram ≥ 0.97)" as
   **IMPROVABLE-SAFE** — `agent/memory/dedup.py:66-76` implements `is_duplicate()` with zero
   non-test callers. Both are right about their own store; the two are different code paths, so the
   programme splits them: **B13** writes the check for the rekal store, **C2** wires the existing
   function in the enhanced store.
2. **Dedup maintenance vs consolidation.** `3.7/#4` says a dedupe maintenance operation is
   **ALREADY** present (rekal `deduplicate(project, dry_run)`, `engine:1623-1664`). `3.12` says
   consolidation/merge is **NEW-SAFE TO ADD** — nothing in the enhanced store merges, and the rekal
   operation *deletes*. The capability is not the same one: **C3** is new work and is explicitly
   non-destructive (§4/F4).
3. **Vector arm.** `3.3/#3` says the vector arm is **ALREADY** implemented (conditional on
   Chroma/`mempalace`). `3.15` says lazy/CPU embeddings are **NOT SUPPORTED in this branch** and
   that `vector_backend` / `vector_model` have no reader. Both audits were taken at
   `3272b009a2`; **this branch differs** — `agent/memory/embeddings.py` exists,
   `build_vector_support()` reads `vector_backend` at `embeddings.py:348`, and the table is
   `memory_embeddings` (`store.py:151`). `3.15`'s finding is therefore **stale for
   `memory-enhance-2026`**, which is why **C7** is a verification item and **C8** is a latency
   item, not implementations from scratch.
4. **Session-start retrieval.** `3.10` lists it **NEW-SAFE TO ADD** and `3.13` lists the bootstrap
   protocol **NEW-SAFE TO ADD**; they describe one capability (recall before turn 1) counted twice.
   The programme carries one item, **D6**.
5. **Event time.** `3.2/#7` and `3.6/#7` both report the missing `occurred_start`/`occurred_end`
   columns as **NEW-SAFE**, and `3.4/#5` reports the coarse-date path as **IMPROVABLE-SAFE**
   because it is unreachable. One root cause, one item: **B3**.
6. **Consistency of classification rest on this branch.** Spot-checked in the worktree:
   `evidence.confidence()` and `is_duplicate()` still have no callers, `backend.initialize()` still
   ignores `agent_context`, `topics()` is still `GROUP BY memory_type`, and the tool enum still
   advertises all 7 enhanced actions unconditionally. Those rows (C1, C2, C4, C5, D3) are current,
   not stale.
