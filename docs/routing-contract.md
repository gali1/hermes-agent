# Smart model routing — shared behavioural contract

Two Hermes deployments run two independently written routers:

| | `hermes-agent-fastembed` (oklab-1) | `hermes-agent` (bro-2) |
|---|---|---|
| module | `agent/model_router.py` (2 444 lines) | `agent/model_routing.py` (853 lines) |
| entry point | `select_model(...)` + turn-boundary wiring in `agent/agent_runtime_helpers.py` | `maybe_apply_model_routing(agent, text)` |
| explain | `explain(decision, profile)` (+ `hermes model routing explain`) | `explain_routing(agent, ...)` (used by `hermes_cli/subcommands/model.py`) |
| multimodal | `agent/image_routing.py` + modality fields on candidates | vision/audio/modality flags inside the router |
| layout | capability-state machine, weighted multi-factor score | tier + cost-weight scoring |

They are **not** the same implementation and are not meant to become one. This
document is the contract they must both satisfy, so a config or an expectation
can move between the hosts without changing what the agent does.

Sides are named by host, never as "this repo": the same file is checked into
both trees, and a relative label would mean the opposite thing in each.

## C1 — Enablement and inertness
Routing applies only when `smart_model_routing.enabled` is true. When false or
absent, the router returns "no decision" and the normal model selection is
untouched: no catalog read, no scoring, no switch. Any exception inside routing
is logged and swallowed — routing must never break a turn.

## C2 — Deterministic classification first
Classification is rule-based and clock-free, and runs **before** any model call.
It yields: complexity tier, token estimate, tool requirement, input/output
modality requirements, and structured-output need. An optional LLM-assisted
classifier exists only as a switch, never as the default path.

## C3 — Eligibility before ranking
Hard constraints are applied before scoring, and a model failing one is never
selected even if it is cheapest:

- deprecated models excluded;
- a model that cannot emit text cannot answer a text turn;
- required capability explicitly unsupported ⇒ excluded (unknown is *not*
  excluded unless `allow_unknown_metadata: false`, so a sparse catalog cannot
  empty the candidate set);
- context window smaller than the estimated input ⇒ excluded;
- max output smaller than the estimated output ⇒ excluded;
- complex/advanced turns impose a capability floor, not merely a penalty.

## C4 — Ranking factors
Score is a bounded **additive** combination, and the factor list is **not the same
on the two sides** — the contract no longer pretends it is:

- **oklab-1** (`agent/model_router.py:1087-1110`): capability, cost, reliability,
  context fit, continuity, weighted by `strategy` — `cost`:
  `0.35 / 0.55 / 0.05 / 0.05 / 0.0`; `quality`: `0.65 / 0.05 / 0.15 / 0.10 / 0.05`.
- **bro-2** (`agent/model_routing.py:307-333`): capability minus an
  unknown-capability penalty minus a cost penalty; context is a **hard gate**
  (`:293`), not a score term; continuity is an incumbent rule, not a term.

`reliability` is an **operator-declared preference, not a measured success rate**:
nothing assigns `ModelCandidate.reliability`, the only source is
`smart_model_routing.reliability`, and unknown reads as `0.5` (cold start). Because
it is additive it can **promote** an equal-scored rival as well as demote one, so the
older "only ever demotes / only more conservative" claim was wrong. Band: maximum
swing is `0.05` under `strategy=cost` and `0.15` under `quality`; C5 keeps the
incumbent when `gain <= stability_threshold`, so it decides first-turn picks and
ties. It cannot alone dislodge an incumbent **at the default `stability_threshold` of
0.15** — that is a threshold-dependent statement, not an absolute one: any configured
threshold below 0.15 makes the 0.15 quality-strategy swing decisive on its own.

### C4a — What "cost" means (oklab-1)

`estimate_cost` is the total of `estimate_cost_breakdown`'s terms, all in USD per
million tokens (the catalog's unit): uncached input at `cost_in`; cache-hit input at
`cost_cache_read` (declared override or catalog) **falling back to `cost_in`** when no
discount is published; the answer at `cost_out`; reasoning tokens at `cost_out` and
only for a model reporting reasoning support; images as a fixed 1600-token input
equivalent (`_IMAGE_TOKEN_EQUIVALENT`) because models.dev has no per-image price; and
**expected retries** = subtotal × `(1−p)/p` from a declared `reliability`, capped at
1.0, and `0` when no reliability is declared. Cache credit is caller-supplied and only
from the second routed turn (turn 0 populates the cache), and is clamped to the input
estimate. `charge` (the pool median for an unpriced model) prices every term, so an
unpriced model can never claim a discount.

With no cache credit, no reasoning tokens, no images and no declared reliability the
total is exactly the old `input*in + output*out` — a config that declares none of the
new terms prices identically to before. A model whose required capability is
**UNKNOWN** gets no cost credit on a reasoning task (`cost_score = 0`): honestly
priced reasoning tokens would otherwise let an unconfirmed non-reasoner win hard
prompts on price, which items 3/4 forbid. bro-2's cost penalty is unchanged.


## C5 — Stability (anti-flapping)
The incumbent model keeps the turn unless a rival beats it by more than
`stability_threshold` (default `0.15`; `min_improvement` is the same knob under
its older name) **or** a hard constraint forces the switch. Ties and borderline
wins keep the current model — a switch costs a prompt-cache invalidation.

**Tie rule (total order).** Equal scores resolve on the provider-qualified key
`provider/model` — the joined string compared literally, not a `(provider, name)`
tuple, which orders differently when one provider name is a prefix of another — and
not on declaration order and not on latency; no implementation
reads a latency field, and `ModelCandidate.latency_ms` is documented as unused. The
same rule decides the *first* pick of a session, where no incumbent exists yet, so it
is observable behaviour, not an implementation detail.

**Incumbent identity is `(provider, model)`, not the model name.** One model id can be
offered by two providers; a name-only match lets the "kept current" path report a
provider the session is not on, and makes a cross-provider move to an identically
named model unreachable (C8). Name-only matching is allowed **only** when the caller
has no provider to compare with.

Two falsifiers on bro-2 (`tests/agent/test_model_routing.py`): an incumbent `dup` on provider
`b` must switch to a cheaper `dup` on provider `a` (a name-only match would stay on `b`), and
the turn-boundary hook must call `switch_model` when only the provider changes for an
identically named model (the name-only "unchanged" check would return early).

## C6 — `preferred_models_only`
Restricts the *preferred pool*; it never overrides C3. A turn that needs tools
or a modality may select any eligible model, not only a preferred one.
**RESOLVED — shared default `false`.** A restriction needs an explicit opt-in:
with `true` and no `models` list, routing silently stops selecting, which is a
worse fresh-install surprise than routing across the eligible catalog. bro-2
already defaults to `false`; fastembed changes its default or documents `true`
as a deliberate local override. Until then, both hosts should still set the key
explicitly in `config.yaml` rather than relying on the default.
**The flag — not the shape of the model list — decides whether the catalog may be
consulted.** A non-empty `smart_model_routing.models` is the preferred pool, and
with `true` it is the *whole* pool: a modality/tool gap must keep the incumbent,
never reach the provider catalog. Reporting the list's key name as the pool scope
was a real defect on oklab-1 (a strict config still widened, and the CLI preview
disagreed with the runtime); any implementation that returns a scope value must
return one the widening guard recognises.

## C7 — Mixture-of-agents
If MoA is active for the turn, routing is skipped: the operator chose a
fan-out and replacing the model would silently dismantle it.

## C8 — Cross-provider selection
A switch may cross providers, but must resolve credentials through the target
provider's existing profile and rebuild credential pools, prompt-cache flags and
the compressor exactly as a manual switch does. Provider pools are never mixed:
a cross-provider move clears the previous provider's pool rather than borrowing
from it. Neither side may switch by mutating the model name alone.

**Mechanism — delegation first (agreed with bro-2).** The switch is delegated to the
runtime's model-swap entry point (`agent.switch_model`) whenever one exists, because
that is the seam that re-resolves endpoint and credentials by construction and the
seam both sides' falsifiers already exercise. In-place re-resolution is the
**fallback**, not a parallel design, and it is allowed only when `switch_model` is
absent — the implementation must name that absence in the reason it reports. The
fallback must satisfy the same postconditions (target-profile credentials, previous
pool cleared, prompt-cache flags, compressor) and is verified by the same falsifier;
without that, "otherwise managed in place" becomes a permanent way for one side to
stay put.

## C9 — Multimodal continuity — **C9a: implemented on oklab-1, partial on bro-2; C9b: nowhere**

**C9a — input modality.** The requirement must be derived from the attachments present
at the turn boundary, never from message text. Falsifier: a message carrying an image
part must not select a model whose input modality excludes images, and must not raise.

- *oklab-1*: **IMPLEMENTED.** `message_text_and_attachments()` splits a turn into text and
  modality hints (text/image/audio/pdf/video; unknown parts contribute nothing), and the
  loop passes both to `route_turn(..., attachments=...)`, so the modality reaches
  classification instead of being flattened by the old `str()` coercion. An image
  attachment makes `vision` a hard requirement: a model declaring `supports_vision: false`
  is excluded as `unsupported_vision` and cannot win the turn. Limits that remain: an
  UNDECLARED `supports_vision` reads as unknown and is still allowed (unknown ≠
  unsupported), and `SPEC_FIELDS` cannot declare image *output*, so an image-edit request
  is not yet protected on the output side (that is C9b).
- *bro-2*: NOT IMPLEMENTED for string prompts, but the crash is fixed. The turn boundary
  (`agent/turn_context.py:1117-1118`) forwards the payload verbatim, and that payload can be a
  multi-part content list — vision via the OpenAI-compatible server, per
  `agent/agent_runtime_helpers.py:3358` — so `classify_request` used to raise `AttributeError`
  (`body.strip()` on a list, swallowed at `model_routing.py:731`) and every such turn was
  routing-inert by accident. It now reads text, image, audio and file parts and ORs their
  flags with the caller's, so a list payload neither raises nor selects a text-only model for
  an image part. Still missing: attachments delivered beside a *string* prompt are handled by
  `image_routing.py` after selection, so the string path cannot state its modality.
- Both: `image_routing.py` is a **separate axis applied after selection**, which cannot
  satisfy C9a.

**C9b — output modality and attachment continuity.** An image-edit follow-up must keep
the original attachment and require image-output capability.

- *Enforceable on oklab-1*: `supports_image_output` and `supports_audio_output` are
  declarable `smart_model_routing.models` overrides, so an image-output request
  (`expects_image`, which puts `image` in `output_modalities` and `output_image` in the
  required capabilities) excludes a model that declares it cannot emit images —
  `excluded_reason: unsupported_output_image`. A model that declares nothing stays
  UNKNOWN and therefore eligible: unknown is not the same claim as unsupported. Verified
  by test: `test_image_output_request_excludes_a_model_that_cannot_emit_images`.
- *Recorded on oklab-1*: `Decision.input_modalities` / `.output_modalities` carry what the
  decision was made for, so a follow-up turn can ask what was promised.
- *Derived from the turn on oklab-1*: the classifier now recognises an image-output
  request from the prompt itself (a generation verb within 40 characters of an artefact
  noun: generate/create/make/draw/paint/render/produce/design + image/picture/
  illustration/drawing/artwork/logo/icon/poster/avatar). Negative controls are pinned by
  test, because the naive "contains the word image" rule would route every attachment
  question through an image-output gate: "what is in this image?" and "explain how image
  generation works" both stay text.
- *Still missing on BOTH sides*: re-attaching the original image on a follow-up edit
  turn. Until that lands, an image-edit turn is routed correctly but the attachment
  continuity is the caller's job, so do not imply end-to-end image-edit support.

**Invariant that does hold:** neither side may satisfy a modality requirement by
dropping the attachment.

## C10 — Explainability
Every decision carries a human-readable reason and a machine `skip_reason`, drawn
from one vocabulary both sides accept: `no_candidates`, `no_eligible_model`,
`already_current`, `within_stability_threshold`, `moa_active`, `disabled`, `error`
(oklab-1 pins the set in `SKIP_REASONS`).

**`moa_active` — implemented on oklab-1, still absent on bro-2.** The skip used to be
invisible on both sides (oklab-1 at the call site, bro-2 deep in the router where it
returns `None` before a `Decision` exists). oklab-1 now (a) delegates the decision into
`route_turn(moa_active=...)`, which records `ROUTING_SKIP_ATTR`
(`_last_routing_skip_reason`) before returning `None`, and (b) emits it in the machine
surface: `routing_report(..., moa_active=True)` returns `skip_reason: "moa_active"`,
`selected: null`, `kept_current: true` and a human `reason`, while keeping the
candidate rows so "what would routing have chosen" stays answerable — a skip must not
look like an empty catalog. bro-2 still returns `None` before any decision exists, so no
dict there can carry the reason. C10 is satisfied on oklab-1 and open on bro-2.

**The machine surface is a dict, and it is normative.** oklab-1 calls it
`routing_report()` (`model_router.py:2361-2455`), bro-2 calls it `explain_routing()`
(`model_routing.py:735-821`): same shape, different name. Either name is acceptable
locally, but the dict — not any text renderer — is what automation may parse, and a
text renderer must be derived from it so the two cannot disagree. oklab-1's
`explain()` (`:1486-1513`) is **text only** and is not an automation surface.

"why not X" is part of the contract: the dict carries one row **per candidate** with
`excluded_reason` (empty when the candidate was eligible). Verified on oklab-1: every
row carries `excluded_reason`, `eligible`, `score`, `provider`, `model`, cost fields.

## C11 — Config compatibility
Canonical key first, explicit legacy alias second, and **no silent ignores**:

| canonical | accepted alias | behaviour when both set |
|---|---|---|
| `stability_threshold` | `min_improvement` | canonical wins, both accepted |
| `reliability_rerank` (tool search) | `reliability` | canonical wins + warning |
| `strategy` | — | invalid value ⇒ `automatic` |

Unknown keys under a routing/tool-search section produce one warning naming the
key and stating it is not in effect.

## C12 — One applier per setting, whatever the input surface
A routing setting is applied by exactly one function. Text (`/model-routing 3`), a
numbered reply, and an inline button must reach the same applier and the same writer
(`update_routing`); a button is an input method, never a second code path. Platforms that
render buttons are an optimisation: where an adapter exposes no picker (or a send fails),
the command must still work as the numbered text flow, and no setting may exist only
behind a button. Correspondingly, a button flow applies nothing until the final
confirmation, and its result text is the block read back from disk — not the caller's
optimistic echo.

## C13 — Consent before routing is switched on
`enabled` is the one routing key that changes behaviour on every later turn, so the write
that sets it to true needs the operator's agreement on any surface a human is driving:
* The CLI's `enable` verb prints the resolved settings and candidate pool (the same
  `_print_routing_status` renderer `status` uses) and asks before writing. `-y/--yes` is
  the assume-yes escape hatch, matching `configure`.
* A caller with no terminal — a script, cron, the gateway — is never prompted: it cannot
  answer, and a prompt there would hang instead of doing what was asked. Its behaviour is
  unchanged from before this contract.
* Turning it off never prompts. Disabling restores the single-model path, so there is
  nothing to warn about, and a gate there would only make recovery slower.
* Asking to enable something already enabled is not a state change and is not prompted.
* A preview is not a write: the renderer is called with `saved_note=False` before the
  question, so the "Saved in: config.yaml" line can never appear on a state that is not
  on disk.

## C14 — A refresh is a cache operation, and the footer is a claim about the file
Two small truths that the CLI surface must keep:

* **Catalog refresh reuses the engine's cache.** `hermes model routing refresh`
  force-refreshes the same `agent.models_dev` registry that `_catalog_models` resolves
  candidates from — no second cache, no second network path, and selection itself never
  fetches (C1). A refresh writes the catalog cache only: `smart_model_routing` is not
  touched, so a refresh can never enable, narrow or widen a pool by itself. When the
  catalog cannot be fetched the verb says so and exits non-zero rather than reporting a
  success it did not have; the cached copy keeps being used either way.
* **"Saved in: config.yaml" is a statement about the file.** The renderer prints that
  footer only when the loaded config actually contains a `smart_model_routing` section
  (`saved_note=None` means *ask the file*). An absent section is off-by-default, not
  saved — printing the footer there would describe a block the operator does not have.
  `saved_note=False` remains the pre-write preview form (C13).
* **Bare `hermes model routing` follows `hermes model`.** With a terminal it opens the
  interactive flow; with no terminal it prints the read-only status instead of blocking.
  `status` stays reachable as its own verb in both cases.

## C15 — Every runtime-read setting is reachable, and nothing else is a control
Directive §4 in one line: a control exists only for a setting the engine reads, and
every setting the engine reads has a control. On oklab-1 that means:

* **Reachable from both interactive surfaces.** `enabled`, `strategy`,
  `stability_threshold`, `cost_optimization`, `preferred_models_only`,
  `switch_on_cost`, `allow_unknown_metadata`, `exclude`, `weights` and `reliability`
  are settable from `hermes model routing configure` (flags **and** the interactive
  flow) and from `/model-routing` → Preferences. Every one of them is also settable
  without a TTY, so the gateway, scripts and people share one code path.
* **One parser, one writer.** Typed values (`exclude a,b`, `weights dim=value`,
  `reliability model=rate`) are parsed by `agent.model_router.parse_model_ids` /
  `parse_weights` / `parse_reliability` and written by `update_routing`. A surface
  renders and dispatches; it does not decide what a payload means. The chat accepts
  exactly what the CLI accepts, and both reject with `validate_routing`'s message.
* **Weights are merged, never re-derived.** An edit starts from the values *as
  written* in `smart_model_routing.weights`, so repeating the same command is a
  no-op. Merging over the normalised effective mapping (which the reader rescales to
  sum to 1) moves every number on each run and would rewrite the file forever.
  `weights` applies under `strategy: automatic`; `cost` and `quality` substitute their
  own fixed vectors by design, so a custom weight set under those strategies is inert
  — the menu says which strategy is active for exactly that reason.
* **No control for a key nothing reads.** `llm_assisted_classification` is still
  parsed and round-tripped (an operator's file keeps its value) but no code path
  consults it, so no menu offers it any more. C2 keeps classification deterministic.
* **Nothing is written before the confirmation.** Pending values live in the flow
  state only; Cancel, and a flow that times out, leave `config.yaml` byte-identical
  (C13).

## C16 — A verb that changes nothing does not change the file
`disable` is the OFF state, and an *absent* `smart_model_routing` block is also the OFF
state (`routing_settings` reads it as `enabled: false`; absence is the shipped default).
So `disable` on a config that never carried a block has nothing to do. This tree must not
materialise the section for it:

* `hermes model routing disable` on an absent block → exit 0, prints `already off — nothing
  to change`, reads nothing but the state, and leaves `config.yaml` byte-identical.
* `hermes model routing configure --disable` on an absent block → the same rule in the
  shared writer: `written: false`, `changed: []`, `no_op` naming the reason.
* The rule is scoped to the *absent* case. A present block is still written when the
  value changes (`disable` while on turns it off and keeps every other preference), and a
  present block already at `enabled: false` is not rewritten.
* `enable` is unaffected: an absent block plus `enabled: true` is a real change and writes.

Why it matters: every later reader treats a present block as an opt-in record, `status`
prints the "Saved in: config.yaml" footer for it, and `setup_quick.py` writes its own
explicit `enabled: False` on purpose. A `disable` that manufactures one blurs "the user
opted in and turned it off" with "the user never opted in", for a command with no effect.

Residual, deliberately not changed: when the state already matches and the block *is*
present, the verb still prints `Saved to config.yaml`. That statement is about the
resulting config, which does hold that state; it was left alone rather than widening the
diff.

## C17 — A no-op reply may not claim a write, and "single-model mode" is disable on the file too

C16 governs the file; this governs what the chat says about it, and closes the one
spelling of disable that still changed the file.

* **The gateway reports a write only when the file changed.** `_model_routing_commit`
  reads the writer's own `written` flag before composing its reply. When the writer
  returns `written: false` (the C16 absent-block no-op), the reply is
  `Nothing saved — <no_op reason>`, it says `config.yaml was not modified`, and it never
  carries `✅ Saved` or `Takes effect on the next turn`. This is C14's rule (a footer is a
  claim about the file) applied to §6.3's result message; the CLI has held it since M13.
* **`Switch to single-model mode` produces disable's change-set.** §6.3 makes it the same
  action, so it may not add `preferred_models_only: false` unconditionally: on a config
  that never carried a block that extra key made the *writer* see a real change and
  materialise `{enabled: false, preferred_models_only: false}` where `2`/`disable` wrote
  nothing. The flag is cleared only when it is actually on (pending or saved), and a
  pending `models` selection is always dropped.
* **Both input surfaces.** The typed applier and the button applier carry the same
  change-set logic (C12); the rule is asserted through the real dispatch path and through
  `_model_routing_button_action`.

Control that keeps the assertions honest: `enable` on the same blockless config must
still print `✅ Saved smart_model_routing` and must still write, and `single` with the
flag genuinely on must still turn it off.

## C18 — Picker state is per (chat, message), and one finished picker is finished

`/model-routing` sends a toggle list whose buttons carry indexes into server-side state. Two
pickers can be alive in one chat (the user asks again before finishing), Telegram can deliver
the same callback twice, and a plugin reload can replace the adapter between the send and the
tap. The state therefore belongs to a message, not to a chat:

* **A tap applies to the picker it came from.** The handler resolves state by
  `(chat_id, message_id)`; a chat slot belonging to a *different* message is not this tap's
  state, and the process-wide registry is the fallback (it survives an adapter swap).
* **Finishing retires that message's entry, and only that entry.** Done and Cancel retire the
  tapped message in both the chat slot (only when the slot *is* that message) and the
  registry. Leaving the registry entry behind lets a repeat Done tap find the consumed state
  and run `on_selection_done` a second time — a second write for one confirmation, which is
  C17's class of defect reached through the adapter instead of the writer.
* **Retirement is not a chat-wide reset.** Tapping Done on an older picker while a newer one
  is live must leave the newer one's slot intact, or a live picker becomes answerable only
  through the registry and dies if its entry is pruned (bounded size, 30-minute TTL).

Control that keeps the assertions honest: a tap from an older message must still apply to
that message's options, and a picker must still answer after the adapter's per-chat maps are
cleared (simulating a plugin reload).

## C19 — A page the user must answer with text has to say how

The numbered models page is not a menu graphic: every caller of it is a reply the user answers
by typing. It is reachable only where no button picker was used — a text-only platform, or a
button platform whose toggle send failed — so it must name the accepted answers:

* one compact line, not a prose dump: the numbered form (`1,3`, first = highest priority), the
  word forms (`all`, `none`, `back`, `cancel`), and the exact-name form (`add <model>` /
  `remove <model>`);
* the pager stays one line (`Models 1–10 of 132 · next / prev`) rather than a second sentence
  of instructions;
* removing the instruction line is a regression even though the row list still renders: a
  user who cannot see what to type is the reported "I type something and get the same text
  back" loop, reached through the page instead of the flow.

Control that keeps this honest: a button platform's typed `3` must send the picker and never
this page (C12: same interaction as the 🎛 button), so the page only exists where the line
above is the only guidance available.

## C20 — `tool_calls_only` keeps the default model primary

The fourth strategy is not a weighting of the other three: it removes selection from the turn.

* An **eligible incumbent is never displaced** — not for price, not for a higher score, not for
  a closer capability fit, and not by `switch_on_cost`. `rank_candidates` returns it before the
  hysteresis block, so none of those knobs are consulted.
* The only thing that moves the turn is **ineligibility**: the incumbent failed a hard
  requirement (a modality it does not accept, a context window it cannot hold, a tier floor),
  so it never reaches the "keep the incumbent" branch and the specialist is chosen instead.
* The specialist is then chosen **capability-first, cost as tiebreak** — the pool's job in this
  strategy is to cover a requirement, not to win on price.
* `providers.<provider>.models` therefore reads as *the assistant plus its specialists*, and a
  model listed there is not thereby a candidate to take over ordinary conversation.

Consequence worth stating in the reply, not just here: the routing hook runs once per **turn**,
before the model is invoked. `tool_calls_only` decides *which model serves the turn*.
Sub-task delegation inside a turn is a separate mechanism — see **C21**, which now exists and
is tested. C20's own limit stands: this hook still never sends a file to a specialist.

## C21 — Capability delegation is a sub-call, not a second router

`agent/capability_delegation.py` answers a different question from C20's: *which model performs one
sub-task inside a turn the primary model still owns*. The primary is never displaced by it.

* **Audio is a transcript, not a model call.** The capability the primary lacks for audio is
  speech-to-text, which is infrastructure here; the row below shows it working, with the PyAV
  incompatibility (`faster_whisper` wants `av.open(metadata_errors=...)`, PyAV 19 removed it)
  routed around by an isolated venv behind the documented command-provider seam.
* **Portable first, native part only as an opt-in.** Nothing in this tree builds a `file` or
  `input_audio` part — `gateway/platforms/api_server.py` rejects both and `agent/image_routing.py`
  states that a PDF must never become a vision part — so the default path is the one every provider
  accepts: a **pdf is extracted locally** (`tools.read_extract.extract_document_text`) and the text
  is sent; an **audio file is transcribed** with the existing STT stack
  (`tools.transcription_tools.transcribe_audio`) and the transcript is sent; text-like files are
  inlined (truncated, marked); images travel as `image_url` data URLs. `transport="native"` sends
  the provider part instead (`file` / `input_audio`) and reports the provider's own words when it
  refuses. Video has **no** transport at all: the layer refuses with "extract frames or text first".
* **The specialist is chosen by the routing pool and the routing rules.** `preferred_models_only`
  is honoured, capability leads and price only breaks ties, and the config's own model is excluded —
  delegating to the model already holding the conversation is a no-op with a price tag.
* **A refusal is reported verbatim, never retried into a success.** Provider 410/403/404 markers
  ("deprecated", "access is disabled", "no such model") are recorded in
  `$HERMES_HOME/state/capability_delegation/unusable.json` for 24 h so the next resolve skips that
  model instead of walking into the same wall. An ordinary `400` is **not** remembered: that is a
  malformed part, which is this layer's bug and must stay visible.
* **No success without text.** Empty content, a decode failure, an oversized body (20 MB default) or
  a missing specialist all produce `success: false` with a reason; nothing is fabricated.

Verified on this host (live config, deepseek primary, gpt-6-luna specialist):

| case | result |
|---|---|
| image (PNG) → `gpt-6-luna`, asked for the visible string | `DELEGATE-OK-7319` returned |
| pdf (688 B, real xref) → extraction (80 chars) → `gpt-6-luna` | `PDF-OK-4412` returned: the specialist saw text, never bytes |
| audio (Ogg Vorbis) → local STT → transcript | **works**: `stt.provider: local_command` → `HERMES_LOCAL_STT_COMMAND` → `~/.hermes/tools/stt-venv` (isolated faster-whisper); the transcript is the delegation result and **no model request is spent** |
| `gemini-3-flash` (catalog-listed, provider-retired) | 410 `ModelDeprecated` reported, then skipped on the next resolve |
| audio / video specialists on opencode-zen | **none live**: `gemini-3.5/3.7-flash` answer 403 "model access is disabled" — audio and video delegation are unavailable until a live multimodal specialist exists |

That last row is the honest boundary of this feature on this provider, not a code gap: the layer
picks correctly and reports what the provider says.

## C22 — Capability registry: native, via-Hermes, unknown, unavailable

`agent/capability_registry.py` is the one place that answers "can this model do X *here*", and it
refuses to flatten four different answers into one boolean:

| state | meaning | consequence |
|---|---|---|
| `native` | the model/provider accepts it (catalog metadata, overlaid by the operator's declarations) | send the part |
| `hermes` | the model cannot take it, but this installation converts it: PDF text layer, OCR, speech-to-text, image data-URL | send the converted form |
| `unknown` | nobody declared it and metadata is silent | eligible only when the routing config allows unknown metadata, and reported as unknown |
| `unavailable` | the provider refused the model (retired/forbidden, memoised 24 h) or the local facility is absent (no tesseract, no STT) | do not claim it works |

The distinction matters most for PDFs: a model with no native PDF input is **not** a model that
cannot handle a PDF — Hermes extracts the text (or OCRs a scan) and sends text. Reporting that as
"unsupported" would misdescribe the system, not the model.

Verified on this host: `deepseek-v4.1-flash` → native image/tools/structured, **hermes** for pdf +
audio + ocr, unavailable video; `gpt-6-luna` → native image/pdf, hermes audio/ocr.

## C23 — OCR is a fallback that must detect the extractor's own placeholder

`tools.read_extract` does not return an empty string for a scanned page: it returns a bracketed
`[NEEDS OCR: pages 1 …]` **instruction addressed to the model**. Treating that as text made a scan
look readable, and a specialist answered "the page isn't available" from the instruction itself —
a failure that looks like a model problem and is not one. `capability_delegation.usable_text()`
strips those notices; only then is the length threshold meaningful, and only then does OCR run.

Order for a pdf: **text layer → OCR (poppler + tesseract) → native `file` part**, last resort only.

Verified end to end: an image-only PDF (no text layer) → OCR read `MAGIC-OCR-8842` → `gpt-6-luna`
returned exactly that string.

## C24 — Video is decomposed, not demanded

No provider in this pool accepts a native video part, and waiting for one is not a design. A clip
becomes a labelled, timestamped text representation with the tools already on the host:

```
ffprobe  -> duration + dimensions
ffmpeg   -> sampled frames (scene changes first, evenly spaced as a fallback), capped
tesseract-> on-screen text per frame, with the frame's timestamp
ffmpeg   -> 16 kHz mono wav -> local STT -> speech transcript
```

`agent/video_context.py` assembles that into `VIDEO / SPEECH / ON SCREEN / NOTES` and the result is
sent as **text** to a text-reading specialist (`resolve_specialist("text")`), because asking for a
model with native video input would refuse work this installation can already do. Frames are kept
in a temp dir and deleted unless `keep_frames=True`; sending frames to a vision model is opt-in
(`frame_image_parts`), because a frame costs tokens and OCR costs nothing.

Verified: a 6.1 s clip carrying on-screen codes and an audio track → 8 frames, all OCR'd
(`MAGIC-VIDEO-5510`, `ACTION ITEM ship OCR path`), speech transcribed locally → `gpt-6-luna`
answered *"shows MAGIC-VIDEO-5510 until about 3.1 seconds, then ACTION ITEM ship…"*, i.e. it
reasoned over timestamps it never saw as video.

## C25 — Several operations per request, bounded

One request can need more than one capability (a recording that must be transcribed *and* have its
slides read). `max_operations` (default 1, replaces `max_per_request` — still honoured) allows
several, but never unbounded:

| guard | mechanism |
|---|---|
| operation count | `max_operations`, and one operation per `(path, modality)` |
| time | `budget_seconds` — no new operation starts past the deadline |
| money | `budget_usd` — estimated cost accumulates as results arrive; the rest is skipped |
| repeats | per-turn memo keyed by `turn_id`/`session_id` plus the material marker (TTL 30 min): the injected note is not enough, because the loop rebuilds the request from history on the next model call and would delegate again |
| recursion | thread-local guard on the middleware |
| partial results | a skipped or failed operation becomes a labelled note; the primary answers from what came back |
| audit | one `delegation` line per operation + one `delegation_batch` line with `operations`, `est_cost_usd`, `budget_hit` |

Operations are independent by construction, so they run in parallel (`ThreadPoolExecutor`, max 4);
`parallel: false` runs them in order. Verified live: one turn with an audio note and a scanned
invoice → two operations (audio → `stt/local`, pdf → `gpt-6-luna`), one answer containing the OCR
code, `delegation_batch` showing `operations: 2`.

## C26 — Path trust: only adapter-produced files are delegated on remote surfaces

On `telegram`, `api_server` and any other remote platform, the delegation planner accepts **only**
the adapter's own `saved at:` note pointing inside Hermes' media caches
(`cache/documents`, `cache/images`, `cache/audio`, `cache/video`, `cache/media`) plus the local Bot
API server's own directories (`telegram-bot-api/local`, `telegram-bot-api/data` — in `--local` mode
the adapter reads telegram files straight off disk there), resolved with
symlinks and `..` collapsed (`Path.resolve(strict=True)` then `relative_to(root)`). A path that was
typed, pasted, recalled from memory, or echoed by a tool or a web page delegates nothing.

Why it is not merely tidy: a typed path that is treated as an attachment lets anyone who can put
text in front of the agent — another member of a group chat, a web page the agent fetched — make it
read a local file and send the contents to a third-party provider. One such source was already
found in the wild: recalled memory injected into the user message contained my own scratch paths,
and four `platform=telegram` events were produced for files nobody had attached.

On the **CLI** typed paths stay allowed: it is the operator's own machine, and refusing them would
break every local workflow. `strip_injected_context()` removes `<memory-context>` and system-note
blocks before scanning, and tool/web output is never scanned at all (only the last user message is).

## C27 — Cost: measured, estimated, or free — never silently zero

| `cost_basis` | meaning | how produced |
|---|---|---|
| `usage` | measured | the provider reported token usage; priced from the specialist's own rates |
| `estimated_from_bytes` / `_binary` | estimated, and labelled | provider silent: `bytes/3` input tokens (pessimistic) + the configured output allowance (floor 1024) |
| `free_local` | genuinely free | local STT/OCR: no model call was made |
| `failed` | nothing was spent | the operation failed before a call |

`cost_known` is true only for `usage` and `free_local`. **A money budget charges the estimate**,
because a budget that treats a silent provider as free never trips — which is the same as having no
budget. `delegation_batch.cost_known` is the AND of its members, and its `cost_basis` reads
`mixed_or_estimated` when they differ. Verified: a 2 MB audio payload with a silent provider
estimates above a 1-cent budget; a 4-delegation run with the budget below the per-op estimate stops
launching after the first.

## C28 — api_server parity means its part types, not a workaround

`gateway/platforms/api_server.py:466-468` defines the whole surface:

```
_TEXT_PART_TYPES   = {"text", "input_text", "output_text"}
_IMAGE_PART_TYPES  = {"image_url", "input_image"}
_FILE_PART_TYPES   = {"file", "input_file"}       # normaliser raises before any provider call
```

Accepted = **five** part types (text/input_text/output_text/image_url/input_image), plus a bare
string and a list item that is a bare string, which are stringified. By design, and documented
rather than routed around:

* `file` / `input_file` parts are refused at normalisation, not at the provider.
* A `data:application/pdf` URL is refused for both `file` and `image_url` ("Only image data URLs
  are supported"), so a document reaches a model on this surface **only as extracted text** — which
  is what the delegation layer produces anyway.
* No audio parts (`audio_api: False`), no magic-byte inspection of inbound image payloads.

So api_server parity is: the same delegation *behaviour* (note → local extraction/OCR/STT →
labelled context → primary answers), over the part types this route accepts.

## C29 — Telegram inbound size: this install is not bound by the 20 MB hosted cap

`telegram.extra.base_url: http://127.0.0.1:8081/bot` with `telegram.extra.local_mode: true`: a
self-hosted local Bot API server. The hosted-API 20 MB **download** limit therefore does not apply
(2 GB send cap retained), and inbound files land on disk where the adapter records their paths —
which is exactly the material C26 trusts.

## C30 — Document limits are enforced and announced

| guard | where | behaviour |
|---|---|---|
| zip bomb | `read_extract._zip_inflation_error`, before any inflation | refuses when the central directory declares > 200 MB uncompressed, or a ratio > 200:1 above 8 MB |
| xlsx rows | `_MAX_XLSX_ROWS_PER_SHEET` = 5000, 256 cols | appends `…[truncated: showed 5000 of 6000 rows on sheet 'Q3' (limit 5000)]` |
| csv/tsv charset | `decode_text_payload`: BOM → utf-8-sig/utf-16/utf-32, else strict utf-8, else cp1252 | a cp1252 file no longer arrives as `Ren\ufffd M\ufffdller`; a UTF-16 file no longer arrives NUL-separated |
| csv/tsv truncation | `clip_text_with_counts`, on a line boundary | `truncated: showed 4762 of 9001 rows, 119991 of 230205 chars` — never a silent cut mid-row |
| sqlite | `mode=ro&immutable=1` | read-only; 200 tables / 5 preview rows / 80 chars per cell |
| macros | never read | `vbaProject.bin` is not opened; nothing is executed |
| formulas | cached values only | `=A1*B1` surfaces as its cached result; the formula text is invisible |

Verified on generated files: a 12 KB package declaring 315 MB → *"Not a safe DOCX: … refusing to
inflate it"*; a 6000-row sheet → the truncation line above; a 230 KB CSV → the counted clip above.

## C31 — A finished result ends the re-derivation

When a delegation succeeds, the injected block states that the content was extracted from the actual
file and is complete, and **the call that carries it sets `tool_choice: "none"`** (when the request
has tools). This is a stop condition, not a hint: measured twice, the primary re-ran its own
extraction over the same file and hit the turn budget without answering.

Guards: only on success (a failed delegation must leave tools available, or the model cannot recover
itself), only when `tools` is present, never overwriting a `tool_choice` the caller set, the caller's
request is not mutated in place, and `capability_delegation.stop_tool_use_after_delegation: false`
turns it off. A `tool_stop` line is written to the audit trail.

## Extensions that are allowed to differ
`switch_on_cost`, `llm_assisted_classification`, `weights`, `reliability`,
`exclude`, `models` (per-provider lists), and MoA detection are local
extensions. They may change *which eligible model wins* — an additive term promotes as
well as demotes — but none may relax C1–C3 or C6. Of these,
`llm_assisted_classification` has no caller on either tree: it is accepted in config and
reported, never acted on, and therefore has no control on any surface (C15).

## Verification
```bash
# fastembed (oklab-1)
.venv/bin/python -m pytest tests/agent/test_model_router.py \
    tests/agent/test_model_router_cross_provider.py tests/agent/test_image_routing.py \
    tests/hermes_cli/test_model_routing_cli.py -q

# bro-2 (this copy): 85 passed
/home/bro/hermes-venv/bin/python -m pytest tests/agent/test_model_routing.py -q
```

## Open reconciliation items

Resolved this round — oklab-1 `1f1ab5ad32` (doc truth) + `8a848064c8` (incumbent
identity); bro-2 `5ed600a3b3` (same identity, plus the C9a content-parts fix):
reliability reclassified as operator-declared preference (C4), tie rule and incumbent
identity stated (C5), explain surface named with the dict normative (C10), and the
"only more conservative" claim withdrawn. bro-2 mirrored the incumbent identity and the
joined tie-break key, fixed the content-parts crash, and refreshed the line references in
the commit that carries this revision.

Still open:

1. **C9a** — oklab-1 done (`message_text_and_attachments` → hard `vision` requirement).
   bro-2 still cannot state modality for attachments delivered beside a *string* prompt,
   because `image_routing.py` runs after selection. Owner: bro-2, using oklab-1's helper
   as the reference shape.
2. **C10 `moa_active` dict row** — done on oklab-1 (`routing_report(moa_active=True)`);
   bro-2 returns before a decision object exists, so it needs the same early-skip record.
   Owner: bro-2.
2. **bro-2 defect — fixed, wiring still open** — `classify_request` no longer raises on a
   content-parts list and derives the modality flags from the parts (regression tests in
   `tests/agent/test_model_routing.py`). What remains is item 1's string-prompt path.
   Owner: bro-2, then both.
4. **C8 form (closed)** — delegation-first agreed; see C8 above.
5. **Name collision to document** — C4 `reliability` (routing preference) and the tool
   search's `reliability_rerank` (call-outcome re-rank) are unrelated knobs with similar
   names; never let one config imply the other.
