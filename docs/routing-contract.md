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

## C7 — Mixture-of-agents
If MoA is active for the turn, routing is skipped: the operator chose a
fan-out and replacing the model would silently dismantle it.

## C8 — Cross-provider selection
A switch may cross providers, but must resolve credentials through the target
provider's existing profile and rebuild credential pools, prompt-cache flags and
the compressor exactly as a manual switch does. Provider pools are never mixed:
a cross-provider move clears the previous provider's pool rather than borrowing
from it. fastembed performs the re-resolution itself at the turn boundary;
bro-2 delegates to the agent's existing `switch_model`, which rebuilds the pools,
prompt-cache flags and compressor exactly as a manual switch would (verified in
`agent/model_routing.py: maybe_apply_model_routing`). Neither side may switch by
mutating the model name alone.

## C9 — Multimodal continuity — **NOT IMPLEMENTED, both sides (status, not contract text)**

Called "settled" in an earlier revision. Measured, it is not implemented anywhere, and
both sides fail differently.

**C9a — input modality.** The requirement must be derived from the attachments present
at the turn boundary, never from message text. Falsifier: a message carrying an image
part must not select a model whose input modality excludes images, and must not raise.

- *oklab-1*: NOT IMPLEMENTED. `agent/conversation_loop.py:1585` coerces the message with
  `str(...)` before `route_turn`, so attachments never reach selection; `expects_image`
  has test-only callers. A text-only model can win an image-output request because an
  UNKNOWN output modality passes the filter (`model_router.py:904`) and `SPEC_FIELDS`
  (`:339-353`) cannot declare image output.
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
the original attachment and require image-output capability. Neither side records a
modality on the decision (`Decision` has no such field), so this cannot be tested yet.

**Invariant that does hold:** neither side may satisfy a modality requirement by
dropping the attachment.

## C10 — Explainability
Every decision carries a human-readable reason and a machine `skip_reason`, drawn
from one vocabulary both sides accept: `no_candidates`, `no_eligible_model`,
`already_current`, `within_stability_threshold`, `moa_active`, `disabled`, `error`
(oklab-1 pins the set in `SKIP_REASONS`; `moa_active` is reserved — nothing emits it
yet, because the MoA path does not consult the router).

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

## Extensions that are allowed to differ
`switch_on_cost`, `llm_assisted_classification`, `weights`, `reliability`,
`exclude`, `models` (per-provider lists), and MoA detection are local
extensions. They may change *which eligible model wins* — an additive term promotes as
well as demotes — but none may relax C1–C3 or C6.

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

1. **C9a wiring** — the one genuinely unimplemented clause. Both sides must derive the
   modality requirement from attachments at the turn boundary. Owner: both.
2. **bro-2 defect — fixed, wiring still open** — `classify_request` no longer raises on a
   content-parts list and derives the modality flags from the parts (regression tests in
   `tests/agent/test_model_routing.py`). What remains is item 1's string-prompt path.
   Owner: bro-2, then both.
3. **C8 form** — self-managed switching there vs `switch_model` delegation here. Both
   are acceptable in behaviour; decide whether the delegation form becomes the shared
   contract. Owner: both.
4. **Name collision to document** — C4 `reliability` (routing preference) and the tool
   search's `reliability_rerank` (call-outcome re-rank) are unrelated knobs with similar
   names; never let one config imply the other.
