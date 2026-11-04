# Smart model routing — shared behavioural contract

Two Hermes deployments run two independently written routers:

| | `hermes-agent-fastembed` (oklab-1) | `hermes-agent` (bro-2) |
|---|---|---|
| module | `agent/model_router.py` (2 444 lines) | `agent/model_routing.py` (741 lines) |
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
Score is a bounded combination of capability fit, cost, context fit, continuity,
and (fastembed only) reliability, using `strategy` (`automatic` | `cost` |
`quality`) to set the cost weight. Reliability is a per-model 0..1 success rate
and only ever *demotes*: it cannot promote a model past an eligibility rule.

## C5 — Stability (anti-flapping)
The incumbent model keeps the turn unless a rival beats it by more than
`stability_threshold` (default `0.15`; `min_improvement` is the same knob under
its older name) **or** a hard constraint forces the switch. Ties and borderline
wins keep the current model — a switch costs a prompt-cache invalidation.

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

## C9 — Multimodal continuity
Image/audio/pdf input requires a model whose input modality supports it, and an
image-edit follow-up keeps the original image attached and requires image output
capability. Neither side may satisfy a modality requirement by dropping the
attachment.

## C10 — Explainability
Every decision carries a human-readable reason and a machine `skip_reason`
(`no_candidates`, `no_eligible_model`, `already_current`,
`within_stability_threshold`, `moa_active`, `disabled`, `error`, …) — these are
the exact tokens bro-2 emits in `agent/model_routing.py`. Exclusions must be
explainable per model: "why not X" is part of the contract, not a debug nicety.

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
extensions. Each only makes selection *more* conservative; none may relax C1–C10.

## Verification
```bash
# fastembed (oklab-1)
.venv/bin/python -m pytest tests/agent/test_model_router.py \
    tests/agent/test_model_router_cross_provider.py tests/agent/test_image_routing.py \
    tests/hermes_cli/test_model_routing_cli.py -q

# bro-2 (this copy): 71 passed
/home/bro/hermes-venv/bin/python -m pytest tests/agent/test_model_routing.py -q
```

## Open reconciliation items
1. Reliability as a ranking factor exists only in fastembed's model router; either
   port it as an opt-in weight or record it as a permanent extension.
2. Turn-boundary cross-provider switching (C8) is implemented differently on the
   two sides (self-managed there, `switch_model` delegation here); decide whether
   the delegation form is acceptable as the shared contract.
3. Explain surface: two CLI shapes (`explain` vs `explain_routing`); unify the
   output fields before sharing automation that parses them.
