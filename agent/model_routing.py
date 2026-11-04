"""Deterministic, opt-in model routing.

The selection here is arithmetic, not inference: a request is classified from cheap lexical
signals, ineligible models are dropped by hard constraints, and the survivors are ranked by
expected cost adjusted for capability. No LLM is consulted, so routing a "15 * 8" question
costs nothing beyond a few string scans.

Design constraints that shaped this module:

* **Eligibility before ranking.** A model that cannot accept the request's modality, context
  or tool requirement is never scored — a cheap model that cannot do the job must not win.
* **Hysteresis.** A new model replaces the current one only when it is better by more than
  ``min_improvement``, so a session does not oscillate between near-equal models.
* **Inert by default.** ``smart_model_routing.enabled`` is false unless a user sets it, and
  the disabled path returns ``None`` before touching the agent at all.

Deliberately *not* here: an LLM classifier, embedding similarity, or a learned ranker. Those
were evaluated and rejected — the deterministic path is measurable and cannot silently change
behaviour between runs.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Ordered least -> most capable. Index doubles as the capability tier.
TIERS: Tuple[str, ...] = ("simple", "routine", "moderate", "complex", "advanced", "specialized")

# Roughly 4 characters per token; good enough to compare models, not to bill a customer.
_CHARS_PER_TOKEN = 4
_DEFAULT_OUTPUT_TOKENS = 800
_MAX_OUTPUT_TOKENS = 8000

_CODE_FENCE = re.compile(r"```|\bdef \b|\bclass \b|\bfunction\b|\bSELECT\b|\bimport \b", re.I)
_MATH = re.compile(r"\b\d+\s*[-+*/^]\s*\d+\b|\bintegral\b|\bderivative\b|\bprove\b", re.I)
_MULTI_PART = re.compile(r"(^|\n)\s*(\d+[.)]|[-*])\s+\S", re.M)
_DEEP_REASON = re.compile(
    r"\b(architect|design|refactor|debug|root cause|trade-?off|analyse|analyze|"
    r"compare|evaluate|why does|optimi[sz]e|migrate|benchmark|prove)\b", re.I)
_TRIVIAL = re.compile(r"^\s*(hi|hello|thanks|what is \d+ ?[-+*/] ?\d+|define \w+)\s*[?.!]?\s*$", re.I)


@dataclass(frozen=True)
class RequestProfile:
    """What the request needs, derived deterministically from its text."""

    tier: str
    est_input_tokens: int
    est_output_tokens: int
    needs_tools: bool = False
    needs_vision: bool = False
    needs_audio: bool = False
    needs_structured: bool = False
    reasons: Tuple[str, ...] = ()

    @property
    def tier_rank(self) -> int:
        try:
            return TIERS.index(self.tier)
        except ValueError:
            return 2


@dataclass(frozen=True)
class ModelOption:
    """One candidate. Prices are per 1M tokens in the same currency for all candidates.

    Capability flags are tri-state on purpose: ``None`` means *unknown* (the catalog is
    silent), which is penalised but never treated as a refusal — spec §9.2, "missing
    metadata is not proof that a model lacks a capability". Only an explicit ``False``
    disqualifies a candidate.
    """

    name: str
    provider: str = ""
    context_window: int = 0
    max_output: int = 0
    cost_in: float = 0.0
    cost_out: float = 0.0
    supports_tools: Optional[bool] = None
    supports_vision: Optional[bool] = None
    supports_audio: Optional[bool] = None
    supports_structured: Optional[bool] = None
    tier_rank: int = 3
    deprecated: bool = False
    supports_text_output: Optional[bool] = None


# Skip reasons: the machine-readable answer to "why did nothing switch?". Prose in ``reason``
# is for humans and may be reworded; these strings are the contract two implementations compare.
SKIP_DISABLED = "disabled"                        # smart_model_routing.enabled is false
SKIP_NO_CANDIDATES = "no_candidates"              # no models configured and no catalog entries
SKIP_NO_ELIGIBLE = "no_eligible_model"            # every candidate failed a hard constraint
SKIP_ALREADY_CURRENT = "already_current"          # the current model is the top-ranked one
SKIP_WITHIN_THRESHOLD = "within_stability_threshold"  # a better model exists but not by enough
SKIP_ERROR = "error"                              # routing raised and the current model was kept


@dataclass(frozen=True)
class RoutingDecision:
    model: Optional[str]
    provider: str
    reason: str
    score: float
    profile: RequestProfile
    candidates: int = 0
    # Empty when the decision switches (or when the model is unknown); otherwise one of the
    # SKIP_* constants above.
    skip_reason: str = ""


def estimate_tokens(text: str) -> int:
    """Token estimate delegated to the tree's single estimator.

    ``agent.model_metadata.estimate_tokens_rough`` already exists and is calibrated against
    real usage; a second estimator here would drift from it. The local fallback only covers
    the case where that module cannot be imported (e.g. a trimmed test environment).
    """
    if not text:
        return 0
    try:
        from agent.model_metadata import estimate_tokens_rough
        return max(1, min(int(estimate_tokens_rough(text)), 2_000_000))
    except Exception:  # pragma: no cover - defensive
        return max(1, min(len(text) // _CHARS_PER_TOKEN, 2_000_000))


def classify_request(
    text: str,
    *,
    has_images: bool = False,
    has_audio: bool = False,
    has_files: bool = False,
    needs_tools: bool = False,
) -> RequestProfile:
    """Classify a request into a capability tier using lexical signals only.

    Length is a weak signal on its own — a short question can need deep reasoning and a long
    one can be a copy-paste — so the tier comes from the strongest signal present, and length
    only breaks ties.
    """
    body = text or ""
    tokens = estimate_tokens(body)
    reasons: List[str] = []
    tier = 1  # routine

    if _TRIVIAL.match(body.strip()):
        tier = 0
        reasons.append("trivial phrasing")
    elif _MATH.search(body) and tokens < 400:
        tier = max(tier, 1)
        reasons.append("arithmetic")
    if _CODE_FENCE.search(body):
        tier = max(tier, 3)
        reasons.append("code")
    if _MULTI_PART.search(body):
        tier = max(tier, 2)
        reasons.append("multi-part")
    if _DEEP_REASON.search(body):
        tier = max(tier, 3)
        reasons.append("reasoning verbs")
    if tokens > 2000:
        tier = max(tier, 3)
        reasons.append("long context")
    if tokens > 8000:
        tier = max(tier, 4)
        reasons.append("very long context")
    if needs_tools:
        tier = max(tier, 2)
        reasons.append("tools required")

    modality = has_images or has_audio or has_files
    if modality:
        tier = max(tier, 5)
        reasons.append("modality required")

    return RequestProfile(
        tier=TIERS[min(tier, len(TIERS) - 1)],
        est_input_tokens=tokens,
        est_output_tokens=min(max(_DEFAULT_OUTPUT_TOKENS, tokens // 2), _MAX_OUTPUT_TOKENS),
        needs_tools=needs_tools,
        needs_vision=has_images,
        needs_audio=has_audio,
        needs_structured=False,
        reasons=tuple(reasons),
    )


def estimate_cost(option: ModelOption, profile: RequestProfile) -> float:
    """Expected cost in currency units (prices are per 1M tokens)."""
    return (profile.est_input_tokens * option.cost_in
            + profile.est_output_tokens * option.cost_out) / 1_000_000.0


def is_eligible(option: ModelOption, profile: RequestProfile,
                allow_unknown: bool = True) -> bool:
    """Hard constraints. A model failing any of these is never scored.

    Only an explicit ``False`` capability disqualifies — unknown (``None``) passes here and
    is penalised in :func:`score_model` instead, so a sparse catalog cannot silently empty
    the candidate set.
    """
    if option.deprecated:
        return False
    # A model that only emits images cannot answer a text request.
    if option.supports_text_output is False:
        return False
    if not allow_unknown:
        # Strict mode: an unknown capability counts as unsupported.
        if profile.needs_tools and option.supports_tools is not True:
            return False
        if profile.needs_vision and option.supports_vision is not True:
            return False
        if profile.needs_audio and option.supports_audio is not True:
            return False
    if profile.needs_vision and option.supports_vision is False:
        return False
    if profile.needs_audio and option.supports_audio is False:
        return False
    if profile.needs_structured and option.supports_structured is False:
        return False
    if profile.needs_tools and option.supports_tools is False:
        return False
    if option.context_window and option.context_window < profile.est_input_tokens:
        return False
    if option.max_output and option.max_output < profile.est_output_tokens:
        return False
    # Demanding work gets a capability floor, not merely a penalty. Without this a very cheap
    # low-tier model can absorb a complex request purely on price (spec §8.2: do not choose a
    # cheaper model when it substantially raises the chance of a wrong or retried answer).
    # Scoped to complex/advanced: "specialized" means a modality requirement, which the
    # capability flags above already enforce, not a demand for the most capable model.
    if 3 <= profile.tier_rank <= 4 and option.tier_rank < profile.tier_rank:
        return False
    return True


def score_model(option: ModelOption, profile: RequestProfile, cheapest: float,
                fallback_cost: float = 0.0, cost_weight: float = 0.45) -> float:
    """Higher is better.

    Capability adequacy is a gate, not a bonus: a model below the request's tier is penalised
    hard so cost can never promote an under-powered model. Cost is normalised against the
    cheapest eligible candidate, so scores stay comparable within one request. Unknown
    capability metadata costs a small, fixed amount rather than disqualifying.

    A model that publishes no price is charged the median known price, never zero: treating
    missing pricing as free makes unknown-cost models win every comparison, which is how an
    image model ended up answering an arithmetic question against the live catalog.
    """
    deficit = max(0, profile.tier_rank - option.tier_rank)
    capability = 1.0 - 0.35 * deficit
    unknown = 0.0
    if profile.needs_tools and option.supports_tools is None:
        unknown += 0.15
    if profile.needs_vision and option.supports_vision is None:
        unknown += 0.15
    if profile.needs_audio and option.supports_audio is None:
        unknown += 0.15
    has_cost = bool(option.cost_in or option.cost_out)
    cost = estimate_cost(option, profile) if has_cost else fallback_cost
    cost_penalty = 0.0 if cheapest <= 0 else min(0.9, cost_weight * (cost / cheapest - 1.0))
    return capability - unknown - cost_penalty



def rank_candidates(
    options: Sequence[ModelOption],
    profile: RequestProfile,
    settings: Optional["RoutingSettings"] = None,
) -> Tuple[List[Tuple[float, ModelOption]], float, float]:
    """Rank eligible candidates. The single scoring path shared by selection and ``explain``.

    Returns ``(ranked, cheapest, fallback_cost)``. Keeping one implementation is deliberate:
    an explanation that disagrees with the real choice is worse than no explanation.
    """
    st = settings or RoutingSettings()
    cost_weight = st.cost_weight if st.cost_optimization else 0.0
    eligible = [o for o in options if is_eligible(o, profile, st.allow_unknown_metadata)]
    costs = [estimate_cost(o, profile) for o in eligible if (o.cost_in or o.cost_out)]
    cheapest = min(costs) if costs else 0.0
    fallback_cost = statistics.median(costs) if costs else 0.0
    ranked = sorted(
        ((score_model(o, profile, cheapest, fallback_cost, cost_weight), o) for o in eligible),
        key=lambda pair: (-pair[0], pair[1].name))
    return ranked, cheapest, fallback_cost


def select_model(
    options: Sequence[ModelOption],
    profile: RequestProfile,
    *,
    current: Optional[str] = None,
    min_improvement: float = 0.15,
    settings: Optional["RoutingSettings"] = None,
) -> RoutingDecision:
    """Pick the best eligible model, holding the current one unless clearly beaten."""
    st = settings or RoutingSettings()
    threshold = st.stability_threshold if settings is not None else min_improvement
    if not options:
        return RoutingDecision(
            model=None, provider="", reason="no candidate models configured",
            score=0.0, profile=profile, candidates=0, skip_reason=SKIP_NO_CANDIDATES)
    ranked, _cheapest, _fallback = rank_candidates(options, profile, st)
    if not ranked:
        return RoutingDecision(
            model=None, provider="", reason="no eligible model for this request",
            score=0.0, profile=profile, candidates=0, skip_reason=SKIP_NO_ELIGIBLE)

    best_score, best = ranked[0]
    if current:
        for rank_score, option in ranked:
            if option.name == current:
                if option.name == best.name:
                    return RoutingDecision(
                        model=current, provider=option.provider,
                        reason="current model is already the best for this request",
                        score=rank_score, profile=profile, candidates=len(ranked),
                        skip_reason=SKIP_ALREADY_CURRENT)
                if rank_score >= best_score - threshold:
                    return RoutingDecision(
                        model=current, provider=option.provider,
                        reason=f"keeping current model (within {threshold:.2f} of best)",
                        score=rank_score, profile=profile, candidates=len(ranked),
                        skip_reason=SKIP_WITHIN_THRESHOLD)
                break
    reason = (f"best of {len(ranked)} eligible for {profile.tier}"
              + (f" ({', '.join(profile.reasons)})" if profile.reasons else ""))
    return RoutingDecision(
        model=best.name, provider=best.provider, reason=reason,
        score=best_score, profile=profile, candidates=len(ranked))


def routing_config(agent: Any) -> Dict[str, Any]:
    """Read ``smart_model_routing`` from the agent's config; never raises."""
    try:
        cfg = getattr(agent, "config", None)
        if not isinstance(cfg, dict):
            return {}
        raw = cfg.get("smart_model_routing")
        return raw if isinstance(raw, dict) else {}
    except Exception:  # pragma: no cover - defensive
        return {}


@dataclass(frozen=True)
class RoutingSettings:
    """The one authoritative ``smart_model_routing`` contract.

    Reconciled with the parallel implementation on the lab box, which used
    ``stability_threshold``/``strategy``/``allow_unknown_metadata``. Both spellings are
    accepted so an existing config keeps working; the lab spelling wins when both are present
    because it is the one the CLI documents.

    ``strategy``:
      * ``automatic`` — capability floor plus balanced cost penalty (default)
      * ``cost``      — stronger cost penalty; still honours the capability floor
      * ``quality``   — cost ignored; pick the most capable eligible model
    """

    enabled: bool = False
    strategy: str = "automatic"
    stability_threshold: float = 0.15
    allow_unknown_metadata: bool = True
    cost_optimization: bool = True
    preferred_models_only: bool = False
    # Key presence, not value: a one-model pool is only widened when the operator
    # actually wrote the key. A default must not silently turn a no-op into a switch.
    preferred_models_only_explicit: bool = False
    models: Tuple[Any, ...] = ()

    @property
    def cost_weight(self) -> float:
        return {"cost": 0.75, "quality": 0.0}.get(self.strategy, 0.45)


def parse_settings(raw: Any) -> RoutingSettings:
    """Normalise any historical spelling into :class:`RoutingSettings`. Never raises."""
    if not isinstance(raw, dict):
        return RoutingSettings()
    strategy = str(raw.get("strategy") or "automatic").strip().lower()
    if strategy not in ("automatic", "cost", "quality"):
        strategy = "automatic"
    # ``stability_threshold`` (lab) and ``min_improvement`` (earlier) are the same knob.
    threshold = raw.get("stability_threshold", raw.get("min_improvement", 0.15))
    try:
        threshold = max(0.0, float(threshold))
    except (TypeError, ValueError):
        threshold = 0.15
    models = raw.get("models")
    return RoutingSettings(
        enabled=bool(raw.get("enabled", False)),
        strategy=strategy,
        stability_threshold=threshold,
        allow_unknown_metadata=bool(raw.get("allow_unknown_metadata", True)),
        cost_optimization=bool(raw.get("cost_optimization", True)),
        preferred_models_only=bool(raw.get("preferred_models_only", False)),
        preferred_models_only_explicit="preferred_models_only" in raw,
        models=tuple(models) if isinstance(models, (list, tuple)) else (),
    )


def _tri_flag(entry: Dict[str, Any], key: str) -> Optional[bool]:
    """True/False when stated, None when the entry says nothing (unknown, not unsupported)."""
    if key not in entry or entry[key] is None:
        return None
    return bool(entry[key])


def _catalog_info(provider: str, name: str) -> Any:
    """Cache-only catalog metadata for one model; ``None`` when the catalog is silent."""
    if not provider or not name:
        return None
    try:
        from agent import models_dev
        return models_dev.get_model_info(provider, name, allow_network=False)
    except Exception:  # pragma: no cover - catalog optional in trimmed trees
        logger.debug("model routing: catalog lookup failed for %s/%s", provider, name,
                     exc_info=True)
        return None


def _catalog_attr(info: Any, attr: str, default: Any = 0) -> Any:
    if info is None:
        return default
    return getattr(info, attr, default) or default


def options_from_config(cfg: Dict[str, Any]) -> List[ModelOption]:
    """Build candidates from ``models: [{name, provider, ...}]``; malformed entries skipped.

    ``models[]`` is canonical, and a field stated on the entry always wins. A field the entry
    omits is filled from the models.dev entry for the same provider/model, so overriding just
    the price of a catalog model does not silently discard its context window, modality or
    tool support. A field neither source states stays *unknown* (``None``), never ``False`` —
    missing metadata is not proof a model lacks a capability.
    """
    out: List[ModelOption] = []
    raw = cfg.get("models")
    if not isinstance(raw, (list, tuple)):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        provider = str(entry.get("provider") or "").strip()
        info = _catalog_info(provider, name)
        inputs = tuple(_catalog_attr(info, "input_modalities", ()) or ())
        outputs = tuple(_catalog_attr(info, "output_modalities", ()) or ())

        def stated(key: str) -> bool:
            return key in entry and entry[key] is not None

        def flag(key: str, from_catalog: Optional[bool]) -> Optional[bool]:
            return _tri_flag(entry, key) if key in entry else from_catalog

        try:
            out.append(ModelOption(
                name=name,
                provider=provider,
                context_window=int(entry["context_window"]) if stated("context_window")
                else int(_catalog_attr(info, "context_window")),
                max_output=int(entry["max_output"]) if stated("max_output")
                else int(_catalog_attr(info, "max_output")),
                cost_in=float(entry["cost_in"]) if stated("cost_in")
                else float(_catalog_attr(info, "cost_input")),
                cost_out=float(entry["cost_out"]) if stated("cost_out")
                else float(_catalog_attr(info, "cost_output")),
                supports_tools=flag(
                    "supports_tools",
                    (bool(_catalog_attr(info, "tool_call", False)) or None)
                    if info is not None else None),
                supports_vision=flag(
                    "supports_vision", (("image" in inputs) if inputs else None)),
                supports_audio=flag(
                    "supports_audio", (("audio" in inputs) if inputs else None)),
                supports_structured=flag(
                    "supports_structured",
                    (bool(_catalog_attr(info, "structured_output", False)) or None)
                    if info is not None else None),
                tier_rank=int(entry["tier_rank"]) if stated("tier_rank")
                else (_tier_from_capabilities(info) if info is not None else 3),
                deprecated=bool(entry["deprecated"]) if stated("deprecated")
                else (str(_catalog_attr(info, "status", "")).lower() == "deprecated"),
                supports_text_output=flag(
                    "supports_text_output", (("text" in outputs) if outputs else None)),
            ))
        except (TypeError, ValueError):
            logger.warning("smart_model_routing: skipping malformed model entry %r", entry)
    return out


def _tier_from_capabilities(caps: Any) -> int:
    """Map catalog metadata onto the 0..5 capability tier.

    Reasoning models and large-context models are treated as more capable; this is a
    heuristic on published metadata, not a quality claim about any specific model.
    """
    tier = 2
    if getattr(caps, "reasoning", False):
        tier = max(tier, 4)
    if getattr(caps, "tool_call", False):
        tier = max(tier, 3)
    ctx = int(getattr(caps, "context_window", 0) or 0)
    if ctx >= 200_000:
        tier = max(tier, 4)
    elif ctx >= 100_000:
        tier = max(tier, 3)
    if getattr(caps, "structured_output", False):
        tier = max(tier, 3)
    return min(tier, 5)


def options_from_catalog(provider: str, models: Optional[Sequence[str]] = None) -> List[ModelOption]:
    """Build candidates from the models.dev catalog for one provider.

    Never raises and never blocks on the network: ``allow_network=False`` means only the
    on-disk cache is consulted, so a turn never waits on a catalog fetch.
    """
    out: List[ModelOption] = []
    if not provider:
        return out
    try:
        from agent import models_dev
        names = list(models) if models else list(models_dev.list_provider_models(
            provider, allow_network=False) or [])
    except Exception:
        logger.debug("model routing: catalog unavailable for %r", provider, exc_info=True)
        return out
    for name in names:
        try:
            info = models_dev.get_model_info(provider, name, allow_network=False)
        except Exception:
            info = None
        if info is None:
            # Present in the catalog listing but without metadata: still a candidate, with
            # everything unknown. Excluding it would be "missing metadata == unsupported".
            out.append(ModelOption(name=name, provider=provider))
            continue
        inputs = tuple(getattr(info, "input_modalities", ()) or ())
        outputs = tuple(getattr(info, "output_modalities", ()) or ())
        out.append(ModelOption(
            name=name,
            provider=provider,
            context_window=int(getattr(info, "context_window", 0) or 0),
            max_output=int(getattr(info, "max_output", 0) or 0),
            cost_in=float(getattr(info, "cost_input", 0.0) or 0.0),
            cost_out=float(getattr(info, "cost_output", 0.0) or 0.0),
            supports_tools=bool(getattr(info, "tool_call", False)) or None,
            supports_vision=(("image" in inputs) if inputs else None),
            supports_audio=(("audio" in inputs) if inputs else None),
            supports_structured=bool(getattr(info, "structured_output", False)) or None,
            tier_rank=_tier_from_capabilities(info),
            deprecated=str(getattr(info, "status", "") or "").lower() == "deprecated",
            supports_text_output=(("text" in outputs) if outputs else None),
        ))
    return out


def moa_active(agent: Any) -> bool:
    """True when the agent is currently running the Mixture-of-Agents facade.

    ``bind_moa_runtime`` pins ``provider`` and ``requested_provider`` to ``moa``; the preset is
    a virtual runtime, not a catalog model. Routing must not replace it with a scored candidate
    — the fan-out, its reference models and the aggregator are the user's explicit choice.
    Both fields are checked because a ``/model`` switch pins one before the other.
    """
    for attr in ("provider", "requested_provider"):
        try:
            if str(getattr(agent, attr, "") or "").strip().lower() == "moa":
                return True
        except Exception:  # pragma: no cover - defensive
            continue
    return False


def maybe_apply_model_routing(agent: Any, text: str) -> Optional[RoutingDecision]:
    """Apply routing for one turn when enabled. Returns the decision, or None if inert.

    Switches through the agent's existing ``switch_model`` so credential pools, prompt-cache
    flags and the compressor are rebuilt exactly as a manual switch would rebuild them. Any
    failure is logged and swallowed: routing must never break a turn.
    """
    if moa_active(agent):
        # MoA is a runtime the user chose explicitly; scoring a replacement model would
        # silently dismantle the fan-out. Both implementations skip routing here.
        logger.debug("model routing skipped: mixture-of-agents is active")
        return None
    cfg = routing_config(agent)
    st = parse_settings(cfg)
    if not st.enabled:
        return None
    try:
        options = options_from_config(cfg)
        # A provider that lists exactly one model offers no choice: no catalog scan, no
        # routing decision — unless the operator wrote the key, which makes the intent explicit.
        if len(options) == 1 and not st.preferred_models_only_explicit:
            logger.debug("model routing skipped: single-model preferred pool (%s)",
                         options[0].name)
            return None
        scope = "preferred" if st.preferred_models_only else "preferred+catalog"
        if not options:
            if scope == "preferred":
                # "Preferred models only" with no preferred list means there is nothing to
                # choose between. Falling back to the whole catalog would do the exact
                # opposite of what the setting asks for.
                logger.debug("model routing skipped: preferred_models_only with no models list")
                return None
            # No explicit list: derive candidates from the cached catalog for the current
            # provider (cache only — a turn must never wait on a network fetch).
            options = options_from_catalog(str(getattr(agent, "provider", "") or ""))
        if not options:
            return None
        profile = classify_request(text or "")
        decision = select_model(
            options, profile,
            current=getattr(agent, "model", None), settings=st)
        if not decision.model or decision.model == getattr(agent, "model", None):
            if decision.skip_reason:
                logger.debug("model routing: keeping %s (%s)", getattr(agent, "model", None),
                             decision.skip_reason)
            return decision
        from agent.agent_runtime_helpers import switch_model
        previous = getattr(agent, "model", None)
        switch_model(agent, decision.model, decision.provider or agent.provider)
        logger.info("model routing: %s -> %s (%s)", previous, decision.model, decision.reason)
        return decision
    except Exception:
        logger.warning("model routing failed; keeping the current model", exc_info=True)
        return None

def explain_routing(
    config: Dict[str, Any],
    text: str,
    *,
    current: Optional[str] = None,
    provider: str = "",
    options: Optional[Sequence[ModelOption]] = None,
    has_images: bool = False,
    has_audio: bool = False,
    has_files: bool = False,
    needs_tools: bool = False,
) -> Dict[str, Any]:
    """Explain a routing decision without performing one.

    Every number here comes from the same functions selection uses, so the report cannot
    disagree with the behaviour it describes.
    """
    st = parse_settings(config)
    prof = classify_request(text or "", has_images=has_images, has_audio=has_audio,
                            has_files=has_files, needs_tools=needs_tools)
    cands = list(options) if options is not None else options_from_config(config)
    source = "config"
    if not cands:
        cands = options_from_catalog(provider)
        source = "catalog"
    ranked, cheapest, fallback_cost = rank_candidates(cands, prof, st)
    decision = select_model(cands, prof, current=current, settings=st)
    # Keyed by identity: two providers can expose the same model name, and a name-keyed map
    # made both rows print the same score.
    scores = {id(o): sc for sc, o in ranked}
    rows = []
    for opt in cands:
        eligible = is_eligible(opt, prof, st.allow_unknown_metadata)
        priced = bool(opt.cost_in or opt.cost_out)
        unknown = [k for k, v in (
            ("tools", opt.supports_tools), ("vision", opt.supports_vision),
            ("audio", opt.supports_audio), ("text_out", opt.supports_text_output),
        ) if v is None]
        rows.append({
            "model": opt.name,
            "provider": opt.provider,
            "eligible": eligible,
            "score": round(scores[id(opt)], 4) if eligible else None,
            "tier": opt.tier_rank,
            "cost_in": opt.cost_in,
            "cost_out": opt.cost_out,
            "priced": priced,
            "charged_as": ("published" if priced else "median_unknown"),
            "unknown_capabilities": unknown,
            "excluded_reason": None if eligible else _exclusion_reason(opt, prof, st),
        })
    rows.sort(key=lambda r: (not r["eligible"], -(r["score"] or -99), r["model"]))
    return {
        "enabled": st.enabled,
        "settings": {
            "strategy": st.strategy, "stability_threshold": st.stability_threshold,
            "allow_unknown_metadata": st.allow_unknown_metadata,
            "cost_optimization": st.cost_optimization,
            "preferred_models_only": st.preferred_models_only,
        },
        "candidate_source": source,
        "request": {
            "tier": prof.tier, "est_input_tokens": prof.est_input_tokens,
            "est_output_tokens": prof.est_output_tokens,
            "needs_tools": prof.needs_tools, "needs_vision": prof.needs_vision,
            "needs_audio": prof.needs_audio, "reasons": list(prof.reasons),
        },
        "pricing": {"cheapest_eligible": round(cheapest, 6),
                    "median_fallback": round(fallback_cost, 6),
                    "policy": "unpriced models are charged the median known price, never zero"},
        "selected": decision.model,
        "selected_provider": decision.provider,
        "reason": decision.reason,
        # Machine-readable counterpart of ``reason``: one of the SKIP_* constants, or "" when
        # the decision switches. Stable across implementations; the prose is not. A disabled
        # section reports "disabled" even though the hypothetical selection is still computed.
        "skip_reason": decision.skip_reason if st.enabled else SKIP_DISABLED,
        "current": current,
        "kept_current": bool(decision.model and decision.model == current),
        "eligible_count": len(ranked),
        "candidates": rows,
    }


def _exclusion_reason(option: ModelOption, profile: RequestProfile,
                      st: "RoutingSettings") -> str:
    if option.deprecated:
        return "deprecated"
    if option.supports_text_output is False:
        return "cannot emit text"
    if profile.needs_vision and option.supports_vision is False:
        return "no vision input"
    if profile.needs_audio and option.supports_audio is False:
        return "no audio input"
    if profile.needs_tools and option.supports_tools is False:
        return "no tool calling"
    if option.context_window and option.context_window < profile.est_input_tokens:
        return f"context {option.context_window} < {profile.est_input_tokens}"
    if option.max_output and option.max_output < profile.est_output_tokens:
        return f"max_output {option.max_output} < {profile.est_output_tokens}"
    if 3 <= profile.tier_rank <= 4 and option.tier_rank < profile.tier_rank:
        return f"tier {option.tier_rank} below required {profile.tier_rank}"
    if not st.allow_unknown_metadata:
        needs = (
            ("tools", option.supports_tools, profile.needs_tools),
            ("vision", option.supports_vision, profile.needs_vision),
            ("audio", option.supports_audio, profile.needs_audio),
            ("structured", option.supports_structured, profile.needs_structured),
        )
        for label, value, required in needs:
            if required and value is None:
                return f"{label} capability unknown (strict mode)"
    return "not eligible"
