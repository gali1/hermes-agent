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

#: Phrases that ask for an image BACK (C9b). Requires a generation verb within 40 chars
#: of an artefact noun, because an unanchored "image" match fires on every attachment
#: question and would route description requests through an image-output gate.
_IMAGE_OUTPUT_SIGNALS = re.compile(
    r"(?:\b(?:generate|create|make|draw|paint|render|produce|design)\b[^.?!\n]{0,40}"
    r"\b(?:image|picture|illustration|drawing|artwork|logo|icon|poster|avatar)\b)",
    re.I,
)


@dataclass(frozen=True)
class RequestProfile:
    """What the request needs, derived deterministically from its text.

    The last four fields are cost-model assumptions, not requirements: they describe how much
    of the prompt is expected to hit the provider's prompt cache, how many tokens the model is
    expected to spend thinking, and how many retries a request of this kind is expected to
    take. All default to zero so a profile built by :func:`classify_request` costs exactly what
    it always did; a caller that knows better (a cached agent loop, a retry budget) can set
    them and :func:`estimate_cost` will price the *expected total* rather than one attempt.
    """

    tier: str
    est_input_tokens: int
    est_output_tokens: int
    needs_tools: bool = False
    needs_vision: bool = False
    needs_audio: bool = False
    needs_structured: bool = False
    # C9b: the turn asks for an IMAGE back, not merely about one. Kept separate from
    # needs_vision because one is an input requirement and the other an output one.
    needs_image_output: bool = False
    reasons: Tuple[str, ...] = ()
    cached_input_ratio: float = 0.0
    cache_write_ratio: float = 0.0
    reasoning_tokens: int = 0
    expected_retries: int = 0

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
    # C9b (parity with oklab-1): an output modality must be DECLARABLE, or a model that
    # can only emit text stays unknown (None) and therefore eligible for an image-output
    # request. Unknown stays different from unsupported: only an explicit False excludes.
    supports_image_output: Optional[bool] = None
    supports_audio_output: Optional[bool] = None
    # Prompt-cache prices, per 1M tokens, when the catalog publishes them. ``None`` means
    # unpublished — never zero — so :func:`estimate_cost` falls back to the normal input rate
    # instead of pretending cached tokens are free.
    cost_cache_read: Optional[float] = None
    cost_cache_write: Optional[float] = None


# Skip reasons: the machine-readable answer to "why did nothing switch?". Prose in ``reason``
# is for humans and may be reworded; these strings are the contract two implementations compare.
SKIP_DISABLED = "disabled"                        # smart_model_routing.enabled is false
SKIP_NO_CANDIDATES = "no_candidates"              # no models configured and no catalog entries
SKIP_NO_ELIGIBLE = "no_eligible_model"            # every candidate failed a hard constraint
SKIP_ALREADY_CURRENT = "already_current"          # the current model is the top-ranked one
SKIP_WITHIN_THRESHOLD = "within_stability_threshold"  # a better model exists but not by enough
SKIP_ERROR = "error"                              # routing raised and the current model was kept
# The agent is running the Mixture-of-Agents facade. Routing is skipped deliberately: MoA is a
# runtime the user chose, and scoring a replacement would dismantle the fan-out. This is a
# distinct reason from "disabled" — a status view that reported "disabled" while MoA was active
# would be describing the wrong cause.
SKIP_MOA_ACTIVE = "moa_active"



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


def _content_parts(payload: Any) -> Tuple[str, bool, bool, bool]:
    """Split a message payload into ``(text, has_images, has_audio, has_files)``.

    A turn boundary may hand over a multi-part content list rather than a string — vision on
    the OpenAI-compatible server, attachments on the API/gateway path. Calling ``str()`` on
    that list would count an image payload as prompt text; refusing it raised
    ``AttributeError: 'list' object has no attribute 'strip'`` inside a swallowed ``except``,
    which made every multimodal turn routing-inert by accident. So the parts are read here and
    the modality flags come from what is actually attached, never from the words around it.
    """
    if payload is None:
        return "", False, False, False
    if isinstance(payload, str):
        return payload, False, False, False

    flags = {"image": False, "audio": False, "file": False}
    chunks: List[str] = []

    def _walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            chunks.append(value)
            return
        if isinstance(value, dict):
            # Tool results arrive in this envelope; the parts are inside ``content``.
            if value.get("_multimodal") is True and isinstance(value.get("content"), list):
                _walk(value["content"])
                return
            kind = str(value.get("type") or "").lower()
            if kind in {"text", "input_text", "output_text"}:
                chunk = value.get("text")
                if isinstance(chunk, str):
                    chunks.append(chunk)
                return
            if kind in {"image_url", "input_image", "image"}:
                flags["image"] = True
                return
            if kind in {"input_audio", "audio"}:
                flags["audio"] = True
                return
            if kind in {"file", "input_file", "document"}:
                flags["file"] = True
                return
            # Generic envelope ({"content": [...]}, {"parts": [...]}) or a bare {"text": ...}.
            for key in ("text", "content", "parts"):
                if key in value:
                    _walk(value[key])
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)
            return
        chunks.append(str(value))

    _walk(payload)
    return "\n".join(c for c in chunks if c), flags["image"], flags["audio"], flags["file"]


def _string_image_refs(text: str) -> bool:
    """True when a *string* prompt names an image that will be attached for the turn.

    The reference shape is the peer's ``message_text_and_attachments()``: modality comes from
    what is attached, never from the words. For a string payload the attachment is expressed as
    a local path or an http(s) URL, which is exactly what ``image_routing.extract_image_refs``
    later turns into content parts — so routing asks the same question with the same function
    instead of inferring vision from prose. A path that does not exist is not an attachment.
    """
    if not isinstance(text, str) or not text:
        return False
    try:
        from agent.image_routing import extract_image_refs
        paths, urls = extract_image_refs(text)
    except Exception:  # pragma: no cover - routing must never break a turn
        return False
    return bool(paths or urls)


def classify_request(
    text: Any,
    *,
    has_images: bool = False,
    has_audio: bool = False,
    has_files: bool = False,
    needs_tools: bool = False,
) -> RequestProfile:
    """Classify a request into a capability tier using lexical signals only.

    Length is a weak signal on its own — a short question can need deep reasoning and a long
    one can be a copy-paste — so the tier comes from the strongest signal present, and length
    only breaks ties. ``text`` may be a string or a content-parts list; modality flags found
    in the parts are OR'd with the caller's explicit flags. A string prompt that references an
    image is the same requirement as an image part: the attachment is derived from the payload,
    not from the surrounding words.
    """
    body, part_images, part_audio, part_files = _content_parts(text)
    # C9b: a generation verb near an artefact noun asks for an image back. Deliberately
    # narrow - "what is in this image?" is an INPUT question and must not trip this.
    needs_image_output = bool(_IMAGE_OUTPUT_SIGNALS.search(body))
    if not part_images and _string_image_refs(text):
        part_images = True
    has_images = has_images or part_images
    has_audio = has_audio or part_audio
    has_files = has_files or part_files
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
        needs_image_output=needs_image_output,
        reasons=tuple(reasons),
    )


def _clamp_ratio(value: Any) -> float:
    """A 0..1 share; anything unparsable is 0 so a bad config cannot inflate a cost."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _clamp_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def estimate_cost(
    option: ModelOption,
    profile: RequestProfile,
    *,
    cached_input_ratio: Optional[float] = None,
    cache_write_ratio: Optional[float] = None,
    reasoning_tokens: Optional[int] = None,
    expected_retries: Optional[int] = None,
) -> float:
    """Expected **total** cost of serving the request, in currency units (prices per 1M tokens).

    Spec §7.4: the number ranking compares must be what the request is expected to cost in
    total for the required quality, not the price of one happy-path attempt. Three terms are
    added on top of the naive ``input * cost_in + output * cost_out``:

    * **Cached input.** The catalog publishes ``cost.cache_read`` / ``cost.cache_write``
      (``ModelInfo.cost_cache_read`` / ``cost_cache_write``), so the prompt is split into
      fresh, cache-read and cache-write shares and each share is priced at its own rate. When
      the catalog publishes no cache price for a model, cached tokens are charged the normal
      input rate — a missing price is never read as free.
    * **Reasoning tokens.** These are billed at the *output* rate: models.dev has **no**
      reasoning-specific price field (only ``cost.input``/``cost.output``/``cache_read``/
      ``cache_write``), and providers bill thinking tokens as output. No field was invented
      here for that reason.
    * **Expected retries.** A retry re-sends the identical prompt, which is a prompt-cache read
      when the model publishes a cache price (and a normal input charge when it does not), and
      regenerates the output. So each expected retry costs
      ``input * cache_read_rate + output * cost_out``.

    Every parameter defaults to ``None``, meaning "take it from *profile*", and those profile
    fields default to zero — so existing callers, and any profile built by
    :func:`classify_request`, get exactly the one-attempt cost they got before.

    This function prices *published* rates only. Charging an unpriced model the median known
    price stays in :func:`rank_candidates`/:func:`score_model`, where the median is known.
    """
    read_ratio = _clamp_ratio(
        profile.cached_input_ratio if cached_input_ratio is None else cached_input_ratio)
    write_ratio = _clamp_ratio(
        profile.cache_write_ratio if cache_write_ratio is None else cache_write_ratio)
    reasoning = _clamp_count(
        profile.reasoning_tokens if reasoning_tokens is None else reasoning_tokens)
    retries = _clamp_count(
        profile.expected_retries if expected_retries is None else expected_retries)

    # Overlapping shares would bill more than the prompt: scale both down proportionally.
    total_ratio = read_ratio + write_ratio
    if total_ratio > 1.0:
        read_ratio, write_ratio = read_ratio / total_ratio, write_ratio / total_ratio

    prompt_tokens = max(0.0, float(profile.est_input_tokens))
    read_tokens = prompt_tokens * read_ratio
    write_tokens = prompt_tokens * write_ratio
    fresh_tokens = prompt_tokens - read_tokens - write_tokens

    read_rate = option.cost_cache_read if option.cost_cache_read is not None else option.cost_in
    write_rate = option.cost_cache_write if option.cost_cache_write is not None else option.cost_in
    output_tokens = max(0.0, float(profile.est_output_tokens)) + reasoning

    input_cost = (fresh_tokens * option.cost_in
                  + read_tokens * read_rate
                  + write_tokens * write_rate)
    output_cost = output_tokens * option.cost_out
    first_attempt = (input_cost + output_cost) / 1_000_000.0
    if not retries:
        return first_attempt
    # A retry re-sends the same prompt (cache hit when priced) and regenerates the output.
    retry = (prompt_tokens * read_rate + output_tokens * option.cost_out) / 1_000_000.0
    return first_attempt + retries * retry


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
    # C9b: an image-output request cannot be served by a model that declares it cannot
    # emit images. Undeclared (None) still passes here and is penalised elsewhere, so a
    # sparse catalogue cannot silently empty the candidate set.
    if profile.needs_image_output and option.supports_image_output is False:
        return False
    if not allow_unknown:
        # Strict mode: an unknown capability counts as unsupported.
        if profile.needs_tools and option.supports_tools is not True:
            return False
        if profile.needs_vision and option.supports_vision is not True:
            return False
        if profile.needs_audio and option.supports_audio is not True:
            return False
        if profile.needs_image_output and option.supports_image_output is not True:
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


def _candidate_key(option: ModelOption) -> str:
    """Provider-qualified sort key, spelled exactly as the other host's ``ModelCandidate.key``.

    Sorting on the joined string (not on a ``(provider, name)`` tuple) matters: when one
    provider name is a prefix of another the delimiter orders the two lists differently, so
    the tuple form and the string form would disagree on the same tie.
    """
    return f"{option.provider}/{option.name}"


def widen_for_tools(
    options: Sequence[ModelOption],
    profile: RequestProfile,
    settings: Optional["RoutingSettings"] = None,
    fallback_options: Optional[Sequence[ModelOption]] = None,
) -> Tuple[List[ModelOption], bool]:
    """Apply spec §7.7: a preferred list must not block a tool-capable model. Returns ``(pool, widened)``.

    ``needs_tools`` gates eligibility, but when ``preferred_models_only`` is set and none of the
    preferred models is *known* to call tools, the preferred pool can only ever yield "no eligible
    model" — the request fails even though the catalog holds models that can do the job. When the
    operator opts in (``allow_tool_fallback``), the pool is widened to the catalog and narrowed to
    models whose ``supports_tools`` is explicitly ``True``: "reliably produces the call" is the
    requirement, so an unknown capability is not good enough here even though it stays eligible in
    the ordinary path (§9.2, where missing metadata must not disqualify).

    Nothing happens when any precondition is absent: no tool requirement, no preferred-only
    restriction, no opt-in, no fallback pool to widen to, or a preferred model that *does* declare
    tool support. A widened pool that is empty is discarded, so widening can never leave fewer
    candidates than the caller passed in.
    """
    st = settings or RoutingSettings()
    pool = list(options)
    if not (profile.needs_tools and st.preferred_models_only and st.allow_tool_fallback):
        return pool, False
    if not pool or not fallback_options:
        return pool, False
    # The compatibility check uses the real flag: ``True`` means the catalog says it can call
    # tools. ``None`` (unknown) is not evidence of capability.
    if any(o.supports_tools is True and is_eligible(o, profile, st.allow_unknown_metadata)
           for o in pool):
        return pool, False

    seen = {_candidate_key(o) for o in pool}
    widened_pool = [o for o in pool if o.supports_tools is True]
    for option in fallback_options:
        key = _candidate_key(option)
        if key in seen or option.supports_tools is not True:
            continue
        seen.add(key)
        widened_pool.append(option)
    if not widened_pool:
        return pool, False
    return widened_pool, True


def rank_candidates(
    options: Sequence[ModelOption],
    profile: RequestProfile,
    settings: Optional["RoutingSettings"] = None,
    *,
    fallback_options: Optional[Sequence[ModelOption]] = None,
) -> Tuple[List[Tuple[float, ModelOption]], float, float]:
    """Rank eligible candidates. The single scoring path shared by selection and ``explain``.

    Returns ``(ranked, cheapest, fallback_cost)``. Keeping one implementation is deliberate:
    an explanation that disagrees with the real choice is worse than no explanation.
    ``fallback_options`` is the catalog pool used for the §7.7 tool fallback; without it the
    ranking is exactly what it always was.
    """
    st = settings or RoutingSettings()
    cost_weight = st.cost_weight if st.cost_optimization else 0.0
    pool, _widened = widen_for_tools(options, profile, st, fallback_options)
    eligible = [o for o in pool if is_eligible(o, profile, st.allow_unknown_metadata)]
    costs = [estimate_cost(o, profile) for o in eligible if (o.cost_in or o.cost_out)]
    cheapest = min(costs) if costs else 0.0
    fallback_cost = statistics.median(costs) if costs else 0.0
    ranked = sorted(
        ((score_model(o, profile, cheapest, fallback_cost, cost_weight), o) for o in eligible),
        # Total order on the provider-qualified key: same score and same model name from two
        # providers must still resolve the same way on both hosts. No latency term:
        # ``latency_ms`` is not read anywhere.
        key=lambda pair: (-pair[0], _candidate_key(pair[1])))
    return ranked, cheapest, fallback_cost


def select_model(
    options: Sequence[ModelOption],
    profile: RequestProfile,
    *,
    current: Optional[str] = None,
    current_provider: Optional[str] = None,
    min_improvement: float = 0.15,
    settings: Optional["RoutingSettings"] = None,
    fallback_options: Optional[Sequence[ModelOption]] = None,
) -> RoutingDecision:
    """Pick the best eligible model, holding the current one unless clearly beaten.

    ``fallback_options`` is the catalog pool for the §7.7 tool fallback: when the request needs
    tools, the caller restricted routing to a preferred list, and that list holds no model that
    is known to call tools, the pool is widened to tool-capable catalog models instead of
    returning "no eligible model" — and the reason says so.
    """
    st = settings or RoutingSettings()
    threshold = st.stability_threshold if settings is not None else min_improvement
    if not options:
        return RoutingDecision(
            model=None, provider="", reason="no candidate models configured",
            score=0.0, profile=profile, candidates=0, skip_reason=SKIP_NO_CANDIDATES)
    pool, widened = widen_for_tools(options, profile, st, fallback_options)
    ranked, _cheapest, _fallback = rank_candidates(pool, profile, st)
    if not ranked:
        reason = "no eligible model for this request"
        if profile.needs_tools and st.preferred_models_only:
            reason += " (preferred list has no tool-capable model"
            reason += "; tool fallback found none in the catalog)" if st.allow_tool_fallback \
                else "; set allow_tool_fallback to widen)"
        return RoutingDecision(
            model=None, provider="", reason=reason,
            score=0.0, profile=profile, candidates=0, skip_reason=SKIP_NO_ELIGIBLE)

    best_score, best = ranked[0]
    # tool_calls_only: the configured default model is the assistant for the turn, and the
    # pool exists to serve capabilities it lacks — not to compete with it. An eligible
    # incumbent is never displaced: not for price, not for a better score. (Same semantics
    # as model_router.route_turn; that engine also handles the delegation case, where a
    # specialist executes one sub-task and the default model still answers.)
    if st.strategy == "tool_calls_only" and current:
        for rank_score, option in ranked:
            if option.name != current:
                continue
            if current_provider and option.provider and option.provider != current_provider:
                continue
            return RoutingDecision(
                model=current, provider=option.provider,
                reason=(f"tool_calls_only: default model {current_provider or 'current'}/{current} "
                        f"stays primary (best alternative {best.provider}/{best.name} not used — "
                        "the default model is replaced only when it cannot serve the turn)"),
                score=rank_score, profile=profile, candidates=len(ranked),
                skip_reason=SKIP_ALREADY_CURRENT)
        # The default model is not in the routing pool. It still stays primary: under this
        # strategy the pool is the delegate list, never a rival for the whole turn.
        return RoutingDecision(
            model=current, provider=current_provider or "",
            reason=(f"tool_calls_only: default model {current_provider or ''}/{current} stays primary "
                    f"(not in the routing pool; best pool alternative {best.provider}/{best.name} "
                    "is reserved for capability delegation)"),
            score=0.0, profile=profile, candidates=len(ranked),
            skip_reason=SKIP_ALREADY_CURRENT)
    if current:
        for rank_score, option in ranked:
            if option.name != current:
                continue
            # Incumbent identity is (provider, model). One model id can be offered by two
            # providers, and a name-only match both reports a provider the session is not on
            # and hides a cross-provider move to an identically named model. Name-only is the
            # fallback for callers that cannot tell us the provider.
            if current_provider and option.provider and option.provider != current_provider:
                continue
            if option is best:
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
    if widened:
        # Prose, not a skip reason: the decision is a real switch, so ``skip_reason`` stays "".
        reason += ("; tool fallback: preferred list had no tool-capable model, "
                   f"widened to {len(pool)} catalog candidates")
    return RoutingDecision(
        model=best.name, provider=best.provider, reason=reason,
        score=best_score, profile=profile, candidates=len(ranked))


def routing_config(agent: Any) -> Dict[str, Any]:
    """Read ``smart_model_routing`` for this turn; never raises.

    The agent's own config dict wins when it has one (tests and embedded callers
    build agents that way). A real AIAgent has NO ``config`` attribute, so the old
    body returned ``{}`` on every production turn and the whole feature was inert
    no matter what the user configured - the test fake sets ``self.config``, which
    is exactly why the suite never caught it.

    Falling back to the on-disk user config also means a CLI change takes effect on
    the next turn without restarting the gateway: the loader caches on the file
    signature, so an unchanged file costs one stat.

    Trigger is "attribute absent / not a dict", never "empty dict": a caller that
    explicitly passes ``{}`` means "no routing" and must keep that meaning.
    """
    try:
        cfg = getattr(agent, "config", None)
        if isinstance(cfg, dict):
            raw = cfg.get("smart_model_routing")
            return raw if isinstance(raw, dict) else {}
        from hermes_cli.config import load_config_readonly

        live = load_config_readonly()
        raw = live.get("smart_model_routing") if isinstance(live, dict) else None
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
    # Spec §7.7, opt-in: when a tools-requiring request has no tool-capable model in the
    # preferred list, widen to the catalog instead of returning no candidates. False by
    # default so an existing config routes exactly as before.
    allow_tool_fallback: bool = False
    models: Tuple[Any, ...] = ()

    @property
    def cost_weight(self) -> float:
        return {"cost": 0.75, "quality": 0.0}.get(self.strategy, 0.45)


def parse_settings(raw: Any) -> RoutingSettings:
    """Normalise any historical spelling into :class:`RoutingSettings`. Never raises."""
    if not isinstance(raw, dict):
        return RoutingSettings()
    strategy = str(raw.get("strategy") or "automatic").strip().lower()
    if strategy not in ROUTING_STRATEGIES:
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
        allow_tool_fallback=bool(raw.get("allow_tool_fallback", False)),
        models=tuple(models) if isinstance(models, (list, tuple)) else (),
    )


ROUTING_STRATEGIES = ("automatic", "cost", "quality", "tool_calls_only")
_ROUTING_BOOL_KEYS = ("enabled", "preferred_models_only", "allow_unknown_metadata",
                      "cost_optimization", "allow_tool_fallback")


def validate_routing_config(raw: Any) -> List[str]:
    """Human-readable problems with a ``smart_model_routing`` block; empty means valid.

    Never raises: this runs on user input and on config read back from disk.
    """
    problems: List[str] = []
    if raw is None:
        return problems
    if not isinstance(raw, dict):
        return ["smart_model_routing must be a mapping"]
    if "strategy" in raw:
        strategy = str(raw.get("strategy") or "").strip().lower()
        if strategy not in ROUTING_STRATEGIES:
            problems.append(
                f"strategy must be one of {', '.join(ROUTING_STRATEGIES)} "
                f"(got {raw.get('strategy')!r})")
    if "stability_threshold" in raw:
        try:
            threshold = float(raw["stability_threshold"])
        except (TypeError, ValueError):
            problems.append("stability_threshold must be a number between 0 and 1")
        else:
            if not 0.0 <= threshold <= 1.0:
                problems.append(
                    f"stability_threshold must be between 0 and 1 (got {threshold})")
    if "models" in raw and not isinstance(raw["models"], (list, tuple)):
        problems.append("models must be a list of model names")
    for key in _ROUTING_BOOL_KEYS:
        if key in raw and not isinstance(raw[key], bool):
            problems.append(f"{key} must be true or false")
    return problems


def merge_routing_config(raw: Any, **changes: Any) -> Dict[str, Any]:
    """Return a new ``smart_model_routing`` dict with *changes* applied.

    Keys the caller does not mention are preserved untouched, including any this engine does
    not know about, so a future/foreign key is never dropped by a write from an interface.
    ``None`` means "leave as is".
    """
    merged = dict(raw) if isinstance(raw, dict) else {}
    for key, value in changes.items():
        if value is None:
            continue
        merged[key] = list(value) if isinstance(value, tuple) else value
    return merged


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


def _optional_price(info: Any, attr: str) -> Optional[float]:
    """A per-1M-token price the catalog publishes, or ``None`` when it is silent.

    Zero is treated as unpublished, matching ``has_cost``: a model that publishes no price is
    charged the median, so reading ``0.0`` as "free" here would leak a free model into ranking.
    """
    if info is None:
        return None
    try:
        price = float(getattr(info, attr, None))
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def options_from_config(cfg: Dict[str, Any], default_provider: str = "") -> List[ModelOption]:
    """Build candidates from ``models: [{name, provider, ...}]`` or ``models: [name, ...]``.

    Both shapes are valid: ``validate_routing_config`` describes the field as "a list of
    model names", the CLI and ``--models`` write plain strings, and ``models[]`` entries as
    dicts exist for per-model overrides. Reading only dicts made every saved selection
    invisible to the engine -- the pool came back empty, selection fell through to the whole
    provider catalog, and the interactive summary reported "87 eligible" seconds after the
    user had picked three models. A name with no provider inherits *default_provider*.

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
        if isinstance(entry, str):
            text = entry.strip()
            if not text:
                continue
            provider_part, sep, model_part = text.partition("/")
            entry = {"name": model_part if sep else text,
                     "provider": provider_part if sep else ""}
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        provider = str(entry.get("provider") or "").strip() or default_provider.strip()
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
                supports_image_output=flag("supports_image_output", None),
                supports_audio_output=flag("supports_audio_output", None),
                cost_cache_read=float(entry["cost_cache_read"]) if stated("cost_cache_read")
                else _optional_price(info, "cost_cache_read"),
                cost_cache_write=float(entry["cost_cache_write"]) if stated("cost_cache_write")
                else _optional_price(info, "cost_cache_write"),
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
            supports_image_output=((("image" in outputs) if outputs else None)),
            cost_cache_read=_optional_price(info, "cost_cache_read"),
            cost_cache_write=_optional_price(info, "cost_cache_write"),
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


def maybe_apply_model_routing(agent: Any, text: Any, *,
                              needs_tools: bool = False) -> Optional[RoutingDecision]:
    """Apply routing for one turn when enabled. Returns the decision, or None if inert.

    ``text`` is normally a string but may be a multi-part content list (vision via the
    OpenAI-compatible server), so it is passed through unmodified rather than coerced.
    ``needs_tools`` says the turn will require a tool call; it is what makes the §7.7 tool
    fallback reachable from the real path instead of only from direct ``select_model`` calls.

    Switches through the agent's existing ``switch_model`` so credential pools, prompt-cache
    flags and the compressor are rebuilt exactly as a manual switch would rebuild them. Any
    failure is logged and swallowed: routing must never break a turn.
    """
    if moa_active(agent):
        # MoA is a runtime the user chose explicitly; scoring a replacement model would
        # silently dismantle the fan-out. Both implementations skip routing here.
        logger.debug("model routing skipped: mixture-of-agents is active")
        # Record the reason on the agent so an in-session surface (status, /model-routing) can
        # report WHY nothing switched. Without this the skip was invisible: a debug line is not
        # a record, and a status view would have blamed "disabled" or shown a hypothetical
        # selection that would never happen.
        try:
            agent._routing_skip_reason = SKIP_MOA_ACTIVE
        except Exception:  # pragma: no cover - defensive
            pass
        return None
    cfg = routing_config(agent)
    st = parse_settings(cfg)
    if not st.enabled:
        return None
    try:
        options = options_from_config(
            cfg, default_provider=str(getattr(agent, "provider", "") or ""))
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
        profile = classify_request(text or "", needs_tools=needs_tools)
        # §7.7: the catalog is only consulted when the opt-in can actually matter — a
        # preferred-only pool, a tool requirement, and allow_tool_fallback. Cache-only, so a
        # turn never waits on a network fetch.
        fallback_options = None
        if st.preferred_models_only and st.allow_tool_fallback and profile.needs_tools:
            fallback_options = options_from_catalog(str(getattr(agent, "provider", "") or ""))
        current_provider = getattr(agent, "provider", None)
        # Never select a free model that already refused this window: the provider said no
        # until the reset, so choosing it again would spend a turn to learn that again.
        try:
            from agent import free_model_guard
            options, _dropped = free_model_guard.filter_exhausted(options, cfg)
            if fallback_options:
                fallback_options, _ = free_model_guard.filter_exhausted(fallback_options, cfg)
        except Exception:
            logger.debug("model routing: free-model guard unavailable", exc_info=True)
        decision = select_model(
            options, profile,
            current=getattr(agent, "model", None),
            current_provider=current_provider, settings=st,
            fallback_options=fallback_options)
        # Same identity rule as the selector: switching to an identically named model on a
        # different provider is a real switch, so the name alone is not "unchanged".
        same_identity = (
            decision.model == getattr(agent, "model", None)
            and (not current_provider or not decision.provider
                 or decision.provider == current_provider))
        if not decision.model or same_identity:
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

def _configured_model_names(config: Any) -> List[str]:
    """Names from ``models`` in priority order, whichever shape the entries use."""
    raw = config.get("models") if isinstance(config, dict) else None
    if not isinstance(raw, (list, tuple)):
        return []
    names: List[str] = []
    for entry in raw:
        name = entry.get("name") if isinstance(entry, dict) else entry
        name = str(name or "").strip()
        if name:
            names.append(name)
    return names


def _free_guard_report(config: Any) -> Dict[str, Any]:
    try:
        from agent import free_model_guard
        return free_model_guard.status(free_model_guard.guard_config(config))
    except Exception:
        return {}


def explain_routing(
    config: Dict[str, Any],
    text: Any,
    *,
    current: Optional[str] = None,
    provider: str = "",
    options: Optional[Sequence[ModelOption]] = None,
    has_images: bool = False,
    has_audio: bool = False,
    has_files: bool = False,
    needs_tools: bool = False,
    moa_active: bool = False,
) -> Dict[str, Any]:
    """Explain a routing decision without performing one.

    Every number here comes from the same functions selection uses, so the report cannot
    disagree with the behaviour it describes.
    """
    st = parse_settings(config)
    prof = classify_request(text or "", has_images=has_images, has_audio=has_audio,
                            has_files=has_files, needs_tools=needs_tools)
    cands = (list(options) if options is not None
             else options_from_config(config, default_provider=provider))
    source = "config"
    if not cands:
        cands = options_from_catalog(provider)
        source = "catalog"
    # A free model that already hit its limit this window must never be offered again: the
    # provider has said no until the reset, so re-selecting it burns a turn. Applied after the
    # catalog fallback too, otherwise an exhausted pool "falls back" straight onto the free
    # model it just excluded.
    try:
        from agent import free_model_guard
        cands, exhausted_skipped = free_model_guard.filter_exhausted(cands, config)
    except Exception:
        exhausted_skipped = []
    # §7.7: widen exactly as selection does, so the report cannot describe a pool the engine
    # would not have used. An explicit ``options`` list is widened against the same provider
    # catalog, because selection would do so too when the opt-in is on.
    fallback_pool = None
    if st.preferred_models_only and st.allow_tool_fallback and prof.needs_tools:
        fallback_pool = options_from_catalog(provider)
    cands, widened = widen_for_tools(cands, prof, st, fallback_pool)
    if widened:
        source = f"{source}+catalog (tool fallback)"
    ranked, cheapest, fallback_cost = rank_candidates(cands, prof, st)
    decision = select_model(cands, prof, current=current, current_provider=provider, settings=st)
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
            # "" when eligible, so the field is always a string (contract: one row per
            # candidate, excluded_reason empty when eligible).
            "excluded_reason": "" if eligible else _exclusion_reason(opt, prof, st),
        })
    rows.sort(key=lambda r: (not r["eligible"],
                             -(r["score"] if r["score"] is not None else -99),
                             f"{r['provider']}/{r['model']}"))
    return {
        "enabled": st.enabled,
        "settings": {
            "strategy": st.strategy, "stability_threshold": st.stability_threshold,
            "allow_unknown_metadata": st.allow_unknown_metadata,
            "cost_optimization": st.cost_optimization,
            "preferred_models_only": st.preferred_models_only,
            "allow_tool_fallback": st.allow_tool_fallback,
        },
        "candidate_source": source,
        # True when §7.7 widened the pool past a preferred list that held no tool-capable model.
        "widened_for_tools": widened,
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
        # MoA wins over every other reason when enabled: it is the cause the runtime will act
        # on, and reporting "already_current" or a hypothetical selection beside it would be a
        # statement about a decision that cannot happen this turn.
        "skip_reason": (
            SKIP_MOA_ACTIVE if (st.enabled and moa_active)
            else (decision.skip_reason if st.enabled else SKIP_DISABLED)
        ),
        # Whether the Mixture-of-Agents facade is what suppresses routing. Kept as its own row
        # so a caller can render "MoA active" without parsing skip_reason.
        "moa_active": bool(moa_active),
        "current": current,
        "kept_current": bool(
            decision.model and decision.model == current
            and (not provider or not decision.provider or decision.provider == provider)),
        "eligible_count": len(ranked),
        "candidates": rows,
        # What the operator actually configured, in priority order. The CLI needs this to
        # show the pool it is routing within: "candidates: 4" alone never told anyone whether
        # their selection had been applied.
        "configured_models": _configured_model_names(config),
        "exhausted_skipped": list(exhausted_skipped),
        "free_guard": _free_guard_report(config),
    }


def _exclusion_reason(option: ModelOption, profile: RequestProfile,
                      st: "RoutingSettings") -> str:
    if option.deprecated:
        return "deprecated"
    if option.supports_text_output is False:
        return "cannot emit text"
    if profile.needs_image_output and option.supports_image_output is False:
        return "unsupported_output_image"
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
            ("image_output", option.supports_image_output, profile.needs_image_output),
        )
        for label, value, required in needs:
            if required and value is None:
                return f"{label} capability unknown (strict mode)"
    # Callers only ask after ``is_eligible`` already said no, so this is "no specific
    # reason matched", never "eligible". Wording must not read as the verdict itself.
    return "unclassified"
