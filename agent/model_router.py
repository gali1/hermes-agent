"""Deterministic, capability-aware model selection ("smart model routing").

This module is the *selection* half of dynamic model routing.  It answers one
question: given a request profile and a set of candidate models, which model
should serve it, and why?  It never executes a request, never touches tool
calling, and never rewrites conversation state.

Design constraints (see task spec TASK_1, parts III/V/IX/X):

* **Deterministic.** No LLM call, no network call at selection time.  Model
  metadata comes from :mod:`agent.models_dev`, which is cached on disk and in
  memory; token counts come from :func:`agent.model_metadata.estimate_tokens_rough`
  so there is exactly one token estimator in the tree.
* **Eligibility before ranking.** A model that cannot satisfy a hard
  requirement (context, modality, tools, deprecation) is never ranked.
* **Unknown is not unsupported.** models.dev omits fields for many entries
  (``tool_call`` defaults to ``False`` when absent).  Missing metadata is
  treated as UNKNOWN and merely penalised, per spec §9.2.
* **Stable.** A marginally better score does not replace the current model;
  a switch needs ``stability_threshold`` of improvement or a hard constraint.
* **Off by default.** ``smart_model_routing.enabled`` is ``False``.  Callers
  must consult :func:`is_enabled` before doing anything else, so an untouched
  install behaves exactly as it did before this module existed.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CapabilityState",
    "Complexity",
    "Decision",
    "ModelCandidate",
    "RoutingConfig",
    "TaskProfile",
    "classify_request",
    "configured_models",
    "estimate_cost",
    "explain",
    "filter_eligible",
    "is_enabled",
    "route_turn",
    "routing_pool",
    "score_candidate",
    "select_model",
    "widen_pool",
]


# ---------------------------------------------------------------------------
# Enums / small types
# ---------------------------------------------------------------------------

class CapabilityState(str, Enum):
    """Tri-state capability answer (spec §9.2)."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class Complexity(int, Enum):
    """Task complexity ladder (spec §7)."""

    SIMPLE = 0
    ROUTINE = 1
    MODERATE = 2
    COMPLEX = 3
    ADVANCED = 4

    @property
    def label(self) -> str:
        return self.name.lower()


#: Input modalities a profile can require.  Text is always implied.
_INPUT_MODALITIES = ("text", "image", "audio", "video", "pdf")

# Deterministic lexical signals.  These are *hints*, not the sole input: token
# volume, structure (code fences, numbered lists) and explicit modalities also
# contribute.  Order matters only for readability.
_CODE_SIGNALS = re.compile(
    r"```|\bdef\s+\w+\(|\bclass\s+\w+|Traceback \(most recent call last\)|"
    r"\b(?:refactor|debug|stack ?trace|segfault|regex|sql query|api endpoint)\b",
    re.IGNORECASE,
)
_REASONING_SIGNALS = re.compile(
    r"\b(?:prove|derive|root cause|trade-?offs?|architecture|design a|optimi[sz]e|"
    r"benchmark|algorithm|complexity|why does|analyse|analyze|evaluate|compare|"
    r"step by step|multi-?step|migration plan|post-?mortem)\b",
    re.IGNORECASE,
)
_SIMPLE_SIGNALS = re.compile(
    r"^\s*(?:what|who|when|where|which)\s+(?:is|are|was|were)\b|"
    r"^\s*(?:convert|translate|capital of|define)\b",
    re.IGNORECASE,
)
_STRUCTURED_SIGNALS = re.compile(
    r"\b(?:json|yaml|xml|csv|schema|structured output|return (?:a|an) (?:list|object|array))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Task profile + classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskProfile:
    """What the request needs from a model."""

    complexity: Complexity = Complexity.ROUTINE
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    needs_tools: bool = False
    needs_reasoning: bool = False
    needs_structured_output: bool = False
    input_modalities: Tuple[str, ...] = ("text",)
    output_modalities: Tuple[str, ...] = ("text",)
    #: Context the request must fit into (conversation + prompt + output).
    required_context: int = 0
    #: User pinned a model for this request; selection must not override it.
    pinned_model: str = ""
    #: Which signals fired, for the human-readable explanation.
    signals: Tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.est_input_tokens + self.est_output_tokens


def _estimate_tokens(text: str) -> int:
    """Delegate to the single existing estimator (no second accounting system)."""
    if not text:
        return 0
    try:
        from agent.model_metadata import estimate_tokens_rough

        return int(estimate_tokens_rough(text))
    except Exception:  # pragma: no cover - defensive: estimator unavailable
        return max(1, len(text) // 4)


def classify_request(
    prompt: str,
    *,
    attachments: Sequence[str] = (),
    expects_image: bool = False,
    expects_audio: bool = False,
    expects_video: bool = False,
    conversation_tokens: int = 0,
    tool_available: bool = True,
    pinned_model: str = "",
    expected_output_tokens: Optional[int] = None,
) -> TaskProfile:
    """Classify a request deterministically.

    ``attachments`` are modality hints such as ``"image"``, ``"pdf"``,
    ``"audio"``, ``"video"`` (or file paths, whose suffix is used as a hint).
    ``conversation_tokens`` is the already-known size of the running context;
    it is added to the estimate so a long session cannot be routed into a
    model whose window it no longer fits.
    """
    text = prompt or ""
    signals: List[str] = []

    in_tokens = _estimate_tokens(text) + max(0, int(conversation_tokens))

    # ---- complexity ladder ------------------------------------------------
    # Base tier from volume: short prompts start simple, long ones moderate.
    if in_tokens <= 60:
        level = Complexity.SIMPLE
        signals.append("short-prompt")
    elif in_tokens <= 400:
        level = Complexity.ROUTINE
        signals.append("medium-prompt")
    elif in_tokens <= 2000:
        level = Complexity.MODERATE
        signals.append("long-prompt")
    else:
        level = Complexity.COMPLEX
        signals.append("very-long-prompt")

    code_hit = bool(_CODE_SIGNALS.search(text))
    reasoning_hit = bool(_REASONING_SIGNALS.search(text))
    if code_hit:
        level = Complexity(min(int(level) + 1, int(Complexity.ADVANCED)))
        signals.append("code-signal")
    if reasoning_hit:
        level = Complexity(min(int(level) + 1, int(Complexity.ADVANCED)))
        signals.append("reasoning-signal")
    # Code *and* reasoning together is advanced technical work regardless of
    # how few tokens it took to ask (spec §7: a short request can be advanced).
    if code_hit and reasoning_hit:
        level = Complexity.ADVANCED
        signals.append("technical-work")
    # Multi-part requests (several questions / a numbered plan) are more than
    # their token count suggests.
    if text.count("?") >= 3 or len(re.findall(r"(?m)^\s*\d+[.)]\s", text)) >= 4:
        level = Complexity(min(int(level) + 1, int(Complexity.ADVANCED)))
        signals.append("multi-part")

    # A short, plainly factual question must not be over-classified just
    # because it contains one keyword (spec §7: "do not overclassify").
    if (
        in_tokens <= 40
        and _SIMPLE_SIGNALS.search(text)
        and not _CODE_SIGNALS.search(text)
        and not _REASONING_SIGNALS.search(text)
    ):
        level = Complexity.SIMPLE
        signals.append("factual-question")

    # ---- modalities -------------------------------------------------------
    in_mods: List[str] = ["text"]
    for hint in attachments:
        h = str(hint).strip().lower()
        if not h:
            continue
        if h in _INPUT_MODALITIES and h not in in_mods:
            in_mods.append(h)
            signals.append(f"input:{h}")
        elif h.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
            if "image" not in in_mods:
                in_mods.append("image")
                signals.append("input:image")
        elif h.endswith(".pdf"):
            if "pdf" not in in_mods:
                in_mods.append("pdf")
                signals.append("input:pdf")
        elif h.endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
            if "audio" not in in_mods:
                in_mods.append("audio")
                signals.append("input:audio")
        elif h.endswith((".mp4", ".mov", ".mkv", ".webm")):
            if "video" not in in_mods:
                in_mods.append("video")
                signals.append("input:video")

    out_mods: List[str] = ["text"]
    if expects_image:
        out_mods.append("image")
        signals.append("output:image")
    if expects_audio:
        out_mods.append("audio")
        signals.append("output:audio")
    if expects_video:
        out_mods.append("video")
        signals.append("output:video")

    needs_structured = bool(_STRUCTURED_SIGNALS.search(text))
    if needs_structured:
        signals.append("structured-output")

    # Modality work is not text-reasoning work; a vision question should not be
    # pushed into an expensive reasoner merely for carrying an image.
    needs_reasoning = level >= Complexity.COMPLEX and bool(_REASONING_SIGNALS.search(text))
    if needs_reasoning:
        signals.append("reasoning-required")

    if expected_output_tokens is None:
        # Cheap deterministic default: short answers for simple asks, room for
        # code/analysis when the task is larger.
        out_tokens = {Complexity.SIMPLE: 300, Complexity.ROUTINE: 800,
                      Complexity.MODERATE: 1500, Complexity.COMPLEX: 3000,
                      Complexity.ADVANCED: 4000}[level]
    else:
        out_tokens = max(1, int(expected_output_tokens))

    return TaskProfile(
        complexity=level,
        est_input_tokens=in_tokens,
        est_output_tokens=out_tokens,
        needs_tools=bool(tool_available) and _needs_tools(text, level),
        needs_reasoning=needs_reasoning,
        needs_structured_output=needs_structured,
        input_modalities=tuple(in_mods),
        output_modalities=tuple(out_mods),
        required_context=in_tokens + out_tokens,
        pinned_model=(pinned_model or "").strip(),
        signals=tuple(signals),
    )


def _needs_tools(text: str, level: Complexity) -> bool:
    """Whether the request plausibly requires tool execution.

    Deliberately conservative: only explicit action verbs that cannot be
    answered from the model's own weights.  This decides *which* model may be
    used, never *whether* tools are available (that is the toolset's job).
    """
    if re.search(
        r"\b(?:run|execute|install|deploy|check|fetch|download|upload|open|read the file|"
        r"search (?:the )?web|curl|ssh|docker|systemctl|git (?:status|log|diff))\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return level >= Complexity.COMPLEX and bool(_CODE_SIGNALS.search(text))


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

@dataclass
class ModelCandidate:
    """A model that *could* serve a request, plus what we know about it.

    ``info`` may be ``None`` when models.dev has no entry (a local/custom
    model, or an offline catalog miss).  ``tool_support`` lets a caller pass a
    runtime-verified answer that overrides catalog metadata.
    """

    provider: str
    model: str
    info: Any = None  # agent.models_dev.ModelInfo
    tool_support: Optional[CapabilityState] = None
    #: False when the provider is disabled/without credentials.
    enabled: bool = True
    #: Optional measured latency in ms (used only as a tie-breaker).
    latency_ms: Optional[float] = None
    #: Optional historical success rate 0..1; None means unknown.
    reliability: Optional[float] = None

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    # -- derived metadata ---------------------------------------------------
    def _mods(self, attr: str) -> Tuple[str, ...]:
        mods = getattr(self.info, attr, ()) if self.info is not None else ()
        return tuple(mods or ())

    def supports_input(self, modality: str) -> CapabilityState:
        if modality == "text":
            return CapabilityState.SUPPORTED
        if self.info is None:
            return CapabilityState.UNKNOWN
        mods = self._mods("input_modalities")
        if not mods:
            return CapabilityState.UNKNOWN
        if modality in mods:
            return CapabilityState.SUPPORTED
        # models.dev lists "image" for vision; "attachment" is the older flag.
        if modality == "image" and getattr(self.info, "attachment", False):
            return CapabilityState.SUPPORTED
        return CapabilityState.UNSUPPORTED

    def supports_output(self, modality: str) -> CapabilityState:
        if modality == "text":
            return CapabilityState.SUPPORTED
        if self.info is None:
            return CapabilityState.UNKNOWN
        mods = self._mods("output_modalities")
        if not mods:
            return CapabilityState.UNKNOWN
        return CapabilityState.SUPPORTED if modality in mods else CapabilityState.UNSUPPORTED

    def supports_tools(self) -> CapabilityState:
        if self.tool_support is not None:
            return self.tool_support
        if self.info is None:
            return CapabilityState.UNKNOWN
        # models.dev defaults the flag to False when the field is absent, so a
        # False here means "not confirmed", never "confirmed unsupported".
        return CapabilityState.SUPPORTED if getattr(self.info, "tool_call", False) else CapabilityState.UNKNOWN

    def supports_reasoning(self) -> CapabilityState:
        if self.info is None:
            return CapabilityState.UNKNOWN
        return CapabilityState.SUPPORTED if getattr(self.info, "reasoning", False) else CapabilityState.UNKNOWN

    def context_window(self) -> Optional[int]:
        ctx = getattr(self.info, "context_window", 0) if self.info is not None else 0
        return int(ctx) if ctx and ctx > 0 else None

    def max_output(self) -> Optional[int]:
        out = getattr(self.info, "max_output", 0) if self.info is not None else 0
        return int(out) if out and out > 0 else None

    def is_deprecated(self) -> bool:
        return str(getattr(self.info, "status", "") or "").lower() == "deprecated"

    def costs(self) -> Tuple[Optional[float], Optional[float]]:
        if self.info is None:
            return None, None
        ci = getattr(self.info, "cost_input", 0) or 0
        co = getattr(self.info, "cost_output", 0) or 0
        if ci <= 0 and co <= 0:
            return None, None
        return float(ci), float(co)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "capability": 0.45,
    "cost": 0.30,
    "reliability": 0.10,
    "context": 0.10,
    "continuity": 0.05,
}


@dataclass
class RoutingConfig:
    """Resolved ``smart_model_routing`` settings with safe defaults."""

    enabled: bool = False
    strategy: str = "automatic"  # automatic | cost | quality
    preferred_models_only: bool = True
    cost_optimization: bool = True
    stability_threshold: float = 0.08
    #: Allow a mid-session switch to a *cheaper* model.  Off by default: a
    #: model change invalidates the provider's prompt cache, so a downgrade
    #: must be an explicit choice, not an automatic one.
    switch_on_cost: bool = False
    allow_unknown_metadata: bool = True
    llm_assisted_classification: bool = False
    weights: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    #: model key ("provider/model") or bare model -> 0..1 success rate.
    reliability: Dict[str, float] = field(default_factory=dict)
    #: models the operator never wants auto-selected (still usable manually).
    exclude: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any) -> "RoutingConfig":
        cfg = cls()
        if not isinstance(raw, dict):
            return cfg
        cfg.enabled = bool(raw.get("enabled", cfg.enabled))
        strategy = str(raw.get("strategy", cfg.strategy) or cfg.strategy).lower()
        cfg.strategy = strategy if strategy in ("automatic", "cost", "quality") else "automatic"
        cfg.preferred_models_only = bool(raw.get("preferred_models_only", cfg.preferred_models_only))
        cfg.cost_optimization = bool(raw.get("cost_optimization", cfg.cost_optimization))
        try:
            cfg.stability_threshold = max(0.0, float(raw.get("stability_threshold", cfg.stability_threshold)))
        except (TypeError, ValueError):
            pass
        cfg.allow_unknown_metadata = bool(raw.get("allow_unknown_metadata", cfg.allow_unknown_metadata))
        cfg.switch_on_cost = bool(raw.get("switch_on_cost", cfg.switch_on_cost))
        cfg.llm_assisted_classification = bool(raw.get("llm_assisted_classification", cfg.llm_assisted_classification))
        weights = raw.get("weights")
        if isinstance(weights, dict):
            for name, value in weights.items():
                if name in cfg.weights:
                    try:
                        cfg.weights[name] = max(0.0, float(value))
                    except (TypeError, ValueError):
                        pass
            total = sum(cfg.weights.values())
            if total > 0:  # normalise so the score stays comparable
                cfg.weights = {k: v / total for k, v in cfg.weights.items()}
        rel = raw.get("reliability")
        if isinstance(rel, dict):
            for key, value in rel.items():
                try:
                    cfg.reliability[str(key)] = min(1.0, max(0.0, float(value)))
                except (TypeError, ValueError):
                    continue
        excl = raw.get("exclude")
        if isinstance(excl, (list, tuple)):
            cfg.exclude = tuple(str(x) for x in excl if str(x).strip())
        return cfg


def is_enabled(config: Any) -> bool:
    """True when dynamic selection is switched on.

    Accepts a whole config dict or an already-built :class:`RoutingConfig`.
    Cheap enough to call on every request; the point is that a disabled
    install pays nothing.
    """
    if isinstance(config, RoutingConfig):
        return config.enabled
    if isinstance(config, dict):
        section = config.get("smart_model_routing")
        return bool(isinstance(section, dict) and section.get("enabled"))
    return False


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def _excluded(candidate: ModelCandidate, cfg: RoutingConfig) -> bool:
    if not candidate.enabled:
        return True
    if candidate.key in cfg.exclude or candidate.model in cfg.exclude:
        return True
    return candidate.is_deprecated()


def filter_eligible(
    candidates: Iterable[ModelCandidate],
    profile: TaskProfile,
    config: Optional[RoutingConfig] = None,
) -> List[ModelCandidate]:
    """Apply hard constraints; only survivors may be ranked (spec §15)."""
    cfg = config or RoutingConfig()
    out: List[ModelCandidate] = []
    for cand in candidates:
        if _excluded(cand, cfg):
            continue

        # A pinned model wins over every other constraint except existence:
        # the user's explicit choice is authoritative (spec §10).
        if profile.pinned_model and cand.model == profile.pinned_model:
            out.append(cand)
            continue

        # Context window: unknown is allowed (penalised later), too small is not.
        ctx = cand.context_window()
        if ctx is not None and profile.required_context > ctx:
            continue

        # Output cap: a model that cannot emit the expected answer will truncate.
        cap = cand.max_output()
        if cap is not None and profile.est_output_tokens > cap:
            continue

        # Input modalities.
        ok = True
        for mod in profile.input_modalities:
            state = cand.supports_input(mod)
            if state is CapabilityState.UNSUPPORTED:
                ok = False
                break
            if state is CapabilityState.UNKNOWN and not cfg.allow_unknown_metadata:
                ok = False
                break
        if not ok:
            continue

        # Output modalities.
        for mod in profile.output_modalities:
            state = cand.supports_output(mod)
            if state is CapabilityState.UNSUPPORTED:
                ok = False
                break
            if state is CapabilityState.UNKNOWN and not cfg.allow_unknown_metadata:
                ok = False
                break
        if not ok:
            continue

        # Tools: only a *confirmed* lack disqualifies.
        if profile.needs_tools and cand.supports_tools() is CapabilityState.UNSUPPORTED:
            continue

        if profile.needs_structured_output and cand.info is not None:
            if getattr(cand.info, "structured_output", False) is False and getattr(
                cand.info, "input_modalities", ()
            ) and not cfg.allow_unknown_metadata:
                # Only a hard exclusion in strict mode; otherwise penalised.
                continue

        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Cost + scoring
# ---------------------------------------------------------------------------

def estimate_cost(candidate: ModelCandidate, profile: TaskProfile) -> Optional[float]:
    """Expected USD cost for this request; ``None`` when pricing is unknown."""
    ci, co = candidate.costs()
    if ci is None and co is None:
        return None
    ci = ci or 0.0
    co = co or 0.0
    return (profile.est_input_tokens / 1_000_000.0) * ci + (
        profile.est_output_tokens / 1_000_000.0
    ) * co


def _reliability_of(candidate: ModelCandidate, cfg: RoutingConfig) -> Optional[float]:
    if candidate.reliability is not None:
        return candidate.reliability
    for key in (candidate.key, candidate.model):
        if key in cfg.reliability:
            return cfg.reliability[key]
    return None


def _capability_fit(candidate: ModelCandidate, profile: TaskProfile) -> float:
    """0..1 — can this model do the job, and is it the right *size* of model?"""
    fit = 1.0

    # Reasoning: needed for complex work, wasted (not harmful) on simple work.
    if profile.needs_reasoning:
        state = candidate.supports_reasoning()
        if state is CapabilityState.SUPPORTED:
            fit += 0.10
        elif state is CapabilityState.UNKNOWN:
            fit -= 0.15
    else:
        # Prefer non-reasoning models for simple asks: reasoning tokens are
        # billed as output and add latency with no quality benefit.
        if candidate.supports_reasoning() is CapabilityState.SUPPORTED and profile.complexity <= Complexity.ROUTINE:
            fit -= 0.05

    if profile.needs_tools:
        state = candidate.supports_tools()
        if state is CapabilityState.UNKNOWN:
            fit -= 0.20

    if profile.needs_structured_output and candidate.info is not None:
        if not getattr(candidate.info, "structured_output", False):
            fit -= 0.10

    # Modality breadth beyond what was asked for is not a virtue; modality
    # gaps that survived eligibility are UNKNOWN and already penalised.
    return max(0.0, min(1.25, fit))


def _context_fit(candidate: ModelCandidate, profile: TaskProfile) -> float:
    ctx = candidate.context_window()
    if ctx is None:
        return 0.5  # unknown capacity: neutral, never rewarded
    if profile.required_context <= 0:
        return 1.0
    headroom = ctx / max(1, profile.required_context)
    # Enough room is enough; a 1M-token window buys nothing for a 2K request.
    return min(1.0, 0.6 + 0.4 * min(1.0, headroom / 4.0))


def score_candidate(
    candidate: ModelCandidate,
    profile: TaskProfile,
    config: Optional[RoutingConfig] = None,
    *,
    current_model: str = "",
    cost_ceiling: Optional[float] = None,
) -> float:
    """Weighted score in roughly 0..1.2 (higher is better).

    Mandatory constraints are applied by :func:`filter_eligible`; this is the
    ranking step only.  Cost is expressed relative to ``cost_ceiling`` (the
    most expensive eligible candidate) so the term is stable across requests.
    """
    cfg = config or RoutingConfig()
    w = cfg.weights

    capability = _capability_fit(candidate, profile)

    cost = estimate_cost(candidate, profile)
    if cost is None:
        cost_score = 0.35  # unknown pricing: conservative, never a free lunch
    elif not cfg.cost_optimization:
        cost_score = 1.0
    else:
        # With a single priced candidate there is nothing to compare against;
        # charging it a full penalty would make "unknown" look cheaper.
        ceiling = cost_ceiling if cost_ceiling and cost_ceiling > 0 else None
        cost_score = 1.0 - min(1.0, cost / ceiling) if ceiling else 1.0

    rel = _reliability_of(candidate, cfg)
    # Cold start: unknown reliability is 0.5, never 1.0 (spec §16.3).
    rel_score = 0.5 if rel is None else max(0.0, min(1.0, rel))

    context = _context_fit(candidate, profile)

    continuity = 1.0 if current_model and candidate.model == current_model else 0.0

    if cfg.strategy == "cost":
        w = {"capability": 0.35, "cost": 0.55, "reliability": 0.05, "context": 0.05, "continuity": 0.0}
    elif cfg.strategy == "quality":
        w = {"capability": 0.65, "cost": 0.05, "reliability": 0.15, "context": 0.10, "continuity": 0.05}

    return (
        w.get("capability", 0) * capability
        + w.get("cost", 0) * cost_score
        + w.get("reliability", 0) * rel_score
        + w.get("context", 0) * context
        + w.get("continuity", 0) * continuity
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """The outcome of one selection pass, with the reasoning kept for audits."""

    model: str
    provider: str
    score: float
    est_cost: Optional[float]
    reason: str
    switched: bool = False
    complexity: str = ""
    eligible_count: int = 0
    considered: List[Tuple[str, float]] = field(default_factory=list)
    pinned: bool = False

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}" if self.provider else self.model


def select_model(
    profile: TaskProfile,
    candidates: Sequence[ModelCandidate],
    config: Optional[RoutingConfig] = None,
    *,
    current_model: str = "",
) -> Optional[Decision]:
    """Pick the best eligible model, or ``None`` when nothing qualifies.

    ``current_model`` drives hysteresis: the incumbent is kept unless a rival
    beats it by ``stability_threshold`` (or the incumbent is ineligible).
    """
    cfg = config or RoutingConfig()

    if profile.pinned_model:
        pinned = [c for c in candidates if c.model == profile.pinned_model and c.enabled]
        if pinned:
            cand = pinned[0]
            return Decision(
                model=cand.model,
                provider=cand.provider,
                score=1.0,
                est_cost=estimate_cost(cand, profile),
                reason="user pinned model",
                complexity=profile.complexity.label,
                eligible_count=1,
                considered=[(cand.key, 1.0)],
                pinned=True,
            )
        # Pinned but unknown: say so rather than silently choosing another.
        return None

    eligible = filter_eligible(candidates, profile, cfg)
    if not eligible:
        return None

    costs = [c for c in (estimate_cost(x, profile) for x in eligible) if c is not None]
    ceiling = max(costs) if len(costs) > 1 else None

    scored = [
        (cand, score_candidate(cand, profile, cfg, current_model=current_model, cost_ceiling=ceiling))
        for cand in eligible
    ]
    scored.sort(key=lambda item: (-item[1], item[0].key))

    best, best_score = scored[0]
    considered = [(c.key, round(s, 4)) for c, s in scored[:5]]

    incumbent = None
    if current_model:
        incumbent = next((c for c in eligible if c.model == current_model), None)

    if incumbent is not None and incumbent is not best:
        inc_score = score_candidate(incumbent, profile, cfg, current_model=current_model, cost_ceiling=ceiling)
        gain = best_score - inc_score
        best_cost = estimate_cost(best, profile)
        inc_cost = estimate_cost(incumbent, profile)
        downgrade = (
            best_cost is not None and inc_cost is not None and best_cost < inc_cost
        )
        # A capability gap is not a marginal score difference: an incumbent
        # that cannot reason about an advanced task (or cannot call tools)
        # must not be kept just because the score delta is small (spec §16.2).
        capability_upgrade = (
            (
                profile.needs_reasoning
                and incumbent.supports_reasoning() is not CapabilityState.SUPPORTED
                and best.supports_reasoning() is CapabilityState.SUPPORTED
            )
            or (
                profile.needs_tools
                and incumbent.supports_tools() is CapabilityState.UNSUPPORTED
                and best.supports_tools() is CapabilityState.SUPPORTED
            )
        )
        if gain < cfg.stability_threshold and not capability_upgrade:
            return Decision(
                model=incumbent.model,
                provider=incumbent.provider,
                score=round(inc_score, 4),
                est_cost=inc_cost,
                reason=(
                    f"kept current model: best alternative {best.key} led by only "
                    f"{gain:.3f} < stability {cfg.stability_threshold:.3f}"
                ),
                switched=False,
                complexity=profile.complexity.label,
                eligible_count=len(eligible),
                considered=considered,
            )
        if downgrade and not cfg.switch_on_cost:
            # A cheaper model wins on score, but the incumbent is still fully
            # eligible.  Switching would invalidate the provider's prompt
            # cache mid-conversation, so it stays an explicit user choice.
            return Decision(
                model=incumbent.model,
                provider=incumbent.provider,
                score=round(inc_score, 4),
                est_cost=inc_cost,
                reason=(
                    f"kept current model: {best.key} is cheaper (score +{gain:.3f}) but a "
                    "cost-only downgrade mid-session is disabled (prompt-cache stability)"
                ),
                switched=False,
                complexity=profile.complexity.label,
                eligible_count=len(eligible),
                considered=considered,
            )

    if incumbent is not None and incumbent is best:
        reason = "current model remains the best eligible choice"
        switched = False
    else:
        reason = f"selected over {len(eligible) - 1} other eligible model(s) for {profile.complexity.label} task"
        switched = bool(current_model) and best.model != current_model

    return Decision(
        model=best.model,
        provider=best.provider,
        score=round(best_score, 4),
        est_cost=estimate_cost(best, profile),
        reason=reason,
        switched=switched,
        complexity=profile.complexity.label,
        eligible_count=len(eligible),
        considered=considered,
    )


# ---------------------------------------------------------------------------
# Human-readable explanation (used by CLI + logs)
# ---------------------------------------------------------------------------

def explain(decision: Optional[Decision], profile: Optional[TaskProfile] = None) -> str:
    """One-paragraph, copy-pasteable explanation of a selection."""
    if decision is None:
        base = "no eligible model: every candidate failed a hard constraint (context, modality, tools, or deprecation)"
        if profile and profile.pinned_model:
            base = f"pinned model {profile.pinned_model!r} is not an enabled candidate"
        if profile and profile.signals:
            base += f" [signals: {', '.join(profile.signals)}]"
        return base

    lines = [
        f"model: {decision.key}",
        f"complexity: {decision.complexity} | eligible: {decision.eligible_count} | score: {decision.score:.3f}",
        f"est. cost: {'unknown' if decision.est_cost is None else f'${decision.est_cost:.6f}'}",
        f"reason: {decision.reason}",
    ]
    if decision.considered:
        top = ", ".join(f"{k}={s:.3f}" for k, s in decision.considered)
        lines.append(f"top candidates: {top}")
    if profile is not None:
        lines.append(
            f"tokens: in≈{profile.est_input_tokens} out≈{profile.est_output_tokens} "
            f"| tools={profile.needs_tools} reasoning={profile.needs_reasoning} "
            f"| in={','.join(profile.input_modalities)} out={','.join(profile.output_modalities)}"
        )
        if profile.signals:
            lines.append(f"signals: {', '.join(profile.signals)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: build candidates from the existing catalog
# ---------------------------------------------------------------------------

def candidates_for(
    provider: str,
    models: Iterable[str],
    *,
    enabled: bool = True,
) -> List[ModelCandidate]:
    """Wrap model IDs in candidates using cached models.dev metadata.

    Never raises and never fetches per model beyond the catalog's own cache;
    an unknown model simply carries ``info=None`` (UNKNOWN metadata).
    """
    out: List[ModelCandidate] = []
    try:
        from agent import models_dev
    except Exception:  # pragma: no cover - defensive
        models_dev = None  # type: ignore[assignment]

    for model in models:
        model = str(model).strip()
        if not model:
            continue
        info = None
        if models_dev is not None:
            try:
                info = models_dev.get_model_info(provider, model)
            except Exception:  # pragma: no cover - catalog unavailable
                info = None
        out.append(ModelCandidate(provider=provider, model=model, info=info, enabled=enabled))
    return out


# ---------------------------------------------------------------------------
# Turn-level integration — the call path
# ---------------------------------------------------------------------------
#
# ``route_turn`` is the single entry point the conversation loop calls once per
# user turn, *before* the system prompt is restored/built.  Keeping it there
# means:
#
#   * a disabled install pays one dict lookup and nothing else;
#   * a single-model install pays one list build and returns (spec §4.1);
#   * a switch can only ever happen at a turn boundary, never mid-loop, so the
#     prompt prefix within a model is untouched (spec §10, prompt caching).
#
# It never raises: routing is an optimisation, and a routing bug must not be
# able to break a conversation.

_ROUTER_TURN_ATTR = "_model_router_turn"
_ROUTER_PIN_ATTR = "_model_routing_pinned"
_ROUTER_LAST_ATTR = "_last_routing_decision"

_log = logging.getLogger("hermes.model_router")


def _preferred_models(config: Any, provider: str) -> List[str]:
    """Model ids the user listed for ``provider``, in configured order.

    Reads the shapes that already exist in Hermes config rather than
    introducing a new one: ``providers.<name>.models`` (dict or list, per
    ``normalize_provider_entry``), ``providers.<name>.model``, and the
    ``model`` section (``preferred_models`` / ``models`` / ``default``).
    ``smart_model_routing.candidates`` is accepted as an explicit override.
    """
    if not isinstance(config, dict):
        return []
    out: List[str] = []

    providers = config.get("providers")
    if isinstance(providers, dict):
        entry = providers.get(provider)
        if isinstance(entry, dict):
            models = entry.get("models")
            if isinstance(models, dict):
                out.extend(str(k) for k in models)
            elif isinstance(models, list):
                for item in models:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        mid = item.get("id") or item.get("name")
                        if isinstance(mid, str):
                            out.append(mid)
            single = entry.get("model") or entry.get("default_model")
            if isinstance(single, str):
                out.append(single)

    model_section = config.get("model")
    if isinstance(model_section, dict):
        # Only trust the model section when it describes the same provider;
        # otherwise it would nominate another provider's model as a candidate.
        declared = model_section.get("provider")
        same_provider = not (isinstance(declared, str) and declared.strip()) or (
            declared.strip() == provider
        )
        if same_provider:
            for key in ("preferred_models", "models"):
                val = model_section.get(key)
                if isinstance(val, list):
                    out.extend(str(x) for x in val if isinstance(x, str))
                elif isinstance(val, dict):
                    out.extend(str(k) for k in val)
            default = model_section.get("default")
            if isinstance(default, str):
                out.append(default)

    section = config.get("smart_model_routing")
    if isinstance(section, dict):
        extra = section.get("candidates")
        if isinstance(extra, list):
            out.extend(str(x) for x in extra if isinstance(x, str))

    seen = set()
    uniq: List[str] = []
    for model in out:
        model = model.strip()
        if model and model not in seen:
            seen.add(model)
            uniq.append(model)
    return uniq


def configured_models(config: Any, provider: str) -> List[str]:
    """Public alias for the model list the user configured for ``provider``.

    Read-only; the CLI and the setup wizard use it to explain what routing has
    to choose from without reaching into a private helper.
    """
    return _preferred_models(config, provider)


#: Upper bound on catalog models considered when widening.  A provider can list
#: hundreds; scoring all of them every turn would cost latency for no gain.
_CATALOG_POOL_LIMIT = 60


def _catalog_models(provider: str, limit: int = _CATALOG_POOL_LIMIT) -> List[str]:
    """Model ids models.dev lists for ``provider`` (bounded, read-only).

    Never raises and never fetches: the catalog is already cached by
    :mod:`agent.models_dev`.  An unavailable catalog yields ``[]``, which
    simply means there is nothing to widen to — widening must never be able
    to change behaviour when the operator has not opted in *and* the metadata
    is missing.
    """
    if not provider:
        return []
    try:
        from agent import models_dev

        listed = models_dev.list_provider_models(provider)
    except Exception:  # pragma: no cover - catalog unavailable / defensive
        return []
    out = [str(m).strip() for m in (listed or []) if str(m).strip()]
    return out[: max(1, int(limit))]


def routing_pool(
    config: Any,
    provider: str,
    *,
    current_model: str = "",
) -> Tuple[List[str], str]:
    """The models routing may consider for ``provider``, and the pool scope.

    ``preferred_models_only: true`` (the default) is enforced here and nowhere
    else: the pool is exactly the models the operator listed for the provider
    (plus the model currently in use, which is the incumbent, not a new
    choice), so routing can never select a model that is not on that list.

    With ``preferred_models_only: false`` the returned scope is
    ``"preferred+catalog"``: preferred models are still tried first, and the
    provider's models.dev catalog is consulted *only* when none of them is
    eligible for the request (a tool/modality/context gap).  That is the
    documented purpose of the flag — "allow automatic selection outside
    preferred models for tool-dependent tasks" — not a licence to pick a
    random cheaper model over a perfectly usable preferred one.
    """
    preferred = _preferred_models(config, provider)
    current = str(current_model or "").strip()
    if current and current not in preferred:
        preferred.insert(0, current)

    section = config.get("smart_model_routing") if isinstance(config, dict) else None
    cfg = RoutingConfig.from_dict(section)
    if cfg.preferred_models_only:
        return preferred, "preferred"
    return preferred, "preferred+catalog"


def widen_pool(
    provider: str,
    models: Iterable[str],
    *,
    limit: int = _CATALOG_POOL_LIMIT,
) -> List[str]:
    """``models`` followed by the provider catalog, de-duplicated, order kept.

    Deterministic: the operator's list keeps its priority order and catalog
    additions follow in catalog order, so the same config always produces the
    same pool.
    """
    out: List[str] = []
    seen = set()
    for model in models:
        model = str(model).strip()
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    for model in _catalog_models(provider, limit=limit):
        if model not in seen:
            seen.add(model)
            out.append(model)
    return out


def _configured_default_model(config: Any, provider: str) -> str:
    """The model config.yaml names as this provider's default (\"\" if none)."""
    if not isinstance(config, dict):
        return ""
    model_section = config.get("model")
    if not isinstance(model_section, dict):
        return ""
    declared = model_section.get("provider")
    if isinstance(declared, str) and declared.strip() and declared.strip() != provider:
        return ""
    default = model_section.get("default")
    return default.strip() if isinstance(default, str) else ""


def _configured_provider_keys(config: Any) -> Dict[str, str]:
    """Lower-cased provider key -> the operator's spelling, from ``providers:``.

    Only user-declared profiles are listed: those are the providers whose
    endpoint and credential reference live in ``config.yaml``, so a
    turn-boundary switch has something to resolve.  Built-in providers are
    recognised separately by :func:`_resolve_provider_runtime`.
    """
    out: Dict[str, str] = {}
    if not isinstance(config, dict):
        return out
    providers = config.get("providers")
    if isinstance(providers, dict):
        for name, entry in providers.items():
            if isinstance(name, str) and name.strip() and isinstance(entry, dict):
                out[name.strip().lower()] = name.strip()
    return out


def _split_provider_model(
    config: Any, model: str, *, default_provider: str
) -> Tuple[str, str]:
    """Split a ``"provider/model"`` candidate id when the prefix is configured.

    ``smart_model_routing.candidates`` is a flat list, so a cross-provider
    entry has to name its provider.  The prefix is only treated as a provider
    when it matches a key under ``providers:``; otherwise the id is left whole,
    so a slash-carrying id (an aggregator's ``vendor/model``) keeps pointing at
    the session provider exactly as before.
    """
    text = str(model or "").strip()
    if "/" not in text:
        return default_provider, text
    prefix, rest = text.split("/", 1)
    canonical = _configured_provider_keys(config).get(prefix.strip().lower())
    if not canonical or not rest.strip():
        return default_provider, text
    if canonical.strip().lower() == (default_provider or "").strip().lower():
        return default_provider, rest.strip()
    return canonical, rest.strip()


def _candidates_from_pool(
    config: Any,
    provider: str,
    models: Iterable[str],
    *,
    enabled: bool = True,
) -> List[ModelCandidate]:
    """Build candidates for ``provider`` plus any provider-qualified ids.

    One ``candidates_for`` call per provider keeps the existing metadata
    lookup contract intact; with no qualified ids this is exactly
    ``candidates_for(provider, models)``.
    """
    groups: List[Tuple[str, List[str]]] = []
    index: Dict[str, int] = {}
    for model in models:
        owner, bare = _split_provider_model(config, model, default_provider=provider)
        if owner not in index:
            index[owner] = len(groups)
            groups.append((owner, []))
        groups[index[owner]][1].append(bare)
    out: List[ModelCandidate] = []
    for owner, ids in groups:
        out.extend(candidates_for(owner, ids, enabled=enabled))
    return out


def _call_switch_model(
    agent: Any,
    decision: Decision,
    *,
    runtime: Optional[Dict[str, Any]] = None,
) -> bool:
    """Apply ``decision`` through the existing runtime swap helper.

    ``runtime`` is the target provider's resolved profile (endpoint, key,
    transport).  It is only ever passed for a cross-provider switch, where
    ``agent.switch_model`` would otherwise rebuild the client from the OUTGOING
    provider's ``api_key``/``base_url`` — a request that cannot succeed.
    """
    fn = getattr(agent, "switch_model", None)
    if not callable(fn):
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / mocks without a signature
        params = {}
    accepts_any = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    kwargs: Dict[str, Any] = {}
    if runtime:
        for name in ("api_key", "base_url", "api_mode"):
            value = runtime.get(name)
            if value and (name in params or accepts_any):
                kwargs[name] = value
    if "user_initiated" in params or accepts_any:
        fn(decision.model, decision.provider, user_initiated=False, **kwargs)
    else:
        fn(decision.model, decision.provider, **kwargs)
    return True


_SECRET_PLACEHOLDERS = {"", "no-key-required"}


def _provider_entry(config: Any, provider: str) -> Optional[Dict[str, Any]]:
    """The ``providers:<name>`` block for ``provider`` (case-insensitive)."""
    if not isinstance(config, dict):
        return None
    providers = config.get("providers")
    if not isinstance(providers, dict):
        return None
    wanted = (provider or "").strip().lower()
    for name, entry in providers.items():
        if (
            isinstance(name, str)
            and isinstance(entry, dict)
            and name.strip().lower() == wanted
        ):
            return entry
    return None


def _declared_credential(entry: Optional[Dict[str, Any]]) -> Tuple[str, bool]:
    """``(env var name, inline key present)`` for a provider profile.

    A ``${VAR}`` reference is reported by variable *name*; the value is never
    read into an error message or a log line.
    """
    if not isinstance(entry, dict):
        return "", False
    inline = str(entry.get("api_key") or "").strip()
    if inline.startswith("${") and inline.endswith("}"):
        return inline[2:-1].strip(), False
    return str(entry.get("key_env") or "").strip(), bool(inline)


def _provider_key_env_vars(provider: str, entry: Optional[Dict[str, Any]]) -> List[str]:
    """Credential variables a provider reads, for messages (names only)."""
    var, _inline = _declared_credential(entry)
    if var:
        return [var]
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get((provider or "").strip().lower())
        return [str(v) for v in (getattr(pconfig, "api_key_env_vars", ()) or ()) if str(v)]
    except Exception:  # pragma: no cover - registry unavailable
        return []


def _resolve_provider_runtime(
    config: Any, provider: str, model: str
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Resolve ``provider``'s saved profile into a runtime, or explain the gap.

    Delegates to :func:`hermes_cli.runtime_provider.resolve_runtime_provider`,
    the same provider-profile resolution ``/model`` uses
    (``hermes_cli/model_switch.py:1209``), so the switch inherits that
    provider's endpoint, credential reference and transport instead of a second
    credential code path.  Returns ``(runtime, "")`` on success and
    ``(None, reason)`` when the provider cannot serve a request; every reason
    names credential *variables*, never their values.
    """
    provider = (provider or "").strip()
    if not provider:
        return None, "the selected model has no provider"

    entry = _provider_entry(config, provider)
    if entry is None:
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY

            known = provider.lower() in PROVIDER_REGISTRY
        except Exception:  # pragma: no cover - registry unavailable
            known = False
        if not known:
            # Deliberately resolved no further: the generic resolver falls
            # through to OpenRouter for names it does not know, which would
            # quietly send this model to an aggregator the operator never
            # selected for it.
            return None, (
                f"provider {provider!r} is not configured "
                "(no providers: entry and not a built-in provider)"
            )

    var, inline = _declared_credential(entry)
    if var and not os.environ.get(var, "").strip():
        return None, f"provider {provider!r} has no credentials ({var} is not set)"

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=provider, target_model=model) or {}
    except Exception as err:
        return None, f"provider {provider!r} credentials could not be resolved: {err}"

    api_key = str(runtime.get("api_key") or "").strip()
    base_url = str(runtime.get("base_url") or "").strip()
    if not base_url and not api_key:
        return None, f"provider {provider!r} resolved neither an endpoint nor credentials"
    if api_key in _SECRET_PLACEHOLDERS and (entry is None or var or inline):
        # A placeholder is not a credential: it is what the resolver returns
        # for a local endpoint with no auth.  Only a profile that declares no
        # key at all (a local service) may go out without one.
        names = _provider_key_env_vars(provider, entry)
        hint = f" ({' / '.join(names)} not set)" if names else ""
        return None, f"provider {provider!r} has no API key{hint}"
    return runtime, ""


def route_turn(
    agent: Any,
    user_message: str,
    *,
    attachments: Sequence[str] = (),
    conversation_tokens: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Decision]:
    """Choose the model for one turn and apply it, or return ``None``.

    Returns the :class:`Decision` (so callers can log/explain it) or ``None``
    when routing is off, the model is user-pinned, there is nothing to choose
    between, or no candidate qualifies.
    """
    try:
        if config is None:
            # Read-only fast path: routing never mutates config, and this runs
            # once per user turn (~0.6 ms) instead of the deepcopying
            # ``load_config()`` (~1.2 ms).
            try:
                from hermes_cli.config import load_config_readonly as _load_cfg
            except ImportError:  # pragma: no cover - older tree
                from hermes_cli.config import load_config as _load_cfg

            config = _load_cfg() or {}
        if not is_enabled(config):
            return None
        if getattr(agent, _ROUTER_PIN_ATTR, False):
            return None

        cfg = RoutingConfig.from_dict((config or {}).get("smart_model_routing") or {})
        provider = str(getattr(agent, "provider", "") or "").strip()
        current = str(getattr(agent, "model", "") or "").strip()
        models, pool_scope = routing_pool(config, provider, current_model=current)
        if len(models) <= 1 and pool_scope == "preferred":
            return None  # single-model config: no routing overhead (spec §4.1)

        candidates = _candidates_from_pool(config, provider, models)
        profile = classify_request(
            user_message or "",
            attachments=attachments,
            conversation_tokens=conversation_tokens,
            tool_available=bool(getattr(agent, "tools", None)),
        )
        turn = int(getattr(agent, _ROUTER_TURN_ATTR, 0) or 0)
        if turn == 0 and not cfg.switch_on_cost:
            # First routed turn of a session: nothing is cached yet, so a
            # cost-driven initial selection is not a mid-session downgrade.
            # ``switch_on_cost`` guards the *mid-session* case only.
            cfg = replace(cfg, switch_on_cost=True)
        decision = select_model(profile, candidates, cfg, current_model=current)

        if decision is None and pool_scope != "preferred":
            # Nothing on the operator's preferred list can serve this request.
            # Only now — and only because preferred_models_only is false — is
            # the provider catalog consulted, so a tool/modality gap is not
            # answered by silently keeping an incapable model.
            widened = widen_pool(provider, models)
            if len(widened) > len(models):
                candidates = _candidates_from_pool(config, provider, widened)
                decision = select_model(profile, candidates, cfg, current_model=current)
                if decision is not None and decision.model not in models:
                    decision = replace(
                        decision,
                        reason=decision.reason + "; no preferred model was eligible, "
                        "selected from the provider catalog (preferred_models_only=false)",
                    )
                    _log.info(
                        "routing: widened beyond the preferred list to %s (%s)",
                        decision.key, decision.reason,
                    )

        setattr(agent, _ROUTER_TURN_ATTR, turn + 1)
        if decision is None:
            _log.info(
                "routing: no eligible model among %d candidate(s); keeping %s",
                len(models), current or "(unset)",
            )
            return None

        if decision.model == current:
            _log.info(
                "routing: kept %s (%s) for %s task",
                decision.key, decision.reason, decision.complexity,
            )
            setattr(agent, _ROUTER_LAST_ATTR, decision)
            return decision

        # Mid-session switches need a hard reason: the incumbent must be
        # ineligible for this request, or it is the session's first routed
        # turn on the *configured default* model (no prompt cache and no user
        # choice to protect).  A marginal score win is not enough — changing
        # models invalidates the provider's cached prefix.
        #
        # A session running on a non-default model is a deliberate choice
        # (a gateway /model override, a --model flag, a restored session), so
        # it gets the same protection as a mid-session incumbent.
        eligible = filter_eligible(candidates, profile, cfg)
        incumbent_eligible = any(c.model == current for c in eligible)
        on_configured_default = not _configured_default_model(config, provider) or (
            current == _configured_default_model(config, provider)
        )
        if incumbent_eligible and (turn > 0 or not on_configured_default):
            kept = replace(
                decision,
                model=current,
                provider=provider,
                switched=False,
                reason=(
                    f"kept current model: {decision.key} scored higher but a "
                    "mid-session switch requires a capability gap or ineligible "
                    "incumbent (prompt-cache stability)"
                ),
            )
            _log.info("routing: %s", kept.reason)
            setattr(agent, _ROUTER_LAST_ATTR, kept)
            return kept

        if decision.provider and decision.provider.strip().lower() != provider.strip().lower():
            # Cross-provider selection: the target provider's own profile is
            # the only place its endpoint and credential reference exist, so
            # resolve it here (the same resolution /model uses) and hand it to
            # the swap.  Without this the rebuilt client keeps the outgoing
            # provider's endpoint AND key, so the turn dies instead of moving.
            runtime, why = _resolve_provider_runtime(
                config, decision.provider, decision.model
            )
            if runtime is None:
                # Never guess and never half-switch: name the model and the
                # provider, and keep the session where it is.
                _log.info(
                    "routing: skipped %s: %s; keeping %s",
                    decision.key, why, current or "(unset)",
                )
                return None
            if not _call_switch_model(agent, decision, runtime=runtime):
                _log.info("routing: agent has no switch_model(); keeping %s", current)
                return None
            _log.info(
                "routing: switched %s -> %s (%s) for %s task via %s profile",
                current or "(unset)", decision.key, decision.reason,
                decision.complexity, decision.provider,
            )
            setattr(agent, _ROUTER_LAST_ATTR, decision)
            return decision

        if not _call_switch_model(agent, decision):
            _log.info("routing: agent has no switch_model(); keeping %s", current)
            return None

        _log.info(
            "routing: switched %s -> %s (%s) for %s task",
            current or "(unset)", decision.key, decision.reason, decision.complexity,
        )
        setattr(agent, _ROUTER_LAST_ATTR, decision)
        return decision
    except Exception as err:  # pragma: no cover - routing must never break a turn
        _log.warning("routing: disabled for this turn after error: %s", err)
        return None


# ---------------------------------------------------------------------------
# Preview (CLI `hermes model routing explain`, setup wizard)
# ---------------------------------------------------------------------------

def preview_decision(
    config: Any,
    prompt: str,
    *,
    attachments: Sequence[str] = (),
    conversation_tokens: int = 0,
    provider: str = "",
    current_model: str = "",
    candidates: Optional[Sequence[str]] = None,
) -> Tuple[Optional[Decision], TaskProfile, List[str]]:
    """Select a model for ``prompt`` from config alone.

    Pure: no agent, no network, no config mutation — so it is safe for
    ``hermes model routing explain`` and for the setup wizard to call before
    anything is enabled.  ``candidates`` overrides the configured model list
    (lets an operator preview a multi-model setup before committing to it).

    Returns ``(decision, profile, candidate_models)``.  ``decision`` is
    ``None`` when routing is off, when fewer than two models are in play, or
    when no candidate satisfies the request's hard constraints.
    """
    cfg_section = config.get("smart_model_routing") if isinstance(config, dict) else None
    cfg = RoutingConfig.from_dict(cfg_section)

    prov = str(provider or "").strip()
    if not prov and isinstance(config, dict):
        model_section = config.get("model")
        if isinstance(model_section, dict):
            prov = str(model_section.get("provider") or "").strip()

    if candidates is not None:
        # An explicit list is the whole candidate set (the operator is
        # previewing a hypothetical config), so no model is injected into it
        # and ``preferred_models_only`` does not apply to it.
        models = [str(m).strip() for m in candidates if str(m).strip()]
        current = str(current_model or "").strip()
        if not current:
            default = _configured_default_model(config, prov)
            current = default if default in models else ""
        pool_scope = "explicit"
    else:
        models, pool_scope = routing_pool(
            config, prov, current_model=str(current_model or "").strip()
        )
        current = str(current_model or "").strip() or _configured_default_model(config, prov)

    profile = classify_request(
        prompt or "",
        attachments=attachments,
        conversation_tokens=conversation_tokens,
    )
    if len(models) <= 1 and pool_scope != "preferred+catalog":
        return None, profile, models

    cands = _candidates_from_pool(config, prov, models)
    # A preview is a fresh session's first turn: nothing is prompt-cached yet,
    # so a cost-driven choice is not a mid-session downgrade.
    preview_cfg = replace(cfg, switch_on_cost=True) if not cfg.switch_on_cost else cfg
    decision = select_model(profile, cands, preview_cfg, current_model=current)
    if decision is None and pool_scope == "preferred+catalog":
        widened = widen_pool(prov, models)
        if len(widened) > len(models):
            cands = _candidates_from_pool(config, prov, widened)
            decision = select_model(profile, cands, preview_cfg, current_model=current)
            if decision is not None and decision.model not in models:
                models = widened
                decision = replace(
                    decision,
                    reason=decision.reason + "; no preferred model was eligible, "
                    "selected from the provider catalog (preferred_models_only=false)",
                )
    return decision, profile, models
