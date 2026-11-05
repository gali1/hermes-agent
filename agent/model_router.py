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
* **One scoring path.** Every consumer — the per-turn hook, ``hermes model
  routing explain`` and ``hermes model routing status --json`` — goes through
  :func:`rank_candidates`.  A report can therefore never disagree with what a
  turn would actually do for the same inputs.
* **Shared config contract.** ``smart_model_routing.models`` is the canonical
  candidate list (see :class:`ModelSpec`); the pre-contract flat
  ``candidates`` string list is still read, deprecated, and translated into the
  same specs.  Both shapes end up in the identical scoring path.
* **Off by default.** ``smart_model_routing.enabled`` is ``False``.  Callers
  must consult :func:`is_enabled` before doing anything else, so an untouched
  install behaves exactly as it did before this module existed.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CapabilityState",
    "CandidateReport",
    "Complexity",
    "Decision",
    "ModelCandidate",
    "ModelSpec",
    "RoutingConfig",
    "SKIP_REASONS",
    "SPEC_FIELDS",
    "Selection",
    "TaskProfile",
    "candidates_from_specs",
    "charge_for",
    "classify_request",
    "configured_models",
    "declared_model_list",
    "default_weights",
    "estimate_cost",
    "estimate_cost_breakdown",
    "expected_retry_fraction",
    "explain",
    "filter_eligible",
    "is_enabled",
    "parse_model_ids",
    "parse_reliability",
    "parse_weights",
    "pricing_for",
    "rank_candidates",
    "resolve_candidates",
    "route_turn",
    "routing_pool",
    "routing_preview",
    "routing_report",
    "routing_settings",
    "score_candidate",
    "select_model",
    "specs_from_config",
    "update_routing",
    "validate_routing",
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
    #: Input tokens expected to hit the provider's prompt cache (billed at the
    #: cache-read rate when the catalog prices one).  0 on a cold turn.
    est_cached_input_tokens: int = 0
    #: Reasoning tokens the turn will pay for on top of the answer; billed at the
    #: output rate because that is how providers charge them.
    est_reasoning_tokens: int = 0
    #: Number of image attachments, the basis for the modality term (each is
    #: priced as a fixed token equivalent at the input rate — the catalog has no
    #: per-image surcharge field, and inventing one would be a fiction).
    image_inputs: int = 0
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


#: Content-part types that carry an attachment, mapped to the modality hint the
#: request classifier understands. Anything else in a parts list contributes text
#: (or nothing), never a modality — an unrecognised part must not invent one.
_PART_MODALITY = {
    "image_url": "image",
    "image": "image",
    "input_image": "image",
    "input_audio": "audio",
    "audio": "audio",
    "video": "video",
    "input_video": "video",
}

_FILE_SUFFIX_MODALITY = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image", ".tif": "image", ".tiff": "image",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".ogg": "audio",
    ".flac": "audio",
    ".pdf": "pdf",
    ".mp4": "video", ".mov": "video", ".mkv": "video", ".webm": "video",
}


def _modality_of_part(part: Dict[str, Any]) -> Optional[str]:
    kind = str(part.get("type") or "").strip().lower()
    if kind in _PART_MODALITY:
        return _PART_MODALITY[kind]
    if kind == "file":
        raw = part.get("file")
        filename = ""
        if isinstance(raw, dict):
            filename = str(raw.get("filename") or raw.get("name") or "")
        elif isinstance(raw, str):
            filename = raw
        suffix = Path(filename).suffix.lower() if filename else ""
        return _FILE_SUFFIX_MODALITY.get(suffix, "pdf" if suffix == ".pdf" else None)
    return None


def message_text_and_attachments(user_message: Any) -> Tuple[str, Tuple[str, ...]]:
    """Split one turn's message into the text the router classifies.

    A turn can arrive as a plain string or as a list of content parts (the
    OpenAI-compatible shape a client sends when it mixes text with images, audio
    or files). Routing must see both halves: the text decides complexity, the
    attachments decide the input-modality requirement (contract C9a). Coercing a
    parts list with ``str()`` — the old behaviour — kept the text but dropped the
    attachments, so an image turn could be routed to a model that cannot read
    images.

    Returns ``(text, attachments)`` where ``attachments`` holds modality hints
    (``"image"``, ``"audio"``, ``"pdf"``, ``"video"``), de-duplicated, in first-seen
    order. Unknown part types contribute nothing.
    """
    if user_message is None:
        return "", ()
    if isinstance(user_message, str):
        return user_message, ()
    if isinstance(user_message, dict):
        parts: Sequence[Any] = [user_message]
    elif isinstance(user_message, (list, tuple)):
        parts = user_message
    else:
        return str(user_message), ()

    chunks: List[str] = []
    attachments: List[str] = []
    for part in parts:
        if isinstance(part, str):
            chunks.append(part)
            continue
        if not isinstance(part, dict):
            continue
        modality = _modality_of_part(part)
        if modality and modality not in attachments:
            attachments.append(modality)
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
    return "\n".join(chunks).strip(), tuple(attachments)


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
    cached_input_tokens: int = 0,
    expected_reasoning_tokens: Optional[int] = None,
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
    # One strong signal is enough to say "this is not a trivial ask", however
    # few tokens it took to type it (spec §7: a short request can be advanced).
    # Without this floor a short prompt could never climb past ROUTINE — every
    # volume step starts at SIMPLE, so +1 lands on ROUTINE and stops.  MODERATE
    # is the floor, not COMPLEX: a single keyword must not push work into an
    # expensive reasoner, and the FACTUAL override below still wins for a
    # plainly factual question.
    if code_hit or reasoning_hit:
        level = Complexity(max(int(level), int(Complexity.MODERATE)))
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
    image_inputs = 0
    for hint in attachments:
        h = str(hint).strip().lower()
        if not h:
            continue
        if h == "image":
            image_inputs += 1
        elif h.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
            image_inputs += 1
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

    # Prompt-cache credit is caller-supplied: only the turn hook knows whether the
    # stable prefix was already sent (a cold first turn pays full input price).  It
    # is clamped to the input estimate so it can never manufacture a negative bill.
    cached_tokens = max(0, min(int(cached_input_tokens or 0), in_tokens))

    # Reasoning tokens are billed as output by every provider we price; assume a
    # half-answer's worth of deliberation for a reasoning task, and none otherwise.
    # Explicitly overridable by a caller that measures its own reasoning budget.
    if expected_reasoning_tokens is None:
        reasoning_tokens = int(round(out_tokens * 0.5)) if needs_reasoning else 0
    else:
        reasoning_tokens = max(0, int(expected_reasoning_tokens))
    if reasoning_tokens:
        signals.append("reasoning-tokens")

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
        est_cached_input_tokens=cached_tokens,
        est_reasoning_tokens=reasoning_tokens,
        image_inputs=image_inputs,
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
# Declared model metadata (the shared ``smart_model_routing.models`` contract)
# ---------------------------------------------------------------------------

#: The metadata a ``smart_model_routing.models`` entry may declare.  Every
#: field is an OVERRIDE over the models.dev catalog entry for the same model;
#: ``None`` means "not declared", which is what makes a capability UNKNOWN
#: rather than unsupported.
SPEC_FIELDS = (
    "name",
    "provider",
    "context_window",
    "max_output",
    "cost_in",
    "cost_out",
    "cost_cache_read",
    "tier_rank",
    "supports_tools",
    "supports_vision",
    "supports_audio",
    "supports_structured",
    "supports_text_output",
    "supports_image_output",
    "supports_audio_output",
    "deprecated",
)

_BOOL_SPEC_FIELDS = (
    "supports_tools",
    "supports_vision",
    "supports_audio",
    "supports_structured",
    "supports_text_output",
    "supports_image_output",
    "supports_audio_output",
    "deprecated",
)
_INT_SPEC_FIELDS = ("context_window", "max_output", "tier_rank")
_COST_SPEC_FIELDS = ("cost_in", "cost_out", "cost_cache_read")


def _as_bool(value: Any) -> Optional[bool]:
    """Coerce a declared flag; anything unparseable is "not declared"."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Only a real 0/1 is a declared flag.  Anything else numeric is
        # "not declared" so a malformed value is treated as unknown rather
        # than as a confident true/false.
        return bool(value) if value in (0, 1) else None
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return None


def _as_positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _as_rate(value: Any) -> Optional[float]:
    """A USD-per-million price.  Zero/negative is treated as "unpriced" so an
    unpriced model can never look free (it is charged the median instead)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _as_tier(value: Any) -> Optional[int]:
    """A capability tier rank.  Accepts an int or a tier name."""
    if isinstance(value, str):
        named = {"simple": 1, "routine": 2, "moderate": 3, "complex": 3, "advanced": 4}
        text = value.strip().lower()
        if text in named:
            return named[text]
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= 4 else None


@dataclass(frozen=True)
class ModelSpec:
    """One ``smart_model_routing.models`` entry: declared metadata overrides.

    ``name`` is the model id (may be ``"provider/model"`` when the entry does
    not repeat the provider).  Everything else is optional: an undeclared field
    stays whatever models.dev says, and stays UNKNOWN when the catalog has no
    entry — never "unsupported".
    """

    name: str
    provider: Optional[str] = None
    context_window: Optional[int] = None
    max_output: Optional[int] = None
    cost_in: Optional[float] = None
    cost_out: Optional[float] = None
    cost_cache_read: Optional[float] = None
    tier_rank: Optional[int] = None
    supports_tools: Optional[bool] = None
    supports_vision: Optional[bool] = None
    supports_audio: Optional[bool] = None
    supports_structured: Optional[bool] = None
    supports_text_output: Optional[bool] = None
    supports_image_output: Optional[bool] = None
    supports_audio_output: Optional[bool] = None
    deprecated: Optional[bool] = None

    @classmethod
    def from_entry(cls, entry: Any) -> Optional["ModelSpec"]:
        """Parse one entry; ``None`` when it names no model."""
        if isinstance(entry, str):
            name = entry.strip()
            return cls(name=name) if name else None
        if not isinstance(entry, dict):
            return None
        name = str(entry.get("name") or entry.get("model") or entry.get("id") or "").strip()
        if not name:
            return None
        kwargs: Dict[str, Any] = {"name": name}

        provider = entry.get("provider")
        if isinstance(provider, str) and provider.strip():
            kwargs["provider"] = provider.strip()

        for key in _INT_SPEC_FIELDS:
            if key in entry:
                parsed = _as_positive_int(entry.get(key))
                if parsed is not None:
                    kwargs[key] = parsed
        if "tier_rank" in entry:
            tier = _as_tier(entry.get("tier_rank"))
            if tier is not None:
                kwargs["tier_rank"] = tier
        for key in _COST_SPEC_FIELDS:
            if key in entry:
                rate = _as_rate(entry.get(key))
                if rate is not None:
                    kwargs[key] = rate
        for key in _BOOL_SPEC_FIELDS:
            if key in entry:
                flag = _as_bool(entry.get(key))
                if flag is not None:
                    kwargs[key] = flag
        return cls(**kwargs)

    def describe(self) -> str:
        return self.name


def specs_from_config(config: Any) -> Tuple[List[ModelSpec], str]:
    """The declared candidate specs and where they came from.

    ``smart_model_routing.models`` is canonical.  The pre-contract flat
    ``smart_model_routing.candidates`` list is still honoured — deprecated, and
    only when ``models`` is absent — by translating each ``model`` /
    ``provider/model`` string into the same :class:`ModelSpec` the canonical key
    produces, so both shapes share one scoring path.

    Returns ``([], "")`` when neither key is usable; callers then fall back to
    the provider's configured model list.
    """
    if not isinstance(config, dict):
        return [], ""
    section = config.get("smart_model_routing")
    if not isinstance(section, dict):
        return [], ""

    declared = section.get("models")
    if isinstance(declared, (list, tuple)):
        specs = [spec for spec in (ModelSpec.from_entry(e) for e in declared) if spec]
        if specs:
            return specs, "models"

    legacy = section.get("candidates")
    if isinstance(legacy, (list, tuple)):
        specs = [spec for spec in (ModelSpec.from_entry(e) for e in legacy) if spec]
        if specs:
            return specs, "candidates"

    return [], ""


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

@dataclass
class ModelCandidate:
    """A model that *could* serve a request, plus what we know about it.

    ``info`` may be ``None`` when models.dev has no entry (a local/custom
    model, or an offline catalog miss).  ``tool_support`` lets a caller pass a
    runtime-verified answer that overrides catalog metadata.  ``spec`` carries
    the operator's declared overrides and wins over both — see
    :class:`ModelSpec`.
    """

    provider: str
    model: str
    info: Any = None  # agent.models_dev.ModelInfo
    tool_support: Optional[CapabilityState] = None
    #: False when the provider is disabled/without credentials.
    enabled: bool = True
    #: Optional measured latency in ms.  NOT populated by ``candidates_from_specs`` and
    #: never read by the ranker, so it is not a tie-breaker today: equal scores resolve
    #: alphabetically in ``rank_candidates``.  The field is kept for API compatibility;
    #: do not document it as behaviour until something fills it in.
    latency_ms: Optional[float] = None
    #: Optional historical success rate 0..1; None means unknown.  Only
    #: ``smart_model_routing.reliability`` (operator config) can currently supply a
    #: value — nothing measures outcomes and assigns this field.
    reliability: Optional[float] = None
    #: Declared overrides from ``smart_model_routing.models[]`` (or its
    #: translated legacy equivalent).  ``None`` fields are "not declared".
    spec: Optional["ModelSpec"] = None

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    # -- derived metadata ---------------------------------------------------
    def _mods(self, attr: str) -> Tuple[str, ...]:
        mods = getattr(self.info, attr, ()) if self.info is not None else ()
        return tuple(mods or ())

    def _declared(self, name: str) -> Optional[bool]:
        if self.spec is None:
            return None
        return getattr(self.spec, name, None)

    def supports_input(self, modality: str) -> CapabilityState:
        if modality == "text":
            return CapabilityState.SUPPORTED
        declared = None
        if modality == "image":
            declared = self._declared("supports_vision")
        elif modality == "audio":
            declared = self._declared("supports_audio")
        if declared is not None:
            return CapabilityState.SUPPORTED if declared else CapabilityState.UNSUPPORTED
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
            declared_text = self._declared("supports_text_output")
            if declared_text is not None:
                return CapabilityState.SUPPORTED if declared_text else CapabilityState.UNSUPPORTED
        else:
            # C9b needs an operator-declarable OUTPUT capability. Without one a
            # text-only model stays UNKNOWN (not UNSUPPORTED) and therefore
            # eligible, so an image-edit request cannot exclude it.
            declared_out = self._declared(f"supports_{modality}_output")
            if declared_out is not None:
                return (CapabilityState.SUPPORTED if declared_out
                        else CapabilityState.UNSUPPORTED)
        if self.info is None:
            return CapabilityState.UNKNOWN
        mods = self._mods("output_modalities")
        if not mods:
            return CapabilityState.UNKNOWN
        return CapabilityState.SUPPORTED if modality in mods else CapabilityState.UNSUPPORTED

    def supports_tools(self) -> CapabilityState:
        declared = self._declared("supports_tools")
        if declared is not None:
            return CapabilityState.SUPPORTED if declared else CapabilityState.UNSUPPORTED
        if self.tool_support is not None:
            return self.tool_support
        if self.info is None:
            return CapabilityState.UNKNOWN
        # models.dev defaults the flag to False when the field is absent, so a
        # False here means "not confirmed", never "confirmed unsupported".
        return CapabilityState.SUPPORTED if getattr(self.info, "tool_call", False) else CapabilityState.UNKNOWN

    def supports_structured_output(self) -> CapabilityState:
        declared = self._declared("supports_structured")
        if declared is not None:
            return CapabilityState.SUPPORTED if declared else CapabilityState.UNSUPPORTED
        if self.info is None:
            return CapabilityState.UNKNOWN
        # Same reasoning as ``tool_call``: an absent flag is not a refusal.
        return (
            CapabilityState.SUPPORTED
            if getattr(self.info, "structured_output", False)
            else CapabilityState.UNKNOWN
        )

    def supports_reasoning(self) -> CapabilityState:
        if self.info is None:
            return CapabilityState.UNKNOWN
        return CapabilityState.SUPPORTED if getattr(self.info, "reasoning", False) else CapabilityState.UNKNOWN

    def context_window(self) -> Optional[int]:
        if self.spec is not None and self.spec.context_window:
            return int(self.spec.context_window)
        ctx = getattr(self.info, "context_window", 0) if self.info is not None else 0
        return int(ctx) if ctx and ctx > 0 else None

    def max_output(self) -> Optional[int]:
        if self.spec is not None and self.spec.max_output:
            return int(self.spec.max_output)
        out = getattr(self.info, "max_output", 0) if self.info is not None else 0
        return int(out) if out and out > 0 else None

    def is_deprecated(self) -> bool:
        declared = self._declared("deprecated")
        if declared is not None:
            return bool(declared)
        return str(getattr(self.info, "status", "") or "").lower() == "deprecated"

    def tier_rank(self) -> Optional[int]:
        """Declared capability tier (1..4), or ``None`` when undeclared."""
        if self.spec is not None and self.spec.tier_rank:
            return int(self.spec.tier_rank)
        return None

    def costs(self) -> Tuple[Optional[float], Optional[float]]:
        """Declared-or-catalog ``(cost_in, cost_out)``; ``None`` when unpriced."""
        ci = co = None
        if self.spec is not None:
            ci, co = self.spec.cost_in, self.spec.cost_out
        if ci is None and co is None:
            if self.info is None:
                return None, None
            ci = getattr(self.info, "cost_input", 0) or 0
            co = getattr(self.info, "cost_output", 0) or 0
        ci = float(ci) if ci and ci > 0 else 0.0
        co = float(co) if co and co > 0 else 0.0
        if ci <= 0 and co <= 0:
            return None, None
        return ci, co

    def cache_cost(self) -> Optional[float]:
        """Cache-read price, declared override first, else catalog, else ``None``.

        ``None`` is not "free": the cost model falls back to the input rate, which
        is the conservative reading (a model with no published cache discount is
        billed as if every cached token still cost full price).
        """
        if self.spec is not None and self.spec.cost_cache_read:
            return float(self.spec.cost_cache_read)
        rate = getattr(self.info, "cost_cache_read", None) if self.info is not None else None
        try:
            value = float(rate) if rate else 0.0
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None


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
    """Resolved ``smart_model_routing`` settings with safe defaults.

    The keys shared with the sibling tree's contract are ``enabled``,
    ``strategy``, ``stability_threshold`` (alias ``min_improvement``),
    ``allow_unknown_metadata``, ``cost_optimization`` and
    ``preferred_models_only``.  ``switch_on_cost``, ``llm_assisted_classification``,
    ``weights``, ``reliability`` and ``exclude`` are local extensions that only
    ever make selection *more* conservative.
    """

    enabled: bool = False
    strategy: str = "automatic"  # automatic | cost | quality
    #: Shared contract default with bro-2 (docs/routing-contract.md C6): a restriction needs an
    #: explicit opt-in. With ``true`` and no ``models`` list, routing silently stops selecting,
    #: which is a worse fresh-install surprise than routing across the eligible catalog.
    preferred_models_only: bool = False
    #: True when the operator wrote the key, so a single-model provider list can stay a no-op
    #: without pretending the default was ``true``.
    preferred_models_only_explicit: bool = False
    cost_optimization: bool = True
    #: A rival must beat the incumbent by MORE than this to take the turn
    #: (ties and borderline wins keep the current model).
    stability_threshold: float = 0.15
    #: Allow a mid-session switch to a *cheaper* model.  Off by default: a
    #: model change invalidates the provider's prompt cache, so a downgrade
    #: must be an explicit choice, not an automatic one.
    switch_on_cost: bool = False
    allow_unknown_metadata: bool = True
    llm_assisted_classification: bool = False
    weights: Dict[str, float] = field(default_factory=dict)
    #: model key ("provider/model") or bare model -> 0..1 success rate.
    reliability: Dict[str, float] = field(default_factory=dict)
    #: models the operator never wants auto-selected (still usable manually).
    exclude: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # A fresh dict per instance: the module-level default is shared state.
        self.weights = dict(_DEFAULT_WEIGHTS) if not self.weights else dict(self.weights)

    @classmethod
    def from_dict(cls, raw: Any) -> "RoutingConfig":
        cfg = cls()
        if not isinstance(raw, dict):
            return cfg

        def _flag(name: str, default: bool) -> bool:
            """A boolean knob: only an unambiguous value can turn it on.

            ``bool("no")`` is ``True``, so a quoted YAML scalar would silently
            enable a feature the user switched off.  ``_as_bool`` is the one
            place that knows what "true" means (and ``None`` means "unusable",
            which falls back to the default — :func:`validate_routing` reports
            it as an error instead of guessing).
            """
            if name not in raw:
                return default
            parsed = _as_bool(raw.get(name))
            return default if parsed is None else parsed

        cfg.enabled = _flag("enabled", cfg.enabled)
        strategy = str(raw.get("strategy", cfg.strategy) or cfg.strategy).lower()
        cfg.strategy = strategy if strategy in ("automatic", "cost", "quality") else "automatic"
        cfg.preferred_models_only = _flag("preferred_models_only", cfg.preferred_models_only)
        cfg.preferred_models_only_explicit = "preferred_models_only" in raw
        cfg.cost_optimization = _flag("cost_optimization", cfg.cost_optimization)
        # ``stability_threshold`` is canonical; ``min_improvement`` is the
        # sibling tree's alias for the same knob.
        threshold = raw.get("stability_threshold", raw.get("min_improvement", cfg.stability_threshold))
        try:
            cfg.stability_threshold = max(0.0, float(threshold))
        except (TypeError, ValueError):
            pass
        cfg.allow_unknown_metadata = _flag("allow_unknown_metadata", cfg.allow_unknown_metadata)
        cfg.switch_on_cost = _flag("switch_on_cost", cfg.switch_on_cost)
        cfg.llm_assisted_classification = _flag(
            "llm_assisted_classification", cfg.llm_assisted_classification)
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

    def settings(self) -> Dict[str, Any]:
        """The settings block of the shared JSON report (exact key set)."""
        return {
            "strategy": self.strategy,
            "stability_threshold": self.stability_threshold,
            "allow_unknown_metadata": self.allow_unknown_metadata,
            "cost_optimization": self.cost_optimization,
            "preferred_models_only": self.preferred_models_only,
        }


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
        if not isinstance(section, dict) or "enabled" not in section:
            return False
        # Same coercion as :meth:`RoutingConfig.from_dict`: a quoted ``"no"``
        # must not switch routing on just because it is a non-empty string.
        return _as_bool(section.get("enabled")) is True
    return False


# ---------------------------------------------------------------------------
# The config surface: one read / validate / mutate / preview path
# ---------------------------------------------------------------------------
#
# ``smart_model_routing`` is deliberately absent from ``DEFAULT_CONFIG``: an
# untouched install loads byte-identically because nothing materialises the
# block, and ``is_enabled`` sees no section.  Everything that READS or WRITES
# the block goes through this section, so the CLI, the setup wizard, the gateway
# and ``validate_config_structure`` can never disagree about the schema.

#: Every key a ``smart_model_routing`` block may carry.  The first group is the
#: shared cross-tree contract; the rest are knobs :class:`RoutingConfig` already
#: reads, so exposing them adds no new engine behaviour.
ROUTING_KEYS: Tuple[str, ...] = (
    "enabled",
    "strategy",
    "stability_threshold",
    "models",
    "preferred_models_only",
    "cost_optimization",
    "switch_on_cost",
    "allow_unknown_metadata",
    "llm_assisted_classification",
    "exclude",
    "weights",
    "reliability",
)

#: Deprecated spellings the engine still honours.  Read and migrated, never
#: written by :func:`update_routing`.
ROUTING_LEGACY_KEYS: Tuple[str, ...] = ("candidates", "min_improvement")

#: Accepted ``strategy`` values (see :class:`RoutingConfig`).
ROUTING_STRATEGIES: Tuple[str, ...] = ("automatic", "cost", "quality")

_BOOLEAN_ROUTING_KEYS: Tuple[str, ...] = (
    "enabled",
    "preferred_models_only",
    "cost_optimization",
    "switch_on_cost",
    "allow_unknown_metadata",
    "llm_assisted_classification",
)


def _routing_section_raw(config: Any) -> Dict[str, Any]:
    """The raw ``smart_model_routing`` mapping of *config* (``{}`` when absent)."""
    if not isinstance(config, dict):
        return {}
    section = config.get("smart_model_routing")
    return section if isinstance(section, dict) else {}


def _routing_block_from_disk(config: Any = None) -> Dict[str, Any]:
    """The block to read: *config* when given, else the on-disk config.

    Total by construction — an unreadable or absent config is an absent block,
    which means "off", never an exception.
    """
    if config is not None:
        return _routing_section_raw(config)
    try:
        from hermes_cli.config import load_config_readonly

        return _routing_section_raw(load_config_readonly())
    except Exception:  # pragma: no cover - defensive: reading config never raises here
        return {}


def _model_names(block: Dict[str, Any]) -> List[str]:
    """Model ids declared by a block, canonical ``models`` first, legacy second."""
    for key in ("models", "candidates"):
        entries = block.get(key)
        if not isinstance(entries, (list, tuple)):
            continue
        names: List[str] = []
        for entry in entries:
            if isinstance(entry, str):
                name = entry.strip()
            elif isinstance(entry, dict):
                name = str(entry.get("name") or entry.get("model") or "").strip()
            else:
                name = ""
            if name and name not in names:
                names.append(name)
        if names:
            return names
    return []


def routing_settings(config: Any = None) -> Dict[str, Any]:
    """The normalised effective ``smart_model_routing`` block.

    *config* is a whole config mapping; ``None`` reads the live config from
    disk.  An absent (or malformed) block yields the disabled defaults — this
    function never raises, because every report and status path calls it.
    """
    block = _routing_block_from_disk(config)
    cfg = RoutingConfig.from_dict(block)
    return {
        "enabled": bool(cfg.enabled),
        "strategy": cfg.strategy,
        "stability_threshold": float(cfg.stability_threshold),
        "models": _model_names(block),
        "preferred_models_only": bool(cfg.preferred_models_only),
        "cost_optimization": bool(cfg.cost_optimization),
        "switch_on_cost": bool(cfg.switch_on_cost),
        "allow_unknown_metadata": bool(cfg.allow_unknown_metadata),
        "llm_assisted_classification": bool(cfg.llm_assisted_classification),
        "exclude": list(cfg.exclude),
        "weights": dict(cfg.weights),
        "reliability": dict(cfg.reliability),
    }


def validate_routing(block: Any) -> List[str]:
    """Every reason *block* is not a usable ``smart_model_routing`` section.

    ``[]`` means valid.  Each message names the offending key, so a caller can
    print the list as-is and the user knows which line to fix.  An absent block
    (``None`` / ``{}``) is valid: absence means "off".

    This validates the block's *shape and values* only — it never scores, never
    resolves candidates and never touches the network, so
    ``validate_config_structure`` can call it on every startup.
    """
    problems: List[str] = []
    if block is None:
        return problems
    if not isinstance(block, dict):
        return [f"smart_model_routing must be a mapping, got {type(block).__name__}"]

    known = set(ROUTING_KEYS) | set(ROUTING_LEGACY_KEYS)
    for key in block:
        if str(key) not in known:
            problems.append(
                f"smart_model_routing.{key} is not a recognised routing option "
                f"(known: {', '.join(ROUTING_KEYS)})")

    for key in _BOOLEAN_ROUTING_KEYS:
        if key in block and _as_bool(block[key]) is None:
            problems.append(
                f"smart_model_routing.{key} must be true or false, got {block[key]!r}")

    if "strategy" in block:
        strategy = block["strategy"]
        if not isinstance(strategy, str) or strategy.strip().lower() not in ROUTING_STRATEGIES:
            problems.append(
                f"smart_model_routing.strategy must be one of "
                f"{', '.join(ROUTING_STRATEGIES)}, got {strategy!r}")

    thresholds = [key for key in ("stability_threshold", "min_improvement") if key in block]
    if len(thresholds) == 2:
        problems.append(
            "smart_model_routing.stability_threshold and smart_model_routing.min_improvement "
            "are the same knob — set only stability_threshold")
    for key in thresholds:
        value = block[key]
        try:
            number = float(value)
        except (TypeError, ValueError):
            problems.append(
                f"smart_model_routing.{key} must be a number between 0 and 1, got {value!r}")
            continue
        if not (0.0 <= number <= 1.0):
            problems.append(
                f"smart_model_routing.{key} must be between 0 and 1, got {value!r}")

    for key in ("models", "candidates", "exclude"):
        if key not in block:
            continue
        entries = block[key]
        if not isinstance(entries, (list, tuple)):
            problems.append(
                f"smart_model_routing.{key} must be a YAML list, got {type(entries).__name__}")
            continue
        for index, entry in enumerate(entries):
            if key == "exclude":
                if not isinstance(entry, str) or not entry.strip():
                    problems.append(
                        f"smart_model_routing.exclude[{index}] must be a model name, got {entry!r}")
                continue
            if isinstance(entry, str):
                if not entry.strip():
                    problems.append(f"smart_model_routing.{key}[{index}] is an empty model name")
                continue
            if not isinstance(entry, dict):
                problems.append(
                    f"smart_model_routing.{key}[{index}] must be a model name or a mapping, "
                    f"got {type(entry).__name__}")
                continue
            name = entry.get("name", entry.get("model"))
            if not isinstance(name, str) or not name.strip():
                problems.append(
                    f"smart_model_routing.{key}[{index}].name is required for a mapping entry")
            for field_name in entry:
                if str(field_name) not in SPEC_FIELDS and str(field_name) != "model":
                    problems.append(
                        f"smart_model_routing.{key}[{index}].{field_name} is not a declared "
                        f"model field (known: {', '.join(SPEC_FIELDS)})")

    if "weights" in block:
        weights = block["weights"]
        if not isinstance(weights, dict):
            problems.append(
                f"smart_model_routing.weights must be a mapping, got {type(weights).__name__}")
        else:
            total = 0.0
            for name, value in weights.items():
                if str(name) not in _DEFAULT_WEIGHTS:
                    problems.append(
                        f"smart_model_routing.weights.{name} is not a scoring dimension "
                        f"(known: {', '.join(_DEFAULT_WEIGHTS)})")
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    problems.append(
                        f"smart_model_routing.weights.{name} must be a number, got {value!r}")
                    continue
                if number < 0:
                    problems.append(
                        f"smart_model_routing.weights.{name} must not be negative, got {value!r}")
                    continue
                total += number
            if total <= 0:
                problems.append("smart_model_routing.weights must have at least one positive value")

    if "reliability" in block:
        reliability = block["reliability"]
        if not isinstance(reliability, dict):
            problems.append(
                f"smart_model_routing.reliability must be a mapping of model → rate, "
                f"got {type(reliability).__name__}")
        else:
            for name, value in reliability.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    problems.append(
                        f"smart_model_routing.reliability.{name} must be a rate between 0 and 1, "
                        f"got {value!r}")
                    continue
                if not (0.0 <= number <= 1.0):
                    problems.append(
                        f"smart_model_routing.reliability.{name} must be between 0 and 1, "
                        f"got {value!r}")
    return problems


def _jsonable(value: Any) -> Any:
    """*value* with tuples turned into lists, so a returned block is JSON-clean."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Typed-input parsers — one implementation for every surface
# ---------------------------------------------------------------------------
#
# ``exclude``, ``weights`` and ``reliability`` are read by the engine, so every
# interface that can set them (CLI flags, the CLI's interactive flow, the
# gateway's preferences steps) must read a typed value the same way.  Parsing
# lives here, next to the schema, rather than in each surface: two parsers is
# how ``a,b`` starts meaning different things in two places.

def parse_model_ids(text: Any) -> List[str]:
    """Comma- or whitespace-separated model ids → ordered, de-duplicated list.

    ``None`` and ``""`` yield ``[]`` (an explicit "none"), so a caller can use
    one parser for "set" and "clear".
    """
    if text is None:
        return []
    chunks = [str(chunk) for chunk in text] if isinstance(text, (list, tuple)) else [str(text)]
    out: List[str] = []
    for chunk in chunks:
        for part in re.split(r"[,\s]+", chunk.strip()):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


def _parse_assignments(text: Any, *, what: str) -> Dict[str, Any]:
    """``name=value`` pairs → mapping, numbers where they parse.

    A missing ``=`` raises :class:`ValueError` (a typed value the user can fix
    without reading the source).  A value that is not a number is passed
    through as a string so :func:`validate_routing` names the offending key and
    the accepted range instead of a second, differently-worded parser message.
    """
    pairs: Dict[str, Any] = {}
    chunks = [str(chunk) for chunk in text] if isinstance(text, (list, tuple)) else [str(text or "")]
    for chunk in chunks:
        for part in re.split(r"[,\s]+", chunk.strip()):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"{what} entry {part!r} must look like name=value")
            name, _, raw = part.partition("=")
            name, raw = name.strip(), raw.strip()
            if not name:
                raise ValueError(f"{what} entry {part!r} has no name")
            try:
                pairs[name] = float(raw)
            except (TypeError, ValueError):
                pairs[name] = raw
    return pairs


def default_weights() -> Dict[str, float]:
    """A copy of the engine's default scoring weights (never the shared dict)."""
    return dict(_DEFAULT_WEIGHTS)


def parse_weights(text: Any, *, base: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge ``capability=0.5,cost=0.3`` over *base* (the current weights).

    Only the dimensions named in the text change; the rest keep their current
    (or default) value, so a user can tune one number without retyping five.
    """
    merged: Dict[str, Any] = dict(base or {})
    merged.update(_parse_assignments(text, what="weights"))
    return merged


def parse_reliability(text: Any, *, base: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge ``model=0.98,other=0.95`` over *base* (declared success rates)."""
    merged: Dict[str, Any] = dict(base or {})
    merged.update(_parse_assignments(text, what="reliability"))
    return merged


def update_routing(
    changes: Dict[str, Any], *, config: Any = None, dry_run: bool = False,
) -> Dict[str, Any]:
    """Change ``smart_model_routing``: validate → mutate → atomic save → read back.

    *changes* is the subset of keys to set.  Nothing is written unless the
    merged block validates: an invalid value raises :class:`ValueError` naming
    the offending key and leaves config.yaml byte-identical.  The returned
    ``settings`` are read back **from disk** after the write, so a caller that
    prints them cannot print its own optimism.

    Any mapping, list and scalar outside ``smart_model_routing`` is preserved
    (the write is a re-serialisation of the raw file through ``save_config``,
    which keeps comments and env-var templates).

    ``config`` is for read-only callers (tests, previews): when it is passed the
    base block comes from *config* and **nothing is written**, exactly as with
    ``dry_run=True``.  The CLI and the gateway pass ``config=None``.
    """
    if not isinstance(changes, dict) or not changes:
        raise ValueError("update_routing expects a non-empty mapping of smart_model_routing keys")

    explicit_base = config is not None
    read_only = bool(dry_run) or explicit_base

    from hermes_cli import config as hermes_config

    path = None if explicit_base else hermes_config.get_config_path()
    if explicit_base:
        base_raw: Dict[str, Any] = {}
        base_block = _routing_block_from_disk(config)
    else:
        # Fail closed: an unreadable/unparseable file must not be replaced by a
        # lone routing block (the read-then-write collapse-to-empty trap).
        base_raw = hermes_config.require_readable_config_before_write(path)
        base_block = _routing_section_raw(base_raw)

    merged = dict(base_block)
    for key, value in changes.items():
        merged[str(key)] = value

    stale_unknown = {
        key for key in base_block
        if str(key) not in ROUTING_KEYS and str(key) not in ROUTING_LEGACY_KEYS}

    problems = [
        problem for problem in validate_routing(merged)
        if not any(problem.startswith(f"smart_model_routing.{key} is not a recognised")
                   for key in stale_unknown)]
    if problems:
        raise ValueError("invalid smart_model_routing: " + "; ".join(problems))

    warnings = [
        f"smart_model_routing.{key} is not a recognised routing option; left untouched"
        for key in sorted(stale_unknown)]

    changed = sorted(
        str(key) for key in changes if _jsonable(base_block.get(key)) != _jsonable(merged.get(key)))

    result: Dict[str, Any] = {
        "dry_run": read_only,
        "written": False,
        "changed": changed,
        "warnings": warnings,
        "settings": routing_settings({"smart_model_routing": merged}),
        "block": _jsonable(merged),
        "path": "",
    }
    if read_only:
        return result

    base_raw["smart_model_routing"] = merged
    hermes_config.save_config(base_raw)

    # Read the file back rather than trusting the object we just wrote: the
    # stripper, the env-ref template pass and the YAML round-trip all get a say.
    try:
        on_disk = hermes_config.read_user_config_raw(path)
    except Exception:  # pragma: no cover - defensive
        on_disk = {}
    if "smart_model_routing" not in on_disk:
        on_disk = hermes_config.read_raw_config()
    written = _routing_section_raw(on_disk)
    result.update({
        "written": True,
        "settings": routing_settings({"smart_model_routing": written}),
        "block": _jsonable(written),
        "path": str(path),
    })
    return result


def routing_preview(
    prompt: str,
    config: Any = None,
    *,
    attachments: Sequence[str] = (),
    conversation_tokens: int = 0,
    provider: str = "",
    current_model: str = "",
    candidates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """What routing would do for *prompt*, and why — one scoring pass.

    Reuses :func:`_preview_selection` (the same pass behind ``explain`` and
    ``status --json``) and :func:`explain` for the prose, so a preview can never
    disagree with the turn it describes.  Read-only and total: any failure is
    reported as ``skip_reason: "error"``, never raised.

    ``config`` defaults to the live config on disk.  A disabled block still
    previews the hypothetical choice, with ``skip_reason: "disabled"``.
    """
    source = config if config is not None else None
    cfg = _routing_block_from_disk(source)
    settings = routing_settings({"smart_model_routing": cfg})
    enabled = bool(settings["enabled"])

    doc: Dict[str, Any] = {
        "prompt": prompt or "",
        "enabled": enabled,
        "settings": settings,
        "request": {},
        "candidates": [],
        "models": [],
        "selected": "",
        "selected_provider": "",
        "reason": "",
        "skip_reason": "",
        "kept_current": False,
        "eligible_count": 0,
        "explanation": "",
    }
    try:
        full_config: Any = source if source is not None else _load_config_for_preview()
        selection, models = _preview_selection(
            full_config if isinstance(full_config, dict) else {},
            prompt or "",
            attachments=attachments,
            conversation_tokens=conversation_tokens,
            provider=provider,
            current_model=current_model,
            candidates=candidates,
        )
    except Exception as err:  # pragma: no cover - defensive: a preview never breaks a CLI
        doc["reason"] = f"routing preview failed: {err}"
        doc["skip_reason"] = "error"
        return doc

    profile = selection.profile
    decision = selection.decision()
    doc.update({
        "request": {
            "tier": profile.complexity.label,
            "est_input_tokens": profile.est_input_tokens,
            "est_output_tokens": profile.est_output_tokens,
            "needs_tools": profile.needs_tools,
            "needs_reasoning": profile.needs_reasoning,
            "reasons": list(profile.signals),
        },
        "candidates": [record.as_dict() for record in selection.records],
        "models": list(models),
        "eligible_count": selection.eligible_count,
        "kept_current": bool(selection.kept_current),
        "selected": decision.model if decision else "",
        "selected_provider": decision.provider if decision else "",
        "reason": decision.reason if decision else selection.reason,
        "skip_reason": selection.skip_reason or "",
        "explanation": explain(decision, profile),
    })
    if not enabled:
        # Same convention as ``routing_report``: the hypothetical choice is
        # shown, but the machine field says it would not be applied.
        doc["skip_reason"] = "disabled"
    return doc


def _load_config_for_preview() -> Dict[str, Any]:
    """The live config for previews (``{}`` when it cannot be read)."""
    try:
        from hermes_cli.config import load_config_readonly

        loaded = load_config_readonly()
        return loaded if isinstance(loaded, dict) else {}
    except Exception:  # pragma: no cover - defensive
        return {}


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

#: Request tiers at or above this rank demand a tiered model: a candidate that
#: DECLARES a lower ``tier_rank`` is ineligible rather than merely penalised.
#: An undeclared tier is unknown metadata (penalised, never a refusal), which
#: is what keeps an untouched config behaving exactly as before.
_TIER_FLOOR = int(Complexity.COMPLEX)

#: Flat score cost of one required capability whose metadata is unknown.
_UNKNOWN_CAPABILITY_PENALTY = 0.15


def required_capabilities(profile: TaskProfile) -> Tuple[str, ...]:
    """Capabilities the request cannot do without, in a stable order.

    Text output is required by every request but listed last, so an exclusion
    reason names the most specific gap first.
    """
    caps: List[str] = []
    if profile.needs_reasoning:
        caps.append("reasoning")
    if profile.needs_tools:
        caps.append("tools")
    for modality in profile.input_modalities:
        if modality == "text" or not modality:
            continue
        if modality == "image":
            caps.append("vision")
        elif modality == "audio":
            caps.append("audio")
        else:
            caps.append(f"input_{modality}")
    for modality in profile.output_modalities:
        if modality == "text" or not modality:
            continue
        caps.append(f"output_{modality}")
    if profile.needs_structured_output:
        caps.append("structured")
    caps.append("text_output")
    return tuple(caps)


def capability_state(candidate: ModelCandidate, capability: str) -> CapabilityState:
    """Tri-state answer for one required capability (spec §9.2)."""
    if capability == "text_output":
        return candidate.supports_output("text")
    if capability == "reasoning":
        return candidate.supports_reasoning()
    if capability == "tools":
        return candidate.supports_tools()
    if capability == "vision":
        return candidate.supports_input("image")
    if capability == "audio":
        return candidate.supports_input("audio")
    if capability == "structured":
        return candidate.supports_structured_output()
    if capability.startswith("input_"):
        return candidate.supports_input(capability[len("input_"):])
    if capability.startswith("output_"):
        return candidate.supports_output(capability[len("output_"):])
    return CapabilityState.UNKNOWN


def unknown_required_capabilities(
    candidate: ModelCandidate, profile: TaskProfile
) -> Tuple[str, ...]:
    """Required capabilities this candidate's metadata does not answer."""
    return tuple(
        cap
        for cap in required_capabilities(profile)
        if capability_state(candidate, cap) is CapabilityState.UNKNOWN
    )


def _exclusion_reason(
    candidate: ModelCandidate, profile: TaskProfile, cfg: RoutingConfig
) -> str:
    """Why ``candidate`` cannot serve this request (``""`` when it can).

    Hard constraints only: an ineligible model never wins on price, and the
    reason is reported verbatim in the CLI's ``candidates[].excluded_reason``.
    """
    if not candidate.enabled:
        return "not_enabled"
    if candidate.key in cfg.exclude or candidate.model in cfg.exclude:
        return "excluded"
    if candidate.is_deprecated():
        return "deprecated"

    # A pinned model wins over every other constraint except existence: the
    # user's explicit choice is authoritative (spec §10).
    if profile.pinned_model and candidate.model == profile.pinned_model:
        return ""

    # Context window: unknown is allowed (penalised later), too small is not.
    ctx = candidate.context_window()
    if ctx is not None and profile.required_context > ctx:
        return "context_window_too_small"

    # Output cap: a model that cannot emit the expected answer will truncate.
    cap = candidate.max_output()
    if cap is not None and profile.est_output_tokens > cap:
        return "max_output_too_small"

    # Tier floor: for complex/advanced requests a declared below-tier model is
    # ineligible, not merely expensive-looking.
    tier = candidate.tier_rank()
    if tier is not None and int(profile.complexity) >= _TIER_FLOOR and tier < int(profile.complexity):
        return "below_tier"

    # Capability requirements: unsupported disqualifies always; unknown
    # disqualifies only in strict mode (``allow_unknown_metadata: false``).
    for capability in required_capabilities(profile):
        state = capability_state(candidate, capability)
        if state is CapabilityState.UNSUPPORTED:
            return f"unsupported_{capability}"
        if state is CapabilityState.UNKNOWN and not cfg.allow_unknown_metadata:
            return f"unknown_{capability}"
    return ""


def filter_eligible(
    candidates: Iterable[ModelCandidate],
    profile: TaskProfile,
    config: Optional[RoutingConfig] = None,
) -> List[ModelCandidate]:
    """Apply hard constraints; only survivors may be ranked (spec §15)."""
    cfg = config or RoutingConfig()
    return [c for c in candidates if not _exclusion_reason(c, profile, cfg)]


# ---------------------------------------------------------------------------
# Cost + scoring
# ---------------------------------------------------------------------------

#: Tokens one image attachment is charged as.  Vision models bill an image as a
#: token block, not as a separate line item; models.dev publishes no per-image
#: price, so a fixed equivalent keeps images priced without inventing a rate.
#: 1600 is the upper end of the common 1024×1024 tiling on current vision APIs.
_IMAGE_TOKEN_EQUIVALENT = 1600

#: Most one retry can add to a request's cost.  A model that fails half the time
#: expects one retry; past that the candidate is unreliable enough that price is
#: not the interesting number.
_MAX_RETRY_FRACTION = 1.0


def expected_retry_fraction(reliability: Optional[float]) -> float:
    """Expected retries per successful request for a declaration, capped.

    ``reliability`` is the operator-declared success rate (the only source the
    engine has — nothing measures outcomes).  Unknown means ``0.0``: the cost
    model does not invent a retry charge for a model nobody has characterised.
    """
    if reliability is None:
        return 0.0
    try:
        p = float(reliability)
    except (TypeError, ValueError):
        return 0.0
    if p >= 1.0:
        return 0.0
    if p <= 0.0:
        return _MAX_RETRY_FRACTION
    return min(_MAX_RETRY_FRACTION, (1.0 - p) / p)


def estimate_cost_breakdown(
    candidate: ModelCandidate,
    profile: TaskProfile,
    *,
    charge: Optional[Tuple[Optional[float], Optional[float]]] = None,
    cache_charge: Optional[float] = None,
    reliability: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Expected USD cost for this request, term by term; ``None`` when unpriced.

    Terms (all in USD per million tokens, the catalog's normalised unit):

    - ``input`` — input tokens that are not cache hits, at ``cost_in``;
    - ``cached_input`` — cache-hit tokens at the cache-read price, falling back
      to ``cost_in`` when no discount is published;
    - ``output`` — the answer, at ``cost_out``;
    - ``reasoning`` — deliberation tokens at ``cost_out``, and only for models
      that actually report reasoning support (a model that cannot reason does
      not get billed for reasoning it will not emit);
    - ``modality`` — image attachments as token equivalents at ``cost_in``;
    - ``retries`` — the subtotal times the expected retry fraction derived from
      a declared reliability.

    ``charge`` overrides the candidate's own ``(cost_in, cost_out)`` — the
    selection path passes the pool median there so an unpriced model is never
    treated as free.  When a charge override is in play the cache rate defaults
    to that same input rate: an unpriced model cannot claim a discount.
    """
    if charge is not None:
        ci, co = charge
    else:
        ci, co = candidate.costs()
    if ci is None and co is None:
        return None
    ci = float(ci or 0.0)
    co = float(co or 0.0)

    cached_tokens = max(0, min(int(profile.est_cached_input_tokens or 0), int(profile.est_input_tokens or 0)))
    billable_input = max(0, int(profile.est_input_tokens or 0) - cached_tokens)

    own_in, own_out = candidate.costs()
    declared_cache = candidate.cache_cost()
    # Own/declared cache price applies whenever the candidate is being charged its
    # own input price (the normal case, and also charge=None).  A model priced at
    # the pool median has no published rate of its own to discount from.
    charged_own_prices = charge is None or own_in is not None or own_out is not None
    if cache_charge is not None:
        cache_rate = float(cache_charge)
    elif charged_own_prices and declared_cache is not None:
        cache_rate = declared_cache
    else:
        cache_rate = ci

    reasoning_tokens = int(profile.est_reasoning_tokens or 0)
    if reasoning_tokens and candidate.supports_reasoning() is not CapabilityState.SUPPORTED:
        # Only a model that reports reasoning support spends these tokens.
        reasoning_tokens = 0

    image_tokens = max(0, int(profile.image_inputs or 0)) * _IMAGE_TOKEN_EQUIVALENT

    input_cost = (billable_input / 1_000_000.0) * ci
    cached_cost = (cached_tokens / 1_000_000.0) * cache_rate
    output_cost = (int(profile.est_output_tokens or 0) / 1_000_000.0) * co
    reasoning_cost = (reasoning_tokens / 1_000_000.0) * co
    modality_cost = (image_tokens / 1_000_000.0) * ci

    subtotal = input_cost + cached_cost + output_cost + reasoning_cost + modality_cost
    retry_fraction = expected_retry_fraction(reliability)
    retry_cost = subtotal * retry_fraction

    return {
        "units": "usd_per_million_tokens",
        "input": round(input_cost, 12),
        "cached_input": round(cached_cost, 12),
        "output": round(output_cost, 12),
        "reasoning": round(reasoning_cost, 12),
        "modality": round(modality_cost, 12),
        "subtotal": round(subtotal, 12),
        "retries": round(retry_cost, 12),
        "expected_retries": round(retry_fraction, 6),
        "total": round(subtotal + retry_cost, 12),
        "tokens": {
            "billable_input": billable_input,
            "cached_input": cached_tokens,
            "output": int(profile.est_output_tokens or 0),
            "reasoning": reasoning_tokens,
            "image_equivalent": image_tokens,
        },
        "charged": {"cost_in": ci, "cost_out": co, "cost_cache_read": cache_rate},
        "policy": (
            "reasoning_at_output_rate; images_as_input_token_equivalents; "
            "cache_read_falls_back_to_input; retries_from_declared_reliability"
        ),
    }


def estimate_cost(
    candidate: ModelCandidate,
    profile: TaskProfile,
    *,
    charge: Optional[Tuple[Optional[float], Optional[float]]] = None,
    cache_charge: Optional[float] = None,
    reliability: Optional[float] = None,
) -> Optional[float]:
    """Expected USD cost for this request; ``None`` when pricing is unknown.

    Total of :func:`estimate_cost_breakdown`.  With no cache credit, no
    reasoning tokens, no images and no declared reliability the total is exactly
    the input-plus-output estimate it has always been.
    """
    breakdown = estimate_cost_breakdown(
        candidate, profile, charge=charge, cache_charge=cache_charge, reliability=reliability
    )
    return None if breakdown is None else breakdown["total"]


def _median_pair(candidates: Sequence[ModelCandidate]) -> Optional[Tuple[float, float]]:
    """Median known ``(cost_in, cost_out)`` across ``candidates``.

    Computed per rate so a pool that prices input but not output still yields a
    usable charge.  ``None`` when nothing in the pool is priced at all.
    """
    ins: List[float] = []
    outs: List[float] = []
    for cand in candidates:
        ci, co = cand.costs()
        if ci is not None:
            ins.append(float(ci))
        if co is not None:
            outs.append(float(co))
    if not ins and not outs:
        return None
    return (median(ins) if ins else 0.0, median(outs) if outs else 0.0)


def charge_for(
    candidate: ModelCandidate, median_pair: Optional[Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """The prices a candidate is charged at: its own, else the pool median."""
    ci, co = candidate.costs()
    if ci is not None or co is not None:
        return (ci or 0.0, co or 0.0)
    return median_pair


def pricing_for(
    profile: TaskProfile,
    candidates: Sequence[ModelCandidate],
    *,
    median_pair: Optional[Tuple[float, float]] = None,
    config: Optional[RoutingConfig] = None,
) -> Dict[str, Any]:
    """The ``pricing`` block of the shared report.

    ``median_fallback`` is the pair charged to unpriced models (never zero);
    ``cheapest_eligible`` is the estimated cost of the cheapest eligible
    candidate at its charged price.  ``median_pair`` lets the caller supply the
    same pair the scoring path used, so the report cannot describe a different
    charge than the one that decided the turn.  ``config`` supplies the declared
    reliabilities so the retry term matches the one ranking used.
    """
    cfg = config or RoutingConfig()
    if median_pair is None:
        median_pair = _median_pair(candidates)
    costs = [
        estimate_cost(
            c, profile,
            charge=charge_for(c, median_pair),
            reliability=_reliability_of(c, cfg),
        )
        for c in candidates
    ]
    known = [c for c in costs if c is not None]
    return {
        "cheapest_eligible": round(min(known), 9) if known else None,
        "median_fallback": (
            {"cost_in": median_pair[0], "cost_out": median_pair[1]}
            if median_pair is not None
            else None
        ),
        "policy": "unpriced_models_charged_the_median_known_price",
    }


def _reliability_of(candidate: ModelCandidate, cfg: RoutingConfig) -> Optional[float]:
    if candidate.reliability is not None:
        return candidate.reliability
    for key in (candidate.key, candidate.model):
        if key in cfg.reliability:
            return cfg.reliability[key]
    return None


def _capability_fit(candidate: ModelCandidate, profile: TaskProfile) -> float:
    """0..1.25 — can this model do the job, and is it the right *size* model?"""
    fit = 1.0

    # Reasoning: needed for complex work, wasted (not harmful) on simple work.
    if profile.needs_reasoning:
        if candidate.supports_reasoning() is CapabilityState.SUPPORTED:
            fit += 0.10
    else:
        # Prefer non-reasoning models for simple asks: reasoning tokens are
        # billed as output and add latency with no quality benefit.
        if candidate.supports_reasoning() is CapabilityState.SUPPORTED and profile.complexity <= Complexity.ROUTINE:
            fit -= 0.05

    # Unknown capability metadata is penalised FLAT per required capability
    # (never treated as a refusal): 0.15 each, per the shared contract.
    fit -= _UNKNOWN_CAPABILITY_PENALTY * len(unknown_required_capabilities(candidate, profile))

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
    current_provider: str = "",
    cost_ceiling: Optional[float] = None,
    charge: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> float:
    """Weighted score in roughly 0..1.2 (higher is better).

    Mandatory constraints are applied by :func:`filter_eligible`; this is the
    ranking step only.  Cost uses ``charge`` (own prices, else the pool median)
    and is expressed relative to ``cost_ceiling`` (the most expensive eligible
    candidate) so the term is stable across requests.
    """
    cfg = config or RoutingConfig()
    w = cfg.weights

    capability = _capability_fit(candidate, profile)

    rel = _reliability_of(candidate, cfg)
    cost = estimate_cost(candidate, profile, charge=charge, reliability=rel)
    if cost is None:
        cost_score = 0.35  # no price anywhere: conservative, never a free lunch
    elif not cfg.cost_optimization:
        cost_score = 1.0
    else:
        # With a single priced candidate there is nothing to compare against;
        # charging it a full penalty would make "unknown" look cheaper.
        ceiling = cost_ceiling if cost_ceiling and cost_ceiling > 0 else None
        cost_score = 1.0 - min(1.0, cost / ceiling) if ceiling else 1.0

    if profile.needs_reasoning and candidate.supports_reasoning() is not CapabilityState.SUPPORTED:
        # Spec §7 items 3/4: when the task requires reasoning, a model whose
        # reasoning support is unconfirmed must not become the pick merely
        # because its price is lower — that is the "under-serve complex work to
        # save money" failure, and honestly-priced reasoning tokens would
        # otherwise hand it the whole cost term.
        cost_score = 0.0

    # Cold start: unknown reliability is 0.5, never 1.0 (spec §16.3).
    rel_score = 0.5 if rel is None else max(0.0, min(1.0, rel))

    context = _context_fit(candidate, profile)

    continuity = 1.0 if (current_model and candidate.model == current_model
                         and (not current_provider
                              or candidate.provider.strip().lower() == current_provider.strip().lower())) else 0.0

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
# Selection — ONE ranking path for the turn hook, `explain` and `status --json`
# ---------------------------------------------------------------------------

#: Every ``skip_reason`` value the shared contract allows (``""`` = a switch).
SKIP_REASONS = (
    "",
    "disabled",
    "no_candidates",
    "no_eligible_model",
    "already_current",
    "within_stability_threshold",
    # Contract C10 lists this reason; nothing emits it yet (the MoA path does not
    # consult the router), but the vocabulary must accept it or a future emitter
    # would be rejected for a reason the contract already promises.
    "moa_active",
    "error",
)


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
    skip_reason: str = ""
    kept_current: bool = False
    #: Input/output modalities the decision was made for (contract C9). Without
    #: this record C9b cannot be tested: a later turn (an image-edit follow-up,
    #: say) has no way to know what the previous decision actually promised.
    input_modalities: Tuple[str, ...] = ("text",)
    output_modalities: Tuple[str, ...] = ("text",)

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}" if self.provider else self.model


@dataclass
class CandidateReport:
    """One row of the shared report's ``candidates[]`` array."""

    model: str
    provider: str
    eligible: bool
    score: Optional[float]
    tier: str
    cost_in: Optional[float]
    cost_out: Optional[float]
    priced: bool
    charged_as: str
    unknown_capabilities: Tuple[str, ...]
    excluded_reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "eligible": self.eligible,
            "score": self.score,
            "tier": self.tier,
            "cost_in": self.cost_in,
            "cost_out": self.cost_out,
            "priced": self.priced,
            "charged_as": self.charged_as,
            "unknown_capabilities": list(self.unknown_capabilities),
            "excluded_reason": self.excluded_reason,
        }


@dataclass
class Selection:
    """A complete selection pass: the choice plus every input to it."""

    profile: TaskProfile
    records: List[CandidateReport] = field(default_factory=list)
    eligible_count: int = 0
    best_key: str = ""
    model: str = ""
    provider: str = ""
    score: float = 0.0
    est_cost: Optional[float] = None
    reason: str = ""
    skip_reason: str = ""
    kept_current: bool = False
    current: str = ""
    pinned: bool = False
    considered: List[Tuple[str, float]] = field(default_factory=list)
    pricing: Dict[str, Any] = field(default_factory=dict)

    def decision(self) -> Optional[Decision]:
        """The :class:`Decision` for this pass (``None`` when nothing was chosen)."""
        if not self.model:
            return None
        return Decision(
            model=self.model,
            provider=self.provider,
            score=round(self.score, 4),
            est_cost=self.est_cost,
            reason=self.reason,
            switched=bool(self.current) and self.model != self.current,
            complexity=self.profile.complexity.label,
            eligible_count=self.eligible_count,
            considered=list(self.considered),
            pinned=self.pinned,
            skip_reason=self.skip_reason,
            kept_current=self.kept_current,
            input_modalities=tuple(self.profile.input_modalities),
            output_modalities=tuple(self.profile.output_modalities),
        )


def _capability_gap(
    incumbent: ModelCandidate, best: ModelCandidate, profile: TaskProfile
) -> bool:
    """True when ``best`` closes a capability gap the incumbent has.

    A gap is a hard reason to change models mid-session; a marginal score win
    is not (spec §16.2).
    """
    if (
        profile.needs_reasoning
        and incumbent.supports_reasoning() is not CapabilityState.SUPPORTED
        and best.supports_reasoning() is CapabilityState.SUPPORTED
    ):
        return True
    return bool(
        profile.needs_tools
        and incumbent.supports_tools() is CapabilityState.UNSUPPORTED
        and best.supports_tools() is CapabilityState.SUPPORTED
    )


def _tier_label(candidate: ModelCandidate) -> str:
    tier = candidate.tier_rank()
    if tier is None:
        return "unknown"
    for level in Complexity:
        if int(level) == tier:
            return level.label
    return str(tier)


def rank_candidates(
    profile: TaskProfile,
    candidates: Sequence[ModelCandidate],
    config: Optional[RoutingConfig] = None,
    *,
    current_model: str = "",
    current_provider: str = "",
    mid_session: bool = False,
) -> Selection:
    """Rank ``candidates`` for ``profile`` and pick one.  THE scoring path.

    Order of business, matching the shared contract:

    1. eligibility before ranking (an ineligible model never wins on price);
    2. one score per eligible model, priced by the pool median when a model
       carries no price of its own;
    3. hysteresis — the incumbent is kept unless it is ineligible or a rival
       beats it by MORE than ``stability_threshold``;
    4. ``mid_session`` additionally keeps an eligible incumbent that merely
       scores higher (a switch would drop the provider's prompt cache).

    The turn hook, ``preview_decision`` and the CLI report all call this
    function, so a report can never disagree with behaviour for the same
    inputs.
    """
    cfg = config or RoutingConfig()
    current = str(current_model or "").strip()
    selection = Selection(profile=profile, current=current)
    candidates = list(candidates or [])

    # ---- pinned model -----------------------------------------------------
    if profile.pinned_model:
        pinned = [c for c in candidates if c.model == profile.pinned_model and c.enabled]
        if pinned:
            cand = pinned[0]
            selection.model, selection.provider = cand.model, cand.provider
            selection.score, selection.pinned = 1.0, True
            selection.est_cost = estimate_cost(
                cand, profile, reliability=_reliability_of(cand, cfg)
            )
            selection.reason = "user pinned model"
            selection.eligible_count = 1
            selection.considered = [(cand.key, 1.0)]
            selection.best_key = cand.key
            selection.records = [
                _record(
                    c, profile, cfg, None,
                    1.0 if c is cand else None,
                    _exclusion_reason(c, profile, cfg),
                )
                for c in candidates
            ]
            return selection
        # Pinned but unknown: say so rather than silently choosing another.
        selection.reason = f"pinned model {profile.pinned_model!r} is not an enabled candidate"
        selection.skip_reason = "no_eligible_model"
        selection.records = [
            _record(c, profile, cfg, None, None, _exclusion_reason(c, profile, cfg))
            for c in candidates
        ]
        return selection

    if not candidates:
        selection.reason = "no candidate models configured"
        selection.skip_reason = "no_candidates"
        selection.pricing = pricing_for(profile, [])
        return selection

    # ---- eligibility ------------------------------------------------------
    eligible = filter_eligible(candidates, profile, cfg)
    selection.eligible_count = len(eligible)
    median_pair = _median_pair(eligible) or _median_pair(candidates)
    selection.pricing = pricing_for(profile, eligible, median_pair=median_pair, config=cfg)

    if not eligible:
        selection.reason = (
            "no eligible model: every candidate failed a hard constraint "
            "(context, modality, tools, tier, or deprecation)"
        )
        selection.skip_reason = "no_eligible_model"
        selection.records = [
            _record(c, profile, cfg, median_pair, None, _exclusion_reason(c, profile, cfg))
            for c in candidates
        ]
        return selection

    costs = [
        cost
        for cost in (
            estimate_cost(
                c, profile,
                charge=charge_for(c, median_pair),
                reliability=_reliability_of(c, cfg),
            )
            for c in eligible
        )
        if cost is not None
    ]
    ceiling = max(costs) if len(costs) > 1 else None

    scored = [
        (
            cand,
            score_candidate(
                cand, profile, cfg,
                current_model=current,
                current_provider=current_provider,
                cost_ceiling=ceiling,
                charge=charge_for(cand, median_pair),
            ),
        )
        for cand in eligible
    ]
    scored.sort(key=lambda item: (-item[1], item[0].key))

    best, best_score = scored[0]
    selection.best_key = best.key
    selection.considered = [(c.key, round(s, 4)) for c, s in scored[:5]]

    by_key = {(c.provider, c.model): s for c, s in scored}
    selection.records = []
    for cand in candidates:
        score = by_key.get((cand.provider, cand.model))
        selection.records.append(
            _record(
                cand, profile, cfg, median_pair,
                score,
                "" if score is not None else _exclusion_reason(cand, profile, cfg),
            )
        )

    # The incumbent is identity-based, not name-based: one model id can exist on two
    # providers, and a name-only match lets the "kept current" path report a provider
    # the session is not actually on. Falls back to the name only when the caller has
    # no provider to compare with (configs that omit it keep working).
    incumbent = None
    if current:
        named = [c for c in eligible if c.model == current]
        if current_provider:
            incumbent = next(
                (c for c in named
                 if c.provider.strip().lower() == current_provider.strip().lower()),
                None,
            )
        if incumbent is None:
            incumbent = named[0] if named else None

    def _keep(cand: ModelCandidate, reason: str, skip_reason: str) -> Selection:
        selection.model, selection.provider = cand.model, cand.provider
        selection.score = by_key.get((cand.provider, cand.model), 0.0)
        selection.est_cost = estimate_cost(
            cand, profile,
            charge=charge_for(cand, median_pair),
            reliability=_reliability_of(cand, cfg),
        )
        selection.reason = reason
        selection.skip_reason = skip_reason
        selection.kept_current = True
        return selection

    if incumbent is not None and incumbent is best:
        return _keep(
            incumbent, "current model remains the best eligible choice", "already_current"
        )

    if incumbent is not None and incumbent is not best:
        inc_score = by_key.get((incumbent.provider, incumbent.model), 0.0)
        gain = best_score - inc_score
        best_cost = estimate_cost(
            best, profile,
            charge=charge_for(best, median_pair),
            reliability=_reliability_of(best, cfg),
        )
        inc_cost = estimate_cost(
            incumbent, profile,
            charge=charge_for(incumbent, median_pair),
            reliability=_reliability_of(incumbent, cfg),
        )
        downgrade = best_cost is not None and inc_cost is not None and best_cost < inc_cost

        # Hysteresis: ties and borderline wins keep the current model.  A
        # capability gap is not marginal — the incumbent cannot do the job.
        if gain <= cfg.stability_threshold and not _capability_gap(incumbent, best, profile):
            return _keep(
                incumbent,
                f"kept current model: best alternative {best.key} led by only "
                f"{gain:.3f} < stability {cfg.stability_threshold:.3f}",
                "within_stability_threshold",
            )
        if mid_session:
            # A capable incumbent that merely scores lower is still serving the
            # session; only an *ineligible* incumbent forces a mid-session
            # change, because switching would invalidate the cached prompt
            # prefix.  (A fresh turn — what `explain` / `status` describe — has
            # no cache to protect and may switch on a capability gap.)
            return _keep(
                incumbent,
                f"kept current model: {best.key} scored higher but a "
                "mid-session switch requires a capability gap or ineligible "
                "incumbent (prompt-cache stability)",
                "within_stability_threshold",
            )
        if downgrade and not cfg.switch_on_cost:
            return _keep(
                incumbent,
                f"kept current model: {best.key} is cheaper (score +{gain:.3f}) but a "
                "cost-only downgrade mid-session is disabled (prompt-cache stability)",
                "within_stability_threshold",
            )

    selection.model, selection.provider = best.model, best.provider
    selection.score = best_score
    selection.est_cost = estimate_cost(
        best, profile,
        charge=charge_for(best, median_pair),
        reliability=_reliability_of(best, cfg),
    )
    selection.reason = (
        f"selected over {len(eligible) - 1} other eligible model(s) for "
        f"{profile.complexity.label} task"
    )
    selection.skip_reason = ""
    selection.kept_current = False
    return selection


def _record(
    candidate: ModelCandidate,
    profile: TaskProfile,
    cfg: RoutingConfig,
    median_pair: Optional[Tuple[float, float]],
    score: Optional[float],
    excluded_reason: str,
) -> CandidateReport:
    """One report row, priced at what the model would actually be charged."""
    own_in, own_out = candidate.costs()
    charged = charge_for(candidate, median_pair)
    priced = own_in is not None or own_out is not None
    return CandidateReport(
        model=candidate.model,
        provider=candidate.provider,
        eligible=not excluded_reason,
        score=round(score, 4) if score is not None else None,
        tier=_tier_label(candidate),
        cost_in=round(charged[0], 6) if charged else None,
        cost_out=round(charged[1], 6) if charged else None,
        priced=priced,
        charged_as="own" if priced else ("median" if charged else "unknown"),
        unknown_capabilities=unknown_required_capabilities(candidate, profile),
        excluded_reason=excluded_reason,
    )


def select_model(
    profile: TaskProfile,
    candidates: Sequence[ModelCandidate],
    config: Optional[RoutingConfig] = None,
    *,
    current_model: str = "",
    current_provider: str = "",
    mid_session: bool = False,
) -> Optional[Decision]:
    """Pick the best eligible model, or ``None`` when nothing qualifies.

    Thin wrapper over :func:`rank_candidates` — the single scoring path.
    ``current_model`` drives hysteresis: the incumbent is kept unless a rival
    beats it by more than ``stability_threshold`` (or the incumbent is
    ineligible).
    """
    selection = rank_candidates(
        profile, candidates, config, current_model=current_model,
        current_provider=current_provider, mid_session=mid_session,
    )
    return selection.decision()


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

        # Cache-only: selection must never touch the network (contract), so a
        # stale or absent catalog simply means there is nothing to widen to.
        listed = models_dev.list_provider_models(provider, allow_network=False)
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

    ``preferred_models_only`` defaults to **false** (shared contract C6). A provider
    that declares exactly one model is still treated as ``"preferred"`` — there is no
    choice to make and the catalog scan would be wasted — unless the operator wrote
    the key explicitly, which ``RoutingConfig.preferred_models_only_explicit`` records.

    ``preferred_models_only: true`` is enforced here and nowhere else: the pool is
    exactly the models the operator listed for the provider (plus the model currently
    in use, which is the incumbent, not a new choice), so routing can never select a
    model that is not on that list.

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
    # C6 of the shared contract (docs/routing-contract.md): widening is the default, but a
    # provider that lists exactly ONE model has nothing to choose between — consulting the
    # catalog there would add ranking work to a config that asked for a single model. That
    # stays a no-op unless the operator explicitly sets preferred_models_only.
    if cfg.preferred_models_only or (len(preferred) <= 1 and not cfg.preferred_models_only_explicit):
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


def _spec_provider_and_model(
    config: Any, spec: ModelSpec, *, default_provider: str
) -> Tuple[str, str]:
    """``(provider, model)`` for one declared spec.

    An explicit ``provider`` wins; a ``provider/model`` name is split only when
    the prefix names a provider this install actually has configured (so an
    aggregator's ``vendor/model`` id is never mistaken for a provider).
    """
    provider = (spec.provider or "").strip()
    name = spec.name.strip()
    if provider:
        head, sep, rest = name.partition("/")
        if sep and rest.strip() and head.strip().lower() == provider.lower():
            name = rest.strip()
        return provider, name
    return _split_provider_model(config, name, default_provider=default_provider)


def candidates_from_specs(
    config: Any,
    specs: Sequence[ModelSpec],
    *,
    default_provider: str = "",
    enabled: bool = True,
) -> List[ModelCandidate]:
    """Build candidates from declared specs, metadata overlaid on the catalog.

    One ``candidates_for`` call per provider keeps the cached-catalog contract
    (and the existing test seam) intact.  A model the catalog does not know is
    still a candidate: its ``info`` is ``None``, which reads as UNKNOWN
    metadata, never as a refusal.
    """
    groups: List[Tuple[str, List[Tuple[str, ModelSpec]]]] = []
    index: Dict[str, int] = {}
    for spec in specs:
        provider, model = _spec_provider_and_model(
            config, spec, default_provider=default_provider
        )
        if not model:
            continue
        if provider not in index:
            index[provider] = len(groups)
            groups.append((provider, []))
        groups[index[provider]][1].append((model, spec))

    out: List[ModelCandidate] = []
    for provider, items in groups:
        built = {
            cand.model: cand
            for cand in candidates_for(provider, [m for m, _ in items], enabled=enabled)
        }
        for model, spec in items:
            cand = built.get(model) or ModelCandidate(
                provider=provider, model=model, info=None, enabled=enabled
            )
            cand.spec = spec
            out.append(cand)
    return out


def _declared_pool_scope(config: Any) -> str:
    """The pool scope of the declared ``models``/``candidates`` list.

    The list's *shape* is not what decides whether routing may look at the
    provider catalog — ``preferred_models_only`` is.  Reporting ``"models"``
    here (the key the list came from) used to slip past every ``== "preferred"``
    guard, so a strict config still widened and the CLI preview, which only
    widens on ``"preferred+catalog"``, disagreed with the runtime.
    """
    section = config.get("smart_model_routing") if isinstance(config, dict) else None
    cfg = RoutingConfig.from_dict(section)
    return "preferred" if cfg.preferred_models_only else "preferred+catalog"


def declared_model_list(
    config: Any, provider: str, *, current_model: str = ""
) -> Tuple[List[str], str, str]:
    """The declared model ids, with NO catalog read.

    ``(models, candidate_source, pool_scope)``.  Cheap enough for the per-turn
    hot path: a single-model install must not pay for any metadata lookups.
    """
    current = str(current_model or "").strip()
    specs, source = specs_from_config(config)
    if specs:
        names = [spec.name.strip() for spec in specs if spec.name.strip()]
        if current and current not in names:
            names.insert(0, current)
        return names, source, _declared_pool_scope(config)
    models, pool_scope = routing_pool(config, provider, current_model=current)
    return models, "preferred", pool_scope


def resolve_candidates(
    config: Any,
    provider: str,
    *,
    current_model: str = "",
    explicit_models: Optional[Sequence[str]] = None,
) -> Tuple[List[ModelCandidate], str, str]:
    """The candidate set a turn/CLI report works from.

    Returns ``(candidates, candidate_source, pool_scope)``.

    * ``explicit_models`` (the CLI's preview override) is the whole set;
    * ``smart_model_routing.models`` — or its deprecated ``candidates``
      equivalent — is used verbatim when present;
    * otherwise the provider's configured model list is used, exactly as
      before, with ``preferred_models_only`` deciding whether the catalog may
      be consulted as a fallback.

    The incumbent is always represented (prepended, never duplicated) so
    hysteresis can compare against it even when it is not on the list.
    """
    current = str(current_model or "").strip()

    if explicit_models is not None:
        specs = [
            spec
            for spec in (ModelSpec.from_entry(item) for item in explicit_models)
            if spec
        ]
        candidates = candidates_from_specs(config, specs, default_provider=provider)
        return candidates, "explicit", "explicit"

    specs, source = specs_from_config(config)
    if specs:
        candidates = candidates_from_specs(config, specs, default_provider=provider)
        if current and not any(c.model == current for c in candidates):
            incumbent = candidates_for(provider, [current])
            candidates.insert(
                0,
                incumbent[0]
                if incumbent
                else ModelCandidate(provider=provider, model=current, info=None),
            )
        return candidates, source, _declared_pool_scope(config)

    models, pool_scope = routing_pool(config, provider, current_model=current)
    return _candidates_from_pool(config, provider, models), "preferred", pool_scope


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


#: Agent attribute carrying the reason the last turn was not routed. Read by the
#: turn log and by anything that wants to explain a turn that kept its model.
ROUTING_SKIP_ATTR = "_last_routing_skip_reason"


def _record_routing_skip(agent: Any, reason: str) -> None:
    """Make a routing skip visible (contract C10). Never raises."""
    try:
        setattr(agent, ROUTING_SKIP_ATTR, reason)
    except Exception:  # pragma: no cover - defensive: exotic agent objects
        pass
    _log.debug("model routing skipped: %s", reason)


def _conversation_token_estimate(messages: Any, agent: Any = None) -> int:
    """Rough token count for the turn's request, or ``0``.

    Used only after routing is confirmed enabled, so a disabled install never
    pays for it.  Includes the cached system prompt and the tool schemas when
    the agent has them: 50+ tools add 20-30k tokens on their own, which is
    exactly the difference between a model that fits and one that does not.
    Never raises — a malformed history degrades to "unknown size".
    """
    if not messages:
        return 0
    try:
        from agent.model_metadata import (
            estimate_messages_tokens_rough,
            estimate_request_tokens_rough,
        )

        system_prompt = str(getattr(agent, "_cached_system_prompt", "") or "")
        tools = getattr(agent, "tools", None) if agent is not None else None
        try:
            if system_prompt or tools:
                return int(
                    estimate_request_tokens_rough(
                        list(messages), system_prompt=system_prompt, tools=tools or None
                    )
                )
            return int(estimate_messages_tokens_rough(list(messages)))
        except TypeError:  # pragma: no cover - (messages)-only test seams
            return int(estimate_messages_tokens_rough(messages))
    except Exception:  # pragma: no cover - defensive
        return 0


def _cached_prefix_tokens(agent: Any) -> int:
    """Tokens of the stable prompt prefix a provider can serve from its cache.

    Only the system prompt is byte-stable across turns; conversation history
    grows and is not guaranteed identical, so it is never credited.  ``0`` when
    there is no cached prompt — a cold turn pays full input price.
    """
    if agent is None:
        return 0
    text = str(getattr(agent, "_cached_system_prompt", "") or "")
    return _estimate_tokens(text) if text else 0


def route_turn(
    agent: Any,
    user_message: str,
    *,
    attachments: Sequence[str] = (),
    conversation_tokens: int = 0,
    conversation_history: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    moa_active: bool = False,
) -> Optional[Decision]:
    """Choose the model for one turn and apply it, or return ``None``.

    Returns the :class:`Decision` (so callers can log/explain it) or ``None``
    when routing is off, the model is user-pinned, there is nothing to choose
    between, or no candidate qualifies.

    ``conversation_tokens`` is the size the caller already knows; when it is 0
    and ``conversation_history`` is given, the size is estimated here (a model
    whose context window cannot hold the live conversation must be excluded
    before ranking, not after it fails).

    ``moa_active`` names the caller's mixture-of-agents state. The skip has to be
    visible (contract C10: every skip is explainable), so it is recorded on the
    agent rather than left implicit at the call site.
    """
    if moa_active:
        # C10: a skip nobody can see is indistinguishable from a bug. Record the
        # reason where a later reader (and the turn log) can find it, then leave
        # the model alone — scoring a replacement would dismantle the fan-out.
        _record_routing_skip(agent, "moa_active")
        return None
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

        if conversation_tokens <= 0 and conversation_history:
            # Context fit is a hard gate (C3): a model whose window cannot hold
            # the live conversation must be excluded, not selected and failed.
            conversation_tokens = _conversation_token_estimate(conversation_history, agent)

        cfg = RoutingConfig.from_dict((config or {}).get("smart_model_routing") or {})
        provider = str(getattr(agent, "provider", "") or "").strip()
        current = str(getattr(agent, "model", "") or "").strip()
        models, _source, pool_scope = declared_model_list(
            config, provider, current_model=current
        )
        if len(models) <= 1 and pool_scope == "preferred":
            return None  # single-model config: no routing overhead (spec §4.1)

        candidates, _source, pool_scope = resolve_candidates(
            config, provider, current_model=current
        )
        models = [c.model for c in candidates]

        turn = int(getattr(agent, _ROUTER_TURN_ATTR, 0) or 0)
        profile = classify_request(
            user_message or "",
            attachments=attachments,
            conversation_tokens=conversation_tokens,
            # Prompt-cache credit only from the second routed turn: turn 0 is the
            # cold request that populates the cache, so it pays full input price.
            cached_input_tokens=_cached_prefix_tokens(agent) if turn > 0 else 0,
            tool_available=bool(getattr(agent, "tools", None)),
        )
        # A session running on a non-default model is a deliberate choice (a
        # gateway /model override, a --model flag, a restored session), so it
        # gets the same protection as a mid-session incumbent.
        default_model = _configured_default_model(config, provider)
        mid_session = turn > 0 or bool(default_model and current != default_model)
        if turn == 0 and not cfg.switch_on_cost:
            # First routed turn of a session: nothing is cached yet, so a
            # cost-driven initial selection is not a mid-session downgrade.
            # ``switch_on_cost`` guards the *mid-session* case only.
            cfg = replace(cfg, switch_on_cost=True)

        # ONE ranking path: the same function `explain` and `status --json` use.
        selection = rank_candidates(
            profile, candidates, cfg, current_model=current,
            current_provider=provider, mid_session=mid_session,
        )

        if selection.skip_reason == "no_eligible_model" and pool_scope == "preferred+catalog":
            # Nothing on the operator's preferred list can serve this request.
            # Only now — and only because preferred_models_only is false — is
            # the provider catalog consulted, so a tool/modality gap is not
            # answered by silently keeping an incapable model.
            widened = widen_pool(provider, models)
            if len(widened) > len(models):
                candidates = _candidates_from_pool(config, provider, widened)
                selection = rank_candidates(
                    profile, candidates, cfg, current_model=current,
                    current_provider=provider, mid_session=mid_session,
                )
                if selection.model and selection.model not in models:
                    selection.reason = (
                        selection.reason + "; no preferred model was eligible, "
                        "selected from the provider catalog (preferred_models_only=false)"
                    )
                    _log.info(
                        "routing: widened beyond the preferred list to %s (%s)",
                        f"{selection.provider}/{selection.model}", selection.reason,
                    )

        setattr(agent, _ROUTER_TURN_ATTR, turn + 1)
        decision = selection.decision()
        if decision is None:
            _log.info(
                "routing: no eligible model among %d candidate(s); keeping %s",
                len(models), current or "(unset)",
            )
            return None

        same_provider = (not decision.provider or not provider
                         or decision.provider.strip().lower() == provider.strip().lower())
        if decision.model == current and same_provider:
            _log.info(
                "routing: kept %s (%s) for %s task",
                decision.key, decision.reason, decision.complexity,
            )
            setattr(agent, _ROUTER_LAST_ATTR, decision)
            return decision

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

def _report_provider(config: Any, provider: str = "") -> str:
    """The provider a report resolves candidates against."""
    prov = str(provider or "").strip()
    if not prov and isinstance(config, dict):
        model_section = config.get("model")
        if isinstance(model_section, dict):
            prov = str(model_section.get("provider") or "").strip()
    return prov


def _preview_selection(
    config: Any,
    prompt: str,
    *,
    attachments: Sequence[str] = (),
    conversation_tokens: int = 0,
    provider: str = "",
    current_model: str = "",
    candidates: Optional[Sequence[str]] = None,
) -> Tuple[Selection, List[str]]:
    """The shared pass behind `explain` and `status --json`.

    Pure: no agent, no network, no config mutation.  Returns the selection and
    the model list it was made from.  A preview is a *fresh session's first
    turn* — nothing is prompt-cached yet — so it is never ``mid_session``.
    """
    cfg = RoutingConfig.from_dict(
        config.get("smart_model_routing") if isinstance(config, dict) else None
    )
    prov = _report_provider(config, provider)

    current = str(current_model or "").strip()
    if not current:
        current = _configured_default_model(config, prov)

    cands, _source, pool_scope = resolve_candidates(
        config, prov, current_model=current, explicit_models=candidates
    )
    models = [c.model for c in cands]

    profile = classify_request(
        prompt or "",
        attachments=attachments,
        conversation_tokens=conversation_tokens,
    )
    preview_cfg = cfg
    if not cfg.switch_on_cost:
        # Nothing is cached on a fresh session, so a cost-driven choice is not
        # a mid-session downgrade.
        preview_cfg = replace(cfg, switch_on_cost=True)

    selection = rank_candidates(profile, cands, preview_cfg, current_model=current)
    if selection.skip_reason == "no_eligible_model" and pool_scope == "preferred+catalog":
        widened = widen_pool(prov, models)
        if len(widened) > len(models):
            cands = _candidates_from_pool(config, prov, widened)
            selection = rank_candidates(profile, cands, preview_cfg, current_model=current)
            if selection.model and selection.model not in models:
                models = widened
                selection.reason = (
                    selection.reason + "; no preferred model was eligible, "
                    "selected from the provider catalog (preferred_models_only=false)"
                )
    return selection, models


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

    Returns ``(decision, profile, candidate_models)``.  ``decision`` is
    ``None`` when routing is off, when fewer than two models are in play, or
    when no candidate satisfies the request's hard constraints.
    """
    prov = _report_provider(config, provider)
    if candidates is not None:
        names = [str(m).strip() for m in candidates if str(m).strip()]
        scope = "explicit"
    else:
        names, _source, scope = declared_model_list(
            config, prov, current_model=str(current_model or "").strip()
        )
    profile = classify_request(
        prompt or "",
        attachments=attachments,
        conversation_tokens=conversation_tokens,
    )
    if len(names) <= 1 and scope != "preferred+catalog":
        # Nothing to route: say so instead of inventing a choice.
        return None, profile, names

    selection, models = _preview_selection(
        config,
        prompt,
        attachments=attachments,
        conversation_tokens=conversation_tokens,
        provider=provider,
        current_model=current_model,
        candidates=candidates,
    )
    return selection.decision(), selection.profile, models


def routing_report(
    config: Any,
    prompt: str = "",
    *,
    attachments: Sequence[str] = (),
    conversation_tokens: int = 0,
    provider: str = "",
    current_model: str = "",
    candidates: Optional[Sequence[str]] = None,
    moa_active: bool = False,
) -> Dict[str, Any]:
    """The ``hermes model routing status --json`` document.

    Exactly the shared key set::

        enabled, settings, candidate_source, request, pricing, selected,
        selected_provider, reason, skip_reason, current, kept_current,
        eligible_count, candidates[]

    Read-only and total: every failure inside selection becomes
    ``skip_reason: "error"`` with the reason attached, so the CLI can still
    exit 0 (only an unreadable config is a non-zero exit, and that is decided
    by the caller).  A disabled section computes the hypothetical selection and
    reports ``skip_reason: "disabled"``.
    """
    cfg = RoutingConfig.from_dict(
        config.get("smart_model_routing") if isinstance(config, dict) else None
    )
    enabled = is_enabled(config)

    try:
        selection, _models = _preview_selection(
            config,
            prompt,
            attachments=attachments,
            conversation_tokens=conversation_tokens,
            provider=provider,
            current_model=current_model,
            candidates=candidates,
        )
        _specs, source = specs_from_config(config)
        if candidates is not None:
            source = "explicit"
        elif not source:
            source = "preferred"
        profile = selection.profile
        records = selection.records
        pricing = selection.pricing or pricing_for(profile, [])
        selected = selection.model
        selected_provider = selection.provider if selected else ""
        reason = selection.reason
        skip_reason = selection.skip_reason or ""
        current = selection.current
        kept_current = bool(selection.kept_current)
        eligible_count = selection.eligible_count
    except Exception as err:  # pragma: no cover - defensive: report never breaks
        profile = TaskProfile()
        records = []
        pricing = pricing_for(profile, [])
        selected = ""
        selected_provider = ""
        reason = f"routing report failed: {err}"
        skip_reason = "error"
        current = str(current_model or "").strip()
        kept_current = False
        eligible_count = 0
        _specs, source = specs_from_config(config)

    if not enabled:
        # Disabled is a strict no-op for execution, but the report still says
        # what routing WOULD do — with "disabled" as the skip reason.
        skip_reason = "disabled"

    if moa_active:
        # C10: a Mixture-of-Agents turn is skipped before any candidate can be
        # chosen. Say so in the machine surface instead of leaving the reader to
        # infer it from a decision that never happened. The candidate rows stay:
        # they are the answer to "what would routing have chosen", and hiding them
        # would make the skip look like a failure.
        skip_reason = "moa_active"
        selected = None
        selected_provider = ""
        kept_current = True
        reason = "mixture-of-agents is active this turn; model routing is skipped"

    return {
        "enabled": enabled,
        "settings": cfg.settings(),
        "candidate_source": source,
        "request": {
            "tier": profile.complexity.label,
            "est_input_tokens": profile.est_input_tokens,
            "est_output_tokens": profile.est_output_tokens,
            "needs_tools": profile.needs_tools,
            "needs_vision": "image" in profile.input_modalities,
            "needs_audio": "audio" in profile.input_modalities,
            "reasons": list(profile.signals),
        },
        "pricing": pricing,
        "selected": selected,
        "selected_provider": selected_provider,
        "reason": reason,
        "skip_reason": skip_reason,
        "current": current,
        "kept_current": kept_current,
        "eligible_count": eligible_count,
        "candidates": [record.as_dict() for record in records],
    }
