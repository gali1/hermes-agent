"""One place that answers "can this model do X here, today?" — and says how it knows.

Four states, deliberately distinct because they lead to different decisions:

``native``       the model/provider accepts the modality itself (catalog metadata, overlaid by any
                 operator declaration in ``smart_model_routing.models[]``).
``hermes``       the model cannot take it, but *Hermes* can convert it into something it can:
                 PDF text extraction, speech-to-text, the image data-URL path.  A model without
                 native PDF support is still perfectly usable for a PDF - calling that
                 "unsupported" would be wrong about the system, not about the model.
``unknown``      nobody declared it and the catalog is silent.  Eligible only when the routing
                 config allows unknown metadata, and reported as unknown so a caller can say so
                 instead of guessing.
``unavailable``  the model was refused by the provider (retired/forbidden) and that refusal is
                 remembered for a while, or the local facility does not exist (no OCR binary, no
                 STT backend).

Native and Hermes capability must never be conflated in a report: the first is what the model can
do, the second is what this installation can do *around* the model.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

NATIVE = "native"
HERMES = "hermes"
UNKNOWN = "unknown"
UNAVAILABLE = "unavailable"

_STATE_RANK = {NATIVE: 3, HERMES: 2, UNKNOWN: 1, UNAVAILABLE: 0}


@dataclass(frozen=True)
class Capability:
    name: str
    state: str
    source: str
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.state in (NATIVE, HERMES)

    def as_dict(self) -> Dict[str, str]:
        return {"capability": self.name, "state": self.state,
                "source": self.source, "detail": self.detail}


# --------------------------------------------------------------------------- local facilities


def _binary(name: str) -> bool:
    return shutil.which(name) is not None


def local_facilities() -> Dict[str, Capability]:
    """What THIS installation can convert, independent of any model."""
    facilities: Dict[str, Capability] = {
        "pdf_text": Capability(
            "pdf_text", HERMES, "tools.read_extract.extract_document_text",
            "text layer read locally; no model call"),
        "audio_stt": Capability(
            "audio_stt", UNAVAILABLE, "tools.transcription_tools.transcribe_audio",
            "no STT provider configured"),
        "ocr": Capability(
            "ocr", UNAVAILABLE, "tesseract", "tesseract not installed"),
        "video_frames": Capability(
            "video_frames", UNAVAILABLE, "ffmpeg", "frame extraction not wired yet"),
    }
    try:
        from tools.transcription_tools import is_stt_enabled

        if is_stt_enabled():
            facilities["audio_stt"] = Capability(
                "audio_stt", HERMES, "tools.transcription_tools.transcribe_audio",
                "STT provider configured")
    except Exception:  # noqa: BLE001 - availability probing must never raise
        pass
    if _binary("tesseract"):
        facilities["ocr"] = Capability(
            "ocr", HERMES, "tesseract",
            "scanned pages can be transcribed locally" if _binary("pdftoppm")
            else "tesseract present but pdftoppm missing: images only")
    if _binary("ffmpeg"):
        facilities["video_frames"] = Capability(
            "video_frames", UNAVAILABLE, "ffmpeg",
            "ffmpeg present; frame extraction pipeline not implemented")
    return facilities


# --------------------------------------------------------------------------- model capability


def _candidate(provider: str, model: str, config: Any = None):
    from agent import model_router as mr

    specs, _source = mr.specs_from_config(config if config is not None else mr._load_config_for_preview())
    pool = mr.candidates_from_specs(
        config if config is not None else mr._load_config_for_preview(), specs,
        default_provider=provider or "")
    for candidate in pool:
        if candidate.model == model and (not provider or
                                         candidate.provider.lower() == provider.lower()):
            return candidate
    return None


def _modality_source(state: str) -> str:
    return "smart_model_routing.models / catalog metadata" if state != UNKNOWN else "no metadata"


def model_capabilities(provider: str, model: str, config: Any = None) -> Dict[str, Capability]:
    """Native capabilities as declared for this model, plus what Hermes can do around it."""
    from agent import model_router as mr

    candidate = _candidate(provider, model, config)
    facilities = local_facilities()
    out: Dict[str, Capability] = {}

    if candidate is None:
        out["_model"] = Capability("_model", UNKNOWN, "candidates", "model not in the routing pool")
        return out

    refused = set()
    try:
        from agent.capability_delegation import remembered_unusable

        refused = remembered_unusable()
    except Exception:  # noqa: BLE001
        pass
    if (candidate.provider.lower(), candidate.model) in refused:
        out["_model"] = Capability(
            "_model", UNAVAILABLE, "capability_delegation.unusable.json",
            "provider refused this model recently (retired/forbidden)")
        return out

    def _input(modality: str) -> Capability:
        state = candidate.supports_input(modality)
        value = {mr.CapabilityState.SUPPORTED: NATIVE,
                 mr.CapabilityState.UNSUPPORTED: UNKNOWN if False else UNAVAILABLE,
                 mr.CapabilityState.UNKNOWN: UNKNOWN}.get(state, UNKNOWN)
        return Capability(f"input_{modality}", value, "catalog metadata")

    for modality in ("image", "audio", "video", "pdf"):
        out[f"input_{modality}"] = _input(modality)

    # Hermes conversion turns an unsupported *model* input into a usable *system* capability.
    out["pdf"] = Capability(
        "pdf",
        NATIVE if out["input_pdf"].state == NATIVE else (
            HERMES if facilities["pdf_text"].usable else UNAVAILABLE),
        "model + local extraction" if out["input_pdf"].state != NATIVE else "model",
        "native pdf input" if out["input_pdf"].state == NATIVE
        else "Hermes extracts the text layer and sends text")
    out["audio"] = Capability(
        "audio",
        NATIVE if out["input_audio"].state == NATIVE else (
            HERMES if facilities["audio_stt"].usable else UNAVAILABLE),
        "model + local STT" if out["input_audio"].state != NATIVE else "model",
        "native audio input" if out["input_audio"].state == NATIVE
        else "Hermes transcribes locally and sends text")
    out["image"] = out["input_image"]
    out["video"] = Capability(
        "video",
        NATIVE if out["input_video"].state == NATIVE else (
            UNAVAILABLE),
        "model" if out["input_video"].state == NATIVE else "no local pipeline",
        "native video input" if out["input_video"].state == NATIVE
        else "frame extraction not implemented")
    out["ocr"] = facilities["ocr"]

    structured = getattr(candidate, "supports_output", None)
    if callable(structured):
        try:
            out["structured_output"] = Capability(
                "structured_output",
                NATIVE if structured("json") or structured("text") else UNKNOWN,
                "catalog metadata")
        except Exception:  # noqa: BLE001
            out["structured_output"] = Capability("structured_output", UNKNOWN, "catalog metadata")
    tools_state = mr.CapabilityState.UNKNOWN
    try:
        tools_state = candidate.supports_tools() if hasattr(candidate, "supports_tools") else tools_state
    except Exception:  # noqa: BLE001
        pass
    out["tools"] = Capability(
        "tools",
        {mr.CapabilityState.SUPPORTED: NATIVE, mr.CapabilityState.UNSUPPORTED: UNAVAILABLE}.get(
            tools_state, UNKNOWN),
        "catalog metadata")
    return out


def can_serve(provider: str, model: str, capability: str, config: Any = None) -> Capability:
    """The answer delegation needs: native, via Hermes, unknown, or unavailable."""
    caps = model_capabilities(provider, model, config)
    if capability in caps:
        return caps[capability]
    for value in caps.values():
        if value.name == capability:
            return value
    return Capability(capability, UNKNOWN, "no entry")


def describe(provider: str, model: str, config: Any = None) -> Dict[str, Any]:
    """A report body: what the model does, what Hermes does around it, and what is missing."""
    caps = model_capabilities(provider, model, config)
    native = sorted(k for k, v in caps.items() if v.state == NATIVE and not k.startswith("input_")
                    and not k.startswith("_"))
    via_hermes = sorted(k for k, v in caps.items() if v.state == HERMES)
    unavailable = sorted(k for k, v in caps.items() if v.state == UNAVAILABLE)
    return {
        "provider": provider,
        "model": model,
        "native": native,
        "via_hermes": via_hermes,
        "unavailable": unavailable,
        "detail": {k: v.as_dict() for k, v in caps.items()},
    }
