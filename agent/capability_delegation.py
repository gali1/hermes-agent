"""Specialist task delegation — a capability-specific sub-call the turn's own model owns.

Two different jobs live near each other and must not be confused:

* :mod:`agent.model_router` decides **which model serves a turn**.  One model, one turn, chosen
  before the model is invoked (``conversation_loop`` calls ``route_turn`` once).
* this module decides **which model performs one sub-task inside a turn the primary model still
  owns**.  The primary is not displaced: it asks for one thing (read this scan, transcribe this
  recording, review this diff), a declared specialist does it out of band, and the text comes
  back as ordinary context for the primary to answer from.

It is deliberately a thin layer over infrastructure that already exists:

* :mod:`agent.auxiliary_client` — ``call_llm`` / ``async_call_llm`` accept an explicit
  ``provider`` and ``model`` plus arbitrary ``messages``; the specialist call is that client with
  the router's pool as its candidate list.  No second provider client is built here.
* :mod:`agent.model_router` — eligibility, capability declarations, scoring and the
  preferred-models restriction are reused verbatim, so a specialist is chosen by the same rules
  that pick a turn's model (capability first, price as the tiebreak).

Honest limits this module encodes rather than hides:

* a model *declaring* a media input does not make the provider accept the transport for it —
  ``input_audio`` and ``file`` parts are provider-specific, so a rejection is returned as a
  failure with the provider's own words, never retried into a fake success;
* video needs frame extraction first; there is no silent fallback that pretends otherwise;
* an unknown capability is not a capability: a model whose metadata is silent is only eligible
  when the routing config allows unknown metadata, and that is reported in the choice reason.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import codecs
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agent import model_router as mr

logger = logging.getLogger(__name__)

#: Modalities a specialist call can carry today, and the part shape used for each.
TRANSPORTED_MODALITIES: Tuple[str, ...] = ("image", "audio", "pdf", "text", "document", "video")

#: Canonical modality names. These win over the alias table: "document" is a modality, not a
#: colloquial way of saying pdf, and letting the alias table decide turned a .xlsx into a "pdf"
#: with an application/pdf mime type.
KNOWN_MODALITIES = frozenset({"image", "audio", "video", "pdf", "text", "document", "unknown"})


def normalize_modality(value: str) -> str:
    explicit = (value or "").strip().lower()
    if explicit in KNOWN_MODALITIES:
        return explicit
    return MODALITY_ALIASES.get(explicit, explicit)


#: Spoken aliases an operator or a model may use.
MODALITY_ALIASES: Dict[str, str] = {
    "img": "image", "photo": "image", "picture": "image", "screenshot": "image",
    "voice": "audio", "sound": "audio", "recording": "audio", "speech": "audio",
    "doc": "pdf", "document": "pdf", "scan": "pdf", "scanned": "pdf",
    "movie": "video", "clip": "video", "film": "video",
    "spreadsheet": "document", "deck": "document", "slides": "document", "word": "document",
}

_SUFFIX_MODALITY: Dict[str, str] = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".gif": "image", ".bmp": "image", ".tif": "image", ".tiff": "image", ".heic": "image",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".ogg": "audio",
    ".oga": "audio", ".opus": "audio", ".flac": "audio", ".aac": "audio", ".aiff": "audio",
    ".pdf": "pdf",
    ".docx": "document", ".doc": "document", ".xlsx": "document", ".xls": "document",
    ".xlsm": "document", ".xlsb": "document", ".pptx": "document", ".ppt": "document",
    ".odt": "document", ".ods": "document", ".odp": "document", ".rtf": "document",
    ".epub": "document", ".ipynb": "document", ".db": "document", ".sqlite": "document",
    ".mp4": "video", ".mov": "video", ".mkv": "video", ".webm": "video", ".avi": "video",
    ".txt": "text", ".md": "text", ".markdown": "text", ".csv": "text", ".tsv": "text",
    ".json": "text", ".yaml": "text", ".yml": "text", ".log": "text", ".py": "text",
    ".js": "text", ".ts": "text", ".sh": "text", ".sql": "text", ".html": "text",
}

_AUDIO_FORMAT: Dict[str, str] = {
    ".mp3": "mp3", ".mpga": "mp3", ".m4a": "mp4", ".mp4": "mp4", ".mpeg": "mpeg",
    ".mpga": "mp3", ".wav": "wav", ".webm": "webm", ".ogg": "ogg", ".oga": "ogg",
    ".flac": "flac", ".aac": "aac", ".opus": "opus", ".aiff": "aiff",
}

#: Refuse before spending a request on a body no provider accepts.
DEFAULT_MAX_BYTES = 20 * 1024 * 1024
#: Inline text is capped so a 40 MB log cannot eat the specialist's whole window.
MAX_INLINE_TEXT_CHARS = 120_000


#: Providers retire models without warning: the catalog cache keeps listing them and the pool
#: keeps scoring them. A rejected specialty call is therefore remembered for a while so the next
#: resolve cannot route the same dead model into the same wall.
UNUSABLE_TTL_S = 24 * 3600
_DEPRECATED_MARKERS = (
    "modeldeprecated", "has been deprecated", "model is deprecated", "410",
    "model access is disabled", "access is disabled", "does not exist", "no such model",
    "model not found", "404",
)


def unusable_path() -> Path:
    """Where the "this model refused a specialty call" memo lives."""
    root = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(root) / "state" / "capability_delegation" / "unusable.json"


def _load_unusable() -> Dict[str, Dict[str, Any]]:
    path = unusable_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    now = time.time()
    return {k: v for k, v in data.items()
            if isinstance(v, dict) and (now - float(v.get("at") or 0)) < UNUSABLE_TTL_S}


def remembered_unusable() -> set:
    """``{(provider, model)}`` the provider refused a specialty call for, inside the TTL."""
    return {tuple(key.split("\x00", 1)) for key in _load_unusable()}


def remember_unusable(provider: str, model: str, reason: str = "") -> None:
    """Record a refusal so the next resolve does not pick the same model again."""
    path = unusable_path()
    data = _load_unusable()
    data[f"{provider}\x00{model}"] = {"at": time.time(), "reason": str(reason)[:200]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.chmod(path, 0o600)
    except OSError as exc:  # pragma: no cover - a memo that cannot be written must not break a call
        logger.info("could not persist the unusable-model memo: %s", exc)


def forget_unusable(provider: str = "", model: str = "") -> None:
    """Drop one memo entry (or all of them) - the operator's override after a provider fix."""
    path = unusable_path()
    data = _load_unusable()
    if not provider and not model:
        data = {}
    else:
        data.pop(f"{provider}\x00{model}", None)
    try:
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError:
        pass


def looks_unusable_error(error: str) -> bool:
    low = (error or "").lower()
    return any(marker in low for marker in _DEPRECATED_MARKERS)


def infer_modality(path: str, *, override: str = "") -> str:
    """Best-effort modality for a file, from an explicit name or its suffix."""
    explicit = (override or "").strip().lower()
    if explicit:
        return normalize_modality(explicit)
    return _SUFFIX_MODALITY.get(Path(path).suffix.lower(), "unknown")


@dataclass
class SpecialistChoice:
    """Who will do the sub-task, and why the router picked them."""

    provider: str
    model: str
    modality: str
    reason: str
    score: float = 0.0
    candidates_considered: List[str] = field(default_factory=list)
    #: The chosen model's own declared prices (USD per 1M tokens), for the audit record.
    cost_in: float = 0.0
    cost_out: float = 0.0


@dataclass
class DelegationResult:
    """The outcome of one specialist call — success is only ever claimed with text in hand."""

    success: bool
    text: str = ""
    provider: str = ""
    model: str = ""
    modality: str = ""
    bytes_sent: int = 0
    reason: str = ""
    error: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    est_cost_usd: float = 0.0
    #: False when the provider reported no token usage: est_cost_usd is then an ESTIMATE derived
    #: from the payload size and the specialist's declared rates, not a measured figure. Billing
    #: audit needs that distinction, and a money budget needs a non-zero number even when the
    #: provider is silent - treating unknown cost as 0.0 means the budget can never trip.
    cost_known: bool = True
    #: "usage" (measured), "estimated_from_bytes" (provider silent), "free_local" (STT/OCR).
    cost_basis: str = "usage"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "modality": self.modality,
            "bytes_sent": self.bytes_sent,
            "reason": self.reason,
            "error": self.error,
            "usage": dict(self.usage),
            "est_cost_usd": round(self.est_cost_usd, 8),
            "cost_known": self.cost_known,
            "cost_basis": self.cost_basis,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict())


# ---------------------------------------------------------------------------
# pool
# ---------------------------------------------------------------------------


def _routing_config(config: Any) -> mr.RoutingConfig:
    """The live routing block as a validated config object (unknown keys dropped)."""
    import dataclasses

    block = dict((config or {}).get("smart_model_routing") or {}) if isinstance(config, dict) else {}
    fields = {f.name for f in dataclasses.fields(mr.RoutingConfig)}
    return mr.RoutingConfig(**{k: v for k, v in block.items() if k in fields})


def _live_config() -> Any:
    from hermes_cli.config import load_config_readonly

    return load_config_readonly()


def _pool(config: Any) -> Tuple[Any, mr.RoutingConfig, List[mr.ModelCandidate], str]:
    cfg = _routing_config(config)
    specs, source = mr.specs_from_config(config)
    pool = mr.candidates_from_specs(config, specs, default_provider=_default_provider(config))
    return config, cfg, pool, source


def _default_provider(config: Any) -> str:
    model_block = (config or {}).get("model") or {}
    return str(model_block.get("provider") or model_block.get("provider_name") or "")


def _primary(config: Any) -> Tuple[str, str]:
    model_block = (config or {}).get("model") or {}
    return (
        str(model_block.get("provider") or ""),
        str(model_block.get("default") or model_block.get("model") or ""),
    )


def resolve_specialist(
    modality: str,
    *,
    config: Any = None,
    exclude: Sequence[Tuple[str, str]] = (),
    include_primary: bool = False,
) -> Optional[SpecialistChoice]:
    """Pick the declared model best suited to carrying ``modality``.

    Uses the routing pool and the routing eligibility rules, so ``preferred_models_only`` is
    respected: with it on, a model outside the operator's list is never chosen.  The config's
    own model is skipped by default — delegating a task to the model already holding the
    conversation is not delegation, and it would be a no-op with a cost.
    """
    want = normalize_modality(modality)
    if want not in TRANSPORTED_MODALITIES:
        return None

    cfg_obj = config if config is not None else _live_config()
    _, cfg, pool, _source = _pool(cfg_obj)

    if not pool and not cfg.preferred_models_only:
        # Nothing declared: widening is the existing behaviour for a capability gap.
        provider = _default_provider(cfg_obj) or "opencode"
        names = mr.widen_pool(provider, [])
        with_info = [n for n in names]
        # Candidates still need specs; build minimal ones so metadata does the deciding.
        specs = [mr.ModelSpec(name=n, provider=provider) for n in with_info]
        pool = mr.candidates_from_specs(cfg_obj, specs, default_provider=provider)

    profile = mr.classify_request(
        "delegated capability task",
        attachments=(want,),
        tool_available=True,
    )

    primary_provider, primary_model = _primary(cfg_obj)
    blocked = {(p.strip().lower(), m) for p, m in exclude}
    if not include_primary and primary_model:
        blocked.add((primary_provider.strip().lower(), primary_model))

    dead = remembered_unusable()
    eligible = [
        c for c in mr.filter_eligible(pool, profile, cfg)
        if (c.provider.strip().lower(), c.model) not in blocked
        and (c.provider.strip().lower(), c.model) not in dead
    ]
    if not eligible:
        logger.info("no declared specialist accepts %s (pool=%d)", want, len(pool))
        return None

    # Capability first, price as the tiebreak: the same weights `tool_calls_only` uses when it
    # is forced to replace the primary, because the job here is the same — serve a requirement.
    deleg_cfg = mr.RoutingConfig(
        enabled=True,
        preferred_models_only=cfg.preferred_models_only,
        allow_unknown_metadata=cfg.allow_unknown_metadata,
        weights={"capability": 0.60, "cost": 0.25, "reliability": 0.10,
                 "context": 0.05, "continuity": 0.0},
        reliability=cfg.reliability,
        exclude=cfg.exclude,
    )
    scored = sorted(
        ((c, mr.score_candidate(c, profile, deleg_cfg, current_model=primary_model,
                                current_provider=primary_provider))
         for c in eligible),
        key=lambda item: (-item[1], item[0].key),
    )
    best, best_score = scored[0]
    return SpecialistChoice(
        provider=best.provider,
        model=best.model,
        modality=want,
        reason=(
            f"capability-first choice for {want} input among {len(eligible)} eligible "
            f"declared model(s); price only breaks ties"
        ),
        score=round(best_score, 4),
        candidates_considered=[c.key for c, _ in scored],
        cost_in=float(getattr(best.spec, "cost_in", 0.0) or 0.0),
        cost_out=float(getattr(best.spec, "cost_out", 0.0) or 0.0),
    )


# ---------------------------------------------------------------------------
# message building
# ---------------------------------------------------------------------------


def _extract_pdf_text(path: Path) -> str:
    """Text layer of a PDF/DOCX/XLSX via the repo's own extractor; "" when there is none."""
    try:
        from tools.read_extract import extract_document_text

        return extract_document_text(str(path)) or ""
    except Exception as exc:  # noqa: BLE001 - a scan with no text layer is expected, not a crash
        logger.info("local extraction failed for %s: %s", path.name, exc)
        return ""


def decode_text_payload(data: bytes) -> Tuple[str, str]:
    """(text, encoding). BOM first, then strict UTF-8, then cp1252.

    Measured before this: a cp1252 spreadsheet arrived as ``Ren\ufffd M\ufffdller`` and a UTF-16
    one as NUL-separated garbage, because everything was decoded as UTF-8 with errors="replace".
    """
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig", errors="replace"), "utf-8-sig"
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return data.decode("utf-16", errors="replace"), "utf-16"
    if data.startswith(codecs.BOM_UTF32_LE) or data.startswith(codecs.BOM_UTF32_BE):
        return data.decode("utf-32", errors="replace"), "utf-32"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        pass
    # latin-1 cannot fail, so this always answers — but say that it was a guess, because the same
    # bytes read through read_file are refused rather than shown as mojibake.
    return data.decode("latin-1", errors="replace"), "latin-1"


def clip_text_with_counts(body: str, limit: int) -> Tuple[str, str]:
    """(clipped, note) - cut on a line boundary and say how much was kept.

    A block cut mid-row with no counts lets a reader believe they saw the whole file.
    """
    if len(body) <= limit:
        return body, ""
    head = body[:limit]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[:cut]
    shown_rows = head.count("\n") + 1
    total_rows = body.count("\n") + 1
    return head, (f"truncated: showed {shown_rows} of {total_rows} rows, "
                  f"{len(head)} of {len(body)} chars")


def usable_text(text: str) -> str:
    """Drop the extractor's placeholder notices and keep only real content.

    `tools.read_extract` does not return "" for a scanned page: it returns a bracketed
    "[NEEDS OCR: pages 1 …]" instruction to the *model*.  Measuring that as text made OCR look
    unnecessary and sent the instruction itself to the specialist — which then (correctly)
    answered "the page isn't available".  Strip the notice, then decide.
    """
    kept = [line for line in (text or "").splitlines() if "[needs ocr" not in line.lower()]
    return "\n".join(kept).strip()


def ocr_available() -> bool:
    """Local OCR needs both the rasteriser and the engine; one without the other is a trap."""
    return bool(shutil.which("pdftoppm")) and bool(shutil.which("tesseract"))


def _ocr_pdf(path: Path, *, dpi: int = 200, max_pages: int = 20,
             lang: str = "eng") -> str:
    """Transcribe a scanned PDF locally with poppler + tesseract; "" when unavailable."""
    if not ocr_available():
        return ""
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-ocr-") as tmp:
            subprocess.run(
                ["pdftoppm", "-r", str(dpi), "-png", "-f", "1", "-l", str(max_pages),
                 str(path), str(Path(tmp) / "page")],
                check=True, capture_output=True, timeout=300,
            )
            pages = sorted(Path(tmp).glob("page*.png"))
            chunks = []
            for page in pages:
                done = subprocess.run(["tesseract", str(page), "stdout", "-l", lang],
                                      capture_output=True, text=True, timeout=180)
                if done.returncode == 0 and done.stdout.strip():
                    chunks.append(done.stdout)
            return "\n".join(chunks).strip()
    except Exception as exc:  # noqa: BLE001 - OCR is best-effort, never fatal
        logger.info("OCR failed for %s: %s", path.name, exc)
        return ""


def _extract_document_text(path: Path) -> str:
    """xlsx/csv/docx/pptx/odt/... via the readers this tree already ships."""
    try:
        from tools.read_extract import extract_document_text

        return extract_document_text(str(path)) or ""
    except Exception as exc:  # noqa: BLE001 - an unreadable deck is a note, not a crash
        logger.info("document extraction failed for %s: %s", path.name, exc)
        return ""


def _transcribe_file(path: Path) -> Tuple[bool, str, str]:
    """Local/cloud STT through the existing transcription stack. Returns (ok, text, why)."""
    try:
        from tools.transcription_tools import transcribe_audio

        result = transcribe_audio(str(path), source="capability_delegation")
    except Exception as exc:  # noqa: BLE001
        return False, "", f"{type(exc).__name__}: {str(exc)[:200]}"
    if isinstance(result, dict) and result.get("success"):
        return True, str(result.get("transcript") or ""), ""
    why = result.get("error") if isinstance(result, dict) else "transcription returned no result"
    return False, "", str(why or "transcription produced no text")[:200]


def _usage_of(response: Any) -> Dict[str, int]:
    """Prompt/completion tokens when the provider reports them; {} when it does not."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if isinstance(usage, dict):
        return {key: int(value) for key, value in usage.items()
                if key in ("input_tokens", "output_tokens", "total_tokens")
                and isinstance(value, int)}
    if usage is None:
        return {}
    out: Dict[str, int] = {}
    for key, attr in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
                      ("total_tokens", "total_tokens")):
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            out[key] = value
    return out


#: Used ONLY when the provider reports no usage, and labelled as an estimate. It deliberately
#: OVERCOUNTS: a budget that under-charges a silent provider is a budget that never trips.
#: 3 bytes/token is pessimistic for text, and the output allowance takes the configured cap when
#: one exists rather than assuming a short reply.
_EST_BYTES_PER_TOKEN = 3
_EST_OUTPUT_TOKENS_FLOOR = 1024


def _output_token_allowance() -> int:
    try:
        from hermes_cli.config import load_config_readonly

        block = (load_config_readonly() or {}).get("capability_delegation") or {}
        cap = int(block.get("max_output_tokens") or 0)
    except Exception:  # noqa: BLE001
        cap = 0
    return max(cap, _EST_OUTPUT_TOKENS_FLOOR)


def _price(choice: SpecialistChoice, usage: Dict[str, int], *,
           bytes_sent: int = 0, modality: str = "") -> Tuple[float, str]:
    """(usd, basis) — measured when the provider reports usage, labelled estimate when it does not."""
    if usage:
        usd = (usage.get("input_tokens", 0) * choice.cost_in
               + usage.get("output_tokens", 0) * choice.cost_out) / 1_000_000.0
        return usd, "usage"
    if not (choice.cost_in or choice.cost_out):
        return 0.0, "free_local"
    est_in = max(1, int(bytes_sent or 0) // _EST_BYTES_PER_TOKEN)
    usd = (est_in * choice.cost_in
           + _output_token_allowance() * choice.cost_out) / 1_000_000.0
    binary = modality in ("audio", "video", "image", "pdf", "document")
    return usd, ("estimated_from_bytes_binary" if binary else "estimated_from_bytes")


def _read_limited(path: Path, max_bytes: int) -> Tuple[bytes, str]:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"{path.name} is {size / (1024 * 1024):.1f} MB, above the {max_bytes / (1024 * 1024):.1f} MB "
            "delegation limit — convert or split it first"
        )
    return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"


#: path -> (VideoContext, expiry). The decomposed representation is expensive (STT), so it is built
#: once and reused to attach the verbatim on-screen timeline to the result the primary receives.
_VIDEO_CONTEXT_CACHE: Dict[str, tuple] = {}
_VIDEO_CONTEXT_TTL_S = 900.0


def _cache_video_context(path: str, context: Any) -> None:
    import time as _time

    _VIDEO_CONTEXT_CACHE[str(path)] = (context, _time.time() + _VIDEO_CONTEXT_TTL_S)


def cached_video_timeline(path: str, limit: int = 40) -> str:
    """Verbatim OCR lines with their timestamps, so the primary never has to re-read frames."""
    import time as _time

    entry = _VIDEO_CONTEXT_CACHE.get(str(path))
    if not entry:
        return ""
    context, expiry = entry
    if expiry <= _time.time():
        _VIDEO_CONTEXT_CACHE.pop(str(path), None)
        return ""
    lines = []
    for frame in getattr(context, "frames", [])[:limit]:
        text = " ".join((frame.ocr_text or "").split())
        if text:
            lines.append(f"[{frame.timestamp_s:6.2f}s] {text}")
    if not lines:
        return ""
    return ("RAW ON-SCREEN TIMELINE (verbatim OCR, sampled frames):\n"
            + "\n".join(lines))


def build_delegation_messages(
    path: str,
    instruction: str,
    modality: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_frames: int = 8,
    transport: str = "auto",
) -> Tuple[List[Dict[str, Any]], int, str]:
    """Messages for the specialist: one text part with the task, one part carrying the file.

    Returns ``(messages, bytes_sent, note)``. Raises ``ValueError`` for anything that cannot be
    transported — a refusal here is cheaper and more honest than a provider 400.

    ``transport`` selects how a pdf/audio is carried:

    * ``"auto"`` (default) — pdf: extract the text layer locally and send the text; if there is
      no text layer (a scan), send the bytes as a native ``file`` part and say so.  audio:
      transcribe with the existing STT stack and send the transcript; if STT is unavailable,
      fall back to a native ``input_audio`` part.
    * ``"text"`` / ``"stt"`` — the portable path only; raise if it cannot produce text.
    * ``"native"`` — skip extraction/transcription and send the provider part directly.
    """
    if not (instruction or "").strip():
        raise ValueError("delegation needs an instruction: what the specialist should return")
    target = Path(os.path.expanduser(str(path or "")))
    if not str(path or "").strip():
        raise ValueError("delegation needs a file path")
    if not target.exists() or not target.is_file():
        raise ValueError(f"no such file: {target}")

    want = normalize_modality(modality)
    data, mime = _read_limited(target, max_bytes)
    header = (
        f"{instruction.strip()}\n\n"
        f"— Delegated by the primary assistant. Return only the requested result; "
        f"the assistant will present it to the user."
    )

    if want == "text":
        body, encoding = decode_text_payload(data)
        body, clip_note = clip_text_with_counts(body, MAX_INLINE_TEXT_CHARS)
        note = f"inlined {target.name} as text ({encoding})"
        if clip_note:
            note += f" — {clip_note}"
            body += f"\n…[{clip_note}]"
        return ([{"role": "user", "content": f"{header}\n\n--- {target.name} ---\n{body}"}],
                len(body.encode("utf-8")), note)

    if want == "image":
        url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        note = f"sent {target.name} as an image_url part ({mime})"
        return ([{"role": "user", "content": [
            {"type": "text", "text": header},
            {"type": "image_url", "image_url": {"url": url}},
        ]}], len(data), note)

    if want == "audio":
        stt_why = ""
        if transport != "native":
            ok, transcript, stt_why = _transcribe_file(target)
            if ok and transcript.strip():
                clipped = transcript[:MAX_INLINE_TEXT_CHARS]
                note = (f"transcribed {target.name} with the STT stack "
                        f"({len(transcript)} chars) and sent the transcript")
                return ([{"role": "user", "content": (
                    f"{header}\n\n--- {target.name} (transcript) ---\n{clipped}")}],
                        len(clipped.encode("utf-8")), note)
            if transport == "stt":
                raise ValueError(
                    f"could not transcribe {target.name}: {stt_why}. Configure an STT provider "
                    "or set transport='native' to send the audio itself.")
        fmt = _AUDIO_FORMAT.get(target.suffix.lower(), "wav")
        suffix = f" (after STT failed: {stt_why})" if stt_why else ""
        note = (f"sent {target.name} as an input_audio part ({fmt}){suffix} — no audio part "
                "shape is guaranteed on every provider")
        return ([{"role": "user", "content": [
            {"type": "text", "text": header},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(data).decode("ascii"), "format": fmt}},
        ]}], len(data), note)

    if want == "document":
        extracted = "" if transport == "native" else usable_text(_extract_document_text(target))
        if extracted:
            note = (f"extracted {target.name} locally with the document readers "
                    f"({len(extracted)} chars, structure preserved) and sent the text")
            return ([{"role": "user", "content": f"[{note}]\n\n{extracted[:MAX_INLINE_TEXT_CHARS]}"}],
                    len(extracted), note)
        note = f"no text could be extracted from {target.name}"
        return ([{"role": "user", "content": f"[{note}]"}], 0, note)

    if want == "video":
        if transport == "native":
            note = f"{target.name}: no provider here accepts a native video part"
            return ([{"role": "user", "content": f"[{note}]"}], 0, note)
        from agent.video_context import build_video_context

        context = build_video_context(str(target), max_frames=max_frames, do_ocr=True,
                                      do_transcribe=True, transcribe=_transcribe_file)
        _cache_video_context(str(target), context)
        text = context.as_text()
        with_text = sum(1 for frame in context.frames if frame.ocr_text.strip())
        note = (f"decomposed {target.name} locally into a timestamped representation "
                f"({with_text} of {len(context.frames)} sampled frame(s) had readable text, "
                f"speech {'transcribed' if context.transcript else 'unavailable'})")
        return ([{"role": "user", "content": f"[{note}]\n\n{text}"}], len(text), note)

    if want == "pdf":
        extracted = "" if transport == "native" else usable_text(_extract_pdf_text(target))
        ocr_used = False
        if transport != "native" and len(extracted) < 40 and ocr_available():
            ocr_text = _ocr_pdf(target)
            if len(ocr_text.strip()) > len(extracted.strip()):
                extracted, ocr_used = ocr_text, True
        if extracted.strip():
            clipped = extracted[:MAX_INLINE_TEXT_CHARS]
            how = "OCR (tesseract)" if ocr_used else "text layer"
            note = (f"{how} on {target.name} locally ({len(extracted)} chars, portable path) "
                    "and sent the text")
            return ([{"role": "user", "content": (
                f"{header}\n\n--- {target.name} (extracted text) ---\n{clipped}")}],
                    len(clipped.encode("utf-8")), note)
        if transport == "text":
            raise ValueError(
                f"could not extract text from {target.name}: it may be a scan with no text "
                "layer. Set transport='native' to send the bytes as a file part.")
        url = f"data:application/pdf;base64,{base64.b64encode(data).decode('ascii')}"
        note = (f"sent {target.name} as a file part (application/pdf) after local extraction "
                "produced nothing — not every provider accepts this part type")
        return ([{"role": "user", "content": [
            {"type": "text", "text": header},
            {"type": "file", "file": {"filename": target.name, "file_data": url}},
        ]}], len(data), note)

    raise ValueError(
        f"{want} cannot be delegated as-is: no supported transport part exists for it. "
        "Extract frames or text first (ffmpeg / read_file), then delegate that."
    )


# ---------------------------------------------------------------------------
# the call
# ---------------------------------------------------------------------------


async def delegate_capability(
    path: str,
    instruction: str,
    *,
    modality: str = "",
    config: Any = None,
    timeout: float = 120.0,
    max_tokens: int = 4000,
    max_bytes: int = DEFAULT_MAX_BYTES,
    exclude: Sequence[Tuple[str, str]] = (),
    choice: Optional[SpecialistChoice] = None,
    transport: str = "auto",
) -> DelegationResult:
    """Run one capability-specific sub-task on a declared specialist and return its text."""
    try:
        want = infer_modality(path, override=modality)
    except Exception as exc:  # pragma: no cover - defensive
        return DelegationResult(success=False, error=str(exc))

    # ---- audio: transcribe, do not rent a model -------------------------------------
    # The capability the primary model lacks here is speech-to-text, which is infrastructure and
    # already exists in this tree. Returning the transcript beats paying a specialist to read
    # text the primary can read itself; `transport="native"` still sends the audio to a model.
    if want == "audio" and transport != "native":
        try:
            size = Path(os.path.expanduser(str(path or ""))).stat().st_size
        except OSError as exc:
            return DelegationResult(success=False, modality="audio",
                                    error=f"no such file: {path} ({exc})")
        ok, transcript, why = await asyncio.to_thread(
            _transcribe_file, Path(os.path.expanduser(str(path))))
        if ok and transcript.strip():
            return DelegationResult(
                success=True, text=transcript.strip(), provider="stt", model="local",
                modality="audio", bytes_sent=size, cost_known=True, cost_basis="free_local",
                reason="transcribed locally with the STT stack — no model call was needed",
            )
        return DelegationResult(
            success=False, modality="audio", bytes_sent=size,
            error=(f"could not transcribe {Path(path).name}: {why}. Install/point an STT provider, "
                   "or set transport='native' to send the audio to a specialist model."),
        )

    if choice is None:
        # A video or a document is *decomposed or extracted* here, so the specialist that matters
        # is one that reads text - asking for a model with native video input would refuse a task
        # this installation can already serve.
        resolve_as = "text" if want in ("video", "document") else want
        choice = resolve_specialist(resolve_as, config=config, exclude=exclude)
    if choice is None:
        return DelegationResult(
            success=False,
            modality=want,
            error=(
                f"no declared specialist accepts {want} input — add one to the routing pool "
                "(smart_model_routing.models) or relax preferred_models_only"
            ),
        )

    try:
        # extraction and STT are blocking: keep them off the event loop.
        messages, sent, note = await asyncio.to_thread(
            build_delegation_messages, path, instruction, want,
            max_bytes=max_bytes, transport=transport)
    except ValueError as exc:
        return DelegationResult(
            success=False, provider=choice.provider, model=choice.model,
            modality=want, error=str(exc), reason=choice.reason,
        )

    from agent import auxiliary_client as aux

    try:
        response = await aux.async_call_llm(
            task="capability_delegation",
            provider=choice.provider,
            model=choice.model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - the provider's own words are the report
        detail = f"{type(exc).__name__}: {str(exc)[:400]}"
        if looks_unusable_error(detail):
            remember_unusable(choice.provider, choice.model, detail)
        return DelegationResult(
            success=False, provider=choice.provider, model=choice.model, modality=want,
            bytes_sent=sent, reason=f"{choice.reason}; {note}",
            error=detail,
        )

    text = ""
    try:
        text = aux.extract_content_or_reasoning(response) or ""
    except Exception as exc:  # noqa: BLE001
        return DelegationResult(
            success=False, provider=choice.provider, model=choice.model, modality=want,
            bytes_sent=sent, reason=f"{choice.reason}; {note}",
            error=f"could not read the specialist response: {type(exc).__name__}: {exc}",
        )
    text = text.strip()
    if not text:
        return DelegationResult(
            success=False, provider=choice.provider, model=choice.model, modality=want,
            bytes_sent=sent, reason=f"{choice.reason}; {note}",
            error="the specialist returned no content",
        )
    if want == "video":
        timeline = cached_video_timeline(path)
        if timeline and "RAW ON-SCREEN TIMELINE" not in text:
            # The specialist summarises and drops timestamps; the primary then wants to re-read
            # frames with vision to pin a second (measured: 8 vision calls, an exhausted turn).
            # Hand it the verbatim timeline instead, at no extra cost.
            text = f"{text}\n\n{timeline}"
    usage = _usage_of(response)
    usd, basis = _price(choice, usage, bytes_sent=sent, modality=want)
    return DelegationResult(
        success=True, text=text, provider=choice.provider, model=choice.model,
        modality=want, bytes_sent=sent, reason=f"{choice.reason}; {note}",
        usage=usage, est_cost_usd=usd, cost_known=bool(usage), cost_basis=basis,
    )


def delegate_capability_sync(
    path: str, instruction: str, **kwargs: Any
) -> DelegationResult:
    """Blocking wrapper for callers that are not already inside an event loop."""
    return asyncio.run(delegate_capability(path, instruction, **kwargs))
