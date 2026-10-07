"""Content-aware compression of tool outputs (Headroom-derived, Phases 1-2).

Makes tool outputs smaller before they reach the model without losing
information:

* **Lossless folds** (:mod:`agent.content_compression.lossless`) are
  default-on. Each fold is information-preserving by construction and
  round-trip verified before adoption, so enabling them never loses data.
* **Content router + type compressors** (Phases 2) route each output to a
  compressor for its shape. The lossless tiers are free: JSON arrays of
  objects render to a ``[N]{schema}`` CSV block (verified to re-parse to the
  exact same list of dicts), config stanzas fold, tabular text folds. The
  lossy tiers (JSON item drop, log line drop, search match drop, diff
  context/hunk/file drop) only run when the original can be persisted through
  the ``retrieval_writer`` and the returned hint is embedded in the drop
  marker.
* **Dense-line elision** (:mod:`agent.content_compression.dense_lines`) is
  lossy and default-off. When enabled, the original must remain retrievable:
  :func:`compress_tool_output` only elides after a ``retrieval_writer``
  persists the original and returns a hint (e.g. a sandbox path), which is
  embedded in the elision marker. With no writer/hint, elision is skipped.

Everything is fail-open: :func:`compress_tool_output` never raises, every
type compressor is wrapped, and the original text is returned on any failure.

Config lives under ``compression.content_aware`` in ``config.yaml`` (see
:mod:`agent.content_compression.config`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from . import config as _config
from . import config_compressor as _config_compressor
from . import dense_lines as _dense_lines
from . import diff_compressor as _diff_compressor
from . import json_crusher as _json_crusher
from . import log_compressor as _log_compressor
from . import lossless as _lossless
from . import router as _router
from . import search_compressor as _search_compressor
from . import tabular as _tabular
from .dense_lines import elide_dense_lines
from .lossless import fold_lossless, strip_ansi
from .protection import is_critical_line

__all__ = [
    "CompressionResult",
    "compress_tool_output",
    "detect_content_type",
    "elide_dense_lines",
    "fold_lossless",
    "strip_ansi",
    "is_critical_line",
    "compress_json",
    "compress_log",
    "compress_search",
    "compress_diff",
    "compress_config",
    "compress_tabular",
]

detect_content_type = _router.detect_content_type
compress_json = _json_crusher.compress_json
compress_log = _log_compressor.compress_log
compress_search = _search_compressor.compress_search
compress_diff = _diff_compressor.compress_diff
compress_config = _config_compressor.compress_config
compress_tabular = _tabular.compress_tabular

_TYPE_COMPRESSORS = {
    "json": (_json_crusher, "compress_json"),
    "log": (_log_compressor, "compress_log"),
    "search": (_search_compressor, "compress_search"),
    "diff": (_diff_compressor, "compress_diff"),
    "config": (_config_compressor, "compress_config"),
    "tabular": (_tabular, "compress_tabular"),
}

# Throwaway hint for the no-side-effect "would this drop anything?" probe.
# Never adopted and never surfaced to the caller.
_PROBE_HINT = "<probe>"


@dataclass
class CompressionResult:
    """Outcome of :func:`compress_tool_output`.

    ``changed`` is True when ``text`` differs from the input. ``lossless`` is
    True when no information was dropped (folds/lossless type renders only, or
    nothing applied); it is False exactly when a lossy stage fired (type drop
    or dense-line elision). ``elided_lines`` is the number of dense lines
    replaced by markers; ``dropped_units`` is the number of units (JSON items,
    log lines, search matches, diff lines, config lines) dropped by the type
    compressor. ``strategy`` names the last stage that produced the final text:
    ``"lossless"``, ``"json"``, ``"log"``, ``"search"``, ``"diff"``,
    ``"config"``, ``"tabular"``, ``"dense"``, or ``"none"``.
    """

    text: str
    changed: bool
    lossless: bool
    elided_lines: int
    original_chars: int
    new_chars: int
    strategy: str = "none"
    dropped_units: int = 0


def compress_tool_output(
    content,
    *,
    tool_name: str = "",
    tool_use_id: str = "",
    env=None,
    retrieval_writer: Optional[Callable[[str], Optional[str]]] = None,
) -> CompressionResult:
    """Compress a tool output for the model context. Never raises.

    ``retrieval_writer`` is called lazily -- only when a lossy stage would
    actually drop content (dense-line elision, or a type compressor that
    returned no lossless improvement) -- with the ORIGINAL text, and must
    return a retrieval hint (e.g. the sandbox path where it persisted the
    original) or None. If it returns None (or is not provided), lossy stages
    are skipped entirely: lossy compression without a retrieval path is never
    applied.

    ``tool_name``/``tool_use_id``/``env`` are accepted for call-site symmetry
    and are available to a retrieval writer via closure; ``tool_name`` is also
    used as the weak router prior.
    """
    if not isinstance(content, str):
        return CompressionResult(
            text=content,
            changed=False,
            lossless=True,
            elided_lines=0,
            original_chars=0,
            new_chars=0,
            strategy="none",
            dropped_units=0,
        )

    original = content
    original_chars = len(original)

    def _unchanged() -> CompressionResult:
        return CompressionResult(
            text=original,
            changed=False,
            lossless=True,
            elided_lines=0,
            original_chars=original_chars,
            new_chars=original_chars,
            strategy="none",
            dropped_units=0,
        )

    try:
        cfg = _config.get_content_compression_config()
        if not cfg.get("enabled", True):
            return _unchanged()

        try:
            min_savings = int(cfg.get("min_savings_chars", 64))
        except (TypeError, ValueError):
            min_savings = 64
        if original_chars < min_savings:
            return _unchanged()

        text = original
        strategy = "none"
        if cfg.get("lossless_folds", True):
            text = _lossless.fold_lossless(text)
            if text != original:
                strategy = "lossless"

        type_dropped = 0
        if cfg.get("type_compression", True) and text:
            text, strategy, type_dropped = _apply_type_compression(
                text, original, tool_name, retrieval_writer, cfg, strategy
            )

        elided_lines = 0
        if cfg.get("dense_line_elision", False) and text:
            # Probe without a hint first: the retrieval write is the only
            # side effect in this path and must not happen when there is
            # nothing to elide.
            _, would_elide = _dense_lines.elide_dense_lines(text)
            if would_elide:
                hint = None
                if retrieval_writer is not None:
                    try:
                        hint = retrieval_writer(original)
                    except Exception:
                        hint = None
                if hint:
                    candidate, elided_lines = _dense_lines.elide_dense_lines(
                        text, retrieval_hint=hint
                    )
                    if elided_lines:
                        text = candidate
                        strategy = "dense"
                    else:
                        elided_lines = 0
                # No writer/hint -> skip elision. Never drop information
                # without a retrieval path.

        if text != original and len(text) < original_chars:
            return CompressionResult(
                text=text,
                changed=True,
                lossless=(elided_lines == 0 and type_dropped == 0),
                elided_lines=elided_lines,
                original_chars=original_chars,
                new_chars=len(text),
                strategy=strategy if strategy != "none" else "lossless",
                dropped_units=type_dropped,
            )
        return _unchanged()
    except Exception:
        return _unchanged()


def _apply_type_compression(
    text: str,
    original: str,
    tool_name: str,
    retrieval_writer,
    cfg: dict,
    current_strategy: str,
) -> tuple[str, str, int]:
    """Run the routed type compressor. Returns ``(text, strategy, dropped)``.

    Routing runs against the pre-fold ``original`` because the lossless folds
    can rewrite structural lines inside pretty JSON / diffs and make the
    folded text unrecognizable. The compressor input is the current ``text``
    when it still routes to the same type (the common case), else the
    original.

    The first call is lossless-only. If it yields no improvement, a
    no-side-effect dry run (throwaway hint) checks whether the compressor
    would actually drop content; only then does the writer run (once) and the
    compressor re-run with ``allow_lossy=True``. Any exception is treated as
    "no improvement".
    """
    try:
        content_type, _confidence = _router.detect_content_type(original, tool_name=tool_name)
    except Exception:
        return text, current_strategy, 0
    entry = _TYPE_COMPRESSORS.get(content_type)
    if entry is None:
        return text, current_strategy, 0
    module, attr = entry
    try:
        compressor = getattr(module, attr)
    except Exception:
        return text, current_strategy, 0

    source = original
    if text != original:
        try:
            if _router.detect_content_type(text, tool_name=tool_name)[0] == content_type:
                source = text
        except Exception:
            source = original

    kwargs = _type_kwargs(content_type, cfg)
    candidate, _dropped = _safe_compress(
        compressor, source, allow_lossy=False, retrieval_hint=None, kwargs=kwargs
    )
    if candidate is not None and len(candidate) < len(text):
        return candidate, content_type, 0

    if retrieval_writer is not None:
        # Dry-run with a throwaway hint to learn whether the compressor would
        # actually drop anything. The retrieval write is the only side effect
        # in this path and must not happen when nothing would be dropped.
        probe, would_drop = _safe_compress(
            compressor,
            source,
            allow_lossy=True,
            retrieval_hint=_PROBE_HINT,
            kwargs=kwargs,
        )
        if would_drop > 0 and probe is not None and len(probe) < len(text):
            hint = None
            try:
                hint = retrieval_writer(original)
            except Exception:
                hint = None
            if hint:
                candidate, dropped = _safe_compress(
                    compressor,
                    source,
                    allow_lossy=True,
                    retrieval_hint=hint,
                    kwargs=kwargs,
                )
                if candidate is not None and len(candidate) < len(text):
                    return candidate, content_type, dropped
    return text, current_strategy, 0


def _type_kwargs(content_type: str, cfg: dict) -> dict:
    def _as_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    if content_type == "json":
        return {"max_items": _as_int(cfg.get("json_max_items"), 15)}
    if content_type == "log":
        return {"max_lines": _as_int(cfg.get("log_max_lines"), 100)}
    if content_type == "search":
        return {"max_total": _as_int(cfg.get("search_max_total"), 30)}
    return {}


def _safe_compress(compressor, text, *, allow_lossy, retrieval_hint, kwargs):
    """Call a type compressor; return ``(candidate_or_None, dropped)``."""
    try:
        result = compressor(
            text,
            allow_lossy=allow_lossy,
            retrieval_hint=retrieval_hint,
            **kwargs,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            return None, 0
        candidate, dropped = result
        if not isinstance(candidate, str) or not isinstance(dropped, int):
            return None, 0
        return candidate, max(0, dropped)
    except Exception:
        return None, 0
