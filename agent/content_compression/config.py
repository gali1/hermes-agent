"""Config access for content-aware tool-output compression.

Reads ``compression.content_aware`` from the user's ``config.yaml`` via
``hermes_cli.config.load_config()`` (lazily imported to avoid any import
cycle) and caches the merged result for a short TTL so the agent loop does
not re-parse YAML per tool call. Any failure returns the defaults -- the
compression path is strictly fail-open.

Defaults::

    {"enabled": True, "lossless_folds": True,
     "dense_line_elision": False, "min_savings_chars": 64,
     "type_compression": True, "json_max_items": 15,
     "log_max_lines": 100, "search_max_total": 30,
     "read_lifecycle": True, "net_cost_gate": False}

``dense_line_elision`` and the lossy tier of the type compressors are lossy and
therefore gated on a retrievable original; lossless folds and lossless type
renders are default-on. ``min_savings_chars`` is the minimum input length
before compression is attempted at all (fold markers have fixed overhead, so
tiny results are left alone).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

__all__ = ["get_content_compression_config", "reset_cache"]

DEFAULT_CONTENT_AWARE_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "lossless_folds": True,
    "dense_line_elision": False,
    "min_savings_chars": 64,
    "type_compression": True,
    "json_max_items": 15,
    "log_max_lines": 100,
    "search_max_total": 30,
    "read_lifecycle": True,
    "net_cost_gate": False,
}

_CACHE_TTL_SECONDS = 30.0

_cache_lock = threading.Lock()
_cache_value: Dict[str, Any] | None = None
_cache_at: float = 0.0


def _coerce(section: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the user section over the defaults, type-checking each key."""
    merged = dict(DEFAULT_CONTENT_AWARE_CONFIG)
    for key, default in DEFAULT_CONTENT_AWARE_CONFIG.items():
        if key not in section:
            continue
        value = section[key]
        if isinstance(default, bool):
            if isinstance(value, bool):
                merged[key] = value
        elif isinstance(default, int):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = int(value)
    return merged


def _load() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        compression = config.get("compression")
        if not isinstance(compression, dict):
            return dict(DEFAULT_CONTENT_AWARE_CONFIG)
        section = compression.get("content_aware")
        if not isinstance(section, dict):
            return dict(DEFAULT_CONTENT_AWARE_CONFIG)
        return _coerce(section)
    except Exception:
        return dict(DEFAULT_CONTENT_AWARE_CONFIG)


def get_content_compression_config() -> Dict[str, Any]:
    """Return the effective ``compression.content_aware`` settings.

    Cached for 30 seconds. Never raises: any load failure yields the
    defaults. Callers may mutate the returned dict freely (it is a copy).
    """
    global _cache_value, _cache_at

    now = time.monotonic()
    with _cache_lock:
        cached = _cache_value
        if cached is not None and (now - _cache_at) < _CACHE_TTL_SECONDS:
            return dict(cached)

    value = _load()

    with _cache_lock:
        _cache_value = value
        _cache_at = time.monotonic()
    return dict(value)


def reset_cache() -> None:
    """Drop the cached config (test hook and config-change invalidation)."""
    global _cache_value, _cache_at
    with _cache_lock:
        _cache_value = None
        _cache_at = 0.0
