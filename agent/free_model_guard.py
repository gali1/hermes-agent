"""Free-tier model quota guard.

Problem it solves: a free model (`*-free`) can stop answering mid-task when its daily
allowance runs out, and routing would happily pick it again on the very next turn. This
module watches for that, switches to a paid model the operator named, and refuses to
re-select the exhausted free model until the window rolls over.

Design rules:
  * One state file, one event log, no second config store. Config lives in the existing
    ``smart_model_routing.free_guard`` block.
  * Detection is evidence-based: the provider's own failure classification
    (rate_limit / billing / upstream_rate_limit), never a guess from a slow response.
  * Exhaustion is scoped to a WINDOW (a date that rolls at ``reset_hour``), so the ban
    lifts by itself the next day. Nothing has to be cleaned up by hand.
  * Every decision is written down: switches and usage land in an append-only JSONL log,
    because "why did it move off the free model at 14:20" must be answerable later.
  * Failures here never break a turn: every entry point swallows its own exceptions.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_RESET_HOUR = 0
_FREE_SUFFIXES = ("-free", ":free", "_free")
_QUOTA_REASONS = frozenset({"rate_limit", "billing", "upstream_rate_limit"})


# --------------------------------------------------------------------------- config
def guard_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The ``smart_model_routing.free_guard`` block.

    Accepts either the whole config or the ``smart_model_routing`` block itself, because
    callers hold both: ``explain_routing`` is handed the routing block, the CLI holds the full
    file. Getting this wrong silently fell through to the live config file, so an explicit
    ``free_guard: {enabled: false}`` was ignored and the guard acted anyway.
    """
    def _block_from(source: Dict[str, Any]) -> Dict[str, Any]:
        block = source.get("free_guard")
        if isinstance(block, dict):
            return block
        if block is not None:
            return {}
        # No ``free_guard`` key: on by default, with the first paid model as the target.
        # A free model that stops mid-task is a cost problem the operator asked us to stop.
        models = source.get("models")
        return {"enabled": True,
                "fallback": _first_paid(models) if isinstance(models, (list, tuple)) else ""}

    if isinstance(config, dict):
        nested = config.get("smart_model_routing")
        if isinstance(nested, dict):
            return _block_from(nested)
        if any(key in config for key in ("models", "free_guard", "preferred_models_only",
                                         "strategy", "enabled")):
            return _block_from(config)
        return {}
    try:
        from hermes_cli.config import load_config
        routing = (load_config() or {}).get("smart_model_routing")
        if isinstance(routing, dict):
            return _block_from(routing)
    except Exception:
        pass
    return {}


def guard_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    block = guard_config(cfg)
    return bool(block.get("enabled", True))


def reset_hour(cfg: Optional[Dict[str, Any]] = None) -> int:
    try:
        hour = int(guard_config(cfg).get("reset_hour", DEFAULT_RESET_HOUR))
    except (TypeError, ValueError):
        return DEFAULT_RESET_HOUR
    return hour if 0 <= hour <= 23 else DEFAULT_RESET_HOUR


def configured_fallback(cfg: Optional[Dict[str, Any]] = None) -> str:
    return str(guard_config(cfg).get("fallback") or "").strip()


def _first_paid(models: Sequence[Any]) -> str:
    for entry in models:
        name = entry.get("name") if isinstance(entry, dict) else entry
        name = str(name or "").strip()
        if name and not is_free_model(name):
            return name
    return ""


# --------------------------------------------------------------------------- identity
def is_free_model(name: Any) -> bool:
    """True when the name marks the model as a free tier.

    Naming, not price: a paid model can be priced at zero in a stale catalog, but the
    ``-free`` suffix is what the provider's free tier is actually called.
    """
    text = str(name or "").strip().lower()
    return bool(text) and any(text.endswith(sfx) for sfx in _FREE_SUFFIXES)


# --------------------------------------------------------------------------- state
def _state_dir() -> Path:
    base = os.environ.get("HERMES_STATE_DIR")
    root = Path(base) if base else Path.home() / ".hermes" / "state"
    return root


def state_path() -> Path:
    return _state_dir() / "free_model_guard.json"


def events_path() -> Path:
    return _state_dir() / "free_model_guard_events.jsonl"


def window_key(now: Optional[datetime] = None, hour: Optional[int] = None) -> str:
    """The quota window a moment belongs to: the local date, rolled at *hour*.

    A provider reset at 02:00 means 01:30 belongs to the previous day's window, so the
    exhausted flag does not lift seven minutes before the allowance actually resets.
    """
    moment = now or datetime.now()
    boundary = int(DEFAULT_RESET_HOUR if hour is None else hour)
    return (moment - timedelta(hours=boundary)).strftime("%Y-%m-%d")


def window_end(now: Optional[datetime] = None, hour: Optional[int] = None) -> str:
    """ISO timestamp when the current window ends (next reset boundary)."""
    moment = now or datetime.now()
    boundary = int(DEFAULT_RESET_HOUR if hour is None else hour)
    start = moment - timedelta(hours=boundary)
    end = datetime(start.year, start.month, start.day) + timedelta(days=1, hours=boundary)
    return end.isoformat(timespec="seconds")


def _read_state() -> Dict[str, Any]:
    try:
        with open(state_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(state: Dict[str, Any]) -> None:
    path = state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        logger.debug("free-model guard: state write failed", exc_info=True)


def _event(kind: str, **fields: Any) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": kind}
    record.update({k: v for k, v in fields.items() if v is not None})
    try:
        path = events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        logger.debug("free-model guard: event append failed", exc_info=True)


def _current(state: Dict[str, Any], cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """State for the live window; rolls the exhausted flags when the window changes."""
    key = window_key(hour=reset_hour(cfg))
    if state.get("window") != key:
        # A new window clears every exhausted flag: the allowance was reset by the provider,
        # so carrying a ban forward would strand the operator on a paid model for no reason.
        state["window"] = key
        state["models"] = {}
    if not isinstance(state.get("models"), dict):
        state["models"] = {}
    return state


# --------------------------------------------------------------------------- API
def record_usage(model: Any, provider: Any, tokens: int = 0,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
    """Count a completed request (and its tokens) against the model's window."""
    name = str(model or "").strip()
    if not name:
        return
    try:
        state = _current(_read_state(), cfg)
        entry = state["models"].setdefault(name, {})
        entry["provider"] = str(provider or entry.get("provider") or "")
        entry["requests"] = int(entry.get("requests") or 0) + 1
        entry["tokens"] = int(entry.get("tokens") or 0) + max(0, int(tokens or 0))
        entry["last_used"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_state(state)
        _event("usage", model=name, provider=provider, tokens=int(tokens or 0),
               requests=entry["requests"])
    except Exception:
        logger.debug("free-model guard: usage record failed", exc_info=True)


def is_exhausted(model: Any, cfg: Optional[Dict[str, Any]] = None) -> bool:
    name = str(model or "").strip()
    if not name:
        return False
    try:
        state = _current(_read_state(), cfg)
        entry = state["models"].get(name)
        return bool(isinstance(entry, dict) and entry.get("exhausted_at"))
    except Exception:
        return False


def mark_exhausted(model: Any, provider: Any, reason: str,
                   cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Ban a FREE model until the window rolls. Returns True when this call set the flag.

    Paid models are deliberately out of scope: their rate limits are transient and the
    existing cooldown/failover machinery already handles them. Only a free tier has a
    daily allowance that is genuinely gone until the reset.
    """
    name = str(model or "").strip()
    if not name or not is_free_model(name):
        return False
    try:
        state = _current(_read_state(), cfg)
        entry = state["models"].setdefault(name, {})
        already = bool(entry.get("exhausted_at"))
        entry["provider"] = str(provider or entry.get("provider") or "")
        entry["exhausted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry["reason"] = str(reason or "quota")
        entry["until"] = window_end(hour=reset_hour(cfg))
        _write_state(state)
        if not already:
            _event("exhausted", model=name, provider=provider, reason=reason,
                   until=entry["until"])
            logger.warning("free model %s exhausted (%s); blocked until %s",
                           name, reason, entry["until"])
        return not already
    except Exception:
        logger.debug("free-model guard: mark exhausted failed", exc_info=True)
        return False


def exhausted_models(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    try:
        state = _current(_read_state(), cfg)
        return sorted(n for n, e in state["models"].items()
                      if isinstance(e, dict) and e.get("exhausted_at"))
    except Exception:
        return []


def filter_exhausted(options: Sequence[Any],
                     cfg: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], List[str]]:
    """Drop exhausted free models from a candidate pool; return (kept, dropped names)."""
    if not guard_enabled(cfg):
        return list(options), []
    blocked = set(exhausted_models(cfg))
    if not blocked:
        return list(options), []
    kept, dropped = [], []
    for opt in options:
        name = str(getattr(opt, "name", "") or "")
        if name in blocked:
            dropped.append(name)
        else:
            kept.append(opt)
    return kept, dropped


def switch_off_exhausted(agent: Any, reason: Any = None) -> bool:
    """Mid-turn rescue: the current (free) model just failed — move to the paid fallback.

    Called from the failover path with the provider's classification, so it never fires on
    a slow response. Returns True when it actually moved the agent.
    """
    cfg: Optional[Dict[str, Any]] = None
    try:
        name = str(getattr(agent, "model", "") or "").strip()
        provider = str(getattr(agent, "provider", "") or "").strip()
        if not is_free_model(name) or not guard_enabled(cfg):
            return False
        reason_text = getattr(reason, "value", reason)
        reason_text = str(reason_text or "").strip().lower()
        if reason_text not in _QUOTA_REASONS:
            return False
        target = configured_fallback(cfg) or _first_paid(_load_routing_models())
        if not target or target == name:
            # No paid target configured: record the exhaustion so routing stops choosing the
            # free model, but let the normal fallback chain decide what happens next.
            mark_exhausted(name, provider, reason_text, cfg)
            return False
        mark_exhausted(name, provider, reason_text, cfg)
        from agent.agent_runtime_helpers import switch_model
        switch_model(agent, target, provider or None)
        _event("switch", model=target, provider=provider, reason=reason_text,
               **{"from": name})
        logger.warning("free model %s hit its limit (%s); switched to %s",
                       name, reason_text, target)
        return True
    except Exception:
        logger.debug("free-model guard: mid-turn switch failed", exc_info=True)
        return False


def _load_routing_models() -> List[Any]:
    try:
        from hermes_cli.config import load_config
        routing = (load_config() or {}).get("smart_model_routing")
        if isinstance(routing, dict):
            models = routing.get("models")
            return list(models) if isinstance(models, (list, tuple)) else []
    except Exception:
        pass
    return []


def status(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything the CLI needs to explain the guard, computed fresh from disk."""
    state = _current(_read_state(), cfg)
    models = {n: e for n, e in (state.get("models") or {}).items() if isinstance(e, dict)}
    return {
        "enabled": guard_enabled(cfg),
        "window": state.get("window") or window_key(hour=reset_hour(cfg)),
        "resets_at": window_end(hour=reset_hour(cfg)),
        "fallback": configured_fallback(cfg),
        "exhausted": sorted(n for n, e in models.items() if e.get("exhausted_at")),
        "usage": {n: {"requests": int(e.get("requests") or 0),
                      "tokens": int(e.get("tokens") or 0),
                      "last_used": e.get("last_used"),
                      "exhausted_at": e.get("exhausted_at")}
                  for n, e in sorted(models.items())},
        "events_path": str(events_path()),
    }
