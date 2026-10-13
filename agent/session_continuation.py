"""Persistent, reboot-gated automatic session continuation for the CLI.

A user can arm continuation for the current session with
``/continue-after-reboot``.  The intent is persisted under the profile-scoped
Hermes home in ``session_continuation.json``; it is honored only after the
machine's boot token changes (Linux ``/proc/sys/kernel/random/boot_id``, else
psutil's boot time), and only once.  The bare interactive CLI checks the marker
at startup and resumes the recorded session through the normal ``--resume``
path.

States: ``armed`` (intent established, awaiting a reboot) -> ``pending`` (a
reboot was observed, awaiting the first eligible CLI login) -> ``consumed``
(resumed once) / ``failed`` (resume attempted and failed) / ``cancelled``
(revoked, e.g. by an explicit exit or ``/new``).  A single marker is kept:
arming again replaces it.

Guarantees encoded here:
- No crash, disconnect, terminal-close or atexit path arms intent; only the
  explicit slash command calls :func:`arm`.
- A marker is honored only after its boot token differs from the current one
  (an already-``pending`` marker, written during such a transition, stays
  eligible until consumed).
- At most one resume per marker: :func:`consume` transitions it to ``consumed``
  and consumed/failed/cancelled markers are never returned by
  :func:`evaluate_at_startup`.
- Stale markers expire after :data:`DEFAULT_TTL_SECONDS` (72h from ``armed_at``).
- Missing, unreadable or corrupt state yields ``None`` (corrupt/inconsistent
  files are quarantined as ``<name>.corrupt-<ts>``); reads never raise.

CLI integration: ``cli.py`` calls :func:`evaluate_at_startup` at REPL startup,
``hermes_cli/cli_loops_mixin.py`` owns the slash handlers and the
explicit-termination revoke.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
STATE_FILENAME = "session_continuation.json"
DEFAULT_TTL_SECONDS = 72 * 60 * 60
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")

STATUS_ARMED = "armed"
STATUS_PENDING = "pending"
STATUS_CONSUMED = "consumed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_ALL_STATUSES = frozenset({STATUS_ARMED, STATUS_PENDING, STATUS_CONSUMED, STATUS_FAILED, STATUS_CANCELLED})
_ELIGIBLE_STATUSES = frozenset({STATUS_ARMED, STATUS_PENDING})


def _state_path(home: Optional[Path | str] = None) -> Path:
    base = Path(home) if home is not None else get_hermes_home()
    return base / STATE_FILENAME


def _iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if ts is None else ts, tz=timezone.utc).isoformat()


def _parse_iso(value: Any) -> Optional[float]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


def current_boot_token() -> str:
    """Boot token that changes on every reboot.

    Linux's ``/proc/sys/kernel/random/boot_id`` when readable, else
    ``boottime:<psutil.boot_time()>``; ``"unknown"`` on any error.
    """
    try:
        token = BOOT_ID_PATH.read_text(encoding="utf-8", errors="replace").strip()
        if token:
            return token
    except Exception:
        pass
    try:
        import psutil

        return f"boottime:{int(psutil.boot_time())}"
    except Exception:
        return "unknown"


def _valid_state(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("schema") != SCHEMA_VERSION:
        return False
    if not isinstance(data.get("session_id"), str) or not data["session_id"].strip():
        return False
    if data.get("status") not in _ALL_STATUSES:
        return False
    if _parse_iso(data.get("armed_at")) is None:
        return False
    if not isinstance(data.get("armed_boot_token"), str) or not data["armed_boot_token"]:
        return False
    return True


def _quarantine(path: Path) -> None:
    try:
        target = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
        if target.exists():
            target = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
        path.rename(target)
        logger.warning("Quarantined invalid session-continuation state to %s", target)
    except OSError as exc:
        logger.warning("Could not quarantine invalid session-continuation state %s: %s", path, exc)


def read_state(home: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """The validated marker, or None.

    Missing/unreadable state yields None (fail open); corrupt or inconsistent
    JSON is renamed to ``<name>.corrupt-<ts>`` and yields None.  Never raises.
    """
    path = _state_path(home)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Could not read session-continuation state %s: %s", path, exc)
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        _quarantine(path)
        return None
    if not _valid_state(data):
        _quarantine(path)
        return None
    return data


def _write_state(state: Dict[str, Any], home: Optional[Path | str] = None) -> None:
    atomic_json_write(_state_path(home), state, indent=2, sort_keys=True, mode=0o600, fsync_dir=True)


def arm(session_id: str, *, reason: str = "", home: Optional[Path | str] = None) -> Dict[str, Any]:
    """Persist continuation intent for *session_id* and return the written marker.

    Re-arming replaces any previous marker (single-marker design).  Raises
    ``ValueError`` when *session_id* is empty; a failure here must never leave a
    half-written file (``atomic_json_write`` is temp-file + replace).
    """
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required to arm session continuation")
    state: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "session_id": sid,
        "status": STATUS_ARMED,
        "reason": str(reason or ""),
        "armed_at": _iso(),
        "armed_boot_token": current_boot_token(),
        "attempted_at": None,
        "consumed_at": None,
        "error": None,
    }
    _write_state(state, home)
    return state


def cancel(session_id: Optional[str] = None, *, reason: str = "", home: Optional[Path | str] = None) -> bool:
    """Revoke an eligible marker; True when one was revoked.

    With *session_id* set only that session's marker is revoked; ``None``
    revokes unconditionally (used by ``/cancel-continuation``).  Terminal
    markers (consumed/failed/cancelled) are left untouched.
    """
    state = read_state(home)
    if not state or state.get("status") not in _ELIGIBLE_STATUSES:
        return False
    if session_id is not None and state.get("session_id") != str(session_id):
        return False
    updated = {**state, "status": STATUS_CANCELLED}
    if reason:
        updated["reason"] = str(reason)
    _write_state(updated, home)
    return True


def _transition(session_id: str, expected_status: str, new_status: str, home) -> Optional[Dict[str, Any]]:
    """Re-read and atomically move an eligible marker to *new_status*; None on a lost race."""
    state = read_state(home)
    if not state or state.get("status") != expected_status or state.get("session_id") != session_id:
        return None
    updated = {**state, "status": new_status}
    _write_state(updated, home)
    return updated


def evaluate_at_startup(
    *,
    home: Optional[Path | str] = None,
    session_exists: Optional[Callable[[str], bool]] = None,
    now: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Return ``{"session_id", "armed_at", "reason"}`` when the marker is eligible.

    Eligible means: status ``armed`` with a different boot token than now (a
    reboot happened) or status ``pending``; not expired; and — when
    *session_exists* is provided — the session still exists (an exception from
    the callable is treated as not existing).  On the armed→rebooted transition
    the marker is atomically moved to ``pending`` before returning.  A marker
    whose session is gone is marked ``failed``.  Terminal statuses never fire.
    """
    state = read_state(home)
    if not state:
        return None
    status = state.get("status")
    if status not in _ELIGIBLE_STATUSES:
        return None
    session_id = state["session_id"]
    armed_ts = _parse_iso(state.get("armed_at"))
    if armed_ts is None:
        return None
    now_ts = time.time() if now is None else float(now)
    if now_ts - armed_ts > DEFAULT_TTL_SECONDS:
        cancel(session_id, reason="expired", home=home)
        return None
    if status == STATUS_ARMED:
        if state.get("armed_boot_token") == current_boot_token():
            return None
        transitioned = _transition(session_id, STATUS_ARMED, STATUS_PENDING, home)
        if transitioned is None:
            return None
        state = transitioned
    if session_exists is not None:
        try:
            exists = bool(session_exists(session_id))
        except Exception as exc:
            logger.warning("Session existence check failed for %s: %s", session_id, exc)
            exists = False
        if not exists:
            mark_failed(session_id, "recorded session no longer exists", home=home)
            return None
    return {
        "session_id": session_id,
        "armed_at": state.get("armed_at"),
        "reason": state.get("reason") or "",
    }


def consume(session_id: str, *, home: Optional[Path | str] = None) -> None:
    """Mark the eligible marker for *session_id* consumed (at most one resume per marker)."""
    state = read_state(home)
    if not state or state.get("session_id") != str(session_id) or state.get("status") not in _ELIGIBLE_STATUSES:
        return
    updated = {**state, "status": STATUS_CONSUMED, "consumed_at": _iso(), "error": None}
    _write_state(updated, home)


def mark_failed(session_id: str, error: Any, *, home: Optional[Path | str] = None) -> None:
    """Mark an eligible marker failed; terminal markers are never re-opened."""
    state = read_state(home)
    if not state or state.get("session_id") != str(session_id) or state.get("status") not in _ELIGIBLE_STATUSES:
        return
    updated = {**state, "status": STATUS_FAILED, "attempted_at": _iso(), "error": str(error)}
    _write_state(updated, home)


def status(home: Optional[Path | str] = None) -> Dict[str, Any]:
    """Diagnostics snapshot: state file path, existence, and parsed marker."""
    path = _state_path(home)
    return {"path": str(path), "exists": path.exists(), "state": read_state(home)}
