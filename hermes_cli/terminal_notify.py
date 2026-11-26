"""Terminal-native desktop notifications: OSC 9 and Warp's OSC 777 CLI-agent protocol.

OSC 9 (``ESC ] 9 ; <body> BEL``): Ghostty, iTerm2, Kitty and WezTerm raise an OS notification;
others drop it. OSC 777 (``ESC ] 777 ; notify ; warp://cli-agent ; <json> BEL``): Warp's
structured CLI-agent protocol (tab status + notification mailbox).

Inside the running CLI, ``HermesCLI._ring_bell`` sends ``notification_sequence()`` through the
prompt_toolkit output on the app loop (a second writer on the tty would splice into an in-flight kitty
pet frame). ``write_tty`` is the no-app path: ``/dev/tty`` because ``patch_stdout``'s wrapper strips raw
escapes, falling back to ``sys.stdout`` when ``/dev/tty`` can't be opened (Windows, no controlling
terminal). Never raises.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys

_C0_AND_DEL = re.compile(r"[\x00-\x1f\x7f]")
_WARP_PROTOCOL_VERSION = 1
# Last Warp release per channel that set WARP_CLI_AGENT_PROTOCOL_VERSION but could not render
# structured payloads (Warp's should-use-structured.sh). Bash compares lexicographically; so do we.
_WARP_LAST_BROKEN = {"stable": "v0.2026.03.25.08.24.stable_05", "preview": "v0.2026.03.25.08.24.preview_05"}


def write_tty(seq: str) -> None:
    """Write raw escapes to /dev/tty, falling back to sys.stdout. Never raises."""
    try:
        with open("/dev/tty", "w", encoding="utf-8") as tty:
            tty.write(seq)
        return
    except OSError:
        pass
    try:
        sys.stdout.write(seq)
        sys.stdout.flush()
    except Exception:
        pass


def osc9(body: str) -> str:
    """OSC 9 sequence with C0 controls and DEL stripped from the body."""
    return f"\x1b]9;{_C0_AND_DEL.sub('', body)}\x07"


def warp_supported(env=None) -> bool:
    """True when running in a Warp build that can render OSC 777 agent payloads."""
    env = os.environ if env is None else env
    client = env.get("WARP_CLIENT_VERSION", "")
    if env.get("TERM_PROGRAM") != "WarpTerminal" or not env.get("WARP_CLI_AGENT_PROTOCOL_VERSION") or not client:
        return False
    return not any(channel in client and client <= last_broken for channel, last_broken in _WARP_LAST_BROKEN.items())


def warp_osc777(event: str, detail: str, session_id: str = "") -> str:
    """OSC 777 ``warp://cli-agent`` notification; ``event`` is ``stop`` or ``permission_request``."""
    try:
        advertised = int(os.environ.get("WARP_CLI_AGENT_PROTOCOL_VERSION", "1"))
    except ValueError:
        advertised = 1
    cwd = os.getcwd()
    payload = {"v": min(advertised, _WARP_PROTOCOL_VERSION), "agent": "hermes", "event": event,
               "session_id": session_id, "cwd": cwd, "project": os.path.basename(cwd)}
    payload["summary" if event == "permission_request" else "response"] = detail[:200]
    return f"\x1b]777;notify;warp://cli-agent;{json.dumps(payload, separators=(',', ':'))}\x07"


def notification_sequence(context: str, *, prompt: bool, session_id: str = "", detail: str = "") -> str:
    """OSC 9 (plus Warp OSC 777 when supported) for a blocking prompt or turn end."""
    seq = osc9(f"Hermes: {context}")
    if warp_supported():
        event = "permission_request" if prompt else "stop"
        seq += warp_osc777(event, detail or context, session_id)
    return seq


# --- OSC 7501: Program Status Protocol -------------------------------------
# ``ESC ] 7501 ; state=...:app=...:msg=... ESC \`` — the agent tells the terminal what it is doing
# (idle, working, done, blocked, error). Herdr, cmux, Ghostty and Rex consume it; every other
# terminal ignores an unknown OSC, so emitting it is safe everywhere. The record is what populates
# an agent sidebar: a program that never reports is just a terminal to those tools.
_OSC7501_END = "\x1b\\"
_OSC7501_KINDS = frozenset({"permission", "question", "auth"})
_OSC7501_OFF = {"0", "false", "no", "off"}


def status_enabled(env=None) -> bool:
    """True unless ``HERMES_OSC_STATUS`` is set to 0/false/no/off."""
    env = os.environ if env is None else env
    return (env.get("HERMES_OSC_STATUS", "1") or "1").strip().lower() not in _OSC7501_OFF


def osc7501(state: str, *, app: str = "hermes", kind: str = "", msg: str = "") -> str:
    """One OSC 7501 record. ``msg`` is base64 (the protocol's encoding for a human line)."""
    fields = [f"state={state}"]
    if kind in _OSC7501_KINDS:
        fields.append(f"kind={kind}")
    safe_app = _C0_AND_DEL.sub("", app or "")
    if safe_app:
        fields.append(f"app={safe_app}")
    if msg:
        fields.append("msg=" + base64.b64encode(msg.encode("utf-8", "replace")).decode("ascii"))
    return "\x1b]7501;" + ":".join(fields) + _OSC7501_END
