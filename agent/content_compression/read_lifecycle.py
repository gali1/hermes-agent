"""Read-lifecycle handling for old tool results (Headroom-derived, Phase 3a).

A long transcript accumulates read results the model must no longer trust:
a file read and then modified later is **stale**, and a file read again with
different content is **superseded**. Headroom measures roughly 75% of read
bytes in a long session as stale or superseded. This module rewrites those
old read results to a one-line marker before the compressor's dedup and
summarize passes, so the transcript stops carrying bytes that contradict the
current state of the workspace.

Pure, deterministic, and fail-open: :func:`apply_read_lifecycle` never raises
and returns the input messages unchanged on any error. Only results strictly
before ``boundary`` are rewritten; the protected tail is never touched.
"""

from __future__ import annotations

import json

__all__ = [
    "READ_TOOL_NAMES",
    "WRITE_TOOL_HINTS",
    "extract_path",
    "apply_read_lifecycle",
]

READ_TOOL_NAMES = frozenset({"read_file", "read"})
WRITE_TOOL_HINTS = ("write", "edit", "patch", "create", "delete", "move", "rename", "apply")

_PATH_KEYS = frozenset({"path", "file_path", "filepath", "filename", "file"})

_STALE_MARKER = (
    "[read_file] {path} — stale: modified later by {tool}; "
    "re-read for current contents"
)
_SUPERSEDED_MARKER = "[read_file] {path} — superseded by a newer read of the same file"
_SKIP_PREFIXES = ("[read_file]", "[Duplicate tool output")


def extract_path(args) -> str | None:
    """Return the file path named by a tool call's arguments, or None.

    ``args`` may be a dict or a JSON string. Keys are matched
    case-insensitively against ``path``/``file_path``/``filepath``/
    ``filename``/``file``; the first non-empty string wins. A ``files`` list
    of dicts is also accepted, in which case the first path found in it is
    returned. Never raises.
    """
    try:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                return None
        if not isinstance(args, dict):
            return None

        files_value = None
        for key, value in args.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if lowered in _PATH_KEYS:
                if isinstance(value, str) and value.strip():
                    return value
            elif lowered == "files" and isinstance(value, list):
                files_value = value

        if files_value is not None:
            for item in files_value:
                path = extract_path(item)
                if path:
                    return path
        return None
    except Exception:
        return None


def _is_write_tool(name: str) -> bool:
    """True when ``name`` looks like a file-mutating tool (read tools excluded)."""
    if not isinstance(name, str) or not name:
        return False
    lowered = name.lower()
    if lowered.startswith("read"):
        return False
    return any(hint in lowered for hint in WRITE_TOOL_HINTS)


def _extract_call(tool_call) -> tuple[str, str]:
    """Return ``(name, arguments)`` for a dict- or object-shaped tool call."""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function")
        if not isinstance(fn, dict):
            fn = {}
        name = fn.get("name") or ""
        args = fn.get("arguments")
        return str(name), args if isinstance(args, str) else ""
    fn = getattr(tool_call, "function", None)
    if fn is None:
        return "", ""
    name = getattr(fn, "name", None) or ""
    args = getattr(fn, "arguments", None)
    return str(name), args if isinstance(args, str) else ""


def _extract_call_id(tool_call) -> str:
    if isinstance(tool_call, dict):
        cid = tool_call.get("id")
    else:
        cid = getattr(tool_call, "id", None)
    return cid if isinstance(cid, str) else ""


def apply_read_lifecycle(
    messages: list[dict], boundary: int
) -> tuple[list[dict], int]:
    """Replace stale/superseded read results before ``boundary``.

    Returns ``(new_messages, replacements)``. A read result is *stale* when a
    later write-ish tool call targets the same path, and *superseded* when the
    same path is read again later; stale takes precedence. Results at index
    ``>= boundary`` are never touched, as are non-string, empty, or
    already-marked contents. Deterministic and fail-open: any exception
    returns the original list with a replacement count of 0.
    """
    try:
        if not messages or not isinstance(boundary, int) or boundary <= 0:
            return messages, 0

        call_info: dict[str, tuple[str, str | None]] = {}
        reads_by_path: dict[str, list[int]] = {}
        writes_by_path: dict[str, list[tuple[int, str]]] = {}

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            for tool_call in msg.get("tool_calls") or []:
                name, args = _extract_call(tool_call)
                call_id = _extract_call_id(tool_call)
                path = extract_path(args)
                if call_id:
                    call_info[call_id] = (name, path)
                if not path:
                    continue
                if name in READ_TOOL_NAMES:
                    reads_by_path.setdefault(path, []).append(idx)
                elif _is_write_tool(name):
                    writes_by_path.setdefault(path, []).append((idx, name))

        result = list(messages)
        replacements = 0
        for idx, msg in enumerate(messages):
            if idx >= boundary:
                continue
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            call_id = msg.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in call_info:
                continue
            name, path = call_info[call_id]
            if name not in READ_TOOL_NAMES or not path:
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content:
                continue
            if content.startswith(_SKIP_PREFIXES):
                continue

            later_writes = [w for w in writes_by_path.get(path, ()) if w[0] > idx]
            if later_writes:
                tool = min(later_writes, key=lambda item: item[0])[1]
                marker = _STALE_MARKER.format(path=path, tool=tool)
            elif any(r > idx for r in reads_by_path.get(path, ())):
                marker = _SUPERSEDED_MARKER.format(path=path)
            else:
                continue

            result[idx] = {**msg, "content": marker}
            replacements += 1

        return result, replacements
    except Exception:
        return messages, 0
