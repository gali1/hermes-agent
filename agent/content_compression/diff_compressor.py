"""Git-diff compressor (Headroom-derived, stdlib-only).

Lossy and gated: without ``allow_lossy`` AND a ``retrieval_hint`` the input is
returned byte-identical. When enabled it keeps every ``diff --git`` / ``---`` /
``+++`` / ``@@`` header and every ``+`` / ``-`` change line, and caps the
removable parts only:

* context lines per hunk (``max_context_lines``),
* hunks per file (``max_hunks_per_file``),
* files per diff (``max_files``).

Each drop is replaced by a marker carrying the retrieval hint, and the number
of dropped lines is returned. Deterministic and fail-open.
"""

from __future__ import annotations

__all__ = ["compress_diff"]


def compress_diff(
    text,
    *,
    retrieval_hint: str | None = None,
    allow_lossy: bool = False,
    max_context_lines: int = 2,
    max_hunks_per_file: int = 10,
    max_files: int = 20,
) -> tuple[str, int]:
    """Return ``(text, dropped_lines)``; never larger than the input."""
    if not isinstance(text, str) or not text:
        return text, 0
    if not (allow_lossy and retrieval_hint):
        return text, 0

    lines, trailing = _split(text)
    n = len(lines)
    file_starts = [i for i, line in enumerate(lines) if line.startswith("diff --git ")]
    if not file_starts:
        return text, 0

    file_ranges = [
        (start, file_starts[k + 1] if k + 1 < len(file_starts) else n)
        for k, start in enumerate(file_starts)
    ]

    keep = [True] * n
    markers: dict[int, list[str]] = {}
    dropped = 0

    for fi, (start, end) in enumerate(file_ranges):
        if fi >= max_files:
            for i in range(start, end):
                if keep[i]:
                    keep[i] = False
                    dropped += 1
            _add_marker(
                markers,
                start,
                f"... {len(file_ranges) - max_files} files omitted (retrieve: {retrieval_hint})",
            )
            continue

        hunk_starts = [i for i in range(start, end) if lines[i].startswith("@@ ")]
        if not hunk_starts:
            continue
        hunk_ranges = [
            (hstart, hunk_starts[k + 1] if k + 1 < len(hunk_starts) else end)
            for k, hstart in enumerate(hunk_starts)
        ]

        dropped_hunks = 0
        first_dropped_hunk = None
        for hi, (hstart, hend) in enumerate(hunk_ranges):
            if hi >= max_hunks_per_file:
                if first_dropped_hunk is None:
                    first_dropped_hunk = hstart
                dropped_hunks += 1
                for i in range(hstart, hend):
                    if keep[i]:
                        keep[i] = False
                        dropped += 1
                continue

            ctx_kept = 0
            i = hstart + 1
            while i < hend:
                line = lines[i]
                if _is_context_line(line):
                    if ctx_kept < max_context_lines:
                        ctx_kept += 1
                        i += 1
                        continue
                    run_start = i
                    while i < hend and _is_context_line(lines[i]):
                        keep[i] = False
                        dropped += 1
                        i += 1
                    marker_at = i if i < hend else run_start
                    _add_marker(
                        markers,
                        marker_at,
                        f"... {i - run_start} context lines omitted (retrieve: {retrieval_hint})",
                    )
                    continue
                i += 1

        if dropped_hunks:
            _add_marker(
                markers,
                first_dropped_hunk,
                f"... {dropped_hunks} hunks omitted (retrieve: {retrieval_hint})",
            )

    if dropped == 0:
        return text, 0

    out: list[str] = []
    for i in range(n):
        out.extend(markers.get(i, ()))
        if keep[i]:
            out.append(lines[i])
    candidate = "\n".join(out) + ("\n" if trailing else "")
    if len(candidate) >= len(text):
        return text, 0
    return candidate, dropped


def _split(text: str) -> tuple[list[str], bool]:
    if text == "":
        return [], False
    trailing = text.endswith("\n")
    body = text[:-1] if trailing else text
    return body.split("\n"), trailing


def _add_marker(markers: dict[int, list[str]], index: int, marker: str) -> None:
    markers.setdefault(index, []).append(marker)


def _is_context_line(line: str) -> bool:
    """True for removable unified-diff context lines.

    ``+``/``-`` change lines, ``\\ No newline at end of file`` markers, and
    anything unrecognized are never context. The hunk header itself is skipped
    by the caller (the scan starts after it).
    """
    if line.startswith(("+", "-", "\\", "@@", "diff ")):
        return False
    return line.startswith(" ") or line == ""
