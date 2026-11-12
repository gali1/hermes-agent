#!/usr/bin/env python3
"""Apply the mechanical fixes from skill-lint: a platform note (P2) and a
prefer-the-native-tool line (P3).

Deliberately does NOT touch P1 (blanket bans): most are legitimate safety
warnings ("Do NOT use nohup"), and rewriting those would remove real protection.
P1 is reported for review instead.

Idempotent: a file that already carries the block is skipped.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(os.path.expanduser("~/.hermes/skills"))
HITS_PATH = Path("/tmp/skill-lint.json")
if not HITS_PATH.exists():
    import subprocess
    import sys

    subprocess.run([sys.executable, str(Path(__file__).with_name("skill-lint.py"))], check=False)
HITS = json.loads(HITS_PATH.read_text())

PLATFORM = """> **Platform note.** Commands here are POSIX-first. On Windows: paths live under
> `%LOCALAPPDATA%\\hermes\\…`, the `sqlite3` CLI is usually absent (use `python -c`
> with the `sqlite3` stdlib and forward slashes), and a `python3 - <<'EOF'` heredoc
> mangles `C:\\…` paths inside Git Bash. Prefer the native Hermes tool when one
> answers the question.
"""

NATIVE = """> **Prefer the native tool first.** Before reading raw files or querying a store
> directly, check whether a Hermes tool already answers it (`session_search` for
> conversation history, `read_file` / `search_files` for files, `browser_*` for
> live pages). Drop to raw access only for what the tool cannot express — counts,
> bulk exports, one-off joins.
"""

MARKERS = ("Platform note.", "Prefer the native tool first.")


def block_for(need_platform: bool, need_native: bool) -> str:
    parts = []
    if need_platform:
        parts.append(PLATFORM)
    if need_native:
        parts.append(NATIVE)
    return "\n".join(parts)


def insert_after_title(text: str, block: str) -> str | None:
    lines = text.split("\n")
    in_fm = False
    for i, line in enumerate(lines):
        if i == 0 and line.strip() == "---":
            in_fm = True
            continue
        if in_fm and line.strip() == "---":
            in_fm = False
            continue
        if not in_fm and line.startswith("# "):
            return "\n".join(lines[: i + 1] + ["", block.rstrip()] + lines[i + 1:])
    return None


def main() -> int:
    need_platform = {rel for rel, _ in HITS["P2"]}
    need_native = {rel for rel, _ in HITS["P3"]}
    targets = need_platform | need_native
    changed, skipped, failed = [], [], []
    for rel in sorted(targets):
        f = ROOT / rel
        try:
            text = f.read_text(encoding="utf-8", errors="surrogateescape")
        except OSError as e:
            failed.append((rel, str(e)))
            continue
        if any(m in text for m in MARKERS) and (
            (rel not in need_platform or "Platform note." in text)
            and (rel not in need_native or "Prefer the native tool first." in text)
        ):
            skipped.append(rel)
            continue
        block = block_for(rel in need_platform, rel in need_native)
        new = insert_after_title(text, block)
        if new is None:
            failed.append((rel, "no '# ' title line to anchor to"))
            continue
        f.write_text(new, encoding="utf-8", errors="surrogateescape")
        changed.append(rel)

    print(f"targets      : {len(targets)}")
    print(f"patched      : {len(changed)}")
    print(f"already had  : {len(skipped)}")
    print(f"skipped/fail : {len(failed)}")
    for rel, why in failed[:8]:
        print(f"  ! {rel} — {why}")
    print(f"P1 left for review: {len(HITS['P1'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
