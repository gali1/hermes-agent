#!/usr/bin/env python3
"""Lint Hermes skills for the defect class that made an agent take a 4-step detour.

Three failure modes, all mechanical enough to detect with high precision:

  P1 BLANKET-BAN     "do not use/fall back to <tool>" with no carve-out for the case
                     the tool actually handles. One agent read such a line and
                     hand-rolled SQL for a question the tool answers directly.
  P2 POSIX-ONLY      commands that cannot run as written on Windows (sqlite3 CLI,
                     /home/ paths, bash heredoc `python3 - <<EOF`, $HOME) with no
                     platform note anywhere in the file.
  P3 NO-NATIVE-HINT  skill drives raw files/DBs and never tells the reader to try the
                     native Hermes tool first.

Read-only. Prints counts and the worst offenders; writes nothing.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.hermes/skills"))

BAN = re.compile(
    r"(?:do not|don't|never|avoid)\s+(?:fall back to|use|call|rely on)\s+[`']?([a-z_][a-z0-9_]{2,})",
    re.I,
)
CARVE_OUT = re.compile(r"except when|unless the question|for (?:last|recent)|carve|pass limit", re.I)

POSIX_PATS = {
    "sqlite3-cli": re.compile(r"(?<![\w/])sqlite3\s+[~\w$/\"]", re.M),
    "home-path": re.compile(r"/home/[A-Za-z_]"),
    "bash-heredoc": re.compile(r"python3?\s+-\s*<<\s*'?EOF'?"),
    "dollar-home": re.compile(r"(?<!\w)\$HOME/"),
    "apt-get": re.compile(r"\bapt(?:-get)?\s+install\b"),
    "systemctl": re.compile(r"\bsystemctl\s+(?:restart|start|stop|status)\b"),
}
PLATFORM_NOTE = re.compile(r"windows|%LOCALAPPDATA%|cross-platform|platform note", re.I)

RAWSTORE = re.compile(r"state\.db|\.sqlite|/etc/nginx|\.json\b.*(?:grep|jq)|sqlite3", re.I)
NATIVE_HINT = re.compile(
    r"native tool|use the tool|try the tool|tool first|prefer the (?:native|tool)|"
    r"session_search|browser_|read_file|search_files|terminal\(",
    re.I,
)


def main() -> int:
    files = sorted(ROOT.rglob("SKILL.md"))
    hits = {"P1": [], "P2": [], "P3": []}
    scanned = 0
    for f in files:
        if ".archive" in f.parts:
            continue
        scanned += 1
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(f.relative_to(ROOT))

        # P1: a ban whose same paragraph carries no carve-out
        for m in BAN.finditer(t):
            para_start = t.rfind("\n\n", 0, m.start())
            para_end = t.find("\n\n", m.end())
            para = t[para_start:para_end if para_end != -1 else len(t)]
            if not CARVE_OUT.search(para):
                hits["P1"].append((rel, m.group(0)[:70]))
                break

        # P2: posix patterns with no platform note anywhere in the file
        if not PLATFORM_NOTE.search(t):
            found = [k for k, rx in POSIX_PATS.items() if rx.search(t)]
            if found:
                hits["P2"].append((rel, ",".join(found)))

        # P3: raw store work with no hint toward the native tool
        if RAWSTORE.search(t) and not NATIVE_HINT.search(t):
            hits["P3"].append((rel, "raw-store, no native-tool hint"))

    print(f"scanned SKILL.md: {scanned}\n")
    for k, label in (("P1", "blanket ban, no carve-out"),
                     ("P2", "POSIX-only, no platform note"),
                     ("P3", "raw store, no native-tool hint")):
        print(f"{k} {label}: {len(hits[k])}")
    print()
    for k in ("P1", "P2", "P3"):
        if hits[k]:
            print(f"--- {k} sample (first 12 of {len(hits[k])}) ---")
            for rel, why in hits[k][:12]:
                print(f"  {rel}  [{why}]")
            print()
    Path("/tmp/skill-lint.json").write_text(json.dumps(hits, indent=1))
    print("full list: /tmp/skill-lint.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
