#!/usr/bin/env python3
"""Ledger engine for the ``continuation-until-completion`` skill. Standard library only.

A mission lives in ``<dir>/ledger.json``; ``<dir>/checkpoint.md`` is the human-readable resume record.

Commands
--------
init        create the ledger from a mission JSON file (refuses to clobber without --force)
add         append one task (auto id)
status      summary counts + the next action
next        the single next actionable task (dependency-aware, blocked-aware)
start       mark a task in_progress
done        mark complete; refuses without --evidence
verify      record an acceptance check: --result pass|fail (fail returns the task to in_progress)
block       mark blocked with a reason; --needs-user when only the user can resolve it
unblock     blocked -> not_started
waive       drop a task with a written reason (the only legitimate way to skip)
checkpoint  write + print the resume record
resume      print everything a fresh session needs
complete    refuse while anything is open; succeed only when all tasks are complete/waived
guard       anti-abandonment check: exits 3 when actionable work is open (do not end the turn)
handoff     end-of-turn record; refuses a note that does not name the open in-progress task

Exit codes: 0 success, 1 refusal (unfinished work), 2 usage/state error,
            3 open actionable work (``guard``: you may not stop here).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("not_started", "in_progress", "needs_verification", "blocked", "complete", "waived")
KINDS = ("work", "bookkeeping")   # bookkeeping (docs, ledger upkeep, commits) never before substantive work
STATUS_RANK = {"in_progress": 0, "needs_verification": 1, "not_started": 2}
FINISHED = ("complete", "waived")
OPEN = ("not_started", "in_progress", "needs_verification", "blocked")


# --------------------------------------------------------------------------- plumbing
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def die(msg: str, code: int = 2) -> "NoReturn":  # noqa: F821 - string annotation keeps py3.8 happy
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def ledger_path(d: str | Path) -> Path:
    return Path(d).expanduser().resolve() / "ledger.json"


def checkpoint_path(d: str | Path) -> Path:
    return Path(d).expanduser().resolve() / "checkpoint.md"


def load(d: str | Path) -> dict:
    p = ledger_path(d)
    if not p.is_file():
        die(f"no ledger at {p} (run `init` first)")
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        die(f"ledger is not valid JSON: {e}")


LOCK_STALE_SECONDS = 300


def lock_path(d: str | Path) -> Path:
    return Path(d).expanduser().resolve() / ".ledger.lock"


def acquire(d: str | Path) -> None:
    """One writer at a time: a cron run and an interactive session must not drive the same ledger together.

    The lock is a file holding the pid; it is stolen when it is stale (a crashed run leaves it behind).
    """
    p = lock_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            age = time.time() - p.stat().st_mtime
            holder = p.read_text().strip() or "?"
        except OSError:
            age, holder = 0, "?"
        if age < LOCK_STALE_SECONDS:
            die(f"another continuation run holds the lock ({p}, pid {holder}, {int(age)}s old); "
                f"wait for it to finish or remove the file if that process is gone")
        p.unlink(missing_ok=True)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        die(f"another continuation run just took the lock ({p})")
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))


def release(d: str | Path) -> None:
    try:
        lock_path(d).unlink(missing_ok=True)
    except OSError:
        pass


def save(d: str | Path, led: dict) -> None:
    """Atomic, 0600: a crash mid-write never leaves a truncated ledger behind."""
    d = Path(d).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    led["updated"] = now()
    fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".ledger-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(led, indent=2) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, ledger_path(d))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def find(led: dict, tid: str) -> dict:
    for t in led["tasks"]:
        if t["id"] == tid:
            return t
    die(f"unknown task {tid!r}; known: {', '.join(t['id'] for t in led['tasks'])}")


def ordered(items: "dict | list[dict]") -> list[dict]:
    """Tasks in priority order. Accepts the ledger or a bare task list (both are used)."""
    tasks = items["tasks"] if isinstance(items, dict) else list(items)
    return sorted(tasks, key=lambda t: (t.get("priority", 99), t["id"]))


def deps_ready(task: dict, led: dict) -> bool:
    finished = {t["id"] for t in led["tasks"] if t["status"] in FINISHED}
    missing = [d for d in task.get("deps") or [] if d not in finished]
    return not missing


def waiting_on(task: dict, led: dict) -> list[str]:
    finished = {t["id"] for t in led["tasks"] if t["status"] in FINISHED}
    return [d for d in task.get("deps") or [] if d not in finished]


def pick_next(led: dict) -> tuple[dict | None, str]:
    """Substantive work before bookkeeping, in-progress before unstarted, then priority.

    ``kind`` outranks ``priority`` on purpose: a commit/docs/ledger task that happens to be priority 1 must not
    push the actual implementation to the back of the queue.
    """
    ready = [t for t in led["tasks"] if t["status"] in STATUS_RANK and deps_ready(t, led)]
    if ready:
        return min(ready, key=lambda t: (0 if t.get("kind", "work") == "work" else 1,
                                        STATUS_RANK[t["status"]], t.get("priority", 99), t["id"])), ""
    open_tasks = [t for t in led["tasks"] if t["status"] in OPEN]
    if not open_tasks:
        return None, "all tasks are complete or waived"
    parts = []
    for t in ordered(open_tasks):
        why = t.get("blocked_reason") or ""
        deps = waiting_on(t, led)
        if deps:
            why = (why + " " if why else "") + f"waiting on {', '.join(deps)}"
        parts.append(f"{t['id']} ({why or t['status']})")
    return None, "every open task is blocked or waiting: " + "; ".join(parts)


def open_tasks(led: dict) -> list[dict]:
    return [t for t in ordered(led) if t["status"] not in FINISHED]


def validate_tasks(tasks: list[dict]) -> None:
    ids = [t.get("id") for t in tasks]
    if len(ids) != len(set(ids)):
        die(f"duplicate task ids: {ids}")
    for t in tasks:
        if not t.get("id") or not t.get("title"):
            die(f"task needs id and title: {t}")
        st = t.get("status", "not_started")
        if st not in STATUSES:
            die(f"{t['id']}: unknown status {st!r} (expected one of {', '.join(STATUSES)})")
        t["status"] = st
        kind = t.get("kind", "work")
        if kind not in KINDS:
            die(f"{t['id']}: unknown kind {kind!r} (expected work or bookkeeping)")
        t["kind"] = kind
    known = set(ids)
    for t in tasks:
        for dep in t.get("deps") or []:
            if dep not in known:
                die(f"{t['id']}: dependency {dep!r} does not exist")


# --------------------------------------------------------------------------- renderers
def render_checkpoint(led: dict, note: str = "", state_dir: str = "") -> str:
    done = [t for t in ordered(led) if t["status"] == "complete"]
    wip = [t for t in ordered(led) if t["status"] == "in_progress"]
    verify = [t for t in ordered(led) if t["status"] == "needs_verification"]
    remaining = [t for t in ordered(led) if t["status"] in ("not_started", "blocked")]
    blockers = led.get("blockers") or []
    nxt, why = pick_next(led)
    lines = [f"# Checkpoint — {led['mission']}", "", f"Last updated: {now()}", "", "## Mission",
             led["mission"], "", "## Governing instructions"]
    lines += [f"- {g}" for g in led.get("governing") or ["(none recorded)"]] or ["- (none recorded)"]
    lines += ["", "## Completed and verified"]
    lines += [f"- {t['id']} — {t['title']} — evidence: {t.get('evidence') or '(none recorded)'}" for t in done] \
        or ["- (nothing verified yet)"]
    lines += ["", "## In progress"]
    lines += [f"- {t['id']} — {t['title']} — {t.get('next_action') or 'continue'}" for t in wip] \
        or ["- (nothing in progress)"]
    if verify:
        lines += ["", "## Implemented, awaiting verification"]
        lines += [f"- {t['id']} — {t['title']} — acceptance: {t.get('acceptance') or '(unspecified)'}"
                  for t in verify]
    lines += ["", "## Remaining tasks"]
    lines += [f"- [{t['status']}] {t['id']} (p{t.get('priority', 99)}) — {t['title']}" for t in remaining] \
        or ["- (none)"]
    lines += ["", "## Blockers"]
    for b in blockers:
        flag = "USER INPUT REQUIRED" if b.get("needs_user") else "external"
        lines.append(f"- {b['id']}: {b['reason']} [{flag}]")
    if not blockers:
        lines.append("- (none)")
    lines += ["", "## Next action",
              (f"{nxt['id']} — {nxt['title']}: {nxt.get('next_action') or 'execute it'}" if nxt else f"none — {why}"),
              "", "## Resume command"]
    lines.append(f"python3 scripts/continuation.py --dir {state_dir or '<state-dir>'} next    # {nxt['id']} {nxt['title']}"
                 if nxt else "(no actionable task: see Blockers)")
    lines += ["", "## Resume condition"]
    lines.append(led.get("resume_condition") or
                 (f"{nxt['id']}: ledger readable and the next action's tools available" if nxt
                  else "ledger readable; resolve the blockers first"))
    if note:
        lines += ["", "## Note", note]
    return "\n".join(lines) + "\n"


def render_resume(led: dict) -> str:
    nxt, why = pick_next(led)
    lines = [f"Mission: {led['mission']}", "Governing instructions:"]
    lines += [f"  - {g}" for g in led.get("governing") or ["(none recorded)"]]
    counts = {s: sum(1 for t in led["tasks"] if t["status"] == s) for s in STATUSES}
    lines.append("Tasks: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    lines.append("Completed and verified:")
    done = [t for t in ordered(led) if t["status"] == "complete"]
    lines += [f"  - {t['id']} — {t['title']} — {t.get('evidence') or '(no evidence recorded)'}" for t in done] \
        or ["  - (none)"]
    wip = [t for t in ordered(led) if t["status"] in ("in_progress", "needs_verification")]
    lines.append("In progress / awaiting verification:")
    lines += [f"  - {t['id']} [{t['status']}] — {t['title']}" for t in wip] or ["  - (none)"]
    lines.append("Blockers:")
    for b in led.get("blockers") or []:
        lines.append(f"  - {b['id']}: {b['reason']}" + ("  <-- USER INPUT REQUIRED" if b.get("needs_user") else ""))
    if not (led.get("blockers") or []):
        lines.append("  - (none)")
    lines.append("Next action: " + (f"{nxt['id']} — {nxt['title']}: {nxt.get('next_action') or 'execute it'}"
                                    if nxt else f"none — {why}"))
    lines.append("Resume condition: " + (led.get("resume_condition") or "ledger readable and tools available"))
    lines.append(f"Last updated: {led.get('updated')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- commands
def cmd_init(args) -> int:
    path = ledger_path(args.dir)
    if path.exists() and not args.force:
        die(f"ledger already exists at {path} (use --force to replace)", 2)
    if args.file:
        src = json.loads(Path(args.file).expanduser().read_text())
        led = {"mission": src.get("mission") or args.mission or "(unnamed mission)",
               "governing": src.get("governing") or args.governing or [],
               "resume_condition": src.get("resume_condition", ""),
               "tasks": src.get("tasks") or [], "blockers": src.get("blockers") or []}
    else:
        led = {"mission": args.mission or "(unnamed mission)", "governing": args.governing or [],
               "resume_condition": "", "tasks": [], "blockers": []}
    if not led["tasks"]:
        die("a mission needs at least one task (--file with a tasks list)")
    for t in led["tasks"]:
        t.setdefault("status", "not_started")
        t.setdefault("priority", 99)
        t.setdefault("deps", [])
    validate_tasks(led["tasks"])
    save(args.dir, led)
    print(f"ledger created: {path} ({len(led['tasks'])} tasks)")
    return 0


def cmd_add(args) -> int:
    led = load(args.dir)
    tid = args.id or f"T{len(led['tasks']) + 1}"
    if any(t["id"] == tid for t in led["tasks"]):
        die(f"task {tid} already exists")
    led["tasks"].append({"id": tid, "title": args.title, "source": args.source or "", "priority": args.priority,
                         "kind": args.kind,
                         "deps": args.deps or [], "status": "not_started", "acceptance": args.acceptance or "",
                         "needs": args.needs or "", "next_action": args.next or ""})
    validate_tasks(led["tasks"])
    save(args.dir, led)
    print(f"added {tid} — {args.title}")
    return 0


def cmd_status(args) -> int:
    led = load(args.dir)
    counts = {s: sum(1 for t in led["tasks"] if t["status"] == s) for s in STATUSES}
    print(f"mission: {led['mission']}")
    print("counts: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    for t in ordered(led):
        print(f"  [{t['status']:18s}] {t['id']:4s} p{t.get('priority', 99)} {t['title']}")
    nxt, why = pick_next(led)
    print("next: " + (f"{nxt['id']} — {nxt['title']}" if nxt else f"none — {why}"))
    return 0


def cmd_next(args) -> int:
    led = load(args.dir)
    nxt, why = pick_next(led)
    if nxt is None:
        print(f"no actionable task: {why}")
        return 1 if open_tasks(led) else 0
    print(f"{nxt['id']}\t{nxt['title']}")
    if nxt.get("next_action"):
        print(f"action: {nxt['next_action']}")
    if nxt.get("acceptance"):
        print(f"acceptance: {nxt['acceptance']}")
    if nxt.get("needs"):
        print(f"needs: {nxt['needs']}")
    return 0


def cmd_start(args) -> int:
    led = load(args.dir)
    t = find(led, args.id)
    if t["status"] == "blocked":
        die(f"{t['id']} is blocked ({t.get('blocked_reason')}); unblock it first")
    missing = waiting_on(t, led)
    if missing:
        die(f"{t['id']} depends on unfinished task(s): {', '.join(missing)}; do those first")
    t["status"] = "in_progress"
    t["started"] = now()
    if args.next:
        t["next_action"] = args.next
    save(args.dir, led)
    print(f"{t['id']} in progress")
    print(f"next: {t.get('next_action') or t.get('acceptance') or 'see the register'}")
    return 0


def cmd_done(args) -> int:
    led = load(args.dir)
    t = find(led, args.id)
    if not args.evidence.strip():
        die(f"{t['id']}: --evidence is required; completion without evidence is not completion")
    t["status"] = "complete"
    t["evidence"] = args.evidence
    t["completed"] = now()
    led["blockers"] = [b for b in led.get("blockers") or [] if b["id"] != t["id"]]
    save(args.dir, led)
    print(f"{t['id']} complete — {args.evidence}")
    return 0


def cmd_verify(args) -> int:
    led = load(args.dir)
    t = find(led, args.id)
    if args.result == "pass":
        if not args.evidence.strip():
            die(f"{t['id']}: a passing verification needs --evidence")
        t["status"] = "complete"
        t["evidence"] = args.evidence
        t["verification"] = "pass"
        t["completed"] = now()
    else:
        t["status"] = "in_progress"
        t["verification"] = "fail"
        t["notes"] = (t.get("notes", "") + f"\nverification failed: {args.evidence}").strip()
    save(args.dir, led)
    print(f"{t['id']} verification {args.result}" + (f" — {args.evidence}" if args.evidence else ""))
    return 0


def cmd_block(args) -> int:
    led = load(args.dir)
    t = find(led, args.id)
    t["status"] = "blocked"
    t["blocked_reason"] = args.reason
    led.setdefault("blockers", [])
    for b in led["blockers"]:
        if b["id"] == t["id"]:
            b.update({"reason": args.reason, "needs_user": bool(args.needs_user)})
            break
    else:
        led["blockers"].append({"id": t["id"], "reason": args.reason, "needs_user": bool(args.needs_user),
                                "since": now()})
    save(args.dir, led)
    print(f"{t['id']} blocked — {args.reason}" + (" (needs user)" if args.needs_user else ""))
    return 0


def cmd_unblock(args) -> int:
    led = load(args.dir)
    t = find(led, args.id)
    if t["status"] != "blocked":
        die(f"{t['id']} is not blocked (status: {t['status']})")
    t["status"] = "not_started"
    t.pop("blocked_reason", None)
    led["blockers"] = [b for b in led.get("blockers") or [] if b["id"] != t["id"]]
    if args.note:
        t["notes"] = (t.get("notes", "") + f"\nunblocked: {args.note}").strip()
    save(args.dir, led)
    print(f"{t['id']} unblocked")
    return 0


def cmd_waive(args) -> int:
    led = load(args.dir)
    t = find(led, args.id)
    if not args.reason.strip():
        die(f"{t['id']}: waiving requires a written --reason")
    t["status"] = "waived"
    t["waived_reason"] = args.reason
    led["blockers"] = [b for b in led.get("blockers") or [] if b["id"] != t["id"]]
    save(args.dir, led)
    print(f"{t['id']} waived — {args.reason}")
    return 0


def cmd_checkpoint(args) -> int:
    led = load(args.dir)
    nxt_id = (pick_next(led)[0] or {}).get("id")
    if args.note:
        # Every checkpoint names its task id, so a resuming agent can match the note to the register.
        led["resume_condition"] = f"{nxt_id}: {args.note}" if nxt_id else args.note
    text = render_checkpoint(led, args.note or "", str(Path(args.dir).expanduser().resolve()))
    p = checkpoint_path(args.dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(p, text)
    save(args.dir, led)
    print(text, end="")
    print(f"\ncheckpoint written: {p}")
    return 0


def atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ckpt-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def cmd_resume(args) -> int:
    led = load(args.dir)
    print(render_resume(led))
    return 0


def is_actionable(t: dict, led: dict) -> bool:
    """Open, owned work whose dependencies are met — the definition ``guard`` and ``next`` share."""
    if t["status"] not in ("not_started", "in_progress", "needs_verification"):
        return False
    return not waiting_on(t, led)


def cmd_guard(args) -> int:
    """The anti-abandonment check. Run it before ending any turn.

    Exit 3 means: actionable work is still open — you may not stop here, and you may not ask the user whether to
    continue (the register already answers that). Exit 0 means the turn may legitimately end: nothing actionable
    is left, or everything left is blocked with ``needs_user``.
    """
    led = load(args.dir)
    action = [t for t in ordered(led) if is_actionable(t, led)]
    needs_user = [b for b in led.get("blockers") or [] if b.get("needs_user")]
    stalled = []
    limit = max(1, args.stall_minutes) * 60
    for t in action:
        if t["status"] != "in_progress" or not t.get("started"):
            continue
        try:
            age = time.time() - datetime.fromisoformat(t["started"]).timestamp()
        except ValueError:
            continue
        if age > limit:
            stalled.append((t, int(age // 60)))

    if not action:
        print("guard: no actionable work left — a wrap-up turn is legitimate")
        for b in needs_user:
            print(f"  awaiting user: {b['id']} — {b.get('reason', '')}")
        return 0

    print(f"guard: {len(action)} actionable task(s) still open — DO NOT end the turn here")
    for t in action:
        kind = t.get("kind", "work")
        print(f"  {t['id']} [{t['status']}/{kind}] {t['title']}")
        line = t.get("next_action") or t.get("acceptance") or ""
        if line:
            print(f"     do: {line}")
    for t, minutes in stalled:
        print(f"  stalled: {t['id']} has been in_progress for {minutes}m with no outcome recorded")
    if needs_user:
        for b in needs_user:
            print(f"  (separately awaiting user: {b['id']} — {b.get('reason', '')})")
    print("resume: python3 continuation.py next && work it; asking the user to continue is a violation "
          "when the register already authorises it")
    return 3


def cmd_handoff(args) -> int:
    """Record a deliberate stop. Refuses a note that ignores an in-progress work task.

    A handoff is the only sanctioned way to end a turn with open work: it must name every in-progress task, so
    "I said I was starting and then stopped" leaves a trace instead of looking like progress.
    """
    led = load(args.dir)
    note = (args.note or "").strip()
    if not note:
        die("handoff needs --note saying exactly what is unfinished and where it resumes")
    # Every actionable task must be named — not only the in-progress ones, because the worst case is
    # announcing a task, never starting it, and stopping: nothing is "in progress" and the work vanishes.
    open_ids = [t["id"] for t in ordered(led) if is_actionable(t, led)]
    missing = [i for i in open_ids if i not in note]
    if missing:
        die(f"handoff note must name the open task(s) {', '.join(missing)}; a stop that does not say what "
            f"was left undone is how work gets dropped (waive or block them instead if they are truly done)", 1)
    led.setdefault("handoffs", [])
    led["handoffs"].append({"at": now(), "note": note, "open": open_ids})
    save(args.dir, led)
    if args.checkpoint:
        text = render_checkpoint(led, note, str(Path(args.dir).expanduser().resolve()))
        atomic_write(checkpoint_path(args.dir), text)
    print(f"handoff recorded ({len(open_ids)} open: {', '.join(open_ids) or 'none'})")
    nxt, why = pick_next(led)
    print("next: " + (f"{nxt['id']} — {nxt['title']}" if nxt else f"none — {why}"))
    return 0


def cmd_complete(args) -> int:
    led = load(args.dir)
    todo = open_tasks(led)
    if todo:
        print(f"REFUSED: {len(todo)} task(s) are not complete and not waived:", file=sys.stderr)
        for t in todo:
            why = t.get("blocked_reason") or t["status"]
            print(f"  - {t['id']} [{why}] {t['title']}", file=sys.stderr)
        print("complete or waive (with a reason) every task before claiming completion", file=sys.stderr)
        return 1
    done = [t for t in ordered(led) if t["status"] == "complete"]
    waived = [t for t in ordered(led) if t["status"] == "waived"]
    print(f"MISSION COMPLETE: {led['mission']}")
    print(f"verified tasks: {len(done)}, waived: {len(waived)}")
    for t in done:
        print(f"  - {t['id']} {t['title']} — {t.get('evidence') or '(no evidence)'}")
    for t in waived:
        print(f"  ~ {t['id']} {t['title']} — waived: {t['waived_reason']}")
    return 0


# --------------------------------------------------------------------------- cli
def build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="continuation.py", description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=os.path.expanduser("~/.hermes/state/continuation"),
                   help="state directory holding ledger.json and checkpoint.md")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="create the ledger")
    i.add_argument("--file", help="mission JSON: {mission, governing, tasks, resume_condition}")
    i.add_argument("--mission")
    i.add_argument("--governing", action="append")
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)

    a = sub.add_parser("add", help="append a task")
    a.add_argument("--id")
    a.add_argument("--title", required=True)
    a.add_argument("--source")
    a.add_argument("--priority", type=int, default=99)
    a.add_argument("--kind", choices=KINDS, default="work")
    a.add_argument("--deps", nargs="*")
    a.add_argument("--acceptance")
    a.add_argument("--needs")
    a.add_argument("--next", dest="next")
    a.set_defaults(fn=cmd_add)

    sub.add_parser("status", help="summary + next").set_defaults(fn=cmd_status)
    sub.add_parser("next", help="the single next actionable task").set_defaults(fn=cmd_next)
    sub.add_parser("resume", help="everything a fresh session needs").set_defaults(fn=cmd_resume)
    sub.add_parser("complete", help="completion gate (refuses while work is open)").set_defaults(fn=cmd_complete)
    g = sub.add_parser("guard", help="anti-abandonment check: exit 3 while actionable work is open")
    g.add_argument("--stall-minutes", type=int, default=30, dest="stall_minutes")
    g.set_defaults(fn=cmd_guard)
    h = sub.add_parser("handoff", help="record a deliberate stop; note must name in-progress tasks")
    h.add_argument("--note", required=True)
    h.add_argument("--checkpoint", action="store_true", help="also rewrite checkpoint.md")
    h.set_defaults(fn=cmd_handoff)

    s = sub.add_parser("start", help="mark in_progress")
    s.add_argument("id")
    s.add_argument("--next", dest="next")
    s.set_defaults(fn=cmd_start)

    d = sub.add_parser("done", help="mark complete (needs evidence)")
    d.add_argument("id")
    d.add_argument("--evidence", required=True)
    d.set_defaults(fn=cmd_done)

    v = sub.add_parser("verify", help="record an acceptance check")
    v.add_argument("id")
    v.add_argument("--result", choices=("pass", "fail"), required=True)
    v.add_argument("--evidence", default="")
    v.set_defaults(fn=cmd_verify)

    b = sub.add_parser("block", help="mark blocked")
    b.add_argument("id")
    b.add_argument("--reason", required=True)
    b.add_argument("--needs-user", action="store_true")
    b.set_defaults(fn=cmd_block)

    u = sub.add_parser("unblock", help="blocked -> not_started")
    u.add_argument("id")
    u.add_argument("--note")
    u.set_defaults(fn=cmd_unblock)

    w = sub.add_parser("waive", help="drop a task with a written reason")
    w.add_argument("id")
    w.add_argument("--reason", required=True)
    w.set_defaults(fn=cmd_waive)

    c = sub.add_parser("checkpoint", help="write + print the resume record")
    c.add_argument("--note")
    c.set_defaults(fn=cmd_checkpoint)
    return p


READ_ONLY = ("status", "next", "resume")


def main(argv: list[str] | None = None) -> int:
    args = build().parse_args(argv)
    if args.cmd in READ_ONLY:
        return args.fn(args)
    acquire(args.dir)
    try:
        return args.fn(args)
    finally:
        release(args.dir)


if __name__ == "__main__":
    raise SystemExit(main())