---
name: continuation-until-completion
description: Resume unfinished work from a ledger until verified done.
version: 1.1.0
author: Hermes Agent (Gadafi Ali)
license: MIT
metadata:
  hermes:
    tags: [continuation, resume, ledger, checkpoint, mission, guard, handoff, anti-abandonment]
    related_skills: [weekly-review-planning, development-workflow]
---

# Continuation until completion

Purpose: never mistake a pause, a report, or a finished sub-task for completion. Work from a durable ledger,
execute the next actionable task, verify it, persist state before any interruption, and refuse to declare
done while anything is unverified.

## 0. The contract (read before §1)

Four failure modes this skill exists to prevent. Each one is a violation even when the work itself was correct,
because it converts "unfinished" into "silently abandoned" or "unnecessarily stalled":

1. **Announce-and-stop.** Writing "starting T3 now" and ending the turn without doing T3. If a turn contains the
   words that promise work, the same turn must contain the tool calls that do it.
2. **Asking for permission the register already grants.** "Shall I begin?" / "Say go and I'll start." The ledger
   entry *is* the authorisation: `next` names the task, its `action` and its `acceptance`. Asking again burns the
   user's turn and is the single most expensive form of stalling.
3. **Progress-report-only turns.** A turn that restates state and does nothing while actionable work exists.
   Status is not progress; it is only legitimate when nothing is actionable.
4. **Silent abandonment.** Stopping with open work and no record of what was left running.

### The only legal ways to end a turn

- **(a) Verified completion** — the acceptance check ran, real output backs it, and the ledger was updated
  (`done`/`verify` with `--evidence`).
- **(b) A genuine blocker** — `block --needs-user` with a *specific question* (a credential, a permission, a
  destructive action, an ambiguous authorisation boundary). Not "should I proceed?".
- **(c) A recorded handoff** — `handoff --note "..."` naming every open task, when the turn must end for an
  external reason (time, interruption, the user's new instruction).

Anything else is abandonment.

### The ritual (cheap, and it is what makes the difference)

```bash
S=~/.hermes/skills/productivity/continuation-until-completion/scripts/continuation.py
python3 $S --dir <state> next      # what to do — do not re-derive it, do not ask about it
#  ... start / work / verify ...
python3 $S --dir <state> guard     # exit 3 = you may NOT end this turn; exit 0 = wrap-up is legitimate
```

`guard` exits **3** while actionable work is open, printing each task's concrete next action, and **0** only when
the register holds nothing actionable (tasks blocked `--needs-user` are named but do not hold the turn). Run it
as the last thing before finishing a turn; treating exit 3 as "keep going" is the whole discipline.

`handoff` refuses (exit 1) a note that does not name every actionable task — so "I said I was starting and then
stopped" cannot look like progress. `start` records `started`, so `guard --stall-minutes N` can report a task
that has been *in progress* with nothing to show for it.

## When to Use

- A mission spans more than one turn and state must survive pauses, compaction or a new session.
- The user says continue/resume/finish, or asks why work stopped.
- Several tasks share dependencies and one blocked item must not stall the rest.
- You are about to report progress and need to prove whether the work is actually done.

Do not use it for a single-shot task, or as a substitute for an actual scheduler when unattended resumption
is required (see §7).

Hard truth this skill encodes: a skill cannot wake an agent. Only a scheduler or an orchestrator can. If
unattended resumption is required, use `cronjob_manage` (Hermes) or an external runner — and say plainly that
the skill alone does not fire by itself.

## 1. Recover (always first)

1. `python3 scripts/continuation.py resume --dir <state-dir>` — prints mission, verified work, blockers,
   next action and resume condition. If the ledger exists, trust it over memory of the conversation.
2. If no ledger: reconstruct from the active task, the governing documents, `git status`/`git log`, existing
   artifacts and test logs. Never invent a requirement or claim to have read a document you do not have.
3. Record the governing references in the ledger (`--governing`) before doing substantive work.

## 2. Organize

One task per verifiable deliverable, with: `id`, `title`, `source`, `priority` (1 = highest), `deps`,
`status`, `kind` (`work` | `bookkeeping`), `acceptance` (a command or observable check), `needs` (user input /
credential / authorization).

`kind` is what stops the queue from being eaten by bookkeeping: commits, docs and ledger upkeep never run
before substantive work, however low their priority number. Use `bookkeeping` for exactly those.

Status values: `not_started`, `in_progress`, `needs_verification`, `blocked`, `complete`, `waived`.
`waived` requires a written reason and is the only way to legitimately drop a task.

Create the ledger:

```bash
python3 scripts/continuation.py init --dir <state-dir> --file mission.json
```

The mission JSON is a **seed**. After the first `init` the ledger is authoritative: change state with
`start`/`done`/`block`/`verify`, which record evidence and timestamps. Re-running `init --force` from a stale
JSON silently reverts tasks the CLI already completed (a task marked `complete` with evidence will come back
`in_progress`) — re-init only from a JSON you just updated, carrying the existing `status`/`evidence` across.

Splitting a task mid-flight: add `T6a`/`T6b` and keep the parent honest. Marking `T6` complete while its title
still names work you did not do means the remaining half is never picked up again.

## 3. Choose the next action (decision rule)

Ask, in order:

1. Can I make meaningful authorized progress with what I already have? → `next`, then execute it.
2. Is it already in progress? → finish it before starting anything new.
3. Is a task blocked on something outside my control? → record the blocker, take the next independent task
   (`next` skips blocked tasks automatically).
4. Did something fail? → diagnose, correct, re-run, and record the real result. Never mark complete on a
   plausible assumption.
5. Ambiguous but reversible and low-risk? → take the most reasonable reading, record the assumption, continue.
6. Missing user-specific information, credentials, or irreversible authorization? → finish every independent
   task first, then ask exactly one targeted question.
7. Everything complete and verified? → `complete` (it will refuse if anything is open).

Never manufacture a blocker because the next step needs reasoning, another tool call, or a test run.

## 4. Execute

Do the task with real tools and real output. Evidence goes into the ledger as the command plus its result:

```bash
python3 scripts/continuation.py start T3 --dir <state-dir>
python3 scripts/continuation.py done  T3 --dir <state-dir> --evidence "pytest tests/x.py -> 12 passed"
# or, when the task demands a check beyond doing it:
python3 scripts/continuation.py verify T4 --dir <state-dir> --result pass --evidence "curl ... -> 200"
```

`verify` exists because "implemented" and "verified" are different claims. A task whose acceptance check has
not run is `needs_verification`, not `complete`. `start` refuses a task whose dependencies are unfinished, so
the ledger cannot drift into out-of-order work.

Transcribe evidence, never compose it: read the tail of the log and the exit code first (`tail -3 <log>`,
`echo $?`), then paste those numbers into `--evidence`. Writing the string you expect before the output exists
turns the ledger into fiction — a passing-looking `--evidence "246 passed, exit 0"` typed from memory once
recorded a suite that had actually failed (`1 failed, 251 passed, exit 1`). If the numbers contradict the task,
`verify <id> --result fail` and put the task back to work; do not edit the claim.

Joint runs: every mutating command takes a `<state-dir>/.ledger.lock` (stale after 5 min, so a crashed run does
not wedge the mission); `status`, `next` and `resume` never take it. That serialises ledger writes between a
cron run and an interactive session — it does **not** serialise file edits, see §9.

## 5. Persist before any pause

```bash
python3 scripts/continuation.py checkpoint --dir <state-dir> --note "what happens next"
```

Writes `checkpoint.md` next to the ledger with exactly these fields: Mission; Governing instructions;
Completed and verified; In progress; Remaining tasks; Blockers; Next action; **Resume command** (the exact
command plus `# <task id> <title>`); Resume condition (always prefixed with the task id); Last updated.
Every command a checkpoint names carries its task id, so a resuming agent can match it to the register instead
of guessing which task the note belonged to. Also print the same block in the final response when the user is
the only consumer.

Save a checkpoint whenever: a task becomes complete, a blocker is found, a long command is started, or the
session is about to end. Not just when asked.

## 6. Genuine blockers vs ordinary work

A blocker is: absent authorization, a missing credential or permission, an unavailable external dependency, an
irreversible choice only the user can make, or a requirement that cannot be inferred.

Ask with: the exact decision needed, why it is needed, the recommended option, and what will resume on answer.
Keep every independent task moving meanwhile. Do not re-ask what the conversation or the ledger already
answers.

## 7. When unattended continuation is genuinely required

Verify, do not assume:

- Does the mechanism persist past the session? (a file does; a chat message does not)
- Does it trigger without a new user message? (only a scheduler/orchestrator does)
- Can it reach the same files, tools and credentials?
- Can it recover the ledger and detect a previous failure without duplicating work?

If only a skill + ledger are available, state that resumption is manual and give the exact resume command.
Never describe a saved skill or reminder as a background process.

Concrete Hermes setup — job fields that worked, the guardrail prompt, how the model is resolved (and the pin
trap), and how to prove the job actually fired: `references/cron-watchdog.md`.

## 8. Testing this skill's machinery

`scripts/test_continuation.py` is executable evidence, not documentation (24 stdlib tests, each invoking the
real CLI in a subprocess). It covers normal continuation, interruption recovery across processes, completed
task preservation, dependency refusal, tool failure, missing user input, partial completion with a live
blocker, completion refusal, waiver rules, atomic writes, work-before-bookkeeping ordering, the
checkpoint/resume task-id contract and single-writer locking. Run it after any change:

```bash
python3 scripts/test_continuation.py -v
```

A skill change that is not exercised by these tests is unverified. When a scenario is missing for a real
failure you hit, add the test first, then fix the script.

## 9. Pitfalls

- Two runs, one worktree: the ledger lock protects the register, not the code. Before editing, run
  `git status --porcelain` and treat unexpected modifications as another run's work in progress — pick a
  different task instead of racing it. When a write fails loudly mid-run, trust that failure over your own
  summary of the run: a summary can say "done" while a patch never applied.
- Re-running a verified task wastes the session: check the ledger before touching anything.
- Marking `complete` because the edit landed: run the acceptance check or leave it `needs_verification`.
- Letting a blocker stop all work: `next` exists precisely to route around it.
- Reporting progress instead of executing: a report is not a deliverable.
- Trusting memory of state at session start: the ledger is the source of truth; if the two disagree, the
  ledger wins and you re-verify.
- Delegating the work to a scheduled run and then assuming it succeeded: re-run the acceptance check yourself.
  A subagent's final message is a self-report, not evidence.
- Evidence without a locator is unverifiable: for a long suite or a background job also record the log path
  (and the job/run id), or the next session is trusting a result it cannot re-read.
