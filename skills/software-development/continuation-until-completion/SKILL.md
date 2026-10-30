---
name: continuation-until-completion
description: Resume a multi-turn mission from durable state until done.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [continuation, mission, checkpoint, recovery, autonomy]
    related_skills: [dynamic-workflow, watchers, hermes-agent]
---

# Continuation until completion

A pause is not completion. A progress report is not execution. This skill is the procedure for resuming a long
mission from durable state instead of from conversational memory, and for refusing to call it done early.

## When to use

- The user says "continue", "keep going", "until complete", or resumes a mission from an earlier session.
- Work spans more turns than one context window can hold.
- A multi-deliverable task has items in different states (done / partial / blocked).

Not for: a single self-contained request, or a task whose whole state fits in the current turn.

## The state, not the memory

Every mission keeps four files in `~/.hermes/missions/<slug>/`:

| file | role |
|---|---|
| `ledger.json` | machine-readable task list: id, guide, section, status, evidence, next action, priority |
| `ledger.md` | rendered view of the same data, for humans; regenerate, never hand-edit |
| `checkpoint.md` | the resume brief: mission, done, in progress, remaining, blockers, next action, resume condition |
| `mission_status.py` | the state machine: `status`, `next`, `blocked`, `check` |

Statuses are fixed: `COMPLETE` (verified, evidence named), `PARTIAL` (deliverable exists, a named part does not),
`BLOCKED` (needs something outside this machine), `VERIFY` (believed, unproven), `TODO`, `IN_PROGRESS`.

**Never** mark something `COMPLETE` because code compiles, a file exists, or a test was written. `COMPLETE`
requires an observed result.

## Procedure

1. **Recover.** `python3 ~/.hermes/missions/<slug>/mission_status.py status`. If it exits 1, work remains — do not
   re-plan from scratch and do not re-do `COMPLETE` rows.
2. **Integrity first.** `mission_status.py check`. A `BLOCKED` row with no named operator action is not a blocker,
   it is unfinished work; a `COMPLETE` row with no evidence is not complete.
3. **Pick the next action.** `mission_status.py next` — explicit priority, then furthest-along. Execute *that*,
   not the easiest adjacent thing. Priority is authored in the ledger, not inferred from status: substantive
   gaps outrank roll-ups, roll-ups outrank bookkeeping, so `next` never answers "update the register".
4. **Execute and verify.** Run the real command, read the real output. Record the evidence string in the ledger
   row (the tuple list in `mission_ledger.py`, then regenerate).
5. **Update status honestly.** Partial progress that leaves a named requirement unmet is `PARTIAL` with the
   unmet part spelled out — never `COMPLETE` plus a caveat in prose. When the unmet part is a separate piece of
   work, split it into its own **child row** first (`G2-17-MCP1`, `G2-17-MCP2`) and leave the umbrella open: an
   umbrella may only close when every child is `COMPLETE` or `BLOCKED` with a named operator action. Closing the
   umbrella while a child is open hides the gap behind one word, and the next session reads it as finished.
6. **Checkpoint before yielding.** Write `checkpoint.md` (format below) whenever you pause, hand back, or expect
   an interruption. Update it before a long background run, not after.
7. **Repeat.** Only when every row is `COMPLETE` or documented `BLOCKED` do you write the completion report.

## Decision rule

| situation | action |
|---|---|
| Authorized, meaningful progress possible with current tools | Do it now. No question, no plan restatement. |
| A tool/endpoint/file is missing | Try a supported alternative, then record the limitation. |
| A step failed | Diagnose, apply a reasonable correction, verify. Never silently mark done. |
| One task blocked, others independent | Work the others; record the blocker with its operator action. |
| Ambiguous but reversible and low-risk | Take the most defensible reading, record the assumption, proceed. |
| Needs information only the user has, an unavailable credential, or an irreversible consequential choice | Finish all independent work, checkpoint, then ask **one** question saying exactly what is needed and why. |
| Everything verified | Final verification, then completion report. |

Manufacturing a blocker because the next step needs reasoning, another tool call, or a test run is a failure of
this skill, not a valid stop.

## Checkpoint format (write it verbatim)

```markdown
# Checkpoint — <mission>

**Mission:** one sentence.
**Governing instructions:** paths/refs to the guides, with precedence.
**Completed and verified:** bullet list, each with its evidence (command + result, commit, URL).
**In progress:** the task id and its latest verified state.
**Remaining:** ordered list, ids from the ledger.
**Blockers:** each with the exact issue, what was tried, and the operator action required.
**Next action:** `<ledger id>` — one executable step (a command, a file, a decision).
**Resume condition:** what must exist before work continues.
**Last updated:** ISO timestamp + host.
```

A checkpoint that another agent cannot resume from is a status update, not a checkpoint. Test it: open a fresh
session, read only the checkpoint and the ledger, and see whether the next action is unambiguous.

Lead the next action with its ledger id (`G2-16 — in <repo>: run …`). The id is what lets a resuming session match
the checkpoint to the register mechanically; without it the checkpoint and `mission_status.py next` can drift apart
and nothing notices.

## Deferred and user-queued work

When the user hands you a task and says "later", "at the end of the day", "when convenient", or "don't do this
now":

1. Add a row at the **lowest** priority so it cannot pre-empt mission work, with the instruction and its
   constraints recorded in the evidence field.
2. If the environment has a scheduler, also create a **one-shot** job for the stated time, and put the guardrails
   in the job prompt itself — a scheduled run is a fresh session with no chat context and cannot ask questions.
3. Do not touch the target now. "Queued" means queued; doing it early is exactly the interruption the user asked
   you to avoid.

State the guardrails explicitly rather than relying on good judgement at fire time: which paths may be staged,
which git commands are forbidden (`git add -A`, `stash`, `reset`, `clean`), what to do on divergence (stop and
report, never resolve), and what "done" looks like (`git show --stat HEAD` lists only the new paths and the
pre-existing dirty entries are still there). `references/scheduling-and-deferred-work.md` has the full recipe and
the zero-token scheduler probe.

## Unattended continuation

A recurring job is the only thing that continues work while nobody is watching, and it is legitimate only once
the user has authorised the token spend. When they have:

- Create it with the skill loaded, not just a prompt:
  `cronjob_manage(action="create", skills=["continuation-until-completion"], workdir=<repo>, continuity=True,
  schedule="every 2h", repeat=<n>, deliver="origin", enabled_toolsets=["terminal", "files"])`. `continuity`
  lets each fire see its own previous output so a run does not redo what the last one finished; a bounded
  `repeat` stops it ticking forever by accident; a narrow toolset stops it wandering.
- Put the guardrails in the prompt, because a scheduled run is a fresh session that cannot ask questions: no
  push/reset/clean/stash/`git add -A`, check `git status --porcelain` before editing and stop on foreign edits,
  secret scan before any commit, no commit without an observed test run, and on a genuine blocker do one
  independent task rather than inventing the missing credential.
- Verify the agent path with **one** manual fire whose `prompt` override is read-only ("report the next action
  and the repo state; do not modify anything"). The script probe in the references proves the trigger fires;
  this proves an agent actually continues.
- **Cadence is anchored to the last run, not to now.** `every 2h` means "two hours after the previous fire", so
  right after a manual run the next slot can be almost two hours out — that is the schedule working, not a bug.
  When the user wants work to start now, fire it with `cronjob_manage(action="run", job_id=...)` and say plainly
  that the recurring slot is unchanged.
- **The shortest safe interval is the run duration plus a gap.** Below that, two runs occupy the same working tree.
  Do not shorten the interval to look responsive; shorten it only after exclusion is enforced at a boundary, and
  say what the faster cadence costs (one full agent turn per slot, even when the run finds nothing to do).
- Record the job id and the exact disable call (`cronjob_manage(action="pause", job_id=...)`) in the checkpoint,
  so a later session can stop it without hunting for it.

**A run's report is a self-report.** Before repeating its numbers, re-derive them: `git log -1` and the commit's
`--stat`, the generator's `--check`, the tests it cited, and the headline suite count on a clean tree. A run that
says "321 passed" may be right — the number only becomes yours once you have seen it. Cron runs and delegated
children both report success the way a contractor reports a finished roof.

**Verify the integration, not the component.** A run can build every piece of a capability and still not deliver it.
Check the claim from the user's side: if it says tools are "registered in the registry", run the command that lists
the registry; if it says an endpoint works, call it; if it says a gate is armed, trip it. A client library plus a CLI
verb is not the same product as an agent that can use it, and the difference is invisible in a test count.

**Enforce exclusion below the prompt.** A step in a job prompt is advisory: a run will skip "acquire the lock first"
and commit anyway, and nothing notices until two writers interleave. Put mutual exclusion at a boundary the agent
cannot bypass — a pre-commit hook that refuses to commit while another party holds the lock (the owner passes via an
env token). Prompt-level guards catch mistakes; boundary-level guards catch non-compliance.

**Check that the gate is installed before trusting it.** A hook, cron entry, or scan that is configured but not wired
silently protects nothing — `git config core.hooksPath` unset means `git commit` runs no hooks at all, and the suite
stays green while the secret scan never executes. Verify the mechanism fires (trip it deliberately) rather than
reading its configuration file.

## Mirroring a diverged remote

A mission that runs for days will drift from its upstream. Reconciling is routine; doing it carelessly is how both
sides' work is lost.

1. **Fetch the real refs before concluding anything is missing.** `git fetch <remote> '+refs/heads/*:refs/remotes/srv/*'`,
   then read `refs/remotes/srv/*`. A `git bundle` someone made earlier may omit branches, so "that ref does not
   exist" derived from a bundle is not evidence — a ref declared `BLOCKED` for being absent can turn out to be one
   fetch away, and the register then carries a false blocker that outlives the session.
2. **Measure the divergence in both directions** before reading any diff: `git rev-list --count srv/main..main`,
   `git rev-list --count main..srv/main`, `git log --oneline main..srv/main`, `git diff --stat main...srv/main`.
3. **Take a recovery point**: `git branch backup/pre-mirror-<date>`. Cheap, and the only thing that makes the next
   step reversible.
4. **Trial-merge to see the real shape**, `git merge --no-commit --no-ff srv/main`. Read `git status --short` and
   group the conflicts.
5. **Abort rather than improvise.** A conflict set touching the same feature on both sides (add/add on the same
   module) is a design decision, not a text conflict. `git merge --abort`, keep the backup and the fetched refs, and
   record a child row per group with the decision already made: take theirs and re-apply our safety checks, keep
   both surfaces, rebuild generated assets from source instead of hand-merging them. One run resolves one group.
6. **Never claim "ours is superior in every aspect".** Enumerate remote-only and local-only capabilities from the
   diff and say where upstream leads. A comparison that only lists local wins is a marketing sentence, and it
   survives into reports as fact.

`references/diverged-remote-reconciliation.md` has the fetch-with-credentials recipe, the comparison commands and
the conflict triage table.

## Honest limits

- **A skill cannot wake a dead session.** It encodes how to continue; it does not schedule anything.
- **A cron job is a real scheduler only if it actually runs an agent turn.** Hermes cron can do that; see
  *Unattended continuation* above for the shape and the verification. Never imply autonomy that is not
  configured, and never leave an unbounded recurring job running without saying so.
- **Background processes are not persistence.** A suite running in the background dies with the host; the
  checkpoint is what survives.
- **Context compaction is not memory.** Anything not in the ledger or checkpoint will be lost; write it down
  before the turn that might lose it.

## Failure handling

- `mission_status.py check` failing → fix the ledger before doing work; an inconsistent register misleads the
  next session more than an empty one.
- A test that fails for environmental reasons (no network, no credit, missing binary) → record it as skipped with
  the reason and the exact command to run it; do not delete it to make the suite green.
- Evidence you cannot reproduce → demote the row to `VERIFY` and say why.
- Two rows contradicting each other → resolve by instruction precedence (later guide over earlier where it is
  additive, the user's latest message over everything), then record the resolution in the row's evidence.

## Test scenarios

Run these against the real mission directory, not a mock. Record expected / actual / pass / evidence.

1. **Normal continuation** — `mission_status.py next` returns the highest-priority open task, not a completed one.
2. **Interruption recovery** — a fresh process reading only `checkpoint.md` + `ledger.json` names the same next
   action the checkpoint names.
3. **Completed-task preservation** — `next` never returns a `COMPLETE` row.
4. **Tool failure** — a failed command is recorded with its error and the row stays open.
5. **Missing user input** — `blocked` lists only rows that name an operator action; `PARTIAL` rows never claim to
   be blocked.
6. **Partial completion** — with blockers present, `next` still returns actionable work (`status` exits 1).
7. **Completion refusal** — while any row is not `COMPLETE`, `status` exits 1 and prints `NOT COMPLETE`.
8. **Scheduler and persistence** — a one-shot script job actually fires, reads the durable state, and leaves a
   report; remove it afterwards.

A test passes only when its output was observed. "The file exists" is not a passing test.
