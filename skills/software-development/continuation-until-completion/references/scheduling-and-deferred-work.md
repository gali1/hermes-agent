# Scheduling and deferred work

Depth for three related situations: proving that a scheduler really runs unattended, running an agent
unattended once the user has authorised it, and queueing a user-requested task for later without letting it
pre-empt the mission.

## Probe the scheduler without spending tokens

Before claiming anything about unattended continuation, prove the trigger fires. A script-only job does that at
zero model cost:

```bash
# ~/.hermes/scripts/<mission>-status.sh — the job body, exits 0 so a state of "work remains" is not an error
#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/.hermes/missions/<slug>" || { echo "mission directory missing"; exit 0; }
echo "== mission state $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 mission_status.py status || true
python3 mission_status.py next || true
```

```text
cronjob_manage(action="create", name="<mission>-probe", schedule="in 2m", repeat=1,
               no_agent=True, script="<mission>-status.sh", deliver="local")
```

Then verify, in this order — each is a separate fact:

1. the job's `state` moves from `scheduled` to fired (`cronjob_manage(action="list")`);
2. an artefact exists under `~/.hermes/cron/output/<job_id>/` with a timestamp;
3. its contents show the *real* mission state, not a placeholder;
4. remove the job (`action="remove"`) so no duplicate lingers.

`deliver="local"` keeps the probe out of the user's chat. `no_agent=True` is what makes it free: the scheduler
runs the script and delivers stdout verbatim.

What this proves: scheduling, execution, state recovery and reporting work unattended. What it does **not**
prove: that an *agent* continues unattended — that needs a job with `prompt` + `skills`, which spends the user's
tokens on every fire and therefore needs explicit authorization. Say which of the two you tested.

## Running an agent unattended

Once the user says to enable it, the job needs the skill loaded, not merely a prompt that describes the skill:

```text
cronjob_manage(action="create", name="<mission>-continuation",
               skills=["continuation-until-completion"],
               workdir="/abs/path/to/repo",        # anchors file/terminal tools and injects repo context
               continuity=True,                    # each fire sees its own previous output → no repeat work
               schedule="every 2h", repeat=6,      # bounded: 6 fires, not forever by accident
               deliver="origin",                   # the report lands where the user asked
               enabled_toolsets=["terminal", "files"],
               prompt=<the procedure + guardrails>)
```

Prompt requirements, because a cron session cannot ask questions:

- recover state first (`mission_status.py status` / `check` / `next`) and refuse to redo `COMPLETE` rows;
- execute exactly ONE task per fire, so a run is reviewable and reversible;
- the git prohibitions verbatim: no `push`, `reset`, `clean`, `stash`, `checkout -- .`, no `git add -A`/`git add .`;
- `git status --porcelain` before editing, and stop-and-report if the tree is dirty in ways the run did not create
  (the user or another session may be mid-edit);
- secret scan before any commit; no commit without an observed test run; targeted tests by default, full suite only
  for cross-cutting changes (it costs minutes and tokens);
- on a genuine blocker, do one independent task and name the blocker rather than inventing a credential or host;
- a hard cap on the reply length (a few lines: what, evidence, next, blocked) so a tick is not a wall of text.

Verify the agent path with **one manual fire** carrying a read-only override:

```text
cronjob_manage(action="run", job_id="<id>",
               prompt="VERIFICATION FIRE — read-only: run mission_status.py status and next, state the next
                       action and the repo state (branch, dirty). Do NOT modify or commit anything.")
```

`action="run"` returns immediately and the outcome re-enters the conversation later as its own message; do not
poll for it. The override applies to that fire only and does not change the stored prompt.

Record in the checkpoint: the job id, cadence, `repeat`, and the exact disable call
(`cronjob_manage(action="pause", job_id="...")`). A future session must be able to stop an unattended job
without rediscovering it.

## Verify what a run claims before repeating it

A cron run's report is a self-report, and so is a delegated child's. Re-derive the load-bearing numbers:

| claim | re-derive with |
|---|---|
| "shipped in commit X" | `git log -1 --format='%h %s'`, `git show --stat HEAD`, `git status --porcelain` (must be clean) |
| "generated doc is current" | the generator's own `--check` subcommand, read its exit code |
| "N tests passed" | run the cited nodes, or the whole suite on a clean tree — then quote the number you saw |
| "coverage moved from A to B" | recompute the counts from the artefact itself (e.g. `len(json['rows'])`), not the prose |

If a claim cannot be reproduced, say so plainly and demote the ledger row to `VERIFY` rather than carrying the
run's optimism into the next report.

## Queueing a task for later

A queued task needs both a ledger row and (when a scheduler exists) a job, because they answer different
questions: the row stops the work being forgotten, the job is what actually starts it.

- Ledger row: lowest priority, evidence field carries the user's constraints and the reason it is deferred.
- Job: one-shot, at the stated time, `deliver` back to the user, toolsets restricted to what the task needs
  (`["terminal", "files"]` for repo work — no browser, no web).

Write the prompt as if the executing session knows nothing (it does): the source path, the target layout, the
verification commands, and the prohibitions. A prompt that says "ship the skill" gets `git add -A`; a prompt that
says "stage only these paths, never `git add -A`, stop and report if the branch diverged" gets a clean commit.

Include an explicit escape hatch — "if this would turn into a merge, conflict resolution or a push, stop and
report" — because a scheduled run cannot ask a question and will otherwise improvise.

## Ordering with mission work

The deferred row must sort below every substantive mission task (a numeric priority well past the real gaps).
If it does not, `mission_status.py next` starts answering with the side quest, and the mission stalls behind
something the user explicitly deprioritised.
