#!/usr/bin/env python3
"""Executable evidence for the continuation ledger (stdlib unittest, no pytest needed).

Each test runs the real CLI in a subprocess and inspects the real ledger/checkpoint files, so the
"interruption recovery" case genuinely crosses a process boundary — the same thing that happens when a
session ends and a later invocation resumes.

Run: python3 scripts/test_continuation.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLI = HERE / "continuation.py"


def cli(state: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the CLI once, in its own process, with a clean state directory."""
    return subprocess.run([sys.executable, str(CLI), "--dir", str(state), *args],
                          capture_output=True, text=True, timeout=60)


def init_mission(state: Path, tasks: list[dict], mission: str = "test mission",
                 governing: list[str] | None = None) -> None:
    spec = {"mission": mission, "governing": governing or ["guide A", "guide B"], "tasks": tasks}
    f = state.parent / f"{state.name}-mission.json"
    f.write_text(json.dumps(spec))
    r = cli(state, "init", "--file", str(f))
    assert r.returncode == 0, r.stderr


def ledger(state: Path) -> dict:
    return json.loads((state / "ledger.json").read_text())


def task(state: Path, tid: str) -> dict:
    return next(t for t in ledger(state)["tasks"] if t["id"] == tid)


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="ctl-")
        self.state = Path(self.tmp.name) / "state"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def assertOk(self, r: subprocess.CompletedProcess) -> subprocess.CompletedProcess:
        self.assertEqual(r.returncode, 0, f"stdout={r.stdout!r} stderr={r.stderr!r}")
        return r


def basic_tasks() -> list[dict]:
    return [
        {"id": "T1", "title": "scaffold", "priority": 1, "acceptance": "pytest -k scaffold"},
        {"id": "T2", "title": "wire it up", "priority": 1, "deps": ["T1"],
         "acceptance": "pytest tests/test_wire.py"},
        {"id": "T3", "title": "docs", "priority": 2, "acceptance": "docs build"},
        {"id": "T4", "title": "polish", "priority": 3, "deps": ["T3"]},
    ]


class TestNormalContinuation(Base):
    def test_next_is_highest_priority_ready_task(self):
        init_mission(self.state, basic_tasks())
        r = self.assertOk(cli(self.state, "next"))
        self.assertTrue(r.stdout.startswith("T1"), r.stdout)
        self.assertIn("acceptance: pytest -k scaffold", r.stdout)

    def test_priority_respected_after_completion(self):
        init_mission(self.state, basic_tasks())
        self.assertOk(cli(self.state, "done", "T1", "--evidence", "pytest -k scaffold -> 3 passed"))
        r = self.assertOk(cli(self.state, "next"))
        self.assertTrue(r.stdout.startswith("T2"), r.stdout)


class TestDependencyOrdering(Base):
    def test_task_with_unfinished_dependency_is_skipped(self):
        init_mission(self.state, basic_tasks())
        # T2 (p1) depends on T1; T3 (p2) is independent -> after T1 T2 wins, but with T1 open only T1 is ready
        r = self.assertOk(cli(self.state, "next"))
        self.assertTrue(r.stdout.startswith("T1"))
        self.assertOk(cli(self.state, "done", "T1", "--evidence", "done"))
        self.assertOk(cli(self.state, "done", "T2", "--evidence", "done"))
        r = self.assertOk(cli(self.state, "next"))
        self.assertTrue(r.stdout.startswith("T3"), r.stdout)
        # T4 still cannot start: T3 is open
        self.assertNotIn("T4", r.stdout)


class TestInterruptionRecovery(Base):
    def test_state_survives_a_process_boundary_with_a_checkpoint(self):
        init_mission(self.state, basic_tasks())
        self.assertOk(cli(self.state, "start", "T1", "--next", "finish the scaffold"))
        r = self.assertOk(cli(self.state, "checkpoint", "--note", "resume by running the wire tests"))
        ck = (self.state / "checkpoint.md").read_text()
        for head in ("## Mission", "## Governing instructions", "## Completed and verified", "## In progress",
                     "## Remaining tasks", "## Blockers", "## Next action", "## Resume condition",
                     "Last updated:"):
            self.assertIn(head, ck, f"missing section {head}")
        self.assertIn("T1", ck)
        self.assertIn("resume by running the wire tests", ck)
        self.assertIn("T1", r.stdout)

        # a NEW process (a later session) recovers without the conversation
        resume = self.assertOk(cli(self.state, "resume"))
        self.assertIn("Mission: test mission", resume.stdout)
        self.assertIn("guide A", resume.stdout)
        self.assertIn("T1 [in_progress]", resume.stdout)
        self.assertIn("Next action: T1", resume.stdout)
        self.assertIn("Resume condition: T1: resume by running the wire tests", resume.stdout)
        self.assertEqual(task(self.state, "T1")["status"], "in_progress")

    def test_start_refuses_a_task_whose_dependencies_are_unfinished(self):
        init_mission(self.state, basic_tasks())
        r = cli(self.state, "start", "T4")            # T4 depends on T3
        self.assertEqual(r.returncode, 2)
        self.assertIn("depends on unfinished", r.stderr)
        self.assertEqual(task(self.state, "T4")["status"], "not_started")


class TestCompletedPreservation(Base):
    def test_verified_task_is_not_offered_again_and_keeps_evidence(self):
        init_mission(self.state, basic_tasks())
        self.assertOk(cli(self.state, "done", "T1", "--evidence", "pytest -k scaffold -> 3 passed"))
        for _ in range(3):
            r = self.assertOk(cli(self.state, "next"))
            self.assertNotIn("T1 ", r.stdout)
        self.assertIn("pytest -k scaffold -> 3 passed", (self.state / "ledger.json").read_text())
        status = self.assertOk(cli(self.state, "status")).stdout
        self.assertIn("[complete          ] T1", status)

    def test_completion_without_evidence_is_rejected(self):
        init_mission(self.state, basic_tasks())
        r = cli(self.state, "done", "T1")
        self.assertEqual(r.returncode, 2)
        self.assertIn("evidence", r.stderr)
        r = cli(self.state, "done", "T1", "--evidence", "  ")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(task(self.state, "T1")["status"], "not_started")


class TestToolFailure(Base):
    def test_blocked_task_is_recorded_and_independent_work_continues(self):
        tasks = [{"id": "T1", "title": "needs a broken tool", "priority": 1},
                 {"id": "T2", "title": "independent work", "priority": 2}]
        init_mission(self.state, tasks)
        r = self.assertOk(cli(self.state, "block", "T1", "--reason", "tool x returns 500"))
        self.assertIn("blocked", r.stdout)
        self.assertEqual(task(self.state, "T1")["status"], "blocked")
        self.assertEqual(ledger(self.state)["blockers"][0]["reason"], "tool x returns 500")
        nxt = self.assertOk(cli(self.state, "next"))
        self.assertTrue(nxt.stdout.startswith("T2"), nxt.stdout)      # routed around the failure
        # the failure is not silently forgotten
        resume = self.assertOk(cli(self.state, "resume")).stdout
        self.assertIn("T1: tool x returns 500", resume)
        # recovery: unblock restores it and clears the blocker
        self.assertOk(cli(self.state, "unblock", "T1", "--note", "tool fixed"))
        self.assertEqual(task(self.state, "T1")["status"], "not_started")
        self.assertEqual(ledger(self.state)["blockers"], [])

    def test_verification_failure_returns_task_to_work(self):
        init_mission(self.state, basic_tasks())
        self.assertOk(cli(self.state, "start", "T1"))
        r = self.assertOk(cli(self.state, "verify", "T1", "--result", "fail", "--evidence", "1 test failed"))
        self.assertIn("verification fail", r.stdout)
        self.assertEqual(task(self.state, "T1")["status"], "in_progress")
        self.assertIn("verification failed: 1 test failed", task(self.state, "T1")["notes"])
        nxt = self.assertOk(cli(self.state, "next"))
        self.assertTrue(nxt.stdout.startswith("T1"))                  # retried, not skipped
        self.assertOk(cli(self.state, "verify", "T1", "--result", "pass", "--evidence", "3 passed"))
        self.assertEqual(task(self.state, "T1")["status"], "complete")


class TestMissingUserInput(Base):
    def test_user_blocker_is_flagged_and_other_work_still_moves(self):
        tasks = [{"id": "T1", "title": "needs a token", "priority": 1},
                 {"id": "T2", "title": "local work", "priority": 2}]
        init_mission(self.state, tasks)
        self.assertOk(cli(self.state, "block", "T1", "--reason", "needs GITHUB_TOKEN", "--needs-user"))
        resume = self.assertOk(cli(self.state, "resume")).stdout
        self.assertIn("USER INPUT REQUIRED", resume)
        self.assertIn("needs GITHUB_TOKEN", resume)
        nxt = self.assertOk(cli(self.state, "next"))
        self.assertTrue(nxt.stdout.startswith("T2"), nxt.stdout)
        # only-blocked missions say so instead of inventing work
        self.assertOk(cli(self.state, "block", "T2", "--reason", "needs approval"))
        r = cli(self.state, "next")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no actionable task", r.stdout)


class TestPartialCompletion(Base):
    def test_independent_tasks_complete_while_one_stays_blocked(self):
        tasks = [{"id": "T1", "title": "blocked external", "priority": 1},
                 {"id": "T2", "title": "independent", "priority": 2},
                 {"id": "T3", "title": "depends on blocked", "priority": 3, "deps": ["T1"]}]
        init_mission(self.state, tasks)
        self.assertOk(cli(self.state, "block", "T1", "--reason", "no network"))
        self.assertOk(cli(self.state, "done", "T2", "--evidence", "pytest -> 5 passed"))
        self.assertEqual(task(self.state, "T2")["status"], "complete")
        self.assertEqual(task(self.state, "T1")["status"], "blocked")
        r = cli(self.state, "next")
        self.assertEqual(r.returncode, 1)
        self.assertIn("waiting on T1", r.stdout)                     # explains why T3 cannot start


class TestCompletionGate(Base):
    def test_refuses_completion_while_anything_is_open(self):
        init_mission(self.state, basic_tasks())
        self.assertOk(cli(self.state, "done", "T1", "--evidence", "ok"))
        r = cli(self.state, "complete")
        self.assertEqual(r.returncode, 1)
        self.assertIn("REFUSED", r.stderr)
        for tid in ("T2", "T3", "T4"):
            self.assertIn(tid, r.stderr)

    def test_completes_only_when_every_task_is_complete_or_waived(self):
        init_mission(self.state, basic_tasks())
        for tid in ("T1", "T2", "T3"):
            self.assertOk(cli(self.state, "done", tid, "--evidence", f"{tid} verified"))
        self.assertEqual(cli(self.state, "complete").returncode, 1)   # T4 still open
        r = self.assertOk(cli(self.state, "waive", "T4", "--reason", "out of scope for this milestone"))
        self.assertIn("waived", r.stdout)
        r = self.assertOk(cli(self.state, "complete"))
        self.assertIn("MISSION COMPLETE", r.stdout)
        self.assertIn("waived: 1", r.stdout)

    def test_waive_requires_a_written_reason(self):
        init_mission(self.state, basic_tasks())
        r = cli(self.state, "waive", "T1", "--reason", " ")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(task(self.state, "T1")["status"], "not_started")


class TestOrderingAndCheckpointContract(Base):
    def test_substantive_work_outranks_bookkeeping_regardless_of_priority(self):
        tasks = [{"id": "T1", "title": "write the commit", "priority": 1, "kind": "bookkeeping"},
                 {"id": "T2", "title": "implement the feature", "priority": 5}]
        init_mission(self.state, tasks)
        r = self.assertOk(cli(self.state, "next"))
        self.assertTrue(r.stdout.startswith("T2"), r.stdout)
        # with only bookkeeping left, it is offered (never a dead end)
        self.assertOk(cli(self.state, "done", "T2", "--evidence", "pytest -> 5 passed"))
        r = self.assertOk(cli(self.state, "next"))
        self.assertTrue(r.stdout.startswith("T1"), r.stdout)

    def test_started_work_still_beats_unstarted_work_of_the_same_kind(self):
        tasks = [{"id": "T1", "title": "half done", "priority": 9},
                 {"id": "T2", "title": "not started", "priority": 1}]
        init_mission(self.state, tasks)
        self.assertOk(cli(self.state, "start", "T1"))
        r = self.assertOk(cli(self.state, "next"))
        self.assertTrue(r.stdout.startswith("T1"), r.stdout)

    def test_checkpoint_and_resume_name_the_task_id_next_to_every_command(self):
        init_mission(self.state, basic_tasks())
        self.assertOk(cli(self.state, "start", "T1"))
        self.assertOk(cli(self.state, "checkpoint", "--note", "run the wire tests"))
        ck = (self.state / "checkpoint.md").read_text()
        self.assertIn("## Resume command", ck)
        self.assertIn(f"{self.state} next    # T1", ck)
        self.assertIn("T1: run the wire tests", ck)
        resume = self.assertOk(cli(self.state, "resume")).stdout
        self.assertIn("Next action: T1", resume)
        self.assertIn("T1: run the wire tests", resume)

    def test_unknown_kind_is_rejected(self):
        spec = {"mission": "m", "tasks": [{"id": "T1", "title": "t", "kind": "errand"}]}
        f = self.state.parent / "badkind.json"
        f.write_text(json.dumps(spec))
        r = subprocess.run([sys.executable, str(CLI), "--dir", str(self.state / "k"), "init", "--file", str(f)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown kind", r.stderr)


class TestSingleWriter(Base):
    def test_a_second_writer_is_refused_while_the_lock_is_fresh(self):
        init_mission(self.state, basic_tasks())
        lock = self.state / ".ledger.lock"
        lock.write_text("999999")
        try:
            r = cli(self.state, "done", "T1", "--evidence", "not really")
            self.assertEqual(r.returncode, 2, r.stdout)
            self.assertIn("holds the lock", r.stderr)
            self.assertEqual(task(self.state, "T1")["status"], "not_started")   # untouched
        finally:
            lock.unlink(missing_ok=True)

    def test_a_stale_lock_is_stolen_and_released(self):
        import os
        import time
        init_mission(self.state, basic_tasks())
        lock = self.state / ".ledger.lock"
        lock.write_text("12345")
        old = time.time() - 900
        os.utime(lock, (old, old))
        r = self.assertOk(cli(self.state, "done", "T1", "--evidence", "ok"))
        self.assertIn("T1 complete", r.stdout)
        self.assertFalse(lock.exists(), "the lock must be released after the command")

    def test_read_only_commands_do_not_take_the_lock(self):
        init_mission(self.state, basic_tasks())
        lock = self.state / ".ledger.lock"
        lock.write_text("999999")
        try:
            self.assertOk(cli(self.state, "next"))      # reads keep working during a run
        finally:
            lock.unlink(missing_ok=True)


class TestLedgerHygiene(Base):
    def test_writes_are_atomic_and_leave_no_temp_files(self):
        init_mission(self.state, basic_tasks())
        for _ in range(5):
            self.assertOk(cli(self.state, "start", "T1"))
            self.assertOk(cli(self.state, "checkpoint"))
        self.assertEqual([p.name for p in self.state.iterdir() if p.name.startswith(".")], [])
        self.assertEqual(sorted(p.name for p in self.state.iterdir()), ["checkpoint.md", "ledger.json"])
        self.assertEqual(oct((self.state / "ledger.json").stat().st_mode & 0o777), "0o600")
        json.loads((self.state / "ledger.json").read_text())          # never truncated

    def test_unknown_task_and_bad_status_are_errors(self):
        init_mission(self.state, basic_tasks())
        self.assertEqual(cli(self.state, "done", "T99", "--evidence", "x").returncode, 2)
        spec = {"mission": "m", "tasks": [{"id": "T1", "title": "t", "status": "finished"}]}
        f = self.state.parent / "bad.json"
        f.write_text(json.dumps(spec))
        r = subprocess.run([sys.executable, str(CLI), "--dir", str(self.state / "other"), "init", "--file", str(f)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown status", r.stderr)

    def test_init_refuses_to_clobber_without_force(self):
        init_mission(self.state, basic_tasks())
        r = cli(self.state, "init", "--mission", "other", "--file", str(self.state.parent / "state-mission.json"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("already exists", r.stderr)

class TestGuard(unittest.TestCase):
    """The anti-abandonment gate: exit 3 means "do not end the turn here".

    These pin the failure that motivated the command: a run that announces a task, asks the user whether to
    proceed, and stops — leaving work open that the register had already authorised.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="ctl-guard-")
        self.state = Path(self.tmp.name) / "state"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_open_work_forbids_stopping(self):
        init_mission(self.state, basic_tasks())
        r = cli(self.state, "guard")
        self.assertEqual(r.returncode, 3, r.stdout)
        self.assertIn("DO NOT end the turn", r.stdout)
        self.assertIn("T1", r.stdout)
        # The directive must carry the concrete action, so continuing needs no re-planning or permission.
        self.assertIn("do: ", r.stdout)
        self.assertIn("asking the user to continue is a violation", r.stdout)

    def test_finished_ledger_allows_a_wrap_up_turn(self):
        init_mission(self.state, basic_tasks())
        for tid in ("T1", "T2", "T3", "T4"):
            cli(self.state, "done", tid, "--evidence", f"{tid} done")
        r = cli(self.state, "guard")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("no actionable work left", r.stdout)

    def test_needs_user_blockers_do_not_block_the_turn_but_are_named(self):
        init_mission(self.state, basic_tasks())
        for tid in ("T1", "T3", "T4"):
            cli(self.state, "done", tid, "--evidence", f"{tid} done")
        cli(self.state, "block", "T2", "--reason", "needs a credential", "--needs-user")
        r = cli(self.state, "guard")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("awaiting user: T2", r.stdout)

    def test_a_dependency_gated_task_is_not_actionable(self):
        tasks = [
            {"id": "T1", "title": "first", "priority": 1, "status": "not_started"},
            {"id": "T2", "title": "second", "priority": 2, "status": "not_started", "deps": ["T1"]},
        ]
        init_mission(self.state, tasks)
        r = cli(self.state, "guard")
        self.assertEqual(r.returncode, 3)
        self.assertIn("T1", r.stdout)
        self.assertNotIn("T2 [", r.stdout)      # gated work is not what guard demands next

    def test_start_records_the_time_so_a_stall_can_be_detected(self):
        init_mission(self.state, basic_tasks())
        cli(self.state, "start", "T1")
        self.assertIn("started", task(self.state, "T1"))
        # Backdate it and guard reports the stall instead of silently re-listing the task.
        led = ledger(self.state)
        for t in led["tasks"]:
            if t["id"] == "T1":
                t["started"] = "2020-01-01T00:00:00+00:00"
        (self.state / "ledger.json").write_text(json.dumps(led))
        r = cli(self.state, "guard", "--stall-minutes", "5")
        self.assertEqual(r.returncode, 3)
        self.assertIn("stalled: T1", r.stdout)


class TestHandoff(unittest.TestCase):
    """deliberate stops must be recorded, and must name what was left open."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="ctl-handoff-")
        self.state = Path(self.tmp.name) / "state"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_handoff_without_a_note_is_refused(self):
        init_mission(self.state, basic_tasks())
        self.assertEqual(cli(self.state, "handoff", "--note", "  ").returncode, 2)

    def test_handoff_must_name_every_open_task(self):
        init_mission(self.state, basic_tasks())
        r = cli(self.state, "handoff", "--note", "stopping")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("T1", r.stderr)

    def test_handoff_naming_the_open_work_is_recorded(self):
        init_mission(self.state, basic_tasks())
        # T1 and T3 are the actionable ones; T2/T4 are gated by their deps, so they are not demanded.
        r = cli(self.state, "handoff", "--note", "T1 half done; T3 not touched", "--checkpoint")
        self.assertEqual(r.returncode, 0, r.stderr)
        led = ledger(self.state)
        self.assertEqual(led["handoffs"][-1]["open"], ["T1", "T3"])
        self.assertTrue((self.state / "checkpoint.md").is_file())

    def test_a_stop_that_announces_a_task_never_started_is_still_caught(self):
        """The exact defect: say "starting T1 now", never start it, end the turn."""
        init_mission(self.state, basic_tasks())
        self.assertNotIn("started", task(self.state, "T1"))
        r = cli(self.state, "handoff", "--note", "wrapping up")
        self.assertEqual(r.returncode, 1)
        self.assertIn("T1", r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
