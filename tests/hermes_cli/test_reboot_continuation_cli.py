"""Integration tests for reboot-gated session continuation on the CLI surface.

Real imports against a temp ``HERMES_HOME``. The ``cli.main()`` auto-resume hook is
exercised with ``_build_cli_from_args`` patched to capture the resume value, and the
slash handlers are driven through ``CLILoopsMixin`` with a minimal fake CLI.
"""

from __future__ import annotations

import atexit

import pytest

from agent import session_continuation as sc


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def boot(monkeypatch):
    token = {"value": "boot-A"}
    monkeypatch.setattr(sc, "current_boot_token", lambda: token["value"])
    return token


def _patch_main_for_resume_capture(monkeypatch, cli):
    """Patch cli.main()'s side effects so a fake HermesCLI captures the resume kwarg."""
    captured = {}

    def _build(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget,
               verbose, compact, resume, checkpoints, pass_session_id, ignore_rules, skills):
        captured["resume"] = resume
        return type("FakeCLI", (), {"run": lambda self: None, "system_prompt": None, "agent": None})()

    monkeypatch.setattr(cli, "_build_cli_from_args", _build)
    monkeypatch.setattr(cli, "_start_worktree_setup", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_install_single_query_signal_handlers", lambda _cli: None)
    return captured


def _run_main(cli, **kwargs):
    try:
        cli.main(**kwargs)
    finally:
        atexit.unregister(cli._run_cleanup)


class FakeCLI:
    """Minimal stand-in exposing the attribute/method surface the mixin handlers use."""

    def __init__(self, session_id="sid-live"):
        self.session_id = session_id
        self.new_title = "<unset>"

    def _split_destructive_skip(self, cmd_original):
        return "", False

    def _confirm_destructive_slash(self, *_a, **_k):
        return "yes"

    def new_session(self, title=None):
        self.new_title = title


def test_registry_exposes_cli_only_commands(home):
    from hermes_cli.commands import COMMAND_REGISTRY, resolve_command
    from cli import HermesCLI

    names = {cmd.name for cmd in COMMAND_REGISTRY}
    assert "continue-after-reboot" in names
    assert "cancel-continuation" in names
    for name in ("continue-after-reboot", "cancel-continuation"):
        cmd = resolve_command(name)
        assert cmd is not None and cmd.cli_only is True and cmd.category == "Session"
    assert resolve_command("reboot-continue").name == "continue-after-reboot"
    assert resolve_command("cancel-reboot-continue").name == "cancel-continuation"
    assert HermesCLI._slash_handler("continue-after-reboot") == ("_handle_continue_after_reboot_command", True)
    assert HermesCLI._slash_handler("cancel-continuation") == ("_handle_cancel_continuation_command", True)


def test_eligible_marker_sets_resume_once_and_consumes(home, boot, monkeypatch):
    import cli

    session_id = "20260101_000000_abc123"
    sc.arm(session_id, home=home, reason="manual")
    boot["value"] = "boot-B"
    monkeypatch.setattr(cli, "_reboot_continuation_session_exists", lambda sid: sid == session_id)

    captured = _patch_main_for_resume_capture(monkeypatch, cli)
    _run_main(cli)
    assert captured["resume"] == session_id
    assert sc.read_state(home=home)["status"] == sc.STATUS_CONSUMED

    captured_second = _patch_main_for_resume_capture(monkeypatch, cli)
    _run_main(cli)
    assert captured_second["resume"] is None


def test_no_marker_leaves_resume_unchanged(home, boot, monkeypatch):
    import cli

    captured = _patch_main_for_resume_capture(monkeypatch, cli)
    _run_main(cli)
    assert captured["resume"] is None
    assert sc.read_state(home=home) is None


def test_explicit_resume_wins_and_marker_stays_armed(home, boot, monkeypatch):
    import cli

    sc.arm("sid-a", home=home)
    boot["value"] = "boot-B"

    captured = _patch_main_for_resume_capture(monkeypatch, cli)
    _run_main(cli, resume="explicit-sid")
    assert captured["resume"] == "explicit-sid"
    assert sc.read_state(home=home)["status"] == sc.STATUS_ARMED


def test_missing_session_marks_failed_and_does_not_resume(home, boot, monkeypatch):
    import cli

    sc.arm("sid-gone", home=home)
    boot["value"] = "boot-B"
    monkeypatch.setattr(cli, "_reboot_continuation_session_exists", lambda _sid: False)

    captured = _patch_main_for_resume_capture(monkeypatch, cli)
    _run_main(cli)
    assert captured["resume"] is None
    assert sc.read_state(home=home)["status"] == sc.STATUS_FAILED


def test_real_session_db_check_resumes(home, boot, monkeypatch):
    import cli
    import hermes_state
    from hermes_state import SessionDB

    db_path = home / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id="sid-real", source="cli")
    db.close()
    # Production's acquire() resolves _default_db_path(); point it at this test's home.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

    sc.arm("sid-real", home=home)
    boot["value"] = "boot-B"

    captured = _patch_main_for_resume_capture(monkeypatch, cli)
    _run_main(cli)
    assert captured["resume"] == "sid-real"
    assert sc.read_state(home=home)["status"] == sc.STATUS_CONSUMED


def test_startup_evaluation_failure_does_not_raise(home, boot, monkeypatch):
    import cli

    sc.arm("sid-boom", home=home)
    boot["value"] = "boot-B"

    def _boom(**_kwargs):
        raise RuntimeError("evaluation exploded")

    monkeypatch.setattr(sc, "evaluate_at_startup", _boom)
    captured = _patch_main_for_resume_capture(monkeypatch, cli)
    _run_main(cli)
    assert captured["resume"] is None
    assert sc.read_state(home=home)["status"] == sc.STATUS_FAILED


def test_explicit_exit_cancels_marker(home, boot):
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    for verb in ("exit", "quit"):
        sc.arm("sid-exit", home=home)
        fake = FakeCLI(session_id="sid-exit")
        assert CLILoopsMixin._cmd_exit(fake, verb) is False
        assert sc.read_state(home=home)["status"] == sc.STATUS_CANCELLED


def test_exit_with_unknown_argument_keeps_marker(home, boot):
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    sc.arm("sid-exit", home=home)
    fake = FakeCLI(session_id="sid-exit")
    assert CLILoopsMixin._cmd_exit(fake, "exit --nope") is True
    assert sc.read_state(home=home)["status"] == sc.STATUS_ARMED


def test_clear_cancels_marker_for_session_being_left(home, boot):
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    class ClearFake(FakeCLI):
        _app = None
        console = type("C", (), {"clear": lambda self: None})()

        def new_session(self, silent=False, title=None):
            self.new_title = title

        def show_banner(self):
            pass

        def _print_random_tip(self):
            pass

    sc.arm("sid-old", home=home)
    fake = ClearFake(session_id="sid-old")
    CLILoopsMixin._cmd_clear(fake, "/clear")
    assert sc.read_state(home=home)["status"] == sc.STATUS_CANCELLED


def test_new_cancels_marker_for_session_being_left(home, boot):
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    sc.arm("sid-old", home=home)
    fake = FakeCLI(session_id="sid-old")
    CLILoopsMixin._cmd_new(fake, "/new")
    assert fake.new_title is None
    assert sc.read_state(home=home)["status"] == sc.STATUS_CANCELLED


def test_new_confirmation_cancel_keeps_marker(home, boot):
    from hermes_cli.cli_loops_mixin import CLILoopsMixin


    class DecliningFake(FakeCLI):
        def _confirm_destructive_slash(self, *_a, **_k):
            return None

    sc.arm("sid-old", home=home)
    fake = DecliningFake(session_id="sid-old")
    assert CLILoopsMixin._cmd_new(fake, "/new") is True
    assert sc.read_state(home=home)["status"] == sc.STATUS_ARMED


def test_arm_command_writes_marker_with_live_session_id(home, boot):
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    CLILoopsMixin._handle_continue_after_reboot_command(FakeCLI("sid-live"), "/continue-after-reboot")
    state = sc.read_state(home=home)
    assert state["session_id"] == "sid-live"
    assert state["status"] == sc.STATUS_ARMED
    assert state["armed_boot_token"] == "boot-A"

    # Arming a different session replaces the single marker.
    CLILoopsMixin._handle_continue_after_reboot_command(FakeCLI("sid-live-2"), "/continue-after-reboot")
    state = sc.read_state(home=home)
    assert state["session_id"] == "sid-live-2"
    assert state["armed_at"]


def test_cancel_command_revokes_marker(home, boot):
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    sc.arm("sid-live", home=home)
    CLILoopsMixin._handle_cancel_continuation_command(FakeCLI("sid-live"), "/cancel-continuation")
    assert sc.read_state(home=home)["status"] == sc.STATUS_CANCELLED


def test_cancel_command_without_marker_is_a_noop(home, boot):
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    CLILoopsMixin._handle_cancel_continuation_command(FakeCLI("sid-live"), "/cancel-continuation")
    assert sc.read_state(home=home) is None
