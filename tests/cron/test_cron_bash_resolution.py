"""Bash resolution for .sh cron scripts on Windows.

Regression: ``C:\\Windows\\System32\\bash.exe`` (the WSL launcher) precedes Git Bash on
PATH, so ``shutil.which("bash")`` returned the stub. With no WSL distro installed every
``.sh`` cron job died with "The file cannot be accessed by the system." — an error with
no actionable content, repeated on every tick. Git Bash must win, and the stub must never
be handed to subprocess.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron import scheduler_script  # noqa: E402


def _fake_git_bash(tmp_path: Path) -> Path:
    p = tmp_path / "Program Files" / "Git" / "bin" / "bash.exe"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


def _fake_wsl_stub(tmp_path: Path) -> Path:
    p = tmp_path / "Windows" / "System32" / "bash.exe"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


def test_is_wsl_stub_recognises_system32(tmp_path, monkeypatch):
    root = tmp_path / "Windows"
    stub = root / "System32" / "bash.exe"
    stub.parent.mkdir(parents=True)
    stub.write_text("", encoding="utf-8")
    monkeypatch.setenv("SystemRoot", str(root))

    assert scheduler_script._is_wsl_stub(str(stub)) is True


def test_is_wsl_stub_rejects_git_bash(tmp_path, monkeypatch):
    root = tmp_path / "Windows"
    (root / "System32").mkdir(parents=True)
    monkeypatch.setenv("SystemRoot", str(root))
    git = _fake_git_bash(tmp_path)

    assert scheduler_script._is_wsl_stub(str(git)) is False


def test_git_bash_wins_over_wsl_stub(tmp_path, monkeypatch):
    """The core regression: a stub on PATH must not be selected when Git Bash exists."""
    git = _fake_git_bash(tmp_path)
    stub = _fake_wsl_stub(tmp_path)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(scheduler_script, "_git_bash_candidates", lambda: [str(git)])
    monkeypatch.setattr(scheduler_script.shutil, "which", lambda _name: str(stub))

    assert scheduler_script._resolve_bash() == str(git)


def test_stub_alone_yields_no_bash(tmp_path, monkeypatch):
    """With only the stub available, return None so the caller emits a useful error."""
    stub = _fake_wsl_stub(tmp_path)
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(scheduler_script, "_git_bash_candidates", lambda: [])
    monkeypatch.setattr(scheduler_script.shutil, "which", lambda _name: str(stub))

    assert scheduler_script._resolve_bash() is None


def test_argv_error_is_actionable_when_no_bash(tmp_path, monkeypatch):
    """The failure message must name the real problem, not a bare Windows error."""
    script = tmp_path / "job.sh"
    script.write_text("#!/usr/bin/env bash\necho hi\n", encoding="utf-8")

    monkeypatch.setattr(scheduler_script, "_resolve_bash", lambda: None)
    argv, _env, err = scheduler_script._script_argv(script)

    assert argv is None
    assert err is not None
    assert "WSL" in err and "Git for Windows" in err


def test_argv_uses_resolved_bash(tmp_path, monkeypatch):
    script = tmp_path / "job.sh"
    script.write_text("#!/usr/bin/env bash\necho hi\n", encoding="utf-8")
    monkeypatch.setattr(scheduler_script, "_resolve_bash", lambda: r"C:\Git\bin\bash.exe")

    argv, _env, err = scheduler_script._script_argv(script)

    assert err is None
    assert argv == [r"C:\Git\bin\bash.exe", str(script)]


@pytest.mark.skipif(sys.platform == "win32", reason="posix resolution path")
def test_posix_returns_a_real_bash():
    resolved = scheduler_script._resolve_bash()
    assert resolved and Path(resolved).is_file()
