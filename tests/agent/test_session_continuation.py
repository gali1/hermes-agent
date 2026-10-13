"""Unit tests for agent.session_continuation — reboot-gated continuation markers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
    """Mutable fake boot token: tests flip it to simulate a reboot."""
    token = {"value": "boot-A"}
    monkeypatch.setattr(sc, "current_boot_token", lambda: token["value"])
    return token


def _state_file(home: Path) -> Path:
    return home / sc.STATE_FILENAME


class TestBootToken:
    def test_returns_nonempty_string(self):
        assert isinstance(sc.current_boot_token(), str)
        assert sc.current_boot_token()

    def test_prefers_boot_id_file(self, tmp_path, monkeypatch):
        fake = tmp_path / "boot_id"
        fake.write_text("file-boot-token\n", encoding="utf-8")
        monkeypatch.setattr(sc, "BOOT_ID_PATH", fake)
        assert sc.current_boot_token() == "file-boot-token"

    def test_never_raises_when_every_source_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sc, "BOOT_ID_PATH", tmp_path / "missing-boot-id")
        monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil -> ImportError
        assert sc.current_boot_token() == "unknown"


class TestArmAndEvaluate:
    def test_same_boot_token_not_eligible(self, home, boot):
        sc.arm("sid-a")

        assert sc.evaluate_at_startup(session_exists=lambda _sid: True) is None
        assert sc.read_state()["status"] == sc.STATUS_ARMED

    def test_missing_marker_not_eligible(self, home, boot):
        assert sc.evaluate_at_startup(session_exists=lambda _sid: True) is None

    def test_changed_boot_token_eligible_until_consumed(self, home, boot):
        sc.arm("sid-a", reason="manual")
        boot["value"] = "boot-B"

        marker = sc.evaluate_at_startup(session_exists=lambda _sid: True)
        assert marker["session_id"] == "sid-a"
        assert marker["reason"] == "manual"
        assert marker["armed_at"]
        assert sc.read_state()["status"] == sc.STATUS_PENDING

        # Pending survives a failed startup (crash between evaluate and consume).
        assert sc.evaluate_at_startup(session_exists=lambda _sid: True)["session_id"] == "sid-a"

        sc.consume("sid-a")
        assert sc.read_state()["status"] == sc.STATUS_CONSUMED
        assert sc.evaluate_at_startup(session_exists=lambda _sid: True) is None

    def test_consume_ignores_other_session(self, home, boot):
        sc.arm("sid-a")
        sc.consume("sid-b")
        assert sc.read_state()["status"] == sc.STATUS_ARMED

    def test_cancel_never_eligible(self, home, boot):
        sc.arm("sid-a")
        assert sc.cancel("sid-a", reason="explicit exit") is True
        boot["value"] = "boot-B"

        assert sc.evaluate_at_startup(session_exists=lambda _sid: True) is None
        state = sc.read_state()
        assert state["status"] == sc.STATUS_CANCELLED
        assert state["reason"] == "explicit exit"

    def test_cancel_other_session_keeps_marker(self, home, boot):
        sc.arm("sid-a")
        assert sc.cancel("sid-b") is False
        assert sc.read_state()["status"] == sc.STATUS_ARMED

    def test_cancel_unconditional(self, home, boot):
        sc.arm("sid-a")
        assert sc.cancel() is True
        assert sc.read_state()["status"] == sc.STATUS_CANCELLED

    def test_expired_ttl_not_eligible(self, home, boot):
        sc.arm("sid-a")
        boot["value"] = "boot-B"
        later = time.time() + sc.DEFAULT_TTL_SECONDS + 60

        assert sc.evaluate_at_startup(session_exists=lambda _sid: True, now=later) is None
        assert sc.read_state()["status"] == sc.STATUS_CANCELLED

    def test_fresh_marker_not_expired_by_now(self, home, boot):
        sc.arm("sid-a")
        boot["value"] = "boot-B"
        assert sc.evaluate_at_startup(session_exists=lambda _sid: True, now=time.time()) is not None

    def test_missing_session_marks_failed(self, home, boot):
        sc.arm("sid-a")
        boot["value"] = "boot-B"

        assert sc.evaluate_at_startup(session_exists=lambda _sid: False) is None
        state = sc.read_state()
        assert state["status"] == sc.STATUS_FAILED
        assert "no longer exists" in state["error"]
        assert state["attempted_at"]

        # A failed marker never re-fires, even once the session reappears.
        assert sc.evaluate_at_startup(session_exists=lambda _sid: True) is None

    def test_session_exists_exception_is_treated_as_missing(self, home, boot):
        sc.arm("sid-a")
        boot["value"] = "boot-B"

        def _boom(_sid):
            raise RuntimeError("state db down")

        assert sc.evaluate_at_startup(session_exists=_boom) is None
        assert sc.read_state()["status"] == sc.STATUS_FAILED

    def test_failed_marker_never_refires(self, home, boot):
        sc.arm("sid-a")
        boot["value"] = "boot-B"
        sc.mark_failed("sid-a", "resume exploded")

        assert sc.evaluate_at_startup(session_exists=lambda _sid: True) is None
        assert sc.read_state()["status"] == sc.STATUS_FAILED

    def test_mark_failed_ignores_consumed_marker(self, home, boot):
        sc.arm("sid-a")
        sc.consume("sid-a")
        sc.mark_failed("sid-a", "late failure")
        assert sc.read_state()["status"] == sc.STATUS_CONSUMED

    def test_rearm_replaces_previous_marker(self, home, boot):
        sc.arm("sid-a", reason="first")
        sc.arm("sid-b", reason="second")

        state = sc.read_state()
        assert state["session_id"] == "sid-b"
        assert state["status"] == sc.STATUS_ARMED
        assert state["reason"] == "second"
        assert [p.name for p in home.glob(f"{sc.STATE_FILENAME}*")] == [sc.STATE_FILENAME]

    def test_arm_requires_session_id(self, home, boot):
        with pytest.raises(ValueError):
            sc.arm("")

    @pytest.mark.platforms("posix")
    def test_arm_writes_restricted_mode(self, home, boot):
        sc.arm("sid-a")
        mode = _state_file(home).stat().st_mode & 0o777
        assert mode == 0o600


class TestStateFile:
    def test_missing_state_is_none(self, home):
        assert sc.read_state() is None
        assert sc.status() == {"path": str(_state_file(home)), "exists": False, "state": None}

    def test_status_reports_marker(self, home, boot):
        sc.arm("sid-a")
        snapshot = sc.status()
        assert snapshot["exists"] is True
        assert snapshot["state"]["session_id"] == "sid-a"

    def test_corrupt_json_is_quarantined(self, home):
        _state_file(home).write_text("{not json", encoding="utf-8")

        assert sc.read_state() is None
        assert not _state_file(home).exists()
        quarantined = list(home.glob(f"{sc.STATE_FILENAME}.corrupt-*"))
        assert len(quarantined) == 1
        assert quarantined[0].read_text(encoding="utf-8") == "{not json"

    def test_inconsistent_json_is_quarantined(self, home):
        _state_file(home).write_text(json.dumps({"schema": 1, "session_id": ""}), encoding="utf-8")

        assert sc.read_state() is None
        assert list(home.glob(f"{sc.STATE_FILENAME}.corrupt-*"))

    def test_unknown_status_is_quarantined(self, home):
        state = sc.arm("sid-a")
        _state_file(home).write_text(json.dumps({**state, "status": "bogus"}), encoding="utf-8")

        assert sc.read_state() is None
        assert list(home.glob(f"{sc.STATE_FILENAME}.corrupt-*"))

    def test_read_state_never_raises_on_oserror(self, home, monkeypatch):
        _state_file(home).write_text("{}", encoding="utf-8")

        def _boom(*_a, **_k):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "read_text", _boom)
        assert sc.read_state() is None
