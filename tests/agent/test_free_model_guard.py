"""Free-tier quota guard: detection, blocking, switching, accounting.

Each test pins one promise the feature makes to the operator:
  * a free model that hit its limit is not offered again this window
  * the ban lifts by itself when the window rolls
  * the switch goes to the configured paid model, promptly
  * usage and switch events are written down
"""
import json
from datetime import datetime, timedelta

import pytest

from agent import free_model_guard as guard


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STATE_DIR", str(tmp_path))
    yield tmp_path


CFG = {"models": ["paid-one", "minimax-m3.1-flash-free", "paid-two"],
       "free_guard": {"enabled": True, "fallback": "paid-one", "reset_hour": 0}}


class TestIdentity:
    @pytest.mark.parametrize("name,expected", [
        ("minimax-m3.1-flash-free", True),
        ("glm-5-free", True),
        ("provider:free", True),
        ("glm-5.3-flash", False),
        ("big-pickle", False),
        ("", False),
        (None, False),
    ])
    def test_free_is_decided_by_name(self, name, expected):
        assert guard.is_free_model(name) is expected


class TestExhaustion:
    def test_mark_then_blocked(self):
        assert guard.mark_exhausted("minimax-m3.1-flash-free", "p", "rate_limit", CFG) is True
        assert guard.is_exhausted("minimax-m3.1-flash-free", CFG) is True
        assert guard.exhausted_models(CFG) == ["minimax-m3.1-flash-free"]

    def test_marking_twice_is_not_a_new_event(self):
        assert guard.mark_exhausted("m-free", "p", "rate_limit", CFG) is True
        assert guard.mark_exhausted("m-free", "p", "rate_limit", CFG) is False

    def test_new_window_lifts_the_ban(self):
        guard.mark_exhausted("m-free", "p", "rate_limit", CFG)
        state = json.loads(guard.state_path().read_text())
        assert state["models"]["m-free"]["exhausted_at"]
        # Simulate yesterday's state on disk: the flag must not survive the roll.
        state["window"] = "1999-01-01"
        guard.state_path().write_text(json.dumps(state))
        assert guard.is_exhausted("m-free", CFG) is False
        assert guard.exhausted_models(CFG) == []

    def test_window_respects_reset_hour(self):
        cfg = {"free_guard": {"enabled": True, "reset_hour": 2}}
        moment = datetime(2026, 11, 11, 1, 30)
        assert guard.window_key(moment, 2) == "2026-11-10"
        assert guard.window_key(moment, 0) == "2026-11-11"


class TestFiltering:
    @staticmethod
    def _opts(*names):
        from agent.model_routing import ModelOption
        return [ModelOption(name=n, provider="p") for n in names]

    def test_exhausted_free_model_is_dropped(self):
        guard.mark_exhausted("m-free", "p", "rate_limit", CFG)
        kept, dropped = guard.filter_exhausted(self._opts("paid-one", "m-free"), CFG)
        assert [o.name for o in kept] == ["paid-one"]
        assert dropped == ["m-free"]

    def test_paid_models_are_never_dropped(self):
        # Exhaustion is a free-tier concept: a paid model's rate limit is transient and the
        # cooldown path owns it, so the guard must refuse to ban one.
        assert guard.mark_exhausted("paid-one", "p", "rate_limit", CFG) is False
        kept, dropped = guard.filter_exhausted(self._opts("paid-one", "paid-two"), CFG)
        assert [o.name for o in kept] == ["paid-one", "paid-two"]
        assert dropped == []

    def test_guard_disabled_changes_nothing(self):
        guard.mark_exhausted("m-free", "p", "rate_limit", CFG)
        cfg = {"free_guard": {"enabled": False}}
        kept, dropped = guard.filter_exhausted(self._opts("m-free"), cfg)
        assert [o.name for o in kept] == ["m-free"] and dropped == []


class TestSwitch:
    class _Agent:
        def __init__(self, model, provider="p"):
            self.model = model
            self.provider = provider

    def test_quota_failure_switches_to_configured_paid_model(self, monkeypatch):
        switched = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda a, m, p=None: switched.append((m, p)))
        monkeypatch.setattr(guard, "guard_config", lambda cfg=None: CFG["free_guard"])
        monkeypatch.setattr(guard, "_load_routing_models", lambda: CFG["models"])
        agent = self._Agent("minimax-m3.1-flash-free")

        assert guard.switch_off_exhausted(agent, "rate_limit") is True
        assert switched and switched[0][0] == "paid-one"
        assert guard.is_exhausted("minimax-m3.1-flash-free", CFG) is True

    def test_non_quota_reason_does_not_switch(self, monkeypatch):
        switched = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda a, m, p=None: switched.append(m))
        monkeypatch.setattr(guard, "guard_config", lambda cfg=None: CFG["free_guard"])
        agent = self._Agent("minimax-m3.1-flash-free")

        assert guard.switch_off_exhausted(agent, "context_overflow") is False
        assert switched == []
        assert guard.is_exhausted("minimax-m3.1-flash-free", CFG) is False

    def test_paid_model_is_untouched(self, monkeypatch):
        switched = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda a, m, p=None: switched.append(m))
        monkeypatch.setattr(guard, "guard_config", lambda cfg=None: CFG["free_guard"])
        agent = self._Agent("paid-two")

        assert guard.switch_off_exhausted(agent, "rate_limit") is False
        assert switched == []


class TestAccounting:
    def test_usage_counts_requests_and_tokens(self):
        guard.record_usage("m-free", "p", 1200)
        guard.record_usage("m-free", "p", 800)
        state = json.loads(guard.state_path().read_text())
        entry = state["models"]["m-free"]
        assert entry["requests"] == 2 and entry["tokens"] == 2000

    def test_events_are_appended(self):
        guard.mark_exhausted("m-free", "p", "rate_limit", CFG)
        guard.record_usage("m-free", "p", 10)
        lines = [json.loads(l) for l in guard.events_path().read_text().splitlines()]
        assert [e["event"] for e in lines] == ["exhausted", "usage"]

    def test_status_reports_state(self):
        guard.record_usage("m-free", "p", 50)
        guard.mark_exhausted("m-free", "p", "rate_limit", CFG)
        status = guard.status(CFG["free_guard"])
        assert status["enabled"] is True
        assert status["exhausted"] == ["m-free"]
        assert status["usage"]["m-free"]["requests"] == 1
        assert status["resets_at"].endswith("00:00:00")
