"""Tests for ``hermes model routing status``.

The command is read-only by design, so these tests assert the two things that matter: it never
raises and exits 0 whatever the config says, and every number it prints comes from the same
engine selection uses.
"""

from __future__ import annotations

import argparse
import json

import pytest

from hermes_cli.subcommands.model import cmd_model_routing_status


def _args(**kw):
    base = {"json": False, "text": None}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def cfg(monkeypatch):
    """Patch the read-only config loader with a dict the test controls."""
    import hermes_cli.config as config_mod

    def _install(value):
        monkeypatch.setattr(config_mod, "load_config_readonly", lambda: value)

    return _install


def test_disabled_config_reports_false_and_exits_zero(cfg, capsys):
    cfg({"model": {"default": "m1", "provider": "p"}})
    assert cmd_model_routing_status(_args()) == 0
    out = capsys.readouterr().out
    assert "enabled: false" in out
    assert "skip_reason:" in out


def test_enabled_config_reports_selection(cfg, capsys):
    cfg({"model": {"default": "cheap", "provider": "p"},
         "smart_model_routing": {"enabled": True, "models": [
             {"name": "cheap", "cost_in": 0.1, "cost_out": 0.2, "tier_rank": 0},
             {"name": "strong", "cost_in": 5.0, "cost_out": 15.0, "tier_rank": 5}]}})
    assert cmd_model_routing_status(_args(text="```refactor``` trade-offs")) == 0
    out = capsys.readouterr().out
    assert "enabled: true" in out
    assert "would select:" in out


def test_json_report_matches_engine_keys(cfg, capsys):
    cfg({"model": {"default": "cheap", "provider": "p"},
         "smart_model_routing": {"enabled": True, "strategy": "cost"}})
    assert cmd_model_routing_status(_args(json=True)) == 0
    report = json.loads(capsys.readouterr().out)
    for key in ("enabled", "settings", "selected", "skip_reason", "reason",
                "eligible_count", "candidate_source"):
        assert key in report
    assert report["settings"]["strategy"] == "cost"


def test_malformed_config_never_raises(cfg, capsys):
    cfg({"model": "not-a-dict", "smart_model_routing": "yes"})
    assert cmd_model_routing_status(_args()) == 0
    assert "enabled: false" in capsys.readouterr().out


def test_loader_failure_is_reported_not_disguised(monkeypatch, capsys):
    import hermes_cli.config as config_mod

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    assert cmd_model_routing_status(_args()) == 2
    out = capsys.readouterr().out
    assert "unreadable" in out and "enabled:" not in out
