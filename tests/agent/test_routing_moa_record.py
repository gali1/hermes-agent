"""The MoA skip must be a record, not a debug line.

Routing deliberately stands down while the Mixture-of-Agents facade is running. That skip used
to exist only as ``logger.debug`` + ``return None``, so no surface could say why nothing
switched: a status view either blamed "disabled" or showed a hypothetical selection that could
never happen that turn. These tests pin the record.
"""

from __future__ import annotations

from agent.model_routing import (
    SKIP_DISABLED,
    SKIP_MOA_ACTIVE,
    explain_routing,
    maybe_apply_model_routing,
    moa_active,
)

CFG = {
    "enabled": True,
    "strategy": "automatic",
    "stability_threshold": 0.15,
    "models": [],
}


class _MoaAgent:
    """A real MoA agent: ``bind_moa_runtime`` pins provider and requested_provider to "moa"."""

    _routing_skip_reason: str = ""

    def __init__(self, provider: str = "moa", config=None):
        self.provider = provider
        self.requested_provider = provider
        self.model = "some-model"
        self.config = CFG if config is None else config


def test_moa_skip_reason_is_reported_when_enabled():
    report = explain_routing(CFG, "hello", current="x", provider="p", moa_active=True)
    assert report["moa_active"] is True
    assert report["skip_reason"] == SKIP_MOA_ACTIVE


def test_moa_has_its_own_row():
    """A caller must be able to render "MoA active" without parsing skip_reason."""
    assert "moa_active" in explain_routing(CFG, "hello", current="x", provider="p")


def test_disabled_still_reads_as_disabled():
    """Routing being off is the cause when it is off; MoA is not the reason to report."""
    report = explain_routing({"enabled": False}, "hello", current="x", provider="p",
                             moa_active=True)
    assert report["skip_reason"] == SKIP_DISABLED


def test_no_moa_row_when_inactive():
    report = explain_routing(CFG, "hello", current="x", provider="p")
    assert report["moa_active"] is False
    assert report["skip_reason"] != SKIP_MOA_ACTIVE


def test_engine_records_the_skip_on_the_agent():
    agent = _MoaAgent()
    assert moa_active(agent) is True
    assert maybe_apply_model_routing(agent, "hello") is None
    assert agent._routing_skip_reason == SKIP_MOA_ACTIVE


def test_non_moa_agent_does_not_get_the_marker():
    agent = _MoaAgent(provider="opencode-zen")
    maybe_apply_model_routing(agent, "hello")
    assert getattr(agent, "_routing_skip_reason", None) != SKIP_MOA_ACTIVE


def test_renderer_shows_the_moa_line():
    from hermes_cli.subcommands.model import _render_routing

    report = explain_routing(CFG, "hello", current="x", provider="p", moa_active=True)
    out = _render_routing(report)
    assert "skip_reason: moa_active" in out
    assert "moa: active" in out
