"""Tests for agent/content_compression/net_cost.py (Phase 3c).

Hermetic: pure formula tests plus an integration check of the opt-in gate in
ContextCompressor. No network; the summary LLM call is always mocked.
"""

import math
from unittest.mock import patch

import pytest

from agent.content_compression import config as cc_config
from agent.content_compression.net_cost import (
    DEFAULT_EXPECTED_READS,
    DEFAULT_P_ALIVE,
    DEFAULT_READ_COST,
    DEFAULT_WRITE_COST,
    net_mutation_gain,
    should_compress,
)
from agent.context_compressor import ContextCompressor


class TestNetMutationGain:
    def test_defaults(self):
        assert DEFAULT_EXPECTED_READS == 10.0
        assert DEFAULT_P_ALIVE == 1.0
        assert DEFAULT_WRITE_COST == 1.25
        assert DEFAULT_READ_COST == 0.1

    def test_break_even_reads_formula(self):
        delta, summary = 1000, 1000
        break_even = 11.5 * summary / delta
        assert abs(net_mutation_gain(delta, summary, expected_reads=break_even)) < 1e-6
        assert net_mutation_gain(delta, summary, expected_reads=break_even + 0.1) > 0
        assert net_mutation_gain(delta, summary, expected_reads=break_even - 0.1) < 0

    def test_break_even_scales_with_summary(self):
        assert abs(net_mutation_gain(2000, 4000, expected_reads=11.5 * 4000 / 2000)) < 1e-6
        assert abs(net_mutation_gain(500, 250, expected_reads=11.5 * 250 / 500)) < 1e-6

    def test_default_reads_gain_sign(self):
        assert net_mutation_gain(1000, 500) > 0
        assert net_mutation_gain(1000, 1000) < 0
        assert net_mutation_gain(100, 100) < 0

    def test_should_compress(self):
        assert should_compress(1000, 500) is True
        assert should_compress(1000, 1000) is False
        assert should_compress(1000, 1000, expected_reads=12.0) is True

    def test_negative_inputs_clamped(self):
        assert net_mutation_gain(-100, -50) == 0.0
        assert net_mutation_gain(-100, 500) == net_mutation_gain(0, 500)
        assert net_mutation_gain(-100, 500) < 0
        assert net_mutation_gain(500, -100) == net_mutation_gain(500, 0)

    def test_expected_reads_clamped(self):
        assert net_mutation_gain(1000, 1000, expected_reads=-5.0) == net_mutation_gain(
            1000, 1000, expected_reads=0.0
        )

    def test_p_alive_clamped(self):
        assert net_mutation_gain(1000, 1000, p_alive=5.0) == net_mutation_gain(
            1000, 1000, p_alive=1.0
        )
        assert net_mutation_gain(1000, 1000, p_alive=-3.0) == net_mutation_gain(
            1000, 1000, p_alive=0.0
        )
        assert net_mutation_gain(1000, 1000, p_alive=0.0) > 0

    def test_nan_guards(self):
        nan = float("nan")
        assert net_mutation_gain(nan, 1000) == net_mutation_gain(0, 1000)
        assert net_mutation_gain(1000, nan) == net_mutation_gain(1000, 0)
        assert net_mutation_gain(1000, 1000, expected_reads=nan) == net_mutation_gain(
            1000, 1000, expected_reads=0.0
        )
        assert net_mutation_gain(1000, 1000, p_alive=nan) == net_mutation_gain(
            1000, 1000, p_alive=1.0
        )

    def test_custom_costs(self):
        assert net_mutation_gain(100, 100, write_cost=0.5, read_cost=0.5) > 0
        assert net_mutation_gain(100, 100, write_cost=2.0, read_cost=0.1) < 0

    def test_deterministic(self):
        assert net_mutation_gain(1234, 567) == net_mutation_gain(1234, 567)


def _make_compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=2,
            quiet_mode=True,
        )


def _messages(content="Message"):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{content} {i}"}
        for i in range(9)
    ]


class TestNetCostGateIntegration:
    def test_gate_disabled_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(
            cc_config,
            "get_content_compression_config",
            lambda: {"read_lifecycle": False, "net_cost_gate": False},
        )
        c = _make_compressor()
        messages = _messages()
        with patch.object(c, "_generate_summary", return_value="Summary") as mock_summary:
            result = c.compress(messages, current_tokens=90_000)
        assert mock_summary.call_count == 1
        assert len(result) < len(messages)

    def test_gate_enabled_skips_when_savings_too_small(self, monkeypatch):
        monkeypatch.setattr(
            cc_config,
            "get_content_compression_config",
            lambda: {"read_lifecycle": False, "net_cost_gate": True},
        )
        c = _make_compressor()
        messages = _messages()
        with patch.object(c, "_generate_summary", return_value="Summary") as mock_summary:
            result = c.compress(messages, current_tokens=90_000)
        assert mock_summary.call_count == 0
        assert len(result) == len(messages)
        assert result == messages
        assert c._last_compress_aborted is True

    def test_gate_enabled_allows_worthwhile_compression(self, monkeypatch):
        monkeypatch.setattr(
            cc_config,
            "get_content_compression_config",
            lambda: {"read_lifecycle": False, "net_cost_gate": True},
        )
        c = _make_compressor()
        c.max_summary_tokens = 100
        messages = _messages(content="x" * 400)
        with patch.object(c, "_generate_summary", return_value="Summary") as mock_summary:
            result = c.compress(messages, current_tokens=90_000)
        assert mock_summary.call_count == 1
        assert len(result) < len(messages)
        assert c._last_compress_aborted is False
