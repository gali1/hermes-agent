"""Cost is the expected total cost of the request, term by term (Section 7 item 4).

Regression for the M9/D-3 gap: ``estimate_cost`` priced only input and output
tokens.  Cached input, reasoning tokens, image attachments and expected retries
were invisible, so a model with a cache discount looked as expensive as one
without, a reasoner's deliberation was free, and an operator-declared
reliability never affected the number the ranker compared.

Each term gets its own assertion here, plus the two invariants that must not
change: the legacy two-term total is byte-identical when no advanced term
applies, and an ineligible model still never wins on price.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent import model_router as mr  # noqa: E402
from agent.models_dev import ModelInfo  # noqa: E402


def _cand(
    model,
    *,
    cost_in=10.0,
    cost_out=20.0,
    cache_read=None,
    reasoning=False,
    in_mods=("text",),
    out_mods=("text",),
    ctx=200_000,
):
    """A candidate with exactly the metadata a cost assertion needs."""
    return mr.ModelCandidate(
        provider="test",
        model=model,
        info=ModelInfo(
            id=model,
            name=model,
            family="",
            provider_id="test",
            reasoning=reasoning,
            tool_call=True,
            input_modalities=tuple(in_mods),
            output_modalities=tuple(out_mods),
            context_window=ctx,
            max_output=8_192,
            cost_input=cost_in,
            cost_output=cost_out,
            cost_cache_read=cache_read,
        ),
    )


def _profile(**kw):
    return mr.TaskProfile(**kw)


def _bd(candidate, profile, **kw):
    """A non-``None`` breakdown (the assertion doubles as the unpriced check)."""
    breakdown = mr.estimate_cost_breakdown(candidate, profile, **kw)
    assert breakdown is not None
    return breakdown


# ---------------------------------------------------------------------------
# Backwards compatibility: no advanced term, same total
# ---------------------------------------------------------------------------


def test_legacy_two_term_total_is_unchanged():
    p = _profile(est_input_tokens=1_000_000, est_output_tokens=1_000_000)
    c = _cand("m", cost_in=3.0, cost_out=15.0)
    assert mr.estimate_cost(c, p) == pytest.approx(18.0)
    b = _bd(c, p)
    assert b["total"] == pytest.approx(18.0)
    assert b["cached_input"] == b["reasoning"] == b["modality"] == b["retries"] == 0.0
    assert b["policy"].startswith("reasoning_at_output_rate")


def test_breakdown_terms_sum_to_the_total():
    p = _profile(
        est_input_tokens=200_000,
        est_output_tokens=1_000,
        est_cached_input_tokens=50_000,
        est_reasoning_tokens=500,
        image_inputs=1,
    )
    b = _bd(_cand("m", reasoning=True, cache_read=1.0), p)
    terms = b["input"] + b["cached_input"] + b["output"] + b["reasoning"] + b["modality"]
    assert b["subtotal"] == pytest.approx(terms)
    assert b["total"] == pytest.approx(b["subtotal"] + b["retries"])


# ---------------------------------------------------------------------------
# Cached input
# ---------------------------------------------------------------------------


def test_cached_input_is_billed_at_the_cache_read_rate():
    p = _profile(est_input_tokens=1_000_000, est_cached_input_tokens=400_000)
    c = _cand("m", cost_in=10.0, cost_out=0.0, cache_read=1.0)
    b = _bd(c, p)
    assert b["input"] == pytest.approx(6.0)          # 600k tokens at $10/M
    assert b["cached_input"] == pytest.approx(0.4)   # 400k tokens at $1/M
    assert b["total"] == pytest.approx(6.4)


def test_cached_input_falls_back_to_the_input_rate_without_a_published_discount():
    p = _profile(est_input_tokens=1_000_000, est_cached_input_tokens=400_000)
    c = _cand("m", cost_in=10.0, cost_out=0.0, cache_read=None)
    b = _bd(c, p)
    assert b["cached_input"] == pytest.approx(4.0)   # no discount invented
    assert b["total"] == pytest.approx(10.0)


def test_declared_cache_read_override_wins_over_the_catalog():
    p = _profile(est_input_tokens=1_000_000, est_cached_input_tokens=500_000)
    spec = mr.ModelSpec.from_entry(
        {"name": "m", "cost_in": 10.0, "cost_out": 0.0, "cost_cache_read": 0.5}
    )
    c = mr.ModelCandidate("test", "m", None, spec=spec)
    b = _bd(c, p)
    assert b["charged"]["cost_cache_read"] == 0.5
    assert b["cached_input"] == pytest.approx(0.25)


def test_cache_credit_can_never_exceed_the_input_estimate():
    p = _profile(est_input_tokens=1_000, est_cached_input_tokens=50_000)
    b = _bd(_cand("m", cache_read=1.0), p)
    assert b["tokens"]["cached_input"] == 1_000
    assert b["tokens"]["billable_input"] == 0


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


def test_reasoning_tokens_are_charged_at_the_output_rate():
    p = _profile(est_input_tokens=0, est_output_tokens=1_000, est_reasoning_tokens=2_000)
    b = _bd(_cand("m", cost_out=30.0, reasoning=True), p)
    assert b["output"] == pytest.approx(0.03)
    assert b["reasoning"] == pytest.approx(0.06)
    assert b["total"] == pytest.approx(0.09)


def test_a_non_reasoning_model_is_not_charged_for_reasoning_tokens():
    p = _profile(est_output_tokens=1_000, est_reasoning_tokens=2_000)
    b = _bd(_cand("m", cost_out=30.0, reasoning=False), p)
    assert b["reasoning"] == 0.0
    assert b["tokens"]["reasoning"] == 0


def test_classification_derives_reasoning_tokens_only_for_reasoning_tasks():
    hard = mr.classify_request(
        "Analyse why this service deadlocks under load and propose an architecture "
        "change. Explain the root cause step by step and derive the trade-offs.\n"
        "```python\ndef handler():\n    pass\n```"
    )
    assert hard.needs_reasoning is True
    assert hard.est_reasoning_tokens > 0
    assert "reasoning-tokens" in hard.signals

    easy = mr.classify_request("What is the capital of France?")
    assert easy.est_reasoning_tokens == 0


# ---------------------------------------------------------------------------
# Modality
# ---------------------------------------------------------------------------


def test_image_inputs_are_charged_as_input_token_equivalents():
    p = _profile(est_input_tokens=0, image_inputs=2)
    b = _bd(_cand("m", cost_in=10.0, cost_out=0.0), p)
    assert b["tokens"]["image_equivalent"] == 2 * mr._IMAGE_TOKEN_EQUIVALENT
    assert b["modality"] == pytest.approx(2 * 1600 / 1_000_000 * 10.0)


def test_classification_counts_image_attachments():
    p = mr.classify_request(
        "compare these", attachments=["a.png", "b.jpg", "notes.pdf"], expects_image=True
    )
    assert p.image_inputs == 2
    assert p.est_cached_input_tokens == 0


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


def test_retry_term_uses_the_declared_reliability():
    p = _profile(est_input_tokens=1_000_000)
    c = _cand("m", cost_in=10.0, cost_out=0.0)
    b = _bd(c, p, reliability=0.5)
    assert b["expected_retries"] == pytest.approx(1.0)
    assert b["retries"] == pytest.approx(10.0)  # one expected retry
    assert b["total"] == pytest.approx(20.0)


def test_unknown_or_perfect_reliability_adds_no_retry_cost():
    p = _profile(est_input_tokens=1_000_000)
    c = _cand("m", cost_in=10.0, cost_out=0.0)
    for value in (None, 1.0, 1.5):
        b = _bd(c, p, reliability=value)
        assert b["retries"] == 0.0


def test_retry_fraction_is_capped():
    assert mr.expected_retry_fraction(0.0) == mr._MAX_RETRY_FRACTION
    assert mr.expected_retry_fraction(-1.0) == mr._MAX_RETRY_FRACTION


# ---------------------------------------------------------------------------
# Charging policy + ranking invariants
# ---------------------------------------------------------------------------


def test_unpriced_model_is_still_none_and_a_charge_cannot_buy_a_cache_discount():
    p = _profile(est_input_tokens=1_000_000, est_cached_input_tokens=500_000)
    unpriced = mr.ModelCandidate("test", "mystery", None)
    assert mr.estimate_cost(unpriced, p) is None
    b = _bd(unpriced, p, charge=(4.0, 8.0))
    # Charged at the median input rate for every input token: no invented discount.
    assert b["charged"]["cost_cache_read"] == 4.0
    assert b["total"] == pytest.approx(4.0)


def test_cheap_model_that_cannot_meet_the_task_never_wins_on_price():
    # An image edit needs image input *and* image output.
    p = mr.classify_request(
        "make it brighter", attachments=["shot.png"], expects_image=True
    )
    cheap_text_only = _cand("cheap", cost_in=0.01, cost_out=0.01)
    capable = _cand(
        "capable",
        cost_in=10.0,
        cost_out=20.0,
        in_mods=("text", "image"),
        out_mods=("text", "image"),
    )
    d = mr.select_model(p, [cheap_text_only, capable])
    assert d is not None and d.model == "capable"
    assert mr.estimate_cost(cheap_text_only, p) < mr.estimate_cost(capable, p)


def test_cheaper_unconfirmed_reasoner_does_not_win_a_reasoning_task():
    p = mr.classify_request(
        "Analyse why this service deadlocks under load and propose an architecture "
        "change. Explain the root cause step by step and derive the trade-offs.\n"
        "```python\ndef handler():\n    pass\n```"
    )
    assert p.needs_reasoning is True
    cheap = _cand("cheap", cost_in=0.01, cost_out=0.01, reasoning=False)
    reasoner = _cand("reasoner", cost_in=10.0, cost_out=20.0, reasoning=True)
    d = mr.select_model(p, [cheap, reasoner])
    assert d is not None and d.model == "reasoner"
    # The cheap model is genuinely cheaper — it still must not be picked, because
    # its reasoning support is unconfirmed (Section 7 items 3 and 4).
    assert mr.estimate_cost(cheap, p) < mr.estimate_cost(reasoner, p)


def test_cache_discount_survives_the_ranking_charge_path():
    """The ranker always passes ``charge``; that must not erase a real discount."""
    p = _profile(est_input_tokens=1_000_000, est_cached_input_tokens=800_000)
    headline = _cand("headline", cost_in=10.0, cost_out=20.0)
    discounted = _cand("discounted", cost_in=12.0, cost_out=20.0, cache_read=0.5)
    assert mr.estimate_cost(discounted, p) < mr.estimate_cost(headline, p)
    selection = mr.rank_candidates(p, [headline, discounted], mr.RoutingConfig())
    assert selection.model == "discounted"
    assert selection.pricing["cheapest_eligible"] == pytest.approx(
        mr.estimate_cost(discounted, p)
    )


def test_report_pricing_uses_the_same_retry_term_as_ranking():
    cfg = mr.RoutingConfig(reliability={"test/flaky": 0.5})
    p = _profile(est_input_tokens=1_000_000)
    flaky = _cand("flaky", cost_in=10.0, cost_out=0.0)
    steady = _cand("steady", cost_in=10.0, cost_out=0.0)
    selection = mr.rank_candidates(p, [flaky, steady], cfg)
    own = mr.estimate_cost(flaky, p, reliability=0.5)
    assert selection.pricing["cheapest_eligible"] is not None
    # The flaky model's retry term makes it dearer than the steady one, and the
    # report agrees with the ranking path's own number.
    assert own == pytest.approx(20.0)
    assert selection.pricing["cheapest_eligible"] == pytest.approx(10.0)
