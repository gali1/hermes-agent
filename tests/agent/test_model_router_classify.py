"""``classify_request`` — the tier ladder, and the short-prompt ceiling (M1 fix).

The gap this pins (M0 finding, spec §7 item 3 "complex work must not be
under-classified to save money"): every volume band started the ladder at
SIMPLE or ROUTINE, and each strong signal added only +1, so a short prompt
carrying exactly one strong code/reasoning signal could never climb past
ROUTINE — "fix this kernel module segfault ..." and "prove the safety
properties of a distributed consensus protocol ..." both came out `routine`.

The fix floors a code/reasoning-bearing prompt at MODERATE.  It deliberately
does *not* jump to COMPLEX/ADVANCED on one keyword: over-classification is the
mirror-image defect (spec §7), and ADVANCED still requires code **and**
reasoning.  The rest of this file pins the classifications that were already
correct, so the fix cannot move a tier that was right.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent import model_router as mr  # noqa: E402


# ---------------------------------------------------------------------------
# the gap
# ---------------------------------------------------------------------------

def test_short_code_prompt_is_no_longer_capped_at_routine():
    prompt = "fix this kernel module segfault that only appears under memory pressure"
    profile = mr.classify_request(prompt)
    assert profile.complexity is mr.Complexity.MODERATE
    assert profile.complexity is not mr.Complexity.ROUTINE
    assert "short-prompt" in profile.signals
    assert "code-signal" in profile.signals


def test_short_reasoning_prompt_is_no_longer_capped_at_routine():
    prompt = "prove the safety properties of a distributed consensus protocol under byzantine faults"
    profile = mr.classify_request(prompt)
    assert profile.complexity is mr.Complexity.MODERATE
    assert profile.complexity is not mr.Complexity.ROUTINE
    assert "reasoning-signal" in profile.signals


def test_one_signal_does_not_reach_complex_or_advanced():
    """The floor is MODERATE: a single keyword must not buy an expensive reasoner."""
    for prompt in (
        "debug this",
        "optimise this",
        "why does it fail",
        "fix this segfault",
    ):
        profile = mr.classify_request(prompt)
        assert profile.complexity in (mr.Complexity.SIMPLE, mr.Complexity.MODERATE), prompt
        assert profile.complexity is not mr.Complexity.ADVANCED, prompt


def test_short_code_and_reasoning_prompt_is_advanced():
    prompt = "analyse the root cause of this deadlock and refactor the handler"
    profile = mr.classify_request(prompt)
    assert profile.complexity is mr.Complexity.ADVANCED
    assert "technical-work" in profile.signals
    assert profile.needs_reasoning is True


# ---------------------------------------------------------------------------
# the classifications that were already right
# ---------------------------------------------------------------------------

def test_greeting_stays_simple():
    assert mr.classify_request("hi").complexity is mr.Complexity.SIMPLE


@pytest.mark.parametrize(
    "prompt",
    [
        "hi",
        "what is 2+2",
        "What is 2 + 2?",
        "What is 15 multiplied by 8?",
        "What is the capital of France?",
        "summarise this email",
        "describe this",
    ],
)
def test_plain_short_prompts_stay_simple(prompt):
    profile = mr.classify_request(prompt)
    assert profile.complexity is mr.Complexity.SIMPLE, prompt
    assert profile.complexity is not mr.Complexity.ADVANCED


def test_a_long_prompt_still_scales_by_volume():
    prompt = "Summarise the following notes. " * 60
    profile = mr.classify_request(prompt)
    assert profile.complexity in (mr.Complexity.MODERATE, mr.Complexity.COMPLEX)
    assert profile.needs_reasoning is False


def test_a_very_long_prompt_with_one_signal_is_advanced():
    """The floor never *lowers* a tier: volume plus one signal still reaches ADVANCED."""
    profile = mr.classify_request(("summarise this section. " * 400) + "debug the parser")
    assert profile.complexity is mr.Complexity.ADVANCED


def test_a_medium_prompt_with_one_signal_is_moderate():
    prompt = ("the deployment notes cover the rollout order, the rollback path and the "
              "on-call rotation for every service in the cluster. " * 3) + "debug it"
    profile = mr.classify_request(prompt)
    assert profile.complexity is mr.Complexity.MODERATE


def test_a_factual_question_without_signals_is_not_over_classified():
    profile = mr.classify_request("what is a monad")
    assert profile.complexity is mr.Complexity.SIMPLE
    assert "factual-question" in profile.signals


def test_classification_is_still_deterministic():
    prompt = "fix this kernel module segfault that only appears under memory pressure"
    assert mr.classify_request(prompt) == mr.classify_request(prompt)


def test_the_tier_reaches_the_turn_level_decision():
    """The router must act on the corrected tier, not just report it."""
    from agent.models_dev import ModelInfo

    def _cand(model, *, tier_cost):
        info = ModelInfo(
            id=model, name=model, family="", provider_id="test", reasoning=False,
            tool_call=True, attachment=False, structured_output=False,
            input_modalities=("text",), output_modalities=("text",),
            context_window=200_000, max_output=8_192, cost_input=tier_cost,
            cost_output=tier_cost * 2, status="",
        )
        return mr.ModelCandidate(provider="test", model=model, info=info)

    profile = mr.classify_request(
        "fix this kernel module segfault that only appears under memory pressure")
    decision = mr.select_model(profile, [_cand("cheap", tier_cost=0.1)])
    assert decision is not None
    assert decision.complexity == mr.Complexity.MODERATE.label
