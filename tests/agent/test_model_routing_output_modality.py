"""C9b parity tests: an image-output request must be able to exclude a text-only model.

Written by the peer on oklab-1 against this tree's own shapes; kept as its own file so a
review does not have to diff a large existing module.
"""
from __future__ import annotations

from agent.model_routing import (
    ModelOption,
    RequestProfile,
    RoutingSettings,
    _exclusion_reason,
    classify_request,
    is_eligible,
)


def _option(**kwargs) -> ModelOption:
    base = dict(
        name="probe", provider="test", context_window=200_000, max_output=8_192,
        cost_in=1.0, cost_out=2.0,
    )
    base.update(kwargs)
    return ModelOption(**base)


def _settings() -> RoutingSettings:
    return RoutingSettings()


def test_generation_verb_near_an_artefact_noun_asks_for_an_image_back():
    for prompt in (
        "generate an image of a red fox",
        "Draw me a picture of the harbour",
        "render the chart as an image",
    ):
        assert classify_request(prompt).needs_image_output is True, prompt


def test_merely_mentioning_an_image_stays_an_input_question():
    for prompt in (
        "what is in this image?",
        "explain how image generation works",
        "summarise this picture for me",
    ):
        assert classify_request(prompt).needs_image_output is False, prompt


def test_a_model_declaring_no_image_output_is_excluded_with_a_reason():
    option = _option(supports_image_output=False)
    profile = classify_request("generate an image of a red fox")
    assert profile.needs_image_output is True
    assert _exclusion_reason(option, profile, _settings()) == "unsupported_output_image"


def test_undeclared_output_capability_is_not_excluded_for_the_output_modality():
    """Unknown is not the same claim as unsupported.

    This tree excludes unknown capabilities under ``allow_unknown_metadata`` (strict is the
    default), so the honest assertion is narrower: whatever the reason, it must not be the
    output-modality one. With strict metadata relaxed the model is fully eligible, which
    is what an operator opts into when the catalogue is sparse.
    """
    option = _option(supports_image_output=None)
    profile = RequestProfile(tier="routine", est_input_tokens=10, est_output_tokens=10,
                             needs_image_output=True)
    # lenient metadata: an undeclared output capability does not exclude
    assert is_eligible(option, profile, True) is True
    # strict metadata: it excludes, but never with the OUTPUT-modality reason, because
    # nothing ever declared that the model cannot emit images
    assert is_eligible(option, profile, False) is False
    strict = RoutingSettings(allow_unknown_metadata=False)
    assert _exclusion_reason(option, profile, strict) == "image_output capability unknown (strict mode)"
