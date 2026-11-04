"""Follow-up turns must keep the image they refer to.

Scenario: the user asks for/about an image, then replies ``make it brighter``.
That follow-up carries no attachment of its own, so it must be routed through
the same image-input decision as a fresh attachment (``agent/image_routing.py``)
and the original image reference must be present in the request that is built
for the model — not dropped, and not sent to a model that cannot see it.
"""

from __future__ import annotations

import base64

import pytest

from agent import image_routing as ir

# 1x1 transparent PNG, valid magic bytes so `_file_to_data_url` embeds it
# verbatim instead of transcoding it through Pillow.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def poster(tmp_path):
    path = tmp_path / "poster.png"
    path.write_bytes(_PNG)
    return path


def _vision_cfg() -> dict:
    return {
        "agent": {"image_input_mode": "auto"},
        "model": {"provider": "anthropic", "default": "claude-sonnet-4"},
    }


def _dropped_history(path) -> list:
    """History as it comes back after a session-DB resume / compression pass.

    The native turn's image part has been flattened to text (run_agent.py
    persist path, context_compressor._strip_historical_media), leaving the
    ``[Image attached at: ...]`` hint that build_native_content_parts wrote.
    """
    return [
        {"role": "user", "content": f"make me a poster\n\n[Image attached at: {path}]"},
        {"role": "assistant", "content": "Here is your poster."},
    ]


# ─── carried_image_refs ──────────────────────────────────────────────────────


def test_carried_refs_recovered_from_flattened_history(poster):
    paths, urls = ir.carried_image_refs(_dropped_history(poster))
    assert paths == [str(poster)]
    assert urls == []


def test_carried_refs_also_recovered_from_text_mode_hint(poster):
    """Text mode writes a different hint (gateway._enrich_message_with_vision)."""
    history = [
        {
            "role": "user",
            "content": (
                "[The user sent an image~ Here's what I can see:\nA poster.]\n"
                f"[If you need a closer look, use vision_analyze with image_url: {poster} ~]"
                "\n\nmake me a poster"
            ),
        },
        {"role": "assistant", "content": "Here is your poster."},
    ]
    paths, _ = ir.carried_image_refs(history)
    assert paths == [str(poster)]


def test_no_carry_when_prior_image_part_is_still_live(poster):
    """Do not re-ship pixels the request already contains."""
    live = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"make me a poster\n\n[Image attached at: {poster}]"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(_PNG).decode()}"}},
        ],
    }
    paths, urls = ir.carried_image_refs([live, {"role": "assistant", "content": "done"}])
    assert (paths, urls) == ([], [])


def test_no_carry_when_history_has_no_image(poster):
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert ir.carried_image_refs(history) == ([], [])


def test_carried_refs_ignore_missing_files(poster, tmp_path):
    history = [{"role": "user", "content": "[Image attached at: /nope/missing.png]"}]
    assert ir.carried_image_refs(history) == ([], [])


# ─── is_image_edit_followup ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "make it brighter",
        "Make it brighter please",
        "can you crop it?",
        "darken this a bit",
        "upscale it 2x",
        "make the image warmer",
    ],
)
def test_image_edit_followups_recognised(text):
    assert ir.is_image_edit_followup(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "what's the weather in oslo?",
        "thanks!",
        "make it quick, I'm late",
        "write me a summary of the meeting",
        "",
    ],
)
def test_non_image_followups_not_recognised(text):
    assert ir.is_image_edit_followup(text) is False


# ─── the required behaviour, end to end through the routing helpers ──────────


def test_followup_routes_to_vision_model_with_original_image_in_request(poster, monkeypatch):
    """Follow-up + dropped image → native route, original bytes in the request."""
    history = _dropped_history(poster)
    text = "make it brighter"

    assert ir.is_image_edit_followup(text) is True
    paths, urls = ir.carried_image_refs(history)
    assert paths == [str(poster)]

    # Routing instructions reused, not re-invented: the decision is the same
    # one a fresh attachment goes through (models.dev is not reachable in
    # tests, so the capability lookup is stubbed). Patch the module object we
    # hold rather than the dotted path: sibling tests purge sys.modules.
    monkeypatch.setattr(ir, "_lookup_supports_vision", lambda *a: True)
    assert ir.decide_image_input_mode("anthropic", "claude-sonnet-4", _vision_cfg()) == "native"

    parts, skipped = ir.build_native_content_parts(text, paths, image_urls=urls or None)
    assert skipped == []
    images = [p for p in parts if p.get("type") == "image_url"]
    assert len(images) == 1
    url = images[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _PNG
    # and the edit instruction itself survives alongside the pixels
    assert "make it brighter" in parts[0]["text"]


def test_followup_to_text_only_model_keeps_reference(poster):
    """A model that cannot take pixels must still get the original reference.

    ``decide_image_input_mode`` returns ``"text"`` here, so the caller runs the
    vision_analyze pipeline on the *carried* path — the reference is preserved
    instead of the attachment being silently dropped.
    """
    cfg = {"agent": {"image_input_mode": "auto"}, "model": {"provider": "deepseek", "default": "deepseek-v4-pro"}}
    assert ir.decide_image_input_mode("deepseek", "deepseek-v4-pro", cfg) == "text"

    paths, _ = ir.carried_image_refs(_dropped_history(poster))
    assert paths == [str(poster)]
