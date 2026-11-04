"""Gateway dispatch: an image-edit follow-up must keep the image it refers to.

Reproduces the multi-turn case: turn 1 attaches/generates an image, turn 2 is
a plain-text edit request ("make it brighter"). The follow-up event has no
``media_urls`` of its own, so before the fix the whole image branch in
``GatewayRunner._prepare_inbound_message_text`` (gateway/run.py:10251) was
skipped: ``decide_image_input_mode`` was never consulted, nothing was buffered
for native attachment, and the request built at run.py:18532 was plain text —
the image survived only if the previous turn's raw base64 part happened to
still be in memory (the session-DB persist path at run_agent.py:1844-1854 and
compression at context_compressor.py:492-574 both replace it).
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
    )
    runner.adapters = {}
    runner._pending_native_image_paths_by_session = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="273403055",
        chat_type="dm",
        user_id="42",
        user_name="Maxim",
    )


def _followup_event(text: str = "make it brighter") -> MessageEvent:
    """A plain-text follow-up: no attachments of its own."""
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source())


@pytest.fixture()
def poster(tmp_path):
    path = tmp_path / "poster.png"
    path.write_bytes(_PNG)
    return path


def _history(path) -> list:
    """Turn 1 as it looks after session-DB persist / compression flattened it."""
    return [
        {"role": "user", "content": f"make me a poster\n\n[Image attached at: {path}]"},
        {"role": "assistant", "content": "Here is your poster."},
    ]


def _patch_routing(monkeypatch, *, vision: bool, cfg: dict):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(
        "agent.image_routing._lookup_supports_vision",
        lambda provider, model, config: vision,
    )


def test_followup_buffers_original_image_for_vision_model(monkeypatch, poster):
    runner = _make_runner()
    source = _source()
    _patch_routing(
        monkeypatch,
        vision=True,
        cfg={"agent": {"image_input_mode": "auto"}, "model": {"provider": "anthropic", "default": "claude-sonnet-4"}},
    )

    text = asyncio.run(
        runner._prepare_inbound_message_text(
            event=_followup_event(),
            source=source,
            history=_history(poster),
        )
    )

    key = runner._session_key_for_source(source)
    assert runner._pending_native_image_paths_by_session[key] == [str(poster)]

    # Exactly what run.py:18532-18539 does with that buffer.
    from agent.image_routing import build_native_content_parts

    parts, skipped = build_native_content_parts(
        text, runner._consume_pending_native_image_paths(key)
    )
    assert skipped == []
    images = [p for p in parts if p.get("type") == "image_url"]
    assert len(images) == 1, f"follow-up request dropped the image: {parts!r}"
    assert base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]) == _PNG
    assert "make it brighter" in parts[0]["text"]


def test_followup_to_text_only_model_preserves_reference(monkeypatch, poster):
    runner = _make_runner()
    source = _source()
    _patch_routing(
        monkeypatch,
        vision=False,
        cfg={"agent": {"image_input_mode": "auto"}, "model": {"provider": "deepseek", "default": "deepseek-v4-pro"}},
    )

    seen = {}

    async def fake_enrich(user_text, image_paths):
        seen["paths"] = image_paths
        return f"[vision summary of {image_paths[0]}]\n\n{user_text}"

    monkeypatch.setattr(runner, "_enrich_message_with_vision", fake_enrich)

    text = asyncio.run(
        runner._prepare_inbound_message_text(
            event=_followup_event(),
            source=source,
            history=_history(poster),
        )
    )

    assert seen["paths"] == [str(poster)], "text-only follow-up dropped the image reference"
    assert str(poster) in text
    assert runner._pending_native_image_paths_by_session.get(
        runner._session_key_for_source(source)
    ) is None


def test_unrelated_followup_does_not_reattach_image(monkeypatch, poster):
    runner = _make_runner()
    source = _source()
    _patch_routing(
        monkeypatch,
        vision=True,
        cfg={"agent": {"image_input_mode": "auto"}, "model": {"provider": "anthropic", "default": "claude-sonnet-4"}},
    )

    text = asyncio.run(
        runner._prepare_inbound_message_text(
            event=_followup_event("what's the weather in oslo?"),
            source=source,
            history=_history(poster),
        )
    )

    assert text == "what's the weather in oslo?"
    assert runner._pending_native_image_paths_by_session == {}


def test_followup_does_not_duplicate_a_live_image_part(monkeypatch, poster):
    """If the previous turn still carries the pixels, do not ship them twice."""
    runner = _make_runner()
    source = _source()
    _patch_routing(
        monkeypatch,
        vision=True,
        cfg={"agent": {"image_input_mode": "auto"}, "model": {"provider": "anthropic", "default": "claude-sonnet-4"}},
    )

    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"make me a poster\n\n[Image attached at: {poster}]"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(_PNG).decode()},
                },
            ],
        },
        {"role": "assistant", "content": "Here is your poster."},
    ]

    asyncio.run(
        runner._prepare_inbound_message_text(
            event=_followup_event(),
            source=source,
            history=history,
        )
    )

    assert runner._pending_native_image_paths_by_session == {}
