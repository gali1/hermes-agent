"""Specialist task delegation: the primary model keeps the conversation, a specialist does one job.

These tests exercise the call path itself — who is chosen, what is sent, what comes back — not the
turn-level routing decision (that is tests/agent/test_model_router.py).
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from agent import capability_delegation as cd

# --------------------------------------------------------------------------- fixtures


def _info(modalities):
    return types.SimpleNamespace(
        input_modalities=tuple(modalities), output_modalities=("text",))


def _cand(model, modalities, *, cost_in=1.0, cost_out=1.0, provider="test"):
    from agent.model_router import ModelCandidate, ModelSpec
    return ModelCandidate(
        provider=provider, model=model, info=_info(modalities),
        spec=ModelSpec(name=model, provider=provider, cost_in=cost_in, cost_out=cost_out))


def _config(models, *, preferred_only=False, primary="primary-model"):
    return {
        "model": {"provider": "test", "default": primary},
        "smart_model_routing": {
            "enabled": True,
            "preferred_models_only": preferred_only,
            "models": list(models),
        },
        "providers": {"test": {"models": list(models)}},
    }


@pytest.fixture
def pool(monkeypatch):
    """Patch the pool the resolver sees, so no catalog or live config is involved."""
    def _install(candidates):
        monkeypatch.setattr(cd, "_pool", lambda config: (config, cd._routing_config(config),
                                                        candidates, "test"))
        return candidates
    return _install


@pytest.fixture
def aux(monkeypatch):
    """Capture the out-of-band call and return a canned response."""
    calls = []

    async def fake_async_call_llm(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="SPECIALIST TEXT", reasoning_content=None))])

    from agent import auxiliary_client as aux_mod
    monkeypatch.setattr(aux_mod, "async_call_llm", fake_async_call_llm, raising=False)
    monkeypatch.setattr(
        aux_mod, "extract_content_or_reasoning",
        lambda resp: resp.choices[0].message.content, raising=False)
    return calls


# --------------------------------------------------------------------------- resolution


def test_resolver_picks_the_model_that_declares_the_modality(pool):
    pool([_cand("primary-model", ["text"]),
          _cand("reader", ["text", "pdf"]),
          _cand("talker", ["text", "audio"])])
    choice = cd.resolve_specialist("pdf", config=_config(["primary-model", "reader", "talker"]))
    assert choice is not None
    assert choice.model == "reader"
    assert "capability-first" in choice.reason


def test_resolver_never_delegates_to_the_conversations_own_model(pool):
    """With only the primary able to do it, there is no specialist — not a self-delegation."""
    pool([_cand("primary-model", ["text", "pdf"])])
    assert cd.resolve_specialist("pdf", config=_config(["primary-model"])) is None


def test_resolver_reports_when_no_declared_model_accepts_the_modality(pool):
    pool([_cand("primary-model", ["text"]), _cand("talker", ["text", "audio"])])
    assert cd.resolve_specialist("pdf", config=_config(["primary-model", "talker"])) is None


def test_resolver_honours_selected_models_only(pool):
    """preferred_models_only must not silently widen to a catalog model."""
    pool([_cand("primary-model", ["text"])])
    assert cd.resolve_specialist(
        "pdf", config=_config(["primary-model"], preferred_only=True)) is None


def test_resolver_prefers_the_cheaper_of_two_capable_specialists(pool):
    pool([_cand("primary-model", ["text"]),
          _cand("pricey", ["text", "pdf"], cost_in=50.0, cost_out=50.0),
          _cand("cheap", ["text", "pdf"], cost_in=0.1, cost_out=0.1)])
    choice = cd.resolve_specialist(
        "pdf", config=_config(["primary-model", "pricey", "cheap"]))
    assert choice.model == "cheap", "price breaks ties between equally capable specialists"


# --------------------------------------------------------------------------- message building


def test_image_is_sent_as_a_data_url_part(tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    messages, sent, note = cd.build_delegation_messages(str(img), "what is this?", "image")
    parts = messages[0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert sent == img.stat().st_size
    assert "image_url" in note


def test_audio_is_sent_as_an_input_audio_part_with_the_container_format(tmp_path):
    voice = tmp_path / "memo.ogg"
    voice.write_bytes(b"OggS" + b"1" * 32)
    messages, _sent, note = cd.build_delegation_messages(str(voice), "transcribe", "audio")
    part = messages[0]["content"][1]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "ogg"
    assert part["input_audio"]["data"]


def test_pdf_is_sent_as_a_file_part(tmp_path):
    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n" + b"2" * 32)
    messages, _sent, _note = cd.build_delegation_messages(str(doc), "extract the totals", "pdf")
    part = messages[0]["content"][1]
    assert part["type"] == "file"
    assert part["file"]["filename"] == "scan.pdf"
    assert part["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_text_is_inlined_and_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "MAX_INLINE_TEXT_CHARS", 20)
    log = tmp_path / "app.log"
    log.write_text("x" * 200)
    messages, sent, note = cd.build_delegation_messages(str(log), "summarise", "text")
    body = messages[0]["content"]
    assert "truncated by the delegation layer" in body
    assert sent <= 4096
    assert "truncated" in note


def test_video_is_refused_rather_than_pretended(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError) as exc:
        cd.build_delegation_messages(str(clip), "what happens?", "video")
    assert "no supported transport part" in str(exc.value)


def test_oversized_file_is_refused_before_any_call(tmp_path):
    big = tmp_path / "huge.pdf"
    big.write_bytes(b"%PDF" + b"0" * (2 * 1024 * 1024))
    with pytest.raises(ValueError) as exc:
        cd.build_delegation_messages(str(big), "read", "pdf", max_bytes=1024 * 1024)
    assert "delegation limit" in str(exc.value)


# --------------------------------------------------------------------------- the call


def test_delegation_returns_the_specialists_text(pool, aux, tmp_path):
    pool([_cand("primary-model", ["text"]), _cand("reader", ["text", "pdf"])])
    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n" + b"2" * 32)

    result = cd.delegate_capability_sync(
        str(doc), "extract the totals", modality="pdf",
        config=_config(["primary-model", "reader"]))

    assert result.success is True
    assert result.text == "SPECIALIST TEXT"
    assert (result.provider, result.model) == ("test", "reader")
    assert len(aux) == 1, "exactly one out-of-band call"
    call = aux[0]
    assert call["provider"] == "test" and call["model"] == "reader"
    assert call["task"] == "capability_delegation"
    assert call["messages"][0]["content"][1]["type"] == "file"


def test_no_specialist_means_no_call_and_an_honest_error(pool, aux, tmp_path):
    pool([_cand("primary-model", ["text"])])
    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n")

    result = cd.delegate_capability_sync(
        str(doc), "extract", modality="pdf", config=_config(["primary-model"]))

    assert result.success is False
    assert "no declared specialist" in result.error
    assert aux == [], "a missing specialist must not spend a request"


def test_provider_rejection_is_reported_not_swallowed(pool, monkeypatch, tmp_path):
    pool([_cand("primary-model", ["text"]), _cand("reader", ["text", "audio"])])

    async def boom(**kwargs):
        raise RuntimeError("400 input_audio is not supported by this model")

    from agent import auxiliary_client as aux_mod
    monkeypatch.setattr(aux_mod, "async_call_llm", boom, raising=False)

    clip = tmp_path / "memo.mp3"
    clip.write_bytes(b"ID3" + b"0" * 32)
    # transport="native" is the path that actually spends a provider request for audio
    result = cd.delegate_capability_sync(
        str(clip), "transcribe", modality="audio", config=_config(["primary-model", "reader"]),
        transport="native")

    assert result.success is False
    assert "input_audio is not supported" in result.error
    assert result.text == ""


def test_empty_specialist_reply_is_a_failure(pool, monkeypatch, tmp_path):
    pool([_cand("primary-model", ["text"]), _cand("reader", ["text", "pdf"])])

    async def empty(**kwargs):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="   ", reasoning_content=None))])

    from agent import auxiliary_client as aux_mod
    monkeypatch.setattr(aux_mod, "async_call_llm", empty, raising=False)

    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    result = cd.delegate_capability_sync(
        str(doc), "extract", modality="pdf", config=_config(["primary-model", "reader"]))

    assert result.success is False
    assert "no content" in result.error


def test_the_primary_model_is_never_the_delegate(pool, aux, tmp_path):
    """Delegating to the model that is already answering is a no-op with a price tag."""
    pool([_cand("primary-model", ["text", "pdf"])])
    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    result = cd.delegate_capability_sync(
        str(doc), "extract", modality="pdf", config=_config(["primary-model"]))
    assert result.success is False
    assert aux == []


def test_result_json_is_serialisable(pool, aux, tmp_path):
    pool([_cand("primary-model", ["text"]), _cand("reader", ["text", "pdf"])])
    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    result = cd.delegate_capability_sync(
        str(doc), "extract", modality="pdf", config=_config(["primary-model", "reader"]))
    payload = json.loads(result.to_json())
    assert payload["success"] is True and payload["model"] == "reader"


# --------------------------------------------------------------------------- retired models


def test_a_remembered_dead_specialist_is_skipped(pool, tmp_path, monkeypatch):
    """The provider retired the model but the catalog cache still lists it: skip, don't retry."""
    monkeypatch.setattr(cd, "unusable_path", lambda: tmp_path / "unusable.json")
    pool([_cand("primary-model", ["text"]),
          _cand("retired", ["text", "pdf"]),
          _cand("live", ["text", "pdf"], cost_in=5.0, cost_out=5.0)])
    cd.remember_unusable("test", "retired", "410 ModelDeprecated")
    choice = cd.resolve_specialist(
        "pdf", config=_config(["primary-model", "retired", "live"]))
    assert choice is not None and choice.model == "live"


def test_a_provider_refusal_is_remembered(pool, monkeypatch, tmp_path):
    monkeypatch.setattr(cd, "unusable_path", lambda: tmp_path / "unusable.json")
    pool([_cand("primary-model", ["text"]), _cand("retired", ["text", "pdf"])])

    async def gone(**kwargs):
        raise RuntimeError("Error code: 410 - ModelDeprecated: Model retired has been deprecated.")

    from agent import auxiliary_client as aux_mod
    monkeypatch.setattr(aux_mod, "async_call_llm", gone, raising=False)

    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    result = cd.delegate_capability_sync(
        str(doc), "extract", modality="pdf", config=_config(["primary-model", "retired"]))

    assert result.success is False
    assert ("test", "retired") in cd.remembered_unusable()
    assert cd.resolve_specialist(
        "pdf", config=_config(["primary-model", "retired"])) is None


def test_an_ordinary_bad_request_is_not_memorised(pool, monkeypatch, tmp_path):
    """A malformed part is our bug, not a dead model — remembering it would hide a real specialist."""
    monkeypatch.setattr(cd, "unusable_path", lambda: tmp_path / "unusable.json")
    pool([_cand("primary-model", ["text"]), _cand("reader", ["text", "pdf"])])

    async def bad(**_kwargs):
        raise RuntimeError("Error code: 400 - invalid request: unsupported content part")

    from agent import auxiliary_client as aux_mod
    monkeypatch.setattr(aux_mod, "async_call_llm", bad, raising=False)

    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    result = cd.delegate_capability_sync(
        str(doc), "extract", modality="pdf", config=_config(["primary-model", "reader"]))
    assert result.success is False
    assert cd.remembered_unusable() == set()


def test_forget_unusable_clears_the_memo(tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "unusable_path", lambda: tmp_path / "unusable.json")
    cd.remember_unusable("test", "retired", "410")
    assert ("test", "retired") in cd.remembered_unusable()
    cd.forget_unusable("test", "retired")
    assert cd.remembered_unusable() == set()


# --------------------------------------------------------------------------- tool surface


def test_the_tool_is_registered_in_the_delegation_toolset():
    import tools.capability_delegation_tools  # noqa: F401  (registers on import)
    from tools.registry import registry

    assert "delegate_capability" in registry.get_tool_names_for_toolset("delegation")
    assert registry.get_tool_to_toolset_map().get("delegate_capability") == "delegation"


def test_tool_refuses_without_a_path_or_instruction():
    import asyncio
    from tools.capability_delegation_tools import _handle_delegate_capability

    out = asyncio.run(_handle_delegate_capability({"path": "", "instruction": "read it"}))
    assert "needs both" in out


def test_tool_returns_the_delegation_json(pool, aux, tmp_path):
    import asyncio
    from tools.capability_delegation_tools import _handle_delegate_capability

    pool([_cand("primary-model", ["text"]), _cand("reader", ["text", "pdf"])])
    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n")

    cfg = _config(["primary-model", "reader"])
    import agent.capability_delegation as mod
    original = mod._live_config
    mod._live_config = lambda: cfg  # delegate_capability reads live config when none is passed
    try:
        out = asyncio.run(_handle_delegate_capability(
            {"path": str(doc), "instruction": "extract the totals"}))
    finally:
        mod._live_config = original

    payload = json.loads(out)
    assert payload["success"] is True
    assert payload["text"] == "SPECIALIST TEXT"
    assert payload["model"] == "reader"


# --------------------------------------------------------------------------- transports


def test_pdf_with_a_text_layer_is_extracted_and_sent_as_text(tmp_path, monkeypatch):
    """Portable by default: extraction works on every provider, a `file` part does not."""
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(cd, "_extract_pdf_text", lambda p: "VAULT TOTAL: 412 GB")
    messages, sent, note = cd.build_delegation_messages(str(doc), "extract the total", "pdf")
    body = messages[0]["content"]
    assert isinstance(body, str)
    assert "VAULT TOTAL: 412 GB" in body
    assert "text layer" in note and "portable path" in note
    assert sent == len("VAULT TOTAL: 412 GB")


def test_a_scan_falls_back_to_the_native_part_and_says_so(tmp_path, monkeypatch):
    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n" + b"2" * 32)
    monkeypatch.setattr(cd, "_extract_pdf_text", lambda p: "")
    messages, sent, note = cd.build_delegation_messages(str(doc), "read", "pdf")
    part = messages[0]["content"][1]
    assert part["type"] == "file"
    assert "after local extraction produced nothing" in note
    assert sent == doc.stat().st_size


def test_pdf_transport_text_refuses_a_scan(tmp_path, monkeypatch):
    doc = tmp_path / "scan.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(cd, "_extract_pdf_text", lambda p: "")
    with pytest.raises(ValueError) as exc:
        cd.build_delegation_messages(str(doc), "read", "pdf", transport="text")
    assert "no text layer" in str(exc.value)


def test_audio_is_transcribed_locally_by_default(tmp_path, monkeypatch):
    voice = tmp_path / "memo.ogg"
    voice.write_bytes(b"OggS" + b"1" * 32)
    monkeypatch.setattr(cd, "_transcribe_file", lambda p: (True, "the disk is failing", ""))
    messages, sent, note = cd.build_delegation_messages(str(voice), "summarise", "audio")
    body = messages[0]["content"]
    assert isinstance(body, str) and "the disk is failing" in body
    assert "transcribed" in note and "STT" in note


def test_audio_transport_stt_reports_why_transcription_failed(tmp_path, monkeypatch):
    voice = tmp_path / "memo.ogg"
    voice.write_bytes(b"OggS" + b"1" * 32)
    monkeypatch.setattr(cd, "_transcribe_file", lambda p: (False, "", "no STT provider configured"))
    with pytest.raises(ValueError) as exc:
        cd.build_delegation_messages(str(voice), "transcribe", "audio", transport="stt")
    assert "no STT provider configured" in str(exc.value)


def test_audio_native_skips_transcription(tmp_path, monkeypatch):
    voice = tmp_path / "memo.mp3"
    voice.write_bytes(b"ID3" + b"0" * 32)
    calls = []
    monkeypatch.setattr(cd, "_transcribe_file", lambda p: calls.append(p) or (True, "x", ""))
    messages, _sent, note = cd.build_delegation_messages(
        str(voice), "transcribe", "audio", transport="native")
    assert calls == [], "transport='native' must not spend a transcription"
    assert messages[0]["content"][1]["type"] == "input_audio"
    assert "not guaranteed" in note.replace("no audio part shape is guaranteed", "not guaranteed")


def test_transport_is_threaded_through_the_call(pool, monkeypatch, tmp_path):
    pool([_cand("primary-model", ["text"]), _cand("reader", ["text", "pdf"])])
    seen = {}

    async def fake(**kwargs):
        seen["messages"] = kwargs["messages"]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="OK", reasoning_content=None))])

    from agent import auxiliary_client as aux_mod
    monkeypatch.setattr(aux_mod, "async_call_llm", fake, raising=False)
    monkeypatch.setattr(aux_mod, "extract_content_or_reasoning", lambda r: "OK", raising=False)
    monkeypatch.setattr(cd, "_extract_pdf_text", lambda p: "EXTRACTED BODY")

    doc = tmp_path / "r.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    import asyncio
    result = asyncio.run(cd.delegate_capability(
        str(doc), "read it", modality="pdf", config=_config(["primary-model", "reader"]),
        transport="text"))
    assert result.success is True
    assert "EXTRACTED BODY" in seen["messages"][0]["content"]


# --------------------------------------------------------------------------- audio = STT


def test_audio_returns_the_transcript_without_touching_a_model(pool, aux, tmp_path, monkeypatch):
    """The missing capability is speech-to-text, not a model: no request should be spent."""
    pool([_cand("primary-model", ["text"])])
    monkeypatch.setattr(cd, "_transcribe_file", lambda p: (True, "the disk is failing", ""))
    voice = tmp_path / "memo.ogg"
    voice.write_bytes(b"OggS" + b"1" * 32)

    import asyncio
    result = asyncio.run(cd.delegate_capability(
        str(voice), "transcribe this", modality="audio",
        config=_config(["primary-model"])))

    assert result.success is True
    assert result.text == "the disk is failing"
    assert result.provider == "stt" and result.model == "local"
    assert aux == [], "no specialist request for an audio transcript"


def test_audio_without_stt_reports_the_stt_failure_not_a_fake_result(pool, aux, tmp_path, monkeypatch):
    pool([_cand("primary-model", ["text"])])
    monkeypatch.setattr(cd, "_transcribe_file", lambda p: (False, "", "no STT provider configured"))
    voice = tmp_path / "memo.ogg"
    voice.write_bytes(b"OggS" + b"1" * 32)

    import asyncio
    result = asyncio.run(cd.delegate_capability(
        str(voice), "transcribe this", modality="audio",
        config=_config(["primary-model"])))

    assert result.success is False
    assert "no STT provider configured" in result.error
    assert aux == []


# --------------------------------------------------------------------------- OCR for scans


def test_a_scan_is_ocrd_before_falling_back_to_bytes(tmp_path, monkeypatch):
    """A scanned page has no text layer; OCR is the honest path, a file part is the last resort."""
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.7\n" + b"2" * 32)
    monkeypatch.setattr(cd, "_extract_pdf_text", lambda p: "")
    monkeypatch.setattr(cd, "ocr_available", lambda: True)
    monkeypatch.setattr(cd, "_ocr_pdf", lambda p, **k: "SCANNED INVOICE TOTAL 412")

    messages, sent, note = cd.build_delegation_messages(str(scan), "extract the total", "pdf")
    body = messages[0]["content"]
    assert isinstance(body, str), "OCR text is sent as text, not as bytes"
    assert "SCANNED INVOICE TOTAL 412" in body
    assert "OCR" in note and "tesseract" in note


def test_without_ocr_a_scan_still_falls_back_but_says_why(tmp_path, monkeypatch):
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.7\n" + b"2" * 32)
    monkeypatch.setattr(cd, "_extract_pdf_text", lambda p: "")
    monkeypatch.setattr(cd, "ocr_available", lambda: False)

    messages, _sent, note = cd.build_delegation_messages(str(scan), "extract", "pdf")
    assert messages[0]["content"][1]["type"] == "file"
    assert "after local extraction produced nothing" in note


def test_ocr_is_not_used_when_the_text_layer_has_content(tmp_path, monkeypatch):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(cd, "_extract_pdf_text", lambda p: "a real text layer with plenty of words "
                                                           "to exceed the forty character floor")
    called = []
    monkeypatch.setattr(cd, "_ocr_pdf", lambda p, **k: called.append(p) or "OCR")
    messages, _sent, note = cd.build_delegation_messages(str(doc), "read", "pdf")
    assert called == [], "OCR costs seconds per page: never run it on a readable pdf"
    assert "text layer" in note


# --------------------------------------------------------------------------- extractor placeholder


def test_the_needs_ocr_placeholder_is_not_treated_as_content(tmp_path, monkeypatch):
    """`read_extract` hands back "[NEEDS OCR: pages 1 …]" for a scan — an instruction, not text."""
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(b"%PDF-1.7\n")
    monkeypatch.setattr(cd, "_extract_pdf_text",
                        lambda p: "[NEEDS OCR: pages 1 of this PDF are scanned images …]")
    assert cd.usable_text("[NEEDS OCR: pages 1 …]\nreal line") == "real line"
    monkeypatch.setattr(cd, "ocr_available", lambda: True)
    monkeypatch.setattr(cd, "_ocr_pdf", lambda p, **k: "MAGIC-OCR-8842")
    messages, _sent, note = cd.build_delegation_messages(str(scan), "read", "pdf")
    assert "MAGIC-OCR-8842" in messages[0]["content"]
    assert "OCR" in note


def test_usage_and_cost_are_reported_for_a_model_call(pool, monkeypatch, tmp_path):
    pool([_cand("primary-model", ["text"]), _cand("reader", ["text", "pdf"],
                                                  cost_in=0.30, cost_out=1.20)])

    async def with_usage(**_kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="OK", reasoning_content=None))],
            usage=types.SimpleNamespace(prompt_tokens=1000, completion_tokens=500,
                                        total_tokens=1500))

    from agent import auxiliary_client as aux_mod
    monkeypatch.setattr(aux_mod, "async_call_llm", with_usage, raising=False)
    monkeypatch.setattr(aux_mod, "extract_content_or_reasoning", lambda r: "OK", raising=False)
    monkeypatch.setattr(cd, "_extract_pdf_text", lambda p: "a real text layer, long enough to be used")

    doc = tmp_path / "r.pdf"
    doc.write_bytes(b"%PDF-1.7\n")
    result = cd.delegate_capability_sync(str(doc), "read", modality="pdf",
                                         config=_config(["primary-model", "reader"]))
    assert result.success is True
    assert result.usage["total_tokens"] == 1500
    # 1000 in * $0.30/M + 500 out * $1.20/M
    assert abs(result.est_cost_usd - (0.0003 + 0.0006)) < 1e-9


def test_local_stt_reports_zero_cost(pool, aux, tmp_path, monkeypatch):
    pool([_cand("primary-model", ["text"])])
    monkeypatch.setattr(cd, "_transcribe_file", lambda p: (True, "hello", ""))
    voice = tmp_path / "m.ogg"
    voice.write_bytes(b"OggS" + b"1" * 16)
    import asyncio
    result = asyncio.run(cd.delegate_capability(str(voice), "transcribe", modality="audio",
                                                config=_config(["primary-model"])))
    assert result.success and result.est_cost_usd == 0.0 and result.usage == {}
