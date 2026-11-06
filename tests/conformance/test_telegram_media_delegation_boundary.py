"""The adapter boundary that could not be tested without an operator send.

Until now every claim about "Telegram works" rested either on unit tests of the plugin or on synthetic
rows written into ``events.jsonl`` — which bypass the adapter completely. A real inbound message was the
only missing piece, and it was missing because only the operator can send one.

This drives the **real** adapter method with a **real** Telegram ``Update``, stubbing only two things: the
Bot API network call (``get_file`` / ``download_as_bytearray``) and the agent turn (``handle_message``).
The seam proven here is exactly the unproven one:

    Update -> adapter media handler -> cached file under a trusted root -> the sentence the model reads
           -> capability_delegation.plan() decides on that path

Fixtures are generated per test inside the sandboxed HERMES_HOME (conftest forbids touching the real one).
Byte validity is not the point here — modality routing is; real decoding is covered by the CLI acceptance
rows and the transcription/video suites.
"""

from __future__ import annotations

import datetime as dt
import struct
import zlib
from types import SimpleNamespace
from pathlib import Path

import pytest
from telegram import Chat, Document, Message, Update, User

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _pdf_bytes(text: str = "SCAN MAGIC-PDF-7719") -> bytes:
    content = f"BT /F1 18 Tf 72 700 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    start = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, start)
    return bytes(out)


def _png_bytes(size=(64, 32)) -> bytes:
    w, h = size
    raw = b"".join(b"\x00" + bytes((200, 40, 40)) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


CASES = [
    # (file_name, mime_type, expected_modality, must_delegate, blob)
    ("scan.pdf", "application/pdf", "pdf", True, _pdf_bytes()),
    ("voice_note.ogg", "audio/ogg", "audio", True, b"OggS\x00\x02" + b"\x00" * 900),
    ("clip.mp4", "video/mp4", "video", True, b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 900),
    # Office files are extracted locally by the plugin (no model call, no cost), so a plan item DOES
    # exist — it is the local extractor, not a specialist. The runtime may still answer through
    # read_file; that is why the xlsx row is labelled primary-native in the report.
    ("costs.xlsx",
     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "document", "local",
     b"PK\x03\x04" + b"\x00" * 900),
    ("photo.png", "image/png", "image", False, _png_bytes()),
]


class _FakeFile:
    """Stands in for telegram.File: the bytes are whatever the test generated."""

    def __init__(self, blob: bytes):
        self._blob = blob
        self.file_path = "documents/file_1"

    async def download_as_bytearray(self) -> bytearray:
        return bytearray(self._blob)


async def _returns(value):
    return value


@pytest.fixture
def adapter():
    a = TelegramAdapter(PlatformConfig(enabled=True, token="123456:TESTTOKEN"))
    a._bot = SimpleNamespace()  # the adapter leaves this None until start(); tests need a stub
    # Authorization and gating have their own tests; this one is about the media seam.
    a._is_user_authorized_from_message = lambda *args, **kwargs: True
    a._should_process_message = lambda *args, **kwargs: True
    a._should_observe_unmentioned_group_message = lambda *args, **kwargs: False
    captured: dict = {}
    # Some branches (video documents, generic documents) return without calling handle_message, so the
    # event object itself is captured as it is built and inspected after the handler mutates it.
    original_build = a._build_message_event

    def _build(msg, msg_type, update_id=None):
        event = original_build(msg, msg_type, update_id=update_id)
        captured["event"] = event
        return event

    async def _capture(event):
        captured["handled"] = event

    a._build_message_event = _build  # type: ignore[method-assign]
    a.handle_message = _capture  # type: ignore[method-assign]
    a._captured = captured  # type: ignore[attr-defined]
    return a


def _update(file_name: str, mime_type: str, size: int, blob: bytes) -> Update:
    user = User(id=852568909, first_name="Gadafi", is_bot=False, username="gadafi")
    chat = Chat(id=852568909, type=Chat.PRIVATE, first_name="Gadafi", username="gadafi")
    doc = Document(file_id="f1", file_unique_id="u1", file_name=file_name,
                   mime_type=mime_type, file_size=size)
    msg = Message(message_id=7, date=dt.datetime.now(dt.timezone.utc), chat=chat, from_user=user,
                  caption="test delegation", document=doc)
    # PTB resolves file downloads through the bot bound to the object, so the stub must be bound to
    # both the medium and the message; otherwise get_file() raises before our stub is ever called.
    bot = SimpleNamespace(get_file=lambda *a, **k: _returns(_FakeFile(blob)), defaults=None)
    doc.set_bot(bot)
    msg.set_bot(bot)
    return Update(update_id=700001, message=msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("file_name,mime_type,modality,should_delegate,blob", CASES)
async def test_adapter_delivers_a_trusted_path_and_the_plugin_plans_on_it(
        adapter, file_name, mime_type, modality, should_delegate, blob):
    plugin = pytest.importorskip("capability_delegation")
    await adapter._handle_media_message(_update(file_name, mime_type, len(blob), blob), None)

    event = adapter._captured.get("event")
    assert event is not None, "the adapter never produced a MessageEvent"
    assert event.media_urls, "no media path reached the agent-facing event"
    cached_path = Path(event.media_urls[0])

    roots = [Path(r).resolve() for r in plugin.media_roots()]
    resolved = cached_path.resolve()
    assert any(resolved == r or r in resolved.parents for r in roots), (
        f"{resolved} is outside the plugin's trusted roots {roots}")
    assert resolved.exists(), "the adapter named a file that is not on disk"
    assert resolved.read_bytes()[:8] == blob[:8], "the cached bytes are not the uploaded bytes"

    note = (f"[{modality} '{file_name}' saved at: {cached_path}]\n\n"
            f"{event.text or 'test delegation'}")
    plan = plugin.plan([{"role": "user", "content": note}], model="deepseek-v4.1-flash",
                       provider="opencode-zen", platform="telegram")

    refusals = getattr(plan, "refusals", [])  # Plan carries refusals; a bare list does not
    if should_delegate == "local":
        assert plan, "document: nothing planned at all for adapter-delivered material"
        assert not refusals, f"document: unexpected refusal {refusals}"
        assert plan[0].modality == "document"
        assert Path(plan[0].path).resolve() == resolved
        return
    if should_delegate:
        assert plan, f"{modality}: nothing planned for a path the adapter itself produced"
        assert not refusals, f"{modality}: refused what the adapter delivered: {refusals}"
        assert Path(plan[0].path).resolve() == resolved
    else:
        # Office files and images are deliberately primary-native: read_file / native vision handle
        # them, so there is no delegation and no refusal — delivery + readability is the claim.
        assert not plan, f"{modality}: unexpected delegation {[p.modality for p in plan]}"
        assert not refusals, f"{modality}: unexpected refusal {refusals}"
