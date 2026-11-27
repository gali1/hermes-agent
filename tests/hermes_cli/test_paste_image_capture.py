"""Pasted images: bytes staged in ~/.hermes/pastes/ + a local-path reference.

Covers the contract pasted *text* already has — stage, verify, reference, resolve — plus the
adversarial cases: a hostile source name, a counter restart in the same second, a source
named with the wrong container, junk from a bitmap extraction, and a vanished attachment.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import pytest

from hermes_cli import clipboard, paste_image


# ── fixtures: minimal real containers ───────────────────────────────────


def _png(width: int = 3, height: int = 2, colour: bytes = b"\xff\x00\x00") -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + colour * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _jpeg(width: int = 640, height: int = 480, app1_pad: int = 200_000) -> bytes:
    """JPEG whose frame header sits behind a large EXIF region — the fixed-4KB-read trap.

    APP1 segments cap at 64 KiB (2-byte length), so a multi-MB EXIF block is a *run* of
    them; the parser must walk every segment to reach SOF0.
    """
    out = b"\xff\xd8"
    remaining = app1_pad
    while remaining > 0:
        take = min(remaining, 60_000)
        payload = b"\x00" * take
        out += b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
        remaining -= take
    frame = struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00"
    return out + b"\xff\xc0" + struct.pack(">H", len(frame) + 2) + frame + b"\xff\xd9"


def _gif(width: int = 12, height: int = 7) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00"


def _bmp(width: int = 20, height: int = 10) -> bytes:
    return b"BM" + b"\x00" * 16 + struct.pack("<ii", width, height)


def _webp(width: int = 300, height: int = 200) -> bytes:
    head = b"RIFF" + struct.pack("<I", 22) + b"WEBP" + b"VP8X" + struct.pack("<I", 10)
    head += b"\x00" + b"\x00\x00\x00"
    head += (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    return head


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def pastes(tmp_path) -> Path:
    directory = tmp_path / "pastes"
    directory.mkdir()
    return directory


# ── staging ─────────────────────────────────────────────────────────────


def test_stage_copies_bytes_exactly(pastes, tmp_path):
    src = tmp_path / "shot.png"
    src.write_bytes(_png(4, 5))
    staged = paste_image.stage_path(src, pastes, 1)
    assert staged.parent == pastes
    assert staged.suffix == ".png"
    assert staged.read_bytes() == src.read_bytes()
    assert _sha(staged) == _sha(src)
    assert "[Pasted image #" in paste_image.placeholder(1, staged, origin=src)


def test_stage_rejects_non_image(pastes, tmp_path):
    src = tmp_path / "notes.png"  # lies about being an image
    src.write_text("definitely not an image")
    with pytest.raises(ValueError):
        paste_image.stage_path(src, pastes, 1)


def test_stage_rejects_missing_source(pastes, tmp_path):
    with pytest.raises(FileNotFoundError):
        paste_image.stage_path(tmp_path / "nope.png", pastes, 1)


def test_counter_restart_never_overwrites(pastes, tmp_path):
    """Two CLI sessions, same counter, same second: the second paste gets _2."""
    src_a, src_b = tmp_path / "a.png", tmp_path / "b.png"
    src_a.write_bytes(_png(1, 1, b"\x00\x00\xff"))
    src_b.write_bytes(_png(2, 2, b"\x00\xff\x00"))
    first = paste_image.stage_path(src_a, pastes, 1)
    second = paste_image.stage_path(src_b, pastes, 1)
    assert first != second
    assert second.exists() and first.exists()
    assert src_a.read_bytes() in first.read_bytes()
    assert src_b.read_bytes() in second.read_bytes()


def test_repaste_reuses_the_same_file(pastes, tmp_path):
    src = tmp_path / "shot.png"
    src.write_bytes(_png(8, 8))
    first = paste_image.stage_path(src, pastes, 1)
    second = paste_image.stage_path(src, pastes, 2)
    assert first == second
    assert len([p for p in pastes.iterdir() if p.suffix == ".png"]) == 1


def test_settle_suffix_fixes_a_lying_name(pastes, tmp_path):
    src = tmp_path / "photo.png"  # JPEG bytes under a .png name
    src.write_bytes(_jpeg(100, 50, app1_pad=1024))
    staged = paste_image.stage_path(src, pastes, 1)
    assert staged.suffix == ".jpg"
    assert paste_image.sniff_format(staged)[0] == "JPEG"


# ── dimensions (header parsing, not Pillow) ────────────────────────────


@pytest.mark.parametrize("builder,dims,name", [
    (_png, (3, 2), "PNG"),
    (_gif, (12, 7), "GIF"),
    (_bmp, (20, 10), "BMP"),
    (_webp, (300, 200), "WEBP"),
])
def test_dimensions_from_headers(tmp_path, builder, dims, name):
    src = tmp_path / f"x.{name.lower()}"
    src.write_bytes(builder())
    assert paste_image.dimensions(src) == dims
    assert paste_image.sniff_format(src)[0] == name
    assert paste_image.describe(src) == f"{dims[0]}x{dims[1]} {name}"


def test_jpeg_dimensions_behind_large_exif(tmp_path):
    src = tmp_path / "big.jpg"
    src.write_bytes(_jpeg(1920, 1080, app1_pad=300_000))
    assert paste_image.dimensions(src) == (1920, 1080)


def test_dimensions_of_junk_is_none(tmp_path):
    src = tmp_path / "junk.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\ntruncated")
    assert paste_image.sniff_format(src)[0] == "PNG"
    assert paste_image.dimensions(src) is None


# ── reference round-trip, including hostile names ──────────────────────


def test_placeholder_shape(pastes, tmp_path):
    src = tmp_path / "shot.png"
    src.write_bytes(_png(1280, 720))
    staged = paste_image.stage_path(src, pastes, 3)
    text = paste_image.placeholder(3, staged, origin=src)
    assert text.startswith("[Pasted image #3: 1280x720 PNG (copied from ")
    assert text.endswith(f") \u2192 {staged}]")


def test_hostile_source_name_still_resolves(pastes, tmp_path):
    """A file legally named 'x ] y → z.png' must not break the reference."""
    src = tmp_path / "x ] y \u2192 z.png"
    src.write_bytes(_png(5, 5))
    staged = paste_image.stage_path(src, pastes, 7)
    text = paste_image.placeholder(7, staged, origin=src)
    assert "]" not in text.split("(copied from ", 1)[1].split(") \u2192 ", 1)[0]
    assert paste_image.refs(text, pastes) == [staged]


def test_refs_falls_back_to_index_when_reference_is_unreadable(pastes, tmp_path):
    """A name containing ']' can truncate the readable part; the sidecar still resolves it."""
    src = tmp_path / "odd ] name.png"
    src.write_bytes(_png(5, 5))
    staged = paste_image.stage_path(src, pastes, 4)
    broken = "[Pasted image #4: 5x5 PNG \u2192 ] truncated by the bracket"
    assert paste_image.refs(broken, pastes) == [staged]


def test_refs_ignores_text_paste_placeholders(pastes, tmp_path):
    text = "[Pasted text #1: 3 lines \u2192 /tmp/paste_1_120000.txt]"
    assert paste_image.refs(text, pastes) == []


def test_refs_dedupes_repeated_reference(pastes, tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(_png())
    staged = paste_image.stage_path(src, pastes, 1)
    ref = paste_image.placeholder(1, staged)
    assert paste_image.refs(f"{ref} and again {ref}", pastes) == [staged]


def test_refs_drops_vanished_file(pastes, tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(_png())
    staged = paste_image.stage_path(src, pastes, 1)
    ref = paste_image.placeholder(1, staged)
    staged.unlink()
    assert paste_image.refs(ref, pastes) == []


# ── clipboard: file drop and bitmap ────────────────────────────────────


def test_clipboard_file_wins_over_bitmap(pastes, tmp_path, monkeypatch):
    src = tmp_path / "copied.jpg"
    src.write_bytes(_jpeg(40, 30, app1_pad=512))
    monkeypatch.setattr(clipboard, "clipboard_image_files", lambda: [src])
    called = []
    monkeypatch.setattr(clipboard, "save_clipboard_image", lambda dest: called.append(dest) or False)
    staged, origin = paste_image.capture_clipboard(pastes, 1)
    assert origin == src
    assert staged.read_bytes() == src.read_bytes()
    assert called == []


def test_bitmap_extraction_is_staged_and_validated(pastes, monkeypatch):
    def fake_save(dest):
        dest.write_bytes(_png(640, 400))
        return True

    monkeypatch.setattr(clipboard, "clipboard_image_files", lambda: [])
    monkeypatch.setattr(clipboard, "save_clipboard_image", fake_save)
    staged, origin = paste_image.capture_clipboard(pastes, 2)
    assert origin is None
    assert staged.suffix == ".png"
    assert paste_image.dimensions(staged) == (640, 400)
    assert "(copied from" not in paste_image.placeholder(2, staged)


def test_bitmap_junk_is_rejected(pastes, monkeypatch):
    def fake_save(dest):
        dest.write_bytes(b"a partial write, no container magic at all")
        return True

    monkeypatch.setattr(clipboard, "clipboard_image_files", lambda: [])
    monkeypatch.setattr(clipboard, "save_clipboard_image", fake_save)
    assert paste_image.capture_clipboard(pastes, 3) is None
    assert not [p for p in pastes.iterdir() if p.name.startswith("paste_")]


def test_no_clipboard_image(pastes, monkeypatch):
    monkeypatch.setattr(clipboard, "clipboard_image_files", lambda: [])
    monkeypatch.setattr(clipboard, "save_clipboard_image", lambda dest: False)
    assert paste_image.capture_clipboard(pastes, 4) is None


def test_uri_list_parsing(tmp_path):
    shot = tmp_path / "my shot.png"
    shot.write_bytes(_png())
    payload = "# comment\r\n" + shot.as_uri() + "\r\nfile:///does/not/exist.png\r\n"
    assert clipboard._parse_uri_list(payload) == [shot]


def test_uri_list_ignores_non_images(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("x")
    assert clipboard._parse_uri_list(doc.as_uri()) == []


# ── sidecar index ──────────────────────────────────────────────────────


def test_index_is_fail_open(pastes, tmp_path):
    (pastes / paste_image.INDEX_NAME).write_text("{not json")
    assert paste_image.load_index(pastes) == []
    src = tmp_path / "a.png"
    src.write_bytes(_png())
    staged = paste_image.stage_path(src, pastes, 1)
    assert json.loads((pastes / paste_image.INDEX_NAME).read_text())["entries"][0]["path"] == str(staged)


def test_index_prunes_vanished_entries(pastes, tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(_png())
    staged = paste_image.stage_path(src, pastes, 1)
    staged.unlink()
    assert paste_image.load_index(pastes) == []


# ── CLI wiring: /paste <path> and submit-time resolution ───────────────


class _Buf:
    def __init__(self):
        self.text = ""
        self.cursor_position = 0

    def insert_text(self, text):
        self.text = self.text[: self.cursor_position] + text + self.text[self.cursor_position:]
        self.cursor_position += len(text)


class _App:
    is_running = True

    def __init__(self):
        self.current_buffer = _Buf()
        self.invalidated = 0

    def invalidate(self):
        self.invalidated += 1


def _shell(monkeypatch, pastes):
    from hermes_cli.cli_commands_session_tools import CLICommandsSessionToolsMixin
    from hermes_cli.cli_terminal_mixin import CLITerminalMixin

    class Shell(CLICommandsSessionToolsMixin, CLITerminalMixin):
        def __init__(self):
            self._attached_images = []
            self._tui_paste_counter = 0
            self._image_counter = 0
            self._app = _App()

    monkeypatch.setattr(paste_image, "paste_dir", lambda: pastes)
    return Shell()


def test_paste_command_path_stages_and_inserts_reference(monkeypatch, pastes, tmp_path, capsys):
    src = tmp_path / "screen shot.png"
    src.write_bytes(_png(800, 600))
    shell = _shell(monkeypatch, pastes)
    shell._handle_paste_command(f'/paste "{src}"')
    buf_text = shell._app.current_buffer.text
    assert buf_text.startswith("[Pasted image #1: 800x600 PNG (copied from ")
    staged = paste_image.refs(buf_text, pastes)
    assert len(staged) == 1
    assert _sha(staged[0]) == _sha(src)
    assert shell._app.invalidated == 1


def test_paste_command_without_image(monkeypatch, pastes, capsys):
    shell = _shell(monkeypatch, pastes)
    monkeypatch.setattr(clipboard, "has_clipboard_image", lambda: False)
    monkeypatch.setattr(clipboard, "clipboard_image_files", lambda: [])
    shell._handle_paste_command("/paste")
    assert shell._app.current_buffer.text == ""


def test_paste_command_missing_path(monkeypatch, pastes, capsys):
    shell = _shell(monkeypatch, pastes)
    shell._handle_paste_command("/paste /no/such/image.png")
    assert shell._app.current_buffer.text == ""


def test_resolver_attaches_staged_and_dedupes(monkeypatch, pastes, tmp_path):
    from hermes_cli.cli_stream_mixin import CLIStreamMixin

    src = tmp_path / "a.png"
    src.write_bytes(_png())
    staged = paste_image.stage_path(src, pastes, 1)
    monkeypatch.setattr(paste_image, "paste_dir", lambda: pastes)

    class Shell(CLIStreamMixin):
        pass

    shell = Shell()
    ref = paste_image.placeholder(1, staged)
    # The gesture already attached it; submit must not attach a second copy.
    assert shell._resolve_paste_image_refs(ref, [staged]) == [staged]
    assert shell._resolve_paste_image_refs(ref, []) == [staged]


def test_resolver_drops_vanished_attachment(monkeypatch, pastes, tmp_path):
    from hermes_cli.cli_stream_mixin import CLIStreamMixin

    gone = tmp_path / "gone.png"
    gone.write_bytes(_png())
    gone.unlink()  # deleted between the gesture and the submit

    class Shell(CLIStreamMixin):
        pass

    assert Shell()._resolve_paste_image_refs("hello", [gone]) == []


def test_text_paste_expansion_untouched(monkeypatch, tmp_path):
    """A text placeholder still expands to contents and never becomes an image."""
    from hermes_cli.cli_stream_mixin import CLIStreamMixin

    paste_file = tmp_path / "paste_1_120000.txt"
    paste_file.write_text("line one\nline two\n", encoding="utf-8")

    class Shell(CLIStreamMixin):
        pass

    shell = Shell()
    text = f"see [Pasted text #1: 2 lines \u2192 {paste_file}]"
    assert shell._expand_paste_references(text) == "see line one\nline two\n"
    assert shell._resolve_paste_image_refs(text, []) == []
