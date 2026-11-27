"""Pasted images get the same treatment as pasted text.

A collapsed text paste lands in ``~/.hermes/pastes/`` behind a
``[Pasted text #n: N lines → path]`` placeholder. These tests pin the image half of that
contract: the bytes are staged in the same directory (never re-encoded when the source is a
file), the placeholder names the local path, and the path round-trips back out of the text
so the turn carries the image.

No clipboard is touched — ``hermes_cli.clipboard`` is patched at the seam.
"""

import struct
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest

from cli import HermesCLI
from hermes_cli import paste_capture
from hermes_cli.paste_capture import (
    PASTE_IMAGE_REF_RE,
    capture_clipboard_image,
    describe_image,
    image_dimensions,
    image_placeholder,
    image_refs,
    stage_image_file,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _png(width: int, height: int, rgb=(10, 20, 30)) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _jpeg(width: int, height: int, *, app1_bytes: int = 0) -> bytes:
    """Minimal JPEG: SOI, optional fat APP1 (EXIF), SOF0 with the frame size, then EOI."""
    out = b"\xff\xd8"
    if app1_bytes:
        payload = b"Exif\x00\x00" + b"\x00" * (app1_bytes - 6)
        out += b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    sof0 = b"\x08" + struct.pack(">HH", height, width) + b"\x03" + b"\x00" * 9
    out += b"\xff\xc0" + struct.pack(">H", len(sof0) + 2) + sof0
    return out + b"\xff\xd9"


# ── format / dimension sniffing ──────────────────────────────────────────


def test_png_dimensions_come_from_the_container_header(tmp_path):
    src = tmp_path / "shot.png"
    src.write_bytes(_png(37, 11))
    assert image_dimensions(src) == (37, 11)
    assert describe_image(src) == "37x11 PNG"


def test_jpeg_dimensions_survive_a_large_exif_segment(tmp_path):
    """Regression: the frame header sits *behind* a multi-megabyte APP1 in real photos, so
    the segment walk must stream instead of giving up inside a fixed read window."""
    src = tmp_path / "photo.jpg"
    src.write_bytes(_jpeg(5000, 3215, app1_bytes=60_000))
    assert image_dimensions(src) == (5000, 3215)
    assert describe_image(src) == "5000x3215 JPEG"


def test_non_image_reports_no_dimensions(tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("hello")
    assert image_dimensions(src) is None
    assert describe_image(src) == "TXT"


# ── staging into the paste directory ─────────────────────────────────────


def test_stage_image_file_copies_bytes_and_names_like_a_text_paste(tmp_path):
    src = tmp_path / "source image ü.png"
    src.write_bytes(_png(64, 48))
    staged = stage_image_file(src, tmp_path / "pastes", 3)

    assert staged.read_bytes() == src.read_bytes()          # byte-identical, no re-encode
    assert staged.parent.name == "pastes"
    assert staged.name.startswith("paste_3_")
    assert staged.suffix == ".png"
    assert src.exists()                                       # source is untouched


def test_stage_image_file_preserves_a_jpeg_suffix(tmp_path):
    src = tmp_path / "photo.jpeg"
    src.write_bytes(_jpeg(120, 90))
    staged = stage_image_file(src, tmp_path / "pastes", 1)
    assert staged.suffix == ".jpeg"
    assert staged.stat().st_size == src.stat().st_size


def test_stage_image_file_rejects_a_non_image(tmp_path):
    src = tmp_path / "fake.png"
    src.write_bytes(b"this is not a png")
    with pytest.raises(ValueError):
        stage_image_file(src, tmp_path / "pastes", 1)
    assert not (tmp_path / "pastes").exists()


def test_stage_image_file_rejects_a_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        stage_image_file(tmp_path / "nope.png", tmp_path / "pastes", 1)


# ── placeholder contract ─────────────────────────────────────────────────


def test_placeholder_names_the_staged_path_and_round_trips(tmp_path):
    staged = stage_image_file(_write(tmp_path / "a b.png", _png(8, 8)), tmp_path / "pastes", 7)
    placeholder = image_placeholder(7, staged)

    assert placeholder == f"[Pasted image #7: 8x8 PNG \u2192 {staged}]"
    assert image_refs("look: " + placeholder) == [staged]
    # The text-paste placeholder regex must never swallow an image reference.
    assert PASTE_IMAGE_REF_RE.findall(placeholder) == [("7", str(staged))]


def test_placeholder_records_the_origin_of_a_copied_file(tmp_path):
    src = _write(tmp_path / "from share.jpg", _jpeg(20, 10))
    staged = stage_image_file(src, tmp_path / "pastes", 2)
    placeholder = image_placeholder(2, staged, origin=src)

    assert "(copied from " in placeholder
    assert image_refs(placeholder) == [staged]


def test_image_refs_ignores_text_pastes_and_dedupes(tmp_path):
    staged = stage_image_file(_write(tmp_path / "a.png", _png(4, 4)), tmp_path / "pastes", 1)
    text_placeholder = f"[Pasted text #1: 3 lines \u2192 {tmp_path / 'note.txt'}]"
    image = image_placeholder(2, staged)
    assert image_refs(text_placeholder) == []
    assert image_refs(f"{image} and again {image}") == [staged]


# ── clipboard capture ────────────────────────────────────────────────────


def test_capture_prefers_a_clipboard_image_file(tmp_path):
    src = _write(tmp_path / "copied.jpg", _jpeg(30, 20))
    with patch("hermes_cli.clipboard.clipboard_image_files", return_value=[src]), \
         patch("hermes_cli.clipboard.save_clipboard_image") as bitmap:
        staged, origin = capture_clipboard_image(tmp_path / "pastes", 5)

    assert origin == src
    assert staged.read_bytes() == src.read_bytes()
    bitmap.assert_not_called()  # a file copy is byte-exact — no bitmap round-trip


def test_capture_falls_back_to_a_clipboard_bitmap(tmp_path):
    def fake_save(dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_png(320, 240))
        return True

    with patch("hermes_cli.clipboard.clipboard_image_files", return_value=[]), \
         patch("hermes_cli.clipboard.save_clipboard_image", side_effect=fake_save):
        staged, origin = capture_clipboard_image(tmp_path / "pastes", 4)

    assert origin is None
    assert (staged.suffix, image_dimensions(staged)) == (".png", (320, 240))


def test_capture_renames_a_mislabelled_payload(tmp_path):
    """Windows file-drop extraction always writes ``.png`` even for JPEG bytes."""
    def fake_save(dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_jpeg(40, 30))
        return True

    with patch("hermes_cli.clipboard.clipboard_image_files", return_value=[]), \
         patch("hermes_cli.clipboard.save_clipboard_image", side_effect=fake_save):
        staged, _origin = capture_clipboard_image(tmp_path / "pastes", 6)

    assert staged.suffix == ".jpg"
    assert image_dimensions(staged) == (40, 30)


def test_capture_returns_none_when_the_clipboard_has_no_image(tmp_path):
    with patch("hermes_cli.clipboard.clipboard_image_files", return_value=[]), \
         patch("hermes_cli.clipboard.save_clipboard_image", return_value=False):
        assert capture_clipboard_image(tmp_path / "pastes", 9) is None


# ── CLI wiring ───────────────────────────────────────────────────────────


def test_paste_command_receives_its_argument():
    """``/paste <path>`` needs the raw command, i.e. the dispatch entry passes cmd_original."""
    assert HermesCLI._SLASH_DISPATCH["paste"] == ("_handle_paste_command", True)


def test_resolve_paste_image_refs_attaches_staged_images(tmp_path):
    staged = stage_image_file(_write(tmp_path / "a.png", _png(5, 5)), tmp_path / "pastes", 1)
    text = "here it is " + image_placeholder(1, staged)

    got = HermesCLI._resolve_paste_image_refs(object(), text, [])
    assert got == [staged]

    # Already attached by the paste gesture → never attached twice.
    assert HermesCLI._resolve_paste_image_refs(object(), text, [staged]) == [staged]


def test_resolve_paste_image_refs_skips_paths_that_vanished(tmp_path):
    gone = tmp_path / "pastes" / "paste_1_000000.png"
    text = image_placeholder(1, gone)
    assert HermesCLI._resolve_paste_image_refs(object(), text, []) == []


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path
