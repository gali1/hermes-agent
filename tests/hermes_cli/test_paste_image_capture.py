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
    load_index,
    remember_paste,
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

    got = _cli_instance()._resolve_paste_image_refs(text, [])
    assert got == [staged]

    # Already attached by the paste gesture → never attached twice.
    assert _cli_instance()._resolve_paste_image_refs(text, [staged]) == [staged]


def test_resolve_paste_image_refs_skips_paths_that_vanished(tmp_path):
    gone = tmp_path / "pastes" / "paste_1_000000.png"
    text = image_placeholder(1, gone)
    assert _cli_instance()._resolve_paste_image_refs(text, []) == []


# ── hostile file names, repeat pastes, and the index fallback ────────────


def test_reference_survives_brackets_in_the_source_name(tmp_path):
    """A `]` in the copied file's name used to truncate the reference: the tail stopped at the
    bracket and the image vanished at submit time."""
    src = _write(tmp_path / "bracket ]name.png", _png(16, 8))
    paste_dir = tmp_path / "pastes"
    staged = stage_image_file(src, paste_dir, 11)
    placeholder = image_placeholder(11, staged, origin=src)

    assert "]" not in placeholder.split("\u2192")[0]        # origin part is escaped for display
    assert image_refs(placeholder, paste_dir) == [staged]


def test_reference_survives_an_arrow_in_the_source_name(tmp_path):
    """U+2192 inside a file name used to split the tail at the wrong arrow."""
    src = _write(tmp_path / "arrow \u2192 in name.png", _png(9, 6))
    paste_dir = tmp_path / "pastes"
    staged = stage_image_file(src, paste_dir, 12)
    placeholder = image_placeholder(12, staged, origin=src)

    assert image_refs(placeholder, paste_dir) == [staged]


def test_literal_path_wins_over_the_index(tmp_path):
    """A recalled older prompt must keep its own image even though counters restart per session."""
    paste_dir = tmp_path / "pastes"
    first = stage_image_file(_write(tmp_path / "a.png", _png(4, 4)), paste_dir, 1)
    second = stage_image_file(_write(tmp_path / "b.png", _png(5, 5)), paste_dir, 1)  # same counter, new session
    assert first != second
    assert image_refs(image_placeholder(1, first), paste_dir) == [first]


def test_the_same_bytes_are_staged_once(tmp_path):
    """Pasting the same screenshot twice produced two identical files and two copies to the model."""
    paste_dir = tmp_path / "pastes"
    src = _write(tmp_path / "shot.png", _png(20, 10))
    one = stage_image_file(src, paste_dir, 21)
    two = stage_image_file(src, paste_dir, 22)

    assert one == two
    assert sorted(p.name for p in paste_dir.glob("*.png")) == [one.name]


def test_the_same_bytes_are_staged_once_after_the_counter_restarts(tmp_path):
    """Counters restart every CLI session, and the index is keyed by counter — so the second
    session's ``#1`` overwrote the record for the first session's ``#1`` and the byte dedupe
    could no longer see the earlier paste. The bytes have to answer for themselves."""
    paste_dir = tmp_path / "pastes"
    src = _write(tmp_path / "shot.png", _png(20, 10))
    first = stage_image_file(src, paste_dir, 1)

    other = _write(tmp_path / "other.png", _png(30, 20))
    stage_image_file(other, paste_dir, 1, dedupe=False)      # new session, same counter
    (paste_dir / paste_capture.INDEX_NAME).unlink()          # index now knows nothing at all

    again = stage_image_file(src, paste_dir, 1)

    assert again == first
    # two images were ever pasted, so two files exist — no third copy of the same bytes
    assert len(list(paste_dir.glob("*.png"))) == 2


def test_corrupt_clipboard_bitmap_is_rejected(tmp_path):
    """The bitmap branch used to trust the extractor: junk bytes became a broken attachment."""
    def fake_save(dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"not an image at all")
        return True

    paste_dir = tmp_path / "pastes"
    with patch("hermes_cli.clipboard.clipboard_image_files", return_value=[]), \
         patch("hermes_cli.clipboard.save_clipboard_image", side_effect=fake_save):
        assert capture_clipboard_image(paste_dir, 31) is None

    assert not list(paste_dir.glob("*.png"))


def test_index_drops_entries_whose_file_vanished(tmp_path):
    paste_dir = tmp_path / "pastes"
    staged = stage_image_file(_write(tmp_path / "a.png", _png(3, 3)), paste_dir, 41)
    assert [e["id"] for e in load_index(paste_dir)] == [41]

    staged.unlink()
    assert load_index(paste_dir) == []


def test_remember_paste_keeps_the_latest_entry_per_counter(tmp_path):
    paste_dir = tmp_path / "pastes"
    first = _write(tmp_path / "a.png", _png(3, 3))
    second = _write(tmp_path / "b.png", _png(4, 4))
    remember_paste(paste_dir, 7, first)
    remember_paste(paste_dir, 7, second)

    entries = load_index(paste_dir)
    assert len(entries) == 1 and Path(entries[0]["path"]) == second


def test_resolve_drops_an_attachment_that_no_longer_exists(tmp_path):
    gone = tmp_path / "pastes" / "paste_9_000000.png"
    assert _cli_instance()._resolve_paste_image_refs("no refs here", [gone]) == []


def test_staged_bytes_match_the_source_sha256(tmp_path):
    """The acceptance bar: a file source is copied byte-for-byte, not re-encoded."""
    import hashlib
    src = _write(tmp_path / "shot.png", _png(120, 80))
    staged = stage_image_file(src, tmp_path / "pastes", 51)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert digest(staged) == digest(src)
    assert staged.stat().st_size == src.stat().st_size


def test_same_counter_in_the_same_second_does_not_overwrite(tmp_path):
    """Counters restart every CLI session; two pastes must never share one name."""
    paste_dir = tmp_path / "pastes"
    one = stage_image_file(_write(tmp_path / "one.png", _png(6, 6)), paste_dir, 61)
    two = stage_image_file(_write(tmp_path / "two.png", _png(7, 7)), paste_dir, 61)
    assert one != two
    assert one.exists() and two.exists()
    assert image_dimensions(one) == (6, 6) and image_dimensions(two) == (7, 7)


def test_hostile_source_name_round_trips_to_the_staged_bytes(tmp_path):
    """`x ] y -> z.png` used to truncate the reference and silently drop the image."""
    import hashlib
    paste_dir = tmp_path / "pastes"
    src = _write(tmp_path / "x ] y \u2192 z.png", _png(21, 13))
    staged = stage_image_file(src, paste_dir, 71)
    placeholder = image_placeholder(71, staged, origin=src)
    resolved = image_refs(f"what is this? {placeholder}", paste_dir)
    assert resolved == [staged]
    assert hashlib.sha256(resolved[0].read_bytes()).hexdigest() == \
        hashlib.sha256(src.read_bytes()).hexdigest()


def test_text_paste_expansion_is_unaffected_by_an_image_placeholder(tmp_path):
    """Text pastes keep their own contract: expand to contents, never eaten by the image path."""
    from hermes_cli.cli_stream_mixin import CLIStreamMixin
    note = tmp_path / "note.txt"
    note.write_text("expanded body")
    staged = stage_image_file(_write(tmp_path / "a.png", _png(3, 3)), tmp_path / "pastes", 81)
    text = (f"[Pasted text #1: 1 lines \u2192 {note}] and "
            f"{image_placeholder(81, staged)}")
    out = object.__new__(CLIStreamMixin)._expand_paste_references(text)
    assert "expanded body" in out
    assert "[Pasted text #" not in out
    assert image_placeholder(81, staged) in out          # image reference left intact
    assert image_refs(out, tmp_path / "pastes") == [staged]


def _cli_instance():
    """A HermesCLI with no __init__ side effects — enough to call a pure method."""
    return object.__new__(HermesCLI)


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_capture_attaches_once_and_keeps_the_badge_in_step(tmp_path, monkeypatch):
    """The attachment badge and the placeholder must agree on the image's number.

    The badge row is numbered from ``_image_counter``; a capture that appended to
    ``_attached_images`` without touching it displayed "Image #0" for a placeholder that
    said "#1".
    """
    from hermes_cli.cli_terminal_input import _format_image_attachment_badges

    monkeypatch.setattr("hermes_cli.cli_terminal_mixin.get_hermes_home", lambda: tmp_path)
    src = _write(tmp_path / "copied.png", _png(24, 16))
    cli = _cli_instance()
    cli._attached_images = []
    cli._image_counter = 0
    cli._tui_paste_counter = 0

    with patch("hermes_cli.clipboard.clipboard_image_files", return_value=[src]):
        first = cli._capture_paste_image()
        second = cli._capture_paste_image()

    assert first is not None and second is not None
    path, placeholder = first
    again_path, again_placeholder = second
    assert placeholder.startswith("[Pasted image #1:")
    assert again_path == path and again_placeholder == placeholder  # one file, one attachment
    assert cli._attached_images == [path]
    assert (tmp_path / "pastes").is_dir()  # staged in the paste dir, not beside the source
    assert "#1" in _format_image_attachment_badges(cli._attached_images, cli._image_counter)

