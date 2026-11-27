"""Pasted images share the text-paste contract: bytes staged in ``~/.hermes/pastes/``.

A large text paste collapses to ``paste_<n>_<HHMMSS>.txt`` and the composer keeps
``[Pasted text #n: L lines -> path]``; the agent expands it at submit time. This module
gives an image the same treatment: the bytes are copied into that same directory (never
re-encoded when the clipboard carries a *file*), and the composer keeps

    [Pasted image #3: 1280x720 PNG (copied from /home/x/shot.png) -> paste_3_215002.png]

a plain local path that is readable for the rest of the session and after the clipboard
has moved on.

Source order, deliberately:

1. an image *file* on the clipboard (Explorer/Finder/``file://`` URI list) — byte-exact,
   no re-encode, and any path the OS can open works (other volumes, mapped drives, SMB,
   NFS);
2. a raw clipboard bitmap — handed to :mod:`hermes_cli.clipboard`, then checked against
   the same magic-byte bar as a file source (a broken extraction must not become a
   broken attachment).

Header parsing is stdlib only (PNG/GIF/BMP/JPEG/WebP/TIFF); Pillow is a fallback when it
happens to be importable, never a requirement. This module is a leaf: it imports
``hermes_constants``/``clipboard`` lazily inside functions so ``cli.py`` can import it
from any mixin without a cycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import struct
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Containers a paste may carry. Mirrors ``clipboard._FILE_IMAGE_EXTS``.
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"})

ARROW = "\u2192"

#: Readable reference. Group 1 is the counter; group 2 is whatever sits between the last
#: arrow and the closing bracket (the staged path). A hostile file name can end the
#: reference early, which is why this is never the only resolution route.
REF_RE = re.compile(r"\[Pasted image #(\d+):[^\]]*?\u2192\s*(.+?)\]")

#: Counter-only probe. Survives a name containing ``]`` or an arrow.
COUNTER_RE = re.compile(r"\[Pasted image #(\d+):")

#: Sidecar: counter -> path (+ sha, size, origin). Fail-open everywhere — it only exists to
#: recover an ambiguous reference and to recognise a re-paste of the same bytes.
INDEX_NAME = ".image-index.json"
_INDEX_MAX = 400

#: (magic, display name, canonical suffix)
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "PNG", ".png"),
    (b"\xff\xd8\xff", "JPEG", ".jpg"),
    (b"GIF87a", "GIF", ".gif"),
    (b"GIF89a", "GIF", ".gif"),
    (b"BM", "BMP", ".bmp"),
    (b"II*\x00", "TIFF", ".tiff"),
    (b"MM\x00*", "TIFF", ".tiff"),
)


# ── directory + format sniffing ─────────────────────────────────────────


def paste_dir() -> Path:
    """``<hermes home>/pastes`` — the directory text pastes already use."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "pastes"
    except Exception:
        home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        return Path(home) / "pastes"


def sniff_format(path: Path | str) -> tuple[str, str] | None:
    """(display name, canonical suffix) from magic bytes; None when it is not an image."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
    except OSError:
        return None
    for magic, name, ext in _MAGIC:
        if head.startswith(magic):
            return name, ext
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP", ".webp"
    return None


def _png_dims(head: bytes):
    if len(head) >= 24 and head[12:16] == b"IHDR":
        return struct.unpack(">II", head[16:24])
    return None


def _gif_dims(head: bytes):
    if len(head) >= 10:
        return struct.unpack("<HH", head[6:10])
    return None


def _bmp_dims(head: bytes):
    if len(head) >= 26:
        width, height = struct.unpack("<ii", head[18:26])
        return abs(width), abs(height)
    return None


def _jpeg_dims(path: Path):
    """Walk JPEG segments to the first SOFn. Streamed on purpose: a multi-MB EXIF APP1
    must not hide the frame header behind a fixed-size read."""
    try:
        with open(path, "rb") as fh:
            if fh.read(2) != b"\xff\xd8":
                return None
            while True:
                marker = b""
                # Resync: fill bytes and stray data before a marker are legal.
                while marker != b"\xff":
                    marker = fh.read(1)
                    if not marker:
                        return None
                code = fh.read(1)
                while code == b"\xff":  # fill byte run
                    code = fh.read(1)
                if not code:
                    return None
                m = code[0]
                if m in (0xD8, 0xD9, 0x01) or 0xD0 <= m <= 0xD7:  # standalone markers
                    continue
                if m == 0xDA:  # scan started without a frame header
                    return None
                raw = fh.read(2)
                if len(raw) < 2:
                    return None
                seg_len = struct.unpack(">H", raw)[0]
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    frame = fh.read(5)
                    if len(frame) < 5:
                        return None
                    height, width = struct.unpack(">HH", frame[1:5])
                    return width, height
                fh.seek(seg_len - 2, 1)  # length counts its own two bytes
    except OSError:
        return None


def _webp_dims(head: bytes):
    if len(head) < 30:
        return None
    chunk = head[12:16]
    if chunk == b"VP8X":
        return (int.from_bytes(head[24:27], "little") + 1,
                int.from_bytes(head[27:30], "little") + 1)
    if chunk == b"VP8 ":
        width, height = struct.unpack("<HH", head[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and len(head) >= 25:
        bits = int.from_bytes(head[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def dimensions(path: Path | str) -> tuple[int, int] | None:
    """(width, height) from the container header; Pillow only when the header fails."""
    path = Path(path)
    fmt = sniff_format(path)
    if fmt is not None:
        try:
            with open(path, "rb") as fh:
                head = fh.read(32)
        except OSError:
            return None
        name = fmt[0]
        if name == "PNG":
            return _png_dims(head)
        if name == "GIF":
            return _gif_dims(head)
        if name == "BMP":
            return _bmp_dims(head)
        if name == "JPEG":
            return _jpeg_dims(path)
        if name == "WEBP":
            return _webp_dims(head)
    try:  # TIFF and exotica: only if Pillow is already installed
        from PIL import Image  # noqa: PLC0415
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def describe(path: Path | str) -> str:
    """``1280x720 PNG`` when the header parses, otherwise just ``PNG``/the suffix."""
    path = Path(path)
    fmt = sniff_format(path)
    name = fmt[0] if fmt else (path.suffix.lstrip(".").upper() or "IMAGE")
    dims = dimensions(path)
    return f"{dims[0]}x{dims[1]} {name}" if dims else name


# ── sidecar index ───────────────────────────────────────────────────────


def _sha256(path: Path | str) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def load_index(directory: Path | str | None = None) -> list[dict]:
    """Counter -> path entries, oldest first, vanished files dropped. Never raises."""
    directory = Path(directory) if directory is not None else paste_dir()
    try:
        raw = json.loads((directory / INDEX_NAME).read_text(encoding="utf-8"))
        entries = raw.get("entries") if isinstance(raw, dict) else raw
    except (OSError, ValueError):
        return []
    out: list[dict] = []
    for entry in entries or []:
        try:
            path = Path(entry["path"])
            if not path.is_file():
                continue
            out.append({"id": int(entry["id"]), "path": str(path), "sha": entry.get("sha"),
                        "size": entry.get("size"), "origin": entry.get("origin")})
        except (KeyError, TypeError, ValueError):
            continue
    return out[-_INDEX_MAX:]


def remember(directory: Path | str, counter: int, path: Path, *, sha: str | None = None,
             size: int | None = None, origin: Path | str | None = None) -> None:
    """Record counter -> path so an ambiguous reference is still resolvable."""
    try:
        directory = Path(directory)
        entries = [e for e in load_index(directory) if e["id"] != int(counter)]
        if size is None:
            size = path.stat().st_size if path.is_file() else None
        entries.append({
            "id": int(counter), "path": str(path),
            "sha": sha if sha is not None else _sha256(path), "size": size,
            "origin": str(origin) if origin else None,
            "at": datetime.now().isoformat(timespec="seconds")})
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / (INDEX_NAME + ".tmp")
        tmp.write_text(json.dumps({"entries": entries[-_INDEX_MAX:]}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, directory / INDEX_NAME)
    except Exception as exc:  # bookkeeping must never break a paste
        logger.debug("paste image index not updated: %s", exc)


def _same_bytes_staged(directory: Path | str, sha: str, size: int) -> Path | None:
    """An existing paste file with these exact bytes — the same image pasted twice."""
    for entry in reversed(load_index(directory)):
        if entry.get("sha") != sha:
            continue
        candidate = Path(entry["path"])
        try:
            if candidate.stat().st_size == size:
                return candidate
        except OSError:
            continue
    return None


# ── staging ─────────────────────────────────────────────────────────────


def next_path(directory: Path, counter: int, suffix: str = ".png") -> Path:
    """``paste_<counter>_<HHMMSS><suffix>``, never reusing an existing name.

    Counters restart every CLI session, so two pastes in the same second (or a restart
    within the same second) must not overwrite: ``_2``, ``_3``, ... disambiguate.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"paste_{counter}_{datetime.now().strftime('%H%M%S')}"
    ext = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    candidate = directory / f"{stem}{ext}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{n}{ext}"
        n += 1
    return candidate


def _settle_suffix(path: Path) -> Path:
    """Rename to the real container suffix when the name lied (a bitmap extract always
    lands as ``.png``, even when the clipboard actually held JPEG or GIF bytes)."""
    fmt = sniff_format(path)
    if fmt is None:
        return path
    wanted = fmt[1]
    aliases = {wanted}
    if wanted == ".jpg":
        aliases.add(".jpeg")
    if wanted == ".tiff":
        aliases.add(".tif")
    if path.suffix.lower() in aliases:
        return path
    target = path.with_suffix(wanted)
    try:
        if target.exists():
            target.unlink()
        path.rename(target)
        return target
    except OSError as exc:
        logger.debug("could not settle suffix %s -> %s: %s", path, target, exc)
        return path


def stage_path(src: Path | str, directory: Path, counter: int, *, dedupe: bool = True) -> Path:
    """Copy an image file into *directory* byte-for-byte and return the staged path.

    The copy is verified (size, then sha256) before it is published under its final name,
    so a short read — a network mount that dropped mid-copy — never survives. With
    *dedupe*, bytes already staged return the existing file: one screenshot pasted twice
    is one file and one attachment.
    """
    src = Path(os.path.expanduser(str(src)))
    if not src.is_file():
        raise FileNotFoundError(f"not a file: {src}")
    fmt = sniff_format(src)  # trust the bytes, not the extension
    if fmt is None:
        raise ValueError(f"not a readable image: {src}")
    size = src.stat().st_size
    sha = _sha256(src)
    if dedupe and sha:
        existing = _same_bytes_staged(directory, sha, size)
        if existing is not None:
            remember(directory, counter, existing, sha=sha, size=size, origin=src)
            return existing
    suffix = src.suffix if src.suffix.lower() in IMAGE_EXTS else fmt[1]
    dest = next_path(directory, counter, suffix)
    try:
        if src.resolve() == dest.resolve():
            remember(directory, counter, dest, sha=sha, size=size, origin=src)
            return dest
    except OSError:
        pass
    part = dest.with_name(dest.name + ".part")
    try:
        shutil.copyfile(src, part)
        if part.stat().st_size != size:
            raise OSError(f"short copy: {src} -> {dest}")
        if sha and _sha256(part) != sha:
            raise OSError(f"digest mismatch after copy: {src} -> {dest}")
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    dest = _settle_suffix(dest)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise OSError(f"staged image is empty: {dest}")
    remember(directory, counter, dest, sha=sha, size=size, origin=src)
    return dest


def capture_clipboard(directory: Path, counter: int) -> tuple[Path, Path | None] | None:
    """Stage whatever image the clipboard holds.

    Returns ``(staged, source)`` — *source* is set when a clipboard *file* was copied (the
    placeholder names where it came from), ``None`` for a raw bitmap — or ``None`` when the
    clipboard holds nothing readable.
    """
    from hermes_cli.clipboard import clipboard_image_files, save_clipboard_image

    for src in clipboard_image_files():
        try:
            return stage_path(src, directory, counter), src
        except Exception as exc:  # gone, unreadable, dead network mount
            logger.debug("clipboard image file %s could not be staged: %s", src, exc)

    dest = next_path(directory, counter, ".png")
    try:
        if save_clipboard_image(dest):
            dest = _settle_suffix(dest)
            # Same bar as a file source: junk from a bitmap extraction is never passed on.
            if not dest.is_file() or dest.stat().st_size == 0 or sniff_format(dest) is None:
                dest.unlink(missing_ok=True)
                return None
            size = dest.stat().st_size
            sha = _sha256(dest)
            if sha:
                existing = _same_bytes_staged(directory, sha, size)
                if existing is not None:
                    dest.unlink(missing_ok=True)
                    remember(directory, counter, existing, sha=sha, size=size)
                    return existing, None
            remember(directory, counter, dest, sha=sha, size=size)
            return dest, None
    except Exception as exc:
        logger.debug("clipboard bitmap extraction failed: %s", exc)
    dest.unlink(missing_ok=True)
    return None


# ── reference in / out ──────────────────────────────────────────────────

_ORIGIN_ESCAPES = {"[": "(", "]": ")", ARROW: "->"}


def _display_origin(origin: Path | str) -> str:
    """A copied file may be named with ``]`` or an arrow; either would end the readable
    reference early, so both are neutralised in the *displayed* origin only."""
    text = str(origin)
    for raw, safe in _ORIGIN_ESCAPES.items():
        text = text.replace(raw, safe)
    return text


def placeholder(counter: int, path: Path | str, *, origin: Path | str | None = None) -> str:
    """The composer reference for a staged image — same contract as a text placeholder."""
    label = describe(path)
    if origin is not None and Path(origin) != Path(path):
        return (f"[Pasted image #{counter}: {label} "
                f"(copied from {_display_origin(origin)}) {ARROW} {path}]")
    return f"[Pasted image #{counter}: {label} {ARROW} {path}]"


def refs(text: str | None, directory: Path | str | None = None) -> list[Path]:
    """Local image paths named by paste placeholders in *text*, in order, de-duplicated.

    Resolution order per counter: the literal path inside the reference (it names exactly
    the file the prompt means, so an older recalled prompt keeps its own image), then the
    sidecar index for references whose readable part is ambiguous or predates the index.
    """
    if not isinstance(text, str) or "[Pasted image #" not in text:
        return []
    directory = Path(directory) if directory is not None else paste_dir()
    indexed = {entry["id"]: Path(entry["path"]) for entry in load_index(directory)}
    literal = {int(counter): Path(raw.strip())
               for counter, raw in REF_RE.findall(text)}
    out: list[Path] = []
    for raw_counter in COUNTER_RE.findall(text):
        counter = int(raw_counter)
        for candidate in (literal.get(counter), indexed.get(counter)):
            if candidate is None:
                continue
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            if candidate not in out:
                out.append(candidate)
            break
    return out


__all__ = [
    "ARROW", "COUNTER_RE", "IMAGE_EXTS", "INDEX_NAME", "REF_RE", "capture_clipboard",
    "describe", "dimensions", "load_index", "next_path", "paste_dir", "placeholder",
    "refs", "remember", "sniff_format", "stage_path",
]
