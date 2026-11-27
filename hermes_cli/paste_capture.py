"""Pasted-image capture — the image half of ``~/.hermes/pastes/``.

Text pastes collapse to ``paste_<n>_<HHMMSS>.txt`` plus a ``[Pasted text #n: N lines →
path]`` placeholder that the agent expands at send time. This module gives images the
identical treatment: the bytes land in the same directory and the composer receives
``[Pasted image #n: 1280x720 PNG → C:\\...\\pastes\\paste_3_231502.png]`` — a local path
that is readable immediately, exactly like a text paste.

Two clipboard sources, in this order (a file copy is byte-exact, so it wins over a bitmap
round-trip):

1. an image *file* on the clipboard (Explorer / Finder / ``file://`` URI list) — copied
   without re-encoding; any path the OS can read works, including other local volumes,
   mapped drives, SMB shares and NFS mounts;
2. a raw clipboard bitmap — extracted through :mod:`hermes_cli.clipboard`.

No third-party deps: dimensions come from the container header (PNG/GIF/BMP/JPEG/WebP);
Pillow is used only as a fallback when it happens to be importable.
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

#: Image containers a paste may carry. Keep in sync with ``hermes_cli.clipboard._FILE_IMAGE_EXTS``.
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"})

#: ``[Pasted image #3: 1280x720 PNG (copied from E:\a.png) → C:\...\paste_3_231502.png]``
#: Group 1 is the counter, group 2 the local path (the last thing before the closing bracket).
PASTE_IMAGE_REF_RE = re.compile(r"\[Pasted image #(\d+):[^\]]*?\u2192\s*(.+?)\]")

#: Counter-only probe. A path may legally contain ``]`` or ``→`` (a file the user copied can
#: be named anything), which makes the human-readable reference ambiguous — so the index below
#: is the authoritative counter → path map, and this scan is what survives such a name.
PASTE_IMAGE_COUNTER_RE = re.compile(r"\[Pasted image #(\d+):")

#: Sidecar in the paste directory. Small, append-only by rewrite, and fail-open: it only exists
#: to recover the path when the readable reference is ambiguous, and to spot a re-paste of the
#: same bytes. Never trusted without re-checking the file on disk.
INDEX_NAME = ".image-index.json"
_INDEX_MAX = 400

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "PNG", ".png"),
    (b"\xff\xd8\xff", "JPEG", ".jpg"),
    (b"GIF87a", "GIF", ".gif"),
    (b"GIF89a", "GIF", ".gif"),
    (b"BM", "BMP", ".bmp"),
    (b"II*\x00", "TIFF", ".tiff"),
    (b"MM\x00*", "TIFF", ".tiff"),
)


# ── format / dimension sniffing ──────────────────────────────────────────


def _detect_format(path: Path) -> tuple[str, str] | None:
    """(display name, canonical suffix) from magic bytes; None when it is not an image."""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
    except OSError:
        return None
    for magic, name, ext in _MAGIC:
        if head.startswith(magic):
            return name, ext
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP", ".webp"
    return None


def _png_size(head: bytes) -> tuple[int, int] | None:
    if len(head) >= 24 and head[12:16] == b"IHDR":
        return struct.unpack(">II", head[16:24])
    return None


def _gif_size(head: bytes) -> tuple[int, int] | None:
    if len(head) >= 10:
        return struct.unpack("<HH", head[6:10])
    return None


def _bmp_size(head: bytes) -> tuple[int, int] | None:
    if len(head) >= 26:
        w, h = struct.unpack("<ii", head[18:26])
        return abs(w), abs(h)
    return None


def _jpeg_size(path: Path) -> tuple[int, int] | None:
    """Walk JPEG segments (streamed, so a multi-MB EXIF APP1 cannot hide the frame header)
    until the first SOFn marker, then read its height/width."""
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                byte = f.read(1)
                while byte and byte != b"\xff":  # resync on fill bytes / stray data
                    byte = f.read(1)
                if not byte:
                    return None
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker:
                    return None
                m = marker[0]
                if m in (0xD8, 0xD9, 0x01) or 0xD0 <= m <= 0xD7:
                    continue
                if m == 0xDA:  # start of scan with no frame header seen: not a readable JPEG
                    return None
                raw_len = f.read(2)
                if len(raw_len) < 2:
                    return None
                seg_len = struct.unpack(">H", raw_len)[0]
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    frame = f.read(5)
                    if len(frame) < 5:
                        return None
                    h, w = struct.unpack(">HH", frame[1:5])
                    return w, h
                f.seek(seg_len - 2, 1)
    except OSError:
        return None


def _webp_size(path: Path) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as f:
            head = f.read(30)
    except OSError:
        return None
    if len(head) < 30:
        return None
    chunk = head[12:16]
    if chunk == b"VP8X":
        w = int.from_bytes(head[24:27], "little") + 1
        h = int.from_bytes(head[27:30], "little") + 1
        return w, h
    if chunk == b"VP8 " and len(head) >= 30:
        w, h = struct.unpack("<HH", head[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if chunk == b"VP8L" and len(head) >= 25:
        bits = int.from_bytes(head[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def image_dimensions(path: Path) -> tuple[int, int] | None:
    """(width, height) from the container header, Pillow only if the header path fails."""
    fmt = _detect_format(path)
    if fmt is not None:
        try:
            with open(path, "rb") as f:
                head = f.read(32)
        except OSError:
            return None
        name = fmt[0]
        if name == "PNG":
            return _png_size(head)
        if name == "GIF":
            return _gif_size(head)
        if name == "BMP":
            return _bmp_size(head)
        if name == "JPEG":
            return _jpeg_size(path)
        if name == "WEBP":
            return _webp_size(head)
    try:  # TIFF and anything exotic: only if Pillow is already present
        from PIL import Image  # noqa: PLC0415 — optional dependency
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def describe_image(path: Path) -> str:
    """Human-sized label for the placeholder, e.g. ``1280x720 PNG`` or ``PNG``."""
    fmt = _detect_format(path)
    name = fmt[0] if fmt else (path.suffix.lstrip(".").upper() or "IMAGE")
    dims = image_dimensions(path)
    return f"{dims[0]}x{dims[1]} {name}" if dims else name


# ── paste index (counter → path, and byte dedupe) ────────────────────────


def _default_paste_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "pastes"
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")) / "pastes"


def _sha256_file(path: Path) -> str | None:
    """Streamed digest; None when the file cannot be read."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def load_index(paste_dir: Path | str | None = None) -> list[dict]:
    """Counter → path entries, oldest first, with vanished files dropped.

    Fail-open by design: a corrupt or missing index only costs the fallback resolution path
    (the readable reference still works for well-behaved file names).
    """
    directory = Path(paste_dir) if paste_dir is not None else _default_paste_dir()
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


def remember_paste(paste_dir: Path | str, counter: int, path: Path, *,
                   sha: str | None = None, size: int | None = None,
                   origin: Path | str | None = None) -> None:
    """Record counter → path so an ambiguous reference can still be resolved later."""
    try:
        directory = Path(paste_dir)
        entries = [e for e in load_index(directory) if e["id"] != int(counter)]
        if size is None:
            size = path.stat().st_size if path.is_file() else None
        entries.append({
            "id": int(counter), "path": str(path),
            "sha": sha if sha is not None else _sha256_file(path), "size": size,
            "origin": str(origin) if origin else None,
            "at": datetime.now().isoformat(timespec="seconds")})
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / (INDEX_NAME + ".tmp")
        tmp.write_text(json.dumps({"entries": entries[-_INDEX_MAX:]}, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, directory / INDEX_NAME)
    except Exception as e:  # never let bookkeeping break a paste
        logger.debug("paste index not updated: %s", e)


def _already_staged(paste_dir: Path | str, sha: str, size: int) -> Path | None:
    """An existing paste file with these exact bytes/size — the same image pasted twice."""
    for entry in reversed(load_index(paste_dir)):
        if entry.get("sha") != sha:
            continue
        candidate = Path(entry["path"])
        try:
            if candidate.stat().st_size == size:
                return candidate
        except OSError:
            continue
    return None


# ── staging into the paste directory ─────────────────────────────────────


def next_paste_path(paste_dir: Path, counter: int, suffix: str = ".png") -> Path:
    """``<paste_dir>/paste_<counter>_<HHMMSS><suffix>`` — same shape as collapsed text pastes.

    Never reuses a name that already exists: a counter restarting in a second session must not
    overwrite an image pasted earlier in the day (`_2`, `_3`, … disambiguate).
    """
    paste_dir.mkdir(parents=True, exist_ok=True)
    stem = f"paste_{counter}_{datetime.now().strftime('%H%M%S')}"
    ext = suffix.lower()
    candidate = paste_dir / f"{stem}{ext}"
    n = 2
    while candidate.exists():
        candidate = paste_dir / f"{stem}_{n}{ext}"
        n += 1
    return candidate


def _settle_extension(path: Path) -> Path:
    """Rename *path* to its real container suffix when the clipboard lied about the format
    (Windows file-drop extraction names every payload ``.png`` even for JPEG bytes)."""
    fmt = _detect_format(path)
    if fmt is None:
        return path
    if path.suffix.lower() not in {fmt[1], ".jpeg" if fmt[1] == ".jpg" else fmt[1]}:
        target = path.with_suffix(fmt[1])
        try:
            if target.exists():
                target.unlink()
            path.rename(target)
            return target
        except OSError as e:
            logger.debug("could not rename %s -> %s: %s", path, target, e)
    return path


def stage_image_file(src: Path | str, paste_dir: Path, counter: int, *,
                     dedupe: bool = True) -> Path:
    """Copy an image file into the paste directory, byte-for-byte, and return its path.

    Works for any path the OS can read (other volumes, mapped drives, SMB, NFS). The copy is
    verified against the source digest before it is published; a partial copy never survives.
    With *dedupe*, re-staging bytes that are already in the directory returns the existing
    file instead of duplicating it (pasting the same screenshot twice is one file, one image).
    """
    src = Path(os.path.expanduser(str(src)))
    if not src.is_file():
        raise FileNotFoundError(f"not a file: {src}")
    # Trust the bytes, not the name: a renamed .txt would only become a broken attachment.
    fmt = _detect_format(src)
    if fmt is None:
        raise ValueError(f"not a readable image: {src}")
    size = src.stat().st_size
    sha = _sha256_file(src)
    if dedupe and sha:
        existing = _already_staged(paste_dir, sha, size)
        if existing is not None:
            remember_paste(paste_dir, counter, existing, sha=sha, size=size, origin=src)
            return existing
    suffix = src.suffix if src.suffix.lower() in IMAGE_EXTS else fmt[1]
    dest = next_paste_path(paste_dir, counter, suffix)
    if src.resolve() == dest.resolve():
        return dest
    part = dest.with_name(dest.name + ".part")
    try:
        shutil.copyfile(src, part)
        if part.stat().st_size != size:
            raise OSError(f"short copy: {src} -> {dest}")
        if sha and _sha256_file(part) != sha:
            raise OSError(f"digest mismatch after copy: {src} -> {dest}")
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    dest = _settle_extension(dest)
    if not dest.exists() or dest.stat().st_size == 0:
        raise OSError(f"staged image is empty: {dest}")
    remember_paste(paste_dir, counter, dest, sha=sha, size=size, origin=src)
    return dest


def capture_clipboard_image(paste_dir: Path, counter: int) -> tuple[Path, Path | None] | None:
    """Stage whatever image the clipboard holds into *paste_dir*.

    Returns ``(staged_path, source_path)`` — *source_path* is set when a clipboard *file* was
    copied (so the placeholder can name where it came from), ``None`` for a raw bitmap — or
    ``None`` when the clipboard holds no image we can read.
    """
    from hermes_cli.clipboard import clipboard_image_files, save_clipboard_image

    files = clipboard_image_files()
    for src in files:
        try:
            return stage_image_file(src, paste_dir, counter), src
        except Exception as e:  # unreadable source (permissions, dead network mount, gone)
            logger.debug("clipboard image file %s could not be staged: %s", src, e)

    dest = next_paste_path(paste_dir, counter, ".png")
    try:
        if save_clipboard_image(dest):
            dest = _settle_extension(dest)
            # A bitmap extraction can hand back junk (a partial PNG, a mislabelled container):
            # hold it to the same bar as a file source instead of passing a broken image on.
            if not dest.exists() or dest.stat().st_size == 0 or _detect_format(dest) is None:
                dest.unlink(missing_ok=True)
                return None
            size = dest.stat().st_size
            sha = _sha256_file(dest)
            if sha:
                existing = _already_staged(paste_dir, sha, size)
                if existing is not None:
                    dest.unlink(missing_ok=True)
                    remember_paste(paste_dir, counter, existing, sha=sha, size=size)
                    return existing, None
            remember_paste(paste_dir, counter, dest, sha=sha, size=size)
            return dest, None
    except Exception as e:
        logger.debug("clipboard bitmap extraction failed: %s", e)
    dest.unlink(missing_ok=True)
    return None


_ORIGIN_ESCAPES = {"[": "(", "]": ")", chr(0x2192): "->"}


def _display_origin(origin: Path | str) -> str:
    """Source path as it may appear inside the reference: brackets and arrows in a file name
    would otherwise end or split the reference early."""
    text = str(origin)
    for bad, good in _ORIGIN_ESCAPES.items():
        text = text.replace(bad, good)
    return text


def image_placeholder(counter: int, path: Path, *, origin: Path | None = None) -> str:
    """The composer reference for a staged image — same contract as a text-paste placeholder."""
    label = describe_image(path)
    if origin is not None and Path(origin) != Path(path):
        return f"[Pasted image #{counter}: {label} (copied from {_display_origin(origin)}) \u2192 {path}]"
    return f"[Pasted image #{counter}: {label} \u2192 {path}]"


def image_refs(text: str | None, paste_dir: Path | str | None = None) -> list[Path]:
    """Local image paths referenced by paste placeholders in *text*, in order, de-duplicated.

    The literal path in the reference is tried first (it names exactly the file the prompt
    refers to, so a recalled older prompt keeps its own image); the paste index is the fallback
    for references whose readable part is ambiguous — a source file named with `]` or `→` — and
    for pastes staged before the index existed.
    """
    if not isinstance(text, str) or "[Pasted image #" not in text:
        return []
    directory = Path(paste_dir) if paste_dir is not None else _default_paste_dir()
    indexed = {entry["id"]: Path(entry["path"]) for entry in load_index(directory)}
    literal = {int(counter): Path(raw.strip()) for counter, raw in PASTE_IMAGE_REF_RE.findall(text)}
    out: list[Path] = []
    for raw_counter in PASTE_IMAGE_COUNTER_RE.findall(text):
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
    "IMAGE_EXTS", "INDEX_NAME", "PASTE_IMAGE_COUNTER_RE", "PASTE_IMAGE_REF_RE",
    "capture_clipboard_image", "describe_image", "image_dimensions", "image_placeholder",
    "image_refs", "load_index", "next_paste_path", "remember_paste", "stage_image_file",
]
