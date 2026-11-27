"""Clipboard image extraction and text write for macOS, Windows, Linux, and WSL2.

No Python deps — only OS-level CLI tools: macOS osascript (always present) / pngpaste (optional)
plus a file-url fallback for Finder copies; Windows and WSL2 PowerShell via WinForms,
Get-Clipboard, then a file-drop fallback; Linux wl-paste (Wayland), xclip (X11).
"""

import base64
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hermes_constants import is_wsl as _is_wsl

logger = logging.getLogger(__name__)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_TEXT = dict(capture_output=True, text=True, encoding='utf-8', errors='replace', stdin=subprocess.DEVNULL)
_PS_FLAGS = ("-NoProfile", "-NonInteractive")
_FILE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _probe(argv: list, timeout: int, ok, *, missing: str | None = None) -> bool:
    """Run a text-mode probe; True when it ran and ``ok(result)`` holds.
    A missing executable logs *missing* (when given); every other failure is silent."""
    try:
        return bool(ok(subprocess.run(argv, timeout=timeout, **_TEXT)))
    except FileNotFoundError:
        if missing:
            logger.debug(missing)
    except Exception:
        pass
    return False


def _pipe_to_file(argv: list, dest: Path) -> bool:
    """Run *argv* with stdout redirected into *dest*; True when a non-empty file resulted."""
    with open(dest, "wb") as f:
        subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.DEVNULL, timeout=5, check=True)
    return _nonempty(dest)


def _linux_backends():
    """(enabled, has_image, save) in Linux fallthrough order: WSL → Wayland → X11
    (a failed WSL probe falls through — WSLg might have wl-paste or xclip working)."""
    return (
        (_is_wsl(), _wsl_has_image, _wsl_save),
        (bool(os.environ.get("WAYLAND_DISPLAY")), _wayland_has_image, _wayland_save),
        (True, _xclip_has_image, _xclip_save))


def save_clipboard_image(dest: Path) -> bool:
    """Save the clipboard image to *dest* as PNG; True when an image was found and written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        # pngpaste first (fast, handles more formats); osascript is the always-present fallback.
        return _macos_pngpaste(dest) or _macos_osascript(dest)
    return (_windows_save if sys.platform == "win32" else _linux_save)(dest)


def has_clipboard_image() -> bool:
    """Quick check: does the clipboard currently contain an image?"""
    if sys.platform == "darwin":
        return _macos_has_image()
    if sys.platform == "win32":
        return _windows_has_image()
    return any(enabled and has() for enabled, has, _ in _linux_backends())


# ── Text write (native tools, mirrors ui-tui/src/lib/clipboard.ts) ──────

def _write_clipboard_commands(data: bytes) -> list:
    """(argv, run_kwargs) candidates for writing *data*, in platform fallback order."""
    # PowerShell decodes piped stdin with the system ANSI code page (e.g. CP936), not UTF-8, so
    # stdin-based writes mangle CJK/emoji. Base64 the UTF-8 bytes and decode inside PowerShell
    # instead (same approach as the TUI's writeClipboardText).
    b64 = base64.b64encode(data).decode("ascii")
    ps_argv = [*_PS_FLAGS, "-Command", "Set-Clipboard -Value ([System.Text.Encoding]::UTF8"
               f".GetString([System.Convert]::FromBase64String('{b64}')))"]
    ps_kw, pipe = {"stdin": subprocess.DEVNULL}, {"input": data}
    linux = sys.platform not in ("darwin", "win32")
    return [(argv, kw) for enabled, argv, kw in (
        (sys.platform == "darwin", ["pbcopy"], pipe),
        (sys.platform == "win32", ["powershell", *ps_argv], ps_kw),
        (linux and _is_wsl(), ["powershell.exe", *ps_argv], ps_kw),
        (linux and os.environ.get("WAYLAND_DISPLAY"), ["wl-copy", "--type", "text/plain"], pipe),
        (linux, ["xclip", "-selection", "clipboard", "-in"], pipe),
        (linux, ["xsel", "--clipboard", "--input"], pipe),
    ) if enabled]


def is_remote_shell_session(env=None) -> bool:
    """True inside an SSH session (mirrors ui-tui/src/lib/terminalSetup.ts). Over SSH, native
    clipboard tools write the REMOTE machine's clipboard (or an X-forwarded one), which is almost
    never what the user wants — OSC 52 reaches the LOCAL terminal instead."""
    e = os.environ if env is None else env
    return bool(e.get("SSH_CONNECTION") or e.get("SSH_TTY") or e.get("SSH_CLIENT"))


def write_clipboard_text(text: str) -> bool:
    """Write *text* to the clipboard via native tools; fallback order matches the TUI: pbcopy →
    Windows/WSL PowerShell Set-Clipboard → wl-copy → xclip → xsel. Returns True if any backend
    succeeded; callers fall back to OSC 52 on False."""
    for argv, kw in _write_clipboard_commands(text.encode("utf-8")):
        try:
            if subprocess.run(argv, timeout=10, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, **kw).returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


# ── macOS ────────────────────────────────────────────────────────────────

def _osascript(expr: str, timeout: int = 3) -> str:
    """stdout of an osascript expression; "" on any failure."""
    try:
        r = subprocess.run(["osascript", "-e", expr], timeout=timeout, **_TEXT)
        return r.stdout if r.returncode == 0 else ""
    except Exception as e:
        logger.debug("osascript probe failed: %s", e)
        return ""


def _macos_has_bitmap(info: str) -> bool:
    return "«class PNGf»" in info or "«class TIFF»" in info


def _macos_clipboard_file_image() -> Path | None:
    """Local image path when the clipboard holds a Finder file-url: Cmd+C on a file puts
    «class furl» (no bitmap) on the pasteboard, yet other apps paste it as an image."""
    path = Path(_osascript("POSIX path of (the clipboard as «class furl»)").strip())
    return path if path.suffix.lower() in _FILE_IMAGE_EXTS and os.path.isfile(path) else None


def _macos_has_image() -> bool:
    info = _osascript("clipboard info")
    return _macos_has_bitmap(info) or (
        "«class furl»" in info and _macos_clipboard_file_image() is not None)


def _macos_save_file_image(dest: Path) -> bool:
    src = _macos_clipboard_file_image()
    if src is None:
        return False
    try:
        shutil.copyfile(src, dest)
        if _is_png_file(dest) or (_convert_to_png(dest) and _is_png_file(dest)):
            return True
    except OSError as e:
        logger.debug("clipboard file-url extract failed: %s", e)
    dest.unlink(missing_ok=True)
    return False


def _macos_pngpaste(dest: Path) -> bool:
    """pngpaste (brew install pngpaste) — fastest, cleanest."""
    try:
        r = subprocess.run(["pngpaste", str(dest)], stdin=subprocess.DEVNULL, capture_output=True, timeout=3)
        return r.returncode == 0 and _nonempty(dest)
    except FileNotFoundError:
        pass  # pngpaste not installed
    except Exception as e:
        logger.debug("pngpaste failed: %s", e)
    return False


def _macos_osascript(dest: Path) -> bool:
    """osascript extraction (always available): bitmap PNGf, else a Finder file-url. One
    `clipboard info` listing gates both — the furl read is a pasteboard *content* access."""
    info = _osascript("clipboard info")
    if not _macos_has_bitmap(info):
        return "«class furl»" in info and _macos_save_file_image(dest)
    script = f'''try
  set imgData to the clipboard as «class PNGf»
  set f to open for access POSIX file "{dest}" with write permission
  write imgData to f
  close access f
on error
  return "fail"
end try
'''
    try:
        r = subprocess.run(["osascript", "-e", script], timeout=5, **_TEXT)
        return r.returncode == 0 and "fail" not in r.stdout and _nonempty(dest)
    except Exception as e:
        logger.debug("osascript clipboard extract failed: %s", e)
    return False


# ── PowerShell (native Windows powershell/pwsh + WSL2 powershell.exe) ─────

_FILEDROP_IMAGE_EXTS = ",".join(f"'{e}'" for e in sorted(_FILE_IMAGE_EXTS))
_PS_FILEDROP_HIT = (
    "try { "
    "$files = Get-Clipboard -Format FileDropList -ErrorAction Stop;"
    f"$exts = @({_FILEDROP_IMAGE_EXTS});"
    "$hit = $files | Where-Object { $exts -contains ([System.IO.Path]::GetExtension($_).ToLowerInvariant()) } | Select-Object -First 1;"
)

# (has_image, extract-as-base64-PNG) script pairs, tried in order.
_PS_IMAGE_STRATEGIES = (
    (  # .NET System.Windows.Forms.Clipboard
        "Add-Type -AssemblyName System.Windows.Forms;"
        "[System.Windows.Forms.Clipboard]::ContainsImage()",
        "Add-Type -AssemblyName System.Windows.Forms;"
        "Add-Type -AssemblyName System.Drawing;"
        "$img = [System.Windows.Forms.Clipboard]::GetImage();"
        "if ($null -eq $img) { exit 1 }"
        "$ms = New-Object System.IO.MemoryStream;"
        "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png);"
        "[System.Convert]::ToBase64String($ms.ToArray())"),
    (  # Get-Clipboard -Format Image (System.Drawing.Image or WPF BitmapSource)
        "try { "
        "$img = Get-Clipboard -Format Image -ErrorAction Stop;"
        "if ($null -ne $img) { 'True' } else { 'False' }"
        "} catch { 'False' }",
        "try { "
        "Add-Type -AssemblyName System.Drawing;"
        "Add-Type -AssemblyName PresentationCore;"
        "Add-Type -AssemblyName WindowsBase;"
        "$img = Get-Clipboard -Format Image -ErrorAction Stop;"
        "if ($null -eq $img) { exit 1 }"
        "$ms = New-Object System.IO.MemoryStream;"
        "if ($img -is [System.Drawing.Image]) {"
        "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)"
        "} elseif ($img -is [System.Windows.Media.Imaging.BitmapSource]) {"
        "$enc = New-Object System.Windows.Media.Imaging.PngBitmapEncoder;"
        "$enc.Frames.Add([System.Windows.Media.Imaging.BitmapFrame]::Create($img));"
        "$enc.Save($ms)"
        "} else { exit 2 }"
        "[System.Convert]::ToBase64String($ms.ToArray())"
        "} catch { exit 1 }"),
    (  # copied image *file* (Explorer file drop)
        _PS_FILEDROP_HIT
        + "if ($null -ne $hit) { 'True' } else { 'False' }"
        "} catch { 'False' }",
        _PS_FILEDROP_HIT
        + "if ($null -eq $hit) { exit 1 }"
        "[System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($hit))"
        "} catch { exit 1 }"))


def _ps_clipboard(exe: str, timeout: int, label: str, dest: Path | None = None) -> bool:
    """Probe (*dest* None) or extract (base64 PNG → *dest*) the Windows clipboard image via *exe*.
    A missing *exe* ends the whole chain (every script needs the same binary); any other failure
    logs (and drops a partial *dest*) then tries the next strategy."""
    for check, extract in _PS_IMAGE_STRATEGIES:
        try:
            argv = [exe, *_PS_FLAGS, "-Command", check if dest is None else extract]
            r = subprocess.run(argv, timeout=timeout, **_TEXT)
            if dest is None:
                if r.returncode == 0 and "True" in r.stdout:
                    return True
            elif r.returncode == 0 and r.stdout.strip():
                dest.write_bytes(base64.b64decode(r.stdout.strip(), validate=True))
                if _nonempty(dest):
                    return True
        except FileNotFoundError:
            logger.debug("%s not found — clipboard unavailable", exe)
            return False
        except Exception as e:
            logger.debug("%s clipboard image %s failed: %s", label,
                         "check" if dest is None else "extraction", e)
            if dest is not None:
                dest.unlink(missing_ok=True)
    return False


_ps_exe: str | None | bool = False  # resolved PowerShell executable; False = not yet checked


def _get_ps_exe() -> str | None:
    """First working PowerShell — ``powershell`` (5.1, always present) or ``pwsh`` (7+,
    optional) — cached per process; None when neither runs."""
    global _ps_exe
    if _ps_exe is False:
        _ps_exe = next((name for name in ("powershell", "pwsh") if _probe(
            [name, *_PS_FLAGS, "-Command", "echo ok"], 5,
            lambda r: r.returncode == 0 and "ok" in r.stdout)), None)
    return _ps_exe


def _windows_has_image() -> bool:
    ps = _get_ps_exe()
    return ps is not None and _ps_clipboard(ps, 5, "Windows")


def _windows_save(dest: Path) -> bool:
    ps = _get_ps_exe()
    if ps is None:
        logger.debug("No PowerShell found — Windows clipboard image paste unavailable")
        return False
    return _ps_clipboard(ps, 15, "Windows", dest)


# ── Clipboard image *files* (Explorer/Finder/URI-list copies) ────────────
#
# Distinct from the bitmap strategies above: copying a file in Explorer puts a FileDrop list
# on the clipboard, not a bitmap. Reading the file directly keeps the original bytes (no
# re-encode) and works for any path the OS can open — other volumes, mapped drives, SMB
# shares, NFS mounts. Paths travel as base64 so non-ASCII names survive the PowerShell hop.

_PS_FILE_LIST = (
    "try { "
    "$files = Get-Clipboard -Format FileDropList -ErrorAction Stop;"
    f"$exts = @({_FILEDROP_IMAGE_EXTS});"
    "@($files) | Where-Object { $exts -contains "
    "([System.IO.Path]::GetExtension($_.ToString()).ToLowerInvariant()) } | ForEach-Object { "
    "$p = $_.ToString(); if (Test-Path -LiteralPath $p -PathType Leaf) { "
    "[System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($p)) } }"
    "} catch { }"
)


def _windows_file_images() -> list[Path]:
    ps = _get_ps_exe()
    if ps is None:
        return []
    try:
        r = subprocess.run([ps, *_PS_FLAGS, "-Command", _PS_FILE_LIST], timeout=8, **_TEXT)
    except Exception as e:
        logger.debug("Windows clipboard file-list probe failed: %s", e)
        return []
    if r.returncode != 0:
        return []
    out: list[Path] = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            candidate = Path(base64.b64decode(line, validate=True).decode("utf-8"))
        except Exception:
            continue
        if candidate.suffix.lower() in _FILE_IMAGE_EXTS and os.path.isfile(candidate):
            out.append(candidate)
    return out


_URI_LIST_COMMANDS = (
    (("wl-paste", "--type", "text/uri-list"), True),
    (("xclip", "-selection", "clipboard", "-t", "text/uri-list", "-o"), False),
)


def _parse_uri_list(text: str) -> list[Path]:
    """``file://`` entries of a text/uri-list payload, as local paths."""
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname
    out: list[Path] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.lower().startswith("file://"):
            continue
        candidate = Path(url2pathname(unquote(urlparse(line).path)))
        if candidate.suffix.lower() in _FILE_IMAGE_EXTS and os.path.isfile(candidate):
            out.append(candidate)
    return out


def _linux_file_images() -> list[Path]:
    for argv, missing_log in _URI_LIST_COMMANDS:
        try:
            r = subprocess.run(argv, timeout=3, **_TEXT)
        except FileNotFoundError:
            if missing_log:
                logger.debug("%s not installed — URI-list clipboard unavailable", argv[0])
            continue
        except Exception as e:
            logger.debug("%s uri-list read failed: %s", argv[0], e)
            continue
        if r.returncode == 0:
            found = _parse_uri_list(r.stdout)
            if found:
                return found
    return []


def clipboard_image_files() -> list[Path]:
    """Image files currently sitting on the clipboard (copied in a file manager), if any."""
    try:
        if sys.platform == "darwin":
            single = _macos_clipboard_file_image()
            return [single] if single else []
        if sys.platform == "win32":
            return _windows_file_images()
        if _is_wsl():
            found = _windows_file_images_from_wsl()
            if found:
                return found
        return _linux_file_images()
    except Exception as e:  # a blocked clipboard must never break a paste
        logger.debug("clipboard image-file probe failed: %s", e)
        return []


def _windows_file_images_from_wsl() -> list[Path]:
    """WSL2: the Windows clipboard holds Windows paths — translate them to /mnt paths."""
    try:
        r = subprocess.run(["powershell.exe", *_PS_FLAGS, "-Command", _PS_FILE_LIST], timeout=8, **_TEXT)
    except Exception:
        return []
    out: list[Path] = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            win = Path(base64.b64decode(line, validate=True).decode("utf-8"))
        except Exception:
            continue
        try:
            translated = Path(subprocess.run(
                ["wslpath", "-u", str(win)], timeout=3, **_TEXT).stdout.strip() or "")
        except Exception:
            continue
        if translated and translated.suffix.lower() in _FILE_IMAGE_EXTS and os.path.isfile(translated):
            out.append(translated)
    return out


# ── Linux: WSL (powershell.exe) → Wayland (wl-paste) → X11 (xclip) ───────

def _linux_save(dest: Path) -> bool:
    return any(enabled and save(dest) for enabled, _, save in _linux_backends())


def _wsl_has_image() -> bool:
    return _ps_clipboard("powershell.exe", 8, "WSL")


def _wsl_save(dest: Path) -> bool:
    return _ps_clipboard("powershell.exe", 15, "WSL", dest)


_WAYLAND_MIME_PREFERENCE = ("image/png", "image/jpeg", "image/bmp", "image/gif", "image/webp")
_WL_LIST_TYPES = ["wl-paste", "--list-types"]
_WL_MISSING = "wl-paste not installed — Wayland clipboard unavailable"


def _wayland_has_image() -> bool:
    return _probe(_WL_LIST_TYPES, 3, lambda r: r.returncode == 0 and any(
        t.startswith("image/") for t in r.stdout.splitlines()), missing=_WL_MISSING)


def _wayland_save(dest: Path) -> bool:
    try:
        types_r = subprocess.run(_WL_LIST_TYPES, timeout=3, **_TEXT)
        types = types_r.stdout.splitlines() if types_r.returncode == 0 else ()
        mime = next((m for m in _WAYLAND_MIME_PREFERENCE if m in types), None)  # PNG preferred
        if not mime:
            return False
        # save_clipboard_image() promises a PNG. Wayland can offer JPEG/GIF/WebP/BMP payloads,
        # so every non-PNG result is normalized (and re-verified) before reporting success.
        if _pipe_to_file(["wl-paste", "--type", mime], dest) and (
                mime == "image/png" or (_convert_to_png(dest) and _is_png_file(dest))):
            return True
        dest.unlink(missing_ok=True)
    except FileNotFoundError:
        logger.debug(_WL_MISSING)
    except Exception as e:
        logger.debug("wl-paste clipboard extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False


def _convert_to_png(path: Path) -> bool:
    """Convert an image file to PNG in-place: Pillow first (likely in the venv), then ImageMagick.
    When neither works the file is left as-is — BMP is still usable for most APIs."""
    try:
        from PIL import Image
        Image.open(path).save(path, "PNG")
        return True
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Pillow BMP→PNG conversion failed: %s", e)
    tmp = path.with_suffix(".bmp")
    try:
        path.rename(tmp)
        r = subprocess.run(["convert", str(tmp), "png:" + str(path)], stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=5)
        if r.returncode == 0 and _nonempty(path):
            tmp.unlink(missing_ok=True)
            return True
        tmp.rename(path)  # convert failed — restore the original file
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            logger.debug("ImageMagick not installed — cannot convert BMP to PNG")
        else:
            logger.debug("ImageMagick BMP→PNG conversion failed: %s", e)
        if tmp.exists() and not path.exists():
            tmp.rename(path)
    return _nonempty(path)


def _is_png_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(len(_PNG_SIGNATURE)) == _PNG_SIGNATURE
    except OSError:
        return False


_XCLIP_TARGETS = ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"]


def _xclip_has_image() -> bool:
    return _probe(_XCLIP_TARGETS, 3, lambda r: r.returncode == 0 and "image/png" in r.stdout)


def _xclip_save(dest: Path) -> bool:
    if not _probe(_XCLIP_TARGETS, 3, lambda r: "image/png" in r.stdout,
                  missing="xclip not installed — X11 clipboard image paste unavailable"):
        return False
    try:
        return _pipe_to_file(["xclip", "-selection", "clipboard", "-t", "image/png", "-o"], dest)
    except Exception as e:
        logger.debug("xclip image extraction failed: %s", e)
        dest.unlink(missing_ok=True)
    return False
