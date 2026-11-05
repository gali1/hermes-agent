"""Video, done with the tools that exist here: ffmpeg frames + local OCR + local STT.

The point is that no model needs native video input. A clip becomes a labelled, timestamped text
representation - what is on screen, what is said, and how long it runs - and that text is what a
specialist (or the primary) reasons over. Frames are only sent to a vision model when asked for,
because a frame costs tokens and OCR costs nothing.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Frame:
    index: int
    timestamp_s: float
    path: str
    ocr_text: str = ""


@dataclass
class VideoContext:
    path: str
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    frames: List[Frame] = field(default_factory=list)
    transcript: str = ""
    transcript_error: str = ""
    notes: List[str] = field(default_factory=list)

    def as_text(self, *, max_chars: int = 6000) -> str:
        head = [f"VIDEO: {Path(self.path).name} ({self.duration_s:.1f}s"
                + (f", {self.width}x{self.height}" if self.width else "") + ")"]
        body: List[str] = []
        if self.transcript.strip():
            body.append("SPEECH (local STT, untimed single pass):\n" + self.transcript.strip())
        elif self.transcript_error:
            body.append(f"SPEECH: unavailable ({self.transcript_error})")
        onscreen = [f for f in self.frames if f.ocr_text.strip()]
        if onscreen:
            body.append("ON SCREEN (OCR per sampled frame, timestamped):")
            for frame in onscreen:
                snippet = " ".join(frame.ocr_text.split())[:400]
                body.append(f"  [{frame.timestamp_s:6.1f}s] {snippet}")
        elif self.frames:
            body.append(f"ON SCREEN: no readable text in {len(self.frames)} sampled frame(s)")
        if self.notes:
            body.append("NOTES: " + "; ".join(self.notes))
        text = "\n".join(head + [""] + body)
        return text[:max_chars]


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


def probe(path: Path) -> Dict[str, float]:
    """Duration and dimensions; {} when ffprobe cannot read the file."""
    try:
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            return {}
        data = json.loads(done.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        out: Dict[str, float] = {}
        if data.get("format", {}).get("duration"):
            out["duration_s"] = float(data["format"]["duration"])
        if stream.get("width"):
            out["width"] = float(stream["width"])
        if stream.get("height"):
            out["height"] = float(stream["height"])
        return out
    except Exception:  # noqa: BLE001
        return {}


def extract_frames(path: Path, *, out_dir: Path, max_frames: int = 8,
                   duration_s: float = 0.0, scene_change: bool = True) -> List[Frame]:
    """Sample frames: scene changes first (they carry the information), evenly spaced otherwise."""
    max_frames = max(1, min(int(max_frames), 24))
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: List[Frame] = []
    patterns: List[str] = []
    if scene_change:
        patterns.append("gt(scene\\,0.30)")
    if not patterns or duration_s > 0:
        fps = max_frames / duration_s if duration_s > 0 else 1.0 / 30.0
        patterns.append(f"fps={max(fps, 1e-6):.6f}")
    for pattern in patterns:
        for existing in out_dir.glob("f_*.png"):
            existing.unlink()
        for existing in out_dir.glob("f_*.jpg"):
            existing.unlink()
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-vf", pattern,
                 "-frames:v", str(max_frames), "-vsync", "vfr",
                 str(out_dir / "f_%03d.jpg")],
                check=True, capture_output=True, timeout=300)
        except Exception:  # noqa: BLE001 - a filter ffmpeg dislikes is not a fatal error
            continue
        files = sorted(out_dir.glob("f_*.jpg"))
        if files:
            step = duration_s / len(files) if duration_s else 0.0
            for index, file in enumerate(files):
                frames.append(Frame(index=index, timestamp_s=round(step * index, 2),
                                    path=str(file)))
            return frames
    return frames


def ocr_frame(frame: Frame, *, lang: str = "eng") -> Frame:
    """Read on-screen text locally; a frame with no text is normal, not a failure."""
    if not shutil.which("tesseract"):
        return frame
    try:
        done = subprocess.run(["tesseract", frame.path, "stdout", "-l", lang],
                              capture_output=True, text=True, timeout=120)
        if done.returncode == 0:
            text = " ".join(done.stdout.split())
            if len(text) >= 8:  # one or two characters is noise from a photo, not on-screen text
                frame.ocr_text = text
    except Exception:  # noqa: BLE001
        pass
    return frame


def extract_audio(path: Path, *, out_dir: Path) -> Tuple[Optional[str], str]:
    """16 kHz mono wav, the shape the STT provider already accepts."""
    if not ffmpeg_available():
        return None, "ffmpeg unavailable"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio.wav"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn",
                        "-ac", "1", "-ar", "16000", str(wav)],
                       check=True, capture_output=True, timeout=600)
        if wav.exists() and wav.stat().st_size > 1024:
            return str(wav), ""
        return None, "no audio track"
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()[-200:]
        return None, f"no audio track ({detail})" if detail else "no audio track"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)[:200]


def build_video_context(path: str, *, max_frames: int = 8, do_ocr: bool = True,
                        do_transcribe: bool = True, keep_frames: bool = False,
                        transcribe=None) -> VideoContext:
    """Frames + OCR + speech -> one structured representation. Never raises."""
    target = Path(path)
    context = VideoContext(path=str(target))
    if not ffmpeg_available():
        context.notes.append("ffmpeg/ffprobe not installed")
        return context

    info = probe(target)
    context.duration_s = float(info.get("duration_s", 0.0) or 0.0)
    context.width = int(info.get("width", 0) or 0)
    context.height = int(info.get("height", 0) or 0)
    if not info:
        context.notes.append("ffprobe could not read the file (is it a real video?)")

    tmp = Path(tempfile.mkdtemp(prefix="hermes-video-"))
    try:
        context.frames = extract_frames(target, out_dir=tmp / "frames", max_frames=max_frames,
                                        duration_s=context.duration_s)
        if do_ocr and context.frames:
            context.frames = [ocr_frame(frame) for frame in context.frames]
        if do_transcribe:
            wav, why = extract_audio(target, out_dir=tmp / "audio")
            if wav:
                if transcribe is None:
                    from agent.capability_delegation import _transcribe_file

                    transcribe = _transcribe_file
                ok, text, error = transcribe(Path(wav))
                if ok and text.strip():
                    context.transcript = text.strip()
                else:
                    context.transcript_error = error or "STT returned nothing"
            else:
                context.transcript_error = why
        if keep_frames:
            context.notes.append(f"frames kept at {tmp / 'frames'}")
            tmp = None  # type: ignore[assignment]  # caller asked to keep them
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return context


def frame_image_parts(context: VideoContext, limit: int = 4) -> List[Dict[str, object]]:
    """Image parts for a vision specialist - opt-in, because each frame costs tokens."""
    parts: List[Dict[str, object]] = []
    for frame in context.frames[:max(0, limit)]:
        try:
            import base64

            data = Path(frame.path).read_bytes()
            parts.append({
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()},
            })
        except Exception:  # noqa: BLE001
            continue
    return parts
