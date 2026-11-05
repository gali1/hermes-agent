"""Video, done with the tools that exist here: ffmpeg frames + local OCR + local STT.

The point is that no model needs native video input. A clip becomes a labelled, timestamped text
representation - what is on screen, what is said, and how long it runs - and that text is what a
specialist (or the primary) reasons over. Frames are only sent to a vision model when asked for,
because a frame costs tokens and OCR costs nothing.

Real-world clips are long, sometimes silent, and local STT is slower than real time, so every
stage here is bounded by a wall-clock budget (``time_budget_s``). When a stage cannot finish, it
stops and says what it actually covered instead of pretending the clip was read end to end.
Speaker labels are *not* produced: the local STT has no diarization, and guessing who spoke is
worse than saying it is unknown.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_TIME_BUDGET_S = 240.0
# How much audio the first STT call is allowed to cover: it measures throughput, so it should be
# a fraction of the clip, not the whole thing, and never more than this many seconds.
STT_PROBE_MAX_S = 20.0
# After the probe, audio is transcribed in chunks this long, checking the deadline between them,
# so one optimistic throughput estimate can only overrun by one chunk's worth of audio.
STT_CHUNK_S = 60.0

SPEAKER_LABEL_NOTE = ("SPEAKERS: unavailable (local STT has no diarization; "
                      "speech is unattributed - do not guess who spoke)")
Transcriber = Callable[[Path], Tuple[bool, str, str]]


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
    audio_duration_s: float = 0.0
    covered_s: float = 0.0
    stt_realtime_factor: float = 0.0  # measured wall seconds per audio second
    timings: Dict[str, float] = field(default_factory=dict)

    def as_text(self, *, max_chars: int = 6000) -> str:
        head = [f"VIDEO: {Path(self.path).name} ({self.duration_s:.1f}s"
                + (f", {self.width}x{self.height}" if self.width else "") + ")"]
        body: List[str] = []
        if self.transcript.strip():
            span = ""
            if self.audio_duration_s > 0:
                span = f", 0:00-{_fmt_mmss(self.covered_s)} of {_fmt_mmss(self.audio_duration_s)}"
                if self.covered_s >= self.audio_duration_s - 1.0:
                    span += " (full)"
            body.append(f"SPEECH (local STT{span}):\n" + self.transcript.strip())
            body.append(SPEAKER_LABEL_NOTE)
            if self.transcript_error:
                body.append(f"SPEECH: partial ({self.transcript_error})")
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
        if len(text) <= max_chars:
            return text
        transcript_chars = len(self.transcript.strip())
        total = len(text)
        marker = ""
        keep = max_chars
        for _ in range(4):  # the marker quotes the numbers, so its own length moves them
            marker = (f"\n[TRUNCATED: transcript {transcript_chars} chars; "
                      f"showing {max(0, keep)} of {total} chars]")
            settled = max(0, max_chars - len(marker))
            if settled == keep:
                break
            keep = settled
        if max_chars <= len(marker):  # the marker alone does not fit; honour the hard cap
            return text[:max_chars]
        keep = max(0, min(keep, max_chars - len(marker)))
        return text[:keep] + marker


def _fmt_mmss(seconds: float) -> str:
    """0:00 style timestamps, the way a coverage claim reads."""
    total = int(max(0.0, round(float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


def probe(path: Path, *, timeout_s: float = 60.0) -> Dict[str, float]:
    """Duration, dimensions and whether there is an audio stream; {} when unreadable."""
    if timeout_s <= 0:
        return {}
    try:
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,width,height,duration:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=max(1.0, timeout_s))
        if done.returncode != 0:
            return {}
        data = json.loads(done.stdout or "{}")
        streams = data.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        out: Dict[str, float] = {"has_audio": 1.0 if audio is not None else 0.0}
        if data.get("format", {}).get("duration"):
            out["duration_s"] = float(data["format"]["duration"])
        if video.get("width"):
            out["width"] = float(video["width"])
        if video.get("height"):
            out["height"] = float(video["height"])
        if audio and audio.get("duration"):
            out["audio_duration_s"] = float(audio["duration"])
        return out
    except Exception:  # noqa: BLE001
        return {}


def _showinfo_times(stderr: str) -> List[float]:
    return [float(value) for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", stderr or "")]


def _frame_pass(path: Path, pattern: str, out_dir: Path, max_frames: int,
                timeout_s: float) -> Tuple[List[Frame], bool]:
    """One ffmpeg sampling pass. Timestamps come from showinfo, not from an average step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob("f_*.jpg")):
        stale.unlink()
    timed_out = False
    try:
        done = subprocess.run(
            ["ffmpeg", "-v", "info", "-y", "-i", str(path), "-vf", pattern,
             "-frames:v", str(max_frames), "-vsync", "vfr",
             str(out_dir / "f_%03d.jpg")],
            capture_output=True, text=True, timeout=max(1.0, timeout_s))
    except subprocess.TimeoutExpired:
        return [], True
    except Exception:  # noqa: BLE001 - a filter ffmpeg dislikes is not a fatal error
        return [], timed_out
    stamps = _showinfo_times(done.stderr)
    files = sorted(out_dir.glob("f_*.jpg"))
    frames: List[Frame] = []
    for index, file in enumerate(files):
        stamp = stamps[index] if index < len(stamps) else 0.0
        frames.append(Frame(index=index, timestamp_s=round(stamp, 2), path=str(file)))
    return frames, timed_out


def extract_frames(path: Path, *, out_dir: Path, max_frames: int = 8,
                   duration_s: float = 0.0, scene_change: bool = True,
                   deadline: Optional[float] = None,
                   stats: Optional[Dict[str, Any]] = None) -> List[Frame]:
    """Sample frames across the whole clip, capped at ``max_frames`` and at ``deadline``.

    Scene cuts are sampled first because they carry the information; the rest of the slots are
    filled with evenly spaced frames so a clip of a static screen with changing text still gets
    coverage of its whole timeline. Timestamps are read back from ffmpeg (``showinfo``), so a
    scene-change frame is labelled with its real position rather than an averaged one.
    """
    max_frames = max(1, min(int(max_frames), 24))
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = stats if stats is not None else {}
    stats.setdefault("requested", max_frames)
    stats.setdefault("budget_hit", False)
    frames: List[Frame] = []
    budget = (lambda: _remaining(deadline)) if deadline is not None else (lambda: 3600.0)

    if scene_change and budget() > 3.0:
        frames, timed_out = _frame_pass(path, "select='gt(scene\\,0.30)',showinfo",
                                        out_dir / "scene", max_frames, budget())
        if timed_out:
            stats["budget_hit"] = True

    if len(frames) < max_frames:
        if budget() <= 1.0:
            stats["budget_hit"] = True
        else:
            fps = max_frames / duration_s if duration_s > 0 else 1.0 / 30.0
            fps = max(fps, 1e-6)
            even, timed_out = _frame_pass(path, f"fps={fps:.6f},showinfo", out_dir / "even",
                                          max_frames, budget())
            if timed_out:
                stats["budget_hit"] = True
            if even:
                gap = (duration_s / max_frames) if duration_s > 0 else 0.0
                chosen = list(frames)
                for candidate in even:
                    if len(chosen) >= max_frames:
                        break
                    if all(abs(candidate.timestamp_s - taken.timestamp_s) >= max(0.2, gap / 2.0)
                           for taken in chosen):
                        chosen.append(candidate)
                frames = chosen

    frames = sorted(frames, key=lambda item: item.timestamp_s)[:max_frames]
    for index, frame in enumerate(frames):
        frame.index = index
    stats["frames"] = len(frames)
    return frames


def ocr_frame(frame: Frame, *, lang: str = "eng", timeout_s: float = 120.0) -> Frame:
    """Read on-screen text locally; a frame with no text is normal, not a failure."""
    if not shutil.which("tesseract") or timeout_s <= 0:
        return frame
    try:
        done = subprocess.run(["tesseract", frame.path, "stdout", "-l", lang],
                              capture_output=True, text=True, timeout=max(1.0, timeout_s))
        if done.returncode == 0:
            text = " ".join(done.stdout.split())
            if len(text) >= 8:  # one or two characters is noise from a photo, not on-screen text
                frame.ocr_text = text
    except Exception:  # noqa: BLE001
        pass
    return frame


def extract_audio(path: Path, *, out_dir: Path, timeout_s: float = 600.0) -> Tuple[Optional[str], str]:
    """16 kHz mono wav, the shape the STT provider already accepts."""
    if not ffmpeg_available():
        return None, "ffmpeg unavailable"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio.wav"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn",
                        "-ac", "1", "-ar", "16000", str(wav)],
                       check=True, capture_output=True, timeout=max(2.0, timeout_s))
        if wav.exists() and wav.stat().st_size > 1024:
            return str(wav), ""
        return None, "no audio track"
    except subprocess.TimeoutExpired:
        return None, "time budget exhausted during audio extraction (budget)"
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()[-200:]
        return None, f"no audio track ({detail})" if detail else "no audio track"
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)[:200]


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getframerate():
                return handle.getnframes() / float(handle.getframerate())
    except Exception:  # noqa: BLE001 - fall back to ffprobe for odd containers
        pass
    try:
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return float((done.stdout or "0").strip() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _clip_wav(source: str, target: Path, start_s: float, length_s: float) -> Optional[Path]:
    """Cut ``[start, start+length)`` out of the extracted wav; None when ffmpeg refuses."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start_s):.3f}",
                        "-t", f"{max(0.1, length_s):.3f}", "-i", source,
                        "-ac", "1", "-ar", "16000", str(target)],
                       check=True, capture_output=True, timeout=max(
                           5.0, min(120.0, length_s * 2.0)))
    except Exception:  # noqa: BLE001
        return None
    return target if target.exists() and target.stat().st_size > 1024 else None


def _call_transcriber(transcribe: Transcriber, audio: Path) -> Tuple[float, bool, str, str]:
    started = time.monotonic()
    try:
        ok, text, error = transcribe(audio)
    except Exception as exc:  # noqa: BLE001 - a provider crash is reported, not raised
        return time.monotonic() - started, False, "", f"{type(exc).__name__}: {str(exc)[:200]}"
    return time.monotonic() - started, bool(ok), str(text or ""), str(error or "")


def _stt_with_budget(wav: str, duration_s: float, *, out_dir: Path, deadline: float,
                     transcribe: Transcriber) -> Tuple[str, float, str, float]:
    """Transcribe as much of the audio as the remaining budget pays for, from the start.

    A first, short clip measures actual throughput (wall seconds per audio second) on this
    machine and provider; that measurement - not a guessed constant - decides how much audio the
    rest of the budget can pay for. Audio is then cut with ffmpeg, chunk by chunk, and the
    deadline is re-checked before every call, so a too-optimistic estimate from the probe can
    only overrun by one chunk rather than by the whole remaining track. The covered span is
    returned so the caller can state exactly what was transcribed.
    """
    if duration_s <= 0.05:
        return "", 0.0, "audio track is empty", 0.0
    remaining = _remaining(deadline)
    if remaining <= 0.05:
        return "", 0.0, "time budget exhausted before speech-to-text (budget)", 0.0

    available = max(1.0, remaining * 0.5)  # never let the measuring call eat the whole budget
    probe_s = min(duration_s, max(1.0, min(STT_PROBE_MAX_S, duration_s / 4.0, available)))
    single_pass = probe_s >= duration_s - 1.0
    if single_pass:
        probe_s = duration_s

    part = _clip_wav(wav, out_dir / "stt_part0.wav", 0.0, probe_s)
    if part is None:  # conversion failed; hand the full wav over rather than nothing
        part, probe_s = Path(wav), duration_s
        single_pass = True
    elapsed, ok, text, error = _call_transcriber(transcribe, part)
    if not ok and not text.strip():
        return "", 0.0, error or "STT returned nothing", 0.0
    parts = [text.strip()] if text.strip() else []
    covered = probe_s
    wall = elapsed
    rtf = (elapsed / probe_s) if probe_s > 0 else 0.0
    worst_rate = rtf
    if single_pass:
        return "\n".join(parts).strip(), covered, "", rtf

    chunk_index = 1
    failure = ""
    while covered < duration_s - 0.5:
        remaining = _remaining(deadline)
        reserve = min(3.0, max(0.5, remaining * 0.05))  # notes and cleanup must still fit
        spendable = remaining - reserve
        if spendable <= 0.5:
            break
        chunk = min(STT_CHUNK_S, duration_s - covered)
        if worst_rate > 0 and worst_rate * chunk > spendable:
            chunk = spendable / worst_rate
        if chunk < 2.0:  # another model load would cost more than the audio is worth
            break
        clip = _clip_wav(wav, out_dir / f"stt_part{chunk_index}.wav", covered, chunk)
        if clip is None:
            break
        elapsed, ok, text, error = _call_transcriber(transcribe, clip)
        wall += elapsed
        if not ok and not text.strip():
            failure = f"STT failed after {_fmt_mmss(covered)} ({error})"
            break
        if text.strip():
            parts.append(text.strip())
        covered = min(duration_s, covered + chunk)
        if chunk > 0:
            worst_rate = max(worst_rate, elapsed / chunk)
        chunk_index += 1

    rtf = (wall / covered) if covered > 0 else 0.0
    return "\n".join(parts).strip(), covered, failure, rtf


def build_video_context(path: str, *, max_frames: int = 8, do_ocr: bool = True,
                        do_transcribe: bool = True, keep_frames: bool = False,
                        transcribe: Optional[Transcriber] = None,
                        time_budget_s: float = DEFAULT_TIME_BUDGET_S) -> VideoContext:
    """Frames + OCR + speech -> one structured representation, inside a wall-clock budget.

    Never raises. Every stage stops at ``time_budget_s`` and leaves a coverage note saying how
    much of the clip was actually read; nothing is invented for the part that was not.
    """
    started = time.monotonic()
    try:
        budget = max(0.0, float(time_budget_s))
    except (TypeError, ValueError):
        budget = DEFAULT_TIME_BUDGET_S
    # Keep a beat for assembling the notes/timings and cleaning up, so the call lands inside the
    # budget it was given rather than a second over it.
    deadline = started + max(0.0, budget - (1.0 if budget > 2.0 else 0.0))
    target = Path(path)
    context = VideoContext(path=str(target))
    if not ffmpeg_available():
        context.notes.append("ffmpeg/ffprobe not installed")
        context.timings["total_s"] = round(time.monotonic() - started, 2)
        return context

    stage = time.monotonic()
    info = probe(target, timeout_s=min(60.0, _remaining(deadline) or 1.0))
    context.timings["probe_s"] = round(time.monotonic() - stage, 2)
    context.duration_s = float(info.get("duration_s", 0.0) or 0.0)
    context.width = int(info.get("width", 0) or 0)
    context.height = int(info.get("height", 0) or 0)
    context.audio_duration_s = float(info.get("audio_duration_s", 0.0) or 0.0)
    if not info:
        context.notes.append("ffprobe could not read the file (is it a real video?)")

    tmp = Path(tempfile.mkdtemp(prefix="hermes-video-"))
    try:
        stage = time.monotonic()
        stats: Dict[str, object] = {}
        context.frames = extract_frames(target, out_dir=tmp / "frames", max_frames=max_frames,
                                        duration_s=context.duration_s, deadline=deadline,
                                        stats=stats)
        context.timings["frames_s"] = round(time.monotonic() - stage, 2)
        if stats.get("budget_hit"):
            requested = stats.get("requested", max_frames)
            context.notes.append(
                f"frames: {len(context.frames)} of {int(requested)} sampled (budget)")

        if do_ocr and context.frames:
            stage = time.monotonic()
            read = 0
            for frame in context.frames:
                if _remaining(deadline) <= 0.5:
                    break
                ocr_frame(frame, timeout_s=min(120.0, _remaining(deadline)))
                read += 1
            context.timings["ocr_s"] = round(time.monotonic() - stage, 2)
            if read < len(context.frames):
                context.notes.append(f"OCR: {read} of {len(context.frames)} frames read (budget)")

        if do_transcribe:
            _transcribe_with_budget(context, target, tmp, deadline, transcribe,
                                    has_audio=None if not info else bool(info.get("has_audio")))

        if keep_frames:
            context.notes.append(f"frames kept at {tmp / 'frames'}")
            tmp = None  # type: ignore[assignment]  # caller asked to keep them
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    context.timings["total_s"] = round(time.monotonic() - started, 2)
    return context


def _transcribe_with_budget(context: VideoContext, target: Path, tmp: Path, deadline: float,
                            transcribe: Optional[Transcriber],
                            has_audio: Optional[bool] = None) -> None:
    """Audio extraction + STT, with the coverage claim attached to the context."""
    if has_audio is False:
        # ffprobe already answered: do not let ffmpeg's error text stand in for the fact.
        context.transcript_error = "no audio track"
        context.notes.append("no audio stream in the container")
        return
    if _remaining(deadline) <= 0.5:
        context.transcript_error = "time budget exhausted before audio extraction (budget)"
        context.notes.append(f"audio transcribed 0:00-0:00 of {_fmt_mmss(context.duration_s)} (budget)")
        return
    if transcribe is None:
        from agent.capability_delegation import _transcribe_file

        transcribe = _transcribe_file

    stage = time.monotonic()
    wav, why = extract_audio(target, out_dir=tmp / "audio",
                             timeout_s=min(600.0, _remaining(deadline)))
    context.timings["audio_s"] = round(time.monotonic() - stage, 2)
    if not wav:
        context.transcript_error = why or "no audio track"
        if why.startswith("time budget"):
            context.notes.append(
                f"audio transcribed 0:00-0:00 of {_fmt_mmss(context.duration_s)} (budget)")
        return

    duration = _wav_duration(Path(wav))
    if duration <= 0:
        duration = context.duration_s
    context.audio_duration_s = duration or context.audio_duration_s

    stage = time.monotonic()
    text, covered, error, rtf = _stt_with_budget(
        wav, duration, out_dir=tmp / "audio", deadline=deadline, transcribe=transcribe)
    context.timings["stt_s"] = round(time.monotonic() - stage, 2)
    context.covered_s = round(covered, 2)
    context.stt_realtime_factor = round(rtf, 3)
    if text.strip():
        context.transcript = text.strip()
        if error:
            context.transcript_error = error  # partial speech still carries the failure reason
    else:
        context.transcript_error = error or "STT returned nothing"
    total = context.audio_duration_s or duration
    if total > 0:
        if covered >= total - 1.0:
            context.notes.append(
                f"audio transcribed 0:00-{_fmt_mmss(total)} of {_fmt_mmss(total)} (complete)")
        else:
            context.notes.append(
                f"audio transcribed 0:00-{_fmt_mmss(covered)} of {_fmt_mmss(total)} (budget)")


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
