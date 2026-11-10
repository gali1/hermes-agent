"""Video, done with the tools that exist here: ffmpeg frames + local OCR + local STT.

The point is that no model needs native video input. A clip becomes a labelled, timestamped text
representation - what is on screen, what is said, and how long it runs - and that text is what a
specialist (or the primary) reasons over. Frames are only sent to a vision model when asked for,
because a frame costs tokens and OCR costs nothing.

Real-world clips are long, sometimes silent, and local STT is slower than real time, so every
stage here is bounded by a wall-clock budget (``time_budget_s``) and speech-to-text by its own
(``stt_budget_s``, which defaults to the same value). When a stage cannot finish, it stops and
says what it actually covered instead of pretending the clip was read end to end.
Speaker labels are *not* produced: the local STT has no diarization, and guessing who spoke is
worse than saying it is unknown.

Three things make a five-minute clip fit in a few minutes of wall clock instead of never
finishing: long silences are cut out before any audio reaches the model, the remaining speech is
split into short chunks that are decoded in parallel across cores (and stitched back in time
order), and OCR runs alongside STT rather than after it.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_TIME_BUDGET_S = 240.0
# How much audio a single small-budget STT call may take: when there is not enough time left for
# a full parallel wave, one short clip measures throughput so the coverage claim stays truthful.
STT_PROBE_MAX_S = 20.0
# Speech is decoded in chunks this long. Short chunks are what make parallel decoding pay: with
# four workers a 5-minute clip becomes a handful of 30 s jobs instead of one long queue.
STT_CHUNK_S = 30.0
# Runs of silence at least this long are cut out before decoding. Whisper invents text on
# silence, and VAD inside the model still costs a forward pass per frame, so the cheapest
# silence is silence that never reaches the model.
STT_MIN_SILENCE_S = 0.8
STT_SILENCE_DB = -35.0
# Default chunk workers: one per core, capped, because past that ctranslate2 spends its time
# contending (measured: 4 workers 0.81 aggregate RTF vs 8 workers 0.73 on an 8-thread box under
# load). ``stt.video.workers`` overrides it.
STT_WORKERS_DEFAULT = max(1, min(4, os.cpu_count() or 1))
STT_BEAM_SIZE_DEFAULT = 1
STT_COMPUTE_TYPE_DEFAULT = "int8"
DEFAULT_STT_MODEL = "base"
STT_MODEL_ENV = "HERMES_VIDEO_STT_MODEL"
# A chunk needs about this much budget to be worth starting: below it, one clipped probe is all
# the time there is.
STT_MIN_CHUNK_BUDGET_S = 6.0

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
    # Provenance of the speech pass, so a reader (and the measurement script) can tell what ran.
    silence_skipped_s: float = 0.0
    stt_chunks: int = 0
    stt_workers: int = 1
    stt_workers_requested: int = 0  # what the configuration asked for, before the load guard
    host_load_start: float = 0.0    # load1 when the pass began, so "partial" names its cause
    stt_model: str = ""
    stt_backend: str = ""  # "faster-whisper" (in-process) or "provider" (configured STT stack)
    time_budget_s: float = 0.0
    stt_budget_s: float = 0.0

    def as_text(self, *, max_chars: int = 6000) -> str:
        head = [f"VIDEO: {Path(self.path).name} ({self.duration_s:.1f}s"
                + (f", {self.width}x{self.height}" if self.width else "") + ")"]
        body: List[str] = []
        if self.transcript.strip():
            span = ""
            if self.audio_duration_s > 0:
                span = f", 0:00-{_fmt_mmss(self.covered_s)} of {_fmt_mmss(self.audio_duration_s)}"
                if self.covered_s >= self.audio_duration_s - 1.0:
                    span += " (complete)"
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


def _video_stt_config() -> Dict[str, Any]:
    """``stt.video.*`` from the Hermes config; {} when unavailable (tests, CLI tools).

    Only the video path reads this, so a five-minute clip can be tuned (smaller model, more
    workers, a longer speech budget) without changing what voice-message transcription does.
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:  # noqa: BLE001 - no config is a normal state for a library call
        return {}
    stt = config.get("stt") if isinstance(config, dict) else None
    if not isinstance(stt, dict):
        return {}
    video = stt.get("video")
    out: Dict[str, Any] = dict(video) if isinstance(video, dict) else {}
    local: Dict[str, Any] = stt["local"] if isinstance(stt.get("local"), dict) else {}
    if not out.get("language"):
        out["language"] = local.get("language") or stt.get("language") or ""
    return out


_STT_WORKER_SOURCE = '''"""Chunk decoder for agent/video_context.py - faster-whisper over slices of one wav.

Written at run time into the video temp directory and executed by an interpreter that has
faster-whisper installed. It exists because the video path needs three things the general STT
provider does not offer: ONE model load per worker instead of one per chunk, the decoding
settings this ctranslate2 build accepts (int8, beam_size=1, Silero VAD), and one JSON line per
finished chunk, so a worker killed at the deadline still reports what it really decoded.

argv: <audio wav> <plan json> <out jsonl> <config json>
plan:  [[index, start_s, end_s], ...]   config: {model, device, compute_type, cpu_threads,
         beam_size, language}
"""
import json
import sys
import time
import wave

import numpy as np
from faster_whisper import WhisperModel

audio_path, plan_path, out_path, config_path = sys.argv[1:5]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
with open(plan_path, encoding="utf-8") as handle:
    plan = json.load(handle)

model = WhisperModel(config["model"], device=config.get("device") or "cpu",
                     compute_type=config.get("compute_type") or "int8",
                     cpu_threads=int(config.get("cpu_threads") or 0))
with wave.open(audio_path, "rb") as handle:
    rate = handle.getframerate()
    channels = handle.getnchannels()
    raw = handle.readframes(handle.getnframes())
audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
if channels > 1:
    audio = audio.reshape(-1, channels).mean(axis=1)

kwargs = {"beam_size": int(config.get("beam_size") or 1), "vad_filter": True,
          "condition_on_previous_text": False,
          "no_speech_threshold": 0.6, "log_prob_threshold": -1.0,
          "vad_parameters": {"min_silence_duration_ms": 500}}
if config.get("language"):
    kwargs["language"] = config["language"]

with open(out_path, "a", encoding="utf-8") as out:
    for index, start, end in plan:
        clip = audio[int(float(start) * rate):int(float(end) * rate)]
        began = time.monotonic()
        try:
            segments, _info = model.transcribe(clip, **kwargs)
            # Same AND-gate the local STT stack uses (openai-whisper's own rule): non-speech
            # *and* low confidence, so quiet real speech survives and silence noise does not.
            text = " ".join(
                segment.text.strip() for segment in segments
                if not (float(getattr(segment, "no_speech_prob", 0.0)) > 0.6
                        and float(getattr(segment, "avg_logprob", 0.0)) < -1.0)
            ).strip()
            record = {"i": int(index), "ok": True, "text": text,
                      "wall": round(time.monotonic() - began, 3)}
        except Exception as exc:
            record = {"i": int(index), "ok": False, "text": "",
                      "error": ("%s: %s" % (type(exc).__name__, exc))[:200],
                      "wall": round(time.monotonic() - began, 3)}
        out.write(json.dumps(record) + "\\n")
        out.flush()
'''


def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


_PYTHON_PROBE_CACHE: Dict[str, bool] = {}


def _interpreter_has_faster_whisper(python: str) -> bool:
    """Does this interpreter import faster-whisper? Probed once per path, cached."""
    if not python or not Path(python).exists():
        return False
    if python in _PYTHON_PROBE_CACHE:
        return _PYTHON_PROBE_CACHE[python]
    ok = False
    try:
        done = subprocess.run([python, "-c", "import faster_whisper"], timeout=60,
                              capture_output=True)
        ok = done.returncode == 0
    except Exception:  # noqa: BLE001
        ok = False
    _PYTHON_PROBE_CACHE[python] = ok
    return ok


def _stt_worker_python(config: Dict[str, Any]) -> Optional[str]:
    """The interpreter that can run the chunk worker: env, then us, then the STT command's.

    The configured STT provider is a command template (``HERMES_LOCAL_STT_COMMAND``), and its
    first word is usually the interpreter that already has faster-whisper installed - the same
    one that answers voice messages. Reusing it means the video path gets the model settings it
    needs without touching that provider.
    """
    candidates: List[str] = []
    explicit = os.getenv("HERMES_VIDEO_STT_PYTHON") or config.get("python")
    if isinstance(explicit, str) and explicit.strip():
        candidates.append(explicit.strip())
    if _faster_whisper_available():
        candidates.append(sys.executable)  # this very process can run the worker
    template = (os.getenv("HERMES_LOCAL_STT_COMMAND") or "").strip()
    if template:
        try:
            first = shlex.split(template.format(input_path="x", output_dir="y", model="base",
                                                language="en"))[0]
        except (ValueError, KeyError, IndexError):
            first = ""
        if first and ("python" in Path(first).name.lower()):
            candidates.append(first)
    candidates.append(sys.executable)
    for candidate in candidates:
        if _interpreter_has_faster_whisper(candidate):
            return candidate
    return None


def _partition_chunks(count: int, workers: int) -> List[List[int]]:
    """Contiguous blocks of chunk indexes, one per worker: coverage then grows in time order."""
    workers = max(1, min(workers, count))
    base, extra = divmod(count, workers)
    blocks: List[List[int]] = []
    start = 0
    for worker in range(workers):
        size = base + (1 if worker < extra else 0)
        blocks.append(list(range(start, start + size)))
        start += size
    return [block for block in blocks if block]


def _parallel_worker_decode(wav: str, chunks: Sequence[Tuple[float, float]], *, out_dir: Path,
                            deadline: float, python: str, model: str, workers: int,
                            beam_size: int, compute_type: str, device: str,
                            cpu_threads: int, language: Optional[str]
                            ) -> Tuple[List[Optional[Tuple[float, bool, str, str]]], float, str]:
    """Decode chunks in ``workers`` processes, each with one model load, killed at the deadline.

    Each worker writes a JSON line per finished chunk as soon as it is done, so whatever was
    decoded before the deadline survives even when the process is terminated.
    """
    results: List[Optional[Tuple[float, bool, str, str]]] = [None] * len(chunks)
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_script = out_dir / "stt_worker.py"
    worker_script.write_text(_STT_WORKER_SOURCE, encoding="utf-8")
    config_path = out_dir / "stt_worker_config.json"
    config_path.write_text(json.dumps({
        "model": model, "device": device or "cpu", "compute_type": compute_type or "int8",
        "cpu_threads": max(1, int(cpu_threads)), "beam_size": max(1, int(beam_size)),
        "language": language or ""}), encoding="utf-8")

    started = time.monotonic()
    processes: List[Tuple[Any, Path, int, Path]] = []
    for worker, block in enumerate(_partition_chunks(len(chunks), workers)):
        plan_path = out_dir / f"stt_plan_{worker}.json"
        plan_path.write_text(json.dumps(
            [[position, round(chunks[position][0], 3), round(chunks[position][1], 3)]
             for position in block]), encoding="utf-8")
        result_path = out_dir / f"stt_result_{worker}.jsonl"
        result_path.write_text("", encoding="utf-8")
        log_path = out_dir / f"stt_worker_{worker}.err"
        with open(log_path, "wb") as log_handle:
            processes.append((subprocess.Popen(
                [python, str(worker_script), wav, str(plan_path), str(result_path),
                 str(config_path)],
                stdout=subprocess.DEVNULL, stderr=log_handle,
                preexec_fn=lambda: os.nice(10)), result_path, len(block),
                log_path))

    stopped_at_deadline = False
    try:
        while True:
            alive = [item for item in processes if item[0].poll() is None]
            complete = all(
                sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
                >= expected for _proc, path, expected, _log in processes)
            if complete or not alive or _remaining(deadline) <= 0.4:
                break
            time.sleep(0.2)
    finally:
        for proc, _path, _expected, _log in processes:
            if proc.poll() is None:
                # Remember that the stop was ours, not the worker's: the diagnosis below must not
                # report our own SIGTERM as a decoder failure (lane1: starved worker, not a crash).
                stopped_at_deadline = True
                proc.terminate()
        for proc, _path, _expected, _log in processes:
            try:
                proc.wait(timeout=3.0)
            except Exception:  # noqa: BLE001
                proc.kill()

    error = ""
    wall_total = 0.0
    lines_read = 0
    for _proc, path, _expected, _log in processes:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                position = int(record["i"])
            except (ValueError, KeyError, TypeError):
                continue
            if not 0 <= position < len(results):
                continue
            lines_read += 1
            wall = float(record.get("wall") or 0.0)
            results[position] = (wall, bool(record.get("ok")),
                                 str(record.get("text") or ""), str(record.get("error") or ""))
            wall_total += wall
    missing = [position for position, result in enumerate(results) if result is None]
    if missing:
        error = "time budget reached before a chunk finished (budget)"
    if lines_read == 0:
        # No chunk at all came back: the worker died before decoding (model load, OOM, ...).
        detail = ""
        for _proc, _path, _expected, log_path in processes:
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:  # noqa: BLE001
                tail = ""
            if tail:
                detail = tail.splitlines()[-1][:200]
                break
        # A worker that dies before its first line is the one failure nobody can diagnose later:
        # the temp log is gone by the time anyone looks. Record how it died, now, in the answer.
        exits = []
        for proc, _path, _expected, _log in processes:
            code = proc.poll()
            if code is None:
                exits.append("still running")
            elif code < 0:
                exits.append(f"killed by signal {-code}")
            else:
                exits.append(f"exit {code}")
        if stopped_at_deadline:
            # A starved decoder we stopped at the deadline is a budget stop, not a dead decoder.
            # Saying otherwise sends the reader looking for a crash that never happened.
            error = ("time budget reached before a chunk finished (budget; the decoder was still "
                     "running and was stopped at the deadline)")
        else:
            error = (f"faster-whisper worker produced no output "
                     f"({detail or 'no stderr'}; {'; '.join(sorted(set(exits))) or 'status unknown'})")
    elif all(result is not None and not result[1] and not result[2].strip()
             for result in results):
        first = results[0]
        error = f"faster-whisper worker failed ({first[3] if first else 'no output'})"
    if wall_total <= 0:
        wall_total = time.monotonic() - started
    return results, wall_total, error


def _resolve_stt_model(explicit: Optional[str], config: Dict[str, Any]) -> str:
    for candidate in (explicit, os.getenv(STT_MODEL_ENV), config.get("model")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return DEFAULT_STT_MODEL


def _available_cpu_slots() -> int:
    """Cores that are actually free right now, not cores that exist.

    Measured: with the box at load 16 and four STT workers, a 5-minute clip covered only 60 s of
    audio in 239 s (RTF 10.2) and a one-word live turn took 19 s. STT must yield to live traffic,
    so the worker count follows the load average instead of the core count.
    """
    cores = os.cpu_count() or 2
    try:
        load1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        return max(1, cores // 2)
    free = cores - int(round(load1))
    return max(1, min(cores, free))


def _requested_workers(explicit: Optional[int], config: Dict[str, Any]) -> int:
    """What was asked for, before the load guard trimmed it — only used to explain a shortfall."""
    for candidate in (explicit, config.get("workers")):
        try:
            if candidate is not None:
                return max(1, min(16, int(candidate)))
        except (TypeError, ValueError):
            continue
    return max(1, (os.cpu_count() or 2) // 2)


def _resolve_stt_workers(explicit: Optional[int], config: Dict[str, Any]) -> int:
    """Workers: explicit, then ``stt.video.workers``, then half the free cores (never more)."""
    requested = None
    for candidate in (explicit, config.get("workers")):
        try:
            if candidate is not None:
                requested = max(1, min(16, int(candidate)))
                break
        except (TypeError, ValueError):
            continue
    if requested is not None:
        # An operator who names a worker count gets it: the load guard is for the default path,
        # where nobody chose anything and starving live traffic would be our decision, not theirs.
        return requested
    return max(1, min(max(1, (os.cpu_count() or 2) // 2), _available_cpu_slots()))


def _resolve_stt_budget(explicit: Optional[float], config: Dict[str, Any],
                        time_budget_s: float) -> float:
    """Speech budget: caller, then ``stt.video.budget_s``, then the overall budget."""
    for candidate in (explicit, config.get("budget_s")):
        try:
            if candidate is not None:
                return max(0.0, float(candidate))
        except (TypeError, ValueError):
            continue
    return time_budget_s


def _resolve_stt_engine(*, transcribe: Optional[Transcriber], config: Dict[str, Any]
                        ) -> Tuple[str, Optional[str], Optional[Transcriber]]:
    """Which decoder to use: ("supplied"|"faster-whisper"|"provider", python, callable).

    A caller-supplied transcriber always wins, so the configured STT stack stays in charge when
    the caller named it. Otherwise the chunk worker is preferred when an interpreter can run it
    (the video path needs int8/beam=1/one model load), and the configured provider is the
    fallback - still parallelised per chunk, just without control of its decode settings.
    """
    if transcribe is not None:
        return "supplied", None, transcribe
    backend = str(config.get("backend") or "auto").strip().lower()
    if backend not in ("provider", "command", "local_command"):
        python = _stt_worker_python(config)
        if python:
            return "faster-whisper", python, None
    from agent.capability_delegation import _transcribe_file

    return "provider", None, _transcribe_file


def _speech_spans(wav: str, duration_s: float, *, timeout_s: float,
                  noise_db: float = STT_SILENCE_DB,
                  min_silence_s: float = STT_MIN_SILENCE_S) -> Optional[List[Tuple[float, float]]]:
    """Spans that contain something other than silence; None when ffmpeg cannot tell us.

    ``silencedetect`` writes its events to stderr; inverting them gives the speech spans, which
    is what gets decoded. Returning None (rather than guessing) keeps the fallback honest: with
    no answer, the whole track is treated as speech and nothing is skipped or claimed skipped.
    """
    if timeout_s <= 1.0 or duration_s <= 0:
        return None
    try:
        done = subprocess.run(
            ["ffmpeg", "-v", "info", "-nostdin", "-i", wav, "-af",
             f"silencedetect=noise={noise_db:g}dB:duration={min_silence_s:g}",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=max(2.0, timeout_s))
    except Exception:  # noqa: BLE001
        return None
    stderr = done.stderr or ""
    events = re.findall(r"silence_(start|end):\s*(-?[0-9]+(?:\.[0-9]+)?)", stderr)
    if not events and "silencedetect" not in stderr:
        return None  # ffmpeg refused the filter: no basis for skipping anything
    spans: List[Tuple[float, float]] = []
    cursor = 0.0
    for kind, value in events:
        stamp = max(0.0, min(duration_s, float(value)))
        if kind == "start":
            if stamp - cursor > 0.05:
                spans.append((cursor, stamp))
            cursor = max(cursor, stamp)
        else:
            cursor = max(cursor, stamp)
    if duration_s - cursor > 0.05:
        spans.append((cursor, duration_s))
    padded: List[Tuple[float, float]] = []
    for start, end in spans:
        start, end = max(0.0, start - 0.12), min(duration_s, end + 0.12)
        if padded and start - padded[-1][1] <= 0.1:
            padded[-1] = (padded[-1][0], max(end, padded[-1][1]))
        else:
            padded.append((start, end))
    return padded


def _plan_chunks(spans: Sequence[Tuple[float, float]], *, chunk_s: float,
                 join_gap_s: float = 1.5) -> List[Tuple[float, float]]:
    """Cut speech spans into decode jobs of at most ``chunk_s`` seconds, in time order."""
    pieces: List[Tuple[float, float]] = []
    for start, end in spans:
        position = start
        while end - position > 0.2:
            take = min(max(0.5, chunk_s), end - position)
            pieces.append((position, position + take))
            position += take
    chunks: List[Tuple[float, float]] = []
    for start, end in pieces:
        if (chunks and start - chunks[-1][1] <= join_gap_s
                and end - chunks[-1][0] <= max(0.5, chunk_s)):
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return chunks


@dataclass
class SttOutcome:
    """What the speech pass actually produced, and exactly which seconds it read."""

    text: str = ""
    covered_s: float = 0.0
    error: str = ""
    rtf: float = 0.0  # wall seconds per covered audio second (see _apply_stt_outcome)
    transcribed_s: float = 0.0  # audio seconds actually decoded by the workers
    silence_skipped_s: float = 0.0
    audio_duration_s: float = 0.0
    chunks: int = 0
    wall_s: float = 0.0


def _transcribe_chunks(wav: str, chunks: Sequence[Tuple[float, float]], *, out_dir: Path,
                       deadline: float, transcribe: Transcriber,
                       workers: int) -> Tuple[List[Optional[Tuple[float, bool, str, str]]],
                                              float, str]:
    """Decode chunks in parallel waves, in time order, stopping at ``deadline``.

    A wave is at most ``workers`` chunks; each runs on its own daemon thread so the deadline can
    be honoured by joining with a timeout instead of waiting for the slowest job. Chunks that do
    not finish are left unset - the caller reports the coverage it really has rather than
    pretending the audio in flight was read.
    """
    results: List[Optional[Tuple[float, bool, str, str]]] = [None] * len(chunks)
    wall_total = 0.0
    per_chunk_wall = 0.0
    failure = ""
    index = 0

    def _run(position: int) -> None:
        start, end = chunks[position]
        clip = _clip_wav(wav, out_dir / f"stt_c{position:03d}.wav", start, end - start)
        if clip is None:
            results[position] = (0.0, False, "", "could not cut chunk")
            return
        results[position] = _call_transcriber(transcribe, clip)

    while index < len(chunks):
        remaining = _remaining(deadline)
        if remaining <= 0.5:
            break
        # First wave is a single chunk: it measures this machine's real throughput before any
        # parallelism is assumed, so a slow host loses one chunk's worth of budget, not all of it.
        size = 1 if per_chunk_wall <= 0 else max(
            1, min(workers, int(remaining * 0.85 / per_chunk_wall)))
        batch = list(range(index, min(len(chunks), index + size)))
        started = time.monotonic()
        threads = [threading.Thread(target=_run, args=(position,), daemon=True) for position in batch]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=max(0.0, _remaining(deadline)))
        spent = time.monotonic() - started
        done = [position for position in batch if results[position] is not None]
        if not done:
            failure = "time budget reached before a chunk finished (budget)"
            break
        per_chunk_wall = spent
        wall_total += sum(results[position][0] for position in done)  # type: ignore[index]
        index = batch[-1] + 1
        if len(done) < len(batch):
            failure = "time budget reached mid-chunk (budget)"
            break
        failed = [p for p in done
                  if not results[p][1] and not results[p][2].strip()]  # type: ignore[index]
        if failed:
            detail = results[failed[0]][3]  # type: ignore[index]
            failure = f"STT failed at {_fmt_mmss(chunks[failed[0]][0])} ({detail})"
            break
    return results, wall_total, failure


def _stt_job(target: Path, tmp: Path, *, deadline: float,
             transcribe: Optional[Transcriber], worker_python: Optional[str],
             duration_s: float, workers: int, chunk_s: float, silence_db: float,
             min_silence_s: float, model: str, beam_size: int, compute_type: str,
             device: str, cpu_threads: int, language: Optional[str],
             timings: Dict[str, float]) -> SttOutcome:
    """Audio extraction, silence cutting, parallel decoding and the coverage arithmetic."""
    outcome = SttOutcome(audio_duration_s=duration_s)
    started = time.monotonic()
    if _remaining(deadline) <= 0.5:
        outcome.error = "time budget exhausted before audio extraction (budget)"
        outcome.wall_s = time.monotonic() - started
        return outcome

    stage = time.monotonic()
    wav, why = extract_audio(target, out_dir=tmp / "audio",
                             timeout_s=min(600.0, _remaining(deadline)))
    timings["audio_s"] = round(time.monotonic() - stage, 2)
    if not wav:
        outcome.error = why or "no audio track"
        outcome.wall_s = time.monotonic() - started
        return outcome

    duration = _wav_duration(Path(wav)) or duration_s
    outcome.audio_duration_s = duration
    if duration <= 0.05:
        outcome.error = "audio track is empty"
        outcome.wall_s = time.monotonic() - started
        return outcome

    stage = time.monotonic()
    spans = _speech_spans(wav, duration, timeout_s=min(30.0, max(2.0, _remaining(deadline) * 0.25)),
                          noise_db=silence_db, min_silence_s=min_silence_s)
    timings["silence_s"] = round(time.monotonic() - stage, 2)
    if spans is None:
        chunks = _plan_chunks([(0.0, duration)], chunk_s=chunk_s)
    elif not spans:
        outcome.covered_s = duration
        outcome.silence_skipped_s = duration
        outcome.error = "no speech detected (silence detector found only silence)"
        outcome.wall_s = time.monotonic() - started
        return outcome
    else:
        chunks = _plan_chunks(spans, chunk_s=chunk_s)
    outcome.chunks = len(chunks)
    if not chunks:
        outcome.error = "no speech detected (silence detector found only silence)"
        outcome.wall_s = time.monotonic() - started
        return outcome

    # With too little time left for a parallel wave, plan one short probe chunk instead of a
    # queue: the coverage arithmetic then describes exactly that probe.
    remaining = _remaining(deadline)
    full_plan = True
    if remaining < STT_MIN_CHUNK_BUDGET_S:
        start = chunks[0][0] if chunks[0][0] > 1.0 else 0.0
        probe_s = min(duration - start,
                      max(1.0, min(STT_PROBE_MAX_S, duration / 4.0, remaining * 0.5)))
        chunks = [(start, start + probe_s)]
        full_plan = False
        outcome.chunks = 1
    else:
        outcome.chunks = len(chunks)

    if worker_python:
        results, wall_total, failure = _parallel_worker_decode(
            wav, chunks, out_dir=tmp / "audio", deadline=deadline, python=worker_python,
            model=model, workers=workers, beam_size=beam_size, compute_type=compute_type,
            device=device, cpu_threads=cpu_threads, language=language)
        timings["stt_workers"] = len(_partition_chunks(len(chunks), workers))
    else:
        results, wall_total, failure = _transcribe_chunks(
            wav, chunks, out_dir=tmp / "audio", deadline=deadline,
            transcribe=transcribe, workers=workers)

    parts: List[str] = []
    covered_end = 0.0
    done = 0
    chunk_failure = ""
    for position, (start, end) in enumerate(chunks):
        result = results[position]
        if result is None:
            break
        _elapsed, ok, text, detail = result
        if text.strip():
            parts.append(text.strip())
        covered_end = end
        done += 1
        if not ok:
            chunk_failure = f"STT failed at {_fmt_mmss(start)} ({detail or 'no text'})"
            break  # partial text is kept, but nothing after a failed chunk is claimed
    transcribed_s = sum(chunks[i][1] - chunks[i][0] for i in range(done))
    outcome.text = "\n".join(parts).strip()
    complete = done == len(chunks) and full_plan
    outcome.covered_s = duration if complete else covered_end
    outcome.silence_skipped_s = max(0.0, outcome.covered_s - transcribed_s)
    # Elapsed wall seconds per second of audio ACTUALLY COVERED. Two wrong denominators came before
    # this one: transcribed_s (decoded chunk seconds, which ignores parallelism and truncation) and
    # wall_total in the parallel path, which is the SUM of per-chunk decode times, not the clock. With
    # one worker a 5:01 clip reported 2.59 for a pass that really took 238.89 s over 60 s of audio —
    # the stage clock is the only numerator that means what the name says.
    stage_wall = time.monotonic() - started
    outcome.rtf = (stage_wall / outcome.covered_s) if outcome.covered_s > 0.05 else 0.0
    outcome.transcribed_s = transcribed_s
    outcome.wall_s = time.monotonic() - started
    if done < len(chunks):
        outcome.error = chunk_failure or failure or (
            f"time budget reached after {_fmt_mmss(covered_end)} (budget)")
    return outcome


def _apply_stt_outcome(context: VideoContext, outcome: SttOutcome) -> None:
    """Attach the transcript and the coverage line that says what was really read."""
    if outcome.audio_duration_s:
        context.audio_duration_s = outcome.audio_duration_s
    context.covered_s = round(outcome.covered_s, 2)
    context.stt_realtime_factor = round(outcome.rtf, 3)
    context.timings["covered_s"] = round(outcome.covered_s, 2)
    context.timings["decoded_audio_s"] = round(outcome.transcribed_s, 2)
    context.silence_skipped_s = round(outcome.silence_skipped_s, 2)
    context.stt_chunks = outcome.chunks
    if outcome.text.strip():
        context.transcript = outcome.text.strip()
        if outcome.error:
            context.transcript_error = outcome.error  # partial speech keeps the reason
    else:
        context.transcript_error = outcome.error or "STT returned nothing"
    total = context.audio_duration_s or context.duration_s
    skipped = context.silence_skipped_s
    if "no speech detected" in (outcome.error or ""):
        context.notes.append(
            f"audio transcribed 0:00-{_fmt_mmss(outcome.covered_s)} of {_fmt_mmss(total)} "
            f"(no speech detected; {_fmt_mmss(skipped)} of silence skipped)")
        return
    if total <= 0:
        return
    if outcome.covered_s >= total - 1.0:
        note = f"audio transcribed 0:00-{_fmt_mmss(total)} of {_fmt_mmss(total)} (complete"
        note += f"; {skipped:.0f}s of silence skipped)" if skipped >= 0.5 else ")"
        context.notes.append(note)
    else:
        # The parenthetical must not lie about why the span stopped: a failed decode is not a
        # budget stop, and saying so is the difference between "read less" and "broke".
        error_text = (outcome.error or "").strip()
        if not error_text or "budget" in error_text.lower():
            reason = "budget"
        elif "worker" in error_text.lower():
            # A dead decoder is not a budget stop, and "partial" alone hides that: say it.
            reason = "decoder failed: " + error_text[:120]
        else:
            reason = "partial: " + error_text[:120]
        if reason == "budget":
            # "partial" without a cause reads as a defect. Name it: a loaded host is the usual one.
            if context.stt_workers_requested and context.stt_workers < context.stt_workers_requested:
                reason += (f"; speech ran on {context.stt_workers} of "
                           f"{context.stt_workers_requested} workers because the host was loaded "
                           f"(load {context.host_load_start:.1f} on {os.cpu_count()} cores)")
            elif context.host_load_start >= 1.0:
                reason += (f"; the {context.time_budget_s:.0f}s wall budget ran out on a loaded host "
                           f"(load {context.host_load_start:.1f} on {os.cpu_count()} cores)")
        context.notes.append(
            f"audio transcribed 0:00-{_fmt_mmss(outcome.covered_s)} of {_fmt_mmss(total)} "
            f"({reason})")


def build_video_context(path: str, *, max_frames: int = 8, do_ocr: bool = True,
                        do_transcribe: bool = True, keep_frames: bool = False,
                        transcribe: Optional[Transcriber] = None,
                        time_budget_s: float = DEFAULT_TIME_BUDGET_S,
                        stt_budget_s: Optional[float] = None,
                        stt_model: Optional[str] = None,
                        stt_workers: Optional[int] = None,
                        stt_chunk_s: Optional[float] = None) -> VideoContext:
    """Frames + OCR + speech -> one structured representation, inside a wall-clock budget.

    Never raises. Frames and OCR stop at ``time_budget_s``; speech-to-text stops at
    ``stt_budget_s`` (default: the same value, so existing callers see no change). The two run
    concurrently, so the wall time is roughly the slower stage, not their sum. A coverage note
    always says how much of the audio was actually read; nothing is invented for the part that
    was not.

    ``stt_model`` / ``stt_workers`` / ``stt_chunk_s`` default to ``stt.video.*`` in the config
    (then to base / cpu_count-capped workers / 30 s). A transcriber passed in by the caller is
    used as-is, so the configured STT stack stays in charge when one is supplied.

    Optional ``stt.video`` config keys: ``model``, ``workers``, ``chunk_s``, ``budget_s``,
    ``beam_size``, ``compute_type``, ``device``, ``cpu_threads``, ``language``, ``python``,
    ``min_silence_s``, ``silence_db``, and ``backend`` (``auto`` uses the chunk worker,
    ``provider`` forces the configured STT stack).
    """
    started = time.monotonic()
    try:
        budget = max(0.0, float(time_budget_s))
    except (TypeError, ValueError):
        budget = DEFAULT_TIME_BUDGET_S
    config = _video_stt_config()
    speech_budget = _resolve_stt_budget(stt_budget_s, config, budget)
    workers = _resolve_stt_workers(stt_workers, config)
    model = _resolve_stt_model(stt_model, config)
    try:
        chunk_s = float(stt_chunk_s if stt_chunk_s is not None else config.get("chunk_s")
                        or STT_CHUNK_S)
    except (TypeError, ValueError):
        chunk_s = STT_CHUNK_S
    chunk_s = min(120.0, max(2.0, chunk_s))

    # Keep a beat for assembling the notes/timings and cleaning up, so the call lands inside the
    # budget it was given rather than a second over it.
    deadline = started + max(0.0, budget - (1.0 if budget > 2.0 else 0.0))
    stt_deadline = started + max(0.0, speech_budget - (1.0 if speech_budget > 2.0 else 0.0))
    target = Path(path)
    try:
        _load1 = float(os.getloadavg()[0])
    except (OSError, AttributeError):
        _load1 = 0.0
    context = VideoContext(path=str(target), time_budget_s=round(budget, 2),
                           stt_budget_s=round(speech_budget, 2), stt_workers=workers,
                           stt_model=model, host_load_start=round(_load1, 2),
                           stt_workers_requested=_requested_workers(stt_workers, config))
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

        no_audio = "has_audio" in info and not info["has_audio"]
        if do_transcribe and no_audio:
            # ffprobe already answered: do not let ffmpeg's error text stand in for the fact.
            context.transcript_error = "no audio track"
            context.notes.append("no audio stream in the container")
            context.notes.append(
                f"audio transcribed 0:00-0:00 of {_fmt_mmss(context.duration_s)} "
                "(no audio track)")
        speech = _start_speech_pass(context, target, tmp, stt_deadline=stt_deadline,
                                    transcribe=transcribe, workers=workers, chunk_s=chunk_s,
                                    model=model, config=config) if (
            do_transcribe and not no_audio) else None

        # OCR runs while the chunks decode: the two stages share the clock, not a queue.
        read = 0
        if do_ocr and context.frames:
            stage = time.monotonic()
            for frame in context.frames:
                if _remaining(deadline) <= 0.5:
                    break
                ocr_frame(frame, timeout_s=min(120.0, _remaining(deadline)))
                read += 1
            context.timings["ocr_s"] = round(time.monotonic() - stage, 2)
            if read < len(context.frames):
                context.notes.append(f"OCR: {read} of {len(context.frames)} frames read (budget)")

        if speech is not None:
            _finish_speech_pass(context, speech[0], speech[1], stt_deadline)

        if keep_frames:
            context.notes.append(f"frames kept at {tmp / 'frames'}")
            tmp = None  # type: ignore[assignment]  # caller asked to keep them
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    context.timings["total_s"] = round(time.monotonic() - started, 2)
    return context


def _start_speech_pass(context: VideoContext, target: Path, tmp: Path, *, stt_deadline: float,
                       transcribe: Optional[Transcriber], workers: int, chunk_s: float,
                       model: str, config: Dict[str, Any]
                       ) -> Tuple[threading.Thread, Dict[str, Any]]:
    """Kick off STT on a worker thread so OCR proceeds in parallel."""
    box: Dict[str, Any] = {}

    def _job() -> None:
        stage = time.monotonic()
        try:
            backend, python, callable_ = _resolve_stt_engine(transcribe=transcribe,
                                                             config=config)
            box["backend"] = backend
            cpu_threads = int(config.get("cpu_threads") or
                              max(1, (os.cpu_count() or 1) // max(1, workers)))
            box["outcome"] = _stt_job(
                target, tmp, deadline=stt_deadline, transcribe=callable_,
                worker_python=python,
                duration_s=context.audio_duration_s or context.duration_s, workers=workers,
                chunk_s=chunk_s, silence_db=float(config.get("silence_db") or STT_SILENCE_DB),
                min_silence_s=float(config.get("min_silence_s") or STT_MIN_SILENCE_S),
                model=model, beam_size=int(config.get("beam_size") or STT_BEAM_SIZE_DEFAULT),
                compute_type=str(config.get("compute_type") or STT_COMPUTE_TYPE_DEFAULT),
                device=str(config.get("device") or "cpu"), cpu_threads=cpu_threads,
                language=(config.get("language") or "").strip() or None,
                timings=context.timings)
        except Exception as exc:  # noqa: BLE001 - a stage crash is reported, never raised
            box["backend"] = box.get("backend", "supplied")
            box["outcome"] = SttOutcome(
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
                audio_duration_s=context.audio_duration_s or context.duration_s)
        finally:
            context.timings["stt_s"] = round(time.monotonic() - stage, 2)

    thread = threading.Thread(target=_job, name="video-stt", daemon=True)
    thread.start()
    return thread, box


def _finish_speech_pass(context: VideoContext, thread: threading.Thread,
                        box: Dict[str, Any], stt_deadline: float) -> None:
    """Join the speech thread and attach whatever it really produced."""
    thread.join(timeout=max(0.5, _remaining(stt_deadline) + 5.0))
    context.stt_backend = str(box.get("backend") or "")
    outcome = box.get("outcome")
    if outcome is None:
        outcome = SttOutcome(
            error="time budget exhausted during speech-to-text (budget)",
            audio_duration_s=context.audio_duration_s or context.duration_s)
        context.timings.setdefault("stt_s", 0.0)
    _apply_stt_outcome(context, outcome)


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
