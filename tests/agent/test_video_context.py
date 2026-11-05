"""Behaviour of the video-to-text path that a model would otherwise have to guess at.

The clips here are generated with ffmpeg in the test, so the tests prove the same ffmpeg/OCR
path the product uses - no fixture binaries. Local STT is replaced by a stub (the real Whisper
`base` model runs at roughly real time; a unit test must not wait for it).
"""

from __future__ import annotations

import re
import subprocess
import time
import wave
from pathlib import Path

import pytest

from agent import video_context as vc

pytestmark = pytest.mark.skipif(not vc.ffmpeg_available(),
                                reason="ffmpeg and ffprobe are required for these tests")


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, timeout=180)


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("video-context")
    short = root / "short.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=10:d=4",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:d=4",
            "-vf", "drawtext=text='HELLO 42':fontcolor=white:fontsize=36:x=24:y=24",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(short))
    silent = root / "silent.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=10:d=4",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(silent))
    ten = root / "ten.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=10:d=10",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:d=10",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(ten))
    return {"short": short, "silent": silent, "ten": ten}


def test_no_audio_track_is_reported_and_speech_is_never_invented(clips):
    def must_not_run(audio):
        raise AssertionError("STT must not be called for a video without an audio stream")

    ctx = vc.build_video_context(str(clips["silent"]), transcribe=must_not_run)

    assert ctx.transcript == ""
    assert ctx.transcript_error == "no audio track"
    text = ctx.as_text()
    assert "SPEECH: unavailable (no audio track)" in text
    assert "SPEAKERS:" not in text  # there is nothing to attribute
    assert "no audio stream in the container" in ctx.notes


def test_stt_stops_at_budget_and_states_the_covered_span(clips, monkeypatch):
    """A slow transcriber must not be allowed to eat the whole budget silently."""
    monkeypatch.setattr(vc, "extract_frames", lambda *a, **k: [])

    def slow_stub(audio: Path):
        with wave.open(str(audio), "rb") as handle:
            seconds = handle.getnframes() / handle.getframerate()
        time.sleep(seconds * 0.25)  # simulate ~4x faster than real time
        return True, f"[covered {seconds:.1f}s]", ""

    ctx = vc.build_video_context(str(clips["ten"]), max_frames=2, do_ocr=False,
                                 transcribe=slow_stub, time_budget_s=2.0)

    assert ctx.transcript, "the part that was transcribed must be kept"
    assert 0 < ctx.covered_s < ctx.audio_duration_s - 0.5
    budget_notes = [n for n in ctx.notes if n.startswith("audio transcribed 0:00-")]
    assert budget_notes, f"no coverage note in {ctx.notes!r}"
    assert budget_notes[0].endswith("(budget)")
    assert budget_notes[0] in ctx.as_text()
    assert f"of {vc._fmt_mmss(ctx.audio_duration_s)}" in budget_notes[0]
    assert ctx.stt_realtime_factor > 0


def test_budget_exhausted_before_frames_states_coverage(clips):
    ctx = vc.build_video_context(str(clips["short"]), time_budget_s=0.0,
                                 transcribe=lambda audio: pytest.fail("STT ran past the budget"))

    assert ctx.frames == []
    assert any(n.startswith("frames: 0 of ") and n.endswith("(budget)") for n in ctx.notes)
    assert any(n.startswith("audio transcribed 0:00-0:00 of ") and n.endswith("(budget)")
               for n in ctx.notes)
    assert "time budget exhausted" in ctx.transcript_error


def test_as_text_marks_truncated_output_with_counts():
    ctx = vc.VideoContext(path="talk.mp4", duration_s=300.0, width=640, height=360,
                          transcript="word " * 4000, audio_duration_s=300.0, covered_s=300.0)

    text = ctx.as_text(max_chars=400)

    assert len(text) <= 400
    match = re.search(r"\[TRUNCATED: transcript (\d+) chars; showing (\d+) of (\d+) chars\]",
                      text)
    assert match, text[-160:]
    transcript_chars, shown, total = (int(g) for g in match.groups())
    assert transcript_chars == len(ctx.transcript.strip())
    assert total > shown > 0
    assert text.startswith("VIDEO: talk.mp4")


def test_speaker_labels_are_declared_unavailable_not_guessed():
    ctx = vc.VideoContext(path="panel.mp4", duration_s=60.0, transcript="one two three",
                          audio_duration_s=60.0, covered_s=60.0)

    text = ctx.as_text()

    assert text.count("SPEAKERS: unavailable") == 1
    assert "no diarization" in text


def test_short_clip_yields_frames_and_transcript(clips):
    seen = {}

    def stub(audio: Path):
        with wave.open(str(audio), "rb") as handle:
            seen["rate"] = handle.getframerate()
            seen["channels"] = handle.getnchannels()
        return True, "hello from the stub", ""

    ctx = vc.build_video_context(str(clips["short"]), max_frames=4, transcribe=stub)

    assert 0 < ctx.duration_s < 6
    assert ctx.width == 320 and ctx.height == 240
    assert 1 <= len(ctx.frames) <= 4
    assert all(frame.path.endswith(".jpg") for frame in ctx.frames)
    stamps = [frame.timestamp_s for frame in ctx.frames]
    assert stamps == sorted(stamps)
    assert stamps[-1] < ctx.duration_s + 1.0
    assert seen == {"rate": 16000, "channels": 1}
    text = ctx.as_text()
    assert "hello from the stub" in text
    assert "SPEECH (local STT" in text and "ON SCREEN" in text
    assert ctx.transcript_error == ""
