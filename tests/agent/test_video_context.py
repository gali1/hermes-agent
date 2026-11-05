"""Behaviour of the video-to-text path that a model would otherwise have to guess at.

The clips here are generated with ffmpeg in the test, so the tests prove the same ffmpeg/OCR
path the product uses - no fixture binaries. Local STT is replaced by a stub (the real Whisper
`base` model runs at roughly real time; a unit test must not wait for it).
"""

from __future__ import annotations

import re
import struct
import subprocess
import threading
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
    # Three identifiable tones separated by two long silences: enough to prove that silences are
    # not decoded and that chunks come back in time order (each tone names its own position).
    tones = root / "tones.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=10:d=10",
            "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=16000:duration=2",
            "-f", "lavfi", "-i", "aevalsrc=0:s=16000:d=2",
            "-f", "lavfi", "-i", "sine=frequency=600:sample_rate=16000:duration=2",
            "-f", "lavfi", "-i", "aevalsrc=0:s=16000:d=2",
            "-f", "lavfi", "-i", "sine=frequency=900:sample_rate=16000:duration=2",
            "-filter_complex", "[1:a][2:a][3:a][4:a][5:a]concat=n=5:v=0:a=1[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(tones))
    hush = root / "hush.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=10:d=4",
            "-f", "lavfi", "-i", "aevalsrc=0:s=16000:d=4",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(hush))
    return {"short": short, "silent": silent, "ten": ten, "tones": tones, "hush": hush}


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


# --- silence skipping, parallel chunks, coverage honesty --------------------------------------


def _dominant_tone(audio: Path) -> str:
    """Name the tone in a chunk from its zero-crossing rate: 300 Hz -> 'tone300'.

    Only samples above the silence floor are counted: a chunk carries a little padded silence at
    its edges, and dividing by the padded duration would read 600 Hz as 500 Hz.
    """
    with wave.open(str(audio), "rb") as handle:
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    count = len(raw) // 2
    samples = [value for value in struct.unpack(f"<{count}h", raw) if abs(value) > 512]
    if len(samples) < 2 or not rate:
        return "silence"
    crossings = sum(1 for first, second in zip(samples, samples[1:])
                    if (first < 0) != (second < 0))
    frequency = crossings * rate / (2.0 * len(samples))
    nearest = min((300, 600, 900), key=lambda tone: abs(tone - frequency))
    return f"tone{nearest}"


def test_silence_is_skipped_and_parallel_chunks_stitch_in_time_order(clips):
    """Two long silences cost nothing, three tones are decoded at once, order is preserved."""
    calls: list[tuple[str, float]] = []
    active = 0
    peak = 0
    lock = threading.Lock()

    def tone_stub(audio: Path):
        nonlocal active, peak
        with wave.open(str(audio), "rb") as handle:
            seconds = handle.getnframes() / handle.getframerate()
        tone = _dominant_tone(audio)
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append((tone, round(seconds, 3)))
        time.sleep(0.2)  # hold the worker open so parallel launch is observable
        with lock:
            active -= 1
        return True, tone, ""

    ctx = vc.build_video_context(str(clips["tones"]), max_frames=1, do_ocr=False,
                                 transcribe=tone_stub, stt_workers=3, time_budget_s=30)

    assert ctx.transcript.split() == ["tone300", "tone600", "tone900"], ctx.transcript
    assert peak >= 2, f"chunks ran one after another: {calls!r}"
    assert ctx.stt_chunks >= 3
    assert ctx.stt_workers == 3
    decoded = sum(seconds for _tone, seconds in calls)
    assert decoded <= ctx.audio_duration_s - 3.0, (
        f"{decoded:.1f}s decoded of {ctx.audio_duration_s:.1f}s - silence was not skipped")
    assert ctx.silence_skipped_s >= 3.0
    line = next(n for n in ctx.notes if n.startswith("audio transcribed 0:00-"))
    assert "(complete" in line and "silence skipped" in line, line
    assert line in ctx.as_text()
    assert ctx.covered_s >= ctx.audio_duration_s - 0.5


def test_a_failed_chunk_stops_the_coverage_claim_at_the_gap(clips):
    def flaky_stub(audio: Path):
        tone = _dominant_tone(audio)
        return (False, "", "decoder blew up") if tone == "tone600" else (True, tone, "")

    ctx = vc.build_video_context(str(clips["tones"]), max_frames=1, do_ocr=False,
                                 transcribe=flaky_stub, stt_workers=3, time_budget_s=30)

    assert ctx.transcript.split() == ["tone300"], (
        "text after a failed chunk must not be claimed as covered audio")
    assert ctx.covered_s < ctx.audio_duration_s - 1.0
    assert "decoder blew up" in ctx.transcript_error
    line = next(n for n in ctx.notes if n.startswith("audio transcribed 0:00-"))
    assert line.endswith("(partial)"), line  # not a budget stop - say what it was


def test_all_silence_audio_is_declared_not_transcribed(clips):
    def must_not_run(audio):
        raise AssertionError("silence must never reach the model")

    ctx = vc.build_video_context(str(clips["hush"]), max_frames=1, do_ocr=False,
                                 transcribe=must_not_run)

    assert ctx.transcript == ""
    assert "no speech detected" in ctx.transcript_error
    notes = [n for n in ctx.notes if n.startswith("audio transcribed 0:00-")]
    assert notes and "no speech detected" in notes[0], notes
    assert "SPEECH: unavailable" in ctx.as_text()


def test_ocr_and_stt_stages_run_concurrently(clips, monkeypatch):
    def slow_ocr(frame, *, lang="eng", timeout_s=120.0):
        time.sleep(0.6)
        return frame

    def slow_stt(audio):
        time.sleep(1.2)
        return True, "overlap", ""

    monkeypatch.setattr(vc, "ocr_frame", slow_ocr)

    ctx = vc.build_video_context(str(clips["short"]), max_frames=4, transcribe=slow_stt,
                                 time_budget_s=60)

    assert len(ctx.frames) >= 2
    assert ctx.timings["ocr_s"] >= 1.2 and ctx.timings["stt_s"] >= 1.2
    # Sequential stages would cost ocr+stt; overlapping ones cost about the slower stage.
    assert ctx.timings["total_s"] <= ctx.timings["ocr_s"] + ctx.timings["stt_s"] - 0.5, ctx.timings


def test_stt_budget_is_separable_from_the_overall_budget(clips):
    def slow_stub(audio: Path):
        with wave.open(str(audio), "rb") as handle:
            seconds = handle.getnframes() / handle.getframerate()
        time.sleep(seconds * 0.25)  # ~4x faster than real time
        return True, f"[{seconds:.1f}s]", ""

    generous = vc.build_video_context(str(clips["ten"]), max_frames=1, do_ocr=False,
                                      transcribe=slow_stub, time_budget_s=1.5,
                                      stt_budget_s=40.0)
    tight = vc.build_video_context(str(clips["ten"]), max_frames=1, do_ocr=False,
                                   transcribe=slow_stub, time_budget_s=1.5)

    assert generous.stt_budget_s == 40.0 and tight.stt_budget_s == 1.5
    assert generous.covered_s >= generous.audio_duration_s - 1.0
    assert any(n.endswith("(complete)") for n in generous.notes), generous.notes
    assert tight.covered_s < generous.covered_s
    assert any(n.endswith("(budget)") for n in tight.notes), tight.notes


def test_complete_coverage_line_states_the_exact_span(clips):
    ctx = vc.build_video_context(str(clips["ten"]), max_frames=1, do_ocr=False,
                                 transcribe=lambda audio: (True, "hello", ""))

    total = vc._fmt_mmss(ctx.audio_duration_s)
    line = f"audio transcribed 0:00-{total} of {total} (complete)"
    text = ctx.as_text()

    assert line in ctx.notes
    assert line in text
    assert f"SPEECH (local STT, 0:00-{total} of {total} (complete))" in text
    assert "SPEAKERS: unavailable" in text


def test_stt_budget_model_and_workers_come_from_config(clips, monkeypatch):
    monkeypatch.setattr(vc, "_video_stt_config",
                        lambda: {"budget_s": 7.5, "model": "tiny", "workers": 2})

    ctx = vc.build_video_context(str(clips["short"]), max_frames=1, do_ocr=False,
                                 transcribe=lambda audio: (True, "x", ""))

    assert ctx.stt_budget_s == 7.5
    assert ctx.stt_model == "tiny"
    assert ctx.stt_workers == 2
    # A caller argument still beats the config; no config falls back to the overall budget.
    assert vc._resolve_stt_model("base", {"model": "tiny"}) == "base"
    assert vc._resolve_stt_model(None, {}) == vc.DEFAULT_STT_MODEL
    assert vc._resolve_stt_budget(12.0, {"budget_s": 7.5}, 240.0) == 12.0
    assert vc._resolve_stt_budget(None, {}, 240.0) == 240.0


def test_chunk_planning_splits_long_speech_and_merges_short_gaps():
    assert vc._plan_chunks([(0.0, 70.0)], chunk_s=30.0) == [
        (0.0, 30.0), (30.0, 60.0), (60.0, 70.0)]
    assert vc._plan_chunks([(0.0, 10.0), (11.0, 20.0)], chunk_s=30.0) == [(0.0, 20.0)]
    assert vc._plan_chunks([(0.0, 2.0), (6.0, 8.0)], chunk_s=30.0) == [(0.0, 2.0), (6.0, 8.0)]


def test_worker_partition_gives_each_process_contiguous_time_order():
    assert vc._partition_chunks(9, 4) == [[0, 1, 2], [3, 4], [5, 6], [7, 8]]
    assert vc._partition_chunks(3, 8) == [[0], [1], [2]]  # never more workers than chunks
    assert vc._partition_chunks(1, 4) == [[0]]


def test_engine_resolution_keeps_a_supplied_transcriber_and_honours_config():
    def supplied(audio):
        return True, "x", ""

    backend, python, callable_ = vc._resolve_stt_engine(transcribe=supplied, config={})
    assert (backend, python) == ("supplied", None) and callable_ is supplied

    backend, python, callable_ = vc._resolve_stt_engine(
        transcribe=None, config={"backend": "provider"})
    assert backend == "provider" and python is None and callable_ is not None
