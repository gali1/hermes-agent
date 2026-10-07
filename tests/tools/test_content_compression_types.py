"""Tests for Phase 2 content router + per-type compressors.

Hermetic: tmp_path only, no network, no real HERMES_HOME. Config is faked via
monkeypatching agent.content_compression.config.get_content_compression_config.
"""

import json

import pytest

from agent.content_compression import (
    CompressionResult,
    compress_config,
    compress_diff,
    compress_json,
    compress_log,
    compress_search,
    compress_tabular,
    compress_tool_output,
    detect_content_type,
)
from agent.content_compression import config as cc_config
from agent.content_compression import json_crusher
from agent.content_compression.adaptive_sizer import compute_optimal_k
from agent.content_compression.config import DEFAULT_CONTENT_AWARE_CONFIG
from agent.content_compression.json_crusher import parse_csv_schema


def _cfg(**over):
    cfg = dict(DEFAULT_CONTENT_AWARE_CONFIG)
    cfg.update(over)
    return cfg


@pytest.fixture
def set_cc_config(monkeypatch):
    """Force the effective compression config for compress_tool_output."""

    def _apply(**over):
        cfg = _cfg(**over)
        monkeypatch.setattr(cc_config, "get_content_compression_config", lambda: dict(cfg))
        return cfg

    return _apply


# ── router ────────────────────────────────────────────────────────────


class TestRouter:
    def test_json_array_of_objects(self):
        kind, confidence = detect_content_type(json.dumps([{"id": i} for i in range(5)]))
        assert kind == "json"
        assert confidence >= 0.9

    def test_json_object(self):
        kind, confidence = detect_content_type('{"a": 1, "b": [1, 2]}')
        assert kind == "json"
        assert confidence >= 0.9

    def test_search(self):
        text = "src/main.py:12:def foo():\nsrc/util.py:3:import os\nlib/x.py:9:x = 1\n"
        kind, confidence = detect_content_type(text)
        assert kind == "search"
        assert confidence >= 0.6

    def test_log(self):
        text = "".join(
            f"2026-01-01 10:00:{i:02d} {'ERROR' if i % 2 else 'INFO'} event {i}\n"
            for i in range(10)
        )
        kind, confidence = detect_content_type(text)
        assert kind == "log"
        assert confidence >= 0.5

    def test_diff(self):
        text = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-old\n"
            "+new\n"
            " ctx\n"
        )
        kind, confidence = detect_content_type(text)
        assert kind == "diff"
        assert confidence >= 0.7

    def test_config_yaml(self):
        text = "name: app\nversion: 1\nserver:\n  host: localhost\n  port: 8080\n"
        kind, confidence = detect_content_type(text)
        assert kind == "config"
        assert confidence >= 0.6

    def test_config_toml(self):
        text = "[server]\nhost = 'localhost'\nport = 8080\n"
        kind, confidence = detect_content_type(text)
        assert kind == "config"
        assert confidence >= 0.6

    def test_tabular_csv(self):
        text = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
        kind, confidence = detect_content_type(text)
        assert kind == "tabular"
        assert confidence >= 0.6

    def test_html(self):
        text = (
            "<!DOCTYPE html><html><head><title>x</title></head>"
            "<body><div>hi</div><span>a</span><script>var x=1;</script></body></html>"
        )
        kind, confidence = detect_content_type(text)
        assert kind == "html"
        assert confidence >= 0.7

    def test_code(self):
        text = (
            "import os\n"
            "from sys import path\n"
            "\n"
            "class Gamma:\n"
            "    pass\n"
            "\n"
            "def alpha():\n"
            "    return 1\n"
            "\n"
            "def beta():\n"
            "    return 2\n"
        )
        kind, confidence = detect_content_type(text)
        assert kind == "code"
        assert confidence >= 0.5

    def test_fallback_text(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "It was a dark and stormy night, and the rain fell hard."
        )
        assert detect_content_type(text) == ("text", 0.5)

    def test_empty_is_text_zero(self):
        assert detect_content_type("") == ("text", 0.0)
        assert detect_content_type("   \n") == ("text", 0.0)

    def test_tool_name_prior_search(self):
        text = "The quick brown fox jumps over the lazy dog and keeps running far away."
        kind, confidence = detect_content_type(text, tool_name="search_files")
        assert kind == "search"
        assert confidence <= 0.6

    def test_tool_name_prior_terminal(self):
        text = "The quick brown fox jumps over the lazy dog and keeps running far away."
        kind, confidence = detect_content_type(text, tool_name="terminal")
        assert kind == "log"
        assert confidence <= 0.6

    def test_tool_name_prior_read_file_code(self):
        text = "import os\nfrom sys import path\nclass A:\n    pass\ndef f():\n    pass\n"
        assert detect_content_type(text, tool_name="read_file")[0] == "code"

    def test_prior_does_not_override_structural_json(self):
        text = json.dumps([{"id": i} for i in range(5)])
        assert detect_content_type(text, tool_name="terminal")[0] == "json"


# ── adaptive sizer ────────────────────────────────────────────────────


class TestAdaptiveSizer:
    def test_fast_path_small_input(self):
        assert compute_optimal_k(["a", "b", "c"]) == 3
        assert compute_optimal_k(["a", "b", "c"], max_k=2) == 2

    def test_never_exceeds_max_k(self):
        items = [f"item {i} unique token {i * 3}" for i in range(100)]
        assert compute_optimal_k(items, min_k=5, max_k=7) <= 7

    def test_knee_below_total_for_redundant_tail(self):
        items = [f"distinct sentence number {i} with words {i * 7}" for i in range(12)]
        items += ["the same repeated line"] * 48
        k = compute_optimal_k(items, min_k=5, max_k=30)
        assert 5 <= k < len(items)

    def test_deterministic(self):
        items = [f"row {i} has value {i * 13}" for i in range(50)]
        assert compute_optimal_k(items) == compute_optimal_k(items)

    def test_accepts_pairs_and_dicts(self):
        texts = [f"entry number {i} with token {i * 11}" for i in range(40)]
        pairs = [(text, 0.0) for text in texts]
        dicts = [{"text": text, "score": 0.0} for text in texts]
        assert compute_optimal_k(texts) == compute_optimal_k(pairs)
        assert compute_optimal_k(texts) == compute_optimal_k(dicts)

    def test_handles_empty(self):
        assert compute_optimal_k([]) == 0


# ── JSON crusher ──────────────────────────────────────────────────────


def _flat_items(n=20):
    return [
        {"id": i, "name": f"user_{i}", "status": "ok" if i % 3 else "pending"}
        for i in range(n)
    ]


def _lossy_items(n=60):
    """Array whose CSV-schema render cannot be adopted (comma in a key)."""
    items = []
    for i in range(n):
        items.append({"idx": i, "value": 10, "ts,ms": i, "msg": "ok"})
    items[30]["msg"] = "error: failed request"
    items[40]["value"] = 999999
    return items


class TestJsonCrusher:
    def test_lossless_render_round_trips(self):
        items = _flat_items()
        text = json.dumps(items)
        out, dropped = compress_json(text)
        assert dropped == 0
        assert len(out) < len(text)
        assert out.startswith("[20]{")
        assert parse_csv_schema(out) == items

    def test_schema_header_dedups_repeated_keys(self):
        items = _flat_items()
        out, _ = compress_json(json.dumps(items))
        assert out.count("name:") == 1
        assert out.count("status:") == 1

    def test_duplicate_items_round_trip(self):
        items = [{"id": 1, "v": "x"}] * 10
        out, dropped = compress_json(json.dumps(items))
        assert dropped == 0
        assert parse_csv_schema(out) == items

    def test_render_round_trips_commas_quotes_nulls_newlines(self):
        items = [
            {"id": 1, "msg": 'a, "b"', "note": None, "empty": "", "lit": "null"},
            {"id": 2, "msg": "line1\nline2", "note": "n", "empty": "e", "lit": "x"},
            {"id": 3, "msg": "plain", "note": None, "empty": "e2", "lit": "null"},
            {"id": 4, "msg": "quote\"end", "note": "n2", "empty": "", "lit": "y"},
            {"id": 5, "msg": "tail, comma", "note": "n3", "empty": "e3", "lit": "z"},
        ]
        out, dropped = compress_json(json.dumps(items))
        assert dropped == 0
        assert parse_csv_schema(out) == items

    def test_lossy_retains_first_last_error_anomaly(self):
        items = _lossy_items()
        text = json.dumps(items, separators=(",", ":"))
        out, dropped = compress_json(
            text, allow_lossy=True, retrieval_hint="/sandbox/orig.json", max_items=15
        )
        assert dropped == 45
        assert "(retrieve: /sandbox/orig.json)" in out
        assert "failed request" in out
        assert "999999" in out
        assert '"idx":0' in out
        assert '"idx":59' in out
        assert "omitted" in out

    def test_lossy_requires_hint(self):
        text = json.dumps(_lossy_items(), separators=(",", ":"))
        out, dropped = compress_json(text, allow_lossy=True)
        assert dropped == 0
        assert "omitted" not in out

    def test_lossy_requires_flag(self):
        text = json.dumps(_lossy_items(), separators=(",", ":"))
        out, dropped = compress_json(text, retrieval_hint="/x")
        assert dropped == 0
        assert "omitted" not in out

    def test_malformed_json_unchanged(self):
        text = "{not json at all"
        assert compress_json(text) == (text, 0)

    def test_pretty_object_compacted_losslessly(self):
        text = json.dumps({"a": [1, 2, 3], "b": {"c": 4}}, indent=2)
        out, dropped = compress_json(text)
        assert dropped == 0
        assert len(out) < len(text)
        assert json.loads(out) == json.loads(text)

    def test_non_dict_array_compacted(self):
        text = json.dumps([1, 2, 3, 4, 5, 6], indent=2)
        out, dropped = compress_json(text)
        assert dropped == 0
        assert len(out) < len(text)
        assert json.loads(out) == [1, 2, 3, 4, 5, 6]


# ── log compressor ────────────────────────────────────────────────────


def _log_text():
    lines = [f"2026-01-01 10:00:{i % 60:02d} INFO step {i}" for i in range(300)]
    lines[100] = "2026-01-01 10:01:00 ERROR first error"
    lines[99] = "2026-01-01 10:00:59 context before first error"
    lines[200] = "2026-01-01 10:03:20 ERROR last error"
    for k, pos in enumerate([110, 120, 130, 140, 150, 160, 170, 180, 190, 195]):
        lines[pos] = f"2026-01-01 10:02:{k:02d} ERROR error number {k}"
    for b in range(30):
        lines.append("Traceback (most recent call last):")
        for f in range(10):
            lines.append(f'  File "/app/mod{b}.py", line {f + 1}, in func{f}')
            lines.append(f"    do_work_{f}()")
        lines.append(f"RuntimeError: boom {b}")
    for i in range(40):
        lines.append(f"2026-01-01 10:05:00 WARN warning {i}")
    lines.append("===== 5 failed, 95 passed, 2 skipped in 1.23s =====")
    return "\n".join(lines) + "\n"


class TestLogCompressor:
    def test_collapses_runs_losslessly_without_hint(self):
        text = "same line\n" * 5 + "end\n"
        out, dropped = compress_log(text)
        assert dropped == 0
        assert "(repeated 5 times)" in out
        assert len(out) < len(text)

    def test_lossy_gated_by_hint(self):
        text = _log_text()
        out, dropped = compress_log(text, max_lines=50)
        assert dropped == 0
        assert "omitted" not in out
        out2, dropped2 = compress_log(text, allow_lossy=True, retrieval_hint="/x", max_lines=50)
        assert dropped2 > 0
        assert "(retrieve: /x)" in out2
        assert len(out2) < len(text)

    def test_errors_and_context_retained(self):
        out, _ = compress_log(
            _log_text(), allow_lossy=True, retrieval_hint="/x", max_lines=50
        )
        assert "first error" in out
        assert "context before first error" in out

    def test_error_count_capped_at_ten(self):
        lines = [f"2026-01-01 10:00:{i % 60:02d} INFO filler {i}" for i in range(300)]
        for k in range(15):
            lines[20 + k * 15] = f"2026-01-01 10:01:{k:02d} ERROR marker-{k:02d}"
        text = "\n".join(lines) + "\n"
        out, dropped = compress_log(text, allow_lossy=True, retrieval_hint="/x", max_lines=50)
        assert dropped > 0
        assert "marker-00" in out
        assert "marker-14" in out
        assert "filler 19" in out
        assert sum(1 for line in out.splitlines() if "ERROR" in line) <= 10

    def test_stack_traces_capped_and_trimmed(self):
        out, _ = compress_log(
            _log_text(), allow_lossy=True, retrieval_hint="/x", max_lines=50
        )
        assert sum(1 for line in out.splitlines() if "Traceback" in line) <= 3
        assert '"/app/mod0.py", line 6' in out
        assert '"/app/mod0.py", line 7' not in out

    def test_summary_line_retained(self):
        out, _ = compress_log(
            _log_text(), allow_lossy=True, retrieval_hint="/x", max_lines=50
        )
        assert "5 failed, 95 passed, 2 skipped" in out

    def test_warnings_deduped_and_capped(self):
        text = "".join(f"WARN repeated warning\n" for _ in range(10)) + "".join(
            f"WARN unique warning {i}\n" for i in range(20)
        )
        text += "".join(f"INFO filler {i}\n" for i in range(200))
        out, dropped = compress_log(text, allow_lossy=True, retrieval_hint="/x", max_lines=50)
        assert dropped > 0
        assert sum(1 for line in out.splitlines() if "WARN" in line) <= 5


# ── search compressor ─────────────────────────────────────────────────


def _listing(files=3, per_file=20):
    lines = []
    for f in range(files):
        for i in range(per_file):
            lines.append(f"src/mod{f}.py:{i + 1}:match number {i} in file {f}")
    return "\n".join(lines) + "\n"


class TestSearchCompressor:
    def test_lossy_gated_by_hint(self):
        text = _listing()
        assert compress_search(text) == (text, 0)

    def test_per_file_cap_and_markers(self):
        text = _listing()
        out, dropped = compress_search(
            text,
            allow_lossy=True,
            retrieval_hint="/x",
            max_total=30,
            max_per_file=5,
            max_files=15,
        )
        assert dropped > 0
        for f in range(3):
            kept = [ln for ln in out.splitlines() if ln.startswith(f"src/mod{f}.py:")]
            assert len(kept) <= 5
        assert out.count("more matches in") == 3
        assert out.count("(retrieve: /x)") == 1
        assert len(out) < len(text)

    def test_first_and_last_always_kept(self):
        text = _listing()
        out, _ = compress_search(
            text, allow_lossy=True, retrieval_hint="/x", max_total=10, max_per_file=2
        )
        assert "src/mod0.py:1:match number 0 in file 0" in out
        assert "src/mod2.py:20:match number 19 in file 2" in out

    def test_max_files_cap(self):
        text = _listing(files=5, per_file=10)
        out, dropped = compress_search(
            text,
            allow_lossy=True,
            retrieval_hint="/x",
            max_total=30,
            max_per_file=5,
            max_files=2,
        )
        assert dropped > 0
        assert "src/mod0.py:" in out
        assert "src/mod1.py:" in out
        for f in (2, 3):
            assert f"src/mod{f}.py:" not in out
        # The last match overall is always kept, even in a dropped file.
        assert "src/mod4.py:10:match number 9 in file 4" in out

    def test_error_lines_score_higher(self):
        lines = [f"src/a.py:{i + 1}:ordinary match {i}" for i in range(30)]
        lines[15] = "src/a.py:16:ERROR failed to connect"
        text = "\n".join(lines) + "\n"
        out, _ = compress_search(
            text, allow_lossy=True, retrieval_hint="/x", max_total=5, max_per_file=5
        )
        assert "ERROR failed to connect" in out

    def test_parses_line_col_form(self):
        lines = [f"src/a.py:{i + 1}:5:match {i}" for i in range(30)]
        text = "\n".join(lines) + "\n"
        out, dropped = compress_search(
            text, allow_lossy=True, retrieval_hint="/x", max_total=10, max_per_file=5
        )
        assert dropped > 0
        assert "src/a.py:1:5:match 0" in out


# ── diff compressor ───────────────────────────────────────────────────


def _diff_text():
    first = ["diff --git a/a.py b/a.py", "--- a/a.py", "+++ b/a.py", "@@ -1,24 +1,24 @@"]
    first += [f" ctx line {i}" for i in range(10)]
    first += ["-old line", "+new line"]
    first += [f" ctx tail {i}" for i in range(10)]
    second = ["diff --git a/b.py b/b.py", "--- a/b.py", "+++ b/b.py", "@@ -1,3 +1,3 @@"]
    second += ["-x", "+y", " z"]
    return "\n".join(first + second) + "\n"


class TestDiffCompressor:
    def test_gated_by_hint(self):
        text = _diff_text()
        assert compress_diff(text) == (text, 0)

    def test_keeps_changes_and_headers_caps_context(self):
        text = _diff_text()
        out, dropped = compress_diff(
            text, allow_lossy=True, retrieval_hint="/x", max_context_lines=2
        )
        assert dropped > 0
        for header in (
            "diff --git a/a.py b/a.py",
            "--- a/a.py",
            "+++ b/a.py",
            "@@ -1,24 +1,24 @@",
            "diff --git a/b.py b/b.py",
        ):
            assert header in out
        assert "-old line" in out
        assert "+new line" in out
        assert "-x" in out
        assert "+y" in out
        assert "ctx line 1" in out
        assert "ctx line 2" not in out
        assert "ctx tail 9" not in out
        assert out.count("context lines omitted") == 2

    def test_hunks_per_file_cap(self):
        lines = ["diff --git a/a.py b/a.py", "--- a/a.py", "+++ b/a.py"]
        for h in range(15):
            lines.append(f"@@ -{h + 1},2 +{h + 1},2 @@")
            lines.append(f"-old {h}")
            lines.append(f"+new {h}")
        text = "\n".join(lines) + "\n"
        out, dropped = compress_diff(
            text, allow_lossy=True, retrieval_hint="/x", max_hunks_per_file=3
        )
        assert dropped > 0
        assert "-old 0" in out
        assert "+new 2" in out
        assert "-old 3" not in out
        assert "hunks omitted" in out

    def test_files_cap(self):
        lines = []
        for f in range(5):
            lines.append(f"diff --git a/f{f}.py b/f{f}.py")
            lines.append(f"--- a/f{f}.py")
            lines.append(f"+++ b/f{f}.py")
            lines.append("@@ -1,1 +1,1 @@")
            lines.append(f"-old {f}")
            lines.append(f"+new {f}")
        text = "\n".join(lines) + "\n"
        out, dropped = compress_diff(text, allow_lossy=True, retrieval_hint="/x", max_files=2)
        assert dropped > 0
        assert "+new 1" in out
        assert "+new 2" not in out
        assert "files omitted" in out


# ── config compressor ─────────────────────────────────────────────────


class TestConfigCompressor:
    def test_lossless_fold_without_hint(self):
        block = "".join(f"line_{i}: value_{i} with some extra text\n" for i in range(5))
        text = block * 3
        out, dropped = compress_config(text)
        assert dropped == 0
        assert len(out) < len(text)
        assert "(repeats 5 lines" in out

    def test_comment_elision_gated_by_hint(self):
        comments = "".join(
            f"# explanatory comment number {i} with several words\n" for i in range(30)
        )
        text = comments + "name: app\nversion: 1\n" + "".join(
            f"key_{i}: value_{i}\n" for i in range(20)
        )
        out_plain, dropped_plain = compress_config(text)
        assert dropped_plain == 0
        assert "# explanatory comment number 0" in out_plain
        out, dropped = compress_config(text, allow_lossy=True, retrieval_hint="/x")
        assert dropped == 30
        assert "# explanatory comment number 0" not in out
        assert "(retrieve: /x)" in out
        assert len(out) < len(text)

    def test_block_scalar_skips_elision(self):
        text = "script: |\n  # not a comment\n  echo hi\n" + "".join(
            f"key_{i}: value_{i}\n" for i in range(20)
        )
        out, dropped = compress_config(text, allow_lossy=True, retrieval_hint="/x")
        assert dropped == 0
        assert "# not a comment" in out

    def test_toml_multiline_skips_elision(self):
        text = 'note = """\n# not a comment\n"""\n' + "".join(
            f"key_{i} = {i}\n" for i in range(20)
        )
        _, dropped = compress_config(text, allow_lossy=True, retrieval_hint="/x")
        assert dropped == 0


# ── tabular compressor ────────────────────────────────────────────────


class TestTabularCompressor:
    def test_markdown_table_round_trip(self):
        rows = [{"id": str(i), "name": f"user_{i}", "status": "ok"} for i in range(20)]
        body = "\n".join(f"| {r['id']} | {r['name']} | {r['status']} |" for r in rows)
        text = f"| id | name | status |\n| --- | --- | --- |\n{body}\n"
        out, dropped = compress_tabular(text)
        assert dropped == 0
        assert len(out) < len(text)
        assert out.startswith("[20]{")
        assert parse_csv_schema(out) == rows

    def test_ragged_csv_unchanged(self):
        text = "a,b,c\n1,2,3\n4,5\n6,7,8\n"
        assert compress_tabular(text) == (text, 0)

    def test_duplicate_headers_unchanged(self):
        text = "a,a,b\n1,2,3\n4,5,6\n7,8,9\n"
        assert compress_tabular(text) == (text, 0)

    def test_compact_csv_not_smaller_unchanged(self):
        text = "id,name,status\n" + "".join(f"{i},user_{i},ok\n" for i in range(20))
        assert compress_tabular(text) == (text, 0)


# ── compress_tool_output integration ──────────────────────────────────


class TestCompressToolOutputTypes:
    def test_json_lossless_strategy(self, set_cc_config):
        set_cc_config()
        text = json.dumps(_flat_items(), indent=2)
        result = compress_tool_output(text)
        assert isinstance(result, CompressionResult)
        assert result.strategy == "json"
        assert result.changed is True
        assert result.lossless is True
        assert result.dropped_units == 0
        assert len(result.text) < len(text)

    def test_lossy_json_strategy_with_writer(self, set_cc_config):
        set_cc_config()
        text = json.dumps(_lossy_items(), separators=(",", ":"))
        seen = []

        def _writer(original):
            seen.append(original)
            return "/sandbox/orig.txt"

        result = compress_tool_output(text, retrieval_writer=_writer)
        assert result.strategy == "json"
        assert result.dropped_units == 45
        assert result.lossless is False
        assert "(retrieve: /sandbox/orig.txt)" in result.text
        assert seen == [text]

    def test_lossy_json_no_writer_no_drop(self, set_cc_config):
        set_cc_config()
        text = json.dumps(_lossy_items(), separators=(",", ":"))
        result = compress_tool_output(text)
        assert result.dropped_units == 0
        assert "omitted" not in result.text
        assert result.lossless is True

    def test_writer_not_called_when_lossless_suffices(self, set_cc_config):
        set_cc_config()
        calls = []
        text = json.dumps(_flat_items(), indent=2)
        result = compress_tool_output(text, retrieval_writer=lambda o: calls.append(o) or "/x")
        assert result.strategy == "json"
        assert calls == []

    def test_tabular_strategy(self, set_cc_config):
        set_cc_config()
        rows = [{"id": str(i), "name": f"user_{i}"} for i in range(20)]
        body = "\n".join(f"| {r['id']} | {r['name']} |" for r in rows)
        text = f"| id | name |\n| --- | --- |\n{body}\n"
        result = compress_tool_output(text)
        assert result.strategy == "tabular"
        assert result.changed is True

    def test_lossless_fold_strategy(self, set_cc_config):
        set_cc_config()
        text = "this is a repeated log line\n" * 5 + "end\n"
        result = compress_tool_output(text)
        assert result.strategy == "lossless"
        assert result.lossless is True

    def test_dense_strategy(self, set_cc_config):
        set_cc_config(dense_line_elision=True)
        text = "x" * 1200 + "\n" + "y" * 1200
        result = compress_tool_output(text, retrieval_writer=lambda o: "/x")
        assert result.strategy == "dense"
        assert result.lossless is False
        assert result.elided_lines == 2

    def test_fail_open_when_type_compressor_raises(self, set_cc_config, monkeypatch):
        set_cc_config()

        def _boom(*args, **kwargs):
            raise RuntimeError("crusher exploded")

        monkeypatch.setattr(json_crusher, "compress_json", _boom)
        text = json.dumps(_flat_items(), separators=(",", ":"))
        result = compress_tool_output(text)
        assert isinstance(result, CompressionResult)
        assert result.text == text
        assert result.changed is False

    def test_disabled_config_byte_identical(self, set_cc_config):
        set_cc_config(enabled=False)
        text = json.dumps(_flat_items(), indent=2)
        result = compress_tool_output(text)
        assert result.text == text
        assert result.changed is False
        assert result.strategy == "none"

    def test_type_compression_off_byte_identical(self, set_cc_config):
        set_cc_config(type_compression=False)
        text = json.dumps(_flat_items(), separators=(",", ":"))
        result = compress_tool_output(text)
        assert result.text == text
        assert result.strategy == "none"

    def test_result_defaults(self):
        result = CompressionResult(
            text="x",
            changed=False,
            lossless=True,
            elided_lines=0,
            original_chars=1,
            new_chars=1,
        )
        assert result.strategy == "none"
        assert result.dropped_units == 0
