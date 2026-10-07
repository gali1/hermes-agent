"""Tests for agent/content_compression -- Headroom-derived Phase 1.

Hermetic: tmp_path only, no network, no real HERMES_HOME. Config is faked via
monkeypatching agent.content_compression.config.get_content_compression_config.
"""

import pytest

from agent.content_compression import (
    CompressionResult,
    compress_tool_output,
    elide_dense_lines,
    fold_lossless,
    is_critical_line,
    strip_ansi,
)
from agent.content_compression import config as cc_config
from agent.content_compression import lossless as lossless_mod
from agent.content_compression.config import DEFAULT_CONTENT_AWARE_CONFIG
from agent.content_compression.dense_lines import (
    HEAD_CHARS,
    TAIL_CHARS,
    is_dense_line,
)
from agent.content_compression.lossless import (
    collapse_runs,
    expand_runs,
    fold_path_listing,
    fold_repeated_blocks,
    path_unheading,
    unfold_repeated_blocks,
)
from agent.content_compression.protection import CRITICAL_LINE_RE, MUST_KEEP_RE, is_error_line
from tools.tool_result_storage import maybe_persist_tool_result


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


class _FakeEnv:
    """Minimal sandbox env: execute() + get_temp_dir() like BaseEnvironment."""

    def __init__(self, temp_dir, returncode=0):
        self.temp_dir = str(temp_dir)
        self.returncode = returncode
        self.calls = []

    def get_temp_dir(self):
        return self.temp_dir

    def execute(self, cmd, timeout=None, stdin_data=None):
        self.calls.append({"cmd": cmd, "timeout": timeout, "stdin_data": stdin_data})
        return {"output": "", "returncode": self.returncode}


# ── strip_ansi ────────────────────────────────────────────────────────


class TestStripAnsi:
    def test_removes_color_codes(self):
        assert strip_ansi("\x1b[31mERROR\x1b[0m plain") == "ERROR plain"

    def test_no_op_without_ansi(self):
        text = "plain log line\nsecond line\n"
        assert strip_ansi(text) == text

    def test_deterministic(self):
        text = "\x1b[1;32mok\x1b[0m"
        assert strip_ansi(text) == strip_ansi(text)


# ── collapse_runs ─────────────────────────────────────────────────────


class TestCollapseRuns:
    def test_collapses_identical_run(self):
        text = "warn\nwarn\nwarn\nunique\n"
        assert collapse_runs(text) == "warn\n... (repeated 3 times)\nunique\n"

    def test_run_of_two(self):
        assert collapse_runs("a\na\nb") == "a\n... (repeated 2 times)\nb"

    def test_reconstruction_round_trip(self):
        text = "x\nx\nx\ny\nz\nz\nend\n"
        assert expand_runs(collapse_runs(text)) == text

    def test_no_op_when_no_runs(self):
        text = "alpha\nbeta\ngamma\n"
        assert collapse_runs(text) == text

    def test_no_op_on_empty(self):
        assert collapse_runs("") == ""

    def test_deterministic(self):
        text = "r\nr\ns\n"
        assert collapse_runs(text) == collapse_runs(text)


# ── fold_repeated_blocks ──────────────────────────────────────────────


BLOCK = "alpha payload line 01\nbeta payload line 02\ngamma payload line 03"


class TestFoldRepeatedBlocks:
    def test_folds_repeated_block(self):
        text = f"section one\n{BLOCK}\nseparator\nsection two\n{BLOCK}\n"
        folded = fold_repeated_blocks(text)
        assert folded != text
        assert "... (repeats 3 lines from 5 lines back)" in folded

    def test_reconstruction_round_trip(self):
        text = f"section one\n{BLOCK}\nseparator\nsection two\n{BLOCK}\n"
        assert unfold_repeated_blocks(fold_repeated_blocks(text)) == text

    def test_no_op_when_no_repeat(self):
        text = "one\ntwo\nthree\nfour\n"
        assert fold_repeated_blocks(text) == text

    def test_no_op_when_repeat_too_short(self):
        # A 2-line block only folds when the marker is smaller than the block.
        text = "a\nb\nc\na\nb\nc\n"
        assert unfold_repeated_blocks(fold_repeated_blocks(text)) == text

    def test_deterministic(self):
        text = f"head\n{BLOCK}\nmid\n{BLOCK}\n"
        assert fold_repeated_blocks(text) == fold_repeated_blocks(text)


# ── fold_path_listing ─────────────────────────────────────────────────


class TestFoldPathListing:
    def test_folds_listing(self):
        text = "src/pkg/a.py\nsrc/pkg/b.py\nlib/x.py\nlib/y.py\n"
        folded = fold_path_listing(text)
        assert folded == "src/pkg/\na.py\nb.py\nlib/\nx.py\ny.py\n"

    def test_reconstruction_round_trip(self):
        text = "src/pkg/a.py\nsrc/pkg/b.py\nlib/x.py\nlib/y.py\n"
        assert path_unheading(fold_path_listing(text)) == text

    def test_no_op_for_non_paths(self):
        text = "hello world\nfoo bar\n"
        assert fold_path_listing(text) == text

    def test_no_op_for_single_row(self):
        text = "src/pkg/a.py\n"
        assert fold_path_listing(text) == text

    def test_preserves_non_matching_lines(self):
        text = "not a path\nsrc/pkg/a.py\nsrc/pkg/b.py\n"
        folded = fold_path_listing(text)
        assert folded.startswith("not a path\n")
        assert path_unheading(folded) == text

    def test_deterministic(self):
        text = "a/b/c.txt\na/b/d.txt\n"
        assert fold_path_listing(text) == fold_path_listing(text)


# ── fold_lossless ─────────────────────────────────────────────────────


class TestFoldLossless:
    def test_composes_folds_and_round_trips(self):
        original = (
            "\x1b[32mok\x1b[0m\n"
            f"{BLOCK}\n"
            f"{BLOCK}\n"
            "src/pkg/a.py\nsrc/pkg/b.py\n"
        )
        folded = fold_lossless(original)
        assert len(folded) < len(original)
        assert "(repeats" in folded
        assert "src/pkg/" in folded
        # Inverses in reverse application order reproduce the de-ANSI'd input.
        reconstructed = expand_runs(unfold_repeated_blocks(path_unheading(folded)))
        assert reconstructed == strip_ansi(original)

    def test_collapses_runs(self):
        text = "same line here\nsame line here\nsame line here\ntail\n"
        folded = fold_lossless(text)
        assert "... (repeated 3 times)" in folded
        assert expand_runs(folded) == text

    def test_no_op_plain_text(self):
        text = "alpha beta gamma\n"
        assert fold_lossless(text) == text

    def test_fail_open_on_fold_error(self, monkeypatch):
        def _boom(text):
            raise RuntimeError("fold exploded")

        monkeypatch.setattr(lossless_mod, "collapse_runs", _boom)
        text = "a\n" * 100
        assert fold_lossless(text) == text

    def test_deterministic(self):
        text = "r\nr\nr\n" + "a\nb\nc\nd\n"
        assert fold_lossless(text) == fold_lossless(text)


# ── protection ────────────────────────────────────────────────────────


class TestProtection:
    def test_regexes_compiled(self):
        assert CRITICAL_LINE_RE.search("ERROR: boom")
        assert MUST_KEEP_RE.search("--verbose")

    @pytest.mark.parametrize(
        "line",
        [
            "Traceback (most recent call last):",
            "ERROR: something failed",
            "fatal: not a git repository",
            "AssertionError: boom",
            "CRITICAL failure",
            "do not modify this file",
            "/usr/lib/python3.so",
            "--verbose",
            "0x7fff2038",
        ],
    )
    def test_critical_lines(self, line):
        assert is_critical_line(line) is True

    @pytest.mark.parametrize("line", ["normal log line", "hello world", ""])
    def test_non_critical_lines(self, line):
        assert is_critical_line(line) is False

    def test_non_string_is_not_critical(self):
        assert is_critical_line(None) is False

    @pytest.mark.parametrize("line", ["ERROR " + "x" * 400, "Traceback (most recent call last):", "panic: runtime error"])
    def test_error_lines(self, line):
        assert is_error_line(line) is True

    @pytest.mark.parametrize("line", ["/usr/lib/" + "x" * 400, "--verbose", "do not modify this file", "normal line"])
    def test_must_keep_is_not_an_error_line(self, line):
        # Must-keep tokens protect lossy token compression (Phase 2), not
        # dense-line elision: base64 blobs contain ALLCAPS runs and are
        # safely elidable because the marker keeps the original retrievable.
        assert is_error_line(line) is False


# ── dense elision ─────────────────────────────────────────────────────


class TestDenseLines:
    def test_dense_line_detected(self):
        assert is_dense_line("x" * 300) is True

    def test_short_line_skipped(self):
        assert is_dense_line("x" * 299) is False

    def test_spaced_line_skipped(self):
        assert is_dense_line("word " * 100) is False

    def test_tab_line_skipped(self):
        assert is_dense_line("x" * 250 + "\t" + "y" * 250) is False

    def test_json_line_skipped(self):
        assert is_dense_line('{"key":"' + "x" * 400 + '"}') is False

    def test_critical_line_skipped(self):
        assert is_dense_line("ERROR " + "x" * 400) is False

    def test_must_keep_line_is_dense(self):
        # A long path/ALLCAPS line is dense and elidable when a retrieval
        # path exists; only error semantics block elision.
        assert is_dense_line("/usr/lib/" + "x" * 400) is True

    def test_base64_blob_elided_with_hint(self):
        blob = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==" * 30
        text = "header\n" + blob + "\nfooter"
        result, n = elide_dense_lines(text, retrieval_hint="/tmp/hermes-results/orig.txt")
        assert n == 1
        assert "[elided" in result
        assert "(retrieve: /tmp/hermes-results/orig.txt)" in result
        assert "header" in result and "footer" in result

    def test_elides_above_thresholds(self):
        text = "x" * 1200 + "\n" + "y" * 1200
        result, n = elide_dense_lines(text)
        assert n == 2
        assert result != text
        assert result.count("[elided") == 2
        first = result.split("\n")[0]
        assert first.startswith("x" * HEAD_CHARS)
        assert first.endswith("x" * TAIL_CHARS)

    def test_includes_retrieval_hint(self):
        text = "x" * 1200 + "\n" + "y" * 1200
        result, n = elide_dense_lines(text, retrieval_hint="/tmp/hermes-results/orig.txt")
        assert n == 2
        assert "(retrieve: /tmp/hermes-results/orig.txt)" in result

    def test_no_hint_has_no_retrieve_marker(self):
        text = "x" * 1200 + "\n" + "y" * 1200
        result, n = elide_dense_lines(text)
        assert n == 2
        assert "retrieve:" not in result

    def test_below_dense_total_unchanged(self):
        text = "x" * 1900
        assert elide_dense_lines(text) == (text, 0)

    def test_no_dense_lines_unchanged(self):
        text = "prose line\n" * 100
        assert elide_dense_lines(text) == (text, 0)

    def test_unchanged_when_hint_makes_it_larger(self):
        # 7 * 300 = 2100 dense chars clears MIN_DENSE_TOTAL_CHARS, but each
        # 775-char replacement (with the hint) exceeds its 300-char source.
        text = "\n".join("x" * 299 + chr(ord("a") + i) for i in range(7))
        result, n = elide_dense_lines(text, retrieval_hint="h" * 500)
        assert (result, n) == (text, 0)

    def test_deterministic(self):
        text = "x" * 1200 + "\n" + "y" * 1200
        assert elide_dense_lines(text) == elide_dense_lines(text)


# ── compress_tool_output ──────────────────────────────────────────────


class TestCompressToolOutput:
    def test_non_string_unchanged(self):
        result = compress_tool_output(12345)
        assert result.text == 12345
        assert result.changed is False

    def test_disabled_config_byte_identical(self, set_cc_config):
        set_cc_config(enabled=False)
        text = "same line here\n" * 20
        result = compress_tool_output(text)
        assert result.text == text
        assert result.changed is False
        assert isinstance(result, CompressionResult)

    def test_below_min_savings_unchanged(self, set_cc_config):
        set_cc_config()
        text = "tiny"
        result = compress_tool_output(text)
        assert result.text == text
        assert result.changed is False

    def test_no_savings_unchanged(self, set_cc_config):
        set_cc_config()
        text = "\n".join(f"unique log line {i:03d}" for i in range(40)) + "\n"
        result = compress_tool_output(text)
        assert result.text == text
        assert result.changed is False

    def test_lossless_only_never_loses_information(self, set_cc_config):
        set_cc_config()
        text = "this is a repeated log line\n" * 5 + "end\n"
        result = compress_tool_output(text)
        assert result.changed is True
        assert result.lossless is True
        assert result.elided_lines == 0
        assert result.new_chars < result.original_chars
        assert expand_runs(result.text) == text

    def test_lossy_without_writer_skips_elision(self, set_cc_config):
        set_cc_config(dense_line_elision=True)
        text = "x" * 1200 + "\n" + "y" * 1200
        result = compress_tool_output(text)
        assert "[elided" not in result.text
        assert result.elided_lines == 0
        assert result.lossless is True

    def test_lossy_with_writer_embeds_hint(self, set_cc_config):
        set_cc_config(dense_line_elision=True)
        text = "x" * 1200 + "\n" + "y" * 1200
        seen = []

        def _writer(original):
            seen.append(original)
            return "/sandbox/orig.txt"

        result = compress_tool_output(text, retrieval_writer=_writer)
        assert result.elided_lines == 2
        assert result.lossless is False
        assert "[elided" in result.text
        assert "(retrieve: /sandbox/orig.txt)" in result.text
        assert seen == [text]

    def test_writer_returning_none_skips_elision(self, set_cc_config):
        set_cc_config(dense_line_elision=True)
        text = "x" * 1200 + "\n" + "y" * 1200
        result = compress_tool_output(text, retrieval_writer=lambda _: None)
        assert "[elided" not in result.text
        assert result.lossless is True

    def test_writer_raising_skips_elision(self, set_cc_config):
        set_cc_config(dense_line_elision=True)

        def _writer(_):
            raise RuntimeError("sandbox down")

        text = "x" * 1200 + "\n" + "y" * 1200
        result = compress_tool_output(text, retrieval_writer=_writer)
        assert "[elided" not in result.text
        assert result.changed is False

    def test_writer_not_called_when_nothing_to_elide(self, set_cc_config):
        set_cc_config(dense_line_elision=True)
        calls = []
        text = "\n".join(f"unique log line {i:03d}" for i in range(40)) + "\n"
        compress_tool_output(text, retrieval_writer=lambda o: calls.append(o) or "/x")
        assert calls == []

    def test_fail_open_when_fold_raises(self, set_cc_config, monkeypatch):
        set_cc_config()

        def _boom(text):
            raise RuntimeError("fold exploded")

        monkeypatch.setattr(lossless_mod, "collapse_runs", _boom)
        text = "this is a repeated log line\n" * 5 + "end\n"
        result = compress_tool_output(text)
        assert result.text == text
        assert result.changed is False

    def test_min_savings_config_respected(self, set_cc_config):
        set_cc_config(min_savings_chars=1000)
        text = "this is a repeated log line\n" * 5 + "end\n"
        assert len(text) < 1000
        result = compress_tool_output(text)
        assert result.text == text


# ── maybe_persist_tool_result integration ─────────────────────────────


class TestMaybePersistIntegration:
    REPEATED = "this is a repeated log line\n" * 5 + "end\n"
    DENSE = "x" * 2500 + "\n" + "y" * 2500

    def test_lossless_compression_when_config_on(self, set_cc_config):
        set_cc_config()
        result = maybe_persist_tool_result(
            content=self.REPEATED,
            tool_name="terminal",
            tool_use_id="tc_cc",
            env=None,
            threshold=100_000,
        )
        assert result != self.REPEATED
        assert "... (repeated 5 times)" in result
        assert len(result) < len(self.REPEATED)

    def test_byte_identical_when_config_off(self, set_cc_config):
        set_cc_config(enabled=False)
        result = maybe_persist_tool_result(
            content=self.REPEATED,
            tool_name="terminal",
            tool_use_id="tc_cc",
            env=None,
            threshold=100_000,
        )
        assert result == self.REPEATED

    def test_read_file_threshold_inf_untouched(self, set_cc_config):
        set_cc_config()
        result = maybe_persist_tool_result(
            content=self.REPEATED,
            tool_name="read_file",
            tool_use_id="tc_rf",
            env=None,
            threshold=float("inf"),
        )
        assert result == self.REPEATED

    def test_lossy_path_writes_original_and_embeds_path(self, set_cc_config, tmp_path):
        set_cc_config(dense_line_elision=True)
        env = _FakeEnv(tmp_path)
        result = maybe_persist_tool_result(
            content=self.DENSE,
            tool_name="terminal",
            tool_use_id="tc_dense",
            env=env,
            threshold=100_000,
        )
        assert "[elided" in result
        expected_path = f"{env.temp_dir}/hermes-results/tc_dense.txt"
        assert f"(retrieve: {expected_path})" in result
        assert len(env.calls) == 1
        assert env.calls[0]["stdin_data"] == self.DENSE

    def test_lossy_write_failure_skips_elision(self, set_cc_config, tmp_path):
        set_cc_config(dense_line_elision=True)
        env = _FakeEnv(tmp_path, returncode=1)
        result = maybe_persist_tool_result(
            content=self.DENSE,
            tool_name="terminal",
            tool_use_id="tc_dense_fail",
            env=env,
            threshold=100_000,
        )
        assert "[elided" not in result
        assert result == self.DENSE
        assert len(env.calls) == 1

    def test_lossy_path_no_env_skips_elision(self, set_cc_config):
        set_cc_config(dense_line_elision=True)
        result = maybe_persist_tool_result(
            content=self.DENSE,
            tool_name="terminal",
            tool_use_id="tc_dense_noenv",
            env=None,
            threshold=100_000,
        )
        assert "[elided" not in result
        assert result == self.DENSE

    def test_writer_not_invoked_when_no_dense_lines(self, set_cc_config, tmp_path):
        set_cc_config(dense_line_elision=True)
        env = _FakeEnv(tmp_path)
        maybe_persist_tool_result(
            content=self.REPEATED,
            tool_name="terminal",
            tool_use_id="tc_nodense",
            env=env,
            threshold=100_000,
        )
        assert env.calls == []


# ── config loading ────────────────────────────────────────────────────


class TestConfigLoading:
    def test_defaults_without_user_config(self):
        cc_config.reset_cache()
        assert cc_config.get_content_compression_config() == DEFAULT_CONTENT_AWARE_CONFIG

    def test_user_override_loaded(self):
        import hermes_constants

        home = hermes_constants.get_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            "compression:\n"
            "  content_aware:\n"
            "    dense_line_elision: true\n"
            "    min_savings_chars: 10\n",
            encoding="utf-8",
        )
        cc_config.reset_cache()
        cfg = cc_config.get_content_compression_config()
        assert cfg["dense_line_elision"] is True
        assert cfg["min_savings_chars"] == 10
        assert cfg["enabled"] is True

    def test_load_failure_returns_defaults(self, monkeypatch):
        import hermes_cli.config as hermes_config

        def _boom():
            raise RuntimeError("config exploded")

        monkeypatch.setattr(hermes_config, "load_config", _boom)
        cc_config.reset_cache()
        assert cc_config.get_content_compression_config() == DEFAULT_CONTENT_AWARE_CONFIG

    def test_ttl_cache_and_reset(self, monkeypatch):
        import hermes_cli.config as hermes_config

        monkeypatch.setattr(
            hermes_config,
            "load_config",
            lambda: {"compression": {"content_aware": {"min_savings_chars": 10}}},
        )
        cc_config.reset_cache()
        assert cc_config.get_content_compression_config()["min_savings_chars"] == 10

        monkeypatch.setattr(
            hermes_config,
            "load_config",
            lambda: {"compression": {"content_aware": {"min_savings_chars": 20}}},
        )
        assert cc_config.get_content_compression_config()["min_savings_chars"] == 10

        cc_config.reset_cache()
        assert cc_config.get_content_compression_config()["min_savings_chars"] == 20
