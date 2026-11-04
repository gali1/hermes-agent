"""TASK-2 (anytool): reliability-aware re-ranking + bounded zero-match LLM fallback.

Invariants under test:

* reliability scoring is pure/deterministic, neutral under 3 calls and at a >=40%
  success rate, and always inside the 0.2..1.0 band;
* re-ranking happens AFTER admission, so it is a permutation: same candidate set,
  same result count, only the order changes;
* the zero-match pass runs only on an empty lexical result, sends a capped catalog
  listing, makes one attempt under a hard timeout, filters proposals against the
  real corpus, and returns nothing on any failure;
* both keys are off by default and a disabled key is a strict no-op.
"""

import json
import sqlite3
import time
from types import SimpleNamespace

import pytest


def _td(name, desc="", props=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props or {},
                "required": required or [],
            },
        },
    }


def _entry(name, desc="", toolset="mcp-anytool-test"):
    """A CatalogEntry whose BM25 tokens are built exactly as build_catalog does."""
    from tools.tool_search_catalog import CatalogEntry, _entry_search_text, _tokenize

    td = _td(name, desc)
    entry = CatalogEntry(name=name, description=desc, schema=td, source="mcp",
                         source_name=toolset)
    entry._tokens = _tokenize(_entry_search_text(td, "anytool-test"))
    return entry


def _register(names, toolset, desc="Look up a widget by id"):
    """Register MCP tools in the live registry and return their tool-defs."""
    from tools.registry import registry

    def _handler(args, **kwargs):
        return json.dumps({"ok": True})

    defs = []
    for name in names:
        td = _td(name, desc)
        registry.register(name=name, handler=_handler, schema=td, toolset=toolset)
        defs.append(td)
    return defs


def _reply(text):
    """A minimal auxiliary-response stand-in (``choices[0].message.content``)."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


# ---------------------------------------------------------------------------
# 1. Reliability scoring
# ---------------------------------------------------------------------------


class TestReliabilityMultiplier:
    def test_neutral_under_three_calls(self):
        from tools.tool_search_reliability import MAX_MULTIPLIER, reliability_multiplier

        # Two failures on two calls is not evidence: the band needs 3+ calls.
        assert reliability_multiplier(0, 0) == MAX_MULTIPLIER
        assert reliability_multiplier(1, 0) == MAX_MULTIPLIER
        assert reliability_multiplier(2, 0) == MAX_MULTIPLIER
        assert reliability_multiplier(2, 2) == MAX_MULTIPLIER

    def test_neutral_at_or_above_forty_percent(self):
        from tools.tool_search_reliability import MAX_MULTIPLIER, reliability_multiplier

        assert reliability_multiplier(3, 2) == MAX_MULTIPLIER     # 66%
        assert reliability_multiplier(10, 4) == MAX_MULTIPLIER    # exactly 40%
        assert reliability_multiplier(100, 95) == MAX_MULTIPLIER

    def test_floor_at_zero_success_and_band_width(self):
        from tools.tool_search_reliability import MIN_MULTIPLIER, reliability_multiplier

        assert reliability_multiplier(10, 0) == MIN_MULTIPLIER
        assert reliability_multiplier(3, 0) == MIN_MULTIPLIER
        for calls in (3, 4, 7, 25, 500):
            for ok in range(calls + 1):
                value = reliability_multiplier(calls, ok)
                assert 0.2 <= value <= 1.0, (calls, ok, value)

    def test_monotonic_in_success_rate(self):
        from tools.tool_search_reliability import reliability_multiplier

        values = [reliability_multiplier(10, ok) for ok in range(11)]
        assert values == sorted(values)
        assert values[0] < values[1] < values[2]  # strictly improving below neutral

    def test_scoring_reads_no_clock(self, monkeypatch):
        """Purity: the same counts answer the same under a booby-trapped clock."""
        import tools.tool_search_reliability as rel

        monkeypatch.setattr(rel.time, "time", lambda: (_ for _ in ()).throw(
            AssertionError("scoring must not read the clock")))
        assert rel.reliability_multiplier(9, 1) == rel.reliability_multiplier(9, 1)
        scored = [(1.0, _entry("anytool_clock_probe", "widget lookup"))]
        assert rel.rerank_scored(scored, {"anytool_clock_probe": 0.5}) == scored

    def test_multiplier_is_deterministic_for_a_history(self):
        from tools.tool_search_reliability import multipliers_from_counts

        counts = {"a": {"calls": 9, "ok": 2}, "b": {"calls": 4, "ok": 4},
                  "c": {"calls": 12, "ok": 0}}
        assert multipliers_from_counts(counts) == multipliers_from_counts(dict(counts))

    def test_neutral_tools_are_omitted_from_the_mapping(self):
        from tools.tool_search_reliability import multipliers_from_counts

        mapping = multipliers_from_counts({
            "solid": {"calls": 20, "ok": 20},
            "thin": {"calls": 2, "ok": 0},
            "broken": {"calls": 10, "ok": 0},
        })
        assert "solid" not in mapping and "thin" not in mapping
        assert mapping["broken"] == 0.2


# ---------------------------------------------------------------------------
# 2. Re-ranking: ordering changes, set does not
# ---------------------------------------------------------------------------


class TestRerank:
    def _twins(self):
        # Identical descriptions -> identical BM25 scores -> a stable tie.
        return [_entry("anytool_alpha_widget_lookup", "Look up a widget by id"),
                _entry("anytool_beta_widget_lookup", "Look up a widget by id")]

    def test_ordering_changes_but_the_set_does_not(self):
        from tools.tool_search_catalog import search_catalog

        catalog = self._twins()
        baseline = [e.name for e in search_catalog(catalog, "widget lookup", limit=5)]
        reranked = [e.name for e in search_catalog(
            catalog, "widget lookup", limit=5,
            multipliers={"anytool_alpha_widget_lookup": 0.2})]

        assert baseline == ["anytool_alpha_widget_lookup", "anytool_beta_widget_lookup"]
        assert reranked[0] == "anytool_beta_widget_lookup"      # order moved
        assert set(reranked) == set(baseline)                    # nothing hidden
        assert len(reranked) == len(baseline)                    # nothing dropped

    def test_demotion_lands_at_its_adjusted_score(self):
        """The multiplier scales the score: the demoted hit sits exactly where s*0.2 ranks."""
        from tools.tool_search_reliability import rerank_scored

        strong = _entry("anytool_widget_lookup", "Look up a widget by id")
        middle = _entry("anytool_widget_middle", "Look up a widget")
        floor = _entry("anytool_widget_floor", "Look up a widget")
        out = rerank_scored([(1.0, strong), (0.5, middle), (0.15, floor)],
                            {"anytool_widget_lookup": 0.2})
        # 1.0*0.2 = 0.2, which is still above the 0.15 candidate and below 0.5.
        assert [e.name for _, e in out] == [
            "anytool_widget_middle", "anytool_widget_lookup", "anytool_widget_floor"]
        assert len(out) == 3

    def test_exact_name_match_stays_first_under_a_penalty(self):
        from tools.tool_search_catalog import search_catalog

        catalog = self._twins() + [_entry("anytool_other", "Something else entirely")]
        hits = [e.name for e in search_catalog(
            catalog, "anytool_beta_widget_lookup", limit=1,
            multipliers={"anytool_beta_widget_lookup": 0.2})]
        assert hits == ["anytool_beta_widget_lookup"]

    def test_no_multipliers_is_byte_identical(self):
        from tools.tool_search_catalog import search_catalog

        catalog = self._twins()
        assert (search_catalog(catalog, "widget lookup", limit=2)
                == search_catalog(catalog, "widget lookup", limit=2, multipliers=None)
                == search_catalog(catalog, "widget lookup", limit=2, multipliers={}))

    def test_rerank_scored_is_a_pure_permutation(self):
        from tools.tool_search_reliability import rerank_scored

        alpha, beta = self._twins()
        scored = [(2.0, alpha), (1.0, beta)]
        snapshot = list(scored)
        out = rerank_scored(scored, {"anytool_beta_widget_lookup": 1.0,
                                     "anytool_alpha_widget_lookup": 0.2})
        assert scored == snapshot                     # input untouched
        assert sorted(id(e) for _, e in out) == sorted(id(e) for _, e in snapshot)
        assert len(out) == len(snapshot)

    def test_low_reliability_never_removes_a_hit_from_the_pool(self):
        from tools.tool_search_reliability import rerank_scored

        entries = [_entry(f"anytool_pool_{index}", "widget lookup tool") for index in range(6)]
        scored = [(1.0, entry) for entry in entries]
        out = rerank_scored(scored, {entry.name: 0.2 for entry in entries})
        assert {e.name for _, e in out} == {e.name for e in entries}
        assert len(out) == 6


# ---------------------------------------------------------------------------
# 3. Outcome window (newest 8000 rows / 30 days)
# ---------------------------------------------------------------------------


def _make_state_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
        "role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, "
        "timestamp REAL NOT NULL)")
    conn.commit()
    return conn


class TestOutcomeWindow:
    def test_window_defaults_are_the_contract(self):
        from tools.tool_search_reliability import WINDOW_DAYS, WINDOW_ROWS

        assert WINDOW_ROWS == 8000
        assert WINDOW_DAYS == 30.0

    def test_counts_only_tool_rows_inside_the_window(self, tmp_path):
        from tools.tool_search_reliability import load_outcome_counts

        db = tmp_path / "state.db"
        conn = _make_state_db(db)
        now = 1_800_000_000.0
        day = 86400.0
        rows = [
            ("ok_tool", json.dumps({"ok": True}), now - day),
            ("ok_tool", json.dumps({"ok": True}), now - 2 * day),
            ("bad_tool", json.dumps({"error": "boom"}), now - day),
            ("bad_tool", "Error: nope", now - 3 * day),
            ("guard_tool", json.dumps({"error": "duplicate", "guardrail_refusal": True}), now),
            ("old_tool", json.dumps({"ok": True}), now - 31 * day),  # outside 30 days
        ]
        for name, content, ts in rows:
            conn.execute("INSERT INTO messages (session_id, role, content, tool_name, timestamp) "
                         "VALUES ('s', 'tool', ?, ?, ?)", (content, name, ts))
        conn.execute("INSERT INTO messages (session_id, role, content, tool_name, timestamp) "
                     "VALUES ('s', 'assistant', 'text', 'not_a_tool', ?)", (now,))
        conn.execute("INSERT INTO messages (session_id, role, content, tool_name, timestamp) "
                     "VALUES ('s', 'tool', '{}', '', ?)", (now,))
        conn.commit()
        conn.close()

        counts = load_outcome_counts(db_path=db, now=now)
        assert counts["ok_tool"] == {"calls": 2, "ok": 2, "fail": 0}
        assert counts["bad_tool"] == {"calls": 2, "ok": 0, "fail": 2}
        # A guardrail refusal is not a failure and is not counted at all.
        assert "guard_tool" not in counts
        assert "old_tool" not in counts
        assert "not_a_tool" not in counts and "" not in counts

    def test_newest_row_cap_is_applied(self, tmp_path):
        from tools.tool_search_reliability import load_outcome_counts

        db = tmp_path / "state.db"
        conn = _make_state_db(db)
        now = 1_800_000_000.0
        for index in range(6):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_name, timestamp) "
                "VALUES ('s', 'tool', ?, ?, ?)",
                (json.dumps({"error": "x"}), f"tool_{index}", now - index))
        conn.commit()
        conn.close()

        counts = load_outcome_counts(db_path=db, now=now, max_rows=3)
        # Newest three by id DESC: the last three inserts (tool_5, tool_4, tool_3).
        assert set(counts) == {"tool_3", "tool_4", "tool_5"}

    def test_missing_or_corrupt_db_is_no_signal(self, tmp_path):
        from tools.tool_search_reliability import load_outcome_counts

        assert load_outcome_counts(db_path=tmp_path / "absent.db", now=1.0) == {}

        broken = tmp_path / "broken.db"
        broken.write_text("not a database", encoding="utf-8")
        assert load_outcome_counts(db_path=broken, now=1.0) == {}

    def test_schema_without_messages_table_is_no_signal(self, tmp_path):
        from tools.tool_search_reliability import load_outcome_counts

        db = tmp_path / "empty.db"
        sqlite3.connect(db).close()
        assert load_outcome_counts(db_path=db, now=1.0) == {}

    def test_truncated_envelope_is_no_evidence(self):
        """A prefix cut mid-JSON must not be guessed at in either direction."""
        from tools.tool_search_reliability import counts_from_rows

        counts = counts_from_rows([
            ("tool_a", '{"ok": true, "blob": "xxxxxxxx'),   # truncated -> skipped
            ("tool_a", "plain successful output"),           # counted as success
        ])
        assert counts["tool_a"] == {"calls": 1, "ok": 1, "fail": 0}


# ---------------------------------------------------------------------------
# 4. Zero-match LLM fallback
# ---------------------------------------------------------------------------


class TestZeroMatchFallback:
    NAMES = ("anytool_zebra_sync_files", "anytool_zebra_report")

    @pytest.fixture
    def defs(self):
        return _register(self.NAMES, "mcp-anytool-zeromatch")

    @staticmethod
    def _config(**overrides):
        from tools.tool_search import ToolSearchConfig

        raw = {"enabled": "on", "zero_match_llm": True}
        raw.update(overrides)
        return ToolSearchConfig.from_raw(raw)

    def _dispatch(self, defs, proposer, config=None, query="zebraherd quux"):
        from tools.tool_search import dispatch_tool_search

        return json.loads(dispatch_tool_search(
            {"queries": [query]}, current_tool_defs=defs,
            config=config or self._config(), zero_match_proposer=proposer))

    def test_disabled_is_a_strict_no_op(self, defs):
        """Off by default: the proposer is never called, even on an empty result."""
        from tools.tool_search import dispatch_tool_search

        def _explode(messages, timeout):
            raise AssertionError("zero-match model pass must not run when disabled")

        result = json.loads(dispatch_tool_search(
            {"queries": ["zebraherd quux"]}, current_tool_defs=defs,
            config=self._config(zero_match_llm=False), zero_match_proposer=_explode))
        assert result["results"][0]["matches"] == []

    def test_falls_back_on_empty_lexical_result(self, defs):
        calls = []

        def _proposer(messages, timeout):
            calls.append((messages, timeout))
            return _reply(json.dumps(["anytool_zebra_sync_files"]))

        result = self._dispatch(defs, _proposer)
        group = result["results"][0]

        assert len(calls) == 1
        assert group["matches"] == ["anytool_zebra_sync_files"]
        assert result["tools"]["anytool_zebra_sync_files"]["source"] == "mcp"
        # The prompt carries names + one-line descriptions, never schemas.
        prompt = calls[0][0][-1]["content"]
        assert "anytool_zebra_sync_files" in prompt
        assert "parameters" not in prompt

    def test_hallucinated_names_are_dropped(self, defs):
        def _proposer(messages, timeout):
            return _reply(json.dumps(
                ["anytool_does_not_exist", "anytool_zebra_report", "terminal"]))

        result = self._dispatch(defs, _proposer)
        assert result["results"][0]["matches"] == ["anytool_zebra_report"]
        assert "anytool_does_not_exist" not in result["tools"]

    def test_all_hallucinated_keeps_the_empty_result(self, defs):
        def _proposer(messages, timeout):
            return _reply('["ghost_one", "ghost_two"]')

        result = self._dispatch(defs, _proposer)
        group = result["results"][0]
        assert group["matches"] == []
        assert result["tools"] == {}
        # The pre-existing empty-result hint still applies.
        assert "available_sources" in group

    def test_model_failure_is_silent(self, defs):
        def _proposer(messages, timeout):
            raise RuntimeError("provider exploded")

        result = self._dispatch(defs, _proposer)
        assert result["results"][0]["matches"] == []

    def test_unparsable_answer_is_silent(self, defs):
        result = self._dispatch(defs, lambda messages, timeout: _reply(""))
        assert result["results"][0]["matches"] == []

        result = self._dispatch(
            defs, lambda messages, timeout: SimpleNamespace(choices=[]))
        assert result["results"][0]["matches"] == []

    def test_timeout_is_enforced_and_bounded(self, defs):
        def _slow(messages, timeout):
            time.sleep(0.6)
            return _reply('["anytool_zebra_report"]')

        started = time.monotonic()
        result = self._dispatch(defs, _slow,
                                config=self._config(zero_match_llm_timeout=0.05))
        elapsed = time.monotonic() - started

        assert result["results"][0]["matches"] == []
        assert elapsed < 0.5, f"fallback exceeded its wall-clock cap: {elapsed:.2f}s"

    def test_not_called_when_lexical_search_matched(self, defs):
        def _explode(messages, timeout):
            raise AssertionError("fallback must only run on an empty lexical result")

        result = self._dispatch(defs, _explode, query="zebra report")
        assert result["results"][0]["matches"] == ["anytool_zebra_report"]

    def test_pass_count_per_call_is_bounded(self, defs):
        """A batch of empty queries must not multiply the fallback cost."""
        from tools.tool_search import _MAX_ZERO_MATCH_PASSES, dispatch_tool_search

        calls = []

        def _proposer(messages, timeout):
            calls.append(messages[-1]["content"])
            return _reply("[]")

        queries = [f"zebraherd quux {index}" for index in range(7)]
        json.loads(dispatch_tool_search(
            {"queries": queries}, current_tool_defs=defs,
            config=self._config(), zero_match_proposer=_proposer))
        assert len(calls) == _MAX_ZERO_MATCH_PASSES == 3

    def test_catalog_listing_is_capped(self):
        from tools.tool_search_zero_match import (
            MAX_CATALOG_CHARS, MAX_CATALOG_ENTRIES, catalog_listing)

        entries = [_entry(f"anytool_bulk_{index}", "x" * 400) for index in range(500)]
        listing = catalog_listing(entries)
        lines = [line for line in listing.splitlines() if line.strip()]

        assert len(lines) <= MAX_CATALOG_ENTRIES
        assert len(listing) <= MAX_CATALOG_CHARS

    def test_parse_proposals_is_tolerant(self):
        from tools.tool_search_zero_match import parse_proposals

        assert parse_proposals('```json\n["a_tool", "b_tool"]\n```') == ["a_tool", "b_tool"]
        assert parse_proposals('[{"name": "c_tool"}]') == ["c_tool"]
        assert parse_proposals("d_tool\ne_tool") == ["d_tool", "e_tool"]
        assert parse_proposals("") == [] and parse_proposals(None) == []

    def test_proposals_are_capped(self):
        from tools.tool_search_zero_match import MAX_PROPOSALS, parse_proposals

        text = json.dumps([f"tool_{index}" for index in range(50)])
        assert len(parse_proposals(text)) == MAX_PROPOSALS


# ---------------------------------------------------------------------------
# 5. Dispatch wiring + config defaults
# ---------------------------------------------------------------------------


class TestDispatchWiring:
    def _defs(self):
        return _register(("anytool_wire_alpha", "anytool_wire_beta"),
                         "mcp-anytool-wiring")

    def test_rerank_is_wired_through_dispatch(self, monkeypatch):
        import tools.tool_search_reliability as rel
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search

        defs = self._defs()

        def _names(raw):
            out = json.loads(dispatch_tool_search(
                {"queries": ["widget"]}, current_tool_defs=defs,
                config=ToolSearchConfig.from_raw(raw)))
            return out["results"][0]["matches"]

        baseline = _names({})
        assert len(baseline) == 2, baseline
        top, other = baseline[0], baseline[1]
        monkeypatch.setattr(rel, "load_reliability_multipliers",
                            lambda **kwargs: {top: 0.2})
        reranked = _names({"reliability_rerank": True})

        assert reranked[0] == other          # the discounted hit moved down
        assert set(reranked) == set(baseline)  # and nothing was lost

    def test_disabled_rerank_never_touches_the_store(self, monkeypatch):
        import tools.tool_search_reliability as rel
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search

        def _explode(**kwargs):
            raise AssertionError("reliability store must not be read when disabled")

        monkeypatch.setattr(rel, "load_reliability_multipliers", _explode)
        out = json.loads(dispatch_tool_search(
            {"queries": ["widget"]}, current_tool_defs=self._defs(),
            config=ToolSearchConfig.from_raw({})))
        assert out["results"][0]["matches"]

    def test_default_config_documents_both_keys_off(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        from tools.tool_search import ToolSearchConfig

        raw = DEFAULT_CONFIG["tools"]["tool_search"]
        assert raw["reliability_rerank"] is False
        assert raw["zero_match_llm"] is False
        assert raw["zero_match_llm_timeout"] == 4.0

        config = ToolSearchConfig.from_raw({})
        assert config.reliability_rerank is False and config.zero_match_llm is False

    def test_flags_accept_only_explicit_true(self):
        from tools.tool_search import ToolSearchConfig

        assert ToolSearchConfig.from_raw({"zero_match_llm": "on"}).zero_match_llm is True
        assert ToolSearchConfig.from_raw({"zero_match_llm": True}).zero_match_llm is True
        for junk in ("off", "maybe", 0, "", None, [], {}):
            assert ToolSearchConfig.from_raw({"zero_match_llm": junk}).zero_match_llm is False
        config = ToolSearchConfig.from_raw({"reliability_rerank": "yes",
                                            "zero_match_llm_timeout": 99})
        assert config.reliability_rerank is True
        assert config.zero_match_llm_timeout == 15.0
