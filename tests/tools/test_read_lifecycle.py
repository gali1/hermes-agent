"""Tests for agent/content_compression/read_lifecycle.py (Phase 3a).

Hermetic: pure functions only, no network, no HERMES_HOME writes.
"""

from agent.content_compression.read_lifecycle import (
    READ_TOOL_NAMES,
    WRITE_TOOL_HINTS,
    apply_read_lifecycle,
    extract_path,
)


def _call(call_id, name, args):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
        ],
    }


def _result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ObjectCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _Fn(name, arguments)


STALE = (
    "[read_file] {path} — stale: modified later by {tool}; "
    "re-read for current contents"
)
SUPERSEDED = "[read_file] {path} — superseded by a newer read of the same file"


class TestExtractPath:
    def test_dict_path(self):
        assert extract_path({"path": "a.py"}) == "a.py"

    def test_json_string(self):
        assert extract_path('{"path": "a.py"}') == "a.py"

    def test_key_variants_case_insensitive(self):
        assert extract_path({"file_path": "b.py"}) == "b.py"
        assert extract_path({"Filepath": "c.py"}) == "c.py"
        assert extract_path({"FILENAME": "d.py"}) == "d.py"
        assert extract_path({"file": "e.py"}) == "e.py"

    def test_first_path_key_wins(self):
        assert extract_path({"file_path": "first.py", "path": "second.py"}) == "first.py"

    def test_files_list(self):
        assert extract_path({"files": [{"path": "f.py"}, {"path": "g.py"}]}) == "f.py"
        assert extract_path({"files": [{"nope": 1}, {"file_path": "g.py"}]}) == "g.py"

    def test_invalid_inputs(self):
        assert extract_path(None) is None
        assert extract_path("not json") is None
        assert extract_path({"nope": 1}) is None
        assert extract_path({"path": 123}) is None
        assert extract_path({"path": "   "}) is None
        assert extract_path({"files": "not-a-list"}) is None


class TestWriteToolHints:
    def test_constants(self):
        assert READ_TOOL_NAMES == frozenset({"read_file", "read"})
        assert "write" in WRITE_TOOL_HINTS and "patch" in WRITE_TOOL_HINTS

    def test_read_tools_never_writeish(self):
        from agent.content_compression.read_lifecycle import _is_write_tool

        assert _is_write_tool("read_file") is False
        assert _is_write_tool("read") is False
        assert _is_write_tool("read_write_file") is False

    def test_write_tools_detected(self):
        from agent.content_compression.read_lifecycle import _is_write_tool

        assert _is_write_tool("write_file") is True
        assert _is_write_tool("apply_patch") is True
        assert _is_write_tool("edit_file") is True
        assert _is_write_tool("delete_path") is True
        assert _is_write_tool("terminal") is False
        assert _is_write_tool("") is False


class TestStaleReads:
    def test_read_then_write_replaced(self):
        messages = [
            {"role": "user", "content": "read then edit"},
            _call("r1", "read_file", '{"path": "a.py"}'),
            _result("r1", "line1\nline2\n" * 100),
            _call("w1", "write_file", '{"path": "a.py", "content": "new"}'),
            _result("w1", '{"ok": true}'),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 1
        assert result[2]["content"] == STALE.format(path="a.py", tool="write_file")
        assert result[0] == messages[0]
        assert result[4] == messages[4]

    def test_stale_takes_precedence_over_superseded(self):
        messages = [
            _call("r1", "read_file", '{"path": "a.py"}'),
            _result("r1", "old"),
            _call("w1", "patch", '{"path": "a.py"}'),
            _result("w1", "patched"),
            _call("r2", "read_file", '{"path": "a.py"}'),
            _result("r2", "new"),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 1
        assert result[1]["content"] == STALE.format(path="a.py", tool="patch")

    def test_path_key_variants_match(self):
        messages = [
            _call("r1", "read_file", '{"file_path": "a.py"}'),
            _result("r1", "old"),
            _call("w1", "write_file", '{"Filename": "a.py"}'),
            _result("w1", "ok"),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 1
        assert "stale" in result[1]["content"]

    def test_earliest_later_write_names_marker(self):
        messages = [
            _call("r1", "read_file", '{"path": "a.py"}'),
            _result("r1", "old"),
            _call("w1", "patch", '{"path": "a.py"}'),
            _result("w1", "ok"),
            _call("w2", "write_file", '{"path": "a.py"}'),
            _result("w2", "ok"),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 1
        assert "modified later by patch" in result[1]["content"]

    def test_object_shaped_tool_calls(self):
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_ObjectCall("r1", "read_file", '{"path": "a.py"}')],
            },
            _result("r1", "old"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_ObjectCall("w1", "write_file", '{"path": "a.py"}')],
            },
            _result("w1", "ok"),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 1
        assert "stale" in result[2]["content"]


class TestSupersededReads:
    def test_older_read_replaced_newest_kept(self):
        messages = [
            _call("r1", "read_file", '{"path": "b.py"}'),
            _result("r1", "old content " * 50),
            _call("r2", "read_file", '{"path": "b.py"}'),
            _result("r2", "new content " * 50),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 1
        assert result[1]["content"] == SUPERSEDED.format(path="b.py")
        assert result[3]["content"] == "new content " * 50

    def test_three_reads_only_newest_kept(self):
        messages = [
            _call("r1", "read_file", '{"path": "b.py"}'),
            _result("r1", "v1"),
            _call("r2", "read_file", '{"path": "b.py"}'),
            _result("r2", "v2"),
            _call("r3", "read_file", '{"path": "b.py"}'),
            _result("r3", "v3"),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 2
        assert result[1]["content"] == SUPERSEDED.format(path="b.py")
        assert result[3]["content"] == SUPERSEDED.format(path="b.py")
        assert result[5]["content"] == "v3"

    def test_different_paths_not_superseded(self):
        messages = [
            _call("r1", "read_file", '{"path": "a.py"}'),
            _result("r1", "a"),
            _call("r2", "read_file", '{"path": "b.py"}'),
            _result("r2", "b"),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 0
        assert result == messages


class TestBoundary:
    def test_tail_never_touched(self):
        messages = [
            _call("r1", "read_file", '{"path": "b.py"}'),
            _result("r1", "old content " * 50),
            _call("r2", "read_file", '{"path": "b.py"}'),
            _result("r2", "new content " * 50),
        ]
        result, replaced = apply_read_lifecycle(messages, 4)
        assert replaced == 1
        assert result[3]["content"] == "new content " * 50

    def test_boundary_at_or_before_index_protects_it(self):
        messages = [
            _call("r1", "read_file", '{"path": "b.py"}'),
            _result("r1", "old content " * 50),
            _call("r2", "read_file", '{"path": "b.py"}'),
            _result("r2", "new content " * 50),
        ]
        result, replaced = apply_read_lifecycle(messages, 1)
        assert replaced == 0
        assert result == messages

    def test_zero_and_negative_boundary_noop(self):
        messages = [
            _call("r1", "read_file", '{"path": "b.py"}'),
            _result("r1", "old"),
            _call("r2", "read_file", '{"path": "b.py"}'),
            _result("r2", "new"),
        ]
        assert apply_read_lifecycle(messages, 0) == (messages, 0)
        assert apply_read_lifecycle(messages, -3) == (messages, 0)


class TestSkips:
    def test_non_string_empty_and_marker_contents_skipped(self):
        messages = [
            _call("r1", "read_file", '{"path": "c.py"}'),
            _result("r1", [{"type": "text", "text": "multimodal"}]),
            _call("r2", "read_file", '{"path": "c.py"}'),
            _result("r2", ""),
            _call("r3", "read_file", '{"path": "c.py"}'),
            _result("r3", "[read_file] c.py — superseded by a newer read of the same file"),
            _call("r4", "read_file", '{"path": "c.py"}'),
            _result("r4", "[Duplicate tool output — same content as a more recent call]"),
            _call("r5", "read_file", '{"path": "c.py"}'),
            _result("r5", "newest"),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 0
        assert result == messages

    def test_read_without_path_skipped(self):
        messages = [
            _call("r1", "read_file", "{}"),
            _result("r1", "old"),
            _call("r2", "read_file", '{"path": "c.py"}'),
            _result("r2", "new"),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 0

    def test_tool_message_without_matching_call_skipped(self):
        messages = [
            {"role": "tool", "tool_call_id": "orphan", "content": "old"},
            _call("r2", "read_file", '{"path": "c.py"}'),
            _result("r2", "new"),
        ]
        result, replaced = apply_read_lifecycle(messages, len(messages))
        assert replaced == 0


class TestFailOpen:
    def test_bad_messages_returned_unchanged(self):
        messages = [{"role": "assistant", "tool_calls": 5}]
        result, replaced = apply_read_lifecycle(messages, 5)
        assert result is messages
        assert replaced == 0

    def test_none_messages(self):
        assert apply_read_lifecycle(None, 5) == (None, 0)

    def test_empty_messages(self):
        assert apply_read_lifecycle([], 5) == ([], 0)

    def test_deterministic(self):
        messages = [
            _call("r1", "read_file", '{"path": "a.py"}'),
            _result("r1", "old"),
            _call("w1", "write_file", '{"path": "a.py"}'),
            _result("w1", "ok"),
        ]
        first = apply_read_lifecycle(messages, len(messages))
        second = apply_read_lifecycle(messages, len(messages))
        assert first == second
