"""Tests for agent/content_compression/tokens.py (Phase 3b).

Hermetic: pure functions only, no network, no HERMES_HOME writes.
"""

import json

from agent.content_compression.tokens import (
    estimate_messages_tokens_structured,
    estimate_tokens_structured,
)


class TestEstimateTokensStructured:
    def test_empty_and_non_string(self):
        assert estimate_tokens_structured("") == 0
        assert estimate_tokens_structured(None) == 0
        assert estimate_tokens_structured(123) == 0

    def test_non_empty_always_at_least_one(self):
        for text in ["a", "{", " ", "1", "\n"]:
            assert estimate_tokens_structured(text) >= 1

    def test_json_within_structural_bounds(self):
        payload = json.dumps(
            [
                {"id": i, "name": f"item-{i}", "active": i % 2 == 0, "score": i * 1.5}
                for i in range(40)
            ],
            indent=2,
        )
        estimate = estimate_tokens_structured(payload)
        assert len(payload) // 6 <= estimate <= len(payload) // 3

    def test_punctuation_dense_json_beats_chars_over_four(self):
        payload = json.dumps(
            {f"key_{i}": i for i in range(200)}, separators=(",", ":")
        )
        estimate = estimate_tokens_structured(payload)
        assert estimate > len(payload) // 4
        assert estimate <= len(payload) // 3

    def test_json_scalars(self):
        assert estimate_tokens_structured("true") == 1
        assert estimate_tokens_structured("null") == 1
        assert estimate_tokens_structured('"hello world"') == 2

    def test_prose_within_word_bounds(self):
        prose = "The quick brown fox jumps over the lazy dog. " * 30
        estimate = estimate_tokens_structured(prose)
        assert len(prose) // 5 <= estimate <= len(prose) // 3

    def test_code_is_sane(self):
        code = "def add(a, b):\n    return a + b\n\n" * 20
        estimate = estimate_tokens_structured(code)
        assert 0 < estimate <= len(code)
        assert estimate >= len(code) // 6

    def test_plain_identifier_heavy_code(self):
        code = "\n".join(f"value_{i} = compute_{i}(arg_{i})" for i in range(50))
        estimate = estimate_tokens_structured(code)
        assert 0 < estimate <= len(code)

    def test_deterministic(self):
        for text in ["plain prose here", '{"a": [1, 2, 3]}', "x = (a + b) * c;"]:
            assert estimate_tokens_structured(text) == estimate_tokens_structured(text)


class TestEstimateMessagesTokensStructured:
    def test_sums_string_contents(self):
        messages = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": "read_file output"},
        ]
        expected = estimate_tokens_structured("hello world") + estimate_tokens_structured(
            "read_file output"
        )
        assert estimate_messages_tokens_structured(messages) == expected

    def test_skips_multimodal_and_non_dict(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "x" * 100}]},
            {"role": "user", "content": {"_multimodal": True, "content": []}},
            "not a dict",
            {"role": "user", "content": "kept"},
        ]
        assert estimate_messages_tokens_structured(messages) == estimate_tokens_structured(
            "kept"
        )

    def test_empty_inputs(self):
        assert estimate_messages_tokens_structured([]) == 0
        assert estimate_messages_tokens_structured(None) == 0

    def test_missing_content(self):
        assert estimate_messages_tokens_structured([{"role": "user"}]) == 0
