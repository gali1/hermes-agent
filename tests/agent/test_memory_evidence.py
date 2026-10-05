"""Unit tests for the evidence quality and confidence primitives."""

from agent.memory.evidence import SOURCE_QUALITY, confidence, source_quality


def test_source_quality():
    assert source_quality("user") == 0.9
    assert source_quality("conversation") == 0.7
    assert source_quality("tool") == 0.6
    assert source_quality("file") == 0.8
    assert source_quality("git") == 0.8
    assert source_quality("agent") == 0.5
    assert source_quality("memory") == 0.6
    assert source_quality("inferred") == 0.3
    assert source_quality("graph") == 0.3
    assert source_quality("FILE") == 0.8
    assert source_quality("unknown-kind") == 0.5
    assert source_quality(None) == 0.5
    assert source_quality("") == 0.5
    assert all(0.0 <= value <= 1.0 for value in SOURCE_QUALITY.values())


def test_confidence():
    assert abs(confidence() - 0.5) < 1e-9
    assert confidence(proof_count=50, source_quality_value=0.9, temporal_consistency=0.9) > \
        confidence(proof_count=1, source_quality_value=0.3, temporal_consistency=0.3)
    assert confidence(proof_count=1000, source_quality_value=1.0, temporal_consistency=1.0) <= 0.95
    assert confidence(proof_count=1, source_quality_value=0.0, contradiction_penalty=5.0) == 0.0
    assert 0.0 <= confidence(proof_count=0, source_quality_value=0.0,
                             contradiction_penalty=0.5, temporal_consistency=0.0) <= 0.95
    assert confidence(proof_count=5, source_quality_value=0.7) > confidence(proof_count=1, source_quality_value=0.7)
