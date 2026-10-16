"""Contract: no dependency gate may exceed the runtime Python floor.

Regression guard for the 3.14-gated core dependencies that made an install on
3.12 resolve with no dependencies at all and then fail to import.
``requires-python`` may declare a *lower* bridge floor (3.11, so a pre-PM
install can run ``hermes update`` once); the gates must still admit the
runtime floor the project actually supports.
"""

import re
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Lowest interpreter the full dependency set must resolve on.
RUNTIME_FLOOR = (3, 12)


def _requires_python_floor() -> tuple[int, int]:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^requires-python\s*=\s*">=(\d+)\.(\d+)', text, re.M)
    assert match, "requires-python missing from pyproject.toml"
    return int(match.group(1)), int(match.group(2))


def _dependency_gates() -> set[tuple[int, int]]:
    text = PYPROJECT.read_text(encoding="utf-8")
    return {
        (int(major), int(minor))
        for major, minor in re.findall(r"python_version\s*>=\s*'(\d+)\.(\d+)'", text)
    }


def test_dependency_markers_admit_the_runtime_floor():
    gates = _dependency_gates()
    assert gates, "expected at least one python_version gate on the core dependencies"
    too_high = sorted(gate for gate in gates if gate > RUNTIME_FLOOR)
    assert not too_high, (
        f"dependency gates {too_high} exceed the runtime floor {RUNTIME_FLOOR}; "
        "an install on that interpreter would resolve without them"
    )


def test_requires_python_floor_does_not_exceed_the_runtime_floor():
    floor = _requires_python_floor()
    assert floor <= RUNTIME_FLOOR, (
        f"requires-python floor {floor} exceeds the supported runtime floor {RUNTIME_FLOOR}"
    )
