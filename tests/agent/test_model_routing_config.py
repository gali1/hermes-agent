"""The shared ``smart_model_routing`` config surface (M1).

Behaviour contracts, not snapshots:

* a config file with **no** routing block loads unchanged and reads as
  "disabled" — absence is the shipped state, so nothing may materialise it;
* every read goes through :func:`agent.model_router.routing_settings`, every
  write through :func:`update_routing`, every check through
  :func:`validate_routing`, so the CLI, the wizard, the gateway and
  ``validate_config_structure`` cannot disagree about the schema;
* an invalid block is rejected *before* anything is written, and a write that
  fails leaves the previous file byte-identical;
* a successful write re-serialises the file through ``save_config``: comments,
  env-var templates and every unrelated key survive, and the caller's reported
  state is the one read back **from disk**.

Every test runs against a scratch ``HERMES_HOME`` under the mission scratch dir
and removes it afterwards — the live ``~/.hermes/config.yaml`` is never touched.
"""

from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent import model_router as mr  # noqa: E402

_CONFIG = """\
# Hermes config (test fixture) — comments must survive a routing write.
_config_version: 51
model:
  default: model-a
  provider: prov
providers:
  prov:
    models:
      - model-a
      - model-b
display:
  skin: slate  # keep me
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated HERMES_HOME, removed by pytest at teardown.

    ``tmp_path`` rather than a directory under the real ``~/.hermes``: both keep
    the live config out of reach, and ``tests/home_io_guard.py`` correctly
    refuses any fixture that opens a path inside the real home — even a scratch
    subdirectory of it.  The point of the isolation is "not the live file", and
    this is the strongest form of it the suite allows.
    """
    import hermes_cli.config as cfg

    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    (path / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    yield path
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()


def _write(home: Path, text: str) -> None:
    import hermes_cli.config as cfg

    (home / "config.yaml").write_text(text, encoding="utf-8")
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()


def _read(home: Path) -> str:
    return (home / "config.yaml").read_text(encoding="utf-8")


def _raw(home: Path) -> dict:
    import hermes_cli.config as cfg

    cfg._RAW_CONFIG_CACHE.clear()
    return cfg.read_raw_config()


# ---------------------------------------------------------------------------
# reading: absence means "off", never an exception
# ---------------------------------------------------------------------------

def test_absent_block_reads_as_disabled_defaults(home):
    settings = mr.routing_settings()
    assert settings["enabled"] is False
    assert settings["strategy"] == "automatic"
    assert settings["stability_threshold"] == pytest.approx(0.15)
    assert settings["models"] == []


def test_absent_block_is_never_materialised(home):
    """A fresh config must load and re-save without gaining a routing block."""
    import hermes_cli.config as cfg

    assert "smart_model_routing" not in cfg.load_config()
    cfg.save_config(cfg.load_config())
    assert "smart_model_routing" not in _raw(home)


def test_disabling_an_absent_block_is_not_a_write(home):
    """Absence is the OFF state, so ``disable`` must not materialise a block.

    The complement of ``test_absent_block_is_never_materialised`` for the write
    path: a verb whose only change is ``enabled: false`` on a file that never
    carried a routing block has nothing to do, and a command that changes
    nothing must not change the file.
    """
    before = _read(home)

    result = mr.update_routing({"enabled": False})

    assert _read(home) == before, "the file must stay byte-identical"
    assert "smart_model_routing" not in _raw(home), "no block may be created"
    assert result["written"] is False
    assert result["changed"] == []
    assert result["settings"]["enabled"] is False


def test_disabling_a_present_block_still_writes(home):
    """The no-op rule is scoped to an *absent* block; a real change still lands."""
    _write(home, _CONFIG + "smart_model_routing:\n  enabled: true\n  strategy: cost\n")

    result = mr.update_routing({"enabled": False})

    assert result["written"] is True
    assert result["changed"] == ["enabled"]
    block = _raw(home)["smart_model_routing"]
    assert block["enabled"] is False
    assert block["strategy"] == "cost", "other preferences survive a disable"


def test_routing_settings_reads_the_block_from_disk(home):
    _write(home, _CONFIG + "smart_model_routing:\n  enabled: true\n  strategy: cost\n  stability_threshold: 0.4\n  models:\n    - model-b\n    - model-a\n")
    settings = mr.routing_settings()
    assert settings["enabled"] is True
    assert settings["strategy"] == "cost"
    assert settings["stability_threshold"] == pytest.approx(0.4)
    assert settings["models"] == ["model-b", "model-a"]


def test_routing_settings_accepts_a_config_mapping():
    settings = mr.routing_settings({"smart_model_routing": {"enabled": True, "models": ["m"]}})
    assert settings["enabled"] is True
    assert settings["models"] == ["m"]


@pytest.mark.parametrize("block", ["yes", 3, ["enabled"], {"enabled": "maybe"}])
def test_routing_settings_never_raises(block):
    """A malformed block is reported as the disabled defaults, not as truthy soup."""
    settings = mr.routing_settings({"smart_model_routing": block})
    assert settings["enabled"] is False
    assert mr.is_enabled({"smart_model_routing": block}) is False
    assert settings["strategy"] == "automatic"


def test_legacy_candidates_list_is_read_as_models():
    settings = mr.routing_settings({"smart_model_routing": {"candidates": ["m1", "m2"]}})
    assert settings["models"] == ["m1", "m2"]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block", [None, {}, {"enabled": True}, {"models": ["m"]}])
def test_absent_and_partial_blocks_are_valid(block):
    assert mr.validate_routing(block) == []


def test_a_canonical_block_is_valid():
    block = {
        "enabled": True,
        "strategy": "quality",
        "stability_threshold": 0.2,
        "models": ["a", {"name": "b", "tier_rank": 3}],
        "exclude": ["c"],
        "weights": {"cost": 1.0},
        "reliability": {"a": 0.9},
    }
    assert mr.validate_routing(block) == []


@pytest.mark.parametrize(
    ("block", "key"),
    [
        ({"enabled": "sometimes"}, "enabled"),
        ({"strategy": "cheapest"}, "strategy"),
        ({"stability_threshold": "high"}, "stability_threshold"),
        ({"stability_threshold": 2.0}, "stability_threshold"),
        ({"min_improvement": 0.1, "stability_threshold": 0.2}, "min_improvement"),
        ({"models": "a,b"}, "models"),
        ({"models": [""]}, "models"),
        ({"models": [{"tier_rank": 2}]}, "models"),
        ({"models": [{"name": "a", "tier": 2}]}, "tier"),
        ({"exclude": [3]}, "exclude"),
        ({"weights": {"cost": -1.0}}, "weights.cost"),
        ({"weights": {"vibes": 1.0}}, "weights.vibes"),
        ({"reliability": {"a": 4}}, "reliability.a"),
        ({"stability_treshold": 0.2}, "stability_treshold"),
    ],
)
def test_every_message_names_the_offending_key(block, key):
    problems = mr.validate_routing(block)
    assert problems, f"{block!r} should not validate"
    assert any(key in problem for problem in problems), problems


def test_a_non_mapping_block_is_rejected_with_its_type():
    problems = mr.validate_routing(["enabled"])
    assert problems and "smart_model_routing" in problems[0]


# ---------------------------------------------------------------------------
# the write path
# ---------------------------------------------------------------------------

def test_update_routing_writes_and_reads_back_from_disk(home):
    result = mr.update_routing(
        {"enabled": True, "strategy": "cost", "models": ["model-a", "model-b"]})

    assert result["written"] is True and result["dry_run"] is False
    assert result["path"].endswith("config.yaml")
    assert result["changed"] == ["enabled", "models", "strategy"]
    assert result["settings"]["enabled"] is True
    assert result["settings"]["strategy"] == "cost"
    assert result["settings"]["models"] == ["model-a", "model-b"]

    # ...and the file itself agrees with what was reported.
    on_disk = _raw(home)["smart_model_routing"]
    assert on_disk["enabled"] is True
    assert on_disk["models"] == ["model-a", "model-b"]

    # a second, partial update edits only what it names
    second = mr.update_routing({"enabled": False})
    assert second["changed"] == ["enabled"]
    assert second["settings"]["strategy"] == "cost"
    assert second["settings"]["models"] == ["model-a", "model-b"]


def test_a_round_trip_write_is_byte_identical_apart_from_the_block(home):
    before = _read(home)
    mr.update_routing({"enabled": True, "models": ["model-a"]})
    after = _read(home)

    diff = [line for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=0)]
    removed = [line for line in diff if line.startswith("-") and not line.startswith("---")]
    added = [line for line in diff if line.startswith("+") and not line.startswith("+++")]
    assert removed == [], f"the write rewrote existing lines: {removed}"
    assert "+smart_model_routing:" in added
    # everything added belongs to the new block, at its own indentation
    for line in added:
        body = line[1:]
        assert body == "smart_model_routing:" or body.startswith("  "), body

    assert before.count("#") == after.count("#")
    assert "skin: slate  # keep me" in after
    assert "model-a" in after and "provider: prov" in after


def test_update_routing_reports_only_real_changes(home):
    mr.update_routing({"enabled": True, "strategy": "automatic"})
    again = mr.update_routing({"enabled": True, "strategy": "automatic"})
    assert again["changed"] == []


def test_invalid_input_raises_and_writes_nothing(home):
    before = _read(home)
    with pytest.raises(ValueError) as excinfo:
        mr.update_routing({"enabled": True, "stability_threshold": 5})
    assert "stability_threshold" in str(excinfo.value)
    assert _read(home) == before
    assert "smart_model_routing" not in _raw(home)


def test_an_unknown_key_is_refused_rather_than_written(home):
    before = _read(home)
    with pytest.raises(ValueError) as excinfo:
        mr.update_routing({"enabled": True, "stability_treshold": 0.1})
    assert "stability_treshold" in str(excinfo.value)
    assert _read(home) == before


def test_a_failed_write_leaves_the_previous_file_intact(home, monkeypatch):
    import hermes_cli.config as cfg

    mr.update_routing({"enabled": False})
    before = _read(home)

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cfg, "save_config", _boom)
    with pytest.raises(OSError):
        mr.update_routing({"enabled": True})
    assert _read(home) == before


def test_dry_run_writes_nothing_but_reports_the_block(home):
    before = _read(home)
    result = mr.update_routing({"enabled": True}, dry_run=True)
    assert result["written"] is False and result["dry_run"] is True
    assert result["settings"]["enabled"] is True
    assert _read(home) == before


def test_a_config_mapping_makes_the_call_read_only(home):
    before = _read(home)
    result = mr.update_routing(
        {"enabled": True}, config={"smart_model_routing": {"enabled": False}})
    assert result["written"] is False
    assert result["settings"]["enabled"] is True
    assert _read(home) == before


def test_a_stale_unknown_key_is_preserved_with_a_warning(home):
    _write(home, _CONFIG + "smart_model_routing:\n  enabled: true\n  from_the_future: 1\n")
    result = mr.update_routing({"strategy": "quality"})
    assert result["written"] is True
    assert any("from_the_future" in warning for warning in result["warnings"])
    on_disk = _raw(home)["smart_model_routing"]
    assert on_disk["from_the_future"] == 1
    assert on_disk["strategy"] == "quality"


def test_tuples_and_nested_mappings_round_trip(home):
    result = mr.update_routing({
        "enabled": True,
        "exclude": ("model-b",),
        "reliability": {"model-a": 0.8},
    })
    assert result["settings"]["exclude"] == ["model-b"]
    assert result["settings"]["reliability"] == {"model-a": 0.8}
    assert isinstance(result["block"]["exclude"], list)


# ---------------------------------------------------------------------------
# config.py integration: validate_config_structure + migration
# ---------------------------------------------------------------------------

def test_validate_config_structure_is_silent_without_a_block(home):
    import hermes_cli.config as cfg

    issues = [i for i in cfg.validate_config_structure(cfg.load_config())
              if "smart_model_routing" in i.message]
    assert issues == []


def test_validate_config_structure_rejects_a_bad_block(home):
    import hermes_cli.config as cfg

    _write(home, _CONFIG + "smart_model_routing:\n  enabled: true\n  strategy: cheapest\n")
    issues = [i for i in cfg.validate_config_structure(cfg.load_config())
              if "smart_model_routing" in i.message]
    assert issues, "a bad strategy must be reported"
    assert any(i.severity == "error" for i in issues)
    assert any("strategy" in i.message for i in issues)


def test_validate_config_structure_accepts_what_update_routing_writes(home):
    import hermes_cli.config as cfg

    mr.update_routing({"enabled": True, "models": ["model-a", "model-b"]})
    issues = [i for i in cfg.validate_config_structure(cfg.load_config())
              if "smart_model_routing" in i.message]
    assert issues == []


def test_migration_accepts_a_legacy_block(home):
    import hermes_cli.config as cfg

    _write(home, _CONFIG + (
        "smart_model_routing:\n"
        "  enabled: true\n"
        "  min_improvement: 0.3\n"
        "  candidates:\n"
        "    - model-a\n"
        "    - model-b\n"))
    results = cfg.migrate_config(interactive=False, quiet=True)
    block = _raw(home)["smart_model_routing"]
    assert block["stability_threshold"] == pytest.approx(0.3)
    assert block["models"] == ["model-a", "model-b"]
    assert "min_improvement" not in block and "candidates" not in block
    assert block["enabled"] is True  # never toggled by a migration
    assert any("min_improvement" in entry for entry in results["config_added"])
    assert mr.validate_routing(block) == []


def test_migration_leaves_a_canonical_block_alone(home):
    import hermes_cli.config as cfg

    _write(home, _CONFIG + "smart_model_routing:\n  enabled: false\n  strategy: cost\n")
    before = _read(home)
    cfg.migrate_config(interactive=False, quiet=True)
    assert _raw(home)["smart_model_routing"] == {"enabled": False, "strategy": "cost"}
    assert _read(home) == before


def test_migration_does_not_invent_the_block(home):
    import hermes_cli.config as cfg

    cfg.migrate_config(interactive=False, quiet=True)
    assert "smart_model_routing" not in _raw(home)


# ---------------------------------------------------------------------------
# preview: what would be selected, and why
# ---------------------------------------------------------------------------

def _info(model, *, provider="prov", cost_in=1.0, cost_out=2.0):
    from agent.models_dev import ModelInfo

    return ModelInfo(
        id=model, name=model, family="", provider_id=provider, reasoning=False,
        tool_call=True, attachment=False, structured_output=False,
        input_modalities=("text",), output_modalities=("text",),
        context_window=200_000, max_output=8_192, cost_input=cost_in,
        cost_output=cost_out, status="",
    )


@pytest.fixture
def two_models(monkeypatch):
    def _candidates_for(provider, models, *, enabled=True):
        prices = {"model-a": (5.0, 15.0), "model-b": (0.2, 0.4)}
        return [
            mr.ModelCandidate(
                provider=provider, model=name, info=_info(name, cost_in=prices[name][0],
                                                          cost_out=prices[name][1]))
            for name in models if name in prices
        ]

    monkeypatch.setattr(mr, "candidates_for", _candidates_for)


def test_preview_says_what_would_be_selected_and_why(home, two_models):
    _write(home, _CONFIG + "smart_model_routing:\n  enabled: true\n  models:\n    - model-a\n    - model-b\n")
    doc = mr.routing_preview("What is 2 + 2?")

    assert doc["enabled"] is True
    assert doc["skip_reason"] == ""
    assert doc["request"]["tier"] == "simple"
    assert doc["selected"] == "model-b"          # the cheap one for a simple ask
    assert doc["selected_provider"] == "prov"
    assert doc["reason"]
    assert "model-b" in doc["explanation"]
    assert {row["model"] for row in doc["candidates"]} == {"model-a", "model-b"}


def test_preview_on_a_disabled_install_reports_the_hypothetical(home, two_models):
    _write(home, _CONFIG + "smart_model_routing:\n  models:\n    - model-a\n    - model-b\n")
    doc = mr.routing_preview("What is 2 + 2?")
    assert doc["enabled"] is False
    assert doc["skip_reason"] == "disabled"
    assert doc["selected"] == "model-b"


def test_preview_with_one_model_reports_the_only_candidate(home, two_models):
    """One candidate is not a choice — the preview says so without inventing a rival."""
    _write(home, _CONFIG + "smart_model_routing:\n  enabled: true\n  models:\n    - model-a\n")
    doc = mr.routing_preview("What is 2 + 2?")
    assert doc["models"] == ["model-a"]
    assert doc["eligible_count"] == 1
    assert doc["selected"] == "model-a"
    assert len(doc["candidates"]) == 1


def test_preview_uses_the_tier_the_turn_would_use(home, two_models):
    """A short but hard prompt is previewed at the tier classification gives it."""
    _write(home, _CONFIG + "smart_model_routing:\n  enabled: true\n  models:\n    - model-a\n    - model-b\n")
    doc = mr.routing_preview(
        "fix this kernel module segfault that only appears under memory pressure")
    assert doc["request"]["tier"] == mr.Complexity.MODERATE.label
    assert "code-signal" in doc["request"]["reasons"]


def test_preview_is_read_only(home, two_models):
    before = _read(home)
    mr.routing_preview("anything")
    assert _read(home) == before
