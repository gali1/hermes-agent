"""``hermes model`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_yes_flag


def build_model_parser(
    subparsers, *, cmd_model: Callable, cmd_model_routing: Callable | None = None
) -> None:
    """Attach the ``model`` subcommand to ``subparsers``."""
    model_parser = subparsers.add_parser(
        "model", help="Select default model and provider, or configure dynamic model routing",
        description=(
            "Interactively select your inference provider and default model.\n\n"
            "`hermes model routing` inspects or changes dynamic model selection\n"
            "(smart_model_routing); `hermes model routing configure` edits it."
        ))
    model_parser.add_argument(
        "--refresh", action="store_true",
        help="Wipe the model picker disk cache and re-fetch every provider's live /v1/models list.")
    model_parser.add_argument(
        "--portal-url", help="Portal base URL for Nous login (default: production portal)")
    model_parser.add_argument(
        "--inference-url",
        help="Inference API base URL for Nous login (default: production inference API)")
    model_parser.add_argument(
        "--client-id", default=None,
        help="OAuth client id to use for Nous login (default: hermes-cli)")
    model_parser.add_argument("--scope", default=None, help="OAuth scope to request for Nous login")
    model_parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not attempt to open the browser automatically during Nous login")
    model_parser.add_argument(
        "--timeout", type=float, default=15.0,
        help="HTTP request timeout in seconds for Nous login (default: 15)")
    model_parser.add_argument(
        "--ca-bundle", help="Path to CA bundle PEM file for Nous TLS verification")
    model_parser.add_argument(
        "--insecure", action="store_true",
        help="Disable TLS verification for Nous login (testing only)")
    model_parser.set_defaults(func=cmd_model)

    # -------------------------------------------------------------------------
    # model routing — dynamic model selection (smart_model_routing)
    # -------------------------------------------------------------------------
    # Additive: `hermes model` and its flags still reach the interactive picker,
    # so an untouched install keeps its exact CLI surface.  The handler is
    # injected (falling back to a lazy import) so this module never imports main.
    if cmd_model_routing is None:

        def _lazy_routing(args):  # noqa: ANN001 - injected handler shape
            from hermes_cli.main import cmd_model_routing as _handler

            return _handler(args)

        routing_handler = _lazy_routing
    else:
        routing_handler = cmd_model_routing

    model_sub = model_parser.add_subparsers(dest="model_command")
    routing_parser = model_sub.add_parser(
        "routing",
        help="Inspect or change dynamic model selection",
        description=(
            "Dynamic model selection (smart_model_routing) picks the cheapest "
            "model that can actually serve each request.\n\n"
            "Off by default: while disabled no model is ever switched and the "
            "agent behaves exactly as before."
        ),
    )
    routing_sub = routing_parser.add_subparsers(dest="routing_command")

    _status = routing_sub.add_parser(
        "status",
        help="Show whether dynamic selection is on and what it can choose from",
        description=(
            "Print the resolved routing settings and the candidate pool.\n\n"
            "With --json, prints the shared machine-readable report: the exact\n"
            "key set enabled/settings/candidate_source/request/pricing/selected/\n"
            "selected_provider/reason/skip_reason/current/kept_current/\n"
            "eligible_count/candidates (sorted keys, 2-space indent). Read-only."
        ),
    )
    _status.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit the shared JSON report (sort_keys, indent=2) instead of text",
    )
    _status.add_argument(
        "prompt",
        nargs="*",
        help="Optional user message to classify for the report (default: none)",
    )
    _status.add_argument(
        "--current-model",
        default=None,
        help="Treat this model as the session's incumbent (default: config default)",
    )
    _status.add_argument(
        "--provider",
        default=None,
        help="Provider to resolve candidates against (default: model.provider)",
    )
    _status.add_argument(
        "--attachments",
        default=None,
        metavar="image,pdf,...",
        help="Modality hints for the request (image, pdf, audio, video)",
    )
    _status.add_argument(
        "--conversation-tokens",
        type=int,
        default=0,
        metavar="N",
        help="Size of the running context, added to the input estimate",
    )

    _enable = routing_sub.add_parser(
        "enable", help="Turn dynamic model selection on (persists to config.yaml)"
    )
    _enable.add_argument(
        "--strategy",
        choices=["automatic", "cost", "quality", "tool_calls_only"],
        default=None,
        help="Selection bias (default: automatic — capability first, then cost; tool_calls_only keeps the default model primary)",
    )
    _enable.add_argument(
        "--stability-threshold",
        type=float,
        default=None,
        metavar="N",
        help="Score a rival must beat the current model by before switching (default 0.15)",
    )
    _enable.add_argument(
        "--switch-on-cost",
        action="store_true",
        default=None,
        help="Allow a mid-session switch to a cheaper model (breaks the prompt cache)",
    )
    _enable.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "If no preferred model can serve a request, fall back to the "
            "provider's catalog instead of keeping an incapable model"
        ),
    )
    _enable.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Skip the confirmation prompt (for scripts and cron)",
    )

    routing_sub.add_parser(
        "disable", help="Turn dynamic model selection off (persists to config.yaml)"
    )

    _explain = routing_sub.add_parser(
        "explain",
        help="Show the decision dynamic selection would make for a prompt",
    )
    _explain.add_argument(
        "prompt",
        nargs="+",
        help="The user message to classify (quote it if it contains spaces)",
    )
    _explain.add_argument(
        "--candidates",
        default=None,
        metavar="M1,M2,...",
        help="Preview against these model ids instead of the configured list",
    )
    _explain.add_argument(
        "--current-model",
        default=None,
        help="Treat this model as the session's incumbent (default: config default)",
    )
    _explain.add_argument(
        "--provider",
        default=None,
        help="Provider to resolve candidates against (default: model.provider)",
    )
    _explain.add_argument(
        "--attachments",
        default=None,
        metavar="image,pdf,...",
        help="Modality hints for the request (image, pdf, audio, video)",
    )
    _explain.add_argument(
        "--conversation-tokens",
        type=int,
        default=0,
        metavar="N",
        help="Size of the running context, added to the input estimate",
    )

    _refresh = routing_sub.add_parser(
        "refresh",
        help="Refresh the model catalog dynamic selection chooses from",
        description=(
            "Force-refresh the models.dev catalog that routing resolves candidates\n"
            "from — the same cache agent.models_dev already serves (no second\n"
            "cache, no per-message fetch).\n\n"
            "Read-only with respect to config.yaml: this touches the catalog cache,\n"
            "never smart_model_routing. Useful after a provider adds models, when\n"
            "`status` still shows the old pool."
        ),
    )
    _refresh.add_argument(
        "--provider",
        default=None,
        help="Report this provider's pool (default: model.provider from config)",
    )
    _refresh.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit the result as JSON (refreshed/providers/models/candidates/elapsed_ms)",
    )

    _configure = routing_sub.add_parser(
        "configure",
        help="Review and change every routing setting (interactive, or with flags)",
        description=(
            "Review the resolved settings and change them.\n\n"
            "In a terminal with no flags this runs the interactive flow: current\n"
            "state, enable/disable, all eligible models vs only your selected\n"
            "models, model selection through the normal picker, preferences,\n"
            "before/after summary with a preview, then confirmation.\n\n"
            "With flags it is fully scriptable and needs no TTY.  Nothing is\n"
            "written until the end, so cancelling leaves config.yaml unchanged.\n"
            "Every write goes through agent.model_router.update_routing and is\n"
            "read back from disk.  Verify afterwards with:\n"
            "  hermes model routing status"
        ),
    )
    _configure.add_argument(
        "--models",
        action="append",
        default=None,
        metavar="M1,M2,...",
        help=(
            "Route only among these model ids, in priority order "
            "(comma-separated; repeat the flag to add more)"
        ),
    )
    _configure.add_argument(
        "--all-models",
        "--all-eligible",
        dest="all_models",
        action="store_true",
        default=False,
        help="Route across the whole eligible provider catalog (clears the selected list)",
    )
    _configure_mode = _configure.add_mutually_exclusive_group()
    _configure_mode.add_argument(
        "--enable", action="store_true", default=False,
        help="Turn dynamic model selection on")
    _configure_mode.add_argument(
        "--disable", action="store_true", default=False,
        help="Turn dynamic model selection off")
    _configure.add_argument(
        "--strategy",
        choices=["automatic", "cost", "quality", "tool_calls_only"],
        default=None,
        help="Selection bias (default: automatic — capability first, then cost; tool_calls_only keeps the default model primary)",
    )
    _configure.add_argument(
        "--stability-threshold",
        "--threshold",
        dest="stability_threshold",
        type=float,
        default=None,
        metavar="N",
        help="Score a rival must beat the current model by before switching (0-1)",
    )
    _configure.add_argument(
        "--cost-optimization",
        dest="cost_optimization",
        action="store_true",
        default=None,
        help="Prefer cheaper models that can still serve the request",
    )
    _configure.add_argument(
        "--no-cost-optimization",
        dest="cost_optimization",
        action="store_false",
        help="Prefer capability over price",
    )
    _configure.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="M1,M2,...",
        help=(
            "Never auto-select these model ids (comma-separated; repeat the flag to "
            "add more). They stay usable through an explicit model choice"
        ),
    )
    _configure.add_argument(
        "--clear-exclude",
        action="store_true",
        default=False,
        help="Empty the never-auto-select list",
    )
    _configure.add_argument(
        "--allow-unknown-metadata",
        dest="allow_unknown_metadata",
        action="store_true",
        default=None,
        help=(
            "Treat missing capability metadata as unknown (penalised), not unsupported "
            "— the default"
        ),
    )
    _configure.add_argument(
        "--no-allow-unknown-metadata",
        "--strict-metadata",
        dest="allow_unknown_metadata",
        action="store_false",
        help="Strict: a model missing a required capability's metadata is ineligible",
    )
    _configure.add_argument(
        "--switch-on-cost",
        dest="switch_on_cost",
        action="store_true",
        default=None,
        help=(
            "Allow a mid-session switch to a cheaper model (invalidates the provider's "
            "prompt cache)"
        ),
    )
    _configure.add_argument(
        "--no-switch-on-cost",
        dest="switch_on_cost",
        action="store_false",
        help="Keep the session's model even when a cheaper one would do — the default",
    )
    _configure.add_argument(
        "--weights",
        action="append",
        default=None,
        metavar="DIM=N,...",
        help=(
            "Scoring weights for the automatic strategy, merged over the current "
            "values (capability, cost, reliability, context, continuity)"
        ),
    )
    _configure.add_argument(
        "--weights-default",
        action="store_true",
        default=False,
        help="Reset the scoring weights to the engine defaults",
    )
    _configure.add_argument(
        "--reliability",
        action="append",
        default=None,
        metavar="MODEL=RATE,...",
        help=(
            "Declared success rate per model (0-1), merged over the current values; "
            "expected retries are priced from it"
        ),
    )
    _configure.add_argument(
        "--clear-reliability",
        action="store_true",
        default=False,
        help="Forget every declared reliability rate",
    )
    _configure.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Validate and show what would change without writing config.yaml")
    _configure.add_argument(
        "--json", action="store_true", default=False,
        help="Print the writer's result (settings read back from disk) as JSON")
    add_yes_flag(
        _configure, "Assume yes: skip the final confirmation prompt")

    routing_parser.set_defaults(func=routing_handler)


# ---------------------------------------------------------------------------
# The shared writer layer
# ---------------------------------------------------------------------------
# Every interface — `hermes model routing` (the parser above), the setup wizard and the
# gateway (`gateway/slash_commands_model.py`) — reaches config.yaml through the functions
# below: one read -> merge -> validate -> atomic write -> read-back path, so no surface can
# disagree with another about what was saved. They stay importable library entry points even
# where the parser dispatches through the injected handler.


def cmd_model_routing_status(args) -> int:
    """Print the routing contract and the decision the next turn would make.

    Read-only by construction: it calls the same pure functions selection calls, so the report
    cannot disagree with the behaviour it describes. Always exits 0 — a disabled or misconfigured
    routing section is a state to report, not a failure.
    """
    import json

    from agent.model_routing import explain_routing
    from hermes_cli.config import load_config_readonly

    try:
        cfg = load_config_readonly()
    except Exception as exc:  # a diagnostic must say "unreadable", never pretend routing is off
        print(f"routing: config unreadable ({exc.__class__.__name__}: {exc})")
        return 2
    cfg = cfg if isinstance(cfg, dict) else {}
    raw = cfg.get("smart_model_routing")
    model_cfg = cfg.get("model")
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    current = str(model_cfg.get("default") or "").strip() or None
    provider = str(model_cfg.get("provider") or "").strip()
    text = getattr(args, "text", None) or (
        "Refactor the storage layer and explain the trade-offs.")
    report = explain_routing(raw if isinstance(raw, dict) else {}, text,
                             current=current, provider=provider)
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    # One renderer for status and configure: two copies drifted, and the pool line landed in
    # only one of them, so `status` kept reporting a count without the models behind it.
    print(_render_routing(report))
    print(f"reason: {report['reason']}")
    return 0


_PROBE_PROMPT = "Refactor the storage layer and explain the trade-offs."


def _routing_report(cfg, raw, text: str = _PROBE_PROMPT):
    """Explain a candidate routing block using the same engine `status` uses."""
    from agent.model_routing import explain_routing

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return explain_routing(
        raw if isinstance(raw, dict) else {}, text,
        current=str(model_cfg.get("default") or "").strip() or None,
        provider=str(model_cfg.get("provider") or "").strip())


def _render_routing(report) -> str:
    """The one renderer, matching `status` so the two can never disagree."""
    label = "would select" if report["enabled"] else "would select if enabled"
    lines = [
        f"enabled: {str(report['enabled']).lower()}",
        f"strategy: {report['settings']['strategy']} "
        f"(stability_threshold={report['settings']['stability_threshold']})",
        f"request tier: {report['request']['tier']}",
    ]
    # Show the pool itself, not just its size. "candidates: 4" told nobody whether their
    # selection had been applied; naming the models makes that answerable at a glance.
    pool = report.get("configured_models") or []
    if pool:
        scope = "only these" if report["settings"].get("preferred_models_only") else "priority order"
        lines.append(f"selected models ({len(pool)}, {scope}):")
        for index, name in enumerate(pool, 1):
            current = report.get("current") == name
            lines.append(f"  {index}. {name}{'  ← current' if current else ''}")
    else:
        lines.append("selected models: all eligible from the provider catalog")
    lines.append(f"candidates: {report['eligible_count']} eligible from {report['candidate_source']}")
    lines.append(f"current: {report.get('current') or '(unset)'}")
    lines.append(f"{label}: {report['selected'] or '(none)'}")
    lines.append(f"skip_reason: {report['skip_reason'] or '(switches)'}")
    if report.get("exhausted_skipped"):
        lines.append(f"blocked this window (free-tier limit): {', '.join(report['exhausted_skipped'])}")
    guard = report.get("free_guard") or {}
    if guard.get("enabled"):
        detail = f"free-model guard: on (window {guard.get('window')}, resets {guard.get('resets_at')}"
        if guard.get("fallback"):
            detail += f", fallback {guard['fallback']}"
        detail += ")"
        lines.append(detail)
        for name in guard.get("exhausted") or []:
            usage = (guard.get("usage") or {}).get(name) or {}
            lines.append(f"  exhausted: {name} — {usage.get('requests', 0)} requests, "
                         f"{usage.get('tokens', 0)} tokens before the limit")
        counted = [(n, u) for n, u in (guard.get("usage") or {}).items() if u.get("requests")]
        if counted:
            lines.append("  usage today: " + ", ".join(
                f"{n} ({u['requests']} req/{u['tokens']} tok)" for n, u in counted[:6]))
    if report.get("moa_active"):
        # Say it in words, not just as a skip_reason code: the selection shown above will not
        # happen while the fan-out is running, and a reader must not take it as a prediction.
        lines.append("moa: active — routing is skipped while Mixture-of-Agents runs")
    return "\n".join(lines)

def write_routing_config(changes: dict) -> dict:
    """Apply *changes* to ``smart_model_routing`` and return the outcome.

    The single writer for every interface: CLI, setup wizard and gateway all call this, so
    there is exactly one read → merge → validate → atomic write → read-back path. Returns
    ``{"ok": bool, "problems": [...], "saved": <block read back from disk>}``; on failure the
    previous file is left intact and ``saved`` is the unchanged block.
    """
    from agent.model_routing import merge_routing_config, validate_routing_config
    from hermes_cli.config import atomic_config_replace, get_config_path, read_user_config_raw

    path = get_config_path()
    data = dict(read_user_config_raw(path) or {})
    old_raw = data.get("smart_model_routing")
    new_raw = merge_routing_config(old_raw, **changes)

    problems = validate_routing_config(new_raw)
    if problems:
        return {"ok": False, "problems": problems, "saved": old_raw, "before": old_raw,
                "after": new_raw, "data": data}

    data["smart_model_routing"] = new_raw
    atomic_config_replace(path, data)
    saved = (read_user_config_raw(path) or {}).get("smart_model_routing")
    return {"ok": True, "problems": [], "saved": saved, "before": old_raw,
            "after": new_raw, "data": data}


def _routing_apply(*, changes: dict, assume_yes: bool) -> int:
    """Read, merge, validate, preview, confirm, write atomically, then read back.

    The single writer shared by every interface. A failed or cancelled run writes nothing.
    """
    from agent.model_routing import merge_routing_config
    from hermes_cli.config import get_config_path, read_user_config_raw

    try:
        data = dict(read_user_config_raw(get_config_path()) or {})
    except Exception as exc:
        print(f"routing: config unreadable ({exc.__class__.__name__}: {exc})")
        return 2

    old_raw = data.get("smart_model_routing")
    new_raw = merge_routing_config(old_raw, **changes)

    before = _routing_report(data, old_raw)
    after = _routing_report(data, new_raw)
    print("--- before ---")
    print(_render_routing(before))
    print("--- after ---")
    print(_render_routing(after))

    # Enabling changes behaviour on every turn, so it always needs an explicit confirmation
    # that includes the preview (spec: defaults are safe).
    if new_raw.get("enabled") and not assume_yes:
        try:
            answer = input("Enable smart model routing? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("routing: cancelled — nothing written")
            return 0

    try:
        result = write_routing_config(changes)
    except Exception as exc:
        print(f"routing: write failed ({exc.__class__.__name__}: {exc}); previous config intact")
        return 2
    if not result["ok"]:
        for problem in result["problems"]:
            print(f"routing: {problem}")
        print("routing: nothing written")
        return 1

    print("--- saved (read back from disk) ---")
    print(_render_routing(_routing_report(data, result["saved"])))
    print("\nverify: hermes model routing status")
    return 0


def cmd_model_routing_enable(args) -> int:
    return _routing_apply(changes={"enabled": True}, assume_yes=bool(getattr(args, "yes", False)))


def cmd_model_routing_disable(args) -> int:
    return _routing_apply(changes={"enabled": False}, assume_yes=True)


def cmd_model_routing_configure(args) -> int:
    changes: dict = {}
    if getattr(args, "strategy", None):
        changes["strategy"] = args.strategy
    if getattr(args, "threshold", None) is not None:
        changes["stability_threshold"] = float(args.threshold)
    if getattr(args, "all_eligible", False):
        changes["models"] = []
    elif getattr(args, "models", None) is not None:
        raw = str(args.models).strip()
        changes["models"] = [m.strip() for m in raw.split(",") if m.strip()] if raw else []
    if getattr(args, "selected_only", False):
        changes["preferred_models_only"] = True
    if getattr(args, "allow_unknown", None) is not None:
        changes["allow_unknown_metadata"] = bool(args.allow_unknown)
    if getattr(args, "cost_opt", None) is not None:
        changes["cost_optimization"] = bool(args.cost_opt)
    if not changes:
        import sys
        from hermes_cli.setup import is_noninteractive

        if is_noninteractive() or not sys.stdin.isatty():
            print("routing: nothing to change — pass at least one option "
                  "(--strategy, --threshold, --models, --all-eligible, --selected-only, ...)")
            return 1
        return _routing_interactive()
    return _routing_apply(changes=changes, assume_yes=bool(getattr(args, "yes", False)))


def _routing_interactive() -> int:
    """The interactive routing flow: state → action → options → summary → confirm → write.

    Every prompt comes from ``hermes_cli.setup`` so keybindings, cancel semantics and the
    non-interactive fallbacks match the provider/model pickers. Cancelling at any step returns
    without writing.
    """
    from agent.model_routing import merge_routing_config, options_from_catalog
    from hermes_cli.config import get_config_path, read_user_config_raw
    from hermes_cli.setup import is_noninteractive, prompt_choice, prompt_multi_choice, prompt_yes_no

    if is_noninteractive():
        print("routing: no interactive terminal available — use `hermes model routing configure "
              "--strategy/--threshold/--models`")
        return 1

    try:
        data = dict(read_user_config_raw(get_config_path()) or {})
    except Exception as exc:
        print(f"routing: config unreadable ({exc.__class__.__name__}: {exc})")
        return 2

    model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
    provider = str(model_cfg.get("provider") or "").strip()
    old_raw = data.get("smart_model_routing")
    block = old_raw if isinstance(old_raw, dict) else {}

    print("--- current ---")
    print(_render_routing(_routing_report(data, block)))
    print()

    if not provider:
        print("routing: no provider configured — run `hermes model` first")
        return 1

    changes: dict = {}
    actions = ["Enable routing", "Disable routing", "Choose models", "Preferences", "Cancel"]
    choice = prompt_choice("What do you want to change?", actions, default=4)

    if choice == 4:
        print("routing: cancelled — nothing written")
        return 0

    if choice == 0:
        changes["enabled"] = True
    elif choice == 1:
        changes["enabled"] = False
    elif choice == 2:
        mode = prompt_choice(
            "Which models may routing use?",
            ["All eligible models", "Only my selected models"], default=0)
        if mode == 0:
            changes["models"] = []
            changes["preferred_models_only"] = False
        else:
            candidates = options_from_catalog(provider)
            names = [o.name for o in candidates]
            if not names:
                print(f"routing: no catalog models found for provider {provider!r}")
                return 1
            if len(names) > 50:
                print(f"  {len(names)} models available — type 'all', 'none', or numbers.")
            preselected = [names.index(m) for m in block.get("models") or [] if m in names]
            picked = prompt_multi_choice("Select models (priority = list order)",
                                         names, preselected=preselected)
            if picked is None:
                print("routing: cancelled — nothing written")
                return 0
            if not picked:
                print("routing: no models selected — nothing written")
                return 0
            changes["models"] = [names[i] for i in picked]
            changes["preferred_models_only"] = True
    elif choice == 3:
        strategies = ["automatic", "cost", "quality", "tool_calls_only"]
        current = str(block.get("strategy") or "automatic")
        idx = prompt_choice("Strategy?", strategies,
                            default=strategies.index(current) if current in strategies else 0)
        changes["strategy"] = strategies[idx]
        try:
            raw_threshold = input(
                f"stability_threshold [{block.get('stability_threshold', 0.15)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw_threshold = ""
        if raw_threshold:
            try:
                changes["stability_threshold"] = float(raw_threshold)
            except ValueError:
                print(f"routing: {raw_threshold!r} is not a number — nothing written")
                return 1

    new_raw = merge_routing_config(old_raw, **changes)
    print("\n--- after ---")
    print(_render_routing(_routing_report(data, new_raw)))

    if not prompt_yes_no("Save these changes?", default=False):
        print("routing: cancelled — nothing written")
        return 0

    try:
        result = write_routing_config(changes)
    except Exception as exc:
        print(f"routing: write failed ({exc.__class__.__name__}: {exc}); previous config intact")
        return 2
    if not result["ok"]:
        for problem in result["problems"]:
            print(f"routing: {problem}")
        print("routing: nothing written")
        return 1

    print("\n--- saved (read back from disk) ---")
    print(_render_routing(_routing_report(data, result["saved"])))
    print("\nverify: hermes model routing status")
    return 0
