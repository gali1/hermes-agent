"""``hermes model`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_model_parser(subparsers, *, cmd_model: Callable) -> None:
    """Attach the ``model`` subcommand to ``subparsers``."""
    model_parser = subparsers.add_parser(
        "model", help="Select default model and provider",
        description="Interactively select your inference provider and default model")
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
    # ``hermes model`` alone stays the interactive picker; the optional subcommands are
    # diagnostics that must never fall through to it.
    model_subs = model_parser.add_subparsers(dest="model_command", metavar="<command>")
    routing_parser = model_subs.add_parser(
        "routing", help="Inspect dynamic model routing (smart_model_routing)")
    routing_parser.set_defaults(func=cmd_model_routing_status)
    routing_subs = routing_parser.add_subparsers(dest="routing_command", metavar="<command>")
    status_parser = routing_subs.add_parser(
        "status",
        help="Report the routing contract and the model the next turn would pick",
        description="Read-only. Never switches the model, never writes config.")
    status_parser.add_argument(
        "--json", action="store_true", help="Emit the full routing report as JSON.")
    status_parser.add_argument(
        "--text", default=None,
        help="Prompt to classify (default: a representative probe prompt).")
    status_parser.set_defaults(func=cmd_model_routing_status)


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
    print(f"enabled: {str(report['enabled']).lower()}")
    print(f"strategy: {report['settings']['strategy']} "
          f"(stability_threshold={report['settings']['stability_threshold']})")
    print(f"request tier: {report['request']['tier']}")
    print(f"candidates: {report['eligible_count']} eligible "
          f"from {report['candidate_source']}")
    print(f"current: {report['current'] or '(unset)'}")
    label = "would select" if report["enabled"] else "would select if enabled"
    print(f"{label}: {report['selected'] or '(none)'}")
    print(f"skip_reason: {report['skip_reason'] or '(switches)'}")
    print(f"reason: {report['reason']}")
    return 0
