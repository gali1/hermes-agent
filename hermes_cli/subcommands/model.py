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
        choices=["automatic", "cost", "quality"],
        default=None,
        help="Selection bias (default: automatic — capability first, then cost)",
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
        choices=["automatic", "cost", "quality"],
        default=None,
        help="Selection bias (default: automatic — capability first, then cost)",
    )
    _configure.add_argument(
        "--stability-threshold",
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
        "--dry-run", action="store_true", default=False,
        help="Validate and show what would change without writing config.yaml")
    _configure.add_argument(
        "--json", action="store_true", default=False,
        help="Print the writer's result (settings read back from disk) as JSON")
    add_yes_flag(
        _configure, "Assume yes: skip the final confirmation prompt")

    routing_parser.set_defaults(func=routing_handler)
