"""``hermes model`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_model_parser(
    subparsers, *, cmd_model: Callable, cmd_model_routing: Callable | None = None
) -> None:
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

    routing_sub.add_parser(
        "status",
        help="Show whether dynamic selection is on and what it can choose from",
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
        help="Score a rival must beat the current model by before switching (default 0.08)",
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

    routing_parser.set_defaults(func=routing_handler)
