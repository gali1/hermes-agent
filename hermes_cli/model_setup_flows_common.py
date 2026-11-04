"""Shared skeleton for the ``_model_flow_*`` wizards: banner, credentials, model list, picker,
persist, confirmation — one copy of each step. Prompt strings and the config keys written (and
their insertion order = config.yaml key order) are behavior; keep them byte-identical.

main_provider_setup / auth / config imports are lazy on purpose: main.py imports the flows (import
cycle) and tests patch ``hermes_cli.config.load_config`` etc. at call time.
"""

from __future__ import annotations

import contextlib
import subprocess
from urllib.parse import urlparse

from hermes_cli.cli_output import line_input
from hermes_cli.config import clear_model_endpoint_credentials

_HTTP = ("http://", "https://")


def _note_setup_failure(failure_class: str) -> None:
    """Why a flow returns without a pick, for the provider-setup metric (no-op outside a tracked flow)."""
    from hermes_cli.observability.shared_metrics_setup import note_provider_setup_failure

    note_provider_setup_failure(failure_class)


def _say(*lines: str) -> None:
    """``print`` each line (``""`` = blank line); one call per banner block."""
    print("\n".join(lines))


def _ask(prompt: str, *, secret: bool = False, raw: bool = False, cancel_msg: str | None = "\nCancelled.",
         on_cancel=None):
    """One stripped prompt answer. ``raw`` uses builtin ``input`` (numbered fallbacks),
    ``secret`` the masked prompt, else ``line_input``. Ctrl-C/EOF prints *cancel_msg*
    (``None`` = silent) and returns *on_cancel*."""
    if secret:
        from hermes_cli.secret_prompt import masked_secret_prompt as fn
    else:
        fn = input if raw else line_input
    try:
        return fn(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        if cancel_msg is not None:
            print(cancel_msg)
        return on_cancel


def _existing_api_key_for_model_flow(provider_id: str, pconfig) -> tuple[str, str]:
    """Resolve an existing wizard credential without changing its storage."""
    from hermes_cli.auth import _resolve_api_key_provider_secret
    return _resolve_api_key_provider_secret(provider_id, pconfig)


def _ensure_flow_api_key(provider_id: str, pconfig, *, missing_hint=()) -> tuple[str, str, bool]:
    """Resolve the stored key, print *missing_hint* lines when none exists, then run
    ``_prompt_api_key`` (users can replace a stale key in-flow via K/R/C).

    Returns ``(existing_key, resolved_key, abort)``.
    """
    from hermes_cli.main_provider_setup import _prompt_api_key
    existing_key, existing_source = _existing_api_key_for_model_flow(provider_id, pconfig)
    if not existing_key:
        for line in missing_hint:
            print(line)
    resolved, abort = _prompt_api_key(pconfig, existing_key, provider_id=provider_id, existing_source=existing_source)
    return existing_key, resolved, abort


def _load_config_model_section() -> tuple[dict, dict]:
    """Return ``(cfg, cfg["model"])`` with the model section coerced to a dict."""
    from hermes_cli.config import load_config
    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    return cfg, model


def _begin_model_config(selected: str, provider: str) -> tuple[dict, dict]:
    """Record *selected* as the model choice and open the config model section
    with ``provider`` set; callers set endpoint fields then ``_commit_model_config``."""
    from hermes_cli.auth import _save_model_choice
    _save_model_choice(selected)
    cfg, model = _load_config_model_section()
    model["provider"] = provider
    return cfg, model


def _commit_model_config(cfg: dict) -> None:
    """Persist *cfg* and deactivate any OAuth provider."""
    from hermes_cli.auth import deactivate_provider
    from hermes_cli.config import save_config
    save_config(cfg)
    deactivate_provider()


def _persist_model(selected: str, provider: str, *, base_url: str | None = None, api_mode: str | None = None,
                   drop_base_url: bool = False, drop_api_mode: bool = False, clear_creds: bool = True,
                   finish=None) -> dict:
    """The standard persist step: ``_save_model_choice`` → model section with ``provider``,
    ``base_url`` then ``api_mode`` (that order is the config.yaml key order) → scrub inline
    endpoint credentials (``clear_creds``; ``drop_api_mode`` also pops ``api_mode``) →
    *finish(cfg, model)* for extra sections → save + deactivate OAuth provider."""
    cfg, model = _begin_model_config(selected, provider)
    if base_url is not None:
        model["base_url"] = base_url
    if api_mode is not None:
        model["api_mode"] = api_mode
    if drop_base_url:
        model.pop("base_url", None)
    if clear_creds:
        clear_model_endpoint_credentials(model, clear_api_mode=drop_api_mode)
    elif drop_api_mode:
        model.pop("api_mode", None)
    if finish is not None:
        finish(cfg, model)
    _commit_model_config(cfg)
    return model


def _finish_model(selected, provider: str, done: str, *, no_change: str = "No change.", **persist_kw):
    """``_persist_model`` + confirmation line when *selected*, else the no-change line.
    Returns the saved model section (None when nothing was selected)."""
    if not selected:
        print(no_change)
        return None
    model = _persist_model(selected, provider, **persist_kw)
    print(done)
    return model


def _activate_provider_model(selected, provider_id: str, base_url: str, done: str,
                             no_change: str | None = "No change.") -> None:
    """OAuth-provider persist: model choice + ``_update_config_for_provider`` (which owns
    the auth-state bookkeeping), then *done*; *no_change* (``None`` = silent) otherwise."""
    from hermes_cli.auth import _save_model_choice, _update_config_for_provider
    if not selected:
        if no_change is not None:
            print(no_change)
        return
    _save_model_choice(selected)
    _update_config_for_provider(provider_id, base_url)
    print(done)


def _ensure_dict_section(cfg: dict, key: str) -> dict:
    """Return ``cfg[key]`` as a dict, replacing a missing/non-dict value."""
    section = cfg.get(key)
    if not isinstance(section, dict):
        section = {}
    cfg[key] = section
    return section


def _pick_model_or_prompt(model_list, prompt: str, **kwargs):
    """Radio picker when *model_list* is non-empty, else a free-text ``line_input``
    (None on Ctrl-C/EOF)."""
    from hermes_cli.auth import _prompt_model_selection
    if model_list:
        return _prompt_model_selection(model_list, **kwargs)
    return _ask(prompt, cancel_msg=None)


def _login_retry_context(args) -> tuple[str, str]:
    """(retry_command, service_host) for a login helper's failure copy, from the ``ProviderConfig``
    among its positional args. Nous keeps ``hermes portal``; every other OAuth provider is retried
    with ``hermes auth add <provider>`` and named by its own portal host (``hermes login`` no longer
    exists). Falls back to ``hermes model`` when no provider config is in play."""
    pconfig = next((a for a in args if hasattr(a, "id") and hasattr(a, "portal_base_url")), None)
    if pconfig is None:
        return "hermes model", "the sign-in service"
    provider_id = str(getattr(pconfig, "id", "") or "")
    host = urlparse(str(getattr(pconfig, "portal_base_url", "") or "")).hostname or "the sign-in service"
    if provider_id == "nous":
        return "hermes portal", host
    return (f"hermes auth add {provider_id}" if provider_id else "hermes model"), host


def _run_login(login_fn, *args, **kwargs) -> bool:
    """Run an OAuth login helper; print plain failure copy (what happened + retry command) and
    return False on SystemExit / any exception. The retry command and service host come from the
    provider config passed to the helper, so a MiniMax failure never says ``hermes portal`` /
    ``portal.nousresearch.com``. Helpers that print their own copy raise ``SystemExit(1)`` with no
    message, which stays silent; a SystemExit that carries a message (or a non-cancel code from a
    helper that printed nothing) gets a one-line explanation so the user is never left with no
    output."""
    from hermes_cli.auth_error_copy import sign_in_failure_lines
    retry_command, service_host = _login_retry_context(args)
    try:
        login_fn(*args, **kwargs)
    except SystemExit as exc:
        if exc.code in (130, None, 0):
            print("Sign-in was cancelled.")
        elif isinstance(exc.code, str) and exc.code.strip():
            print(f"Sign-in did not complete: {exc.code.strip()}")
            print(f"Run `{retry_command}` to try again.")
        else:
            print(f"Sign-in did not complete; run `{retry_command}` to try again.")
        return False
    except Exception as exc:
        for line in sign_in_failure_lines(exc, service_host=service_host, retry_command=retry_command):
            print(line)
        return False
    return True


def _oauth_gate(logged_in: bool, name: str, login_fn, *login_args, fresh_name: str = "", recheck=None) -> bool:
    """Login-or-reuse step for OAuth providers. Not logged in → start a login. Logged in →
    use / reauthenticate (``force_new_login``; *recheck()* must then report logged in) /
    cancel. Returns False when the flow must stop."""
    if not logged_in:
        _say(f"Not logged into {name}. Starting login...", "")
        return _run_login(login_fn, *login_args)
    _say(f"  {name} credentials: ✓", "")
    choice = _prompt_auth_credentials_choice(f"{name} credentials:")
    if choice == "cancel":
        return False
    if choice == "reauth":
        _say(f"Starting a fresh {fresh_name or name} login...", "")
        if not _run_login(login_fn, *login_args, force_new_login=True):
            return False
        if recheck is not None and not recheck():
            print("Login failed.")
            return False
    return True


def _models_dev_merged(provider_id: str, curated) -> list:
    """models.dev agentic models for *provider_id* plus curated ids not yet listed
    (case-insensitive). Empty list when models.dev has nothing / is unavailable."""
    mdev_models: list = []
    with contextlib.suppress(Exception):
        from agent.models_dev import list_agentic_models
        mdev_models = list_agentic_models(provider_id)
    if not mdev_models:
        return []
    seen = {m.lower() for m in mdev_models}
    merged = list(mdev_models)
    for m in curated:
        if m.lower() not in seen:
            merged.append(m)
            seen.add(m.lower())
    return merged


def _show_curated(model_list) -> None:
    if model_list:
        print(f'  Showing {len(model_list)} curated models — use "Enter custom model name" for others.')


def _prune_replaced_custom_model_config_credentials(base_url: str, *, provider_name: str = "") -> None:
    """Drop stale ``model_config`` ("the credential under ``model.api_key``") entries from inactive
    custom pools: after an explicit custom-endpoint switch an old pool still carrying that source
    points at the previous endpoint and could be selected before the fresh config."""
    try:
        from agent.credential_pool import CUSTOM_POOL_PREFIX, custom_provider_pool_key_candidates
        from hermes_cli.auth import read_credential_pool, write_credential_pool

        # A keyed ``providers.<key>`` endpoint stores under the durable slug while
        # legacy pools keep ``custom:<display-name>``; every identity the active
        # endpoint may occupy must be skipped or its own legacy pool gets pruned.
        # See #100413.
        active_pool_keys = {
            str(key).strip().lower()
            for key in custom_provider_pool_key_candidates(base_url, provider_name=provider_name or None)}
        if not active_pool_keys:
            return
        pools = read_credential_pool(None)
        if not isinstance(pools, dict):
            return
        for pool_key, entries in pools.items():
            if (
                not isinstance(pool_key, str)
                or not pool_key.startswith(CUSTOM_POOL_PREFIX)
                or pool_key in active_pool_keys
                or not isinstance(entries, list)):
                continue
            retained = [e for e in entries if not (isinstance(e, dict) and e.get("source") == "model_config")]
            if len(retained) != len(entries):
                removed_ids = [
                    str(e["id"]) for e in entries
                    if isinstance(e, dict) and e.get("source") == "model_config" and e.get("id")]
                write_credential_pool(pool_key, retained, removed_ids=removed_ids)
    except Exception:
        return


def _curses_choice(title: str, rows: list, default_idx: int):
    """``_curses_prompt_choice`` index, or None when curses is unavailable (piped stdin,
    non-TTY) so the caller can fall back to a numbered prompt."""
    try:
        from hermes_cli.setup import _curses_prompt_choice
        return _curses_prompt_choice(title, rows, default_idx)
    except Exception:
        return None


def _radiolist(title: str, items: list, default_idx: int = 0, **kw):
    """``curses_radiolist`` index (-1 = cancelled), or None when curses is unavailable so the
    caller can fall back to a numbered prompt."""
    try:
        from hermes_cli.curses_ui import curses_radiolist
        return curses_radiolist(title, items, selected=default_idx, cancel_returns=-1, **kw)
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        return None


def _print_numbered(title: str, rows: list, default_idx: int) -> None:
    print(title)
    for i, row in enumerate(rows, 1):
        marker = "→" if (i - 1) == default_idx else " "
        print(f"  {marker} {i}. {row}")


def _prompt_auth_credentials_choice(title: str) -> str:
    """Prompt for reuse / reauthenticate / cancel with the standard radio UI.

    Returns one of ``"use"``, ``"reauth"``, ``"cancel"``. Falls back to a
    numbered prompt when curses is unavailable (piped stdin, non-TTY).
    """
    choices = ["Use existing credentials", "Reauthenticate (new OAuth login)", "Cancel"]
    idx = _curses_choice(title, choices, 0)
    if idx is not None and idx >= 0:
        print()
        return ("use", "reauth", "cancel")[idx]
    _print_numbered(title, choices, 0)
    print()
    choice = _ask("  Choice [1/2/3]: ", raw=True, cancel_msg=None, on_cancel="1")
    return {"2": "reauth", "3": "cancel"}.get(choice, "use")


# ---------------------------------------------------------------------------
# Smart model routing (optional post-selection step)
# ---------------------------------------------------------------------------
#: Providers whose model setup offers the routing step today.  The helper below
#: is provider-agnostic; this tuple is the only OpenCode-specific gate, so
#: another provider joins by adding its id here (and nothing else changes).
ROUTING_SETUP_PROVIDERS = ("opencode-zen", "opencode-go")


def routing_setup_applies(provider_id: str) -> bool:
    """True when smart model routing can be offered for *provider_id*."""
    return str(provider_id or "").strip().lower() in ROUTING_SETUP_PROVIDERS


def _choice_index(title: str, rows: list, default_idx: int = 0):
    """Radio (curses) or numbered fallback → index, *default_idx* on empty answer,
    None on cancel — same conventions as ``_prompt_auth_credentials_choice``."""
    idx = _curses_choice(title, rows, default_idx)
    if idx is not None:
        print()
        return idx if idx >= 0 else None
    _print_numbered(title, rows, default_idx)
    print()
    answer = _ask("  Choice: ", raw=True, cancel_msg=None, on_cancel="")
    if not answer:
        return default_idx
    try:
        idx = int(answer) - 1
    except ValueError:
        return default_idx
    return idx if 0 <= idx < len(rows) else default_idx


def _multi_pick_models(model_list, *, preselected=(), prompt: str = "Add model: ") -> list:
    """Ordered subset of *model_list* picked one at a time ("add another" loop).

    Built on the existing single-model picker (``_pick_model_or_prompt`` →
    ``_prompt_model_selection``: curses radio, numbered fallback, search, custom
    name), so it inherits that picker's UI and needs no new picker framework.
    The user picks a model per pass and ends the loop with the picker's own
    Skip/Ctrl-C; already-picked models drop out of the next pass.

    Provider-agnostic: it only needs ids, so any provider can adopt it.  Each
    pick is validated against *model_list* and rejected with a clear message
    when it is not a known model.
    """
    known = [str(m) for m in model_list]
    chosen: list = []
    # An existing routing block's models are preserved even when today's discovery
    # no longer offers them: re-running setup must never silently drop a choice.
    for model in preselected or ():
        model = str(model).strip()
        if model and model not in chosen:
            chosen.append(model)
    while True:
        remaining = [m for m in known if m not in chosen]
        if not remaining:
            break
        if chosen:
            print(f"  Selected so far ({len(chosen)}): {', '.join(chosen)}")
        picked = _pick_model_or_prompt(
            remaining, prompt, current_model=chosen[-1] if chosen else "")
        if not picked:
            break
        picked = str(picked).strip()
        if picked not in known:
            print(f"  '{picked}' is not one of {len(known)} models offered here — not added.")
            continue
        if picked in chosen:
            print(f"  '{picked}' is already selected.")
            continue
        chosen.append(picked)
    return chosen


def _print_routing_readback(result: dict) -> None:
    """Report the routing block *as read back from disk* (never our own optimism)."""
    settings = (result or {}).get("settings") or {}
    models = list(settings.get("models") or [])
    state = "enabled" if settings.get("enabled") else "disabled"
    print(f"  ✓ Smart model routing {state}"
          f" — {'selected models only' if settings.get('preferred_models_only') else 'all eligible models'}")
    if models:
        print(f"    models ({len(models)}): {', '.join(models)}")
    print(f"    saved in: {result.get('path') or 'config.yaml'}")
    print("    verify with: hermes model routing status")


def offer_model_routing(*, provider_id: str, model_list=(), current_model: str = "",
                        existing_block=None) -> dict | None:
    """Optional smart-model-routing step, run directly after the model choice.

    Offered only when :func:`routing_setup_applies` holds, only in an
    interactive stdin, and only when there is something to route across.  The
    default answer is **skip**: nothing is written and ``config.yaml`` stays
    byte-identical.  Any write goes through ``agent.model_router.update_routing``
    (validate → atomic save → read back), so an invalid choice writes nothing.

    An existing ``smart_model_routing`` block is never reset: the first prompt
    offers to keep it, and a change starts from the models already listed.

    Returns the block read back from disk, or None when nothing was written.
    """
    if not routing_setup_applies(provider_id):
        return None
    from hermes_cli.setup import is_interactive_stdin
    if not is_interactive_stdin():
        # Non-interactive / CI setup must not block or touch routing config.
        return None
    try:
        from agent import model_router as router
        from hermes_cli.config import load_config
    except Exception:  # pragma: no cover - setup must never die here
        return None
    try:
        raw_block = (load_config() or {}).get("smart_model_routing")
    except Exception:  # pragma: no cover - defensive
        raw_block = None
    if existing_block is not None:
        raw_block = existing_block
    settings = router.routing_settings()
    existing_models = [str(m) for m in settings.get("models") or []]
    has_block = isinstance(raw_block, dict) and bool(raw_block)
    model_list = [str(m) for m in (model_list or []) if str(m).strip()]

    print()
    print("  Smart model routing (optional)")
    print("  Routes each request to the model that fits it — deterministic, no extra API call.")

    if has_block:
        state = "on" if settings.get("enabled") else "off"
        print(f"  Currently {state}"
              f"{f' with {len(existing_models)} model(s)' if existing_models else ''}.")
        rows = ["Keep the current routing configuration (default)",
                "Change the routing configuration"]
        if settings.get("enabled"):
            rows.append("Turn smart model routing off")
        idx = _choice_index("Smart model routing", rows, 0)
        if idx is None or idx == 0:
            print("  Keeping the current routing configuration.")
            return None
        if idx == len(rows) - 1 and settings.get("enabled"):
            try:
                result = router.update_routing({"enabled": False})
            except ValueError as exc:
                print(f"  ✗ Smart model routing not changed: {exc}")
                return None
            print("  Smart model routing turned off.")
            _print_routing_readback(result)
            return result
        mode_rows = ["Route across all eligible models",
                     "Choose the models routing may use"]
        default_mode = 1 if settings.get("preferred_models_only") else 0
        mode_idx = _choice_index("Routing mode", mode_rows, default_mode)
        if mode_idx is None:
            print("  Nothing changed.")
            return None
        mode_all = mode_idx == 0
    else:
        if len(model_list) < 2:
            print(f"  Only {len(model_list)} model(s) discovered for {provider_id}; routing needs at"
                  " least two, so there is nothing to choose between. Left off.")
            return None
        rows = ["Skip — keep one model for every request (default)",
                "Route across all eligible models",
                "Choose the models routing may use"]
        idx = _choice_index("Smart model routing", rows, 0)
        if idx is None or idx == 0:
            print("  Smart model routing left off — nothing changed.")
            return None
        mode_all = idx == 1

    if mode_all:
        changes = {"enabled": True, "preferred_models_only": False}
    else:
        if not model_list:
            print("  No model list available for this provider — routing left unchanged.")
            return None
        print("  Pick the models routing may use, one at a time; choose Skip when done.")
        unknown = [m for m in existing_models if m not in model_list]
        if unknown:
            print(f"  Keeping {', '.join(unknown)} from the current configuration"
                  f" (not in today's {provider_id} list).")
        picked = _multi_pick_models(model_list, preselected=existing_models,
                                    prompt="Add routing model: ")
        if len(picked) < 2:
            print(f"  Smart model routing needs at least two models — {len(picked)} selected,"
                  " so nothing was saved.")
            return None
        problems = router.validate_routing(
            {"enabled": True, "preferred_models_only": True, "models": picked})
        if problems:
            print("  ✗ Smart model routing not saved: " + "; ".join(problems))
            return None
        changes = {"enabled": True, "preferred_models_only": True, "models": picked}

    try:
        result = router.update_routing(changes)
    except ValueError as exc:
        print(f"  ✗ Smart model routing not saved: {exc}")
        return None
    _print_routing_readback(result)
    return result
