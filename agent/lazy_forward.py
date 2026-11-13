"""Lazy forwarding helpers shared by the AIAgent façade and its mixins.

Targets are resolved on every call, so ``patch("<module>.<name>")`` in tests still
intercepts and the heavy agent modules stay off the ``run_agent`` import path.
"""
import importlib


def lazy_attr(module: str, name: str):
    """Resolve ``module.name`` at call time."""
    return getattr(importlib.import_module(module), name)


def forward(module: str, name: str, *, static: bool = False):
    """Build an AIAgent method that lazily forwards to ``module.name``
    (``target(self, *args, **kwargs)``; ``static=True`` drops ``self``)."""
    if static:
        def forwarder(*args, **kwargs):
            return lazy_attr(module, name)(*args, **kwargs)
    else:
        def forwarder(self, *args, **kwargs):
            return lazy_attr(module, name)(self, *args, **kwargs)
    forwarder.__name__ = forwarder.__qualname__ = name
    forwarder.__doc__ = f"Forwarder — see ``{module}.{name}``."
    # The bare ``*args, **kwargs`` signature lies to introspection. Callers that decide what to pass
    # by reading the target's parameters (the router does exactly this before switching a live agent)
    # see VAR_KEYWORD, conclude the target takes anything, and hand it a keyword it never accepted:
    # "switch_model() got an unexpected keyword argument 'user_initiated'", after which routing
    # disables itself for the turn and the model the operator asked for is never used. Exposing the
    # real signature through ``__wrapped__`` (which ``inspect.signature`` follows, stripping ``self``
    # from a bound method) makes every lazily forwarded method tell the truth.
    try:
        setattr(forwarder, "__wrapped__", lazy_attr(module, name))
    except Exception:  # pragma: no cover - a module that cannot import yet still forwards at call time
        pass
    return staticmethod(forwarder) if static else forwarder


def forward_static(module: str, name: str):
    return forward(module, name, static=True)
