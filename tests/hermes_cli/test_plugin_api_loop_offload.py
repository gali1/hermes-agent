"""Plugin API routes that are ``async def`` but never await must not run on the dashboard loop.

Captured (2026-11-12 17:09:38,858, logs/agent.log):

    WARNING hermes_cli.web_server: event loop stalled 12.1s (GIL pressure suspected)

Timeline evidence, same window:

* 17:09:16,191 ``plugins/hub rebuilt in 5.505s`` — a dashboard page load (2x the usual 2.7-2.9s).
* 17:09:25,466 ``plugins/memory-wiki/dashboard/__pycache__/memory_wiki_core.cpython-314.pyc`` is
  written — the first-ever import of memory-wiki's dashboard core, i.e. the first
  ``/api/plugins/memory-wiki/*`` request in that process.
* 17:09:38,858 the heartbeat (2s tick, 5s threshold) reports 12.1s of drift, so the loop was held
  from ~17:09:25.4 to 17:09:38.8.

The handler is the blocker: ``GET /overview`` is ``async def`` and calls
``build_memory_overview(db)`` inline, which scans a 348MB ``state.db`` — measured 10.657s on this
host for ``subject_limit=50, day_limit=30`` (and 5.809s for ``/subjects``).

The host cannot rewrite the plugin, and ``async def`` is not evidence of non-blocking, so the mount
point demotes the provable case: a coroutine handler with no await/async-for/async-with opcode has
nothing to suspend on, so re-registering it as a plain ``def`` moves it to Starlette's threadpool
without changing its semantics.
"""

from __future__ import annotations

import asyncio
import inspect
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx
from fastapi import APIRouter, FastAPI

from hermes_cli.web_server_dashboard import (
    _offload_blocking_plugin_routes,
    _route_is_blocking_sync_work,
)


# ── the detector ────────────────────────────────────────────────────────────────────────────────


async def _plain_blocking(limit: int = 3):  # async def, nothing to await
    return {"limit": limit}


async def _awaits():
    await asyncio.sleep(0)


async def _async_with():
    import contextlib

    async with contextlib.AsyncExitStack():
        pass


async def _async_for():
    async def _gen():
        yield 1

    async for _ in _gen():
        pass


def _sync_but_blocking():
    return {}


async def _takes_request(request):  # loop-owned object: must stay on the loop
    return {}


def test_detects_coroutine_without_suspension():
    assert _route_is_blocking_sync_work(_plain_blocking) is True


@pytest.mark.parametrize(
    "endpoint",
    [_awaits, _async_with, _async_for, _sync_but_blocking, _takes_request],
    ids=["awaits", "async-with", "async-for", "plain-def", "takes-request"],
)
def test_leaves_everything_else_alone(endpoint):
    assert _route_is_blocking_sync_work(endpoint) is False


# ── the mount-time rewrite ─────────────────────────────────────────────────────────────────────


def _build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/slow")
    async def slow(limit: int = 3):
        # Blocking work wearing ``async def`` — exactly memory-wiki's /overview shape.
        time.sleep(0.6)
        return {"limit": limit, "done": True}

    @router.get("/ping")
    async def ping():
        return {"ok": True}

    @router.get("/genuinely-async")
    async def genuinely_async():
        await asyncio.sleep(0)
        return {"ok": True}

    return router


def _probe(app: FastAPI, work_s: float = 0.6) -> tuple[float, int, dict]:
    """Fire /slow, then time /ping and count loop ticks while /slow is in flight."""

    async def _scenario():
        ticks = 0
        stop = asyncio.Event()

        async def _heartbeat():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.create_task(_heartbeat())
        await asyncio.sleep(0)  # let the heartbeat task actually start
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            started = time.perf_counter()
            slow = asyncio.create_task(client.get("/slow?limit=7"))
            # Measure from /slow's start: when the handler runs on the loop this await is itself
            # delayed, so timing only the /ping round-trip would read ~0 either way.
            await asyncio.sleep(0.05)
            ping = await client.get("/ping")
            ping_latency = time.perf_counter() - started
            slow_resp = await slow
        stop.set()
        await beat
        assert ping.status_code == 200
        return ping_latency, ticks, slow_resp.json()

    return asyncio.run(_scenario())


def test_offloaded_route_keeps_the_loop_serving_other_requests():
    """Control: mounted as-is the handler holds the loop; after the rewrite it does not."""
    before = FastAPI()
    before.include_router(_build_router())
    blocked_latency, blocked_ticks, _ = _probe(before)
    # The control proves the harness measures the stall at all: /ping could not be served until
    # /slow's body returned, so it waited out the rest of the sleep and the loop never ticked.
    assert blocked_latency > 0.3, f"/ping should have waited on the blocking body, took {blocked_latency:.3f}s"
    assert blocked_ticks <= 5, f"loop should be frozen, ticked {blocked_ticks}x"

    router = _build_router()
    moved = _offload_blocking_plugin_routes(router, "probe-plugin")
    # Both are coroutines that never await — /ping is trivial but still sync work on the loop.
    assert moved == ["GET /slow", "GET /ping"], moved

    after = FastAPI()
    after.include_router(router)
    free_latency, free_ticks, body = _probe(after)

    assert free_latency < 0.25, f"/ping should not wait on /slow, took {free_latency:.3f}s"
    assert free_ticks >= 8, f"loop should keep ticking, ticked {free_ticks}x"
    # The handler is unchanged: same body, same query-parameter resolution.
    assert body == {"limit": 7, "done": True}


def test_genuinely_async_routes_are_not_rewritten():
    """A handler that really awaits keeps its coroutine identity (and its loop)."""
    router = _build_router()
    _offload_blocking_plugin_routes(router, "probe-plugin")

    by_path = {route.path: route for route in router.routes}
    assert inspect.iscoroutinefunction(by_path["/genuinely-async"].endpoint)
    assert not inspect.iscoroutinefunction(by_path["/slow"].endpoint)
    assert not inspect.iscoroutinefunction(by_path["/ping"].endpoint)

    app = FastAPI()
    app.include_router(router)

    async def _call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return (await client.get("/genuinely-async")).json()

    assert asyncio.run(_call()) == {"ok": True}
