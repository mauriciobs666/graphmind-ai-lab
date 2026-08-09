"""Component tier: the full poll -> match -> dedupe -> launch loop, driven
against an in-memory stub MCP server via
`mcp.shared.memory.create_connected_server_and_client_session` — the same
technique `falkor-chat/server/tests/test_mcp_client.py` already uses to verify
client-side logic without a real transport (plan §11 tier 2). No real process,
no real socket, deterministic.
"""

from __future__ import annotations

import asyncio
import logging
import re

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_monitor.client import ServerConnection
from mcp_monitor.config import ServerConfig, WatchConfig
from mcp_monitor.watch import run_watch

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ── a stub in-memory MCP server whose one tool's answer we control ──────────


class _StubToolServer:
    """A tiny FastMCP server whose `get_status` tool returns a value we flip
    from the test, mirroring `fake_mcp_server`'s state-file trick but entirely
    in-memory (no subprocess, no file)."""

    def __init__(self) -> None:
        self.value = "idle"
        self.mcp = FastMCP("stub-status-server")

        @self.mcp.tool()
        def get_status() -> dict:
            return {"value": self.value}


def _connection_for(stub: _StubToolServer) -> ServerConnection:
    config = ServerConfig(name="stub", transport="http", url="http://unused")
    return ServerConnection(
        "stub",
        config,
        session_cm_factory=lambda: create_connected_server_and_client_session(stub.mcp),
    )


def _watch(**overrides) -> WatchConfig:
    defaults = dict(
        name="w1",
        server="stub",
        tool="get_status",
        args={},
        interval_seconds=0.01,
        pattern=re.compile(r'"value":\s*"READY"'),
        repeat_trigger=False,
        command=["true"],
    )
    defaults.update(overrides)
    return WatchConfig(**defaults)


class _RecordingLauncher:
    """Stands in for `launcher.launch` — records every call instead of
    spawning a real subprocess."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []
        self.event = asyncio.Event()

    async def __call__(self, watch, server_name, raw_result, matched_text, *, logger):
        self.calls.append((watch.name, server_name, raw_result, matched_text))
        self.event.set()


async def _run_until(coro_task, event: asyncio.Event, *, timeout: float = 2.0) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    finally:
        coro_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await coro_task


# ── tests ─────────────────────────────────────────────────────────────────


async def test_watch_fires_when_poll_result_matches_pattern():
    stub = _StubToolServer()
    stub.value = "READY"
    connection = _connection_for(stub)
    watch = _watch()
    recorder = _RecordingLauncher()
    logger = logging.getLogger("test")

    task = asyncio.create_task(
        run_watch(watch, connection, server_name="stub", logger=logger, launch=recorder)
    )
    await _run_until(task, recorder.event)

    assert len(recorder.calls) == 1
    name, server_name, raw_result, matched_text = recorder.calls[0]
    assert name == "w1"
    assert server_name == "stub"
    assert '"READY"' in raw_result
    assert matched_text == '"value": "READY"'

    await connection.aclose()


async def test_watch_does_not_fire_when_no_match():
    stub = _StubToolServer()
    stub.value = "idle"
    connection = _connection_for(stub)
    watch = _watch(interval_seconds=0.01)
    recorder = _RecordingLauncher()
    logger = logging.getLogger("test")

    task = asyncio.create_task(
        run_watch(watch, connection, server_name="stub", logger=logger, launch=recorder)
    )
    # Give it a handful of poll cycles, then confirm nothing fired.
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert recorder.calls == []
    await connection.aclose()


async def test_repeat_trigger_off_suppresses_second_identical_match(caplog):
    """AC-5 (off branch), driven within one process run (plan §6's open
    question: dedupe state is in-memory only — see docs/BACKLOG.md)."""
    stub = _StubToolServer()
    stub.value = "READY"
    connection = _connection_for(stub)
    watch = _watch(interval_seconds=0.01, repeat_trigger=False)
    recorder = _RecordingLauncher()
    logger = logging.getLogger("test")

    task = asyncio.create_task(
        run_watch(watch, connection, server_name="stub", logger=logger, launch=recorder)
    )
    await _run_until(task, recorder.event)
    # Let a few more poll cycles happen; the same "READY" text keeps matching
    # but must not re-fire since repeat_trigger is False.
    await asyncio.sleep(0.1)

    assert len(recorder.calls) == 1
    await connection.aclose()


async def test_repeat_trigger_on_fires_every_time():
    """AC-5 (on branch)."""
    stub = _StubToolServer()
    stub.value = "READY"
    connection = _connection_for(stub)
    watch = _watch(interval_seconds=0.01, repeat_trigger=True)
    recorder = _RecordingLauncher()
    logger = logging.getLogger("test")

    task = asyncio.create_task(
        run_watch(watch, connection, server_name="stub", logger=logger, launch=recorder)
    )
    # Wait for several matches to accumulate.
    while len(recorder.calls) < 3:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(recorder.calls) >= 3
    await connection.aclose()


async def test_poll_failure_is_logged_and_watch_keeps_running(caplog):
    """FR-10/AC-7: a poll failure (tool call raises) is logged and the watch
    continues — it must not crash the loop or stop polling."""

    class _RaisingServer:
        def __init__(self) -> None:
            self.mcp = FastMCP("raising-server")

            @self.mcp.tool()
            def get_status() -> dict:
                raise RuntimeError("boom")

    stub = _RaisingServer()
    config = ServerConfig(name="stub", transport="http", url="http://unused")
    connection = ServerConnection(
        "stub",
        config,
        session_cm_factory=lambda: create_connected_server_and_client_session(stub.mcp),
    )
    watch = _watch(interval_seconds=0.01)
    recorder = _RecordingLauncher()
    logger = logging.getLogger("test-poll-failure")

    caplog.set_level(logging.WARNING, logger="test-poll-failure")
    task = asyncio.create_task(
        run_watch(watch, connection, server_name="stub", logger=logger, launch=recorder)
    )
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert recorder.calls == []
    assert any("poll failed" in record.message for record in caplog.records)
    await connection.aclose()
