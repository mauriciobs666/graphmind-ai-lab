"""Unit tests for `mcp_monitor.launcher` — stdin/env payload shape, argv-exec,
exit-code logging. No real subprocess: `asyncio.create_subprocess_exec` is
stubbed via the injectable `create_subprocess_exec` parameter (plan §11 tier 1).
"""

from __future__ import annotations

import json

import pytest

from mcp_monitor.config import WatchConfig
from mcp_monitor.launcher import build_env, build_payload, launch

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _watch(**overrides) -> WatchConfig:
    import re

    defaults = dict(
        name="w1",
        server="s1",
        tool="t1",
        args={},
        interval_seconds=1.0,
        pattern=re.compile("x"),
        repeat_trigger=False,
        command=["echo", "hi"],
    )
    defaults.update(overrides)
    return WatchConfig(**defaults)


class _StubProcess:
    def __init__(self, returncode: int = 0, stdin_sink: list[bytes] | None = None):
        self.returncode = returncode
        self._stdin_sink = stdin_sink

    async def communicate(self, input: bytes | None = None):
        if self._stdin_sink is not None and input is not None:
            self._stdin_sink.append(input)
        return (b"", b"")


class _ListLogger:
    """Minimal logger stand-in recording (level, message %-formatted) tuples."""

    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def _record(self, level, msg, *args):
        self.records.append((level, msg % args if args else msg))

    def info(self, msg, *args):
        self._record("INFO", msg, *args)

    def warning(self, msg, *args):
        self._record("WARNING", msg, *args)

    def debug(self, msg, *args):
        self._record("DEBUG", msg, *args)

    def error(self, msg, *args):
        self._record("ERROR", msg, *args)


# ── pure payload/env builders ────────────────────────────────────────────


def test_build_payload_shape():
    payload = build_payload(
        watch_name="w1",
        server_name="s1",
        tool_name="t1",
        raw_result="the raw text",
        matched_text="matched",
    )
    assert payload["watch"] == "w1"
    assert payload["server"] == "s1"
    assert payload["tool"] == "t1"
    assert payload["raw_result"] == "the raw text"
    assert payload["matched_text"] == "matched"
    # matched_at is an ISO-8601 UTC timestamp, e.g. 2026-08-08T12:34:56Z
    assert payload["matched_at"].endswith("Z")


def test_build_env_adds_three_identifiers_without_dropping_base_env():
    env = build_env(watch_name="w1", server_name="s1", tool_name="t1", base_env={"PATH": "/bin"})
    assert env["PATH"] == "/bin"
    assert env["MCP_MONITOR_WATCH_NAME"] == "w1"
    assert env["MCP_MONITOR_SERVER_NAME"] == "s1"
    assert env["MCP_MONITOR_TOOL_NAME"] == "t1"


# ── launch() with a stubbed subprocess ───────────────────────────────────


async def test_launch_sends_payload_json_on_stdin_and_sets_env_vars():
    watch = _watch(command=["my-command", "--flag"])
    stdin_sink: list[bytes] = []
    captured_kwargs = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured_kwargs["args"] = args
        captured_kwargs["kwargs"] = kwargs
        return _StubProcess(returncode=0, stdin_sink=stdin_sink)

    logger = _ListLogger()
    await launch(
        watch,
        "s1",
        "raw result text",
        "matched bit",
        logger=logger,
        create_subprocess_exec=fake_create_subprocess_exec,
    )

    # argv-exec, exactly watch.command, never a shell string.
    assert captured_kwargs["args"] == ("my-command", "--flag")
    env = captured_kwargs["kwargs"]["env"]
    assert env["MCP_MONITOR_WATCH_NAME"] == "w1"
    assert env["MCP_MONITOR_SERVER_NAME"] == "s1"
    assert env["MCP_MONITOR_TOOL_NAME"] == "t1"

    assert len(stdin_sink) == 1
    payload = json.loads(stdin_sink[0])
    assert payload["raw_result"] == "raw result text"
    assert payload["matched_text"] == "matched bit"
    assert payload["watch"] == "w1"

    # Exit code logged at INFO.
    assert any(level == "INFO" and "code 0" in msg for level, msg in logger.records)


async def test_launch_logs_error_and_returns_when_spawn_fails():
    watch = _watch(command=["/no/such/binary"])

    async def failing_create_subprocess_exec(*args, **kwargs):
        raise OSError("no such file or directory")

    logger = _ListLogger()
    # Must not raise — a spawn failure is logged, not propagated (plan §7).
    await launch(
        watch,
        "s1",
        "raw",
        "matched",
        logger=logger,
        create_subprocess_exec=failing_create_subprocess_exec,
    )

    assert any(level == "ERROR" for level, _ in logger.records)


async def test_launch_logs_nonzero_exit_code():
    watch = _watch()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _StubProcess(returncode=17)

    logger = _ListLogger()
    await launch(
        watch,
        "s1",
        "raw",
        "matched",
        logger=logger,
        create_subprocess_exec=fake_create_subprocess_exec,
    )

    assert any(level == "INFO" and "code 17" in msg for level, msg in logger.records)
