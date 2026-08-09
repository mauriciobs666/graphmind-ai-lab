"""Subprocess launch: stdin JSON + env identifiers, exit-code logging (plan §5/§7).

The payload travels **only** on stdin, as a single JSON document; three env
vars (`MCP_MONITOR_WATCH_NAME`/`_SERVER_NAME`/`_TOOL_NAME`) carry small
identifiers as a convenience for commands that don't want to parse JSON. The
raw result and matched text are deliberately not duplicated into argv or env
— argv length caps, shell-quoting of arbitrary matched content, and
`ps`-visible env leakage are all avoided by keeping content on stdin only.

`launch()` is the whole body of the supervisory task `watch.py` schedules per
match (`asyncio.create_task(launch(...))`, never awaited by the watch loop
itself) — it spawns, feeds stdin, awaits completion, and logs the exit code,
so "triggered" alone isn't the only signal available for after-the-fact
debugging (FR-11).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from .config import WatchConfig

#: Injectable seam for `test_launcher.py` — stub this instead of monkeypatching
#: `asyncio.create_subprocess_exec` globally.
CreateSubprocessExec = Callable[..., Awaitable[asyncio.subprocess.Process]]


def build_payload(
    *, watch_name: str, server_name: str, tool_name: str, raw_result: str, matched_text: str
) -> dict[str, Any]:
    """The stdin JSON document (plan §5) — pure, so it's testable without a subprocess."""
    return {
        "watch": watch_name,
        "server": server_name,
        "tool": tool_name,
        "raw_result": raw_result,
        "matched_text": matched_text,
        "matched_at": _now_iso(),
    }


def build_env(
    *, watch_name: str, server_name: str, tool_name: str, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    """The launched process's environment: the parent's env plus three identifiers."""
    env = dict(base_env if base_env is not None else os.environ)
    env["MCP_MONITOR_WATCH_NAME"] = watch_name
    env["MCP_MONITOR_SERVER_NAME"] = server_name
    env["MCP_MONITOR_TOOL_NAME"] = tool_name
    return env


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def launch(
    watch: WatchConfig,
    server_name: str,
    raw_result: str,
    matched_text: str,
    *,
    logger: logging.LoggerAdapter | logging.Logger,
    create_subprocess_exec: CreateSubprocessExec = asyncio.create_subprocess_exec,
) -> None:
    """Launch `watch.command`, feed it the match payload on stdin, log its exit.

    `watch.command` is consumed as an **argv array** via
    `asyncio.create_subprocess_exec(*watch.command, ...)` — never a shell
    string (`shell=True` is not used). Nothing needs interpolating into the
    command line (content travels on stdin), and argv-exec sidesteps
    shell-quoting ambiguity entirely for whatever operator-chosen command
    runs. An operator wanting shell features puts `["bash", "-c", "..."]` in
    `command` themselves.

    A spawn failure (bad path, permission denied) is logged at `ERROR` and
    swallowed here — never raised out of this coroutine — since this runs as
    its own `asyncio.create_task(...)`, unattached to anything that awaits it
    for errors.
    """
    payload = build_payload(
        watch_name=watch.name,
        server_name=server_name,
        tool_name=watch.tool,
        raw_result=raw_result,
        matched_text=matched_text,
    )
    env = build_env(watch_name=watch.name, server_name=server_name, tool_name=watch.tool)

    try:
        proc = await create_subprocess_exec(
            *watch.command,
            stdin=asyncio.subprocess.PIPE,
            env=env,
        )
    except OSError as exc:
        logger.error("command failed to launch: command=%r error=%s", watch.command, exc)
        return

    stdin_bytes = json.dumps(payload).encode("utf-8")
    try:
        await proc.communicate(input=stdin_bytes)
    except Exception as exc:  # noqa: BLE001 — a supervisory task must never die silently
        logger.error("command %r raised while running: %s", watch.command, exc)
        return

    logger.info("command %r exited with code %s", watch.command, proc.returncode)
