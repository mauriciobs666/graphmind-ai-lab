#!/usr/bin/env python3
"""The `fake-test` MCP server — mcp-monitor's own FR-12 genericity-proving server.

A minimal stdio server, one tool (`get_status`), built purpose-built for
mcp-monitor's tests and demo — not an existing repo component (FR-12). It is
deliberately **stdio**, where falkor-chat's is **HTTP**: exercising both
transports, not just a different tool signature, is a strictly stronger proof
that mcp-monitor's client layer (`mcp_monitor/client.py`) is transport-agnostic
(plan §9).

`get_status()` reads whatever was last written to a small JSON state file
(path from `FAKE_MCP_STATE_FILE`, `{"value": "idle"}` if the file is absent or
unreadable) on every call — so a test or the demo can externally,
deterministically flip what the *next* poll sees via `set_state.py`, without
any FalkorDB, network, or external process beyond this one.

Modeled on `cpg/mcp/server.py`'s mechanics (a `FastMCP` stdio server built the
same way falkor-chat's own MCP tools are), not its content — this server's
only job is to read one file and hand back its contents.

Nothing here writes to stdout: the stdio transport owns it. `mcp.run()`
handles that itself; this module adds no `print()`s of its own.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

#: Same env var name `mcp_monitor`'s example config and `set_state.py` use.
STATE_FILE_ENV = "FAKE_MCP_STATE_FILE"

#: Used both when the env var is unset and when the file itself is missing —
#: a fresh fake server with nothing written yet reports "idle" either way.
DEFAULT_STATE_PATH = "mcp-monitor-fake-state.json"
DEFAULT_STATE: dict[str, str] = {"value": "idle"}


def state_file_path() -> Path:
    return Path(os.environ.get(STATE_FILE_ENV) or DEFAULT_STATE_PATH)


def read_state() -> dict:
    """Read the state file, tolerating "not written yet" and "corrupt" alike.

    Never raises: a tool body that raises kills a stdio session that is not
    reconnected mid-session (same posture as `cpg/mcp/server.py`).
    """
    path = state_file_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return dict(DEFAULT_STATE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return dict(DEFAULT_STATE)


mcp = FastMCP(
    name="fake-test",
    instructions=(
        "Test/demo-only MCP server for mcp-monitor (FR-12/AC-4). One tool, "
        "get_status, returns whatever set_state.py last wrote to "
        f"${STATE_FILE_ENV}."
    ),
)


@mcp.tool()
def get_status() -> dict:
    """Return whatever was last written to `$FAKE_MCP_STATE_FILE`."""
    return read_state()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
