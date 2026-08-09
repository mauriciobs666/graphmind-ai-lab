#!/usr/bin/env python3
"""Flip `fake_mcp_server`'s state file — the deterministic trigger for tests/demo.

Overwrites `$FAKE_MCP_STATE_FILE` (or `--file`) with `{"value": VALUE}`, so the
next `get_status()` poll sees a chosen value. Used to drive AC-4 (a distinct
watch fires on this second server), AC-5 (flip to the pattern's value, then
back, then again, to exercise `repeat_trigger` on/off), and manually while
developing a watch config.

    fake_mcp_server/set_state.py READY
    fake_mcp_server/set_state.py idle --file /tmp/mcp-monitor-fake-state.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# `set_state.py` is run by path (`fake_mcp_server/set_state.py`, mirroring
# `server.py`'s own launch convention), not installed as a package, so its
# sibling `server.py` is only importable once this directory is on sys.path.
# This must run *before* the `server` import below.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import DEFAULT_STATE_PATH, STATE_FILE_ENV  # noqa: E402


def write_state(value: str, *, path: Path) -> None:
    path.write_text(json.dumps({"value": value}), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Flip fake_mcp_server's state file.")
    parser.add_argument("value", help="written as {\"value\": VALUE}")
    parser.add_argument(
        "--file",
        default=None,
        help=f"state file path (default: ${STATE_FILE_ENV} or ./{DEFAULT_STATE_PATH})",
    )
    args = parser.parse_args(argv)

    path = Path(args.file or os.environ.get(STATE_FILE_ENV) or DEFAULT_STATE_PATH)
    write_state(args.value, path=path)
    print(f"wrote {{'value': {args.value!r}}} to {path}")


if __name__ == "__main__":
    main()
