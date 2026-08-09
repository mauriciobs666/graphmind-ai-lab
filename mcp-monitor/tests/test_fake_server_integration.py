"""Component tier: a REAL `fake_mcp_server/server.py` stdio subprocess and a
real mcp-monitor watch loop (`mcp_monitor.watch.run_watch` +
`mcp_monitor.client.ServerConnection`, not stubs) — plan §11 tier 3. State is
driven via `fake_mcp_server/set_state.py`'s `write_state` helper; launched
commands are real `python3` invocations of a small script (built inline, in
`tmp_path`) that appends its start timestamp to a marker file, so overlapping
timestamps prove AC-6's parallelism and repeated READY flips prove AC-5's
on/off dedupe switch.

No FalkorDB, no LM Studio, no Docker — safe to run in the default suite.
Interval seconds are kept small (~0.2-0.5s) to bound wall-clock cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import pytest

from mcp_monitor.client import ServerConnection
from mcp_monitor.config import ServerConfig, WatchConfig
from mcp_monitor.launcher import launch
from mcp_monitor.watch import run_watch

pytestmark = pytest.mark.anyio

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_SERVER = REPO_ROOT / "fake_mcp_server" / "server.py"
SET_STATE = REPO_ROOT / "fake_mcp_server" / "set_state.py"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _write_state(state_file: Path, value: str) -> None:
    state_file.write_text(json.dumps({"value": value}), encoding="utf-8")


def _fake_server_connection(state_file: Path) -> ServerConnection:
    config = ServerConfig(
        name="fake-test",
        transport="stdio",
        command=sys.executable,
        args=[str(FAKE_SERVER)],
        env={"FAKE_MCP_STATE_FILE": str(state_file)},
    )
    return ServerConnection("fake-test", config)


def _marker_launch_script(tmp_path: Path, marker: Path) -> Path:
    """A tiny real script the launched command execs: appends a start
    timestamp (with microsecond precision) to `marker`, one line per launch —
    used to prove AC-6 (concurrent launches interleave, not serialize)."""
    script = tmp_path / "marker_launch.py"
    script.write_text(
        "import sys, time, json\n"
        "payload = json.loads(sys.stdin.read())\n"
        f"with open({str(marker)!r}, 'a') as f:\n"
        "    f.write(f\"{time.time()!r} {payload['matched_text']}\\n\")\n"
    )
    return script


async def test_ac4_second_server_genericity_watch_fires_on_state_flip(tmp_path):
    """AC-4: a watch against the fake (second, distinct) MCP server fires when
    its own pattern matches — the same mcp-monitor machinery (client, watch,
    launcher) that also drives the falkor-chat watch, proving genericity."""
    state_file = tmp_path / "state.json"
    _write_state(state_file, "idle")
    marker = tmp_path / "marker.log"
    marker.write_text("")
    script = _marker_launch_script(tmp_path, marker)

    connection = _fake_server_connection(state_file)
    watch = WatchConfig(
        name="fake-genericity",
        server="fake-test",
        tool="get_status",
        args={},
        interval_seconds=0.2,
        pattern=re.compile(r'"value":\s*"READY"'),
        repeat_trigger=False,
        command=[sys.executable, str(script)],
    )
    logger = logging.getLogger("test-ac4")

    task = asyncio.create_task(
        run_watch(watch, connection, server_name="fake-test", logger=logger, launch=launch)
    )
    try:
        await asyncio.sleep(0.3)  # one poll while still idle: no match
        assert marker.read_text() == ""

        _write_state(state_file, "READY")
        # wait for a poll to see it and the launched command to finish
        for _ in range(50):
            if marker.read_text().strip():
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("watch never fired within the timeout")
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await connection.aclose()

    lines = marker.read_text().strip().splitlines()
    assert len(lines) == 1
    assert '"value": "READY"' in lines[0]


async def test_ac5_repeat_trigger_off_fires_once_then_on_fires_again(tmp_path):
    """AC-5: repeat_trigger off suppresses a second identical match; a fresh
    watch (fresh dedupe state) with repeat_trigger on fires again on the same
    text — the escape hatch for a watch that wants every occurrence."""
    state_file = tmp_path / "state.json"
    _write_state(state_file, "READY")
    marker_off = tmp_path / "marker_off.log"
    marker_off.write_text("")
    script = _marker_launch_script(tmp_path, marker_off)

    connection = _fake_server_connection(state_file)
    watch_off = WatchConfig(
        name="off",
        server="fake-test",
        tool="get_status",
        args={},
        interval_seconds=0.15,
        pattern=re.compile(r'"value":\s*"READY"'),
        repeat_trigger=False,
        command=[sys.executable, str(script)],
    )
    logger = logging.getLogger("test-ac5-off")

    task = asyncio.create_task(
        run_watch(watch_off, connection, server_name="fake-test", logger=logger, launch=launch)
    )
    # Several poll cycles, all seeing the same "READY" text.
    await asyncio.sleep(0.9)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await connection.aclose()

    off_lines = [line for line in marker_off.read_text().splitlines() if line.strip()]
    assert len(off_lines) == 1, "repeat_trigger=false must fire at most once on identical text"

    # A separate watch (fresh dedupe state) with repeat_trigger on, against the
    # same still-READY state, fires on every poll.
    marker_on = tmp_path / "marker_on.log"
    marker_on.write_text("")
    script_on = _marker_launch_script(tmp_path, marker_on)
    connection2 = _fake_server_connection(state_file)
    watch_on = WatchConfig(
        name="on",
        server="fake-test",
        tool="get_status",
        args={},
        interval_seconds=0.15,
        pattern=re.compile(r'"value":\s*"READY"'),
        repeat_trigger=True,
        command=[sys.executable, str(script_on)],
    )
    logger2 = logging.getLogger("test-ac5-on")
    task2 = asyncio.create_task(
        run_watch(watch_on, connection2, server_name="fake-test", logger=logger2, launch=launch)
    )
    await asyncio.sleep(0.9)
    task2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task2
    await connection2.aclose()

    on_lines = [line for line in marker_on.read_text().splitlines() if line.strip()]
    assert len(on_lines) >= 3, "repeat_trigger=true must fire again on every poll"


async def test_ac6_parallel_launches_do_not_block_each_other(tmp_path):
    """AC-6: a match fires a new launch in parallel with an already-running
    one — proven by a slow launched command whose own start/finish timestamps
    interleave with a second match's launch rather than serializing."""
    state_file = tmp_path / "state.json"
    _write_state(state_file, "idle")
    marker = tmp_path / "marker.log"
    marker.write_text("")

    # A launched script that sleeps before recording, so a second, faster
    # match launched while the first is still "running" proves overlap.
    slow_script = tmp_path / "slow_launch.py"
    slow_script.write_text(
        "import sys, time, json\n"
        "start = time.time()\n"
        "payload = json.loads(sys.stdin.read())\n"
        "time.sleep(0.5)\n"
        f"with open({str(marker)!r}, 'a') as f:\n"
        "    f.write(f\"{start!r} {time.time()!r} {payload['matched_text']}\\n\")\n"
    )

    connection = _fake_server_connection(state_file)
    watch = WatchConfig(
        name="parallel",
        server="fake-test",
        tool="get_status",
        args={},
        interval_seconds=0.2,
        # Every non-overlapping match in one poll's result is a candidate —
        # two distinguishable values in the state text both match a
        # value-agnostic pattern, firing two launches from the SAME poll.
        pattern=re.compile(r'"value":\s*"\w+"'),
        repeat_trigger=True,
        command=[sys.executable, str(slow_script)],
    )
    logger = logging.getLogger("test-ac6")

    task = asyncio.create_task(
        run_watch(watch, connection, server_name="fake-test", logger=logger, launch=launch)
    )
    try:
        _write_state(state_file, "GO")
        # Two poll cycles, each firing one (slow, 0.5s) launch in parallel.
        await asyncio.sleep(1.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await connection.aclose()

    lines = [line for line in marker.read_text().splitlines() if line.strip()]
    assert len(lines) >= 2, "expected at least two launched commands to complete"

    # Parse (start, end) pairs and check at least two overlap in wall-clock
    # time — proof the watch loop did not wait for one launch before starting
    # the next (FR-9/AC-6).
    intervals = []
    for line in lines:
        start_s, end_s, _ = line.split(" ", 2)
        intervals.append((float(start_s), float(end_s)))

    def overlaps(a, b) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    assert any(
        overlaps(intervals[i], intervals[j])
        for i in range(len(intervals))
        for j in range(i + 1, len(intervals))
    ), f"no two launches overlapped: {intervals}"
