"""K-069: the shipped `config/logging.json` routes each (logger, level) to exactly one handler,
or to none, after the app's own import side effects.

The start scripts and the Dockerfile pass that file as uvicorn's `--log-config`. uvicorn applies it
**before** it imports `falkorchat.app`, and that import builds `FastMCP(...)`, which calls
`logging.basicConfig(level=INFO, format="%(message)s")`. So the test applies the config first and
imports the app second, the same order uvicorn uses, then checks two things:

- **Structure.** For every (logger, level) cell, the number of handlers a record reaches. 1 means
  printed once and 0 means dropped. A `propagate` slip shows up as 2, a missing handler as 0.
- **Rendering.** Real records are emitted and the captured streams are read back: each line
  printed exactly once, on the right stream, with an ISO 8601 timestamp + offset and a level.

The test runs in a **subprocess** because `dictConfig` is process-global. Applied in-process,
`falkorchat.propagate: false` would blind every `caplog` test that runs after it.

It needs no FalkorDB. The child gets a clean env with `FALKORDB_PORT=1` and every optional
subsystem off. Importing the app opens no connection (`db.LazyFalkorDB`), and the port proves it:
a connection attempt there fails at once instead of reaching a real instance.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

LOG_CONFIG = Path(__file__).resolve().parents[2] / "config" / "logging.json"

_CHILD = r"""
import json, logging, logging.config, sys
logging.config.dictConfig(json.load(open(sys.argv[1])))
import falkorchat.app  # noqa: F401  -- FastMCP's basicConfig runs here, as under uvicorn

D, I, W = logging.DEBUG, logging.INFO, logging.WARNING
CELLS = [
    ("falkorchat.retry", D), ("falkorchat.retry", I), ("falkorchat.retry", W),
    ("uvicorn", I), ("uvicorn.error", I), ("uvicorn.error", W), ("uvicorn.access", I),
    ("httpx", I), ("httpx", W), ("mcp.server.lowlevel.server", I), ("root", W),
]

def reach(name, level):
    lg = logging.getLogger(None if name == "root" else name)
    if not lg.isEnabledFor(level):
        return 0
    n, c = 0, lg
    while c is not None:
        n += sum(1 for h in c.handlers if level >= h.level)
        if not c.propagate:
            break
        c = c.parent
    if n == 0 and logging.lastResort is not None and level >= logging.lastResort.level:
        return 1  # Python's lastResort handler would print it, bare
    return n

with open(sys.argv[2], "w") as f:
    json.dump({f"{n}@{logging.getLevelName(l)}": reach(n, l) for n, l in CELLS}, f)

logging.getLogger("falkorchat.retry").debug("K069-FC-DEBUG")
logging.getLogger("falkorchat.retry").info("K069-FC-INFO")
logging.getLogger("falkorchat.retry").warning("K069-FC-WARN")
logging.getLogger("uvicorn.error").info("K069-UV-INFO")
logging.getLogger("uvicorn.access").info(
    '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5", "GET", "/K069-ACCESS", "1.1", 200
)
logging.getLogger("httpx").info("K069-TP-INFO")
logging.getLogger("httpx").warning("K069-TP-WARN")
logging.getLogger("mcp.server.lowlevel.server").info("K069-MCP-INFO")
"""

EXPECTED_REACH = {
    "falkorchat.retry@DEBUG": 0,
    "falkorchat.retry@INFO": 1,
    "falkorchat.retry@WARNING": 1,
    "uvicorn@INFO": 1,
    "uvicorn.error@INFO": 1,
    "uvicorn.error@WARNING": 1,
    "uvicorn.access@INFO": 1,
    "httpx@INFO": 0,  # third-party INFO is dropped on purpose (root at WARNING)
    "httpx@WARNING": 1,
    "mcp.server.lowlevel.server@INFO": 0,  # mcp's per-request lines, dropped on purpose
    "root@WARNING": 1,
}

# marker -> (stream it must appear on exactly once, required text after the level)
PRINTED = {
    "K069-FC-INFO": ("stderr", "INFO:     falkorchat.retry: K069-FC-INFO"),
    "K069-FC-WARN": ("stderr", "WARNING:  falkorchat.retry: K069-FC-WARN"),
    "K069-UV-INFO": ("stderr", "INFO:     K069-UV-INFO"),
    "K069-ACCESS": ("stdout", 'INFO:     127.0.0.1:5 - "GET /K069-ACCESS HTTP/1.1" 200'),
    "K069-TP-WARN": ("stderr", "WARNING:  httpx: K069-TP-WARN"),
}
DROPPED = ("K069-FC-DEBUG", "K069-TP-INFO", "K069-MCP-INFO")
_TS = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d ")


def _run_child(tmp_path: Path) -> tuple[dict, str, str]:
    out = tmp_path / "reach.json"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "FALKORDB_HOST": "127.0.0.1",
        "FALKORDB_PORT": "1",
        "FALKORCHAT_WS_ID": "logcfg-probe",
        "FALKORCHAT_ENABLE_AGENT": "0",
        "FALKORCHAT_WORKFLOW_ENABLED": "0",
        "FALKORCHAT_STOREFRONT_ENABLED": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(LOG_CONFIG), str(out)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text()), proc.stdout, proc.stderr


def test_shipped_log_config_routes_each_logger_level_to_exactly_one_handler_or_none(tmp_path):
    reach, stdout, stderr = _run_child(tmp_path)

    assert reach == EXPECTED_REACH

    streams = {"stdout": stdout.splitlines(), "stderr": stderr.splitlines()}
    for marker, (stream, text) in PRINTED.items():
        other = "stdout" if stream == "stderr" else "stderr"
        hits = [ln for ln in streams[stream] if marker in ln]
        assert len(hits) == 1, f"{marker}: {len(hits)} lines on {stream}: {hits}"
        assert not [ln for ln in streams[other] if marker in ln], f"{marker} leaked to {other}"
        assert _TS.match(hits[0]), f"{marker}: no ISO 8601 + offset timestamp: {hits[0]!r}"
        assert hits[0][_TS.match(hits[0]).end():].startswith(text), hits[0]
    for marker in DROPPED:
        assert marker not in stdout + stderr, f"{marker} should be dropped"
