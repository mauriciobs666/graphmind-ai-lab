"""`start_demo.sh`'s storefront turn-worker cap (K-078 RCA DEF-1, fix O1): the demo exports
`FALKORCHAT_STOREFRONT_TURN_WORKERS=2` to uvicorn by default, and a caller's own value wins.

The real script runs end-to-end in a throwaway copy of the `falkor-chat/` tree, the same
way `test_opencode_config_resolution.py` drives it: every callee (sibling scripts, the
venv's `pip`/`uvicorn`, `redis-cli`, `salesperson/build.sh`) is a stub, and the uvicorn
stub records the env it was exec'd with. No FalkorDB, no LM Studio, no network, no real
venv; never touches `reference`.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../falkor-chat
_SCRIPTS = _REPO_ROOT / "scripts"

_RECORD_ENV = f'''#!{sys.executable}
import json, os, sys
with open(os.environ["STUB_RECORD"], "a") as fh:
    fh.write(json.dumps({{"argv0": os.path.basename(sys.argv[0]),
                         "turn_workers": os.environ.get("FALKORCHAT_STOREFRONT_TURN_WORKERS")}}) + "\\n")
'''

# PONG for the readiness pings; echoing the args back satisfies step 5's Agent check,
# which greps the query output for the agent id (present in the CYPHER parameter).
_REDIS_CLI = '#!/bin/sh\necho PONG\necho "$*"\n'

_CALLEES = ("start_falkordb.sh", "ensure_demo_models.sh", "create_workspace.sh",
            "seed_demo.sh", "seed_catalog.sh", "verify_catalog.sh",
            "seed_salesperson.sh", "verify_salesperson.sh")


def _exe(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_demo(tmp: Path, **extra: str) -> tuple[subprocess.CompletedProcess, list[dict]]:
    root = tmp / "falkor-chat"
    (root / "scripts" / "shlib").mkdir(parents=True)
    shutil.copy2(_SCRIPTS / "start_demo.sh", root / "scripts" / "start_demo.sh")
    shutil.copy2(_SCRIPTS / "shlib" / "opencode_config.sh",
                 root / "scripts" / "shlib" / "opencode_config.sh")
    (root / "config").mkdir()
    (root / "config" / "logging.json").write_text("{}")
    for callee in _CALLEES:
        _exe(root / "scripts" / callee, _RECORD_ENV)
    venv = root / "server" / ".venv" / "bin"
    _exe(venv / "pip", _RECORD_ENV)
    _exe(venv / "uvicorn", _RECORD_ENV)
    spa = tmp / "salesperson"  # the script's default SALESPERSON_DIR (<repo-root>/salesperson)
    _exe(spa / "build.sh", "#!/bin/sh\nexit 0\n")
    (spa / "dist").mkdir()
    (spa / "dist" / "index.html").write_text("<html></html>")
    stubbin = tmp / "stubbin"
    _exe(stubbin / "redis-cli", _REDIS_CLI)
    record = tmp / "record.jsonl"
    env = {"HOME": str(tmp), "PATH": f"{stubbin}:/usr/bin:/bin",
           "STUB_RECORD": str(record),
           "FALKORCHAT_OPENCODE_CONFIG": "/explicit/opencode.json", **extra}
    out = subprocess.run(["bash", str(root / "scripts" / "start_demo.sh")], env=env,
                         capture_output=True, text=True, timeout=60)
    calls = ([json.loads(line) for line in record.read_text().splitlines()]
             if record.exists() else [])
    return out, calls


def _uvicorn_turn_workers(out, calls) -> str | None:
    seen = [c["turn_workers"] for c in calls if c["argv0"] == "uvicorn"]
    assert seen, f"uvicorn never ran: {calls}\n{out.stdout}\n{out.stderr}"
    return seen[-1]


def test_start_demo_exports_two_turn_workers_by_default(tmp_path):
    out, calls = _run_demo(tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert _uvicorn_turn_workers(out, calls) == "2"


def test_start_demo_turn_workers_override_wins(tmp_path):
    out, calls = _run_demo(tmp_path, FALKORCHAT_STOREFRONT_TURN_WORKERS="3")
    assert out.returncode == 0, out.stdout + out.stderr
    assert _uvicorn_turn_workers(out, calls) == "3"
