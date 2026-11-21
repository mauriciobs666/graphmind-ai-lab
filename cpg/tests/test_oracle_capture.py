"""P1-47 capture wrapper: fence, shims, export, normalize (offline; the live container check is injected)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "eval" / "tools"
sys.path.insert(0, str(TOOLS))
import oracle_capture as O  # noqa: E402

OK = lambda port: ["ccg-falkordb-tw-%d" % port]  # noqa: E731


def test_fence_accepts_explicit_throwaway():
    assert O.check_fence({"FALKORDB_HOST": "127.0.0.1", "FALKORDB_PORT": "16400"}, OK) == (
        "127.0.0.1", 16400, "ccg-falkordb-tw-16400")


@pytest.mark.parametrize("env", [
    {}, {"FALKORDB_HOST": "127.0.0.1"}, {"FALKORDB_PORT": "16400"},          # unset = the named mutant
    {"FALKORDB_HOST": "127.0.0.1", "FALKORDB_PORT": "6379"},
    {"FALKORDB_HOST": "127.0.0.1", "FALKORDB_PORT": "6390"},
    {"FALKORDB_HOST": "10.0.0.5", "FALKORDB_PORT": "16400"},
    {"FALKORDB_HOST": "127.0.0.1", "FALKORDB_PORT": "x"}])
def test_fence_refuses(env):
    with pytest.raises(O.FenceRefused):
        O.check_fence(env, OK)


def test_fence_refuses_unlabelled_or_ambiguous_container():
    env = {"FALKORDB_HOST": "127.0.0.1", "FALKORDB_PORT": "16400"}
    for names in ([], ["a", "b"]):
        with pytest.raises(O.FenceRefused):
            O.check_fence(env, lambda p, n=names: n)


def test_run_refuses_with_port_unset_and_never_runs_command(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("FALKORDB_PORT", "FALKORDB_HOST")}
    marker = tmp_path / "ran"
    r = subprocess.run([sys.executable, str(TOOLS / "oracle_capture.py"), "run", "--root", str(tmp_path),
                        "--log", str(tmp_path / "l"), "--", "touch", str(marker)], env=env, capture_output=True, text=True)
    assert r.returncode == 3 and "FENCE REFUSED" in r.stderr and not marker.exists()


def _run_shimmed(tmp_path, code, name="app.py"):
    root = tmp_path / "oracle" / "src"
    root.mkdir(parents=True)
    (root / name).write_text(code)
    log = tmp_path / "log.jsonl"
    env = dict(os.environ, ORACLE_LOG=str(log), ORACLE_ROOT=str(root), PYTHONPATH=str(TOOLS / "oracle_shims"),
               PATH=f"{TOOLS / 'oracle_shims'}:{os.environ['PATH']}")
    subprocess.run([sys.executable, str(root / name)], env=env, capture_output=True, timeout=60)
    return root, log, env


def test_python_shim_attributes_function_grain_and_unattributed(tmp_path):
    code = ("import urllib.request\n"
            "def fetch_orders():\n"
            "    try: urllib.request.urlopen('http://127.0.0.1:9/api/orders/7')\n"
            "    except Exception: pass\n"
            "fetch_orders()\n")
    _, log, _ = _run_shimmed(tmp_path, code)
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert rows[0]["method"] == "GET" and rows[0]["path"] == "/api/orders/7"
    assert rows[0]["frame"] == {"file": "app.py", "func": "fetch_orders"}
    # a call whose stack has no frame under ORACLE_ROOT is unattributed (frame null), never guessed
    out = subprocess.run([sys.executable, "-c", "import urllib.request\ntry: urllib.request.urlopen('http://127.0.0.1:9/x')\nexcept Exception: pass"],
                         env=dict(os.environ, ORACLE_LOG=str(log), ORACLE_ROOT=str(tmp_path / "oracle" / "src"),
                                  PYTHONPATH=str(TOOLS / "oracle_shims")), capture_output=True)
    last = json.loads(log.read_text().splitlines()[-1])
    assert last["path"] == "/x" and last["frame"] is None


def test_curl_shim_records_parent_script_at_file_grain(tmp_path):
    root = tmp_path / "oracle" / "src"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "go.sh").write_text("#!/usr/bin/env bash\ncurl -s -X POST http://127.0.0.1:9/api/x/1 >/dev/null 2>&1\n")
    log = tmp_path / "log.jsonl"
    env = dict(os.environ, ORACLE_LOG=str(log), ORACLE_ROOT=str(root), PATH=f"{TOOLS / 'oracle_shims'}:{os.environ['PATH']}")
    subprocess.run(["bash", str(root / "scripts" / "go.sh")], env=env, timeout=60)
    row = json.loads(log.read_text().splitlines()[0])
    assert (row["method"], row["path"], row["frame"]) == ("POST", "/api/x/1", {"file": "scripts/go.sh", "func": None})


def test_normalize_and_summary_split_testclient_and_unattributed(tmp_path, capsys):
    log = tmp_path / "l.jsonl"
    rows = [{"src": "py", "method": "GET", "host": "h", "port": 1, "path": "/api/a?x=1", "frame": {"file": "f.py", "func": "g"}, "cls": "app"},
            {"src": "py", "method": "GET", "host": "testserver", "port": 80, "path": "/api/a", "frame": {"file": "t.py", "func": "t"}, "cls": "testclient"},
            {"src": "sh", "method": "GET", "host": "h", "port": 1, "path": "/api/b", "frame": None, "cls": "app"}]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "e.jsonl"
    assert O.main(["normalize", "--log", str(log), "--out", str(out)]) == 0
    capsys.readouterr()
    assert O.main(["summary", "--edges", str(out)]) == 0
    s = json.loads(capsys.readouterr().out)
    assert (s["rows"], s["testclient"], s["unattributed"], s["attributed"], s["distinctEdges"]) == (3, 1, 1, 1, 1)
