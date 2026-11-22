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


def test_attribute_spa_maps_first_non_wrapper_frame_through_a_source_map(tmp_path):
    # generated line 1 col 0 -> sources[0] (client.ts) line 1; line 2 col 0 -> sources[1] (endpoints.ts) line 5
    # mappings "AAAA;AAIA": seg1 = [0,0,0,0]; seg2 = [0,+0 src... sources idx +1, line +4, col 0]
    sm = {"version": 3, "sources": ["../src/api/client.ts", "../src/api/endpoints.ts"], "mappings": "AAAA;ACIA"}
    dist = tmp_path / "dist" / "assets"
    dist.mkdir(parents=True)
    (dist / "index-X.js.map").write_text(json.dumps(sm))
    raw = tmp_path / "raw.jsonl"
    stack = ("Error\n    at window.fetch (http://h:1/shop/assets/index-X.js:9:9)\n"
             "    at apiFetch (http://h:1/shop/assets/index-X.js:1:1)\n"
             "    at getState (http://h:1/shop/assets/index-X.js:2:1)\n")
    raw.write_text(json.dumps({"method": "GET", "url": "http://h:1/shop/api/state", "stack": stack}) + "\n"
                   + json.dumps({"method": "GET", "url": "http://h:1/shop/api/x", "stack": "Error\n    at z (http://h:1/other.js:1:1)"}) + "\n")
    out = tmp_path / "o.jsonl"
    assert O.main(["attribute-spa", "--raw", str(raw), "--dist", str(tmp_path / "dist"), "--out", str(out)]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows[0]["frame"] == {"file": "salesperson/src/api/endpoints.ts", "func": None} and rows[0]["path"] == "/shop/api/state"
    assert rows[1]["frame"] is None


def test_runner_refuses_through_the_fence_before_creating_anything(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("FALKORDB_PORT", "FALKORDB_HOST")}
    for extra in ({}, {"FALKORDB_HOST": "127.0.0.1", "FALKORDB_PORT": "6390"}, {"FALKORDB_HOST": "127.0.0.1", "FALKORDB_PORT": "6379"}):
        r = subprocess.run([sys.executable, str(TOOLS / "oracle_runner.py"), "--external", "--root", str(tmp_path)], env={**env, **extra},
                           capture_output=True, text=True)
        assert r.returncode == 3 and "FENCE REFUSED" in r.stderr
    assert list(tmp_path.iterdir()) == []


# --- runner-owned throwaway (container-start path) -------------------------------------------------------------
import types  # noqa: E402

import oracle_runner as RN  # noqa: E402


def _inspect_stdout(labels, ports):
    return json.dumps(labels) + "|" + json.dumps(ports)


GOOD_L = {"ccg-falkordb": "1", "ccg-falkordb.throwaway": "1"}
GOOD_P = {"6379/tcp": [{"HostIp": "127.0.0.1", "HostPort": "16400"}]}


def _fake(stdout, rc=0):
    return lambda *a, **k: types.SimpleNamespace(returncode=rc, stdout=stdout, stderr="")


def test_docker_run_cmd_is_loopback_labelled_and_never_pulls():
    c = RN.docker_run_cmd("ccg-falkordb-tw-16400", 16400)
    assert c[c.index("-p") + 1] == "127.0.0.1:16400:6379"
    assert c.count("--label") == 2 and "ccg-falkordb=1" in c and "ccg-falkordb.throwaway=1" in c
    assert c[c.index("--pull") + 1] == "never" and c[-1] == RN.IMAGE == "falkordb/falkordb:v4.18.11"


def test_verify_started_accepts_the_good_shape():
    RN.verify_started("n", 16400, _fake(_inspect_stdout(GOOD_L, GOOD_P)))


@pytest.mark.parametrize("labels,ports", [
    ({"ccg-falkordb": "1"}, GOOD_P),                                              # throwaway label missing
    ({"ccg-falkordb.throwaway": "1"}, GOOD_P),                                    # ccg-falkordb label missing
    ({}, GOOD_P),
    (GOOD_L, {"6379/tcp": [{"HostIp": "0.0.0.0", "HostPort": "16400"}]}),         # non-loopback bind
    (GOOD_L, {"6379/tcp": [{"HostIp": "127.0.0.1", "HostPort": "16401"}]}),       # other port
    (GOOD_L, {"6379/tcp": [{"HostIp": "127.0.0.1", "HostPort": "16400"}, {"HostIp": "::", "HostPort": "16400"}]}),
    (GOOD_L, {}),                                                                 # unpublished
])
def test_verify_started_refuses_bad_shapes(labels, ports):
    with pytest.raises(O.FenceRefused):
        RN.verify_started("n", 16400, _fake(_inspect_stdout(labels, ports)))


def test_start_throwaway_removes_what_it_started_when_verification_fails():
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        out = ""
        if cmd[:3] == ["docker", "inspect", "-f"]:
            out = _inspect_stdout({}, GOOD_P)
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    with pytest.raises(O.FenceRefused):
        RN.start_throwaway(run)
    assert any(c[:3] == ["docker", "rm", "-f"] and c[3].startswith("ccg-falkordb-tw-") for c in calls)


def test_start_throwaway_refuses_a_missing_local_image_without_pulling():
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")
    with pytest.raises(SystemExit):
        RN.start_throwaway(run)
    assert not any(c[:2] == ["docker", "run"] or c[:2] == ["docker", "pull"] for c in calls)


def test_remove_throwaway_never_touches_a_foreign_or_the_16390_container():
    calls = []
    for n in ("ccg-falkordb", "falkordb-dev", "ccg-falkordb-tw-16390"):
        RN.remove_throwaway(n, lambda *a, **k: calls.append(a))
    assert calls == []
