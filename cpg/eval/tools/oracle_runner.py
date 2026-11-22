#!/usr/bin/env python3
"""oracle_runner.py - the P1-47 capture runner (record ``ccg-oracle-v1``; ``-p1-ml.md`` §6.5).

    FALKORDB_HOST=127.0.0.1 FALKORDB_PORT=<throwaway port> \\
      oracle_runner.py --root ~/.local/share/ccg-eval/<EVAL12> --phases prep,session,entry,suites,post [--run-id ID]

Every phase first passes ``oracle_capture.check_fence`` exactly as ``oracle_capture.py run`` does (explicit
FALKORDB_HOST/PORT, loopback, not 6379/6390/16390, CCG_ORACLE_RUN set, exactly one throwaway container labelled ccg-falkordb=1,
ccg-falkordb.throwaway=1 and ccg-oracle.run=$CCG_ORACLE_RUN publishing that port on 127.0.0.1 only); a refusal exits 3 before anything starts. By default the runner starts
its OWN unauthenticated throwaway (falkor-chat cannot authenticate; the ``ccg-falkordb.sh`` throwaways are
password-protected): the locally present pinned image ``falkordb/falkordb:v4.18.11`` (``--pull never``), bound to
127.0.0.1 on a free port in 16400-16499, named ``ccg-falkordb-tw-<port>``, labelled ccg-falkordb=1 and
ccg-falkordb.throwaway=1 and a per-run ccg-oracle.run=<random token> (exported as CCG_ORACLE_RUN; ``--external``
needs the caller to set it and label their instance); the labels and the loopback bind are verified with ``docker inspect`` before anything
uses it, it is exported as FALKORDB_HOST/FALKORDB_PORT, and it is removed in a ``finally``. ``--external`` instead
uses the instance named by the caller's environment (old behaviour). Only a container this run started is removed.

Phases (each re-runnable; logs append under ``<root>/oracle/runs/<run-id>/``):
  prep     create_workspace / seed_demo / seed_catalog / verify_catalog / seed_salesperson / verify_salesperson
           against the throwaway, run from the exported tree (the entry points that need only the throwaway)
  session  start uvicorn from the EXPORT on a free port (16500+) with the built SPA (``<root>/oracle/spa-dist``) and a
           random presenter key (kept in memory and the child's environment only), run ``oracle_session.mjs`` (S01-S14)
           under Windows node.exe, stop the server. A turn timeout in an LLM-backed step stops the run with
           exit 5 and ``blocked: model not loaded`` in ``session-blocked.txt``; LM Studio state is never changed.
  entry    ``load_test.sh`` (it starts its own server on a free port and runs ``load_append.py``)
  suites   falkor-chat/server pytest and ``scripts/test_queries.sh``, with the shims (TestClient edges = a class outside
           the denominator)
  post     ``attribute-spa`` + ``normalize`` + ``summary`` (counts only on stdout)
Exit: 0 ok, 2 usage, 3 fence refused, 4 bad state, 5 blocked.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_capture as OC  # noqa: E402

REPO = OC.REPO
WS = "demo"
WIN_OUT = r"C:\Windows\Temp\ccg-oracle-out"
WIN_OUT_WSL = "/mnt/c/Windows/Temp/ccg-oracle-out"


IMAGE = "falkordb/falkordb:v4.18.11"
TW_PREFIX = "ccg-falkordb-tw-"


def docker_run_cmd(name: str, port: int, run_id: str, image: str = IMAGE) -> list[str]:
    return ["docker", "run", "-d", "--pull", "never", "--name", name, "--label", "ccg-falkordb=1",
            "--label", "ccg-falkordb.throwaway=1", "--label", f"{OC.RUN_LABEL}={run_id}", "-p", f"127.0.0.1:{port}:6379", "--memory", "2g", "--memory-swap", "2g", image]


def verify_started(name: str, port: int, run_id: str, run=subprocess.run) -> None:
    """Raise FenceRefused unless the container carries both labels and publishes 6379/tcp ONLY on 127.0.0.1:<port>."""
    r = run(["docker", "inspect", "-f", "{{json .Config.Labels}}|{{json .NetworkSettings.Ports}}", name],
            capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise OC.FenceRefused(f"docker inspect {name} failed")
    labels_s, ports_s = r.stdout.strip().split("|", 1)
    labels, ports = json.loads(labels_s) or {}, json.loads(ports_s) or {}
    if labels.get("ccg-falkordb") != "1" or labels.get("ccg-falkordb.throwaway") != "1":
        raise OC.FenceRefused(f"{name} lacks the ccg-falkordb labels")
    if labels.get(OC.RUN_LABEL) != run_id:
        raise OC.FenceRefused(f"{name} lacks this run's {OC.RUN_LABEL} label")
    binds = ports.get("6379/tcp") or []
    if len(binds) != 1 or binds[0].get("HostIp") != "127.0.0.1" or binds[0].get("HostPort") != str(port):
        raise OC.FenceRefused(f"{name} is not published only on 127.0.0.1:{port}: {binds}")


def start_throwaway(run_id: str, run=subprocess.run) -> tuple[str, int]:
    """Start and verify our own throwaway; on any failure remove what was started and raise."""
    if run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        raise SystemExit(f"image {IMAGE} not present locally (no pull is allowed)")
    taken = set(run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split())
    port = next((q for q in range(16400, 16500) if f"{TW_PREFIX}{q}" not in taken and
                 socket.socket().connect_ex(("127.0.0.1", q)) != 0), None)
    if port is None:
        raise SystemExit("no free port in 16400-16499")
    name = f"{TW_PREFIX}{port}"
    if run(docker_run_cmd(name, port, run_id), capture_output=True, text=True).returncode != 0:
        run(["docker", "rm", "-f", name], capture_output=True)
        raise SystemExit(f"docker run failed for {name}")
    try:
        verify_started(name, port, run_id, run)
        for _ in range(60):
            if "PONG" in run(["redis-cli", "-h", "127.0.0.1", "-p", str(port), "ping"], capture_output=True, text=True).stdout:
                return name, port
            time.sleep(0.5)
        raise OC.FenceRefused(f"{name} did not answer PONG")
    except BaseException:
        remove_throwaway(name, run)
        raise


def remove_throwaway(name: str, run=subprocess.run) -> None:
    if name.startswith(TW_PREFIX) and name != "ccg-falkordb-tw-16390":
        run(["docker", "rm", "-f", name], capture_output=True)


def free_port(start=16500, end=16599) -> int:
    for p in range(start, end):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port in 16500-16599")


class Ctx:
    def __init__(self, root: Path, run_id: str):
        self.root = root
        self.src = root / "oracle" / "src"
        self.fc = self.src / "falkor-chat"
        self.runs = root / "oracle" / "runs" / run_id
        self.runs.mkdir(parents=True, exist_ok=True)
        self.log = self.runs / "py-sh.jsonl"      # shim rows (python + curl)
        self.log.touch()

    def env(self, extra=None) -> dict:
        e = dict(os.environ)
        e.update({"ORACLE_LOG": str(self.log), "ORACLE_ROOT": str(self.src),
                  "PATH": f"{OC.SHIMS}:{e.get('PATH', '')}",
                  "PYTHONPATH": ":".join(x for x in (str(OC.SHIMS), str(self.fc / "server"), e.get("PYTHONPATH", "")) if x)})
        e.update(extra or {})
        return e

    def sh(self, name, cmd, cwd, extra=None, timeout=1800) -> int:
        t0 = time.time()
        with open(self.runs / f"{name}.out", "wb") as fh:
            try:
                rc = subprocess.call(cmd, cwd=cwd, env=self.env(extra), stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                rc = 124
        print(json.dumps({"cmd": name, "rc": rc, "seconds": round(time.time() - t0)}))
        return rc


def prepare_export(c: Ctx) -> None:
    if not c.fc.is_dir():
        raise SystemExit("export missing: run oracle_capture.py export first")
    venv = c.fc / "server" / ".venv"
    if not venv.exists():
        venv.symlink_to(REPO / "falkor-chat" / "server" / ".venv")
    out = subprocess.run([str(venv / "bin" / "python"), "-c", "import falkorchat;print(falkorchat.__file__)"],
                         env=c.env(), capture_output=True, text=True)
    if not out.stdout.strip().startswith(str(c.fc / "server")):
        raise SystemExit(f"server code does not come from the export: {out.stdout.strip()!r} {out.stderr[-200:]}")


def opencode_config() -> str:
    r = subprocess.run(["bash", "-c", f". {REPO}/falkor-chat/scripts/shlib/opencode_config.sh; resolve_opencode_config 2>/dev/null; echo $FALKORCHAT_OPENCODE_CONFIG"],
                       capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1]


def phase_prep(c: Ctx) -> int:
    base = {"FALKORCHAT_WS_ID": WS, "FALKORCHAT_USER_ID": "u1", "EMBEDDING_DIM": "1024"}
    for name, cmd in (("create_workspace", ["scripts/create_workspace.sh", WS]), ("seed_demo", ["scripts/seed_demo.sh", WS]),
                      ("seed_catalog", ["scripts/seed_catalog.sh", WS]), ("verify_catalog", ["scripts/verify_catalog.sh"]),
                      ("seed_salesperson", ["scripts/seed_salesperson.sh", WS]), ("verify_salesperson", ["scripts/verify_salesperson.sh", WS])):
        if c.sh(name, cmd, c.fc, base, 600) != 0:
            print(f"prep step {name} failed; see {c.runs / (name + '.out')}", file=sys.stderr)
            return 4
    return 0


def phase_session(c: Ctx, chrome_port: int, turn_s: int) -> int:
    key = secrets.token_urlsafe(24)            # throwaway-only; never written anywhere
    port = free_port()
    spa = c.root / "oracle" / "spa-dist"
    env = c.env({"FALKORCHAT_WS_ID": WS, "FALKORCHAT_USER_ID": "u1", "FALKORCHAT_EMBEDDING_DIM": "1024",
                 "FALKORCHAT_ENABLE_AGENT": "1", "FALKORCHAT_AGENT_ID": "assistant", "FALKORCHAT_WORKFLOW_ENABLED": "1",
                 "FALKORCHAT_TRIGGER_DEF_KEY": "salesperson", "FALKORCHAT_TRIGGER_DEF_VERSION": "v9",
                 "FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH": "0", "FALKORCHAT_STOREFRONT_ENABLED": "1",
                 "FALKORCHAT_STOREFRONT_DIR": str(spa), "FALKORCHAT_STOREFRONT_TURN_WORKERS": "2",
                 "FALKORCHAT_STOREFRONT_PRESENTER_KEY": key, "FALKORCHAT_OPENCODE_CONFIG": opencode_config()})
    srv_log = open(c.runs / "server.out", "wb")
    srv = subprocess.Popen([str(c.fc / "server/.venv/bin/uvicorn"), "falkorchat.app:app", "--host", "0.0.0.0", "--port", str(port),
                            "--log-config", str(c.fc / "config/logging.json")], cwd=c.fc / "server", env=env,
                           stdout=srv_log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/shop/api/health", timeout=2).read()
                break
            except Exception:  # noqa: BLE001
                if srv.poll() is not None:
                    print("server exited early; see server.out", file=sys.stderr)
                    return 4
                time.sleep(1)
        else:
            print("server not healthy in 120 s", file=sys.stderr)
            return 4
        raw_win = rf"{WIN_OUT}\spa-raw-{port}.jsonl"
        nodeenv = dict(os.environ, ORACLE_PRESENTER_KEY=key, WSLENV="ORACLE_PRESENTER_KEY")
        with open(c.runs / "session.out", "wb") as fh:
            rc = subprocess.call(["node.exe", subprocess.check_output(["wslpath", "-w", str(HERE / "oracle_session.mjs")], text=True).strip(),
                                  "--base", f"http://127.0.0.1:{port}", "--out", raw_win, "--chrome-port", str(chrome_port),
                                  "--turn-timeout-s", str(turn_s)], env=nodeenv, stdout=fh, stderr=subprocess.STDOUT, timeout=3000)
        shutil.copy(f"{WIN_OUT_WSL}/spa-raw-{port}.jsonl", c.runs / "spa-raw.jsonl")
        steps = [json.loads(l) for l in (c.runs / "session.out").read_text().splitlines() if l.startswith("{") and '"step"' in l]
        print(json.dumps({"session_rc": rc, "port": port, "steps": [(s["step"][:3], s["outcome"], s["detail"][:40]) for s in steps]}))
        if any(s["step"][:3] in ("S03", "S04", "S05") and "timeout" in s["detail"] for s in steps):
            (c.runs / "session-blocked.txt").write_text("blocked: model not loaded (LLM-backed turn timed out); LM Studio state not changed\n")
            print("blocked: model not loaded", file=sys.stderr)
            return 5
        return 0
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=20)
        except Exception:  # noqa: BLE001
            os.killpg(srv.pid, signal.SIGKILL)


def phase_entry(c: Ctx) -> int:
    return 0 if c.sh("load_test", ["scripts/load_test.sh"], c.fc,
                     {"SERVER_PORT": str(free_port()), "LOAD_MESSAGES": "200", "LOAD_WORKERS": "4"}, 900) == 0 else 4


def phase_suites(c: Ctx) -> int:
    a = c.sh("pytest_falkor_chat_server", [str(c.fc / "server/.venv/bin/python"), "-m", "pytest", "-q"], c.fc / "server", None, 3000)
    b = c.sh("test_queries", ["scripts/test_queries.sh"], c.fc, None, 1800)
    return 0 if (a, b) == (0, 0) else 4


def phase_post(c: Ctx, dist: Path) -> int:
    spa_log = c.runs / "spa.jsonl"
    if (c.runs / "spa-raw.jsonl").exists():
        OC.main(["attribute-spa", "--raw", str(c.runs / "spa-raw.jsonl"), "--dist", str(dist), "--out", str(spa_log)])
    allp = c.runs / "all.jsonl"
    allp.write_text("".join((p.read_text() if p.exists() else "") for p in (c.log, spa_log)))
    OC.main(["normalize", "--log", str(allp), "--out", str(c.runs / "edges.jsonl")])
    return OC.main(["summary", "--edges", str(c.runs / "edges.jsonl")])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--phases", default="prep,session,entry,suites,post")
    p.add_argument("--run-id", default="run1")
    p.add_argument("--chrome-port", type=int, default=9333)
    p.add_argument("--turn-timeout-s", type=int, default=150)
    p.add_argument("--external", action="store_true", help="use the instance named by FALKORDB_HOST/FALKORDB_PORT")
    a = p.parse_args(argv)
    owned = None
    try:
        if not a.external:
            os.environ[OC.RUN_ENV] = secrets.token_hex(8)     # per-run owner label value (fence admits only this)
            try:
                owned, port = start_throwaway(os.environ[OC.RUN_ENV])
            except OC.FenceRefused as e:
                print(f"FENCE REFUSED: {e}", file=sys.stderr)
                return 3
            os.environ["FALKORDB_HOST"], os.environ["FALKORDB_PORT"] = "127.0.0.1", str(port)
            print(json.dumps({"throwaway": owned, "port": port}))
        return _run(a)
    finally:
        if owned:
            remove_throwaway(owned)
            print(json.dumps({"removed": owned}))


def _run(a) -> int:
    try:
        OC.check_fence()
    except OC.FenceRefused as e:
        print(f"FENCE REFUSED: {e}", file=sys.stderr)
        return 3
    root = Path(a.root).expanduser()
    c = Ctx(root, a.run_id)
    prepare_export(c)
    for ph in a.phases.split(","):
        try:
            OC.check_fence()                     # re-asserted before every phase
        except OC.FenceRefused as e:
            print(f"FENCE REFUSED: {e}", file=sys.stderr)
            return 3
        rc = {"prep": lambda: phase_prep(c), "session": lambda: phase_session(c, a.chrome_port, a.turn_timeout_s),
              "entry": lambda: phase_entry(c), "suites": lambda: phase_suites(c),
              "post": lambda: phase_post(c, root / "oracle" / "spa-dist")}[ph]()
        print(json.dumps({"phase": ph, "rc": rc}))
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
