"""The holder as a real process on a real Unix socket (plan §5.2 FR-8, FR-10)."""

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest
from holder_harness import (
    CALLER_SECRET,
    LAUNCHER,
    OTHER_SECRET,
    UID,
    Env,
    HolderProc,
    StandIn,
    body_for,
    make_cert,
    raw_connect,
    tree_search,
    wait_for,
)

from llmgateway import framing
from llmgateway.failures import classify_exception, classify_response
from llmgateway.types import FailureClass

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cert(tmp_path_factory):
    return make_cert(tmp_path_factory.mktemp("tls"))


@pytest.fixture
def env(tmp_path, cert):
    standin = StandIn(*cert)
    e = Env(tmp_path / "root", standin, cert[0])
    e.add_caller("service", OTHER_SECRET)
    yield e
    standin.close()


@pytest.fixture
def proc(env):
    p = HolderProc(env)
    yield p
    p.stop()


def test_both_callers_are_served_over_the_real_socket(env, proc):
    a = proc.exchange()
    b = proc.exchange(caller="service", secret=OTHER_SECRET)
    assert (a.status, b.status) == (200, 200) and a.authenticated and b.authenticated
    assert len(env.standin.seen) == 2
    mode = os.stat(env.socket_path).st_mode & 0o777
    assert mode == 0o666  # the MAC authenticates, not the mode


def test_refusals_cross_the_socket_as_authenticated_499_exchanges(env, proc):
    r = proc.exchange(body=body_for(model="claude-3-5-sonnet-latest"))
    assert r.authenticated and (r.status, r.refusal_code) == (499, "model_not_pinned")
    assert classify_response(r.status, r.content_type, r.body) is FailureClass.REJECTED
    r = proc.exchange(secret=OTHER_SECRET)  # wrong credential
    assert r.refusal_code == "bad_credential" and not r.authenticated
    assert env.standin.seen == []


def test_stale_timestamp_over_the_socket(env, proc):
    r = proc.exchange(ts=int(time.time()) - 120)
    assert r.refusal_code == "stale_timestamp" and r.authenticated
    assert proc.exchange().status == 200  # time resynced


def test_replayed_frame_is_refused_over_the_socket(env, proc):
    frame, nonce = framing.build_request(
        CALLER_SECRET,
        caller="broker",
        method="POST",
        endpoint=env.endpoint,
        path="/v1/messages",
        run_id="eval-1",
        body=body_for(),
    )

    def send():
        s = raw_connect(env.socket_path)
        s.sendall(frame)
        payload = framing.read_frame(s, max_bytes=1 << 20, timeout=5)
        s.close()
        return framing.parse_reply(payload, CALLER_SECRET, nonce)

    assert send().status == 200
    assert send().refusal_code == "replayed_nonce"
    assert len(env.standin.seen) == 1


def test_oversized_frame_is_dropped_before_it_is_read(env):
    p = HolderProc(env, extra={"max_frame_bytes": 4096})
    try:
        s = raw_connect(env.socket_path)
        start = time.monotonic()
        s.sendall((10_000_000).to_bytes(4, "big"))
        assert dropped(s)  # closed, no reply
        assert time.monotonic() - start < 2  # refused on the declared size, not after a timeout
        big = framing.build_request(
            CALLER_SECRET,
            caller="broker",
            method="POST",
            endpoint=env.endpoint,
            path="/v1/messages",
            run_id="eval-1",
            body=b"x" * 8000,
        )[0]
        s = raw_connect(env.socket_path)
        s.sendall(big)
        assert dropped(s)
        assert p.exchange().status == 200
    finally:
        p.stop()


def test_slow_pre_auth_frame_is_cut_off_and_does_not_block_others(env):
    p = HolderProc(env, extra={"read_timeout": 0.5})
    try:
        s = raw_connect(env.socket_path)
        s.sendall(b"\x00\x00")  # half a length prefix, then silence
        start = time.monotonic()
        assert dropped(s)
        assert time.monotonic() - start < 3
        drip = raw_connect(env.socket_path)
        drip.sendall((100).to_bytes(4, "big"))
        stop = threading.Event()

        def trickle():
            while not stop.is_set():
                try:
                    drip.sendall(b"x")
                except OSError:
                    return
                time.sleep(0.1)

        threading.Thread(target=trickle, daemon=True).start()
        assert p.exchange().status == 200  # a stalled peer does not hold the accept loop
        time.sleep(1.0)  # past the 0.5 s read deadline
        stop.set()
        assert dropped(drip)
    finally:
        p.stop()


def dropped(sock):
    """The peer closed (or reset) without sending any reply."""
    try:
        return sock.recv(10) == b""
    except ConnectionResetError:
        return True


def _readable(sock):
    import select

    return bool(select.select([sock], [], [], 0.1)[0])


def test_per_connection_limit_drops_the_excess(env):
    p = HolderProc(env, extra={"max_connections": 2, "read_timeout": 3.0})
    try:
        idle = [raw_connect(env.socket_path) for _ in range(2)]
        time.sleep(0.2)
        extra = raw_connect(env.socket_path)
        start = time.monotonic()
        assert dropped(extra)  # refused at once, not after the 3 s frame deadline
        assert time.monotonic() - start < 1.5
        [c.close() for c in idle]
        assert wait_for(lambda: _ok(p), 3)
    finally:
        p.stop()


def _ok(p):
    try:
        return p.exchange().status == 200
    except OSError:
        return False


def test_unauthenticated_flood_is_refused_cheaply_and_service_continues(env, proc):
    garbage = os.urandom(300)
    for i in range(150):
        s = raw_connect(env.socket_path)
        if i % 3 == 0:
            s.sendall(len(garbage).to_bytes(4, "big") + garbage)
        elif i % 3 == 1:
            s.sendall(
                framing.build_request(
                    OTHER_SECRET,
                    caller="broker",
                    method="POST",
                    endpoint=env.endpoint,
                    path="/v1/messages",
                    run_id="eval-1",
                    body=body_for(),
                )[0]
            )
        s.close()
    assert proc.exchange().status == 200
    assert len(env.standin.seen) == 1  # none of the flood reached the provider


def test_stopped_holder_is_connection_class_only(env):
    with pytest.raises(framing.HolderUnreachable) as err:
        framing.exchange(
            env.socket_path,
            UID,
            CALLER_SECRET,
            caller="broker",
            endpoint=env.endpoint,
            path="/v1/messages",
            body=body_for(),
        )
    assert classify_exception(err.value) is FailureClass.CONNECTION


def test_a_fake_holder_in_an_agent_writable_directory_is_refused_by_callers(env, tmp_path):
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    fake_dir.chmod(0o777)
    path = str(fake_dir / "holder.sock")
    with pytest.raises(framing.HolderUnreachable):  # writable by others: anyone could swap it
        framing.exchange(
            path, UID, CALLER_SECRET, caller="broker", endpoint="e", path="/p", body=b"{}"
        )
    fake_dir.chmod(0o711)
    with pytest.raises(framing.HolderUnreachable):  # owned by the wrong account
        framing.exchange(
            path, UID + 1, CALLER_SECRET, caller="broker", endpoint="e", path="/p", body=b"{}"
        )


def test_a_fake_holder_that_does_not_know_the_secret_cannot_forge_a_reply(tmp_path):
    d = tmp_path / "fake2"
    d.mkdir()
    d.chmod(0o711)
    path = str(d / "holder.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)

    def serve():
        c, _ = srv.accept()
        payload = framing.read_frame(c, max_bytes=1 << 20, timeout=5)
        hdr, _body = framing.parse_request(payload)
        c.sendall(
            framing.build_response(
                OTHER_SECRET,
                request_nonce=hdr.nonce,
                close_reason="completed",
                status=200,
                content_type="application/json",
                body=b'{"fake":1}',
            )
        )
        c.close()

    threading.Thread(target=serve, daemon=True).start()
    with pytest.raises(framing.ReplyRejected):
        framing.exchange(
            path, UID, CALLER_SECRET, caller="broker", endpoint="e", path="/p", body=b"{}"
        )
    srv.close()


# --- FR-8 (1)-(4): the key never appears ---


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": str(cwd)},
    ).stdout


def test_key_is_nowhere_after_hosted_calls_and_failures(env, tmp_path):
    marker = env.marker.encode()
    control = subprocess.Popen(
        ["sleep", "30"], env={"PATH": os.environ["PATH"], "PLANTED": env.marker}
    )
    scratch = tmp_path / "scratch-repo"
    scratch.mkdir()
    _git(scratch, "init", "-q")
    (scratch / "a.txt").write_text("tracked\n")
    _git(scratch, "add", "a.txt")
    _git(scratch, "commit", "-qm", "init")
    p = HolderProc(env)
    try:
        assert p.exchange().status == 200
        env.standin.responder = lambda s: (
            401,
            {"Content-Type": "application/json"},
            json.dumps(
                {
                    "type": "error",
                    "error": {"message": "invalid x-api-key " + s.headers["x-api-key"]},
                }
            ).encode(),
        )
        failing = p.exchange(run_id="eval-2")
        assert failing.status == 401 and marker not in failing.body
        os.rename(env.key_path, str(env.key_path) + ".gone")  # a missing key fails closed
        assert p.exchange(run_id="eval-3").refusal_code == "key_missing"
        os.rename(str(env.key_path) + ".gone", env.key_path)
        pid = p.proc.pid
        proc_files = {
            f: pathlib.Path(f"/proc/{pid}/{f}").read_bytes() for f in ("environ", "cmdline")
        }
        for name, data in proc_files.items():
            assert marker not in data, name
        assert (
            marker in pathlib.Path(f"/proc/{control.pid}/environ").read_bytes()
        )  # positive control
        log_p = _git(scratch, "log", "-p", "--all")
        assert env.marker not in log_p
    finally:
        out, err = p.stop()
        control.kill()
    assert env.marker not in out + err
    hits = tree_search([env.root], marker)
    assert hits == [
        str(env.key_path)
    ]  # the key file only; not state, ledger, calls.log, WAL, socket dir
    assert tree_search([scratch], marker) == []
    (scratch / "leak.txt").write_text(env.marker)
    assert tree_search([scratch], marker) == [str(scratch / "leak.txt")]  # positive control
    assert wait_for(lambda: True)


def test_rotation_and_restart_keep_authentication(env):
    p = HolderProc(env)
    try:
        assert p.exchange().status == 200
        env.key_path.write_text("sk-ant-FAKE-rotated-9999\n")  # key rotation: no restart, no edit
        assert p.exchange(run_id="eval-r").status == 200
        assert env.standin.seen[-1].headers["x-api-key"] == "sk-ant-FAKE-rotated-9999"
        new = bytes(range(9, 41))
        env.add_caller("broker", new)  # credential rotation: old one is bad_credential at once
        assert p.exchange().refusal_code == "bad_credential"
        assert p.exchange(secret=new).status == 200
    finally:
        p.stop()
    p2 = HolderProc(env)  # restart
    try:
        assert p2.exchange(secret=new).status == 200
    finally:
        p2.stop()


def test_ledger_survives_a_restart_and_the_callers_records_are_irrelevant(env):
    env.write_config(run_bounds=[{"match": "eval-*", "max_tokens": 2000}])
    p = HolderProc(env)
    body = body_for(1000)
    try:
        assert p.exchange(body=body).status == 200  # settles 26
        r = p.exchange(body=body_for(1900))
        assert r.refusal_code == "run_bound_reached"
    finally:
        p.stop()
    caller_db = pathlib.Path(env.root) / "records.sqlite3"
    caller_db.write_text("the caller deletes its own rows; the holder never reads them")
    caller_db.unlink()
    p = HolderProc(env)
    try:
        assert p.exchange(body=body_for(1900)).refusal_code == "run_bound_reached"
        assert p.exchange(body=body).status == 200
    finally:
        p.stop()


def test_a_second_holder_on_the_same_state_refuses_to_start(env, proc):
    spec = {
        "config": str(env.config_path),
        "uid": UID,
        "anchor": str(env.root),
        "state_root": env.state_root,
        "socket": str(env.root / "run" / "second.sock"),
    }
    out = subprocess.run(
        [sys.executable, str(LAUNCHER), json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert (
        out.returncode == 2 and out.stdout.startswith("REFUSED") and "held by another" in out.stdout
    )


def test_holder_start_refusals_over_a_real_process(env):
    def launch():
        spec = {
            "config": str(env.config_path),
            "uid": UID,
            "anchor": str(env.root),
            "state_root": env.state_root,
            "socket": env.socket_path,
        }
        return subprocess.run(
            [sys.executable, str(LAUNCHER), json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=20,
            input="",
        )

    env.write_config(holder={"unmatched": "refuse"})
    assert "window" in launch().stdout
    env.write_config(holder=env.holder_table(window_seconds=0))
    assert launch().returncode == 2
    env.write_config()
    env.config_path.chmod(0o666)
    assert "owner-protected" in launch().stdout


# --- the entry shim on an installed layout, exactly as the unit runs it ---


@pytest.fixture
def installed(tmp_path):
    root = tmp_path / "opt" / "llmgateway"
    for sub in ("src", "bin", "deploy"):
        shutil.copytree(
            REPO / sub, root / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for dirpath, dirs, files in os.walk(root):
        os.chmod(dirpath, 0o755)
        for f in files:
            os.chmod(os.path.join(dirpath, f), 0o644)
    (tmp_path / "opt").chmod(0o755)
    tmp_path.chmod(0o755)
    return root


def run_shim(root, *args, cwd="/", env=None):
    return subprocess.run(
        ["/usr/bin/python3", "-I", str(root / "bin" / "keyholder.py"), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", **(env or {})},
    )


def test_unit_command_line_runs_on_the_installed_layout(installed, tmp_path):
    res = run_shim(installed)  # ExecStart: no arguments
    assert "ModuleNotFoundError" not in res.stderr and "Traceback" not in res.stderr
    assert (
        res.returncode == 2 and "refusing to start" in res.stderr
    )  # reached main; owner is not root
    # PYTHONPATH and cwd cannot redirect the import (they are ignored under -I)
    evil = tmp_path / "evil" / "llmgateway"
    evil.mkdir(parents=True)
    (evil / "__init__.py").write_text("raise SystemExit('PLANTED CODE RAN')")
    (evil / "keyholder.py").write_text("raise SystemExit('PLANTED CODE RAN')")
    res = run_shim(
        installed, cwd=str(tmp_path / "evil"), env={"PYTHONPATH": str(tmp_path / "evil")}
    )
    assert "PLANTED" not in res.stderr + res.stdout and res.returncode == 2


def test_check_deployment_code_only_on_the_installed_layout(installed, tmp_path):
    base = [
        "check-deployment",
        "--code-only",
        "--root",
        str(installed),
        "--anchor",
        str(tmp_path),
        "--trusted-uid",
        str(UID),
    ]
    ok = run_shim(installed, *base)
    assert ok.returncode == 0 and ok.stdout == ""
    (installed / "src" / "llmgateway" / "keyholder.py").chmod(0o666)
    bad = run_shim(installed, *base)
    assert bad.returncode == 1 and "group/other-writable" in bad.stdout


def test_log_without_state_fails_cleanly(installed):
    res = run_shim(installed, "log")
    assert res.returncode == 1 and "Traceback" not in res.stderr
