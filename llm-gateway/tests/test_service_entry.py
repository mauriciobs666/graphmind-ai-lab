"""The service's start-up and its edges: the entry shim, the credential reader, the socket
directory, the peer check (plan §3.4; security Pass 6 on credential intake)."""

import json
import os
import pathlib
import socket
import subprocess
import time

import pytest
from gwstandin import write_config
from holder_harness import CALLER_SECRET, wait_for

from llmgateway.client import CallFailed, Client
from llmgateway.service import (
    Service,
    ServiceSettings,
    ServiceStartError,
    read_credential,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIM = ROOT / "bin" / "service.py"
UID = os.getuid()


# --- bin/service.py ---


def test_the_shim_refuses_without_isolated_mode(tmp_path):
    res = subprocess.run(
        ["/usr/bin/python3", str(SHIM), "--help"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path)},
        cwd="/",
    )
    assert res.returncode != 0 and "-I" in res.stderr and "Traceback" not in res.stderr


def test_the_shim_runs_isolated_from_its_own_src_and_ignores_the_environment(tmp_path):
    """Started as the unit starts it (`python3 -I`), against a code root it finds beside itself;
    `LLMGATEWAY_CONFIG` and PYTHONPATH change nothing."""
    code = tmp_path / "opt"
    shutil_copy(ROOT / "bin", code / "bin")
    shutil_copy(ROOT / "src", code / "src")
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "llmgateway.py").write_text("raise SystemExit('evil package imported')\n")
    cfg = tmp_path / "evil.toml"
    cfg.write_text('[providers.x]\nkind = "lmstudio"\nbase_url = "http://127.0.0.1:1"\n')
    run = tmp_path / "run"
    proc = subprocess.Popen(
        [
            "/usr/bin/python3",
            "-I",
            str(code / "bin" / "service.py"),
            "--socket",
            str(run / "s.sock"),
            "--state-dir",
            str(tmp_path / "st"),
            "--credential",
            str(tmp_path / "absent.cred"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(evil), "LLMGATEWAY_CONFIG": str(cfg)},
        cwd=str(evil),
    )
    try:
        assert wait_for(lambda: (run / "s.sock").exists(), 10), proc.poll()
        with pytest.raises(CallFailed) as info:
            Client(str(run / "s.sock")).call("x", [{"role": "user", "content": "q"}])
        # /etc/llmgateway/gateway.toml is absent or unprotected here: no call site exists, and
        # the environment's config was not read
        assert info.value.failure_class == "config"
    finally:
        proc.terminate()
        _, err = proc.communicate(timeout=10)
    assert "evil package" not in err and "Traceback" not in err
    assert "ready" in err


def shutil_copy(src, dst):
    import shutil

    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))


# --- the credential ---


def cred(tmp_path, *, text=None, mode=0o600, name="c.cred"):
    p = tmp_path / name
    p.write_text((CALLER_SECRET.hex() if text is None else text) + "\n")
    p.chmod(mode)
    return str(p)


def test_a_private_regular_credential_is_read(tmp_path):
    assert read_credential(cred(tmp_path)) == CALLER_SECRET


@pytest.mark.parametrize(
    "kw",
    [{"mode": 0o644}, {"mode": 0o640}, {"mode": 0o604}, {"text": "ab" * 31}, {"text": "zz" * 32}],
)
def test_a_credential_that_is_too_open_or_malformed_is_refused(tmp_path, kw):
    with pytest.raises(ValueError):
        read_credential(cred(tmp_path, **kw))


def test_a_symlink_a_hard_link_and_another_owner_are_refused(tmp_path):
    real = cred(tmp_path)
    link = tmp_path / "link.cred"
    link.symlink_to(real)
    with pytest.raises(OSError):
        read_credential(str(link))
    os.link(real, tmp_path / "hard.cred")
    with pytest.raises(ValueError):
        read_credential(real)
    with pytest.raises(ValueError):
        read_credential(cred(tmp_path, name="other.cred"), UID + 1)


# --- the socket ---


def make_service(tmp_path, **kw):
    src = write_config(tmp_path / "root", "[scopes.plain]\n")
    return Service(
        ServiceSettings(
            config_source=src,
            socket_path=kw.pop("socket_path", str(tmp_path / "run" / "s.sock")),
            state_dir=str(tmp_path / "state"),
            credential_path=str(tmp_path / "absent.cred"),
            **kw,
        )
    )


def test_a_socket_directory_that_is_not_private_is_refused(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(ServiceStartError, match="private"):
        make_service(tmp_path, socket_path=str(shared / "s.sock")).bind()


def test_the_socket_path_must_not_be_a_regular_file_and_a_stale_socket_is_replaced(tmp_path):
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    (run / "s.sock").write_text("not a socket")
    with pytest.raises(ServiceStartError, match="not ours"):
        make_service(tmp_path).bind()
    (run / "s.sock").unlink()
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(run / "s.sock"))
    dead.close()  # a socket file nobody listens on
    svc = make_service(tmp_path)
    svc.bind()
    assert (run / "s.sock").stat().st_mode & 0o777 == 0o600
    svc.close()
    assert not (run / "s.sock").exists()


def test_a_peer_of_another_uid_is_closed_without_an_answer(tmp_path):
    svc = make_service(tmp_path, own_uid=UID + 1)  # every real peer now looks foreign
    # the directory check wants the service's own uid: bind through a private dir of the test uid
    svc._uid = UID
    svc.bind()
    svc._uid = UID + 1
    import threading

    threading.Thread(target=svc.serve_forever, daemon=True).start()
    try:
        with pytest.raises(CallFailed) as info:
            Client(str(tmp_path / "run" / "s.sock"), timeout=3).call("x", [])
        assert info.value.failure_class == "connection"
    finally:
        svc.close()


def test_a_slow_or_oversized_frame_is_dropped(tmp_path):
    import threading

    svc = make_service(tmp_path, read_timeout=0.3, max_frame_bytes=64)
    svc.bind()
    threading.Thread(target=svc.serve_forever, daemon=True).start()
    path = str(tmp_path / "run" / "s.sock")
    try:
        big = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        big.connect(path)
        big.settimeout(2)
        big.sendall((1000).to_bytes(4, "big"))
        assert big.recv(10) == b""  # closed, nothing answered
        big.close()
        slow = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        slow.connect(path)
        slow.settimeout(3)
        slow.sendall((50).to_bytes(4, "big") + b"{")
        t0 = time.monotonic()
        assert slow.recv(10) == b""
        assert time.monotonic() - t0 < 2.5
        slow.close()
        # bad JSON / a non-object / a duplicate key get a refusal or a drop, never a traceback
        for payload in (b'{"op":"call","op":"load"}', b"[1]"):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(path)
            s.settimeout(2)
            s.sendall(len(payload).to_bytes(4, "big") + payload)
            raw = s.recv(4096)
            if raw:
                assert json.loads(raw[4:])["ok"] is False
            s.close()
    finally:
        svc.close()
