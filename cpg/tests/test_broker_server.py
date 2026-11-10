"""P1-29 — the broker process: frames, the listener, the singleton, the closed environment, the upstream exchange, the
client.

Spec: `docs/plans/connected-code-graph-egress-spec3.md` §2.0 (process), §2.2 (framing), §2.3 (transport and bounds),
§2.5 (a missing or latched broker fails the call), §10.1 tests (14) (frames), (16), (17), (18), (19), (20) (the stderr
part is in `test_broker_audit.py`). Process tests start the real ``python -m ccg.broker.server`` under a temporary
directory, a temporary socket and a temporary audit directory; none touches the real ``~/.local/state`` or ``cpg/.audit``.
No test opens a connection to anything but 127.0.0.1 or a Unix socket in the temporary directory.
"""
from __future__ import annotations

import json
import os
import pwd
import signal
import socket
import ssl
import struct
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from broker_helpers import (Harness, Upstream, egress, http_reply, load_system, model_call, write_manifest)
from ccg.broker import framing, server, structured
from ccg.broker.client import BrokerClient, BrokerError, BrokerUnavailable
from ccg.broker.server import BrokerServer

CPG = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_path():
    """A private directory under the user's cache dir, not under `/tmp`: the broker process checks every ancestor of
    its directory strictly (section 2.0, no sticky exemption), and `/tmp` is a sticky world-writable ancestor. The
    base is under the home, not the checkout, to keep the AF_UNIX path (limit 108 bytes) short."""
    base = Path.home() / ".cache" / "ccg-broker-tests"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=base))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def frame(raw: bytes) -> bytes:
    return struct.pack(">I", len(raw)) + raw


# ------------------------------------------------------------------ framing


@pytest.mark.parametrize("raw", [
    b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}', b'[1e999]', b"", b"{", b'{"a":1} x',
    b"\xff\xfe", b"[" * 65 + b"]" * 65, b'{"a":' * 65 + b"1" + b"}" * 65, b"9223372036854775809", b"-9223372036854775809",
    b"1" * 5000,
])
def test_strict_parse_refuses(raw):
    with pytest.raises(framing.FrameError):
        framing.parse_strict(raw)


@pytest.mark.parametrize("raw,val", [
    (b"[" * 64 + b"]" * 64, None), (b"9223372036854775807", 9223372036854775807), (b"9223372036854775808", 2**63), (b"-9223372036854775808", -2**63),
    (b'{"a":[1,2.5,null,"\\u00e9"]}', {"a": [1, 2.5, None, "é"]}), (b"  true ", True),
])
def test_strict_parse_accepts(raw, val):
    out = framing.parse_strict(raw)
    if val is not None:
        assert out == val


def test_depth_counts_containers_through_objects_and_arrays_alike():
    ok = b'{"a":[' * 32 + b"1" + b"]}" * 32  # 64 containers
    framing.parse_strict(ok)
    with pytest.raises(framing.FrameError):
        framing.parse_strict(b'[{"a":' * 33 + b"1" + b"}]" * 33)  # 66


def test_encode_refuses_nan_and_oversize():
    with pytest.raises(ValueError):
        framing.encode({"a": float("nan")})
    with pytest.raises(framing.FrameError):
        framing.encode({"a": "x" * 100}, max_len=50)
    assert framing.encode({"a": 1}) == struct.pack(">I", 7) + b'{"a":1}'


def test_read_frame_checks_the_length_before_reading_the_payload():
    got = []

    def recv(n):
        got.append(n)
        return struct.pack(">I", 1000)[:n] if len(got) == 1 else b"x" * n
    with pytest.raises(framing.FrameError, match="exceeds"):
        framing.read_frame(recv, 999)
    assert got == [4]  # nothing past the header was requested


# ------------------------------------------------------------------ the listener, in process


@pytest.fixture
def served(tmp_path):
    hh = Harness(tmp_path)
    d = tmp_path / "sock"
    d.mkdir(mode=0o700)
    srv = BrokerServer(hh.broker, d / "b.sock", log=hh.logs.append)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield hh, srv
    srv.shutdown()


def raw(path, payload: bytes, timeout=5.0) -> bytes:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(path))
        s.sendall(payload)
        data = b""
        while True:
            try:
                c = s.recv(65536)
            except ConnectionResetError:
                break
            if not c:
                break
            data += c
        return data
    finally:
        s.close()


def test_the_socket_is_0600_and_the_only_listener(served):
    hh, srv = served
    assert oct(os.stat(srv.sock_path).st_mode & 0o777) == "0o600"
    assert srv.sock.family == socket.AF_UNIX and srv.sock.type == socket.SOCK_STREAM


@pytest.mark.parametrize("payload", [
    frame(b'{"op":"open_run","op":"x"}'),                               # duplicate key
    frame(b'{"op":"open_run","systemId":NaN}'),                         # NaN
    frame(b'{"op":"request","token":"x","request":' + b"[" * 70 + b"]" * 70 + b"}"),  # depth > 64
    frame(b"not json"), frame(b""), struct.pack(">I", 4 * 1024 * 1024 + 64 * 1024 + 1),  # over the cap, header only
    frame(b'{"op":"unknown"}'), frame(b"[1]"),                           # not an operation
])
def test_a_malformed_frame_closes_with_no_reply_and_no_record(served, payload):
    hh, srv = served
    assert raw(srv.sock_path, payload) == b""
    assert hh.records() == []
    assert any("malformed" in m for m in hh.logs)  # counted on the log, never recorded


def test_a_frame_at_depth_64_is_accepted_and_answered(served):
    hh, srv = served
    deep = b'{"op":"request","token":"x","request":' + b'{"a":' * 61 + b"1" + b"}" * 61 + b"}"
    framing.parse_strict(deep)  # exactly 63 containers deep + the wrapper: within the bound
    reply = framing.parse_strict(raw(srv.sock_path, frame(deep))[4:])
    assert reply["decision"]["reason"] == "token-invalid"
    assert len(hh.records("decision")) == 1


def test_a_peer_of_another_uid_is_closed_with_no_reply_no_record(tmp_path):
    hh = Harness(tmp_path)
    d = tmp_path / "s"
    d.mkdir(mode=0o700)
    srv = BrokerServer(hh.broker, d / "b.sock", peer_uid=lambda c: os.getuid() + 1, log=hh.logs.append)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        msg = frame(b'{"op":"open_run","runKind":"answer","systemId":"shop","codebases":"*","override":null}')
        assert raw(srv.sock_path, msg) == b""
        assert hh.records() == [] and len(hh.broker.runs) == 0
        assert any("another uid" in m for m in hh.logs)
    finally:
        srv.shutdown()


def test_the_real_peer_credential_reader_returns_our_uid(served):
    hh, srv = served
    a, b = socket.socketpair()
    try:
        assert server.so_peercred_uid(a) == os.getuid()
    finally:
        a.close(), b.close()


def test_a_live_socket_is_never_unlinked_and_a_stale_one_is_replaced(served, tmp_path):
    hh, srv = served
    ino = os.stat(srv.sock_path).st_ino
    with pytest.raises(server.PreconditionError, match="already listening"):
        BrokerServer(hh.broker, srv.sock_path)
    assert os.stat(srv.sock_path).st_ino == ino
    assert raw(srv.sock_path, frame(b'{"op":"close_run","token":"x"}'))  # still answers
    stale = tmp_path / "stale.sock"
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(stale))
    s.close()  # leaves the file, nobody listens
    assert stale.exists()
    srv2 = BrokerServer(hh.broker, stale)
    srv2.shutdown()
    notsock = tmp_path / "plain"
    notsock.write_text("x")
    with pytest.raises(server.PreconditionError, match="not a socket"):
        BrokerServer(hh.broker, notsock)
    assert notsock.read_text() == "x"


def test_one_request_one_reply_per_connection_and_the_reply_is_a_frame(served):
    hh, srv = served
    out = raw(srv.sock_path, frame(b'{"op":"open_run","runKind":"answer","systemId":"shop","codebases":"*","override":null}'))
    (n,) = struct.unpack(">I", out[:4])
    reply = json.loads(out[4:])
    assert n == len(out) - 4 and reply["ok"] and len(reply["grant"]["token"]) == 64


# ------------------------------------------------------------------ the client (§2.5)


def test_client_round_trip_and_refusal_text(served):
    hh, srv = served
    c = BrokerClient(str(srv.sock_path), timeout=5)
    g = c.open_run("answer", "shop", "*")
    assert g.effective == {"front": "local-only", "docs": "hosted", "lib": "hosted"} and g.system_local_only
    d, resp = c.request(g.token, model_call(), {"messages": []})
    assert d.allowed and d.rule == "R1-LOCAL-ENDPOINT" and resp.status == 200 and resp.body == b'{"ok":1}'
    assert resp.content_type == "application/json"  # the upstream's header, carried through the reply
    d, resp = c.request(g.token, model_call(endpoint="https://api.hosted.example.com:443"), {"m": 1})
    assert not d.allowed and resp is None
    assert d.refusal() == "refused by egress policy (R5-SYSTEM-LOCAL-MODEL: system-local-only-model)"
    c.close_run(g.token)
    d, _ = c.request(g.token, model_call(), {"m": 1})
    assert d.reason == "token-invalid"
    with pytest.raises(BrokerError) as e:
        c.open_run("answer", "shop", ["nosuch"])
    assert e.value.code == "open-run-invalid"


def test_client_with_no_broker_fails_and_never_connects_directly(tmp_path, monkeypatch):
    families = []
    real = socket.socket

    class Spy(real):
        def __init__(self, family=-1, *a, **kw):
            families.append(socket.AddressFamily(family) if family != -1 else None)
            super().__init__(family, *a, **kw)
    monkeypatch.setattr(socket, "socket", Spy)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("a direct connection was attempted"))
    c = BrokerClient(str(tmp_path / "absent.sock"), timeout=2)
    with pytest.raises(BrokerUnavailable):
        c.open_run("answer", "shop", "*")
    with pytest.raises(BrokerUnavailable):
        c.request("0" * 64, model_call(), {"m": 1})
    assert families and set(families) == {socket.AF_UNIX}


def test_client_with_a_latched_broker_fails_with_the_reason(served):
    hh, srv = served
    c = BrokerClient(str(srv.sock_path), timeout=5)
    g = c.open_run("answer", "shop", "*")
    hh.audit.latch()
    with pytest.raises(BrokerError) as e:
        c.request(g.token, model_call(), {"m": 1})
    assert e.value.code == "audit-unwritable"
    with pytest.raises(BrokerError) as e:
        c.open_run("answer", "shop", "*")
    assert e.value.code == "audit-unwritable"


def test_client_treats_a_reply_it_cannot_read_as_unavailable(tmp_path):
    p = tmp_path / "x.sock"
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(p))
    s.listen(1)

    def serve():
        conn, _ = s.accept()
        conn.recv(10)
        conn.sendall(frame(b"not json"))
        conn.close()
    threading.Thread(target=serve, daemon=True).start()
    with pytest.raises(BrokerUnavailable):
        BrokerClient(str(p), timeout=3).open_run("answer", "shop", "*")
    s.close()


def test_client_socket_discovery(monkeypatch):
    from ccg.broker import client
    monkeypatch.setenv("CCG_BROKER_SOCKET", "/x/y.sock")
    assert client.default_socket_path() == "/x/y.sock"
    monkeypatch.delenv("CCG_BROKER_SOCKET")
    home = pwd.getpwuid(os.getuid()).pw_dir
    monkeypatch.setenv("HOME", "/not/the/passwd/home")  # the passwd entry decides, not $HOME
    assert client.default_socket_path() == f"{home}/.local/state/ccg/broker/broker.sock"


def test_client_module_imports_no_policy_or_server():
    src = (CPG / "ccg" / "broker" / "client.py").read_text()
    for mod in ("server", "policy", "structured", "addr", "audit"):
        assert f"ccg.broker.{mod}" not in src and f"import {mod}" not in src, mod
    assert "from ccg.broker import framing\n" in src  # its one broker import: the frames


# ------------------------------------------------------------------ (18) the upstream sees exactly §2.3's request


def real_harness(tmp_path, upstream: Upstream, **kw):
    sysm = {"shop": load_system(tmp_path, eg=egress(port=upstream.port))}
    return Harness(tmp_path, systems=sysm, exchanger=structured.exchange, **kw)


def test_upstream_sees_exactly_the_request_line_and_headers(tmp_path):
    up = Upstream(http_reply(200, b'{"answer":42}'))
    try:
        hh = real_harness(tmp_path, up)
        g = hh.open("answer", "*")
        r = hh.ask(g["token"], model_call(endpoint=f"http://127.0.0.1:{up.port}"), {"messages": []})
        assert r["decision"]["decision"] == "allow" and r["response"]["status"] == 200
        want = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{up.port}\r\nContent-Type: application/json\r\n"
                "Content-Length: 15\r\nAccept: application/json\r\nAccept-Encoding: identity\r\n"
                'User-Agent: ccg-broker/1\r\n\r\n{"messages":[]}').encode()
        assert up.requests == [want]
        [close] = hh.records("close")
        assert close["closeReason"] == "completed" and close["httpStatus"] == 200
        assert close["bytesOut"] == len(want)
    finally:
        up.close()


def test_a_get_sends_no_body_headers(tmp_path):
    up = Upstream(http_reply(200, b"[]"))
    try:
        hh = real_harness(tmp_path, up)
        g = hh.open("answer", "*")
        hh.ask(g["token"], model_call(payloadClass="none", endpoint=f"http://127.0.0.1:{up.port}",
                                      target={"method": "GET", "path": "/v1/chat/completions"}))
        assert up.requests[0].split(b"\r\n") == [
            b"GET /v1/chat/completions HTTP/1.1", f"Host: 127.0.0.1:{up.port}".encode(), b"Accept: application/json",
            b"Accept-Encoding: identity", b"User-Agent: ccg-broker/1", b"", b""]
    finally:
        up.close()


def test_a_redirect_is_returned_not_followed(tmp_path):
    up = Upstream(http_reply(302, b"", (("Location", "/elsewhere"),)))
    try:
        hh = real_harness(tmp_path, up)
        g = hh.open("answer", "*")
        r = hh.ask(g["token"], model_call(endpoint=f"http://127.0.0.1:{up.port}"), {"m": 1})
        assert r["response"]["status"] == 302 and len(up.requests) == 1
    finally:
        up.close()


def test_chunked_and_read_until_close_responses(tmp_path):
    def chunked(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: text/plain\r\n\r\n"
                     b"5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\n\r\n")

    def until_close(conn):
        conn.sendall(b"HTTP/1.0 200 OK\r\n\r\nabc")
    for reply, body in ((chunked, b"hello world"), (until_close, b"abc")):
        up = Upstream(reply)
        try:
            hh = real_harness(tmp_path / reply.__name__ if (tmp_path / reply.__name__).mkdir() is None else tmp_path, up)
            c = hh.open("answer", "*")
            r = hh.ask(c["token"], model_call(endpoint=f"http://127.0.0.1:{up.port}"), {"m": 1})
            import base64
            assert base64.b64decode(r["response"]["bodyB64"]) == body
        finally:
            up.close()


# ------------------------------------------------------------------ (19) the bounds


def raw_reply(total: int, extra_headers: bytes = b""):
    """A response whose wire size is exactly ``total`` bytes."""
    head = b"HTTP/1.1 200 OK\r\n" + extra_headers + b"Content-Length: %d\r\n\r\n"
    n = total - len(head % 0)
    while len(head % n) + n != total:
        n += total - (len(head % n) + n)
    return lambda conn: conn.sendall((head % n) + b"x" * n)


def _bound_run(tmp_path, monkeypatch, wire_bytes: int, bound: int = 600, **kw):
    monkeypatch.setattr(framing, "MAX_RESPONSE_BYTES", bound)
    up = Upstream(kw.pop("reply", None) or raw_reply(wire_bytes))
    try:
        hh = real_harness(tmp_path, up)
        g = hh.open("answer", "*")
        r = hh.ask(g["token"], model_call(endpoint=f"http://127.0.0.1:{up.port}"), {"m": 1})
        return r, hh.records("close")[0]
    finally:
        up.close()


def test_a_response_at_the_bound_completes_and_one_byte_over_is_discarded(tmp_path, monkeypatch):
    r, close = _bound_run(tmp_path / "a", monkeypatch, 600) if (tmp_path / "a").mkdir() is None else None
    assert r["response"]["status"] == 200 and close["closeReason"] == "completed" and close["bytesIn"] == 600
    r, close = _bound_run(tmp_path / "b", monkeypatch, 601) if (tmp_path / "b").mkdir() is None else None
    assert r["response"] is None and r["decision"]["decision"] == "allow"
    assert close["closeReason"] == "response-too-large" and close["bytesIn"] == 601 and close["httpStatus"] is None


def test_a_gzip_content_encoding_is_discarded(tmp_path, monkeypatch):
    def gz(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 2\r\n\r\n\x1f\x8b")
    r, close = _bound_run(tmp_path, monkeypatch, 0, bound=10**6, reply=gz)
    assert r["response"] is None and close["closeReason"] == "upstream-error"


def test_a_trickling_server_hits_the_wall_clock_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(framing, "UPSTREAM_TIMEOUT_S", 1.0)

    def trickle(conn):
        for b in b"HTTP/1.1 200 OK\r\nX-Slow: " + b"a" * 40:  # a byte every 0.2 s: never a per-read timeout of 1 s
            conn.sendall(bytes([b]))
            time.sleep(0.2)
    up = Upstream(trickle)
    try:
        hh = real_harness(tmp_path, up)
        g = hh.open("answer", "*")
        t0 = time.monotonic()
        r = hh.ask(g["token"], model_call(endpoint=f"http://127.0.0.1:{up.port}"), {"m": 1})
        took = time.monotonic() - t0
        assert r["response"] is None and hh.records("close")[0]["closeReason"] == "timeout"
        assert took < 3.0  # the deadline is the whole exchange (1 s), not per read (the trickle lasts ~13 s)
    finally:
        up.close()


def test_connection_refused_is_an_upstream_error_with_a_close_record(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens
    sysm = {"shop": load_system(tmp_path, eg=egress(port=port))}
    hh = Harness(tmp_path, systems=sysm, exchanger=structured.exchange)
    g = hh.open("answer", "*")
    r = hh.ask(g["token"], model_call(endpoint=f"http://127.0.0.1:{port}"), {"m": 1})
    assert r["decision"]["decision"] == "allow" and r["response"] is None
    assert hh.records("close")[0]["closeReason"] == "upstream-error"


def test_the_reply_frame_cap_admits_a_full_size_base64_response():
    # MAX_RESPONSE_BYTES raw bytes travel as base64 (4/3): the frame cap must not be smaller than that
    n = framing.MAX_RESPONSE_BYTES
    assert framing.max_reply_frame() >= (n + 2) // 3 * 4 + 1000
    assert framing.max_request_frame() == framing.MAX_REQUEST_BYTES + 64 * 1024


def test_the_exact_constants_of_section_2_3():
    assert (framing.MAX_REQUEST_BYTES, framing.MAX_RESPONSE_BYTES, framing.UPSTREAM_TIMEOUT_S) == (
        4 * 2**20, 16 * 2**20, 300)
    assert structured.CA_BUNDLE == "/etc/ssl/certs/ca-certificates.crt"


# ------------------------------------------------------------------ (17) the closed environment


HOME = pwd.getpwuid(os.getuid()).pw_dir


def test_closed_home_comes_from_the_passwd_entry_not_the_environment(monkeypatch):
    monkeypatch.setenv("HOME", "/not/the/passwd/home")
    assert server.closed_environment(None)["HOME"] == HOME
    assert server.passwd_home() == HOME


def test_closed_environment_is_exactly_four_variables():
    assert server.closed_environment(None) == {"PATH": "/usr/bin:/bin", "HOME": HOME, "LANG": "C.UTF-8"}
    assert server.closed_environment("/a") == {"PATH": "/usr/bin:/bin", "HOME": HOME, "LANG": "C.UTF-8",
                                               "CCG_AUDIT_DIR": "/a"}


POLLUTED = {"PATH": "/x", "HOME": "/x", "LANG": "xx", "PYTHONPATH": "/evil", "SSL_CERT_FILE": "/evil.pem",
            "SSL_CERT_DIR": "/evil", "LOCALDOMAIN": "evil", "RES_OPTIONS": "ndots:9", "HOSTALIASES": "/evil",
            "HTTPS_PROXY": "http://evil", "CCG_AUDIT_DIR": "/aud", "OTHER": "1"}


def test_reexec_replaces_the_environment_and_isolates_the_interpreter():
    calls = []
    server.reexec_if_needed(["--system", "m.yaml"], dict(POLLUTED), execve=lambda *a: calls.append(a),
                            executable="/py/bin/python", isolated=0)
    [(exe, argv, env)] = calls
    assert exe == "/py/bin/python"
    assert argv == ["/py/bin/python", "-I", "-m", "ccg.broker.server", "--system", "m.yaml"]
    assert env == {"PATH": "/usr/bin:/bin", "HOME": HOME, "LANG": "C.UTF-8", "CCG_AUDIT_DIR": "/aud"}


@pytest.mark.parametrize("environ,isolated,skips", [
    ({"PATH": "/usr/bin:/bin", "HOME": HOME, "LANG": "C.UTF-8"}, 1, True),
    ({"PATH": "/usr/bin:/bin", "HOME": HOME, "LANG": "C.UTF-8", "CCG_AUDIT_DIR": "/a"}, 1, True),
    ({"PATH": "/usr/bin:/bin", "HOME": HOME, "LANG": "C.UTF-8"}, 0, False),                 # right env, no -I
    ({"PATH": "/usr/bin:/bin", "HOME": HOME, "LANG": "C.UTF-8", "PYTHONPATH": "/e"}, 1, False),  # -I, extra variable
    ({"PATH": "/usr/bin:/bin", "HOME": "/other", "LANG": "C.UTF-8"}, 1, False),
    ({"PATH": "/usr/bin:/bin", "HOME": HOME}, 1, False),
])
def test_reexec_is_skipped_only_when_already_exactly_closed(environ, isolated, skips):
    calls = []
    out = server.reexec_if_needed([], dict(environ), execve=lambda *a: calls.append(a), executable="/p", isolated=isolated)
    assert (out is False) == skips and bool(calls) != skips


def test_tls_context_names_only_the_fixed_bundle():
    seen = []

    def create(*a, **kw):
        seen.append((a, kw))
        return ssl.create_default_context()
    structured.make_tls_context(create)
    assert seen == [((), {"cafile": "/etc/ssl/certs/ca-certificates.crt"})]  # no default verify paths


def test_the_real_tls_context_ignores_ssl_cert_file(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "/nonexistent")
    ctx = structured.make_tls_context()
    assert ctx.check_hostname and ctx.verify_mode == ssl.CERT_REQUIRED and ctx.cert_store_stats()["x509_ca"] > 0


def test_server_module_imports_nothing_before_the_reexec_guard():
    src = (CPG / "ccg" / "broker" / "server.py").read_text()
    head = src[: src.index('if __name__ == "__main__":')]
    imports = [ln for ln in head.splitlines() if ln.startswith(("import ", "from "))]
    assert imports == ["import os", "import pwd", "import sys"]


# ------------------------------------------------------------------ process tests: the real broker


def start(tmp_path: Path, *, args=None, env=None, cwd=None, manifest=None, sock=None, audit=None, wait=True,
          name="b") -> subprocess.Popen:
    d = tmp_path / name
    d.mkdir(exist_ok=True, mode=0o700)
    manifest = manifest or write_manifest(d)
    sock = sock or d / "broker.sock"
    audit = audit or d / "audit"
    e = {"PATH": os.environ["PATH"], "CCG_AUDIT_DIR": str(audit), **(env or {})}
    argv = [sys.executable, "-m", "ccg.broker.server"] + (args if args is not None else ["--system", str(manifest), "--socket", str(sock)])
    p = subprocess.Popen(argv, cwd=str(cwd or d), env=e, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    p.sock = sock
    p.audit = audit
    if wait:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if p.poll() is not None:
                raise AssertionError(f"broker exited {p.returncode}: {p.stderr.read().decode()}")
            try:
                c = socket.socket(socket.AF_UNIX)
                c.connect(str(sock))
                c.close()
                return p
            except OSError:
                time.sleep(0.1)
        p.kill()
        raise AssertionError("broker did not listen in time")
    return p


def stop(p: subprocess.Popen) -> int:
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
    try:
        rc = p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        raise
    p.stderr.close()
    return rc


def test_process_starts_serves_and_stops_cleanly(tmp_path):
    p = start(tmp_path)
    try:
        g = BrokerClient(str(p.sock), timeout=5).open_run("answer", "shop", "*")
        assert len(g.token) == 64
        lines = (p.audit / "egress.jsonl").read_text().splitlines()
        assert json.loads(lines[0])["auditDir"] == str(p.audit) and json.loads(lines[0])["runId"] == g.run_id
    finally:
        assert stop(p) == 0
    assert not p.sock.exists()  # a clean stop removes the socket


def test_the_process_runs_isolated_under_the_closed_environment(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    pp = tmp_path / "pp"
    pp.mkdir()
    marks = []
    for where in (cwd, pp):  # planted `json`, in the working directory and on PYTHONPATH
        m = where / "MARK"
        (where / "json.py").write_text(f"open({str(m)!r}, 'w').write('x')\nraise ImportError('planted')\n")
        marks.append(m)
    env = {"PYTHONPATH": str(pp), "SSL_CERT_FILE": "/evil.pem", "LOCALDOMAIN": "evil", "RES_OPTIONS": "ndots:9",
           "HOSTALIASES": "/evil", "HTTPS_PROXY": "http://evil.invalid", "OTHER": "1"}
    p = start(tmp_path, env=env, cwd=cwd)
    try:
        environ = dict(kv.split("=", 1) for kv in Path(f"/proc/{p.pid}/environ").read_text().split("\0") if kv)
        assert environ == {"PATH": "/usr/bin:/bin", "HOME": HOME, "LANG": "C.UTF-8", "CCG_AUDIT_DIR": str(p.audit)}
        cmd = Path(f"/proc/{p.pid}/cmdline").read_text().split("\0")
        assert cmd[1:4] == ["-I", "-m", "ccg.broker.server"]
        BrokerClient(str(p.sock), timeout=5).open_run("answer", "shop", "*")  # it works, so the real json was imported
    finally:
        assert stop(p) == 0
    assert [m.exists() for m in marks] == [False, False]


def test_a_planted_module_wins_without_the_reexec(tmp_path):
    """Positive control: the planted module is imported when the interpreter is not isolated, so the check above
    can fail."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "json.py").write_text(f"open({str(cwd / 'MARK')!r}, 'w').write('x')\nraise ImportError('planted')\n")
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, ''); import ccg.broker.framing"], cwd=cwd,
                       capture_output=True)
    assert (cwd / "MARK").exists() and r.returncode != 0


# -- (16) singleton and directory checks


def test_a_second_broker_with_another_audit_dir_exits_69_and_leaves_the_first_alone(tmp_path):
    a = start(tmp_path)
    try:
        ino = os.stat(a.sock).st_ino
        d = tmp_path / "b"
        second = start(tmp_path, wait=False, sock=a.sock, audit=tmp_path / "other-audit", name="b2",
                       manifest=write_manifest(tmp_path / "b"))
        assert second.wait(timeout=15) == 69
        err = second.stderr.read().decode()
        assert "broker.lock" in err
        second.stderr.close()
        assert os.stat(a.sock).st_ino == ino and not (tmp_path / "other-audit").exists()
        BrokerClient(str(a.sock), timeout=5).open_run("answer", "shop", "*")  # the first still serves
    finally:
        stop(a)


def test_a_live_socket_without_the_lock_exits_69_and_is_not_unlinked(tmp_path):
    d = tmp_path / "live"
    d.mkdir(mode=0o700)
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(d / "broker.sock"))
    s.listen(1)
    try:
        p = start(tmp_path, wait=False, sock=d / "broker.sock", name="live")
        assert p.wait(timeout=15) == 69 and b"already listening" in p.stderr.read()
        p.stderr.close()
        assert (d / "broker.sock").exists()
    finally:
        s.close()


def test_a_stale_socket_file_is_replaced(tmp_path):
    d = tmp_path / "st"
    d.mkdir(mode=0o700)
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(d / "broker.sock"))
    s.close()
    p = start(tmp_path, sock=d / "broker.sock", name="st")
    try:
        BrokerClient(str(p.sock), timeout=5).open_run("answer", "shop", "*")
    finally:
        stop(p)


@pytest.mark.parametrize("mk,why", [
    (lambda d: os.chmod(d, 0o755), "mode"),
    (lambda d: os.chmod(d, 0o770), "mode"),
    (lambda d: os.chmod(d.parent, 0o770), "ancestor"),
])
def test_directory_checks_refuse_the_start(tmp_path, mk, why):
    outer = tmp_path / "outer"
    outer.mkdir(mode=0o700)
    d = outer / "dir"
    d.mkdir(mode=0o700)
    mk(d)
    man = write_manifest(tmp_path, name="m.yaml")
    p = start(tmp_path, wait=False, sock=d / "broker.sock", manifest=man, name="x")
    assert p.wait(timeout=15) == 69
    assert why in p.stderr.read().decode()
    p.stderr.close()


def test_a_broker_directory_under_the_sticky_tmp_exits_69_on_the_ancestor_check():
    # pins the server.py call site: strict (no sticky exemption), so `/tmp` (1777) as an ancestor refuses the start
    base = Path(tempfile.mkdtemp(dir="/tmp", prefix="ccg-sticky-"))
    p = None
    try:
        d = base / "dir"
        d.mkdir(mode=0o700)
        man = write_manifest(base, name="m.yaml")
        p = start(base, wait=False, sock=d / "broker.sock", manifest=man, name="x")
        assert p.wait(timeout=15) == 69
        err = p.stderr.read().decode()
        p.stderr.close()
        assert "ancestor /tmp is group- or world-writable (1777)" in err
    finally:
        if p is not None and p.poll() is None:  # a regression would leave a live broker: do not orphan it
            p.kill()
            p.wait()
        shutil.rmtree(base, ignore_errors=True)


def test_a_symlinked_broker_directory_and_lock_file_refuse_the_start(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    os.symlink(real, tmp_path / "link")
    man = write_manifest(tmp_path, name="m.yaml")
    p = start(tmp_path, wait=False, sock=tmp_path / "link" / "broker.sock", manifest=man, name="x")
    assert p.wait(timeout=15) == 69 and b"plain directory" in p.stderr.read()
    p.stderr.close()
    d = tmp_path / "lk"
    d.mkdir(mode=0o700)
    os.symlink(tmp_path / "elsewhere", d / "broker.lock")
    p = start(tmp_path, wait=False, sock=d / "broker.sock", manifest=man, name="y")
    assert p.wait(timeout=15) == 69
    p.stderr.close()
    assert not (tmp_path / "elsewhere").exists()  # O_NOFOLLOW: the link was not followed (nothing created)


def test_an_unsafe_audit_file_refuses_the_start(tmp_path):
    aud = tmp_path / "aud"
    aud.mkdir(mode=0o700)
    os.symlink(tmp_path / "x", aud / "egress.jsonl")
    p = start(tmp_path, wait=False, audit=aud, name="a")
    assert p.wait(timeout=15) == 69
    p.stderr.close()


def test_configuration_errors_exit_64(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    bad = d / "bad.yaml"
    bad.write_text("schemaVersion: 1\n")
    out = {}
    cases = {
        "no-system": [],
        "unknown-arg": ["--system", str(write_manifest(d)), "--bogus"],
        "bad-manifest": ["--system", str(bad), "--socket", str(d / "s.sock")],
        "missing-file": ["--system", str(d / "nope.yaml"), "--socket", str(d / "s.sock")],
        "s6-public-literal": ["--system", str(write_manifest(d, eg=egress(local_host="8.8.8.8"), name="s6.yaml")),
                              "--socket", str(d / "s.sock")],
        "twice": ["--system", str(write_manifest(d, name="a.yaml")), "--system", str(write_manifest(d, name="b.yaml")),
                  "--socket", str(d / "s.sock")],
    }
    for name, args in cases.items():
        p = start(tmp_path, args=args, wait=False, cwd=d, name="cfg")
        out[name] = (p.wait(timeout=15), p.stderr.read().decode())
        p.stderr.close()
    assert {k: v[0] for k, v in out.items()} == {k: 64 for k in cases}
    assert "S6" in out["s6-public-literal"][1] and "bad.yaml" in out["bad-manifest"][1]
    assert not (d / "s.sock").exists()


# ------------------------------------------------------------------ review fixes: F2, F3, catch-all, frames, strict dir


def _hosted(tmp_path, **kw):
    return Harness(tmp_path, systems={"shop": load_system(tmp_path, repos=(("a", "hosted"),))}, **kw)


def _hosted_allow(hh):
    from broker_helpers import hosted_call
    g = hh.open("answer", ["a"])
    return hh.broker.handle({"op": "request", "token": g["token"], "request": hosted_call(codebases=["a"]),
                             "body": {"m": 1}})


def test_a_failing_tls_factory_still_ends_the_allow_in_an_upstream_error_close(tmp_path):
    def boom():
        raise FileNotFoundError("/etc/ssl/certs/ca-certificates.crt")
    hh = _hosted(tmp_path)
    hh.broker.tls_factory = boom
    r = _hosted_allow(hh)
    assert r["decision"]["decision"] == "allow" and r["response"] is None
    [close] = hh.records("close")
    assert close["closeReason"] == "upstream-error" and close["decisionAuditId"] == r["decision"]["auditId"]
    assert hh.records("decision")[-1]["decision"] == "allow" and not hh.audit.latched


def test_a_failing_exchange_still_ends_the_allow_in_an_upstream_error_close(tmp_path):
    def boom(*a, **kw):
        raise RuntimeError("bug")
    hh = _hosted(tmp_path, exchanger=boom)
    r = _hosted_allow(hh)
    assert r["response"] is None and [c["closeReason"] for c in hh.records("close")] == ["upstream-error"]


def test_a_handler_defect_closes_the_connection_and_is_logged_by_type(served):
    hh, srv = served

    def bad(msg):
        raise RuntimeError("secret " * 1000)
    hh.broker.handle = bad
    assert raw(srv.sock_path, frame(b'{"op":"close_run"}')) == b""
    assert any(m == "handler failed: RuntimeError" for m in hh.logs)
    assert all(len(m) < 200 for m in hh.logs)  # the exception text is never logged
    del hh.broker.handle  # the server still serves afterwards
    assert raw(srv.sock_path, frame(b'{"op":"close_run","token":"x"}'))


def test_a_lone_surrogate_frame_is_malformed_and_the_broker_keeps_serving(served):
    hh, srv = served
    bad = b'{"op":"request","token":"a","request":{"target":{"path":"\\ud800"}},"body":null}'
    assert raw(srv.sock_path, frame(bad)) == b""
    assert hh.records() == [] and not hh.audit.latched
    assert reply_ok(raw(srv.sock_path, frame(b'{"op":"open_run","runKind":"answer","systemId":"shop","codebases":"*","override":null}')))


def reply_ok(out: bytes) -> bool:
    return json.loads(out[4:])["ok"] is True


def test_log_lines_never_carry_attacker_sized_text(served):
    hh, srv = served
    big = b'{"' + b"k" * 100_000 + b'":1,"' + b"k" * 100_000 + b'":2}'
    assert raw(srv.sock_path, frame(big)) == b""
    assert raw(srv.sock_path, frame(b'{"op":"' + b"o" * 100_000 + b'"}')) == b""
    assert len(hh.logs) == 2 and all(len(m) < 300 for m in hh.logs)


def test_a_junk_codebases_declaration_in_a_vulndb_run_is_codebases_in_vulndb(tmp_path):
    from test_broker_policy import _vuln
    hh = Harness(tmp_path)
    g = hh.open("vulndb", [])
    for junk in (["UPPER"], ["a"] * 9, "front", 5, {"a": 1}, ["front"]):
        hh.ask(g["token"], _vuln(codebases=junk))
        assert hh.last_decision()["reason"] == "codebases-in-vulndb", junk
    for ok_empty in ([], None):
        hh.ask(g["token"], _vuln(codebases=ok_empty))
    assert [d["reason"] for d in hh.records("decision")[-2:]] == [None, "codebases-not-subset"]


def test_a_body_key_that_does_not_fit_the_record_is_null_payload_fields_with_a_digest(tmp_path):
    import hashlib
    hh = Harness(tmp_path)
    g = hh.open("answer", "*")
    hh.ask(g["token"], model_call(), {"k" * 65: 1, "ok": 2})
    d = hh.last_decision()
    assert d["decision"] == "allow" and d["payloadFields"] is None and not hh.audit.latched
    want = hashlib.sha256(json.dumps(sorted(["k" * 65, "ok"]), separators=(",", ":")).encode()).hexdigest()
    assert d["rejected"] == {"payloadFields": want}
    hh.ask(g["token"], model_call(), {"k" * 64: 1})  # at the bound it is recorded
    assert hh.last_decision()["payloadFields"] == ["k" * 64]


def test_a_caller_path_that_cannot_be_an_allow_list_path_is_junk(tmp_path):
    hh = Harness(tmp_path)
    g = hh.open("answer", "*")
    for bad in ("/a b", "/a?x=1", "/a\n", "a", "/\u00e9", "/a#f"):
        hh.ask(g["token"], model_call(target={"method": "POST", "path": bad}), {"m": 1})
        d = hh.last_decision()
        assert d["target"]["path"] is None and "target.path" in d["rejected"] and d["reason"] == "path-not-allowed", bad


def test_the_process_refuses_a_sticky_world_writable_ancestor(tmp_path):
    """`main` follows section 2.0 strictly: `/tmp` (sticky, world-writable) is refused, no exemption."""
    d = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        man = write_manifest(tmp_path, name="m.yaml")
        os.chmod(d, 0o700)
        p = start(tmp_path, wait=False, sock=d / "broker.sock", manifest=man, name="sticky")
        assert p.wait(timeout=15) == 69
        assert "ancestor" in p.stderr.read().decode()
        p.stderr.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_each_layer_clips_caller_text_on_its_own(served, monkeypatch):
    # framing: a duplicate key and an unknown op, parsed or dispatched directly
    with pytest.raises(framing.FrameError) as e:
        framing.parse_strict(b'{"' + b"k" * 50_000 + b'":1,"' + b"k" * 50_000 + b'":2}')
    assert len(str(e.value)) < 200
    hh, srv = served
    with pytest.raises(framing.FrameError) as e:
        hh.broker.handle({"op": "o" * 50_000})
    assert len(str(e.value)) < 200
    # server: a frame or message error that carries a huge message is cut before it reaches the log
    def huge_frame(recv, cap):
        raise framing.FrameError("x" * 50_000)
    monkeypatch.setattr(framing, "read_frame", huge_frame)
    assert raw(srv.sock_path, frame(b"{}")) == b""
    monkeypatch.undo()

    def huge_msg(msg):
        raise framing.FrameError("y" * 50_000)
    hh.broker.handle = huge_msg
    assert raw(srv.sock_path, frame(b"{}")) == b""
    assert len(hh.logs) == 2 and all(len(m) < 300 for m in hh.logs)
