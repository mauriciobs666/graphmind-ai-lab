"""`client.py` is a single stdlib file components byte-copy (plan §3.1, §3.10): it imports no
other part of the package, and its bytes are pinned so a drift is a deliberate act."""

import ast
import contextlib
import hashlib
import json
import os
import pathlib
import socket
import struct
import subprocess
import sys
import threading
import time

import pytest

from llmgateway import client as client_module
from llmgateway.client import (
    PERMANENT_CLASSES,
    TEMPORARY_CLASSES,
    CallFailed,
    Client,
    EmbeddingsNotAvailable,
    HostedFromCallerProcess,
    default_socket_path,
)
from llmgateway.types import FailureClass

CLIENT = pathlib.Path(client_module.__file__)
# Update deliberately: every component that copied the file must take the new bytes.
CLIENT_SHA256 = "0d1424d289718a9e1495ba367b9e308c68ce0f4580d46ee782a05dc9ef34e0fb"


def test_the_file_imports_only_the_standard_library():
    tree = ast.parse(CLIENT.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import needs the package"
            names.add(node.module.split(".")[0])
    assert names <= set(sys.stdlib_module_names) and "llmgateway" not in names


def test_the_file_imports_and_works_from_an_empty_sys_path(tmp_path):
    """The proof by execution: `-I` drops PYTHONPATH, the script directory and user site, the
    file is loaded by path from a scratch copy, and `llmgateway` must not appear."""
    copy = tmp_path / "copied" / "client.py"
    copy.parent.mkdir()
    copy.write_bytes(CLIENT.read_bytes())
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('gwclient', {str(copy)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['gwclient'] = m\n"
        "spec.loader.exec_module(m)\n"
        "assert not [n for n in sys.modules if n.startswith('llmgateway')], 'package imported'\n"
        "try:\n"
        f"    m.Client({str(tmp_path / 'absent' / 's.sock')!r}).call('x', [])\n"
        "except m.CallFailed as exc:\n"
        "    print(exc.failure_class)\n"
    )
    res = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(tmp_path)
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "connection"


def test_the_bytes_are_pinned():
    assert hashlib.sha256(CLIENT.read_bytes()).hexdigest() == CLIENT_SHA256


def test_failure_classes_agree_with_the_package():
    assert TEMPORARY_CLASSES | PERMANENT_CLASSES == {c.value for c in FailureClass}
    assert TEMPORARY_CLASSES == {c.value for c in FailureClass if c.temporary}


def test_default_socket_path_order():
    assert (
        default_socket_path({"LLMGATEWAY_SERVICE_SOCKET": "/a/s", "XDG_RUNTIME_DIR": "/r"})
        == "/a/s"
    )
    assert default_socket_path({"XDG_RUNTIME_DIR": "/r"}) == "/r/llmgateway/service.sock"
    assert default_socket_path({}).endswith("/llmgateway/service.sock")


class FakeService:
    """One-shot Unix server answering with `reply_bytes` (raw, so a test can send bad frames)."""

    def __init__(self, directory, reply_bytes, *, hang=False, trickle=0.0):
        directory.mkdir(mode=0o700, exist_ok=True)
        self.path = str(directory / "s.sock")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        self.sock.listen(4)
        self.requests = []
        self.reply_bytes, self.hang, self.trickle = reply_bytes, hang, trickle
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.sock.accept()
        with conn, contextlib.suppress(OSError, struct.error):
            (size,) = struct.unpack(">I", conn.recv(4))
            data = b""
            while len(data) < size:
                data += conn.recv(size - len(data))
            self.requests.append(json.loads(data))
            if self.hang:
                threading.Event().wait(2)
            else:
                if self.trickle:
                    for i in range(len(self.reply_bytes)):
                        conn.sendall(self.reply_bytes[i : i + 1])
                        threading.Event().wait(self.trickle)
                else:
                    conn.sendall(self.reply_bytes)

    def close(self):
        self.sock.close()


def frame(obj):
    data = json.dumps(obj).encode()
    return struct.pack(">I", len(data)) + data


@pytest.fixture
def fake(tmp_path):
    made = []

    def make(reply_bytes, **kw):
        f = FakeService(tmp_path / f"d{len(made)}", reply_bytes, **kw)
        made.append(f)
        return f

    yield make
    for f in made:
        f.close()


OK = {
    "ok": True,
    "result": {
        "call_id": "c1",
        "provider": "p",
        "model": "m",
        "hosted": True,
        "message": {"role": "assistant", "content": "hi"},
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "provider_detail": {"stats": {"a": 1}},
        "raw": {},
    },
}


def error(**kw):
    return {
        "ok": False,
        "error": {"type": "CallFailed", "class": "rate_limit", "message": "m", **kw},
    }


def test_a_call_sends_only_what_the_caller_named_and_reads_the_reply(fake):
    f = fake(frame(OK))
    reply = Client(f.path).call(
        "site", [{"role": "user", "content": "q"}], run_id="eval-1", settings={"max_tokens": 5}
    )
    assert (reply.model, reply.hosted, reply.usage.output_tokens) == ("m", True, 2)
    assert reply.provider_detail == {"stats": {"a": 1}}
    (sent,) = f.requests
    assert sent["v"] == 1 and sent["op"] == "call"
    assert set(sent["request"]) == {
        "call_site",
        "messages",
        "scopes",
        "run_id",
        "settings",
        "options",
    }


def test_errors_carry_class_code_and_attempts(fake):
    f = fake(frame(error(code="run_unmatched", call_id="c9", classes=["rate_limit", "run_bound"])))
    with pytest.raises(CallFailed) as info:
        Client(f.path).call("site", [])
    e = info.value
    assert (e.failure_class, e.code, e.call_id, e.classes) == (
        "rate_limit",
        "run_unmatched",
        "c9",
        ("rate_limit", "run_bound"),
    )
    assert e.temporary


@pytest.mark.parametrize(
    "kind, exc",
    [
        ("HostedFromCallerProcess", HostedFromCallerProcess),
        ("EmbeddingsNotAvailable", EmbeddingsNotAvailable),
    ],
)
def test_typed_errors(fake, kind, exc):
    f = fake(
        frame({"ok": False, "error": {"type": kind, "class": "policy_refused", "message": "no"}})
    )
    with pytest.raises(exc):
        Client(f.path).call("site", [])


def test_unreadable_replies_fail_config_and_a_silent_service_times_out(fake, tmp_path):
    for raw in (frame([1]), struct.pack(">I", 3) + b"{no", frame({"ok": True, "result": {}})):
        with pytest.raises(CallFailed) as info:
            Client(fake(raw).path).call("site", [])
        assert info.value.failure_class == "config"
    huge = struct.pack(">I", client_module.MAX_REPLY_BYTES + 1)
    with pytest.raises(CallFailed) as info:
        Client(fake(huge).path).call("site", [])
    assert info.value.failure_class == "config"
    with pytest.raises(CallFailed) as info:
        Client(fake(b"", hang=True).path, timeout=0.2).call("site", [])
    assert info.value.failure_class == "timeout"


def test_an_absent_service_is_connection_and_a_shared_directory_is_refused(tmp_path):
    with pytest.raises(CallFailed) as info:
        Client(str(tmp_path / "none" / "s.sock")).call("site", [])
    assert info.value.failure_class == "connection"
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)
    with pytest.raises(CallFailed) as info:
        Client(str(shared / "s.sock")).call("site", [])
    assert info.value.failure_class == "config"


def test_a_reply_that_trickles_past_the_total_deadline_times_out(fake):
    """Each byte arrives inside any per-`recv` timeout; only a total deadline stops it."""
    start = time.monotonic()
    with pytest.raises(CallFailed) as info:
        Client(fake(frame(OK), trickle=0.05).path, timeout=0.5).call("site", [])
    assert info.value.failure_class == "timeout"
    assert time.monotonic() - start < 1.5


def test_a_service_run_by_another_uid_is_refused_after_connecting(fake, monkeypatch):
    f = fake(frame(OK))
    monkeypatch.setattr(Client, "_check_directory", lambda self: None)
    other = os.getuid() + 1
    monkeypatch.setattr(client_module.os, "getuid", lambda: other)
    with pytest.raises(CallFailed, match="another account") as info:
        Client(f.path).call("site", [])
    assert info.value.failure_class == "config"
    assert f.requests == []  # nothing was sent to the foreign peer


def test_a_request_over_the_frame_cap_is_refused_before_sending(fake):
    f = fake(frame(OK))
    big = [{"role": "user", "content": "x" * client_module.MAX_REQUEST_BYTES}]
    with pytest.raises(CallFailed) as info:
        Client(f.path).call("site", big)
    assert info.value.failure_class == "rejected"
    assert f.requests == []
