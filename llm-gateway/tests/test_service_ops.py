"""The service's `load`, `provider_op` and `call` ops carry the caller's scope, run id and the
marker source into the policy (security review K2/K3), and a connection never leaks its slot.
In-process `Service.handle`, no sockets except the slot test."""

import socket
import threading

import pytest
from gwstandin import write_config

from llmgateway.service import Service, ServiceSettings

CONFIG = """
[providers.lm]
kind = "lmstudio"
base_url = "http://127.0.0.1:1"
model = "m1"
[providers.anth]
kind = "anthropic"
base_url = "https://api.anthropic.com"
model = "claude-x"
[scopes.plain]
[scopes.secret]
marker = "restricted"
[scope_sources.ccg]
prefix = "ccg:"
[call_sites.mixed]
component = "a.b"
scopes = ["plain"]
chain = ["anth", "lm"]
[call_sites.local_load]
component = "a.b"
scopes = ["plain"]
chain = ["lm"]
load_mode = "explicit"
"""
MSGS = [{"role": "user", "content": "hi"}]


class Markers:
    def __init__(self, answer="restricted"):
        self.answer, self.asked = answer, []

    def marker(self, scope_id):
        self.asked.append(scope_id)
        return self.answer


def make(tmp_path, markers=None, **kw):
    return Service(
        ServiceSettings(
            config_source=write_config(tmp_path / "root", CONFIG),
            socket_path=str(tmp_path / "run" / "s.sock"),
            state_dir=str(tmp_path / "state"),
            credential_path=str(tmp_path / "absent.cred"),
            marker_source=markers,
            **kw,
        )
    )


def refusal_text(reply):
    assert reply["ok"] is False and reply["error"]["class"] == "policy_refused", reply
    return reply["error"]["message"]


OPS = {
    "provider_op": {"op": "provider_op", "call_site": "mixed", "name": "lmstudio.models"},
    "load": {"op": "load", "call_site": "local_load"},
}


@pytest.mark.parametrize("op", OPS)
def test_the_requests_scopes_reach_the_policy(tmp_path, op):
    svc = make(tmp_path)
    reply = svc.handle({"v": 1, **OPS[op], "scopes": ["secret"]})
    assert "restricted scope 'secret'" in refusal_text(reply)


@pytest.mark.parametrize("op", OPS)
def test_the_run_id_reaches_the_request_validation(tmp_path, op):
    svc = make(tmp_path)
    assert "run_id must match" in refusal_text(svc.handle({"v": 1, **OPS[op], "run_id": "bad id!"}))


@pytest.mark.parametrize("op", [*OPS, "call"])
def test_the_marker_source_reaches_every_scope_policy(tmp_path, op):
    markers = Markers("restricted")
    svc = make(tmp_path, markers)
    if op == "call":
        message = {
            "op": "call",
            "request": {"call_site": "mixed", "messages": MSGS, "scopes": ["ccg:x"]},
        }
    else:
        message = {**OPS[op], "scopes": ["ccg:x"]}
    assert "restricted scope 'ccg:x'" in refusal_text(svc.handle({"v": 1, **message}))
    assert markers.asked == ["ccg:x"]


def test_without_a_marker_source_a_source_backed_scope_is_unreadable(tmp_path):
    reply = make(tmp_path).handle({"v": 1, **OPS["provider_op"], "scopes": ["ccg:x"]})
    assert "marker of scope 'ccg:x' is unreadable" in refusal_text(reply)  # fails closed


def test_an_unsupported_protocol_version_is_refused_for_every_op(tmp_path):
    svc = make(tmp_path)
    for message in (*OPS.values(), {"op": "embed"}, {"op": "call", "request": {}}):
        assert "unsupported protocol version 2" in refusal_text(svc.handle({"v": 2, **message}))
        assert "unsupported protocol version None" in refusal_text(svc.handle(dict(message)))


def test_a_state_dir_in_the_deployment_table_is_used(tmp_path):
    cfg = CONFIG + '\n[deployment]\nstate_dir = "%s"\n' % (tmp_path / "deployed-state")
    src = write_config(tmp_path / "root", cfg)
    svc = Service(
        ServiceSettings(
            config_source=src,
            socket_path=str(tmp_path / "run" / "s.sock"),
            credential_path=str(tmp_path / "absent.cred"),
        )
    )
    assert svc.records.path.startswith(str(tmp_path / "deployed-state"))


def test_connections_release_their_slot_so_the_service_keeps_answering(tmp_path):
    svc = make(tmp_path, read_timeout=0.3, max_connections=2)
    svc.bind()
    threading.Thread(target=svc.serve_forever, daemon=True).start()
    path = str(tmp_path / "run" / "s.sock")
    try:
        for _ in range(6):  # three times the slots: each is a dropped (over-cap-less, empty) frame
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(path)
            s.settimeout(3)
            s.sendall((10**9).to_bytes(4, "big"))
            assert s.recv(10) == b""  # closed by the service, slot released in `finally`
            s.close()
        payload = b'{"v":1,"op":"embed"}'
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(path)
        s.settimeout(3)
        s.sendall(len(payload).to_bytes(4, "big") + payload)
        assert s.recv(4) != b""  # answered: a leaked slot would have closed this one unread
        s.close()
    finally:
        svc.close()
