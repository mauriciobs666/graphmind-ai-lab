"""DirectTransport against a real in-process HTTP stand-in (plan §3.2; FR-11, AC-4, AC-7)."""

import socket
import threading

import pytest
from gwstandin import StandIn, write_config

from llmgateway.config import GatewayConfig
from llmgateway.failures import Condition, classify_condition, classify_exception
from llmgateway.transport import DirectTransport, TransportFailure, WireRequest
from llmgateway.types import FailureClass, HostedFromCallerProcess


def wire(base_url, **kw):
    kw.setdefault("body", b'{"a":1}')
    return WireRequest(
        provider_id="p", kind="openai_compat", base_url=base_url, path="/v1/x", hosted=False, **kw
    )


UNPROTECTED = GatewayConfig(protected=False)  # loopback IP literals, any port


def test_exchange_roundtrip_returns_status_body_content_type_and_sends_body_verbatim():
    s = StandIn((200, "application/json; charset=utf-8", b'{"ok":true}'))
    try:
        r = DirectTransport(UNPROTECTED).exchange(wire(s.url, headers={"X-Extra": "1"}))
    finally:
        s.close()
    assert (r.status, r.body, r.content_type) == (
        200,
        b'{"ok":true}',
        "application/json; charset=utf-8",
    )
    got = s.received[0]
    assert got["path"] == "/v1/x" and got["body"] == b'{"a":1}'
    assert got["headers"]["Content-Type"] == "application/json" and got["headers"]["X-Extra"] == "1"


def test_error_status_is_a_response_not_an_exception_and_redirect_not_followed():
    s = StandIn((503, "text/plain", b"down"))
    s2 = StandIn((302, "text/plain", b""))
    try:
        assert DirectTransport(UNPROTECTED).exchange(wire(s.url)).status == 503
        assert DirectTransport(UNPROTECTED).exchange(wire(s2.url)).status == 302
    finally:
        s.close(), s2.close()


def test_base_url_path_prefix_is_kept():
    s = StandIn()
    try:
        DirectTransport(UNPROTECTED).exchange(wire(s.url + "/prefix/"))
    finally:
        s.close()
    assert s.received[0]["path"] == "/prefix/v1/x"


@pytest.fixture
def no_sockets(monkeypatch):
    attempts = []

    def deny(self, address):
        attempts.append(address)
        raise AssertionError(f"a socket connect to {address} was attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    return attempts


@pytest.mark.parametrize(
    "url",
    [
        "https://203.0.113.5",  # a remote IP literal
        "http://localhost:1234",  # a hostname is never local
        "https://api.anthropic.com",
        "http://127.0.0.1@203.0.113.5:80",  # userinfo trick
        "http://10.0.0.5:1234",  # private but not loopback
    ],
)
def test_hosted_or_non_local_url_is_refused_before_any_socket(url, no_sockets):
    with pytest.raises(HostedFromCallerProcess):
        DirectTransport(UNPROTECTED).exchange(wire(url))
    assert no_sockets == []


def test_protected_config_allow_list_decides(tmp_path):
    s = StandIn()
    other = StandIn()
    try:
        cfg = write_config(tmp_path / "r", f'[deployment]\nlocal_endpoints = ["{s.url}"]\n')
        t = DirectTransport(cfg.get)  # callable: re-evaluated on every exchange
        assert t.exchange(wire(s.url)).status == 200
        with pytest.raises(HostedFromCallerProcess):  # loopback, but not listed
            t.exchange(wire(other.url))
    finally:
        s.close(), other.close()
    assert other.received == []


def test_unreachable_port_raises_an_oserror_classified_connection():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(OSError) as e:
        DirectTransport(UNPROTECTED).exchange(wire(f"http://127.0.0.1:{port}"))
    assert classify_exception(e.value) is FailureClass.CONNECTION


def test_silent_server_times_out_classified_timeout():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    conns = []
    t = threading.Thread(target=lambda: conns.append(srv.accept()), daemon=True)
    t.start()
    try:
        with pytest.raises(TimeoutError) as e:
            DirectTransport(UNPROTECTED).exchange(
                wire(f"http://127.0.0.1:{srv.getsockname()[1]}", timeout_s=0.2)
            )
        assert classify_exception(e.value) is FailureClass.TIMEOUT
    finally:
        t.join(2)
        for c, _ in conns:
            c.close()
        srv.close()


def test_dropped_connection_is_a_connection_failure():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def drop():
        c, _ = srv.accept()
        c.recv(65536)
        c.close()

    t = threading.Thread(target=drop, daemon=True)
    t.start()
    try:
        with pytest.raises((OSError, TransportFailure)) as e:
            DirectTransport(UNPROTECTED).exchange(wire(f"http://127.0.0.1:{srv.getsockname()[1]}"))
        cls = (
            classify_exception(e.value)
            if isinstance(e.value, OSError)
            else classify_condition(e.value.condition)
        )
        assert cls is FailureClass.CONNECTION
    finally:
        t.join(2)
        srv.close()


def test_response_over_the_bound_is_a_connection_class_failure():
    s = StandIn((200, "application/json", b"x" * 100))
    try:
        with pytest.raises(TransportFailure) as e:
            DirectTransport(UNPROTECTED, max_response_bytes=10).exchange(wire(s.url))
    finally:
        s.close()
    assert e.value.condition is Condition.BROKER_RESPONSE_TOO_LARGE


def test_reaches_hosted_is_false():
    assert DirectTransport(UNPROTECTED).reaches_hosted is False
