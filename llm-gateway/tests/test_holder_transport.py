"""`HolderTransport` against a scripted fake holder on a real Unix socket (plan §3.3, §3.4):
what a caller must do (sign with a fresh nonce, verify the reply, map what did not complete) and
must refuse (a local request, an unprotected configuration, a holder in its own uid)."""

import os
import socket
import threading

import pytest

from llmgateway import framing, keyholder
from llmgateway.config import GatewayConfig
from llmgateway.failures import Condition
from llmgateway.transport import (
    DEFAULT_HOLDER_SOCKET,
    HolderTransport,
    TransportFailure,
    WireRequest,
)
from llmgateway.types import CallFailed, FailureClass

SECRET = bytes(range(32))
UID = os.getuid()


class FakeHolder:
    """Serves each connection through `handler(header, body, nonce) -> frame | None | 'hang'`."""

    def __init__(self, directory, handler):
        self.path = str(directory / "h.sock")
        self.handler = handler
        self.requests = []
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        self.sock.listen(8)
        self.threads = []
        self.stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        self.sock.settimeout(0.1)
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except (TimeoutError, OSError):
                continue
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn):
        try:
            payload = framing.read_frame(conn, max_bytes=1 << 20, timeout=5)
            header, body = framing.parse_request(payload)
            self.requests.append((header, body))
            out = self.handler(header, body)
            if out == "hang":
                self.stop.wait(3)
            elif out is not None:
                conn.sendall(out)
        finally:
            conn.close()

    def close(self):
        self.stop.set()
        self.sock.close()


def reply(secret=SECRET, *, reason="completed", status=200, ctype="application/json", body=b"{}"):
    def handler(header, _body):
        return framing.build_response(
            secret,
            request_nonce=header.nonce,
            close_reason=reason,
            status=status,
            content_type=ctype,
            body=body,
        )

    return handler


def config(path, **over):
    deployment = {"holder_socket": path, "holder_uid": UID, **over}
    return GatewayConfig(protected=True, deployment=deployment)


def wire(**kw):
    base = dict(
        provider_id="anth",
        kind="anthropic",
        base_url="https://api.example.com",
        path="/v1/messages",
        body=b'{"model":"m"}',
        hosted=True,
        run_id="eval-1",
        timeout_s=2.0,
    )
    return WireRequest(**{**base, **kw})


@pytest.fixture
def holder(tmp_path):
    made = []

    def make(handler):
        h = FakeHolder(tmp_path, handler)
        made.append(h)
        return h

    yield make
    for h in made:
        h.close()


def transport(path, **kw):
    return HolderTransport(config(path), SECRET, own_uid=-1, **kw)


def test_default_socket_matches_the_holders():
    assert DEFAULT_HOLDER_SOCKET == keyholder.DEFAULT_SOCKET


def test_a_completed_exchange_is_signed_and_returned_as_is(holder):
    h = holder(reply(status=429, body=b'{"e":1}'))
    got = transport(h.path).exchange(wire())
    assert (got.status, got.body, got.content_type) == (429, b'{"e":1}', "application/json")
    ((header, body),) = h.requests
    assert (header.caller, header.endpoint, header.path, header.run_id) == (
        "service",
        "https://api.example.com",
        "/v1/messages",
        "eval-1",
    )
    assert header.mac == framing.request_mac(
        SECRET,
        caller="service",
        ts=header.ts,
        nonce=header.nonce,
        method="POST",
        endpoint=header.endpoint,
        path=header.path,
        run_id="eval-1",
        body=body,
    )


def test_every_exchange_signs_a_fresh_nonce(holder):
    h = holder(reply())
    t = transport(h.path)
    for _ in range(3):
        t.exchange(wire())
    nonces = [hdr.nonce for hdr, _ in h.requests]
    assert len(set(nonces)) == 3


def test_a_refusal_exchange_comes_back_for_the_engine_to_classify(holder):
    h = holder(
        reply(
            status=499,
            ctype=framing.REFUSAL_CONTENT_TYPE,
            body=framing.refusal_body("run_unmatched"),
        )
    )
    got = transport(h.path).exchange(wire())
    assert got.status == 499 and got.content_type == framing.REFUSAL_CONTENT_TYPE


@pytest.mark.parametrize(
    "reason, condition",
    [
        ("upstream-error", Condition.BROKER_UPSTREAM_ERROR),
        ("timeout", Condition.BROKER_TIMEOUT),
        ("response-too-large", Condition.BROKER_RESPONSE_TOO_LARGE),
    ],
)
def test_an_exchange_that_did_not_complete_is_a_transport_failure(holder, reason, condition):
    h = holder(reply(reason=reason, status=0, ctype="", body=b""))
    with pytest.raises(TransportFailure) as info:
        transport(h.path).exchange(wire())
    assert info.value.condition is condition


def test_a_reply_signed_by_another_secret_is_rejected_as_config(holder):
    h = holder(reply(secret=bytes(range(1, 33))))
    with pytest.raises(CallFailed) as info:
        transport(h.path).exchange(wire())
    assert info.value.failure_class is FailureClass.CONFIG


def test_an_unsigned_reply_is_rejected(holder):
    h = holder(reply(secret=None))
    with pytest.raises(CallFailed) as info:
        transport(h.path).exchange(wire())
    assert info.value.failure_class is FailureClass.CONFIG


def test_a_silent_holder_is_a_timeout_and_a_missing_socket_is_unreachable(holder, tmp_path):
    h = holder(lambda header, body: "hang")
    with pytest.raises(TransportFailure) as info:
        transport(h.path).exchange(wire(timeout_s=0.3))
    assert info.value.condition is Condition.SOCKET_TIMEOUT
    with pytest.raises(TransportFailure) as info:
        transport(str(tmp_path / "absent.sock")).exchange(wire())
    assert info.value.condition is Condition.HOLDER_SOCKET_UNREACHABLE


def test_a_socket_directory_not_owned_by_the_holder_is_unreachable(holder):
    h = holder(reply())
    t = HolderTransport(config(h.path, holder_uid=UID + 1), SECRET, own_uid=-1)
    with pytest.raises(TransportFailure) as info:
        t.exchange(wire())
    assert info.value.condition is Condition.HOLDER_SOCKET_UNREACHABLE
    assert h.requests == []  # nothing was sent to a fake holder


@pytest.mark.parametrize(
    "cfg, request_",
    [
        (GatewayConfig(protected=False, deployment={"holder_uid": 1}), {}),
        (config("/x/h.sock", holder_uid=None), {}),
        (config("/x/h.sock", holder_uid=True), {}),
        (config("relative.sock", holder_uid=1), {}),
        (config("/x/h.sock", holder_uid=UID), {}),  # a holder in the caller's own uid
        (config("/x/h.sock", holder_uid=1), {"hosted": False}),  # a local step is not for us
    ],
)
def test_misconfiguration_and_misrouting_fail_config_before_a_socket_opens(
    cfg, request_, monkeypatch
):
    def boom(*a, **k):
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket.socket, "connect", boom)
    with pytest.raises(CallFailed) as info:
        HolderTransport(cfg, SECRET).exchange(wire(**request_))
    assert info.value.failure_class is FailureClass.CONFIG


def test_the_secret_is_checked_and_never_shown():
    with pytest.raises(ValueError):
        HolderTransport(config("/x"), b"short")
    assert SECRET.hex() not in repr(HolderTransport(config("/x"), SECRET))
    assert HolderTransport.reaches_hosted is True
