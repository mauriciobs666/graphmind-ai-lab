"""Security Pass 7, P7-1: a keyed provider is hosted whatever its origin, so it is never sent
direct and keyless. `DirectTransport` refuses it from the provider table (not only from the
wire flag), the residency wire request carries the real flag, and `provider_op` skips it."""

import http.client
import socket

import pytest
from gwstandin import write_config
from service_harness import LocalStandIn

from llmgateway.config import GatewayConfig
from llmgateway.load import Residency
from llmgateway.providers import REGISTRY
from llmgateway.records import RecordStore
from llmgateway.service import Service, ServiceSettings
from llmgateway.transport import DirectTransport, WireRequest, WireResponse
from llmgateway.types import HostedFromCallerProcess

CONFIG = """
[deployment]
local_endpoints = ["{url}"]

[providers.plain]
kind = "lmstudio"
base_url = "{url}"
model = "m1"

[providers.keyed]
kind = "lmstudio"
base_url = "{url}"
model = "m1"
key_ref = "k1"
pinned_models = ["m1"]
max_tokens_ceiling = 100

[scopes.plain]

[call_sites.plain]
component = "answer.x"
scopes = ["plain"]
chain = ["plain"]

[call_sites.keyed]
component = "answer.x"
scopes = ["plain"]
chain = ["keyed"]
"""


@pytest.fixture
def lab(tmp_path, monkeypatch):
    local = LocalStandIn()
    src = write_config(tmp_path / "root", CONFIG.format(url=local.url))
    yield local, src, tmp_path
    local.close()


@pytest.fixture
def no_connect(monkeypatch):
    calls = []

    def boom(*args, **kw):
        calls.append(args)
        raise AssertionError("a connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", boom)
    return calls


def wire(url, provider_id, *, hosted=False):
    return WireRequest(
        provider_id=provider_id,
        kind="lmstudio",
        base_url=url,
        path="/api/v0/models",
        body=b"",
        method="GET",
        hosted=hosted,
    )


def test_direct_transport_refuses_a_keyed_provider_even_when_the_wire_says_local(lab, no_connect):
    local, src, _ = lab
    transport = DirectTransport(src.get)
    with pytest.raises(HostedFromCallerProcess):
        transport.exchange(wire(local.url, "keyed", hosted=False))
    assert no_connect == [] and local.received == []


def test_direct_transport_refuses_any_request_flagged_hosted(lab, no_connect):
    local, src, _ = lab
    with pytest.raises(HostedFromCallerProcess):
        DirectTransport(src.get).exchange(wire(local.url, "plain", hosted=True))
    assert no_connect == [] and local.received == []


def test_direct_transport_still_serves_a_plain_local_provider(lab):
    local, src, _ = lab
    got = DirectTransport(src.get).exchange(wire(local.url, "plain"))
    assert got.status == 200 and len(local.received) == 1


class Spy:
    reaches_hosted = False

    def __init__(self):
        self.seen = []

    def exchange(self, request):
        self.seen.append(request)
        return WireResponse(200, b'{"data": []}', "application/json")


@pytest.mark.parametrize("provider, expected", [("plain", False), ("keyed", True)])
def test_residency_wire_requests_carry_the_real_hosted_flag(lab, provider, expected):
    _, src, _ = lab
    config = src.get()
    spy = Spy()
    Residency(spy, REGISTRY.get("lmstudio"), provider, config.providers[provider]).loaded()
    assert [r.hosted for r in spy.seen] == [expected]


def test_residency_flag_is_never_weaker_than_the_step_flag(lab):
    _, src, _ = lab
    spy = Spy()
    Residency(
        spy, REGISTRY.get("lmstudio"), "plain", src.get().providers["plain"], hosted=True
    ).loaded()
    assert spy.seen[0].hosted is True


def service(lab):
    local, src, tmp = lab
    return Service(
        ServiceSettings(
            config_source=src,
            socket_path=str(tmp / "s.sock"),
            state_dir=str(tmp / "state"),
            credential_path=str(tmp / "absent.cred"),
        )
    )


def test_provider_op_on_a_keyed_provider_sends_nothing(lab, no_connect):
    local, _, _ = lab
    reply = service(lab).handle(
        {"v": 1, "op": "provider_op", "call_site": "keyed", "name": "lmstudio.models"}
    )
    assert reply["ok"] is False and reply["error"]["class"] == "policy_refused"
    assert reply["error"]["type"] == "CallFailed"  # refused by routing, not by the transport
    assert no_connect == [] and local.received == []


def test_provider_op_wire_request_carries_the_provider_hosted_flag(lab, monkeypatch):
    local, _, _ = lab
    svc = service(lab)
    spy = Spy()
    monkeypatch.setattr(svc, "transport", spy)
    reply = svc.handle(
        {"v": 1, "op": "provider_op", "call_site": "plain", "name": "lmstudio.models"}
    )
    assert reply["ok"] is True and [r.hosted for r in spy.seen] == [False]
    config: GatewayConfig = svc.s.config_source.get()
    assert config.provider_is_hosted("keyed") and not config.provider_is_hosted("plain")
    # provider_op is no model call: it writes no attempt record
    assert RecordStore(svc.s.state_dir).attempts() == []
