"""Helpers for the provider-adapter tests (D5): a real stdlib stand-in server, a gateway built
with the real provider registry, and a transport that may carry a hosted step to the stand-in
(DirectTransport refuses hosted by design, so hosted adapters are exercised through this)."""

from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit

from gwstandin import StandIn, write_config

from llmgateway.engine import AttemptsFailed, Gateway
from llmgateway.records import RecordStore
from llmgateway.transport import DirectTransport, WireResponse
from llmgateway.types import Request

MSGS = ({"role": "user", "content": "hello"},)
SCHEMA_RF = {
    "type": "json_schema",
    "json_schema": {
        "name": "ans",
        "schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
    },
}
SCHEMA_JSON = '{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}'


class ForwardTransport:
    """Sends every step, hosted or not, to one stand-in (the real holder is D7/D8's)."""

    reaches_hosted = True

    def __init__(self, url):
        parts = urlsplit(url)
        self._host, self._port = parts.hostname, parts.port

    def exchange(self, request):
        conn = http.client.HTTPConnection(self._host, self._port, timeout=5)
        try:
            conn.request(
                request.method,
                request.path,
                body=request.body,
                headers={**request.headers, "Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            return WireResponse(resp.status, resp.read(), resp.getheader("Content-Type"))
        finally:
            conn.close()


def provider_table(kind, url, *, hosted=False, model="m1", extra=""):
    if hosted:
        return (
            f'kind = "{kind}"\nbase_url = "https://api.example.com"\nmodel = "{model}"\n'
            f'key_ref = "k1"\npinned_models = ["{model}", "other-pinned"]\n{extra}'
        )
    return f'kind = "{kind}"\nbase_url = "{url}"\nmodel = "{model}"\n{extra}'


class Env:
    """One provider `p` behind call site `cs`; `last` is the newest body the stand-in received."""

    def __init__(self, tmp_path, kind, replies=(), *, hosted=False, model="m1", extra="",
                 optional=()):  # fmt: skip
        self.standin = StandIn(*replies)
        cfg = f"""
[deployment]
local_endpoints = ["{self.standin.url}"]

[providers.p]
{provider_table(kind, self.standin.url, hosted=hosted, model=model, extra=extra)}
[scopes.plain]

[call_sites.cs]
component = "answer.cs"
scopes = ["plain"]
chain = ["p"]
optional_settings = {json.dumps(list(optional))}
"""
        self.src = write_config(tmp_path / "root", cfg)
        self.records = RecordStore(tmp_path / "state")
        transport = ForwardTransport(self.standin.url) if hosted else DirectTransport(self.src.get)
        self.gateway = Gateway(self.src, transport, self.records)
        self.kind = kind

    def call(self, **kw):
        return self.gateway.call(Request("cs", MSGS, **kw))

    def fail(self, **kw) -> AttemptsFailed:
        try:
            self.call(**kw)
        except AttemptsFailed as exc:
            return exc
        raise AssertionError("the call succeeded")

    @property
    def sent(self):
        return self.standin.received

    @property
    def last(self) -> bytes:
        return self.standin.received[-1]["body"]

    def close(self):
        self.standin.close()


def json_reply(obj, status=200):
    return (status, "application/json", json.dumps(obj).encode())
