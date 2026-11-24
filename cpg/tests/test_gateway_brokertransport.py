"""D10 — `ccg.gateway.brokertransport.BrokerTransport` through the real `llmgateway.Gateway`, a real `Broker` served on a
real socket (`BrokerServer` + `BrokerClient`) and a `ScriptedHolder` double for the hosted forward (docs/plans/llm-gateway.md
§3.2, §3.3, §3.6; egress-spec5 §3; FR-3, FR-11, FR-12, AC-2, AC-21). Local calls end in a `FakeExchange` (no provider is
contacted); the gateway's installed manifest and the broker's manifest are separate files, as in production."""
from __future__ import annotations

import json
import os
import threading

import pytest

from broker_helpers import FakeExchange, Harness, hosted_egress, load_system, write_manifest
from gateway_helpers import ScriptedHolder, load_vectors, secret_of
from gwstandin import write_config
from ccg.broker import client as client_mod
from ccg.broker.holderclient import HolderClient
from ccg.broker.server import BrokerServer
from ccg.broker.structured import Exchange
from ccg.gateway.brokertransport import BrokerTransport
from ccg.gateway.manifestmarkers import ManifestMarkers
from llmgateway import framing as gw
from llmgateway.engine import AttemptsFailed, Gateway
from llmgateway.policy import PolicyRefused
from llmgateway.records import RecordStore
from llmgateway.types import FailureClass as F
from llmgateway.types import Request

UID = os.getuid()
REFUSAL = "application/vnd.llmgateway.refusal+json"
HOSTED = "https://api.hosted.example.com:443"
MSGS = ({"role": "user", "content": "hi"},)
LM_OK = json.dumps({"model": "m", "choices": [{"message": {"role": "assistant", "content": "local"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2}}).encode()
CLAUDE_OK = json.dumps({"id": "msg_1", "model": "claude-x", "content": [{"type": "text", "text": "hosted"}],
                        "usage": {"input_tokens": 11, "output_tokens": 7}}).encode()

TOML = f"""
[deployment]
local_endpoints = ["http://127.0.0.1:1234"]

[providers.lm]
kind = "lmstudio"
base_url = "http://127.0.0.1:1234"
model = "m"

[providers.claude]
kind = "anthropic"
base_url = "{HOSTED}"
key_ref = "/var/lib/llmholder/keys/anthropic.key"
model = "claude-x"
pinned_models = ["claude-x"]
max_tokens_ceiling = 4096

[scope_sources.ccg-manifest]
prefix = "ccg:"

[call_sites.local]
component = "answer.interpret"
scopes = ["ccg:system:shop"]
chain = ["lm"]
retry = {{ max = 1, on = ["connection", "timeout", "outage"] }}
settings = {{ max_tokens = 64 }}

[call_sites.hosted]
component = "answer.interpret"
scopes = ["ccg:system:shop"]
chain = ["claude"]
retry = {{ max = 1, on = ["connection", "timeout", "outage"] }}
settings = {{ max_tokens = 64 }}

[call_sites.hosted_or_local]
component = "answer.summarize"
scopes = ["ccg:system:shop"]
chain = ["claude"]
local_chain = ["lm"]
settings = {{ max_tokens = 64 }}
"""


class Stack:
    """Gateway -> BrokerTransport -> BrokerClient -> BrokerServer -> Broker (R1 FakeExchange, R7 ScriptedHolder)."""

    def __init__(self, tmp_path, *, broker_repos, gateway_repos=None, reply=None):
        self.tmp = tmp_path
        vec = load_vectors()
        self.secret = secret_of(vec)
        self.reply = reply or (lambda hdr: dict(status=200, ctype="application/json", body=CLAUDE_OK))

        def script(payload):
            hdr, body = gw.parse_request(payload)
            self.holder_seen.append((hdr, body))
            r = self.reply(hdr)
            return gw.build_response(self.secret, request_nonce=hdr.nonce, close_reason=r.get("reason", "completed"),
                                     status=r.get("status", 200), content_type=r.get("ctype", "application/json"),
                                     body=r.get("body", b"{}"))

        self.holder_seen: list = []
        self.holder = ScriptedHolder(tmp_path / "run", script)
        eg = hosted_egress(HOSTED, paths=["/v1/messages"])
        eg["local"][0]["paths"] = ["/v1/chat/completions"]
        d = tmp_path / "broker"
        d.mkdir()
        self.exchange = FakeExchange(Exchange("completed", 200, "application/json", LM_OK, 10, 20))
        self.hh = Harness(d, systems={"shop": load_system(d, repos=broker_repos, eg=eg)}, exchanger=self.exchange,
                          hosted_exchanger=HolderClient(self.holder.path, UID, self.secret, timeout=10))
        self.srv = BrokerServer(self.hh.broker, d / "b.sock", log=self.hh.logs.append)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.client = client_mod.BrokerClient(str(d / "b.sock"), timeout=30)
        self.grant = self.client.open_run("answer", "shop", "*")
        # the gateway's own installed copy of the manifest, under an anchored root
        self.root = tmp_path / "root"
        (self.root / "etc" / "llmgateway" / "ccg").mkdir(parents=True)
        for p in (self.root, self.root / "etc", self.root / "etc" / "llmgateway", self.root / "etc" / "llmgateway" / "ccg"):
            p.chmod(0o755)
        self.installed = self.root / "etc" / "llmgateway" / "ccg" / "system.yaml"
        self.install(gateway_repos or broker_repos)
        self.markers = ManifestMarkers(self.installed, {UID}, anchor=self.root)
        self.records = RecordStore(tmp_path / "state")
        self.gateway = Gateway(write_config(self.root, TOML), BrokerTransport(self.client, lambda: self.grant.token),
                               self.records, marker_source=self.markers)

    def install(self, repos):
        p = write_manifest(self.installed.parent, repos=repos)
        os.replace(p, self.installed)
        self.installed.chmod(0o644)

    def call(self, site="local", *repos, **kw):
        return self.gateway.call(Request(site, MSGS, scopes=tuple(f"ccg:repo:{r}" for r in repos), **kw))

    def attempts(self):
        return self.records.attempts()

    def close(self):
        self.srv.sock.close()
        self.holder.close()


HOSTED_REPOS = (("docs", "hosted"), ("lib", "hosted"))
MIXED_REPOS = (("front", "local-only"), ("docs", "hosted"))


@pytest.fixture
def stack(tmp_path):
    made = []

    def make(**kw):
        s = Stack(tmp_path, **kw)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def test_a_local_call_goes_through_the_broker_to_the_local_model(stack):
    s = stack(broker_repos=HOSTED_REPOS)
    r = s.call("local", "docs", "lib", run_id="eval-1")
    assert (r.provider, r.hosted, r.message["content"]) == ("lm", False, "local")
    [call] = s.exchange.calls
    assert (call["method"], call["path"], call["port"]) == ("POST", "/v1/chat/completions", 1234)
    assert json.loads(call["body"])["model"] == "m"
    [d] = [x for x in s.hh.records("decision")]
    assert (d["component"], d["decision"], d["rule"]) == ("answer.interpret", "allow", d["rule"]) and d["rule"].startswith("R1")
    assert s.holder.connections == 0
    [row] = s.attempts()
    assert (row["outcome"], row["hosted"], row["component"]) == ("ok", 0, "answer.interpret")


def test_the_component_comes_from_the_call_site(stack):
    s = stack(broker_repos=HOSTED_REPOS)
    s.call("hosted_or_local", "docs")  # hosted chain over unrestricted repos
    [d] = s.hh.records("decision")
    assert d["component"] == "answer.summarize"


def test_a_hosted_call_is_forwarded_to_the_holder_with_the_run_id_and_returns_its_reply(stack):
    s = stack(broker_repos=HOSTED_REPOS)
    r = s.call("hosted", "docs", "lib", run_id="eval-7")
    assert (r.provider, r.hosted, r.model, r.message["content"]) == ("claude", True, "claude-x", "hosted")
    assert (r.usage.input_tokens, r.usage.output_tokens) == (11, 7)
    [(hdr, body)] = s.holder_seen
    assert (hdr.caller, hdr.endpoint, hdr.path, hdr.run_id) == ("broker", HOSTED, "/v1/messages", "eval-7")
    assert json.loads(body)["model"] == "claude-x"
    assert s.exchange.calls == []  # the broker sent nothing itself
    assert s.hh.last_decision()["rule"] == "R7-HOSTED-SERVICE"
    [row] = s.attempts()
    assert (row["outcome"], row["hosted"]) == ("ok", 1)


def test_a_hosted_call_without_a_run_id_reaches_the_holder_with_none(stack):
    s = stack(broker_repos=HOSTED_REPOS)
    s.call("hosted", "docs")
    assert s.holder_seen[0][0].run_id == ""


@pytest.mark.parametrize("code,cls", [
    ("bad_credential", F.CONFIG), ("stale_timestamp", F.CONFIG), ("replayed_nonce", F.CONFIG),
    ("unknown_caller", F.CONFIG), ("endpoint_unknown", F.POLICY_REFUSED), ("path_not_allowed", F.POLICY_REFUSED),
    ("https_required", F.POLICY_REFUSED), ("address_forbidden", F.POLICY_REFUSED), ("model_not_pinned", F.REJECTED),
    ("limit_missing", F.REJECTED), ("limit_over_ceiling", F.REJECTED), ("body_invalid", F.REJECTED),
    ("run_id_missing", F.RUN_BOUND), ("run_unmatched", F.RUN_BOUND), ("run_bound_reached", F.RUN_BOUND),
    ("window_ceiling_reached", F.RUN_BOUND), ("release_cap_reached", F.RUN_BOUND), ("key_missing", F.CONFIG),
])
def test_every_holder_refusal_fails_with_its_class_after_one_attempt_and_is_never_retried(stack, code, cls):
    body = json.dumps({"holderRefusal": code}).encode()
    s = stack(broker_repos=HOSTED_REPOS, reply=lambda hdr: dict(status=499, ctype=REFUSAL, body=body))
    with pytest.raises(AttemptsFailed) as e:
        s.call("hosted", "docs", run_id="eval-1")
    assert e.value.classes == (cls,) and e.value.code == code
    assert len(s.holder_seen) == 1 and len(s.attempts()) == 1
    closes = s.hh.records("close")
    assert [(c["httpStatus"], c["closeReason"]) for c in closes] == [(499, "completed")]


def test_a_provider_5xx_through_the_holder_is_outage_and_not_retried_unless_listed(stack):
    s = stack(broker_repos=HOSTED_REPOS, reply=lambda hdr: dict(status=503, body=b'{"error":{"type":"overloaded_error"}}'))
    with pytest.raises(AttemptsFailed) as e:
        s.call("hosted", "docs", run_id="eval-1")
    assert e.value.classes == (F.OUTAGE, F.OUTAGE)  # the site retries outage once
    assert len(s.holder_seen) == 2 and [r["kind"] for r in s.attempts()] == ["first", "retry"]


@pytest.mark.parametrize("reason,cls", [("upstream-error", F.CONNECTION), ("response-too-large", F.CONNECTION),
                                         ("timeout", F.TIMEOUT)])
def test_a_broker_close_without_a_response_maps_to_its_class_and_is_retried_once(stack, reason, cls):
    s = stack(broker_repos=HOSTED_REPOS)
    s.exchange.result = Exchange(reason, None, None, None, 0, 0)
    with pytest.raises(AttemptsFailed) as e:
        s.call("local", "docs")
    assert e.value.classes == (cls, cls)  # the site retries connection/timeout once
    assert [r["kind"] for r in s.attempts()] == ["first", "retry"]
    assert len(s.exchange.calls) == 2


def test_a_broker_denial_is_policy_refused_and_never_retried_and_the_gateways_marker_cannot_loosen_it(stack):
    # the gateway's installed manifest says hosted; the broker's manifest says front is local-only (AC-21)
    s = stack(broker_repos=MIXED_REPOS, gateway_repos=(("front", "hosted"), ("docs", "hosted")))
    with pytest.raises(AttemptsFailed) as e:
        s.call("hosted", "front", "docs", run_id="eval-1")
    assert e.value.classes == (F.POLICY_REFUSED,)
    assert len(s.attempts()) == 1
    assert s.holder.connections == 0 and s.exchange.calls == []
    assert s.hh.last_decision()["decision"] == "deny"
    assert s.hh.records("close") == []


def test_a_restricted_scope_never_reaches_the_broker_with_a_hosted_chain(stack):
    s = stack(broker_repos=MIXED_REPOS)
    with pytest.raises(PolicyRefused) as e:
        s.call("hosted", "front")
    assert e.value.reason == "restricted_hosted"
    assert s.hh.records("decision") == [] and s.holder.connections == 0


def test_a_restricted_scope_runs_the_call_sites_local_chain_through_the_broker(stack):
    s = stack(broker_repos=MIXED_REPOS)
    r = s.call("hosted_or_local", "front")
    assert (r.provider, r.hosted) == ("lm", False)
    assert s.holder.connections == 0 and len(s.exchange.calls) == 1


def test_the_marker_is_read_on_every_call_so_a_tightened_manifest_applies_to_the_next_call(stack):
    s = stack(broker_repos=HOSTED_REPOS)
    assert s.call("hosted_or_local", "docs").hosted is True
    s.install((("docs", "local-only"), ("lib", "hosted")))  # tightened; the broker still has its start-time copy (D-4)
    r = s.call("hosted_or_local", "docs")
    assert (r.provider, r.hosted) == ("lm", False)
    assert len(s.holder_seen) == 1


def test_an_unreadable_installed_manifest_refuses_a_hosted_call_before_the_broker(stack):
    s = stack(broker_repos=HOSTED_REPOS)
    s.installed.write_text("repos: [")
    with pytest.raises(PolicyRefused) as e:
        s.call("hosted", "docs")
    assert e.value.reason == "marker_unreadable"
    assert s.hh.records("decision") == [] and s.holder.connections == 0


def test_an_unreachable_broker_is_config_and_never_retried(stack):
    s = stack(broker_repos=HOSTED_REPOS)
    s.srv.sock.close()
    os.unlink(s.srv.sock_path)
    with pytest.raises(AttemptsFailed) as e:
        s.call("local", "docs")
    assert e.value.classes == (F.CONFIG,) and len(s.attempts()) == 1


class StubClient:
    """Records the one message `BrokerTransport` sends; answers an allow with a canned completed response."""

    def __init__(self):
        self.sent: list[dict] = []

    def _call(self, msg):
        self.sent.append(msg)
        return {"ok": True, "decision": {"decision": "allow", "rule": "R1", "reason": None, "auditId": "a"},
                "response": {"status": 200, "contentType": "application/json", "bodyB64": "e30="},
                "closeReason": "completed"}


def wire(**over):
    from llmgateway.transport import WireRequest
    base = dict(provider_id="lm", kind="lmstudio", base_url="http://127.0.0.1:1234", path="/v1/chat/completions",
                body=b'{"a":1}', hosted=False, component="answer.interpret",
                scopes=("ccg:system:shop", "ccg:repo:docs", "ccg:repo:lib", "other:scope"), run_id="eval-2")
    base.update(over)
    return WireRequest(**base)


def test_the_broker_message_carries_component_purpose_endpoint_target_codebases_and_run_id():
    c = StubClient()
    resp = BrokerTransport(c, lambda: "tok").exchange(wire())
    assert (resp.status, resp.body, resp.content_type) == (200, b"{}", "application/json")
    [msg] = c.sent
    assert msg == {"op": "request", "token": "tok", "body": {"a": 1}, "request": {
        "component": "answer.interpret", "purpose": "model-call", "payloadClass": "repo-derived",
        "endpoint": "http://127.0.0.1:1234", "target": {"method": "POST", "path": "/v1/chat/completions"},
        "codebases": ["docs", "lib"], "gatewayRunId": "eval-2"}}


def test_a_bodyless_get_is_payload_class_none_and_a_missing_run_id_is_not_sent():
    c = StubClient()
    BrokerTransport(c, lambda: "t").exchange(wire(method="GET", body=b"", run_id=None, path="/api/v0/models"))
    req = c.sent[0]["request"]
    assert (c.sent[0]["body"], req["payloadClass"], req["target"]["method"]) == (None, "none", "GET")
    assert "gatewayRunId" not in req


def test_a_non_json_body_is_a_config_failure_before_anything_is_sent():
    from llmgateway.types import CallFailed
    c = StubClient()
    with pytest.raises(CallFailed) as e:
        BrokerTransport(c, lambda: "t").exchange(wire(body=b"not json"))
    assert e.value.failure_class is F.CONFIG and c.sent == []


def test_it_reaches_hosted_only_through_the_broker():
    assert BrokerTransport.reaches_hosted is True
