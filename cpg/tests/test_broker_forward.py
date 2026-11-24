"""D9a — an R7 allow is forwarded to the key holder, never sent by the broker (egress-spec5 sections 2-5, 7 items 1, 3-7).

Unit altitude: `Broker.handle` with the recording `hosted_exchanger` double of `broker_helpers.Harness`; the refusal
passthrough and the AC-21 checks also run against a real `HolderClient` and a scripted `AF_UNIX` listener that counts
connections. Expected values are literals written here.
"""
from __future__ import annotations

import base64
import hashlib
import os
import threading

import pytest

from broker_helpers import FakeHostedExchanger, Harness, hosted_call, load_system, model_call
from gateway_helpers import ScriptedHolder
from ccg.broker import client as client_mod
from ccg.broker.holderclient import HolderClient
from ccg.broker.server import BrokerServer
from llmgateway import framing as gw

REFUSAL = "application/vnd.llmgateway.refusal+json"
SECRET = bytes(range(32))
ENDPOINT = "https://api.hosted.example.com:443"


def sub(tmp_path, name):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return d


def all_hosted(tmp_path, **kw) -> Harness:
    return Harness(tmp_path, systems={"shop": load_system(tmp_path, repos=(("a", "hosted"), ("b", "hosted")))}, **kw)


def ask_hosted(hh: Harness, request=None, body=None, codebases=("a",)):
    g = hh.open("answer", list(codebases))
    return hh.ask(g["token"], request if request is not None else hosted_call(codebases=list(codebases)),
                  {"m": 1} if body is None else body)


# ------------------------------------------------------------------ item 1: the forward


def test_an_r7_allow_calls_the_hosted_exchanger_once_and_nothing_else(tmp_path):
    def boom(*a, **k):
        raise AssertionError("must not be called for R7")
    hh = all_hosted(tmp_path, iface=boom)
    hh.broker.tls_factory = boom
    r = ask_hosted(hh, hosted_call(codebases=["a"], gatewayRunId="eval-7"), {"m": 1})
    assert r["decision"]["rule"] == "R7-HOSTED-SERVICE" and r["closeReason"] == "completed"
    assert hh.hosted.calls == [{"endpoint": ENDPOINT, "method": "POST", "path": "/v1/chat/completions",
                                "body": b'{"m":1}', "gateway_run_id": "eval-7"}]
    assert hh.resolver.calls == [] and hh.exchange.calls == []
    dec, close = hh.last_decision(), hh.records("close")[-1]
    assert dec["resolvedAddr"] is None and dec["privateAddress"] is False and dec["requestMode"] == "hosted"
    assert dec["bodySha256"] == hashlib.sha256(hh.hosted.calls[0]["body"]).hexdigest() and dec["bytes"] == 7
    assert (close["closeReason"], close["httpStatus"], close["bytesOut"], close["bytesIn"]) == ("completed", 200, 300, 400)
    assert r["response"]["status"] == 200


def test_the_path_sent_is_the_entrys_string_and_a_private_flag_changes_nothing(tmp_path):
    from broker_helpers import hosted_egress
    eg = hosted_egress(privateAddress=True)
    hh = Harness(tmp_path, systems={"shop": load_system(tmp_path, repos=(("a", "hosted"),), eg=eg)})
    ask_hosted(hh)
    assert hh.hosted.calls[0]["path"] == "/v1/chat/completions"
    assert hh.last_decision()["privateAddress"] is False and hh.resolver.calls == []


def test_r1_and_r4_still_use_the_broker_exchanger_and_the_resolver(tmp_path):
    hh = all_hosted(tmp_path)
    g = hh.open("answer", ["a"])
    hh.ask(g["token"], model_call(codebases=["a"], endpoint="http://127.0.0.1:1234"), {"m": 1})
    idp = {"component": "access.idp", "purpose": "idp", "payloadClass": "repo-names-and-identity",
           "endpoint": "https://idp.example.com:443", "target": {"method": "POST", "path": "/who"}, "codebases": ["a"]}
    hh.ask(g["token"], idp, {"repoNames": ["a"], "identity": {"subject": "alice"}})
    assert [r["rule"] for r in hh.records("decision")] == ["R1-LOCAL-ENDPOINT", "R4-NAMED-SERVICE"]
    assert len(hh.exchange.calls) == 2 and hh.resolver.calls == ["idp.example.com."] and hh.hosted.calls == []


# ------------------------------------------------------------------ item 4: no holder configured


def test_without_a_holder_an_r7_allow_closes_upstream_error_and_connects_nowhere(tmp_path):
    hh = all_hosted(tmp_path, hosted_exchanger=None)
    r = ask_hosted(hh)
    assert r["decision"]["decision"] == "allow" and r["response"] is None and r["closeReason"] == "upstream-error"
    [close] = hh.records("close")
    assert (close["closeReason"], close["bytesOut"], close["bytesIn"], close["httpStatus"]) == (
        "upstream-error", 0, 0, None)
    assert hh.exchange.calls == [] and hh.resolver.calls == [] and not hh.audit.latched
    assert hh.last_decision()["rule"] == "R7-HOSTED-SERVICE" and hh.last_decision()["resolvedAddr"] is None


# ------------------------------------------------------------------ item 5: gatewayRunId

HASH = lambda raw: hashlib.sha256(raw).hexdigest()  # noqa: E731
BAD = [  # (raw value, the JSON text whose digest is recorded)
    (123, b"123"), ("", b'""'), ("a" * 129, b'"' + b"a" * 129 + b'"'), ("a b", b'"a b"'), ("a|b", b'"a|b"'),
    ("a\nb", b'"a\\nb"'), ("café", '"café"'.encode()), ("-a", b'"-a"'), (".a", b'".a"'), ("a/b", b'"a/b"'),
    (["x"], b'["x"]'), (True, b"true"), ({"a": 1}, b'{"a":1}'),
]


@pytest.mark.parametrize("value", ["eval-1", "A", "0", "run.1:a-b", "a_b", "a" * 128, "9" + "z" * 127])
def test_a_valid_gateway_run_id_reaches_the_exchanger_and_leaves_rejected_empty(tmp_path, value):
    hh = all_hosted(tmp_path)
    ask_hosted(hh, hosted_call(codebases=["a"], gatewayRunId=value))
    assert hh.hosted.calls[0]["gateway_run_id"] == value and hh.last_decision()["rejected"] == {}


@pytest.mark.parametrize("variant", ["missing", "null"])
def test_an_absent_gateway_run_id_is_none_with_no_rejected_entry(tmp_path, variant):
    hh = all_hosted(tmp_path)
    req = hosted_call(codebases=["a"])
    if variant == "null":
        req["gatewayRunId"] = None
    ask_hosted(hh, req)
    assert hh.hosted.calls[0]["gateway_run_id"] is None
    assert hh.last_decision()["rejected"] == {}


@pytest.mark.parametrize("raw,text", BAD, ids=[repr(b)[:20] for b, _ in BAD])
def test_an_invalid_gateway_run_id_is_none_and_its_digest_is_rejected(tmp_path, raw, text):
    hh = all_hosted(tmp_path)
    ask_hosted(hh, hosted_call(codebases=["a"], gatewayRunId=raw))
    assert hh.hosted.calls[0]["gateway_run_id"] is None
    assert hh.last_decision()["rejected"] == {"gatewayRunId": HASH(text)}
    assert hh.last_decision()["decision"] == "allow" and not hh.audit.latched  # never a deny reason of its own


def test_a_lone_surrogate_gateway_run_id_is_rejected_by_digest_and_never_latches(tmp_path):
    hh = all_hosted(tmp_path)
    ask_hosted(hh, hosted_call(codebases=["a"], gatewayRunId="\ud800"))
    assert hh.hosted.calls[0]["gateway_run_id"] is None
    assert list(hh.last_decision()["rejected"]) == ["gatewayRunId"] and not hh.audit.latched


def test_gateway_run_id_is_not_in_any_audit_record(tmp_path):
    hh = all_hosted(tmp_path)
    ask_hosted(hh, hosted_call(codebases=["a"], gatewayRunId="secret-run-77"))
    assert b"secret-run-77" not in (hh.audit_dir / "egress.jsonl").read_bytes()


def test_an_r1_call_ignores_gateway_run_id_and_a_denied_request_never_calls_the_exchanger(tmp_path):
    hh = all_hosted(tmp_path)
    g = hh.open("answer", ["a"])
    hh.ask(g["token"], model_call(codebases=["a"], gatewayRunId="eval-1"), {"m": 1})
    assert [r["rule"] for r in hh.records("decision")] == ["R1-LOCAL-ENDPOINT"] and hh.records("decision")[0]["rejected"] == {}
    assert hh.hosted.calls == []
    bad = hosted_call(codebases=["nosuch"], gatewayRunId="eval-1")  # codebases-not-subset
    assert hh.ask(g["token"], bad, {"m": 1})["decision"]["decision"] == "deny" and hh.hosted.calls == []


# ------------------------------------------------------------------ item 3: the refusal passthrough


def refusal_holder(tmp_path, code="run_unmatched"):
    def script(payload):
        hdr, _ = gw.parse_request(payload)
        return gw.build_response(SECRET, request_nonce=hdr.nonce, close_reason="completed", status=499,
                                 content_type=REFUSAL, body=gw.refusal_body(code))
    return ScriptedHolder(tmp_path / "run", script)


def _assert_one_refusal(hh: Harness):
    [dec] = hh.records("decision")
    [close] = hh.records("close")
    assert (dec["decision"], dec["rule"]) == ("allow", "R7-HOSTED-SERVICE")
    assert (close["httpStatus"], close["closeReason"], close["decisionAuditId"]) == (499, "completed", dec["auditId"])
    assert not hh.audit.latched


def test_a_holder_refusal_reaches_the_caller_as_a_completed_response(tmp_path):
    h = refusal_holder(tmp_path)
    hh = all_hosted(sub(tmp_path, "b"), hosted_exchanger=HolderClient(h.path, os.getuid(), SECRET))
    try:
        r = ask_hosted(hh, hosted_call(codebases=["a"], gatewayRunId="zzz"))
    finally:
        h.close()
    assert r["closeReason"] == "completed" and r["decision"]["decision"] == "allow"
    assert r["response"]["status"] == 499 and r["response"]["contentType"] == REFUSAL
    assert base64.b64decode(r["response"]["bodyB64"]) == b'{"holderRefusal": "run_unmatched"}'
    _assert_one_refusal(hh)


def test_a_holder_refusal_reaches_a_served_broker_client_intact(tmp_path):
    h = refusal_holder(tmp_path)
    hh = all_hosted(sub(tmp_path, "b"), hosted_exchanger=HolderClient(h.path, os.getuid(), SECRET))
    d = tmp_path / "sock"
    d.mkdir(mode=0o700)
    srv = BrokerServer(hh.broker, d / "b.sock", log=hh.logs.append)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = client_mod.BrokerClient(str(d / "b.sock"), timeout=10)
        grant = c.open_run("answer", "shop", ["a"])
        decision, resp = c.request(grant.token, hosted_call(codebases=["a"], gatewayRunId="zzz"), {"m": 1})
    finally:
        srv.shutdown()
        h.close()
    assert decision.allowed and decision.rule == "R7-HOSTED-SERVICE"
    assert resp is not None and (resp.status, resp.content_type) == (499, REFUSAL)
    assert resp.body == b'{"holderRefusal": "run_unmatched"}'
    _assert_one_refusal(hh)


@pytest.mark.parametrize("status", [499, 404, 500, 503, 429])
def test_any_completed_exchange_keeps_its_response_whatever_the_status(tmp_path, status):
    from ccg.broker.structured import Exchange
    ex = FakeHostedExchanger(Exchange("completed", status, "text/plain", b"nope", 10, 20))
    hh = all_hosted(tmp_path, hosted_exchanger=ex)
    r = ask_hosted(hh)
    assert r["response"] == {"status": status, "contentType": "text/plain", "bodyB64": "bm9wZQ=="}


# ------------------------------------------------------------------ item 6: R5/R6 refuse independently of the gateway


GATEWAY_FIELDS = {"marker": "unrestricted", "scope": "all", "unrestricted": True, "gatewayRunId": "eval-1"}


def counting_holder(tmp_path) -> ScriptedHolder:
    return ScriptedHolder(tmp_path / "run", lambda payload: None)


def _wired(tmp_path, h: ScriptedHolder) -> Harness:
    client, calls = HolderClient(h.path, os.getuid(), SECRET, timeout=1), []

    def counting(*args):
        calls.append(args)
        return client(*args)

    hh = Harness(sub(tmp_path, "b"), hosted_exchanger=counting)
    hh.forwarded = calls
    return hh


def _pr_to_hosted(**over):
    r = {"component": "answer.pr-adapter", "purpose": "pr-addon", "payloadClass": "repo-derived",
         "endpoint": ENDPOINT, "target": {"method": "POST", "path": "/v1/chat/completions"}, "codebases": ["front"]}
    r.update(over)
    return r


@pytest.mark.parametrize("run_scope", [["docs"], ["front"], ["front", "docs", "lib"]])
@pytest.mark.parametrize("extras", [{}, GATEWAY_FIELDS])
def test_r5_refuses_a_hosted_model_call_in_a_local_only_system_whatever_the_request_carries(tmp_path, run_scope, extras):
    h = counting_holder(tmp_path)
    try:
        hh = _wired(tmp_path, h)
        g = hh.open("answer", run_scope)
        r = hh.ask(g["token"], hosted_call(codebases=list(run_scope), **extras), {"m": 1})
    finally:
        h.close()
    assert r["decision"] == {**r["decision"], "decision": "deny", "rule": "R5-SYSTEM-LOCAL-MODEL",
                             "reason": "system-local-only-model"} and r["response"] is None
    assert hh.records("close") == [] and hh.forwarded == [] and h.connections == 0
    assert hh.last_decision()["decision"] == "deny"


@pytest.mark.parametrize("component,purpose", [("answer.pr-adapter", "pr-addon")])
@pytest.mark.parametrize("extras", [{}, GATEWAY_FIELDS])
def test_r6_refuses_a_non_model_call_to_the_hosted_endpoint_in_a_local_only_run(tmp_path, component, purpose, extras):
    h = counting_holder(tmp_path)
    try:
        hh = _wired(tmp_path, h)
        g = hh.open("answer", ["front"])
        r = hh.ask(g["token"], _pr_to_hosted(component=component, purpose=purpose, **extras), {"m": 1})
    finally:
        h.close()
    assert (r["decision"]["decision"], r["decision"]["rule"], r["decision"]["reason"]) == (
        "deny", "R6-STICKY-LOCAL", "sticky-local-only")
    assert hh.records("close") == [] and hh.forwarded == [] and h.connections == 0
    assert hh.last_decision()["requestMode"] == "local-only"


def test_r5_and_r6_do_not_depend_on_having_a_holder_at_all(tmp_path):
    hh = Harness(tmp_path, hosted_exchanger=None)
    g = hh.open("answer", ["docs"])
    assert hh.ask(g["token"], hosted_call(codebases=["docs"]), {"m": 1})["decision"]["rule"] == "R5-SYSTEM-LOCAL-MODEL"
    g = hh.open("answer", ["front"])
    assert hh.ask(g["token"], _pr_to_hosted(), {"m": 1})["decision"]["rule"] == "R6-STICKY-LOCAL"


# ------------------------------------------------------------------ item 7: unknown request keys are inert

VOLATILE = {"auditId", "ts", "decisionAuditId"}


def _stable(records):
    return [{k: v for k, v in r.items() if k not in VOLATILE and k != "runId"} for r in records]


@pytest.mark.parametrize("build", [lambda: hosted_call(codebases=["a"]), lambda: hosted_call(codebases=["nosuch"]),
                                   lambda: model_call(codebases=["a"], endpoint="http://127.0.0.1:1234")])
def test_an_unknown_request_key_changes_no_verdict_and_no_record_field(tmp_path, build):
    plain, extra = all_hosted(sub(tmp_path, "p")), all_hosted(sub(tmp_path, "e"))
    for hh, add in ((plain, {}), (extra, {"marker": "unrestricted", "scope": "x", "unrestricted": True, "other": [1]})):
        g = hh.open("answer", ["a"])
        hh.ask(g["token"], {**build(), **add}, {"m": 1})
    assert _stable(plain.records("decision")) == _stable(extra.records("decision"))
    assert _stable(plain.records("close")) == _stable(extra.records("close"))
    assert plain.hosted.calls == extra.hosted.calls
