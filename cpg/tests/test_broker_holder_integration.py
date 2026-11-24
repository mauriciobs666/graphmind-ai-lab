"""D9a — the broker forwards R7 allows to the REAL key holder (a subprocess on a real Unix socket) in front of a logging
HTTPS stand-in provider (egress-spec5 section 7 items 3 and 12, in-process broker: `Broker.handle` and a served
`BrokerServer` + `BrokerClient`; the real broker subprocess is D9b's, which owns `main`).

The fake key (`sk-ant-FAKE-...`) is planted in the holder's key file only; it must appear in none of the broker's
files, logs, argv, environment or records (a planted copy is found: the positive control).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading

import pytest

from broker_helpers import Harness, hosted_call, hosted_egress, load_system
from gateway_helpers import GATEWAY_TESTS  # noqa: F401  (puts llm-gateway/tests on sys.path for `holder_harness`)
from ccg.broker import client as client_mod
from ccg.broker.holderclient import HolderClient
from ccg.broker.server import BrokerServer
from holder_harness import (CALLER_SECRET, MODEL, UID, Env, HolderProc, OTHER_SECRET, StandIn, body_for, make_cert,
                            tree_search)

REFUSAL = "application/vnd.llmgateway.refusal+json"


@pytest.fixture(scope="module")
def cert(tmp_path_factory):
    return make_cert(tmp_path_factory.mktemp("tls"))


@pytest.fixture
def env(tmp_path, cert):
    standin = StandIn(*cert)
    e = Env(tmp_path / "root", standin, cert[0])
    yield e
    standin.close()


def stack(tmp_path, env: Env, *, endpoint=None, paths=("/v1/messages",), holder_table=None, extra_entries=()):
    """A running holder, a broker `Harness` whose only hosted entry is `endpoint`, wired through a real `HolderClient`."""
    if holder_table is not None:
        env.write_config(holder=env.holder_table(**holder_table))
    proc = HolderProc(env)
    ep = endpoint or env.endpoint
    eg = hosted_egress(ep, paths=list(paths))
    eg["hosted"] += [{"id": f"h{i}", "kind": "model", "endpoint": e, "paths": list(paths)}
                     for i, e in enumerate(extra_entries)]
    d = tmp_path / "broker"
    d.mkdir()
    hh = Harness(d, systems={"shop": load_system(d, repos=(("a", "hosted"),), eg=eg)},
                 hosted_exchanger=HolderClient(env.socket_path, UID, CALLER_SECRET, timeout=20))
    return proc, hh, ep


def call_hosted(hh: Harness, endpoint: str, *, run_id="eval-1", body=None, path="/v1/messages") -> dict:
    g = hh.open("answer", ["a"])
    req = hosted_call(endpoint=endpoint, codebases=["a"], target={"method": "POST", "path": path})
    if run_id is not None:
        req["gatewayRunId"] = run_id
    return hh.ask(g["token"], req, body if body is not None else json.loads(body_for()))


def refusal_code(r: dict) -> str:
    assert r["response"]["status"] == 499 and r["response"]["contentType"] == REFUSAL
    return json.loads(base64.b64decode(r["response"]["bodyB64"]))["holderRefusal"]


def test_a_200_settles_and_returns_the_provider_body_and_the_key_stays_in_the_holder(tmp_path, env):
    proc, hh, ep = stack(tmp_path, env)
    try:
        r = call_hosted(hh, ep)
    finally:
        proc.stop()
    assert r["decision"]["rule"] == "R7-HOSTED-SERVICE" and r["closeReason"] == "completed"
    assert r["response"]["status"] == 200
    assert json.loads(base64.b64decode(r["response"]["bodyB64"]))["usage"]["output_tokens"] == 7
    [seen] = env.standin.seen  # the holder reached the provider, with the key the broker never held
    assert seen.headers["x-api-key"] == env.marker and seen.path == "/v1/messages"
    [dec] = hh.records("decision")
    [close] = hh.records("close")
    assert dec["resolvedAddr"] is None and (close["httpStatus"], close["closeReason"]) == (200, "completed")
    assert close["bytesOut"] > len(body_for()) and close["bytesIn"] > 0
    # the marker is in no broker file, log, argv or environment; a planted copy is found (positive control)
    needle = env.marker.encode()
    roots = [str(tmp_path / "broker")]  # the audit dir is inside it
    assert tree_search(roots, needle) == []
    assert not any(env.marker in m for m in hh.logs) and not any(env.marker in a for a in sys.argv)
    assert not any(env.marker in v for v in os.environ.values())
    (hh.audit_dir / "planted.txt").write_text(env.marker)
    assert tree_search(roots, needle) == [str(hh.audit_dir / "planted.txt")]


REFUSALS = {  # code -> a ccg-shaped call that provokes it (cf. llm-gateway/tests/test_holder_core.py CASES)
    "run_id_missing": dict(run_id=None),
    "run_unmatched": dict(run_id="adhoc-1"),
    "model_not_pinned": dict(body=json.loads(body_for(model="claude-sonnet-4-5-latest"))),
    "limit_missing": dict(body={"model": MODEL, "messages": []}),
    "limit_over_ceiling": dict(body=json.loads(body_for(5000))),
    "body_invalid": dict(body=json.loads(body_for(100, tools=[{"name": "x"}]))),
}


@pytest.mark.parametrize("code", sorted(REFUSALS))
def test_a_holder_refusal_reaches_the_caller_intact_as_a_completed_499(tmp_path, env, code):
    proc, hh, ep = stack(tmp_path, env)
    try:
        r = call_hosted(hh, ep, **REFUSALS[code])
    finally:
        proc.stop()
    assert r["closeReason"] == "completed" and r["decision"]["decision"] == "allow"
    assert refusal_code(r) == code
    [dec] = hh.records("decision")
    [close] = hh.records("close")
    assert (close["httpStatus"], close["closeReason"], close["decisionAuditId"]) == (499, "completed", dec["auditId"])
    assert env.standin.seen == [] and not hh.audit.latched


def test_an_unregistered_endpoint_and_a_disallowed_path_are_refused_by_the_holder(tmp_path, env):
    proc, hh, ep = stack(tmp_path, env, paths=("/v1/messages", "/v1/complete"),
                         extra_entries=("https://127.0.0.1:9",))
    try:
        assert refusal_code(call_hosted(hh, "https://127.0.0.1:9")) == "endpoint_unknown"
        assert refusal_code(call_hosted(hh, ep, path="/v1/complete")) == "path_not_allowed"
    finally:
        proc.stop()
    assert env.standin.seen == [] and [c["httpStatus"] for c in hh.records("close")] == [499, 499]


def test_a_refusal_reaches_a_served_broker_client_and_the_stand_in_log_stays_empty(tmp_path, env):
    proc, hh, ep = stack(tmp_path, env)
    d = tmp_path / "sock"
    d.mkdir(mode=0o700)
    srv = BrokerServer(hh.broker, d / "b.sock", log=hh.logs.append)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = client_mod.BrokerClient(str(d / "b.sock"), timeout=30)
        grant = c.open_run("answer", "shop", ["a"])
        req = hosted_call(endpoint=ep, codebases=["a"], target={"method": "POST", "path": "/v1/messages"})
        decision, resp = c.request(grant.token, req, json.loads(body_for()))  # no gatewayRunId: run_id_missing
        decision2, ok = c.request(grant.token, {**req, "gatewayRunId": "eval-9"}, json.loads(body_for()))
    finally:
        srv.shutdown()
        proc.stop()
    assert decision.allowed and (resp.status, resp.content_type) == (499, REFUSAL)
    assert json.loads(resp.body) == {"holderRefusal": "run_id_missing"}
    assert decision2.allowed and ok.status == 200 and len(env.standin.seen) == 1


def test_a_missing_run_id_is_refused_under_refuse_and_served_in_the_shared_bucket_under_cap(tmp_path, env):
    proc, hh, ep = stack(tmp_path, env)
    try:
        assert refusal_code(call_hosted(hh, ep, run_id=None)) == "run_id_missing"
    finally:
        proc.stop()
    (tmp_path / "second").mkdir()
    proc2, hh2, ep = stack(tmp_path / "second", env, holder_table={"unmatched": "cap", "default_bound": 5000})
    try:
        r = call_hosted(hh2, ep, run_id=None)
    finally:
        proc2.stop()
    assert r["response"]["status"] == 200 and len(env.standin.seen) == 1


def test_a_hosted_entry_at_the_default_port_is_keyed_by_the_canonical_origin(tmp_path, env):
    """`base_url` is written without the port; the broker sends the canonical `https://127.0.0.1:443` and the holder must
    still find the provider (spec5 s3 item 2). Nothing listens on 443, so the exchange ends past the lookup, not in
    `endpoint_unknown`."""
    env.write_config(providers={"anth": env.provider(base_url="https://127.0.0.1")})
    proc, hh, ep = stack(tmp_path, env, endpoint="https://127.0.0.1:443")
    try:
        r = call_hosted(hh, ep)
    finally:
        proc.stop()
    assert r["decision"]["decision"] == "allow"
    # the holder found the provider (no `endpoint_unknown` 499), then could not connect to port 443: its own signed
    # `upstream-error` reply, which arrived (bytesIn > 0) - unlike a stopped holder, where nothing was exchanged
    assert r["decision"]["decision"] == "allow" and r["response"] is None and r["closeReason"] == "upstream-error"
    [close] = hh.records("close")
    assert close["bytesOut"] > 0 and close["bytesIn"] > 0 and close["httpStatus"] is None
    assert env.standin.seen == []


def test_a_stopped_holder_is_upstream_error_and_nothing_else_is(tmp_path, env):
    proc, hh, ep = stack(tmp_path, env)
    proc.stop()
    r = call_hosted(hh, ep)
    assert r["decision"]["decision"] == "allow" and r["response"] is None and r["closeReason"] == "upstream-error"
    [close] = hh.records("close")
    assert close["closeReason"] == "upstream-error" and close["bytesOut"] == 0 and not hh.audit.latched


def test_a_wrong_credential_is_the_unsigned_bad_credential_refusal_as_a_completed_499(tmp_path, env):
    proc, hh, ep = stack(tmp_path, env)
    hh.broker.hosted_exchanger = HolderClient(env.socket_path, UID, OTHER_SECRET, timeout=20)
    try:
        r = call_hosted(hh, ep)
    finally:
        proc.stop()
    assert r["closeReason"] == "completed" and refusal_code(r) == "bad_credential" and env.standin.seen == []


def test_an_unknown_caller_is_the_unsigned_unknown_caller_refusal_as_a_completed_499(tmp_path, env):
    (env.root / "state" / "registry" / "broker.secret").unlink()
    proc, hh, ep = stack(tmp_path, env)
    try:
        r = call_hosted(hh, ep)
    finally:
        proc.stop()
    assert r["closeReason"] == "completed" and refusal_code(r) == "unknown_caller" and env.standin.seen == []
