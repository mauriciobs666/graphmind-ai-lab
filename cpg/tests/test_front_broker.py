"""P1-26 dispatch 2 (6): the front's two broker calls against the real broker policy and the real ``cpg/ccg/system.yaml`` egress section (``-egress-spec3.md`` §2.1, §2.4a, §10.2).

``Broker`` runs in process over the shipped manifest with an exchange stub standing in for LM Studio; a bridge gives ``ModelRun`` the three ``BrokerClient`` methods over
``Broker.handle`` (the same messages the socket carries). The test shows that the ``local`` entry's ``paths`` are what the client calls (the model list and the chat
completion), that a local-only repository's question is allowed by ``R1``, that the answer run's decisions are audited, and that a path the entry does not list is denied.

    cd cpg && .venv/bin/python -m pytest tests/test_front_broker.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from broker_helpers import Harness
from front_helpers import FrontReader
from front_model_helpers import MODEL, draft_reply
from test_front_documents import structured, world

from ccg import manifest as mf
from ccg.broker import policy, structured as broker_structured
from ccg.broker.client import BrokerError, Decision, Response, RunGrant
from ccg.front import Front
from ccg.front import interpret as I

SHIPPED = Path(__file__).resolve().parent.parent / "ccg" / "system.yaml"


class Bridge:
    """``BrokerClient``'s three methods over ``Broker.handle``."""

    def __init__(self, h):
        self.h = h

    def _call(self, msg):
        r = self.h.broker.handle(msg)
        if r.get("ok") is not True:
            raise BrokerError(str(r.get("error")), str(r.get("message", "")))
        return r

    def open_run(self, run_kind, system_id, codebases, override=None):
        g = self._call({"op": "open_run", "runKind": run_kind, "systemId": system_id, "codebases": codebases, "override": override})["grant"]
        return RunGrant(g["runId"], g["token"], g["effective"], g["systemLocalOnly"], g["auditId"])

    def close_run(self, token):
        self._call({"op": "close_run", "token": token})

    def request(self, token, request, body=None):
        r = self._call({"op": "request", "token": token, "request": request, "body": body})
        d, resp = r["decision"], r.get("response")
        return (Decision(d["decision"], d["rule"], d["reason"], d["auditId"]),
                Response(resp["status"], resp["contentType"], base64.b64decode(resp["bodyB64"])) if resp else None)


def lm_studio(replies):
    sent = []

    def exchange(scheme, host, port, addr, method, path, body, **kw):
        sent.append((method, path, json.loads(body) if body else None))
        payload = {"data": [MODEL]} if path == "/api/v0/models" else {"choices": [{"message": {"content": json.dumps(replies.pop(0))}}]}
        return broker_structured.Exchange("completed", 200, "application/json", json.dumps(payload).encode(), 10, 10)
    return exchange, sent


def shipped_harness(tmp_path, replies):
    exchange, sent = lm_studio(replies)
    m = mf.load_manifest(SHIPPED)
    h = Harness(tmp_path, systems={m.system: policy.LoadedSystem.from_manifest(m)}, exchanger=exchange)
    return h, sent, m


def test_the_shipped_local_entry_lists_exactly_the_two_paths_the_client_calls():
    m = mf.load_manifest(SHIPPED)
    (entry,) = m.egress["local"]
    assert sorted(entry["paths"]) == sorted([I.MODELS_PATH, I.CHAT_PATH]) == ["/api/v0/models", "/v1/chat/completions"]
    assert entry["endpoint"] == I.LOCAL_ENDPOINT


def test_the_front_over_the_real_broker_policy_is_allowed_by_the_local_entry_and_audited(tmp_path):
    g, _ = world()
    h, sent, m = shipped_harness(tmp_path, [draft_reply()])
    repos = tuple(r.repo_id for r in m.repos)
    f = Front((g.repo,), lambda a, d: structured(g), interpreter=I.Interpreter(), run_factory=I.run_factory(m.system, repos, Bridge(h)))
    doc = f("if I touch m.py::A who gets hit", reader=FrontReader(g))
    assert doc["interpretation"]["status"] == "resolved"
    assert [(s[0], s[1]) for s in sent] == [("GET", "/api/v0/models"), ("POST", "/v1/chat/completions")]
    decisions = h.records("decision")
    assert [(d["decision"], d["rule"], d["component"], d["purpose"]) for d in decisions] == [("allow", "R1-LOCAL-ENDPOINT", "answer.interpret", "model-call")] * 2
    assert len(h.records("policy")) == 1                                      # one run for the question


def test_a_path_the_entry_does_not_list_is_denied_by_the_policy(tmp_path):
    h, _, m = shipped_harness(tmp_path, [])
    grant = Bridge(h).open_run("answer", m.system, "*")
    req = {"component": "answer.interpret", "purpose": "model-call", "payloadClass": "none", "endpoint": I.LOCAL_ENDPOINT,
           "target": {"method": "GET", "path": "/api/v0/embeddings"}, "codebases": [r.repo_id for r in m.repos]}
    decision, resp = Bridge(h).request(grant.token, req, None)
    assert not decision.allowed and decision.reason == "path-not-allowed" and resp is None
