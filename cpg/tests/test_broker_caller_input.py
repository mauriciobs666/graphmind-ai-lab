"""P1-29 — caller-controlled strings never latch the broker or crash the audit path (review F1, M1).

Spec: `docs/plans/connected-code-graph-egress-spec3.md` §5.1 ("a caller-supplied value that does not fit its field is
recorded as `null` ... so caller input can never make a record invalid") and §10.1 test (14). Every string slot a
caller controls is crossed with hostile value classes. Per case the broker must survive: not latched, a following
legitimate `open_run` still works, every record on disk validates (or none was written), and no upstream
connection is made unless the request is a legitimate allow. The slot list is derived from `policy.sanitize`'s own
output keys plus the message shapes, so a new sanitised field fails `test_the_sweep_covers_every_sanitised_field`
until it joins the sweep.
"""
from __future__ import annotations

import json

import pytest

from broker_helpers import Harness, model_call
from ccg.broker import framing, policy

HOSTILE = {
    "lone-surrogate": "\ud800",
    "surrogate-in-text": "/a\udfffb",
    "65-chars": "a" * 65,
    "257-chars": "/" + "a" * 256,
    "301-chars": "http://" + "a" * 294,
    "10k-chars": "x" * 10_000,
    "nul": "\x00a",
    "control": "a\x1fb\x7f",
    "u2028": "a b",
    "newline": "a\nb",
    "non-ascii": "café",
    "int": 123,
    "nested": {"a": [1, {"b": "\ud800"}]},
    "list": ["\ud800", "x" * 70],
    "none": None,
}

# every key `sanitize` returns, each exercised below through the slot named here
SANITISED_SLOTS = {"component", "purpose", "payloadClass", "method", "path", "endpoint", "codebases"}


def _request(slot: str, v):
    r = model_call()
    if slot in ("component", "purpose", "payloadClass", "endpoint"):
        r[slot] = v
    elif slot == "method":
        r["target"] = {"method": v, "path": "/v1/chat/completions"}
    elif slot == "path":
        r["target"] = {"method": "POST", "path": v}
    elif slot == "codebases":
        r["codebases"] = [v]
    elif slot == "codebases-list":
        r["codebases"] = v
    elif slot == "target":
        r["target"] = v
    return r


def _messages(slot: str, v, token: str):
    """The messages a hostile value in ``slot`` produces."""
    if slot == "token":
        return [{"op": "request", "token": v, "request": model_call(), "body": {"m": 1}},
                {"op": "close_run", "token": v}]
    if slot == "request":
        return [{"op": "request", "token": token, "request": v, "body": {"m": 1}}]
    if slot == "body-key":
        return [{"op": "request", "token": token, "request": model_call(), "body": {v if isinstance(v, str) else "k": 1}}]
    if slot == "body-value":
        return [{"op": "request", "token": token, "request": model_call(), "body": {"k": v}}]
    if slot == "body-whole":
        return [{"op": "request", "token": token, "request": model_call(), "body": v}]
    if slot in ("identity-subject", "identity-issuer", "identity-key", "repo-name"):
        idp = {"component": "access.idp", "purpose": "idp", "payloadClass": "repo-names-and-identity",
               "endpoint": "https://idp.example.com:443", "target": {"method": "POST", "path": "/who"},
               "codebases": ["front"]}
        ident = {"subject": "alice"}
        names = ["front"]
        if slot == "identity-subject":
            ident = {"subject": v}
        elif slot == "identity-issuer":
            ident = {"subject": "a", "issuer": v}
        elif slot == "identity-key":
            ident = {"subject": "a", (v if isinstance(v, str) else "k"): 1}
        else:
            names = [v]
        return [{"op": "request", "token": token, "request": idp, "body": {"repoNames": names, "identity": ident}}]
    if slot in ("systemId", "runKind", "override", "open-codebases"):
        m = {"op": "open_run", "runKind": "answer", "systemId": "shop", "codebases": "*", "override": None}
        key = {"systemId": "systemId", "runKind": "runKind", "override": "override", "open-codebases": "codebases"}[slot]
        m[key] = v if slot != "open-codebases" else [v]
        return [m]
    if slot == "op":
        return [{"op": v}]
    return [{"op": "request", "token": token, "request": _request(slot, v), "body": {"m": 1}}]


SLOTS = sorted(SANITISED_SLOTS | {"codebases-list", "target", "token", "request", "body-key", "body-value", "body-whole",
                                  "identity-subject", "identity-issuer", "identity-key", "repo-name", "systemId",
                                  "runKind", "override", "open-codebases", "op"})


def test_the_sweep_covers_every_sanitised_field():
    clean, _ = policy.sanitize({}, 1)
    assert {k for k in clean if not k.startswith("_")} == SANITISED_SLOTS


@pytest.mark.parametrize("via", ["wire", "direct"])
@pytest.mark.parametrize("slot", SLOTS)
@pytest.mark.parametrize("cls", list(HOSTILE))
def test_hostile_caller_string_never_latches_or_invalidates_a_record(tmp_path, slot, cls, via):
    """``wire``: through the strict frame parser, as the socket delivers it. ``direct``: the decoded message goes
    straight to `Broker.handle`, so each layer (parser, sanitiser, record writer) is tested without the one above."""
    v = HOSTILE[cls]
    h = Harness(tmp_path)
    token = h.open("answer", "*")["token"]
    legit_allow = slot.startswith("body") or slot == "identity-subject"  # a valid value there is a real allow
    n_before = len(h.records())
    for msg in _messages(slot, v, token):
        parsed = msg
        if via == "wire":
            wire = json.dumps(msg).encode()  # ensure_ascii: a lone surrogate travels as the escape \ud800
            try:
                parsed = framing.parse_strict(wire)
            except framing.FrameError:
                continue  # a malformed frame: the connection closes with no record (§2.2)
        try:
            h.broker.handle(parsed)
        except framing.FrameError:
            continue  # an unknown op: the connection closes with no reply
    assert not h.audit.latched, (slot, cls, h.logs)
    recs = h.records()  # read_records validates every record against the schema
    assert len(recs) >= n_before
    if not legit_allow:
        assert h.exchange.calls == [], (slot, cls)
    assert h.broker.handle({"op": "open_run", "runKind": "answer", "systemId": "shop", "codebases": "*",
                            "override": None})["ok"]
