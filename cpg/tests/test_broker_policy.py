"""P1-29 — the broker core's policy: runs, mode resolution, the rule table, address validation, S6.

Spec: `docs/plans/connected-code-graph-egress-spec3.md` §2.1, §2.4a, §3.2-§3.5, §4.3 (S6), §10.1 tests (10)-(13), (20)
(token part) and the S6 paragraph. Each test drives `Broker.handle` with injected resolver, interface reader, exchange
and clock, and reads the verdict back from the audit records. Expected rule ids, reasons and addresses are literals
written here; the mutants that turn each red are named in the report of the step.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from broker_helpers import (BRIDGE_NET, HOST_ADDR, FakeExchange, Harness, Resolver, egress, fake_iface, hosted_call,
                            load_system, model_call, write_manifest)
from ccg import manifest as mf
from ccg.broker import addr as addr_mod
from ccg.broker import policy

TOKEN_RE = r"[0-9a-f]{64}"


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


_UNSET = object()


def decide(h: Harness, token, request, body=_UNSET) -> dict:
    """Ask and return the decision record. A POST with no body argument gets a small one (None means none)."""
    if body is _UNSET:
        body = {"m": 1} if (request.get("target") or {}).get("method") == "POST" else None
    h.ask(token, request, body)
    return h.last_decision()


def verdict(d: dict) -> tuple:
    return d["decision"], d["rule"], d["reason"]


# ------------------------------------------------------------------ (1) open_run and tokens


def test_open_run_grant_and_policy_record(h):
    g = h.open("answer", ["docs", "lib"])
    assert set(g) == {"runId", "token", "effective", "systemLocalOnly", "auditId"}
    assert g["effective"] == {"docs": "hosted", "lib": "hosted"}
    assert g["systemLocalOnly"] is True  # `front` is local-only by default and part of the system
    [rec] = h.records("policy")
    assert rec["runId"] == g["runId"] and rec["auditId"] == g["auditId"]
    assert rec["codebases"] == ["docs", "lib"] and rec["systemId"] == "shop" and rec["runKind"] == "answer"
    assert rec["auditDir"] == str(h.audit_dir)


def test_star_expands_to_every_repository(h):
    g = h.open("answer", "*")
    assert g["effective"] == {"front": "local-only", "docs": "hosted", "lib": "hosted"}
    assert h.records("policy")[0]["codebases"] == ["front", "docs", "lib"]


def test_token_is_256_bit_hex_and_unique(h):
    toks = {h.open("answer", "*")["token"] for _ in range(20)}
    assert len(toks) == 20
    import re
    assert all(re.fullmatch(TOKEN_RE, t) for t in toks)


def test_run_table_holds_only_the_hash(h):
    g = h.open("answer", "*")
    keys = list(h.broker.runs._runs)
    assert keys == [hashlib.sha256(g["token"].encode()).hexdigest()]
    assert g["token"] not in repr(h.broker.runs._runs)


@pytest.mark.parametrize("kind,cbs", [
    ("vulndb", ["front"]), ("vulndb", "*"), ("refresh", []), ("answer", []), ("refresh", ["front", "front"]),
    ("answer", ["nosuch"]), ("answer", "all"), ("answer", [1]), ("nonsense", ["front"]), ("answer", None),
])
def test_open_run_cardinality_is_refused_and_writes_no_record(h, kind, cbs):
    r = h.broker.handle({"op": "open_run", "runKind": kind, "systemId": "shop", "codebases": cbs, "override": None})
    assert r["ok"] is False and r["error"] == "open-run-invalid"
    assert h.records() == []
    assert len(h.broker.runs) == 0


def test_open_run_unknown_system_and_override(h):
    for sid, ov in (("nosuch", None), ("shop", "both")):
        r = h.broker.handle({"op": "open_run", "runKind": "answer", "systemId": sid, "codebases": "*", "override": ov})
        assert r["error"] == "open-run-invalid"
    assert h.records() == []


def test_vulndb_run_has_exactly_no_codebases(h):
    g = h.open("vulndb", [])
    assert g["effective"] == {}
    assert h.records("policy")[0]["codebases"] == []


def test_token_expires_at_close_run(h):
    g = h.open("answer", "*")
    assert verdict(decide(h, g["token"], model_call()))[0] == "allow"
    assert h.broker.handle({"op": "close_run", "token": g["token"]}) == {"ok": True}
    assert h.broker.handle({"op": "close_run", "token": g["token"]}) == {"ok": True}  # idempotent
    d = decide(h, g["token"], model_call())
    assert verdict(d) == ("deny", "R0-INVALID", "token-invalid") and d["runId"] is None
    assert len(h.records("policy")) == 1  # close_run writes no record


def test_token_expires_after_24_hours(h):
    g = h.open("answer", "*")
    h.clock.t += 24 * 3600 - 1
    assert verdict(decide(h, g["token"], model_call()))[0] == "allow"
    h.clock.t += 1
    assert verdict(decide(h, g["token"], model_call())) == ("deny", "R0-INVALID", "token-invalid")


@pytest.mark.parametrize("tok", [None, "", "x", "A" * 64, "0" * 64, 7, ["a"]])
def test_absent_or_unknown_token(h, tok):
    h.open("answer", "*")
    d = decide(h, tok, model_call())
    assert verdict(d) == ("deny", "R0-INVALID", "token-invalid")
    assert d["runId"] is None and d["requestMode"] is None
    assert h.exchange.calls == [] and h.resolver.calls == []


# ------------------------------------------------------------------ (2) mode resolution


@pytest.mark.parametrize("repo,override,eff,rule,refused", [
    ("front", None, "local-only", "P1-DEFAULT", False),
    ("front", "local", "local-only", "P1-DEFAULT", False),
    ("front", "hosted", "local-only", "P2-OVERRIDE-REFUSED", True),
    ("docs", None, "hosted", "P1-DEFAULT", False),
    ("docs", "hosted", "hosted", "P1-DEFAULT", False),
    ("docs", "local", "local-only", "P3-OVERRIDE-LOCAL", False),
])
def test_mode_resolution_table(h, repo, override, eff, rule, refused):
    g = h.open("refresh", [repo], override)
    [rec] = h.records("policy")
    default = "local-only" if repo == "front" else "hosted"
    assert rec["modes"] == [{"repoId": repo, "default": default, "effective": eff, "rule": rule}]
    assert rec["overrideRefused"] is refused and rec["override"] == override
    assert g["effective"] == {repo: eff}  # a refused override does not abort the run


def test_system_local_only_covers_the_whole_system(tmp_path):
    allhosted = Harness(tmp_path, systems={"shop": load_system(tmp_path, repos=(("a", "hosted"), ("b", "hosted")))})
    assert allhosted.open("answer", ["a"])["systemLocalOnly"] is False
    # an override-to-local on one run codebase makes the system local-only for this run
    assert allhosted.open("answer", ["a"], "local")["systemLocalOnly"] is True
    # a local-only repository outside the run's codebases still makes it so
    sub = tmp_path / "m"
    sub.mkdir()
    mixed = Harness(sub, systems={"shop": load_system(sub, repos=(("a", "hosted"), ("b", "local-only")))})
    assert mixed.open("answer", ["a"])["systemLocalOnly"] is True


# ------------------------------------------------------------------ (3) R1 / R5 / R6 / R7 and their precedence


def test_local_model_call_is_allowed_by_r1_in_every_mode(h):
    g = h.open("answer", "*")
    d = decide(h, g["token"], model_call())
    assert verdict(d) == ("allow", "R1-LOCAL-ENDPOINT", None) and d["requestMode"] == "local-only"
    assert d["resolvedAddr"] == "127.0.0.1" and h.resolver.calls == []  # local entries are never resolved


def test_hosted_model_call_in_a_mixed_system_is_r5_even_when_only_hosted_repos_are_touched(h):
    g = h.open("answer", ["docs"])  # `docs` is hosted, but `front` makes the system local-only
    d = decide(h, g["token"], hosted_call(codebases=["docs"]))
    assert verdict(d) == ("deny", "R5-SYSTEM-LOCAL-MODEL", "system-local-only-model") and d["requestMode"] == "hosted"
    d = decide(h, g["token"], model_call(codebases=["docs"]))
    assert verdict(d) == ("allow", "R1-LOCAL-ENDPOINT", None)  # row 8.7: interpreted by the local model


def test_hosted_model_call_is_r7_when_no_repository_is_local_only(tmp_path):
    hh = Harness(tmp_path, systems={"shop": load_system(tmp_path, repos=(("a", "hosted"), ("b", "hosted")))})
    g = hh.open("answer", "*")
    d = decide(hh, g["token"], hosted_call(codebases=["a", "b"]))
    assert verdict(d) == ("allow", "R7-HOSTED-SERVICE", None) and d["requestMode"] == "hosted"
    assert hh.resolver.calls == ["api.hosted.example.com."]
    assert hh.exchange.calls[0]["addr"] == "93.184.216.34" and hh.exchange.calls[0]["tls_context"] == "tls"


def test_r5_precedes_r6_for_a_local_only_answer(h):
    g = h.open("answer", ["front"])
    d = decide(h, g["token"], hosted_call(codebases=["front"]))
    assert verdict(d) == ("deny", "R5-SYSTEM-LOCAL-MODEL", "system-local-only-model") and d["requestMode"] == "local-only"


def _pr_to_hosted(**over):  # purpose pr-addon to a hosted-section endpoint: no R4 entry, so only R6 / R99 can apply
    return {"component": "answer.pr-adapter", "purpose": "pr-addon", "payloadClass": "repo-derived",
            "endpoint": "https://api.hosted.example.com:443", "target": {"method": "POST", "path": "/v1/chat/completions"},
            "codebases": ["docs"], **over}


def test_r6_is_sticky_and_r99_is_the_default(h, tmp_path):
    g = h.open("answer", ["front"])
    assert verdict(decide(h, g["token"], _pr_to_hosted(codebases=["front"]), {})) == (
        "deny", "R6-STICKY-LOCAL", "sticky-local-only")
    g2 = h.open("answer", ["docs"])  # hosted-only scope: nothing matches, the default deny applies
    assert verdict(decide(h, g2["token"], _pr_to_hosted(), {})) == ("deny", "R99-DEFAULT-DENY", "no-rule-matched")


def test_an_answer_run_scope_is_the_run_not_the_declaration(h):
    g = h.open("answer", "*")  # touches `front`
    d = decide(h, g["token"], _pr_to_hosted(codebases=["docs"]), {})  # declares only the hosted repository
    assert verdict(d) == ("deny", "R6-STICKY-LOCAL", "sticky-local-only") and d["requestMode"] == "local-only"


def test_a_refresh_run_scope_is_the_requests_codebases(h):
    g = h.open("refresh", ["front", "docs"])
    hosted_only = {"component": "access.check", "purpose": "access-check", "payloadClass": "repo-derived",
                   "endpoint": "https://api.hosted.example.com:443",
                   "target": {"method": "POST", "path": "/v1/chat/completions"}}
    d = decide(h, g["token"], {**hosted_only, "codebases": ["docs"]}, {})
    assert verdict(d) == ("deny", "R99-DEFAULT-DENY", "no-rule-matched") and d["requestMode"] == "hosted"
    d = decide(h, g["token"], {**hosted_only, "codebases": ["front"]}, {})
    assert verdict(d) == ("deny", "R6-STICKY-LOCAL", "sticky-local-only") and d["requestMode"] == "local-only"
    d = decide(h, g["token"], {**hosted_only, "codebases": ["front", "docs"]}, {})
    assert d["requestMode"] == "local-only"


def test_a_refresh_declaration_that_cannot_be_trusted_never_narrows_the_scope(h):
    g = h.open("refresh", ["front", "docs"])
    for bad in (["nosuch"], ["docs", "docs"], "front", None):
        d = decide(h, g["token"], {"component": "access.check", "purpose": "access-check",
                                   "payloadClass": "repo-derived", "endpoint": "https://api.hosted.example.com:443",
                                   "target": {"method": "POST", "path": "/v1/chat/completions"}, "codebases": bad}, {})
        assert d["rule"] == "R0-INVALID" and d["requestMode"] == "local-only"


def test_override_runs_resolve_modes_into_the_scope(h):
    g = h.open("answer", ["docs"], "local")  # hosted-default with a local override stays local: row 8.11
    assert g["effective"] == {"docs": "local-only"}
    d = decide(h, g["token"], hosted_call(codebases=["docs"]))
    assert d["requestMode"] == "local-only" and verdict(d)[1] == "R5-SYSTEM-LOCAL-MODEL"


# ------------------------------------------------------------------ (4) R3 and R4


def _vuln(**over):
    return {"component": "vulndb.refresh", "purpose": "vulndb-refresh", "payloadClass": "none",
            "endpoint": "https://vulndb.example.com:443", "target": {"method": "GET", "path": "/feed"},
            "codebases": [], **over}


def test_r3_allows_only_a_bodiless_get_of_a_vulndb_run(h):
    g = h.open("vulndb", [])
    d = decide(h, g["token"], _vuln())
    assert verdict(d) == ("allow", "R3-VULNDB", None) and d["requestMode"] == "hosted"
    assert d["payloadFields"] == [] and d["bytes"] == 0 and d["bodySha256"] is None
    # a POST (body) does not match R3; it falls through to the default deny
    d = decide(h, g["token"], _vuln(target={"method": "POST", "path": "/feed"}, payloadClass="vuln-feed-request"), {"q": 1})
    assert verdict(d) == ("deny", "R99-DEFAULT-DENY", "no-rule-matched")
    d = decide(h, g["token"], _vuln(target={"method": "POST", "path": "/feed"}), {"q": 1})  # payloadClass stays `none`
    assert verdict(d) == ("deny", "R99-DEFAULT-DENY", "no-rule-matched")
    # R3's endpoint must be a vulndb entry
    d = decide(h, g["token"], _vuln(endpoint="https://idp.example.com:443", target={"method": "GET", "path": "/who"}))
    assert verdict(d) == ("deny", "R99-DEFAULT-DENY", "no-rule-matched")


def test_r3_is_closed_to_other_run_kinds(h):
    g = h.open("answer", "*")
    d = decide(h, g["token"], _vuln(codebases=["front"]))
    assert verdict(d) == ("deny", "R0-INVALID", "component-not-in-run-kind")


def _identity(**over):
    return {"repoNames": ["front"], "identity": {"subject": "alice"}, **over}


def _idp_request(**over):
    return {"component": "access.idp", "purpose": "idp", "payloadClass": "repo-names-and-identity",
            "endpoint": "https://idp.example.com:443", "target": {"method": "POST", "path": "/who"},
            "codebases": ["front"], **over}


def test_r4_allows_the_closed_body_even_in_a_local_only_scope(h):
    g = h.open("answer", ["front"])  # M = local-only: only R1, R3, R4 can allow anything
    body = _identity(identity={"subject": "alice", "issuer": "https://idp.example.com:443"})
    d = decide(h, g["token"], _idp_request(component="access.idp"), body)
    assert verdict(d) == ("allow", "R4-NAMED-SERVICE", None) and d["requestMode"] == "local-only"
    assert d["payloadFields"] == ["identity.issuer", "identity.subject", "repoNames"]
    assert d["bodySha256"] == hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()
    assert d["bytes"] == len(json.dumps(body, separators=(",", ":")).encode())


def test_r4_needs_an_entry_of_the_same_kind(h):
    g = h.open("answer", ["front"])
    d = decide(h, g["token"], _idp_request(component="access.check", purpose="access-check"), _identity())
    assert verdict(d) == ("deny", "R6-STICKY-LOCAL", "sticky-local-only")  # the idp entry is not an access-check entry
    d = decide(h, g["token"], _idp_request(component="access.check", purpose="access-check",
                                           endpoint="https://acl.example.com:443", target={"method": "POST", "path": "/check"}),
               _identity())
    assert verdict(d) == ("allow", "R4-NAMED-SERVICE", None)


@pytest.mark.parametrize("name,body", [
    ("extra key", {**_identity(), "note": "x"}),
    ("extra identity key", _identity(identity={"subject": "a", "extra": 1})),
    ("name outside the system", _identity(repoNames=["nosuch"])),
    ("repeated name", _identity(repoNames=["front", "front"])),
    ("empty names", _identity(repoNames=[])),
    ("non-list names", _identity(repoNames="front")),
    ("non-string name", _identity(repoNames=[1])),
    ("long subject", _identity(identity={"subject": "a" * 257})),
    ("empty subject", _identity(identity={"subject": ""})),
    ("space in subject", _identity(identity={"subject": "a b"})),
    ("control char", _identity(identity={"subject": "a\x01"})),
    ("trailing newline", _identity(identity={"subject": "alice\n"})),
    ("non-ascii", _identity(identity={"subject": "alicé"})),
    ("issuer not an idp", _identity(identity={"subject": "a", "issuer": "https://acl.example.com:443"})),
    ("issuer unknown", _identity(identity={"subject": "a", "issuer": "https://evil.example.com:443"})),
    ("no subject", _identity(identity={})),
    ("not an object", ["front"]),
])
def test_r4_body_shape_violations_are_payload_shape(h, name, body):
    g = h.open("answer", ["front"])
    assert verdict(decide(h, g["token"], _idp_request(), body)) == ("deny", "R4-NAMED-SERVICE", "payload-shape"), name
    assert h.exchange.calls == []


def test_r4_with_the_wrong_payload_class_is_payload_shape(h):
    g = h.open("answer", ["front"])
    assert verdict(decide(h, g["token"], _idp_request(payloadClass="repo-derived"), _identity())) == (
        "deny", "R4-NAMED-SERVICE", "payload-shape")


def test_r4_subject_bound_accepts_the_edges(h):
    g = h.open("answer", ["front"])
    for s in ("a", "!" * 256, "~" * 256):
        assert verdict(decide(h, g["token"], _idp_request(), _identity(identity={"subject": s})))[0] == "allow"


# ------------------------------------------------------------------ (10) R0, every reason and the order


def _ok(**over):
    return model_call(**over)


R0_CASES = [  # (reason, token?, request) each failing exactly that check
    ("enum-invalid", True, _ok(component="answer.nosuch")),
    ("enum-invalid", True, _ok(purpose="nosuch")),
    ("enum-invalid", True, _ok(payloadClass="nosuch")),
    ("enum-invalid", True, _ok(target={"method": "PUT", "path": "/v1/chat/completions"})),
    ("component-not-in-run-kind", True, _ok(component="build.frontend", purpose="dependency-download")),
    ("component-not-in-run-kind", True, _ok(component="vulndb.refresh", purpose="vulndb-refresh")),
    ("component-purpose-mismatch", True, _ok(purpose="idp")),
    ("codebases-empty", True, _ok(codebases=[])),
    ("codebases-not-subset", True, _ok(codebases=["front", "front"])),
    ("codebases-not-subset", True, _ok(codebases=["nosuch"])),
    ("codebases-not-subset", True, _ok(codebases="front")),
    ("endpoint-not-canonical", True, _ok(endpoint="HTTP://127.0.0.1:1234")),
    ("endpoint-not-canonical", True, _ok(endpoint="http://127.0.0.1:1234/")),
    ("endpoint-not-canonical", True, _ok(endpoint="http://127.0.0.1")),
    ("endpoint-not-canonical", True, _ok(endpoint=None)),
    ("path-not-allowed", True, _ok(target={"method": "POST", "path": "/v1/chat/completions/"})),
    ("path-not-allowed", True, _ok(target={"method": "POST", "path": "/v1/chat/completions?x=1"})),
    ("path-not-allowed", True, _ok(target={"method": "POST", "path": None})),
]


@pytest.mark.parametrize("reason,_t,request_", R0_CASES)
def test_every_r0_reason(h, reason, _t, request_):
    g = h.open("answer", "*")
    d = decide(h, g["token"], request_, {"messages": []})
    assert verdict(d) == ("deny", "R0-INVALID", reason)
    assert h.exchange.calls == [] and h.resolver.calls == []


def test_r0_method_body_mismatch_and_body_too_large(h, monkeypatch):
    g = h.open("answer", "*")
    get_with_body = _ok(payloadClass="none", target={"method": "GET", "path": "/v1/chat/completions"})
    assert verdict(decide(h, g["token"], get_with_body, {"a": 1}))[2] == "method-body-mismatch"
    get_with_class = _ok(target={"method": "GET", "path": "/v1/chat/completions"})
    assert verdict(decide(h, g["token"], get_with_class))[2] == "method-body-mismatch"
    assert verdict(decide(h, g["token"], _ok(), None))[2] == "method-body-mismatch"  # POST with no body
    from ccg.broker import framing
    monkeypatch.setattr(framing, "MAX_REQUEST_BYTES", 10)
    assert verdict(decide(h, g["token"], _ok(), {"a": "x" * 30}))[2] == "body-too-large"
    assert verdict(decide(h, g["token"], _ok(), {"a": 1}))[0] == "allow"  # 7 serialized bytes: under the bound


def test_r0_codebases_in_vulndb(h):
    g = h.open("vulndb", [])
    assert verdict(decide(h, g["token"], _vuln(codebases=["front"])))[2] == "codebases-in-vulndb"
    assert verdict(decide(h, g["token"], _vuln(codebases=[])))[0] == "allow"
    assert verdict(decide(h, g["token"], _vuln(codebases=None)))[2] == "codebases-not-subset"


def test_codebases_empty_applies_to_refresh_too(h):
    g = h.open("refresh", ["front"])
    d = decide(h, g["token"], {"component": "access.idp", "purpose": "idp", "payloadClass": "repo-names-and-identity",
                               "endpoint": "https://idp.example.com:443", "target": {"method": "POST", "path": "/who"},
                               "codebases": []}, _identity())
    assert verdict(d)[2] == "codebases-empty"


# Each case fails the named earlier check AND a later one; the earlier reason must win (a reorder mutant fails).
BAD = {
    "token-invalid": dict(token=False),
    "enum-invalid": dict(component="answer.nosuch"),
    "component-not-in-run-kind": dict(component="build.frontend"),  # its purpose, model-call, is a mismatch as well
    "component-purpose-mismatch": dict(purpose="idp"),
    "codebases-empty": dict(codebases=[]),
    "codebases-not-subset": dict(codebases=["nosuch"]),
    "endpoint-not-canonical": dict(endpoint="HTTP://x"),
    "path-not-allowed": dict(target={"method": "POST", "path": "/nope"}),
    "method-body-mismatch": dict(body=None),
}
ORDER = list(BAD)


@pytest.mark.parametrize("i", range(len(ORDER) - 1))
def test_r0_checks_run_in_order(h, i):
    earlier, later = ORDER[i], ORDER[i + 1]
    # combine the earlier fault with the faults of every later check that can coexist with it
    combo: dict = {}
    for name in reversed(ORDER[i:]):
        combo.update(BAD[name])
    # `endpoint-not-canonical` and `path-not-allowed` cannot coexist (no entry matches a bad endpoint): drop the path
    if "endpoint" in combo:
        combo.pop("target", None)
    g = h.open("answer", "*")
    use_token = combo.pop("token", True)
    body = combo.pop("body", {"m": 1})
    req = _ok(**combo)
    tok = g["token"] if use_token else "0" * 64
    d = decide(h, tok, req, body)
    assert d["reason"] == earlier, (earlier, later)


def test_r0_order_path_beats_method_and_size(h, monkeypatch):
    from ccg.broker import framing
    monkeypatch.setattr(framing, "MAX_REQUEST_BYTES", 3)
    g = h.open("answer", "*")
    bad_path_and_body = _ok(target={"method": "GET", "path": "/nope"}, payloadClass="none")
    assert decide(h, g["token"], bad_path_and_body, {"a": "x" * 50})["reason"] == "path-not-allowed"
    assert decide(h, g["token"], _ok(payloadClass="none", target={"method": "GET", "path": "/v1/chat/completions"}),
                  {"a": "x" * 50})["reason"] == "method-body-mismatch"  # beats body-too-large


# component <-> purpose (§2.4a second table), written out here independently of the code's table
COMPONENT_PURPOSE = {
    "build.git": {"git-fetch"},
    "build.frontend": {"dependency-download", "import-name-lookup"},
    "build.indexer": {"dependency-download", "import-name-lookup"},
    "build.pkgmgr": {"dependency-download", "import-name-lookup"},
    "answer.interpret": {"model-call"},
    "answer.summarize": {"model-call"},
    "answer.pr-adapter": {"pr-addon"},
    "access.idp": {"idp"},
    "access.check": {"access-check"},
    "vulndb.refresh": {"vulndb-refresh"},
}
ALL_PURPOSES = ["git-fetch", "dependency-download", "model-call", "idp", "access-check", "pr-addon", "vulndb-refresh",
                "import-name-lookup"]
RUN_FOR = {"build": "refresh", "access": "refresh", "answer": "answer", "vulndb": "vulndb"}


@pytest.mark.parametrize("component", list(COMPONENT_PURPOSE))
def test_component_purpose_mismatch_for_each_row(h, component):
    kind = RUN_FOR[component.split(".")[0]]
    g = h.open(kind, [] if kind == "vulndb" else ["front"])
    for purpose in ALL_PURPOSES:
        d = decide(h, g["token"], {"component": component, "purpose": purpose, "payloadClass": "repo-derived",
                                   "endpoint": "http://127.0.0.1:1234",
                                   "target": {"method": "POST", "path": "/v1/chat/completions"},
                                   "codebases": [] if kind == "vulndb" else ["front"]}, {})
        mismatch = d["reason"] == "component-purpose-mismatch"
        assert mismatch == (purpose not in COMPONENT_PURPOSE[component]), (component, purpose, d["reason"])


@pytest.mark.parametrize("kind,ok,bad", [
    ("refresh", ["build.frontend", "access.idp"], ["answer.interpret", "vulndb.refresh"]),
    ("answer", ["answer.interpret", "access.check"], ["build.git", "vulndb.refresh"]),
    ("vulndb", ["vulndb.refresh"], ["build.git", "answer.interpret", "access.idp"]),
])
def test_components_per_run_kind(h, kind, ok, bad):
    g = h.open(kind, [] if kind == "vulndb" else ["front"])
    cb = [] if kind == "vulndb" else ["front"]
    for c in ok + bad:
        purpose = sorted(COMPONENT_PURPOSE[c])[0]
        d = decide(h, g["token"], {"component": c, "purpose": purpose, "payloadClass": "none",
                                   "endpoint": "http://127.0.0.1:1234",
                                   "target": {"method": "GET", "path": "/v1/chat/completions"}, "codebases": cb})
        assert (d["reason"] == "component-not-in-run-kind") == (c in bad), (kind, c, d["reason"])


# ------------------------------------------------------------------ (11) R2, R8, R9 deny even a local endpoint


@pytest.mark.parametrize("component,purpose,rule,reason", [
    ("build.git", "git-fetch", "R2-GIT-OWN-REMOTE", "not-in-p1:git-fetch"),
    ("build.frontend", "dependency-download", "R8-CONTROLLED-REGISTRY", "not-in-p1:controlled-download"),
    ("build.indexer", "import-name-lookup", "R9-IMPORT-NAME-SERVICE", "not-in-p1:import-name-service"),
])
def test_r2_r8_r9_deny_a_local_endpoint(h, component, purpose, rule, reason):
    g = h.open("refresh", ["front"])
    d = decide(h, g["token"], {"component": component, "purpose": purpose, "payloadClass": "repo-derived",
                               "endpoint": "http://127.0.0.1:1234",
                               "target": {"method": "POST", "path": "/v1/chat/completions"}, "codebases": ["front"]},
               {"x": 1})
    assert verdict(d) == ("deny", rule, reason)
    assert h.exchange.calls == [] and h.resolver.calls == []


def test_r2_r8_r9_deny_a_hosted_and_an_unlisted_endpoint_with_the_same_reason(h):
    g = h.open("refresh", ["docs"])
    for ep, path in (("https://api.hosted.example.com:443", "/v1/chat/completions"), ("https://nope.example.com:443", "/x")):
        d = decide(h, g["token"], {"component": "build.git", "purpose": "git-fetch", "payloadClass": "repo-derived",
                                   "endpoint": ep, "target": {"method": "POST", "path": path}, "codebases": ["docs"]},
                   {"x": 1})
        assert verdict(d) == ("deny", "R2-GIT-OWN-REMOTE", "not-in-p1:git-fetch")


# ------------------------------------------------------------------ the deny vocabulary and rule ids are closed


def test_deny_reasons_are_the_21_of_section_3_5():
    assert len(policy.DENY_REASONS) == 21 and len(set(policy.DENY_REASONS)) == 21
    schema = json.load(open(policy.__file__.replace("policy.py", "../schemas/egress-audit-v1.schema.json")))
    assert set(schema["$defs"]["denyReason"]["enum"]) == set(policy.DENY_REASONS)
    assert policy.RULES == ("R0-INVALID", "R2-GIT-OWN-REMOTE", "R8-CONTROLLED-REGISTRY", "R9-IMPORT-NAME-SERVICE",
                            "R1-LOCAL-ENDPOINT", "R3-VULNDB", "R4-NAMED-SERVICE", "R5-SYSTEM-LOCAL-MODEL",
                            "R6-STICKY-LOCAL", "R7-HOSTED-SERVICE", "R99-DEFAULT-DENY")


def test_every_decision_reason_is_in_the_vocabulary_and_allow_has_none(h):
    g = h.open("answer", "*")
    for r in (model_call(), hosted_call(codebases=["docs"]), model_call(purpose="idp"), model_call(endpoint="x")):
        h.ask(g["token"], r, {"m": 1})
    for d in h.records("decision"):
        assert (d["reason"] is None) == (d["decision"] == "allow")
        assert d["reason"] is None or d["reason"] in policy.DENY_REASONS


# ------------------------------------------------------------------ (12) resolution and the forbidden set


def hosted_system(tmp_path, **kw):
    return {"shop": load_system(tmp_path, repos=(("a", "hosted"), ("b", "hosted")), **kw)}


def _hosted_ask(hh, token):
    return decide(hh, token, hosted_call(codebases=["a"]))


def test_no_resolver_call_for_any_deny(tmp_path):
    hh = Harness(tmp_path, systems=hosted_system(tmp_path))
    g = hh.open("answer", ["a"])
    for r in (hosted_call(purpose="idp", codebases=["a"]), hosted_call(codebases=["nosuch"]),
              hosted_call(endpoint="https://unlisted.example.com:443", codebases=["a"]),
              _pr_to_hosted(codebases=["a"])):
        assert hh.broker.handle({"op": "request", "token": g["token"], "request": r, "body": {}})["decision"]["decision"] == "deny"
    assert hh.resolver.calls == []
    assert hh.exchange.calls == []


def test_the_absolute_name_is_resolved_once_and_the_first_address_is_used(tmp_path):
    r = Resolver({"api.hosted.example.com.": ["93.184.216.34", "93.184.216.99"]})
    hh = Harness(tmp_path, systems=hosted_system(tmp_path), resolver=r)
    g = hh.open("answer", ["a"])
    d = _hosted_ask(hh, g["token"])
    assert verdict(d) == ("allow", "R7-HOSTED-SERVICE", None)
    assert r.calls == ["api.hosted.example.com."]
    assert d["resolvedAddr"] == "93.184.216.34" and hh.exchange.calls[0]["addr"] == "93.184.216.34"


FORBIDDEN = [
    "::ffff:127.0.0.1", "::127.0.0.1", "64:ff9b::7f00:1", "2002:7f00:1::", "0.1.2.3", "240.0.0.1", "255.255.255.255",
    "127.0.0.1", "169.254.169.254", "192.0.0.8", "198.19.0.1", "224.0.0.1", "::", "::1", "fe80::1", "ff02::1",
    "2001:0:4136:e378:8000:63bf:3fff:fdd2", "64:ff9b:1::1",
    "172.17.0.5",       # inside a bridge-kind interface's network
    "172.17.0.1",       # the bridge gateway (a host interface address of the reader's bridge net)
    "192.168.0.1",      # a default gateway
    HOST_ADDR,          # an address of a host interface
    "10.1.2.3", "172.31.0.1", "192.168.7.7", "100.64.0.9", "fd00::1",  # private, no opt-in
]


@pytest.mark.parametrize("bad", FORBIDDEN)
def test_each_forbidden_form_is_forbidden_address(tmp_path, bad):
    r = Resolver({"api.hosted.example.com.": [bad]})
    hh = Harness(tmp_path, systems=hosted_system(tmp_path), resolver=r)
    g = hh.open("answer", ["a"])
    d = _hosted_ask(hh, g["token"])
    assert verdict(d) == ("deny", "R0-INVALID", "forbidden-address")
    assert d["resolvedAddr"] is None and hh.exchange.calls == []


def test_one_bad_address_among_several_denies(tmp_path):
    r = Resolver({"api.hosted.example.com.": ["93.184.216.34", "10.0.0.1"]})
    hh = Harness(tmp_path, systems=hosted_system(tmp_path), resolver=r)
    d = _hosted_ask(hh, hh.open("answer", ["a"])["token"])
    assert d["reason"] == "forbidden-address"


def test_a_local_entry_address_is_forbidden_for_a_non_local_name(tmp_path):
    r = Resolver({"api.hosted.example.com.": ["127.0.0.1"]})
    hh = Harness(tmp_path, systems=hosted_system(tmp_path), resolver=r)
    assert _hosted_ask(hh, hh.open("answer", ["a"])["token"])["reason"] == "forbidden-address"


def test_private_address_opt_in_admits_private_but_never_a_host_address(tmp_path):
    eg = egress()
    eg["hosted"][0]["privateAddress"] = True
    for addr, ok in (("10.1.2.3", True), ("192.168.7.7", True), (HOST_ADDR, False), ("192.168.0.1", False),
                     ("172.17.0.9", False), ("127.0.0.1", False)):
        r = Resolver({"api.hosted.example.com.": [addr]})
        sub = tmp_path / addr.replace(":", "_").replace(".", "_")
        sub.mkdir()
        hh = Harness(sub, systems={"shop": load_system(sub, repos=(("a", "hosted"),), eg=eg)}, resolver=r)
        d = _hosted_ask(hh, hh.open("answer", ["a"])["token"])
        assert (d["decision"] == "allow") == ok, addr
        assert d["privateAddress"] is ok


def test_resolve_failed_and_address_check_unavailable(tmp_path):
    for table, expect in (({"api.hosted.example.com.": []}, "resolve-failed"),
                          ({"api.hosted.example.com.": ["not-an-address"]}, "resolve-failed")):
        sub = tmp_path / expect / str(len(table))
        sub.mkdir(parents=True, exist_ok=True)
        hh = Harness(sub, systems=hosted_system(sub), resolver=Resolver(table))
        assert _hosted_ask(hh, hh.open("answer", ["a"])["token"])["reason"] == expect
    sub = tmp_path / "gone"
    sub.mkdir()
    hh = Harness(sub, systems=hosted_system(sub), resolver=Resolver({"api.hosted.example.com.": []}))
    hh.resolver.table.pop("api.hosted.example.com.")  # raises OSError
    assert _hosted_ask(hh, hh.open("answer", ["a"])["token"])["reason"] == "resolve-failed"

    def unreadable():
        raise addr_mod.IfaceUnavailable("ip: not found")
    sub = tmp_path / "noip"
    sub.mkdir()
    hh = Harness(sub, systems=hosted_system(sub), iface=unreadable)
    d = _hosted_ask(hh, hh.open("answer", ["a"])["token"])
    assert verdict(d) == ("deny", "R0-INVALID", "address-check-unavailable") and hh.exchange.calls == []


def test_an_ip_literal_remote_entry_needs_no_resolution_but_is_still_checked(tmp_path):
    eg = egress()
    eg["hosted"][0]["endpoint"] = "https://93.184.216.40:443"
    sub = tmp_path
    hh = Harness(sub, systems={"shop": load_system(sub, repos=(("a", "hosted"),), eg=eg)})
    d = decide(hh, hh.open("answer", ["a"])["token"], hosted_call(endpoint="https://93.184.216.40:443", codebases=["a"]))
    assert d["decision"] == "allow" and d["resolvedAddr"] == "93.184.216.40" and hh.resolver.calls == []
    eg["hosted"][0]["endpoint"] = "https://10.0.0.5:443"
    sub2 = tmp_path / "p"
    sub2.mkdir()
    hh = Harness(sub2, systems={"shop": load_system(sub2, repos=(("a", "hosted"),), eg=eg)})
    d = decide(hh, hh.open("answer", ["a"])["token"], hosted_call(endpoint="https://10.0.0.5:443", codebases=["a"]))
    assert d["reason"] == "forbidden-address"


# ------------------------------------------------------------------ S6 (spec §10.1, P1-29's part)


def _s6(tmp_path, local_host, iface=None):
    eg = egress(local_host=local_host)
    m = mf.load_manifest(write_manifest(tmp_path, eg=eg))
    return addr_mod.check_local_entries(m.egress, iface or fake_iface())


def test_s6_admits_loopback_and_a_host_interface_address(tmp_path):
    assert _s6(tmp_path, "127.0.0.1") == []
    assert _s6(tmp_path, "127.5.5.5") == []
    assert _s6(tmp_path, HOST_ADDR) == []
    assert _s6(tmp_path, "localhost") == []
    assert _s6(tmp_path, "[::1]") == []


@pytest.mark.parametrize("host", ["8.8.8.8", "192.168.0.77", "192.168.0.1", "[fe80::1]", "10.255.255.254"])
def test_s6_refuses_a_public_a_lan_non_host_and_the_gateways(tmp_path, host):
    # 192.168.0.1 and fe80::1 are this host's default gateways; 192.168.0.77 is a LAN address that is no host address
    msgs = _s6(tmp_path, host)
    assert len(msgs) == 1 and msgs[0].startswith("S6:")


def test_s6_never_admits_a_gateway_even_if_the_reader_lists_it_as_an_address(tmp_path):
    iface = fake_iface(addresses=(HOST_ADDR, "192.168.0.1"))
    assert len(_s6(tmp_path, "192.168.0.1", iface)) == 1


def test_s6_loopback_needs_no_interface_data(tmp_path):
    def boom():
        raise AssertionError("must not read the interfaces for a loopback literal")
    assert _s6(tmp_path, "127.0.0.1", boom) == []


def test_s6_unreadable_interface_data_raises(tmp_path):
    def unreadable():
        raise addr_mod.IfaceUnavailable("x")
    with pytest.raises(addr_mod.IfaceUnavailable):
        _s6(tmp_path, HOST_ADDR, unreadable)


def test_an_entry_whose_address_the_host_loses_is_denied_on_the_next_r1_request(tmp_path):
    state = {"addrs": (HOST_ADDR, "127.0.0.1", "::1")}

    def iface():
        return fake_iface(addresses=state["addrs"])()
    sub = tmp_path
    eg = egress(port=5555, local_host=HOST_ADDR)
    hh = Harness(sub, systems={"shop": load_system(sub, eg=eg)}, iface=iface)
    g = hh.open("answer", "*")
    ask = model_call(endpoint=f"http://{HOST_ADDR}:5555")
    d = decide(hh, g["token"], ask)
    assert verdict(d) == ("allow", "R1-LOCAL-ENDPOINT", None) and d["resolvedAddr"] == HOST_ADDR
    assert len(hh.exchange.calls) == 1
    state["addrs"] = ("127.0.0.1", "::1")  # the interface went away
    d = decide(hh, g["token"], ask)
    assert verdict(d) == ("deny", "R0-INVALID", "forbidden-address") and d["resolvedAddr"] is None
    assert len(hh.exchange.calls) == 1  # no connection attempted


def test_the_ip_reader_runs_the_absolute_ip_binary_with_the_listed_arguments():
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        import subprocess
        out = {
            "address": '[{"ifname":"br-1","linkinfo":{"info_kind":"bridge"},"addr_info":[{"local":"172.20.0.1","prefixlen":16}]},'
                       '{"ifname":"eth0","addr_info":[{"local":"192.168.0.64","prefixlen":24}]}]',
            "-6": '[{"dst":"default","gateway":"fe80::1"}]',
            "route": '[{"dst":"default","gateway":"192.168.0.1"}]',
        }
        key = "address" if "address" in argv else ("-6" if "-6" in argv else "route")
        return subprocess.CompletedProcess(argv, 0, out[key].encode(), b"")
    data = addr_mod.read_interfaces(run)
    assert [a[:1] for a in seen] == [["/usr/bin/ip"]] * 3
    assert seen[0] == ["/usr/bin/ip", "-j", "-d", "address", "show"]
    assert seen[1] == ["/usr/bin/ip", "-j", "route", "show", "default"]
    assert seen[2] == ["/usr/bin/ip", "-6", "-j", "route", "show", "default"]
    assert [str(n) for n in data.bridge_nets] == ["172.20.0.0/16"]  # only the bridge-kind interface
    assert sorted(map(str, data.addresses)) == ["172.20.0.1", "192.168.0.64"]
    assert sorted(map(str, data.gateways)) == ["192.168.0.1", "fe80::1"]
