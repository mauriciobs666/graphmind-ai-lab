"""P1-29 — the broker's audit trail: record fields, the schema, caller junk, the latch, the file checks, the token.

Spec: `docs/plans/connected-code-graph-egress-spec3.md` §5.1-§5.2, §2.5, §10.1 tests (9), (14) (the record part), (15),
(20). The expected field sets are the tables of §5.2 written out here as literals.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat

import jsonschema
import pytest

from broker_helpers import FakeExchange, Harness, load_system, model_call
from ccg.broker import audit
from ccg.broker.audit import AuditUnwritable, AuditWriter, PreconditionError, read_records, validate_record

COMMON = {"schemaVersion", "recordType", "auditId", "ts"}
POLICY_FIELDS = COMMON | {"runId", "runKind", "systemId", "codebases", "override", "modes", "overrideRefused",
                          "systemLocalOnly", "allowListDigest", "auditDir"}
DECISION_FIELDS = COMMON | {"runId", "component", "purpose", "payloadClass", "target", "endpoint", "codebases",
                            "rejected", "resolvedAddr", "privateAddress", "requestMode", "decision", "rule", "reason",
                            "transport", "payloadFields", "bytes", "bodySha256"}
CLOSE_FIELDS = COMMON | {"decisionAuditId", "runId", "bytesOut", "bytesIn", "httpStatus", "durationMs", "closeReason"}
UUID4 = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
TS = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def run_one(h: Harness, **over):
    g = h.open("answer", "*")
    h.ask(g["token"], model_call(**over), {"messages": []})
    return g


# ------------------------------------------------------------------ (9) the fields are exactly §5.2's


def test_record_fields_are_exactly_the_three_tables(h):
    run_one(h)
    [pol], [dec], [cls] = h.records("policy"), h.records("decision"), h.records("close")
    assert set(pol) == POLICY_FIELDS and set(dec) == DECISION_FIELDS and set(cls) == CLOSE_FIELDS
    import re
    for r in (pol, dec, cls):
        assert r["schemaVersion"] == 1 and re.fullmatch(UUID4, r["auditId"]) and re.fullmatch(TS, r["ts"])
    assert re.fullmatch("[0-9a-f]{64}", pol["allowListDigest"])
    assert dec["transport"] == "structured" and cls["decisionAuditId"] == dec["auditId"]
    assert cls["closeReason"] == "completed" and cls["httpStatus"] == 200 and (cls["bytesOut"], cls["bytesIn"]) == (100, 200)
    assert dec["runId"] == cls["runId"] == pol["runId"]


def test_records_are_one_json_line_each_in_order(h):
    run_one(h)
    lines = (h.audit_dir / "egress.jsonl").read_bytes().split(b"\n")
    assert lines[-1] == b"" and [json.loads(x)["recordType"] for x in lines[:-1]] == ["policy", "decision", "close"]
    assert oct(os.stat(h.audit_dir / "egress.jsonl").st_mode & 0o777) == "0o600"


def test_a_deny_has_no_close_and_an_allow_has_exactly_one(h):
    g = h.open("answer", "*")
    h.ask(g["token"], model_call(), {"m": 1})
    h.ask(g["token"], model_call(purpose="idp"), {"m": 1})
    assert [r["recordType"] for r in h.records()] == ["policy", "decision", "close", "decision"]


def test_allow_decision_fields(h):
    run_one(h)
    d = h.records("decision")[0]
    assert d["payloadFields"] == ["messages"] and d["bytes"] == 15
    assert d["bodySha256"] == hashlib.sha256(b'{"messages":[]}').hexdigest()
    assert d["resolvedAddr"] == "127.0.0.1" and d["privateAddress"] is False and d["requestMode"] == "local-only"


def test_the_policy_digest_is_over_egress_and_default_modes(h):
    run_one(h)
    sysm = h.systems["shop"]
    want = hashlib.sha256(json.dumps({"egress": sysm.egress, "modes": {"front": "local-only", "docs": "hosted",
                                                                       "lib": "hosted"}},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert h.records("policy")[0]["allowListDigest"] == want
    # a different allow-list or mode gives a different digest
    other = load_system(h.tmp / "o", repos=(("front", "hosted"), ("docs", "hosted"), ("lib", "hosted"))) \
        if (h.tmp / "o").mkdir() is None else None
    assert other.digest != sysm.digest


# ------------------------------------------------------------------ the schema is closed


def _sample(h) -> dict:
    run_one(h)
    return {"policy": h.records("policy")[0], "decision": h.records("decision")[0], "close": h.records("close")[0]}


def test_schema_rejects_missing_extra_and_out_of_enum_fields(h):
    for kind, rec in _sample(h).items():
        validate_record(rec)
        for f in list(rec):
            bad = {k: v for k, v in rec.items() if k != f}
            with pytest.raises(jsonschema.ValidationError):
                validate_record(bad)
        with pytest.raises(jsonschema.ValidationError):
            validate_record({**rec, "extra": 1})


@pytest.mark.parametrize("kind,field,value", [
    ("decision", "transport", "proxy"), ("decision", "rule", "R2-GIT-OWN-REMOTE-X"), ("decision", "reason", "other"),
    ("decision", "component", "build.nosuch"), ("decision", "requestMode", "both"),
    ("close", "closeReason", "killed"), ("policy", "runKind", "build"), ("policy", "override", "both"),
    ("decision", "auditId", "ABCDEF00-0000-4000-8000-000000000000"), ("decision", "ts", "2026-11-10T10:00:00Z"),
    ("close", "httpStatus", 99), ("decision", "bytes", -1), ("policy", "auditDir", "relative/dir"),
])
def test_schema_enums_admit_only_p1_values(h, kind, field, value):
    rec = copy.deepcopy(_sample(h)[kind])
    rec[field] = value
    with pytest.raises(jsonschema.ValidationError):
        validate_record(rec)


def test_schema_ties_reason_to_decision(h):
    d = copy.deepcopy(_sample(h)["decision"])
    d["reason"] = "no-rule-matched"  # an allow with a reason
    with pytest.raises(jsonschema.ValidationError):
        validate_record(d)
    d["decision"], d["rule"] = "deny", "R99-DEFAULT-DENY"
    validate_record(d)
    d["reason"] = None  # a deny without one
    with pytest.raises(jsonschema.ValidationError):
        validate_record(d)


def test_schema_is_closed_for_nested_rows(h):
    p = copy.deepcopy(_sample(h)["policy"])
    p["modes"][0]["extra"] = 1
    with pytest.raises(jsonschema.ValidationError):
        validate_record(p)
    d = copy.deepcopy(_sample(h)["decision"])
    d["target"]["extra"] = 1
    with pytest.raises(jsonschema.ValidationError):
        validate_record(d)


# ------------------------------------------------------------------ (14) caller junk is recorded null, with a digest


JUNK = [
    ("component", {"component": {"a": 1}}, "component", {"a": 1}),
    ("purpose", {"purpose": "x" * 500}, "purpose", "x" * 500),
    ("payloadClass", {"payloadClass": 5}, "payloadClass", 5),
    ("target.method", {"target": {"method": ["GET"], "path": "/p"}}, "target.method", ["GET"]),
    ("target.path", {"target": {"method": "POST", "path": "/" + "a" * 300}}, "target.path", "/" + "a" * 300),
    ("endpoint", {"endpoint": "HTTP://X:1\n"}, "endpoint", "HTTP://X:1\n"),
    ("codebases", {"codebases": ["Front!"]}, "codebases", ["Front!"]),
]


@pytest.mark.parametrize("name,over,rkey,raw", JUNK)
def test_caller_junk_is_null_with_the_digest_of_its_raw_json(h, name, over, rkey, raw):
    g = h.open("answer", "*")
    h.ask(g["token"], model_call(**over), {"m": 1})
    d = h.records("decision")[-1]  # read_records validated it against the schema
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert d["rejected"] == {rkey: digest}
    field = {"target.method": ("target", "method"), "target.path": ("target", "path")}.get(name, (name,))
    val = d
    for k in field:
        val = val[k]
    assert val is None


def test_codebases_longer_than_the_repository_count_is_junk(h):
    g = h.open("answer", "*")
    h.ask(g["token"], model_call(codebases=["front", "docs", "lib", "front"]), {"m": 1})
    d = h.records("decision")[-1]
    assert d["codebases"] is None and set(d["rejected"]) == {"codebases"}


def test_a_request_that_is_not_an_object_still_records(h):
    g = h.open("answer", "*")
    for junk in ("x", 5, None, [1]):
        r = h.broker.handle({"op": "request", "token": g["token"], "request": junk, "body": None})
        assert r["decision"]["reason"] == "enum-invalid"
    assert len(h.records("decision")) == 4


def test_clean_request_has_empty_rejected(h):
    run_one(h)
    assert h.records("decision")[0]["rejected"] == {}


# ------------------------------------------------------------------ (15) the latch


class FailingWrite:
    """``os.write`` that fails (or short-writes) on the n-th call."""

    def __init__(self, fail_at: int, short: bool = False):
        self.n, self.fail_at, self.short = 0, fail_at, short

    def __call__(self, fd, data):
        self.n += 1
        if self.n == self.fail_at:
            if self.short:
                return os.write(fd, data[: len(data) // 2])
            raise OSError(28, "No space left on device")
        return os.write(fd, data)


def harness_with(tmp_path, write):
    hh = Harness(tmp_path)
    hh.audit._write = write
    return hh


def test_policy_write_failure_issues_no_token_and_latches(tmp_path):
    hh = harness_with(tmp_path, FailingWrite(1))
    r = hh.broker.handle({"op": "open_run", "runKind": "answer", "systemId": "shop", "codebases": "*", "override": None})
    assert r["ok"] is False and r["error"] == "audit-unwritable" and "token" not in json.dumps(r)
    assert len(hh.broker.runs) == 0 and hh.audit.latched
    r = hh.broker.handle({"op": "open_run", "runKind": "answer", "systemId": "shop", "codebases": "*", "override": None})
    assert r["error"] == "audit-unwritable"  # refused from now on, though the disk would take it again


def test_decision_write_failure_makes_no_upstream_connection_and_refuses_later_calls(tmp_path):
    hh = harness_with(tmp_path, FailingWrite(2))  # the policy record is write 1
    g = hh.open("answer", "*")
    r = hh.broker.handle({"op": "request", "token": g["token"], "request": model_call(), "body": {"m": 1}})
    assert r == {"ok": False, "error": "audit-unwritable", "message": r["message"]}
    assert hh.exchange.calls == []
    for op in ({"op": "open_run", "runKind": "answer", "systemId": "shop", "codebases": "*", "override": None},
               {"op": "request", "token": g["token"], "request": model_call(), "body": {"m": 1}}):
        assert hh.broker.handle(op)["error"] == "audit-unwritable"
    assert hh.exchange.calls == []


def test_a_short_write_latches(tmp_path):
    hh = harness_with(tmp_path, FailingWrite(2, short=True))
    g = hh.open("answer", "*")
    r = hh.broker.handle({"op": "request", "token": g["token"], "request": model_call(), "body": {"m": 1}})
    assert r["error"] == "audit-unwritable" and hh.audit.latched and hh.exchange.calls == []


def test_fsync_failure_latches(tmp_path):
    hh = Harness(tmp_path)

    def bad_fsync(fd):
        raise OSError(5, "I/O error")
    hh.audit._fsync = bad_fsync
    r = hh.broker.handle({"op": "open_run", "runKind": "answer", "systemId": "shop", "codebases": "*", "override": None})
    assert r["error"] == "audit-unwritable" and len(hh.broker.runs) == 0


def test_a_close_that_cannot_be_written_still_returns_the_response_then_latches(tmp_path):
    hh = harness_with(tmp_path, FailingWrite(3))  # policy, decision, then the close fails
    g = hh.open("answer", "*")
    r = hh.broker.handle({"op": "request", "token": g["token"], "request": model_call(), "body": {"m": 1}})
    assert r["ok"] and r["decision"]["decision"] == "allow" and r["response"]["status"] == 200
    assert hh.audit.latched
    assert hh.broker.handle({"op": "request", "token": g["token"], "request": model_call(), "body": {"m": 1}})["error"] == "audit-unwritable"


def test_an_exchange_in_flight_when_another_call_trips_the_latch_completes(tmp_path):
    hh = Harness(tmp_path)

    def exchange(*a, **kw):
        hh.audit.latch()  # another call tripped the latch while this exchange was running
        return FakeExchange()(*a, **kw)
    hh.broker.exchanger = exchange
    g = hh.open("answer", "*")
    r = hh.broker.handle({"op": "request", "token": g["token"], "request": model_call(), "body": {"m": 1}})
    assert r["ok"] and r["response"]["status"] == 200  # the response returns; its close could not be written
    assert [x["recordType"] for x in hh.records()] == ["policy", "decision"]


def test_a_broker_built_record_that_fails_validation_latches(h):
    rec = h.audit.record("close", decisionAuditId="x", runId="y", bytesOut=1, bytesIn=1, httpStatus=200, durationMs=1,
                         closeReason="completed")
    with pytest.raises(AuditUnwritable):
        h.audit.append(rec)
    assert h.audit.latched


def test_a_file_not_ending_in_a_newline_is_repaired_at_start(tmp_path):
    d = tmp_path / "a"
    d.mkdir(mode=0o700)
    f = d / "egress.jsonl"
    f.write_bytes(b'{"half":')
    os.chmod(f, 0o600)
    logs = []
    w = AuditWriter(d, log=logs.append)
    assert f.read_bytes() == b'{"half":\n' and logs and "newline" in logs[0]
    w.close()
    # reading skips a trailing partial line and the repaired (invalid) line is reported, not silently skipped
    f.write_bytes(b"")
    w = AuditWriter(d, log=logs.append)
    rec = w.record("close", decisionAuditId="0" * 8 + "-0000-4000-8000-" + "0" * 12, runId="0" * 8 + "-0000-4000-8000-" + "0" * 12,
                   bytesOut=0, bytesIn=0, httpStatus=None, durationMs=0, closeReason="timeout")
    w.append(rec)
    w.close()
    f.write_bytes(f.read_bytes() + b'{"partial":')
    assert [r["recordType"] for r in read_records(f)] == ["close"]


def test_start_refuses_a_symlinked_group_readable_or_hard_linked_file(tmp_path):
    def fresh(name):
        d = tmp_path / name
        d.mkdir(mode=0o700)
        return d
    d = fresh("sym")
    (tmp_path / "elsewhere").write_text("")
    os.chmod(tmp_path / "elsewhere", 0o600)  # an otherwise acceptable target: only the link itself is wrong
    os.symlink(tmp_path / "elsewhere", d / "egress.jsonl")
    with pytest.raises(PreconditionError):
        AuditWriter(d)
    d = fresh("grp")
    (d / "egress.jsonl").write_text("")
    os.chmod(d / "egress.jsonl", 0o640)
    with pytest.raises(PreconditionError, match="permission bits"):
        AuditWriter(d)
    d = fresh("hard")
    (d / "egress.jsonl").write_text("")
    os.chmod(d / "egress.jsonl", 0o600)
    os.link(d / "egress.jsonl", tmp_path / "other-name")
    with pytest.raises(PreconditionError, match="hard links"):
        AuditWriter(d)
    d = fresh("fifo")
    os.mkfifo(d / "egress.jsonl", 0o600)
    with pytest.raises(PreconditionError):
        AuditWriter(d)


def test_start_refuses_a_symlinked_or_writable_audit_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    os.symlink(real, tmp_path / "link")
    with pytest.raises(PreconditionError, match="plain directory"):
        AuditWriter(tmp_path / "link")
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    os.chmod(open_dir, 0o770)
    with pytest.raises(PreconditionError, match="writable"):
        AuditWriter(open_dir)
    os.chmod(open_dir, 0o755)  # group/other readable is fine: only writable is refused
    AuditWriter(open_dir).close()


def test_a_missing_audit_directory_is_created_private(tmp_path):
    w = AuditWriter(tmp_path / "new" / "audit")
    assert stat.S_IMODE(os.stat(tmp_path / "new" / "audit").st_mode) == 0o700
    assert (tmp_path / "new" / "audit" / "egress.jsonl").exists()
    w.close()


def test_the_default_audit_dir_is_the_codes_own_location_not_the_working_directory(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    cpg = Path(__file__).resolve().parent.parent  # written from this test's own location, not from the module
    out = subprocess.run([sys.executable, "-c", "from ccg.broker import audit; print(audit.DEFAULT_AUDIT_DIR)"],
                         cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(cpg / ".audit")  # imported from another working directory


def _stat_of(table):
    """A stat reader answering from ``{path: (mode, uid)}``; anything else is a plain 0755 directory of ours."""
    from types import SimpleNamespace

    def stat_fn(p):
        mode, uid = table.get(str(p), (0o040755, os.getuid()))
        return SimpleNamespace(st_mode=mode, st_uid=uid)
    return stat_fn


@pytest.mark.parametrize("mode,owner,allow_sticky,ok", [
    (0o041777, 0, False, False),        # /tmp-like (sticky, root): spec section 2.0 has no exception
    (0o041777, 0, True, True),          # the test-only exemption: sticky, owned by root
    (0o041777, "me", True, True),       # sticky, owned by the broker's own uid
    (0o041777, "other", True, False),   # sticky but owned by someone else: another uid could swap our subtree
    (0o040777, 0, True, False),         # world-writable without the sticky bit
    (0o040775, "me", True, False),      # group-writable
    (0o040775, "me", False, False),
    (0o040755, "other", False, True),   # not writable: ownership is not this check's business
])
def test_ancestor_check_follows_section_2_0_and_the_exemption_has_an_owner_conjunct(mode, owner, allow_sticky, ok):
    me = os.getuid()
    uid = {0: 0, "me": me, "other": me + 1}[owner]
    anc = "/some/ancestor"
    why = audit._ancestors_ok(audit.Path(anc + "/dir"), me, allow_sticky=allow_sticky, stat_fn=_stat_of({anc: (mode, uid)}))
    assert (why is None) == ok, why


def test_require_dir_is_strict_by_default_and_real_group_writable_ancestor_is_refused(tmp_path):
    d = tmp_path / "x" / "broker"
    os.makedirs(tmp_path / "x", mode=0o700)
    audit.require_dir(d, exact_mode=0o700, create=True, check_ancestors=True, allow_sticky=True)  # /tmp is sticky
    with pytest.raises(PreconditionError, match="ancestor"):
        audit.require_dir(d, exact_mode=0o700, create=True, check_ancestors=True)  # strict: /tmp refused
    os.chmod(tmp_path / "x", 0o770)
    with pytest.raises(PreconditionError, match="ancestor"):
        audit.require_dir(d, exact_mode=0o700, create=True, check_ancestors=True, allow_sticky=True)


def test_the_audit_line_is_ascii_so_a_line_splitting_reader_sees_one_record(h):
    g = h.open("answer", "*")
    h.ask(g["token"], model_call(), {"k\u2028": "v\u2028\u0085"})
    raw = (h.audit_dir / "egress.jsonl").read_bytes()
    assert raw.isascii() and len(raw.decode().splitlines()) == 3  # policy, decision, close: no extra splits
    [dec] = h.records("decision")
    assert dec["payloadFields"] == ["k\u2028"]


def test_a_validation_failure_does_not_log_the_offending_value(h):
    secret = "S" * 5000
    rec = h.audit.record("close", decisionAuditId=secret, runId="y", bytesOut=1, bytesIn=1, httpStatus=200,
                         durationMs=1, closeReason="completed")
    with pytest.raises(AuditUnwritable) as e:
        h.audit.append(rec)
    assert len(str(e.value)) < 400 and all(len(m) < 400 for m in h.logs)


# ------------------------------------------------------------------ (20) the token is never written


def test_the_token_is_in_no_record_and_no_log(tmp_path, capfd):
    hh = Harness(tmp_path)
    g = hh.open("answer", "*")
    for r in (model_call(), model_call(purpose="idp"), model_call(component="x" + g["token"])):
        hh.ask(g["token"], r, {"m": 1})
    hh.ask("0" * 63 + "1", model_call(), {"m": 1})  # an unknown token is not recorded either
    raw = (hh.audit_dir / "egress.jsonl").read_text()
    assert g["token"] not in raw and "0" * 63 + "1" not in raw
    assert all(g["token"] not in line for line in hh.logs)
    cap = capfd.readouterr()
    assert g["token"] not in cap.out + cap.err
    # the hash is not recorded either: nothing in the file lets a reader present a token
    assert hashlib.sha256(g["token"].encode()).hexdigest() not in raw
