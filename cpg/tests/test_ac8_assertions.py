"""P1-31: AC-8 rows 8.1-8.16, the preconditions P-a..P-g and the in-run controls (`cpg/sandbox/ac8/assertions.py`).

Spec: `docs/plans/connected-code-graph-egress-spec3.md` §8.1-§8.2. `ac8_helpers.Scenario` writes a GOOD scenario (records made by the
real broker); each test breaks one thing and reads the row, precondition or control that must turn red, with the other rows
unchanged where that is the point. Row statuses are asserted as literals.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

import ac8_helpers as H
from ac8_helpers import Scenario, row
import assertions as A
import window as W


@pytest.fixture
def sc(tmp_path):
    return Scenario(tmp_path)


def statuses(rep):
    return {r["id"]: r["status"] for r in rep["rows"]}


def bad_pre(rep):
    return sorted(k for k, v in rep["preconditions"].items() if not v["ok"])


def bad_controls(rep):
    return sorted(c["id"] for c in rep["controls"] if not c["ok"])


def edit_egress(sc, fn):
    """``fn`` edits the records in place, or returns a new list of records (a filter)."""
    recs = sc.egress()
    out = fn(recs)
    sc.write_egress(out if isinstance(out, list) and out and all(isinstance(x, dict) for x in out) else recs)


GOOD = {"8.1": "pass", "8.2": "vacuous", "8.3": "pass", "8.4": "vacuous", "8.5": "pass", "8.6": "pass", "8.7": "pass", "8.8": "pass",
        "8.9": "pass", "8.10": "pass", "8.11": "pass", "8.12": "vacuous", "8.13": "pass", "8.14": "pass", "8.15": "pass",
        "8.16": "pass", "close": "pass"}


# ------------------------------------------------------------------ the good scenario, and what "pass" means


def test_good_scenario_passes_with_each_row_as_the_table_says(sc):
    rep = sc.evaluate()
    assert rep["overall"] == "pass" and statuses(rep) == GOOD and bad_pre(rep) == [] and bad_controls(rep) == []


def test_without_strace_row_8_16_is_unrun_and_the_run_is_incomplete_not_passed(tmp_path):
    rep = Scenario(tmp_path, strace=False).evaluate()
    assert statuses(rep) == {**GOOD, "8.16": "unrun"} and rep["overall"] == "incomplete"
    assert "unrun, not passed" in row(rep, "8.16")["details"][0]


def test_a_failure_outranks_an_unrun_row(tmp_path):
    s = Scenario(tmp_path, strace=False)
    edit_egress(s, lambda r: [x for x in r if x.get("reason") != "system-local-only-model"])
    assert s.evaluate()["overall"] == "fail"


CLI_DRIVER = """
import sys
sys.path.insert(0, %r); sys.path.insert(0, %r)
import assertions
from ccg.broker import addr
def _nx(_name):          # the CLI's own default-resolver path, with no DNS query sent
    raise OSError('NXDOMAIN')
addr.default_resolver = _nx
sys.exit(assertions._main(sys.argv[1:]))
"""


def run_cli(*args):
    """The real ``assertions._main`` in a subprocess, its default resolver replaced by one that answers nothing and sends nothing."""
    cpg = str(H.AC8.parent.parent)
    return subprocess.run([sys.executable, "-c", CLI_DRIVER % (str(H.AC8), cpg), *map(str, args)], capture_output=True, text=True, cwd=cpg)


def test_cli_exit_0_pass_3_incomplete_1_fail_exactly(tmp_path):
    # the CLI's argv audit runs the real runner function: the fixture's digest is the real one
    good = Scenario(tmp_path / "g")
    out = tmp_path / "r.json"
    r = run_cli(good.dir, "--json", out)
    assert r.returncode == 0 and json.loads(out.read_text())["overall"] == "pass" and json.loads(out.read_text())["resolver"] == "real", r.stdout
    nostrace = Scenario(tmp_path / "n", strace=False)
    r = run_cli(nostrace.dir)
    assert r.returncode == 3 and "INCOMPLETE" in r.stdout
    broken = Scenario(tmp_path / "b")
    (broken.det / "flush.marker").unlink()
    r = run_cli(broken.dir)
    assert r.returncode == 1 and "AC-8 part 1: FAIL" in r.stdout
    assert run_cli().returncode == 64


def test_cli_stub_resolver_is_a_self_test_and_caps_at_incomplete(tmp_path):
    out = tmp_path / "r.json"
    r = run_cli(Scenario(tmp_path / "g").dir, "--stub-resolver", "--json", out)
    rep = json.loads(out.read_text())
    assert r.returncode == 3 and rep["overall"] == "incomplete" and rep["selfTest"] == ["stub-resolver"] and rep["resolver"].startswith("stub")


# ------------------------------------------------------------------ F3: a self-test stand-in never passes


def set_facts(sc, **kv):
    f = json.loads((sc.dir / "facts.json").read_text())
    f.update(kv)
    (sc.dir / "facts.json").write_text(json.dumps(f))


@pytest.mark.parametrize("flags", [["stub-model"], ["store-standin"], ["witness-parse"], ["stub-resolver"], ["stub-model", "store-standin"]])
def test_a_run_that_used_a_self_test_stand_in_is_incomplete_never_pass(sc, flags):
    set_facts(sc, selfTest=flags)
    rep = sc.evaluate()
    assert rep["overall"] == "incomplete" and rep["selfTest"] == sorted(flags) and statuses(rep) == GOOD
    assert any("self-test stand-ins" in r for r in rep["incompleteReasons"])
    assert "self-test stand-ins: " + ", ".join(sorted(flags)) in A.render(rep)


def test_a_run_with_no_stand_in_and_no_unrun_row_reaches_pass(sc):
    set_facts(sc, selfTest=[])
    rep = sc.evaluate()
    assert rep["overall"] == "pass" and rep["selfTest"] == [] and rep["incompleteReasons"] == []


@pytest.mark.parametrize("bad", ["__absent__", "stub-model", [1], None])
def test_facts_without_a_selftest_list_fail_closed(sc, bad):
    f = json.loads((sc.dir / "facts.json").read_text())
    f.pop("selfTest")
    if bad != "__absent__":
        f["selfTest"] = bad
    (sc.dir / "facts.json").write_text(json.dumps(f))
    rep = sc.evaluate()
    assert rep["overall"] == "fail" and bad_pre(rep) == ["S-facts"]


def test_the_evaluation_own_stub_resolver_is_recorded_too(sc):
    rep = sc.evaluate(self_test=["stub-resolver"])
    assert rep["overall"] == "incomplete" and rep["selfTest"] == ["stub-resolver"]


def test_a_failure_outranks_a_self_test_incomplete(sc):
    set_facts(sc, selfTest=["stub-model"])
    (sc.det / "flush.marker").unlink()
    rep = sc.evaluate()
    assert rep["overall"] == "fail" and rep["incompleteReasons"] == []


def test_real_argv_audit_agrees_with_the_fixture_digest(sc):
    digest, hits, ran = A._run_argv_audit("pysrc2cpg", ["/src", "--language", "pythonsrc"])
    assert ran and hits == [] and digest == H.list_digest()
    assert A._run_argv_audit("kotlin2cpg", ["/src", "--jar4import-url", "x"])[1] == ["--jar4import-url"]


# ------------------------------------------------------------------ 8.1


def conn_line(ts):
    return json.dumps({"ts": ts, "srcAddr": "172.18.0.2", "srcPort": 4000, "dstAddr": "172.18.0.3", "dstPort": 443, "bytesOut": 0, "bytesIn": 1}) + "\n"


def test_8_1_a_conn_in_the_window_fails(sc):
    (sc.det / "conn.jsonl").write_text(conn_line(W.format_ts(sc.start.replace(microsecond=0) + (sc.flush - sc.start) / 2)))
    rep = sc.evaluate()
    assert row(rep, "8.1")["status"] == "fail" and "conn-in-window" in " ".join(row(rep, "8.1")["details"])
    assert rep["overall"] == "fail"


def test_8_1_a_conn_outside_the_window_fails_too(sc):
    (sc.det / "conn.jsonl").write_text(conn_line("2020-01-01T00:00:00.000Z"))
    assert "conn-outside-window:before-start" in " ".join(row(sc.evaluate(), "8.1")["details"])


def test_8_1_allow_touching_a_local_only_codebase_must_be_r1_or_r4(sc):
    def mut(recs):
        d = next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow" and r["codebases"] == ["front"])
        d["rule"] = "R7-HOSTED-SERVICE"
    edit_egress(sc, mut)
    rep = sc.evaluate()
    assert row(rep, "8.1")["status"] == "fail" and "rule R7" in " ".join(row(rep, "8.1")["details"])


def test_8_1_empty_witness_set_fails_it_does_not_pass_over_nothing(sc):
    edit_egress(sc, lambda recs: [r for r in recs if not (r["recordType"] in ("decision", "close") and r.get("codebases") != ["docs"]
                                                          and r.get("decisionAuditId") is None and r["recordType"] == "decision"
                                                          and r["decision"] == "allow")])
    r = row(sc.evaluate(), "8.1")
    assert r["status"] == "fail" and "empty" in " ".join(r["details"])


# ------------------------------------------------------------------ 8.2, 8.4, 8.12: vacuous, and not vacuous when the impossible happens


def test_8_2_vacuous_lists_private_address_allows_and_fails_on_an_r4_allow(sc):
    def mut(recs):
        d = next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow")
        d["privateAddress"] = True
    edit_egress(sc, mut)
    r = row(sc.evaluate(), "8.2")
    assert r["status"] == "vacuous" and any("privateAddress=true" in x for x in r["details"])

    def mut2(recs):
        d = next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow")
        d["rule"] = "R4-NAMED-SERVICE"
    edit_egress(sc, mut2)
    assert row(sc.evaluate(), "8.2")["status"] == "fail"


def test_8_4_a_vulndb_run_makes_the_row_fail_not_vacuous(sc):
    edit_egress(sc, lambda recs: next(r for r in recs if r["recordType"] == "policy").update(runKind="vulndb"))
    assert row(sc.evaluate(), "8.4")["status"] == "fail"


def test_8_12_an_r8_allow_makes_the_row_fail_not_vacuous(sc):
    edit_egress(sc, lambda recs: next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow").update(rule="R8-CONTROLLED-REGISTRY"))
    assert row(sc.evaluate(), "8.12")["status"] == "fail"


# ------------------------------------------------------------------ 8.3


def test_8_3_missing_payload_shape_controls_fail(sc):
    def mut(recs):
        n = 0
        for r in recs:
            if r["recordType"] == "decision" and r.get("reason") == "payload-shape" and n < 1:
                r["reason"] = "no-rule-matched"
                n += 1
    edit_egress(sc, mut)
    rep = sc.evaluate()
    assert row(rep, "8.3")["status"] == "fail" and "C1" in bad_controls(rep)


def test_8_3_an_r4_allow_with_an_extra_field_fails(sc):
    def mut(recs):
        d = next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow")
        d.update(rule="R4-NAMED-SERVICE", payloadClass="repo-names-and-identity", payloadFields=["repoNames", "identity.subject", "extra"])
    edit_egress(sc, mut)
    assert "carries more than" in " ".join(row(sc.evaluate(), "8.3")["details"])


# ------------------------------------------------------------------ 8.5, 8.6


def mutate_launched(sc, fn):
    recs = sc.sandbox()
    fn(next(r for r in recs if not r["refused"]))
    sc.write_sandbox(recs)


def test_8_5_launcher_without_network_none_fails(sc):
    mutate_launched(sc, lambda r: r.update(launcherArgv=["docker", "run", "--rm", "--read-only"]))
    assert row(sc.evaluate(), "8.5")["status"] == "fail"


def test_8_5_two_network_flags_fail_and_a_host_network_fails(sc):
    mutate_launched(sc, lambda r: r.update(launcherArgv=["docker", "run", "--network", "none", "--network", "host"]))
    assert row(sc.evaluate(), "8.5")["status"] == "fail"
    mutate_launched(sc, lambda r: r.update(launcherArgv=["docker", "run", "--network", "host"]))
    assert row(sc.evaluate(), "8.5")["status"] == "fail"


def test_8_5_namespace_backend_needs_unshare_all_without_share_net(sc):
    mutate_launched(sc, lambda r: r.update(backend="namespace", launcherArgv=["bwrap", "--unshare-all"]))
    assert row(sc.evaluate(), "8.5")["status"] == "pass"
    mutate_launched(sc, lambda r: r.update(backend="namespace", launcherArgv=["bwrap", "--unshare-all", "--share-net"]))
    assert row(sc.evaluate(), "8.5")["status"] == "fail"
    mutate_launched(sc, lambda r: r.update(backend="namespace", launcherArgv=["bwrap"]))
    assert row(sc.evaluate(), "8.5")["status"] == "fail"


def test_8_5_a_dependency_download_allow_fails(sc):
    edit_egress(sc, lambda recs: next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow").update(purpose="dependency-download"))
    assert "dependency-download" in " ".join(row(sc.evaluate(), "8.5")["details"])


def test_8_5_missing_r8_control_fails(sc):
    edit_egress(sc, lambda recs: [r.update(rule="R99-DEFAULT-DENY") for r in recs if r.get("rule") == "R8-CONTROLLED-REGISTRY"])
    rep = sc.evaluate()
    assert row(rep, "8.5")["status"] == "fail" and "C4" in bad_controls(rep)


def test_8_6_a_hit_from_the_runners_own_audit_function_fails(sc):
    rep = sc.evaluate(argv_auditor=lambda fe, argv: (H.list_digest(), ["--jar4import-url"], True))
    assert row(rep, "8.6")["status"] == "fail"


def test_8_6_the_audit_must_run_to_completion(sc):
    assert row(sc.evaluate(argv_auditor=lambda fe, argv: (None, [], False)), "8.6")["status"] == "fail"


def test_8_6_changed_flag_lists_fail_it_cannot_re_run_against_the_lists_at_listDigest(sc):
    mutate_launched(sc, lambda r: r["flagCheck"].update(listDigest="0" * 64))
    assert "listDigest differs" in " ".join(row(sc.evaluate(), "8.6")["details"])


def test_8_6_recorded_forbidden_hits_in_a_launched_record_fail(sc):
    mutate_launched(sc, lambda r: r["flagCheck"].update(forbiddenHits=["--x"]))
    assert row(sc.evaluate(), "8.6")["status"] == "fail"


@pytest.mark.parametrize("key", ["HTTPS_PROXY", "http_proxy", "GOPROXY", "NUGET_PACKAGES", "NPM_CONFIG_REGISTRY", "npm_config_registry",
                                 "PIP_INDEX_URL", "UV_INDEX", "LD_PRELOAD", "BASH_ENV", "JAVA_TOOL_OPTIONS", "WSL_INTEROP", "CCG_RUN_TOKEN",
                                 "NODE_OPTIONS", "PYTHONPATH", "NO_PROXY"])
def test_8_6_a_forbidden_env_name_fails(sc, key):
    mutate_launched(sc, lambda r: r["envKeys"].append(key))
    assert key in " ".join(row(sc.evaluate(), "8.6")["details"])


@pytest.mark.parametrize("key", ["PATH", "JAVA_HOME", "PYTHONSAFEPATH", "GOTOOLCHAIN", "DOTNET_NOLOGO", "JAVA_OPTS", "LANG"])
def test_8_6_pass_list_names_are_not_forbidden(sc, key):
    mutate_launched(sc, lambda r: r["envKeys"].append(key))
    assert row(sc.evaluate(), "8.6")["status"] == "pass"


def test_8_6_no_launched_record_fails(sc):
    sc.write_sandbox([r for r in sc.sandbox() if r["refused"]])
    assert row(sc.evaluate(), "8.6")["status"] == "fail"


def test_8_6_the_forbidden_flag_control_must_be_refused_with_exit_64(sc):
    recs = sc.sandbox()
    next(r for r in recs if r["refused"]).update(exitCode=0, refused=False, refusalReason=None)
    sc.write_sandbox(recs)
    rep = sc.evaluate()
    assert "C7" in bad_controls(rep) and row(rep, "8.13")["status"] == "fail"


# ------------------------------------------------------------------ 8.7, 8.8, 8.9


def test_8_7_interpretation_by_a_non_local_rule_fails(sc):
    def mut(recs):
        d = next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow" and r["codebases"] == ["docs"])
        d["rule"] = "R7-HOSTED-SERVICE"
    edit_egress(sc, mut)
    rep = sc.evaluate()
    assert row(rep, "8.7")["status"] == "fail" and row(rep, "8.9")["status"] == "fail"


def test_8_7_missing_hosted_control_fails(sc):
    edit_egress(sc, lambda recs: [r.update(rule="R99-DEFAULT-DENY") for r in recs if r.get("rule") == "R5-SYSTEM-LOCAL-MODEL"])
    rep = sc.evaluate()
    assert row(rep, "8.7")["status"] == "fail" and "C2" in bad_controls(rep)


def test_8_8_a_decision_with_requestMode_hosted_in_a_local_answer_run_fails(sc):
    def mut(recs):
        run = next(r for r in recs if r["recordType"] == "policy" and r["runKind"] == "answer" and "front" in r["codebases"])["runId"]
        next(r for r in recs if r["recordType"] == "decision" and r["runId"] == run and r["decision"] == "deny").update(requestMode="hosted")
    edit_egress(sc, mut)
    assert row(sc.evaluate(), "8.8")["status"] == "fail"


def test_8_8_an_allow_by_r7_in_a_local_answer_run_fails(sc):
    def mut(recs):
        run = next(r for r in recs if r["recordType"] == "policy" and r["runKind"] == "answer" and "front" in r["codebases"])["runId"]
        next(r for r in recs if r["recordType"] == "decision" and r["runId"] == run and r["decision"] == "allow").update(rule="R7-HOSTED-SERVICE")
    edit_egress(sc, mut)
    assert row(sc.evaluate(), "8.8")["status"] == "fail"


def test_8_8_no_answer_run_touching_a_local_only_repository_fails(sc):
    edit_egress(sc, lambda recs: [r.update(codebases=["docs"], modes=[m for m in r["modes"] if m["repoId"] == "docs"])
                                  for r in recs if r["recordType"] == "policy" and r["runKind"] == "answer" and "front" in r["codebases"]])
    assert "empty" in " ".join(row(sc.evaluate(), "8.8")["details"])


def test_8_9_a_model_call_allow_by_a_hosted_rule_in_a_system_local_run_fails(sc):
    def mut(recs):
        d = next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow" and r["codebases"] == ["front"])
        d["rule"] = "R7-HOSTED-SERVICE"
    edit_egress(sc, mut)
    assert row(sc.evaluate(), "8.9")["status"] == "fail"


# ------------------------------------------------------------------ 8.10, 8.11, 8.13, 8.14, 8.15, close


def test_8_10_and_8_11_need_their_override_policy_records(sc):
    edit_egress(sc, lambda recs: [r for r in recs if not (r["recordType"] == "policy" and r["override"] == "hosted")])
    rep = sc.evaluate()
    assert row(rep, "8.10")["status"] == "fail" and row(rep, "8.11")["status"] == "pass" and "C8" in bad_controls(rep)
    edit_egress(sc, lambda recs: [r for r in recs if not (r["recordType"] == "policy" and r["override"] == "local")])
    rep = sc.evaluate()
    assert row(rep, "8.11")["status"] == "fail" and "C9" in bad_controls(rep)


def test_8_10_a_refused_override_must_leave_effective_local_only(sc):
    def mut(recs):
        p = next(r for r in recs if r["recordType"] == "policy" and r["override"] == "hosted")
        for m in p["modes"]:
            m["effective"] = "hosted"
    edit_egress(sc, mut)
    assert row(sc.evaluate(), "8.10")["status"] == "fail"


def test_8_13_a_launched_record_carrying_the_flag_fails(sc):
    mutate_launched(sc, lambda r: r["toolArgv"].append("--jar4import-url"))
    assert row(sc.evaluate(), "8.13")["status"] == "fail"


def test_8_14_a_record_failing_its_schema_fails(sc):
    edit_egress(sc, lambda recs: recs[1].update(bogus=1))
    r = row(sc.evaluate(), "8.14")
    assert r["status"] == "fail" and "invalid broker record" in " ".join(r["details"])


def test_8_14_an_invalid_sandbox_record_and_an_unparseable_line_fail(sc):
    recs = sc.sandbox()
    recs[0]["refusalReason"] = "because"
    sc.write_sandbox(recs)
    assert "invalid sandbox record" in " ".join(row(sc.evaluate(), "8.14")["details"])
    with (sc.audit / "egress.jsonl").open("a") as f:
        f.write("this is not json\n")
    assert "not JSON" in " ".join(row(sc.evaluate(), "8.14")["details"])


def test_8_14_no_records_at_all_fails(sc):
    sc.write_egress([])
    sc.write_sandbox([])
    assert row(sc.evaluate(), "8.14")["status"] == "fail"


def test_8_15_an_allow_declaring_no_codebases_in_a_local_run_fails(sc):
    edit_egress(sc, lambda recs: next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow" and r["codebases"] == ["front"]).update(codebases=[]))
    assert row(sc.evaluate(), "8.15")["status"] == "fail"


def test_close_an_allow_without_a_close_is_an_interrupted_exchange_and_fails(sc):
    edit_egress(sc, lambda recs: recs.remove(next(r for r in recs if r["recordType"] == "close")))
    r = row(sc.evaluate(), "close")
    assert r["status"] == "fail" and "interrupted exchange" in " ".join(r["details"])


def test_close_two_closes_for_one_allow_fail(sc):
    def mut(recs):
        c = next(r for r in recs if r["recordType"] == "close")
        recs.append({**c, "auditId": "99999999-9999-4999-8999-999999999999"})
    edit_egress(sc, mut)
    assert "2 close" in " ".join(row(sc.evaluate(), "close")["details"])


def test_a_bad_allow_outside_the_window_is_not_a_row_input_but_p_d_fails(sc):
    def mut(recs):
        d = next(r for r in recs if r["recordType"] == "decision" and r["decision"] == "allow" and r["codebases"] == ["docs"])
        d.update(rule="R7-HOSTED-SERVICE", ts="2020-01-01T00:00:00.000Z")
    edit_egress(sc, mut)
    rep = sc.evaluate()
    assert row(rep, "8.7")["status"] == "pass" and "P-d" in bad_pre(rep)


def test_records_outside_the_window_are_not_rows_inputs_but_fail_p_d(sc):
    edit_egress(sc, lambda recs: recs[0].update(ts="2020-01-01T00:00:00.000Z"))
    rep = sc.evaluate()
    assert "P-d" in bad_pre(rep) and "starts-late" in " ".join(rep["preconditions"]["P-d"]["problems"])


# ------------------------------------------------------------------ the controls, as the audit shows them


@pytest.mark.parametrize("cid,pred", [
    ("C2", lambda r: r.get("rule") == "R5-SYSTEM-LOCAL-MODEL"),
    ("C3", lambda r: r.get("rule") == "R2-GIT-OWN-REMOTE"),
    ("C5", lambda r: r.get("reason") == "token-invalid"),
    ("C6", lambda r: r.get("reason") == "codebases-empty"),
])
def test_a_missing_control_is_reported_and_fails_the_run(sc, cid, pred):
    edit_egress(sc, lambda recs: [r for r in recs if not pred(r)])
    rep = sc.evaluate()
    assert cid in bad_controls(rep) and rep["overall"] == "fail"


def test_c2_the_hosted_control_must_be_in_a_system_local_only_run(sc):
    def mut(recs):
        run = next(r["runId"] for r in recs if r.get("rule") == "R5-SYSTEM-LOCAL-MODEL")
        next(r for r in recs if r["recordType"] == "policy" and r["runId"] == run).update(systemLocalOnly=False)
    edit_egress(sc, mut)
    assert "C2" in bad_controls(sc.evaluate())


def test_c3_must_target_a_local_endpoint_with_a_listed_path(sc):
    edit_egress(sc, lambda recs: [r.update(endpoint=H.HOSTED) for r in recs if r.get("rule") == "R2-GIT-OWN-REMOTE"])
    assert "C3" in bad_controls(sc.evaluate())


def test_c7_needs_docker_events_showing_no_container_for_the_refused_invocation(sc):
    refused = next(r for r in sc.sandbox() if r["refused"])
    (sc.dir / "docker-events.jsonl").write_text(json.dumps(
        {"Type": "container", "Action": "create", "Actor": {"ID": "x", "Attributes": {"name": f"ccg-sbx-{refused['auditId']}"}}}) + "\n")
    assert "C7" in bad_controls(sc.evaluate())
    (sc.dir / "docker-events.jsonl").write_text("")
    assert "C7" in bad_controls(sc.evaluate())
    (sc.dir / "docker-events.jsonl").unlink()
    assert "C7" in bad_controls(sc.evaluate())


# ------------------------------------------------------------------ P-a, P-b, P-c


def put_probe(sc, comp, phase, text):
    (sc.dir / "probes" / f"{phase}-{comp}.txt").write_text(text)


@pytest.mark.parametrize("kw", [{"fail": ("A1b-v4",)}, {"fail": ("A1b-v6",)}, {"drop": ("A1b-v4",)}])
def test_p_a_a_default_route_fails(sc, kw):
    put_probe(sc, "answer", "end", H.netprobe_text(**kw))
    assert bad_pre(sc.evaluate()) == ["P-a"]


def test_p_a_a_reachable_host_target_fails(sc):
    text = H.netprobe_text().replace("192.168.0.64:6379 ENETUNREACH", "192.168.0.64:6379 CONNECTED", 1).replace(
        "CHECK A1d-tcp PASS 192.168.0.64:6379", "CHECK A1d-tcp FAIL 192.168.0.64:6379")
    text = text.replace("fail=0", "fail=1")
    put_probe(sc, "front", "start", text)
    rep = sc.evaluate()
    assert bad_pre(rep) == ["P-a"] and "192.168.0.64:6379" in " ".join(rep["preconditions"]["P-a"]["problems"])


def test_p_a_a_reachable_udp_target_fails(sc):
    text = H.netprobe_text().replace("CHECK A1d-udp PASS 192.168.0.64:53 ENETUNREACH", "CHECK A1d-udp FAIL 192.168.0.64:53 SENT").replace("fail=0", "fail=1")
    put_probe(sc, "answer", "end", text)
    rep = sc.evaluate()
    assert bad_pre(rep) == ["P-a"] and "A1d-udp" in " ".join(rep["preconditions"]["P-a"]["problems"])


def test_p_a_a_probe_that_probed_too_few_targets_fails(sc):
    text = "\n".join(l for l in H.netprobe_text().splitlines() if "172.17.0.1" not in l)
    put_probe(sc, "answer", "start", text)
    assert "P-a" in bad_pre(sc.evaluate())


def test_p_a_a_probe_that_never_finished_fails_all_three(sc):
    put_probe(sc, "answer", "start", "CHECK A1b-v4 PASS x\n")
    assert bad_pre(sc.evaluate())[:3] == ["P-a", "P-b", "P-c"]  # (and P-d: the probe sent nothing the detector filed)


def test_p_a_the_positive_control_must_accept_the_store_and_the_detector_or_the_probe_is_broken(sc):
    put_probe(sc, "answer", "start", H.netprobe_text(skip_pos=True))
    assert "positive control 172.18.0.4:6379" in " ".join(sc.evaluate()["preconditions"]["P-a"]["problems"])
    put_probe(sc, "answer", "start", H.netprobe_text(skip_det=True))
    assert "positive control (detector)" in " ".join(sc.evaluate()["preconditions"]["P-a"]["problems"])
    put_probe(sc, "answer", "start", H.netprobe_text(fail=("posctl-tcp",)))
    assert "P-a" in bad_pre(sc.evaluate())


def test_p_a_each_component_and_each_phase_is_required(sc):
    (sc.dir / "probes" / "end-front.txt").unlink()
    assert bad_pre(sc.evaluate())[:3] == ["P-a", "P-b", "P-c"]
    facts = json.loads((sc.dir / "facts.json").read_text())
    facts["probes"] = [p for p in facts["probes"] if not (p["component"] == "front" and p["phase"] == "start")]
    (sc.dir / "facts.json").write_text(json.dumps(facts))
    assert "no start probe recorded" in " ".join(sc.evaluate()["preconditions"]["P-a"]["problems"])


def test_p_a_no_components_named_fails(sc):
    facts = json.loads((sc.dir / "facts.json").read_text())
    facts["components"] = []
    (sc.dir / "facts.json").write_text(json.dumps(facts))
    assert bad_pre(sc.evaluate())[:3] == ["P-a", "P-b", "P-c"]


def test_p_b_dns_resolving_fails_only_p_b(sc):
    put_probe(sc, "answer", "end", H.netprobe_text(fail=("A2-ahosts",)))
    assert bad_pre(sc.evaluate()) == ["P-b"]


@pytest.mark.parametrize("cid", ["A3-files", "A3-broker-present", "A3-named", "A3-abstract", "A3-connect", "A3-readonly", "A7"])
def test_p_c_each_host_socket_check_fails_only_p_c(sc, cid):
    put_probe(sc, "front", "start", H.netprobe_text(fail=(cid,)))
    assert bad_pre(sc.evaluate()) == ["P-c"]


# ------------------------------------------------------------------ P-d


def test_p_d_missing_marker_fails_the_precondition_and_the_row(sc):
    (sc.det / "flush.marker").unlink()
    rep = sc.evaluate()
    assert "P-d" in bad_pre(rep) and row(rep, "8.1")["status"] == "fail"


def test_p_d_probe_connections_must_match_what_the_probes_report(sc):
    lines = (sc.det / "probe.jsonl").read_text().splitlines()
    (sc.det / "probe.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    assert "P-d" in bad_pre(sc.evaluate())
    (sc.det / "probe.jsonl").write_text("\n".join(lines + lines[:1]) + "\n")  # a forged extra
    assert "P-d" in bad_pre(sc.evaluate())
    (sc.det / "probe.jsonl").unlink()
    assert "probe.jsonl does not exist" in " ".join(sc.evaluate()["preconditions"]["P-d"]["problems"])


def test_p_d_a_probe_connection_is_not_a_conn_the_zero_conn_verdict_ignores_it(sc):
    assert (sc.det / "probe.jsonl").read_text() and row(sc.evaluate(), "8.1")["status"] == "pass"


# ------------------------------------------------------------------ the detector control (C-detector)


def test_the_detector_control_must_hold(sc):
    (sc.dir / "detector-control" / "conn.jsonl").write_text("")
    assert "C-detector" in bad_pre(sc.evaluate())


# ------------------------------------------------------------------ P-e


def test_p_e_a_non_invalid_host_fails(sc):
    eg = H.egress_section()
    eg["hosted"][0]["endpoint"] = "https://api.example.com:443"
    H.write_manifest(sc.dir / "manifest.yaml", eg=eg)
    assert "not under .invalid" in " ".join(sc.evaluate()["preconditions"]["P-e"]["problems"])


def test_p_e_a_resolver_that_answers_a_scenario_name_fails(sc):
    rep = sc.evaluate(resolver=lambda name: ["93.184.216.34"])
    assert "P-e" in bad_pre(rep) and "validated an address" in " ".join(rep["preconditions"]["P-e"]["problems"])


def test_p_e_the_positive_control_catches_a_check_that_never_flags(sc, monkeypatch):
    from ccg.broker import addr
    monkeypatch.setattr(addr, "validate", lambda *a, **k: (_ for _ in ()).throw(addr.AddressDenied("resolve-failed")))
    assert "positive control failed" in " ".join(sc.evaluate()["preconditions"]["P-e"]["problems"])


def test_p_e_a_different_denial_reason_is_not_resolve_failed(sc):
    from ccg.broker import addr
    rep = sc.evaluate(iface=lambda: (_ for _ in ()).throw(addr.IfaceUnavailable("x")), resolver=lambda n: ["93.184.216.34"])
    assert "address-check-unavailable" in " ".join(rep["preconditions"]["P-e"]["problems"])


def test_p_e_needs_a_hosted_model_and_an_idp_entry(sc):
    eg = H.egress_section()
    eg["hosted"] = []
    H.write_manifest(sc.dir / "manifest.yaml", eg=eg)
    assert "no hosted model entry" in " ".join(sc.evaluate()["preconditions"]["P-e"]["problems"])
    eg = H.egress_section()
    eg["outbound"] = []
    H.write_manifest(sc.dir / "manifest.yaml", eg=eg)
    assert "no outbound idp entry" in " ".join(sc.evaluate()["preconditions"]["P-e"]["problems"])


def test_p_e_the_manifest_must_match_every_policy_records_digest(sc):
    edit_egress(sc, lambda recs: next(r for r in recs if r["recordType"] == "policy").update(allowListDigest="0" * 64))
    assert "allowListDigest differs" in " ".join(sc.evaluate()["preconditions"]["P-e"]["problems"])


# ------------------------------------------------------------------ P-f, P-g


def test_p_f_a_system_without_a_local_only_or_without_a_hosted_repository_fails(sc):
    H.write_manifest(sc.dir / "manifest.yaml", repos=(("front", "local-only"), ("docs", "local-only")))
    assert "no hosted-default repository" in " ".join(sc.evaluate()["preconditions"]["P-f"]["problems"])
    H.write_manifest(sc.dir / "manifest.yaml", repos=(("front", "hosted"), ("docs", "hosted")))
    assert "no local-only repository" in " ".join(sc.evaluate()["preconditions"]["P-f"]["problems"])


def test_p_g_no_interpret_allow_fails(sc):
    edit_egress(sc, lambda recs: [r.update(decision="deny", rule="R99-DEFAULT-DENY", reason="no-rule-matched", resolvedAddr=None, payloadFields=None)
                                  for r in recs if r["recordType"] == "decision" and r["decision"] == "allow"])
    assert "no R1 allow from answer.interpret" in " ".join(sc.evaluate()["preconditions"]["P-g"]["problems"])


def test_p_g_summary_enabled_needs_a_summarize_allow(sc):
    facts = json.loads((sc.dir / "facts.json").read_text())
    facts["summaryEnabled"] = True
    (sc.dir / "facts.json").write_text(json.dumps(facts))
    assert "answer.summarize" in " ".join(sc.evaluate()["preconditions"]["P-g"]["problems"])


def test_p_g_every_repository_needs_a_non_refused_sandbox_record(sc):
    sc.write_sandbox([r for r in sc.sandbox() if r["codebases"] != ["docs"]])
    assert "docs" in " ".join(sc.evaluate()["preconditions"]["P-g"]["problems"])
    recs = sc.sandbox()
    for r in recs:
        r.update(refused=True, refusalReason="usage", exitCode=64, launcherArgv=[])
    sc.write_sandbox(recs)
    assert "no non-refused sandbox record for repository front" in " ".join(sc.evaluate()["preconditions"]["P-g"]["problems"])


# ------------------------------------------------------------------ the §7.3 escape probes


@pytest.mark.parametrize("kw", [{"fail": ("A1c",)}, {"drop": ("A7",)}, {"drop": ("A4a-memfd",)}, {"checks": 20}, {"drop": ("A1e-udp",)}, {"fail": ("A1d-tcp",)}])
def test_escape_probe_problems_fail_e_7_3(sc, kw):
    (sc.dir / "probes" / "escape-container.txt").write_text(H.escape_text(**kw))
    assert "E-7.3" in bad_pre(sc.evaluate())


def test_escape_probe_missing_or_unfinished_fails(sc):
    (sc.dir / "probes" / "escape-container.txt").write_text("CHECK A1a PASS\n")
    assert "did not run to completion" in " ".join(sc.evaluate()["preconditions"]["E-7.3"]["problems"])
    (sc.dir / "probes" / "escape-container.txt").unlink()
    assert "missing" in " ".join(sc.evaluate()["preconditions"]["E-7.3"]["problems"])


# ------------------------------------------------------------------ row 8.16 and the trace parser


CONNECT_V4 = '123 100.500000 connect(7, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("93.184.216.34")}, 16) = -1 EINPROGRESS (x)'
CONNECT_V6 = '123 100.500000 connect(7, {sa_family=AF_INET6, sin6_port=htons(443), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2606:4700::1111", &sin6_addr), sin6_scope_id=0}, 28) = 0'


def test_parse_trace_families_and_unfinished_lines():
    ev = A.parse_trace("\n".join([
        CONNECT_V4, CONNECT_V6,
        '124 100.6 connect(3, {sa_family=AF_UNIX, sun_path="/x"}, 110) = 0',
        '124 100.7 sendto(5, "x", 1, 0, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("10.0.0.1")}, 16) = 1',
        '[pid 125] 100.8 connect(8, {sa_family=AF_INET, sin_port=htons(80), sin_addr=inet_addr("1.2.3.4")}, 16 <unfinished ...>',
        '125 100.9 <... connect resumed>) = -1 EINPROGRESS',
        '125 101.0 sendto(5, "x", 1, 0, NULL, 0) = 1',
        '125 101.1 sendmsg(5, {msg_name={sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("8.8.8.8")}, msg_namelen=16}, 0) = 1',
        "garbage"]))
    assert [(e["call"], e["family"], e["addr"], e["port"]) for e in ev] == [
        ("connect", "AF_INET", "93.184.216.34", 443), ("connect", "AF_INET6", "2606:4700::1111", 443), ("connect", "AF_UNIX", None, None),
        ("sendto", "AF_INET", "10.0.0.1", 53), ("connect", "AF_INET", "1.2.3.4", 80), ("sendmsg", "AF_INET", "8.8.8.8", 53)]
    assert ev[4]["pid"] == "125"


def test_check_trace_matches_allows_and_flags_the_rest():
    ev = A.parse_trace(CONNECT_V4 + "\n" + CONNECT_V6)
    assert A.check_trace(ev, [("93.184.216.34", 443, 100.0)], 0, 1000)[0].startswith("a connection to 2606:4700::1111:443")
    assert A.check_trace(ev, [("93.184.216.34", 443, 100.0), ("2606:4700:0::1111", 443, 100.0)], 0, 1000) == []
    assert "matches no unused allow" in A.check_trace(A.parse_trace(CONNECT_V4), [("93.184.216.34", 8443, 100.0)], 0, 1000)[0]  # right address, wrong port
    assert len(A.check_trace(ev, [], 0, 1000)) == 2


def test_check_trace_a_connect_before_its_allow_was_decided_is_a_violation_and_the_window_bounds_apply():
    ev = A.parse_trace(CONNECT_V4)
    assert A.check_trace(ev, [("93.184.216.34", 443, 101.0)], 0, 1000)
    assert A.check_trace(ev, [], 200, 300) == []  # outside the window
    assert A.check_trace(ev, [], 0, 50) == []


def test_check_trace_resolver_traffic_is_flagged_as_such_and_other_families_are_not_ignored():
    ev = A.parse_trace('1 5.0 sendto(5, "x", 1, 0, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("10.0.0.1")}, 16) = 1\n'
                       '1 6.0 connect(5, {sa_family=AF_VSOCK, svm_cid=2, svm_port=1}, 16) = -1 EACCES\n'
                       '1 7.0 sendto(5, "x", 1, 0, {sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, 12) = 1\n')
    out = A.check_trace(ev, [], 0, 100)
    assert out[0].startswith("resolver traffic to 10.0.0.1:53") and "unexpected or unparsable family AF_VSOCK" in out[1] and len(out) == 2


def test_8_16_flags_a_connection_that_matches_no_allow(sc):
    with (sc.dir / "trace" / "broker.strace").open("a") as f:
        f.write(f'100 {sc.start.timestamp() + 1:.6f} connect(9, {{sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("93.184.216.34")}}, 16) = 0\n')
    r = row(sc.evaluate(), "8.16")
    assert r["status"] == "fail" and "93.184.216.34:443" in " ".join(r["details"])


def test_8_16_the_positive_control_must_flag_the_unlisted_dial(sc):
    (sc.dir / "trace" / "control.strace").write_text('200 5.0 connect(3, {sa_family=AF_UNIX, sun_path="/x"}, 110) = 0\n')
    assert "positive control failed" in " ".join(row(sc.evaluate(), "8.16")["details"])
    (sc.dir / "trace" / "control.strace").write_text(
        '200 5.0 connect(3, {sa_family=AF_INET, sin_port=htons(1), sin_addr=inet_addr("127.0.0.1")}, 16) = -1 ECONNREFUSED (x)\n')
    assert "positive control failed" in " ".join(row(sc.evaluate(), "8.16")["details"])  # flagged, but not the dial it names


def test_8_16_missing_traces_fail_when_strace_is_declared_provisioned(sc):
    (sc.dir / "trace" / "control.strace").unlink()
    assert row(sc.evaluate(), "8.16")["status"] == "fail"
    (sc.dir / "trace" / "broker.strace").unlink()
    assert "missing" in " ".join(row(sc.evaluate(), "8.16")["details"])


def test_8_16_without_a_window_fails(sc):
    (sc.det / "start.marker").unlink()
    assert row(sc.evaluate(), "8.16")["status"] == "fail"


def test_8_16_an_allow_decision_must_precede_its_connect(sc):
    # move every traced connect to well before the allows: the broker connected before it decided
    p = sc.dir / "trace" / "broker.strace"
    p.write_text("".join(l.replace(l.split()[1], "1.000000") if "AF_INET" in l else l for l in p.read_text().splitlines(True)))
    r = row(sc.evaluate(), "8.16")  # outside the window: not checked (the row says "in W"), but then the trace shows no call for the allows
    assert r["status"] == "fail" and "0 IP call(s) in the window" in " ".join(r["details"])


# ================================================================== analyst F1-F9 / security M1-M6 coverage

# -- the escape-probe required ids: every one of the helper's own literal list, dropped one at a time (F5 i)


def test_the_required_escape_ids_are_the_specs_set():
    assert tuple(A.ESCAPE_REQUIRED) == H.ESCAPE_IDS


@pytest.mark.parametrize("cid", H.ESCAPE_IDS)
def test_dropping_any_one_required_escape_check_fails_e_7_3(sc, cid):
    (sc.dir / "probes" / "escape-container.txt").write_text(H.escape_text(drop=(cid,)))
    rep = sc.evaluate()
    assert "E-7.3" in bad_pre(rep) and any(p.startswith(cid + ":") for p in rep["preconditions"]["E-7.3"]["problems"])


@pytest.mark.parametrize("cid", H.ESCAPE_IDS)
def test_one_failing_required_escape_check_fails_e_7_3(sc, cid):
    (sc.dir / "probes" / "escape-container.txt").write_text(H.escape_text(fail=(cid,)))
    assert "E-7.3" in bad_pre(sc.evaluate())


# -- empty witness sets, one test per witness-bearing row (F5 ii), and the rows 8.5/8.6 over zero launched records (F7)


def drop_interpret_allows(sc):
    ids = {d["auditId"] for d in sc.egress() if d["recordType"] == "decision" and d["decision"] == "allow" and d["component"] == "answer.interpret"}
    assert ids
    sc.write_egress([r for r in sc.egress() if r["auditId"] not in ids and r.get("decisionAuditId") not in ids])


@pytest.mark.parametrize("rid_", ["8.1", "8.7", "8.9", "8.15"])
def test_a_witness_bearing_row_over_an_empty_witness_set_fails(sc, rid_):
    drop_interpret_allows(sc)
    r = row(sc.evaluate(), rid_)
    assert r["status"] == "fail" and "the witness set is empty: the check ran over nothing" in r["details"]


def test_8_8_over_no_answer_run_touching_a_local_only_repository_fails_with_the_empty_witness_message(sc):
    pol = [p for p in sc.egress() if p["recordType"] == "policy" and p.get("runKind") == "answer"
           and any(m["effective"] == "local-only" and m["repoId"] == "front" for m in p["modes"])]
    assert pol
    ids = {p["runId"] for p in pol}
    sc.write_egress([r for r in sc.egress() if r.get("runId") not in ids])
    r = row(sc.evaluate(), "8.8")
    assert r["status"] == "fail" and "the witness set is empty: the check ran over nothing" in r["details"]


@pytest.mark.parametrize("rid_", ["8.5", "8.6"])
def test_8_5_and_8_6_over_zero_launched_sandbox_records_fail(sc, rid_):
    sc.write_sandbox([r for r in sc.sandbox() if r["refused"] is True])
    r = row(sc.evaluate(), rid_)
    assert r["status"] == "fail" and "no non-refused sandbox record in the window" in " ".join(r["details"])


# -- the P-a target floor: shorten only the TARGETS line (F5 iii)


def shorten_targets(sc, which):
    for f in (sc.dir / "probes").glob("*-answer.txt"):
        import re
        t = f.read_text()
        m = re.search(r"TARGETS tcp=(\d+) udp=(\d+)", t)
        tcp, udp = int(m.group(1)), int(m.group(2))
        new = f"TARGETS tcp={tcp - 1 if which == 'tcp' else tcp} udp={udp - 1 if which == 'udp' else udp}"
        f.write_text(t.replace(m.group(0), new))


@pytest.mark.parametrize("which", ["tcp", "udp"])
def test_p_a_a_targets_line_one_short_of_the_host_snapshot_fails_p_a_alone(sc, which):
    shorten_targets(sc, which)
    rep = sc.evaluate()
    assert bad_pre(rep) == ["P-a"] and any("fewer than the host snapshot needs" in p for p in rep["preconditions"]["P-a"]["problems"])


def test_p_a_the_done_line_must_agree_with_the_check_lines(sc):
    f = sc.dir / "probes" / "start-answer.txt"
    t = f.read_text().replace("CHECK A1d-tcp PASS 192.168.0.64:53 ENETUNREACH", "CHECK A1d-tcp FAIL 192.168.0.64:53 ENETUNREACH")
    assert "FAIL 192.168.0.64:53" in t
    f.write_text(t)   # the DONE line still says fail=0
    assert any("fail count disagrees" in p for p in sc.evaluate()["preconditions"]["P-a"]["problems"])


# -- P-b: the two positive legs (F4)


@pytest.mark.parametrize("cid", ["A2-localhost", "A2-embedded"])
def test_p_b_the_getent_positive_legs_are_required_and_must_pass(sc, cid):
    for kw in ({"drop": (cid,)}, {"fail": (cid,)}):
        for f in (sc.dir / "probes").glob("start-answer.txt"):
            f.write_text(H.netprobe_text(**kw))
        rep = sc.evaluate()
        assert bad_pre(rep) == ["P-b"] and any(p.endswith(f"{cid}: " + ("not run" if "drop" in kw else "[('FAIL', 'x')]"))
                                               for p in rep["preconditions"]["P-b"]["problems"])


# -- window membership at the marker instants, and the sandbox half of window coverage (F5 iv)


def test_in_window_is_inclusive_at_both_marker_instants_and_false_just_outside(sc):
    s = A.load(sc.dir)
    f = lambda t: s.in_window({"ts": W.format_ts(t)})  # noqa: E731
    from datetime import timedelta
    assert f(s.start) and f(s.flush)
    assert not f(s.start - timedelta(milliseconds=1)) and not f(s.flush + timedelta(milliseconds=1))
    assert not s.in_window({"ts": "garbage"}) and not s.in_window({})


def test_a_row_input_exactly_on_the_start_marker_is_a_row_input(sc):
    allow = next(d for d in sc.egress() if d["recordType"] == "decision" and d["decision"] == "allow" and d["component"] == "answer.interpret")
    recs = sc.egress()
    for r in recs:
        if r["auditId"] == allow["auditId"]:
            r["rule"] = "R3-VULNDB"   # an allow that "cannot exist" (row 8.2 fails on it, only if it is in the window)
    sc.write_egress(recs)
    # the window opens at the instant of the earliest record, so every record is at or after the start marker; the allow is on/after it
    first = min(W.parse_ts(r["ts"]) for r in recs + sc.sandbox())
    sc.write_markers(sc.det, first, sc.flush)
    assert row(sc.evaluate(), "8.2")["status"] == "fail"


def test_the_sandbox_records_bound_the_window_too(sc):
    from datetime import timedelta
    recs = sc.sandbox()
    late = W.format_ts(sc.flush + timedelta(seconds=30))
    recs[0]["ts"] = late
    sc.write_sandbox(recs)
    assert any(p.startswith("window-ends-early") for p in sc.evaluate()["preconditions"]["P-d"]["problems"])
    recs[0]["ts"] = W.format_ts(sc.start - timedelta(seconds=30))
    sc.write_sandbox(recs)
    assert any(p.startswith("window-starts-late") for p in sc.evaluate()["preconditions"]["P-d"]["problems"])


# -- P-d: probe.jsonl


def test_p_d_a_malformed_probe_line_fails(sc):
    with (sc.det / "probe.jsonl").open("a") as f:
        f.write("not json\n")
    assert any("probe.jsonl:" in p and "not JSON" in p for p in sc.evaluate()["preconditions"]["P-d"]["problems"])


def test_p_d_a_probe_record_without_a_label_is_malformed(sc):
    recs = [json.loads(x) for x in (sc.det / "probe.jsonl").read_text().splitlines()]
    del recs[0]["probe"]
    (sc.det / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert "P-d" in bad_pre(sc.evaluate())


def test_c5_needs_the_token_less_request_to_have_no_run(sc):
    recs = sc.egress()
    pol = next(r for r in recs if r["recordType"] == "policy")
    for r in recs:
        if r.get("rule") == "R0-INVALID" and r.get("reason") == "token-invalid":
            r["runId"] = pol["runId"]
    sc.write_egress(recs)
    assert bad_controls(sc.evaluate()) == ["C5"]


# -- P-d: every probe record is attributed to one (component, phase) (M4)


def probe_recs(sc):
    return [json.loads(x) for x in (sc.det / "probe.jsonl").read_text().splitlines()]


def write_probe_recs(sc, recs):
    (sc.det / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def p_d(rep):
    return " | ".join(rep["preconditions"]["P-d"]["problems"])


def test_p_d_the_good_scenario_attributes_twelve_probe_records_to_four_probes(sc):
    recs = probe_recs(sc)
    assert len(recs) == 12 and {r["probe"] for r in recs} == {"answer/start", "answer/end", "front/start", "front/end"}
    assert sc.evaluate()["preconditions"]["P-d"]["ok"]


def test_p_d_a_forged_extra_record_under_a_known_label_fails_even_when_the_total_is_compensated(sc):
    recs = probe_recs(sc)
    # swap one real record for one more from the other component (the global count is unchanged: the old equality passes)
    recs[0] = {**recs[0], "probe": "front/end", "srcAddr": "172.18.0.11"}
    write_probe_recs(sc, recs)
    msg = p_d(sc.evaluate())
    assert "ports" in msg and "answer/start" in msg


def test_p_d_a_record_with_an_unknown_label_fails(sc):
    recs = probe_recs(sc)
    recs[0]["probe"] = "intruder/start"
    write_probe_recs(sc, recs)
    assert "matches no expected (component, phase) probe" in p_d(sc.evaluate())


def test_p_d_a_record_from_the_wrong_source_address_fails(sc):
    recs = probe_recs(sc)
    recs[0]["srcAddr"] = "172.18.0.99"
    write_probe_recs(sc, recs)
    assert "is not the component's address" in p_d(sc.evaluate())


@pytest.mark.parametrize("delta", [-60, 60])
def test_p_d_a_record_outside_its_probes_own_interval_fails(sc, delta):
    from datetime import timedelta
    recs = probe_recs(sc)
    recs[0]["ts"] = W.format_ts(W.parse_ts(recs[0]["ts"]) + timedelta(seconds=delta))
    write_probe_recs(sc, recs)
    assert "outside that probe's own interval" in p_d(sc.evaluate())


def test_p_d_a_record_at_the_edge_of_the_slack_is_accepted(sc):
    from datetime import timedelta
    probes = json.loads((sc.dir / "facts.json").read_text())["probes"]
    e = next(p for p in probes if p["component"] == "answer" and p["phase"] == "start")
    recs = probe_recs(sc)
    recs[0]["ts"] = W.format_ts(W.parse_ts(e["t1"]) + A.PROBE_SLACK)
    write_probe_recs(sc, recs)
    assert sc.evaluate()["preconditions"]["P-d"]["ok"]


def test_p_d_a_missing_record_for_a_target_fails(sc):
    recs = probe_recs(sc)
    write_probe_recs(sc, [r for r in recs if not (r["probe"] == "front/end" and r["dstPort"] == 443)])
    assert "front/end" in p_d(sc.evaluate())


def test_p_d_the_same_port_twice_instead_of_each_target_once_fails(sc):
    recs = probe_recs(sc)
    for r in recs:
        if r["probe"] == "front/end" and r["dstPort"] == 443:
            r["dstPort"] = 80
    write_probe_recs(sc, recs)
    assert "expected exactly [80, 443, 3128]" in p_d(sc.evaluate())


@pytest.mark.parametrize("key", ["ip", "t0", "t1"])
def test_p_d_facts_must_state_each_probes_address_and_interval(sc, key):
    f = json.loads((sc.dir / "facts.json").read_text())
    del f["probes"][0][key]
    (sc.dir / "facts.json").write_text(json.dumps(f))
    assert "facts.json lacks the probe's ip, t0/t1" in p_d(sc.evaluate())


# -- P-e before the run, and only for hosts that passed the .invalid check (F2, security m1)


def test_pre_e_manifest_alone_passes_with_a_resolver_that_answers_nothing(sc):
    assert A.pre_e_manifest(sc.dir / "manifest.yaml", H._nxdomain, H._no_iface) == []


def test_pre_e_manifest_alone_fails_on_a_resolver_that_answers_a_scenario_name(sc):
    probs = A.pre_e_manifest(sc.dir / "manifest.yaml", lambda name: ["93.184.216.34"], H._no_iface)
    assert probs and all("validated an address" in p or "not resolve-failed" in p for p in probs)


def test_pre_e_never_resolves_a_host_that_is_not_under_invalid(sc):
    eg = H.egress_section()
    eg["outbound"][0]["endpoint"] = "https://idp.example.com:443"
    H.write_manifest(sc.dir / "manifest.yaml", eg=eg)
    asked = []

    def spy(name):
        asked.append(name)
        raise OSError("NXDOMAIN")
    probs = A.pre_e_manifest(sc.dir / "manifest.yaml", spy, H._no_iface)
    assert any("idp.example.com" in p and "not under .invalid" in p for p in probs)
    assert asked and not any("example.com" in a for a in asked) and any("ac8.invalid" in a for a in asked)


def test_pre_e_cli_exit_codes_without_any_dns(sc):
    ok = run_cli("pre-e", sc.dir / "manifest.yaml")          # the patched default resolver answers nothing
    assert ok.returncode == 0 and "P-e (resolver: real): ok" in ok.stdout
    stub = run_cli("pre-e", sc.dir / "manifest.yaml", "--stub-resolver")
    assert stub.returncode == 0 and "stub, no DNS query made" in stub.stdout
    eg = H.egress_section()
    eg["hosted"][0]["endpoint"] = "https://api.example.com:443"
    H.write_manifest(sc.dir / "manifest.yaml", eg=eg)
    bad = run_cli("pre-e", sc.dir / "manifest.yaml", "--stub-resolver")
    assert bad.returncode == 1 and "FAIL P-e" in bad.stdout
    assert run_cli("pre-e").returncode == 64


# -- row 8.16 (security m2): the trace set, a non-empty trace, one-to-one allow matching, unparsable connects


def test_parse_trace_sendmmsg_socket_and_an_unparsable_connect():
    ev = A.parse_trace("\n".join([
        '1 5.0 sendmmsg(5, [{msg_hdr={msg_name={sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("10.0.0.1")}, msg_namelen=16}}], 1, 0) = 1',
        "1 5.1 socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL)) = 3",
        "1 5.2 socket(AF_INET, SOCK_STREAM|SOCK_CLOEXEC, IPPROTO_IP) = 4",
        "1 5.3 connect(4, 0x7ffc0, 16) = -1 EFAULT (Bad address)",
        '1 5.4 sendmmsg(5, [{msg_hdr={msg_name=NULL, msg_namelen=0}}], 1, 0) = 1']))
    assert [(e["call"], e["family"], e["addr"]) for e in ev] == [
        ("sendmmsg", "AF_INET", "10.0.0.1"), ("socket", "AF_PACKET", None), ("socket", "AF_INET", None), ("connect", "UNPARSED", None)]


def test_check_trace_socket_domains_and_unparsable_connects_are_violations():
    ev = A.parse_trace("1 5.1 socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL)) = 3\n1 5.2 socket(AF_INET6, SOCK_DGRAM, 0) = 4\n"
                       "1 5.25 socket(AF_UNIX, SOCK_STREAM, 0) = 5\n1 5.3 connect(4, 0x7ffc0, 16) = -1 EFAULT (x)\n")
    out = A.check_trace(ev, [], 0, 100)
    assert len(out) == 2 and "AF_PACKET" in out[0] and "unparsable family UNPARSED" in out[1]


def test_check_trace_one_allow_legitimises_one_connect_only():
    two = A.parse_trace(CONNECT_V4 + "\n" + CONNECT_V4.replace("100.500000", "100.600000"))
    assert A.check_trace(two, [("93.184.216.34", 443, 100.0)], 0, 1000) == [
        "a connection to 93.184.216.34:443 matches no unused allow decided before it (connect)"]
    assert A.check_trace(two, [("93.184.216.34", 443, 100.0), ("93.184.216.34", 443, 100.0)], 0, 1000) == []


def test_8_16_a_trace_with_no_network_call_at_all_fails(sc):
    (sc.dir / "trace" / "broker.strace").write_text("")
    r = row(sc.evaluate(), "8.16")
    assert r["status"] == "fail" and "captured nothing" in " ".join(r["details"])


def test_8_16_a_trace_with_fewer_ip_calls_than_allows_fails(sc):
    p = sc.dir / "trace" / "broker.strace"
    lines = p.read_text().splitlines(True)
    ip = [i for i, l in enumerate(lines) if "AF_INET" in l]
    del lines[ip[0]]
    p.write_text("".join(lines))
    r = row(sc.evaluate(), "8.16")
    assert r["status"] == "fail" and "every allow implies a connect" in " ".join(r["details"])


def test_8_16_a_second_connect_to_an_allowed_address_is_a_violation(sc):
    p = sc.dir / "trace" / "broker.strace"
    first = next(l for l in p.read_text().splitlines() if "AF_INET" in l)
    with p.open("a") as f:
        f.write(first.replace(first.split()[1], f"{sc.start.timestamp() + 2:.6f}") + "\n")
    r = row(sc.evaluate(), "8.16")
    assert r["status"] == "fail" and "matches no unused allow" in " ".join(r["details"])


@pytest.mark.parametrize("final_residue,code", [(None, "marker-missing:final"), (2, "final-residue: 2 connection(s)")])
def test_p_d_a_missing_or_nonzero_final_marker_fails_p_d_and_row_8_1(sc, final_residue, code):
    sc.write_markers(sc.det, sc.start, sc.flush, final_residue=final_residue)
    rep = sc.evaluate()
    assert any(p.startswith(code) for p in rep["preconditions"]["P-d"]["problems"])
    r = row(rep, "8.1")
    assert r["status"] == "fail" and any(d.startswith(code) for d in r["details"])
    assert rep["overall"] == "fail"


def test_p_d_a_nonzero_flush_residue_in_a_scenario_fails_p_d_and_row_8_1(sc):
    sc.write_markers(sc.det, sc.start, sc.flush, residue=3)
    rep = sc.evaluate()
    assert any(p.startswith("flush-residue: 3 connection(s)") for p in rep["preconditions"]["P-d"]["problems"])
    r = row(rep, "8.1")
    assert r["status"] == "fail" and any(d.startswith("flush-residue: 3") for d in r["details"])
    assert rep["overall"] == "fail"


def test_p_d_a_probe_that_reports_sending_a_different_count_than_the_detector_filed_fails(sc):
    f = sc.dir / "probes" / "start-answer.txt"
    f.write_text(f.read_text().replace("PROBE-SENT det=3", "PROBE-SENT det=2"))
    assert "the detector filed 12 probe connection(s) but the probes report sending 11" in p_d(sc.evaluate())
