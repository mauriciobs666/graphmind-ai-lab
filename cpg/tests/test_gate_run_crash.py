"""P1-32b round C: the marker state machine under REAL ``SIGKILL`` (``os.fork``; the child kills itself at the planted point, the parent reads what is
on disk and runs the next command), and the per-anchor / pin / lock / build-isolation behaviour added with it. Offline and synthetic; nothing here is
held-out data. A ``BaseException`` stand-in cannot see a window between two syscalls; a killed process can."""
import json
import os
import signal
from pathlib import Path

import pytest

import gate_helpers as H
from gate_helpers import G
from gate_run_helpers import Env, many_world, GATE, OBS

import gate_run as R


# ------------------------------------------------------------------------------------------------ the fork harness
def in_child(fn) -> int:
    """Run ``fn`` in a forked child; -> the wait status. A normal end is exit 0, an exception exit 70, a planted kill a SIGKILL signal."""
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            fn()
        except BaseException:                                   # noqa: BLE001 - the child must never fall back into pytest
            code = 70
        os._exit(code)
    _, status = os.waitpid(pid, 0)
    return status


def was_sigkilled(status: int) -> bool:
    return os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL


def suicide():
    os.kill(os.getpid(), signal.SIGKILL)


def kill_at_link(*, after: bool):
    """Patch ``os.link`` so that the child dies right before (or right after) it links a marker into place."""
    real = os.link

    def link(src, dst, *a, **k):
        if str(dst).endswith(".marker") and not after:
            suicide()
        out = real(src, dst, *a, **k)
        if str(dst).endswith(".marker") and after:
            suicide()
        return out
    os.link = link


def kill_after_rename_of(suffix: str):
    """Patch ``os.replace`` so that the child dies right AFTER the rename that puts a file ending ``suffix`` in place."""
    real = os.replace

    def replace(src, dst, *a, **k):
        out = real(src, dst, *a, **k)
        if str(dst).endswith(suffix):
            suicide()
        return out
    os.replace = replace


def score(e, monkeypatch, capsys):
    """``gate.py score`` over the Env's custodian, exactly as the CLI would be called; -> (exit status, stdout JSON or None)."""
    monkeypatch.setenv("HOME", str(e.home))
    argv = ["score", "--gate-id", GATE, "--lists", str(e.lists), "--run", str(e.run_files()[0]), "--custodian", str(e.cust), "--port", "16420"]
    rc = G.main(argv, snapshot=e.snap, repo=e.repo, preflight_fn=lambda repo, gid: G.preflight(repo, gid, observed=OBS))
    cap = capsys.readouterr()
    return rc, (json.loads(cap.out) if cap.out.strip() else None), cap.err


# ------------------------------------------------------------------------------------------------ N1 (a): killed after run.json is renamed in
def test_n1a_a_kill_after_the_run_json_rename_leaves_a_run_that_the_resume_finalises_without_asking_and_score_reads(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 2)
    status = in_child(lambda: (kill_after_rename_of("run-" + G.list_key([json.loads(l) for l in e.lists.read_text().splitlines()]) + ".json"), e.go()))
    assert was_sigkilled(status)
    assert e.marker()["state"] == "started" and len(e.run_files()) == 1                  # the plant: a complete run, a marker that never became `asked`
    e.asked.clear()
    out = e.go(resume=True)
    assert e.asked == [] and e.marker()["state"] == "asked" and out["asked"] == 2        # finalised, nothing asked
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rep is not None and rep["scored"] == 2 and rc in (0, 1), err


def test_n1a_a_run_json_of_another_list_or_snapshot_is_not_a_finalisation_and_the_resume_is_refused(tmp_path):
    e = Env(tmp_path)
    many_world(e, 2)
    e.go()
    key = e.marker()["listSha256"]
    for field, value in (("listKey", "b" * 64), ("snapshotId", "c" * 32)):
        doc = e.marker()
        (e.cust / f"gate-run-{key}.marker").write_text(json.dumps({**doc, "state": "started"}) + "\n")
        run = e.run_files()[0]
        good = run.read_text()
        run.write_text(json.dumps({**json.loads(good), field: value}) + "\n")
        with pytest.raises(G.Refusal):
            e.go(resume=True)
        assert e.marker()["state"] == "started"                                          # not finalised on a mismatching file
        run.write_text(good)


def test_n1a_a_run_json_that_does_not_parse_is_not_a_finalisation(tmp_path):
    e = Env(tmp_path)
    e.go()
    key = e.marker()["listSha256"]
    (e.cust / f"gate-run-{key}.marker").write_text(json.dumps({**e.marker(), "state": "started"}) + "\n")
    e.run_files()[0].write_text("{torn")
    with pytest.raises(G.Refusal):
        e.go(resume=True)
    assert e.marker()["state"] == "started"


# ------------------------------------------------------------------------------------------------ N1 (b): the marker is never empty or torn
def test_n1b_a_kill_between_the_marker_create_and_its_payload_never_leaves_an_empty_marker_and_the_next_run_succeeds(tmp_path):
    e = Env(tmp_path)
    status = in_child(lambda: (kill_at_link(after=False), e.go()))
    assert was_sigkilled(status)
    assert [m.read_bytes() for m in e.markers()] == []                                    # no marker at all: nothing started, nothing spent
    out = e.go()
    assert out["asked"] == 1 and e.marker()["state"] == "asked"


def test_n1b_a_kill_right_after_the_marker_is_linked_leaves_a_complete_started_marker_and_the_resume_finishes(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    many_world(e, 2)
    status = in_child(lambda: (kill_at_link(after=True), e.go()))
    assert was_sigkilled(status)
    assert e.marker()["state"] == "started" and e.marker()["pins"]["snapshotId"] == H.SNAP and not list(e.cust.glob("run-*"))
    e.go(resume=True)
    assert len(e.asked) == 2 and e.marker()["state"] == "asked"
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rep is not None, err


def test_n1b_an_existing_marker_still_refuses_and_is_left_byte_for_byte(tmp_path):
    e = Env(tmp_path)
    e.go()
    (m,) = e.markers()
    before = m.read_bytes()
    with pytest.raises(G.Refusal, match="already run"):
        G.first_run_marker(e.cust, m.name[len("gate-run-"):-len(".marker")], {"state": "started", "pins": {}})
    assert m.read_bytes() == before and [p.name for p in e.cust.glob(".gate-run-*")] == []


def test_n1_no_code_path_removes_a_marker():
    src = "".join((Path(G.__file__).parent / n).read_text() for n in ("gate.py", "gate_run.py"))
    import re
    assert not re.search(r"\.unlink\([^)]*marker|os\.(remove|unlink)\([^)]*marker|rmtree\([^)]*marker", src)


# ================================================================================================ per-anchor recording (Q1) and the engine-error policy
from clients.common import AnchorNotResolved, BadRequest, NoCredentials, StoreAuthError, StoreError  # noqa: E402
from gate_run_helpers import FakeClient  # noqa: E402


def raising_at(k, exc):
    def hook(req, n):
        if n == k:
            raise exc
    return hook


def test_a_harness_error_on_one_anchor_is_recorded_the_loop_continues_the_run_json_is_written_the_marker_is_asked_and_score_gives_blocked(
        tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    recs = many_world(e, 3)
    e.ask_hook = raising_at(2, G.HarnessError("a path /home/x and an id leaked in the message"))
    out = e.go()
    assert len(e.asked) == 3 and out["asked"] == 3 and e.marker()["state"] == "asked"
    run = json.loads(e.run_files()[0].read_text())
    faulty = [a for a, v in run["answers"].items() if "harnessError" in v]
    assert len(faulty) == 1 and run["answers"][faulty[0]] == {"harnessError": "HarnessError"}
    lines = [json.loads(l) for l in e.answers_file().read_text().splitlines()]
    assert {"anchorId": faulty[0], "harnessError": "HarnessError"} in lines
    assert "leaked" not in e.answers_file().read_text() and "leaked" not in e.run_files()[0].read_text()
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rep["verdict"] == "blocked" and rep["erroredItems"] == 1 and rep["harnessErrors"] == {"HarnessError": 1}


def test_an_exception_the_harness_cannot_place_is_a_harness_error_for_that_anchor(tmp_path):
    e = Env(tmp_path)
    many_world(e, 2)
    e.ask_hook = raising_at(1, ValueError("boom"))
    e.go()
    run = json.loads(e.run_files()[0].read_text())
    assert sorted(v.get("harnessError") for v in run["answers"].values() if "harnessError" in v) == ["ValueError"] and e.marker()["state"] == "asked"


def engine(cause):
    """What ``make_ask`` raises for an exception that came out of the engine: the wrapper, its cause the engine's own exception."""
    try:
        raise R.EngineAskError(type(cause).__name__) from cause
    except R.EngineAskError as e:
        return e


@pytest.mark.parametrize("exc,expect", [
    (engine(ValueError("x")), {"outcome": "engine-error", "exceptionClass": "ValueError"}),
    (engine(StoreError("the answer does not validate")), {"outcome": "engine-error", "exceptionClass": "StoreError"}),
    (G.HarnessError("h"), {"harnessError": "HarnessError"}),
    (BadRequest("b"), {"harnessError": "BadRequest"}),
    (KeyError("k"), {"harnessError": "KeyError"}),
    (ConnectionError("c"), None), (TimeoutError("t"), None), (engine(ConnectionError("c")), None),
    (NoCredentials("n"), None), (StoreAuthError("a"), None), (G.Refusal("r"), None), (R.ResolveRefusal({"ok": 0, "none": 0, "ambiguous": 0, "id-mismatch": 0, "total": 0}), None),
])
def test_the_ask_failure_policy_places_an_exception_by_where_it_was_raised(exc, expect):
    assert R.ask_failure_entry(exc) == expect


def client_raising(exc_by_call):
    """A client stub: the n-th ask (1-based) raises ``exc_by_call[n]`` when present, else answers with the world's answer."""
    class C(FakeClient):
        def ask(self, request=None, **kw):
            self.calls.append(request)
            exc = exc_by_call.get(len(self.calls))
            if exc:
                raise exc
            return self.doc
    return C


def engine_env(tmp_path, exc_by_call, n=3):
    e = Env(tmp_path)
    many_world(e, n)
    c = client_raising(exc_by_call)(doc=e.w.answer())
    stamp = {"snapshotId": H.SNAP, "verdictLogOffset": 0, "verdictLogHash": G.EMPTY_VERDICT_LOG_HASH}
    e.engine_ask = lambda b, s: R.make_ask(c, stamp=stamp, in_graph=lambda req: True)
    return e, c


def test_planted_an_engine_stub_raising_value_error_on_one_anchor_is_a_miss_engine_error_verdict_fail_and_resume_does_not_re_ask(tmp_path, monkeypatch, capsys):
    e, c = engine_env(tmp_path, {1: ValueError("secret-message /home/u id=abc")})
    e.go(ask_factory=e.engine_ask)
    run = json.loads(e.run_files()[0].read_text())
    errs = [v for v in run["answers"].values() if v.get("outcome") == "engine-error"]
    assert errs == [{"outcome": "engine-error", "exceptionClass": "ValueError"}] and e.marker()["state"] == "asked"
    assert "secret-message" not in e.answers_file().read_text() + e.run_files()[0].read_text()
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rep["verdict"] == "fail" and rep["engineErrors"]["byClass"] == {"ValueError": 1} and rep["harnessErrors"] == {}
    assert rep["misses"]["byReason"].get("engine-error") == 1


def test_planted_a_kill_after_an_engine_error_then_resume_never_re_asks_the_errored_anchor(tmp_path):
    plan = {1: ValueError("v")}
    e, c = engine_env(tmp_path, plan, n=3)

    def kill_third(b, s):
        inner = e.engine_ask(b, s)

        def ask(req):
            if len(c.calls) == 2:                                          # two asks are on disk (the engine error and one answer): die before the third
                suicide()
            return inner(req)
        return ask
    status = in_child(lambda: e.go(ask_factory=kill_third))
    assert was_sigkilled(status) and e.marker()["state"] == "started"
    lines = [json.loads(l) for l in e.answers_file().read_text().splitlines()]
    assert len(lines) == 2 and sum(1 for l in lines if l.get("outcome") == "engine-error") == 1
    c.calls.clear()
    plan.clear()
    e.go(ask_factory=e.engine_ask, resume=True)
    assert len(c.calls) == 1                                               # only the never-asked anchor: the engine-error anchor is not re-rolled
    run = json.loads(e.run_files()[0].read_text())
    assert sum(1 for v in run["answers"].values() if v.get("outcome") == "engine-error") == 1 and len(run["answers"]) == 3


def test_planted_the_same_stub_raising_a_connection_error_is_failed_transient_and_records_nothing_for_it(tmp_path):
    e, c = engine_env(tmp_path, {2: ConnectionError("down")})
    with pytest.raises(R.RunFailed) as ei:
        e.go(ask_factory=e.engine_ask)
    assert ei.value.state == "failed:transient" and e.marker()["state"] == "failed:transient" and not e.run_files()
    assert len(e.answers_file().read_text().splitlines()) == 1


@pytest.mark.parametrize("exc", [NoCredentials("n"), StoreAuthError("a")])
def test_a_run_wide_fault_stays_failed_deterministic_and_is_not_recorded_per_anchor(tmp_path, exc):
    e, c = engine_env(tmp_path, {2: exc})
    with pytest.raises(R.RunFailed) as ei:
        e.go(ask_factory=e.engine_ask)
    assert ei.value.state == "failed:deterministic" and not e.run_files() and len(e.answers_file().read_text().splitlines()) == 1


def test_a_bad_request_from_the_engine_is_the_harness_s_error_not_the_engine_s(tmp_path):
    e, c = engine_env(tmp_path, {1: BadRequest("b")})
    e.go(ask_factory=e.engine_ask)
    run = json.loads(e.run_files()[0].read_text())
    assert [v for v in run["answers"].values() if "harnessError" in v] == [{"harnessError": "HarnessError"}]


def test_an_engine_error_beside_a_harness_error_stays_blocked(tmp_path, monkeypatch, capsys):
    e, c = engine_env(tmp_path, {1: ValueError("v"), 2: BadRequest("b")})
    e.go(ask_factory=e.engine_ask)
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rep["verdict"] == "blocked" and rep["engineErrors"]["count"] == 1 and rep["erroredItems"] == 1


# ================================================================================================ N3: the run's flow flag and withheld set are pinned and used by score
def commit_flow_decision(e, verdict="TRUE:measured"):
    dec = e.repo / "cpg" / "eval" / "decisions" / "ccg-p1-flow-python-v2-decision.json"
    dec.parent.mkdir(parents=True, exist_ok=True)
    dec.write_text(json.dumps({"decision": {"verdict": verdict}}))
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "flow decision")


def with_flow_item(e):
    flow = H.item(2, H.contract("http", "GET /flowonly"), H.sym("svc/f.py::g"), stratum="flow")
    e.set_records([e.w.rec, flow])
    return flow


def test_n3_the_marker_pins_the_flow_flag_the_withheld_counts_and_the_probe_map_hash(tmp_path):
    e = Env(tmp_path)
    with_flow_item(e)
    e.go()
    pins = e.marker()["pins"]
    assert pins["flowFlag"] is False and pins["withheld"] == {"flow": 1}
    import hashlib
    assert pins["probeMapSha256"] == hashlib.sha256(e.probes.read_bytes()).hexdigest()


def test_n3_a_decision_flip_between_run_and_score_does_not_change_the_withheld_set_or_the_verdict(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    flow = with_flow_item(e)
    e.go()
    commit_flow_decision(e)                                                           # the committed decision now says TRUE; the run was made at FALSE
    assert G.flow_flag(e.repo / G.FLOW_DECISION) is True
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rep is not None, err
    assert rep["scored"] == 1 and rep["flow"] == {"status": "flow-unavailable, not closed", "flag": False, "withheld": 1} and rep["verdict"] != "fail"
    assert flow["itemId"] not in (e.cust / "gate-outcomes.jsonl").read_text()


def test_n3_a_resume_after_the_decision_flipped_is_refused_naming_the_pin(tmp_path):
    e = Env(tmp_path)
    with_flow_item(e)
    status = in_child(lambda: (setattr(e, "ask_hook", lambda req, n: suicide()), e.go()))
    assert was_sigkilled(status) and e.marker()["state"] == "started"
    commit_flow_decision(e)
    with pytest.raises(G.Refusal, match="flowFlag"):
        e.go(resume=True)


@pytest.mark.parametrize("drop", ["flowFlag", "withheld"])
def test_n3_score_refuses_a_run_file_without_the_recorded_flag_or_withheld_set(tmp_path, monkeypatch, capsys, drop):
    e = Env(tmp_path)
    e.go()
    f = e.run_files()[0]
    run = json.loads(f.read_text())
    run.pop(drop)
    f.write_text(json.dumps(run))
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rc == 2 and rep is None and "flow" in err


def test_n3_score_refuses_when_the_run_files_flag_differs_from_the_markers_pin(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    e.go()
    f = e.run_files()[0]
    f.write_text(json.dumps({**json.loads(f.read_text()), "flowFlag": True}))
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rc == 2 and rep is None and "flow" in err


def test_n3_score_refuses_a_withheld_set_that_the_lists_do_not_reproduce(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    e.go()
    f = e.run_files()[0]
    f.write_text(json.dumps({**json.loads(f.read_text()), "withheld": {"flow": 3}}))
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rc == 2 and rep is None and "withheld" in err


# ================================================================================================ N6: lock, pins
def hold_lock(e):
    import fcntl
    key = e.marker()["listSha256"]
    fd = os.open(e.cust / f"gate-run-{key}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_n6_two_live_resumes_cannot_interleave_the_second_is_refused_and_asks_nothing(tmp_path):
    e = Env(tmp_path)
    many_world(e, 3)
    status = in_child(lambda: (setattr(e, "ask_hook", lambda req, n: suicide() if n == 2 else None), e.go()))
    assert was_sigkilled(status)
    e.asked.clear()
    fd = hold_lock(e)                                                                 # another live process holds the run
    try:
        with pytest.raises(G.Refusal, match="lock|another"):
            e.go(resume=True)
    finally:
        os.close(fd)
    assert e.asked == [] and len(e.answers_file().read_text().splitlines()) == 1
    e.go(resume=True)                                                                 # once released, the resume proceeds
    assert e.marker()["state"] == "asked"


def test_n6_a_real_concurrent_resume_in_a_child_holding_the_lock_is_refused(tmp_path):
    e = Env(tmp_path)
    many_world(e, 3)
    in_child(lambda: (setattr(e, "ask_hook", lambda req, n: suicide() if n == 2 else None), e.go()))
    ready_r, ready_w = os.pipe()                                                      # child -> parent: "I am inside my ask, holding the lock"
    go_r, go_w = os.pipe()                                                            # parent -> child: "carry on"
    signal.alarm(60)                                                                  # a hang is a failure, not a stuck suite
    pid = os.fork()
    if pid == 0:
        def slow(req, n):
            if n == 1:                                                                # the first ask of the resume only
                os.write(ready_w, b"x")
                os.read(go_r, 1)
        e.ask_hook = slow
        try:
            e.go(resume=True)
            os._exit(0)
        except BaseException:                                                         # noqa: BLE001
            import traceback
            traceback.print_exc()
            os.write(ready_w, b"!")                                                   # never leave the parent waiting on a child that died
            os._exit(70)
    try:
        os.read(ready_r, 1)
        with pytest.raises(G.Refusal, match="lock|another"):
            e.go(resume=True)
    finally:
        os.write(go_w, b"y")
        _, st = os.waitpid(pid, 0)
        signal.alarm(0)
    assert os.WEXITSTATUS(st) == 0 and e.marker()["state"] == "asked"
    assert len(e.answers_file().read_text().splitlines()) == 3                       # no duplicated line


def test_n6_the_probe_map_changed_between_run_and_resume_is_refused(tmp_path):
    e = Env(tmp_path)
    many_world(e, 2)
    in_child(lambda: (setattr(e, "ask_hook", lambda req, n: suicide()), e.go()))
    e.set_probes({"x" * 32: "y" * 32})
    with pytest.raises(G.Refusal, match="probeMapSha256"):
        e.go(resume=True)


# ================================================================================================ N4: the gate build never touches the developer's `ccg` work area
def test_n4_the_build_gets_a_private_work_root_keys_root_and_audit_dir_under_the_custodian_root(tmp_path):
    e = Env(tmp_path)
    e.go()
    kw = e.refresh_kw
    assert set(kw) == {"work_root", "keys_root", "audit_dir", "manifest"}
    key = e.marker()["listSha256"]
    cust = e.cust.resolve()
    paths = [Path(kw[k]).resolve() for k in ("work_root", "keys_root", "audit_dir")]
    assert len(set(paths)) == 3 and all(p.parent == cust and key in p.name for p in paths)


def test_n4_a_failed_earlier_attempt_leaves_private_areas_that_the_next_attempt_replaces_not_reuses(tmp_path):
    e = Env(tmp_path)

    def failing_refresh(system, repo_root=None, **kw):
        (Path(kw["work_root"]) / "stale-bundle").write_text("x")
        raise ValueError("build")
    with pytest.raises(ValueError):
        e.go(refresh_fn=failing_refresh)
    assert e.markers() == []                                                          # nothing was spent
    e.go()
    assert not (Path(e.refresh_kw["work_root"]) / "stale-bundle").exists()


def test_n4_a_store_that_already_holds_the_live_graph_is_refused_before_any_build_or_marker(tmp_path):
    e = Env(tmp_path)
    with pytest.raises(G.Refusal, match="live graph"):
        e.go(live_graph_present=lambda: True)
    assert e.events == [] and e.markers() == [] and not [p for p in e.cust.iterdir() if p.name.startswith(("gate-build-", "gate-work-", "gate-keys-", "gate-audit-"))]


def test_n4_an_empty_store_passes_and_a_resume_does_not_ask_the_question(tmp_path):
    e = Env(tmp_path)
    many_world(e, 2)
    in_child(lambda: (setattr(e, "ask_hook", lambda req, n: suicide()), e.go(live_graph_present=lambda: False)))
    e.go(resume=True, live_graph_present=lambda: True)                              # the store holds the run's own graph by then
    assert e.marker()["state"] == "asked"


def test_n4_the_default_deps_probe_the_store_for_the_live_graph_before_building(tmp_path):
    """The probe is a read-only query as the ``raw`` role (no reader role may run ``GRAPH.LIST``): the server's "empty key" answer is the absence (D1; the
    real store is ``test_gate_rehearsal_defects_live.py``)."""
    import argparse
    from redis.exceptions import ResponseError
    seen = {}

    class DB:
        def __init__(self, graphs):
            self.graphs = graphs

        def select_graph(self, k):
            graphs = self.graphs

            class Graph:
                def ro_query(self, cypher, params=None, timeout=None):
                    if k not in graphs:
                        raise ResponseError("Invalid graph operation on empty key")
                    return [[1]]
            return Graph()
    for graphs, expect in ((["ccg_ccg"], True), (["other"], False)):
        a = argparse.Namespace(system="ccg", host="127.0.0.1", port=16420, graph=None)
        deps = R._default_deps(a, db_factory=lambda h, p, g=graphs: (seen.setdefault("hp", (h, p)), DB(g))[1])
        assert deps["live_graph_present"]() is expect
    assert seen["hp"] == ("127.0.0.1", 16420)


def test_n4_a_real_refresh_through_the_gate_build_writes_nothing_outside_the_custodian_root(tmp_path, monkeypatch):
    """The real ``ccg.refresh.refresh`` (offline; the swap and the store are the refresh tests' fakes), with the developer's defaults pointed at sentinel
    directories: after the gate build they are still empty, and the gate's own areas hold the bundles, the key and the audit."""
    import test_refresh as TR
    from ccg import atrest, refresh as RF, sandbox_audit
    dev = tmp_path / "dev"
    dev_work, dev_keys, dev_audit = dev / "work", dev / "keys", dev / "audit"
    for d in (dev_work, dev_keys, dev_audit):
        d.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(atrest, "WORK_ROOT", dev_work)
    monkeypatch.setattr(atrest, "KEY_ROOT", dev_keys)
    monkeypatch.setenv("CCG_AUDIT_DIR", str(dev_audit))
    (tmp_path / "w").mkdir()
    world = TR.World(tmp_path / "w")
    target, _fakes = TR.make_target()
    e = Env(tmp_path)

    def real_refresh(system, repo_root=None, **kw):
        e.refresh_kw = kw
        return RF.refresh(system, target=target, repo_root=world.root, built_at="2027-01-01T00:00:00Z", now=TR.NOW, **{**kw, "manifest": world.manifest})
    e.go(refresh_fn=real_refresh, system=TR.SYSTEM)
    assert not list(dev.rglob("*.json")) and not [p for p in dev.rglob("*") if p.is_file()]                       # the developer's area is untouched
    gate_files = [p for k in ("work_root", "keys_root", "audit_dir") for p in Path(e.refresh_kw[k]).rglob("*") if p.is_file()]
    assert gate_files and all(e.cust.resolve() in p.resolve().parents for p in gate_files)


# ================================================================================================ round D, P3-1: an environment fault is never an engine error
import errno  # noqa: E402


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EACCES, errno.EMFILE, None])
def test_p31_any_oserror_while_recording_an_answer_is_failed_transient_and_resumable(tmp_path, monkeypatch, code):
    e = Env(tmp_path)
    many_world(e, 2)
    real = R._append_line

    def broken(path, obj):
        raise OSError(code, "x") if code else OSError("x")
    monkeypatch.setattr(R, "_append_line", broken)
    with pytest.raises(R.RunFailed) as ei:
        e.go()
    assert ei.value.state == "failed:transient" == e.marker()["state"]
    monkeypatch.setattr(R, "_append_line", real)                                   # space freed: the resume goes through
    e.go(resume=True)
    assert e.marker()["state"] == "asked"


FAMILIES = {                                                                        # one plant per environment family, raised INSIDE the engine's ask
    "oserror-enospc": lambda: OSError(errno.ENOSPC, "no space"), "oserror-edquot": lambda: OSError(errno.EDQUOT, "quota"),
    "oserror-eio": lambda: OSError(errno.EIO, "io"), "oserror-eacces": lambda: PermissionError(errno.EACCES, "perm"),
    "oserror-plain": lambda: OSError("x"), "memory": lambda: MemoryError(), "recursion": lambda: RecursionError("deep"),
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_p31_an_environment_fault_inside_an_ask_is_never_recorded_as_an_engine_error_the_run_stops(tmp_path, family):
    e, c = engine_env(tmp_path, {2: FAMILIES[family]()})
    with pytest.raises(R.RunFailed):
        e.go(ask_factory=e.engine_ask)
    assert not e.run_files() and e.marker()["state"].startswith("failed:")
    assert all("outcome" not in json.loads(l) and "harnessError" not in json.loads(l) for l in e.answers_file().read_text().splitlines())
    assert len(e.answers_file().read_text().splitlines()) == 1


@pytest.mark.parametrize("family", ["oserror-enospc", "oserror-eio", "oserror-plain"])
def test_p31_an_oserror_inside_an_ask_is_failed_transient_and_the_resume_re_asks_that_anchor(tmp_path, family):
    plan = {2: FAMILIES[family]()}
    e, c = engine_env(tmp_path, plan)
    with pytest.raises(R.RunFailed) as ei:
        e.go(ask_factory=e.engine_ask)
    assert ei.value.state == "failed:transient"
    plan.clear()
    c.calls.clear()
    e.go(ask_factory=e.engine_ask, resume=True)
    assert len(c.calls) == 2 and e.marker()["state"] == "asked"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_p31_the_policy_returns_none_for_every_environment_family_wrapped_or_bare(family):
    assert R.ask_failure_entry(FAMILIES[family]()) is None and R.ask_failure_entry(engine(FAMILIES[family]())) is None


# ================================================================================================ round D, P3-2: `run` evaluates every refusal `score` can already know
from gate_run_helpers import DROP  # noqa: E402


@pytest.mark.parametrize("extra", [{"kind": DROP}, {"kind": "re-gate"}, {"kind": "re-gate", "movedOut": {"http": -1}}, {"kind": "nonsense"}],
                         ids=["no-kind", "re-gate-without-movedOut", "re-gate-bad-movedOut", "unknown-kind"])
def test_p32_a_committed_record_that_score_would_refuse_stops_run_before_the_marker_with_nothing_spent(tmp_path, extra):
    e = Env(tmp_path, record_extra=extra)
    with pytest.raises(G.PrecheckRefusal, match="gate kind|movedOut"):
        e.go()
    assert e.markers() == [] and e.events == [] and not list(e.cust.glob("run-*")) and not [p for p in e.cust.iterdir() if p.name.startswith("gate-")]


def test_p32_the_cli_exits_5_for_such_a_record_and_prints_no_content(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    e = Env(tmp_path, record_extra={"kind": DROP})
    assert R.main(e.argv(), deps=e.deps()) == R.EXIT_RESOLVE == 5
    cap = capsys.readouterr()
    assert cap.err.count("REFUSED") == 1 and "SENTINEL" not in cap.err and e.markers() == []


def test_p32_a_valid_record_still_runs_and_the_same_function_is_what_score_calls(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path)
    e.go()
    assert score(e, monkeypatch, capsys)[1] is not None
    import inspect
    assert "score_inputs_check" in inspect.getsource(G.main) and "score_inputs_check" in inspect.getsource(R._run_gate)


# ================================================================================================ round D, P3-3: the repeat-stall conversion (-p1-gate-marker-ml.md §1.1.3)
def store_timeout():
    """A store error that WRAPS a timeout: transient (the cause is), but not a ``TimeoutError`` itself, so it never converts."""
    try:
        raise StoreError("the store timed out") from TimeoutError("t")
    except StoreError as e:
        return e


def stall_env(tmp_path, stalls_on=lambda spec: spec["key"] == "GET /a0", exc=lambda: TimeoutError("deadline"), n=3):
    """A client that raises ``exc()`` whenever the first anchor is asked while ``stalls_on`` says so, and answers every other ask."""
    e = Env(tmp_path)
    many_world(e, n)
    box = {"on": True, "exc": exc}

    class C(FakeClient):
        def ask(self, request=None, **kw):
            self.calls.append(request)
            if box["on"] and stalls_on(request["anchor"]):
                raise box["exc"]()
            return self.doc
    c = C(doc=e.w.answer())
    stamp = {"snapshotId": H.SNAP, "verdictLogOffset": 0, "verdictLogHash": G.EMPTY_VERDICT_LOG_HASH}
    e.stall_ask = lambda b, s: R.make_ask(c, stamp=stamp, in_graph=lambda req: True)
    e.canary_calls = []
    e.canary_ok = True

    def canary():
        e.canary_calls.append(1)
        if not e.canary_ok:
            raise TimeoutError("canary deadline")
    e.canary = canary
    e.box, e.client = box, c
    return e


def answers_text(e) -> str:
    return "".join(f.read_text() for f in e.cust.glob("run-*.answers.jsonl"))


def stop(e, **kw):
    """One process lifetime of the run: it stops on the stalled anchor (or, after the conversion, completes)."""
    first = not e.markers()
    try:
        e.go(ask_factory=e.stall_ask, canary=e.canary, resume=not first, **kw)
    except R.RunFailed as exc:
        return exc.state
    return "done"


def test_p33_stops_one_and_two_are_failed_transient_stop_three_with_the_canary_answering_is_an_engine_error_and_the_run_completes(tmp_path, monkeypatch, capsys):
    e = stall_env(tmp_path)
    assert stop(e) == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 1}
    assert stop(e) == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 2}
    assert stop(e) == "done" and e.marker()["state"] == "asked" and len(e.canary_calls) == 3
    assert e.marker()["stall"] is None                                               # the count is spent with the conversion
    lines = [json.loads(l) for l in e.answers_file().read_text().splitlines()]
    assert {"anchorId": lines[0]["anchorId"], "outcome": "engine-error", "exceptionClass": "TimeoutError", "via": "repeat-stall", "stalls": 3} in lines
    assert len(lines) == 3
    rc, rep, err = score(e, monkeypatch, capsys)
    assert rep["verdict"] == "fail" and rep["misses"]["byReason"]["engine-error"] == 1
    assert rep["engineErrors"]["repeatStall"] == {"count": 1, "byStratum": {"http": 1}} and rep["engineErrors"]["count"] == 0


def test_p33_a_failed_canary_is_plain_transient_and_resets_the_count(tmp_path):
    e = stall_env(tmp_path)
    assert stop(e) == "failed:transient" and stop(e) == "failed:transient"
    e.canary_ok = False
    assert stop(e) == "failed:transient" and e.marker()["stall"] is None
    e.canary_ok = True
    assert stop(e) == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 1}      # the count starts over


@pytest.mark.parametrize("other", [lambda: ConnectionError("c"), lambda: ConnectionResetError("r"), lambda: OSError(errno.EIO, "io"),
                                   lambda: OSError(errno.ENOSPC, "full"), lambda: store_timeout()])
def test_p33_only_a_timeout_converts_any_other_stop_resets_the_count_and_is_never_recorded(tmp_path, other):
    e = stall_env(tmp_path)
    assert stop(e) == "failed:transient" and stop(e) == "failed:transient" and e.marker()["stall"]["count"] == 2
    e.box["exc"] = other
    assert stop(e) in ("failed:transient", "failed:deterministic") and e.marker()["stall"] is None
    e.box["exc"] = lambda: TimeoutError("t")
    assert stop(e) == "failed:transient" and e.marker()["stall"]["count"] == 1
    assert answers_text(e) == ""


@pytest.mark.parametrize("exc", [lambda: ConnectionError("c"), lambda: OSError(errno.EIO, "io"), lambda: OSError(errno.ENOSPC, "full"), lambda: store_timeout()])
def test_p33_a_non_timeout_failure_never_converts_however_often_it_repeats(tmp_path, exc):
    e = stall_env(tmp_path, exc=exc)
    for _ in range(6):
        stop(e)
    assert e.marker()["state"].startswith("failed:") and not e.run_files() and answers_text(e) == ""


def test_p33_progress_resets_the_count(tmp_path):
    e = stall_env(tmp_path, stalls_on=lambda spec: spec["key"] in ("GET /a0", "GET /a1"))
    assert stop(e) == "failed:transient" and stop(e) == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 2}
    stalling = e.client.ask

    def ask(request=None, **kw):                                                     # the first anchor answers from now on, the second one stalls
        if request["anchor"]["key"] == "GET /a0":
            e.client.calls.append(request)
            return e.client.doc
        return stalling(request, **kw)
    e.client.ask = ask
    assert stop(e) == "failed:transient" and e.marker()["stall"] == {"index": 1, "count": 1}   # progress: the second anchor's count starts at 1, not 3


def test_p33_without_a_canary_the_conversion_never_happens(tmp_path):
    e = stall_env(tmp_path)
    for i in range(5):
        try:
            e.go(ask_factory=e.stall_ask, resume=bool(i))
        except R.RunFailed:
            pass
    assert e.marker()["state"] == "failed:transient" and not e.run_files()


@pytest.mark.parametrize("field,value", [("k", 2), ("deadlineS", 1), ("canaryRequest", "RETURN 2")])
def test_p33_k_the_deadline_and_the_canary_request_are_pinned_a_resume_with_another_value_is_refused(tmp_path, field, value):
    e = stall_env(tmp_path)
    assert stop(e) == "failed:transient"
    assert set(e.marker()["pins"]["stall"]) == {"k", "deadlineS", "canaryRequest"}
    over = {"stall_k": value} if field == "k" else {"stall_deadline_s": value} if field == "deadlineS" else {"canary_request": value}
    with pytest.raises(G.Refusal, match="stall"):
        stop(e, **over)


def test_p33_a_sigkill_before_the_stall_write_keeps_the_previous_marker_whole_and_the_count_resumes_from_it(tmp_path):
    e = stall_env(tmp_path)
    assert stop(e) == "failed:transient"
    real = R._write_marker

    def kill_on_stall(path, doc):
        if doc["state"] == "failed:transient":
            suicide()
        return real(path, doc)

    def child():
        R._write_marker = kill_on_stall
        stop(e)
    assert was_sigkilled(in_child(child))
    m = e.marker()                                                                   # complete, parseable: the atomic rewrite never half-happened
    assert m["state"] == "started" and m["stall"] == {"index": 0, "count": 1}
    assert stop(e) == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 2}


def test_p33_a_sigkill_after_the_stall_write_keeps_the_count_and_the_conversion_still_happens(tmp_path):
    e = stall_env(tmp_path)
    assert stop(e) == "failed:transient"
    real = R._write_marker

    def kill_after_stall(path, doc):
        real(path, doc)
        if doc["state"] == "failed:transient":
            suicide()

    def child():
        R._write_marker = kill_after_stall
        stop(e)
    assert was_sigkilled(in_child(child))
    assert e.marker()["state"] == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 2}
    assert stop(e) == "done" and e.marker()["state"] == "asked"


def test_p33_the_conversion_leaves_the_finalise_path_intact_a_kill_after_run_json_still_finalises(tmp_path):
    e = stall_env(tmp_path)
    assert stop(e) == "failed:transient" and stop(e) == "failed:transient"
    key = e.marker()["listSha256"]
    assert was_sigkilled(in_child(lambda: (kill_after_rename_of(f"run-{key}.json"), stop(e))))
    assert e.marker()["state"] == "started" and len(e.run_files()) == 1                  # the stalled anchor was converted, the run completed, the kill hit last
    e.client.calls.clear()
    assert e.go(resume=True)["finalised"] is True and e.marker()["state"] == "asked" and e.client.calls == []


def test_p33_a_timeout_is_still_not_placed_by_the_policy_function(tmp_path):
    assert R.ask_failure_entry(TimeoutError("t")) is None and R.ask_failure_entry(engine(TimeoutError("t"))) is None
