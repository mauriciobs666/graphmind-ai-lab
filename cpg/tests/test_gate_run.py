"""P1-32b gate harness: asking (one question per anchor, cache and snapshot-id order), AC-24 repeat checks, custody, preflight and the CLI.

Offline and synthetic; the CLI test injects the snapshot and never touches a store."""
import json
from pathlib import Path

import pytest

import gate_helpers as H
from gate_helpers import G


# ---------------------------------------------------------------- run_asks
def test_run_asks_reads_snapshot_id_before_the_first_question_after_the_cache_check():
    w, events = H.World(), []
    snap = w.snap
    orig = snap.snapshot_id
    snap.snapshot_id = lambda: (events.append("snapshot-id"), orig())[1]
    run = G.run_asks([w.rec], snap, lambda req: (events.append("ask"), w.answer())[1], lambda: (events.append("cache"), 0)[1])
    assert events == ["cache", "snapshot-id", "ask"] and run["snapshotId"] == H.SNAP
    assert list(run["answers"]) == [w.cp] and run["answers"][w.cp]["request"] == {
        "useCase": "impact", "direction": "impact", "anchors": [{"type": "contract", "cpId": w.cp, "kind": "http", "keyNorm": "GET /a"}]}


def test_run_asks_refuses_a_non_empty_cache_before_any_question():
    w, asked = H.World(), []
    with pytest.raises(G.Refusal, match="cache is not empty"):
        G.run_asks([w.rec], w.snap, lambda r: asked.append(r), lambda: 3)
    assert asked == []


def test_run_asks_asks_once_per_anchor_for_all_its_items():
    w = H.World()
    r2 = H.item(2, w.anchor, H.sym("svc/other.py::g"))
    asked = []
    G.run_asks([w.rec, r2], w.snap, lambda req: (asked.append(req), w.answer())[1], lambda: 0)
    assert len(asked) == 1


def test_a_contract_anchor_absent_from_the_graph_is_not_asked_and_is_an_anchor_not_in_graph_miss():
    w = H.World()
    absent = H.item(2, H.contract("http", "GET /not/in/graph"), H.sym("svc/c.py::call"))
    asked = []
    run = G.run_asks([w.rec, absent], w.snap, lambda req: (asked.append(req), w.answer())[1], lambda: 0)
    assert len(asked) == 1 and absent["anchorId"] not in run["answers"]
    res = G.score_gate([w.rec, absent], run, w.snap, ())
    assert res["items"][absent["itemId"]] == {"stratum": "http", "anchorId": absent["anchorId"], "counting": True, "outcome": "miss",
                                              "reason": "anchor-not-in-graph"}
    assert res["items"][w.rec["itemId"]]["outcome"] == "hit"


def test_out_of_scope_held_out_item_is_key_error_not_asked_scored_and_not_a_miss():
    w = H.World()
    oos = H.item(2, w.anchor, H.sym("cpg/ccg/x.py::g"))
    res = G.score_gate([w.rec, oos], w.run(), w.snap, ["cpg/"])
    assert res["items"][oos["itemId"]]["outcome"] == "out-of-scope" and res["items"][oos["itemId"]]["triage"] == "key-error"
    d = G.decide_gate(res)
    assert d["misses"] == 0 and d["scored"] == 1


def test_symbol_and_config_anchors_are_asked_in_their_own_mode():
    rec = H.item(3, H.sym("svc/a.py::f"), H.sym("svc/b.py::g"))
    req = G.build_request(rec, rec["anchorId"])
    assert req["anchors"] == [{"type": "symbol", "symbolId": rec["anchorId"], "qname": "svc/a.py::f", "repoId": "r", "path": "svc/a.py"}]
    assert "scope" not in req


def _unit_world():
    c1, c2, c3 = (H.contract("mcp_tool", t) for t in ("alpha", "beta", "gamma"))
    ids = {t: G._mk.contract_id(c) for t, c in zip("abc", (c1, c2, c3))}
    unit = {"type": "unit", "repoId": "r", "rootPath": "svc", "lang": "python"}
    rec = H.item(1, unit, H.cfg("compose.yaml", "services.s"), serves=[c1, c2, c3])
    others = [H.item(10 + k, c, H.sym("a.py::f")) for k, c in enumerate((c1, c1, c3))]       # c1 has 2 items, c3 has 1, c2 has none
    return ids, rec, others


def test_unit_probe_cross_check_is_fewest_items_then_smallest_cpid():
    ids, rec, others = _unit_world()
    assert G.unit_probe(rec, [rec, *others]) == ids["b"]
    tie = G.unit_probe(rec, [rec, others[0], H.item(20, rec["serves"][1], H.sym("a.py::f")), others[2]])
    assert tie == min(ids.values())


def test_unit_anchor_with_no_served_point_is_a_harness_error():
    rec = H.item(1, {"type": "unit", "repoId": "r", "rootPath": "s", "lang": "python"}, H.cfg("c.yaml", "a"))
    with pytest.raises(G.HarnessError):
        G.unit_probe(rec, [rec])


def _snap_with(ids):
    snap = H.DictSnapshot()
    for cp in ids.values():
        snap.add_node("ContractPoint", cp)
    return snap


def test_m4_the_unit_anchor_is_asked_from_the_s47_probe_map_not_from_serves():
    ids, rec, others = _unit_world()
    asked = []
    probes = {rec["anchorId"]: ids["c"]}                                  # S47 says gamma; the record-local rule would say beta
    run = G.run_asks([rec, *others], _snap_with(ids), lambda req: (asked.append(req), {})[1], lambda: 0, probes=probes)
    assert [a["anchors"][0]["cpId"] for a in asked if a["anchors"][0]["cpId"] in ids.values()][0:1] == [ids["c"]] and ids["b"] not in run["answers"]
    assert run["probes"] == probes and run["probeCrossCheckDiffers"] == 1


def test_m4_two_unit_anchors_sharing_a_probe_are_both_recorded_and_asked_once():
    ids, rec, others = _unit_world()
    unit2 = {"type": "unit", "repoId": "r", "rootPath": "svc2", "lang": "python"}
    rec2 = H.item(2, unit2, H.cfg("compose.yaml", "services.t"), serves=[rec["serves"][0]])
    probes = {rec["anchorId"]: ids["a"], rec2["anchorId"]: ids["a"]}
    asked = []
    run = G.run_asks([rec, rec2], _snap_with(ids), lambda r: (asked.append(r), {})[1], lambda: 0, probes=probes)
    assert len(asked) == 1 and run["probes"] == probes


def test_m4_a_unit_item_with_no_probe_entry_fails_closed_before_any_question():
    ids, rec, others = _unit_world()
    asked = []
    for probes in (None, {}, {"other": ids["a"]}):
        with pytest.raises(G.HarnessError, match="probe"):
            G.run_asks([rec, *others], _snap_with(ids), lambda r: asked.append(r), lambda: 0, probes=probes)
    assert asked == []


def test_m4_a_probe_point_no_record_can_locate_is_a_harness_error():
    ids, rec, _ = _unit_world()
    with pytest.raises(G.HarnessError, match="locat"):
        G.run_asks([rec], _snap_with(ids), lambda r: {}, lambda: 0, probes={rec["anchorId"]: "f" * 32})


def test_m4_a_probe_point_outside_serves_is_located_through_another_record():
    ids, rec, others = _unit_world()
    extra = H.contract("mcp_tool", "delta")
    cp = G._mk.contract_id(extra)
    snap = _snap_with({**ids, "d": cp})
    asked = []
    G.run_asks([rec, H.item(30, extra, H.sym("a.py::f"))], snap, lambda r: (asked.append(r), {})[1], lambda: 0, probes={rec["anchorId"]: cp})
    assert asked[-1]["anchors"][0] == {"type": "contract", "cpId": cp, "kind": "mcp_tool", "keyNorm": "delta"}


def test_m4_score_reads_the_probe_from_the_run_file_and_scores_the_unit_item():
    ids, rec, _ = _unit_world()
    cp = ids["c"]
    req = G.build_request(rec, cp)
    run = {"snapshotId": H.SNAP, "probes": {rec["anchorId"]: cp}, "answers": {cp: {"request": req, "answer": H.answer(req, [])}}}
    o = G.score_gate([rec], run, _snap_with(ids), ())["items"][rec["itemId"]]
    assert o["outcome"] == "miss" and o["reason"] == "no-qualifying-item"
    with pytest.raises(G.HarnessError, match="probe"):
        G.score_gate([rec], {k: v for k, v in run.items() if k != "probes"}, _snap_with(ids), ())


def test_m4_read_probe_map_reads_only_through_the_custody_guard(tmp_path):
    home = tmp_path / "home"
    area = G._ac.custodian_area(home) / "abc"
    area.mkdir(parents=True)
    (area / "s47-probe.json").write_text(json.dumps({"u1": "c" * 32}))
    assert G.read_probe_map(area / "s47-probe.json", home=home) == {"u1": "c" * 32}
    (tmp_path / "elsewhere.json").write_text("{}")
    with pytest.raises(G.Refusal):
        G.read_probe_map(tmp_path / "elsewhere.json", home=home)
    (area / "bad.json").write_text("[1]")
    with pytest.raises(G.HarnessError):
        G.read_probe_map(area / "bad.json", home=home)


# ---------------------------------------------------------------- AC-24 repeats (C5)
def rep(n, hit):
    return {"repeat": n, "answer": {"interpretation": {"status": "resolved", "provenance": {"cacheHit": hit}}}}


def test_all_repeats_uncached_is_clean():
    assert G.check_repeat_cache_hits([rep(n, False) for n in range(1, 6)]) == {"stop": False, "voided": False, "badRepeats": []}


def test_cache_hit_on_repeat_1_stops_the_run():
    assert G.check_repeat_cache_hits([rep(1, True)] + [rep(n, False) for n in range(2, 6)])["stop"] is True


def test_cache_hit_on_a_later_repeat_voids_that_items_repeats_without_stopping():
    r = G.check_repeat_cache_hits([rep(1, False), rep(3, True)])
    assert r == {"stop": False, "voided": True, "badRepeats": [3]}


def test_a_missing_cache_hit_field_is_not_a_pass():
    r = G.check_repeat_cache_hits([{"repeat": 1, "answer": {"interpretation": {"status": "resolved"}}}])
    assert r["stop"] is True


# ---------------------------------------------------------------- custody
def test_write_custodian_refuses_a_path_outside_the_custodian_root(tmp_path):
    with pytest.raises(G.Refusal, match="custodian"):
        G.write_custodian(tmp_path / "elsewhere", "x.jsonl", [{"a": 1}], home=tmp_path / "home")


def test_write_custodian_writes_under_the_root(tmp_path):
    root = tmp_path / "home" / ".local" / "share" / "ccg-eval" / "abc"
    f = G.write_custodian(root, "x.jsonl", [{"b": 2}, {"a": 1}], home=tmp_path / "home")
    assert f.read_text() == '{"b": 2}\n{"a": 1}\n'


def test_a2_library_refusals_do_not_carry_the_cli_prefix(tmp_path):
    """The CLI adds ``REFUSED:`` once; a library message that starts with it prints it twice."""
    home = tmp_path / "home"
    (tmp_path / "elsewhere.json").write_text("{}")
    with pytest.raises(G.Refusal) as e1:
        G.read_probe_map(tmp_path / "elsewhere.json", home=home)
    with pytest.raises(G.Refusal) as e2:
        G.write_custodian(tmp_path / "elsewhere", "x.jsonl", [{"a": 1}], home=home)
    assert "custodian" in str(e1.value) and "custodian" in str(e2.value)
    assert "REFUSED" not in str(e1.value) and "REFUSED" not in str(e2.value)


def test_write_custodian_refuses_the_repo_tree(tmp_path):
    with pytest.raises(G.Refusal):
        G.write_custodian(G._CPG.parent / "cpg" / "eval", "leak.json", {"x": 1}, home=tmp_path)


# ---------------------------------------------------------------- preflight on a throwaway repo
YAML_EM = "exclusions:\n  paths: [\"cpg/\"]\nprotocol:\n  documents: []\n  protocolString: \"\"\n"
YAML_SYS = "repos:\n  - exclusions:\n      paths: [\"cpg/\"]\n"
OBS = {"algorithm": {"id": "bfs-v1", "version": "1.2"}, "defaultDepthBound": 2}


def prepared(tmp_path, **extra):
    repo = H.make_repo(tmp_path)
    for rel in G.HARNESS_FILES:
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"# {rel}\n")
    (repo / "cpg/eval/eval-manifest.yaml").write_text(YAML_EM)
    (repo / "cpg/ccg").mkdir(exist_ok=True)
    (repo / "cpg/ccg/system.yaml").write_text(YAML_SYS)
    H.git(repo, "add", "-A")
    H.git(repo, "commit", "-qm", "harness")
    t = G.observe_tree(repo)
    rec = {"schema": "ccg-prereg-v1", "stratum": "ac5-gate-1", "packSetHash": t["packSetHash"], "coreTreeHash": t["coreTreeHash"], **OBS, **extra}
    (repo / "cpg/eval/preregistration.jsonl").write_text(json.dumps(rec) + "\n")
    H.git(repo, "add", "-A")
    H.git(repo, "commit", "-qm", "record")
    return repo


def test_preflight_first_contact_when_everything_is_unchanged(tmp_path):
    repo = prepared(tmp_path)
    p = G.preflight(repo, "ac5-gate-1", observed=OBS)
    assert p["label"] == "first-contact" and p["recordCommit"] == H.head(repo)
    assert p["firstContact"]["firstContact"] is True and p["harnessPin"]["afterAdaptation"] is False and p["scope"]["agree"] is True


def test_preflight_a_harness_edit_after_the_record_is_after_adaptation(tmp_path):
    repo = prepared(tmp_path)
    (repo / "cpg/eval/tools/gate.py").write_text("# edited\n")
    H.git(repo, "commit", "-qam", "edit")
    p = G.preflight(repo, "ac5-gate-1", observed=OBS)
    assert p["label"] == "after-adaptation" and p["harnessPin"]["differ"] == ["cpg/eval/tools/gate.py"]
    assert p["firstContact"]["firstContact"] is False or p["firstContact"]["dirtyPaths"] == 0


def test_preflight_each_observed_field_differing_is_after_adaptation(tmp_path):
    repo = prepared(tmp_path)
    p = G.preflight(repo, "ac5-gate-1", observed={**OBS, "defaultDepthBound": 3})
    assert p["label"] == "after-adaptation" and p["firstContact"]["fieldsDiffering"] == ["defaultDepthBound"]


def test_preflight_a_dirty_cpg_tree_is_after_adaptation(tmp_path):
    repo = prepared(tmp_path)
    (repo / "cpg/ccg/a.py").write_text("A = 5\n")
    p = G.preflight(repo, "ac5-gate-1", observed=OBS)
    assert p["label"] == "after-adaptation" and p["firstContact"]["dirtyPaths"] == 1


# ---------------------------------------------------------------- M5: the record precedes the build; the label is carried
def test_m5_run_context_returns_the_label_pins_and_hashes_in_a_json_form(tmp_path):
    repo = prepared(tmp_path)
    pf = G.preflight(repo, "ac5-gate-1", observed=OBS)
    ctx = G.run_context(repo, pf)
    assert json.loads(json.dumps(ctx)) == ctx
    assert ctx["label"] == "first-contact" and ctx["recordCommit"] == H.head(repo) == ctx["head"]
    assert ctx["harnessPin"]["afterAdaptation"] is False and set(ctx["harnessPin"]["sha256"]) == set(G.HARNESS_FILES)
    assert all(len(v) == 64 for v in ctx["harnessPin"]["sha256"].values())


def test_m5_no_record_commit_found_is_refused(tmp_path):
    repo = prepared(tmp_path)
    pf = G.preflight(repo, "ac5-gate-1", observed=OBS)
    with pytest.raises(G.Refusal, match="record"):
        G.run_context(repo, {**pf, "recordCommit": None})


def test_m5_a_record_commit_that_is_not_an_ancestor_of_head_is_refused(tmp_path):
    repo = prepared(tmp_path)
    pf = G.preflight(repo, "ac5-gate-1", observed=OBS)
    base = H.head(repo)
    H.git(repo, "checkout", "-q", "-b", "side")
    (repo / "cpg/eval/side.txt").write_text("s\n")
    H.git(repo, "add", "-A")
    H.git(repo, "commit", "-qm", "side")
    side = H.head(repo)
    H.git(repo, "checkout", "-q", "main")
    assert side != base
    with pytest.raises(G.Refusal, match="ancestor"):
        G.run_context(repo, {**pf, "recordCommit": side})


def test_m5_a_commit_after_the_record_that_touches_cpg_outside_eval_is_refused_and_eval_or_docs_are_not(tmp_path):
    repo = prepared(tmp_path)
    pf = G.preflight(repo, "ac5-gate-1", observed=OBS)
    (repo / "cpg/eval/notes.json").write_text("{}\n")
    (repo / "cpg/README.md").write_text("# doc\n")
    H.git(repo, "add", "-A")
    H.git(repo, "commit", "-qm", "eval and doc only")
    assert G.run_context(repo, pf)["label"] == "first-contact"
    (repo / "cpg/ccg/a.py").write_text("A = 2\n")
    H.git(repo, "commit", "-qam", "code")
    with pytest.raises(G.Refusal, match="1 file"):
        G.run_context(repo, pf)


def test_m5_an_after_adaptation_label_is_refused_unless_explicitly_allowed_and_is_then_recorded(tmp_path):
    repo = prepared(tmp_path)
    (repo / "cpg/eval/tools/gate.py").write_text("# edited\n")
    H.git(repo, "commit", "-qam", "edit")
    pf = G.preflight(repo, "ac5-gate-1", observed=OBS)
    assert pf["label"] == "after-adaptation"
    with pytest.raises(G.Refusal, match="after-adaptation"):
        G.run_context(repo, pf)
    ctx = G.run_context(repo, pf, allow_after_adaptation=True)
    assert ctx["label"] == "after-adaptation" and ctx["afterAdaptationAllowed"] is True and ctx["harnessPin"]["differ"] == ["cpg/eval/tools/gate.py"]


# ---------------------------------------------------------------- M7: the gate kind and the moved-out counts come from the committed record
def test_m7_gate_kind_reads_the_committed_record_and_refuses_when_it_is_silent():
    assert G.gate_kind({"kind": "gate"}) == ("gate", None)
    assert G.gate_kind({"kind": "re-gate", "movedOut": {"http": 2}}) == ("re-gate", {"http": 2})
    for bad in ({}, {"kind": "regate"}, {"kind": "re-gate"}, {"kind": "re-gate", "movedOut": {"http": "2"}}, {"kind": "re-gate", "movedOut": []},
                {"kind": "gate", "movedOut": {"http": 1}}):
        with pytest.raises(G.Refusal):
            G.gate_kind(bad)


# ---------------------------------------------------------------- M8: the snapshot factory
class FakeDB:
    made = []

    def __init__(self, host, port):
        FakeDB.made.append((host, port))

    def select_graph(self, name):
        FakeDB.made.append(name)
        return "graph-" + name


@pytest.mark.parametrize("port", [None, 6379, 6390, 16390, 16400])
def test_m8_the_snapshot_factory_refuses_a_missing_or_forbidden_port_before_connecting(port):
    FakeDB.made = []
    with pytest.raises(G.Refusal, match="port"):
        G.snapshot_graph("127.0.0.1", port, None, "ccg", db_factory=FakeDB)
    assert FakeDB.made == []


def test_m8_the_graph_is_derived_from_the_system_and_must_agree_with_an_explicit_one():
    FakeDB.made = []
    assert G.snapshot_graph("127.0.0.1", 16420, None, "ccg", db_factory=FakeDB) == "graph-ccg_ccg"
    assert FakeDB.made == [("127.0.0.1", 16420), "ccg_ccg"]
    assert G.snapshot_graph("127.0.0.1", 16420, "ccg_ccg", "ccg", db_factory=FakeDB) == "graph-ccg_ccg"
    FakeDB.made = []
    with pytest.raises(G.Refusal, match="disagree"):
        G.snapshot_graph("127.0.0.1", 16420, "ccg_other", "ccg", db_factory=FakeDB)
    assert FakeDB.made == []


def test_m8_a_non_loopback_host_is_refused():
    with pytest.raises(G.Refusal, match="host"):
        G.snapshot_graph("10.0.0.5", 16420, None, "ccg", db_factory=FakeDB)


# ---------------------------------------------------------------- CLI score
OBS_FN = lambda repo, gid, **kw: G.preflight(repo, gid, observed=OBS, **kw)       # noqa: E731


class ScoreEnv:
    """A custodian area under a throwaway HOME, a throwaway repo with a committed record, and a run file carrying the preflight block."""

    def __init__(self, tmp_path, monkeypatch, record=None, label="first-contact", run_pin=H.SNAP):
        monkeypatch.setenv("HOME", str(tmp_path))
        self.cust = tmp_path / ".local" / "share" / "ccg-eval" / "e12"
        self.cust.mkdir(parents=True)
        self.shapes = {"scriptsCounting": 0, "withheld": 0, "byShape": {}}
        self.flow_flag, self.withheld = False, {}                  # what the run recorded; a test sets them before set_records/write_run
        self.repo = prepared(tmp_path / "r", **({"kind": "gate"} if record is None else record))
        self.w = H.World(qname="svc/SENTINEL_QNAME.py::SentinelFn")
        self.lists = self.cust / "ac5-http.jsonl"
        self.lists.write_text(json.dumps(self.w.rec) + "\n")
        pf = G.preflight(self.repo, "ac5-gate-1", observed=OBS)
        self.ctx = {**G.run_context(self.repo, pf, allow_after_adaptation=True), "label": label}
        self.run = self.cust / "run.json"
        self.dev_probe = tmp_path / "dev-probe.json"
        self.dev_probe.write_text(json.dumps(self.probe_doc()))
        self.key = G.list_key([self.w.rec])
        self.write_marker("asked")
        self.write_run(run_pin)

    def set_records(self, recs):
        """The list changes, so its key does: the marker and the run file follow it (as they would for a run of that list)."""
        self.lists.write_text("".join(json.dumps(r) + "\n" for r in recs))
        self.key = G.list_key(recs)
        self.withheld = G.withhold_strata(recs, G.gated_strata(self.flow_flag))[1]       # what a run of these lists would have recorded
        self.shapes = {"scriptsCounting": sum(1 for r in recs if r["derivedStratum"] == "scripts" and r.get("counting")), "withheld": 0, "byShape": {}}   # literal: no list here holds a shape
        self.write_marker("asked")
        self.write_run()

    def write_marker(self, state, key=None):
        (self.cust / f"gate-run-{key or self.key}.marker").write_text(
            json.dumps({"state": state, "pins": {"flowFlag": self.flow_flag, "devProbeSha256": H.sha_of(self.dev_probe)}}) + "\n")

    def write_run(self, pin=H.SNAP, **extra):
        run = {**self.w.run(snapshot_id=pin), "preflight": self.ctx, "listKey": self.key, "flowFlag": self.flow_flag, "withheld": self.withheld, "withheldShapes": self.shapes, **extra}
        self.run.write_text(json.dumps(run))

    def probe_doc(self, **kw):
        return H.dev_probe_doc(self.repo, record_commit=self.ctx["recordCommit"], **kw)

    def argv(self, **over):
        a = {"--gate-id": "ac5-gate-1", "--lists": str(self.lists), "--run": str(self.run), "--custodian": str(self.cust), "--port": "16420",
             "--dev-probe": str(self.dev_probe)}
        a.update(over)
        return ["score", *[x for k, v in a.items() if v is not None for x in (k, v)]]

    def main(self, argv=None, snapshot="w", **kw):
        return G.main(argv or self.argv(), snapshot=self.w.snap if snapshot == "w" else snapshot, repo=self.repo, preflight_fn=OBS_FN, **kw)


def test_cli_score_prints_counts_only_and_writes_outcomes_to_the_custodian(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    rc = e.main()
    out = capsys.readouterr().out
    assert rc == 1                                           # one item: minimums not met, so not a pass
    rep = json.loads(out)
    assert rep["verdict"] == "minimums-not-met" and rep["scored"] == 1 and rep["label"] == "first-contact" and rep["kind"] == "gate"
    assert rep["flow"]["status"] == "flow-unavailable, not closed" and rep["flow"]["flag"] is False
    for secret in ("SENTINEL", e.w.rec["itemId"], e.w.rec["anchorId"], e.w.cp, e.w.sid):
        assert secret not in out
    rows = [json.loads(l) for l in (e.cust / "gate-outcomes.jsonl").read_text().splitlines()]
    assert rows[0]["itemId"] == e.w.rec["itemId"] and rows[0]["outcome"] == "hit"
    assert [json.loads(l)["stratum"] for l in (e.cust / "gate-exposure.jsonl").read_text().splitlines()] == ["http"]


# ---- A2 (ruling §1.1.7): score requires the marker of the run's list key to be in state `asked`
def _refused_without_output(e, capsys, **kw):
    rc = e.main(**kw)
    cap = capsys.readouterr()
    assert rc == 2 and cap.err.count("REFUSED") == 1 and cap.out == ""
    assert not (e.cust / "gate-outcomes.jsonl").exists() and not (e.cust / "gate-report.json").exists()
    for secret in ("SENTINEL", e.w.rec["itemId"], e.w.rec["anchorId"], e.key):
        assert secret not in cap.err
    return cap.err


@pytest.mark.parametrize("state", ["started", "failed:transient", "failed:deterministic", "anything-else"])
def test_a2_score_refuses_unless_the_marker_is_asked(tmp_path, monkeypatch, capsys, state):
    e = ScoreEnv(tmp_path, monkeypatch)
    e.write_marker(state)
    assert "marker" in _refused_without_output(e, capsys)


def test_a2_score_refuses_when_there_is_no_marker(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    (e.cust / f"gate-run-{e.key}.marker").unlink()
    assert "marker" in _refused_without_output(e, capsys)


@pytest.mark.parametrize("text", ["not json", "[1]", "{}", '{"state": 7}', ""])
def test_a2_score_refuses_an_unreadable_or_malformed_marker(tmp_path, monkeypatch, capsys, text):
    e = ScoreEnv(tmp_path, monkeypatch)
    (e.cust / f"gate-run-{e.key}.marker").write_text(text)
    assert "marker" in _refused_without_output(e, capsys)


def test_a2_the_marker_of_another_list_does_not_serve(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    (e.cust / f"gate-run-{e.key}.marker").unlink()
    e.write_marker("asked", key="b" * 64)
    assert "marker" in _refused_without_output(e, capsys)


@pytest.mark.parametrize("listkey", [None, "b" * 64])
def test_a2_a_run_file_whose_list_key_is_missing_or_not_the_lists_is_refused(tmp_path, monkeypatch, capsys, listkey):
    e = ScoreEnv(tmp_path, monkeypatch)
    run = json.loads(e.run.read_text())
    run.pop("listKey")
    e.run.write_text(json.dumps({**run, **({} if listkey is None else {"listKey": listkey})}))
    if listkey:
        e.write_marker("asked", key=listkey)                      # even a marker for the run's own (wrong) key does not serve the --lists
    assert "list" in _refused_without_output(e, capsys)


def test_a2_score_stays_re_runnable_on_asked(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    assert e.main() == 1 and e.main() == 1
    assert json.loads((e.cust / "gate-report.json").read_text())["verdict"] == "minimums-not-met"


def test_a2_a_marker_written_by_gate_run_is_what_score_reads(tmp_path):
    """The marker file name and state vocabulary are gate_run's, not a copy that could drift."""
    from gate_run_helpers import Env
    e = Env(tmp_path)
    e.go()
    run = json.loads(e.run_files()[0].read_text())
    assert e.marker()["state"] == "asked" and run["listKey"] == G.list_key([e.w.rec])
    assert G.marker_state(e.cust, run["listKey"]) == "asked"


# ---- A2: gate_run today never writes the per-anchor harness-error entry (the score_gate shape is forward-compatible)
def test_a2_gate_run_writes_no_harness_error_entry_today(tmp_path):
    from gate_run_helpers import Env
    e = Env(tmp_path)
    e.go()
    run = json.loads(e.run_files()[0].read_text())
    assert all(set(v) == {"request", "answer"} for v in run["answers"].values())


def test_a2_a_post_marker_harness_error_is_recorded_for_its_anchor_and_the_run_is_complete(tmp_path):
    """Round C: Q1 of the Pass 2 review. The run no longer ends on a per-anchor harness error: the line is persisted, `run.json` written, marker `asked`."""
    from gate_run_helpers import Env
    e = Env(tmp_path)
    e.ask_hook = lambda req, n: (_ for _ in ()).throw(G.HarnessError("planted"))
    e.go()
    assert [json.loads(l) for l in e.answers_file().read_text().splitlines()] == [{"anchorId": e.w.cp, "harnessError": "HarnessError"}]
    assert json.loads(e.run_files()[0].read_text())["answers"] == {e.w.cp: {"harnessError": "HarnessError"}} and e.marker()["state"] == "asked"


def test_a2_score_of_a_run_with_an_errored_anchor_reports_blocked_with_counts_and_a_class_only(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    e.write_run(answers={e.w.cp: {"harnessError": "StoreError"}})
    rc = e.main()
    cap = capsys.readouterr()
    rep = json.loads(cap.out)
    assert rc == 1 and rep["verdict"] == "blocked" and rep["harnessErrors"] == {"StoreError": 1} and rep["erroredItems"] == 1
    for secret in ("SENTINEL", e.w.rec["itemId"], e.w.rec["anchorId"], e.w.cp):
        assert secret not in cap.out + cap.err


def test_cli_score_refuses_a_custodian_dir_outside_the_area(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = G.main(["score", "--gate-id", "g", "--lists", "x", "--run", "y", "--custodian", str(tmp_path / "elsewhere"), "--dev-probe", "z"])
    assert rc == 2 and "REFUSED" in capsys.readouterr().err


def test_cli_score_stops_before_scoring_on_a_stored_gold_renormalize_difference(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    stale = dict(e.w.rec, anchorId="0" * 32)
    e.lists.write_text(json.dumps(stale) + "\n")
    rc = e.main()
    assert rc == 2 and "renormalize" in capsys.readouterr().err and not (e.cust / "gate-outcomes.jsonl").exists()


@pytest.mark.parametrize("which", ["--lists", "--run"])
def test_m8_score_reads_lists_and_run_only_from_the_custodian_area(tmp_path, monkeypatch, capsys, which):
    e = ScoreEnv(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f").write_text((e.lists if which == "--lists" else e.run).read_text())
    rc = e.main(e.argv(**{which: str(outside / "f")}))
    err = capsys.readouterr().err
    assert rc == 2 and "custodian" in err and "REFUSED: REFUSED" not in err and not (e.cust / "gate-outcomes.jsonl").exists()


def test_m8_a_symlink_out_of_custody_and_a_dotdot_path_are_refused(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    (tmp_path / "out.jsonl").write_text(e.lists.read_text())
    (e.cust / "link.jsonl").symlink_to(tmp_path / "out.jsonl")
    assert e.main(e.argv(**{"--lists": str(e.cust / "link.jsonl")})) == 2
    assert e.main(e.argv(**{"--lists": str(e.cust / ".." / ".." / ".." / ".." / "out.jsonl")})) == 2


@pytest.mark.parametrize("port", ["6379", "6390", "16390", "16400"])
def test_m8_score_refuses_a_forbidden_port_before_touching_a_store(tmp_path, monkeypatch, capsys, port):
    e = ScoreEnv(tmp_path, monkeypatch)
    assert e.main(e.argv(**{"--port": port}), snapshot=None) == 2
    assert "port" in capsys.readouterr().err


def test_m8_score_refuses_a_missing_port_and_a_graph_that_disagrees_with_the_system(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    assert e.main(e.argv(**{"--port": None}), snapshot=None) == 2
    assert e.main(e.argv(**{"--graph": "ccg_other", "--system": "ccg"}), snapshot=None) == 2


def test_m2_score_rereads_the_live_snapshot_id_and_stops_when_the_run_file_disagrees(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch, run_pin="b" * 32)                      # plant P3: the file's pin equals the answers' stamp, not the live node
    run = json.loads(e.run.read_text())
    for entry in run["answers"].values():
        entry["answer"]["stamp"]["snapshotId"] = "b" * 32
    e.run.write_text(json.dumps(run))
    rc = e.main()
    assert rc == 2 and "Snapshot" in capsys.readouterr().err and not (e.cust / "gate-outcomes.jsonl").exists()


def test_m2_score_refuses_a_dirty_cpg_tree(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    (e.repo / "cpg/ccg/a.py").write_text("A = 9\n")
    assert e.main() == 2 and "dirty" in capsys.readouterr().err


def test_m2_score_refuses_a_run_file_with_no_preflight_block(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    run = json.loads(e.run.read_text())
    run.pop("preflight")
    e.run.write_text(json.dumps(run))
    assert e.main() == 2 and "preflight" in capsys.readouterr().err


def test_m2_the_report_carries_the_run_label_and_the_pins(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch, label="after-adaptation")
    e.main()
    rep = json.loads(capsys.readouterr().out)
    assert rep["label"] == "after-adaptation" and rep["preflight"]["recordCommit"] == e.ctx["recordCommit"]
    assert rep["preflight"]["harnessPin"]["afterAdaptation"] is False and "sha256" in rep["preflight"]["harnessPin"]


def test_m7_a_record_that_does_not_state_its_kind_makes_score_refuse(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch, record={"note": "no kind"})
    assert e.main() == 2 and "kind" in capsys.readouterr().err


def test_m7_moved_out_is_no_longer_an_operator_input(tmp_path, monkeypatch):
    e = ScoreEnv(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        e.main(e.argv(**{"--moved-out": "{}"}))


def test_m7_a_regate_record_needs_the_flags_file_and_a_gate_record_refuses_one(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch, record={"kind": "re-gate", "movedOut": {"http": 1}})
    assert e.main() == 2 and "flags" in capsys.readouterr().err
    flags = e.cust / "flags.jsonl"
    flags.write_text(json.dumps({"itemId": e.w.rec["itemId"], "flag": "fresh-cluster"}) + "\n")
    assert e.main(e.argv(**{"--flags": str(flags)})) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["kind"] == "re-gate" and rep["verdict"] == "all hits, composition not met"          # 75 counting items are missing
    g = ScoreEnv(tmp_path / "g", monkeypatch)
    (g.cust / "flags.jsonl").write_text("")
    assert g.main(g.argv(**{"--flags": str(g.cust / "flags.jsonl")})) == 2


def test_m1_flow_items_are_withheld_and_only_counted(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    flow = H.item(2, H.contract("http", "GET /flowonly"), H.sym("svc/f.py::g"), stratum="flow")
    e.set_records([e.w.rec, flow])
    e.main()
    rep = json.loads(capsys.readouterr().out)
    assert rep["flow"] == {"status": "flow-unavailable, not closed", "flag": False, "withheld": 1} and rep["scored"] == 1
    assert flow["itemId"] not in (e.cust / "gate-outcomes.jsonl").read_text()


def test_m7_the_moved_out_counts_of_the_record_reach_the_verdict(tmp_path, monkeypatch, capsys):
    def run(moved):
        e = ScoreEnv(tmp_path / f"m{moved}", monkeypatch, record={"kind": "re-gate", "movedOut": {"http": moved}})
        recs = [H.item(n, e.w.anchor, H.sym(e.w.qname), stratum=("http", "mcp", "scripts")[n % 3]) for n in range(1, 81)]
        e.set_records(recs)
        fresh = [r for r in recs if r["derivedStratum"] == "http"][:20] + [r for r in recs if r["derivedStratum"] == "mcp"][:15]
        flags = e.cust / "flags.jsonl"
        ids = {r["itemId"] for r in fresh}
        flags.write_text("".join(json.dumps({"itemId": r["itemId"], "flag": "fresh-cluster" if r["itemId"] in ids else "kept"}) + "\n" for r in recs))
        rc = e.main(e.argv(**{"--flags": str(flags)}))
        return rc, json.loads(capsys.readouterr().out)
    rc, rep = run(20)
    assert (rc, rep["verdict"], rep["kind"]) == (0, "pass", "re-gate")
    rc, rep = run(25)                                        # the record says 25 http items moved out, only 20 fresh http items are present
    assert (rc, rep["verdict"]) == (1, "all hits, composition not met")


def test_m8_the_flags_file_is_read_only_from_the_custodian_area(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch, record={"kind": "re-gate", "movedOut": {}})
    (tmp_path / "flags.jsonl").write_text("")
    assert e.main(e.argv(**{"--flags": str(tmp_path / "flags.jsonl")})) == 2 and "custodian" in capsys.readouterr().err


@pytest.mark.parametrize("block", [{"label": "x", "recordCommit": "a" * 40}, {"label": "first-contact"}, {"label": "first-contact", "recordCommit": ""}, "text"])
def test_m2_a_malformed_preflight_block_is_refused(tmp_path, monkeypatch, capsys, block):
    e = ScoreEnv(tmp_path, monkeypatch)
    run = json.loads(e.run.read_text())
    run["preflight"] = block
    e.run.write_text(json.dumps(run))
    assert e.main() == 2 and "preflight" in capsys.readouterr().err


def test_m2_a_harness_edit_after_the_run_downgrades_the_label_at_score(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)                      # the run recorded first-contact
    (e.repo / "cpg/eval/tools/gate.py").write_text("# edited after the run\n")
    H.git(e.repo, "commit", "-qam", "edit")
    e.main()
    rep = json.loads(capsys.readouterr().out)
    assert rep["label"] == "after-adaptation" and rep["scoreLabel"] == "after-adaptation" and rep["preflight"]["label"] == "first-contact"


def test_m1_a_committed_true_flow_decision_makes_flow_a_gated_stratum_at_score(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)
    dec = e.repo / "cpg" / "eval" / "decisions" / "ccg-p1-flow-python-v2-decision.json"
    dec.parent.mkdir(parents=True, exist_ok=True)
    dec.write_text(json.dumps({"decision": {"verdict": "TRUE:measured"}}))
    H.git(e.repo, "add", "-A")
    H.git(e.repo, "commit", "-qm", "decision")
    e.flow_flag = True                                       # the run was made while the committed decision said TRUE
    flow = H.item(2, H.contract("http", "GET /flowonly"), H.sym("svc/f.py::g"), stratum="flow")
    e.set_records([e.w.rec, flow])
    e.main()
    rep = json.loads(capsys.readouterr().out)
    assert "flow" not in rep and rep["scored"] == 2           # gated: scored (as a miss: not in graph), no withheld block


def test_an_unexpected_exception_class_is_exit_4_not_the_not_a_pass_exit_1(tmp_path, monkeypatch, capsys):
    e = ScoreEnv(tmp_path, monkeypatch)

    class Boom:
        def __getattr__(self, n):
            raise AttributeError(n)
    assert e.main(snapshot=Boom()) == 4 and "AttributeError" in capsys.readouterr().err
