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


def test_unit_anchor_is_asked_from_its_probe_fewest_items_then_smallest_cpid():
    c1, c2, c3 = (H.contract("mcp_tool", t) for t in ("alpha", "beta", "gamma"))
    ids = {t: G._mk.contract_id(c) for t, c in zip("abc", (c1, c2, c3))}
    unit = {"type": "unit", "repoId": "r", "rootPath": "svc", "lang": "python"}
    rec = H.item(1, unit, H.cfg("compose.yaml", "services.s"), serves=[c1, c2, c3])
    others = [H.item(10 + k, c, H.sym("a.py::f")) for k, c in enumerate((c1, c1, c3))]       # c1 has 2 items, c3 has 1, c2 has none
    probe = G.unit_probe(rec, [rec, *others])
    assert probe == ids["b"]
    assert G.build_request(rec, probe)["anchors"][0] == {"type": "contract", "cpId": ids["b"], "kind": "mcp_tool", "keyNorm": "beta"}
    tie = G.unit_probe(rec, [rec, others[0], H.item(20, c2, H.sym("a.py::f")), others[2]])    # c2 now has 1, c3 has 1, c1 has 1: smallest cpId
    assert tie == min(ids.values())


def test_unit_anchor_with_no_served_point_is_a_harness_error():
    rec = H.item(1, {"type": "unit", "repoId": "r", "rootPath": "s", "lang": "python"}, H.cfg("c.yaml", "a"))
    with pytest.raises(G.HarnessError):
        G.unit_probe(rec, [rec])


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


def test_write_custodian_refuses_the_repo_tree(tmp_path):
    with pytest.raises(G.Refusal):
        G.write_custodian(G._CPG.parent / "cpg" / "eval", "leak.json", {"x": 1}, home=tmp_path)


# ---------------------------------------------------------------- preflight on a throwaway repo
YAML_EM = "exclusions:\n  paths: [\"cpg/\"]\nprotocol:\n  documents: []\n  protocolString: \"\"\n"
YAML_SYS = "repos:\n  - exclusions:\n      paths: [\"cpg/\"]\n"
OBS = {"algorithm": {"id": "bfs-v1", "version": "1.2"}, "defaultDepthBound": 2}


def prepared(tmp_path):
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
    rec = {"schema": "ccg-prereg-v1", "stratum": "ac5-gate-1", "packSetHash": t["packSetHash"], "coreTreeHash": t["coreTreeHash"], **OBS}
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


# ---------------------------------------------------------------- CLI score: counts only, custodian outputs
def test_cli_score_prints_counts_only_and_writes_outcomes_to_the_custodian(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    cust = tmp_path / ".local" / "share" / "ccg-eval" / "e12"
    cust.mkdir(parents=True)
    w = H.World(qname="svc/SENTINEL_QNAME.py::SentinelFn")
    lists = cust / "ac5-http.jsonl"
    lists.write_text(json.dumps(w.rec) + "\n")
    run = cust / "run.json"
    run.write_text(json.dumps(w.run()))
    rc = G.main(["score", "--gate-id", "ac5-gate-1", "--lists", str(lists), "--run", str(run), "--custodian", str(cust)], snapshot=w.snap)
    out = capsys.readouterr().out
    assert rc == 1                                           # one item: minimums not met, so not a pass
    rep = json.loads(out)
    assert rep["verdict"] == "minimums-not-met" and rep["scored"] == 1
    for secret in ("SENTINEL", w.rec["itemId"], w.rec["anchorId"], w.cp, w.sid):
        assert secret not in out
    rows = [json.loads(l) for l in (cust / "gate-outcomes.jsonl").read_text().splitlines()]
    assert rows[0]["itemId"] == w.rec["itemId"] and rows[0]["outcome"] == "hit"


def test_cli_score_refuses_a_custodian_dir_outside_the_area(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = G.main(["score", "--gate-id", "g", "--lists", "x", "--run", "y", "--custodian", str(tmp_path / "elsewhere")])
    assert rc == 2 and "REFUSED" in capsys.readouterr().err


def test_cli_score_stops_before_scoring_on_a_stored_gold_renormalize_difference(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    cust = tmp_path / ".local" / "share" / "ccg-eval" / "e12"
    cust.mkdir(parents=True)
    w = H.World()
    stale = dict(w.rec, anchorId="0" * 32)
    (cust / "l.jsonl").write_text(json.dumps(stale) + "\n")
    (cust / "run.json").write_text(json.dumps(w.run()))
    rc = G.main(["score", "--gate-id", "g", "--lists", str(cust / "l.jsonl"), "--run", str(cust / "run.json"), "--custodian", str(cust)], snapshot=w.snap)
    assert rc == 2 and "renormalize" in capsys.readouterr().err and not (cust / "gate-outcomes.jsonl").exists()
