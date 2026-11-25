"""P1-32b gate harness, verification layer (``docs/plans/connected-code-graph-p1-gates.md`` P1-32b; EP §12.3, ``-p1-ml.md`` §6.1, §6.3).

Offline and synthetic: throwaway git repos, injected observations. No held-out data, no store."""
import json
from pathlib import Path

import pytest

import gate_helpers as H
from gate_helpers import G

OBS = {"packSetHash": "p" * 64, "coreTreeHash": "c" * 64, "algorithm": {"id": "bfs-v1", "version": "1.2"}, "defaultDepthBound": 2,
       "dirty": [], "clean": True}
REC = {"schema": "ccg-prereg-v1", "stratum": "ac5-gate-1", "packSetHash": "p" * 64, "coreTreeHash": "c" * 64,
       "algorithm": {"id": "bfs-v1", "version": "1.2"}, "defaultDepthBound": 2}


# ---------------------------------------------------------------- four-field comparison and the first-contact refusal
def test_all_four_equal_and_clean_is_first_contact():
    v = G.first_contact(REC, OBS)
    assert v["firstContact"] is True and v["label"] == "first-contact" and v["fieldsDiffering"] == []


@pytest.mark.parametrize("field,value", [
    ("packSetHash", "q" * 64), ("coreTreeHash", "d" * 64),
    ("algorithm", {"id": "bfs-v1", "version": "1.1"}), ("algorithm", {"id": "bfs-v2", "version": "1.2"}),
    ("algorithm", {"id": "bfs-v1", "version": "1.20"}), ("defaultDepthBound", 3), ("defaultDepthBound", "2")])
def test_each_prereg_field_differing_refuses_first_contact(field, value):
    v = G.first_contact(REC, {**OBS, field: value})
    assert v["firstContact"] is False and v["label"] == "after-adaptation" and v["fieldsDiffering"] == [field]


def test_a_dirty_cpg_tree_refuses_first_contact_though_all_four_equal():
    v = G.first_contact(REC, {**OBS, "dirty": [" M cpg/ccg/a.py"], "clean": False})
    assert v["firstContact"] is False and v["fieldsDiffering"] == [] and v["dirtyPaths"] == 1


def test_a_dirty_list_alone_refuses_even_if_clean_flag_is_missing():
    obs = {k: v for k, v in OBS.items() if k != "clean"}
    assert G.first_contact(REC, {**obs, "dirty": ["?? cpg/x"]})["firstContact"] is False


def test_a_dirty_list_with_a_contradicting_clean_flag_still_refuses():
    assert G.first_contact(REC, {**OBS, "dirty": ["?? cpg/x"], "clean": True})["firstContact"] is False
    assert G.first_contact(REC, {**OBS, "dirty": [], "clean": False})["firstContact"] is False


def test_a_missing_observed_field_is_a_difference_not_a_pass():
    obs = {k: v for k, v in OBS.items() if k != "defaultDepthBound"}
    assert G.first_contact(REC, obs)["fieldsDiffering"] == ["defaultDepthBound"]


# ---------------------------------------------------------------- the tree observation (real hashes, throwaway repo)
def test_observe_tree_clean_then_dirty_under_any_cpg_path(tmp_path):
    repo = H.make_repo(tmp_path)
    clean = G.observe_tree(repo)
    assert clean["clean"] is True and clean["dirty"] == []
    (repo / "cpg" / "README.md").write_text("# edited\n")          # an .md under cpg/: not in coreTreeHash, but still an uncommitted change
    d = G.observe_tree(repo)
    assert d["clean"] is False and d["coreTreeHash"] == clean["coreTreeHash"]
    assert G.first_contact({**REC, "packSetHash": d["packSetHash"], "coreTreeHash": d["coreTreeHash"]}, {**OBS, **d})["firstContact"] is False


def test_observe_tree_untracked_file_under_cpg_is_dirty(tmp_path):
    repo = H.make_repo(tmp_path)
    (repo / "cpg" / "eval" / "stray.txt").write_text("x")
    assert G.observe_tree(repo)["clean"] is False


def test_observe_tree_hashes_follow_the_working_tree(tmp_path):
    repo = H.make_repo(tmp_path)
    a = G.observe_tree(repo)
    (repo / "cpg/ccg/a.py").write_text("A = 2\n")
    b = G.observe_tree(repo)
    assert b["coreTreeHash"] != a["coreTreeHash"] and b["packSetHash"] == a["packSetHash"]
    H.git(repo, "checkout", "--", "cpg/ccg/a.py")
    (repo / "cpg/packs/p/pack.yaml").write_text("rules: [1]\n")
    c = G.observe_tree(repo)
    assert c["packSetHash"] != a["packSetHash"] and c["coreTreeHash"] == a["coreTreeHash"]


# ---------------------------------------------------------------- the committed record
def _prereg(tmp_path, recs):
    p = tmp_path / "prereg.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def test_read_prereg_returns_the_committed_record(tmp_path):
    p = _prereg(tmp_path, [{"schema": "other"}, REC])
    got = G.read_prereg(p, "ac5-gate-1", committed_text=lambda _p: p.read_text())
    assert got["packSetHash"] == "p" * 64


def test_read_prereg_refuses_an_uncommitted_record(tmp_path):
    p = _prereg(tmp_path, [REC])
    with pytest.raises(G.Refusal, match="not committed"):
        G.read_prereg(p, "ac5-gate-1", committed_text=lambda _p: "")


@pytest.mark.parametrize("recs", [[], [{**REC, "stratum": "ac5-gate-2"}], [REC, REC]])
def test_read_prereg_needs_exactly_one_record_for_the_gate(tmp_path, recs):
    p = _prereg(tmp_path, recs)
    with pytest.raises(G.Refusal, match="exactly one"):
        G.read_prereg(p, "ac5-gate-1", committed_text=lambda _p: p.read_text())


def test_read_prereg_refuses_a_record_without_a_compared_field(tmp_path):
    bad = {k: v for k, v in REC.items() if k != "algorithm"}
    p = _prereg(tmp_path, [bad])
    with pytest.raises(G.Refusal, match="algorithm"):
        G.read_prereg(p, "ac5-gate-1", committed_text=lambda _p: p.read_text())


# ---------------------------------------------------------------- stored-gold renormalize-check
def test_renormalize_stored_gold_clean_and_difference():
    anchor = H.contract("http", "GET /a/{id}")
    good = H.item(1, anchor, H.sym("a.py::f"))
    assert G.renormalize_stored_gold([good])["differences"] == 0
    stale = dict(good, anchorId="0" * 32)
    rep = G.renormalize_stored_gold([stale])
    assert rep["differences"] == 1 and rep["checked"] == 1
    with pytest.raises(G.Refusal, match="stops before scoring"):
        G.require_clean_renormalize(rep)


def test_renormalize_ignores_symbol_anchors_and_checks_contract_impacted():
    sym_item = H.item(1, H.sym("a.py::f"), H.sym("b.py::g"))
    assert G.renormalize_stored_gold([sym_item])["checked"] == 0
    c_item = H.item(2, H.sym("a.py::f"), H.contract("mcp_tool", "t"))
    assert G.renormalize_stored_gold([c_item])["checked"] == 1       # impactedId is md5 of the locator here, not a contract id
    assert G.renormalize_stored_gold([c_item])["differences"] == 1


# ---------------------------------------------------------------- held-out manifest
def test_verify_heldout_ok_and_modified(tmp_path):
    from split_manifest import build_manifest
    d = tmp_path / "ho"
    d.mkdir()
    for s in G.STRATA + ("reserve",):
        (d / f"ac5-{s}.jsonl").write_text(json.dumps({"itemId": s, "counting": True, "anchorId": s}) + "\n")
    ecf = tmp_path / "EVAL_COMMIT"
    ecf.write_text("e" * 40)
    m = build_manifest(d, "e" * 40, "proto", 0)
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(m))
    assert G.verify_heldout(mp, d, ecf) == {"ok": True, "problems": []}
    (d / "ac5-http.jsonl").write_text("x\n")
    bad = G.verify_heldout(mp, d, ecf)
    assert bad["ok"] is False and any("ac5-http.jsonl" in p for p in bad["problems"])
    ecf.write_text("f" * 40)
    assert any("evalCommit" in p for p in G.verify_heldout(mp, d, ecf)["problems"])


# ---------------------------------------------------------------- R-6 pin strings
def _doc(tmp_path, rel, version):
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"# T\n\n> **Status:** active · **Version:** {version} · **Owner:** x\n")


def test_pin_report_reports_a_difference_and_does_not_edit(tmp_path):
    _doc(tmp_path, "docs/a.md", "1.9")
    _doc(tmp_path, "docs/b.md", "1.5")
    mf = {"protocol": {"documents": [{"path": "docs/a.md", "version": "1.4"}, {"path": "docs/b.md", "version": "1.5"}],
                       "protocolString": "docs/a.md@1.4+docs/b.md@1.5"}}
    before = (tmp_path / "docs/a.md").read_text()
    rep = G.pin_report(mf, tmp_path, {"protocol": "docs/a.md@1.4+docs/b.md@1.5"})
    assert rep["differences"] == [{"path": "docs/a.md", "recorded": "1.4", "committed": "1.9"}]
    assert rep["protocolStringConsistent"] is True and rep["heldoutProtocolEqual"] is True
    assert (tmp_path / "docs/a.md").read_text() == before


def test_pin_report_flags_an_inconsistent_protocol_string_and_a_missing_doc(tmp_path):
    mf = {"protocol": {"documents": [{"path": "docs/gone.md", "version": "1"}], "protocolString": "other"}}
    rep = G.pin_report(mf, tmp_path, {"protocol": "nope"})
    assert rep["protocolStringConsistent"] is False and rep["heldoutProtocolEqual"] is False
    assert rep["differences"][0]["committed"] is None


# ---------------------------------------------------------------- harness self-pin
def test_harness_pin_same_then_edit_after_record_is_after_adaptation(tmp_path):
    repo = H.make_repo(tmp_path)
    rec = H.head(repo)
    files = ("cpg/eval/tools/gate.py", "cpg/eval/tools/match_keys.py")
    assert G.harness_pin(repo, rec, files)["afterAdaptation"] is False
    (repo / files[0]).write_text("# gate v2\n")
    pin = G.harness_pin(repo, rec, files)
    assert pin["afterAdaptation"] is True and pin["differ"] == [files[0]] and pin["files"][files[1]]["same"] is True


def test_harness_pin_edit_in_a_committed_later_commit_still_differs(tmp_path):
    repo = H.make_repo(tmp_path)
    rec = H.head(repo)
    (repo / "cpg/eval/tools/match_keys.py").write_text("# mk 2\n")
    H.git(repo, "commit", "-qam", "later")
    assert G.harness_pin(repo, rec, ("cpg/eval/tools/match_keys.py",))["afterAdaptation"] is True


def test_harness_pin_rescored_unchanged_exception_is_reported_not_silent(tmp_path):
    repo = H.make_repo(tmp_path)
    rec = H.head(repo)
    (repo / "cpg/eval/tools/gate.py").write_text("# fix\n")
    pin = G.harness_pin(repo, rec, ("cpg/eval/tools/gate.py",), rescored_unchanged=True)
    assert pin["afterAdaptation"] is False and pin["rescoreException"] is True and pin["differ"] == ["cpg/eval/tools/gate.py"]
    assert G.harness_pin(repo, rec, ("cpg/eval/tools/gate.py",), rescored_unchanged=False)["afterAdaptation"] is True


def test_harness_pin_a_file_the_record_lacks_is_a_difference(tmp_path):
    repo = H.make_repo(tmp_path)
    rec = H.head(repo)
    (repo / "cpg/eval/tools/new.py").write_text("x\n")
    assert G.harness_pin(repo, rec, ("cpg/eval/tools/new.py",))["afterAdaptation"] is True


# ---------------------------------------------------------------- M3: the self-pin is built from the import graph
def _tools(tmp_path, files):
    d = tmp_path / "tools"
    d.mkdir()
    for name, text in files.items():
        (d / name).write_text(text)
    return d


def test_m3_the_real_harness_pin_covers_the_run_wiring_and_the_snapshot_transport():
    names = {Path(f).name for f in G.HARNESS_FILES}
    assert {"gate.py", "gate_run.py", "snapshot_export.py", "match_keys.py", "split_manifest.py", "noise.py", "oracle.py", "ac24_smoke.py",
            "export_kit.py", "baseline.py", "withheld_shapes.py"} <= names
    assert "cpg/eval/EVAL_COMMIT" in G.HARNESS_FILES and all(f.startswith("cpg/eval/") for f in G.HARNESS_FILES)


@pytest.fixture(params=[
    ("top-level", "import planted\n"),
    ("from-import", "from planted import x\n"),
    ("lazy-in-function", "def f():\n    import planted\n"),
    ("lazy-from-in-function", "def f():\n    from planted import x\n"),
    ("sys.path then import", "import sys\nsys.path.insert(0, '.')\nimport planted\n"),
    ("import_module", "import importlib\nimportlib.import_module('planted')\n"),
    ("__import__", "__import__('planted')\n"),
    ("relative", "from . import planted\n"),
], ids=lambda p: p[0])
def axis_text(request):
    return request.param


@pytest.mark.parametrize("entry", ["gate.py", "gate_run.py"])
def test_m3_a_new_import_in_either_entry_point_joins_the_pin(tmp_path, entry, axis_text):
    axis, text = axis_text
    d = _tools(tmp_path, {"gate.py": "x = 1\n", "gate_run.py": "y = 1\n", "planted.py": "z = 1\n", "other.py": "w = 1\n"})
    before = {Path(f).name for f in G.harness_files(d)}
    assert "planted.py" not in before
    (d / entry).write_text(text)
    after = {Path(f).name for f in G.harness_files(d)}
    assert "planted.py" in after, axis
    assert "other.py" not in after


def test_m3_the_closure_is_transitive_and_ignores_names_outside_the_tools_dir(tmp_path):
    d = _tools(tmp_path, {"gate.py": "import json\nimport a\n", "gate_run.py": "", "a.py": "def f():\n    import b\n", "b.py": "import os\n", "c.py": ""})
    assert sorted(Path(f).name for f in G.harness_files(d)) == ["EVAL_COMMIT", "a.py", "b.py", "gate.py", "gate_run.py"]


def test_m3_a_pinned_file_edited_after_the_record_is_after_adaptation(tmp_path):
    repo = H.make_repo(tmp_path)
    (repo / "cpg/eval/tools/gate_run.py").write_text("# run v1\n")
    H.git(repo, "add", "-A")
    H.git(repo, "commit", "-qm", "run")
    rec = H.head(repo)
    (repo / "cpg/eval/tools/gate_run.py").write_text("# run v2\n")
    assert G.harness_pin(repo, rec, ("cpg/eval/tools/gate_run.py",))["afterAdaptation"] is True


def test_rescore_changes_lists_only_changed_items():
    assert G.rescore_changes({"a": "hit", "b": "miss"}, {"a": "hit", "b": "hit", "c": "hit"}) == ["b", "c"]


def test_record_commit_finds_the_commit_that_added_the_line(tmp_path):
    repo = H.make_repo(tmp_path)
    line = json.dumps(REC)
    (repo / "cpg/eval/preregistration.jsonl").write_text(line + "\n")
    H.git(repo, "add", "-A")
    H.git(repo, "commit", "-qm", "record")
    rec = H.head(repo)
    (repo / "cpg/eval/preregistration.jsonl").write_text(line + "\n" + json.dumps({"x": 1}) + "\n")
    H.git(repo, "commit", "-qam", "append")
    assert G.record_commit(repo, "cpg/eval/preregistration.jsonl", line) == rec


# ---------------------------------------------------------------- scope
EXCL = {"exclusions": {"paths": ["cpg/", "docs/BACKLOG.md"], "contentCheck": {"files": [{"path": "claude/x.md", "class": "doc"}]}}}


def test_scope_patterns_include_content_check_files():
    assert G.scope_patterns(EXCL) == ["cpg/", "docs/BACKLOG.md", "claude/x.md"]


def test_scope_report_flags_out_of_scope_held_out_item_as_key_error_never_dropped():
    inside = H.item(1, H.contract("http", "GET /a"), H.sym("svc/a.py::f"))
    outside = H.item(2, H.contract("http", "GET /b"), H.sym("cpg/ccg/x.py::g"))
    rep = G.scope_report([inside, outside], G.scope_patterns(EXCL))
    assert rep == {"outOfScope": 1, "triage": {"key-error": 1}, "itemIds": [outside["itemId"]]}


def test_scope_disagreement_between_eval_and_system_manifest_is_reported():
    same = {"repos": [{"exclusions": {"paths": ["cpg/", "docs/BACKLOG.md"], "contentCheck": {"files": ["claude/x.md"]}}}]}
    assert G.scope_disagreement(EXCL, same)["agree"] is True
    drift = {"repos": [{"exclusions": {"paths": ["cpg/"], "contentCheck": {"files": ["claude/x.md"]}}}]}
    d = G.scope_disagreement(EXCL, drift)
    assert d["agree"] is False and d["onlyInEval"] == 1 and d["onlyInSystem"] == 0


# ---------------------------------------------------------------- build at EVAL_COMMIT
def test_build_at_eval_commit_clones_checks_out_and_calls_refresh(tmp_path):
    repo = H.make_repo(tmp_path)
    ev = H.head(repo)
    (repo / "cpg/ccg/a.py").write_text("A = 9\n")
    H.git(repo, "commit", "-qam", "later")
    calls = []
    dest = tmp_path / "custodian" / "build"
    dest.parent.mkdir()
    out = G.build_at_eval_commit(ev, dest, repo, refresh_fn=lambda system, repo_root, **kw: (calls.append((system, Path(repo_root))), H.clean_report())[1])
    assert out == dest and H.head(dest) == ev and (dest / "cpg/ccg/a.py").read_text() == "A = 1\n"
    assert calls == [("ccg", dest)]
    assert (repo / "cpg/ccg/a.py").read_text() == "A = 9\n"                 # the user's tree is untouched


def test_build_at_eval_commit_refuses_a_non_hex_commit_and_an_existing_dir(tmp_path):
    repo = H.make_repo(tmp_path)
    with pytest.raises(G.Refusal, match="40-hex"):
        G.build_at_eval_commit("main", tmp_path / "b", repo, refresh_fn=lambda *a, **k: None)
    (tmp_path / "b").mkdir()
    with pytest.raises(G.Refusal, match="exists"):
        G.build_at_eval_commit(H.head(repo), tmp_path / "b", repo, refresh_fn=lambda *a, **k: None)
