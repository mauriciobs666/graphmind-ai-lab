"""P1-05 (5b) — bundle assembly and writing (`extractors/joern/normalize/assemble_bundle.py`, `blame.py`).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-05 behaviours (3), (4), (5), (5b), (6), (7) and the determinism
finding of the P1-04 gate (raw `d`/`cb` order varies between Joern runs). Rules: `-bundle-spec2.md` §1.4 (G-4), `-spec4.md` §2
(empty file, `ownedRegions`, `use == def`), spec3 §5. The facts come from `normalize_assembly_helpers.py` (a class defined
twice, repeated `(point, cond)` pairs, a `use == def` edge, two raw `data` edges with one id); a real git repository holds the
source, so `git blame` runs for real. Expected values are written by hand or read from the output's own records, never
recomputed with `ccg.ids`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ccg import atrest, bundle, ids
from extractors.joern import run as joern_run
from extractors.joern.normalize import assemble_bundle, blame, core, python
from normalize_assembly_helpers import COMMIT_DATE, KINDS, SRC, full_facts, git_repo, shuffled

UUID = "11111111-1111-4111-8111-111111111111"
BUILT_AT = "2026-01-01T00:00:00Z"
COMMIT_Z = "2026-01-02T03:04:05Z"


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    path = tmp_path_factory.mktemp("repo")
    return path, git_repo(path, {"r.py": SRC})


def assemble(repo, facts=None, *, root=".", **kw):
    path, commit = repo
    kw.setdefault("built_at", BUILT_AT)
    return assemble_bundle.assemble(lang="python", unit_root=root, repo_id="r", facts=facts or full_facts(), stage=path,
                                    astgen_dir=None, repo=path, commit=commit, audit_ids=[UUID],
                                    extract_script=joern_run.EXTRACT_SC, **kw)


def of(records, kind):
    return [r for r in records if r["type"] == kind]


def qnames(records):
    return {r["id"]: r["qname"] for r in of(records, "symbol")}


# --- decorators belong to their owner (gate P2-2) --------------------------------------------------------------------
def decorators(records):
    q = qnames(records)
    return sorted((q[r["targetId"]], r["ordinal"], r["name"]) for r in of(records, "decorator"))


WANT_DECORATORS = [("r.py::B#2.x.<get>", 0, "property"), ("r.py::B#2.x.<set>", 0, "x.setter"), ("r.py::B.x.<get>", 0, "property")]


def test_each_of_two_same_named_members_of_a_redefined_class_gets_its_own_decorator_record(repo):
    records, _ = assemble(repo)
    assert decorators(records) == WANT_DECORATORS
    assert len({r["id"] for r in of(records, "decorator")}) == 3


@pytest.mark.parametrize("seed", range(40))
def test_decorator_records_do_not_depend_on_the_order_of_deco_m_and_t(repo, seed):
    records, _ = assemble(repo, shuffled(full_facts(), seed, kinds=("deco", "m", "t")))
    assert decorators(records) == WANT_DECORATORS


def test_a_call_that_takes_a_lambda_is_a_decorator_of_the_lambda_and_does_not_widen_its_span(tmp_path):
    src = "def run(f):\n    pass\n\n\nx = run(\n    lambda: 1)\n"
    path = tmp_path / "repo"
    commit = git_repo(path, {"l.py": src})
    from normalize_helpers import py_facts
    facts = py_facts({"l.py": src})
    lam = next(m for m in facts["m"] if m["name"] == "<lambda>0")
    assert lam["l"] == 6
    facts["deco"].append({"k": "deco", "m": lam["id"], "target": lam["full"], "o": 0, "name": "run", "code": "run(lambda: 1)",
                          "l": 5, "c": 5, "args": []})
    records, _ = assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None,
                                          repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC)
    [deco] = of(records, "decorator")
    sym = next(r for r in of(records, "symbol") if r["id"] == deco["targetId"])
    assert (sym["kind"], deco["name"], deco["line"], sym["lineStart"]) == ("lambda", "run", 5, 6)


# --- determinism: shuffles of every kind assemble reads --------------------------------------------------------------
def bundle_bytes(records):
    return bundle.dumps(records)


@pytest.fixture(scope="module")
def reference(repo):
    return bundle_bytes(assemble(repo)[0])


@pytest.mark.parametrize("seed", range(40))
def test_the_bundle_is_byte_identical_under_shuffles_of_every_kind_of_fact(repo, reference, seed):
    assert bundle_bytes(assemble(repo, shuffled(full_facts(), seed))[0]) == reference


@pytest.mark.parametrize("kind", KINDS)
def test_the_bundle_is_byte_identical_when_any_one_kind_is_shuffled_alone(repo, reference, kind):
    for seed in range(6):
        assert bundle_bytes(assemble(repo, shuffled(full_facts(), seed, kinds=(kind,)))[0]) == reference


def test_the_fixture_has_the_shapes_the_determinism_claim_needs(repo):
    f = full_facts()
    bodies = [m["full"] for m in f["m"] if m["name"] == "<body>"]
    assert len(bodies) == 2 and bodies[0] == bodies[1]                       # a class defined twice
    pairs = [(e["pid"], e["cond"]) for e in f["cb"]]
    assert len(pairs) != len(set(pairs))                                     # repeated (point, cond) pairs
    assert any(d["uk"] == "cs" and d["dk"] == "cs" and d["uid"] == d["did"] for d in f["d"])    # use == def
    assert any(e["pid"] == e["cond"] for e in f["cb"])                       # a CDG self-edge
    shuffled_orders = {tuple(e["pid"] + e["cond"] for e in shuffled(f, s)["cb"]) for s in range(8)}
    assert len(shuffled_orders) > 1                                          # and the shuffle really reorders


def test_two_raw_data_edges_with_one_id_keep_the_one_that_sorts_first_whatever_their_file_order(repo):
    for seed in range(12):
        records, report = assemble(repo, shuffled(full_facts(), seed, kinds=("d",)))
        a0 = [d for d in of(records, "data") if d["toPath"].startswith("a0.a.b.c")]
        assert [(d["status"], d["reason"]) for d in a0] == [("possible", "field-truncated")]
        assert report["dataDropped"] == {"duplicate": 1}


def test_a_data_edge_whose_use_is_its_own_def_is_kept_with_distinct_paths(repo):
    records, _ = assemble(repo)
    [selfedge] = [d for d in of(records, "data") if d["use"] == d["def"]]
    assert (selfedge["toPath"], selfedge["fromPath"], selfedge["status"], selfedge["reason"]) == ("recv", "r", "confirmed", "intra-flow")


# --- G-4 --------------------------------------------------------------------------------------------------------------
def test_g4_filter_drops_self_edges_and_repeated_pairs_and_keeps_first_seen_order():
    assert assemble_bundle.g4_filter([("a", "c"), ("b", "b"), ("a", "c"), ("b", "c"), ("c", "c"), ("a", "c")]) == (
        [("a", "c"), ("b", "c")], 2, 2)


def test_a_controlled_by_record_with_point_equal_cond_is_dropped_and_repeated_pairs_collapse(repo):
    records, report = assemble(repo)
    conds = {r["id"]: r["code"] for r in of(records, "cond")}
    calls = {r["id"]: r["calleeName"] for r in of(records, "callsite")}
    pairs = sorted((calls.get(r["point"], conds.get(r["point"])), conds[r["cond"]]) for r in of(records, "controlledBy"))
    assert pairs == [("g", "self.a"), ("g", "self.b"), ("self.b", "self.a")]
    assert all(r["point"] != r["cond"] for r in of(records, "controlledBy"))
    assert report["controlledBy"] == {"rawEdges": 6, "selfEdgesDropped": 1, "duplicatePairsDropped": 2, "unmapped": 0,
                                      "written": 3, "condToCond": 1}


def test_a_self_edge_in_a_bundle_is_rejected_by_the_validator_so_the_filter_is_load_bearing(repo):
    records, _ = assemble(repo)
    cond = of(records, "cond")[0]
    bad = records + [{"type": "controlledBy", "id": "0" * 32, "point": cond["id"], "cond": cond["id"]}]
    with pytest.raises(bundle.BundleError):
        bundle.dumps(bad)


# --- (6) validation, (7) the writer ------------------------------------------------------------------------------------
def build(repo, tmp_path, **kw):
    path, commit = repo
    facts_file = tmp_path / "facts.jsonl"
    facts_file.write_text("".join(json.dumps(r) + "\n" for k in KINDS for r in full_facts()[k]), encoding="utf-8")
    from ccg.units import Unit
    from ccg import ids
    unit = Unit(ids.unit_id("r", ".", "python"), "r", ".", "python")
    raw = joern_run.RawFacts(unit, commit, facts_file, (UUID,), path, tmp_path, None, tmp_path / "x", 1)
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    kw.setdefault("built_at", BUILT_AT)
    return unit, assemble_bundle.build_bundle(unit, raw, path, commit, system="sys1", root=work, **kw), work


def test_every_bundle_build_bundle_writes_passes_the_validator_and_reads_back(repo, tmp_path):
    _, written, _ = build(repo, tmp_path)
    records = bundle.read_bundle(written)                       # parses and validates (ccg.bundle.loads)
    assert records[0]["type"] == "header" and records[0]["history"] == "available"
    assert records[0]["counts"]["symbol"] == len(of(records, "symbol"))


def test_build_bundle_fills_the_report_it_is_given(repo, tmp_path):
    report = {}
    build(repo, tmp_path, report=report)
    assert report["symbols"] == 7 and report["filesParsed"] == 1 and report["counts"]["symbol"] == 7


def test_the_header_names_the_frontend_of_the_language_and_the_fixed_projection_fields(tmp_path, repo):
    records, _ = assemble(repo)
    h = records[0]
    assert (h["tools"]["frontend"], h["flowProjected"], h["egressMode"], h["fieldDepthCap"]) == ("pysrc2cpg", True, "offline", 3)
    ts, _ = assemble_ts(tmp_path, None)
    assert ts[0]["tools"]["frontend"] == "jssrc2cpg"


def test_a_built_tree_that_git_cannot_resolve_is_an_error_not_an_empty_tree(tmp_path):
    path = tmp_path / "repo"
    commit = git_repo(path, {"r.py": SRC})
    empty = {k: [] for k in KINDS}
    with pytest.raises(subprocess.CalledProcessError):
        assemble_bundle.assemble(lang="python", unit_root="missing", repo_id="r", facts=empty, stage=path, astgen_dir=None, repo=path,
                                 commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)


def test_the_bundle_is_written_under_bundles_system_as_unit_dot_build_with_private_modes_even_under_umask_0(repo, tmp_path):
    old = os.umask(0)
    try:
        unit, written, work = build(repo, tmp_path)
    finally:
        os.umask(old)
    assert written == work / "bundles" / "sys1" / f"{unit.unitId}.{repo[1]}.bundle.jsonl"
    assert oct(written.stat().st_mode & 0o777) == "0o600"
    assert oct(written.parent.stat().st_mode & 0o777) == "0o700"


def test_the_only_writer_is_ccg_atrest_write_bundle(repo, tmp_path, monkeypatch):
    calls = []

    def spy(system, unit, build_id, records, root=None):
        calls.append((system, unit, build_id, records[0]["type"], root))
        return tmp_path / "sentinel"

    monkeypatch.setattr(atrest, "write_bundle", spy)
    unit, written, work = build(repo, tmp_path, build="b42")
    assert written == tmp_path / "sentinel"
    assert calls == [("sys1", unit.unitId, "b42", "header", work)]


def test_the_build_defaults_to_the_commit_and_a_raw_extraction_of_another_commit_is_refused(repo, tmp_path):
    path, commit = repo
    unit, written, _ = build(repo, tmp_path)
    assert written.name.endswith(f".{commit}.bundle.jsonl")
    from ccg.units import Unit
    raw = joern_run.RawFacts(unit, "f" * 40, tmp_path / "facts.jsonl", (UUID,), path, tmp_path, None, tmp_path / "x", 1)
    with pytest.raises(assemble_bundle.BuildError, match="extracted at"):
        assemble_bundle.build_bundle(unit, raw, path, commit, system="sys1", root=tmp_path / "work")


# --- (4) the interpreter, (5) history ----------------------------------------------------------------------------------
def test_assemble_refuses_any_interpreter_but_cpython_3_12_before_it_touches_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 13, 0, "final", 0))
    with pytest.raises(core.RuntimeRefused):
        assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=full_facts(), stage=tmp_path / "none",
                                 astgen_dir=None, repo=tmp_path / "none", commit="0" * 40, audit_ids=[UUID],
                                 extract_script=tmp_path / "none")


@pytest.fixture(scope="module")
def two_commits(tmp_path_factory):
    path = tmp_path_factory.mktemp("full")
    git_repo(path, {"r.py": SRC})
    env = {**os.environ, "GIT_AUTHOR_DATE": COMMIT_DATE, "GIT_COMMITTER_DATE": COMMIT_DATE, "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x.invalid"}
    (path / "other.txt").write_text("x\n")
    for a in (("add", "-A"), ("commit", "-q", "-m", "c2")):
        subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True, env=env)
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    shallow = tmp_path_factory.mktemp("shallow") / "clone"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{path}", str(shallow)], check=True, capture_output=True, env=env)
    return path, shallow, head


def test_a_shallow_clone_is_a_build_error_and_a_full_one_is_not(two_commits):
    full, shallow, head = two_commits
    with pytest.raises(blame.HistoryError, match="shallow"):
        assemble((shallow, head))
    records, _ = assemble((full, head))
    assert records[0]["history"] == "available"


def test_a_shallow_clone_is_refused_for_javascript_too(two_commits):
    full, shallow, head = two_commits
    with pytest.raises(blame.HistoryError):
        assemble_bundle.assemble(lang="javascript", unit_root=".", repo_id="r", facts=full_facts(), stage=shallow, astgen_dir=None,
                                 repo=shallow, commit=head, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC)


# --- (3) the empty file and the span-plus-ownedRegions fallback ---------------------------------------------------------
@pytest.fixture(scope="module")
def history(tmp_path_factory):
    """Three commits over `h.py` (3 lines) and an empty `e.py`: c1 creates both, c2 rewrites line 2, c3 line 3."""
    path = tmp_path_factory.mktemp("hist")
    stamp = ["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"]
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@x.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x.invalid"}
    commits = []
    texts = ["a = 1\nb = 2\nc = 3\n", "a = 1\nb = 22\nc = 3\n", "a = 1\nb = 22\nc = 33\n"]
    subprocess.run(["git", "-C", str(path), "init", "-q", "-b", "main"], check=True, capture_output=True, env=env)
    for i, text in enumerate(texts):
        (path / "h.py").write_text(text)
        (path / "e.py").write_text("")
        e = {**env, "GIT_AUTHOR_DATE": stamp[i], "GIT_COMMITTER_DATE": stamp[i]}
        if i == 2:
            (path / "e.py").write_text("")
        subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, capture_output=True, env=e)
        subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", f"c{i + 1}", "--allow-empty"], check=True,
                       capture_output=True, env=e)
        commits.append(subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                                      text=True).stdout.strip())
    return path, commits, ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"]


def empty_and_h_facts():
    from normalize_helpers import py_facts
    return py_facts({"h.py": "a = 1\nb = 22\nc = 33\n", "e.py": ""})


def run_history(history, facts=None, patch=None):
    path, commits, _ = history
    return assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts or empty_and_h_facts(), stage=path,
                                    astgen_dir=None, repo=path, commit=commits[2], audit_ids=[UUID],
                                    extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)


def test_an_empty_file_has_line_count_0_a_one_line_module_and_the_path_last_commit_as_its_history(history):
    records, report = run_history(history)
    [f] = [r for r in of(records, "file") if r["path"] == "e.py"]
    [mod] = [r for r in of(records, "symbol") if r["path"] == "e.py"]
    assert f["lineCount"] == 0
    assert (mod["kind"], mod["lineStart"], mod["lineEnd"]) == ("module", 1, 1)
    assert mod["fingerprint"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"     # sha256 of the empty text
    # e.py has no blamed line; its last touching commit is c1 (c2 and c3 leave it alone)
    assert (mod["lastChangedAt"], mod["lastChangedCommit"]) == (history[2][0], history[1][0])
    assert report["blameFallbackFiles"] == ["e.py"]


def test_a_file_with_blamed_lines_never_takes_the_fallback(history):
    records, report = run_history(history)
    [mod] = [r for r in of(records, "symbol") if r["path"] == "h.py"]
    assert (mod["lastChangedAt"], mod["lastChangedCommit"]) == (history[2][2], history[1][2])      # line 3 changed in c3
    assert report["blameFallbackFiles"] == ["e.py"]


def patched_scopes(monkeypatch, **attrs):
    real = python.build_scopes

    def build(B):
        scopes, m2s, deco = real(B)
        for s in scopes.values():
            if s.file == "h.py" and s.kind == "module":
                for k, v in attrs.items():
                    setattr(s, k, v)
        return scopes, m2s, deco

    monkeypatch.setattr(python, "build_scopes", build)


def test_the_owned_regions_of_a_symbol_count_for_its_last_change(history, monkeypatch):
    patched_scopes(monkeypatch, lineEnd=2, owned=[(3, 3)])           # span lines 1-2 (last touched in c2) plus line 3 (c3)
    records, _ = run_history(history)
    [mod] = [r for r in of(records, "symbol") if r["path"] == "h.py"]
    assert (mod["lineStart"], mod["lineEnd"]) == (1, 2)
    assert (mod["lastChangedAt"], mod["lastChangedCommit"]) == (history[2][2], history[1][2])


def test_without_owned_regions_the_same_span_is_last_changed_in_c2(history, monkeypatch):
    patched_scopes(monkeypatch, lineEnd=2)
    records, _ = run_history(history)
    [mod] = [r for r in of(records, "symbol") if r["path"] == "h.py"]
    assert (mod["lastChangedAt"], mod["lastChangedCommit"]) == (history[2][1], history[1][1])


def test_the_fallback_needs_no_blamed_line_in_the_span_and_in_its_owned_regions(history, monkeypatch):
    # a span past the end of the file has no blamed line; an owned region on line 2 does (c2), so no fallback to the path's
    # last commit (c3 would be touched by `git log`); with no owned region the fallback is the path's last commit
    patched_scopes(monkeypatch, line=50, lineEnd=51, owned=[(2, 2)])
    _, report = run_history(history)
    assert report["blameFallbackFiles"] == ["e.py"]                      # h.py did not fall back
    records, _ = run_history(history)
    [mod] = [r for r in of(records, "symbol") if r["path"] == "h.py"]
    assert (mod["lastChangedAt"], mod["lastChangedCommit"]) == (history[2][1], history[1][1])


def test_the_fallback_is_the_paths_last_commit_when_neither_span_nor_regions_have_a_blamed_line(history, monkeypatch):
    patched_scopes(monkeypatch, line=50, lineEnd=51)
    records, report = run_history(history)
    [mod] = [r for r in of(records, "symbol") if r["path"] == "h.py"]
    assert (mod["lastChangedAt"], mod["lastChangedCommit"]) == (history[2][2], history[1][2])
    assert sorted(report["blameFallbackFiles"]) == ["e.py", "h.py"]


# --- blame.py -----------------------------------------------------------------------------------------------------------
def test_blame_maps_each_line_to_its_commit_and_each_commit_to_its_committer_time(history):
    path, commits, stamps = history
    per, times = blame.blame(path, commits[2], "h.py")
    assert per == {1: commits[0], 2: commits[1], 3: commits[2]}
    assert times == {commits[0]: 1767225600, commits[1]: 1769904000, commits[2]: 1772323200}


def test_last_changed_is_the_latest_committer_time_over_all_regions_and_ties_take_the_lower_commit_id():
    per = {1: "bb", 2: "aa", 3: "cc"}
    times = {"bb": 100, "aa": 100, "cc": 50}
    assert blame.last_changed(per, times, [(1, 3)]) == ("1970-01-01T00:01:40Z", "aa")
    assert blame.last_changed(per, times, [(3, 3)]) == ("1970-01-01T00:00:50Z", "cc")
    assert blame.last_changed(per, times, [(3, 3), (1, 1)]) == ("1970-01-01T00:01:40Z", "bb")
    assert blame.last_changed(per, times, [(1, 1), (3, 3)]) == ("1970-01-01T00:01:40Z", "bb")


def test_last_changed_over_lines_that_are_not_blamed_is_a_value_error():
    with pytest.raises(ValueError, match="no blame"):
        blame.last_changed({1: "aa"}, {"aa": 1}, [(5, 9)])


def test_path_last_commit_is_the_committer_date_and_id_of_the_last_commit_touching_the_path(history):
    path, commits, stamps = history
    assert blame.path_last_commit(path, commits[2], "h.py") == (stamps[2], commits[2])
    assert blame.path_last_commit(path, commits[2], "e.py") == (stamps[0], commits[0])
    assert blame.path_last_commit(path, commits[1], "h.py") == (stamps[1], commits[1])      # up to the built commit only


# --- ports of helpers ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rel, role", [("src/app.py", "primary"), ("a/examples/x.py", "example"), ("a/samples/x.py", "example"),
                                       ("a/example_x.py", "example"), ("a/sample_x.py", "example"), ("a/x.example.py", "example"),
                                       ("a/x.sample.py", "example"), ("examples.py", "example"), ("exam.py", "primary"), ("a/Examples/x.py", "example")])
def test_file_role(rel, role):
    assert assemble_bundle.file_role(rel) == role


@pytest.mark.parametrize("rel, vendored", [("a/node_modules/x.js", True), ("vendor/x.py", True), ("a/_vendor/b/x.py", True),
                                           ("a/third_party/x.py", True), ("a/third-party/x.py", True), ("a/vendored/x.py", True),
                                           ("vendor.py", False), ("a/vendors/x.py", False), ("a/b/x.py", False)])
def test_is_vendored_looks_at_directories_only(rel, vendored):
    assert assemble_bundle.is_vendored(rel) is vendored


def test_callee_name_of_a_new_expression_is_the_last_segment_of_the_constructed_name():
    c = {"name": "<operator>.new", "code": "new  ns.Foo(1)"}
    assert assemble_bundle.callee_name(c) == "Foo"
    assert assemble_bundle.callee_name({"name": "<operator>.new", "code": "new (x)()"}) == "<dynamic>"
    assert assemble_bundle.callee_name({"name": "", "code": "x"}) == "<dynamic>"
    assert assemble_bundle.callee_name({"name": "f", "code": "f()"}) == "f"
    assert assemble_bundle.callee_name({"name": "<operator>.new", "code": "new Foo(1)"}) == "Foo"     # no dot: group 1, not the whole match


# --- symbol, param and file record fields ----------------------------------------------------------------------------------
def test_symbols_carry_their_parent_their_complexity_and_params_their_line(repo):
    records, _ = assemble(repo)
    q = qnames(records)
    syms = of(records, "symbol")
    assert {r["qname"]: (q.get(r.get("parentId")), r["complexity"]) for r in syms} == {
        "r.py::<module>": (None, 1), "r.py::B": ("r.py::<module>", 1), "r.py::B#2": ("r.py::<module>", 1),
        "r.py::B.x.<get>": ("r.py::B", 1), "r.py::B#2.x.<get>": ("r.py::B#2", 3),      # 1 + its two conditions
        "r.py::B#2.x.<set>": ("r.py::B#2", 1), "r.py::g": ("r.py::<module>", 1)}
    assert all("parentId" not in r for r in syms if r["kind"] == "module")
    assert sorted((q[r["symbolId"]], r["index"], r["line"]) for r in of(records, "param")) == [
        ("r.py::B#2.x.<get>", -1, 12), ("r.py::B#2.x.<set>", -1, 19), ("r.py::B#2.x.<set>", 0, 19),
        ("r.py::B.x.<get>", -1, 6), ("r.py::g", 0, 23)]


def test_a_param_reported_at_line_0_is_recorded_at_line_1(repo):
    facts = full_facts()
    g = next(m for m in facts["m"] if m["name"] == "g")
    for p in facts["p"]:
        if p["m"] == g["id"]:
            p["l"] = 0
    records, _ = assemble(repo, facts)
    q = qnames(records)
    assert [r["line"] for r in of(records, "param") if q[r["symbolId"]] == "r.py::g"] == [1]


def test_a_symbol_without_a_parent_scope_has_no_parent_id(tmp_path):
    facts = ts_facts()
    facts["m"] = []                                                         # no `:program`, so no module to fall back to
    path = tmp_path / "repo"
    commit = git_repo(path, {"a.ts": TS_SRC})
    records, _ = assemble_bundle.assemble(lang="javascript", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None,
                                          repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC,
                                          built_at=BUILT_AT)
    [i] = [r for r in of(records, "symbol") if r["qname"] == "a.ts::I"]
    assert "parentId" not in i


def test_line_count_is_the_number_of_lines_with_or_without_a_final_newline(tmp_path):
    from normalize_helpers import py_facts
    files = {"a.py": "x = 1\ny = 2", "b.py": "x = 1\ny = 2\n", "c.py": "x = 1\n\n", "d.py": "x = 1"}
    path = tmp_path / "repo"
    commit = git_repo(path, files)
    records, _ = assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=py_facts(files), stage=path,
                                          astgen_dir=None, repo=path, commit=commit, audit_ids=[UUID],
                                          extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)
    assert {r["path"]: r["lineCount"] for r in of(records, "file")} == {"a.py": 2, "b.py": 2, "c.py": 2, "d.py": 1}


# --- call-site kinds and resolutions, dropped synthetic calls, return and condition points ------------------------------
def assemble_py(tmp_path, src, edit):
    from normalize_helpers import py_facts
    path = tmp_path / "repo"
    commit = git_repo(path, {"r.py": src})
    facts = py_facts({"r.py": src})
    edit(facts, next(m for m in facts["m"] if m["name"] == "f"))
    return assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None, repo=path,
                                    commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)


def cs_fact(f, id_, name, line, mfn="", in_=(), out=(), code=None):
    return {"k": "cs", "m": f["id"], "id": str(id_), "name": name, "mfn": mfn, "l": line, "c": 1, "code": code or f"{name}()",
            "dt": "STATIC_DISPATCH", "in": [dict(i) for i in in_], "out": list(out), "args": []}


CALL_CASES = [   # name, mfn, in, out  ->  expected (callKind, resolution, externalTarget, status, reason)
    ("Foo", "r.Foo", [], [], ("new", "unresolved", None, "possible", "name-match")),
    ("super", "<unknownFullName>", [], [], ("super", "unresolved", None, "possible", "name-match")),
    ("getcwd", "os.getcwd", [], ["os.getcwd"], ("call", "external", "os.getcwd", "possible", "name-match")),
    ("pr", "__builtin.pr", [], ["__builtin.pr"], ("call", "external", "__builtin.pr", "possible", "name-match")),
    ("unk", "<unknownFullName>", [], ["os.x"], ("call", "unresolved", None, "possible", "name-match")),
    ("noext", "m.noext", [], [], ("call", "unresolved", None, "possible", "name-match")),
    ("nomfn", "", [], ["o.nomfn"], ("call", "unresolved", None, "possible", "name-match")),
    ("hasin", "m.hasin", [{"id": "zzz", "full": "q"}], ["o.hasin"], ("call", "unresolved", None, "possible", "name-match")),
    ("unkout", "m.unkout", [], ["<unknownFullName>.y"], ("call", "unresolved", None, "possible", "name-match")),
]


def test_each_call_gets_its_kind_and_a_resolution_from_what_the_cpg_knows_about_its_target(tmp_path):
    src = "def f():\n" + "".join(f"    v{i} = {name}()\n" for i, (name, *_) in enumerate(CALL_CASES))

    def edit(facts, f):
        for i, (name, mfn, in_, out, _) in enumerate(CALL_CASES):
            op = "<operator>.new" if name == "Foo" else name
            facts["cs"].append(cs_fact(f, 100 + i, op, i + 2, mfn, in_, out, code="new Foo()" if name == "Foo" else None))
    records, _ = assemble_py(tmp_path, src, edit)
    got = {r["calleeName"]: (r["callKind"], r["resolution"], r.get("externalTarget"), r["status"], r["reason"], r["line"], r["targets"])
           for r in of(records, "callsite")}
    assert got == {name: (*want, i + 2, []) for i, (name, _, _, _, want) in enumerate(CALL_CASES)}


def test_calls_to_one_callee_in_one_scope_are_numbered_by_position_whatever_the_fact_order(tmp_path):
    src = "def f(): g()\n\ndef g():\n    pass\n\ndef h():\n    g()\n    g()\n"
    for order in ((0, 1, 2), (2, 1, 0), (1, 2, 0)):
        def edit(facts, f, order=order):
            h = next(m for m in facts["m"] if m["name"] == "h")
            calls = [cs_fact(f, 400, "g", 1, code="g()"), cs_fact(h, 401, "g", 7, code="g()"), cs_fact(h, 402, "g", 8, code="g()")]
            facts["cs"] += [calls[i] for i in order]
        records, _ = assemble_py(tmp_path / str(order[0]) / str(order[1]), src, edit)
        q = qnames(records)
        assert sorted((q[r["symbolId"]], r["ordinal"], r["line"]) for r in of(records, "callsite")) == [
            ("r.py::f", 0, 1), ("r.py::h", 0, 7), ("r.py::h", 1, 8)]


def test_instantiating_a_class_targets_its_init_not_a_method_declared_before_it(tmp_path):
    src = "class K:\n    def m(self):\n        pass\n\n    def __init__(self):\n        pass\n\n\ndef f():\n    K()\n"

    def edit(facts, f):
        body = next(m for m in facts["m"] if m["name"] == "<body>")
        facts["cs"].append({**cs_fact(f, 500, "K", 10, "r.py:<module>.K", code="K()"), "in": [{"id": body["id"], "full": body["full"]}]})
    records, _ = assemble_py(tmp_path, src, edit)
    [call] = of(records, "callsite")
    assert [t["qname"] for t in call["targets"]] == ["r.py::K.__init__"]


def arg(i, e, an="", f=()):
    return {"i": i, "an": an, "f": list(f), "e": e}


LIT = lambda c: {"o": "lit", "c": c}


def test_call_arguments_become_value_sets_by_path_except_receivers_unknowns_and_over_deep_paths(tmp_path):
    src = "def g(*a, **k):\n    pass\n\n\ndef f():\n    g(1, 'x', k=2)\n"

    def edit(facts, f):
        call = cs_fact(f, 600, "g", 6, code="g(1, 'x', k=2)")
        call["args"] = [arg(0, {"o": "name", "n": "recv"}),                     # the receiver: no value set
                        arg(1, LIT("1")), arg(1, LIT("9")),                      # a path seen twice: the first wins
                        arg(2, LIT("'x'"), f=["a", "b"]),                        # a field path
                        arg(3, LIT("2"), an="k"),                                # a keyword argument
                        arg(4, {"o": "zzz"}),                                    # runtime-built: dropped
                        arg(5, {"o": "call", "n": "h"}),                         # point-derived: kept
                        arg(6, LIT("3"), f=["a", "b", "c"]),                     # at the depth cap: kept
                        arg(7, LIT("4"), f=["a", "b", "c", "d"])]                # past it: dropped
        facts["cs"].append(call)
    records, _ = assemble_py(tmp_path, src, edit)
    got = sorted((r["arg"], r["values"]) for r in of(records, "valueSet"))
    assert got == sorted([("a0", [{"const": 1}]), ("a1.a.b", [{"const": "x"}]), ("kw.k", [{"const": 2}]),
                          ("a4", [{"unknown": "point-derived"}]), ("a5.a.b.c", [{"const": 3}])])
    [call] = of(records, "callsite")
    assert {r["callSiteId"] for r in of(records, "valueSet")} == {call["id"]}


def test_a_value_set_path_given_twice_takes_the_first_argument_in_canonical_order_whatever_the_args_order(tmp_path):
    src = "def g(*a):\n    pass\n\n\ndef f():\n    g(1)\n"
    seen = set()
    for order in ((0, 1), (1, 0)):
        def edit(facts, f, order=order):
            call = cs_fact(f, 700, "g", 6, code="g(1)")
            args = [arg(1, LIT("9")), arg(1, LIT("1"))]
            call["args"] = [args[i] for i in order]
            facts["cs"].append(call)
        records, _ = assemble_py(tmp_path / str(order[0]), src, edit)
        seen.add(tuple((r["arg"], json.dumps(r["values"])) for r in of(records, "valueSet")))
    assert seen == {(("a0", '[{"const": 1}]'),)}


@pytest.mark.parametrize("a, b", [
    (arg(1, LIT("9")), arg(1, LIT("1"))),                                             # the canonical text decides, not the order given
    (arg(1, LIT("'\u00e9'")), arg(1, LIT("'z'"))),                                    # unescaped text: U+00E9 sorts after "z"
    ({"i": 1, "an": "", "f": [], "e": LIT("9")}, {"i": 1, "e": LIT("1"), "an": "", "f": []}),   # keys sorted: the key order given is no matter
])
def test_canonical_args_order_by_position_then_the_whole_json_with_sorted_keys_and_unescaped_text(a, b):
    for args in ([a, b], [b, a]):
        assert assemble_bundle.canonical_args(args)[0] is b
    assert [x["i"] for x in assemble_bundle.canonical_args([arg(2, LIT("1")), arg(1, LIT("2")), arg(10, LIT("1"))])] == [1, 2, 10]


def test_a_decorator_arg_path_given_twice_takes_the_first_argument_in_canonical_order_whatever_the_args_order(tmp_path):
    seen = set()
    for order in ((0, 1), (1, 0)):
        def edit(facts, f, order=order):
            [d0] = [d for d in facts["deco"] if d["m"] == f["id"]]
            args = [arg(1, LIT("9")), arg(1, LIT("1"))]
            d0["args"] = [args[i] for i in order]
        records, _ = assemble_py(tmp_path / str(order[0]), DECO_SRC, edit)
        [d] = [r for r in of(records, "decorator") if r["name"] == "deco"]
        seen.add(json.dumps(d["args"]))
    assert seen == {'[{"arg": "a0", "values": [{"const": 1}]}]'}


def test_the_targets_of_a_call_do_not_depend_on_the_order_of_its_in_list(tmp_path):
    src = "def g():\n    pass\n\n\ndef h():\n    pass\n\n\ndef f():\n    g()\n"
    seen = set()
    for rev in (False, True):
        def edit(facts, f, rev=rev):
            ins = [{"id": m["id"], "full": m["full"]} for m in facts["m"] if m["name"] in ("g", "h")]
            call = cs_fact(f, 710, "g", 10, code="g()", in_=ins[::-1] if rev else ins)
            facts["cs"].append(call)
        records, _ = assemble_py(tmp_path / str(rev), src, edit)
        [c] = of(records, "callsite")
        seen.add(tuple(t["qname"] for t in c["targets"]))
    assert len(seen) == 1


DECO_SRC = "def deco(*a, **k):\n    pass\n\n\n@deco(1, 'x', k=2)\ndef f():\n    pass\n"


def test_decorator_arguments_are_recorded_by_path_sorted_and_skip_unknown_and_over_deep_ones(tmp_path):
    def edit(facts, f):
        [d0] = [d for d in facts["deco"] if d["m"] == f["id"]]
        d0["args"] = [arg(1, LIT("1")), arg(2, LIT("'x'"), f=["a"]), arg(3, LIT("2"), an="k"), arg(4, {"o": "zzz"}),
                      arg(5, LIT("5"), f=["a", "b", "c", "d"]), arg(6, {"o": "call", "n": "h"})]
    records, _ = assemble_py(tmp_path, DECO_SRC, edit)
    [d] = [r for r in of(records, "decorator") if r["name"] == "deco"]
    assert d["args"] == [{"arg": "a0", "values": [{"const": 1}]}, {"arg": "a1.a", "values": [{"const": "x"}]},
                         {"arg": "a5", "values": [{"unknown": "point-derived"}]}, {"arg": "kw.k", "values": [{"const": 2}]}]
    assert (d["ordinal"], d["line"], d["form"]) == (0, 5, "decorator")


def test_a_decorator_on_the_first_line_is_recorded_at_line_1(tmp_path):
    records, _ = assemble_py(tmp_path, "@deco\ndef f():\n    pass\n", lambda facts, f: None)
    assert [(r["name"], r["line"]) for r in of(records, "decorator")] == [("deco", 1)]


def test_imports_are_ordered_by_position_not_by_walk_order_and_are_not_dynamic(tmp_path):
    src = "def f():\n    import json\n\n\nimport os\nfrom sys import path\n"
    records, _ = assemble_py(tmp_path, src, lambda facts, f: None)
    assert [(r["ordinal"], r["line"], r["module"], r["dynamic"]) for r in of(records, "import")] == [
        (0, 2, "json", False), (1, 5, "os", False), (2, 6, "sys", False)]


def test_a_call_whose_name_is_not_in_the_source_is_dropped_and_counted_by_name_for_the_twelve_most_common(tmp_path):
    names = ["syn0", "syn0"] + [f"syn{i}" for i in range(1, 13)]

    def edit(facts, f):
        facts["cs"] += [cs_fact(f, 200 + i, n, 2) for i, n in enumerate(names)]
        facts["cs"].append(cs_fact(f, 300, "", 2))                                # an unnamed call is no call site at all
        facts["cs"].append({**cs_fact(f, 301, "orphan", 2), "m": "no-such-method"})   # nor is one of an unmapped method
    records, report = assemble_py(tmp_path, "def f():\n    pass\n", edit)
    assert of(records, "callsite") == []
    assert report["syntheticCallsDropped"] == {"total": 14, "byName": {"syn0": 2, **{f"syn{i}": 1 for i in range(1, 12)}}}


def test_returns_and_conditions_are_recorded_at_their_lines_clamped_to_1_and_a_long_condition_is_cut_at_200(tmp_path):
    src = "g = lambda: 0\n\nf = lambda: 1\n\ndef f():\n    pass\n"
    long_code = "a " + "word " * 80

    def edit(facts, f):
        facts["r"].append({"k": "r", "m": f["id"], "id": "5001", "l": 0, "c": 0})
        facts["r"].append({"k": "r", "m": f["id"], "id": "5002", "l": 6, "c": 4})
        facts["c"].append({"k": "c", "m": f["id"], "id": "6001", "l": 0, "c": 0, "code": long_code, "lab": "CALL"})
        facts["c"].append({"k": "c", "m": f["id"], "id": "6002", "l": 6, "c": 0, "code": "ok", "lab": "CALL"})
    records, report = assemble_py(tmp_path, src, edit)
    q = qnames(records)
    assert [(r["ordinal"], r["line"]) for r in of(records, "ret") if q[r["symbolId"]] == "r.py::f"] == [(0, 1), (1, 6)]
    lams = [r for r in of(records, "ret") if "<anon>" in q[r["symbolId"]]]
    assert sorted((q[r["symbolId"]], r["ordinal"], r["line"], r["id"] == ids.point_id(r["symbolId"], "ret", 0)) for r in lams) == [
        ("r.py::<anon>#1", 0, 1, True), ("r.py::<anon>#2", 0, 3, True)]
    assert report["syntheticRetsForExpressionLambdas"] == 2
    assert [(r["ordinal"], r["line"], r["code"]) for r in of(records, "cond")] == [(0, 1, core.collapse(long_code, 200)), (1, 6, "ok")]
    assert len(of(records, "cond")[0]["code"]) == len(core.collapse(long_code, 200)) <= 200


# --- constants and imports ---------------------------------------------------------------------------------------------
CONST_SRC = """import os
from pkg import thing as t
from pkg import other


PREFIX = "/api"
DUP = 1
DUP = 2
ALIAS = t
UNK = zzz()


class K:
    ATTR = "x"
    ATTR2 = other


def f():
    import json
"""


def const_unit(tmp_path):
    from normalize_helpers import py_facts
    files = {"r.py": CONST_SRC, "pkg.py": "thing = 1\nother = 2\n"}
    path = tmp_path / "repo"
    commit = git_repo(path, files)
    facts = py_facts(files)
    mod = next(m for m in facts["m"] if m["name"] == "<module>" and m["full"] == "r.py:<module>")
    body = next(m for m in facts["m"] if m["name"] == "<body>")

    def const(m, name, line, e):
        facts["const"].append({"k": "const", "m": m["id"], "name": name, "l": line, "e": e})
    const(mod, "PREFIX", 6, {"o": "lit", "c": "'/api'"})
    const(mod, "DUP", 7, {"o": "lit", "c": "1"})
    const(mod, "DUP", 8, {"o": "lit", "c": "2"})                       # a name bound twice is no constant
    const(mod, "ALIAS", 9, {"o": "name", "n": "t"})
    const(mod, "UNK", 10, {"o": "call", "n": "zzz"})                   # an unknown value is no constant
    const(mod, "LOW", 0, {"o": "lit", "c": "3"})
    const(body, "ATTR", 14, {"o": "lit", "c": "'x'"})
    const(body, "ATTR2", 15, {"o": "name", "n": "other"})
    f = next(m for m in facts["m"] if m["name"] == "f")
    const(f, "LOCAL", 19, {"o": "lit", "c": "1"})                      # only module- and type-level names are constants
    return assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None, repo=path,
                                    commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)


def test_module_and_type_level_names_bound_once_to_a_known_value_are_constants(tmp_path):
    records, _ = const_unit(tmp_path)
    got = {r["qname"]: (r["line"], r["values"]) for r in of(records, "constant") if r["qname"].startswith("r.py::")}
    assert got == {"r.py::PREFIX": (6, [{"const": "/api"}]), "r.py::LOW": (1, [{"const": 3}]),
                   "r.py::ALIAS": (9, [{"ref": "pkg.py::thing"}]),
                   "r.py::K.ATTR": (14, [{"const": "x"}]),
                   "r.py::K.ATTR2": (15, [{"ref": "pkg.py::other"}])}


def test_a_constant_names_a_sibling_constant_an_imported_name_or_a_submodule(tmp_path):
    from normalize_helpers import py_facts
    src = "from pkg import sub\nfrom pkg import thing\nSUB = sub\nTH = thing\nPFX = 'a'\nCP = PFX\nATT = sub.v\n"
    files = {"r.py": src, "pkg/__init__.py": "thing = 1\n", "pkg/sub.py": "v = 1\n"}
    path = tmp_path / "repo"
    commit = git_repo(path, files)
    facts = py_facts(files)
    mod = next(m for m in facts["m"] if m["full"] == "r.py:<module>")
    for name, line, e in (("SUB", 3, {"o": "name", "n": "sub"}), ("TH", 4, {"o": "name", "n": "thing"}),
                          ("PFX", 5, {"o": "lit", "c": "'a'"}), ("CP", 6, {"o": "name", "n": "PFX"}),
                          ("ATT", 7, {"o": "attr", "b": {"o": "name", "n": "sub"}, "f": "v"})):
        facts["const"].append({"k": "const", "m": mod["id"], "name": name, "l": line, "e": e})
    records, _ = assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None,
                                          repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC,
                                          built_at=BUILT_AT)
    assert {r["qname"]: r["values"] for r in of(records, "constant")} == {
        "r.py::SUB": [{"importRef": {"module": "pkg", "name": "sub"}}],          # a submodule, not a name inside pkg
        "r.py::TH": [{"ref": "pkg/__init__.py::thing"}], "r.py::PFX": [{"const": "a"}], "r.py::CP": [{"ref": "r.py::PFX"}],
        "r.py::ATT": [{"ref": "pkg/sub.py::v"}]}


def test_each_import_is_a_record_with_its_ordinal_its_resolved_path_and_the_symbol_it_sits_in(tmp_path):
    records, _ = const_unit(tmp_path)
    q = qnames(records)
    [rfile] = [f for f in of(records, "file") if f["path"] == "r.py"]
    got = [(r["ordinal"], r["line"], r["module"], r["names"], r.get("resolvedPath"), q.get(r.get("symbolId")))
           for r in of(records, "import") if r["fileId"] == rfile["id"]]
    assert got == [(0, 1, "os", [], None, None), (1, 2, "pkg", [{"name": "thing", "alias": "t"}], "pkg.py", None),
                   (2, 3, "pkg", [{"name": "other"}], "pkg.py", None), (3, 19, "json", [], None, "r.py::f")]


# --- data and controlledBy edges the assembler must drop or reshape -----------------------------------------------------
def test_data_edges_without_a_symbol_a_use_point_or_a_def_point_are_dropped_and_counted_and_paths_are_reshaped(repo):
    facts = full_facts()
    getter2 = next(m for m in facts["m"] if m["name"] == "x" and m["l"] == 12)
    d = lambda **kw: {"k": "d", "m": getter2["id"], "tf": [], "ff": [], "uk": "cs", "uid": "7001", "th": "a0", "h": "p0",
                      "dk": "p", "did": getter2["id"], "di": 0, **kw}
    facts["d"] += [d(m="no-such-method"), d(uid="9999"), d(dk="cs", did="9999", h="r"),
                   d(uk="r", uid="5003", dk="cs", did="7001", th="", h="r"),
                   d(uk="r", uid="5001", dk="cs", did="7001", th="kw.k", h="r")]
    records, report = assemble(repo, facts)
    assert report["dataDropped"] == {"duplicate": 1, "no-symbol": 1, "no-use-point": 1, "no-def-point": 1}
    got = {(r["toPath"], r["fromPath"], r["status"], r["reason"]) for r in of(records, "data")}
    assert ("", "r", "confirmed", "intra-flow") in got and ("kw.k", "r", "confirmed", "intra-flow") in got


def test_a_controlled_by_edge_with_an_unknown_point_or_condition_is_counted_unmapped_and_not_written(repo):
    facts = full_facts()
    getter2 = next(m for m in facts["m"] if m["name"] == "x" and m["l"] == 12)
    cb = lambda pk, pid, cond, m=getter2["id"]: {"k": "cb", "m": m, "pk": pk, "pid": pid, "cond": cond}
    facts["cb"] += [cb("cs", "7001", "nope"), cb("cs", "nope", "6001"), cb("cs", "7001", "6001", m="no-such-method")]
    records, report = assemble(repo, facts)
    assert report["controlledBy"] == {"rawEdges": 8, "selfEdgesDropped": 1, "duplicatePairsDropped": 2, "unmapped": 2,
                                      "written": 3, "condToCond": 1}
    assert len(of(records, "controlledBy")) == 3


# --- assemble(): the file filter, the frontend switch, the collision guard ----------------------------------------------
TS_SRC = "interface I {\n  a: 1;\n}\n"


def ts_facts():
    meta = dict(file="a.ts", pfull="")
    return {**{k: [] for k in KINDS}, "f": [{"k": "f", "name": "a.ts"}],
            "m": [{"k": "m", "id": "1", "name": ":program", "full": "a.ts::program", "l": 1, "le": 3, "c": 0, "pk": "TYPE_DECL",
                   "pid": "t0", "code": "", **meta}],
            "t": [{"k": "t", "id": "t0", "name": ":program", "full": "a.ts::program", "l": 1, "c": 0, "pk": "NAMESPACE_BLOCK",
                   "pid": "x", **meta},
                  {"k": "t", "id": "t1", "name": "I", "full": "a.ts::program:I", "l": 1, "c": 0, "pk": "BLOCK", "pid": "x", **meta}]}


def assemble_ts(tmp_path, astgen_dir):
    path = tmp_path / "repo"
    commit = git_repo(path, {"a.ts": TS_SRC})
    return assemble_bundle.assemble(lang="javascript", unit_root=".", repo_id="r", facts=ts_facts(), stage=path, astgen_dir=astgen_dir,
                                    repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)


def type_span(records):
    [i] = [r for r in of(records, "symbol") if r["qname"] == "a.ts::I"]
    return i["lineStart"], i["lineEnd"]


def test_a_javascript_types_span_comes_from_the_babel_ast_when_astgen_ran(tmp_path):
    astgen = tmp_path / "astgen"
    astgen.mkdir()
    ast_ = {"body": [{"type": "TSInterfaceDeclaration", "id": {"name": "I"}, "loc": {"start": {"line": 1, "column": 0}},
                      "end": len("interface I {\n  a: 1;\n}")}]}
    (astgen / "a.ts.json").write_text(json.dumps({"ast": ast_}), encoding="utf-8")
    records, _ = assemble_ts(tmp_path, astgen)
    assert type_span(records) == (1, 3)


def test_a_javascript_unit_without_astgen_output_keeps_the_cpgs_one_line_type_span(tmp_path):
    records, _ = assemble_ts(tmp_path, None)
    assert type_span(records) == (1, 1)


def test_a_python_unit_never_reads_babel_output_even_when_a_directory_is_given(tmp_path):
    astgen = tmp_path / "astgen"
    astgen.mkdir()
    (astgen / "r.py.json").write_text("not json", encoding="utf-8")
    path = tmp_path / "repo"
    commit = git_repo(path, {"r.py": SRC})
    assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=full_facts(), stage=path, astgen_dir=astgen, repo=path,
                             commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)


def test_vendored_and_missing_files_make_no_symbols(tmp_path):
    from normalize_helpers import py_facts
    src = "def f():\n    pass\n"
    path = tmp_path / "repo"
    commit = git_repo(path, {"r.py": src, "vendor/v.py": src})
    facts = py_facts({"r.py": src, "vendor/v.py": src})
    facts["f"].append({"k": "f", "name": "gone.py"})                       # listed by the CPG, absent from the staged tree
    records, _ = assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None,
                                          repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC,
                                          built_at=BUILT_AT)
    assert sorted(qnames(records).values()) == ["r.py::<module>", "r.py::f"]


def test_two_scopes_with_one_qname_are_a_build_error(tmp_path, monkeypatch):
    import copy
    real = python.build_scopes

    def doubled(B):
        scopes, m2s, d = real(B)
        f = next(s for s in scopes.values() if s.base == "f")
        twin = copy.copy(f)
        twin.key = ("twin",) + tuple(f.key)
        scopes[twin.key] = twin
        return scopes, m2s, d
    monkeypatch.setattr(python, "build_scopes", doubled)
    from normalize_helpers import py_facts
    src = "def f():\n    pass\n"
    path = tmp_path / "repo"
    commit = git_repo(path, {"r.py": src})
    with pytest.raises(assemble_bundle.BuildError, match=r"qname collision on 1 symbols, e\.g\. r\.py::f"):
        assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=py_facts({"r.py": src}), stage=path, astgen_dir=None,
                                 repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC, built_at=BUILT_AT)


# --- locate_callees: where each call's callee identifier is, from the source -------------------------------------------
def callee_builder(tmp_path, text, lang="javascript"):
    (tmp_path / "a.src").write_text(text, encoding="utf-8")
    return core.Builder(lang, ".", "r", {}, core.Source(tmp_path))


def call(id_, name, line, col, code=None):
    return {"id": id_, "name": name, "l": line, "c": col, "code": code or f"{name}()"}


def where(tmp_path, text, calls, lang="javascript"):
    return assemble_bundle.locate_callees(callee_builder(tmp_path, text, lang), "a.src", calls)


def test_a_callee_is_found_on_its_calls_line_at_the_column_of_its_name(tmp_path):
    assert where(tmp_path, "x = foo(1)\n", [call("1", "foo", 1, 4)]) == {"1": (1, 4)}
    assert where(tmp_path, "x = foo(1)\n", [call("1", "foo", 1, 5)], "python") == {"1": (1, 4)}   # Python columns are 1-based


def test_a_callee_on_a_later_line_is_placed_by_its_line_and_its_column_within_that_line(tmp_path):
    assert where(tmp_path, "xx a.\n  b(1)\n", [call("1", "b", 1, 3)]) == {"1": (2, 2)}


def test_a_call_whose_name_never_occurs_after_its_start_is_absent(tmp_path):
    assert where(tmp_path, "bar(1)\n", [call("1", "foo", 1, 0)]) == {}
    assert where(tmp_path, "foo(1)\nbar(2)\n", [call("1", "foo", 2, 0)]) == {}


def test_a_name_inside_a_longer_identifier_or_without_a_call_paren_is_not_the_callee(tmp_path):
    assert where(tmp_path, "foobar(1); xfoo(2); foo + foo(3)", [call("1", "foo", 1, 0)]) == {"1": (1, 26)}
    assert where(tmp_path, "foo ?. (1); foo<T>(2)", [call("1", "foo", 1, 0)]) == {"1": (1, 0)}


def test_a_new_expression_needs_no_paren(tmp_path):
    c = {"id": "1", "name": "<operator>.new", "l": 1, "c": 4, "code": "new Foo"}
    assert where(tmp_path, "x = new Foo;", [c]) == {"1": (1, 8)}


def test_calls_sharing_a_start_and_name_take_successive_occurrences_shortest_code_then_lowest_id_first(tmp_path):
    text = "foo(1) + foo(2)\n"
    long_, short = call("1", "foo", 1, 0, "foo(1) + foo(2)"), call("2", "foo", 1, 0, "foo(1)")
    for calls in ([long_, short], [short, long_]):
        assert where(tmp_path, text, calls) == {"2": (1, 0), "1": (1, 9)}
    a, b = call("9", "foo", 1, 0, "foo(1)"), call("8", "foo", 1, 0, "foo(2)")
    for calls in ([a, b], [b, a]):
        assert where(tmp_path, text, calls) == {"8": (1, 0), "9": (1, 9)}
    assert where(tmp_path, "foo(1)\n", [a, b]) == {"8": (1, 0)}          # more calls than occurrences: the rest are absent


def test_a_column_outside_the_line_is_treated_as_the_start_of_the_line(tmp_path):
    assert where(tmp_path, "foo(1)\n", [call("1", "foo", 1, 50)]) == {"1": (1, 0)}
    assert where(tmp_path, "foo(1)\n", [call("1", "foo", 1, 0)], "python") == {"1": (1, 0)}      # python col 0 is below the 1-based base
    assert where(tmp_path, "foo(1)\n", [call("1", "foo", 1, 6)]) == {}         # exactly at the line's end: inside, so nothing is left to find


def test_the_search_window_reaches_400_lines_past_the_calls_line(tmp_path):
    c = [call("1", "foo", 1, 0)]
    assert where(tmp_path, "\n" * 401 + "foo(1)", c) == {}                        # foo on line 402: outside the window
    assert where(tmp_path, "\n" * 400 + "foo(1)", c) == {"1": (401, 0)}          # on line 401: the window's last line


def test_canonical_order_is_the_whole_facts_json_with_sorted_keys_and_unescaped_text():
    x, y = {"z": 0, "a": 1}, {"m": 1}
    assert assemble_bundle.canonical_order([y, x]) == [x, y]               # sorted keys: x's text starts at "a", not "z"
    e, z = {"n": "\u00e9"}, {"n": "z"}
    assert assemble_bundle.canonical_order([e, z]) == [z, e]               # unescaped, so U+00E9 sorts after "z"


# --- a class instantiation resolves to the class's constructor ----------------------------------------------------------
INIT_SRC = "class K:\n    def __init__(self):\n        pass\n\n    def __init__(self, a):\n        pass\n\n\ndef mk():\n    return K()\n"


@pytest.mark.parametrize("seed", [None, 0, 1, 2, 3, 4, 5, 6, 7])
def test_instantiating_a_class_with_a_redefined_init_targets_the_last_one_for_every_fact_order(tmp_path, seed):
    from normalize_helpers import py_facts
    path = tmp_path / "repo"
    commit = git_repo(path, {"k.py": INIT_SRC})
    facts = py_facts({"k.py": INIT_SRC}, shuffle=seed)
    body = next(m for m in facts["m"] if m["name"] == "<body>")
    mk = next(m for m in facts["m"] if m["name"] == "mk")
    facts["cs"].append({"k": "cs", "m": mk["id"], "id": "9001", "name": "K", "mfn": "k.py:<module>.K", "l": 10, "c": 12,
                        "code": "K()", "dt": "STATIC_DISPATCH", "in": [{"id": body["id"], "full": body["full"]}], "out": [], "args": []})
    records, _ = assemble_bundle.assemble(lang="python", unit_root=".", repo_id="r", facts=facts, stage=path, astgen_dir=None,
                                          repo=path, commit=commit, audit_ids=[UUID], extract_script=joern_run.EXTRACT_SC)
    [call] = of(records, "callsite")
    assert [t["qname"] for t in call["targets"]] == ["k.py::K.__init__#2"]


def test_a_git_failure_is_an_error_not_an_empty_history(history, tmp_path):
    path, commits, _ = history
    with pytest.raises(subprocess.CalledProcessError):
        blame.blame(path, commits[2], "no/such/file.py")
    with pytest.raises(subprocess.CalledProcessError):
        blame.require_full_history(tmp_path)                     # not a repository
    with pytest.raises(subprocess.CalledProcessError):
        blame.path_last_commit(path, "0" * 40, "h.py")
