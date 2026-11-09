"""contact.json (development-contact list for the two proving-repo corpora; ML note v1.7 s3.1, R6) is re-derived
from HEAD by make_contact.py and must not drift; the derivation's rules are pinned on synthetic trees."""
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

FF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spikes", "frontend-fitness")
MC_PATH = os.environ.get("MAKE_CONTACT_PATH", os.path.join(FF, "make_contact.py"))
CONTACT_JSON = os.path.join(FF, "contact.json")


def _load():
    spec = importlib.util.spec_from_file_location("make_contact_under_test", MC_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mc = _load()


def _tree(tmp_path, files):
    for p, t in files.items():
        f = tmp_path / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(t)
    return str(tmp_path)


BASE = {
    "falkor-chat/server/falkorchat/__init__.py": "",
    "falkor-chat/server/falkorchat/alpha.py": "x = 1\n",
    "falkor-chat/server/falkorchat/beta.py": "x = 1\n",
    "falkor-chat/server/falkorchat/sub/gamma.py": "x = 1\n",
    "falkor-chat/server/falkorchat/sub/__init__.py": "",
    "falkor-chat/server/tests/test_delta.py": "x = 1\n",
    "falkor-chat/server/tests/conftest.py": "x = 1\n",
    "salesperson/src/api/client.ts": "x\n",
    "salesperson/src/api/client.test.ts": "x\n",
    "salesperson/src/ui/Panel.tsx": "x\n",
    "salesperson/vite.config.ts": "x\n",
}


def _derive(tmp_path, extra):
    d = mc.derive(_tree(tmp_path, {**BASE, **extra}))
    return {k: set(v["files"]) for k, v in d["corpora"].items()}, d


# ---------------------------------------------------------------- the committed file
def test_contact_json_matches_rederivation_at_its_derivedfrom_tree():
    ok, msg = mc.check(CONTACT_JSON)
    assert ok, msg


def test_contact_json_shape_and_flow_anchors_kept_apart():
    d = json.load(open(CONTACT_JSON))
    assert re.fullmatch(r"[0-9a-f]{40}", d["derivedFromTree"]) and "derivedFrom" not in d
    assert set(d["flowAnchors"]) == {"python", "javascript"}
    for lang, paths in d["flowAnchors"].items():
        assert paths == sorted(set(paths)) and paths
        assert not any(p.startswith(("./", "falkor-chat/", "salesperson/")) for p in paths)
    assert "flowAnchors" not in d["corpora"]["python"] and "flowAnchors" not in d["corpora"]["javascript"]


def test_contact_json_evidence_is_rederivable_from_head():
    d = json.load(open(CONTACT_JSON))
    for corpus, c in d["corpora"].items():
        for f, x in c["files"].items():
            assert not re.search(r"(-ml|-coordination)\.md$|docs/reviews/|(HISTORY|BACKLOG)\.md$|test_frontend_fitness", x["evidence"][0]["artifact"])
            assert 1 <= len(x["evidence"]) <= 3
            assert len({e["artifact"] for e in x["evidence"]}) == len(x["evidence"])
            for e in x["evidence"]:
                assert not e["artifact"].startswith("cpg/spikes/frontend-fitness/")
                body = subprocess.run(["git", "show", d["derivedFromTree"] + ":" + e["artifact"]], capture_output=True, text=True, cwd=FF).stdout
                assert e["match"] in body, (f, e)


def test_contact_json_known_anchors_and_counts():
    d = json.load(open(CONTACT_JSON))["corpora"]
    js, py = d["javascript"]["files"], d["python"]["files"]
    assert {"src/api/client.ts", "src/api/endpoints.ts"} <= set(js)          # S14 anchors (SL-1/SL-4)
    assert {"falkorchat/storefront_api.py", "falkorchat/app.py", "falkorchat/api.py"} <= set(py)
    assert d["python"]["corpusFiles"] >= d["python"]["contactFiles"] > 0
    assert d["javascript"]["corpusFiles"] >= d["javascript"]["contactFiles"] > 0


# ---------------------------------------------------------------- derivation rules on synthetic trees
def test_repo_rooted_path_and_absolute_prefix(tmp_path):
    got, _ = _derive(tmp_path, {
        "cpg/tests/test_a.py": "x = 'falkor-chat/server/falkorchat/alpha.py'\n",
        "cpg/tests/test_b.py": "p = '/home/u/repo/salesperson/src/api/client.ts'\n"})
    assert got["python"] == {"falkorchat/alpha.py"} and got["javascript"] == {"src/api/client.ts"}


def test_path_normalisation_dot_slash_and_trailing_punctuation(tmp_path):
    got, _ = _derive(tmp_path, {"docs/plans/connected-code-graph-rule-pack-spec.md":
                                "see ./falkor-chat/server/falkorchat/beta.py. And (falkor-chat/server/falkorchat/sub/gamma.py), done\n"})
    assert got["python"] == {"falkorchat/beta.py", "falkorchat/sub/gamma.py"}


def test_path_marker_needs_a_slash_boundary(tmp_path):
    got, _ = _derive(tmp_path, {"cpg/tests/test_a.py": "x = 'xfalkor-chat/server/falkorchat/alpha.py'\n"})
    assert got["python"] == set()


def test_dot_slash_prefix_on_relative_form(tmp_path):
    got, _ = _derive(tmp_path, {"cpg/tests/t.py": "open('./falkorchat/alpha.py'); './src/api/client.ts'\n"})
    assert got["python"] == {"falkorchat/alpha.py"} and got["javascript"] == {"src/api/client.ts"}


def test_marker_found_after_a_non_boundary_first_occurrence(tmp_path):
    got, _ = _derive(tmp_path, {"cpg/tests/t.py": "/w/xsalesperson/salesperson/src/api/client.ts\n"})
    assert got["javascript"] == {"src/api/client.ts"}


def test_corpus_relative_path_and_qname_suffix(tmp_path):
    got, d = _derive(tmp_path, {"docs/plans/connected-code-graph-rule-pack-spec.md": "`falkorchat/alpha.py:12`, `src/api/client.ts::apiFetch`\n"})
    assert got["python"] == {"falkorchat/alpha.py"} and got["javascript"] == {"src/api/client.ts"}
    assert d["corpora"]["python"]["files"]["falkorchat/alpha.py"]["forms"] == ["rel"]


def test_relative_form_ignored_in_fixture_sources(tmp_path):
    got, _ = _derive(tmp_path, {"cpg/fixtures/s/expected/a.yaml": "path: src/api/client.ts\npath2: tests/conftest.py\n",
                                "docs/test-plans/connected-code-graph-fixtures.md": "fx-svc:tests/conftest.py\n"})
    assert got["python"] == set() and got["javascript"] == set()
    got, _ = _derive(tmp_path, {"cpg/fixtures/s/expected/a.yaml": "path: salesperson/src/api/client.ts\n"})
    assert got["javascript"] == {"src/api/client.ts"}              # the repo-rooted form still counts everywhere


def test_root_level_file_needs_repo_rooted_form(tmp_path):
    got, _ = _derive(tmp_path, {"cpg/tests/t.py": "vite.config.ts\n"})
    assert got["javascript"] == set()
    got, _ = _derive(tmp_path, {"cpg/tests/t.py": "salesperson/vite.config.ts\n"})
    assert got["javascript"] == {"vite.config.ts"}


def test_corpus_unique_basename_with_symbol_or_line(tmp_path):
    got, _ = _derive(tmp_path, {"docs/plans/connected-code-graph-rule-pack-spec.md": "`alpha.py::f` and `Panel.tsx:40`, but beta.py alone\n"})
    assert got["python"] == {"falkorchat/alpha.py"} and got["javascript"] == {"src/ui/Panel.tsx"}


def test_ambiguous_basename_is_not_matched(tmp_path):
    tree = {"falkor-chat/server/other/alpha.py": "x\n", "cpg/tests/t.py": "alpha.py::f\n"}
    got, _ = _derive(tmp_path, tree)
    assert got["python"] == set()


def test_dotted_qname_resolves_to_longest_module_prefix(tmp_path):
    got, _ = _derive(tmp_path, {"cpg/tests/t.py": "a = 'falkorchat.sub.gamma.Cls.meth'\nb = 'falkorchat.alpha:app'\nc = 'falkorchat.beta.fn'\n"})
    assert got["python"] == {"falkorchat/sub/gamma.py", "falkorchat/alpha.py", "falkorchat/beta.py"}


def test_dotted_qname_package_and_nonmodule(tmp_path):
    got, _ = _derive(tmp_path, {"cpg/tests/t.py": "falkorchat.sub.nope falkorchat.zzz falkorchat.egg-info falkorchat\n"})
    assert got["python"] == {"falkorchat/sub/__init__.py"}          # `falkorchat.sub.nope` -> package `sub`; the rest name no module


def test_dotted_qname_needs_a_boundary(tmp_path):
    got, _ = _derive(tmp_path, {"cpg/tests/t.py": "myfalkorchat.alpha other.falkorchat.beta\n"})
    assert got["python"] == set()


EXPECTED_SOURCE = {
    "cpg/fixtures/s/f.yaml": "fixtures", "cpg/tests/golden/g.json": "golden", "cpg/spikes/vertical-slice/probe.py": "slice",
    "cpg/spikes/vertical-slice/tests/test_v.py": "tests", "cpg/tests/test_z.py": "tests", "cpg/spikes/store/tests/test_s.py": "tests",
    "cpg/packs/p/pack.yaml": "packs", "docs/plans/connected-code-graph-rule-pack-spec.md": "docs", "docs/test-reports/connected-code-graph-p0-slice-report.md": "slice", "cpg/ccg/mod.py": "cpg-other"}


def test_slice_report_read_only_in_section_4(tmp_path):
    rep = ("# r\n\n## 3. x\nfalkor-chat/server/falkorchat/alpha.py\n\n## 4. Proving-repo anchors the slice touched (eval-protocol 8.2)\n"
           "falkor-chat/server/falkorchat/beta.py\nsalesperson/src/api/client.ts\n\n## 5. Bundle spec items\nfalkor-chat/server/falkorchat/sub/gamma.py\n")
    got, d = _derive(tmp_path, {"docs/test-reports/connected-code-graph-p0-slice-report.md": rep})
    assert got["python"] == {"falkorchat/beta.py"} and got["javascript"] == {"src/api/client.ts"}
    assert d["corpora"]["python"]["files"]["falkorchat/beta.py"]["sources"] == ["slice"]
    early = "## 5. earlier\nx\n## 4. Proving-repo anchors the slice touched\nfalkor-chat/server/falkorchat/beta.py\n## 5. next\nfalkor-chat/server/falkorchat/alpha.py\n"
    got, _ = _derive(tmp_path / "e", {"docs/test-reports/connected-code-graph-p0-slice-report.md": early})      # the end marker is searched after the heading
    assert got["python"] == {"falkorchat/beta.py"}
    got, _ = _derive(tmp_path / "b", {"docs/test-reports/connected-code-graph-p0-slice-report.md": "## 3. x\nfalkor-chat/server/falkorchat/alpha.py\n"})           # no section 4: nothing
    assert got["python"] == set()
    got, _ = _derive(tmp_path / "c", {"docs/test-reports/connected-code-graph-p0-slice-report.md": "## 4. Proving-repo anchors the slice touched\nfalkor-chat/server/falkorchat/alpha.py\n"})  # to EOF
    assert got["python"] == {"falkorchat/alpha.py"}


def test_only_the_rule_pack_spec_among_plans_requirements_and_test_plans(tmp_path):
    text = "falkor-chat/server/falkorchat/alpha.py\n"
    got, _ = _derive(tmp_path, {"docs/plans/connected-code-graph-bundle-spec3.md": text, "docs/plans/connected-code-graph.md": text,
                                "docs/plans/connected-code-graph-graph.md": text, "docs/requirements/connected-code-graph.md": text,
                                "docs/test-plans/connected-code-graph-fixtures.md": text, "docs/plans/connected-code-graph-answer-spec.md": text})
    assert got["python"] == set()
    got, _ = _derive(tmp_path / "rp", {"docs/plans/connected-code-graph-rule-pack-spec.md": text})
    assert got["python"] == {"falkorchat/alpha.py"}


def test_each_search_source_is_searched(tmp_path):
    for art in EXPECTED_SOURCE:
        got, d = _derive(tmp_path / art.replace("/", "_"), {art: ("## 4. Proving-repo anchors the slice touched\n" if "slice-report" in art else "")
                                                         + "falkor-chat/server/falkorchat/alpha.py\n"})
        assert got["python"] == {"falkorchat/alpha.py"}, art
        assert mc.source_of(art) == EXPECTED_SOURCE[art]
        assert d["searchSources"][EXPECTED_SOURCE[art]] >= 1


def test_out_of_scope_artifacts_are_not_searched(tmp_path):
    text = "falkor-chat/server/falkorchat/alpha.py\n"
    got, _ = _derive(tmp_path, {"cpg/spikes/frontend-fitness/labels/x.jsonl": text, "cpg/spikes/frontend-fitness/README.md": text,
                                "cpg/tests/test_contact.py": text, "docs/plans/salesperson-ui.md": text, "README.md": text,
                                "docs/reviews/connected-code-graph-r.md": text, "docs/test-reports/connected-code-graph-rep.md": text,
                                "docs/plans/connected-code-graph-p0-ml.md": text, "docs/reviews/connected-code-graph-x-ml.md": text,
                                "docs/plans/connected-code-graph-coordination.md": text, "docs/HISTORY.md": text, "docs/BACKLOG.md": text,
                                "cpg/HISTORY.md": text, "cpg/notes-coordination.md": text, "cpg/notes-ml.md": text, "cpg/BACKLOG.md": text, "cpg/tests/test_frontend_fitness.py": text,
                                "falkor-chat/server/falkorchat/beta.py": text})
    assert got["python"] == set()


def test_evidence_capped_at_three_and_prefers_path_form(tmp_path):
    extra = {f"cpg/tests/test_{i}.py": "falkorchat/alpha.py\n" for i in range(5)}
    extra["docs/plans/connected-code-graph-rule-pack-spec.md"] = "falkor-chat/server/falkorchat/alpha.py\n"
    _, d = _derive(tmp_path, extra)
    x = d["corpora"]["python"]["files"]["falkorchat/alpha.py"]
    assert len(x["evidence"]) == 3 and x["artifacts"] == 6 and x["evidence"][0]["form"] == "path"


def test_test_files_outside_the_js_prod_pattern_are_not_corpus(tmp_path):
    got, d = _derive(tmp_path, {"cpg/tests/t.py": "salesperson/src/api/client.test.ts salesperson/e2e/flow.ts salesperson/src/__tests__/z.ts\n",
                                "salesperson/e2e/flow.ts": "x\n", "salesperson/src/__tests__/z.ts": "x\n", "salesperson/src/a.spec.tsx": "x\n"})
    assert got["javascript"] == set() and d["corpora"]["javascript"]["corpusFiles"] == 3


def test_evidence_lists_distinct_artifacts(tmp_path):
    both = "falkor-chat/server/falkorchat/alpha.py and falkorchat/alpha.py and alpha.py::f and falkorchat.alpha\n"
    _, d = _derive(tmp_path, {"cpg/tests/t1.py": both, "cpg/tests/t2.py": both})
    ev = d["corpora"]["python"]["files"]["falkorchat/alpha.py"]["evidence"]
    assert [e["artifact"] for e in ev] == ["cpg/tests/t1.py", "cpg/tests/t2.py"]


def _flow(rows):
    return "".join(json.dumps({"path": r, "ordinal": i}) + "\n" for i, r in enumerate(rows))


def test_flow_anchors_separate_normalised_distinct_and_per_corpus(tmp_path):
    sd = "cpg/spikes/frontend-fitness/sample/"
    got, d = _derive(tmp_path, {
        sd + "python-flow.jsonl": _flow(["./falkorchat/beta.py", "falkor-chat/server/falkorchat/beta.py", "falkorchat\\sub\\gamma.py",
                                         "tests/test_delta.py", "falkorchat/alpha.py"]),
        sd + "javascript-flow.jsonl": _flow(["src/ui/Panel.tsx", "./src/ui/Panel.tsx", "salesperson/src/api/client.ts"]),
        sd + "python-flow.brief.md": "falkorchat/zzz.py", sd + "javascript-flow-tier2.jsonl": _flow(["src/other.ts"])})
    assert d["flowAnchors"]["python"] == ["falkorchat/alpha.py", "falkorchat/beta.py", "falkorchat/sub/gamma.py", "tests/test_delta.py"]
    assert d["flowAnchors"]["javascript"] == ["src/api/client.ts", "src/ui/Panel.tsx"]
    assert got["python"] == set() and got["javascript"] == set()         # flow anchors are not contact `files`


def test_flow_sample_not_a_contact_source(tmp_path):
    got, _ = _derive(tmp_path, {"cpg/spikes/frontend-fitness/sample/python-flow.jsonl": _flow(["falkorchat/alpha.py"])})
    assert got["python"] == set()


def _git(repo, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q")
    for p, t in {**BASE, "cpg/tests/t.py": "falkor-chat/server/falkorchat/alpha.py\n"}.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True); (r / p).write_text(t)
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "c1")
    return r


def test_derived_from_tree_pins_the_tree_read(tmp_path):
    r = _repo(tmp_path)
    tree1 = _git(r, "rev-parse", "HEAD^{tree}")
    out = tmp_path / "contact.json"
    out.write_text(mc.render(mc.derive(repo=str(r))))
    assert json.loads(out.read_text())["derivedFromTree"] == tree1
    assert mc.check(str(out), str(r)) == (True, "ok")
    # a later commit that adds a source naming another file, and edits an excluded document, cannot drift the check
    (r / "cpg/tests/t2.py").write_text("falkor-chat/server/falkorchat/beta.py\n")
    (r / "docs/reviews").mkdir(parents=True); (r / "docs/reviews/connected-code-graph-x.md").write_text("falkor-chat/server/falkorchat/beta.py\n")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "c2")
    assert mc.check(str(out), str(r)) == (True, "ok")
    assert set(mc.derive(repo=str(r))["corpora"]["python"]["files"]) == {"falkorchat/alpha.py", "falkorchat/beta.py"}   # HEAD differs, pinned tree does not


def test_cli_from_and_check(tmp_path):
    r = _repo(tmp_path)
    sha1 = _git(r, "rev-parse", "HEAD")
    tree1 = _git(r, "rev-parse", "HEAD^{tree}")
    (r / "cpg/tests/t2.py").write_text("falkor-chat/server/falkorchat/beta.py\n"); _git(r, "add", "-A"); _git(r, "commit", "-qm", "c2")
    env = dict(os.environ, CONTACT_REPO=str(r))
    out = tmp_path / "c.json"
    run = lambda *a: subprocess.run([sys.executable, MC_PATH, *a], env=env, capture_output=True, text=True)
    assert run("--out", str(out), "--from", sha1).returncode == 0
    d = json.loads(out.read_text())
    assert d["derivedFromTree"] == tree1 and set(d["corpora"]["python"]["files"]) == {"falkorchat/alpha.py"}
    assert run("--out", str(out), "--check").returncode == 0
    assert run("--out", str(out)).returncode == 0                      # default: HEAD
    d2 = json.loads(out.read_text())
    assert d2["derivedFromTree"] == _git(r, "rev-parse", "HEAD^{tree}") and set(d2["corpora"]["python"]["files"]) == {"falkorchat/alpha.py", "falkorchat/beta.py"}
    out.write_text(out.read_text().replace('"artifacts": 1', '"artifacts": 9'))
    assert run("--out", str(out), "--check").returncode == 1


def test_check_fails_on_difference_unresolved_missing_or_non_tree_hash(tmp_path):
    r = _repo(tmp_path)
    out = tmp_path / "contact.json"
    good = json.loads(mc.render(mc.derive(repo=str(r))))
    def put(o):
        out.write_text(mc.render(o))
        return mc.check(str(out), str(r))
    assert put(good) == (True, "ok")
    bad = json.loads(json.dumps(good)); bad["corpora"]["python"]["files"]["falkorchat/alpha.py"]["artifacts"] += 1
    assert put(bad)[0] is False
    bad = json.loads(json.dumps(good)); bad["flowAnchors"]["python"].append("falkorchat/beta.py")
    assert put(bad)[0] is False
    bad = json.loads(json.dumps(good)); bad["derivedFromTree"] = "0" * 40
    assert "does not resolve" in put(bad)[1]
    bad = json.loads(json.dumps(good)); bad["derivedFromTree"] = None
    assert put(bad)[0] is False
    bad = json.loads(json.dumps(good)); bad["derivedFromTree"] = _git(r, "rev-parse", "HEAD")      # a commit sha is not a tree hash
    assert "does not resolve" in put(bad)[1]
    # a changed input: the pinned tree is an older one than the content was derived from
    old_tree = _git(r, "rev-parse", "HEAD^{tree}")
    (r / "cpg/tests/t3.py").write_text("falkor-chat/server/falkorchat/beta.py\n"); _git(r, "add", "-A"); _git(r, "commit", "-qm", "c3")
    new = json.loads(mc.render(mc.derive(repo=str(r))))
    assert new["derivedFromTree"] != old_tree and new["corpora"] != good["corpora"]
    new["derivedFromTree"] = old_tree
    ok, msg = put(new)
    assert ok is False and "differs" in msg
    out.unlink()
    assert mc.check(str(out), str(r))[0] is False


def test_check_survives_a_history_rewrite_that_keeps_the_tree(tmp_path):
    r = _repo(tmp_path)
    out = tmp_path / "contact.json"
    out.write_text(mc.render(mc.derive(repo=str(r))))
    assert mc.check(str(out), str(r)) == (True, "ok")
    # "rebase": the same tree under a different, parentless commit; the original commit becomes unreachable
    old = _git(r, "rev-parse", "HEAD")
    new = _git(r, "commit-tree", "HEAD^{tree}", "-m", "rewritten")
    _git(r, "reset", "-q", "--hard", new)
    assert _git(r, "rev-parse", "HEAD") != old
    assert subprocess.run(["git", "-C", str(r), "merge-base", "--is-ancestor", old, "HEAD"]).returncode != 0
    assert mc.check(str(out), str(r)) == (True, "ok")


def test_deterministic(tmp_path):
    t = _tree(tmp_path, {**BASE, "cpg/tests/t.py": "falkorchat.alpha falkorchat/beta.py\n"})
    assert mc.render(mc.derive(t)) == mc.render(mc.derive(t))
