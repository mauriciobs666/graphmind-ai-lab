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
def test_contact_json_matches_rederivation():
    assert os.path.exists(CONTACT_JSON), "run make_contact.py"
    assert open(CONTACT_JSON).read() == mc.render(mc.derive()), "contact.json drifted: re-run make_contact.py and review the diff"


def test_contact_json_evidence_is_rederivable_from_head():
    d = json.load(open(CONTACT_JSON))
    for corpus, c in d["corpora"].items():
        for f, x in c["files"].items():
            assert 1 <= len(x["evidence"]) <= 3
            assert len({e["artifact"] for e in x["evidence"]}) == len(x["evidence"])
            for e in x["evidence"]:
                assert not e["artifact"].startswith("cpg/spikes/frontend-fitness/")
                body = subprocess.run(["git", "show", "HEAD:" + e["artifact"]], capture_output=True, text=True, cwd=FF).stdout
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
    got, _ = _derive(tmp_path, {"docs/plans/connected-code-graph-x.md":
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
    got, d = _derive(tmp_path, {"docs/reviews/connected-code-graph-y.md": "`falkorchat/alpha.py:12`, `src/api/client.ts::apiFetch`\n"})
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
    got, _ = _derive(tmp_path, {"docs/test-reports/connected-code-graph-r.md": "`alpha.py::f` and `Panel.tsx:40`, but beta.py alone\n"})
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
    "cpg/packs/p/pack.yaml": "packs", "docs/plans/connected-code-graph-p.md": "docs", "docs/reviews/connected-code-graph-r.md": "docs",
    "docs/test-plans/connected-code-graph-t.md": "docs", "docs/test-reports/connected-code-graph-rep.md": "docs", "cpg/ccg/mod.py": "cpg-other"}


def test_each_search_source_is_searched(tmp_path):
    for art in EXPECTED_SOURCE:
        got, d = _derive(tmp_path / art.replace("/", "_"), {art: "falkor-chat/server/falkorchat/alpha.py\n"})
        assert got["python"] == {"falkorchat/alpha.py"}, art
        assert mc.source_of(art) == EXPECTED_SOURCE[art]
        assert d["searchSources"][EXPECTED_SOURCE[art]] >= 1


def test_out_of_scope_artifacts_are_not_searched(tmp_path):
    text = "falkor-chat/server/falkorchat/alpha.py\n"
    got, _ = _derive(tmp_path, {"cpg/spikes/frontend-fitness/labels/x.jsonl": text, "cpg/spikes/frontend-fitness/README.md": text,
                                "cpg/tests/test_contact.py": text, "docs/plans/salesperson-ui.md": text, "README.md": text,
                                "falkor-chat/server/falkorchat/beta.py": text})
    assert got["python"] == set()


def test_evidence_capped_at_three_and_prefers_path_form(tmp_path):
    extra = {f"cpg/tests/test_{i}.py": "falkorchat/alpha.py\n" for i in range(5)}
    extra["docs/plans/connected-code-graph-z.md"] = "falkor-chat/server/falkorchat/alpha.py\n"
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


def test_deterministic(tmp_path):
    t = _tree(tmp_path, {**BASE, "cpg/tests/t.py": "falkorchat.alpha falkorchat/beta.py\n"})
    assert mc.render(mc.derive(t)) == mc.render(mc.derive(t))
