"""P1-57a (red first): the `fetch` pack seeds only in a JavaScript/TypeScript source file (defect D8).

Plan: `docs/plans/connected-code-graph-p1-linking.md` P1-57 (57a). Cause: `docs/test-reports/connected-code-graph-p1-report.md` D8: the seed
matched any callee named `fetch`, so Python's `self.session.fetch(item_id)` was read as a keyless HTTP consumer. `fetch` is a JavaScript global;
the seed now carries `scope.path`, a positive file gate on the extension.

``tests/data/p1_57a_fetch_sites.json`` holds the real extractor records of the proving repository's two JS/TS `fetch` sites (regenerate with
``tests/data/p1_57a_fetch_sites_regen.py``); the census guards that the gate matches at least one real file path.
"""
import copy
import json
from pathlib import Path

import pytest

from packrt_helpers import call, file_, sym, vs
from ccg import units
from ccg.packrt import run_packs
from ccg.packrt.match import text_hit
from ccg.rulepack import load_pack

CPG = Path(__file__).resolve().parents[1]
#: the shipped file, never `packrt_helpers.pack` (which can read the spec block, where the gate is not yet written: the spec owes it at the close)
FETCH = load_pack(CPG / "packs" / "fetch" / "pack.yaml")
GOLDEN = json.loads((CPG / "tests" / "data" / "p1_57a_fetch_sites.json").read_text())["records"]

SEEDS = ["a.ts", "web/a.tsx", "web/a.js", "a.jsx", "a.cjs", "a.mjs", "a.cts", "a.mts", "deep/er/path/x.ts"]
#: a planted wrong input per axis the gate could wrongly accept: another language, a shell, docs, a near-extension, an extension that is only a
#: directory, a bare name, an extension that is only a prefix of the suffix.
NOT_SEEDS = ["a.py", "svc/state/stores.py", "run.sh", "README.md", "a.tsx.bak", "foo.ts/bar.py", "ts", "a.json", "a.tsx.py", "a.jsonl", "a.ts~", "foots", "a_tsx", "app/Page.vue"]


def site(path):
    """One function in `path` holding a literal `fetch("/api/x")` call (a leaf: no caller)."""
    return [file_("f1", path, lang="javascript"), sym("go", "go", fid="f1", path=path),
            call("fe", "go", "fetch", line=5, resolution="unresolved"), vs("fe", "a0", "/api/x")]


@pytest.mark.parametrize("path", SEEDS)
def test_a_js_or_ts_file_seeds_the_fetch_lifting(path):
    out = run_packs([site(path)], [FETCH])
    assert [c["key"] for c in out.consumers] == ["GET /api/x"]
    assert len(out.lifted) == 1


@pytest.mark.parametrize("path", NOT_SEEDS)
def test_any_other_file_makes_no_fact_and_no_lifted_leaf(path):
    out = run_packs([site(path)], [FETCH])
    assert out.consumers == []
    assert out.lifted == []


def test_a_python_method_named_fetch_is_not_a_consumer():
    """The D8 shape: `self.session.fetch(item_id)` (no literal, an unresolved member call) in a Python file."""
    recs = [file_("f1", "svc/state/stores.py"), sym("load", "load", fid="f1", path="svc/state/stores.py", qname="svc/state/stores.py::DbStore.load"),
            call("fe", "load", "fetch", line=14, resolution="unresolved"), vs("fe", "a0", {"unknown": "point-derived"})]
    out = run_packs([recs], [FETCH])
    assert out.consumers == [] and out.lifted == []


def test_the_real_js_and_ts_fetch_sites_still_seed():
    paths = sorted(r["path"] for r in GOLDEN if r["type"] == "file")
    assert paths == ["falkor-chat/web/app.js", "salesperson/src/api/client.ts"]
    out = run_packs([GOLDEN], [FETCH])
    assert len(out.lifted) == 2, out.lifted


def test_census_the_gate_matches_at_least_one_real_file_path_and_rejects_the_python_twin():
    """Real records (the extractor's file paths, not written strings): renaming their files to `.py` removes every seed, so the matching above
    is the gate's doing, not an artifact of the record shape."""
    py = copy.deepcopy(GOLDEN)
    for r in py:
        if r["type"] == "file":
            r["path"] = r["path"].rsplit(".", 1)[0] + ".py"
        if "path" in r and r["type"] == "symbol":
            r["path"] = r["path"].rsplit(".", 1)[0] + ".py"
    out = run_packs([py], [FETCH])
    assert out.lifted == [] and out.consumers == []


# ---------------------------------------------------------------------------------------------------- the gate is tied to the extractor's own extensions

def js_family(source_ext=None) -> set[str]:
    """The extensions the extractor treats as JavaScript/TypeScript source (`ccg.units._SOURCE_EXT`): the only files that ever get a `file` record."""
    return {e for e, lang in (units._SOURCE_EXT if source_ext is None else source_ext).items() if lang == "javascript"}


def uncovered(exts) -> list[str]:
    """Extensions the shipped seed's `scope.path` does not match (a `fetch` call in such a file would silently stop seeding)."""
    (scope,) = [s["scope"]["path"] for lf in FETCH["liftings"] for s in lf["seeds"]]
    return sorted(e for e in exts if not text_hit(scope, "src/x" + e, path=True))


def test_census_the_gate_matches_every_extension_the_extractor_emits_for_javascript():
    exts = js_family()
    assert exts == {".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"}    # the real set is non-empty and as documented
    assert uncovered(exts) == []


def test_a_new_javascript_family_extension_would_fail_the_census():
    """Mutant: the extractor learns one more extension; the census must name it (the regex is a hand-written copy of that set)."""
    grown = {**units._SOURCE_EXT, ".xjs": "javascript"}
    assert uncovered(js_family(grown)) == [".xjs"]


def test_the_gate_has_no_alternative_the_extractor_never_emits():
    """Both directions: every alternative of the regex is an extension the extractor emits (`.vue` is never a source file, so it is not there)."""
    (scope,) = [s["scope"]["path"] for lf in FETCH["liftings"] for s in lf["seeds"]]
    alts = scope["regex"].split("(", 1)[1].rstrip(")").split("|")
    assert {"." + a for a in alts} == js_family()
