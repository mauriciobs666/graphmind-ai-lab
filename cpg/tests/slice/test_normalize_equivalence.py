"""P1-05 behaviour (2) — the ported normalizer reproduces the vertical slice's stored bundles (**5a: Python half**).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-05 (2); the rule is `docs/plans/connected-code-graph-p1.md` §8 ("Equivalence before improvement") and D7; D9 is the skip rule below. The oracle is the slice's
S14a stored bundle for `falkor-chat/server`, `cpg/.ccg-work/vertical-slice/s14a/<unitId>.bundle.jsonl` (sha256 pinned
below), with the raw facts and staged tree beside it. These tests carry the `slice` marker and skip when an artifact is
absent: **a skip is unrun, never green.** The run summary must read `0 skipped`.

The first dispatch covers what the Python frontend decides: every `symbol` record's identity, kind, name, parent, span
and fingerprint (all but `complexity` and `lastChanged*`, which need conds and blame; 5b) and every `import` record on every field. Expected values are read from the stored bundle, never from the code under test. The whole-bundle byte
comparison (and the JS unit, `salesperson`) is the second dispatch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ccg import ids
from extractors.joern.normalize import core, python

CPG = Path(__file__).resolve().parent.parent.parent
S14A = CPG / ".ccg-work" / "vertical-slice" / "s14a"
REPO_ID = "graphmind-ai-lab"
FCS_ROOT = "falkor-chat/server"
FCS_BUNDLE = S14A / "0fb899742d4d3687495c30357ca16253.bundle.jsonl"
#: sha256 of the stored bundle recorded 2026-11-10 (read from the file in this run).
FCS_BUNDLE_SHA = "a0cae513fb6795605a5818942942869fbb1c575de7cdbd4a264ff82833ba1c86"
#: counts recorded in the stored bundle's header (`counts`).
FCS_COUNTS = {"symbol": 4686, "import": 926, "file": 107}
#: stored `import` records that carry a `symbolId` (counted in the stored bundle in this run).
FCS_IMPORTS_WITH_SYMBOL = 87


def need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"slice artifact absent: {path} (UNRUN, not green)")
    return path


def stored(kind: str) -> list[dict]:
    out = []
    for line in need(FCS_BUNDLE).read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o.get("type") == kind:
            out.append(o)
    return out


def facts_of():
    cpg = need(S14A / "fcs" / "cpg")
    found = sorted(cpg.glob("*.facts.jsonl"))
    assert len(found) == 1
    return core.load_facts(found[0])


@pytest.fixture(scope="module")
def built():
    facts = facts_of()
    B = core.Builder("python", FCS_ROOT, REPO_ID, facts, core.Source(need(S14A / "fcs" / "stage")))
    scopes, _, _ = python.build_scopes(B)
    python.apply_py_spans(B, scopes)
    for s in scopes.values():
        s.sid = ids.symbol_id("python", REPO_ID, s.qname)
    return B, scopes


@pytest.mark.slice
def test_the_stored_bundle_is_the_artifact_recorded_with_its_sha256_and_counts():
    assert hashlib.sha256(need(FCS_BUNDLE).read_bytes()).hexdigest() == FCS_BUNDLE_SHA
    header = json.loads(FCS_BUNDLE.read_text(encoding="utf-8").splitlines()[0])
    assert {k: header["counts"][k] for k in FCS_COUNTS} == FCS_COUNTS


@pytest.mark.slice
def test_every_stored_symbol_is_rebuilt_field_for_field(built):
    B, scopes = built
    want = {r["id"]: r for r in stored("symbol")}
    assert len(want) == FCS_COUNTS["symbol"]
    got = {}
    for s in scopes.values():
        rec = {"id": s.sid, "qname": s.qname, "path": B.P(s.file), "fileId": ids.file_id(REPO_ID, B.P(s.file)),
               "kind": s.kind, "name": core.sym_name(s), "nameNorm": core.name_norm(core.sym_name(s)),
               "lineStart": core.span_start(B, s), "lineEnd": max(s.lineEnd, core.span_start(B, s)),
               "fingerprint": core.fingerprint(B, s)}
        if s.parent is not None and s.kind != "module":
            rec["parentId"] = s.parent.sid
        got[s.sid] = rec
    assert set(got) == set(want)
    keys = ("qname", "path", "fileId", "kind", "name", "nameNorm", "lineStart", "lineEnd", "fingerprint", "parentId")
    bad = [(i, k) for i, r in got.items() for k in keys if r.get(k) != want[i].get(k)]
    assert bad == []


@pytest.mark.slice
def test_every_stored_import_is_rebuilt_field_for_field(built):
    B, scopes = built
    files = sorted(f for f in core.lang_files(B.facts, "python") if B.src.exists(f))
    got = []
    for f in files:
        for k, r in enumerate(python.py_imports(B, f)):
            fid = ids.file_id(REPO_ID, B.P(f))
            rec = {"id": ids.import_id(fid, k), "fileId": fid, "line": r["line"], "ordinal": k, "module": r["module"],
                   "names": r["names"], "dynamic": r["dynamic"]}
            if r.get("resolvedPath"):
                rec["resolvedPath"] = r["resolvedPath"]
            sid = core.import_symbol_id(B, scopes, f, r["line"])
            if sid:
                rec["symbolId"] = sid
            got.append(rec)
    want = [{k: v for k, v in r.items() if k != "type"} for r in stored("import")]
    assert sum("symbolId" in r for r in want) == FCS_IMPORTS_WITH_SYMBOL
    assert len(want) == FCS_COUNTS["import"]
    key = lambda r: r["id"]
    assert sorted(got, key=key) == sorted(want, key=key)


@pytest.mark.slice
def test_no_symbol_of_the_stored_unit_needed_the_orphan_parent_fallback(built):
    assert built[0].report["pythonParentFallbacks"] == 0
