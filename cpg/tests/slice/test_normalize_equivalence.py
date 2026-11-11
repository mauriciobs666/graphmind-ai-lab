"""P1-05 behaviour (2) — the ported normalizer reproduces the vertical slice's stored bundles (**5a: Python half**).

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § P1-05 (2); the rule is `docs/plans/connected-code-graph-p1.md` §8 ("Equivalence before improvement") and D7; D9 is the skip rule below. The oracle is the slice's
S14a stored bundle for `falkor-chat/server`, `cpg/.ccg-work/vertical-slice/s14a/<unitId>.bundle.jsonl` (sha256 pinned
below), with the raw facts and staged tree beside it. These tests carry the `slice` marker and skip when an artifact is
absent: **a skip is unrun, never green.** The run summary must read `0 skipped`.

**5b adds the whole bundle**, both units (`falkor-chat/server` and `salesperson`): the bundle the production code writes through
`ccg.atrest.write_bundle` is compared with the stored one under three definitions (strict bytes with the slice's own
`extract.sc` as the hashed script; bytes with the three header fields that are inputs or stamps exempt; the multiset of
records), see `test_whole_bundle_*`.

The first dispatch covers what the Python frontend decides: every `symbol` record's identity, kind, name, parent, span
and fingerprint (all but `complexity` and `lastChanged*`, which need conds and blame; 5b) and every `import` record on every field. Expected values are read from the stored bundle, never from the code under test. The whole-bundle byte
comparison (and the JS unit, `salesperson`) is the second dispatch.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from ccg import atrest, ids
from ccg.units import Unit
from extractors.joern import run
from extractors.joern.normalize import assemble_bundle, core, python

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


# -- 5b: the whole bundle, both units -------------------------------------------------------------------------------
REPO = CPG.parent
SLICE_SCRIPT = CPG / "spikes" / "vertical-slice" / "extract.sc"
SP_ROOT = "salesperson"
SP_BUNDLE = S14A / "c2341bca0404a6a3eadab1d72ba5f4db.bundle.jsonl"
SP_BUNDLE_SHA = "f2ef6f5add18691d94cecfcb05a742e03d3e06f00cf3fedea462d79886e13e65"
UNITS = {"fcs": (FCS_ROOT, "python", FCS_BUNDLE, FCS_BUNDLE_SHA), "sp": (SP_ROOT, "javascript", SP_BUNDLE, SP_BUNDLE_SHA)}
#: header fields that are an input or a stamp, not content: spec3 §8.5 (`builtAt`, `sandboxAuditIds`) and the hash of the
#: script file, which differs between the slice's `extract.sc` and P1-04's port by its header comment alone.
EXEMPT = ("builtAt", "sandboxAuditIds")


def raw_of(tag: str, commit: str, audit_ids) -> run.RawFacts:
    root, lang = UNITS[tag][:2]
    cpg = need(S14A / tag / "cpg")
    facts = sorted(cpg.glob("*.facts.jsonl"))
    astgen = sorted(cpg.glob("*.astgen"))
    unit = Unit(ids.unit_id(REPO_ID, root, lang), REPO_ID, root, lang)
    return run.RawFacts(unit, commit, facts[0], tuple(audit_ids), need(S14A / tag / "stage"), cpg,
                        astgen[0] if astgen else None, cpg / "none", 0)


def stored_lines(tag: str) -> list[bytes]:
    return need(UNITS[tag][2]).read_bytes().split(b"\n")[:-1]


@pytest.fixture(scope="module", params=["fcs", "sp"])
def written(request, tmp_path_factory):
    """(tag, written bytes, written path) of the unit's bundle, built by `build_bundle` through `ccg.atrest`, with the stored
    header's own `builtAt`, audit ids and the slice's script, so that nothing in it is exempt."""
    tag = request.param
    root, lang, bundle_file, _ = UNITS[tag]
    need(bundle_file)
    header = json.loads(stored_lines(tag)[0])
    work = tmp_path_factory.mktemp("work")
    os.chmod(work, 0o700)
    raw = raw_of(tag, header["builtCommit"], header["sandboxAuditIds"])
    path = assemble_bundle.build_bundle(raw.unit, raw, REPO, header["builtCommit"], system="slice", root=work,
                                        built_at=header["builtAt"], extract_script=SLICE_SCRIPT)
    return tag, path.read_bytes(), path


@pytest.mark.slice
@pytest.mark.parametrize("tag", ["fcs", "sp"])
def test_the_stored_bundles_are_the_artifacts_recorded_with_their_sha256(tag):
    assert hashlib.sha256(need(UNITS[tag][2]).read_bytes()).hexdigest() == UNITS[tag][3]


@pytest.mark.slice
def test_whole_bundle_is_byte_identical_to_the_stored_one(written):
    """Definition 1 (strict): every byte, the header included; the inputs the stamps come from are the stored header's."""
    tag, data, _ = written
    assert data == need(UNITS[tag][2]).read_bytes()


@pytest.mark.slice
def test_whole_bundle_is_identical_with_the_ported_extract_script_but_for_its_hash(written, tmp_path):
    """Definition 2: with the production `extract.sc` the only difference is `tools.extractScriptHash`, and it is that file's sha256."""
    tag, _, _ = written
    header = json.loads(stored_lines(tag)[0])
    raw = raw_of(tag, header["builtCommit"], header["sandboxAuditIds"])
    os.chmod(tmp_path, 0o700)
    path = assemble_bundle.build_bundle(raw.unit, raw, REPO, header["builtCommit"], system="slice", root=tmp_path,
                                        built_at=header["builtAt"])
    got, want = path.read_bytes().split(b"\n")[:-1], stored_lines(tag)
    gh, wh = json.loads(got[0]), json.loads(want[0])
    assert gh["tools"].pop("extractScriptHash") == hashlib.sha256(run.EXTRACT_SC.read_bytes()).hexdigest()
    assert wh["tools"].pop("extractScriptHash") == hashlib.sha256(SLICE_SCRIPT.read_bytes()).hexdigest()
    assert gh == wh
    assert got[1:] == want[1:]


@pytest.mark.slice
def test_whole_bundle_records_are_the_same_multiset_per_type(written):
    """Definition 3 (weakest): per record type, the same multiset of canonical lines, and the same header counts."""
    tag, data, _ = written
    def by_type(lines):
        out = collections.defaultdict(collections.Counter)
        for ln in lines[1:]:
            out[json.loads(ln)["type"]][ln] += 1
        return out
    got, want = by_type(data.split(b"\n")[:-1]), by_type(stored_lines(tag))
    assert {t: sum(c.values()) for t, c in got.items()} == {t: sum(c.values()) for t, c in want.items()}
    assert got == want
    assert json.loads(data.split(b"\n")[0])["counts"] == json.loads(stored_lines(tag)[0])["counts"]


@pytest.mark.slice
def test_the_written_bundle_is_private_0600_in_a_0700_directory(written):
    _, _, path = written
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert path.parent.name == "slice" and path.parent.parent.name == "bundles"


# -- 5b live tier: two real Joern runs of one stage ------------------------------------------------------------------
PINNED = "defca60ff16534aa12a74e35856de9be98bfc710"


def _without_exempt(data: bytes) -> list[bytes]:
    lines = data.split(b"\n")[:-1]
    header = json.loads(lines[0])
    for k in EXEMPT:
        header.pop(k)
    return [json.dumps(header, sort_keys=True).encode()] + lines[1:]


@pytest.mark.slice
@pytest.mark.live
def test_two_real_joern_runs_of_one_unit_give_one_bundle_and_it_is_the_stored_one(tmp_path, monkeypatch):
    """The raw `d`/`cb` facts differ in file order between Joern runs (they agree as multisets); the bundle must not. The unit is
    `salesperson` (JS, the smaller one): two full extractions through the runner at the slice's pinned commit, each normalized
    with the stored header's `builtAt` and the slice's script, then compared with each other and with the stored bundle,
    header `sandboxAuditIds` (new audit ids every run) exempt."""
    monkeypatch.setenv("CCG_AUDIT_DIR", str(tmp_path / "audit"))     # never the real cpg/.audit/sandbox.jsonl
    stored_header = json.loads(stored_lines("sp")[0])
    unit = Unit(ids.unit_id(REPO_ID, SP_ROOT, "javascript"), REPO_ID, SP_ROOT, "javascript")
    area = tmp_path / ".ccg-work"
    area.mkdir(mode=0o700)
    raws = [run.extract_unit(REPO, unit, [unit], PINNED, area / f"run{i}", area=area) for i in (1, 2)]
    facts = [core.load_facts(r.facts) for r in raws]
    for kind in ("d", "cb"):                                          # same multiset, whatever the order
        key = lambda f: json.dumps(f, sort_keys=True)
        assert sorted(map(key, facts[0][kind])) == sorted(map(key, facts[1][kind]))
    out = []
    for i, raw in enumerate(raws, start=1):
        path = assemble_bundle.build_bundle(unit, raw, REPO, PINNED, system="live", root=area, build=f"run{i}",
                                            built_at=stored_header["builtAt"], extract_script=SLICE_SCRIPT)
        out.append(path.read_bytes())
    assert raws[0].sandbox_audit_ids != raws[1].sandbox_audit_ids          # every run has its own audit ids ...
    assert _without_exempt(out[0]) == _without_exempt(out[1])             # ... and they are all that tells the bundles apart
    assert _without_exempt(out[0]) == _without_exempt(out[1]) == _without_exempt(need(SP_BUNDLE).read_bytes())
