"""P1-15 — `extractors/joern/normalize/merge_scip.py`: the merge rule of `docs/plans/connected-code-graph-p1-ml.md` §2.2.

J is Joern's in-corpus targets of a call site, S the in-corpus target SCIP gives the same ``(file, line, col)``.
The rule's table is written out case by case below; no candidate is ever dropped, so ``R_m >= R_j``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scip_helpers as h
from ccg import ids
from extractors.indexers import scip
from extractors.joern.normalize import merge_scip as ms

CPG = Path(__file__).resolve().parent.parent
LANG, REPO = "python", "proj"


def sym(q: str) -> str:
    return ids.symbol_id(LANG, REPO, q)


Q_HELPER, Q_PUT, Q_OTHER, Q_INIT = "pkg/a.py::helper", "pkg/a.py::Box.put", "pkg/a.py::other", "pkg/a.py::Box.__init__"
KNOWN = {q: sym(q) for q in (Q_HELPER, Q_PUT, Q_OTHER, Q_INIT)}
ENCLOSING = sym("pkg/b.py::run")


def tgt(q: str) -> dict:
    return {"symbolId": KNOWN[q], "qname": q}


def site(name: str, ordinal: int, targets: list[str] = (), *, external: str | None = None) -> dict:
    rec = {"type": "callsite", "id": ids.callsite_id(ENCLOSING, name, ordinal), "symbolId": ENCLOSING, "calleeName": name,
           "ordinal": ordinal, "callKind": "call", "line": 4}
    if targets:
        rec.update(resolution="resolved", targets=sorted((tgt(q) for q in targets), key=lambda t: t["symbolId"]),
                   status="confirmed", reason="resolved-call")
    elif external:
        rec.update(resolution="external", externalTarget=external, targets=[], status="possible", reason="name-match")
    else:
        rec.update(resolution="unresolved", targets=[], status="possible", reason="name-match")
    return rec


def index_of(*occ: tuple[int, int, str], defs_in="pkg/a.py") -> scip.ScipIndex:
    """A one-document index of ``pkg/b.py`` references (0-based line/col as SCIP writes them) plus the definitions."""
    a = h.document(defs_in, [h.occurrence([0, 4, 10], h.HELPER, h.DEF), h.occurrence([4, 6, 9], h.BOX, h.DEF),
                             h.occurrence([5, 8, 11], h.BOX_PUT, h.DEF), h.occurrence([6, 8, 13], h.BOX_INIT, h.DEF),
                             h.occurrence([9, 4, 9], h.OTHER, h.DEF)])
    b = h.document("pkg/b.py", [h.occurrence([ln, c, c + 3], s) for ln, c, s in occ])
    return scip.parse_bytes(h.index([a, b]))


def merge(sites, idx, positions=None, known=KNOWN, prefix=""):
    positions = positions if positions is not None else {s["id"]: ("pkg/b.py", 4, 4) for s in sites}
    return ms.merge_scip(sites, positions, idx, known, prefix=prefix, lang=LANG)


def only(res):
    assert len(res.callsites) == 1
    return res.callsites[0]


def tgt_ids(rec):
    return [t["symbolId"] for t in rec["targets"]]


# --- the table ---------------------------------------------------------------------------------------------------


def test_agree_confirms_the_site_with_the_scip_target():
    s = site("helper", 0, [Q_HELPER])
    res = merge([s], index_of((3, 4, h.HELPER)))
    r = only(res)
    assert (r["status"], r["reason"], r["resolution"]) == ("confirmed", "resolved-call", "resolved")
    assert r["targets"] == [tgt(Q_HELPER)]
    assert res.candidates == {} and res.counts["agree"] == 1


def test_agree_keeps_the_other_joern_targets_as_possible_candidates():
    s = site("helper", 0, [Q_HELPER, Q_OTHER])
    res = merge([s], index_of((3, 4, h.HELPER)))
    r = only(res)
    assert tgt_ids(r) == [KNOWN[Q_HELPER]] and r["status"] == "confirmed"
    assert res.candidates[s["id"]] == [{**tgt(Q_OTHER), "status": "possible", "reason": "resolution-unverified"}]


def test_upgrade_an_unresolved_name_match_site():
    s = site("put", 0)
    res = merge([s], index_of((3, 4, h.BOX_PUT)))
    r = only(res)
    assert (r["status"], r["reason"], r["resolution"]) == ("confirmed", "resolved-call", "resolved")
    assert r["targets"] == [tgt(Q_PUT)] and res.counts["upgrade"] == 1 and res.candidates == {}


def test_upgrade_drops_the_externalTarget_of_a_formerly_external_site():
    s = site("put", 0, external="some.lib.put")
    r = only(merge([s], index_of((3, 4, h.BOX_PUT))))
    assert r["resolution"] == "resolved" and "externalTarget" not in r


def test_upgrade_a_class_instantiation_goes_to_the_constructor():
    s = site("Box", 0)
    r = only(merge([s], index_of((3, 4, h.BOX))))
    assert r["targets"] == [tgt(Q_INIT)] and r["status"] == "confirmed"


def test_disagree_keeps_both_and_marks_every_candidate_possible_resolution_unverified():
    s = site("helper", 0, [Q_OTHER])
    res = merge([s], index_of((3, 4, h.HELPER)))
    r = only(res)
    # the site keeps the schema-valid name-match; the candidates carry resolution-unverified (see merge_scip's docstring)
    assert (r["status"], r["reason"], r["resolution"], r["targets"]) == ("possible", "name-match", "unresolved", [])
    got = {c["qname"]: (c["status"], c["reason"]) for c in res.candidates[s["id"]]}
    assert got == {Q_OTHER: ("possible", "resolution-unverified"), Q_HELPER: ("possible", "resolution-unverified")}
    assert res.counts["disagree"] == 1


def test_scip_external_against_an_in_corpus_joern_target_is_a_disagreement():
    s = site("len", 0, [Q_HELPER])
    res = merge([s], index_of((3, 4, h.LEN)))
    r = only(res)
    assert (r["status"], r["reason"]) == ("possible", "name-match")
    assert [c["qname"] for c in res.candidates[s["id"]]] == [Q_HELPER]


def test_scip_local_against_an_in_corpus_joern_target_is_a_disagreement():
    s = site("helper", 0, [Q_HELPER])
    r = only(merge([s], index_of((3, 4, h.LOCAL))))
    assert (r["status"], r["reason"]) == ("possible", "name-match")


def test_scip_external_with_no_joern_target_leaves_the_site_alone():
    s = site("len", 0)
    res = merge([s], index_of((3, 4, h.LEN)))
    assert res.callsites == [s] and res.counts["silent"] == 1


def test_scip_silent_leaves_joern_unchanged():
    s = site("helper", 0, [Q_HELPER])
    res = merge([s], index_of((20, 0, h.HELPER)))
    assert res.callsites == [s]


def test_a_symbol_not_in_the_bundle_counts_as_external():
    s = site("put", 0)
    res = merge([s], index_of((3, 4, h.BOX_PUT)), known={Q_HELPER: KNOWN[Q_HELPER]})
    assert res.callsites == [s]


def test_two_distinct_in_corpus_scip_targets_at_one_position_leave_the_site_alone():
    s = site("helper", 0, [Q_HELPER])
    res = merge([s], index_of((3, 4, h.HELPER), (3, 4, h.BOX_PUT)))
    assert res.callsites == [s] and res.counts["ambiguous"] == 1


def test_a_site_without_a_position_is_unchanged():
    s = site("helper", 0)
    res = merge([s], index_of((3, 4, h.HELPER)), positions={})
    assert res.callsites == [s]


def test_the_unit_prefix_is_part_of_the_qname():
    q = "svc/pkg/a.py::helper"
    s = site("helper", 0)
    s["targets"] = []
    known = {q: sym(q)}
    res = merge([s], index_of((3, 4, h.HELPER)), known=known, prefix="svc")
    assert only(res)["targets"] == [{"symbolId": sym(q), "qname": q}]


# --- SCIP-only ---------------------------------------------------------------------------------------------------


def test_scip_only_calls_are_reported_not_synthesized():
    s = site("helper", 0, [Q_HELPER])
    idx = index_of((3, 4, h.HELPER), (10, 8, h.BOX_PUT), (12, 0, h.BOX), (13, 0, h.LEN))
    res = merge([s], idx)
    assert len(res.callsites) == 1                                   # no record is minted
    assert [(o.file, o.line, o.col, o.qname, o.symbolId) for o in res.scip_only] == [
        ("pkg/b.py", 11, 8, Q_PUT, KNOWN[Q_PUT]), ("pkg/b.py", 13, 0, Q_INIT, KNOWN[Q_INIT])]   # the class: its constructor
    assert res.counts["scip_only"] == 2


def test_a_reference_that_is_not_callable_is_not_scip_only():
    idx = index_of((3, 4, h.BOX))
    res = merge([], idx, known={Q_HELPER: KNOWN[Q_HELPER]})            # `Box` has no known constructor here
    assert res.scip_only == []


# --- invariants --------------------------------------------------------------------------------------------------

J_CASES = [[], [Q_HELPER], [Q_OTHER], [Q_HELPER, Q_OTHER], [Q_PUT, Q_OTHER]]
S_CASES = [None, h.HELPER, h.OTHER, h.BOX_PUT, h.LEN, h.LOCAL, h.BOX]


@pytest.mark.parametrize("j", J_CASES)
@pytest.mark.parametrize("s_sym", S_CASES)
def test_no_candidate_is_ever_dropped(j, s_sym):
    s = site("helper", 0, j)
    idx = index_of(*([(3, 4, s_sym)] if s_sym else []))
    res = merge([s], idx)
    r = only(res)
    kept = set(tgt_ids(r)) | {c["symbolId"] for c in res.candidates.get(s["id"], [])}
    assert {KNOWN[q] for q in j} <= kept                                # R_m >= R_j, site by site
    if r["status"] == "confirmed":
        assert r["reason"] == "resolved-call" and r["targets"]
    else:
        assert r["targets"] == []
    assert r["symbolId"] == s["symbolId"] and r["id"] == s["id"] and r["ordinal"] == s["ordinal"]


def test_inputs_are_not_mutated_and_the_output_is_deterministic():
    sites = [site("helper", 0, [Q_OTHER]), site("put", 0)]
    pos = {sites[0]["id"]: ("pkg/b.py", 4, 4), sites[1]["id"]: ("pkg/b.py", 7, 8)}
    before = copy.deepcopy(sites)
    idx = index_of((3, 4, h.HELPER), (6, 8, h.BOX_PUT))
    a, b = merge(sites, idx, pos), merge(sites, idx, pos)
    assert sites == before and a.callsites == b.callsites and a.candidates == b.candidates


def test_merged_records_validate_against_the_bundle_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((CPG / "ccg" / "schemas" / "bundle-v1.schema.json").read_text())
    v = jsonschema.Draft202012Validator({"$ref": "#/$defs/callsite", "$defs": schema["$defs"]})
    sites = [site("a", 0, [Q_OTHER]), site("b", 0), site("c", 0, [Q_HELPER]), site("d", 0, external="x.y")]
    pos = {s["id"]: ("pkg/b.py", 4 + i, 4) for i, s in enumerate(sites)}
    idx = index_of((3, 4, h.HELPER), (4, 4, h.BOX_PUT), (5, 4, h.HELPER), (6, 4, h.LEN))
    for r in merge(sites, idx, pos).callsites:
        v.validate(r)


def test_every_code_the_merge_writes_is_in_the_registry():
    import yaml
    rows = {r["code"] for r in yaml.safe_load((CPG / "ccg" / "registry" / "reasons.yaml").read_text())}
    assert {"resolved-call", "name-match", ms.CANDIDATE_REASON, ms.DISAGREE_SITE_REASON} <= rows


# --- the real-tool golden: scip-typescript 0.6.0 over a toy project, and the bundle Joern + the normalizer built for it ---

REAL = CPG / "tests" / "data" / "scip"


@pytest.fixture(scope="module")
def real():
    """``tests/data/scip/real_ts.{scip,bundle.jsonl,positions.json}`` (sources in ``real_ts_src/``): generated once by
    ``run_indexer`` (the live path) and ``extract_unit`` + ``build_bundle`` over a git copy of those sources, unit root ``web``."""
    records = [json.loads(l) for l in (REAL / "real_ts.bundle.jsonl").read_text().splitlines()]
    report = {"callsitePositions": json.loads((REAL / "real_ts.positions.json").read_text())}
    known = {r["qname"]: r["id"] for r in records if r["type"] == "symbol"}
    return {"records": records, "report": report, "known": known, "scip": scip.parse_scip(REAL / "real_ts.scip"),
            "sites": {r["calleeName"]: r for r in records if r["type"] == "callsite"}}


def real_merge(real, **kw):
    pos = ms.positions_from_bundle(real["records"], real["report"])
    sites = [real["sites"][n] for n in ("put", "café", "Box", "Plain")]
    return ms.merge_scip(sites, pos.sites, real["scip"], real["known"], prefix="web", lang="javascript",
                         astral=pos.astral, **kw), pos


def test_positions_from_a_stored_bundle_match_the_real_tool_columns(real):
    """The `café()` case: an `é` precedes the callee on the line, and both tools count columns in code points."""
    pos = ms.positions_from_bundle(real["records"], real["report"])
    by_name = {n: pos.sites[real["sites"][n]["id"]] for n in ("put", "café", "Box", "Plain")}
    assert by_name == {"put": ("b.ts", 5, 20), "café": ("b.ts", 5, 9), "Box": ("b.ts", 3, 31), "Plain": ("b.ts", 4, 6)}
    assert pos.astral == frozenset()
    for p in by_name.values():
        assert p in real["scip"].refs                                  # every Joern site position is a SCIP reference position


def test_the_real_merge_agrees_upgrades_and_leaves_the_rest(real):
    res, _ = real_merge(real)
    out = {r["calleeName"]: r for r in res.callsites}
    assert (out["put"]["status"], out["put"]["targets"][0]["qname"]) == ("confirmed", "web/a.ts::Box.put")
    assert (out["café"]["status"], out["café"]["targets"][0]["qname"]) == ("confirmed", "web/a.ts::café")
    assert (out["Box"]["status"], out["Box"]["reason"], out["Box"]["targets"][0]["qname"]) == (
        "confirmed", "resolved-call", "web/a.ts::Box.<init>")
    assert out["Plain"] == real["sites"]["Plain"]
    assert {k: res.counts[k] for k in ("agree", "upgrade", "silent", "ctor_unmapped", "astral_skipped")} == {
        "agree": 2, "upgrade": 1, "silent": 1, "ctor_unmapped": 1, "astral_skipped": 0}


def test_constructor_mapping_real_bundle(real):
    """Every class of the stored real javascript bundle with an explicit constructor resolves, through the real
    scip-typescript symbol ``Box#`<constructor>`().``, to the bundle's constructor symbol; a class without one is counted
    ``ctor_unmapped`` and never synthesised. (The bundle names it ``<class qname>.<init>``, not ``.constructor``.)"""
    known, recs = real["known"], real["records"]
    classes = {r["qname"] for r in recs if r["type"] == "symbol" and r["kind"] == "type"}
    explicit = {q for q in classes if f"{q}.<init>" in known}
    assert explicit == {"web/a.ts::Box"} and classes - explicit == {"web/a.ts::Plain"}
    ctor_refs = [t for ts in real["scip"].refs.values() for t in ts if t.chain and t.chain[-1] == "<constructor>" and t.callable]
    assert ctor_refs
    for t in ctor_refs:
        assert ms._resolve(t, known, "web", "javascript") == ("web/a.ts::Box.<init>", known["web/a.ts::Box.<init>"])
    counts = ms.Counter()
    type_refs = [t for ts in real["scip"].refs.values() for t in ts if t.is_type]
    got = {t.chain: ms._resolve(t, known, "web", "javascript", counts) for t in type_refs}
    assert got[("Box",)] == ("web/a.ts::Box.<init>", known["web/a.ts::Box.<init>"]) and got[("Plain",)] is None
    assert counts["ctor_unmapped"] == len([t for t in type_refs if t.chain == ("Plain",)]) > 0
    assert not any(q.endswith("::Plain.<init>") or q.endswith("::Plain.constructor") for q in known)


def test_the_javascript_constructor_descriptor_has_a_distinct_name_from_python():
    assert ms.INIT_NAME == {"python": "__init__", "javascript": "<init>"} and ms.SCIP_CTOR == "<constructor>"


def test_a_site_whose_line_has_an_astral_character_before_the_callee_is_left_alone(real):
    cid = real["sites"]["put"]["id"]
    rep = {"callsitePositions": {**real["report"]["callsitePositions"], cid: ["b.ts", 5, 20, True]}}
    pos = ms.positions_from_bundle(real["records"], rep)
    assert pos.astral == {cid}
    res, _ = real_merge(real)
    sites = [real["sites"]["put"]]
    out = ms.merge_scip(sites, pos.sites, real["scip"], real["known"], prefix="web", lang="javascript", astral=pos.astral)
    assert out.callsites == sites and out.counts["astral_skipped"] == 1 and out.counts["agree"] == 0


def test_positions_for_unknown_sites_or_a_different_line_are_dropped(real):
    cid = real["sites"]["put"]["id"]
    rep = {"callsitePositions": {cid: ["b.ts", 99, 0, False], "deadbeef": ["b.ts", 1, 0, False]}}
    assert ms.positions_from_bundle(real["records"], rep).sites == {}


def test_a_non_callable_in_corpus_reference_is_not_scip_only():
    attr = "scip-python python proj 0.1 `pkg.a`/Box#attr."
    a = h.document("pkg/a.py", [h.occurrence([2, 4, 8], attr, h.DEF), h.occurrence([0, 4, 10], h.HELPER, h.DEF)])
    b = h.document("pkg/b.py", [h.occurrence([3, 4, 8], attr), h.occurrence([5, 4, 10], h.HELPER)])
    idx = scip.parse_bytes(h.index([a, b]))
    known = {"pkg/a.py::Box.attr": sym("pkg/a.py::Box.attr"), Q_HELPER: KNOWN[Q_HELPER]}
    res = ms.merge_scip([], {}, idx, known, prefix="", lang=LANG)
    assert [o.qname for o in res.scip_only] == [Q_HELPER]
