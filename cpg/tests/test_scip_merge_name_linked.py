"""P1-05 5d — `merge_scip(..., name_linked=...)`: the Joern targets 5d dropped from a call site are part of ``J``.

Plan: `docs/plans/connected-code-graph-p1-extraction.md` § 5d (`merge_scip`). A site 5d demoted carries its dropped Joern targets in
``report["nameLinked"][callSiteId]``; ``J`` for such a site is the surviving bundle ``targets`` plus the entry (the method
note's "fixed, never tuned" rule is unchanged: only what ``J`` is read from changes). The three pinned rows over a demoted site:
``S in J`` (the site becomes `confirmed`/`resolved-call` to S), ``S not in J`` (the site stays as 5d left it: demoted, or its
type-backed targets; the disagreement is still counted and its candidates reported), no entry (as before).
"""
from __future__ import annotations

import copy

import pytest

import scip_helpers as h
from test_scip_merge import (KNOWN, Q_HELPER, Q_OTHER, Q_PUT, ENCLOSING, index_of, merge as base_merge, only, site, tgt, tgt_ids)


def merge(sites, idx, name_linked, positions=None):
    from extractors.joern.normalize import merge_scip as ms
    positions = positions if positions is not None else {s["id"]: ("pkg/b.py", 4, 4) for s in sites}
    return ms.merge_scip(sites, positions, idx, KNOWN, prefix="", lang="python", name_linked=name_linked)


def demoted(name="helper"):
    return site(name, 0)                                      # what 5d leaves of a fully demoted site: unresolved, no targets


def cand(q):
    return {**tgt(q), "status": "possible", "reason": "resolution-unverified"}


# --- S in J ---------------------------------------------------------------------------------------------------------
def test_scip_agreeing_with_a_dropped_target_confirms_the_demoted_site():
    s = demoted()
    res = merge([s], index_of((3, 4, h.HELPER)), {s["id"]: [tgt(Q_HELPER)]})
    r = only(res)
    assert (r["status"], r["reason"], r["resolution"], r["targets"]) == ("confirmed", "resolved-call", "resolved", [tgt(Q_HELPER)])
    assert res.counts["agree"] == 1 and res.counts["upgrade"] == 0 and res.candidates == {}


def test_the_other_dropped_targets_of_the_agreeing_site_are_candidates():
    s = demoted()
    res = merge([s], index_of((3, 4, h.HELPER)), {s["id"]: [tgt(Q_HELPER), tgt(Q_OTHER)]})
    assert tgt_ids(only(res)) == [KNOWN[Q_HELPER]]
    assert res.candidates[s["id"]] == [cand(Q_OTHER)]


def test_a_partially_demoted_site_agreeing_with_its_kept_target_lists_the_dropped_one_as_a_candidate():
    s = site("put", 0, [Q_PUT])                              # type-backed target kept; Q_HELPER dropped into the entry
    res = merge([s], index_of((3, 4, h.BOX_PUT)), {s["id"]: [tgt(Q_HELPER)]})
    assert tgt_ids(only(res)) == [KNOWN[Q_PUT]]
    assert res.candidates[s["id"]] == [cand(Q_HELPER)]


def test_a_partially_demoted_site_agreeing_with_a_dropped_target_takes_it_and_keeps_the_kept_one_as_candidate():
    s = site("helper", 0, [Q_PUT])
    res = merge([s], index_of((3, 4, h.HELPER)), {s["id"]: [tgt(Q_HELPER)]})
    r = only(res)
    assert (r["status"], r["targets"]) == ("confirmed", [tgt(Q_HELPER)])
    assert res.candidates[s["id"]] == [cand(Q_PUT)]


# --- S not in J -----------------------------------------------------------------------------------------------------
def test_scip_disagreeing_with_every_dropped_target_leaves_the_demoted_site_as_5d_left_it():
    s = demoted()
    res = merge([s], index_of((3, 4, h.OTHER)), {s["id"]: [tgt(Q_HELPER)]})
    assert only(res) == s
    assert res.counts["disagree"] == 1
    assert {c["qname"] for c in res.candidates[s["id"]]} == {Q_HELPER, Q_OTHER}


def test_scip_disagreeing_with_a_partially_demoted_site_keeps_its_type_backed_targets():
    s = site("put", 0, [Q_PUT])
    res = merge([s], index_of((3, 4, h.OTHER)), {s["id"]: [tgt(Q_HELPER)]})
    r = only(res)
    assert r == s and r["status"] == "confirmed" and tgt_ids(r) == [KNOWN[Q_PUT]]
    assert res.counts["disagree"] == 1
    assert {c["qname"] for c in res.candidates[s["id"]]} == {Q_PUT, Q_HELPER, Q_OTHER}


def test_scip_external_against_a_demoted_site_is_a_counted_disagreement_and_changes_nothing():
    s = demoted("len")
    res = merge([s], index_of((3, 4, h.LEN)), {s["id"]: [tgt(Q_HELPER)]})
    assert only(res) == s and res.counts["disagree"] == 1
    assert [c["qname"] for c in res.candidates[s["id"]]] == [Q_HELPER]


def test_scip_silent_leaves_a_demoted_site_alone():
    s = demoted()
    res = merge([s], index_of(), {s["id"]: [tgt(Q_HELPER)]})
    assert only(res) == s and res.counts["silent"] == 1


# --- no entry: as before --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nl", [None, {}, {"someOtherSiteId": [{"symbolId": "x", "qname": "y"}]}])
def test_a_site_without_an_entry_is_merged_exactly_as_before(nl):
    s = demoted()
    idx = index_of((3, 4, h.HELPER))
    res, base = merge([s], idx, nl), base_merge([s], idx)
    assert res.callsites == base.callsites and res.candidates == base.candidates and res.counts == base.counts
    assert res.counts["upgrade"] == 1                        # an unresolved site with SCIP's target is an upgrade, not an agreement


def test_the_keyword_defaults_to_none_so_every_shipped_caller_is_unchanged():
    import inspect
    from extractors.joern.normalize import merge_scip as ms
    p = inspect.signature(ms.merge_scip).parameters
    assert p["name_linked"].default is None and p["name_linked"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(p)[:4] == ["callsites", "positions", "scip", "known"] and "astral" in p


def test_name_linked_is_not_mutated():
    s = demoted()
    nl = {s["id"]: [tgt(Q_HELPER)]}
    before = copy.deepcopy(nl)
    merge([s], index_of((3, 4, h.HELPER)), nl)
    assert nl == before


def test_no_dropped_target_is_ever_lost_by_the_merge():
    """R_m >= R_j with J read from the entry: every dropped target is a target or a candidate of its site, whatever SCIP says."""
    for s_sym in (None, h.HELPER, h.OTHER, h.BOX_PUT, h.LEN, h.LOCAL):
        s = demoted()
        idx = index_of(*([(3, 4, s_sym)] if s_sym else []))
        res = merge([s], idx, {s["id"]: [tgt(Q_HELPER)]})
        r = only(res)
        seen = set(tgt_ids(r)) | {c["symbolId"] for c in res.candidates.get(s["id"], [])}
        silent = s_sym is None
        assert silent or KNOWN[Q_HELPER] in seen
