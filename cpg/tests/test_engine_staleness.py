"""P1-39a: the staleness block and the stamp of an answer (``docs/plans/connected-code-graph-p1-answers.md`` P1-39 behaviours (1) and (2); ``-answer-spec2.md`` §3.5, §4.3;
``-answer-spec.md`` §4.2, §4.3; FR-7, FR-7a, AC-9, AC-9b).

The block lists exactly the units on a chain (never a unit only the walk's flag gates read), each with the built commit and time of **its own** ``Unit`` node, the state
refresh recorded (``fresh``, ``stale-kept`` with the failed time and the value-free code, ``not-mapped`` with its code and no build fields). The stamp is read from the
``:Snapshot`` node and carries the P1 verdict-log values (offset 0, the empty-log hash) and no clock. Everything read is graph data: malformed values are refused with a
static message that never carries them. Offline: ``tests/engine_walk_helpers``' ``MemoryReader``; ``snapshot`` is added by :class:`R` below.
"""
from __future__ import annotations

import copy
import json
import time

import pytest
from engine_walk_helpers import G, MemoryReader

from ccg import answer
from ccg.engine import bfs, staleness as S
from ccg.engine.reader import GraphDataError
from ccg.staleness import FAIL_REASONS

C1, C2, C3 = "a" * 40, "b" * 40, "c" * 40
T1, T2, T3 = "2026-11-01T10:00:00Z", "2026-11-02T11:30:05Z", "2026-11-03T00:00:00Z"
SNAP = "0123456789abcdef0123456789abcdef"
EMPTY = "c482b1cf54631b37eb4572dc86a68fb1"
UA, UB, UC, UD = (c * 32 for c in "abcd")


class R(MemoryReader):
    """``MemoryReader`` plus the ``snapshot`` read (no parameters), answering ``snapshot_rows``."""

    def __init__(self, g, snapshot_rows=None):
        super().__init__(g)
        self.snapshot_rows = [(SNAP, 0, EMPTY, "1.1", "1.0")] if snapshot_rows is None else snapshot_rows

    def read(self, name, params=None):
        if name == "snapshot":
            self.calls.append((name, dict(params or {})))
            return list(self.snapshot_rows)
        return super().read(name, params)


def unit(g, uid, *, repo="repo-a", root="svc", lang="python", status="fresh", commit=C1, built=T1, failed=None, reason=None):
    g.rows.nodes["Unit"] = [n for n in g.rows.nodes["Unit"] if n["unitId"] != uid]
    props = dict(unitId=uid, repoId=repo, rootPath=root, language=lang, buildStatus=status, flowAvailable=True)
    for k, v in (("builtCommit", commit), ("builtAt", built), ("failedAt", failed), ("failReason", reason)):
        if v is not None:
            props[k] = v
    g.node("Unit", **props)


def graph():
    g = G(unit=UA)
    unit(g, UA, root="a", commit=C1, built=T1)
    unit(g, UB, root="b", commit=C2, built=T2)
    unit(g, UC, root="c", status="stale-kept", commit=C3, built=T3, failed=T2, reason="extract:failed")
    unit(g, UD, root="d", status="not-mapped", commit=None, built=None, reason="manifest:never-built")
    return g


def entries(ids, g=None):
    return S.unit_entries(R(g or graph()), ids)


# ------------------------------------------------------------------------------------------- (1) one entry per unit, each from its own Unit node

def test_a_fresh_unit_carries_its_own_built_commit_and_time_and_nothing_else():
    assert entries([UA]) == [{"unitId": UA, "repoId": "repo-a", "rootPath": "a", "lang": "python", "buildStatus": "fresh", "builtCommit": C1, "builtAt": T1}]


def test_each_unit_has_its_own_commit_and_time_never_a_neighbours_or_the_snapshots():
    got = {e["unitId"]: (e["builtCommit"], e["builtAt"]) for e in entries([UA, UB])}
    assert got == {UA: (C1, T1), UB: (C2, T2)}


def test_a_stale_kept_unit_surfaces_the_last_good_build_the_failed_time_and_the_value_free_code():
    assert entries([UC]) == [{"unitId": UC, "repoId": "repo-a", "rootPath": "c", "lang": "python", "buildStatus": "stale-kept", "builtCommit": C3, "builtAt": T3,
                              "failedAt": T2, "reason": "extract:failed"}]


def test_a_never_built_unit_is_not_mapped_with_its_reason_and_carries_no_build_fields():
    (e,) = entries([UD])
    assert e == {"unitId": UD, "repoId": "repo-a", "rootPath": "d", "lang": "python", "buildStatus": "not-mapped", "reason": "manifest:never-built"}


@pytest.mark.parametrize("code", FAIL_REASONS)
def test_every_closed_failure_code_is_carried_as_the_reason_of_a_not_mapped_and_a_stale_kept_unit(code):
    g = graph()
    unit(g, UD, status="not-mapped", commit=None, built=None, reason=code)
    unit(g, UC, status="stale-kept", failed=T2, reason=code)
    assert {e["unitId"]: e["reason"] for e in entries([UC, UD], g)} == {UC: code, UD: code}


def test_the_entries_are_sorted_by_repo_root_and_unit_and_ask_only_for_the_given_units_in_one_read():
    g = graph()
    unit(g, UB, repo="repo-a", root="a")                      # same repo and root as UA: the unit id decides
    unit(g, UC, repo="repo-0", root="z", status="fresh", commit=C1, built=T1)
    r = R(g)
    got = S.unit_entries(r, [UB, UC, UA])
    assert [e["unitId"] for e in got] == [UC, UA, UB]
    assert r.calls == [("units", {"ids": sorted([UA, UB, UC])})]


def test_no_units_means_no_read_and_an_empty_list():
    r = R(graph())
    assert S.unit_entries(r, []) == [] and r.calls == []


def test_a_unit_named_twice_is_listed_once():
    assert [e["unitId"] for e in entries([UA, UA, UA])] == [UA]


def test_a_unit_without_a_unit_row_is_refused():
    with pytest.raises(GraphDataError, match="no Unit row"):
        entries(["f" * 32])


def test_one_missing_unit_among_present_ones_is_refused_not_dropped():
    with pytest.raises(GraphDataError, match="no Unit row"):
        entries([UA, "f" * 32, UB])


# ------------------------------------------------------------------------------------------- hostile unit rows: refused, never repaired, never echoed

PLANT = "PLANTED-VALUE-‮\n<script>"


def _hostile():
    cases = {}
    def bad(name, uid=UA, **kw):
        cases[name] = (uid, kw)
    for v in (None, 5, "", "A" * 40, "a" * 39, "a" * 41, "a" * 64, "g" * 40, PLANT, " " + "a" * 40, "a" * 40 + "\n"):
        bad(f"commit={v!r}", commit=v)
    for v in (None, 5, "", "2026-11-01", "2026-11-01T10:00:00", "2026-11-01T10:00:00+00:00", "2026-13-01T10:00:00Z", "2026-11-31T10:00:00Z", "2026-11-01T25:00:00Z",
              "2026-11-01 10:00:00Z", "2026-11-01T10:00:00Z\n", "٢026-11-01T10:00:00Z", PLANT):
        bad(f"built={v!r}", built=v)
    for v in (5, "", "x:y", PLANT, None):
        bad(f"fresh reason={v!r}" if v is not None else "fresh failedAt", reason=v, failed=T2 if v is None else None)
    for v in (None, 5, "", "Repo", "a b", PLANT):
        bad(f"repo={v!r}", repo=v)
    for v in (None, 5, "Py", "1py", "", PLANT):
        bad(f"lang={v!r}", lang=v)
    bad("root=None", root=None)
    bad("root=5", root=5)
    return cases


@pytest.mark.parametrize("name", sorted(_hostile()))
def test_a_malformed_fresh_unit_row_is_refused_with_a_static_message(name):
    uid, kw = _hostile()[name]
    g = graph()
    unit(g, uid, **{**dict(root="a"), **kw})
    with pytest.raises(GraphDataError) as e:
        entries([uid], g)
    assert "PLANTED" not in str(e.value) and str(e.value) == str(e.value).strip() and str(e.value).isascii()


@pytest.mark.parametrize("kw", [
    dict(failed=None), dict(failed="nope"), dict(failed=5), dict(reason=None), dict(reason="free text"), dict(reason=PLANT), dict(reason=5), dict(commit=None), dict(built=None),
    dict(failed=T2 + "\n")])
def test_a_malformed_stale_kept_row_is_refused(kw):
    g = graph()
    row = dict(status="stale-kept", commit=C3, built=T3, failed=T2, reason="extract:failed") | kw
    unit(g, UC, **row)
    with pytest.raises(GraphDataError) as e:
        entries([UC], g)
    assert "PLANTED" not in str(e.value) and "free text" not in str(e.value)


@pytest.mark.parametrize("kw", [dict(reason=None), dict(reason="free text"), dict(reason=PLANT), dict(commit=C1), dict(built=T1), dict(failed=T2), dict(reason=7)])
def test_a_malformed_not_mapped_row_is_refused(kw):
    g = graph()
    row = dict(status="not-mapped", commit=None, built=None, reason="manifest:never-built") | kw
    unit(g, UD, **row)
    with pytest.raises(GraphDataError) as e:
        entries([UD], g)
    assert "PLANTED" not in str(e.value) and "free text" not in str(e.value)


@pytest.mark.parametrize("status", [None, "", "FRESH", "stale", "built", 3, PLANT])
def test_an_unknown_build_status_is_refused(status):
    g = graph()
    unit(g, UA, status=status)
    with pytest.raises(GraphDataError) as e:
        entries([UA], g)
    assert "PLANTED" not in str(e.value)


def test_a_fresh_unit_that_carries_a_failure_marker_is_refused():
    g = graph()
    unit(g, UA, failed=T2)
    with pytest.raises(GraphDataError):
        entries([UA], g)
    unit(g, UA, reason="extract:failed")
    with pytest.raises(GraphDataError):
        entries([UA], g)


@pytest.mark.parametrize("rows", [[(UA,)], [(UA,) + (None,) * 8], [("x",) + (None,) * 9], [(5,) + (None,) * 9], [(UA, "repo-a", "a", "python", "fresh", C1, T1, None, None, True, None)]])
def test_a_malformed_units_reply_is_refused(rows):
    class Bad(R):
        def read(self, name, params=None):
            return rows if name == "units" else super().read(name, params)
    with pytest.raises(GraphDataError):
        S.unit_entries(Bad(graph()), [UA])


def test_a_units_reply_about_a_unit_that_was_not_asked_for_is_refused_even_when_the_count_matches():
    class Other(R):
        def read(self, name, params=None):
            return super().read(name, {"ids": [UB]}) if name == "units" else super().read(name, params)
    with pytest.raises(GraphDataError):
        S.unit_entries(Other(graph()), [UA])


def test_the_order_does_not_depend_on_the_order_the_reader_returns_the_rows_in():
    class Reversed(R):
        def read(self, name, params=None):
            return list(reversed(super().read(name, params)))
    g = graph()
    unit(g, UA, root="z"), unit(g, UB, root="a"), unit(g, UC, root="a", status="fresh", commit=C1, built=T1)
    assert [e["unitId"] for e in S.unit_entries(Reversed(g), [UA, UB, UC])] == [UB, UC, UA]


def test_a_units_reply_that_repeats_a_unit_is_refused():
    class Dup(R):
        def read(self, name, params=None):
            rows = super().read(name, params)
            return rows + rows if name == "units" else rows
    with pytest.raises(GraphDataError):
        S.unit_entries(Dup(graph()), [UA])


# ------------------------------------------------------------------------------------------- the units on a chain

def hop(label, ident, edge=None):
    return {"node": (label, ident), "state": ("value", ""), "edge": edge}


def item(*nodes, kind="symbol"):
    chain = [hop(*n) for n in nodes]
    return bfs.Item(kind, nodes[-1][1], 1, 0, len(chain) - 1, chain)


def test_the_units_of_a_chain_are_those_of_its_nodes_that_carry_one():
    facts = {("Symbol", "s1"): {"unit": UA}, ("CallSite", "c1"): {"unit": UB}, ("Symbol", "s2"): {"unit": UC}, ("Link", "l"): {"status": "confirmed"}}
    it = item(("Symbol", "s1"), ("CallSite", "c1"), ("Link", "l"), ("Symbol", "s2"))
    assert S.chain_units([it], facts) == {UA, UB, UC}


def test_a_unit_that_is_on_no_chain_is_not_selected_though_facts_and_the_walk_know_it():
    facts = {("Symbol", "s1"): {"unit": UA}, ("Symbol", "other"): {"unit": UB}, ("CallSite", "off"): {"unit": UC}}
    assert S.chain_units([item(("Symbol", "s1"))], facts) == {UA}
    assert S.chain_units([], facts) == set()


def test_every_item_of_the_listing_contributes_its_chain():
    facts = {("Symbol", "s1"): {"unit": UA}, ("Symbol", "s2"): {"unit": UB}}
    assert S.chain_units([item(("Symbol", "s1")), item(("Symbol", "s2"))], facts) == {UA, UB}


@pytest.mark.parametrize("bad", [5, "", None, ["x"]])
def test_a_chain_node_whose_unit_fact_is_malformed_is_refused(bad):
    facts = {("Symbol", "s1"): {"unit": bad}}
    if bad is None:
        assert S.chain_units([item(("Symbol", "s1"))], facts) == set()
        return
    with pytest.raises(GraphDataError):
        S.chain_units([item(("Symbol", "s1"))], facts)


def test_a_node_with_no_facts_contributes_no_unit():
    assert S.chain_units([item(("Symbol", "ghost"))], {}) == set()


def depends_graph():
    g = G(unit=UA)
    unit(g, UA, root="a", commit=C1, built=T1)
    unit(g, UB, root="b", commit=C2, built=T2)                  # a second built unit the walk's gates read but no chain reaches
    unit(g, UD, status="not-mapped", commit=None, built=None, reason="manifest:never-built", root="d")
    g.sym("F"), g.sym("H", unit=UA)
    g.sym("Far", unit=UB)
    g.site("c1", "F"), g.link("L", provider="H"), g.via("c1", "L")
    g.site("c2", "F"), g.link("LU", to_unit=UD, status="possible", reason="target-not-mapped"), g.via("c2", "LU", status="possible", reason="target-not-mapped")
    return g


def walk_of(g):
    r = R(g)
    w = bfs.walk(r, ("Symbol", "F"), column=bfs.DEPENDS_ON)
    return r, w


def test_a_walk_over_a_link_into_a_never_built_unit_lists_it_as_not_mapped_and_only_the_units_on_chains():
    r, w = walk_of(depends_graph())
    block = S.staleness_block(r, w.items(), w.facts)
    assert [(u["unitId"], u["buildStatus"]) for u in block["units"]] == [(UA, "fresh"), (UD, "not-mapped")]
    assert UB not in {u["unitId"] for u in block["units"]}
    assert block["units"][1]["reason"] == "manifest:never-built" and "builtCommit" not in block["units"][1]


def test_extra_units_from_lifted_and_bound_entries_are_listed_too():
    r, w = walk_of(depends_graph())
    block = S.staleness_block(r, w.items(), w.facts, extra_units=[UB])
    assert {u["unitId"] for u in block["units"]} == {UA, UB, UD}


def test_a_unit_hop_whose_unit_is_not_not_mapped_is_refused_because_v9_would_refuse_it():
    g = depends_graph()
    unit(g, UD, status="fresh", commit=C1, built=T1, root="d")
    r, w = walk_of(g)
    with pytest.raises(GraphDataError, match="not-mapped"):
        S.staleness_block(r, w.items(), w.facts)


def test_a_stale_kept_unit_behind_a_unit_hop_is_refused_too():
    g = depends_graph()
    unit(g, UD, status="stale-kept", commit=C1, built=T1, failed=T2, reason="extract:failed", root="d")
    r, w = walk_of(g)
    with pytest.raises(GraphDataError, match="not-mapped"):
        S.staleness_block(r, w.items(), w.facts)


def test_the_block_never_lists_a_unit_only_the_facts_know():
    r, w = walk_of(depends_graph())
    facts = {**w.facts, ("Symbol", "Far"): {"unit": UB}, ("CallSite", "off"): {"unit": UC}}
    assert [u["unitId"] for u in S.staleness_block(r, w.items(), facts)["units"]] == [UA, UD]


def test_a_not_mapped_unit_that_is_not_a_unit_hop_is_a_plain_entry():
    g = depends_graph()
    r, w = walk_of(g)
    facts = dict(w.facts) | {("Symbol", "H"): {"unit": UD}}
    block = S.staleness_block(r, w.items(), facts)
    assert {u["unitId"]: u["buildStatus"] for u in block["units"]} == {UA: "fresh", UD: "not-mapped"}


def test_an_empty_listing_has_an_empty_block_and_reads_nothing():
    r = R(graph())
    assert S.staleness_block(r, [], {}) == {"units": []} and r.calls == []


# ------------------------------------------------------------------------------------------- (2) the stamp

def test_the_stamp_is_the_snapshot_id_the_offset_zero_and_the_empty_log_hash():
    assert S.stamp(R(graph())) == {"snapshotId": SNAP, "verdictLogOffset": 0, "verdictLogHash": EMPTY}


def test_the_stamp_constants_are_the_pinned_literals_and_the_answers_own():
    assert S.stamp(R(graph()))["verdictLogHash"] == "c482b1cf54631b37eb4572dc86a68fb1" == answer.EMPTY_VERDICT_LOG_HASH
    assert S.VERDICT_LOG_OFFSET == 0


def test_the_snapshot_id_comes_from_the_node_not_from_a_constant():
    other = "f" * 32
    assert S.stamp(R(graph(), [(other, 0, EMPTY, "1.1", "1.0")]))["snapshotId"] == other


def test_the_stamp_is_the_same_bytes_every_time_and_carries_no_clock(monkeypatch):
    first = json.dumps(S.stamp(R(graph())), sort_keys=True).encode()
    monkeypatch.setattr(time, "time", lambda: 1.0)
    monkeypatch.setattr(time, "time_ns", lambda: 1)
    second = json.dumps(S.stamp(R(graph())), sort_keys=True).encode()
    assert first == second
    assert set(S.stamp(R(graph()))) == {"snapshotId", "verdictLogOffset", "verdictLogHash"}


def test_the_stamp_reads_the_snapshot_once_with_no_parameters():
    r = R(graph())
    S.stamp(r)
    assert r.calls == [("snapshot", {})]


@pytest.mark.parametrize("rows", [
    [], [(SNAP, 0, EMPTY, "1.1", "1.0")] * 2,
    [(SNAP, 1, EMPTY, "1.1", "1.0")], [(SNAP, 0, "e" * 32, "1.1", "1.0")], [(SNAP, None, None, "1.1", "1.0")], [(SNAP, False, EMPTY, "1.1", "1.0")],
    [(SNAP, 0.0, EMPTY, "1.1", "1.0")], [(SNAP, "0", EMPTY, "1.1", "1.0")],
    [(None, 0, EMPTY, "1.1", "1.0")], [(5, 0, EMPTY, "1.1", "1.0")], [("0123", 0, EMPTY, "1.1", "1.0")], [(SNAP.upper(), 0, EMPTY, "1.1", "1.0")],
    [(SNAP + "0", 0, EMPTY, "1.1", "1.0")], [(SNAP + "\n", 0, EMPTY, "1.1", "1.0")], [(PLANT, 0, EMPTY, "1.1", "1.0")],
    [(SNAP, 0, EMPTY)], [(SNAP,)]])
def test_a_snapshot_that_is_missing_doubled_malformed_or_not_the_p1_verdict_log_is_refused(rows):
    with pytest.raises(GraphDataError) as e:
        S.stamp(R(graph(), rows))
    assert "PLANTED" not in str(e.value) and "e" * 32 not in str(e.value)


# ------------------------------------------------------------------------------------------- the objects satisfy ccg.answer

def golden():
    return json.loads((answer.HERE.parent / "tests" / "golden" / "answer-ac1.json").read_bytes())


def golden_graph(doc):
    g = G(unit=doc["staleness"]["units"][0]["unitId"])
    g.rows.nodes["Unit"] = []
    for u in doc["staleness"]["units"]:
        unit(g, u["unitId"], repo=u["repoId"], root=u["rootPath"], lang=u["lang"], commit=u["builtCommit"], built=u["builtAt"])
    return g


def test_the_golden_answers_staleness_and_stamp_are_reproduced_from_the_graph_and_validate():
    doc = golden()
    r = R(golden_graph(doc), [(doc["stamp"]["snapshotId"], 0, EMPTY, "1.1", "1.0")])
    ids = [u["unitId"] for u in doc["staleness"]["units"]]
    built = {"units": S.unit_entries(r, ids)}
    assert built == doc["staleness"] and S.stamp(r) == doc["stamp"]
    answer.validate(doc | {"staleness": built, "stamp": S.stamp(r)})


@pytest.mark.parametrize("status", ["stale-kept", "not-mapped"])
def test_a_stale_kept_or_not_mapped_entry_validates_inside_the_goldens_document(status):
    doc = golden()
    g = golden_graph(doc)
    victim = doc["staleness"]["units"][0]["unitId"]
    if status == "stale-kept":
        unit(g, victim, repo="graphmind-ai-lab", root="falkor-chat/server", status=status, commit=C1, built=T1, failed=T2, reason="bundle:invalid")
    else:
        unit(g, victim, repo="graphmind-ai-lab", root="falkor-chat/server", status=status, commit=None, built=None, reason="manifest:never-built")
    r = R(g, [(doc["stamp"]["snapshotId"], 0, EMPTY, "1.1", "1.0")])
    mine = S.unit_entries(r, [u["unitId"] for u in doc["staleness"]["units"]])
    assert [u["buildStatus"] for u in mine if u["unitId"] == victim] == [status]
    answer.validate(copy.deepcopy(doc) | {"staleness": {"units": mine}})
