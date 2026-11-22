"""P1-32b gate harness: EP §12.1's fail procedure (floor E12 = 35), the report units and the cumulative record. Offline, synthetic."""
import json

import pytest

import gate_helpers as H
from gate_helpers import G

STRATA = list(G.STRATA)


def result(rows, **extra):
    """rows: (id, stratum, outcome, counting, anchor)."""
    items = {i: {"stratum": s, "outcome": o, "counting": c, "anchorId": a, **({"reason": "no-qualifying-item", "stage": "none"} if o == "miss" else
                                                                                {"hops": 2, "interFunctionHops": 1} if o == "hit" else {})}
             for i, s, o, c, a in rows}
    return {"items": items, "defects": {}, "harnessErrors": {}, "maskedComparisons": 0, "stops": {}, **extra}


def hits(prefix, n, stratum, counting=True, anchor_every=2):
    return [(f"{prefix}{k:03d}", stratum, "hit", counting, f"{prefix}a{k // anchor_every:03d}") for k in range(n)]


def gate1_rows(per=15):
    return [r for s in STRATA for r in hits(s[:2], per, s)]


# ---------------------------------------------------------------- gate 1
def test_gate1_all_hits_over_the_minimums_passes():
    d = G.decide_gate(result(gate1_rows()))
    assert d["verdict"] == "pass" and d["H"] == 75 and d["minimumsMet"] is True and d["misses"] == 0


def test_gate1_74_counting_items_is_not_a_pass():
    d = G.decide_gate(result(gate1_rows()[:-1]))
    assert d["verdict"] == "minimums-not-met" and d["H"] == 74


def test_gate1_a_stratum_at_exactly_8_items_passes():
    rows = [r for s in STRATA[:-1] for r in hits(s[:2], 17, s)] + hits("fl", 8, "flow")
    d = G.decide_gate(result(rows))
    assert d["verdict"] == "pass" and d["byStratum"]["flow"] == 8 and d["H"] == 76


def test_gate1_a_stratum_below_8_is_not_a_pass():
    rows = [r for s in STRATA[:-1] for r in hits(s[:2], 20, s)] + hits("fl", 7, "flow")
    d = G.decide_gate(result(rows))
    assert d["verdict"] == "minimums-not-met" and d["H"] == 87 and d["byStratum"]["flow"] == 7


def test_gate1_one_miss_fails_and_reports_the_cluster_and_stratum_to_move():
    rows = gate1_rows()
    rows[3] = (rows[3][0], rows[3][1], "miss", True, "AX")
    d = G.decide_gate(result(rows))
    assert d["verdict"] == "fail" and d["misses"] == 1 and d["missClusters"] == ["AX"] and d["movedOutByStratum"] == {"http": 1}


def test_gate1_non_counting_items_do_not_count_toward_75():
    rows = gate1_rows(per=14) + hits("nc", 10, "http", counting=False)
    d = G.decide_gate(result(rows))
    assert d["verdict"] == "minimums-not-met" and d["H"] == 70 and d["scored"] == 80


def test_out_of_scope_items_are_neither_hits_nor_misses_and_never_fail_the_gate():
    rows = gate1_rows()
    res = result(rows)
    res["items"]["oos"] = {"stratum": "http", "outcome": "out-of-scope", "triage": "key-error", "counting": True, "anchorId": "z"}
    d = G.decide_gate(res)
    assert d["verdict"] == "pass" and d["misses"] == 0 and d["scored"] == 75


def test_a_c5_stamp_stop_is_reported_as_stopped_whatever_else():
    d = G.decide_gate(result(gate1_rows(), stops={"C5:stamp:snapshotId": 1}))
    assert d["verdict"] == "stopped"


def test_a_void_item_stops_the_gate():
    res = result(gate1_rows())
    res["items"]["v"] = {"stratum": "http", "outcome": "void", "counting": True, "anchorId": "z"}
    assert G.decide_gate(res)["verdict"] == "stopped"


@pytest.mark.parametrize("extra,blocker", [({"defects": {"request-echo": 1}}, "engine-defects"), ({"harnessErrors": {"class-X": 1}}, "harness-errors"),
                                           ({"maskedComparisons": 1}, "masked-comparisons")])
def test_listed_defects_errors_and_masked_comparisons_block_a_pass(extra, blocker):
    d = G.decide_gate(result(gate1_rows(), **extra))
    assert d["verdict"] == "blocked" and d["blockers"] == [blocker]


def test_a_miss_is_a_fail_even_when_blockers_are_listed():
    rows = gate1_rows()
    rows[0] = (rows[0][0], rows[0][1], "miss", True, "AX")
    assert G.decide_gate(result(rows, defects={"x": 1}))["verdict"] == "fail"


# ---------------------------------------------------------------- re-gate: the floor
KEPT_STRATA = ("http", "mcp", "scripts", "flow")


def kept_rows(per=10):
    return [r for s in KEPT_STRATA for r in hits("k" + s[:2], per, s)]


def regate(fresh_counting, fresh_noncounting=0, kept=11, moved_out=None, miss=False):
    rows = kept_rows(kept)
    flags = {r[0]: "kept" for r in rows}
    fresh = hits("fr", fresh_counting, "wiring") + hits("fn", fresh_noncounting, "wiring", counting=False)
    rows += fresh
    flags.update({r[0]: "fresh-cluster" for r in fresh})
    if miss:
        rows[0] = (rows[0][0], rows[0][1], "miss", True, "AX")
    return G.decide_gate(result(rows), flags, moved_out), flags


def test_regate_all_hits_with_34_fresh_counting_is_all_hits_floor_not_met():
    d, _ = regate(34)
    assert d["verdict"] == "all hits, floor not met" and d["freshCounting"] == 34 and d["floorMet"] is False and d["compositionMet"] is True


def test_regate_35_fresh_counting_items_passes_at_the_floor():
    d, _ = regate(35)
    assert d["verdict"] == "pass" and d["floorMet"] is True and d["freshCounting"] == 35


def test_regate_non_counting_fresh_items_do_not_reach_the_floor():
    d, _ = regate(34, fresh_noncounting=10)
    assert d["verdict"] == "all hits, floor not met" and d["freshCounting"] == 34


def test_regate_kept_cluster_fresh_items_do_not_count_as_fresh():
    rows = kept_rows() + hits("fr", 34, "wiring") + hits("kc", 5, "wiring")
    flags = {r[0]: "fresh-cluster" if r[0].startswith("fr") else "kept-cluster-fresh" if r[0].startswith("kc") else "kept" for r in rows}
    d = G.decide_gate(result(rows), flags)
    assert d["freshCounting"] == 34 and d["verdict"] == "all hits, floor not met"


def test_regate_a_miss_is_a_fail_even_with_35_fresh():
    d, _ = regate(40, miss=True)
    assert d["verdict"] == "fail"


def test_regate_composition_needs_75_counting_items():
    d, _ = regate(35, kept=8)                                       # 32 kept + 35 fresh = 67 < 75: the floor is met, the composition is not
    assert d["verdict"] == "all hits, composition not met" and d["floorMet"] is True and d["H"] == 67


def test_regate_composition_needs_8_items_in_every_stratum():
    rows = hits("fr", 76, "wiring") + hits("ke", 7, "http")
    flags = {r[0]: "fresh-cluster" if r[0].startswith("fr") else "kept" for r in rows}
    d = G.decide_gate(result(rows), flags)
    assert d["verdict"] == "all hits, composition not met" and d["floorMet"] is True


def test_regate_composition_needs_fresh_items_for_each_stratum_that_had_a_miss():
    d, _ = regate(35, moved_out={"wiring": 35, "http": 2})
    assert d["verdict"] == "all hits, composition not met" and d["compositionShortfall"] == {"http": 2}


def test_regate_blockers_block_a_would_be_pass():
    rows = kept_rows() + hits("fr", 35, "wiring")
    flags = {r[0]: "fresh-cluster" if r[0].startswith("fr") else "kept" for r in rows}
    d = G.decide_gate(result(rows, defects={"request-echo": 1}), flags)
    assert d["verdict"] == "blocked" and d["floorMet"] is True


def test_first_run_marker_refuses_a_rerun_on_the_same_list(tmp_path):
    G.first_run_marker(tmp_path, "ab" * 32, {"x": 1})
    with pytest.raises(G.Refusal, match="already run"):
        G.first_run_marker(tmp_path, "ab" * 32, {"x": 1})
    G.first_run_marker(tmp_path, "cd" * 32, {"x": 1})            # another list is fine


# ---------------------------------------------------------------- intervals
def test_zero_miss_wilson_lower_bounds_match_the_spec():
    assert G._wilson(159, 159)["lower"] == 0.976 and G._wilson(62, 62)["lower"] == 0.942 and G._wilson(58, 58)["lower"] == 0.938
    assert G._wilson(35, 35)["lower"] == 0.901


def test_cumulative_record_108_of_110_matches_the_spec_interval():
    w = G._wilson(108, 110)
    assert (w["lower"], w["upper"]) == (0.936, 0.995)


def test_ratio_interval_by_hand():
    r = G.ratio_interval([(2, 2), (2, 1), (2, 2)])
    assert r == {"n": 3, "R": 0.833, "lower": 0.116, "upper": 1.0}
    assert G.ratio_interval([(2, 2)])["R"] is None and G.ratio_interval([(0, 0), (2, 2)])["n"] == 1


def test_t975_table_and_tail():
    assert G.t975(29) == 2.045 and G.t975(2) == 4.303
    assert abs(G.t975(60) - 2.000) < 0.002 and abs(G.t975(1000) - 1.962) < 0.001


# ---------------------------------------------------------------- the report: units, counts only
def test_report_units_item_anchor_cluster_and_counts_only():
    rows = [("i1", "http", "hit", True, "A1"), ("i2", "http", "miss", True, "A1"), ("i3", "http", "hit", True, "A2"),
            ("i4", "mcp", "hit", True, "A3"), ("i5", "mcp", "hit", True, "A4")]
    res = result(rows)
    rep = G.build_report(res, G.decide_gate(res), cluster_of={"A1": "C1", "A2": "C1", "A3": "C2", "A4": "C3"})
    http = rep["recall"]["byStratum"]["http"]
    assert (http["items"]["k"], http["items"]["n"]) == (2, 3)
    assert (http["anchors"]["k"], http["anchors"]["n"]) == (1, 2)           # A1 has a miss
    assert (http["clusters"]["k"], http["clusters"]["n"]) == (0, 1)         # C1 holds A1's miss
    assert rep["recall"]["pooled"]["clusterUnit"] == "cluster"
    assert rep["verdict"] == "fail" and rep["misses"]["count"] == 1 and rep["misses"]["triage"] == {c: 0 for c in G.TRIAGE}
    text = json.dumps(rep)
    for secret in ("i1", "i2", "A1", "A2", "A3"):
        assert f'"{secret}"' not in text


def test_report_anchor_as_cluster_when_no_map_is_given():
    res = result(hits("a", 4, "http"))
    rep = G.build_report(res, G.decide_gate(res))
    assert rep["recall"]["pooled"]["clusterUnit"] == "anchor" and rep["recall"]["pooled"]["clusters"] == rep["recall"]["pooled"]["anchors"]


def test_report_chain_length_median_and_p90_by_stratum():
    res = result(hits("a", 4, "http"))
    for i, h in zip(sorted(res["items"]), (1, 2, 3, 10)):
        res["items"][i]["interFunctionHops"] = h
    rep = G.build_report(res, G.decide_gate(res))
    assert rep["chainLength"]["http"]["median"] == 2.5 and rep["chainLength"]["http"]["n"] == 4 and rep["chainLength"]["http"]["p90"] == 10


def test_report_miss_flags_are_counted_per_class():
    res = result(hits("a", 2, "http"))
    res["items"]["m1"] = {"stratum": "http", "outcome": "miss", "counting": True, "anchorId": "z", "reason": "anchor-not-in-graph"}
    res["items"]["m2"] = {"stratum": "http", "outcome": "miss", "counting": True, "anchorId": "y", "reason": "no-qualifying-item", "stage": "none",
                          "extractionAbsent": True, "depthBoundCut": True}
    m = G.build_report(res, G.decide_gate(res))["misses"]
    assert m["anchorNotInGraph"] == 1 and m["extractionAbsent"] == 1 and m["depthBoundCut"] == 1 and m["byReason"] == {
        "anchor-not-in-graph": 1, "no-qualifying-item": 1}


def test_regate_report_fresh_recall_kept_check_and_cumulative_record():
    rows = kept_rows() + hits("fr", 35, "wiring")
    flags = {r[0]: "fresh-cluster" if r[0].startswith("fr") else "kept" for r in rows}
    res = result(rows)
    d = G.decide_gate(res, flags)
    exposure = [{"itemId": f"old{k}", "firstGate": 1, "hit": k >= 2} for k in range(75)]      # gate 1 at 73/75 on items no longer listed
    rep = G.build_report(res, d, flags=flags, exposure=exposure)
    r = rep["regate"]
    assert rep["verdict"] == "pass" and (r["freshCluster"]["k"], r["freshCluster"]["n"]) == (35, 35) and r["freshCluster"]["wilson"]["lower"] == 0.901
    assert r["keptRegressionCheck"]["n"] == 40 and "not recall evidence" in r["keptRegressionCheck"]["note"]
    cum = r["cumulativeFirstExposure"]["pooled"]
    assert (cum["k"], cum["n"], cum["lower"], cum["upper"]) == (73 + 40 + 35, 75 + 40 + 35, cum["lower"], cum["upper"])
    assert "0.951" in r["note"] and "35/35" in r["claim"] and "0.951" not in r["claim"]


def test_regate_cumulative_record_matches_the_specs_108_of_110():
    rows = hits("fr", 35, "wiring")
    flags = {r[0]: "fresh-cluster" for r in rows}
    res = result(rows)
    prior = [{"itemId": f"old{k}", "firstGate": 1, "hit": k >= 2} for k in range(75)]
    cum = G.build_report(res, G.decide_gate(res, flags), flags=flags, exposure=prior)["regate"]["cumulativeFirstExposure"]["pooled"]
    assert (cum["k"], cum["n"], cum["lower"], cum["upper"]) == (108, 110, 0.936, 0.995)


def test_regate_cumulative_counts_each_item_once_and_by_stratum_when_known():
    rows = hits("fr", 35, "wiring")
    flags = {r[0]: "fresh-cluster" for r in rows}
    res = result(rows)
    prior = [{"itemId": "fr000", "firstGate": 1, "hit": False}, {"itemId": "old1", "firstGate": 1, "hit": True}]
    cum = G.build_report(res, G.decide_gate(res, flags), flags=flags, exposure=prior, exposure_strata={"fr000": "wiring", "old1": "http"}
                         )["regate"]["cumulativeFirstExposure"]
    assert (cum["pooled"]["k"], cum["pooled"]["n"]) == (1 + 34, 2 + 34)         # fr000 is counted once, at its first gate (a miss there)
    assert (cum["byStratum"]["wiring"]["k"], cum["byStratum"]["wiring"]["n"]) == (34, 35) and cum["byStratum"]["http"]["n"] == 1


def test_report_excludes_out_of_scope_items_from_every_figure():
    res = result(hits("a", 4, "http"))
    res["items"]["oos"] = {"stratum": "http", "outcome": "out-of-scope", "triage": "key-error", "counting": True, "anchorId": "zz"}
    rep = G.build_report(res, G.decide_gate(res))
    assert rep["H_all"] == 4 and rep["recall"]["pooled"]["items"]["n"] == 4 and rep["recall"]["byStratum"]["http"]["items"]["n"] == 4
    assert rep["outOfScope"] == {"count": 1, "triage": {"key-error": 1}}


def test_regate_report_claims_nothing_when_it_is_not_a_pass():
    rows = kept_rows(11) + hits("fr", 34, "wiring")
    flags = {r[0]: "fresh-cluster" if r[0].startswith("fr") else "kept" for r in rows}
    res = result(rows)
    rep = G.build_report(res, G.decide_gate(res, flags), flags=flags)
    assert rep["verdict"] == "all hits, floor not met" and rep["regate"]["claim"] == "no AC-5 pass claimed" and rep["regate"]["floorMet"] is False
