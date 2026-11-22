"""P1-36 noise tooling (``docs/plans/connected-code-graph-p1-gates.md`` P1-36; EP §11; ``-p1-ml.md`` §5.1, §6.3-6.4).

Offline and synthetic: answers, exclusions and the snapshot are injected; no store, no model."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "eval" / "tools"
sys.path.insert(0, str(TOOLS))
import noise as N  # noqa: E402

EXCL = ["cpg/", "docs/**/connected-code-graph*", "docs/BACKLOG.md"]


def item(sid, path="src/a.py", status="possible", reason="name-only-provider", distance=1, unit="u1"):
    it = {"subject": {"type": "symbol", "symbolId": sid, "path": path, "lineStart": 1}, "repoId": "r", "unitId": unit,
          "status": status, "directness": "direct" if distance == 1 else "indirect", "distance": distance,
          "viaData": False, "chain": []}
    if status == "possible":
        it["reason"] = reason
    return it


def answer(anchor, items=(), members=()):
    groups = [{"kind": "other-possible-callees", "key": "k", "reason": "resolution-unverified",
               "count": len(members), "members": list(members)}] if members else []
    return {"anchorId": anchor, "items": list(items), "groups": groups}


# ---------------------------------------------------------------- bands
@pytest.mark.parametrize("d,b", [(1, 1), (2, 2), (3, 3), (4, 3), (9, 3)])
def test_band_edges(d, b):
    assert N.band(d) == b


# ---------------------------------------------------------------- scope
@pytest.mark.parametrize("path,out", [
    ("cpg/x.py", True), ("cpgx/y.py", False), ("docs/plans/connected-code-graph-p1.md", True),
    ("docs/plans/other.md", False), ("docs/BACKLOG.md", True), ("docs/BACKLOG.mdx", False),
    ("docs/a/b/connected-code-graph.md", True)])
def test_path_excluded(path, out):
    assert N.path_excluded(path, EXCL) is out


def test_unit_subject_is_never_excluded():
    it = {"subject": {"type": "unit", "unitId": "u9"}, "unitId": "u9"}
    assert N.item_path(it) is None
    assert N.build_frame([answer("A", [dict(item("x"), subject=it["subject"])])], EXCL)["frame"]


def test_out_of_scope_items_leave_the_frame_and_are_counted_per_answer_and_reason():
    a = answer("A", [item("s1"), item("s2", path="cpg/z.py"), item("s3", path="cpg/y.py", reason="unknown-key"),
                     item("s4", path="cpg/w.py", status="confirmed")],
               members=[item("m1", path="cpg/m.py")])
    f = N.build_frame([a, answer("B", [item("s5")])], EXCL)
    assert [r["itemId"] for r in f["frame"]] == ["u1:s1", "u1:s5"]
    assert f["outOfScope"] == {"A": {"name-only-provider": 2, "unknown-key": 1, "confirmed": 1}}
    assert f["burden"] == {"inScope": [1, 1], "allListed": [5, 1]}   # per answer, in anchorId order


def test_group_members_are_frame_items():
    a = answer("A", [item("s1")], members=[item("m1"), item("m2", distance=2)])
    f = N.build_frame([a], EXCL)
    assert sorted(r["itemId"] for r in f["frame"]) == ["u1:m1", "u1:m2", "u1:s1"]


def test_strata_are_reason_by_band_and_confirmed_is_a_census():
    a = answer("A", [item("a", distance=1), item("b", distance=3), item("c", distance=4),
                     item("d", reason="unknown-key", distance=2), item("e", status="confirmed", distance=1)])
    f = N.build_frame([a], EXCL)
    assert f["strata"] == {("name-only-provider", 1): 1, ("name-only-provider", 3): 2, ("unknown-key", 2): 1}
    assert [r["itemId"] for r in f["census"]] == ["u1:e"]


# ---------------------------------------------------------------- allocation (hand-computed)
def test_allocation_minima_then_largest_remainder_then_cap_then_largest_unsampled():
    # minima 5,5,3 (=13); remainder 87 over N=303: quotas 57/28/0, remainders 129/216/261; the 2 leftovers go to
    # c and b; c is capped at 3 so its extra goes to the largest unsampled stratum, a.
    assert N.allocate({"a": 200, "b": 100, "c": 3}) == {"a": 63, "b": 34, "c": 3}


def test_allocation_cap_and_remainder_order():
    # minima 5,5; remainder 90 over 1006: a quota 0 rem 540, b quota 89 rem 466; the one leftover goes to a.
    assert N.allocate({"a": 6, "b": 1000}) == {"a": 6, "b": 94}


def test_minima_exceeding_the_target_are_the_sample():
    sizes = {f"s{i}": 10 for i in range(25)}
    got = N.allocate(sizes)
    assert got == {k: 5 for k in sizes} and sum(got.values()) == 125


def test_a_frame_smaller_than_the_target_is_sampled_whole():
    assert N.allocate({"a": 4, "b": 40}) == {"a": 4, "b": 40}


def test_allocation_is_exact_integer_not_float_rounded():
    # 3 equal strata, remainder 85 -> 28.33 each: two get 29 only through integer remainders; ties break by key
    got = N.allocate({"a": 1000, "b": 1000, "c": 1000})
    assert got == {"a": 34, "b": 33, "c": 33} and all(isinstance(v, int) for v in got.values())


# ---------------------------------------------------------------- draw
def _expected(seed, cands, k):
    return sorted(cands, key=lambda c: hashlib.sha256(f"{seed}|{c[0]}|{c[1]}".encode()).hexdigest())[:k]


def test_draw_orders_by_sha256_of_seed_anchor_and_item():
    cands = [("A", f"u1:s{i}") for i in range(30)]
    frame = [{"anchorId": a, "itemId": i, "reason": "r", "band": 1, "status": "possible"} for a, i in cands]
    s = N.draw(frame, {("r", 1): 5}, "g1", "snap1")
    assert [(r["anchorId"], r["itemId"]) for r in s] == _expected("ccg-noise-v1|g1|snap1", cands, 5)


@pytest.mark.parametrize("gate,snap", [("g2", "snap1"), ("g1", "snap2")])
def test_draw_moves_with_gate_id_and_snapshot_id(gate, snap):
    cands = [("A", f"u1:s{i}") for i in range(30)]
    frame = [{"anchorId": a, "itemId": i, "reason": "r", "band": 1, "status": "possible"} for a, i in cands]
    base = N.draw(frame, {("r", 1): 5}, "g1", "snap1")
    assert N.draw(frame, {("r", 1): 5}, gate, snap) != base


def test_draw_stays_within_each_stratum():
    frame = [{"anchorId": "A", "itemId": f"x{i}", "reason": "r", "band": 1 + i % 2, "status": "possible"} for i in range(20)]
    s = N.draw(frame, {("r", 1): 3, ("r", 2): 2}, "g", "s")
    assert sum(r["band"] == 1 for r in s) == 3 and sum(r["band"] == 2 for r in s) == 2


# ---------------------------------------------------------------- CLI
def _write_answers(p, answers):
    p.write_text("".join(json.dumps(a) + "\n" for a in answers))


def test_cli_draw_refuses_a_custodian_root_inside_the_repo(tmp_path, capsys):
    _write_answers(tmp_path / "a.jsonl", [answer("A", [item("s1")])])
    rc = N.main(["draw", "--answers", str(tmp_path / "a.jsonl"), "--exclusions", str(tmp_path / "e.json"),
                 "--gate-id", "g", "--snapshot-id", "s", "--custodian", str(Path(N.REPO) / "cpg" / "noise-out")])
    assert rc == 2 and not (Path(N.REPO) / "cpg" / "noise-out").exists()


def test_cli_draw_writes_frame_and_sample_under_the_custodian_and_prints_counts_only(tmp_path, capsys):
    ans = [answer("ANCHOR-SECRET", [item(f"s{i}", distance=1 + i % 3) for i in range(12)] + [item("c", status="confirmed")])]
    _write_answers(tmp_path / "a.jsonl", ans)
    (tmp_path / "e.json").write_text(json.dumps(EXCL))
    cust = tmp_path / "cust"
    rc = N.main(["draw", "--answers", str(tmp_path / "a.jsonl"), "--exclusions", str(tmp_path / "e.json"),
                 "--gate-id", "g", "--snapshot-id", "s", "--custodian", str(cust)])
    out = capsys.readouterr().out
    assert rc == 0
    rep = json.loads(out)
    assert rep["frame"] == 13 and rep["census"] == 1 and rep["sample"] == 12
    assert "ANCHOR-SECRET" not in out and "u1:s" not in out
    assert len((cust / "noise-frame.jsonl").read_text().splitlines()) == 13
    assert len((cust / "noise-sample.jsonl").read_text().splitlines()) == 12


# ---------------------------------------------------------------- estimates (hand-computed, EP §11.3)
from fractions import Fraction as Fr  # noqa: E402


def frow(i, status="possible", reason="r1", band_=1, split="possible", anchor="A"):
    return {"anchorId": anchor, "itemId": f"i{i}", "status": status, "reason": None if status == "confirmed" else reason,
            "band": band_, "distance": band_, "split": split}


def est_fixture():
    rows, labels, n = [], {}, 0

    def add(count, labelled, reals, **kw):
        nonlocal n
        for j in range(count):
            n += 1
            rows.append(frow(n, **kw))
            if j < labelled:
                labels[("A", f"i{n}")] = "real" if j < reals else "false"
    add(10, 4, 3, reason="r1", band_=1)          # N=10 n=4 t=3 f=1
    add(20, 5, 1, reason="r2", band_=2)          # N=20 n=5 t=1 f=4
    add(3, 3, 3, status="confirmed", band_=1)    # census: 3 real at band 1
    add(3, 3, 2, status="confirmed", band_=2)    # census: 2 real 1 false at band 2
    return rows, labels


def test_pooled_estimates_are_the_ratio_of_stratified_sums():
    rows, labels = est_fixture()
    e = N.estimate(rows, labels)
    assert (e["What"], e["Rhat"], e["RC"]) == (Fr(37, 2), Fr(23, 2), 5)
    assert e["Q"] == Fr(37, 2) / Fr(33, 2) == Fr(37, 33)
    assert e["possiblePrecision"] == Fr(23, 2) / 30
    assert e["confirmedPrecision"] == N.wilson(5, 6)


def test_by_band_restricts_numerator_and_both_denominator_terms():
    rows, labels = est_fixture()
    b = N.estimate(rows, labels)["byBand"]
    assert b[1]["Q"] == Fr(5, 2) / (3 + Fr(15, 2)) == Fr(5, 21)          # What 2.5, Rhat 7.5, RC_1 3
    assert b[2]["Q"] == Fr(16) / (2 + 4) == Fr(8, 3)                      # What 16, Rhat 4, RC_2 2
    assert 3 not in b


def test_false_confirmed_items_are_counted_as_defects():
    rows, labels = est_fixture()
    assert N.estimate(rows, labels)["falseConfirmed"] == 1


def test_possible_precision_per_reason_and_band_has_wilson_and_a_descriptive_flag():
    rows, labels = est_fixture()
    e = N.estimate(rows, labels)
    assert e["byReason"]["r1"] == {**N.wilson(3, 4), "descriptive": True}
    assert e["byReason"]["r2"]["k"] == 1 and e["byReason"]["r2"]["n"] == 5
    assert e["reasonsUnder005"] == []


def test_a_stratum_with_no_labels_is_an_error_not_a_silent_skip():
    rows, labels = est_fixture()
    rows.append(frow(999, reason="r3", band_=3))
    with pytest.raises(ValueError, match="r3"):
        N.estimate(rows, labels)


def test_split_by_own_edge_status_post_stratifies_within_each_stratum():
    rows, labels = est_fixture()
    for r in rows[:5]:
        r["split"] = "confirmed-in-flag-true"      # 5 of r1's 10 items; the labelled ones among them: i1..i4 -> t=3 f=1
    sp = N.estimate(rows, labels)["bySplit"]
    # r1 stratum: 5 items in the flag-true class, labelled there: i1..i4 (t=3,f=1) -> What 5*1/4, Rhat 5*3/4
    assert sp["confirmed-in-flag-true"]["What"] == Fr(5, 4) and sp["confirmed-in-flag-true"]["Rhat"] == Fr(15, 4)
    # the other class: r1's 5 unlabelled items borrow the stratum-wide rate (1/4 false), plus all of r2
    assert sp["possible"]["What"] == Fr(5, 4) + 16 and sp["possible"]["Rhat"] == Fr(15, 4) + 4


def test_split_class_is_flag_true_only_when_every_possible_hop_is_flow_unavailable():
    def it(hops, reason):
        x = item("s", reason=reason)
        x["chain"] = [{"kind": "Symbol"}] + [{"status": "possible", "reason": h} for h in hops]
        return x
    assert N.split_class(it(["flow-unavailable:python"], "flow-unavailable:python")) == "confirmed-in-flag-true"
    assert N.split_class(it(["flow-unavailable:python", "name-only-provider"], "flow-unavailable:python")) == "possible"
    assert N.split_class(it(["unknown-key"], "unknown-key")) == "possible"
    assert N.split_class(item("s", reason="flow-unavailable:go")) == "confirmed-in-flag-true"   # empty chain: the item reason
    assert N.split_class(item("s", status="confirmed")) == "confirmed"


# ---------------------------------------------------------------- burden, reduction, thresholds
@pytest.mark.parametrize("vals,p90", [([5], 5), (list(range(1, 11)), 9), (list(range(1, 12)), 10),
                                      (list(range(62, 0, -1)), 56), ([3, 1, 2], 3)])
def test_p90_is_the_exact_ceil_rank(vals, p90):
    assert N.p90(vals) == p90


def test_median_of_an_even_count_is_the_exact_midpoint():
    assert N.median([4, 1, 2, 9]) == Fr(3, 1) and N.median([1, 2]) == Fr(3, 2) and N.median([7, 1, 3]) == 3


def test_reduction_is_sum_over_sum_and_flagged_only_when_baseline_recall_is_lower():
    r = N.reduction([40, 60], [10, 10], (9, 10), (9, 10))
    assert r["reduction"] == Fr(5) and r["flagged"] is False
    assert N.reduction([40, 60], [10, 10], (9, 10), (8, 10))["flagged"] is True   # 8/10 < 9/10
    assert N.reduction([40, 60], [10, 10], (9, 10), (18, 20))["flagged"] is False  # equal fractions, different n
    assert N.reduction([1], [0], (1, 1), (1, 1))["reduction"] is None            # no system burden


def test_thresholds_are_exact_and_only_q_is_gated_at_gate_one():
    th = N.thresholds(q=Fr(4), w=Fr(40), r=Fr(10), rc=2, p90_in=100, reduction=Fr(5))
    assert th["Q"]["met"] and th["possiblePrecision"]["met"] and th["p90InScope"]["met"] and th["reduction"]["met"]
    assert th["Q"]["gated"] and not th["possiblePrecision"]["gated"]
    assert not N.thresholds(q=Fr(41, 10), w=Fr(41), r=Fr(8), rc=2, p90_in=100, reduction=Fr(5))["Q"]["met"]   # Q > 4
    assert N.thresholds(q=Fr(2), w=Fr(4), r=Fr(1), rc=9, p90_in=100, reduction=Fr(5))["possiblePrecision"]["met"]  # pp = 0.2 exactly
    assert not N.thresholds(q=Fr(2), w=Fr(41), r=Fr(10), rc=9, p90_in=100, reduction=Fr(5))["possiblePrecision"]["met"]  # 10/51 < 0.2
    assert not N.thresholds(q=Fr(1), w=Fr(1), r=Fr(1), rc=1, p90_in=101, reduction=Fr(5))["p90InScope"]["met"]
    assert not N.thresholds(q=Fr(1), w=Fr(1), r=Fr(1), rc=1, p90_in=100, reduction=Fr(49, 10))["reduction"]["met"]
    assert th["Q"]["limit"] == 4 and th["possiblePrecision"]["limit"] == "0.20" and th["p90InScope"]["limit"] == 100 and th["reduction"]["limit"] == 5


# ---------------------------------------------------------------- agreement and kappa
def _labs(s):
    return {("A", f"i{i}"): c for i, c in enumerate(s)}


def test_agreement_is_exact_and_kappa_corrects_for_chance():
    a = _labs("rrrrffffrr")      # r=real f=false
    b = _labs("rrrfffffrf")      # disagree at index 3 and 9
    full = lambda d: {k: ("real" if v == "r" else "false") for k, v in d.items()}  # noqa: E731
    g = N.agreement(full(a), full(b))
    assert (g["n"], g["agree"]) == (10, 8)
    # po 0.8; a: 6 real 4 false, b: 4 real 6 false -> pe = 0.6*0.4 + 0.4*0.6 = 0.48; kappa = 0.32/0.52
    assert g["kappa"] == round(0.32 / 0.52, 3)
    assert g["labelsUnstable"] is False                     # 8/10 is exactly 0.8: not below


def test_below_point_eight_marks_the_labels_unstable_and_at_point_eight_does_not():
    full = lambda s: {("A", f"i{i}"): ("real" if c == "r" else "false") for i, c in enumerate(s)}  # noqa: E731
    assert N.agreement(full("rrrrrrrrrr"), full("rrrrrrrrff"))["labelsUnstable"] is False      # 8/10
    assert N.agreement(full("rrrrrrrrrr"), full("rrrrrrrfff"))["labelsUnstable"] is True       # 7/10
    assert N.agreement(full("r" * 19 + "f"), full("r" * 14 + "f" * 6))["labelsUnstable"] is True    # 15/20
    assert N.agreement(full("r" * 19 + "f"), full("r" * 15 + "f" * 5))["labelsUnstable"] is False   # 16/20 = 0.8


# ---------------------------------------------------------------- bootstrap over answers
def boot_fixture():
    rows, labels, n = [], {}, 0
    for a in range(6):
        for h, (reason, bd) in enumerate([("r1", 1), ("r2", 2)]):
            for j in range(4):
                n += 1
                rows.append(frow(n, reason=reason, band_=bd, anchor=f"A{a}"))
                labels[(f"A{a}", f"i{n}")] = "real" if (a + j + h) % 3 == 0 else "false"
        n += 1
        rows.append(frow(n, status="confirmed", anchor=f"A{a}"))
        labels[(f"A{a}", f"i{n}")] = "real" if a % 2 == 0 else "false"
    return rows, labels


def test_bootstrap_q_matches_an_independent_resample_of_answers():
    import random
    rows, labels = boot_fixture()
    anchors = sorted({r["anchorId"] for r in rows})
    B = 40
    got = N.bootstrap(rows, labels, "g1", B=B)["Q"]
    rng = random.Random("ccg-noise-boot-v1|g1")
    qs = []
    for _ in range(B):
        pick = [rng.randrange(len(anchors)) for _ in anchors]
        rs = [r for i in pick for r in rows if r["anchorId"] == anchors[i]]
        qs.append(N.estimate(rs, labels)["Q"])
    qs.sort()
    assert (got["lower"], got["upper"]) == (qs[int(B * 0.025)], qs[int(B * 0.975) - 1])


def test_bootstrap_indices_are_250_and_9749_at_b_10000():
    assert N.percentile_indices(10000) == (250, 9749)


def test_bootstrap_is_deterministic_and_moves_with_the_gate_id():
    rows, labels = boot_fixture()
    a = N.bootstrap(rows, labels, "g1", B=60)["Q"]
    assert a == N.bootstrap(list(reversed(rows)), labels, "g1", B=60)["Q"]          # input order does not matter
    assert a != N.bootstrap(rows, labels, "g2", B=60)["Q"]


def test_a_resample_stratum_without_labels_uses_the_full_sample_rate():
    # one answer carries the only labelled r2 items; resamples that miss it must not drop r2 or raise
    rows, labels = boot_fixture()
    for r in rows:
        if r["reason"] == "r2" and r["anchorId"] != "A0":
            labels.pop((r["anchorId"], r["itemId"]), None)
    out = N.bootstrap(rows, labels, "g1", B=50)["Q"]
    assert out["lower"] is not None and out["upper"] is not None


# ---------------------------------------------------------------- Q-NC5: unbound links with multi-unit providers
def test_unbound_links_whose_key_has_providers_in_more_than_one_unit_are_counted():
    snap = {"links": [{"keyNorm": "k1", "bound": False, "status": "confirmed"},
                      {"keyNorm": "k1", "bound": False, "status": "possible"},
                      {"keyNorm": "k1", "bound": True, "status": "confirmed"},     # bound: not the shape
                      {"keyNorm": "k2", "bound": False, "status": "confirmed"},     # one unit only
                      {"keyNorm": "k3", "bound": False, "status": "confirmed"}],    # no provider
            "providers": [{"keyNorm": "k1", "unitId": "u1"}, {"keyNorm": "k1", "unitId": "u2"},
                          {"keyNorm": "k2", "unitId": "u1"}, {"keyNorm": "k2", "unitId": "u1"}]}
    assert N.unbound_multi_unit(snap) == {"unboundMultiUnit": 2, "unboundMultiUnitConfirmed": 1}


# ---------------------------------------------------------------- spot-check draw
def test_spot_check_is_a_hash_sort_of_the_labelled_set_of_20():
    keys = [("A", f"i{i}") for i in range(50)]
    got = N.spot_check(keys, "g1", "snap1")
    seed = "ccg-noise-v1|g1|snap1|spot"
    assert got == sorted(keys, key=lambda k: hashlib.sha256(f"{seed}|{k[0]}|{k[1]}".encode()).hexdigest())[:20]
    assert N.spot_check(keys[:7], "g1", "snap1") != [] and len(N.spot_check(keys[:7], "g1", "snap1")) == 7


# ---------------------------------------------------------------- score CLI, end to end
def _draw(tmp_path, answers):
    _write_answers(tmp_path / "a.jsonl", answers)
    (tmp_path / "e.json").write_text(json.dumps(EXCL))
    cust = tmp_path / "cust"
    assert N.main(["draw", "--answers", str(tmp_path / "a.jsonl"), "--exclusions", str(tmp_path / "e.json"),
                   "--gate-id", "g", "--snapshot-id", "s", "--custodian", str(cust)]) == 0
    return cust


def _score(tmp_path, cust, extra=()):
    return N.main(["score", "--answers", str(tmp_path / "a.jsonl"), "--exclusions", str(tmp_path / "e.json"),
                   "--gate-id", "g", "--snapshot-id", "s", "--custodian", str(cust), "--boot", "30", *extra])


def _scoring_answers():
    out = []
    for a in range(4):
        items = [item(f"p{a}{j}", reason="r1" if j % 2 else "r2", distance=1 + j % 3) for j in range(12)]
        items += [item(f"c{a}", status="confirmed"), item(f"x{a}", path="cpg/out.py")]
        out.append(answer(f"SECRET{a}", items))
    return out


def _label_all(cust, name="noise-labels.jsonl", flip=()):
    rows = [json.loads(l) for l in (cust / "noise-sample.jsonl").read_text().splitlines()]
    rows += [json.loads(l) for l in (cust / "noise-census.jsonl").read_text().splitlines()]
    lab = [{"anchorId": r["anchorId"], "itemId": r["itemId"], "distance": r["distance"],
            "label": ("false" if i in flip else "real") if r["status"] == "possible" and i % 2 else "real"} for i, r in enumerate(rows)]
    (cust / name).write_text("".join(json.dumps(x) + "\n" for x in lab))
    return lab


def test_score_reports_counts_only_with_thresholds_and_the_scope_counts(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    capsys.readouterr()
    _label_all(cust)
    (cust / "noise-baseline.jsonl").write_text("".join(json.dumps({"anchorId": f"SECRET{a}", "count": 200}) + "\n" for a in range(4)))
    (cust / "noise-recall.json").write_text(json.dumps({"system": [9, 10], "baseline": [9, 10]}))
    assert _score(tmp_path, cust) == 0
    out = capsys.readouterr().out
    rep = json.loads(out)
    assert "SECRET" not in out and "u1:p" not in out
    assert rep["outOfScope"]["perAnswer"] == [1, 1, 1, 1] and rep["outOfScope"]["byReason"] == {"name-only-provider": 4}
    assert rep["burden"]["inScope"]["p90"] == 13 and rep["burden"]["allListed"]["p90"] == 14
    assert rep["reduction"]["inScope"] == round(800 / 52, 3) and rep["reduction"]["allListed"] == round(800 / 56, 3)
    assert set(rep["thresholds"]) == {"Q", "possiblePrecision", "p90InScope", "reduction"}
    assert rep["thresholds"]["p90InScope"]["met"] is True
    assert rep["estimates"]["confirmedPrecision"]["k"] == 4
    assert rep["labelsUnstable"] is None            # no second labeller supplied: nothing to say, not "stable"


def _spot(cust):
    assert N.main(["spot", "--gate-id", "g", "--snapshot-id", "s", "--custodian", str(cust)]) == 0
    return [json.loads(l) for l in (cust / "noise-spot.jsonl").read_text().splitlines()]


def _labels2(cust, spot, flip_all=False, drop=0, extra=None):
    first = {(r["anchorId"], r["itemId"]): r["label"] for r in (json.loads(l) for l in (cust / "noise-labels.jsonl").read_text().splitlines())}
    rows = [{"anchorId": s["anchorId"], "itemId": s["itemId"],
             "label": ({"real": "false", "false": "real"}[first[(s["anchorId"], s["itemId"])]] if flip_all else first[(s["anchorId"], s["itemId"])])}
            for s in spot][drop:]
    if extra:
        rows.append(extra)
    (cust / "noise-labels2.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


def test_score_marks_possible_precision_labels_unstable_below_point_eight_agreement(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    _label_all(cust)
    _labels2(cust, _spot(cust), flip_all=True)
    capsys.readouterr()
    assert _score(tmp_path, cust) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["agreement"]["labelsUnstable"] is True and rep["estimates"]["possiblePrecisionNote"] == "labels unstable"


def test_score_accepts_a_second_labeller_who_relabelled_exactly_the_spot_keys(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    _label_all(cust)
    _labels2(cust, _spot(cust))
    capsys.readouterr()
    assert _score(tmp_path, cust) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["agreement"]["n"] == 20 and rep["labelsUnstable"] is False


def test_score_refuses_a_second_labeller_file_that_is_not_the_spot_keys(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    lab = _label_all(cust)
    spot = _spot(cust)
    spotkeys = {(s["anchorId"], s["itemId"]) for s in spot}
    other = next(l for l in lab if (l["anchorId"], l["itemId"]) not in spotkeys)
    _labels2(cust, spot, drop=1, extra={"anchorId": other["anchorId"], "itemId": other["itemId"], "label": "real"})  # same size, wrong key
    capsys.readouterr()
    assert _score(tmp_path, cust) == 4
    assert capsys.readouterr().out == ""
    _labels2(cust, spot, drop=1)                                                                                  # a subset
    assert _score(tmp_path, cust) == 4


def test_score_refuses_a_second_labeller_file_with_keys_beyond_the_spot_list(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    lab = _label_all(cust)
    spot = _spot(cust)
    spotkeys = {(s["anchorId"], s["itemId"]) for s in spot}
    other = next(l for l in lab if (l["anchorId"], l["itemId"]) not in spotkeys)
    _labels2(cust, spot, extra={"anchorId": other["anchorId"], "itemId": other["itemId"], "label": "real"})
    capsys.readouterr()
    assert _score(tmp_path, cust) == 4


def test_score_refuses_a_second_labeller_file_without_a_spot_list(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    _label_all(cust)
    _label_all(cust, "noise-labels2.jsonl")
    assert _score(tmp_path, cust) == 4


def test_agreement_without_overlap_is_not_measured_not_stable():
    g = N.agreement({("A", "x"): "real"}, {})
    assert g["n"] == 0 and g["labelsUnstable"] is None and g["note"] == "not measured"


def test_score_with_an_empty_second_labeller_file_reports_not_measured(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    _label_all(cust)
    _spot(cust)
    (cust / "noise-labels2.jsonl").write_text("")
    capsys.readouterr()
    assert _score(tmp_path, cust) == 4          # an empty file is not the 20 spot keys


# ---------------------------------------------------------------- complete label coverage (never a vacuous gate)
def _drop(cust, pred):
    rows = [json.loads(l) for l in (cust / "noise-labels.jsonl").read_text().splitlines()]
    kept = [r for r in rows if not pred(r)]
    assert len(kept) < len(rows)
    (cust / "noise-labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in kept))


def _refused(tmp_path, cust, capsys):
    capsys.readouterr()
    assert _score(tmp_path, cust) == 4
    cap = capsys.readouterr()
    assert cap.out == "" and "SECRET" not in cap.err and "u1:" not in cap.err


def test_score_refuses_an_empty_labels_file(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    (cust / "noise-labels.jsonl").write_text("")
    _refused(tmp_path, cust, capsys)


def test_score_refuses_labels_missing_one_census_item(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    _label_all(cust)
    census = {json.loads(l)["itemId"] for l in (cust / "noise-census.jsonl").read_text().splitlines()}
    gone = []
    _drop(cust, lambda r: r["itemId"] in census and not gone and not gone.append(1))
    _refused(tmp_path, cust, capsys)


def test_score_refuses_labels_missing_one_drawn_item(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    _label_all(cust)
    drawn = {json.loads(l)["itemId"] for l in (cust / "noise-sample.jsonl").read_text().splitlines()}
    gone = []
    _drop(cust, lambda r: r["itemId"] in drawn and not gone and not gone.append(1))
    _refused(tmp_path, cust, capsys)


def test_score_refuses_labels_missing_a_whole_stratum(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    _label_all(cust)
    first = json.loads((cust / "noise-sample.jsonl").read_text().splitlines()[0])
    sample = [json.loads(l) for l in (cust / "noise-sample.jsonl").read_text().splitlines()]
    ids = {r["itemId"] for r in sample if (r["reason"], r["band"]) == (first["reason"], first["band"])}
    _drop(cust, lambda r: r["itemId"] in ids)
    _refused(tmp_path, cust, capsys)


def test_score_rejects_a_label_outside_the_draw(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    lab = _label_all(cust)
    (cust / "noise-labels.jsonl").write_text(json.dumps({"anchorId": "SECRET0", "itemId": "nope", "label": "real", "distance": 1}) + "\n")
    assert _score(tmp_path, cust) == 4


def test_spot_subcommand_writes_the_second_labeller_list_under_the_custodian(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    lab = _label_all(cust)
    capsys.readouterr()
    assert N.main(["spot", "--gate-id", "g", "--snapshot-id", "s", "--custodian", str(cust)]) == 0
    rep = json.loads(capsys.readouterr().out)
    got = [json.loads(l) for l in (cust / "noise-spot.jsonl").read_text().splitlines()]
    assert rep == {"labelled": len(lab), "spot": 20} and len(got) == 20
    assert N.main(["spot", "--gate-id", "g", "--snapshot-id", "s", "--custodian", str(Path(N.REPO) / "x")]) == 2


def test_unreadable_answers_exit_4(tmp_path):
    assert N.main(["draw", "--answers", str(tmp_path / "missing"), "--exclusions", str(tmp_path / "e.json"),
                   "--gate-id", "g", "--snapshot-id", "s", "--custodian", str(tmp_path / "c")]) == 4


def test_allocation_cap_binds_when_a_minimum_plus_its_share_would_pass_n_h():
    # a: N=5, minimum 5, share 90*5//305 = 1 would make 6 > 5; the capped slot goes to b, which totals 100
    assert N.allocate({"a": 5, "b": 300}) == {"a": 5, "b": 95}


def test_allocation_floors_quotas_so_the_sample_is_never_over_target():
    # six equal strata: remainder 70 -> 11.67 each; rounding would give 12 each (102 in all)
    got = N.allocate({k: 1000 for k in "abcdef"})
    assert got == {"a": 17, "b": 17, "c": 17, "d": 17, "e": 16, "f": 16} and sum(got.values()) == 100


def test_bootstrap_fallback_rate_is_the_full_sample_rate_not_zero():
    rows, labels, n = [], {}, 0
    for a in range(3):
        for j in range(4):
            n += 1
            rows.append(frow(n, anchor=f"A{a}"))
            if a == 0:
                labels[(f"A{a}", f"i{n}")] = "real" if j < 2 else "false"
    out = N.bootstrap(rows, labels, "g1", B=60)["Q"]
    assert out["used"] == 60 and out["lower"] == out["upper"] == Fr(1)
