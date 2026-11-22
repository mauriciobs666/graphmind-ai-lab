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


def _baseline_files(tmp_path, cust, baseline=None, recall=None, hits=None, key=False, record=2):
    """The custodian-side inputs of the reduction: baseline rows, recall pairs, ``system-hits.jsonl`` and the gate record."""
    hits = _hits() if hits is None else hits
    rows = list((_baseline() if baseline is None else baseline).values())
    (cust / "noise-baseline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (cust / "noise-recall.json").write_text(json.dumps(_recall(hits) if recall is None else recall))
    (cust / "system-hits.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hits))
    if key:
        (cust / "noise-baseline-key.jsonl").write_text("".join(json.dumps(dict(r, count=r["count"] // 2, countInScope=r["countInScope"] // 2)) + "\n" for r in rows))
        (cust / "noise-recall-key.json").write_text(json.dumps(_recall(hits)))
    (tmp_path / "gate.json").write_text(json.dumps({"defaultDepthBound": record}))
    return ["--gate-record", str(tmp_path / "gate.json")]


def test_score_reports_counts_only_with_thresholds_and_the_scope_counts(tmp_path, capsys):
    cust = _draw(tmp_path, _bounded(_scoring_answers()))
    capsys.readouterr()
    _label_all(cust)
    extra = _baseline_files(tmp_path, cust)
    assert _score(tmp_path, cust, extra) == 0
    out = capsys.readouterr().out
    rep = json.loads(out)
    assert "SECRET" not in out and "u1:p" not in out
    assert rep["outOfScope"]["perAnswer"] == [1, 1, 1, 1] and rep["outOfScope"]["byReason"] == {"name-only-provider": 4}
    assert rep["burden"]["inScope"]["p90"] == 13 and rep["burden"]["allListed"]["p90"] == 14
    assert rep["reduction"]["inScope"] == round(300 / 52, 3) and rep["reduction"]["allListed"] == round(600 / 56, 3)
    assert rep["baseline"]["unresolved"] == 1 and "keyScoped" not in rep
    assert set(rep["thresholds"]) == {"Q", "possiblePrecision", "p90InScope", "reduction"}
    assert rep["thresholds"]["p90InScope"]["met"] is True
    assert rep["estimates"]["confirmedPrecision"]["k"] == 4
    assert rep["labelsUnstable"] is None            # no second labeller supplied: nothing to say, not "stable"


def test_score_cli_reads_the_key_scoped_variant_beside_the_headline(tmp_path, capsys):
    cust = _draw(tmp_path, _bounded(_scoring_answers()))
    _label_all(cust)
    extra = _baseline_files(tmp_path, cust, key=True)
    capsys.readouterr()
    assert _score(tmp_path, cust, extra) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["keyScoped"]["reduction"]["inScope"] == round(150 / 52, 3) and rep["reduction"]["inScope"] == round(300 / 52, 3)


def _refused_with(tmp_path, cust, capsys, extra):
    capsys.readouterr()
    assert _score(tmp_path, cust, extra) == 4
    cap = capsys.readouterr()
    assert cap.out == "" and "SECRET" not in cap.err and "u1:" not in cap.err
    return cap.err


def test_score_cli_refuses_a_baseline_without_the_gate_record_and_each_bad_input(tmp_path, capsys):
    cust = _draw(tmp_path, _bounded(_scoring_answers()))
    _label_all(cust)
    extra = _baseline_files(tmp_path, cust)
    assert "gate-record" in _refused_with(tmp_path, cust, capsys, [])
    (tmp_path / "gate.json").write_text(json.dumps({"defaultDepthBound": 3}))
    assert "bound" in _refused_with(tmp_path, cust, capsys, extra)
    (tmp_path / "gate.json").write_text(json.dumps({"defaultDepthBound": 2}))
    rows = [json.loads(l) for l in (cust / "noise-baseline.jsonl").read_text().splitlines()]
    (cust / "noise-baseline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + [rows[0]]))      # a duplicated anchor row
    assert "duplicate" in _refused_with(tmp_path, cust, capsys, extra)
    (cust / "noise-baseline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (cust / "system-hits.jsonl").unlink()
    _refused_with(tmp_path, cust, capsys, extra)
    _baseline_files(tmp_path, cust)
    (cust / "noise-recall.json").write_text(json.dumps(dict(_recall(), itemSetSha256="f" * 64)))
    assert "itemSetSha256" in _refused_with(tmp_path, cust, capsys, extra)


def test_score_cli_refuses_a_key_variant_with_only_one_of_its_two_files(tmp_path, capsys):
    cust = _draw(tmp_path, _bounded(_scoring_answers()))
    _label_all(cust)
    extra = _baseline_files(tmp_path, cust, key=True)
    (cust / "noise-recall-key.json").unlink()
    assert "come together" in _refused_with(tmp_path, cust, capsys, extra)


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


# ================================================================ P1-36 store-side note §7 consumer edits
# (``docs/plans/connected-code-graph-p1-noise-export-graph.md`` §7; ``docs/reviews/connected-code-graph-p1-noise-export-ml.md``)

# ---------------------------------------------------------------- item 4 (m1): the frame is unique on (anchorId, itemId)
def test_build_frame_refuses_one_planted_duplicate_item_with_the_count_only():
    a = answer("SECRET-A", [item("s1"), item("s2")], members=[item("s1")])        # s1 listed as item and as group member
    with pytest.raises(ValueError, match=r"1 duplicate") as e:
        N.build_frame([a], EXCL)
    assert "SECRET-A" not in str(e.value) and "s1" not in str(e.value)


def test_build_frame_accepts_the_same_item_under_two_anchors():
    f = N.build_frame([answer("A", [item("s1")]), answer("B", [item("s1")])], EXCL)
    assert len(f["frame"]) == 2


def test_build_frame_counts_duplicates_not_just_detects_them():
    a = answer("A", [item("s1"), item("s2")], members=[item("s1"), item("s2")])
    with pytest.raises(ValueError, match=r"2 duplicate"):
        N.build_frame([a], EXCL)


def test_build_frame_refuses_a_possible_item_whose_reason_is_not_a_string():
    bad = item("s1")
    del bad["reason"]
    with pytest.raises(ValueError, match=r"1 possible item"):
        N.build_frame([answer("A", [bad, item("s2")])], EXCL)
    bad2 = item("s1")
    bad2["reason"] = None
    with pytest.raises(ValueError, match=r"1 possible item"):
        N.build_frame([answer("A", [bad2])], EXCL)


def test_cli_draw_exits_4_on_a_duplicate_item(tmp_path):
    _write_answers(tmp_path / "a.jsonl", [answer("A", [item("s1")], members=[item("s1")])])
    (tmp_path / "e.json").write_text(json.dumps(EXCL))
    assert N.main(["draw", "--answers", str(tmp_path / "a.jsonl"), "--exclusions", str(tmp_path / "e.json"),
                   "--gate-id", "g", "--snapshot-id", "s", "--custodian", str(tmp_path / "c")]) == 4


# ---------------------------------------------------------------- item 5 (m2): agreement split by status, possible-item note
def test_agreement_carries_the_wilson_interval_of_the_raw_agreement():
    full = lambda s: {("A", f"i{i}"): ("real" if c == "r" else "false") for i, c in enumerate(s)}  # noqa: E731
    g = N.agreement(full("r" * 16 + "f" * 4), full("r" * 20))
    assert g["wilson"] == N.wilson(16, 20) and g["wilson"]["lower"] is not None
    assert N.agreement({}, {})["wilson"]["s"] is None


def test_split_agreement_counts_confirmed_and_possible_keys_separately():
    first = {("A", f"c{i}"): "real" for i in range(8)} | {("A", f"p{i}"): "real" for i in range(12)}
    second = dict(first)
    second[("A", "c0")] = "false"                       # a confirmed disagreement
    second[("A", "p0")] = "false"                       # two possible disagreements
    second[("A", "p1")] = "false"
    status = {k: ("confirmed" if k[1].startswith("c") else "possible") for k in first}
    sp = N.split_agreement(first, second, status)
    assert (sp["confirmed"]["n"], sp["confirmed"]["agree"]) == (8, 7)
    assert (sp["possible"]["n"], sp["possible"]["agree"]) == (12, 10)
    assert sp["possible"]["wilson"] == N.wilson(10, 12)
    assert sp["possible"]["rawAgreement"] == round(10 / 12, 3)


def test_split_agreement_with_no_possible_keys_is_not_measured_for_that_class():
    first = {("A", "c0"): "real"}
    sp = N.split_agreement(first, dict(first), {("A", "c0"): "confirmed"})
    assert sp["possible"]["n"] == 0 and sp["possible"]["rawAgreement"] is None and sp["possible"]["wilson"]["s"] is None


@pytest.mark.parametrize("n_possible,unstable,note", [
    (9, False, "agreement not measured on possible items"), (9, True, "agreement not measured on possible items"),
    (0, None, "agreement not measured on possible items"), (10, True, "labels unstable"), (10, False, None), (20, False, None)])
def test_possible_precision_note_needs_ten_possible_spot_keys(n_possible, unstable, note):
    assert N.possible_precision_note(unstable, n_possible) == note


def _confirmed_heavy_answers():
    out = []
    for a in range(4):
        items = [item(f"p{a}{j}", reason="r1", distance=1 + j % 3) for j in range(2)]
        items += [item(f"c{a}{j}", status="confirmed") for j in range(30)]
        out.append(answer(f"SECRET{a}", items))
    return out


def test_score_says_agreement_not_measured_on_possible_items_when_the_spot_keys_are_mostly_confirmed(tmp_path, capsys):
    cust = _draw(tmp_path, _confirmed_heavy_answers())
    _label_all(cust)
    _labels2(cust, _spot(cust), flip_all=True)             # overall agreement 0: would read "labels unstable"
    capsys.readouterr()
    assert _score(tmp_path, cust) == 0
    rep = json.loads(capsys.readouterr().out)
    ag = rep["agreement"]
    assert ag["possible"]["n"] < 10 and ag["confirmed"]["n"] + ag["possible"]["n"] == 20
    assert rep["estimates"]["possiblePrecisionNote"] == "agreement not measured on possible items"
    assert ag["wilson"]["n"] == 20


def test_score_reports_the_split_and_keeps_labels_unstable_when_ten_or_more_spot_keys_are_possible(tmp_path, capsys):
    cust = _draw(tmp_path, _scoring_answers())
    _label_all(cust)
    _labels2(cust, _spot(cust), flip_all=True)
    capsys.readouterr()
    assert _score(tmp_path, cust) == 0
    rep = json.loads(capsys.readouterr().out)
    ag = rep["agreement"]
    assert ag["possible"]["n"] >= 10 and ag["possible"]["n"] + ag["confirmed"]["n"] == 20
    assert ag["possible"]["agree"] == 0 and rep["estimates"]["possiblePrecisionNote"] == "labels unstable"


# ---------------------------------------------------------------- item 1: burden pairing, in the score reduction and in the bootstrap
def _bounded(answers, bound=2):
    return [dict(a, depthBound={"bound": bound}) for a in answers]


def _labels_for(answers):
    f = N.build_frame(answers, EXCL)
    drawn = N.draw(f["frame"], N.allocate(f["strata"]), "g", "s")
    return {(r["anchorId"], r["itemId"]): "real" for r in drawn + f["census"]}


def brow(anchor, count=200, in_scope=100, type_="symbol", bound=2, by_name=(2, 3), unresolved=False, own=10):
    row = {"anchorId": anchor, "type": type_, "count": count, "countInScope": in_scope, "countConfirmedOwn": own,
           "byName": {"resolvedSite": by_name[0], "unresolvedSite": by_name[1]}, "bound": bound}
    if unresolved:
        row["unresolved"] = True
    return row


def _baseline():
    return {"SECRET0": brow("SECRET0"), "SECRET1": brow("SECRET1"), "SECRET2": brow("SECRET2", type_="contract"),
            "SECRET3": brow("SECRET3", 0, 0, type_="unresolved", by_name=(0, 0), unresolved=True, own=0)}


def hit_row(anchor, qname, lines, hit=True, key_error=False, type_="symbol", **imp):
    impacted = {"type": type_, "repoId": "r", "path": "a.py", ("qname" if type_ == "symbol" else "keyPath"): qname, **imp}
    return {"anchorId": anchor, "impacted": impacted, "lines": lines, "hit": hit, "keyError": key_error}


def _hits():
    rows = [hit_row(f"SECRET{i % 4}", f"q{i}", [i, i + 5]) for i in range(10)]
    rows[9]["keyError"] = True
    return rows


def _recall(hits=None):
    hits = _hits() if hits is None else hits
    return {"system": [9, 10], "baseline": [8, 10], "systemSym": [8, 9], "baselineSym": [7, 9],
            "paired": {"both": 7, "systemOnly": 2, "baselineOnly": 1, "neither": 0}, "configGold": 1,
            "itemSetSha256": N.item_set_sha256(hits),
            "exKeyError": {"system": [8, 9], "baseline": [7, 9], "systemSym": [7, 8], "baselineSym": [6, 8],
                           "itemSetSha256": N.item_set_sha256([h for h in hits if not h["keyError"]])}}


def _run(answers=None, baseline=None, recall=None, hits=None, depth=2, **kw):
    answers = _bounded(_scoring_answers()) if answers is None else answers
    hits = _hits() if hits is None else hits
    return N.score(answers, EXCL, _labels_for(answers), "g", "s", B=30, baseline=_baseline() if baseline is None else baseline,
                   recall=_recall(hits) if recall is None else recall, system_hits=hits, depth_bound=depth, **kw)


def test_in_scope_reduction_uses_count_in_scope_and_all_listed_uses_count():
    rep = _run()
    # in-scope burden 13 per answer, all-listed 14; baseline count 200 / countInScope 100 on three answers
    assert rep["reduction"]["inScope"] == round(300 / 52, 3)        # crossed with count would be 600/52 = 11.538
    assert rep["reduction"]["allListed"] == round(600 / 56, 3)      # crossed with countInScope would be 300/56 = 5.357
    assert 300 / 52 != 600 / 52 != 300 / 56


def test_the_reduction_threshold_judges_the_in_scope_pairing():
    b = _baseline()
    for a in b.values():
        a["count"], a["countInScope"] = (1000, 10) if a["count"] else (0, 0)    # all-listed passes 5x, in-scope does not
    rep = _run(baseline=b)
    assert rep["reduction"]["allListed"] > 5 > rep["reduction"]["inScope"]
    assert rep["thresholds"]["reduction"]["met"] is False


def _independent_reduction_interval(rows, anchors, num, den, B, gate="g1"):
    import random
    rng = random.Random("ccg-noise-boot-v1|" + gate)
    vals = []
    for _ in range(B):
        pick = [anchors[rng.randrange(len(anchors))] for _ in anchors]
        d = sum(den[a] for a in pick)
        if d:
            vals.append(Fr(sum(num[a] for a in pick), d))
    vals.sort()
    return vals[int(len(vals) * 0.025)], vals[int(len(vals) * 0.975) - 1]


def test_bootstrap_pairs_each_baseline_column_with_its_own_burden_and_gives_each_its_own_interval():
    rows, labels = boot_fixture()
    anchors = [f"A{a}" for a in range(6)]
    extra = {"A1": 3, "A3": 7, "A4": 2}                  # all-listed burden = in-scope burden + out-of-scope items
    all_b = {a: sum(r["anchorId"] == a for r in rows) + extra.get(a, 0) for a in anchors}
    in_b = {a: sum(r["anchorId"] == a for r in rows) for a in anchors}
    base = {a: {"count": 40 + 11 * i * i, "countInScope": 5 + 3 * i} for i, a in enumerate(anchors)}
    B = 60
    got = N.bootstrap(rows, labels, "g1", B=B, anchors=anchors, baseline=base, all_burden=all_b)
    want_in = _independent_reduction_interval(rows, anchors, {a: base[a]["countInScope"] for a in anchors}, in_b, B)
    want_all = _independent_reduction_interval(rows, anchors, {a: base[a]["count"] for a in anchors}, all_b, B)
    assert (got["reduction"]["lower"], got["reduction"]["upper"]) == want_in
    assert (got["reductionAll"]["lower"], got["reductionAll"]["upper"]) == want_all
    assert want_in != want_all
    crossed = _independent_reduction_interval(rows, anchors, {a: base[a]["count"] for a in anchors}, in_b, B)
    assert crossed != want_in


def test_bootstrap_with_a_baseline_needs_the_all_listed_burden():
    rows, labels = boot_fixture()
    with pytest.raises(ValueError, match="all-listed"):
        N.bootstrap(rows, labels, "g1", B=5, baseline={"A0": {"count": 1, "countInScope": 1}})


def test_score_interval_for_the_in_scope_reduction_is_the_countinscope_pairing():
    rep = _run()
    a = _bounded(_scoring_answers())
    f = N.build_frame(a, EXCL)
    anchors = sorted(x["anchorId"] for x in a)
    base = _baseline()
    boot = N.bootstrap(f["frame"], _labels_for(a), "g", 30, anchors, base, dict(zip(anchors, f["burden"]["allListed"])))
    assert rep["reduction"]["interval"] == N._ci(boot["reduction"])
    assert rep["reduction"]["intervalAllListed"] == N._ci(boot["reductionAll"])


# ---------------------------------------------------------------- item 2: refusals (ValueError with counts only; the CLI turns each into exit 4)
def test_item_set_sha256_is_the_pinned_canonical_form():
    rows = [hit_row("B1", "p.f", [10, 9, 2]), hit_row("a1", "x.y", [5], type_="config"),
            hit_row("é", "q", [1, 3])]
    want = "\n".join(sorted(["B1|symbol:r:a.py:p.f|2,9,10", "a1|config:r:a.py:x.y|5", "é|symbol:r:a.py:q|1,3"]))
    assert want.split("\n") == ["B1|symbol:r:a.py:p.f|2,9,10", "a1|config:r:a.py:x.y|5", "é|symbol:r:a.py:q|1,3"]  # code-point order
    assert N.item_set_sha256(rows) == hashlib.sha256(want.encode("utf-8")).hexdigest()
    assert N.item_set_sha256(list(reversed(rows))) == N.item_set_sha256(rows)
    assert N.item_set_sha256([hit_row("B1", "p.f", [10, 9, 3])]) != N.item_set_sha256([hit_row("B1", "p.f", [10, 9, 2])])


def test_the_baseline_pair_k_n_must_be_coherent():
    for bad in ([5, 4], [1, 0], [0, 0], [-1, 5], [2, -3]):
        r = _recall()
        r["baseline"] = bad
        with pytest.raises(ValueError, match="noise-recall"):
            _run(recall=r)


@pytest.mark.parametrize("key", ["system", "systemSym", "baselineSym"])
def test_every_recall_pair_is_checked_not_only_the_headline_baseline(key):
    r = _recall()
    r[key] = [10, 9]
    with pytest.raises(ValueError, match="noise-recall"):
        _run(recall=r)


def test_the_ex_key_error_pair_is_checked_too():
    r = _recall()
    r["exKeyError"]["baseline"] = [9, 8]
    with pytest.raises(ValueError, match="noise-recall"):
        _run(recall=r)


def test_recall_with_a_different_item_set_hash_is_refused():
    r = _recall()
    r["itemSetSha256"] = "0" * 64
    with pytest.raises(ValueError, match="itemSetSha256"):
        _run(recall=r)
    hits = _hits()
    with pytest.raises(ValueError, match="itemSetSha256"):        # the same recall against a system-hits.jsonl that moved
        _run(hits=hits[:-1] + [hit_row("SECRET0", "other", [1])], recall=_recall(hits))


def test_the_ex_key_error_figure_carries_its_own_hash_over_the_reduced_set():
    r = _recall()
    r["exKeyError"]["itemSetSha256"] = r["itemSetSha256"]           # the full-set hash on the reduced figure
    with pytest.raises(ValueError, match="exKeyError"):
        _run(recall=r)


def test_a_recall_without_the_hash_is_refused():
    r = _recall()
    del r["itemSetSha256"]
    with pytest.raises(ValueError, match="itemSetSha256"):
        _run(recall=r)


def test_baseline_anchor_set_must_equal_the_answers():
    missing = _baseline()
    del missing["SECRET2"]
    with pytest.raises(ValueError, match=r"1 answer\(s\) without a baseline row, 0"):
        _run(baseline=missing)
    extra = _baseline()
    extra["SECRET9"] = brow("SECRET9")
    with pytest.raises(ValueError, match=r"0 answer\(s\) without a baseline row, 1"):
        _run(baseline=extra)


def test_count_in_scope_must_lie_between_zero_and_count():
    over = _baseline()
    over["SECRET1"]["countInScope"] = 201
    with pytest.raises(ValueError, match=r"1 baseline row\(s\) .*countInScope"):
        _run(baseline=over)
    neg = _baseline()
    neg["SECRET1"]["countInScope"] = -1
    with pytest.raises(ValueError, match=r"1 baseline row\(s\) .*countInScope"):
        _run(baseline=neg)
    edge = _baseline()
    edge["SECRET1"]["countInScope"] = 200                           # == count is allowed
    edge["SECRET0"]["countInScope"] = 0                             # 0 is allowed
    _run(baseline=edge)


def test_an_unresolved_row_must_have_count_zero():
    b = _baseline()
    b["SECRET3"]["count"] = 4
    with pytest.raises(ValueError, match=r"1 unresolved baseline row\(s\)"):
        _run(baseline=b)


def test_a_baseline_row_bound_different_from_the_record_is_refused():
    b = _baseline()
    b["SECRET2"]["bound"] = 3
    with pytest.raises(ValueError, match=r"1 baseline row\(s\) .*bound"):
        _run(baseline=b)


def test_an_answer_depth_bound_different_from_the_record_is_refused():
    ans = _bounded(_scoring_answers())
    ans[1]["depthBound"]["bound"] = 3
    with pytest.raises(ValueError, match=r"1 answer\(s\) .*depthBound"):
        _run(answers=ans)
    with pytest.raises(ValueError, match=r"4 answer\(s\) .*depthBound"):
        _run(depth=3, baseline={k: dict(v, bound=3) for k, v in _baseline().items()})


def test_an_answer_without_a_depth_bound_is_refused_when_a_baseline_is_scored():
    ans = _scoring_answers()
    with pytest.raises(ValueError, match=r"4 answer\(s\) .*depthBound"):
        _run(answers=ans)


def test_a_baseline_needs_the_record_depth_bound_the_hits_and_the_recall():
    ans = _bounded(_scoring_answers())
    lab = _labels_for(ans)
    with pytest.raises(ValueError, match="depth bound"):
        N.score(ans, EXCL, lab, "g", "s", B=5, baseline=_baseline(), recall=_recall(), system_hits=_hits())
    with pytest.raises(ValueError, match="system-hits"):
        N.score(ans, EXCL, lab, "g", "s", B=5, baseline=_baseline(), recall=_recall(), depth_bound=2)


def test_the_key_scoped_variant_is_held_to_the_same_refusals():
    key = {a: dict(r) for a, r in _baseline().items()}
    key["SECRET1"]["countInScope"] = 500
    with pytest.raises(ValueError, match="key-scoped"):
        _run(baseline_key=key, recall_key=_recall())
    key = {a: dict(r) for a, r in _baseline().items()}
    r = _recall()
    r["itemSetSha256"] = "1" * 64
    with pytest.raises(ValueError, match="key-scoped"):
        _run(baseline_key=key, recall_key=r)
    with pytest.raises(ValueError, match="key-scoped"):
        _run(baseline_key=_baseline())                                  # variant baseline without its recall


def test_refusal_messages_carry_counts_only():
    b = _baseline()
    b["SECRET1"]["countInScope"] = 999
    with pytest.raises(ValueError) as e:
        _run(baseline=b)
    assert "SECRET" not in str(e.value)


# ---------------------------------------------------------------- item 3: reports (counts only)
def test_reduction_says_when_the_baseline_recalls_more_and_equal_recall_is_neither_flag_nor_higher():
    assert N.reduction([40], [10], (9, 10), (10, 10)) == {"reduction": Fr(4), "flagged": False, "baselineRecallHigher": True}
    assert N.reduction([40], [10], (9, 10), (18, 20))["baselineRecallHigher"] is False
    assert N.reduction([40], [10], (9, 10), (8, 10))["baselineRecallHigher"] is False


def test_report_burden_and_reduction_by_anchor_type_baseline_and_system():
    by = _run()["reduction"]["byType"]
    assert set(by) == {"symbol", "contract", "unresolved"}
    assert by["symbol"] == {"anchors": 2, "baselineCount": 400, "baselineInScope": 200, "baselineConfirmedOwn": 20,
                            "systemInScope": 26, "systemAllListed": 28,
                            "reductionInScope": round(200 / 26, 3), "reductionAllListed": round(400 / 28, 3)}
    assert by["contract"]["baselineInScope"] == 100 and by["contract"]["systemAllListed"] == 14
    assert by["unresolved"]["anchors"] == 1 and by["unresolved"]["reductionInScope"] == 0.0


def test_report_by_name_unresolved_rows_and_the_non_symbol_system_items():
    ans = _bounded(_scoring_answers())
    ans[0]["items"] += [{"subject": {"type": "config", "entryId": "e1", "path": "x.yml"}, "repoId": "r", "unitId": "u1",
                         "status": "confirmed", "directness": "direct", "distance": 2, "viaData": False, "chain": []}]
    ans[1]["groups"] = [{"kind": "k", "key": "k", "reason": "resolution-unverified", "count": 1, "members": [
        {"subject": {"type": "unit", "unitId": "u7"}, "repoId": "r", "unitId": "u7", "status": "possible",
         "reason": "name-only-provider", "directness": "direct", "distance": 1, "viaData": False, "chain": []}]}]
    ans[2]["items"].append(item("o", path="cpg/zz.py"))             # a symbol: not counted
    rep = _run(answers=ans)
    assert rep["baseline"] == {"anchors": 4, "unresolved": 1, "byName": {"resolvedSite": 6, "unresolvedSite": 9},
                               "nonSymbolSystemItems": 2}


def test_report_the_recall_pairs_with_wilson_intervals_the_paired_table_and_the_config_count():
    rc = _run()["recall"]
    assert rc["system"] == N.wilson(9, 10) and rc["baseline"] == N.wilson(8, 10)
    assert rc["systemSym"] == N.wilson(8, 9) and rc["baselineSym"] == N.wilson(7, 9)
    assert rc["paired"] == {"both": 7, "systemOnly": 2, "baselineOnly": 1, "neither": 0} and rc["configGold"] == 1
    assert rc["exKeyError"]["system"] == N.wilson(8, 9) and rc["exKeyError"]["baselineSym"] == N.wilson(6, 8)
    assert rc["flagged"] is True and rc["baselineRecallHigher"] is False
    assert rc["symbolOnly"] == {"flagged": True, "baselineRecallHigher": False}


def test_report_names_the_figure_a_burden_ratio_and_says_so_when_the_baseline_recalls_more():
    assert _run()["reduction"]["kind"] == "burden ratio" and "note" not in _run()["reduction"]
    r = _recall()
    r["baseline"], r["baselineSym"] = [10, 10], [9, 9]
    rep = _run(recall=r)
    assert rep["reduction"]["baselineRecallHigher"] is True and rep["reduction"]["flagged"] is False
    assert "not at equal recall" in rep["reduction"]["note"] and "upper bound" in rep["reduction"]["note"]
    assert rep["recall"]["symbolOnly"] == {"flagged": False, "baselineRecallHigher": True}       # 9/9 > 8/9


def test_the_key_scoped_variant_is_reported_beside_the_headline_with_its_own_numbers():
    key = {a: dict(r, count=r["count"] // 4, countInScope=r["countInScope"] // 4) for a, r in _baseline().items()}   # 50 / 25
    rk = _recall()
    rk["baseline"] = [9, 10]
    rep = _run(baseline_key=key, recall_key=rk)
    ks = rep["keyScoped"]
    assert ks["reduction"]["inScope"] == round(75 / 52, 3) and ks["reduction"]["allListed"] == round(150 / 56, 3)
    assert ks["reduction"]["flagged"] is False and rep["reduction"]["flagged"] is True      # each with its own recall pair
    assert ks["recall"]["baseline"] == N.wilson(9, 10) and ks["baseline"]["unresolved"] == 1
    assert rep["reduction"]["inScope"] == round(300 / 52, 3)                                  # the headline is untouched
    assert ks["reduction"]["interval"] != rep["reduction"]["interval"]
    assert "keyScoped" not in _run()
