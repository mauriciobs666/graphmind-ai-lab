"""P1-36 naive over-lister, snapshot reader, item-level recall, dominance (``docs/plans/connected-code-graph-p1-noise-export-graph.md``
§3-§7). Offline: the oracle is the note's §4.1 table, written here as sets derived from the rules, not read from the code."""
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "eval" / "tools"))
import baseline as B  # noqa: E402
import baseline_helpers as H  # noqa: E402
import noise  # noqa: E402

A_ = H.anchor


def S(s):
    return set(s.split())


def run(anchors, bound=2, **kw):
    return B.naive_over_lister(H.snapshot(), anchors, bound, **kw)


def one(kind, ident, bound=2, **kw):
    return run([A_(kind, ident)], bound, **kw)[0]


# ------------------------------------------------------------------------------------------------ §4.1 table
TABLE = [
    # (kind, id, bound, scope, items, countConfirmedOwn)
    ("symbol", "A", 2, "kind", S("B C E F G H P2 P3 Q"), 5),
    ("symbol", "A", 1, "kind", S("B E H P2 P3 Q"), 4),
    ("symbol", "A", 0, "kind", S("H P2 P3 Q"), 3),
    ("symbol", "A", 3, "kind", S("B C D E F G H P2 P3 Q"), 6),
    ("symbol", "A", 2, "key", S("B C E F G H P3"), 4),
    ("contract", "CP1", 2, "kind", S("A B C E F G H P2 P3 Q R"), 7),
    ("contract", "CP1", 3, "kind", S("A B C D E F G H P2 P3 Q R S"), 9),
    ("contract", "CP1", 2, "key", S("A B C E F G H P3 R"), 6),
    ("config", "CE1", 2, "kind", S("A H P2 P3 Q"), 4),
    ("config", "CE1", 3, "kind", S("A H P2 P3 Q R"), 5),
    ("config", "CE1", 2, "key", S("A H P3"), 3),
    ("symbol", "K", 2, "kind", S("F"), 0),
    ("symbol", "D", 2, "kind", set(), 0),
    ("symbol", "M", 2, "kind", set(), 0),
]


@pytest.mark.parametrize("kind,ident,bound,scope,items,confirmed", TABLE,
                         ids=[f"{k}-{i}-b{b}-{s}" for k, i, b, s, _, _ in TABLE])
def test_worked_graph_rows(kind, ident, bound, scope, items, confirmed):
    r = one(kind, ident, bound, kind_scope=scope)
    assert r["items"] == sorted(items)
    assert r["count"] == len(items) == len(r["items"])
    assert r["countConfirmedOwn"] == confirmed
    assert r["type"] == kind and r["anchorId"] == f"{kind}:{ident}"


def test_in_scope_filter_drops_the_ts_symbols():
    r = one("symbol", "A", 2, in_scope=lambda s: not s["path"].endswith(".ts"))
    assert (r["count"], r["countInScope"]) == (9, 6)
    assert one("symbol", "A", 2)["countInScope"] == 9          # None: everything in scope


def test_byname_counts_items_resting_on_name_rows():
    a = one("symbol", "A")
    assert a["byName"] == {"resolvedSite": 0, "unresolvedSite": 3}       # E F G; B and C also hold a call row
    k = one("symbol", "K")
    assert k["byName"] == {"resolvedSite": 1, "unresolvedSite": 0}       # F reaches K by a name row on a resolved site


def test_byseed_partitions_the_items_and_sums_to_count():
    want = {
        ("symbol", "A", "kind"): dict(kindWideOnly=4, nameOnly=3, providerOnly=0, consumerOrConfigOnly=0, other=2),
        ("contract", "CP1", "kind"): dict(kindWideOnly=2, nameOnly=3, providerOnly=4, consumerOrConfigOnly=2, other=0),
        ("contract", "CP1", "key"): dict(kindWideOnly=0, nameOnly=3, providerOnly=4, consumerOrConfigOnly=2, other=0),
        ("config", "CE1", "kind"): dict(kindWideOnly=4, nameOnly=0, providerOnly=0, consumerOrConfigOnly=1, other=0),
    }
    for (k, i, sc), part in want.items():
        r = one(k, i, 2, kind_scope=sc)
        assert r["bySeed"] == part, (k, i, sc)
        assert sum(r["bySeed"].values()) == r["count"]


def test_byseed_mixed_reach_is_other():
    # R is reached from the consumer seed H; make it also a caller of provider A: reached from two seed classes
    snap = H.snapshot()
    snap["site"].append({"siteId": "sRa", "owner": "R", "unitId": "u1", "path": "r.py", "line": 3, "calleeName": "a"})
    snap["call"].append({"site": "sRa", "callee": "A", "status": "confirmed", "reason": "resolved-call"})
    r = B.naive_over_lister(snap, [A_("contract", "CP1")], 2)[0]
    assert r["bySeed"]["other"] == 1 and r["bySeed"]["consumerOrConfigOnly"] == 1      # R mixed; H alone


def test_confirmed_own_needs_every_call_on_the_chain_confirmed():
    snap = H.snapshot()
    next(c for c in snap["call"] if (c["site"], c["callee"]) == ("sBc", "A"))["status"] = "possible"      # B -> A
    r = B.naive_over_lister(snap, [A_("symbol", "A")], 2)[0]
    assert r["count"] == 9 and r["countConfirmedOwn"] == 3       # B (possible call) and C (chain through B) drop out: H P2 P3 remain


def test_kind_membership_is_not_seeded_from_the_reached_callers():
    snap = H.snapshot()
    snap["provides"].append({"symbolId": "D", "cpId": "CP3", "status": "confirmed", "reason": "r"})     # D (cost 3) provides the mcp tool
    r = B.naive_over_lister(snap, [A_("symbol", "A")], 3)[0]
    assert "D" in r["items"] and "M" not in r["items"]           # A's kind is http only: M (the mcp provider) stays out


def test_byseed_name_only_takes_precedence_over_kind_wide_for_a_member_reached_by_a_name_row():
    snap = H.snapshot()
    snap["site"].append({"siteId": "sP3n", "owner": "P3", "unitId": "u2", "path": "p3.ts", "line": 1, "calleeName": "a"})
    snap["name"].append({"site": "sP3n", "candidate": "A"})       # P3 is in Kind(http) and a name-row caller of A
    r = B.naive_over_lister(snap, [A_("symbol", "A")], 2)[0]
    assert r["bySeed"] == dict(kindWideOnly=3, nameOnly=4, providerOnly=0, consumerOrConfigOnly=0, other=2)


def test_result_is_in_anchorid_order_one_row_per_anchor():
    rows = run([A_("symbol", "K", "z"), A_("symbol", "A", "b"), A_("contract", "CP1", "m")])
    assert [r["anchorId"] for r in rows] == ["b", "m", "z"]


def test_unresolved_anchor_row_is_burden_zero_not_an_error():
    r = run([{"anchorId": "u1", "type": "unresolved"}])[0]
    assert r == {"anchorId": "u1", "type": "unresolved", "count": 0, "countInScope": 0, "countConfirmedOwn": 0,
                 "items": [], "unresolved": True}


# ------------------------------------------------------------------------------------------------ refusals
def test_raises_on_negative_bound_direction_scope_unit_duplicate():
    with pytest.raises(ValueError, match="bound"):
        run([A_("symbol", "A")], -1)
    with pytest.raises(ValueError, match="direction"):
        run([A_("symbol", "A")], 2, direction="depends-on")
    with pytest.raises(ValueError, match="kind_scope"):
        run([A_("symbol", "A")], 2, kind_scope="both")
    with pytest.raises(ValueError, match="unit"):
        run([{"anchorId": "x", "type": "unit", "unitId": "u1"}])
    with pytest.raises(ValueError, match="duplicate"):
        run([A_("symbol", "A", "same"), A_("symbol", "D", "same")])


def test_bound_is_a_required_argument():
    with pytest.raises(TypeError):
        B.naive_over_lister(H.snapshot(), [A_("symbol", "A")])


@pytest.mark.parametrize("anchor", [A_("symbol", "Z"), A_("contract", "CPX"), A_("config", "CEX")])
def test_unknown_id_on_a_resolved_answer_raises_counts_only(anchor):
    with pytest.raises(ValueError) as e:
        run([anchor])
    msg = str(e.value)
    assert "1" in msg and anchor[{"symbol": "symbolId", "contract": "cpId", "config": "entryId"}[anchor["type"]]] not in msg


def test_unknown_anchor_type_and_missing_key_raise():
    with pytest.raises(ValueError):
        run([{"anchorId": "x", "type": "bogus"}])
    with pytest.raises(ValueError):
        run([{"anchorId": "x", "type": "symbol"}])


# ------------------------------------------------------------------------------------------------ properties
def test_shuffled_records_give_the_identical_result():
    base = run([A_("symbol", "A"), A_("contract", "CP1"), A_("config", "CE1")])
    snap = H.snapshot()
    rng = random.Random(7)
    for rows in snap.values():
        rng.shuffle(rows)
    got = B.naive_over_lister(snap, [A_("config", "CE1"), A_("contract", "CP1"), A_("symbol", "A")], 2)
    assert got == base


def test_call_cycle_and_self_call_terminate_and_the_anchor_is_removed():
    snap = H.snapshot()
    snap["site"] += [{"siteId": "sAB", "owner": "A", "unitId": "u1", "path": "a.py", "line": 1, "calleeName": "b"},
                     {"siteId": "sAA", "owner": "A", "unitId": "u1", "path": "a.py", "line": 2, "calleeName": "a"}]
    snap["call"] += [{"site": "sAB", "callee": "B", "status": "confirmed", "reason": "resolved-call"},
                     {"site": "sAA", "callee": "A", "status": "confirmed", "reason": "resolved-call"}]
    r = B.naive_over_lister(snap, [A_("symbol", "A")], 9)[0]
    assert "A" not in r["items"] and {"B", "C", "D"} <= set(r["items"])


def test_a_contract_anchor_symbol_that_provides_and_consumes_takes_cost_zero():
    snap = H.snapshot()
    snap["site"].append({"siteId": "sAk", "owner": "A", "unitId": "u1", "path": "a.py", "line": 4, "calleeName": "x"})
    snap["consumes"].append({"site": "sAk", "cpId": "CP1", "status": "confirmed", "reason": "call"})
    r = B.naive_over_lister(snap, [A_("contract", "CP1")], 2)[0]
    assert r["items"] == sorted(S("A B C E F G H P2 P3 Q R"))      # A at 0, not 1: C, F, G stay inside the bound


def test_the_anchor_does_not_mutate_its_input():
    snap = H.snapshot()
    before = copy.deepcopy(snap)
    B.naive_over_lister(snap, [A_("symbol", "A")], 2)
    assert snap == before


# ------------------------------------------------------------------------------------------------ read_snapshot
def test_read_snapshot_round_trips_the_export(tmp_path):
    p = H.write_snapshot(tmp_path / "s.jsonl")
    snap = B.read_snapshot(p)
    want = H.snapshot()
    for t, rows in want.items():
        assert snap[t] == rows, t
    assert snap["dropped"] == {"call": 0, "name": 0, "consumes": 0, "via": 0}


def _set(t, field, value, which=0):
    def mutate(recs):
        seen = -1
        for tt, r in recs:
            if tt == t:
                seen += 1
                if seen == which:
                    r[field] = value
        return recs
    return mutate


@pytest.mark.parametrize("t,field", [
    ("site", "owner"), ("call", "callee"), ("name", "candidate"), ("provides", "symbolId"), ("provides", "cpId"),
    ("consumes", "cpId"), ("link", "cpId"), ("via", "linkId"), ("bound", "linkId"), ("bound", "entryId")])
def test_closure_break_raises_naming_the_record_type_and_a_count(tmp_path, t, field):
    p = H.write_snapshot(tmp_path / "s.jsonl", _set(t, field, "nope"))
    with pytest.raises(ValueError) as e:
        B.read_snapshot(p)
    assert t in str(e.value) and "1" in str(e.value) and "nope" not in str(e.value)


def test_ownerless_site_rows_are_dropped_and_counted_not_an_error(tmp_path):
    def drop(recs):
        return [(t, r) for t, r in recs if not (t == "site" and r["siteId"] in ("sBc", "sQk"))]
    snap = B.read_snapshot(H.write_snapshot(tmp_path / "s.jsonl", drop))
    # sBc carries one call and one name; sQk one consumes and two via rows
    assert snap["dropped"] == {"call": 1, "name": 1, "consumes": 1, "via": 2}
    assert all(r["site"] != "sBc" for r in snap["call"] + snap["name"])
    assert all(r["site"] != "sQk" for r in snap["consumes"] + snap["via"])


def test_duplicate_key_raises(tmp_path):
    def dup(recs):
        i = next(i for i, (t, _) in enumerate(recs) if t == "symbol")
        recs.insert(i, copy.deepcopy(recs[i]))
        return recs
    with pytest.raises(ValueError, match="symbol"):
        B.read_snapshot(H.write_snapshot(tmp_path / "s.jsonl", dup))


def test_composite_keys_are_part_of_the_duplicate_rule(tmp_path):
    def dup(recs):
        i = next(i for i, (t, _) in enumerate(recs) if t == "bound")
        recs.append(copy.deepcopy(recs[i]))
        return recs
    with pytest.raises(ValueError, match="bound"):
        B.read_snapshot(H.write_snapshot(tmp_path / "s.jsonl", dup))
    def distinct(recs):
        recs.append(("bound", {"linkId": "L1", "entryId": "CE1", "siteId": "sHk", "ord": 1}))
    B.read_snapshot(H.write_snapshot(tmp_path / "s2.jsonl", distinct))


@pytest.mark.parametrize("edit,why", [
    (_set("meta", "exportVersion", 2), "exportVersion"),
    (_set("meta", "graphModelVersion", "1.0"), "graphModelVersion"),
])
def test_meta_gate(tmp_path, edit, why):
    with pytest.raises(ValueError, match=why):
        B.read_snapshot(H.write_snapshot(tmp_path / "s.jsonl", edit))


def test_meta_must_come_first_and_alone(tmp_path):
    with pytest.raises(ValueError, match="meta"):
        B.read_snapshot(H.write_snapshot(tmp_path / "a.jsonl", lambda r: r[1:] + r[:1]))
    with pytest.raises(ValueError, match="meta"):
        B.read_snapshot(H.write_snapshot(tmp_path / "b.jsonl", lambda r: r + [r[0]]))


def test_missing_field_and_unknown_type_raise(tmp_path):
    def miss(recs):
        for t, r in recs:
            if t == "cp":
                del r["keyNorm"]
                break
    with pytest.raises(ValueError, match="cp"):
        B.read_snapshot(H.write_snapshot(tmp_path / "a.jsonl", miss))
    with pytest.raises(ValueError, match="bogus"):
        B.read_snapshot(H.write_snapshot(tmp_path / "b.jsonl", lambda r: r + [("bogus", {"x": 1})]))


def test_optional_fields_may_be_absent_and_a_read_snapshot_feeds_the_over_lister(tmp_path):
    snap = B.read_snapshot(H.write_snapshot(tmp_path / "s.jsonl"))
    r = B.naive_over_lister(snap, [A_("symbol", "A")], 2)[0]
    assert r["count"] == 9


def test_an_odd_length_owned_regions_list_is_refused_not_silently_truncated():
    snap = H.snapshot()
    next(x for x in snap["symbol"] if x["symbolId"] == "E")["ownedRegions"] = [7, 9, 11]
    with pytest.raises(ValueError, match="ownedRegions"):
        B.baseline_recall(snap, B.naive_over_lister(snap, [A_("symbol", "A", "a")], 2), [_gi("a", "e.py", "e.py::E", [8])])


# ------------------------------------------------------------------------------------------------ item-set hash (N2)
def _gi(anchor_id, path, qname, lines, **kw):
    return {"anchorId": anchor_id, "impacted": {"type": "symbol", "repoId": "r1", "path": path, "qname": qname},
            "lines": lines, "hit": kw.get("hit", False), "keyError": kw.get("keyError", False)}


def test_item_set_sha256_is_over_sorted_code_point_lines_without_trailing_newline():
    items = [_gi("b", "p.py", "p.py::f", [9, 10, 2]), _gi("a", "q.py", "q.py::g", [5]),
             {"anchorId": "a", "impacted": {"type": "config", "repoId": "r1", "path": "c.yml", "keyPath": "k.v"}, "lines": [1],
              "hit": False, "keyError": False}]
    text = "\n".join(sorted(["b|symbol:r1:p.py:p.py::f|2,9,10", "a|symbol:r1:q.py:q.py::g|5", "a|config:r1:c.yml:k.v|1"]))
    assert B.itemSetSha256(items) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert B.itemSetSha256(list(reversed(items))) == B.itemSetSha256(items)


def test_item_set_sha256_distinguishes_lines_and_orders_by_code_point_not_locale():
    assert B.itemSetSha256([_gi("a", "p", "q", [1])]) != B.itemSetSha256([_gi("a", "p", "q", [2])])
    assert B.itemSetSha256([_gi("a", "p", "q", [10, 9])]) == B.itemSetSha256([_gi("a", "p", "q", [9, 10])])   # numeric sort of lines
    up, lo = _gi("B", "p", "q", [1]), _gi("a", "p", "q", [1])
    text = "\n".join(["B|symbol:r1:p:q|1", "a|symbol:r1:p:q|1"])         # 'B' (66) before 'a' (97)
    assert B.itemSetSha256([lo, up]) == hashlib.sha256(text.encode()).hexdigest()


def test_item_set_sha256_of_nothing_is_the_empty_hash():
    assert B.itemSetSha256([]) == hashlib.sha256(b"").hexdigest()


# ------------------------------------------------------------------------------------------------ §6 recall predicate
def _rows_for(*anchors, bound=2, **kw):
    return B.naive_over_lister(H.snapshot(), list(anchors), bound, **kw)


def test_owned_regions_are_paired_before_the_h1_credit():
    # E: span 1-3, ownedRegions [7, 9] flat; gold line 8 hits the owned pair, line 6 hits nothing
    base = _rows_for(A_("symbol", "A", "a"))
    hit = _gi("a", "e.py", "e.py::E", [8])
    miss = _gi("a", "e.py", "e.py::E", [6])
    r = B.baseline_recall(H.snapshot(), base, [hit, miss])
    assert r["baseline"] == [1, 2]


def test_credit_is_per_anchor_never_pooled():
    base = _rows_for(A_("symbol", "A", "a"), A_("symbol", "D", "d"))          # D lists nothing; E is only A's
    r = B.baseline_recall(H.snapshot(), base, [_gi("a", "e.py", "e.py::E", [2]), _gi("d", "e.py", "e.py::E", [2])])
    assert r["baseline"] == [1, 2]


def test_a_gold_anchor_without_a_baseline_row_raises():
    with pytest.raises(ValueError, match="anchor"):
        B.baseline_recall(H.snapshot(), _rows_for(A_("symbol", "A", "a")), [_gi("zz", "e.py", "e.py::E", [2])])


def test_config_gold_is_never_a_hit_and_is_counted():
    base = _rows_for(A_("symbol", "A", "a"))
    cfg = {"anchorId": "a", "impacted": {"type": "config", "repoId": "r1", "path": "e.py", "keyPath": "x"}, "lines": [2],
           "hit": True, "keyError": False}
    r = B.baseline_recall(H.snapshot(), base, [cfg, _gi("a", "e.py", "e.py::E", [2], hit=True)])
    assert r["configGold"] == 1
    assert r["baseline"] == [1, 2] and r["baselineSym"] == [1, 1]
    assert r["system"] == [2, 2] and r["systemSym"] == [1, 1]


def test_paired_two_by_two_and_exkeyerror_and_flags():
    base = _rows_for(A_("symbol", "A", "a"))
    items = [_gi("a", "e.py", "e.py::E", [2], hit=True),                 # both
             _gi("a", "b.py", "b.py::B", [2], hit=False),                 # baselineOnly
             _gi("a", "zz.py", "zz.py::Z", [2], hit=True),                # systemOnly
             _gi("a", "yy.py", "yy.py::Y", [2], hit=False, keyError=True)]  # neither, key error
    r = B.baseline_recall(H.snapshot(), base, items)
    assert r["paired"] == {"both": 1, "baselineOnly": 1, "systemOnly": 1, "neither": 1}
    assert r["system"] == [2, 4] and r["baseline"] == [2, 4]
    assert r["itemSetSha256"] == B.itemSetSha256(items)
    ex = r["exKeyError"]
    assert ex["system"] == [2, 3] and ex["baseline"] == [2, 3]
    assert ex["itemSetSha256"] == B.itemSetSha256(items[:3]) != r["itemSetSha256"]


@pytest.mark.parametrize("sys_k,base_k,reduction,baseline_more", [
    (3, 1, True, False),        # baseline recalls less: flagged
    (1, 3, False, True),        # baseline recalls more: reported, not flagged
    (2, 2, False, False),
])
def test_flag_is_exact_cross_multiplication_in_both_directions(sys_k, base_k, reduction, baseline_more):
    # n_sys == n_base == 4 would hide a swapped operand: use different n through excluded key errors is not needed;
    # the cross-multiplied form is checked directly on the figures
    f = B.recall_flags({"system": [sys_k, 4], "baseline": [base_k, 4]})
    assert (f["reduction"], f["baselineRecallsMore"]) == (reduction, baseline_more)


def test_flag_uses_cross_multiplication_with_unequal_denominators():
    # 1/2 vs 2/5: baseline_k*system_n = 1*5 = 5 < system_k*baseline_n = 2*2 = 4 is False; naive rate compare would say 0.5 > 0.4
    f = B.recall_flags({"system": [2, 5], "baseline": [1, 2]})
    assert f["reduction"] is False and f["baselineRecallsMore"] is True
    f = B.recall_flags({"system": [3, 5], "baseline": [1, 2]})        # 0.6 vs 0.5: 1*5 < 3*2
    assert f["reduction"] is True


# ------------------------------------------------------------------------------------------------ dominance (M1, N3)
def _si(anchor_id, sym, *edges):
    return {"anchorId": anchor_id, "symbolId": sym, "edges": list(edges)}


def test_dominance_splits_class_a_and_b_and_counts_absent_items_as_violations():
    base = run([A_("symbol", "A", "a")])        # items B C E F G H P2 P3 Q
    sys_items = [_si("a", "B", "CALLS"), _si("a", "C", "CALLS", "CALLS"), _si("a", "D", "CALLS"),   # D: A violation
                 _si("a", "H", "DATA"), _si("a", "S", "->", "CALLS"), _si("a", "R", "HAS_PARAM"),   # B: S, R violate
                 _si("a", "P3", "CALLS", "DATA", "->")]                                              # first excluded: DATA
    d = B.dominance(sys_items, base)
    assert d["A"] == {"checked": 3, "violations": 1}
    assert d["B"]["checked"] == 4 and d["B"]["violations"] == 2
    assert d["B"]["byEdge"] == {"DATA": {"checked": 2, "violations": 0}, "->": {"checked": 1, "violations": 1},
                                "HAS_PARAM": {"checked": 1, "violations": 1}}


@pytest.mark.parametrize("edge", sorted(B.EXCLUDED_EDGES))
def test_every_excluded_edge_type_makes_class_b(edge):
    d = B.dominance([_si("a", "B", edge)], run([A_("symbol", "A", "a")]))
    assert d["A"]["checked"] == 0 and d["B"]["checked"] == 1


def test_excluded_edge_set_is_the_documented_six():
    assert B.EXCLUDED_EDGES == {"->", "DATA", "CONTROLLED_BY", "HAS_PARAM", "HAS_RETURN", "HAS_COND"}


def test_dominance_is_per_anchor_and_refuses_an_unknown_anchor():
    base = run([A_("symbol", "A", "a"), A_("symbol", "D", "d")])
    d = B.dominance([_si("d", "B", "CALLS")], base)               # B is A's item, not D's
    assert d["A"] == {"checked": 1, "violations": 1}
    with pytest.raises(ValueError, match="anchor"):
        B.dominance([_si("nope", "B", "CALLS")], base)


# ------------------------------------------------------------------------------------------------ Q-NC5 view (§5)
def test_nc5_view_projects_links_and_providers_and_agrees_with_the_noise_counter():
    v = B.nc5_view(H.snapshot())
    assert sorted((l["keyNorm"], l["bound"], l["status"]) for l in v["links"]) == [
        ("GET /shop/api/x", False, "possible"), ("GET /shop/api/x", True, "confirmed"), ("GET /shop/api/y", False, "possible")]
    assert sorted((p["keyNorm"], p["unitId"]) for p in v["providers"]) == [
        ("GET /shop/api/x", "u1"), ("GET /shop/api/x", "u2"), ("GET /shop/api/y", "u1"), ("tool", "u1")]
    assert noise.unbound_multi_unit(v) == {"unboundMultiUnit": 1, "unboundMultiUnitConfirmed": 0}      # L3 only


def test_nc5_view_refuses_one_key_under_two_kinds():
    snap = H.snapshot()
    snap["cp"].append({"cpId": "CP9", "kind": "grpc", "keyNorm": "GET /shop/api/x"})
    with pytest.raises(ValueError, match="kind"):
        B.nc5_view(snap)


def test_nc5_view_skips_links_without_a_contract_point():
    snap = H.snapshot()
    snap["link"].append({"linkId": "L9", "kind": "http", "status": "possible", "reason": "r", "cpId": None, "provider": None})
    assert len(B.nc5_view(snap)["links"]) == 3
