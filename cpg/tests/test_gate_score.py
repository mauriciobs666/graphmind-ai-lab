"""P1-32b gate harness, C5 assertions and the hit rule H1-H4 (EP §5.2, §5.2.1 HA-1..HA-5). Offline, synthetic, in-memory gate snapshot.

Every refusal has a planted input that must turn it red, with the expected outcome written here and not taken from the engine."""
import copy

import pytest

import gate_helpers as H
from gate_helpers import G


@pytest.fixture
def w():
    return H.World()


def hit(o):
    return o["outcome"] == "hit"


# ---------------------------------------------------------------- positive control and H1
def test_baseline_is_a_hit_with_chain_length(w):
    o, res = w.outcome()
    assert hit(o) and o["hops"] == 4 and o["interFunctionHops"] == 1
    assert res["defects"] == {} and res["harnessErrors"] == {} and res["stops"] == {}


def test_h1_wrong_qname_is_a_miss(w):
    other = w.snap.add_symbol("4" * 32, "svc/other.py::g", 5, 20)
    sub = H.sym_subject(other, "svc/other.py::g", 5, 20)
    chain = w.chain()[:-1] + [H.hopn("Symbol", other, "HAS_CALL", "<-")]
    w.snap.add_edge("HAS_CALL", ("Symbol", other), ("CallSite", w.cs))
    o, _ = w.outcome(w.answer([H.ans_item(sub, chain)]))
    assert o["outcome"] == "miss" and o["stage"] == "none"


def test_h1_right_qname_span_off_the_line_is_a_miss_no_id_only_branch():
    w = H.World(lines=(30,), span=(5, 20))                  # the symbol is e by qname, but no line of L lies in its span
    o, _ = w.outcome()
    assert o["outcome"] == "miss" and o["stage"] == "none"


def test_h1_owned_region_credits_a_line_outside_the_span():
    w = H.World(lines=(30,), span=(5, 20))
    w.snap.symbols[w.sid]["ownedRegions"] = [[28, 32]]
    o, _ = w.outcome(w.answer([ans := H.ans_item(H.sym_subject(w.sid, w.qname, 5, 20, ownedRegions=[[28, 32]]), w.chain())]))
    assert hit(o)


def test_h1_eligible_ancestor_credits_a_duplicate_suffix_symbol():
    w = H.World(qname="svc/c.py::C.x#2.check")
    w.rec = H.item(1, w.anchor, H.sym("svc/c.py::C.x.check"), lines=(10,))      # the key holds E(qname)
    o, _ = w.outcome()
    assert hit(o)


def test_h1_anon_symbol_is_credited_to_its_enclosing_eligible_function():
    w = H.World(qname="svc/c.py::run.<anon>#1")
    w.rec = H.item(1, w.anchor, H.sym("svc/c.py::run"), lines=(10,))
    assert hit(w.outcome()[0])


# ---------------------------------------------------------------- H2
@pytest.mark.parametrize("status", ["dismissed", "stale", None])
def test_h2_status_other_than_confirmed_or_possible_is_a_miss(w, status):
    it = w.item()
    it["status"] = status
    o, _ = w.outcome(w.answer([it]))
    assert o["outcome"] == "miss" and o["stage"] == "H1"


def test_h2_possible_counts(w):
    assert hit(w.outcome(w.answer([w.item(status="possible")]))[0])


# ---------------------------------------------------------------- H3 shape
def test_h3_empty_chain_is_a_miss(w):
    it = w.item()
    it["chain"] = []
    assert w.outcome(w.answer([it]))[0]["outcome"] == "miss"


def test_h3_first_hop_not_the_asked_anchor_is_a_miss(w):
    it = w.item()
    it["chain"][0] = H.hop0("ContractPoint", "9" * 32)
    w.snap.add_node("ContractPoint", "9" * 32)
    assert w.outcome(w.answer([it]))[0]["outcome"] == "miss"


def test_h3_last_hop_not_the_item_is_a_miss(w):
    it = w.item()
    it["chain"] = it["chain"][:-1]
    assert w.outcome(w.answer([it]))[0]["outcome"] == "miss"


# ---------------------------------------------------------------- H4
def test_h4_group_member_is_enumerated_individually_and_hits(w):
    assert hit(w.outcome(w.answer([], groups=[H.group([w.item(status="possible")])]))[0])


def test_h4_a_group_header_with_a_count_only_is_never_x(w):
    g = H.group([])
    g["count"] = 1
    assert w.outcome(w.answer([], groups=[g]))[0]["outcome"] == "miss"


# ---------------------------------------------------------------- HA-1 chain replay, planted inputs (a)-(h)
def test_ha1a_middle_hop_id_absent_from_the_snapshot_is_a_miss(w):
    it = w.item()
    it["chain"][2]["id"] = "f" * 32
    o, res = w.outcome(w.answer([it]))
    assert o["outcome"] == "miss" and o["stage"] == "H2" and res["harnessErrors"] == {}


def test_ha1b_middle_hop_real_node_with_no_edge_to_either_neighbour_is_a_miss(w):
    stray = w.snap.add_node("CallSite", "5" * 32)
    it = w.item()
    it["chain"][2]["id"] = stray
    assert w.outcome(w.answer([it]))[0]["outcome"] == "miss"


def test_ha1c_reversed_dir_is_a_miss(w):
    it = w.item()
    it["chain"][1]["dir"] = "->"
    assert w.outcome(w.answer([it]))[0]["outcome"] == "miss"


def test_ha1d_edgetype_that_does_not_join_the_pair_is_a_miss(w):
    it = w.item()
    it["chain"][2]["edgeType"] = "CONSUMES"         # allow-listed, kind pair (CallSite, ContractPoint) is not (CallSite, Link)
    assert w.outcome(w.answer([it]))[0]["outcome"] == "miss"


def test_ha1d2_allowlisted_pair_type_with_no_stored_edge_is_a_miss(w):
    w.snap.edges.discard(("VIA", ("CallSite", w.cs), ("Link", w.link)))
    assert w.outcome()[0]["outcome"] == "miss"


@pytest.mark.parametrize("bad", ["CALLS; MATCH (n) DETACH DELETE n", "calls", "TO_UNIT", ""])
def test_ha1e_edgetype_outside_the_allowlist_is_a_harness_error_and_never_queried(w, bad):
    it = w.item()
    it["chain"][1]["edgeType"] = bad
    before = w.snap.queries
    o, res = w.outcome(w.answer([it]))
    assert o["outcome"] == "error" and res["harnessErrors"] == {"edgeType-not-allowed": 1}
    assert w.snap.queries == before                      # rejected before any query


def test_ha1e_bad_dir_is_a_harness_error(w):
    it = w.item()
    it["chain"][1]["dir"] = "=>"
    assert w.outcome(w.answer([it]))[1]["harnessErrors"] == {"dir": 1}


def test_ha1f_name_matched_call_via_calls_name_resolves_to_is_a_hit():
    w = H.World()
    cs2, callee = "6" * 32, "7" * 32
    w.snap.add_node("CallSite", cs2)
    w.snap.add_symbol(callee, w.qname, 5, 20)             # callee symbol is e (same qname, other id): E equal, span holds the line
    w.snap.add_edge("CALLS_NAME", ("CallSite", cs2), ("Symbol", callee))
    chain = w.chain()[:3] + [H.hopn("CallSite", cs2, "HAS_CALL", "<-")]
    w.snap.add_edge("HAS_CALL", ("Symbol", w.sid), ("CallSite", cs2))
    chain = w.chain()[:-1] + [H.hopn("Symbol", w.sid, "HAS_CALL", "<-", ), H.hopn("CallSite", cs2, "HAS_CALL", "->"),
                              H.hopn("Symbol", callee, "CALLS_NAME", "->", status="possible", reason="name-match")]
    it = H.ans_item(H.sym_subject(callee, w.qname, 5, 20), chain, status="possible")
    assert hit(w.outcome(w.answer([it]))[0])


def test_ha1g_class_d_row_against_a_snapshot_with_no_such_edges_is_a_harness_error(w):
    it = w.item()
    it["chain"][2]["edgeType"] = "READS"
    o, res = w.outcome(w.answer([it]))
    assert o["outcome"] == "error" and res["harnessErrors"] == {"class-D-not-loaded": 1}


def test_ha1g_class_d_row_is_replayed_once_the_snapshot_loads_it(w):
    w.snap.add_edge("READS", ("CallSite", w.cs), ("ContractPoint", w.cp))
    it = w.item()
    it["chain"] = [H.hop0("ContractPoint", w.cp), H.hopn("CallSite", w.cs, "READS", "<-"), H.hopn("Symbol", w.sid, "HAS_CALL", "<-")]
    o, res = w.outcome(w.answer([it]))
    assert hit(o) and res["harnessErrors"] == {}


@pytest.mark.parametrize("etype", ["LAUNCHES", "DEFINED_BY", "DEPENDS_ON", "FROM_DEP"])
def test_ha1h_class_x_row_is_a_harness_error(w, etype):
    it = w.item()
    it["chain"][1]["edgeType"] = etype
    assert w.outcome(w.answer([it]))[1]["harnessErrors"] == {"class-X": 1}


def test_ha1_fabricated_chain_item_is_a_miss_positive_control():
    """A fabricated chain between real nodes (no edge joins them) scores a miss, however plausible its shape."""
    w = H.World()
    fab = [H.hop0("ContractPoint", w.cp), H.hopn("Symbol", w.sid, "PROVIDES", "<-")]    # allow-listed pair, but no PROVIDES edge stored
    it = H.ans_item(H.sym_subject(w.sid, w.qname, 5, 20), fab)
    o, _ = w.outcome(w.answer([it]))
    assert o["outcome"] == "miss"


def test_ha1_replay_function_directly():
    w = H.World()
    assert G.replay_chain(w.chain(), w.snap) == ("ok", "")
    assert G.replay_chain([], w.snap)[0] == "fail"
    assert G.replay_chain([H.hop0("Unit", "x")], w.snap) == ("error", "hop-kind")


def test_edge_allowlist_is_the_21_types_of_the_spec():
    assert len(G.EDGES) == 21
    assert {t for t, (c, _) in G.EDGES.items() if c == "X"} == {"DEFINED_BY", "LAUNCHES", "DEPENDS_ON", "FROM_DEP"}
    assert {t for t, (c, _) in G.EDGES.items() if c == "D"} == {"READS", "WRITES", "BOUND_BY", "RAN"}


# ---------------------------------------------------------------- HA-2 subject read from the snapshot
def test_ha2_moved_lines_list_a_mismatch_while_h1_decides_on_the_snapshot_span(w):
    it = w.item()
    it["subject"]["lineStart"], it["subject"]["lineEnd"] = 100, 110            # answer copy moved off the recorded line
    o, res = w.outcome(w.answer([it]))
    assert hit(o) and res["defects"] == {"subject-lineStart": 1, "subject-lineEnd": 1}


def test_ha2_altered_qname_lists_a_mismatch_and_h1_still_uses_the_snapshot(w):
    it = w.item()
    it["subject"]["qname"] = "svc/other.py::zzz"
    o, res = w.outcome(w.answer([it]))
    assert hit(o) and res["defects"] == {"subject-qname": 1}


def test_ha2_answer_span_that_would_credit_is_not_trusted():
    w = H.World(lines=(30,), span=(5, 20))
    it = w.item()
    it["subject"]["lineStart"], it["subject"]["lineEnd"] = 25, 35               # the answer's copy would hold line 30
    o, res = w.outcome(w.answer([it]))
    assert o["outcome"] == "miss" and res["defects"]["subject-lineStart"] == 1


def test_ha2_subject_absent_from_the_snapshot_is_a_listed_defect_and_a_miss(w):
    it = w.item()
    it["subject"]["symbolId"] = "e" * 32
    o, res = w.outcome(w.answer([it]))
    assert o["outcome"] == "miss" and res["defects"] == {"subject-not-in-snapshot": 1}


# ---------------------------------------------------------------- HA-3 request echo
def _echo(w, mutate):
    ans = w.answer()
    mutate(ans["interpretation"])
    return w.outcome(ans)


def test_ha3_second_anchor_added_is_a_miss_and_a_listed_defect(w):
    o, res = _echo(w, lambda i: i["request"]["anchors"].append({"type": "contract", "cpId": "8" * 32, "kind": "http", "keyNorm": "x"}))
    assert o["outcome"] == "miss" and o["reason"] == "request-unequal" and res["defects"] == {"request-echo": 1}


def test_ha3_scope_present_is_a_miss(w):
    o, res = _echo(w, lambda i: i["request"].update(scope={"repoIds": ["r"]}))
    assert o["reason"] == "request-unequal" and res["defects"] == {"request-echo": 1}


def test_ha3_direction_flipped_is_a_miss(w):
    o, _ = _echo(w, lambda i: i["request"].update(direction="depends-on"))
    assert o["reason"] == "request-unequal"


def test_ha3_ambiguous_status_is_a_miss_reported_with_lookups_and_no_defect(w):
    ans = w.answer(status="ambiguous")
    o, res = w.outcome(ans)
    assert o["outcome"] == "miss" and o["reason"] == "not-resolved" and res["defects"] == {}


# ---------------------------------------------------------------- HA-4 stamp
@pytest.mark.parametrize("mutate,field", [
    (lambda a: a["stamp"].update(snapshotId="a" * 31 + "b"), "snapshotId"),
    (lambda a: a["stamp"].update(verdictLogOffset=1), "verdictLogOffset"),
    (lambda a: a["stamp"].update(verdictLogHash=G.EMPTY_VERDICT_LOG_HASH[:-1] + "0"), "verdictLogHash"),
    (lambda a: a["stamp"].update(verdictLogOffset=False), "verdictLogOffset")])
def test_ha4_each_stamp_failure_voids_the_answer_and_stops_the_run(w, mutate, field):
    ans = w.answer()
    mutate(ans)
    o, res = w.outcome(ans)
    assert o["outcome"] == "void" and res["stops"] == {f"C5:stamp:{field}": 1}


def test_ha4_the_hash_is_the_literal_not_the_engines_constant():
    from ccg import answer as A
    assert G.EMPTY_VERDICT_LOG_HASH == "c482b1cf54631b37eb4572dc86a68fb1" == A.EMPTY_VERDICT_LOG_HASH


def test_ha4_stamp_is_checked_against_the_pinned_id_read_from_the_snapshot(w):
    assert w.outcome(snapshot_id="b" * 32)[0]["outcome"] == "void"


# ---------------------------------------------------------------- HA-5 masking (config)
def _stored(raw):
    from ccg.assemble import rows
    return rows._mask_key_path(raw)                                                  # noqa: SLF001 - the producer's own masking


def _cfg_world(answer_key, gold_key):
    snap = H.DictSnapshot()
    cp = "c" * 32
    snap.add_node("ContractPoint", cp)
    entry = snap.add_entry("e" * 32, "deploy/compose.yaml", answer_key)
    rec = H.item(1, {"type": "contract", "kind": "http", "keyNorm": "GET /a"}, H.cfg("deploy/compose.yaml", gold_key), lines=(3,))
    rec["anchorId"] = cp
    chain = [H.hop0("ContractPoint", cp), H.hopn("ConfigEntry", entry, "WRITES", "<-")]
    snap.add_edge("WRITES", ("ConfigEntry", entry), ("ContractPoint", cp))
    subj = {"type": "config", "entryId": entry, "path": "deploy/compose.yaml", "line": 3, "keyPath": answer_key}
    ans = H.answer(H.request([cp]), [H.ans_item(subj, chain)])
    run = {"snapshotId": H.SNAP, "answers": {cp: {"request": H.request([cp]), "answer": ans}}}
    return G.score_gate([rec], run, snap, ())


def test_ha5_config_prefix_rule_without_masking_is_listed_nowhere():
    res = _cfg_world("services.web.ports[0]", "services.web")
    assert res["items"][next(iter(res["items"]))]["outcome"] == "hit" and res["maskedComparisons"] == 0


def test_ha5_failing_segments_mask_to_the_same_string_and_are_listed():
    # a 24-character run fails the token screen (T2): two different keys under one parent mask to the same string, and the comparison is listed
    a, b = "services." + "a" * 24, "services." + "b" * 24
    assert G.mask_keypath(a) == G.mask_keypath(b) == "services.[#]"
    res = _cfg_world(a, b)
    assert res["items"][next(iter(res["items"]))]["outcome"] == "hit" and res["maskedComparisons"] == 1


def test_ha5_a_listed_masked_comparison_blocks_the_verdict():
    d = G.decide_gate({"items": {"i": {"outcome": "hit", "stratum": "http", "counting": True, "anchorId": "a"}}, "defects": {},
                       "harnessErrors": {}, "maskedComparisons": 2, "stops": {}})
    assert d["verdict"] == "blocked" and "masked-comparisons" in d["blockers"]


# ---------------------------------------------------------------- tests added from the mutation table
def test_ha1_each_hop_node_is_checked_for_existence_even_where_an_edge_is_recorded(w):
    """The spec lists the existence check separately; the in-memory snapshot can hold an edge whose end node is gone."""
    for idx in range(4):
        snap = H.World().snap
        label, nid = [(h["kind"], h["id"]) for h in w.chain()][idx]
        snap.nodes.discard((label, nid))
        assert G.replay_chain(w.chain(), snap) == ("fail", "node-absent"), idx


def test_ha1_a_stored_edge_between_a_kind_pair_the_type_cannot_join_still_fails():
    w = H.World()
    w.snap.add_edge("CONSUMES", ("CallSite", w.cs), ("Link", w.link))              # CONSUMES joins CallSite to ContractPoint, not to Link
    chain = w.chain()
    chain[2]["edgeType"] = "CONSUMES"
    assert G.replay_chain(chain, w.snap) == ("fail", "kind-pair")


def test_h3_first_hop_must_be_the_asked_anchor_even_when_the_chain_is_otherwise_real():
    w = H.World()
    cp2, link2 = "8" * 32, "9" * 32
    w.snap.add_node("ContractPoint", cp2)
    w.snap.add_node("Link", link2)
    w.snap.add_edge("ON", ("Link", link2), ("ContractPoint", cp2))
    w.snap.add_edge("VIA", ("CallSite", w.cs), ("Link", link2))
    chain = [H.hop0("ContractPoint", cp2), H.hopn("Link", link2, "ON", "<-"), H.hopn("CallSite", w.cs, "VIA", "<-"), H.hopn("Symbol", w.sid, "HAS_CALL", "<-")]
    assert G.replay_chain(chain, w.snap) == ("ok", "")                              # a real chain, but from another anchor
    o, _ = w.outcome(w.answer([H.ans_item(H.sym_subject(w.sid, w.qname, *w.span), chain)]))
    assert o["outcome"] == "miss"


def test_h3_last_hop_must_be_the_item_even_when_the_chain_is_otherwise_real():
    w = H.World()
    other = w.snap.add_symbol("4" * 32, "svc/other.py::g", 5, 20)
    w.snap.add_edge("HAS_CALL", ("Symbol", other), ("CallSite", w.cs))
    chain = w.chain()[:-1] + [H.hopn("Symbol", other, "HAS_CALL", "<-")]
    o, _ = w.outcome(w.answer([H.ans_item(H.sym_subject(w.sid, w.qname, *w.span), chain)]))     # the item is w.sid, the chain ends at another symbol
    assert o["outcome"] == "miss"


def test_h4_a_group_header_carrying_a_subject_and_a_chain_is_still_never_x(w):
    g = H.group([])
    g.update(count=1, subject=H.sym_subject(w.sid, w.qname, *w.span), chain=w.chain(), status="possible")
    assert w.outcome(w.answer([], groups=[g]))[0]["outcome"] == "miss"


def test_ha3_use_case_must_equal_the_submitted_one(w):
    o, res = _echo(w, lambda i: i["request"].update(useCase="other"))
    assert o["reason"] == "request-unequal" and res["defects"] == {"request-echo": 1}


@pytest.mark.parametrize("answer_key,hit_expected", [
    ("services.web[0]", True), ("services.web.ports", True), ("services.webapp", False), ("services.web", True), ("services.we", False)])
def test_h1_config_keypath_equal_or_prefix_at_a_segment_boundary(answer_key, hit_expected):
    res = _cfg_world(answer_key, "services.web")
    assert (res["items"][next(iter(res["items"]))]["outcome"] == "hit") is hit_expected


def test_h1_config_in_another_file_is_a_miss():
    snap = H.DictSnapshot()
    cp = "c" * 32
    snap.add_node("ContractPoint", cp)
    entry = snap.add_entry("e" * 32, "deploy/other.yaml", "services.web")
    rec = H.item(1, H.contract("http", "GET /a"), H.cfg("deploy/compose.yaml", "services.web"), lines=(3,))
    rec["anchorId"] = cp
    snap.add_edge("WRITES", ("ConfigEntry", entry), ("ContractPoint", cp))
    subj = {"type": "config", "entryId": entry, "path": "deploy/other.yaml", "line": 3, "keyPath": "services.web"}
    chain = [H.hop0("ContractPoint", cp), H.hopn("ConfigEntry", entry, "WRITES", "<-")]
    ans = H.answer(H.request([cp]), [H.ans_item(subj, chain)])
    res = G.score_gate([rec], {"snapshotId": H.SNAP, "answers": {cp: {"request": H.request([cp]), "answer": ans}}}, snap, ())
    assert res["items"][rec["itemId"]]["outcome"] == "miss"


def test_ha5_listing_when_the_store_holds_the_masked_form_and_the_gold_is_raw():
    # round C: the store never holds the answer-side `[#]` form, only the loader's `[#…]`; these two cases are the realistic replacement of the
    # earlier `[#]` fixtures
    a = "services." + "a" * 24
    res = _cfg_world(_stored(a), a)
    assert res["items"][next(iter(res["items"]))]["outcome"] == "hit" and res["maskedComparisons"] == 1


def test_ha5_listing_when_both_sides_are_already_the_stored_masked_form():
    s = _stored("services." + "b" * 24)
    res = _cfg_world(s, s)
    assert res["items"][next(iter(res["items"]))]["outcome"] == "hit" and res["maskedComparisons"] == 1


# ---------------------------------------------------------------- B4: the store's flat ownedRegions (the loader's shape)
def _flat_store(owned, qname="svc/c.py::call", span=(5, 20)):
    sid = "3" * 32
    props = H.loader_symbol_props(sid, qname, *span, owned=owned)
    return sid, props, G.StoreSnapshot(H.LoaderRunner([props]))


def test_b4_the_loader_writes_ownedRegions_flat_and_the_snapshot_hands_match_keys_pairs():
    sid, props, snap = _flat_store([[28, 32], [40, 41]])
    assert props["ownedRegions"] == [28, 32, 40, 41]                       # the producer's shape, not a hand-written one
    assert snap.symbol(sid)["ownedRegions"] == [[28, 32], [40, 41]]
    assert snap.symbols_of_path("svc/c.py")[0]["ownedRegions"] == [[28, 32], [40, 41]]


def _split(store, world_snap):
    """Symbols from the loader-fed ``StoreSnapshot``, everything else from the dict world."""
    class Both:
        def __getattr__(self, n):
            return getattr(store if n in ("symbol", "symbols_of_path") else world_snap, n)
    return Both()


def test_b4_a_hit_through_a_flat_owned_region_scores_without_crashing():
    w = H.World(lines=(30,), span=(5, 20))
    sid, props, store = _flat_store([[28, 32]])
    w.snap.add_edge("HAS_CALL", ("Symbol", sid), ("CallSite", w.cs))
    w.snap.add_node("Symbol", sid)
    chain = w.chain()
    chain[-1]["id"] = sid
    ans = H.answer(w.request(), [H.ans_item(H.sym_subject(sid, w.qname, 5, 20, ownedRegions=[28, 32]), chain)])    # the engine's copy is flat too
    res = G.score_gate([w.rec], {"snapshotId": H.SNAP, "answers": {w.cp: {"request": w.request(), "answer": ans}}}, _split(store, w.snap), ())
    assert res["items"][w.rec["itemId"]]["outcome"] == "hit" and res["defects"] == {}


def test_b4_a_miss_still_runs_the_source_check_over_flat_rows():
    w = H.World(lines=(99,), span=(5, 20))
    _, _, store = _flat_store([[28, 32]])
    run = {"snapshotId": H.SNAP, "answers": {w.cp: {"request": w.request(), "answer": H.answer(w.request(), [])}}}
    o = G.score_gate([w.rec], run, _split(store, w.snap), ())["items"][w.rec["itemId"]]
    assert o["outcome"] == "miss" and o["extractionAbsent"] is True


def test_b4_a_nested_pair_snapshot_is_still_accepted_as_a_regression():
    w = H.World(lines=(30,), span=(5, 20))
    w.snap.symbols[w.sid]["ownedRegions"] = [[28, 32]]
    sub = H.sym_subject(w.sid, w.qname, 5, 20, ownedRegions=[28, 32])
    o, res = w.outcome(w.answer([H.ans_item(sub, w.chain())]))
    assert o["outcome"] == "hit" and res["defects"] == {}


def test_b4_an_odd_length_stored_region_list_is_a_harness_error_not_a_crash():
    props = H.loader_symbol_props("3" * 32, "svc/c.py::call", 5, 20)
    props["ownedRegions"] = [28, 32, 40]
    with pytest.raises(G.HarnessError):
        G.StoreSnapshot(H.LoaderRunner([props])).symbol("3" * 32)


# ---------------------------------------------------------------- B5 at the scorer: a harness error is neither hit nor miss
@pytest.mark.parametrize("mut,code", [(lambda c: c[1].update(edgeType="X'; MATCH (n) DELETE n //"), "edgeType-not-allowed"),
                                      (lambda c: c[1].update(dir="<->"), "dir"),
                                      (lambda c: c[1].update(kind="link"), "hop-kind"),
                                      (lambda c: c[1].pop("id"), "hop-shape"),
                                      (lambda c: c.__setitem__(1, "not a dict"), "hop-shape")])
def test_b5_a_harness_error_chain_is_an_error_item_not_a_miss(w, mut, code):
    it = w.item()
    mut(it["chain"])
    o, res = w.outcome(w.answer([it]))
    assert o["outcome"] == "error" and o["reason"] == code and res["harnessErrors"] == {code: 1}
    assert G.decide_gate(res)["verdict"] == "blocked"


def test_b5_a_hit_by_another_candidate_stays_a_hit_and_the_error_still_blocks(w):
    bad = w.item()
    bad["chain"][1]["edgeType"] = "NOPE"
    o, res = w.outcome(w.answer([bad, w.item()]))
    assert o["outcome"] == "hit" and G.decide_gate(res)["verdict"] == "blocked"


def test_b5_a_plain_chain_failure_is_still_a_miss(w):
    it = w.item()
    it["chain"][1]["dir"] = "->"
    assert w.outcome(w.answer([it]))[0]["outcome"] == "miss"


# ---------------------------------------------------------------- m1: distance
def test_m1_a_hit_records_the_engines_distance_and_the_symbol_hop_count(w):
    it = w.item()
    it["distance"] = 7
    o, _ = w.outcome(w.answer([it]))
    assert o["distance"] == 7 and o["interFunctionHops"] == 1


def test_m1_a_config_hit_reports_a_nonzero_distance():
    sid = "5" * 32
    snap = H.DictSnapshot()
    snap.add_entry(sid, "deploy/compose.yaml", "services.web")
    rec = H.item(1, H.contract("http", "GET /a"), H.cfg("deploy/compose.yaml", "services.web"), stratum="wiring")
    cp = rec["anchorId"]
    snap.add_node("ContractPoint", cp)
    snap.add_edge("WRITES", ("ConfigEntry", sid), ("ContractPoint", cp))
    subj = {"type": "config", "entryId": sid, "path": "deploy/compose.yaml", "keyPath": "services.web"}
    chain = [H.hop0("ContractPoint", cp), H.hopn("ConfigEntry", sid, "WRITES", "<-", status="confirmed")]
    it = H.ans_item(subj, chain)
    it["distance"] = 2
    req = H.request([cp])
    run = {"snapshotId": H.SNAP, "answers": {cp: {"request": req, "answer": H.answer(req, [it])}}}
    o = G.score_gate([rec], run, snap, ())["items"][rec["itemId"]]
    assert o["outcome"] == "hit" and o["interFunctionHops"] == 0 and o["distance"] == 2


@pytest.mark.parametrize("copy", [[28, 33], [28, 32, 40, 41], None, [28, 32, 40]])
def test_b4_an_answer_copy_of_ownedRegions_that_differs_from_the_store_is_an_engine_defect(copy):
    w = H.World(lines=(30,), span=(5, 20))
    w.snap.symbols[w.sid]["ownedRegions"] = [[28, 32]]
    sub = H.sym_subject(w.sid, w.qname, 5, 20, **({"ownedRegions": copy} if copy is not None else {}))
    o, res = w.outcome(w.answer([H.ans_item(sub, w.chain())]))
    assert o["outcome"] == "hit" and res["defects"] == {"subject-ownedRegions": 1}          # credit on the snapshot's values; the difference is listed


# ---------------------------------------------------------------- A2: a per-anchor harness-error entry in run.json
# run.json shape: ``answers[<asked anchor id>] = {"harnessError": "<exception class name>"}`` in place of ``{"request", "answer"}``.
def _errored_run(w, cls="StoreError"):
    run = w.run()
    run["answers"][w.cp] = {"harnessError": cls}
    return run


def test_a2_an_anchor_entry_with_a_harness_error_makes_its_item_neither_hit_nor_miss(w):
    res = G.score_gate([w.rec], _errored_run(w), w.snap, ())
    o = res["items"][w.rec["itemId"]]
    assert o["outcome"] == "error" and o["reason"] == "StoreError" and res["harnessErrors"] == {"StoreError": 1}
    assert G.decide_gate(res, gated_strata=G.STRATA_GATED)["verdict"] == "blocked"


def test_a2_a_miss_beside_an_errored_anchor_is_blocked_not_fail(w):
    other = H.item(2, H.contract("http", "GET /b"), H.sym("svc/other.py::g"))
    run = _errored_run(w)
    run["answers"][other["anchorId"]] = {"request": H.request([other["anchorId"]]), "answer": H.answer(H.request([other["anchorId"]]), [])}
    w.snap.add_node("ContractPoint", other["anchorId"])
    res = G.score_gate([w.rec, other], run, w.snap, ())
    assert res["items"][other["itemId"]]["outcome"] == "miss" and res["items"][w.rec["itemId"]]["outcome"] == "error"
    d = G.decide_gate(res, gated_strata=G.STRATA_GATED)
    assert d["verdict"] == "blocked" and d["misses"] == 1 and d["errored"] == 1


def test_a2_two_items_on_one_errored_anchor_are_both_errors(w):
    twin = H.item(2, w.anchor, H.sym("svc/other.py::g"))
    res = G.score_gate([w.rec, twin], _errored_run(w), w.snap, ())
    assert [o["outcome"] for o in res["items"].values()] == ["error", "error"]


def test_a2_an_out_of_scope_item_on_an_errored_anchor_stays_out_of_scope(w):
    res = G.score_gate([w.rec], _errored_run(w), w.snap, ["svc/*"])
    assert res["items"][w.rec["itemId"]]["outcome"] == "out-of-scope" and res["harnessErrors"] == {}


@pytest.mark.parametrize("bad", ["Store Error: id 3333", "", 7, None, {"x": 1}, "a" * 200])
def test_a2_a_malformed_class_name_is_reported_as_unclassified_never_echoed(w, bad):
    run = w.run()
    run["answers"][w.cp] = {"harnessError": bad}
    res = G.score_gate([w.rec], run, w.snap, ())
    assert res["items"][w.rec["itemId"]]["outcome"] == "error" and res["harnessErrors"] == {"unclassified": 1}


def test_a2_an_entry_with_both_an_answer_and_a_harness_error_is_an_error(w):
    run = w.run()
    run["answers"][w.cp]["harnessError"] = "StoreError"
    assert G.score_gate([w.rec], run, w.snap, ())["items"][w.rec["itemId"]]["outcome"] == "error"


def test_a2_the_report_names_the_class_and_counts_never_an_id(w):
    res = G.score_gate([w.rec], _errored_run(w), w.snap, ())
    rep = G.build_report(res, G.decide_gate(res, gated_strata=G.STRATA_GATED), gated_strata=G.STRATA_GATED, withheld={}, flow_flag=False)
    assert rep["verdict"] == "blocked" and rep["erroredItems"] == 1 and rep["harnessErrors"] == {"StoreError": 1} and rep["scored"] == 0
    assert w.rec["itemId"] not in str(rep) and w.cp not in str(rep)


# ---------------------------------------------------------------- N2 (round C): H1 on a config key compares in the loader's STORED form
def test_n2_a_screen_failing_gold_segment_against_the_loader_masked_store_value_is_a_hit_and_is_listed():
    raw = "services." + "a" * 24 + ".ports"
    stored = _stored(raw)
    assert "[#…]" in stored and G.mask_keypath(raw) != stored                         # the two forms differ: the plant is real
    res = _cfg_world(stored, raw)                                                    # the store (and the answer) hold the stored form; the gold is raw
    out = res["items"][next(iter(res["items"]))]
    assert out["outcome"] == "hit" and res["maskedComparisons"] == 1 and res["defects"] == {}


def test_n2_a_gold_already_in_the_stored_form_also_hits():
    stored = _stored("services." + "a" * 24 + ".ports")
    res = _cfg_world(stored, stored)
    assert res["items"][next(iter(res["items"]))]["outcome"] == "hit"


def test_n2_a_different_unmasked_segment_is_still_a_miss_against_a_stored_masked_key():
    stored = _stored("services." + "a" * 24 + ".ports")
    res = _cfg_world(stored, "other." + "a" * 24 + ".ports")
    assert res["items"][next(iter(res["items"]))]["outcome"] == "miss"


# ---------------------------------------------------------------- round C: a per-anchor ENGINE error is a miss `engine-error` (-p1-gate-marker-ml.md §3)
ENGINE_ERR = {"outcome": "engine-error", "exceptionClass": "ValueError"}


def _run_with(w, entry):
    return {"snapshotId": H.SNAP, "answers": {w.cp: entry}}


def test_c_an_engine_error_entry_scores_every_item_of_the_anchor_as_a_miss_with_reason_engine_error():
    w = H.World()
    res = G.score_gate([w.rec], _run_with(w, dict(ENGINE_ERR)), w.snap, ())
    o = res["items"][w.rec["itemId"]]
    assert o["outcome"] == "miss" and o["reason"] == "engine-error" and o["exceptionClass"] == "ValueError"
    assert res["engineErrors"] == {"ValueError": 1} and res["harnessErrors"] == {}


def test_c_an_engine_error_is_in_the_fail_tier_and_a_harness_error_beside_it_stays_blocked():
    w = H.World()
    res = G.score_gate([w.rec], _run_with(w, dict(ENGINE_ERR)), w.snap, ())
    assert G.decide_gate(res)["verdict"] == "fail" and G.decide_gate(res)["misses"] == 1
    w2 = H.World()
    w2.cp2 = "9" * 32
    rec2 = H.item(2, H.contract("http", "GET /b"), H.sym("svc/c.py::call"))
    w2.snap.add_node("ContractPoint", rec2["anchorId"])
    run = {"snapshotId": H.SNAP, "answers": {w2.cp: dict(ENGINE_ERR), rec2["anchorId"]: {"harnessError": "HarnessError"}}}
    res2 = G.score_gate([w2.rec, rec2], run, w2.snap, ())
    assert G.decide_gate(res2)["verdict"] == "blocked"


def test_c_an_engine_error_entry_that_also_carries_an_answer_is_not_trusted():
    w = H.World()
    res = G.score_gate([w.rec], _run_with(w, {**ENGINE_ERR, "answer": w.answer(), "request": w.request()}), w.snap, ())
    assert res["items"][w.rec["itemId"]]["outcome"] == "error"


def test_c_the_exception_class_of_an_engine_error_is_an_identifier_only():
    w = H.World()
    res = G.score_gate([w.rec], _run_with(w, {"outcome": "engine-error", "exceptionClass": "bad: /home/x id=1"}), w.snap, ())
    assert res["items"][w.rec["itemId"]]["exceptionClass"] == "unclassified"


def test_c_the_report_gives_engine_errors_on_their_own_line_per_stratum_and_class_apart_from_harness_errors():
    w = H.World()
    res = G.score_gate([w.rec], _run_with(w, dict(ENGINE_ERR)), w.snap, ())
    rep = G.build_report(res, G.decide_gate(res), gated_strata=G.STRATA_GATED)
    assert rep["engineErrors"] == {"count": 1, "byStratum": {"http": 1}, "byClass": {"ValueError": 1},
                                "repeatStall": {"count": 0, "byStratum": {}}}
    assert rep["harnessErrors"] == {} and rep["erroredItems"] == 0
