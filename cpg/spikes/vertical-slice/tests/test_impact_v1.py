"""S17-L2: ``impact.py``'s ``bfs-v1`` against graph model v1 (``docs/plans/connected-code-graph-p0-graph.md`` section 3, 5).

Hand-built graphs through the ``MemoryReader`` seam; the ``live`` tests touch only fresh
``ccg_p0slice__build_l2*`` keys on ``ccg-falkordb`` (127.0.0.1:6390), deleted at the end.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import impact as I  # noqa: E402
import load as L  # noqa: E402
import probe_s14 as P  # noqa: E402
from test_live_slice import MemoryReader, _sym, h  # noqa: E402

FC = "function-level-continuation"

# ---------------------------------------------------------------------------------------------
# M2: value-mode labelling (section 3.3, section 5 (a)-(c)).
# ---------------------------------------------------------------------------------------------


def ret_graph(*, ret_reached: bool):
    """cB (in B) calls the anchor A; X calls B. With ``ret_reached`` B's `rB` returns cB's result (`r.y`)."""
    data = [("Ret", "rB", "CallSite", "cB", "r.y", "ret", "confirmed")] if ret_reached else []
    return MemoryReader(
        calls=[("A", "cB", "confirmed"), ("B", "cX", "confirmed")],
        owners=[("CallSite", "cB", "B"), ("Ret", "rB", "B"), ("CallSite", "cX", "X")],
        data=data,
    )


def test_a_caller_reached_through_a_ret_of_the_reached_function_is_confirmed():
    items = _sym(I.walk(ret_graph(ret_reached=True), ("Symbol", "A")).items())
    assert set(items) == {"B", "X"}
    x = items["X"]
    assert (x.status, x.possible_hops, x.distance, x.reason) == ("confirmed", 0, 2, None)
    assert all(s["edge"] is None or "reason" not in s["edge"] for s in x.chain)


def test_a_caller_of_a_function_whose_return_the_change_does_not_reach_is_possible_with_the_reason():
    items = _sym(I.walk(ret_graph(ret_reached=False), ("Symbol", "A")).items())
    x = items["X"]
    assert (x.status, x.possible_hops, x.distance, x.reason) == ("possible", 1, 2, FC)
    calls = [s["edge"] for s in x.chain if s["edge"] and s["edge"]["type"] == "CALLS"]
    assert calls == [{"type": "CALLS", "dir": "<-", "status": "confirmed"},
                     {"type": "CALLS", "dir": "<-", "status": "possible", "reason": FC}]


def test_the_direct_caller_of_the_anchor_stays_confirmed_whether_or_not_a_ret_is_reached():
    for ret in (True, False):
        b = _sym(I.walk(ret_graph(ret_reached=ret), ("Symbol", "A")).items())["B"]
        assert (b.status, b.possible_hops, b.reason) == ("confirmed", 0, None), ret


def test_param_and_cond_owned_points_continue_at_function_level_too():
    for label in ("Param", "Cond"):
        g = MemoryReader(calls=[("B", "cX", "confirmed")], owners=[(label, "pt", "B"), ("CallSite", "cX", "X")])
        x = _sym(I.walk(g, (label, "pt")).items())["X"]
        assert (x.status, x.reason) == ("possible", FC), label


def test_via_data_is_true_exactly_when_the_items_best_path_has_a_data_hop():
    with_ret = _sym(I.walk(ret_graph(ret_reached=True), ("Symbol", "A")).items())
    assert with_ret["X"].via_data is True      # the confirmed route to X goes through the DATA hop to `rB`
    assert with_ret["B"].via_data is False     # B's own best path is the call site, no DATA hop
    without = _sym(I.walk(ret_graph(ret_reached=False), ("Symbol", "A")).items())
    assert without["B"].via_data is False and without["X"].via_data is False


def test_a_provider_reached_at_function_level_makes_its_consumers_possible_but_an_anchor_provider_does_not():
    g = MemoryReader(
        calls=[("A", "cF", "confirmed")],
        owners=[("CallSite", "cF", "F"), ("CallSite", "cons", "G")],
        to=[("L", "F", "confirmed")], via=[("cons", "L", "confirmed")],
    )
    g_item = _sym(I.walk(g, ("Symbol", "A")).items())["G"]
    assert (g_item.status, g_item.reason) == ("possible", FC)
    assert [s["edge"]["type"] for s in g_item.chain[1:]] == ["CALLS", "HAS_CALL", "TO", "VIA", "HAS_CALL"]
    assert g_item.chain[3]["edge"]["reason"] == FC  # the TO hop out of the function-level provider carries it
    direct = _sym(I.walk(g, ("Symbol", "F")).items())["G"]  # F itself changed: its Link hop is confirmed
    assert (direct.status, direct.reason) == ("confirmed", None)


def test_a_function_reached_both_ways_is_one_item_and_the_anchor_is_never_its_own_item():
    g = ret_graph(ret_reached=True)
    res = I.walk(g, ("Symbol", "A"))
    assert [i.symbol_id for i in res.items()] == ["B", "X"]
    g.calls.append(("B", "cA", "confirmed"))  # recursion back into the anchor
    g.owners.append(("CallSite", "cA", "A"))
    assert "A" not in {i.symbol_id for i in I.walk(g, ("Symbol", "A")).items()}


# ---------------------------------------------------------------------------------------------
# bfs-v1 guards.
# ---------------------------------------------------------------------------------------------


def test_a_two_point_data_cycle_with_matching_paths_terminates_with_two_states():
    g = MemoryReader(data=[("CallSite", "c2", "CallSite", "c1", "r", "a0", "confirmed"),
                           ("CallSite", "c1", "CallSite", "c2", "a0", "r", "confirmed")])
    res = I.walk(g, ("CallSite", "c1"), ("path", "r"))
    assert res.state_count == 2 and res.levels == 2 and res.frontier_sizes == [1, 1]
    assert res.best[("CallSite", "c1", "path", "r")].cost == (0, 0, 0)  # the revisit did not replace the anchor


def test_an_equal_cost_second_arrival_is_not_re_expanded_so_state_counts_stay_exact():
    # n is reached at level 1 directly and at level 2 through c1, both at cost (0, 0); m hangs off n.
    # Re-expanding on `<=` would walk m again at level 3 and move m's best entry to the later, longer arrival.
    g = MemoryReader(data=[
        ("CallSite", "n", "CallSite", "c0", "r", "a0", "confirmed"),
        ("CallSite", "c1", "CallSite", "c0", "r", "a0", "confirmed"),
        ("CallSite", "n", "CallSite", "c1", "a0", "a0", "confirmed"),
        ("CallSite", "m", "CallSite", "n", "a0", "a0", "confirmed"),
    ])
    res = I.walk(g, ("CallSite", "c0"), ("path", "r"))
    assert res.state_count == 4 and res.levels == 3 and res.frontier_sizes == [1, 2, 1]
    assert res.best[("CallSite", "n", "path", "a0")].cost == (0, 0, 1)
    assert res.best[("CallSite", "n", "path", "a0")].pred == ("CallSite", "c0", "path", "r")
    assert res.best[("CallSite", "m", "path", "a0")].cost == (0, 0, 2)


def test_a_data_self_edge_terminates():
    g = MemoryReader(data=[("CallSite", "c", "CallSite", "c", "r", "r", "confirmed")])
    res = I.walk(g, ("CallSite", "c"), ("path", "r"))
    assert res.state_count == 1 and res.levels == 1


def test_a_strictly_smaller_arrival_is_re_expanded_and_improves_everything_downstream():
    # n is reached by a possible edge at level 1 and by two confirmed hops at level 2; m hangs off n
    g = MemoryReader(data=[
        ("CallSite", "n", "CallSite", "c0", "r", "a0", "possible"),
        ("CallSite", "c1", "CallSite", "c0", "r", "a0", "confirmed"),
        ("CallSite", "n", "CallSite", "c1", "a0", "a0", "confirmed"),
        ("CallSite", "m", "CallSite", "n", "a0", "a0", "confirmed"),
    ])
    res = I.walk(g, ("CallSite", "c0"), ("path", "r"))
    assert res.best[("CallSite", "n", "path", "a0")].cost[:2] == (0, 0)
    assert res.best[("CallSite", "m", "path", "a0")].cost[:2] == (0, 0)
    assert res.best[("CallSite", "m", "path", "a0")].pred == ("CallSite", "n", "path", "a0")


def _call_chain(n_callers, anchor_kind):
    """A chain of ``n_callers`` symbols, each calling the one before; the anchor is a Symbol or a CallSite."""
    calls = [(f"S{i}", f"c{i + 1}", "confirmed") for i in range(n_callers)]
    owners = [("CallSite", f"c{i + 1}", f"S{i + 1}") for i in range(n_callers)]
    if anchor_kind == "symbol":
        return MemoryReader(calls=calls, owners=owners), ("Symbol", "S0")
    owners[0] = ("CallSite", "c1", "S1")
    return MemoryReader(calls=calls[1:], owners=owners), ("CallSite", "c1")


def test_the_default_level_cap_is_64_and_a_walk_of_exactly_64_levels_passes():
    g, anchor = _call_chain(32, "callsite")  # 1 level to S1, then 2 per further caller, plus the closing empty level
    res = I.walk(g, anchor, ("path", "r"))
    assert res.levels == 64 and len(res.items()) == 32


def test_a_walk_of_65_levels_raises_loudly_under_the_default_cap():
    g, anchor = _call_chain(32, "symbol")  # 2 levels per caller plus the closing one
    with pytest.raises(I.ImpactError, match="exceeded 64 levels"):
        I.walk(g, anchor)
    g, anchor = _call_chain(31, "symbol")  # 63 levels
    assert I.walk(g, anchor).levels == 63


READER_CALLS = {
    "calls_in": lambda r: r.calls_in(["a"]),
    "owner_of": lambda r: r.owner_of("CallSite", ["a"]),
    "data_in": lambda r: r.data_in("CallSite", ["a"]),
    "to_in": lambda r: r.to_in(["a"]),
    "via_in": lambda r: r.via_in(["a"]),
    "provides_in": lambda r: r.provides_in(["a"]),
    "provides_out": lambda r: r.provides_out(["a"]),
}


@pytest.mark.parametrize("method", sorted(READER_CALLS))
def test_every_reader_raises_on_a_reply_that_reaches_the_resultset_size_and_accepts_one_below(method):
    assert I.RESULT_SET_LIMIT == 10_000
    call = READER_CALLS[method]
    with pytest.raises(I.ImpactError, match="RESULTSET_SIZE"):
        call(I.CypherReader(lambda q, p: [["x"]] * 10_000))
    assert len(call(I.CypherReader(lambda q, p: [["x"]] * 9_999))) == 9_999


def test_a_full_reply_in_a_later_chunk_raises_too():
    replies = iter([[["x"]], [["x"]] * 10_000])
    reader = I.CypherReader(lambda q, p: next(replies))
    with pytest.raises(I.ImpactError, match="RESULTSET_SIZE"):
        reader.calls_in([str(i) for i in range(1500)])


def test_probe_s14_v2_gate_ends_every_non_ret_item_so_it_has_fewer_states_than_v0():
    g = MemoryReader(
        calls=[("A", "cB", "confirmed"), ("B", "cX", "confirmed")],
        owners=[("CallSite", "cB", "B"), ("Ret", "rB", "B"), ("CallSite", "cX", "X")],
    )
    v0 = I.walk(g, ("Symbol", "A"))
    with P.Gate():
        v2 = I.walk(g, ("Symbol", "A"))
    assert {i.symbol_id for i in v0.items()} == {"B", "X"}
    assert {i.symbol_id for i in v2.items()} == {"B"}  # B is reported but not walked further
    assert v2.state_count < v0.state_count
    assert I._expand is not None and I.walk(g, ("Symbol", "A")).state_count == v0.state_count  # restored on exit


# ---------------------------------------------------------------------------------------------
# Live (ccg-falkordb): D8 sparse status read back by the reader, fresh build keys only.
# ---------------------------------------------------------------------------------------------

HOST, PORT = "127.0.0.1", 6390


@pytest.fixture(scope="module")
def live_store():
    try:
        return L.LiveStore(HOST, PORT)
    except FileNotFoundError as exc:
        pytest.skip(f"ccg-falkordb admin password file not readable ({exc.filename}); see cpg/infra/README.md")
    except Exception as exc:  # noqa: BLE001 -- any connection failure is "unreachable"
        pytest.skip(f"ccg-falkordb unreachable at {HOST}:{PORT} ({type(exc).__name__}); start it with cpg/infra/ccg-falkordb.sh up")


@pytest.mark.live
def test_d8_a_walk_over_sparse_edges_reads_the_absent_status_as_the_default_and_a_stored_one_as_is(live_store):
    key = L.check_key(L.BUILD_PREFIX + "l2d8")
    if live_store.exists(key):
        live_store.delete(key)
    try:
        live_store.query(key, "CREATE (:Symbol {symbolId: 'A'}), (:Symbol {symbolId: 'B'}), (:Symbol {symbolId: 'P'}), "
                              "(:CallSite {callSiteId: 'cB'}), (:CallSite {callSiteId: 'cP'}), "
                              "(:CallSite {callSiteId: 'c0'}), (:CallSite {callSiteId: 'u1'}), (:CallSite {callSiteId: 'u2'})")
        for cs, sym in (("cB", "B"), ("cP", "P")):
            live_store.query(key, "MATCH (s:Symbol {symbolId: $s}), (c:CallSite {callSiteId: $c}) CREATE (s)-[:HAS_CALL]->(c)", {"s": sym, "c": cs})
        # CALLS: A <- cB stored bare (default), A <- cP stored with a non-default status
        live_store.query(key, "MATCH (c:CallSite {callSiteId: 'cB'}), (a:Symbol {symbolId: 'A'}) CREATE (c)-[:CALLS]->(a)")
        live_store.query(key, "MATCH (c:CallSite {callSiteId: 'cP'}), (a:Symbol {symbolId: 'A'}) CREATE (c)-[:CALLS {status: 'possible', reason: 'name-match'}]->(a)")
        # DATA: u1 <- c0 stored bare, u2 <- c0 stored possible
        live_store.query(key, "MATCH (u:CallSite {callSiteId: 'u1'}), (d:CallSite {callSiteId: 'c0'}) CREATE (u)-[:DATA {fromPath: 'r', toPath: 'a0'}]->(d)")
        live_store.query(key, "MATCH (u:CallSite {callSiteId: 'u2'}), (d:CallSite {callSiteId: 'c0'}) "
                              "CREATE (u)-[:DATA {fromPath: 'r', toPath: 'a1', status: 'possible', reason: 'field-truncated'}]->(d)")
        reader = I.CypherReader(lambda q, p=None: live_store.query(key, q, p).result_set)
        assert reader.calls_in(["A"]) == [("A", "cB", "confirmed"), ("A", "cP", "possible")]
        items0 = _sym(I.walk(reader, ("Symbol", "A")).items())
        assert items0["P"].reason is None and items0["B"].reason is None  # a stored possible is not function-level
        assert reader.data_in("CallSite", ["c0"]) == [("c0", "CallSite", "u1", "r", "a0", "confirmed"),
                                                      ("c0", "CallSite", "u2", "r", "a1", "possible")]
        items = _sym(I.walk(reader, ("Symbol", "A")).items())
        assert (items["B"].status, items["P"].status) == ("confirmed", "possible")
        p_calls = [st["edge"] for st in items["P"].chain if st["edge"] and st["edge"]["type"] == "CALLS"]
        assert p_calls == [{"type": "CALLS", "dir": "<-", "status": "possible"}]  # no reason on the stored-possible hop
    finally:
        if live_store.exists(key):
            live_store.delete(key)
