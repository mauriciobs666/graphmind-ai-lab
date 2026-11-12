"""P1-22 the bounded result, unit tier: everything reachable within the bound, each item on a chain that recomputes to its own cost, and an independent oracle.

Contract: ``docs/plans/connected-code-graph-answer-spec2.md`` §4.1 (the bounded result is everything reachable within the bound; the walk keeps every arrival no other
arrival dominates, and expands a state once per kept arrival) and §4.2 (``unexplored`` from the final set of non-dominated arrivals); ``-p1-answers.md`` P1-22 (3), (5), (7).
The review's planted graphs (``docs/reviews/connected-code-graph-p1-22.md`` F1, F2) are built here from the ``G`` helper.
"""
from __future__ import annotations

import random

import pytest
from engine_walk_helpers import G, MemoryReader

from ccg.engine import bfs
from ccg.engine import state as S
from test_engine_bfs import edge_types, items_by_id, nodes, run

#: the edge types whose row has distance increment 1 (inter-function hops): the distance of a chain, recomputed from the chain alone
INTER_FUNCTION = {"CALLS", "CALLS_NAME", "VIA", "CONSUMES", "BOUND_BY"}


def recompute(item):
    """(possible hops, distance, length) of an item read off its own chain."""
    edges = [s["edge"] for s in item.chain[1:]]
    return (sum(e["status"] == "possible" for e in edges), sum(e["type"] in INTER_FUNCTION for e in edges), len(edges))


def assert_chains_recompute(w):
    for i in w.items():
        assert recompute(i) == (i.possible_hops, i.distance, i.length), (i.kind, i.id)
        assert i.distance <= w.bound and all(s["edge"] is None or s["edge"]["status"] in ("confirmed", "possible") for s in i.chain)


# --------------------------------------------------------------------------- F1: the planted graph, and F2: the route-length twins

def planted(route: str):
    """A is the anchor. ``c2`` (path ``r``) is reached by a confirmed route of distance 2 (``c1`` -> Ret of F -> F -> ``c3`` -> ``c2``) and by a ``name-match`` route of
    distance 1: directly (``short``), through five ``DATA`` hops (``long``), or confirmed (``confirmed``). ``c2``'s Ret leads to ``Gs``; ``c4`` calls ``Gs``: ``H`` lies at distance 2
    by the name route and at distance 3 by the confirmed one."""
    g = G()
    for s in ("A", "F", "Gs", "X", "Z", "H"):
        g.sym(s)
    g.site("c1", "X"), g.site("c2", "X"), g.site("c3", "Z"), g.site("c4", "H"), g.ret("R", "F"), g.ret("R2", "Gs")
    g.calls("c1", "A"), g.data("Ret", "R", "CallSite", "c1", "r", "ret"), g.calls("c3", "F"), g.data("CallSite", "c2", "CallSite", "c3", "r", "r")
    g.data("Ret", "R2", "CallSite", "c2", "r", "ret"), g.calls("c4", "Gs")
    if route == "short":
        g.calls_name("c2", "A", "a")
    elif route == "confirmed":
        g.calls("c2", "A")
    elif route == "long":
        g.site("cx", "X", callee="a"), g.calls_name("cx", "A", "a")
        prev = "cx"
        for i in range(5):
            g.site(f"cy{i}", "X"), g.data("CallSite", f"cy{i}", "CallSite", prev, "r", "r")
            prev = f"cy{i}"
        g.data("CallSite", "c2", "CallSite", prev, "r", "r")
    return g


def test_f1_the_planted_graphs_item_chain_recomputes_to_the_items_own_cost():
    w = run(planted("short"), ("Symbol", "A"))
    h = items_by_id(w)["H"]
    assert (h.possible_hops, h.distance, h.status, h.reason) == (1, 2, "possible", "name-match")
    assert edge_types(h)[0] == ("CALLS_NAME", "<-") and edge_types(h)[-1] == ("HAS_CALL", "<-") and recompute(h) == (1, 2, h.length)
    assert [s["edge"]["status"] for s in h.chain[1:]].count("possible") == 1       # the review's chain was 9 hops, all confirmed, three CALLS
    assert_chains_recompute(w)


@pytest.mark.parametrize("route", ["short", "long", "confirmed"])
def test_every_twin_lists_h_on_a_chain_that_recomputes(route):
    w = run(planted(route), ("Symbol", "A"))
    assert "H" in items_by_id(w) and items_by_id(w)["H"].distance == 2
    assert_chains_recompute(w)


def test_f2_the_item_set_does_not_depend_on_the_length_of_the_possible_route():
    short, long_ = run(planted("short"), ("Symbol", "A")), run(planted("long"), ("Symbol", "A"))
    key = lambda w: {(i.kind, i.id): (i.possible_hops, i.distance, i.status) for i in w.items()}   # noqa: E731
    assert key(short) == key(long_) and key(short)[("symbol", "H")] == (1, 2, "possible")
    assert long_.unexplored == short.unexplored == 1


def test_the_all_confirmed_twins_items_are_a_subset_of_the_bounded_walks_and_h_is_confirmed_there():
    conf, short = run(planted("confirmed"), ("Symbol", "A")), run(planted("short"), ("Symbol", "A"))
    assert {(i.kind, i.id) for i in conf.items()} <= {(i.kind, i.id) for i in short.items()}
    assert items_by_id(conf)["H"].status == "confirmed"


def test_both_arrivals_at_the_shared_state_are_kept_neither_dominating_the_other():
    w = run(planted("short"), ("Symbol", "A"))
    assert sorted(e.cost[:2] for e in w.arrivals[("CallSite", "c2", "path", "r")]) == [(0, 2), (1, 1)]
    assert w.best[("CallSite", "c2", "path", "r")].cost[:2] == (0, 2)           # the display entry is the lexicographic minimum


def test_the_cut_edge_comes_from_the_kept_arrival_at_the_bound_only():
    assert run(planted("short"), ("Symbol", "A")).cut == frozenset({("CALLS", "c4", "Gs")})


# --------------------------------------------------------------------------- the (0,2), (9,0), (5,1) arrival order

def three_pairs():
    """One state, T = (CallSite t, path r), reached three times in this order by chains with pairs (0,2), (9,0), (5,1): none dominates another."""
    g = G()
    for s in ("Q", "F", "Gx", "Hh", "W"):
        g.sym(s)
    g.param("p", "Q", index=0)
    g.site("t", "W"), g.site("ca", "Q"), g.site("cb", "F"), g.site("cx", "Q")
    # (0,2): two confirmed calls
    g.data("CallSite", "ca", "Param", "p", "p0", "a0"), g.calls("ca", "F"), g.param("pf", "F", index=0), g.data("CallSite", "cb", "Param", "pf", "p0", "a0")
    g.calls("cb", "Gx"), g.param("pg", "Gx", index=0), g.data("CallSite", "t", "Param", "pg", "p0", "r")
    # (9,0): nine possible DATA hops, no call
    poss = {"status": "possible", "reason": "field-truncated"}
    for i in range(8):
        g.site(f"u{i}", "Q")
        g.data("CallSite", f"u{i}", *(("Param", "p") if i == 0 else ("CallSite", f"u{i-1}")), "p0" if i == 0 else "r", "r", **poss)
    g.data("CallSite", "t", "CallSite", "u7", "r", "r", **poss)
    # (5,1): one call, three confirmed padding hops, five possible hops
    g.data("CallSite", "cx", "Param", "p", "p0", "a0"), g.calls("cx", "Hh"), g.param("ph", "Hh", index=0)
    g.site("n0", "Hh"), g.data("CallSite", "n0", "Param", "ph", "p0", "r")
    for i in (1, 2):
        g.site(f"n{i}", "Hh"), g.data("CallSite", f"n{i}", "CallSite", f"n{i-1}", "r", "r")
    prev = "n2"
    for i in range(4):
        g.site(f"m{i}", "Hh"), g.data("CallSite", f"m{i}", "CallSite", prev, "r", "r", **poss)
        prev = f"m{i}"
    g.data("CallSite", "t", "CallSite", prev, "r", "r", **poss)
    return g


def test_three_arrivals_with_pairs_02_90_51_in_that_order_are_all_kept():
    w = run(three_pairs(), ("Param", "p"), S.path_state("p0"), bound=2)
    arr = w.arrivals[("CallSite", "t", "path", "r")]
    by_level = sorted(arr, key=lambda e: e.cost[2])
    assert [e.cost[:2] for e in by_level] == [(0, 2), (9, 0), (5, 1)]                  # the arrival order the review's incomplete fix mishandles
    assert [e.cost[2] for e in by_level] == sorted(e.cost[2] for e in by_level) and len({e.cost[2] for e in by_level}) == 3
    w_ = items_by_id(w)["W"]
    assert (w_.possible_hops, w_.distance) == (0, 2)                                     # the best chain among the three
    assert_chains_recompute(w)


# --------------------------------------------------------------------------- an independent oracle: DP over (state, distance <= bound)

def _rule_kind(column, label, edge_type, direction):
    """The item rules, typed again here (not imported)."""
    if column == bfs.IMPACT:
        if label == "Symbol" and edge_type in ("HAS_CALL", "HAS_PARAM", "HAS_RETURN", "HAS_COND") and direction == "<-":
            return "symbol"
        if label == "ConfigEntry" and edge_type == "BOUND_BY" and direction == "->":
            return "config"
    else:
        if label == "Symbol" and edge_type in ("CALLS", "CALLS_NAME", "TO") and direction == "->":
            return "symbol"
        if label == "Unit" and edge_type == "TO_UNIT":
            return "unit"
    return None


def oracle(reader, anchor, state, column, bound):
    """Minimum possible hops per (state, distance <= bound) by relaxation to a fixed point, written without the walk's bookkeeping (no kept sets, no levels, no
    predecessor entries): it asks ``bfs._expand`` only for the row semantics of single states. Returns ``(items, cut)``: ``{(kind, id): (possible, distance)}`` and
    the cut edges of the keys that hold a non-dominated pair at the bound."""
    by_group = bfs._group_rows(bfs.ROWS, column)
    start = (anchor[0], anchor[1], state[0], state[1])
    best = {(start, 0): 0}
    arrive = {}                      # (key, edge type, dir, distance) -> min possible hops
    cutedges = {}                    # key -> edges a row would traverse from it past the bound (any distance: only keys at the bound are used)
    work = [(start, 0, 0)]
    facts = {}
    seen_expand = set()
    while work:
        key, d, p = work.pop()
        if best.get((key, d)) != p or (key, d, p) in seen_expand:
            continue
        seen_expand.add((key, d, p))
        for mv in bfs._expand(reader, by_group, [bfs.Entry((p, d, 0, 0), None, None, key)], facts):
            own = 1 if mv.step.status == "possible" else 0
            nd = d + mv.row.inc
            if nd > bound:
                cutedges.setdefault((key, d), set()).add(bfs._edge_id(mv.row, key[1], mv.step.id))
                continue
            to = (mv.step.label, mv.step.id, mv.to_state[0], mv.to_state[1])
            a = (to, mv.edge.type, mv.edge.dir, nd)
            if arrive.get(a, 99) > p + own:
                arrive[a] = p + own
            if best.get((to, nd), 99) > p + own:
                best[(to, nd)] = p + own
                work.append((to, nd, p + own))
    items = {}
    for (to, et, di, nd), p in arrive.items():
        kind = _rule_kind(column, to[0], et, di)
        if kind and to[:2] != start[:2]:
            cur = items.get((kind, to[1]))
            if cur is None or (p, nd) < cur:
                items[(kind, to[1])] = (p, nd)
    per_key = {}
    for (key, d), p in best.items():
        per_key.setdefault(key, []).append((p, d))
    cut = set()
    for key, pairs in per_key.items():
        for p, d in pairs:
            dominated = any((q, e) != (p, d) and q <= p and e <= d for q, e in pairs)
            if not dominated and d == bound:
                cut |= cutedges.get((key, d), set())
    return items, frozenset(cut)


def random_world(seed):
    rnd = random.Random(seed)
    g = G()
    syms = [g.sym(f"s{i}") for i in range(rnd.randint(3, 6))]
    sites, params, rets, conds = [], [], [], []
    for s in syms:
        for j in range(rnd.randint(1, 3)):
            sites.append(g.site(f"{s}_c{j}", s, callee="n"))
        for j in range(rnd.randint(0, 2)):
            params.append(g.param(f"{s}_p{j}", s, index=j))
        if rnd.random() < 0.7:
            rets.append(g.ret(f"{s}_r", s))
        if rnd.random() < 0.5:
            conds.append(g.cond(f"{s}_k", s))
    poss = lambda: {"status": "possible", "reason": "field-truncated"} if rnd.random() < 0.3 else {}   # noqa: E731
    for c in sites:
        roll = rnd.random()
        if roll < 0.55:
            g.calls(c, rnd.choice(syms), **({"status": "possible", "reason": "name-match"} if rnd.random() < 0.2 else {}))
        elif roll < 0.75:
            g.calls_name(c, rnd.choice(syms), "n")
    for _ in range(rnd.randint(4, 18)):
        ul, u = rnd.choice([("CallSite", c) for c in sites] + [("Ret", r) for r in rets] + [("Cond", k) for k in conds])
        dl, d = rnd.choice([("CallSite", c) for c in sites] + [("Param", p) for p in params])
        fp = rnd.choice(["r", "r.x"]) if dl == "CallSite" else rnd.choice(["p0", "p0.a", "p1"])
        tp = {"CallSite": rnd.choice(["a0", "a1.k", "r"]), "Ret": "ret", "Cond": ""}[ul]
        g.data(ul, u, dl, d, fp, tp, **poss())
    for _ in range(rnd.randint(0, 4) if conds else 0):
        g.controlled(*rnd.choice([("CallSite", c) for c in sites] + [("Ret", r) for r in rets]), rnd.choice(conds))
    cps = [g.cp(f"cp{i}", f"GET /k{i}") for i in range(rnd.randint(1, 2))]
    for cp in cps:
        for s in rnd.sample(syms, rnd.randint(0, 2)):
            g.provides(s, cp, **({"status": "possible", "reason": "runtime-built"} if rnd.random() < 0.2 else {}))
        for c in rnd.sample(sites, min(len(sites), rnd.randint(0, 3))):
            g.consumes(c, cp, **({"status": "possible", "reason": "runtime-built"} if rnd.random() < 0.2 else {}))
    for i in range(rnd.randint(0, 3)):
        lk = g.link(f"L{i}", provider=rnd.choice(syms), cp=rnd.choice(cps), status=rnd.choice(["confirmed", "possible"]), reason="port-shared")
        vs = rnd.sample(sites, min(len(sites), rnd.randint(1, 3)))
        for c in vs:
            g.via(c, lk, status=rnd.choice(["confirmed", "possible"]), reason="port-shared")
        for j in range(rnd.randint(0, 2)):
            e = g.entry(f"e{i}{j}", key_path=f"k{i}{j}")
            for c in rnd.sample(vs, rnd.randint(1, len(vs))):
                g.bound(lk, e, c, ord=j)
    anchors = [(("Symbol", rnd.choice(syms)), S.VALUE, bfs.IMPACT), (("Symbol", rnd.choice(syms)), S.CONTRACT, bfs.IMPACT), (("Symbol", rnd.choice(syms)), S.KEY, bfs.IMPACT),
               (("ContractPoint", cps[0]), S.CONTRACT, bfs.IMPACT), (("Symbol", rnd.choice(syms)), S.VALUE, bfs.DEPENDS_ON), (("CallSite", rnd.choice(sites)), S.path_state("r"), bfs.IMPACT)]
    if params:
        anchors.append((("Param", rnd.choice(params)), S.path_state(rnd.choice(["p0", "p0.a"])), bfs.IMPACT))
    if rets:
        anchors.append((("Ret", rnd.choice(rets)), S.VALUE, bfs.DEPENDS_ON))
    ents = [n["entryId"] for n in g.rows.nodes.get("ConfigEntry", [])]
    if ents:
        anchors.append((("ConfigEntry", rnd.choice(ents)), S.VALUE, bfs.IMPACT))
    return g, anchors


@pytest.mark.parametrize("seed", range(120))
def test_the_walk_agrees_with_the_independent_dp_on_items_costs_and_unexplored_at_every_bound(seed):
    g, anchors = random_world(seed)
    for anchor, st, column in anchors:
        for bound in (1, 2, 3):
            w = bfs.walk(MemoryReader(g), anchor, st, column=column, bound=bound)
            items, cut = oracle(MemoryReader(g), anchor, st, column, bound)
            got = {(i.kind, i.id): (i.possible_hops, i.distance) for i in w.items()}
            assert got == items, (seed, anchor, st, column, bound)
            assert w.cut == cut and w.unexplored == len(cut), (seed, anchor, st, column, bound)
            assert_chains_recompute(w)


def test_the_random_worlds_exercise_the_bound_not_trivially():
    cut = multi = items = 0
    for seed in range(120):
        g, anchors = random_world(seed)
        for anchor, st, column in anchors:
            w = bfs.walk(MemoryReader(g), anchor, st, column=column, bound=2)
            cut += w.unexplored > 0
            multi += any(len(v) > 1 for v in w.arrivals.values())
            items += len(w.items())
    assert cut > 40 and multi > 20 and items > 500
