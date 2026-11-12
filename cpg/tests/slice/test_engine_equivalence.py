"""P1-22 equivalence: the engine's walk, restricted to the slice's rows, is the slice's walk (``cpg/spikes/vertical-slice/impact.py``).

Plan: ``docs/plans/connected-code-graph-p1-answers.md`` P1-22 (1). Two oracles, both the slice's own code (never a golden copied from the engine):

* **random graphs** (always run): the same ``GraphRows`` through ``impact.py`` (its own ``Reader`` seam, written here from the rows) and through ``ccg.engine.bfs`` over
  the *slice variant* of the shipped row table (the rows the slice follows, plus its function-level state: a table variant, which the walk interprets like any
  table). The visited state sets and the item sets must be equal; seeds are fixed.
* **the stored slice bundles** (``slice`` marker, D9: absent artifacts skip, reported unrun): the six slice walks (anchor ``apiFetch.p0`` at ``p0``, ``p0.path``,
  ``p0.method``, ``p0.query``, ``p0.nothing``, and the value walk from ``buildUrl``) reproduce the recorded ``-p0-graph.md`` §3.3 figures (21 items each, states
  98 / 84 / 84 / 82 / 81 / 83); the V2-gated rows (``probe_s14.py``'s what-if) give 19 / 5 / 5 / 3 / 2 / 4 **states** and one item; the shipped (precise) table's
  figures are pinned from what the run observed, and ``apiFetch`` stays ``confirmed``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine_walk_helpers import G, MemoryReader  # noqa: E402
from packrt_helpers import CPG, S14A  # noqa: E402
from test_packrt_conformance import SHA256  # noqa: E402

from ccg.engine import bfs  # noqa: E402
from ccg.engine import state as S  # noqa: E402


def _slice_impact():
    spec = importlib.util.spec_from_file_location("slice_impact", CPG / "spikes/vertical-slice" / "impact.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["slice_impact"] = mod
    spec.loader.exec_module(mod)
    return mod


I = _slice_impact()


class SliceReader:
    """``impact.Reader`` over ``GraphRows``: one method per (edge type, direction) the slice follows, batches of ids, sorted."""

    def __init__(self, rows):
        self.r, self.edges = rows, rows.edges

    def E(self, a, b, c):
        return self.edges.get((a, b, c), [])

    def calls_in(self, ids):
        return sorted((e["t"], e["s"], e.get("p", {}).get("status", "confirmed")) for e in self.E("CallSite", "CALLS", "Symbol") if e["t"] in ids)

    def owner_of(self, label, ids):
        rel = {"CallSite": "HAS_CALL", "Param": "HAS_PARAM", "Ret": "HAS_RETURN", "Cond": "HAS_COND"}[label]
        return sorted((e["t"], e["s"]) for e in self.E("Symbol", rel, label) if e["t"] in ids)

    def data_in(self, label, ids):
        if label not in I.DEF_LABELS:
            return []
        out = []
        for (src, et, dst), lst in self.edges.items():
            if et == "DATA" and dst == label:
                out += [(e["t"], src, e["s"], e["p"]["fromPath"], e["p"]["toPath"], e["p"].get("status", "confirmed")) for e in lst if e["t"] in ids]
        return sorted(out)

    def to_in(self, ids):
        status = {n["linkId"]: n["status"] for n in self.r.nodes.get("Link", [])}
        return sorted((e["t"], e["s"], status[e["s"]]) for e in self.E("Link", "TO", "Symbol") if e["t"] in ids)

    def via_in(self, ids):
        return sorted((e["t"], e["s"], e["p"]["status"]) for e in self.E("CallSite", "VIA", "Link") if e["t"] in ids)

    def provides_in(self, ids):
        return sorted((e["t"], e["s"]) for e in self.E("Symbol", "PROVIDES", "ContractPoint") if e["t"] in ids)

    def provides_out(self, ids):
        return sorted((e["s"], e["t"]) for e in self.E("Symbol", "PROVIDES", "ContractPoint") if e["s"] in ids)


def slice_table():
    """The shipped table restricted to the rows ``impact.py`` follows, plus its ``function`` state: a point other than a ``Ret`` enters ``function``, from which
    ``CALLS <-`` and ``TO <-`` continue (the slice's M2 baseline). A variant table: the walk has no knowledge of it."""
    keep = {"to_in:contract", "provides_out", "owner:Ret", "owner:CallSite:contract", "data_in:Param", "data_in:CallSite", "via_in:value", "via_in:contract", "provides_in"}
    out = []
    for r in bfs.ROWS:
        if r.column != bfs.IMPACT:
            continue
        if r.id in ("calls_in", "to_in:value"):
            out.append(dataclasses.replace(r, kinds=("value", "function")))
        elif r.id in keep:
            out.append(r)
        elif r.id in ("owner:CallSite", "owner:Param", "owner:Cond"):
            out.append(dataclasses.replace(r, to="function"))
    return out


V2_LIKE_IDS = {"calls_in", "to_in:value", "to_in:contract", "provides_out", "owner:CallSite", "owner:Param", "owner:Ret", "owner:Cond", "owner:CallSite:contract",
               "data_in:Param", "data_in:CallSite", "via_in:value", "via_in:contract", "provides_in"}


def v2_like_table():
    """``probe_s14.py``'s V2 what-if as a table: the slice's rows with a non-``Ret`` point ending its function (the shipped owner rows) and no ``function`` state."""
    return [r for r in bfs.ROWS if r.column == bfs.IMPACT and r.id in V2_LIKE_IDS]


# --------------------------------------------------------------------------- random graphs, always run

def random_graph(seed: int) -> tuple[G, list]:
    rnd = random.Random(seed)
    g = G()
    syms = [g.sym(f"s{i}") for i in range(rnd.randint(3, 7))]
    sites, params, rets, conds = [], [], [], []
    for s in syms:
        for j in range(rnd.randint(0, 3)):
            sites.append(g.site(f"{s}_c{j}", s))
        for j in range(rnd.randint(0, 2)):
            params.append(g.param(f"{s}_p{j}", s, index=j))
        for j in range(rnd.randint(0, 1)):
            rets.append(g.ret(f"{s}_r{j}", s))
        for j in range(rnd.randint(0, 2)):
            conds.append(g.cond(f"{s}_k{j}", s))
    for c in sites:
        if rnd.random() < 0.7:
            g.calls(c, rnd.choice(syms), **({"status": "possible", "reason": "name-match"} if rnd.random() < 0.2 else {}))
    for _ in range(rnd.randint(2, 14)):
        use_label, use = rnd.choice([("CallSite", c) for c in sites] + [("Ret", r) for r in rets] + [("Cond", k) for k in conds] or [("CallSite", sites[0])])
        def_label, d = rnd.choice([("CallSite", c) for c in sites] + [("Param", p) for p in params])
        from_path = rnd.choice(["r", "r.x"]) if def_label == "CallSite" else rnd.choice(["p0", "p0.a", "p1"])
        to_path = {"CallSite": rnd.choice(["a0", "a1.k", "r"]), "Ret": rnd.choice(["ret", "ret.x"]), "Cond": ""}[use_label]
        g.data(use_label, use, def_label, d, from_path, to_path, **({"status": "possible", "reason": "field-truncated"} if rnd.random() < 0.2 else {}))
    for _ in range(rnd.randint(0, 5) if conds else 0):
        g.controlled(*rnd.choice([("CallSite", c) for c in sites] + [("Ret", r) for r in rets] + [("Cond", k) for k in conds] or [("Cond", conds[0])]), rnd.choice(conds))
    cps = [g.cp(f"cp{i}", f"GET /k{i}") for i in range(rnd.randint(0, 2))]
    for cp in cps:
        for s in rnd.sample(syms, rnd.randint(0, 2)):
            g.provides(s, cp)
    for i in range(rnd.randint(0, 2)):
        lk = g.link(f"L{i}", provider=rnd.choice(syms), cp=rnd.choice(cps) if cps else None, status=rnd.choice(["confirmed", "possible"]), reason="key-match")
        for c in rnd.sample(sites, min(len(sites), rnd.randint(0, 3))):
            g.via(c, lk, status=rnd.choice(["confirmed", "possible"]), reason="key-match")
    anchors = [("Symbol", rnd.choice(syms), S.VALUE), ("Symbol", rnd.choice(syms), S.CONTRACT)]
    if params:
        anchors += [("Param", rnd.choice(params), S.path_state(rnd.choice(["p0", "p0.a", "p1"])))]
    if sites:
        anchors += [("CallSite", rnd.choice(sites), S.path_state("r"))]
    if rets:
        anchors += [("Ret", rnd.choice(rets), S.path_state("ret"))]
    if cps:
        anchors += [("ContractPoint", cps[0], S.CONTRACT)]
    return g, [((lab, ident), st) for lab, ident, st in anchors]


@pytest.mark.parametrize("seed", range(60))
def test_over_random_graphs_the_slice_variant_visits_exactly_the_slice_states_and_lists_exactly_its_items(seed):
    g, anchors = random_graph(seed)
    reader, table = MemoryReader(g), slice_table()
    for anchor, st in anchors:
        ref = I.walk(SliceReader(g.rows), anchor, st)
        got = bfs.walk(reader, anchor, st, table=table, bound=60)
        assert got.visited == ref.visited, (seed, anchor, st)
        assert {i.id for i in got.items()} == {i.symbol_id for i in ref.items()}, (seed, anchor, st)


def test_the_random_graphs_are_not_trivial():
    """The oracle comparison is only worth something if the walks reach something: across the seeds, many states and items and some cycle through a Ret."""
    reached = cyc = 0
    for seed in range(60):
        g, anchors = random_graph(seed)
        for anchor, st in anchors:
            w = bfs.walk(MemoryReader(g), anchor, st, table=slice_table(), bound=60)
            reached += len(w.items())
            cyc += any(w.best[k].edge is not None and w.best[k].edge.type == "DATA" for k in w.best)
    assert reached > 300 and cyc > 50


# --------------------------------------------------------------------------- the stored slice

SLICE_WALKS = ("p0", "p0.path", "p0.method", "p0.query", "p0.nothing", "buildUrl")
RECORDED_STATES = (98, 84, 84, 82, 81, 83)         # -p0-graph.md §3.3, V0 (as built)
RECORDED_V2_STATES = (19, 5, 5, 3, 2, 4)           # -p0-graph.md §3.3, with the V2 gate



def _load(name):
    spec = importlib.util.spec_from_file_location("slice_" + name, CPG / "spikes/vertical-slice" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["slice_" + name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def slice_rows():
    for name, sha in SHA256.items():
        assert hashlib.sha256((S14A / name).read_bytes()).hexdigest() == sha, f"{name}: not the artifact the oracle was pinned to"
    bundles = [[json.loads(line) for line in (S14A / name).read_text().splitlines()] for name in sorted(SHA256)]
    return _load("load").build_rows(bundles, _load("assemble").assemble(bundles))


@pytest.fixture(scope="module")
def slice_walks(slice_rows):
    qn = {s["qname"]: s["symbolId"] for s in slice_rows.nodes["Symbol"]}
    api, bu = qn["salesperson/src/api/client.ts::apiFetch"], qn["salesperson/src/api/client.ts::buildUrl"]
    index = {p["pointId"]: p["index"] for p in slice_rows.nodes["Param"]}
    p0 = next(e["t"] for e in slice_rows.edges[("Symbol", "HAS_PARAM", "Param")] if e["s"] == api and index[e["t"]] == 0)
    anchors = {path: (("Param", p0), S.path_state(path)) for path in SLICE_WALKS[:5]}
    anchors["buildUrl"] = (("Symbol", bu), S.VALUE)
    return SimpleNamespace(rows=slice_rows, anchors=anchors, api=api, reader=MemoryReader(SimpleNamespace(rows=slice_rows)))


slice_only = [pytest.mark.slice, pytest.mark.skipif(not S14A.is_dir() or not list(S14A.glob("*.bundle.jsonl")),
                                                   reason="stored s14a slice bundles absent (cpg/.ccg-work is gitignored): UNRUN, not green")]


def _marked(fn):
    for m in slice_only:
        fn = m(fn)
    return fn


@_marked
def test_the_slice_variant_reproduces_the_recorded_slice_figures_and_the_slice_walks_own_sets(slice_walks):
    states, items = [], []
    for name in SLICE_WALKS:
        anchor, st = slice_walks.anchors[name]
        ref = I.walk(SliceReader(slice_walks.rows), anchor, st)
        got = bfs.walk(slice_walks.reader, anchor, st, table=slice_table(), bound=60)
        assert got.visited == ref.visited and {i.id for i in got.items()} == {i.symbol_id for i in ref.items()}, name
        states.append(len(got.visited)), items.append(len(got.items()))
    assert tuple(states) == RECORDED_STATES and items == [21] * 6


@_marked
def test_the_v2_gated_rows_give_the_recorded_state_counts_and_the_one_item_the_gate_leaves(slice_walks):
    got = [bfs.walk(slice_walks.reader, *slice_walks.anchors[name], table=v2_like_table(), bound=60) for name in SLICE_WALKS]
    assert tuple(len(w.visited) for w in got) == RECORDED_V2_STATES and [len(w.items()) for w in got] == [1] * 6


FUNCTION_LEVEL = "a caller the slice listed only through a function-level continuation: the function holds the changed point but its return is not reached"
CALLEE = "a callee the change enters through CALLS ->/HAS_PARAM -> (an argument reaching an index-matched parameter), which the slice never walked"
CONTROL = "an owner reached through CONTROLLED_BY then a Ret: a callee's guarded return reaches its caller (possible: the caller site is a name candidate)"
_CLIENT, _ENDPOINTS, _HOOKS = "salesperson/src/api/client.ts::", "salesperson/src/api/endpoints.ts::", "salesperson/src/api/hooks.ts::"
#: the 20 function-level items the slice lists for every walk that reaches ``apiFetch``'s callers, none of which the precise rows list
_TWENTY = {*(_ENDPOINTS + n for n in ("joinSession", "getCatalog", "presenterSession", "presenterResetAll", "getMessages", "participantGet", "getHealth", "advanceOrder",
                                      "presenterParticipants", "resetMine", "getState", "postMessage")),
           *(_HOOKS + n for n in ("useMessages.queryFn", "useShopState.queryFn", "useCatalog.queryFn", "usePostMessage.onError", "useHealth.queryFn", "useJoin.mutationFn",
                                  "usePostMessage.mutationFn", "useAdvanceOrder.mutationFn"))}
#: walk -> (dropped qname set, {added qname: reason}): the named delta against the slice's 21-item function-level set (``-p0-graph.md`` §3.3); every walk is listed, none special-cased
DELTAS = {
    "p0": ({_HOOKS + "useMessages.queryFn", _HOOKS + "useShopState.queryFn", _HOOKS + "useCatalog.queryFn", _HOOKS + "usePostMessage.onError"},
           {_CLIENT + "ApiError.<init>": CALLEE, _CLIENT + "timedSignal": CALLEE, _CLIENT + "buildUrl": CALLEE,
            _HOOKS + "usePresenterResetAll.onError": CONTROL, _HOOKS + "usePresenterParticipants": CONTROL}),
    "p0.path": (_TWENTY, {_CLIENT + "ApiError.<init>": CALLEE, _CLIENT + "buildUrl": CALLEE}),
    "p0.method": (_TWENTY, {_CLIENT + "ApiError.<init>": CALLEE}),
    "p0.query": (_TWENTY, {_CLIENT + "buildUrl": CALLEE}),
    "p0.nothing": (_TWENTY, {}),
    "buildUrl": (_TWENTY, {}),
}
#: regression pin (what the run observed under the shipped table at bound 2, not an oracle): (items, states, unexplored, maximum item distance)
SHIPPED_OBSERVED = {"p0": (22, 157, 5, 2), "p0.path": (3, 18, 0, 1), "p0.method": (2, 10, 0, 1), "p0.query": (2, 22, 0, 1), "p0.nothing": (1, 2, 0, 0), "buildUrl": (1, 4, 0, 1)}


@_marked
def test_each_slice_walk_is_pinned_as_a_named_delta_against_the_slices_21_item_set_and_as_observed_counts(slice_walks):
    qname = {s["symbolId"]: s["qname"] for s in slice_walks.rows.nodes["Symbol"]}
    observed = {}
    for name in SLICE_WALKS:
        anchor, st = slice_walks.anchors[name]
        slice_items = {i.id for i in bfs.walk(slice_walks.reader, anchor, st, table=slice_table(), bound=60).items()}
        w = bfs.walk(slice_walks.reader, anchor, st)
        got = {i.id: i for i in w.items()}
        dropped, added = {qname[i] for i in slice_items - set(got)}, {qname[i]: got[i] for i in set(got) - slice_items}
        want_dropped, want_added = DELTAS[name]
        assert dropped == want_dropped and set(added) == set(want_added), name
        for q, item in added.items():                       # the reason is checked against the chain, not only written down
            edges = [(s["edge"]["type"], s["edge"]["dir"]) for s in item.chain[1:]]
            assert (("CALLS", "->") in edges and ("HAS_PARAM", "->") in edges) if want_added[q] == CALLEE else ("CONTROLLED_BY", "<-") in edges, (name, q)
        observed[name] = (len(got), len(w.visited), w.unexplored, max([i.distance for i in got.values()] or [0]))
    assert observed == SHIPPED_OBSERVED


@_marked
def test_api_fetch_stays_confirmed_as_the_direct_resolved_caller_of_the_anchor_symbol(slice_walks):
    w = bfs.walk(slice_walks.reader, *slice_walks.anchors["buildUrl"])
    (item,) = w.items()
    assert item.id == slice_walks.api and (item.status, item.distance, item.possible_hops) == ("confirmed", 1, 0)
    assert [s["edge"]["type"] for s in item.chain[1:]] == ["CALLS", "HAS_CALL"]
