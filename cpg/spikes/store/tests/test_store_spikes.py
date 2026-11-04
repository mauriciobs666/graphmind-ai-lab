"""S20/S21 (P0-15): the store spikes' own code -- key guards, the synthetic graph, the SSpaths item derivation, the
BFS instrumentation. Pure tests run everywhere; ``-m live`` ones touch only ``ccg_p0spike__test_*`` keys on
``ccg-falkordb`` (127.0.0.1:6390) and delete them (S20/S21 live rules)."""
import itertools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import bench_reach as R  # noqa: E402
import bfs as B  # noqa: E402
import swap_probe as W  # noqa: E402
import synth as S  # noqa: E402

I = B.I


# ---------------------------------------------------------------------------------------------
# Key guards (live rules: only ccg_p0spike and ccg_p0spike__*; swap probes only ccg_p0spike__swap_*).
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["ccg_p0spike", "ccg_p0spike__build_synth", "ccg_p0spike__x"])
def test_spike_key_accepts_own_keys(key):
    assert S.check_spike_key(key) == key


@pytest.mark.parametrize("key", ["ccg_p0slice", "ccg_p0slice__build_x", "ccg_p0spike_x", "ccg_p0spikeX", "xccg_p0spike", "", None,
                                 "ccg_p0spike__", "ccg_p0spike__a b", "telemetry{ccg_p0spike}", "falkordb", "ccg_p0spike\n"])
def test_spike_key_refuses_everything_else(key):
    with pytest.raises(S.SynthError):
        S.check_spike_key(key)


@pytest.mark.parametrize("key", ["ccg_p0spike__swap_live", "telemetry{ccg_p0spike__swap_live}", "ccg_p0spike__swap_x1"])
def test_swap_guard_accepts_swap_graphs_and_their_telemetry_stream(key):
    assert W.guard(key) == key


@pytest.mark.parametrize("key", ["ccg_p0spike", "ccg_p0spike__build_synth", "telemetry{ccg_p0spike}", "telemetry{ccg_p0slice}",
                                 "telemetry{ccg_p0spike__swap_x", "ccg_p0spike__swap_x}", "ccg_p0spike__swap_", "ccg_p0slice", ""])
def test_swap_guard_refuses_everything_else(key):
    with pytest.raises(S.SynthError):
        W.guard(key)


def test_spike_store_is_the_loaders_zero_retry_connection_and_refuses_the_dev_port():
    import load as L
    assert issubclass(S.SpikeStore, L.LiveStore)
    with pytest.raises(L.LoadError):
        S.SpikeStore("127.0.0.1", 6379)


def test_spike_store_uses_the_spike_guard_not_the_loaders():
    st = S.SpikeStore.__new__(S.SpikeStore)  # no connection: every method must refuse before touching it
    for call in (lambda: st.exists("ccg_p0slice"), lambda: st.query("ccg_p0slice", "RETURN 1"),
                 lambda: st.ro_query("ccg_p0slice", "RETURN 1"), lambda: st.rename("ccg_p0spike", "ccg_p0slice"),
                 lambda: st.delete("ccg_p0slice"), lambda: st.create_unique_constraint("ccg_p0slice", "Unit", "unitId"),
                 lambda: st.memory("ccg_p0slice"), lambda: st.constraints("ccg_p0slice")):
        with pytest.raises(S.SynthError):
            call()


# ---------------------------------------------------------------------------------------------
# The synthetic graph.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth60():
    return S.Synth(60)


def test_generation_is_deterministic(synth60):
    again = S.Synth(60)
    for group in (("CallSite", "CALLS", "Symbol"), ("CallSite", "DATA", "CallSite"), ("Cond", "CONTROLLED_BY", "Cond")):
        assert list(synth60.edges(group)) == list(again.edges(group))
    assert list(synth60.nodes("Symbol")) == list(again.nodes("Symbol"))
    assert S.Synth(60, seed=1).anchors() != synth60.anchors()


def test_every_edge_endpoint_exists_and_no_edge_is_duplicated(synth60):
    ids = {l: {r[{"Repo": "repoId", "Unit": "unitId", "File": "fileId", "Symbol": "symbolId", "CallSite": "callSiteId",
                  "Name": "nameKey"}.get(l, "pointId")] for r in synth60.nodes(l)} for l in S.NODE_ORDER}
    for group in S.EDGE_ORDER:
        seen = set()
        for r in synth60.edges(group):
            assert r["s"] in ids[group[0]], (group, r)
            assert r["t"] in ids[group[2]], (group, r)
            key = (r["s"], r["t"], tuple(sorted((r.get("p") or {}).items())))
            assert key not in seen, (group, r)
            seen.add(key)


def test_densities_are_near_the_v2_estimate_per_file():
    s = S.Synth(200)
    nodes = sum(1 for l in S.NODE_ORDER for _ in s.nodes(l)) / 200
    edges = sum(1 for g in S.EDGE_ORDER for _ in s.edges(g)) / 200
    assert 420 <= nodes <= 490 and 2200 <= edges <= 2600, (nodes, edges)  # graph note §6.4: ~455 / ~2,400


def test_the_planted_cycles_are_present_and_closed(synth60):
    cycles = synth60.cond_cycles()
    assert cycles, "no cond->cond CONTROLLED_BY cycle planted"
    cc = {(r["s"], r["t"]) for r in synth60.edges(("Cond", "CONTROLLED_BY", "Cond"))}
    for ring in cycles:
        assert len(ring) >= 3
        for a, b in zip(ring, ring[1:] + ring[:1]):
            assert (a, b) in cc, (a, b)
    data = {(r["s"], r["t"]) for g in (("CallSite", "DATA", "CallSite"),) for r in synth60.edges(g)}
    for pl in synth60.planted():
        hub_call = S.Synth.call_id(pl["function"], pl["hubCall"])
        assert (hub_call, hub_call) in data  # the self-edge
        two = [u for (u, d) in data if d == hub_call and u != hub_call and (hub_call, u) in data]
        assert two, "no two-point DATA cycle through the hub call"


def test_a_planted_function_calls_its_hub_so_walks_from_the_hub_reach_the_cycle(synth60):
    calls = {(r["s"], r["t"]) for r in synth60.edges(("CallSite", "CALLS", "Symbol"))}
    for pl in synth60.planted():
        assert (S.Synth.call_id(pl["function"], pl["hubCall"]), pl["hub"]) in calls


def test_anchors_span_hub_and_non_hub(synth60):
    a = S.Synth(300).anchors(50)
    assert len(a) == 50 and len({x["symbolId"] for x in a}) == 50
    cls = {c: [x["indegree"] for x in a if x["class"] == c] for c in ("hub", "mid", "low")}
    assert len(cls["hub"]) == 15 and min(cls["hub"]) > max(cls["mid"]) and max(cls["low"]) <= 2


def test_call_edges_carry_the_sspaths_weight_and_possible_ones_say_so(synth60):
    rows = list(synth60.edges(("CallSite", "CALLS", "Symbol")))
    assert all("w" in r["p"] for r in rows)
    assert {r["p"]["w"] for r in rows if "status" in r["p"]} == {1000}
    assert {r["p"]["w"] for r in rows if "status" not in r["p"]} == {1}
    assert all(r["p"] == {"w": 1} for r in synth60.edges(("Symbol", "HAS_CALL", "CallSite")))


def test_batches_never_exceed_batch_max():
    sizes = [len(c) for c in S.stream_batches(iter({"i": i} for i in range(45_000)))]
    assert sizes == [20_000, 20_000, 5_000] and max(sizes) <= S.BATCH_MAX
    assert [len(c) for c in S.stream_batches(iter(range(10)), 3)] == [3, 3, 3, 1]
    for bad in (0, S.BATCH_MAX + 1):
        with pytest.raises(S.SynthError):
            list(S.stream_batches(iter(range(3)), bad))


def test_load_synth_writes_every_row_in_batches_with_the_timeout_and_checks_created_counts():
    class Res:
        def __init__(self, n):
            self.relationships_created = n

    class Store:
        def __init__(self, short=False):
            self.q, self.short, self.tm = [], short, 7000

        def timeout_max(self):
            return self.tm

        def exists(self, key):
            return False

        def query(self, key, cypher, params=None, timeout=None):
            self.q.append((key, cypher, len(params["rows"]) if params and "rows" in params else None, timeout))
            n = len(params["rows"]) if params and "rows" in params else 0
            return Res(n - 1 if self.short and "CREATE (a)-[" in cypher else n)

        def create_unique_constraint(self, *a):
            self.q.append(("constraint", "constraint", None, None))

        def constraints(self, key):
            return [{"label": l, "properties": [p], "status": "OPERATIONAL"} for l, p in S.L.constraint_statements()]

        def memory(self, key):
            return {"total_graph_sz_mb": 1}

        def used_memory(self):
            return 0

        def rename(self, a, b):
            self.q.append(("rename", "rename", None, None))

    st = Store()
    synth = S.Synth(20)
    rep = S.load_synth(st, synth, "ccg_p0spike__build_t", "ccg_p0spike__t", batch_size=500, log=lambda *_: None)
    writes = [q for q in st.q if q[2] is not None]
    assert writes and max(q[2] for q in writes) <= 500 and rep["maxBatch"] <= 500
    assert {q[3] for q in writes} == {7000}                      # the explicit timeout rides every load query (D16)
    assert sum(1 for q in st.q if q[0] == "rename") == 1 and st.q[-1][0] == "rename"   # RENAME last
    assert rep["nodeTotal"] == sum(1 for l in S.NODE_ORDER for _ in synth.nodes(l))
    assert rep["edgeTotal"] == sum(1 for g in S.EDGE_ORDER for _ in synth.edges(g))
    # indexes come first, constraints after every write
    first_write = next(i for i, q in enumerate(st.q) if q[2] is not None)
    assert all("INDEX" in q[1] for q in st.q[:first_write])
    assert max(i for i, q in enumerate(st.q) if q[2] is not None) < min(i for i, q in enumerate(st.q) if q[0] == "constraint")
    with pytest.raises(S.SynthError, match="edges created"):
        S.load_synth(Store(short=True), synth, "ccg_p0spike__build_t", "ccg_p0spike__t", log=lambda *_: None)
    with pytest.raises(S.SynthError):
        S.load_synth(st, synth, "ccg_p0slice__build_x", "ccg_p0spike__t", log=lambda *_: None)


# ---------------------------------------------------------------------------------------------
# SSpaths rows -> items, comparison, decision rule.
# ---------------------------------------------------------------------------------------------

CONF = "confirmed"


def row(ids, types, statuses):
    return [ids, types, statuses, 0.0]


def test_items_from_paths_labels_by_bfs_v1_rule_and_keeps_the_best_path_per_symbol():
    rows = [
        row(["A", "c1"], ["CALLS"], [CONF]),                                            # ends on a call site: not an item
        row(["A", "c1", "B"], ["CALLS", "HAS_CALL"], [CONF, CONF]),                      # B at distance 1, confirmed
        row(["A", "c1", "B", "c2", "C"], ["CALLS", "HAS_CALL", "CALLS", "HAS_CALL"], [CONF] * 4),   # C: hop 2 is function-level
        row(["A", "c1", "B", "c9", "C"], ["CALLS", "HAS_CALL", "CALLS", "HAS_CALL"], [CONF, CONF, "possible", CONF]),  # worse path to C? same poss
        row(["A", "c3", "D"], ["CALLS", "HAS_CALL"], ["possible", CONF]),                 # D: first hop possible
        row(["A", "c4", "A"], ["CALLS", "HAS_CALL"], [CONF, CONF]),                      # the anchor itself: never an item
    ]
    items = {r[0]: r for r in R.items_from_paths("A", rows)}
    assert set(items) == {"B", "C", "D"}
    assert items["B"][1:5] == ["confirmed", 1, 0, 2]
    assert items["C"][1:5] == ["possible", 2, 1, 4]         # one possible hop: the function-level continuation
    assert items["C"][5] == ["A", "c1", "B", "c2", "C"]       # tie on cost broken by node ids ("c2" < "c9")
    assert items["D"][1:5] == ["possible", 1, 1, 2]


def test_items_from_paths_prefers_fewer_possible_hops_then_distance_then_length():
    p_long = row(["A", "x", "B", "y", "C"], ["CALLS", "HAS_CALL", "CALLS", "HAS_CALL"], [CONF] * 4)
    p_short_possible_first = row(["A", "z", "C"], ["CALLS", "HAS_CALL"], ["possible", CONF])
    item = R.items_from_paths("A", [p_short_possible_first, p_long])
    c = next(r for r in item if r[0] == "C")
    # both have one possible hop; the shorter distance wins
    assert c[5] == ["A", "z", "C"] and c[2] == 1


def test_items_from_paths_is_independent_of_row_order():
    rows = [row(["A", f"c{i}", f"S{i % 3}"], ["CALLS", "HAS_CALL"], [CONF, CONF]) for i in range(9)]
    assert R.items_from_paths("A", rows) == R.items_from_paths("A", rows[::-1])
    assert R.items_from_paths("A", rows) == R.items_from_paths("A", rows[4:] + rows[:4])


def test_compare_counts_missing_extra_and_agreement():
    bfs_items = [["B", "confirmed", 1, 0, 2, [[["Symbol", "A"], ["value", ""], None], [["CallSite", "c1"], ["path", "r"], {}], [["Symbol", "B"], ["function", ""], {}]]],
                 ["C", "possible", 2, 1, 4, [[["Symbol", "A"], [], None]]], ["E", "confirmed", 1, 0, 2, []]]
    ss = [["B", "confirmed", 1, 0, 2, ["A", "c1", "B"]], ["C", "confirmed", 2, 0, 4, ["A", "q"]], ["F", "confirmed", 1, 0, 2, []]]
    c = R.compare(bfs_items, ss)
    assert (c["bfsItems"], c["sspathsItems"], c["common"], c["missingFromSSpaths"], c["extraInSSpaths"], c["itemSetEqual"]) == (3, 3, 2, 1, 1, False)
    assert (c["statusAgree"], c["distanceAgree"], c["chainAgree"]) == (1, 2, 1)
    assert R.compare(bfs_items[:1], ss[:1])["itemSetEqual"] is True


def test_percentile_is_nearest_rank():
    xs = list(range(1, 101))
    assert (R.percentile(xs, 50), R.percentile(xs, 95), R.percentile(xs, 99), R.percentile(xs, 100)) == (50, 95, 99, 100)
    assert R.percentile([7], 99) == 7 and R.percentile([3, 1, 2], 50) == 2


def _dec(identical=True, equal=True, bfs=1.0, ss=0.5):
    return [{"sspathsRepeatIdentical": identical, "compare": {"itemSetEqual": equal}, "bfsSeconds": [bfs] * 20, "sspathsSeconds": [ss] * 20}]


def test_decision_rule_needs_all_three_conditions_and_the_30_percent_margin():
    assert R.adopt_decision(_dec())["adopt"] is True
    assert R.adopt_decision(_dec(identical=False))["adopt"] is False      # rule 1
    assert R.adopt_decision(_dec(equal=False))["adopt"] is False          # rule 2
    assert R.adopt_decision(_dec(ss=0.71))["adopt"] is False              # rule 3: 29 % lower is not enough
    assert R.adopt_decision(_dec(ss=0.70))["adopt"] is True               # exactly 30 % lower is
    assert R.adopt_decision(_dec(ss=1.5))["rule3_p95_30pct_lower"] is False


# ---------------------------------------------------------------------------------------------
# bfs.py: the instrumentation around impact.walk (hand-built graph through the reader seam).
# ---------------------------------------------------------------------------------------------


class Mem(B.SplitRows):
    """An in-memory reader with ``bfs-v1``'s seam plus ``controlled_in``; ``rows_limit`` simulates the reply cap."""

    def __init__(self, calls=(), owners=(), data=(), controlled=(), limit=None):
        self.calls, self.owners, self.data, self.controlled = list(calls), list(owners), list(data), list(controlled)
        self.limit = limit
        self.run = lambda cypher, params: []

    def calls_in(self, ids):
        return sorted((s, c, st) for s, c, st in self.calls if s in ids)

    def owner_of(self, label, ids):
        return sorted((p, s) for l, p, s in self.owners if l == label and p in ids)

    def data_in(self, label, ids):
        if label not in I.DEF_LABELS:
            return []
        return sorted((d, ul, u, fp, tp, st) for dl, d, ul, u, fp, tp, st in self.data if dl == label and d in ids)

    def to_in(self, ids):
        return []

    def via_in(self, ids):
        return []

    def provides_in(self, ids):
        return []

    def provides_out(self, ids):
        return []

    def controlled_in(self, ids):
        return sorted((c, l, p) for c, l, p in self.controlled if c in ids)


def cycle_graph():
    """c1 (in B) calls the anchor A; c1's result reaches cond k1; k1 -> k2 -> k3 -> k1 guard (cond->cond) and k3 guards c2 in C."""
    return Mem(calls=[("A", "c1", "confirmed")],
               owners=[("CallSite", "c1", "B"), ("Cond", "k1", "B"), ("Cond", "k2", "B"), ("Cond", "k3", "B"), ("CallSite", "c2", "C")],
               data=[("CallSite", "c1", "Cond", "k1", "r", "", "confirmed")],
               controlled=[("k1", "Cond", "k2"), ("k2", "Cond", "k3"), ("k3", "Cond", "k1"), ("k3", "CallSite", "c2")])


def run_walk(reader, *, controlled_by, cap=None):
    t = B.Tracker(cap=cap)
    with B.instrumented(t, controlled_by=controlled_by):
        return I.walk(reader, ("Symbol", "A"), I.VALUE, max_levels=(cap or 60) + 2), t


def test_a_cond_to_cond_controlled_by_cycle_terminates_and_each_cond_is_visited_once():
    res, t = run_walk(cycle_graph(), controlled_by=True)
    conds = sorted(k for k in res.visited if k[0] == "Cond")
    assert sorted({k[1] for k in conds}) == ["k1", "k2", "k3"]
    assert [k[1] for k in conds].count("k2") == [k[1] for k in conds].count("k3") == 1 and len(conds) <= 4  # k1: a DATA arrival and a CONTROLLED_BY one
    assert {i.symbol_id for i in res.items()} == {"B", "C"}                       # the guarded call site c2 reaches C
    assert res.levels < 20 and len(t.levels) == res.levels


def test_without_the_controlled_by_row_the_cycle_is_never_walked():
    res, _ = run_walk(cycle_graph(), controlled_by=False)
    assert {k[1] for k in res.visited if k[0] == "Cond"} == {"k1"}
    assert {i.symbol_id for i in res.items()} == {"B"}


def test_the_walk_restores_impact_py_after_instrumenting():
    expand, chunk = I._expand, I.CHUNK
    run_walk(cycle_graph(), controlled_by=True)
    assert I._expand is expand and I.CHUNK == chunk
    with pytest.raises(RuntimeError):
        with B.instrumented(B.Tracker(), controlled_by=True, chunk=5):
            raise RuntimeError
    assert I._expand is expand and I.CHUNK == chunk


def test_the_level_cap_stops_the_walk_by_its_own_termination_test():
    g = Mem(calls=[("A", "c1", "confirmed"), ("B", "c2", "confirmed"), ("C", "c3", "confirmed")],
            owners=[("CallSite", "c1", "B"), ("CallSite", "c2", "C"), ("CallSite", "c3", "D")])
    full, _ = run_walk(g, controlled_by=False)
    assert {i.symbol_id for i in full.items()} == {"B", "C", "D"}
    for cap, want in ((2, {"B"}), (4, {"B", "C"}), (6, {"B", "C", "D"})):
        capped, t = run_walk(g, controlled_by=False, cap=cap)
        assert {i.symbol_id for i in capped.items()} == want, cap
        assert len(t.levels) == cap and t.level == cap + 1            # exactly `cap` levels expanded, then the cap stopped it


def test_segment_reader_follows_only_calls_and_has_call():
    class Q(B.SegmentReader):
        pass

    seen = []
    r = Q(lambda cypher, params: seen.append(cypher) or [])
    assert r.owner_of("Param", ["p"]) == [] and r.owner_of("Cond", ["k"]) == [] and r.owner_of("Ret", ["r"]) == []
    assert r.data_in("CallSite", ["c"]) == [] and r.to_in(["s"]) == [] and r.via_in(["l"]) == []
    assert r.provides_in(["x"]) == [] and r.provides_out(["s"]) == [] and seen == []
    r.owner_of("CallSite", ["c"])
    r.calls_in(["s"])
    assert len(seen) == 2


def test_split_rows_halves_a_full_reply_and_raises_only_when_one_id_fills_it(monkeypatch):
    monkeypatch.setattr(I, "RESULT_SET_LIMIT", 10)
    calls = []

    def run(cypher, params):
        ids = params["ids"]
        calls.append(len(ids))
        return [[i, k] for i in ids for k in range(3)]       # 3 rows per id

    r = B.SegmentReader(run)
    rows = r._rows("q", [f"s{i}" for i in range(8)])         # 24 rows >= 10: split to 4 -> 12 >= 10: split to 2 -> 6 rows
    assert len(rows) == 24 and sorted(rows) == rows and r.splits == 3 and calls[0] == 8
    assert len({tuple(x) for x in rows}) == 24               # nothing lost, nothing doubled

    def fat(cypher, params):
        return [[params["ids"][0], k] for k in range(10)]

    with pytest.raises(I.ImpactError, match="one id fills"):
        B.SegmentReader(fat)._rows("q", ["only"])


def test_provenance_names_the_file_and_its_function_state():
    p = B.impact_provenance()
    assert p["file"] == "cpg/spikes/vertical-slice/impact.py" and len(p["sha256"]) == 64 and p["hasFunctionState"] is True
    assert p["algorithm"] == "bfs-v1"


def test_canonical_items_digest_is_stable_and_sensitive():
    res, _ = run_walk(cycle_graph(), controlled_by=True)
    a, b = B.canonical_items(res), B.canonical_items(run_walk(cycle_graph(), controlled_by=True)[0])
    assert B.digest(a) == B.digest(b)
    b[0][2] += 1
    assert B.digest(a) != B.digest(b)
    c = B.canonical_items(res)
    c[0][5][-1][2] = {"type": "X"}
    assert B.digest(a) != B.digest(c)                        # the chain is in the digest


# ---------------------------------------------------------------------------------------------
# live: a 12-file graph on a ccg_p0spike__test_* key.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_store():
    try:
        return S.SpikeStore("127.0.0.1", 6390)
    except FileNotFoundError as exc:
        pytest.skip(f"ccg-falkordb admin password file not readable ({exc.filename})")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ccg-falkordb unreachable ({type(exc).__name__}); start it with cpg/infra/ccg-falkordb.sh up")


@pytest.mark.live
def test_live_small_synth_loads_equals_generator_counts_and_both_walks_agree(live_store):
    st = live_store
    key, build = "ccg_p0spike__test_small", "ccg_p0spike__build_testsmall"
    synth = S.Synth(40)
    for k in (key, build):
        if st.exists(k):
            st.delete(k)
    try:
        rep = S.load_synth(st, synth, build, key, log=lambda *_: None)
        assert all(c[1] == "OPERATIONAL" for c in rep["constraints"])
        n = st.ro_query(key, "MATCH (n) RETURN count(n)").result_set[0][0]
        e = st.ro_query(key, "MATCH ()-[e]->() RETURN count(e)").result_set[0][0]
        assert (n, e) == (rep["nodeTotal"], rep["edgeTotal"])
        g = st.db.select_graph(key)
        order = synth.hubs()
        for i in order[:3] + order[len(order) // 2:len(order) // 2 + 3]:
            a = {"symbolId": synth.sym_ids[i]}
            rb = R.run_bfs(g, a["symbolId"], "segment", 6, 2)
            rs = R.run_sspaths(g, a["symbolId"], 3, 2)
            assert rb["identical"] and rs["identical"]
            if not rs["truncated"]:
                assert R.compare(rb["items"], rs["items"])["itemSetEqual"], a
        # the planted cond cycle is walked by the full mode and terminates
        pl = synth.planted()[0]
        full = B.answer(g, pl["hub"], mode="full", levels=6)
        assert full["levels"] <= 8 and full["nItems"] > 0
    finally:
        for k in (key, build):
            if st.exists(k):
                st.delete(k)


# ---------------------------------------------------------------------------------------------
# Gate follow-ups (m3): the truncation flag, summarize(), the EVAL swap guard.
# ---------------------------------------------------------------------------------------------


class _FakeRes:
    def __init__(self, n):
        self.result_set = [[["A", "c1", "B"], ["CALLS", "HAS_CALL"], ["confirmed", "confirmed"], 2.0] for _ in range(n)]


class _FakeGraph:
    def __init__(self, n):
        self.n = n

    def ro_query(self, cypher, params, timeout=None):
        return _FakeRes(self.n)


@pytest.mark.parametrize("rows,truncated", [(9_999, False), (10_000, True), (10_001, True), (0, False)])
def test_sspaths_answer_flags_a_reply_that_reached_the_row_cap(rows, truncated):
    got = R.sspaths_answer(_FakeGraph(rows), "A", 3)
    assert got["truncated"] is truncated and got["rows"] == rows and R.TRUNCATED_AT == 10_000


def _seg(depth, cls, indeg, bfs_items, ss_items, missing, extra, equal, truncated, bfs_s, ss_s, repeats=2):
    return {"kind": "segment", "depth": depth, "anchor": {"symbolId": f"{cls}{indeg}", "class": cls, "indegree": indeg}, "repeats": repeats,
            "bfs": {"identical": True, "nItems": bfs_items, "seconds": bfs_s, "reads": {"HAS_CALL": [1.0, 2.0], "other": [10.0]},
                    "levelMs": [5.0, 6.0], "splits": 0, "states": 7, "levels": 3},
            "sspaths": {"identical": True, "nItems": ss_items, "truncated": truncated, "seconds": ss_s},
            "compare": {"bfsItems": bfs_items, "sspathsItems": ss_items, "common": bfs_items - missing, "missingFromSSpaths": missing,
                        "extraInSSpaths": extra, "itemSetEqual": equal, "statusAgree": bfs_items - missing,
                        "distanceAgree": bfs_items - missing - 1, "chainAgree": bfs_items - missing - 2}}


def test_summarize_reproduces_hand_computed_figures(tmp_path):
    import json
    recs = [
        {"kind": "meta", "impact": {"sha256": "x"}, "timeoutMax": 0},
        _seg(3, "hub", 900, 100, 40, 60, 0, False, True, [4.0, 5.0], [1.0, 1.0]),
        _seg(3, "hub", 800, 50, 30, 25, 5, False, True, [2.0, 3.0], [1.0, 1.0]),
        _seg(3, "low", 1, 4, 4, 0, 0, True, False, [0.1, 0.1], [0.05, 0.05]),
        {"kind": "full", "levels": 4, "anchor": {"symbolId": "h", "class": "hub", "indegree": 9}, "repeats": 2,
         "bfs": {"identical": True, "nItems": 9, "seconds": [1.0, 3.0], "splits": 2, "states": 77, "levels": 5,
                 "reads": {"DATA": [20.0]}, "levelMs": [7.0]}},
        {"kind": "end"},
    ]
    path = tmp_path / "b.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in recs))
    out = R.summarize(path)
    d3 = out["segment"][3]
    assert d3["anchors"] == 3 and d3["itemSetEqual"] == "1/3" and d3["bfsRepeatIdentical"] == "3/3"
    assert d3["statusAgreeOnCommon"] == "69/69" and d3["distanceAgreeOnCommon"] == "66/69" and d3["chainAgreeOnCommon"] == "63/69"
    hub = d3["perClass"]["hub"]
    assert hub["missingFraction"] == round(85 / 150, 4)            # missing / BFS items, not extra / anything
    assert hub["sspathsTruncated"] == 2 and hub["itemSetEqual"] == 0 and hub["bfsItemsMedian"] == 75.0
    assert d3["perClass"]["low"]["itemSetEqual"] == 1 and d3["perClass"]["low"]["missingFraction"] == 0.0
    assert d3["bfsP95"] == 5.0 and d3["sspathsP95"] == 1.0         # nearest rank over 6 samples
    assert d3["decisionRule"]["rule2_item_set_equal"] is False and d3["decisionRule"]["adopt"] is False
    assert out["full"][4]["splits"] == 2 and out["full"][4]["maxStates"] == 77 and out["full"][4]["max"] == 3.0
    assert out["perReadMs"]["HAS_CALL"]["n"] == 6 and out["perReadMs"]["DATA"]["max"] == 20.0
    assert out["perReadMsAll"]["n"] == 10 and out["perLevelMs"]["n"] == 7


def test_swap_script_checks_all_three_keys_before_its_first_rename():
    text = W.SWAP_SCRIPT
    first_rename = text.index("redis.call('RENAME'")
    for guard_text in ("EXISTS', KEYS[2]) == 0 then return 'refused: build absent'",
                       "EXISTS', KEYS[1]) == 0 then return 'refused: live absent'",
                       "EXISTS', KEYS[3]) == 1 then return 'refused: prev present'"):
        assert guard_text in text and text.index(guard_text) < first_rename
    assert text.count("redis.call('RENAME'") == 2 and "refused" not in W.OLD_SCRIPT.replace("refused: build absent", "")
    # P2-m2: live moves to prev BEFORE build moves to live (the reverse destroys the old snapshot and leaves prev = new)
    assert text.index("redis.call('RENAME', KEYS[1], KEYS[3])") < text.index("redis.call('RENAME', KEYS[2], KEYS[1])")


@pytest.mark.live
@pytest.mark.parametrize("case,expect,after", [
    ("happy", "swapped", {"live": "new", "build": None, "prev": "old"}),
    ("build-absent", "refused: build absent", {"live": "old", "build": None, "prev": None}),
    ("live-absent", "refused: live absent", {"live": None, "build": "new", "prev": None}),
    ("prev-present", "refused: prev present", {"live": "old", "build": "new", "prev": "older"}),
])
def test_live_swap_script_refuses_before_moving_anything(live_store, case, expect, after):
    p = W.Probe(live_store)
    live, build, prev = (W.PREFIX + n + "t" for n in ("live", "build", "prev"))
    p.wipe(live, build, prev)
    try:
        if case != "live-absent":
            p.make(live, "old")
        if case != "build-absent":
            p.make(build, "new")
        if case == "prev-present":
            p.make(prev, "older")
        got = p.cmd("EVAL", W.SWAP_SCRIPT, 3, live, build, prev, keys=(live, build, prev))
        assert (got.decode() if isinstance(got, bytes) else got) == expect
        assert dict(zip((("live", "build", "prev")), p.state(live, build, prev).values())) == after
    finally:
        p.wipe(live, build, prev)
