"""S13 (P0-12) tests for ``load.py`` and ``impact.py``.

Pure-logic tests run everywhere (a fake store, a hand-built graph). The ``live`` tests need the
dedicated ``ccg-falkordb`` (127.0.0.1:6390, reservation R-B) and the gitignored S11 bundles
(``cpg/.ccg-work/vertical-slice/run1``); they skip, with the reason, when either is missing:

    cd cpg && .venv/bin/python -m pytest spikes/vertical-slice/tests/test_live_slice.py          # pure
    cd cpg && .venv/bin/python -m pytest -m live spikes/vertical-slice/tests/test_live_slice.py  # + live

The live run loads ``ccg_p0slice`` twice (SL-6), checks SL-3/SL-4, measures SL-5, and writes the
measurements S14 quotes to ``<run dir>/s13-measurements.json``. Only the keys ``ccg_p0slice`` and
``ccg_p0slice__build_*`` are touched.
"""
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from redis.backoff import NoBackoff
from redis.retry import Retry

HERE = Path(__file__).resolve().parents[1]
CPG = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CPG))

import impact as I  # noqa: E402
import load as L  # noqa: E402
from assemble import assemble  # noqa: E402
from ccg.bundle import read_bundle  # noqa: E402
from ccg.ids import contract_point_id, link_id, name_key  # noqa: E402

RUN_DIR = Path(os.environ.get("CCG_SLICE_RUN", CPG / ".ccg-work" / "vertical-slice" / "run1"))
MEASUREMENTS = RUN_DIR / "s13-measurements.json"
HOST, PORT = "127.0.0.1", 6390
PINNED_COMMIT = "defca60ff16534aa12a74e35856de9be98bfc710"

# ---------------------------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------------------------


class Result:
    def __init__(self, created=None):
        self.relationships_created = created


class FakeStore:
    """Records every call in order; ``constraint_script`` is the status list each poll returns."""

    def __init__(self, timeout_max=0, constraint_script=None, short_edges=False, exists=False):
        self.ops: list[tuple] = []
        self._timeout_max = timeout_max
        self.script = list(constraint_script or [])
        self.polls = 0
        self.short_edges = short_edges
        self._exists = exists
        self.n_constraints = 0

    def timeout_max(self):
        return self._timeout_max

    def exists(self, key):
        L.check_key(key)
        return self._exists

    def query(self, key, cypher, params=None, timeout=None):
        L.check_key(key)
        n = len(params["rows"]) if params and "rows" in params else 0
        self.ops.append(("query", key, cypher, n, timeout, params))
        if "CREATE (a)-" in cypher or "]->(b)" in cypher:
            return Result(n - 1 if self.short_edges else n)
        return Result()

    def create_unique_constraint(self, key, label, prop):
        L.check_key(key)
        self.ops.append(("constraint", key, label, prop))
        self.n_constraints += 1

    def constraints(self, key):
        self.ops.append(("poll", key))
        self.polls += 1
        assert self.polls < 500, "runaway constraint polling"
        status = self.script[min(self.polls - 1, len(self.script) - 1)] if self.script else "OPERATIONAL"
        if isinstance(status, str):
            status = [status] * self.n_constraints
        return [{"type": "UNIQUE", "label": f"L{i}", "properties": ["x"], "entity": "NODE", "status": s}
                for i, s in enumerate(status)]

    def rename(self, src, dst):
        L.check_key(src), L.check_key(dst)
        self.ops.append(("rename", src, dst))

    def delete(self, key):
        L.check_key(key)
        self.ops.append(("delete", key))

    def memory(self, key):
        return {"total_graph_sz_mb": 1}

    def used_memory(self):
        return 0

    def kinds(self):
        return [o[0] for o in self.ops]


class MemoryReader:
    """The impact reader over plain tuples: a second implementation of the seven edge rows."""

    def __init__(self, calls=(), owners=(), data=(), to=(), via=(), provides=()):
        self.calls, self.owners, self.data_edges = list(calls), list(owners), list(data)
        self.to, self.via, self.provides = list(to), list(via), list(provides)

    @classmethod
    def from_rows(cls, rows):
        g = rows.edges
        owners = []
        for (src, et, dst), lst in g.items():
            if et in ("HAS_CALL", "HAS_PARAM", "HAS_RETURN", "HAS_COND"):
                owners += [(dst, r["t"], r["s"]) for r in lst]
        link_status = {n["linkId"]: n["status"] for n in rows.nodes["Link"]}
        return cls(
            calls=[(r["t"], r["s"], r.get("p", {}).get("status", "confirmed")) for r in g.get(("CallSite", "CALLS", "Symbol"), [])],
            owners=owners,
            data=[(s, r["s"], d, r["t"], r["p"]["fromPath"], r["p"]["toPath"], r["p"].get("status", "confirmed"))
                  for (s, et, d), lst in g.items() if et == "DATA" for r in lst],
            to=[(r["t"], r["s"], link_status[r["s"]]) for r in g.get(("Link", "TO", "Symbol"), [])],
            via=[(r["t"], r["s"], r["p"]["status"]) for r in g.get(("CallSite", "VIA", "Link"), [])],
            provides=[(r["s"], r["t"]) for r in g.get(("Symbol", "PROVIDES", "ContractPoint"), [])],
        )

    def calls_in(self, ids):
        return sorted((s, c, st) for s, c, st in self.calls if s in ids)

    def owner_of(self, label, ids):
        return sorted((p, s) for lab, p, s in self.owners if lab == label and p in ids)

    def data_in(self, label, ids):
        if label not in I.DEF_LABELS:
            return []
        return sorted((d, ul, u, fp, tp, st) for ul, u, dl, d, fp, tp, st in self.data_edges if dl == label and d in ids)

    def to_in(self, ids):
        return sorted((s, link, st) for link, s, st in self.to if s in ids)

    def via_in(self, ids):
        return sorted((link, c, st) for c, link, st in self.via if link in ids)

    def provides_in(self, ids):
        return sorted((cp, s) for s, cp in self.provides if cp in ids)

    def provides_out(self, ids):
        return sorted((s, cp) for s, cp in self.provides if s in ids)


# ---------------------------------------------------------------------------------------------
# Batching, ids, escapes, keys.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,size,expected", [
    (0, 20000, []), (1, 20000, [1]), (20000, 20000, [20000]), (20001, 20000, [20000, 1]),
    (45000, 20000, [20000, 20000, 5000]), (7, 3, [3, 3, 1]),
])
def test_batches_split_exactly(n, size, expected):
    rows = list(range(n))
    out = list(L.batches(rows, size))
    assert [len(b) for b in out] == expected
    assert [x for b in out for x in b] == rows  # nothing lost, nothing reordered


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        list(L.batches([1], 0))


def test_batch_ceiling_is_d16s_20k():
    assert L.BATCH_MAX == 20_000


def _big_rows(n_params):
    rows = L.GraphRows()
    rows.snapshot = {"snapshotId": "s" * 8}
    rows.nodes["Param"] = [{"pointId": f"{i:032x}", "index": 0, "name": "x", "line": 1} for i in range(n_params)]
    rows.edges[("Symbol", "HAS_PARAM", "Param")] = [{"s": "a" * 32, "t": f"{i:032x}"} for i in range(n_params)]
    return rows


def test_load_writes_in_batches_of_at_most_20k_and_loses_no_row():
    store = FakeStore()
    rep = L.LoadReport(build_key=L.BUILD_PREFIX + "x")
    L.load_build(store, _big_rows(45_000), rep.build_key, rep, poll_interval=0)
    sizes = [o[3] for o in store.ops if o[0] == "query" and o[5] and "rows" in o[5]]
    assert sizes == [20000, 20000, 5000, 20000, 20000, 5000]  # 3 node batches, then 3 edge batches
    assert rep.max_batch == 20000 and rep.batches == 6


def test_explicit_query_timeout_only_when_timeout_max_is_nonzero():
    for tmax, expected in ((0, None), (7000, 7000)):
        store = FakeStore(timeout_max=tmax)
        rep = L.LoadReport()
        L.load_build(store, _big_rows(3), L.BUILD_PREFIX + "x", rep, poll_interval=0)
        writes = [o for o in store.ops if o[0] == "query" and o[5] and "rows" in o[5]]
        assert writes and all(o[4] == expected for o in writes)
        assert rep.query_timeout_used is (expected is not None)


@pytest.mark.parametrize("key,ok", [
    ("ccg_p0slice", True), ("ccg_p0slice__build_abc123", True), ("ccg_p0slice__build_dens_ab", True),
    ("falkor_chat", False), ("cpg_falkorchat", False), ("ccg_p0slice2", False), ("ccg_p0slice__build_", False),
    ("ccg_p0slice__build_a b", False), ("ccg_p0slice\n", False), ("telemetry{ccg_p0slice}", False), ("", False),
    ("ccg_smoke", False), ("xccg_p0slice", False),
])
def test_r_b_key_guard(key, ok):
    if ok:
        assert L.check_key(key) == key
    else:
        with pytest.raises(L.LoadError):
            L.check_key(key)


def test_live_store_refuses_falkordb_dev_port():
    with pytest.raises(L.LoadError):
        L.LiveStore("127.0.0.1", 6379, pw="x")


def test_live_store_uses_one_connection_with_retries_off_and_a_socket_timeout_of_minutes(monkeypatch, tmp_path):
    import falkordb

    seen = {}

    class FakeDB:
        def __init__(self, **kw):
            seen.update(kw)
            self.connection = type("C", (), {"ping": lambda self: True})()

    monkeypatch.setattr(falkordb, "FalkorDB", FakeDB)
    (tmp_path / "falkordb-admin.pass").write_text("s3cret\n")
    monkeypatch.setenv("CCG_CONFIG_DIR", str(tmp_path))
    L.LiveStore("127.0.0.1", 6390)
    assert seen["host"] == "127.0.0.1" and seen["port"] == 6390
    # a retried write could run twice. retry=None is NOT zero retries: redis-py then installs its own
    # default policy, so the loader passes an explicit zero-retry object (S17 note section 4 / 5 Retries)
    retry = seen["retry"]
    assert isinstance(retry, Retry) and retry._retries == 0
    assert isinstance(retry._backoff, NoBackoff)
    assert seen["socket_timeout"] >= 120  # "minutes": a client timeout abandons the answer, not the work
    assert seen["password"] == "s3cret"  # read from the file, stripped
    assert L.LiveStore("127.0.0.1", 6390, pw="given") and seen["password"] == "given"


def test_identifiers_are_validated_before_they_reach_cypher_text():
    for bad in ("Symbol) DETACH DELETE (n", "a-b", "1x", "", "A B"):
        with pytest.raises(ValueError):
            L._ident(bad)
    assert L.node_statement("Symbol") == "UNWIND $rows AS r CREATE (n:Symbol) SET n = r"
    assert L.edge_statement("CallSite", "DATA", "Param", True) == (
        "UNWIND $rows AS r MATCH (a:CallSite {callSiteId: r.s}) MATCH (b:Param {pointId: r.t}) "
        "CREATE (a)-[e:DATA]->(b) SET e = r.p")
    assert L.edge_statement("Unit", "IN_REPO", "Repo", False).endswith("CREATE (a)-[e:IN_REPO]->(b)")
    with pytest.raises(ValueError):
        L.node_statement("Symbol) RETURN 1 //")


NASTY = 'a"b\\c\nd\teé中{}[]`\'$rows -- ; MATCH (n) DETACH DELETE n'


def test_values_travel_as_parameters_never_in_cypher_text():
    rows = L.GraphRows()
    rows.snapshot = {"snapshotId": "q"}
    rows.nodes["Cond"] = [{"pointId": "p" * 32, "line": 1, "code": NASTY}]
    store = FakeStore()
    L.load_build(store, rows, L.BUILD_PREFIX + "q", L.LoadReport(), poll_interval=0)
    writes = [o for o in store.ops if o[0] == "query" and o[5] and "rows" in o[5]]
    assert len(writes) == 1
    assert "DETACH" not in writes[0][2] and NASTY not in writes[0][2]
    assert writes[0][5]["rows"][0]["code"] == NASTY  # verbatim, no escaping applied by us


def test_clean_strips_and_drops_none_and_flatten_regions():
    assert L._clean({"a": "  x\n", "b": None, "c": 3, "d": [1, 2]}) == {"a": "x", "c": 3, "d": [1, 2]}
    assert L._flatten_regions([[1, 2], [5, 9]]) == [1, 2, 5, 9]


# ---------------------------------------------------------------------------------------------
# The load sequence (graph note §3.5), order and gates.
# ---------------------------------------------------------------------------------------------


def _mini_rows():
    rows = L.GraphRows()
    rows.snapshot = {"snapshotId": "abc", "systemId": "p0slice"}
    rows.nodes["Repo"] = [{"repoId": "r"}]
    rows.nodes["Unit"] = [{"unitId": "u"}]
    rows.edges[("Unit", "IN_REPO", "Repo")] = [{"s": "u", "t": "r"}]
    return rows


def _run_load(store, **kw):
    return L.load(store, [], {}, rows=_mini_rows(), **kw)


def _phase(op):
    if op[0] == "query":
        c = op[2]
        if c.startswith("CREATE INDEX"):
            return "index"
        if c.startswith("CREATE (s:Snapshot)"):
            return "snapshot"
        return "edge" if "]->(b)" in c else "node"
    return {"constraint": "constraint", "poll": "poll", "rename": "rename", "delete": "delete"}[op[0]]


def test_sequence_is_index_node_edge_constraint_poll_snapshot_rename():
    store = FakeStore()
    rep = _run_load(store, poll_interval=0)
    phases = [_phase(o) for o in store.ops]
    collapsed = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert collapsed == ["index", "node", "edge", "constraint", "poll", "snapshot", "rename"]
    assert store.ops[-1] == ("rename", L.BUILD_PREFIX + "abc", L.FINAL_KEY)
    assert phases.count("snapshot") == 1 and phases.index("snapshot") == len(phases) - 2
    assert all(o[1] == L.BUILD_PREFIX + "abc" for o in store.ops[:-1])  # everything lands on the build key
    assert rep.build_key == L.BUILD_PREFIX + "abc" and rep.final_key == L.FINAL_KEY


def test_indexes_precede_every_write_and_cover_the_point_labels():
    stmts = L.index_statements()
    for label in ("Param", "Ret", "Cond"):  # S-1: pointId range indexes are the edge-load anchors
        assert f"CREATE INDEX FOR (n:{label}) ON (n.pointId)" in stmts
    assert "CREATE INDEX FOR (n:File) ON (n.path)" in stmts and "CREATE INDEX FOR (n:Symbol) ON (n.qname)" in stmts
    assert {l for l, _ in L.constraint_statements()} == {
        "Symbol", "CallSite", "ContractPoint", "Link", "Name", "Unit", "File", "Param", "Ret", "Cond", "Dependency"}
    for label, prop in L.constraint_statements():  # a constraint needs its index first
        assert f"CREATE INDEX FOR (n:{label}) ON (n.{prop})" in stmts


def test_constraints_are_polled_until_operational():
    store = FakeStore(constraint_script=[["UNDER CONSTRUCTION"] * 10, ["OPERATIONAL"] * 5 + ["UNDER CONSTRUCTION"] * 5, "OPERATIONAL"])
    rep = _run_load(store, poll_interval=0)
    assert store.polls == 3
    assert {c["status"] for c in rep.constraints} == {"OPERATIONAL"}
    assert store.kinds()[-2:] == ["query", "rename"]  # the snapshot is still after the last poll


def test_failed_constraint_fails_the_build_before_snapshot_and_rename():
    store = FakeStore(constraint_script=[["OPERATIONAL"] * 9 + ["FAILED"]])
    ticks = iter(range(0, 10_000, 100))  # a fake clock: a mutant that keeps polling hits the deadline quickly
    with pytest.raises(L.LoadError, match="constraint FAILED"):
        _run_load(store, poll_interval=0, clock=lambda: next(ticks), sleep=lambda s: None)
    assert store.polls == 1  # fails at once, not after polling out the clock
    assert not any(_phase(o) in ("snapshot", "rename") for o in store.ops)


def test_pending_constraints_that_never_finish_time_out_instead_of_proceeding():
    store = FakeStore(constraint_script=[["UNDER CONSTRUCTION"] * 10])
    t = iter(range(0, 10_000, 100))
    with pytest.raises(L.LoadError, match="not OPERATIONAL"):
        L.load_build(store, _mini_rows(), L.BUILD_PREFIX + "abc", L.LoadReport(), poll_interval=0, poll_timeout=250,
                     clock=lambda: next(t), sleep=lambda s: None)
    assert not any(_phase(o) in ("snapshot", "rename") for o in store.ops)


def test_missing_constraints_are_not_mistaken_for_done():
    # the server lists fewer constraints than were created (still being registered): keep polling
    class Slow(FakeStore):
        def constraints(self, key):
            res = super().constraints(key)
            return res[: self.polls] if self.polls < 4 else res

    store = Slow()
    _run_load(store, poll_interval=0)
    assert store.polls >= 4


def test_edge_batch_that_creates_fewer_edges_than_rows_is_an_error():
    with pytest.raises(L.LoadError, match="endpoint did not match"):
        _run_load(FakeStore(short_edges=True), poll_interval=0)


def test_existing_build_key_is_refused_unless_drop_stale():
    with pytest.raises(L.LoadError, match="already exists"):
        _run_load(FakeStore(exists=True), poll_interval=0)
    store = FakeStore(exists=True)
    _run_load(store, drop_stale=True, poll_interval=0)
    assert store.ops[0] == ("delete", L.BUILD_PREFIX + "abc")


def test_load_never_names_a_key_outside_r_b():
    store = FakeStore()
    _run_load(store, poll_interval=0)
    for o in store.ops:
        for part in o[1:3]:
            if isinstance(part, str) and part.startswith("ccg"):
                L.check_key(part)


# ---------------------------------------------------------------------------------------------
# build_rows on a hand-made bundle: every output field against its input.
# ---------------------------------------------------------------------------------------------


def h(n):
    return f"{n:032x}"


def mini_bundle():
    unit, repo = h(1), "repo"
    recs = [
        {"type": "header", "unitId": unit, "unit": {"repoId": repo, "rootPath": "svc", "lang": "python"},
         "builtCommit": "c" * 40, "builtTree": "t" * 40, "builtAt": "2026-01-02T03:04:05Z", "bundleSchemaVersion": "1.0"},
        {"type": "file", "id": h(10), "repoId": repo, "unitId": unit, "path": "svc/a.py", "kind": "code", "role": "primary",
         "lang": "python", "contentHash": "0" * 64, "lineCount": 9},
        {"type": "symbol", "id": h(20), "repoId": repo, "lang": "python", "qname": "svc/a.py::f", "fileId": h(10),
         "path": "svc/a.py", "kind": "function", "name": "f", "nameNorm": "f", "lineStart": 3, "lineEnd": 8,
         "fingerprint": "f" * 64, "complexity": 1, "ownedRegions": [[4, 5], [7, 7]]},
        {"type": "symbol", "id": h(21), "repoId": repo, "lang": "python", "qname": "svc/a.py::g", "fileId": h(10),
         "path": "svc/a.py", "kind": "function", "name": "g", "nameNorm": "g", "lineStart": 1, "lineEnd": 2,
         "fingerprint": "e" * 64, "complexity": 1},
        {"type": "symbol", "id": h(22), "repoId": repo, "lang": "javascript", "qname": "svc/b.ts::g", "fileId": h(10),
         "path": "svc/b.ts", "kind": "function", "name": "g", "nameNorm": "g", "lineStart": 1, "lineEnd": 2,
         "fingerprint": "d" * 64, "complexity": 1},
        {"type": "param", "id": h(30), "symbolId": h(20), "index": 0, "name": "x", "line": 3},
        {"type": "ret", "id": h(31), "symbolId": h(20), "ordinal": 0, "line": 8},
        {"type": "cond", "id": h(32), "symbolId": h(20), "ordinal": 0, "line": 5, "code": "x > 1"},
        {"type": "callsite", "id": h(40), "symbolId": h(20), "calleeName": "g", "ordinal": 0, "callKind": "call", "line": 6,
         "resolution": "resolved", "targets": [{"symbolId": h(21), "qname": "svc/a.py::g"}, {"symbolId": h(99), "qname": "x::gone"}],
         "status": "confirmed", "reason": "resolved-call"},
        {"type": "callsite", "id": h(41), "symbolId": h(20), "calleeName": "<dynamic>", "ordinal": 0, "callKind": "call",
         "line": 7, "resolution": "unresolved", "targets": [], "status": "possible", "reason": "name-match"},
        {"type": "data", "id": h(50), "use": h(40), "def": h(30), "toPath": "a0", "fromPath": "p0.k", "status": "confirmed", "reason": "intra-flow"},
        {"type": "data", "id": h(51), "use": h(31), "def": h(40), "toPath": "ret", "fromPath": "r", "status": "confirmed", "reason": "intra-flow"},
        {"type": "data", "id": h(52), "use": h(40), "def": h(40), "toPath": "recv", "fromPath": "r", "status": "confirmed", "reason": "intra-flow"},
        {"type": "data", "id": h(53), "use": h(40), "def": h(77), "toPath": "a1", "fromPath": "r", "status": "confirmed", "reason": "intra-flow"},
        {"type": "controlledBy", "id": h(60), "point": h(40), "cond": h(32)},
        {"type": "controlledBy", "id": h(61), "point": h(32), "cond": h(32)},
        {"type": "controlledBy", "id": h(62), "point": h(40), "cond": h(32)},
    ]
    return recs


def mini_facts():
    return {"providers": [], "consumers": [], "links": []}


def test_build_rows_maps_every_record_field_by_field():
    rows = L.build_rows([mini_bundle()], mini_facts())
    n = {k: v for k, v in rows.nodes.items()}
    assert n["Repo"] == [{"repoId": "repo", "headCommit": "c" * 40}]
    assert n["Unit"] == [{"unitId": h(1), "repoId": "repo", "rootPath": "svc", "language": "python", "builtCommit": "c" * 40,
                          "builtTree": "t" * 40, "builtAt": "2026-01-02T03:04:05Z", "buildStatus": "fresh"}]
    assert n["File"] == [{"fileId": h(10), "repoId": "repo", "unitId": h(1), "path": "svc/a.py", "kind": "code", "role": "primary",
                          "lang": "python", "contentHash": "0" * 64}]
    f, g, g_js = n["Symbol"]
    assert g_js["symbolId"] == h(22)
    assert f == {"symbolId": h(20), "repoId": "repo", "unitId": h(1), "kind": "function", "name": "f", "nameNorm": "f",
                 "qname": "svc/a.py::f", "path": "svc/a.py", "lineStart": 3, "lineEnd": 8, "fingerprint": "f" * 64,
                 "ownedRegions": [4, 5, 7, 7], "lang": "python", "complexity": 1, "depth": 0}
    assert "ownedRegions" not in g and "lastChangedAt" not in g  # absent stays absent, never null
    assert n["Param"] == [{"pointId": h(30), "index": 0, "name": "x", "line": 3}]
    assert n["Ret"] == [{"pointId": h(31), "line": 8}]
    assert n["Cond"] == [{"pointId": h(32), "line": 5, "code": "x > 1"}]
    assert {c["callSiteId"] for c in n["CallSite"]} == {h(40), h(41)}
    cs = next(c for c in n["CallSite"] if c["callSiteId"] == h(40))
    assert cs == {"callSiteId": h(40), "repoId": "repo", "unitId": h(1), "path": "svc/a.py", "line": 6, "calleeName": "g"}
    assert [x["name"] for x in n["Name"]] == ["g"]  # no Name for <dynamic>
    assert n["Name"][0]["lang"] == "python" and len(n["Name"][0]["nameKey"]) == 32


def test_build_rows_edges_field_by_field_and_loader_side_drops():
    rows = L.build_rows([mini_bundle()], mini_facts())
    e = rows.edges
    assert e[("Unit", "IN_REPO", "Repo")] == [{"s": h(1), "t": "repo"}]
    assert e[("File", "DEFINES", "Symbol")] == [{"s": h(10), "t": h(t)} for t in (20, 21, 22)]
    assert e[("Symbol", "HAS_PARAM", "Param")] == [{"s": h(20), "t": h(30)}]
    assert e[("Symbol", "HAS_CALL", "CallSite")] == [{"s": h(20), "t": h(40)}, {"s": h(20), "t": h(41)}]
    assert e[("CallSite", "CALLS", "Symbol")] == [{"s": h(40), "t": h(21)}]  # D8: the default status/reason is not stored
    assert rows.dropped["calls-target-not-in-snapshot"] == 1
    nk = rows.nodes["Name"][0]["nameKey"]
    assert e[("CallSite", "CALLS_NAME", "Name")] == [{"s": h(40), "t": nk}]
    assert nk == name_key("python", "g")
    assert e[("Name", "RESOLVES_TO", "Symbol")] == [{"s": nk, "t": h(21)}]  # the python `g`, not the javascript one
    data = {k: v for k, v in e.items() if k[1] == "DATA"}
    assert data[("CallSite", "DATA", "Param")] == [
        {"s": h(40), "t": h(30), "p": {"fromPath": "p0.k", "toPath": "a0"}}]  # D8
    assert data[("Ret", "DATA", "CallSite")][0]["p"]["toPath"] == "ret"
    # a call site that feeds itself with different paths is a real fact (loop-carried), kept
    assert data[("CallSite", "DATA", "CallSite")] == [
        {"s": h(40), "t": h(40), "p": {"fromPath": "r", "toPath": "recv"}}]
    assert rows.dropped["data-endpoint-missing"] == 1
    # G-4: the cond->cond self-edge is dropped and the repeated (point, cond) pair written once
    assert e[("CallSite", "CONTROLLED_BY", "Cond")] == [{"s": h(40), "t": h(32)}]
    assert ("Cond", "CONTROLLED_BY", "Cond") not in e
    assert rows.dropped["controlled-by-self-edge"] == 1 and rows.dropped["duplicate-CONTROLLED_BY"] == 1
    assert rows.edge_count() == sum(len(v) for v in e.values())


def test_snapshot_is_deterministic_and_keys_the_build():
    a = L.build_rows([mini_bundle()], mini_facts()).snapshot
    b = L.build_rows([mini_bundle()], mini_facts()).snapshot
    assert a == b and set(a) == {"snapshotId", "systemId", "builtAt", "vocabVersion", "bundleSchemaVersion", "manifestHash"}
    other = mini_bundle()
    other[0]["builtTree"] = "u" * 40
    assert L.build_rows([other], mini_facts()).snapshot["snapshotId"] != a["snapshotId"]


def test_link_facts_become_nodes_and_edges_with_lifted_chain_props():
    cp_key = "GET /x"
    facts = {
        "providers": [{"handlerId": h(21), "key": cp_key, "status": "possible", "reason": "partial-key"},
                      {"handlerId": h(77), "key": cp_key, "status": "confirmed", "reason": "key-match"}],
        "consumers": [{"siteId": h(40), "keys": [cp_key, "GET /y"], "partialAlternatives": True}],
        "links": [{"id": h(70), "kind": "http", "keyNorm": cp_key, "status": "confirmed", "reason": "key-match",
                   "matcher": "exact", "provider": {"anchorId": h(21)},
                   "sites": [{"siteId": h(40), "consumerKey": cp_key, "status": "confirmed", "reason": "key-match",
                              "liftedVia": [{"ord": 0, "kind": "callsite", "id": h(40), "line": 6, "urlPath": "a0"},
                                            {"ord": 1, "kind": "symbol", "id": h(20), "line": 3, "urlPath": "p0.path",
                                             "verbPath": "p0.method", "viaSiteId": h(40), "viaLine": 6}]},
                             {"siteId": h(40), "consumerKey": "GET /y", "status": "possible", "reason": "partial-key",
                              "liftedVia": [{"ord": 0, "kind": "callsite", "id": h(40), "line": 6, "urlPath": "a0"}]}]}],
    }
    rows = L.build_rows([mini_bundle()], facts)
    assert rows.dropped["provider-handler-missing"] == 1  # h(77) has no Symbol node
    cps = {c["keyNorm"]: c for c in rows.nodes["ContractPoint"]}
    assert set(cps) == {cp_key, "GET /y"} and cps[cp_key]["kind"] == "http" and cps[cp_key]["key"] == cp_key
    assert cps[cp_key]["cpId"] == contract_point_id("http", cp_key)
    assert rows.nodes["Link"] == [{"linkId": h(70), "kind": "http", "status": "confirmed", "reason": "key-match",
                                   "matcher": "exact"}]  # D6: no candidateCount on Link
    e = rows.edges
    assert e[("Symbol", "PROVIDES", "ContractPoint")] == [
        {"s": h(21), "t": cps[cp_key]["cpId"], "p": {"status": "possible", "reason": "partial-key"}}]
    assert e[("Link", "TO", "Symbol")] == [{"s": h(70), "t": h(21)}]
    assert e[("Link", "ON", "ContractPoint")] == [{"s": h(70), "t": cps[cp_key]["cpId"]}]
    assert [r["p"]["consumerKey"] for r in e[("CallSite", "VIA", "Link")]] == [cp_key, "GET /y"]  # two keys, two edges
    assert e[("CallSite", "CONSUMES", "ContractPoint")][0]["p"] == {"status": "possible", "reason": "lifted-role"}
    assert e[("Link", "LIFTED_VIA", "CallSite")] == [
        {"s": h(70), "t": h(40), "p": {"ord": 0, "siteId": h(40), "line": 6, "urlPath": "a0"}}]  # repeated hop written once
    assert e[("Link", "LIFTED_VIA", "Symbol")] == [
        {"s": h(70), "t": h(20), "p": {"ord": 1, "siteId": h(40), "line": 3, "urlPath": "p0.path", "verbPath": "p0.method",
                                       "viaSiteId": h(40), "viaLine": 6}}]
    assert set(rows.edges) <= set(L.EDGE_ORDER)


# ---------------------------------------------------------------------------------------------
# The BFS on a tiny hand-built graph.
# ---------------------------------------------------------------------------------------------

S = lambda n: ("Symbol", f"S{n}")  # noqa: E731


def chain_graph():
    """A <- cB (in B) <- cC (in C) <- cA (in A): a call cycle A -> C -> B -> A (G-3)."""
    return MemoryReader(
        calls=[("A", "cB", "confirmed"), ("B", "cC", "confirmed"), ("C", "cA", "confirmed")],
        owners=[("CallSite", "cB", "B"), ("CallSite", "cC", "C"), ("CallSite", "cA", "A")],
    )


def _sym(ids):
    return {i.symbol_id: i for i in ids}


def test_call_chain_distances_hops_and_chain_nodes_are_exact():
    res = I.walk(chain_graph(), ("Symbol", "A"))
    items = _sym(res.items())
    assert set(items) == {"B", "C"}  # the anchor is not its own item
    assert (items["B"].distance, items["B"].hops, items["B"].status, items["B"].length) == (1, 2, "confirmed", 2)
    assert (items["C"].distance, items["C"].hops, items["C"].length) == (2, 4, 4)
    assert [s["node"] for s in items["C"].chain] == [("Symbol", "A"), ("CallSite", "cB"), ("Symbol", "B"),
                                                      ("CallSite", "cC"), ("Symbol", "C")]
    assert [s["edge"]["type"] for s in items["C"].chain[1:]] == ["CALLS", "HAS_CALL", "CALLS", "HAS_CALL"]
    assert all(s["edge"]["dir"] == "<-" for s in items["C"].chain[1:])
    assert items["C"].chain[0]["edge"] is None


def test_the_visited_set_stops_a_cycle_and_counts_states():
    res = I.walk(chain_graph(), ("Symbol", "A"))
    # the anchor re-entered by the cycle is a new (function-level) state, whose own call site is no improvement
    assert res.levels == 7
    assert res.frontier_sizes == [1, 1, 1, 1, 1, 1, 1]
    assert res.peak_frontier == 1
    assert res.state_count == 7 and res.states_by_kind() == {"function": 3, "path": 3, "value": 1}
    assert res.field_path_states == 0


def test_max_levels_guards_a_runaway_walk():
    with pytest.raises(I.ImpactError, match="exceeded 3 levels"):
        I.walk(chain_graph(), ("Symbol", "A"), max_levels=3)


def test_edge_direction_is_against_the_stored_edge():
    # B calls A, so A changing impacts B; B changing does not impact A
    g = MemoryReader(calls=[("A", "cB", "confirmed")], owners=[("CallSite", "cB", "B")])
    assert [i.symbol_id for i in I.walk(g, ("Symbol", "A")).items()] == ["B"]
    assert I.walk(g, ("Symbol", "B")).items() == []


def dg():
    """In B: cB calls A (the anchor). cB's result r feeds g (a0) via r.x and ret via r.y; p0.a feeds cB's a0."""
    return MemoryReader(
        calls=[("A", "cB", "confirmed")],
        owners=[("CallSite", "cB", "B"), ("CallSite", "g", "B"), ("Ret", "rB", "B"), ("Param", "p0", "B"), ("Cond", "k", "B")],
        data=[("CallSite", "g", "CallSite", "cB", "r.x", "a0", "confirmed"),
              ("Ret", "rB", "CallSite", "cB", "r.y", "ret", "confirmed"),
              ("Cond", "k", "CallSite", "cB", "r", "", "confirmed"),
              ("CallSite", "cB", "Param", "p0", "p0.a", "a0", "confirmed"),
              ("CallSite", "cB", "Param", "p0", "p0.ab", "a1", "confirmed"),
              ("CallSite", "cB", "Param", "p0", "p0", "a2", "confirmed")],
    )


def _data_reached(res):
    return {k[1]: (k[2], k[3]) for k in res.visited if res.best[k].edge and res.best[k].edge.type == "DATA"}


@pytest.mark.parametrize("path,expected", [
    ("r", {"g", "rB", "k"}),            # the whole result changed: every use
    ("r.x", {"g", "k"}),                # a field changed: its uses, plus the whole-value use `r`
    ("r.y", {"rB", "k"}),
    ("r.xy", {"k"}),                    # `r.x` is not a prefix of `r.xy` at a segment boundary
    ("r.x.deep", {"g", "k"}),           # a deeper field is covered by the use of `r.x`
    ("q", set()),
])
def test_data_edges_are_path_matched(path, expected):
    res = I.walk(dg(), ("CallSite", "cB"), ("path", path))
    assert set(_data_reached(res)) == expected
    assert _data_reached(res).get("g", ("path", "a0")) == ("path", "a0")
    # a use the path does not touch is absent from the whole walk, enclosing function aside
    for absent in {"g", "rB", "k"} - expected:
        assert not any(k[1] == absent for k in res.visited)


def test_whole_value_state_matches_every_data_edge_and_params_filter_by_path():
    res = I.walk(dg(), ("CallSite", "cB"), I.VALUE)
    assert set(_data_reached(res)) == {"g", "rB", "k"}
    for path, expected in (
        ("p0.a", {("cB", "a0"), ("cB", "a2")}),    # the use of `p0.a` and the use of the whole `p0`
        ("p0.ab", {("cB", "a1"), ("cB", "a2")}),   # `p0.a` is not a prefix of `p0.ab` at a segment boundary
        ("p0", {("cB", "a0"), ("cB", "a1"), ("cB", "a2")}),
        ("p0.zz", {("cB", "a2")}),                 # only the whole-`p0` use covers an unrelated field
        ("p1", set()),
    ):
        r = I.walk(dg(), ("Param", "p0"), ("path", path))
        assert {(k[1], k[3]) for k in r.visited if r.best[k].edge and r.best[k].edge.type == "DATA"} == expected, path
    assert I.path_overlaps("p0.a", "p0.ab") is False and I.path_overlaps("p0", "p0.ab") is True
    assert I.path_overlaps("p1", "p10") is False and I.path_overlaps("a[0]", "a") is True


def test_path_state_after_a_data_hop_is_the_uses_to_path():
    res = I.walk(dg(), ("CallSite", "cB"), ("path", "r.x"))
    key = next(k for k in res.visited if k[1] == "g")
    assert key == ("CallSite", "g", "path", "a0")
    assert res.best[key].cost == (0, 0, 1) and res.best[key].edge.as_dict() == {"type": "DATA", "dir": "<-", "status": "confirmed"}
    assert res.field_path_states == 1 + 0  # only `r.x` itself has a field segment; `a0` has none


def test_possible_hops_rank_before_distance_and_a_better_path_wins():
    # D is reached by a possible direct call (poss 1, dist 1) and by a confirmed two-call route (poss 0, dist 2):
    # B's `Ret` carries the call's result, so the continuation out of B is not function-level
    g = MemoryReader(
        calls=[("A", "cD", "possible"), ("A", "cB", "confirmed"), ("B", "cD2", "confirmed")],
        owners=[("CallSite", "cD", "D"), ("CallSite", "cB", "B"), ("CallSite", "cD2", "D"), ("Ret", "rB", "B")],
        data=[("Ret", "rB", "CallSite", "cB", "r", "ret", "confirmed")],
    )
    d = _sym(I.walk(g, ("Symbol", "A")).items())["D"]
    assert (d.status, d.possible_hops, d.distance) == ("confirmed", 0, 2)
    only_possible = MemoryReader(calls=[("A", "cD", "possible")], owners=[("CallSite", "cD", "D")])
    d = _sym(I.walk(only_possible, ("Symbol", "A")).items())["D"]
    assert (d.status, d.possible_hops, d.distance, d.hops) == ("possible", 1, 1, 2)


def test_ties_break_on_the_predecessors_stable_key():
    g = MemoryReader(calls=[("A", "cz", "confirmed"), ("A", "ca", "confirmed")],
                     owners=[("CallSite", "cz", "D"), ("CallSite", "ca", "D")])
    d = _sym(I.walk(g, ("Symbol", "A")).items())["D"]
    assert d.chain[1]["node"] == ("CallSite", "ca")


def link_graph():
    return MemoryReader(
        calls=[("F", "cG", "confirmed")],
        owners=[("CallSite", "cons", "F"), ("CallSite", "cG", "G")],
        to=[("L", "P", "confirmed")], via=[("cons", "L", "confirmed")], provides=[("P", "CP")],
    )


def test_contract_mode_from_a_contract_point_lists_consumers_at_distance_one():
    res = I.walk(link_graph(), ("ContractPoint", "CP"), I.CONTRACT)
    items = _sym(res.items())
    assert set(items) == {"F"}  # the provider P is the contract's own side, not an impacted item; no walk past F
    f = items["F"]
    assert (f.distance, f.hops, f.status) == (1, 4, "confirmed")
    assert [(s["node"], s["edge"] and (s["edge"]["type"], s["edge"]["dir"])) for s in f.chain] == [
        (("ContractPoint", "CP"), None), (("Symbol", "P"), ("PROVIDES", "<-")), (("Link", "L"), ("TO", "<-")),
        (("CallSite", "cons"), ("VIA", "<-")), (("Symbol", "F"), ("HAS_CALL", "<-"))]
    assert res.states_by_kind() == {"contract": 4, "end": 1}


def test_a_provider_whose_key_changed_switches_to_contract_mode_through_provides():
    res = I.walk(link_graph(), ("Symbol", "P"), I.KEY)
    assert [i.symbol_id for i in res.items()] == ["F"]
    assert [s["edge"] and s["edge"]["type"] for s in res.items()[0].chain] == [None, "PROVIDES", "PROVIDES", "TO", "VIA", "HAS_CALL"]


def test_a_changed_provider_body_reaches_consumers_in_value_mode_and_their_callers():
    res = I.walk(link_graph(), ("Symbol", "P"), I.VALUE)
    items = _sym(res.items())
    assert set(items) == {"F", "G"} and (items["F"].distance, items["G"].distance) == (1, 2)
    assert {(k[0], k[2], k[3]) for k in res.visited if k[1] == "cons"} == {("CallSite", "path", "r")}


def test_link_status_possible_marks_the_item_possible():
    g = link_graph()
    g.via = [("cons", "L", "possible")]
    f = _sym(I.walk(g, ("ContractPoint", "CP"), I.CONTRACT).items())["F"]
    assert (f.status, f.possible_hops, f.distance) == ("possible", 1, 1)


def test_cypher_reader_never_returns_a_possibly_truncated_reply():
    reader = I.CypherReader(lambda q, p: [["x"]] * I.RESULT_SET_LIMIT)
    with pytest.raises(I.ImpactError, match="RESULTSET_SIZE"):
        reader.calls_in(["a"])
    seen = []
    reader = I.CypherReader(lambda q, p: seen.append(len(p["ids"])) or [])
    reader.calls_in([str(i) for i in range(2500)])
    assert seen == [1000, 1000, 500]  # chunked, one query each


def test_cypher_reader_queries_are_parameterized_and_read_only():
    seen = []
    reader = I.CypherReader(lambda q, p: seen.append((q, p)) or [])
    reader.calls_in(["a"]), reader.owner_of("Param", ["a"]), reader.data_in("CallSite", ["a"]), reader.to_in(["a"])
    reader.via_in(["a"]), reader.provides_in(["a"]), reader.provides_out(["a"])
    assert reader.data_in("Ret", ["a"]) == []  # a Ret is never a DATA def
    assert len(seen) == 7
    for q, p in seen:
        assert p == {"ids": ["a"]} and "$ids" in q and not any(w in q for w in ("CREATE", "SET ", "DELETE", "MERGE"))
    assert "(c:CallSite)-[e:CALLS]->(s:Symbol" in seen[0][0] and "-[:HAS_PARAM]->(p:Param {pointId: i})" in seen[1][0]
    assert "-[d:DATA]->(p:CallSite {callSiteId: i})" in seen[2][0]
    assert "(l:Link)-[:TO]->(s:Symbol" in seen[3][0] and "(c:CallSite)-[v:VIA]->(l:Link" in seen[4][0]
    assert "(s:Symbol)-[:PROVIDES]->(cp:ContractPoint {cpId: i})" in seen[5][0] and "(s:Symbol {symbolId: i})-[:PROVIDES]->" in seen[6][0]


# ---------------------------------------------------------------------------------------------
# Real bundles: rows against records (no database).
# ---------------------------------------------------------------------------------------------


def _bundle_paths():
    out = {}
    for p in sorted(RUN_DIR.glob("*.bundle.jsonl")):
        with open(p) as f:
            hdr = json.loads(f.readline())
        out[hdr["unit"]["rootPath"]] = p
    return out


@pytest.fixture(scope="module")
def real():
    paths = _bundle_paths()
    if set(paths) != {"falkor-chat/server", "salesperson"}:
        pytest.skip(f"S11 bundles not found under {RUN_DIR} (gitignored; run spikes/vertical-slice/run_extract.sh)")
    bundles = {root: read_bundle(p) for root, p in paths.items()}
    ordered = [bundles["falkor-chat/server"], bundles["salesperson"]]
    facts = assemble(ordered)
    return {"bundles": bundles, "ordered": ordered, "facts": facts, "rows": L.build_rows(ordered, facts)}


def _count(records, t):
    return sum(1 for r in records if r["type"] == t)


def test_real_rows_counts_match_records_exactly(real):
    recs = [r for b in real["ordered"] for r in b]
    rows, facts = real["rows"], real["facts"]
    n = {k: len(v) for k, v in rows.nodes.items()}
    assert n["File"] == _count(recs, "file") and n["Symbol"] == _count(recs, "symbol")
    assert n["Param"] == _count(recs, "param") and n["Ret"] == _count(recs, "ret") and n["Cond"] == _count(recs, "cond")
    assert n["CallSite"] == _count(recs, "callsite") and n["Unit"] == 2 and n["Repo"] == 1
    assert n["Link"] == len(facts["links"])
    assert n["ContractPoint"] == len({p["key"] for p in facts["providers"]} | {k for c in facts["consumers"] for k in c["keys"]})
    e = Counter()
    for (s, t, d), lst in rows.edges.items():
        e[t] += len(lst)
    assert e["HAS_CALL"] == n["CallSite"] and e["DEFINES"] == n["Symbol"] and e["HAS_FILE"] == n["File"]
    assert e["HAS_PARAM"] == n["Param"] and e["HAS_RETURN"] == n["Ret"] and e["HAS_COND"] == n["Cond"]
    assert e["CALLS"] == sum(len(r["targets"]) for r in recs if r["type"] == "callsite")  # none dropped: every target is in a bundle
    assert e["CALLS_NAME"] == n["CallSite"] - sum(1 for r in recs if r["type"] == "callsite" and r["calleeName"] == "<dynamic>")
    assert e["DATA"] == _count(recs, "data") and e["CONTROLLED_BY"] == _count(recs, "controlledBy")  # bundles are already G-4-clean
    assert e["PROVIDES"] == len(facts["providers"]) and e["TO"] == e["ON"] == len(facts["links"])
    assert e["VIA"] == sum(len(l["sites"]) for l in facts["links"])
    assert e["LIFTED_VIA"] == len({(l["id"], s["siteId"], h["ord"]) for l in facts["links"] for s in l["sites"] for h in s["liftedVia"]})
    assert rows.dropped == {}, rows.dropped


def test_real_name_nodes_and_resolves_to_match_an_independent_oracle(real):
    recs = [r for b in real["ordered"] for r in b]
    lang = {r["id"]: r["lang"] for r in recs if r["type"] == "symbol"}
    symbols = defaultdict(set)
    for r in recs:
        if r["type"] == "symbol":
            symbols[(r["lang"], r["name"])].add(r["id"])
    names = {(lang[r["symbolId"]], r["calleeName"]) for r in recs if r["type"] == "callsite" and r["calleeName"] != "<dynamic>"}
    rows = real["rows"]
    assert {n["nameKey"]: (n["lang"], n["name"]) for n in rows.nodes["Name"]} == {name_key(l, n): (l, n) for l, n in names}
    want = {(name_key(l, n), s) for l, n in names for s in symbols[(l, n)]}
    got = {(r["s"], r["t"]) for r in rows.edges[("Name", "RESOLVES_TO", "Symbol")]}
    assert got == want and len(rows.edges[("Name", "RESOLVES_TO", "Symbol")]) == len(want)
    assert {c["cpId"]: c["keyNorm"] for c in rows.nodes["ContractPoint"]} == {
        contract_point_id("http", k): k for k in {p["key"] for p in real["facts"]["providers"]} | {k for c in real["facts"]["consumers"] for k in c["keys"]}}
    assert {l["linkId"] for l in rows.nodes["Link"]} == {l["id"] for l in real["facts"]["links"]}
    assert all(l["id"] == link_id("http", l["keyNorm"], l["provider"]["anchorId"], l["consumer"]["anchorId"]) for l in real["facts"]["links"])


def test_real_snapshot_fields_come_from_the_headers(real):
    heads = [next(r for r in b if r["type"] == "header") for b in real["ordered"]]
    snap = real["rows"].snapshot
    assert snap["builtAt"] == max(h["builtAt"] for h in heads) and len({h["builtAt"] for h in heads}) == 2
    assert snap["bundleSchemaVersion"] == heads[0]["bundleSchemaVersion"] and snap["systemId"] == "p0slice"
    manifest = sorted([h["unitId"], h["builtCommit"], h["builtTree"]] for h in heads)
    import hashlib
    assert snap["manifestHash"] == hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    from ccg.ids import tuple_id
    assert snap["snapshotId"] == tuple_id(["snapshot", "p0slice", snap["manifestHash"]])


def _find_symbol(real, qname):
    for r in (x for b in real["ordered"] for x in b):
        if r["type"] == "symbol" and r["qname"] == qname:
            return r
    raise KeyError(qname)


def test_real_rows_contain_the_ac1_data_rows_of_spec_5_10(real):
    recs = {r["id"]: r for b in real["ordered"] for r in b if "id" in r}
    api = _find_symbol(real, "salesperson/src/api/client.ts::apiFetch")["id"]
    bu = _find_symbol(real, "salesperson/src/api/client.ts::buildUrl")["id"]

    def cs_of(symbol, callee):
        hits = [r for r in recs.values() if r["type"] == "callsite" and r["symbolId"] == symbol and r["calleeName"] == callee]
        assert len(hits) == 1
        return hits[0]["id"]

    def param0(symbol):
        return next(r["id"] for r in recs.values() if r["type"] == "param" and r["symbolId"] == symbol and r["index"] == 0)

    ret_bu = [r["id"] for r in recs.values() if r["type"] == "ret" and r["symbolId"] == bu]
    want = {
        ("CallSite", cs_of(api, "fetch"), "CallSite", cs_of(api, "buildUrl"), "r", "a0"),
        ("CallSite", cs_of(api, "buildUrl"), "Param", param0(api), "p0.path", "a0"),
        ("CallSite", cs_of(api, "fetch"), "Param", param0(api), "p0.method", "a1.method"),
    } | {("Ret", r, "Param", param0(bu), "p0", "ret") for r in ret_bu}
    have = {(s, r["s"], d, r["t"], r["p"]["fromPath"], r["p"]["toPath"])
            for (s, t, d), lst in real["rows"].edges.items() if t == "DATA" for r in lst}
    assert want <= have


# ---------------------------------------------------------------------------------------------
# Live: ccg-falkordb.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store():
    try:
        s = L.LiveStore(HOST, PORT)
    except FileNotFoundError as exc:
        pytest.skip(f"ccg-falkordb admin password file not readable ({exc.filename}); see cpg/infra/README.md")
    except Exception as exc:  # noqa: BLE001 -- any connection failure is "unreachable"
        pytest.skip(f"ccg-falkordb unreachable at {HOST}:{PORT} ({type(exc).__name__}); start it with cpg/infra/ccg-falkordb.sh up")
    return s


def run_on(store, key=L.FINAL_KEY):
    return lambda q, p=None: store.query(key, q, p).result_set


MEAS: dict = {}


def facts_snapshot(store, real):
    """Everything SL-6 compares, from one loaded graph: link set, link edges, BFS answers."""
    run = run_on(store)
    snap = {
        "links": run("MATCH (l:Link) RETURN l.linkId, l.kind, l.status, l.reason, l.matcher ORDER BY l.linkId"),
        "via": run("MATCH (c:CallSite)-[v:VIA]->(l:Link) RETURN l.linkId, c.callSiteId, v.status, v.reason, v.consumerKey ORDER BY l.linkId, c.callSiteId, v.consumerKey"),
        "to": run("MATCH (l:Link)-[:TO]->(s:Symbol) RETURN l.linkId, s.symbolId ORDER BY l.linkId"),
        "lifted": run("MATCH (l:Link)-[r:LIFTED_VIA]->(t) RETURN l.linkId, r.siteId, r.ord, labels(t)[0], coalesce(t.callSiteId, t.symbolId), r.line, r.urlPath, r.verbPath ORDER BY l.linkId, r.siteId, r.ord"),
    }
    reader = I.CypherReader(run)
    answers = {}
    for key in sorted({l["keyNorm"] for l in real["facts"]["links"]}):
        cp = run("MATCH (cp:ContractPoint {keyNorm: $k}) RETURN cp.cpId", {"k": key})[0][0]
        answers["contract " + key] = _answer(I.walk(reader, ("ContractPoint", cp), I.CONTRACT))
    for qn in ("buildUrl", "apiFetch"):
        sid = _find_symbol(real, f"salesperson/src/api/client.ts::{qn}")["id"]
        answers["value " + qn] = _answer(I.walk(reader, ("Symbol", sid)))
    return {"graph": snap, "answers": answers}


def _answer(res):
    return {
        "items": [[it.symbol_id, it.distance, it.possible_hops, it.status, it.length, it.hops,
                   [[s["node"][0], s["node"][1], s["state"][0], s["state"][1], s["edge"]] for s in it.chain]]
                  for it in res.items()],
        **I.state_summary(res),
    }


@pytest.fixture(scope="module")
def loaded(store, real):
    """Load twice from the same bundles (SL-6); the first load's facts are read before the second replaces it."""
    rep1 = L.load(store, real["ordered"], real["facts"], drop_stale=True)
    assert store.exists(L.FINAL_KEY) and not store.exists(rep1.build_key)
    snap1 = facts_snapshot(store, real)
    rep2 = L.load(store, real["ordered"], real["facts"], drop_stale=True)
    snap2 = facts_snapshot(store, real)
    MEAS["loads"] = [_report_json(rep1), _report_json(rep2)]
    return {"rep1": rep1, "rep2": rep2, "snap1": snap1, "snap2": snap2, "run": run_on(store),
            "reader": I.CypherReader(run_on(store))}


def _report_json(rep):
    d = {k: v for k, v in rep.__dict__.items() if k != "constraints"}
    d["constraints"] = [[c["label"], c["properties"], c["status"]] for c in rep.constraints]
    return d


@pytest.mark.live
def test_sl3_load_integrity(store, real, loaded):
    run = loaded["run"]
    cons = store.constraints(L.FINAL_KEY)
    assert len(cons) == len(L.constraint_statements()) == 11
    assert {(c["label"], c["properties"][0]) for c in cons} == set(L.constraint_statements())
    assert all(c["type"] == "UNIQUE" and c["entity"] == "NODE" for c in cons)
    assert [c["status"] for c in cons] == ["OPERATIONAL"] * 11  # none FAILED, none pending
    snaps = run("MATCH (s:Snapshot) RETURN s")
    assert len(snaps) == 1
    props = snaps[0][0].properties
    assert props == real["rows"].snapshot  # field by field
    # RENAME done: the final key exists, the build key is gone, nothing else of ours is left
    keys = sorted(k for k in store.conn.keys("*") if k.startswith("ccg_p0slice"))
    assert keys == [L.FINAL_KEY] or keys == [L.FINAL_KEY, "telemetry{ccg_p0slice}"] or set(keys) <= {L.FINAL_KEY}
    assert not [k for k in store.conn.keys("*") if "__build_" in k]
    # and the stored graph is exactly the rows: every label and every edge type, count for count
    rows = real["rows"]
    got_nodes = dict(run("MATCH (n) RETURN labels(n)[0], count(n)"))
    want_nodes = {k: len(v) for k, v in rows.nodes.items()}
    want_nodes["Snapshot"] = 1
    assert got_nodes == want_nodes
    got_edges = dict(run("MATCH ()-[e]->() RETURN type(e), count(e)"))
    want_edges = Counter()
    for (s, t, d), lst in rows.edges.items():
        want_edges[t] += len(lst)
    assert got_edges == dict(want_edges)
    rep = loaded["rep2"]
    assert rep.node_total == sum(got_nodes.values()) and rep.edge_total == sum(got_edges.values())
    assert rep.max_batch <= L.BATCH_MAX and rep.timeout_max == 0 and rep.query_timeout_used is False
    assert rep.dropped == {}
    MEAS["sl3"] = {"constraints": [[c["label"], c["status"]] for c in cons], "snapshot": dict(props),
                   "keys": keys, "nodes": got_nodes, "edges": got_edges}


@pytest.mark.live
def test_values_with_quotes_newlines_and_unicode_round_trip_through_the_driver(store):
    key = L.BUILD_PREFIX + "escape_probe"
    if store.exists(key):
        store.delete(key)
    try:
        store.query(key, "UNWIND $rows AS r CREATE (n:Cond) SET n = r", {"rows": [{"pointId": "e1", "line": 3, "code": NASTY}]})
        got = store.query(key, "MATCH (n:Cond {pointId: 'e1'}) RETURN n.code, n.line").result_set
        assert got == [[NASTY, 3]]
        injected = store.query(key, "MATCH (n) RETURN count(n)").result_set
        assert injected == [[1]]  # the `DETACH DELETE` inside the value stayed data
    finally:
        store.delete(key)


def _qn_id(real, qname):
    return _find_symbol(real, qname)["id"]


def _site_symbol(real):
    out = {}
    for b in real["ordered"]:
        for r in b:
            if r["type"] == "callsite":
                out[r["id"]] = r["symbolId"]
    return out


@pytest.mark.live
def test_sl4_contract_mode_lists_the_endpoints_caller_confirmed_at_distance_one(store, real, loaded):
    run, reader = loaded["run"], loaded["reader"]
    key = "GET /shop/api/health"
    cp = run("MATCH (cp:ContractPoint {keyNorm: $k}) RETURN cp.cpId", {"k": key})
    assert len(cp) == 1
    res = I.walk(reader, ("ContractPoint", cp[0][0]), I.CONTRACT)
    items = res.items()
    get_health = _qn_id(real, "salesperson/src/api/endpoints.ts::getHealth")
    assert [i.symbol_id for i in items] == [get_health]
    it = items[0]
    assert (it.distance, it.possible_hops, it.status, it.hops, it.length) == (1, 0, "confirmed", 4, 4)
    handler = next(p["handlerId"] for p in real["facts"]["providers"] if p["key"] == key and "shop" in json.dumps(p["mounts"]) and p["qname"].endswith("build_storefront_router.health"))
    link_id = run("MATCH (l:Link)-[:ON]->(:ContractPoint {keyNorm: $k}) RETURN l.linkId", {"k": key})[0][0]
    site = next(s["siteId"] for l in real["facts"]["links"] if l["id"] == link_id for s in l["sites"])
    assert [(s["node"], s["edge"] and (s["edge"]["type"], s["edge"]["dir"], s["edge"]["status"])) for s in it.chain] == [
        (("ContractPoint", cp[0][0]), None),
        (("Symbol", handler), ("PROVIDES", "<-", "confirmed")),
        (("Link", link_id), ("TO", "<-", "confirmed")),
        (("CallSite", site), ("VIA", "<-", "confirmed")),
        (("Symbol", get_health), ("HAS_CALL", "<-", "confirmed")),
    ]
    link = run("MATCH (l:Link {linkId: $i}) RETURN l.kind, l.status, l.reason, l.matcher, keys(l)", {"i": link_id})[0]
    assert link[:4] == ["http", "confirmed", "key-match", "exact"]
    assert "candidateCount" not in link[4]  # D6: N is derived per VIA edge at render time
    rendered = I.render_item(run, it)
    hops = [x["lifted"] for x in rendered if "lifted" in x]
    qn = lambda name: _qn_id(real, "salesperson/src/api/client.ts::" + name)  # noqa: E731
    fetch_cs = next(h["id"] for h in hops if h["name"] == "fetch")
    assert [(h["ord"], h["label"], h["path"], h["line"], h["urlPath"], h["verbPath"], h["name"]) for h in hops] == [
        (0, "CallSite", "salesperson/src/api/endpoints.ts", 17, "a0.path", "a0.method", "apiFetch"),
        (1, "Symbol", "salesperson/src/api/client.ts", 107, "p0.path", "p0.method", "salesperson/src/api/client.ts::apiFetch"),
        (2, "Symbol", "salesperson/src/api/client.ts", 73, "p0", None, "salesperson/src/api/client.ts::buildUrl"),
        (3, "CallSite", "salesperson/src/api/client.ts", 118, "a0", "a1.method", "fetch"),
    ]
    assert hops[0]["id"] == site and hops[1]["id"] == qn("apiFetch") and hops[2]["id"] == qn("buildUrl") and hops[3]["id"] == fetch_cs
    api_id, sites = qn("apiFetch"), _site_symbol(real)
    via = next(c for c, sym in sites.items() if sym == api_id and _line_of(real, c) == 108)  # `buildUrl(...)` inside apiFetch
    assert (hops[1]["viaSiteId"], hops[1]["viaLine"]) == (via, 108)
    # the rendered chain sits where the Link hop was: ... Link, <4 lifted hops>, CallSite, Symbol
    kinds = ["lifted" if "lifted" in x else x["node"][0] for x in rendered]
    assert kinds == ["ContractPoint", "Symbol", "Link"] + ["lifted"] * 4 + ["CallSite", "Symbol"]
    _check_hop_lines_against_source(hops)
    MEAS["sl4Contract"] = {"key": key, "item": get_health, "distance": it.distance, "hops": it.hops, "lifted": hops,
                           "state": I.state_summary(res)}


@pytest.mark.live
def test_lifted_chain_is_filtered_by_consumer_site_and_ordered(store):
    key = L.BUILD_PREFIX + "lifted_probe"
    if store.exists(key):
        store.delete(key)
    try:
        run = run_on(store, key)
        store.query(key, "CREATE (:Link {linkId: 'L'}), (:CallSite {callSiteId: 'c1', path: 'a.ts', line: 1, calleeName: 'x'}), "
                         "(:CallSite {callSiteId: 'c2', path: 'a.ts', line: 2, calleeName: 'y'}), (:Symbol {symbolId: 's', qname: 'q', path: 'b.ts'})")
        edges = [("c1", 0, "c1"), ("s", 1, "c1"), ("c2", 0, "c2"), ("s", 1, "c2"), ("c1", 2, "c2")]
        for tgt, ord_, site in edges:
            lab = "Symbol {symbolId: $t}" if tgt == "s" else "CallSite {callSiteId: $t}"
            store.query(key, f"MATCH (l:Link {{linkId: 'L'}}), (t:{lab}) CREATE (l)-[:LIFTED_VIA {{ord: $o, siteId: $s, line: 9, urlPath: 'a0'}}]->(t)",
                        {"t": tgt, "o": ord_, "s": site})
        assert [(h["ord"], h["id"]) for h in I.lifted_chain(run, "L", "c1")] == [(0, "c1"), (1, "s")]
        assert [(h["ord"], h["id"]) for h in I.lifted_chain(run, "L", "c2")] == [(0, "c2"), (1, "s"), (2, "c1")]
        assert I.lifted_chain(run, "L", "nope") == []
    finally:
        store.delete(key)


def _line_of(real, site_id):
    for b in real["ordered"]:
        for r in b:
            if r["type"] == "callsite" and r["id"] == site_id:
                return r["line"]


def _check_hop_lines_against_source(hops):
    repo = CPG.parent
    want = {17: ("salesperson/src/api/endpoints.ts", "apiFetch"), 107: ("salesperson/src/api/client.ts", "function apiFetch"),
            73: ("salesperson/src/api/client.ts", "function buildUrl"), 118: ("salesperson/src/api/client.ts", "fetch(")}
    for h in hops:
        path, needle = want[h["line"]]
        assert h["path"] == path
        p = subprocess.run(["git", "-C", str(repo), "show", f"{PINNED_COMMIT}:{path}"], capture_output=True, text=True)
        if p.returncode != 0:
            return  # the pinned commit is not in this clone; the graph-side assertions above stand
        assert needle in p.stdout.splitlines()[h["line"] - 1]


@pytest.mark.live
def test_sl4_every_link_contract_answer_matches_the_assembled_facts(store, real, loaded):
    run, reader = loaded["run"], loaded["reader"]
    facts = real["facts"]
    site_sym = _site_symbol(real)
    by_key = defaultdict(list)
    for l in facts["links"]:
        by_key[l["keyNorm"]].append(l)
    hop_counts = []
    for key, links in sorted(by_key.items()):
        cp = run("MATCH (cp:ContractPoint {keyNorm: $k}) RETURN cp.cpId", {"k": key})[0][0]
        res = I.walk(reader, ("ContractPoint", cp), I.CONTRACT)
        want = {site_sym[s["siteId"]] for l in links for s in l["sites"]}
        got = {i.symbol_id: i for i in res.items()}
        assert set(got) == want, key
        for i in got.values():
            assert (i.distance, i.hops, i.length) == (1, 4, 4) and i.status == ("confirmed" if all(
                s["status"] == "confirmed" for l in links for s in l["sites"] if site_sym[s["siteId"]] == i.symbol_id) else "possible")
    for l in facts["links"]:
        for s in l["sites"]:
            n = run("MATCH (:Link {linkId: $l})-[r:LIFTED_VIA]->() WHERE r.siteId = $s RETURN count(r)", {"l": l["id"], "s": s["siteId"]})[0][0]
            assert n == len(s["liftedVia"])
            got = I.lifted_chain(run, l["id"], s["siteId"])  # the rendered chain, hop by hop, against the assembled hop
            assert [(g["ord"], g["id"], g["line"], g["urlPath"], g["verbPath"], g["viaSiteId"], g["viaLine"], g["path"]) for g in got] == [
                (h["ord"], h["id"], h["line"], h["urlPath"], h["verbPath"], h.get("viaSiteId"), h.get("viaLine"), h["path"])
                for h in s["liftedVia"]]
        for sid in {s["siteId"] for s in l["sites"]}:
            hop_counts.append({"linkId": l["id"], "keyNorm": l["keyNorm"], "consumer": l["consumer"]["qname"], "siteId": sid,
                               "liftedHops": next(len(s["liftedVia"]) for s in l["sites"] if s["siteId"] == sid),
                               "status": l["status"], "answerHops": 4})
    assert len(hop_counts) >= 11 and all(c["liftedHops"] >= 1 for c in hop_counts)
    MEAS["linkChains"] = hop_counts


def _closure(real, anchor):
    """Independent oracle: reverse call-graph closure from the bundle records alone."""
    callers = defaultdict(set)
    for b in real["ordered"]:
        for r in b:
            if r["type"] == "callsite":
                for t in r["targets"]:
                    callers[t["symbolId"]].add(r["symbolId"])
    dist, frontier = {anchor: 0}, [anchor]
    while frontier:
        nxt = []
        for s in frontier:
            for c in sorted(callers[s]):
                if c not in dist:
                    dist[c] = dist[s] + 1
                    nxt.append(c)
        frontier = nxt
    dist.pop(anchor)
    return dist


@pytest.mark.live
def test_sl4_value_mode_from_buildurl_lists_apifetch_and_its_endpoints_callers_through_path_matched_data(store, real, loaded):
    run, reader = loaded["run"], loaded["reader"]
    bu = _qn_id(real, "salesperson/src/api/client.ts::buildUrl")
    api = _qn_id(real, "salesperson/src/api/client.ts::apiFetch")
    res = I.walk(reader, ("Symbol", bu))
    items = _sym(res.items())
    want = _closure(real, bu)
    assert {k: v.distance for k, v in items.items()} == want  # item set and distances, field by field
    # M2 / section 3.3: the direct callers of the anchor stay confirmed; every item further out rests on a
    # function-level continuation (no `arg(i)`/`Ret` rows yet): `possible`, with the reason, never `confirmed`
    direct = {k for k, i in items.items() if i.distance == 1}
    assert direct == {api}
    assert items[api].status == "confirmed" and items[api].possible_hops == 0 and items[api].reason is None
    far = [i for i in items.values() if i.distance > 1]
    assert len(far) == len(items) - 1 == 20
    assert all(i.status == "possible" and i.possible_hops >= 1 and i.reason == "function-level-continuation" for i in far)
    assert all(i.hops == 2 * i.distance for i in far if not i.via_data)  # a DATA detour only ever lengthens the chain
    assert any(i.via_data for i in far) and not items[api].via_data
    assert items[api].distance == 1
    callers_of_api = {sym for b in real["ordered"] for r in b if r["type"] == "callsite" and r["symbolId"] != api
                      for sym in [r["symbolId"]] if any(t["symbolId"] == api for t in r["targets"])}
    ep = {s for s in callers_of_api if _sym_path(real, s) == "salesperson/src/api/endpoints.ts"}
    assert ep and all(items[s].distance == 2 for s in ep)
    assert _qn_id(real, "salesperson/src/api/endpoints.ts::getHealth") in ep
    # the path-matched DATA hop: buildUrl's call site in apiFetch (result `r`) feeds fetch's a0, nothing else
    bu_cs = next(r["id"] for b in real["ordered"] for r in b if r["type"] == "callsite" and r["symbolId"] == api and r["calleeName"] == "buildUrl")
    fetch_cs = next(r["id"] for b in real["ordered"] for r in b if r["type"] == "callsite" and r["symbolId"] == api and r["calleeName"] == "fetch")
    by_data = {k[1]: (k[2], k[3]) for k in res.visited if res.best[k].edge and res.best[k].edge.type == "DATA" and res.best[k].pred[1] == bu_cs}
    assert by_data == {fetch_cs: ("path", "a0")}
    MEAS["sl4Value"] = {"anchor": "buildUrl", "items": len(items), "apiFetch": items[api].distance, "endpointsCallers": len(ep),
                        "maxDistance": max(i.distance for i in items.values()), "state": I.state_summary(res)}


def _sym_path(real, sid):
    return next(r["path"] for b in real["ordered"] for r in b if r["type"] == "symbol" and r["id"] == sid)


@pytest.mark.live
def test_sl4_data_path_match_filters_live_exactly_like_the_bundle_records(store, real, loaded):
    reader = loaded["reader"]
    api = _qn_id(real, "salesperson/src/api/client.ts::apiFetch")
    p0 = next(r["id"] for b in real["ordered"] for r in b if r["type"] == "param" and r["symbolId"] == api and r["index"] == 0)
    datas = [r for b in real["ordered"] for r in b if r["type"] == "data" and r["def"] == p0]
    site_of = _site_symbol(real)
    fields = {}
    for path in ("p0", "p0.path", "p0.method", "p0.query", "p0.body", "p0.nothing", "p1"):
        res = I.walk(reader, ("Param", p0), ("path", path))
        got = {(k[0], k[1], k[3]) for k in res.visited if res.best[k].edge and res.best[k].edge.type == "DATA"
               and res.best[k].pred[1] == p0}
        want = {(("CallSite" if d["use"] in site_of else "Ret" if d["use"] in _rets(real) else "Cond"), d["use"], d["toPath"])
                for d in datas if I.path_overlaps(path, d["fromPath"])}
        assert got == want, path
        fields[path] = len(got)
        # the memory reader over the rows agrees with the database on the whole walk (differential)
        mem = I.walk(MemoryReader.from_rows(real["rows"]), ("Param", p0), ("path", path))
        assert mem.visited == res.visited and {k: e.cost for k, e in mem.best.items()} == {k: e.cost for k, e in res.best.items()}
    assert fields["p0.path"] < fields["p0"] and fields["p0.method"] < fields["p0"]
    assert fields["p0.nothing"] == 0 and fields["p1"] == 0
    assert fields["p0.path"] > 0 and fields["p0.method"] > 0
    mem = I.walk(MemoryReader.from_rows(real["rows"]), ("Param", p0), I.VALUE)
    live = I.walk(reader, ("Param", p0), I.VALUE)
    assert len(live.visited) == len(mem.visited) and live.visited == mem.visited
    MEAS["pathMatchFanOut"] = fields


def _rets(real):
    return {r["id"] for b in real["ordered"] for r in b if r["type"] == "ret"}


@pytest.mark.live
def test_sl4_ac1_data_rows_are_in_the_graph(store, real, loaded):
    run = loaded["run"]
    api = _qn_id(real, "salesperson/src/api/client.ts::apiFetch")
    bu = _qn_id(real, "salesperson/src/api/client.ts::buildUrl")
    q = ("MATCH (u:CallSite {calleeName: $uc})<-[:HAS_CALL]-(:Symbol {symbolId: $us}) "
         "MATCH (u)-[d:DATA {fromPath: $fp, toPath: $tp}]->(x) WHERE labels(x)[0] = $xl AND ($xc IS NULL OR x.calleeName = $xc) "
         "RETURN count(d), min(coalesce(d.status, 'confirmed')), min(coalesce(d.reason, 'intra-flow'))")  # D8: sparse
    rows = run(q, {"uc": "fetch", "us": api, "fp": "r", "tp": "a0", "xl": "CallSite", "xc": "buildUrl"})
    assert rows == [[1, "confirmed", "intra-flow"]]
    assert run(q, {"uc": "buildUrl", "us": api, "fp": "p0.path", "tp": "a0", "xl": "Param", "xc": None}) == [[1, "confirmed", "intra-flow"]]
    assert run(q, {"uc": "fetch", "us": api, "fp": "p0.method", "tp": "a1.method", "xl": "Param", "xc": None}) == [[1, "confirmed", "intra-flow"]]
    ret = run("MATCH (:Symbol {symbolId: $b})-[:HAS_RETURN]->(r:Ret)-[d:DATA {fromPath: 'p0', toPath: 'ret'}]->(p:Param) "
              "WHERE p.index = 0 RETURN r.pointId, p.pointId, coalesce(d.status, 'confirmed'), coalesce(d.reason, 'intra-flow') "
              "ORDER BY r.pointId", {"b": bu})
    recs = [r for b in real["ordered"] for r in b]
    rets = {r["id"] for r in recs if r["type"] == "ret" and r["symbolId"] == bu}
    p0 = next(r["id"] for r in recs if r["type"] == "param" and r["symbolId"] == bu and r["index"] == 0)
    want = sorted([d["use"], p0, d["status"], d["reason"]] for d in recs
                  if d["type"] == "data" and d["use"] in rets and d["def"] == p0 and d["fromPath"] == "p0" and d["toPath"] == "ret")
    assert want and ret == want


@pytest.mark.live
def test_sl5_density_and_timing_for_both_units(store, real, loaded):
    out = {}
    for root, tag in (("falkor-chat/server", "fcs"), ("salesperson", "sp")):
        m = L.measure_unit_density(store, real["bundles"][root], tag=tag)
        out[root] = m
        assert not [k for k in store.conn.keys("*") if "__build_" in k]  # the scratch key is gone
        assert m["files"] == _count(real["bundles"][root], "file")
        assert m["mbPerFile"] <= 0.7, f"{root}: {m['mbPerFile']:.3f} MB per file over the 0.7 limit"
        assert m["dropped"] == {}
    combined = loaded["rep2"].memory["total_graph_sz_mb"]
    files = sum(real["rows"].files.values())
    MEAS["sl5"] = {
        "units": out, "combinedTotalGraphMb": combined, "combinedFiles": files, "combinedMbPerFile": combined / files,
        "combinedNodes": loaded["rep2"].node_total, "combinedEdges": loaded["rep2"].edge_total,
        "limitMbPerFile": 0.7,
        "loadSeconds": {"firstLoad": loaded["rep1"].seconds, "secondLoad": loaded["rep2"].seconds},
        "joernParseSecondsFromS11Notes_run1": {"falkor-chat/server": 12.9, "salesperson": 7.1},
        "s11ExtractOverParseFromS11Notes_run1": {"falkor-chat/server": 0.99, "salesperson": 1.20},
        "note": "memory is GRAPH.MEMORY USAGE total_graph_sz_mb (integer MB); S11 figures are quoted from S11-notes.md, not re-measured",
    }
    for root, parse in (("falkor-chat/server", 12.9), ("salesperson", 7.1)):
        s = out[root]["seconds"]
        out[root]["loadOverParse"] = (s["indexes"] + s["nodes"] + s["edges"] + s["constraints"]) / parse


@pytest.mark.live
def test_sl6_two_loads_give_identical_link_sets_and_bfs_answers(loaded, real):
    a, b = loaded["snap1"], loaded["snap2"]
    assert a["graph"] == b["graph"]
    assert a["answers"] == b["answers"]
    assert [row[0] for row in a["graph"]["links"]] == sorted(l["id"] for l in real["facts"]["links"])  # the assembled facts
    assert len(a["graph"]["links"]) == 11 and len(a["answers"]) == 11 + 2 - (11 - len({l["keyNorm"] for l in real["facts"]["links"]}))
    assert loaded["rep1"].snapshot_id == loaded["rep2"].snapshot_id
    assert (loaded["rep1"].node_total, loaded["rep1"].edge_total) == (loaded["rep2"].node_total, loaded["rep2"].edge_total)
    MEAS["sl6"] = {"linkRows": len(a["graph"]["links"]), "viaRows": len(a["graph"]["via"]), "liftedRows": len(a["graph"]["lifted"]),
                   "answers": len(a["answers"]), "identical": True}


@pytest.mark.live
def test_zz_bfs_measurements_and_summary(store, real, loaded):
    """M4: peak frontier and field-path state counts over a spread of anchors; writes the measurements file."""
    run, reader = loaded["run"], loaded["reader"]
    runs = {}
    for key in sorted({l["keyNorm"] for l in real["facts"]["links"]}):
        cp = run("MATCH (cp:ContractPoint {keyNorm: $k}) RETURN cp.cpId", {"k": key})[0][0]
        runs["contract " + key] = I.state_summary(I.walk(reader, ("ContractPoint", cp), I.CONTRACT))
    top = run("MATCH (s:Symbol)<-[c:CALLS]-() RETURN s.qname, s.symbolId, count(c) AS n ORDER BY n DESC, s.qname LIMIT 6")
    for qname, sid, n in top:
        res = I.walk(reader, ("Symbol", sid))
        runs[f"value {qname} (callers={n})"] = {**I.state_summary(res), "items": len(res.items())}
    api = _qn_id(real, "salesperson/src/api/client.ts::apiFetch")
    p0 = next(r["id"] for b in real["ordered"] for r in b if r["type"] == "param" and r["symbolId"] == api and r["index"] == 0)
    for path in ("p0", "p0.path", "p0.method"):
        res = I.walk(reader, ("Param", p0), ("path", path))
        runs[f"param apiFetch.p0 @ {path}"] = {**I.state_summary(res), "items": len(res.items())}
    peak = max(v["peakFrontier"] for v in runs.values())
    MEAS["bfs"] = {"runs": runs, "peakFrontierMax": peak, "statesMax": max(v["states"] for v in runs.values()),
                   "fieldPathStatesMax": max(v["fieldPathStates"] for v in runs.values())}
    assert all(v["levels"] < 64 for v in runs.values())
    MEAS["generatedBy"] = "spikes/vertical-slice/tests/test_live_slice.py (pytest -m live)"
    MEAS["bundles"] = {root: {"files": _count(b, "file"), "symbols": _count(b, "symbol")} for root, b in real["bundles"].items()}
    MEAS["dropped"] = loaded["rep2"].dropped
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    MEASUREMENTS.write_text(json.dumps(MEAS, indent=1, sort_keys=True, default=str) + "\n")
    sl5 = MEAS["sl5"]
    lines = [f"S13 measurements -> {MEASUREMENTS}",
             f"graph: {sl5['combinedNodes']} nodes, {sl5['combinedEdges']} edges, {sl5['combinedTotalGraphMb']} MB over {sl5['combinedFiles']} files",
             f"load seconds (second load): {MEAS['loads'][1]['seconds']}"]
    for root, u in sl5["units"].items():
        lines.append(f"{root}: {u['files']} files, {u['nodes']} nodes, {u['edges']} edges, {u['totalGraphMb']} MB "
                     f"({u['mbPerFile']:.3f} MB/file, limit 0.7), {u['nodesPerFile']:.1f} nodes/file, {u['edgesPerFile']:.1f} edges/file, "
                     f"load/parse {u['loadOverParse']:.3f}")
    lines.append(f"BFS: peak frontier {MEAS['bfs']['peakFrontierMax']}, max states {MEAS['bfs']['statesMax']}, "
                 f"max field-path states {MEAS['bfs']['fieldPathStatesMax']}")
    lines += [f"link chain {c['keyNorm']}: lifted hops {c['liftedHops']}, answer hops {c['answerHops']}" for c in MEAS["linkChains"]]
    print("\n" + "\n".join(lines))
