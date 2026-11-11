"""P1-19 dispatch 1: the loader (``ccg/store/loader.py``) against a fake store: sequence, batching, timeout rule, created-count check,
closed edge list, pre-write refusals, key ownership, failure cleanup. Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §4, §8
P1-19 and ``-p0-graph.md`` §4. Expected values are written out here or read from the assembler/the documents, never from the loader."""
from __future__ import annotations

import re

import pytest
from loader_helpers import Boom, FakeStore, fail_at, tiny_rows

from ccg.assemble import rows as AR
from ccg.store import loader as L
from ccg.sysname import SystemNameError

SYSTEM = "shop"
BUILD_PREFIX = "ccg_shop__build_"


def _build_key(rows):
    return BUILD_PREFIX + rows.snapshot["snapshotId"]


def _kinds(store):
    """The call sequence as coarse kinds: index / node / edge / snapshot / constraint / poll / other."""
    out = []
    for c in store.calls:
        if c[0] == "query":
            q = c[2]
            out.append("index" if q.startswith("CREATE INDEX") else "snapshot" if "(s:Snapshot)" in q else
                       "node" if q.startswith("UNWIND $rows AS r CREATE (n:") else "edge" if "CREATE (a)-[e:" in q else "other-query")
        elif c[0] in ("constraint", "constraints"):
            out.append("constraint" if c[0] == "constraint" else "poll")
        else:
            out.append(c[0])
    return out


def _load(store=None, rows=None, **kw):
    rows = rows or tiny_rows(SYSTEM)
    store = store or FakeStore()
    return store, rows, L.load(store, rows, SYSTEM, **kw)


# ------------------------------------------------------------------------------------- sequence and shape

def test_the_sequence_is_indexes_nodes_edges_constraints_polled_then_snapshot_last():
    store, rows, report = _load()
    kinds = _kinds(store)
    first = {k: kinds.index(k) for k in ("index", "node", "edge", "constraint", "poll", "snapshot")}
    last = {k: len(kinds) - 1 - kinds[::-1].index(k) for k in first}
    assert last["index"] < first["node"] <= last["node"] < first["edge"] <= last["edge"] < first["constraint"]
    assert last["constraint"] < first["poll"] <= last["poll"] < first["snapshot"]
    assert kinds[-1] == "snapshot" and kinds.count("snapshot") == 1          # :Snapshot last, once
    assert kinds.count("index") == 20 and kinds.count("constraint") == 12   # p1-graph §4: 20 indexes, 12 constraints
    assert report.build_key == _build_key(rows) and report.snapshot_id == rows.snapshot["snapshotId"]


def test_every_store_key_is_the_systems_build_key_and_nothing_else_is_ever_touched():
    store, rows, _ = _load()
    keys = {c[1] for c in store.calls if len(c) > 1 and isinstance(c[1], str)}
    assert keys == {_build_key(rows)}
    assert not store.ops("delete")                      # a clean load deletes nothing
    assert "rename" not in {c[0] for c in store.calls}  # the swap's, never the loader's


def test_the_ddl_order_inside_the_load_indexes_before_the_matching_constraint():
    store, _, _ = _load()
    idx = [re.fullmatch(r"CREATE INDEX FOR \(n:(\w+)\) ON \(n\.(\w+)\)", c[2]).groups() for c in store.queries() if c[2].startswith("CREATE INDEX")]
    assert ("ConfigEntry", "entryId") in idx and ("ConfigEntry", "path") in idx and len(set(idx)) == 20
    assert set(store.constraints_made) <= set(idx) and len(store.constraints_made) == 12
    assert ("ConfigEntry", "entryId") in store.constraints_made


def test_node_and_edge_statement_texts():
    assert L.node_statement("Symbol") == "UNWIND $rows AS r CREATE (n:Symbol) SET n = r"
    assert L.edge_statement("Link", "BOUND_BY", "ConfigEntry", True) == (
        "UNWIND $rows AS r MATCH (a:Link {linkId: r.s}) MATCH (b:ConfigEntry {entryId: r.t}) CREATE (a)-[e:BOUND_BY]->(b) SET e = r.p")
    assert L.edge_statement("Unit", "SKIPS", "SkippedFile", False) == (
        "UNWIND $rows AS r MATCH (a:Unit {unitId: r.s}) MATCH (b:SkippedFile {unitId: r.s, path: r.t}) CREATE (a)-[e:SKIPS]->(b)")
    with pytest.raises(ValueError):
        L.node_statement("Symbol) DETACH DELETE (n")


def test_mounted_by_is_created_never_merged_and_both_edges_of_a_pair_are_sent():
    """G6: two edges between one Symbol and CallSite differ only by mountPath and ord. A MERGE on (s, t) would create one."""
    store, rows, _ = _load()
    for c in store.queries():
        assert "MERGE" not in c[2]
    mount = [c for c in store.queries() if "[e:MOUNTED_BY]" in c[2]]
    assert len(mount) == 1 and "CREATE (a)-[e:MOUNTED_BY]->(b)" in mount[0][2]
    assert [r["p"]["mountPath"] for r in mount[0][3]["rows"]] == ["/a", "/b"]
    assert not any("MERGE" in L.edge_statement(*g, True) for g in L.EDGE_ORDER)


def test_a_mixed_props_batch_gets_an_empty_map_for_the_row_without_props():
    store, rows, _ = _load()
    data = [c for c in store.queries() if "(a:CallSite {callSiteId: r.s}) MATCH (b:Param" in c[2] and "[e:DATA]" in c[2]]
    assert len(data) == 1 and data[0][2].endswith("SET e = r.p")
    assert [("p" in r, r["p"]) for r in data[0][3]["rows"]][1] == (True, {})


def test_the_loader_tables_equal_the_assemblers_so_no_group_or_label_can_drift():
    assert L.ID_PROP == AR.ID_PROP and L.NODE_ORDER == AR.NODE_ORDER and L.EDGE_ORDER == AR.EDGE_GROUPS
    assert len(L.EDGE_ORDER) == 34 and ("Symbol", "MOUNTED_BY", "CallSite") in L.EDGE_ORDER


# ------------------------------------------------------------------------------------- batching

def test_batches_never_exceed_batch_max_and_split_exactly():
    assert L.BATCH_MAX == 20_000
    assert [len(b) for b in L.batches(list(range(45_000)))] == [20_000, 20_000, 5_000]
    assert list(L.batches([])) == []
    with pytest.raises(ValueError):
        list(L.batches([1], 0))


def test_a_45000_row_group_loads_in_three_batches_each_checked():
    rows = tiny_rows(SYSTEM)
    rows.edges[("CallSite", "CALLS", "Symbol")] = [{"s": "CallSite1", "t": "Symbol1"} for _ in range(45_000)]
    store, _, report = _load(rows=rows)
    sizes = [len(c[3]["rows"]) for c in store.queries() if "[e:CALLS]" in c[2]]
    assert sizes == [20_000, 20_000, 5_000] and max(len(c[3]["rows"]) for c in store.queries() if c[3] and "rows" in c[3]) <= 20_000
    assert report.max_batch == 20_000


@pytest.mark.parametrize("size", [0, -1, 20_001, 100_000])
def test_a_batch_size_outside_one_to_batch_max_is_refused_before_any_call(size):
    store = FakeStore()
    with pytest.raises(ValueError):
        L.load(store, tiny_rows(SYSTEM), SYSTEM, batch_size=size)
    assert store.calls == []


# ------------------------------------------------------------------------------------- the timeout rule

def test_every_query_carries_a_timeout_at_most_timeout_max_when_it_is_nonzero():
    for cap in (10_000, 3_000, 1):
        store, _, report = _load(FakeStore(timeout_max=cap))
        assert store.queries() and all(0 < c[4] <= cap for c in store.queries()), cap
        assert all(c[2] is not None and 0 < c[2] <= cap for c in store.ops("constraints"))
        assert report.timeout_max == cap and report.query_timeout_used is True


def test_with_timeout_max_zero_no_query_carries_a_timeout():
    store, _, report = _load(FakeStore(timeout_max=0))
    assert all(c[4] is None for c in store.queries()) and report.query_timeout_used is False


# ------------------------------------------------------------------------------------- created-count check

def test_a_lost_edge_in_a_batch_fails_the_build_names_the_group_and_removes_the_build_key():
    store = FakeStore(created_delta=-1)
    with pytest.raises(L.LoadError, match=r"Unit-\[IN_REPO\]->Repo.*1 rows but 0 edges"):
        L.load(store, tiny_rows(SYSTEM), SYSTEM)
    assert store.ops("delete") == [("delete", _build_key(tiny_rows(SYSTEM)))]
    assert not any("(s:Snapshot)" in c[2] for c in store.queries())   # never written: the build is not "complete"


def test_an_extra_edge_is_also_a_failure():
    with pytest.raises(L.LoadError, match="edges created"):
        L.load(FakeStore(created_delta=+1), tiny_rows(SYSTEM), SYSTEM)


def test_a_missing_created_count_fails_closed():
    with pytest.raises(L.LoadError, match="no created-edge count"):
        L.load(FakeStore(created_none=True), tiny_rows(SYSTEM), SYSTEM)


# ------------------------------------------------------------------------------------- constraints

def test_constraints_are_polled_until_operational():
    store, _, report = _load(FakeStore(pending_polls=3), poll_interval=0, sleep=lambda s: None)
    assert len(store.ops("constraints")) == 4 and len(report.constraints) == 12
    assert all(c["status"] == "OPERATIONAL" for c in report.constraints)


def test_a_failed_constraint_fails_the_build_no_snapshot_and_the_build_key_is_removed():
    store = FakeStore(failed_label="Link")
    with pytest.raises(L.LoadError, match=r"FAILED.*Link"):
        L.load(store, tiny_rows(SYSTEM), SYSTEM, poll_timeout=1, poll_interval=0, sleep=lambda s: None)   # small: a mutant must not spin 300 s
    assert not any("(s:Snapshot)" in c[2] for c in store.queries()) and len(store.ops("delete")) == 1


def test_constraints_that_never_become_operational_time_out_with_an_injected_clock():
    ticks = iter(range(0, 10_000, 100))
    store = FakeStore(pending_polls=10**6)
    with pytest.raises(L.LoadError, match="not OPERATIONAL within"):
        L.load(store, tiny_rows(SYSTEM), SYSTEM, poll_timeout=300, poll_interval=0, sleep=lambda s: None, clock=lambda: next(ticks))
    assert len(store.ops("delete")) == 1


# ------------------------------------------------------------------------------------- refusals before any write

def _refused(rows, match, system=SYSTEM):
    store = FakeStore()
    with pytest.raises(L.LoadError, match=match):
        L.load(store, rows, system)
    assert store.calls == []        # not one store call: nothing written, nothing read


@pytest.mark.parametrize("group", [("CallSite", "PROVIDES", "ContractPoint"), ("ConfigEntry", "PROVIDES", "ContractPoint"),
                                   ("ConfigEntry", "CONSUMES", "ContractPoint"), ("ConfigEntry", "WRITES", "ContractPoint"),
                                   ("Symbol", "READS", "ContractPoint"), ("Symbol", "WRITES", "Param"), ("Unit", "NEW_EDGE", "Repo"),
                                   ("Link", "LAUNCHES", "ConfigEntry")])
def test_a_row_in_an_edge_group_outside_the_closed_list_fails_before_any_call(group):
    rows = tiny_rows(SYSTEM)
    rows.edges[group] = [{"s": "x", "t": "y"}]
    _refused(rows, re.escape(str(group[1])))


def test_an_unknown_node_label_fails_before_any_call():
    rows = tiny_rows(SYSTEM)
    # a row that HAS unitId and path: only the unknown-label check can refuse it (the id-property check would not), else it is dropped silently
    rows.nodes["SchemaEntry"] = [{"unitId": "u", "path": "p"}]
    _refused(rows, r"node labels with no load statement.*SchemaEntry")


def test_a_label_injection_in_an_edge_group_is_a_closed_list_failure_not_cypher():
    rows = tiny_rows(SYSTEM)
    rows.edges[("Unit", "IN_REPO]->(x) DETACH DELETE (x", "Repo")] = [{"s": "a", "t": "b"}]
    _refused(rows, "edge groups")


def test_a_node_row_without_its_id_property_is_refused_naming_label_and_index_not_value():
    rows = tiny_rows(SYSTEM)
    del rows.nodes["Symbol"][0]["symbolId"]
    _refused(rows, r"Symbol.*symbolId")


def test_an_edge_row_without_s_and_t_is_refused():
    rows = tiny_rows(SYSTEM)
    rows.edges[("Unit", "IN_REPO", "Repo")] = [{"s": "Unit1"}]
    _refused(rows, "IN_REPO")


@pytest.mark.parametrize("value", [None, "", 0, 7, ["Symbol1"]])
def test_a_node_row_whose_id_value_is_not_a_nonempty_string_is_refused_before_any_call(value):
    rows = tiny_rows(SYSTEM)
    rows.nodes["Symbol"][1]["symbolId"] = value
    _refused(rows, r"Symbol row 1.*symbolId")


@pytest.mark.parametrize("value", [None, ""])
def test_a_skipped_file_row_with_an_empty_unit_id_or_path_is_refused(value):
    for prop in ("unitId", "path"):
        rows = tiny_rows(SYSTEM)
        rows.nodes["SkippedFile"][0][prop] = value
        _refused(rows, rf"SkippedFile row 0.*{prop}")


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s.pop("snapshotId"), "snapshotId"),
    (lambda s: s.update(systemId="other"), "systemId"),
    (lambda s: s.update(graphModelVersion="1.0"), "graphModelVersion"),
])
def test_the_snapshot_row_must_name_this_system_and_model_1_1(mutate, match):
    rows = tiny_rows(SYSTEM)
    mutate(rows.snapshot)
    _refused(rows, match)


# ------------------------------------------------------------------------------------- names and key ownership

@pytest.mark.parametrize("name", ["shop_", "shop__x", "Shop", "", "-shop", "shop*", "ccg__shop", "a" * 49, "shop:x", "shop\n"])
def test_a_system_name_outside_the_whitelist_is_refused_before_any_key_is_formed_or_store_call(name):
    store = FakeStore()
    with pytest.raises(SystemNameError):
        L.load(store, tiny_rows(SYSTEM), name)
    assert store.calls == []


@pytest.mark.parametrize("key", ["ccg_shop", "ccg_shop2", "ccg_shop2__build_1", "ccg_shop__x", "ccg_shop___build_1", "ccg_shop__ov_1",
                                 "ccg_shop__build_", "ccg_shop__build_a b", "ccg_shop__build_a__b", "ccg_other__build_1", "shop",
                                 "shop__build_1", "shop_catalog", "ccg_shop:verdict_lock", "ccg_shop__build_1*", "ccg_shop__build_1\n"])
def test_the_loader_refuses_every_key_it_does_not_own(key):
    with pytest.raises(L.LoadError, match="not owned"):
        L.owned_build_key(SYSTEM, key)


@pytest.mark.parametrize("key", ["ccg_shop__build_1", "ccg_shop__build_a1b2c3", "ccg_shop__build_" + "0123456789abcdef" * 2, "ccg_shop__build_a-b"])
def test_the_loader_owns_this_systems_build_keys(key):
    assert L.owned_build_key(SYSTEM, key) == key


def test_the_owned_key_check_is_per_system():
    assert L.owned_build_key("a-b_c", "ccg_a-b_c__build_7") == "ccg_a-b_c__build_7"
    with pytest.raises(L.LoadError):
        L.owned_build_key("a", "ccg_a-b_c__build_7")


def test_a_preexisting_build_key_refuses_the_load_unless_drop_stale_and_only_then_deletes_it():
    rows = tiny_rows(SYSTEM)
    key = _build_key(rows)
    store = FakeStore(existing={key})
    with pytest.raises(L.LoadError, match="already exists"):
        L.load(store, rows, SYSTEM)
    assert not store.queries() and not store.ops("delete")
    store = FakeStore(existing={key})
    L.load(store, rows, SYSTEM, drop_stale=True)
    assert store.ops("delete") == [("delete", key)] and store.calls.index(("delete", key)) < store.calls.index(store.queries()[0])


# ------------------------------------------------------------------------------------- failure cleanup

@pytest.mark.parametrize("n", [2, 5, 25, 40])
def test_any_failure_mid_load_removes_the_build_key_and_reraises_the_original_error(n):
    rows = tiny_rows(SYSTEM)
    store = FakeStore(fail_on=fail_at(n))
    with pytest.raises(Boom):
        L.load(store, rows, SYSTEM)
    assert store.ops("delete") == [("delete", _build_key(rows))] and _build_key(rows) not in store.keys


def test_an_interrupt_mid_load_also_cleans_up():
    rows = tiny_rows(SYSTEM)
    store = FakeStore(fail_on=fail_at(30, KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        L.load(store, rows, SYSTEM)
    assert store.ops("delete") == [("delete", _build_key(rows))]


def test_a_failing_cleanup_does_not_mask_the_original_error():
    class Stuck(FakeStore):
        def delete(self, key):
            raise RuntimeError("cleanup also failed")
    with pytest.raises(Boom):
        L.load(Stuck(fail_on=fail_at(3)), tiny_rows(SYSTEM), SYSTEM)


def test_a_failure_before_the_build_key_exists_deletes_nothing():
    store = FakeStore(fail_on=fail_at(1))            # the first index statement creates the key; it failed
    with pytest.raises(Boom):
        L.load(store, tiny_rows(SYSTEM), SYSTEM)
    assert store.ops("delete") == []


# ------------------------------------------------------------------------------------- report

def test_the_report_is_filled_in_place_and_returned():
    rep = L.LoadReport()
    store, rows, got = _load(report=rep)
    assert got is rep and rep.node_total == rows.node_count() + 1 and rep.edge_total == rows.edge_count()
    assert rep.nodes["Symbol"] == 2 and rep.edges["Symbol-MOUNTED_BY->CallSite"] == 2
    assert rep.live_key == "ccg_shop" and rep.batches == len([c for c in store.queries() if c[3] and "rows" in c[3]])
    assert set(rep.seconds) >= {"indexes", "nodes", "edges", "constraints"}


def test_zero_retry_is_an_explicit_zero_retry_object_not_the_client_default():
    r = L.zero_retry()
    assert r is not None and r._retries == 0


def test_the_load_report_echoes_unattested_units_as_a_report_line_and_writes_them_to_no_store_call():
    """spec6 'load report (P1-19)': the assembly report's ``unattested_units`` is echoed; the graph carries no attestation property."""
    rows = tiny_rows(SYSTEM)
    rows.unattested_units = ["unit-zz-planted", "unit-aa-planted"]
    store, _, report = _load(rows=rows)
    assert report.unattested_units == ["unit-zz-planted", "unit-aa-planted"] and report.unattested_units is not rows.unattested_units
    assert not any("planted" in repr(c) for c in store.calls)          # a report line only: no query, param or snapshot property carries it
    assert "unattested" not in repr(rows.snapshot).lower()
    store, _, report = _load(rows=tiny_rows(SYSTEM))
    assert report.unattested_units == []


# ------------------------------------------------------------------------------------- LiveStore forwards the per-query timeout

class _RecGraph:
    def __init__(self, log, key):
        self.log, self.key = log, key

    def query(self, *args):
        self.log.append((self.key,) + args)
        return "reply"


class _RecDb:
    """``select_graph(key).query(...)`` recorded with the exact positional arguments the store passed."""

    def __init__(self):
        self.log = []

    def select_graph(self, key):
        return _RecGraph(self.log, key)


def test_livestore_passes_the_timeout_to_every_query_and_none_when_the_cap_is_zero():
    key = BUILD_PREFIX + "1"
    for cap in (10_000, 3_000, 1):
        db = _RecDb()
        store = L.LiveStore(db, SYSTEM)
        store.query(key, "RETURN 1", {"a": 1}, cap)
        assert db.log == [(key, "RETURN 1", {"a": 1}, cap)], cap
    db = _RecDb()
    store = L.LiveStore(db, SYSTEM)
    store.query(key, "RETURN 1", {"a": 1}, None)
    assert db.log == [(key, "RETURN 1", {"a": 1})]                        # no timeout argument at all when the cap is zero


def test_livestore_constraints_poll_forwards_the_timeout_too():
    class Res:
        result_set = []
    db = _RecDb()
    db.select_graph = lambda key: type("G", (), {"query": lambda self, *a: (db.log.append((key,) + a), Res())[1]})()
    L.LiveStore(db, SYSTEM).constraints(BUILD_PREFIX + "1", 3_000)
    assert db.log == [(BUILD_PREFIX + "1", "CALL db.constraints()", None, 3_000)]


# ---- P1-19 dispatch 2: the 4,096-byte bound on an indexed value (live-observed on v4.18.11: > 4,096 bytes is not indexed, an equality
# anchor then MISSES it, and a UNIQUE constraint over it crashes the server; ``tests/test_fuzz_idhash_live.py`` has the observations) ----

INDEXED_VALUE_MAX = 4096


@pytest.mark.parametrize("label,prop,row", [("Name", "nameKey", 0), ("Symbol", "symbolId", 1), ("Repo", "repoId", 0), ("Link", "linkId", 1)])
@pytest.mark.parametrize("value,ok", [("k" * 4096, True), ("é" * 2048, True), ("k" * 4097, False), ("é" * 2049, False), ("k" * 100_000, False)])
def test_an_id_value_over_4096_utf8_bytes_is_refused_before_any_call(label, prop, row, value, ok):
    rows = tiny_rows(SYSTEM)
    rows.nodes[label][row][prop] = value
    store = FakeStore()
    if ok:
        L.load(store, rows, SYSTEM)
        return
    with pytest.raises(L.LoadError, match=rf"{label} row {row}.*{prop}.*4,?096") as ei:
        L.load(store, rows, SYSTEM)
    assert store.calls == [] and value[:20] not in str(ei.value)


@pytest.mark.parametrize("label,prop", [("Symbol", "qname"), ("Symbol", "nameNorm"), ("File", "path"), ("ContractPoint", "keyNorm"), ("ConfigEntry", "path")])
def test_every_other_indexed_property_obeys_the_same_bound_and_an_unindexed_one_does_not(label, prop):
    rows = tiny_rows(SYSTEM)
    rows.nodes[label][0][prop] = "q" * 4097
    with pytest.raises(L.LoadError, match=rf"{label} row 0.*{prop}"):
        L.load(FakeStore(), rows, SYSTEM)
    rows.nodes[label][0][prop] = "q" * 4096
    L.load(FakeStore(), rows, SYSTEM)
    rows.nodes["Name"][0]["name"] = "z" * 100_000                     # ``Name.name`` has no index: any length loads
    L.load(FakeStore(), rows, SYSTEM)


def test_a_lone_surrogate_in_an_indexed_value_is_a_load_error_not_a_unicode_error():
    rows = tiny_rows(SYSTEM)
    rows.nodes["Name"][0]["nameKey"] = "a\ud800b"
    with pytest.raises(L.LoadError, match="Name row 0.*nameKey"):
        L.load(FakeStore(), rows, SYSTEM)


def test_the_bounded_properties_are_exactly_the_ddl_indexes():
    from ccg.store import parse_ddl
    assert L.INDEXED_PROPS == {(s[1], s[2]) for s in parse_ddl() if s[0] == "index"} and len(L.INDEXED_PROPS) == 20


@pytest.mark.parametrize("ws", [" ", "\t", "\n", "\v", "\f", "\r"])
@pytest.mark.parametrize("label,prop", [("Name", "nameKey"), ("File", "path"), ("Symbol", "qname"), ("ContractPoint", "keyNorm")])
@pytest.mark.parametrize("shape", ["{w}x", "x{w}", "{w}x{w}", "{w}"])
def test_an_indexed_value_edged_with_ascii_whitespace_is_refused_before_any_call(ws, label, prop, shape):
    rows = tiny_rows(SYSTEM)
    rows.nodes[label][0][prop] = shape.format(w=ws)
    store = FakeStore()
    with pytest.raises(L.LoadError, match=rf"{label} row 0.*{prop}.*whitespace"):
        L.load(store, rows, SYSTEM)
    assert store.calls == []


@pytest.mark.parametrize("value", ["a b", "a\tb", "a\nb", "\u00a0x", "x\u2003", "\u3000x\u3000", "\u200bx", "x\u0085"])
def test_interior_ascii_whitespace_and_edge_unicode_spaces_load(value):
    rows = tiny_rows(SYSTEM)
    rows.nodes["File"][0]["path"] = value                              # the engine trims only the six ASCII characters, and only at the ends
    L.load(FakeStore(), rows, SYSTEM)


def test_an_unindexed_property_may_be_whitespace_edged():
    rows = tiny_rows(SYSTEM)
    rows.nodes["Name"][0]["name"] = "  padded\n"
    L.load(FakeStore(), rows, SYSTEM)
