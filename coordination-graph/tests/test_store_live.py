"""Live tests of `coordgraph.store` against the EPHEMERAL FalkorDB container only (marker `live`).

The container comes from `tests/fixtures/live.py` (`falkordb/falkordb:v4.18.11`, 127.0.0.1, a free
port, no volume). Every test gets its own graph key inside it, and `graph` re-verifies host, port
and container name before any write: the shared `falkordb-dev` (6379) is never contacted.
Run with `-m live`.
"""
import os
import time
import uuid
from pathlib import Path

import pytest

from coordgraph import store
from coordgraph.discover import discover
from coordgraph.keys import KeyHygieneError
from coordgraph.parse.backlog import parse_backlog
from coordgraph.parse.ledger import parse_ledger
from coordgraph.parse.requirements import parse_requirements
from coordgraph.parse.review import parse_review
from tests.fixtures.live import SHARED_PORTS, falkordb_container  # noqa: F401  (session fixture)

pytestmark = pytest.mark.live
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def conn(falkordb_container):  # noqa: F811
    """`(db, graph)` on a fresh graph key of the ephemeral container, bootstrapped."""
    c = falkordb_container
    assert c.host == "127.0.0.1" and c.port not in SHARED_PORTS and c.port != 6379
    assert c.name.startswith("coordgraph-test-") and c.exists()
    env = {"COORDGRAPH_EPHEMERAL_PORT": str(c.port)}
    db, graph = store.connect(c.host, c.port, f"t-{uuid.uuid4().hex[:10]}", env)
    assert db.connection.connection_pool.connection_kwargs["port"] == c.port
    store.bootstrap(db, graph, graph.name)
    return db, graph


def count(graph, label: str, where: str = "") -> int:
    return graph.query(f"MATCH (n:{label}) {where} RETURN count(n)").result_set[0][0]


def ledger(path="a/docs/plans/x-coordination.md", units=("U1", "U2"), sha="s1", **extra) -> tuple[dict, list[dict]]:
    meta = {"path": path, "kind": "ledger", "component": "a", "slug": "x", "sha256": sha, "bytes": 1,
            "parserVersion": 1, "warnings": [], "rowCount": len(units),
            "citedReviews": [], "resolvedReviews": [], **extra}
    rows = [{"key": f"{path}#{u}", "unitKey": u, "unitLabel": u, "statusToken": "accepted", "ledgerPath": path,
             "slug": "x", "component": "a", "tableIdx": 1, "rowIdx": i, "unit": u, "cites": [], "rowHash": f"h{u}"}
            for i, u in enumerate(units, 1)]
    return meta, rows


def backlog(path, refs, sha="b1") -> tuple[dict, list[dict]]:
    meta = {"path": path, "kind": "backlog", "component": "a", "slug": "BACKLOG", "sha256": sha, "bytes": 1,
            "parserVersion": 1, "warnings": [], "rowCount": len(refs), "scope": "a"}
    rows = [{"ref": r, "itemId": r, "scope": "a", "backlogPath": path, "ord": i, "title": "t", "open": True,
             "itemHash": "h"} for i, r in enumerate(refs, 1)]
    return meta, rows


# --- bootstrap ---------------------------------------------------------------------------------

def test_bootstrap_creates_note_4_2_schema_and_is_idempotent(conn):
    db, graph = conn
    indexes, constraints = store.existing_schema(graph)
    assert indexes == set(store.INDEXES)
    assert constraints == {pair: "OPERATIONAL" for pair in store.UNIQUE}
    nodes = graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
    again = store.bootstrap(db, graph, graph.name)
    assert not again.changed
    assert store.existing_schema(graph) == (indexes, constraints)
    assert graph.query("MATCH (n) RETURN count(n)").result_set[0][0] == nodes == 1
    rows = graph.query("MATCH (m:SyncState {id:'mirror'}) RETURN m.schemaVersion").result_set
    assert rows == [[store.SCHEMA_VERSION]]


def test_unique_constraints_are_enforced_after_bootstrap(conn):
    _, graph = conn
    graph.query("CREATE (:Unit {key:'k'})")
    with pytest.raises(Exception, match="unique constraint violation on node of type Unit"):
        graph.query("CREATE (:Unit {key:'k'})")


def test_bootstrap_on_a_partial_schema_completes_it(conn, falkordb_container):  # noqa: F811
    c = falkordb_container
    db, graph = store.connect(c.host, c.port, f"t-{uuid.uuid4().hex[:10]}", {"COORDGRAPH_EPHEMERAL_PORT": str(c.port)})
    graph.query("CREATE INDEX FOR (n:Source) ON (n.path)")  # as if a previous run died after one index
    result = store.bootstrap(db, graph, graph.name)
    assert ("Source", "path") not in result.created_indexes
    assert ("Source", "path") in result.created_constraints
    indexes, constraints = store.existing_schema(graph)
    assert indexes == set(store.INDEXES) and set(constraints) == set(store.UNIQUE)
    assert len(result.created_indexes) == len(store.INDEXES) - 1


# --- per-source replace ------------------------------------------------------------------------

def test_replace_writes_the_source_and_its_units_with_absent_not_empty(conn):
    _, graph = conn
    meta, rows = ledger(docStatus=None, aliases=[])
    rows[0].update(needs=[], statusText=None, itemId="  T-1 ")  # needs [] = recorded none; statusText absent
    assert store.replace_source(graph, "ledger", meta, rows, synced_at_ms=1234) == 2
    src = graph.query("MATCH (s:Source:Ledger {path:$p}) RETURN properties(s)", {"p": meta["path"]}).result_set[0][0]
    assert src["syncedAt"] == 1234 and src["aliases"] == [] and "docStatus" not in src
    assert src["warnings"] == [] and src["rowCount"] == 2
    u1 = graph.query("MATCH (:Source {path:$p})-[:HAS_UNIT]->(u:Unit {unitKey:'U1'}) RETURN properties(u)",
                     {"p": meta["path"]}).result_set[0][0]
    assert u1["needs"] == [] and "statusText" not in u1 and u1["itemId"] == "T-1"
    u2 = graph.query("MATCH (:Source {path:$p})-[:HAS_UNIT]->(u:Unit {unitKey:'U2'}) RETURN keys(u)",
                     {"p": meta["path"]}).result_set[0][0]
    assert "needs" not in u2  # absent stays absent, distinct from []


def test_free_text_round_trips_through_the_parameter_preamble(conn):
    _, graph = conn
    text = "it's \"q\" \\ back\nline `tick` $p {x:1} \U0001F600 — tab\t,;"
    meta, rows = ledger(units=("U1",))
    rows[0]["deliverable"] = text
    meta["warnings"] = [text, "second"]
    store.replace_source(graph, "ledger", meta, rows)
    got = graph.query("MATCH (s:Source)-[:HAS_UNIT]->(u:Unit) RETURN u.deliverable, s.warnings").result_set[0]
    assert got == [text, [text, "second"]]


def test_a_changed_file_replace_leaves_no_trace_of_the_old_version(conn):
    _, graph = conn
    meta, rows = ledger(units=("U1", "U2", "U3"), sha="v1")
    store.replace_source(graph, "ledger", meta, rows)
    meta2, rows2 = ledger(units=("U2", "U4"), sha="v2")
    assert store.replace_source(graph, "ledger", meta2, rows2) == 2
    assert count(graph, "Source") == 1 and count(graph, "Unit") == 2
    keys = {r[0] for r in graph.query("MATCH (u:Unit) RETURN u.unitKey").result_set}
    assert keys == {"U2", "U4"}
    assert graph.query("MATCH (s:Source) RETURN s.sha256").result_set == [["v2"]]
    orphans = graph.query("MATCH (u:Unit) OPTIONAL MATCH (s:Source)-[:HAS_UNIT]->(u) WITH u, s WHERE s IS NULL "
                          "RETURN count(u)").result_set[0][0]
    assert orphans == 0


def test_replace_does_not_touch_another_source(conn):
    _, graph = conn
    a, ra = ledger("a/docs/plans/x-coordination.md", ("U1",))
    b, rb = ledger("b/docs/plans/y-coordination.md", ("U1", "U2"), sha="other")
    store.replace_source(graph, "ledger", a, ra)
    store.replace_source(graph, "ledger", b, rb)
    a2, ra2 = ledger("a/docs/plans/x-coordination.md", ("U9",), sha="new")
    store.replace_source(graph, "ledger", a2, ra2)
    rows = graph.query("MATCH (s:Source)-[:HAS_UNIT]->(u:Unit) RETURN s.path, count(u) ORDER BY s.path").result_set
    assert rows == [["a/docs/plans/x-coordination.md", 1], ["b/docs/plans/y-coordination.md", 2]]


def test_failed_replace_leaves_the_old_state_untouched(conn):
    _, graph = conn
    meta, rows = ledger(units=("U1", "U2"), sha="old")
    store.replace_source(graph, "ledger", meta, rows, synced_at_ms=111)
    bad_meta, bad_rows = ledger(units=("U1", "U1"), sha="new")  # a duplicate Unit.key: the constraint fires
    with pytest.raises(store.StoreError, match="previous state kept"):
        store.replace_source(graph, "ledger", bad_meta, bad_rows, synced_at_ms=222)
    got = graph.query("MATCH (s:Source)-[:HAS_UNIT]->(u:Unit) RETURN s.sha256, s.syncedAt, u.unitKey "
                      "ORDER BY u.unitKey").result_set
    assert got == [["old", 111, "U1"], ["old", 111, "U2"]]
    assert count(graph, "Source") == 1


def test_failed_replace_of_a_new_path_leaves_no_half_written_source(conn):
    _, graph = conn
    meta, rows = ledger(units=("U1", "U1"))
    with pytest.raises(store.StoreError):
        store.replace_source(graph, "ledger", meta, rows)
    assert count(graph, "Source") == 0 and count(graph, "Unit") == 0


def test_unit_key_clash_across_ledgers_fails_that_file_only(conn):
    _, graph = conn
    a, ra = ledger("a/docs/plans/x-coordination.md", ("U1",))
    store.replace_source(graph, "ledger", a, ra)
    b, rb = ledger("b/docs/plans/y-coordination.md", ("U1",))
    rb[0]["key"] = ra[0]["key"]
    with pytest.raises(store.StoreError):
        store.replace_source(graph, "ledger", b, rb)
    assert graph.query("MATCH (s:Source) RETURN s.path").result_set == [["a/docs/plans/x-coordination.md"]]


def test_scheme_ref_in_two_backlogs_names_both_paths_and_keeps_the_old_state(conn):
    _, graph = conn
    a, ra = backlog("a/docs/BACKLOG.md", ["GAL-F0042"])
    store.replace_source(graph, "backlog", a, ra)
    b, rb = backlog("b/docs/BACKLOG.md", ["GAL-F0001", "GAL-F0042"])
    with pytest.raises(store.StoreError) as exc:
        store.replace_source(graph, "backlog", b, rb)
    msg = str(exc.value)
    assert ("item GAL-F0042 headed in both b/docs/BACKLOG.md (ord 2) and a/docs/BACKLOG.md" in msg
            and "wi move may be in progress" in msg)
    assert graph.query("MATCH (i:Item) RETURN i.ref").result_set == [["GAL-F0042"]]
    # the wi move completes: the holder drops the item, the retry succeeds
    a2, ra2 = backlog("a/docs/BACKLOG.md", [], sha="b2")
    store.replace_source(graph, "backlog", a2, ra2)
    assert store.replace_source(graph, "backlog", b, rb) == 2


def test_zero_rows_still_creates_the_source(conn):
    _, graph = conn
    meta, _ = ledger(units=())
    assert store.replace_source(graph, "ledger", meta, []) == 0
    assert count(graph, "Source", "WHERE n.sha256 = 's1'") == 1 and count(graph, "Unit") == 0
    old, rows = ledger(units=("U1",), sha="full")
    store.replace_source(graph, "ledger", old, rows)
    empty, _ = ledger(units=(), sha="emptied")
    assert store.replace_source(graph, "ledger", empty, []) == 0
    assert graph.query("MATCH (s:Source) RETURN s.sha256").result_set == [["emptied"]]
    assert count(graph, "Unit") == 0


@pytest.mark.parametrize("kind,label", [("review", "Review"), ("requirement", "Requirement")])
def test_childless_kinds_carry_everything_on_the_source(conn, kind, label):
    _, graph = conn
    path = f"a/docs/{kind}s/x.md"
    meta = {"path": path, "kind": kind, "component": "a", "slug": "x", "sha256": "r", "bytes": 1,
            "parserVersion": 1, "warnings": [], "rowCount": 0}
    assert store.replace_source(graph, kind, meta, []) == 1
    assert count(graph, f"Source:{label}") == 1
    assert store.replace_source(graph, kind, dict(meta, sha256="r2")) == 1
    assert count(graph, "Source") == 1


def test_delete_source_removes_the_subgraph(conn):
    _, graph = conn
    meta, rows = ledger()
    store.replace_source(graph, "ledger", meta, rows)
    other, orow = ledger("b/docs/plans/y-coordination.md", ("U1",))
    store.replace_source(graph, "ledger", other, orow)
    assert store.delete_source(graph, meta["path"]) == 3
    assert graph.query("MATCH (s:Source) RETURN s.path").result_set == [["b/docs/plans/y-coordination.md"]]
    assert count(graph, "Unit") == 1


def test_indexed_values_are_normalised_so_the_index_finds_them(conn):
    _, graph = conn
    meta, rows = ledger(itemId="  GAL-F0042 ", slug=" x ")
    rows[0]["statusToken"] = " paused "
    store.replace_source(graph, "ledger", meta, rows)
    q = ("MATCH (s:Source) WHERE s.itemId IN $c RETURN count(s)", {"c": ["GAL-F0042"]})
    assert graph.query(*q).result_set == [[1]]
    assert graph.query("MATCH (l:Ledger {slug:'x'}) RETURN count(l)").result_set == [[1]]
    assert graph.query("MATCH (u:Unit {statusToken:'paused'}) RETURN count(u)").result_set == [[1]]


def test_a_crashing_key_is_refused_before_the_instance_sees_it(conn):
    _, graph = conn
    meta, rows = ledger(units=("U1",))
    rows[0]["key"] += " "
    with pytest.raises(KeyHygieneError):
        store.replace_source(graph, "ledger", meta, rows)
    assert count(graph, "Source") == 0


def test_a_large_parameter_commits(conn):
    _, graph = conn
    meta, rows = ledger(units=tuple(f"U{i}" for i in range(2000)))
    for r in rows:
        r["deliverable"] = "d" * 2048
    t0 = time.monotonic()
    assert store.replace_source(graph, "ledger", meta, rows) == 2000
    assert time.monotonic() - t0 < 10
    assert count(graph, "Unit") == 2000


def test_write_converged_rewrites_a_file_that_moved(conn, tmp_path):
    _, graph = conn
    f = tmp_path / "doc.md"
    f.write_text("v1")
    import hashlib

    def sha():
        return hashlib.sha256(f.read_bytes()).hexdigest()

    state = {"n": 0}

    def parse_current():
        state["n"] += 1
        meta = {"path": "a/docs/reviews/x.md", "kind": "review", "component": "a", "slug": "x", "sha256": sha(),
                "bytes": 2, "parserVersion": 1, "warnings": [], "rowCount": 0}
        if state["n"] == 1:
            f.write_text("v2")  # edited while we were writing
        return "review", meta, []

    assert store.write_converged(graph, parse_current, sha) == 2
    assert graph.query("MATCH (s:Source) RETURN s.sha256").result_set == [[sha()]]


# --- plans: the index claims the note makes ------------------------------------------------------

def plan(graph, q, params=None) -> str:
    return str(graph.explain(q, params))


def test_the_replace_anchors_and_in_list_joins_seek_indexes(conn):
    _, graph = conn
    meta, rows = ledger(itemId="GAL-F0002")
    rows[0]["itemId"] = "T-1"
    store.replace_source(graph, "ledger", meta, rows)
    seeks = [
        ("OPTIONAL MATCH (old:Source {path: $path}) RETURN old", {"path": "p"}),
        ("OPTIONAL MATCH (old:Source {path: $meta.path}) RETURN old", {"meta": {"path": "p"}}),
        ("MATCH (s:Source) WHERE s.itemId IN $c RETURN s", {"c": ["a", "b"]}),
        ("MATCH (u:Unit) WHERE u.itemId IN $c RETURN u", {"c": ["a", "b"]}),
        ("MATCH (i:Item) WHERE i.ref IN $c RETURN i", {"c": ["a", "b"]}),
        ("MATCH (l:Ledger {slug: $s}) RETURN l", {"s": "x"}),
        ("MATCH (b:Backlog {scope: $s}) RETURN b", {"s": "a"}),
        ("MATCH (u:Unit {statusToken: 'paused'}) RETURN u", None),
        ("MATCH (u:Unit {key: $k}) RETURN u", {"k": "k"}),
    ]
    for q, params in seeks:
        assert "Node By Index Scan" in plan(graph, q, params), q
    # derived, deliberately not indexed: the lookup anchors on the ledger first (note 7.3)
    assert "Node By Index Scan" not in plan(graph, "MATCH (u:Unit) WHERE u.unitLabel = $u RETURN u", {"u": "U1"})


def test_in_list_actually_returns_the_matches_through_the_index(conn):
    _, graph = conn
    meta, rows = ledger(units=("U1", "U2"), itemId="GAL-F0002")
    rows[1]["itemId"] = "T-9"
    store.replace_source(graph, "ledger", meta, rows)
    q = "MATCH (u:Unit) WHERE u.itemId IN $c RETURN u.unitKey"
    assert graph.query(q, {"c": ["T-9", "nope"]}).result_set == [["U2"]]
    assert graph.query("MATCH (s:Source) WHERE s.itemId IN $c RETURN s.path", {"c": ["x", "GAL-F0002"]}).result_set \
        == [[meta["path"]]]
    prof = str(graph.profile("MATCH (u:Unit) WHERE u.itemId IN $c RETURN u.unitKey", {"c": ["T-9"]}))
    assert "Node By Index Scan" in prof


# --- archived ledgers stay out of the in-flight reads (AC-7) --------------------------------------

IN_FLIGHT_A = """MATCH (l:Ledger)-[:HAS_UNIT]->(u:Unit)
WHERE coalesce(l.docStatus, '') <> 'archived'
RETURN l.path AS path, u.unitKey AS unit"""
IN_FLIGHT_B = """MATCH (u:Unit {statusToken: 'paused'})
MATCH (l:Ledger {path: u.ledgerPath})
WHERE coalesce(l.docStatus, '') <> 'archived'
RETURN l.path AS path, u.unitKey AS unit"""


def test_a_paused_unit_in_an_archived_ledger_is_in_the_graph_but_absent_from_both_views(conn):
    _, graph = conn
    live, lrows = ledger("a/docs/plans/live-coordination.md", ("L1", "L2"), docStatus="active")
    lrows[1]["statusToken"] = "paused"
    arch, arows = ledger("a/docs/plans/old-coordination.md", ("A1", "A2"), docStatus="archived")
    arows[0]["statusToken"] = "paused"
    for r in arows:  # minimal units: no cell text, no needs, no rowHash
        for k in ("unit", "cites", "rowHash"):
            r.pop(k)
    nohdr, nrows = ledger("a/docs/plans/nohdr-coordination.md", ("N1",))  # no Status header: docStatus absent
    for meta, rows in ((live, lrows), (arch, arows), (nohdr, nrows)):
        store.replace_source(graph, "ledger", meta, rows)
    assert count(graph, "Unit", "WHERE n.ledgerPath = 'a/docs/plans/old-coordination.md'") == 2
    a = {tuple(r) for r in graph.query(IN_FLIGHT_A).result_set}
    b = {tuple(r) for r in graph.query(IN_FLIGHT_B).result_set}
    assert a == {("a/docs/plans/live-coordination.md", "L1"), ("a/docs/plans/live-coordination.md", "L2"),
                 ("a/docs/plans/nohdr-coordination.md", "N1")}
    assert b == {("a/docs/plans/live-coordination.md", "L2")}


def test_archived_flip_replaces_full_units_with_minimal_ones(conn):
    _, graph = conn
    meta, rows = ledger(docStatus="active")
    store.replace_source(graph, "ledger", meta, rows)
    meta2, rows2 = ledger(docStatus="archived", sha="s2")
    for r in rows2:
        for k in ("unit", "cites", "rowHash"):
            r.pop(k)
    store.replace_source(graph, "ledger", meta2, rows2)
    assert graph.query(IN_FLIGHT_A).result_set == []
    keys = graph.query("MATCH (u:Unit) RETURN keys(u) LIMIT 1").result_set[0][0]
    assert "rowHash" not in keys and "unitLabel" in keys
    assert count(graph, "Unit") == 2


# --- the real repo's files, through the real parsers ---------------------------------------------

def _parse(root: Path, f, reviews: set[str], legacy: dict):
    raw = (root / f.path).read_bytes()
    if f.kind == "ledger":
        d = parse_ledger(f.path, raw, reviews)
        return d.source_props(), d.unit_props()
    if f.kind == "backlog":
        d = parse_backlog(f.path, raw, legacy)
        return d.source_props(), d.item_props()
    d = parse_review(f.path, raw) if f.kind == "review" else parse_requirements(f.path, raw)
    return d.source_props(), []


def test_every_real_file_loads_and_reads_back_as_written(conn):
    from coordgraph.scope import load_legacy_paths

    _, graph = conn
    found = discover(REPO)
    reviews = {f.path for f in found if f.kind == "review"}
    legacy = load_legacy_paths(REPO)
    expected_children = 0
    for f in found:
        try:
            meta, rows = _parse(REPO, f, reviews, legacy)
        except Exception as exc:  # a BacklogParseError is the parser's, reported by the dry-run
            pytest.fail(f"{f.path}: {exc}")
        store.replace_source(graph, f.kind, meta, rows)
        expected_children += len(rows)
        sent = store.prepare_meta(f.kind, meta)
        back = graph.query("MATCH (s:Source {path:$p}) RETURN properties(s)", {"p": f.path}).result_set[0][0]
        back.pop("syncedAt")
        assert back == sent, f.path
    assert count(graph, "Source") == len(found)
    assert count(graph, "Unit") + count(graph, "Item") == expected_children
    sample = next(f for f in found if f.kind == "ledger" and "coordination-graph" in f.path)
    _, rows = _parse(REPO, sample, reviews, legacy)
    got = graph.query("MATCH (:Source {path:$p})-[:HAS_UNIT]->(u:Unit) RETURN properties(u) "
                      "ORDER BY u.tableIdx, u.rowIdx", {"p": sample.path}).result_set
    assert [g[0] for g in got] == store.prepare_rows("ledger", rows)
