"""P1-36 custodian-side export (``docs/plans/connected-code-graph-p1-noise-export-graph.md`` §1-§3, §5).

Unit tier: a fake runner (the ``GRAPH.RO_QUERY`` seam) serves the §4.1 records; no store. Live tier (``-m live``): a throwaway
unauthenticated FalkorDB of this test's own (127.0.0.1, port >= 16420, ``--pull never``, labels ``ccg-falkordb=1``
``ccg-falkordb.throwaway=1``) holds the §4.1 graph and the real export is compared with the canonical file. It never touches 6379,
6390, 16390, 16400 or any other running service."""
import io
import json
import re
import socket
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval" / "tools"))
import baseline as B  # noqa: E402
import baseline_helpers as H  # noqa: E402
import snapshot_export as E  # noqa: E402

LINK_ROWS = [
    ["h.ts::H", "h.ts", "GET /shop/api/x", "GET /{?}/shop/api/x", "confirmed", "L1", True],
    ["q.ts::Q", "q.ts", "GET /shop/api/y", "GET /shop/api/y", "possible", "L2", True],
    ["q.ts::Q", "q.ts", "GET /shop/api/x", None, "possible", "L3", False],
]
NC5_COUNT = [[1, 0.0]]
NC5_LIST = [["L3", "possible", "http", "GET /shop/api/x", 2]]


class FakeStore:
    """The runner seam: ``(cypher, params) -> rows``. Serves the fixture records by query text and records every call."""

    def __init__(self, snapshot_ids=("snap-1", "snap-1"), link_rows=None, nc5_count=None, mutate=None):
        self.calls = []
        self.ids = list(snapshot_ids)
        recs = H.records()
        if mutate:
            recs = mutate(recs) or recs
        self.table = {}
        for t, (cypher, fields) in E.QUERIES.items():
            self.table[cypher] = [[r.get(f) for f in fields] for tt, r in recs if tt == t]
        self.table[E.LINKS_QUERY] = LINK_ROWS if link_rows is None else link_rows
        self.table[E.NC5_COUNT_QUERY] = NC5_COUNT if nc5_count is None else nc5_count
        self.table[E.NC5_LIST_QUERY] = NC5_LIST

    def __call__(self, cypher, params=None):
        self.calls.append((cypher, dict(params or {})))
        if cypher == E.SNAPSHOT_QUERY:
            return [[self.ids.pop(0)]]
        rows = self.table[cypher]
        if params and "skip" in params:
            return rows[params["skip"]:params["skip"] + params["limit"]]
        return rows


# ------------------------------------------------------------------------------------------------ query texts
def test_every_list_query_is_paged_with_a_total_order_and_only_reads():
    for t, (cypher, fields) in E.QUERIES.items():
        assert not re.search(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP)\b", cypher), t
        if t in ("meta", "unit"):
            continue
        assert cypher.rstrip().endswith("SKIP $skip LIMIT $limit"), t
        assert "ORDER BY" in cypher, t
    for q in (E.LINKS_QUERY, E.NC5_LIST_QUERY):
        assert q.rstrip().endswith("SKIP $skip LIMIT $limit") and "ORDER BY" in q
    assert "WHERE id >" not in "".join(c for c, _ in E.QUERIES.values())          # no keyset paging on an indexed property
    assert E.PAGE_LIMIT == 5000 and E.QUERY_TIMEOUT_MS <= 10000


def test_query_fields_cover_the_section_3_record_fields():
    want = {"meta": ["snapshotId", "systemId", "graphModelVersion"], "unit": ["unitId", "repoId", "language", "buildStatus", "flowAvailable"],
            "symbol": ["symbolId", "unitId", "repoId", "qname", "path", "lineStart", "lineEnd", "ownedRegions"],
            "site": ["siteId", "owner", "unitId", "path", "line", "calleeName"], "call": ["site", "callee", "status", "reason"],
            "name": ["site", "candidate"], "cp": ["cpId", "kind", "keyNorm"], "provides": ["symbolId", "cpId", "status", "reason"],
            "consumes": ["site", "cpId", "status", "reason"], "link": ["linkId", "kind", "status", "reason", "cpId", "provider"],
            "via": ["site", "linkId", "status", "reason", "consumerKey"], "entry": ["entryId", "unitId", "path", "keyPath", "launchKind"],
            "bound": ["linkId", "entryId", "siteId", "ord"]}
    assert {t: list(f) for t, (_, f) in E.QUERIES.items()} == want
    assert list(E.QUERIES) == list(want)                                   # export group order


# ------------------------------------------------------------------------------------------------ read-only transport
class StubGraph:
    def __init__(self):
        self.seen = []

    def query(self, *a, **k):
        raise AssertionError("a write-capable GRAPH.QUERY was used")

    def ro_query(self, q, params=None, timeout=None):
        self.seen.append((q, params, timeout))
        return type("R", (), {"result_set": [["x"]]})()


def test_the_real_runner_uses_ro_query_with_a_timeout_and_params():
    g = StubGraph()
    rows = E.RoRunner(g)("RETURN 1", {"skip": 0})
    assert rows == [["x"]] and g.seen == [("RETURN 1", {"skip": 0}, E.QUERY_TIMEOUT_MS)]


# ------------------------------------------------------------------------------------------------ paging
def test_paging_stops_on_a_short_page_and_continues_on_a_full_one():
    calls = []

    def run(cypher, params=None):
        calls.append(params["skip"])
        return [[i] for i in range(params["skip"], min(params["skip"] + params["limit"], 12))]
    assert [r[0] for r in E.paged(run, "Q SKIP $skip LIMIT $limit", limit=5)] == list(range(12))
    assert calls == [0, 5, 10]
    calls.clear()
    assert len(E.paged(run, "Q", limit=6)) == 12 and calls == [0, 6, 12]        # exact multiple: one more (empty) read


def test_default_page_limit_is_below_resultset_size():
    seen = []
    E.paged(lambda c, p: seen.append(p) or [], "Q")
    assert seen == [{"skip": 0, "limit": 5000}]


# ------------------------------------------------------------------------------------------------ the export
def test_export_lines_are_the_canonical_file_in_group_order():
    assert E.export_snapshot(FakeStore()) == H.canonical_lines()


def _null_optionals(recs):
    for t, r in recs:
        if t == "symbol":
            r.setdefault("ownedRegions", None)        # the store returns null for a missing optional property
        if t == "via":
            r.setdefault("consumerKey", None)
        if t == "entry":
            r.setdefault("launchKind", None)


def test_optional_fields_are_absent_when_null_and_required_nulls_are_kept():
    recs = [json.loads(l) for l in E.export_snapshot(FakeStore(mutate=_null_optionals))]
    sym = next(r for r in recs if r["t"] == "symbol" and r["symbolId"] == "A")
    assert "ownedRegions" not in sym
    link3 = next(r for r in recs if r["t"] == "link" and r["linkId"] == "L3")
    assert link3["provider"] is None and "provider" in link3
    via3 = next(r for r in recs if r["t"] == "via" and r["linkId"] == "L3")
    assert "consumerKey" not in via3
    entry = next(r for r in recs if r["t"] == "entry")
    assert "launchKind" not in entry and entry["keyPath"] == "a.b"


# ------------------------------------------------------------------------------------------------ link dump (§2)
def test_link_dump_splits_unprovided_and_keeps_the_note_example_row():
    prov, unprov = E.dump_links(FakeStore())
    assert prov[0] == {"consumerKey": "GET /{?}/shop/api/x", "element": "h.ts::H", "keyNorm": "GET /shop/api/x", "linkId": "L1",
                       "path": "h.ts", "status": "confirmed"}
    assert [r["linkId"] for r in prov] == ["L1", "L2"] and [r["linkId"] for r in unprov] == ["L3"]
    assert "hasProvider" not in prov[0]


def test_link_dump_moves_keyless_provider_rows_to_unprovided_and_distincts_to_best_status_then_lowest_id():
    rows = [["f::a", "f", "GET /k", None, "confirmed", "L2", True], ["f::a", "f", "GET /k", None, "possible", "L9", True],
            ["f::a", "f", "GET /k", None, "confirmed", "L5", True], ["f::b", "f", None, None, "possible", "L4", True],
            ["f::a", "f", "GET /k", None, "possible", "L1", False]]
    prov, unprov = E.dump_links(FakeStore(link_rows=rows))
    assert [(r["element"], r["linkId"]) for r in prov] == [("f::a", "L2")]
    assert sorted(r["linkId"] for r in unprov) == ["L1", "L4"]


def test_link_stats_count_the_shop_scope_and_catch_all():
    prov = [{"keyNorm": "GET /files/{**}"}, {"keyNorm": "GET /shop/api/x"}]
    unprov = [{"keyNorm": "GET /shop/api/x"}, {"keyNorm": "GET /{?}/shop/api/y"}, {"keyNorm": "GET /other"}, {"keyNorm": None}]
    assert E.link_stats(prov, unprov) == {"links": 2, "linksUnprovided": 4, "unprovidedShopApi": 2, "keyNormCatchAll": 1}


# ------------------------------------------------------------------------------------------------ Q-NC5 (§5)
def test_nc5_dump_converts_float_sum_and_lists_the_rows():
    d = E.nc5_dump(FakeStore())
    assert d["unboundMultiUnit"] == 1 and d["unboundMultiUnitConfirmed"] == 0 and isinstance(d["unboundMultiUnitConfirmed"], int)
    assert d["rows"] == [{"keyNorm": "GET /shop/api/x", "kind": "http", "linkId": "L3", "providerUnits": 2, "status": "possible"}]


def test_nc5_dump_of_an_empty_result_is_zero_zero():
    assert E.nc5_dump(FakeStore(nc5_count=[[0, None]]))["unboundMultiUnitConfirmed"] == 0


# ------------------------------------------------------------------------------------------------ run_export: guard, files
def test_snapshot_id_is_read_first_and_last_and_the_files_are_written(tmp_path):
    st = FakeStore()
    counts = E.run_export(st, "snap-1", tmp_path)
    assert st.calls[0][0] == E.SNAPSHOT_QUERY and st.calls[-1][0] == E.SNAPSHOT_QUERY
    assert sum(1 for c, _ in st.calls if c == E.SNAPSHOT_QUERY) == 2
    assert (tmp_path / "snapshot-adjacency.jsonl").read_text().splitlines() == H.canonical_lines()
    assert [json.loads(l)["linkId"] for l in (tmp_path / "links.jsonl").read_text().splitlines()] == ["L1", "L2"]
    assert [json.loads(l)["linkId"] for l in (tmp_path / "links-unprovided.jsonl").read_text().splitlines()] == ["L3"]
    assert json.loads((tmp_path / "nc5-unbound.jsonl").read_text())["linkId"] == "L3"
    assert counts["records"]["symbol"] == 15 and counts["unboundMultiUnit"] == 1 and counts["dropped"] == {"call": 0, "name": 0, "consumes": 0, "via": 0}
    B.read_snapshot(tmp_path / "snapshot-adjacency.jsonl")


def test_a_wrong_first_snapshot_id_is_refused_before_any_other_read(tmp_path):
    st = FakeStore(snapshot_ids=("other", "other"))
    with pytest.raises(E.SnapshotMismatch):
        E.run_export(st, "snap-1", tmp_path)
    assert len(st.calls) == 1 and list(tmp_path.iterdir()) == []


def test_a_refresh_during_the_export_is_an_error_and_writes_nothing(tmp_path):
    st = FakeStore(snapshot_ids=("snap-1", "snap-2"))
    with pytest.raises(E.SnapshotMismatch):
        E.run_export(st, "snap-1", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_two_snapshot_nodes_are_refused(tmp_path):
    class Two(FakeStore):
        def __call__(self, cypher, params=None):
            return [["snap-1"], ["snap-1"]] if cypher == E.SNAPSHOT_QUERY else super().__call__(cypher, params)
    with pytest.raises(E.SnapshotMismatch):
        E.run_export(Two(), "snap-1", tmp_path)


def test_the_independent_nc5_count_must_equal_the_export_projection(tmp_path):
    with pytest.raises(ValueError, match="nc5"):
        E.run_export(FakeStore(nc5_count=[[2, 0.0]]), "snap-1", tmp_path)
    with pytest.raises(ValueError, match="nc5"):
        E.run_export(FakeStore(nc5_count=[[1, 1.0]]), "snap-1", tmp_path)


def test_a_closure_break_in_the_export_is_an_error(tmp_path):
    def bad(recs):
        for t, r in recs:
            if t == "provides":
                r["cpId"] = "ghost"
                break
    with pytest.raises(ValueError, match="provides"):
        E.run_export(FakeStore(mutate=bad), "snap-1", tmp_path)


def test_ownerless_site_rows_are_counted_in_dropped(tmp_path):
    counts = E.run_export(FakeStore(mutate=lambda recs: [(t, r) for t, r in recs if not (t == "site" and r["siteId"] == "sBc")]),
                          "snap-1", tmp_path)
    assert counts["dropped"]["call"] == 1 and counts["dropped"]["name"] == 1


# ------------------------------------------------------------------------------------------------ main
def test_main_refuses_a_custodian_inside_the_repo():
    assert E.main(["--graph", "ccg_x", "--snapshot-id", "snap-1", "--custodian", str(E.REPO / "cpg" / "x")], runner=FakeStore()) == 2


def test_main_prints_counts_only_and_no_ids_or_paths(tmp_path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = E.main(["--graph", "ccg_x", "--snapshot-id", "snap-1", "--custodian", str(tmp_path)], runner=FakeStore())
    out = buf.getvalue()
    assert rc == 0
    rep = json.loads(out)

    def leaves(o):
        return [x for v in o.values() for x in (leaves(v) if isinstance(v, dict) else [v])]
    assert all(isinstance(v, int) for v in leaves(rep))
    for needle in ("h.ts", "q.ts", "GET /", "snap-1", str(tmp_path)):
        assert needle not in out


def test_main_exit_4_on_a_snapshot_mismatch(tmp_path):
    assert E.main(["--graph", "ccg_x", "--snapshot-id", "nope", "--custodian", str(tmp_path)], runner=FakeStore()) == 4


# ------------------------------------------------------------------------------------------------ live tier
IMAGE = "falkordb/falkordb:v4.18.11"
FORBIDDEN_PORTS = {6379, 6390, 16390, 16400}


def _free_port():
    published = subprocess.run(["docker", "ps", "--format", "{{.Ports}}"], capture_output=True, text=True, check=True).stdout
    taken = {int(p) for p in re.findall(r":(\d+)->", published)} | FORBIDDEN_PORTS
    for port in range(16420, 16520):
        if port in taken:
            continue
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port in 16420-16519")


@pytest.fixture(scope="module")
def throwaway():
    try:
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"{IMAGE} is not available locally (docker missing or image not pulled; this test never pulls)")
    port = _free_port()
    name = f"ccg-falkordb-tw-{port}"
    subprocess.run(["docker", "run", "-d", "--pull", "never", "--name", name, "-p", f"127.0.0.1:{port}:6379",
                    "--label", "ccg-falkordb=1", "--label", "ccg-falkordb.throwaway=1", IMAGE],
                   capture_output=True, text=True, check=True)
    try:
        from falkordb import FalkorDB
        db = None
        for _ in range(60):
            try:
                db = FalkorDB(host="127.0.0.1", port=port)
                db.connection.ping()
                break
            except Exception:  # noqa: BLE001 -- not up yet
                time.sleep(0.5)
        else:
            pytest.fail("throwaway FalkorDB did not come up")
        yield db
    finally:
        labels = subprocess.run(["docker", "inspect", "-f", "{{index .Config.Labels \"ccg-falkordb.throwaway\"}}", name],
                                capture_output=True, text=True).stdout.strip()
        if labels == "1":                      # only ever remove the container this fixture started
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def load_graph(db, key):
    """The §4.1 records as the model's nodes and relationships (labels and edge names of ``-p1-graph.md``)."""
    g = db.select_graph(key)
    by = {}
    for t, r in H.records():
        by.setdefault(t, []).append(r)
    m = by["meta"][0]
    g.query("CREATE (:Snapshot {snapshotId: $a, systemId: $b, graphModelVersion: $c})",
            {"a": m["snapshotId"], "b": m["systemId"], "c": m["graphModelVersion"]})
    for q, t in [
        ("UNWIND $rows AS r CREATE (:Unit {unitId: r.unitId, repoId: r.repoId, language: r.language, buildStatus: r.buildStatus, flowAvailable: r.flowAvailable})", "unit"),
        ("UNWIND $rows AS r CREATE (:Symbol {symbolId: r.symbolId, unitId: r.unitId, repoId: r.repoId, qname: r.qname, path: r.path, lineStart: r.lineStart, lineEnd: r.lineEnd, ownedRegions: r.ownedRegions})", "symbol"),
        ("UNWIND $rows AS r MATCH (s:Symbol {symbolId: r.owner}) CREATE (s)-[:HAS_CALL]->(:CallSite {callSiteId: r.siteId, unitId: r.unitId, path: r.path, line: r.line, calleeName: r.calleeName})", "site"),
        ("UNWIND $rows AS r MATCH (c:CallSite {callSiteId: r.site}), (t:Symbol {symbolId: r.callee}) CREATE (c)-[:CALLS {status: r.status, reason: r.reason}]->(t)", "call"),
        ("UNWIND $rows AS r MATCH (c:CallSite {callSiteId: r.site}), (t:Symbol {symbolId: r.candidate}) CREATE (c)-[:CALLS_NAME]->(:Name {name: r.site + r.candidate})-[:RESOLVES_TO]->(t)", "name"),
        ("UNWIND $rows AS r CREATE (:ContractPoint {cpId: r.cpId, kind: r.kind, keyNorm: r.keyNorm})", "cp"),
        ("UNWIND $rows AS r MATCH (s:Symbol {symbolId: r.symbolId}), (cp:ContractPoint {cpId: r.cpId}) CREATE (s)-[:PROVIDES {status: r.status, reason: r.reason}]->(cp)", "provides"),
        ("UNWIND $rows AS r MATCH (c:CallSite {callSiteId: r.site}), (cp:ContractPoint {cpId: r.cpId}) CREATE (c)-[:CONSUMES {status: r.status, reason: r.reason}]->(cp)", "consumes"),
        ("UNWIND $rows AS r CREATE (:Link {linkId: r.linkId, kind: r.kind, status: r.status, reason: r.reason})", "link"),
        ("UNWIND $rows AS r MATCH (l:Link {linkId: r.linkId}), (cp:ContractPoint {cpId: r.cpId}) CREATE (l)-[:ON]->(cp)", "link"),
        ("UNWIND $rows AS r MATCH (l:Link {linkId: r.linkId}), (p:Symbol {symbolId: r.provider}) CREATE (l)-[:TO]->(p)", "link"),
        ("UNWIND $rows AS r MATCH (c:CallSite {callSiteId: r.site}), (l:Link {linkId: r.linkId}) CREATE (c)-[:VIA {status: r.status, reason: r.reason, consumerKey: r.consumerKey}]->(l)", "via"),
        ("UNWIND $rows AS r CREATE (:ConfigEntry {entryId: r.entryId, unitId: r.unitId, path: r.path, keyPath: r.keyPath})", "entry"),
        ("UNWIND $rows AS r MATCH (l:Link {linkId: r.linkId}), (e:ConfigEntry {entryId: r.entryId}) CREATE (l)-[:BOUND_BY {siteId: r.siteId, ord: r.ord}]->(e)", "bound"),
    ]:
        rows = by[t]
        if "(l)-[:ON]" in q:
            rows = [r for r in rows if r["cpId"] is not None]
        if "(l)-[:TO]" in q:
            rows = [r for r in rows if r["provider"] is not None]
        g.query(q, {"rows": rows})
    return g


@pytest.mark.live
def test_live_export_of_the_worked_graph_is_the_canonical_file(throwaway, tmp_path):
    load_graph(throwaway, "ccg_tw_a")
    run = E.RoRunner(throwaway.select_graph("ccg_tw_a"))
    counts = E.run_export(run, H.SNAPSHOT_ID, tmp_path)
    assert (tmp_path / "snapshot-adjacency.jsonl").read_text().splitlines() == H.canonical_lines()
    assert counts["unboundMultiUnit"] == 1 and counts["unboundMultiUnitConfirmed"] == 0
    prov = [json.loads(l) for l in (tmp_path / "links.jsonl").read_text().splitlines()]
    unprov = [json.loads(l) for l in (tmp_path / "links-unprovided.jsonl").read_text().splitlines()]
    assert prov[0] == {"consumerKey": "GET /{?}/shop/api/x", "element": "h.ts::H", "keyNorm": "GET /shop/api/x", "linkId": "L1",
                       "path": "h.ts", "status": "confirmed"}
    assert [r["linkId"] for r in prov] == ["L1", "L2"] and [(r["linkId"], r["keyNorm"]) for r in unprov] == [("L3", "GET /shop/api/x")]
    assert counts["unprovidedShopApi"] == 1
    nc5 = json.loads((tmp_path / "nc5-unbound.jsonl").read_text())
    assert (nc5["linkId"], nc5["status"], nc5["kind"], nc5["keyNorm"], nc5["providerUnits"]) == ("L3", "possible", "http", "GET /shop/api/x", 2)


@pytest.mark.live
def test_live_export_feeds_the_over_lister_with_the_worked_numbers(throwaway, tmp_path):
    load_graph(throwaway, "ccg_tw_b")
    E.run_export(E.RoRunner(throwaway.select_graph("ccg_tw_b")), H.SNAPSHOT_ID, tmp_path)
    snap = B.read_snapshot(tmp_path / "snapshot-adjacency.jsonl")
    r = B.naive_over_lister(snap, [H.anchor("symbol", "A"), H.anchor("contract", "CP1"), H.anchor("config", "CE1")], 2)
    assert {x["anchorId"]: (x["count"], x["countConfirmedOwn"]) for x in r} == {
        "symbol:A": (9, 5), "contract:CP1": (11, 7), "config:CE1": (5, 4)}


@pytest.mark.live
def test_live_missing_edge_status_reads_as_confirmed_resolved_call(throwaway, tmp_path):
    g = load_graph(throwaway, "ccg_tw_c")
    g.query("MATCH (:CallSite {callSiteId: 'sBc'})-[e:CALLS]->(:Symbol {symbolId: 'A'}) SET e.status = NULL, e.reason = NULL")
    run = E.RoRunner(g)
    row = next(json.loads(l) for l in E.export_snapshot(run) if json.loads(l)["t"] == "call" and json.loads(l)["site"] == "sBc")
    assert (row["status"], row["reason"]) == ("confirmed", "resolved-call")


@pytest.mark.live
def test_live_non_http_links_stay_out_of_the_oracle_dump(throwaway):
    g = load_graph(throwaway, "ccg_tw_f")
    g.query("MATCH (c:CallSite {callSiteId: 'sHk'}) CREATE (c)-[:VIA {status: 'confirmed', reason: 'r'}]->(:Link {linkId: 'L4', kind: 'event', status: 'confirmed', reason: 'r'})")
    prov, unprov = E.dump_links(E.RoRunner(g))
    assert [r["linkId"] for r in prov] == ["L1", "L2"] and [r["linkId"] for r in unprov] == ["L3"]


@pytest.mark.live
def test_live_a_snapshot_id_mismatch_is_refused(throwaway, tmp_path):
    load_graph(throwaway, "ccg_tw_d")
    with pytest.raises(E.SnapshotMismatch):
        E.run_export(E.RoRunner(throwaway.select_graph("ccg_tw_d")), "not-the-id", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.live
def test_live_the_runner_cannot_write(throwaway):
    load_graph(throwaway, "ccg_tw_e")
    with pytest.raises(Exception):  # noqa: B017 -- the server's read-only refusal
        E.RoRunner(throwaway.select_graph("ccg_tw_e"))("CREATE (:X)")
