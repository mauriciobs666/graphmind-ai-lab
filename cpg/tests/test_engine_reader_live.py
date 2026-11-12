"""P1-21 reader layer and anchors, LIVE on a throwaway instance only (``infra/ccg-falkordb.sh --throwaway up``; ``-m live``).

Needs ``CCG_FALKORDB_PORT`` (the throwaway's port; 6379 and 6390 are refused, and the port must be published by a container labelled
``ccg-falkordb.throwaway=1``) and ``CCG_CONFIG_DIR`` (a scratch dir holding the throwaway's ``falkordb-admin.pass``). ``CCG_REQUIRE_LIVE=1`` turns a missing
instance into a failure. The module creates one system's four ACL users (``ccg.store.acl.create_users``, credentials in a tmp dir), loads ``engine_helpers.mini_rows``
through ``ccg.store.loader.load`` as ``ccg_build_<s>``, renames the build key to the live key as admin, and reads it as ``ccg_engine_<s>`` only.
Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §5, §7, §8 P1-21 (b)-(f); ``docs/plans/connected-code-graph-p1-answers.md`` P1-21."""
from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from engine_helpers import EVIL_QNAME, mini_rows
from loader_helpers import admin_password, skip_or_fail, throwaway_port
from test_engine_reader import EXPECTED

from ccg import sysname
from ccg.engine import anchors as A
from ccg.engine import config, reader
from ccg.engine.reader import CypherReader, ReaderError, ReadTimeout, ResultTooLarge
from ccg.store import acl
from ccg.store import loader as L

HOST = "127.0.0.1"
SYSTEM = f"p121x{os.getpid() % 1000}"
OTHER = f"p121y{os.getpid() % 1000}"
LIVE = sysname.live_key(SYSTEM)
pytestmark = pytest.mark.live


class _Redacted(dict):
    def __repr__(self):
        return "{<redacted>}"


class _World(SimpleNamespace):
    def __repr__(self):
        return "<world: throwaway instance, redacted>"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance (or the docker CLI cannot start): start one with cpg/infra/ccg-falkordb.sh --throwaway up and export "
                     "CCG_FALKORDB_PORT and CCG_CONFIG_DIR (a scratch dir)")
    try:
        pw = admin_password()
    except FileNotFoundError as exc:
        skip_or_fail(f"throwaway admin password not readable ({exc})")
    admin = L.connect(HOST, port, username=None, password=pw)
    cred = tmp_path_factory.mktemp("ccg-credentials-p121")
    users, passwords = [], _Redacted()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CCG_CREDENTIALS_DIR", str(cred))
        for system in (SYSTEM, OTHER):
            users += acl.create_users(admin, system)
            passwords[system] = _Redacted(acl.read_credentials(system))
    w = _World(port=port, admin=admin, cred=cred, users=users)
    try:
        for system in (SYSTEM, OTHER):
            rows = mini_rows(system)
            build_db = L.connect(HOST, port, username=sysname.acl_user("build", system), password=passwords[system]["build"])
            report = L.load(L.LiveStore(build_db, system), rows, system)
            admin.connection.rename(report.build_key, sysname.live_key(system))
        w.engine_db = L.connect(HOST, port, username=sysname.acl_user("engine", SYSTEM), password=passwords[SYSTEM]["engine"])
        w.engine_other_db = L.connect(HOST, port, username=sysname.acl_user("engine", OTHER), password=passwords[OTHER]["engine"])
        yield w
    finally:
        for pattern in (f"ccg_{SYSTEM}*", f"ccg_{OTHER}*"):
            for key in list(admin.connection.scan_iter(match=pattern)):
                admin.connection.delete(key)
        for user in users:
            admin.connection.execute_command("ACL", "DELUSER", user)


@pytest.fixture
def rd(world):
    return CypherReader(world.engine_db, SYSTEM)


def _counts(world):
    g = world.admin.select_graph(LIVE)
    return (g.ro_query("MATCH (n) RETURN count(n)").result_set[0][0], g.ro_query("MATCH ()-[e]->() RETURN count(e)").result_set[0][0])


def _params(text):
    out = {}
    for p in sorted(set(re.findall(r"\$([a-z]+)", text))):
        out[p] = {"ids": ["s_f"], "pairs": [["l1", "c_fetch1"]], "repo": "repo-a", "path": "app/a.py", "qname": "app.Outer", "name": "run",
                  "key": "GET /shop/api/health", "kind": "http"}[p]
    return out


# ------------------------------------------------------------------------------------- the texts against the plan and the engine

def test_the_texts_are_the_plans_section_5_read_from_the_document():
    doc = Path(__file__).resolve().parents[2] / "docs" / "plans" / "connected-code-graph-p1-graph.md"
    if not doc.is_file():
        pytest.skip("the plan document is not in this checkout")
    text = doc.read_text()
    sec = text[text.index("## 5. Reader queries"):text.index("## 6. `Link` answer size")]
    norm = lambda s: re.sub(r"\s+", " ", s.replace("`", "")).strip()           # noqa: E731
    body = norm(sec)
    for name, cypher in reader.QUERIES.items():
        shown = cypher
        head, _, label = name.partition(":")
        if head in ("owner_of", "data_in", "data_out", "controlled_out"):
            idprop = "callSiteId" if label == "CallSite" else "pointId"
            shown = shown.replace(f"(p:{label} {{{idprop}: i}})", "(p:<Label> {<idProp>: i})")
            for rel in ("HAS_CALL", "HAS_PARAM", "HAS_RETURN", "HAS_COND"):
                shown = shown.replace(f"[:{rel}]", "[:<HAS_REL>]") if head == "owner_of" else shown
        assert norm(shown) in body, name


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_registered_text_runs_on_the_engine_credential_and_returns_rows_as_tuples(rd, name):
    rows = rd.read(name, _params(EXPECTED[name]))
    assert all(isinstance(r, tuple) for r in rows)


def test_explain_of_every_text_seeks_an_index_never_scans_a_label_except_the_two_by_design(world):
    scans = {}
    for name, text in EXPECTED.items():
        plan = str(world.engine_db.select_graph(LIVE).explain(text, _params(text)))
        scans[name] = ("Node By Index Scan" in plan, "Label Scan" in plan, plan)
    by_design = {"cp_keys", "snapshot"}
    for name, (index, label, plan) in scans.items():
        if name in by_design:
            assert label, name
        else:
            assert index and not label, (name, plan)
    control = str(world.engine_db.select_graph(LIVE).explain("MATCH (s:Symbol) WHERE toLower(s.qname) = $q RETURN s.symbolId", {"q": "x"}))
    assert "Node By Label Scan" in control


def test_explain_without_parameter_values_says_missing_parameters_so_the_harness_must_supply_them(world):
    with pytest.raises(Exception, match="(?i)missing parameters"):
        world.engine_db.select_graph(LIVE).explain(EXPECTED["file_by_path"])


# ------------------------------------------------------------------------------------- read-only, live: role and server

def test_the_engine_role_cannot_write_and_the_server_refuses_a_write_on_the_read_path(world):
    before = _counts(world)
    g = world.engine_db.select_graph(LIVE)
    with pytest.raises(Exception, match="(?i)noperm|permission"):
        g.query("CREATE (:Evil)")
    with pytest.raises(Exception, match="(?i)read-only|ro_query"):
        g.ro_query("CREATE (:Evil)")
    for cmd in (("DEL", LIVE), ("SET", LIVE + "x", "1"), ("RENAME", LIVE, LIVE + "z"), ("GRAPH.DELETE", LIVE), ("EXPIRE", LIVE, 0), ("FLUSHALL",)):
        with pytest.raises(Exception, match="(?i)noperm|permission"):
            world.engine_db.connection.execute_command(*cmd)
    assert _counts(world) == before and world.admin.connection.exists(LIVE)


def test_every_registered_text_leaves_the_graph_exactly_as_it_was(world, rd):
    before = _counts(world)
    for name, text in EXPECTED.items():
        rd.read(name, _params(text))
    assert _counts(world) == before


def test_with_the_lint_removed_a_planted_write_text_is_still_refused_by_the_server_and_the_role(world, monkeypatch):
    before = _counts(world)
    monkeypatch.setitem(reader.QUERIES, "planted", "MATCH (s:Symbol {symbolId: $id}) SET s.qname = $v")
    monkeypatch.setattr(reader, "assert_read_only", lambda text: None)
    with pytest.raises(ReaderError, match="planted"):
        CypherReader(world.engine_db, SYSTEM).read("planted", {"id": "s_f", "v": "pwned"})
    assert _counts(world) == before
    assert world.admin.select_graph(LIVE).ro_query("MATCH (s:Symbol {symbolId: 's_f'}) RETURN s.qname").result_set == [["app.Outer.f"]]


def test_the_reader_cannot_reach_another_systems_graph(world):
    other = world.engine_db.select_graph(sysname.live_key(OTHER))
    with pytest.raises(Exception, match="(?i)noperm|permission"):
        other.ro_query("MATCH (r:Repo) RETURN r.repoId")
    assert CypherReader(world.engine_other_db, OTHER).read("repo_exists", {"repo": "repo-a"}) == [("repo-a",)]


def test_for_system_connects_with_the_stored_engine_credential(world, monkeypatch):
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(world.cred))
    r = CypherReader.for_system(SYSTEM, HOST, world.port)
    assert r.read("repo_exists", {"repo": "repo-b"}) == [("repo-b",)]
    with pytest.raises(Exception, match="(?i)noperm|permission"):
        r._graph.query("CREATE (:Evil)")                                  # noqa: SLF001 - proving which credential the reader holds
    with pytest.raises(ReaderError, match="6379"):
        CypherReader.for_system(SYSTEM, HOST, 6379)


# ------------------------------------------------------------------------------------- timeout and bounded replies, live

def test_the_timeout_reaches_the_server_and_a_slow_read_raises_read_timeout(world, monkeypatch):
    monkeypatch.setitem(reader.QUERIES, "slow_probe", "MATCH (a), (b), (c), (d), (e) RETURN count(*)")
    with pytest.raises(ReadTimeout):
        CypherReader(world.engine_db, SYSTEM, timeout_ms=1).read("slow_probe", {})


def test_a_timeout_above_the_servers_timeout_max_is_a_reader_error_not_a_timeout(world):
    world.admin.connection.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", 3000)
    try:
        with pytest.raises(ReaderError, match="TIMEOUT_MAX") as ei:
            CypherReader(world.engine_db, SYSTEM, timeout_ms=10_000).read("repo_exists", {"repo": "repo-a"})
        assert not isinstance(ei.value, ReadTimeout)
        assert CypherReader(world.engine_db, SYSTEM, timeout_ms=3000).read("repo_exists", {"repo": "repo-a"}) == [("repo-a",)]
    finally:
        world.admin.connection.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", 10000)


def test_the_resultset_size_constant_is_the_servers_and_a_reply_that_fills_it_raises(world, rd):
    got = world.admin.connection.execute_command("GRAPH.CONFIG", "GET", "RESULTSET_SIZE")
    assert int(got[1]) == config.RESULTSET_SIZE
    g = world.admin.select_graph(LIVE)
    g.query("UNWIND range(1, $n) AS i CREATE (:ContractPoint {cpId: 'bulk' + toString(i), kind: 'bulk', keyNorm: 'k' + toString(i)})", {"n": config.RESULTSET_SIZE + 5})
    try:
        with pytest.raises(ResultTooLarge, match="cp_keys"):
            rd.read("cp_keys", {"kind": "bulk"})
    finally:
        g.query("MATCH (c:ContractPoint {kind: 'bulk'}) DELETE c")
    assert len(rd.read("cp_keys", {"kind": "http"})) == 3


# ------------------------------------------------------------------------------------- the reads' contents (C3)

def test_owner_join_returns_path_repo_unit_and_lines_of_the_owner_for_every_point_label(rd):
    for label, pid in (("Param", "p1"), ("Ret", "r1"), ("Cond", "k1")):
        (row,) = rd.read(f"owner_of:{label}", {"ids": [pid]})
        assert row == (pid, "s_f", "repo-a", "ua", "app.Outer.f", "app/a.py", 10, 30, None)
    (row,) = rd.read("owner_of:CallSite", {"ids": ["c_res"]})
    assert row[:6] == ("c_res", "s_caller", "repo-a", "ua", "app.caller", "app/a.py")
    (row,) = rd.read("owner_of:Param", {"ids": ["p1"]})
    assert row[1] == "s_f"


def test_an_edge_reason_and_status_come_back_with_d8_defaults_coalesced_in_the_query(rd):
    (row,) = rd.read("calls_in", {"ids": ["s_run1"]})
    assert row == ("s_run1", "c_res", "repo-a", "ua", "app/a.py", 401, "run", "confirmed", "resolved-call")
    rows = rd.read("data_in:Param", {"ids": ["p1"]})
    assert rows == [("p1", "CallSite", "c_res", 401, "p0", "p0", "possible", "field-truncated"),
                    ("p1", "CallSite", "c_unres", 402, "p0.a", "p0", "confirmed", "intra-flow")]
    (row,) = rd.read("provides_in", {"ids": ["cp_items"]})
    assert row[-2:] == ("possible", "runtime-built")
    (row,) = rd.read("provides_in", {"ids": ["cp_health"]})
    assert row[-2:] == (None, None)                                         # PROVIDES is not coalesced: the plan gives it no default


def test_lifted_of_returns_unit_repo_and_site_per_link_and_site(rd):
    rows = rd.read("lifted_of", {"pairs": [["l1", "c_fetch1"], ["l1", "c_fetch2"]]})
    assert rows[0] == ("l1", "c_fetch1", 0, "CallSite", "c_fetch1", "repo-a", "ua", "app/a.py", 601, "/api/health", "GET", "c_fetch1", 601, "fetch")
    assert rows[1] == ("l1", "c_fetch1", 1, "Symbol", "s_caller2", "repo-a", "ua", "app/a.py", 600, None, None, None, None, "app.caller2")
    assert rows[2][:5] == ("l1", "c_fetch2", 0, "CallSite", "c_fetch2")
    assert [(r[1], r[2]) for r in rows] == [("c_fetch1", 0), ("c_fetch1", 1), ("c_fetch2", 0)]


def test_name_in_on_a_resolved_site_returns_only_the_unresolved_candidate_with_count_one_and_on_an_unresolved_site_every_candidate_with_zero(rd):
    rows = rd.read("name_in", {"ids": ["s_run1", "s_run2"]})
    by = {(r[0], r[1]): r[-1] for r in rows}
    assert by == {("s_run2", "c_res"): 1, ("s_run1", "c_unres"): 0, ("s_run2", "c_unres"): 0}
    out = rd.read("name_out", {"ids": ["c_res", "c_unres"]})
    assert {(r[0], r[1]): r[-1] for r in out} == {("c_res", "s_run2"): 1, ("c_unres", "s_run1"): 0, ("c_unres", "s_run2"): 0}


def test_candidate_count_counts_the_links_of_equal_consumer_key_self_included(rd):
    assert rd.read("candidate_count", {"pairs": [["c_fetch1", "l1"]]}) == [("c_fetch1", "l1", 2)]
    assert rd.read("candidate_count", {"pairs": [["c_fetch2", "l1"]]}) == [("c_fetch2", "l1", 1)]


def test_bound_out_returns_only_the_chain_of_the_asked_site_as_valid_bound_columns(rd):
    a = rd.read("bound_out", {"pairs": [["l1", "c_fetch1"]]})
    assert [(r[1], r[2], r[3]) for r in a] == [("c_fetch1", 0, "eX"), ("c_fetch1", 1, "eL")]
    assert a[0][8] == "services.api.url" and a[0][9] is None and a[0][16:19] == (8000, "explicit", None)
    assert a[1][8] is None and a[1][9] == "command" and a[1][16:19] == (8001, "pack-default", "fastapi")       # a launch-origin entry: launchKind, no keyPath
    b = rd.read("bound_out", {"pairs": [["l1", "c_fetch2"]]})
    assert [(r[2], r[3]) for r in b] == [(0, "eZ"), (1, "eX")]


def test_bound_of_link_returns_an_entry_once_per_edge_in_entry_site_ord_order_and_the_pick_is_the_lowest_site_ord(rd):
    rows = rd.read("bound_of_link", {"ids": ["l1"]})
    assert [(r[1], r[2], r[3]) for r in rows] == [("eL", "c_fetch1", 1), ("eX", "c_fetch1", 0), ("eX", "c_fetch2", 1), ("eZ", "c_fetch2", 0)]
    picks = rd.bound_of_link(["l1"])
    assert [(p.entry_id, p.site_id, p.ord, p.port, p.port_origin) for p in picks] == [("eX", "c_fetch1", 0, 8000, "explicit"), ("eZ", "c_fetch2", 0, None, None)]
    assert all(p.key_path for p in picks)


def test_the_expansions_the_walk_needs_answer_on_the_loaded_graph(rd):
    assert rd.read("to_in", {"ids": ["s_handler"]}) == [("s_handler", "l1", "http", "confirmed", "binding-confirmed", "exact")]
    assert [r[1] for r in rd.read("via_in", {"ids": ["l1"]})] == ["c_fetch1", "c_fetch2"]
    assert rd.read("to_unit_out", {"ids": ["l3"]}) == [("l3", "un", "repo-a", "svc/new", "not-mapped")]
    assert [r[1:] for r in rd.read("controlled_in", {"ids": ["k1"]})] == [("CallSite", "c_res", 401), ("Param", "p1", 11)]
    assert rd.read("has_out:Param", {"ids": ["s_f"]}) == [("s_f", "p1", 11, 0, "x")]
    assert rd.read("provides_out", {"ids": ["s_handler"]})[0][1:4] == ("cp_health", "http", "GET /shop/api/health")
    (row,) = rd.read("units", {"ids": ["ub"]})
    assert row[4] == "stale-kept" and row[9] is False and row[8] == "extract:failed"
    (snap,) = rd.read("snapshot", {})
    assert snap[1:3] == (0, "c482b1cf54631b37eb4572dc86a68fb1") and snap[3] == "1.1"


def test_the_recipe_after_these_reads_is_the_texts_by_name(rd):
    rd.read("to_in", {"ids": ["s_handler"]})
    rd.read("owner_of:Ret", {"ids": ["r1"]})
    assert rd.recipe_queries() == [{"name": "owner_of:Ret", "cypher": EXPECTED["owner_of:Ret"]}, {"name": "to_in", "cypher": EXPECTED["to_in"]}]


# ------------------------------------------------------------------------------------- anchors, live

def test_the_three_reads_on_the_loaded_store_are_repo_scoped(rd):
    got = A.resolve_location(rd, "repo-b", "app/a.py")
    assert (got.state, got.unit_id) == ("mapped", "ub")
    got = A.resolve_location(rd, "repo-a", "app/a.py")
    assert (got.state, got.unit_id) == ("mapped", "ua")
    got = A.resolve_location(rd, "repo-a", "tests/test_a.py", line=3)
    assert (got.state, got.unit_id, got.reason) == ("not mapped", "ua", "test")
    assert A.resolve_location(rd, "repo-b", "tests/test_a.py").state == "not found"
    for path in ("app/types.d.ts", "stubs/x.pyi", "vendor/x.py", "app/b.py/../a.py"):
        assert A.resolve_location(rd, "repo-a", path).state == "not found", path
    with pytest.raises(A.UnknownRepo):
        A.resolve_location(rd, "repo-z", "app/a.py")


def test_file_and_line_rule_on_the_loaded_store(rd):
    def at(line, repo="repo-a", path="app/a.py"):
        return A.resolve_location(rd, repo, path, line=line)
    assert at(17).anchors[0].symbol_id == "s_g" and at(17).anchors[0].qname == "app.Outer.f.g"
    assert at(25).anchors[0].symbol_id == "s_f" and at(60).anchors[0].symbol_id == "s_outer"
    assert at(122).anchors[0].symbol_id == "s_span" and at(122).via == "span"                # span 120-125 and a region 120-125 tie: the span wins
    assert at(132).anchors[0].symbol_id == "s_dec" and at(132).via == "owned-region"          # outside every span but inside an owned region
    assert at(160).anchors[0].symbol_id == "s_dec" and at(160).via == "span"
    got = at(202)
    assert got.status == "ambiguous" and [o.symbol_id for o in got.options] == ["s_t1", "s_t2"]
    assert at(5000).detail == "no-symbol-at-line"
    assert at(5, repo="repo-b").anchors[0].symbol_id == "s_b_outer" and at(50, repo="repo-b").detail == "no-symbol-at-line"


def test_file_and_symbol_and_the_symbol_anchors_carry_qname_on_the_loaded_store(rd):
    got = A.resolve_location(rd, "repo-a", "app/a.py", symbol="app.Outer.f")
    assert got.anchors == (A.SymbolAnchor("s_f", "app.Outer.f", "repo-a", "app/a.py", "ua", 10, 30),)
    got = A.resolve_symbol(rd, "repo-b", "app.Outer")
    assert got.anchors[0].symbol_id == "s_b_outer" and got.anchors[0].unit_id == "ub"
    got = A.resolve_symbol_by_name(rd, "RUN")
    assert got.status == "ambiguous" and [o.symbol_id for o in got.options] == ["s_run1", "s_run2"]
    assert A.resolve_symbol_by_name(rd, "RUN", repo="repo-b").status == "unresolved"


def test_a_hostile_qname_is_inert_data_the_anchor_carries_it_and_the_graph_is_unchanged(world, rd):
    before = _counts(world)
    got = A.resolve_location(rd, "repo-a", "app/b.py", line=3)
    assert got.anchors[0].qname == EVIL_QNAME
    again = A.resolve_symbol(rd, "repo-a", EVIL_QNAME)                         # and asked back as a parameter, it matches itself
    assert again.status == "resolved" and again.anchors[0].symbol_id == "s_evil"
    hostile_repo = "x'}) DETACH DELETE (n) //"
    with pytest.raises(A.AnchorError):
        A.resolve_location(rd, hostile_repo, "app/a.py")
    got = A.resolve_location(rd, "repo-a", "app/a.py'}) DETACH DELETE (n) //", line=3)
    assert got.state == "not found"
    assert _counts(world) == before


def test_contract_anchors_on_the_loaded_store(rd):
    got = A.resolve_contract(rd, "GET /shop/api/health")
    assert got.anchors == (A.ContractAnchor("cp_health", "http", "GET /shop/api/health"),)
    got = A.resolve_contract(rd, "GET /shop/api/items/7", kind="http")
    assert got.status == "resolved" and got.anchors[0].cp_id == "cp_items" and got.via == "path-template"
    got = A.resolve_contract(rd, "billing.invoice.create", kind="mcp_tool")
    assert got.status == "unresolved"                        # mcp_tool links exact only: a longer key is no prefix match
    assert A.resolve_contract(rd, "DELETE /shop/api/items/7", kind="http").status == "unresolved"
    assert A.resolve_contract(rd, "GET /shop/api/items/7").status == "unresolved"


def test_config_anchors_on_the_loaded_store_never_a_launch_entry(rd):
    got = A.resolve_config(rd, "repo-a", "cfg/app.yaml", key_path="services.api.port")
    assert got.anchors == (A.ConfigAnchor("eZ", "repo-a", "cfg/app.yaml", "services.api.port", "ua", 9),)
    got = A.resolve_config(rd, "repo-a", "cfg/app.yaml")
    assert got.status == "ambiguous" and [o.entry_id for o in got.options] == ["eZ", "eX"]        # eL (a launch entry) is not an option
    assert A.resolve_config(rd, "repo-a", "cfg/nothing.yaml", key_path="a").state == "not found"
