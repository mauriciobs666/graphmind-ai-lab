"""S17-L1: the loader writes graph model v1 (``docs/plans/connected-code-graph-p0-graph.md``).

Pure tests (a fake store, hand-built bundles) run everywhere; the ``live`` tests need ``ccg-falkordb``
(127.0.0.1:6390) and touch only fresh ``ccg_p0slice__build_l1*`` keys, deleted at the end (R-B).
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import load as L  # noqa: E402
from ccg.store import parse_ddl  # noqa: E402
from test_live_slice import FakeStore, _mini_rows, h, mini_bundle, mini_facts  # noqa: E402

# ---------------------------------------------------------------------------------------------
# D10 / section 1.6: the index and constraint set comes from ddl_v1.cypher, applied by the raw command.
# ---------------------------------------------------------------------------------------------


def test_index_statements_are_the_ddl_files_indexes_in_file_order():
    assert L.index_statements() == [s[3] for s in parse_ddl() if s[0] == "index"]
    assert len(L.index_statements()) == 17 and "CREATE INDEX FOR (n:SkippedFile) ON (n.repoId)" in L.index_statements()


def test_constraints_applied_are_the_ddl_files_constraints_in_file_order():
    store = FakeStore()
    L.load_build(store, _mini_rows(), L.BUILD_PREFIX + "x", L.LoadReport(), poll_interval=0)
    applied = [(o[2], o[3]) for o in store.ops if o[0] == "constraint"]
    assert applied == [(s[1], s[2]) for s in parse_ddl() if s[0] == "constraint"] and len(applied) == 10


def test_live_store_creates_constraints_by_the_raw_command_never_the_falkordb_py_helper(monkeypatch):
    import falkordb

    cmds, helper = [], []

    class Conn:
        def ping(self):
            return True

        def execute_command(self, *a):
            cmds.append(a)

    class Graph:
        def create_node_unique_constraint(self, *a):  # silently creates a missing index first: masks DDL gaps
            helper.append(a)

    class FakeDB:
        def __init__(self, **kw):
            self.connection = Conn()

        def select_graph(self, key):
            return Graph()

    monkeypatch.setattr(falkordb, "FalkorDB", FakeDB)
    store = L.LiveStore("127.0.0.1", 6390, pw="x")
    store.create_unique_constraint(L.BUILD_PREFIX + "k", "Symbol", "symbolId")
    assert not helper
    assert cmds == [("GRAPH.CONSTRAINT", "CREATE", L.BUILD_PREFIX + "k", "UNIQUE", "NODE", "Symbol", "PROPERTIES", 1, "symbolId")]


# ---------------------------------------------------------------------------------------------
# D7: Symbol extras (depth from parentId, signature only when present), File lang/contentHash.
# ---------------------------------------------------------------------------------------------


def _sym(n, parent=None, **extra):
    r = {"type": "symbol", "id": h(n), "repoId": "repo", "lang": "java", "qname": f"svc/A.java::s{n}", "fileId": h(10),
         "path": "svc/A.java", "kind": "method", "name": f"s{n}", "nameNorm": f"s{n}", "lineStart": n, "lineEnd": n + 1,
         "fingerprint": "f" * 64, "complexity": 2 + n}
    if parent is not None:
        r["parentId"] = h(parent)
    r.update(extra)
    return r


def _bundle_with(*symbols):
    recs = mini_bundle()
    return [r for r in recs if r["type"] in ("header", "file")] + list(symbols)


def test_symbol_depth_is_the_nesting_depth_computed_from_parent_id_and_parent_id_is_not_stored():
    # declared child-before-parent on purpose: depth must not depend on record order
    syms = [_sym(23, parent=22), _sym(22, parent=21), _sym(21, parent=20), _sym(20)]
    rows = L.build_rows([_bundle_with(*syms)], mini_facts())
    depth = {n["symbolId"]: n["depth"] for n in rows.nodes["Symbol"]}
    assert depth == {h(20): 0, h(21): 1, h(22): 2, h(23): 3}
    assert all("parentId" not in n for n in rows.nodes["Symbol"])


def test_symbol_with_an_unknown_parent_is_a_root_and_a_parent_cycle_does_not_hang():
    rows = L.build_rows([_bundle_with(_sym(20, parent=99), _sym(21, parent=22), _sym(22, parent=21))], mini_facts())
    depth = {n["symbolId"]: n["depth"] for n in rows.nodes["Symbol"]}
    assert depth[h(20)] == 0
    assert set(depth) == {h(20), h(21), h(22)}


def test_symbol_carries_complexity_lang_and_signature_only_when_the_record_has_one():
    rows = L.build_rows([_bundle_with(_sym(20, signature="(int,String)"), _sym(21))], mini_facts())
    a, b = rows.nodes["Symbol"]
    assert (a["complexity"], a["lang"], a["signature"]) == (22, "java", "(int,String)")
    assert (b["complexity"], b["lang"]) == (23, "java") and "signature" not in b


def test_file_carries_lang_and_content_hash():
    [f] = L.build_rows([mini_bundle()], mini_facts()).nodes["File"]
    assert (f["lang"], f["contentHash"], f["kind"], f["role"]) == ("python", "0" * 64, "code", "primary")


# ---------------------------------------------------------------------------------------------
# D6: N ("may call any of N") is derived per VIA edge at render time.
# ---------------------------------------------------------------------------------------------

import render as R  # noqa: E402

# (consumer symbol, call site, link, VIA.consumerKey)
VIA_ROWS = [
    ("sym-A", "s1", "L1", "GET /x"), ("sym-A", "s1", "L2", "GET /x"),          # s1 key x reaches L1, L2
    ("sym-A", "s2", "L1", "GET /y"), ("sym-A", "s2", "L3", "GET /y"), ("sym-A", "s2", "L4", "GET /y"),  # s2 key y: L1, L3, L4
    ("sym-A", "s5", "L7", "GET /x"),                                            # another site of A, same key x -> L7 too
    ("sym-B", "s3", "L5", "GET /z"),                                            # sole candidate
    ("sym-B", "s4", "L6", "GET /x"),                                            # other consumer, same key text: not A's
]


def test_one_link_reached_by_two_sites_with_different_keys_has_two_different_n_self_included():
    assert R.candidate_count(VIA_ROWS, "s1", "L1", "GET /x") == 3  # L1, L2 (s1) and L7 (s5): same consumer symbol, key x
    assert R.candidate_count(VIA_ROWS, "s2", "L1", "GET /y") == 3  # L1, L3, L4: key y
    rows = [r for r in VIA_ROWS if r[1] != "s5"]
    assert R.candidate_count(rows, "s1", "L1", "GET /x") == 2
    assert R.candidate_count(rows, "s2", "L1", "GET /y") == 3
    assert R.candidate_count(rows, "s1", "L1", "GET /x") != R.candidate_count(rows, "s2", "L1", "GET /y")


def test_a_sole_candidate_has_n_equal_one_and_another_consumers_same_key_is_not_counted():
    assert R.candidate_count(VIA_ROWS, "s3", "L5", "GET /z") == 1
    assert R.candidate_count(VIA_ROWS, "s4", "L6", "GET /x") == 1  # sym-A also has key x, but it is another consumer symbol


def test_a_link_is_counted_once_however_many_sites_reach_it_with_that_key():
    rows = VIA_ROWS + [("sym-A", "s6", "L2", "GET /x")]
    assert R.candidate_count(rows, "s1", "L1", "GET /x") == 3


def test_a_missing_via_edge_is_an_error_not_a_zero():
    with pytest.raises(KeyError):
        R.candidate_count(VIA_ROWS, "s1", "L5", "GET /x")


def test_a_site_reaching_one_link_under_two_keys_has_one_n_per_edge():
    rows = VIA_ROWS + [("sym-A", "s1", "L1", "GET /y")]
    assert R.candidate_count(rows, "s1", "L1", "GET /x") == 3
    assert R.candidate_count(rows, "s1", "L1", "GET /y") == 3  # L1, L3, L4 (+ nothing else under y)
    with pytest.raises(KeyError):
        R.candidate_count(rows, "s1", "L1", "GET /nope")


# ---------------------------------------------------------------------------------------------
# D8: DATA / CALLS status and reason are stored only when not the default.
# ---------------------------------------------------------------------------------------------


def _edit(recs, rid, **kw):
    for r in recs:
        if r.get("id") == rid:
            r.update(kw)
    return recs


def test_calls_status_and_reason_are_stored_only_when_not_the_default_each_on_its_own():
    recs = mini_bundle()
    _edit(recs, h(40), status="possible")                      # reason stays the default
    rows = L.build_rows([recs], mini_facts())
    assert rows.edges[("CallSite", "CALLS", "Symbol")] == [{"s": h(40), "t": h(21), "p": {"status": "possible"}}]
    recs = _edit(mini_bundle(), h(40), status="possible", reason="name-match")
    assert L.build_rows([recs], mini_facts()).edges[("CallSite", "CALLS", "Symbol")][0]["p"] == {
        "status": "possible", "reason": "name-match"}
    recs = _edit(mini_bundle(), h(40), reason="alias")           # the status stays the default
    assert L.build_rows([recs], mini_facts()).edges[("CallSite", "CALLS", "Symbol")][0]["p"] == {"reason": "alias"}


def test_data_non_default_status_and_reason_are_kept_the_paths_always():
    recs = _edit(mini_bundle(), h(50), status="possible", reason="field-truncated")
    rows = L.build_rows([recs], mini_facts())
    assert rows.edges[("CallSite", "DATA", "Param")][0]["p"] == {
        "fromPath": "p0.k", "toPath": "a0", "status": "possible", "reason": "field-truncated"}
    assert rows.edges[("Ret", "DATA", "CallSite")][0]["p"] == {"fromPath": "r", "toPath": "ret"}


def test_a_group_mixing_rows_with_and_without_props_writes_an_empty_map_for_the_bare_ones():
    rows = L.GraphRows()
    rows.snapshot = {"snapshotId": "m"}
    rows.edges[("CallSite", "CALLS", "Symbol")] = [{"s": "a", "t": "b", "p": {"status": "possible"}}, {"s": "c", "t": "d"}]
    store = FakeStore()
    L.load_build(store, rows, L.BUILD_PREFIX + "m", L.LoadReport(), poll_interval=0)
    [w] = [o for o in store.ops if o[0] == "query" and o[5] and "rows" in o[5]]
    assert [r["p"] for r in w[5]["rows"]] == [{"status": "possible"}, {}]  # a missing p is a driver error live
    assert "SET e = r.p" in w[2]
    assert rows.edges[("CallSite", "CALLS", "Symbol")][1] == {"s": "c", "t": "d"}  # the rows themselves are not mutated


# ---------------------------------------------------------------------------------------------
# D9: Dependency nodes and DEPENDS_ON {version, manifestPath, line, scope}.
# ---------------------------------------------------------------------------------------------


def _dep(n, **kw):
    r = {"type": "dependency", "id": h(80 + n), "depId": h(70 + n), "ecosystem": "PyPI", "package": f"pkg{n}",
         "version": ">=1", "scope": "runtime", "manifestPath": "svc/pyproject.toml", "line": 3 + n}
    r.update(kw)
    return r


def test_dependency_records_become_nodes_and_depends_on_edges_with_scope():
    recs = mini_bundle() + [_dep(1), _dep(2, scope="dev"), _dep(1, id=h(90), scope="dev", line=40)]  # pkg1 declared twice
    rows = L.build_rows([recs], mini_facts())
    assert rows.nodes["Dependency"] == [
        {"depId": h(71), "ecosystem": "PyPI", "package": "pkg1"}, {"depId": h(72), "ecosystem": "PyPI", "package": "pkg2"}]
    e = rows.edges[("Unit", "DEPENDS_ON", "Dependency")]
    assert e == [
        {"s": h(1), "t": h(71), "p": {"version": ">=1", "manifestPath": "svc/pyproject.toml", "line": 4, "scope": "runtime"}},
        {"s": h(1), "t": h(72), "p": {"version": ">=1", "manifestPath": "svc/pyproject.toml", "line": 5, "scope": "dev"}},
        {"s": h(1), "t": h(71), "p": {"version": ">=1", "manifestPath": "svc/pyproject.toml", "line": 40, "scope": "dev"}},
    ]
    assert ("Unit", "DEPENDS_ON", "Dependency") in L.EDGE_ORDER and "Dependency" in L.NODE_ORDER


# ---------------------------------------------------------------------------------------------
# Section 1.3: SkippedFile nodes and SKIPS edges, from header.skippedFiles only.
# ---------------------------------------------------------------------------------------------


def test_header_skipped_files_become_skipped_file_nodes_and_skips_edges():
    recs = mini_bundle()
    recs[0]["skippedFiles"] = [{"path": "svc/test_a.py", "reason": "test"}, {"path": "svc/x.tsx", "reason": "frontend-ignore"}]
    rows = L.build_rows([recs], mini_facts())
    assert rows.nodes["SkippedFile"] == [
        {"repoId": "repo", "unitId": h(1), "path": "svc/test_a.py", "reason": "test"},
        {"repoId": "repo", "unitId": h(1), "path": "svc/x.tsx", "reason": "frontend-ignore"}]
    assert rows.edges[("Unit", "SKIPS", "SkippedFile")] == [{"s": h(1), "t": "svc/test_a.py"}, {"s": h(1), "t": "svc/x.tsx"}]
    # a coverage fact only: no DEFINES, HAS_FILE, kind or role
    assert not any(k[2] == "SkippedFile" and k[1] != "SKIPS" for k in rows.edges)
    assert all(set(n) == {"repoId", "unitId", "path", "reason"} for n in rows.nodes["SkippedFile"])


def test_a_bundle_without_skipped_files_yields_no_skipped_file_rows():
    rows = L.build_rows([mini_bundle()], mini_facts())
    assert not rows.nodes.get("SkippedFile") and ("Unit", "SKIPS", "SkippedFile") not in rows.edges


def test_skips_edge_statement_matches_the_skipped_file_by_unit_and_path_not_by_a_missing_id():
    stmt = L.edge_statement("Unit", "SKIPS", "SkippedFile", False)
    assert "(b:SkippedFile {unitId: r.s, path: r.t})" in stmt and "(a:Unit {unitId: r.s})" in stmt
    assert L.node_statement("SkippedFile") == "UNWIND $rows AS r CREATE (n:SkippedFile) SET n = r"


def test_impact_reader_coalesces_the_absent_calls_and_data_status_to_the_default():
    # D8 minimum so a sparse graph reads right until impact.py's own unit adds the behavioural test
    import impact as I

    seen = []
    reader = I.CypherReader(lambda q, p: seen.append(q) or [])
    reader.calls_in(["a"]), reader.data_in("CallSite", ["a"])
    assert "coalesce(e.status, 'confirmed')" in seen[0] and "coalesce(d.status, 'confirmed')" in seen[1]


# ---------------------------------------------------------------------------------------------
# Live (ccg-falkordb): the lookup scope, the loaded v1 shapes, the D6 Cypher. Fresh build keys only.
# ---------------------------------------------------------------------------------------------

import lookup as K  # noqa: E402

HOST, PORT = "127.0.0.1", 6390


@pytest.fixture(scope="module")
def live_store():
    try:
        return L.LiveStore(HOST, PORT)
    except FileNotFoundError as exc:
        pytest.skip(f"ccg-falkordb admin password file not readable ({exc.filename}); see cpg/infra/README.md")
    except Exception as exc:  # noqa: BLE001 -- any connection failure is "unreachable"
        pytest.skip(f"ccg-falkordb unreachable at {HOST}:{PORT} ({type(exc).__name__}); start it with cpg/infra/ccg-falkordb.sh up")


class Build:
    """A fresh ``ccg_p0slice__build_l1<name>`` key loaded from rows, always deleted."""

    def __init__(self, store, name, rows):
        self.store, self.key, self.rows = store, L.check_key(L.BUILD_PREFIX + "l1" + name), rows

    def __enter__(self):
        if self.store.exists(self.key):
            self.store.delete(self.key)
        L.load_build(self.store, self.rows, self.key, L.LoadReport(), snapshot=False, poll_interval=0.2)
        self.run = lambda q, p=None: self.store.query(self.key, q, p).result_set
        return self

    def __exit__(self, *exc):
        if self.store.exists(self.key):
            self.store.delete(self.key)


def _unit_bundle(n, repo, *, mapped, skipped=(), symbols=()):
    """One unit of ``repo``; ``mapped`` paths get a File + the given symbols, ``skipped`` go in the header."""
    unit = h(100 * n + 1)
    recs = [{"type": "header", "unitId": unit, "unit": {"repoId": repo, "rootPath": "svc", "lang": "python"},
             "builtCommit": "c" * 40, "builtTree": "t" * 40, "builtAt": "2026-01-02T03:04:05Z", "bundleSchemaVersion": "1.0"}]
    if skipped:
        recs[0]["skippedFiles"] = [{"path": p, "reason": "test"} for p in skipped]
    for i, path in enumerate(mapped):
        fid = h(100 * n + 10 + i)
        recs.append({"type": "file", "id": fid, "repoId": repo, "unitId": unit, "path": path, "kind": "code",
                     "role": "primary", "lang": "python", "contentHash": "0" * 64, "lineCount": 9})
        for j, (name, ls, le) in enumerate(symbols):
            recs.append({"type": "symbol", "id": h(100 * n + 50 + j), "repoId": repo, "lang": "python", "qname": f"{path}::{name}",
                         "fileId": fid, "path": path, "kind": "function", "name": name, "nameNorm": name, "lineStart": ls,
                         "lineEnd": le, "fingerprint": "f" * 64, "complexity": 1})
    return recs


@pytest.mark.live
def test_lookup_is_repo_scoped_the_same_path_mapped_in_two_repos_and_skipped_in_a_third(live_store):
    path = "svc/a.py"
    bundles = [_unit_bundle(1, "repoA", mapped=[path], symbols=[("fa", 1, 5), ("ga", 6, 9)]),
               _unit_bundle(2, "repoB", mapped=[], skipped=[path]),
               _unit_bundle(3, "repoC", mapped=[path], symbols=[("fc", 2, 4)])]
    with Build(live_store, "scope", L.build_rows(bundles, mini_facts())) as b:
        a, bb, c = h(101), h(201), h(301)
        assert K.lookup_path(b.run, "repoA", path) == {"state": "mapped", "unitId": a}
        assert K.lookup_path(b.run, "repoB", path) == {"state": "not mapped", "unitId": bb, "reason": "test"}
        assert K.lookup_path(b.run, "repoC", path) == {"state": "mapped", "unitId": c}
        assert K.lookup_path(b.run, "repoD", path) == {"state": "not found"}          # a repo the store does not hold
        assert K.lookup_path(b.run, "repoA", "svc/types.d.ts") == {"state": "not found"}  # declaration file: in neither set
        # the File+line anchor read returns only the queried repo's symbols
        assert sorted(r[0] for r in K.file_symbols(b.run, "repoA", path)) == [h(150), h(151)]
        assert sorted(r[0] for r in K.file_symbols(b.run, "repoC", path)) == [h(350)]
        assert K.file_symbols(b.run, "repoB", path) == []                             # skipped there: no symbols
        # a SkippedFile is a coverage fact only
        assert b.run("MATCH (s:SkippedFile)-[r]-() RETURN type(r), count(*)") == [["SKIPS", 1]]
        assert b.run("MATCH (s:SkippedFile) RETURN count(s)") == [[1]]


@pytest.mark.live
def test_loaded_v1_shapes_round_trip_symbol_extras_edge_props_and_sparse_status(live_store):
    recs = mini_bundle()
    recs[0]["skippedFiles"] = [{"path": "svc/skip_a.py", "reason": "test"}]
    recs.append({"type": "dependency", "id": h(90), "depId": h(91), "ecosystem": "PyPI", "package": "x", "version": "1",
                 "scope": "dev", "manifestPath": "svc/p.toml", "line": 2})
    facts = {"providers": [{"handlerId": h(21), "key": "GET /x", "status": "possible", "reason": "partial-key"}],
             "consumers": [], "links": []}
    recs.append(_sym(24, parent=20, signature="(a)"))
    with Build(live_store, "shapes", L.build_rows([recs], facts)) as b:
        [sym] = b.run("MATCH (s:Symbol {symbolId: $i}) RETURN s.depth, s.complexity, s.lang, s.signature, s.ownedRegions, keys(s)", {"i": h(20)})
        assert sym[:5] == [0, 1, "python", None, [4, 5, 7, 7]] and "parentId" not in sym[5]
        [child] = b.run("MATCH (s:Symbol {symbolId: $i}) RETURN s.depth, s.signature", {"i": h(24)})
        assert child == [1, "(a)"]
        assert b.run("MATCH (f:File {fileId: $i}) RETURN f.lang, f.contentHash", {"i": h(10)}) == [["python", "0" * 64]]
        assert b.run("MATCH ()-[e:CALLS]->() RETURN keys(e)") == [[[]]]  # the one stored CALLS has the default: no props
        assert b.run("MATCH (:Unit)-[e:DEPENDS_ON]->(d:Dependency) RETURN e.scope, e.version, d.package") == [["dev", "1", "x"]]
        assert b.run("MATCH (:Symbol)-[e:PROVIDES]->(:ContractPoint) RETURN e.status, e.reason") == [["possible", "partial-key"]]
        assert b.run("MATCH (:Unit)-[:SKIPS]->(s:SkippedFile) RETURN s.path, s.reason, s.repoId") == [["svc/skip_a.py", "test", "repo"]]
        # the sparse DATA edges read back with the default through impact.py's reader
        import impact as I
        reader = I.CypherReader(b.run)
        assert {st for *_, st in reader.data_in("Param", [h(30)])} == {"confirmed"}
        assert {st for _, _, st in reader.calls_in([h(21)])} == {"confirmed"}


@pytest.mark.live
def test_candidate_count_cypher_agrees_with_the_row_helper(live_store):
    """D6 on a hand-built store: one consumer symbol, two sites with different keys reaching a shared link."""
    rows = L.GraphRows()
    rows.snapshot = {"snapshotId": "d6"}
    ids = {k: h(500 + i) for i, k in enumerate(["symA", "symB", "s1", "s2", "s3", "L1", "L2", "L3", "P1", "P2", "P3"])}
    i = ids
    rows.nodes["Symbol"] = [{"symbolId": i[k]} for k in ("symA", "symB", "P1", "P2", "P3")]
    rows.nodes["CallSite"] = [{"callSiteId": i[k]} for k in ("s1", "s2", "s3")]
    rows.nodes["Link"] = [{"linkId": i[k]} for k in ("L1", "L2", "L3")]
    rows.edges[("Symbol", "HAS_CALL", "CallSite")] = [{"s": i["symA"], "t": i["s1"]}, {"s": i["symA"], "t": i["s2"]},
                                                      {"s": i["symB"], "t": i["s3"]}]
    via = [("s1", "L1", "GET /x"), ("s1", "L2", "GET /x"), ("s2", "L1", "GET /y"), ("s2", "L3", "GET /y"), ("s3", "L1", "GET /x")]
    rows.edges[("CallSite", "VIA", "Link")] = [{"s": i[s], "t": i[l], "p": {"status": "confirmed", "reason": "key-match", "consumerKey": k}}
                                               for s, l, k in via]
    owner = {"s1": "symA", "s2": "symA", "s3": "symB"}
    flat = [(i[owner[s]], i[s], i[l], k) for s, l, k in via]
    import render as R
    with Build(live_store, "d6", rows) as b:
        for s, l, k in via:
            got = b.run(R.CANDIDATE_COUNT_CYPHER, {"site": i[s], "link": i[l], "key": k})
            assert got == [[R.candidate_count(flat, i[s], i[l], k)]], (s, l, k)
        assert b.run(R.CANDIDATE_COUNT_CYPHER, {"site": i["s1"], "link": i["L1"], "key": "GET /x"}) == [[2]]
        assert b.run(R.CANDIDATE_COUNT_CYPHER, {"site": i["s2"], "link": i["L1"], "key": "GET /y"}) == [[2]]
        assert b.run(R.CANDIDATE_COUNT_CYPHER, {"site": i["s3"], "link": i["L1"], "key": "GET /x"}) == [[1]]


@pytest.mark.live
def test_lookup_order_is_mapped_then_skipped_then_not_found_a_path_in_both_sets_of_one_repo_is_mapped(live_store):
    both, only_skipped = "svc/both.py", "svc/only_skipped.py"
    bundles = [_unit_bundle(1, "repoA", mapped=[both], skipped=[both, only_skipped], symbols=[("f", 1, 5)])]
    with Build(live_store, "order", L.build_rows(bundles, mini_facts())) as b:
        assert b.run("MATCH (f:File {path: $p, repoId: 'repoA'}) RETURN count(f)", {"p": both}) == [[1]]
        assert b.run("MATCH (s:SkippedFile {path: $p, repoId: 'repoA'}) RETURN count(s)", {"p": both}) == [[1]]  # in both
        assert K.lookup_path(b.run, "repoA", both) == {"state": "mapped", "unitId": h(101)}
        assert K.lookup_path(b.run, "repoA", only_skipped) == {"state": "not mapped", "unitId": h(101), "reason": "test"}
        assert K.lookup_path(b.run, "repoA", "svc/nowhere.py") == {"state": "not found"}


@pytest.mark.live
def test_candidate_count_counts_a_link_once_when_two_sites_of_one_symbol_reach_it_under_the_same_key(live_store):
    """D6's ``count(DISTINCT l)``: s1 and s4 (both symA) reach L1 under 'GET /x'; s1 also reaches L2."""
    rows = L.GraphRows()
    rows.snapshot = {"snapshotId": "d6dup"}
    i = {k: h(700 + n) for n, k in enumerate(["symA", "s1", "s4", "L1", "L2"])}
    rows.nodes["Symbol"] = [{"symbolId": i["symA"]}]
    rows.nodes["CallSite"] = [{"callSiteId": i["s1"]}, {"callSiteId": i["s4"]}]
    rows.nodes["Link"] = [{"linkId": i["L1"]}, {"linkId": i["L2"]}]
    rows.edges[("Symbol", "HAS_CALL", "CallSite")] = [{"s": i["symA"], "t": i["s1"]}, {"s": i["symA"], "t": i["s4"]}]
    via = [("s1", "L1"), ("s1", "L2"), ("s4", "L1")]
    rows.edges[("CallSite", "VIA", "Link")] = [{"s": i[s], "t": i[l], "p": {"status": "confirmed", "reason": "key-match", "consumerKey": "GET /x"}}
                                               for s, l in via]
    flat = [(i["symA"], i[s], i[l], "GET /x") for s, l in via]
    with Build(live_store, "d6dup", rows) as b:
        for s, l, n in (("s1", "L1", 2), ("s1", "L2", 2), ("s4", "L1", 2)):
            got = b.run(R.CANDIDATE_COUNT_CYPHER, {"site": i[s], "link": i[l], "key": "GET /x"})
            assert got == [[n]] == [[R.candidate_count(flat, i[s], i[l], "GET /x")]], (s, l)
