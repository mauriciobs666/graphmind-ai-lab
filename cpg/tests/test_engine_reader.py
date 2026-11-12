"""P1-21 reader layer, unit half (a fake database; the live half is ``tests/test_engine_reader_live.py``).

Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §5 (the texts), §7 (the ``ccg_engine_<s>`` role), §8 P1-21 (a), (c)-(f);
``docs/plans/connected-code-graph-p1-answers.md`` P1-21 behaviour (1), (6). The expected texts below are a pinned copy of §5, written out here and not
read from ``ccg.engine.reader``; ``test_engine_reader_live.py::test_the_texts_are_the_plans`` also checks them against the plan document itself."""
from __future__ import annotations

import re

import pytest
from engine_helpers import FakeDb

from ccg import sysname
from ccg.engine import config, reader
from ccg.engine.reader import CypherReader, GraphDataError, ReaderError, ReadTimeout, ResultTooLarge

POINT_ID = {"CallSite": "callSiteId", "Param": "pointId", "Ret": "pointId", "Cond": "pointId"}
OWNER_REL = {"CallSite": "HAS_CALL", "Param": "HAS_PARAM", "Ret": "HAS_RETURN", "Cond": "HAS_COND"}

EXPECTED: dict[str, str] = {
    # anchors
    "repo_exists": "MATCH (r:Repo {repoId: $repo}) RETURN r.repoId",
    "file_by_path": "MATCH (f:File {path: $path}) WHERE f.repoId = $repo RETURN f.fileId, f.unitId",
    "skipped_by_path": "MATCH (s:SkippedFile {path: $path}) WHERE s.repoId = $repo RETURN s.unitId, s.reason",
    "symbols_of_file": "MATCH (f:File {path: $path})-[:DEFINES]->(s:Symbol) WHERE f.repoId = $repo RETURN s.symbolId, s.qname, s.lineStart, s.lineEnd, s.ownedRegions, s.depth",
    "symbol_by_qname": "MATCH (s:Symbol {qname: $qname}) WHERE s.repoId = $repo RETURN s.symbolId, s.unitId, s.path, s.lineStart, s.lineEnd, s.ownedRegions",
    "symbol_by_name": "MATCH (s:Symbol {nameNorm: $name}) RETURN s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart",
    "config_by_path": "MATCH (e:ConfigEntry {path: $path}) WHERE e.repoId = $repo RETURN e.entryId, e.unitId, e.line, e.keyPath, e.launchKind",
    "cp_by_key": "MATCH (cp:ContractPoint {keyNorm: $key}) RETURN cp.cpId, cp.kind, cp.keyNorm",
    "cp_keys": "MATCH (cp:ContractPoint) WHERE cp.kind = $kind RETURN cp.cpId, cp.keyNorm",
    "snapshot": "MATCH (s:Snapshot) RETURN s.snapshotId, s.verdictLogOffset, s.verdictLogHash, s.graphModelVersion, s.bundleSchemaVersion",
    "units": "UNWIND $ids AS i MATCH (u:Unit {unitId: i}) RETURN u.unitId, u.repoId, u.rootPath, u.language, u.buildStatus, u.builtCommit, u.builtAt, u.failedAt, u.failReason, u.flowAvailable",
    # expansions, impact
    "calls_in": "UNWIND $ids AS i MATCH (c:CallSite)-[e:CALLS]->(s:Symbol {symbolId: i}) RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, coalesce(e.status, 'confirmed'), coalesce(e.reason, 'resolved-call')",
    "name_in": "UNWIND $ids AS i MATCH (c:CallSite)-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(s:Symbol {symbolId: i}) OPTIONAL MATCH (c)-[r:CALLS]->(s) WITH i, c, s, r WHERE r IS NULL OPTIONAL MATCH (c)-[d:DISMISSED]->(s) WITH i, c, d WHERE d IS NULL OPTIONAL MATCH (c)-[x:CALLS]->() RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, count(x)",
    "controlled_in": "UNWIND $ids AS i MATCH (p)-[:CONTROLLED_BY]->(k:Cond {pointId: i}) RETURN i, labels(p)[0], coalesce(p.callSiteId, p.pointId), p.line",
    "to_in": "UNWIND $ids AS i MATCH (l:Link)-[:TO]->(s:Symbol {symbolId: i}) RETURN i, l.linkId, l.kind, l.status, l.reason, l.matcher",
    "via_in": "UNWIND $ids AS i MATCH (c:CallSite)-[v:VIA]->(l:Link {linkId: i}) RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, v.status, v.reason, v.consumerKey",
    "consumes_in": "UNWIND $ids AS i MATCH (c:CallSite)-[e:CONSUMES]->(cp:ContractPoint {cpId: i}) RETURN i, c.callSiteId, c.repoId, c.unitId, c.path, c.line, c.calleeName, e.status, e.reason",
    "on_in": "UNWIND $ids AS i MATCH (l:Link)-[:ON]->(cp:ContractPoint {cpId: i}) RETURN i, l.linkId, l.kind, l.status, l.reason, l.matcher",
    "provides_in": "UNWIND $ids AS i MATCH (s:Symbol)-[e:PROVIDES]->(cp:ContractPoint {cpId: i}) RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, e.status, e.reason",
    "bound_in": "UNWIND $ids AS i MATCH (l:Link)-[b:BOUND_BY]->(e:ConfigEntry {entryId: i}) RETURN i, l.linkId, l.kind, l.status, l.reason, b.siteId, b.ord",
    # expansions, depends-on
    "calls_out": "UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[e:CALLS]->(s:Symbol) RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, coalesce(e.status, 'confirmed'), coalesce(e.reason, 'resolved-call')",
    "name_out": "UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(s:Symbol) OPTIONAL MATCH (c)-[r:CALLS]->(s) WITH i, c, s, r WHERE r IS NULL OPTIONAL MATCH (c)-[d:DISMISSED]->(s) WITH i, c, s, d WHERE d IS NULL OPTIONAL MATCH (c)-[x:CALLS]->() RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, count(x)",
    "has_out:CallSite": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_CALL]->(p:CallSite) RETURN i, p.callSiteId, p.line, p.calleeName",
    "has_out:Param": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_PARAM]->(p:Param) RETURN i, p.pointId, p.line, p.index, p.name",
    "has_out:Ret": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_RETURN]->(p:Ret) RETURN i, p.pointId, p.line",
    "has_out:Cond": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[:HAS_COND]->(p:Cond) RETURN i, p.pointId, p.line, p.code",
    "consumes_out": "UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[e:CONSUMES]->(cp:ContractPoint) RETURN i, cp.cpId, cp.kind, cp.keyNorm, e.status, e.reason",
    "provides_out": "UNWIND $ids AS i MATCH (s:Symbol {symbolId: i})-[e:PROVIDES]->(cp:ContractPoint) RETURN i, cp.cpId, cp.kind, cp.keyNorm, e.status, e.reason",
    "via_out": "UNWIND $ids AS i MATCH (c:CallSite {callSiteId: i})-[v:VIA]->(l:Link) RETURN i, l.linkId, l.kind, l.status, l.reason, l.matcher, v.status, v.reason, v.consumerKey",
    "to_out": "UNWIND $ids AS i MATCH (l:Link {linkId: i})-[:TO]->(s:Symbol) RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart",
    "to_unit_out": "UNWIND $ids AS i MATCH (l:Link {linkId: i})-[:TO_UNIT]->(u:Unit) RETURN i, u.unitId, u.repoId, u.rootPath, u.buildStatus",
    # evidence
    "lifted_of": "UNWIND $pairs AS q MATCH (l:Link {linkId: q[0]})-[r:LIFTED_VIA]->(t) WHERE r.siteId = q[1] RETURN q[0], q[1], r.ord, labels(t)[0], coalesce(t.callSiteId, t.symbolId), t.repoId, t.unitId, t.path, r.line, r.urlPath, r.verbPath, r.viaSiteId, r.viaLine, coalesce(t.calleeName, t.qname) ORDER BY q[0], q[1], r.ord",
    "bound_out": "UNWIND $pairs AS q MATCH (l:Link {linkId: q[0]})-[b:BOUND_BY]->(e:ConfigEntry) WHERE b.siteId = q[1] RETURN q[0], q[1], b.ord, e.entryId, e.repoId, e.unitId, e.path, e.line, e.keyPath, e.launchKind, e.protocol, e.port, e.envVar, e.mountPath, e.service, e.argv0, b.port, b.portOrigin, b.packId ORDER BY q[0], q[1], b.ord",
    "bound_of_link": "UNWIND $ids AS i MATCH (l:Link {linkId: i})-[b:BOUND_BY]->(e:ConfigEntry) RETURN i, e.entryId, b.siteId, b.ord, e.repoId, e.unitId, e.path, e.line, e.keyPath, e.launchKind, b.port, b.portOrigin, b.packId ORDER BY i, e.entryId, b.siteId, b.ord",
    "candidate_count": "UNWIND $pairs AS q MATCH (c:CallSite {callSiteId: q[0]})-[v:VIA]->(l:Link {linkId: q[1]}) MATCH (s:Symbol)-[:HAS_CALL]->(c) MATCH (s)-[:HAS_CALL]->(:CallSite)-[v2:VIA]->(l2:Link) WHERE v2.consumerKey = v.consumerKey RETURN q[0], q[1], count(DISTINCT l2)",
}
# the templated names, one text per label of the row's set (the ``<Label>`` token is chosen by the engine from a fixed set, never by a caller)
for _lab in ("CallSite", "Param", "Ret", "Cond"):
    EXPECTED[f"owner_of:{_lab}"] = (f"UNWIND $ids AS i MATCH (s:Symbol)-[:{OWNER_REL[_lab]}]->(p:{_lab} {{{POINT_ID[_lab]}: i}}) "
                                    "RETURN i, s.symbolId, s.repoId, s.unitId, s.qname, s.path, s.lineStart, s.lineEnd, s.ownedRegions")
for _lab in ("Param", "CallSite"):
    EXPECTED[f"data_in:{_lab}"] = (f"UNWIND $ids AS i MATCH (u)-[d:DATA]->(p:{_lab} {{{POINT_ID[_lab]}: i}}) RETURN i, labels(u)[0], coalesce(u.callSiteId, u.pointId), "
                                   "u.line, d.fromPath, d.toPath, coalesce(d.status, 'confirmed'), coalesce(d.reason, 'intra-flow')")
for _lab in ("CallSite", "Ret", "Cond"):
    EXPECTED[f"data_out:{_lab}"] = (f"UNWIND $ids AS i MATCH (p:{_lab} {{{POINT_ID[_lab]}: i}})-[d:DATA]->(t) RETURN i, labels(t)[0], coalesce(t.callSiteId, t.pointId), "
                                    "t.line, d.fromPath, d.toPath, coalesce(d.status, 'confirmed'), coalesce(d.reason, 'intra-flow')")
for _lab in ("CallSite", "Ret", "Cond", "Param"):
    EXPECTED[f"controlled_out:{_lab}"] = (f"UNWIND $ids AS i MATCH (p:{_lab} {{{POINT_ID[_lab]}: i}})-[:CONTROLLED_BY]->(k:Cond) RETURN i, k.pointId, k.line, k.code")

NAME_RE = re.compile(r"^[a-z_]+(:[A-Za-z]+)?$")
PARAMS_OF = {"anchor": {"repo", "path", "qname", "name", "key", "kind"}}


def _params_for(text: str) -> dict:
    out: dict = {}
    for p in sorted(set(re.findall(r"\$([a-z]+)", text))):
        out[p] = ["x"] if p == "ids" else [["a", "b"]] if p == "pairs" else "x"
    return out


# ------------------------------------------------------------------------------------- (a) the texts

def test_every_text_of_the_plan_is_the_one_in_the_code_character_for_character():
    assert dict(reader.QUERIES) == EXPECTED


def test_every_name_is_a_recipe_name():
    for name in reader.QUERIES:
        assert NAME_RE.match(name), name


def test_config_values_are_the_plans():
    assert (config.DEFAULT_DEPTH_BOUND, config.LEVEL_GUARD, config.CHUNK, config.RESULTSET_SIZE) == (2, 64, 1000, 10000)
    assert 0 < config.QUERY_TIMEOUT_MS <= 10_000                 # the pinned instance's TIMEOUT_MAX: a larger value is refused by the server
    assert config.OPTIONS_MAX >= 2


# ------------------------------------------------------------------------------------- the read-only lint

WRITES = [
    "CREATE (n:X) RETURN n", "MATCH (n) SET n.a = 1", "MATCH (n) DELETE n", "MATCH (n) DETACH DELETE n", "MERGE (n:X {a: 1})",
    "MATCH (n) REMOVE n.a", "DROP INDEX x", "CALL db.idx.fulltext.createNodeIndex('X', 'a')", "LOAD CSV FROM 'x' AS r RETURN r",
    "MATCH (n) FOREACH (x IN [1] | CREATE (:Y))", "create (n:X)", "match (n)\nset n.a = 1", "MATCH (n) RETURN n;", "MATCH (n) RETURN n; MATCH (m) DELETE m",
    "MATCH (n) RETURN n // CREATE", "MATCH (n) /* x */ RETURN n", "MATCH (n:`SET`) RETURN n", "MATCH (n) RETURN 'a'", 'MATCH (n) RETURN "confirmed"',
    "MATCH (n) WHERE n.a = 1 OR n.b = 2 RETURN n", "MATCH (n) RETURN n\tOR", "UNWIND $ids AS i CREATE (:X)", "MATCH (n) RETURN n UNION CALL db.labels()",
    "MATCH (n)-[e:CALLS]->(m) SET\te.a = 1", "MATCH (n) RETURN n SET n.a = 1",
]


@pytest.mark.parametrize("text", WRITES)
def test_the_lint_refuses_every_write_shape_and_every_other_non_read_construct(text):
    with pytest.raises(ReaderError):
        reader.assert_read_only(text)


@pytest.mark.parametrize("word", ["CREATE", "MERGE", "SET", "DELETE", "DETACH", "REMOVE", "DROP", "CALL", "FOREACH", "LOAD", "CSV", "OR"])
def test_each_forbidden_word_is_refused_on_its_own(word):
    for text in (f"MATCH (n) {word} RETURN n", f"MATCH (n) {word.lower()} RETURN n", f"MATCH (n) WHERE n.a = $a {word.title()} (n.b)"):
        with pytest.raises(ReaderError):
            reader.assert_read_only(text)


@pytest.mark.parametrize("text", [
    "MATCH (n:`Label`) RETURN n", "MATCH (n)\nRETURN n", "MATCH (n) RETURN n\t", "MATCH (n) RETURN n\r", "MATCH (n) RETURN \u00e9", "MATCH (n) RETURN n\x00",
    "MATCH (n) RETURN 'confirmed", "MATCH (n) RETURN confirmed'", "MATCH (n) RETURN 'confirmed' + 'x", "", "   ", None, 5,
])
def test_the_lint_refuses_a_text_that_is_not_one_plain_line_of_the_engines_shape(text):
    with pytest.raises(ReaderError):
        reader.assert_read_only(text)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_lint_accepts_every_registered_text(name):
    reader.assert_read_only(EXPECTED[name])


def test_the_lint_ignores_a_write_word_that_is_part_of_an_identifier():
    reader.assert_read_only("MATCH (n:Offset) WHERE n.settled = $v RETURN n.created_at, n.reset")


def test_a_text_that_is_not_registered_is_never_sent(monkeypatch):
    db = FakeDb()
    r = CypherReader(db, "shop")
    for name in ("evil", "owner_of:Symbol", "owner_of:", "owner_of", "has_out:CallSite ", "calls_in; DROP", "", "owner_of:CallSite\n"):
        with pytest.raises(ReaderError, match="unknown query"):
            r.read(name, {"ids": ["a"]})
    assert db.calls == []


def test_a_registered_text_that_fails_the_lint_is_refused_at_read_time_not_only_at_import(monkeypatch):
    monkeypatch.setitem(reader.QUERIES, "planted", "MATCH (n) SET n.a = 1")
    db = FakeDb()
    with pytest.raises(ReaderError, match="read-only"):
        CypherReader(db, "shop").read("planted", {})
    assert db.calls == []


def test_the_table_check_that_runs_at_import_refuses_a_bad_name_and_a_write_text(monkeypatch):
    reader._validate_table()
    monkeypatch.setitem(reader.QUERIES, "Bad-Name", "MATCH (n) RETURN n")
    with pytest.raises(ReaderError, match="recipe name"):
        reader._validate_table()
    monkeypatch.delitem(reader.QUERIES, "Bad-Name")
    monkeypatch.setitem(reader.QUERIES, "planted", "MATCH (n) DELETE n")
    with pytest.raises(ReaderError, match="read-only"):
        reader._validate_table()


def test_no_text_has_an_or_or_a_string_literal_beyond_the_d8_defaults():
    for name, text in EXPECTED.items():
        assert not re.search(r"\bOR\b", text), name
    literals = {m for t in EXPECTED.values() for m in re.findall(r"'([^']*)'", t)}
    assert literals == {"confirmed", "resolved-call", "intra-flow"}


# ------------------------------------------------------------------------------------- construction: the key and the role

def test_the_reader_selects_the_live_key_of_a_validated_system_name_only():
    db = FakeDb()
    CypherReader(db, "shop")
    assert db.keys == [sysname.live_key("shop")]
    for bad in ("Shop", "shop_", "a__b", "shop2'] DEL x", "", "x" * 49, "../shop", "shop\n", None, 7):
        db = FakeDb()
        with pytest.raises(ReaderError, match="invalid system name"):
            CypherReader(db, bad)
        assert db.keys == [] and db.calls == []


def test_for_system_connects_as_the_engine_role_and_refuses_the_dev_port(monkeypatch):
    seen = {}

    def fake_connect(host, port, *, username, password, socket_timeout=300):
        seen.update(host=host, port=port, username=username)
        return FakeDb()

    monkeypatch.setattr(reader.L, "connect", fake_connect)
    monkeypatch.setattr(reader.acl, "read_credentials", lambda system: {"engine": "pw", "build": "x", "verdict": "y", "raw": "z"})
    CypherReader.for_system("shop", "127.0.0.1", 16999)
    assert seen == {"host": "127.0.0.1", "port": 16999, "username": "ccg_engine_shop"}
    with pytest.raises(ReaderError):
        CypherReader.for_system("Bad_", "127.0.0.1", 16999)


# ------------------------------------------------------------------------------------- one call: read-only, parameterized, timed

def test_every_registered_text_runs_as_ro_query_with_the_timeout_and_parameters_only():
    for name, text in EXPECTED.items():
        db = FakeDb()
        r = CypherReader(db, "shop")
        params = _params_for(text)
        r.read(name, params)
        calls = db.ro_calls()
        assert calls, name
        for _, key, sent, sent_params, timeout in calls:
            assert key == "ccg_shop" and sent == text and timeout == config.QUERY_TIMEOUT_MS, name
            assert set(sent_params) == set(re.findall(r"\$([a-z]+)", text)), name


def test_the_timeout_is_the_readers_and_a_changed_one_is_sent():
    db = FakeDb()
    CypherReader(db, "shop", timeout_ms=1234).read("repo_exists", {"repo": "r"})
    assert db.ro_calls()[0][4] == 1234
    for bad in (0, -1, 1.5, True, "5"):
        with pytest.raises(ReaderError):
            CypherReader(FakeDb(), "shop", timeout_ms=bad)


def test_a_hostile_value_is_a_parameter_never_part_of_the_text():
    db = FakeDb()
    evil = "x'}) DETACH DELETE (n) //"
    CypherReader(db, "shop").read("file_by_path", {"repo": evil, "path": "a\"); MATCH (m) DELETE m //"})
    (_, _, text, params, _), = db.ro_calls()
    assert text == EXPECTED["file_by_path"] and params == {"repo": evil, "path": "a\"); MATCH (m) DELETE m //"}


def test_missing_extra_and_mistyped_parameters_are_refused_before_any_call():
    db = FakeDb()
    r = CypherReader(db, "shop")
    for name, params in (("file_by_path", {"repo": "r"}), ("file_by_path", {"repo": "r", "path": "p", "extra": "x"}), ("file_by_path", {"repo": 1, "path": "p"}),
                         ("file_by_path", {"repo": "r", "path": None}), ("calls_in", {"ids": "abc"}), ("calls_in", {"ids": [1, 2]}), ("calls_in", {"ids": [None]}),
                         ("lifted_of", {"pairs": [["a"]]}), ("lifted_of", {"pairs": [["a", "b", "c"]]}), ("lifted_of", {"pairs": ["ab"]}), ("lifted_of", {"pairs": [[1, 2]]}),
                         ("calls_in", {}), ("calls_in", None)):
        with pytest.raises(ReaderError):
            r.read(name, params)
    assert db.calls == []


def test_a_server_error_is_a_reader_error_and_a_timeout_is_a_read_timeout():
    class ResponseError(Exception):
        pass

    r = CypherReader(FakeDb({EXPECTED["repo_exists"]: ResponseError("Query timed out")}), "shop")
    with pytest.raises(ReadTimeout):
        r.read("repo_exists", {"repo": "r"})
    r = CypherReader(FakeDb({EXPECTED["repo_exists"]: ResponseError("The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value")}), "shop")
    with pytest.raises(ReaderError, match="TIMEOUT_MAX") as ei:
        r.read("repo_exists", {"repo": "r"})
    assert not isinstance(ei.value, ReadTimeout)
    r = CypherReader(FakeDb({EXPECTED["repo_exists"]: ResponseError("NOPERM x")}), "shop")
    with pytest.raises(ReaderError, match="repo_exists"):
        r.read("repo_exists", {"repo": "r"})


# ------------------------------------------------------------------------------------- chunking and bounded replies

def _ids(n):
    return [f"id{i:05d}" for i in range(n)]


def test_ids_are_chunked_at_1000_and_every_chunk_is_a_separate_call():
    db = FakeDb({EXPECTED["calls_in"]: lambda p: [[i] + [None] * 8 for i in p["ids"]]})
    rows = CypherReader(db, "shop").read("calls_in", {"ids": _ids(2500)})
    assert [len(c[3]["ids"]) for c in db.ro_calls()] == [1000, 1000, 500]
    assert len(rows) == 2500


def test_a_full_reply_halves_the_chunk_and_the_halved_size_sticks():
    full = config.RESULTSET_SIZE
    sizes = []

    def reply(p):
        sizes.append(len(p["ids"]))
        return [[p["ids"][0]] + [None] * 8] * (full if len(p["ids"]) > 300 else len(p["ids"]))

    db = FakeDb({EXPECTED["calls_in"]: reply})
    CypherReader(db, "shop").read("calls_in", {"ids": _ids(1600)})
    assert sizes == [1000, 500, 250, 250, 250, 250, 250, 250, 100]


def test_a_reply_that_stays_full_at_one_id_raises_and_never_truncates():
    full = [["id"] + [None] * 8] * config.RESULTSET_SIZE
    db = FakeDb({EXPECTED["calls_in"]: lambda p: full})
    with pytest.raises(ResultTooLarge, match="calls_in"):
        CypherReader(db, "shop").read("calls_in", {"ids": ["a", "b", "c"]})
    assert [len(c[3]["ids"]) for c in db.ro_calls()] == [3, 1]


def test_a_one_row_below_the_limit_is_returned_whole():
    n = config.RESULTSET_SIZE - 1
    db = FakeDb({EXPECTED["calls_in"]: lambda p: [[f"r{i}"] + [None] * 8 for i in range(n)]})
    assert len(CypherReader(db, "shop").read("calls_in", {"ids": ["a"]})) == n


def test_an_anchor_read_with_a_full_reply_raises_there_is_nothing_to_halve():
    db = FakeDb({EXPECTED["cp_keys"]: lambda p: [["c", "k"]] * config.RESULTSET_SIZE})
    with pytest.raises(ResultTooLarge):
        CypherReader(db, "shop").read("cp_keys", {"kind": "http"})


def test_pairs_chunk_like_ids():
    db = FakeDb()
    CypherReader(db, "shop").read("lifted_of", {"pairs": [[f"l{i}", f"s{i}"] for i in range(2100)]})
    assert [len(c[3]["pairs"]) for c in db.ro_calls()] == [1000, 1000, 100]


def test_an_empty_id_list_makes_no_call():
    db = FakeDb()
    assert CypherReader(db, "shop").read("calls_in", {"ids": []}) == [] and db.calls == []


# ------------------------------------------------------------------------------------- rows: deterministic, typed, untrusted

def test_rows_come_back_in_a_total_order_whatever_the_server_sent_and_none_sorts_last():
    rows = [["b", None, 2], ["a", 3, 1], ["b", 1, 1], ["a", None, 0], ["a", 3, 0]]
    db = FakeDb({EXPECTED["repo_exists"]: rows})
    got = CypherReader(db, "shop").read("repo_exists", {"repo": "r"})
    assert got == [("a", 3, 0), ("a", 3, 1), ("a", None, 0), ("b", 1, 1), ("b", None, 2)]
    db2 = FakeDb({EXPECTED["repo_exists"]: list(reversed(rows))})
    assert CypherReader(db2, "shop").read("repo_exists", {"repo": "r"}) == got


def test_a_list_cell_is_a_tuple_so_a_row_is_hashable_and_sortable():
    db = FakeDb({EXPECTED["repo_exists"]: [["b", [3, 4]], ["a", [1, 2, 3, 4]], ["a", None], ["a", []]]})
    got = CypherReader(db, "shop").read("repo_exists", {"repo": "r"})
    assert got == [("a", ()), ("a", (1, 2, 3, 4)), ("a", None), ("b", (3, 4))] and len(set(got)) == 4


def test_the_recipe_lists_every_text_run_once_by_name_sorted_and_deduplicated():
    db = FakeDb()
    r = CypherReader(db, "shop")
    r.read("to_in", {"ids": ["a"]})
    r.read("calls_in", {"ids": ["a"]})
    r.read("to_in", {"ids": ["b"]})
    r.read("owner_of:Ret", {"ids": ["a"]})
    r.read("calls_in", {"ids": []})                                 # no call was made: not recorded
    assert r.recipe_queries() == [{"name": n, "cypher": EXPECTED[n]} for n in ("calls_in", "owner_of:Ret", "to_in")]
    assert CypherReader(FakeDb(), "shop").recipe_queries() == []


def test_a_query_that_failed_is_still_recorded_as_run_but_one_never_sent_is_not():
    db = FakeDb({EXPECTED["repo_exists"]: RuntimeError("boom")})
    r = CypherReader(db, "shop")
    with pytest.raises(ReaderError):
        r.read("repo_exists", {"repo": "r"})
    with pytest.raises(ReaderError):
        r.read("evil", {})
    assert [q["name"] for q in r.recipe_queries()] == ["repo_exists"]


# ------------------------------------------------------------------------------------- bound_of_link: the first-row pick (-answer-spec2 §3.3)

def _bol(i, entry, site, ord_, keypath, launch=None, port=None, origin=None, pack=None):
    return [i, entry, site, ord_, "repo", "unit", "cfg/x.json", 3, keypath, launch, port, origin, pack]


def test_bound_of_link_picks_per_link_and_entry_the_lowest_site_then_ord_row_with_a_keypath():
    rows = [
        _bol("L1", "E1", "S2", 0, "a.b", port=8000, origin="explicit"),
        _bol("L1", "E1", "S1", 1, "a.b", port=9000, origin="compose"),
        _bol("L1", "E1", "S1", 0, "a.b", port=7000, origin="pack-default", pack="fastapi"),
        _bol("L1", "E2", "S1", 0, None, launch="command"),            # a launchFact-origin entry: never an item
        _bol("L2", "E1", "S9", 4, "z"),
    ]
    db = FakeDb({EXPECTED["bound_of_link"]: lambda p: rows})
    got = CypherReader(db, "shop").bound_of_link(["L1", "L2"])
    assert [(g.link_id, g.entry_id, g.site_id, g.ord, g.port, g.port_origin, g.pack_id) for g in got] == [
        ("L1", "E1", "S1", 0, 7000, "pack-default", "fastapi"), ("L2", "E1", "S9", 4, None, None, None)]
    assert all(g.key_path for g in got)


def test_an_entry_that_carries_both_a_keypath_and_a_launch_kind_is_not_an_item():
    rows = [_bol("L1", "E1", "S1", 0, "a.b", launch="command"), _bol("L1", "E2", "S1", 1, "c.d"), _bol("L1", "E3", "S1", 2, None)]
    got = CypherReader(FakeDb({EXPECTED["bound_of_link"]: lambda p: rows}), "shop").bound_of_link(["L1"])
    assert [g.entry_id for g in got] == ["E2"]


def test_bound_of_link_reads_nothing_it_was_not_asked_for_and_rejects_a_malformed_row():
    for bad in ([["L1", "E1"]], [_bol("L1", "E1", "S1", 0, "a") + ["extra"]], [_bol("L1", "E1", "S1", "0", "a")], [_bol("L1", None, "S1", 0, "a")]):
        db = FakeDb({EXPECTED["bound_of_link"]: bad})
        with pytest.raises(GraphDataError, match="bound_of_link"):
            CypherReader(db, "shop").bound_of_link(["L1"])
    assert CypherReader(FakeDb(), "shop").bound_of_link([]) == []
    with pytest.raises(ReaderError):
        CypherReader(FakeDb(), "shop").bound_of_link("L1")


# ------------------------------------------------------------------------------------- an error names no graph data (the store is untrusted text)

SENTINEL = "SENTINEL-GRAPH-DATA-7f3a"


def _clean(err: BaseException) -> None:
    """The message, the repr and the displayed chain carry no sentinel: ``from None`` (no ``__cause__``), so a traceback does not echo server text either."""
    assert SENTINEL not in str(err) and SENTINEL not in repr(err), (str(err), repr(err))
    assert err.__cause__ is None and err.__suppress_context__ is True


class _Leaky:
    """A cell value of a type the reader does not accept, whose repr and str leak."""
    def __repr__(self):
        return SENTINEL
    __str__ = __repr__


def _server_error(text):
    class ResponseError(Exception):
        pass
    return ResponseError(text)


def test_a_malformed_cell_names_the_query_and_the_type_never_the_value():
    db = FakeDb({EXPECTED["repo_exists"]: [[_Leaky()]]})
    with pytest.raises(GraphDataError, match="repo_exists") as ei:
        CypherReader(db, "shop").read("repo_exists", {"repo": "r"})
    assert SENTINEL not in str(ei.value) and SENTINEL not in repr(ei.value)


def test_a_malformed_bound_of_link_row_carries_no_row_text():
    rows = [["L1", SENTINEL, "S1", "0", "a", None, None, None, "k", None, None, None, None]]
    with pytest.raises(GraphDataError) as ei:
        CypherReader(FakeDb({EXPECTED["bound_of_link"]: rows}), "shop").bound_of_link(["L1"])
    assert SENTINEL not in str(ei.value) and SENTINEL not in repr(ei.value)
    with pytest.raises(GraphDataError) as ei:
        CypherReader(FakeDb({EXPECTED["bound_of_link"]: [["L1", SENTINEL]]}), "shop").bound_of_link(["L1"])
    assert SENTINEL not in str(ei.value) and SENTINEL not in repr(ei.value)


@pytest.mark.parametrize("server_text", [
    f"Type mismatch: expected Integer but was String {SENTINEL}",
    f"Query timed out {SENTINEL}",
    f"The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value {SENTINEL}",
])
def test_a_server_error_carries_no_server_text_in_the_message_or_the_chain(server_text):
    r = CypherReader(FakeDb({EXPECTED["repo_exists"]: _server_error(server_text)}), "shop")
    with pytest.raises(ReaderError) as ei:
        r.read("repo_exists", {"repo": "r"})
    _clean(ei.value)


def test_a_result_too_large_names_the_query_and_the_limit_never_a_row(monkeypatch):
    monkeypatch.setattr(config, "RESULTSET_SIZE", 3)
    db = FakeDb({EXPECTED["calls_in"]: lambda p: [[SENTINEL] + [SENTINEL] * 8] * 3})
    with pytest.raises(ResultTooLarge) as ei:
        CypherReader(db, "shop").read("calls_in", {"ids": ["a"]})
    assert SENTINEL not in str(ei.value) and SENTINEL not in repr(ei.value)
    db = FakeDb({EXPECTED["cp_keys"]: lambda p: [[SENTINEL, SENTINEL]] * 3})
    with pytest.raises(ResultTooLarge) as ei:
        CypherReader(db, "shop").read("cp_keys", {"kind": "http"})
    assert SENTINEL not in str(ei.value) and SENTINEL not in repr(ei.value)


def test_a_caller_hostile_parameter_value_is_not_echoed_in_a_shape_error():
    r = CypherReader(FakeDb(), "shop")
    for name, params in (("repo_exists", {"repo": 5, "x": SENTINEL}), ("calls_in", {"ids": [SENTINEL, 1]}), ("repo_exists", {SENTINEL: "x"})):
        with pytest.raises(ReaderError) as ei:
            r.read(name, params)
        assert SENTINEL not in str(ei.value)


def test_the_resultset_size_constant_is_the_one_the_infra_status_check_compares_against():
    from pathlib import Path
    script = (Path(__file__).resolve().parent.parent / "infra" / "ccg-falkordb.sh").read_text()
    m = re.search(r"^RESULTSET_SIZE_IMAGE=(\d+)$", script, re.M)
    assert m and int(m.group(1)) == config.RESULTSET_SIZE
