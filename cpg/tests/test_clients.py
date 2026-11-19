"""P1-25a: the shared client code and the CLI (``docs/plans/connected-code-graph-p1-answers.md`` P1-25 behaviours (1)-(5), (8), (9); ``-answer-spec2.md`` §2, §5.1, §9 Q-1).

Offline: a real engine over an in-memory reader (``tests/engine_answer_helpers.AnswerReader``) for ``ask``/``resolve``, and a fake falkordb-py client for ``raw_query`` and
``refresh``. The database-side half of AC-21 (the server refusing a write, ``GRAPH.QUERY`` and ``GRAPH.PROFILE`` for the raw role) is live and P1-42's; here the unit
asserts what the client controls: ``GRAPH.RO_QUERY`` only, the ``ccg_raw_<system>`` credential only, the time limit on every call, and that a refusal is typed.

    cd cpg && .venv/bin/python -m pytest tests/test_clients.py
"""
from __future__ import annotations

import copy
import io
import json
import os
import pwd
from types import SimpleNamespace

import pytest
import test_engine_answer as TA
from engine_answer_helpers import AnswerReader
from test_answer import b_12, b_flow, b_notice, golden

from ccg import answer, atrest, sysname
from ccg import refresh as refresh_mod
from ccg.engine import answer_build as AB
from ccg.engine import anchors as A
from ccg.engine import config
from ccg.render import render_markdown, render_text
from ccg.render.common import flow_notice_lines
from ccg.render.common import all_items
from clients import common
from clients.cli import main as cli
from clients.common import (AnchorNotResolved, BadRequest, Client, NoCredentials, NotBuilt, RawRefused, RefreshFailed, StoreError)

SYSTEM = "shop"
SOCKET_TIMEOUTS: list[float] = []
PASSWORDS = {"build": "pw-build", "verdict": "pw-verdict", "engine": "pw-engine", "raw": "pw-raw"}


# ------------------------------------------------------------------------------------------------------------------ fixtures

class ClientReader(AnswerReader):
    """``AnswerReader`` plus the resolver's ``repo_exists`` read; ``client_over`` makes a fresh one per call, as the live client opens a reader per call."""

    def __init__(self, g, snapshot_rows=None):
        super().__init__(g, snapshot_rows)
        self.g = g

    def _x_repo_exists(self, repo):
        return [(repo,)] if any(n.get("repoId") == repo for nodes in self._nodes.values() for n in nodes.values()) else []


def world(flow: bool):
    g, ids_ = TA.caller_world(flow=flow)
    return g, ids_, ClientReader(g)


def req(g, qname="m.py::A", use_case="impact"):
    return {"useCase": use_case, "anchor": {"kind": "symbol", "repo": g.repo, "qname": qname}}


def client_over(reader, **kw):
    return Client(SYSTEM, host="127.0.0.1", port=16399, reader_factory=lambda system, host, port: ClientReader(reader.g), **kw)


def write_credentials(system=SYSTEM, passwords=PASSWORDS):
    lines = "".join(f"{sysname.acl_user(role, system)} {pw}\n" for role, pw in passwords.items())
    atrest.write_acl(system, lines.encode())


class FakeGraph:
    """What the raw role may do: ``ro_query``. A write is refused the way the server refuses it; ``query``/``profile`` must never be called."""

    def __init__(self, db):
        self.db = db

    def ro_query(self, text, params=None, timeout=None):
        self.db.calls.append(("ro_query", text, params, timeout))
        if timeout is not None and timeout > self.db.timeout_max:
            raise RuntimeError("The query TIMEOUT parameter value cannot exceed the TIMEOUT_MAX configuration parameter value")
        if any(w in text.upper() for w in ("CREATE", "DELETE", "SET ", "MERGE", "REMOVE")):
            raise RuntimeError("graph.RO_QUERY is to be executed only on read-only queries")
        return SimpleNamespace(header=[[1, "n"], [1, "m"]], result_set=[[1, "x"], [2, FakeNode()]])

    def query(self, *a, **k):
        self.db.forbidden.append("GRAPH.QUERY")
        self.db.stamp = "changed"
        raise AssertionError("raw_query must never call GRAPH.QUERY")

    def profile(self, *a, **k):
        self.db.forbidden.append("GRAPH.PROFILE")
        raise AssertionError("raw_query must never call GRAPH.PROFILE")


class FakeNode:
    labels = ["Symbol"]
    properties = {"qname": "m.py::A"}


class FakeDB:
    def __init__(self, timeout_max=10_000):
        self.calls, self.forbidden, self.keys, self.stamp, self.timeout_max = [], [], [], "s0", timeout_max

    def select_graph(self, key):
        self.keys.append(key)
        return FakeGraph(self)


def raw_client(db, system=SYSTEM, opened=None, **kw):
    def connect(host, port, *, username, password, socket_timeout=None):
        (opened if opened is not None else []).append((host, port, username, password))
        if socket_timeout is not None:
            SOCKET_TIMEOUTS.append(socket_timeout)
        return db
    return Client(system, host="127.0.0.1", port=16399, raw_connect=connect, **kw)


# ------------------------------------------------------------------------------------------------------------------ (1) ask, structured

def test_a_structured_impact_ask_returns_the_engines_canonical_document_unchanged():
    g, ids_, reader = world(flow=True)
    doc = client_over(reader).ask(req(g))
    expected = AB.build_answer(ClientReader(g), A.resolve_symbol(ClientReader(g), g.repo, "m.py::A").anchors[0])
    assert answer.dumps(doc) == answer.dumps(expected)
    assert [i["subject"]["symbolId"] for i in doc["items"]] == [ids_["C"]]
    assert doc["interpretation"]["request"]["direction"] == "impact"
    answer.validate(doc)


def test_depends_on_asks_the_other_column():
    g, ids_ = TA.depends_world_g(), None
    g, ids_ = TA.hexify(g)
    reader = ClientReader(g)
    qname = next(s["qname"] for s in g.rows.nodes["Symbol"] if s["symbolId"] == ids_["getHealth"])
    doc = client_over(reader).ask(req(g, qname, "depends-on"))
    assert doc["interpretation"]["request"]["direction"] == "depends-on"
    assert doc["items"] or doc["groups"]


def test_the_use_case_is_not_silently_defaulted_or_widened():
    g, _, reader = world(flow=True)
    c = client_over(reader)
    for bad_case in ("test-gap", "rca", "change-review", ""):
        with pytest.raises(NotBuilt) as e:
            c.ask({**req(g), "useCase": bad_case})
        assert e.value.code == "not-built"
        assert "impact" in e.value.detail["built"]
    with pytest.raises(BadRequest):
        c.ask({"anchor": req(g)["anchor"]})                         # no use case
    with pytest.raises(NotBuilt):
        c.ask({**req(g), "scope": {"repoIds": [g.repo]}})            # a scope is not built: never ignored
    with pytest.raises(BadRequest):
        c.ask({**req(g), "depth": 5})                                # no depth bound field (spec2 §4.1)
    with pytest.raises(BadRequest):
        c.ask({**req(g), "anchor": {"kind": "symbol", "repo": g.repo}})            # missing qname
    with pytest.raises(BadRequest):
        c.ask({**req(g), "anchor": {"kind": "symbol", "repo": g.repo, "qname": "m.py::A", "x": 1}})
    with pytest.raises(BadRequest):
        c.ask({**req(g), "anchor": {"kind": "nope"}})
    with pytest.raises(BadRequest):
        c.ask("impact of A")                                        # a bare string is not a structured request
    with pytest.raises(BadRequest):
        c.ask()                                                     # neither
    with pytest.raises(BadRequest):
        c.ask(req(g), question="who calls A")                       # both


def test_an_anchor_that_is_not_one_candidate_is_a_typed_error_with_the_candidates(monkeypatch):
    g, ids_, reader = world(flow=True)
    c = client_over(reader)
    with pytest.raises(AnchorNotResolved) as e:
        c.ask(req(g, "m.py::NoSuch"))
    assert e.value.code == "anchor-unresolved" and e.value.detail["status"] == "unresolved"
    a1 = A.SymbolAnchor("1" * 32, "m.py::A", g.repo, "m.py", "u1", 3, 9)
    a2 = A.SymbolAnchor("2" * 32, "m.py::A", g.repo, "n.py", "u1", 4, 8)
    monkeypatch.setattr(common.A, "resolve_symbol", lambda reader, repo, qname: A.Resolution("ambiguous", options=(a1, a2), options_total=2))
    with pytest.raises(AnchorNotResolved) as e:
        c.ask(req(g))
    assert e.value.code == "anchor-ambiguous"
    assert [x["symbolId"] for x in e.value.detail["candidates"]] == ["1" * 32, "2" * 32]


def test_cache_bypass_has_no_effect_on_a_structured_request_and_is_not_recorded():
    g, _, reader = world(flow=True)
    a = client_over(reader).ask(req(g))
    b = client_over(ClientReader(g)).ask(req(g), cache_bypass=True)
    assert answer.dumps(a) == answer.dumps(b)
    assert "provenance" not in b["interpretation"]
    with pytest.raises(BadRequest):
        client_over(reader).ask(req(g), cache_bypass="yes")


# ------------------------------------------------------------------------------------------------------------------ (1) ask, plain language: the seam

def test_a_question_without_a_front_is_a_typed_not_built_error_and_reads_nothing():
    calls = []
    c = Client(SYSTEM, host="h", port=16399, reader_factory=lambda *a: calls.append(a))
    with pytest.raises(NotBuilt) as e:
        c.ask(question="who calls A?")
    assert e.value.detail["surface"] == "plain-language-question" and calls == []
    for blank in ("", "   "):
        with pytest.raises(BadRequest):
            c.ask(question=blank)


def test_a_question_goes_through_the_front_seam_with_the_bypass_and_its_answer_is_validated():
    g, _, reader = world(flow=True)
    seen = []

    def front(question, *, reader, cache_bypass):
        seen.append((question, cache_bypass))
        return b_12()

    doc = client_over(reader, front=front).ask(question="who calls A?", cache_bypass=True)
    assert seen == [("who calls A?", True)] and doc == b_12()
    bad = b_12()
    bad["items"][0]["status"] = "maybe"
    with pytest.raises(StoreError):
        client_over(reader, front=lambda q, **kw: bad).ask(question="x")


# ------------------------------------------------------------------------------------------------------------------ (8) validate first

def test_an_invalid_answer_from_the_builder_never_leaves_ask():
    g, _, reader = world(flow=True)
    bad = b_12()
    bad["answerSchemaVersion"] = "9.9"
    with pytest.raises(StoreError):
        client_over(reader, builder=lambda *a, **k: bad).ask(req(g))


def test_render_answer_validates_before_any_renderer_runs(monkeypatch):
    spy = []
    monkeypatch.setattr(common, "render_text", lambda d: spy.append("text") or "T")
    monkeypatch.setattr(common, "render_markdown", lambda d: spy.append("markdown") or "M")
    bad = b_12()
    bad["items"][0]["chain"][0]["state"] = "nowhere"
    for fmt in ("json", "text", "markdown"):
        with pytest.raises(answer.AnswerError):
            common.render_answer(bad, fmt)
    assert spy == []
    assert common.render_answer(b_12(), "text") == "T" and spy == ["text"]
    with pytest.raises(BadRequest):
        common.render_answer(b_12(), "html")


# ------------------------------------------------------------------------------------------------------------------ (9) the flow notice

def test_the_notice_is_in_the_returned_document_and_in_every_form():
    g, _, reader = world(flow=False)
    doc = client_over(reader).ask(req(g))
    assert doc["answerSchemaVersion"] == "1.2" and doc["notices"] == [{"code": "flow-not-enumerated", "langs": ["python"]}]
    assert json.loads(common.render_answer(doc, "json"))["notices"] == doc["notices"]
    lines = flow_notice_lines(doc)
    assert lines and all(line in common.render_answer(doc, "text") for line in lines)
    assert common.render_answer(doc, "text") == render_text(doc) and common.render_answer(doc, "markdown") == render_markdown(doc)
    _, _, flow_true = world(flow=True)
    assert "notices" not in client_over(flow_true).ask(req(g))


def run_cli(argv, client):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, client_factory=lambda system: client, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_the_cli_prints_the_canonical_document_with_its_notice_intact():
    g, _, reader = world(flow=False)
    code, out, err = run_cli(["ask", "--request", json.dumps(req(g))], client_over(reader))
    assert (code, err) == (0, "")
    doc = json.loads(out)
    assert doc["notices"] == [{"code": "flow-not-enumerated", "langs": ["python"]}]
    assert out.encode() == answer.dumps(doc) and answer.is_canonical(out.encode())


def test_the_cli_text_and_markdown_modes_are_the_renderers_and_carry_the_notice():
    g, _, reader = world(flow=False)
    doc = client_over(reader).ask(req(g))
    for fmt, render in (("text", render_text), ("markdown", render_markdown)):
        code, out, _ = run_cli(["ask", "--format", fmt, "--request", json.dumps(req(g))], client_over(ClientReader(g)))
        assert code == 0 and out == render(doc)
        assert all(line in out for line in flow_notice_lines(doc))


class DocClient:
    """A client whose ``ask`` returns a given document (the CLI's exit codes and rendering do not depend on how it was made)."""

    def __init__(self, doc):
        self.doc, self.asked = doc, []

    def ask(self, request=None, *, question=None, cache_bypass=False):
        self.asked.append((request, question, cache_bypass))
        return self.doc


def zero_item_notice():
    d = b_notice()
    d["items"], d["groups"], d["staleness"]["units"] = [], [], []
    return d


def confirmed_only():
    d = golden()
    assert d["items"] and all(i["status"] == "confirmed" for i in all_items(d))
    return d


# (document, expected exit for --fail-on possible, for --fail-on any); written out by hand from the documents' items
FAIL_ON_CASES = [("b_notice (possible items, notice)", b_notice, 1, 1), ("b_flow (the same items, no notice)", b_flow, 1, 1),
                 ("a notice and no item", zero_item_notice, 0, 0), ("confirmed items only", confirmed_only, 0, 1)]


@pytest.mark.parametrize("name,make,on_possible,on_any", FAIL_ON_CASES, ids=[c[0] for c in FAIL_ON_CASES])
def test_fail_on_exit_codes_follow_the_items_only_and_a_notice_changes_none(name, make, on_possible, on_any):
    doc = make()
    answer.validate(doc)
    for level, want in (("possible", on_possible), ("any", on_any)):
        code, out, _ = run_cli(["ask", "--request", "{}", "--fail-on", level], DocClient(doc))
        assert code == want, (name, level)
        assert out.encode() == answer.dumps(doc)                     # printed whether or not it trips, notice included
    code, _, _ = run_cli(["ask", "--request", "{}"], DocClient(doc))
    assert code == 0                                               # no --fail-on: never 1


def test_a_notice_alone_does_not_trip_any_and_removing_it_does_not_change_a_code():
    with_notice, without = zero_item_notice(), zero_item_notice()
    del without["notices"]
    assert not cli.fail_on_tripped(with_notice, "any") and not cli.fail_on_tripped(without, "any")
    possible = b_notice()
    stripped = copy.deepcopy(possible)
    del stripped["notices"]
    for level in ("possible", "any"):
        assert cli.fail_on_tripped(possible, level) == cli.fail_on_tripped(stripped, level) is True
    with pytest.raises(BadRequest):
        cli.fail_on_tripped(possible, "all")


def test_a_group_member_counts_for_fail_on():
    d = {"interpretation": {"status": "resolved"}, "items": [], "groups": [{"members": [{"status": "confirmed"}]}]}
    assert cli.fail_on_tripped(d, "any") and not cli.fail_on_tripped(d, "possible")
    d["groups"][0]["members"].append({"status": "possible"})
    assert cli.fail_on_tripped(d, "possible")


def test_the_cli_validates_before_it_renders_or_prints():
    bad = b_12()
    bad["items"][0]["status"] = "maybe"
    for argv in (["--format", "text"], ["--format", "markdown"], []):
        code, out, err = run_cli(["ask", "--request", "{}", *argv], DocClient(bad))
        assert code == 2 and out == ""
        assert json.loads(err)["error"]["code"] == "invalid-answer"


def test_the_cli_passes_the_question_and_the_bypass_flag_and_a_structured_request_unchanged():
    d = DocClient(b_12())
    run_cli(["ask", "--question", "who calls A?", "--cache-bypass"], d)
    run_cli(["ask", "--request", '{"useCase": "impact", "anchor": {"kind": "name", "name": "A"}}'], d)
    assert d.asked == [(None, "who calls A?", True), ({"useCase": "impact", "anchor": {"kind": "name", "name": "A"}}, None, False)]


def test_the_cli_reports_typed_errors_on_stderr_with_exit_2_and_nothing_on_stdout():
    g, _, reader = world(flow=True)
    for argv, code_name in ((["ask", "--question", "x"], "not-built"),
                            (["ask", "--request", json.dumps({**req(g), "useCase": "rca"})], "not-built"),
                            (["ask", "--request", "not json"], "bad-request"),
                            (["ask", "--request", json.dumps(req(g, "m.py::NoSuch"))], "anchor-unresolved")):
        code, out, err = run_cli(argv, client_over(reader))
        assert (code, out) == (2, "") and json.loads(err)["error"]["code"] == code_name, argv


def test_a_usage_error_exits_2():
    with pytest.raises(SystemExit) as e:
        cli.main(["ask"])                                            # neither --request nor --question
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        cli.main(["ask", "--request", "{}", "--fail-on", "sometimes"])


# ------------------------------------------------------------------------------------------------------------------ (2) resolve

def test_resolve_lists_the_candidates_with_their_ids():
    g, ids_, reader = world(flow=True)
    out = client_over(reader).resolve({"kind": "symbol", "repo": g.repo, "qname": "m.py::A"})
    assert out["status"] == "resolved" and out["via"] == "qname"
    (cand,) = out["candidates"]
    assert cand["type"] == "symbol" and cand["symbolId"] == ids_["A"] and cand["repoId"] == g.repo and cand["unitId"] and cand["lineStart"] == 3
    none = client_over(reader).resolve({"kind": "symbol", "repo": g.repo, "qname": "m.py::NoSuch"})
    assert none["status"] == "unresolved" and none["candidates"] == [] and none["detail"] == "no-such-symbol"


def test_resolve_finds_a_contract_point_and_a_config_entry_by_their_keys():
    g, ids_ = TA.hexify(TA.link_world_g())
    reader = ClientReader(g)
    cp = TA.cp_anchor(g, ids_, "cp1")
    out = client_over(reader).resolve({"kind": "contract", "key": cp.key_norm})
    assert [c["cpId"] for c in out["candidates"]] == [cp.cp_id] and out["candidates"][0]["type"] == "contract"
    cfg = TA.cfg_anchor(g, ids_, "e1")
    out = client_over(reader).resolve({"kind": "config", "repo": cfg.repo_id, "path": cfg.path, "keyPath": cfg.key_path})
    assert [c["entryId"] for c in out["candidates"]] == [cfg.entry_id] and out["candidates"][0]["keyPath"] == cfg.key_path


def test_resolve_rejects_an_unusable_anchor_and_the_cli_prints_the_candidates():
    g, ids_, reader = world(flow=True)
    with pytest.raises(BadRequest):
        client_over(reader).resolve({"kind": "symbol", "repo": "BAD REPO", "qname": "x"})
    code, out, _ = run_cli(["resolve", "--anchor", json.dumps({"kind": "symbol", "repo": g.repo, "qname": "m.py::A"})], client_over(reader))
    assert code == 0 and json.loads(out)["candidates"][0]["symbolId"] == ids_["A"]


# ------------------------------------------------------------------------------------------------------------------ (3) refresh

def test_refresh_calls_p1_20_as_the_build_and_verdict_users_with_the_scope(monkeypatch):
    write_credentials()
    opened, got = [], {}

    def connect(host, port, *, username, password):
        opened.append((host, port, username, password))
        return SimpleNamespace(role=username)

    def fake_refresh(system, scope=None, override=None, *, target=None, **kw):
        got.update(system=system, scope=scope, override=override, target=target)
        return refresh_mod.RefreshReport(system, scope=tuple(scope) if scope else None, snapshot_id="s1",
                                         units=[refresh_mod.UnitReport("u1", "repo-a", "svc", "python", "fresh")])

    monkeypatch.setattr(refresh_mod, "refresh", fake_refresh)
    out = Client(SYSTEM, host="127.0.0.1", port=16399, connect=connect).refresh(["repo-a"])
    assert opened == [("127.0.0.1", 16399, "ccg_build_shop", "pw-build"), ("127.0.0.1", 16399, "ccg_verdict_shop", "pw-verdict")]
    assert (got["system"], got["scope"], got["override"]) == (SYSTEM, ["repo-a"], None)
    assert got["target"].build_conn.role == "ccg_build_shop" and got["target"].verdict_conn.role == "ccg_verdict_shop" and got["target"].store.system == SYSTEM
    assert out["snapshotId"] == "s1" and out["scope"] == ["repo-a"] and out["units"][0]["status"] == "fresh" and out["report"][0].startswith("refresh shop:")


def test_refresh_failures_and_bad_scopes_are_typed(monkeypatch):
    write_credentials()
    c = Client(SYSTEM, host="h", port=16399, connect=lambda *a, **k: object())
    with pytest.raises(BadRequest):
        c.refresh("repo-a")                                          # a bare string, not a list
    monkeypatch.setattr(refresh_mod, "refresh", lambda *a, **k: (_ for _ in ()).throw(refresh_mod.RefreshError("no unit left")))
    with pytest.raises(RefreshFailed):
        c.refresh()


# ------------------------------------------------------------------------------------------------------------------ (4), (5) raw_query

def test_raw_query_runs_ro_query_as_the_raw_credential_on_the_live_graph_with_a_time_limit():
    write_credentials()
    db, opened = FakeDB(), []
    out = raw_client(db, opened=opened).raw_query("MATCH (s:Symbol) RETURN s.qname, s LIMIT 2")
    assert opened == [("127.0.0.1", 16399, "ccg_raw_shop", "pw-raw")]            # the raw user and its password, not engine's or build's
    assert db.keys == ["ccg_shop"]                                              # the live graph
    assert db.calls == [("ro_query", "MATCH (s:Symbol) RETURN s.qname, s LIMIT 2", None, config.QUERY_TIMEOUT_MS)]
    assert db.forbidden == []
    assert out["columns"] == ["n", "m"] and out["rows"] == [[1, "x"], [2, {"labels": ["Symbol"], "properties": {"qname": "m.py::A"}}]]
    assert out["timeoutMs"] == config.QUERY_TIMEOUT_MS and out["system"] == SYSTEM


def test_the_time_limit_is_configurable_sent_every_call_and_the_servers_timeout_max_refuses_a_larger_one():
    write_credentials()
    db = FakeDB(timeout_max=5_000)
    c = raw_client(db)
    c.raw_query("RETURN 1", timeout_ms=5_000)
    assert db.calls[-1][3] == 5_000
    with pytest.raises(RawRefused) as e:
        c.raw_query("RETURN 1", timeout_ms=5_001)
    assert "TIMEOUT_MAX" in str(e.value)
    with pytest.raises(RawRefused):
        c.raw_query("RETURN 1")                                              # the 10,000 default is above this instance's TIMEOUT_MAX
    for bad in (0, -1, True, 1.5, "10"):
        with pytest.raises(BadRequest):
            c.raw_query("RETURN 1", timeout_ms=bad)


def test_a_write_is_refused_by_the_database_typed_and_the_stamp_is_unchanged():
    write_credentials()
    db, opened = FakeDB(), []
    c = raw_client(db, opened=opened)
    for text in ("CREATE (n:Symbol {qname: 'x'})", "MATCH (s:Snapshot) SET s.snapshotId = 'z'", "MATCH (n) DELETE n"):
        with pytest.raises(RawRefused) as e:
            c.raw_query(text)
        assert e.value.code == "raw-refused"
    assert len(opened) == 3 and {o[2:] for o in opened} == {("ccg_raw_shop", "pw-raw")}      # one connection per call, always the raw role: no retry as another role
    assert db.stamp == "s0" and db.forbidden == []                           # GRAPH.QUERY / GRAPH.PROFILE never reached
    assert {call[0] for call in db.calls} == {"ro_query"}
    assert not [m for m in dir(Client) if m.lower() in ("query", "profile", "graph_query", "graph_profile")]


def test_raw_query_does_not_alter_the_text_or_pass_params():
    write_credentials()
    db = FakeDB()
    text = "\n MATCH (s:Symbol)\nWHERE s.qname = 'a'   RETURN s \n"
    raw_client(db).raw_query(text)
    assert db.calls[0][1:3] == (text, None)
    for bad in ("", "   ", None, 5):
        with pytest.raises(BadRequest):
            raw_client(db).raw_query(bad)


def test_the_credential_and_the_graph_follow_the_system_argument():
    write_credentials("shop", PASSWORDS)
    write_credentials("billing-eu", {r: f"{p}-eu" for r, p in PASSWORDS.items()})
    for system, pw in (("shop", "pw-raw"), ("billing-eu", "pw-raw-eu")):
        db, opened = FakeDB(), []
        raw_client(db, system=system, opened=opened).raw_query("RETURN 1")
        assert opened == [("127.0.0.1", 16399, f"ccg_raw_{system}", pw)] and db.keys == [f"ccg_{system}"]


def test_the_asker_is_the_launching_os_user():
    write_credentials()
    out = raw_client(FakeDB()).raw_query("RETURN 1")
    assert out["asker"] == pwd.getpwuid(os.getuid()).pw_name == common.asker()


def test_missing_credentials_are_a_typed_error_and_nothing_is_opened():
    opened = []
    with pytest.raises(NoCredentials):
        raw_client(FakeDB(), opened=opened).raw_query("RETURN 1")
    assert opened == []


def test_other_server_failures_are_typed_without_claiming_a_refusal():
    write_credentials()

    class Down:
        def select_graph(self, key):
            raise ConnectionError("connection refused")

    opened = []
    with pytest.raises(StoreError) as e:
        raw_client(Down(), opened=opened).raw_query("RETURN 1")
    assert not isinstance(e.value, RawRefused) and "ConnectionError" in str(e.value)
    assert opened == [("127.0.0.1", 16399, "ccg_raw_shop", "pw-raw")]        # a failure never falls back to another role


def test_the_cli_raw_prints_json_and_a_refusal_is_exit_2():
    write_credentials()
    code, out, _ = run_cli(["raw", "--cypher", "RETURN 1", "--timeout-ms", "2000"], raw_client(FakeDB()))
    body = json.loads(out)
    assert code == 0 and body["timeoutMs"] == 2000 and body["columns"] == ["n", "m"]
    code, out, err = run_cli(["raw", "--cypher", "CREATE (n)"], raw_client(FakeDB()))
    assert (code, out) == (2, "") and json.loads(err)["error"]["code"] == "raw-refused"


def test_the_cli_refresh_prints_the_summary(monkeypatch):
    write_credentials()
    monkeypatch.setattr(refresh_mod, "refresh", lambda system, scope=None, **kw: refresh_mod.RefreshReport(system, snapshot_id="s9"))
    code, out, _ = run_cli(["refresh"], Client(SYSTEM, host="h", port=16399, connect=lambda *a, **k: object()))
    assert code == 0 and json.loads(out)["snapshotId"] == "s9"


# ------------------------------------------------------------------------------------------------------------------ system, endpoint

def test_system_resolution_prefers_the_argument_then_the_environment_then_the_manifest():
    assert common.resolve_system("alpha", {"CCG_SYSTEM": "beta"}) == "alpha"
    assert common.resolve_system(None, {"CCG_SYSTEM": "beta"}) == "beta"
    assert common.resolve_system(None, {}) == "graphmind-ai-lab"              # ccg/system.yaml
    for bad in ("Bad Name", "a" * 49, "a--b"):
        with pytest.raises(BadRequest):
            common.resolve_system(bad, {})


def test_the_endpoint_defaults_and_refuses_falkordb_dev():
    assert common.endpoint({}) == ("127.0.0.1", 6390)
    assert common.endpoint({"CCG_FALKORDB_HOST": "db", "CCG_FALKORDB_PORT": "16390"}) == ("db", 16390)
    for bad in ("6379", "x", "0", "70000"):
        with pytest.raises(BadRequest):
            common.endpoint({"CCG_FALKORDB_PORT": bad})
    with pytest.raises(BadRequest):
        Client(SYSTEM, host="h", port=6379)
    with pytest.raises(BadRequest):
        common._open_raw("h", 6379, username="u", password="p")


# ------------------------------------------------------------------------------------------------------------------ review F1: a store failure is typed, never exit 1

def free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("argv", [["ask", "--request", json.dumps({"useCase": "impact", "anchor": {"kind": "symbol", "repo": "r", "qname": "q"}}), "--fail-on", "any"],
                                  ["resolve", "--anchor", json.dumps({"kind": "symbol", "repo": "r", "qname": "q"})], ["refresh"], ["raw", "--cypher", "RETURN 1"]],
                         ids=["ask", "resolve", "refresh", "raw"])
def test_an_unreachable_store_is_a_typed_error_with_exit_2_never_a_traceback_or_exit_1(argv):
    write_credentials()
    port = free_port()
    code, out, err = run_cli(argv, Client(SYSTEM, host="127.0.0.1", port=port))               # the real connectors against a closed port
    assert (code, out) == (2, "")
    assert json.loads(err)["error"]["code"] in ("store-error", "bad-request") and "Traceback" not in err and "PW" not in err


@pytest.mark.parametrize("exc", [ConnectionError("down"), RuntimeError("boom"), KeyError("k"), ValueError("v")])
def test_no_failure_exits_1_the_fail_on_code_even_one_nobody_anticipated(exc):
    write_credentials()

    def boom(*a, **k):
        raise exc

    g, _, reader = world(flow=True)
    clients = [Client(SYSTEM, host="h", port=16399, reader_factory=boom), Client(SYSTEM, host="h", port=16399, connect=boom, raw_connect=boom)]
    for c in clients:
        for argv in (["ask", "--request", json.dumps(req(g)), "--fail-on", "any"], ["resolve", "--anchor", json.dumps(req(g)["anchor"])], ["refresh"], ["raw", "--cypher", "RETURN 1"]):
            code, out, err = run_cli(argv, c)
            assert code == 2 and out == "" and json.loads(err)["error"]["code"], (argv, err)
    class Exploding:
        def ask(self, *a, **k):
            raise exc
    code, out, err = run_cli(["ask", "--request", "{}", "--fail-on", "any"], Exploding())
    assert code == 2 and out == "" and json.loads(err)["error"]["code"] == "internal-error" and str(exc.args[0]) not in err


# ------------------------------------------------------------------------------------------------------------------ review F2: no silent truncation

def test_a_reply_that_fills_resultset_size_is_a_typed_error_and_one_row_less_is_returned():
    write_credentials()

    class Rows(FakeDB):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def select_graph(self, key):
            n = self.n
            return SimpleNamespace(ro_query=lambda text, params=None, timeout=None: SimpleNamespace(header=[[1, "x"]], result_set=[[i] for i in range(n)]))

    out = raw_client(Rows(config.RESULTSET_SIZE - 1)).raw_query("UNWIND range(1,9999) AS x RETURN x")
    assert len(out["rows"]) == config.RESULTSET_SIZE - 1
    with pytest.raises(common.ResultTooLarge) as e:
        raw_client(Rows(config.RESULTSET_SIZE)).raw_query("UNWIND range(1,10000) AS x RETURN x")
    assert e.value.code == "result-too-large"


# ------------------------------------------------------------------------------------------------------------------ review F3: the survivors

def test_the_cli_hands_args_system_to_the_client_factory_and_the_default_factory_builds_a_client_of_it(monkeypatch):
    seen = []
    cli.main(["--system", "alpha", "raw", "--cypher", "RETURN 1"], client_factory=lambda system: seen.append(system) or (_ for _ in ()).throw(BadRequest("stop")),
             stdout=io.StringIO(), stderr=io.StringIO())
    cli.main(["raw", "--cypher", "RETURN 1"], client_factory=lambda system: seen.append(system) or (_ for _ in ()).throw(BadRequest("stop")),
             stdout=io.StringIO(), stderr=io.StringIO())
    assert seen == ["alpha", None]
    built = []

    class Spy:
        def __init__(self, system):
            built.append(system)
            raise BadRequest("stop")

    monkeypatch.setattr(cli, "Client", Spy)
    cli.main(["--system", "beta", "raw", "--cypher", "RETURN 1"], stdout=io.StringIO(), stderr=io.StringIO())
    assert built == ["beta"]


class Recorder:
    def __init__(self):
        self.calls = []

    def refresh(self, scope=None):
        self.calls.append(("refresh", scope))
        return {"ok": True}

    def resolve(self, anchor):
        self.calls.append(("resolve", anchor))
        return {"ok": True}

    def ask(self, request=None, *, question=None, cache_bypass=False):
        self.calls.append(("ask", request))
        return b_12()


def test_refresh_scope_and_stdin_arguments_reach_the_client(monkeypatch):
    r = Recorder()
    run_cli(["refresh", "--scope", "a", "b"], r)
    run_cli(["refresh"], r)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"useCase": "impact", "anchor": {"kind": "name", "name": "A"}}'))
    run_cli(["ask", "--request", "-"], r)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"kind": "name", "name": "A"}'))
    run_cli(["resolve", "--anchor", "-"], r)
    assert r.calls == [("refresh", ["a", "b"]), ("refresh", None), ("ask", {"useCase": "impact", "anchor": {"kind": "name", "name": "A"}}),
                       ("resolve", {"kind": "name", "name": "A"})]
    write_credentials()
    monkeypatch.setattr("sys.stdin", io.StringIO("RETURN 1\n"))
    code, out, _ = run_cli(["raw", "--cypher", "-"], raw_client(FakeDB()))
    assert code == 0 and json.loads(out)["columns"] == ["n", "m"]


def test_an_ambiguous_resolution_reports_the_candidate_total_beyond_the_cap(monkeypatch):
    g, _, reader = world(flow=True)
    a1 = A.SymbolAnchor("1" * 32, "q", g.repo, "m.py", "u1", 3, 9)
    a2 = A.SymbolAnchor("2" * 32, "q", g.repo, "n.py", "u1", 4, 8)
    monkeypatch.setattr(common.A, "resolve_symbol", lambda r, repo, qname: A.Resolution("ambiguous", options=(a1, a2), options_total=25))
    out = client_over(reader).resolve({"kind": "symbol", "repo": g.repo, "qname": "q"})
    assert out["status"] == "ambiguous" and out["candidatesTotal"] == 25 and len(out["candidates"]) == 2


PLANTED = "PW-planted-s3cr3t-9f2a"


def test_a_credential_never_appears_in_any_error_output(monkeypatch):
    write_credentials(passwords={"build": PLANTED + "b", "verdict": PLANTED + "v", "engine": PLANTED + "e", "raw": PLANTED})
    seen = []

    def capture(c, argv):
        code, out, err = run_cli(argv, c)
        seen.append(out + err)
        assert code != 0

    for text in ("CREATE (n)", "RETURN 1 // timeout_max"):
        capture(raw_client(FakeDB(timeout_max=1)), ["raw", "--cypher", text])
    capture(raw_client(FakeDB()), ["raw", "--cypher", "CREATE (n)"])

    class Down:
        def select_graph(self, key):
            raise ConnectionError("refused")
    capture(raw_client(Down()), ["raw", "--cypher", "RETURN 1"])
    capture(Client(SYSTEM, host="127.0.0.1", port=free_port()), ["refresh"])
    capture(Client(SYSTEM, host="127.0.0.1", port=free_port()), ["resolve", "--anchor", json.dumps({"kind": "symbol", "repo": "r", "qname": "q"})])
    monkeypatch.setattr(common.acl, "read_credentials", lambda system: (_ for _ in ()).throw(common.acl.AclError(f"bad {PLANTED}")))
    capture(raw_client(FakeDB()), ["raw", "--cypher", "RETURN 1"])
    capture(Client(SYSTEM, host="127.0.0.1", port=free_port()), ["refresh"])
    assert len(seen) == 8 and not any(PLANTED in text for text in seen)


# ------------------------------------------------------------------------------------------------------------------ review F5: the socket timeout follows the limit

def test_the_socket_timeout_passed_to_the_connection_exceeds_the_requested_time_limit():
    write_credentials()
    SOCKET_TIMEOUTS.clear()
    for ms in (1_000, 10_000, 90_000):
        raw_client(FakeDB(timeout_max=100_000)).raw_query("RETURN 1", timeout_ms=ms)
    assert SOCKET_TIMEOUTS == [ms / 1000 + common.SOCKET_MARGIN_S for ms in (1_000, 10_000, 90_000)]
    assert all(t > ms / 1000 for t, ms in zip(SOCKET_TIMEOUTS, (1_000, 10_000, 90_000)))


def test_open_raw_hands_the_socket_timeout_to_falkordb_with_zero_retries(monkeypatch):
    import falkordb
    got = {}
    monkeypatch.setattr(falkordb, "FalkorDB", lambda **kw: got.update(kw) or object())
    common._open_raw("h", 16399, username="u", password="p", socket_timeout=12.5)
    assert got["socket_timeout"] == 12.5 and got["username"] == "u" and got["port"] == 16399 and got["retry"] is not None
    common._open_raw("h", 16399, username="u", password="p")
    assert got["socket_timeout"] == config.QUERY_TIMEOUT_MS / 1000 + common.SOCKET_MARGIN_S


# ------------------------------------------------------------------------------------------------------------------ review F6: pinned current behaviour (open question for the architect)

def test_fail_on_exits_3_for_an_ambiguous_or_unresolved_document_and_0_without_it():
    from test_answer import b_amb, b_unres
    for make in (b_amb, b_unres):
        doc = make()
        answer.validate(doc)
        for level in ("possible", "any"):
            code, out, _ = run_cli(["ask", "--request", "{}", "--fail-on", level], DocClient(doc))
            assert code == 3 and out.encode() == answer.dumps(doc), (make.__name__, level)      # printed, and neither 0 nor the tripped 1
        code, out, _ = run_cli(["ask", "--request", "{}"], DocClient(doc))
        assert code == 0 and out.encode() == answer.dumps(doc)
        code, _, _ = run_cli(["ask", "--request", "{}", "--format", "text", "--fail-on", "any"], DocClient(doc))
        assert code == 3


def test_no_broker_module_is_loaded_by_ask_resolve_or_raw_and_refresh_imports_it_lazily():
    import subprocess
    import sys
    code = """
import json, sys
sys.path[:0] = [%r, %r]
import clients.common as c, clients.cli.main
assert not [m for m in sys.modules if m.startswith('ccg.broker')], 'import'
assert 'ccg.refresh' not in sys.modules
from engine_answer_helpers import AnswerReader
import test_engine_answer as TA
g, ids = TA.caller_world(flow=True)
class R(AnswerReader):
    def _x_repo_exists(self, repo): return [(repo,)]
cl = c.Client('shop', host='h', port=16399, reader_factory=lambda *a: R(g), raw_connect=lambda *a, **k: (_ for _ in ()).throw(ConnectionError('x')))
cl.resolve({'kind': 'symbol', 'repo': g.repo, 'qname': 'm.py::A'})
cl.ask({'useCase': 'impact', 'anchor': {'kind': 'symbol', 'repo': g.repo, 'qname': 'm.py::A'}})
try: cl.raw_query('RETURN 1')
except c.ClientError: pass
assert not [m for m in sys.modules if m.startswith('ccg.broker')], sorted(m for m in sys.modules if m.startswith('ccg.broker'))
print('ok')
""" % (str(__import__("pathlib").Path(__file__).parent.parent), str(__import__("pathlib").Path(__file__).parent))      # this tree's ccg/clients first, not the editable install's
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).parent.parent),
                       env={"PATH": os.environ["PATH"], "CCG_CREDENTIALS_DIR": "/nonexistent"})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-600:]


# ------------------------------------------------------------------------------------------------------------------ review Pass 2 minor findings (P1-25b cleanup)

@pytest.mark.parametrize("where", ["reader", "refresh"])
def test_the_store_failure_message_names_the_exception_type_never_its_text(where):
    """M51: the "not reachable" / "refresh failed" messages are built from the type name; the exception's own text (which may hold a password) never leaves."""
    write_credentials()

    def boom(*a, **k):
        raise ConnectionError(f"AUTH ccg_engine_shop {PLANTED} refused")

    c = Client(SYSTEM, host="h", port=16399, reader_factory=boom, connect=boom)
    argv = ["refresh"] if where == "refresh" else ["resolve", "--anchor", json.dumps({"kind": "symbol", "repo": "r", "qname": "q"})]
    code, out, err = run_cli(argv, c)
    payload = json.loads(err)["error"]
    assert (code, out, payload["code"]) == (2, "", "store-error") and "ConnectionError" in payload["message"]
    assert PLANTED not in err and "refused" not in err


def test_the_client_safe_text_is_the_refresh_reports_rule():
    """``clients.common._safe`` duplicates ``ccg.refresh.safe_text`` on purpose: importing ``ccg.refresh`` would load the broker client (``-egress-spec3.md`` §2.1). Pinned equal."""
    import random
    rnd = random.Random(25)
    alphabet = "ab\\\n\r\t\x00\x1b\x7f é€😀" + "".join(chr(c) for c in range(0x20, 0x7f))
    cases = ["", "plain", "a\nb", "\x1b[31mred", "back\\slash", "é", "😀" * 50, "x" * 200, "y" * 120, "z" * 121] + \
            ["".join(rnd.choice(alphabet) for _ in range(rnd.randrange(0, 300))) for _ in range(300)]
    for cap in (120, 50, 10):
        assert [common._safe(c, cap) for c in cases] == [refresh_mod.safe_text(c, cap) for c in cases], cap
    assert [common._safe(c) for c in cases] == [refresh_mod.safe_text(c) for c in cases]          # the defaults agree too


@pytest.mark.parametrize("argv", [["resolve", "--anchor", json.dumps({"kind": "symbol", "repo": "r", "qname": "q"})],
                                  ["ask", "--request", json.dumps({"useCase": "impact", "anchor": {"kind": "symbol", "repo": "r", "qname": "q"}}), "--fail-on", "any"],
                                  ["refresh"], ["raw", "--cypher", "RETURN 1"]], ids=["resolve", "ask", "refresh", "raw"])
def test_a_refused_credential_is_a_distinct_typed_error_pointing_at_the_credentials_and_leaks_no_password(argv):
    from redis.exceptions import AuthenticationError
    write_credentials(passwords={"build": PLANTED + "b", "verdict": PLANTED + "v", "engine": PLANTED + "e", "raw": PLANTED})

    def refused(*a, **k):
        raise AuthenticationError(f"invalid username-password pair {PLANTED}")

    c = Client(SYSTEM, host="h", port=16399, reader_factory=refused, connect=refused, raw_connect=refused)
    code, out, err = run_cli(argv, c)
    payload = json.loads(err)["error"]
    assert (code, out, payload["code"]) == (2, "", "store-auth"), err
    assert "credential" in payload["message"] and "reachable" not in payload["message"] and PLANTED not in err
    role = {"resolve": "engine", "ask": "engine", "refresh": "build or verdict", "raw": "raw"}[argv[0]]
    assert f"the {role} credential of {SYSTEM!r}" in payload["message"]


def test_an_unreachable_store_is_still_not_reported_as_a_credential_problem():
    def down(*a, **k):
        raise ConnectionError("Error 111 connecting")
    c = Client(SYSTEM, host="h", port=16399, reader_factory=down)
    with pytest.raises(StoreError, match="not reachable"):
        c.resolve({"kind": "symbol", "repo": "r", "qname": "q"})


# ------------------------------------------------------------------------------------------------------------------ P1-25b: the MCP server (a thin wrapper over Client)

def mcp_call(name, arguments, client=None, factory=None):
    """One tool call through FastMCP's own argument validation and dispatch: ``(isError, text, systems the factory was asked for)``."""
    import asyncio
    from clients.mcp import server as S
    asked = []

    def make(system):
        asked.append(system)
        return client
    srv = S.build_server(factory or make)
    res = asyncio.run(srv.call_tool(name, arguments))
    (content,) = res.content
    return res.isError, content.text, asked


class FullRecorder(Recorder):
    def ask(self, request=None, *, question=None, cache_bypass=False):
        self.calls.append(("ask", request, question, cache_bypass))
        return b_12()

    def raw_query(self, cypher, *, timeout_ms=None):
        self.calls.append(("raw", cypher, timeout_ms))
        return {"rows": []}


def test_the_mcp_server_lists_exactly_the_four_tools_with_their_arguments():
    import asyncio
    from clients.mcp import server as S
    tools = {t.name: t for t in asyncio.run(S.build_server().list_tools())}
    assert sorted(tools) == ["ask", "raw_query", "refresh", "resolve"]
    props = {n: sorted(t.inputSchema["properties"]) for n, t in tools.items()}
    assert props == {"ask": ["cacheBypass", "question", "request", "system"], "resolve": ["anchor", "system"], "refresh": ["scope", "system"],
                     "raw_query": ["cypher", "system", "timeoutMs"]}
    assert tools["resolve"].inputSchema["required"] == ["anchor"] and tools["raw_query"].inputSchema["required"] == ["cypher"]
    assert "required" not in tools["ask"].inputSchema


def test_mcp_ask_returns_the_canonical_bytes_of_the_engines_document_with_its_notice_intact():
    g, _, reader = world(flow=False)
    expected = client_over(reader).ask(req(g))
    assert expected["notices"]
    is_error, text, asked = mcp_call("ask", {"request": req(g)}, client_over(ClientReader(g)))
    assert not is_error and text.encode() == answer.dumps(expected) and answer.is_canonical(text.encode())
    assert json.loads(text)["notices"] == expected["notices"] and asked == [None]
    code, out, _ = run_cli(["ask", "--request", json.dumps(req(g))], client_over(ClientReader(g)))
    assert code == 0 and out == text                                                          # the canonical bytes, the CLI's exactly


def test_mcp_passes_every_argument_through_unchanged():
    """The silent corruptions: a dropped or defaulted ``system``, ``cacheBypass``, ``question``, ``scope`` or ``timeoutMs`` changes no output shape, only what was asked."""
    r = FullRecorder()
    request = {"useCase": "impact", "anchor": {"kind": "name", "name": "A"}}
    assert mcp_call("ask", {"request": request, "system": "beta"}, r)[2] == ["beta"]
    assert mcp_call("ask", {"question": "who calls A?", "cacheBypass": True, "system": "gamma"}, r)[2] == ["gamma"]
    mcp_call("ask", {"question": "who calls A?"}, r)
    assert mcp_call("resolve", {"anchor": {"kind": "name", "name": "A"}}, r)[2] == [None]
    assert mcp_call("resolve", {"anchor": {"kind": "name", "name": "A"}, "system": "zeta"}, r)[2] == ["zeta"]
    assert mcp_call("refresh", {"scope": ["a", "b"], "system": "delta"}, r)[2] == ["delta"]
    mcp_call("refresh", {}, r)
    assert mcp_call("raw_query", {"cypher": "RETURN 1", "timeoutMs": 1234, "system": "eps"}, r)[2] == ["eps"]
    mcp_call("raw_query", {"cypher": "MATCH (n) RETURN n"}, r)
    assert r.calls == [("ask", request, None, False), ("ask", None, "who calls A?", True), ("ask", None, "who calls A?", False),
                       ("resolve", {"kind": "name", "name": "A"}), ("resolve", {"kind": "name", "name": "A"}), ("refresh", ["a", "b"]), ("refresh", None),
                       ("raw", "RETURN 1", 1234), ("raw", "MATCH (n) RETURN n", None)]


def test_mcp_resolve_refresh_and_raw_return_the_clis_json(monkeypatch):
    g, _, reader = world(flow=True)
    anchor = req(g)["anchor"]
    is_error, text, _ = mcp_call("resolve", {"anchor": anchor}, client_over(reader))
    assert not is_error and text + "\n" == run_cli(["resolve", "--anchor", json.dumps(anchor)], client_over(reader))[1]
    assert json.loads(text)["candidates"][0]["symbolId"]
    write_credentials()
    is_error, text, _ = mcp_call("raw_query", {"cypher": "RETURN 1"}, raw_client(FakeDB()))
    _, out, _ = run_cli(["raw", "--cypher", "RETURN 1"], raw_client(FakeDB()))
    assert not is_error and text + "\n" == out and json.loads(text)["timeoutMs"] == config.QUERY_TIMEOUT_MS
    r = Recorder()
    assert mcp_call("refresh", {"scope": ["x"]}, r)[:2] == (False, '{"ok":true}') and r.calls == [("refresh", ["x"])]


def test_every_non_answer_result_is_one_compact_sorted_json_line_in_both_clients():
    """``dump_json`` is the one form (sorted keys, no spaces, UTF-8 text): the CLI prints it plus a newline, the MCP tool returns it."""
    g, _, reader = world(flow=True)
    anchor = req(g)["anchor"]
    _, text, _ = mcp_call("resolve", {"anchor": anchor}, client_over(reader))
    value = json.loads(text)
    assert text == json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) and list(value) == sorted(value)
    assert run_cli(["resolve", "--anchor", json.dumps(anchor)], client_over(reader))[1] == text + "\n"
    assert common.dump_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'


def test_mcp_typed_errors_are_tool_errors_carrying_the_clis_payload_and_nothing_else():
    g, _, reader = world(flow=True)
    cases = [("ask", {"request": {**req(g), "scope": ["r"]}}, ["ask", "--request", json.dumps({**req(g), "scope": ["r"]})], "not-built"),
             ("ask", {"request": req(g, use_case="vuln")}, ["ask", "--request", json.dumps(req(g, use_case="vuln"))], "not-built"),
             ("ask", {"question": "who calls A?"}, ["ask", "--question", "who calls A?"], "not-built"),
             ("ask", {"request": req(g), "question": "q"}, None, "bad-request"),
             ("resolve", {"anchor": {"kind": "nope"}}, ["resolve", "--anchor", '{"kind": "nope"}'], "bad-request")]
    for name, args, argv, code in cases:
        is_error, text, _ = mcp_call(name, args, client_over(ClientReader(g)))
        assert is_error and json.loads(text)["error"]["code"] == code, (name, args, text)
        if argv:
            assert text + "\n" == run_cli(argv, client_over(ClientReader(g)))[2]
    write_credentials()
    is_error, text, _ = mcp_call("raw_query", {"cypher": "CREATE (n)"}, raw_client(FakeDB()))
    assert is_error and json.loads(text)["error"]["code"] == "raw-refused"


def test_mcp_ask_validates_first_and_returns_nothing_of_an_invalid_document():
    bad = b_12()
    del bad["stamp"]
    is_error, text, _ = mcp_call("ask", {"request": {}}, DocClient(bad))
    assert is_error and json.loads(text) == {"error": {"code": "invalid-answer", "message": "the answer does not validate; nothing was returned"}}
    assert "items" not in text and "stamp" not in text


@pytest.mark.parametrize("exc", [ConnectionError(PLANTED), RuntimeError(PLANTED), KeyError(PLANTED)])
def test_mcp_never_answers_with_a_traceback_or_an_exceptions_text(exc):
    class Exploding:
        def ask(self, *a, **k):
            raise exc
    is_error, text, _ = mcp_call("ask", {"request": {}}, Exploding())
    assert is_error and json.loads(text)["error"] == {"code": "internal-error", "message": f"unexpected failure ({type(exc).__name__})"}
    is_error, text, _ = mcp_call("ask", {"request": {}}, factory=lambda system: (_ for _ in ()).throw(BadRequest("bad system")))
    assert is_error and json.loads(text)["error"]["code"] == "bad-request"


def _server_env(tmp_path, **extra):
    env = {"PATH": os.environ["PATH"], "CCG_FORBID_REAL_CREDENTIALS": "1", "CCG_CREDENTIALS_DIR": str(tmp_path / "cred"), "CCG_CONFIG_DIR": str(tmp_path / "cfg"),
           "CCG_SYSTEM": SYSTEM, "CCG_FALKORDB_HOST": "127.0.0.1", "CCG_FALKORDB_PORT": str(free_port())}
    return {**env, **extra}


def test_the_stdio_server_starts_serves_the_tools_and_answers_typed_errors_end_to_end(tmp_path):
    """``python -m clients.mcp.server`` over a real stdio session (the mcp SDK's client): the tools are listed, a question is ``not-built``, and ``raw_query`` against
    a closed port is a typed ``store-error`` (or ``no-credentials``), never a crash of the server."""
    import asyncio
    import sys
    from pathlib import Path
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cpg = Path(__file__).resolve().parent.parent
    params = StdioServerParameters(command=sys.executable, args=["-m", "clients.mcp.server"], cwd=str(cpg), env=_server_env(tmp_path))

    async def session():
        async with stdio_client(params, errlog=open(os.devnull, "w")) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()
                names = sorted(t.name for t in (await s.list_tools()).tools)
                q = await s.call_tool("ask", {"question": "who calls A?"})
                raw = await s.call_tool("raw_query", {"cypher": "RETURN 1"})
                return names, q, raw

    names, q, raw = asyncio.run(asyncio.wait_for(session(), 60))
    assert names == ["ask", "raw_query", "refresh", "resolve"]
    assert q.isError and json.loads(q.content[0].text)["error"]["code"] == "not-built"
    assert raw.isError and json.loads(raw.content[0].text)["error"]["code"] in ("store-error", "no-credentials")


@pytest.mark.parametrize("extra,code", [({"CCG_FALKORDB_PORT": "6379"}, "bad-request"), ({"CCG_SYSTEM": "Bad Name!"}, "bad-request")])
def test_the_stdio_server_refuses_a_bad_configuration_at_start_with_exit_2(tmp_path, extra, code):
    import subprocess
    import sys
    from pathlib import Path
    r = subprocess.run([sys.executable, "-m", "clients.mcp.server"], cwd=str(Path(__file__).resolve().parent.parent), env=_server_env(tmp_path, **extra),
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert r.returncode == 2 and r.stdout == "" and json.loads(r.stderr.strip().splitlines()[-1])["error"]["code"] == code, r.stderr[-400:]


# ------------------------------------------------------------------------------------------------------------------ P1-25b review F3, F5

@pytest.mark.parametrize("tool,args", [
    ("raw_query", {"cypher": "RETURN 1", "timeoutMs": True}), ("raw_query", {"cypher": "RETURN 1", "timeoutMs": "5"}), ("raw_query", {"cypher": "RETURN 1", "timeoutMs": 2.5}),
    ("raw_query", {"cypher": "RETURN 1", "timeoutMs": 2.0}), ("raw_query", {"cypher": 5}), ("raw_query", {"cypher": "RETURN 1", "system": 7}),
    ("ask", {"question": "who calls A?", "cacheBypass": 1}), ("ask", {"question": "who calls A?", "cacheBypass": "yes"}), ("ask", {"question": 42}),
    ("ask", {"request": "not an object"}), ("refresh", {"scope": "abc"}), ("refresh", {"scope": [1, 2]}), ("resolve", {"anchor": "abc"}), ("resolve", {"anchor": [1]}),
], ids=lambda v: json.dumps(v) if isinstance(v, dict) else v)
def test_a_wrong_argument_type_is_never_coerced_and_is_the_typed_bad_request(tool, args):
    """F3: the tools declare ``Any``, so FastMCP coerces nothing and ``Client`` rejects the type with its own typed error (never pydantic's text, never a silent 1)."""
    write_credentials()
    is_error, text, _ = mcp_call(tool, args, factory=lambda system: Client(system, host="127.0.0.1", port=16399, raw_connect=lambda *a, **k: FakeDB()))
    assert is_error and json.loads(text)["error"]["code"] == "bad-request", text
    assert "pydantic" not in text and "validation error" not in text


def test_a_json_array_sent_as_a_string_is_still_parsed_by_fastmcp_and_a_scalar_string_is_not():
    r = FullRecorder()
    mcp_call("refresh", {"scope": '["a"]'}, r)
    mcp_call("raw_query", {"cypher": "RETURN 1", "timeoutMs": 2000}, r)
    assert r.calls == [("refresh", ["a"]), ("raw", "RETURN 1", 2000)]


def test_an_auth_failure_is_recognised_through_a_subclass_of_the_drivers_class():
    """F5: the match walks the MRO, so a driver subclass (a future redis-py split, a wrapper) is still ``store-auth``."""
    from redis.exceptions import AuthenticationError

    class TokenExpired(AuthenticationError):
        pass

    def refused(*a, **k):
        raise TokenExpired(f"expired {PLANTED}")

    write_credentials()
    c = Client(SYSTEM, host="h", port=16399, reader_factory=refused, raw_connect=refused, connect=refused)
    for call in (lambda: c.resolve({"kind": "symbol", "repo": "r", "qname": "q"}), lambda: c.raw_query("RETURN 1"), lambda: c.refresh()):
        with pytest.raises(common.StoreAuthError) as info:
            call()
        assert PLANTED not in str(info.value)
    assert common._is_auth_failure(TokenExpired()) and not common._is_auth_failure(ConnectionError())
