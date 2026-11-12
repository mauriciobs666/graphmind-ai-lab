"""P1-39b the answer builder, LIVE on a throwaway instance only (``infra/ccg-falkordb.sh --throwaway up``; ``-m live``): store -> walk -> build -> validate.

Needs ``CCG_FALKORDB_PORT`` (the throwaway's port; 6379 and 6390 are refused) and ``CCG_CONFIG_DIR`` (a scratch dir holding the throwaway's ``falkordb-admin.pass``);
``CCG_REQUIRE_LIVE=1`` turns a missing instance into a failure. Loads the union of the builder tests' worlds (each in a repository of its own, ids rewritten to 32-hex) through
``ccg.store.loader.load`` as ``ccg_build_<s>``, renames the build key to the live key as admin, and reads it as ``ccg_engine_<s>`` only (``CypherReader``). The oracle is the unit
tier's own reader: every answer is built over ``CypherReader`` and over ``AnswerReader`` on the same rows and the two documents must be byte-identical, and each must pass
``ccg.answer.validate`` and equal what ``loads(dumps(...))`` gives back.
Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §5, ``-p1-answers.md`` P1-39 (0)-(3)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from engine_answer_helpers import AnswerReader, hexify
from engine_walk_helpers import union
from loader_helpers import admin_password, skip_or_fail, snapshot_row, throwaway_port

import test_engine_answer as TA
from ccg import answer, sysname
from ccg.engine import answer_build as AB
from ccg.engine.reader import CypherReader
from ccg.store import acl
from ccg.store import loader as L

HOST = "127.0.0.1"
SYSTEM = f"p139b{os.getpid() % 1000}"
pytestmark = pytest.mark.live

PARTS = {"caller": TA.caller_world_g, "flagfalse": TA.flag_false_two_callers_g, "link": TA.link_world_g, "depends": TA.depends_world_g, "fold": TA.fold_world_g,
         "data": TA.data_reached_sites_g}
#: (part, anchor kind, readable name, question)
REQUESTS = [("caller", "symbol", "A", "impact"), ("flagfalse", "symbol", "A", "impact"), ("link", "contract", "cp1", "impact"), ("link", "symbol", "H", "impact"),
            ("link", "config", "e1", "impact"), ("depends", "symbol", "getHealth", "depends-on"), ("fold", "symbol", "F", "depends-on"), ("data", "symbol", "F", "depends-on"),
            ("data", "symbol", "A", "impact")]


def _parts():
    parts = {}
    for name, make in PARTS.items():
        g = make()
        for nodes in g.rows.nodes.values():
            for n in nodes:
                if "repoId" in n:
                    n["repoId"] = f"repo-{name}"
        parts[name] = g
    return parts


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
    cred = tmp_path_factory.mktemp("ccg-credentials-p139b")
    graph, ids_ = hexify(union(_parts()))
    graph.rows.snapshot = snapshot_row(SYSTEM, "p139b")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CCG_CREDENTIALS_DIR", str(cred))
        users = acl.create_users(admin, SYSTEM)
        creds = acl.read_credentials(SYSTEM)
    w = SimpleNamespace(port=port, admin=admin, graph=graph, ids=ids_, snapshot=graph.rows.snapshot)
    try:
        build_db = L.connect(HOST, port, username=sysname.acl_user("build", SYSTEM), password=creds["build"])
        report = L.load(L.LiveStore(build_db, SYSTEM), graph.rows, SYSTEM)
        admin.connection.rename(report.build_key, sysname.live_key(SYSTEM))
        w.db = L.connect(HOST, port, username=sysname.acl_user("engine", SYSTEM), password=creds["engine"])
        yield w
    finally:
        for key in list(admin.connection.scan_iter(match=f"ccg_{SYSTEM}*")):
            admin.connection.delete(key)
        for user in users:
            admin.connection.execute_command("ACL", "DELUSER", user)


def _anchor(world, part, kind, name):
    node = world.ids[f"{part}:{name}"]
    return {"symbol": TA.sym_anchor, "contract": TA.cp_anchor, "config": TA.cfg_anchor}[kind](world.graph, {name: node}, name)


def _memory(world):
    s = world.snapshot
    return AnswerReader(world.graph, [(s["snapshotId"], s["verdictLogOffset"], s["verdictLogHash"], s["graphModelVersion"], s["bundleSchemaVersion"])])


@pytest.mark.parametrize("part,kind,name,question", REQUESTS, ids=[f"{r[0]}-{r[2]}-{r[3]}" for r in REQUESTS])
def test_an_answer_built_over_the_store_is_valid_and_byte_identical_to_the_one_over_the_in_memory_reader(world, part, kind, name, question):
    a = _anchor(world, part, kind, name)
    live = AB.build_answer(CypherReader(world.db, SYSTEM), a, question=question)
    mem = AB.build_answer(_memory(world), a, question=question)
    answer.validate(live)
    assert answer.dumps(live) == answer.dumps(mem)
    assert answer.loads(answer.dumps(live)) == live
    assert live["items"] or live["groups"], "an empty answer proves nothing"
    assert live["stamp"]["snapshotId"] == world.snapshot["snapshotId"]


def test_a_live_rebuild_gives_the_same_facts_bytes_and_every_text_it_ran_is_in_the_closed_table(world):
    from ccg.engine.reader import QUERIES
    a = _anchor(world, "link", "contract", "cp1")
    rd = CypherReader(world.db, SYSTEM)
    one, two = AB.build_answer(rd, a), AB.build_answer(CypherReader(world.db, SYSTEM), a)
    assert answer.facts_bytes(one) == answer.facts_bytes(two)
    assert one["recipe"]["queries"] and all(QUERIES[q["name"]] == q["cypher"] for q in one["recipe"]["queries"])
    assert {"lifted_of", "bound_out", "candidate_count", "snapshot", "units"} <= {q["name"] for q in one["recipe"]["queries"]}


def test_live_an_anchor_the_store_does_not_hold_is_a_build_error(world):
    a = _anchor(world, "caller", "symbol", "A")
    gone = type(a)(a.symbol_id.replace(a.symbol_id[0], "0" if a.symbol_id[0] != "0" else "1", 1), a.qname, a.repo_id, a.path, a.unit_id, a.line_start, a.line_end)
    with pytest.raises(AB.BuildError, match="anchor"):
        AB.build_answer(CypherReader(world.db, SYSTEM), gone)
