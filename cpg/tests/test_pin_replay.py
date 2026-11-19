"""P1-25 (6), AC-16 as pin-and-replay (``docs/plans/connected-code-graph-eval-protocol-ml.md`` §10.6; ``-answer-spec.md`` §6 ``facts_bytes``).

One structured request (the pin) submitted through the MCP ``ask`` tool, the CLI and the engine itself gives byte-equal ``facts_bytes``. The pin is also derived the way
§10.6 takes it, from the ``request`` a resolved answer echoes, and replaying it gives the same facts. Planted inputs: the direction flipped in one replay changes the facts;
a change outside the facts (``staleness.units[].builtAt``, spec2 §4.3) does not.

**The raw-recipe leg is the weaker one, by decision** (``-answer-spec.md`` §3/§4.1): re-running ``recipe.queries`` yields rows, never an answer document, so it is compared on
the item set ``(subject id, status, ((kind, id) of each chain hop))`` over ``items`` plus ``groups[].members``. It needs the store, so it is live. The driver below
(:func:`raw_item_set`) re-runs the recipe's own texts, verbatim, through ``Client.raw_query`` (the ``ccg_raw_<s>`` role, parameters as the ``CYPHER`` prefix) and derives
items from the rows by the ``bfs-v1`` description (``docs/plans/connected-code-graph-p0-graph.md`` §3), importing no engine function. **Its coverage is the call column of a
flag-true unit:** ``CALLS <-`` and ``CALLS_NAME``/``RESOLVES_TO`` (``name_in``) from the anchor symbol, then ``HAS_CALL <-`` to the owner, where the walk ends. Any other walk
read in the recipe that returns rows is a derivation the driver cannot make, and the leg FAILS (never skips), as §10.6 requires. The harness's full driver is GC's (§10.6).

    cd cpg && .venv/bin/python -m pytest tests/test_pin_replay.py                      # offline legs
    cd cpg && .venv/bin/python -m pytest -m live tests/test_pin_replay.py              # + the live legs (throwaway instance: test_engine_answer_live.py's setup)
"""
from __future__ import annotations

import asyncio
import copy
import io
import json
import os
from types import SimpleNamespace

import pytest
from engine_answer_helpers import AnswerReader, hexify
from engine_walk_helpers import G

from ccg import answer, sysname
from ccg.engine import anchors as A
from ccg.engine import answer_build as AB
from clients import common
from clients.cli import main as cli
from clients.common import Client
from clients.mcp import server as mcp_server

SYSTEM = "shop"
HOST = "127.0.0.1"


# ------------------------------------------------------------------------------------------------------------------ the world and the pins

def chain_world_g(*, flow: bool) -> G:
    """``B`` calls ``A`` (a resolved ``CALLS``), ``F`` calls ``A`` by a ``possible`` ``CALLS`` edge (``shell-dynamic``), ``E`` calls a name that resolves to ``A``
    (a ``possible`` name match), ``C`` calls ``B``, ``D`` calls ``C``.
    Flag-true, the walk ends at each caller (items ``B``, ``E``, ``F``); flag-false, function-level continuation also reaches ``C`` (``possible``, distance 2)."""
    g = G(unit="u1", flow=flow)
    g.sym("A", "m.py::A", path="m.py", start=3, end=9)
    for name in ("B", "C", "D", "E", "F"):
        g.sym(name, f"{name.lower()}.py::{name}", path=f"{name.lower()}.py", start=1, end=20)
    g.site("sB", "B", line=5, callee="A", path="b.py")
    g.calls("sB", "A")
    g.site("sC", "C", line=6, callee="B", path="c.py")
    g.calls("sC", "B")
    g.site("sD", "D", line=7, callee="C", path="d.py")
    g.calls("sD", "C")
    g.site("sE", "E", line=8, callee="A", path="e.py")
    g.calls_name("sE", "A", name="A")
    g.site("sF", "F", line=9, callee="A", path="f.py")
    g.calls("sF", "A", status="possible", reason="shell-dynamic")
    return g


class PinReader(AnswerReader):
    """``AnswerReader`` plus the resolver's ``repo_exists`` read (as ``tests/test_clients.ClientReader``)."""

    def _x_repo_exists(self, repo):
        return [(repo,)] if any(n.get("repoId") == repo for nodes in self._nodes.values() for n in nodes.values()) else []


def pin(qname: str, use_case: str = "impact", repo: str = "repo-a") -> dict:
    return {"useCase": use_case, "anchor": {"kind": "symbol", "repo": repo, "qname": qname}}


def pin_from_echo(doc: dict) -> dict:
    """§10.6's pin: the ``request`` a resolved answer echoes, as a structured request (one symbol anchor; P1 echoes no other kind for these use cases here)."""
    request = doc["interpretation"]["request"]
    (anchor,) = request["anchors"]
    assert anchor["type"] == "symbol", anchor
    return {"useCase": request["useCase"], "anchor": {"kind": "symbol", "repo": anchor["repoId"], "qname": anchor["qname"]}}


def item_set(doc: dict) -> set[tuple]:
    """The comparison key of the raw leg: ``(subject id, status, ((kind, id), ...))`` per item and group member."""
    out = set()
    for it in [*doc["items"], *(m for g in doc["groups"] for m in g["members"])]:
        subject = it["subject"]
        sid = subject.get("symbolId") or subject.get("entryId") or subject.get("unitId")
        out.add((sid, it["status"], tuple((h["kind"], h["id"]) for h in it["chain"])))
    return out


# ------------------------------------------------------------------------------------------------------------------ the three structured legs

def engine_leg(reader_factory, request: dict) -> dict:
    """The engine itself: resolve the anchor and build the answer, no client code."""
    a = request["anchor"]
    res = A.resolve_symbol(reader_factory(), a["repo"], a["qname"])
    assert res.status == "resolved", res.status
    return AB.build_answer(reader_factory(), res.anchors[0], question=request["useCase"])


def cli_leg(client_factory, request: dict, system: str | None = None) -> bytes:
    out, err = io.StringIO(), io.StringIO()
    argv = (["--system", system] if system else []) + ["ask", "--request", json.dumps(request)]
    code = cli.main(argv, client_factory=client_factory, stdout=out, stderr=err)
    assert (code, err.getvalue()) == (0, ""), err.getvalue()
    return out.getvalue().encode()


def mcp_leg(client_factory, request: dict, system: str | None = None) -> bytes:
    srv = mcp_server.build_server(client_factory)
    args = {"request": request, **({"system": system} if system else {})}
    res = asyncio.run(srv.call_tool("ask", args))
    (content,) = res.content
    assert not res.isError, content.text
    return content.text.encode()


def three_legs(reader_factory, client_factory, request: dict, system: str | None = None) -> dict[str, bytes]:
    legs = {"engine": answer.facts_bytes(engine_leg(reader_factory, request)),
            "cli": answer.facts_bytes(answer.loads(cli_leg(client_factory, request, system))),
            "mcp": answer.facts_bytes(answer.loads(mcp_leg(client_factory, request, system)))}
    return legs


# ------------------------------------------------------------------------------------------------------------------ offline: MCP, CLI and engine agree

def offline(flow: bool):
    g, ids_ = hexify(chain_world_g(flow=flow))
    reader_factory = lambda: PinReader(g)                                                      # noqa: E731 - a fresh reader per read, as the live client opens one
    client_factory = lambda system: Client(system or SYSTEM, host=HOST, port=16399, reader_factory=lambda *a: PinReader(g))  # noqa: E731
    return g, ids_, reader_factory, client_factory


PINS = [(True, "m.py::A", "impact"), (False, "m.py::A", "impact"), (True, "d.py::D", "depends-on"), (False, "d.py::D", "depends-on"), (True, "c.py::C", "impact")]


@pytest.mark.parametrize("flow,qname,use_case", PINS, ids=[f"{'flagtrue' if p[0] else 'flagfalse'}-{p[1]}-{p[2]}" for p in PINS])
def test_the_pin_through_mcp_the_cli_and_the_engine_gives_byte_equal_facts(flow, qname, use_case):
    _, _, reader_factory, client_factory = offline(flow)
    legs = three_legs(reader_factory, client_factory, pin(qname, use_case))
    assert legs["engine"] == legs["cli"] == legs["mcp"]
    doc = answer.loads(mcp_leg(client_factory, pin(qname, use_case)))
    assert doc["items"] or doc["groups"], "an empty answer proves nothing"
    assert ("notices" in doc) == (not flow)                                                  # the flag-false notice is a fact and survives every leg


def test_the_pin_taken_from_an_echoed_request_replays_to_the_same_facts():
    _, _, reader_factory, client_factory = offline(False)
    first = answer.loads(mcp_leg(client_factory, pin("m.py::A")))
    replay = pin_from_echo(first)
    assert replay == pin("m.py::A")
    legs = three_legs(reader_factory, client_factory, replay)
    assert legs["engine"] == legs["cli"] == legs["mcp"] == answer.facts_bytes(first)


def test_the_system_argument_reaches_the_client_on_every_leg():
    """A wrapper that dropped ``system`` would still answer from the default system; the factory sees what each leg was given."""
    _, _, _, client_factory = offline(True)
    seen = []

    def spy(system):
        seen.append(system)
        return client_factory(system)
    cli_leg(spy, pin("m.py::A"), system="other")
    mcp_leg(spy, pin("m.py::A"), system="other")
    assert seen == ["other", "other"]


def test_planted_a_flipped_direction_changes_the_facts():
    _, ids_, reader_factory, client_factory = offline(False)
    impact = three_legs(reader_factory, client_factory, pin("c.py::C", "impact"))
    flipped = three_legs(reader_factory, client_factory, pin("c.py::C", "depends-on"))
    assert impact["mcp"] != flipped["mcp"] and impact["cli"] != flipped["cli"] and impact["engine"] != flipped["engine"]
    assert flipped["engine"] == flipped["cli"] == flipped["mcp"]


def test_planted_b_a_change_outside_the_facts_compares_equal_and_a_fact_change_does_not():
    _, _, _, client_factory = offline(False)
    doc = answer.loads(mcp_leg(client_factory, pin("m.py::A")))
    moved = copy.deepcopy(doc)
    for unit in moved["staleness"]["units"]:
        unit["builtAt"] = "2030-01-01T00:00:00Z"
    assert answer.dumps(moved) != answer.dumps(doc) and answer.facts_bytes(moved) == answer.facts_bytes(doc)
    dropped = copy.deepcopy(doc)
    dropped["items"] = dropped["items"][:-1]
    assert answer.facts_bytes(dropped) != answer.facts_bytes(doc)


def test_the_item_set_key_sees_status_and_every_chain_hop():
    _, _, _, client_factory = offline(False)
    doc = answer.loads(mcp_leg(client_factory, pin("m.py::A")))
    base = item_set(doc)
    assert {(s, st) for s, st, _ in base} and any(st == "possible" for _, st, _ in base) and any(st == "confirmed" for _, st, _ in base)
    for change in ("status", "hop"):
        d = copy.deepcopy(doc)
        it = d["items"][0]
        if change == "status":
            it["status"] = "possible" if it["status"] == "confirmed" else "confirmed"
        else:
            it["chain"][1]["id"] = "f" * 32
        assert item_set(d) != base, change


# ------------------------------------------------------------------------------------------------------------------ the raw-recipe driver

WALK_READS = {"calls_in", "name_in", "owner_of:CallSite", "to_in", "data_in:CallSite"}
NOT_WALK_READS = {"snapshot", "units", "symbol_by_qname"}           # the stamp, the staleness and the anchor's own read: no item comes from them


class Underivable(AssertionError):
    """The rows hold something the driver cannot turn into items: the raw leg fails (§10.6: never skipped)."""


def raw_item_set(doc: dict, run) -> set[tuple]:
    """The item set of ``doc``'s ``recipe.queries`` re-run through ``run(cypher, params) -> rows`` (no engine import). Call column, flag-true units, impact only."""
    queries = {q["name"]: q["cypher"] for q in doc["recipe"]["queries"]}
    unknown = set(queries) - WALK_READS - NOT_WALK_READS
    if unknown:
        raise Underivable(f"the recipe ran walk reads this driver does not derive: {sorted(unknown)}")
    request = doc["interpretation"]["request"]
    if request["direction"] != "impact":
        raise Underivable("the driver derives the impact column only")
    (anchor,) = request["anchors"]
    root = ("Symbol", anchor["symbolId"])

    def read(name, ids_):
        if name not in queries:
            return []
        return run(queries[name], {"ids": sorted(ids_)})

    # level 1, from the anchor symbol (state value): CALLS <- keeps the edge status; a name-only call site (CALLS_NAME -> Name -> RESOLVES_TO) is possible
    sites: dict[str, bool] = {}                                       # call site id -> its hop is possible
    for row in read("calls_in", [anchor["symbolId"]]):
        sites[row[1]] = row[7] == "possible"
    for row in read("name_in", [anchor["symbolId"]]):
        sites.setdefault(row[1], True)
    if read("to_in", [anchor["symbolId"]]):
        raise Underivable("a Link reaches the anchor: link hops are not derived here")
    if not sites:
        return set()
    # level 2, from each call site (state path r): DATA <- would widen the walk; HAS_CALL <- reaches the owner, where a flag-true walk ends
    if read("data_in:CallSite", sites):
        raise Underivable("a DATA edge enters a reached call site: data hops are not derived here")
    items = set()
    for row in read("owner_of:CallSite", sites):
        site, owner = row[0], row[1]
        status = "possible" if sites[site] else "confirmed"
        items.add((owner, status, (root, ("CallSite", site), ("Symbol", owner))))
    return items


def cypher_with_params(cypher: str, params: dict) -> str:
    """The recipe text, unchanged, behind FalkorDB's ``CYPHER k=v`` parameter prefix (what a driver sends for ``$ids``)."""
    head = " ".join(f"{k}={json.dumps(v)}" for k, v in sorted(params.items()))
    return f"CYPHER {head} {cypher}"


def test_the_raw_driver_refuses_a_recipe_it_cannot_derive_rather_than_passing():
    _, _, _, client_factory = offline(False)
    flag_false = answer.loads(mcp_leg(client_factory, pin("m.py::A")))                       # function-level continuation reads calls_out, name_out, has_out
    with pytest.raises(Underivable, match="does not derive"):
        raw_item_set(flag_false, lambda cypher, params: [])
    _, _, _, client_factory = offline(True)
    depends = answer.loads(mcp_leg(client_factory, pin("d.py::D", "depends-on")))
    with pytest.raises(Underivable):
        raw_item_set(depends, lambda cypher, params: [])


def test_cypher_with_params_keeps_the_recipe_text_verbatim():
    text = "UNWIND $ids AS i MATCH (n {id: i}) RETURN i"
    assert cypher_with_params(text, {"ids": ["b", "a"]}) == 'CYPHER ids=["b", "a"] ' + text


# ------------------------------------------------------------------------------------------------------------------ live: the same legs over the store, and the raw leg

LIVE_SYSTEM = f"p125pin{os.getpid() % 1000}"


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    from loader_helpers import admin_password, skip_or_fail, snapshot_row, throwaway_port

    from ccg.store import acl
    from ccg.store import loader as L
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance: start one with cpg/infra/ccg-falkordb.sh --throwaway up and export CCG_FALKORDB_PORT and CCG_CONFIG_DIR (a scratch dir)")
    try:
        pw = admin_password()
    except FileNotFoundError as exc:
        skip_or_fail(f"throwaway admin password not readable ({exc})")
    admin = L.connect(HOST, port, username=None, password=pw)
    cred = tmp_path_factory.mktemp("ccg-credentials-p125pin")
    g, ids_ = hexify(chain_world_g(flow=True))
    g.node("Repo", repoId=g.repo, headCommit="c" * 40)                                   # the resolver's ``repo_exists`` read (offline, ``PinReader`` answers it)
    g.rows.snapshot = snapshot_row(LIVE_SYSTEM, "p125pin")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CCG_CREDENTIALS_DIR", str(cred))
        users = acl.create_users(admin, LIVE_SYSTEM)
        creds = acl.read_credentials(LIVE_SYSTEM)
    try:
        build_db = L.connect(HOST, port, username=sysname.acl_user("build", LIVE_SYSTEM), password=creds["build"])
        report = L.load(L.LiveStore(build_db, LIVE_SYSTEM), g.rows, LIVE_SYSTEM)
        admin.connection.rename(report.build_key, sysname.live_key(LIVE_SYSTEM))
        s = g.rows.snapshot
        snap = [(s["snapshotId"], s["verdictLogOffset"], s["verdictLogHash"], s["graphModelVersion"], s["bundleSchemaVersion"])]
        yield SimpleNamespace(port=port, cred=cred, g=g, ids=ids_, memory=lambda: PinReader(g, snap))
    finally:
        for key in list(admin.connection.scan_iter(match=f"ccg_{LIVE_SYSTEM}*")):
            admin.connection.delete(key)
        for user in users:
            admin.connection.execute_command("ACL", "DELUSER", user)


@pytest.fixture
def live_env(live, monkeypatch):
    """The environment the MCP server and the CLI are configured by (``CCG_SYSTEM``, ``CCG_FALKORDB_HOST``/``PORT``, the credentials directory)."""
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(live.cred))
    monkeypatch.setenv("CCG_SYSTEM", LIVE_SYSTEM)
    monkeypatch.setenv("CCG_FALKORDB_HOST", HOST)
    monkeypatch.setenv("CCG_FALKORDB_PORT", str(live.port))
    return live


@pytest.mark.live
@pytest.mark.parametrize("qname,use_case", [("m.py::A", "impact"), ("d.py::D", "depends-on")])
def test_live_the_pin_through_mcp_the_cli_and_the_engine_gives_byte_equal_facts(live_env, qname, use_case):
    from ccg.engine.reader import CypherReader
    reader_factory = lambda: CypherReader.for_system(LIVE_SYSTEM, HOST, live_env.port)      # noqa: E731 - the engine role, as the engine itself runs
    legs = three_legs(reader_factory, lambda system: Client(system), pin(qname, use_case))   # the default factories: configured by the environment only
    assert legs["engine"] == legs["cli"] == legs["mcp"]
    assert legs["engine"] == answer.facts_bytes(engine_leg(live_env.memory, pin(qname, use_case)))      # and the store agrees with the in-memory rows


@pytest.mark.live
def test_live_the_raw_recipe_rerun_reproduces_the_item_set_the_weaker_leg(live_env):
    doc = answer.loads(mcp_leg(lambda system: Client(system), pin("m.py::A")))
    expected = item_set(doc)
    ids_ = live_env.ids
    assert {(s, st, chain[1]) for s, st, chain in expected} == {(ids_["B"], "confirmed", ("CallSite", ids_["sB"])),
                                                                (ids_["E"], "possible", ("CallSite", ids_["sE"])),       # name_in
                                                                (ids_["F"], "possible", ("CallSite", ids_["sF"]))}       # calls_in, a possible CALLS edge
    raw = Client(LIVE_SYSTEM)
    ran = []

    def run(cypher, params):
        ran.append(cypher)
        return raw.raw_query(cypher_with_params(cypher, params))["rows"]
    assert raw_item_set(doc, run) == expected
    assert ran and set(ran) <= {q["cypher"] for q in doc["recipe"]["queries"]}             # only the recipe's own texts ran, as the raw role


@pytest.mark.live
def test_live_the_raw_leg_detects_a_planted_difference(live_env):
    """The comparison is not vacuous over the store: an answer whose one item's status is flipped no longer matches the rows."""
    doc = answer.loads(mcp_leg(lambda system: Client(system), pin("m.py::A")))
    raw = Client(LIVE_SYSTEM)
    derived = raw_item_set(doc, lambda cypher, params: raw.raw_query(cypher_with_params(cypher, params))["rows"])
    planted = copy.deepcopy(doc)
    planted["items"][0]["status"] = "possible" if planted["items"][0]["status"] == "confirmed" else "confirmed"
    assert derived == item_set(doc) and derived != item_set(planted)
