"""P1-32b Pass 4, P4-1, live tier (``pytest -m live``): a REAL hang, through the real ``Client`` over a real ``CypherReader`` over a real socket, then
``make_ask``, ``_Persisting`` and the marker. Own throwaway FalkorDB from port 16430 up (``test_gate_rehearsal_defects_live.store``: local image,
``--pull never``, 127.0.0.1, labels ``ccg-falkordb=1`` ``ccg-falkordb.throwaway=1``, removed afterwards; never 6379/6390/16390/16400).

* a black-holed product connection (a listening socket that never answers: the product hangs, the host is healthy) with the canary on the direct, healthy
  connection: stops 1 and 2 are ``failed:transient``, stop 3 converts the anchor (``via: repeat-stall``);
* the whole store paused (``docker pause``: the host is down, the canary fails too): never converts, however often it stops.
Invented data (``engine_helpers.mini_rows``)."""
import socket
import subprocess

import pytest
from engine_helpers import mini_rows
from test_gate_rehearsal_defects_live import HOST, SYSTEM, creds, store  # noqa: F401  (the container and credential fixtures)

import gate_helpers as H  # noqa: F401
from gate_helpers import G
from gate_run_helpers import Env, many_world
import gate_run as R
from ccg import sysname
from ccg.engine import config
from ccg.engine.reader import CypherReader
from ccg.store import acl
from ccg.store import loader as L
from clients.common import Client

pytestmark = pytest.mark.live
SOCKET_TIMEOUT_S = 1.5
STAMP = {"snapshotId": H.SNAP, "verdictLogOffset": 0, "verdictLogHash": G.EMPTY_VERDICT_LOG_HASH}


@pytest.fixture
def live_store(store, creds):
    rows = mini_rows(SYSTEM)
    rep = L.load(L.LiveStore(L.connect(HOST, store.port, username=None, password=store.admin_pw), SYSTEM), rows, SYSTEM)
    store.admin.connection.rename(rep.build_key, sysname.live_key(SYSTEM))
    try:
        yield store
    finally:
        store.admin.connection.delete(sysname.live_key(SYSTEM))


def engine_db(port):
    return L.connect(HOST, port, username=sysname.acl_user("engine", SYSTEM), password=acl.read_credentials(SYSTEM)["engine"], socket_timeout=SOCKET_TIMEOUT_S)


def client_on(port):
    """The production ``Client`` and ``CypherReader`` over a real connection to ``port`` (a short socket timeout so the hang is seen in seconds)."""
    return Client(SYSTEM, host=HOST, port=port, reader_factory=lambda system, host, p: CypherReader(engine_db(port), system))     # connects per ask, as ``for_system`` does


def canary_on(port):
    """The host-health canary on its own connection: the fixed non-held-out read at the pinned deadline."""
    graph = engine_db(port).select_graph(sysname.live_key(SYSTEM))
    return lambda: graph.ro_query(R.CANARY_REQUEST, None, round(R.STALL_DEADLINE_S * 1000))


@pytest.fixture
def black_hole():
    """A socket that accepts connections (the kernel completes the handshake) and never answers: what a hung product looks like to its client."""
    s = socket.socket()
    s.bind((HOST, 0))
    s.listen(16)
    try:
        yield s.getsockname()[1]
    finally:
        s.close()


def env(tmp_path):
    e = Env(tmp_path)
    many_world(e, 3)
    return e


def stop(e, client, canary):
    first = not e.markers()
    try:
        e.go(ask_factory=lambda b, s: R.make_ask(client, stamp=STAMP, in_graph=lambda req: True), canary=canary, resume=not first)
    except R.RunFailed as exc:
        return exc.state
    return "done"


def answers(e):
    f = list(e.cust.glob("run-*.answers.jsonl"))
    return [__import__("json").loads(l) for l in f[0].read_text().splitlines()] if f else []


def test_live_a_control_ask_on_the_healthy_store_answers_the_canary(live_store):
    canary_on(live_store.port)()                                                              # the canary itself works on this store


def test_live_the_real_hang_chain_is_a_StoreError_over_a_timeout_and_the_third_stop_converts(live_store, black_hole, tmp_path):
    from clients.common import StoreError
    client = client_on(black_hole)
    with pytest.raises(StoreError) as ei:
        client.ask({"useCase": "impact", "anchor": {"kind": "contract", "key": "GET /a0", "contractKind": "http"}})
    names = [type(c).__name__ for c in R.exception_chain(ei.value)]
    assert names[0] == "StoreError" and "TimeoutError" in names and not isinstance(ei.value, TimeoutError), names     # the outer type is never the timeout
    e = env(tmp_path)
    canary = canary_on(live_store.port)
    assert stop(e, client, canary) == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 1} and answers(e) == []
    assert stop(e, client, canary) == "failed:transient" and e.marker()["stall"] == {"index": 0, "count": 2} and answers(e) == []
    assert stop(e, client, canary) == "failed:transient"                                      # stop 3: the hung anchor is converted, the next one hangs
    first = answers(e)
    assert len(first) == 1 and {k: v for k, v in first[0].items() if k != "anchorId"} == \
        {"outcome": "engine-error", "exceptionClass": "TimeoutError", "via": "repeat-stall", "stalls": 3}


def test_live_a_paused_store_fails_the_canary_too_so_it_never_converts(live_store, tmp_path):
    name = f"ccg-falkordb-tw-{live_store.port}"
    e = env(tmp_path)
    client, canary = client_on(live_store.port), canary_on(live_store.port)
    subprocess.run(["docker", "pause", name], check=True, capture_output=True)
    try:
        for _ in range(3):
            assert stop(e, client, canary) == "failed:transient"
    finally:
        subprocess.run(["docker", "unpause", name], check=True, capture_output=True)
    assert answers(e) == [] and e.marker()["stall"] is None and not e.run_files()


def test_live_a_hang_on_an_established_connection_has_the_analysts_chain_and_is_a_stall_the_rule_may_count(live_store):
    """The pause arrives after the connection exists: the timeout is raised by ``ro_query`` inside the reader (``StoreError`` <- ``ReaderError`` <- redis
    ``TimeoutError`` <- the socket's ``TimeoutError``), the shape the offline hand-built classes never had."""
    from clients.common import StoreError
    db = engine_db(live_store.port)
    client = Client(SYSTEM, host=HOST, port=live_store.port, reader_factory=lambda system, host, p: CypherReader(db, system))
    spec = {"useCase": "impact", "anchor": {"kind": "location", "repo": "repo-a", "path": "app/a.py", "symbol": "app.Outer"}}
    assert client.resolve(spec["anchor"])["status"] == "resolved"                                   # the control: it answers while the store is up
    name = f"ccg-falkordb-tw-{live_store.port}"
    subprocess.run(["docker", "pause", name], check=True, capture_output=True)
    try:
        with pytest.raises(StoreError) as ei:
            client.ask(spec)
    finally:
        subprocess.run(["docker", "unpause", name], check=True, capture_output=True)
    names = [f"{type(c).__module__.split('.')[0]}.{type(c).__name__}" for c in R.exception_chain(ei.value)]
    assert names[0].endswith("StoreError") and names[-1] == "builtins.TimeoutError" and "redis.TimeoutError" in names, names
    assert not isinstance(ei.value, TimeoutError) and R.classify(ei.value) == "transient" and R.ask_failure_entry(ei.value) is None
    assert R.converts_on_repeat_stall(ei.value) is True
