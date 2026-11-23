"""The dress rehearsal's D1 and D4 against the REAL component, live tier (``pytest -m live``): a throwaway FalkorDB of this module's own that is the lab's
standard shape -- ``requirepass`` on the default user, the four real ACL users of ``ccg.store.acl.create_users``, ``TIMEOUT_MAX`` 10000 -- on a port from 16430
up (local image, ``--pull never``, 127.0.0.1, labels ``ccg-falkordb=1`` ``ccg-falkordb.throwaway=1``, removed afterwards; never 6379/6390/16390/16400).
The offline suite stood fakes where these failed: a credential-less ``FalkorDB(host, port)`` and a canary asking 30000 ms of a store that refuses above 10000.
The graph is a tiny synthetic one (invented ids)."""
import argparse
import os
import re
import secrets
import socket
import subprocess
import time
import types

import pytest
from engine_helpers import mini_rows
from redis.exceptions import AuthenticationError, ResponseError

import gate_helpers as H  # noqa: F401  (puts eval/tools on sys.path)
from gate_helpers import G
import gate_run as R
from ccg import sysname
from ccg.store import acl
from ccg.store import loader as L

pytestmark = pytest.mark.live
IMAGE = "falkordb/falkordb:v4.18.11"
FORBIDDEN_PORTS = {6379, 6390, 16390, 16400}
SYSTEM = "gtlive4"
HOST = "127.0.0.1"
TIMEOUT_MAX = 10_000


def _free_port():
    published = subprocess.run(["docker", "ps", "--format", "{{.Ports}}"], capture_output=True, text=True, check=True).stdout
    taken = {int(p) for p in re.findall(r":(\d+)->", published)} | FORBIDDEN_PORTS
    for port in range(16430, 16520):
        if port in taken:
            continue
        with socket.socket() as s:
            if s.connect_ex((HOST, port)) != 0:
                return port
    raise RuntimeError("no free port in 16430-16519")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    try:
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"{IMAGE} is not available locally (docker missing or image not pulled; this test never pulls)")
    from falkordb import FalkorDB
    port, admin_pw = _free_port(), secrets.token_urlsafe(16)
    name = f"ccg-falkordb-tw-{port}"
    subprocess.run(["docker", "run", "-d", "--pull", "never", "--name", name, "-p", f"{HOST}:{port}:6379", "--label", "ccg-falkordb=1",
                    "--label", "ccg-falkordb.throwaway=1", "-e", f"REDIS_ARGS=--requirepass {admin_pw}", IMAGE], capture_output=True, text=True, check=True)
    try:
        admin = None
        for _ in range(60):
            try:
                admin = FalkorDB(host=HOST, port=port, password=admin_pw)
                admin.connection.ping()
                break
            except Exception:  # noqa: BLE001 -- not up yet
                time.sleep(0.5)
        else:
            pytest.fail("throwaway FalkorDB did not come up")
        admin.connection.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", str(TIMEOUT_MAX))
        creds = tmp_path_factory.mktemp("ccg-credentials")
        os.chmod(creds, 0o700)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CCG_CREDENTIALS_DIR", str(creds))
            acl.create_users(admin, SYSTEM)
        yield types.SimpleNamespace(port=port, admin=admin, creds=creds, admin_pw=admin_pw)
    finally:
        labels = subprocess.run(["docker", "inspect", "-f", "{{index .Config.Labels \"ccg-falkordb.throwaway\"}}", name], capture_output=True, text=True).stdout.strip()
        if labels == "1":                      # only ever remove the container this fixture started
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture
def creds(store, monkeypatch):
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(store.creds))


def deps(store):
    a = argparse.Namespace(system=SYSTEM, host=HOST, port=store.port, graph=None)
    return R._default_deps(a)                                    # no injection: the real credential path, the real Client


def test_live_the_store_is_the_lab_standard_the_default_user_is_refused(store):
    from falkordb import FalkorDB
    with pytest.raises(AuthenticationError):
        FalkorDB(host=HOST, port=store.port).select_graph("x").ro_query("RETURN 1")      # D1's premise: the credential-less reader the harness used


def test_live_the_live_graph_probe_and_the_snapshot_reader_work_on_the_password_protected_store_as_a_read_only_role(store, creds):
    d = deps(store)
    assert d["live_graph_present"]() is False                                              # absent: the "empty key" answer, as a role that cannot GRAPH.LIST
    rows = mini_rows(SYSTEM)
    rep = L.load(L.LiveStore(L.connect(HOST, store.port, username=None, password=store.admin_pw), SYSTEM), rows, SYSTEM)
    store.admin.connection.rename(rep.build_key, sysname.live_key(SYSTEM))
    try:
        assert d["live_graph_present"]() is True
        snap = d["snapshot_factory"](None)
        assert snap.snapshot_id() == rows.snapshot["snapshotId"]
        g = G.snapshot_graph(HOST, store.port, None, SYSTEM)                               # what `gate.py score` opens
        with pytest.raises(Exception, match="(?i)no permissions"):                         # read-only: a write command is refused by the ACL, not by the harness
            g.query("CREATE (:X)")
        assert d["engine_snapshot_id"]() == snap.snapshot_id()                              # the engine reader and the snapshot reader agree
    finally:
        store.admin.connection.delete(sysname.live_key(SYSTEM))


def test_live_the_default_canary_answers_at_the_stores_real_timeout_max_and_the_old_literal_is_refused(store, creds):
    rows = mini_rows(SYSTEM)
    rep = L.load(L.LiveStore(L.connect(HOST, store.port, username=None, password=store.admin_pw), SYSTEM), rows, SYSTEM)
    store.admin.connection.rename(rep.build_key, sysname.live_key(SYSTEM))
    try:
        d = deps(store)
        assert round(R.STALL_DEADLINE_S * 1000) <= TIMEOUT_MAX
        out = d["canary"]()                                                                  # raises RawRefused when the deadline is above TIMEOUT_MAX
        assert out["rows"] == [[rows.snapshot["snapshotId"]]] and out["timeoutMs"] == round(R.STALL_DEADLINE_S * 1000)
        from clients.common import Client, RawRefused
        with pytest.raises(RawRefused):                                                      # the pre-fix deadline (30000 ms) on this very store
            Client(SYSTEM, host=HOST, port=store.port).raw_query(R.CANARY_REQUEST, timeout_ms=30_000)
    finally:
        store.admin.connection.delete(sysname.live_key(SYSTEM))
