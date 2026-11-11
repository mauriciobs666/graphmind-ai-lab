"""P1-19 dispatch 1, LIVE on a throwaway instance only (``infra/ccg-falkordb.sh --throwaway up``; ``-m live``).

Needs ``CCG_FALKORDB_PORT`` (the throwaway's port; 6379 and 6390 are refused, and the port must be published by a container labelled
``ccg-falkordb.throwaway=1``) and ``CCG_CONFIG_DIR`` (the scratch directory holding the throwaway's ``falkordb-admin.pass``; ``~/.config/ccg`` is
refused). It also sets ``CCG_CREDENTIALS_DIR`` to a tmp dir, so nothing under ``~/.config/ccg`` is read or written. The ACL users it creates
live and die inside the throwaway: ``ccg.store.acl.create_users`` makes the four ``-p1-graph.md`` §7 users (``tests/test_acl_live.py`` asserts the grant
matrix); the load and the swap run under them, never as admin.
**A gate sets ``CCG_REQUIRE_LIVE=1``** (``CCG_REQUIRE_LIVE=1 pytest -m live tests/test_loader_live.py``, and asserts ``0 skipped``): a missing instance,
configuration or docker CLI is then a FAILURE (an error per test), not the default skip that exits 0.
Contract: ``-p1-graph.md`` §4, §7, §8 P1-19 (a), (b); ``-p1-linking.md`` P1-19 behaviours (1), (5), (6)."""
from __future__ import annotations

import itertools
import multiprocessing
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from loader_helpers import EMPTY_HASH, FORBIDDEN_PORTS, admin_password, skip_or_fail, throwaway_port, tiny_rows
from test_ddl_live import EXPECTED_CONSTRAINT_PAIRS, EXPECTED_INDEX_PAIRS

from ccg import sysname
from ccg.store import acl
from ccg.store import loader as L
from ccg.store import swap as SW

HOST = "127.0.0.1"
_counter = itertools.count(1)


# ------------------------------------------------------------------------------------- pure guards (no network)

def test_the_live_tests_refuse_the_shared_ports(monkeypatch):
    for port in FORBIDDEN_PORTS:
        monkeypatch.setenv("CCG_FALKORDB_PORT", str(port))
        assert throwaway_port() is None


def test_connect_refuses_falkordb_dev_before_any_socket():
    with pytest.raises(L.LoadError, match="6379"):
        L.connect(HOST, 6379, username=None, password="x")


def test_the_store_refuses_a_key_it_does_not_own_before_touching_the_connection():
    class Sentinel:
        def __getattr__(self, name):
            raise AssertionError("the connection was used")

    store = L.LiveStore(Sentinel(), "shop")
    for key in ("ccg_shop", "ccg_shop2", "shop", "ccg_shop__x", "ccg_shop___build_1", "ccg_shop2__build_1"):
        with pytest.raises(L.LoadError, match="not owned"):
            store.exists(key)
        with pytest.raises(L.LoadError, match="not owned"):
            store.delete(key)
        with pytest.raises(L.LoadError, match="not owned"):
            store.query(key, "RETURN 1")


def test_require_live_turns_the_no_instance_skip_into_a_failure(monkeypatch):
    monkeypatch.delenv("CCG_REQUIRE_LIVE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        skip_or_fail("no instance")
    monkeypatch.setenv("CCG_REQUIRE_LIVE", "0")
    with pytest.raises(pytest.skip.Exception):
        skip_or_fail("no instance")
    monkeypatch.setenv("CCG_REQUIRE_LIVE", "1")
    with pytest.raises(pytest.fail.Exception, match="CCG_REQUIRE_LIVE=1.*no instance"):
        skip_or_fail("no instance")


def _run_live_module_without_an_instance(require: bool) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("CCG_FALKORDB_PORT", "CCG_CONFIG_DIR", "CCG_REQUIRE_LIVE")}
    if require:
        env["CCG_REQUIRE_LIVE"] = "1"
    cwd = Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, "-m", "pytest", "-m", "live", "tests/test_loader_live.py", "-q", "-p", "no:cacheprovider"],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)


def test_the_gate_command_with_no_instance_skips_and_exits_zero_without_the_switch_and_fails_with_it():
    """End to end on this very module, with no throwaway configured (so no network is touched)."""
    quiet = _run_live_module_without_an_instance(require=False)
    assert quiet.returncode == 0 and re.search(r"\d+ skipped", quiet.stdout) and "failed" not in quiet.stdout.splitlines()[-1]
    strict = _run_live_module_without_an_instance(require=True)
    last = strict.stdout.strip().splitlines()[-1]
    assert strict.returncode != 0 and "skipped" not in last and re.search(r"\d+ error", last), last


# ------------------------------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def _credentials_dir(tmp_path, monkeypatch):
    """Every test: ACL credentials in a tmp dir, never ``~/.config/ccg``."""
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(tmp_path / "cred"))


@pytest.fixture(scope="module")
def instance(tmp_path_factory):
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance (or the docker CLI cannot start): start one with cpg/infra/ccg-falkordb.sh --throwaway up and "
                     "export CCG_FALKORDB_PORT and CCG_CONFIG_DIR (a scratch dir)")
    try:
        pw = admin_password()
    except FileNotFoundError as exc:
        skip_or_fail(f"throwaway admin password not readable ({exc})")
    admin = L.connect(HOST, port, username=None, password=pw)
    return SimpleNamespace(port=port, admin=admin)


@pytest.fixture
def env(instance):
    from ccg import atrest                                  # another module may import this fixture without the autouse one above
    real = Path.home() / ".config" / "ccg"
    cred = Path(atrest.credentials_dir()).resolve()
    if cred == real.resolve() or real.resolve() in cred.parents:
        pytest.fail("CCG_CREDENTIALS_DIR is unset or names ~/.config/ccg: this fixture creates ACL users and writes their credentials file", pytrace=False)
    system = f"tl{next(_counter)}x{os.getpid() % 1000}"
    admin = instance.admin
    users = acl.create_users(admin, system)                  # the four §7 users, passwords through ``ccg.atrest.write_acl``
    pw = acl.read_credentials(system)
    build_db = L.connect(HOST, instance.port, username=sysname.acl_user("build", system), password=pw["build"])
    verdict_db = L.connect(HOST, instance.port, username=sysname.acl_user("verdict", system), password=pw["verdict"])
    live = sysname.live_key(system)
    e = SimpleNamespace(system=system, admin=admin, build_db=build_db, verdict_db=verdict_db, live=live, lock=sysname.lock_key(system),
                        port=instance.port)
    e.cleanup_keys = [live, e.lock]
    try:
        yield e
    finally:
        pre = sysname.build_key(system, "x")[:-1]
        keys = set(e.cleanup_keys) | {k.decode() if isinstance(k, bytes) else k for k in admin.connection.scan_iter(match=f"{live}*")}
        keys |= {k.decode() if isinstance(k, bytes) else k for k in admin.connection.scan_iter(match=f"{pre}*")}
        for key in keys:
            admin.connection.delete(key)
        for user in users:
            admin.connection.execute_command("ACL", "DELUSER", user)


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def _plant_live(env, marker="old"):
    """A live graph the way a previous swap left it: one Marker node and the empty-log Snapshot."""
    g = env.admin.select_graph(env.live)
    g.query("CREATE (:Marker {v: $v}), (:Snapshot {snapshotId: 'old', verdictLogOffset: 0, verdictLogHash: $h})", {"v": marker, "h": EMPTY_HASH})
    return g


def _count(g, cypher):
    return g.ro_query(cypher).result_set[0][0]


# ------------------------------------------------------------------------------------- (a) the load, every group

@pytest.mark.live
def test_a_full_load_under_the_build_credential_writes_every_group_and_the_ddl_set(env):
    rows = tiny_rows(env.system)
    store = L.LiveStore(env.build_db, env.system)
    report = L.load(store, rows, env.system)
    bkey = sysname.build_key(env.system, rows.snapshot["snapshotId"])
    assert report.build_key == bkey and not env.admin.connection.exists(env.live)         # the loader never touches live
    g = env.admin.select_graph(bkey)
    for label, label_rows in rows.nodes.items():
        assert _count(g, f"MATCH (n:{label}) RETURN count(n)") == len(label_rows), label
    assert _count(g, "MATCH (n:Snapshot) RETURN count(n)") == 1
    for (src, etype, dst), edge_rows in rows.edges.items():
        n = _count(g, f"MATCH (:{src})-[e:{etype}]->(:{dst}) RETURN count(e)")
        assert n == len(edge_rows), (src, etype, dst)
    assert _count(g, "MATCH ()-[e]->() RETURN count(e)") == rows.edge_count()
    # G6: two MOUNTED_BY edges between one Symbol and CallSite, distinguished by ord and mountPath
    got = g.ro_query("MATCH (:Symbol {symbolId: 'Symbol1'})-[e:MOUNTED_BY]->(:CallSite {callSiteId: 'CallSite1'}) "
                     "RETURN e.ord, e.mountPath ORDER BY e.ord").result_set
    assert got == [[0, "/a"], [1, "/b"]]
    # D8: an edge without status props stays without them
    assert g.ro_query("MATCH (:CallSite {callSiteId: 'CallSite2'})-[e:DATA]->() RETURN e.status").result_set == [[None]]
    assert g.ro_query("MATCH (u:Unit) RETURN u.unitId, u.flowAvailable ORDER BY u.unitId").result_set == [["Unit1", True], ["Unit2", False]]
    snap = g.ro_query("MATCH (s:Snapshot) RETURN s.snapshotId, s.systemId, s.verdictLogOffset, s.verdictLogHash").result_set
    assert snap == [[rows.snapshot["snapshotId"], env.system, 0, EMPTY_HASH]]
    # (a): db.indexes() equals the DDL set; every constraint OPERATIONAL
    idx = set()
    for r in g.ro_query("CALL db.indexes()").result_set:
        assert r[6] == "NODE" and r[7] == "OPERATIONAL" and all(r[2][p] == ["RANGE"] for p in r[1]), r
        idx |= {(r[0], p) for p in r[1]}
    assert idx == EXPECTED_INDEX_PAIRS and len(idx) == 20
    cons = g.ro_query("CALL db.constraints()").result_set
    assert {(c[1], c[2][0]) for c in cons} == EXPECTED_CONSTRAINT_PAIRS and len(cons) == 12 and all(c[4] == "OPERATIONAL" for c in cons)
    assert report.timeout_max == 10000 and report.query_timeout_used and report.node_total == rows.node_count() + 1
    env.cleanup_keys.append(bkey)


@pytest.mark.live
def test_the_constraint_really_enforces_after_the_load(env):
    rows = tiny_rows(env.system)
    L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
    bkey = sysname.build_key(env.system, rows.snapshot["snapshotId"])
    with pytest.raises(Exception, match="(?i)constraint|unique"):
        env.admin.select_graph(bkey).query("CREATE (:ConfigEntry {entryId: 'ConfigEntry1'})")


@pytest.mark.live
def test_the_connection_carries_a_zero_retry_object_not_the_client_default(env):
    kwargs = L.LiveStore(env.build_db, env.system).db.connection.connection_pool.connection_kwargs
    assert kwargs["retry"] is not None and kwargs["retry"]._retries == 0


@pytest.mark.live
def test_the_load_works_under_a_nonzero_timeout_max_and_reports_it(env):
    env.admin.connection.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", 3000)
    try:
        rows = tiny_rows(env.system)
        report = L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
        assert report.timeout_max == 3000 and report.query_timeout_used
        env.cleanup_keys.append(report.build_key)
    finally:
        env.admin.connection.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", 10000)


@pytest.mark.live
def test_a_multi_batch_group_splits_at_batch_max_and_every_batch_is_count_checked(env):
    rows = tiny_rows(env.system)
    rows.nodes["CallSite"] = [{"callSiteId": f"CallSite{i}"} for i in range(1, 2001)]
    rows.edges[("CallSite", "DATA", "CallSite")] = [{"s": f"CallSite{1 + i % 2000}", "t": f"CallSite{1 + (i * 7 + 1) % 2000}"} for i in range(45_000)]
    report = L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
    g = env.admin.select_graph(report.build_key)
    assert _count(g, "MATCH (:CallSite)-[e:DATA]->(:CallSite) RETURN count(e)") == 45_000
    assert report.max_batch == 20_000 and report.edges["CallSite-DATA->CallSite"] == 45_000
    env.cleanup_keys.append(report.build_key)


@pytest.mark.live
def test_a_duplicate_node_id_that_an_edge_references_is_caught_by_the_created_count_check(env):
    rows = tiny_rows(env.system)
    rows.nodes["Link"].append({"linkId": "Link1", "tag": "dup"})
    with pytest.raises(L.LoadError, match=r"CallSite-\[VIA\]->Link: 1 rows but 2 edges"):
        L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
    assert not env.admin.connection.exists(sysname.build_key(env.system, rows.snapshot["snapshotId"]))


@pytest.mark.live
def test_a_planted_edge_with_a_missing_endpoint_fails_the_build_and_removes_the_build_key(env):
    rows = tiny_rows(env.system)
    rows.edges[("Unit", "IN_REPO", "Repo")] = [{"s": "Unit1", "t": "Repo1"}, {"s": "Unit2", "t": "NoSuchRepo"}]
    bkey = sysname.build_key(env.system, rows.snapshot["snapshotId"])
    with pytest.raises(L.LoadError, match=r"Unit-\[IN_REPO\]->Repo: 2 rows but 1 edges"):
        L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
    assert not env.admin.connection.exists(bkey)                      # removed with the build credential's +del


@pytest.mark.live
def test_a_constraint_violation_in_the_data_fails_the_build_and_removes_the_key(env):
    rows = tiny_rows(env.system)
    # an id no edge references twice: a duplicate of a referenced one is caught earlier, by the created-count check (two matches, two edges)
    rows.nodes["ContractPoint"] += [{"cpId": "cpDup"}, {"cpId": "cpDup"}]
    with pytest.raises(L.LoadError, match=r"FAILED.*ContractPoint"):
        L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
    assert not env.admin.connection.exists(sysname.build_key(env.system, rows.snapshot["snapshotId"]))


# ------------------------------------------------------------------------------------- (5) kill and refusal

class _KillAfter:
    """A store that SIGKILLs its own process at the n-th query: a kill mid-load, nothing in ``finally`` runs."""

    def __init__(self, inner, n):
        self.inner, self.n, self.i = inner, n, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def query(self, *a, **k):
        self.i += 1
        if self.i == self.n:
            os.kill(os.getpid(), signal.SIGKILL)
        return self.inner.query(*a, **k)


def _child_load(port, system, n):
    db = L.connect(HOST, port, username=sysname.acl_user("build", system), password=acl.read_credentials(system)["build"])
    L.load(_KillAfter(L.LiveStore(db, system), n), tiny_rows(system), system)


@pytest.mark.live
def test_a_mid_load_kill_leaves_the_live_key_intact_and_the_build_key_removable(env):
    _plant_live(env)
    rows = tiny_rows(env.system)
    bkey = sysname.build_key(env.system, rows.snapshot["snapshotId"])
    p = multiprocessing.get_context("fork").Process(target=_child_load, args=(env.port, env.system, 60))
    p.start()
    p.join(60)
    assert p.exitcode == -signal.SIGKILL
    # live untouched
    g = env.admin.select_graph(env.live)
    assert g.ro_query("MATCH (m:Marker) RETURN m.v").result_set == [["old"]]
    assert g.ro_query("MATCH (s:Snapshot) RETURN s.snapshotId").result_set == [["old"]]
    # the build key is there, incomplete (no Snapshot), and the build credential removes it (and only it)
    assert env.admin.connection.exists(bkey)
    assert _count(env.admin.select_graph(bkey), "MATCH (s:Snapshot) RETURN count(s)") == 0
    assert env.build_db.connection.delete(bkey) == 1 and not env.admin.connection.exists(bkey)
    assert env.admin.select_graph(env.live).ro_query("MATCH (m:Marker) RETURN count(m)").result_set == [[1]]


@pytest.mark.live
def test_a_swap_of_the_incomplete_build_a_kill_left_is_refused_and_live_stays(env):
    _plant_live(env)
    rows = tiny_rows(env.system)
    bkey = sysname.build_key(env.system, rows.snapshot["snapshotId"])
    p = multiprocessing.get_context("fork").Process(target=_child_load, args=(env.port, env.system, 60))
    p.start()
    p.join(60)
    env.cleanup_keys.append(bkey)
    with pytest.raises(SW.SwapError, match="Snapshot"):
        SW.swap(env.build_db, env.verdict_db, env.system, rows.snapshot["snapshotId"])
    assert env.admin.select_graph(env.live).ro_query("MATCH (m:Marker) RETURN m.v").result_set == [["old"]]
    assert not env.admin.connection.exists(env.lock)


@pytest.mark.live
def test_foreign_keys_beside_this_system_are_never_touched_by_a_load_and_swap(env):
    """``<s>2``, ``<s>__x``, ``<s>_`` and a bare falkor-chat style ``shop``: planted by admin, content compared afterwards."""
    s, admin = env.system, env.admin
    foreign = [f"ccg_{s}2", f"ccg_{s}__x", f"ccg_{s}_", f"ccg_{s}2__build_{'a' * 8}", "shop_p119a_neighbour"]
    for k in foreign:
        admin.select_graph(k).query("CREATE (:Marker {v: $v})", {"v": k})
        env.cleanup_keys.append(k)
    rows = tiny_rows(s)
    report = L.load(L.LiveStore(env.build_db, s), rows, s)
    SW.swap(env.build_db, env.verdict_db, s, rows.snapshot["snapshotId"])
    for k in foreign:
        assert admin.select_graph(k).ro_query("MATCH (m:Marker) RETURN m.v").result_set == [[k]]
    assert report.build_key not in {k.decode() if isinstance(k, bytes) else k for k in admin.connection.scan_iter(match=f"ccg_{s}__build_*")}


# ------------------------------------------------------------------------------------- the swap, under the real grants

@pytest.mark.live
def test_load_then_swap_replaces_live_keeps_no_old_snapshot_and_releases_the_lock(env):
    _plant_live(env)
    rows = tiny_rows(env.system)
    L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
    # negative control: the build credential holds live write-only, so the grants this test runs under are the real ones
    with pytest.raises(Exception, match="(?i)permission|noperm"):
        env.build_db.select_graph(env.live).query("RETURN 1")
    res = SW.swap(env.build_db, env.verdict_db, env.system, rows.snapshot["snapshotId"])
    assert res.renamed and res.released and res.replayed_before == 0 and res.replayed_after == 0
    g = env.admin.select_graph(env.live)
    assert g.ro_query("MATCH (s:Snapshot) RETURN s.snapshotId, s.verdictLogOffset, s.verdictLogHash").result_set == [
        [rows.snapshot["snapshotId"], 0, EMPTY_HASH]]
    assert _count(g, "MATCH (m:Marker) RETURN count(m)") == 0                     # the old live is gone
    assert _count(g, "MATCH (n:Symbol) RETURN count(n)") == 2
    left = {k.decode() if isinstance(k, bytes) else k for k in env.admin.connection.scan_iter(match=f"ccg_{env.system}*")}
    assert left == {env.live}                                                      # no build key, no prev, no lock


@pytest.mark.live
def test_the_first_build_swaps_onto_a_system_with_no_live_key(env):
    rows = tiny_rows(env.system)
    L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
    assert not env.admin.connection.exists(env.live)
    assert SW.swap(env.build_db, env.verdict_db, env.system, rows.snapshot["snapshotId"]).renamed
    assert _count(env.admin.select_graph(env.live), "MATCH (n:Snapshot) RETURN count(n)") == 1


@pytest.mark.live
def test_a_held_lock_blocks_the_swap_on_the_real_server_and_is_left_alone(env):
    rows = tiny_rows(env.system)
    L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
    env.admin.connection.set(env.lock, "holder", px=30_000)
    with pytest.raises(SW.SwapError, match="lock"):
        SW.swap(env.build_db, env.verdict_db, env.system, rows.snapshot["snapshotId"], wait_s=0)
    assert _s(env.admin.connection.get(env.lock)) == "holder" and not env.admin.connection.exists(env.live)
    env.cleanup_keys.append(sysname.build_key(env.system, rows.snapshot["snapshotId"]))


@pytest.mark.live
def test_the_release_on_the_real_server_never_deletes_a_lock_another_holder_took(env):
    rows = tiny_rows(env.system)
    L.load(L.LiveStore(env.build_db, env.system), rows, env.system)
    res = SW.swap(env.build_db, env.verdict_db, env.system, rows.snapshot["snapshotId"],
                  hook=lambda name: env.admin.connection.set(env.lock, "other", px=30_000) if name == "before_release" else None)
    assert res.released is False and _s(env.admin.connection.get(env.lock)) == "other"


@pytest.mark.live
def test_the_lock_expires_by_its_ttl_when_a_swap_dies_holding_it(env):
    """A swap that dies holding the lock does not wedge the system: the lock is ``SET NX PX``, so it lapses and the next swap proceeds."""
    rows = tiny_rows(env.system)
    L.load(L.LiveStore(env.build_db, env.system), rows, env.system)

    def die(name):
        if name == "locked":
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        SW.swap(env.build_db, env.verdict_db, env.system, rows.snapshot["snapshotId"], ttl_ms=300, hook=die)
    assert not env.admin.connection.exists(env.lock)                               # the finally released it
    env.admin.connection.set(env.lock, "dead-holder", px=300)                      # what a SIGKILL would have left
    assert SW.swap(env.build_db, env.verdict_db, env.system, rows.snapshot["snapshotId"], wait_s=3).renamed
