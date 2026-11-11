"""P1-20, LIVE on a throwaway instance only (``-m live``): ``refresh`` over a real ``LiveStore`` and the real ``swap``.

Same plumbing and the same rules as ``test_loader_live.py``: ``cpg/infra/ccg-falkordb.sh --throwaway up`` (the script refuses 6379 and 6390),
``CCG_FALKORDB_PORT`` naming it, a scratch ``CCG_CONFIG_DIR`` holding the throwaway's ``falkordb-admin.pass`` (``~/.config/ccg`` is refused),
and ``CCG_CREDENTIALS_DIR`` pointed at a tmp dir by the ``instance`` fixture. The ACL users are the ``build`` and ``verdict`` lines of
``-p1-graph.md`` §7, read by ``loader_helpers.acl_setuser_lines()`` (the test-local grant helper; ``ccg.store.acl.create_users`` of P1-19 dispatch 2
replaces it once the gate has accepted that unit). The refresh loads and swaps under those two users, never as admin; the work area, the bundles and
the HMAC key live under ``tmp_path``.
**A gate sets ``CCG_REQUIRE_LIVE=1``** (``CCG_REQUIRE_LIVE=1 pytest -m live tests/test_refresh_live.py``, and asserts ``0 skipped``): with it a missing
instance or configuration is a FAILURE, not the default skip that exits 0. A test removes only the keys of its own random system name.
"""
from __future__ import annotations

import pytest
from loader_helpers import EMPTY_HASH
from test_loader_live import env, instance  # noqa: F401  (fixtures: the throwaway, and one system's ACL users and cleanup)
from test_refresh import ALL_UNITS, World, commit, failing, git, uid

from ccg import refresh as RF
from ccg import sysname
from ccg.store import loader as L


def _target(env):
    return RF.Target(L.LiveStore(env.build_db, env.system), env.build_db, env.verdict_db)


@pytest.fixture
def world(tmp_path, env):
    return World(tmp_path, system=env.system)


def _units(env, key):
    g = env.admin.select_graph(key)
    return {r[0]: r[1:] for r in g.ro_query(
        "MATCH (u:Unit) RETURN u.unitId, u.buildStatus, u.failReason, u.builtCommit ORDER BY u.unitId").result_set}


def _count(env, key, cypher):
    return env.admin.select_graph(key).ro_query(cypher).result_set[0][0]


@pytest.mark.live
def test_a_refresh_loads_and_swaps_the_assembled_graph_onto_the_live_key(world, env):
    rep = RF.refresh(env.system, **world.kw(_target(env)))
    assert rep.swap.renamed and rep.swap.released and rep.load.build_key == sysname.build_key(env.system, rep.snapshot_id)
    assert not env.admin.connection.exists(rep.load.build_key) and env.admin.connection.exists(env.live)
    head = git(world.repo(), "rev-parse", "HEAD")
    assert _units(env, env.live) == {uid(*u, repo="r1"): ["fresh", None, head] for u in ALL_UNITS}
    assert _count(env, env.live, "MATCH (s:Snapshot) RETURN count(s)") == 1
    assert env.admin.select_graph(env.live).ro_query("MATCH (s:Snapshot) RETURN s.snapshotId, s.systemId, s.verdictLogOffset, s.verdictLogHash"
                                                     ).result_set == [[rep.snapshot_id, env.system, 0, EMPTY_HASH]]
    assert _count(env, env.live, "MATCH (n:ContractPoint) RETURN count(n)") == 0        # host-only fixture: nothing links, nothing breaks


@pytest.mark.live
def test_a_failed_unit_is_stale_kept_in_the_live_graph_and_a_never_built_one_is_not_mapped(world, env):
    RF.refresh(env.system, **world.kw(_target(env)))
    c2 = commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n", "extra.sh": "#!/bin/sh\n"})
    victim, never = uid("conf/app.yaml", "config"), uid("extra.sh", "shell")
    rep = RF.refresh(env.system, **world.kw(_target(env), now=2_000_000_000.0,
                                              build=failing((victim, "extract:timeout"), (never, "extract:failed"))))
    got = _units(env, env.live)
    head1 = next(u.built_commit for u in rep.stale_kept)
    assert got[victim] == ["stale-kept", "extract:timeout", head1] and head1 != c2
    assert got[never] == ["not-mapped", "extract:failed", None]
    assert got[uid("bin/run.sh", "shell")][0] == "fresh"
    row = env.admin.select_graph(env.live).ro_query("MATCH (u:Unit {unitId: $u}) RETURN u.failedAt", {"u": victim}).result_set
    assert row == [["2033-05-18T03:33:20Z"]]


@pytest.mark.live
def test_a_refresh_that_cannot_assemble_leaves_the_live_graph_untouched(world, env):
    g = env.admin.select_graph(env.live)
    g.query("CREATE (:Marker {v: 'old'}), (:Snapshot {snapshotId: 'old', verdictLogOffset: 0, verdictLogHash: $h})", {"h": EMPTY_HASH})
    with pytest.raises(RF.RefreshError, match="nothing to load"):
        RF.refresh(env.system, **world.kw(_target(env), build=failing(*[(uid(*u), "extract:failed") for u in ALL_UNITS])))
    assert _count(env, env.live, "MATCH (m:Marker) RETURN count(m)") == 1
