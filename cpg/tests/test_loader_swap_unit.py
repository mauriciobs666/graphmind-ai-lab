"""P1-19 dispatch 1: ``ccg/store/swap.py`` against fake connections: the lock, the pre-``RENAME`` replay as the build credential, the plain
``RENAME build live``, the post-``RENAME`` check and replay as the verdict credential (never admin), release by compare-and-delete,
zero kept snapshots. Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §7, §8 P1-19 (c); ``-p0-graph.md`` §6 rows 8, 9, 11."""
from __future__ import annotations

import pytest
from loader_helpers import EMPTY_HASH, fake_pair

from ccg.store import swap as SW
from ccg.sysname import SystemNameError

SYSTEM, BUILD_ID = "shop", "abc123"
LIVE, BUILD, LOCK = "ccg_shop", "ccg_shop__build_abc123", "ccg_shop:verdict_lock"


def _pair(*, live_present=True, build_present=True, build_stamp=(0, EMPTY_HASH), foreign=()):
    keys = {k: "x" for k in foreign}
    if live_present:
        keys[LIVE] = "old"
    if build_present:
        keys[BUILD] = "new"
    b, v, log, shared, stamps = fake_pair(SW.RELEASE_LUA, keys=keys)
    if build_present and build_stamp is not None:
        stamps[BUILD] = build_stamp
    if live_present:
        stamps[LIVE] = (0, EMPTY_HASH)
    return b, v, log, shared, stamps


def _ops(log, who=None, op=None):
    return [e for e in log if (who is None or e[0] == who) and (op is None or e[1] == op)]


def test_the_happy_path_order_and_who_does_what():
    b, v, log, shared, stamps = _pair()
    replays = []
    res = SW.swap(b, v, SYSTEM, BUILD_ID, replay=lambda conn, key, system: replays.append((conn.name, key, system)) or 0)
    shape = [(e[0], e[1]) + ((e[2],) if e[1] in ("ro_query", "query") else ()) for e in log]
    assert shape == [("build", "set"), ("build", "exists"), ("build", "ro_query", BUILD), ("build", "get"),
                     ("build", "rename"), ("build", "get"), ("verdict", "query", LIVE), ("build", "eval")]
    assert replays == [("build", BUILD, SYSTEM), ("verdict", LIVE, SYSTEM)]    # pre on the build key as build; post on live as verdict
    assert _ops(log, op="rename") == [("build", "rename", BUILD, LIVE)]
    assert LOCK not in shared and BUILD not in shared and shared[LIVE] == "new"
    assert (res.released, res.renamed, res.replayed_before, res.replayed_after) == (True, True, 0, 0)
    assert res.lock_held_after_rename is True


def test_the_lock_is_set_nx_with_a_ttl_and_its_value_is_the_token_released_by_compare_and_delete():
    b, v, log, shared, _ = _pair()
    SW.swap(b, v, SYSTEM, BUILD_ID, ttl_ms=1234)
    (_, _, key, token, nx, px), = _ops(log, "build", "set")
    assert (key, nx, px) == (LOCK, True, 1234) and len(token) >= 16
    (_, _, n, k, tok), = _ops(log, "build", "eval")
    assert (n, k, tok) == (1, LOCK, token)
    assert SW.RELEASE_LUA == "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"
    assert 0 < SW.LOCK_TTL_MS <= 10_000


def test_two_swaps_hold_different_tokens():
    tokens = []
    for _ in range(2):
        b, v, log, _, _ = _pair()
        SW.swap(b, v, SYSTEM, BUILD_ID)
        tokens.append(_ops(log, "build", "set")[0][3])
    assert tokens[0] != tokens[1]


def test_a_held_lock_refuses_the_swap_with_no_rename_and_leaves_the_holders_lock():
    b, v, log, shared, _ = _pair()
    shared[LOCK] = "someone-else"
    with pytest.raises(SW.SwapError, match="lock"):
        SW.swap(b, v, SYSTEM, BUILD_ID, wait_s=0)
    assert not _ops(log, op="rename") and shared[LOCK] == "someone-else" and shared[BUILD] == "new"
    assert not _ops(log, op="eval")           # never held it, so never releases it


def test_the_lock_is_always_released_when_a_step_fails_and_the_original_error_surfaces():
    class Boom(RuntimeError):
        pass

    def boom(conn, key, system):
        raise Boom("replay")

    b, v, log, shared, _ = _pair()
    with pytest.raises(Boom):
        SW.swap(b, v, SYSTEM, BUILD_ID, replay=boom)
    assert LOCK not in shared and not _ops(log, op="rename") and shared[BUILD] == "new" and shared[LIVE] == "old"

    b, v, log, shared, _ = _pair()
    with pytest.raises(SW.SwapError):
        SW.swap(b, v, SYSTEM, BUILD_ID, hook=lambda name: (_ for _ in ()).throw(SW.SwapError("planted")) if name == "before_rename" else None)
    assert LOCK not in shared and not _ops(log, op="rename")


def test_a_release_never_deletes_another_holders_lock():
    """The lock expired and another process took it mid-swap: our compare-and-delete must leave theirs and report it."""
    b, v, log, shared, _ = _pair()
    res = SW.swap(b, v, SYSTEM, BUILD_ID, hook=lambda name: shared.__setitem__(LOCK, "other") if name == "before_release" else None)
    assert shared[LOCK] == "other" and res.released is False and res.renamed is True


def test_a_lock_lost_before_the_rename_aborts_with_nothing_renamed():
    b, v, log, shared, _ = _pair()
    with pytest.raises(SW.SwapError, match="lost before RENAME"):
        SW.swap(b, v, SYSTEM, BUILD_ID, hook=lambda name: shared.pop(LOCK) if name == "before_rename" else None)
    assert not _ops(log, op="rename") and shared[BUILD] == "new" and shared[LIVE] == "old"


def test_a_missing_build_key_is_refused_and_the_lock_released():
    b, v, log, shared, _ = _pair(build_present=False)
    with pytest.raises(SW.SwapError, match="build key"):
        SW.swap(b, v, SYSTEM, BUILD_ID)
    assert LOCK not in shared and shared[LIVE] == "old" and not _ops(log, op="rename")


def test_a_build_without_its_snapshot_is_incomplete_and_refused():
    b, v, log, shared, _ = _pair(build_stamp=None)
    with pytest.raises(SW.SwapError, match="Snapshot"):
        SW.swap(b, v, SYSTEM, BUILD_ID)
    assert LOCK not in shared and shared[LIVE] == "old" and not _ops(log, op="rename")


@pytest.mark.parametrize("stamp", [(1, EMPTY_HASH), (0, "deadbeef"), (None, None)])
def test_a_build_stamp_other_than_the_empty_log_is_refused_before_the_rename(stamp):
    b, v, log, shared, _ = _pair(build_stamp=stamp)
    with pytest.raises(SW.SwapError, match="verdict log"):
        SW.swap(b, v, SYSTEM, BUILD_ID)
    assert not _ops(log, op="rename") and LOCK not in shared


def test_a_live_stamp_that_differs_after_the_rename_is_a_post_rename_failure_that_still_releases():
    b, v, log, shared, stamps = _pair()
    with pytest.raises(SW.SwapError, match="after RENAME") as info:
        SW.swap(b, v, SYSTEM, BUILD_ID, hook=lambda name: stamps.__setitem__(LIVE, (3, EMPTY_HASH)) if name == "after_rename" else None)
    assert info.value.renamed is True and LOCK not in shared and shared[LIVE] == "new"


def test_the_pre_rename_failure_carries_renamed_false():
    b, v, log, shared, _ = _pair(build_stamp=(9, EMPTY_HASH))
    with pytest.raises(SW.SwapError) as info:
        SW.swap(b, v, SYSTEM, BUILD_ID)
    assert info.value.renamed is False


def test_the_post_rename_work_is_never_done_by_the_build_credential_and_never_by_admin():
    b, v, log, _, _ = _pair()
    SW.swap(b, v, SYSTEM, BUILD_ID)
    after = log[[i for i, e in enumerate(log) if e[1] == "rename"][0] + 1:]
    graph_ops_on_live = [e for e in after if e[1] in ("query", "ro_query") and e[2] == LIVE]
    assert graph_ops_on_live and all(e[0] == "verdict" for e in graph_ops_on_live)
    assert not [e for e in log if e[0] == "build" and e[1] in ("query", "ro_query") and e[2] == LIVE]   # build never reads live
    assert not [e for e in log if e[1] in ("delete",)]                                                 # nobody DELs anything
    # the module takes exactly two connections: there is no third, admin-shaped, parameter
    import inspect
    assert list(inspect.signature(SW.swap).parameters)[:4] == ["build_conn", "verdict_conn", "system", "build_id"]


def test_zero_old_snapshots_are_kept_and_no_key_outside_live_build_lock_is_touched():
    b, v, log, shared, _ = _pair(foreign=["ccg_shop2", "ccg_shop2__build_abc123", "ccg_shop__x", "shop"])
    SW.swap(b, v, SYSTEM, BUILD_ID)
    touched = {e[2] for e in log if len(e) > 2 and isinstance(e[2], str) and e[1] in ("set", "get", "exists", "rename", "ro_query", "query")}
    touched |= {e[3] for e in log if e[1] == "rename"}
    assert touched <= {LIVE, BUILD, LOCK}
    assert set(shared) == {LIVE, "ccg_shop2", "ccg_shop2__build_abc123", "ccg_shop__x", "shop"}     # old live replaced, nothing kept as prev


@pytest.mark.parametrize("name", ["shop_", "shop__x", "Shop", "", "a" * 49, "shop*", "ccg__shop"])
def test_an_invalid_system_name_is_refused_before_any_connection_use(name):
    b, v, log, _, _ = _pair()
    with pytest.raises(SystemNameError):
        SW.swap(b, v, name, BUILD_ID)
    assert log == []


@pytest.mark.parametrize("build_id", ["", "a:b", "a*", "a__b", "a b", "-a", "a\n", "../x", None])
def test_an_invalid_build_id_is_refused_before_any_connection_use(build_id):
    b, v, log, _, _ = _pair()
    with pytest.raises((ValueError, TypeError)):
        SW.swap(b, v, SYSTEM, build_id)
    assert log == []


def test_the_empty_log_stamp_constants_equal_the_assemblers_and_the_documented_literal():
    from ccg.assemble import snapshot as SN
    assert SW.EMPTY_VERDICT_LOG_HASH == SN.EMPTY_VERDICT_LOG_HASH == EMPTY_HASH and SW.VERDICT_LOG_OFFSET == SN.VERDICT_LOG_OFFSET == 0


def test_the_verdict_replay_hook_is_a_no_op_returning_zero_and_swap_uses_the_module_hook_by_default(monkeypatch):
    b, v, log, _, _ = _pair()
    assert SW.replay_verdicts(b, BUILD, SYSTEM) == 0 and SW.replay_verdicts(v, LIVE, SYSTEM) == 0
    assert log == []                                  # it does not touch a connection in P1
    calls = []
    monkeypatch.setattr(SW, "replay_verdicts", lambda conn, key, system: calls.append((conn.name, key)) or 0)
    SW.swap(b, v, SYSTEM, BUILD_ID)
    assert calls == [("build", BUILD), ("verdict", LIVE)]


def test_the_first_build_has_no_live_key_and_swaps_all_the_same():
    b, v, log, shared, _ = _pair(live_present=False)
    res = SW.swap(b, v, SYSTEM, BUILD_ID)
    assert shared[LIVE] == "new" and BUILD not in shared and res.renamed


def test_the_result_carries_the_phase_times_the_ttl_rule_needs():
    b, v, _, _, _ = _pair()
    import itertools
    ticks = itertools.count()                      # every clock() call advances 1.0: each phase then reads a positive, whole figure
    res = SW.swap(b, v, SYSTEM, BUILD_ID, clock=lambda: float(next(ticks)))
    assert res.replay_s > 0 and res.rename_s > 0 and res.remainder_s > 0 and res.total_s > res.replay_s and res.ttl_ms == SW.LOCK_TTL_MS


def test_the_stamp_reads_carry_the_callers_timeout_and_none_by_default():
    for given in (None, 7_500):
        b, v, log, shared, _ = _pair()
        SW.swap(b, v, SYSTEM, BUILD_ID, **({} if given is None else {"timeout": given}))
        assert b.timeouts == [("ro_query", BUILD, given)] and v.timeouts == [("query", LIVE, given)]


def test_a_lock_lost_after_the_rename_is_reported_not_fatal_and_the_remainder_still_runs():
    """RENAME has run: live holds the new graph, so raising would report a failed swap that happened. The loss is on the result
    (``lock_held_after_rename`` False, ``released`` False because the compare-and-delete finds another holder) and the remainder replay
    still runs (offset-based, so idempotent), as the verdict credential."""
    b, v, log, shared, _ = _pair()
    replays = []
    res = SW.swap(b, v, SYSTEM, BUILD_ID, replay=lambda conn, key, system: replays.append((conn.name, key)) or 0,
                  hook=lambda name: shared.__setitem__(LOCK, "other") if name == "after_rename" else None)
    assert res.renamed is True and res.lock_held_after_rename is False and res.released is False
    assert shared[LOCK] == "other" and shared[LIVE] == "new"
    assert replays == [("build", BUILD), ("verdict", LIVE)]


def test_the_lock_check_after_the_rename_reads_the_lock_key_as_the_lock_holder_and_after_the_rename():
    b, v, log, shared, _ = _pair()
    SW.swap(b, v, SYSTEM, BUILD_ID)
    i = [e[1] for e in log].index("rename")
    assert log[i + 1] == ("build", "get", LOCK)
