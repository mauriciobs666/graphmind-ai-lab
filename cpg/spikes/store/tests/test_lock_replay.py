"""S22 (P0-16): the verdict lock, log and swap replay (``lock_replay.py``). Non-live tests drive the protocol through a
fake connection and graph (order, fsync, torn tail, release by token); ``-m live`` ones run the fault cases on
``ccg-falkordb`` against ``ccg_p0spike__lkt*`` keys only and delete them: B6 (a verdict issued during a build survives
the swap), P2-3 (lock expired mid-swap), SIGKILL between append and patch (the next verdict or swap re-converges)."""
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import lock_replay as LR  # noqa: E402


# ---------------------------------------------------------------------------------------------
# The log.
# ---------------------------------------------------------------------------------------------


def test_log_append_fsyncs_every_entry(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    log = LR.Log(tmp_path / "v.jsonl")
    assert log.append({"linkId": "a", "verdict": "d", "by": "x"}) == 1
    assert len(calls) >= 1
    n = len(calls)
    assert log.append({"linkId": "b", "verdict": "d", "by": "x"}) == 2
    assert len(calls) > n


def test_log_torn_tail_is_ignored_and_not_glued_onto(tmp_path):
    p = tmp_path / "v.jsonl"
    log = LR.Log(p)
    log.append({"linkId": "a", "verdict": "d", "by": "x"})
    with open(p, "ab") as f:
        f.write(b'{"linkId": "b", "verd')          # a kill mid-write
    assert [e["linkId"] for e in log.entries()] == ["a"]
    assert log.append({"linkId": "c", "verdict": "d", "by": "x"}) == 2
    assert [e["linkId"] for e in log.entries()] == ["a", "c"]


# ---------------------------------------------------------------------------------------------
# The protocol over a fake connection/graph: order, remainder, catch-up, release by token.
# ---------------------------------------------------------------------------------------------


class FakeConn:
    def __init__(self):
        self.kv, self.ops = {}, []

    def set(self, key, val, nx=False, px=None):
        self.ops.append(("set", key))
        if nx and key in self.kv:
            return None
        self.kv[key] = val
        return True

    def eval(self, lua, n, key, token):
        assert lua == LR.RELEASE_LUA
        self.ops.append(("release", key))
        if self.kv.get(key) == token:
            del self.kv[key]
            return 1
        return 0


class FakeEngine(LR.Engine):
    """Graphs are ``{key: {"off": int, "v": {linkId: verdict}}}``; ``rename`` moves a dict."""

    def __init__(self, tmp_path, graphs, hook=None):
        self.fc = FakeConn()
        self.graphs = graphs
        self.events = []
        super().__init__(SimpleNamespace(connection=self.fc), "ccg_p0spike__fake", LR.Log(tmp_path / "v.jsonl"), 1000, hook)

    def _offset(self, key):
        return self.graphs[key]["off"]

    def _patch(self, key, entry, off):
        self.events.append(("patch", key, entry["linkId"], off))
        g = self.graphs[key]
        g["v"][entry["linkId"]] = entry["verdict"]
        g["off"] = max(g["off"], off)

    def _rename(self, src, dst):
        self.events.append(("rename", src, dst))
        self.graphs[dst] = self.graphs.pop(src)


LIVE, BUILD = "ccg_p0spike__fake", "ccg_p0spike__fake_b"


def _graphs():
    return {LIVE: {"off": 0, "v": {}}, BUILD: {"off": 0, "v": {}}}


def test_swap_order_lock_replay_rename_remainder_release(tmp_path):
    e = FakeEngine(tmp_path, _graphs())
    e.log.append({"linkId": "L1", "verdict": "d", "by": "x"})
    res = e.swap(BUILD)
    assert [x[0] for x in e.events] == ["patch", "rename", ], e.events
    assert e.events[0][1] == BUILD and e.events[1] == ("rename", BUILD, LIVE)
    assert [o[0] for o in e.fc.ops] == ["set", "release"]
    assert res["replayed"] == 1 and res["remainder"] == 0 and res["released"] is True
    assert e.graphs[LIVE]["v"] == {"L1": "d"} and e.graphs[LIVE]["off"] == 1


def test_swap_replays_an_entry_appended_after_the_pre_rename_replay(tmp_path):
    """P2-3: a verdict that lands between the pre-RENAME replay and the RENAME is replayed on the new live key."""
    holder = {}

    def hook(name):
        if name == "before_rename":                      # the lock "expired": another verdict logs and patches OLD live
            holder["e"].log.append({"linkId": "L9", "verdict": "late", "by": "y"})
            holder["e"]._patch(LIVE, {"linkId": "L9", "verdict": "late"}, 1)

    e = holder["e"] = FakeEngine(tmp_path, _graphs(), hook)
    res = e.swap(BUILD)
    assert res["remainder"] == 1
    assert e.graphs[LIVE]["v"] == {"L9": "late"} and e.graphs[LIVE]["off"] == 1


def test_verdict_replays_an_unapplied_earlier_entry_before_its_own(tmp_path):
    """A crash between append and patch left entry 0 unapplied; the next verdict converges both."""
    e = FakeEngine(tmp_path, _graphs())
    e.log.append({"linkId": "L1", "verdict": "crashed", "by": "x"})   # appended, never patched
    res = e.verdict("L2", "d", "x")
    assert res["replayed"] == 2
    assert e.graphs[LIVE]["v"] == {"L1": "crashed", "L2": "d"} and e.graphs[LIVE]["off"] == 2
    assert [ev[3] for ev in e.events] == [1, 2]                       # in log order, one offset step each


def test_verdict_appends_under_the_lock_and_releases_by_token(tmp_path):
    seen = []
    holder = {}

    def hook(name):
        seen.append((name, holder["e"].lock.key in holder["e"].fc.kv, len(holder["e"].log.entries())))

    e = holder["e"] = FakeEngine(tmp_path, _graphs(), hook)
    e.verdict("L1", "d", "x")
    assert seen[0] == ("locked", True, 0) and seen[1] == ("after_append", True, 1)
    assert e.fc.kv == {}                                              # released


def test_release_does_not_delete_another_holders_lock():
    c = FakeConn()
    lock = LR.Lock(c, "ccg_p0spike__fake:verdict_lock", 100)
    t1 = lock.acquire()
    c.kv[lock.key] = "someone-else"                                   # expired, then re-taken
    assert lock.release(t1) is False
    assert c.kv[lock.key] == "someone-else"


def test_guard_refuses_foreign_keys():
    for bad in ("ccg_p0slice", "ccg_p0spike", "ccg_p0slice:verdict_lock", "ccg_p0spike__x:other", ""):
        with pytest.raises(LR.LockError):
            LR.guard(bad)
    assert LR.guard("ccg_p0spike__lkt1:verdict_lock")


# ---------------------------------------------------------------------------------------------
# Live fault tests (ccg_p0spike__lkt* keys only).
# ---------------------------------------------------------------------------------------------

_n = [0]


@pytest.fixture
def world(tmp_path):
    db = LR.connect()
    _n[0] += 1
    live = f"ccg_p0spike__lkt{os.getpid()}x{_n[0]}"
    build = live + "_b"
    keys = (live, build, live + ":verdict_lock")
    LR.drop(db, *keys)
    LR.make_graph(db, live, 5, 0, "old")
    LR.make_graph(db, build, 5, 0, "new")                  # assembled while the log was empty: stamped offset 0
    yield SimpleNamespace(db=db, live=live, build=build, log=LR.Log(tmp_path / "v.jsonl"), path=tmp_path / "v.jsonl")
    LR.drop(db, *keys)


def marker(w):
    return w.db.select_graph(w.live).ro_query("MATCH (s:Snapshot) RETURN s.marker").result_set[0][0]


@pytest.mark.live
def test_live_lock_nx_contention_expiry_and_compare_and_delete(world):
    w = world
    lock = LR.Lock(w.db.connection, w.live + ":verdict_lock", 10_000)   # long: the NX/token part must not race the ttl
    t1 = lock.acquire()
    assert t1 and lock.acquire() is None                    # NX
    assert lock.release("not-the-token") is False and w.db.connection.exists(lock.key)
    assert lock.release(t1) is True and not w.db.connection.exists(lock.key)
    lock = LR.Lock(w.db.connection, w.live + ":verdict_lock", 300)
    t2 = lock.acquire()
    time.sleep(0.8)                                         # PX expiry
    assert not w.db.connection.exists(lock.key)
    assert lock.release(t2) is False                        # expired token releases nothing
    assert lock.acquire(wait_s=1.0)                         # and a waiter gets it


@pytest.mark.live
def test_live_offset_never_moves_backwards(world):
    """Two patches racing after a lock expiry may land out of order: the offset is a high-water mark."""
    w = world
    eng = LR.Engine(w.db, w.live, w.log, ttl_ms=5000)
    eng._patch(w.live, {"linkId": "L1", "verdict": "a", "by": "x"}, 5)
    eng._patch(w.live, {"linkId": "L2", "verdict": "b", "by": "x"}, 3)
    assert eng._offset(w.live) == 5


@pytest.mark.live
def test_live_verdict_issued_during_a_build_is_present_after_the_swap(world):
    """B6."""
    w = world
    eng = LR.Engine(w.db, w.live, w.log, ttl_ms=5000)
    eng.verdict("L1", "dismissed", "ann")                   # lands on the OLD live key while the build is assembled
    assert LR.link_verdict(w.db, w.live, "L1") == "dismissed"
    assert LR.link_verdict(w.db, w.build, "L1") is None
    res = eng.swap(w.build)
    assert marker(w) == "new" and res["replayed"] == 1 and res["remainder"] == 0 and res["released"] is True
    assert LR.link_verdict(w.db, w.live, "L1") == "dismissed"
    assert eng._offset(w.live) == 1
    assert not w.db.connection.exists(w.live + ":verdict_lock")


@pytest.mark.live
def test_live_lock_expired_mid_swap_verdict_on_old_key_is_present_after_the_swap(world):
    """P2-3: PX forced short, a sleep before RENAME; a verdict lands on the old key; present afterwards."""
    w = world
    swap_eng = LR.Engine(w.db, w.live, w.log, ttl_ms=300,
                         hook=lambda n: time.sleep(0.9) if n == "before_rename" else None)
    out = {}

    def late_verdict():
        time.sleep(0.1)
        e2 = LR.Engine(LR.connect(), w.live, w.log, ttl_ms=5000)
        out["r"] = e2.verdict("L2", "late", "bob", wait_s=3.0)       # waits for the 300 ms expiry
        out["old_marker_at_patch"] = marker(w)

    th = threading.Thread(target=late_verdict)
    th.start()
    res = swap_eng.swap(w.build)
    th.join(10)
    assert out["old_marker_at_patch"] == "old"                       # it really patched the OLD key
    assert marker(w) == "new"
    assert res["remainder"] == 1                                     # the remainder replay found it
    assert LR.link_verdict(w.db, w.live, "L2") == "late"
    assert swap_eng._offset(w.live) == 1


@pytest.mark.live
def test_live_swap_release_leaves_a_lock_another_holder_took_after_expiry(world):
    w = world
    swap_eng = LR.Engine(w.db, w.live, w.log, ttl_ms=300,
                         hook=lambda n: time.sleep(0.8) if n == "before_rename" else None)

    def slow_holder():
        time.sleep(0.1)
        e2 = LR.Engine(LR.connect(), w.live, w.log, ttl_ms=10_000,
                       hook=lambda n: time.sleep(1.2) if n == "after_append" else None)
        e2.verdict("L3", "slow", "cy", wait_s=3.0)

    th = threading.Thread(target=slow_holder)
    th.start()
    res = swap_eng.swap(w.build)                                     # returns while the slow holder still holds
    assert res["released"] is False                                  # compare-and-delete: not our token
    assert w.db.connection.exists(w.live + ":verdict_lock")          # the other holder's lock survives
    th.join(15)
    assert LR.link_verdict(w.db, w.live, "L3") == "slow"
    assert swap_eng._offset(w.live) == 1                             # applied once, not twice


_CHILD = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {here!r})
    import lock_replay as LR
    db = LR.connect()
    def hook(name):
        if name == "after_append":
            print("APPENDED", flush=True); time.sleep(60)
    LR.Engine(db, {live!r}, LR.Log({log!r}), ttl_ms=600, hook=hook).verdict("L1", "killed", "dan")
""")


def _kill_between_append_and_patch(w):
    code = _CHILD.format(here=str(HERE), live=w.live, log=str(w.path))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "APPENDED"
    p.kill()                                                         # SIGKILL
    p.wait()
    eng = LR.Engine(w.db, w.live, w.log, ttl_ms=5000)
    assert len(w.log.entries()) == 1                                 # fsynced before the kill
    assert eng._offset(w.live) == 0 and LR.link_verdict(w.db, w.live, "L1") is None   # never patched
    assert w.db.connection.exists(w.live + ":verdict_lock")          # the dead holder's lock waits out its PX
    return eng


@pytest.mark.live
def test_live_kill_between_append_and_patch_next_verdict_reconverges(world):
    w = world
    eng = _kill_between_append_and_patch(world)
    res = eng.verdict("L2", "next", "eve", wait_s=3.0)
    assert res["replayed"] == 2
    assert LR.link_verdict(w.db, w.live, "L1") == "killed" and LR.link_verdict(w.db, w.live, "L2") == "next"
    assert eng._offset(w.live) == 2


@pytest.mark.live
def test_live_kill_between_append_and_patch_next_swap_reconverges(world):
    w = world
    eng = _kill_between_append_and_patch(world)
    res = eng.swap(w.build, wait_s=3.0)
    assert res["replayed"] == 1 and res["remainder"] == 0
    assert marker(w) == "new"
    assert LR.link_verdict(w.db, w.live, "L1") == "killed" and eng._offset(w.live) == 1


# ---------------------------------------------------------------------------------------------
# ACL rules (acl_probe.py): the lock grants are key-scoped selectors, never flat commands on the key patterns.
# ---------------------------------------------------------------------------------------------

import acl_probe as AP  # noqa: E402


def test_acl_tokens_keep_a_selector_as_one_argument():
    toks = AP.tokens("~k +a +b (~lock +set +get +del +eval) +info")
    assert toks == ["~k", "+a", "+b", "(~lock +set +get +del +eval)", "+info"]


@pytest.mark.parametrize("user", ["ccg_build", "ccg_verdict"])
def test_selector_variant_grants_lock_commands_only_inside_the_selector(user):
    rule = AP.acl_rules(user, "selector")
    root = rule[: rule.index("(")]
    for cmd in ("+set", "+get", "+del"):
        assert cmd not in root.split() and cmd in rule[rule.index("("):]
    assert f"~{AP.LOCK}" in rule[rule.index("("):]


@pytest.mark.parametrize("user", ["ccg_engine", "ccg_raw_p0"])
def test_read_users_get_no_list_no_write_and_no_percent_r(user):
    rule = AP.acl_rules(user, "selector")
    assert "+graph.ro_query" in rule and "+info" in rule
    for bad in ("graph.list", "+graph.query", "graph.profile", "%R~", "+eval", "+set", "+del"):
        assert bad not in rule


@pytest.mark.live
def test_live_acl_matrix_selector_variant_matches_proposal_and_flat_grants_leak_del():
    r = AP.run(out=None)
    assert r["usersLeft"] == []
    assert all(d == {"allowedButNotExpected": [], "expectedButNotAllowed": []} for d in r["variants"]["selector"]["diff"].values())
    assert all(v == "ok" for v in r["variants"]["selector"]["constructor"].values())
    flat = r["variants"]["flat+get"]["diff"]
    assert "EVAL: DEL graph key in script" in flat["ccg_verdict"]["allowedButNotExpected"]
    assert "EVAL release (GET+DEL in script)" in r["variants"]["spec"]["diff"]["ccg_build"]["expectedButNotAllowed"]
    assert r["constructor_without_info"].startswith("fail")
