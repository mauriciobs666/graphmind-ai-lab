"""P1-19 dispatch 2, LIVE on a throwaway instance only (``infra/ccg-falkordb.sh --throwaway up``; ``-m live``): the ``ccg.store.acl.create_users`` users
and the grant matrix of ``docs/plans/connected-code-graph-p1-graph.md`` §8 P1-19 (c), against ``ccg_shop`` planted beside ``ccg_shop2``,
``ccg_shop__x`` and ``ccg_shop___build_1`` (the keys of the refused names ``shop__x`` and ``shop_``).

Needs ``CCG_FALKORDB_PORT`` (a throwaway's port, published by a container labelled ``ccg-falkordb.throwaway=1``; 6379 and 6390 are refused) and
``CCG_CONFIG_DIR`` (a scratch directory holding the throwaway's ``falkordb-admin.pass``; ``~/.config/ccg`` is refused). Every test also sets
``CCG_CREDENTIALS_DIR`` to a tmp dir, so nothing under ``~/.config/ccg`` is read or written. The system names ``shop``, ``shop2`` and ``rot`` are
used literally: do not run two of these modules against one throwaway at once. ``CCG_REQUIRE_LIVE=1`` makes a missing instance a failure."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from loader_helpers import admin_password, skip_or_fail, throwaway_port
from redis.exceptions import AuthenticationError, ResponseError

from ccg import atrest, sysname
from ccg.store import acl
from ccg.store import loader as L
from ccg.store.swap import RELEASE_LUA

HOST = "127.0.0.1"
SHOP, SHOP2 = "shop", "shop2"
TOK = "tok-1"
LIVE, B1, B9 = sysname.live_key(SHOP), sysname.build_key(SHOP, "1"), sysname.build_key(SHOP, "9")
O1, O9 = sysname.overlay_key(SHOP, "1"), sysname.overlay_key(SHOP, "9")
LOCK = sysname.lock_key(SHOP)
S2, S2B, S2O, S2LOCK = (sysname.live_key(SHOP2), sysname.build_key(SHOP2, "1"), sysname.overlay_key(SHOP2, "1"), sysname.lock_key(SHOP2))
NOPERM = ("noperm", "no permissions", "can't run", "permission")
ROLES = sysname.ROLES


class _World(SimpleNamespace):
    """The fixture value. Its repr names no secret: a failing test prints its arguments, and a password in that output is a leak (a planted
    failure run, ``--tb=long``, grepped for the passwords found the default repr printing them)."""

    def __repr__(self):
        return "<world: throwaway instance, redacted>"


class _Redacted(dict):
    def __repr__(self):
        return "{<redacted>}"


@pytest.fixture(autouse=True)
def _credentials_dir(tmp_path, monkeypatch):
    """Every test: credentials in a tmp dir (never ``~/.config/ccg``)."""
    monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(tmp_path / "cred"))
    assert Path(os.environ["CCG_CREDENTIALS_DIR"]).resolve() != (Path.home() / ".config" / "ccg" / "keys").resolve()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    port = throwaway_port()
    if port is None:
        skip_or_fail("no throwaway instance (or the docker CLI cannot start): start one with cpg/infra/ccg-falkordb.sh --throwaway up and "
                     "export CCG_FALKORDB_PORT and CCG_CONFIG_DIR (a scratch dir)")
    try:
        pw = admin_password()
    except FileNotFoundError as exc:
        skip_or_fail(f"throwaway admin password not readable ({exc})")
    admin = L.connect(HOST, port, username=None, password=pw)
    w = _World(port=port, admin=admin, passwords=_Redacted(), conns={}, users=[], config_dir=os.environ["CCG_CONFIG_DIR"])     # captured before conftest's per-test override
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CCG_CREDENTIALS_DIR", str(tmp_path_factory.mktemp("ccg-credentials-world")))
        for system in (SHOP, SHOP2):
            w.users += acl.create_users(admin, system, template=_template())
            w.passwords[system] = _Redacted(acl.read_credentials(system))
    for system in (SHOP, SHOP2):
        for role in ROLES:
            w.conns[system, role] = L.connect(HOST, port, username=sysname.acl_user(role, system), password=w.passwords[system][role])
    w.all_passwords = tuple(p for d in w.passwords.values() for p in d.values())     # a tuple of str: only ever iterated, never asserted on directly
    try:
        yield w
    finally:
        _wipe(admin)
        for user in w.users:
            admin.connection.execute_command("ACL", "DELUSER", user)


def _template():
    """``None`` (the data file ships), or, for a mutation check, the lines of §7 in a scratch copy of the plan named by ``CCG_ACL_TEST_NOTE`` (an
    unset variable in the committed run). ``create_users`` validates either against the closed grammar, so a mutant must stay inside the ceiling."""
    note = os.environ.get("CCG_ACL_TEST_NOTE")
    if not note:
        return None
    return acl.parse_acl_lines(Path(note).read_text().split("## 7. ACL users", 1)[1].split("```", 2)[1])


def _wipe(admin):
    for pattern in ("ccg_shop*", "ccg_rot*", "p119b_*"):
        for key in list(admin.connection.scan_iter(match=pattern)):
            admin.connection.delete(key)


def _conn(w, system, role):
    return w.conns[system, role].connection


def _plant(w):
    """A known keyspace: a live graph with its Snapshot stamp, a build key, an overlay, the lock held by ``TOK``, and the same for ``shop2``."""
    a = w.admin
    _wipe(a)
    a.connection.execute_command("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", 10000)         # a mutated grant that allowed CONFIG SET must not leak into the next probe
    for key in (LIVE, S2):
        a.select_graph(key).query("CREATE (:Marker {v: 'm'}), (:Snapshot {snapshotId: 's', verdictLogOffset: 0, verdictLogHash: 'h'})")
    for key in (B1, O1, S2B, S2O):
        a.select_graph(key).query("CREATE (:Marker {v: 'm'})")
    for key in (LOCK, S2LOCK):
        a.connection.set(key, TOK)


WATCHED = [LIVE, B1, B9, O1, O9, LOCK, S2, S2B, S2O, S2LOCK]


def _digest(w):
    """What an unwanted write would change: existence, TTL, content, and the whole keyspace size."""
    a, out = w.admin, {"dbsize": w.admin.connection.dbsize()}
    for key in WATCHED:
        if not a.connection.exists(key):
            out[key] = None
        elif key not in (LOCK, S2LOCK):
            out[key] = ("graph", a.connection.pttl(key), a.select_graph(key).ro_query("MATCH (n) RETURN count(n)").result_set[0][0])
        else:
            out[key] = ("str", a.connection.pttl(key), a.connection.get(key))
    out["timeout_max"] = a.connection.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")
    return out


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def _is_refusal(exc: Exception) -> bool:
    return isinstance(exc, ResponseError) and any(t in str(exc).lower() for t in NOPERM)


def _q(key, cypher):
    return lambda c: c.execute_command("GRAPH.QUERY", key, cypher)


def _cmd(*args):
    return lambda c: c.execute_command(*args)


def _constraint(c):
    c.execute_command("GRAPH.QUERY", B1, "CREATE INDEX FOR (n:T) ON (n.p)")
    return c.execute_command("GRAPH.CONSTRAINT", "CREATE", B1, "UNIQUE", "NODE", "T", "PROPERTIES", "1", "p")


PATCH = "MATCH (s:Snapshot) WITH s, s.verdictLogOffset AS o SET s.verdictLogOffset = o + 1 RETURN o, s.verdictLogOffset"
DEL_LIVE_LUA = "return redis.call('DEL', KEYS[1])"
DEL_HARDCODED_LUA = f"return redis.call('DEL', '{LIVE}')"

# (role, probe id, call, allowed). A call is a function of the role's raw redis connection.
BUILD_OK = [
    ("GRAPH.QUERY build key", _q(B1, "CREATE (:T {p: 1})")), ("CREATE INDEX on build key", _q(B1, "CREATE INDEX FOR (n:T) ON (n.p)")),
    ("GRAPH.CONSTRAINT on build key", _constraint), ("GRAPH.RO_QUERY build key", _cmd("GRAPH.RO_QUERY", B1, "MATCH (n) RETURN count(n)")),
    ("GRAPH.EXPLAIN build key", _cmd("GRAPH.EXPLAIN", B1, "MATCH (n) RETURN n")), ("GRAPH.CONFIG GET", _cmd("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")),
    ("EXPIRE build key", _cmd("EXPIRE", B1, 1000)), ("EXPIRE overlay key", _cmd("EXPIRE", O1, 1000)), ("DEL build key", _cmd("DEL", B1)),
    ("DEL overlay key", _cmd("DEL", O1)), ("RENAME build -> live", _cmd("RENAME", B1, LIVE)), ("RENAME overlay -> live", _cmd("RENAME", O1, LIVE)),
    ("EXISTS live", _cmd("EXISTS", LIVE)), ("INFO", _cmd("INFO")), ("lock SET NX PX", _cmd("SET", LOCK, "t", "NX", "PX", 5000)),
    ("lock GET", _cmd("GET", LOCK)), ("lock compare-and-delete EVAL", _cmd("EVAL", RELEASE_LUA, 1, LOCK, TOK)),
]
BUILD_NO = [
    ("GRAPH.QUERY live", _q(LIVE, "RETURN 1")), ("GRAPH.RO_QUERY live", _cmd("GRAPH.RO_QUERY", LIVE, "RETURN 1")),
    ("GRAPH.EXPLAIN live", _cmd("GRAPH.EXPLAIN", LIVE, "RETURN 1")), ("EXPIRE live 0", _cmd("EXPIRE", LIVE, 0)), ("EXPIRE live 100", _cmd("EXPIRE", LIVE, 100)),
    ("DEL live", _cmd("DEL", LIVE)), ("UNLINK live", _cmd("UNLINK", LIVE)), ("RENAME live -> build", _cmd("RENAME", LIVE, B9)),
    ("RENAME live -> overlay", _cmd("RENAME", LIVE, O9)), ("EVAL DEL live (declared key)", _cmd("EVAL", DEL_LIVE_LUA, 1, LIVE)),
    ("EVAL DEL live (hardcoded, lock declared)", _cmd("EVAL", DEL_HARDCODED_LUA, 1, LOCK)),
    ("GRAPH.CONFIG SET", _cmd("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", 7777)), ("UNLINK build key", _cmd("UNLINK", B1)), ("UNLINK overlay key", _cmd("UNLINK", O1)), ("GRAPH.DELETE build key", _cmd("GRAPH.DELETE", B1)),
    ("GRAPH.DELETE live", _cmd("GRAPH.DELETE", LIVE)), ("GRAPH.LIST", _cmd("GRAPH.LIST")), ("GRAPH.PROFILE build key", _cmd("GRAPH.PROFILE", B1, "MATCH (n) RETURN n")),
    ("GRAPH.QUERY shop2 live", _q(S2, "RETURN 1")), ("GRAPH.RO_QUERY shop2 live", _cmd("GRAPH.RO_QUERY", S2, "RETURN 1")),
    ("GRAPH.QUERY shop2 build", _q(S2B, "RETURN 1")), ("DEL shop2 live", _cmd("DEL", S2)), ("DEL shop2 build", _cmd("DEL", S2B)),
    ("DEL shop2 overlay", _cmd("DEL", S2O)), ("EXPIRE shop2 build", _cmd("EXPIRE", S2B, 100)), ("EXISTS shop2 live", _cmd("EXISTS", S2)),
    ("RENAME build -> shop2 live", _cmd("RENAME", B1, S2)), ("RENAME shop2 build -> build", _cmd("RENAME", S2B, B9)),
    ("shop2 lock SET", _cmd("SET", S2LOCK, "x")), ("shop2 lock GET", _cmd("GET", S2LOCK)), ("shop2 lock EVAL", _cmd("EVAL", RELEASE_LUA, 1, S2LOCK, TOK)),
    ("KEYS *", _cmd("KEYS", "*")), ("FLUSHALL", _cmd("FLUSHALL")), ("ACL SETUSER", _cmd("ACL", "SETUSER", "p119b_x", "on")),
    ("CONFIG GET", _cmd("CONFIG", "GET", "save")), ("SET on build key", _cmd("SET", B9, "x")),
]
VERDICT_OK = [
    ("read live offset and patch it in one GRAPH.QUERY", _q(LIVE, PATCH)), ("INFO", _cmd("INFO")), ("lock SET NX PX", _cmd("SET", LOCK, "t", "NX", "PX", 5000)),
    ("lock GET", _cmd("GET", LOCK)), ("lock compare-and-delete EVAL", _cmd("EVAL", RELEASE_LUA, 1, LOCK, TOK)),
]
VERDICT_NO = [
    ("GRAPH.RO_QUERY live", _cmd("GRAPH.RO_QUERY", LIVE, "RETURN 1")), ("GRAPH.EXPLAIN live", _cmd("GRAPH.EXPLAIN", LIVE, "RETURN 1")),
    ("EXPIRE live", _cmd("EXPIRE", LIVE, 0)), ("DEL live", _cmd("DEL", LIVE)), ("UNLINK live", _cmd("UNLINK", LIVE)), ("RENAME live away", _cmd("RENAME", LIVE, B9)),
    ("RENAME build -> live", _cmd("RENAME", B1, LIVE)), ("RENAME build -> build", _cmd("RENAME", B1, B9)), ("EXISTS live", _cmd("EXISTS", LIVE)),
    ("GRAPH.QUERY build key", _q(B1, "RETURN 1")), ("DEL build key", _cmd("DEL", B1)), ("EXPIRE build key", _cmd("EXPIRE", B1, 0)),
    ("DEL overlay", _cmd("DEL", O1)), ("GRAPH.QUERY overlay", _q(O1, "RETURN 1")), ("EVAL DEL live (declared key)", _cmd("EVAL", DEL_LIVE_LUA, 1, LIVE)),
    ("EVAL DEL live (hardcoded, lock declared)", _cmd("EVAL", DEL_HARDCODED_LUA, 1, LOCK)), ("GRAPH.CONFIG GET", _cmd("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")),
    ("GRAPH.CONFIG SET", _cmd("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", 7777)), ("GRAPH.DELETE live", _cmd("GRAPH.DELETE", LIVE)), ("GRAPH.LIST", _cmd("GRAPH.LIST")),
    ("GRAPH.PROFILE live", _cmd("GRAPH.PROFILE", LIVE, "MATCH (n) RETURN n")), ("GRAPH.QUERY shop2 live", _q(S2, "RETURN 1")), ("DEL shop2 live", _cmd("DEL", S2)),
    ("shop2 lock SET", _cmd("SET", S2LOCK, "x")), ("shop2 lock GET", _cmd("GET", S2LOCK)), ("KEYS *", _cmd("KEYS", "*")), ("FLUSHALL", _cmd("FLUSHALL")),
    ("lock DEL (not compare-and-delete)", _cmd("EXPIRE", LOCK, 0)),
]
READER_OK = [
    ("GRAPH.RO_QUERY live", _cmd("GRAPH.RO_QUERY", LIVE, "MATCH (s:Snapshot) RETURN s.verdictLogOffset")), ("GRAPH.RO_QUERY overlay", _cmd("GRAPH.RO_QUERY", O1, "RETURN 1")),
    ("GRAPH.EXPLAIN live", _cmd("GRAPH.EXPLAIN", LIVE, "MATCH (n) RETURN n")), ("INFO", _cmd("INFO")),
]
READER_NO = [
    ("GRAPH.QUERY live", _q(LIVE, "RETURN 1")), ("GRAPH.QUERY live (write)", _q(LIVE, "CREATE (:X)")), ("GRAPH.QUERY overlay", _q(O1, "RETURN 1")),
    ("GRAPH.RO_QUERY build key", _cmd("GRAPH.RO_QUERY", B1, "RETURN 1")), ("GRAPH.EXPLAIN build key", _cmd("GRAPH.EXPLAIN", B1, "RETURN 1")),
    ("EXPIRE live", _cmd("EXPIRE", LIVE, 0)), ("EXPIRE overlay", _cmd("EXPIRE", O1, 0)), ("DEL live", _cmd("DEL", LIVE)), ("DEL overlay", _cmd("DEL", O1)),
    ("UNLINK live", _cmd("UNLINK", LIVE)), ("RENAME live away", _cmd("RENAME", LIVE, B9)), ("RENAME overlay -> live", _cmd("RENAME", O1, LIVE)),
    ("EXISTS live", _cmd("EXISTS", LIVE)), ("lock SET", _cmd("SET", LOCK, "x", "NX", "PX", 5000)), ("lock GET", _cmd("GET", LOCK)),
    ("lock EVAL", _cmd("EVAL", RELEASE_LUA, 1, LOCK, TOK)), ("GRAPH.CONFIG GET", _cmd("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")),
    ("GRAPH.CONFIG SET", _cmd("GRAPH.CONFIG", "SET", "TIMEOUT_MAX", 7777)), ("GRAPH.DELETE live", _cmd("GRAPH.DELETE", LIVE)), ("GRAPH.LIST", _cmd("GRAPH.LIST")),
    ("GRAPH.PROFILE live", _cmd("GRAPH.PROFILE", LIVE, "MATCH (n) RETURN n")), ("GRAPH.RO_QUERY shop2 live", _cmd("GRAPH.RO_QUERY", S2, "RETURN 1")),
    ("DEL shop2 live", _cmd("DEL", S2)), ("KEYS *", _cmd("KEYS", "*")), ("FLUSHALL", _cmd("FLUSHALL")),
]
MATRIX = ([("build", n, f, True) for n, f in BUILD_OK] + [("build", n, f, False) for n, f in BUILD_NO]
          + [("verdict", n, f, True) for n, f in VERDICT_OK] + [("verdict", n, f, False) for n, f in VERDICT_NO]
          + [(r, n, f, True) for r in ("engine", "raw") for n, f in READER_OK] + [(r, n, f, False) for r in ("engine", "raw") for n, f in READER_NO])


@pytest.mark.live
@pytest.mark.parametrize("role,name,call,allowed", MATRIX, ids=[f"{r}:{'ok' if a else 'NO'}:{n}" for r, n, _f, a in MATRIX])
def test_the_grant_matrix_of_section_8_p1_19_c(world, role, name, call, allowed):
    _plant(world)
    before = _digest(world)
    conn = _conn(world, SHOP, role)
    try:
        call(conn)
        err = None
    except Exception as exc:       # noqa: BLE001
        err = exc
    after = _digest(world)
    if allowed:
        assert err is None, f"{role} {name} failed: {err!r}"
    else:
        assert err is not None and _is_refusal(err), f"{role} {name} was NOT refused (result or error: {err!r})"
        assert after == before, f"{role} {name} was refused but the keyspace changed"
        leaked = any(p in str(err) for p in world.all_passwords)
        assert not leaked


@pytest.mark.live
def test_the_allowed_probes_do_what_they_claim(world):
    """Positive controls: a permitted probe really acts, so a refusal elsewhere is not a dead connection."""
    _plant(world)
    v, b = _conn(world, SHOP, "verdict"), _conn(world, SHOP, "build")
    v.execute_command("GRAPH.QUERY", LIVE, PATCH)
    assert world.admin.select_graph(LIVE).ro_query("MATCH (s:Snapshot) RETURN s.verdictLogOffset").result_set == [[1]]
    assert _conn(world, SHOP, "build").execute_command("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")[1] == 10000
    b.execute_command("RENAME", B1, LIVE)
    assert not world.admin.connection.exists(B1) and world.admin.select_graph(LIVE).ro_query("MATCH (n) RETURN count(n)").result_set == [[1]]
    world.admin.connection.delete(LOCK)
    assert b.execute_command("SET", LOCK, "t1", "NX", "PX", 5000) in (True, b"OK") and v.execute_command("SET", LOCK, "t2", "NX", "PX", 5000) is None
    assert _s(v.execute_command("GET", LOCK)) == "t1"
    assert v.execute_command("EVAL", RELEASE_LUA, 1, LOCK, "wrong") == 0 and v.execute_command("EVAL", RELEASE_LUA, 1, LOCK, "t1") == 1
    assert not world.admin.connection.exists(LOCK)
    assert b.execute_command("EXPIRE", O1, 1000) == 1 and 0 < world.admin.connection.pttl(O1) <= 1_000_000
    assert b.execute_command("DEL", O1) == 1 and not world.admin.connection.exists(O1)


@pytest.mark.live
def test_the_engine_reads_live_through_falkordb_py_and_cannot_write_it(world):
    _plant(world)
    db = world.conns[SHOP, "engine"]
    assert db.select_graph(LIVE).ro_query("MATCH (s:Snapshot) RETURN s.snapshotId").result_set == [["s"]]
    with pytest.raises(ResponseError) as ei:
        db.select_graph(LIVE).query("CREATE (:X)")
    assert _is_refusal(ei.value)
    assert world.admin.select_graph(LIVE).ro_query("MATCH (x:X) RETURN count(x)").result_set == [[0]]


@pytest.mark.live
def test_shop2_users_hold_nothing_of_shop(world):
    """The matrix's other direction: each ``shop2`` role against every ``shop`` key."""
    _plant(world)
    before = _digest(world)
    shop_keys = (LIVE, B1, O1)
    for role in ("build", "verdict", "engine", "raw"):
        c = _conn(world, SHOP2, role)
        calls = [("GRAPH.QUERY", k, "RETURN 1") for k in shop_keys] + [("GRAPH.RO_QUERY", k, "RETURN 1") for k in shop_keys] \
            + [("DEL", k) for k in shop_keys] + [("EXPIRE", k, 0) for k in shop_keys] + [("RENAME", S2B, LIVE), ("RENAME", S2O, B9),
                                                                                       ("SET", LOCK, "x"), ("GET", LOCK), ("EVAL", RELEASE_LUA, 1, LOCK, TOK)]
        if role == "build":
            calls += [("RENAME", B1, S2), ("RENAME", S2B, LIVE)]
        for args in calls:
            with pytest.raises(ResponseError) as ei:
                c.execute_command(*args)
            assert _is_refusal(ei.value), (role, args, str(ei.value))
    assert _digest(world) == before


FLAT_OLD = ["on", ">p119b-flat-pw", "resetkeys", "resetchannels", "-@all", "+graph.query", "+graph.ro_query", "+info", "~ccg_shop*"]


@pytest.mark.live
def test_the_old_flat_pattern_is_the_negative_control_it_grants_shop2_and_the_real_user_does_not(world):
    _plant(world)
    admin = world.admin
    admin.connection.execute_command("ACL", "SETUSER", "p119b_flat_ctl", *FLAT_OLD)
    try:
        flat = L.connect(HOST, world.port, username="p119b_flat_ctl", password="p119b-flat-pw").connection
        real = _conn(world, SHOP, "build")
        for key in (S2, S2B):                                        # the same command on the same key: two verdicts
            assert flat.execute_command("GRAPH.RO_QUERY", key, "MATCH (n:Marker) RETURN count(n)")[1] == [[1]]
            with pytest.raises(ResponseError) as ei:
                real.execute_command("GRAPH.RO_QUERY", key, "MATCH (n:Marker) RETURN count(n)")
            assert _is_refusal(ei.value)
        flat.execute_command("GRAPH.QUERY", S2, "CREATE (:Planted)")      # the flat pattern WRITES the neighbour's live graph
        assert admin.select_graph(S2).ro_query("MATCH (p:Planted) RETURN count(p)").result_set == [[1]]
    finally:
        admin.connection.execute_command("ACL", "DELUSER", "p119b_flat_ctl")


@pytest.mark.live
@pytest.mark.parametrize("bad", ["shop__x", "shop_"])
def test_the_refused_names_are_refused_before_any_key_exists_and_the_pattern_alone_would_grant_their_keys(world, bad):
    _wipe(world.admin)
    keys = {"shop__x": [f"ccg_{bad}"], "shop_": [f"ccg_{bad}", f"ccg_{bad}__build_1", f"ccg_{bad}:verdict_lock"]}[bad]
    assert not any(world.admin.connection.exists(k) for k in keys)                       # no key exists yet
    with pytest.raises(sysname.SystemNameError):
        sysname.check_system_name(bad)
    with pytest.raises((sysname.SystemNameError, ValueError)):
        sysname.build_key(bad, "1")
    with pytest.raises(sysname.SystemNameError):
        acl.create_users(world.admin, bad)
    users = {_s(u) for u in world.admin.connection.execute_command("ACL", "USERS")}
    assert not [u for u in users if u.endswith(("_" + bad, bad))]                                    # no user of the refused name either
    assert not any(world.admin.connection.exists(k) for k in keys)                       # still no key: the refusal came first
    # what the whitelist prevents: ``shop``'s own §7 pattern ``~ccg_shop__*`` matches these keys, so ``shop``'s build credential would hold them
    granted = f"ccg_{bad}__build_1" if bad == "shop_" else f"ccg_{bad}"
    world.admin.select_graph(granted).query("CREATE (:Marker {v: 'theirs'})")
    got = _conn(world, SHOP, "build").execute_command("GRAPH.RO_QUERY", granted, "MATCH (m:Marker) RETURN m.v")
    assert got[1] == [["theirs"]] or got[1] == [[b"theirs"]]
    if bad == "shop_":                                                                   # while the bare live key of ``shop_`` is NOT matched
        world.admin.select_graph(f"ccg_{bad}").query("CREATE (:Marker)")
        with pytest.raises(ResponseError) as ei:
            _conn(world, SHOP, "build").execute_command("GRAPH.RO_QUERY", f"ccg_{bad}", "RETURN 1")
        assert _is_refusal(ei.value)


@pytest.mark.live
def test_create_users_leaves_default_plus_exactly_the_four_users_per_system(world):
    users = {_s(u) for u in world.admin.connection.execute_command("ACL", "USERS")}
    assert {sysname.acl_user(r, s) for s in (SHOP, SHOP2) for r in ROLES} <= users
    assert all(u == "default" or sysname.acl_user_re().match(u) for u in users - {"p119b_flat_ctl"}), users


# ------------------------------------------------------------------------------------- create_users on the server: rotation, reset, fail closed

def _auth_fails(world, user, password) -> bool:
    try:
        L.connect(HOST, world.port, username=user, password=password).connection.execute_command("INFO")
    except (AuthenticationError, ResponseError) as exc:
        return "wrongpass" in str(exc).lower() or "invalid username-password" in str(exc).lower() or "auth" in str(exc).lower()
    return False


@pytest.mark.live
def test_a_second_create_users_rotates_passwords_and_resets_a_hand_added_grant(world):
    system = "rot"
    admin = world.admin
    try:
        acl.create_users(admin, system)
        first = _Redacted(acl.read_credentials(system))
        user = sysname.acl_user("engine", system)
        admin.connection.execute_command("ACL", "SETUSER", user, "~ccg_other*", "+graph.query", "+del", ">hand-added-pw")     # drift
        hand_added_works = not _auth_fails(world, user, "hand-added-pw")
        assert hand_added_works
        acl.create_users(admin, system)
        second = _Redacted(acl.read_credentials(system))
        rotated = all(first[r] != second[r] for r in ROLES)
        assert rotated
        for role in ROLES:
            old_is_dead = _auth_fails(world, sysname.acl_user(role, system), first[role])               # the old secret no longer authenticates
            assert old_is_dead
            L.connect(HOST, world.port, username=sysname.acl_user(role, system), password=second[role]).connection.execute_command("INFO")
        hand_added_dead = _auth_fails(world, user, "hand-added-pw")                                     # reset dropped the added password
        assert hand_added_dead
        info = admin.connection.execute_command("ACL", "GETUSER", user)
        flat = repr(info)
        assert "ccg_other" not in flat and "+del" not in flat and "+graph.ro_query" in flat
    finally:
        for r in ROLES:
            admin.connection.execute_command("ACL", "DELUSER", sysname.acl_user(r, system))


@pytest.mark.live
def test_a_credentials_file_failure_leaves_every_user_disabled(world, monkeypatch):
    system = "rot"
    admin = world.admin

    def boom(*_a, **_k):
        raise OSError("planted")

    monkeypatch.setattr(atrest, "write_acl", boom)
    try:
        with pytest.raises(acl.AclError):
            acl.create_users(admin, system)
        for role in ROLES:
            flags = admin.connection.execute_command("ACL", "GETUSER", sysname.acl_user(role, system))
            asdict = {_s(k): v for k, v in zip(flags[::2], flags[1::2])}
            assert "off" in [_s(f) for f in asdict["flags"]] and asdict["passwords"] == []
    finally:
        for r in ROLES:
            admin.connection.execute_command("ACL", "DELUSER", sysname.acl_user(r, system))


@pytest.mark.live
def test_no_password_reaches_stdout_stderr_or_a_file_but_the_credentials_file(world, tmp_path):
    """``create_users`` in a child process (its stdout and stderr captured): the passwords appear in neither, and the only file that holds them is the 0600 one."""
    child = ("import os, sys, pathlib\n"
             "from ccg.store import acl, loader\n"
             "pw = (pathlib.Path(os.environ['CCG_CONFIG_DIR']) / 'falkordb-admin.pass').read_text().strip()\n"
             "admin = loader.connect('127.0.0.1', int(os.environ['CCG_FALKORDB_PORT']), username=None, password=pw)\n"
             "print(acl.create_users(admin, 'rot'))\n")
    cred = tmp_path / "child-cred"
    env = dict(os.environ, CCG_CREDENTIALS_DIR=str(cred), CCG_CONFIG_DIR=world.config_dir)
    try:
        run = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=60, env=env, cwd=Path(__file__).resolve().parents[1],
                             stdin=subprocess.DEVNULL)
        assert run.returncode == 0, run.stderr[-300:]
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CCG_CREDENTIALS_DIR", str(cred))
            pws = set(acl.read_credentials("rot").values())
        assert len(pws) == 4
        in_output = any(p in run.stdout + run.stderr for p in pws)
        assert not in_output
        holders = [p for p in cred.rglob("*") if p.is_file() and any(p_.encode() in p.read_bytes() for p_ in pws)]
        assert holders == [cred / "rot.acl"] and oct(holders[0].stat().st_mode & 0o777) == "0o600"
    finally:
        for r in ROLES:
            world.admin.connection.execute_command("ACL", "DELUSER", sysname.acl_user(r, "rot"))


@pytest.mark.live
def test_rotation_ends_the_established_sessions_of_the_old_password(world):
    """``CLIENT KILL USER`` after each ``SETUSER``: a connection opened with the old password is dead after the next ``create_users``."""
    system, admin = "rot", world.admin
    try:
        acl.create_users(admin, system)
        old = {r: L.connect(HOST, world.port, username=sysname.acl_user(r, system), password=acl.read_credentials(system)[r]) for r in ROLES}
        for r in ROLES:
            assert old[r].connection.execute_command("INFO")                                  # established and working
        acl.create_users(admin, system)
        for r in ROLES:
            with pytest.raises(Exception):
                old[r].connection.execute_command("INFO")                                      # killed; its re-AUTH with the old password fails
        new = acl.read_credentials(system)
        assert L.connect(HOST, world.port, username=sysname.acl_user("engine", system), password=new["engine"]).connection.execute_command("INFO")
    finally:
        for r in ROLES:
            admin.connection.execute_command("ACL", "DELUSER", sysname.acl_user(r, system))


@pytest.mark.live
def test_a_failed_rotation_ends_the_sessions_disables_the_users_and_removes_the_file(world, monkeypatch):
    system, admin = "rot", world.admin
    try:
        acl.create_users(admin, system)
        cred_file = atrest.acl_path(system)
        assert cred_file.exists()
        live_conn = L.connect(HOST, world.port, username=sysname.acl_user("build", system), password=acl.read_credentials(system)["build"])
        assert live_conn.connection.execute_command("INFO")
        monkeypatch.setattr(atrest, "write_acl", lambda *a, **k: (_ for _ in ()).throw(OSError("planted")))
        with pytest.raises(acl.AclError, match="the users are disabled"):
            acl.create_users(admin, system)
        with pytest.raises(Exception):
            live_conn.connection.execute_command("INFO")
        assert not cred_file.exists()
    finally:
        for r in ROLES:
            admin.connection.execute_command("ACL", "DELUSER", sysname.acl_user(r, system))


@pytest.mark.live
def test_a_bad_credentials_dir_on_a_rotation_leaves_the_working_users_untouched(world, monkeypatch):
    system, admin = "rot", world.admin
    try:
        acl.create_users(admin, system)
        pw = _Redacted(acl.read_credentials(system))
        monkeypatch.setenv("CCG_CREDENTIALS_DIR", str(atrest.REPO_ROOT / "cpg" / "p119c-bad"))
        with pytest.raises(acl.AclError, match="credentials directory"):
            acl.create_users(admin, system)
        still_works = L.connect(HOST, world.port, username=sysname.acl_user("engine", system), password=pw["engine"]).connection.execute_command("INFO")
        assert still_works                                                                    # not reset, not disabled
    finally:
        for r in ROLES:
            admin.connection.execute_command("ACL", "DELUSER", sysname.acl_user(r, system))
