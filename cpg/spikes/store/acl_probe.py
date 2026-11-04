"""S22 (P0-16) -- the ACL matrix and the lock grants, probed on ``ccg-falkordb``.

Contract: proposal ``docs/plans/connected-code-graph.md`` §3.5 (ACL users, lock grants P2-3), graph note §6.4, V5, V16-V18.

Four users (``ccg_build``, ``ccg_verdict``, ``ccg_engine``, ``ccg_raw_p0``) are created on the instance (``ACL SETUSER`` is
instance-wide), exercised cell by cell against ``ccg_p0spike__acl*`` keys, and deleted in a ``finally`` (``ACL DELUSER``).
Three ACL variants are run, because the lock grant's wording decides what the lock key costs:

* ``selector`` -- the lock grants (``+set +get +del +eval``) live in a Redis selector bound to the lock key alone;
* ``flat+get`` -- the same four commands granted flat on the user's key patterns;
* ``spec`` -- proposal §3.5 to the letter: flat ``+set +del +eval`` (no ``+get``).

Passwords are random per run, held in memory, never printed or written. Each cell is ``allowed``, ``NOPERM-cmd``,
``NOPERM-key`` or ``other:<error>``; ``results/acl_matrix.json`` has the observed cells, the expectation
(:data:`EXPECT`, from proposal §3.5) and the diff per variant.

Usage: ``python acl_probe.py`` (from ``cpg/``; writes ``results/acl_matrix.json``)
"""

from __future__ import annotations

import json
import secrets
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lock_replay as LR  # noqa: E402

LIVE = "ccg_p0spike__acl"
BUILD_A, BUILD_B, PREV = "ccg_p0spike__acl_ba", "ccg_p0spike__acl_bb", "ccg_p0spike__acl_pv"
LOCK = LIVE + ":verdict_lock"
FOREIGN = "ccg_p0spike__zforeign"
FOREIGN_STR = "ccg_p0spike__zforeign_str"
TELEMETRY = "telemetry{" + LIVE + "}"
USERS = ("ccg_build", "ccg_verdict", "ccg_engine", "ccg_raw_p0")
HOME = {"ccg_build": BUILD_A, "ccg_verdict": LIVE, "ccg_engine": LIVE, "ccg_raw_p0": LIVE}

_LOCK_CMDS = ("+set", "+get", "+del", "+eval")


def acl_rules(user: str, variant: str) -> str:
    """The ACL rule string (without ``on >pw``) for ``user`` under ``variant``."""
    live_pat, sys_pat = f"~{LIVE}", f"~{LIVE}*"
    lock_pat = f"~{LOCK}"
    lock_flat = {"selector": "", "flat+get": "+set +get +del +eval", "spec": "+set +del +eval"}[variant]
    lock_sel = f"({lock_pat} +set +get +del +eval)" if variant == "selector" else ""
    if user == "ccg_build":       # write + RENAME + EXPIRE on the system's keys (+ EVAL for the guarded swap script)
        base = f"{sys_pat} +graph.query +graph.constraint +rename +expire +exists +info +eval"
        if variant != "selector":  # flat grants cannot be key-scoped: +eval is already there, add the rest flat
            base = f"{sys_pat} +graph.query +graph.constraint +rename +expire +exists +info {lock_flat}"
        return f"{base} {lock_sel}".strip()
    if user == "ccg_verdict":     # GRAPH.QUERY on the live key only + the lock key
        if variant == "selector":
            return f"{live_pat} +graph.query +info {lock_sel}"
        return f"{live_pat} {lock_pat} +graph.query +info {lock_flat}"
    if user == "ccg_engine":      # read
        return f"{live_pat} ~{LIVE}__ov_* +graph.ro_query +graph.explain +info"
    if user == "ccg_raw_p0":      # graph note V5/V16/V17: no GRAPH.LIST, +info for the constructor
        return f"{live_pat} ~{LIVE}__ov_* +graph.ro_query +graph.explain +info"
    raise ValueError(user)


#: Expected-allowed cells per user, from proposal §3.5 (+ `info` everywhere: the falkordb-py constructor needs it, V17;
#: engine reads RO_QUERY/EXPLAIN on the live key -- the proposal says only "(read)").
EXPECT = {
    "ccg_build": {"GRAPH.QUERY", "RENAME", "EXPIRE", "GRAPH.CONSTRAINT", "INFO", "SET lock", "EVAL release (GET+DEL in script)",
                  "DEL lock", "EVAL swap script (RENAME build->live)"},
    "ccg_verdict": {"GRAPH.QUERY", "INFO", "SET lock", "EVAL release (GET+DEL in script)", "DEL lock"},
    "ccg_engine": {"GRAPH.RO_QUERY", "GRAPH.EXPLAIN", "INFO"},
    "ccg_raw_p0": {"GRAPH.RO_QUERY", "GRAPH.EXPLAIN", "INFO"},
}

SWAP_LUA = ("if redis.call('EXISTS', KEYS[2]) == 0 then return 'refused: build absent' end "
            "redis.call('RENAME', KEYS[1], KEYS[3]) redis.call('RENAME', KEYS[2], KEYS[1]) return 'swapped'")


def tokens(rule: str) -> list[str]:
    """Split a rule string into ACL SETUSER arguments; a parenthesised selector is one argument."""
    toks, cur = [], None
    for t in rule.split():
        if cur is None and t.startswith("("):
            cur = [t]
        elif cur is not None:
            cur.append(t)
        else:
            toks.append(t)
            continue
        if t.endswith(")"):
            toks.append(" ".join(cur))
            cur = None
    return toks


def classify(fn) -> str:
    try:
        fn()
        return "allowed"
    except Exception as e:  # noqa: BLE001 -- the error text is the datum
        msg = str(e)
        if "NOPERM" in msg or "no permissions" in msg.lower():
            return "NOPERM-key" if "access a key" in msg.lower() else "NOPERM-cmd"
        return "other:" + msg.splitlines()[0][:100]


def fixtures(db) -> None:
    keys = (LIVE, BUILD_A, BUILD_B, PREV, FOREIGN, LOCK, FOREIGN_STR)
    LR.drop(db, *[k for k in keys if k not in (FOREIGN_STR,)])
    if db.connection.exists(FOREIGN_STR):
        db.connection.delete(FOREIGN_STR)
    for g in (LIVE, BUILD_A, FOREIGN):
        LR.make_graph(db, g, 3, 0, g)
    db.connection.set(FOREIGN_STR, "x")
    for _ in range(80):      # telemetry{LIVE} is created lazily after the first queries (S21: ~1 s; polled up to 8 s)
        db.select_graph(LIVE).ro_query("MATCH (n:Link) RETURN count(n)")
        if db.connection.exists(TELEMETRY):
            return
        time.sleep(0.1)
    raise AssertionError("telemetry stream not created within 8 s")


def cells(user: str, c, db_admin) -> dict:
    """Run every matrix cell as ``user`` over raw connection ``c``; the admin ``db_admin`` resets state between cells."""
    home = HOME[user]
    out: dict[str, str] = {}

    def reset():
        a = db_admin.connection
        for k in (LOCK,):
            a.delete(k)
        for g in (LIVE, BUILD_A, FOREIGN):
            if not a.exists(g):
                LR.make_graph(db_admin, g, 3, 0, g)
        for g in (BUILD_B, PREV):
            if a.exists(g):
                db_admin.select_graph(g).delete()
        a.persist(BUILD_A)
        a.persist(LIVE)

    def cell(name, fn):
        reset()
        out[name] = classify(fn)

    q = "MATCH (n:Link) RETURN count(n)"
    cell("GRAPH.QUERY", lambda: c.execute_command("GRAPH.QUERY", home, q))
    cell("GRAPH.RO_QUERY", lambda: c.execute_command("GRAPH.RO_QUERY", home, q))
    cell("GRAPH.EXPLAIN", lambda: c.execute_command("GRAPH.EXPLAIN", home, q))
    cell("GRAPH.PROFILE", lambda: c.execute_command("GRAPH.PROFILE", home, q))
    cell("GRAPH.LIST", lambda: c.execute_command("GRAPH.LIST"))
    cell("RENAME", lambda: c.execute_command("RENAME", BUILD_A, BUILD_B))
    cell("EXPIRE", lambda: c.execute_command("EXPIRE", home, 3600))
    cell("DEL (graph key)", lambda: c.execute_command("DEL", home))
    cell("GRAPH.CONSTRAINT", lambda: c.execute_command("GRAPH.CONSTRAINT", "CREATE", home, "UNIQUE", "NODE", "Link",
                                                       "PROPERTIES", 1, "linkId"))
    cell("INFO", lambda: c.execute_command("INFO"))
    cell("SET lock", lambda: c.execute_command("SET", LOCK, "tok", "NX", "PX", 5000))
    cell("EVAL release (GET+DEL in script)",
         lambda: (db_admin.connection.set(LOCK, "tok", px=5000), c.execute_command("EVAL", LR.RELEASE_LUA, 1, LOCK, "tok"))[1])
    cell("DEL lock", lambda: (db_admin.connection.set(LOCK, "tok", px=5000), c.execute_command("DEL", LOCK))[1])
    cell("XLEN telemetry", lambda: c.execute_command("XLEN", TELEMETRY))
    cell("foreign RO_QUERY", lambda: c.execute_command("GRAPH.RO_QUERY", FOREIGN, q))
    cell("foreign QUERY", lambda: c.execute_command("GRAPH.QUERY", FOREIGN, q))
    cell("EVAL: GET foreign key in script", lambda: c.execute_command("EVAL", "return redis.call('GET', ARGV[1])", 0, FOREIGN_STR))
    cell("EVAL: DEL graph key in script", lambda: c.execute_command("EVAL", "return redis.call('DEL', KEYS[1])", 1, home))
    cell("EVAL swap script (RENAME build->live)", lambda: c.execute_command("EVAL", SWAP_LUA, 3, LIVE, BUILD_A, PREV))
    return out


def constructor_ok(user: str, pw: str) -> str:
    try:
        LR.connect(user, pw)             # the constructor alone (V17); PING is not granted and is not the test
        return "ok"
    except Exception as e:  # noqa: BLE001
        return "fail:" + str(e).splitlines()[0][:100]


def run(out: Path | None = None) -> dict:
    admin = LR.connect()
    a = admin.connection
    existing = [x for x in a.execute_command("ACL", "USERS") if x in USERS]
    if existing:
        raise SystemExit(f"refusing: ACL users already exist on the instance: {existing}")
    pw = {u: secrets.token_urlsafe(18) for u in USERS}
    result: dict = {"variants": {}, "expect": {u: sorted(v) for u, v in EXPECT.items()}}
    try:
        for variant in ("selector", "flat+get", "spec"):
            fixtures(admin)
            vr: dict = {"rules": {u: acl_rules(u, variant) for u in USERS}, "cells": {}, "constructor": {}, "diff": {}}
            for u in USERS:
                a.execute_command("ACL", "SETUSER", u, "reset", "on", ">" + pw[u], "resetkeys", "-@all", *tokens(acl_rules(u, variant)))
                vr["constructor"][u] = constructor_ok(u, pw[u])
                udb = LR.connect(u, pw[u]) if vr["constructor"][u] == "ok" else None
                if udb is None:
                    vr["cells"][u] = {"constructor": vr["constructor"][u]}
                    continue
                vr["cells"][u] = cells(u, udb.connection, admin)
                allowed = {k for k, v in vr["cells"][u].items() if v == "allowed"}
                vr["diff"][u] = {"allowedButNotExpected": sorted(allowed - EXPECT[u]), "expectedButNotAllowed": sorted(EXPECT[u] - allowed)}
            result["variants"][variant] = vr
        # V17: the constructor needs +info
        a.execute_command("ACL", "SETUSER", "ccg_raw_p0", "-info")
        result["constructor_without_info"] = constructor_ok("ccg_raw_p0", pw["ccg_raw_p0"])
    finally:
        for u in USERS:
            try:
                a.execute_command("ACL", "DELUSER", u)
            except Exception:  # noqa: BLE001
                pass
        LR.drop(admin, LIVE, BUILD_A, BUILD_B, PREV, FOREIGN, LOCK)
        if a.exists(FOREIGN_STR):
            a.delete(FOREIGN_STR)
    result["usersLeft"] = [x for x in a.execute_command("ACL", "USERS") if x in USERS]
    if out:
        out.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    r = run(HERE / "results" / "acl_matrix.json")
    for v, vr in r["variants"].items():
        print(v, "constructor", vr["constructor"], "\n  diff", json.dumps(vr["diff"]))
    print("without +info:", r["constructor_without_info"], "usersLeft:", r["usersLeft"])
