"""Shared fixtures for ``test_loader_*.py`` (P1-19): a small ``GraphRows`` that exercises every edge group of
``docs/plans/connected-code-graph-p1-graph.md`` §4, a fake ``Store``, a fake swap connection, and the throwaway-instance
plumbing of the live tests. Nothing here touches ``~/.config/ccg``, ``falkordb-dev`` or ``ccg-falkordb``."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccg import ids
from ccg.assemble.rows import EDGE_GROUPS, ID_PROP, GraphRows

#: the empty verdict log's stamp (answer spec §4.2; ``-p1-graph.md`` §3), written out here, not read from ``ccg``
EMPTY_HASH = "c482b1cf54631b37eb4572dc86a68fb1"


def snapshot_row(system: str = "shop", tag: str = "m1") -> dict:
    return {"snapshotId": ids.tuple_id(["snapshot", system, tag]), "systemId": system, "graphModelVersion": "1.1",
            "bundleSchemaVersion": "1.1", "vocabVersion": "1", "manifestHash": "m" * 64, "packSetHash": "p" * 32,
            "codeHash": "c" * 64, "auditSetHash": "a" * 64, "verdictLogOffset": 0, "verdictLogHash": EMPTY_HASH}


def tiny_rows(system: str = "shop", *, tag: str = "m1") -> GraphRows:
    """Two nodes of every labelled label, one ``SkippedFile``, one edge per closed-list group (two for ``MOUNTED_BY`` between the
    same ``Symbol`` and ``CallSite``, G6, and two for ``CallSite DATA Param``, one with props and one without)."""
    r = GraphRows()
    for label, prop in ID_PROP.items():
        for i in (1, 2):
            row = {prop: f"{label}{i}", "tag": f"{label}-{i}"}
            if label == "Unit":
                row["flowAvailable"] = i == 1
            r.nodes.setdefault(label, []).append(row)
    r.nodes["SkippedFile"] = [{"repoId": "Repo1", "unitId": "Unit1", "path": "a/b.py", "reason": "test"}]
    for src, etype, dst in EDGE_GROUPS:
        if dst == "SkippedFile":
            r.edges[(src, etype, dst)] = [{"s": "Unit1", "t": "a/b.py"}]
            continue
        row = {"s": f"{src}1", "t": f"{dst}{2 if src == dst else 1}"}
        r.edges[(src, etype, dst)] = [row]
    r.edges[("Symbol", "MOUNTED_BY", "CallSite")] = [
        {"s": "Symbol1", "t": "CallSite1", "p": {"ord": 0, "mountPath": "/a"}},
        {"s": "Symbol1", "t": "CallSite1", "p": {"ord": 1, "mountPath": "/b"}},
    ]
    r.edges[("CallSite", "DATA", "Param")] = [
        {"s": "CallSite1", "t": "Param1", "p": {"status": "possible", "reason": "field-truncated"}},
        {"s": "CallSite2", "t": "Param2"},
    ]
    r.snapshot = snapshot_row(system, tag)
    return r


def expected_edge_counts(rows: GraphRows) -> dict[str, int]:
    return {etype: sum(len(v) for (s, t, d), v in rows.edges.items() if t == etype) for etype in {k[1] for k in rows.edges}}


class FakeStore:
    """Records every call; ``created`` shifts the reported created-edge count (to plant a lost edge)."""

    def __init__(self, *, timeout_max: int = 10000, existing=(), pending_polls: int = 0, failed_label: str | None = None,
                 created_delta: int = 0, created_none: bool = False, fail_on=None):
        self.calls: list[tuple] = []
        self.keys = set(existing)
        self._timeout_max, self.pending_polls, self.failed_label = timeout_max, pending_polls, failed_label
        self.created_delta, self.created_none, self.fail_on = created_delta, created_none, fail_on
        self.constraints_made: list[tuple[str, str]] = []

    # -- the Store protocol ------------------------------------------------------------------------
    def timeout_max(self) -> int:
        self.calls.append(("timeout_max",))
        return self._timeout_max

    def exists(self, key: str) -> bool:
        self.calls.append(("exists", key))
        return key in self.keys

    def query(self, key, cypher, params=None, timeout=None):
        self.calls.append(("query", key, cypher, params, timeout))
        if self.fail_on and self.fail_on(len([c for c in self.calls if c[0] == "query"]), cypher):
            raise self.fail_on.exc
        self.keys.add(key)
        n = len(params["rows"]) if params and "rows" in params else 0
        if "CREATE (a)-[e:" in cypher:
            return SimpleNamespace(relationships_created=None if self.created_none else n + self.created_delta)
        return SimpleNamespace(relationships_created=0)

    def create_unique_constraint(self, key, label, prop):
        self.calls.append(("constraint", key, label, prop))
        self.constraints_made.append((label, prop))

    def constraints(self, key, timeout=None):
        self.calls.append(("constraints", key, timeout))
        status = "PENDING" if self.pending_polls > 0 else "OPERATIONAL"
        if self.pending_polls > 0:
            self.pending_polls -= 1
        return [{"type": "UNIQUE", "label": l, "properties": [p], "entity": "NODE",
                 "status": "FAILED" if l == self.failed_label else status} for l, p in self.constraints_made]

    def delete(self, key):
        self.calls.append(("delete", key))
        self.keys.discard(key)

    # -- views -------------------------------------------------------------------------------------
    def ops(self, name):
        return [c for c in self.calls if c[0] == name]

    def queries(self):
        return self.ops("query")


class Boom(RuntimeError):
    pass


def fail_at(n: int, exc: BaseException | None = None):
    """``fail_on`` predicate: raise on the n-th query."""
    def pred(i, cypher):
        return i == n
    pred.exc = exc or Boom("planted")
    return pred


# ------------------------------------------------------------------------------------------------ swap fakes

class FakeRedis:
    """The few commands ``swap`` uses, with the lock script's real semantics for the one script the module ships."""

    def __init__(self, log, name, shared, release_lua, stamps):
        self.log, self.name, self.shared, self.release_lua, self.stamps = log, name, shared, release_lua, stamps

    def set(self, key, val, nx=False, px=None):
        self.log.append((self.name, "set", key, val, nx, px))
        if nx and key in self.shared:
            return None
        self.shared[key] = val
        return True

    def get(self, key):
        self.log.append((self.name, "get", key))
        return self.shared.get(key)

    def exists(self, key):
        self.log.append((self.name, "exists", key))
        return int(key in self.shared)

    def rename(self, src, dst):
        self.log.append((self.name, "rename", src, dst))
        if src not in self.shared:
            raise RuntimeError("no such key")
        self.shared[dst] = self.shared.pop(src)
        if src in self.stamps:
            self.stamps[dst] = self.stamps.pop(src)
        return True

    def delete(self, *keys):
        self.log.append((self.name, "delete") + keys)
        return sum(self.shared.pop(k, None) is not None for k in keys)

    def eval(self, script, numkeys, *args):
        self.log.append((self.name, "eval", numkeys) + args)
        if script != self.release_lua:
            raise RuntimeError("unknown script")
        key, token = args
        if self.shared.get(key) == token:
            del self.shared[key]
            return 1
        return 0


class FakeGraph:
    def __init__(self, db, key):
        self.db, self.key = db, key

    def _run(self, op, cypher, params, timeout):
        self.db.log.append((self.db.name, op, self.key, cypher))
        self.db.timeouts.append((op, self.key, timeout))
        stamp = self.db.stamps.get(self.key)
        rows = [] if stamp is None else [list(stamp)]
        return SimpleNamespace(result_set=rows)

    def ro_query(self, cypher, params=None, timeout=None):
        return self._run("ro_query", cypher, params, timeout)

    def query(self, cypher, params=None, timeout=None):
        return self._run("query", cypher, params, timeout)


class FakeDb:
    """A falkordb-py-shaped object: ``.connection`` (redis) and ``.select_graph``."""

    def __init__(self, log, name, shared, stamps, release_lua):
        self.log, self.name, self.stamps = log, name, stamps
        self.timeouts: list[tuple] = []      # (op, key, timeout) of every graph query, as the caller passed it
        self.connection = FakeRedis(log, name, shared, release_lua, stamps)

    def select_graph(self, key):
        return FakeGraph(self, key)


def fake_pair(release_lua: str, *, keys=None):
    """(build_db, verdict_db, log, shared, stamps). ``shared`` is the keyspace (graph keys map to a marker); ``stamps`` maps a graph
    key to the ``(verdictLogOffset, verdictLogHash)`` of its one ``Snapshot`` (absent: no Snapshot). ``RENAME`` moves both."""
    log: list[tuple] = []
    shared: dict = {} if keys is None else dict(keys)
    stamps: dict = {}
    return (FakeDb(log, "build", shared, stamps, release_lua), FakeDb(log, "verdict", shared, stamps, release_lua),
            log, shared, stamps)


# ------------------------------------------------------------------------------------------------ live plumbing

#: ports the live tests refuse outright (falkordb-dev, the shared ccg-falkordb)
FORBIDDEN_PORTS = (6379, 6390)


def throwaway_port() -> int | None:
    """The port of a throwaway instance named by ``CCG_FALKORDB_PORT``, or None. It must be published by a container that carries the
    label ``ccg-falkordb.throwaway=1`` (the label only ``infra/ccg-falkordb.sh --throwaway`` sets); anything else is not used."""
    raw = os.environ.get("CCG_FALKORDB_PORT")
    if not raw or not raw.isdigit() or int(raw) in FORBIDDEN_PORTS:
        return None
    port = int(raw)
    try:
        out = subprocess.run(["docker", "ps", "--filter", "label=ccg-falkordb.throwaway=1", "--format", "{{.Ports}}"],
                             capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return port if f"127.0.0.1:{port}->" in out else None


def admin_password() -> str:
    """The throwaway's admin password, from ``$CCG_CONFIG_DIR`` only (the tests never read ``~/.config/ccg``)."""
    d = os.environ.get("CCG_CONFIG_DIR")
    if not d or Path(d).expanduser().resolve() == (Path.home() / ".config" / "ccg").resolve():
        raise FileNotFoundError("CCG_CONFIG_DIR must name a scratch directory, not ~/.config/ccg")
    return (Path(d) / "falkordb-admin.pass").read_text().strip()


def skip_or_fail(reason: str) -> None:
    """A live test's "no instance" exit: a skip, unless ``CCG_REQUIRE_LIVE=1`` (the gate's switch), which makes it a failure so a gate or CI
    cannot pass with every live test silently skipped."""
    if os.environ.get("CCG_REQUIRE_LIVE") == "1":
        pytest.fail(f"CCG_REQUIRE_LIVE=1 and the live tests cannot run: {reason}", pytrace=False)
    pytest.skip(reason)
