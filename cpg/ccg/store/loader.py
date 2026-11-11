"""The store loader (C-1001 P1-19, dispatch 1): graph rows -> a complete, constrained ``build`` key. The swap is ``ccg.store.swap``.

Contract: ``docs/plans/connected-code-graph-p1-graph.md`` §4 (DDL and load groups), §7 (credentials), §8 P1-19 and
``docs/plans/connected-code-graph-p0-graph.md`` §4, §6 (load protocol and measured ceilings). A port of the slice's
``spikes/vertical-slice/load.py`` (``LiveStore``, ``zero_retry``, ``batches``, ``poll_constraints``), re-keyed by system name.

The sequence, in this order and no other, on ``build_key(system, snapshotId)``:

1. refuse, before any store call, rows the loader may not write (an edge group outside the closed list, an unknown label, a row without its id property, an indexed string over 4,096 UTF-8 bytes or edged with ASCII whitespace, a ``:Snapshot`` that names another system or model);
2. the DDL's indexes on the EMPTY key; nodes, then edges, in ``UNWIND`` batches of at most :data:`BATCH_MAX` rows over ONE connection
   with retries ZERO; every query carries ``timeout`` = ``TIMEOUT_MAX`` whenever that is nonzero; every edge batch is count-checked;
3. the DDL's ``UNIQUE`` constraints, polled to ``OPERATIONAL`` (a ``FAILED`` fails the build);
4. ``:Snapshot`` last: its presence means "complete".

The loader owns **build keys only**: it never reads or writes the live key (the swap renames onto it) and refuses any key that is not
``ccg_<system>__build_<token>`` of this system (:func:`owned_build_key`), so a neighbouring system, a ``shop``-style graph or an overlay
can never be reached through it. A failed load removes its own build key (``DEL``, the one removal the build credential holds) and
re-raises; a killed one leaves the key behind, incomplete (no ``:Snapshot``), for the next load's ``drop_stale`` or the operator.
No Cypher text here is built from row data: labels and relationship types come from the closed tables below, validated as identifiers.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from ccg import indexed
from ccg.indexed import INDEXED_PROPS, INDEXED_VALUE_MAX_BYTES, INDEX_TRIMMED_CHARS  # noqa: F401  (re-exported: the loader's public names)
from ccg.store import parse_ddl
from ccg.sysname import build_key, check_system_name, live_key

BATCH_MAX = 20_000                  # p0-graph §4, §6 row 14: the hard ceiling measured is 100,000 and is never a target
SOCKET_TIMEOUT_S = 300              # writes ignore TIMEOUT; a client timeout abandons the answer, not the work
CONSTRAINT_POLL_TIMEOUT_S = 300
FORBIDDEN_PORT = 6379               # falkordb-dev: never touched
GRAPH_MODEL_VERSION = "1.1"         # tested equal to ``ccg.assemble.snapshot.GRAPH_MODEL_VERSION``

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class LoadError(RuntimeError):
    """A refusal or a failed build. Messages hold labels, groups, counts and key names, never a row value."""


def _ident(name: str) -> str:
    if not isinstance(name, str) or not _IDENT.fullmatch(name):
        raise ValueError(f"not a safe Cypher identifier: {name!r}")
    return name


# ---------------------------------------------------------------------------------------------
# Model tables (``-p1-graph.md`` §4). Owned here, not imported from the assembler: the loader is the last gate, so its closed list
# must not move with a producer's. ``tests/test_loader_unit.py`` asserts equality with the assembler's, so neither can drift alone.
# ---------------------------------------------------------------------------------------------

#: label -> its id property. ``SkippedFile`` has none: it is matched by ``(unitId, path)`` as an edge target.
ID_PROP = {
    "Repo": "repoId", "Unit": "unitId", "File": "fileId", "Symbol": "symbolId", "CallSite": "callSiteId",
    "Param": "pointId", "Ret": "pointId", "Cond": "pointId", "Name": "nameKey", "ContractPoint": "cpId", "ConfigEntry": "entryId",
    "Link": "linkId", "Dependency": "depId", "Pack": "packId",
}
NODE_ORDER = ("Repo", "Unit", "File", "Symbol", "Param", "Ret", "Cond", "CallSite", "Name", "ContractPoint", "ConfigEntry", "Link",
              "Dependency", "SkippedFile", "Pack")

#: The closed list of edge groups: v1's 29 plus the five of 1.1. A row in any other group fails the build before any write.
EDGE_ORDER = (
    ("Unit", "IN_REPO", "Repo"), ("Unit", "HAS_FILE", "File"), ("File", "DEFINES", "Symbol"),
    ("Symbol", "HAS_CALL", "CallSite"), ("Symbol", "HAS_PARAM", "Param"), ("Symbol", "HAS_RETURN", "Ret"),
    ("Symbol", "HAS_COND", "Cond"), ("CallSite", "CALLS", "Symbol"), ("CallSite", "CALLS_NAME", "Name"),
    ("Name", "RESOLVES_TO", "Symbol"),
    ("CallSite", "DATA", "Param"), ("CallSite", "DATA", "CallSite"), ("Ret", "DATA", "Param"),
    ("Ret", "DATA", "CallSite"), ("Cond", "DATA", "Param"), ("Cond", "DATA", "CallSite"),
    ("CallSite", "CONTROLLED_BY", "Cond"), ("Ret", "CONTROLLED_BY", "Cond"), ("Cond", "CONTROLLED_BY", "Cond"),
    ("Param", "CONTROLLED_BY", "Cond"),
    ("Symbol", "PROVIDES", "ContractPoint"), ("CallSite", "CONSUMES", "ContractPoint"),
    ("CallSite", "VIA", "Link"), ("Link", "TO", "Symbol"), ("Link", "ON", "ContractPoint"),
    ("Link", "LIFTED_VIA", "CallSite"), ("Link", "LIFTED_VIA", "Symbol"),
    ("Unit", "DEPENDS_ON", "Dependency"), ("Unit", "SKIPS", "SkippedFile"),
    ("Link", "BOUND_BY", "ConfigEntry"), ("ConfigEntry", "LAUNCHES", "File"), ("ConfigEntry", "LAUNCHES", "Unit"),
    ("Symbol", "MOUNTED_BY", "CallSite"), ("Link", "TO_UNIT", "Unit"),
)

SNAPSHOT_STATEMENT = "CREATE (s:Snapshot) SET s = $props"

# The indexed-value predicate and the property list live in ``ccg.indexed`` (shared with the assembler, which refuses the one unit first; this
# preflight is the fail-closed backstop).


# ---------------------------------------------------------------------------------------------
# Keys.
# ---------------------------------------------------------------------------------------------


def owned_build_key(system: str, key: str) -> str:
    """``key`` if it is a build key of ``system`` (formed by ``ccg.sysname``), else :class:`LoadError`.

    Not owned: the live key, any other system's key (``<s>2``, ``<s>__x``, ``<s>_``), an overlay, the lock, a bare ``shop``-style graph.
    The check round-trips the suffix through ``sysname.build_key``, so the token grammar is the one the ACL pattern relies on."""
    prefix = live_key(system) + "__build_"
    if isinstance(key, str) and key.startswith(prefix) and len(key) > len(prefix):
        try:
            if build_key(system, key[len(prefix):]) == key:
                return key
        except ValueError:
            pass
    raise LoadError(f"refusing key {key!r}: not owned by the loader of system {system!r} (only its build keys are)")


# ---------------------------------------------------------------------------------------------
# Statements.
# ---------------------------------------------------------------------------------------------


def batches(rows: list, size: int = BATCH_MAX) -> Iterator[list]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def index_statements() -> list[str]:
    """Part 1 of ``ddl_v1.cypher``, in file order: the DDL file is the one home of the set."""
    return [s[3] for s in parse_ddl() if s[0] == "index"]


def constraint_statements() -> list[tuple[str, str]]:
    """Part 2 of the DDL file, in file order: ``(label, property)`` of each ``UNIQUE`` constraint."""
    return [(_ident(s[1]), _ident(s[2])) for s in parse_ddl() if s[0] == "constraint"]


def node_statement(label: str) -> str:
    return f"UNWIND $rows AS r CREATE (n:{_ident(label)}) SET n = r"


def edge_statement(src: str, etype: str, dst: str, with_props: bool) -> str:
    """``CREATE``, never ``MERGE``: two ``MOUNTED_BY`` edges between one ``Symbol`` and ``CallSite`` differ only by ``mountPath`` and ``ord``
    (``-p1-graph.md`` G6). Both endpoints are matched by an index that exists before the load."""
    # a SkippedFile has no id: it is matched by (unit, path), the unit id being the edge's source id
    target = "unitId: r.s, path: r.t" if dst == "SkippedFile" else f"{_ident(ID_PROP[dst])}: r.t"
    text = (f"UNWIND $rows AS r MATCH (a:{_ident(src)} {{{_ident(ID_PROP[src])}: r.s}}) "
            f"MATCH (b:{_ident(dst)} {{{target}}}) CREATE (a)-[e:{_ident(etype)}]->(b)")
    return text + " SET e = r.p" if with_props else text


# ---------------------------------------------------------------------------------------------
# Stores: the thin seam over the database (a fake in the unit tests, ``LiveStore`` for real).
# ---------------------------------------------------------------------------------------------


class Store(Protocol):
    def timeout_max(self) -> int: ...
    def exists(self, key: str) -> bool: ...
    def query(self, key: str, cypher: str, params: dict | None = None, timeout: int | None = None) -> Any: ...
    def create_unique_constraint(self, key: str, label: str, prop: str) -> None: ...
    def constraints(self, key: str, timeout: int | None = None) -> list[dict]: ...
    def delete(self, key: str) -> None: ...


def zero_retry():
    """An explicit zero-retry policy. ``retry=None`` is the client default, which redis-py replaces with its own retrying policy: a
    retried write could run twice (``-p0-graph.md`` §4)."""
    from redis.backoff import NoBackoff
    from redis.retry import Retry
    return Retry(NoBackoff(), 0)


def connect(host: str, port: int, *, username: str | None, password: str, socket_timeout: float = SOCKET_TIMEOUT_S):
    """A falkordb-py client: one connection, retries zero, a socket timeout of minutes. ``username`` None is the default (admin) user.

    No ``PING``: the role users do not hold ``+ping``. Port 6379 (``falkordb-dev``) is refused before any socket exists."""
    if port == FORBIDDEN_PORT:
        raise LoadError(f"port {FORBIDDEN_PORT} is falkordb-dev; this loader never touches it")
    from falkordb import FalkorDB
    return FalkorDB(host=host, port=port, username=username, password=password, socket_timeout=socket_timeout,
                    socket_connect_timeout=10, retry=zero_retry())


class LiveStore:
    """The store over one falkordb-py client (``connect``). Every key goes through :func:`owned_build_key` BEFORE the connection is used."""

    def __init__(self, db, system: str):
        self.system = check_system_name(system)
        self.db = db

    @property
    def conn(self):
        return self.db.connection

    def _own(self, key: str) -> str:
        return owned_build_key(self.system, key)

    def timeout_max(self) -> int:
        return int(self.conn.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")[1])

    def exists(self, key: str) -> bool:
        key = self._own(key)
        return bool(self.conn.exists(key))

    def query(self, key, cypher, params=None, timeout=None):
        key = self._own(key)
        graph = self.db.select_graph(key)
        return graph.query(cypher, params, timeout) if timeout else graph.query(cypher, params)

    def create_unique_constraint(self, key, label, prop):
        # the raw command, never falkordb-py's create_node_unique_constraint: that helper silently creates a missing range index first,
        # which would mask a gap in ddl_v1.cypher (falkordb-quirks.md, 2026-11-01)
        key = self._own(key)
        self.conn.execute_command("GRAPH.CONSTRAINT", "CREATE", key, "UNIQUE", "NODE", _ident(label), "PROPERTIES", 1, _ident(prop))

    def constraints(self, key, timeout=None):
        res = self.query(key, "CALL db.constraints()", None, timeout)
        return [{"type": r[0], "label": r[1], "properties": r[2], "entity": r[3], "status": r[4]} for r in res.result_set]

    def delete(self, key):
        """``DEL`` of a build key: the removal the build credential holds (``+del`` on ``…__build_*``); ``GRAPH.DELETE`` is granted to nobody."""
        key = self._own(key)
        self.conn.delete(key)

    def used_memory(self) -> int:
        return int(self.conn.info("memory")["used_memory"])

    def memory(self, key) -> dict:
        """``GRAPH.MEMORY USAGE`` (integer MB, coarse). The role users do not hold it: measure with a credential that does."""
        key = self._own(key)
        raw = self.conn.execute_command("GRAPH.MEMORY", "USAGE", key)
        return {raw[i]: raw[i + 1] for i in range(0, len(raw), 2)}


# ---------------------------------------------------------------------------------------------
# The load.
# ---------------------------------------------------------------------------------------------


@dataclass
class LoadReport:
    system: str = ""
    build_key: str = ""
    live_key: str = ""
    snapshot_id: str = ""
    nodes: dict[str, int] = field(default_factory=dict)
    edges: dict[str, int] = field(default_factory=dict)
    node_total: int = 0
    edge_total: int = 0
    batches: int = 0
    max_batch: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    timeout_max: int = 0
    query_timeout_used: bool = False
    constraints: list[dict] = field(default_factory=list)
    used_memory: dict[str, int] = field(default_factory=dict)
    #: ids of the loaded units no sandboxed invocation produced (the assembly report's ``unattested_units``): a report line only, never written
    #: to the graph (``-bundle-spec6.md``, "load report (P1-19)")
    unattested_units: list[str] = field(default_factory=list)


def _preflight(rows, system: str) -> None:
    """Everything refusable without a store: runs before the first call, so a refusal writes and reads nothing."""
    snap = rows.snapshot
    if not isinstance(snap, dict) or not isinstance(snap.get("snapshotId"), str):
        raise LoadError("the :Snapshot row has no snapshotId")
    if snap.get("systemId") != system:
        raise LoadError(f"the :Snapshot row's systemId is not this load's system {system!r}")
    if snap.get("graphModelVersion") != GRAPH_MODEL_VERSION:
        raise LoadError(f"the :Snapshot row's graphModelVersion is not {GRAPH_MODEL_VERSION!r}")
    unknown_nodes = sorted(set(rows.nodes) - set(NODE_ORDER))
    if unknown_nodes:
        raise LoadError(f"node labels with no load statement: {unknown_nodes}")
    unknown_edges = sorted(set(rows.edges) - set(EDGE_ORDER))
    if unknown_edges:
        raise LoadError(f"edge groups with no load statement: {unknown_edges}")
    for label, label_rows in rows.nodes.items():
        props = (ID_PROP[label],) if label in ID_PROP else ("unitId", "path")
        for i, row in enumerate(label_rows):
            for prop in props:
                if not isinstance(row, dict) or not isinstance(row.get(prop), str) or not row[prop]:
                    raise LoadError(f"{label} row {i} lacks {prop} (absent, or not a non-empty string)")
            for prop in indexed.props_of(label):
                bad = indexed.check(row.get(prop))
                if bad is None:
                    continue
                if bad.reason == indexed.NOT_STR:
                    raise LoadError(f"{label} row {i}: {prop} is not a string")
                if bad.reason == indexed.UNICODE:
                    raise LoadError(f"{label} row {i}: {prop} is not valid Unicode text")
                if bad.reason == indexed.EMPTY:
                    raise LoadError(f"{label} row {i}: {prop} is an empty string, which an equality anchor on the engine's index never matches")
                if bad.reason == indexed.WHITESPACE:
                    raise LoadError(f"{label} row {i}: {prop} starts or ends with ASCII whitespace, which the engine's index trims "
                                    "(an equality anchor would miss it or match other values)")
                raise LoadError(f"{label} row {i}: {prop} is {bad.size:,} UTF-8 bytes, over the {bad.limit:,} an indexed value may hold "
                                "(the engine neither indexes it nor survives a unique constraint over it)")
    for group, edge_rows in rows.edges.items():
        for i, row in enumerate(edge_rows):
            if not isinstance(row, dict) or "s" not in row or "t" not in row:
                raise LoadError(f"{group[0]}-[{group[1]}]->{group[2]} row {i} lacks s or t")


def _write(store: Store, key: str, cypher: str, params: dict, timeout: int | None, report: LoadReport) -> Any:
    report.batches += 1
    report.max_batch = max(report.max_batch, len(params.get("rows", ())))
    return store.query(key, cypher, params, timeout)


def poll_constraints(store: Store, key: str, *, expected: int, interval: float = 0.25, timeout: float = CONSTRAINT_POLL_TIMEOUT_S,
                     sleep=time.sleep, clock=time.monotonic, query_timeout: int | None = None) -> list[dict]:
    """Poll ``db.constraints()`` until all ``expected`` are ``OPERATIONAL``; any ``FAILED`` is a failed build."""
    deadline = clock() + timeout
    while True:
        cons = store.constraints(key, query_timeout)
        failed = [c for c in cons if c["status"] == "FAILED"]
        if failed:
            raise LoadError(f"constraint FAILED on {key}: {[(c['label'], c['properties']) for c in failed]}")
        if len(cons) >= expected and all(c["status"] == "OPERATIONAL" for c in cons):
            return cons
        if clock() >= deadline:
            raise LoadError(f"constraints not OPERATIONAL within {timeout}s on {key}: {[c['status'] for c in cons]}")
        sleep(interval)


def _load_build(store: Store, rows, build: str, report: LoadReport, timeout: int | None, batch_size: int, poll_interval: float,
                poll_timeout: float, sleep, clock, measure_memory: bool) -> None:
    if measure_memory:
        report.used_memory["before"] = store.used_memory()  # type: ignore[attr-defined]
    t0 = clock()
    for stmt in index_statements():
        store.query(build, stmt, None, timeout)
    report.seconds["indexes"] = clock() - t0

    t0 = clock()
    for label in NODE_ORDER:
        label_rows = rows.nodes.get(label, [])
        for chunk in batches(label_rows, batch_size):
            _write(store, build, node_statement(label), {"rows": chunk}, timeout, report)
        report.nodes[label] = len(label_rows)
    report.seconds["nodes"] = clock() - t0

    t0 = clock()
    for group in EDGE_ORDER:
        group_rows = rows.edges.get(group, [])
        if not group_rows:
            continue
        src, etype, dst = group
        with_props = any("p" in r for r in group_rows)
        for chunk in batches(group_rows, batch_size):
            if with_props:  # a row without props is a driver error in a mixed batch; an empty map sets nothing
                chunk = [r if "p" in r else {**r, "p": {}} for r in chunk]
            res = _write(store, build, edge_statement(src, etype, dst, with_props), {"rows": chunk}, timeout, report)
            created = getattr(res, "relationships_created", None)
            if created is None:
                raise LoadError(f"{src}-[{etype}]->{dst}: the reply has no created-edge count, so the batch cannot be checked")
            created = int(created)   # falkordb-py returns the statistic as a float (1.0)
            if created != len(chunk):
                raise LoadError(f"{src}-[{etype}]->{dst}: {len(chunk)} rows but {created} edges created (an endpoint did not match)")
        report.edges[f"{src}-{etype}->{dst}"] = len(group_rows)
    report.seconds["edges"] = clock() - t0

    t0 = clock()
    constraints = constraint_statements()
    for label, prop in constraints:
        store.create_unique_constraint(build, label, prop)
    report.constraints = poll_constraints(store, build, expected=len(constraints), interval=poll_interval, timeout=poll_timeout,
                                          sleep=sleep, clock=clock, query_timeout=timeout)
    report.seconds["constraints"] = clock() - t0
    if measure_memory:
        report.used_memory["after"] = store.used_memory()  # type: ignore[attr-defined]

    store.query(build, SNAPSHOT_STATEMENT, {"props": rows.snapshot}, timeout)  # last: its presence means "complete"
    report.node_total = sum(report.nodes.values()) + 1
    report.edge_total = sum(report.edges.values())


def load(store: Store, rows, system: str, report: LoadReport | None = None, *, drop_stale: bool = False, batch_size: int = BATCH_MAX,
         poll_interval: float = 0.25, poll_timeout: float = CONSTRAINT_POLL_TIMEOUT_S, sleep=time.sleep, clock=time.monotonic,
         measure_memory: bool = False) -> LoadReport:
    """Build ``rows`` (an ``ccg.assemble.GraphRows``) into ``build_key(system, rows.snapshot['snapshotId'])`` and return the report.

    Does not swap. An existing build key of the same snapshot is an interrupted load: refused unless ``drop_stale`` (then removed first).
    A failure after the key exists removes it and re-raises; ``measure_memory`` needs a store with ``used_memory()`` (the report's
    ``used_memory`` before the first index and after the last constraint is ``OPERATIONAL``)."""
    system = check_system_name(system)
    if not isinstance(batch_size, int) or not 1 <= batch_size <= BATCH_MAX:
        raise ValueError(f"batch_size must be 1..{BATCH_MAX}, got {batch_size!r}")
    _preflight(rows, system)
    build = build_key(system, rows.snapshot["snapshotId"])
    report = report if report is not None else LoadReport()
    report.system, report.build_key, report.live_key = system, build, live_key(system)
    report.snapshot_id, report.dropped = rows.snapshot["snapshotId"], dict(rows.dropped)
    report.unattested_units = list(rows.unattested_units)

    report.timeout_max = int(store.timeout_max())
    timeout = report.timeout_max or None  # explicit and <= TIMEOUT_MAX whenever the cap is nonzero (p0-graph §4)
    report.query_timeout_used = timeout is not None
    if store.exists(build):
        if not drop_stale:
            raise LoadError(f"{build} already exists (an interrupted load?); rerun with drop_stale")
        store.delete(build)
    t0 = clock()
    try:
        _load_build(store, rows, build, report, timeout, batch_size, poll_interval, poll_timeout, sleep, clock, measure_memory)
    except BaseException:
        try:
            if store.exists(build):
                store.delete(build)
        except Exception:  # noqa: BLE001 -- cleanup is best effort and must not mask the failure that caused it
            pass
        raise
    report.seconds["total"] = clock() - t0
    return report
