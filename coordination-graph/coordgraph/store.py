"""The graph store: bootstrap and per-source replace (graph note v4 sections 3.2, 3.3, 4.2, 4.3).

The Cypher is the note's, verbatim: one literal statement per kind (labels cannot be
parameters), each atomic, each a pure function of `($path, $meta, $rows)`, so a retry is
harmless and a failed statement leaves the previous subgraph untouched.

Write rules (note section 3.2, "the store adds nothing the parser did not decide"):

* absent, not empty: a `None` map value is omitted, never stored;
* `safe_key()` is **asserted** on every UNIQUE value (`Source.path`, `Unit.key`, `Item.ref`) and
  `norm_value()` is **applied** to every indexed non-unique one (`Source.itemId`, `Unit.itemId`,
  `Ledger.slug`, `Backlog.scope`, `Unit.statusToken`), so whatever a parser forgot never reaches an
  index. A UNIQUE value is never rewritten here (the `~n`/hash form is the parser's), only refused:
  a bad one is an instance outage, not a query error;
* a non-unique indexed value must be non-empty and <= 512 UTF-8 bytes, else it is refused (FalkorDB
  silently leaves a value over 4,096 bytes unindexed, so an equality lookup would miss it);
* a NUL byte in any string fails the whole replace (`Failed to parse query parameters`), so it is
  replaced by U+FFFD and the source gains a warning naming how many;
* `warnings` and `rowCount` are supplied by the parser and written as given; the store adds only
  `syncedAt` (and `mtimeNs` when the caller has it).

`falkordb` is imported lazily, so the offline paths (`dryrun`) never load it.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .keys import MAX_KEY_BYTES, KeyHygieneError, norm_value, safe_key

SCHEMA_VERSION = 4  # the graph note version the DDL below implements
POLL_TIMEOUT_S = 5.0
SHARED_PORTS = frozenset({6379, 6390, 3000})
SHARED_KB_PORTS = frozenset({8200})  # the real falkor-chat (`COORDGRAPH_KB_URL`'s default)
EPHEMERAL_ENV = "COORDGRAPH_EPHEMERAL_PORT"
SHARED_KEYS = frozenset({"coordination"})  # the only graph shared mode may write: the mirror (note 5.5 step 5)
SHARED_CONFIRM_ENV = "COORDGRAPH_SHARED_CONFIRM"
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")

# note section 4.2. UNIQUE entries get their supporting RANGE index first (index before constraint).
UNIQUE: tuple[tuple[str, str], ...] = (
    ("Source", "path"), ("Unit", "key"), ("Item", "ref"), ("KbFile", "path"), ("KbSection", "key"),
)
RANGE_ONLY: tuple[tuple[str, str], ...] = (
    ("Ledger", "slug"), ("Backlog", "scope"), ("Unit", "statusToken"),
    ("Source", "itemId"), ("Unit", "itemId"),
)
INDEXES: tuple[tuple[str, str], ...] = UNIQUE + RANGE_ONLY

# kind -> (kind label, relationship, child label); review/requirement carry no children.
KINDS: dict[str, tuple[str, str | None, str | None]] = {
    "ledger": ("Ledger", "HAS_UNIT", "Unit"),
    "backlog": ("Backlog", "HAS_ITEM", "Item"),
    "review": ("Review", None, None),
    "requirement": ("Requirement", None, None),
}

_DELETE_OLD = """OPTIONAL MATCH (old:Source {path: $path})
OPTIONAL MATCH (old)-[]->(child)
WITH old, collect(child) AS children
FOREACH (c IN children | DETACH DELETE c)
FOREACH (o IN CASE WHEN old IS NULL THEN [] ELSE [old] END | DETACH DELETE o)
WITH count(*) AS _
"""


def _replace_statement(kind: str) -> str:
    label, rel, child = KINDS[kind]
    head = f"{_DELETE_OLD}CREATE (s:Source:{label})\nSET s = $meta\n"
    if rel is None:
        return head + "RETURN count(s) AS written"
    return (head + f"WITH s\nUNWIND $rows AS row\nCREATE (s)-[:{rel}]->(u:{child})\n"
            "SET u = row\nRETURN count(u) AS written")


REPLACE = {kind: _replace_statement(kind) for kind in KINDS}

DELETE_SOURCE = ("MATCH (s:Source {path:$path}) OPTIONAL MATCH (s)-[]->(c) "
                 "WITH s, collect(c) AS cs FOREACH (c IN cs | DETACH DELETE c) DELETE s")


class StoreError(RuntimeError):
    """A write or bootstrap that did not complete; the message names what and why."""


class UnsafeValue(StoreError, KeyHygieneError):
    """A key or indexed value the write boundary refuses. A `StoreError` so every caller that already
    reports a failed source (`sync`, `parity`, `drift`) handles it; still a `KeyHygieneError` for `except` sites
    that name the hygiene failure. (`keys.py` cannot subclass `StoreError`: `store` imports it.)"""


class RefusedEndpoint(StoreError):
    """The target FalkorDB is not the explicitly named ephemeral one."""


def require_ephemeral(host: str, port: int, env: Mapping[str, str] | None = None) -> None:
    """Refuse any endpoint but an explicitly announced ephemeral one.

    Allowed only on loopback, when `COORDGRAPH_EPHEMERAL_PORT` names exactly this port, and never on
    a shared port (6379 and the other known shared instances). The shared instance is reached only through
    `connect_shared`; this guard has no override.
    """
    env = os.environ if env is None else env
    declared = (env.get(EPHEMERAL_ENV) or "").strip()
    if host not in LOOPBACK_HOSTS:
        raise RefusedEndpoint(f"refusing non-loopback FalkorDB host {host!r}")
    if port in SHARED_PORTS:
        raise RefusedEndpoint(f"refusing shared FalkorDB port {port}")
    if declared != str(port):
        raise RefusedEndpoint(
            f"refusing FalkorDB port {port}: set {EPHEMERAL_ENV}={port} to confirm it is an ephemeral instance"
        )


def require_ephemeral_kb_url(url: str) -> None:
    """Refuse a falkor-chat REST endpoint that could be the shared one.

    The mirror graph is ephemeral-only (`require_ephemeral`), so a KB sync that POSTs to the shared
    falkor-chat would write real documents while observing an unrelated, empty graph. The two endpoints are
    named together: loopback host, and not a shared falkor-chat port (the unset default is one).
    """
    parts = urlsplit(url)
    if parts.hostname not in LOOPBACK_HOSTS:
        raise RefusedEndpoint(f"refusing non-loopback falkor-chat host {parts.hostname!r} for an ephemeral mirror")
    if parts.port in SHARED_KB_PORTS:
        raise RefusedEndpoint(
            f"refusing falkor-chat port {parts.port} (the shared instance, and the default when "
            "COORDGRAPH_KB_URL is unset): point COORDGRAPH_KB_URL at the scratch falkor-chat of this ephemeral run"
        )


def connect(host: str, port: int, key: str, env: Mapping[str, str] | None = None):
    """`(db, graph)` for an ephemeral instance; raises `RefusedEndpoint` otherwise."""
    require_ephemeral(host, port, env)
    from falkordb import FalkorDB  # lazy: offline paths never import it

    db = FalkorDB(host=host, port=port)
    return db, db.select_graph(key)


def require_shared(host: str, port: int, key: str, confirm: str | None, env: Mapping[str, str] | None = None) -> None:
    """Authorize one deliberate connection to the shared instance (the counterpart of `require_ephemeral`).

    `confirm` must be exactly the graph key about to be written, and that key must be the coordination mirror's
    (`SHARED_KEYS`): a confirmation can never name another component's graph. The port must be a shared one
    (`SHARED_PORTS`), so the two modes are exact complements. Loopback only. Never alongside an ephemeral
    declaration: `COORDGRAPH_EPHEMERAL_PORT` set means the caller believes it is on a scratch instance, so both at
    once is a mistake, not a mode.
    """
    env = os.environ if env is None else env
    if host not in LOOPBACK_HOSTS:
        raise RefusedEndpoint(f"refusing non-loopback FalkorDB host {host!r}")
    if (env.get(EPHEMERAL_ENV) or "").strip():
        raise RefusedEndpoint(f"refusing shared access while {EPHEMERAL_ENV} is set: pick one of the two modes")
    if port not in SHARED_PORTS:
        raise RefusedEndpoint(f"refusing FalkorDB port {port}: not a shared instance port {sorted(SHARED_PORTS)}")
    if key not in SHARED_KEYS:
        raise RefusedEndpoint(f"refusing graph {key!r} on the shared instance: not the coordination mirror "
                              f"({', '.join(sorted(SHARED_KEYS))})")
    if confirm != key:
        raise RefusedEndpoint(
            f"refusing shared FalkorDB {host}:{port}: --confirm / {SHARED_CONFIRM_ENV} must equal the graph key "
            f"{key!r}, got {confirm!r}"
        )


def connect_shared(host: str, port: int, key: str, confirm: str | None, env: Mapping[str, str] | None = None):
    """`(db, graph)` for the shared instance; raises `RefusedEndpoint` unless `confirm == key`."""
    require_shared(host, port, key, confirm, env)
    from falkordb import FalkorDB  # lazy: offline paths never import it

    db = FalkorDB(host=host, port=port)
    return db, db.select_graph(key)


def require_endpoint(host: str, port: int, key: str, confirm: str | None, env: Mapping[str, str] | None = None) -> None:
    """The pure guard `open_graph` applies, with no connection: lets a caller refuse before touching the network."""
    if confirm is None:
        require_ephemeral(host, port, env)
    else:
        require_shared(host, port, key, confirm, env)


def open_graph(host: str, port: int, key: str, confirm: str | None = None, env: Mapping[str, str] | None = None):
    """The one choke point commands use: ephemeral by default, shared only when `confirm` is given.

    `confirm=None` is `connect` (the ephemeral guard, unchanged); any other value, even an empty one, is
    `connect_shared`, so a blank or wrong confirmation refuses instead of silently falling back.
    """
    if confirm is None:
        return connect(host, port, key, env)
    return connect_shared(host, port, key, confirm, env)


def require_shared_kb_url(url: str) -> None:
    """The shared-mode counterpart of `require_ephemeral_kb_url`: the mirror is the shared graph, so the KB
    endpoint must be the real falkor-chat (loopback, a shared falkor-chat port), never a scratch one.
    The real falkor-chat (8200) fronts the `falkordb-dev` container published on 6379, the port a shared
    mirror connects to by default."""
    parts = urlsplit(url)
    if parts.hostname not in LOOPBACK_HOSTS:
        raise RefusedEndpoint(f"refusing non-loopback falkor-chat host {parts.hostname!r}")
    if parts.port not in SHARED_KB_PORTS:
        raise RefusedEndpoint(
            f"refusing falkor-chat port {parts.port} for a shared mirror: COORDGRAPH_KB_URL must be the shared "
            f"falkor-chat ({', '.join(str(p) for p in sorted(SHARED_KB_PORTS))})"
        )


# --- the write boundary -------------------------------------------------------------------

def _present(props: Mapping) -> dict:
    return {k: v for k, v in props.items() if v is not None}


def _unique(value, what: str) -> str:
    if not isinstance(value, str) or not value or safe_key(value) != value:
        raise UnsafeValue(f"{what} is not a safe UNIQUE value: {value!r}")
    return value


def _indexed(props: dict, *names: str) -> None:
    for name in names:
        if name in props:
            props[name] = value = norm_value(props[name])
            if not value or len(value.encode("utf-8")) > MAX_KEY_BYTES:
                raise UnsafeValue(f"indexed {name} must be non-empty and <= {MAX_KEY_BYTES} bytes: {value[:60]!r}")


def _scrub(value, count: list[int]):
    """`value` with every NUL in any string (nested in lists/maps) replaced by U+FFFD; `count[0]` tallies."""
    if isinstance(value, str):
        n = value.count("\x00")
        if n:
            count[0] += n
            return value.replace("\x00", "\ufffd")
        return value
    if isinstance(value, list):
        return [_scrub(v, count) for v in value]
    if isinstance(value, Mapping):
        return {k: _scrub(v, count) for k, v in value.items()}
    return value


def prepare_meta(kind: str, meta: Mapping) -> dict:
    """The `:Source` map to write: `None`s dropped, the key asserted, indexed values normalised."""
    props = _present(meta)
    _unique(props.get("path"), "Source.path")
    _indexed(props, "itemId")
    if kind == "ledger":
        _indexed(props, "slug")
    if kind == "backlog":
        _indexed(props, "scope")
    return props


def prepare_rows(kind: str, rows: list[Mapping]) -> list[dict]:
    out = []
    for row in rows:
        props = _present(row)
        if kind == "ledger":
            _unique(props.get("key"), "Unit.key")
            _indexed(props, "itemId", "statusToken")
        elif kind == "backlog":
            _unique(props.get("ref"), "Item.ref")
        out.append(props)
    return out


def write_form(kind: str, meta: Mapping, rows: list[Mapping] | None = None) -> tuple[dict, list[dict]]:
    """The exact `($meta minus syncedAt/mtimeNs, $rows)` a replace writes: NULs scrubbed (with a source
    warning), `None`s dropped, keys asserted, indexed values normalised. Parity expects this form."""
    count = [0]
    props = prepare_meta(kind, _scrub(dict(meta), count))
    out = prepare_rows(kind, _scrub(list(rows or []), count))
    if count[0]:
        props["warnings"] = [*props.get("warnings", []), f"{count[0]} NUL byte(s) replaced by U+FFFD"]
    return props, out


# --- bootstrap (note section 4.3) --------------------------------------------------------

def existing_schema(graph) -> tuple[set[tuple[str, str]], dict[tuple[str, str], str]]:
    """`({(label, prop)}, {(label, prop): status})` of the single-property indexes and constraints.

    `GRAPH.QUERY`, not `RO_QUERY`: on an absent key `RO_QUERY` errors, `QUERY` materialises it.
    """
    idx = graph.query("CALL db.indexes() YIELD label, properties RETURN label, properties")
    cons = graph.query("CALL db.constraints() YIELD label, properties, status RETURN label, properties, status")
    indexes = {(r[0], p) for r in idx.result_set for p in r[1]}
    constraints = {(r[0], p): r[2] for r in cons.result_set for p in r[1]}
    return indexes, constraints


@dataclass
class BootstrapResult:
    created_indexes: list[tuple[str, str]] = field(default_factory=list)
    created_constraints: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.created_indexes or self.created_constraints)


def bootstrap(db, graph, key: str, timeout: float = POLL_TIMEOUT_S) -> BootstrapResult:
    """Read-then-create: indexes first, then UNIQUE constraints, poll to OPERATIONAL, then `SyncState`.

    Re-running DDL errors on this build, so only what is missing is created; a second run changes
    nothing. Raises `StoreError` on a `FAILED` constraint or one still not `OPERATIONAL` after `timeout`.
    """
    result = BootstrapResult()
    indexes, constraints = existing_schema(graph)
    for label, prop in INDEXES:
        if (label, prop) not in indexes:
            graph.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
            result.created_indexes.append((label, prop))
    for label, prop in UNIQUE:
        if (label, prop) not in constraints:
            db.execute_command("GRAPH.CONSTRAINT", "CREATE", key, "UNIQUE", "NODE", label,
                               "PROPERTIES", "1", prop)
            result.created_constraints.append((label, prop))
    deadline = time.monotonic() + timeout
    while True:
        _, constraints = existing_schema(graph)
        status = {pair: constraints.get(pair) for pair in UNIQUE}
        failed = [pair for pair, s in status.items() if s == "FAILED"]
        if failed:
            raise StoreError(f"constraint FAILED: {failed}")
        # `status` is keyed by the expected pairs, so an empty listing reads None, never all-operational
        if all(s == "OPERATIONAL" for s in status.values()):
            break
        if time.monotonic() > deadline:
            raise StoreError(f"constraints not OPERATIONAL after {timeout}s: {status}")
        time.sleep(0.05)
    graph.query("MERGE (m:SyncState {id:'mirror'}) SET m.schemaVersion = $v", {"v": SCHEMA_VERSION})
    return result


# --- per-source replace (note section 3.2-3.3) -----------------------------------------------

def replace_source(graph, kind: str, meta: Mapping, rows: list[Mapping] | None = None, *,
                   synced_at_ms: int | None = None, mtime_ns: int | None = None) -> int:
    """Atomically replace one source's subgraph; returns the children written.

    Zero rows is safe: the `:Source` is created before the `UNWIND`, so only the `RETURN` row is
    dropped (children written read 0). On any failure the previous subgraph is untouched.
    """
    if kind not in KINDS:
        raise StoreError(f"unknown source kind {kind!r}")
    rows = rows or []
    if rows and KINDS[kind][1] is None:
        raise StoreError(f"a {kind} source has no children, got {len(rows)} rows")
    props, prepared = write_form(kind, meta, rows)
    props["syncedAt"] = int(time.time() * 1000) if synced_at_ms is None else synced_at_ms
    if mtime_ns is not None:
        props["mtimeNs"] = mtime_ns
    params = {"path": props["path"], "meta": props, "rows": prepared}
    try:
        res = graph.query(REPLACE[kind], params)
    except Exception as exc:  # falkordb ResponseError and friends: the statement rolled back
        raise _explain(graph, kind, props["path"], params["rows"], exc) from exc
    return res.result_set[0][0] if res.result_set else 0


def _explain(graph, kind: str, path: str, rows: list[dict], exc: Exception) -> StoreError:
    text = str(exc)
    if kind == "backlog" and "unique constraint violation" in text and "Item" in text:
        refs = [r["ref"] for r in rows]
        held = graph.query("MATCH (i:Item) WHERE i.ref IN $refs RETURN i.ref, i.backlogPath",
                           {"refs": refs}).result_set
        other = [(ref, bp) for ref, bp in held if bp != path]
        if other:
            ref, bp = other[0]
            ord_here = next((r.get("ord") for r in rows if r["ref"] == ref), "?")
            return StoreError(
                f"item {ref} headed in both {path} (ord {ord_here}) and {bp}: "
                f"unique constraint violation on node of type Item; a wi move may be in progress"
            )
    return StoreError(f"{path}: replace failed, previous state kept: {text}")


def delete_source(graph, path: str) -> int:
    """Delete one source and its children (a file that disappeared); returns nodes deleted."""
    res = graph.query(DELETE_SOURCE, {"path": _unique(path, "Source.path")})
    return res.nodes_deleted


class ConvergenceError(StoreError):
    """The file kept changing under every write round."""


def write_converged(graph, parse_current: Callable[[], tuple[str, Mapping, list[Mapping]]],
                    current_sha: Callable[[], str], rounds: int = 3) -> int:
    """Replace, then re-hash the file; re-parse and write again if it moved (note section 3.3).

    Closes the stale-parse race without a lock: the last writer to commit always re-checks against
    the file's final content. Returns the rounds used; `ConvergenceError` after `rounds`.
    """
    for n in range(1, rounds + 1):
        kind, meta, rows = parse_current()
        replace_source(graph, kind, meta, rows)
        if current_sha() == meta["sha256"]:
            return n
    raise ConvergenceError(f"file still changing after {rounds} write rounds")
