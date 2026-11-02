"""S13 (P0-12) -- load the slice's fact bundles and the S12 link facts into ``ccg-falkordb``.

Contract: ``docs/plans/connected-code-graph-p0-2.md`` S13 row and D16; graph note
``docs/plans/connected-code-graph-graph.md`` §3.4 (indexes, constraints), §3.5 (load sequence),
§6.3 (points, ``DATA``); bundle spec v1.3 §9.1/§1.11 (record -> graph).

The sequence (graph note §3.5), in this order and no other:

1. a fresh key ``ccg_p0slice__build_<snapshotId>``, then its indexes (the key is empty);
2. nodes, then edges grouped by (source label, type, target label), as ``UNWIND $rows`` batches of at
   most :data:`BATCH_MAX` rows (D16) over one connection: retries off, a socket timeout of minutes, and an
   explicit query ``timeout`` whenever ``TIMEOUT_MAX`` is nonzero;
3. the ``UNIQUE`` constraints, polled to ``OPERATIONAL`` (any ``FAILED`` fails the build);
4. the ``:Snapshot`` node, last (its presence means "complete");
5. ``RENAME`` to ``ccg_p0slice``.

Reservation R-B: only the keys ``ccg_p0slice`` and ``ccg_p0slice__build_*`` are ever touched
(:func:`check_key` guards every store call), and port 6379 is refused outright.

Row building (:func:`build_rows`) is pure: no database, no clock. Every value reaches Cypher as a
parameter; label and relationship-type names come from the fixed tables below and are validated.

Usage: ``python load.py --host 127.0.0.1 --port 6390 BUNDLE.jsonl [BUNDLE.jsonl ...] [--drop-stale]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # cpg/

from ccg.bundle import read_bundle  # noqa: E402
from ccg.ids import contract_point_id, name_key, tuple_id  # noqa: E402
from ccg.store import parse_ddl  # noqa: E402

SYSTEM = "p0slice"
FINAL_KEY = f"ccg_{SYSTEM}"
BUILD_PREFIX = f"{FINAL_KEY}__build_"
BATCH_MAX = 20_000  # D16: until S23 records a ceiling, every P0 write uses at most 20k rows
SOCKET_TIMEOUT_S = 300  # "a socket timeout of minutes": writes ignore TIMEOUT, a client timeout abandons the answer, not the work
CONSTRAINT_POLL_TIMEOUT_S = 300
FORBIDDEN_PORT = 6379  # falkordb-dev: never touched

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEY = re.compile(rf"{FINAL_KEY}(__build_[A-Za-z0-9_]+)?")


class LoadError(RuntimeError):
    pass


def check_key(key: str) -> str:
    """R-B: only ``ccg_p0slice`` and ``ccg_p0slice__build_*`` may be touched."""
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise LoadError(f"refusing key {key!r}: R-B allows only {FINAL_KEY!r} and {BUILD_PREFIX!r}*")
    return key


def _ident(name: str) -> str:
    if not _IDENT.fullmatch(name):
        raise ValueError(f"not a safe Cypher identifier: {name!r}")
    return name


# ---------------------------------------------------------------------------------------------
# Model tables.
# ---------------------------------------------------------------------------------------------

#: label -> its id property (graph note §3.4, bundle spec §1.11 S-1: Param/Ret/Cond carry ``pointId``)
ID_PROP = {
    "Repo": "repoId", "Unit": "unitId", "File": "fileId", "Symbol": "symbolId", "CallSite": "callSiteId",
    "Param": "pointId", "Ret": "pointId", "Cond": "pointId", "Name": "nameKey",
    "ContractPoint": "cpId", "Link": "linkId", "Dependency": "depId",
}
#: ``SkippedFile`` has no id property (S17 section 1.3): it is written by :func:`node_statement` and matched
#: by ``(unitId, path)`` as an edge target; its indexes are the DDL's ``path`` and ``repoId``
NODE_ORDER = ("Repo", "Unit", "File", "Symbol", "Param", "Ret", "Cond", "CallSite", "Name", "ContractPoint", "Link",
              "Dependency", "SkippedFile")

#: D8: a reader coalesces an absent ``status`` / ``reason`` to these, so the loader stores only a non-default
DEFAULT_STATUS = {"CALLS": {"status": "confirmed", "reason": "resolved-call"},
                  "DATA": {"status": "confirmed", "reason": "intra-flow"}}

#: keep Cypher fixed text for the few relationship groups a bundle can produce, ordered
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
)

#: ``parentId`` is deliberately absent (D7): the loader turns it into ``depth``
_SYMBOL_PROPS = ("kind", "name", "nameNorm", "qname", "path", "lineStart", "lineEnd", "fingerprint",
                 "lastChangedAt", "lastChangedCommit", "complexity", "lang", "signature")


def _clean(props: dict[str, Any]) -> dict[str, Any]:
    """Drop absent values and strip strings (an unstripped indexed value is silently missed; §3.4)."""
    out: dict[str, Any] = {}
    for k, v in props.items():
        if v is None:
            continue
        out[k] = v.strip() if isinstance(v, str) else v
    return out


def _symbol_depths(parents: dict[str, str | None]) -> dict[str, int]:
    """Nesting depth per symbol id from ``parentId``: a root (no parent, or a parent outside the snapshot)
    is 0. Order-independent; a parent cycle (invalid input) terminates instead of looping."""
    depth: dict[str, int] = {}
    for sid in parents:
        chain, cur = [], sid
        while cur in parents and cur not in depth and cur not in chain:
            chain.append(cur)
            cur = parents[cur]
        base = depth[cur] if cur in depth else -1  # an unknown parent (or a cycle) puts the chain's head at 0
        for node in reversed(chain):
            base += 1
            depth[node] = base
    return depth


def _non_default(etype: str, status: str, reason: str) -> dict[str, str]:
    """D8: the ``status`` / ``reason`` of a ``CALLS`` / ``DATA`` edge that differ from the default, each on its own."""
    default = DEFAULT_STATUS[etype]
    return {k: v for k, v in (("status", status), ("reason", reason)) if v != default[k]}


def _flatten_regions(regions: list[list[int]]) -> list[int]:
    """``ownedRegions`` ``[[a, b], ...]`` -> ``[a, b, ...]`` (a nested array is not a property value)."""
    return [n for pair in regions for n in pair]


# ---------------------------------------------------------------------------------------------
# Rows.
# ---------------------------------------------------------------------------------------------


@dataclass
class GraphRows:
    nodes: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    #: (source label, type, target label) -> rows ``{"s": id, "t": id, "p": {...}}``
    edges: dict[tuple[str, str, str], list[dict]] = field(default_factory=lambda: defaultdict(list))
    #: why a fact was not written, by reason (loader-side counts, S13)
    dropped: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    units: list[dict] = field(default_factory=list)
    files: dict[str, int] = field(default_factory=dict)  # unitId -> file count
    snapshot: dict = field(default_factory=dict)

    def node_count(self) -> int:
        return sum(len(v) for v in self.nodes.values())

    def edge_count(self) -> int:
        return sum(len(v) for v in self.edges.values())


def _add_node(rows: GraphRows, seen: dict[str, set], label: str, props: dict) -> None:
    ident = props[ID_PROP[label]]
    if ident in seen[label]:
        return
    seen[label].add(ident)
    rows.nodes[label].append(_clean(props))


def _add_edge(rows: GraphRows, seen: set, src: tuple[str, str], etype: str, dst: tuple[str, str],
              props: dict | None = None, key: tuple = ()) -> bool:
    """Add one edge; an identical edge (same key) is added once. Returns whether it was added."""
    ident = (src, etype, dst, key)
    if ident in seen:
        rows.dropped[f"duplicate-{etype}"] += 1
        return False
    seen.add(ident)
    row: dict[str, Any] = {"s": src[1], "t": dst[1]}
    if props:
        row["p"] = props
    rows.edges[(src[0], etype, dst[0])].append(row)
    return True


def build_rows(bundles: list[list[dict]], facts: dict, *, system_id: str = SYSTEM) -> GraphRows:
    """Graph rows for the given bundles' records and the S12 link ``facts`` (``assemble.assemble``)."""
    rows = GraphRows()
    seen_nodes: dict[str, set] = defaultdict(set)
    seen_edges: set = set()

    point_label: dict[str, str] = {}  # point id -> Param | Ret | Cond | CallSite
    symbol_ids: set[str] = set()
    sym_lang: dict[str, str] = {}
    sym_path: dict[str, str] = {}
    sym_unit: dict[str, str] = {}
    by_name: dict[tuple[str, str], list[str]] = defaultdict(list)
    headers: list[dict] = []
    parents: dict[str, str | None] = {}

    # pass 1: index every point and symbol, so edges can find both ends whichever bundle holds them
    for records in bundles:
        header = next(r for r in records if r["type"] == "header")
        headers.append(header)
        for r in records:
            t = r["type"]
            if t == "symbol":
                symbol_ids.add(r["id"])
                sym_lang[r["id"]] = r["lang"]
                sym_path[r["id"]] = r["path"]
                sym_unit[r["id"]] = header["unitId"]
                by_name[(r["lang"], r["name"])].append(r["id"])
                parents[r["id"]] = r.get("parentId")
            elif t in ("param", "ret", "cond", "callsite"):
                point_label[r["id"]] = {"param": "Param", "ret": "Ret", "cond": "Cond", "callsite": "CallSite"}[t]

    depths = _symbol_depths(parents)

    # pass 2: nodes and the edges that follow from one record
    repo_head: dict[str, str] = {}
    for records in bundles:
        header = next(r for r in records if r["type"] == "header")
        unit_id, repo_id = header["unitId"], header["unit"]["repoId"]
        repo_head.setdefault(repo_id, header["builtCommit"])
        _add_node(rows, seen_nodes, "Repo", {"repoId": repo_id, "headCommit": repo_head[repo_id]})
        _add_node(rows, seen_nodes, "Unit", {
            "unitId": unit_id, "repoId": repo_id, "rootPath": header["unit"]["rootPath"],
            "language": header["unit"]["lang"], "builtCommit": header["builtCommit"],
            "builtTree": header["builtTree"], "builtAt": header["builtAt"], "buildStatus": "fresh",
        })
        _add_edge(rows, seen_edges, ("Unit", unit_id), "IN_REPO", ("Repo", repo_id))
        rows.units.append({"unitId": unit_id, "rootPath": header["unit"]["rootPath"], "language": header["unit"]["lang"]})
        rows.files[unit_id] = 0
        for skipped in header.get("skippedFiles", ()):  # S17 section 1.3: a coverage fact, one source (the header)
            rows.nodes["SkippedFile"].append({"repoId": repo_id, "unitId": unit_id, "path": skipped["path"],
                                              "reason": skipped["reason"]})
            _add_edge(rows, seen_edges, ("Unit", unit_id), "SKIPS", ("SkippedFile", skipped["path"]))
        for r in records:
            t = r["type"]
            if t == "file":
                rows.files[unit_id] += 1
                _add_node(rows, seen_nodes, "File", {
                    "fileId": r["id"], "repoId": r["repoId"], "unitId": unit_id, "path": r["path"],
                    "kind": r["kind"], "role": r["role"], "lang": r["lang"], "contentHash": r["contentHash"]})
                _add_edge(rows, seen_edges, ("Unit", unit_id), "HAS_FILE", ("File", r["id"]))
            elif t == "symbol":
                props = {"symbolId": r["id"], "repoId": r["repoId"], "unitId": unit_id}
                props.update({k: r.get(k) for k in _SYMBOL_PROPS})
                props["depth"] = depths[r["id"]]
                if r.get("ownedRegions"):
                    props["ownedRegions"] = _flatten_regions(r["ownedRegions"])
                _add_node(rows, seen_nodes, "Symbol", props)
                _add_edge(rows, seen_edges, ("File", r["fileId"]), "DEFINES", ("Symbol", r["id"]))
            elif t == "param":
                _add_node(rows, seen_nodes, "Param", {"pointId": r["id"], "index": r["index"], "name": r["name"], "line": r["line"]})
                _add_edge(rows, seen_edges, ("Symbol", r["symbolId"]), "HAS_PARAM", ("Param", r["id"]))
            elif t == "ret":
                _add_node(rows, seen_nodes, "Ret", {"pointId": r["id"], "line": r["line"]})
                _add_edge(rows, seen_edges, ("Symbol", r["symbolId"]), "HAS_RETURN", ("Ret", r["id"]))
            elif t == "cond":
                _add_node(rows, seen_nodes, "Cond", {"pointId": r["id"], "line": r["line"], "code": r["code"]})
                _add_edge(rows, seen_edges, ("Symbol", r["symbolId"]), "HAS_COND", ("Cond", r["id"]))
            elif t == "callsite":
                sid = r["symbolId"]
                _add_node(rows, seen_nodes, "CallSite", {
                    "callSiteId": r["id"], "repoId": repo_id, "unitId": unit_id, "path": sym_path[sid],
                    "line": r["line"], "calleeName": r["calleeName"]})
                _add_edge(rows, seen_edges, ("Symbol", sid), "HAS_CALL", ("CallSite", r["id"]))
                for target in r["targets"]:
                    if target["symbolId"] not in symbol_ids:
                        rows.dropped["calls-target-not-in-snapshot"] += 1
                        continue
                    _add_edge(rows, seen_edges, ("CallSite", r["id"]), "CALLS", ("Symbol", target["symbolId"]),
                              _non_default("CALLS", r["status"], r["reason"]), key=(r["status"], r["reason"]))
                if r["calleeName"] != "<dynamic>":
                    lang = sym_lang[sid]
                    nk = name_key(lang, r["calleeName"])
                    _add_node(rows, seen_nodes, "Name", {"nameKey": nk, "lang": lang, "name": r["calleeName"]})
                    _add_edge(rows, seen_edges, ("CallSite", r["id"]), "CALLS_NAME", ("Name", nk))
            elif t == "dependency":
                _add_node(rows, seen_nodes, "Dependency", {"depId": r["depId"], "ecosystem": r["ecosystem"], "package": r["package"]})
                _add_edge(rows, seen_edges, ("Unit", unit_id), "DEPENDS_ON", ("Dependency", r["depId"]),
                          {"version": r["version"], "manifestPath": r["manifestPath"], "line": r["line"], "scope": r["scope"]},
                          key=(r["manifestPath"], r["line"], r["scope"]))
            elif t == "data":
                _add_data(rows, seen_edges, point_label, r)
            elif t == "controlledBy":
                _add_controlled_by(rows, seen_edges, point_label, r)

    # Name -> every symbol of that language and name (graph note §6.1 item 6)
    for n in rows.nodes["Name"]:
        for sym in sorted(by_name.get((n["lang"], n["name"]), ())):
            _add_edge(rows, seen_edges, ("Name", n["nameKey"]), "RESOLVES_TO", ("Symbol", sym))

    _add_link_facts(rows, seen_nodes, seen_edges, facts, symbol_ids, point_label)

    manifest = sorted([h["unitId"], h["builtCommit"], h["builtTree"]] for h in headers)
    manifest_hash = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    rows.snapshot = _clean({
        "snapshotId": tuple_id(["snapshot", system_id, manifest_hash]), "systemId": system_id,
        "builtAt": max(h["builtAt"] for h in headers), "vocabVersion": "p0-slice",
        "bundleSchemaVersion": headers[0]["bundleSchemaVersion"], "manifestHash": manifest_hash,
    })
    rows.dropped = dict(rows.dropped)
    return rows


def _add_data(rows: GraphRows, seen: set, point_label: dict, r: dict) -> None:
    # No self-edge rule here: a call site whose argument derives from its own result (a loop-carried
    # value, ``use == def`` with different paths) is a real fact. Only the CDG self-edge is dropped (G-4).
    use, dfn = r["use"], r["def"]
    ul, dl = point_label.get(use), point_label.get(dfn)
    if ul not in ("CallSite", "Ret", "Cond") or dl not in ("Param", "CallSite"):
        rows.dropped["data-endpoint-missing"] += 1
        return
    props = {"fromPath": r["fromPath"], "toPath": r["toPath"], **_non_default("DATA", r["status"], r["reason"])}
    _add_edge(rows, seen, (ul, use), "DATA", (dl, dfn), props, key=(r["fromPath"], r["toPath"]))


def _add_controlled_by(rows: GraphRows, seen: set, point_label: dict, r: dict) -> None:
    """G-4: a self-edge is dropped and a (point, cond) pair is written once."""
    point, cond = r["point"], r["cond"]
    if point == cond:
        rows.dropped["controlled-by-self-edge"] += 1
        return
    pl = point_label.get(point)
    if pl is None or point_label.get(cond) != "Cond":
        rows.dropped["controlled-by-endpoint-missing"] += 1
        return
    _add_edge(rows, seen, (pl, point), "CONTROLLED_BY", ("Cond", cond))


def _add_link_facts(rows: GraphRows, seen_nodes: dict, seen_edges: set, facts: dict,
                    symbol_ids: set[str], point_label: dict) -> None:
    cp_cache: dict[str, str] = {}

    def cp(key: str) -> str:
        if key not in cp_cache:
            cid = contract_point_id("http", key)
            _add_node(rows, seen_nodes, "ContractPoint", {"cpId": cid, "kind": "http", "key": key, "keyNorm": key})
            cp_cache[key] = cid
        return cp_cache[key]

    for p in facts["providers"]:
        if p["handlerId"] not in symbol_ids:
            rows.dropped["provider-handler-missing"] += 1
            continue
        _add_edge(rows, seen_edges, ("Symbol", p["handlerId"]), "PROVIDES", ("ContractPoint", cp(p["key"])),
                  {"status": p["status"], "reason": p["reason"]})
    for c in facts["consumers"]:
        for key in c["keys"]:
            status = "possible" if c["partialAlternatives"] else "confirmed"
            _add_edge(rows, seen_edges, ("CallSite", c["siteId"]), "CONSUMES", ("ContractPoint", cp(key)),
                      {"status": status, "reason": "lifted-role"}, key=(key,))
    for link in facts["links"]:
        lid = link["id"]
        _add_node(rows, seen_nodes, "Link", {
            "linkId": lid, "kind": link["kind"], "status": link["status"], "reason": link["reason"],
            "matcher": link["matcher"]})  # D6: no candidateCount; N is derived per VIA edge (render.py)
        _add_edge(rows, seen_edges, ("Link", lid), "TO", ("Symbol", link["provider"]["anchorId"]))
        _add_edge(rows, seen_edges, ("Link", lid), "ON", ("ContractPoint", cp(link["keyNorm"])))
        for site in link["sites"]:
            _add_edge(rows, seen_edges, ("CallSite", site["siteId"]), "VIA", ("Link", lid),
                      {"status": site["status"], "reason": site["reason"], "consumerKey": site["consumerKey"]},
                      key=(site["consumerKey"],))
            for hop in site["liftedVia"]:
                label = "CallSite" if hop["kind"] == "callsite" else "Symbol"
                props = {"ord": hop["ord"], "siteId": site["siteId"], "line": hop["line"], "urlPath": hop["urlPath"]}
                for k in ("verbPath", "viaSiteId", "viaLine"):
                    if hop.get(k) is not None:
                        props[k] = hop[k]
                _add_edge(rows, seen_edges, ("Link", lid), "LIFTED_VIA", (label, hop["id"]), props,
                          key=(site["siteId"], hop["ord"]))


# ---------------------------------------------------------------------------------------------
# Cypher text (fixed templates; data only ever travels as parameters).
# ---------------------------------------------------------------------------------------------


def batches(rows: list, size: int = BATCH_MAX) -> Iterator[list]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def index_statements() -> list[str]:
    """Part 1 of ``ccg/store/ddl_v1.cypher`` (S17 section 1.6), in file order: the DDL file is the one home."""
    return [s[3] for s in parse_ddl() if s[0] == "index"]


def constraint_statements() -> list[tuple[str, str]]:
    """Part 2 of the DDL file, in file order: ``(label, property)`` of each ``UNIQUE`` constraint."""
    return [(_ident(s[1]), _ident(s[2])) for s in parse_ddl() if s[0] == "constraint"]


def node_statement(label: str) -> str:
    return f"UNWIND $rows AS r CREATE (n:{_ident(label)}) SET n = r"


def edge_statement(src: str, etype: str, dst: str, with_props: bool) -> str:
    s = ID_PROP[src]
    # a SkippedFile has no id: it is matched by (unit, path), the unit id being the edge's source id
    target = "unitId: r.s, path: r.t" if dst == "SkippedFile" else f"{_ident(ID_PROP[dst])}: r.t"
    text = (f"UNWIND $rows AS r MATCH (a:{_ident(src)} {{{_ident(s)}: r.s}}) "
            f"MATCH (b:{_ident(dst)} {{{target}}}) CREATE (a)-[e:{_ident(etype)}]->(b)")
    return text + " SET e = r.p" if with_props else text


SNAPSHOT_STATEMENT = "CREATE (s:Snapshot) SET s = $props"


# ---------------------------------------------------------------------------------------------
# Stores: the thin seam over the database (a fake in tests, ``LiveStore`` for real).
# ---------------------------------------------------------------------------------------------


class Store(Protocol):
    def timeout_max(self) -> int: ...
    def exists(self, key: str) -> bool: ...
    def query(self, key: str, cypher: str, params: dict | None = None, timeout: int | None = None) -> Any: ...
    def create_unique_constraint(self, key: str, label: str, prop: str) -> None: ...
    def constraints(self, key: str) -> list[dict]: ...
    def rename(self, src: str, dst: str) -> None: ...
    def delete(self, key: str) -> None: ...
    def memory(self, key: str) -> dict: ...
    def used_memory(self) -> int: ...


def zero_retry():
    """An explicit zero-retry policy. ``retry=None`` is the client default, which redis-py replaces with its
    own retrying policy: a retried write could run twice (S17 note section 4)."""
    from redis.backoff import NoBackoff
    from redis.retry import Retry
    return Retry(NoBackoff(), 0)


def password() -> str:
    path = Path(os.environ.get("CCG_CONFIG_DIR", Path.home() / ".config" / "ccg")) / "falkordb-admin.pass"
    return path.read_text().strip()


class LiveStore:
    """One persistent connection, retries off, socket timeout of minutes; every key goes through :func:`check_key`."""

    def __init__(self, host: str, port: int, pw: str | None = None):
        if port == FORBIDDEN_PORT:
            raise LoadError(f"port {FORBIDDEN_PORT} is falkordb-dev; this loader never touches it")
        from falkordb import FalkorDB
        self.db = FalkorDB(host=host, port=port, password=pw if pw is not None else password(),
                           socket_timeout=SOCKET_TIMEOUT_S, socket_connect_timeout=10, retry=zero_retry())
        self.conn = self.db.connection
        self.conn.ping()

    def timeout_max(self) -> int:
        return int(self.conn.execute_command("GRAPH.CONFIG", "GET", "TIMEOUT_MAX")[1])

    def exists(self, key: str) -> bool:
        return bool(self.conn.exists(check_key(key)))

    def query(self, key, cypher, params=None, timeout=None):
        graph = self.db.select_graph(check_key(key))
        return graph.query(cypher, params, timeout) if timeout else graph.query(cypher, params)

    def create_unique_constraint(self, key, label, prop):
        # the raw command, never falkordb-py's create_node_unique_constraint: that helper silently creates a
        # missing range index first, which would mask a gap in ddl_v1.cypher (falkordb-quirks.md, 2026-11-01)
        self.conn.execute_command("GRAPH.CONSTRAINT", "CREATE", check_key(key), "UNIQUE", "NODE",
                                  _ident(label), "PROPERTIES", 1, _ident(prop))

    def constraints(self, key):
        res = self.db.select_graph(check_key(key)).query("CALL db.constraints()")
        return [{"type": r[0], "label": r[1], "properties": r[2], "entity": r[3], "status": r[4]} for r in res.result_set]

    def rename(self, src, dst):
        self.conn.rename(check_key(src), check_key(dst))

    def delete(self, key):
        self.db.select_graph(check_key(key)).delete()

    def used_memory(self):
        return int(self.conn.info("memory")["used_memory"])

    def memory(self, key):
        raw = self.conn.execute_command("GRAPH.MEMORY", "USAGE", check_key(key))
        return {raw[i]: raw[i + 1] for i in range(0, len(raw), 2)}


# ---------------------------------------------------------------------------------------------
# The load.
# ---------------------------------------------------------------------------------------------


@dataclass
class LoadReport:
    build_key: str = ""
    final_key: str = ""
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
    memory: dict = field(default_factory=dict)


def _write(store: Store, key: str, cypher: str, params: dict, timeout: int | None, report: LoadReport) -> Any:
    report.batches += 1
    report.max_batch = max(report.max_batch, len(params.get("rows", ())))
    return store.query(key, cypher, params, timeout)


def load_build(store: Store, rows: GraphRows, build_key: str, report: LoadReport, *, snapshot: bool = True,
               poll_interval: float = 0.25, poll_timeout: float = CONSTRAINT_POLL_TIMEOUT_S,
               batch_size: int = BATCH_MAX, sleep=time.sleep, clock=time.monotonic) -> None:
    """Steps 1-4 of the sequence on ``build_key``; the caller renames."""
    check_key(build_key)
    report.timeout_max = store.timeout_max()
    timeout = report.timeout_max or None  # D16: explicit when TIMEOUT_MAX is nonzero
    report.query_timeout_used = timeout is not None

    t0 = clock()
    for stmt in index_statements():
        store.query(build_key, stmt)
    report.seconds["indexes"] = clock() - t0

    t0 = clock()
    for label in NODE_ORDER:
        label_rows = rows.nodes.get(label, [])
        for chunk in batches(label_rows, batch_size):
            _write(store, build_key, node_statement(label), {"rows": chunk}, timeout, report)
        report.nodes[label] = len(label_rows)
    report.seconds["nodes"] = clock() - t0

    t0 = clock()
    unknown = set(rows.edges) - set(EDGE_ORDER)
    if unknown:
        raise LoadError(f"edge groups with no load statement: {sorted(unknown)}")
    for group in EDGE_ORDER:
        group_rows = rows.edges.get(group, [])
        if not group_rows:
            continue
        src, etype, dst = group
        with_props = any("p" in r for r in group_rows)
        for chunk in batches(group_rows, batch_size):
            if with_props:  # a row without props is a driver error in a mixed batch; an empty map sets nothing
                chunk = [r if "p" in r else {**r, "p": {}} for r in chunk]
            res = _write(store, build_key, edge_statement(src, etype, dst, with_props), {"rows": chunk}, timeout, report)
            created = getattr(res, "relationships_created", None)
            if created is not None and created != len(chunk):
                raise LoadError(f"{src}-[{etype}]->{dst}: {len(chunk)} rows but {created} edges created "
                                "(an endpoint did not match)")
        report.edges[f"{src}-{etype}->{dst}"] = len(group_rows)
    report.seconds["edges"] = clock() - t0

    t0 = clock()
    constraints = constraint_statements()
    for label, prop in constraints:
        store.create_unique_constraint(build_key, label, prop)
    report.constraints = poll_constraints(store, build_key, expected=len(constraints), interval=poll_interval,
                                          timeout=poll_timeout, sleep=sleep, clock=clock)
    report.seconds["constraints"] = clock() - t0

    if snapshot:
        store.query(build_key, SNAPSHOT_STATEMENT, {"props": rows.snapshot})  # last: its presence means "complete"

    report.node_total = sum(report.nodes.values()) + (1 if snapshot else 0)
    report.edge_total = sum(report.edges.values())


def poll_constraints(store: Store, key: str, *, expected: int, interval: float = 0.25,
                     timeout: float = CONSTRAINT_POLL_TIMEOUT_S, sleep=time.sleep, clock=time.monotonic) -> list[dict]:
    """Poll ``db.constraints()`` until all ``expected`` are ``OPERATIONAL``; any ``FAILED`` is a failed build."""
    deadline = clock() + timeout
    while True:
        cons = store.constraints(key)
        failed = [c for c in cons if c["status"] == "FAILED"]
        if failed:
            raise LoadError(f"constraint FAILED on {key}: {[(c['label'], c['properties']) for c in failed]}")
        if len(cons) >= expected and all(c["status"] == "OPERATIONAL" for c in cons):
            return cons
        if clock() >= deadline:
            raise LoadError(f"constraints not OPERATIONAL within {timeout}s on {key}: {[c['status'] for c in cons]}")
        sleep(interval)


def load(store: Store, bundles: list[list[dict]], facts: dict, *, final_key: str = FINAL_KEY,
         drop_stale: bool = False, rows: GraphRows | None = None, **kw) -> LoadReport:
    """The full sequence: build key, indexes, nodes, edges, constraints, ``:Snapshot``, ``RENAME``."""
    check_key(final_key)
    rows = rows if rows is not None else build_rows(bundles, facts)
    report = LoadReport(final_key=final_key, snapshot_id=rows.snapshot["snapshotId"], dropped=dict(rows.dropped))
    report.build_key = check_key(BUILD_PREFIX + rows.snapshot["snapshotId"])
    if store.exists(report.build_key):
        if not drop_stale:
            raise LoadError(f"{report.build_key} already exists (an interrupted load?); rerun with drop_stale")
        store.delete(report.build_key)
    t0 = time.monotonic()
    load_build(store, rows, report.build_key, report, **kw)
    report.memory = store.memory(report.build_key)
    t1 = time.monotonic()
    store.rename(report.build_key, final_key)
    report.seconds["rename"] = time.monotonic() - t1
    report.seconds["total"] = time.monotonic() - t0
    return report


def measure_unit_density(store: Store, bundle: list[dict], *, tag: str, **kw) -> dict:
    """SL-5 half: load ONE unit alone (no links, no snapshot) into a scratch build key, measure, delete it."""
    rows = build_rows([bundle], {"providers": [], "consumers": [], "links": []})
    key = check_key(f"{BUILD_PREFIX}dens_{tag}")
    if store.exists(key):
        store.delete(key)
    report = LoadReport(build_key=key, dropped=dict(rows.dropped))
    try:
        before = store.used_memory()
        t0 = time.monotonic()
        load_build(store, rows, key, report, snapshot=False, **kw)
        report.seconds["total"] = time.monotonic() - t0
        report.memory = store.memory(key)
        used_delta = store.used_memory() - before
    finally:
        if store.exists(key):
            store.delete(key)
    unit_id = rows.units[0]["unitId"]
    files = rows.files[unit_id]
    total_mb = report.memory["total_graph_sz_mb"]
    return {
        "unitId": unit_id, "rootPath": rows.units[0]["rootPath"], "files": files,
        "nodes": report.node_total, "edges": report.edge_total, "memory": report.memory,
        "totalGraphMb": total_mb, "mbPerFile": total_mb / files, "instanceUsedMemoryDeltaBytes": used_delta,
        "instanceUsedMemoryDeltaMbPerFile": used_delta / 2**20 / files, "nodesPerFile": report.node_total / files,
        "edgesPerFile": report.edge_total / files, "seconds": report.seconds, "dropped": report.dropped,
    }


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from assemble import assemble

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", nargs="+")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--drop-stale", action="store_true")
    args = ap.parse_args(argv)
    bundles = [read_bundle(b) for b in args.bundles]
    facts = assemble(bundles)
    report = load(LiveStore(args.host, args.port), bundles, facts, drop_stale=args.drop_stale)
    print(json.dumps({k: v for k, v in report.__dict__.items() if k != "constraints"}, sort_keys=True, indent=1))
    print(f"constraints: {[(c['label'], c['status']) for c in report.constraints]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
