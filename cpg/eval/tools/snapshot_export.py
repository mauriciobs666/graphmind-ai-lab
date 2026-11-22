#!/usr/bin/env python3
"""snapshot_export.py - custodian-side store export for the noise and oracle tooling (P1-36; ``docs/plans/connected-code-graph-p1-noise-export-graph.md`` §1-§3, §5).

    snapshot_export.py --graph ccg_<s> --snapshot-id S --custodian ROOT --port P [--host H --user U --password-file F]

Runs the Cypher texts of §2, §3 and §5 as ``GRAPH.RO_QUERY`` (the ``ccg_raw_<s>`` role; no write-capable call exists here) against the
live key and writes, under ``--custodian`` (refused inside the repo): ``snapshot-adjacency.jsonl`` (the 14-record export, ``baseline.read_snapshot``'s
input), ``links.jsonl`` and ``links-unprovided.jsonl`` (``oracle.py``'s Link set), ``nc5-unbound.jsonl`` (the Q-NC5 list form).
``Snapshot.snapshotId`` is read first and last and must equal ``--snapshot-id`` both times: a refresh during the export is an error and
nothing is written; the files are staged under temporary names and published only after every check passes. The written export is re-read with ``read_snapshot`` (closure) and the independent Q-NC5 count is compared with
``noise.unbound_multi_unit(nc5_view(...))``. stdout is a counts-only JSON report. Exit: 0 ok, 2 usage/refused, 4 bad input or mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import baseline  # noqa: E402
from noise import unbound_multi_unit  # noqa: E402

PAGE_LIMIT = 5000                 # below RESULTSET_SIZE (10,000)
QUERY_TIMEOUT_MS = 10000          # <= the store's TIMEOUT_MAX
EXPORT_VERSION = baseline.EXPORT_VERSION
OPTIONAL = {"ownedRegions", "consumerKey", "keyPath", "launchKind"}      # absent from the record when null
_PAGE = " SKIP $skip LIMIT $limit"

SNAPSHOT_QUERY = "MATCH (s:Snapshot) RETURN s.snapshotId"

# t -> (cypher, fields in RETURN order). Group order is the export order.
QUERIES = {
    "meta": ("MATCH (s:Snapshot) RETURN s.snapshotId, s.systemId, s.graphModelVersion",
             ("snapshotId", "systemId", "graphModelVersion")),
    "unit": ("MATCH (u:Unit) RETURN u.unitId, u.repoId, u.language, u.buildStatus, u.flowAvailable ORDER BY u.unitId",
             ("unitId", "repoId", "language", "buildStatus", "flowAvailable")),
    "symbol": ("MATCH (s:Symbol) RETURN s.symbolId, s.unitId, s.repoId, s.qname, s.path, s.lineStart, s.lineEnd, s.ownedRegions "
               "ORDER BY s.symbolId" + _PAGE,
               ("symbolId", "unitId", "repoId", "qname", "path", "lineStart", "lineEnd", "ownedRegions")),
    "site": ("MATCH (s:Symbol)-[:HAS_CALL]->(c:CallSite) RETURN c.callSiteId, s.symbolId, c.unitId, c.path, c.line, c.calleeName "
             "ORDER BY c.callSiteId" + _PAGE,
             ("siteId", "owner", "unitId", "path", "line", "calleeName")),
    "call": ("MATCH (c:CallSite)-[e:CALLS]->(t:Symbol) RETURN c.callSiteId, t.symbolId, coalesce(e.status,'confirmed'), "
             "coalesce(e.reason,'resolved-call') ORDER BY c.callSiteId, t.symbolId" + _PAGE,
             ("site", "callee", "status", "reason")),
    "name": ("MATCH (c:CallSite)-[:CALLS_NAME]->(:Name)-[:RESOLVES_TO]->(t:Symbol) RETURN c.callSiteId, t.symbolId "
             "ORDER BY c.callSiteId, t.symbolId" + _PAGE,
             ("site", "candidate")),
    "cp": ("MATCH (cp:ContractPoint) RETURN cp.cpId, cp.kind, cp.keyNorm ORDER BY cp.cpId" + _PAGE, ("cpId", "kind", "keyNorm")),
    "provides": ("MATCH (s:Symbol)-[e:PROVIDES]->(cp:ContractPoint) RETURN s.symbolId, cp.cpId, e.status, e.reason "
                 "ORDER BY s.symbolId, cp.cpId" + _PAGE,
                 ("symbolId", "cpId", "status", "reason")),
    "consumes": ("MATCH (c:CallSite)-[e:CONSUMES]->(cp:ContractPoint) RETURN c.callSiteId, cp.cpId, e.status, e.reason "
                 "ORDER BY c.callSiteId, cp.cpId" + _PAGE,
                 ("site", "cpId", "status", "reason")),
    "link": ("MATCH (l:Link) OPTIONAL MATCH (l)-[:ON]->(cp:ContractPoint) OPTIONAL MATCH (l)-[:TO]->(p:Symbol) "
             "RETURN l.linkId, l.kind, l.status, l.reason, cp.cpId, p.symbolId ORDER BY l.linkId" + _PAGE,
             ("linkId", "kind", "status", "reason", "cpId", "provider")),
    "via": ("MATCH (c:CallSite)-[v:VIA]->(l:Link) RETURN c.callSiteId, l.linkId, v.status, v.reason, v.consumerKey "
            "ORDER BY c.callSiteId, l.linkId" + _PAGE,
            ("site", "linkId", "status", "reason", "consumerKey")),
    "entry": ("MATCH (e:ConfigEntry) RETURN e.entryId, e.unitId, e.path, e.keyPath, e.launchKind ORDER BY e.entryId" + _PAGE,
              ("entryId", "unitId", "path", "keyPath", "launchKind")),
    "bound": ("MATCH (l:Link)-[b:BOUND_BY]->(e:ConfigEntry) RETURN l.linkId, e.entryId, b.siteId, b.ord "
              "ORDER BY l.linkId, e.entryId, b.siteId, b.ord" + _PAGE,
              ("linkId", "entryId", "siteId", "ord")),
}

LINKS_QUERY = (
    "MATCH (c:CallSite)-[v:VIA]->(l:Link) WHERE l.kind = 'http' "
    "MATCH (s:Symbol)-[:HAS_CALL]->(c) "
    "OPTIONAL MATCH (l)-[:ON]->(cp:ContractPoint) "
    "OPTIONAL MATCH (l)-[:TO]->(p:Symbol) "
    "RETURN s.qname AS element, s.path AS path, cp.keyNorm AS keyNorm, v.consumerKey AS consumerKey, "
    "l.status AS status, l.linkId AS linkId, p.symbolId IS NOT NULL AS hasProvider "
    "ORDER BY element, linkId, c.callSiteId" + _PAGE)

_NC5_HEAD = ("MATCH (s:Symbol)-[:PROVIDES]->(cp:ContractPoint) "
             "WITH cp, count(DISTINCT s.unitId) AS nu WHERE nu > 1 "
             "MATCH (l:Link)-[:ON]->(cp) "
             "OPTIONAL MATCH (l)-[b:BOUND_BY]->(:ConfigEntry) ")
NC5_COUNT_QUERY = (_NC5_HEAD + "WITH l, count(b) AS nb WHERE nb = 0 "
                   "RETURN count(l) AS unboundMultiUnit, sum(CASE WHEN l.status = 'confirmed' THEN 1 ELSE 0 END) AS unboundMultiUnitConfirmed")
NC5_LIST_QUERY = (_NC5_HEAD + "WITH l, cp, nu, count(b) AS nb WHERE nb = 0 "
                  "RETURN l.linkId, l.status, cp.kind, cp.keyNorm, nu ORDER BY l.linkId" + _PAGE)

_STATUS_RANK = {"confirmed": 0, "possible": 1}


class SnapshotMismatch(ValueError):
    """The store's ``Snapshot.snapshotId`` is not the gate's, or changed during the export."""


class RoRunner:
    """The only transport: ``GRAPH.RO_QUERY`` with a timeout. ``graph`` is a falkordb ``Graph``; ``query`` is never called."""

    def __init__(self, graph, timeout_ms: int = QUERY_TIMEOUT_MS):
        self._graph, self._timeout = graph, timeout_ms

    def __call__(self, cypher: str, params: dict | None = None) -> list:
        return self._graph.ro_query(cypher, params, self._timeout).result_set


def paged(run, cypher: str, limit: int = PAGE_LIMIT) -> list:
    """Every row of a ``SKIP $skip LIMIT $limit`` query; stops on a short page (the ORDER BY is a total order)."""
    rows, skip = [], 0
    while True:
        page = run(cypher, {"skip": skip, "limit": limit})
        rows += page
        if len(page) < limit:
            return rows
        skip += limit


def _snapshot_id(run) -> str:
    rows = run(SNAPSHOT_QUERY, None)
    if len(rows) != 1:
        raise SnapshotMismatch(f"snapshot: {len(rows)} Snapshot nodes, expected 1")
    return rows[0][0]


def _record(t: str, fields, row) -> dict:
    rec = dict(zip(fields, row))
    if t == "meta":
        rec["exportVersion"] = EXPORT_VERSION
    return {k: v for k, v in rec.items() if not (v is None and k in OPTIONAL)}


def export_snapshot(run) -> list:
    """The export's lines (§3): canonical JSON, ``meta`` first, groups in table order, each in its query's total order."""
    lines = []
    for t, (cypher, fields) in QUERIES.items():
        rows = paged(run, cypher) if cypher.rstrip().endswith(_PAGE.strip()) else run(cypher, None)
        lines += [json.dumps({"t": t, **_record(t, fields, r)}, sort_keys=True, ensure_ascii=False) for r in rows]
    return lines


def dump_links(run) -> tuple:
    """``(provided, unprovided)`` rows for ``oracle.py`` (§2): distinct on ``(element, keyNorm)`` keeping the best status, then the lowest ``linkId``."""
    best: dict = {}
    for element, path, key, ckey, status, link_id, has_provider in paged(run, LINKS_QUERY):
        row = {"consumerKey": ckey, "element": element, "keyNorm": key, "linkId": link_id, "path": path, "status": status}
        group = "provided" if has_provider and key is not None else "unprovided"
        k = (group, element, key)
        rank = (_STATUS_RANK.get(status, 2), link_id)
        if k not in best or rank < best[k][0]:
            best[k] = (rank, row)
    out = {g: sorted((r for (gg, _, _), (_, r) in best.items() if gg == g), key=lambda r: (r["element"], r["keyNorm"] or ""))
           for g in ("provided", "unprovided")}
    return out["provided"], out["unprovided"]


def _key_path(key: str | None) -> str:
    return "" if key is None else key.split(" ", 1)[-1]


def link_stats(provided: list, unprovided: list) -> dict:
    shop = sum(1 for r in unprovided if _key_path(r["keyNorm"]).startswith(("/shop/api", "/{?}/shop/api")))
    return {"links": len(provided), "linksUnprovided": len(unprovided), "unprovidedShopApi": shop,
            "keyNormCatchAll": sum(1 for r in provided if "{**}" in (r["keyNorm"] or ""))}


def nc5_dump(run) -> dict:
    """The §5 independent check: counts (``sum`` is a float, converted) and the list form."""
    row = run(NC5_COUNT_QUERY, None)[0]
    rows = [{"keyNorm": k, "kind": kind, "linkId": lid, "providerUnits": int(nu), "status": st}
            for lid, st, kind, k, nu in paged(run, NC5_LIST_QUERY)]
    return {"unboundMultiUnit": int(row[0] or 0), "unboundMultiUnitConfirmed": int(row[1] or 0), "rows": rows}


def _write(path: Path, lines: list) -> None:
    path.write_text("".join(l + "\n" for l in lines), encoding="utf-8")


def run_export(run, snapshot_id: str, out_dir) -> dict:
    """Export everything under ``out_dir`` and return the counts-only report; raises ``SnapshotMismatch`` / ``ValueError``."""
    out = Path(out_dir)
    if _snapshot_id(run) != snapshot_id:
        raise SnapshotMismatch("snapshot: the store's snapshotId differs from --snapshot-id (first read)")
    lines = export_snapshot(run)
    provided, unprovided = dump_links(run)
    nc5 = nc5_dump(run)
    if _snapshot_id(run) != snapshot_id:
        raise SnapshotMismatch("snapshot: the snapshotId changed during the export (last read)")
    out.mkdir(parents=True, exist_ok=True)
    files = {"snapshot-adjacency.jsonl": lines, "links.jsonl": [json.dumps(r, sort_keys=True) for r in provided],
             "links-unprovided.jsonl": [json.dumps(r, sort_keys=True) for r in unprovided],
             "nc5-unbound.jsonl": [json.dumps(r, sort_keys=True) for r in nc5["rows"]]}
    tmp = {name: out / f".{name}.tmp" for name in files}
    try:
        for name, body in files.items():
            _write(tmp[name], body)
        snap = baseline.read_snapshot(tmp["snapshot-adjacency.jsonl"])
        view = unbound_multi_unit(baseline.nc5_view(snap))
        if (view["unboundMultiUnit"], view["unboundMultiUnitConfirmed"]) != (nc5["unboundMultiUnit"], nc5["unboundMultiUnitConfirmed"]):
            raise ValueError("nc5: the independent store count differs from the export projection")
        for name, path in tmp.items():                 # published only after every check passed
            path.replace(out / name)
    finally:
        for path in tmp.values():
            path.unlink(missing_ok=True)
    return {"records": {t: len(snap[t]) for t in QUERIES}, "dropped": snap["dropped"], **link_stats(provided, unprovided),
            "unboundMultiUnit": nc5["unboundMultiUnit"], "unboundMultiUnitConfirmed": nc5["unboundMultiUnitConfirmed"]}


def _connect(a):
    from falkordb import FalkorDB
    pw = Path(a.password_file).expanduser().read_text().strip() if a.password_file else None
    db = FalkorDB(host=a.host, port=a.port, username=a.user, password=pw, socket_timeout=120, socket_connect_timeout=10)
    return RoRunner(db.select_graph(a.graph))


def main(argv=None, runner=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--custodian", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True, help="no default: the shared store is never the implicit target")
    ap.add_argument("--user")
    ap.add_argument("--password-file")
    a = ap.parse_args(argv)
    out = Path(a.custodian).expanduser().resolve()
    if out == REPO or REPO in out.parents:
        print(f"REFUSED: --custodian {a.custodian} is inside the repo; the export belongs under the custodian root", file=sys.stderr)
        return 2
    try:
        from redis.exceptions import RedisError
    except ImportError:                                # the transport library is absent: nothing of its to catch
        RedisError = OSError                           # noqa: N806
    try:
        counts = run_export(runner or _connect(a), a.snapshot_id, out)
    except (ValueError, OSError, RedisError) as e:                 # SnapshotMismatch is a ValueError; messages carry counts, never ids
        print(f"export refused: {e}", file=sys.stderr)
        return 4
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
