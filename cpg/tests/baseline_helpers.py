"""The 15-symbol worked graph of ``docs/plans/connected-code-graph-p1-noise-export-graph.md`` §4.1, as export records.

Shared by ``test_baseline.py`` (pure) and ``test_snapshot_export.py`` (the live tier loads the same records into a
throwaway FalkorDB and compares the export with ``canonical_lines``). The tables below are the note's table, not a
dump of any implementation."""
import json

SNAPSHOT_ID = "snap-1"
SYSTEM_ID = "sys"

TS = {"H", "P3", "Q"}          # paths ending ``.ts`` (the ``in_scope`` rows of §4.1); the rest are ``.py``
SYMBOLS = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "M", "P2", "P3", "Q", "R", "S"]


def path_of(sym):
    return f"{sym.lower()}.ts" if sym in TS else f"{sym.lower()}.py"


def _symbol(sym):
    p = path_of(sym)
    rec = {"symbolId": sym, "unitId": "u2" if sym in TS else "u1", "repoId": "r1", "qname": f"{p}::{sym}",
           "path": p, "lineStart": 1, "lineEnd": 5}
    if sym == "E":
        rec["lineStart"], rec["lineEnd"], rec["ownedRegions"] = 1, 3, [7, 9]
    return rec


CALLS = [("B", "A"), ("C", "B"), ("D", "C"), ("F", "E"), ("R", "H"), ("S", "R")]      # (owner, callee)
NAMES = [("B", "A"), ("C", "B"), ("E", "A"), ("G", "B"), ("F", "K")]                   # (owner, candidate)
CONSUMERS = [("H", "CP1", "confirmed"), ("Q", "CP2", "possible")]                      # (owner, cp, status)


def site_id(owner, tag=""):
    return f"s{owner}{tag}"


def records():
    """``[(t, record)]`` in the export's group order, each group in its sort order."""
    recs = [("meta", {"snapshotId": SNAPSHOT_ID, "systemId": SYSTEM_ID, "graphModelVersion": "1.1", "exportVersion": 1})]
    recs += [("unit", {"unitId": u, "repoId": "r1", "language": lang, "buildStatus": "built", "flowAvailable": True})
             for u, lang in (("u1", "python"), ("u2", "typescript"))]
    recs += [("symbol", _symbol(s)) for s in sorted(SYMBOLS)]
    sites = {}      # siteId -> owner
    for owner, callee in CALLS:
        sites[site_id(owner, "c")] = owner
    for owner, cand in NAMES:
        if owner == "F":            # F's site is the one resolved to E; its name row rides on it
            continue
        sites[site_id(owner, "n")] = owner
    for owner, _, _ in CONSUMERS:
        sites[site_id(owner, "k")] = owner
    recs += [("site", {"siteId": sid, "owner": o, "unitId": "u2" if o in TS else "u1", "path": path_of(o), "line": 10,
                       "calleeName": "callee"}) for sid, o in sorted(sites.items())]
    calls = sorted((site_id(o, "c"), c) for o, c in CALLS)
    recs += [("call", {"site": s, "callee": c, "status": "confirmed", "reason": "resolved-call"}) for s, c in calls]
    names = []
    for o, cand in NAMES:
        # B->A and C->B ride on the resolved call site; F->K on F's resolved site; E->A, G->B on their own sites
        names.append((site_id(o, "c") if o in {"B", "C", "F"} else site_id(o, "n"), cand))
    recs += [("name", {"site": s, "candidate": c}) for s, c in sorted(names)]
    cps = [("CP1", "http", "GET /shop/api/x"), ("CP2", "http", "GET /shop/api/y"), ("CP3", "mcp_tool", "tool")]
    recs += [("cp", {"cpId": i, "kind": k, "keyNorm": n}) for i, k, n in cps]
    provides = sorted([("A", "CP1"), ("P3", "CP1"), ("P2", "CP2"), ("M", "CP3")])
    recs += [("provides", {"symbolId": s, "cpId": c, "status": "confirmed", "reason": "route"}) for s, c in provides]
    recs += [("consumes", {"site": site_id(o, "k"), "cpId": c, "status": st, "reason": "call"})
             for o, c, st in sorted(CONSUMERS, key=lambda r: site_id(r[0], "k"))]
    recs += [("link", {"linkId": "L1", "kind": "http", "status": "confirmed", "reason": "key", "cpId": "CP1", "provider": "A"}),
             ("link", {"linkId": "L2", "kind": "http", "status": "possible", "reason": "key", "cpId": "CP2", "provider": "P2"}),
             ("link", {"linkId": "L3", "kind": "http", "status": "possible", "reason": "key", "cpId": "CP1", "provider": None})]
    recs += [("via", {"site": "sHk", "linkId": "L1", "status": "confirmed", "reason": "key", "consumerKey": "GET /{?}/shop/api/x"}),
             ("via", {"site": "sQk", "linkId": "L2", "status": "possible", "reason": "key", "consumerKey": "GET /shop/api/y"}),
             ("via", {"site": "sQk", "linkId": "L3", "status": "possible", "reason": "key"})]
    recs += [("entry", {"entryId": "CE1", "unitId": "u1", "path": "cfg.yml", "keyPath": "a.b"})]
    recs += [("bound", {"linkId": "L1", "entryId": "CE1", "siteId": "sHk", "ord": 0})]
    return recs


def snapshot():
    """The in-memory form ``read_snapshot`` returns: ``{t: [record, ...]}`` (``meta`` is a one-record list)."""
    out = {}
    for t, r in records():
        out.setdefault(t, []).append(dict(r))
    return out


def canonical_lines():
    """The export file's lines: canonical JSON, nulls of optional fields absent, ``link.cpId``/``provider`` kept as null."""
    lines = []
    for t, r in records():
        r = {k: v for k, v in r.items() if not (v is None and k not in ("cpId", "provider"))}
        lines.append(json.dumps({"t": t, **r}, sort_keys=True, ensure_ascii=False))
    return lines


def write_snapshot(path, mutate=None):
    """Write the canonical export; ``mutate(recs)`` may edit the ``[(t, record)]`` list first (planted defects)."""
    recs = records()
    if mutate:
        recs = mutate(recs) or recs
    with open(path, "w", encoding="utf-8") as f:
        for t, r in recs:
            f.write(json.dumps({"t": t, **r}, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def anchor(kind, ident, anchor_id=None):
    key = {"symbol": "symbolId", "contract": "cpId", "config": "entryId"}[kind]
    return {"anchorId": anchor_id or f"{kind}:{ident}", "type": kind, key: ident}
