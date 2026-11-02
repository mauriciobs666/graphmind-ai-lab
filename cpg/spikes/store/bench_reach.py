"""S20 (P0-15) -- BFS (``bfs-v1``) vs ``algo.SSpaths`` on the 1,000-file synthetic graph; the decision by §4.5's rule.

Contract: ``docs/plans/connected-code-graph-p0.md`` §4.5; ``docs/plans/connected-code-graph-p0-2.md`` S20 row.

Decision rule (§4.5): adopt ``SSpaths`` for a segment only if all three hold --
  1. 20/20 repeats identical, chains included;  2. its item set equals the BFS's;
  3. p95 answer latency at least 30 % lower at depth >= 3.
Otherwise the BFS. ``SSpaths`` takes one ``relDirection``, so it can stand in only for a single-direction, edge-typed
segment: **call-only reverse reach** (``CALLS`` + ``HAS_CALL``, incoming). The full ``bfs-v1`` walk is benchmarked
alone (latency, repeat-diff); ``bfs.py`` says which ``impact.py`` it is.

``SSpaths`` call: ``relTypes ['CALLS','HAS_CALL']``, ``relDirection 'incoming'``, ``weightProp 'w'`` (confirmed 1,
possible 1000, on both edge types), ``maxLen 2 * D``, ``pathCount`` = :data:`PATH_COUNT`. The engine returns **paths**,
not best-path-per-node; ``pathCount`` is a global, weight-ordered path budget, and ``RESULTSET_SIZE`` (also 10,000)
separately caps the rows of a reply; :func:`items_from_paths` keeps, per end symbol, the
lightest returned path and labels it by ``bfs-v1``'s rule (hop 1 by its own edge status, every later hop ``possible``
by function-level continuation), so the two sides are compared on one vocabulary.

Usage::

    python bench_reach.py run --out results/bench_reach.jsonl [--repeats 20] [--only segment|full]
    python bench_reach.py summarize results/bench_reach.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bfs as B  # noqa: E402
import synth as S  # noqa: E402

GRAPH = "ccg_p0spike"
READ_TIMEOUT_MS = 60_000  # explicit on every read: the engine's default read TIMEOUT is 1,000 ms (measured; a hub-frontier read exceeds it)
# pathCount is a global path budget (count(*) = pathCount while paths remain, results/sspaths_pathcount_probe.json); this value
# equals RESULTSET_SIZE, so in the row-returning benchmark either cap alone would give 10,000 rows. See sspaths_agg_check.py.
PATH_COUNT = 10_000
TRUNCATED_AT = 10_000  # a reply of this many rows may have been cut (RESULTSET_SIZE or pathCount)
SEGMENT_DEPTHS = (1, 3, 5)  # call hops D; bfs levels = 2 * D; SSpaths maxLen = 2 * D
FULL_LEVELS = (4, 6)  # 6 levels = distance <= 3 (the §6.4 depth-3 cap)
SSPATHS_QUERY = (
    "MATCH (a:Symbol {symbolId: $id}) CALL algo.SSpaths({sourceNode: a, relTypes: ['CALLS','HAS_CALL'], "
    "relDirection: 'incoming', weightProp: 'w', maxLen: $maxlen, pathCount: $pc}) YIELD path, pathWeight "
    "RETURN [n IN nodes(path) | coalesce(n.symbolId, n.callSiteId)], [r IN relationships(path) | type(r)], "
    "[r IN relationships(path) | coalesce(r.status, 'confirmed')], pathWeight"
)


# ---------------------------------------------------------------------------------------------
# SSpaths rows -> items (pure).
# ---------------------------------------------------------------------------------------------


def items_from_paths(anchor: str, rows: list[list]) -> list[list]:
    """``[symbol, status, distance, possible hops, length, node-id chain]`` per end symbol, like ``bfs.canonical_items``.

    A row is ``[node ids, relationship types, relationship statuses, pathWeight]``; only a path that ends on a
    ``Symbol`` through ``HAS_CALL`` (edge types alternate ``CALLS``, ``HAS_CALL``) is an item, and the anchor is none.
    Per end symbol the best path is the minimum of ``(possible hops, distance, length, node ids)``."""
    best: dict[str, tuple] = {}
    for ids, types, statuses, _w in rows:
        if not types or types[-1] != "HAS_CALL" or ids[-1] == anchor:
            continue
        calls = [i for i, t in enumerate(types) if t == "CALLS"]
        distance = len(calls)
        poss = sum(1 for k, i in enumerate(calls) if k > 0 or statuses[i] != "confirmed")
        cost = (poss, distance, len(types), tuple(ids))
        if ids[-1] not in best or cost < best[ids[-1]]:
            best[ids[-1]] = cost
    return sorted(([sym, "possible" if c[0] else "confirmed", c[1], c[0], c[2], list(c[3])] for sym, c in best.items()),
                  key=lambda r: r[0])


def sspaths_answer(graph, anchor: str, depth: int, *, pc: int = PATH_COUNT, timeout: int | None = READ_TIMEOUT_MS) -> dict:
    t0 = time.perf_counter()
    rows = graph.ro_query(SSPATHS_QUERY, {"id": anchor, "maxlen": 2 * depth, "pc": pc}, timeout).result_set
    t_query = time.perf_counter() - t0
    items = items_from_paths(anchor, rows)
    return {"items": items, "digest": B.digest(items), "seconds": time.perf_counter() - t0, "queryseconds": t_query,
            "rows": len(rows), "nItems": len(items), "truncated": len(rows) >= TRUNCATED_AT}


# ---------------------------------------------------------------------------------------------
# Comparison (pure).
# ---------------------------------------------------------------------------------------------


def compare(bfs_items: list[list], ss_items: list[list]) -> dict:
    """Item-set equality and agreement on the common items; BFS chains are ``[[label, id], state, edge]`` steps."""
    b = {r[0]: r for r in bfs_items}
    s = {r[0]: r for r in ss_items}
    common = sorted(set(b) & set(s))
    return {
        "bfsItems": len(b), "sspathsItems": len(s), "common": len(common), "missingFromSSpaths": len(set(b) - set(s)),
        "extraInSSpaths": len(set(s) - set(b)), "itemSetEqual": set(b) == set(s),
        "statusAgree": sum(1 for k in common if b[k][1] == s[k][1]),
        "distanceAgree": sum(1 for k in common if b[k][2] == s[k][2]),
        "chainAgree": sum(1 for k in common if [st[0][1] for st in b[k][5]] == s[k][5]),
    }


def percentile(xs: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of a non-empty list."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, -(-len(xs) * q // 100) - 1))]


def adopt_decision(rows: list[dict]) -> dict:
    """The §4.5 rule over per-(anchor, depth >= 3) records: ``rows`` hold the three facts for one depth set."""
    r1 = all(r["sspathsRepeatIdentical"] for r in rows)
    r2 = all(r["compare"]["itemSetEqual"] for r in rows)
    bfs_p95 = percentile([x for r in rows for x in r["bfsSeconds"]], 95)
    ss_p95 = percentile([x for r in rows for x in r["sspathsSeconds"]], 95)
    r3 = ss_p95 <= 0.7 * bfs_p95
    return {"rule1_repeats_identical": r1, "rule2_item_set_equal": r2, "rule3_p95_30pct_lower": r3,
            "bfsP95": bfs_p95, "sspathsP95": ss_p95, "adopt": r1 and r2 and r3}


# ---------------------------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------------------------


def _read_stats(trackers: list[B.Tracker]) -> dict:
    by: dict[str, list[float]] = {}
    for t in trackers:
        for _lvl, tag, sec, _ids, _rows in t.reads:
            by.setdefault(tag, []).append(sec * 1000)
    return {tag: [round(x, 3) for x in v] for tag, v in sorted(by.items())}


def run_bfs(graph, anchor, mode, levels, repeats):
    secs, digests, trackers, last = [], [], [], None
    for _ in range(repeats):
        r = B.answer(graph, anchor, mode=mode, levels=levels, timeout=READ_TIMEOUT_MS)
        secs.append(r["seconds"])
        digests.append(r["digest"])
        trackers.append(r["tracker"])
        last = r
    level_ms = [round(t[3] * 1000, 3) for tr in trackers for t in tr.levels]
    return {"seconds": [round(x, 4) for x in secs], "digests": digests, "identical": len(set(digests)) == 1,
            "nItems": last["nItems"], "states": last["states"], "levels": last["levels"], "splits": last["splits"],
            "reads": _read_stats(trackers), "levelMs": level_ms, "items": last["items"]}


def run_sspaths(graph, anchor, depth, repeats):
    secs, digests, last = [], [], None
    for _ in range(repeats):
        r = sspaths_answer(graph, anchor, depth)
        secs.append(r["seconds"])
        digests.append(r["digest"])
        last = r
    return {"seconds": [round(x, 4) for x in secs], "digests": digests, "identical": len(set(digests)) == 1,
            "nItems": last["nItems"], "rows": last["rows"], "truncated": last["truncated"], "items": last["items"]}


def run(out: Path, repeats: int, only: str | None, hub_repeats_deep: int) -> None:
    store = S.SpikeStore("127.0.0.1", 6390)
    graph = store.db.select_graph(GRAPH)
    synth = S.Synth(1000)
    anchors = synth.anchors()
    prov = B.impact_provenance()
    tmax = store.timeout_max()
    done: set = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "segment":
                done.add(("segment", r["depth"], r["anchor"]["symbolId"]))
            elif r["kind"] == "full":
                done.add(("full", r["levels"], r["anchor"]["symbolId"]))
    with out.open("a") as fh:
        fh.write(json.dumps({"kind": "meta", "impact": prov, "timeoutMax": tmax, "anchors": anchors, "repeats": repeats,
                             "pathCount": PATH_COUNT, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        fh.flush()
        mid = next(a for a in anchors if a["class"] == "mid")
        for depth in SEGMENT_DEPTHS:  # warm the plan cache for every query text
            B.answer(graph, mid["symbolId"], mode="segment", levels=2 * depth)
            sspaths_answer(graph, mid["symbolId"], depth)
            B.answer(graph, mid["symbolId"], mode="full", levels=4)
        if only in (None, "segment"):
            for depth in SEGMENT_DEPTHS:
                for a in anchors:
                    if ("segment", depth, a["symbolId"]) in done:
                        continue
                    n = hub_repeats_deep if (a["class"] == "hub" and depth >= 5) else repeats
                    rb = run_bfs(graph, a["symbolId"], "segment", 2 * depth, n)
                    rs = run_sspaths(graph, a["symbolId"], depth, n)
                    cmp = compare(rb["items"], rs["items"])
                    fh.write(json.dumps({"kind": "segment", "depth": depth, "anchor": a, "repeats": n,
                                         "bfs": {k: v for k, v in rb.items() if k != "items"},
                                         "sspaths": {k: v for k, v in rs.items() if k != "items"}, "compare": cmp}) + "\n")
                    fh.flush()
        if only in (None, "full"):
            for levels in FULL_LEVELS:
                for a in anchors:
                    if ("full", levels, a["symbolId"]) in done:
                        continue
                    n = hub_repeats_deep if (a["class"] == "hub" and levels >= 6) else repeats
                    rb = run_bfs(graph, a["symbolId"], "full", levels, n)
                    fh.write(json.dumps({"kind": "full", "levels": levels, "anchor": a, "repeats": n,
                                         "bfs": {k: v for k, v in rb.items() if k != "items"}}) + "\n")
                    fh.flush()
        fh.write(json.dumps({"kind": "end", "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


# ---------------------------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------------------------


def summarize(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    meta = next(r for r in recs if r["kind"] == "meta")
    out: dict = {"impact": meta["impact"], "timeoutMax": meta["timeoutMax"], "segment": {}, "full": {}}
    for depth in sorted({r["depth"] for r in recs if r["kind"] == "segment"}):
        rows = [r for r in recs if r["kind"] == "segment" and r["depth"] == depth]
        flat = lambda key, side: [x for r in rows for x in r[side][key]]  # noqa: E731
        per_class = {}
        for c in ("hub", "mid", "low"):
            cr = [r for r in rows if r["anchor"]["class"] == c]
            if not cr:
                continue
            per_class[c] = {
                "anchors": len(cr), "bfsItemsMedian": statistics.median(r["bfs"]["nItems"] for r in cr),
                "sspathsItemsMedian": statistics.median(r["sspaths"]["nItems"] for r in cr),
                "itemSetEqual": sum(r["compare"]["itemSetEqual"] for r in cr),
                "sspathsTruncated": sum(r["sspaths"]["truncated"] for r in cr),
                "bfsP95": percentile([x for r in cr for x in r["bfs"]["seconds"]], 95),
                "sspathsP95": percentile([x for r in cr for x in r["sspaths"]["seconds"]], 95),
                "bfsRepeatIdentical": sum(r["bfs"]["identical"] for r in cr),
                "sspathsRepeatIdentical": sum(r["sspaths"]["identical"] for r in cr),
                "missingFraction": round(sum(r["compare"]["missingFromSSpaths"] for r in cr) /
                                         max(1, sum(r["compare"]["bfsItems"] for r in cr)), 4),
            }
        dec_rows = [{"sspathsRepeatIdentical": r["sspaths"]["identical"], "compare": r["compare"],
                     "bfsSeconds": r["bfs"]["seconds"], "sspathsSeconds": r["sspaths"]["seconds"]} for r in rows]
        out["segment"][depth] = {
            "anchors": len(rows), "repeatsPerAnchor": sorted({r["repeats"] for r in rows}),
            "bfsRepeatIdentical": f"{sum(r['bfs']['identical'] for r in rows)}/{len(rows)}",
            "sspathsRepeatIdentical": f"{sum(r['sspaths']['identical'] for r in rows)}/{len(rows)}",
            "itemSetEqual": f"{sum(r['compare']['itemSetEqual'] for r in rows)}/{len(rows)}",
            "statusAgreeOnCommon": f"{sum(r['compare']['statusAgree'] for r in rows)}/{sum(r['compare']['common'] for r in rows)}",
            "distanceAgreeOnCommon": f"{sum(r['compare']['distanceAgree'] for r in rows)}/{sum(r['compare']['common'] for r in rows)}",
            "chainAgreeOnCommon": f"{sum(r['compare']['chainAgree'] for r in rows)}/{sum(r['compare']['common'] for r in rows)}",
            "bfsP50": percentile(flat("seconds", "bfs"), 50), "bfsP95": percentile(flat("seconds", "bfs"), 95),
            "sspathsP50": percentile(flat("seconds", "sspaths"), 50), "sspathsP95": percentile(flat("seconds", "sspaths"), 95),
            "perClass": per_class, "decisionRule": adopt_decision(dec_rows) if depth >= 3 else "n/a (depth < 3)",
        }
    reads: dict[str, list[float]] = {}
    for r in recs:
        if r["kind"] in ("segment", "full"):
            for tag, v in r["bfs"]["reads"].items():
                reads.setdefault(tag, []).extend(v)
    out["perReadMs"] = {tag: {"n": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95), "p99": percentile(v, 99), "max": max(v)}
                        for tag, v in sorted(reads.items())}
    allv = [x for v in reads.values() for x in v]
    out["perReadMsAll"] = {"n": len(allv), "p50": percentile(allv, 50), "p95": percentile(allv, 95), "p99": percentile(allv, 99), "max": max(allv)}
    lv = [x for r in recs if r["kind"] in ("segment", "full") for x in r["bfs"]["levelMs"]]
    out["perLevelMs"] = {"n": len(lv), "p50": percentile(lv, 50), "p95": percentile(lv, 95), "p99": percentile(lv, 99), "max": max(lv)}
    for lvl in sorted({r["levels"] for r in recs if r["kind"] == "full"}):
        rows = [r for r in recs if r["kind"] == "full" and r["levels"] == lvl]
        out["full"][lvl] = {
            "anchors": len(rows), "repeatsPerAnchor": sorted({r["repeats"] for r in rows}),
            "bfsRepeatIdentical": f"{sum(r['bfs']['identical'] for r in rows)}/{len(rows)}",
            "p50": percentile([x for r in rows for x in r["bfs"]["seconds"]], 50),
            "p95": percentile([x for r in rows for x in r["bfs"]["seconds"]], 95),
            "max": max(x for r in rows for x in r["bfs"]["seconds"]),
            "splits": sum(r["bfs"]["splits"] for r in rows),
            "itemsMedianByClass": {c: statistics.median(r["bfs"]["nItems"] for r in rows if r["anchor"]["class"] == c)
                                   for c in ("hub", "mid", "low") if any(r["anchor"]["class"] == c for r in rows)},
            "maxStates": max(r["bfs"]["states"] for r in rows),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--repeats", type=int, default=20)
    r.add_argument("--hub-repeats-deep", type=int, default=5)
    r.add_argument("--only", choices=("segment", "full"))
    s = sub.add_parser("summarize")
    s.add_argument("path", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        run(args.out, args.repeats, args.only, args.hub_repeats_deep)
    else:
        print(json.dumps(summarize(args.path), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
