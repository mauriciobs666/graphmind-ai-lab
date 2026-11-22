#!/usr/bin/env python3
"""baseline.py - the naive over-lister baseline and its item-level recall (P1-36; ``docs/plans/connected-code-graph-p1-noise-export-graph.md``
§3-§7, ``docs/plans/connected-code-graph-p1-ml.md`` §5.1 (3)).

Pure and stdlib-only (plus ``match_keys.h1_symbol_credit``, imported, never reimplemented): no store, no clock, no randomness.
``read_snapshot`` loads and validates the custodian-side adjacency export (``snapshot_export.py``); ``naive_over_lister`` lists, per
anchor, every symbol the baseline would hand a reader (reverse call reach to the bound plus the kind-wide contract members);
``baseline_recall`` scores the baseline against the harness's gold-item list with the §6 hit rule; ``dominance`` and
``nc5_view`` are the M1 check and the Q-NC5 projection. Counts only leave this module: ids appear in no error message.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from match_keys import h1_symbol_credit  # noqa: E402

EXPORT_VERSION = 1
GRAPH_MODEL_VERSION = "1.1"
EXCLUDED_EDGES = frozenset({"->", "DATA", "CONTROLLED_BY", "HAS_PARAM", "HAS_RETURN", "HAS_COND"})

# t -> (required fields, key fields).  Optional fields (``ownedRegions``, ``consumerKey``, ``keyPath``, ``launchKind``) may be absent.
_SPEC = {
    "meta": (("snapshotId", "systemId", "graphModelVersion", "exportVersion"), ()),
    "unit": (("unitId", "repoId", "language", "buildStatus", "flowAvailable"), ("unitId",)),
    "symbol": (("symbolId", "unitId", "repoId", "qname", "path", "lineStart", "lineEnd"), ("symbolId",)),
    "site": (("siteId", "owner", "unitId", "path", "line", "calleeName"), ("siteId",)),
    "call": (("site", "callee", "status", "reason"), ("site", "callee")),
    "name": (("site", "candidate"), ("site", "candidate")),
    "cp": (("cpId", "kind", "keyNorm"), ("cpId",)),
    "provides": (("symbolId", "cpId", "status", "reason"), ("symbolId", "cpId")),
    "consumes": (("site", "cpId", "status", "reason"), ("site", "cpId")),
    "link": (("linkId", "kind", "status", "reason", "cpId", "provider"), ("linkId",)),
    "via": (("site", "linkId", "status", "reason"), ("site", "linkId")),
    "entry": (("entryId", "unitId", "path"), ("entryId",)),
    "bound": (("linkId", "entryId", "siteId", "ord"), ("linkId", "entryId", "siteId", "ord")),
}
_SITE_ROWS = ("call", "name", "consumes", "via")
# (t, field, target t, target field): a non-null value must exist in the target (``link.cpId`` may be null)
_CLOSURE = (("site", "owner", "symbol", "symbolId"), ("call", "callee", "symbol", "symbolId"),
            ("name", "candidate", "symbol", "symbolId"), ("provides", "symbolId", "symbol", "symbolId"),
            ("provides", "cpId", "cp", "cpId"), ("consumes", "cpId", "cp", "cpId"), ("link", "cpId", "cp", "cpId"),
            ("via", "linkId", "link", "linkId"), ("bound", "linkId", "link", "linkId"),
            ("bound", "entryId", "entry", "entryId"))


# ------------------------------------------------------------------------------------------------ snapshot reader
def read_snapshot(path) -> dict:
    """The export file -> ``{t: [record, ...]}`` (``meta`` a one-record list) plus ``dropped``: the counts of ``call``/``name``/
    ``consumes``/``via`` rows on a site that has no ``site`` record (no ``HAS_CALL`` owner), which are removed. Any other
    closure break, duplicate key, missing field or unknown record type raises ``ValueError`` naming the type and a count."""
    snap: dict = {t: [] for t in _SPEC}
    first = True
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec.pop("t", None)
            if first and t != "meta":
                raise ValueError("snapshot: the meta record must come first")
            first = False
            if t not in _SPEC:
                raise ValueError(f"snapshot: unknown record type {t!r} at line {n}")
            if t == "meta" and snap["meta"]:
                raise ValueError("snapshot: more than one meta record")
            snap[t].append(rec)
    if not snap["meta"]:
        raise ValueError("snapshot: the meta record must come first")
    meta = snap["meta"][0]
    for t, (req, keys) in _SPEC.items():
        bad = sum(1 for r in snap[t] if any(k not in r for k in req))
        if bad:
            raise ValueError(f"{t}: {bad} record(s) missing a required field")
        if keys:
            seen = [tuple(r[k] for k in keys) for r in snap[t]]
            if len(seen) != len(set(seen)):
                raise ValueError(f"{t}: {len(seen) - len(set(seen))} duplicate key(s)")
    if meta["exportVersion"] != EXPORT_VERSION:
        raise ValueError(f"meta: exportVersion is not {EXPORT_VERSION}")
    if meta["graphModelVersion"] != GRAPH_MODEL_VERSION:
        raise ValueError(f"meta: graphModelVersion is not {GRAPH_MODEL_VERSION!r}")
    sites = {r["siteId"] for r in snap["site"]}
    dropped = {}
    for t in _SITE_ROWS:
        kept = [r for r in snap[t] if r["site"] in sites]
        dropped[t] = len(snap[t]) - len(kept)
        snap[t] = kept
    snap["dropped"] = dropped
    ids = {t: {r[k] for r in snap[t] for k in (_SPEC[t][1][:1])} for t in ("symbol", "cp", "link", "entry")}
    for t, field, target, tfield in _CLOSURE:
        bad = sum(1 for r in snap[t] if r[field] is not None and r[field] not in ids[target])
        if bad:
            raise ValueError(f"{t}.{field}: {bad} record(s) not in {target}.{tfield}")
    return snap


# ------------------------------------------------------------------------------------------------ over-lister
class _Index:
    """Lookups over a snapshot dict, built once per call."""

    def __init__(self, snap: dict):
        self.symbols = {s["symbolId"]: s for s in snap["symbol"]}
        self.cps = {c["cpId"]: c for c in snap["cp"]}
        self.entries = {e["entryId"] for e in snap["entry"]}
        self.owner = {s["siteId"]: s["owner"] for s in snap["site"]}
        self.resolved_sites = {c["site"] for c in snap["call"] if c["site"] in self.owner}
        self.call_in: dict = {}          # callee -> {owner: confirmed}
        for c in snap["call"]:
            o = self.owner.get(c["site"])
            if o is not None:
                d = self.call_in.setdefault(c["callee"], {})
                d[o] = d.get(o, False) or c["status"] == "confirmed"
        self.name_in: dict = {}          # candidate -> [(owner, resolvedSite)]
        for n in snap["name"]:
            o = self.owner.get(n["site"])
            if o is not None:
                self.name_in.setdefault(n["candidate"], []).append((o, n["site"] in self.resolved_sites))
        self.provides: dict = {}         # cpId -> {symbolId: confirmed}
        self.provided_by: dict = {}      # symbolId -> {cpId}
        for p in snap["provides"]:
            d = self.provides.setdefault(p["cpId"], {})
            d[p["symbolId"]] = d.get(p["symbolId"], False) or p["status"] == "confirmed"
            self.provided_by.setdefault(p["symbolId"], set()).add(p["cpId"])
        self.consumes: dict = {}         # cpId -> {owner: confirmed}
        self.consumed_by: dict = {}      # owner -> {cpId}
        for c in snap["consumes"]:
            o = self.owner.get(c["site"])
            if o is not None:
                d = self.consumes.setdefault(c["cpId"], {})
                d[o] = d.get(o, False) or c["status"] == "confirmed"
                self.consumed_by.setdefault(o, set()).add(c["cpId"])
        self.link_cp = {l["linkId"]: l["cpId"] for l in snap["link"]}
        self.bound_links: dict = {}      # entryId -> {linkId}
        for b in snap["bound"]:
            self.bound_links.setdefault(b["entryId"], set()).add(b["linkId"])
        self.vias = [(self.owner.get(v["site"]), v["linkId"], v["status"] == "confirmed") for v in snap["via"]
                     if v["site"] in self.owner]

    def callers(self, sym: str, names: str = "all", confirmed_only: bool = False) -> dict:
        """``{owner: confirmed}``: owners of a ``call`` row to ``sym`` or (``names``) of a ``name`` row with ``sym`` as candidate.
        ``names``: ``all`` | ``none`` | ``no-resolved`` (ignore name rows of sites that have a call row) | ``no-unresolved``."""
        out = {o: c for o, c in self.call_in.get(sym, {}).items() if c or not confirmed_only}
        if names != "none" and not confirmed_only:
            for o, resolved in self.name_in.get(sym, ()):
                if names == "no-resolved" and resolved or names == "no-unresolved" and not resolved:
                    continue
                out.setdefault(o, False)
        return out


def _seeds(ix: _Index, anchor: dict):
    """``[(symbolId, cost, class, confirmed)]`` for the anchor (rule 2) and the contract points ``CP(anchor)`` (rule 4)."""
    t = anchor["type"]
    if t == "symbol":
        s = anchor["symbolId"]
        return [(s, 0, "anchor", True)], ix.provided_by.get(s, set()) | ix.consumed_by.get(s, set())
    if t == "contract":
        cp = anchor["cpId"]
        seeds = [(s, 0, "provider", c) for s, c in ix.provides.get(cp, {}).items()]
        seeds += [(o, 1, "consumer", c) for o, c in ix.consumes.get(cp, {}).items()]
        return seeds, {cp}
    links = ix.bound_links.get(anchor["entryId"], set())
    seeds = [(o, 2, "consumer", conf) for o, lid, conf in ix.vias if lid in links and o is not None]
    return seeds, {ix.link_cp[l] for l in links if ix.link_cp.get(l) is not None}


def _reach(seeds, bound: int, callers) -> dict:
    dist: dict = {}
    for s, c in seeds:
        if c <= bound and c < dist.get(s, bound + 1):
            dist[s] = c
    heap = [(c, s) for s, c in dist.items()]
    heapq.heapify(heap)
    while heap:
        c, s = heapq.heappop(heap)
        if c > dist[s] or c >= bound:
            continue
        for o in callers(s):
            if c + 1 < dist.get(o, bound + 1):
                dist[o] = c + 1
                heapq.heappush(heap, (c + 1, o))
    return dist


def _kind_members(ix: _Index, cps: set, kind_scope: str) -> dict:
    """``{symbolId: confirmed}``: providers and consumer owners of the kind-wide (or key) contract points (rule 4)."""
    if kind_scope == "kind":
        kinds = {ix.cps[c]["kind"] for c in cps}
        cps = {c for c, r in ix.cps.items() if r["kind"] in kinds}
    out: dict = {}
    for cp in cps:
        for src in (ix.provides.get(cp, {}), ix.consumes.get(cp, {})):
            for s, conf in src.items():
                out[s] = out.get(s, False) or conf
    return out


def _compute(ix: _Index, anchor: dict, bound: int, kind_scope: str, names: str = "all"):
    seeds, cps = _seeds(ix, anchor)
    reach = _reach([(s, c) for s, c, _, _ in seeds], bound, lambda s: ix.callers(s, names))
    kind = _kind_members(ix, cps, kind_scope)
    drop = {anchor["symbolId"]} if anchor["type"] == "symbol" else set()
    return seeds, reach, kind, (set(reach) | set(kind)) - drop


def _own_confirmed(ix: _Index, seeds, bound: int, kind: dict) -> set:
    """Rule 6: confirmed seeds, chains of confirmed ``call`` rows from them within the bound, confirmed kind members."""
    conf_seeds = [(s, c) for s, c, _, ok in seeds if ok]
    chain = _reach(conf_seeds, bound, lambda s: ix.callers(s, confirmed_only=True))
    return set(chain) | {s for s, ok in kind.items() if ok}


def _by_seed(ix: _Index, seeds, bound: int, reach: dict, kind: dict, items: set) -> dict:
    call_only = {}
    for cls in {c for _, _, c, _ in seeds}:
        call_only[cls] = set(_reach([(s, c) for s, c, k, _ in seeds if k == cls], bound, lambda s: ix.callers(s, "none")))
    reached_by_call = set().union(*call_only.values()) if call_only else set()
    part = dict(kindWideOnly=0, nameOnly=0, providerOnly=0, consumerOrConfigOnly=0, other=0)
    for s in items:
        if s in kind and s not in reach:
            part["kindWideOnly"] += 1
        elif s not in reached_by_call:
            part["nameOnly"] += 1
        else:
            classes = {c for c, r in call_only.items() if s in r}
            key = {"provider": "providerOnly", "consumer": "consumerOrConfigOnly"}.get(next(iter(classes)), "other") \
                if len(classes) == 1 else "other"
            part[key] += 1
    return part


def _validate(snapshot: dict, anchors: list, bound, direction: str, kind_scope: str) -> None:
    if direction != "impact":
        raise ValueError("direction: only 'impact' is supported at P1")
    if kind_scope not in ("kind", "key"):
        raise ValueError("kind_scope: must be 'kind' or 'key'")
    if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
        raise ValueError("bound: must be an int >= 0")
    ids = {"symbol": ("symbolId", {s["symbolId"] for s in snapshot["symbol"]}),
           "contract": ("cpId", {c["cpId"] for c in snapshot["cp"]}),
           "config": ("entryId", {e["entryId"] for e in snapshot["entry"]})}
    seen, absent = set(), 0
    for a in anchors:
        aid, t = a.get("anchorId"), a.get("type")
        if aid is None:
            raise ValueError("anchor: missing anchorId")
        if aid in seen:
            raise ValueError("anchor: 1 duplicate anchorId")
        seen.add(aid)
        if t == "unit":
            raise ValueError("anchor: a unit anchor arrives as the contract anchor of its probe, not as type 'unit'")
        if t == "unresolved":
            continue
        if t not in ids:
            raise ValueError("anchor: unknown anchor type")
        key, known = ids[t]
        if key not in a:
            raise ValueError(f"anchor: a {t} anchor needs {key}")
        absent += a[key] not in known
    if absent:
        raise ValueError(f"anchor: {absent} resolved anchor(s) absent from the snapshot (a build or export mismatch)")


def naive_over_lister(snapshot: dict, anchors: list, bound: int, *, direction: str = "impact",
                      kind_scope: str = "kind", in_scope=None) -> list:
    """The baseline's burden, one row per anchor in ``anchorId`` order (§4). ``bound`` has no default: callers read it from the
    ``ac5-gate-1`` record's ``defaultDepthBound``."""
    _validate(snapshot, anchors, bound, direction, kind_scope)
    ix = _Index(snapshot)
    rows = []
    for a in sorted(anchors, key=lambda a: a["anchorId"]):
        if a["type"] == "unresolved":
            rows.append({"anchorId": a["anchorId"], "type": "unresolved", "count": 0, "countInScope": 0,
                         "countConfirmedOwn": 0, "items": [], "unresolved": True})
            continue
        seeds, reach, kind, items = _compute(ix, a, bound, kind_scope)
        confirmed = _own_confirmed(ix, seeds, bound, kind) & items
        by_name = {key: len(items - _compute(ix, a, bound, kind_scope, mode)[3])
                   for key, mode in (("resolvedSite", "no-resolved"), ("unresolvedSite", "no-unresolved"))}
        rows.append({"anchorId": a["anchorId"], "type": a["type"], "count": len(items),
                     "countInScope": len(items) if in_scope is None else sum(1 for s in items if in_scope(ix.symbols[s])),
                     "countConfirmedOwn": len(confirmed), "byName": by_name,
                     "bySeed": _by_seed(ix, seeds, bound, reach, kind, items), "items": sorted(items)})
    return rows


# ------------------------------------------------------------------------------------------------ item-level recall (§6)
def _gold_line(g: dict) -> str:
    e = g["impacted"]
    ident = e["qname"] if "qname" in e else e.get("keyPath")
    return f"{g['anchorId']}|{e['type']}:{e['repoId']}:{e['path']}:{ident}|{','.join(str(l) for l in sorted(g['lines']))}"


def itemSetSha256(items: list) -> str:  # noqa: N802  (the record's own key name)
    """sha256 (UTF-8) of the gold items' ``anchorId|type:repoId:path:qname-or-keyPath|l1,l2`` lines, sorted by code point and
    joined by ``\\n`` with no trailing newline."""
    return hashlib.sha256("\n".join(sorted(_gold_line(g) for g in items)).encode("utf-8")).hexdigest()


def _paired(x: dict) -> dict:
    o = x.get("ownedRegions")
    if not o:
        return x
    if len(o) % 2:
        raise ValueError("symbol: an ownedRegions list of odd length")
    return {**x, "ownedRegions": [[o[i], o[i + 1]] for i in range(0, len(o), 2)]}


def recall_flags(fig: dict) -> dict:
    """Exact cross-multiplied statements (§6): ``reduction`` iff baseline recall is lower, ``baselineRecallsMore`` iff higher."""
    out = {}
    for suffix, sk, bk in (("", "system", "baseline"), ("Sym", "systemSym", "baselineSym")):
        if sk in fig:
            (sk_, sn), (bk_, bn) = fig[sk], fig[bk]
            out["reduction" + suffix] = bk_ * sn < sk_ * bn
            out["baselineRecallsMore" + suffix] = bk_ * sn > sk_ * bn
    return out


def _figures(items: list, base_hit: list) -> dict:
    sym = [i for i, g in enumerate(items) if g["impacted"]["type"] == "symbol"]
    sys_hit = [bool(g["hit"]) for g in items]
    both = sum(s and b for s, b in zip(sys_hit, base_hit))
    s_only = sum(s and not b for s, b in zip(sys_hit, base_hit))
    b_only = sum(b and not s for s, b in zip(sys_hit, base_hit))
    fig = {"system": [sum(sys_hit), len(items)], "baseline": [sum(base_hit), len(items)],
           "systemSym": [sum(sys_hit[i] for i in sym), len(sym)], "baselineSym": [sum(base_hit[i] for i in sym), len(sym)],
           "paired": {"both": both, "systemOnly": s_only, "baselineOnly": b_only,
                      "neither": len(items) - both - s_only - b_only},
           "configGold": sum(1 for g in items if g["impacted"]["type"] == "config"),
           "itemSetSha256": itemSetSha256(items)}
    fig["flags"] = recall_flags(fig)
    return fig


def baseline_recall(snapshot: dict, baseline_rows: list, items: list) -> dict:
    """§6 figures. ``items``: the harness's scored counting gold items, ``{anchorId, impacted, lines, hit, keyError}``; a gold item
    is a baseline hit iff it is symbol-typed and some baseline item **of its own anchor** passes ``h1_symbol_credit``."""
    rows = {r["anchorId"]: r for r in baseline_rows}
    if len(rows) != len(baseline_rows):
        raise ValueError("baseline: duplicate anchorId in the baseline rows")
    missing = len({g["anchorId"] for g in items} - set(rows))
    if missing:
        raise ValueError(f"anchor: {missing} gold anchor(s) have no baseline row")
    symbols = {s["symbolId"]: _paired(s) for s in snapshot["symbol"]}
    hits = [g["impacted"]["type"] == "symbol"
            and any(h1_symbol_credit(symbols[s], g["impacted"], g["lines"]) for s in rows[g["anchorId"]]["items"])
            for g in items]
    out = _figures(items, hits)
    keep = [i for i, g in enumerate(items) if not g.get("keyError")]
    ex = _figures([items[i] for i in keep], [hits[i] for i in keep])
    out["exKeyError"] = {k: ex[k] for k in ("system", "baseline", "systemSym", "baselineSym", "paired", "configGold",
                                            "itemSetSha256", "flags")}
    return out


# ------------------------------------------------------------------------------------------------ dominance (§7, M1/N3)
def dominance(system_items: list, baseline_rows: list) -> dict:
    """Every system symbol item (``{anchorId, symbolId, edges: [edge type, ...]}``) must be in the baseline's items for its anchor.
    Class A: no excluded edge in the chain (expected 0 violations); class B: the chain has one, split by the first."""
    rows = {r["anchorId"]: set(r["items"]) for r in baseline_rows}
    unknown = len({s["anchorId"] for s in system_items} - set(rows))
    if unknown:
        raise ValueError(f"anchor: {unknown} system anchor(s) have no baseline row")
    a = {"checked": 0, "violations": 0}
    b = {"checked": 0, "violations": 0}
    by_edge: dict = {}
    for it in system_items:
        miss = it["symbolId"] not in rows[it["anchorId"]]
        first = next((e for e in it["edges"] if e in EXCLUDED_EDGES), None)
        for tally in ((a,) if first is None else (b, by_edge.setdefault(first, {"checked": 0, "violations": 0}))):
            tally["checked"] += 1
            tally["violations"] += miss
    return {"A": a, "B": {**b, "byEdge": by_edge}}


# ------------------------------------------------------------------------------------------------ Q-NC5 (§5)
def nc5_view(snapshot: dict) -> dict:
    """The ``noise.unbound_multi_unit`` input projected from the export: ``links`` (every link with a contract point) and ``providers``."""
    cps = {c["cpId"]: c for c in snapshot["cp"]}
    kinds: dict = {}
    for c in cps.values():
        kinds.setdefault(c["keyNorm"], set()).add(c["kind"])
    clash = sum(1 for k in kinds.values() if len(k) > 1)
    if clash:
        raise ValueError(f"cp: {clash} keyNorm(s) occur under more than one kind")
    bound = {b["linkId"] for b in snapshot["bound"]}
    units = {s["symbolId"]: s["unitId"] for s in snapshot["symbol"]}
    links = [{"keyNorm": cps[l["cpId"]]["keyNorm"], "bound": l["linkId"] in bound, "status": l["status"]}
             for l in snapshot["link"] if l["cpId"] is not None]
    pairs = sorted({(cps[p["cpId"]]["keyNorm"], units[p["symbolId"]]) for p in snapshot["provides"]})
    return {"links": links, "providers": [{"keyNorm": k, "unitId": u} for k, u in pairs]}
