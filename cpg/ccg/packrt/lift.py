"""Role lifting to a fixpoint with ordered ``LIFTED_VIA`` evidence (proposal §3.3 "B2"; rule-pack spec §1.6 ``liftings``).

The fixpoint is the S12 recognizer's ``lift_role`` (``cpg/spikes/vertical-slice/assemble.py``, the conformance oracle),
parameterized by the pack's seeds and companions instead of the hard-coded ``fetch``/``a0``/``a1.method``, and extended
with status propagation: a lifted slot is as weak as the weakest ``data`` record on its chain (a ``possible`` ``data``
record, ``shell-dynamic`` or ``field-truncated``, makes the consumer ``possible``; the slice ignored ``data`` status).

Nodes: ``("slot", callsiteId, argPath)`` -- the value at that call argument has the role; ``("param", symbolId, "p<i>[.f]")``
-- the function's parameter has it. A predecessor is ``(node, kind, extra, weak)`` one step *towards the seed*; ``weak`` is the
``data`` record that crossed that step when it was not ``confirmed``.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ccg.packrt.extract import Corpus
from ccg.packrt.httpkey import UNKNOWN, VERBS
from ccg.packrt.match import clause_hit

_PARAM_PATH = re.compile(r"^p(\d+)(.*)$")
_RESULT_PATH = re.compile(r"^r(.*)$")


def lift_role(corpus: Corpus, seeds: list[tuple[str, str]]) -> dict[tuple, tuple | None]:
    """Fixpoint of a role from seed ``(callsiteId, argPath)`` slots; returns ``node -> predecessor``."""
    pred: dict[tuple, tuple | None] = {}
    work: list[tuple] = []
    for cid, path in sorted(seeds):
        node = ("slot", cid, path)
        pred[node] = None
        work.append(node)

    def add(node: tuple, parent: tuple, kind: str, extra=None, weak=None) -> None:
        if node not in pred:
            pred[node] = (parent, kind, extra, weak)
            work.append(node)

    def weak_of(*records: dict) -> str | None:
        return next((r["id"] for r in records if r.get("status", "confirmed") != "confirmed"), None)

    while work:
        node = work.pop(0)
        if node[0] == "slot":
            _, cid, path = node
            site = corpus.callsites[cid]
            for d, def_path in sorted(corpus.flows_into(cid, path), key=lambda x: x[0]["id"]):
                dp = corpus.params.get(d["def"])
                if dp is not None:
                    if _PARAM_PATH.match(def_path):
                        add(("param", site["symbolId"], def_path), node, "param-def", None, weak_of(d))
                    continue
                dcs = corpus.callsites.get(d["def"])
                m = _RESULT_PATH.match(def_path)
                if dcs is None or not m:
                    continue
                rest = m.group(1)  # "" or ".status": the call result's own path
                for t in sorted(dcs["targets"], key=lambda t: t["symbolId"]):
                    for rid in corpus.rets_of.get(t["symbolId"], ()):
                        for rd, rdef in sorted(corpus.flows_into(rid, "ret" + rest), key=lambda x: x[0]["id"]):
                            if rd["def"] not in corpus.params:
                                continue
                            pm = _PARAM_PATH.match(rdef)
                            if pm:
                                add(("slot", dcs["id"], f"a{pm.group(1)}{pm.group(2)}"), node, "call-ret",
                                    (t["symbolId"], rdef), weak_of(d, rd))
        else:
            _, sym_id, ppath = node
            pm = _PARAM_PATH.match(ppath)
            for caller in corpus.callers_of.get(sym_id, ()):
                add(("slot", caller["id"], f"a{pm.group(1)}{pm.group(2)}"), node, "caller")
    return pred


def chain_of(pred: dict, node: tuple) -> list[tuple]:
    """``node`` and its predecessors up to the seed, consumer side first."""
    out, cur = [], node
    while cur is not None:
        out.append(cur)
        cur = pred[cur][0] if pred[cur] else None
    return out


def weakest(pred: dict, node: tuple) -> str | None:
    """The first non-``confirmed`` ``data`` record id on ``node``'s chain, consumer side first; ``None`` when all are."""
    return next((pred[n][3] for n in chain_of(pred, node) if pred[n] and pred[n][3]), None)


def lifted_hops(corpus: Corpus, pred: dict, node: tuple, verb_pred: dict | None) -> list[dict]:
    """The ordered ``LIFTED_VIA`` chain for the url slot ``node``: consumer site first, seed last.

    A callsite hop is the consumer call (or the seed); a symbol hop is a function whose parameter carries the role
    (``urlPath`` = that parameter path) or a callee through whose return value the role flows. A symbol hop reached
    through an inner call site names it (``viaSiteId``/``viaLine``).
    """
    chain = [(n, pred[n]) for n in chain_of(pred, node)]
    verb_paths = _verb_paths(verb_pred)
    hops: list[dict] = []
    for i, (n, p) in enumerate(chain):
        if n[0] == "param":
            hops.append(_symbol_hop(corpus, n[1], n[2], verb_paths.get(("symbol", n[1]))))
            continue
        site = corpus.callsites[n[1]]
        if i == 0 or p is None:
            hops.append({
                "kind": "callsite", "id": site["id"], "path": corpus.symbols[site["symbolId"]]["path"],
                "line": site["line"], "callee": site["calleeName"], "urlPath": n[2],
                "verbPath": verb_paths.get(("callsite", site["id"])),
            })
        elif chain[i - 1][0][0] == "param":
            hops[-1]["viaSiteId"], hops[-1]["viaLine"] = site["id"], site["line"]
        if p is not None and p[1] == "call-ret":
            tid, rdef = p[2]
            hops.append(_symbol_hop(corpus, tid, rdef, verb_paths.get(("symbol", tid))))
    for i, h in enumerate(hops):
        h["ord"] = i
    return hops


def _symbol_hop(corpus: Corpus, sym_id: str, ppath: str, verb_path: str | None) -> dict:
    sym = corpus.symbols[sym_id]
    idx = int(_PARAM_PATH.match(ppath).group(1))
    line = next((p["line"] for p in corpus.params.values() if p["symbolId"] == sym_id and p["index"] == idx),
                sym["lineStart"])
    return {"kind": "symbol", "id": sym_id, "qname": sym["qname"], "path": sym["path"], "line": line,
            "lineStart": sym["lineStart"], "urlPath": ppath, "verbPath": verb_path}


def _verb_paths(verb_pred: dict | None) -> dict:
    """``{(kind, id): path}`` for every node the verb role reached (first path in sort order wins)."""
    out: dict = {}
    for vn in sorted(verb_pred or ()):
        key = ("callsite", vn[1]) if vn[0] == "slot" else ("symbol", vn[1])
        out.setdefault(key, vn[2])
    return out


def _verbs_at(corpus: Corpus, chain_slots: list[str], verb_nodes: dict, seed_ids: set, comp: dict) -> tuple[list[str], bool]:
    """``(verbs, partial)`` for a consumer: the first call on its lifting chain (consumer side first) whose lifted companion
    slot resolves; else the companion's ``bareDefault`` for a seed call with no second argument; else its ``unknown``.
    ``partial`` is true when that slot also holds an unresolved alternative (``[GET, unknown]``)."""
    unknown = comp.get("unknown", "*")
    for cid in chain_slots:
        alts: set[str] = set()
        partial = False
        for _, _, path in sorted(verb_nodes.get(cid, ())):
            for a in corpus.slot_alternatives(cid, path) or ():
                if a == UNKNOWN:
                    partial = True
                else:
                    v = a.upper()
                    alts.add(v if v in VERBS else unknown)
        if alts:
            return sorted(alts), partial
    seed = chain_slots[-1]
    head = comp["arg"].split(".")[0]
    if "bareDefault" in comp and seed in seed_ids \
            and not any(k[0] == seed and k[1].startswith(head) for k in corpus.value_sets) \
            and not any(d["toPath"].startswith(head) for d in corpus.data_by_use.get(seed, ())):
        return [comp["bareDefault"]], False
    return [unknown], False


def run_lifting(corpus: Corpus, pack_id: str, lifting: dict) -> list[dict]:
    """Every lifted ``slot`` node of ``lifting`` (sorted), resolved or not, with its hops, verbs and weakest link."""
    seeds = [(c["id"], seed["arg"]) for seed in lifting["seeds"] for c in sorted(corpus.callsites.values(), key=lambda c: c["id"])
             if clause_hit("callsite", seed["call"], c, corpus)]
    seed_ids = {cid for cid, _ in seeds}
    url_pred = lift_role(corpus, seeds)
    companion = next((c for c in lifting.get("companions", ()) if c["role"] == "verb"), None)
    verb_pred = None
    verb_slots: dict[str, list[tuple]] = defaultdict(list)
    if companion:
        verb_pred = lift_role(corpus, [(cid, companion["arg"]) for cid in sorted(seed_ids)])
        for n in verb_pred:
            if n[0] == "slot":
                verb_slots[n[1]].append(n)
    has_child = {p[0] for p in url_pred.values() if p}
    out = []
    for node in sorted(n for n in url_pred if n[0] == "slot"):
        _, cid, path = node
        site = corpus.callsites[cid]
        alts = corpus.slot_alternatives(cid, path)
        resolved = alts is not None and not all(a == UNKNOWN for a in alts)
        item = {
            "packId": pack_id, "lifting": lifting["id"], "siteId": cid, "symbolId": site["symbolId"],
            "path": corpus.symbols[site["symbolId"]]["path"], "line": site["line"], "urlPath": path,
            "alternatives": alts or [], "resolved": resolved, "leaf": node not in has_child,
            "hops": lifted_hops(corpus, url_pred, node, verb_pred), "verbs": None, "partialAlternatives": False,
            "weakenedBy": weakest(url_pred, node),
        }
        if resolved and companion:
            chain_slots = [n[1] for n in chain_of(url_pred, node) if n[0] == "slot"]
            verbs, verb_partial = _verbs_at(corpus, chain_slots, verb_slots, seed_ids, companion)
            item["verbs"] = verbs
            item["partialAlternatives"] = verb_partial or UNKNOWN in alts
        elif resolved:
            item["partialAlternatives"] = UNKNOWN in alts
        item["status"], item["reason"] = ("possible", "partial-key") if (
            not resolved or item["partialAlternatives"] or item["weakenedBy"]) else ("confirmed", "lifted-role")
        out.append(item)
    return out
