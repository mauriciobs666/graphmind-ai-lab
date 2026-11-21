"""P1-42, pure helpers (no Docker, no store): the AC-level checkers over canonical answer documents.

Plan: `docs/plans/connected-code-graph-p1.md` P1-42; `docs/test-plans/connected-code-graph-p1.md` §7 (P1T-17..25).
Nothing here reads the code under test: every expectation is the requirements text (`docs/requirements/connected-code-graph.md` AC-6, 9, 16, 17, 18, 21, 25) or a manifest row.
"""
from __future__ import annotations

import json

#: AC-6 per kind: the P1 rows (P1T-12/25). AC6-DYN-1..3 gate P2 (P1-57 Q-DYN B): "dynamic import not met at P1".
AC6_KINDS = {"url": ["AC6-URL-1", "AC6-URL-2", "AC6-URL-3"], "example-config": ["AC6-EX-1", "AC6-EX-4"], "prompt-mention": ["AC6-PM-1", "AC6-PM-2", "AC6-PM-3"]}
AC6_FLOOR = {"url": 3, "example-config": 2, "prompt-mention": 3}


def find_item(doc: dict, repo: str, qname: str):
    """The item of ``doc`` whose subject is ``qname`` (or a function nested in it, ``<qname>.<anon>#n``) in ``repo``; None when absent."""
    for it in doc["items"]:
        s = it["subject"]
        q = s.get("qname")
        if it["repoId"] == repo and q and (q == qname or q.startswith(qname + ".<anon>#")):
            return it
    return None


def ac6_item_violations(rid: str, link_reason: str, item) -> list[str]:
    """One AC-6 instance: the planted consumer's item is listed, `possible` (never `confirmed`), carries the stated reason, and not a flow reason."""
    if item is None:
        return [f"{rid}: the planted consumer is not listed in the answer"]
    bad = []
    if item["status"] != "possible":
        bad.append(f"{rid}: item status is {item['status']!r}, expected 'possible' (AC-6: none shown as confirmed)")
    reason = item.get("reason")
    if reason != link_reason:
        bad.append(f"{rid}: item reason is {reason!r}, expected {link_reason!r}")
    if isinstance(reason, str) and reason.startswith("flow-unavailable"):
        bad.append(f"{rid}: the reason is the flow notice {reason!r}, not the planted unprovable kind")
    return bad


# ---- AC-18 ------------------------------------------------------------------------------------------------------------------------------
LOCATED = ("Symbol", "CallSite", "ConfigEntry")


def _has_loc(h: dict) -> bool:
    return isinstance(h.get("path"), str) and bool(h["path"]) and isinstance(h.get("line"), int) and not isinstance(h["line"], bool) and h["line"] >= 1


def ac18_violations(doc: dict, label: str, counts: dict | None = None) -> list[str]:
    """Every item (and every folded group member) of ``doc`` shows its full chain with file and line at each step.

    A hop that is a code element (`Symbol`, `CallSite`, `ConfigEntry`) carries `path` and `line`; a `Link` hop carries `lifted` entries each with `path` and `line` (the sites the
    link was lifted through), or, when it has none (the contract-point walk's `ON` hop, the `BOUND_BY` hop to a config entry), an adjacent code-element hop that does; a `ContractPoint` hop is a key, not a code location, and must carry its `keyNorm`.
    The chain starts at a resolved anchor of the request and ends at the item's subject. ``counts`` (when given) accumulates the hop kinds seen."""
    bad = []
    interp = doc["interpretation"]
    anchors = {a.get("symbolId") or a.get("cpId") or a.get("entryId") for a in interp.get("request", {}).get("anchors", [])}
    items = [*doc["items"], *(m for g in doc["groups"] for m in g["members"])]
    for n, it in enumerate(items):
        tag = f"{label} item {n} ({it['subject'].get('qname') or it['subject'].get('keyNorm') or it['subject'].get('type')})"
        chain = it.get("chain") or []
        if not chain:
            bad.append(f"{tag}: empty chain")
            continue
        if chain[0]["id"] not in anchors:
            bad.append(f"{tag}: the chain starts at {chain[0]['id']}, not at a request anchor {sorted(a for a in anchors if a)}")
        s = it["subject"]
        sid = s.get("symbolId") or s.get("entryId") or s.get("unitId")
        if sid and chain[-1]["id"] != sid:
            bad.append(f"{tag}: the chain ends at {chain[-1]['id']}, not at the subject {sid}")
        if s.get("type") == "symbol" and not (s.get("path") and isinstance(s.get("lineStart"), int)):
            bad.append(f"{tag}: the subject carries no path/lineStart")
        for i, h in enumerate(chain):
            kind = h["kind"]
            if counts is not None:
                counts[kind] = counts.get(kind, 0) + 1
            if kind in LOCATED:
                if not _has_loc(h):
                    bad.append(f"{tag}: hop {i} ({kind}) has no file and line: path={h.get('path')!r} line={h.get('line')!r}")
            elif kind == "Link":
                lifted = h.get("lifted") or []
                if lifted:
                    for l in lifted:
                        if not _has_loc(l):
                            bad.append(f"{tag}: hop {i} (Link) lifted entry {l.get('ord')} has no file and line")
                else:
                    near = [chain[j] for j in (i - 1, i + 1) if 0 <= j < len(chain) and chain[j]["kind"] in LOCATED]
                    if not any(_has_loc(c) for c in near):
                        bad.append(f"{tag}: hop {i} (Link) has no lifted entry and no adjacent located hop (CallSite, Symbol, ConfigEntry) with file and line")
                for b in h.get("bound") or []:
                    if "ord" not in b:
                        bad.append(f"{tag}: hop {i} (Link) has a bound entry without an ord")
            elif kind == "ContractPoint":
                if not h.get("keyNorm"):
                    bad.append(f"{tag}: hop {i} (ContractPoint) has no keyNorm")
            else:
                bad.append(f"{tag}: hop {i} has the unexpected kind {kind!r}")
            if i > 0 and kind != "ContractPoint" and not h.get("edgeType"):
                bad.append(f"{tag}: hop {i} ({kind}) names no edge")
    return bad


# ---- AC-25 ------------------------------------------------------------------------------------------------------------------------------
def ac25_shape_violations(doc: dict, label: str) -> list[str]:
    """Every item is marked direct or indirect with its distance (direct is distance 1, never beyond the stated bound); items grouped by repository and component
    (`staleness.units[].rootPath`), then by distance, confirmed before possible (the order of the document)."""
    bad = []
    roots = {u["unitId"]: u["rootPath"] for u in doc["staleness"]["units"]}
    bound = doc["depthBound"].get("bound")
    prev = None
    for it in doc["items"]:
        d = it["distance"]
        if it.get("directness") not in ("direct", "indirect"):
            bad.append(f"{label}: item {it['subject'].get('qname')} has directness {it.get('directness')!r}")
        elif (it["directness"] == "direct") != (d == 1):
            bad.append(f"{label}: item {it['subject'].get('qname')} is {it['directness']} at distance {d}")
        if not isinstance(d, int) or d < 1 or (bound is not None and d > bound):
            bad.append(f"{label}: item {it['subject'].get('qname')} distance {d!r} outside 1..{bound}")
        if it["status"] == "possible" and not it.get("reason"):
            bad.append(f"{label}: possible item {it['subject'].get('qname')} carries no reason")
        key = (it["repoId"], roots.get(it["unitId"], "?"), d, 0 if it["status"] == "confirmed" else 1)
        if prev is not None and (key[:2], key[2:]) < (prev[:2], prev[2:]):
            bad.append(f"{label}: item {it['subject'].get('qname')} {key} is out of order after {prev} (repository, component, distance, confirmed before possible)")
        prev = key
    return bad


# ---- AC-16 raw leg ----------------------------------------------------------------------------------------------------------------------
def cypher_with_params(cypher: str, params: dict) -> str:
    """The recipe text, unchanged, behind FalkorDB's ``CYPHER k=v`` parameter prefix."""
    head = " ".join(f"{k}={json.dumps(v)}" for k, v in sorted(params.items()))
    return f"CYPHER {head} {cypher}"


#: (hop kind, edge, dir) -> the recipe read that walks that edge, with the hop's previous element as ``$ids``; the row's column 0 is that id and ``col`` holds the hop's id,
#: ``stat`` the column holding the edge status (None: the read has none, so the hop's status is not compared)
HOP_READS = {
    ("CallSite", "CALLS", "<-"): ("calls_in", 1, 7),
    ("CallSite", "CALLS_NAME", "<-"): ("name_in", 1, None),
    ("Symbol", "HAS_CALL", "<-"): ("owner_of:CallSite", 1, None),
    ("CallSite", "HAS_CALL", "->"): ("has_out:CallSite", 1, None),
    ("Symbol", "CALLS", "->"): ("calls_out", 1, 7),
    ("Symbol", "CALLS_NAME", "->"): ("name_out", 1, None),
    ("Link", "TO", "<-"): ("to_in", 1, 3),
    ("Symbol", "TO", "->"): ("to_out", 1, None),
    ("CallSite", "VIA", "<-"): ("via_in", 1, 7),
    ("Link", "VIA", "->"): ("via_out", 1, 6),
    ("Link", "ON", "<-"): ("on_in", 1, 3),
    ("ConfigEntry", "BOUND_BY", "->"): ("bound_of_link", 1, None),
}
