#!/usr/bin/env python3
"""split_manifest.py — the AC-5 split, the held-out manifest and the report leak check (S49 second half, P0-42b).

Spec: eval-protocol ``docs/plans/connected-code-graph-eval-protocol-ml.md`` §9.1-§9.4, §12.1, §13.6 (v1.5) and
EP2 §1.4; the P1 verbs (P1-32a) are specified in ``docs/plans/connected-code-graph-p1-gates.md`` P1-32 and
``docs/plans/connected-code-graph-p1-ml.md`` §6.2/§6.4.

Verbs (counts, ids and paths only on stdout; never item content):
  split         --match F --context C --contact F --eval-commit SHA --heldout-dir D --dev-dir D [--report F] [--log F]
                  §9.1: clusters (B1), contact, share, step 3b (served anchors), step 4 (minimums, eligibility
                  guard), step 5 (trigger, H and H_all), step 6 (reserve). Writes ``ac5-<stratum>.jsonl`` x5,
                  ``ac5-reserve.jsonl`` and an empty ``exposure.jsonl`` into the held-out dir, ``ac5-<stratum>.jsonl``
                  into the dev dir. Exit 0 even when the trigger fires (``trigger: true`` is printed and recorded).
  manifest      --heldout-dir D --eval-commit SHA --protocol STR --contact F --out F [--raw-root D --raw REL ...]   §9.3
  verify        --manifest F --heldout-dir D [--raw-root D]       (alias ``--verify F``)                           §9.3
  contact       --in RAW --out work/contact.jsonl   raw {locator, source} lines -> {anchorId, locator, source}      §8.2
  check-report  --report F (--heldout-dir D | --heldout F...) [--dev F...] [--contact F]  (alias ``--check-report F``)  §9.4
  regate-list   --match F --context C --contact F --eval-commit SHA --manifest F [--missed F] [--fresh F] [--promote-reserve]
                --heldout-dir D --dev-dir D [--report F]                                                          §12.1 3-5
                  the next list: kept clusters + fresh items. Re-derives gate 1's split (its lists must hash to the held-out
                  ``--manifest``, else exit 1 and nothing is written), moves the clusters of ``--missed``
                  (jsonl ``{anchorId}``) to dev, re-applies the served-anchor rule, discards a fresh item already in
                  dev/held-out/reserve, routes the rest (dev if its cluster or S(u) holds an anchor in dev or C; joins a
                  kept cluster, not counted fresh; else a fresh cluster). Writes ``ac5-*.jsonl``, ``ac5-reserve.jsonl``,
                  ``regate-flags.jsonl`` (itemId -> kept / kept-cluster-fresh / fresh-cluster) into a NEW held-out dir
                  (an existing ac5 list is never overwritten) and the whole dev list into the dev dir. Reports counts,
                  the E12 fresh-counting figure against ``FRESH_FLOOR`` (35) and the composition shortfall; the pass / "all
                  hits, floor not met" decision is the harness's (P1-32b).
  exposure      --exposure F --gate N --results F       appends ``{itemId, firstGate, hit}`` for items not yet on record  §12.1 6
  apply-corrections --match-scoped F --marks-template F --verify F --out-match F --corrections F                    §7.4
  ambiguity     --questions D --aid-map F --out F --log F      the §10.3 join (QW/AN1/AN2) -> kept questions, gold normalized
  split-ac24    --in F --eval-commit SHA --heldout-dir D --dev-dir D    §10.2: 6 per (group, lang) cell held out, ambiguous
                  by language (lowest rank dev); the three scripted-step verbs replace ``s47_scripted_steps.py``
  freeze-check  --record SHA [--s47 SHA] [--repo D] --heldout F... [--dev F...] [--contact F] [--out F] [--probe-map F]
                [--positive-control]                                                                                -p1-ml.md 6.2
                  ``check-report`` once per file of ``git diff --name-only <S47> <record>`` minus the recorded exclusions
                  (custodian outputs; paths whose commits are all non-P1). Exit 0 clean, 1 hits, 2 / 4 a failure (never a
                  skip). Hits are 12-hex probe hashes; ``--probe-map`` writes the private hash -> itemId map (custodian root).
  oracle-candidates --in edges.jsonl --out F      observed edges ``{consumer, key, lines?}`` -> key-item records for ``--heldout``

Inputs: ``--match`` is match_keys' ``scope`` output (``counting``, ``inScope`` per record); ``--context`` is
``{"units": [{"rootPath","lang"}]}``; ``--contact`` is ``work/contact.jsonl`` as the ``contact`` verb writes it (the only accepted shape). Exit: 0 ok, 1 findings (a leak, a
manifest mismatch), 2 usage, 4 missing or invalid input.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import match_keys as _mk                                    # noqa: E402

STRATA = _mk.STRATA                                         # http, mcp, scripts, wiring, flow: the tie order
SEED_PREFIX = "ccg-p0f-split-v1|"                           # EP §9.1
SHARE_NUM, SHARE_DEN = 7, 10                                # share 0.70, exact integer arithmetic
MIN_STRATUM_ITEMS = 8                                       # per-stratum gate-1 minimum (all items)
MIN_COUNTING = 75                                           # AC-5's 75, tested on counting items H
RESERVE_TARGET = 35                                         # E12
MIN_BARE_PROBE_LEN = 8                                      # EP §9.4 v1.5: a BARE probe shorter than this is dropped
MANIFEST_SCHEMA = "ccg-heldout-manifest-v1"
canon, sha = _mk.canon, _mk.sha


# =============================================================================================
# split (EP §9.1)
# =============================================================================================

def _ceil_share(n: int) -> int:
    return -(-n * SHARE_NUM // SHARE_DEN)


def rank_of(seed: str, anchor_id: str) -> str:
    return sha(f"{seed}|{anchor_id}")


def served_sets(records: list[dict], units: list[dict]) -> dict[str, set]:
    """S(u) per unit anchorId (EP §3.5): contract anchors whose member ``provider.path`` falls in u (longest
    prefix), plus every contract locator in the ``serves[]`` of a record anchored at u."""
    unit_ids = {}
    for u in units:
        loc = {"type": "unit", "repoId": _mk.REPO_ID, "rootPath": u["rootPath"], "lang": u["lang"]}
        unit_ids[(u["rootPath"], u["lang"])] = _mk.locator_id(loc)
    S: dict[str, set] = {uid: set() for uid in unit_ids.values()}
    for g in records:
        for m in g["members"]:
            a = m["anchor"]
            if a["type"] == "contract":
                p = (m.get("provider") or {}).get("path")
                if p:
                    u = _mk.unit_of(p, units)
                    if u is not None:
                        S[unit_ids[(u["rootPath"], u["lang"])]].add(g["anchorId"])
            elif a["type"] == "unit":
                S.setdefault(g["anchorId"], set())
                for s in m.get("serves") or []:
                    S[g["anchorId"]].add(_mk.locator_id(s))
    return S


def _served_in_dev(ctx: dict, c: dict) -> bool:
    """Any anchor of S(u), for a unit u in cluster ``c``, that sits in ANOTHER cluster that is in dev."""
    for u in c["units"]:
        for a in ctx["S"][u]:
            if a in ctx["C"] and a not in c["anchors"]:          # contacted: public, even with no item/cluster (EP §8.2)
                return True
            k = ctx["cluster_of"].get(a)
            if k is not None and k != c["id"] and ctx["side"][k] == "dev":
                return True
    return False


def split(records: list[dict], units: list[dict], contact_ids, eval_commit: str) -> dict:
    """EP §9.1 over the in-scope union. -> dict with ``side`` per cluster, report counts and the move log."""
    for g in records:
        if "counting" not in g:
            raise ValueError("record without 'counting': run match_keys scope first")
    for g in records:
        if not isinstance(g.get("inScope"), bool):
            raise ValueError("record without a boolean 'inScope': run match_keys scope first (EP §9.1)")
    records = [g for g in records if g["inScope"]]
    seed = SEED_PREFIX + eval_commit
    C = set(contact_ids)
    by_anchor: dict[str, list] = {}
    for g in records:
        by_anchor.setdefault(g["anchorId"], []).append(g)
    S = served_sets(records, units)
    items_in = {a: len(v) for a, v in by_anchor.items()}

    # probe per unit anchor present in U: fewest items in U, ties by smallest cpId (hex order)
    probe = {}
    for uid, served in S.items():
        if uid in by_anchor and served:
            probe[uid] = min(served, key=lambda a: (items_in.get(a, 0), a))

    # step 1: union-find
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in by_anchor:
        find(a)
    for uid, p in probe.items():
        parent[find(uid)] = find(p)
    groups: dict[str, set] = {}
    for a in list(parent):
        groups.setdefault(find(a), set()).add(a)
    clusters = []
    for anchors in groups.values():
        items = [g for a in anchors for g in by_anchor.get(a, ())]
        if not items:
            continue
        cnt: dict[str, int] = {}
        for g in items:
            cnt[g["derivedStratum"]] = cnt.get(g["derivedStratum"], 0) + 1
        primary = min(STRATA, key=lambda s: (-cnt.get(s, 0), STRATA.index(s)))
        rk = min(rank_of(seed, a) for a in anchors)
        clusters.append({"anchors": anchors, "items": items, "rank": rk, "primary": primary,
                         "key": min(anchors), "units": sorted(a for a in anchors if a in S)})
    clusters.sort(key=lambda c: (c["rank"], c["key"]))
    for i, c in enumerate(clusters):
        c["id"] = i
    cluster_of = {a: c["id"] for c in clusters for a in c["anchors"]}
    side = {c["id"]: "held" for c in clusters}
    log: list = []

    def to_dev(c, rule):
        side[c["id"]] = "dev"
        log.append({"rule": rule, "cluster": c["key"], "stratum": c["primary"]})

    # step 2: contact
    contacted = {c["id"] for c in clusters if c["anchors"] & C}
    for c in clusters:
        if c["id"] in contacted:
            to_dev(c, "contact")

    # step 3: share
    for s in STRATA:
        A = [c for c in clusters if c["primary"] == s and side[c["id"]] != "dev"]
        T = _ceil_share(sum(len(c["items"]) for c in A))
        held = 0
        for c in A:
            if held >= T:
                side[c["id"]] = "dev"
            else:
                held += len(c["items"])

    ctx = {"S": S, "cluster_of": cluster_of, "side": side, "C": C}

    def served_in_dev(c):
        return _served_in_dev(ctx, c)

    # step 3b
    served_moves = 0
    changed = True
    while changed:
        changed = False
        for c in clusters:
            if side[c["id"]] == "held" and served_in_dev(c):
                to_dev(c, "served-anchor")
                served_moves += 1
                changed = True

    def counts():
        Hs = {s: 0 for s in STRATA}
        H = 0
        for c in clusters:
            if side[c["id"]] == "held":
                for g in c["items"]:
                    Hs[g["derivedStratum"]] += 1
                    H += bool(g["counting"])
        return Hs, H

    def eligible(c):
        return side[c["id"]] == "dev" and not (c["anchors"] & C) and not served_in_dev(c)

    # step 4
    marked: list = []
    while True:
        Hs, H = counts()
        short = [s for s in STRATA if s not in marked and Hs[s] < MIN_STRATUM_ITEMS]
        if short:
            s = min(short, key=lambda x: (Hs[x], STRATA.index(x)))
            cand = [c for c in clusters if eligible(c) and any(g["derivedStratum"] == s for g in c["items"])]
            if cand:
                side[cand[0]["id"]] = "held"
                log.append({"rule": "minimum-stratum", "cluster": cand[0]["key"], "stratum": s})
            else:
                marked.append(s)
        elif H < MIN_COUNTING:
            cand = [c for c in clusters if eligible(c) and any(g["counting"] for g in c["items"])]
            if not cand:
                break
            side[cand[0]["id"]] = "held"
            log.append({"rule": "minimum-counting", "cluster": cand[0]["key"], "stratum": cand[0]["primary"]})
        else:
            break

    # step 5 / 6
    Hs, H = counts()
    H_all = sum(Hs.values())
    trigger = H < MIN_COUNTING or bool(marked)
    reserve_counting = 0
    if not trigger:
        for c in sorted((c for c in clusters if side[c["id"]] == "held"), key=lambda c: (c["rank"], c["key"]), reverse=True):
            if reserve_counting >= RESERVE_TARGET:
                break
            n_count = sum(bool(g["counting"]) for g in c["items"])
            if n_count < 1:
                continue
            side[c["id"]] = "reserve"
            Hs2, H2 = counts()
            if H2 >= MIN_COUNTING and all(Hs2[s] >= MIN_STRATUM_ITEMS for s in STRATA):
                reserve_counting += n_count
                log.append({"rule": "reserve", "cluster": c["key"], "stratum": c["primary"]})
            else:
                side[c["id"]] = "held"
    Hs, H = counts()
    held = [c for c in clusters if side[c["id"]] == "held"]
    report = {
        "H": H, "H_all": sum(Hs.values()), "Hs": Hs, "markedStrata": marked, "trigger": trigger,
        "servedAnchorMoves": served_moves, "reserveCounting": reserve_counting,
        "reserveClusters": sum(1 for c in clusters if side[c["id"]] == "reserve"),
        "clusters": len(clusters),
        "heldClustersPerStratum": {s: sum(1 for c in held if c["primary"] == s) for s in STRATA},
        "devClusters": {"contacted": sum(1 for c in clusters if side[c["id"]] == "dev" and c["id"] in contacted),
                        "other": sum(1 for c in clusters if side[c["id"]] == "dev" and c["id"] not in contacted)},
        "contactAnchors": len(C),
    }
    return {"clusters": clusters, "side": side, "report": report, "log": log, "seed": seed,
            "S": S, "cluster_of": cluster_of, "C": C, "probe": probe}


def regate_served(res: dict, missed_anchors) -> int:
    """EP §12.1 steps 3-3b (library; the ``regate-list`` verb is P1): every cluster holding a missed anchor moves
    to dev, then the served-anchor rule is re-applied to every kept (held) AND reserve unit cluster. -> number of
    served-anchor moves (not misses)."""
    side = res["side"]
    for a in missed_anchors:
        k = res["cluster_of"].get(a)
        if k is not None:
            side[k] = "dev"
    moved, changed = 0, True
    while changed:
        changed = False
        for c in res["clusters"]:
            if side[c["id"]] in ("held", "reserve") and _served_in_dev(res, c):
                side[c["id"]] = "dev"
                moved += 1
                changed = True
    return moved


def route_fresh(res: dict, fresh: list[dict]) -> list[dict]:
    """EP §12.1 step 4 routing, in this order. ``dev``: the anchor is in C, or in a dev cluster, or the item sits on
    a unit whose served set S(u) holds an anchor in C or in a dev cluster (v1.2, P2-1). ``kept-cluster-fresh``: the
    anchor is in a kept (held) cluster, so the item joins it and does not count as fresh. Otherwise ``fresh-cluster``
    (an anchor with no cluster, or in the never-exposed reserve). -> [{itemId, anchorId, flag}]."""
    side, S, C = res["side"], res["S"], res["C"]

    def exposed(x):
        k = res["cluster_of"].get(x)
        return x in C or (k is not None and side[k] == "dev")

    out = []
    for g in fresh:
        a = g["anchorId"]
        k = res["cluster_of"].get(a)
        if exposed(a) or any(exposed(x) for x in S.get(a, ())):
            flag = "dev"
        elif k is not None and side[k] == "held":
            flag = "kept-cluster-fresh"
        else:
            flag = "fresh-cluster"
        out.append({"itemId": g["itemId"], "anchorId": a, "flag": flag})
    return out


FRESH_FLOOR = 35                                            # E12: counting items on fresh clusters a re-gate needs


class Gate1Mismatch(Exception):
    """The re-derived gate-1 state is not the one the held-out manifest recorded (a different contact, commit or match)."""


def gate1_problems(res: dict, manifest: dict, eval_commit: str, n_contact: int) -> list[str]:
    """Compare the gate-1 lists re-derived by ``split`` (hashed in memory, same bytes ``write_split`` writes) with the
    manifest's ``files`` (the five ``ac5-<s>.jsonl`` and ``ac5-reserve.jsonl``; ``ac24.jsonl`` is not a split product),
    and its ``evalCommit``, ``splitSeed`` and ``contactAnchors``. -> field names only."""
    bad = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        bad.append("schema")
    if manifest.get("evalCommit") != eval_commit:
        bad.append("evalCommit")
    if manifest.get("splitSeed") != SEED_PREFIX + eval_commit:
        bad.append("splitSeed")
    if manifest.get("contactAnchors") != n_contact:
        bad.append("contactAnchors")
    listed = {f.get("name"): f for f in manifest.get("files") or []}
    for name, data in split_files(res)[0].items():
        f = listed.get(name)
        if f is None:
            bad.append(f"{name}: not listed")
            continue
        now = entry_of(name, data)
        bad += [f"{name}: {k}" for k in ("sha256", "bytes", "items") if now[k] != f.get(k)]
    return bad


def regate_list(records: list[dict], units: list[dict], contact_ids, eval_commit: str, missed, fresh: list[dict],
                promote_reserve: bool = False, manifest: dict | None = None) -> dict:
    """EP §12.1 steps 3-4: the next gate's list. Re-derives gate 1's state with ``split``, moves every cluster holding
    a ``missed`` anchor to dev and re-applies the served-anchor rule (``regate_served``), discards a fresh item already
    in dev, held-out or the reserve (or repeated in ``fresh``), routes the rest with ``route_fresh``, and optionally
    promotes the surviving reserve clusters as fresh-cluster items. -> {res, held, dev, reserve, flags, report}; ``held``
    is the new list (kept clusters, joined items, fresh-cluster items), ``flags`` maps its itemIds to ``kept`` /
    ``kept-cluster-fresh`` / ``fresh-cluster``. With a ``manifest`` the re-derived gate-1 lists must hash to its ``files``,
    else ``Gate1Mismatch`` (before anything is written)."""
    for g in fresh:
        if "counting" not in g or not isinstance(g.get("inScope"), bool):
            raise ValueError("fresh record without 'counting' and a boolean 'inScope': run match_keys scope first")
    res = split(records, units, contact_ids, eval_commit)
    if manifest is not None:
        bad = gate1_problems(res, manifest, eval_commit, len(set(contact_ids)))
        if bad:
            raise Gate1Mismatch(bad)
    side, clusters = res["side"], res["clusters"]
    moved_out = {s: 0 for s in STRATA}
    miss_clusters = set()
    for a in sorted(set(missed)):
        k = res["cluster_of"].get(a)
        if k is None or side[k] != "held":
            raise ValueError("a missed anchor is not in a kept (held-out) cluster of the gate list")
        miss_clusters.add(k)
    for k in miss_clusters:
        for g in clusters[k]["items"]:
            moved_out[g["derivedStratum"]] += 1
    served_moves = regate_served(res, missed)
    known = {g["itemId"] for c in clusters for g in c["items"]}
    out_of_scope = sum(1 for g in fresh if not g["inScope"])
    discarded, new_fresh = 0, []
    for g in fresh:
        if not g["inScope"]:
            continue
        if g["itemId"] in known:
            discarded += 1
            continue
        known.add(g["itemId"])
        new_fresh.append(g)
    res["S"] = served_sets([g for c in clusters for g in c["items"]] + new_fresh, units)    # a fresh unit item has its S(u)
    flag_of = {r["itemId"]: r["flag"] for r in route_fresh(res, new_fresh)}
    held, dev, reserve, flags = [], [], [], {}
    for c in clusters:
        for g in c["items"]:
            sd = side[c["id"]]
            if sd == "held":
                held.append(g)
                flags[g["itemId"]] = "kept"
            elif sd == "reserve" and promote_reserve:
                held.append(g)
                flags[g["itemId"]] = "fresh-cluster"
            elif sd == "reserve":
                reserve.append(g)
            else:
                dev.append(g)
    for g in new_fresh:
        f = flag_of[g["itemId"]]
        if f == "dev":
            dev.append(g)
        else:
            held.append(g)
            flags[g["itemId"]] = f
    Hs = {s: sum(1 for g in held if g["derivedStratum"] == s) for s in STRATA}
    H = sum(1 for g in held if g["counting"])
    fresh_items = [g for g in held if flags[g["itemId"]] == "fresh-cluster"]
    fresh_by = {s: sum(1 for g in fresh_items if g["derivedStratum"] == s) for s in STRATA}
    fresh_counting = sum(1 for g in fresh_items if g["counting"])
    report = {
        "H": H, "Hs": Hs, "kept": sum(1 for f in flags.values() if f == "kept"),
        "keptClusterFresh": sum(1 for f in flags.values() if f == "kept-cluster-fresh"),
        "freshCluster": len(fresh_items), "freshCounting": fresh_counting, "freshByStratum": fresh_by,
        "floor": FRESH_FLOOR, "floorMet": fresh_counting >= FRESH_FLOOR,
        "discarded": discarded, "outOfScope": out_of_scope,
        "freshToDev": sum(1 for f in flag_of.values() if f == "dev"),
        "missedClusters": len(miss_clusters), "movedOutByStratum": moved_out,
        "shortfallByStratum": {s: max(0, moved_out[s] - fresh_by[s]) for s in STRATA if moved_out[s]},
        "minimumsMet": H >= MIN_COUNTING and all(v >= MIN_STRATUM_ITEMS for v in Hs.values()),
        "servedAnchorMoves": served_moves, "reserveLeft": len(reserve),
        "reservePromoted": sum(len(c["items"]) for c in clusters if side[c["id"]] == "reserve") if promote_reserve else 0,
    }
    return {"res": res, "held": held, "dev": dev, "reserve": reserve, "flags": flags, "report": report}


def write_regate(out: dict, heldout_dir: Path, dev_dir: Path) -> None:
    """``ac5-<stratum>.jsonl`` x5 + ``ac5-reserve.jsonl`` + ``regate-flags.jsonl`` into the held-out dir (refused if it
    already holds an ``ac5-*`` file: a re-gate is a NEW list, EP §12.1 step 7), and the whole dev list into the dev dir."""
    if any(heldout_dir.glob("ac5-*.jsonl")):
        raise ValueError("the held-out directory already holds an ac5 list; a re-gate never overwrites a list")
    for s in STRATA:
        write_jsonl(heldout_dir / f"ac5-{s}.jsonl", [out_record(g) for g in out["held"] if g["derivedStratum"] == s])
        write_jsonl(dev_dir / f"ac5-{s}.jsonl", [out_record(g) for g in out["dev"] if g["derivedStratum"] == s])
    write_jsonl(heldout_dir / "ac5-reserve.jsonl", [out_record(g) for g in out["reserve"]])
    write_jsonl(heldout_dir / "regate-flags.jsonl", [{"itemId": i, "flag": f} for i, f in out["flags"].items()])


def out_record(g: dict) -> dict:
    m0 = g["members"][0]
    return {"itemId": g["itemId"], "anchorId": g["anchorId"], "impactedId": g["impactedId"],
            "derivedStratum": g["derivedStratum"], "counting": bool(g["counting"]),
            "anchor": m0["anchor"], "impacted": m0["impacted"], "provider": m0.get("provider"),
            "serves": m0.get("serves"), "lines": sorted({l for m in g["members"] for l in m.get("lines") or []}),
            "chain": m0.get("chain"), "writers": g["sessions"], "originals": [m["original"] for m in g["members"]]}


def _jsonl_bytes(rows) -> bytes:
    rows = sorted(rows, key=lambda r: r["itemId"])
    return "".join(canon(r) + "\n" for r in rows).encode("utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_jsonl_bytes(rows))


def split_files(res: dict):
    """The files ``write_split`` writes, as bytes in memory: ``(held, dev)`` = {name: bytes} (held-out: ``ac5-<s>.jsonl`` x5 +
    ``ac5-reserve.jsonl``). Nothing touches the disk, so a check can hash the gate-1 lists without writing held-out data."""
    held = {s: [] for s in STRATA}
    dev = {s: [] for s in STRATA}
    reserve = []
    for c in res["clusters"]:
        for g in c["items"]:
            s = res["side"][c["id"]]
            r = out_record(g)
            if s == "held":
                held[g["derivedStratum"]].append(r)
            elif s == "dev":
                dev[g["derivedStratum"]].append(r)
            else:
                reserve.append(r)
    h = {f"ac5-{s}.jsonl": _jsonl_bytes(held[s]) for s in STRATA}
    h["ac5-reserve.jsonl"] = _jsonl_bytes(reserve)
    return h, {f"ac5-{s}.jsonl": _jsonl_bytes(dev[s]) for s in STRATA}


def write_split(res: dict, heldout_dir: Path, dev_dir: Path) -> None:
    held, dev = split_files(res)
    for n, b in held.items():
        (heldout_dir / n).parent.mkdir(parents=True, exist_ok=True)
        (heldout_dir / n).write_bytes(b)
    for n, b in dev.items():
        (dev_dir / n).parent.mkdir(parents=True, exist_ok=True)
        (dev_dir / n).write_bytes(b)
    (heldout_dir / "exposure.jsonl").write_text("", encoding="utf-8")


# =============================================================================================
# manifest and --verify (EP §9.3)
# =============================================================================================

def file_entry(path: Path) -> dict:
    return entry_of(path.name, path.read_bytes())


def entry_of(name: str, data: bytes) -> dict:
    """Hash, byte size and line count first (a corrupted file must still be measurable); per-line fields best effort."""
    lines = [l for l in data.split(b"\n") if l.strip()]
    rows = []
    for l in lines:
        try:
            rows.append(json.loads(l.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            rows.append({})
    return {"name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "items": len(lines), "countingItems": sum(1 for r in rows if isinstance(r, dict) and r.get("counting")),
            "anchors": len({r.get("anchorId") for r in rows if isinstance(r, dict)})}


def listed_files(heldout_dir: Path) -> list[Path]:
    names = [f"ac5-{s}.jsonl" for s in STRATA] + ["ac5-reserve.jsonl"]
    if (heldout_dir / "ac24.jsonl").exists():
        names.append("ac24.jsonl")
    return [heldout_dir / n for n in names]


def build_manifest(heldout_dir: Path, eval_commit: str, protocol: str, contact_anchors: int,
                   raw_root: Path | None = None, raw: list[str] = ()) -> dict:
    return {
        "schema": MANIFEST_SCHEMA, "evalCommit": eval_commit, "protocol": protocol,
        "splitSeed": SEED_PREFIX + eval_commit,
        "files": [file_entry(p) for p in listed_files(heldout_dir)],
        "contactAnchors": contact_anchors,
        "rawArtifacts": [{"path": r, "sha256": sha_file(Path(raw_root) / r)} for r in raw],
    }


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_manifest(manifest: dict, heldout_dir: Path, raw_root: Path | None = None) -> list[str]:
    """-> list of problems (names and field names only)."""
    bad = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        bad.append("schema")
    if manifest.get("splitSeed") != SEED_PREFIX + str(manifest.get("evalCommit")):
        bad.append("splitSeed")
    listed = {f.get("name") for f in manifest.get("files") or []}
    required = [f"ac5-{s}.jsonl" for s in STRATA] + ["ac5-reserve.jsonl"]
    if (heldout_dir / "ac24.jsonl").exists():
        required.append("ac24.jsonl")
    for n in required:
        if n not in listed:
            bad.append(f"{n}: not listed")
    for f in manifest.get("files") or []:
        p = heldout_dir / f["name"]
        if not p.is_file():
            bad.append(f"{f['name']}: missing")
            continue
        now = file_entry(p)
        for k in ("sha256", "bytes", "items"):
            if now[k] != f.get(k):
                bad.append(f"{f['name']}: {k}")
    for r in manifest.get("rawArtifacts") or []:
        p = Path(raw_root if raw_root is not None else heldout_dir.parent) / r["path"]
        if not p.is_file():
            bad.append(f"{r['path']}: missing")
        elif sha_file(p) != r.get("sha256"):
            bad.append(f"{r['path']}: sha256")
    return bad


# =============================================================================================
# check-report (EP §9.4 v1.5, EP2 §1.4)
# =============================================================================================

def check_report(report: str, heldout: list[dict], dev: list[dict], contact: list[dict]):
    """-> ``(hits, n_probes)``; hits are probe hashes only. ``match_keys.leak_check`` does the work, with the floor
    (EP §9.4 v1.5: only a BARE probe shorter than this is dropped) set by this module's ``MIN_BARE_PROBE_LEN``."""
    return _mk.leak_check(report, heldout, dev, contact, bare_floor=MIN_BARE_PROBE_LEN)


# =============================================================================================
# contact (EP §8.2 "The lists")
# =============================================================================================

def normalize_contact_locator(loc: dict) -> dict:
    """EP2 §1.2 ``normalize_contract`` (contract; optional ``form`` key, default provider) and EP §2.3 ``E`` (symbol);
    paths repo-relative POSIX; ``repoId`` defaulted."""
    loc = _mk.nfc_strip(dict(loc))
    form = loc.pop("form", "provider")
    t = loc.get("type")
    if t in ("symbol", "unit"):
        loc.setdefault("repoId", _mk.REPO_ID)
    if isinstance(loc.get("path"), str):
        loc["path"] = _mk._norm_path(loc["path"])
    if t == "contract":
        loc = _mk.normalize_contract(loc, form)
    elif t == "symbol":
        loc["qname"] = _mk.E(loc["qname"])
    elif t != "unit":
        raise ValueError(f"locator type {t!r} is not an anchor type (contract, symbol, unit): EP §8.2")
    return loc


def build_contact(rows: list[dict]) -> list[dict]:
    """-> [{anchorId, locator, source}] sorted by anchorId, deduplicated on (anchorId, source)."""
    out: dict = {}
    for r in rows:
        loc = normalize_contact_locator(r["locator"])
        rec = {"anchorId": _mk.locator_id(loc), "locator": loc, "source": r.get("source")}
        out[(rec["anchorId"], canon(rec["source"]))] = rec
    return [out[k] for k in sorted(out)]


def read_contact(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    for r in rows:
        if not {"anchorId", "locator", "source"} <= set(r):
            raise ValueError("contact file line without anchorId/locator/source: produce it with the 'contact' verb")
    return rows


def heldout_files(dir_: Path) -> list[Path]:
    return sorted(dir_.glob("ac5-*.jsonl")) + ([dir_ / "ac24.jsonl"] if (dir_ / "ac24.jsonl").exists() else [])


def check_heldout_list(paths: list[Path]) -> str | None:
    """EP §9.4 'Files': an explicit list omitting any of the six ac5 files, or an existing ac24.jsonl, is refused."""
    names = {Path(p).name for p in paths}
    if not names:
        return "holds no ac5-*.jsonl"
    need = [f"ac5-{s}.jsonl" for s in STRATA] + ["ac5-reserve.jsonl"]
    for p in paths:
        d = Path(p).parent
        if (d / "ac24.jsonl").exists():
            need.append("ac24.jsonl")
        break
    miss = [n for n in need if n not in names]
    return f"held-out file list omits {miss}" if miss else None


# =============================================================================================
# the S47 scripted steps as verbs (EP §7.4 correction, §10.3 join, §10.2 AC-24 split; P1-32a)
# =============================================================================================
# These replace ``s47_scripted_steps.py``, kept as the replay record the tests compare against byte for byte.

AC24_SEED_PREFIX = "ccg-p0f-ac24-v1|"                       # EP §10.2
AC24_HELD_PER_CELL = 6                                      # per (group, lang) cell of unambiguous questions
AC24_AMBIGUOUS_DEV_PER_LANG = 1                             # the lowest-rank ambiguous question of a language goes to dev
CORRECTION_REASON = "EP7.4: incorrect, note gives no locator"


def _jl(p: Path) -> list[dict]:
    return _read_jsonl(p)


def _dump_jsonl_plain(p: Path, rows) -> None:
    """Canonical lines in the given order (the scripted steps' own writer; ``write_jsonl`` sorts by itemId)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(canon(x) + "\n" for x in rows), encoding="utf-8")


def apply_corrections(match: dict, template: list[dict], verify: list[dict]):
    """EP §7.4: an ``incorrect`` UNION item whose note gives no locator is removed from the match file. -> (match, log)
    where ``log`` has one row per removed item. A verdict whose ``n`` has no template row is an error (KeyError)."""
    tmpl = {x["n"]: x for x in template}
    bad = [v["n"] for v in verify if v["verdict"] == "incorrect" and tmpl[v["n"]]["kind"] == "union"]
    m = json.loads(json.dumps(match))
    log = []
    for n in bad:
        iid = tmpl[n]["itemId"]
        m["records"] = [r for r in m["records"] if r["itemId"] != iid]
        log.append({"n": n, "itemId": iid, "action": "removed", "reason": CORRECTION_REASON, "verifier": "analyst (AI)"})
    return m, log


def ambiguity_join(qw: list[dict], amap: list[dict], labels: dict):
    """EP §10.3: join the two annotators' labels (``labels``: writer -> rows with ``aid``/``label``) with QW's intent.
    -> (final, outcomes, rewrites, stats): ``final`` is [(question, outcome)] for the kept questions, ``outcomes`` a
    Counter in first-seen order, ``rewrites`` the gold-form rewrites by (lang, group)."""
    aid_to_qid = {r["aid"]: r["qid"] for r in amap}
    lab: dict = {}
    for w in ("AN1", "AN2"):
        for r in labels[w]:
            lab.setdefault(r["aid"], {})[w] = r["label"]
    byq = {aid_to_qid[k]: v for k, v in lab.items()}
    out, agree, final = collections.Counter(), 0, []
    for q in qw:
        a1, a2 = byq[q["qid"]]["AN1"], byq[q["qid"]]["AN2"]
        agree += a1 == a2
        if a1 != a2:
            res = "drop-disagree"
        elif a1 == "ambiguous" and q["intent"] == "ambiguous":
            res = "ambiguity"
        elif a1 == "unambiguous" and q["intent"] == "unambiguous":
            res = "unambiguous"
        elif a1 == "ambiguous":
            res = "drop-writer-unamb-ann-amb"
        else:
            res = "drop-writer-amb-ann-unamb"
        out[res] += 1
        if res in ("ambiguity", "unambiguous"):
            final.append((q, res))
    n = len(qw)
    p1 = sum(byq[q["qid"]]["AN1"] == "ambiguous" for q in qw) / n
    p2 = sum(byq[q["qid"]]["AN2"] == "ambiguous" for q in qw) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = round((agree / n - pe) / (1 - pe), 4) if pe < 1 else None
    rw: collections.Counter = collections.Counter()

    def normgold(g, q):
        g = json.loads(json.dumps(g))
        for an in g["anchors"]:
            if an["type"] == "contract":
                nn = _mk.normalize_contract(dict(an), "provider")
                if nn != an:
                    rw[(q["lang"], q["group"])] += 1
                an.clear()
                an.update(nn)
        return g

    recs = []
    for q, res in final:
        r = {"qid": q["qid"], "lang": q["lang"], "group": q["group"], "text": q["text"], "outcome": res,
             "original": {"gold": q["gold"], "goldOptions": q["goldOptions"], "ambiguityKind": q["ambiguityKind"]}}
        if res == "unambiguous":
            r["gold"] = normgold(q["gold"], q)
            r["goldForm"] = "provider"
        else:
            r["ambiguityKind"] = q["ambiguityKind"]
            r["goldOptions"] = [normgold(o, q) for o in q["goldOptions"]]
        recs.append(r)
    return recs, out, rw, {"agree": agree, "n": n, "kappa": kappa}


def split_ac24(recs: list[dict], eval_commit: str):
    """EP §10.2: rank by ``sha(seed|qid)``; per (group, lang) cell of unambiguous questions the first 6 by rank are
    held out and the rest dev; per language the first ambiguous question by rank is dev and the rest held out.
    -> (held, dev), each sorted by qid. Input records are not modified."""
    seed = AC24_SEED_PREFIX + eval_commit
    recs = [dict(r, rank=_mk.sha(seed + "|" + r["qid"])) for r in recs]
    held, dev = [], []
    cells: dict = collections.defaultdict(list)
    for r in recs:
        if r["outcome"] == "unambiguous":
            cells[(r["group"], r["lang"])].append(r)
    for v in cells.values():
        v.sort(key=lambda r: r["rank"])
        held += v[:AC24_HELD_PER_CELL]
        dev += v[AC24_HELD_PER_CELL:]
    amb: dict = collections.defaultdict(list)
    for r in recs:
        if r["outcome"] == "ambiguity":
            amb[r["lang"]].append(r)
    for v in amb.values():
        v.sort(key=lambda r: r["rank"])
        dev += v[:AC24_AMBIGUOUS_DEV_PER_LANG]
        held += v[AC24_AMBIGUOUS_DEV_PER_LANG:]
    return sorted(held, key=lambda x: x["qid"]), sorted(dev, key=lambda x: x["qid"])


# =============================================================================================
# exposure (EP §9.2 ``exposure.jsonl``, §12.1 step 6; P1-32a)
# =============================================================================================

def _exposure_rows(rows: list[dict], what: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        iid, gate, hit = r.get("itemId"), r.get("firstGate"), r.get("hit")
        if not (isinstance(iid, str) and iid and type(gate) is int and gate >= 1 and type(hit) is bool):
            raise ValueError(f"{what}: a row without a string itemId, an integer firstGate >= 1 and a boolean hit")
        if iid in out:
            raise ValueError(f"{what}: an item appears twice")
        out[iid] = {"itemId": iid, "firstGate": gate, "hit": hit}
    return out


def append_exposure(existing: list[dict], gate: int, results: list[dict]):
    """The exposure record a gate run appends: each held-out item is counted once, at the FIRST gate that exposed it
    (EP §12.1 step 6), so an item already on record is left exactly as it was. -> (rows sorted by itemId, appended,
    already). A gate below one already recorded, a non-boolean ``hit`` or a repeated item is an error."""
    if type(gate) is not int or gate < 1:
        raise ValueError("gate must be an integer >= 1")
    cur = _exposure_rows(existing, "exposure record")
    if cur and gate < max(r["firstGate"] for r in cur.values()):
        raise ValueError("gate is earlier than a gate already on record")
    new: dict[str, dict] = {}
    for r in results:
        iid, hit = r.get("itemId"), r.get("hit")
        if not (isinstance(iid, str) and iid and type(hit) is bool):
            raise ValueError("results: a row without a string itemId and a boolean hit")
        if iid in new:
            raise ValueError("results: an item appears twice")
        new[iid] = {"itemId": iid, "firstGate": gate, "hit": hit}
    fresh = {k: v for k, v in new.items() if k not in cur}
    rows = sorted({**cur, **fresh}.values(), key=lambda r: r["itemId"])
    return rows, len(fresh), len(new) - len(fresh)


# =============================================================================================
# freeze-check (-p1-ml.md §6.2) and the oracle candidates (§6.4)
# =============================================================================================

S47_COMMIT = "511b14cf"                                     # the split commit: the path set starts here
P1_COMMIT_RE = re.compile(r"C-1001:? P1")                   # a commit is P1 iff its MESSAGE contains this (§6.2): a search,
#   never a prefix match: real subjects read ``cpg C-1001 P1-13: ...`` and ``cpg C-1001 P1-48 + P1-02: ...``. Over-inclusion
#   only scans more files; a missed P1 commit makes the check report clean without covering a builder file.
CUSTODIAN_FILES = ("cpg/eval/heldout-manifest.json", "cpg/eval/preregistration.jsonl")
CUSTODIAN_DIRS = ("cpg/eval/dev/", "cpg/eval/decisions/")   # already public as dev or C; ``cpg/eval/tools/`` is NOT one


def _git(repo: Path, *args: str) -> bytes:
    r = subprocess.run(["git", "-C", str(repo), "-c", "core.quotepath=off", *args], capture_output=True)
    if r.returncode != 0:
        raise ValueError(f"git {args[0]} failed: {r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


def path_set(repo: Path, s47: str, record: str) -> list[str]:
    """``git diff --name-only <S47> <record>`` (no rename detection: a renamed file is its new path, as a delete + add)."""
    out = _git(repo, "diff", "--name-only", "--no-renames", "-z", s47, record)
    return sorted(p for p in out.decode("utf-8").split("\0") if p)


def commit_subjects(repo: Path, s47: str, record: str) -> dict[str, list[str]]:
    """path -> full messages (``%B``) of the non-merge commits in ``S47..record`` that touched it."""
    out = _git(repo, "log", "--no-merges", "--no-renames", "--name-only", "--format=\x01%B\x02", f"{s47}..{record}").decode("utf-8")
    msgs: dict[str, list[str]] = {}
    for chunk in out.split("\x01")[1:]:
        msg, _, names = chunk.partition("\x02")
        for line in names.splitlines():
            if line:
                msgs.setdefault(line, []).append(msg)
    return msgs


def is_p1_commit(message: str) -> bool:
    return P1_COMMIT_RE.search(message) is not None


def classify_path(path: str, subjects: list[str]):
    """-> ``None`` (stays in the set) or the exclusion reason. A custodian output is excluded first; a path is excluded
    as non-P1 only when it has commits in the range and EVERY one is non-P1 (a mixed commit counts as P1; a path with no
    attributable commit cannot be proved non-P1 and stays)."""
    if path in CUSTODIAN_FILES or any(path.startswith(d) for d in CUSTODIAN_DIRS):
        return "custodian-output"
    if subjects and not any(is_p1_commit(x) for x in subjects):
        return "non-p1-commits"
    return None


def read_at(repo: Path, rev: str, path: str) -> bytes | None:
    """The committed bytes of ``path`` at ``rev``; ``None`` when the path is absent there (deleted by the range);
    ValueError when it exists but cannot be read as a file (a submodule, a tree)."""
    if not _git(repo, "ls-tree", "-z", rev, "--", path):
        return None
    return _git(repo, "cat-file", "blob", f"{rev}:{path}")


def check_text(text: str, heldout: list[dict], dev: list[dict], contact: list[dict]):
    """One ``check-report`` run over one text -> ``(rc, hits, n_probes)``; rc 0 clean, 1 hits, 4 unreadable/invalid."""
    try:
        hits, n = check_report(text, heldout, dev, contact)
    except (ValueError, KeyError, TypeError) as e:
        return 4, [], 0
    return (1 if hits else 0), hits, n


def probe_map(heldout: list[dict]) -> dict[str, list[str]]:
    """The custodian's reverse map: 12-hex probe hash -> the held-out record ids carrying that probe."""
    out: dict[str, set] = {}
    for rec in heldout:
        who = rec.get("itemId") or rec.get("qid") or "?"
        for _, v in _mk.record_probes(rec):
            out.setdefault(sha(_mk.collapse(v))[:12], set()).add(who)
    return {h: sorted(v) for h, v in sorted(out.items())}


def positive_control(text: str, heldout: list[dict], dev: list[dict], contact: list[dict]) -> bool:
    """A held-out probe planted in a scratch copy of one artifact must be reported as EXACTLY one new hit. The planted
    probe is a 12-hex id probe the artifact does not already hit, so exactly one probe can match it."""
    _, before, _ = check_text(text, heldout, dev, contact)
    for rec in heldout:
        for k, v in _mk.record_probes(rec):
            if k == "id" and len(v) == 12 and sha(_mk.collapse(v))[:12] not in before:
                rc, after, _ = check_text(text + "\n" + v + "\n", heldout, dev, contact)
                return rc == 1 and sorted(set(after) - set(before)) == [sha(_mk.collapse(v))[:12]]
    return False


def freeze_check(repo: Path, s47: str, record: str, heldout: list[dict], dev: list[dict], contact: list[dict]) -> dict:
    """-p1-ml.md §6.2: ``check-report`` once per file over ``git diff --name-only <S47> <record>`` minus the recorded
    exclusions, on the text committed at ``record``. A file that cannot be read or checked is ``failed`` (exit 4),
    never skipped; a file absent at ``record`` is listed as ``deleted``. -> {paths, counts, exit}."""
    paths = path_set(repo, s47, record)
    subj = commit_subjects(repo, s47, record)
    rows, n_probes = [], 0
    for p in paths:
        why = classify_path(p, subj.get(p, []))
        if why:
            rows.append({"path": p, "status": "excluded", "reason": why})
            continue
        try:
            blob = read_at(repo, record, p)
        except ValueError:
            rows.append({"path": p, "status": "failed", "reason": "unreadable at the record commit"})
            continue
        if blob is None:
            rows.append({"path": p, "status": "deleted", "reason": "absent at the record commit"})
            continue
        rc, hits, n = check_text(blob.decode("utf-8", errors="replace"), heldout, dev, contact)
        n_probes = max(n_probes, n)
        if rc in (2, 4):
            rows.append({"path": p, "status": "failed", "reason": f"check-report exit {rc}"})
        else:
            rows.append({"path": p, "status": "checked", "hits": hits})
    cnt = collections.Counter(r["status"] for r in rows)
    exc = collections.Counter(r["reason"] for r in rows if r["status"] == "excluded")
    n_hits = sum(len(r.get("hits", ())) for r in rows)
    code = 4 if cnt["failed"] else (1 if n_hits else 0)
    return {"paths": rows, "probes": n_probes, "diffPaths": len(paths), "checked": cnt["checked"], "failed": cnt["failed"],
            "deleted": cnt["deleted"], "excluded": dict(exc), "hits": n_hits, "exit": code}


def oracle_candidates(edges: list[dict]) -> list[dict]:
    """P1-47's observed edges ``{consumer, key[, lines]}`` as key-item records (``ccg-key-v1``: anchor = the contract key,
    impacted = the consumer element), so ``--heldout`` can carry them as extra probes (-p1-ml.md §6.4)."""
    out: dict[str, dict] = {}
    for e in edges:
        key, cons = _mk.nfc_strip(dict(e["key"])), _mk.nfc_strip(dict(e["consumer"]))
        if key.get("type") != "contract":
            raise ValueError("an oracle edge's key must be a contract locator")
        key = _mk.normalize_contract(key, "provider")
        cons.setdefault("repoId", _mk.REPO_ID)
        if isinstance(cons.get("path"), str):
            cons["path"] = _mk._norm_path(cons["path"])
        aid, iid = _mk.locator_id(key), _mk.locator_id(cons)
        item = _mk._ids.tuple_id(["evalItem", aid, iid])
        lines = sorted({int(x) for x in e.get("lines") or []})
        out[item] = {"itemId": item, "anchorId": aid, "impactedId": iid,
                     "originals": [{"schema": "ccg-key-v1", "anchor": key, "impacted": cons, "lines": lines}]}
    return [out[k] for k in sorted(out)]


# =============================================================================================
# CLI
# =============================================================================================

def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _alias(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--verify" and len(argv) > 1:
        return ["verify", "--manifest", argv[1]] + argv[2:]
    if argv and argv[0] == "--check-report" and len(argv) > 1:
        return ["check-report", "--report", argv[1]] + argv[2:]
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    P = lambda **kw: dict(type=Path, **kw)

    def add(name, *flags):
        sp = sub.add_parser(name)
        for f, kw in flags:
            sp.add_argument(f, **kw)

    add("split", ("--match", P(required=True)), ("--context", P(required=True)), ("--contact", P(required=True)),
        ("--eval-commit", dict(required=True)), ("--heldout-dir", P(required=True)), ("--dev-dir", P(required=True)),
        ("--report", P()), ("--log", P()))
    add("manifest", ("--heldout-dir", P(required=True)), ("--eval-commit", dict(required=True)),
        ("--protocol", dict(required=True)), ("--contact", P(required=True)), ("--out", P(required=True)),
        ("--raw-root", P()), ("--raw", dict(nargs="*", default=[])))
    add("verify", ("--manifest", P(required=True)), ("--heldout-dir", P(required=True)), ("--raw-root", P()))
    add("check-report", ("--report", P(required=True)), ("--heldout", P(nargs="+")), ("--heldout-dir", P()),
        ("--dev", P(nargs="*", default=[])), ("--contact", P()))
    add("contact", ("--in", dict(dest="inp", required=True, type=Path)), ("--out", P(required=True)))
    add("freeze-check", ("--record", dict(required=True)), ("--s47", dict(default=S47_COMMIT)), ("--repo", P(default=Path("."))),
        ("--heldout", P(required=True, nargs="+")), ("--dev", P(nargs="*", default=[])), ("--contact", P()),
        ("--out", P()), ("--probe-map", P()), ("--positive-control", dict(action="store_true")))
    add("oracle-candidates", ("--in", dict(dest="inp", required=True, type=Path)), ("--out", P(required=True)))
    add("regate-list", ("--match", P(required=True)), ("--context", P(required=True)), ("--contact", P(required=True)),
        ("--eval-commit", dict(required=True)), ("--manifest", P(required=True)), ("--missed", P()), ("--fresh", P()),
        ("--promote-reserve", dict(action="store_true")), ("--heldout-dir", P(required=True)),
        ("--dev-dir", P(required=True)), ("--report", P()))
    add("exposure", ("--exposure", P(required=True)), ("--gate", dict(required=True, type=int)), ("--results", P(required=True)))
    add("apply-corrections", ("--match-scoped", P(required=True)), ("--marks-template", P(required=True)),
        ("--verify", P(required=True)), ("--out-match", P(required=True)), ("--corrections", P(required=True)))
    add("ambiguity", ("--questions", P(required=True)), ("--aid-map", P(required=True)), ("--out", P(required=True)),
        ("--log", P(required=True)))
    add("split-ac24", ("--in", dict(dest="inp", required=True, type=Path)), ("--eval-commit", dict(required=True)),
        ("--heldout-dir", P(required=True)), ("--dev-dir", P(required=True)))
    a = ap.parse_args(_alias(list(sys.argv[1:] if argv is None else argv)))
    try:
        if a.verb == "split":
            m = json.loads(a.match.read_text(encoding="utf-8"))
            units = json.loads(a.context.read_text(encoding="utf-8")).get("units") or []
            contact = {r["anchorId"] for r in read_contact(a.contact)}
            res = split(m["records"], units, contact, a.eval_commit)
            write_split(res, a.heldout_dir, a.dev_dir)
            rep = res["report"]
            if a.report:
                a.report.parent.mkdir(parents=True, exist_ok=True)
                a.report.write_text(canon(rep) + "\n", encoding="utf-8")
            if a.log:
                write_jsonl(a.log, [dict(l, itemId=f"{i:06d}") for i, l in enumerate(res["log"])])
            print(f"H: {rep['H']}\nH_all: {rep['H_all']}\nHs: {canon(rep['Hs'])}\nmarked: {canon(rep['markedStrata'])}\n"
                  f"served-anchor: {rep['servedAnchorMoves']}\nreserve-counting: {rep['reserveCounting']}\n"
                  f"trigger: {str(rep['trigger']).lower()}")
            return 0
        if a.verb == "manifest":
            n = len({r["anchorId"] for r in read_contact(a.contact)})
            man = build_manifest(a.heldout_dir, a.eval_commit, a.protocol, n, a.raw_root, a.raw)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(man, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"files: {len(man['files'])}\nraw: {len(man['rawArtifacts'])}\nout: {a.out}")
            return 0
        if a.verb == "contact":
            rows = build_contact(_read_jsonl(a.inp))
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text("".join(canon(r) + "\n" for r in rows), encoding="utf-8")
            print(f"contact-anchors: {len({r['anchorId'] for r in rows})}\nlines: {len(rows)}\nout: {a.out}")
            return 0
        if a.verb == "oracle-candidates":
            rows = oracle_candidates(_read_jsonl(a.inp))
            write_jsonl(a.out, rows)
            print(f"candidates: {len(rows)}\nout: {a.out}")
            return 0
        if a.verb == "freeze-check":
            why = check_heldout_list(a.heldout)
            if why:
                print(f"split_manifest: {why}", file=sys.stderr)
                return 2
            top = Path(_git(a.repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
            for q in (a.out, a.probe_map):
                if q is not None and (q.resolve() == top or top in q.resolve().parents):
                    print("split_manifest: --out / --probe-map hold held-out-derived hashes and must not sit inside the repo tree",
                          file=sys.stderr)
                    return 2
            ho = [r for p in a.heldout for r in _read_jsonl(p)]
            dev = [r for p in a.dev for r in _read_jsonl(p)]
            contact = [r["locator"] for r in read_contact(a.contact)] if a.contact else []
            if a.probe_map:
                a.probe_map.parent.mkdir(parents=True, exist_ok=True)
                a.probe_map.write_text(json.dumps(probe_map(ho), indent=1, sort_keys=True) + "\n", encoding="utf-8")
            rep = freeze_check(a.repo, a.s47, a.record, ho, dev, contact)
            if a.out:
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(canon(rep) + "\n", encoding="utf-8")
            print(f"diff-paths: {rep['diffPaths']}\nexcluded: {canon(rep['excluded'])}\ndeleted: {rep['deleted']}\n"
                  f"checked: {rep['checked']}\nfailed: {rep['failed']}\nprobes: {rep['probes']}\nhits: {rep['hits']}")
            for r in rep["paths"]:
                for h in r.get("hits", ()):
                    print(f"hit: {r['path']} {h}")
                if r["status"] == "failed":
                    print(f"failed: {r['path']}")
            code = rep["exit"]
            if a.positive_control:
                sample = next((r for r in rep["paths"] if r["status"] == "checked"), None)
                text = read_at(a.repo, a.record, sample["path"]).decode("utf-8", errors="replace") if sample else ""
                ok = positive_control(text, ho, dev, contact)
                print(f"control: {'ok' if ok else 'FAILED'}")
                if not ok:
                    code = 4
            return code
        if a.verb == "regate-list":
            m = json.loads(a.match.read_text(encoding="utf-8"))
            units = json.loads(a.context.read_text(encoding="utf-8")).get("units") or []
            contact = {r["anchorId"] for r in read_contact(a.contact)}
            missed = [r["anchorId"] for r in _read_jsonl(a.missed)] if a.missed else []
            fresh = json.loads(a.fresh.read_text(encoding="utf-8"))["records"] if a.fresh else []
            manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
            try:
                out = regate_list(m["records"], units, contact, a.eval_commit, missed, fresh, a.promote_reserve, manifest)
            except Gate1Mismatch as e:
                print("split_manifest: the gate-1 state re-derived from --match/--context/--contact/--eval-commit does not "
                      f"match the manifest ({', '.join(e.args[0])}); nothing written", file=sys.stderr)
                return 1
            write_regate(out, a.heldout_dir, a.dev_dir)
            rep = out["report"]
            if a.report:
                a.report.parent.mkdir(parents=True, exist_ok=True)
                a.report.write_text(canon(rep) + "\n", encoding="utf-8")
            print(f"H: {rep['H']}\nHs: {canon(rep['Hs'])}\nkept: {rep['kept']}\nkept-cluster-fresh: {rep['keptClusterFresh']}\n"
                  f"fresh-cluster: {rep['freshCluster']}\nfresh-counting: {rep['freshCounting']}\nfloor: {rep['floor']}\n"
                  f"floor-met: {str(rep['floorMet']).lower()}\ndiscarded: {rep['discarded']}\nout-of-scope: {rep['outOfScope']}\n"
                  f"fresh-to-dev: {rep['freshToDev']}\nmissed-clusters: {rep['missedClusters']}\n"
                  f"moved-out: {canon(rep['movedOutByStratum'])}\nshortfall: {canon(rep['shortfallByStratum'])}\n"
                  f"served-anchor: {rep['servedAnchorMoves']}\nminimums-met: {str(rep['minimumsMet']).lower()}")
            return 0
        if a.verb == "exposure":
            existing = _read_jsonl(a.exposure) if a.exposure.exists() else []
            rows, n_new, n_old = append_exposure(existing, a.gate, _read_jsonl(a.results))
            a.exposure.parent.mkdir(parents=True, exist_ok=True)
            tmp = a.exposure.with_name(a.exposure.name + ".tmp")
            tmp.write_text("".join(canon(r) + "\n" for r in rows), encoding="utf-8")
            tmp.replace(a.exposure)
            seen = {r.get("itemId") for r in existing}
            hits = sum(1 for r in _read_jsonl(a.results) if r["itemId"] not in seen and r["hit"])       # newly appended hits
            print(f"appended: {n_new}\nalready: {n_old}\nhits: {hits}\ntotal: {len(rows)}")
            return 0
        if a.verb == "apply-corrections":
            m0 = json.loads(a.match_scoped.read_text(encoding="utf-8"))
            m1, log = apply_corrections(m0, _jl(a.marks_template), _jl(a.verify))
            a.out_match.parent.mkdir(parents=True, exist_ok=True)
            a.out_match.write_text(canon(m1) + "\n", encoding="utf-8")
            a.corrections.parent.mkdir(parents=True, exist_ok=True)
            a.corrections.write_text("".join(json.dumps(x) + "\n" for x in log), encoding="utf-8")
            print(f"corrections: {len(log)}\nrecords: {len(m0['records'])} -> {len(m1['records'])}")
            return 0
        if a.verb == "ambiguity":
            recs, out, rw, st = ambiguity_join(_jl(a.questions / "QW.jsonl"), _jl(a.aid_map),
                                               {w: _jl(a.questions / f"{w}.jsonl") for w in ("AN1", "AN2")})
            _dump_jsonl_plain(a.out, recs)
            a.log.parent.mkdir(parents=True, exist_ok=True)
            a.log.write_text(json.dumps({"outcomes": dict(out), "rewrites": {str(k): v for k, v in rw.items()}}), encoding="utf-8")
            print(f"agreement: {st['agree']}/{st['n']}\noutcomes: {dict(out)}\nkappa: {st['kappa']}\n"
                  f"gold-rewrites: {sum(rw.values())}")
            return 0
        if a.verb == "split-ac24":
            held, dev = split_ac24(_jl(a.inp), a.eval_commit)
            _dump_jsonl_plain(a.heldout_dir / "ac24.jsonl", held)
            _dump_jsonl_plain(a.dev_dir / "ac24.jsonl", dev)
            print(f"held-out: {len(held)}\ndev: {len(dev)}")
            return 0
        if a.verb == "verify":
            man = json.loads(a.manifest.read_text(encoding="utf-8"))
            bad = verify_manifest(man, a.heldout_dir, a.raw_root)
            print(f"checked: {len(man.get('files') or []) + len(man.get('rawArtifacts') or [])}\nmismatches: {len(bad)}")
            for b in bad:
                print(f"mismatch: {b}")
            return 1 if bad else 0
        if a.verb == "check-report":
            if bool(a.heldout) == bool(a.heldout_dir):
                print("split_manifest: give exactly one of --heldout-dir or --heldout", file=sys.stderr)
                return 2
            files = a.heldout or heldout_files(a.heldout_dir)
            why = check_heldout_list(files)
            if why:
                print(f"split_manifest: {why}", file=sys.stderr)
                return 2
            ho = [r for p in files for r in _read_jsonl(p)]
            dev = [r for p in a.dev for r in _read_jsonl(p)]
            contact = [r["locator"] for r in read_contact(a.contact)] if a.contact else []
            hits, n = check_report(a.report.read_text(encoding="utf-8"), ho, dev, contact)
            print(f"probes: {n}\nhits: {len(hits)}")
            for h in hits:
                print(f"hit: {h}")
            return 1 if hits else 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"split_manifest: {e}", file=sys.stderr)
        return 4
    return 2


if __name__ == "__main__":
    sys.exit(main())
