#!/usr/bin/env python3
"""split_manifest.py — the AC-5 split, the held-out manifest and the report leak check (S49 second half, P0-42b).

Spec: eval-protocol ``docs/plans/connected-code-graph-eval-protocol-ml.md`` §9.1-§9.4, §13.6 (v1.5) and
EP2 §1.4. Not built here (needed from P1, EP §13.6): ``regate-list``, ``exposure``; and the AC-24 half of
``split`` (§10.2), ``apply-corrections`` and ``ambiguity``.

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

Inputs: ``--match`` is match_keys' ``scope`` output (``counting``, ``inScope`` per record); ``--context`` is
``{"units": [{"rootPath","lang"}]}``; ``--contact`` is ``work/contact.jsonl`` as the ``contact`` verb writes it (the only accepted shape). Exit: 0 ok, 1 findings (a leak, a
manifest mismatch), 2 usage, 4 missing or invalid input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    """EP §12.1 step 4 routing (library). A fresh item on an anchor in a dev cluster, or in C, goes to ``dev``;
    on a kept (held) cluster it is ``kept-cluster-fresh`` and does not count toward the fresh floor; on an anchor
    with no cluster it is ``fresh-cluster``. -> [{itemId, anchorId, flag}]."""
    out = []
    for g in fresh:
        a = g["anchorId"]
        k = res["cluster_of"].get(a)
        if a in res["C"] or (k is not None and res["side"][k] == "dev"):
            flag = "dev"
        elif k is not None:
            flag = "kept-cluster-fresh"
        else:
            flag = "fresh-cluster"
        out.append({"itemId": g["itemId"], "anchorId": a, "flag": flag})
    return out


def out_record(g: dict) -> dict:
    m0 = g["members"][0]
    return {"itemId": g["itemId"], "anchorId": g["anchorId"], "impactedId": g["impactedId"],
            "derivedStratum": g["derivedStratum"], "counting": bool(g["counting"]),
            "anchor": m0["anchor"], "impacted": m0["impacted"], "provider": m0.get("provider"),
            "serves": m0.get("serves"), "lines": sorted({l for m in g["members"] for l in m.get("lines") or []}),
            "chain": m0.get("chain"), "writers": g["sessions"], "originals": [m["original"] for m in g["members"]]}


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["itemId"])
    path.write_text("".join(canon(r) + "\n" for r in rows), encoding="utf-8")


def write_split(res: dict, heldout_dir: Path, dev_dir: Path) -> None:
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
    for s in STRATA:
        write_jsonl(heldout_dir / f"ac5-{s}.jsonl", held[s])
        write_jsonl(dev_dir / f"ac5-{s}.jsonl", dev[s])
    write_jsonl(heldout_dir / "ac5-reserve.jsonl", reserve)
    (heldout_dir / "exposure.jsonl").write_text("", encoding="utf-8")


# =============================================================================================
# manifest and --verify (EP §9.3)
# =============================================================================================

def file_entry(path: Path) -> dict:
    """Hash, byte size and line count first (a corrupted file must still be measurable); per-line fields best effort."""
    data = path.read_bytes()
    lines = [l for l in data.split(b"\n") if l.strip()]
    rows = []
    for l in lines:
        try:
            rows.append(json.loads(l.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            rows.append({})
    return {"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
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
