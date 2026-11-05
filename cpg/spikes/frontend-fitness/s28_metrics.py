#!/usr/bin/env python3
"""S28 (P0-21) flow-projection metrics for ONE language: ML §7.3 on `truth/<lang>.flow-final.jsonl`, §8.4 provisional.

    cd cpg && .venv/bin/python spikes/frontend-fitness/s28_metrics.py <lang>

Reads  : <work>/joern/<lang>/s28-run.json (+ the facts files extract.sc wrote, off and on), s27-run.json (parse timings),
         sample/<lang>-flow.jsonl, truth/<lang>.flow-final.jsonl, adjudication/<lang>-flow-audit.jsonl (convention tags),
         truth/<lang>.defs.jsonl (read only, census reach), corpora.yaml, the clean corpus tree.
Writes : results/projection-<lang>.json, results/projection-<lang>.answers.jsonl,
         adjudication/<lang>-flow-misses.jsonl, adjudication/<lang>-flow-spurious.jsonl  (`proposedCause` / `verdict` empty:
         the data-scientist decides).
Never touches measure.sc, the S27 results, adjudication/<lang>-misses*.jsonl, or the report.  Definitions: s28_core.DEFINITIONS
and the `definitions` block of the result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s28_core as C  # noqa: E402
import yaml  # noqa: E402

WORK = C.CPG / ".ccg-work" / "frontend-fitness" / "joern"
Z = 1.959964
SYNTH_NAMES = {"<module>", "<global>", "<body>", ":program", "<clinit>", "<fakeNew>", "<includes>"}


def is_synth(name):
    return name in SYNTH_NAMES or name.endswith("<metaClassAdapter>") or name.startswith("<unresolved")

# default-ignored directory segments of the shipped frontends (report §a1.4, results/frontend-defaults-probe.json)
DEFAULT_IGNORED = {"javascript": {"i18n", "test", "tests"}, "java": {"test"}, "cpp": {"test", "tests"}}
EXTRACT_KEYS = {"python": "extract", "javascript": "extract", "go": "extract", "java": "extract", "csharp": "extract-guard", "cpp": "extract"}


def wilson(k, n):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def prop(k, n):
    return {"k": k, "n": n, "p": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < 3e-12:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    p = 0.5 * _ibeta(df / 2, 0.5, x)
    return 1 - p if t > 0 else p


def t_quantile(q, df):
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ratio_ci(rows):
    """rows: [(m_i, y_i)] -> ML §7.3 linearised ratio estimator with t(0.975, n-1)."""
    rows = [(m, y) for m, y in rows if m > 0]
    n = len(rows)
    sm = sum(m for m, _ in rows)
    sy = sum(y for _, y in rows)
    if n < 2 or sm == 0:
        return {"n": n, "R": None}
    R = sy / sm
    mbar = sm / n
    V = sum((y - R * m) ** 2 for m, y in rows) / (n * (n - 1) * mbar * mbar)
    t = t_quantile(0.975, n - 1)
    h = t * math.sqrt(V)
    return {"n": n, "pairs": sm, "connected": sy, "R": round(R, 4), "t": round(t, 4), "se": round(math.sqrt(V), 5),
            "ci95": [round(max(0.0, R - h), 4), round(min(1.0, R + h), 4)]}


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ method matching
def last_name(qname: str) -> str:
    q = qname.split("::")[-1]
    q = q.split("#")[0]
    return q.split(".")[-1]


def match_method(F, byfile, s):
    name = last_name(s["qname"])
    cands = []
    for m in byfile.get(s["path"], []):
        if is_synth(m["name"]):
            continue
        if m["l"] <= s["lineEnd"] and m["le"] >= s["lineStart"]:
            cands.append(m)
    if not cands:
        return None, "no-method-in-file" if byfile.get(s["path"]) else "file-absent-from-cpg"
    eq = [m for m in cands if m["name"] == name]
    pool = eq or [m for m in cands if abs(m["l"] - s["lineStart"]) <= 3 and abs(m["le"] - s["lineEnd"]) <= 3]
    if not pool:
        return None, "no-method-of-that-span"
    pool.sort(key=lambda m: (abs(m["l"] - s["lineStart"]) + abs(m["le"] - s["lineEnd"]), int(m["id"])))
    how = "name+span" if eq else "span-only"
    if len(pool) > 1 and (abs(pool[0]["l"] - s["lineStart"]) + abs(pool[0]["le"] - s["lineEnd"])) == (abs(pool[1]["l"] - s["lineStart"]) + abs(pool[1]["le"] - s["lineEnd"])):
        how += "+tie"
    return pool[0], how


# ------------------------------------------------------------------------------------------------ labels
def ckey(ep):
    p = ep["point"]
    if p == "param":
        return ("param", ep["index"])
    if p == "call":
        return ("call", ep["calleeName"], ep["ordinal"])
    if p == "ret":
        return ("ret", ep["ordinal"])
    return ("cond", C.collapse(ep["code"]))


def pair_id(ordinal, pr):
    return (ordinal, pr["kind"], ckey(pr["source"]), ckey(pr["sink"]))


def load_tags(lang):
    tags = defaultdict(set)
    removed = defaultdict(list)  # pair id -> [convention] for pairs ruled invalid in Part F
    fp = HERE / "adjudication" / f"{lang}-flow-audit.jsonl"
    for line in open(fp, encoding="utf-8"):
        d = json.loads(line)
        if "pair" not in d:
            continue
        pid = pair_id(d["ordinal"], d["pair"])
        conv = d.get("convention") or ""
        if d.get("decision") == "valid" and conv:
            tags[pid].add(conv)
        if d.get("decision") == "invalid":
            removed[pid].append(conv or "invalid")
    return tags, removed


# ------------------------------------------------------------------------------------------------ one configuration
def analyse_cfg(lang, cfg, run, samp, gold, tags, removed, src, commit):
    F = C.load_facts(Path(run["factsFile"]), lang)
    stats = {k: (int(v) if isinstance(v, str) and v.lstrip("-").isdigit() else v) for k, v in F["stats"].items()}
    F["stats"] = stats
    byfile = defaultdict(list)
    for mid, m in F["m"].items():
        byfile[C.file_of(F, m)].append(m)
    # ---- every method: counts (descriptive density)
    tot = Counter()
    perfile = defaultdict(lambda: Counter())
    allM = {}
    for mid, m in F["m"].items():
        f = C.file_of(F, m)
        if not f or f.startswith("<"):
            tot["methodsNoFile"] += 1
            continue
        lines = src.lines(f)
        if lines is None:
            tot["methodsFileNotInTree"] += 1
            continue
        M = C.build_method(lang, F, src, mid, f)
        synth_name = is_synth(m["name"])
        tot["methods"] += 1
        tot["points"] += M.points
        tot["dataEdges"] += len(M.data)
        tot["ctlEdges"] += len(M.ctl)
        tot["callSites"] += len(M.calls)
        tot["syntheticCalls"] += M.synthetic
        tot["droppedData"] += M.droppedData
        tot["droppedCtl"] += M.droppedCtl
        pf = perfile[f]
        pf["methods"] += 0 if synth_name else 1
        pf["callSites"] += len(M.calls)
        pf["points"] += M.points
        pf["edges"] += len(M.data) + len(M.ctl)
        allM[mid] = M if False else None
    # (build again only for the labelled ones below, to keep memory flat)
    # ---- labelled methods
    mm = {}
    for s in samp:
        m, how = match_method(F, byfile, s)
        mm[s["ordinal"]] = (m, how)
    ignored = DEFAULT_IGNORED.get(lang, set())
    methods_out = []
    answers = {}  # (ordinal, idx) -> dict
    proj_pairs = {}  # ordinal -> (M, data set, ctl set)
    per_m = []
    for s, g in zip(samp, gold):
        assert s["ordinal"] == g["ordinal"] and s["path"] == g["path"]
        m, how = mm[s["ordinal"]]
        segs = set(s["path"].split("/")[:-1]) | {Path(s["path"]).stem}
        M = C.build_method(lang, F, src, m["id"], C.file_of(F, m)) if m else None
        rec = {"ordinal": s["ordinal"], "path": s["path"], "qname": s["qname"], "span": [s["lineStart"], s["lineEnd"]],
               "matched": bool(m), "matchHow": how,
               "joernMethod": ({"name": m["name"], "span": [m["l"], m["le"]], "params": {str(k): v[1] for k, v in sorted(M.params.items())}} if m else None),
               "defaultIgnoredSegment": sorted(set(s["path"].split("/")[:-1]) & ignored)}
        methods_out.append(rec)
        resolved = []
        for idx, pr in enumerate(g["pairs"]):
            a = {"ordinal": g["ordinal"], "idx": idx, "kind": pr["kind"], "confidence": pr.get("confidence"), "pair": {"source": pr["source"], "sink": pr["sink"]},
                 "tags": sorted(tags.get(pair_id(g["ordinal"], pr), [])), "provenance": pr.get("provenance")}
            if M is None:
                a.update(status="method-absent", connected=False, reason=how)
                answers[(g["ordinal"], idx)] = a
                resolved.append(None)
                continue
            sk, sd = C.resolve_endpoint(M, pr["source"])
            tk, td = C.resolve_endpoint(M, pr["sink"])
            a["endpointDetail"] = {"source": sd, "sink": td}
            resolved.append((sk, tk))
            if sk is None or tk is None or sk == tk:
                a.update(status="endpoint-unresolved" if (sk is None or tk is None) else "endpoint-same-point", connected=False)
                a["unresolvedEnds"] = [n for n, k in (("source", sk), ("sink", tk)) if k is None]
            elif pr["kind"] == "data":
                hops = C.reach(M, sk, tk)
                a.update(status="connected" if hops else "not-connected", connected=bool(hops), hops=hops)
                if not hops:
                    a["diag"] = {"sourceOut": [C.point_label(M, y) for y in sorted(M.adj.get(sk, ()), key=str)][:8],
                                 "sinkIn": [C.point_label(M, x) for (x, y) in sorted(M.data, key=str) if y == tk][:8]}
            else:
                ok = (tk, sk) in M.ctl
                a.update(status="connected" if ok else "not-connected", connected=ok)
                if not ok:
                    a["diag"] = {"sinkControlledBy": [C.point_label(M, c) for (p, c) in sorted(M.ctl, key=str) if p == tk][:8]}
            answers[(g["ordinal"], idx)] = a
        proj_pairs[g["ordinal"]] = (M, resolved)
        per_m.append((g["ordinal"], M, resolved))
    # ---- recall
    A = list(answers.values())
    def rc(sel):
        k = sum(1 for a in sel if a["connected"])
        return prop(k, len(sel))
    data = [a for a in A if a["kind"] == "data"]
    ctl = [a for a in A if a["kind"] == "control"]
    sure = [a for a in A if a["confidence"] == "sure"]
    # cluster CI per method
    byo = defaultdict(list)
    for a in A:
        byo[a["ordinal"]].append(a)
    rows = [(len(v), sum(1 for a in v if a["connected"])) for v in byo.values()]
    res = {
        "pooled": rc(A), "data": rc(data), "control": rc(ctl), "sureOnly": rc(sure),
        "clusterInterval": {"pooled": ratio_ci(rows),
                            "data": ratio_ci([(sum(1 for a in v if a["kind"] == "data"), sum(1 for a in v if a["kind"] == "data" and a["connected"])) for v in byo.values()]),
                            "control": ratio_ci([(sum(1 for a in v if a["kind"] == "control"), sum(1 for a in v if a["kind"] == "control" and a["connected"])) for v in byo.values()])},
        "status": dict(Counter(a["status"] for a in A)),
        "statusByKind": {k: dict(Counter(a["status"] for a in A if a["kind"] == k)) for k in ("data", "control")},
        "dataConnectedHops": dict(sorted(Counter(a.get("hops") for a in data if a["connected"]).items())),
    }
    dc = sum(1 for a in data if a["connected"])
    res["dataSingleEdgeShare"] = prop(sum(1 for a in data if a.get("hops") == 1), dc)
    res["dataChainOnlyShare"] = prop(sum(1 for a in data if a["connected"] and a.get("hops", 1) >= 2), dc)
    eu = [a for a in A if a["status"] == "endpoint-unresolved"]
    res["endpointUnresolvedShare"] = prop(len(eu), len(A))
    res["endpointUnresolvedByClass"] = dict(Counter(tuple_name(a) for a in eu))
    fully = sum(1 for v in byo.values() if v and all(a["connected"] for a in v))
    res["methodsFullyRecovered"] = prop(fully, len(byo))
    res["methodsWithPairs"] = len(byo)
    res["methodsMatched"] = prop(sum(1 for m, _ in mm.values() if m), len(samp))
    # strata
    strata = {}
    tagnames = sorted({t for a in A for t in a["tags"]})
    for t in tagnames:
        sel = [a for a in A if t in a["tags"]]
        strata["convention:" + t] = {**rc(sel), "byKind": dict(Counter(a["kind"] for a in sel)), "status": dict(Counter(a["status"] for a in sel))}
    for pv in sorted({a["provenance"] for a in A if a["provenance"]}):
        sel = [a for a in A if a["provenance"] == pv]
        strata["provenance:" + pv] = {**rc(sel), "status": dict(Counter(a["status"] for a in sel))}
    # method-absent pairs in a default-ignored directory = a frontend fact, own stratum
    ign = {r["ordinal"] for r in methods_out if r["defaultIgnoredSegment"]}
    strata["frontend-default-ignored-dir"] = {**rc([a for a in A if a["ordinal"] in ign]), "methods": sorted(ign), "status": dict(Counter(a["status"] for a in A if a["ordinal"] in ign))}
    absent = {r["ordinal"] for r in methods_out if not r["matched"]}
    strata["method-absent"] = {**rc([a for a in A if a["ordinal"] in absent]), "methods": sorted(absent),
                               "reasons": {str(r["ordinal"]): r["matchHow"] for r in methods_out if not r["matched"]},
                               "inDefaultIgnoredDir": sorted(absent & ign), "otherAbsent": sorted(absent - ign)}
    res["strata"] = strata
    # recall excluding the default-ignored-dir methods (the as-shipped headline stays the pooled figure)
    keep = [a for a in A if a["ordinal"] not in ign]
    res["pooledExcludingDefaultIgnoredDirs"] = rc(keep)
    res["pooledOnMatchedMethods"] = rc([a for a in A if a["ordinal"] not in absent])
    # ---- spurious
    spur_all = []
    for ordinal, M, resolved in per_m:
        if M is None:
            continue
        g = next(x for x in gold if x["ordinal"] == ordinal)
        gd, gc = set(), set()
        unresolved_ends = defaultdict(set)
        for pr, r in zip(g["pairs"], resolved):
            if r and r[0] is not None and r[1] is not None:
                (gd if pr["kind"] == "data" else gc).add((r[0], r[1]) if pr["kind"] == "data" else (r[1], r[0]))
            elif r:
                for e in r:
                    if e is not None:
                        unresolved_ends[pr["kind"]].add(e)
        for (d_, u_) in sorted(M.data, key=str):
            if (d_, u_) not in gd:
                spur_all.append((ordinal, "data", M, d_, u_, unresolved_ends))
        for (p_, c_) in sorted(M.ctl, key=str):
            if (p_, c_) not in gc:
                spur_all.append((ordinal, "control", M, c_, p_, unresolved_ends))
    nproj = sum(len(M.data) + len(M.ctl) for _, M, _ in per_m if M is not None)
    res["projectionPairsOnLabelledMethods"] = {"data": sum(len(M.data) for _, M, _ in per_m if M), "control": sum(len(M.ctl) for _, M, _ in per_m if M)}
    res["spuriousPairs"] = prop(len(spur_all), nproj)
    res["spuriousByKind"] = dict(Counter(k for _, k, *_ in spur_all))
    res["methodsWithProjectedPairs"] = sum(1 for _, M, _ in per_m if M is not None and (M.data or M.ctl))
    # ---- density, timing
    nfiles = len([f for f in perfile if perfile[f]["methods"] > 0])
    loc = sum(len(src.lines(f) or []) for f in perfile if perfile[f]["methods"] > 0)
    res["density"] = {
        "cpgFilesWithMethods": nfiles, "kLOC": round(loc / 1000, 2), "cpgMethods": stats["methods"] if stats else None,
        "nonSyntheticMethodsPerFile": round(sum(perfile[f]["methods"] for f in perfile) / nfiles, 2) if nfiles else None,
        "callSitesPerFile": round(sum(perfile[f]["callSites"] for f in perfile) / nfiles, 2) if nfiles else None,
        "projectedPoints": tot["points"], "projectedDataEdges": tot["dataEdges"], "projectedControlEdges": tot["ctlEdges"],
        "pointsPerFile": round(tot["points"] / nfiles, 2) if nfiles else None,
        "edgesPerFile": round((tot["dataEdges"] + tot["ctlEdges"]) / nfiles, 2) if nfiles else None,
        "edgesPerMethod": round((tot["dataEdges"] + tot["ctlEdges"]) / tot["methods"], 3) if tot["methods"] else None,
        "dataEdgesPerMethod": round(tot["dataEdges"] / tot["methods"], 3) if tot["methods"] else None,
        "controlEdgesPerMethod": round(tot["ctlEdges"] / tot["methods"], 3) if tot["methods"] else None,
        "syntheticCallsDropped": tot["syntheticCalls"], "dataFactsDropped": tot["droppedData"], "controlFactsDropped": tot["droppedCtl"],
        "rawCdgEdges": stats["cdgRaw"] if stats else None, "rawCdgSelfEdges": stats["cdgSelf"] if stats else None,
        "rawDataFacts": stats["dataRaw"] if stats else None,
        "methodsWithoutFile": tot["methodsNoFile"], "methodsFileNotInTree": tot["methodsFileNotInTree"],
    }
    return F, res, answers, spur_all, mm, methods_out, per_m


def tuple_name(a):
    return "+".join(a.get("unresolvedEnds", []))


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    a = ap.parse_args()
    lang = a.lang
    base = WORK / lang
    if lang == "kotlin":
        return kotlin_unmeasured(base)
    s28 = json.loads((base / "s28-run.json").read_text())["runs"]
    s27 = json.loads((base / "s27-run.json").read_text())["runs"]
    cfgs = yaml.safe_load(open(HERE / "corpora.yaml"))["corpora"][lang]
    commit = cfgs["commit"]
    src = C.Src(C.CORPORA / lang / "src")
    samp = [json.loads(l) for l in open(HERE / "sample" / f"{lang}-flow.jsonl", encoding="utf-8")]
    gold = [json.loads(l) for l in open(HERE / "truth" / f"{lang}.flow-final.jsonl", encoding="utf-8")]
    tags, removed = load_tags(lang)
    ek = EXTRACT_KEYS[lang]
    out = {"lang": lang, "variant": "as-shipped (headline)", "truth": f"truth/{lang}.flow-final.jsonl", "goldenMethods": len(gold),
           "goldenPairs": sum(len(g["pairs"]) for g in gold), "corpusCommit": commit, "toolchainImage": None, "cfg": {}}
    out["asShippedExtract"] = {c: {"rc": s28[f"{c}.extract"]["rc"], "wallSeconds": s28[f"{c}.extract"]["wallSeconds"],
                                   "factsFile": s28[f"{c}.extract"].get("factsFile")} for c in ("off", "on") if f"{c}.extract" in s28}
    guard = ek.endswith("guard")
    if guard:
        out["variant"] = "as-shipped extract.sc ABORTS; all figures below are from s28-extract-bodiless-guard.sc (extract.sc + 3 one-line guards), secondary"
    allcfg = {}
    for cfg in ("off", "on"):
        run = s28[f"{cfg}.{ek}"]
        if run["rc"] != 0 or not run.get("factsFile"):
            out["cfg"][cfg] = {"error": f"extract rc={run['rc']}"}
            continue
        F, res, answers, spur, mm, methods_out, per_m = analyse_cfg(lang, cfg, run, samp, gold, tags, removed, src, commit)
        cc = s28.get(f"{cfg}.content-check", {})
        res["contentCheck"] = cc.get("contentLines")
        res["extract"] = {k: run.get(k) for k in ("rc", "wallSeconds", "peakMemMiB", "factsMB", "startedUtc")}
        res["extractInternalMs"] = {k: F["stats"][k] for k in ("loadMs", "projectionMs", "totalMs")}
        res["histogramMsPerMethod"] = F["stats"]["histogram"]
        p = s27.get(f"{cfg}.parse", {})
        res["cpgMB"] = p.get("cpgMB")
        res["parse"] = {"wallSeconds": p.get("wallSeconds"), "peakMemMiB": p.get("peakMemMiB")}
        d = res["density"]
        d["cpgMBPerFile"] = round(p["cpgMB"] / d["cpgFilesWithMethods"], 4) if p.get("cpgMB") and d["cpgFilesWithMethods"] else None
        d["extractSecondsPerKLOC"] = round(run["wallSeconds"] / d["kLOC"], 3) if d["kLOC"] else None
        d["parseSecondsPerKLOC"] = round(p["wallSeconds"] / d["kLOC"], 3) if p.get("wallSeconds") and d["kLOC"] else None
        d["projectionMsPerKLOC"] = round(F["stats"]["projectionMs"] / d["kLOC"], 2) if d["kLOC"] else None
        st = s28.get(f"{cfg}.cpg-stats", {}).get("lines") or []
        if st:
            kv = dict(x.split("=") for x in st[0].split()[1:])
            nf = int(kv["files"])
            d["cpgWhole"] = {"nodes": int(kv["nodes"]), "edges": int(kv["edges"]), "fileNodes": nf, "calls": int(kv["calls"]),
                             "nodesPerFile": round(int(kv["nodes"]) / nf, 1), "edgesPerFile": round(int(kv["edges"]) / nf, 1),
                             "mbPerFile": round(p["cpgMB"] / nf, 4) if p.get("cpgMB") else None,
                             "note": "FILE nodes not named `<...>`; per-file figures divide the whole CPG (incl. external/stub nodes) by that count"}
        # projection success (ML §7.3): methods emitted without error / methods in the CPG; plus census reach
        n_cpg = None
        for l in cc.get("contentLines") or []:
            if l.startswith("METHOD_COUNT"):
                n_cpg = int(l.rsplit("=", 1)[1])
        res["projectionSuccess"] = prop(F["stats"]["methods"], n_cpg) if n_cpg else None
        allcfg[cfg] = (res, answers, spur, mm, methods_out, per_m, F)
        out["cfg"][cfg] = res
    # ---- census reach (read-only, informational): named tree-sitter definitions with an emitted Joern METHOD
    try:
        out["censusReach"] = census_reach(lang, allcfg.get("off", (None,)*7)[6] if "off" in allcfg else None)
    except Exception as e:  # noqa: BLE001
        out["censusReach"] = {"error": repr(e)}
    # ---- §8.4 provisional decision
    dec = {}
    for cfg, (res, *_r) in allcfg.items():
        ps = res["projectionSuccess"]
        dec[cfg] = {"pooledRecall": res["pooled"]["p"], "pairs": res["pooled"]["n"], "pairsOK": res["pooled"]["n"] >= 120,
                    "recallOK": (res["pooled"]["p"] or 0) >= 0.95, "projectionSuccess": ps["p"] if ps else None,
                    "preconditionOK": bool(ps and ps["p"] >= 0.95)}
    asshipped_ok = all(c in dec and dec[c]["pairsOK"] and dec[c]["recallOK"] and dec[c]["preconditionOK"] for c in ("off", "on"))
    out["flowAvailableProvisional"] = {
        "value": bool(asshipped_ok and not guard), "perConfig": dec,
        "rule": "ML §8.4: pooled flow pair recall >= 0.95 observed over >= 120 golden pairs in BOTH configurations, and projection success >= 0.95; "
                "provisional: before the data-scientist's adjudication of misses and spurious pairs",
        "wouldBeIfGuarded": bool(asshipped_ok) if guard else None}
    # ---- write
    res_dir = HERE / "results"
    adj_dir = HERE / "adjudication"
    jd = lambda o: json.dumps(o, ensure_ascii=False, sort_keys=True)  # noqa: E731
    if "off" in allcfg:
        res_off, ans_off, spur, mm, methods_out, per_m, F = allcfg["off"]
    else:
        raise SystemExit("no off configuration")
    on = allcfg.get("on")
    out["labelledMethods"] = methods_out
    if on:
        disc = [(k, ans_off[k]["connected"], on[1][k]["connected"]) for k in ans_off if ans_off[k]["connected"] != on[1][k]["connected"]]
        out["offOnDiscordantPairs"] = {"offOnlyConnected": sum(1 for _, a, b in disc if a), "onOnlyConnected": sum(1 for _, a, b in disc if b), "pairs": len(ans_off)}
    # answers
    with open(res_dir / f"projection-{lang}.answers.jsonl", "w", encoding="utf-8") as fh:
        for key in sorted(ans_off):
            r = dict(ans_off[key])
            r["on"] = ({k: on[1][key].get(k) for k in ("status", "connected", "hops")} if on else None)
            fh.write(jd(r) + "\n")
    # misses: every labelled pair missed in either configuration
    nmiss = 0
    with open(adj_dir / f"{lang}-flow-misses.jsonl", "w", encoding="utf-8") as fh:
        for key in sorted(ans_off):
            ao, an = ans_off[key], (on[1][key] if on else None)
            if ao["connected"] and (an is None or an["connected"]):
                continue
            m_ = next(x for x in methods_out if x["ordinal"] == ao["ordinal"])
            row = {"part": "S28", "lang": lang, "projection": "extract.sc+bodiless-guard" if guard else "extract.sc", "ordinal": ao["ordinal"], "pairIndex": ao["idx"], "path": m_["path"], "qname": m_["qname"],
                   "methodSpan": m_["span"], "pair": ao["pair"], "kind": ao["kind"], "confidence": ao["confidence"],
                   "class": ao["status"], "missedOff": not ao["connected"], "missedOn": (not an["connected"]) if an else None,
                   "tags": ao["tags"], "provenance": ao["provenance"], "joernMethod": m_["joernMethod"],
                   "detail": {k: ao.get(k) for k in ("endpointDetail", "unresolvedEnds", "diag", "reason") if ao.get(k) is not None},
                   "onClass": an["status"] if an else None, "proposedCause": ""}
            if ao["status"] == "method-absent":
                row["detail"]["defaultIgnoredSegment"] = m_["defaultIgnoredSegment"]
            fh.write(jd(row) + "\n")
            nmiss += 1
    # spurious sample
    seed = f"ccg-p0a-v1-spur|{lang}|{commit}"
    rows = []
    for ordinal, kind, M, a_, b_, unres in spur:
        # a_ = source/def point, b_ = sink/use point (control: a_ = cond, b_ = point)
        sl, tl = C.point_label(M, a_), C.point_label(M, b_)
        key = f"{ordinal}|{kind}|{sl}|{tl}"
        pid = (ordinal, kind, ckey(C.point_obj(M, a_)), ckey(C.point_obj(M, b_)))
        rows.append({"key": key, "hash": sha(seed + "|" + key), "ordinal": ordinal, "kind": kind, "M": M, "src": a_, "snk": b_, "pid": pid, "unres": unres})
    rows.sort(key=lambda r: (r["hash"], r["key"]))
    pick = rows[:50]
    on_spur = set()
    if on:
        for ordinal, kind, M, a_, b_, unres in on[2]:
            on_spur.add(f"{ordinal}|{kind}|{C.point_label(M, a_)}|{C.point_label(M, b_)}")
    with open(adj_dir / f"{lang}-flow-spurious.jsonl", "w", encoding="utf-8") as fh:
        for rank, r in enumerate(pick):
            M = r["M"]
            m_ = next(x for x in methods_out if x["ordinal"] == r["ordinal"])
            twin = []
            for e in (r["src"], r["snk"]):
                if e[0] in ("cs", "c", "r", "p") and e in r["unres"].get(r["kind"], set()):
                    twin.append(C.point_label(M, e))
            row = {"part": "S28-spur", "lang": lang, "projection": "extract.sc+bodiless-guard" if guard else "extract.sc", "rank": rank, "seed": seed, "hash": r["hash"], "pairKey": r["key"], "ordinal": r["ordinal"],
                   "path": m_["path"], "qname": m_["qname"], "methodSpan": m_["span"], "kind": r["kind"],
                   "pair": {"source": C.point_obj(M, r["src"]), "sink": C.point_obj(M, r["snk"])},
                   "inOn": (r["key"] in on_spur) if on else None,
                   "sharesEndpointWithUnresolvedGolden": twin,
                   "ruledInvalidByPartF": removed.get(r["pid"], []),
                   "verdict": "", "note": ""}
            fh.write(jd(row) + "\n")
    out["spuriousSample"] = {"population": len(rows), "drawn": len(pick), "seed": seed, "rule": "SHA-256(seed|pairKey) ascending, first 50; pairKey = ordinal|kind|source|sink with point labels param:<idx>, call:<name>#<ordinal>@<line>, ret:<ordinal>@<line>, cond:<line>:<code>",
                             "omissionRate": None, "truthIncompleteFlag": ("insufficient (n<20)" if len(pick) < 20 else "pending adjudication (n=%d; raised iff omission rate > 0.05)" % len(pick)),
                             "onOnlyPairsNotInOffPopulation": (len(on_spur - {r["key"] for r in rows}) if on else None),
                             "sharesEndpointWithUnresolvedGolden": sum(1 for r in pick if any(e in r["unres"].get(r["kind"], set()) for e in (r["src"], r["snk"]))),
                             "ruledInvalidByPartF": sum(1 for r in pick if removed.get(r["pid"]))}
    out["missesFile"] = {"rows": nmiss, "path": f"adjudication/{lang}-flow-misses.jsonl"}
    out["definitions"] = C.DEFINITIONS
    out["definitions"]["strata"] = ("convention:* = the Part F audit's `convention` tag on a VALID decision (adjudication/<lang>-flow-audit.jsonl), joined to the "
                                    "final pair by (ordinal, kind, endpoint ids without line); provenance:* = the final-set provenance field; "
                                    "frontend-default-ignored-dir = labelled methods whose path has a directory segment the shipped frontend ignores by default (report §a1.4)")
    (res_dir / f"projection-{lang}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True))
    print("wrote", res_dir / f"projection-{lang}.json")
    return 0


def kotlin_unmeasured(base):
    """Kotlin: the default-overlay stage aborts on the corpus (S27), so the projection cannot run on any CPG without a pipeline
    deviation (--nooverlays), and a --nooverlays CPG has no REACHING_DEF / CDG for extract.sc to project.  Recorded, not decided."""
    s28 = json.loads((base / "s28-run.json").read_text())["runs"]
    gold = [json.loads(l) for l in open(HERE / "truth" / "kotlin.flow-final.jsonl", encoding="utf-8")]
    out = {"lang": "kotlin", "variant": "unmeasured", "truth": "truth/kotlin.flow-final.jsonl", "goldenMethods": len(gold),
           "goldenPairs": sum(len(g["pairs"]) for g in gold), "flowAvailableProvisional": {"value": None, "reason": "unmeasured: not a fail, no rule applied"},
           "attempts": {k: {"rc": v["rc"], "contentLines": v.get("contentLines")} for k, v in s28.items() if k.endswith("content-check")},
           "extractProbe": {"rc": 1, "log": str(base / "off" / "s28-extract.log"),
                            "cause": "importCpg re-applies the default overlays; io.joern.dataflowengineoss.passes.reachingdef.ReachingDefPass fails "
                                     "(java.util.NoSuchElementException: key not found: ...MethodRef) and the script exits 1 before writing a fact"},
           "pipelineDeviationNeeded": "joern-parse --nooverlays (s27_run.py --nooverlays): CPG would carry no REACHING_DEF/CDG, so extract.sc would project nothing",
           "note": "no projection-kotlin.answers.jsonl, no flow-misses / flow-spurious files: nothing was projected"}
    (HERE / "results" / "projection-kotlin.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print("wrote projection-kotlin.json")
    return 0


def census_reach(lang, F):
    if F is None:
        return None
    byfile = defaultdict(list)
    for m in F["m"].values():
        byfile[C.file_of(F, m)].append(m)
    off = C.LINE_OFFSET.get(lang, 0)
    n = hit = 0
    for line in open(HERE / "truth" / f"{lang}.defs.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["k"] != "def" or not d["symbol"] or d["kind"] not in ("fn", "ctor"):
            continue
        n += 1
        for m in byfile.get(d["path"], []):
            if is_synth(m["name"]):
                continue
            if m["l"] <= d["endLine"] and m["le"] >= d["declStartLine"] and (m["name"] == d["name"] or (abs(m["l"] - d["declStartLine"]) <= 2 and abs(m["le"] - d["endLine"]) <= 2)):
                hit += 1
                break
    return {"note": "informational: tree-sitter named definitions (kind fn/ctor, symbol true) from truth/<lang>.defs.jsonl (S27c census, read-only, in flux) with a Joern METHOD in the same file overlapping its span and equal-named or within 2 lines at both ends",
            **prop(hit, n)}


if __name__ == "__main__":
    raise SystemExit(main())
