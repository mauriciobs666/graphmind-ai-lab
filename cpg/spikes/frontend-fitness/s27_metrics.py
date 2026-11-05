#!/usr/bin/env python3
"""S27 (P0-20) provisional metrics for ONE language, from the two `measure.sc` outputs (off / on) and
`truth/<lang>.final.jsonl`. ML §7.1, §7.2, §7.4, §8.1, §8.3 (provisional: `final`, not `final2`).

    cd cpg && .venv/bin/python spikes/frontend-fitness/s27_metrics.py <lang>

Reads  : <work>/joern/<lang>/{off,on}/*.measure.jsonl, <work>/joern/<lang>/s27-run.json (timings),
         sample/<lang>.private.jsonl (siteKey, calleeName, kind: the harness's own file), truth/<lang>.final.jsonl,
         contact.json (python, javascript), frame/<lang>.jsonl (file list).
Writes : results/<lang>.json, results/<lang>.answers.jsonl, adjudication/<lang>-misses.jsonl
Never writes final2 or any other adjudication file. `proposedCause` is always "" (the data-scientist decides).

Definitions fixed here where ML leaves a choice (every one is echoed under `definitions` in the result):
  * labelled scored site: a `final.jsonl` line with scored == true. Internal: T non-empty. External: T empty.
  * T-member <-> Joern METHOD (`locate`): same file and `lineNumber` == defLine + METHOD_LINE_OFFSET[lang]; else the innermost
    METHOD of that file whose line span holds the line, within 8 lines of its start, named like the symbol (the frontends start a
    METHOD at its annotations/modifiers, the truth at the name).
  * Res_c(s): the matched CALL's resolved targets that are non-external, non-vendored and not a synthetic
    (non-symbol) METHOD (NONSYMBOL_NAME). Targets of the other kinds are classed (resolved away / non-symbol).
  * Nm coverage of a T member t: calleeName is one of t's spec short names (last qname segment; for a constructor
    `<init>`/`__init__`/`constructor` also the enclosing type name) AND Joern has a non-external non-vendored METHOD at
    t's location. measure.sc's own `nm` (METHODs named calleeName) is the fan-out figure.
  * found = every T member covered by Res_c or Nm. correct = Res_c non-empty and Res_c subset of T.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPG = HERE.parent.parent
WORK = CPG / ".ccg-work" / "frontend-fitness" / "joern"
Z = 1.959964

# Joern METHOD line vs truth defLine (checked per language by `tMappedAtDefLine` in the result; a mismatch is visible).
METHOD_LINE_OFFSET = {"python": 0, "javascript": 0, "go": 0, "java": 0, "csharp": -1, "cpp": 0, "kotlin": 0}
# Synthetic Joern METHODs that are not bundle symbols (never a resolution target that counts).
NONSYMBOL_NAME = re.compile(r"^(<module>|<body>|<clinit>|<global>|<lambda>\d*|:program|<metaClassCallHandler>|<fakeNew>|<unresolved.*>|<empty>)$")
CTOR_NAMES = {"<init>", "__init__", "constructor"}


def wilson(k, n):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def prop(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": round(k / n, 4) if n else None, "wilson95": [lo, hi]}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    s = sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n
    return min(1.0, 2 * s)


def newcombe_paired(a, b, c, d):
    """difference p_on - p_off; a both, b on-only, c off-only, d neither (Newcombe 1998 method 10)."""
    n = a + b + c + d
    if n == 0:
        return None
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    den = math.sqrt(r1 * r2 * c1 * c2) if r1 * r2 * c1 * c2 else 0
    phi = ((a * d - b * c) / den) if den else 0.0
    dd = p1 - p2
    lo = dd - math.sqrt(max(0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)))
    hi = dd + math.sqrt(max(0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)))
    return {"diff": round(dd, 4), "ci95": [round(lo, 4), round(hi, 4)], "phi": round(phi, 4)}


def nearest_rank(vals, qn, qd):
    if not vals:
        return None
    s = sorted(vals)
    r = -(-qn * len(s) // qd)
    return s[max(1, r) - 1]


def spec_names(qname: str) -> set:
    if not qname:
        return set()
    q = qname.split("::", 1)[-1]
    q = re.sub(r"#\d+$", "", q)
    segs = [re.sub(r"`\d+$", "", s) for s in q.split(".") if s]
    if not segs:
        return set()
    out = {segs[-1]}
    if segs[-1] in CTOR_NAMES and len(segs) >= 2:
        out.add(segs[-2])
    return out


def locate(M, t, off):
    """Joern METHOD (file, lineNumber) standing for truth member t, or None.
    Exact: a METHOD at defLine + off. Else (frontends start a METHOD at its annotations/modifiers, the truth at the name):
    a METHOD of the same file whose [l, le] holds the line and whose name is one of t's spec short names (or a constructor
    `<init>`), innermost (largest l) first."""
    d = t["defLine"] + off
    ms = M["methods"].get(t["path"], {})
    if d in ms:
        return (t["path"], d)
    names = spec_names(t.get("qname") or "") | {"<init>"}
    cand = [m for m in ms.values() if m["l"] <= d <= m["le"] and m["name"] in names and d - m["l"] <= 8]
    if cand:
        m = max(cand, key=lambda x: x["l"])
        return (t["path"], m["l"])
    return None


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_measure(lang, cfg):
    fs = sorted(glob.glob(str(WORK / lang / cfg / "*.measure.jsonl")), key=lambda p: Path(p).stat().st_mtime)
    if not fs:
        return None
    meta, files, methods, sites = None, {}, collections.defaultdict(dict), {}
    for r in read_jsonl(fs[-1]):
        k = r["k"]
        if k == "meta":
            meta = r
        elif k == "file":
            files[r["path"]] = r["methods"]
        elif k == "method":
            methods[r["file"]][r["l"]] = r
        elif k == "site":
            sites[r["key"]] = r
    return {"path": fs[-1], "meta": meta, "files": files, "methods": methods, "sites": sites}


def analyse(lang, cfgname, M, rows, off):
    """per-site answers for one configuration."""
    out = {}
    for ro in rows:
        s = M["sites"].get(ro["siteKey"])
        ans = {"matched": bool(s and s["matched"]), "call": s.get("call") if s else None, "nm": s["nm"] if s else None,
               "res": [], "resolvedSymbols": [], "resolvedExternal": 0, "resolvedVendored": 0, "resolvedNonSymbol": [], "nmList": (s or {}).get("nmList", [])}
        for t in (s["res"] if s else []):
            if t["ext"]:
                ans["resolvedExternal"] += 1
            elif t["vendored"]:
                ans["resolvedVendored"] += 1
            else:
                nm = t["fullName"]
                mm = M["methods"].get(t["file"], {}).get(t["l"])
                name = mm["name"] if mm else ""
                if NONSYMBOL_NAME.match(name):
                    ans["resolvedNonSymbol"].append({"fullName": nm, "file": t["file"], "l": t["l"]})
                else:
                    ans["resolvedSymbols"].append({"fullName": nm, "file": t["file"], "l": t["l"]})
            ans["res"].append(t)
        out[ro["ordinal"]] = ans
    return out


def judge(ro, tr, ans, M, off):
    """found / correct for one scored site in one configuration."""
    T = tr["T"]
    resset = {(r["file"], r["l"]) for r in ans["resolvedSymbols"]}
    members = []
    tkeys = set()
    for t in T:
        loc = locate(M, t, off)
        key = loc or (t["path"], t["defLine"] + off)
        tkeys.add(key)
        via, why = None, None
        if key in resset:
            via = "res"
        else:
            exists = loc
            names = spec_names(t.get("qname") or "")
            if exists and ro["calleeName"] in names:
                via = "nm"
            elif not exists:
                why = "no Joern METHOD at the truth location"
            else:
                why = "name differs: calleeName %r not among the symbol's short names %r" % (ro["calleeName"], sorted(names))
        members.append({"qname": t.get("qname"), "path": t["path"], "defLine": t["defLine"], "role": t["role"], "via": via, "why": why})
    found = bool(T) and all(m["via"] for m in members)
    tset = tkeys
    resolved = bool(ans["resolvedSymbols"])
    correct = resolved and resset <= tset
    edges = (sum(1 for m in members if m["via"]), len(members))
    return {"found": found, "members": members, "resolved": resolved, "correct": correct, "edges": edges,
            "resolvedToDeclOnly": resolved and bool(resset & tset) and all(m["role"] == "decl" for m, tk in zip(T, [locate(M, t, off) or (t["path"], t["defLine"] + off) for t in T]) if tk in resset)}


def config_metrics(lang, per, order, off):
    """per: ordinal -> dict(ro, tr, ans, j). order: scored ordinals. returns the metrics block."""
    internal = [o for o in order if per[o]["tr"]["T"]]
    allsc = order
    found = [o for o in internal if per[o]["j"]["found"]]
    resolved = [o for o in allsc if per[o]["j"]["resolved"]]
    correct = [o for o in resolved if per[o]["j"]["correct"]]
    away = [o for o in internal if (not per[o]["ans"]["resolvedSymbols"]) and (per[o]["ans"]["resolvedExternal"] or per[o]["ans"]["resolvedVendored"]) and not per[o]["ans"]["resolvedNonSymbol"]]
    nonsym = [o for o in allsc if (not per[o]["ans"]["resolvedSymbols"]) and per[o]["ans"]["resolvedNonSymbol"]]
    decl_only = [o for o in resolved if per[o]["j"]["resolvedToDeclOnly"]]
    te = sum(per[o]["j"]["edges"][1] for o in internal)
    fe = sum(per[o]["j"]["edges"][0] for o in internal)
    return {
        "internalSites": len(internal), "scoredSites": len(allsc),
        "recall": prop(len(found), len(internal)),
        "edgeRecall": prop(fe, te),
        "precision": prop(len(correct), len(resolved)),
        "resolvedAway": prop(len(away), len(internal)),
        "resolvedToNonSymbol": len(nonsym),
        "resolvedToDeclarationOnly": prop(len(decl_only), len(resolved)),
        "resolvedToVendoredSites": sum(1 for o in allsc if per[o]["ans"]["resolvedVendored"]),
        "matchedLabelled": prop(sum(1 for o in allsc if per[o]["ans"]["matched"]), len(allsc)),
        "foundSetOrdinals": found, "internalOrdinals": internal, "resolvedOrdinals": resolved, "correctOrdinals": correct,
        "internalMissOrdinals": [o for o in internal if o not in set(found)],
        "wrongResolvedOrdinals": [o for o in resolved if o not in set(correct)],
    }


def gate_recall(m, label="§8.1"):
    n = m["internalSites"]
    k = m["recall"]["k"]
    r = (k / n) if n else None
    verdict = "insufficient (n < 100)" if n < 100 else ("pass (observed >= 0.98)" if r >= 0.98 else "fail")
    return {"n": n, "k": k, "observedRecall": None if r is None else round(r, 4), "verdict": verdict, "maxMissesAt98": (n * 2) // 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    lang = a.lang
    off = METHOD_LINE_OFFSET[lang]
    priv = {r["ordinal"]: r for r in read_jsonl(HERE / "sample" / f"{lang}.private.jsonl")}
    truth = {r["ordinal"]: r for r in read_jsonl(HERE / "truth" / f"{lang}.final.jsonl")}
    runrec = json.loads((WORK / lang / "s27-run.json").read_text())
    Ms = {c: load_measure(lang, c) for c in ("off", "on")}
    if not all(Ms.values()):
        sys.exit(f"{lang}: missing measure output for {[c for c, v in Ms.items() if not v]}")
    order_all = sorted(truth)
    scored = [o for o in order_all if truth[o]["scored"]]
    unscored = [o for o in order_all if not truth[o]["scored"]]
    rows = [priv[o] for o in order_all]

    per_cfg, metrics = {}, {}
    for c, M in Ms.items():
        answers = analyse(lang, c, M, rows, off)
        per = {}
        for o in order_all:
            if truth[o]["scored"]:
                per[o] = {"ro": priv[o], "tr": truth[o], "ans": answers[o]}
                per[o]["j"] = judge(priv[o], truth[o], answers[o], M, off)
        per_cfg[c] = (answers, per)
        metrics[c] = config_metrics(lang, per, scored, off)

    # T mapping diagnostic (how many T members have a Joern METHOD at defLine + offset), per config
    tmap = {}
    for c, M in Ms.items():
        tot = hit = 0
        for o in scored:
            for t in truth[o]["T"]:
                tot += 1
                hit += 1 if locate(M, t, off) else 0
        tmap[c] = {"tMembers": tot, "withJoernMethodAtDefLine": hit}

    # paired off vs on (recall over internal sites)
    internal = metrics["off"]["internalOrdinals"]
    fo, fn = set(metrics["off"]["foundSetOrdinals"]), set(metrics["on"]["foundSetOrdinals"])
    a_ = len([o for o in internal if o in fo and o in fn])
    b_ = len([o for o in internal if o not in fo and o in fn])   # on-only
    c_ = len([o for o in internal if o in fo and o not in fn])   # off-only
    d_ = len(internal) - a_ - b_ - c_
    paired = {"bothFound": a_, "foundOnMissedOff": b_, "foundOffMissedOn": c_, "neither": d_,
              "mcnemarExactP": round(mcnemar_exact(b_, c_), 6), "newcombe": newcombe_paired(a_, b_, c_, d_)}

    # fan-out (census): every matched unresolved frame site
    fan = {}
    for c, M in Ms.items():
        vals = []
        for k, s in M["sites"].items():
            if s["matched"]:
                symres = [t for t in s["res"] if not t["ext"] and not t["vendored"] and not NONSYMBOL_NAME.match((M["methods"].get(t["file"], {}).get(t["l"]) or {"name": ""})["name"])]
                if not symres:
                    vals.append(s["nm"])
        fan[c] = {"matchedUnresolvedSites": len(vals), "median": nearest_rank(vals, 1, 2), "p90": nearest_rank(vals, 9, 10),
                  "p99": nearest_rank(vals, 99, 100), "max": max(vals) if vals else None,
                  "shareOver20": prop(sum(1 for v in vals if v > 20), len(vals))}
    census = {}
    for c, M in Ms.items():
        sites = M["sites"]
        mt = sum(1 for s in sites.values() if s["matched"])
        frame_files = {json.loads(l)["path"] for l in open(HERE / "frame" / f"{lang}.jsonl")}
        have = [f for f in frame_files if f in M["files"]]
        withm = [f for f in have if M["files"][f] > 0]
        census[c] = {"frameSites": len(sites), "matchRate": prop(mt, len(sites)), "frameFiles": len(frame_files),
                     "frameFilesWithFileNode": prop(len(have), len(frame_files)),
                     "frameFilesWithFileNodeAndMethod": prop(len(withm), len(frame_files)),
                     "meta": M["meta"]}

    # strata (python, javascript)
    strata = None
    cj = json.loads((HERE / "contact.json").read_text())["corpora"].get(lang)
    if cj:
        cfiles = set(cj["files"])
        strata = {"contactFiles": sorted(cfiles)}
        for name, pred in (("contact", lambda o: truth[o]["site"].rsplit(":", 1)[0] in cfiles),
                           ("nonContact", lambda o: truth[o]["site"].rsplit(":", 1)[0] not in cfiles)):
            blk = {}
            for c in ("off", "on"):
                inn = [o for o in metrics[c]["internalOrdinals"] if pred(o)]
                fd = [o for o in inn if o in set(metrics[c]["foundSetOrdinals"])]
                allsc = [o for o in scored if pred(o)]
                rs = [o for o in allsc if per_cfg[c][1][o]["j"]["resolved"]]
                cr = [o for o in rs if per_cfg[c][1][o]["j"]["correct"]]
                n = len(inn)
                blk[c] = {"internalSites": n, "recall": prop(len(fd), n), "precision": prop(len(cr), len(rs)),
                          "gate81": ("insufficient (< 50 internal in the non-contact stratum)" if (name == "nonContact" and n < 50) else
                                     (None if name == "contact" else ("pass (observed >= 0.98, n >= 50)" if n and len(fd) / n >= 0.98 else "fail")))}
            blk["scoredSites"] = len([o for o in scored if pred(o)])
            strata[name] = blk
        strata["note"] = "provisional (final.jsonl); S27 reports final2. The contact-file test is the site's path (final.jsonl `site`, path:line) in contact.json files."

    # answers + misses
    def answer_row(o):
        ro, tr = priv[o], truth[o]
        row = {"ordinal": o, "siteKey": ro["siteKey"], "calleeName": ro["calleeName"], "kind": ro["kind"], "scored": tr["scored"],
               "internal": bool(tr["T"]) and tr["scored"], "T": [{"path": t["path"], "defLine": t["defLine"], "role": t["role"], "qname": t.get("qname")} for t in tr["T"]]}
        for c in ("off", "on"):
            an = per_cfg[c][0][o]
            r = {"matched": an["matched"], "nm": an["nm"], "call": an["call"], "res": an["res"]}
            if tr["scored"]:
                j = per_cfg[c][1][o]["j"]
                r.update({"found": j["found"], "resolved": j["resolved"], "correct": j["correct"]})
            row[c] = r
        return row

    answers = [answer_row(o) for o in order_all]
    miss_rows = []
    for o in scored:
        if not truth[o]["T"]:
            continue
        mo = o in set(metrics["off"]["internalMissOrdinals"])
        mn = o in set(metrics["on"]["internalMissOrdinals"])
        if not (mo or mn):
            continue
        tr, ro = truth[o], priv[o]
        row = {"ordinal": o, "siteId": tr["siteId"], "site": tr["site"], "siteKey": ro["siteKey"], "calleeName": ro["calleeName"], "kind": ro["kind"],
               "missedIn": [c for c, f in (("off", mo), ("on", mn)) if f],
               "truth": {k: tr[k] for k in ("T", "N", "external", "dynamic", "basis", "strata", "evidence")},
               "proposedCause": "", "decision": None}
        for c in ("off", "on"):
            an = per_cfg[c][0][o]
            j = per_cfg[c][1][o]["j"]
            row[c] = {"matched": an["matched"], "joernCall": an["call"], "resolvedSymbols": an["resolvedSymbols"],
                      "resolvedExternal": an["resolvedExternal"], "resolvedVendored": an["resolvedVendored"], "resolvedNonSymbol": an["resolvedNonSymbol"],
                      "nm": an["nm"], "nmList": an["nmList"], "found": j["found"], "tMembers": j["members"]}
        miss_rows.append(row)

    res = {
        "lang": lang, "provisional": True, "truth": f"truth/{lang}.final.jsonl",
        "labelledSites": len(order_all), "scoredSites": len(scored), "unscoredSites": len(unscored),
        "unscoredReasons": dict(collections.Counter(truth[o]["notScoredReason"] for o in unscored)),
        "internalSites": metrics["off"]["internalSites"], "externalScoredSites": len(scored) - metrics["off"]["internalSites"],
        "dynamicWithT": sum(1 for o in scored if truth[o]["dynamic"] and truth[o]["T"]),
        "off": {k: v for k, v in metrics["off"].items() if not k.endswith("Ordinals")},
        "on": {k: v for k, v in metrics["on"].items() if not k.endswith("Ordinals")},
        "gate81": {"off": gate_recall(metrics["off"]), "on": gate_recall(metrics["on"])},
        "gate83": {c: {"nP": metrics[c]["precision"]["n"], "P": metrics[c]["precision"]["p"],
                       "verdict": ("unmeasured (n_P < 60)" if metrics[c]["precision"]["n"] < 60 else ("adopted on precision (P < 0.95)" if metrics[c]["precision"]["p"] < 0.95 else "pass (P >= 0.95)"))} for c in ("off", "on")},
        "pairedOffOn": paired, "fanOut": fan, "census": census, "strata": strata,
        "tMemberMapping": tmap, "misses": {"off": len(metrics["off"]["internalMissOrdinals"]), "on": len(metrics["on"]["internalMissOrdinals"]),
                                           "either": len(miss_rows), "offOnly": sum(1 for r in miss_rows if r["missedIn"] == ["off"]),
                                           "onOnly": sum(1 for r in miss_rows if r["missedIn"] == ["on"]), "both": sum(1 for r in miss_rows if len(r["missedIn"]) == 2)},
        "unmatchedLabelled": {c: [{"ordinal": o, "siteKey": priv[o]["siteKey"], "kind": priv[o]["kind"], "internal": bool(truth[o]["T"])}
                                  for o in scored if not per_cfg[c][0][o]["matched"]] for c in ("off", "on")},
        "runs": runrec["runs"], "toolchainImage": runrec.get("toolchainImage"),
        "measureFiles": {c: {"path": Ms[c]["path"], "sha256": hashlib.sha256(Path(Ms[c]["path"]).read_bytes()).hexdigest()} for c in Ms},
        "definitions": __doc__,
    }
    if not a.no_write:
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / f"{lang}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
        with open(HERE / "results" / f"{lang}.answers.jsonl", "w") as fh:
            for r in answers:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(HERE / "adjudication" / f"{lang}-misses.jsonl", "w") as fh:
            for r in miss_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    brief = {k: res[k] for k in ("lang", "labelledSites", "scoredSites", "unscoredSites", "internalSites", "misses", "gate81", "gate83", "pairedOffOn", "tMemberMapping")}
    brief["recall"] = {c: res[c]["recall"] for c in ("off", "on")}
    brief["precision"] = {c: res[c]["precision"] for c in ("off", "on")}
    brief["matched"] = {c: res[c]["matchedLabelled"] for c in ("off", "on")}
    brief["strata"] = strata and {n: strata[n] for n in ("contact", "nonContact")}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
