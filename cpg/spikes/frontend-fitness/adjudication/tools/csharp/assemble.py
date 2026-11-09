import json, collections, math
from pathlib import Path

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF = Path((_CPG+"/spikes/frontend-fitness")); A = FF/"adjudication"
rj = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def wil(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [round(max(0, c-h), 4), round(min(1, c+h), 4)]
inp = {r["ordinal"]: r for r in rj(A/"csharp.input.jsonl")}
ref = rj(FF/"truth/csharp.ref.jsonl"); queue = rj(A/"csharp.queue.jsonl"); dec = rj(A/"csharp.decisions.jsonl")
labels = {r["ordinal"]: r for r in rj(FF/"labels/csharp.jsonl")}
problems = []
key = lambda x: (x["category"], x["file"] if x["category"]=="census" else x["ordinal"]) if x["category"]!="census" else ("census", f'{x["path"]}:{x["defLine"]}') if "path" in x else ("census", x["file"])
qk = collections.Counter(key(x) for x in queue); dk = collections.Counter(key(x) for x in dec)
if qk != dk: problems.append(f"queue/decision key sets differ: only-queue {list((qk-dk))[:5]} only-dec {list((dk-qk))[:5]}")
print("keys queue", sum(qk.values()), "dec", sum(dk.values()), "equal", qk == dk)
D = collections.defaultdict(dict)
for x in dec: D[x["category"]][key(x)[1]] = x
census = {(x["path"], x["defLine"]): x for x in dec if x["category"]=="census"}

# --- decisions file (adjudicated, overturned, swept sites): the 44 needs-adjudication + audit rows with errors (none)
def dline(x, kind):
    return {"ordinal": x["ordinal"], "siteId": x["siteId"], "reasons": inp[x["ordinal"]]["reasons"] if kind=="adjudicated" else [kind],
            "callees": x["decided"], "nonSymbol": x["nonSymbol"], "notACall": x["notACall"], "dynamic": x["dynamic"],
            "wrong": x["wrong"], "errorKind": x["errorKind"], "evidence": x["evidence"], "note": x.get("note"), "basis": x["basis"]}
with open(A/"csharp.jsonl", "w", encoding="utf-8") as f:
    for o, x in sorted(D["needs-adjudication"].items()): f.write(json.dumps(dline(x, "adjudicated"), ensure_ascii=False)+"\n")
    for c in ("audit-U","audit-C","genaudit"):
        for o, x in sorted(D[c].items()):
            if x["wrong"] != "neither" or x["errorKind"] != "none" or not x.get("labelCorrect", True) or x.get("siblingSweep"):
                f.write(json.dumps(dline(x, c), ensure_ascii=False)+"\n")
print("csharp.jsonl lines", sum(1 for _ in open(A/"csharp.jsonl")))

# --- audit.jsonl outcomes
aud = rj(A/"csharp.audit.jsonl"); gen = rj(A/"csharp.gen-audit.jsonl")
out = []
for r in aud:
    x = D["audit-"+r["draw"]][r["ordinal"]]
    r = dict(r); r.update(labelCorrect=x["labelCorrect"], decision=x["decision"], wrong=x["wrong"], errorKind=x["errorKind"], evidence=x["evidence"], note=x.get("note")); out.append(r)
for g in gen:
    x = D["genaudit"][g["ordinal"]]
    out.append({"stratum": "gen-"+g["stratum"], "draw": "GEN", "rank": g["rank"], "ordinal": g["ordinal"], "siteId": g["siteId"], "hash": g["hash"],
                "classes": g["classes"], "labelCorrect": x["labelCorrect"], "decision": x["decision"], "wrong": x["wrong"], "errorKind": x["errorKind"], "evidence": x["evidence"], "note": x.get("note")})
with open(A/"csharp.audit.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print("audit rows", len(out))

# --- final
def stratum_of(r):
    if r["status"] != "agreed": return None
    if r["inCorpusLabel"] or r["declaredLabel"]: return "AI"
    return "AS" if r["reference"]["static"]["kind"] == "silent" else "AE"
final = []; basis_c = collections.Counter(); unscoredR = collections.Counter()
for o in sorted(inp):
    r = inp[o]; lab = labels[o]; st = stratum_of(r); tags = []
    if r["status"]=="agreed": tags.append(st)
    else: tags.append("needs-adjudication")
    nonrefl = [("U", D["audit-U"].get(o)), ("C", D["audit-C"].get(o)), ("gen", D["genaudit"].get(o))]
    for n, x in nonrefl:
        if x: tags.append("audit-"+n if n != "gen" else "genaudit")
    if r["reference"].get("byConstruction"): tags.append("byConstruction")
    if r["declaredLabel"]: tags.append("declared")
    adj = D["needs-adjudication"].get(o)
    if adj: src = adj; basis = "adjudicated"
    elif st == "AS": src = D["audit-U"].get(o) or D["audit-C"].get(o) or D["genaudit"].get(o); basis = "label-only"
    else:
        src = D["audit-U"].get(o) or D["audit-C"].get(o) or D["genaudit"].get(o)
        basis = "audit-confirmed" if src else "agreed"
    if src:
        T = [{"path": t["path"], "qname": t["qname"], "defLine": t["defLine"], "role": t["role"]} for t in src["decided"]]
        N, nac, dyn, sc, nsr = src["nonSymbol"], src["notACall"], src["dynamic"], src["scored"], src["notScoredReason"]
        ev = list(src["evidence"])
    else:
        T = [{"path": c["path"], "qname": c["qname"], "defLine": c["defLine"], "role": census[(c["path"], c["defLine"])]["role"]} for c in lab["callees"] if c["kind"] in ("internal","declared")]
        N, nac, dyn = [], lab["notACall"], lab["dynamic"]; sc = not nac and not (dyn and not T); nsr = "notACall" if nac else None
        ev = [f'{r["site"]["path"]}:{r["site"]["line"]}'] + [f'{t["path"]}:{t["defLine"]}' for t in T]
    ext = sorted({c["package"] for c in lab["callees"] if c["kind"]=="external"})
    final.append({"ordinal": o, "siteId": r["siteId"], "T": T, "N": N, "external": ext, "notACall": nac, "dynamic": dyn, "scored": sc,
                  "notScoredReason": nsr, "basis": basis, "strata": tags, "evidence": ev})
    basis_c[basis] += 1
with open(FF/"truth/csharp.final.jsonl", "w", encoding="utf-8") as f:
    for x in final: f.write(json.dumps(x, ensure_ascii=False)+"\n")
print("final", len(final), dict(basis_c))

# --- noise
N = len(inp); sil = [o for o in inp if inp[o]["reference"]["static"]["kind"]=="silent"]
comp = N-len(sil); agreed_c = sum(1 for o in inp if inp[o]["status"]=="agreed" and o not in sil)
AS = [o for o in inp if stratum_of(inp[o])=="AS"]
def lab_kind(o):
    ks = {c["kind"] for c in labels[o]["callees"]}
    return "declared" if "declared" in ks else "internal" if "internal" in ks else "notACall" if labels[o]["notACall"] else "external"
groups = collections.defaultdict(lambda: [0,0])
for o in inp:
    if o in sil: continue
    g = lab_kind(o); groups[g][1] += 1; groups[g][0] += inp[o]["status"]=="agreed"
# strata for audit table
pop = {"AI": 106, "AE": 120, "AS": 43}; nS = {"AI": 60, "AE": 60, "AS": 30}
au = {}
for s in pop:
    u = [x for x in D["audit-U"].values() if x["stratum"]==s]; c = [x for x in D["audit-C"].values() if x["stratum"]==s]
    k = sum(1 for x in u if not x["labelCorrect"]); kc = sum(1 for x in c if not x["labelCorrect"])
    au[s] = {"N_s": pop[s], "U_n": len(u), "U_errors": k, "U_wilson": wil(k, len(u)), "U_wilsonUpper": wil(k, len(u))[1],
             "C_n": len(c), "C_errors": kc, "C_note": "reported apart, never pooled"}
gen_int = [x for x in D["genaudit"].values() if x["stratum"]=="internal"]; gen_ext = [x for x in D["genaudit"].values() if x["stratum"]=="external"]
W = sum(1 for x in D["needs-adjudication"].values() if x["wrong"] in ("label","both"))
Wr = sum(1 for x in D["needs-adjudication"].values() if x["wrong"] in ("reference","both"))
role_err = sum(1 for x in census.values() if x["labelError"])
point = (W + sum(au[s]["N_s"]*au[s]["U_errors"]/au[s]["U_n"] for s in au))/N
cons = (W + sum(au[s]["N_s"]*au[s]["U_wilsonUpper"] for s in au))/N
# sample-level adjusted coverage from ref.jsonl (500 rows)
refs = ref; wr = sum(1 for r in refs if r["static"]["kind"]!="silent")
excl = sum(1 for r in refs if r["static"]["kind"]=="silent" and r.get("byConstruction"))
den = len(refs)-excl
byc = [o for o in inp if inp[o]["reference"].get("byConstruction")]
# class-line points
cls = [x for x in census.values() if x["form"].startswith("type declaration")]
cls_sites = sorted({s for x in cls for s in x["sites"]})
noise = {"lang": "csharp", "N": N, "comparable": {"n": comp, "referenceSilent": len(sil), "silentAS": len(AS), "silentNotAS": len(sil)-len(AS)},
  "agreement": {"agreed": agreed_c, "comparable": comp, "rate": round(agreed_c/comp,4), "wilson95": wil(agreed_c, comp),
                "byLabelKind": {g: {"agreed": v[0], "n": v[1], "rate": round(v[0]/v[1],4), "wilson95": wil(*v)} for g, v in sorted(groups.items())}},
  "adjudicated": {"n": len(D["needs-adjudication"]), "byWrong": dict(collections.Counter(x["wrong"] for x in D["needs-adjudication"].values())),
                  "byErrorKind": dict(collections.Counter(x["errorKind"] for x in D["needs-adjudication"].values())), "W_labelWrong": W, "referenceWrong": Wr,
                  "referenceWrongOverComparable": {"k": Wr, "n": comp, "rate": round(Wr/comp,4), "wilson95": wil(Wr, comp)}},
  "audit": au,
  "genericAudit": {"external": {"n": len(gen_ext), "errors": sum(1 for x in gen_ext if not x["labelCorrect"]), "wilson95": wil(sum(1 for x in gen_ext if not x["labelCorrect"]), len(gen_ext)), "note": "only 33 labelled-external sites exist in the pool; 40 not reachable"},
                   "internal": {"n": len(gen_int), "errors": sum(1 for x in gen_int if not x["labelCorrect"]), "wilson95": wil(sum(1 for x in gen_int if not x["labelCorrect"]), len(gen_int))},
                   "flag": "see errors above vs ML 5.4 thresholds (>=3 external errors, or internal rate >0.05)", "internalShareOfLabelledInternal": None},
  "handErrorRate": {"definition": "(W + sum N_s*k_s/n_s)/N, draw U only", "point": round(point,6), "conservative_wilsonUpperPerStratum": round(cons,6),
                    "roleErrors": role_err, "withRoleErrors": {"point": round((W+role_err+sum(au[s]["N_s"]*au[s]["U_errors"]/au[s]["U_n"] for s in au))/N,6), "conservative": round((W+role_err+sum(au[s]["N_s"]*au[s]["U_wilsonUpper"] for s in au))/N,6)},
                    "withoutRoleErrors": {"point": round(point,6), "conservative": round(cons,6)}, "noiseLimitedThreshold": 0.05,
                    "noiseLimited_point": point > 0.05, "noiseLimited_conservative": cons > 0.05},
  "byConstruction": {"sites": len(byc), "comparable": sum(1 for o in byc if o not in sil), "referenceSilent": sum(1 for o in byc if o in sil)},
  "sampleAdjustedCoverage": {"sampleRows": len(refs), "withReference": wr, "excludedByConstruction": excl, "denominator": den, "coverage": round(wr/den,4), "wilson95": wil(wr, den)},
  "census": {"entries": len(census), "labelErrors": role_err, "byRole": dict(collections.Counter(x["role"] for x in census.values())), "fcFormDisagreements": sum(1 for x in census.values() if not x["fcFormAgrees"])},
  "unscored": dict(collections.Counter(x["notScoredReason"] for x in final if not x["scored"])), "underivable": sum(1 for x in final if x["notScoredReason"]=="underivable"),
  "openPoint_for_S27_not_decided_here": {"text": "labelled sites whose decided target is a type/class line (new X() with no explicit constructor); S27 to map consistently",
                                           "definitions": [f'{x["path"]}:{x["defLine"]}' for x in cls], "sites": cls_sites, "nTypeDeclarationDefinitions": len(cls)}}
json.dump(noise, open(A/"csharp.noise.json","w"), indent=1)
print(json.dumps({k: noise[k] for k in ("comparable","agreement","genericAudit","handErrorRate","sampleAdjustedCoverage","byConstruction","census","unscored","underivable","openPoint_for_S27_not_decided_here")}, indent=1))
print(json.dumps(au, indent=0))
print(noise["adjudicated"])
print("PROBLEMS", problems)
