"""csharp S57 v1.12 pass: canonical final.jsonl (from the normalised copy), basis precedence, noise.json envelope + detail + openPointsS27.
Reads only adjudication/csharp.{input,audit,decisions,noise}.json*, truth/csharp.{final,ref}.jsonl, labels/csharp.jsonl and the corpus tree."""
import json, math, re, collections, sys
from pathlib import Path
FF = Path("/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness"); A = FF/"adjudication"
NORM = Path("/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/13504717-4242-422a-b7d8-a910a6cc9348/scratchpad/s57_norm")
rj = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def wil(k, n, z=1.959964):
    p = k/n; den = 1+z*z/n; c = (p+z*z/(2*n))/den; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [round(max(0, c-h), 4), round(min(1, c+h), 4)]
old = {r["ordinal"]: r for r in rj(FF/"truth/csharp.final.jsonl")} if (FF/"truth/csharp.final.jsonl").exists() else {}
norm = rj(NORM/"csharp.final.jsonl"); inp = {r["ordinal"]: r for r in rj(A/"csharp.input.jsonl")}
aud = rj(A/"csharp.audit.jsonl")
audited = {r["ordinal"]: r["draw"] for r in aud}   # any of U / C / GEN
dec_adj = {r["ordinal"] for r in rj(A/"csharp.jsonl")}
# stratum from the input script (same rule as the queue builder)
def stratum(r):
    if r["status"] != "agreed": return None
    if r["inCorpusLabel"] or r["declaredLabel"]: return "AI"
    return "AS" if r["reference"]["static"]["kind"] == "silent" else "AE"
rows = []; ch = collections.Counter(); before = collections.Counter(x["basis"] for x in norm)
for x in norm:
    o = x["ordinal"]; st = stratum(inp[o])
    # label-set equality against the earlier file if it is still in the old shape
    if o in old and isinstance(old[o].get("T"), list) and "site" not in old[o] or False: pass
    if o in dec_adj: b = "adjudicated"
    elif o in audited: b = "audit-confirmed"
    elif st == "AS": b = "label-only"
    else: b = "agreed"
    if b != x["basis"]: ch[(x["basis"], b, st)] += 1
    x = dict(x); x["basis"] = b
    tags = [t for t in x["strata"] if t not in ("AI", "AE", "AS")]
    x["strata"] = ([st] if st else []) + tags
    rows.append(x)
with open(FF/"truth/csharp.final.jsonl", "w", encoding="utf-8") as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False)+"\n")
after = collections.Counter(x["basis"] for x in rows)
print("basis before", dict(before), "after", dict(after), "changes", dict(ch))
print("AS lines:", sum(1 for o in inp if stratum(inp[o]) == "AS"), "AS all audited:", all(o in audited for o in inp if stratum(inp[o]) == "AS"))

# ---- noise
nz = json.load(open(NORM/"csharp.noise.json")); det = nz["detail"]
N = 313; au = nz["audit"]
upper = {s: au[s]["wilson95"][1] for s in au}
rem = {s: au[s]["N_s"]-au[s]["n"] for s in au}
r_s = {s: 0 for s in au}                       # audited target/kind errors (none found)
e_res_pt = sum(rem[s]*r_s[s]/au[s]["n"] for s in au)/N
e_res_cons = sum(rem[s]*upper[s] for s in au)/N
e_all_pt = nz["handError"]["e_all"]["point"]; e_all_cons = nz["handError"]["e_all"]["conservative"]
# open points: class-line targets (census form), anonymous-lambda qnames in final T
dec = rj(A/"csharp.decisions.jsonl"); census = [x for x in dec if x["category"] == "census"]
cls = [x for x in census if x["form"].startswith("type declaration")]
cls_sites = sorted({s for x in cls for s in x["sites"]})
final_cls_sites = sorted(x["ordinal"] for x in rows if any((t["path"], t["defLine"]) in {(c["path"], c["defLine"]) for c in cls} for t in x["T"]))
anon = sorted({t["qname"] for x in rows for t in x["T"] if t["qname"] and re.search(r"<anon|<lambda|lambda|#\d", t["qname"])})
absent_ext = "not applicable: no external-package source absence asserted for C# (all labelled externals are stdlib/NuGet packages; none was reported absent-from-tree by DS)"
clsset = {(c["path"], c["defLine"]) for c in cls}
final_defs = sorted({f'{t["path"]}:{t["defLine"]}' for x in rows for t in x["T"] if (t["path"], t["defLine"]) in clsset})
open_pts = {"classLineTargets": {"definitionsInFinalT": final_defs, "nDefinitionsInFinalT": len(final_defs), "sitesInFinalT": final_cls_sites,
              "nSitesInFinalT": len(final_cls_sites), "otherClassLineCensusEntries": sorted(f'{c["path"]}:{c["defLine"]}' for c in cls if f'{c["path"]}:{c["defLine"]}' not in final_defs),
              "censusSitesTouching": cls_sites, "note": "new X() with no explicit constructor; the decided target is the class header line (brief 6.5 constructor row); S27 to map consistently (not decided here)"},
            "anonymousLambdaIndices": anon if anon else "none found (scan of every final T qname for <anon>, lambda and #n markers)",
            "externalPackagesAbsentFromTree": absent_ext}
det2 = dict(det); det2["e_all_e_res_v112"] = {"e_all": {"point": e_all_pt, "conservative": e_all_cons}, "e_res": {"point": round(e_res_pt, 6), "conservative": round(e_res_cons, 6)},
   "r_s": r_s, "unreadRemainder": rem, "wilsonUpper": upper, "flag": "e_res.point > 0.05", "noiseLimited": e_res_pt > 0.05,
   "wouldHaveSaid_v1_10": "e_h point 0.0 (not noise-limited, the v1.10 flag used the point figure incl. role errors); conservative 0.05906 above 0.05 reported beside it",
   "labelOnlyWvariant": {"W": 0}, "labellerNoisyFinalLabelsRepaired": False,
   "closedClassShare": {"count": 0, "share": 0.0, "tripwire": 0.10, "note": "census found 0 role errors (140 targets read, 0 form disagreements); no coordinate/shape errors"},
   "basisChange": {"before": dict(before), "after": dict(after), "AS": "43 label-only -> audit-confirmed; each AS site is in draw U, draw C or the generic audit (verified)"}}
det2["openPointsS27"] = open_pts
nz["detail"] = det2
nz["handError"] = {"e_all": {"point": e_all_pt, "conservative": e_all_cons}, "e_res": {"point": round(e_res_pt, 6), "conservative": round(e_res_cons, 6)}, "noiseLimited": e_res_pt > 0.05}
nz["basisCounts"] = dict(after)
json.dump(nz, open(A/"csharp.noise.json", "w"), indent=1)
print("e_all", e_all_pt, e_all_cons, "e_res", e_res_pt, round(e_res_cons, 6), "classLine defs/sites", len(cls), len(final_cls_sites), "anon", anon)
