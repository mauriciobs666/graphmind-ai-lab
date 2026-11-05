import os, json, sys, time
from pathlib import Path
from collections import defaultdict, Counter
FF = Path(__file__).resolve().parents[3]                      # cpg/spikes/frontend-fitness
CORP = FF.parents[1] / ".ccg-work" / "corpora"                 # cpg/.ccg-work/corpora
S = Path(os.environ.get("FLOW_WORK", "/tmp/ccg-flow-adj"))     # work dir: run/, runfix/, var/ (outside the repo)
TOOLS = Path(__file__).resolve().parent; sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(FF))
import s28_metrics as SM, s28_core as C
W = C.CPG / ".ccg-work" / "frontend-fitness" / "joern"
IGN = SM.DEFAULT_IGNORED
def stat(items):  # items: list of (ordinal, connected)
    k = sum(1 for _, c in items if c); n = len(items)
    byo = defaultdict(lambda: [0, 0])
    for o, c in items: byo[o][0] += 1; byo[o][1] += int(c)
    ci = SM.ratio_ci([tuple(v) for v in byo.values()]) if n else None
    return {"k": k, "n": n, "p": round(k / n, 4) if n else None, "wilson": SM.wilson(k, n) if n else None, "cluster": ci, "methods": len(byo)}
out = {}
for lang in ["python", "javascript", "go", "java", "csharp", "cpp"]:
    A = {f"{a['ordinal']}|{a['idx']}": a for a in map(json.loads, open(FF/"results"/f"projection-{lang}.answers.jsonl"))}
    M = {f"{r['ordinal']}|{r['pairIndex']}": r for r in map(json.loads, open(FF/"adjudication"/f"{lang}-flow-misses.adj.jsonl"))}
    SP = [json.loads(l) for l in open(FF/"adjudication"/f"{lang}-flow-spurious.adj.jsonl")]
    st = json.load(open(S/"var"/f"{lang}.status.json"))["status"]
    samp = {json.loads(l)["ordinal"]: json.loads(l) for l in open(FF/"sample"/f"{lang}-flow.jsonl")}
    ign = lambda o: bool(set(samp[o]["path"].split("/")[:-1]) & IGN.get(lang, set()))
    allp = [(a["ordinal"], a["connected"], k) for k, a in A.items()]
    res = {"asShipped": stat([(o, c) for o, c, _ in allp])}
    overt = {k for k, r in M.items() if r["adjudication"] == "overturned"}
    naming = {k for k, r in M.items() if r["adjudication"] == "endpoint-naming-error"}
    om = [r for r in SP if r["verdict"] == "omission"]
    base = [(o, c, k) for o, c, k in allp if k not in overt and k not in naming]
    adjd = [(o, c) for o, c, _ in base] + [(r["ordinal"], True) for r in om]
    res["adjudicated"] = stat(adjd)
    res["adjudicatedExclDefaultIgnored"] = stat([(o, c) for o, c in adjd if not ign(o)])
    unres = {k for k, r in M.items() if r["class"] == "endpoint-unresolved"}
    res["adjudicatedExclEndpointUnresolved"] = stat([(o, c) for o, c, k in base if k not in unres] + [(r["ordinal"], True) for r in om])
    joinun = {k for k in unres if M[k]["causeSide"] in ("join", "label")}
    res["adjudicatedExclJoinSideUnresolved"] = stat([(o, c) for o, c, k in base if k not in joinun] + [(r["ordinal"], True) for r in om])
    res["adjudicatedExclDefaultIgnoredAndEndpointUnresolved"] = stat([(o, c) for o, c, k in base if k not in unres and not ign(o)] + [(r["ordinal"], True) for r in om if not ign(r["ordinal"])])
    res["adjudicatedMatchedMethodsOnly"] = stat([(o, c) for o, c, k in base if A[k]["status"] != "method-absent"] + [(r["ordinal"], True) for r in om])
    for kind in ("data", "control"):
        res[f"adjudicated_{kind}"] = stat([(o, c) for o, c, k in base if A[k]["kind"] == kind] + [(r["ordinal"], True) for r in om if r["kind"] == kind])
    for tag in ("early-exit", "callee-value", "alias"):
        sel = [(o, c) for o, c, k in base if tag in (A[k].get("tags") or [])]
        res[f"stratum_{tag}"] = stat(sel)
        res[f"stratum_{tag}_matched"] = stat([(o, c) for o, c, k in base if tag in (A[k].get("tags") or []) and A[k]["status"] not in ("method-absent",)])
    # counterfactual projections (all pairs, no adjudication)
    for v in ("R1", "R2", "R3"):
        res[f"counterfactual_{v}"] = stat([(int(k.split("|")[0]), (st[k][v] == "connected") if v != "R3" else bool(st[k][v])) for k in A])
        res[f"counterfactual_{v}_adjudicated"] = stat([(o, ((st[k][v] == "connected") if v != "R3" else bool(st[k][v]))) for o, c, k in base] + [(r["ordinal"], True) for r in om])
    # miss causes
    res["missCauses"] = dict(Counter(r["causeSide"] for r in M.values()))
    res["missClass"] = dict(Counter(r["class"] for r in M.values()))
    res["unresolvedBySide"] = dict(Counter(M[k]["causeSide"] for k in unres))
    res["overturned"] = sorted(overt); res["namingErrors"] = sorted(naming)
    # spurious adjudication
    vc = Counter(r["verdict"] for r in SP); n = len(SP); nd = n - vc.get("duplicate-of-golden", 0); k = vc.get("omission", 0)
    res["spurious"] = {"sampled": n, "verdicts": dict(vc), "omission_all": {"k": k, "n": n, "p": round(k / n, 4) if n else None, "wilson": SM.wilson(k, n) if n else None,
                       "flag": ("insufficient" if n < 20 else (k / n > 0.05))},
                       "omission_exclDuplicates": {"k": k, "n": nd, "p": round(k / nd, 4) if nd else None, "wilson": SM.wilson(k, nd) if nd else None,
                       "flag": ("insufficient" if nd < 20 else (k / nd > 0.05))},
                       "reasons": dict(Counter(r["verdictReason"].split(":")[0] for r in SP if r["verdict"] == "spurious"))}
    # census reach from the current defs.jsonl
    s28 = json.loads((W/lang/"s28-run.json").read_text())["runs"]
    ek = SM.EXTRACT_KEYS[lang]
    F = C.load_facts(Path(s28[f"off.{ek}"]["factsFile"]), lang)
    dp = FF/"truth"/f"{lang}.defs.jsonl"
    res["censusReachNow"] = SM.census_reach(lang, F)
    res["defsFile"] = {"mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(dp))), "lines": sum(1 for _ in open(dp))}
    out[lang] = res
    print(lang, "asShipped", res["asShipped"]["k"], res["asShipped"]["n"], "adjudicated", res["adjudicated"]["k"], res["adjudicated"]["n"], res["adjudicated"]["p"], "census", res["censusReachNow"].get("k"), res["censusReachNow"].get("n"), flush=True)
    json.dump(out, open(S/"metrics.json", "w"), indent=1)
