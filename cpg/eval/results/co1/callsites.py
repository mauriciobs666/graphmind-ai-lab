"""CO-1: read the REAL extractor output (the build's bundles, $CO1_BUNDLES) for every dev impacted element of the scripts stratum:
call sites of the impacted symbol on/near the gold lines, their status/reason, and each argument valueSet. Dev gold only. Output: callsites.json"""
import json, os, sys, glob
from pathlib import Path
REPO = Path(os.environ["P1T_REPO"]); B = Path(os.environ["CO1_BUNDLES"]); OUT = Path(os.environ["P1T_OUT"])
sym, cs_by_sym, vs = {}, {}, {}
for f in glob.glob(str(B / "*.bundle.jsonl")):
    for l in open(f):
        r = json.loads(l); t = r.get("type")
        if t == "symbol": sym[r["id"]] = r
        elif t == "callsite": cs_by_sym.setdefault(r["symbolId"], []).append(r)
        elif t == "valueSet": vs.setdefault(r["callSiteId"], {})[r["arg"]] = r["values"]
by_q = {s["qname"]: s for s in sym.values()}
recs = [json.loads(l) for l in (REPO / "cpg/eval/dev/ac5-scripts.jsonl").read_text().splitlines() if l.strip()]
res = []
for r in recs:
    im = r["impacted"]; s = by_q.get(im["qname"]); a = r["anchor"]
    row = {"itemId": r["itemId"], "anchor": a.get("qname") or a.get("keyNorm"), "impacted": im["qname"], "lines": r["lines"], "symbolFound": bool(s),
           "lang": s["lang"] if s else None, "sites": []}
    if s:
        for c in cs_by_sym.get(s["id"], []):
            near = any(abs(c["line"] - L) <= 1 for L in r["lines"])
            if near:
                row["sites"].append({"line": c["line"], "callee": c["calleeName"], "kind": c.get("callKind"), "status": c["status"], "reason": c["reason"],
                                     "args": vs.get(c["id"], {})})
        row["nSitesInSymbol"] = len(cs_by_sym.get(s["id"], []))
    res.append(row)
(OUT / "callsites.json").write_text(json.dumps(res, indent=1, default=str))
print(len(res), "rows;", sum(1 for x in res if x["symbolFound"]), "impacted symbols found in bundles;", sum(1 for x in res if x["sites"]), "with a site within 1 line of gold")
