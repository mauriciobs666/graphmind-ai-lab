"""CO-1: renders the per-item tables of the report from classification.json + co1-asks.json (no new measurement)."""
import json, os
from pathlib import Path
OUT = Path(os.environ["P1T_OUT"])
rows = json.load(open(OUT / "classification.json")); asks = json.load(open(OUT / "co1-asks.json"))
FAM = {}
for fam, idx in {"A md inline token, >=2 segments, unique": [2,16,17,21,25,27,29,31,39,40,41,42,44], "B md bare basename only": [33,37], "C md front-matter hook": [1,28],
                 "D shell `.` arg `$(dirname)/shlib/x`": [7,9,11,14], "E shell exec arg `../../scripts/x`": [4,5,10,22,26], "F shell literal command word `./scripts/x.sh`": [15],
                 "G shell computed command word (path unrecorded)": [3,8,13,19,20,30,36,45,46,48], "H shell `$HERE/x` argument, 1-segment tail": [32,43],
                 "I parse defect / path inside `bash -c` string / file-read": [0,34,47], "J `-m` package": [18,35], "K Python subprocess argument": [23],
                 "L shell curl URL (http contract)": [6,12,38], "M embedded Python MCP call in shell (mcp contract)": [24]}.items():
    for i in idx: FAM[i] = fam
assert sorted(FAM) == list(range(49))
def carry(r):
    ns = r.get("namingSites", [])
    if not ns:
        return "none"
    b = max(ns, key=lambda s: s.get("tailSegments", 0))
    t = b.get("text") or b.get("tail")
    return f"{'callee' if b['where']=='calleeName' else 'valueSet'} `{t}` ({b['tailSegments']} seg, {b['snapshotSuffixMatches']} snapshot match)"
def short(q): return q.split("::")[0].rsplit("/", 1)[-1] + ("::" + q.split("::")[1] if "::" in q else "")
L = ["| # | item | anchor | impacted element | family | record carries | now | why it misses (the answer for the anchor) |", "|---|---|---|---|---|---|---|---|"]
sa = asks["strata"]["scripts"]["anchorAnswers"]
for r in [x for x in rows if x["stratum"] == "scripts"]:
    aid = next(k for k, v in sa.items() if (v["anchor"].get("qname") or v["anchor"].get("keyNorm")) == r["anchor"])
    a = sa[aid]; n = len(a["ask"]["items"])
    why = (f"answer lists {n} items, none is this element" if r["anchorType"] == "contract" else f"answer lists 0 items; anchor symbol present; {a.get('callsEdgesIn')} CALLS edges in")
    L.append(f"| {r['idx']} | {r['itemId'][:6]} | {short(r['anchor'])} | {short(r['impacted'])} | {FAM[r['idx']]} | {carry(r)} | {'hit' if r['hit'] else 'miss'} | {why} |")
open(OUT / "table-scripts.md", "w").write("\n".join(L) + "\n")
H = ["| # | item | anchor | impacted element | mechanism | now | status/distance or reason |", "|---|---|---|---|---|---|---|"]
for r in [x for x in rows if x["stratum"] == "http"]:
    ar = asks["strata"]["http"]["rows"][r["idx"]]
    H.append(f"| {r['idx']} | {r['itemId'][:6]} | {r['anchor']} | {short(r['impacted'])} | {r['mechanism']} | {'hit' if r['hit'] else 'miss'} | " + (f"{ar['status']}, distance {ar['distance']}" if r["hit"] else "answer lists 10 items; the Markdown `<document>` is not among them (its `curl` call site has no valueSet)") + " |")
open(OUT / "table-http.md", "w").write("\n".join(H) + "\n")
W = ["| # | item | anchor type | config element | mechanism | askable | now | store (ConfigEntry under keyPath / Link BOUND_BY it) | answer |", "|---|---|---|---|---|---|---|---|---|"]
wp = json.load(open(OUT / "wiring-probe.json"))["items"]
for r in [x for x in rows if x["stratum"] == "wiring"]:
    p = wp[r["idx"]]; ar = asks["strata"]["wiring"]["rows"][r["idx"]]
    ans = "not asked" if r["anchorType"] == "unit" else ("0 items" if r["anchorType"] == "symbol" else "22 items, 0 of type config")
    W.append(f"| {r['idx']} | {r['itemId'][:6]} | {r['anchorType']} | {r['impacted'].split('#')[0].rsplit('/',1)[-1]}#{r['impacted'].split('#')[1]} | {r['mechanism']} | {'no' if r['anchorType']=='unit' else 'yes'} | {'not asked' if r['anchorType']=='unit' else 'miss'} | {p['configEntriesUnderKeyPath']} / {p['linkBoundByUnder']} | {ans} |")
open(OUT / "table-wiring.md", "w").write("\n".join(W) + "\n")
print("ok")
