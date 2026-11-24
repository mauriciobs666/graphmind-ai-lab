"""CO-2c offline dev measurement (co2b_dev.py pattern): every tracked *.sh/*.md file of the working tree -> real extractors -> real assembler -> real
answer builder over an in-memory reader; the 49 dev script items of cpg/eval/dev/ac5-scripts.jsonl are judged hit / not-listed by the EP identity
match (cpg/eval/results/co2/asks.py `matches`, imported logic). No store, no build, no held-out file. Prints aggregates only (no noise number).
Run from cpg/:  .venv/bin/python eval/results/co2c/co2c_dev.py OUT.json   (compare two runs with `--diff A.json B.json`)"""
import json, pathlib, subprocess, sys, tempfile
from collections import Counter

CPG = pathlib.Path(__file__).resolve().parents[3]
ROOT = CPG.parent
if sys.argv[1] == "--diff":
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    print("edges only before:", len(set(map(tuple, a["edges"])) - set(map(tuple, b["edges"]))), "only after:", len(set(map(tuple, b["edges"])) - set(map(tuple, a["edges"]))))
    print("hits before/after:", sum(a["hits"].values()), sum(b["hits"].values()), "hit set changed:", [k for k in a["hits"] if a["hits"][k] != b["hits"].get(k)])
    sys.exit(0)
sys.path[:0] = [str(CPG), str(CPG / "tests"), str(CPG / "eval/tools")]
import script_resolver_helpers as H                     # noqa: E402
import match_keys as MK                                 # noqa: E402

files = {}
for f in subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.sh", "*.md"], capture_output=True, text=True).stdout.split():
    try:
        files[f] = (ROOT / f).read_text()
    except Exception:
        pass
w = H.assemble(pathlib.Path(tempfile.mkdtemp()), files)
paths = {s["symbolId"]: s["path"] for s in w.rows.nodes["Symbol"]}
sites = {c["callSiteId"]: (c["path"], c["line"], c["calleeName"]) for c in w.rows.nodes["CallSite"]}
edges = sorted([*sites[e["site"]], paths[e["target"]], e["status"], e["reason"]] for e in H.calls_edges(w) if e["reason"] != "resolved-call")


def regions(s):
    r = s.get("ownedRegions")
    if r and not all(isinstance(x, (list, tuple)) for x in r):
        r = [[r[i], r[i + 1]] for i in range(0, len(r), 2)]
    return r


hits = {}
for line in (CPG / "eval/dev/ac5-scripts.jsonl").read_text().splitlines():
    rec = json.loads(line)
    if rec["anchor"]["type"] != "symbol":
        continue
    key = rec["itemId"]
    try:
        doc = H.ask_impact(w, rec["anchor"]["path"])
    except Exception:
        hits[key] = 0
        continue
    e = rec["impacted"]
    ok = 0
    for x in list(doc["items"]) + [m for g in doc["groups"] for m in g["members"]]:
        s = x["subject"]
        if e["type"] == "symbol" and s.get("type") == "symbol":
            try:
                if MK.h1_symbol_credit({"qname": s["qname"], "lineStart": s["lineStart"], "lineEnd": s["lineEnd"], "ownedRegions": regions(s)},
                                       {"qname": e["qname"]}, rec["lines"]) and x["status"] in ("confirmed", "possible"):
                    ok = 1
            except Exception:
                pass
    hits[key] = ok
json.dump({"edges": edges, "hits": hits}, open(sys.argv[1], "w"))
print("files", len(files), "script edges", len(edges), "dropped", dict(w.rows.dropped), "dev hits", sum(hits.values()), "of", len(hits))
print("edges by reason", dict(Counter(e[4] + "/" + e[5] for e in edges)))
