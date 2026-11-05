#!/usr/bin/env python3
"""Run only after java-flow-relabel.jsonl exists. Writes the F2 diff and the F1 queue."""
import json
FF = "/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness"
def key(p):
    return json.dumps([p["kind"], p["source"], p["sink"]], sort_keys=True)
inp = json.load(open(f"{FF}/adjudication/java-flow-audit.input.json"))
gold = {}
for l in open(f"{FF}/labels/java-flow.jsonl"):
    m = json.loads(l); gold[m["ordinal"]] = m
rel = {}
for l in open(f"{FF}/adjudication/java-flow-relabel.jsonl"):
    m = json.loads(l); rel[m["ordinal"]] = m
nd = 0; stats = {}
with open(f"{FF}/adjudication/java-flow-audit.diff.jsonl", "w") as f:
    for o in inp["F2MethodOrdinals"]:
        g = {key(p): p for p in gold[o]["pairs"]}; r = {key(p): p for p in rel[o]["pairs"]}
        stats[o] = {"golden": len(g), "relabel": len(r), "both": len(g.keys() & r.keys()),
                    "goldenOnly": len(g.keys() - r.keys()), "relabelOnly": len(r.keys() - g.keys())}
        for side, d, other in (("golden-only", g, r), ("relabel-only", r, g)):
            for k in sorted(d.keys() - other.keys()):
                p = d[k]; f.write(json.dumps({"ordinal": o, "path": gold[o]["path"], "side": side, **{x: p[x] for x in ("kind","source","sink","confidence","evidence")}}) + "\n"); nd += 1
with open(f"{FF}/adjudication/java-flow-audit.queue.jsonl", "w") as f:
    for n, d in enumerate(inp["F1"]):
        m = gold[d["ordinal"]]; p = m["pairs"][d["pairIndex"]]
        f.write(json.dumps({"draw": n, "ordinal": d["ordinal"], "pairIndex": d["pairIndex"], "path": m["path"],
                            "qname": m["qname"], "unsure": p["confidence"] == "unsure", "pair": p}) + "\n")
print(json.dumps(stats)); print("diff rows", nd)
print("F1 rows", len(inp["F1"]), "unsure", sum(1 for d in inp["F1"] if gold[d["ordinal"]]["pairs"][d["pairIndex"]]["confidence"]=="unsure"))
