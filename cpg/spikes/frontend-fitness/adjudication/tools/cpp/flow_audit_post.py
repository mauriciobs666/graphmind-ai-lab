#!/usr/bin/env python3
"""After the relabel is on disk: F2 diff (golden vs relabel, six drawn methods) and F1 queue; plus ordinal-5 population check."""
import json, linecache

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF = (_CPG+"/spikes/frontend-fitness")
SRC = (_CPG+"/.ccg-work/corpora/cpp/src")
def ep(e):
    p = e["point"]
    return (p, e["index"]) if p == "param" else (p, e["calleeName"], e["ordinal"]) if p == "call" else (p, e["ordinal"]) if p == "ret" else (p, e["line"], e["code"])
ident = lambda x: (x["kind"], ep(x["source"]), ep(x["sink"]))
gold = {}; 
for l in open(f"{FF}/labels/cpp-flow.jsonl"):
    m = json.loads(l); gold[m["ordinal"]] = m
sample = {}
for l in open(f"{FF}/sample/cpp-flow.jsonl"):
    m = json.loads(l); sample[m["ordinal"]] = m
inp = json.load(open(f"{FF}/adjudication/cpp-flow-audit.input.json"))
rel = {}
for l in open(f"{FF}/adjudication/cpp-flow-relabel.jsonl"):
    m = json.loads(l); rel[m["ordinal"]] = m
tot = {"both": 0, "goldenOnly": 0, "relabelOnly": 0}
with open(f"{FF}/adjudication/cpp-flow-audit.diff.jsonl", "w") as w:
    for d in inp["f2"]["drawn"]:
        o = d["ordinal"]; g = {ident(p): p for p in gold[o]["pairs"]}; r = {ident(p): p for p in rel[o]["pairs"]}
        tot["both"] += len(set(g) & set(r))
        for side, a, b in (("goldenOnly", g, r), ("relabelOnly", r, g)):
            for k in a:
                if k not in b:
                    tot[side] += 1
                    w.write(json.dumps({"method": o, "qname": sample[o]["qname"], "side": side, "pair": a[k]}) + "\n")
print("F2", tot)
with open(f"{FF}/adjudication/cpp-flow-audit.queue.jsonl", "w") as w:
    for n, e in enumerate(inp["f1"]):
        o, i = e["ordinal"], e["pairIndex"]; m = gold[o]; p = m["pairs"][i]; s = sample[o]
        lines = sorted({x["line"] for x in (p["source"], p["sink"]) if "line" in x})
        src = {}
        for ln in lines:
            for q in range(ln, ln + 1):
                src[str(q)] = linecache.getline(f"{SRC}/{m['path']}", q).rstrip("\n")
        w.write(json.dumps({"n": n, "method": o, "pairIndex": i, "unsure": p["confidence"] == "unsure", "path": m["path"], "qname": m["qname"],
                            "lineStart": s["lineStart"], "lineEnd": s["lineEnd"], "pair": p, "sourceLines": src}) + "\n")
print("F1 queue", len(inp["f1"]), "unsure", sum(e["unsure"] for e in inp["f1"]))
# ordinal 5 population check
g5, s5 = gold[5], sample[5]
print("ord5 golden", g5["path"], g5["qname"]); print("ord5 sample", s5["path"], s5["qname"], s5["lineStart"], s5["lineEnd"])
print("ord5 sample source head:", repr(s5["source"][:200]))
print("same path/qname:", g5["path"] == s5["path"] and g5["qname"] == s5["qname"])
for o in gold:
    if (gold[o]["path"], gold[o]["qname"]) != (sample[o]["path"], sample[o]["qname"]): print("MISMATCH ordinal", o)
print("ord5 pair lines:", sorted({x["line"] for p in g5["pairs"] for x in (p["source"], p["sink"]) if "line" in x}))
print("ord5 sample line count in source", s5["source"].count("\n")+1, "span", s5["lineEnd"]-s5["lineStart"]+1)
