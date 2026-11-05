#!/usr/bin/env python3
"""Run AFTER go-flow-relabel.jsonl is on disk. Writes F2 diff and F1 queue.
Pair identity = (kind, source endpoint, sink endpoint); endpoint = point + (index,name | calleeName,ordinal,line | ordinal,line | line,code with whitespace collapsed)."""
import json, re
from pathlib import Path
R = Path(__file__).resolve().parents[3]
ld = lambda p: [json.loads(l) for l in (R/p).read_text().splitlines() if l.strip()]
ep = lambda e: tuple(sorted((k, re.sub(r"\s+", " ", v) if isinstance(v, str) else v) for k, v in e.items()))
key = lambda p: (p["kind"], ep(p["source"]), ep(p["sink"]))
gold = {r["ordinal"]: r for r in ld("labels/go-flow.jsonl")}
rel = {r["ordinal"]: r for r in ld("adjudication/go-flow-relabel.jsonl")}
inp = json.loads((R/"adjudication/go-flow-audit.input.json").read_text())
with open(R/"adjudication/go-flow-audit.diff.jsonl", "w") as f:
    for o in inp["f2Methods"]:
        G = {key(p): p for p in gold[o]["pairs"]}; L = {key(p): p for p in rel[o]["pairs"]}
        for k, p in G.items():
            if k not in L: f.write(json.dumps({"ordinal": o, "side": "golden-only", "pair": p}) + "\n")
        for k, p in L.items():
            if k not in G: f.write(json.dumps({"ordinal": o, "side": "relabel-only", "pair": p}) + "\n")
        print(o, "golden", len(G), "relabel", len(L), "both", len(set(G) & set(L)), "gold-only", len(set(G) - set(L)), "rel-only", len(set(L) - set(G)))
src = {}
for l in (R/"sample/go-flow.jsonl").read_text().splitlines():
    r = json.loads(l); src[r["ordinal"]] = r
def lines(o, p):
    r = src[o]; ls = r["source"].split("\n"); ns = set()
    for e in (p["source"], p["sink"]):
        if "line" in e: ns.add(e["line"])
    for m in re.finditer(r"\b(\d{1,5})\b", p.get("evidence", "")): pass
    return {n: ls[n - r["lineStart"]] for n in sorted(ns) if 0 <= n - r["lineStart"] < len(ls)}
with open(R/"adjudication/go-flow-audit.queue.jsonl", "w") as f:
    for n, d in enumerate(inp["f1"], 1):
        r = gold[d["ordinal"]]; p = r["pairs"][d["pairIndex"]]
        f.write(json.dumps({"n": n, "ordinal": d["ordinal"], "pairIndex": d["pairIndex"], "path": r["path"], "qname": r["qname"],
                            "lineStart": src[d["ordinal"]]["lineStart"], "unsure": p["confidence"] == "unsure",
                            "pair": p, "endpointLines": {str(k): v for k, v in lines(d["ordinal"], p).items()}}) + "\n")
print("queue", len(inp["f1"]))
