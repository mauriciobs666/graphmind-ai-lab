#!/usr/bin/env python3
"""Part F draw for cpp. Phase 'draw': reads labels/cpp-flow.jsonl STRUCTURE ONLY (ordinal, pair index,
confidence) and writes cpp-flow-audit.input.json. Prints only counts + the six method ordinals."""
import hashlib, json, sys

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF = (_CPG+"/spikes/frontend-fitness")
def h(s): return hashlib.sha256(s.encode()).hexdigest()
def main():
    methods = []; pairs = []
    for line in open(f"{FF}/labels/cpp-flow.jsonl"):
        m = json.loads(line); o = m["ordinal"]; methods.append(o)
        for i, p in enumerate(m["pairs"]):
            pairs.append((o, i, p["confidence"] == "unsure"))
    key = lambda o, i: h(f"ccg-p0a-v1-flowaud|cpp|pair|{o}|{i}")
    uns = sorted([p for p in pairs if p[2]], key=lambda p: key(p[0], p[1]))[:20]
    rest = sorted([p for p in pairs if (p[0], p[1]) not in {(u[0], u[1]) for u in uns}], key=lambda p: key(p[0], p[1]))
    f1 = uns + rest[:60 - len(uns)]
    mk = lambda o: h(f"ccg-p0a-v1-flowaud|cpp|method|{o}")
    ms = sorted(methods, key=mk)
    out = {"lang": "cpp", "hashInput": {"pair": "ccg-p0a-v1-flowaud|cpp|pair|<methodOrdinal>|<pairIndex>",
            "method": "ccg-p0a-v1-flowaud|cpp|method|<methodOrdinal>", "pairIndex": "0-based index in the method's pairs array"},
           "nMethods": len(methods), "nPairs": len(pairs), "nUnsureAll": sum(p[2] for p in pairs),
           "f1": [{"ordinal": o, "pairIndex": i, "unsure": u, "hash": key(o, i)} for o, i, u in f1],
           "f2": {"drawn": [{"ordinal": o, "hash": mk(o)} for o in ms[:6]], "extension": [{"ordinal": o, "hash": mk(o)} for o in ms[6:9]]}}
    json.dump(out, open(f"{FF}/adjudication/cpp-flow-audit.input.json", "w"), indent=1)
    print("methods", len(methods), "pairs", len(pairs), "unsureAll", out["nUnsureAll"], "f1", len(f1), "f1unsure", len(uns))
    print("F2 methods", [o for o in ms[:6]])
main()
