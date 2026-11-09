#!/usr/bin/env python3
"""Part F draw for csharp. Reads labels/csharp-flow.jsonl for STRUCTURE ONLY
(ordinal, number of pairs, confidence per pair). Prints/writes no pair content."""
import hashlib, json, sys

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
BASE = (_CPG+"/spikes/frontend-fitness/")
LANG = "csharp"; PFX = "ccg-p0a-v1-flowaud|%s|" % LANG
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
meths, pairs = [], []
for ln in open(BASE + "labels/csharp-flow.jsonl"):
    if not ln.strip(): continue
    m = json.loads(ln); o = m["ordinal"]; meths.append(o)
    for i, p in enumerate(m["pairs"]):
        pairs.append((o, i, p.get("confidence") == "unsure"))
unsure = sorted([p for p in pairs if p[2]], key=lambda p: h(PFX + "pair|%d|%d" % (p[0], p[1])))
rest = sorted([p for p in pairs if not p[2]], key=lambda p: h(PFX + "pair|%d|%d" % (p[0], p[1])))
# all unsure first up to 20, then the remainder (unsure beyond 20 compete with sure) by hash
tail = sorted(unsure[20:] + rest, key=lambda p: h(PFX + "pair|%d|%d" % (p[0], p[1])))
f1 = unsure[:20] + tail[:60 - len(unsure[:20])]
f2 = sorted(meths, key=lambda o: h(PFX + "method|%d" % o))[:6]
out = {"lang": LANG, "hashPrefix": PFX,
       "keyForm": {"pair": PFX + "pair|<ordinal>|<pairIndex0>", "method": PFX + "method|<ordinal>"},
       "population": {"methods": len(meths), "pairs": len(pairs), "unsure": len(unsure)},
       "f1": [{"ordinal": o, "pairIndex": i, "unsure": u, "sha256": h(PFX + "pair|%d|%d" % (o, i))} for o, i, u in f1],
       "f2Methods": [{"ordinal": o, "sha256": h(PFX + "method|%d" % o)} for o in f2]}
json.dump(out, open(BASE + "adjudication/csharp-flow-audit.input.json", "w"), indent=1)
print("methods", len(meths), "pairs", len(pairs), "unsure", len(unsure), "f1", len(f1),
      "f1unsure", sum(1 for p in f1 if p[2]), "F2 ordinals", f2)
