#!/usr/bin/env python3
"""Part F draws for java. Reads labels/java-flow.jsonl STRUCTURE only
(ordinals, pair counts, confidence); prints counts, never labels.
Key = prefix + str(ordinal) + str(pairIndex) (plain concatenation, as the
S27 draw U does with siteId); ties by (ordinal, index)."""
import hashlib, json
FF = "/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness"
PFX = "ccg-p0a-v1-flowaud|java|"
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
pairs, methods = [], []
for line in open(f"{FF}/labels/java-flow.jsonl"):
    m = json.loads(line); o = m["ordinal"]; methods.append(o)
    for i, p in enumerate(m["pairs"]):
        pairs.append((h(PFX + "pair|" + str(o) + str(i)), o, i, p["confidence"] == "unsure"))
pairs.sort(key=lambda t: (t[0], t[1], t[2]))
uns = [t for t in pairs if t[3]][:20]
rest = [t for t in pairs if t not in uns]
f1 = uns + rest[:60 - len(uns)]
meth = sorted(methods, key=lambda o: (h(PFX + "method|" + str(o)), o))
f2 = meth[:6]
json.dump({"seed": PFX, "keyRule": "sha256(prefix+'pair|'+ordinal+pairIndex) / prefix+'method|'+ordinal",
           "nMethods": len(methods), "nPairs": len(pairs), "nUnsureTotal": sum(t[3] for t in pairs),
           "F1": [{"ordinal": t[1], "pairIndex": t[2]} for t in f1],
           "F2MethodOrdinals": f2},
          open(f"{FF}/adjudication/java-flow-audit.input.json", "w"), indent=1)
print("methods", len(methods), "pairs", len(pairs), "unsureTotal", sum(t[3] for t in pairs),
      "F1", len(f1), "F1unsure", len(uns), "F2", f2)
