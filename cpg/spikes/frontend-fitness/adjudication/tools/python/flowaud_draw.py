"""Part F draw for python flow labels. Emits ONLY ordinals / pair indexes / unsure flag (no label content).
Hash key: pair   -> sha256("ccg-p0a-v1-flowaud|python|pair|" + f"{ordinal}|{pairIndex}")  (pairIndex 0-based file order)
          method -> sha256("ccg-p0a-v1-flowaud|python|method|" + f"{ordinal}")
F1: unsure pairs first (ascending hash, up to 20), then remaining slots from the other pairs (ascending hash), total 60.
F2: first 6 methods by ascending hash."""
import json, hashlib, sys
LAB = "../labels/python-flow.jsonl"
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
pairs = []; methods = []
for line in open(LAB):
    m = json.loads(line); methods.append(m["ordinal"])
    for i, p in enumerate(m["pairs"]):
        pairs.append((m["ordinal"], i, p["confidence"] == "unsure"))
key = lambda t: H(f"ccg-p0a-v1-flowaud|python|pair|{t[0]}|{t[1]}")
uns = sorted([t for t in pairs if t[2]], key=key)
oth = sorted([t for t in pairs if not t[2]], key=key)
f1 = uns[:20]; f1 += oth[:60 - len(f1)]
f2 = sorted(methods, key=lambda o: H(f"ccg-p0a-v1-flowaud|python|method|{o}"))
out = {"hash": "sha256", "pairKey": "ccg-p0a-v1-flowaud|python|pair|<ordinal>|<pairIndex>",
       "methodKey": "ccg-p0a-v1-flowaud|python|method|<ordinal>",
       "nMethods": len(methods), "nPairs": len(pairs), "nUnsure": len(uns),
       "F1": [{"ordinal": o, "pairIndex": i, "unsure": u} for o, i, u in f1],
       "F2": {"methods": f2[:6], "extension3": f2[6:9]}}
json.dump(out, open("python-flow-audit.input.json", "w"), indent=1)
print({k: out[k] for k in ("nMethods", "nPairs", "nUnsure")}, "F1", len(f1), "F2", out["F2"])
