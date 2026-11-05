#!/usr/bin/env python3
"""Part F draws for go (ML note 5.4). Emits ONLY ordinals/indexes/confidence/hashes, never pair content.
Hash key: sha256(seed + <ordinal>|<pair index>) for pairs, sha256(seed + <ordinal>) for methods.
Pair index = 0-based position in the method's `pairs` list in labels/go-flow.jsonl."""
import hashlib, json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[3]
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
rows = [json.loads(l) for l in (R/"labels/go-flow.jsonl").read_text().splitlines() if l.strip()]
pairs = []
for r in rows:
    for i, p in enumerate(r["pairs"]):
        pairs.append({"ordinal": r["ordinal"], "pairIndex": i, "confidence": p["confidence"],
                      "hash": H(f"ccg-p0a-v1-flowaud|go|pair|{r['ordinal']}|{i}")})
pairs.sort(key=lambda x: x["hash"])
uns = [p for p in pairs if p["confidence"] == "unsure"]
sure = [p for p in pairs if p["confidence"] != "unsure"]
chosen = uns[:20]
chosen += sure[:60 - len(chosen)] if len(chosen) < 60 else []
chosen = chosen[:60]
meths = sorted(({"ordinal": r["ordinal"], "hash": H(f"ccg-p0a-v1-flowaud|go|method|{r['ordinal']}")} for r in rows), key=lambda x: x["hash"])
out = {"seedPairs": "ccg-p0a-v1-flowaud|go|pair|<ordinal>|<pairIndex>", "seedMethods": "ccg-p0a-v1-flowaud|go|method|<ordinal>",
       "nMethods": len(rows), "nPairs": len(pairs), "nUnsure": len(uns),
       "f1": chosen, "f1UnsureCount": sum(p["confidence"] == "unsure" for p in chosen),
       "f2Methods": [m["ordinal"] for m in meths[:6]], "f2Hashes": meths[:6],
       "f2ExtensionMethods": [m["ordinal"] for m in meths[6:9]],
       "noPairOrdinals": [r["ordinal"] for r in rows if not r["pairs"]]}
(R/"adjudication/go-flow-audit.input.json").write_text(json.dumps(out, indent=1) + "\n")
print({k: v for k, v in out.items() if k not in ("f1", "f2Hashes")})
