"""Part F draws for javascript. Emits only ordinals / pair indexes / unsure flags; never labels.
pair key  = sha256("ccg-p0a-v1-flowaud|javascript|pair|" + ordinal + "|" + pairIndex)  (pairIndex 0-based, file order)
method key= sha256("ccg-p0a-v1-flowaud|javascript|method|" + ordinal)
"""
import hashlib, json
L = "javascript"
LAB = "labels/javascript-flow.jsonl"
OUT = "adjudication/javascript-flow-audit.input.json"
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
pairs, methods = [], []
for line in open(LAB):
    if not line.strip(): continue
    m = json.loads(line)
    methods.append(m["ordinal"])
    for i, p in enumerate(m["pairs"]):
        pairs.append({"ordinal": m["ordinal"], "pairIndex": i, "unsure": p.get("confidence") == "unsure",
                      "key": h(f"ccg-p0a-v1-flowaud|{L}|pair|{m['ordinal']}|{i}")})
pairs.sort(key=lambda r: r["key"])
N_PAIR, N_METH, UCAP = 40, 4, 20
uns = [r for r in pairs if r["unsure"]][:UCAP]
rest = [r for r in pairs if r not in uns][: N_PAIR - len(uns)]
drawn = uns + rest
mk = sorted(methods, key=lambda o: h(f"ccg-p0a-v1-flowaud|{L}|method|{o}"))
msel = mk[:N_METH]
out = {"language": L, "seedPairPrefix": f"ccg-p0a-v1-flowaud|{L}|pair|", "seedMethodPrefix": f"ccg-p0a-v1-flowaud|{L}|method|",
       "keyForm": "prefix + ordinal + '|' + pairIndex (0-based file order); methods: prefix + ordinal",
       "methodsTotal": len(methods), "pairsTotal": len(pairs), "unsureTotal": sum(r["unsure"] for r in pairs),
       "pairs": [{k: r[k] for k in ("ordinal", "pairIndex", "unsure")} for r in drawn],
       "f2Methods": msel, "f2Extension": mk[N_METH:N_METH + 3]}
json.dump(out, open(OUT, "w"), indent=1)
print("methods", len(methods), "pairs", len(pairs), "unsureTotal", out["unsureTotal"])
print("drawn pairs", len(drawn), "of which unsure", len(uns))
print("F2 methods", msel, "extension (if needed)", out["f2Extension"])
print("drawn pairs in F2 methods:", sum(r["ordinal"] in msel for r in drawn))
