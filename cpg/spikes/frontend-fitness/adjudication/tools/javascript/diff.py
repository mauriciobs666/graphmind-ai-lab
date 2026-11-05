import json
def ep(e):
    p=e["point"]
    if p=="param": return ("param",e["index"])
    if p=="call": return ("call",e["calleeName"],e["ordinal"])
    if p=="ret": return ("ret",e["ordinal"])
    return ("cond",e["line"],e["code"])
def key(p): return (p["kind"],ep(p["source"]),ep(p["sink"]))
G={}
for l in open("labels/javascript-flow.jsonl"):
    m=json.loads(l); G[m["ordinal"]]=m
R={}
for l in open("adjudication/javascript-flow-relabel.jsonl"):
    m=json.loads(l); R[m["ordinal"]]=m
rows=[]
for o in sorted(R):
    g={key(p):p for p in G[o]["pairs"]}; r={key(p):p for p in R[o]["pairs"]}
    both=set(g)&set(r)
    for k in sorted(set(g)-set(r),key=str): rows.append({"ordinal":o,"side":"golden-only","pair":g[k]})
    for k in sorted(set(r)-set(g),key=str): rows.append({"ordinal":o,"side":"relabel-only","pair":r[k]})
    print(o,"golden",len(g),"relabel",len(r),"both",len(both),"golden-only",len(set(g)-set(r)),"relabel-only",len(set(r)-set(g)))
with open("adjudication/javascript-flow-audit.diff.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
print("diff rows",len(rows))
