"""After the relabel is on disk: build F1 queue and F2 diff. Run from adjudication/."""
import json, re
SRC = "/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/python/src/"
gold = {}
for l in open("../labels/python-flow.jsonl"):
    m = json.loads(l); gold[m["ordinal"]] = m
samp = {}
for l in open("../sample/python-flow.jsonl"):
    m = json.loads(l); samp[m["ordinal"]] = m
inp = json.load(open("python-flow-audit.input.json"))
cache = {}
def lines(path):
    if path not in cache: cache[path] = open(SRC + path, encoding="utf-8").read().split("\n")
    return cache[path]
def norm(s): return re.sub(r"\s+", " ", s).strip()
def ep(e):
    p = e["point"]
    if p == "param": return ("param", e["index"])
    if p == "call": return ("call", e["calleeName"], e["ordinal"])
    if p == "ret": return ("ret", e["ordinal"])
    return ("cond", e["line"], norm(e["code"]))
def ident(p): return (p["kind"], ep(p["source"]), ep(p["sink"]))
def srclines(m, p):
    path = m["path"]; sm = samp[m["ordinal"]]; ls = set()
    for e in (p["source"], p["sink"]):
        ls.add(e["line"] if "line" in e else sm["lineStart"])
    ev = re.findall(r"[:\s](\d{2,5})(?:-(\d{2,5}))?\b", p.get("evidence", ""))
    return {str(n): lines(path)[n - 1] for n in sorted(ls) if 1 <= n <= len(lines(path))}
# F1
with open("python-flow-audit.queue.jsonl", "w") as o:
    for k, d in enumerate(inp["F1"]):
        m = gold[d["ordinal"]]; p = m["pairs"][d["pairIndex"]]
        o.write(json.dumps({"rank": k, "ordinal": m["ordinal"], "pairIndex": d["pairIndex"], "path": m["path"], "qname": m["qname"],
            "methodSpan": [samp[m["ordinal"]]["lineStart"], samp[m["ordinal"]]["lineEnd"]], "unsure": p["confidence"] == "unsure",
            "pair": p, "sourceLines": srclines(m, p)}) + "\n")
# F2
rel = {}
for l in open("python-flow-relabel.jsonl"):
    m = json.loads(l); rel[m["ordinal"]] = m
n = 0; stats = {}
with open("python-flow-audit.diff.jsonl", "w") as o:
    for ordv in inp["F2"]["methods"]:
        g, r = gold[ordv], rel[ordv]
        gi = {ident(p): p for p in g["pairs"]}; ri = {ident(p): p for p in r["pairs"]}
        both = set(gi) & set(ri)
        stats[ordv] = {"golden": len(gi), "relabel": len(ri), "both": len(both)}
        for side, a, b in (("golden-only", gi, ri), ("relabel-only", ri, gi)):
            for i, p in a.items():
                if i in b: continue
                m = g if side == "golden-only" else r
                o.write(json.dumps({"ordinal": ordv, "path": g["path"], "qname": g["qname"], "side": side, "pair": p,
                    "sourceLines": srclines(m, p)}) + "\n"); n += 1
print(stats, "diff rows", n)
print("F1", sum(1 for _ in open("python-flow-audit.queue.jsonl")), "unsure", sum(json.loads(l)["unsure"] for l in open("python-flow-audit.queue.jsonl")))
