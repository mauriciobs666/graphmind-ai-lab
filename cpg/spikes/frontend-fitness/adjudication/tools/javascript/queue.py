import json
SRC="/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/javascript/src/"
inp=json.load(open("adjudication/javascript-flow-audit.input.json"))
G={}
for l in open("labels/javascript-flow.jsonl"):
    m=json.loads(l); G[m["ordinal"]]=m
S={}
for l in open("sample/javascript-flow.jsonl"):
    m=json.loads(l); S[m["ordinal"]]=m
cache={}
def lines(path):
    if path not in cache: cache[path]=open(SRC+path.removeprefix("src/") if not __import__("os").path.exists(SRC+path) else SRC+path).read().split("\n")
    return cache[path]
out=[]
for r in inp["pairs"]:
    m=G[r["ordinal"]]; p=m["pairs"][r["pairIndex"]]; s=S[r["ordinal"]]
    ls=lines(m["path"]); pts={}
    for e in (p["source"],p["sink"]):
        ln=e.get("line") if e["point"]!="param" else s["lineStart"]
        if ln: pts[ln]=ls[ln-1]
    out.append({"ordinal":r["ordinal"],"pairIndex":r["pairIndex"],"path":m["path"],"qname":m["qname"],
      "methodSpan":[s["lineStart"],s["lineEnd"]],"unsure":r["unsure"],"pair":p,
      "pointLines":{str(k):v for k,v in sorted(pts.items())}})
with open("adjudication/javascript-flow-audit.queue.jsonl","w") as f:
    for q in out: f.write(json.dumps(q)+"\n")
print(len(out),"unsure",sum(q["unsure"] for q in out),"methods",sorted({q["ordinal"] for q in out}))
