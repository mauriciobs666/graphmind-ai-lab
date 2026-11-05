import json, re, collections
from pathlib import Path
FF = Path("/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness"); A = FF/"adjudication"
rj = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
fin = rj(FF/"truth/csharp.final.jsonl"); dec = rj(A/"csharp.decisions.jsonl"); q = rj(A/"csharp.queue.jsonl")
inp = rj(A/"csharp.input.jsonl"); lab = {r["ordinal"]: r for r in rj(FF/"labels/csharp.jsonl")}
census = {(x["path"], x["defLine"]) for x in dec if x["category"]=="census"}
qc = {(x["file"].rsplit(":",1)[0], int(x["file"].rsplit(":",1)[1])) for x in q if x["category"]=="census"}
P = []
if len(fin) != 313 or [x["ordinal"] for x in fin] != list(range(1, 314)): P.append("final not 313 ordinals 1..313")
VB = {"agreed","audit-confirmed","adjudicated","audit-overturned","sweep-overturned","label-only"}
cit = re.compile(r"^[^\s:]+:\d+")
keys = {"ordinal","siteId","T","N","external","notACall","dynamic","scored","notScoredReason","basis","strata","evidence"}
for x in fin:
    if set(x) != keys: P.append(f"keys {x['ordinal']}")
    if x["basis"] not in VB: P.append(f"basis {x['ordinal']}")
    if not x["evidence"] or not all(cit.match(e) for e in x["evidence"]): P.append(f"citation {x['ordinal']}")
    for t in x["T"]:
        if (t["path"], t["defLine"]) not in census: P.append(f"T not in census {x['ordinal']} {t['path']}:{t['defLine']}")
        if t["role"] not in ("impl","decl"): P.append(f"role {x['ordinal']}")
    if x["siteId"] != lab[x["ordinal"]]["siteId"]: P.append(f"siteId {x['ordinal']}")
    if not x["scored"] and not x["notScoredReason"]: P.append(f"unscored w/o reason {x['ordinal']}")
und = sum(1 for x in fin if x["notScoredReason"]=="underivable")
print("final lines", len(fin), "underivable", und, "cap", 0.05*313)
print("census decisions", len(census), "queue", len(qc), "equal", census == qc)
if census != qc: P.append("census sets differ")
# final T sets vs label in-corpus (all label-confirmed)
d = 0
for x, r in zip(fin, inp):
    if {(t["path"], t["defLine"]) for t in x["T"]} != {tuple(p) for p in r["inCorpusLabel"]+r["declaredLabel"]}: d += 1
print("final T differs from label in-corpus set:", d)
# role consistency with census role
role = {(x["path"], x["defLine"]): x["role"] for x in dec if x["category"]=="census"}
print("T role != census role:", sum(1 for x in fin for t in x["T"] if role[(t["path"], t["defLine"])] != t["role"]))
dk = collections.Counter(x["category"] for x in dec); print(dict(dk))
print("final T used defs:", len({(t['path'],t['defLine']) for x in fin for t in x['T']}), "of census", len(census))
print("PROBLEMS", P)
