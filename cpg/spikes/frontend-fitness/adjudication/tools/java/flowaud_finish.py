#!/usr/bin/env python3
"""Annotate the diff (cosmetic cond-code mismatches) and add source lines to the F1 queue."""
import json

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF = (_CPG+"/spikes/frontend-fitness")
SRC = (_CPG+"/.ccg-work/corpora/java/src/")
adj = f"{FF}/adjudication/"
rows = [json.loads(l) for l in open(adj + "java-flow-audit.diff.jsonl")]
for r in rows:
    if r["ordinal"] == 4 and any(e.get("point") == "cond" and "Pet pet" in e.get("code","") + "" or
                                 e.get("point") == "cond" and e.get("code","").startswith(("pet :", "visit :", "Visit visit", "Pet pet"))
                                 for e in (r["source"], r["sink"])):
        r["note"] = "same pair on both sides; cond code text differs only by the declared loop-variable type (golden 'Pet pet : pets' / 'Visit visit : visits' vs relabel 'pet : pets' / 'visit : visits')"
with open(adj + "java-flow-audit.diff.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
cache = {}
q = [json.loads(l) for l in open(adj + "java-flow-audit.queue.jsonl")]
for d in q:
    L = cache.setdefault(d["path"], open(SRC + d["path"]).read().split("\n"))
    ln = sorted({d["pair"]["source"].get("line"), d["pair"]["sink"].get("line")} - {None})
    d["sourceLines"] = {str(n): L[n - 1] for n in ln}
    d["file"] = d["path"]
with open(adj + "java-flow-audit.queue.jsonl", "w") as f:
    for d in q: f.write(json.dumps(d) + "\n")
print(sum("note" in r for r in rows), "annotated of", len(rows))
