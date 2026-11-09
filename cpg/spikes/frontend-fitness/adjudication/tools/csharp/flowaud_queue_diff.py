#!/usr/bin/env python3
"""Run only after csharp-flow-relabel.jsonl exists. Writes F1 queue and F2 diff."""
import json

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
B=(_CPG+"/spikes/frontend-fitness/")
inp=json.load(open(B+"adjudication/csharp-flow-audit.input.json"))
gold={}; 
for l in open(B+"labels/csharp-flow.jsonl"):
    m=json.loads(l); gold[m["ordinal"]]=m
samp={}
for l in open(B+"sample/csharp-flow.jsonl"):
    m=json.loads(l); samp[m["ordinal"]]=m
def lines(o,ls):
    s=samp[o]; src=s["source"].split("\n"); out={}
    for n in sorted(set(ls)):
        i=n-s["lineStart"]
        if 0<=i<len(src): out[str(n)]=src[i]
    return out
def epl(e,o):
    if e["point"]=="param": return [samp[o]["lineStart"]]
    return [e["line"]]
def ek(e):
    p=e["point"]
    if p=="param": return ("param",e["index"],e["name"])
    if p=="call": return ("call",e["calleeName"],e["ordinal"])
    if p=="ret": return ("ret",e["ordinal"])
    return ("cond",e["line"],e["code"])
def pk(p): return (p["kind"],ek(p["source"]),ek(p["sink"]))
with open(B+"adjudication/csharp-flow-audit.queue.jsonl","w") as q:
    for r in inp["f1"]:
        o,i=r["ordinal"],r["pairIndex"]; p=gold[o]["pairs"][i]
        ls=epl(p["source"],o)+epl(p["sink"],o)
        q.write(json.dumps({"ordinal":o,"pairIndex":i,"path":gold[o]["path"],"qname":gold[o]["qname"],
          "lineStart":samp[o]["lineStart"],"unsure":p["confidence"]=="unsure","kind":p["kind"],
          "source":p["source"],"sink":p["sink"],"confidence":p["confidence"],"evidence":p["evidence"],
          "sourceLines":lines(o,ls)})+"\n")
rel={}
for l in open(B+"adjudication/csharp-flow-relabel.jsonl"):
    m=json.loads(l); rel[m["ordinal"]]=m
n=0; summ=[]
with open(B+"adjudication/csharp-flow-audit.diff.jsonl","w") as d:
    for r in inp["f2Methods"]:
        o=r["ordinal"]; g={pk(p):p for p in gold[o]["pairs"]}; x={pk(p):p for p in rel[o]["pairs"]}
        both=len(set(g)&set(x))
        summ.append((o,len(g),len(x),both))
        for side,a,b in (("golden_only",g,x),("relabel_only",x,g)):
            for k_,p in a.items():
                if k_ in b: continue
                ls=epl(p["source"],o)+epl(p["sink"],o)
                d.write(json.dumps({"ordinal":o,"side":side,"path":gold[o]["path"],"qname":gold[o]["qname"],
                  "kind":p["kind"],"source":p["source"],"sink":p["sink"],"confidence":p["confidence"],
                  "evidence":p["evidence"],"sourceLines":lines(o,ls)})+"\n"); n+=1
print("(ordinal,golden,relabel,both)",summ,"diff rows",n)
