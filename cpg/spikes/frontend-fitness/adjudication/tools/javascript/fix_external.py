"""Analyst Pass 4 N2: `external` = the labeller's package from input.jsonl; the DS prose moves into `evidence`."""
import json

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF=(_CPG+'/spikes/frontend-fitness/')
F=[json.loads(l) for l in open(FF+'truth/javascript.final.jsonl')]
IN={}
for l in open(FF+'adjudication/javascript.input.jsonl'):
    r=json.loads(l); IN[r['siteId']]=r
def lab(f): return sorted({c.get('package') for c in IN[f['siteId']]['label']['callees'] if c['kind']=='external'})
before=[{k:v for k,v in f.items() if k not in('external','evidence')} for f in F]
n=0
for f in F:
    l=lab(f)
    if sorted(f['external'])!=l:
        prose='; '.join(f['external']); n+=1
        note=f' [external: {prose}]'
        ev=f['evidence']
        if prose not in ev:
            ev=ev[:300-len(note)].rstrip()+note
        f['evidence']=ev[:300]; f['external']=l
    assert sorted(f['external'])==l
after=[{k:v for k,v in f.items() if k not in('external','evidence')} for f in F]
assert before==after
with open(FF+'truth/javascript.final.jsonl','w') as o:
    for f in F: o.write(json.dumps(f)+'\n')
print('fixed',n,'non-empty external lines',sum(1 for f in F if f['external']),'max evidence',max(len(f['evidence']) for f in F))
