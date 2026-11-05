#!/usr/bin/env python3
"""usage: sweep.py <ordinal> -> up to 10 sibling sites (same cell or same callText last identifier), first 10 by the stratum hash."""
import sys,json,hashlib,re
FF='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness'
inp=[json.loads(l) for l in open(f'{FF}/adjudication/java.input.jsonl')]
samp={json.loads(l)['ordinal']:json.loads(l) for l in open(f'{FF}/sample/java.jsonl')}
def cell(r):
    l=r['label']
    if l['notACall']: return 'notACall'
    return '|'.join(f'{k}:{v}' for k,v in sorted({(c['kind'],c.get('path') or c.get('package')) for c in l['callees']}))
nm=lambda o: re.split(r'[.\s]',samp[o]['callText'])[-1]
o=int(sys.argv[1]); t=inp[o-1]; assert t['ordinal']==o
sib=[r for r in inp if r['ordinal']!=o and (cell(r)==cell(t) or nm(r['ordinal'])==nm(o))]
sib.sort(key=lambda r:hashlib.sha256(f"ccg-p0a-v1-aud|java|sweep|{r['siteId']}".encode()).hexdigest())
print(len(sib),'siblings; first 10:',[r['ordinal'] for r in sib[:10]])
