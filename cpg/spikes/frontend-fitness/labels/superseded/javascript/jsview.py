import json,sys
import os, pathlib as _pl, tempfile as _tf; _CPG = str(_pl.Path(__file__).resolve().parents[5])
S=(_CPG+'/.ccg-work/corpora/javascript/src/')
a,b=int(sys.argv[1]),int(sys.argv[2])
for l in open((_CPG+'/spikes/frontend-fitness/sample/javascript.jsonl')):
    d=json.loads(l)
    if a<=d['ordinal']<=b:
        L=open(S+d['path']).read().split('\n')
        print(d['ordinal'],d['path'],d['line'],d['column'],'|',d['callText'][:60],'|',L[d['line']-1].strip()[:130])
