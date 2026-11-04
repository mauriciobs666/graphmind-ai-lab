import json,os
from lab import *
OUT='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/labels/java.jsonl'
S={}
for l in open('/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/sample/java.jsonl'):
  d=json.loads(l); S[d['ordinal']]=d
D={}
def rec(o,callees,ev,conf='sure',nac=False,dyn=False):
  D[o]=dict(ordinal=o,siteId=S[o]['siteId'],callees=callees,notACall=nac,dynamic=dyn,confidence=conf,evidence=ev[:300])
def I(o,tm,n,ev,ext=None,conf='sure',extra=None): raise SystemExit('unused')
def X(o,pkg,ev,conf='sure'):
  rec(o,[{'kind':'external','package':p} for p in ([pkg] if isinstance(pkg,str) else pkg)],ev,conf)
def K(o,t,n,ev):
  r=[(m['rel'],m['qual'],m['line']) for m in meths if m['type']==t and m['name']==t and m['n']==n]
  if r: c=[{'kind':'internal','path':a,'qname':a+'::'+b+'.<init>','defLine':l} for a,b,l in r]
  else:
    ty=types[t]; c=[{'kind':'internal','path':ty['rel'],'qname':ty['rel']+'::'+ty['qual'],'defLine':ty['line']}]
  rec(o,c,ev)
def M(o,t,name,n,ev):
  ms=[m for m in meths if m['type']==t and m['name']==name and m['n']==n]
  rec(o,[{'kind':'internal','path':m['rel'],'qname':m['rel']+'::'+m['qual']+'.'+m['name'],'defLine':m['line']} for m in ms],ev)
def flush():
  ls=[json.loads(l) for l in open(OUT)]; last=ls[-1]['ordinal']
  with open(OUT,'a') as f:
    o=last+1
    while o in D: f.write(json.dumps(D[o])+'\n'); o+=1
  ls=[json.loads(l) for l in open(OUT)]
  print('lines',len(ls),'last',ls[-1]['ordinal'],'internal/declared',sum(1 for x in ls if any(c['kind'] in('internal','declared') for c in x['callees'])))
