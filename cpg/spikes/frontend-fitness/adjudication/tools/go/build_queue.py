import json,hashlib,collections,os,re

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
ROOT=(_CPG+'/spikes/frontend-fitness')
SRC=(_CPG+'/.ccg-work/corpora/go/src')
LANG='go'
I=[json.loads(l) for l in open(f'{ROOT}/adjudication/go.input.jsonl')]
R={json.loads(l)['siteId']:json.loads(l) for l in open(f'{ROOT}/truth/go.ref.jsonl')}
cache={}
def lines(p):
    if p not in cache:
        try: cache[p]=open(os.path.join(SRC,p),encoding='utf-8',errors='replace').read().split('\n')
        except Exception as e: cache[p]=None
    return cache[p]
def ctx(p,ln,before=5,after=4):
    L=lines(p)
    if L is None: return f'<unreadable {p}>'
    a=max(1,ln-before); b=min(len(L),ln+after)
    return '\n'.join(f'{i}| {L[i-1]}' for i in range(a,b+1))
h=lambda s:hashlib.sha256(s.encode()).hexdigest()
def label_all(i): return bool(i['inCorpusLabel'] or i['declaredLabel'])
def stratum(i):
    if i['status']!='agreed': return None
    k=i['reference']['static']['kind']
    if label_all(i): return 'AI'
    if k=='silent': return 'AS'
    return 'AE'
def cell(i):
    l=i['label']
    if l['notACall']: return 'notACall'
    parts=[]
    for c in l['callees']:
        parts.append(f"{c['kind']}:{c['package'] if c['kind']=='external' else c['path']}")
    return '|'.join(sorted(set(parts))) or 'empty'
strata=collections.defaultdict(list)
for i in I:
    s=stratum(i)
    if s: strata[s].append(i)
print({k:len(v) for k,v in strata.items()})
N={'AI':60,'AE':60,'AS':30}
draw={}  # siteId -> categories
meta={}
audit=[]
for s,L in strata.items():
    order=sorted(L,key=lambda i:h(f"ccg-p0a-v1-aud|{LANG}|{s}|{i['siteId']}"))
    n=min(N[s],len(L)); U=order[:n]; Uids={i['siteId'] for i in U}
    hitcells={cell(i) for i in U}
    C=[]; seen=set()
    for i in order:
        c=cell(i)
        if c in hitcells or c in seen or i['siteId'] in Uids: continue
        seen.add(c); C.append(i)
    for k,i in enumerate(U,1): draw.setdefault(i['siteId'],[]).append(f'audit-U:{s}'); meta[(i['siteId'],'U')]=(s,k,cell(i))
    for k,i in enumerate(C,1): draw.setdefault(i['siteId'],[]).append(f'audit-C:{s}'); meta[(i['siteId'],'C')]=(s,k,cell(i))
    audit.append(dict(stratum=s,N=len(L),n=n,nCells=len({cell(i) for i in L}),cellsHitByU=len(hitcells),nC=len(C)))
print(audit)
json.dump(audit,open(f'{ROOT}/../../../'+'/tmp/x','w')) if False else None
# queue
out=open(f'{ROOT}/adjudication/go.queue.jsonl','w')
cnt=collections.Counter()
def site_line(i,cats):
    r=i['reference']
    return dict(queueCategory=cats[0].split(':')[0],categories=cats,ordinal=i['ordinal'],siteId=i['siteId'],
      file=i['site']['path'],line=i['site']['line'],column=i['site']['column'],callText=i['site']['callText'],
      status=i['status'],reasons=i['reasons'],stratum=stratum(i),cell=cell(i),
      label=i['label'],reference={k:r[k] for k in r if k not in('tool','toolVersion')},
      byConstruction=bool(r.get('byConstruction')),excerpt=ctx(i['site']['path'],i['site']['line']))
for i in I:
    cats=[]
    if i['status']!='agreed': cats.append('adjudicate')
    cats+=draw.get(i['siteId'],[])
    if i['reference']['static']['kind']=='dispatch': cats.append('go-dispatch')
    if not cats: continue
    out.write(json.dumps(site_line(i,cats),ensure_ascii=False)+'\n'); out.flush()
    for c in cats: cnt[c]+=1
# census
items={}
for i in I:
    for c in i['label']['callees']:
        if c['kind'] in('internal','declared'): items.setdefault((c['path'],c['defLine']),set()).add('label-'+c['kind'])
    st=i['reference']['static']
    if st['kind'] in('internal','declared'): items.setdefault((st['path'],st['defLine']),set()).add('ref-static')
    for m in i['reference'].get('impls',[]): items.setdefault((m['path'],m['defLine']),set()).add('ref-impl')
def form(p,ln):
    L=lines(p)
    if L is None: return 'unreadable'
    # join from defLine until '{' or blank / next decl, max 8 lines
    txt=''
    for k in range(ln-1,min(len(L),ln+7)):
        txt+=L[k]+'\n'
        if '{' in L[k]: return 'body(heuristic: "{" at line %d)'%(k+1)
        if k>ln-1 and not L[k].strip(): break
    return 'bodyless-or-unclear(heuristic: no "{" within 8 lines)'
for (p,ln),src in sorted(items.items()):
    L=lines(p); first=L[ln-1].strip() if L and ln<=len(L) else ''
    out.write(json.dumps(dict(queueCategory='census',categories=['census'],file=p,line=ln,sources=sorted(src),
      defLineText=first,fcFormHeuristic=form(p,ln),excerpt=ctx(p,ln,2,7)),ensure_ascii=False)+'\n'); out.flush(); cnt['census']+=1
out.close()
print(dict(cnt))
declared=[k for k,v in items.items() if 'label-declared' in v]
print('census items',len(items),'declared',len(declared))
print('bodyless heuristics',sum(1 for p,l in items if form(p,l).startswith('bodyless')))
# reporting extras
print('byConstruction labelled',sum(1 for i in I if i['reference'].get('byConstruction')))
print('dispatch labelled',[ (i['ordinal'],i['status'],bool(i['reference']['impls'])) for i in I if i['reference']['static']['kind']=='dispatch'])
print('adjudicate reasons',collections.Counter(i['status'] for i in I))
print('unreadable',[p for p,v in cache.items() if v is None])
json.dump(audit,open(f'{ROOT}/adjudication/go.audit-plan.json','w'),indent=1) if False else None
