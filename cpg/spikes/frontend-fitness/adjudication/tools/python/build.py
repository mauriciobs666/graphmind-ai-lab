import json,hashlib,collections,sys

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
R=(_CPG+'/spikes/frontend-fitness/')
SRC=(_CPG+'/.ccg-work/corpora/python/src/')
OUT=R+'adjudication/python.queue.jsonl'
inp=[json.loads(l) for l in open(R+'adjudication/python.input.jsonl')]
smp={}
for l in open(R+'sample/python.jsonl'):
    o=json.loads(l); smp[o['siteId']]=o
ref={}
for l in open(R+'truth/python.ref.jsonl'):
    o=json.loads(l); ref[o['siteId']]=o
cache={}
def lines(p):
    if p not in cache: cache[p]=open(SRC+p,encoding='utf-8',errors='replace').read().split('\n')
    return cache[p]
def excerpt(p,line,before=5,after=4):
    L=lines(p); a=max(1,line-before); b=min(len(L),line+after)
    return '\n'.join(f"{i}| {L[i-1]}" for i in range(a,b+1))
def H(stratum,sid): return hashlib.sha256(f"ccg-p0a-v1-aud|python|{stratum}|{sid}".encode()).hexdigest()
def stratum(r):
    if r['inCorpusLabel']: return 'AI'
    if r['reference']['static']['kind']=='silent': return 'AS'
    return 'AE'
def cell(r):
    l=r['label']
    if l['notACall']: return 'notACall'
    return '|'.join(sorted({(c['kind']+':'+(c.get('path') or c.get('package'))) for c in l['callees']}))
def base(r,cat,extra):
    s=smp[r['siteId']]
    d={'queueCategory':cat,'ordinal':r['ordinal'],'siteId':r['siteId'],'file':s['path'],'line':s['line'],
       'callText':s['callText'],'status':r['status'],'reasons':r['reasons'],'label':r['label'],
       'reference':r['reference'],'excerpt':excerpt(s['path'],s['line'])}
    d.update(extra); return d
out=open(OUT,'w')
def w(d): out.write(json.dumps(d,ensure_ascii=False)+'\n'); out.flush()
counts=collections.Counter()
for r in inp:
    assert smp[r['siteId']]['ordinal']==r['ordinal']
# A. adjudication
for r in inp:
    if r['status']=='needs-adjudication':
        w(base(r,'adjudicate',{})); counts['adjudicate']+=1
# B. audits
agreed=[r for r in inp if r['status']=='agreed']
st=collections.defaultdict(list)
for r in agreed: st[stratum(r)].append(r)
N={k:len(v) for k,v in st.items()}
size={'AI':60,'AE':60,'AS':30}
Uids=set(); audit_meta={}
for k in ('AI','AE','AS'):
    v=sorted(st[k],key=lambda r:H(k,r['siteId']))
    U=v[:size[k]]; Uids|={r['siteId'] for r in U}
    for i,r in enumerate(U):
        w(base(r,'audit-U',{'stratum':k,'drawRank':i+1,'cell':cell(r),'stratumN':N[k]})); counts['audit-U-'+k]+=1
    hitcells={cell(r) for r in U}
    seen=set()
    for r in v[size[k]:]:
        c=cell(r)
        if c in hitcells or c in seen: continue
        seen.add(c)
        w(base(r,'audit-C',{'stratum':k,'cell':c,'stratumN':N[k]})); counts['audit-C-'+k]+=1
json.dump({'stratumN':N,'cellCounts':{k:dict(collections.Counter(cell(r) for r in st[k])) for k in st}},open('strata.json','w'),indent=1)
import re,ast
def form(p,dl):
    L=lines(p); t=L[dl-1] if dl<=len(L) else ''
    m=re.match(r'\s*(async def|def|class)\s',t)
    if not m: return 'not-a-declaration-at-defLine',[]
    ind=len(t)-len(t.lstrip())
    decos=[]; i=dl-2
    while i>=0 and L[i].strip().startswith('@'): decos.append(L[i].strip()); i-=1
    if m.group(1)=='class': return 'class',decos
    j=dl-1; depth=0; sig=''
    while j<len(L):
        sig+=L[j]+'\n'; depth=sum(sig.count(a) for a in '([{')-sum(sig.count(a) for a in ')]}')
        if depth==0 and L[j].rstrip().split('#')[0].rstrip().endswith(':'): break
        j+=1
    k=j+1; body=[]
    while k<len(L) and (not L[k].strip() or len(L[k])-len(L[k].lstrip())>ind): body.append(L[k]); k+=1
    txt='\n'.join(body).strip()
    try:
        tree=ast.parse('def f():\n'+'\n'.join('    '+x.strip() if False else x for x in body) if False else 'if 1:\n'+'\n'.join(body)) if body else None
        stm=tree.body[0].body if tree else []
    except Exception: stm=None
    if stm is None: return ('def-body-unparsed',decos)
    if stm and isinstance(stm[0],ast.Expr) and isinstance(getattr(stm[0],'value',None),ast.Constant) and isinstance(stm[0].value.value,str): stm=stm[1:]
    if not stm: return 'def-docstring-only',decos
    if all(isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and x.value.value is Ellipsis for x in stm): return 'def-ellipsis-body',decos
    if all(isinstance(x,ast.Pass) for x in stm): return 'def-pass-body',decos
    if len(stm)==1 and isinstance(stm[0],ast.Raise): return 'def-raise-only-body',decos
    return 'def-with-body',decos
# C. census
items=collections.defaultdict(lambda:{'sources':set(),'qnames':set(),'sites':[]})
for r in inp:
    for c in r['label']['callees']:
        if c['kind'] in('internal','declared'):
            it=items[(c['path'],c['defLine'])]; it['sources'].add('label-'+c['kind']); it['qnames'].add(c['qname']); it['sites'].append(r['ordinal'])
    rf=r['reference']
    for c in [rf['static']]+rf.get('impls',[]):
        if c.get('path') and c.get('kind') in ('internal',None):
            it=items[(c['path'],c['defLine'])]; it['sources'].add('ref'); it['qnames'].add(c.get('qname')); it['sites'].append(r['ordinal'])
for (p,dl),it in sorted(items.items()):
    L=lines(p); txt=L[dl-1] if dl<=len(L) else ''
    fm,dc=form(p,dl)
    w({'queueCategory':'census','fcForm':fm,'decorators':dc,'file':p,'line':dl,'defLineText':txt.strip(),'sources':sorted(it['sources']),
       'qnames':sorted(q for q in it['qnames'] if q),'nSites':len(set(it['sites'])),'siteOrdinals':sorted(set(it['sites']))[:8],
       'excerpt':excerpt(p,dl,1,8)}); counts['census']+=1
print(dict(counts),N)
