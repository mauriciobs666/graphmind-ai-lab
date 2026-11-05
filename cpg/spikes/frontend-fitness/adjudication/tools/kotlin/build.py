from strata import *
SRC='/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/kotlin/src/'
Q=R+'adjudication/kotlin.queue.jsonl'
open(Q,'w').close()
cache={}
def lines(p):
    if p not in cache: cache[p]=open(SRC+p,encoding='utf-8',errors='replace').read().split('\n')
    return cache[p]
def ctx(p,ln,before=5,after=4):
    L=lines(p); a=max(1,ln-before); b=min(len(L),ln+after)
    return [f'{i}: {L[i-1][:200]}' for i in range(a,b+1)]
def emit(o):
    with open(Q,'a') as f: f.write(json.dumps(o,ensure_ascii=False)+'\n')
def siteitem(cat,r,extra=None):
    s=r['site']
    o={'category':cat,'ordinal':r['ordinal'],'siteId':r['siteId'],'where':f"{s['path']}:{s['line']}",'callText':s['callText'],
       'label':r['label'],'reference':{'static':r['reference']['static'],'impls':r['reference']['impls']},
       'inputStatus':r['status'],'reasons':r['reasons'],'context':ctx(s['path'],s['line'])}
    if extra: o.update(extra)
    emit(o)
# 1 needs-adjudication
for r in rows:
    if r['status']!='agreed': siteitem('adjudicate',r)
# 2 audit draws
by=collections.defaultdict(list)
for r in rows:
    s=stratum(r)
    if s: by[s].append(r)
N={'AI':60,'AE':60,'AS':30}
audit={}
summary={}
for st in ('AI','AE','AS'):
    pop=sorted(by[st],key=lambda r:h(st,r['siteId']))
    U=pop[:N[st]]; Uids={r['siteId'] for r in U}
    hitcells={cell(r) for r in U}
    C=[];seen=set()
    for r in pop:
        c=cell(r)
        if c in hitcells or c in seen: continue
        seen.add(c);C.append(r)
    summary[st]={'population':len(pop),'U':len(U),'cells':len({cell(r) for r in pop}),'cellsHitByU':len(hitcells),'C':len(C)}
    audit[st]=(U,C,pop)
    for i,r in enumerate(U,1): siteitem('audit-U',r,{'stratum':st,'drawRank':i,'cell':cell(r)})
json.dump({k:[r['ordinal'] for r in v[0]] for k,v in audit.items()},open('U.json','w'))
for st in ('AI','AE','AS'):
    for i,r in enumerate(audit[st][1],1): siteitem('audit-C',r,{'stratum':st,'cellRank':i,'cell':cell(r)})
# sweep candidates (conditional): per audit-U site, up to 10 same-cell/same-callee-name sites, hash order, all labelled sites
sw={}
for st in ('AI','AE'):
    for r in audit[st][0]:
        cand=[x for x in rows if x['siteId']!=r['siteId'] and (cell(x)==cell(r) or cname(x)==cname(r))]
        cand.sort(key=lambda x:h(st,x['siteId']))
        sw[str(r['ordinal'])]={'stratum':st,'cell':cell(r),'calleeName':cname(r),'nCandidates':len(cand),'first10':[x['ordinal'] for x in cand[:10]]}
json.dump(sw,open(R+'adjudication/kotlin.sweep-candidates.json','w'),indent=0)
json.dump(summary,open('summary.json','w'),indent=1)
print(summary)
