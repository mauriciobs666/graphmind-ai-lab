import json,hashlib,collections,os
FF='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness'
SRC='/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/javascript/src'
R=[json.loads(l) for l in open(FF+'/adjudication/javascript.input.jsonl')]
def h(st,sid): return hashlib.sha256(f'ccg-p0a-v1-aud|javascript|{st}|{sid}'.encode()).hexdigest()
def rk(r): return (r['reference'].get('static') or {}).get('kind')
def stratum(r):
    if r['status']!='agreed': return None
    L=r['label']
    if L['callees'] and any(c['kind'] in('internal','declared') for c in L['callees']): return 'AI'
    return 'AS' if rk(r)=='silent' else 'AE'
for r in R: r['stratum']=stratum(r)
S={s:[r for r in R if r['stratum']==s] for s in('AI','AE','AS')}
def cell(r):
    cs=r['label']['callees']
    if r['label']['notACall'] or not cs: return 'notACall'
    c=cs[0]; return (c['kind'],c.get('package') or c.get('path'))
def order(st): return sorted(S[st],key=lambda r:h(st,r['siteId']))
U={}
U['AI']=order('AI')[:100]
o=order('AE'); u=o[:100]
loc=[r for r in u if rk(r)=='local']
if len(loc)<40:
    need=40-len(loc); extra=[r for r in o[100:] if rk(r)=='local'][:need]
    drop=[r for r in reversed(u) if rk(r)!='local'][:need]
    ids={r['siteId'] for r in drop}; u=[r for r in u if r['siteId'] not in ids]+extra
    topup=need
else: topup=0
U['AE']=u; U['AS']=order('AS')
C={}
for st in S:
    hit={r['siteId'] for r in U[st]}; hitc={str(cell(r)) for r in U[st]}
    seen={}
    for r in order(st):
        c=str(cell(r))
        if c in hitc or c in seen: continue
        seen[c]=r
    C[st]=list(seen.values())
print({s:len(S[s]) for s in S},{s:len(U[s]) for s in U},{s:len(C[s]) for s in C},'AE local in U',sum(rk(r)=='local' for r in U['AE']),'topup',topup)
print('AE local pop',sum(rk(r)=='local' for r in S['AE']))
def ctx(path,line,n_before=5,n_after=4):
    try: ls=open(f'{SRC}/{path}',encoding='utf-8',errors='replace').read().split('\n')
    except Exception as e: return [f'<unreadable {e}>']
    a=max(1,line-n_before); b=min(len(ls),line+n_after)
    return [f'{i}: {ls[i-1]}' for i in range(a,b+1)]
items={}
def add(r,cat):
    d=items.setdefault(r['siteId'],{'siteId':r['siteId'],'ordinal':r['ordinal'],'where':f"{r['site']['path']}:{r['site']['line']}",'callText':r['site']['callText'],'column':r['site']['column'],'categories':[],'stratum':r['stratum'],'status':r['status'],'reasons':r['reasons'],'label':r['label'],'reference':r['reference'],'context':ctx(r['site']['path'],r['site']['line'])})
    d['categories'].append(cat)
for r in R:
    if r['status']=='needs-adjudication': add(r,'needs-adjudication:'+r['reasons'][0])
for st in S:
    for r in U[st]: add(r,f'audit-U:{st}')
    for r in C[st]: add(r,f'audit-C:{st}')
for r in S['AE']:
    if rk(r)=='local' and r['siteId'] in {x['siteId'] for x in U['AE']}: items[r['siteId']]['categories'].append('js-ref-local-agreed')
# census
cen=collections.defaultdict(set)
for r in R:
    for c in r['label']['callees']:
        if c['kind'] in('internal','declared'): cen[(c['path'],c['defLine'])].add('label:'+c['kind'])
    st=r['reference'].get('static') or {}
    if st.get('kind') in('internal','declared'): cen[(st['path'],st['defLine'])].add('ref-static:'+st['kind'])
    for i in r['reference'].get('impls') or []:
        if isinstance(i,dict) and 'path' in i: cen[(i['path'],i['defLine'])].add('ref-impl')
        else: cen[(i[0],i[1])].add('ref-impl')
out=FF+'/adjudication/javascript.queue.jsonl'
with open(out,'w') as f:
    for d in sorted(items.values(),key=lambda d:d['ordinal']):
        d['item']='site'; f.write(json.dumps(d)+'\n'); f.flush()
    for (p,l),why in sorted(cen.items()):
        c=ctx(p,l,2,5); line=c[2] if len(c)>2 else ''
        t=line.split(': ',1)[-1] if ': ' in line else ''
        guess='body-likely' if ('{' in t or '=>' in t or 'function' in t) else 'unclear'
        f.write(json.dumps({'item':'census','where':f'{p}:{l}','path':p,'defLine':l,'sources':sorted(why),'declLine':t,'fcFormGuess':guess,'context':c})+'\n'); f.flush()
print('sites',len(items),'census',len(cen))
print(collections.Counter(c.split(':')[0]+':'+c.split(':')[1] if c.startswith('audit') else c for d in items.values() for c in d['categories']))
print(collections.Counter(guess for guess in [0]))
