"""Go S57 canonicalisation under ML v1.12 (reads tools/go/orig/*.v1.11 + go.decisions.jsonl + go.input.jsonl; writes truth/go.final.jsonl and adjudication/go.noise.json). Idempotent."""
import json,math,collections,re
A='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/'
rd=lambda p:[json.loads(l) for l in open(A+p) if l.strip()]
old=rd('adjudication/tools/go/orig/go.final.v1.11.jsonl'); onoise=json.load(open(A+'adjudication/tools/go/orig/go.noise.v1.11.json'))
I={i['siteId']:i for i in rd('adjudication/go.input.jsonl')}
D=rd('adjudication/go.decisions.jsonl'); Q=rd('adjudication/go.queue.jsonl')
dsite=[d for d in D if d['kind']=='site']; cens=[d for d in D if d['kind']=='census']
main={d['siteId']:d for d in dsite if d['basis']!='sweep-confirmed'}
swept={d['siteId'] for d in dsite if d['basis']=='sweep-confirmed'}
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0,c-h),4),round(min(1,c+h),4)]
def ev(txt,cites,site):
    s=txt if txt else ''
    c=(cites or [site])[0]
    if len(s)>300: s=s[:297]+'...'
    if not re.search(r'\S+:\d+',s): s=(s+' ' if s else '')+c
    return s[:300] if len(s)>300 else s
fin=[];precedence_changes=[]
for o in old:
    d=main.get(o['siteId']); site=f"{o['site']['path']}:{o['site']['line']}"
    basis=o['basis']
    if o['siteId'] in swept and basis=='agreed':
        basis='audit-confirmed'; precedence_changes.append(o['ordinal'])
    cites=o['citations']
    text=d['evidence'] if d else '; '.join(dict.fromkeys(cites))
    e=ev(text,cites,site) if d else ev('; '.join(dict.fromkeys(cites)),cites,site)
    r=dict(ordinal=o['ordinal'],siteId=o['siteId'],site=site,T=o['T'],N=o['N'],external=o['external'],notACall=o['notACall'],dynamic=o['dynamic'],
      scored=o['scored'],notScoredReason=o['notScoredReason'],basis=basis,strata=o['strata'],evidence=e)
    fin.append(r)
# label-set equality vs old
for r,o in zip(fin,old):
    assert all(r[k]==o[k] for k in('T','N','external','notACall','dynamic','scored','notScoredReason','siteId','ordinal'))
# basis precedence checks
audited={q['siteId'] for q in Q if q['queueCategory']!='census' and any(c.startswith('audit-') for c in q['categories'])}|swept
for r in fin:
    if r['basis'] in('agreed','label-only'): assert r['siteId'] not in audited
    if r['basis']=='audit-confirmed': assert r['siteId'] in audited
    if 'AS' in r['strata']: assert r['basis']!='agreed'
open(A+'truth/go.final.jsonl','w').write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in fin))
N=len(fin); bc=collections.Counter(r['basis'] for r in fin)
# census closure
dl=set()
for i in I.values():
    for c in i['label']['callees']:
        if c['kind']!='external': dl.add((c['path'],c['defLine']))
    st=i['reference']['static']
    if st['kind'] in('internal','declared'): dl.add((st['path'],st['defLine']))
    for m in i['reference'].get('impls',[]): dl.add((m['path'],m['defLine']))
for r in fin:
    for t in r['T']: dl.add((t['path'],t['defLine']))
cset={(c['path'],c['defLine']) for c in cens}
assert dl==cset,(len(dl),len(cset),sorted(dl^cset)[:5])
# audit
au={}
for s in('AI','AE','AS'):
    Ns=onoise['auditTable'][s]; au[s]=dict(N_s=Ns['N_s'],n=Ns['n'],k=Ns['k'],wilson95=Ns['wilson95'])
# residual: r_s = audited target/kind errors in U (AS census -> contributes 0 anyway)
aud=rd('adjudication/go.audit.jsonl')
r_s={s:sum(1 for a in aud if a['row']=='site' and a['draw']=='U' and a['stratum']==s and not a['labelCorrect'] and a['errorKind'] in('target','kind')) for s in au}
per={}
for s,a in au.items():
    rem=a['N_s']-a['n']; up=wilson(r_s[s],a['n'])[1] if rem else 0.0
    per[s]=dict(N_s=a['N_s'],n=a['n'],unreadRemainder=rem,r_s=r_s[s],wilson95Upper=wilson(r_s[s],a['n'])[1],expectedResidual=round(rem*r_s[s]/a['n'],4),conservativeResidual=round(rem*up,4))
e_res=sum(p['expectedResidual'] for p in per.values())/N; e_res_c=sum(p['conservativeResidual'] for p in per.values())/N
W=onoise['adjudicated']['W_labelWrong']
e_all=onoise['handErrorRate']['point']; e_all_c=onoise['handErrorRate']['conservativeWilsonUpper']
e_all_labelonly=round((onoise['adjudicated']['wrong'].get('label',0)+sum(1 for a in aud if a['row']=='site' and a['draw']=='U' and not a['labelCorrect']))/N,4)
# open points
anon={}
for r in fin:
    for t in r['T']:
        if '<anon>' in t['qname'] or re.search(r'\.<anon>|#\d+$',t['qname']):
            anon[(t['path'],t['defLine'])]=t['qname']
def line(p,l):
    return open('/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/go/src/'+p,encoding='utf-8',errors='replace').read().split('\n')[l-1]
classline=[(p,l,line(p,l).strip()[:60]) for (p,l) in sorted(cset) if re.match(r'\s*type\s',line(p,l))]
exts=sorted({e for r in fin for e in r['external']})
nonstd=[e for e in exts if not e.startswith('stdlib') and e!='builtin']
open_pts=dict(classLineTargets=dict(found=len(classline),items=classline,statement='none found: scan of the %d census definition lines for a `type` declaration'%len(cset) if not classline else ''),
  anonymousLambdaIndices=dict(definitions=len(anon),withIndexSuffix=sum(1 for q in anon.values() if re.search(r'#\d+$',q)),bareAnon=sum(1 for q in anon.values() if not re.search(r'#\d+$',q)),items=[dict(path=p,defLine=l,qname=q) for (p,l),q in sorted(anon.items())]),
  externalsAbsentFromTree=dict(statement='none found as generated code: external labels are stdlib/builtin or Go module dependencies, not generated sources in the corpus tree',nonStdlibPackages=nonstd))
un=collections.Counter(r['notScoredReason'] for r in fin if not r['scored'])
noise=dict(lang='go',N=N,comparable=onoise['comparable'],referenceSilent=onoise['referenceSilent']['total'],
 agreement=dict(agreed=onoise['agreement']['overall']['agreed'],comparable=onoise['agreement']['overall']['comparable'],rate=onoise['agreement']['overall']['rate'],wilson95=onoise['agreement']['overall']['wilson95']),
 adjudicated=dict(n=onoise['adjudicated']['n'],W=W,byWrong=onoise['adjudicated']['wrong'],byErrorKind=onoise['adjudicated']['errorKind']),
 referenceError=dict(wrong=onoise['referenceErrorRate']['adjudicatedReferenceWrong'],comparable=onoise['referenceErrorRate']['outOf'],rate=onoise['referenceErrorRate']['rate'],wilson95=onoise['referenceErrorRate']['wilson95']),
 audit=au,
 handError=dict(e_all=dict(point=e_all,conservative=e_all_c),e_res=dict(point=round(e_res,4),conservative=round(e_res_c,4)),noiseLimited=e_res>0.05),
 census=dict(targetsRead=len(cset)),
 closedClasses=dict(count=0,byKind=dict(role=[],coordinate=[],shape=[])),
 unscored=dict(byReason=dict(un),underivable=un.get('underivable',0)),
 basisCounts=dict(bc),
 detail=dict(original_v1_11=onoise,v1_12=dict(perStratumResidual=per,e_all_labelOnlyVariant=e_all_labelonly,
   v1_10_would_have_said=dict(figure='e_h point (with role errors)',point=e_all,conservative=e_all_c,flag=e_all>0.05,note='v1.10 flag used e_h point 0.0268 -> not noise-limited; the conservative 0.0768 was reported beside it'),
   closedClassShare=dict(count=0,share=0.0,cap=0.10,tripwire=False),
   sweep=dict(trigger='AS ordinal 49 (target error), cell external:stdlib:time',sites=sorted(swept),overturns=0,basisNote='ordinals 69 and 360 (re-read in the sweep, not queued) moved agreed -> audit-confirmed by v1.12 step 6 precedence'),
   basisChanges=dict(agreedToAuditConfirmed=precedence_changes),openPointsS27=open_pts,
   note='e_res excludes AS (census, N_s=n); ordinal 49 is corrected in final and leaves no residual')))
json.dump(noise,open(A+'adjudication/go.noise.json','w'),indent=1)
print('basis',dict(bc),'changed',precedence_changes)
print('e_all',e_all,e_all_c,'e_res',noise['handError']['e_res'],'flag',noise['handError']['noiseLimited'])
print('per',json.dumps(per))
print('census',len(cset),'anon defs',len(anon),'classline',len(classline),'nonstd ext',nonstd)
