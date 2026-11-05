import json,math,collections as C,re
R='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/'
A=R+'adjudication/'
S='/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/13504717-4242-422a-b7d8-a910a6cc9348/scratchpad/'
def rj(p): return [json.loads(l) for l in open(p)]
inp=rj(A+'python.input.jsonl'); q=rj(A+'python.queue.jsonl'); dec=rj(A+'python.decisions.jsonl'); ext=rj(S+'s57_python_ds/python.census-ext.jsonl')
smp={o['siteId']:o for o in rj(R+'sample/python.jsonl')}
def wil(k,n,z=1.959964):
    if n==0: return [None,None]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0,c-h),4),round(min(1,c+h),4)]
def key(x): return f"census:{x['file']}:{x['line']}" if x['queueCategory']=='census' else str(x['ordinal'])
qk={key(x):x for x in q}; dk={x['queueKey']:x for x in dec}
assert set(qk)==set(dk) and len(qk)==len(q)==len(dec)
CLOSED={'role','coordinate','shape'}
def rek(x):  # recode schema -> coordinate (v1.11 error-kind set has no 'schema')
    return 'coordinate' if x['errorKind']=='schema' else x['errorKind']
census={}
for x in dec:
    if x['queueCategory']=='census': census[(x['file'],x['line'])]={'role':x['role'],'form':x['form'],'body':x['body'],'targetKind':x['targetKind'],'labelRoleError':x['labelRoleError'],'errorSource':x['errorSource'],'source':'queue-152'}
fcform={(x['file'],x['line']):x['fcForm'] for x in q if x['queueCategory']=='census'}
for e in ext:
    census[(e['path'],e['defLine'])]={'role':e['role'],'form':e['form'],'body':e['body'],'targetKind':None,'labelRoleError':e['labelRoleError'],'errorSource':'none','source':'ds-ext-43'}
by={x['ordinal']:x for x in inp}
# ---- python.jsonl
out=open(A+'python.jsonl','w')
adj=[x for x in dec if x['queueCategory']=='adjudicate']
for x in sorted(adj,key=lambda x:x['ordinal']):
    r=by[x['ordinal']]
    d={'ordinal':x['ordinal'],'siteId':x['siteId'],'reasons':r['reasons'],'decision':x['decision'],'callees':x['callees'],'external':x['external'],
       'nonSymbol':x['nonSymbol'],'notACall':x['notACall'],'dynamic':x['dynamic'],'wrong':x['wrong'],'errorKind':rek(x),'refErrorKind':x['refErrorKind'],'evidence':x['evidence'][:300]}
    if rek(x)!=x['errorKind']: d['recodedFrom']=x['errorKind']
    out.write(json.dumps(d,ensure_ascii=False)+'\n')
out.close()
# ---- audit
def cell_of(o): return qk[str(o)]['cell']
au=open(A+'python.audit.jsonl','w')
audrows=[x for x in dec if x['queueCategory'].startswith('audit')]
for x in sorted(audrows,key=lambda x:(x['queueCategory'],x['stratum'],x['ordinal'])):
    qq=qk[x['queueKey']]
    d={'ordinal':x['ordinal'],'siteId':x['siteId'],'draw':'U' if x['queueCategory']=='audit-U' else 'C','stratum':x['stratum'],'drawRank':qq.get('drawRank'),'cell':qq['cell'],
       'labelCorrect':x['labelCorrect'],'errorKind':rek(x),'wrong':x['wrong'],'refErrorKind':x['refErrorKind'],'outcome':x['basis'],'evidence':x['evidence'][:300]}
    au.write(json.dumps(d,ensure_ascii=False)+'\n')
au.close()
# ---- final
audited={x['ordinal']:x for x in audrows}
adjd={x['ordinal']:x for x in adj}
def stratum(r):
    if r['inCorpusLabel']: return 'AI'
    return 'AS' if r['reference']['static']['kind']=='silent' else 'AE'
fin=open(R+'truth/python.final.jsonl','w'); finrows=[]
for r in inp:
    o=r['ordinal']; s=smp[r['siteId']]; site=f"{s['path']}:{s['line']}"
    if o in adjd: x=adjd[o]; basis='adjudicated'; strata=['adjudicated']+r['reasons']
    elif o in audited:
        x=audited[o]; basis='audit-confirmed'; strata=[x['stratum'],'audit-'+x['queueCategory'][-1]]
    else: x=None; basis='agreed'; strata=[stratum(r)]
    if x is not None:
        T=[{'path':c['path'],'qname':c['qname'],'defLine':c['defLine'],'role':c['role']} for c in x['callees']]
        ext_=[x['external']] if x['external'] else []; N=x['nonSymbol']; nac=x['notACall']; dyn=x['dynamic']; ev=x['evidence']
    else:
        T=[]; ext_=[]
        for c in r['label']['callees']:
            if c['kind']=='external': ext_.append(c)
            else:
                T.append({'path':c['path'],'qname':c['qname'],'defLine':c['defLine'],'role':census[(c['path'],c['defLine'])]['role']})
        N=[]; nac=r['label']['notACall']; dyn=r['label']['dynamic']
        ev=(site+' call site; '+('; '.join(f"{t['path']}:{t['defLine']}" for t in T) or 'external')+'; label: '+r['label']['evidence'])[:300]
    if not re.search(r'[\w/.\-]+\.py:\d+',ev): ev=(site+' '+ev)[:300]
    scored=not(nac or (dyn and not T))
    row={'ordinal':o,'siteId':r['siteId'],'site':site,'T':T,'N':N,'external':ext_,'notACall':nac,'dynamic':dyn,'scored':scored,
         'notScoredReason':None if scored else ('notACall' if nac else 'dynamic-empty'),'basis':basis,'strata':strata,'evidence':ev}
    finrows.append(row); fin.write(json.dumps(row,ensure_ascii=False)+'\n')
fin.close()
# ---- noise
N=len(inp); comp=[r for r in inp if r['reference']['static']['kind']!='silent']; silent=N-len(comp)
agr=[r for r in comp if r['status']=='agreed']
def rate(sub): 
    a=sum(r['status']=='agreed' for r in sub); return {'agreed':a,'comparable':len(sub),'rate':round(a/len(sub),4) if sub else None,'wilson95':wil(a,len(sub))}
lab_in=[r for r in comp if r['inCorpusLabel']]; lab_out=[r for r in comp if not r['inCorpusLabel']]
decl=[r for r in comp if r['declaredLabel']]
flags=C.Counter(k for r in inp for k in ('relaxed','ambiguous','byConstruction','typeCtorMerged') if r['reference'].get(k))
wk=C.Counter((x['wrong'],rek(x)) for x in adj)
W=sum(1 for x in adj if x['wrong'] in('label','both'))
closed_inst=sorted(x['ordinal'] for x in adj if x['wrong'] in('label','both') and rek(x) in CLOSED)
Wres=W-len(closed_inst)
Ns={'AI':92,'AE':71,'AS':4}; ns={'AI':60,'AE':60,'AS':4}
aud={}
for s in Ns:
    U=[x for x in audrows if x['stratum']==s and x['queueCategory']=='audit-U']; Cc=[x for x in audrows if x['stratum']==s and x['queueCategory']=='audit-C']
    k=sum(1 for x in U if not x['labelCorrect']); kres=sum(1 for x in U if not x['labelCorrect'] and rek(x) not in CLOSED)
    aud[s]={'N_s':Ns[s],'n':len(U),'errors_k':k,'residual_errors':kres,'wilson95':wil(k,len(U)),'drawC':{'n':len(Cc),'cell_errors':sum(1 for x in Cc if not x['labelCorrect']),'pooled':False},'flagged':len(U) and k/len(U)>0.05}
    assert len(U)==ns[s]
def e(Wx,resid,upper):
    t=Wx
    for s in aud:
        k=aud[s]['residual_errors' if resid else 'errors_k']; n=aud[s]['n']
        t+=aud[s]['N_s']*((wil(k,n)[1]) if upper else k/n)
    return round(t/N,4)
refW=sum(1 for x in adj if x['wrong'] in('reference','both') and by[x['ordinal']]['reference']['static']['kind']!='silent')
refk={s:sum(1 for x in audrows if x['stratum']==s and x['queueCategory']=='audit-U' and x['refErrorKind']!='none') for s in Ns}
reft=lambda up:round((refW+sum(Ns[s]*(wil(refk[s],ns[s])[1] if up else refk[s]/ns[s]) for s in ('AI','AE')))/len(comp),4)
# census
lab_ref=set()
for r in inp:
    for c in r['label']['callees']:
        if c['kind']!='external': lab_ref.add((c['path'],c['defLine']))
    for c in [r['reference']['static']]+r['reference']['impls']:
        if c.get('path'): lab_ref.add((c['path'],c['defLine']))
decided={(c['path'],c['defLine']) for x in dec if x['queueCategory']!='census' for c in x['callees']}
cens_keys=set(census)
nint=sum(1 for f in finrows if f['T']); typeany=sum(1 for f in finrows if any(census[(t['path'],t['defLine'])]['targetKind']=='type' for t in f['T'] if census[(t['path'],t['defLine'])]['targetKind'])); 
typeall=sum(1 for f in finrows if f['T'] and all(census[(t['path'],t['defLine'])]['targetKind']=='type' for t in f['T'] if census[(t['path'],t['defLine'])]['targetKind']))
ckind=C.Counter()
noise={'lang':'python','N':N,'assembledBy':'qa-engineer FC (mechanical); decisions by data-scientist',
 'comparable':{'comparable':len(comp),'referenceSilent':silent,'share':round(len(comp)/N,4)},
 'agreement':{**rate(comp),'perStratum':{'AI(label has in-corpus target)':rate(lab_in),'AE(label has none)':rate(lab_out),'declared':rate(decl) if decl else None,'relaxed':None,'byConstruction':None,'macro':None},'referenceFlagCounts':dict(flags)},
 'adjudicated':{'n':len(adj),'byWrongAndErrorKind':{f"{a}|{b}":v for (a,b),v in sorted(wk.items())},'W_labelWrong':W,'errorKindRecode':'schema->coordinate for ordinals '+str(sorted(x['ordinal'] for x in adj if x['errorKind']=='schema'))},
 'auditTable':aud,
 'closedClass':{'kinds':sorted(CLOSED),'instancesByOrdinal':{'coordinate':closed_inst,'role':[],'shape':[]},'count':len(closed_inst),'shareOfN':round(len(closed_inst)/N,4),'condIV_le_0.10':len(closed_inst)/N<=0.10,
   'labelRoleErrorsInCensus':sum(1 for v in census.values() if v['labelRoleError'])},
 'handError':{'e_all':e(W,False,False),'e_all_conservative':e(W,False,True),'e_res':e(Wres,True,False),'e_res_conservative':e(Wres,True,True),
   'W':W,'W_residual':Wres,'flagUses':'e_res','noiseLimited':e(Wres,True,False)>0.05,
   'note':'conservative = each stratum Wilson upper bound for k/n (AS: n=N_s=4 drawn in full; literal Wilson bound 0/4 used, a census would give 0)'},
 'referenceError':{'adjudicatedWrongOnComparable':refW,'comparable':len(comp),'rate':round(refW/len(comp),4),'wilson95':wil(refW,len(comp)),'auditRefErrors':refk,'rate_with_audit_extrapolation':reft(False),'conservative':reft(True)},
 'census':{'distinctLabelAndReference':len(lab_ref),'dsDecidedNotInThose':len(decided-lab_ref),'dsExtList':len(ext),
   'targetsRead':len(cens_keys),'queue152':sum(1 for v in census.values() if v['source']=='queue-152'),'ext43':sum(1 for v in census.values() if v['source']=='ds-ext-43'),
   'equalsDistinctLabelRefDecided':cens_keys==(lab_ref|decided),'labelRefAllRead':lab_ref<=cens_keys,'decidedAllRead':decided<=cens_keys,
   'forms':dict(C.Counter(v['form'] for v in census.values()))},
 'sampleLength':{'n_int_final':nint,'classTargetSites_anyTypeTarget':typeany,'classTargetSites_onlyTypeTargets':typeall,'ruleOutcome':'n_int>=150: nothing' if nint>=150 else 'see 5.4','note':'whether class-target sites count toward n_int is an open point for S27, not decided here'},
 'unscored':dict(C.Counter(f['notScoredReason'] for f in finrows if not f['scored'])),'underivable':0,
 'basisCounts':dict(C.Counter(f['basis'] for f in finrows))}
json.dump(noise,open(A+'python.noise.json','w'),indent=1,ensure_ascii=False)
print(json.dumps({k:noise[k] for k in('comparable','agreement','handError','referenceError','census','sampleLength','basisCounts','unscored','closedClass')},indent=1))
print(aud)
