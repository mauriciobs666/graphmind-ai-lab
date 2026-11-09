import json,math,collections,re,hashlib

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
R=(_CPG+'/spikes/frontend-fitness/')
A=R+'adjudication/'
def jl(p): return [json.loads(l) for l in open(p)]
dec=jl(A+'kotlin.decisions.jsonl'); inp=jl(A+'kotlin.input.jsonl'); labels={o['ordinal']:o for o in jl(R+'labels/kotlin.jsonl')}
ref={o['ordinal']:o for o in jl(R+'truth/kotlin.ref.jsonl')}
byord={r['ordinal']:r for r in inp}
census={(d['path'],d['defLine']):d for d in dec if d['category']=='census'}
def wil(k,n,z=1.96):
    if n==0: return (None,None)
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (round(max(0,c-h),4),round(min(1,c+h),4))
def stratum(r):
    if r['status']!='agreed': return None
    if r['inCorpusLabel'] or r['declaredLabel']: return 'AI'
    return 'AS' if r['reference']['static']['kind']=='silent' else 'AE'
CELL={(q['category'],q['ordinal']):q['cell'] for q in jl(A+'kotlin.queue.jsonl') if q['category'] in('audit-U','audit-C')}
# decisions by category/ordinal
D=collections.defaultdict(dict)
for d in dec:
    if d['category']!='census': D[d['category']][d['ordinal']]=d
cap=lambda s:s[:300]
final=[];role_inst=[]
for o in sorted(labels):
    L=labels[o]; r=byord[o]; st=stratum(r)
    d=None;basis=None
    if o in D['adjudicate']: d=D['adjudicate'][o]; basis='adjudicated'
    elif o in D['audit-U'] and D['audit-U'][o]['basis']=='audit-overturned': d=D['audit-U'][o]; basis='audit-overturned'
    elif o in D['audit-C'] and D['audit-C'][o]['basis']=='audit-overturned': d=D['audit-C'][o]; basis='audit-overturned'
    elif o in D['audit-U']: d=D['audit-U'][o]; basis='audit-confirmed'
    elif o in D['audit-C']: d=D['audit-C'][o]; basis='audit-confirmed'
    T=[];ext=[];N=[]
    if d is not None and not (d['category'] in('audit-U','audit-C') and basis=='audit-confirmed'):
        S=d['set']
        for c in S['callees']: T.append({'path':c['path'],'qname':c['qname'],'defLine':c['defLine'],'role':c['role']})
        ext=[e['package'] for e in S['external']]; N=list(S['nonSymbol']); nac=S['notACall']; dyn=S['dynamic']
        scored=d['scored']; ev=d['evidence']
    else:
        # label as is, roles from census (audit-confirmed sets are re-derived from the label and cross-checked in validate)
        for c in L['callees']:
            if c['kind']=='external': ext.append(c['package'])
            else: T.append({'path':c['path'],'qname':c['qname'],'defLine':c['defLine'],'role':census[(c['path'],c['defLine'])]['role']})
        nac=L['notACall']; dyn=L['dynamic']; scored=(not nac) and not (dyn and not T)
        ev=d['evidence'] if d else L['evidence']
        if d is None: basis='agreed'
    for t in T:
        c=census.get((t['path'],t['defLine']))
        if c and c['decision']=='primary-ctor-decl' and any(cc['kind']=='internal' and (cc['path'],cc['defLine'])==(t['path'],t['defLine']) for cc in L['callees']):
            role_inst.append(o)
    tags=[st or 'ADJ']
    if o in D['audit-U']: tags.append('audit-U')
    if o in D['audit-C']: tags.append('audit-C')
    if any(c['kind']=='declared' for c in L['callees']) or r['reference']['static']['kind']=='declared': tags.append('declared')
    for k in ('relaxed','ambiguous','byConstruction'):
        if r['reference'].get(k): tags.append(k)
    s=r['site']
    if not re.search(r'[\w./-]+\.kt:\d+',ev): ev=f"{s['path']}:{s['line']} "+ev
    final.append({'ordinal':o,'siteId':L['siteId'],'T':T,'N':N,'external':ext,'notACall':nac,'dynamic':dyn,'scored':scored,
      'notScoredReason':None if scored else 'underivable','basis':basis,'strata':tags,'site':f"{s['path']}:{s['line']}",'evidence':cap(ev)})
with open(R+'truth/kotlin.final.jsonl','w') as f:
    for x in final: f.write(json.dumps(x,ensure_ascii=False)+'\n')
role_inst_final_T=sorted(set(role_inst)); role_inst=sorted({x for d in dec if d['category']=='census' and d['decision']=='primary-ctor-decl' for x in d['sites']})
# adjudication decisions file
with open(A+'kotlin.jsonl','w') as f:
    for o in sorted(set(D['adjudicate'])|{k for c in('audit-U','audit-C') for k,v in D[c].items() if v['basis']=='audit-overturned'}):
        if o in D['adjudicate']: d=D['adjudicate'][o]; cat='adjudicate'
        else: d=D['audit-U'].get(o) if o in D['audit-U'] and D['audit-U'][o]['basis']=='audit-overturned' else D['audit-C'][o]; cat=d['category']
        f.write(json.dumps({'ordinal':o,'siteId':d['siteId'],'source':cat,'reasons':byord[o]['reasons'],'decision':d['decision'],'basis':d['basis'],'set':d['set'],
           'wrong':d['wrong'],'errorKind':d['errorKind'],'scored':d['scored'],'evidence':d['evidence'],**({'note':d['note']} if 'note' in d else {})},ensure_ascii=False)+'\n')
with open(A+'kotlin.audit.jsonl','w') as f:
    for c in ('audit-U','audit-C','helper-audit'):
        for o in sorted(D[c]):
            d=D[c][o]; x={k:d[k] for k in d if k not in('category',)}; x['draw']={'audit-U':'U','audit-C':'C','helper-audit':'helper'}[c]
            if c=='audit-C': x['pooled']=False
            f.write(json.dumps(x,ensure_ascii=False)+'\n')
with open(A+'kotlin.census.jsonl','w') as f:
    for d in dec:
        if d['category']=='census': f.write(json.dumps(d,ensure_ascii=False)+'\n')
# ---------- noise
N_=430
comp=[r for r in inp if r['reference']['static']['kind']!='silent']
agr=[r for r in comp if r['status']=='agreed']
def cls(r): return 'internal' if (r['inCorpusLabel'] or r['declaredLabel'] or r['inCorpusReference']) else 'external'
agree={'overall':{'comparable':len(comp),'agreed':len(agr),'rate':round(len(agr)/len(comp),4),'wilson95':wil(len(agr),len(comp))}}
for name,fn in (('AI-class(internal label or reference)',lambda r:cls(r)=='internal'),('AE-class(no in-corpus target either side)',lambda r:cls(r)=='external'),
   ('declared(label or reference declared)',lambda r:bool(r['declaredLabel']) or r['reference']['static']['kind']=='declared')):
    c=[r for r in comp if fn(r)]; a=[r for r in c if r['status']=='agreed']
    agree[name]={'comparable':len(c),'agreed':len(a),'rate':round(len(a)/len(c),4) if c else None,'wilson95':wil(len(a),len(c))}
agree['relaxed']=agree['byConstruction']=agree['macro']='not present in Kotlin reference rows (0)'
adj=[D['adjudicate'][o] for o in sorted(D['adjudicate'])]
by=collections.Counter((d['wrong'],d['errorKind']) for d in adj)
closed_kinds={'role','coordinate','shape'}
W_all=sum(1 for d in adj if d['wrong'] in('label','both')); W_res=sum(1 for d in adj if d['wrong'] in('label','both') and d['errorKind'] not in closed_kinds)
pop={'AI':sum(1 for r in inp if stratum(r)=='AI'),'AE':sum(1 for r in inp if stratum(r)=='AE'),'AS':0}
audit={}
for st in('AI','AE'):
    U=[d for d in D['audit-U'].values() if d['stratum']==st]; C=[d for d in D['audit-C'].values() if d['stratum']==st]
    n=len(U); kall=sum(1 for d in U if not d['labelCorrect']); kres=sum(1 for d in U if not d['labelCorrect'] and d['errorKind'] not in closed_kinds)
    kref=sum(1 for d in U if d['wrong'] in('reference','both'))
    audit[st]={'populationNs':pop[st],'n':n,'errorsAll':kall,'errorsResidual':kres,'wilsonAll':wil(kall,n),'wilsonResidual':wil(kres,n),
      'flaggedOnResidual':kres/n>0.05 if n else False,'errorKindsAll':dict(collections.Counter(d['errorKind'] for d in U if not d['labelCorrect'])),
      'referenceErrorsInDraw':kref,
      'drawC':{'n':len(C),'cellErrors':sum(1 for d in C if not d['labelCorrect']),'errorKinds':dict(collections.Counter(d['errorKind'] for d in C if not d['labelCorrect'])),'pooled':False,
               'cellsWithErrors':[CELL[(d['category'],d['ordinal'])] for d in C if not d['labelCorrect']]}}
audit['AS']={'populationNs':0,'n':0,'note':'stratum empty: no agreed site has a silent reference'}
def ext(key,up):
    tot=W_all if key=='all' else W_res
    s=0
    for st in('AI','AE'):
        a=audit[st]; k=a['errorsAll'] if key=='all' else a['errorsResidual']; n=a['n']
        s+=a['populationNs']*((wil(k,n)[1]) if up else k/n)
    return round((tot+s)/N_,4)
noise_flag=ext('res',False)>0.05
# reference error rate
Rw=sum(1 for d in adj if d['wrong'] in('reference','both'))
kref_ext=sum(audit[s]['populationNs']*audit[s]['referenceErrorsInDraw']/audit[s]['n'] for s in('AI','AE'))
phant=[d for d in census.values() if d['decision']=='ref-phantom-impl']
phant_sites=sorted({s for d in phant for s in d['sites']})
oth=[d for d in census.values() if d['decision']=='primary-ctor-decl']
noclass=[d for d in census.values() if d['decision']=='type-confirmed']
dist=set()
for o,L in labels.items():
    for c in L['callees']:
        if c['kind']!='external': dist.add((c['path'],c['defLine']))
for o,rr in ref.items():
    if o not in labels: continue
    st=rr['static']
    if st['kind'] in('internal','declared'): dist.add((st['path'],st['defLine']))
    for i in rr.get('impls',[]): dist.add((i['path'],i['defLine']))
for d in dec:
    if d['category'] in('adjudicate','audit-U','audit-C') and 'set' in d:
        for c in d['set']['callees']: dist.add((c['path'],c['defLine']))
noise={'lang':'kotlin','N':N_,'agreement':agree,
 'adjudicated':{'n':len(adj),'byWrongErrorKind':{f'{k[0]}|{k[1]}':v for k,v in sorted(by.items())},'W_all':W_all,'W_residual':W_res},
 'auditTable':audit,
 'handError':{'e_all':ext('all',False),'e_all_conservative':ext('all',True),'e_res':ext('res',False),'e_res_conservative':ext('res',True),
   'flagUses':'e_res','noiseLimited':noise_flag,'formula':'(W + sum_s N_s*k_s/n_s)/N; conservative: k_s/n_s replaced by Wilson95 upper'},
 'referenceError':{'comparableSites':len(comp),'adjudicatedSitesReferenceWrong':Rw,'extrapolatedFromAuditU':kref_ext,
   'rate':round((Rw+kref_ext)/len(comp),4),
   'rate_conservative':round((Rw+sum(audit[s]['populationNs']*wil(audit[s]['referenceErrorsInDraw'],audit[s]['n'])[1] for s in('AI','AE')))/len(comp),4),
   'phantomImplTargets':len(phant),'phantomImplSites':phant_sites,'note':'phantom reference impls are census-level reference errors (not definitions); not included in T; not in the site-level rate'},
 'closedClasses':{'role_primaryConstructor':{'errorKind':'role','ctorTargets':len(oth),'instancesByOrdinal':role_inst,'nInstances':len(role_inst),'share':round(len(role_inst)/N_,4),
     'correction':'final T keeps the label target; role = decl from census','alsoTargetErrorsCountedResidual':[o for o in role_inst if o not in role_inst_final_T],'instancesCarryingCtorInFinalT':role_inst_final_T,'roleOnlyErrorsAdjudicated':[d['ordinal'] for d in adj if d['errorKind']=='role']},
   'coordinate':{'instances':[]},'shape':{'instances':[]}},
 'closedClassShareCap':0.10,
 'targetErrors':sorted(d['ordinal'] for d in adj if d['errorKind']=='target' and d['wrong'] in('label','both')),
 'census':{'targetsRead':len(census),'distinctLabelReferenceAndDecidedTargets':len(dist),'equal':len(census)==len(dist) and set(census)==dist,
    'decisions':dict(collections.Counter(d['decision'] for d in census.values()))},
 'underivable':{'n':sum(1 for f in final if f['notScoredReason']=='underivable'),'share':round(sum(1 for f in final if not f['scored'])/N_,4),'cap':0.05},
 'helperAudit':{'leads':len(D['helper-audit']),'outcomes':dict(collections.Counter(d['decision'] for d in D['helper-audit'].values())),'ordinals':sorted(D['helper-audit'])},
 'siblingSweep':{'performed':False,'reason':'every audit error is of the closed role class, found and corrected by the complete census; candidates listed in kotlin.sweep-candidates.json'},
 'openPoints':{
   'ordinal327':'two lambda targets recorded (KtorAdminTokenAuthProvider.kt defLines 149 and 261); anonymous-function indices (<anon>#N) not resolved in qname',
   'typeSymbolsWithoutPrimaryConstructor':{'assumption':'type symbols with role impl (data-scientist assumption, not decided here); for S27','items':[f"{d['path']}:{d['defLine']}" for d in noclass]},
   'copy synthesized':[97,354],
   'DS-introduced census rows (targets decided by DS, not listed by label or reference)':[327,388]},
 'basisCounts':dict(collections.Counter(f['basis'] for f in final))}
json.dump(noise,open(A+'kotlin.noise.json','w'),indent=1,ensure_ascii=False)
print(json.dumps({k:noise[k] for k in('handError','referenceError','closedClasses','census','underivable','helperAudit','basisCounts','targetErrors')},indent=1)[:5000])
print(json.dumps(audit,indent=0)[:1500])
