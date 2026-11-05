import json,math,collections,sys,re
sys.path.insert(0,'/tmp')
R='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/'
AD=R+'adjudication/'
J=lambda p:[json.loads(l) for l in open(p)]
dec=J(AD+'cpp.decisions.jsonl'); inp={r['ordinal']:r for r in J(AD+'cpp.input.jsonl')}
refs={r['ordinal']:r for r in J(R+'truth/cpp.ref.jsonl')}
queue=J(AD+'cpp.queue.jsonl')
A={x['ordinal']:x for x in dec if x['queueCategory']=='adjudicate'}
AU={x['ordinal']:x for x in dec if x['queueCategory']=='audit'}
M={x['macroName']:x for x in dec if x['queueCategory']=='macro'}
C={(x['path'],x['defLine']):x for x in dec if x['queueCategory']=='census'}
qa={x['ordinal']:x for x in queue if x['queueCategory']=='audit'}
macro_of={s:m for m,x in M.items() for s in x['sites']}
def wil(k,n,z=1.96):
    if n==0: return [None,None]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0,c-h),4),round(min(1,c+h),4)]
def cell(r):
    l=r['label']
    if l['notACall']: return 'notACall'
    ins=sorted({(c['kind'],c['path']) for c in l['callees'] if c['kind']!='external'})
    if ins: return 'in:'+';'.join(f"{k}@{p}" for k,p in ins)
    return ('ext:'+';'.join(sorted({c['package'] for c in l['callees']}))) if l['callees'] else 'empty'
def stratum(r):
    if r['reference']['static']['kind']=='silent': return 'AS'
    return 'AI' if r['inCorpusLabel'] or r['declaredLabel'] else 'AE'
# ---- outputs: cpp.jsonl (decisions)
out=open(AD+'cpp.jsonl','w')
def tt(c):  # normalise a decided callee
    return {"path":c['path'],"qname":c.get('qname'),"defLine":c['defLine'],"role":c['role'],
            **({"macroGenerated":True} if c.get('macroGenerated') else {}),**({"qnameDerived":True} if c.get('qnameDerived') else {})}
dec_rows={}
for o in sorted(inp):
    r=inp[o]
    if r['status']!='needs-adjudication' and o not in macro_of: continue
    a=A.get(o); m=M.get(macro_of.get(o))
    if a:
        row={"ordinal":o,"siteId":r['siteId'],"reasons":r['reasons'],"via":a['via'],"decision":a['decision'],
             "callees":a['callees'],"external":a['external'],"notACall":a['decision']=='notACall',"dynamic":a['decision']=='dynamic',
             "wrong":a['wrong'],"errorKind":a['errorKind'],"scored":a['scored'],"notScoredReason":a['notScoredReason'],
             "basis":a['basis'],"evidence":a['evidence'][:300]}
    else:  # agreed macro site decided via its macro record
        row={"ordinal":o,"siteId":r['siteId'],"reasons":r['reasons'],"via":"macro:"+m['macroName'],"decision":m['decision'],
             "callees":m['callees'],"external":m['external'],"notACall":m['notACall'],"dynamic":False,"wrong":None,"errorKind":None,
             "scored":not m['notACall'],"notScoredReason":'notACall' if m['notACall'] else None,"basis":m['basis'],"evidence":m['evidence'][:300]}
    dec_rows[o]=row
for o,row in dec_rows.items(): out.write(json.dumps(row)+"\n")
out.close()
# audit file
with open(AD+'cpp.audit.jsonl','w') as f:
    for x in sorted((x for x in AU.values()),key=lambda x:(x['draw'],x['stratum'],qa[x['ordinal']]['drawRank'])):
        f.write(json.dumps({k:x[k] for k in ('ordinal','siteId','path','line','stratum','draw','cell','labelCorrect','errorKind','decision','basis','evidence')}|{"drawRank":qa[x['ordinal']]['drawRank']})+"\n")
with open(AD+'cpp.census.jsonl','w') as f:
    for x in sorted(C.values(),key=lambda x:(x['path'],x['defLine'])):
        f.write(json.dumps({k:v for k,v in x.items() if k!='queueCategory'})+"\n")
# ---- final
def censrole(p,l):
    c=C[(p,l)]; return c['roleDecided'],c
final=[]
for o in sorted(inp):
    r=inp[o]; T=[];N=[];ext=[];nac=False;dyn=False;basis=None;ev=None
    strata=[stratum(r)]
    for k in ('relaxed','ambiguous','byConstruction'):
        if r['reference'].get(k): strata.append(k)
    if o in dec_rows:
        d=dec_rows[o]; basis='adjudicated'
        if d['via'].startswith('macro:'): strata.append('macro')
        if d['wrong']=='label' or d['wrong']=='both':
            pass
        for c in d['callees']:
            (N if c['role']=='N' else T).append(tt(c) if c['role']!='N' else {"path":c['path'],"defLine":c['defLine']})
        ext=d['external']; nac=d['notACall']; dyn=d['decision']=='dynamic'; ev=d['evidence']
        scored=d['scored']; nsr=d['notScoredReason']
        if o in AU:
            strata.append('audit-'+AU[o]['draw'])
    else:
        lab=r['label']
        for c in lab['callees']:
            if c['kind']=='external': ext.append(c); continue
            role,cs=censrole(c['path'],c['defLine'])
            if cs.get('correctedRole'): role='decl'
            if role in('impl','decl'): T.append({"path":c['path'],"qname":c['qname'],"defLine":c['defLine'],"role":role})
            elif role=='type': pass   # DS ruling: wrong target, belongs in neither T nor N
            else: N.append({"path":c['path'],"defLine":c['defLine'],"form":role})
        nac=lab['notACall']; dyn=lab['dynamic']
        if o in AU:
            a=AU[o]; basis='audit-confirmed' if a['labelCorrect'] else 'audit-overturned'; ev=a['evidence']; strata.append('audit-'+a['draw'])
        else: basis='agreed'
        scored=not nac and not(dyn and not T); nsr='notACall' if nac else None
    if not ev: ev=f"{r['site']['path']}:{r['site']['line']}"
    cit=f"{r['site']['path']}:{r['site']['line']}"
    if not re.search(r'\S+:\d+',ev): ev=cit+' '+ev
    if not scored and nsr is None: nsr='dynamic' if dyn else 'underivable'
    final.append({"ordinal":o,"siteId":r['siteId'],"site":cit,"T":T,"N":N,"external":ext,"notACall":nac,"dynamic":dyn,
                  "scored":scored,"notScoredReason":nsr,"basis":basis,"strata":strata,"evidence":ev[:300]})
with open(R+'truth/cpp.final.jsonl','w') as f:
    for x in final: f.write(json.dumps(x)+"\n")
# ---- census closure
distinct=set(C)
lab_ref={(c['path'],c['defLine']) for r in inp.values() for c in r['label']['callees'] if c['kind']!='external'}
for r in refs.values():
    if r['ordinal'] in inp:
        for x in [r['static']]+r.get('impls',[]):
            if x.get('kind') in('internal','declared') and 'path' in x: lab_ref.add((x['path'],x['defLine']))
dec_t={(c['path'],c['defLine']) for x in list(A.values())+list(M.values()) for c in x['callees']}
need=lab_ref|dec_t
missing=sorted({(t['path'],t['defLine']) for x in final for t in x['T']+x['N']}-distinct)
print('closure missing',missing)
# ---- noise
Nn=len(inp)
silent=sum(r['reference']['static']['kind']=='silent' for r in inp.values()); comp=Nn-silent
agreed=[o for o,r in inp.items() if r['status']=='agreed']
ag_comp=[o for o in agreed if inp[o]['reference']['static']['kind']!='silent']
Wl=[o for o,x in A.items() if x['wrong'] in('label','both')]
ker=collections.Counter(A[o]['errorKind'] for o in Wl)
closed_shape=sorted(o for o in Wl if A[o]['errorKind']=='shape' or A[o]['errorKind']=='schema')
roleerr=[x for x in C.values() if x['errorKind']=='role']
role_sites=sorted({s for x in roleerr for s in x['siteOrdinals']})
role_extra=[s for s in role_sites if s not in Wl]
W=len(Wl)
# audit table
tab={}
for s in ('AI','AE','AS'):
    U=[x for x in AU.values() if x['stratum']==s and x['draw']=='U']
    Ns=sum(1 for o in agreed if stratum(inp[o])==s)
    k=sum(not x['labelCorrect'] for x in U)
    tab[s]={"N_s":Ns,"n":len(U),"k":k,"census":len(U)==Ns,"wilson":wil(k,len(U)) if len(U)<Ns else None}
Cd=[x for x in AU.values() if x['draw']=='C']
tab['drawC']={"n":len(Cd),"errors":sum(not x['labelCorrect'] for x in Cd),"cells":[x['cell'] for x in Cd],"pooled":False}
def extrap(use_upper):
    t=0.0
    for s,v in tab.items():
        if s=='drawC' or v['census']: continue
        t+=v['N_s']*((v['wilson'][1]) if use_upper else v['k']/v['n'])
    return t
def rates(W_,extra,subsetN):
    pt=(W_+extra+extrap(False))/subsetN; ub=(W_+extra+extrap(True))/subsetN
    return {"point":round(pt,4),"conservative":round(ub,4)}
res_W=W-len(closed_shape)
e_all=rates(W,len(role_extra),Nn); e_res=rates(res_W,0,Nn)
# macro vs non-macro
macro_set={o for o in inp if o in macro_of}
Wm=[o for o in Wl if o in macro_set]; Wn=[o for o in Wl if o not in macro_set]
def sub(Ws,n): return {"N":n,"W":len(Ws),"residualW":len([o for o in Ws if A[o]['errorKind'] not in('shape','schema')]),"W_over_N":round(len(Ws)/n,4),"wilson":wil(len(Ws),n),
                       "byKind":dict(collections.Counter(A[o]['errorKind'] for o in Ws))}
# reference errors
refW=[o for o,x in A.items() if x['wrong'] in('reference','both') and inp[o]['reference']['static']['kind']!='silent']
relaxed=[o for o,r in inp.items() if r['reference'].get('relaxed')]
tcm=[o for o in inp if refs[o].get('typeCtorMerged')]
byc=[o for o,r in inp.items() if r['reference'].get('byConstruction')]; amb=[o for o,r in inp.items() if r['reference'].get('ambiguous')]
def strat_agree(sel):
    sel=[o for o in sel if inp[o]['reference']['static']['kind']!='silent']
    a=sum(inp[o]['status']=='agreed' for o in sel); return {"comparable":len(sel),"agreed":a,"rate":round(a/len(sel),4) if sel else None,"wilson":wil(a,len(sel))}
declared=[o for o,r in inp.items() if r['declaredLabel']]
# per-cell
cells=collections.defaultdict(lambda:{"sites":0,"labelErrors":0,"residualLabelErrors":0,"auditU":0,"auditUerr":0,"auditC":0,"auditCerr":0})
for o,r in inp.items():
    c=cell(r); d=cells[c]; d['sites']+=1
    if o in Wl: d['labelErrors']+=1; d['residualLabelErrors']+= o not in closed_shape
    if o in AU:
        k='auditU' if AU[o]['draw']=='U' else 'auditC'; d[k]+=1; d[k+'err']+= not AU[o]['labelCorrect']
unders=sum(1 for x in final if x['notScoredReason']=='macro-underdetermined')
und=sum(1 for x in final if x['notScoredReason']=='underivable')
noise={"lang":"cpp","N":Nn,
 "comparable":{"comparable":comp,"silent":silent,"share":round(comp/Nn,4)},
 "agreement":{"agreed_all":len(agreed),"agreed_comparable":len(ag_comp),"rate":round(len(ag_comp)/comp,4),"wilson95":wil(len(ag_comp),comp),
   "note":"AS agreed sites are single-source and excluded from numerator and denominator",
   "perStratum":{"AI_ref_internal":strat_agree([o for o,r in inp.items() if r['reference']['static']['kind']=='internal']),
     "AE_ref_external_local":strat_agree([o for o,r in inp.items() if r['reference']['static']['kind'] in('external','local')]),
     "relaxed":strat_agree(relaxed),"byConstruction":strat_agree(byc),"ambiguous":strat_agree(amb),"declared":strat_agree(declared),
     "macro":strat_agree(sorted(macro_set))}},
 "adjudicated":{"n":len(A),"byWrong":dict(collections.Counter(x['wrong'] for x in A.values())),"byErrorKind":dict(collections.Counter(x['errorKind'] for x in A.values())),
   "W":W,"W_note":"adjudicated sites with the label wrong (wrong in label|both)"},
 "audit":tab,
 "census":{"targetsRead":len(C),"distinctLabelReferenceAndDecidedTargets":len(need),"equal":set(C)==need,"missingFromCensus":len(need-set(C)),"extraInCensus":len(set(C)-need),"roleErrors":[{"path":x['path'],"defLine":x['defLine'],"siteOrdinals":x['siteOrdinals']} for x in roleeer] if False else [{"path":x['path'],"defLine":x['defLine'],"siteOrdinals":x['siteOrdinals']} for x in roleerr],"fcDiffers":sum(bool(x['fcDiffers']) for x in C.values())},
 "closedClasses":{"shape":{"ordinals":closed_shape,"note":"empty callees with neither notACall nor dynamic; found by the script/schema scan of every label line, corrected in final"},
   "role":{"ordinals":role_sites,"note":"census item gmock-function-mocker_test.cc:126; the site (69) is already in W as a target error, so it adds nothing to W and stays residual",
           "alsoInW":[s for s in role_sites if s in Wl]},
   "instances":len(closed_shape)+len(role_sites),"share_of_N":round((len(closed_shape)+len(role_sites))/Nn,4),"cap":0.10},
 "handErrorRate":{"e_all":e_all,"e_res":e_res,"W_residual":res_W,"extrapolatedTerm_point":round(extrap(False),3),"extrapolatedTerm_conservative":round(extrap(True),3),
   "note":"AE and AS audits are censuses (no extrapolation); AI term N_s*k/n and N_s*Wilson upper (k=0, n=60); draw C not pooled"},
 "noiseLimited":e_res['point']>0.05,"noiseLimitedBasis":"e_res point estimate > 0.05","conservativeAlso_above_0.05":e_res['conservative']>0.05,
 "referenceError":{"adjudicatedRefErrors_comparable":len(refW),"comparable":comp,"point":round(len(refW)/comp,4),"wilson95":wil(len(refW),comp),
   "conservative_with_AI_audit_upper":round((len(refW)+tab['AI']['N_s']*(tab['AI']['wilson'][1] or 0))/comp,4)},
 "macroStratum":{"viaMacro":sub(Wm,len(macro_set)),"nonMacro":sub(Wn,Nn-len(macro_set)),
   "macroUnderdetermined":{"count":unders,"share":round(unders/Nn,4),"cap":0.30},
   "relabelCost":{"macroSites":len(macro_set),"distinctMacros":len(M),"finalLinesWhoseBasisWouldChange":sum(1 for x in final if 'macro' in x['strata'])}},
 "relaxed":{"n":len(relaxed),"referenceErrors":len([o for o in refW if o in relaxed]),"share":round(len([o for o in refW if o in relaxed])/len(relaxed),4),"cap":0.10},
 "typeCtorMerged":{"sites":len(tcm),"referenceErrors":len([o for o in refW if o in tcm]),"withdrawAt":3},
 "byConstruction":{"n":len(byc),"referenceErrors":len([o for o in refW if o in byc])},"ambiguous":{"n":len(amb),"referenceErrors":len([o for o in refW if o in amb])},
 "unscored":dict(collections.Counter(x['notScoredReason'] for x in final if not x['scored'])),"underivable":{"count":und,"share":round(und/Nn,4),"cap":0.05},
 "perCellErrors":dict(sorted(cells.items())),
 "siblingSweep":"not triggered: 0 audit errors"}
json.dump(noise,open(AD+'cpp.noise.json','w'),indent=1)
print(json.dumps({k:noise[k] for k in('comparable','agreement','adjudicated','handErrorRate','noiseLimited','referenceError','relaxed','typeCtorMerged','unscored','underivable')},indent=1)[:5000])
print(json.dumps(noise['macroStratum'],indent=1)); print(json.dumps(noise['closedClasses'])); print(tab)
