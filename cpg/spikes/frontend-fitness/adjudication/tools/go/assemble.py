import json,collections,re,math,sys
A='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/'
rd=lambda p:[json.loads(l) for l in open(A+p) if l.strip()]
I=rd('adjudication/go.input.jsonl'); Q=rd('adjudication/go.queue.jsonl'); D=rd('adjudication/go.decisions.jsonl')
byid={i['siteId']:i for i in I}
def wilson(k,n,z=1.96):
    if n==0: return [None,None]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0,c-h),4),round(min(1,c+h),4)]
# decisions
sd=[d for d in D if d['kind']=='site']; cd=[d for d in D if d['kind']=='census']
# ord 27: keep audit-confirmed for final; sweep line separate
final_dec={}
sweeps=[]
for d in sd:
    if d['basis']=='sweep-confirmed': sweeps.append(d); continue
    assert d['siteId'] not in final_dec; final_dec[d['siteId']]=d
role={(c['path'],c['defLine']):c['role'] for c in cd}
cens_ok=all(c['labelOK'] for c in cd)
qsite={q['siteId']:q for q in Q if q['queueCategory']!='census'}
def cite_of(d): return d['citations']
def ev(d):
    s=d['evidence']; c=d['citations'][0] if d['citations'] else ''
    if len(s)>300: s=s[:297]+'...'
    if c and c not in s: s=(s[:300-len(c)-3]+' ['+c+']') if len(s)+len(c)+3>300 else s+' ['+c+']'
    return s
def dec_out(d):
    x=d['decision']
    return dict(callees=x['callees'],external=x['external'],nonSymbol=x['nonSymbol'],notACall=x['notACall'],dynamic=x['dynamic'])
# go.jsonl
out=[]
for d in sorted([d for d in final_dec.values() if d['basis'] in('adjudicated','audit-overturned')]+sweeps,key=lambda d:d['ordinal']):
    i=byid[d['siteId']]
    out.append(dict(ordinal=d['ordinal'],siteId=d['siteId'],reasons=i['reasons'] if d['basis']!='sweep-confirmed' else ['sibling-sweep'],
      decided=dec_out(d),wrong=d['wrong'],errorKind=d['errorKind'],basis=d['basis'],scored=d['scored'],evidence=ev(d),citations=d['citations']))
open(A+'adjudication/go.jsonl','w').write(''.join(json.dumps(o,ensure_ascii=False)+'\n' for o in out))
# audit.jsonl
aud=[]
for q in Q:
    if q['queueCategory']=='census': continue
    for c in q['categories']:
        if c.startswith('audit-'):
            dr,s=c.split(':')[0][-1],c.split(':')[1]
            d=final_dec[q['siteId']]
            aud.append(dict(row='site',draw=dr,stratum=s,ordinal=q['ordinal'],siteId=q['siteId'],cell=q['cell'],labelCorrect=d['labelCorrect'],errorKind=d['errorKind'],basis=d['basis'],evidence=ev(d),citations=d['citations'],pooledIntoRate=(dr=='U')))
for d in sweeps:
    aud.append(dict(row='sibling-sweep',draw=None,stratum=d['stratum'],ordinal=d['ordinal'],siteId=d['siteId'],cell='external:stdlib:time',labelCorrect=d['labelCorrect'],errorKind=d['errorKind'],basis=d['basis'],evidence=ev(d),citations=d['citations'],pooledIntoRate=False))
for c in cd:
    aud.append(dict(row='census',path=c['path'],defLine=c['defLine'],form=c['form'],body=c['body'],role=c['role'],rule8=c['rule8'],labelOK=c['labelOK'],errorKind=c['errorKind'],citations=c['citations']))
open(A+'adjudication/go.audit.jsonl','w').write(''.join(json.dumps(o,ensure_ascii=False)+'\n' for o in aud))
# final
def stratum(i):
    if i['status']!='agreed': return None
    if i['inCorpusLabel'] or i['declaredLabel']: return 'AI'
    return 'AS' if i['reference']['static']['kind']=='silent' else 'AE'
fin=[]
for i in sorted(I,key=lambda i:i['ordinal']):
    d=final_dec.get(i['siteId']); sw=[s for s in sweeps if s['siteId']==i['siteId']]
    st=[]
    s0=stratum(i)
    if s0: st.append(s0)
    if i['declaredLabel']: st.append('declared')
    if i['reference'].get('byConstruction'): st.append('byConstruction')
    if i['reference']['static']['kind']=='dispatch': st.append('go-dispatch')
    if i['reference']['static']['kind']=='silent': st.append('reference-silent')
    if sw: st.append('sibling-swept')
    cite=[f"{i['site']['path']}:{i['site']['line']}"]
    if d:
        x=d['decision']; T=[dict(path=c['path'],qname=c['qname'],defLine=c['defLine'],role=c['role']) for c in x['callees']]
        basis=d['basis']; N=x['nonSymbol']; ext=x['external']; nac=x['notACall']; dyn=x['dynamic']; scored=d['scored']; cite=d['citations']+cite
    else:
        T=[]
        for c in i['label']['callees']:
            if c['kind'] in('internal','declared'): T.append(dict(path=c['path'],qname=c['qname'],defLine=c['defLine'],role=role[(c['path'],c['defLine'])]))
        N=[]; ext=sorted({c['package'] for c in i['label']['callees'] if c['kind']=='external'}); nac=i['label']['notACall']; dyn=i['label']['dynamic']
        basis='agreed'; scored=not nac and not (dyn and not T)
    nsr=None
    if not scored: nsr='notACall' if nac else ('dynamic' if dyn else 'underivable')
    fin.append(dict(ordinal=i['ordinal'],siteId=i['siteId'],site=i['site'],T=T,N=N,external=ext,notACall=nac,dynamic=dyn,scored=scored,notScoredReason=nsr,basis=basis,strata=st,citations=cite))
open(A+'truth/go.final.jsonl','w').write(''.join(json.dumps(o,ensure_ascii=False)+'\n' for o in fin))
# noise
N=len(I); sil=[i for i in I if i['reference']['static']['kind']=='silent']
comp=[i for i in I if i['reference']['static']['kind']!='silent']
agr=[i for i in comp if i['status']=='agreed']
def rate(L):
    a=sum(1 for i in L if i['status']=='agreed'); return dict(comparable=len(L),agreed=a,rate=round(a/len(L),4) if L else None,wilson95=wilson(a,len(L)))
labin=lambda i:bool(i['inCorpusLabel'] or i['declaredLabel'])
per={'overall':rate(comp),'labelInCorpus(AI-side)':rate([i for i in comp if labin(i)]),'labelNonInternal(AE-side)':rate([i for i in comp if not labin(i)]),
 'declared':rate([i for i in comp if i['declaredLabel']]),'byConstruction':rate([i for i in comp if i['reference'].get('byConstruction')]),
 'dispatch':rate([i for i in comp if i['reference']['static']['kind']=='dispatch'])}
adj=[d for d in final_dec.values() if d['basis']=='adjudicated']
wrong=collections.Counter(d['wrong'] for d in adj); ek=collections.Counter(d['errorKind'] for d in adj)
W=wrong['label']+wrong['both']
strata={}
for s in('AI','AE','AS'):
    pop=[i for i in I if stratum(i)==s]; rows=[a for a in aud if a['row']=='site' and a['draw']=='U' and a['stratum']==s]
    n=len(rows); k=sum(1 for a in rows if not a['labelCorrect'])
    cr=[a for a in aud if a['row']=='site' and a['draw']=='C' and a['stratum']==s]
    strata[s]=dict(N_s=len(pop),n=n,k=k,wilson95=wilson(k,n),census=(n==len(pop)),drawC=dict(n=len(cr),errors=sum(1 for a in cr if not a['labelCorrect'])),
      exactCount=(k if n==len(pop) else None),flagged=(k/n>0.05 if n else False))
def contrib(s,upper=False):
    x=strata[s]
    if x['census']: return x['k']
    return x['N_s']*((wilson(x['k'],x['n'])[1]) if upper else x['k']/x['n'])
pt=(W+sum(contrib(s) for s in strata))/N; cons=(W+sum(contrib(s,True) for s in strata))/N
cells=collections.defaultdict(lambda:[0,0])
for a in aud:
    if a['row']=='site':
        c=cells[(a['stratum'],a['draw'],a['cell'])]; c[0]+=1; c[1]+=(not a['labelCorrect'])
celltab=[dict(stratum=k[0],draw=k[1],cell=k[2],n=v[0],errors=v[1]) for k,v in sorted(cells.items())]
refwrong=[d for d in adj if d['wrong'] in('reference','both')]
refcomp=len(comp)
bc=[i for i in I if i['reference'].get('byConstruction')]
bcs=collections.Counter(stratum(i) or 'adjudicated' for i in bc)
bc_adj_ref=sum(1 for i in bc if final_dec.get(i['siteId']) and final_dec[i['siteId']]['basis']=='adjudicated' and final_dec[i['siteId']]['wrong'] in('reference','both'))
unscored=collections.Counter(f['notScoredReason'] for f in fin if not f['scored'])
noise=dict(lang='go',N=N,comparable=len(comp),referenceSilent=dict(total=len(sil),adjudicated=sum(1 for i in sil if i['status']!='agreed'),AS=sum(1 for i in sil if i['status']=='agreed')),
 agreement=per,adjudicated=dict(n=len(adj),wrong=dict(wrong),errorKind=dict(ek),W_labelWrong=W),
 auditTable=strata,cellErrorTable=celltab,
 handErrorRate=dict(point=round(pt,4),conservativeWilsonUpper=round(cons,4),pointWithoutRoleErrors=round(pt,4),conservativeWithoutRoleErrors=round(cons,4),roleErrors=0,
   formula='(W + sum_s N_s*k_s/n_s, AS exact count)/N',noiseLimitedByPoint=pt>0.05,conservativeExceeds0_05=cons>0.05),
 referenceErrorRate=dict(adjudicatedReferenceWrong=len(refwrong),outOf=refcomp,rate=round(len(refwrong)/refcomp,4),wilson95=wilson(len(refwrong),refcomp),note='audits judged labels only; reference errors counted from adjudicated sites'),
 byConstruction=dict(labelled=len(bc),byStratum=dict(bcs),adjudicatedReferenceErrors=bc_adj_ref),
 role=dict(censusItems=len(cd),labelErrors=sum(1 for c in cd if not c['labelOK'])),
 scored=dict(scored=sum(f['scored'] for f in fin),unscored=dict(unscored),underivableShare=unscored['underivable']/N),
 siblingSweep=dict(trigger='AS ordinal 49, cell external:stdlib:time',sites=[s['ordinal'] for s in sweeps],overturns=sum(1 for s in sweeps if not s['labelCorrect'])))
json.dump(noise,open(A+'adjudication/go.noise.json','w'),indent=1)
print(json.dumps(noise,indent=1))
