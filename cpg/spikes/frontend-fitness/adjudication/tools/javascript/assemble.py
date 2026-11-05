import json,math,collections
FF='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/'
rd=lambda p:[json.loads(l) for l in open(FF+p)]
IN={r['siteId']:r for r in rd('adjudication/javascript.input.jsonl')}
REF={r['siteId']:r for r in rd('truth/javascript.ref.jsonl')}
LAB=rd('labels/javascript.jsonl')
D=rd('adjudication/javascript.decisions.jsonl')
DS={d['siteId']:d for d in D if d['item']=='site'}
DC={(d['path'],d['defLine']):d for d in D if d['item']=='census'}
def wil(k,n,z=1.959964):
    if n==0: return [None,None]
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); a=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [round((c-a)/d,4),round((c+a)/d,4)]
def role(p,l):
    c=DC.get((p,l)); return c['role'] if c and c['role'] else None
def w(path,rows):
    with open(FF+path,'w') as f:
        for r in rows: f.write(json.dumps(r)+'\n')
# decisions: adjudicated sites + census exceptions
dec=[]
for d in sorted(DS.values(),key=lambda d:d['ordinal']):
    if d['basis']=='adjudicated':
        dec.append({'ordinal':d['ordinal'],'siteId':d['siteId'],'reasons':d['reasons'],'decided':d['decided'],'wrong':d['wrong'],'errorKind':d['errorKind'],'evidence':d['evidence']})
for d in D:
    if d['item']=='census' and d['basis']!='census-confirmed':
        dec.append({'item':'census','path':d['path'],'defLine':d['defLine'],'decided':{'form':d['form'],'role':d['role'],'symbol':d['symbol'],'decision':d['decision']},'wrong':d['wrong'],'errorKind':d['errorKind'],'evidence':d['evidence']})
w('adjudication/javascript.jsonl',dec)
# audit
aud=[]
for d in sorted(DS.values(),key=lambda d:d['ordinal']):
    if d['basis']=='audit-confirmed':
        draw=[c for c in d['categories'] if c.startswith('audit-')][0]
        aud.append({'ordinal':d['ordinal'],'siteId':d['siteId'],'draw':draw.split(':')[0].split('-')[1],'stratum':draw.split(':')[1],'refLocal':'js-ref-local-agreed' in d['categories'],'labelCorrect':d['labelCorrect'],'wrong':d['wrong'],'errorKind':d['errorKind'],'evidence':d['evidence']})
for d in D:
    if d['item']=='census':
        aud.append({'item':'census','path':d['path'],'defLine':d['defLine'],'form':d['form'],'body':d['body'],'role':d['role'],'labelError':d['labelError'],'errorKind':d['errorKind'],'wrong':d['wrong'],'evidence':d['evidence']})
w('adjudication/javascript.audit.jsonl',aud)
# final
fin=[]
cnt=collections.Counter()
for L in sorted(LAB,key=lambda x:x['ordinal']):
    sid=L['siteId']; r=IN[sid]; d=DS.get(sid); rk=r['reference']['static']['kind']
    if d: dec_=d['decided']; basis=d['basis']
    else: dec_=None; basis='agreed'
    if dec_:
        T=[{'path':c['path'],'qname':c['qname'],'defLine':c['defLine'],'role':c['role']} for c in dec_['callees']]
        ext=dec_.get('external'); nac=dec_['notACall']; dyn=dec_['dynamic']
        ev=d['evidence']
    else:
        T=[{'path':c['path'],'qname':c['qname'],'defLine':c['defLine'],'role':role(c['path'],c['defLine'])} for c in r['label']['callees'] if c['kind'] in('internal','declared')]
        ext=sorted({c.get('package') for c in r['label']['callees'] if c['kind']=='external'}) or None
        nac=r['label']['notACall']; dyn=r['label']['dynamic']
        ev=f"{r['site']['path']}:{r['site']['line']} agreed: {r['label'].get('evidence','')}"[:300]
    if d and d['basis']=='audit-confirmed' and rk=='silent': pass
    strata=[]
    st=r.get('stratum') if False else None
    if d:
        strata=[c for c in d['categories']]
    else:
        strata=['AI' if T else 'AE']
        if rk=='local': strata.append('ref-local')
    if rk=='silent' and 'AS' not in ' '.join(strata): strata.append('AS')
    fin.append({'ordinal':L['ordinal'],'siteId':sid,'site':f"{r['site']['path']}:{r['site']['line']}",'T':T,'N':[],'external':ext,'notACall':nac,'dynamic':dyn,'scored':True,'notScoredReason':None,'basis':basis,'strata':strata,'evidence':ev})
    cnt[basis]+=1
w('truth/javascript.final.jsonl',fin)
print(cnt)
# noise
N=len(LAB); comparable=[s for s in IN if REF[s]['static']['kind']!='silent']
agreed_c=[s for s in comparable if IN[s]['status']=='agreed']
nc=len(comparable); ka=len(agreed_c)
def strat_of(s):
    d=DS.get(s)
    return None
pop={'AI':161,'AE':318,'AS':1}
audit={}
for st in pop:
    U=[a for a in aud if a.get('draw')=='U' and a['stratum']==st]; C=[a for a in aud if a.get('draw')=='C' and a['stratum']==st]
    k=sum(a['wrong'] in('label','both') for a in U); n=len(U)
    census_all=(n==pop[st])
    audit[st]={'n':n,'errors':k,'wilson95':wil(k,n),'population':pop[st],'censusOfStratum':census_all,
      'drawC':{'n':len(C),'errors':sum(a['wrong'] in('label','both') for a in C),'cells':len(C)}}
audit['AE']['refLocalCell']={'n':sum(a['refLocal'] for a in aud if a.get('draw')=='U' and a['stratum']=='AE'),'population':121,'errors':0}
W=sum(1 for d in DS.values() if d['basis']=='adjudicated' and d['wrong'] in('label','both'))
def eh(upper):
    s=W
    for st,a in audit.items():
        if a['censusOfStratum']: s+=pop[st]*a['errors']/a['n']; continue
        s+=pop[st]*(a['wilson95'][1] if upper else a['errors']/a['n'])
    return s/N
refw=sum(1 for d in DS.values() if d['basis']=='adjudicated' and d['wrong'] in('reference','both'))
out={'lang':'javascript','N':N,'comparable':nc,'comparableShare':round(nc/N,4),'agreedComparable':ka,'agreementRate':round(ka/nc,4),'agreementWilson95':wil(ka,nc),
 'agreedAllIncludingAS':sum(1 for r in IN.values() if r['status']=='agreed'),
 'perStratumAgreement':{'AI':{'comparable':161,'agreed':161},'AE':{'comparable':318,'agreed':318},'note':'AS (1) is reference-silent, single-source, excluded; relaxed/byConstruction/declared/macro strata do not occur in javascript (not computed).'},
 'adjudicated':{'total':len(dec)-sum(1 for x in dec if x.get('item')=='census'),'byWrong':dict(collections.Counter(d['wrong'] for d in DS.values() if d['basis']=='adjudicated')),'byErrorKind':dict(collections.Counter(d['errorKind'] for d in DS.values() if d['basis']=='adjudicated')),'W_labelWrong':W},
 'auditTable':audit,
 'handErrorRate':{'withRoleErrors':round(eh(False),4),'withoutRoleErrors':round(eh(False),4),'conservativeWilsonBoundWithRole':round(eh(True),4),'conservativeWilsonBoundWithoutRole':round(eh(True),4),'roleErrors':0,'noiseLimited':eh(False)>0.05,'note':'AS stratum audited as a census (n=N_s=1) so its term is exact; draw C excluded from the rate.'},
 'referenceErrorRate':{'wrongSites':refw,'comparable':nc,'rate':round(refw/nc,4),'wilson95':wil(refw,nc),'breakdown':{'param-t attributed internal':20,'local':23},'censusReferenceErrors':1,'censusNote':'composerNotice.ts:28 is a census-level reference error (parameter, not definition); not added to the site count'},
 'census':{'entries':len(DC),'roleImpl':sum(1 for d in DC.values() if d['role']=='impl'),'notADefinition':sum(1 for d in DC.values() if d['decision']=='not-a-definition'),'roleErrors':0},
 'unscored':{'underivable':0,'notACall':sum(f['notACall'] for f in fin),'share':0.0},
 'testRetest':'56 sites, 55 agree (ledger; not recomputed)'}
json.dump(out,open(FF+'adjudication/javascript.noise.json','w'),indent=1)
print(json.dumps(out,indent=1)[:3500])
