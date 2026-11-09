"""S57 python, v1.12 canonicalisation. Run from cpg/spikes/frontend-fitness:  python3 adjudication/tools/python/build_v112.py
Inputs: adjudication/python.{input,queue,decisions}.jsonl, truth/python.final.jsonl (canonical, labels unchanged),
adjudication/tools/python/python.census-ext.jsonl (43 DS-decided callees), tools/python/python.noise.v1.11-original.json.
Outputs: adjudication/python.census.jsonl (195 rows), adjudication/python.noise.json (canonical envelope + detail)."""
import json,math,re,collections as C
from pathlib import Path
ROOT=Path('.'); A=ROOT/'adjudication'; T=A/'tools'/'python'

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
SRC=Path((_CPG+'/.ccg-work/corpora/python/src'))
rj=lambda p:[json.loads(l) for l in open(p)]
fin=rj(ROOT/'truth/python.final.jsonl'); inp=rj(A/'python.input.jsonl'); q=rj(A/'python.queue.jsonl'); dec=rj(A/'python.decisions.jsonl'); ext=rj(T/'python.census-ext.jsonl')
orig=json.load(open(T/'python.noise.v1.11-original.json'))
def wil(k,n,z=1.959964):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0,c-h),4),round(min(1,c+h),4)]
N=len(fin)
# ---- basis precedence (v1.12 step 6), from the FC's own records
adj={x['ordinal'] for x in dec if x['queueCategory']=='adjudicate'}
audd={x['ordinal'] for x in dec if x['queueCategory'].startswith('audit')}
def stratum(r):
    if r['inCorpusLabel']: return 'AI'
    return 'AS' if r['reference']['static']['kind']=='silent' else 'AE'
agreed_ord={r['ordinal']:stratum(r) for r in inp if r['status']=='agreed'}
dev=[]
for f in fin:
    o=f['ordinal']
    want='adjudicated' if o in adj else 'audit-confirmed' if o in audd else ('label-only' if agreed_ord[o]=='AS' else 'agreed')
    if f['basis']!=want: dev.append((o,f['basis'],want))
    if o in agreed_ord and agreed_ord[o] not in f['strata']: dev.append((o,'missing stratum tag',agreed_ord[o]))
assert not dev,dev
# ---- census file: 152 FC/DS rows + 43 DS-decided extension = targetsRead
fc={(x['file'],x['line']):x for x in q if x['queueCategory']=='census'}
rows=[]
for x in dec:
    if x['queueCategory']!='census': continue
    f=fc[(x['file'],x['line'])]
    rows.append({'path':x['file'],'defLine':x['line'],'qnames':f['qnames'],'source':'label-reference','fcForm':f['fcForm'],'fcDecorators':f['decorators'],
      'form':x['form'],'body':x['body'],'role':x['role'],'targetKind':x['targetKind'],'labelRoleError':x['labelRoleError'],'errorSource':x['errorSource'],
      'decision':x['decision'],'evidence':x['evidence']})
for e in ext:
    L=(SRC/e['path']).read_text().split('\n')
    rows.append({'path':e['path'],'defLine':e['defLine'],'qnames':[e['qname']],'source':'ds-decided-callee','fcForm':e['form'],'fcDecorators':[],
      'fcLine':L[e['defLine']-1].strip()[:120],'form':e['form'],'body':e['body'],'role':e['role'],'targetKind':'anonymous-lambda' if e['form']=='lambda' else 'callable',
      'labelRoleError':e['labelRoleError'],'errorSource':'none','decidedAtOrdinal':e['decidedAt'],
      'decision':'FC read the line from source: expression/def with a body, undecorated; role impl','evidence':f"{e['path']}:{e['defLine']}"})
keys=[(r['path'],r['defLine']) for r in rows]; assert len(set(keys))==len(keys)==195
with open(A/'python.census.jsonl','w') as fh:
    for r in rows: fh.write(json.dumps(r,ensure_ascii=False)+'\n')
cens={(r['path'],r['defLine']):r for r in rows}
# completeness: distinct (path,defLine) of labels, reference and final T
lab=set()
for r in inp:
    for c in r['label']['callees']:
        if c['kind']!='external': lab.add((c['path'],c['defLine']))
    for c in [r['reference']['static']]+r['reference']['impls']:
        if c.get('path'): lab.add((c['path'],c['defLine']))
finT={(t['path'],t['defLine']) for f in fin for t in f['T']}
assert set(cens)==lab|finT, (len(lab|finT),len(cens))
# ---- v1.12 figures
Ns={'AI':92,'AE':71,'AS':4}; ns={'AI':60,'AE':60,'AS':4}
audit_rows=rj(A/'python.audit.jsonl')
rk={}
for s in Ns:
    U=[a for a in audit_rows if a['stratum']==s and a['draw']=='U']; assert len(U)==ns[s]
    rk[s]=sum(1 for a in U if not a['labelCorrect'] and a['errorKind'] in('target','kind'))
k_all={s:sum(1 for a in audit_rows if a['stratum']==s and a['draw']=='U' and not a['labelCorrect']) for s in Ns}
adjd=rj(A/'python.jsonl')
W=sum(1 for a in adjd if a['wrong'] in('label','both')); Wlab=sum(1 for a in adjd if a['wrong']=='label')
unread={s:Ns[s]-ns[s] for s in Ns}
e_res=sum(unread[s]*rk[s]/ns[s] for s in Ns)/N
e_res_c=sum(unread[s]*wil(rk[s],ns[s])[1] for s in Ns)/N
e_all=(W+sum(Ns[s]*k_all[s]/ns[s] for s in Ns))/N
e_all_c=(W+sum(Ns[s]*wil(k_all[s],ns[s])[1] for s in Ns))/N
closed=sorted(a['ordinal'] for a in adjd if a['errorKind'] in('role','coordinate','shape'))
# v1.10 (flag used the figure WITH role errors: W + extrapolated audit)
v110=e_all
# ---- openPointsS27
tl=[f for f in fin if f['T']]
isclass=lambda t: cens[(t['path'],t['defLine'])]['form']=='class'
classsites=[f for f in tl if any(isclass(t) for t in f['T'])]
classonly=[f for f in tl if all(isclass(t) for t in f['T'])]
anon={}
for f in fin:
    for t in f['T']:
        if '<anon>' in (t['qname'] or ''): anon.setdefault(t['qname'],[]).append(f['ordinal'])
extpk=sorted({e for f in fin for e in f['external']})
top={p.name for p in SRC.iterdir()}
classdefs=sorted({(t['path'],t['defLine'],t['qname']) for f in classsites for t in f['T'] if isclass(t)})
openpts={
 'classLineTargets':{'n_sitesWithAnyClassLineTarget':len(classsites),'n_sitesClassLineOnly':len(classonly),
   'sitesAnyOrdinals':[f['ordinal'] for f in classsites],'sitesClassOnlyOrdinals':[f['ordinal'] for f in classonly],
   'definitions':[{'path':p,'defLine':d,'qname':n} for p,d,n in classdefs],
   'note':'T is a class line standing for an implicit constructor (class has no __init__, or reference-only mapping); whether such sites count toward n_int/recall is open for S27'},
 'anonymousLambdaIndices':[{'qname':k,'ordinals':v,'path_lines':sorted({f"{t['path']}:{t['defLine']}" for f in fin if f['ordinal'] in v for t in f['T'] if t['qname']==k})} for k,v in sorted(anon.items())],
 'externalsAbsentFromTree':{'found':[p for p in extpk if p.split(':')[0] in top or p.split(':')[0] in ('falkorchat','tests')],'externalPackagesLabelled':extpk,
   'note':'none found: no labelled external names a package whose source is in the corpus tree (top-level: %s)'%sorted(top)}}
nint=len(tl)
env={'lang':'python','N':N,'comparable':209,'referenceSilent':31,
 'agreement':{'agreed':163,'comparable':209,'rate':round(163/209,4),'wilson95':wil(163,209)},
 'adjudicated':{'n':len(adjd),'W':W,'byWrong':dict(C.Counter(a['wrong'] for a in adjd)),'byErrorKind':dict(C.Counter(a['errorKind'] for a in adjd))},
 'referenceError':{'wrong':27,'comparable':209,'rate':round(27/209,4),'wilson95':wil(27,209)},
 'audit':{s:{'N_s':Ns[s],'n':ns[s],'k':k_all[s],'wilson95':wil(k_all[s],ns[s]),'r_residual':rk[s],'unreadRemainder':unread[s],'wilson95_residual':wil(rk[s],ns[s])} for s in Ns},
 'handError':{'e_all':{'point':round(e_all,4),'conservative':round(e_all_c,4)},'e_res':{'point':round(e_res,4),'conservative':round(e_res_c,4)},'noiseLimited':e_res>0.05},
 'census':{'targetsRead':len(cens),'file':'adjudication/python.census.jsonl','rows':len(rows),'distinctLabelReferenceFinalT':len(lab|finT),'equal':set(cens)==lab|finT},
 'closedClasses':{'count':len(closed),'byKind':{'role':[],'coordinate':closed,'shape':[]}},
 'unscored':{'byReason':{},'underivable':0},
 'basisCounts':dict(C.Counter(f['basis'] for f in fin)),
}
detail=dict(orig)
detail['v1.12']={'e_all_W_only_labelWrong':round(Wlab/N,4),'e_all_W_labelWrongOrBoth':round(W/N,4),'e_res_formula':'sum_s (N_s-n_s)*r_s/n_s / N, r_s = audited target/kind errors',
  'closedClassShare':round(len(closed)/N,4),'closedClassTripwire':0.10,'v1.10WouldHaveSaid':{'figure_with_role_errors':round(v110,4),'noiseLimited':v110>0.05},
  'v1.11Said':{'e_res':orig['handError']['e_res'],'noiseLimited':orig['handError']['noiseLimited']},
  'n_int':{'withClassLineOnlySites':nint,'excludingClassLineOnlySites':nint-len(classonly)},
  'basisRuling':'precedence adjudicated > audit-confirmed > label-only > agreed verified against FC records; 0 deviations; the 4 AS sites (all re-read in draw U) are audit-confirmed; 0 label-only',
  'siblingSweep':'none required: no target/kind error in any audit draw (draws U and C: 0 errors)',
  'dsReasoningNote':'adjudicated sites decided from source by DS; DS error not estimated by this procedure'}
detail['openPointsS27']=openpts
env['detail']=detail
json.dump(env,open(A/'python.noise.json','w'),indent=1,ensure_ascii=False)
print(json.dumps({'basis':env['basisCounts'],'e_all':env['handError']['e_all'],'e_res':env['handError']['e_res'],'flag':env['handError']['noiseLimited'],'v110':v110,'n_int':[nint,nint-len(classonly)],'class_any':len(classsites),'class_only':len(classonly),'census':env['census'],'anon':anon,'ext_absent':openpts['externalsAbsentFromTree']['found']},indent=1))
