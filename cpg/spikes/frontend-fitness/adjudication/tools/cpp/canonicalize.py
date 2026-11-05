"""cpp: v1.12 canonical truth/cpp.final.jsonl and adjudication/cpp.noise.json (run: python3 canonicalize.py).
Inputs (all cpp): cpp.final.v111.jsonl + cpp.noise.v111.json (this dir, the pre-v1.12 outputs of assemble.py),
../cpp.input.jsonl, ../cpp.jsonl, ../cpp.audit.jsonl, ../cpp.census.jsonl, ../cpp.decisions.jsonl."""
import json,math,re,collections,pathlib
H=pathlib.Path(__file__).resolve().parent; AD=H.parent.parent; R=AD.parent
J=lambda p:[json.loads(l) for l in open(p)]
def W(rows,p): open(p,'w').write("".join(json.dumps(r)+"\n" for r in rows))
# 1. errorKind schema -> shape in the decision files
chg=collections.Counter()
for f in ('cpp.decisions.jsonl','cpp.jsonl'):
    rows=J(AD/f)
    for r in rows:
        if r.get('errorKind')=='schema': r['errorKind']='shape'; chg[f]+=1
    W(rows,AD/f)
inp=J(AD/'cpp.input.jsonl'); old=J(H/'cpp.final.v111.jsonl'); adj={r['ordinal']:r for r in J(AD/'cpp.jsonl')}
aud={r['ordinal']:r for r in J(AD/'cpp.audit.jsonl')}; orig=json.load(open(H/'cpp.noise.v111.json'))
# 2. canonical final: external dicts -> str, N dicts -> str, basis by v1.12 precedence, plain strata tags
fin=[]; BAS=collections.Counter(); chgbasis=collections.Counter()
def ext_s(e): return e if isinstance(e,str) else (e.get('package') or e.get('header') or json.dumps(e,sort_keys=True))
for x in old:
    o=x['ordinal']; y=dict(x)
    y['external']=[ext_s(e) for e in x['external']]
    y['N']=[n if isinstance(n,str) else f"{n['path']}:{n['defLine']} {n.get('form','N')}" for n in x['N']]
    if o in adj: b='audit-overturned' if (o in aud and not aud[o]['labelCorrect'] and False) else 'adjudicated'
    elif o in aud: b='audit-confirmed' if aud[o]['labelCorrect'] else 'audit-overturned'
    elif 'AS' in x['strata']: b='label-only'
    else: b='agreed'
    if b!=x['basis']: chgbasis[(x['basis'],b)]+=1
    y['basis']=b; y['strata']=list(dict.fromkeys(x['strata'])); BAS[b]+=1
    # label-set equality guard
    for k in ('siteId','site','T','notACall','dynamic','scored','notScoredReason'): assert y[k]==x[k]
    fin.append(y)
assert len(fin)==187
W(fin,R/'truth'/'cpp.final.jsonl')
# 3. noise v1.12
def wil(k,n,z=1.96):
    if n==0: return None
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0,c-h),4),round(min(1,c+h),4)]
N=187; au=orig['audit']
Wl=[o for o,r in adj.items() if r['wrong'] in('label','both')]
Wonly=[o for o,r in adj.items() if r['wrong']=='label']
shape=sorted(o for o in Wl if adj[o]['errorKind']=='shape'); role=orig['closedClasses']['role']['ordinals']
strata={}
for s in('AI','AE','AS'):
    a=au[s]; U=[r for r in aud.values() if r['stratum']==s and r['draw']=='U']
    k_all=sum(not r['labelCorrect'] for r in U); r_s=sum((not r['labelCorrect']) and r['errorKind'] in('target','kind') for r in U)
    strata[s]=dict(N_s=a['N_s'],n=len(U),k=k_all,r=r_s,remainder=a['N_s']-len(U),census=len(U)==a['N_s'],
                   wilson95=None if len(U)==a['N_s'] else wil(k_all,len(U)))
ep=sum(v['remainder']*v['r']/v['n'] for v in strata.values()); ec=sum(v['remainder']*(v['wilson95'][1] if v['wilson95'] else 0) for v in strata.values())
ea_p=sum(v['N_s']*v['k']/v['n'] for v in strata.values() if not v['census']); ea_c=sum(v['N_s']*v['wilson95'][1] for v in strata.values() if v['wilson95'])
role_extra=[o for o in role if o not in Wl]
e_all={'point':round((len(Wl)+len(role_extra)+ea_p)/N,4),'conservative':round((len(Wl)+len(role_extra)+ea_c)/N,4)}
e_res={'point':round(ep/N,4),'conservative':round(ec/N,4)}
byk=collections.Counter(r['errorKind'] for r in adj.values() if r['errorKind'])  # macro-record-only site has None
ad=orig['adjudicated']; ad_bw=dict(collections.Counter(r['wrong'] for r in adj.values() if r['wrong']))
# adjudicated = the 89 queue sites (site 152 is decided via its macro record only, wrong None)
a89=[r for r in adj.values() if r['wrong']]
ad_bk=dict(collections.Counter(r['errorKind'] for r in a89)); ad_bw=dict(collections.Counter(r['wrong'] for r in a89))
re_=orig['referenceError']
rel=orig['relaxed']; tcm=orig['typeCtorMerged']
cc={'count':len(set(shape)|set(role)),'byKind':{'role':sorted(role),'coordinate':[],'shape':shape}}
# open points
ext=collections.defaultdict(list)
for x in fin:
    for e in x['external']:
        if e.startswith('header:'): ext[e].append(x['ordinal'])
detail=dict(
 v111Original=orig,
 v112=dict(
  e_all_note="labeller noise, disclosure only; W=label|both adjudicated, plus role instance not already in W (0), plus AI audit extrapolation (0 errors)",
  W=len(Wl),W_labelOnlyVariant=len(Wonly),e_all_labelOnlyVariant=round((len(Wonly)+ea_p)/N,4),
  e_res_formula="sum_s (N_s-n_s)*r_s/n_s / N, r_s = audited target|kind errors; AE and AS are censuses and contribute 0",
  perStratum=strata,drawC=orig['audit']['drawC'],
  noiseLimited_v112=e_res['point']>0.05,
  verdictByVersion={'v1.10':{'figure':'e_h with role errors (W + Sum N_s k_s/n_s)/N','value':e_all['point'],'noiseLimited':e_all['point']>0.05},
                    'v1.11':{'figure':'e_res = residual W (46, shape excluded) /N','value':orig['handErrorRate']['e_res']['point'],'noiseLimited':True},
                    'v1.12':{'figure':'e_res (unread remainder x audited rate)','value':e_res['point'],'noiseLimited':e_res['point']>0.05}},
  readout="labeller noisy, final labels repaired (e_all>0.05, e_res<=0.05); corrected sites rest on DS's source reading (analyst Pass 3: 0 wrong in 14 for C++)",
  macroStratum=orig['macroStratum'],
  relaxed=dict(n=rel['n'],referenceErrors=rel['referenceErrors'],share=rel['share'],wilson95=wil(rel['referenceErrors'],rel['n']),cap=rel['cap']),
  typeCtorMerged=dict(sites=tcm['sites'],referenceErrors=tcm['referenceErrors'],wilson95=wil(tcm['referenceErrors'],tcm['sites']),withdrawAt=tcm['withdrawAt'],ordinals=[66,73,80,136,162,176]),
  closedClassNotes=dict(shape="errorKind 'schema' in the S57 decisions renamed 'shape' (v1.12): empty callees, neither notACall nor dynamic",
                        role="gmock-function-mocker_test.cc:126 (site 69); the site is also in W as a target error and stays a residual candidate only via the audit, which found none",
                        share_of_N=round(cc['count']/N,4),tripwire=0.10),
  census_note=orig['census'],
  openPointsS27=dict(
   classLineTargets="none found: scan of the 148 distinct final T (path, defLine) lines for a class/struct head line (regex ^(template<..>)?(class|struct)\\b) returned 0; the 3 census rows decided 'type' (gmock-actions.h:508, gmock-matchers.h:2040, gtest-param-util.h:230) are wrong targets and are in no final T",
   anonymousLambdaIndices="none: no final T qname carries a lambda index; one qname contains an anonymous-namespace marker (<anonns>), googlemock/test/gmock-matchers-misc_test.cc BacktrackingMaxBPMState.Compute",
   externalsPackageAbsentFromTree=dict(note="external header names whose source is absent from the corpus tree (no 'absl' path under cpg/.ccg-work/corpora/cpp/src)",byName={k:v for k,v in sorted(ext.items())}),
   macroGeneratedTargets=sum(1 for x in fin for t in x['T'] if t.get('macroGenerated')),
   )))
unsc=collections.Counter(x['notScoredReason'] for x in fin if not x['scored'])
noise=dict(lang='cpp',N=N,comparable=orig['comparable']['comparable'],referenceSilent=orig['comparable']['silent'],
 agreement=dict(agreed=orig['agreement']['agreed_comparable'],comparable=orig['comparable']['comparable'],rate=orig['agreement']['rate'],wilson95=orig['agreement']['wilson95']),
 adjudicated=dict(n=len(a89),W=len(Wl),byWrong=ad_bw,byErrorKind=ad_bk),
 referenceError=dict(wrong=re_['adjudicatedRefErrors_comparable'],comparable=re_['comparable'],rate=re_['point'],wilson95=re_['wilson95']),
 audit={s:dict(N_s=v['N_s'],n=v['n'],k=v['k'],wilson95=v['wilson95']) for s,v in strata.items()},
 handError=dict(e_all=e_all,e_res=e_res,noiseLimited=e_res['point']>0.05),
 census=dict(targetsRead=orig['census']['targetsRead']),
 closedClasses=cc,unscored=dict(byReason=dict(unsc),underivable=unsc.get('underivable',0)),
 basisCounts=dict(BAS),detail=detail)
json.dump(noise,open(AD/'cpp.noise.json','w'),indent=1)
print('schema->shape',dict(chg)); print('basis changes',dict(chgbasis)); print('basisCounts',dict(BAS))
print('e_all',e_all,'e_res',e_res,'flag',noise['handError']['noiseLimited']); print('ad',ad_bw,ad_bk); print('cc',cc)
