"""v1.12 canonicalisation for javascript: final.jsonl (from the normalised copy) and noise.json."""
import json,math,re,collections,sys
FF='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/'
NORM=sys.argv[1]  # dir holding the normalised javascript.final.jsonl / noise.json
rd=lambda p:[json.loads(l) for l in open(p)]
old=rd(FF+'truth/javascript.final.jsonl'); new=rd(NORM+'javascript.final.jsonl')
def key(f):
    ext=f['external']; ext=[ext] if isinstance(ext,str) else (ext or [])
    return (f['ordinal'],f['siteId'],json.dumps(f['T'],sort_keys=True),tuple(sorted(ext)),f['notACall'],f['dynamic'],f['scored'])
assert [key(f) for f in old]==[key(f) for f in new],'label content differs'
IN={r['siteId']:r for r in rd(FF+'adjudication/javascript.input.jsonl')}
assert [f['siteId'] for f in new]==[r['siteId'] for r in IN.values()]
Q={q['siteId']:q for q in rd(FF+'adjudication/javascript.queue.jsonl') if q['item']=='site'}
# basis precedence
AUD={d['siteId'] for d in rd(FF+'adjudication/javascript.audit.jsonl') if d.get('siteId')}
ADJ={d['siteId'] for d in rd(FF+'adjudication/javascript.jsonl') if d.get('siteId')}
before=collections.Counter(f['basis'] for f in new); chg=[]
for f in new:
    s=f['siteId']; strat=set(f['strata'])
    want='adjudicated' if s in ADJ else 'audit-confirmed' if s in AUD else ('label-only' if 'AS' in strat else 'agreed')
    if f['basis']!=want: chg.append((f['ordinal'],f['basis'],want)); f['basis']=want
assert all(isinstance(x,str) for f in new for x in f['strata'])
n199=[f for f in new if f['ordinal']==199][0]
if 'AS' not in n199['strata']: n199['strata'].append('AS')
with open(FF+'truth/javascript.final.jsonl','w') as o:
    for f in new: o.write(json.dumps(f)+'\n')
print('basis before',dict(before),'after',dict(collections.Counter(f['basis'] for f in new)),'changed',chg)
# noise
n=json.load(open(NORM+'javascript.noise.json'))
def up(k,nn,z=1.959964):
    p=k/nn; d=1+z*z/nn; c=p+z*z/(2*nn); a=z*math.sqrt(p*(1-p)/nn+z*z/(4*nn*nn)); return (c+a)/d
Ns={'AI':161,'AE':318,'AS':1}; ns={'AI':100,'AE':100,'AS':1}; r={'AI':0,'AE':0,'AS':0}
N=523
rem={s:Ns[s]-ns[s] for s in Ns}
e_res_pt=sum(rem[s]*r[s]/ns[s] for s in Ns)/N
e_res_cons=sum(rem[s]*(0 if rem[s]==0 else up(r[s],ns[s])) for s in Ns)/N
n['handError']['e_res']={'point':round(e_res_pt,4),'conservative':round(e_res_cons,4)}
n['handError']['noiseLimited']=e_res_pt>0.05
anon=[]
for f in new:
    for t in f['T']:
        if t['qname'] and re.search(r'<anon>#\d+',t['qname']): anon.append({'ordinal':f['ordinal'],'site':f['site'],'qname':t['qname'],'defLine':t['defLine']})
pk=collections.Counter(p for f in new for p in f['external'])
d=n['detail']
d['v1.12']={'e_all':n['handError']['e_all'],'e_res':n['handError']['e_res'],
 'perStratum':{s:{'N_s':Ns[s],'n':ns[s],'unreadRemainder':rem[s],'r_s_targetKindErrors':r[s],'wilsonUpper':(None if rem[s]==0 else round(up(0,ns[s]),4))} for s in Ns},
 'flag':'e_res point 0.0 <= 0.05: not noise-limited','closedClassShare':0.0,'closedClassTripwire':0.10,
 'v1.10_would_have_said':'e_h = 0.0 (point), conservative 0.0339, not noise-limited; same verdict. v1.12 e_res conservative differs (unread remainder only).'}
d['openPointsS27']={
 'classLineTargets':'none found: the 103 census entries read by DS are 86 function declarations, 9 arrow functions, 6 expression-bodied arrows, 1 explicit constructor (src/api/client.ts:35, ApiError.<init>, not a class line; class at :25) and 1 parameter (composerNotice.ts:28, not a definition); no target sits on a class line.',
 'anonymousIndexQnames':{'count':len(anon),'note':'JS <anon>#n indices appear in these final T qnames (S57 compares (path, defLine); S27 needs the list)','entries':anon},
 'externalsAbsentFromTree':{'note':'every external label names a third-party package or builtin whose source is not in the corpus tree (JS has no vendored node_modules in src); no generated-source gap analogous to Java OpenAPI DTOs was found.','distinctPackages':len(pk),'packages':dict(sorted(pk.items()))}}
json.dump(n,open(FF+'adjudication/javascript.noise.json','w'),indent=1)
print(n['handError'],len(anon),len(pk))
# NOTE: run fix_external.py afterwards (Pass 4 N2), then refresh detail.openPointsS27.externalsAbsentFromTree from the final file.
