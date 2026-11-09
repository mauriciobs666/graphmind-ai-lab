import json,re,collections as C

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
R=(_CPG+'/spikes/frontend-fitness/'); A=R+'adjudication/'
S=(_CPG+'/spikes/frontend-fitness/adjudication/tools/python/')
rj=lambda p:[json.loads(l) for l in open(p)]
fin=rj(R+'truth/python.final.jsonl'); inp=rj(A+'python.input.jsonl'); q=rj(A+'python.queue.jsonl'); dec=rj(A+'python.decisions.jsonl')
adj=rj(A+'python.jsonl'); aud=rj(A+'python.audit.jsonl'); ext=rj(S+'python.census-ext.jsonl'); noise=json.load(open(A+'python.noise.json'))
ok=True
def chk(name,c,info=''):
    global ok; ok&=bool(c); print(('PASS ' if c else 'FAIL ')+name,info)
chk('final N=240 unique ordinals+siteIds',len(fin)==240 and len({f['ordinal'] for f in fin})==240 and {f['siteId'] for f in fin}=={r['siteId'] for r in inp})
chk('final key set uniform',len({frozenset(f) for f in fin})==1)
chk('every final has path:line citation',all(re.search(r'[\w/.\-]+\.py:\d+',f['evidence']) for f in fin))
chk('evidence <=300',all(len(f['evidence'])<=300 for f in fin))
B={'agreed','audit-confirmed','adjudicated','audit-overturned','sweep-overturned','label-only'}
chk('basis valid',all(f['basis'] in B for f in fin),dict(C.Counter(f['basis'] for f in fin)))
und=sum(1 for f in fin if f['notScoredReason']=='underivable')
chk('underivable <=0.05N',und/240<=0.05,und)
chk('T-roles valid',all(t['role'] in('impl','decl') for f in fin for t in f['T']))
census={(x['file'],x['line']) for x in dec if x['queueCategory']=='census'}|{(e['path'],e['defLine']) for e in ext}
chk('census size 195',len(census)==195)
chk('every final T in census',all((t['path'],t['defLine']) in census for f in fin for t in f['T']),sum(len(f['T']) for f in fin))
# key sets
qk={(f"census:{x['file']}:{x['line']}" if x['queueCategory']=='census' else str(x['ordinal'])) for x in q}
chk('decisions keys == queue keys',qk=={x['queueKey'] for x in dec})
qadj={x['ordinal'] for x in q if x['queueCategory']=='adjudicate'}
chk('python.jsonl ordinals == queue adjudicate',qadj=={x['ordinal'] for x in adj} and len(adj)==73)
qa={(x['queueCategory'],x['ordinal']) for x in q if x['queueCategory'].startswith('audit')}
chk('audit rows == queue audit draws',qa=={('audit-'+a['draw'],a['ordinal']) for a in aud} and len(aud)==131)
chk('audit: draw C separate/not pooled',sum(a['draw']=='C' for a in aud)==7 and noise['audit']['AI']['n']==60)
chk('final basis consistent with category',all((f['basis']=='adjudicated')==(f['ordinal'] in qadj) for f in fin) and all((f['basis']=='audit-confirmed')==(('audit-U',f['ordinal']) in qa or ('audit-C',f['ordinal']) in qa) for f in fin))
chk('adjudication errorKind closed set',all(a['errorKind'] in('target','kind','role','coordinate','shape','none') for a in adj),dict(C.Counter(a['errorKind'] for a in adj)))
chk('decision citations',all(re.search(r'\.py:\d+',a['evidence']) for a in adj+aud))
# agreed unaudited T equals label
byo={r['ordinal']:r for r in inp}
bad=[f['ordinal'] for f in fin if f['basis']=='agreed' and [(t['path'],t['defLine']) for t in f['T']]!=[(c['path'],c['defLine']) for c in byo[f['ordinal']]['label']['callees'] if c['kind']!='external']]
chk('agreed T == label',not bad,bad)
# coordinate instances
ci=noise['closedClasses']['byKind']['coordinate']
cl={(x['file'],x['line']) for x in dec if x['queueCategory']=='census' and x['errorSource']=='label'}
slip={(c['path'],c['defLine']) for o in ci for c in byo[o]['label']['callees']}
chk('8 coordinate sites: label lines found by census read',slip<=cl,(sorted(slip),len(ci)))
fx={f['ordinal']:f for f in fin}
chk('coordinate corrected in final (T line differs from label line)',all(not({(t['path'],t['defLine']) for t in fx[o]['T']}&{(c['path'],c['defLine']) for c in byo[o]['label']['callees']}) for o in ci))
chk('targetsRead == distinct label+reference+decided',noise['census']['targetsRead']==195 and noise['census']['equal'])
chk('n_int',noise['detail']['v1.12']['n_int']['withClassLineOnlySites']==sum(1 for f in fin if f['T']),noise['detail']['v1.12']['n_int']['withClassLineOnlySites'])
print('ALL OK' if ok else 'FAILURES')
