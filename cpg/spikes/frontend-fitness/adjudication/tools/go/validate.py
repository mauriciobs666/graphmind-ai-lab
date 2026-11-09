import json,re,collections

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
A=(_CPG+'/spikes/frontend-fitness/')
rd=lambda p:[json.loads(l) for l in open(A+p) if l.strip()]
I=rd('adjudication/go.input.jsonl');Q=rd('adjudication/go.queue.jsonl');D=rd('adjudication/go.decisions.jsonl')
F=rd('truth/go.final.jsonl');G=rd('adjudication/go.jsonl');AU=rd('adjudication/go.audit.jsonl')
ok=True
def chk(c,m):
    global ok
    print(('PASS ' if c else 'FAIL ')+m); ok&=bool(c)
cit=re.compile(r'^[\w./\-]+:\d+$')
chk(len(F)==373 and len({f['siteId'] for f in F})==373,'final N=373 unique')
chk({f['siteId'] for f in F}=={i['siteId'] for i in I},'final siteIds == input')
chk([f['ordinal'] for f in F]==sorted(f['ordinal'] for f in F),'final ordinal order')
BASES={'agreed','audit-confirmed','adjudicated','audit-overturned','sweep-overturned','label-only'}
chk(all(f['basis'] in BASES for f in F),'valid basis '+str(dict(collections.Counter(f['basis'] for f in F))))
chk(all(f['citations'] and all(cit.match(c) for c in f['citations']) for f in F),'every final line has path:line citations')
chk(all(f['scored'] or f['notScoredReason'] for f in F),'unscored have reason')
und=sum(1 for f in F if f['notScoredReason']=='underivable'); chk(und/373<=0.05,f'underivable {und}/373 <= 0.05')
chk(all(f['T'] or f['N'] or f['external'] or f['notACall'] or f['dynamic'] for f in F),'every final has a target set or flag')
chk(all(set(t)=={'path','qname','defLine','role'} and t['role'] in('impl','decl') for f in F for t in f['T']),'T entries well-formed')
dq={q['siteId'] for q in Q if q['queueCategory']!='census'}
dsite=[d for d in D if d['kind']=='site']; dsq={d['siteId'] for d in dsite if d['basis']!='sweep-confirmed'}
chk(dsq==dq,f'decision site keys (non-sweep) == queue site keys ({len(dq)})')
sw={d['siteId'] for d in dsite if d['basis']=='sweep-confirmed'}
chk(sw<= {i['siteId'] for i in I},'sweep siteIds labelled'); print('  sweep not in queue:',sorted(d['ordinal'] for d in dsite if d['siteId'] in sw-dq))
cq={(q['file'],q['line']) for q in Q if q['queueCategory']=='census'}; cd={(d['path'],d['defLine']) for d in D if d['kind']=='census'}
chk(cq<=cd and len(cd)==138,f'census: queue keys {len(cq)} subset of decision keys {len(cd)}; extras {sorted(cd-cq)}')
used={(t['path'],t['defLine']) for f in F for t in f['T']}
chk(used<=cd,f'every final T target in census (missing: {sorted(used-cd)[:5]})')
chk(all((d['ordinal'],d['siteId']) in {(i['ordinal'],i['siteId']) for i in I} for d in dsite),'decision ordinal/siteId pairs match input')
chk(len(G)==39 and all(g['evidence'] and len(g['evidence'])<=300 and re.search(r'[\w./\-]+:\d+',g['evidence']) for g in G),f'go.jsonl {len(G)} lines, evidence<=300 with path:line')
chk(all(g['wrong'] in('label','reference','both','neither') and g['errorKind'] in('target','role','schema','kind','none') for g in G),'go.jsonl enums')
au=[a for a in AU if a['row']=='site']
chk(len(au)==sum(len([c for c in q['categories'] if c.startswith('audit-')]) for q in Q if q['queueCategory']!='census'),f'audit site rows {len(au)} == queue draws')
chk(all(a['pooledIntoRate']==(a['draw']=='U') for a in au),'draw C never pooled')
# adjudicated/overturned sites in final carry decision callees
gd={g['siteId']:g for g in G}
chk(all(f['basis'] not in('adjudicated','audit-overturned') or json.dumps(sorted(f['external']))==json.dumps(sorted(gd[f['siteId']]['decided']['external'])) for f in F),'final matches go.jsonl for adjudicated')
chk(next(f for f in F if f['ordinal']==27)['basis']=='audit-confirmed' and next(f for f in F if f['ordinal']==49)['basis']=='audit-overturned','ord 27 audit-confirmed, ord 49 audit-overturned')
print(dict(collections.Counter(f['basis'] for f in F)), 'ALL OK' if ok else 'FAILURES')
