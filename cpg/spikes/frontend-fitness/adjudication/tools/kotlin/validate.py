import json,re,collections
R='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/'; A=R+'adjudication/'
jl=lambda p:[json.loads(l) for l in open(p)]
fin=jl(R+'truth/kotlin.final.jsonl'); dec=jl(A+'kotlin.jsonl'); aud=jl(A+'kotlin.audit.jsonl'); cen=jl(A+'kotlin.census.jsonl')
q=jl(A+'kotlin.queue.jsonl'); dd=jl(A+'kotlin.decisions.jsonl'); noise=json.load(open(A+'kotlin.noise.json')); inp=jl(A+'kotlin.input.jsonl')
err=[]
def chk(c,m):
    if not c: err.append(m)
chk(len(fin)==430,'final N'); chk([f['ordinal'] for f in fin]==list(range(1,431)),'ordinals')
BAS={'agreed','audit-confirmed','adjudicated','audit-overturned','sweep-overturned','label-only'}
cite=re.compile(r'[\w./-]+\.kt:\d+')
ckeys=set(map(tuple,[sorted(f.keys()) for f in fin])); chk(len(ckeys)==1,'final key sets differ')
sid={r['ordinal']:r['siteId'] for r in inp}
for f in fin:
    chk(f['basis'] in BAS,f"basis {f['ordinal']}"); chk(f['siteId']==sid[f['ordinal']],f"siteId {f['ordinal']}")
    chk(cite.search(f['site']) is not None and cite.search(f['evidence'] or '') is not None or (cite.search(f['site']) and f['basis']=='agreed'),f"cite {f['ordinal']}")
    chk(len(f['evidence'])<=300,f"evlen {f['ordinal']}")
    for t in f['T']: chk(t['role'] in('impl','decl'),f"role {f['ordinal']}")
    chk(bool(f['T'] or f['N'] or f['external'] or f['notACall'] or f['dynamic']),f"empty {f['ordinal']}")
# evidence citations: how many final lines have path:line in evidence itself
print('final evidence with path:line:',sum(1 for f in fin if cite.search(f['evidence'])),'of',len(fin),'; site path:line on all:',all(cite.search(f['site']) for f in fin))
# decisions & audits cites
for d in dec+aud+[x for x in cen]: chk(cite.search(d['evidence']) and len(d['evidence'])<=300 or d.get('category')=='census' and cite.search(d['evidence']),f"dec cite {d.get('ordinal',d.get('path'))}")
print('decision evidence >300:',[d.get('ordinal') for d in dec+aud if len(d['evidence'])>300])
# key sets: decisions vs queue vs final
qk={(x['category'],x['ordinal']) for x in q if 'ordinal' in x}; dk={(x['category'],x['ordinal']) for x in dd if 'ordinal' in x}
chk(qk==dk,'queue vs decisions item keys'); print('queue',len(q),'decisions',len(dd)); 
qc={(x['path'],x['defLine']) for x in q if x['category']=='census'}; dc={(x['path'],x['defLine']) for x in cen}; chk(qc<=dc,'census keys queue subset of decisions'); print('census extra vs queue:',sorted(dc-qc))
chk({d['ordinal'] for d in dd if d['category']=='adjudicate'}=={x['ordinal'] for x in dec if x['source']=='adjudicate'},'adjudicate in output')
chk(len(dd)==len(q)+len(dc-qc),'decision count == queue + extra census')
for a,b in((set(x['siteId'] for x in q if 'siteId' in x),set(x['siteId'] for x in dd if 'siteId' in x)),): chk(a==b,'siteId sets')
# census membership of T
cs={(c['path'],c['defLine']):c for c in cen}
missing=[(f['ordinal'],t['path'].split('/')[-1],t['defLine']) for f in fin for t in f['T'] if (t['path'],t['defLine']) not in cs]
print('T targets not in census:',missing)
bad=[(f['ordinal'],t['defLine'],t['role'],cs[(t['path'],t['defLine'])]['role']) for f in fin for t in f['T'] if (t['path'],t['defLine']) in cs and cs[(t['path'],t['defLine'])]['role']!=t['role']]
print('T role != census role:',bad)
print('T at census role none:',[(f['ordinal']) for f in fin for t in f['T'] if cs.get((t['path'],t['defLine']),{}).get('role')=='none'])
# audit-confirmed sites: label equals final?
lab={o['ordinal']:o for o in jl(R+'labels/kotlin.jsonl')}
diff=[]
for f in fin:
    if f['basis'] in('agreed','audit-confirmed'):
        l=lab[f['ordinal']]; lt={(c['path'],c['defLine']) for c in l['callees'] if c['kind']!='external'}; ft={(t['path'],t['defLine']) for t in f['T']}
        if lt!=ft: diff.append(f['ordinal'])
print('agreed/audit-confirmed whose final T != label targets:',diff)
# audit-confirmed decision sets equal final
dm={(x['category'],x['ordinal']):x for x in dd if 'ordinal' in x}
dif2=[]
for f in fin:
    for c in('audit-U','audit-C'):
        x=dm.get((c,f['ordinal']))
        if x and x['basis']=='audit-confirmed':
            ds={(t['path'],t['defLine'],t['role']) for t in x['set']['callees']}; fs={(t['path'],t['defLine'],t['role']) for t in f['T']}
            if ds!=fs or sorted(e['package'] for e in x['set']['external'])!=sorted(f['external']): dif2.append(f['ordinal'])
print('audit-confirmed decision set != final:',dif2)
print('underivable',sum(1 for f in fin if f['notScoredReason']=='underivable'),'unscored',sum(1 for f in fin if not f['scored']),'cap',0.05*430)
print('basis',collections.Counter(f['basis'] for f in fin)); print('scored false',[f['ordinal'] for f in fin if not f['scored']])
print('audit lines',len(aud),collections.Counter(a['draw'] for a in aud),'C pooled any',any(a.get('pooled') for a in aud if a['draw']=='C'))
print('dec lines',len(dec),collections.Counter(x['source'] for x in dec),'census lines',len(cen))
print('keysets dec',len({tuple(sorted(x.keys())) for x in dec}),'aud',len({tuple(sorted(x.keys())) for x in aud}))
print('ERRORS',err)
