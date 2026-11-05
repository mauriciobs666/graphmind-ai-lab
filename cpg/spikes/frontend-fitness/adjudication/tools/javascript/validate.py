import json,re
FF='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/'
rd=lambda p:[json.loads(l) for l in open(FF+p)]
Q=rd('adjudication/javascript.queue.jsonl'); D=rd('adjudication/javascript.decisions.jsonl')
F=rd('truth/javascript.final.jsonl'); A=rd('adjudication/javascript.audit.jsonl'); J=rd('adjudication/javascript.jsonl')
LAB={l['siteId'] for l in rd('labels/javascript.jsonl')}
err=[]
qs={q['siteId'] for q in Q if q['item']=='site'}; ds={d['siteId'] for d in D if d['item']=='site'}
qc={(q['path'],q['defLine']) for q in Q if q['item']=='census'}; dc={(d['path'],d['defLine']) for d in D if d['item']=='census'}
print('queue/decision site keys equal',qs==ds,len(qs),'census equal',qc==dc,len(qc))
print('final keys == labels',{f['siteId'] for f in F}==LAB,len(F),len({f['siteId'] for f in F}))
print('queue sites subset of final',qs<=LAB)
BAS={'agreed','audit-confirmed','adjudicated','audit-overturned','sweep-overturned','label-only'}
pat=re.compile(r'[\w./@-]+\.\w+:\d+')
print('bad basis',sum(f['basis'] not in BAS for f in F))
print('no citation',sum(not pat.search(f['evidence']) and not pat.search(f['site']) for f in F),'no citation in evidence',sum(not pat.search(f['evidence']) for f in F))
print('decision evidence >300 or uncited',sum(len(d['evidence'])>300 or not pat.search(d['evidence']) for d in D))
print('final evidence >300',sum(len(f['evidence'])>300 for f in F))
und=sum((f['notScoredReason']=='underivable') for f in F); print('underivable',und,und/len(F),'cap 0.05 ok',und/len(F)<=0.05)
print('T roles',{t['role'] for f in F for t in f['T']}, 'T without role',sum(t['role'] is None for f in F for t in f['T']))
print('final T target not in census',sum((t['path'],t['defLine']) not in dc for f in F for t in f['T']))
print('audit lines',len(A),'jsonl lines',len(J))
print('internal decisions with empty T',sum(1 for f in F if not f['T'] and not f['external'] and not f['notACall']))
