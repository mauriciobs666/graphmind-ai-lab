import json,collections
FF='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness'; AD=FF+'/adjudication'
rd=lambda p:[json.loads(l) for l in open(p) if l.strip()]
fin=rd(FF+'/truth/java.final.jsonl'); lab=rd(FF+'/labels/java.jsonl'); aud=rd(AD+'/java.audit.jsonl'); adj=rd(AD+'/java.jsonl'); inp=rd(AD+'/java.input.jsonl')
noise=json.load(open(AD+'/java.noise.json')); err=[]
A={a['siteId'] for a in aud}; J={a['siteId'] for a in adj}
for f,l in zip(fin,lab):
    if f['ordinal'] not in(425,486):
        li={(c['path'],c['defLine']) for c in l['callees'] if c['kind']!='external'}; le={c['package'] for c in l['callees'] if c['kind']=='external'}
        if li!={(t['path'],t['defLine']) for t in f['T']} or le!=set(f['external']): err.append(('label changed',f['ordinal']))
    exp='adjudicated' if f['siteId'] in J else 'audit-confirmed' if f['siteId'] in A else 'label-only' if 'AS' in f['strata'] else 'agreed'
    if f['basis']!=exp: err.append(('basis',f['ordinal'],f['basis'],exp))
    if not any(s in f['strata'] for s in('AI','AE','AS')) and f['siteId'] not in J: err.append(('stratum tag',f['ordinal']))
assert noise['basisCounts']==dict(collections.Counter(f['basis'] for f in fin))
print('errors',err, dict(collections.Counter(f['basis'] for f in fin)))
