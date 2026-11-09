import json,re,collections

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF=(_CPG+'/spikes/frontend-fitness'); AD=FF+'/adjudication'
rd=lambda p:[json.loads(l) for l in open(p) if l.strip()]
fin=rd(FF+'/truth/java.final.jsonl'); q=rd(AD+'/java.queue.jsonl'); dec=rd(AD+'/java.decisions.jsonl'); aud=rd(AD+'/java.audit.jsonl'); adj=rd(AD+'/java.jsonl'); lab=rd(FF+'/labels/java.jsonl')
err=[]
def chk(c,m):
    if not c: err.append(m)
chk(len(fin)==500,'final != 500')
chk([f['ordinal'] for f in fin]==list(range(1,501)),'ordinals not 1..500')
chk({f['siteId'] for f in fin}=={l['siteId'] for l in lab},'siteId set != labels')
BASIS={'agreed','audit-confirmed','adjudicated','audit-overturned','sweep-overturned','label-only'}
cite=re.compile(r'^[\w./-]+\.java:\d+')
for f in fin:
    chk(f['basis'] in BASIS,f"basis {f['ordinal']}")
    chk(bool(cite.match(f['citation'] or '')),f"citation {f['ordinal']}")
    chk(set(f)=={'ordinal','siteId','T','N','external','notACall','dynamic','scored','notScoredReason','basis','strata','citation'},f"keys {f['ordinal']}")
    chk(bool(f['T'] or f['external'] or f['N'] or f['notACall'] or f['dynamic']),f"empty truth {f['ordinal']}")
    for t in f['T']: chk(t['role'] in('impl','decl') and t['defLine']>0,f"T {f['ordinal']}")
und=sum(1 for f in fin if f['notScoredReason']=='underivable'); chk(und/500<=0.05,'underivable cap')
fo={f['ordinal']:f for f in fin}
for o,pk in((425,'org.mockito'),(486,'org.assertj.core.api')):
    chk(fo[o]['external']==[pk] and not fo[o]['T'] and fo[o]['basis']=='adjudicated',f'corr {o}')
# key sets
qs={(x['ordinal'],x['siteId']) for x in q if x['category']!='census'}
ds={(x['ordinal'],x['siteId']) for x in dec if x['category']!='census'}
chk(qs==ds,f'site keys queue vs decisions {len(qs)} {len(ds)}')
qc={(x['file'],x['line']) for x in q if x['category']=='census'}; dc={(x['path'],x['defLine']) for x in dec if x['category']=='census'}
chk(qc==dc,'census keys')
qcat=collections.Counter(x['category'] for x in q); dcat=collections.Counter(x['category'] for x in dec)
chk(qcat==dcat,f'category counts {qcat} {dcat}')
chk(len(aud)==252-0 and all(a['labelCorrect'] is not None for a in aud),'audit outcomes')
chk({(a['draw'],a['stratum'],a['siteId']) for a in aud}=={(x['category'][6],x['category'][8:],x['siteId']) for x in q if x['category'].startswith('audit')},'audit keys vs queue')
chk({a['siteId'] for a in adj}=={x['siteId'] for x in q if x['category']=='adjudicate'},'java.jsonl keys')
for a in adj: chk(len(a['evidence'])<=300 and cite.search(a['evidence']) is not None,f"adj evidence {a['ordinal']}")
# every final site in a decision/queue category or agreed
dsd={s for _,s in ds}
for f in fin:
    if f['basis'] in('audit-confirmed','adjudicated'): chk(f['siteId'] in dsd,f"no decision {f['ordinal']}")
    if f['basis']=='agreed': chk(f['siteId'] not in dsd,f"agreed but decided {f['ordinal']}")
# final unchanged vs label except 425/486
for f,l in zip(fin,lab):
    if f['ordinal'] in(425,486): continue
    li={(c['path'],c['defLine']) for c in l['callees'] if c['kind']!='external'}; le={c['package'] for c in l['callees'] if c['kind']=='external'}
    chk(li=={(t['path'],t['defLine']) for t in f['T']} and le==set(f['external']),f"differs from label {f['ordinal']}")
# decided callees agree with final T for adjudicated impl-ctor sites
for d in dec:
    if d['category']=='adjudicate' and d['decision']=='confirmed':
        chk({(c['path'],c['defLine'],c['role']) for c in d['decided']['callees']}=={(t['path'],t['defLine'],t['role']) for t in fo[d['ordinal']]['T']},f"role mismatch decided vs final {d['ordinal']}")
for d in dec:
    if d['category']!='census':
        for c in d.get('callees',[]):
            chk(any((t['path'],t['defLine'],t['role'])==(c['path'],c['defLine'],c['role']) for t in fo[d['ordinal']]['T']) or d['ordinal'] in(425,486),f"decision callee role vs final {d['ordinal']} {c['defLine']}")
print('basis',dict(collections.Counter(f['basis'] for f in fin)),'underivable',und)
print('ERRORS',len(err));[print(' ',e) for e in err[:30]]
