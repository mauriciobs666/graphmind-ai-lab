import json,re,collections

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
R=(_CPG+'/spikes/frontend-fitness/')
J=lambda p:[json.loads(l) for l in open(R+p)]
fin=J('truth/cpp.final.jsonl'); dec=J('adjudication/cpp.decisions.jsonl'); q=J('adjudication/cpp.queue.jsonl'); inp=J('adjudication/cpp.input.jsonl')
adj=J('adjudication/cpp.jsonl'); aud=J('adjudication/cpp.audit.jsonl'); cen=J('adjudication/cpp.census.jsonl'); noise=json.load(open(R+'adjudication/cpp.noise.json'))
ok=True
def chk(n,c,extra=''):
    global ok; ok&=bool(c); print(('PASS ' if c else 'FAIL ')+n,extra)
BASIS={'agreed','audit-confirmed','adjudicated','audit-overturned','sweep-overturned','label-only'}
chk('final N=187, ordinals 1..187 once',[x['ordinal'] for x in fin]==list(range(1,188)))
chk('siteIds match input',[x['siteId'] for x in fin]==[x['siteId'] for x in inp])
chk('basis valid',all(x['basis'] in BASIS for x in fin),dict(collections.Counter(x['basis'] for x in fin)))
chk('citation path:line every line',all(re.search(r'[\w./-]+:\d+',x['evidence']) and len(x['evidence'])<=300 for x in fin))
chk('site citation present',all(re.fullmatch(r'.+:\d+',x['site']) for x in fin))
chk('unscored reasons valid',all(x['scored'] or x['notScoredReason'] in('notACall','dynamic','macro-underdetermined','underivable') for x in fin),dict(collections.Counter(x['notScoredReason'] for x in fin if not x['scored'])))
chk('underivable <=0.05',sum(x['notScoredReason']=='underivable' for x in fin)/187<=0.05)
chk('scored has T or external or N',all((x['T'] or x['external'] or x['N']) for x in fin if x['scored']))
chk('sites 33 dynamic empty T',fin[32]['dynamic'] and not fin[32]['T'] and not fin[32]['scored'])
chk('sites 74,146 notACall',fin[73]['notACall'] and fin[145]['notACall'])
for name,rows,cat in(('decisions-vs-queue(adjudicate)',[x['ordinal'] for x in dec if x['queueCategory']=='adjudicate'],'adjudicate'),('audit',[x['ordinal'] for x in dec if x['queueCategory']=='audit'],'audit')):
    qs=sorted(x['ordinal'] for x in q if x['queueCategory']==cat); chk('keys '+name,sorted(rows)==qs,len(qs))
chk('keys macro',sorted(x['macroName'] for x in dec if x['queueCategory']=='macro')==sorted(x['macroName'] for x in q if x['queueCategory']=='macro'))
dc={(x['path'],x['defLine']) for x in dec if x['queueCategory']=='census'}; qc={(x['path'],x['defLine']) for x in q if x['queueCategory']=='census'}
chk('keys census: queue 146 subset of decisions 175, 29 appended',qc<=dc and len(dc)==175 and len(dc-qc)==29,(len(qc),len(dc)))
chk('audit file keys',sorted((x['ordinal']) for x in aud)==sorted(x['ordinal'] for x in q if x['queueCategory']=='audit'),len(aud))
chk('census file == decisions census (175)',len(cen)==175 and {(x['path'],x['defLine']) for x in cen}==dc)
chk('typeCtorMerged 6/0',noise['detail']['v112']['typeCtorMerged']['sites']==6 and noise['detail']['v112']['typeCtorMerged']['referenceErrors']==0)
chk('targetsRead equals distinct',noise['census']['targetsRead']==175==noise['detail']['v112']['census_note']['distinctLabelReferenceAndDecidedTargets'] and noise['detail']['v112']['census_note']['equal'])
chk('no type-row target in T/N',not any((t['path'],t['defLine']) in {(x['path'],x['defLine']) for x in cen if x.get('roleDecided')=='type'} for f in fin for t in f['T']+f['N']))
chk('adjudication file covers 89 adjudicate + macro sites',{x['ordinal'] for x in adj}>= {x['ordinal'] for x in q if x['queueCategory']=='adjudicate'},len(adj))
chk('decisions have citation',all(re.search(r'[\w./-]+:\d+',x['evidence']) for x in adj))
mac={s for x in dec if x['queueCategory']=='macro' for s in x['sites']}
chk('macro sites all carry macro stratum',all(('macro' in f['strata'])==(f['ordinal'] in mac) for f in fin),len(mac))
cs={(x['path'],x['defLine']) for x in cen}
miss=sorted({(t['path'],t['defLine']) for f in fin for t in f['T']+f['N']}-cs)
chk('census closure (every T/N in census)',not miss,f'{len(miss)} missing')
chk('noise N, W',noise['N']==187 and noise['adjudicated']['W']==49)
chk('no errorKind schema left',not any(x.get('errorKind')=='schema' for x in dec+adj))
chk('no label-only/dup basis deviations: AS audited->audit-confirmed',all(f['basis']!='label-only' for f in fin) and all(f['basis']=='audit-confirmed' for f in fin if 'AS' in f['strata'] and f['basis']!='adjudicated'))
chk('e_res flag v1.12',noise['handError']['noiseLimited'] is False and noise['handError']['e_res']['point']<=0.05)
print('ALL OK' if ok else 'INCONSISTENCIES'); 
