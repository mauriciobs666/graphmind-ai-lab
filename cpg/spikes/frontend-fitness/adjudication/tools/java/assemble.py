#!/usr/bin/env python3
import json, math, collections, re

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF = (_CPG+'/spikes/frontend-fitness')
AD = f'{FF}/adjudication'
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
inp = rd(f'{AD}/java.input.jsonl')
queue = rd(f'{AD}/java.queue.jsonl')
dec = rd(f'{AD}/java.decisions.jsonl')
auddraw = rd(f'{AD}/java.audit.jsonl')
byord = {r['ordinal']: r for r in inp}


def wilson(k, n, z=1.959964):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 4), round(min(1, c + h), 4))


# ---- decision indexes
site_dec = {}  # (category, siteId) -> decision
census_dec = {}
for d in dec:
    if d['category'] == 'census':
        census_dec[(d['path'], d['defLine'])] = d
    else:
        assert (d['category'], d['siteId']) not in site_dec
        site_dec[(d['category'], d['siteId'])] = d
roles = {k: v['role'] for k, v in census_dec.items()}

# strata (same rule as build_queue)
def stratum(r):
    if r['status'] != 'agreed': return None
    if r['inCorpusLabel'] or r['declaredLabel']: return 'AI'
    return 'AS' if r['reference']['static']['kind'] == 'silent' else 'AE'

# ---- java.audit.jsonl
audrows = []
for a in auddraw:
    d = site_dec[(f"audit-{a['draw']}-{a['stratum']}", a['siteId'])]
    a = dict(a)
    a['labelCorrect'] = d['labelCorrect']; a['errorKind'] = d['errorKind']; a['wrong'] = d['wrong']
    a['basis'] = d['basis']; a['citations'] = d['citations']; a['evidence'] = d['evidence']
    if 'overloadCheck' in d: a['overloadCheck'] = d['overloadCheck']
    audrows.append(a)
with open(f'{AD}/java.audit.jsonl', 'w') as f:
    for a in audrows: f.write(json.dumps(a, ensure_ascii=False, sort_keys=True) + '\n')

# ---- java.jsonl (adjudicated decisions, §5.4 step 6 shape)
adjud = []
for r in inp:
    if r['status'] != 'needs-adjudication': continue
    d = site_dec[('adjudicate', r['siteId'])]
    dc = d['decided']
    ev = d['evidence']
    cit = d['citations'][0]
    if len(ev) > 300: ev = ev[:300]
    if cit not in ev: ev = (cit + ' ' + ev)[:300]
    adjud.append(dict(ordinal=r['ordinal'], siteId=r['siteId'], reasons=r['reasons'],
                      callees=[dict(path=c['path'], defLine=c['defLine'], role=c['role']) for c in dc['callees']],
                      external=dc.get('external', []), nonSymbol=[], notACall=dc['notACall'], dynamic=dc['dynamic'],
                      wrong=d['wrong'], errorKind=d['errorKind'], evidence=ev, citations=d['citations']))
with open(f'{AD}/java.jsonl', 'w') as f:
    for a in adjud: f.write(json.dumps(a, ensure_ascii=False, sort_keys=True) + '\n')

# ---- final
audited = {}
for a in audrows: audited[a['siteId']] = a
final = []
for r in inp:
    o, sid = r['ordinal'], r['siteId']
    st = stratum(r)
    tags = [st] if st else ['needs-adjudication']
    lab = r['label']
    T, ext = [], []
    if st is None:
        d = site_dec[('adjudicate', sid)]
        if d['decision'] == 'overturned':
            basis = 'adjudicated'
            ext = d['decided'].get('external', [])
            assert not d['decided']['callees']
        else:
            basis = 'adjudicated'
        callees = [] if d['decision'] == 'overturned' else None
        if callees is None:
            for c in lab['callees']:
                if c['kind'] in ('internal', 'declared'):
                    T.append(dict(path=c['path'], qname=c['qname'], defLine=c['defLine'], role=roles[(c['path'], c['defLine'])]))
                else: ext.append(c['package'])
    else:
        for c in lab['callees']:
            if c['kind'] in ('internal', 'declared'):
                T.append(dict(path=c['path'], qname=c['qname'], defLine=c['defLine'], role=roles[(c['path'], c['defLine'])]))
            else: ext.append(c['package'])
        if st == 'AS': basis = 'label-only'
        elif sid in audited: basis = 'audit-confirmed'
        else: basis = 'agreed'
    if any(c['kind'] == 'declared' for c in lab['callees']) and st: tags.append('declared')
    # citation: a path:line of the corpus
    if st is None: cite = site_dec[('adjudicate', sid)]['citations'][0]
    elif sid in audited: cite = audited[sid]['citations'][0]
    else:
        sm = r['site']; cite = None
    final.append(dict(ordinal=o, siteId=sid, T=T, N=[], external=sorted(set(ext)), notACall=bool(lab['notACall']),
                      dynamic=bool(lab['dynamic']), scored=True, notScoredReason=None, basis=basis, strata=tags,
                      citation=cite))
# citation for undrawn agreed sites: sample file:line
samp = {s['ordinal']: s for s in rd(f'{FF}/sample/java.jsonl')}
for f_ in final:
    if f_['citation'] is None:
        s = samp[f_['ordinal']]; f_['citation'] = f"{s['path']}:{s['line']}"
with open(f'{FF}/truth/java.final.jsonl', 'w') as f:
    for x in final: f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n')

# ---- noise
N = len(inp)
silent = sum(1 for r in inp if r['reference']['static']['kind'] == 'silent')
comparable = N - silent
agreed_cmp = sum(1 for r in inp if r['status'] == 'agreed' and stratum(r) != 'AS')
Ns = {s: sum(1 for r in inp if stratum(r) == s) for s in ('AI', 'AE', 'AS')}
auds = {}
for s in Ns:
    U = [a for a in audrows if a['draw'] == 'U' and a['stratum'] == s]
    C = [a for a in audrows if a['draw'] == 'C' and a['stratum'] == s]
    k = sum(1 for a in U if not a['labelCorrect'])
    kc = sum(1 for a in C if not a['labelCorrect'])
    lo, hi = wilson(k, len(U))
    auds[s] = dict(N_s=Ns[s], n=len(U), census=(len(U) == Ns[s]), errors=k, wilson=[lo, hi],
                   drawC=dict(n=len(C), errors=kc, note='read and corrected, reported apart, never pooled'))
adj_by_wrong = collections.Counter(a['wrong'] for a in adjud)
adj_by_kind = collections.Counter(a['errorKind'] for a in adjud)
W = sum(1 for a in adjud if a['wrong'] in ('label', 'both'))
label_role_err = sum(1 for d in census_dec.values() if d['errorKind'] == 'role' and d['wrong'] in ('label', 'both'))
ref_role_err = sum(1 for d in census_dec.values() if d['errorKind'] == 'role' and d['wrong'] in ('reference', 'both'))
audit_role_err = 0


def eh(use_upper_ai, include_role=True):
    tot = W + (label_role_err if include_role else 0)
    for s in Ns:
        a = auds[s]
        if s == 'AI' and not use_upper_ai and a['census']:
            k_over_n = 0 if a['errors'] == 0 else a['errors'] / a['n']
            tot += Ns[s] * k_over_n
        elif use_upper_ai == 'point':
            tot += Ns[s] * a['errors'] / a['n']
        else:
            tot += Ns[s] * a['wilson'][1]
    return tot / N


point = (W + sum(Ns[s] * auds[s]['errors'] / auds[s]['n'] for s in Ns)) / N
cons_ai_census = (W + Ns['AE'] * auds['AE']['wilson'][1] + Ns['AS'] * auds['AS']['wilson'][1] + Ns['AI'] * (0 if auds['AI']['census'] else auds['AI']['wilson'][1])) / N
cons_ai_wilson = (W + sum(Ns[s] * auds[s]['wilson'][1] for s in Ns)) / N
# per stratum agreement (comparable)
def agree(pred):
    rs = [r for r in inp if pred(r) and r['reference']['static']['kind'] != 'silent']
    a = sum(1 for r in rs if r['status'] == 'agreed')
    return dict(agreed=a, comparable=len(rs), rate=round(a / len(rs), 4) if rs else None, wilson=list(wilson(a, len(rs))))
per = dict(overall=agree(lambda r: True), AI=agree(lambda r: stratum(r) == 'AI'), AE=agree(lambda r: stratum(r) == 'AE'),
           declared=agree(lambda r: bool(r['declaredLabel'])),
           relaxed=agree(lambda r: r['reference'].get('relaxed')),
           byConstruction=agree(lambda r: r['reference'].get('byConstruction')),
           macro=dict(note='not applicable to java'))
# reference errors
ref_cmp_errors = ref_role_err   # reference-side errors found among comparable sites (census role); audits judged labels only
ref_rate = dict(comparable=comparable, referenceErrorsFound=ref_cmp_errors, rate=round(ref_cmp_errors / comparable, 4),
                wilson=list(wilson(ref_cmp_errors, comparable)),
                found=[dict(kind='role', path=d['path'], defLine=d['defLine'], ordinals=d['ordinals'],
                            note='reference gives static kind internal at a bodyless interface method; correct role is decl (rule 8)')
                       for d in census_dec.values() if d['errorKind'] == 'role' and d['wrong'] in ('reference', 'both')],
                caveat='audit draws judged the label only (every audited wrong=neither); undrawn agreed sites were not re-decided, so this is a floor, not an estimate over the comparable sites')
ctor = [o for o in (178, 247, 290, 452)]
type_lines = {(d['path'], d['defLine']) for d in census_dec.values()}
qd = {(q['file'], q['line']): q for q in queue if q['category'] == 'census'}
impl_type_sites = sorted(f_['ordinal'] for f_ in final
                         if any(qd.get((t['path'], t['defLine']), {}).get('declLineIsType') for t in f_['T']))
noise = dict(
    lang='java', N=N, labelled=N,
    comparable=dict(silentReference=silent, comparable=comparable, share=round(comparable / N, 4)),
    agreementRate=dict(agreed=agreed_cmp, comparable=comparable, rate=round(agreed_cmp / comparable, 4), wilson95=list(wilson(agreed_cmp, comparable)),
                       note='AS (67, reference silent, single-source) excluded; script-level agreed=484 of 500 includes AS'),
    perStratumAgreement=per,
    adjudicated=dict(n=len(adjud), byWrong=dict(adj_by_wrong), byErrorKind=dict(adj_by_kind), W=W),
    auditTable=auds,
    handErrorRate=dict(
        point=round(point, 4), formula=f'(W={W} + sum_s N_s*k_s/n_s={sum(Ns[s]*auds[s]["errors"]/auds[s]["n"] for s in Ns)}) / N={N}',
        conservativeAIcensus=round(cons_ai_census, 4), conservativeAIwilsonUpper=round(cons_ai_wilson, 4),
        withRoleErrors=dict(labelRoleErrors=label_role_err, point=round(point, 4), conservativeAIcensus=round(cons_ai_census, 4), conservativeAIwilsonUpper=round(cons_ai_wilson, 4)),
        withoutRoleErrors=dict(point=round(point, 4), conservativeAIcensus=round(cons_ai_census, 4), conservativeAIwilsonUpper=round(cons_ai_wilson, 4)),
        flagThreshold=0.05, noiseLimited=bool(max(point, cons_ai_wilson) > 0.05 and point > 0.05),
        note='flag uses the figure with role errors (point); label role errors = 0, so with and without are equal. Conservative bounds are below 0.05 as well.'),
    referenceErrorRate=ref_rate,
    helperAudit=dict(overloadQuestion=dict(ordinals=[278, 305], result='single overload correct in both (278 -> OwnerMapper.java:27 only; 305 -> :23 only); no overload fan-out error', source='java.decisions.jsonl overloadCheck'),
                     defLineNameCheck='0 of all internal/declared label callees had a defLine text lacking the callee name (FC mechanical check)',
                     drawUAIerrors=auds['AI']['errors']),
    scoring=dict(scored=sum(1 for f_ in final if f_['scored']), underivable=0, underivableShare=0.0, cap=0.05),
    basisCounts=dict(collections.Counter(f_['basis'] for f_ in final)),
    openPoints=[dict(id='implicit-constructor-target',
                     text='Sites 178, 247, 290 and 452 (`new X`, X with no explicit constructor) are finalised with T = the class declaration line, role impl (per decision file: "target is the class decl per brief"). Open for the Joern matching step to check whether Joern emits the implicit constructor at the class line or elsewhere; not decided here.',
                     sites=ctor, otherSitesWithTypeDeclLineTarget=[o for o in impl_type_sites if o not in ctor])],
)
json.dump(noise, open(f'{AD}/java.noise.json', 'w'), indent=1, ensure_ascii=False, sort_keys=True)
print(json.dumps(noise, indent=1, ensure_ascii=False)[:6000])
