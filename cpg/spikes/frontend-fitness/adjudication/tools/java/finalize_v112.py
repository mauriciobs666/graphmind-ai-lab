#!/usr/bin/env python3
"""java S57 FC, v1.12 canonical shape. Reads the v1.11 files kept under tools/java/v111/, the audit/decision files and the
public sample; writes truth/java.final.jsonl and adjudication/java.noise.json (canonical, validate_final.py)."""
import json, math, re, collections

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF = (_CPG+'/spikes/frontend-fitness')
AD = FF + '/adjudication'; V = AD + '/tools/java/v111'
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
old = rd(V + '/java.final.v111.jsonl'); oldn = json.load(open(V + '/java.noise.v111.json'))
aud = rd(AD + '/java.audit.jsonl'); adj = {a['siteId']: a for a in rd(AD + '/java.jsonl')}
samp = {s['ordinal']: s for s in rd(FF + '/sample/java.jsonl')}
labs = rd(FF + '/labels/java.jsonl'); refs = rd(FF + '/truth/java.ref.jsonl'); inp = rd(AD + '/java.input.jsonl'); queue = rd(AD + '/java.queue.jsonl')
assert len(old) == 500 and len(aud) == 252


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

audited = {a['siteId']: a for a in aud}          # draws U and C, any stratum
fin = []
for o in old:
    sid = o['siteId']; tags = [t for t in o['strata']]
    s = samp[o['ordinal']]
    site = f"{s['path']}:{s['line']}"
    if sid in adj:
        basis = 'adjudicated'; ev = adj[sid]['evidence']
    elif sid in audited:
        basis = 'audit-confirmed'; ev = audited[sid]['evidence']
        assert audited[sid]['labelCorrect'] is True
    elif 'AS' in tags:
        basis = 'label-only'; ev = site
    else:
        basis = 'agreed'; ev = site
    if not re.search(r'\S+:\d+', ev): ev = site + '; ' + ev
    tags = [t for t in tags if t != 'needs-adjudication']
    fin.append(dict(ordinal=o['ordinal'], siteId=sid, site=site, T=o['T'], N=o['N'], external=o['external'],
                    notACall=o['notACall'], dynamic=o['dynamic'], scored=o['scored'], notScoredReason=o['notScoredReason'],
                    basis=basis, strata=tags, evidence=ev))
# label-set equality with the v1.11 file (T, external, flags unchanged)
for a, b in zip(fin, old):
    for k in ('T', 'N', 'external', 'notACall', 'dynamic', 'scored', 'notScoredReason'):
        assert a[k] == b[k], (a['ordinal'], k)
before = collections.Counter(o['basis'] for o in old)
with open(FF + '/truth/java.final.jsonl', 'w') as f:
    for r in fin: f.write(json.dumps(r, ensure_ascii=False) + '\n')
after = collections.Counter(r['basis'] for r in fin)
# AS: listed re-read sites must be exactly the AS draw-U sites
asU = sorted(a['ordinal'] for a in aud if a['stratum'] == 'AS' and a['draw'] == 'U')
changed = sorted(r['ordinal'] for r, o in zip(fin, old) if r['basis'] != o['basis'])
assert changed == asU and len(changed) == 40

# noise
N = 500
Ns = {'AI': 100, 'AE': 317, 'AS': 67}
n_ = {s: sum(1 for a in aud if a['draw'] == 'U' and a['stratum'] == s) for s in Ns}
k_ = {s: sum(1 for a in aud if a['draw'] == 'U' and a['stratum'] == s and not a['labelCorrect']) for s in Ns}
kt = {s: sum(1 for a in aud if a['draw'] == 'U' and a['stratum'] == s and not a['labelCorrect'] and a['errorKind'] in ('target', 'kind')) for s in Ns}
up = {s: wilson(k_[s], n_[s])[1] for s in Ns}
W = oldn['adjudicated']['W']
e_all_pt = (W + sum(Ns[s] * k_[s] / n_[s] for s in Ns)) / N
e_all_cons = (W + sum(Ns[s] * up[s] for s in Ns)) / N
e_all_cons_AIcensus = (W + sum(Ns[s] * up[s] for s in ('AE', 'AS'))) / N
e_res_pt = sum((Ns[s] - n_[s]) * kt[s] / n_[s] for s in Ns) / N
e_res_cons = sum((Ns[s] - n_[s]) * up[s] for s in Ns) / N
# census
S = set()
for l in labs:
    for c in l['callees']:
        if c['kind'] != 'external': S.add((c['path'], c['defLine']))
for r in refs:
    st = r['static']
    if st['kind'] in ('internal', 'declared'): S.add((st['path'], st['defLine']))
    for i in r['impls']: S.add((i['path'], i['defLine']))
Fs = {(t['path'], t['defLine']) for r in fin for t in r['T']}
assert Fs <= S
targets = len(S | Fs)
# open points
cls = {(q['file'], q['line']): q for q in queue if q['category'] == 'census' and q['declLineIsType']}
cl_sites = collections.defaultdict(list)
for r in fin:
    for t in r['T']:
        if (t['path'], t['defLine']) in cls: cl_sites[(t['path'], t['defLine'])].append(r['ordinal'])
anon = [t['qname'] for r in fin for t in r['T'] if re.search(r'#\d|<anon|lambda|\$\d', t['qname'] or '')]
dto = sorted(r['ordinal'] for r in fin if 'org.springframework.samples.petclinic.rest.dto' in r['external'])
open_pts = dict(
    classLineTargets=dict(count=len(cl_sites), siteCount=sum(len(v) for v in cl_sites.values()),
        definitions=[dict(path=p, defLine=l, role='impl', sites=sorted(v)) for (p, l), v in sorted(cl_sites.items())],
        note='`new X` with no explicit constructor: T is the class declaration line, role impl (decided by DS per the brief). S27 must check how Joern places the implicit constructor.'),
    anonymousLambdaIndices=dict(found=len(anon), note='none found: no qname in final T carries #n/<anon>/lambda/$n; the two `<init>` qnames are named constructors' if not anon else anon),
    externalsAbsentFromTree=dict(package='org.springframework.samples.petclinic.rest.dto', count=len(dto), sites=dto,
        note='OpenAPI-generated DTOs: package source is absent from the corpus tree (no dto directory under cpg/.ccg-work/corpora/java/src); labelled external'))
hop = oldn.copy()
detail = dict(original_v111=oldn, v112=dict(
    basisRuling='precedence adjudicated > audit-confirmed (any stratum, U or C) > label-only > agreed; the 40 AS draw-U lines moved label-only -> audit-confirmed, each checked against java.audit.jsonl (draw U, stratum AS, labelCorrect true) and equal to the coordinator list',
    basisBefore=dict(before), basisAfter=dict(after), basisChangedOrdinals=changed,
    e_all=dict(point=e_all_pt, conservativeAllWilson=e_all_cons, conservativeAIcensus=e_all_cons_AIcensus, W=W, note='labeller property, drives no gate'),
    e_res=dict(point=e_res_pt, conservative=e_res_cons, unreadRemainder={s: Ns[s] - n_[s] for s in Ns}, r_s=kt, wilsonUpper={s: round(up[s], 4) for s in Ns},
               formula='sum_s (N_s-n_s)*r_s/n_s / N; AI census (n=N_s) contributes 0'),
    v110Flag=dict(figure='e_h with role errors', point=round(e_all_pt, 4), conservative=round(e_all_cons, 4), flagged=False),
    closedClassShare=dict(count=0, share=0.0, cap=0.10, note='no label role/coordinate/shape error found. The one census role error is on the reference side (OwnerMapper.java:27, ordinal 278) and is not in any final line.'),
    census=dict(targetsRead=targets, distinctLabelRefTargets=len(S), distinctFinalT=len(Fs), note='93 = distinct (path, defLine) of labels, reference static/impls and final T (final T is a subset: 425/486 dropped Person.getFirstName)'),
    helperAudit=oldn['helperAudit'],
    openPointsS27=open_pts))
noise = dict(lang='java', N=N, comparable=oldn['comparable']['comparable'], referenceSilent=oldn['comparable']['silentReference'],
    agreement=dict(agreed=oldn['agreementRate']['agreed'], comparable=oldn['agreementRate']['comparable'], rate=oldn['agreementRate']['rate'], wilson95=oldn['agreementRate']['wilson95']),
    adjudicated=dict(n=16, W=W, byWrong=oldn['adjudicated']['byWrong'], byErrorKind=oldn['adjudicated']['byErrorKind']),
    referenceError=dict(wrong=1, comparable=417, rate=oldn['referenceErrorRate']['rate'], wilson95=oldn['referenceErrorRate']['wilson']),
    audit={s: dict(N_s=Ns[s], n=n_[s], k=k_[s], wilson95=[round(wilson(k_[s], n_[s])[0], 4), round(up[s], 4)]) for s in Ns},
    handError=dict(e_all=dict(point=round(e_all_pt, 4), conservative=round(e_all_cons, 4)), e_res=dict(point=round(e_res_pt, 4), conservative=round(e_res_cons, 4)), noiseLimited=bool(e_res_pt > 0.05)),
    census=dict(targetsRead=targets), closedClasses=dict(count=0, byKind=dict(role=[], coordinate=[], shape=[])),
    unscored=dict(byReason={}, underivable=0), basisCounts=dict(after), detail=detail)
json.dump(noise, open(AD + '/java.noise.json', 'w'), indent=1, ensure_ascii=False)
print('before', dict(before)); print('after', dict(after)); print('e_all', e_all_pt, e_all_cons, e_all_cons_AIcensus, 'e_res', e_res_pt, e_res_cons)
print('targetsRead', targets, 'classLine', len(cl_sites), sum(len(v) for v in cl_sites.values()), 'dto', len(dto), 'anon', len(anon))
