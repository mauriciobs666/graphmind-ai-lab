#!/usr/bin/env python3
"""S57 java FC: strata, audit draws, census, decision queue. Reads only adjudication/java.input.jsonl
(derived from labels+ref), truth/java.ref.jsonl, sample/java.jsonl (public), corpus src."""
import json, hashlib, os, re, collections

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF = (_CPG+'/spikes/frontend-fitness')
SRC = (_CPG+'/.ccg-work/corpora/java/src')
import os, pathlib as _pl, tempfile as _tf; _SCRATCH = os.environ.get('CCG_SCRATCH') or _tf.gettempdir()
SC = (_SCRATCH+'/s57_java')
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
inp = rd(f'{FF}/adjudication/java.input.jsonl')
ref = {r['ordinal']: r for r in rd(f'{FF}/truth/java.ref.jsonl')}
samp = {r['ordinal']: r for r in rd(f'{FF}/sample/java.jsonl')}
assert len(inp) == 500


def h(stratum, sid):
    return hashlib.sha256(f'ccg-p0a-v1-aud|java|{stratum}|{sid}'.encode()).hexdigest()


def stratum(r):
    if r['status'] != 'agreed':
        return None
    lab = r['label']
    inc = bool(r['inCorpusLabel'] or r['declaredLabel'])
    if inc:
        return 'AI'
    if r['reference']['static']['kind'] == 'silent':
        return 'AS'
    return 'AE'


def cell(r):
    lab = r['label']
    if lab['notACall']:
        return 'notACall'
    parts = sorted({(c['kind'], c.get('path') or c.get('package')) for c in lab['callees']})
    return '|'.join(f'{k}:{v}' for k, v in parts)


strata = collections.defaultdict(list)
for r in inp:
    s = stratum(r)
    if s:
        strata[s].append(r)
SIZE = {'AI': 100, 'AE': 100, 'AS': 40}
draws = {}  # stratum -> {'U':[rows], 'C':[rows]}
for s, rows in strata.items():
    rows = sorted(rows, key=lambda r: h(s, r['siteId']))
    U = rows[:SIZE[s]]
    usedcells = {cell(r) for r in U}
    C, seen = [], set()
    for r in rows[SIZE[s]:]:
        c = cell(r)
        if c not in usedcells and c not in seen:
            seen.add(c); C.append(r)
    draws[s] = dict(U=U, C=C, N=len(rows))
    print(s, 'N', len(rows), 'U', len(U), 'C', len(C))


def ctx(path, line, before=5, after=5):
    L = open(f'{SRC}/{path}', encoding='utf-8', errors='replace').read().split('\n')
    a, b = max(1, line - before), min(len(L), line + after)
    return '\n'.join(f'{i}| {L[i-1]}' for i in range(a, b + 1))


def item(category, r, extra=None):
    sm = samp[r['ordinal']]
    d = dict(category=category, ordinal=r['ordinal'], siteId=r['siteId'],
             file=sm['path'], line=sm['line'], column=sm['column'], callText=sm['callText'],
             status=r['status'], reasons=r['reasons'], label=r['label'],
             reference=r['reference'], source=ctx(sm['path'], sm['line']))
    if extra:
        d.update(extra)
    return d


out = open(f'{FF}/adjudication/java.queue.jsonl', 'w')
audit_idx = []
counts = collections.Counter()


def emit(d):
    out.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + '\n'); out.flush()
    counts[d['category']] += 1


# 1. needs-adjudication, no sampling
for r in inp:
    if r['status'] == 'needs-adjudication':
        emit(item('adjudicate', r, dict(stratum=None)))
# 2. audit draws
for s in ('AI', 'AE', 'AS'):
    for rank, r in enumerate(draws[s]['U'], 1):
        emit(item(f'audit-U-{s}', r, dict(stratum=s, drawRank=rank, cell=cell(r), hash=h(s, r['siteId']))))
        audit_idx.append(dict(draw='U', stratum=s, rank=rank, ordinal=r['ordinal'], siteId=r['siteId'], cell=cell(r)))
    for rank, r in enumerate(draws[s]['C'], 1):
        emit(item(f'audit-C-{s}', r, dict(stratum=s, drawRank=rank, cell=cell(r), hash=h(s, r['siteId']))))
        audit_idx.append(dict(draw='C', stratum=s, rank=rank, ordinal=r['ordinal'], siteId=r['siteId'], cell=cell(r)))

# 3. census
cen = collections.OrderedDict()  # (path,defLine) -> info


def add(path, line, src, kindsrc, ordinal):
    e = cen.setdefault((path, line), dict(path=path, defLine=line, labelKinds=set(), refKinds=set(), ordinals=[]))
    e[kindsrc].add(src)
    if ordinal not in e['ordinals']:
        e['ordinals'].append(ordinal)


for r in inp:
    o = r['ordinal']
    for c in r['label']['callees']:
        if c['kind'] in ('internal', 'declared'):
            add(c['path'], c['defLine'], c['kind'], 'labelKinds', o)
    st = r['reference']['static']
    if st['kind'] in ('internal', 'declared'):
        add(st['path'], st['defLine'], st['kind'], 'refKinds', o)
    for im in r['reference']['impls']:
        add(im['path'], im['defLine'], im.get('kind', 'impl'), 'refKinds', o)


def form(path, line):
    """mechanical classification: scan from defLine to first '{' or ';' (outside parens)."""
    L = open(f'{SRC}/{path}', encoding='utf-8', errors='replace').read().split('\n')
    txt, depth = '', 0
    for i in range(line - 1, min(len(L), line + 14)):
        for ch in L[i]:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            elif depth <= 0 and ch == '{': return 'body', txt.strip() + '{'
            elif depth <= 0 and ch == ';': return 'bodyless', txt.strip() + ';'
            txt += ch
        txt += ' '
    return 'unknown', txt.strip()[:200]


census_rows = []
for (p, l), e in cen.items():
    fm, sig = form(p, l)
    lk, rk = sorted(e['labelKinds']), sorted(e['refKinds'])
    flags = []
    if 'internal' in lk and fm == 'bodyless': flags.append('label-internal-at-bodyless')
    if 'declared' in lk and fm == 'body': flags.append('label-declared-at-body')
    if 'internal' in rk and fm == 'bodyless': flags.append('ref-internal-at-bodyless')
    if 'declared' in rk and fm == 'body': flags.append('ref-declared-at-body')
    first = e['ordinals'][0]
    # whether the line starts a type declaration (class/interface/enum/record) -> implicit ctor / type site
    L = open(f'{SRC}/{p}', encoding='utf-8', errors='replace').read().split('\n')
    srcline = L[l - 1].strip() if l - 1 < len(L) else ''
    is_type = bool(re.search(r'\b(class|interface|enum|record)\s+\w+', srcline))
    d = dict(category='census', censusId=len(census_rows) + 1, file=p, line=l, labelKinds=lk, refKinds=rk,
             fcForm=fm, fcSignature=sig[:300], declLineIsType=is_type, flags=flags, ordinals=e['ordinals'],
             source=ctx(p, l, 2, 8))
    census_rows.append(d)
    emit(d)

json.dump(dict(counts=counts, strataN={s: draws[s]['N'] for s in draws}), open(f'{SC}/counts.json', 'w'), default=dict)
with open(f'{FF}/adjudication/java.audit.jsonl', 'w') as f:
    # draws and (pending) outcomes
    for a in audit_idx:
        a['labelCorrect'] = None
        f.write(json.dumps(a, sort_keys=True) + '\n')
print(dict(counts))
print('census', len(census_rows), 'flags', collections.Counter(x for d in census_rows for x in d['flags']),
      'forms', collections.Counter(d['fcForm'] for d in census_rows), 'types', sum(d['declLineIsType'] for d in census_rows))
print('label-kind distinct declared:', len({(d['file'], d['line']) for d in census_rows if 'declared' in d['labelKinds']}),
      'ref-declared:', len({(d['file'], d['line']) for d in census_rows if 'declared' in d['refKinds']}))
