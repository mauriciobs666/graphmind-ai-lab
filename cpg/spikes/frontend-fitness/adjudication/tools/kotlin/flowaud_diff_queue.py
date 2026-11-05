#!/usr/bin/env python3
"""Run AFTER kotlin-flow-relabel.jsonl exists. Writes kotlin-flow-audit.diff.jsonl (F2) and
kotlin-flow-audit.queue.jsonl (F1). Pair identity: kind + endpoint keys; param=index, call=(calleeName,ordinal),
ret=ordinal, cond=(line, whitespace-collapsed code); evidence/confidence/line-of-call/ret/name ignored."""
import json, os, re
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
FF = os.path.join(A, '..')
CORP = os.path.join(FF, '..', '..', '.ccg-work', 'corpora', 'kotlin', 'src')
inp = json.load(open(os.path.join(A, 'kotlin-flow-audit.input.json')))
gold = {r['ordinal']: r for r in map(json.loads, open(os.path.join(FF, 'labels', 'kotlin-flow.jsonl')))}
rel = {r['ordinal']: r for r in map(json.loads, open(os.path.join(A, 'kotlin-flow-relabel.jsonl')))}
smp = {r['ordinal']: r for r in map(json.loads, open(os.path.join(FF, 'sample', 'kotlin-flow.jsonl')))}
ws = lambda s: re.sub(r'\s+', ' ', s).strip()
def ek(e):
    p = e['point']
    if p == 'param': return ('param', e['index'])
    if p == 'call': return ('call', e['calleeName'], e['ordinal'])
    if p == 'ret': return ('ret', e['ordinal'])
    return ('cond', e['line'], ws(e['code']))
key = lambda p: (p['kind'], ek(p['source']), ek(p['sink']))
diff = []; stats = {}
for o in inp['f2Methods']:
    g = {key(p): p for p in gold[o]['pairs']}; r = {key(p): p for p in rel[o]['pairs']}
    both = set(g) & set(r)
    stats[o] = {'gold': len(g), 'relabel': len(r), 'both': len(both)}
    for kk in g:
        if kk not in r: diff.append({'ordinal': o, 'qname': gold[o]['qname'], 'side': 'goldOnly', 'pair': g[kk]})
    for kk in r:
        if kk not in g: diff.append({'ordinal': o, 'qname': gold[o]['qname'], 'side': 'relabelOnly', 'pair': r[kk]})
with open(os.path.join(A, 'kotlin-flow-audit.diff.jsonl'), 'w') as f:
    for d in diff: f.write(json.dumps(d) + '\n')
print('F2 per-method', stats); print('goldOnly', sum(d['side'] == 'goldOnly' for d in diff), 'relabelOnly', sum(d['side'] == 'relabelOnly' for d in diff))
src = {}
def line(path, n):
    if path not in src: src[path] = open(os.path.join(CORP, path), encoding='utf-8', errors='replace').read().split('\n')
    return src[path][n - 1] if 0 < n <= len(src[path]) else None
q = []
for i, d in enumerate(inp['f1']):
    o = d['ordinal']; m = gold[o]; p = m['pairs'][d['pairIndex']]
    ls = set()
    for e in (p['source'], p['sink']):
        ls.add(e['line'] if 'line' in e else smp[o]['lineStart'])
    q.append({'rank': i + 1, 'ordinal': o, 'pairIndex': d['pairIndex'], 'path': m['path'], 'qname': m['qname'],
              'methodSpan': [smp[o]['lineStart'], smp[o]['lineEnd']], 'unsure': p.get('confidence') == 'unsure',
              'pair': p, 'sourceLines': [{'line': n, 'text': line(m['path'], n)} for n in sorted(ls)]})
with open(os.path.join(A, 'kotlin-flow-audit.queue.jsonl'), 'w') as f:
    for x in q: f.write(json.dumps(x) + '\n')
print('queue', len(q), 'unsure', sum(x['unsure'] for x in q), 'kind', {k: sum(x['pair']['kind'] == k for x in q) for k in ('data', 'control')})
