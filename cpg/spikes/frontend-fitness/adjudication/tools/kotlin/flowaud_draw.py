#!/usr/bin/env python3
"""Part F draw for kotlin. Reads labels/kotlin-flow.jsonl ONLY for structure (ordinal, pair index,
confidence) and writes adjudication/kotlin-flow-audit.input.json. Prints no label content.
Hash message: seed + "<ordinal>|<pairIndex0>" (pair: 0-based position in the method's pairs list);
method: seed + "<ordinal>"."""
import json, hashlib, os
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
L = 'kotlin'
rows = [json.loads(l) for l in open(os.path.join(H, '..', 'labels', f'{L}-flow.jsonl'))]
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
pairs = []
for r in rows:
    for i, p in enumerate(r['pairs']):
        pairs.append({'ordinal': r['ordinal'], 'pairIndex': i, 'unsure': p.get('confidence') == 'unsure',
                      'hash': sha(f'ccg-p0a-v1-flowaud|{L}|pair|{r["ordinal"]}|{i}')})
uns = sorted([p for p in pairs if p['unsure']], key=lambda p: p['hash'])[:20]
ids = {(p['ordinal'], p['pairIndex']) for p in uns}
rest = sorted([p for p in pairs if (p['ordinal'], p['pairIndex']) not in ids], key=lambda p: p['hash'])
f1 = (uns + rest)[:60]
ms = sorted(({'ordinal': r['ordinal'], 'hash': sha(f'ccg-p0a-v1-flowaud|{L}|method|{r["ordinal"]}')} for r in rows), key=lambda m: m['hash'])
out = {'language': L, 'nMethods': len(rows), 'nPairs': len(pairs), 'nUnsurePairs': sum(p['unsure'] for p in pairs),
       'hashMessage': 'pair: ccg-p0a-v1-flowaud|kotlin|pair|<ordinal>|<pairIndex0>; method: ccg-p0a-v1-flowaud|kotlin|method|<ordinal>',
       'f1': [{k: p[k] for k in ('ordinal', 'pairIndex', 'unsure', 'hash')} for p in f1],
       'f2Methods': [m['ordinal'] for m in ms[:6]], 'f2Extension': [m['ordinal'] for m in ms[6:9]]}
json.dump(out, open(os.path.join(H, f'{L}-flow-audit.input.json'), 'w'), indent=1)
print({k: out[k] for k in ('nMethods', 'nPairs', 'nUnsurePairs', 'f2Methods', 'f2Extension')}, 'f1', len(f1), 'unsureInDraw', sum(p['unsure'] for p in f1))
print('f1 ordinals present:', sorted({p['ordinal'] for p in f1}))
