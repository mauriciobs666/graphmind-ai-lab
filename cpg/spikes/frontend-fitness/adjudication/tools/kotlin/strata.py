import json,hashlib,re,collections

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
R=(_CPG+'/spikes/frontend-fitness/')
rows=[json.loads(l) for l in open(R+'adjudication/kotlin.input.jsonl')]
ref={json.loads(l)['ordinal']:json.loads(l) for l in open(R+'truth/kotlin.ref.jsonl')}
def h(stratum,sid): return hashlib.sha256(f'ccg-p0a-v1-aud|kotlin|{stratum}|{sid}'.encode()).hexdigest()
def stratum(r):
    if r['status']!='agreed': return None
    lab_in=bool(r['inCorpusLabel'] or r['declaredLabel'])
    k=r['reference']['static']['kind']
    if lab_in: return 'AI'
    if k=='silent': return 'AS'
    return 'AE'
def cell(r):
    L=r['label']
    if L['notACall']: return 'notACall'
    c=L['callees'][0] if L['callees'] else None
    if c is None: return 'nocallee'
    return c['kind']+'|'+(c['package'] if c['kind']=='external' else c['path'])
def cname(r):
    t=r['site']['callText']
    t=re.sub(r'\s+','',t)
    m=re.findall(r'[A-Za-z_][A-Za-z_0-9]*',t)
    return m[-1] if m else ''
