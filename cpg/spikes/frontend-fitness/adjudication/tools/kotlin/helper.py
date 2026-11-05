import json,re,subprocess,sys,io
_o=sys.stdout; sys.stdout=io.StringIO()
import build as B
sys.stdout=_o
from build import *
SR=SRC
byord={r['ordinal']:r for r in rows}
labs=[json.loads(l) for l in open(R+'labels/kotlin.jsonl')]
n=0
for o in labs:
    cs=[c for c in o['callees'] if c['kind']!='external']
    if len(cs)<3: continue
    name=re.sub(r'#\d+$','',cs[0]['qname'].split('::')[1].split('.')[-1])
    if name=='map': continue
    out=subprocess.run(['grep','-rnE',r'fun\s+'+name+r'\s*\(',SR,'--include=*.kt'],capture_output=True,text=True).stdout.strip().split('\n')
    allp={(l.split(':')[0].replace(SR,''),int(l.split(':')[1])) for l in out if l}
    lab={(c['path'],c['defLine']) for c in cs}
    miss=sorted(allp-lab)
    if not miss: continue
    siteitem('helper-audit',byord[o['ordinal']],{'helper':'labeller-scripts/kotlin/w.py panel()/DEF() fan-out','calleeName':name,
      'labelledDefs':sorted(lab),'sameNameDeclsInCorpusNotInLabel':[{'where':f'{p}:{d}','line':lines(p)[d-1].strip()[:160]} for p,d in miss],
      'question':'Does the receiver type at the site include these implementers (label fan-out incomplete) or not (correctly excluded)?'})
    n+=1
print('helper items',n)
