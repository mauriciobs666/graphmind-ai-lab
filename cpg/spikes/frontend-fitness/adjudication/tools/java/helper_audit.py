import sys,re,json,collections
sys.path.insert(0,'/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/labeller-scripts/java')
sys.dont_write_bytecode=True
import lab
SRC=lab.R
# duplicate simple type names (types dict is keyed by simple name -> later file overwrites)
import os
names=collections.defaultdict(list)
for d,_,fs in os.walk(SRC):
  for f in fs:
    if f.endswith('.java'):
      rel=os.path.relpath(os.path.join(d,f),SRC)
      for i,l in enumerate(open(os.path.join(d,f),errors='replace').read().split('\n'),1):
        m=re.match(r'^\s*(?:(?:public|private|protected|static|final|abstract|sealed)\s+)*(class|interface|enum|record)\s+(\w+)',l)
        if m: names[m.group(2)].append((rel,i))
print('types seen by regex:',len(names),'dup simple names:',{k:v for k,v in names.items() if len(v)>1})
print('lab.types',len(lab.types),'lab.meths',len(lab.meths))
# method key collisions (type,name,n) with >1 entry
c=collections.Counter((m['type'],m['name'],m['n']) for m in lab.meths)
print('overload-keys with >1 hit:',[(k,v) for k,v in c.items() if v>1][:20])
# labels: check defLine text
L=[json.loads(l) for l in open('labels/java.jsonl')]
bad=[];multi=[]
for x in L:
  ic=[c for c in x['callees'] if c['kind'] in('internal','declared')]
  if len(ic)>1: multi.append((x['ordinal'],[(c['kind'],c['path'].split('/')[-1],c['defLine']) for c in ic]))
  for c in ic:
    lines=open(f"{SRC}/{c['path']}",errors='replace').read().split('\n')
    t=lines[c['defLine']-1]
    nm=c['qname'].split('::')[-1].split('.')[-1]
    if nm=='<init>': nm=c['qname'].split('::')[-1].split('.')[-2]
    if nm not in t: bad.append((x['ordinal'],c['path'].split('/')[-1],c['defLine'],nm,t.strip()[:80]))
print('multi-callee labels',len(multi));[print(' ',m) for m in multi]
print('defLine text lacks callee name:',len(bad));[print(' ',b) for b in bad]
