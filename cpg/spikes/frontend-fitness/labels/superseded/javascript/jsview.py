import json,sys
S='/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/javascript/src/'
a,b=int(sys.argv[1]),int(sys.argv[2])
for l in open('/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/sample/javascript.jsonl'):
    d=json.loads(l)
    if a<=d['ordinal']<=b:
        L=open(S+d['path']).read().split('\n')
        print(d['ordinal'],d['path'],d['line'],d['column'],'|',d['callText'][:60],'|',L[d['line']-1].strip()[:130])
