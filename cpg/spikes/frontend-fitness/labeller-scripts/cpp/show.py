import json,sys
a,b=int(sys.argv[1]),int(sys.argv[2])
for l in open('/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/sample/cpp.jsonl'):
    d=json.loads(l)
    if a<=d['ordinal']<=b:
        ex=[x for x in d['excerpt'].split('\n') if x.startswith(str(d['line'])+'|')]
        print(d['ordinal'],d['path'].replace('googletest','GT').replace('googlemock','GM'),d['line'],d['column'],d['callText'][:70],'|',(ex[0] if ex else '')[:120].split('|',1)[-1].strip())
