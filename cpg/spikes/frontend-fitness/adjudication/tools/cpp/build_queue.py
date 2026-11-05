import json,re,hashlib,collections,sys
R='/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/'
SRC='/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/cpp/src/'
OUT=R+'adjudication/cpp.queue.jsonl'
rows=[json.loads(l) for l in open(R+'adjudication/cpp.input.jsonl')]
refs={json.loads(l)['ordinal']:json.loads(l) for l in open(R+'truth/cpp.ref.jsonl')}
cache={}
def lines(p):
    if p not in cache:
        try: cache[p]=open(SRC+p,encoding='utf-8',errors='replace').read().split('\n')
        except FileNotFoundError: cache[p]=None
    return cache[p]
def exc(p,ln,before=5,after=4):
    L=lines(p)
    if L is None: return f"<missing {p}>"
    a=max(1,ln-before); b=min(len(L),ln+after)
    return "\n".join(f"{i}| {L[i-1]}" for i in range(a,b+1))
def h(stratum,sid): return hashlib.sha256(f"ccg-p0a-v1-aud|cpp|{stratum}|{sid}".encode()).hexdigest()
MAC=re.compile(r'[A-Z][A-Z0-9_]*')
def macro_name(r):
    t=r['site']['callText'].strip()
    return t if MAC.fullmatch(t) else None
def cell(r):
    l=r['label']
    if l['notACall']: return 'notACall'
    ins=sorted({(c['kind'],c['path']) for c in l['callees'] if c['kind']!='external'})
    if ins: return 'in:'+';'.join(f"{k}@{p}" for k,p in ins)
    return 'ext:'+';'.join(sorted({c['package'] for c in l['callees']})) if l['callees'] else 'empty'
def stratum(r):
    k=r['reference']['static']['kind']
    if k=='silent': return 'AS'
    return 'AI' if r['inCorpusLabel'] or r['declaredLabel'] else 'AE'
def base(cat,r):
    return {"queueCategory":cat,"ordinal":r['ordinal'],"siteId":r['siteId'],"file":r['site']['path'],"line":r['site']['line'],
            "callText":r['site']['callText'],"status":r['status'],"reasons":r['reasons'],"label":r['label'],
            "reference":r['reference'],"macroName":macro_name(r)}
def defex(r):
    d={}
    for p,l in list(r['inCorpusLabel'])+list(r['declaredLabel'])+list(r['inCorpusReference']):
        d[f"{p}:{l}"]=exc(p,l,1,3)
    return d
out=open(OUT,'w'); counts=collections.Counter()
def emit(o): out.write(json.dumps(o)+"\n"); out.flush(); counts[o['queueCategory']]+=1
# 1 adjudicate (non-macro and macro-flagged; macro unsure sites are decided via macro items but listed once here with decisionVia)
macro_sites=[r for r in rows if macro_name(r)]
for r in rows:
    if r['status']!='needs-adjudication': continue
    o=base('adjudicate',r); o['excerpt']=exc(r['site']['path'],r['site']['line']); o['targetExcerpts']=defex(r)
    o['emptyCalleeNoNotACall']= (not r['label']['callees'] and not r['label']['notACall'])
    if o['macroName']: o['decisionVia']='macro:'+o['macroName']
    emit(o)
# 2 audits
strata=collections.defaultdict(list)
for r in rows:
    if r['status']=='agreed': strata[stratum(r)].append(r)
N={'AI':60,'AE':60,'AS':30}
auditmeta={}
for s,lst in strata.items():
    lst.sort(key=lambda r:h(s,r['siteId']))
    U=lst[:N[s]]; hit={cell(r) for r in U}
    seen=set(); C=[]
    for r in lst[N[s]:]:
        c=cell(r)
        if c in hit or c in seen: continue
        seen.add(c); C.append(r)
    auditmeta[s]={'N_s':len(lst),'nU':len(U),'nC':len(C),'cells':len({cell(r) for r in lst})}
    for draw,L in (('U',U),('C',C)):
        for i,r in enumerate(L,1):
            o=base('audit',r); o.update(stratum=s,draw=draw,drawRank=i,cell=cell(r),excerpt=exc(r['site']['path'],r['site']['line']),targetExcerpts=defex(r))
            emit(o)
json.dump(auditmeta,open('/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/13504717-4242-422a-b7d8-a910a6cc9348/scratchpad/s57_cpp/auditmeta.json','w'))
# 3 census
def form(p,dl):
    L=lines(p)
    if L is None: return 'missing-file','',0
    txt='';depth=0
    for i in range(dl-1,min(len(L),dl+29)):
        s=re.sub(r'//.*','',L[i]); txt+=s+'\n'
        for ch in s:
            if ch=='(':depth+=1
            elif ch==')':depth-=1
            elif depth<=0 and ch=='{': return 'body-hint',txt,i+1
            elif depth<=0 and ch==';': return 'bodyless-hint',txt,i+1
    return 'unclear',txt,0
cens=collections.OrderedDict()
for r in rows:
    for c in r['label']['callees']:
        if c['kind'] in('internal','declared'):
            d=cens.setdefault((c['path'],c['defLine']),{'labelKinds':set(),'qnames':set(),'src':set(),'sites':[]})
            d['labelKinds'].add(c['kind']);d['qnames'].add(c['qname']);d['src'].add('label');d['sites'].append(r['ordinal'])
    ref=refs[r['ordinal']]
    for x in [ref['static']]+ref.get('impls',[]):
        if x.get('kind') in('internal','declared') and 'path' in x:
            d=cens.setdefault((x['path'],x['defLine']),{'labelKinds':set(),'qnames':set(),'src':set(),'sites':[]})
            d['qnames'].add(x.get('qname'));d['src'].add('reference:'+x['kind']);d['sites'].append(r['ordinal'])
for (p,dl),d in sorted(cens.items()):
    f,t,endl=form(p,dl)
    emit({"queueCategory":"census","path":p,"defLine":dl,"labelKinds":sorted(d['labelKinds']),"sources":sorted(d['src']),
          "qnames":sorted(q for q in d['qnames'] if q),"siteOrdinals":sorted(set(d['sites'])),"fcFormHint":f,"fcFormEndLine":endl,
          "excerpt":exc(p,dl,1,6),"labelDeclaredAndBody": ('declared' in d['labelKinds'] and f=='body-hint') ,
          "labelInternalAndBodyless": ('internal' in d['labelKinds'] and f=='bodyless-hint')})
# 4 macros
groups=collections.OrderedDict()
for r in macro_sites: groups.setdefault(macro_name(r),[]).append(r)
def define(m):
    import subprocess
    res=subprocess.run(['grep','-rnE',r'^\s*#\s*define\s+'+re.escape(m)+r'\b','--include=*.h','--include=*.cc','--include=*.inc','.'],cwd=SRC,capture_output=True,text=True).stdout.strip().split('\n')
    res=[x for x in res if x]
    ds=[]
    for x in res[:3]:
        p,ln,_=x.split(':',2); p=p[2:]; ln=int(ln); L=lines(p); k=ln
        while L[k-1].rstrip().endswith('\\') and k<len(L): k+=1
        ds.append({"path":p,"line":ln,"text":"\n".join(f"{i}| {L[i-1]}" for i in range(ln,min(k,ln+12)+1))})
    return ds,len(res)
for m,L in groups.items():
    ds,n=define(m)
    emit({"queueCategory":"macro","macroName":m,"nSites":len(L),"nUnsure":sum(r['label']['confidence']=='unsure' for r in L),
          "defines":ds,"nDefineMatches":n,
          "sites":[{"ordinal":r['ordinal'],"siteId":r['siteId'],"file":r['site']['path'],"line":r['site']['line'],"status":r['status'],"reasons":r['reasons'],
                    "label":r['label'],"reference":r['reference'],"inAdjudicateQueue":r['status']=='needs-adjudication',
                    "excerpt":exc(r['site']['path'],r['site']['line'],3,3)} for r in L]})
print(dict(counts)); print(auditmeta)
print('macro sites',len(macro_sites),'macros',len(groups),'unsure macro sites',sum(r['label']['confidence']=='unsure' for r in macro_sites))
