from build import *
import sys
# census
tg=collections.OrderedDict()
def add(p,d,src,kind,o):
    e=tg.setdefault((p,d),{'labelKinds':set(),'refKinds':set(),'sites':[]})
    e[src].add(kind); e['sites'].append(o)
for r in rows:
    for c in r['label']['callees']:
        if c['kind'] in('internal','declared'): add(c['path'],c['defLine'],'labelKinds',c['kind'],r['ordinal'])
    st=r['reference']['static']
    if st['kind'] in('internal','declared'): add(st['path'],st['defLine'],'refKinds',st['kind'],r['ordinal'])
    for i in r['reference']['impls']: add(i['path'],i['defLine'],'refKinds','impl',r['ordinal'])
KW=re.compile(r'\b(abstract|expect|external)\b')
def form(p,d):
    L=lines(p); txt='';depth=0
    first=L[d-1]
    for i in range(d-1,min(len(L),d+14)):
        for ch_i,ch in enumerate(L[i]):
            if ch in '(<[': depth+=1
            elif ch in ')>]': depth-=1 if ch!='>' or L[i][ch_i-1]!='-' else 0
            elif ch=='{' and depth<=0: return 'brace-body',i+1
            elif ch=='=' and depth<=0:
                nx=L[i][ch_i+1:ch_i+2]; pv=L[i][ch_i-1:ch_i]
                if nx!='=' and pv not in ('!','<','>','=') : return 'expression-body',i+1
        nxt=L[i+1] if i+1<len(L) else ''
        if depth<=0 and re.match(r'\s*(\}|(public|private|protected|internal|override|open|abstract|suspend|inline|fun|val|var|class|interface|object|@)\b)',nxt) : return 'bodyless',i+1
    return 'undetermined',d
n=0
for (p,d),e in tg.items():
    f,endl=form(p,d)
    first=lines(p)[d-1].strip()
    kind='type' if re.search(r'\b(class|interface|object)\b',first) else ('fun' if re.search(r'\bfun\b',first) else 'other')
    plain = f in('brace-body','expression-body') and kind=='fun' and not KW.search(first) and not re.search(r'\binterface\b',first)
    emit({'category':'census','where':f'{p}:{d}','path':p,'defLine':d,'labelKinds':sorted(e['labelKinds']),'refKinds':sorted(e['refKinds']),
      'sites':sorted(set(e['sites']))[:15],'nSites':len(set(e['sites'])),'declLine':first[:200],'fcForm':f,'fcDeclKind':kind,
      'fcPlainlyBody':plain,'needsDS':(not plain) or ('declared' in e['labelKinds']) ,
      'possibleRoleError':('declared' in e['labelKinds'] and plain) or ('internal' in e['labelKinds'] and f=='bodyless'),
      'context':ctx(p,d,2,7)})
    n+=1
print(n, collections.Counter((x['fcForm'],x['fcDeclKind']) for x in map(json.loads,open(Q)) if x['category']=='census'))
