import json,re,sys,os
import os, pathlib as _pl, tempfile as _tf; _CPG = str(_pl.Path(__file__).resolve().parents[5])
S=(_CPG+'/.ccg-work/corpora/javascript/src/')
OUT=(_CPG+'/spikes/frontend-fitness/labels/javascript.jsonl')
SAMPLE=[json.loads(l) for l in open((_CPG+'/spikes/frontend-fitness/sample/javascript.jsonl'))]
src_cache={}
def lines(p):
    if p not in src_cache: src_cache[p]=open(S+p).read().split('\n')
    return src_cache[p]
JESTDOM={'toBeInTheDocument','toBeVisible','toHaveTextContent','toBeDisabled','toBeEnabled','toHaveAttribute','toHaveValue','toHaveFocus','toHaveClass','toBeChecked','toHaveAccessibleName','toHaveDisplayValue','toContainElement','toBeEmptyDOMElement','toHaveStyle','toHaveAccessibleDescription','toBeRequired','toBeInvalid','toBeValid'}
def ext(pkg): return {'kind':'external','package':pkg}
def find(p,pat,start=0):
    L=lines(p)
    for i,l in enumerate(L):
        if i>=start and re.search(pat,l): return i+1
    raise Exception('nf '+p+pat)
def I(p,chain,pat,kind='internal',start=0):
    return {'kind':kind,'path':p,'qname':p+'::'+chain,'defLine':find(p,pat,start)}
def auto(d):
    p=d['path']; ct=d['callText']; e2e=p.startswith('tests/e2e')
    m=re.match(r'^expect\b',ct)
    if m:
        if e2e or ct.startswith('expect.poll'): return ext('@playwright/test')
        mm=re.search(r'\.(\w+)$',ct)
        if ct=='expect' or not mm: return ext('vitest')
        return ext('@testing-library/jest-dom' if mm.group(1) in JESTDOM else 'vitest')
    if e2e and re.match(r'^(page|test)\b',ct): return ext('@playwright/test')
    if re.match(r'^(vi\.|it$|it\.|describe|afterEach|beforeEach|beforeAll|afterAll)',ct): return ext('vitest')
    if re.match(r'^(screen\.|render$|rerender$|waitFor$|act$|within$|cleanup$)',ct): return ext('@testing-library/react')
    if re.match(r'^user\.',ct): return ext('@testing-library/user-event')
    if re.match(r'^(String$|JSON\.|Object\.|new (Date|Error|Promise|Set|Map|URLSearchParams|AbortController|Intl\.)|setTimeout|clearTimeout|Promise\.|Array\.|Number|requestAnimationFrame)',ct): return ext('builtin')
    if re.match(r'^(useState|useCallback|useMemo|useEffect|createContext|useContext|useRef|useLayoutEffect)$',ct): return ext('react')
    if ct in('useTranslation',): return ext('react-i18next')
    if ct in('useLocation','useNavigate','MemoryRouter','createBrowserRouter','Navigate','useParams','Outlet','RouterProvider','Link'): return ext('react-router-dom')
    return None
OVR={}
def put(o,callees,conf='sure',ev='',na=False,dyn=False):
    OVR[o]=dict(callees=callees,notACall=na,dynamic=dyn,confidence=conf,evidence=ev)
def write(a,b):
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT): done.add(json.loads(l)['ordinal'])
    last=max(done) if done else 0
    assert a==last+1,(a,last)
    out=[]; miss=[]
    for d in SAMPLE:
        o=d['ordinal']
        if not(a<=o<=b): continue
        if o in OVR: r=OVR[o]
        else:
            x=auto(d)
            if x is None: miss.append(o); continue
            r=dict(callees=[x],notACall=False,dynamic=False,confidence='sure',evidence='external library/builtin/global (auto) '+d['path']+':'+str(d['line']))
        out.append(dict(ordinal=o,siteId=d['siteId'],callees=r['callees'],notACall=r['notACall'],dynamic=r['dynamic'],confidence=r['confidence'],evidence=r['evidence'][:300]))
    if miss: print('MISSING',miss); return
    with open(OUT,'a') as f:
        for r in out: f.write(json.dumps(r)+'\n')
    n=sum(1 for r in out if any(c['kind'] in('internal','declared') for c in r['callees']))
    print('wrote',len(out),'internal-ish',n)
