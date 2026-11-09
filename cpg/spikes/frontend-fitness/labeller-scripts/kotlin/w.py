import json,pickle,os
import os, pathlib as _pl, tempfile as _tf; _SCRATCH = os.environ.get('CCG_SCRATCH') or _tf.gettempdir()
S=pickle.load(open((_SCRATCH+'/S.pkl'),'rb'))

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[4])
OUT=(_CPG+'/spikes/frontend-fitness/labels/kotlin.jsonl')
KP='KtorAdminLibrary/src/main/kotlin/ir/amirroid/ktoradmin/'
TP='KtorAdminLibrary/src/test/kotlin/ir/amirroid/ktoradmin/'
SP='sample/src/main/kotlin/ir/amirreza/'
FP='templates/fluent/src/main/kotlin/ir/amirroid/ktoradmin/templates/fluent/'
def I(path,chain,line,kind='internal'):
    for a,b in (('K:',KP),('T:',TP),('S:',SP),('F:',FP)):
        if path.startswith(a): path=b+path[2:]
    return {"kind":kind,"path":path,"qname":path+"::"+chain,"defLine":line}
def D(path,chain,line): return I(path,chain,line,'declared')
def E(pkg): return {"kind":"external","package":pkg}
def count():
    n=0;last=0
    if os.path.exists(OUT):
        for l in open(OUT):
            d=json.loads(l);last=d['ordinal']
            if any(c['kind']!='external' for c in d['callees']): n+=1
    return last,n
def L(o,callees,ev,conf='sure',notACall=False,dynamic=False):
    last,n=count()
    assert o==last+1,(o,last)
    s=S[o-1]
    rec={"ordinal":o,"siteId":s['siteId'],"callees":callees,"notACall":notACall,"dynamic":dynamic,"confidence":conf,"evidence":ev[:300]}
    with open(OUT,'a') as f: f.write(json.dumps(rec)+'\n')
SRC=(_CPG+'/.ccg-work/corpora/kotlin/src/')
import re
def _ln(path,pat):
    out=[]
    for i,l in enumerate(open(SRC+path),1):
        if re.search(pat,l): out.append(i)
    return out
def panel(m,recv='AdminPanel'):
    decl={'AdminPanel':KP+'panels/AdminPanel.kt','AdminJdbcTable':KP+'panels/AdminJdbcTable.kt','AdminMongoCollection':KP+'panels/AdminMongoCollection.kt'}
    res=[];iface=None
    for k,p in decl.items():
        if _ln(p,r'^\s*fun '+m+r'\('): iface=k;break
    p=decl[iface]; ln=_ln(p,r'^\s*fun '+m+r'\(')[0]
    res.append(D(p,iface+'.'+m,ln))
    def add(path,chain):
        for l in _ln(path,r'override fun '+m+r'\('): res.append(I(path,chain+'.'+m,l))
    jd = recv in('AdminJdbcTable','AdminPanel'); mg= recv in('AdminMongoCollection','AdminPanel')
    tf=KP.replace('main','test')+'TestFixtures.kt'
    for l in _ln(tf,r'override fun '+m+r'\('):
        if l<60 and jd: res.append(I(tf,'TestJdbcTable.'+m,l))
        elif l>=60 and recv=='AdminPanel': res.append(I(tf,'adminPanel.<anon>#1.'+m,l))
    if mg:
        add(KP.replace('main','test')+'repository/MongoClientRepositoryTest.kt','MongoClientRepositoryTest.mongoPanel.<anon>#1')
        add(KP.replace('main','test')+'filters/MongoFiltersTest.kt','MongoFiltersTest.mongoPanel.<anon>#1')
    if jd: add(SP+'panels/CustomManualPanel.kt','CustomManualPanel')
    return res
def DEF(path,chain,regex,nth=0,kind='internal'):
    p=path
    for a,b in (('K:',KP),('T:',TP),('S:',SP),('F:',FP)):
        if path.startswith(a): p=b+path[2:]
    ls=_ln(p,regex)
    return I(path,chain,ls[nth],kind)
