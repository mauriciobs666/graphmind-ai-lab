"""Blind F2 relabel of javascript flow methods 2,5,8,11 from sample/javascript-flow.jsonl + brief only."""
import json
def P(i,n): return {"point":"param","index":i,"name":n}
def C(n,o,l): return {"point":"call","calleeName":n,"ordinal":o,"line":l}
def R(o,l): return {"point":"ret","ordinal":o,"line":l}
def K(l,c): return {"point":"cond","line":l,"code":c}
def d(s,k,e,c="sure"): return {"kind":"data","source":s,"sink":k,"confidence":c,"evidence":e}
def c(s,k,e,c_="sure"): return {"kind":"control","source":s,"sink":k,"confidence":c_,"evidence":e}
out=[]
# 2 formatDate
F="src/i18n/format.ts"
cd=K(26,"value instanceof Date")
out.append({"ordinal":2,"path":F,"qname":F+"::formatDate","pairs":[
 d(P(0,"value"),cd,"value tested by instanceof, format.ts:26"),
 d(P(0,"value"),C("Date",0,26),"new Date(value), format.ts:26"),
 d(P(0,"value"),C("format",0,27),"date = value (true branch) :26, date passed to format :27"),
 d(C("Date",0,26),C("format",0,27),"date = value instanceof Date ? value : new Date(value) :26; format(date) :27"),
 d(P(1,"locale"),C("DateTimeFormat",0,27),"locale arg of new Intl.DateTimeFormat, format.ts:27"),
 d(P(2,"options"),C("DateTimeFormat",0,27),"options arg of new Intl.DateTimeFormat, format.ts:27"),
 d(C("DateTimeFormat",0,27),C("format",0,27),"receiver of .format, format.ts:27"),
 d(C("format",0,27),R(0,27),"return of format(date), format.ts:27"),
 c(cd,C("Date",0,26),"ternary false branch makes new Date(value) execute, format.ts:26")]})
# 5 DeadTurnNotice
F="src/components/message/DeadTurnNotice.tsx"
cd=K(13,"!visible")
out.append({"ordinal":5,"path":F,"qname":F+"::DeadTurnNotice","pairs":[
 d(P(0,"{ visible }"),cd,"destructured prop visible tested, DeadTurnNotice.tsx:13"),
 d(C("t",0,19),R(1,14),"t(...) result is JSX child of the returned <p>, DeadTurnNotice.tsx:14-20","unsure"),
 c(cd,R(0,13),"if (!visible) return null, DeadTurnNotice.tsx:13"),
 c(cd,R(1,14),"early return on :13 skips return at :14"),
 c(cd,C("t",0,19),"early return on :13 skips the t(...) call in the later return expression :19")],
 "unlisted":[]})
# 8 / 11 errorMessageFor
def emf(ordn,F,first,cases,calls,rets,dflt_ret,nrets):
    """cases: list of (line,label,[call idx or None],ret ordinal)"""
    pass
F="src/views/PresenterResetAllControl.tsx"
na=K(23,"!action")
cs=[K(25,"'reread'"),K(29,"'nothingChangedRetry'"),K(33,"'unhandled'"),K(36,"'clearPresenter'")]
t=[C("t",0,28),C("t",1,32),C("t",2,35)]
pairs=[d(P(0,"action"),na,"!action tested, PresenterResetAllControl.tsx:23"),
 d(P(0,"action"),t[2],"action.status argument of t(), :35"),
 d(t[0],R(1,28),"return t(...) :28"),d(t[1],R(2,32),"return t(...) :32"),d(t[2],R(3,35),"return t(...) :35"),
 c(na,R(0,23),"if (!action) return null :23"),
 c(na,cs[0],"early return :23 skips the switch, first case label :25")]
for a,b in zip(cs,cs[1:]): pairs.append(c(a,b,f"case chain: label at :{b['line']} evaluated only if :{a['line']} did not match"))
pairs+= [c(cs[0],t[0],"case body :28"),c(cs[0],R(1,28),"case body :28"),
 c(cs[1],t[1],"case body :32"),c(cs[1],R(2,32),"case body :32"),
 c(cs[2],t[2],"case body :35"),c(cs[2],R(3,35),"case body :35"),
 c(cs[3],R(4,39),"case body return null :39"),
 c(cs[3],R(5,41),"default body (reached when no case matched) :41, nearest preceding case label","unsure")]
out.append({"ordinal":8,"path":F,"qname":F+"::errorMessageFor","pairs":pairs})
F="src/components/sheets/ResetControl.tsx"
na=K(24,"!action")
cs=[K(26,"'reread'"),K(30,"'nothingChangedRetry'"),K(33,"'unscopedAlarm'"),K(36,"'unhandled'"),K(39,"'clearParticipant'")]
t=[C("t",0,29),C("t",1,32),C("t",2,35),C("t",3,38)]
pairs=[d(P(0,"action"),na,"!action tested, ResetControl.tsx:24"),
 d(P(0,"action"),t[3],"action.status argument of t(), :38"),
 d(t[0],R(1,29),"return t(...) :29"),d(t[1],R(2,32),"return t(...) :32"),d(t[2],R(3,35),"return t(...) :35"),d(t[3],R(4,38),"return t(...) :38"),
 c(na,R(0,24),"if (!action) return null :24"),
 c(na,cs[0],"early return :24 skips the switch, first case label :26")]
for a,b in zip(cs,cs[1:]): pairs.append(c(a,b,f"case chain: label at :{b['line']} evaluated only if :{a['line']} did not match"))
pairs+= [c(cs[0],t[0],"case body :29"),c(cs[0],R(1,29),"case body :29"),
 c(cs[1],t[1],"case body :32"),c(cs[1],R(2,32),"case body :32"),
 c(cs[2],t[2],"case body :35"),c(cs[2],R(3,35),"case body :35"),
 c(cs[3],t[3],"case body :38"),c(cs[3],R(4,38),"case body :38"),
 c(cs[4],R(5,42),"case body return null :42"),
 c(cs[4],R(6,44),"default body :44, nearest preceding case label","unsure")]
out.append({"ordinal":11,"path":F,"qname":F+"::errorMessageFor","pairs":pairs})
# fix mismatch: method 8 t ordinals computed above; ordinals by calleeName ascending line
with open("adjudication/javascript-flow-relabel.jsonl","w") as f:
    for m in out: f.write(json.dumps(m)+"\n")
print([ (m["ordinal"],len(m["pairs"])) for m in out])
