"""Blind relabel of the 6 F2 methods (from sample/python-flow.jsonl + brief only). Hand-authored pairs; this script only formats."""
import json
samp = {}
for l in open("../sample/python-flow.jsonl"):
    m = json.loads(l); samp[m["ordinal"]] = m
P = lambda i, n: {"point": "param", "index": i, "name": n}
C = lambda n, o, l: {"point": "call", "calleeName": n, "ordinal": o, "line": l}
R = lambda o, l: {"point": "ret", "ordinal": o, "line": l}
K = lambda l, c: {"point": "cond", "line": l, "code": c}
def d(s, t, ev, conf="sure"): return {"kind": "data", "source": s, "sink": t, "confidence": conf, "evidence": ev}
def c(s, t, ev, conf="sure"): return {"kind": "control", "source": s, "sink": t, "confidence": conf, "evidence": ev}
M = {}
# --- 3: _warn_once
f = "modelconfig.py"
k847 = K(847, "key in self._warned")
M[3] = ([
 d(P(0,"kind"), k847, "key=(kind,ref) 846; tested 847"), d(P(1,"ref"), k847, "key=(kind,ref) 846; tested 847"),
 d(P(-1,"self"), k847, "self._warned read in cond, 847"),
 d(P(0,"kind"), C("add",0,849), "key passed to add, 849"), d(P(1,"ref"), C("add",0,849), "key passed to add, 849"),
 d(P(-1,"self"), C("add",0,849), "receiver self._warned.add, 849"),
 d(P(0,"kind"), C("warning",0,850), "args of _log.warning 850-854"), d(P(1,"ref"), C("warning",0,850), "args of _log.warning 850-854"),
 c(k847, R(0,848), "if key in self._warned: return, 847-848"),
 c(k847, C("add",0,849), "early return 848 skips add 849"), c(k847, C("warning",0,850), "early return 848 skips warning 850"),
], [])
# --- 10: _scalar_equal
c1 = K(116, "isinstance(expected, bool) or isinstance(actual, bool)"); c2 = K(116, "isinstance(expected, bool)")
c3 = K(118, "isinstance(expected, (int, float)) and isinstance(actual, (int, float))"); c4 = K(118, "isinstance(expected, (int, float))")
c5 = K(124, "isinstance(expected, (int, float)) or isinstance(actual, (int, float))"); c6 = K(124, "isinstance(expected, (int, float))")
I = lambda o, l: C("isinstance", o, l)
M[10] = ([
 d(P(0,"expected"), I(0,116), "isinstance(expected,bool) 116"), d(P(1,"actual"), I(1,116), "isinstance(actual,bool) 116"),
 d(P(0,"expected"), I(2,118), "118"), d(P(1,"actual"), I(3,118), "118"),
 d(P(0,"expected"), I(4,124), "124"), d(P(1,"actual"), I(5,124), "124"),
 d(P(0,"expected"), C("float",0,123), "float(expected) 123"), d(P(1,"actual"), C("float",1,123), "float(actual) 123"),
 d(P(0,"expected"), C("_canon_str",0,126), "126"), d(P(1,"actual"), C("_canon_str",1,126), "126"),
 d(P(0,"expected"), R(0,117), "return expected is actual 117"), d(P(1,"actual"), R(0,117), "return expected is actual 117"),
 d(I(0,116), c2, "result is the operand cond 116"), d(I(1,116), c1, "result is right operand of header 116"),
 d(I(2,118), c4, "operand 118"), d(I(3,118), c3, "right operand of header 118"),
 d(I(4,124), c6, "operand 124"), d(I(5,124), c5, "right operand of header 124"),
 d(C("float",0,123), C("abs",0,123), "abs(float(expected)-float(actual)) 123"), d(C("float",1,123), C("abs",0,123), "123"),
 d(C("abs",0,123), R(1,123), "returned via comparison, 123"),
 d(C("_canon_str",0,126), R(3,126), "== comparison returned 126"), d(C("_canon_str",1,126), R(3,126), "126"),
 c(c2, I(1,116), "a or isinstance(actual..): operand guards call 116"),
 c(c1, R(0,117), "if ...: return 116-117"),
 c(c1, c3, "after early return 117, next if header 118"), c(c1, c4, "operand of 118 header evaluated after 117 exit"), c(c1, I(2,118), "first call of 118 header after 117 exit"),
 c(c4, I(3,118), "a and isinstance(actual..) guards call 118"),
 c(c3, R(1,123), "if 118: return 123"), c(c3, C("abs",0,123), "body 123"), c(c3, C("float",0,123), "body 123"), c(c3, C("float",1,123), "body 123"),
 c(c3, c5, "after early return 123, next if 124"), c(c3, c6, "operand of 124 header after 123 exit"), c(c3, I(4,124), "first call of 124 header after 123 exit"),
 c(c6, I(5,124), "a or isinstance(actual..) guards call 124"),
 c(c5, R(2,125), "if 124: return False 125"), c(c5, C("_canon_str",0,126), "after early return 125"), c(c5, C("_canon_str",1,126), "after early return 125"), c(c5, R(3,126), "after early return 125, 126"),
], [])
# --- 16: order_belongs_to_customer
k760 = K(760, 'order is None or order["customerId"] != customer_id')
M[16] = ([
 d(P(-1,"self"), C("append",0,758), "self.calls.append 758"), d(P(1,"customer_id"), C("append",0,758), "tuple arg 758"), d(P(2,"order_id"), C("append",0,758), "tuple arg 758"),
 d(P(-1,"self"), C("get",0,759), "self.orders.get 759"), d(P(2,"order_id"), C("get",0,759), "arg 759"),
 d(C("get",0,759), k760, "order = ... get 759; tested 760"), d(P(1,"customer_id"), k760, "compared in cond 760"),
 d(C("get",0,759), R(1,762), 'order["status"] in returned dict 762'),
 c(k760, R(0,761), "if ...: return 760-761"), c(k760, R(1,762), "return 762 skipped by early return 761"),
], [])
# --- 21
k724 = K(724, "i in range(3)")
M[21] = ([
 d(C("range",0,724), k724, "for i in range(3) header 724"), d(C("range",0,724), C("_seed_row",0,725), "loop var i used in f-strings 725-726"),
 d(C("_old_embedding",0,726), C("_seed_row",0,725), "embedding= arg 726"), d(P(1,"conn"), C("_seed_row",0,725), "arg 725"),
 d(P(0,"migrate_repo"), C("_count_unmigrated",0,727), "arg 727"), d(P(0,"migrate_repo"), C("_write_embedding",0,732), "arg 732"),
 d(P(0,"migrate_repo"), C("_count_unmigrated",1,736), "arg 736"),
 c(k724, C("_seed_row",0,725), "loop body 724-726"), c(k724, C("_old_embedding",0,726), "loop body 726"),
], [{"line": 730, "construct": "assert"}, {"line": 739, "construct": "assert"}])
# --- 25
k206 = K(206, 'row["shape"] != "conflicting-facts"')
M[25] = ([
 d(P(0,"row"), k206, 'row["shape"] tested 206'), d(P(0,"row"), C("len",0,213), 'expected = row["expected"] 208; len(expected["values"]) 213'),
 c(k206, R(0,207), "if ...: return 206-207"), c(k206, C("len",0,213), "len at 213 skipped by early return 207"),
], [{"line": 209, "construct": "assert"}, {"line": 213, "construct": "assert"}])
# --- 30
k1052 = K(1052, "categories")
M[30] = ([
 d(P(0,"schema"), C("_describe_dataset_schema",0,1051), "arg 1051"), d(C("_describe_dataset_schema",0,1051), C("format",0,1051), "kwarg dataset_schema 1051"),
 d(C("format",0,1051), R(0,1056), "prompt = format(...) 1051; returned 1056"), d(C("format",1,1053), R(0,1056), "prompt += ...format(...) 1053; returned 1056"),
 d(P(1,"categories"), k1052, "if categories 1052"), d(P(1,"categories"), C("join",0,1054), "join(categories) 1054"),
 d(C("join",0,1054), C("format",1,1053), "kwarg categories= 1053-1054"),
 c(k1052, C("format",1,1053), "if body 1052-1055"), c(k1052, C("join",0,1054), "if body 1054"),
], [])
with open("python-flow-relabel.jsonl", "w") as o:
    for ordv in sorted(M):
        s = samp[ordv]; pairs, unl = M[ordv]
        rec = {"ordinal": ordv, "path": s["path"], "qname": s["qname"], "pairs": pairs}
        if unl: rec["unlisted"] = unl
        o.write(json.dumps(rec) + "\n")
        print(ordv, len(pairs))
