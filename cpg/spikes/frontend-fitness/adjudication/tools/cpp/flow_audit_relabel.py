#!/usr/bin/env python3
"""F2 blind relabel of the six drawn cpp methods (12,14,19,23,27,29), written from sample/cpp-flow.jsonl source only."""
import json

import pathlib as _pl; _CPG = str(_pl.Path(__file__).resolve().parents[5])
FF = (_CPG+"/spikes/frontend-fitness")
meta = {}
for l in open(f"{FF}/sample/cpp-flow.jsonl"):
    m = json.loads(l); meta[m["ordinal"]] = m
P = lambda i, n: {"point": "param", "index": i, "name": n}
C = lambda n, o, l: {"point": "call", "calleeName": n, "ordinal": o, "line": l}
R = lambda o, l: {"point": "ret", "ordinal": o, "line": l}
K = lambda l, c: {"point": "cond", "line": l, "code": c}
def d(s, t, ev, conf="sure"): return {"kind": "data", "source": s, "sink": t, "confidence": conf, "evidence": ev[:300]}
def c(s, t, ev, conf="sure"): return {"kind": "control", "source": s, "sink": t, "confidence": conf, "evidence": ev[:300]}
out = {}
# 12
f = "gtest-param-util.h"
c1, c2, c3 = K(661, "name.empty()"), K(664, "index < name.size()"), K(665, "!IsAlNum(name[index]) && name[index] != '_'")
e, s, a = C("empty", 0, 661), C("size", 0, 664), C("IsAlNum", 0, 665)
out[12] = [d(P(0, "name"), e, f"{f}:661"), d(P(0, "name"), s, f"{f}:664"), d(P(0, "name"), a, f"{f}:665 name[index]"),
 d(P(0, "name"), c3, f"{f}:665 name[index] != '_'"), d(e, c1, f"{f}:661"), d(s, c2, f"{f}:664"), d(a, c3, f"{f}:665"),
 c(c1, R(0, 661), f"{f}:661"), c(c1, s, f"early return {f}:661, size at 664"), c(c1, c2, f"{f}:661-664"), c(c1, R(2, 668), f"{f}:661,668"),
 c(c2, c3, f"loop body {f}:664-665"), c(c2, a, f"loop body {f}:665; IsAlNum is inside c3 text"), c(c3, R(1, 665), f"{f}:665")]
# 14
f = "gmock-matchers.h"
q1, q2, q3 = K(4033, "matchers_.size() == 0"), K(4038, "i != matchers_.size()"), K(4039, "i > 0")
z0, z1, dt = C("size", 0, 4033), C("size", 1, 4038), C("DescribeTo", 0, 4042)
out[14] = [d(P(0, "os"), dt, f"{f}:4042"), d(z0, q1, f"{f}:4033"), d(z1, q2, f"{f}:4038"),
 c(q1, R(0, 4035), f"{f}:4033-4035"), c(q1, z1, f"early return {f}:4033-4036, size at 4038"), c(q1, q2, f"{f}:4036-4038"),
 c(q2, q3, f"loop body {f}:4038-4039"), c(q2, dt, f"loop body {f}:4042")]
# 19
f = "gtest_pred_impl_unittest.cc"
k = K(856, "PredFunction3(v1, v2, v3)")
pf, sc, af = C("PredFunction3", 0, 856), C("AssertionSuccess", 0, 856), C("AssertionFailure", 0, 858)
out[19] = [d(P(3, "v1"), pf, f"{f}:856"), d(P(4, "v2"), pf, f"{f}:856"), d(P(5, "v3"), pf, f"{f}:856"), d(pf, k, f"{f}:856"),
 d(sc, R(0, 856), f"{f}:856"), d(af, R(1, 858), f"{f}:858-861")] + \
 [d(P(i, n), R(1, 858), f"{n} streamed/added into the AssertionFailure expression, {f}:858-861", "unsure") for i, n in
  [(0, "e1"), (1, "e2"), (2, "e3"), (3, "v1"), (4, "v2"), (5, "v3")]] + \
 [c(k, sc, f"{f}:856"), c(k, R(0, 856), f"{f}:856"), c(k, af, f"early return {f}:856, skips 858"), c(k, R(1, 858), f"early return {f}:856, skips 858")]
# 23
f = "gtest-internal-inl.h"
g0, g1, sz, gen, sw = C("GTEST_CHECK_", 0, 305), C("GTEST_CHECK_", 1, 308), C("size", 0, 304), C("Generate", 0, 318), C("swap", 0, 319)
lc = K(314, "range_width >= 2")
u = "macro treated as a call site"
out[23] = [d(P(3, "v"), sz, f"{f}:304"), d(sz, g0, f"size local used in macro arg {f}:304-305; {u}", "unsure"),
 d(P(1, "begin"), g0, f"{f}:305; {u}", "unsure"), d(P(1, "begin"), g1, f"{f}:308; {u}", "unsure"), d(P(2, "end"), g1, f"{f}:308; {u}", "unsure"),
 d(sz, g1, f"{f}:308; {u}", "unsure"), d(P(1, "begin"), lc, f"range_width=end-begin {f}:314"), d(P(2, "end"), lc, f"range_width=end-begin {f}:314"),
 d(P(0, "random"), gen, f"{f}:318"), d(P(1, "begin"), gen, f"range_width {f}:314,318"), d(P(2, "end"), gen, f"range_width {f}:314,318"),
 d(gen, sw, f"selected {f}:316-319"), d(P(1, "begin"), sw, f"selected/last_in_range {f}:315-320"), d(P(2, "end"), sw, f"last_in_range via range_width {f}:314-320"),
 d(P(3, "v"), sw, f"(*v)[..] {f}:319-320"), c(lc, gen, f"loop body {f}:318"), c(lc, sw, f"loop body {f}:319")]
# 27
f = "gtest-printers.cc"
k = K(303, "c >= 0x80"); pc, pt = C("PrintCharAndCodeTo", 0, 304), C("PrintTo", 0, 306)
out[27] = [d(P(0, "c"), k, f"{f}:303"), d(P(0, "c"), pc, f"{f}:304"), d(P(1, "os"), pc, f"{f}:304"), d(P(0, "c"), pt, f"{f}:306"),
 d(P(1, "os"), pt, f"{f}:306"), c(k, pc, f"{f}:303-304"), c(k, pt, f"else branch {f}:305-306")]
# 29
f = "gmock-matchers.cc"
h185, h186, h187 = K(185, "irhs < graph_->RhsSize()"), K(186, "(*seen)[irhs]"), K(187, "!graph_->HasEdge(ilhs, irhs)")
h200, o200 = K(200, "right_[irhs] == kUnused || TryAugment(right_[irhs], seen)"), K(200, "right_[irhs] == kUnused")
rs, he, ta = C("RhsSize", 0, 185), C("HasEdge", 0, 187), C("TryAugment", 0, 200)
out[29] = [d(P(0, "ilhs"), he, f"{f}:187"), d(P(1, "seen"), ta, f"{f}:200"), d(P(1, "seen"), h186, f"{f}:186"), d(rs, h185, f"{f}:185"),
 d(he, h187, f"{f}:187"), d(ta, h200, f"{f}:200"),
 c(h185, h186, f"loop body {f}:185-186"), c(h186, he, f"continue {f}:186, skips 187"), c(h186, h187, f"continue {f}:186"),
 c(h186, h200, f"continue {f}:186"), c(h186, o200, f"continue {f}:186"), c(h186, ta, f"continue {f}:186"),
 c(h187, h200, f"continue {f}:187"), c(h187, o200, f"continue {f}:187"), c(h187, ta, f"continue {f}:187"),
 c(o200, ta, f"short-circuit {f}:200"), c(h200, R(0, 204), f"{f}:200-204")]
with open(f"{FF}/adjudication/cpp-flow-relabel.jsonl", "w") as w:
    for o in sorted(out):
        m = meta[o]
        w.write(json.dumps({"ordinal": o, "path": m["path"], "qname": m["qname"], "pairs": out[o]}) + "\n")
print({o: len(v) for o, v in sorted(out.items())})
