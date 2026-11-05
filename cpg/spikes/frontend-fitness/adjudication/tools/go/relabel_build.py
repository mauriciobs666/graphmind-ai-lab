#!/usr/bin/env python3
"""Builds adjudication/go-flow-relabel.jsonl from hand-written pair specs (blind relabel of the 6 F2 methods).
Specs written by the FC from sample/go-flow.jsonl source + flow brief only."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[2]
P = lambda i, n: {"point": "param", "index": i, "name": n}
C = lambda n, o, l: {"point": "call", "calleeName": n, "ordinal": o, "line": l}
T = lambda o, l: {"point": "ret", "ordinal": o, "line": l}
K = lambda l, c: {"point": "cond", "line": l, "code": c}
def d(s, t, ev, cf="sure"): return {"kind": "data", "source": s, "sink": t, "confidence": cf, "evidence": ev}
def c(s, t, ev, cf="sure"): return {"kind": "control", "source": s, "sink": t, "confidence": cf, "evidence": ev}
M = []

# ---- 9 MustTextUnmarshaler binder.go 403-418
b = P(-1, "b"); sp = P(0, "sourceParam"); dst = P(1, "dest")
vf = C("ValueFunc", 0, 408); se0 = C("setError", 0, 410); ef0 = C("ErrorFunc", 0, 410)
ut = C("UnmarshalText", 0, 414); se1 = C("setError", 1, 415); ef1 = C("ErrorFunc", 1, 415)
r0, r1, r2 = T(0, 405), T(1, 411), T(2, 417)
c404 = K(404, "b.failFast && b.errors != nil"); c409 = K(409, 'tmp == ""'); c414 = K(414, "err != nil")
M.append((9, "binder.go", "binder.go::ValueBinder.MustTextUnmarshaler", [
 d(b, c404, "b.failFast/b.errors read binder.go:404"), d(b, r0, "return b :405"), d(b, vf, "b.ValueFunc receiver :408"),
 d(b, ef0, "b.ErrorFunc receiver :410"), d(b, se0, "b.setError receiver :410"), d(b, r1, "return b :411"),
 d(b, ef1, "b.ErrorFunc receiver :415"), d(b, se1, "b.setError receiver :415"), d(b, r2, "return b :417"),
 d(sp, vf, "sourceParam arg of ValueFunc :408"), d(sp, ef0, "sourceParam arg of ErrorFunc :410"), d(sp, ef1, "sourceParam arg of ErrorFunc :415"),
 d(dst, ut, "dest.UnmarshalText receiver :414"),
 d(vf, c409, "tmp := ValueFunc :408; tmp == \"\" :409"), d(vf, ef0, "tmp in []string{tmp} :410"), d(vf, ut, "[]byte(tmp) arg :414"), d(vf, ef1, "tmp in []string{tmp} :415"),
 d(ef0, se0, "setError(ErrorFunc(...)) :410"), d(ef1, se1, "setError(ErrorFunc(...)) :415"),
 d(ut, c414, "err := dest.UnmarshalText; err != nil :414"), d(ut, ef1, "err passed as last arg of ErrorFunc :415"),
 c(c404, r0, "if guard return b :404-405"), c(c404, vf, "early return skips :408"), c(c404, c409, "early return skips :409"),
 c(c404, c414, "early return skips :414"), c(c404, ut, "early return skips :414 init"), c(c404, r2, "early return skips :417"),
 c(c409, se0, "if tmp==\"\" body :409-410"), c(c409, ef0, "if body :410"), c(c409, r1, "if body return :411"),
 c(c409, c414, "early return skips :414"), c(c409, ut, "early return skips :414 init"), c(c409, r2, "early return skips :417"),
 c(c414, se1, "if err != nil body :414-415"), c(c414, ef1, "if body :415"),
], []))

# ---- 16 toMiddlewareOrPanic middleware/middleware.go 91-97
cf = P(0, "config"); tm = C("ToMiddleware", 0, 92); pn = C("panic", 0, 94); rr = T(0, 96); cc = K(93, "err != nil")
M.append((16, "middleware/middleware.go", "middleware/middleware.go::toMiddlewareOrPanic", [
 d(cf, tm, "config.ToMiddleware() receiver :92"), d(tm, cc, "mw, err := ...; err != nil :92-93"), d(tm, pn, "panic(err) :94"), d(tm, rr, "return mw :96"),
 c(cc, pn, "if err != nil { panic(err) } :93-94"), c(cc, rr, "panic exits, skipping return :93-96", "unsure"),
], []))

# ---- 17 BenchmarkBind_Query perf_bench_test.go 187-201 (no returns)
bb = P(0, "b"); nw = C("New", 0, 188); gt = C("GET", 0, 189); ra = C("ReportAllocs", 0, 193); rt = C("ResetTimer", 0, 194)
nr = C("NewRequest", 0, 196); sh = C("ServeHTTP", 0, 199); lc = K(197, "i < b.N")
M.append((17, "perf_bench_test.go", "perf_bench_test.go::BenchmarkBind_Query", [
 d(bb, ra, "b.ReportAllocs() :193"), d(bb, rt, "b.ResetTimer() :194"), d(bb, lc, "b.N read in loop condition :197"),
 d(nw, gt, "e := New(); e.GET :188-189"), d(nw, sh, "e.ServeHTTP :199"), d(nr, sh, "req arg :196,199"),
 c(lc, sh, "for i < b.N body :197-199"),
], []))

# ---- 22 isForbiddenOverrideMethod middleware/method_override.go 83-90
m = P(0, "m"); ef = C("EqualFold", 0, 85); rg = K(84, "_, forbidden := range []string{http.MethodGet, http.MethodHead, http.MethodOptions, http.MethodTrace, http.MethodConnect}")
ci = K(85, "strings.EqualFold(m, forbidden)")
M.append((22, "middleware/method_override.go", "middleware/method_override.go::isForbiddenOverrideMethod", [
 d(m, ef, "m is arg of strings.EqualFold :85"), d(ef, ci, "call result is the if condition :85"),
 c(rg, ef, "range body evaluates EqualFold :84-85"), c(rg, ci, "range body if :84-85"),
 c(ci, T(0, 86), "if EqualFold { return true } :85-86"),
 c(ci, T(1, 89), "return true inside loop skips final return :86,89", "unsure"),
], []))

# ---- 26 BenchmarkValueBinder_BindInt64_single binder_test.go 2926-2939
bb = P(0, "b"); ct = C("createTestContext", 0, 2927); qp = C("QueryParamsBinder", 0, 2934)
i64 = C("Int64", 0, 2937); be = C("BindError", 0, 2937); ra = C("ReportAllocs", 0, 2929); rt = C("ResetTimer", 0, 2930); lc = K(2935, "i < b.N")
M.append((26, "binder_test.go", "binder_test.go::BenchmarkValueBinder_BindInt64_single", [
 d(bb, ra, "b.ReportAllocs() :2929"), d(bb, rt, "b.ResetTimer() :2930"), d(bb, lc, "b.N loop bound :2935"),
 d(ct, qp, "c := createTestContext; QueryParamsBinder(c) :2927,2934"), d(qp, i64, "binder.Int64 receiver :2934,2937"),
 d(i64, be, "Int64(...).BindError() chained receiver :2937"),
 c(lc, i64, "for loop body :2935-2937"), c(lc, be, "for loop body :2935-2937"),
], []))

# ---- 30 isTrustedPeer scheme.go 83-98
ra_ = P(0, "remoteAddr"); ck = P(1, "checker")
sh_ = C("SplitHostPort", 0, 89); ib = C("IndexByte", 0, 93); pi = C("ParseIP", 0, 96); tr = C("trust", 0, 97)
c86 = K(86, "remoteAddr == \"\" || remoteAddr[0] == '@' || remoteAddr[0] == '/'"); c90 = K(90, "err != nil")
c93 = K(93, "i != -1"); c97 = K(97, "ip != nil"); r0, r1 = T(0, 87), T(1, 97)
M.append((30, "scheme.go", "scheme.go::isTrustedPeer", [
 d(ra_, c86, "remoteAddr tested :86"), d(ra_, sh_, "SplitHostPort(remoteAddr) :89"),
 d(ra_, ib, "host = remoteAddr :91 then IndexByte(host) :93"), d(ra_, pi, "host = remoteAddr :91 reaches ParseIP(host) :96"),
 d(ck, r0, "return checker.trustLoopback :87"), d(ck, tr, "checker.trust receiver :97"),
 d(sh_, c90, "host, _, err := SplitHostPort; err != nil :89-90"), d(sh_, ib, "host from SplitHostPort :89,93"), d(sh_, pi, "host from SplitHostPort :89,96"),
 d(ib, c93, "i := IndexByte; i != -1 :93"), d(ib, pi, "host = host[:i] :94 then ParseIP(host) :96"),
 d(pi, c97, "ip := ParseIP; ip != nil :96-97"), d(pi, tr, "checker.trust(ip) :97"), d(pi, r1, "ip != nil is part of returned expression :97", "unsure"),
 d(tr, r1, "return ... && checker.trust(ip) :97"),
 c(c86, r0, "if ... { return } :86-87"), c(c86, sh_, "early return skips :89"), c(c86, c90, "early return skips :90"),
 c(c86, ib, "early return skips :93"), c(c86, c93, "early return skips :93"), c(c86, pi, "early return skips :96"),
 c(c86, c97, "early return skips :97"), c(c86, r1, "early return skips :97"),
 c(c97, tr, "ip != nil && checker.trust(ip) short-circuit :97"),
], []))

with open(R/"go-flow-relabel.jsonl", "w") as f:
    for o, p, q, pairs, unl in sorted(M, key=lambda x: x[0]):
        f.write(json.dumps({"ordinal": o, "path": p, "qname": q, "pairs": pairs, "unlisted": unl}) + "\n")
print([(o, len(pr)) for o, _, _, pr, _ in sorted(M, key=lambda x: x[0])])
