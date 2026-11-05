"""DS adjudication of S28 flow misses and spurious samples -> adjudication/<lang>-flow-{misses,spurious}.adj.jsonl (new files)."""
import os, json, re, sys
from pathlib import Path
from collections import Counter
FF = Path(__file__).resolve().parents[3]                      # cpg/spikes/frontend-fitness
CORP = FF.parents[1] / ".ccg-work" / "corpora"                 # cpg/.ccg-work/corpora
S = Path(os.environ.get("FLOW_WORK", "/tmp/ccg-flow-adj"))     # work dir: run/, runfix/, var/ (outside the repo)
TOOLS = Path(__file__).resolve().parent; sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(FF))
from spur_verdicts import V as SPUR
lang = sys.argv[1]
st = json.load(open(S/"var"/f"{lang}.status.json"))["status"]
answers = {f"{a['ordinal']}|{a['idx']}": a for a in map(json.loads, open(FF/"results"/f"projection-{lang}.answers.jsonl"))}
IGN = {"javascript": {"i18n", "test", "tests"}, "java": {"test"}, "cpp": {"test", "tests"}}.get(lang, set())
LOWERED = re.compile(r"iteratorNonEmptyOrException|_iterator_|_result_\d|_idx_ <")
# explicit endpoint table (lang, ordinal, end-text) -> (side, cause); built from reading source + facts (pts.py / probes)
EP = {
 ("csharp", 6, "ret0@25"): ("label", "endpoint-naming-error:ret-line-off-by-one (return true is on line 26; projection ret at 26)"),
 ("csharp", 25, "cond@39"): ("join", "cond-code-text (label keeps the enclosing parentheses; two conds contain-match)"),
 ("javascript", 7, "cond@43"): ("join", "cond-code-text (Joern prints `opts.body.error` for `opts.body?.error`)"),
 ("csharp", 15, "ToString#0@79"): ("frontend", "csharp-switch-expression-arm-not-lowered (no call/cond node for the arm)"),
 ("csharp", 22, "Trim#2@28"): ("frontend", "csharp-null-conditional-call-not-emitted (`from.Value?.Trim()` has no call node)"),
 ("csharp", 26, "NoResult#0@83"): ("frontend", "csharp-coalesce-operand-call-not-emitted (`result ?? X.NoResult()`)"),
 ("csharp", 26, "cond@83"): ("frontend", "no-cond-node:csharp-??-operand"),
 ("csharp", 27, "OcelotJSection#0@513"): ("frontend", "csharp-coalesce-assignment-call-not-emitted (`??=`)"),
}
CS_DROPPED = {1, 5, 10, 11, 12, 16, 19, 25, 27, 28}   # C# methods where a brace-less if/foreach/lock statement body is absent from the CPG (pts.py)
def epkey(e):
    p = e["point"]
    if p == "param": return f"param{e['index']}"
    if p == "call": return f"{e['calleeName']}#{e['ordinal']}@{e['line']}"
    if p == "ret": return f"ret{e['ordinal']}@{e['line']}"
    return f"cond@{e['line']}"
def endpoint_cause(r, end, det, L):
    e = r["pair"][end]; k = epkey(e); ln = e.get("line"); text = L[ln-1].strip() if ln and ln <= len(L) else ""
    if (lang, r["ordinal"], k) in EP: return EP[(lang, r["ordinal"], k)]
    want, reason = det.get("want"), det.get("reason")
    if want == "cond":
        pc = det.get("projectionConds") or []
        if any(c[0] == ln and LOWERED.search(c[1]) for c in pc): return ("join", "lowered-loop-header-cond (the header's CDG node exists on the line with lowered code)")
        if re.match(r"(case\b|default\b)", text) or (lang == "csharp" and "=>" in text and not text.startswith(("if", "return"))):
            return ("frontend", "no-cond-node:switch-case-label (CDG has the discriminant only; Part F ruling 1 makes each label a cond)")
        if lang == "go" and " range " in f" {text} ": return ("frontend", "no-cond-node:go-range-header")
        if lang == "csharp" and r["ordinal"] in CS_DROPPED: return ("frontend", "csharp-embedded-statement-dropped (brace-less if/foreach/lock body absent from the AST)")
    if want == "call":
        if lang == "go" and text.startswith("defer "): return ("frontend", "call-not-emitted:go-defer")
        if e.get("calleeName") == "<dynamic>": return ("frontend", "call-not-emitted:computed-callee (no CALL node on the line)")
        if lang == "csharp" and re.search(r"\bnew\(", text): return ("join", "callee-name-absent-from-source:csharp-target-typed-new (the join treats it as synthetic)")
        if lang == "csharp" and r["ordinal"] in CS_DROPPED: return ("frontend", "csharp-embedded-statement-dropped (brace-less if/foreach/lock body absent from the AST)")
        if lang == "cpp" and "get<" in text and e["calleeName"] == "get": return ("frontend", "call-not-emitted:cpp-explicit-template-call (`::std::get<I>()`; no CALL node)")
        if lang == "cpp": return ("frontend", "call-not-emitted:cpp-direct-initialization-constructor (`T x(args);` has no CALL node)")
    if want == "ret":
        if lang == "csharp" and r["ordinal"] in CS_DROPPED: return ("frontend", "csharp-embedded-statement-dropped (brace-less if/foreach/lock body absent from the AST)")
        if lang == "csharp" and r["ordinal"] == 10: return ("frontend", "csharp-embedded-statement-dropped (brace-less if/foreach/lock body absent from the AST)")
    return ("undetermined", f"endpoint {want} {reason}")
def data_pattern(r, L):
    s, t = r["pair"]["source"], r["pair"]["sink"]
    sl = L[s["line"]-1] if s.get("line") else ""; tl = L[t["line"]-1] if t.get("line") else ""
    if s["point"] == "call" and re.search(r"\bnew\s+" + re.escape(s["calleeName"]) + r"\b", sl): return ("projection", "ctor-lowering (alloc + <init> block; objFields reads the alloc block as an empty object literal, so the new-expression's value has no def edge)")
    if lang == "go" and s["point"] == "call" and re.search(r"^\s*[\w_]+\s*,\s*[\w_, ]+:?=", sl): return ("frontend", "go-multi-value-assignment (`a, err := f()` lowers to no LHS identifiers; probe: 0 IDENTIFIER nodes on the line)")
    if lang == "go" and r["ordinal"] == 1: return ("frontend", "go-range-loop-breaks-reaching-defs (uses after the range loop have no REACHING_DEF in; probe)")
    if lang == "go" and t["point"] == "cond" and re.fullmatch(r"[\w_]+", t["code"].strip()) and s["point"] == "param": return ("frontend", "go-param-use-reaching-def-from-METHOD (identifier not linked to its parameter; probe: rdIn = METHOD)")
    if lang == "cpp" and (t.get("calleeName", "").isupper() or "GTEST_" in tl or "ADD_FAILURE" in tl): return ("frontend", "macro-expansion (data reaches the expanded body, not the macro call point)")
    if lang == "python" and re.search(r"\bfor\b.+\bin\b", tl) and re.search(r"[\[(]", tl) and not tl.strip().startswith("for "): return ("projection", "nested-scope (comprehension/generator is its own METHOD; intraprocedural projection cannot cross it)")
    if lang == "python" and r["ordinal"] == 17: return ("projection", "nested-scope (comprehension/generator is its own METHOD; intraprocedural projection cannot cross it)")
    if re.search(r"\*\*\w|\.\.\.\w|\{[^}]*\b\w+\s*:", tl) or "[..." in tl or "{..." in tl: return ("projection", "container-literal/spread not traversed (value enters a dict/array/object literal or spread argument)")
    if lang == "javascript" and "=>" in tl: return ("projection", "nested-scope (arrow function is its own METHOD)")
    if lang == "csharp" and "foreach" in sl: return ("projection", "foreach-lowering (`_idx_`-indexed re-evaluated collection; loop variable def not traced)")
    return None
def ctl_pattern(r, L):
    s, t = r["pair"]["source"], r["pair"]["sink"]
    after = " ".join(L[s["line"]: s["line"]+3])
    if lang == "csharp" and r["ordinal"] in CS_DROPPED: return ("frontend", "csharp-embedded-statement-dropped (brace-less if/foreach/lock body absent from the AST)")
    if re.search(r"\b(raise|throw)\b", after): return ("frontend", "cfg-throw-falls-through (throw/raise has the next statement as CFG successor, so no CDG edge; probe: java saveUser, python _validate_path)")
    return None
OV = {  # (lang, ordinal, pairIndex) -> (side, cause[, adjudication]) ; pattern-assigned from source unless marked probed
 ("java",1,0):("frontend","no-reaching-def-path:short-circuit-assigned-local (no path even with every REACHING_DEF source followed; wide probe)"),
 ("java",1,1):("frontend","no-reaching-def-path:short-circuit-assigned-local (no path even with every REACHING_DEF source followed; wide probe)"),
 ("python",6,1):("projection","container-literal/spread not traversed (value enters a dict/array/object literal or spread argument)"),
 ("python",15,4):("projection","container-literal/spread not traversed (identifier used as a dict-literal key)"),
 ("python",19,19):("projection","nested-scope (comprehension/generator is its own METHOD; intraprocedural projection cannot cross it)"),
 ("python",23,4):("projection","bound-method-alias (receiver of an aliased member access is not an argument of the later call)"),
 ("go",6,11):("label","pair-not-real:index-as-selector (Intn result is only a subscript key; Part F ruling 7)","overturned"),
 ("cpp",17,0):("label","pair-not-real:index-as-selector (irhs is only a subscript key; Part F ruling 7)","overturned"),
 ("go",1,13):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",14,0):("frontend","go-param-use-reaching-def-from-METHOD (identifier not linked to its parameter; probed on binder.go:243)"),
 ("go",14,1):("frontend","go-param-use-reaching-def-from-METHOD (identifier not linked to its parameter; probed on binder.go:243)"),
 ("go",14,2):("frontend","go-param-use-reaching-def-from-METHOD (identifier not linked to its parameter; probed on binder.go:243)"),
 ("go",16,5):("frontend","cfg-throw-falls-through (Go panic is an ordinary call; the CFG continues past it)"),
 ("go",21,5):("frontend","cfg-throw-falls-through (Go panic is an ordinary call; the CFG continues past it)"),
 ("go",22,4):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",22,5):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",23,11):("projection","field-write-not-tracked-as-def (value written to limiter.Limiter, read back later)"),
 ("go",23,12):("projection","field-write-not-tracked-as-def (value written to limiter.Limiter, read back later)"),
 ("go",23,20):("frontend","go-multi-value-assignment (`a, ok := m[k]` lowers to no LHS identifiers)"),
 ("go",23,21):("frontend","go-multi-value-return (pattern; not probed)"),
 ("go",25,3):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",25,4):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",25,7):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",25,8):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",25,11):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",25,15):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",25,16):("frontend","go-range-loop-breaks-cfg (point inside/after a range loop; range header has no cond node)"),
 ("go",27,6):("frontend","go-multi-value-assignment (`_, err = f()` lowers to no LHS identifiers)"),
 ("go",27,7):("frontend","go-multi-value-assignment (`_, err = f()` lowers to no LHS identifiers)"),
 ("csharp",3,4):("projection","element-write-not-tracked-as-def (tasks[i] = f(); later tasks read)"),
 ("csharp",10,1):("join","cond-resolved-to-wrong-node (only the operand `options == null` exists on line 13; the contains-join picks it)"),
 ("csharp",19,0):("frontend","csharp-bare-identifier-cond-is-parameter-node (the CDG source carries code `bool disposing`; no use to trace)"),
 ("csharp",26,7):("frontend","csharp-coalesce-operand (`return result ?? ...`; pattern, not probed)"),
 ("csharp",27,5):("frontend","csharp-null-conditional-call (`route?.SelectToken(...)` carries code `route`; pattern, not probed)"),
 ("javascript",5,1):("frontend","jsx-lowering (call inside a JSX child of the returned element; pattern, not probed)"),
}
for i in (12, 17, 18, 20): OV[("python", 9, i)] = ("join", "cond-resolved-to-wrong-node (label `not (self._open or ...)`; Joern prints `not self._open or ...`, so the contains-join picks the operand `self._open`)")
missesOut = []
for r in map(json.loads, open(FF/"adjudication"/f"{lang}-flow-misses.jsonl")):
    k = f"{r['ordinal']}|{r['pairIndex']}"; s = st[k]
    L = (CORP/lang/"src"/r["path"]).read_text(errors="replace").split("\n")
    side, cause, adj = None, None, "miss-stands"
    if r["class"] == "method-absent":
        segs = set(r["path"].split("/")[:-1])
        rsn = answers[k].get("reason")
        if segs & IGN: side, cause = "frontend", "method-absent:frontend-default-ignored-dir"
        elif rsn == "file-absent-from-cpg": side, cause = "frontend", "method-absent:file-not-in-cpg (per-file parse failure, report §a1.4)"
        else: side, cause = "frontend", "method-absent:no-METHOD-in-parsed-file (preprocessor-conditional or parse gap; not individually probed)"
    elif r["class"] == "endpoint-unresolved":
        parts = []
        for end in ("source", "sink"):
            det = r["detail"]["endpointDetail"][end]
            if "reason" in det: parts.append((end,) + endpoint_cause(r, end, det, L))
        sides = {p[1] for p in parts}
        side = "label" if "label" in sides else ("frontend" if "frontend" in sides else ("join" if "join" in sides else "undetermined"))
        cause = " + ".join(f"{p[0]}: {p[2]}" for p in parts)
        if side == "label": adj = "endpoint-naming-error"
    else:
        if r["kind"] == "data":
            if s["R1"] == "connected": side, cause = "projection", "defsOf-ignores-non-assignment-reaching-def (H1: REACHING_DEF runs from an earlier use; relaxed defsOf connects it)"
            elif s["R2"] == "connected": side, cause = "projection", "ctor-lowering (value defined by alloc, receiver of <init>; ctor-receiver rule connects it)"
            elif s["R3"]: side, cause = "projection", "synthetic-lowering-call-opaque (path runs through a frontend-synthetic call, e.g. iterator next/tmp; transparent synthetic calls connect it)"
            else:
                pc = data_pattern(r, L)
                side, cause = pc if pc else ("undetermined", "no DDG path found under R1-R3; not probed")
        else:
            if s["ctlHops"]: side, cause = "definition", f"cdg-indirect ({s['ctlHops']}-step CONTROLLED_BY chain; §7.3 requires a direct edge)"
            elif s.get("sinkIsCond"): side, cause = "projection", "call-point-subsumed-by-cond-point (the call node is itself a CDG source; pointOf maps the CDG target to the cond, never the call)"
            else:
                pc = ctl_pattern(r, L)
                side, cause = pc if pc else ("undetermined", "no CONTROLLED_BY path; not probed")
    ov = OV.get((lang, r["ordinal"], r["pairIndex"]))
    if ov:
        side, cause = ov[0], ov[1]
        if len(ov) > 2: adj = ov[2]
    o = dict(r); o["proposedCause"] = f"{side}:{cause}"; o["causeSide"] = side; o["adjudication"] = adj
    o["counterfactual"] = {"R1": s["R1"] == "connected", "R2": s["R2"] == "connected", "R3": bool(s["R3"])}
    o["adjudicatedBy"] = "data-scientist S28"
    missesOut.append(o)
with open(FF/"adjudication"/f"{lang}-flow-misses.adj.jsonl", "w") as fh:
    for o in missesOut: fh.write(json.dumps(o, ensure_ascii=False, sort_keys=True) + "\n")
spurOut = []
for r in map(json.loads, open(FF/"adjudication"/f"{lang}-flow-spurious.jsonl")):
    v, why = SPUR[lang][r["rank"]]
    o = dict(r); o["verdict"] = v; o["verdictReason"] = why; o["adjudicatedBy"] = "data-scientist S28"
    spurOut.append(o)
with open(FF/"adjudication"/f"{lang}-flow-spurious.adj.jsonl", "w") as fh:
    for o in spurOut: fh.write(json.dumps(o, ensure_ascii=False, sort_keys=True) + "\n")
print(lang, "misses", len(missesOut), Counter(o["causeSide"] for o in missesOut))
for (sd, c), n in Counter((o["causeSide"], o["proposedCause"].split(" (")[0]) for o in missesOut).most_common(): print(f"   {n:4d} {c}")
print("  spurious", Counter(o["verdict"] for o in spurOut))
