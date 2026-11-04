# S56 flow-brief generator — patch to `run_fitness.py`

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** S56 (P0-47a…g), S28 (P0-21)

For `graph-dba` to apply (the data-scientist does not edit code). The template it reads is
`cpg/spikes/frontend-fitness/flow-brief.template.md` (the text between its two marker lines is the
brief; its preamble states what S28 does with the labels). Apply from the repo root:

```
patch -p1 --dry-run < <(sed -n '/^```diff$/,/^```$/{//!p}' cpg/spikes/frontend-fitness/flow-brief.patch.md)
```

(or copy the diff block below into a file and run `git apply --check` / `git apply`). The diff is
against `run_fitness.py` as of commit `ef6e70b8`; `git diff` on that file was empty when it was cut.

**What it adds** (all Joern-independent, deterministic, stdlib only):

- `run_fitness.py flowbrief <lang>` → `sample/<lang>-flow.brief.md`: template body + per-language slots
  (`FLOW_OPS`, `FLOW_OP_EXAMPLES`, `FLOW_COND_FORMS`, `FLOW_SCOPE_NOTES`, `FLOW_RECEIVER_NOTE`; paths from
  `flow_path`/`src_dir`; `{N_METHODS}` = lines of `sample/<lang>-flow.jsonl`) + a verbatim excerpt of
  bundle spec3 §5.4, §5.5, §5.6 (naming rows and the two "which calls" bullets) and §5.7, with two
  redactions that remove the sentences naming the comparison tool (a redaction that stops matching
  raises). Unfilled slot, missing section, missing per-language operator list, missing TS/JS
  short-circuit instruction, any leak pattern (`results/`, `truth/`, `projection`, `CDG`,
  `CONTROLLED_BY`, thresholds, stage ids …), or other than one mention of Joern → exit with the reason.
- `run_fitness.py flowcheck <lang>` → JSON; exit 1 on any error. Opens only `sample/<lang>-flow.jsonl`
  and `labels/<lang>-flow.jsonl`. Checks the schema of the template's **Output** section (ordinal order from
  1 with no skip, `path`/`qname` equal the sample's, endpoint key sets, kind/endpoint compatibility,
  lines inside `lineStart`–`lineEnd`, `calleeName` present in the method text with `ordinal` below its
  occurrence count, cond `code` a substring of the method's whitespace-collapsed text, evidence 1–300
  characters, source ≠ sink, no duplicate pair, `noPairReason` when `pairs` is empty). Reports
  counts, `complete`, `belowFloor120` (advisory; ML §8.4's floor is applied after S28's adjudication).
  This is the machine part of S56's done-when; `data-scientist` still reads the evidence.
- `verify` also re-derives and validates any existing `sample/<lang>-flow.brief.md`.

**Order of use:** S25's `flowdraw` (exists) → `flowbrief <lang>` → gate (data-scientist reads the
filled brief) → dispatch S56 → `flowcheck <lang>` after each resume and at the end → `teco`'s transcript grep.
Also add to `cpg/spikes/frontend-fitness/README.md`: the two commands, `flow-brief.template.md` and
`sample/<lang>-flow.brief.md` (S56 labeller) in the file table, and `labels/<lang>-flow.jsonl`.

**Tested** on a copy of the tree (not the real one): all seven briefs generate, validate, and are
byte-identical across two runs; `flowcheck` accepts a valid hand-written label file and flags a bad
call ordinal, a cond `code` that is not in the method, a cond used as a data source (and
source = sink), a skipped ordinal, an out-of-span line, over-long evidence, and empty `pairs` without a
reason; `patch -p1` applies cleanly to the committed file and reproduces the tested file byte for byte.
Not tested: `verify` end to end (needs `tree_sitter`, absent from the system Python in this session).

```diff
--- a/cpg/spikes/frontend-fitness/run_fitness.py
+++ b/cpg/spikes/frontend-fitness/run_fitness.py
@@ -15,6 +15,8 @@
   flowdraw <lang>   30 golden-flow methods (ML §6.4)            -> sample/<lang>-flow.jsonl
   c5pilot  <lang>   first 50 of the c5 hash sort of the frame   -> truth/<lang>.c5-pilot.jsonl
   brief    <lang>   fill ML §6.5's template                     -> sample/<lang>.brief.md
+  flowbrief <lang>  fill flow-brief.template.md (S56)           -> sample/<lang>-flow.brief.md
+  flowcheck <lang>  schema-check labels/<lang>-flow.jsonl (S56)  -> stdout JSON; exit 1 on any error
   refmap   <lang>   map a SCIP JSON / callgraph output to sites -> truth/<lang>.ref.jsonl | c5 json
   genaudit csharp   the hash-sorted generic/nameof sites of the sample (ML §3.2, ccg-p0a-v1-gen|) -> truth/csharp.gen-audit.jsonl
   verify            re-derive every frame/sample/flow file from corpora.yaml alone; exit 1 on drift
@@ -1923,6 +1925,290 @@
     return {"lang": lang, "bytes": len(txt.encode())}
 
 
+# ------------------------------------------------------------------ flow brief + label check (S56; ML §6.4, §11 N3)
+# The template is flow-brief.template.md (between its BRIEF markers). Same shape as `brief`: deterministic,
+# no network, no Joern, reads only the template, the bundle spec, corpora.yaml-derived paths and
+# sample/<lang>-flow.jsonl. `flowcheck` opens only sample/<lang>-flow.jsonl and labels/<lang>-flow.jsonl.
+
+FLOW_TEMPLATE = HERE / "flow-brief.template.md"
+FLOW_BEGIN, FLOW_END = "<!-- BRIEF-BEGIN -->", "<!-- BRIEF-END -->"
+
+
+def flow_brief_path(lang): return HERE / "sample" / f"{lang}-flow.brief.md"
+def flow_label_path(lang): return HERE / "labels" / f"{lang}-flow.jsonl"
+
+
+# Short-circuit operators per language: bundle spec §5.5 "short-circuit operands that guard a point".
+# Kotlin's `?:` is the null-coalescing operator (JS/C# `??`); Kotlin's infix `and`/`or` do not short-circuit.
+FLOW_OPS = {
+    "python": "`and`, `or`",
+    "javascript": "`&&`, `||` and `??`",
+    "java": "`&&`, `||`",
+    "kotlin": "`&&`, `||` and the elvis operator `?:` (Kotlin's null-coalescing operator; the infix functions `and`/`or` do not short-circuit and do not count)",
+    "csharp": "`&&`, `||` and `??`",
+    "go": "`&&`, `||`",
+    "cpp": "`&&`, `||`",
+}
+# Invented snippets, not from any corpus (the labeller must not meet its own methods in the brief).
+FLOW_OP_EXAMPLES = {
+    "python": "`if token and verify(token): …` has the condition `token` guarding the call `verify`; `v = cache or load(key)` has the condition `cache` guarding `load`.",
+    "javascript": "`props.onClick && props.onClick(e)` has the condition `props.onClick` guarding the call `onClick`; `const v = cached ?? load(key)` has the condition `cached` guarding `load`; `ready || init()` has the condition `ready` guarding `init`.",
+    "java": "`if (user != null && audit.log(user)) { … }` has the condition `user != null` guarding the call `log`; `ok = hit || fetch(id)` has the condition `hit` guarding `fetch`.",
+    "kotlin": "`val v = cache ?: load(key)` has the condition `cache` guarding `load`; `if (a != null && check(a)) { … }` has the condition `a != null` guarding `check`.",
+    "csharp": "`var v = cache ?? Load(key);` has the condition `cache` guarding `Load`; `ok = user != null && Audit(user);` has the condition `user != null` guarding `Audit`.",
+    "go": "`if ok && handle(x) { … }` has the condition `ok` guarding the call `handle`; `done = seen || mark(k)` has the condition `seen` guarding `mark`.",
+    "cpp": "`if (p && p->run(x)) { … }` has the condition `p` guarding the call `run`; `ok = hit || fetch(id);` has the condition `hit` guarding `fetch`.",
+}
+FLOW_COND_FORMS = {
+    "python": "`if`/`elif` conditions, `while` and `for` headers, conditional expressions (`x if c else y`), and `match` `case` labels,",
+    "javascript": "`if` conditions, `while`, `do … while` and `for`/`for…of`/`for…in` headers, ternaries (`c ? a : b`), and `switch` `case` labels,",
+    "java": "`if` conditions, `while`, `do … while` and `for`/enhanced-`for` headers, ternaries, and `switch` `case` labels (with their `when` guard, if any),",
+    "kotlin": "`if` conditions (statement or expression), `while`, `do … while` and `for` headers, and `when` branch conditions,",
+    "csharp": "`if` conditions, `while`, `do … while`, `for` and `foreach` headers, ternaries, `switch` `case` labels (with their `when` guard) and `catch … when (…)` filters,",
+    "go": "`if` conditions, `for` conditions and range clauses, and `switch` `case` labels,",
+    "cpp": "`if` conditions, `while`, `do … while` and `for`/range-`for` headers, ternaries, and `switch` `case` labels,",
+}
+FLOW_SCOPE_NOTES = {
+    "python": "Nested `def`, `class` and `lambda` bodies are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that creates or receives the nested function is this method's.",
+    "javascript": "Nested functions, arrow functions, callbacks and classes are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the callback (`xs.map(x => …)`) is this method's.",
+    "java": "Lambdas, local and anonymous classes are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the lambda is this method's.",
+    "kotlin": "Lambdas (trailing lambdas included), local functions and `object :` expressions are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the lambda is this method's.",
+    "csharp": "Lambdas, anonymous methods and local functions are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the lambda is this method's.",
+    "go": "Function literals are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the function literal, or `go`/`defer` of it, is this method's.",
+    "cpp": "Lambdas and local classes are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the lambda is this method's.",
+}
+FLOW_RECEIVER_NOTE = {
+    "python": "The first parameter of a function that sits directly in a class body (`self`, `cls`), unless it is decorated `@staticmethod`, is the receiver: `param` index -1, and the next parameter is index 0. A read of `self.x` is a read of `param` -1.",
+    "javascript": "`this` is implicit and is not a parameter (bundle spec §5.4): a value read from `this` or `this.x` has no source point, so it makes no pair.",
+    "java": "`this` is implicit and is not a parameter (bundle spec §5.4): a value read from `this` or `this.x` has no source point, so it makes no pair.",
+    "kotlin": "`this` is implicit and is not a parameter (bundle spec §5.4), except the receiver of an *extension function*, which is `param` index -1 (named `this`); the first ordinary parameter is then index 0. A value read from the implicit `this` has no source point.",
+    "csharp": "`this` is implicit and is not a parameter (bundle spec §5.4), except the `this` parameter of an *extension method*, which is `param` index -1; the next parameter is index 0. A value read from the implicit `this` has no source point.",
+    "go": "A method's receiver (`b` in `func (b *T) …`) is `param` index -1, and the first ordinary parameter is index 0.",
+    "cpp": "Bundle spec §5.4 names no receiver parameter for C/C++. Treat the implicit `this` as the JVM's: it is not a parameter, and a value read from it has no source point, so it makes no pair.",
+}
+
+
+def flow_brief_template() -> str:
+    t = FLOW_TEMPLATE.read_text(encoding="utf-8")
+    a = t.index(FLOW_BEGIN) + len(FLOW_BEGIN)
+    b = t.index(FLOW_END, a)
+    return t[a:b].strip("\n") + "\n"
+
+
+def _spec_between(t: str, start: str, end: str) -> str:
+    a = t.index(start)
+    return t[a:t.index(end, a)].rstrip()
+
+
+def flow_spec_excerpt() -> str:
+    """Bundle spec3 §5.4, §5.5, §5.6 (the call-site naming rows and the 'which calls' bullets) and §5.7,
+    verbatim except two redactions: the sentences that name the comparison tool. A redaction that no longer
+    matches the spec text fails loudly, so spec drift cannot slip a tool name into the brief."""
+    t = SPEC3.read_text(encoding="utf-8")
+    s54 = _spec_between(t, "### 5.4 `param`", "### 5.5 ")
+    s55 = _spec_between(t, "### 5.5 `ret` and `cond`", "### 5.6 ")
+    s55n = re.sub(r"\s*\(Joern's CDG sources decide what counts\..*?misses\.\)", "", s55, flags=re.S)
+    if s55n == s55:
+        raise SystemExit("flow brief: spec §5.5 redaction no longer matches the spec text")
+    s56 = _spec_between(t, "### 5.6 `callsite`", "### 5.7 ")
+    head, rows, bullets, cur = [], [], [], None
+    for ln in s56.split("\n"):
+        if ln.startswith("- "):
+            cur = [ln]
+            bullets.append(cur)
+        elif cur is not None and ln.startswith("  "):
+            cur.append(ln)
+        elif ln.startswith("|"):
+            cur = None
+            if ln.startswith(("| Field", "|---")) or any(ln.startswith(f"| `{k}`") for k in ("calleeName", "ordinal", "callKind", "line")):
+                rows.append(ln)
+        elif ln.strip() and not rows:
+            head.append(ln)
+    keep = ["\n".join(b) for b in bullets if b[0].startswith(("- **Which calls are call sites:**", "- **Decorator and annotation"))]
+    if len(keep) != 2:
+        raise SystemExit("flow brief: spec §5.6 bullets no longer match")
+    keep = [k.replace("(Joern's `<operator>.*` calls)", "(language operators)") for k in keep]
+    if any("Joern" in k for k in keep):
+        raise SystemExit("flow brief: spec §5.6 redaction no longer matches the spec text")
+    s56n = "\n\n".join(head) + "\n\n" + "\n".join(rows) + "\n\n" + "\n".join(keep)
+    s57 = _spec_between(t, "### 5.7 The call-site ordinal", "### 5.8 ")
+    return "\n\n".join([s54, s55n, s56n, s57]) + "\n"
+
+
+def spec_version() -> str:
+    t = SPEC3.read_text(encoding="utf-8")
+    m = re.search(r"\*\*Version:\*\* ([0-9.]+)", t.split("\n", 4)[2])
+    return m.group(1) if m else "?"
+
+
+def fill_flow_brief(lang: str) -> str:
+    if not flow_path(lang).exists():
+        raise SystemExit(f"flow brief {lang}: {flow_path(lang)} missing (run flowdraw first)")
+    n = len(read_jsonl(flow_path(lang)))
+    slots = {
+        "{LANG}": LANG_LABEL[lang],
+        "{N_METHODS}": str(n),
+        "{SAMPLE_FILE}": str(flow_path(lang)),
+        "{CORPUS_ROOT}": str(src_dir(lang)),
+        "{LABEL_FILE}": str(flow_label_path(lang)),
+        "{SHORT_CIRCUIT_OPS}": FLOW_OPS[lang],
+        "{SHORT_CIRCUIT_EXAMPLES}": FLOW_OP_EXAMPLES[lang],
+        "{COND_FORMS}": FLOW_COND_FORMS[lang],
+        "{SCOPE_NOTES}": FLOW_SCOPE_NOTES[lang],
+        "{RECEIVER_NOTE}": FLOW_RECEIVER_NOTE[lang],
+        "{SPEC_VERSION}": f"{spec_version()} (`docs/plans/connected-code-graph-bundle-spec3.md`; `-spec4` §2 changes nothing for a named method)",
+    }
+    body = flow_brief_template()
+    for k, v in slots.items():
+        body = body.replace(k, v)
+    return body.rstrip() + "\n\n" + flow_spec_excerpt()
+
+
+FLOW_LEAKS = [r"results/", r"truth/", r"ref\.jsonl", r"adjudication", r"extract\.sc", r"\bprojection\b", r"CONTROLLED_BY",
+              r"\bCDG\b", r"\bDATA\b", r"\bCALLS_NAME\b", r"\bFULL_NAME\b", r"cpg\.bin", r"\b0\.95\b", r"\brecall\b",
+              r"\bthreshold\b", r"flowAvailable", r"labels/[a-z]+\.jsonl", r"\.brief\.md", r"S26|S28|S56|S57"]
+
+
+def validate_flow_brief(lang: str, text: str) -> list[str]:
+    errs = []
+    left = re.findall(r"\{[A-Z_]{2,}\}", text)
+    if left:
+        errs.append(f"unfilled slots: {sorted(set(left))}")
+    for must in ("**Output.**", "**What counts as a condition (`cond`).**", "short-circuit operands that guard a point",
+                 "**What to label: pairs.**", "**Appendix: bundle spec points**", "### 5.4 `param`", "### 5.5 `ret` and `cond`",
+                 "### 5.7 The call-site ordinal", FLOW_OPS[lang], FLOW_OP_EXAMPLES[lang]):
+        if must not in text:
+            errs.append(f"missing {must[:60]}")
+    if lang == "javascript":  # the TS/JS instruction is mandatory and explicit
+        for must in ("`&&`, `||` and `??`", "Every operand of such an operator", "`props.onClick && props.onClick(e)`"):
+            if must not in text:
+                errs.append(f"javascript: missing {must}")
+    for pat in FLOW_LEAKS:
+        if re.search(pat, text):
+            errs.append(f"leak pattern {pat!r}")
+    # `joern` is allowed once, in the prohibition sentence; nothing else about tools of the comparison
+    if len(re.findall(r"joern", text, flags=re.I)) != 1:
+        errs.append("expected exactly one mention of Joern (the prohibition)")
+    return errs
+
+
+def cmd_flowbrief(lang: str) -> dict:
+    txt = fill_flow_brief(lang)
+    errs = validate_flow_brief(lang, txt)
+    if errs:
+        raise SystemExit(f"flowbrief {lang}: {errs}")
+    flow_brief_path(lang).write_text(txt, encoding="utf-8")
+    return {"lang": lang, "bytes": len(txt.encode())}
+
+
+FLOW_ENDPOINTS = {"param": {"index", "name"}, "call": {"calleeName", "ordinal", "line"}, "ret": {"ordinal", "line"},
+                  "cond": {"line", "code"}}
+FLOW_SOURCES = {"data": {"param", "call"}, "control": {"cond"}}
+FLOW_SINKS = {"data": {"call", "ret", "cond"}, "control": {"call", "ret", "cond"}}
+
+
+def _flow_endpoint_errs(e, role: str, kind: str, m: dict, text: str, ctext: str) -> list[str]:
+    if not isinstance(e, dict) or e.get("point") not in FLOW_ENDPOINTS:
+        return [f"{role}: bad point"]
+    p = e["point"]
+    if set(e) != {"point"} | FLOW_ENDPOINTS[p]:
+        return [f"{role}: keys {sorted(e)} != {sorted({'point'} | FLOW_ENDPOINTS[p])}"]
+    errs = []
+    allowed = (FLOW_SOURCES if role == "source" else FLOW_SINKS)[kind]
+    if p not in allowed:
+        errs.append(f"{role}: a {p} cannot be a {kind} {role}")
+    ints = [k for k in ("index", "ordinal", "line") if k in e]
+    if any(type(e[k]) is not int for k in ints):
+        return errs + [f"{role}: non-integer field"]
+    if "line" in e and not (m["lineStart"] <= e["line"] <= m["lineEnd"]):
+        errs.append(f"{role}: line {e['line']} outside {m['lineStart']}-{m['lineEnd']}")
+    if p == "param":
+        if e["index"] < -1 or not isinstance(e["name"], str) or not e["name"]:
+            errs.append(f"{role}: bad param index/name")
+    elif p == "call":
+        nm = e["calleeName"]
+        if not isinstance(nm, str) or not nm or e["ordinal"] < 0:
+            errs.append(f"{role}: bad call")
+        elif nm != "<dynamic>" and e["ordinal"] >= len(re.findall(r"(?<![\w$])" + re.escape(nm) + r"(?![\w$])", text)):
+            errs.append(f"{role}: call {nm}#{e['ordinal']} has fewer occurrences in the method")
+    elif p == "ret":
+        if e["ordinal"] < 0:
+            errs.append(f"{role}: bad ret ordinal")
+    else:
+        c = e["code"]
+        if not isinstance(c, str) or not c.strip() or len(c) > 200 or ws(c) not in ctext:
+            errs.append(f"{role}: cond code is not (<= 200 chars of) the method's text")
+    return errs
+
+
+def cmd_flowcheck(lang: str) -> dict:
+    """Schema-validate labels/<lang>-flow.jsonl against sample/<lang>-flow.jsonl (opens only those two files).
+    Checks ordinal order and resumability, endpoint shapes, kind/endpoint compatibility, lines in span, evidence
+    length, source != sink, duplicate pairs; counts pairs. Advisory counts (ML §8.4's 120-pair floor) are not errors."""
+    sample = read_jsonl(flow_path(lang))
+    rows = []
+    errs: list[str] = []
+    if flow_label_path(lang).exists():
+        for i, ln in enumerate(open(flow_label_path(lang), encoding="utf-8"), 1):
+            if not ln.strip():
+                continue
+            try:
+                rows.append((i, json.loads(ln)))
+            except json.JSONDecodeError as ex:
+                errs.append(f"line {i}: not JSON ({ex.msg})")
+                break
+    stats = {"data": 0, "control": 0, "unsure": 0, "withPair": 0, "withoutPair": 0}
+    for k, (i, o) in enumerate(rows):
+        pre = f"line {i}"
+        if k >= len(sample):
+            errs.append(f"{pre}: more lines than methods ({len(sample)})")
+            break
+        m = sample[k]
+        if not isinstance(o, dict) or not {"ordinal", "path", "qname", "pairs"} <= set(o) or set(o) - {"ordinal", "path", "qname", "pairs", "unlisted", "noPairReason"}:
+            errs.append(f"{pre}: keys {sorted(o) if isinstance(o, dict) else type(o).__name__}")
+            continue
+        if (o["ordinal"], o["path"], o["qname"]) != (m["ordinal"], m["path"], m["qname"]):
+            errs.append(f"{pre}: expected ordinal {m['ordinal']} {m['qname']}")
+            continue
+        text, ctext = m["source"], ws(m["source"])
+        seen = set()
+        if not isinstance(o["pairs"], list):
+            errs.append(f"{pre}: pairs not a list")
+            continue
+        for j, p in enumerate(o["pairs"]):
+            pp = f"{pre} pair {j}"
+            if not isinstance(p, dict) or set(p) != {"kind", "source", "sink", "evidence", "confidence"}:
+                errs.append(f"{pp}: keys")
+                continue
+            if p["kind"] not in FLOW_SOURCES or p["confidence"] not in ("sure", "unsure"):
+                errs.append(f"{pp}: kind/confidence")
+                continue
+            if not isinstance(p["evidence"], str) or not p["evidence"].strip() or len(p["evidence"]) > 300:
+                errs.append(f"{pp}: evidence missing or > 300 chars")
+            ee = _flow_endpoint_errs(p["source"], "source", p["kind"], m, text, ctext) + _flow_endpoint_errs(p["sink"], "sink", p["kind"], m, text, ctext)
+            errs += [f"{pp}: {e}" for e in ee]
+            key = (p["kind"], jdump(p["source"]), jdump(p["sink"]))
+            if p["source"] == p["sink"]:
+                errs.append(f"{pp}: source equals sink")
+            if key in seen:
+                errs.append(f"{pp}: duplicate pair")
+            seen.add(key)
+            stats[p["kind"]] += 1
+            stats["unsure"] += p["confidence"] == "unsure"
+        if o["pairs"]:
+            stats["withPair"] += 1
+        else:
+            stats["withoutPair"] += 1
+            if not (isinstance(o.get("noPairReason"), str) and o["noPairReason"].strip() and len(o["noPairReason"]) <= 200):
+                errs.append(f"{pre}: no pairs and no noPairReason")
+        for u in o.get("unlisted", []):
+            if not (isinstance(u, dict) and set(u) == {"line", "construct"} and type(u["line"]) is int and isinstance(u["construct"], str) and 0 < len(u["construct"]) <= 120):
+                errs.append(f"{pre}: bad unlisted entry")
+    pairs = stats["data"] + stats["control"]
+    return {"lang": lang, "methods": len(sample), "labelled": len(rows), "complete": len(rows) == len(sample) and not errs,
+            "pairs": pairs, **stats, "belowFloor120": pairs < 120, "errors": errs[:60], "nErrors": len(errs)}
+
+
 # ------------------------------------------------------------------ firewall + verify
 
 FIREWALL_PATTERNS = [r"joern", r"cpg\.bin", r"\.cpg\b", r"\bFULL_NAME\b", r"results/", r"truth/", r"ref\.jsonl",
@@ -1974,6 +2260,12 @@
             problems.append(f"{lang}: brief differs")
         if validate_brief(brief_path(lang).read_text(encoding="utf-8") if brief_path(lang).exists() else ""):
             problems.append(f"{lang}: brief invalid")
+        if flow_brief_path(lang).exists():
+            ftxt = flow_brief_path(lang).read_text(encoding="utf-8")
+            if ftxt != fill_flow_brief(lang):
+                problems.append(f"{lang}: flow brief differs")
+            if validate_flow_brief(lang, ftxt):
+                problems.append(f"{lang}: flow brief invalid")
     problems += firewall_scan()
     return {"checked": langs, "problems": problems}
 
@@ -2064,7 +2356,7 @@
     s = sub.add_parser("genaudit")
     s.add_argument("lang", choices=["csharp"])
     s.add_argument("--select", action="store_true", help="S57 only (needs labels/): write the sites to check from source")
-    for c in ("fetch", "census", "frame", "draw", "flowdraw", "c5pilot", "brief", "joern"):
+    for c in ("fetch", "census", "frame", "draw", "flowdraw", "c5pilot", "brief", "flowbrief", "flowcheck", "joern"):
         s = sub.add_parser(c)
         s.add_argument("lang", choices=LANGS)
         if c == "fetch":
@@ -2095,6 +2387,12 @@
         r = cmd_c5pilot(a.lang)
     elif a.cmd == "brief":
         r = cmd_brief(a.lang)
+    elif a.cmd == "flowbrief":
+        r = cmd_flowbrief(a.lang)
+    elif a.cmd == "flowcheck":
+        r = cmd_flowcheck(a.lang)
+        print(json.dumps(r, indent=1))
+        return 1 if r["nErrors"] else 0
     elif a.cmd == "joern":
         r = cmd_joern(a.lang)
     elif a.cmd == "genaudit":
```
