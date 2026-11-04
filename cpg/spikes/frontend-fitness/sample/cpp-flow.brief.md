**You are labelling data flow and control flow inside 30 C/C++ methods, blind.** The
methods are in `/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/sample/cpp-flow.jsonl`, one per line. For each one you write down which values reach which
places, and which conditions decide which operations, by reading the source. Your answers are
compared with a tool's output later; you will not see that output, and you must not use any tool
that analyses code for you.

**What you have.** The codebase at `/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/cpp/src` (read-only), the sample file, and your output
file `/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/labels/cpp-flow.jsonl`. Use only file reading, `grep`/`rg` and `find`. The output file is the only
file you write.

**What you must not do.** Do not run any program that understands code: no Joern or `*2cpg`, no
indexer, no language server (Pyright, `tsc`, `gopls`, clangd, Roslyn, OmniSharp), no compiler or
build tool, no parser or AST library (not even a one-line script that imports one), no
`go doc`/`go vet`. Do not use the web, MCP tools or any knowledge base. Do not open anything outside
`/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/cpp/src` except the sample file and your output file; in particular nothing else under
`cpg/spikes/frontend-fitness/`. Using any of these voids your work.

**Which method a sample line means.** Each line gives `ordinal`, `path` (relative to `/home/mauricio/prg/graphmind-ai-lab/cpg/.ccg-work/corpora/cpp/src`),
`qname`, `lineStart`, `lineEnd` (the method's span in the file) and `source` (its text). Label that
method only. If `source` and the file disagree, the file wins. You may read other files for
context, but only this method's own points count (next section).

**The method's own points.** A *point* is one of four things inside the method, and nothing else is
ever a source or a sink:

- `param`: a parameter, by its 0-based position **as written, not counting the receiver**, plus its
  name as written (`*args` keeps its sigil). Bundle spec §5.4 names no receiver parameter for C/C++. Treat the implicit `this` as the JVM's: it is not a parameter, and a value read from it has no source point, so it makes no pair.
- `call`: a call site. By `calleeName`, the last identifier of the callee expression without type
  arguments (`get` in `self.repo.get(x)`; `Foo` in `new Foo()`; `<dynamic>` for a computed callee),
  its `ordinal` (0-based rank among this method's call sites with the same `calleeName`, ordered by
  line then column of that last identifier; the appendix, §5.7, gives the rule), and the `line` of
  that identifier. A call is a point both as the place where its arguments and receiver arrive and
  as the origin of its result.
- `ret`: a return, by its 0-based `ordinal` among this method's own `return` statements in source
  order (an expression-bodied function has one, ordinal 0), and its `line`.
- `cond`: a condition (next section), by its `line` and its `code`.

Lambdas and local classes are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the lambda is this method's.

**What counts as a condition (`cond`).** Bundle spec §5.5, appended: the headers and tests that
decide which way execution goes: `if` conditions, `while`, `do … while` and `for`/range-`for` headers, ternaries, and `switch` `case` labels, **and short-circuit operands that guard a point.**
The short-circuit operators of C/C++ are `&&`, `||`. **Every operand of such an operator
whose right-hand side contains a call or another condition is a condition in its own right**, in
addition to any `if`/`while`/ternary header it sits in. Example: `if (p && p->run(x)) { … }` has the condition `p` guarding the call `run`; `ok = hit || fetch(id);` has the condition `hit` guarding `fetch`. This holds
whether or not the expression is inside an `if`: a bare statement, an assignment's right side, a
`return` expression and a call argument all count. An operand with no call or condition to its
right guards nothing and is not a condition on its own (it is still part of the text of the
condition it sits in). `code` is the condition's source text as written, whitespace runs collapsed
to one space, cut to 200 characters: for `if (e)`/`while (e)`/a ternary `e ? …`/a short-circuit
operand, the expression `e`; for `for (init; e; step)` the `e`; for a range or `foreach` loop the
header clause (`x in xs`); for a `case` label the label. Constructs that look like decisions but are
not in this list (comprehension filters, optional chaining `?.`, `assert`, `&&=`/`||=`/`??=`,
`finally`, exception edges) are **not** conditions: do not label them, but record each one in the
method's `unlisted` list.

**What to label: pairs.** A pair is `{kind, source, sink, evidence, confidence}`. Label **every**
pair of the two kinds below that you can justify from the source; do not stop at the first few, and
do not add a pair you cannot defend. A source never equals its sink.

*Data pairs* (`kind: "data"`). A value that starts at `source` reaches `sink` **directly**.
- `source` is a `param`, or a `call` (its **result**). Nothing else is a source: a literal, a
  module-level or static name, a field of the implicit `this`, or a value captured from an
  enclosing function has no point, so it makes no pair.
- `sink` is a `call` (the value is one of its arguments or its receiver), a `ret` (the value is
  returned), or a `cond` (the condition's expression reads the value).
- "Directly" means the value travels only through local variables and assignments (a later
  assignment to the same variable replaces the earlier value: the earlier one no longer reaches
  anything after it), aliases, loop variables (a loop variable takes the values of what is
  iterated), destructuring, fields and elements of values held in locals, and language operators
  (arithmetic, comparison, concatenation, string interpolation and template literals, ternary
  results, spread, casts, `await`, object/array/tuple literals). It does **not** travel through a
  call: a value that goes into a call and comes out as its result is two pairs (value → that call;
  that call → whatever uses its result). So for `f(g(x))`: `x` → `g`, and `g` → `f`.
- If a condition's expression reads the value, the sink is the **innermost** condition whose text
  contains the read. The result of a call that is itself an operand of a condition's expression
  goes to that condition.
- Different positions of one call (argument 0, argument 1, receiver) are not distinguished: one
  pair per (source, sink), however many times the value reaches that sink.

*Control pairs* (`kind: "control"`). `source` is a `cond`; `sink` is a `call`, a `ret` or another
`cond`, **directly decided** by it: if execution reaches the condition, one of its outcomes makes
the sink certain to execute and another outcome can prevent it. Read "directly" literally:
- In `if a { if b { f() } }`, `f` is decided by `b`, not by `a`; `b` is decided by `a`.
- An early exit decides what follows it: after `if c { return … }` (or `raise`/`throw`, `break`,
  `continue`, `goto`), every later point in that block that the exit can skip is decided by `c`,
  including later returns.
- A loop's header decides the points in its body. In an `if`/`else if`/`else` chain, each later
  condition and the `else` body are decided by the nearest condition before them, not by the
  whole chain.
- `a AND f()` (a short-circuit operand, whatever the language's spelling): `a` decides `f`.
- A point inside a condition's own expression is evaluated to compute it, so that condition does
  not decide it. Exception edges are not control flow here.

**Worked illustration** (pseudo-code, not from your corpus; `AND` stands for a short-circuit
operator; the method is `handle(req, flag)`, so `req` is `param` index 0 and `flag` index 1):

```
2   if flag AND check(req):
3       out = build(req.body)
4       return out
5   return none
```

Conditions: line 2 `flag AND check(req)` (the header) and line 2 `flag` (an operand guarding
`check`). Data pairs: `req` → `check`; `req` → `build`; `build` → `ret` 0 (through `out`);
`flag` → cond `flag`; `check` → the header cond. Control pairs: cond `flag` → `check`; the header
→ `build`; the header → `ret` 0; the header → `ret` 1 (the early exit on line 4 skips line 5).

**Output.** One JSON object per **method**, one line per method, in **ordinal order, never skipping
a method**, appended to `/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/labels/cpp-flow.jsonl` when that method is finished:

```json
{"ordinal": 7, "path": "src/a/b.py", "qname": "src/a/b.py::Repo.get",
 "pairs": [
  {"kind": "data", "source": {"point": "param", "index": 0, "name": "key"},
   "sink": {"point": "call", "calleeName": "lookup", "ordinal": 0, "line": 44},
   "confidence": "sure", "evidence": "key passed as argument to lookup, b.py:44"},
  {"kind": "data", "source": {"point": "call", "calleeName": "lookup", "ordinal": 0, "line": 44},
   "sink": {"point": "cond", "line": 45, "code": "row is None"},
   "confidence": "sure", "evidence": "row = lookup(key) b.py:44; tested b.py:45"},
  {"kind": "control", "source": {"point": "cond", "line": 45, "code": "row is None"},
   "sink": {"point": "ret", "ordinal": 0, "line": 46},
   "confidence": "sure", "evidence": "if row is None: return default, b.py:45-46"}],
 "unlisted": [{"line": 47, "construct": "comprehension filter"}]}
```

- Endpoint shapes are exactly: `{"point":"param","index":int>=-1,"name":str}`,
  `{"point":"call","calleeName":str,"ordinal":int>=0,"line":int}`,
  `{"point":"ret","ordinal":int>=0,"line":int}`, `{"point":"cond","line":int,"code":str}`.
  `line` is the file's own line number. Every `line` lies within `lineStart`–`lineEnd`.
- `confidence` is `sure` or `unsure`. **`unsure` is better than a guess or a skip.** `evidence` is
  at most 300 characters and cites file:line for what you relied on.
- `unlisted` is optional (`line`, `construct` of at most 120 characters). A method with no pair at
  all gets `"pairs": []` and a `"noPairReason"` of at most 200 characters; expect that to be rare.
- **Stop** after the line for ordinal **30**.
- **Resume:** if you are restarted, read the last ordinal in `/home/mauricio/prg/graphmind-ai-lab/cpg/spikes/frontend-fitness/labels/cpp-flow.jsonl` and continue with the
  next one. Never rewrite earlier lines. Write a method's line only when all of its pairs are done.

**Appendix: bundle spec points** (§5.4 `param`, §5.5 `ret` and `cond`, §5.6 `callsite` in part,
§5.7 the call-site ordinal; bundle spec version 1.2 (`docs/plans/connected-code-graph-bundle-spec3.md`; `-spec4` §2 changes nothing for a named method), inserted by the harness). Where it
says a point is the "enclosing symbol's own", a nested function, lambda or class is a different
symbol.

### 5.4 `param`

Id tuple: `["point", symbolId, "param", index]`. Graph label `Param`.

| Field | Type | Req | Meaning |
|---|---|---|---|
| `symbolId` | | yes | |
| `index` | int | yes | 0-based position as written, excluding the receiver. **The receiver parameter is `-1`**, path head `recv`. The receiver parameters are: Python's first parameter of a function directly in a class body, unless the function is decorated `@staticmethod`; a Go method's receiver (`s` in `func (s *Server) …`); a Kotlin extension function's receiver (named `this`); and a C# extension method's `this` parameter (analyst F4). Implicit `this` (JVM, C#, JS instance methods) is not a `param`. **[P-slice]** |
| `name` | string | yes | as written; `*args`/`**kwargs`/`...rest` keep their sigil |
| `line` | int | yes | |

### 5.5 `ret` and `cond`

- **`ret`.** Id tuple: `["point", symbolId, "ret", ordinal]`, where `ordinal` is 0-based over the
  symbol's own `return` statements in source order. Expression-bodied functions (an arrow
  function with an expression body, a Python `lambda`, a Kotlin `= expr`) have one `ret`,
  ordinal 0. Fields: `symbolId`, `ordinal`, `line`.
- **`cond`.** Id tuple: `["point", symbolId, "cond", ordinal]`, where `ordinal` is 0-based over
  the symbol's own control-structure conditions in source order: `if`/`elif`/`while`/`for`
  headers, ternaries, `case` labels, `catch` guards, and short-circuit operands that guard a
  point. Fields: `symbolId`, `ordinal`, `line`, `code`. `code` is the condition's source text,
  whitespace runs collapsed to one space, truncated to 200 characters (graph note §6.3).
  **[P-slice]**

### 5.6 `callsite`

Id tuple: §3.2 `callSiteId`. A call site is itself a point (graph note §6.3).

| Field | Type | Req | Meaning |
|---|---|---|---|
| `calleeName` | string | yes | the **last identifier** of the callee expression, as written, without type arguments: `fetch`, `apiFetch`, `include_router`, `get`. A computed callee (`obj[k]()`, `(a‖b)()`) gives `<dynamic>`. `new Foo()` gives `Foo`, `super(…)` gives `super`. A shell command or function name, or a prompt's mentioned tool name, is used as written. |
| `ordinal` | int | yes | §5.7 |
| `callKind` | `call` \| `new` \| `super` \| `tagged-template` \| `command` \| `mention` | yes | |
| `line` | int | yes | the line of the callee's last identifier |

- **Which calls are call sites:** every source-level call except language operators. Member
  access, indexing, arithmetic, string concatenation, f-strings and template literals are not call
  sites (language operators); tagged templates are.
- **Decorator and annotation applications are not call sites.** They are `decorator` records
  (§5.12), and the one fact is not emitted twice.

### 5.7 The call-site ordinal

- **Order** the enclosing symbol's own call sites (not those of nested symbols) by the position of
  their callee's last identifier: line, then column, with columns counted in code points.
- **`ordinal`** is the 0-based rank among the call sites with the same `calleeName`.
- **Using the last identifier** makes chained and nested calls total-ordered. In
  `x.foo().foo()` the inner `foo` comes first; in `f(g())`, `f` comes before `g`.
  **[P-slice]**
