# Flow-labeller brief — template (S56)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** S56 (P0-47a…g)

Harness-facing part (never pasted): everything above the BRIEF-BEGIN marker line and below the
BRIEF-END marker line (HTML comments, one per line, at the brief's two ends). The generator
(`run_fitness.py flowbrief <lang>`, patch in `cpg/spikes/frontend-fitness/flow-brief.patch.md`)
pastes only what is between the two markers, fills the slots, and appends the bundle spec excerpt. Output: `sample/<lang>-flow.brief.md`, pasted
into the dispatch as it is (as `sample/<lang>.brief.md` is for S26). Authority: ML note §6.4, §7.3,
§8.4, §11 N3; bundle spec `-spec3` §5.4–§5.7 (+ `-spec4` §2, which does not touch a named method).

**Slots** (all filled by the generator; `validate_flow_brief` fails on a leftover `{UPPER}`):
`{LANG}` · `{N_METHODS}` (lines of `sample/<lang>-flow.jsonl`) · `{SAMPLE_FILE}` · `{CORPUS_ROOT}` ·
`{LABEL_FILE}` (`labels/<lang>-flow.jsonl`) · `{SHORT_CIRCUIT_OPS}` · `{SHORT_CIRCUIT_EXAMPLES}` ·
`{COND_FORMS}` · `{SCOPE_NOTES}` · `{RECEIVER_NOTE}` · `{SPEC_VERSION}`. The per-language values
live in the generator's tables, not here.

**What S28 does with the labels (so the brief asks for nothing else).** Not shown to the labeller.

1. *Join of endpoints to the projection's points,* per method (`path` + `qname` → the bundle symbol):
   `param` by `index` (spec §5.4 numbering, receiver -1; `name` is a sanity check); `ret` by
   `ordinal`; `call` by `callSiteId` = (`calleeName`, `ordinal`), falling back to the one projection
   call site with that `calleeName` on the labeller's `line`; `cond` by `line` and collapsed `code`
   (equal, else one contains the other, on that line). A cond's `ordinal` is **not** asked for: the
   projection numbers conditions from the CDG's population (`normalize.py` sort key), which a
   reader cannot reproduce. An endpoint that matches nothing or several points makes its pair
   `endpoint-unresolved`: **a miss in the recall count**, listed separately in
   `adjudication/<lang>-flow-misses.jsonl` so S28's adjudicator can tell a labeller naming error from
   a projection gap.
2. *Recall (ML §7.3).* Data pair: connected iff a directed chain of ≥ 1 `DATA` edges joins the
   sink point to the source point (access paths ignored, `field-truncated` counts). Control pair:
   connected iff a direct `CONTROLLED_BY` edge runs from the sink point to the source `Cond`
   (cond→cond included). Pooled and per kind, with the §7.3 cluster interval over the methods.
   `unsure` pairs are in the pooled figure (ML: "every pair"); the sure-only figure is reported
   beside it.
3. *Spurious pairs (ML §8.4).* The projection's distinct (def, use) point pairs for `DATA` and
   (point, cond) pairs for `CONTROLLED_BY`, with source = sink pairs removed from both sides,
   compared by point identity with the labelled direct pairs. A direct-pair golden set is why the
   labeller is asked for **direct** pairs only: the projection's edges are direct, and the §7.3
   chain rule already credits transitive reach on the projection side.
4. *Used only by people:* `evidence`, `unlisted`, `noPairReason`. `unlisted` constructs are what
   spec §5.5 says S11 must record when the cond definition misses a class.
5. *Consequence for the sources and sinks asked for.* A bundle `data` edge ends at a `param` or a
   call result (`def`) and at a call, a `ret` or a `cond` (`use`). A field write and a read of an
   implicit `this` have no point, so they are not asked for; call results as sources and conds as
   sinks are asked for because the projection emits those edges and an unlabelled class of them
   would show up as "omissions" in the spurious-pair sample (ML §8.4 flag rule).

<!-- BRIEF-BEGIN -->
**You are labelling data flow and control flow inside {N_METHODS} {LANG} methods, blind.** The
methods are in `{SAMPLE_FILE}`, one per line. For each one you write down which values reach which
places, and which conditions decide which operations, by reading the source. Your answers are
compared with a tool's output later; you will not see that output, and you must not use any tool
that analyses code for you.

**What you have.** The codebase at `{CORPUS_ROOT}` (read-only), the sample file, and your output
file `{LABEL_FILE}`. Use only file reading, `grep`/`rg` and `find`. The output file is the only
file you write.

**What you must not do.** Do not run any program that understands code: no Joern or `*2cpg`, no
indexer, no language server (Pyright, `tsc`, `gopls`, clangd, Roslyn, OmniSharp), no compiler or
build tool, no parser or AST library (not even a one-line script that imports one), no
`go doc`/`go vet`. Do not use the web, MCP tools or any knowledge base. Do not open anything outside
`{CORPUS_ROOT}` except the sample file and your output file; in particular nothing else under
`cpg/spikes/frontend-fitness/`. Using any of these voids your work.

**Which method a sample line means.** Each line gives `ordinal`, `path` (relative to `{CORPUS_ROOT}`),
`qname`, `lineStart`, `lineEnd` (the method's span in the file) and `source` (its text). Label that
method only. If `source` and the file disagree, the file wins. You may read other files for
context, but only this method's own points count (next section).

**The method's own points.** A *point* is one of four things inside the method, and nothing else is
ever a source or a sink:

- `param`: a parameter, by its 0-based position **as written, not counting the receiver**, plus its
  name as written (`*args` keeps its sigil). {RECEIVER_NOTE}
- `call`: a call site. By `calleeName`, the last identifier of the callee expression without type
  arguments (`get` in `self.repo.get(x)`; `Foo` in `new Foo()`; `<dynamic>` for a computed callee),
  its `ordinal` (0-based rank among this method's call sites with the same `calleeName`, ordered by
  line then column of that last identifier; the appendix, §5.7, gives the rule), and the `line` of
  that identifier. A call is a point both as the place where its arguments and receiver arrive and
  as the origin of its result.
- `ret`: a return, by its 0-based `ordinal` among this method's own `return` statements in source
  order (an expression-bodied function has one, ordinal 0), and its `line`.
- `cond`: a condition (next section), by its `line` and its `code`.

{SCOPE_NOTES}

**What counts as a condition (`cond`).** Bundle spec §5.5, appended: the headers and tests that
decide which way execution goes: {COND_FORMS} **and short-circuit operands that guard a point.**
The short-circuit operators of {LANG} are {SHORT_CIRCUIT_OPS}. **Every operand of such an operator
whose right-hand side contains a call or another condition is a condition in its own right**, in
addition to any `if`/`while`/ternary header it sits in. Example: {SHORT_CIRCUIT_EXAMPLES} This holds
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
a method**, appended to `{LABEL_FILE}` when that method is finished:

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
- **Stop** after the line for ordinal **{N_METHODS}**.
- **Resume:** if you are restarted, read the last ordinal in `{LABEL_FILE}` and continue with the
  next one. Never rewrite earlier lines. Write a method's line only when all of its pairs are done.

**Appendix: bundle spec points** (§5.4 `param`, §5.5 `ret` and `cond`, §5.6 `callsite` in part,
§5.7 the call-site ordinal; bundle spec version {SPEC_VERSION}, inserted by the harness). Where it
says a point is the "enclosing symbol's own", a nested function, lambda or class is a different
symbol.
<!-- BRIEF-END -->
