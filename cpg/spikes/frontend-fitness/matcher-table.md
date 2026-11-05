# Site matcher table (ML §7.1)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** C-1001 (P0, S25)

How a frame site (`frame/<lang>.jsonl`) is matched to a Joern `CALL`. **Observed**, not assumed: every row below was
read off a Joern v4.0.579 CPG of a small synthetic probe project per language (`matcher-probe.sc`; probe sources
under `cpg/.ccg-work/frontend-fitness/probe/`, not committed; never a corpus), and then confirmed by running
`measure.sc` over each probe's own frame (8–11 sites each). The rule is transcribed in `measure.sc`'s `nameOk`.

**Rule (ML §7.1).** Same file; Joern's call **line** equals the frame's `nameLine` (after the per-language offset below);
Joern's call **name** is accepted by the kind rule; if several calls match, the one whose **column** is nearest the
frame's `column` (the callee expression's first character) wins, a call with no column (`-1`) sorts last, then the
lower node id. Nothing else is compared.

## Position conventions (observed)

| Frontend | `--language` | Joern line | Joern column | Offset applied to the frame value |
|---|---|---|---|---|
| Python `pysrc2cpg` | `pythonsrc` | 1-based | 1-based | none |
| JS/TS `jssrc2cpg` | `jssrc` | 1-based | **0-based** | column − 1 |
| Java `javasrc2cpg` | `javasrc` | 1-based | 1-based (no column for `<init>`: `-1`) | none |
| Kotlin `kotlin2cpg` | `kotlin` | 1-based | **0-based** | column − 1 |
| C# `csharpsrc2cpg` | `csharpsrc` | **0-based** | **0-based** | line − 1, column − 1 |
| Go `gosrc2cpg` | `golang` | 1-based | 1-based | none |
| C/C++ `c2cpg` | `c` | 1-based | 1-based | none |

**Multi-line chains (S27 inspection).** When a chained call's name sits on a later line than the chain's start
(`new X()\n  .m()`), javasrc2cpg and gosrc2cpg give the CALL the line of the **callee expression's first character**, not of
the name. `measure.sc` therefore tries the name's line first and, only if no call matches there, the frame's `line`
(same name rule, nearest column).

A chain `a.b().c()` yields two Joern calls (`b` with code `a.b()`, `c` with code `a.b().c()`) that **share** line and
column (the start of the whole expression), exactly like the two frame sites; their names tell them apart.

## Call name per frame kind

| Language | plain / method / qualified | constructor (`new X`, `X(…)`, `super(…)`, `this(…)`) | Other kinds |
|---|---|---|---|
| Python | `name` = `calleeName` (`m` for `b.m()`, `f` for `f(2)`, `h` for a function value `h(3)`, `len` for a builtin) | `name` = the class name (`B` for `B(1)`; resolved to `B.__init__`) | decorators are out of the frame |
| JS/TS | `name` = `calleeName`; **optional-chain call `a?.b(x)`: `name` is the whole callee text `a?.b`** (code `a?.b(x)`), found by S27's unmatched-site inspection (`measure.sc` accepts `name` ending `?.`+`calleeName` when `code` starts with `name`) | **`<operator>.new`** with `code` = `new X(…)`; the matcher requires `code` to start with `new ` and contain `calleeName` (the index therefore keeps this one operator call) | tagged template `` tag`x` ``: `name` = `tag` (also synthesized `push`/`__ecma.Array.factory` calls on the same line, never matched). **JSX `<Cmp/>`: no CALL at all, so every JSX site is unmatched by construction** (counted *absent in Joern*) |
| Java | `name` = `calleeName` | `<init>` (`new Base(1)` has no column: `-1`; `super(1)`/`this()` carry a column); the matcher accepts `<init>` or the type name | — |
| Kotlin | `name` = `calleeName`, **or `<init>`**: `B(3)` is a plain-looking frame site and Joern names the call `<init>` | same | trailing lambda `run { 1 }`: one call (the frame dedupes the `call_expression(call_expression, annotated_lambda)` pair to one site). **A delegation-specifier call `class A : B(x)`: no CALL, unmatched by construction** |
| C# | `name` = `calleeName` (`G` for `b.G<int>()`, type arguments dropped, as the frame's `calleeName`; `nameof` for `nameof(b)`) | `<init>` (`new B(2)`, `new List<int>()`) | the `: base(1)` initializer is not a frame site |
| Go | `name` = `calleeName` (`A` for `i.A()` and `t.A()`; a call through a function variable `fn(t)` is `fn`) | none (no constructors) | conversion `int(3)` and builtin `len("ab")` are CALLs named `int` / `len` (labelled `notACall` / external); **a generic instantiation call `g[int](1)` has the EMPTY name**, so it is unmatched by construction (the frame emits it from `type_conversion_expression`) |
| C/C++ | `name` = `calleeName`; a template call keeps its arguments in the name (`tf<int>`), so the matcher also accepts `calleeName + "<"…` | `name` = the type name (`new B(1)` is a call `B`, resolved to the constructor, beside `<operator>.alloc`/`<operator>.new`); a temporary `B(5)` is also `B` | **function-like macro `M(2)`: a CALL named `M` with dispatch `INLINED` (unresolved), beside the calls its expansion makes at the same position** |

## Resolved targets and name-match candidates (what `measure.sc` records)

* `res` = the matched CALL's `CALL` edges to METHOD nodes (`_callOut`), each with file, line, `fullName`, `ext`
  (`isExternal`) and `vendored` (a directory segment of its file is in the corpus's effective `vendoredDirs`). A target that
  is `ext` or `vendored` is **external** for every metric (ML §2, §7.1); a call to an external stub still has `res`
  entries (e.g. `java.util.List.add:…`, `<unresolvedNamespace>.nameof`), so consumers must filter on `ext`.
* `nm` = the count of non-external, non-vendored METHODs whose `name` equals `calleeName` (name-only, corpus-wide; the
  P1 fallback). For a constructor kind the candidates are the methods named like the type (`calleeName`); the candidate
  set does not include `<init>` methods of other classes. `nmList` (sampled sites only) lists them.
* The census match rate (matched frame sites ÷ frame sites) and the unmatched-by-construction classes above are reported
  per configuration; **a matcher error is fixed and the run repeated until zero remain** (ML §7.1); no metric is computed
  before that (S27).

## Census fields (what `measure.sc` records for S27c; `censusSchema` 1)

Observed on the planted sources of `cpg/tests/data/census/real-src/<lang>/` (Joern 4.0.579; frozen outputs `cpg/tests/data/census/<lang>.real.measure.jsonl`,
decorator shapes `results/<lang>.decorator-probe.json`). The field definitions are in `measure.sc`'s header; what differs per frontend:

* **Receiver / `recv`.** `recv` = the call has an argument with `argumentIndex` 0. java, csharp and cpp also put the RECEIVER edge on it; python and
  javascript put the RECEIVER edge on the callee expression and still carry the instance as argument 0; javascript adds an implicit `this` at 0 to a
  plain call (`call(x)` has `this`, `x`); go has no receiver argument (`obj.call("x")` has one argument); a plain python or cpp call has none.
  csharp: a plain unqualified call `F("s")` has **zero** `argument`s (the arguments exist only in `code`).
* **Keyword arguments (python).** `k=v` has `argumentIndex` -1 and AST `order` = source position; `order` ranks such a call by AST `order`; `ai` and `aname` are in the record.
* **Literal `lit` text.** python/java/csharp/cpp/go `LITERAL.code` is the source text (cpp keeps adjacent-string `"a" "b"` as one LITERAL and `R"(raw)"`);
  javascript template literal `` `t` `` is a LITERAL with code `"t"`; python `"a" "b"` and `"c" + "d"` and f-strings are CALLs, java `"t" + "u"` and a static constant `J.K` are CALLs.
* **Annotations (`ann`).** ANNOTATION on the METHOD for java, kotlin, csharp and javascript/typescript; **python decorators are CALLs** (rewrite `f = deco(<METHOD_REF>)`),
  the METHOD itself carries nothing; go and cpp expose no node. javascript ANNOTATIONs have no literal children (arguments only in `code`); java `ANNOTATION_LITERAL.code`
  is **unquoted** (`/x`), kotlin's and csharp's keep their quotes. Per-language detail: `results/<lang>.decorator-probe.json`.
* **Kotlin.** Default overlays abort (`ReachingDefPass`), so the CPG is imported overlay-free (`--param frontendOnly=true`); every call then lacks an enclosing METHOD
  (`meta.callsWithoutMethod` = `meta.calls`), no site matches and no `carg` is written. Coverage, line fidelity and annotations are measured.
* **Reaching definitions (`nreach`, `reach`).** REACHING_DEF sources that are the METHOD entry node or sibling arguments of the same call are not definitions (the engine routes
  every argument into every other of an unresolved call). An assignment's target identifier and its literal are one definition.

## Known unmatched-by-construction classes (not matcher errors)

JSX component use (JS/TS), Kotlin delegation-specifier constructor calls, Go generic instantiation calls, C++ calls that
`c2cpg` does not model (parse-error regions: tree-sitter ERROR nodes mean the frame itself is incomplete there). These
count as *absent in Joern* once S27 has inspected them one by one.

## Not this table: the reference-side by-construction classes (C5)

The classes above are about **Joern** (`measure.sc`, S27). ML §3.2's C5 classes are about the **reference tool** being silent
and are tagged by `run_fitness.py` from the source alone, never from a Joern result (`byConstruction` in `truth/<lang>.ref.jsonl`):
Go `builtin`; C# `generic-callee` (the callee text **ends** in a balanced type-argument list that follows an identifier:
`a.B<int>`, `new C<T>`, `F<Func<int, int>>`; a `<` elsewhere, e.g. `list.Where(x => x < 3).ToList` or a generic argument inside a
receiver call, does not tag) and `nameof`; C++ `preproc-branch` (non-guard `#if` group) and `macro-callee` (an all-caps identifier
callee).

Two constants the ML note does not state, fixed by the harness (`site_tags`) and pinned by tests:

* **Go `builtin` is tagged by `calleeName` alone** against the closed list of the Go spec's predeclared functions (`append cap clear
  close complex copy delete imag len make max min new panic print println real recover`, 18). `callText` is not looked at, so a
  method `x.max()` that is reference-silent would be tagged too and leave the denominator. All 22 builtin sites in the Go sample
  and the one in the pilot are bare names, so the choice has changed no figure; requiring `callText == calleeName` is the
  tightening if a corpus ever has such a method.
* **The C++ include-guard rule.** A top-level `#if`/`#ifdef` whose byte span covers more than **0.8** of the file (`> 0.8 * len(file)`)
  is treated as an include guard: its branches are not `preproc-branch`. Any other `preproc_if`/`ifdef`/`elif`/`else` ancestor of
  the site is. 0.8 is the harness's choice (a guard wraps the whole file bar the license header; the constant is not in ML §3.2). The two lists overlap in kind only: a C# generic call is silent in `scip-dotnet`
yet matched by name in Joern (row "C# `G` for `b.G<int>()`" above), and Go's generic instantiation is the reverse case.
**Not observed:** how `c2cpg` positions a call written inside a macro's argument list on a corpus; the probe has only the
top-level `M(2)` row. S27's census match rate will show it; no matcher rule was added for it here.
