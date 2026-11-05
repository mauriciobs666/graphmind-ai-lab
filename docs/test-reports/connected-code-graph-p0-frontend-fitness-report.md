# Connected code graph — P0(a) frontend-fitness report

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (M10)

Sections: §a1 (S27, below), §a2 (S28, not yet written). Protocol: `docs/plans/connected-code-graph-p0-ml.md` (ML, v1.14).
Recall, precision, remedy and stratum figures were recomputed from `cpg/spikes/frontend-fitness/results/<lang>.answers.jsonl`,
`truth/<lang>.final2.jsonl` and `truth/<lang>.ref.jsonl` by a scratch script (Wilson 95%, z = 1.959964), and the fan-out
shares from `results/<lang>.json`'s `measureFiles`; they agree with `results/<lang>.json`. Match rates, file-node counts, `gate83`
verdicts and the noise-flag figures are read as written in `results/<lang>.json` and `adjudication/<lang>.noise.json`.

## §a1 Frontend fitness: fallback recall, precision, fan-out (S27)

### a1.1 Basis and disclosures

- **Truth is `final2`**, byte-identical to `final` for all six measured languages: no overturn was found. Python's 6 wrong-resolved
  sites were read against source and all are Joern errors. The other languages' miss truth was spot-checked, not read
  exhaustively; **the error rate of the data-scientist's own source readings is not estimated** (the analyst's independent
  re-reads of 15 rows found no label error).
- **Headline = as-shipped Joern, strict ML §7.1 `found`.** The `override` runs (test/i18n path segments renamed; not the product
  configuration) and the lenient name-match figures are labelled secondary. Config `on` equals `off` **by construction** for
  Python, TS/JS and C++ (two parses of one input, 0 discordant sites): the paired test and "worse of the two configurations" are
  vacuous there, so McNemar p = 1.0 is not evidence of "no gap". For Go, Java, C# the gap is measured and is also 0 on recall.
- **Rule change after the results.** ML §2 ("site found") credited name matching without a Joern `CALL` while §7.1 counted an
  absent site as not found: the two contradicted each other. ML v1.14 follows §7.1. The amendment was made after the S27 results were
  seen. It changes no pass/fail outcome (all but Python fail 0.98 either way) but it **decides two §8.1 remedies**:

  | Language | Misses strict / lenient | Reference closes strict / lenient | Remedy strict | Remedy lenient |
  |---|---|---|---|---|
  | TS/JS | 69 / 36 | 50 / 17 | indexer (recall) | recognizer work |
  | C++ | 132 / 80 | 67 / 24 | indexer (recall) | recognizer work |
  | Java | 63 / 9 | 49 / 5 | indexer | indexer |
  | C# | 69 / 43 | 56 / 40 | indexer | indexer (reference reaches exactly 147/150 = 0.980) |
  | Go | 34 / 22 | 10 / 2 | recognizer work | recognizer work |
  | Python | 3 / 3 | 0 / 0 | recognizer work (recall); indexer on precision | same |

  33 of TS/JS's 50 closed misses are JSX component tags, for which jssrc2cpg emits no `CALL` by design. The remedy is therefore a
  property of the rule reading as much as of the frontend.
- **Kotlin: unmeasured (not a fail).** Joern's default-overlay stage aborts on the corpus; no metric, no floor. No §8.1/§8.2/§8.3
  verdict and no `flowAvailable` — not decided by any rule. kotlin2cpg does write a frontend-only CPG (2,824 methods), so the
  language is not "cannot build".
- **Contact stratum.** The decision for Python and TS/JS is the non-contact stratum with the plan's definition (A: the site's own file is not
  in `contact.json`); the sensitivity is in §a1.2.
- **Java §8.3.** As shipped, precision is unmeasured (n_P 47 off / 50 on, below 60) and triggers no adoption; the override line
  (99/108 = 0.917, secondary) would be adopted on precision. The as-shipped 47/47 comes from sampling only the 31% of sites Joern sees.
- **Ground-truth noise (ML §5.4 step 7).** `flagByVersion` per language (noise flag figure; flagged if above 0.05), and the rule was
  revised twice after results:

  | Language | v1.10 (e_all) | v1.11 | v1.12 (e_res, in force) |
  |---|---|---|---|
  | Python | 0.075 flagged | 0.042 | 0.000 |
  | TS/JS | 0.000 | 0.000 | 0.000 |
  | Go | 0.027 | 0.027 | 0.000 |
  | Java | 0.004 | 0.004 | 0.000 |
  | C# | 0.000 | 0.000 | 0.000 |
  | C++ | **0.262 flagged** | 0.246 flagged | 0.000 |
  | Kotlin | 0.065 flagged | 0.021 | 0.000 |

  Under v1.12 no language is noise-limited. Python (v1.10), C++ (v1.10, v1.11) and Kotlin (v1.10) were flagged by earlier versions; C++'s
  e_all of 0.262 (macro-argument labels) is disclosed and every C++ figure here inherits it. **C++ final labels rest on a single
  data-scientist reading** (90 adjudicated, 86 audited, each decided once); the analyst's 18 independent reads found 0 wrong (Wilson upper about
  0.18). The reference's `relaxed` macro-argument matches were 4/46 wrong (0.087, Wilson 0.034–0.203).
- **Fan-out definition.** `nm` counts METHODs named like the callee; for a constructor site only METHODs named like the type, so Java/C# `<init>`
  are not counted and constructor sites show fan-out 0 while recall credits them by type name. The census covers only Joern-matched sites
  (31% of Java frame sites, 16% of C++, 75% of C#): "cost: no" is a statement about the parsed part.

### a1.2 ML §8.5 table, as-shipped (headline), strict `found`, with the lenient counterpart

R = strict recall pooled (k/n, Wilson). Off and on equal on recall. "Lenient" = ML pre-v1.14 reading (secondary).

| Language | R off / on | R lenient | Miss causes (strict, v2) | Fan-out p90, share >5 (off; on equal) | P off / on | Indexer: adopted (recall / cost / precision), strict, with lenient remedy | Fold or list | `flowAvailable` |
|---|---|---|---|---|---|---|---|---|
| Python | 152/155 = 0.9806 [0.945, 0.993]; decision stratum below | 152/155 | function value 3 | 5; 866/8,801 = 9.8% | 110/116 = 0.948 [0.892, 0.976] / same | recall: no (reference closes 0 of 3: recognizer work, function value); cost: no; **precision: yes**, one correct site from flipping (111/116 = 0.957 would pass) | list | pending S28 |
| TS/JS | 115/184 = 0.625 [0.553, 0.692]; decision stratum 96/165 = 0.582 [0.506, 0.654] (with the reference 146/165 = 0.885) | 148/184 = 0.804 | absent 14, function value 18, JSX no-CALL 33, different name 4 | 0; 0/360 | 102/102 [0.964, 1.0] / same | recall: **yes** (165/184 = 0.897 with the reference; closes 50 of 69), lenient: **recognizer work**; cost: no; precision: no | fold | pending S28 |
| Go | 112/146 = 0.767 [0.692, 0.828] | 124/146 | function value 13, absent 12, other 9 (named-interface specs 2, no-CALL 7) | 2; 200/7,581 = 2.6% | 108/108 [0.966, 1.0] / same | recall: no (122/146 = 0.836; closes 10 of 34); recognizer work for function value 13/34 and absent 12/34; lenient: same; cost: no; precision: no | fold | pending S28 |
| Java | 51/114 = 0.447 [0.359, 0.539] | 105/114 = 0.921 | absent 63 (`src/test` excluded) | 0; 1/797 | 47/47, 50/50: unmeasured (n_P < 60) | recall: yes (100/114 = 0.877; closes 49 of 63); lenient: yes; cost: no; precision: unmeasured | list | pending S28 |
| C# | 81/150 = 0.540 [0.460, 0.618] | 107/150 = 0.713 | absent 57, other 12 | 6; 1,788/16,075 = 11.1% | 22/25 = 0.880 [0.700, 0.958]: unmeasured | recall: yes (137/150 = 0.913; closes 56 of 69); lenient: yes (0.980 exactly); cost: no; precision: unmeasured | list | pending S28 |
| C++ | 20/152 = 0.132 [0.087, 0.195] | 72/152 = 0.474 | absent 74, macro 49, other 8, different name 1 | 5; 299/3,538 = 8.5% | 5/5: unmeasured | recall: **yes, but fragile** (87/152 = 0.572; closes 67 of 132, one site over half; lenient: **recognizer work**, 24 of 80); cost: no; precision: unmeasured | list | pending S28 |
| Kotlin | unmeasured | — | — | — | — | not decided | list (default) | not applicable |

Caveats that bind every "indexer adopted" cell (ML §8.1): the reference counterfactual is biased upward because the reference is half of the
truth, and the reference ran with dependencies, so an indexer's offline efficacy is unmeasured. C++'s adoption is exclusion-driven: the
74 absent-in-Joern misses are almost all test-tree files; under the override the same rule gives 17 of 74 closed, so recognizer work, and the largest
class no staging fixes is macro (49 as-shipped rows).

**Python's recall verdict is one site from flipping and rests on the stratum definition.** The decision stratum (A: the site's own file not in
`contact.json`) has 3 misses in 149 sites against a budget of 2 (146/149 = 0.9799 [0.942, 0.993]): fail. Counting a site as contact if its own
file *or any file of its truth set* is a contact file gives 143/145 (2 misses, budget 2): pass; truth-file-only gives 144/146: pass. Pooled 152/155
passes. Python's indexer outcome is driven by the precision point estimate (110/116 = 0.9483 < 0.95); a single extra correct site (111/116 = 0.957
[0.903, 0.981]) would flip it. TS/JS fails under every definition (A: 96/165; B and C: 87/152; contact stratum 19 sites, below 50).

Lenient recall for the five failing languages is at most 0.921 (Java), so the §8.1 screen outcome (fail) does not depend on the reading.
Share of fan-out above 20 (cost trigger): Python 136/8,801, C# 907/16,075, C++ 194/3,538, others 0; no p90 above 6.

### a1.3 Override runs (secondary; renamed `test`/`i18n` path segments)

Miss causes for the override runs were **not** adjudicated.

| Language | R off / on (strict) | Precision off / on | Indexer on recall (strict; lenient) | Note |
|---|---|---|---|---|
| Java | 105/114 = 0.921 [0.857, 0.958] | 99/108 = 0.917 [0.849, 0.956] / 102/111 = 0.919 [0.853, 0.957] | yes (110/114 = 0.965, closes 5 of 9); yes | below 0.98; **precision adopted** (0.917, n_P ≥ 60) |
| TS/JS | pooled 126/184 = 0.685 [0.614, 0.748]; decision stratum 107/165 = 0.648 [0.573, 0.717] | 103/103 [0.964, 1.0] | yes (closes 39 of 58); lenient: recognizer work | fail |
| C++ | 78/152 = 0.513 [0.434, 0.591] | 24/25 = 0.960: unmeasured | no (95/152 = 0.625, closes 17 of 74); no | fail; recognizer work |

### a1.4 Why files are missing from Joern (fitness facts)

Joern 4.0.579 frontend jar constants (image `ccg-toolchain:a82faf5219b3`) and `results/frontend-defaults-probe.json`: for each default-exclusion case
the shipped path parses to 0 methods and the same file under a neutral path parses; the frontends offer `--exclude`/`--exclude-regex` to add but no way to
re-include. In each case the set of missing frame files equals the predicted set exactly.

| Language | Frame files without a FILE node | Mechanism | Class |
|---|---|---|---|
| TS/JS | 5 of 40 (`src/i18n/*`, 18 sites) | jssrc2cpg default ignored segments `i18n`, `test`, `tests` | frontend default exclusion (production code lost) |
| Java | 15 of 64 (every `src/test`, 2,559 sites) | javasrc2cpg default ignored segment `test` | frontend default exclusion |
| C++ | 71 of 132 (21,710 sites) | c2cpg default ignored segments `test`, `tests` | frontend default exclusion |
| Go | 3 of 112 (`rfc9457.go`, `binder_test.go`, `middleware/csrf.go`) | per-file AST-creation exception (`astForCaseClause`; `astForTypeSwitchStatement`), reproduced alone | frontend defect |
| C# | 69 of 547 | "Unable to pre-parse C# file" and AstCreationPass exceptions | frontend defect |

These exclusions contradict the frame and bundle spec (tests are ordinary files), not the truth. Losses inside parsed files: JSX component tags in TS/JS (33
internal sites, no `CALL`), Go generic instantiation `F[T](x)` and calls inside statements gosrc2cpg logs as unhandled (`go`, `defer`, `select`, send)
(7 sites; the statement-kind mechanism is a hypothesis), some C# target-typed `new`/chained calls (hypothesis), and eight C++ rows whose target METHOD
is missing in a parsed file (locator versus emission not separated). Go's two "other" interface rows are named-interface method specs (`router.go:49`, `:40`)
which are symbols under bundle rule 8 but get no METHOD in gosrc2cpg; P1 will meet this on every interface call.

### a1.5 Not computed here

Method coverage, line fidelity, value-set recovery and decorator shapes (ML §7.2) need the census of §a1.6, a sub-unit of S27 (S27c); S30 waits on it.
Egress R7 evidence from the off runs was not examined. "Ineligible as held-back" (ML §3.3 reverse rule): the five external corpora, Go echo, Java
spring-petclinic-rest, C# Ocelot, C++ googletest and Kotlin KtorAdmin.

### a1.6 Spec for the ML §5.3 definitions census (S27c; for a coder, then graph-dba)

The metric definitions are ML §7.2 (parse success, method coverage, line fidelity, value-set recovery, decorator shape); this section carries the operating rules the
coder needs, and the rule sources it must read. **Sequence:** (1) `make_defs_census.py` (no Joern); (2) `measure.sc` change by graph-dba, analyst-gated; (3) rerun of the 12
builds (six languages x off/on; Kotlin frontend-only, below); (4) `s27_census_metrics.py`; (5) fill the four rows of §a1.2 and the S30 shapes. The rerun is needed because the
current `*.measure.jsonl` carries no argument or annotation data. All code lives in `cpg/spikes/frontend-fitness/`.

**1. Definitions census, `truth/<lang>.defs.jsonl`** (frame files only, the frame's file rules and grammars).
- `{"k":"def","path","name","nameKey","nameLine","endLine","declStartLine","symbol","decorators":[{"name","node","literalArgs":[{"name":str|null,"value":str}]}]}`.
  `nameLine` is the line of the name (decorators and annotations excluded); `declStartLine` includes them. `symbol` is set by the per-language table of rule 8 in the bundle
  spec in force (`docs/plans/connected-code-graph-bundle-spec3.md` §4.1, as amended by `-spec4.md`); a bodyless non-symbol is kept with `symbol:false` and left out of coverage.
  Each rule-8 case (the lists in ML §2) gets a planted fixture and an expected `symbol` value in the tests.
- **`nameKey`** is the short name: the last segment after `.`, `::` or a Go receiver; template and type arguments stripped; operators as written (`operator==`), destructors with `~`;
  constructors are `<init>` and the type name (Java/C#/Kotlin/C++), `__init__` (Python), `constructor` (JS); overloads share the short name and are told apart by source order. The accepted
  Joern names per frontend come from `matcher-table.md` and `s27_metrics.py` `spec_names` (the existing rule source; the coder extends it, never redefines it). A census definition with
  no name (arrow function, lambda, anonymous class) matches by position only: a METHOD named `<lambda>N`, `<anon>…` or `:program`-nested whose start line is the definition's.
  JS object-literal and class-field functions take the property key as name; Python nested defs their own name.
- `{"k":"arg",…}` is one record per call argument that is (a) a string literal, (b) a literal-only concatenation, (c) a substitution-free template literal, or (d) an identifier that
  resolves to a same-file constant: a file- or class-scope binding (`final`/`static final`/`const`/`constexpr`/`val`/Go `const`/module-level `UPPER_CASE`) initialised by (a)–(c), with
  no local of the same name in the enclosing function (a shadowing local disqualifies the argument). Fields: `path,line,column,callText,argIndex,argName,valueKind,value`.
  `argIndex` is 0-based over the explicit arguments, the receiver excluded (Joern's argument 1 is the receiver for method calls: subtract it); a keyword or named argument keeps its
  positional index and sets `argName`; spread arguments are skipped. `value` is normalised: decoded escapes, delimiters and prefixes stripped, concatenation folded, template text only.

**2. `measure.sc` additions** (graph-dba): per non-external METHOD, its annotation/decorator names and literal arguments; per `arg` record (joined by `(file, line, column)`), the Joern
argument's node kind, its `Literal` value if it is one, and the value of the single reaching definition when that is a `Literal`. **Kotlin:** the metrics script reads a flag in the `meta` record; for
the frontend-only CPG (`importCpg(cpg, name, false)`, no overlays) coverage, line fidelity and annotations are measured, value-set recovery is recorded as unmeasured, and results are labelled
"frontend-only". Whether `measure.sc` tolerates an overlay-free import is graph-dba's check before the rerun.

**3. `s27_census_metrics.py <lang>`** writes `results/<lang>.census.json` with an `off` and an `on` block, every metric as `{k, n, p, wilson95}`.
- **Matching (one-to-one):** per file, definitions in `(nameLine, source order)`. Candidate METHODs are those in the file not yet claimed, whose name satisfies the name rule, and whose
  start line equals `nameLine` or `declStartLine` at the frontend's offset (`METHOD_LINE_OFFSET`: C# -1, others 0); failing that, the innermost unclaimed METHOD whose span holds
  `nameLine` within 8 lines of its start. Prefer the exact start; ties by smallest line distance, then source order. A METHOD is claimed once; a definition with no candidate is uncovered; files with no FILE node are uncovered.
- **Line fidelity:** exact for every language (ML §7.2): `lineNumber` equals the name-line start (variant: the decorator-inclusive start), and `lineNumberEnd` equals the census end line. No
  "span and name" rule.
- **Parse success:** frame files with a FILE node and at least one non-external METHOD ÷ frame files. Both configurations are computed separately.
- **Value-set recovery:** an `arg` record is recovered if the matched Joern call (§7.1, same siteKey) has that argument as a `Literal` with the same normalised value, or if the argument has exactly one reaching
  definition which is such a `Literal` (needs the dataflow overlay). An `arg` whose call is unmatched counts as not recovered.
- **Decorator shape:** the denominator is definitions with at least one decorator, annotation or attribute node. Name share: Joern exposes the decorator name, compared on the last dotted segment
  (`app.route` equals `route`; Java qualified names reduce to the simple name), case-sensitive. Literal share: over decorators with at least one literal argument, all literal values present. The tree-sitter node
  per language: Python `decorator`; TS/JS `decorator`; Java `annotation`/`marker_annotation`; Kotlin `annotation`; C# `attribute`; C++ `attribute_declaration`; Go none. The Joern node that carries it (ANNOTATION on the METHOD,
  or a call for Python decorators) is unverified: the coder records it from a one-file probe per language, in the format `{lang, node, example}`, before step 2. One worked example per language is a deliverable for S30.

**Acceptance.** (a) One `def` record per tree-sitter definition in every frame file, counts published; (b) every `(path, defLine)` that is a `T` member in `final2` or a reference `static`/`impls` target with a body appears in
`defs.jsonl` with `symbol:true`; (c) 3 hand-counted files per language; (d) per language, a fixture with one planted definition per rule-8 case, one argument of each kind (a)–(d), and one decorator with a literal argument,
each with its expected record; (e) mutants that must fail the tests: drop concatenation folding, drop `marker_annotation`, drop the shadowing check, flip the receiver offset, drop the C# line offset.

## §a2.0 Flow-label audit (Part F)

Protocol `docs/plans/connected-code-graph-p0-ml.md` §5.4 Part F. Decisions: `cpg/spikes/frontend-fitness/adjudication/<lang>-flow-audit.jsonl` (one record per
queued pair and diff row, with file:line and convention tag); corrected sets `truth/<lang>.flow-final.jsonl` (provenance field on added pairs, `partFRemoved` on
methods that lost pairs). Judged from corpus source and the flow brief only. All 400 queued pairs and 61 diff rows are decided; no extension was needed (every
language has at least 20 valid pairs in the audited methods). The draws were recomputed from the golden files and equal the queues, the F2 method sets and the
relabel files for all seven languages.

| Lang | F1 n | invalid (unsure drawn / invalid) | F1 flag | F2 agreed | golden-only valid | relabel-only valid | omission | F2 flag | final pairs (removed / added) |
|---|---|---|---|---|---|---|---|---|---|
| python | 60 | 3 (20 / 3) | no (0.050) | 84 | 0 | 0 | 0/84 | no | 480 (3 / 0) |
| javascript | 40 | 0 (18 / 0) | no | 54 | 9 | 0 | 0/63 | no | 151 (0 / 0) |
| go | 60 | 4 (11 / 4) | **yes** (0.067), adjudicated with second labeller | 85 | 1 | 1 | 1/87 = 0.011 | no | 433 (4 / 4) |
| java | 60 | 0 (5 / 0) | no | 124 | 3 | 4 | 4/131 = 0.031 | no | 465 (0 / 4) |
| csharp | 60 | 2 (20 / 2) | no (0.033) | 62 | 6 | 0 | 0/68 | no | 392 (2 / 0; 5 coordinate-corrected) |
| cpp | 60 | 2 (20 / 2) | no | 74 | 2 | 5 | 5/81 = 0.062 | **yes**, adjudicated with second labeller | 349 (13 / 18; 2 coordinate-corrected) |
| kotlin | 60 | 0 (20 / 0) | no | 74 | 0 | 1 | 1/75 = 0.013 | no | 477 (0 / 1) |

Every one of the 11 invalid F1 pairs was `unsure` in the golden set; 0 of the 286 `sure` pairs drawn were invalid (Wilson 95% upper bound 0.013). Agreed pairs
(present in both sets) are counted valid without a separate read; an invalid agreed pair would shrink the denominator and raise the omission rate slightly.
F2 also found golden-only pairs that are invalid: cpp 3 (`IsValidParamName`, early exit reaching points nested in a later loop), removed in the final set;
csharp 0; relabel-only invalid: go 0, csharp 1, kotlin 1 (none added). The invalid pairs by cause: go `callee-value` 3 and `field-write` 1; python `callee-value` 2
and a generator-expression source 1; csharp `out-arg` 2; cpp `this-field` 1 and `stream-into-call` 1, plus cpp ordinal 5 pair 1 (below).

**Coordinate class (coordinator ruling).** A wrong `line` on an otherwise valid endpoint is a closed coordinate class (tripwire, not a validity error) and is corrected
in place in the final set, with `provenance: partF-coordinate-fix`: csharp ordinal 4, pairs 3 to 7 (switch-arm cond lines 119/120/121 corrected to 120/121/122; pair 6 was
not queued) and cpp ordinal 5, pairs 2 and 7 (`GetInstance` 691 to 692). The csharp F1 is therefore 2 invalid of 60 (the two `out-arg` pairs), not flagged.

**Escalation (ML §5.4).** Flagged: go F1 and cpp F2; each got a fresh second blind labeller (all 30 methods, `adjudication/<lang>-flow-second.jsonl`), and the
adjudicated union is the golden set (`truth/<lang>.flow-final.jsonl`; provenance `partF-second-only-valid` on added pairs). Python sits at exactly 0.050 (3 of 60, not above).
Python's relabel reproduced all 84 pairs of its six methods with no difference, which the transcript check (relabel before `labels/` opened) should confirm.

**Second-labeller adjudication** (every one-sided pair judged from source, records `part: F-second` in `adjudication/<lang>-flow-audit.jsonl`). A pair differing only by a wrong
`ret` line in the second set (go 4, cpp 2 of `GetInstance`) is matched to the golden pair, not one-sided.

| | go | cpp |
|---|---|---|
| second-set pairs / `unsure` | 419 / 9 (2.1%) | 344 / 63 (18.3%); golden has 61 |
| agreed (incl. coordinate-matched) | 413 | 329 |
| golden-only valid / invalid | 20 / 0 | 3 / 12 |
| second-only valid / invalid | 3 / 3 | 15 / 0 |
| disagreement (one-sided / union) | 26 / 439 = 5.9% | 30 / 359 = 8.4% |
| final pairs | 433 | 349 |

Of the cpp second-only valid, 2 were already added from the custodian relabel; of the 12 golden-only invalid, 5 were decided earlier. Golden-only valid pairs are the second
labeller's omissions, mostly the early-exit reading (go: later points after several exits; cpp: `continue`/`return` in loops). The go second-only invalid are an index used as
callee selector, a map key, and a flow through two `append` calls.

**Where the second labeller contradicts a DS ruling.** Go: it independently labelled all 4 pairs ruled invalid (3 `callee-value`, 1 `field-write`), each `unsure`, as the golden
labeller did, so two independent labellers disagree with the `callee-value` and `field-write` rulings; both flagged them `unsure`, and the rulings stand on the brief's
"argument or receiver" text and are tagged for reversal. Cpp: it also labelled `mock_obj` -> `cond@688` (`index-as-selector`, `unsure`), though it took method 5's parameters from the
coordinator's brief; that ruling stands. It does not contain any of the 12 cpp pairs ruled invalid for `stream-into-call`, `this-field` and `nested-block`, which supports those rulings.

**Rulings (once, for all languages; the brief's text is the authority).**
1. *Switch-case discriminant* (JS, C# switch expressions, C++ `switch`, Kotlin `when`, Java): the case label is the test of the subject against the label, so the
   subject read by the label cond is a data pair; `code` is only the label text. The `default`/`else` body is decided by the nearest preceding label, and a later
   label by the one before it. Without this, JS has 13 of 40 invalid; the relabeller's omission of the same 9 pairs is the other reading.
2. *Short-circuit operand* (Go, Python, Java): an operand whose right side contains a call is a cond that decides every call in that right side, nested ones
   included, unless a nearer cond lies between.
3. *Early exit* (C#, Go, C++, others): after `if c { return | raise | throw }` every later point in the method that the exit can skip is decided by `c`, also
   past an enclosing loop and across several exits (the brief's "including later returns"); `break`/`continue` decide the rest of the loop body. Points nested in
   the body of a later conditional or loop belong to that structure's nearest cond, so the early exit does not decide them (cpp `IsValidParamName`). A loop header
   decides all its body points. Handler bodies of `try`/`catch` are plain nested blocks. These pairs are multi-step in Joern's control dependence; S28 should report
   the `early-exit` tag as its own stratum.
4. *Discard arm `_`* (C#): a case label, so a cond; its body is decided by it, and by the previous arm only through it.
5. *Declared-type prefix in `cond.code`* (Java): cosmetic; pair identity ignores it (12 pairs of `JdbcOwnerRepositoryImpl.delete` match).
6. *Property read* (Kotlin): a property of a held value, including the extension receiver `this` (param -1), is a field read and flows directly; it is not a call
   site. Flow through the implicit dispatch `this` (a C++ member initialiser read back as a field) and a write into a field followed by a whole-object use are not
   (the brief's "not asked for").
7. *Also ruled:* `callee-value` (a param or local invoked as `f(...)`, or the result of a call used as callee, is neither argument nor receiver: invalid; an alias of a
   member access `o.m` called later has receiver `o`: valid); `out-arg` (a value written through an `out` argument is not the call's result: invalid); C++ stream `<<` is a
   language operator (operands flow to its result; a streamed operand does not reach the call on its left); macro invocations are call sites; a slice bound is an operand, a
   subscript key only selects (`index-as-selector`: invalid); a source never equals its sink even across loop iterations.

**Pairless methods.** Python ordinal 11 (`fake_make_http_transport`: only a nested `def` and `return _transport`) and Go ordinal 13 (`FormFieldBinder`: `return vb` of
a composite literal; `c` used only inside function literals) have no pair; the custodians' `noPairReason`s stand.

**C++ ordinal 5.** The sample span (685-698) starts at the macro `GTEST_LOCK_EXCLUDED_`; the signature is at `googlemock/src/gmock-spec-builders.cc:683-684`. The only
param pair, `mock_obj` -> `cond@688`, is invalid (`mock_obj` is the subscript key at 687; `file`/`line` reach only field writes), and it is removed from the final
set. S28 must match the param points of this method, and of any method whose sample span omits the signature, on the corpus declaration line, not the span. Its `GetInstance`
line (691, on 692) is corrected under the coordinate class above.

**Coordinate scan.** The scan (call name or cond code present on the recorded line) of all seven golden sets found wrong lines only in csharp ordinal 4 and cpp ordinal 5,
both corrected above; the other languages had none. The Java pair key has no separator (ML §5.4 Part F records it).

## §a2.1 Flow projection fitness, S28 run (provisional; before the data-scientist's adjudication)

Run by `graph-dba`, 2026-11-05. Scripts: `cpg/spikes/frontend-fitness/{s28_run.py, s28_core.py, s28_metrics.py, s28-cpg-stats.sc, s28-extract-bodiless-guard.sc}`; outputs
`results/projection-<lang>.json` (+ `.answers.jsonl`), `adjudication/<lang>-flow-misses.jsonl` (every pair missed in either configuration; `proposedCause` empty),
`adjudication/<lang>-flow-spurious.jsonl` (seeded sample; `verdict` empty). Every figure below is read from those files.

**Basis.** `extract.sc` (S11, unedited) ran through `offline-run.sh --mode offline`, after the D15 content check (`cpg-content-check.sc`, `=ok` for all twelve CPGs), on the CPGs S27
built, off and on (`on` is a network-parsed CPG; the projection itself is offline). Truth = `truth/<lang>.flow-final.jsonl` (python 480 pairs, javascript 151, go 433, java 465, csharp 392,
cpp 349; ordinals 1..30, JS 1..12). Headline = as-shipped, pooled over all labelled pairs including methods Joern never produced. Off and on agree on every pair for all six languages (0 discordant).
Join and point rules are in each result's `definitions`; those the protocol leaves open: a cond matches on line and code ignoring all whitespace (Joern prints `(int,float)` for `(int, float)`);
a call matches by (name, ordinal) only if on the labeller's line, else the one same-name call on that line (`ordinalShift`); csharpsrc2cpg lines are 0-based (callee found on l+1 for 3,516 of 4,000 calls) and are shifted +1;
a qualified c2cpg constructor name is cut to its last segment; calls whose name is absent from the source are synthetic and, with their edges, are not points (as `normalize.py`).

**Kotlin: unmeasured.** The default-overlay stage aborts (ReachingDefPass, `key not found: MethodRef`) in the content check and in `extract.sc` itself (rc 1, no fact written). The only CPG is frontend-only; projecting
it needs `--nooverlays`, a pipeline deviation, and such a CPG has no REACHING_DEF or CDG to project. No `flowAvailable`.

**C#: as-shipped `extract.sc` aborts, both configurations** (0 of 5,897 methods emitted; `SchemaViolationException: OUT edge with label AST to an adjacent BLOCK is mandatory`, METHOD seq 6, from `m.block` on a bodiless method).
Beyond that, 1 of 5,897 METHODs has no AST parent (`m.astParent`). The C# rows below come from `s28-extract-bodiless-guard.sc`, a copy of `extract.sc` with those three accesses guarded (4 changed lines, no logic change); they are secondary and flagged `projection: extract.sc+bodiless-guard` in the adjudication files.

| | python | javascript | go | java | csharp (guarded) | cpp |
|---|---|---|---|---|---|---|
| labelled methods matched to a Joern METHOD | 30/30 | 11/12 | 29/30 | 30/30 | 26/30 | 14/30 |
| **flow pair recall, pooled** (Wilson; cluster 95%) | 307/480 = 0.640 [0.596, 0.681]; [0.564, 0.715] | 60/151 = 0.397 [0.323, 0.477]; [0.241, 0.554] | 246/433 = 0.568 [0.521, 0.614]; [0.513, 0.623] | 314/465 = 0.675 [0.631, 0.716]; [0.621, 0.729] | 227/392 = 0.579 [0.530, 0.627]; [0.462, 0.696] | 116/349 = 0.332 [0.285, 0.383]; [0.172, 0.492] |
| data / control | 191/304 = 0.628 / 116/176 = 0.659 | 36/80 = 0.450 / 24/71 = 0.338 | 145/294 = 0.493 / 101/139 = 0.727 | 150/296 = 0.507 / 164/169 = 0.970 | 150/258 = 0.581 / 77/134 = 0.575 | 65/188 = 0.346 / 51/161 = 0.317 |
| sure-only pooled | 306/459 | 59/133 | 246/425 | 310/460 | 220/359 | 114/298 |
| connected data pairs by a single edge | 189/191 | 36/36 | 142/145 | 145/150 | 147/150 | 65/65 |
| misses: not connected / endpoint-unresolved / method absent | 137 / 36 / 0 | 26 / 56 / 9 | 160 / 19 / 8 | 151 / 0 / 0 | 56 / 79 / 30 | 37 / 25 / 171 |
| recall on matched methods only | 307/480 | 60/142 = 0.423 | 246/425 = 0.579 | 314/465 | 227/362 = 0.627 | 116/178 = 0.652 |
| methods fully recovered | 2/29 | 1/12 | 1/29 | 0/30 | 2/30 | 2/30 |
| spurious pairs (projection pairs not in the labels, labelled methods) | 91/396 = 0.230 | 49/109 = 0.450 | 41/284 = 0.144 | 16/325 = 0.049 | 54/278 = 0.194 | 45/161 = 0.280 |
| spurious sample drawn / truth-incomplete flag | 50 / pending | 49 / pending | 41 / pending | 16 / insufficient (n < 20) | 50 / pending | 45 / pending |
| projection success: emitted / CPG methods | 6254/6254 | 232/232 | 2559/2559 | 441/441 | 0/5897 as-shipped; 5897/5897 guarded | 2744/2744 |
| census reach (informational, `truth/<lang>.defs.jsonl` as read today) | 4046/4046 | 127/133 | 1381/1439 | 387/543 | 3754/4639 | 1903/3490 |
| CPG MB; FILE nodes; nodes / edges per file; MB per file | 18.7; 108; 4,629 / 35,864; 0.173 | 0.78; 41; 483 / 2,952; 0.019 | 4.34; 116; 1,027 / 7,040; 0.037 | 0.60; 87; 196 / 1,037; 0.007 | 7.28; 688; 264 / 1,764; 0.011 | 3.35; 75; 1,218 / 8,132; 0.045 |
| projected points / edges per file; edges per method | 367 / 244; 4.06 | 29 / 24; 3.66 | 166 / 127; 5.47 | 24 / 17; 2.98 | 37 / 19; 2.21 | 204 / 183; 3.68 |
| symbols / call sites per file (non-synthetic methods; located calls) | 44.0 / 184.8 | 5.5 / 13.2 | 23.2 / 105.5 | 5.6 / 14.5 | 8.8 / 27.2 | 45.3 / 89.1 |
| build: parse s / extract s (projection ms; kLOC) | 11.6 / 13.8 (357; 77.0) | 6.4 / 7.9 (47; 3.3) | 9.2 / 7.5 (191; 41.6) | 8.0 / 6.8 (52; 5.7) | 39.8 / 8.2 (181; 60.6) | 10.1 / 7.2 (150; 38.9) |

Peak container memory: extract 427 to 1,278 MiB (python highest). Python and TS/JS cross-check with S14: methods emitted 6,254 and 232 in both (disagreement 0.000, below 0.05). Stopping condition (a): chain-only (2 or more edges) data connections are at most 5 of 150 (java), python 2/191, below the 10% prior everywhere. Condition (b): endpoint-unresolved exceeds 10% of pairs in javascript (56/151) and csharp (79/392), so the adjudicator splits them before any §8.4 decision; python 36/480 and cpp 25/349 are below it. cpp has no unresolved `param` endpoint, so there is no `this` offset. The unresolved endpoints are conds (switch case labels and `for`/`foreach`/`for-of` headers lowered to synthetic conditions such as `iteratorNonEmptyOrException`, `_idx_ < x.Count`) and calls Joern does not emit (Go `defer x.Close()` and computed callees, C++ `::std::get<I>()`), plus C# expression-bodied `ret`s; `detail` in the miss rows lists the projection's points on that line.

**Strata asked for by the audit** (tag = Part F `convention`, valid decisions; off = on):

| Stratum | python | javascript | go | java | csharp | cpp |
|---|---|---|---|---|---|---|
| `early-exit` | 0/5 (control; not connected) | 2/4 | 12/38 | 3/5 | 4/8 (3 unresolved: switch arms) | 10/34 (17 in absent methods; 10/17 on matched) |
| `callee-value` (valid pairs only; invalid ones were removed) | 0/1 | none | none (2 `alias`: 0/2) | none | none | none |
| Part F-ruled-invalid pairs the projection emits (in the spurious sample) | 0 | 0 | 0 | 0 | 0 | 1 |

`early-exit` recall is below the language's pooled figure in python, go (0.316 vs 0.568), java, csharp and cpp (matched methods: 0.588 vs 0.652), and above it in javascript (2/4 vs 0.397); the cells are small. C++ method ordinal 5 matched (`RegisterUseByOnCallOrExpectCall`, Joern span 683 to 698, parameters `mock_obj`/`file`/`line` at index 0 to 2, matched on the declaration span, not 685 to 698); its 10 remaining pairs include no param endpoint (the only param pair was removed in Part F) and 8 are connected.
**Frontend default-ignored directories** (own stratum, a frontend fact): cpp 0/109 pairs in 10 methods under `test`/`tests`; javascript 0/9 in 1 method under `src/i18n`; none for the other languages. Other methods Joern has no METHOD for: cpp 6 methods with no METHOD in a parsed file (62 pairs), go 1 (csrf.go, 8 pairs), csharp 4 (30 pairs, parse defects).
Excluding the default-ignored directories, pooled recall is cpp 116/240 = 0.483, javascript 60/142 = 0.423, others unchanged.

**Why recall is low (checked once, Java).** A parameter's second and later uses are not connected: Joern's REACHING_DEF for `owner` at line 218 of `JdbcOwnerRepositoryImpl.delete` runs from the earlier `owner` identifier at line 217, not from the parameter, and `extract.sc`'s `defsOf` follows an identifier's definition only when it is an assignment target. This probe (`rdprobe.sc`, scratch) is one method; its generality is unverified. Missed pairs also include values built by `new HashMap<>()` and then `put`, tuple and spread literals, `for` headers and switch labels; those mechanisms were not probed. The miss rows carry the evidence; `proposedCause` is the data-scientist's.

**`flowAvailable`, provisional (ML §8.4): false for python, javascript, go, java, csharp and cpp; Kotlin unmeasured.** Pooled recall is below 0.95 in both configurations everywhere (best 0.675), pairs exceed 120 everywhere, and C# additionally fails the precondition as shipped (0/5,897). Omission rate and the truth-incomplete flag await the adjudication of the spurious samples; no decision depends on them, since no language is near 0.95.
