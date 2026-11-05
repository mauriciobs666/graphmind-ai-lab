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

Lenient recall for the others reaches no language above 0.92 except Python, so the §8.1 screen outcome (fail) does not depend on the reading.
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

**Sequence:** (1) `make_defs_census.py` (no Joern); (2) `measure.sc` change by graph-dba, analyst-gated like the earlier edits; (3) rerun of the 12 builds (six languages x
off/on; Kotlin frontend-only, below); (4) `s27_census_metrics.py`; (5) fill the four rows of §a1.2 and the S30 shapes. Steps 1 and 4 are plain Python in
`cpg/spikes/frontend-fitness/`; the rerun is needed because the current `*.measure.jsonl` carries no argument or annotation data.

**1. Definitions census, `truth/<lang>.defs.jsonl`.** Frame files only (the same file rules as `run_fitness.py frame`), tree-sitter grammars as the frame.
- `{"k":"def","path","name","nameKey","nameLine","endLine","declStartLine","symbol","decorators":[{"name","node","literalArgs":[...]}]}`.
  `nameLine` is the line of the name (decorators and annotations excluded); `declStartLine` includes them (the decorator-inclusive variant);
  `symbol` per bundle spec rule 8 for the language (a bodyless non-symbol is recorded with `symbol:false` and excluded from coverage).
- **Name key (`nameKey`):** the short name; constructors `<init>` (Java/C#/Kotlin/C++ also keyed by the type name), Python `__init__`, JS `constructor`;
  overloads keep the same short name with an ordinal `#n` by source order; anonymous functions `<anon>#n` by order within the enclosing definition;
  several definitions on one line are kept as separate records (Python has 1,081 such lines) and matched by name first, then by order.
- `{"k":"arg","path","line","column","callText","argIndex","valueKind","value"}` for each argument that is a string literal, a literal-only
  concatenation, a substitution-free template literal, or a same-file constant (`final`/`const`/`val`/module-level `UPPER_CASE` in Python) initialised by one of these.
  `value` is normalised: decoded escapes, quotes and prefixes stripped, concatenation folded, template text without delimiters; f-strings and raw
  strings with substitution are not census arguments.

**2. `measure.sc` additions** (per non-external METHOD and per frame call argument): METHOD annotation/decorator names with literal arguments; for each
argument whose `(file, line, column)` matches an `arg` record, its node kind, the `Literal` value if it is one, and the value of a single reaching definition if
one exists and is a `Literal`.

**3. Metrics (`s27_census_metrics.py`).** Coverage key: file + name + start line, using the lines the harness already fixes (`s27_metrics.py` `METHOD_LINE_OFFSET`:
C# METHOD line = definition line - 1; elsewhere exact line, else the innermost METHOD whose span holds the line within 8 lines of its start and whose name
matches); this replaces ML §7.2's "start ± 0", which would give C# near zero. Line fidelity: matched methods whose both `lineNumber`/`lineNumberEnd` equal the census
lines (Java/C++ compared on span and name, since modifiers and templates shift the start). Parse success: frame files with a FILE node and at least one non-external
METHOD (the existing `frameFilesWithFileNodeAndMethod`); the frontend's own per-file errors go to a log the runner suppresses, so they are separate evidence (§a1.4).
Both configurations, each language; files without a FILE node count as uncovered. **Value-set recovery:** an argument counts as recovered if the Joern argument is a `Literal`
with the same normalised value, or if its single reaching definition is one; "single reaching definition" is exactly one `ReachingDef` source for the argument, reached
through no call, and requires the dataflow overlay. **Kotlin:** the overlay aborts, so value-set recovery is unmeasured; coverage, line fidelity and annotations can be
measured from the frontend-only CPG loaded without overlays (`--nooverlays`), and the result is labelled as such.

**Decorator/annotation shapes (S30).** tree-sitter node per language: Python `decorator`; TS/JS `decorator`; Java `annotation`/`marker_annotation`; Kotlin `annotation`;
C# `attribute_list`/`attribute`; C++ `attribute_declaration` (`[[...]]`); Go none (record "no decorators"). The Joern node expected to carry it (ANNOTATION on the
METHOD, or a call for Python decorators) is unverified: the coder establishes it with a one-file probe per language before step 2 and records it. The deliverable includes one
worked example per language (definition, decorator, literal arguments, Joern representation) for S30.

**Acceptance.** (a) One `def` record per tree-sitter definition in every frame file, counts published per language; (b) every `(path, defLine)` that appears as a `T` member in `final2`
or a reference `static`/`impls` target with a body must appear in `defs.jsonl` with `symbol:true` (independent of the extractor: 100% or the discrepancies are listed); (c) 3
hand-counted files per language; (d) a mutation test: dropping a node type from the extractor fails (b) or (c).
