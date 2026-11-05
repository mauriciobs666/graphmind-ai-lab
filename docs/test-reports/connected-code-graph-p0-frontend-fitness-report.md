# Connected code graph — P0(a) frontend-fitness report

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (M10)

Sections: §a1 (S27, this document's first section), §a2 (S28, not yet written). Protocol:
`docs/plans/connected-code-graph-p0-ml.md` (ML). Everything below was recomputed from
`cpg/spikes/frontend-fitness/results/<lang>.answers.jsonl`, `truth/<lang>.final2.jsonl` and
`truth/<lang>.ref.jsonl` by a scratch script (Wilson 95%, z = 1.959964), except the fan-out, match-rate
and file-node figures, which are read as written in `results/<lang>.json` (`census`, `fanOut`, `gate83`).
The recomputed recall and precision agree with `results/<lang>.json` for every language and configuration.

## §a1 Frontend fitness: fallback recall, precision, fan-out (S27)

### a1.1 Basis and what changed against the first run

- **Truth is `final2`**, identical to `final` for all six languages: no overturn was found (`cmp` equal;
  `validate_final.py` OK on all six). Python's 6 wrong-resolved sites were read against source and all are
  Joern errors, not truth errors. The other languages' miss truth was spot-checked, not read exhaustively.
- **Headline = as-shipped Joern, strict ML §7.1 `found`** (a site absent in Joern counts as not found).
  Lenient name-match recall and the `override` runs (test/i18n path segments renamed, results/*.override*)
  are labelled secondary. Config `on` equals `off` on recall everywhere (0 discordant sites, McNemar p = 1.0).
- **Kotlin: unmeasured (not a fail).** Joern's default-overlay stage aborts on the corpus; no metric, no floor,
  no §8 decision (not "cannot build": kotlin2cpg writes a frontend-only CPG). Consequence for ML §8: no
  §8.1/§8.2/§8.3 verdict and no `flowAvailable`; the language stays a recognizer/indexer-by-default case,
  to be decided with the stakeholder, not by a rule outcome.
- **Rule disclosure.** ML §2 "site found" was ambiguous (lenient) against §7.1 (strict); it was amended to strict in ML v1.14 after the results were seen. The strict reading was the stated intent in §7.1/§7.2 ("absent in Joern" is a miss cause), and both figures are reported; the language outcomes (all but Python fail the 0.98 screen) do not depend on the choice.
- **Java §8.3 verdict:** as shipped, precision is unmeasured (n_P 47 off / 50 on, below 60), so no precision adoption is triggered; the override line (99/108 = 0.917, secondary) would be adopted on precision. Confirmed: as-shipped is the headline, the override verdict is carried as a labelled secondary because the override is not the product configuration. Caution: the override shows that the as-shipped 47/47 comes from sampling only the 31% of sites Joern sees, not from a precision advantage.
- **Contact stratum (S27(6)):** Python and TS/JS are decided on the non-contact stratum alone (n ≥ 50 holds).

### a1.2 ML §8.5 table, as-shipped (headline)

R = strict recall, pooled sample (k/n, Wilson). For Python/JS the decision stratum is non-contact. Off and on
are equal on recall; where precision differs by configuration it is shown as off / on.

| Language | R off / on | Miss causes (strict, v2 file) | Fan-out p90 off / on | P off / on | Indexer: adopted (recall / cost / precision) | Fold or list | `flowAvailable` | Driving config |
|---|---|---|---|---|---|---|---|---|
| Python | pooled 152/155 = 0.981 [0.945, 0.993]; **non-contact 146/149 = 0.980 [0.942, 0.993], fail (budget 2, 3 misses)**; contact 6/6 (n < 50, not decisive) | function value 3 | 5 / 5 | 110/116 = 0.948 [0.892, 0.976] / same | recall: no (reference adds 0 of the 3 misses, so recognizer work: function value); cost: no; **precision: yes (0.948 < 0.95, n_P 116)** | list | pending S28 | both identical |
| TS/JS | pooled 115/184 = 0.625 [0.553, 0.692]; **non-contact 96/165 = 0.582 [0.506, 0.654], fail** | absent in Joern 14, function value 18, JSX no-CALL 33 (other), different name 4 | 0 / 0 | 102/102 = 1.000 [0.964, 1.0] / same | **recall: yes** (with reference 165/184 = 0.897, closes 50 of 69 misses); cost: no; precision: no | fold | pending S28 | both identical |
| Go | 112/146 = 0.767 [0.692, 0.828] | function value 13, absent in Joern 12, other 7, dispatch 2 | 2 / 2 | 108/108 = 1.000 [0.966, 1.0] / same | recall: no (reference 122/146 = 0.836, closes 10 of 34 < half); recognizer work for function value (13/34) and absent in Joern (12/34); cost: no; precision: no | fold | pending S28 | both identical |
| Java | 51/114 = 0.447 [0.359, 0.539] | absent in Joern 63 (file excluded) | 0 / 0 | 47/47 / 50/50, unmeasured (n_P < 60) | **recall: yes** (reference 100/114 = 0.877, closes 49 of 63); cost: no; precision: unmeasured | list | pending S28 | both identical |
| C# | 81/150 = 0.540 [0.460, 0.618] | absent in Joern 57, other 12 | 6 / 6 | 22/25 = 0.880 [0.700, 0.958], unmeasured (n_P < 60) | **recall: yes** (reference 137/150 = 0.913, closes 56 of 69); cost: no; precision: unmeasured | list | pending S28 | both identical |
| C++ | 20/152 = 0.132 [0.087, 0.195] | absent in Joern 74, macro 49, other 8, different name 1 | 5 / 5 | 5/5, unmeasured (n_P < 60) | **recall: yes** (reference 87/152 = 0.572, closes 67 of 132, just over half); cost: no; precision: unmeasured | list | pending S28 | both identical |
| Kotlin | unmeasured | — | — | — | not decided by rule | list | not applicable | — |

Lenient name-match recall (secondary): Python 152/155 = 0.981, TS/JS pooled 148/184 = 0.804, Go 124/146 = 0.849,
Java 105/114 = 0.921, C# 107/150 = 0.713, C++ 72/152 = 0.474. No language but Python reaches 0.98 even
leniently. Share of fan-out above 20: Python 136/8801 (1.5%), C# 907/16075, C++ 194/3538, others 0
(none has p90 above 20, so cost never triggers). The "share above 5" figure for S06 was not computed.

### a1.3 Override runs (secondary; renamed `test`/`i18n` path segments)

The override is an S27 probe by graph-dba, not the product configuration. Miss causes for the override runs
were **not** adjudicated; the carried-over rows are the as-shipped ones.

| Language | R off / on (strict) | Precision off / on | Indexer on recall | Note |
|---|---|---|---|---|
| Java | 105/114 = 0.921 [0.857, 0.958] | 99/108 = 0.917 [0.849, 0.956] / 102/111 = 0.919 [0.853, 0.957] | yes (reference 110/114 = 0.965, closes 5 of 9) | still below 0.98; **precision adopted** (0.917 < 0.95, n_P ≥ 60) |
| TS/JS | pooled 126/184 = 0.685 [0.614, 0.748]; non-contact 107/165 = 0.648 [0.573, 0.717] | 103/103 = 1.000 [0.964, 1.0] | yes (reference 165/184 = 0.897, closes 39 of 58) | fail |
| C++ | 78/152 = 0.513 [0.434, 0.591] | 24/25 = 0.960, unmeasured | no (reference 95/152 = 0.625, closes 17 of 74) | fail; recognizer work, causes unadjudicated |

### a1.4 Why files are missing from Joern (fitness facts)

Verified by listing the defaults in the Joern 4.0.579 frontend jars (image `ccg-toolchain:a82faf5219b3`) and by
standalone `csharpsrc2cpg`/`gosrc2cpg` runs (offline, scratch output). Treat as hypotheses until graph-dba's probe
is quoted here. In each case the set of missing frame files equals the predicted set exactly.

| Language | Frame files without a FILE node | Mechanism | Class |
|---|---|---|---|
| TS/JS | 5 of 40 (`src/i18n/*`, 18 sites) | `i18n` in jssrc2cpg's default ignored directories | frontend default exclusion (production code lost) |
| Java | 15 of 64 (every `src/test`, 2,559 sites) | path segment `test` in javasrc2cpg's default ignores | frontend default exclusion |
| C++ | 71 of 132 (every `test/` and `tests/` file) | `tests?` path segment in c2cpg `DefaultIgnoredFolders` | frontend default exclusion |
| Go | 3 of 112 (`rfc9457.go`, `binder_test.go`, `middleware/csrf.go`) | per-file AST-creation exception (ujson null in `astForCaseClause`; `None.get` in `astForTypeSwitchStatement`) | frontend defect |
| C# | 69 of 547 | "Unable to pre-parse C# file" (ujson null) and AstCreationPass exceptions | frontend defect |

Defaults are addable-to but not removable from the CLI, so P1 hits the same loss unless the tree is staged. They
contradict the frame and the bundle spec (tests are ordinary files), not the truth. Other no-CALL losses with a
parsed file: JSX component tags in TS/JS (33 internal sites, no CALL emitted), Go generic instantiation `F[T](x)`
and calls inside statements gosrc2cpg logs as unhandled (`go`, `defer`, `select`, send), some C# target-typed `new`/
extension-chain calls (hypothesis). C++ and C# "other" rows and the 6 C++ "target METHOD not emitted" cases
could not be separated into locator versus emission failure.

### a1.5 Not computed here

Method coverage, line fidelity, value-set recovery and decorator shapes (ML §7.2, §5.3) need the census below.
The decorator/annotation shapes S30 needs are therefore not yet available. Egress R7 evidence from the off runs
was not examined. "Ineligible as held-back" (ML §3.3 reverse rule): the five external corpora, Go echo, Java
spring-petclinic-rest, C# Ocelot, C++ googletest and Kotlin KtorAdmin.

### a1.6 Spec for the ML §5.3 definitions census (for a coder)

- **Owner:** `coder`/`tdd-engineer`, in `cpg/spikes/frontend-fitness/` (same tree-sitter grammars and file rules as `run_fitness.py frame`).
- **Script 1, `make_defs_census.py <lang>`.** Inputs: corpus tree `cpg/.ccg-work/corpora/<lang>/src`, the frame file list
  (`frame/<lang>.jsonl` paths), bundle-spec rule 8 symbol table. Output `truth/<lang>.defs.jsonl`, two record kinds:
  - `{"k":"def","path","name","qname","nameLine","endLine","declStartLine","symbol":true|false,"decorators":[{"name","literalArgs":[...]}]}` —
    `nameLine` is the name's line with decorators and annotations excluded, `declStartLine` includes them (the decorator-inclusive variant).
  - `{"k":"arg","path","line","column","callText","argIndex","valueKind":"string|concat|template|const","value"}` — string literals,
    literal-only concatenations, substitution-free template literals, and same-file constants initialised by one of those (ML §5.3).
- **Script 2, `s27_census_metrics.py <lang>`.** Joins `defs.jsonl` with `results` measure output (`method` records: file, name, `l`, `le`):
  method coverage (file, name, start ± 0), line fidelity (both lines equal), reported with Wilson per ML §7.2; files without a FILE node count as uncovered.
- **Needed from `measure.sc` (graph-dba):** per non-external METHOD, its annotation/decorator names and literal arguments; per frame call argument,
  whether it is a `Literal` with its value or has a single reaching definition that is one. Without it, value-set recovery and decorator shapes cannot be computed.
- **Acceptance:** one record per tree-sitter definition in every frame file; counts by language published; a test over a fixed small file per language.
