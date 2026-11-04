# Connected code graph — S25 frontend-fitness setup and S56 flow brief: gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P0, S25 / S56)

## 1. Scope and verdict

**Reviewed.** `cpg/spikes/frontend-fitness/` as committed at `77593e50` (HEAD when the review began) for Python, Java, Kotlin, C#, Go and C++, plus `cpg/tests/test_frontend_fitness.py` (106 tests at that commit), against `docs/plans/connected-code-graph-p0-2.md` rows S25 and S56 and §4.6, `docs/plans/connected-code-graph-p0-ml.md` v1.6, bundle spec3 §5.4–§5.7, and root `REVIEW.md`. `39e69934` (the TS/JS redraw, committed while I worked) also edits `README.md`, `corpora.yaml` and `run_fitness.py`; for the six languages I re-ran `verify` against the then-working tree (no problems) but cite line numbers at `77593e50`. TS/JS (sample, brief, labels, flow, the redraw) is **not** reviewed; the census rule (ML §3.1) was read as text only. `measure.sc` and `matcher-table.md` were read as code; no Joern run exists and none was made. I did not read any `labels/` against `truth/` (that comparison is S57's); I read `labels/` alone for schema and counts.

**Verdict: approve with suggestions.** No blocker. The firewall, the draws and the C5 figures hold (§3). Five majors (F1–F5) need an owner before the stage they gate: F1 before S56 labels are checked, F2 and F3 before S57, F4 before S27, F5 with F1. S56 flow labellers for the six languages may start now.

CPG: considered, not relevant — the artifact is a stand-alone Python/Scala spike harness plus data files with no loaded graph; every claim was checked by re-running the harness, an independent hash sort, Python's `ast`, and planted inputs.

**Falsifiable stopping condition (this gate closes at Pass 2 when all hold; otherwise it reopens on the named item).**
1. `flowcheck` run on the five planted label files in F1 reports an error for each of: unknown param name, param index beyond the signature, `ret` ordinal beyond the method's returns, empty-citation evidence, and reports `complete: false` for a 30-method file of empty `pairs`.
2. `truth/<lang>.c5-rows.jsonl` (or equal) exists for six languages and a test recomputes `pilot_coverage` from it to equal `c5.json` without `.ccg-work/` (F3).
3. ML §3.2 states the v1.4 classes postdate the Ocelot and googletest pilots, and eShopOnWeb's standing under the amended rule is recorded (F2).
4. A plan row or a ledger decision names the owner and the step that produces the S06 contact list and computes the non-contact stratum, for TS/JS **and** Python (F4).
5. Mutants n01, n02, n03, m20, m22, n06, n12 (F5) and the C# receiver-argument planted site (F6) are killed by a test, not only by `verify`.
Anything else found in Pass 2 at major or above reopens; minors are taken or declined by the owner without a Pass 3.

## 2. Findings

Suggested fixes are specifications: none was applied or run (I edit no source). Where a fix needs the checker's own parse, `run_fitness.py`'s tree-sitter `enumerate_defs` is already available to the harness-side checker, which the labeller-facing 'no parser' rule does not bind.

### F1 (major) — `flowcheck` accepts endpoints that S28 counts as recall misses, and calls an all-empty file `complete`

- Evidence (**verified**, planted labels on a scratch copy, `run_fitness.py:2110-2218` at `77593e50`, read in full): on method 1 of `python-flow.jsonl`, `param` with name `zzz_nonexistent` and index 7: 0 errors. `param` index 99: 0. `ret` ordinal 5 on a one-return method: 0. `evidence: "."`: 0. Thirty methods each with `pairs: []` and a `noPairReason`: `complete: true`, `nErrors: 0`. It does reject a call line outside the span, `source == sink`, and a `call` as a control source.
- Why it matters: the template (item 1 of its S28 section) joins `param` by `index`, `ret` by `ordinal`; a labeller naming error becomes `endpoint-unresolved`, "a miss in the recall count". The S56 row's done-condition is "≥ 1 pair per method and cited evidence". `complete: true` on zero pairs invites reading that as done.
- Fix: in `_flow_endpoint_errs`, require the param `name` to occur in the method `source` as a whole word; take the parameter count and the `return` count from `enumerate_defs`/tree-sitter (index ≤ nparams − 1, ret ordinal < returns, or 0 for an expression body) rather than from text; require `evidence` to contain a `:` followed by a digit. Report `complete` as `labelled == methods and nErrors == 0 and withoutPair == 0`, keeping `withoutPair` visible. Add the planted shapes above to the `test_flowcheck_rejects` table. Putting the checks in the same change as their tests is fine; the planted files are the guard's expected values and do not come from the code.

### F2 (major) — C5 amended after the Ocelot and googletest pilots, against ML §3.2's own sentence

- Evidence (**verified**, ML §3.2, `corpora.yaml` `rejectedCandidates`, ledger 2026-11-04 "S25 rulings"): §3.2 says "A class is added only by a revision of this note made before the pilot it would apply to" and "the tool is never swapped to rescue a corpus". The C# and C++ classes arrived in v1.4 after both pilots (0.86 and 0.56) were recorded. `dotnet-architecture/eShopOnWeb` was rejected on C5 ("40/50; all 10 silent sites were generic-named callees … or `nameof`") — under v1.4 that is 40/40.
- Fresh evidence I computed (**run**, `refmap_scip` on the retained `index.json`, prefixes `ccg-p0a-v1-c5b|`, `c5c|`, `c5d|`): C# adjusted 1.00, 1.00, 1.00 (raw 0.84, 0.88, 0.88); C++ adjusted 0.8936, 0.9565, 0.9149 (raw 0.84, 0.88, 0.86; strict 0.52, 0.78, 0.62). The 500-site sample gives C# 422/426 and C++ 427/462 (re-derived from `ref.jsonl`). So the substance holds for C#; C++ at pilot size is a coin flip around 0.90, as ML already says.
- Fix: ML §3.2 gets one sentence recording the amendment as a ruled exception to its own timing rule, and `corpora.yaml` re-states eShopOnWeb (and leveldb) under the amended rule with the outcome. If `data-scientist` wants the exception cleaner, a fresh-seed pilot per rescued corpus (the code path exists) is cheap and pre-Joern.

### F3 (major) — the C5 pilot's per-site answers are not in the repo, so C5 cannot be audited from committed files

- Evidence (**verified**): `truth/<lang>.c5.json` holds aggregates; `c5-pilot.jsonl` holds the 50 sites without answers; `ref.jsonl` covers the sample only. Pilot ∩ sample is 3 (Python), 6 (Java), 3 (Kotlin), 1 (C#), 1 (Go), 0 (C++) sites. I could re-derive the figures only by re-running `refmap` on gitignored `cpg/.ccg-work/frontend-fitness/truth-out/*/index.json` (up to 81 MB each). Result: `c5.json` and all four `ref.jsonl` (C++, C#, Kotlin, Go) reproduce exactly apart from `tool`/`toolVersion`.
- Why it matters: the S25 gate item is "raw and adjusted pilot coverage, excluded sites by class". On a clean clone the gate reads a number it cannot check, and `.ccg-work/` is not durable.
- Fix: `refmap --mode c5` also writes `truth/<lang>.c5-rows.jsonl` (`pilotOrdinal`, `siteId`, `kind`, `relaxed`, `ambiguous`, `typeCtorMerged`, `byConstruction`; 50 short lines). A test recomputes `pilot_coverage` from it and compares with `c5.json`.

### F4 (major) — the S06 development-contact list for the fitness strata has no owning step

- Evidence (**verified** by grep): ML v1.6 §3.1 makes TS/JS pass §8.1 only if the non-contact stratum has recall ≥ 0.98 with ≥ 50 internal sites, via "S06's contact list applied to the 40 files". `docs/plans/connected-code-graph-p0-2.md` row S06 is "acc" and lists no such output; row S27 lists "S25 results; S57 truth" and no contact input; "contact" appears (lines 92, 390, 412, 440) only for the *eval* contact set (S47, S56 and a Deps sentence). ML §10 R6 says the same of Python ("S06 is mine; I will carry it there"), but v1.6 states the stopping condition for TS/JS only, although the Python corpus (`falkor-chat/server`) is the same proving repo. The only trace of the debt is the coordination ledger line "Owed: S06 development-contact list".
- Fix: add to S27's inputs and done-condition: contact list (from S06's rule applied to both proving-repo corpora), non-contact stratum computed and reported; state the Python treatment in ML §3.1 (same rule, or a ruled exemption).

### F5 (major) — eight mutants survive the suite: values and branches that no test pins

- Evidence (**run**, §4): the suite stays green when (a) the Kotlin elvis symbol `?:` becomes `?.` or the word "elvis" is surrounded (n01, m06) or Go's `||` is dropped (n02) in `FLOW_OPS`; (b) `new` leaves `GO_BUILTINS` (n03; 5 of the 22 sampled Go builtin sites are `new`); (c) `make_adjudication_input.compare` stops adding `declared` labels to the label set (m20) or stops reading the reference's `impls` (m22); (d) a computed-path read of `results/` is added to `make_adjudication_input.py` (n06); (e) the `<anon>` exclusion leaves `flow_population` (n12). Only (a) is caught, and only by running `run_fitness.py verify` against the committed briefs.
- Why it matters: (a) is ML v1.5 ruling (b), the instruction that widens the labelled pairs; `validate_flow_brief` checks `FLOW_OPS[lang] in text`, reading its expected value from the constant it checks (root `REVIEW.md`). (c) is the S57 input generator: 44 Java, 21 Kotlin and 11 C# labels are `declared`, 11 Go sites are dispatch with `impls`, and neither path is exercised. (d) is the firewall: the static test counts bare `open(` calls and forbids literals, so a computed path passes.
- Fix: literal per-language operator expectations on the filled brief (Go and Java `&&`, `||`, no `??`; Kotlin `&&`, `||`, `?:`; C++ as Java); a `compare` test with a `declared` label against a `declared` reference and a dispatch site whose `impls` match, then differ; a Go builtin table test over the whole closed list; a flow-population test with an anonymous 6-line method with a branch and a call; forbid `read_text`/`read_bytes`/`glob`/`iterdir`/`os` in the static firewall test.

### F6 (minor) — C# `generic-callee` tags any `<` in the callee text

- Evidence (**run**): `csharp_generic_classes({"callText": "_repo.Setup(x => x.GetAsync(It.IsAny<CancellationToken>())).ReturnsAsync", "calleeName": "ReturnsAsync"})` and `"list.Where(x => x < 3).ToList"` both return `['generic-callee']`; ML's class is "a callee with a type-argument list". Today it does no harm: all 62 sample sites and 6 pilot sites tagged `generic-callee` end in a `<…>` list. A silent site like the planted ones would be wrongly excluded from C5's denominator and wrongly enter the S57 audit draw. Mutant m13 (`"<"` → `">"`) survives for the same reason.
- Fix: tag only when `re.search(r"<[^()]*>$", callText)` (or the tree-sitter generic node); add the two planted sites to `test_site_tags_csharp_and_go_classes`.

### F7 (minor) — S26 briefs name Joern three times and carry step ids; the README says once

- Evidence (**verified**): `grep -ci joern sample/{python,java,kotlin,csharp,go,cpp}.brief.md` is 3 each: the prohibition, "Joern resolves `embedder.embed(…)` to the `Protocol` stub" (line 180) and "how Joern's frontends model these is not verified here" (line 210), inside the verbatim spec excerpt, with "analyst F2(a), F4; S03 Pass 2 N1, N5". README §6 says "it names Joern only to forbid it". The flow briefs are clean: exactly one mention, enforced by `validate_flow_brief`.
- Why it matters: the first sentence tells a blind labeller how the comparison tool treats Protocol stubs, which is exactly the `declared` decision. S26 is done, so this is a carry-forward note for S57's `declared` stratum and a fix for the regenerated TS/JS brief.
- Fix: redact those two sentences in `spec_excerpt` the way `flow_spec_excerpt` does, and enforce the one-mention rule in `validate_brief`. Correct the README sentence.

### F8 (minor) — briefs embed the checkout's absolute path, so `verify` is not path-independent

- Evidence (**run**): a relocated copy of the tree fails `verify` with 12 problems, `brief differs` and `flow brief differs` for all six, nothing else; regenerating the briefs in the copy clears them. README §1 says every file is re-derived "from `corpora.yaml` alone".
- Fix: say so in README §1, or have the brief take `{CORPUS_ROOT}` etc. from an `--root` argument recorded in the file's first line and compared separately.

### F9 (minor) — figures that drifted from the harness

- **Run** (`macro_window_symbols` with the merge off, same index): pilot has 3 `typeCtorMerged` sites, not 4; without the clause it is 40/50 raw, 40/45 = 0.889 adjusted (3 ambiguous, 12 relaxed), not 39/50 and 39/44 = 0.886. On the sample, 10 sites carry the flag (matches ML), 417/500 raw, and the in-class count rises to 39, so 417/461 = 0.9046 where ML §3.2 says 417/462 = 0.903. README §2 says "11 relaxed + 3 ambiguous (39/50 raw)": 11 + 3 is 14 and the pilot has 15 relaxed with the clause. Conclusions (pilot below 0.90 without the clause, sample above it) stand. Wilson intervals quoted in ML §3.2 recompute to 0.852–0.988 (43/45), 0.896–0.945 (427/462), 0.738–0.930 (43/50), 0.810–0.873 (422/500).
- Fix: correct ML §3.2 and README §2 to the harness figures.

### F10 (minor) — stale or missing record fields

- `corpora.yaml` C++ C5 block still holds the strict figures (`withReference: 28`, `silent: 22`, `coverage: 0.56`, `silentTagged: {error-region: 4, preproc-branch: 6}`) against `c5.json` (43 / 7 / 0.86, classes 5 + 1); the C# block's numbers agree with `c5.json` (the brief's "C#/C++" stale-figure note holds only for C++). `truth/java.c5.json` and `truth/python.c5.json` lack `adjustedCoverage` and `exclusionCap*`, unlike the other four. README says 101 tests; the suite at `77593e50` has 106. Go's C8 evidence is "test files in-tree (see README)" and README has none.
- Fix: regenerate the YAML C5 blocks from `c5.json` (one test asserting equality of the shared keys), or delete the duplicates.

### F11 (minor) — `make_adjudication_input.py`: `notACall` against a resolved external reference is auto-"agreed"

- Evidence (**run**, planted three-line fixture, `--root` scratch): label `notACall: true`, reference `{kind: external, package: fmt}` gives `status: agreed, reasons: []` (`compare`, `make_adjudication_input.py:130-135`, returns before comparing external). ML §5.1 step 2 agrees only on "same in-corpus target set, or both external"; here the sources disagree on whether it is a call. Static firewall: the script has one read path (`read_jsonl`, parent dir name must be `labels` or `truth`) and one write path (`adjudication/`), and no other file I/O; I read all 212 lines. `test_adjudication_script_static_firewall` is a bound, not proof: it counts bare `open(` calls and forbids literals, so a `Path.read_text()` of a path built from `"res" + "ults"` passes it (mutant n06, §4). I did not count affected real sites (needs the label-vs-reference comparison).
- Fix: add reason `label-notacall-reference-call` when `notACall` and the reference kind is not `silent`; strengthen the static test to forbid `read_text`/`read_bytes`/`glob`/`iterdir`/`os` anywhere in the module.

### F12 (minor) — flow-brief text

- **Read**: the appended spec §5.5 lists "`catch` guards" as conditions while the per-language lists name them only for C# (`catch … when`); Java, Kotlin and Python lists say nothing, and the template says exception edges are not conditions. State "a `catch`/`except` clause with no guard expression is not a condition".
- The excerpt cites sections the labeller cannot read ("§4.1 rule 5", "§3.2 `callSiteId`", "§5.12", "analyst F4", "graph note §6.3", `[P-slice]`) and the spec's repo path; the scope notes restate rule 5, so no content is lost. Strip the tags in `flow_spec_excerpt`.
- The brief says "the output file is the only file you write", yet five S26 labellers wrote helper scripts under the shared scratchpad (ledger). For S56 name a per-dispatch scratch directory and state what a helper may do (write JSONL; not read the corpus to decide a pair or an ordinal).

### F13 (minor) — tags and informational counts

- `GO_BUILTINS` tags by `calleeName` only (a method `x.max()` that is silent would be excluded); require `callText == calleeName`. All 22 sampled Go sites are bare names, so no effect today.
- The include-guard test is "a top-level `#if`/`#ifdef` spanning more than 0.8 of the file" (`run_fitness.py:1732` at `77593e50`), a constant not in ML. Document it.
- **Run**: 59 of 500 C++ sample sites (11.8%) sit in tree-sitter ERROR regions (8 of them reference-silent); they are in the frame and are informational. README §2 says both "not in the frame" and "10.5 % of frame sites sit in ERROR nodes". Pick one.

## 3. What is solid (each **verified**)

- **Sampling.** An independent SHA-256 sort over `frame/<lang>.jsonl` with the §4.4 seed reproduces all 500 ordinals, `siteId`, path, line, column and callText for the six languages (0 mismatches each; frame sizes 17,358 / 3,693 / 7,754 / 24,414 / 13,362 / 28,490). Five ordinals per language checked against the corpus source at line:column: 29 of 30 matched the whole callText; the 30th (Kotlin ordinal 400) is a multi-line callee, whitespace-collapsed by design, whose first token sits at the stated column. The C# generic-audit order (`ccg-p0a-v1-gen|`, 74 sites = 62 + 12) reproduces independently.
- **Frame completeness (Python).** Python's own `ast` finds 17,358 `Call` nodes outside decorators in the frame's 104 files, equal per file; the three `.py` files not in the frame have zero calls.
- **Reproducibility.** `run_fitness.py verify python java kotlin csharp go cpp` returns no problems at the original path; `srcDigest` equality included. Re-running `refmap` on the retained tool outputs regenerates `ref.jsonl` for C++, C#, Kotlin and Go identically, and `c5.json` identically apart from version arguments.
- **C5 (re-derived).** C++ pilot 43/50 raw, 43/45 adjusted (5 `preproc-branch`, 1 also `macro-callee`; strict 28/50); C# 43/50, 43/43 (6 generic, 1 `nameof`); Go 45/50, 45/49 (1 builtin); Kotlin 50/50. Sample level from `ref.jsonl`: C++ 305/119/3/73 with 123 relaxed, 3 ambiguous, 10 merged, 38 silent in a class (29 + 17 − 8 both → 29 under `#if`, 9 macro-only), comparable 427/500, adjusted 427/462 = 0.924; C# 74 of 78 silent in class; Go 22 builtins of 71 silent. The cap (≤ 10) holds everywhere (max 7); a site counts once. The macro-argument rule (outermost invocation, name-equal, exactly one symbol, ambiguous stays silent) matches ML §5.2 line by line (`run_fitness.py:1682-1716` at `77593e50`).
- **Firewall.** Key sets of `sample/*.jsonl`, `*.private.jsonl`, `*-flow.jsonl` and `frame/*.jsonl` carry none of `static impls tool targets truth joern`; no value contains `joern`, `cpg.bin` or `cpg_`. The six flow briefs contain one Joern mention each and none of the 22 leak patterns; `make_adjudication_input.py` has one reader and one writer.
- **Flow brief vs spec.** Point names and numbering (`param` index/name, receiver −1 rules per language, `ret` ordinal, `cond` line + code, `call` calleeName + ordinal + line) match spec3 §5.4–§5.7; the ruled departures are stated honestly (C++ implicit `this` as the JVM's, flagged as an extension). Flow samples: 30 methods each, 5–25 lines, none `<anon>`, each with a branch.
- **Tests.** 106 pass at `77593e50` (and 143 at current HEAD); a hash-sort test derives its expectation from `hashlib` independently.

## 4. Mutation results

38 mutants run on copies of the `77593e50` tree (`CCG_FF_DIR`, scratchpad `…/scratchpad/mm/`, one-line edits; the 106-test suite taken from that commit; every copy regenerated its own briefs first so no kill is path noise); one more (m25) was skipped because its pattern did not match. **26 killed by the suite, 12 survived.**

| Area | Mutant | Result |
|---|---|---|
| sample hash | m01 input order `key|seed`; m02 seed `sha|lang`; m23 cap 501 | killed (hash-formula test, python reproduction) |
| site column | m03 off by one; m04 bytes not code points | killed (3 and 4 tests) |
| type + ctor merge | m07 clause off; m08 merge key loosened | killed (2 tests each) |
| macro rule | m10 window excludes name line; n08 outermost→innermost; n09 ambiguous ignored | killed |
| C5 arithmetic | m09 cap 12; m11 site counted per class | killed (1 test each) |
| tags | m12 `len` leaves builtins; n04 `nameof` case; n05 guard skip removed; m18 guard 0.8→0.99 | killed |
| flow brief | m05 C# `??` dropped; m16 CDG leak pattern; m17 `S57` pattern; m27 Joern count; m26 params; n11 branch requirement | killed |
| flowcheck | m14 self pair; m15 span +5 | killed (parametrised table) |
| firewall scan | m19 `static` key removed | killed |
| adjudication | m21 `unsure` reason removed | killed |
| survivors, real | m06, n01 Kotlin elvis; n02 Go `||` (all three caught by `verify` only); n03 Go builtin `new`; m20 `declared`; m22 `impls`; n06 computed-path read; n12 `<anon>` | **survive** (F5) |
| survivors, equivalent | m24, n10 duplicate-key suffix (no frame has a duplicate key; unit also unpinned); n07 dead branch (`silent` implies empty in-corpus set); m13 `<`→`>` (near-equivalent: generics have both, see F6) | not findings |

## 5. S26 deviations and the S57 design (item 7)

They do not threaten the S57 design as written; four things carry forward.
1. **Scripted labels.** Java (`lab.py`, 46 lines) is a regex indexer of type and method declarations used to emit `qname`/`defLine`; the call-target decisions are hand-written records. It is text-based, independent of SCIP, and S57 compares `(path, defLine)` only, so correlation with the reference is not introduced. But Java has **0 `unsure`, 0 `dynamic` over 500 sites** while C++ has 71 of 187 `unsure` (38%) and Python 23 of 240: S57's label-noise figure for Java should be read with that in mind, and ds should audit a seeded sample of Java externals as the ledger already asks for JS.
2. **Adjudication load.** `unsure` alone sends 71 C++ sites to adjudication regardless of the reference; three C++ labels have empty `callees` without `notACall` (they fall to `label-external-reference-internal`).
3. **Stop rule.** Four labellers (Go, C#, C++, Kotlin) wrote past the 150th internal site and cut back, Python stopped on it; all six label files validate against `validate_label`, run 1..N contiguously and carry `siteId`s equal to the public sample's (**run**, on labels alone). The last line is the 150th internal for Python, Go, C#, C++, Kotlin; Java reached 500 with 116. A hash-order prefix is a simple random sample, so inverse sampling is intact.
4. **Shared scratchpad.** The C++ labeller read two files that another agent may have overwritten (ledger); no reference or Joern content. Fine for S26; fixed for S56 by F12's third bullet.

## 6. Cross-deliverable citations (item 8)

S06 contact list: unscheduled, see F4. S47 depends on S56 (p0-2 rows S47, S56) and the `Deps` line orders them, so that citation is owned. `flow-brief.template.md`'s reference to `flowAvailable`'s S28 join is harness-facing and never pasted.

## 7. Other observations (no action required)

- Pilot optimism: pilot raw vs sample comparable share is Python 0.90 vs 0.868, Java 0.92 vs 0.834, Go 0.90 vs 0.858, C++ 0.86 vs 0.854. C5 gates the pilot; S57's comparable shares will be lower than the pilots suggest for Java and Go.
- C3: Ocelot 62,678 and googletest 75,153 non-blank lines exceed 60k; the 55.6k and 54.4k figures are cloc-style code lines of frame files only. ML C3 does not say which; data-scientist's "kept" ruling presumes the narrower one.
- `measure.sc` has only seen seven probes. Same-name chained calls (`x.foo().foo()`) map two frame sites to one Joern call (greedy nearest-column, no one-to-one assignment): 0.0–0.2% of frame sites per language, 1 sample site (Kotlin).
- `corpora.yaml` Kotlin `url` is `Amirroid/KtorAdmin`, `project` `amirroid/KtorAdmin`.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading "Always check". Source: F1 (and F5).
- A validator for hand-written labels or endpoints is probed with one planted well-formed but wrong value per field (nonexistent name, out-of-range index, empty citation) and one all-empty file; a field it only type-checks, whose value a later step joins on, is a finding.
- A pinned value in a generated prompt or brief is asserted from a literal in the test, not from the table that generates it.

## Pass 2 (S25 fix round `2b1120ce`, S59 `46cd8219`; HEAD `25964de8`)

All re-checks on a relocated copy of `git archive HEAD` (scratchpad `p2/`), TS/JS and label content excluded. Baseline: `verify` on all seven languages from the relocated path: no problems; `test_frontend_fitness.py` 246 passed, `test_contact.py` passes at the real path.

### Verdict

**S25 (seven languages, labels excluded): approve.** The five-item stopping condition of §1 holds: (1) every planted `flowcheck` shape is rejected and a 30-method empty-pairs file is `complete: false`; (2) `c5-rows.jsonl` exists for all seven, regenerates byte-identical from the retained tool outputs, and `refmap --rows-only` reports `pilotEqualsCommitted: true` for C++, C#, Kotlin and Go; (3) ML v1.7 §3.2 records the exception with a withdrawal condition (sample-level adjusted coverage below 0.90) and keeps eShopOnWeb failed, with reasons I accept; (4) ML v1.7 §3.1 and R6 name the producer and extend the stratum to Python; the plan row is still unwritten (see G3); (5) my 12 re-run mutants: 11 killed by the suite, 1 survives (p07, below).

**S59 (contact list): approve with suggestions.** `contact.json` re-derives; the open point is G1, which should be ruled before S27.

### Dispositions (one line each)

- **F1** fixed, deviation accepted. Planted on the fixed code: unknown name, index 99, index/name mismatch, `ret` ordinal 5, `evidence "."`, empty-pairs file: all rejected; a valid pair passes; a digit-less real citation ("passed to append") is advisory (`evidenceNoLine`). The digit-less rule is reasonable for `L45`-style labels. It is not noise: the committed Java flow labels carry 230 digit-less evidences and 16 evidence errors, Python and Go one method with no pair each, so `complete` is false for those three. Label content is `data-scientist`'s check, not this gate.
- **F2** fixed (ML v1.7 §3.2). **F3** fixed (above). **F4** fixed in ML v1.7; plan row S59 is not in `p0-2.md` yet (G3).
- **F5** fixed except p07: the elvis symbol, Go `||`, Go `new`, `declared`, `impls`, `<anon>` mutants are killed by named tests; the computed-path read is killed when it is a read (the new AST proof forbids `read_text` and the rest and passes the five planted variants). p07 only calls `.exists()` on a computed `results/` path; it leaks existence, not content. Accept; add `exists`/`is_file`/`stat` to `FS_ATTRS` if cheap.
- **F6** fixed, deviation accepted: "balanced `<…>` ending the callee and following an identifier" rejects the receiver-argument and lambda-comparison sites and `Foo<int>.Bar`, accepts `a.B<int>`, `new List<T>`, `F<Func<int, int>>`. My mutant p09 is killed.
- **F7** README corrected (three mentions stated); S26 briefs untouched, fine since S26 is done. **F8** fixed (relocated `verify` clean; the normaliser replaces only the checkout root, but its lazy prefix can swallow an altered path component directly in front of `/spikes/frontend-fitness/`: a nit). **F9** fixed (ML and README now 3 / 40/45 / 417/461). **F10** fixed (cpp block 43; java/python `c5.json` gained the adjusted keys; README 246). **F11** fixed (`label-notacall-reference-call`).
- **F12** not applied, accepted: the six flow label files already exist, so a brief edit now would not match what was labelled; the `catch` ambiguity stays for S28's adjudicator. **F13** documented only, accepted (no effect on data).

### New findings

**G1 (major, S59) — S56 flow-method files are outside `contact.json`, and including them nearly removes Python's non-contact margin.** EP §8.2 lists S56's golden flow methods as contact "as a flow anchor"; `contact.json` states S47 will add them. ML v1.7 §3.1 passes a language only if the non-contact stratum has ≥ 50 internal sites. **Run** (reference-internal sites, labelled prefix 1..240 for Python; TS/JS full census): Python non-contact 93 with `contact.json` alone, **52** if the 23 flow-sample files are added; TS/JS 165 → 69. Python is then two sites above the floor on the reference count; the floor is on the final truth, which differs. Fix: `data-scientist` rules now whether the call-site strata exclude flow-method files (EP §8.2 reads "as a flow anchor", and S25/S26 labels are "not contact" in the same table), and `contact.json` records the ruling either way.

**G2 (minor, S59) — sources.** The coordination ledger is not a legitimate source: it is a log, and whatever it records is in the S14/S30 documents; the same holds for ML v1.7, my review and `test_frontend_fitness.py` (synthetic snippets name `falkorchat.executor`). **Run:** deriving with those four excluded gives an identical file list for both corpora. Keeping them makes `--check` and the drift test fail on any later commit that edits them and names a new corpus file (this Pass 2 included, if it did). Recommend excluding them, or recording `derivedFrom: <sha>` and checking against that tree. Over-listing is in the safe direction: one file (`config.py`) is listed only through bundle-spec worked examples that EP §8.2 calls "not contact", and `tests/conftest.py` through a generic example sentence.

**G3 (minor) — the S59 row.** ML says `architect` assigns the step id and row; `docs/plans/connected-code-graph-p0-2.md` has none and S27's inputs do not list `contact.json`. The ledger decision satisfies the condition above; the row must exist before S27 is dispatched.

### S59 checks

- **Re-derivation:** `make_contact.py --check` exits 0 at HEAD; counts Python 23 of 107 (22 of 104 in the frame), TS/JS 2 of 46 (2 of 40); `--summary` gives internal reference sites non-contact 164 of 254 (Python) and 165 of 181 (TS/JS).
- **HEAD-blob-only:** by reading, the default path uses only `git ls-tree` and `git cat-file --batch` on HEAD; `--tree-root` reads a directory by design and is used by the mutation tests. Not traced at syscall level.
- **Spot checks.** Listed: `executor.py`, `llm.py`, `modelconfig.py` (one review probe, `docs/reviews/connected-code-graph-p0-impl.md:1413`, which cites `llm.py:81`, `executor.py:181`, `modelconfig.py:236`: genuine), `tests/test_services.py` (slice report rows 248-250), `tests/test_app.py` (`docs/reviews/connected-code-graph-rule-pack-spec.md:39`, `tests/test_app.py:835-845`), `src/api/client.ts`. Unlisted (basename search over the same sources, zero hits): `repository.py`, `storefront.py`, `trigger.py`, `db.py`, `schemas.py`; `tests/test_api.py` has hits only as `fx-svc:tests/test_api.py` in fixture-named documents, correctly ignored.
- **Mutants (8 on copies, real repo as `CONTACT_REPO`):** boundary `/`→`/:x` (c1), JS `__tests__` pattern dropped (c2), shortest-prefix dotted resolution (c3), fixtures guard off (c4), ambiguous basename (c5), self-dir exclusion off (c6): all killed, c1, c2, c3 and c5 change no real output and are caught by synthetic-tree tests. Survivors: c7 (`node_modules` exclusion, equivalent on HEAD, unpinned) and c8 (trailing `-`/`/` strip). Nits.
- **Limitations list:** no bare-symbol or directory matches, drift, S56 files: the first two are bounds on a list that over-lists elsewhere and are fine; S56 is G1.
