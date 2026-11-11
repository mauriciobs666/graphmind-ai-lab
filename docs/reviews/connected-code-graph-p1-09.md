# Review: C-1001 P1 unit P1-09 (projection fix and golden sets)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Reviewed the uncommitted P1-09 delivery by `graph-dba` against `docs/plans/connected-code-graph-p1-extraction.md` P1-09 (as revised by `docs/reviews/connected-code-graph-p1-09-ml.md` Pass 1 C1-C7), `docs/plans/connected-code-graph-p1-ml.md` §1 F1 and §3.4, and `docs/plans/connected-code-graph-p1.md`. Read: the diffs of `cpg/extractors/joern/extract.sc`, `cpg/spikes/frontend-fitness/s28_core.py`, `cpg/tests/slice/test_extract_equivalence.py`, `cpg/tests/test_s28_metrics.py`, `cpg/extractors/joern/README.md`; all of `cpg/tests/test_projection_golden.py`, `cpg/tests/slice/test_projection_not_regressed.py`; the goldens in `cpg/tests/golden/projection/`; `cpg/.ccg-work/p1-09/PROGRESS.md`. Not reviewed: methodology (the STOP-condition rulings, the recall figures and the dev-contact spurious rate, plan (6), are `data-scientist`'s). No Joern/JVM was run, so extract.sc was **read, not run**.

**Verdict: approve with suggestions.** No blocker. The projection code reads correct against the plan and the tests I ran pass. Two majors (M1, M2) are test-strength gaps the owner should close before the unit is committed. M3 is a design change that needs a ruling. The unit's overall acceptance still waits on the `data-scientist`'s ruling on the STOP condition (csharp 25|2, 25|6 and the 10 python relabels): the committed `RELABELLED` pin in `tests/test_s28_metrics.py` encodes the delegate's expectation, and changes if the ruling differs.

CPG: considered, not relevant — the artifact is a Scala projection script plus a Python scorer and test data; the project's CPGs are built from the shipped code, so an indexed view would not show the new, uncommitted code, and reading plus pytest settled each question.

### Executed vs read

Executed (all with `ulimit -v 1500000`/`2500000`, `timeout -s KILL`, no Joern):
- `tests/slice/test_projection_not_regressed.py`, `tests/slice/test_extract_equivalence.py`, `tests/test_projection_golden.py`, `tests/test_s28_metrics.py`: 288 passed, 6 skipped (the opt-in recompute), 11 deselected.
- `CCG_S28_RECOMPUTE=1 ... -k p1_join`: 6 passed. This is the opt-in recompute for all six languages, including the committed 29-id manifest and the pinned relabels.
- `tests/slice/test_normalize_equivalence.py -m "not live"`: 20 passed in 110 s. The stored bundles are from the old projection and the suite is green, because it hashes a snapshot of `extract.sc` and the stored facts, not the new output.
- Downstream size, my own check: `assemble_bundle.build_bundle` on the fixed facts built without error. Bundle bytes: `sp` 657,221 to 682,453 (+3.8%); `fcs` 20,224,575 to 21,928,966 (+8.4%). Records: 85,071 to 92,762 (+9%). So R1's 63% growth of `d` costs about 8% of the bundle after the normalizer drops non-point edges. Nothing is capped or fails.
- The delegate's 18 hand-run mutant fact files in `cpg/.ccg-work/p1-09/mut/`, judged by `tests/test_projection_golden.py` through `CCG_PROJ_FACTS_PY/JS`: 17 fail, 1 passes (`m16_r3js_no_temp_guard`). That matches "16 killed + survivors" for the JS guard; I found no artifact for the second reported survivor (the `startsWith(recv + ".")` mutant), so it is unverified.
- My own mutants (below), in scratch copies under `cpg/.ccg-work/gate-p1-09/`.

Read only: every `extract.sc` rule; determinism (no double run was possible); the unchanged non-projection facts beyond what the stored-fact tests show.

## Findings

### Major

**M1. The default-tier goldens and the slice tests cannot detect a change to `extract.sc`.** (verified by reading all of `tests/test_projection_golden.py` and the two slice tests)
Tier 1 judges `tests/golden/projection/facts.{python,javascript}.jsonl`, a recorded output; `extract.sc` appears in that file only inside the `live` test, which is deselected by default and needs Joern. `test_the_fixed_projection_keeps_every_non_projection_fact_of_the_slice` and `test_projection_not_regressed.py` read gitignored `.ccg-work/p1-09/fixed-*.facts.jsonl`, also produced by hand, and skip when absent. Nothing ties any of these files to the current script. The delegate's mutation runs (my replay above: m1..m17 fail) prove the goldens discriminate **if someone reruns Joern with a mutated script and points an env var at it**; the repository suite itself passes unchanged if `extract.sc` is edited after the facts were recorded. Today the timestamps are consistent (script 23:51, facts 23:58/23:59), so this is a latent weakness, not a present drift. Under root `REVIEW.md`, a check whose subject can change without turning it red is a finding.
Suggested: have the generator write `sha256(extract.sc)` beside each recorded facts file (e.g. a `facts.<lang>.sha256` or a header comment) and add a default-tier test asserting the current script's hash equals it. It is red on any byte change to the script until the facts are regenerated (live tier or the delegate's tools). Positive control: append a space to a copy of `extract.sc` and assert the test fails; this is cheap and should be asserted in the test. The same stamp for `fixed-<tag>.facts.jsonl` makes the slice tests honest about staleness, and the stat `cbRewrites` check is then tied to the script too.

**M2. The two new join rules have untested guards (plan condition C3: "one mutant per rule and guard").** (verified: four mutants survive)
Run via the repo's own `MUTANTS` mechanism (`CENSUS_SRC` copies of `s28_core.py`, `-k "not mutant"` of `tests/test_s28_metrics.py`); each of these **passes the whole existing suite**:
- `its = [c for c in online ...]` changed to `M.conds` (the iterator cond need not be on the label's line): survives.
- the paren tier's `online` changed to `M.conds` (same): survives.
- `if not ct and ... _for_header(...)` changed to `if ... _for_header(...)` (the header rule overrides a *several*-`contains` ambiguity). The existing test "a contains match is not overridden" never reaches the guard, because with one `contains` match the earlier `if len(ct) == 1: return` fires first: survives.
- `_for_header`'s `text.startswith(f"for {code}:")` loosened to `f"for {code}" in text` (a `for` inside a comprehension counts as the loop header): survives. The delegate's mutant removes the check altogether (`and True`), which is killed, but the loosened form is the realistic one.
The two line-restriction mutants matter most: they would give a pair a wrong resolved endpoint silently. Killed by (written and run in `cpg/.ccg-work/gate-p1-09/test_gate_extra.py`, using the file's own `cond_facts`/`ctl` helpers; green on the real code, red on each mutant):
1. the only iterator cond is on line 3, the label on line 2: status must be `endpoint-unresolved`;
2. a paren-equal cond only on another line: `endpoint-unresolved`;
3. conds `ITER`, `x`, `xs` on one line with label `x in xs` (two `contains`): `endpoint-unresolved`;
4. source `ys = [y for x in xs]` with an iterator cond on that line: `endpoint-unresolved`.
Suggested: add those four tests (they fail on the P0-only join as well as on the mutants where relevant) plus the four mutants to `MUTANTS`. Note the second guard's meaning is a ruling for the data-scientist too: "only when no tier matched" in the plan, versus "unambiguous" in the code (`not ct`): the code does not apply the header rule when `contains` is ambiguous.

**M3. The unplanned "4f" rule is a design change and needs a plan amendment/ruling.** (read; its goldens and mutants verified)
The plan lists the join fix (a) as scorer-side and the projection causes (1)-(4b); nothing says the `for`-statement header cond receives data from its iterable. `extract.sc` adds exactly that (`if (lang == "python") cdgSources.foreach ...`, plus `isCompIterator`). Without it, (a) resolves the endpoint but no chain reaches it: the 10 `endpoint-unresolved` to `not-connected` relabels. With it, recall can move. It is defensible: it is the projection-side half of (a), python-only, with a negative (no data into a comprehension iterator cond), and its mutants (`m15_forheader_data_off`, `m17_forheader_includes_comp`) are killed. I also checked the sibling-scan (`kids(w.astParent)`) for cross-loop leakage on the one fixture with two sequential `for` statements (`tests_test_logging_config_120.py`, lines 126 and 133): only the first cond gets data; the second, whose iterable has no def in the method, gets none, so no leakage is visible. But the plan's stop list names "a real-cause miss turning into a hit" and "a delta other than the measured +29"; this rule is a candidate for both, and the README omits it. Route to `architect`/`teco`: record it in the P1-09 text as behaviour (4f), and let the `data-scientist` rule on recall accounting. No code change is requested.

### Minor

**m1. R3 survivors: not "defence in depth" for the JS guard.** (verified from facts and mutant run)
`facts.javascript.jsonl` shows the real call `it.next()` has its receiver at argument index 0 and `_iterator_0.next()` at index 1. So the receiver-index test (`lang == "python"` 0 else 1) is what keeps the plain `it.next()` negative green: the name regex `_iterator_\d+` is not exercised by that negative, which is why m16 survives. The name regex is nevertheless the **only** thing that stops a user call whose argument 1 is an identifier equal to the receiver, e.g. `it.next(it)` (code starts with `it.`, `recv` = the argument): inferred from the index layout, not run (no Joern). Ruling: keep the guard, do not accept it as untested; add the negative `it.next(it)` (or `x.next(x)`) to `iter.ts`, regenerate, and keep m16 as a mutant that must die. I could not verify the second survivor (R3 `startsWith(recv + ".")`) because no mutant artifact exists for it; same remedy (a python negative where the receiver is a `tmpN` argument but the code does not start with it), or drop the claim.

**m2. The "no connected pair lost" carve-out is self-attested.** (verified by reading `regressions`)
A shipped `cb` edge missing from the fixed facts is excused whenever `stats.cbRewrites`, written by the same script, lists it; the test then checks only that the listed replacement is emitted. A bug in `nearestReal` (wrongly dropping an edge to a real cond) is excused by its own audit. Add an independent check: each rewritten synthetic cond's `c` record in the fixed facts has the recognised shape (code `iteratorNonEmptyOrException` or a try-body statement) and no replacement is itself synthetic. Cheap, and it removes the self-reference.

**m3. The weakened equivalence is real for deletions, blind to insertions.** (verified by simulation)
`REPLACED_SLICE_LINES` is real: I computed the slice's lines absent from the port and got exactly the 14, and editing a non-projection emit line (`"k" -> "p"` to `"pp"`) in a copy of `extract.sc` is detected (15 lines). But inserting a new line that changes a non-projection fact (`emit(ujson.Obj("k" -> "p", ..., "extra" -> 1))`) is **not** detected: the check is one-sided, so it only constrains what the slice lines became. The remaining protection is the fixed-facts test (gitignored, skips when absent) and the Docker tier. Suggested: also pin the added lines, e.g. assert the set of port lines absent from the slice is a pinned list or digest, or at minimum that no added line contains `"k" -> "<kind>"` for a kind outside `d`/`cb`/`stats`. Docstring at `tests/slice/test_normalize_equivalence.py:142` ("differs ... by its header comment alone") is now false for `extract.sc`.

**m4. JS goldens do not exercise the language argument.** (verified)
Replacing `C.build_method(self.lang, ...)` by `"python"` in a scratch copy of `tests/test_projection_golden.py` leaves all 169 passing (no JS pair uses a `param` endpoint, so `param_map` is never reached); the method-span filter and the smallest-method ordering are likewise inert (169 pass with the span filter removed; with the ordering reversed). Add one JS `param` pair (the R3 negative on `it` is a `param`-sourced pair, but it is `connected: false`, so a wrong mapping that fails to resolve is not distinguished: it asserts `got is not None`, which is the useful part; the survival shows the map is the same for index 0).

**m5. README omits 4f and the new stats keys** (`cbRewritten`, `cbRewrites`) (read). Cover them with M3's amendment.

### Nits
- The `live` tier compares `m p r c cs d cb t` but not `stats`, so `cbRewrites` is not regression-tested there (it is, indirectly, by the not-regressed test).
- The delegate's report says P1-09 plan item (6), spurious rate on the 30 dev-contact methods, is the data-scientist's; I did not check it and make no claim.

## Answers to the brief's questions

1. **extract.sc.** Read against the plan: R1 (the `case _ => defsOf(s, ...)` branch, `seen` bounds loops), R2 (`<init>` and `<operator>.new` receivers, defined by the call, in the `new`-lowering block and for identifiers), R3 (shape-closed recogniser), `tmpMutArgs`, the wild computed key, the call-point `cb`, 4b (python only, `nearestReal`, sorted replacement set, self-edges kept) and 4f follow the plan's text. The pin is real (m3). Determinism: all new collections are sorted (`ends`, `ctrlOf`, `argument` by index) or groupBy over an ordered buffer; **read, not run twice**. Non-projection facts: no new line writes a `p/c/r/cs/const/deco/t/f` record; `usePositions(..., ext=false)` keeps `cs`/`deco` args equal; the stored-fact test confirms equality on both slice units (executed).
2. **Goldens.** The pair expectations are not derived from the fixed output: the corpus pairs are the P0 truth rows (the 173 labelled misses) and the hand cases are authored with negatives. Read: R1's first pair (`arguments` param to `"mentions" in arguments`, `tests_test_executor_agent_955.py:957`, with the earlier `append((name, arguments))`), the R2 `new Foo(p)` to `o.run(1)` pair, the R3 negatives (`next(it)`, `it.__next__()` user, user alias `walker = items.__iter__()`, JS `it.next()`), the zero-data header `for marker in DROPPED`, and the comprehension fixture all match the source. What *is* recorded-output is the facts file that tier 1 judges, so M1 applies. The `live` tier matters (it is the only check that the script still produces the facts) and is correctly skipped by default with "UNRUN" reasons and a `live` marker; it needs `CCG_PROJ_LIVE=1`, which the delegate ran once (not re-run here).
3. **Mutants.** 17 of the delegate's 18 fact-file mutants die in my replay; the JS temp-guard survivor is m1 above (keep and extend, do not accept as is). My own mutants: line index of the header rule read (`lines[line-1]` to `lines[line]`): 4 failed; non-paren replacement only `(`: 1 failed; paren tier first-of-many: 1 failed; `async for` dropped: 1 failed; `srcLines` unset: 4 failed; the four M2 survivors. Test-side: `direct` ignored: 4 failed; control tuple swapped: 56 failed; expectation inverted: 166 failed; `lang` and span/order mutants: m4.
4. **Equivalence edit.** Still a real check of the slice's raw-fact lines for deletions and edits; blind to additions (m3). The raw-fact equality for non-`d`/`cb` kinds is carried by the stored-fact test (executed, passes).
5. **4f and the closed lists.** 4f: a design change needing a ruling (M3). The closed R3 and 4b recognisers: defensible; the plan's "name not in the source" cannot be implemented in a script that has no source text, and `-p1-ml.md` §3.4 itself requires a closed list for 4b. Note the deviation in the plan text.
6. **Downstream.** See Executed: normalizer suites green, bundles build, +8.4% bytes on `fcs`.
7. **Tests.** As reproduced above; 288 passed / 6 skipped.

## What's solid

- The goldens are independent of the output where it matters (truth rows plus authored negatives), and fixtures are real corpus slices with line-preserving padding.
- 17 of 18 mutants die; the 14-line replaced-slice pin is real for edits; the not-regressed comparator is tested on planted records for the multiset, wrong-method and missing-replacement cases.
- The scorer change is gated by `JOIN_P0_ONLY`, the P0 `POOLED` stays untouched and a second constant carries the P1 join; the opt-in recompute over all six languages passes with the committed 29-id manifest.
- The bundle path accepts the larger `d` without a size cliff.

## Open questions

1. Does the `data-scientist` accept the pinned `RELABELLED` (python 10, csharp `25|2`, `25|6`)? If not, the test pin changes.
2. Should the header rule apply when `contains` is ambiguous (M2)? The plan says "only when the existing tiers fail".

## Routing

- `graph-dba`: M1, M2, m1, m2, m3 (a pin of added lines), m4 and m5; nothing blocks the commit except the two majors, which are cheap.
- `architect`/`teco`: M3, with the `data-scientist`'s STOP ruling.
- `data-scientist`: the relabel ruling and Open question 2.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, "Always check". Source: M1 (and the same class in `docs/reviews/connected-code-graph-p1-06.md` if it applies; not re-read here).
- A golden or equivalence test that judges a recorded output of a script must tie that output to the script (a stamped hash) and be red when the script changes; a pin that guards only one side of a diff (deletions but not insertions) is named as such.

## Pass 2

Re-gate of the graph-dba's fixes. **Verdict: approve** (no blocker, no open major). `extract.sc` is unchanged since Pass 1 (diff stat identical, sha256 `6cdb498f...` equals `tests/golden/projection/extract.sc.sha256`), so `s28_core.py` and `extract.sc` can be committed together. Still pending outside this review: the data-scientist's ruling on the STOP relabels. CPG: considered, not relevant — a Scala script and Python scorer; unchanged reason.

Executed (no Joern; `ulimit -v`, `timeout -s KILL`): `tests/test_s28_metrics.py` + `tests/test_projection_golden.py` + `tests/slice/test_projection_not_regressed.py` + `tests/slice/test_extract_equivalence.py`: 305 passed, 6 skipped (opt-in recompute), 12 deselected. Scratch work under `cpg/.ccg-work/gate-p1-09/pass2` and `.../p2mut`. I briefly created and removed `tests/slice/zz_nr_scratch.py` for the slice red cases (gone; `git status` shows no stray file).

- **M1 fixed.** Scratch copy of the golden dir + test: stamp as committed passes; stamp edited -> fails; script given an appended line with the stamp kept -> fails; stamp missing -> fails. Slice tests (scratch copy of `test_projection_not_regressed.py` with `FIXED` redirected): wrong stamp -> 3 failed, missing stamp -> 3 failed, both fail rather than skip (the `assert` sits after the skip guard, which only skips when the facts themselves are absent). `test_extract_equivalence.py` carries the same assertion (read). The `.ccg-work` stamps are gitignored by design; the committed golden stamp is the binding default-tier check.
- **M2 fixed.** Four planted-fact tests are in `test_s28_metrics.py` and four mutants in `MUTANTS`. Fresh scratch copies of `s28_core.py` with each mutation (iterator lookup over `M.conds`, header rule overriding a several-`contains`, loosened header text, paren tier over `M.conds`): each fails 1 test, 41 others pass. The strict-tier pin asserts `endpoint-unresolved` with reason `cond-several-contains`; the single-`contains` case still resolves via `contains` (existing test, passing).
- **M3 resolved.** 4f is now plan item (8) with golden, negative and mutants, and says the risk is optimism (dev-fitted). I judge the rationale (data pairs ending at a `for` header 457/480 without the final form vs 464/480 with) as a legitimate dev adaptation, not leakage, provided the report states it as dev-only; the recall accounting is the data-scientist's. 4f appears in the README.
- **m1 fixed (read, not run).** `facts.javascript.jsonl` now holds `it.next(it)` with `it` at index 0 and 1; `selfNext` golden asserts `connected: false`. With `js_name_regex_removed` the recogniser would accept it and `defsOf` continue through the arguments to the param, so the negative turns connected; that mutant's text occurs exactly once (always-run test passes), its live Joern run was not repeated here. The `startsWith(recv + ".")` survivor claim is withdrawn by the owner and I never verified it; closed.
- **m2 fixed.** `shape_problems` is independent of the script's audit and tested on planted records (accepts both synthetic shapes; flags non-cond, operator-shaped, wrong label, self-replacement, replacement itself rewritten, unknown replacement). It runs on the stored `fcs` and `sp` facts (executed, passes: the not-regressed file went 13 passed).
- **m3 accepted.** `git diff tests/slice/test_normalize_equivalence.py` shows the graph-dba's docstring edit (lines 141-142) plus the other agent's P1-08 `skippedFiles` delta and its assertion; nothing clobbered. The insertion blindness of `REPLACED_SLICE_LINES` is not pinned, but with M1's stamp the fixed-facts equality now binds to the current script wherever the local facts exist; residual, no action.
- **m4 not fully fixed (minor).** New JS pairs ("span of f", "nested lambda tried first") exist, but scratch mutants of the golden test still pass: `build_method(self.lang)` -> `"python"` (184 passed), span filter removed (184 passed), method ordering reversed (184 passed). These are test-harness selection details, not projection behaviour; fix is optional.
- **m5 fixed.** README has the 4f rule, `cbRewritten`/`cbRewrites`, and a "Regenerating the recorded facts" recipe naming the three stamped artifacts.

Routing: nothing for the graph-dba but the optional m4. `teco`: commit `extract.sc` + `s28_core.py` with the goldens, stamp and tests; keep the data-scientist's STOP ruling as the remaining gate before closing the unit.
