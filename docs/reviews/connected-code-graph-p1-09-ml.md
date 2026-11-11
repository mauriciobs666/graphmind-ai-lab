# C-1001 P1-09 — scorer join fixes in `s28_core.resolve_endpoint`: methodology review

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1-09, P1-34)

**Verdict: approve with conditions.** The `s28_core` change may merge once conditions C1–C7 hold. The `extract.sc` projection work does not depend on this note and is not blocked by it.

Scope: the two join fixes named in `docs/plans/connected-code-graph-p1-extraction.md` P1-09 (a: Python `for` CDG node coded `iteratorNonEmptyOrException`; b: `not (self._open or ...)` resolved to the operand), their effect on P0's pinned figures, P1-34's entry check (`docs/plans/connected-code-graph-p1-gates.md`, `docs/plans/connected-code-graph-p1-ml.md` §1 F1, §3.1) and the AC-5 pre-registration. Repo-resident data only; nothing under `~/.local/share/ccg-eval`.

## What I measured (read-only; scratch copies of the scorer, no repo file touched)

Baseline reproduces the pinned P0 figures from the stored facts files (`cpg/.ccg-work/frontend-fitness/joern/<lang>/`, gitignored) by calling `s28_metrics.analyse_cfg`: python 307/480, javascript 60/151, go 246/433, java 314/465 (the plan's 308/481 and 61/152 add the one omission each).

Prototype of both fixes, applied only where today's join returns "no cond" (a) or a `contains` result (b), python, shipped projection:

| Variant | Pooled | New hits | Hits lost | Connected-before pairs whose resolution changed |
|---|---|---|---|---|
| baseline | 307/480 | | | |
| (a) only | 333/480 | 26 (all `control`) | 0 | 0 |
| (b) only | 310/480 | 3 (2 control, 1 data) | 0 | 1: (9,8), still connected |
| (a)+(b) | 336/480 | 29 | 0 | 1: (9,8) |

- All 29 new hits are pairs the adjudication logged as `join:` causes; none is outside the 40 listed.
- **11 of the 40 stay misses on the shipped projection** (10 data pairs into a loop header, 1 pair of (b)): a real projection cause, not a join cause, and the scorer correctly leaves them. They convert only with R1/R3. The F1 table's additivity ("... + contains-join 4 = 459") is therefore an assumption about the final combination, not something the scorer change alone delivers (+29, not +40).
- Fix (b) is a *different* defect from the plan's wording: the `contains` tier is not "picking the wrong operand", it fails because Joern drops redundant grouping parentheses (`not self._open or ...` vs the label `not (self._open or ...)`), so the whole condition is never an equality match. The whole condition exists on that line; an equality tier that ignores parentheses finds it.
- The other `contains` resolutions (ordinal 19, `exc.status is not None and 500 <= exc.status <= 599`) are a *real* lowering difference (Joern splits the conjunction and rewrites the chained comparison to `500 <= tmp5`); the whole condition does not exist as one point. The paren-insensitive tier leaves them untouched (verified: the three `contains` pairs there keep their old resolution and status). Do not "fix" that class by preferring the longest contained operand.
- (a): the 36 unresolved endpoints are 9 distinct label endpoints; all 9 are literally the loop header in source (`for <label code>:` on the label's line). Python-only token: `iteratorNonEmptyOrException` occurs 1307 times in the python facts and 0 times in javascript/go/java/csharp/cpp.
- Fix (b) run on javascript, go, java: no resolution changes, no recall change (csharp, cpp not run: their facts use the guarded extract; see C6).

## Q1 — fix to a join, or a relaxation read off the misses?

A legitimate join fix, conditional on how it is stated and tested. The line between it and R3:

- R3 (as first written) changed *whether a flow chain counts*: a path through an opaque synthetic call was accepted at scoring time. That alters the construct the metric measures (does the projection connect the pair).
- These fixes change only *which point a label names*. `reach()` and the `(tk, sk) in M.ctl` check are untouched, so a chain that does not exist is still not connected. Evidence in the table above: the 11 pairs with a real projection cause stay misses; zero connected pairs flip; every new hit is a pair whose endpoint previously failed to resolve at all.
- Both rules have a statement that does not mention the 481 outcome: (a) a CDG node coded `iteratorNonEmptyOrException` *is* the `for` header's condition (a frontend fact); (b) the frontend prints the same expression without grouping parentheses (a printing fact).

What makes it legitimate is therefore a test that fails on the old join for a stated reason, written on a **synthetic fixture** (the existing `run_cfg` in `cpg/tests/test_s28_metrics.py` builds facts and source in `tmp_path`), not on the corpus pairs. They were designed after reading dev misses, so the dev numbers they produce are "after adaptation" like the rest of F1 (already the premise of §3.1's zero-optimism guard); that is acceptable only because the fresh draw, if entered, is scored by the same frozen function (C5).

## Q2 — contamination

Allowed: the dev adjudicated items (`adjudication/python-flow-*.adj.jsonl`, `truth/*.flow-final.jsonl`, `results/projection-*.answers.jsonl`) to *characterise* the defect and to run the before/after comparison; synthetic fixtures; the committed corpus source. Not allowed: anything under `cpg/eval/` held-out custody, any fresh P1-34 draw or its labels, and any retuning of the rule after seeing a fresh-draw miss.

The fix is checked against, in order:
1. **No change where the old join was already right:** every pair the old join resolved (any tier other than "no cond") keeps the same resolved point and the same status, except the named (9,8) re-resolution, which must stay connected. Measured above: 0 flips.
2. **New hits partitioned by cause:** every new hit is a pair whose adjudicated class is `join:*`. A new hit whose recorded cause is `projection:*`, `frontend:*` or `definition:*` means the join now masks a real miss: stop.
3. **Real-cause misses stay misses:** the 11 above, and the `definition` pair at `test_repository.py:4703→4706` (ordinal 18), must remain unconnected on the shipped projection.
4. **Specificity of rule (a):** it applies only when (i) no existing tier resolved, (ii) the line has exactly one `iteratorNonEmptyOrException` cond, and (iii) the label's source line is the loop header, `for <code>:` or `async for <code>:` after whitespace collapse. Without (iii) a label on a ternary inside a `for` header line would silently map to the iterator cond. All 9 distinct corpus endpoints satisfy (iii). Several iterator conds on one line: unresolved, with the existing `cond-several-*`-style reason.

## Q3 — P0 pinned figures and P1-34's entry value

- **The existing pin cannot see the change.** `test_committed_answers_reproduce_the_pooled_recall_and_cover_exactly_the_truth_pairs` reads the committed `results/projection-<lang>.answers.jsonl` and counts `connected`; it never calls `resolve_endpoint` on facts. "Tests that pin P0 figures are re-run" is vacuous for a scorer edit; the mutants run on synthetic fixtures only. The re-run that means something is a recompute from the facts (what I did above).
- **Acceptable to report (closed list, stated before the change):** python pooled 307/480 → exactly 336/480 on the unchanged projection (+26 control from (a), +3 from (b); with the omission 308/481 → 337/481); the 29 pair ids; the (9,8) re-resolution; javascript, go, java (and csharp, cpp, see C6) byte-identical. Report old-join and new-join figures side by side; the old-join figure is a diagnostic, never a gate input.
- **Stop and ask `teco`:** any connected→unconnected flip; any change in a non-python figure; a delta other than the 29; a new hit not in the join-cause list; the 11 real-cause misses turning into hits.
- **Entry value.** 465/481 stays as written (`-p1-ml.md` §3.1), now defined as "P1-09's python dev recall, computed by the new-join `s28_core.resolve_endpoint`". It does not move because §3.1 derived 465 from the true recall a *fresh* draw needs (0.965), a property of the fresh draw scored by whatever the frozen scorer is, not of the dev join. Moving the scorer changes what the dev count *measures*, not the threshold it is compared with; so the P1-09 report must state: (1) the old-join count, (2) the new-join count, (3) the entry decision from (2), (4) that (1) would not have entered either if that is the case. If the new-join count falls in 461–464 (the alternative-model band in §3.1) the report says so rather than rounding to a decision.
- Do not re-pin P0's `POOLED` table to the new figures; keep P0's pin as the old-join record and add a second, labelled constant for the P1 join.

## Q4 — pre-registration

`coreTreeHash` = SHA-256 over every git-tracked file under `cpg/` except `cpg/eval/**`, `cpg/packs/**`, `cpg/tests/**` and `**/*.md` (`docs/plans/connected-code-graph-eval-protocol-ml.md` §"Pre-registration"). `cpg/spikes/frontend-fitness/s28_core.py` is tracked under `cpg/` and not excluded, so **the scorer edit is inside `coreTreeHash`** (the new tests under `cpg/tests/` are not). `algorithm` is `recipe.algorithm.{id, version}` (the walk); the scorer is not in it. Consequences:
- The scorer, like `extract.sc`, must be committed **before** the `ac5-gate-1`/`ccg-prereg-v1` record is written; an edit after it makes the first-contact comparison fail and the run "after adaptation".
- The hash dependency is incidental: the s28 scorer scores the dev flow measurement, not the AC-5 gate (which scores through `gate.py`/`match_keys.py`). I would not narrow the exclusion now (a narrowing is a protocol change needing its own ratification); I record it so nobody is surprised, and so P1-34's `ccg-p1-flow-python-v1` record (committed before labelling) also names the scorer's commit.

## Conditions before the `s28_core` change merges

- **C1.** Both fixes live only in the cond branch of `resolve_endpoint`, after the existing tiers: (a) as in Q2.4; (b) a tier "equal ignoring whitespace and parentheses" **before** `contains`, requiring exactly one match; `contains` itself unchanged. The rule text in `DEFINITIONS["join"]` is updated to say so.
- **C2.** Synthetic-fixture tests, each failing on the old join for its stated reason: (a) positive (unique header cond resolves), negative (two iterator conds on the line stays unresolved; label not the header text stays unresolved), and "the old tiers win when they match"; (b) positive (`not (a or b)` against conds `not a or b` and `a` resolves to the whole), negative (the ordinal-19 shape: conjunction split by the frontend still resolves exactly as before). A negative control for the claim that only endpoints changed: a fixture where the iterator cond resolves but no chain exists stays `not-connected`.
- **C3.** New mutants in `MUTANTS` for each fix (rule (a) off; uniqueness guard removed; header-text guard removed; paren tier removed; paren tier moved after `contains`), each verified non-no-op by the existing `old in text` assertion.
- **C4.** A recompute check from the facts (opt-in, skipped when `.ccg-work` is absent, because the facts are gitignored): old join vs new join over all six languages; asserts the closed list in Q3 (the 29 python pair ids as a committed manifest, dev data only) and no other difference. P1-09's report line cites its output.
- **C5.** The scorer is frozen (committed) before any P1-34 fresh-draw seed is fixed or label is written; the `-flow-python-v1` record names its commit. No scorer change after a fresh-draw miss, whatever it shows.
- **C6.** csharp and cpp are recomputed too (their facts use the guarded extract): expected zero change, as no `iteratorNonEmptyOrException` and no paren-only mismatch; if either differs, report and stop.
- **C7.** P1-09's report line carries: old-join and new-join dev recall with cluster intervals, the 29 new-hit pair ids by cause, the 11 pairs still missed and why, and a sentence that the 40-miss join figure assumed additivity and that the scorer alone delivers 29.

## Risks and open items
- Fix (a) is Python-specific by token; TS/JS loop headers (`for...of`) were not part of this change and have no analogous measured defect here (their misses are frontend-side per F3). Do not extend it to JS without its own evidence.
- The cluster interval of the new count uses n = 29 methods; the (b) gain is one method (ordinal 9), so it adds nothing to the interval's independent evidence.
- This note changes no number's meaning outside the dev flow scorer. A fresh-draw figure is the only unbiased recall estimate for the fixed pipeline.

## Pass 2 (2026-11-11) — csharp ruling and review of the delivered results

**Verdict: approve with conditions.** Ruling on csharp: **accept**. P1-34's entry value is met exactly on the design data; that has a reporting rule (C8) and a decision consequence (below). Conditions C8–C12 added.

Verified by me, not taken from the delegate: the scorer, on the delivered `s28_core.py` and the fixed facts file `cpg/.ccg-work/p1-09/out/corpus.fixed.facts.jsonl`, gives fixed/old-join 425/480 and fixed/new-join 464/480 (cluster 95% [0.941, 0.992]; 465/481 with the omission). The 16 residual misses are in 8 methods (ordinals 1, 9, 13, 18, 19, 23, 26, 30). I did not recompute the spurious figures, the `d` growth, or the csharp change (csharp facts were not run); those are the delegate's, labelled so below.

### Pass 1 findings — dispositions
- Q1/Q2.1–4 (join rules, guards): fixed, `s28_core.py` diff matches C1 (paren tier before `contains`, unique; `for`-header rule only when no tier matched, guarded by the header text, several iterator conds unresolved).
- C3/C4: mutants and the opt-in recompute test with a committed manifest exist (`RELABELLED`, `join-new-hits-python.json`); C6 fired, see (1).
- Q3 stop list: the stop on "any non-python change" fired on csharp only; ruled below. The 10 python relabels (`endpoint-unresolved` to `not-connected`) are the pre-announced "11 stay misses" class (10 data pairs; the 11th, in (b), is (9,12) which hit): expected, not a finding.

### (1) csharp: accept
- The two pairs (ordinal 25, pair ids 25|2 and 25|6; label wrapped in one outer parenthesis pair) go from `cond-several-contains` (unresolved) to a unique paren-insensitive equality, then `not-connected`. Recall is unchanged (227/392 both) and no connected pair moves. That is exactly the "label names a different point" class, with no chain accepted that was not there.
- Restricting the paren tier to python would put a language fork in a shared function on the premise that a printing fact differs by frontend, which nothing measured here supports, and would hide a correct resolution. The cost of accepting is only that P0's csharp `endpoint-unresolved` count drops by 2; csharp is not a P1 language and no P1 decision reads it.
- Therefore: keep the tier language-agnostic; replace `RELABELLED["csharp"]` "pending ruling" with the ruled entry (the two ids, reason: outer-paren label, join fix, no recall change). Labelled constants: `POOLED` (P0, old join) stays as the P0 pin; the P1 constant (call it `POOLED_P1_JOIN`) is python 336/480 on the shipped projection, with csharp 227/392, and javascript/go/java/cpp equal to P0. The committed manifest keeps its meaning (the 29 python new hits only); the relabel list (12 ids: 10 python, 2 csharp) lives in the test, not in the manifest. The P1-34 text needs no csharp wording (python only); the P1-09 report line states that two csharp pairs re-resolve with recall unchanged. No P0 report or csharp adjudication file is edited. One residual: the tier strips every parenthesis, so it can equate `f(a)` with `fa`; the unique-match and same-line requirements make a false equality negligible, recorded as a known limit.

### (2) Methodology review of the delivered results

**Entry check met exactly, on the design data.** Dev recall 464/480 (465/481) equals the §3.1 entry value to the pair; it was designed on those same 29 methods (R1, the 27 causes, R3, `for` header, transparency), so it carries optimism by construction, which §3.1 acknowledges ("zero optimism" means no allowance for it). Required reporting (C8):
- the figure as 465/481 with its cluster interval and the statement "met with zero margin; designed on the measured methods";
- the four-way table (shipped/old 307, shipped/new 336, fixed/old 425, fixed/new 464) with each cell labelled old/new join, and the old-join cell labelled diagnostic;
- the 16 residual misses by cause and method.

**Fresh-draw decision.** The rule §3.1 fixed says enter, and I do not move it after seeing the figure. But the plan's "expected outcome: not run" no longer holds and P1-34's step text must be rewritten to the present state: entry is satisfied; the decision to spend the 30-method draw is explicit (`teco`/stakeholder), not automatic, because (a) the 0.80 pass probability in §3.1 assumes fresh recall equals dev recall, and the fixes were fitted to dev, so the true figure is more likely below 0.965 than above; (b) at true recall 0.95 the F2 curve gives about 0.55 to pass the 0.95 bar, and at 0.93 about 0.2. Honest framing for the stakeholder: entry is a cost guard, it does not predict a pass; expect a materially lower probability than 0.80. My recommendation is to enter, because the draw is the only unbiased estimate of the fixed pipeline and a "not closed" flag would otherwise rest on a figure everyone knows is inflated; but the draw must be pre-registered exactly as §3.3 states, and the result reported whichever side it falls.

**The additions (4f, R2/R3 recognizers).** Within F1's scope, not a leakage risk, but they are adaptation:
- 4f (a `for` header cond now receives data from its iterable) is the projection half of the F1 row "`for` header" (the index's 455 counts it); the plan's Behaviours list did not name it, which is the plan text's gap, not a method change. The risk is not contamination (dev data only, no held-out) but optimism, already priced above. Condition (C9): 4f is stated as a lowering-semantics rule (the `iteratorNonEmptyOrException` cond depends on the iterable named in `tmp = x.__iter__()`), has a positive golden and a negative (a comprehension iterator, which the code excludes with `isCompIterator`), and a mutant (m15 exists in the delegate's scratch; confirm it is in the committed set).
- R2/R3 recognizers are closed lists keyed on lowering shape (`tmpN.__next__()`, `_iterator_N.next()`), explicitly not on "name absent from source", which is what distinguishes them from R3-as-first-written. JS covers one shape; that is acceptable because JS flow is not in the entry check and F3 bounds JS recall at 0.546 anyway. Condition (C10): no JS recall figure is credited to P1-09; the JS recognizer needs its golden, a negative (a user `it.next()` stays opaque) and a mutant (`m10`, `m16` appear to be these).
- Both are only credible with negatives that stay unconnected; the delegate's goldens carry them per the progress notes, and the analyst gate checks it.

**Spurious rate (delegate's figures, unverified by me): 91/396 = 0.230 shipped/old; 62/396 shipped/new; 92/517 fixed/old; 62/526 fixed/new.** The "does not rise" test (Behaviour 6) must compare like with like: same join, shipped vs fixed: 0.230 to 0.178 (old join), 0.157 to 0.118 (new join); neither rises. The drop from 91 to 62 on the shipped projection is the scorer (more labels resolve, so more projected pairs count as covered), not a better projection: a join fix lowers the spurious rate mechanically, so the headline spurious figure for the gate/decision record is the fixed/new one, reported beside the shipped/new one, never against shipped/old. Condition (C11): report all four cells and say that the old-to-new difference at fixed projection is the join.

**R1 grows `d` records 63% on fcs (24760 to 40429), delegate's figure.** Measurement consequence is not on the dev flow scorer; it is on anything that consumes projected flow edges. The AC-5 noise thresholds (Q at most 4, possible precision at least 0.20, p90 burden at most 100, reduction at least 5x; `-p1-ml.md` §5.3) measure impact-answer volume: they move only if the walk lists flow-derived items. With `flowAvailable` false (the as-built python/flag-false walk lists callers only, F4) the extra `d` edges should not enter answers. Condition (C12): confirm that statement by running the dev anchors (`cpg/eval/dev/ac5-*.jsonl`) before and after, reporting items listed, p90 burden and the bundle size/load time; if flow edges can reach an answer (flag true for any language), re-measure the burden on dev before the gate record. This is a report, not a gate on the merge.

### (3) Before `s28_core.py` and `extract.sc` are committed
- Both are inside `coreTreeHash` (`cpg/` tracked, not `eval`/`packs`/`tests`/`*.md`); the `-ml` note's Q4 stands. Commit them together, after the analyst gate on the implementation and this pass's conditions, and before the `ccg-prereg-v1`/`ac5-gate-1` record and before `ccg-p1-flow-python-v1` is written. Neither file may change afterwards; a fix found later (including from the fresh draw) is an "after adaptation" run, not a first contact.
- The `ccg-p1-flow-python-v1` record names the commit of both files and the scorer's `JOIN_P0_ONLY = False`; P1-34's `--extract` sha256 test already pins the script.
- `JOIN_P0_ONLY = True` stays a diagnostic switch; no gate or decision record may use it.
- Committed expected values (`POOLED_P1_JOIN`, the manifest) are written from the recompute I reproduced (python 336/480 shipped/new), not retyped.
- No held-out file was read by P1-09 (dev facts and committed labels only); the report states it.

### Routing
- `teco`: rewrite P1-34's entry paragraph (entry met at 465/481 with zero margin; the draw is an explicit decision), put the stakeholder question "spend the fresh draw, knowing the pass probability is below 0.80" to the stakeholder; order the core freeze ahead of the prereg record.
- `graph-dba`/`tdd-engineer`: C8–C12 and the csharp constant/records change in `test_s28_metrics.py`.
- `analyst`: gate the implementation, including the negatives for 4f/R2/R3 and the like-for-like spurious report.

## Pass 3 (2026-11-11) — tier order of rule (a) against an ambiguous `contains`

**Ruling: keep the order strict.** The `for`-header rule runs only when no earlier tier matched, an ambiguous (several) `contains` outcome included; it must not override it. This is the current code (`if not ct and ...`) and the plan's wording.

- **Why strict.** An ambiguous `contains` means the label text sits inside several conds on the line, i.e. the line holds something other than a bare loop header (a ternary or boolean inside the iterable, say). The header-text guard (iii) is a prefix test and does not exclude that case, so overriding would silently re-point a label whose text matches real conds to the iterator cond. Declining to resolve an ambiguity is the scorer's existing, conservative convention (`cond-several-*` stays a miss); widening it needs its own evidence, and a later widening is cheaper than a retraction after the freeze.
- **Dependence on the dev data: none.** I ran the permissive variant (header rule overriding `cond-several-contains`) read-only on the shipped and the fixed python facts, new join: zero endpoints reach `cond-several-contains` on a `for`-header line, so it changes no pair's status or connection in either (0 of 480). The 29 new hits and the 10 relabels do not depend on the choice, and neither does the 464/480 figure. (The header rule is python-only; the other languages' lines never satisfy the `for <code>:` text test.)
- **Pin (graph-dba's test).** Synthetic fixture: a `for <code>:` source line with exactly one `iteratorNonEmptyOrException` cond and two further conds on the line, each containing the label text, and no equality match; expected `endpoint-unresolved` with reason `cond-several-contains`. The mutant "header rule also fires when `ct` has several entries" must die on it. Same fixture family should pin the single-`contains` case (one candidate returns `contains`, not the iterator cond), since the code returns on `len(ct) == 1` before the header rule.
- If a future corpus shows such pairs, treat widening as a new scorer change: its own evidence, before any fresh draw is fixed (C5), not a silent test edit.
