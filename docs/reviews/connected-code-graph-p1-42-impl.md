# Review: ccg P1-42 AC-level fixture gate (implementation)

> **Status:** active · **Owner:** `analyst` · **Tracks:** P1-42 (ccg P1)

CPG: considered, not relevant — the subject is test code and a test report over the `ccg` answer/store surface; no `cpg_<component>` Joern graph answers it.

## Pass 1

**Scope.** Commit `35273bad`: `cpg/tests/test_fixture_gate_p1_ac.py`, `cpg/tests/fixture_gate_p1_ac_helpers.py`, the "P1-42" section of `docs/test-reports/connected-code-graph-p1-report.md`, the §7 paragraph of `docs/test-plans/connected-code-graph-p1.md`. Baseline: P1-42 row (`docs/plans/connected-code-graph-p1.md:397`), AC-6/9/16/17/18/21/25 and the decision log in `docs/requirements/connected-code-graph.md`, plan `docs/plans/connected-code-graph.md` §4.1. Root `REVIEW.md` read; no component `REVIEW.md` for `cpg/` found. Growth-stops: new document, no prior pass.

**Verdict: approve with suggestions.** No blocker. The tests assert what the table says for AC-6, 9, 17, 18, 21, 25. AC-16's raw leg is narrower than the report's verdict sentence (F1). No source was modified.

**Executed.** Offline: `cd cpg && .venv/bin/python -m pytest tests/test_fixture_gate_p1_ac.py tests/test_fixture_gate_p1.py` -> 9 passed, 120 deselected. Live: own throwaway `ccg-falkordb-tw-16397` (`CCG_REQUIRE_LIVE=1`; the driver imported into a scratch module, so its 43 tests ran plus my 5 probes) -> 48 passed, 299 s. Removed afterwards; `ccg-falkordb`, `falkordb-dev`, `ccg-falkordb-tw-16390` still running, untouched.

**Figures recomputed (verified).** From the live answers: 9 answers, 51 items, 179 hops (sum of `len(chain)-1`): matches. AC-21 attempts: 6 shapes x 2 (QUERY, PROFILE) + 7 further commands + 6 `raw_query` = 25: matches the probe's construction.

### Findings

**F1 (major, verified) — the AC-16 raw leg proves sufficiency of the recipe, not the item set; the verdict sentence and test-plan wording overstate it.**
`replay_chains` (`test_fixture_gate_p1_ac.py:428-493`) takes the engine's own chains and checks each hop is returned by a recipe read. Probed on all 9 live answers (my `test_omission_depth2_undetected`, deep-copy the answer, then replay):
- Deleting the last distance>=2 item: replay green on 7 of 7 answers that have one.
- Deleting the last distance-1 item: green on 6 of 9; red only on `CF-DF3` and the location ask, where it breaks the first-level call-site set (`:482-492`, symbol-anchored impact only).
- Setting an item's `distance` to 99 and flipping its `status`: green on 9 of 9.
So omissions beyond level 1, every contract-point and depends-on omission, and item-level status/distance/ordering are unchecked. The test plan row says "the raw recipe reproduces the item set" (`docs/test-plans/connected-code-graph-p1.md`, AC-16 row) and the report says it "returns every hop ... of every item"; O-1 admits the gap but the verdict line and per-AC table do not.
**Does the requirement allow closing at P1?** Yes, on the weaker leg, if stated as such. AC-16 asks that re-running the *published recipe* "returns the same facts"; decision S1 (2026-10-26) says a recipe re-run "reproduces the facts" and a from-scratch raw query "may legitimately differ". Plan P1-38 already requires the closure report to state "the narrow coverage of AC-16's raw-recipe leg", and the test plan calls it "the weaker leg, by decision". So no stakeholder decision is needed, but "closed for chain reproducibility" must name the unchecked parts: completeness beyond level 1, item status, distance, grouping. Do not write "AC-16 closed" bare in the P1-38 table.
**Improvement.** Reword the verdict sentence and AC-16 table cell to say replay proves existence of every emitted hop and the level-1 set only; state in O-1 the three silent mutants above. Optional strengthening (design not run by me): derive the reachable symbol set to the depth bound by iterating the recipe's `calls_in`/`name_in`/`owner_of` reads from the anchor, and require every non-flow-downgraded answer item to be in it and every derived symbol with only `confirmed` edges to be listed. A mutant that drops a deep item is the assertion that would catch a wrong design; the planted-omission test should be added in a round before the check is wired.

**F2 (minor, verified) — AC-18 asserts presence of file and line, not correctness.**
`ac18_violations` (`fixture_gate_p1_ac_helpers.py:55-100`, `_has_loc`) accepts any non-empty `path` and `line >= 1`. Probe: every Symbol/CallSite/ConfigEntry hop of `CF-DF3` set to `path='nonexistent/zzz.py'`, `line=99999` -> 0 violations. An engine regression that emitted wrong lines would keep AC-18 green. The literal AC text ("file and line at each step") is presence, so the report's "closes at P1" is a fair reading, but the table should say "present", not "correct".
**Improvement.** Cheap independent check from the materialized fixture trees: hop `path` exists in its repository and `line` <= the file's line count; for a `CallSite` hop, the source line contains the callee's name. The bogus-path mutant above is the assertion.

**F3 (minor, inferred from code) — AC-21 `_refused` counts any exception as a refusal.**
`test_fixture_gate_p1_ac.py:533-538`. I verified a wrong raw password raises `AuthenticationError` at connect, so a wrong credential cannot hide as 25 refusals (red, not silent). But a refusal for another reason (unknown key, typo in the statement, timeout) would count. The ledger records only the first 3 texts. The `+graph.query` mutant covers the ACL path, not the other 22 attempts.
**Improvement.** Assert the refusal text matches `no permissions` (or the `GRAPH.RO_QUERY` write-refusal text for `raw_query`) for each attempt and fail on any other class. Not run.

**F4 (minor, verified by reading) — `test_ac17_the_other_groups_are_not_built_at_p1_observed` has no assertion** (`:239-249`); it records and always passes. The report's "branch and stack-trace questions answer a typed `NotBuilt`" is therefore a one-run observation, not a test. Either assert `isinstance(..., common.NotBuilt)` or say "observed once, unasserted". (The sibling `test_ac6_dynamic_import_is_not_met_at_p1_observed` does assert `(0, 0, 0)`, so it flips red if P2 lands: good.)

**F5 (nit) — AC-9's "closes at P1" is half vacuous.** The decision-state half is `verdictLogOffset == 0` on an empty log (`:319`). The table says so ("until P4"); the P1-38 row should carry it as "stamp form closed; decision-state content open to P4" rather than a bare close.

### Judgements asked for

1. **AC verdicts vs assertions.** AC-6 (three kinds asserted per instance with reason and not-confirmed; dynamic import asserted `(0,0,0)`): follows. AC-9: follows (equal/same-commit/changed-commit stamps asserted). AC-17 (2 of 4 groups, 3 `.d.ts`/`.pyi`/vendored negatives asserted `unresolved` with no candidates): follows. AC-18: follows at presence level (F2). AC-21: follows (25 refusals, view unchanged, positive controls, mutant). AC-25: shape, data-flow and reverse question asserted; control half asserted in the *not-met* form via CD-1/3/4 "not listed", which goes red if the engine starts listing them, so the flag-true re-run must flip it. AC-16: see F1.
2. **Deferrals.** AC-17 queue name: closes at the P2 exit by plan §4.1 table ("AC-17 ... Closes at P2 exit: The queue name") and the 2026-11-01 stakeholder decision; branch is P3, stack trace and test name P5. Not asking it at P1 is permitted; the P1 test plan §12 already excludes "the parts of AC-17 beyond code location and contract point". No decision needed. AC-25 control half: requirements status 2026-11-19 says "not met ... Open until the held-out `flow` stratum reports; the stakeholder accepted this state"; `-p1-gates.md` P1-34 step 4 and P1-34/`-flow-prereg` U9 re-run P1-42's fixtures only if the registry flag flips true, and `-flow-fallback-result-ml.md` says only a stakeholder waiver would close it. Recommendation: leave open past P1-42; carry it into the P1-38 closure table as "open, judged on P1-37's held-out `flow` stratum"; keep the CD-form assertion as the re-run hook. A stakeholder decision is needed only if P1-38 wants to waive it for phase exit; the 2026-11-19 acceptance covers the state, not a phase-exit waiver. Note `docs/plans/connected-code-graph-p1.md:264` still heads "AC-25 (closes at P1)"; lines 270 and 487 say the control half is open: a plan-text inconsistency for the architect, not a defect here.
3. **Execution.** Mutants run: F1 (omission, distance/status), F2 (bogus path/line), F3 (wrong raw password -> `AuthenticationError`, red), pin echo: replaying the `depends-on` echo's `useCase` (`impact`) as a pin gives facts different from the depends-on answer (`False`), so assertion `:409` using `direction` discriminates and O-3 is accurate. The report's own mutation claims (carrier, manifest, raw-leg drops, ACL) I did not re-run; the offline checker mutants pass.

### What's solid
- Expectations come from manifest rows or requirement text, not from the code under test; the planted-input checker mutants (AC-6/18/25, replay) are offline and in the default suite.
- Honest non-closures (AC-6 dynamic import, AC-25 control half) are asserted, not skipped; the replay raises `Underivable` rather than skipping.
- Throwaway discipline held; live result reproduced independently (48 passed incl. the 43).

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, `## Always check`. Source: F1, F2 (this review).
- A check that re-derives or replays an answer's rows from its published source (recipe replay, chain re-read) is probed for omission and for item-level attributes: delete one item (deep and shallow), change one status/distance, replace a located field with a bogus value. Green on any of these means the check proves sufficiency of what was emitted, and the closure wording must say so.

No prune proposed.

## Appendix
Probe script (scratch, not committed): `/tmp/claude-1000/-home-<user>-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/mut/test_mut.py`. Per-answer omission results: `CF-DF` deep/shallow/attr all silent; `CF-DF3` deep silent, shallow red, attr silent; `CF-CD`, `RV-1..3` all silent; contract asks all silent; location ask deep silent, shallow red, attr silent.
