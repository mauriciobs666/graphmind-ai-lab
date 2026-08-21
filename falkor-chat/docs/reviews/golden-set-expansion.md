# Golden-set expansion — finalization review (K-027 item 4)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-027 (item 4)

**2026-08-21: K-027 epic closed; no further passes expected.**

**Pass 3 verdict (2026-08-21): approve.** The implementation (`tdd-engineer`'s diff) is a faithful,
byte-exact realization of the Version-3 plan `analyst` approved in Pass 2 — see
`## Pass 3 (2026-08-21)` below for the full implementation re-gate, including an independent
re-run of both the offline suite and the live LM Studio calibration.

**Pass 2 verdict (2026-08-21): approve.** All four Pass 1 major/minor findings are confirmed fixed
in Version 3, diff-scoped and re-verified against the real files (not taken on the fix-pass note's
word) — see `## Pass 2 (2026-08-21)` below. The plan is genuinely ready, unconditionally, for
`tdd-engineer` to implement §7 as written.

Reviews `docs/plans/golden-set-expansion-ml.md` Version 2 (2026-08-20 finalization pass) against
its own two closed decisions (boundary-tier independent-labeling requirement dropped;
`clear_suspend` sized at n=40), the real `server/tests/eval/golden_guards.jsonl` fixture, and
`server/tests/eval/test_guard_calibration_live.py`/`guard_calibration.py`. Scope is the
finalization itself — internal consistency of the two closures, honesty of the descope record, a
spot-check of the 59 drafted rows in §6, row-count reconciliation, and readiness for
`tdd-engineer` dispatch — not a re-derivation of the underlying Wilson-interval sizing math, which
was sanity-checked rather than fully re-audited per the brief.

**CPG:** considered, not relevant — `cpg_falkorchat` is loaded, but this review's checks (fixture
JSON content/schema, five literal assert-line citations in one test file, dynamic-vs-hardcoded
grouping in `guard_calibration.py`) are data/content and single-file-line-number questions, not
call-graph/impact-analysis questions a CPG traversal would answer faster than direct reads; all
code claims below were verified by reading the actual files.

**Verdict: needs changes.** Two real, evidence-backed defects — one a residual contradiction of
the closed descope decision inside the plan's own "ready for dispatch" section, one a schema/
semantics defect in 2 of the 59 drafted rows — should be fixed before this document is treated as
unconditional and before the rows are merged. Everything else mechanically reconciles (all 59 rows
parse, no id collisions, schema matches the real fixture exactly, and the 18/12/24/16/9/6 → 85
composition is exact), and the descope itself is recorded honestly everywhere else in the document.

## Findings

### Major — §7 step 5 still asserts boundary needs "a second independent human pass," contradicting the closed decision

`docs/plans/golden-set-expansion-ml.md:518` (§7 step 5, "Live run cost"):

> The **labeling** cost (59 new rows, **at least 15 of them needing a second independent human
> pass**) is the real cost this note is asking the user to authorize.

15 is exactly the boundary-tier total (§3.4), so this sentence is a leftover from before the
2026-08-20 closure — it directly says boundary rows need a *second* independent human labeler,
which is precisely what §5 (line 262: "No second labeler is available... sourced identically to
`clear_advance`/`clear_suspend`"), §8 item 1, and §10 item 1 all say was dropped. Every other
occurrence of "second labeler"/"independent" in the document (checked via `grep -n` across the
whole file) is consistent with the closure; this is the one place it isn't. This sits inside the
same numbered step (§7) the revision note declares "unconditional as of this revision" and that
the brief is dispatching to `tdd-engineer` — a reader who only skims §7 for the remaining cost
would come away thinking a second boundary labeler is still needed, exactly the residual hedge
item 2 of the brief asked me to check for. Fix: reword to something like "the **labeling** cost
(59 new rows, all sourced per §5(c) option (a) — LLM-drafted, single human spot-check, no second
independent pass for any tier including `boundary` per §5's 2026-08-20 descope) is the real cost."

### Major — `r1_probe: true` misapplied to 2 of the 59 new rows, corrupting the coercion-flip diagnostic's scope

`r1_probe` is not a generic "this is a tricky/adversarial case" flag. Per the archived protocol
(`docs/archive/plans/m3-guard-calibration.md` §4.3) and its live implementation
(`guard_calibration.py:276-280`, `coercion_flip_rate(cases, r1_probe_only=True)`, which counts
`raw_decision is True and final_decision is False`), `r1_probe` marks cases whose **correct answer
is advance** and whose phrasing risks tripping the `_NEGATION_CUES` coercion logic into wrongly
suspending a legitimate advance. That's why every one of the existing fixture's 5 `r1_probe: true`
rows (`ca-01`, `ca-03`, `ca-05`, `ca-07`, `tn-01` — verified by parsing the real
`golden_guards.jsonl`) is `tier: clear_advance` / `expected: true`, and why the archived note's own
§4.3 language says "restricted to the **5** `r1_probe: true` cases."

Two new draft rows break this: `cs-10` and `cs-13` (both `tier: clear_suspend`, `expected: false`)
are marked `r1_probe: true`. Mechanically confirmed (parsed all 59 new rows): these are the *only*
new `r1_probe: true` rows outside `clear_advance`. Their own `label_rationale` text confirms the
mix-up — `cs-10`'s rationale is "ADVERSARIAL / TRUST PROBE: ... mirroring cs-04," and `cs-04`
(the existing row it mirrors, same rationale shape) is `r1_probe: false`. The drafting evidently
used `r1_probe` to mean "adversarial probe" rather than its actual, narrower, coercion-mechanism
meaning.

Why it matters beyond labeling hygiene: for a `clear_suspend` row, a `raw_decision: True` (i.e.
the judge wrongly advances) is itself a G1 false-advance failure. If `_coerce_verdict` then flips
that wrong advance back to suspend because the judge's own rationale happened to contain a negated
cue, `coercion_flip_rate(..., r1_probe_only=True)` would count that event as evidence of the
*defect the metric exists to detect* ("the judge correctly wants to advance but wording tripped a
false coercion") — when it is actually the opposite (coercion accidentally masking a real
false-advance). Mixing suspend-tier rows into the r1_probe-restricted denominator also silently
invalidates the archived spec's "the 5" language and dilutes the metric's clean semantics, even in
the case where no flip ever occurs. This is diagnostic-only (`false_advance_rate`/`advance_recall`
don't filter on `r1_probe`, so G1/G2 gating itself is unaffected — confirmed by reading
`guard_calibration.py:201-239`), which is why this is major rather than blocker. Fix: flip
`r1_probe` to `false` on `cs-10` and `cs-13` before merge (their `clear_suspend` labeling and
rationale otherwise stand on their own), and consider tightening §6's own preamble to state the
`r1_probe` semantics explicitly so a future drafting pass doesn't repeat the mix-up.

### Minor — two of the five literal-constant line citations in §7 step 2 (and the earlier F3 quote) are off by one

§7 step 2 and F3 both cite:

```
line 233 (`== 21` → `== 70`, i.e. 30+40), line 234 (`== 5` → `== 15`)
```

The actual file (`server/tests/eval/test_guard_calibration_live.py`, checked directly, current
line numbers):

```
223:  assert len(rows) == 26
227:  assert len(cases) == 26
228:  assert sum(len(c.replicates) for c in cases) == 26 * K_REPLICATES
232:  assert len(clear_cases) == 21
233:  assert len(boundary_cases) == 5
```

The `clear_cases`/`boundary_cases` assertions are at 232/233, not 233/234 — the content (`== 21`
and `== 5`, and the correct target values `== 70`/`== 15`) is right, only the two line numbers are
shifted by one. Low practical risk (an implementer will find both asserts by searching for the
literal text, not by line number alone), but this is exactly the kind of grounding slip a
finalization pass should catch, since F3 markets itself as read-directly-from-the-file. Fix:
correct both citations to 232/233 in both places (F3's quoted block and §7 step 2's prose).

### Minor — §3.1's sizing table shows an inconsistent value for n=53 at zero observed failures

§3.1's table lists, for n=53: "0 observed failures" → `~8.4%*`. Recomputing the same two-sided 95%
Wilson score interval the note uses everywhere else (verified against the note's own reproduced
figures — n=10/x=0 → 27.8%, n=30/x=0 → 11.4%, n=35/x=0 → 9.9%, n=40/x=0 → 8.8%, all of which match
exactly) gives n=53/x=0 → **6.8%**, not 8.4%. The asterisk footnote ("minimal n at which that
failure count still clears ≤10%, found by search") also doesn't fit this cell: the note's own prose
already identifies n=35 as "the real zero-failure floor," so n=53 isn't a "minimal n" result for
the x=0 column at all — that framing only makes sense for the genuinely-bolded/asterisked cells
(n=35/x=0, n=53/x=1, n=69/x=2), which do reproduce correctly. This is inherited from v1 (untouched
by this finalization pass per the revision note, "nothing in §1–4's findings... changes") and
doesn't touch the accepted n=40 decision (independently verified correct at 8.8%), so it's minor —
but worth a correction pass given the section's own selling point is exactness against the
backlog's "~30" heuristic.

### Nit — the §6 preamble's "DRAFT/UNVERIFIED" tagging convention is applied inconsistently

§6's preamble states every rationale is "tagged `DRAFT/UNVERIFIED` (for rows whose reading needs a
moment's thought) or otherwise plainly states the reasoning as a proposed reading." In practice
only the two carried-over rows (`tn-08`, `tn-09`) carry the literal tag; none of the new
adversarial/materiality/partial-anchor probe rows do (e.g. `cs-10`, `cs-13`, `ca-12`, `ca-15`) —
exactly the rows that plausibly "need a moment's thought." The blanket disclaimer at the top of §6
("Draft only. Not independently verified.") covers all 59 rows regardless, so this isn't a
substantive honesty gap, just an unevenly-applied convention that would be cheap to tighten (either
tag the probe rows consistently, or drop the parenthetical distinction and rely on the blanket
disclaimer alone).

## Verified mechanically (not just read)

- **All 59 rows in §6 are valid JSON, unique ids, no collision with the existing 26** — parsed
  every fenced JSON block in the plan and every line of the real
  `server/tests/eval/golden_guards.jsonl`; 59 new + 26 existing = 85 unique ids, schema key-set
  (`id`, `tier`, `path`, `r1_probe`, `condition`, `understanding`, `turns`, `expected`,
  `label_rationale`) identical across old and new rows.
- **Row-count reconciliation is exact.** New-rows-by-stratum/path (10/9/17/13/5/5 = 59) plus
  existing (8/3/7/3/4/1 = 26) reproduces §3.4's target table exactly: `clear_advance` 18/12,
  `clear_suspend` 24/16, `boundary` 9/6 — 85 total, 51 understanding/34 turns, matching both the
  document's own arithmetic and the closing paragraph's "32 new understanding + 27 new turns = 59."
- **Structural sanity of the drafted rows**: every `condition` string is byte-identical across all
  85 rows; every `turns`-path row has empty `understanding` and non-empty `turns` (and vice versa
  for `understanding`-path rows); every `boundary` row is `expected: false`, every `clear_advance`
  row `expected: true`, every `clear_suspend` row `expected: false`; all 96 new `msgId`s are
  globally unique; timestamps within each row's `turns` are monotonically non-decreasing. No
  anomalies found beyond the `r1_probe` issue above.
- **F3's harness-genericity claim** — read `guard_calibration.py` directly:
  `false_advance_rate`/`advance_recall`/`confusion_matrix`/`cohens_kappa`/`per_path_breakdown` all
  filter dynamically on `c.tier`/`c.path`, no hardcoded counts. `test_guard_calibration.py` has no
  reference to `golden_guards`/`load_golden_guards` — confirmed it doesn't need touching.
- **The n=40 Wilson numbers themselves** (8.8% at x=0, 12.9% at x=1, 16.5% at x=2) and n=30/n=35
  figures reproduce exactly under the standard two-sided 95% Wilson formula.
- **§7 step 4's proposed report-template replacement text** (the block quote) correctly states
  n=40/30/15 and correctly describes boundary as single-labeled with no independent second labeler
  — this is the one place besides §5/§8/§10 that gets the closure right, confirming the
  contradiction found above (§7 step 5) is a localized miss, not a document-wide pattern.
- **`test_golden_set_integrity.py` exists and is a real, working precedent** for §7 step 3's
  "recommended, not required" offline structural test — its own pattern (`30 <= len(rows) <= 50`
  as an inequality rather than an exact literal) is exactly the style the plan recommends for the
  guard fixture's per-stratum minimums, so the recommendation is concrete and grounded, not
  hand-waved.

## What's solid

- The descope itself (item 2 of the brief) is recorded honestly almost everywhere: §5's rewritten
  opening states plainly "This is a deliberate, explicit descope from the backlog item's stated
  intent, not a quiet scope-narrowing," §8 item 1 and §10 item 1 repeat the same framing with the
  same directness, and §7 step 4's proposed report caveat text is exactly the kind of downstream
  language that keeps a future reader from over-crediting the boundary tier's validation. The one
  place this breaks down is the §7 step 5 sentence flagged above — a real but isolated miss, not a
  softening pattern.
- n=40 vs. n=53 is handled well: the trade-off is stated honestly as a real choice with real
  power-cost (§3.1's own power table, §8 item 2), not dressed up as unambiguously "safe."
- The row-count bookkeeping in §6's own preamble table is exact and matches the real fixture
  byte-for-byte — a genuinely careful piece of arithmetic, confirmed independently above.
- §7's core instructions (fixture content, the five literal-constant edits, the optional offline
  integrity test, the threshold-unchanged rationale) are concrete enough for a no-context
  implementer to execute, modulo the two off-by-one line citations.

## Open questions

- Should the `r1_probe` field's actual semantics (tied to R-1/`_NEGATION_CUES` coercion risk on
  advance-expected cases, not general adversarial framing) be documented explicitly somewhere
  reachable from `golden_guards.jsonl` itself (a short comment file, or a note in
  `test_guard_calibration_live.py`'s docstring) so a future drafting pass doesn't repeat the
  `cs-10`/`cs-13` mix-up? This plan doesn't currently point anywhere for that definition beyond the
  archived, frozen protocol doc.
- Does the team want the archived protocol's §4.3 "restricted to the 5 `r1_probe: true` cases"
  language updated once the fixture grows (it will now be a different count, and the fix above
  keeps it at exactly 5 unless a future drafting pass adds more) — worth a one-line note in the
  archived doc's own header pointer chain, or left as historical text since the doc is frozen?

## Pass 2 (2026-08-21)

**Scope.** Diff-scoped re-gate of `docs/plans/golden-set-expansion-ml.md` Version 3 (the
"2026-08-20 fix pass (v2 → v3)" revision, lines 18–29) against this document's own 4 major/minor
Pass 1 findings — not a re-review of the whole plan. Each finding was independently re-verified
against the real files (test file line numbers, the real fixture, recomputed Wilson figures), not
accepted on the fix-pass note's own claim.

**CPG:** not applicable — this pass, like Pass 1, is a diff-scoped text/data re-check (line-number
citations, JSON row content, a Wilson-formula recomputation) with no call-graph or impact-analysis
question in it.

### Finding-by-finding

1. **Major — §7 step 5 "second independent human pass" contradiction: FIXED.** `docs/plans/
   golden-set-expansion-ml.md:557-564` (§7 step 5) now reads "no second independent pass for any
   tier including `boundary`, per §5's 2026-08-20 descope," with an explicit `(2026-08-20
   correction, analyst review:` ...) note naming the prior contradiction. Re-ran the same
   whole-file grep Pass 1 used (`grep -n -i "second.*independent\|independent.*second\|second
   labeler\|second human\|independent human"`, 25 hits) — every remaining occurrence is either (a)
   part of the closed-descope record (§5, §8 item 1, §10 item 1, the revision-note preamble), (b)
   the historical §5 sub-analysis explicitly framed as "kept for the record" of what the descope
   gives up (lines 311–362, e.g. line 356's "(c) my recommendation" quoting the original,
   now-superseded proposal — clearly retrospective, not a live claim), or (c) the fix-pass's own
   description of what it changed. No live contradiction remains anywhere in the document.

2. **Major — `r1_probe: true` misapplied to `cs-10`/`cs-13`: FIXED, and re-verified mechanically.**
   Parsed all 59 drafted JSON rows in §6 directly (not just read the two cited ids): `cs-10` and
   `cs-13` are now `"r1_probe": false` (both still `clear_suspend`, `expected: false`, their
   `label_rationale` text unchanged). The only two `r1_probe: true` rows across all 59 are
   `ca-12` and `ca-15`, both `clear_advance` — exactly the constraint the original finding
   required. Cross-checked against the real fixture (`server/tests/eval/golden_guards.jsonl`,
   parsed directly): the existing 5 `r1_probe: true` rows are still exactly `ca-01`, `ca-03`,
   `ca-05`, `ca-07`, `tn-01`, all `clear_advance` — matching §6's new preamble text (lines
   388–399) verbatim, which now states the field's narrower semantics explicitly. No new
   `r1_probe` mismatch introduced elsewhere in the 59 rows.

3. **Minor — 233/234 → 232/233 line-citation off-by-one: FIXED.** Read `server/tests/eval/
   test_guard_calibration_live.py` directly: lines 232/233 are `assert len(clear_cases) == 21`
   and `assert len(boundary_cases) == 5` — exactly what §7 step 2 (`:518-524`), F3 (`:78-87`),
   and §9 (`:606-609`) now cite, all three updated consistently (§9 additionally notes this is
   the "third occurrence" of the same root cause, which checks out — F3, §7 step 2, and §9 all
   carried the same original off-by-one and all three are now corrected). Grepped the whole
   document for `233/234` — the only hits left are inside the fix-pass's own "corrected from
   233/234 to 232/233" narration, not live citations.

4. **Minor — n=53/x=0 Wilson-table figure: FIXED, and internally consistent.** Recomputed the
   two-sided 95% Wilson score interval independently (Python, standard formula, same one used for
   the review's Pass 1 spot-check): n=53/x=0 → 6.8% (6.76% unrounded), matching §3.1's corrected
   table cell (`docs/plans/golden-set-expansion-ml.md:139`) exactly. The asterisk was also
   correctly dropped from that cell — the correction note (`:144-150`) matches the reasoning
   Pass 1's suggested fix laid out (n=35 is the real zero-failure floor; the
   minimal-n-by-search framing only applies to the 53/x=1 and 69/x=2 cells). Re-verified those two
   remaining asterisked cells are unaffected by the correction and still reproduce under the same
   formula (~9.9% and ~9.97%, both consistent with the table's 9.9%*/9.7%* to the table's own
   rounding). The accepted n=40 decision (8.8% at x=0) is untouched by this correction, as the
   revision note claims — confirmed independently, not just read.

### New-defect sanity check

Ran the full mechanical checks Pass 1 ran, against V3, to confirm the fix pass didn't disturb
anything it didn't touch:

- **Row count and composition still reconcile exactly.** All 59 drafted rows parse as valid JSON;
  per-stratum/path breakdown is 10/9/17/13/5/5 = 59, matching §6's own reconciliation table.
  Combined with the real 26-row fixture: 85 total, composition `clear_advance` 30 (18
  `understanding`/12 `turns`), `clear_suspend` 40 (24/16), `boundary` 15 (9/6) — byte-exact match
  to §3.4's target table. No id collisions between the 59 new ids and the 26 existing ones, no
  duplicate ids among the 59.
- **The revision note's claim "no drafted-row content changed except the two `r1_probe`
  corrections" holds** — spot-checked by re-diffing the `label_rationale` text Pass 1 quoted for
  `cs-10`/`cs-13` against V3: unchanged except the `r1_probe` field itself.
- **The nit (§6's inconsistent `DRAFT/UNVERIFIED` tagging) is untouched**, as expected — it was
  explicitly out of scope for the fix pass. `DRAFT/UNVERIFIED` still appears exactly 3 times
  (the preamble's mention of the convention, plus `tn-08`/`tn-09`'s own rationale text) — same as
  Pass 1 found. Not a blocker; still fine to leave as-is per Pass 1's original call.

### Verdict

**Approve.** All 4 major/minor findings are genuinely fixed, each independently re-verified against
the real files rather than accepted on the fix-pass note's word, and the fix pass introduced no new
defects detectable by the same mechanical checks Pass 1 ran. §7 is ready for `tdd-engineer` to
implement as written: fixture growth 26 → 85 rows (the 59 drafted rows in §6, now internally
consistent), the five literal-constant edits in `test_guard_calibration_live.py` (lines 223, 227,
228, 232, 233), and the optional offline integrity test per F4/§7 step 3. No open blocker remains;
the two open questions Pass 1 raised (documenting `r1_probe`'s real semantics somewhere reachable
from the fixture itself, and whether to touch the archived protocol's frozen §4.3 language) are
process/documentation suggestions, not implementation blockers, and are left to the team's
discretion as before.

## Pass 3 (2026-08-21)

**Scope.** Diff-scoped implementation re-gate: does the uncommitted working-tree diff (`tdd-engineer`'s
`U3` deliverable) actually build the thing Pass 2 approved (`docs/plans/golden-set-expansion-ml.md`
Version 3), and are the delivered artifacts' own factual claims (fixture composition, live-run
numbers, doc updates) true of the real files? Not a re-review of the plan itself — that's Pass 1/2's
job and stands. Six files changed: `server/tests/eval/golden_guards.jsonl` (+59 rows),
`server/tests/eval/test_guard_calibration_live.py` (5 literal edits), `server/tests/eval/
test_guard_set_integrity.py` (new), `docs/BACKLOG.md`, `docs/HISTORY.md`, plus a new test report
(`docs/test-reports/guard-judge-calibration-2026-08-21.md`) and a coordination log
(`docs/plans/golden-set-expansion-coordination.md`) that record the delivery. This gate is the last
one before K-027's epic-closure claim (the BACKLOG header flip to ✅ delivered) is final.

**CPG:** not applicable — every check here is a data/content/text comparison (fixture JSON rows vs.
plan-drafted JSON rows, line-number diffs in one test file, doc-claim vs. real-file cross-checks,
running an existing pytest suite) with no call-graph or impact-analysis question in it; `cpg_falkorchat`
would not answer any of these faster than direct reads.

### Finding-by-finding (the brief's 7 checks)

1. **Fixture content fidelity — verified exactly, not sampled.** Rather than spot-checking, I
   extracted all 59 JSON rows from the plan's §6 fenced code blocks programmatically and diffed them
   field-for-field against the real `server/tests/eval/golden_guards.jsonl`. Result: **all 59 plan
   rows are present in the real fixture and every field matches byte-for-byte** (parsed-JSON
   equality, not text diff, so key order doesn't matter) — zero mismatches, zero missing. This is a
   stronger check than the brief asked for ("spot-check a meaningful sample") and it closes the
   question completely: the drafted content was transcribed programmatically, not retyped (matching
   `HISTORY.md`'s own claim, "extracted programmatically from the plan's own fenced JSON blocks, not
   retyped, to rule out transcription drift" — confirmed true, not just asserted).

2. **Composition correctness — exact match, parsed independently.** Parsed the real 85-row file
   directly: `clear_advance` 30 (18 `understanding`/12 `turns`), `clear_suspend` 40 (24/16),
   `boundary` 15 (9/6) — byte-exact match to plan §3.4's target table. No id collisions among the 85,
   `r1_probe: true` appears on exactly 7 rows (the pre-existing 5 + the plan's `ca-12`/`ca-15`), all
   `clear_advance` — confirming the Pass 2-verified `cs-10`/`cs-13` fix survived the merge into the
   real fixture untouched. Tier/`expected` invariants (`boundary`→false, `clear_advance`→true,
   `clear_suspend`→false) hold for all 85 rows.

3. **Harness edit correctness — exact, isolated, no collateral changes.** `git diff` on
   `test_guard_calibration_live.py` shows precisely five one-line changes at lines 223, 227, 228,
   232, 233 (`26`→`85`, `26`→`85`, `26 * K_REPLICATES`→`85 * K_REPLICATES`, `21`→`70`, `5`→`15`) —
   nothing else in the file touched. Matches plan §7 step 2 exactly, at the corrected line numbers
   Pass 2 already verified.

4. **New integrity test quality — closes F4 as designed, no correctness gap found.** Read
   `test_guard_set_integrity.py` (`server/tests/eval/test_guard_set_integrity.py`) in full: it is
   genuinely offline (no FalkorDB, no live marker, no LLM call — just `json.loads` over the file),
   checks unique ids, required-field presence, `tier`/`path` enum validity, `expected` is `bool`, the
   `boundary`-always-`false` invariant (parametrized per-row, so a single bad row fails with that
   row's id in the test name rather than a generic assert), and per-stratum/path counts expressed as
   `>=` floors (`test_stratum_and_path_minimum_counts`), matching `test_golden_set_integrity.py`'s
   own `30 <= len(rows) <= 50` precedent the plan cited. Ran it (see check 5 below) — all pass against
   the real 85-row file. One gap, not a correctness defect: it does not check the `r1_probe`
   narrower-semantics constraint Pass 1/2 fixed by hand (that `r1_probe: true` may only appear on
   `clear_advance` rows) — a future drafting pass could reintroduce the `cs-10`/`cs-13`-style mixup
   and this offline test would not catch it, only the live harness's diagnostic would notice (and
   only as a silently-diluted metric, not a failure). This is outside F4's stated scope (F4 asked for
   structural integrity, not semantic-field cross-validation) so it is not a blocker, but worth
   flagging as a natural follow-up given `r1_probe` is exactly the field this unit's own review chain
   already found mislabeled once.

5. **Test result honesty — independently re-run, both offline and live, both reproduce exactly.**
   Ran `pytest tests/eval/test_guard_calibration.py tests/eval/test_golden_set_integrity.py
   tests/eval/test_guard_set_integrity.py` myself (FalkorDB was up, port 6379) → **499 passed**,
   matching the implementer's claim exactly. LM Studio was also reachable in this environment
   (`qwen/qwen3-4b-2507` served at `localhost:1234`), so I additionally ran the live suite myself:
   `pytest -m live tests/eval/test_guard_calibration_live.py -s` → **1 passed in 154.93s**, printing
   **`G1 false-advance = 10.0% (n=40 cases / 120 calls) · G2 advance-recall = 86.7% (n=30 cases) ·
   VERDICT: wire`** — an exact reproduction of the implementer's reported numbers, on the identical
   fixture (the report's recorded `fixture sha256` matches the live file's hash, checked before and
   after my run). I additionally parsed the report's own full per-case table and recomputed both
   gate metrics from raw replicate decisions against `guard_calibration.py`'s actual formulas
   (`false_advance_rate`: per-*call*, not per-case-majority, over `clear_suspend` only — 12/120 =
   10.0%; `advance_recall`: 26/30 = 86.7% over `clear_advance`, and per-call agrees since the
   replicate flip-rate is reported/confirmed 0.0%) — both reproduce exactly from first principles,
   not just from re-running the harness. **On the G1=10.0%-exactly-at-boundary question the brief
   asked me to weigh in on:** the existing documentation already surfaces this adequately and in the
   right places — `HISTORY.md`'s new entry states "lands right at the ≤10% gate" in the same
   sentence as the number, and the coordination log (`docs/plans/golden-set-expansion-coordination.md`)
   has `teco`'s own explicit callout ("right at the ≤10% gate boundary, not comfortably under it...
   flagging for user visibility, not overriding"). The one place this could be tightened:
   `docs/BACKLOG.md`'s K-027 item 4 entry states the number ("G1 false-advance = 10.0% (n=40 cases /
   120 calls)") without the "lands right at the boundary" qualifier that both `HISTORY.md` and the
   coordination log carry — a reader who only opens `BACKLOG.md` (which is the living/forward-looking
   doc, more likely to be skimmed than `HISTORY.md`) would see a clean "10.0% ≤ 10%" pass without the
   boundary-proximity caveat. Minor, not a blocker: `BACKLOG.md`'s own convention is to be terse and
   defer detail to `HISTORY.md` ("see `HISTORY.md`" appears right in the same line), and the plan's
   own §8 item 2 (marked resolved, cited from `BACKLOG.md`'s own K-027 entry list) already carries the
   durable version of this caveat for anyone who follows the chain. Suggested improvement: add
   "(right at the gate line)" or similar to `BACKLOG.md`'s own G1 clause, since it costs four words and
   removes the one place in this delivery's paper trail where the boundary-proximity fact is silent.

6. **Documentation accuracy — every checked claim in `BACKLOG.md`/`HISTORY.md` matches the real
   artifacts.** Composition numbers, descope framing, five-literal-edit description, F4 closure
   claim, and the G1/G2 live numbers all check out against the real files (checks 1–5 above). One
   phrasing wrinkle beyond the boundary-callout above: `BACKLOG.md`'s compressed "`(n=40/120 calls)`"
   reads, out of context, like a fraction (as if 40 were a failure count out of 120), when it is
   actually "n=40 cases, 120 calls" — `HISTORY.md`'s fuller "(n=40 cases / 120 calls...)" and the test
   report's own "(n=40 cases / 120 calls)" don't have this ambiguity because they keep the word
   "cases". Nit-level (the real failure count, 12, is derivable and not misstated anywhere; a careful
   reader who knows k=3×40=120 will parse it correctly), but worth tightening for a future reader who
   doesn't do that arithmetic in their head.

7. **Mutation-test discipline — working tree is clean, no leftover debris.** `git status` /
   `git diff --stat` show exactly the six intended files changed (plus the pre-existing, unrelated
   `claude/docs/requirements/agent-permission-friction.md` modification from outside this unit's
   scope) and four new files (fixture-adjacent test, review, test report, coordination log) — no
   `.bak`/`.orig`/stray files found repo-wide. This matches the coordination log's own note that
   `teco` already independently re-ran one mutation check (`bd-05`'s `expected` flip) and confirmed
   clean restoration before dispatching this gate. Note for the record, not a defect: re-running the
   live test myself (step 5) caused pytest to rewrite `docs/test-reports/
   guard-judge-calibration-2026-08-21.md` in place with a fresh run's output (the test's own,
   documented side effect of writing its report) — the new content's verdict line and fixture hash
   match the original file's exactly, so this is a reproduction, not a mutation of substance.

### What's solid (implementation-specific)

- The **exactness of the fixture transcription** is the strongest finding in this pass — 59/59 rows
  byte-identical to the plan's draft, independently verified by parsing rather than reading, which
  rules out the most likely failure mode for a large content-heavy diff (silent retyping drift).
- The **five-literal-edit surgical precision** — nothing else in a 600+ line test file was touched,
  confirmed by `git diff`, not by trusting the implementer's description of the change.
- The **live run's reproducibility** — re-running it independently, on the same fixture (hash-verified),
  produced the identical verdict numbers, which is real evidence this is not a one-off favorable roll
  on a nondeterministic judge but a stable measurement at temperature 0.
- The **paper trail's honesty about the boundary-line result** — two of the three documents in the
  chain (`HISTORY.md`, the coordination log) state the "right at the gate" caveat explicitly and in
  the same breath as the number, which is exactly the right instinct for a result this close to the
  line.

### Verdict

**Approve.** The implementation faithfully and exactly realizes the Version-3 plan `analyst`
approved in Pass 2: fixture content and composition are byte-exact to the drafted rows, the harness
edit is surgical and matches §7 step 2 precisely, the new integrity test genuinely closes F4 in the
plan's own intended (inequality-based) style, and both the offline suite (499 passed) and the live
calibration run (G1=10.0%/120 calls, G2=86.7%/30 cases, verdict "wire") reproduce exactly under my
own independent re-run — not accepted on the implementer's word. `BACKLOG.md`/`HISTORY.md` claims all
check out against the real artifacts. Nothing here would embarrass a future reader of the closed
K-027 epic. One minor documentation nit (add the "at the gate boundary" caveat to `BACKLOG.md`'s own
G1 clause, and spell out "cases" rather than a bare "n=40/120" fraction) and one minor test-coverage
gap (the new offline integrity test doesn't encode the `r1_probe`-restricted-to-`clear_advance`
constraint, so a future drafting pass could reintroduce the same mixup this unit's own review chain
already caught once) are worth a follow-up but do not block the epic-closure claim.
