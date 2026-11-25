# Connected code graph: methodology review of the `ac5-gate-1` pre-registration record (CO-6b)

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** C-1001 (P1 close-out, CO-6b)

Reviewed, uncommitted: the last line of `cpg/eval/preregistration.jsonl` (`ccg-prereg-v1` / `ac5-gate-1`, 12th line, the 11 earlier lines unchanged) and `docs/plans/connected-code-graph-p1-closeout-prereg.md`. Read: root `REVIEW.md` (growth-stops: the note is new, no growth to report); `docs/plans/connected-code-graph-p1-closeout.md` §1-§3.5, §9-§10, CO-6b/CO-9 rows; `docs/plans/connected-code-graph-p1-closeout-ml.md`; `docs/plans/connected-code-graph-p1-ml.md` §5.3, §6.1-§6.2; `docs/plans/connected-code-graph-eval-protocol-ml.md` (EP) §9.4, §11, §12.1; `cpg/eval/tools/gate.py` (scoring and cap code only) and the header of `withheld_shapes.py`. No held-out file and no custodian directory was read. Recomputed: the record's pins (all 12 harness sha256 and the key-error file sha256 match the tree), the Wilson bounds and the `r^c` table (all agree to the printed digits).

**Verdict: needs changes** (2 blockers, 5 majors). The pins and arithmetic are sound; the defects are in what a pass or a miss will mean, and each is cheap to fix now because R is uncommitted and a harness edit before R only costs a re-pin.

**Falsifiable stopping condition.** The held-out list must not be asked while any of these holds: (S1) the record does not state whether a held-out item of a withheld shape is scored (B1); (S2) the record does not state whether held-out noise will be labelled (B2); (S3) the rehearsal report has no labelled dev `Q_scripts` and a bar-3 verdict, or has one above 4. This review is **refuted** on B1 if `gate.py`'s held-out scoring path is shown to drop withheld-shape items from the scored set (my reading below is from `withheld_counts`, `check_withheld_cap` and the only `item_withheld_shape` excusal at the dev probe; a planted withheld-shape item missing from a held-out-shaped fixture list that still yields `pass` would refute it).

## Findings

### B1 (blocker). On held-out a withheld-shape item is asked and scored, so the scoped claim is unreachable unless the draw holds none

Evidence. `gate.py` uses `item_withheld_shape` to **excuse** an item only in the dev probe (l.1449); on the held-out run it feeds `withheld_counts` (counts beside the hits, l.1127-1128) and `check_withheld_cap` (a refusal at 20 of 79), and `withhold_strata` withholds strata only. The record's own statement says it: "A held-out item of such a shape is scored `not-listed`, not excused" (`withheldShapeStatement`). The engine does not build these shapes, so such an item is a miss, and the verdict is "every item".

Yet the claim sentence ("listed every counting item ... **outside** the named withheld shapes"), the cap ("partial scope"), the plan's `docs/plans/connected-code-graph-p1-closeout.md` §9 ("The withheld script shapes are not a miss") and the predicate's "never spent knowingly" all read as if withheld items leave the scored set. They do not, and the three readings cannot all hold.

Size, labelled estimate: dev shows 10 of 49 (20%). If held-out resembles dev, about 16 of 79 items are withheld-shape (mean of a binomial(79, 10/49); no held-out count was read), the chance of **zero** such items is about 1e-8 under independence (clustering raises it, not to a usable level), and the chance of reaching the cap (20 or more) is about 0.17. So as built the spend is a near-certain miss, and the `r^c` table (§5) does not describe it: the table prices only misses on built shapes.

Fix, one of (recommend 1): (1) make the held-out scorer **not ask or score** withheld-shape scripts items (counts and shape letters beside the hits), net the floors, the cap and every bound to the scored set, and re-pin the harness before R; the claim sentence then holds as written. Objection to carry in the record, not hide: the withheld class is defined by what the engine cannot do (the plan's own objection to option (ii) of Q9); the cap, the verbatim statement and the per-shape counts are the mitigation. (2) Keep scoring, state that a pass requires every withheld-shape item to be listed anyway, state the expected outcome (miss) and its probability as above, and have the stakeholder confirm the spend is wanted as a measurement of the built shapes' misses. Under (2) the cap is disclosure only and the claim sentence loses "outside".

### B2 (blocker). Held-out noise: the record never says whether it will be labelled, and its own rule then makes the scripts claim fail

`scriptsReporting.labelledC` allows "labelling withheld from every human", and `falsified` fires on "Q unmeasured while the ceiling exceeds 4". The ceiling is about 16.9 on dev counts (`(610-34)/34`, from `docs/plans/connected-code-graph-p1-closeout-ml.md`), so with held-out labelling withheld the scripts claim is falsified before the run. The AC-5 noise bound (Q at most 4, EP §11.3) is a gate component, and neither the claim sentence nor the verdict rule says what a listing pass without a measured Q is. Arithmetic: with listed `L` = wrong + real, `Q <= 4` iff real items are at least `L/5`; at dev's 610 listed that is 122 real items against 34 gold-matched, so the bound needs the gold to be about 3.6 times incomplete; a pass of "listed every item" through an answer this large is weak evidence and the sentence must say so.

Fix: decide before R whether a non-pack-author labels held-out answers (an exposed dispatch; EP §11.4, the 20-label spot-check). If yes, state the sample (EP §11.2), seed string and bootstrap as for the rehearsal. If no, the claim sentence carries "noise bound not measured", burden block A and `hit@100` beside it, and the record says AC-5's noise clause stays open (not "met") whatever the listing result.

### M1 (major). The cap-exceeded branch is open-ended

`withheldCap.over` says "the stakeholder is asked". After the custodian's count is known, "raise the cap" is a post-contact degree of freedom. Pre-state the closed set: not spent; the options are stop, or build a shape and take a new F, record and draw; raising the cap or narrowing the predicate after the count is excluded. (Under B1 option 1 also say whether the cap's denominator, 79, is gross counting items; the record does say it is.)

### M2 (major). Dev labelled-Q at the rehearsal is not executable as written

The record says only "by a labeller who is not a pack author". Missing, all fixable by one paragraph in the record or the rehearsal brief:
- **Who:** a named human or a named blind agent role, not a pack author, never the extractor builder; the rubric is EP §11.4 ("some execution exists"), a judgment an LLM labeller can drift on, so a second labeller and the 20-label agreement need a pre-stated floor (none exists in EP §11.4).
- **How many:** EP §11.2 gives at least 100 possible items, minimum 5 per (reason, band) stratum, with a seed built from `gateId` and `snapshotId`: state the rehearsal `gateId` string, or the draw is a free parameter across reruns. State whether the 100 is per scripts stratum or pooled (dev scripts is 22 anchors of 610 listed items; a pooled 100 leaves scripts thin).
- **Bootstrap:** EP §11.3 resamples answers. With 22 scripts answers and a 124-item answer in the dev set the interval is wide and the percentile bootstrap is poorly calibrated; say the bar-3 decision is on the point estimate of Q (EP §11.3: "Gate: Q <= 4" observed) with the interval reported, or the lower bar is on the upper bound; pick one now.
- **Absent figure:** reading 3 of the entry criterion only covers "a figure outside a bar". State that a missing labelled `Q_scripts` is "not met", never "met on ceiling".

If the labelled `Q_scripts` exceeds 4: by the record's rule that is a rehearsal miss (item 3): fix, a new F, a new record, no spend; and `-closeout-ml.md`'s recommendation is then refuted on its own condition (a). The fix is tuning the resolver's `possible` edges against dev, which is allowed but must be disclosed as such and re-run, and it can cost built-shape recall; the record should say the iteration count is reported. Given the arithmetic in B2, I rate a first labelled Q above 4 as a live outcome, not a tail.

### M3 (major). Which noise bars gate the held-out verdict is not stated

`docs/plans/connected-code-graph-p1-ml.md` §5.3: gate 1 "stays gated at Q <= 4; the other three are reported there" and gate from the next gate on. The record's reading 3 and S1 wording treat all four bars as the entry bar, and `docs/plans/connected-code-graph-p1-closeout.md` §1 says the 5x bar holds in both baselines. Say in the record which of the four decide the held-out pass and which are reported; otherwise the reduction and precision bars become a post-hoc choice after the run.

### M4 (major). Hit rule and the coincident-only case

The verdict follows EP §5.2 (a coincident hit is a hit) while `falsified` calls the scripts claim false on any coincident-only item. The record does not say what the AC-5 verdict is then. State: AC-5 listing verdict = pass by the letter, and the sentence "the scripts resolver recovers cross-script links" is not written. Also the stakeholder question closing `docs/plans/connected-code-graph-p1-closeout-ml.md` (credit only concordant hits?) is not in the plan's answered list; the record adopts the default silently. Record the answer or mark the default as taken without one.

### M5 (major). Key-error rulings on dev do not say what they mean for held-out

Two of the three confirmed dev key-errors (`$ROOT` parent directory; a Python module target) are by-design own-directory limits of the resolver: items excused because the engine cannot do them. State that a held-out analogue is **scored as a miss** (EP §12.1 step 2 already makes "recall as run" the gate result and key-error exclusion a side figure), and that the dev rulings are not precedent for held-out triage. Otherwise the same exclusion can be asked after contact.

### m1 (minor). Ordering and the freeze-check's disclosure

The ordering F, R, freeze-check, rehearsal, run is dependency-valid. A better order costs nothing: **R, rehearsal, freeze-check, run** (the check still reads the path set ending at R). A rehearsal miss discards R and, as written, forces a second freeze-check on a new record; each is a held-out read in a custodian process. Doing it last means one read, for a record that survived.

Contamination to disclose in the record:
- The printed 12-hex values are item and anchor id prefixes (EP §9.4 probe set). They add no information to builders only if they are already in the committed manifest or dev set; state that, or that the check prints a hash only on a hit (a clean check prints counts).
- A contact hit moves its cluster to dev **after** the check certified the path set; say that `c`, the floors and the withheld counts are taken after that move, and that the check is re-run at the same R only if the move touched a path in the set.
- Commits after R (ledger, rehearsal report, this review) sit outside the path set; say that no builder dispatch reads held-out material after R.

The stakeholder question in §6 of the note (may the custodian run it) should be asked with this ordering.

### m2 (minor). `c` and the bounds

`claimStatedAfterTheRun` with `c` "at spend time" lets `c` be reported after the result. Commit `c` and the withheld counts in a ledger line before the ask. `c` at most 58 holds only if dropping wiring removed no cluster and, under B1 option 1, withholding removes none; item bounds (0.975 over 151, per stratum) are over the gross count and must be recomputed over the scored set. The bound is a cluster-level recall bound against the writers' gold (incomplete, one distribution); the claim sentence handles this ("not ... shapes absent from the writers' draw").

### m3 (minor). Dev-fit leak paths into the claim

None changes the numbers; each should be named in `devPassCaveat` or the note:
- The withheld set and the predicate (`withheld_shapes.py`) were written after the CO-1 read of all 49 dev items; it is a dev-derived scope. It is pinned in the harness and outside the no-workaround scan's directories, so a dev id inside it would not be caught; add it to the scan, expecting zero hits, and report on dev the flagged set against engine status (a flagged item the engine lists means over-withholding).
- The scan is literal-only (473 literals): it cannot see a dev-shaped regex or token list. Its three hits are the author's reading; the analyst gate decides (the note says so).
- The dev rehearsal saturates by construction (every dev item must be a hit, a withheld shape or a key-error, after fixes read from dev); its pass carries no information about held-out recall, and the `r^c` values 0.95 to 0.99 are priors, not estimates.
- The 3 key-errors and the 10 withheld items leave 36 of 49 as the scripts dev evidence; the held-out claim is about F only (CO-10/11 change core after the gate, `docs/plans/connected-code-graph-p1-closeout.md` §5).

### n1 (nit)

`ccg-noise-ratify-v1` is cited by name with no file; the four bar values are in reading 3, so the record is self-sufficient. A hash-less citation to an editable section is the only exposure; say so or write the file.

## What can and cannot be said

**If the run passes** (and B1, B2 are fixed as above): "the frozen engine at F listed every scored counting item of an untouched draw", with: withheld counts by stratum and shape, `c` and the zero-miss bound `c/(c+3.84)` (0.929 at c = 50, about 0.938 at the ceiling 58, 95% two-sided Wilson, clusters independent), the concordant/coincident split, burden block A and `hit@100`, and "noise bound measured / not measured". It supports nothing on `wiring`, `flow`, withheld shapes, shapes absent from the writers' draw, noise (unless labelled), real-component recall, or the engine after CO-10/11, and not "scripts recall". It is a bound on gold-listed clusters, not a point recall.

**If it misses:** the first run is the result (EP §12.1 step 1); no recall figure beyond as-run; the miss is triaged by category in counts; missed clusters go to dev, the draw is burned and stated, the fresh floor is 35 counting items, and the successor takes id `ac5-gate-1-r<k>`. A miss says the true cluster recall is not near 1 on the built shapes; with c about 50 one miss is the expected outcome unless true recall is above about 0.99 (arithmetic, unmeasured). Under B1 as built a miss can be wholly a withheld-shape miss, which carries no information about the built shapes: report per-shape counts so the triage can separate them.

## Answers to the five questions in the brief

1. Mostly pre-stated (pins, floors, cap number, bound formula, Q4, successor id). Not stated: withheld scoring (B1), the cap branch (M1), held-out noise (B2), which noise bars gate (M3), key-error precedent (M5), `c` ahead of the run (m2).
2. The scoped sentence is not honest as scoped until B1 and B2 are fixed; the n of 151 is gross; the cluster count is unknown but bounded by a formula that is right.
3. Not yet executable (M2). A labelled dev Q above 4 is a rehearsal miss (item 3): fix, new F, new record, no spend.
4. Ordering is valid, better as R, rehearsal, freeze-check, run (m1); disclosure items are listed there.
5. Paths in m3; the entry rule's key-error leg is M5.

## Pass 2

Read: the revised record (last line of `cpg/eval/preregistration.jsonl`), `docs/plans/connected-code-graph-p1-closeout-prereg.md`, close-out plan §3.5, `cpg/eval/results/co6b/no_workaround_scan.py` (header and definition), the diff of harness commit `4aa24d09` (`gate.py` `withhold_shapes`; the 9 test names in `cpg/tests/test_gate_withheld_scoring.py`, which include the planted unlisted-withheld item, a decoy and forged shape fields; not run, `pytest` is not installed here). Recomputed: all 12 harness sha256 and the scan-script sha256 match the tree. No held-out read.

**Verdict: approve with suggestions.** Every blocker and major of pass 1 is fixed in text or code; the stopping conditions S1 to S3 are met by the record. Four new majors below are rehearsal-protocol and successor-rule gaps that are fixable in the record or the rehearsal brief before R is committed; none blocks the pins.

### Dispositions

- **B1:** fixed. `withhold_shapes` drops withheld-shape counting scripts items before the ask and the score; floors, bounds and cap are over the kept set; `score` refuses a list that does not reproduce the recorded shape counts; the record's statement and `scoredSet` agree. My refutation condition is covered by a test name (planted unlisted withheld item is not a miss; a decoy is).
- **B2:** fixed by the stakeholder decision: held-out noise not measured, the claim carries it, the AC-5 noise clause carries to P2, the verdict is the listing verdict only (`noise.whichBarsDecideWhat`). See N2 for the interactions.
- **M1:** fixed (`overCapOutcomes`: closed set). See N3 for the fresh-draw branch.
- **M2:** fixed in `devLabelling` (labeller, `gateId`/seed, two samples, point-estimate rule, absent Q = not met). See N1.
- **M3:** fixed (only the listing verdict decides; S1 reported). Residual in N2.
- **M4:** fixed as a disclosed reading, awaiting objection; the requirements log carries it.
- **M5:** fixed (`keyErrorPrecedent`).
- **m1:** fixed (order R, rehearsal, freeze-check once, ledger line, run; the contact-hit cost is stated).
- **m2:** fixed (`c` and counts in a ledger line before the ask; bounds over the scored set). The `c <= 58` wording is judged in N5.
- **m3:** fixed in part: the predicate is in the scan (3 hits, none in it), the flagged-versus-engine-status block exists, `devPassCaveat` names the dev-derived scope. The scan stays literal-only and is disclosed so.
- **n:** fixed (`noiseRatifyCitation`: log entry, no hash, said so).

### New findings

**N1 (major). The rehearsal seed is re-rollable.** `gateIdAndSeed` makes `k` the rehearsal ordinal "at this tree" and judges bar 3 "on the last run at the tree". Re-running the labelled draw at an unchanged tree gives a new `k`, a new seed and a new sample; rerunning until a point estimate falls at or under 4 is optional stopping on a point-estimate gate. Pre-state: a rerun at an unchanged tree reuses the same seed and the same labels (so it cannot change Q), or bar 3 uses the **largest** Q over runs at the tree. Also: the two labellers are LLM agent roles, so a 18-of-20 agreement is agreement of correlated judges, not independent human validation; the larger-Q rule below the floor is the right conservatism. I accept the floor, and say in the record that the labelled Q is a dev fit statistic that carries nothing about held-out noise.

**N2 (major). Dev bar 3 versus "noise not measured".** The pieces fit (listing-only verdict, claim sentence "noise not measured", ceiling B an upper bound never reported as noise), with three loose ends:
- The dev labelled Q gates the spend, yet the claim does not rest on it; say in `claimSentence` or `noise` that dev Q is not noise evidence for the claimed draw.
- The ratified S1 5x bar and the possible-precision and p90 bars become reported-not-gating at the rehearsal and, on held-out, "only if the custodian's baseline exists". `-p1-ml.md` §5.3 supports gate 1 on Q only, but the plan's §1 says S1 holds in both baselines; state that the bars are deferred to P2's first gate, not waived, and name the gate that enforces them.
- `whichBarsDecideWhat` adds "no unreviewed masked comparison" with no definition in the note; cite the harness's `blocked` class.

**N3 (major). The cap's fresh-draw branch and the Q4 re-gate interact in undefined ways.**
- Outcome (b), "draw fresh scripts items", selects a new list because of a predicate count. State how many redraws are allowed, that each redraw and its count are reported (a redraw until under the cap is a forking path), what the fresh supply is, and whether the withheld old list's scripts items may return as fresh candidates (recommend: not).
- The ledger line carries `w` by shape letter before the ask. If builders read it, an over-cap count tells them which unbuilt shape the held-out draw holds. Say that the shape-letter counts go to custodian and `teco` only and are not a builder input for the fresh draw (the record's "no builder dispatch reads held-out material after R" does not obviously cover a derived count).
- For a re-gate, say whether the 35-item fresh floor and the 25% cap are over the scored set (they should be).

**N4 (major). Successor rules leave two spent outcomes unassigned.** `q4` and `ordering` cover: rehearsal miss (new `gate`), spent miss (`re-gate` with `movedOut`), cap refusal (nothing spent). `gate_kind` accepts `re-gate` with `{}` counts, but the record does not say which kind follows a spent run with no verdict: `minimums-not-met`, a `blocked` run, or a harness error after the once-only marker. Pre-state: such a run is a re-gate with `movedOut` `{}` and the burned draw stated, or a first-gate successor if nothing was asked. Also two ordinals named `r<k>` (record ids `ac5-gate-1-r<k>`, rehearsal ids `ac5-gate-1-dev-r<k>`) mean different things; harmless to the harness, a trap for a reader. The freeze-check permission is "once" per stakeholder: a successor record needs its own permission; say so.

**N5 (judgement on the author's `c <= 58` correction). Right in direction, conditional in fact.** The author is right that removing items cannot raise the bound's `c` **when `c` counts the original S47 clusters that still hold at least one scored item**: the set only shrinks, so 58 stays a ceiling. My earlier wording ("holds only if no cluster was removed") was wrong in that case. The ceiling fails in two cases the record's unconditional sentence does not exclude: (a) if the custodian **re-clusters** the scored set, a withheld item that bridged two anchors can split one cluster into two and raise `c`; (b) on any fresh draw (cap branch (b), a re-gate) there is no 58 at all. Fix the wording: `c` is the count of original S47 cluster ids with at least one scored item, 58 is a ceiling for the first list only, and a fresh draw states its own `c`. The formula itself holds for any `c` if clusters are independent units.

**m4 (minor).** A contact hit after the rehearsal moves clusters to dev, changes the dev gold hash, and sends the record back through a rehearsal on a gold that now holds contacted items the engine was never fit to; a miss there is a new F, a new record and a new freeze-check. The record states the cost; say also that the `once` permission would then be renewed.

**n2 (nit).** The no-workaround item still reads "the three are disclosed", while the criterion's letter ("finds no literal") is unmet; the record admits it and defers to the analyst. Keep that wording; do not let a later document call item 2 "met".

### Stopping condition (revised)

Do not ask the held-out list if any of these hold: no labelled dev `Q_pooled` and `Q_scripts` at the final tree, or either above 4 (and, per N1, the largest over reruns at that tree); a rehearsal miss; no clean freeze-check at R; no ledger line with `c`, `w` by shape letter and the run count; `w >= 20` (stop: closed outcomes (a) or (b)). Falsified if a re-run of the withheld-scoring tests with the planted unlisted withheld item and a decoy does not give "not a miss" and "miss" respectively.

### Proposed REVIEW.md lines

None proposed. Reason: the class of finding here (a gate scorer's behaviour for a scoped-out item differing from the claim text) is caught by the existing "every check can fail and pass" line, and the harness test now plants the item.

## Pass 3

Scope: record `ac5-gate-1-r2` (12th line of `cpg/eval/preregistration.jsonl`) and `docs/plans/connected-code-graph-p1-closeout-prereg.md`, read against `docs/test-reports/connected-code-graph-p1-contact-move-report.md` and the dev files (`cpg/eval/dev/ac5-*.jsonl`, dev data only). Recomputed: the Wilson bound at c = 52 (0.9312), the `r^52` row (0.59, 0.35, 0.21, 0.07), the cap threshold (17 of 64), 120 = 26+30+64, 55 − 3 = 52. No held-out read.

**Verdict: approve with suggestions.** The pure move is sound and the arithmetic is right. Open: one disclosure the claim needs (the selection effect and the thin mcp stratum), one report rule worth adding, and a cap-risk statement.

### Dispositions of pass 2

N1 fixed (same seed and labels on a rerun, the largest Q binds, `gateId` and `t2` separated). N2 fixed (dev Q "not noise evidence", the deferred bars named, HA-5 defined). N3 fixed in the record (one redraw at most, old items excluded, shape-letter counts custodian-only, fresh floor and cap over the re-gate's own sets). N4 fixed (`onSpentRunWithoutVerdict`, `r<k>` versus `t<k>`, renewed permission per successor). N5 fixed (c = original S47 cluster ids with at least one scored item, no re-clustering; 58 first list only). m4 and n2 stand as written.

### Findings

**P3-1 (major). The remaining held-out is a conditioned subset; the claim must say so, and the mcp stratum is nearly one unit.** A pure move is the right route: contacted clusters must leave held-out (EP §8, `-p1-ml.md` §6.2), a re-draw would have shifted every gated count and reopened C (the report's dry run: 26 items pulled in), and held-out strictly subset of the S47 list adds no new selection step by the investigator. It still leaves a selection effect the record does not disclose:
- The moved clusters are exactly the ones whose probe values appear in committed documents. The remainder is "the S47 draw conditioned on not being named in the repo", so it over-represents obscure anchors. The direction of the bias on engine recall is unknown (obscure anchors may be less prose-mentioned, or less regular). Say: "the untouched remainder of the S47 draw after the contact move", not "an untouched draw", and no "random draw" reading.
- The shift is large in mcp: held-out mcp fell 46 to 30 items and 4 to 2 anchors (contact-move report); scripts went 36 to 35 anchors, wiring 7 to 3. Any mcp statement rests on 2 anchors; a per-stratum zero-miss bound at anchor level is at most `2/(2+3.84)` = 0.34. The record's item bound for mcp (0.886) must not be quoted alone: report anchors and clusters beside each per-stratum count.
- The freeze-check detects verbatim probes (ids, qnames, paths, 6-word shingles). The 7 hits in 4 committed files show documents did carry held-out material; a paraphrase or a count-only trace is invisible to it. The clean R2 result supports "no probe literal in the path set", not "no contact"; the claim sentence's "untouched" is bounded by that.
Suggested fix: add these three clauses to `claimSentence` and the note's §6; no new gate.

**P3-2 (major). Report the first-exposure result on the 36 moved items, descriptively, before any fix.** The 31 mcp and scripts items were never asked; the t2 rehearsal is their first scoring and the only fit-free look at held-out-like items the project has. Once the engine is tuned to a miss they are fit. Pre-state: the first run at R2 reports hits, withheld shapes and misses on the 31 gated moved items separately (counts, by stratum), before triage. Do not over-read it: it is 3 clusters (the report: scripts 1 new anchor, mcp 2), a zero-miss bound over 3 units is `3/(3+3.84)` = 0.44, and the items were named in documents builders could read (the freeze-check hits), so it is an optimistic first exposure at best. It informs the `r` prior, nothing more.

**P3-3 (major). The dev predicate share is a warning for the cap, and the record should state the chance.** Dev flags 14 of 64 (21.9%; 10 of 49 at 20.4% before the move; the 15 moved scripts items hold 4 flagged, all family C, 26.7%). The held-out cap refuses at 17 of 64 (26.6%). Labelled estimates, binomial and independent items, and dev is not a random sample of held-out: at p = 0.219 the expected held-out withheld count is about 14 and P(17 or more) is about 0.22; at the moved items' 0.267, about 0.55; at the pre-move 0.204, about 0.14. Clustering fattens the tail. So a cap refusal is a roughly one-in-five outcome, not a tail, and the claim "the cap bounds the scope" has a thin margin (3 items of headroom at the mean). Say so in the record, restate the pre-committed response (closed outcomes a and b, at most one redraw, redraw reported) and note that a redraw is a new blind supply of scripts items whose existence the record does not establish; add the supply count to the ledger line.

**P3-4 (minor). `c <= 52` provenance is honest; add a commitment.** "Derived in memory by the custodian, not printed by any verb" is the right label, and it is consistent with the first record's three-stratum 55 (the custodian's "previously 55", where 58 was the four-stratum ceiling): 55 − 3 = 52. It is still a number no one else can recompute, and the verdict-time `c` carries the same limit. Put in the ledger line a hash of the sorted list of the counted cluster ids (counts-only compatible, custodian-held), so a later audit can check `c` without exposing ids. The `c/(c+3.84)` bound and a table at c = 52 are enough; the stale `c = 50` column is harmless, delete it for tidiness.

**P3-5 (minor). Dev labelling at t2 is sound; two numbers are stale.** The frame is only slightly larger: dev scripts is now 64 items in 23 answers (22 before; the 15 new items sit in one new anchor), mcp 36 in 5. So the sample sizes (pooled at least 100, scripts at least 60, minimum 5 per stratum, census if fewer), the per-tree seed and the rerun rule all stand, and the bootstrap remains poor at about 23 answers. Update the record's wording: the "124-item answer at t1" is a t1 figure; one new answer may carry 15 gold items and a large listing, so report the largest-answer share at t2 (block A already does) and note that Q is read on the point estimate with that share beside it. A new tree (t3) re-draws the sample by design; state that unchanged (item, answer) pairs reuse their t2 labels so a fix cannot buy a new Q on the same answers.

**P3-6 (minor). Superseded-record handling: three loose ends.**
- Nothing in `gate.py` or `gate_run.py` reads `supersedes` (grep: no "supersed"). The old record `ac5-gate-1` stays valid to the harness and the old custody dir `heldout/` (with the contacted clusters) still exists. A run at the old id with the old lists is blocked only if the harness's manifest check rejects the old lists against the committed manifest, which I did not run. Ask the custodian to retire `heldout/` (rename it outside the run path) and have `qa-engineer` run the preflight at `ac5-gate-1` once as a negative control; it must refuse.
- The contact-move report notes the S47 probe map (`s47-probe-map.json`) the harness requires under the custodian area is not listed under `custodian-p1/`; confirm it is present for `heldout2/` before R2's spend, or the run refuses at the start.
- `heldoutManifestSha256` and `devGoldSha256` are pinned in R2; say that the harness compares them to the files at the run, or that the custodian and `teco` do so in the spend checklist (the note leaves it unsaid for these two fields).
Kinds are otherwise coherent: R2 is `kind: gate` (nothing was spent at R), a rehearsal miss gives `ac5-gate-1-r3` kind `gate`, a spent miss or a spent no-verdict run gives `re-gate`, and the freeze-check permission is per successor.

**n3 (nit).** The scan's fourth literal hit (`query`) is a 5-character generic token matched on a substring basis; it needs the analyst's ruling the note already asks for. The scan remains a bound, not proof.

### Stopping condition (revised)

Do not ask the held-out list if any of these hold: no clean freeze-check at R2; no labelled dev `Q_pooled` and `Q_scripts` at the final tree, or either above 4 (largest across reruns); a rehearsal miss; no ledger line with `c` (and the id-list hash), the total `w` and the run count; `w >= 17` (closed outcomes a or b only); `heldout/` still runnable, or the probe map missing under the custodian root. Falsified if the first-exposure counts of P3-2 are not in the rehearsal report before the first fix commit, or if the negative-control preflight at `ac5-gate-1` does not refuse.
