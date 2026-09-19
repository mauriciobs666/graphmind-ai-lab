# K-030 Track 2 Stage 8 Phase 1 — golden-set design + pilot calibration review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (Stage 8 Phase 1)

## Scope & verdict

Diff-scoped review of `data-scientist`'s Stage 8 Phase 1 deliverable (commits `dc52a3b`,
`991b987`): the "Stage 8 Phase 1" section added to
`claude/docs/plans/agent-knowledge-base-strategy-ml.md` (~lines 393-709 — 45-pair AC-2 golden-set
design across all 13 migrated KBs, a 16-pair live pilot against `ws:agent-team`, and the resulting
0.42 score-floor calibration), plus the corresponding `skills/agent-kb-retrieval/SKILL.md` floor
section update. Baseline for grounding checks: `claude/docs/reviews/agent-knowledge-base-strategy4-
stage6.md` Appendix A (the split-family source list) and the coordination ledger
(`claude/docs/plans/agent-knowledge-base-strategy4-coordination.md`) row for what `teco` already
independently re-verified (Wilson CI, 5 documentId spot-checks, pilot-count cross-tally, floor
arithmetic, `audit-team.sh`). Per the brief, I did not re-derive those checks; I recomputed the
Wilson CI/MRR independently by a different method, spot-checked a disjoint sample of 11 documentIds
(falkordb-quirks, coordination-techniques, review-techniques, architect, data-scientist,
graph-dba-reference), live-reran two negative queries and the R3/h39 crowding-out claim against the
real `search_documents` tool, and judged the design/derivation/consistency questions the brief
raised.

**Verdict: approve with suggestions.** Nothing found here should block Stage 8 Phase 2
(`qa-engineer`'s full-set gate) from starting. One Minor internal-consistency defect in the plan's
prose (not the design table, not the floor, not SKILL.md) and two Minor observations worth a
one-line fix each; no blockers, no majors.

**CPG:** not applicable — this is a documentation/ML-methodology deliverable (golden-set design,
pilot calibration, skill-doc prose) with no code-level component to graph.

## Findings

### Minor — "6 plus one held in reserve" misstates the stratum-(e) design; the table actually uses all 7 named families

ml.md:417-418 reads: "a full 6-family stratum-(e) set (Stage 6 review Appendix A names 7 real
split families; I used 6 plus one held in reserve)". I counted the rows actually tagged `e` in the
design table: R1(h13), R2(h21), R3(h39), R4(h14), R5(h22), R6(h40) in `review-techniques.md`'s
section, **plus Q1** (the qa-testing-techniques.md model-bench-attest family) in that KB's own
section — also tagged `a,e` (ml.md:484) and explicitly folded into the stratum-(e) set-recall table
during pilot execution ("Set-recall, stratum (e) (R1, R2, R3, **Q1** — 4 families..." ml.md:591).
That's **all 7** of Appendix A's named families represented as design rows — zero held in reserve
at the design level. (Separately, the *pilot* did execute only 4 of the 7 — R1/R2/R3/Q1 — with
R4/R5/R6 unexecuted; that's a true "4 piloted, 3 not yet" framing, but it isn't what the sentence
says either.) The design table itself is correct and complete — this is a prose-accuracy defect in
one sentence, not a coverage gap. **Suggested fix:** replace ml.md:417-418's parenthetical with
something like "Stage 6 review Appendix A names 7 real split families; all 7 appear as design rows
(R1-R6 + Q1), of which 4 (R1, R2, R3, Q1) were executed in this pilot."

### Minor — R9's query paraphrases its target claim's title closely

The "author independently of cobb" discipline (ml.md:430-436) is explicit that a query should be
grounded in the claim's real `Origin:`/worked-example text, "not a synthetic rewording of the
stored claim itself." R9's query — "A repo-wide lint/smell sweep flagged a violation in a file my
current diff never touched. Should I treat that as something my change introduced, or could it
have been sitting there before my diff even started?" — sits very close to its target's own title,
"A repo-wide smell check can flag a pre-existing violation the diff never touched" (verified live,
documentId `6d88fecb435547dea1d93d67b8995356`), reusing "repo-wide," "smell," and "diff never
touched" nearly verbatim rather than the Origin section's concrete incident (`AGENTS.md`'s
`awk length($0)>700` check, `verify_salesperson.sh`, S11/`start_demo.sh`). This doesn't invalidate
R9's single result (it was the pilot's one rank-2, not a suspiciously easy rank-1, so it isn't
inflating the pilot's numbers), but it's the one design row where the independence discipline the
doc itself sets as the bar visibly slipped. **Suggested fix:** no rerun needed now; when Phase 2 or
a future revision touches this row, reword it around the `AGENTS.md`/`verify_salesperson.sh`
incident instead.

### Observation (no severity — informational) — R1's query names "K-028" and echoes "mandatory default fallback"

R1's query cites "the K-028 v2-to-v3 change" and "a mandatory default fallback transition," both
close to the target document's own `Origin:` line ("`falkor-chat` K-028 workflow-timers, v2→v3,"
"mandatory default fallback arm" — verified live, documentId `d41d743e5c344eccbf46715f54cbab44`).
Unlike R9 above, this is squarely inside the stated methodology (ml.md:430-436 explicitly grounds
queries in the real historical incident's `Origin:` text, not the claim's title/fact wording) —
citing the real incident by name is what "grounded in the real Origin text" is supposed to look
like, not a violation of it. Flagging only so the distinction (Origin-text grounding is in-bounds;
title/fact paraphrase, as in R9, is not) is legible to whoever reviews the remaining 29 rows in
Phase 2.

## What's solid

- **Wilson CI and MRR independently reproduced.** Computed both from scratch via a Python
  closed-form Wilson-score implementation (not by-hand algebra): n=8, x=7 → CI **[0.5291, 0.9776]**
  (rounds to the doc's stated [0.529, 0.978]); MRR over ranks [1,1,1,1,1,1,0.5,0] = **0.8125**,
  exact match.
- **Disjoint spot-check sample, 11 documentIds, all real/ready/on-target.** Checked F4, F5 (both
  falkordb-quirks, unchecked by teco's 5), C3, C4 (coordination-techniques), the R4/h14 pair
  (review-techniques), and S1/D1/D2/P1/P2 (data-scientist, graph-dba-reference, architect) directly
  against `ws:agent-team` — every documentId exists, is `status: ready`, and its title plausibly
  answers the row's query. No orphaned or misfiled documentId found.
- **R3/h39 crowding-out finding independently reproduced live**, not just trusted from the report.
  Re-ran R3's exact prefixed query against `search_documents`: the top 5 came back
  `5a6763497f2740e8aa21efac72b3a1c6` at seq0 (0.232), seq2 (0.311), and seq1 (0.311), then two
  different documents (0.358, 0.362) — i.e. the *same* document occupies 3 of the top-5 slots
  (seq0/1/2) and the true second sibling `ccefd08d853247cd941e...` never appears in the top 5. This
  is an exact, independent reproduction of the doc's claimed mechanism, not a restatement of it.
- **Negative-query scores independently reproduced.** Re-ran N3 and N4's exact prefixed queries
  live: N3's top score 0.5523 (doc: 0.552) and N4's top score 0.4459 (doc: 0.446, correctly the
  React-18-batching claim) both match, and in both cases the actual top hits are genuinely
  off-topic for the query (WSL2/LM-Studio connectivity notes for N3's GPU-passthrough ask; a
  React-18 batching/render-timing claim for N4's RSC/Next.js ask) — the negative-stratum design is
  sound, not accidentally answerable.
- **Golden-set design is well-stratified and proportional.** Row counts by KB (81/332→9,
  86/332→9, 39/332→5, floor-of-1 for the ten small KBs) track the manifest's actual claim counts;
  all five of Recommendation 4's axes (KB-weighting, code/prose split, negative, near-dup,
  multi-facet/sibling) are represented; the 45 rows sum exactly against the per-section counts
  (9+9+5+2+2+2+1+1+1+1+2+2+2+6=45); all 13 KBs get at least 1 row.
- **Floor derivation is methodologically sound and honestly caveated.** Max-surviving-true-positive
  (0.409) vs. min-false-match-on-a-genuine-negative (0.446) is the correct direction and the right
  method; the doc discloses its own same-sample/thin-margin limitation in as much detail as I
  would have wanted to add myself, and correctly routes the actual out-of-sample validation to
  Phase 2's 29 untouched rows rather than overclaiming finality.
- **SKILL.md is internally consistent.** Frontmatter description, floor section, and calling-
  convention step 4 all state 0.42 and the "first, provisional" framing consistently; grepped for
  "disabled"/"provisional" — no stale "disabled" language survives anywhere in the file.
- **Near-dup discrimination (R7, F1) held cleanly** in both live-reproduced and reported form: the
  intended distractor sibling didn't appear anywhere in either top-5, not merely rank below the
  correct answer.

## Open questions

None — the one genuinely open question (whether the h39 crowding-out is systematic or a one-off)
is already correctly flagged as Phase 2's job in the plan's own Risks section, not something this
review needs to adjudicate further.
