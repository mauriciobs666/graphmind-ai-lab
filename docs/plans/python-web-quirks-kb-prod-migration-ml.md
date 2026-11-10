# python-web-quirks KB production migration: the pre-registered paired no-regression rule

> **Status:** active · **Owner:** `data-scientist` · **Tracks:** — (W7b, step 0b) · **Extends:** `docs/plans/python-web-quirks-kb-prod-migration.md`

**Frozen** when `qa-engineer` records this file's sha256 in the ledger, before the first scored run of the 45-pair runner (W7a). After that no number, set membership or definition below may change; a change voids the check (a different rule, written before a new first run, not a revision). Nothing here depends on W7a's output.

## 1. Decision served and what the rule can support

Applied at step 0c (`:8200` 0.6b vs `:8210` 4b, read-only, before any outage) and again at step 6.3 (`:8200` on the 4b vs the step-0c 0.6b record). Red at 0c ends the plan with no outage; red at 6.3 means R1.

**It supports:** "no gross regression of floor-applied top-5 retrieval on the Stage 8 single-answer pairs, with a stated chance of missing a smaller loss". **It cannot support:** "no regression in production", "no pair lost" (a loss of up to 3 net pairs passes), or any claim about queries unlike these 32. The pairs are the author-chosen Stage 8 set, 4 of whose rows were its floor-calibration pilot (`claude/docs/plans/agent-knowledge-base-strategy-ml.md` "Stage 8 Phase 1": the 29 other rows were out-of-sample for the 0.43 floor, the 4b floor was never calibrated on them). The only instrument for a production-quality claim is a new sealed set on the post-migration corpus (B4), optional, not a gate (`docs/plans/python-web-quirks-kb-prod-migration.md` §1).

## 2. Unit of analysis

- **Judged unit: the single-answer pairs, expected n = 32** (`docs/reviews/python-web-quirks-kb-prod-migration.md` M2; AC-2 report: 45 pairs, 32 applicable). Each is one binary outcome per configuration: **hit = the expected `documentId` is in the document-level top-5 (`limit=5`, the skill's literal prefix) AND its best score <= that configuration's floor** (0.6b: 0.43; 4b: 0.42).
- **Excluded from judgment, reported:** the 7 multi-facet family rows (R1-R6, Q1; 14 sibling documents, scored by set-recall; R6's second sibling is the known floor-unstable document) and the 6 negatives N1-N6 (handled by §4). 45 - 7 - 6 = 32. A pair is "non-applicable" only by this answer class in the fixture, never by outcome.
- **The count is a check, not an input:** the fixture must yield exactly 32 judged pairs, 7 families, 6 negatives. Any other count, or any expected id not `ready` in the target, makes the run **invalid** (stop and ask; do not drop the pair).
- **Pairs missed at baseline stay in the table** (concordant misses or gains); nothing is excluded after seeing a result.

## 3. The paired table and the decision rule

Per judged pair, two outcomes on one table: A = 0.6b at 0.43, B = 4b at 0.42, same-day corpus for step 0c. Cells: both hit, **b** = found at A and lost at B, **c** = lost at A and found at B, both miss. Net loss = b - c = (hits at A) - (hits at B).

**STOP (red) if net loss b - c >= 4 of 32** (about 12.5 points of the judged set), **or the §4 negative check fails.** Otherwise green. Chosen over a McNemar-style test because the exact one-sided sign test on discordant pairs cannot fire until b >= 5 with c = 0 (p = 0.031; b=4, c=0 gives 0.0625; b=5, c=1 gives 0.109): at this n a significance rule would miss everything short of a collapse. This is a pre-set practical bound, with the exact sign p reported alongside, descriptively.

**Operating characteristics (exact binomial, my computation, a stylized model: each of the F=28 pairs found at the 0.6b is lost independently with probability p, each of the M=4 pairs missed is gained with probability g; F=28, M=4 are the recorded 2026-09-19 figures, an assumption; the real baseline is measured at 0c and the rule does not depend on it; real pair outcomes are correlated by corpus region, so tails are fatter than shown):**

| scenario | expected b | P(red) |
|---|---|---|
| no change | 0 | 0 |
| symmetric turnover, p=0.07, g=0.5 | 2.0 | 0.024 |
| symmetric turnover, p=0.11, g=0.77 | 3.1 | 0.038 |
| symmetric turnover, p=0.14, g=1.0 (every miss gained) | 3.9 | 0.034 |
| true loss, p=0.07, no gains | 2.0 | 0.129 |
| true loss, p=0.11 | 3.1 | 0.372 |
| true loss, p=0.14 | 3.9 | 0.565 |
| true loss, p=0.18 | 5.0 | 0.768 |
| true loss, p=0.25 | 7.0 | 0.945 |

Reading: false alarm under benign swap-turnover is about 4 percent or less; the rule catches a loss of about a quarter of the found pairs 95 percent of the time, a loss of about one in seven only about 57 percent, and a loss of 2 to 3 pairs with no gains mostly passes (13 to 37 percent). Raising the bar (net >= 5) would cut power further; lowering it (net >= 3) buys 20 more points of power at 7 to 10 percent false alarm under symmetric turnover (the stop at 0c is cheap, at 6.3 it costs an outage): I keep 4 because a red at 6.3 triggers a second outage.

## 4. Negatives, floor-unstable rows, and what is reported but not judged

- **Negatives (judged, hard guard).** The 6 negatives N1-N6: a negative is **admitted** if its top-1 score <= the configuration's floor. **Red if the 4b/0.42 admits more negatives than the 0.6b/0.43 did** (expected 0 at baseline; closest observed pooled negative 0.446 at the 0.6b). n=6 cannot be calibrated; this is a gross guard only.
- **Reported, not judged:** raw (floor-free) top-5 table in the same b/c form (separates ranking loss from floor loss; a floor-applied hit implies a raw hit, so a raw net loss >= 4 would already be red); every lost pair (b) with its raw rank, best score and the 4b score against 0.42, and any whose 4b score lies in (0.42, 0.43] (to the stakeholder, per the plan; they still count as lost); **borderline pairs**, any judged pair or family sibling whose score is within 0.02 of its floor on either side (R6's second sibling ranged 0.4201 to 0.4405 across sessions, `claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md`, "Floor derivation"; its two scores on both models are recorded); family set-recall per family; closest-negative score and its margin to the floor (the B3 staging closest near-negative was 0.422 against 0.42); hit rates with Wilson 95% intervals (the lab's convention); the exact sign p.
- **Baseline sanity (not a stop):** the 0.6b was 28/32 (floor-applied equals raw) on 2026-09-19; a same-day baseline differing by 3 or more is reported as corpus drift.
- **Single run, no rescue.** Each configuration is scored once; no retries, re-labelling, floor adjustment or pair removal after seeing results. A rerun is allowed only for an infrastructure fault (embedding timeout, connection refusal), declared in the report, and then replaces the faulted run entirely.

## 5. On red

Default R1 at 6.3 (plan §6); at 0c the plan stops and the paired table, lost-pair list and borderline list go to the stakeholder, who may choose C1 (not promoting) or a new plan; no re-run, no floor tuning to rescue it. An amber condition does not exist: b >= 3 with net < 4 is green and is listed.

## 6. Risks

- Power is modest (§3): a green result is "not grossly worse", nothing stronger.
- The pairs are the same ones that informed the 0.43 floor derivation (pilot rows) and a saturated Set A informed the 4b choice; neither is held out.
- Same-day baseline at 0c is the 0.6b on `:8200`; at 6.3 it is the stored 0c record, so any corpus change between the two (a capture, a deletion) enters the paired table as noise; the report lists corpus Document/Chunk counts at both times.
