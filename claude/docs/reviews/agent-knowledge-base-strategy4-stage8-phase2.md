# K-030 Track 2 Stage 8 Phase 2 — AC-2 full-set regression gate review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (Stage 8 Phase 2)

## Scope & verdict

Diff-scoped review of `qa-engineer`'s Stage 8 Phase 2 deliverable (unit U6a of
`claude/docs/plans/agent-knowledge-base-strategy4-coordination.md`): `claude/docs/test-plans/
agent-knowledge-base-strategy-ac2.md` (execution plan), `claude/docs/test-reports/
agent-knowledge-base-strategy-ac2-report.md` (results/statistics/floor re-derivation, commit
`f8558f2`), and `skills/agent-kb-retrieval/SKILL.md`'s floor-section update 0.42→0.43 (commit
`658e29b`). Baseline: Stage 8 Phase 1's already-gated design (`claude/docs/plans/
agent-knowledge-base-strategy-ml.md`, "Stage 8 Phase 1") and its review (`claude/docs/reviews/
agent-knowledge-base-strategy4-stage8-phase1.md`) — not re-reviewed here, per brief. Per the
brief, my job was to **live-reproduce** the report's pivotal claims (not just trust its tables),
judge the floor re-derivation and crowding-out/prose-pattern methodology, and check internal
consistency — using `search_documents`/`get_document` against `ws:agent-team`, which `teco`'s own
verification pass did not have access to this round.

**Verdict: needs changes.** One blocker: the single most decision-critical number in the entire
report — R6/h40's second sibling's score, the number that triggered and calibrates the 0.42→0.43
floor re-derivation now shipped in `SKILL.md` — does not reproduce live, and the number I get in
its place is inconsistent with the report's own stated margin in a way that would flip the "0 of
41 wrongly dropped" claim the whole re-derivation rests on. One Major: a DEF-1 reproduction
citation (X1's "gets T1's document" claim) misidentifies the retrieved document. One further Major:
the crowding-out "isolated" judgment is scoped only to the 7 designed stratum-(e) families, but I
found same-document chunk-duplication live in a row outside that stratum (X1) that the report never
checked for it. Everything else spot-checked (3 of 5 pivotal documentIds via direct query
reproduction rather than just `get_document`, per the brief's ask; internal count consistency;
traceability; CPG call) holds up cleanly.

**CPG:** considered, not relevant — `ws:agent-team` is a FalkorDB content workspace of `Document`/
`Chunk` nodes reached through `search_documents`; this gate tests a retrieval contract against
distilled-knowledge prose, not source code, and no `cpg_claude` graph exists or would apply. I
confirm the report's own identical framing (line 29) independently rather than deferring to it.

## Findings

### Blocker — R6/h40's floor-triggering score (0.4201) does not reproduce live; the live value contradicts the report's own margin claim

The report's floor re-derivation (0.42→0.43) is triggered entirely by one number: R6/h40's second
sibling (`5b1b477ff67e4e3b81c57f14899bbe48`) reportedly scoring **0.4201** — "over the 0.42 floor
by 0.0001" (report, "Floor derivation"; identical claim now landed verbatim in `skills/
agent-kb-retrieval/SKILL.md:79`). I built the exact prefixed R6 query from `ml.md`:450 and called
`search_documents(query=..., limit=5)` against `ws:agent-team` twice (byte-identical both times,
so not within-session noise). Live result: top 5 are `af9ffb191c...` (0.2783), `b09fca8c9a...`
(0.4219), `5fabd6dce3...` (0.4285), `64eed8bab7...` (0.4285), **`5b1b477ff6...` at 0.4405** — not
0.4201, and at **rank 5**, not the "rank 3" the report's crowding-out section separately claims for
this same document ("the true second sibling landing at rank 3 by score"). Neither the score nor
the rank the report states for this one pivotal document reproduces; the two live values I measured
(0.4405, rank 5) are internally consistent with each other but not with either of the report's two
claims about the same document.

**Why this matters beyond one wrong number:** three other spot-checked rows reproduced almost to
the 4th decimal place — F4 (0.29177 live vs. 0.2918 reported), N4 (0.44595 live vs. 0.4459/0.446
reported), and R4/h14's whole family (0.39479/0.41112 live vs. 0.3948/0.4111 reported) — so this
isn't generic embedding jitter across the board; it's isolated to the one row carrying the entire
re-derivation's evidentiary weight. If my live value (0.4405) is the trustworthy one, it directly
contradicts the report's central claim "0 of the 41 pooled found true-positive documents are
wrongly dropped" at 0.43 — 0.4405 > 0.43, so this exact document would be wrongly rejected under
the floor the report just shipped. And even at the report's own stated 0.446 closest-negative
value, a true 0.4405 leaves only a 0.0055 margin — an order of magnitude thinner than the report's
claimed 0.0259 gap.

I cannot tell from here whether this is (a) a transcription/data-entry error in the report (the
other three rows argue for this — everything else I checked matched cleanly), or (b) genuine
run-to-run drift in the embedding backend specific to a query sitting this close to several
competing documents' scores (this lab has a documented history of LM Studio backend instability,
`agent-knowledge-base-strategy4-coordination.md` U1). Either explanation is a real problem: (a)
means the shipped 0.43 floor rests on a wrong number and needs re-derivation from a correct one;
(b) means a floor pinned to two decimal places isn't a stable signal at all for a borderline query,
which is a methodology question bigger than this one row.

**Suggested fix — owner `qa-engineer`, with a `data-scientist` consult if (b) is confirmed:**
re-run the R6 query multiple times (ideally at different times of day, to catch backend-state
drift) and record every result; if it stabilizes on a value materially different from 0.4201,
correct the report's table and `SKILL.md`'s derivation section with the true number and re-run the
floor arithmetic (a live 0.4405 would push the floor's safe range uncomfortably close to the 0.446
negative boundary — possibly requiring `data-scientist` to judge whether a fixed floor this thin is
even a sound mechanism, rather than qa-engineer re-deriving a new point value alone). If the score
instead proves unstable run-to-run even under controlled repetition, that instability — not a
specific replacement number — is the finding to land in `SKILL.md`, and the floor's margin
requirement should be widened to absorb it rather than pinned to a value measured once.

### Major — DEF-1's X1 reproduction cites the wrong retrieved document

The report's DEF-1 defect (report lines 268-271) states X1's miss "gets T1's document instead at
rank 1" — T1's expected `documentId` is `aa72815433924cfea2760062daa22b40` (ml.md:665). I ran X1's
exact query live: rank 1 is `31e22f72d811485eaad5a0aadd9c1d30` ("A fixture uniform on the rule's own
anchor dimension proves nothing about the rule as documented"), not T1's document, which does not
appear anywhere in my top 5 at all. I traced `31e22f72...`'s KB origin via
`claude/cobb/scripts/kb-claim-manifest.json` (`/files/claude/tdd-engineer/test-design-techniques.md/
headings[6]/claims`) — it **is** genuinely from T1's own KB file (`test-design-techniques.md`), just
a different claim within it than T1's specific expected row. So the report's broader claim
("cross-KB near-duplicate confusion... a different but topically adjacent claim... from a different
KB file") is directionally right — the rank-1 hit really is from the wrong file — but the specific
citation ("T1's document") names the wrong claim within that file, which would send a follow-up
implementer to the wrong `documentId` if they trusted the report's reproduction steps as written.

**Suggested fix — owner `qa-engineer`:** correct report lines 236-238 and 268-271 to cite
`31e22f72d811485eaad5a0aadd9c1d30` (title "A fixture uniform on the rule's own anchor dimension
proves nothing about the rule as documented") as the actual rank-1 confusion, not T1's own
`aa72815433924cfea2760062daa22b40` — both are `test-design-techniques.md` claims, so the
"different KB file" framing survives, only the specific document name needs fixing.

### Major — the crowding-out "isolated" judgment only checked the 7 designed stratum-(e) rows, but I found the same duplicate-chunk mechanism live in a row outside that stratum

The report's crowding-out conclusion ("6 of 7 designed families... show no crowding-out at all;
only 1 of 7 (R3/h39) does") is scoped entirely to the 7 rows tagged `e` in the golden-set design.
Running X1 live (not a stratum-(e) row) for the DEF-1 check above, I noticed its top 5 holds **two
different chunks of the same document** (`e4504f6ab0f74b45868efb97a4d2ef8b`, at rank 2 score 0.4081
and rank 3 score 0.4176, distinct `chunkId`s) — exactly the same-document-multiple-chunks-in-top-5
mechanism Phase 1's R3/h39 finding described (there, 3 of 5 slots; here, 2 of 5). It didn't cause
X1's own miss (the expected document scored worse than even these duplicate-chunk entries), so it
doesn't overturn any headline number — but it means the report's "1 of 7" isolated-incidence count
only measured the phenomenon where the design happened to look for it, not across the full 45-pair
(or even just the 29-row) result set. The true incidence rate among *all* returned top-5 sets is
uncounted, so "isolated" is evidenced only within a narrow, pre-selected slice, not the full
population the claim reads as covering.

**Suggested fix — owner `qa-engineer` (mechanical, re-uses data already captured in this run) or
`graph-dba`/`cobb` if a corpus-level check is preferred:** run a coverage probe over **every** row's
already-recorded top-5 documentId list (all 45 pooled, not just the 7 stratum-(e) rows) checking
for a repeated documentId within one query's top 5 — a one-pass grep/tally over the raw score
tables already in the report and `ml.md`'s pilot section, no new live queries needed except where a
row's individual documentIds weren't recorded. Report the true count of "queries whose top 5
contains a repeated document" as the actual isolated/systematic denominator, rather than "1 of 7
designed multi-facet families."

## What's solid

- **Score-floor derivation method itself (max surviving TP vs. min negative false match) is the
  right method**, same as Phase 1's already-approved approach — the finding above is about one
  input number being wrong/unstable, not the method.
- **Three of five requested pivotal documentIds reproduced almost exactly**: F4 (0.2918), N4
  (0.4459/0.446), and R4/h14's full family (0.3948/0.4111) all matched the report to the 4th decimal
  live, giving confidence the R6 discrepancy is isolated rather than systemic instability across the
  whole run.
- **P1 and G2 misses both confirmed live** — expected documents genuinely absent from the top 5 in
  both cases, not a data/documentId error (both `get_document`-confirmed real, `ready`, on-topic).
- **Internal count consistency holds throughout**: roster (29) = 24 single-answer + 3
  families(6 docs) + 2 negatives; pooled 45 = 16 + 29 in every subset table; pooled single-answer
  32 = 8+24, hits 28 = 7+21; pooled stratum-(e) 14 siblings/13 found/7 families all cross-check
  cleanly against Phase 1's own numbers. DEF-1's 4/4-misses-are-prose tag claim matches `ml.md`'s
  own tag column exactly for C1/P1/X1/G2.
- **Traceability section's cited paths are all real** and the coordination ledger/`ml.md` are
  genuinely untouched by this unit, as claimed.
- **DEF-1 routed forward rather than treated as a Stage 8 blocker is the right call** — each
  individual miss is one query among many with the document still reachable another way, and the
  aggregate pattern (real, but n=19/n=11) is exactly the kind of "flag, don't gate on" finding
  Recommendation 4's own acceptance bar anticipated; I don't think this needed to block Stage 8's
  acceptance even before the Blocker above, and still don't.
- **CPG call is correct** — no code-level component to graph here, independently confirmed.

## Open questions

- Is R6/h40's score genuinely unstable across separate `search_documents` invocations (a property
  of the embedding backend worth documenting), or was the report's number simply mistranscribed?
  I could not distinguish these from a single review pass — the three cleanly-reproducing spot
  checks argue mildly for transcription error, but I don't have the report's original tool-call
  transcript to compare against. Whoever re-runs R6 per the Blocker's suggested fix should record
  enough repetitions to settle this, since the two explanations call for different remedies (fix
  one number vs. widen the floor's safety margin as a matter of policy). **Resolved by Pass 2,
  below** — `qa-engineer` ruled out transcription error on its side (3/3 stable at 0.4201), and I
  independently re-confirmed my own side is equally stable (3/3 now at 0.4405), so this is genuine
  bimodal cross-session behavior, not drift or a typo. The remaining open question (what causes
  the bimodality, and whether a fixed-point floor is sound at all given it) is correctly escalated
  to `data-scientist`, not something either qa-engineer or I can resolve by re-running more calls.

## Pass 2 (2026-09-19)

**Re-check scope:** unit U6c, a focused re-check of `qa-engineer`'s in-place revision (commit
`25e26a3`) to `claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md` and
`skills/agent-kb-retrieval/SKILL.md` only, per this repo's repeated-gate convention — not a full
re-review of the whole Stage 8 Phase 2 deliverable.

**Verdict: approve.** All three Pass-1 findings addressed; no new findings.

- **Blocker (R6/h40 score instability):** addressed — escalated, not silently resolved, which I
  judge the right disposition. Re-ran R6's exact query once more live: **0.440489292144775 at rank
  5**, byte-identical to both of my Pass-1 runs (3/3 in my session now). This corroborates the
  report's bimodal-per-session characterization (qa-engineer: 3/3 at 0.4201/rank 3; me: 3/3 at
  0.4405/rank 5) rather than slow drift or ordinary jitter. The revision withdraws the false "0 of
  41 wrongly dropped" claim, narrows the floor's guarantee to explicitly exclude R6/h40's second
  sibling as a named residual risk, and routes the methodology question (is a fixed-point floor
  sound given confirmed ~0.02 cross-session instability this close to the negative boundary) to
  `data-scientist` rather than qa-engineer picking a new point value unilaterally. Since neither
  0.4201 nor 0.4405 is provably "the" correct value — both are rock-solid within their own
  sessions — re-deriving a fresh single number would just repeat the original overclaim with more
  confidence; escalating is not overcaution here. `SKILL.md`'s floor section matches the report's
  framing exactly.
- **Major (DEF-1/X1 wrong documentId):** fixed. Live rank-1 for X1 is
  `31e22f72d811485eaad5a0aadd9c1d30`, matching the correction in both report locations (the
  C1-pattern finding and DEF-1's reproduction steps) exactly.
- **Major (crowding-out scoped to 7 rows only):** fixed. Spot-checked one row from each phase, as
  requested. **R8** (Phase 1): live top 5 has `297b3ed4acf44f07bfed46f7de76e49d` (R8's own correct
  answer) at both rank 1 (0.1792) and rank 2 (0.2263) — confirms the report's "duplicate is the
  correct answer, harmless" pattern for this row. **T1** (Phase 2): live top 5 has
  `1ad762f7dd834a20aa924ede2be1c6a8` at both rank 2 (0.3693) and rank 3 (0.3909), while T1's own
  expected document (`aa72815433924cfea2760062daa22b40`) still lands cleanly at rank 1 (0.3463) —
  confirms the "duplicate of an unrelated document, doesn't displace the correct one" pattern. Both
  rows are correctly included in the report's 27/45 list, and the stated count arithmetic (12
  Phase 1 + 15 Phase 2 = 27) checks out against the named rows.

**`teco`'s bucket correction (R5/F5/F6/F9 moved from "(Phase 1)" to "(Phase 2)"):** confirmed
right. All four are design-only rows per `ml.md`'s own Pilot column (`—`, not `✓`), so they belong
in the Phase 2 list; the remaining 12-item Phase 1 list (R2, R3, R7, R8, R9, F1, F2, F3, C1, N1,
N3, N4) are all genuinely `✓`-piloted rows. No further bucketing errors found.

No new findings from this pass.
