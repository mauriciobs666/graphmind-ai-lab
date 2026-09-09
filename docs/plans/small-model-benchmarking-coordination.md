# Small-LLM benchmarking tool (`model-bench/`) — coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (M<n> TBD)

Coordinates delivery of [`small-model-benchmarking.md`](./small-model-benchmarking.md) (v1.1,
`architect`) against [`../requirements/small-model-benchmarking.md`](../requirements/small-model-benchmarking.md)
(Ready for design, `tico`), with statistics owned by
[`small-model-benchmarking-ml.md`](./small-model-benchmarking-ml.md) (v1.1, `data-scientist`).

## Scope of this coordination

Stakeholder decisions, 2026-09-02:

1. **Plan gate first**, with S0 dispatched in parallel (disjoint files). The plan had never been
   independently reviewed — no `docs/reviews/small-model-benchmarking.md` existed at kickoff.
2. **Drive through S3** (first real end-to-end run against a live model), then check back. S4–S8
   are out of scope for this pass and are not queued below.

## Environment notes at dispatch

- **CPG `cpg_falkorchat` is stale.** Built 2026-09-02T12:38:21Z at `4bb96e1` with
  `SOURCE_DIRTY = true`; three commits have landed on `falkor-chat/server` since (`b4cbdc7`,
  `5a5a257`, `673342b`) plus uncommitted working-tree changes. Structural answers from it must be
  confirmed against the files. No CPG exists for `model-bench/` (new component).
- **A separate coordination is open in this tree** (`salesperson-ui`), with uncommitted changes to
  `docs/plans/salesperson-ui*.md`, `docs/reviews/salesperson-ui*.md` and
  `falkor-chat/server/falkorchat/config.py`. No unit here may stage, commit, revert or otherwise
  touch those paths.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 — Plan gate: review plan v1.1 + `-ml` note against the requirements | `analyst` | `a0e3b74e34e1d4c40` | delivered | `docs/reviews/small-model-benchmarking.md` — **needs changes**: 3 blockers, 11 majors, 8 minors, 3 nits | — (is the gate) | 174k tok / 33 tools |
| U2 — S0 component skeleton | `coder` | `aa5b28bd14869593c` | accepted | `model-bench/**` (15 files), root `AGENTS.md` (+8 lines) | teco-verified — see note | 113k tok / 30 tools |
| U3a — Revise the method note; **resumed** for the `verdictMetrics` rename | `data-scientist` | `a394671cfc28bef87` | delivered | `docs/plans/small-model-benchmarking-ml.md` **v1.3** — §3.4 stats contract, then the rename | `analyst` re-gate (Pass 2) → — | 184k tok / 9 tools cumulative |
| U3b — Fold U1's findings + U2's S0 defects + the three decisions into the plan; **resumed** to reconcile vocabulary with the note | `architect` | `a3e258f27b83e764d` | delivered | `docs/plans/small-model-benchmarking.md` **v1.3** — 3 blockers + 13 majors + 8 minors + 4 nits dispositioned, then vocabulary reconciled | `analyst` re-gate (Pass 2) → — | 216k tok / 13 tools cumulative |
| U3c — FR-22a's illustrative clause cites the superseded 4×4 sampling | `tico` | `ae6ffeeb440ee967a` | accepted | `docs/requirements/small-model-benchmarking.md` (+13/−1) | folded into the Pass 2 re-gate | 88k tok / 12 tools |
| U3d — Pass 2 re-gate over plan v1.3 + note v1.3 + the amended requirements | `analyst` | `a0e3b74e34e1d4c40` (resumed) | delivered | `docs/reviews/small-model-benchmarking.md` `## Pass 2` — **approve with suggestions**; 3 blockers + 11 majors closed, 3 new findings | — (is the gate) | 250k tok / 29 tools cumulative |
| U3e — Close N-1, N-2, N-4 in the plan | `architect` | `a3e258f27b83e764d` (resumed) | accepted | `docs/plans/small-model-benchmarking.md` **v1.4** | teco-verified | 254k tok / 12 tools cumulative |
| U3f — Close N-3 in the note (`H` definition regression) | `data-scientist` | `a394671cfc28bef87` (resumed) | accepted | `docs/plans/small-model-benchmarking-ml.md` **v1.4** | teco-verified | 194k tok / 5 tools cumulative |
| U4 — S1 core (fingerprint, results, stats, report; no model calls) | `tdd-engineer` | `ac6ef3c82b078903a` | delivered | commit `ab91419` — 8 modules + 6 test files, **233 tests**, offline | U5a + U5b → — | 258k tok / 70 tools |
| U5a — Gate the S1 diff (engineering) | `analyst` | `aa9d6d24849f63006` | delivered | `docs/reviews/small-model-benchmarking-impl.md` — **needs changes**: 1 blocker, 6 majors, 7 minors, 4 nits | — (is the gate) | 213k tok / 47 tools |
| U5b — Methodology review of `stats.py` | `data-scientist` | `a7fbf4d59bfa1d0da` | delivered | `docs/reviews/small-model-benchmarking-ml.md` — **needs changes**: 1 blocker, 4 majors, 5 minors, 3 nits | — (is the gate) | 168k tok / 35 tools |
| U6a — Fix both gates' findings in the code | `tdd-engineer` (fresh) | `a79396bc49b0280d8` | delivered | 14 files, +1909/−168, **296 tests**; 34 mutations, 0 survivors | U7a + U7b → — | 322k tok / 74 tools |
| U7a — Re-gate the fix round (engineering, `## Pass 2`) | `analyst` | `aa9d6d24849f63006` (resumed) | delivered | `docs/reviews/small-model-benchmarking-impl.md` `## Pass 2` — **approve with suggestions**; 18/18 closed, 1 new major | — (is the gate) | 281k tok / 29 tools |
| U7b — Re-gate the fix round (statistics, `## Pass 2`) | `data-scientist` | `a7fbf4d59bfa1d0da` (resumed) | delivered | `## Pass 2` — **needs changes**: 1 blocker, 1 major, 1 minor (all new) | — (is the gate) | 237k tok / 13 tools |
| U8a — Note: the α ruling, Rule 7 split by path, the unified principle | `data-scientist` | `a394671cfc28bef87` (resumed) | accepted | `docs/plans/small-model-benchmarking-ml.md` **v1.6** | teco-verified | 253k tok / 14 tools |
| U8b — Plan sweep: `PackRef.contentHash` / Appendix A identity triple | `architect` | `a5ca583515c0979f1` (resumed) | accepted | `docs/plans/small-model-benchmarking.md` **v1.6** | teco-verified | 158k tok / 37 tools |
| U8c — Code: B-ML-2, M-ML-6, m-ML-6, P2-1…P2-5 | `tdd-engineer` (fresh) | `accddcf5d6ef280aa` | accepted | commit `95b4c88` — 10 files, **314 tests**; 22 mutations, 1 survivor (equivalent by construction) | U9a + U9b → — | 285k tok / 95 tools |
| U9a — Re-gate statistics (`## Pass 3`) + note v1.7 (3 routed defects) | `data-scientist` (fresh) | `ad7584d05e2136de4` | delivered | commit `89e11e1` — note **v1.7**; `## Pass 3` **needs changes**: 1 major (M-ML-7), 2 minors, 4 nits | — (is the gate) | 200k tok / 54 tools |
| U9b — Re-gate engineering (`## Pass 3`) | `analyst` (fresh) | `a066343742ae42c4e` | delivered | commit `d4d847b` — `## Pass 3` **needs changes**: 1 blocker (P3-1), 6 majors, 5 minors, 3 nits | — (is the gate) | 217k tok / 70 tools |
| U11 — Close both gates' Pass 3 findings (P3-1…P3-7, M-ML-7, m-ML-7, minors) | `tdd-engineer` (fresh) | `aeedc9f1724f1264c` | **killed by a platform 500 mid-run** — work preserved on disk (+428 lines, 6 files; 329 pass / 2 deliberate RED) | `model-bench/**`, uncommitted | — | — |
| U11b — Resume U11 from disk state; close both gates' Pass 3 findings | `tdd-engineer` (fresh, state-recovery brief) | `a8dd64de4bab140c4` | **accepted** — `d55f4d8` | `model-bench/**` (14 files, +1103/−62) | Pass 4 (U12a/U12b) → — | 250k tok / 137 tools |
| U12a — Engineering gate, Pass 4 | `analyst` (fresh) | `a693f15e5c2de88b2` | **accepted** — `e8bedce` | `docs/reviews/small-model-benchmarking-impl.md` `## Pass 4` | self → **approve with suggestions** (0 blockers, 5 majors) | 213k tok / 64 tools |
| U12b — Statistics gate, Pass 4 | `data-scientist` (fresh) | `a6cd549ed24ce965d` | delivered — **approve with suggestions** (1 major M-ML-8, 4 minors, 2 nits); note revised to **v1.8**; one sub-claim sent back for reproduction, uncommitted until answered | `docs/reviews/small-model-benchmarking-ml.md` `## Pass 4`; `docs/plans/small-model-benchmarking-ml.md` v1.8 | self → approve w/ suggestions | 229k tok / 71 tools |
| U13 — Plan v1.8: host-info source, first-call JIT budget, P4-4 stage re-attribution | `architect` | `aee9ac26e4c41f8ea` | **accepted** — `aebb611` | `docs/plans/small-model-benchmarking.md` v1.8 (+434/−62) | U15 → — | 173k tok / 80 tools |
| U15 — Gate plan v1.8 | `analyst` (fresh) | `a86335b5fb049e72c` | **accepted** — `ff499d5` | `docs/reviews/small-model-benchmarking.md` `## Pass 3` | self → **needs changes** (2 blockers, 6 majors) | 174k tok / 55 tools |
| U17 — Plan v1.9: close Pass 3's blockers and majors | `architect` (resumed) | `aee9ac26e4c41f8ea` | **accepted** — `81a3ef7` | `docs/plans/small-model-benchmarking.md` v1.9 | U19 → — | 301k tok / 100 tools |
| U19 — Re-gate plan v1.9 (Pass 4) | `analyst` (fresh) | `a84c263e5998ba953` | **accepted** — `bb0cacf`; recovered from disk after a kill, **verified 2026-09-06** (see the resume section) | `docs/reviews/small-model-benchmarking.md` `## Pass 4` | self → **needs changes** (3 blockers, 5 majors, 5 minors, 1 nit) | — (killed before reporting) |
| U20 — S1: residency element-shape assertion (plan v1.9 S1 DC-1) | `tdd-engineer` | — | queued — specified at v1.9, not yet implemented | `model-bench/**` | re-gate → — | — |
| U21 — Rule on Pass 4's three routed statistical questions (gap-detector right-censoring, threshold margin, §11.7's second denominator) | `data-scientist` (fresh) | `a4e06f8c810bbbbb8` | **delivered** — `fc2fcf6`; all three changed the note | `docs/plans/small-model-benchmarking-ml.md` **v1.12** | `analyst` re-gate → — | 119k tok / 39 tools |
| U22 — Plan v1.10: close all 14 plan-gate Pass 4 findings + fold notes v1.11/v1.12 | `architect` (fresh) | `aaf7ade9ddbc63e8b` | **delivered** — `3e5dc50` (+738/−134); all 14 closed, **no residuals** | `docs/plans/small-model-benchmarking.md` **v1.10** | `analyst` Pass 5 → — | 293k tok / 96 tools |
| U23 — Two items routed back from v1.10: rule (iv-b)'s p50-gate application, and §11.2's stale line number | `data-scientist` | `a4e06f8c810bbbbb8` (resumed) | **delivered** — `5197ce6`; (iv-b) confirmed **by correcting the note** | `docs/plans/small-model-benchmarking-ml.md` **v1.13** | `analyst` Pass 5 → — | 140k tok / 11 tools cumulative |
| U24 — Re-gate plan v1.10 + note v1.13 (Pass 5) | `analyst` (fresh) | `aa9af16f140993a17` | **accepted** — `b9964d1` | `docs/reviews/small-model-benchmarking.md` `## Pass 5` | self → **needs changes** (2 blockers, 4 majors, 4 minors) | 209k tok / 70 tools |
| U25 — Three rulings Pass 5 routed: co-presence shape, `censoringExact` clause 1, `paired_cluster_bootstrap`'s necessity | `data-scientist` | `a4e06f8c810bbbbb8` (resumed) | **delivered** — `ca69cb1`; all three changed the note, **plus a live defect the gate missed** | `docs/plans/small-model-benchmarking-ml.md` **v1.14** | `analyst` Pass 6 → — | 182k tok / 15 tools cumulative |
| U26 — Plan v1.11: close all 10 Pass 5 findings; rule 5 restated honestly | `architect` (fresh) | `a99cd8cce60c76d82` | **delivered** — `85a32e5` (+540/−110); all 10 closed, nothing carried | `docs/plans/small-model-benchmarking.md` **v1.11** | `analyst` Pass 6 → — | 282k tok / 121 tools |
| U27 — Re-gate plan v1.11 + note v1.14 (Pass 6) | `analyst` (fresh) | `a6f3786437e4dab05` | **accepted** — `afca8e0` | `docs/reviews/small-model-benchmarking.md` `## Pass 6` | self → **needs changes** (1 blocker, 2 majors, 2 minors); **S2 may not be dispatched** | 161k tok / 52 tools |
| U28 — P6-1's method half: the continuous instrument's carrier + §3.2e's verdict strings | `data-scientist` | `a4e06f8c810bbbbb8` (resumed) | **delivered** — `0ad0e7a`; both changed the note, **plus one reversal and one unprompted ruling** | `docs/plans/small-model-benchmarking-ml.md` **v1.15** | `analyst` Pass 7 → — | 218k tok / 17 tools cumulative |
| U29 — Plan v1.12: close all 5 Pass 6 findings; the continuous carrier as an S1 edit | `architect` (fresh) | `a74d8052842395e20` | **delivered** — `5b67416` (+521/−62); all 5 closed, **12/12 residuals re-run** | `docs/plans/small-model-benchmarking.md` **v1.12** | `analyst` Pass 7 → — | 283k tok / 91 tools |
| U30 — Four items v1.12 raised: §3.2f's retired wording, the continuous-verdict producer's signature, the homogeneous-family enforcement point, §5.2's `sep_raw` figures | `data-scientist` | `a4e06f8c810bbbbb8` (resumed) | **delivered** — `e290148`; all four changed the note | `docs/plans/small-model-benchmarking-ml.md` **v1.16** | `analyst` Pass 7 → — | 248k tok / 15 tools cumulative |
| U32 — Plan v1.13: absorb note v1.16's four deltas. **Deliberately small** | `architect` (fresh) | `ac827e78b5339f829` | **delivered** — `fbe5741` (+300/−91); stayed small, 5 extras all reported | `docs/plans/small-model-benchmarking.md` **v1.13** | `analyst` Pass 7 → — | 220k tok / 99 tools |
| U33 — Does Rule 8 take a `support` parameter? (Table E's clamp has no route to its only caller) | `data-scientist` | `a4e06f8c810bbbbb8` (resumed) | **delivered** — `1fbdb6f`; recommendation accepted with a **sharper shape**, and the premise replaced | `docs/plans/small-model-benchmarking-ml.md` **v1.17** | `analyst` Pass 7 → — | 267k tok / 5 tools cumulative |
| U31 — Re-gate plan v1.13 + note v1.17 (Pass 7) | `analyst` (fresh) | `a7144f0028209bdc0` (first instance `aaa942cd75fabc2ca` **killed by a session rate limit**, wrote nothing) | **accepted** — `b6222c6` | `docs/reviews/small-model-benchmarking.md` `## Pass 7` | self → **needs changes** (1 blocker, 0 majors, 2 minors); **plan implementable**, S2 one revision away | 190k tok / 51 tools |
| U34 — Plan v1.14: P7-1/2/3 + note v1.17's two deltas. **Intended as the last plan revision** | `architect` (fresh) | `a4a33cf66d7db77f2` | **delivered** — `d6556c3` (+288/−75, *smaller* than v1.13); one extra disclosed (Table E's second residual = P7-2 applied), accepted | `docs/plans/small-model-benchmarking.md` **v1.14** | **narrow re-check** (U35) → — | 225k tok / 95 tools |
| U35 — Narrow re-check of v1.14's delta only (**not** a full Pass 8), per Pass 7's own recommendation | `analyst` (**resumed**, Pass 7's own instance) | `a7144f0028209bdc0` | **accepted** — `4cd22b9`; v1.14's own five items all clean, **the blocker predates v1.14** and came from the table sweep I asked for | `docs/reviews/small-model-benchmarking.md` `## Pass 8 (narrow)` | self → **needs changes** (1 blocker, 2 majors, 0 minors) | 282k tok / 26 tools |
| U36 — P8-1: how is a family-corrected quantile level `α/(2k)` represented exactly? (`permille: int` cannot express it; non-exact at k=3 in any decimal unit) | `data-scientist` (fresh) | `a1aaf4909260e0e14` | **delivered** — `bbbf18e`; exemption refused, **my reasoning overruled, conclusion upheld**; levels become exact `Fraction`s. Caught the note's own §11.2.1 defect (1626 measured against a spelling the code does not use; the code's form diverges **0**) | `docs/plans/small-model-benchmarking-ml.md` **v1.18** | `analyst` re-check → — | 199k tok / 41 tools |
| U37 — Plan v1.15: close P8-1/P8-2/P8-3, re-pair to note v1.18, sweep **all seven** tables against rule 5(b) | `architect` (fresh) | `a522a1704449cdc2e` | **delivered** — `083d174` (+329/−63); found **beyond brief** that the C/G *order* is unsafe (G-first hands the shipped `pct: float` a `Fraction`; `Fraction(1,40)` is `0.025`, not `2.5`). Sweep: A passes via rule 5(b)'s named alternative, B/D/E/F pass, C and G were the two broken. 3 extras disclosed | `docs/plans/small-model-benchmarking.md` **v1.15** | `analyst` Pass 9 (narrow) → — | 264k tok / 94 tools |
| U38 — Narrow re-check of v1.15's delta (Pass 9). Sharpest question: Table G's residuals have an **intermediate state, not a commit**, as their baseline | `analyst` (**fresh** — Pass 8's instance is at 282k tok *and* proposed the exemption the note overruled) | `a32855305cbdf4b2c` | **accepted** — `012cf5d` | `docs/reviews/small-model-benchmarking.md` `## Pass 9 (narrow)` | self → **approve** — 0/0/0/0, the first clean gate in nine rounds; **plan ready for implementation** | 171k tok / 40 tools |
| **— PLAN CLOSED. Implementation begins. Baseline: 389 tests green at `5878014`, verified by me before dispatch. —** | | | | | | |
| U39 — **S1 impl 1 of 3**: §4 S1e Tables **A + B** (the `fingerprint.py` re-key) | `coder` (fresh) | `a8b66dddd2e3a45c4` | **delivered** — `8fc2341`; 6/6 residuals at 0, suite 389 → **472**, ruff clean, scope held. 11 mutations killed; I reproduced the load-bearing one (201 failures). Reports **a plan defect only execution could find** (DC-1 vs Table A's 2nd residual) + 5 interpretations | `model-bench/` source + tests | `analyst` diff-scoped (U40) → — | 225k tok / 64 tools |
| U40 — Diff-scoped review of `8fc2341` + adjudicate the implementer's 6 reported items | `analyst` (fresh) | `ad216ed80e4e38da2` (**killed by a session rate limit**, wrote nothing, left no mutation in the tree — verified on disk; **resumed** 19:22 after the limit reset) | **delivered** — `cbfcda9`, **needs changes**: 7 findings, of which P5-1/2/4 code-side, P5-3/5/6 plan-side, P5-7 a nit | `docs/reviews/small-model-benchmarking-impl.md` `## Pass 5` | self → **needs changes** | — |
| U41 — Fix round on `8fc2341`: **P5-1** (callSurface three-state), **P5-2** (pin the `to_dict` decision), **P5-4** (a test that misses what it is named for) | `coder` (fresh) | `a18e03b6ec3876e7c` | **delivered** — `c523a35` (+207/−26). Closed all four; **overruled the gate's own recommended fix** (a per-arm `""`/`None` sentinel, not the review's `_ABSENT`, which would add a third inhabitant to a plan-typed field and break the deterministic round trip) and refuted two of Pass 5's mechanism claims. 472 → **475 tests**, ruff clean, **7 mutations, 7 killed** — two of them reproduced independently by me before acceptance | `fingerprint.py` + `test_fingerprint.py` + `HISTORY.md` + `AGENTS.md` | U45 → — | 156k tok / 55 tools |
| U42 — Plan **v1.16**: **P5-3** (the rule-5(b) trap in the plan that wrote rule 5(b)), **P5-5** (Appendix A stale), **P5-6** (Table C's site row stale — blocks the next unit) | `architect` (fresh) | `a6e6e78f99baf4ead` | **delivered** — `1ed8599` (+215/−25). Closed P5-3 on **both** sides (the minimal reword leaves the counter-impl alive); rule 5(b) gains the **disowning mention** trap + a standing sweep over all 18 residuals, which **immediately found a third instance in Table C — the next unit's own table**. `Landed:` convention proposed, **accepted** | `docs/plans/small-model-benchmarking.md` **v1.16** | `analyst` re-check → — | 176k tok / 78 tools |
| U45 — Pass 6 re-check of `c523a35`, and four adjudications: the substituted P5-1 fix, the coder's two refutations of Pass 5, my own candidate finding (the model-arm round trip `None → ""`, which may make the new `null` reason unreachable through our own writer), and **P5-7's missing owner** | `analyst` (**resumed** `ad216ed80e4e38da2`, the Pass 5 author) | `ad216ed80e4e38da2` | **delivered** — `51bf15c`, **needs changes**: 0 blockers, 0 majors, **1 minor**, 0 nits. All four adjudications went the coder's way: P5-1/2/3/4/5 **fixed**, P5-7 **withdrawn**, and the gate recorded **two of its own Pass 5 claims as errors** — it ran a mutation applying its own recommended `_ABSENT` design and the suite refused it (2 failed). **P6-1** is my candidate finding, confirmed real | `docs/reviews/small-model-benchmarking-impl.md` `## Pass 6` | self → **needs changes** | 197k tok / 16 tools |
| U46 — **P6-1**: `to_dict` omits `callSurface` on the *value*, not the arm, so a model record diagnosed `null` serialises to one diagnosed `absent`. Unreachable through today's writers, reachable through `model-bench migrate` (§3.4.3) — **the sixth instance of "correct for its current caller, wrong for a caller the plan commits to adding"** | `coder` (**resumed** `a18e03b6ec3876e7c`) | `a18e03b6ec3876e7c` | **delivered** — `f409905`. **Overruled the gate a third time, and again correctly**: the gate's one-liner (omit on the arm) fixes one shape and breaks two, dropping a reference arm's *forbidden* surface so the record reads back **valid**. Shipped the conjunction instead, so omission means one record and not a class of them. Round trip now **total over all nine shapes** — I enumerated them myself by executing the class, and reproduced the mutation applying the gate's own suggestion (1 failed). 475 → **477 tests**, ruff clean | `fingerprint.py` + `test_fingerprint.py` + `HISTORY.md` | U47 → — | 176k tok / 10 tools |
| U47 — Narrow Pass 7 re-check of `f409905`: is the conjunction the right rule or two special cases in one condition; does the new test pin **both** narrowings or only the two tried; and a standing note on the round's own pattern — every gate **finding** held, every gate **suggested fix** did not survive the type the plan closes | `analyst` (**resumed** `ad216ed80e4e38da2`) | `ad216ed80e4e38da2` | **delivered** — `9ea8753`, **needs changes**: 1 minor (**P7-1**, in the test, not the code). Upheld the conjunction as the serialisation image of §3.4.1's `iff`. Found the new round-trip test pins **the two narrowings the coder tried, not the rule** — two others pass 477 green, both reproduced by me. Wrote the round's pattern into §3 and sharpened its own rule: *a suggested fix is not evidence unless you can name the assertion that would catch it being wrong* | `docs/reviews/small-model-benchmarking-impl.md` `## Pass 7` | self → **needs changes** | 214k tok / 7 tools |
| U48 — **P7-1**: parametrize the round-trip test over the **product** of arm kinds × surface values so a wrong `omit` cannot pass; plus two doc corrections (the overstated `migrate` claim, which is the gate's own Pass 6 phrasing inherited into code + `HISTORY.md`, and an eight-vs-nine shape miscount) | `coder` (**resumed** `a18e03b6ec3876e7c`) | `a18e03b6ec3876e7c` | **delivered** — `c19f875`. Grid over the **product** (12 shapes) replaces two hand-picked cells; 6 wrong `omit` conditions killed, including both Pass 7 survivors. Also corrected an overstatement the **gate** authored in Pass 6 and the code inherited (`migrate` does not by definition walk invalid records). 477 → **487 tests** | `test_fingerprint.py` + `to_dict` comment + `HISTORY.md` | **gate skipped — stated exception** → — | 192k tok / 8 tools |
| U43 — **S1 impl 2 of 3**: Tables **C + D + E + G** (the `stats.py` signature round, C→G mandated) | `coder` (**fresh** — the fix-round coder is at 192k tok and this unit needs none of its `fingerprint.py` reasoning) | `a873908e4e6bd1d30` (**killed by a session rate limit** — the *fourth* on this coordination, and the first to leave substantial work on disk; **resumed** 2026-09-08 with a state-recovery brief) | **in-flight** — dispatched at baseline `c19f875`, after md5-verifying `stats.py`/`results.py`/`report.py` byte-identical to `5878014` so every pinned count still held. On disk at the kill: **+434/−41** over 4 files, suite **511 green**, **no mutated file left behind** (checked before believing anything else), Table C's residuals already at target (0 / 0 / 1). **delivered** — `cc28d48` (+885/−238 over 9 files). All four tables complete; 487 → **519 tests**, ruff clean, **13 mutations, 11 killed** with both survivors reported rather than hidden. Three findings against the **plan**, routed to `architect` as U49: **F1** Table E's residuals go blind on a faithful edit (I reproduced the half-application — both residuals read **0** while one test fails), **F2** Table C's third residual moves 1→2 once Table D lands, **F3** Table D's first residual can never reach 0 (substring collision, present at `5878014`). Two judgment calls handed to the gate: `Verdict.bound_by` and the `LatencyBlock` deferral | `model-bench/` source + tests | **U50** → — | 290k tok / 44 tools |
| U49 — Plan **v1.17**: F1 (generalise §7 rule 5(b) to a residual over *text the edit destroys*, then **sweep all 18 residuals for the new shape**), F2, F3 | `architect` (**resumed** `a6e6e78f99baf4ead`) | `a6e6e78f99baf4ead` | **delivered** — `b6f578c` (+272/−49). Rule 5(b) gains a **third form** (a residual over the text the edit *creates*), with its trigger named — **a parameterising edit** — and Table G kept as the contrast proving the trigger is real. Sweep over all 18: **16 robust / 2 fragile (E's pair, replaced) / 1 near-miss cleared**. Measured and **rejected** the cheaper truncated form (reads 4 on the faithful tree). F3's re-derivation found the collision is **two** lines, the second a genuine site — the whole-identifier form's cost is stated, not hidden. I verified 1/1 faithful vs 1/0 half-applied, and 27/29 at `5878014`. **v1.18** (`9bc3b35`) adds the portability clause to §7 rule 5. **It was right and I was wrong on `--ignore-files`** — it held a measurement against my correction and I reproduced its result | `docs/plans/small-model-benchmarking.md` **v1.17** | **accepted on my own re-verification** → — | 266k tok / 58 tools |
| U50 — Diff-scoped gate of `cc28d48`, plus four adjudications: the two stop-and-ask forks the coder decided instead of raising (`Verdict.bound_by`, the `LatencyBlock` deferral), **six deleted tests** (the diff's highest-risk part), and the two self-reported surviving mutants | `analyst` (**fresh** — the impl reviewer is at 214k tok after 7 passes and this is the stage's largest diff) | `a4da085f65d70372d` | **delivered** — `30f4651`, **needs changes**: 0 blockers, **3 majors**, 3 minors, 1 nit. Both stop-and-ask adjudications go the implementer's way. **P8-2** and **P8-3** reproduced by me (a garbage constant still passes 519; `deff=0.5` returns a *narrower* interval instead of raising). **P8-1** is the **seventh** instance of correct-for-its-caller / wrong-for-the-committed-caller. Corrected the implementer's D1 equivalent-mutant claim. **No fourth residual defect** | `docs/reviews/small-model-benchmarking-impl.md` `## Pass 8` | self → **needs changes** | 175k tok / 68 tools |
| U51 — **P8-1's root**, a methodology question the gate deliberately declined to adjudicate: does the `(-1, 1)` support clamp belong on the envelope's **arms** or on the **composed** interval? Clamping the arms makes them tie at `-1.0`, so the audit bullet names the arm that did not bind; clamping the composed interval would *dissolve* the tie rather than patch it. Must hold for `sep_z` too, whose differences are not bounded by 1 | `data-scientist` (fresh) | `ae3ffeedb14d9d0fa` | **delivered** — `a707d09`. Ruling: **composed**, and stated once as a property of the estimand so neither carrier owns a special case (`envelope_arms` returns **unclamped** arms with `clamp=None`; the composer clamps its result; §3.2d's continuous path already did exactly this and does not change). **The honest headline it volunteered: arms-vs-composed is immaterial to the statistics** — clamping and composing commute exactly, so every printed number and every verdict is bit-identical either way. **This is a reporting fix, not a statistical one.** But the defect is *larger* than P8-1 measured and **neither of P8-1's two suggested fixes closes it**: on the separating case `(0,0,38,2)`@1.5, attributing from the unclamped arms still prints `lower bound: exact paired bootstrap`, which is false. `bound_by` gains a third token, **`support bound`**, on a *strict* comparison; the tie-break survives for the genuine structural ties and is finally pinned. **Closes P8-5 as collateral** (compose-and-clamp gets one home). Ten assertions with computed witnesses. A second, unprompted sweep proved `_widen`'s latent inversion hazard cannot fire (138,648 tables) and the note records the premise rather than assuming it | `docs/plans/small-model-benchmarking-ml.md` **v1.19** | **folded into the plan gate on the revision** (the methodology authority *is* the producer; `analyst` declined this question by routing it here) — and **I re-verified every number myself**: all ten witness values to the digit, commutation over **74,046** combinations at n=40 (0 diffs, 0 point violations, 0 zero-exclusion changes, 0 conservatism violations), and the full audit table 0/38/58/132/570/2038 with wrong-arm subset 0/0/6/12/166/954 and distinguishable 58-of-58 / 132-of-132 / 540-of-570. I derived the wrong-arm definition **before** reading the note's and it reproduced exactly | 133k tok / 40 tools |
| U52 — Pass 8 fix round: **P8-2** (restore the lost assertion), **P8-3** (restore the `design_effect < 1.0` refusal), P8-4, P8-5, P8-6, P8-7. **P8-1 explicitly excluded** pending U51's ruling | `coder` (**fresh** — U43's coder is at 290k tok, past the resume threshold) | `ab774117c13e7f953` | **delivered** — `7f865e2` (+431/−43 over 5 files). All six closed, **519 → 550 tests**, ruff clean, no skips. **14 mutations, 13 killed and the 14th a deliberate control**: M14 is P8-1's own tie-break mutant, left surviving as the evidence that clamp placement was untouched. **Two deliberate deviations from the review, both of which I verified and both of which stand**: P8-3 ships `not design_effect >= 1.0` because `< 1.0` is False for a NaN — at `cc28d48`, `envelope_arms(deff=nan)` returned `((-1.0,1.0),(-1.0,1.0))`, a maximally wide interval conjured from a missing number, which I reproduced; and **P8-7 corrects the gate outright** — citing F2 as open would now be *false*, plan v1.17 having closed it, so it wrote the live constraint (don't rename `exact_paired_quantiles`) instead. **P8-5 lands as `_compose()` with its three deltas from Rule 4a documented in its own docstring at the seam**, not in a handoff note | `model-bench/` source + tests + `AGENTS.md` + `HISTORY.md` | **U53** → — | 159k tok / 97 tools |
| U53 — Gate the Pass 8 fix round at `7f865e2` (`## Pass 9`), and adjudicate **two deliberate deviations from its own suggested fixes** — P8-3's NaN-safe spelling and P8-7's outright correction of the review. Highest-risk part named for it: the **six retired test ids** | `analyst` (**resumed** `a4da085f65d70372d` — 175k tok, under the threshold, and it wrote Pass 8) | `a4da085f65d70372d` | **delivered** — `3af905f`, `## Pass 9`, **needs changes**: 0 blockers, **2 majors** (N1, N2), 1 nit (N3). *Killed by the sixth rate-limit event; its skeleton-to-disk habit cost it one section.* **Both deviations upheld, and P8-3's reverses the question I put to it**: `resolving_power` already carried `not design_effect >= 1.0` with the identical `# NaN-safe` comment **at `cc28d48`**, before the round — verified — so U52 moved one more site onto the module's own spelling and **the analyst's own suggested `< 1.0` would have been the divergent one**. That is the **seventh** suggested fix of this coordination overruled on the merits, and the first the analyst overruled *itself* on. The six retired ids cost nothing: resolved at **id level** via `pytest --collect-only` against both trees (`git archive` of `cc28d48` into scratch) — **6 retired, 37 added**, every retired assertion surviving strictly stronger. **`_compose` holds as Rule 4a's seam** and two of its three deltas were *forced*, not chosen — clamping inside it is only correct if `envelope_arms` stops clamping in the same edit, which **is** P8-1 — with one obligation named that Table H already carries as a site row | `docs/reviews/small-model-benchmarking-impl.md` `## Pass 9` | self → **needs changes** — and **I reproduced all three findings before dispatching the fix**: four `design_effect` guard sites in `stats.py`, two NaN-safe (`:376`, `:855`) and two not (`:228`, `:1126`); `paired_cluster_bootstrap(…, design_effect=nan)` returns **`(-1.0, 1.0)`** live; and N1 is sharper than reported — the comment **directly above** `:1126` exists so `verdict()` will not raise the same sentence one layer down (P3-11), which at a NaN is exactly what it does. N3 is real at `AGENTS.md:83` | 238k tok / 21 tools |
| U54 — The clamp pass: fold Rule 4a into the plan as **P8-1 + P8-5's implementation spec**, each shipped-code edit carrying its own residual; plus the §7 rule 5 line-pin clause. **The handoff's premise was wrong and I corrected it in the brief** — Table E's pair pins `_widen`'s *body* at `stats.py:261`, which Rule 4a does not touch (it changes `envelope_arms`' call sites at `:382`/`:387`), so both still read **1/1** at `7f865e2` and the four-edit coupling **shrinks** rather than grows | `architect` (**resumed** `a6e6e78f99baf4ead` at 290k tok — *over* my own threshold, taken deliberately: this pass is about not getting residual arithmetic wrong, which is exactly where its accumulated context earns the cost; briefed to re-check premises harder, not less) | `a6e6e78f99baf4ead` | **delivered** — `b873cc9`, plan **v1.19** (+274/−31, one file). *Killed by the seventh rate-limit event and resumed; it had lost every measurement it held.* **Table H** is the eighth S1e table and the implementation spec for a ruling this plan did not make: two enumerating commands, nine site rows, **six** residuals — one per edit, as scoped. **I re-measured all of it on the live tree and every figure reproduced**: both enumerating commands with their per-file breakdowns (`bound_by` **14** = stats 5 / test_stats 6 / report 2 / test_report 1; `envelope_arms` **20** = stats 5 / test_stats 15), all six residuals **2/0/0/1/1/2** with every stated site line resolving, both new names free (**0/0**), Table E's pair **1/1**, `_widen` byte-identical to `cc28d48`, `model-bench/` byte-identical `7f865e2`→HEAD, and DC-12's partition summing to **24** (A 2, B 4, C 3, D 2, E 2, F 3, G 2, H 6) derived rather than transcribed. **Two design catches a reviewer would otherwise have made**: residual 1 is scoped to the `clamp=` keyword because the bare tuple matches `_widen`'s own docstring *refusing* that default (`stats.py:253`, verified) — v1.16's disowning-mention trap, caught before it was written; and residuals 2/3 are **third-form**, the first *forward* application of §7 rule 5(b)'s parameterising trigger rather than a correction into it. **One §7 rule 3 raise opened and verified real**: Rule 4a cites *plan-gate* P8-1 where it means *impl-gate* P8-1 — two genuinely distinct findings (plan-gate P8-1 is Tables C/G colliding on `stats.py:159`), a citation defect that blocks nothing and is the `data-scientist`'s to fix | `docs/plans/small-model-benchmarking.md` **v1.19** | `analyst` → **held for U53** — Pass 9 gates `_compose` *as shipped*, which is one of Table H's own site rows, so gating the plan against a premise Pass 9 may move would be wasted work | 369k tok / 32 tools |
| U55 — Pass 9 fix round: **N1** (`verdict():1126`), **N2** (`paired_cluster_bootstrap:228`), **N3** (`AGENTS.md:83`'s NaN-unsafe prose for a NaN-safe guard). One respelling at the two sites P8-3's correct diagnosis did not reach, plus three words | `coder` (**resumed** `ab774117c13e7f953` — 159k tok, under the threshold, and it made the P8-3 fix whose predicate this extends; told up front that both its deviations were upheld) | `ab774117c13e7f953` | **delivered** — `93b0e42`. **550 → 560 tests**, ruff clean, no skips, no test retired; the +10 is two single-value tests becoming six-id sweeps. Both majors done **test-first** — symptom reproduced, test written, *confirmed failing* before `stats.py` was touched. Five mutations, four killed and the fifth surviving by design. **The line-pin question I made non-optional came back clean and it saves a unit**: `stats.py` is **1418 lines before and after** at `2/2`, and the coder extracted all ten lines pinned by Tables E, G and H *before* editing and diffed them after — I re-checked each of the ten individually and all are **byte-identical**, so **no re-pinning unit is needed**. That constrained the edit shape (end-of-line comments, not blocks) and the coder said so rather than letting the constraint pass silently. **N3 went wider than the review asked, with a reason I accept**: three words remove a false sentence but would not stop the next editor undoing this round, so the bullet now names all four sites, the one-clause mechanism and *do not simplify* — `AGENTS.md` at 2,059 words, no line over 700 | `model-bench/` source + tests + `AGENTS.md` + `HISTORY.md` | **U58** → — and **I reproduced two of the five mutations myself by copy-restore**: N1-M1 is killed by the **`[nan]` row alone** (the evidence that the *sweep*, not a second example, was the necessary shape), and **N-M5 — P8-1's tie-break control — survives at 560**, so clamp placement is still genuinely untouched and P8-1 is still open for Rule 4a. All four guards now NaN-safe and `grep -rn 'design_effect < 1.0' modelbench/` is **empty**: the class is closed by construction, not by inspection | 190k tok / 26 tools |
| U56 — Plan gate **`## Pass 10`** on v1.19's **Table H** and its ripples (DC-12's twenty-four, §4 S1e's eight, §7 rule 5(b)'s new clause, Table E's *does not move* block, the §7 rule 3 raise). **Briefed off the arithmetic** — I re-measured all of it myself — and onto whether the spec is right and complete: does the residual set discriminate a **half-application**, does the nine-row site list miss a consumer of the widened return type, does the `report.py:338` row cover the `bound_by is None` path P8-2's gap was in | `analyst` (**resumed** `a32855305cbdf4b2c` — 171k tok; the plan-gate instance, which wrote Pass 9 (narrow) and the approve, not the impl-gate one) | `a32855305cbdf4b2c` | **delivered** — `913e159`, `## Pass 10`, **needs changes**: **1 blocker** (P10-1), 2 majors, 1 minor, 1 nit. **None blocked on unbuilt work, none deferred by choice.** Ran parallel to U55 on a moving tree and honoured the brief exactly — every measurement through `git show 7f865e2:`, no pin staleness raised. **Answering the brief's four questions it confirmed three and broke the fourth open in a place I did not suspect**: every consumer of the widened return type *is* enumerated (`_compose` has exactly two call sites, `:458` and `:1183`, both rowed — I verified), the `bound_by is None` path is invariant and correctly absent, and residual 1 discriminates with the third-form trigger correctly identified. **The miss is on the test side**, which no residual can see because all six are `modelbench`-scoped | `docs/reviews/small-model-benchmarking.md` `## Pass 10` | self → **needs changes** — and **I verified the blocker's mechanism by reading the test rather than trusting the sweep**: `test_neither_printed_bound_is_ever_tighter_than_either_arm` compares the printed envelope against `envelope_arms`' return *directly*, so unclamped arms plus a clamped composition makes `lo <= mover[0]` read `-1.0 <= -1.5` — false. Also confirmed `support bound (-1)` at **0** occurrences in the plan against a verbatim pin at `-ml:1465`, and the `lo`/`hi` vs `u_lo`/`u_hi` divergence | 267k tok / 34 tools |
| U57 — Pass 10 fix round on Table H → plan **v1.20**. **P10-1** (add the site row for the test the edit falsifies, pinned by *test name* — `tests/test_stats.py` is moving under U55, which is v1.19's own exact-text-over-line-pin clause applied to the table that introduced it), **P10-2** (the renderer needs `SUPPORT_DIFF_PROPORTIONS`, a `stats.py`→`report.py` crossing the row omits), **P10-3**, P10-4, P10-5 | `architect` (**resumed** `a6e6e78f99baf4ead` at **369k tok** — *well* past my threshold, and recorded here as a judgment rather than an oversight: these five findings are omissions of **prose**, not errors of arithmetic, and re-deriving Table H cold risks introducing the second kind to fix the first. Briefed to keep the reading narrow, told explicitly which of its work the gate confirmed so it re-defends nothing, and given a **stop signal** — if it needs to re-read large sections, say so and I dispatch fresh) | `a6e6e78f99baf4ead` | **delivered** — `9aafc85`, plan **v1.20** (+80/−14). All five closed, none carried. **The context gamble paid**: it reported *"I did not need to re-read large sections — Pass 10's findings section plus five anchored rows was the whole read"*, and 18 tool uses against 32 last round. **P10-3 took both available repairs rather than one**: the row now writes the post-edit body out in Rule 4a's own `u_lo`/`u_hi` notation **and** the residuals are restated over `SUPPORT_DIFF_PROPORTIONS[0]`/`[1]`, which depend on no introduced name at all — so an implementer who varies the locals despite the prescription still scores. Generalised at **§7 rule 5(b): a third-form residual is only a residual if the table writes out the text it pins, in the row, because a command is not a specification.** Table H is now **ten** site rows; residuals stay six, DC-12 twenty-four | `docs/plans/small-model-benchmarking.md` **v1.20** | `analyst` → **held** — re-gate after U59's re-baseline rather than gating a document with three facts I already know are stale | 401k tok / 18 tools |
| U58 — Narrow closure gate on `93b0e42`. **Briefed as a closure pass, not a re-review**, and pointed off everything I verified (four guards, 560 green, ten pins, two mutations) and onto what I cannot check: do the six-id sweeps assert the layer-ordering **property** or six examples of it; did a plan pin-constraint bend the code; does N3's second widening hold; and **is there a fourth path where a NaN or an inf reaches the same clamp and prints as a real interval** | `analyst` (**resumed** `a4da085f65d70372d` — 238k tok, just under the threshold, and it wrote N1/N2/N3) | `a4da085f65d70372d` | **delivered** — impl-gate `## Pass 10`, **needs changes**: 0 blockers, **1 major (N4)**, 1 minor (N5); all three Pass 9 findings closed. **The fourth-path question was the right one to ask and the answer is yes.** **N4:** `not inf >= 1.0` is `False`, so `+inf` clears all four guards; then `sqrt(inf)` → `_widen` → `(-inf, +inf)` → clamp → `(-1.0, 1.0)`, and `verdict()` attributes it to a named instrument. **The clamp is the launderer, not the predicate** — reproduced by me in full: `envelope_arms(deff=inf)` → `((-1.0,1.0),(-1.0,1.0))`, and one `nan` in `diffs` gives `paired_cluster_bootstrap` → **`(-1.0, 1.0)` with the clamp and `(nan, nan)` without it**. That two-line contrast is the whole proof. **It ran its candidate instead of proposing it** (one `math.isfinite` on `_widen`'s result: 560 green, refuses all three shapes, survives Rule 4a) and still left result-guard-vs-input-guard to the implementer, naming its own limit. **My three other questions: six examples not the property** (a real partition of the *rejection* domain with no gap — but both tests look at one side only, **which is exactly where N4 was hiding**); **readability was paid for, nothing bent**; **N3's widening holds and its own three words would not have** | `docs/reviews/small-model-benchmarking-impl.md` `## Pass 10` | self → **needs changes** | 270k tok / 14 tools |
| U59 — Re-baseline Table H to `93b0e42`. **Three stale facts, all found by my own verification rather than by a gate**, and all caused by U55 landing *while U57 was writing*: DC-12's *"byte-identical from `7f865e2` through HEAD"* clause; enumerating command 2 reading **21 not 20** (`test_stats.py` **15 → 16**, the added line being inside the very test U55 turned into a six-id sweep); and P10-5's fresh re-derivation inheriting it. **Plus a convention question I declined to settle myself** | `architect` (**resumed** `a6e6e78f99baf4ead` at **401k tok** — second override of my own threshold on this agent, on the narrower ground that I supplied every number, so the work was three edits plus judgment with no re-derivation) | `a6e6e78f99baf4ead` | **delivered** — `ed33e92`, plan **v1.21** (+63/−18), 9 tool uses. It **confirmed my three numbers rather than taking them**. **Both convention questions came back better than they went out, and one of them overruled me.** *Baselines:* it landed near my position by different reasoning — I argued *the implementer runs it against the tree they have*; it answered that this licenses **re-pointing without re-measuring**, which asserts a count nobody took, so the rule is that a baseline moves **only in the revision that re-runs the commands** — landed tables never, unlanded in the revision that re-measures, DC-12 carrying the gap. *Gloss:* I proposed a flat prohibition and **it rejected it with a reason I accept** — a gloss is what rule 5(a) asks for and Table E's non-site list is the model the plan praises, so the real defect is the restated **total**, §7 rule 4's one-home rule applied to a number. **Sites are stable against additions; totals are not.** Applying the new rule immediately caught **a second copy of both counts in v1.19's own closure block three sections away**, one already stale — removed, not updated. That is the **eighth** suggested fix overruled on the merits in this coordination, and the second where the overruled party was me | `docs/plans/small-model-benchmarking.md` **v1.21** | **U60** → — | 416k tok / 9 tools |
| U60 — Re-gate Table H at plan **v1.21** as plan-gate `## Pass 11` — its own five findings' dispositions plus what v1.21 did. **Briefed to cite `P11-*` with the *plan-gate* prefix**, because the impl gate is writing its own `## Pass 11` concurrently: the precise collision §7's prefix convention exists for, and already open once as a rule 3 raise. Told which numbers I re-verified so it re-measures none of them, and pointed at the two new §7 rule 5 conventions as the highest-value target since both govern Table F's unit and everything after | `analyst` (**resumed** `a32855305cbdf4b2c` at **267k tok**, modestly past threshold — resumed because a re-gate's whole value is the reviewer remembering what it asked for, which the review text alone does not carry) | `a32855305cbdf4b2c` | **delivered** — `68b0d14`, plan-gate `## Pass 11`, **needs changes**: 0 blockers, 1 major, 2 minors, all closeable today, prefix honoured. **Pass 10's five all closed, two further than the finding asked.** **P11-1 answers the belt-and-braces question I put to it, and the answer is that they *can* disagree** — see the trap section, seventh instance. **P11-2: it refused to treat my framing of the split baseline as a decision, correctly**, and found the deciding evidence somewhere I had not looked: DC-12 asserts in the same revision that all six residuals are unchanged across `93b0e42`, a claim obtainable only by running them there, so the document names **two measurement commits for the same six numbers**. **P11-3: the rewritten gloss pins its four sites by *line*, in the file its own table inserts into** — Rule 4a's assertions land at `:1346` and `:1367`, between `:1338` and `:1422`. **Both new conventions upheld and independently witnessed**: re-pointing without re-running would have published `envelope_arms` → **20** at `93b0e42` where the truth is **21** — my framing was one revision from licensing a count nobody took | `docs/reviews/small-model-benchmarking.md` `## Pass 11` | self → **needs changes** — and I verified the major's mechanism in both halves: `report.py:27` is `from modelbench import stats` with `:339` already spelling `stats.LEVEL_CI95_LO`, so the qualified form is that file's own convention; and a two-line probe confirms `-F` matches straight through the qualifier | 311k tok / 10 tools |
| U61 — Plan-gate Pass 11 fix round → **v1.22**. **P11-1**, **P11-2**, **P11-3**, plus the generalisation I offered for it to sharpen or reject | `architect` (**resumed** `a6e6e78f99baf4ead` at **416k tok** — the **third** override, named as such, and **declared the last resume** of this instance) | `a6e6e78f99baf4ead` | **delivered** — `9b3c3fe`, plan **v1.22** (+92/−17), 11 tool uses, reading stayed narrow. All three closed. **It changed my generalisation twice over and both changes are improvements.** I wrote *removing a name widens the match*; it answered that the mechanism is **shortening the pattern** — removing the local was only the reason to shorten — so **a pattern's specificity and its scope's breadth are substitutes**, which explains the residuals that never went wrong as well as the one that did. And it wrote the rule as a **default, not a caution**, on the ground that *a caution does not fire* and this document has watched seven residuals fail on a faithful edit: **a residual is scoped to the files its own site rows name; a wider scope is an exception the table states.** Checkable in seconds, and it would have caught **both** of Table H's forms — v1.19 was package-scoped too and survived only because its pattern was long, i.e. **right by luck**. **Two disclosed extras, both right**: all six of Table H's residuals re-scoped rather than the two under finding (a revision writing a default while leaving four same-table violations is the failure this plan keeps hitting), and **Table F's three re-scoped too**, it being the other unlanded table. Binds **prospectively** — landed tables keep their scopes, because their residuals have already been run against a real faithful implementation, which is stronger evidence than a heuristic | `docs/plans/small-model-benchmarking.md` **v1.22** | **U63** → — and I re-ran all nine myself under the new scopes at `93b0e42`: **2/0/0/1/1/2** and **1/1/1**, every stated before-value holding, plus all four gloss test-name pins unique. Residual 5 is correctly scoped to `report.py`, not `stats.py` — the one place a careless application of the new default would have silently zeroed a residual | 440k tok / 11 tools |
| U62 — impl-gate Pass 10 fix round: **N4** (the `inf`/`nan` laundering), **N5**. Briefed to state how the fix survives **Table H** and to report **line-pin drift** | `coder` (**resumed** `ab774117c13e7f953` — 190k tok) | `ab774117c13e7f953` | **delivered** — `e162ba9`, **560 → 577 tests**, ruff clean. *Killed by the **eighth** rate-limit event mid-report.* **Its transcript said "now the HISTORY entry" while HISTORY.md was already +102 lines on disk — the third time checking the disk first has changed what I did.** Work was complete; only the *report* was lost, **so I produced it myself rather than spending a resume**: both guards **mutation-killed independently** by copy-restore (4 failures and 3 — neither is redundant with the other), restored byte-identical. **Two guards, and the split is reasoned**: `_widen` refuses a non-finite **result** because the violated premise is *bounds are numbers*, a property of the output, so one guard closes all three routes in; `paired_bootstrap` additionally refuses non-finite **data**, being public and returning before any widening. **It found something the review did not have**: the review said the unclamped path returns `(nan, nan)`, *visibly* wrong — often it is not. `sorted()` does not order a NaN, so the two quantile indices land on opposite sides of the garbage. **I scanned 60 seeds and 29 return a mixed pair** — `(-0.25, nan)`, and also `(nan, -0.025)`, which reads as an *inverted* interval. A plausible bound beside a `nan` is the shape most likely to be taken for a rendering glitch | `model-bench/` source + tests + `HISTORY.md` | **U64** → — and the drift I made non-optional **did** happen this time: **+31 lines above all ten `stats.py` pins**, now at 292 · 413 · 418 · 444 · 927 · 1199 · 1215-1218, established by content-match. **Every residual is unaffected** — Table H's six, Table E's pair and Table F's three all hold: the exact-text discipline surviving its largest insertion yet, in the same file where every line pin broke | 246k tok / — |
| U63 — Plan-gate `## Pass 12` on **v1.22**, narrow and the last plan gate before Table F executes. Scoped to what is *unreviewed*: the three P11 dispositions, and the **two disclosed extras and the new rule**, none of which any gate had seen | `analyst` (**fresh** — me honouring my own threshold instead of overriding it a fourth time: the plan-gate instance was at 311k tok *and* had been party to three consecutive rounds of converging reasoning with the architect on exactly this rule) | `a8ac46d0ef9980dfc` | **delivered** — `5a018a4`, **needs changes**: 0 blockers, 3 majors, 2 minors, 1 nit, none carried. *Killed by the ninth rate-limit event with only its skeleton down; resumed and finished.* **The fresh reader earned itself.** It confirmed the new default is correct — and **better than a wash**: for Table H's third-form pair the package scope failed **in both directions at once**, on the good spelling *and* past the forbidden one — then found **both** boundary failures I suspected and could not name. **P12-1 (major) is in the unit that executes next**: Table F's residual 3 is where the new default **inverts its own benefit**, since a later `report.py` line spelling `{"binary", "continuous"}` is not noise but a **second home for the tag set**, which that table's one-home decision forbids. **P12-3 (major): the premise is falsified by the plan's own record** — *landed residuals have hit their targets* is contradicted by impl-gate F3 and F2, both verbatim in the document. **P12-4: *costs no evidence* is vacuous for residuals 2 and 3**, whose *before* is 0 under any scope including an empty one — no evidence at all about the exact pair the clause was written for | `docs/reviews/small-model-benchmarking.md` `## Pass 12` | self → **needs changes** — and I verified the three load-bearing facts: the literal occurs **once** package-wide (`results.py:386`), `report.py`'s kind cross-check really is the plausible second-home site, and both of P12-3's falsifying quotations are in the plan | 110k tok / 7 tools |
| U64 — Re-baseline Table H and the ten `stats.py` site pins to `e162ba9`; enumerating commands `bound_by` **14 → 15** and `envelope_arms` **21 → 22**. **Queued, not dispatched** — held until U63's Pass 12 lands, because folding a gate's findings and a re-baseline into one revision is how P11-1 happened, and that lesson is four units old | `architect` (**fresh** — `a6e6e78f99baf4ead` is at 440k tok and I declared U61 its last resume; this is that declaration being kept, not revisited) | — | **queued** | `docs/plans/small-model-benchmarking.md` **v1.23** | `analyst` → — | — || U64 — **Pass 12's six findings *and* the re-baseline, in one revision** → plan **v1.23**. Re-pins the ten `stats.py` sites to `e162ba9` (292 · 413 · 418 · 444 · 927 · 1199 · 1215-1218, established by content-match) and the two enumerating counts (`bound_by` **14 → 15**, `envelope_arms` **21 → 22**), with all eleven residuals confirmed unaffected. **Combining them is the P11-1 lesson applied, not ignored**: that defect came from two fixes interacting unseen because each was reviewed only against its own finding, so this brief makes *re-read your own diff as a whole and say what any two edits do to each other* an explicit deliverable, and the gate will be briefed the same way | `architect` (**fresh** — the U61 declaration kept, not revisited; `a6e6e78f99baf4ead` retires at 440k tok. Briefed with every number supplied and told to confirm rather than take them) | `ad5f2b5ac13615d36` | **delivered** — `25570d4`, plan **v1.23** (+248/−100). All six closed, none deferred. **The whole-diff cross-check I made a deliverable found eight defects, three of them created by its own edits interacting** — the exact P11-1 shape, caught this time because someone was told to look for it. Two are worth the record: it nearly transcribed **the review's own miscount** (Pass 12 calls Table H's six residuals *four `stats.py`, one `report.py`* — that is five; it is **5 + 1**) and derived it from the table instead; and its own first draft claimed `e162ba9` inserted 31 lines above **every** `stats.py` pin, which is **false** — Tables C and G's `:159`/`:162` sit above the first insertion and did not move. It also corrected v1.21's *seven landed tables* to **six**, and re-baselined Table F off its own initiative, it having been two-baselined (commands at `5878014`, residuals at `93b0e42`) — the next unit to execute now sits on one commit | `docs/plans/small-model-benchmarking.md` **v1.23** | **gate cut by stakeholder decision** → — . I verified instead: its 40-check re-derivation script (40/40), P12-1's package count (**1**, `results.py:386`), the six `Landed:` records against the seventh template match, and all four `never re-widened` sites. **My own grep undercounted the last exactly as the review's had** — markdown bolding splits `never re-**widened**`, so a line-based `grep 'never re-scoped'` reads 3 where the truth is 4 | 249k tok / 108 tools |
| U44 — **S1 impl 3 of 3**: Table **F** (the continuous carrier, `results.py`/`report.py`). Dispatched against **v1.23**, whose re-baseline means its pins resolve against the tree for the first time. Briefed that residual 3 is **package-wide by design** and that a non-zero after-count there means a second home for the tag set, **a real defect rather than a noisy residual** — and, conversely, that if a residual will not reach its target after a faithful edit it is very likely the *specification's* defect, seven prior instances being the reason to say so rather than bend code to satisfy a grep. Carries the whole-diff cross-check that has now paid twice, and adds Table F's own `Landed:` line as the record | `coder` (fresh, `sonnet` — an edit list that writes out its replacement texts and proves itself by grep is execution, not design) | `a5694ae5e8519f383` | **delivered** — `b5dab1f`, **577 → 600 tests**, ruff clean, all three residuals **1 → 0**, six mutations each killed by their pinned tests. Seven of the eight S1e tables now carry a `Landed:` record; only **Table H** remains. **I re-verified rather than accepted**: the three residuals and their `e162ba9` baselines, the suite from `model-bench/` as cwd, and an **independent mutation of the load-bearing raise** — disabling `scored_outcome`'s `measures` check killed two tests, reproducing its M1 exactly, restored byte-identical. Two slips, neither load-bearing and both the *citation-vs-fact* shape its Table G miscitation had: `HISTORY.md`'s summary line said **8 mutations** over a table of **6** (I corrected it), and it cited a `grep -c '^+def test_'` as the source of *18 in `test_results.py`* when that command returns **16** — the figure is right (two parametrized tests × 2 cases reconcile 16 defs to 18 collected, and 18 + 5 = the observed +23), the cited command is not. **Its `Landed:` line is 20 lines against a 3-line template** — substance verified and accurate, length is a milestone-close follow-up in the same class as the over-long revision notes. (was ⏸, resumed 2026-09-08 with the stakeholder's answer: **carrier only, verdict producer sequenced as its own unit before S1 closes**) — was ⏸ "Table F's `report.py:623-789` row cites §4 S1's `compare_report` block, whose continuous branch hands `diffs` to **`continuous_verdict()`** and converges both branches on a union type `Verdict | ContinuousVerdict` — neither exists, no Table F command or residual reaches them, and the plan states it does not write them. Implement the carrier only and sequence the verdict producer as its own unit, or absorb it here?" — relayed 2026-09-08. **The eighth real finding**, and it stopped before touching a file. I verified all of it: `continuous_verdict`/`ContinuousVerdict` appear only in three `stats.py` comments and one `test_stats.py` docstring; `report.py:723` renders `### Family-wise error control` on `len(family) > 1` with **no kind check at all**; the plan's disclaimer is real *(one correction: it sits at `:3954` inside **Table G's** block, not Table F's — the agent's stated range 3723–4213 spans F, G and H; Table F is 3723–3898)*. **This is a hole in the decomposition, not new scope** — line 3720 already ties Table F to S3's done-condition 1 | `model-bench/` source + tests + `HISTORY.md` + Table F's `Landed:` line | **gate cut by stakeholder decision** → — | — |
| U65 — **The verdict producer, and the hole that produced it.** Build `continuous_verdict()` / `ContinuousVerdict` from `docs/plans/small-model-benchmarking-ml.md` §3.4 **Rule 8**, plus §4 S1's family-loop continuous branch and §3.3 (iv)'s mixed-kind refusal. **Not a follow-up and not residue** — the stakeholder's standing rule is that a residual is acceptable only when *blocked on unbuilt work*, and Rule 8 is fully specified, so this is buildable and therefore owed before S1 closes. It is also **not new scope**: the plan's own line 3720 already ties Table F to S3's done-condition 1, and without a verdict the embedder pack's only verdict metric renders nothing. **The decomposition never gave it a table** — it fell between the plan (which disclaims writing it) and the `-ml` note (which specifies it and owns no code) | `tdd-engineer` (fresh, `sonnet`) — **the routing question resolved to a contract**: Rule 8 writes out an exact keyword-only signature, a 13-field sibling return type, **five** refusals (the four it groups plus `support` with `lo >= hi`), exact-rational quantile levels, a derived clamp, and a three-decimal print precision. That is a behaviour contract stated in advance, so test-first is the efficient path, not ceremony | `a87b39a3029c65a30`, then **`a7eeda8867aac16fa`** (fresh, state-recovery, 2026-09-09) | **delivered** — `d45e5ff`, **600 → 627 tests**, ruff clean, 13 mutations. **Killed twice by platform failures first, neither its fault**: the **ninth** rate-limit event mid-run — HTTP 429 session limit, which had already reset by the time I processed the notification). **The disk is why the resume was cheap**: `test_stats.py` **+348 lines / 22 test defs**, `stats.py` **untouched** — it died in the *red* phase, having written the whole Rule 8 contract as tests and no implementation. I resumed the same agent rather than dispatching fresh because those 22 tests encode an undocumented design decision — two helpers, `_family_ci_levels` and `_support_clamp` — whose reasoning exists only in its transcript. **One fact it could not have known when it stopped, and which I supplied:** the missing `ContinuousVerdict` import fails `test_stats.py` at **collection**, so the *entire* suite is un-runnable and the 600 baseline unreachable until the implementation lands — the tree is in a state that cannot be integrated or left, which makes finishing the green phase the whole job rather than a nicety. Dispatched on U44's delivery. **Split at the boundary the note itself draws**: Rule 8 ends *"`report.py` handles a union of the two verdict types; that seam is the plan's"*, so the producer (note-owned, `stats.py`) is this unit and the seam (plan-owned, `report.py`) is **U66**. One mega-brief over both files is the dispatch-sizing failure this coordination has already paid for. Scope fenced hard: `report.py`/`test_report.py` **off-limits**, and U44's deliberate `MetricKindError` raise and its pinning test are **left standing** — retiring them is U66's, and that test's docstring is U66's seam description. Briefed that both of Rule 8's stated engine preconditions already landed at `cc28d48` (Tables E and G), so `levels` and the conditional `clamp` are not to be rebuilt; and warned off the `Fraction(str(α))` vs `Fraction(α)` trap and the *no percentile parameter, no clamp parameter* rule, which is the whole design rather than an omission. Also told to derive any per-file test breakdown from a command it actually ran — the `grep -c '^+def test_'` slip in U44 undercounts parametrized cases. **Then the resume never executed** — the session ended before it ran, and the disk proved it: `stats.py` still carried **zero** of the four symbols, `test_stats.py` still exactly +348. So the second recovery was a **fresh dispatch, not a third resume**, because the reasoning that made resuming right on 2026-09-08 had **expired**. I resumed then to preserve an undocumented decomposition — two helpers, `_family_ci_levels` and `_support_clamp` — living only in that agent's transcript. It is now *legible on disk*: the 348 lines pin both helpers' signatures and behaviour exactly (`_family_ci_levels(0.05, k)` → `1/40, 39/40` · `1/80, 79/80` · `1/120, 119/120`, with `Fraction(0.05) != Fraction(1, 20)` asserted first so a `Fraction(α)` implementation cannot pass; `_support_clamp(None) is None`). **A completed red phase converts a transcript-resident design into a disk-resident one — which is what makes a fresh dispatch cheap, and is a reason to prefer TDD routing where a unit is likely to be interrupted.** The recovery brief fences the inherited tests: green them, never weaken or delete an assertion to reach green, and if one is genuinely *wrong* stop and say so rather than editing it — seven prior instances make *specification defect* a diagnosis worth voicing — **and that instruction immediately earned itself**. Its mutation testing found the **ninth real finding**: the inherited `test_continuous_verdict_mrr_worked_case_from_the_note` **cannot bind the clamp it claims to test** — `diffs = [1.0] * 10` has zero variance, so every resample is identical, the half-width is zero and `_widen` returns the point interval whatever the clamp is. Its docstring asserts the opposite (*"constructed so the clamp binds"*). It reported rather than edited, exactly as briefed, and added a test that does bind. **I reproduced the mutation myself: only the new test kills it.** This is the archetype the mutation-testing clause exists for — a test green on arrival that proves nothing. **I pushed back on its second finding.** It reported `distinguishable`'s strict `>`/`<` as untested and declined it as *"a Monte-Carlo boundary"* not worth the construction effort. Untested is right; unreachable is not — `continuous_verdict([0.0]*8, …)` returns `ci = (0.0, 0.0)` **exactly**, no Monte-Carlo luck involved, and mutating `stats.py:1619` to `>=`/`<=` flips it to `distinguishable=True` while **all 627 tests still pass**. Blocked on nothing, so **deferred by choice** and refused under the standing rule; sent back with the repro — **and it agreed and closed both**, `cf54f5b`: the false docstring corrected to state what the test actually proves (assertion untouched, only the three docstring lines removed in the whole file), and `test_continuous_verdict_is_not_distinguishable_at_an_exact_zero_boundary` added. **627 → 628.** I re-mutated `stats.py:1619` myself: the new test is now the *sole* failure, where the same mutation previously passed all 627. It also recorded the reversal in `HISTORY.md` in its own words rather than quietly amending the number, which is the honest form |  `modelbench/stats.py` + `tests/test_stats.py` + `HISTORY.md` | **gate cut** → — | — |
| U66 — **The report-side seam**: `compare_report` pass 1 resolving each member's kind from its aggregate type, the plan's **§3.3 (iv)** mixed-kind-family refusal (**refuse the whole family's verdicts, not the offending members** — dropping the minority kind shrinks a pre-registered `k` after the results exist; **no arm excluded, nothing raised**, every number through labelled `exploratory — no significance claim`), the family-loop continuous branch that calls U65's producer and forwards `ContinuousMetric.support`, the `Verdict \| ContinuousVerdict` union renderer and its `- decided by:` bullet, the `Family-wise error control` block rendering **only where a Holm ladder actually ran**, and §3.3 (iv)'s two label sites (the family filter at `report.py:767`, which structurally excludes the members needing the label, and the headline fallback). Retires U44's placeholder raise and rewrites its pinning test | `coder` (fresh, `sonnet`) — a wiring seam across enumerated sites, with U65's contract already fixed and mutation-gated | `ae4c487bab5b91f77` | **delivered** — `c926308`, **628 → 634 tests**, ruff clean, 10 mutations, **Table F's `report.py:623-789` row now fully satisfied and the table complete**. Dispatched 2026-09-09 against `cf54f5b` (**628 green**). Sites located against the *current* tree, since U44 shifted `report.py`: `compare_report` `:507`, the family loop `:674`, `holm_steps` `:704`, the un-kind-checked `if len(family) > 1:` at `:753` guarding `:762`'s heading, and the Exploratory filter `:809` that structurally excludes the members needing the refused-family label. Briefed hard on the one thing easy to get subtly wrong: §3.3 (iv) refuses **the whole family's verdicts, not the offending members** — dropping the minority kind shrinks a pre-registered `k` after the results exist — with **no arm excluded and nothing raised**, which is what separates it from DC-10's per-member kind *disagreement*, where the arm **is** excluded. Told to retire U44's placeholder by **reachability, not deletion**: the `scored_outcome` raise stays correct as a contract and must simply stop being reached by a well-formed continuous member — **done exactly that way**, `results.py` untouched. **What it got right unprompted:** the `-ml` §3.2d fold (*the mean over a unit's items*) was **cited, not invented** — I checked the note and it specifies it verbatim; the §4.3 asymmetry counting for one-arm-only and one-arm-scoreable units is implemented rather than silently dropped; and it noticed that `diffs`' order feeds a **seeded** bootstrap *by index* while `set` iteration is hash-randomized per process, so it walks insertion-ordered — a reproducibility defect avoided, which I confirmed across three `PYTHONHASHSEED` values. I re-ran two of its mutations independently, including the mean-vs-first-item near-miss it had to redesign a fixture to catch | `modelbench/report.py` + `tests/test_report.py` | **gate cut** → — | — |
| U67 — **The eleventh real finding, and the first that is mine rather than a delegate's.** `compare_report` calls `continuous_verdict(family=family, …)`; mutate that one argument to `family=[metric]` and **all 634 tests pass**. Not cosmetic: the producer derives its levels from `k = len(family)` as `α/(2k)` and `1 − α/(2k)`, so collapsing `k` to 1 means **the Bonferroni correction silently does not happen** for an all-continuous family — and §3.3 (iv) and Table G exist precisely because such a family has *nowhere else to put it*. Too-narrow intervals, too-confident verdicts, no symptom. **The existing `k = 2` test does not catch it**: `test_an_all_continuous_family_takes_its_correction_in_the_interval_not_a_ladder` asserts the explanatory sentence, the absence of Holm, two bootstrap lines and no exclusion — **every one of which survives the collapse**, while the sentence the report prints becomes false. **Third instance in three units of a test asserting less than its name claims**, each found by mutation and none by the suite | `tdd-engineer` (**fresh, not a resume** — U66's agent is at **292k tok / 92 tools**, past the point where continuing buys anything, and this follow-up is self-contained with an exact repro) | `a74b00eaa79f73979` | **in-flight** — dispatched 2026-09-09. Behaviour is **correct**; only its proof is missing, so scope is tests + `HISTORY.md`, and the brief says to stop and report rather than fix if it concludes otherwise | `tests/test_report.py` | **gate cut** → — | — |
| U16 — Close R-13: `_percentile` definition + denominator under informative missingness | `data-scientist` (fresh) | `a7da5de9c6bbf19a1` | **accepted** — `460940c`; resumed to republish §11.7 with measured values | `docs/plans/small-model-benchmarking-ml.md` v1.9 §11 | re-gate → — | 176k tok / 40 tools |

| U14 — Fix unit: **all Pass 4 majors + minors, both gates** (scope expanded mid-run) | `tdd-engineer` | `af08841933828b12c` | **accepted** — `5878014` | `model-bench/**` (10 files, +1490/−61); 353→389 tests | re-gate (both, fresh) → — | 348k tok / 130 tools |
| U18 — Rule on §3.4 Rule 4's closed-form half + P3-5's seed contract | `data-scientist` (fresh) | `ae4bd1d99239907a6` | **accepted** — `a5f42f6`; recovered from disk after a kill, headline claim reproduced independently | `docs/plans/small-model-benchmarking-ml.md` **v1.11** — the closed form is binding | teco-verified | — (killed before reporting) |
| U10 — Plan sweep (n-ML-7 + §5 stage-scoping, flagged 3×) | `architect` (fresh) | `ae512d667fe7f0c49` | accepted | commit `9b63c5c` — plan **v1.7**; 6 restatements withdrawn, stage table added | teco-verified | 139k tok / 64 tools |
| U7c — Plan sweep: `PackRef.contentHash` is now `str \| None` | `architect` | — | abandoned — **delivered by U8b** in plan v1.6 (`5594be8`), never dispatched separately | — | — | — |
| U6e — Fold the adjudication's sharpened principle into the note | `data-scientist` | — | abandoned — **delivered by U8a** as Rule 3's generalisation in note v1.6 (`a54a667`), never dispatched separately | — | — | — |
| U6b — Republish the `-ml` fixtures at 10 dp | `data-scientist` | `a394671cfc28bef87` (resumed) | accepted | `docs/plans/small-model-benchmarking-ml.md` **v1.5** — + Rule 7, floor rounding corrected | teco-verified | 218k tok / 8 tools |
| U6c — Appendix A `PackRef` + §3.4.1 enumeration are stale | `architect` (fresh) | `a5ca583515c0979f1` | accepted | `docs/plans/small-model-benchmarking.md` **v1.5** | teco-verified | 116k tok / 43 tools |
| U6d — Adjudicate the declined floor-rounding finding | `data-scientist` (reviewer) | `a7fbf4d59bfa1d0da` (resumed) | accepted | ruling: **truncation upheld, reviewer's own ask withdrawn** | — | 179k tok / 2 tools |
| U5 — S2 packs, LM Studio adapter, host info, convo, tooling, runner | `coder` | — | queued (after U4) | `modelbench/{packs,lmstudio,hostinfo,convo,tooling,runner}.py` + tests 7b–12 | `analyst` → — | — |
| U6 — S3 `embedder` pack + `refresh_golden.py`, first live run | `coder` | — | queued (after U5) | `packs/embedder/**`, `scripts/refresh_golden.py`, one stored `RunResult` | `analyst` → — | — |

Unit sizing for U5 is provisional: S2 creates six modules, which is at the split boundary. It is
re-drawn against U4's actual delivery before dispatch.

## Documentation impact scan

| Document | Impact | Owner of the update |
|---|---|---|
| Root `AGENTS.md` | New `model-bench/` bullet in **Structure** + row in **Component docs** | U2 (`coder`), in the same change |
| `model-bench/README.md`, `AGENTS.md` | Created by U2; README states the three non-features (no CI, no gate, no leaderboard) | U2 |
| `model-bench/docs/{BACKLOG.md,HISTORY.md}` | Created by U2; `HISTORY.md` takes an entry per delivered stage | each implementing unit |
| `docs/HISTORY.md` (repo root) | Entry when the coordination closes | teco, at close |
| `docs/requirements/small-model-benchmarking.md` | Stays where it is (its own footnote says so) — no move | — |
| `docs/BACKLOG.md` (repo root) | Carries FR-21a's deferred judged-reply-quality layer as an open item (plan R-4) | S8 — out of this pass's scope, flagged |

## Decisions and events

- **2026-09-02 — kickoff.** Plan read; §7 declares ready to implement, with one verify-during-
  implementation item (R-1: whether `lms ps --json` exposes the KV-cache setting on a loaded model,
  checked during S2/U5). No stakeholder answer is outstanding on the plan itself.

- **2026-09-02 — U1 delivered, verdict `needs changes`.** All three blockers land inside S1–S3, this
  pass's scope, and none is a disagreement with the design — each is a guarantee the plan *states*
  with no mechanism behind it. B-1: the tool-caller instrument ignores clustering, so
  `min_detectable_difference` and `verdict()` are anti-conservative against `-ml` §7.2/R1. B-2: the
  §3.1 metric-agreement cross-check is not constructible (the ranked lists it needs are not an
  artifact), and it is the *only* mitigation behind D1, the plan's central decision. B-3: the BM25
  reference arm has no result-schema representation, so `store()`'s own no-bypass fingerprint rule
  rejects it — decided in S1, surfaced in S3.
- **2026-09-02 — mid-run correction relayed to U2 while in flight.** U1's finding m1 (S0's
  done-condition is unsatisfiable: pytest 9.1.1 exits 5 on zero collection) was `SendMessage`d to
  the running `coder` rather than held. It landed: S0 ships one real test
  (`tests/test_package.py`, version-vs-distribution-metadata) instead of configuring the exit code
  away, and `model-bench/AGENTS.md` records *why* that route was refused.
  **Caveat worth carrying:** the agent's returned summary still asserted "no placeholder test (the
  brief forbids one)" — prose that predated the correction it had already applied. The tree, not
  the report, was the state of record; teco verified against the tree.
- **2026-09-02 — U2 accepted.** `./setup.sh && .venv/bin/python -m pytest -q` → `1 passed`, exit 0;
  `ruff check .` clean; both re-run by teco, not taken on report. Note the done-condition only holds
  with `model-bench/` as the working directory — from the repo root there is no root
  `pyproject.toml`, so `rootdir` becomes the monorepo and collection walks into `mcp-monitor/tests`
  (measured: 8 collection errors, exit 2). That belongs in the plan; it is folded into U3.
- **Follow-up, out of scope, not chased:** root `AGENTS.md` was already past its own ~2 500-word
  smell threshold before this coordination (2 729 words at `HEAD`, 2 823 after U2's two required
  insertions). A bloat sweep is overdue and is nobody's unit here.

- **2026-09-02 — three stakeholder decisions taken, closing the gate's open questions.**
  (1) **Tool-caller sampling → 12 distinct scripts × 1 run, temperature 0**, replacing 4 × 4
  replicates; same run budget, authoring cost lands in S6. Taken on `-ml` §4.5's own argument.
  (2) **`guard-judge` gets no `primaryMetric`** — both class-conditional rates, equal weight, no
  headline number; the stakeholder declined to rank false-advance against false-suspend. Design
  consequence flagged to the `architect`: a pack *without* a primary metric must become a
  first-class case in the manifest schema and `report.py`, which today assume one always exists.
  (3) **The S3 self-check is a diagnostic, never a gate** — below-baseline does not block S3; the
  deviation and its investigation go in the test report.
- **2026-09-02 — U3 split into two file-disjoint parallel units.** B-1 and M-1 are statistics
  findings owned by the method note, not the plan, so `data-scientist` revises the note while
  `architect` revises the plan. The `architect`'s brief forbids restating the note's formulas or
  pinning `stats.py`'s clustering signatures — the gate found silent divergence between the two
  documents, and collapsing to one source of truth is the fix.
- **2026-09-02 — a second, unrelated session is committing to this repository concurrently**
  (the `salesperson-ui` / `falkor-chat` storefront coordination; commits `acb5a2a`, `0efc014`,
  `2f7938d`, `1951d94` landed mid-run). Every brief in this coordination is fenced to its one
  file, and teco verified both integration commits touched only their own units' paths.

- **2026-09-02 — U3b delivered, plan v1.2.** Every blocker fixed rather than declined. Two are worth
  recording because they changed the design rather than patching text. **B-2:** the metric-agreement
  cross-check is rebuilt as a hand-transcribed `metrics_agreement.json` (20 cases, with
  `sourceGitSha`/`sourceSha256`), transcribed manually *because* only 6 of the 20 live in
  `parametrize` tables — a mechanical extractor would capture a third of them and pass. The
  `architect` states the honest residual: this is weaker than v1.1 claimed, and D1 is re-argued on
  what actually carries it. **B-3:** `armKind ∈ {model, deterministic}` with both a required-field
  *and* a forbidden-field map — the forbid half is what makes a BM25 arm declaring
  `modelKey: "bm25"` fail loudly instead of silently passing the fingerprint rule.
- **Decision 2 built as first-class, not special-cased:** `verdictMetrics` (the pre-registered
  family) plus an **explicit** `primaryMetric` that may be `null`; `validate_pack` rejects an
  *omitted* key, and `report.py` has no code path that synthesises a headline from the family.
- **One nit withdrawn rather than fixed** (m7): the gate's proposed `ruff` mechanism was found not
  constructible, and was replaced by `validate_pack`'s AST walk plus `run` failing closed. Recorded
  because a declined finding that is silently dropped is the failure mode the disposition table exists
  to prevent.

- **2026-09-02 — U3a delivered, note v1.2, and it corrects two of its own v1.1 numbers.** New §3.4
  fixes B-1 structurally rather than advisorily — `PairedOutcomes.from_units()` is the only
  constructor and raises on a repeated unit id, `resolving_power()` takes its arguments keyword-only
  **with no defaults** (a `1.0` default would rebuild B-1 by omission), and `verdict()` refuses to
  let McNemar decide whenever `design_effect > 1.0`. Recomputed by exact search: the "fully
  clustered ~65 pp" figure was a `8/n` mnemonic (exact **57.8 pp**), the boundary tier's "53 pp" is
  **47.6 pp**, and v1.1 called the CI width ratio the design effect when Kish DEFF is that ratio
  **squared** — an implementer following v1.1 would have overstated effective *n* by ~2.7×.
  **M-1 is not a numerical defect:** `1.96` is a typographic rounding of the exact constant, moving
  every fixture bound by ≤ 3×10⁻⁴ pp; it is pinned for equality-assertion reproducibility, not
  correctness.
- **2026-09-02 — integration defect caught by teco, not by a gate: the two parallel revisions
  diverged.** Same two concepts, two vocabularies (`verdictMetrics`/`primaryMetric` in the plan vs
  `primaryMetrics`/`headlineMetric` in the note), plus a **semantic** divergence on `guard-judge`'s
  pair — the plan gave a verdict to `advanceRecall`, while the note requires both co-primaries be
  error rates in the **same direction** (`falseAdvanceRate`/`falseSuspendRate`). Routed back to the
  `architect` (resumed) with naming authority, since the manifest schema is the plan's; the note
  will be aligned to whatever it picks. **The lesson is the dispatch's, not the agents':**
  file-disjoint is not interface-disjoint — two units revising documents that reference each other's
  vocabulary need the shared vocabulary pinned in both briefs, or serializing.
- **2026-09-02 — a premise teco gave the stakeholder was wrong, and is being corrected to them.**
  Option 1 of the sampling decision was presented as "same total run budget". It is not: 12×1 is
  **~one quarter** of the previous inference budget (~80 turns per model against ~320). What is
  unchanged is the *authoring* budget. The decision itself still stands on its own merits — the old
  nominal 48 had DEFF = 4 and an effective *n* of 12, so the honest floor (50.0 pp) and MDD₈₀
  (57.8 pp) are **identical before and after**; only the printed *n* changed, and the tool lost a
  claim it could not support rather than losing power. But the stakeholder is owed the consequence:
  the **15–50 pp band is dark**, so the ~30 pp ministral duplicate-instruction defect is not
  resolvable at any observed outcome, and per-turn positions 5+ (n=8, then n=4) are descriptive
  only. Buying it back means 48 distinct scripts × 1 run (floor 12.5 pp), whose binding constraint
  is FR-19 human verification of 36 more scripts, not compute. Recorded in `-ml` §10 with a costed
  reversal trigger — *the first tool-caller comparison returning "not distinguishable" with an
  observed difference in the 15–50 pp band*. **Nothing in this pass is blocked on it**; the decision
  point is before S6, which is out of scope here.

- **2026-09-02 — U3c accepted.** FR-22a's illustration now reads "12 distinct scripts — 4 per shape
  across 3 shapes — run once each at temperature 0"; the requirement's substance is byte-identical.
  `tico` swept the rest and reports FR-15/FR-16/FR-18/FR-20/FR-22 and AC-5 all still true, and that
  **no FR or AC ever named a primary/headline metric for any role** — so stakeholder decision 2
  needed no requirements change at all. It also names the near miss explicitly: the *rejected*
  branch of the sampling question (replicates at temperature > 0) is the one that would have
  changed what FR-18's pinning means.
- **Deferred to the same pre-S6 decision as the sampling budget:** the Out-of-scope bullet
  "Measuring small differences" still claims the lab resolves "differences of roughly 15 percentage
  points and up" at "~20–40 runs per arm". The tool-caller pack now sits below that range
  (floor 50.0 pp). `tico` deliberately did **not** edit it, on the reasoning that decision 1 did not
  falsify it — the old 48-conversation design already had an effective *n* near 12, so the true
  resolving power never changed and the decision only made it visible. Restating the range is a
  scope change the stakeholder owns. **It belongs in the same packet as the 48-distinct-scripts
  question**, not in a wording fix.

- **2026-09-02 — U3b delivered v1.3; the divergence is closed, and the fix outlived the instance.**
  Naming decided as the **synthesis**, not either candidate: `verdictMetrics` + `headlineMetric`.
  `primaryMetric` is **retired rather than redefined** (re-pointing an established name at "may now
  be `null`" is its own trap), and `primaryMetrics` was rejected because it sits one character from
  that retired singular — indistinguishable in a JSON manifest or a diff, in the one field whose
  entire job is pre-registration. `guard-judge`'s pair was adopted from the note verbatim, `@slice`
  suffixes dropped rather than introducing a third vocabulary.
- **Reading the note's new §3.4 surfaced three further divergences teco's message had not listed** —
  `PairedResult` was the plan's own invention and is withdrawn in favour of the note's
  `PairedOutcomes`/`ResolvingPower`/`Verdict`; §3.4's six rules are now named as a binding contract
  in S1's done-conditions; and the note's Rule 6 carried a plan-side obligation nobody had written
  down (`validate` must fail a pack declaring `replicatesPerScript > 1` while only the one-level
  `cluster_bootstrap` exists). **Routing a known defect to the agent that owns the document found
  three more than the coordinator's own cross-check did.**
- **The recurrence, not just the instance, is addressed:** plan §7 now carries a **version-pairing
  block** — plan v1.3 ↔ note v1.2, the shared vocabulary and the shared metric pair named, with the
  standing rule that revising either document must sweep the other in the same pass.
- **Known stale, deliberately not chased:** that pairing block will read "note **v1.2**" once the
  in-flight rename lands the note at v1.3. Flagged into the Pass 2 brief as already-reported rather
  than spending a round trip on one token.

- **2026-09-02 — U3a delivered note v1.3; the rename was not mechanical after all.** Three sentences
  were **wrong** under the new vocabulary rather than merely awkward, and a find-and-replace would
  have shipped all three: §7.3's heading and §3.3 both said *"two headlines rather than one/none"*,
  which is backwards — the pack has **zero** headlines and two verdict metrics — and §4.6 still
  labelled `cleanThroughTurnH` `Primary:`, the last place the retired word did structural work.
  Four occurrences of the retired singular survive **deliberately**, naming it as history: retiring
  a name is only legible if the name still appears somewhere saying it was retired.
- **One trap closed that neither document's wording had closed:** the note now states *why*
  `advanceRecall` carries no verdict — a metric and its own complement are **one test, not two**, so
  counting both inflates *k* to 3 against a difference that is by construction identical, costing
  resolving power (α=0.017 rather than 0.025) for zero information. An implementer reading only
  "printed as a complement" could reasonably have added it to `verdictMetrics` to be thorough; that
  is now explicitly a defect.

- **2026-09-02 23:50 — U3d killed by a platform rate limit (session cap, HTTP 429) before writing
  anything.** Re-dispatched to the **same** `analyst` agent id rather than a cold spawn: a compact
  Pass 2 depends on the reviewer holding its own Pass 1 reasoning, which a fresh agent would have to
  reconstruct from the review document at full cost. State recovery was cheap and was verified
  before re-dispatch — no `## Pass 2` section existed, and every document under review was already
  committed (`5aa7c83`), so nothing was lost and nothing needed reconciling. The re-dispatch brief
  carries the state-recovery instruction explicitly and asks for findings to be written
  incrementally, so a second kill costs partial work rather than all of it.

- **2026-09-02 — U3d delivered: `approve with suggestions`.** All 3 blockers and all 11 majors
  closed; every minor and nit closed or explicitly withdrawn. The reviewer re-derived **every**
  changed figure from scratch (nine MDD₈₀ values, `b_min` at both alphas, guard-judge's four bounds,
  McNemar p at b=12, Rule 5's ρ=1 identity) and both `data-scientist` self-corrections reproduce.
- **B-1 is closed, but narrower than it reads — the most valuable finding of the pass.** Rules 2–5
  genuinely make the wrong thing not typecheck, but **Rule 1 is not the mechanism it appears to
  be**: `from_units` raising on a repeated unit id only fires if the caller passes the *cluster*
  key as the unit id — 48 distinct *conversation* ids drawn from 12 scripts are unique and would be
  accepted. What actually closes B-1 is Rule 6 (`validate` failing `replicatesPerScript > 1`).
  Hence **N-1**, whose part (c) is the one that matters: S1's synthetic clustered fixture must
  assert the unit id is the **cluster** key, or the test passes while testing nothing.
- **N-3 — the residual third divergence, and not one teco predicted.** `-ml` §4.6 still defines `H`
  as *equal to* `min(script length)` where the plan makes it manifest-declared and validated `≤`.
  On its own that is a stale clause; combined with the plan's own new precedence rule (*where the
  two disagree, the note is right*) it becomes a **live regression of M-11**. The version-pairing
  block that was added to prevent recurrence is what makes this one bite — a precedence rule
  propagates staleness instead of containing it.
- **N-2 (S5/S6 scope) — `basis: "by-construction"` is an unverified attestation.** `-ml` §4.5.1(iii)
  prescribes the determinism probe as its evidence, and a grep of the plan finds **zero**
  occurrences under any name: no stage, no done-condition, no budget for its two conversations.
  Closing it needs no new statistics — a non-identical probe degrades `basis` to `assumed`, which
  via Rule 4 automatically moves McNemar out of the decision seat.

- **2026-09-03 — U3e/U3f accepted; plan v1.4 ↔ note v1.4, and the gate's findings closed harder
  than they were raised.** N-1's fix uses the `data-scientist`'s outermost-component **rule** rather
  than a per-pack field list (which would go stale the moment a pack is added): `sampling.pairingKey`
  is ordered pack data and `sampling.analysisUnit` is fixed by rule as its outermost component, with
  no parameter through which a call site could choose otherwise. `validate_pack` enforces it by
  **two independent routes** — structurally (`analysisUnit == pairingKey[0]`) and **by arithmetic**
  over the unit's own values, which is the one that catches a *consistently* wrong choice the
  structural check cannot.
- **The `architect` corrected an overstatement of its own that the gate exposed.** S1's DC-4 had
  claimed Rule 1 was "the mechanism that stops a clustered design reaching `verdict()`". It now names
  Rule 1 a **backstop** and points at Rule 6 and the `sampling` contract as load-bearing — left
  standing, that sentence would have propagated the exact misreading into the S1 brief.
- **N-2 closed with a fail-safe stronger than the gate asked for:** `basis = "by-construction"`
  requires `replicatesPerScript == 1` **and** the probe ran **and** both vectors were identical —
  otherwise `"assumed"`, **including when the probe never ran**. An unrun probe cannot silently buy
  the stronger instrument, so N-2 cannot recur by the omission that created it.
- **The precedence rule that caused N-3 is gone, replaced by ownership.** v1.3's "where the two
  disagree, the note is right" did not resolve a conflict — it propagated a stale clause, turning a
  fixed M-11 back into a live regression. **A blanket precedence rule launders staleness with
  exactly the authority it was given to settle disputes, and the more trustworthy the senior
  document, the more efficiently it does so.** Replaced with three rules: a disagreement is
  *presumed staleness* reconciled by which side changed last; precedence applies only when both are
  current and is split **by ownership, not seniority**; and neither document resolves a disagreement
  by editing the other.
- **Context-budget note for future routing:** `architect` is now at ~254k cumulative tokens and
  `analyst` at ~250k. Per teco's own rule, a further *small, self-contained* follow-up to either
  should be a **fresh dispatch**, not a resume — resuming buys their undocumented reasoning at a
  cost that no longer pays for itself.

- **2026-09-03 — U4 delivered; teco re-ran everything rather than accepting the report.** 233 passed,
  `ruff` clean, `pytest --collect-only` = 233 and `pytest -m live` = 233 **deselected** (so no
  `live`-marked test exists yet and nothing was quietly making real calls under the default run),
  and `grep` over `modelbench/` finds **no** `urllib`/`requests`/`http` import, independently
  confirming S1 is offline.
- **The mutation testing earned its place.** Eleven deliberate breaks, ten killed on the first try —
  and **the one that survived exposed a test passing for the wrong reason**: the older-schema test
  kept `BENCH_SCHEMA_VERSION` at 1, which made "validate against the record's own schema" and
  "validate against the current schema" indistinguishable. Rewritten to move the current schema to 2
  and assert both directions in one load. A reject-everything mutation of `validate()` fails 117
  tests, so the refusal assertions are not passing trivially.
- **A defect no test caught, found by reading rendered output instead of assertions:** when arm B
  won, `verdict()` re-oriented the difference to the winner (`+66.7 pp`) but left the CI in A−B
  orientation (`[-86.2, -29.9]`) — a plausible-looking, internally contradictory line that nothing
  raised on. Fixed test-first. **Worth generalising: for a reporting instrument, "the assertions
  pass" and "the output is coherent" are different questions.**
- **Both gates dispatched fresh rather than resumed**, and for two different reasons: `analyst`'s
  prior instance is at ~250k tokens with its whole reasoning already written into the review, and a
  fresh `data-scientist` **re-deriving** the figures is stronger evidence than the note's own author
  confirming them. Both were told to write findings incrementally, since a gate in this coordination
  was already killed once by a platform rate limit.
- **All doc edits are held until both gates land**, deliberately: three defects the implementer
  found (an unassertable tolerance in the note, and two stale enumerations in the plan) all route to
  documents the two reviewers are **reading right now**. Editing under a reader is the read-write
  race version of the mistake that produced the v1.2 divergence.

- **2026-09-03 — U5a delivered: `needs changes`.** The reviewer ran **29 source mutations** of its
  own against a scratch copy (working tree untouched): **19 killed, 10 survived** — against the
  implementer's own 11. Four of the six majors *are* those surviving mutations, i.e. tests that pass
  against a broken implementation of a stated guarantee. **The lesson for briefs: asking an
  implementer to mutation-test its own work is worth doing and is not a substitute for a reviewer
  doing it independently — the implementer mutates what it was thinking about.**
- **Blocker B-1 is the CI-orientation defect's twin, in a different metric.** Holm–Bonferroni is
  *printed* but never *applied*: `report.py` calls `verdict()` without the `alpha_step` parameter
  `stats.py` built for exactly that purpose, so every metric is decided at plain Bonferroni α/k.
  Reproduced at k=2, the report declares a metric "not distinguishable … does not reach alpha=0.025
  (p=0.031)" and **two lines below** prints its threshold as `0.0500`. Conservative in direction, so
  no false positive — but self-contradictory rendered output, which is the same defect class the
  implementer had already found and fixed once for the CI orientation. **Twice now, in one stage, a
  defect has lived in what the instrument *says* rather than in what it computes.**
- **All four of the implementer's judgement calls were independently confirmed**, including that
  `packs.py` contains no loader (verified line by line: no `hashlib`, no `ast`/`importlib`, no
  row-count check) and that the two stale enumerations are **the documents'** defects, not the
  code's. **DC-5(c) was judged the best-built test in the diff:** mutating the pairing index to
  `pairingKey[-1]` still raises, so assertion (2) stays green and assertion (1) is the only thing
  that catches it — the three-assertion structure was load-bearing exactly as specified.

- **2026-09-03 — U5b delivered: `needs changes`, and the arithmetic came back clean three ways.**
  The reviewer re-derived everything from scratch (60-digit `decimal` Wilson/MOVER-D, exact
  `Fraction` McNemar, independent rational-power bisection for MDD) and got **three-way agreement**
  between its own derivation, the note's published table and the module: all ten MOVER-D bounds, all
  five p-values bit-exact, both `b_min` floor tables, §7.1's exact MDD column, the ρ=1 identity.
  Rules 1, 2, 3 and 5 are genuinely binding in code rather than conventional. **Dispatching this
  fresh rather than resuming the note's author is what made that evidence worth having.**
- **Blocker B-ML-1 — B-1's shape, one layer in.** Rule 4's clustered branch *substitutes* a paired
  bootstrap over the **rows** of the paired table — an i.i.d. resample of observations the design
  effect says are correlated — so it **changes the instrument's name, not its interval**. Measured:
  the CI is identical at DEFF 2, 4 and 7, and is *narrower* than the MOVER-D it replaced; at DEFF=7
  the report calls a 15.0 pp difference distinguishable while its own mandatory line says nothing
  below 105.0 pp can reach significance. Root cause is a **missing primitive**: `cluster_bootstrap`
  computes a single-arm pooled rate, not a paired difference over clusters, and **has no caller**.
- **The reviewer supplied the invariant that catches the whole class**, which is worth more than the
  fix: *no verdict may be `distinguishable` when |diff| < `observable_floor`*. It verified the
  McNemar path satisfies this exhaustively and the clustered path violates it in every row. That is
  a property, not a case — it closes defects nobody has thought of yet.
- **Both gates independently found the Holm defect** (printed but not applied). Independent
  agreement from two reviewers with different briefs is stronger evidence than either alone.
- **Tolerance adjudicated — the defect is the note's, and the implementer was right.** The published
  table is **under-precise, not wrong**: `(34,6,0,0)`'s lower bound is 0.031762869443 against a
  published 0.031763 — a 1.31e-7 gap, 131× the mandated 1e-9 — while the delivered float sits within
  1.44e-16 of truth. Resolution: **keep 1e-9 and republish the fixtures at 10 dp**; the reviewer
  computed the full-precision ten-bound table into the review for the fold-in.
- **Sequencing decision: the two document fixes go first, alone.** The implementer must read the
  note's republished fixtures to assert against them, so U6a depends on U6b's output — and
  dispatching the code fix alongside doc edits would recreate the read-write race this coordination
  has already been bitten by twice.

- **2026-09-03 — U6b: the note's author *declined* a review finding, with numbers, and was right.**
  The reviewer asked for `58.3 → 58.4`; the author showed that rounding a **floor** up makes its own
  printed sentence false — at n=12, α=0.025 the exact floor is `7/12 = 58.333` pp and outcomes are
  attainable only at multiples of `1/12`, so `58.4` puts an attainable **significant** outcome below
  the printed floor and the report then contradicts itself. It conceded the reviewer had found
  something real but misidentified the cells: one α column was ceiling-rounding while the other
  truncated, and **three cells are corrected in the opposite direction** (15.8→15.7, 7.1→7.0,
  46.7→46.6) — at n=38 the *existing* `15.8` was already making the false claim. Generalised as:
  **round each printed bound in the direction that keeps its own claim true.**
- **Rule 7 adopted into the note**, with two refinements that decide whether it works: compare
  against the **exact** floor, never the display-rounded one (or the invariant inherits the
  presentation layer's rounding), and **the converse is not an invariant** — above-floor does not
  imply distinguishable. Routed to U6d for adjudication rather than settled on either author's
  authority; both asked for that.
- **2026-09-03 — U6c: the `architect` removed the enumeration rather than repairing it.**
  `FORBIDDEN_BY_ARM_KIND` is now **derived** (`required(other kind) − required(this kind)`) instead
  of hand-listed, with §3.4.2 declared the owning section — and doing so exposed that §3.4.2 was
  itself missing `modelCapabilitiesPresent`, which the derivation would have inherited as a hole.
  **Two review passes had certified the fourteen-name list** (Pass 2 says "enumerates all fourteen
  model fields") **because each read it against its own adjacent prose rather than against the set
  it complements** — a blind spot no amount of re-reading the same way would have closed.
- **The v1.4 pairing rules could not have prevented this drift, and the `architect` said so plainly:**
  rules 1–3 govern *plan↔note* disagreements, while both defects were **intra-document** (an
  enumeration vs. its own prose; an appendix vs. §3.3). New **rule 4**: appendices, recap tables and
  enumerations are **derived surfaces**, the owning section wins by construction, a change to an
  owning section sweeps its derived surfaces in the same pass, and **where a derived surface can be
  a derivation, it must be.**
- **Flagged, not fixed (carried):** §5's numbered test list is not stage-scoped and nowhere says so
  (both gates had to reason it out); §3.4.2's tier lists are still illustrative; and
  `PackRef.contentHash` is always `""` at S1 while Appendix A still calls it part of the identity
  triple — if the code fix makes it `str | None`, Appendix A must be swept in the same pass, which
  is precisely what rule 4 exists to prevent.

- **2026-09-03 — U6d: the reviewer ran the tie-breaker and ruled against itself.** A counterexample
  exists for **every** cell it had asked to ceiling, and it is always the same one — the exact floor
  itself. The general statement is stronger than the note author's framing: **the floor is an
  *attained* bound.** `b = b_min, c = 0` is always realisable and always reaches α by construction,
  so `b_min/n` is not merely a threshold below which nothing fires — it is an outcome that fires.
  Ceiling therefore has **no correct case**, rather than being the wrong trade-off. Truncation and
  the three corrected cells upheld; `58.3` and `23.3` stand.
- **The principle sharpened, and the sharpening matters:** direction is set by **which side of the
  bound can falsify the sentence it appears in** — not by conservatism. Up for MDD (power increases
  in δ, so rounding down under-delivers the promised 80%), down for the floor. That the two coincide
  with the conservative direction here is *a coincidence, not a theorem*, and a future printed bound
  may not oblige. The operative acceptance test is the **tie-breaker itself** — *is any attainable
  `k/n` a counterexample to the printed sentence?* — which is decidable, cheap, and catches a
  mis-signed rounding rule that the rounding rule cannot.
- **The adjudication also caught that teco's dispatch instruction was incomplete.** The named test
  carries only one of the three corrected cells; the other two live in a **second file**, and
  dispatching as stated would have landed the α=0.05 fix red. It added two further conditions:
  change the assertion *mechanism* rather than the literals (re-rounding inside the test reproduces
  the defect being fixed), and truncate **at print only**, leaving the field exact so Rule 7's guard
  is not weakened. **Routing a fix instruction back through the specialist who raised it caught an
  error in the instruction itself.**
- **Rule 7's converse confirmed non-invariant with a fixture already in the suite:** `(20, 8, 2, 10)`
  — n=40, `|diff| = 15.0 pp` exactly at the α=0.05 floor, `p = 0.109375`, not distinguishable.
  Significance depends on the discordance **split**, not on `b − c`. Asserting the converse would
  have failed against an existing fixture.

- **2026-09-03 — U6a delivered: 233 → 296 tests, both blockers closed, 34 mutations run with
  **zero** survivors** — including all ten the gate had left alive. Two of its *own* new mutations
  initially survived, both for exactly the reason the gate's M-4 names: **a test asserting a
  passthrough field rather than the behaviour it gates.** Both were rewritten onto fixtures where
  the distinction bites (DEFF 1.335, where the exact floor is 20.025 pp and the printed one 20.0, so
  an observed 20.0 pp sits in the gap).
- **B-ML-1 fixed as the *minimal* fix, declared as such, with the reason.** The structurally right
  primitive resamples clusters of paired differences, but `PairedOutcomes` carries one row per
  analysis unit and the grouping could only come from a pack declaring `replicatesPerScript > 1` —
  which **Rule 6 makes a validation error** while only the one-level bootstrap exists. Building it
  now would have had no data to consume and no seam to reach it. The rendered effect is the point:
  the CI that was *identical* at DEFF 2, 4 and 7 now widens ([0.9, 32.7] → [−5.0, 40.0] →
  [−11.5, 48.1]) and the DEFF-7 line stops claiming "resolves ≥100.0 pp with 80% power".
- **B-1's fix required reading the output, not just fixing the decision.** `compare_report` now runs
  **two passes** (Holm is a property of the family, so no verdict can be decided until every p-value
  exists), and the family table gained a `decision` column — because a reader applying a printed
  threshold still reaches the opposite conclusion unless the table states the outcome. **Public API
  change for S2: `holm_thresholds` is gone, replaced by `holm_steps`; `verdict()` gained
  `holm_tested`.**
- **The implementer found a factual error in teco's brief, and it was load-bearing.** The brief
  relayed a sweep claiming naive truncation never misfires; `7/40` is `174.99999999999997` bins in
  IEEE doubles, so naive truncation prints **17.4** where `-ml` §7.1 publishes **17.5**. The
  adjudicator's sweep had covered only the α=0.05 column. **Teco's own first verification appeared
  to refute the finding — because it computed in percentage points while the code computes in
  proportions.** Re-checking in the code's own units confirmed the implementer. *Verify in the units
  the code uses, not the units the document prints.*
- **A methodology gap neither the note nor either review had reached, raised by the implementer:**
  Rule 7's floor is computed at **α/k** while Holm's actual step for a rank-*i* member is the looser
  **α/(k−i)**, and they disagree at the margin (b=6, c=0 at n=40 clears a Holm step of 0.05 while
  its 15.0 pp sits below the printed 17.5 pp floor). Resolved **conservatively** — the decision
  follows the floor the report *prints*, so a verdict can never contradict the honesty line beside
  it — with the forgone Holm gain documented in a named test. Routed to `data-scientist` for a
  clause in §3.3/§7.1.

- **2026-09-03 — U7a: `approve with suggestions`, 0 blockers, all 18 Pass 1 findings fixed.** The
  reviewer re-ran its own Pass 1 mutation set — **10/10 now killed** — then ran **24 fresh
  mutations on the new code, of which 3 survived**. That ratio is the argument for re-gating a fix
  round at all: a clean fix of every named finding still left three untested paths behind it.
- **M-4's class is closed generally, not just where it bit.** The reviewer verified the pinned
  literals are genuinely independent (plain dicts, not derived from the module) and reconcile with
  the plan's own 26 + 4 = 30 enumeration, then tried **a mutation neither side had thought of** —
  *relaxing* a tier (`nonempty`→`present`) rather than deleting a field. Killed by 6 tests.
- **New major P2-1 — a gap that becomes live in the very next stage.** Widening
  `mcnemar_may_decide` to admit `basis == "measured"` into the McNemar seat **survives all 296
  tests**: `"assumed"` is covered at Rule 4's branch, the third enum value is not. `"measured"`
  became reachable *in this commit*, and **S2's runner is what will start producing it** — so this
  is fixed before S2, not after.
- **The `contentHash` seam: the code is right and the plan should follow.** `""` is
  indistinguishable from "a hash was computed and came back empty" in the one field whose job is
  identity; Appendix A's identity triple describes a *loaded* pack (S2's `Pack`), while `PackRef` at
  S1 has no hash to carry, because the AC-3 banner reads each run's own
  `fingerprint.packContentHash`. Suggested wording is in the review for U7c's sweep.
- **Routing note:** `analyst` is now at ~281k cumulative tokens. The next engineering gate is a
  **fresh** dispatch; this reviewer's reasoning is fully written into Pass 1 and Pass 2.

- **2026-09-03 — U7b: `needs changes`, and the new blocker is in the corner nobody parameterised.**
  **B-ML-2:** at DEFF = 1.0 with `basis="assumed"` — which is **every comparison until S2 lands the
  determinism probe** — the decision leaves McNemar and `√1 = 1` widens nothing, so a bare
  percentile interval decides. Measured at n=40: `b=7,c=1` (p=0.070), `b=9,c=2` (p=0.065),
  `b=11,c=3` (p=0.057) all render **distinguishable** where the exact test refuses. Rule 7 misses
  them (all at or above the floor) and the new width test parameterises DEFF over {2,4,7} only.
  Fix: on any non-`by-construction` path the decision becomes a **conjunction** — widened CI
  excludes zero **and** `mcnemar_exact ≤ alpha_step`. Using McNemar as a *veto* does not violate
  Rule 4, whose objection is that it **rejects** too readily; a necessary condition only removes
  rejections.
- **The √DEFF interim is accepted, and the deferral argument endorsed.** The conversion is exact in
  the sense claimed (Kish's DEFF is a variance ratio, half-width scales as `1/√n`), checked against
  a hand computation to 1e-12. The reviewer would **not** build the structural primitive before a
  pack needs it: *the event that falsifies the interim is the same event that unlocks the real one.*
- **Rule 7 is right on one path and wrong on the other.** Demote-and-name is correct on the
  substitute path, but on the McNemar branch the invariant is a **theorem** — re-confirmed
  exhaustively, zero violations over six *n* — so a fire there is a module bug, and demoting
  silently discards the detector property Rule 7's own docstring claims. Split by path: demote on
  `cluster-bootstrap`, **raise** on `mcnemar-exact`.
- **The α gap: the reviewer contests the shipped resolution, by the same principle that settled the
  rounding.** The printed floor's sentence is true only at the **loosest** step a member can face
  (α=0.05 → `6/n`); printed at α/k it is `7/n` and **false** — b=6,c=0 at n=40 is p=0.031, reaches
  significance at a 0.05 step, and sits below the printed 17.5 pp. *Identical falsity class to the
  `15.8` withdrawn in Pass 1.* The shipped conservative choice also reduces Holm to Bonferroni in
  `[6/n, 7/n)` — **charging twice** the price §7.3 already books for a second verdict metric.
- **The generalisation now covers three rulings at once:** *every printed bound takes the rounding
  direction, the α **and** the denominator that keep its own claim true.* The Pass 1 rounding
  ruling, this α ruling and the declined n-ML-1 are three instances of one rule, and the note is to
  state it once rather than three times.
- **The reviewer corrected its own sweep again, and more precisely than teco had:** the miss was not
  "pp versus proportions" but that it swept `math.floor(x*1000)` where the code computes
  `math.floor(x/precision)`. For `7/40`, `x*1000 == 175.0` exactly while `x/0.001 ==
  174.99999999999997`. Re-run against the code's own expression, naive truncation misfires at
  **exactly three points under n ≤ 1000 — n = 10, 20, 40 at α=0.025** — and n=40 is both a published
  §7.1 row and §7.3's `clear_suspend` slice. **The pinning test must keep using that exact
  expression, not an equivalent-looking one.**

- **2026-09-03 — U8b: the `architect` pre-empted the next divergence instead of creating it.**
  Rather than leave three α/k restatements that note v1.6 was about to contradict, it **withdrew**
  them — §3.3(ii), §3.8.2 and §3.9 point 3 now *cite* the note instead of naming an α, and the plan
  states nowhere what the new rule is, keeping only the plan-owned half (correction mandatory,
  threshold printed beside every p-value, **the rule applied must equal the rule printed**). That is
  §7 rules 1–2 *executed* rather than described: the alternative was shipping v1.6 with clauses its
  paired note contradicts on arrival.
- **The `contentHash` judgement was accepted with a reason the gate left implicit:** the divergence
  is not new — S2's `Pack` has carried a total `contentHash: str` since v1.1 while `PackRef` is what
  `compare_report` takes — v1.5 simply had not named it. `None` vs `""` is **the plan's own
  absent-versus-empty rule (§3.4.2) applied one level up**. The stricter alternative (drop the field
  from `PackRef` so `None` is unrepresentable) was **considered and rejected**: it would make the
  report's parameter type depend on whether a pack happens to be loaded, for a field the report is
  *forbidden* to read. Three rules now bound it, including a named totality boundary
  (`Pack.ref()`) — *without one, S2 would have invented one.*
- **A live code-side restatement found that the code fix must check:** `PackMetrics.alpha`
  (`0.05 / k`) in `modelbench/packs.py` is the code-side counterpart of the three clauses just
  withdrawn. **If the note's v1.6 moves the floor to the unadjusted α, whoever implements it must
  sweep `packs.py`, not only `stats.py`.** Carried into U8c's brief.
- **Carried a third time, still unfixed:** §5's numbered test list is not stage-scoped, and **both
  gates have now had to reason it out independently**. A recurring cost that a one-line preface
  would remove — logged rather than chased.

- **2026-09-03 — U8a: the α ruling accepted, and a whole column of the note **deleted**.** §7.1's
  second floor column (58.3 / 46.6 / 23.3 / 17.5 at α=0.025) is gone — *it asserted impossibilities
  that are attainable*. One floor column remains, at the unadjusted α, with the reason stated: the
  floor does not move with *k*; **only the MDD pays for multiplicity.** `ResolvingPower` now carries
  three αs — `alpha_family` (floor), `alpha_mdd` (α/k), `alpha_step` (α/(k−i), known only after
  ranking).
- **The generalisation is now Rule 3, and it resolves as a 2×3 table:** MDD rounds **up** / takes
  the **tightest** α / uses the **floored** `n_eff`; the observable floor rounds **down** / takes the
  **loosest** α / uses the **unfloored** `n_eff`. Each cell is conservative *for its own sentence*,
  and unifying any row makes one bound anti-conservative — which is why n-ML-1 was correctly
  **declined rather than harmonised**.
- **The Rule 7 theorem was re-derived exhaustively before the split was written** — all `(b,c)` with
  `b+c ≤ 400`, zero violations at either α. The note also records *why the split depends on the α
  ruling*: at α/k the McNemar branch is reachable, so `raise` would fire on correct data.
- **The guard's justification did not survive its own pass — the third instance of this pattern.**
  The bin-edge guard was load-bearing on `17.5 pp` at n=40, **which is precisely the cell the α
  ruling deleted**. Swept to n ≤ 2000: with `b_min = 7` naive truncation misfires at n = 5, 10, 20,
  40; with `b_min = 6` — which v1.6's floor always uses — **it never misfires**. The guard is kept
  and re-justified (`b_min` is a function of α, so a future α reopens the hazard, and it costs one
  expression) but is now **defensive, not a regression pin on a published value**. The test's own
  comment must be corrected or it will document a cell that no longer exists.
- **Pattern worth naming, since it has now bitten three times:** *a justification computed against a
  state that the same pass changed.* Each time it was caught only because someone re-derived rather
  than re-read.

### U8c accepted — and the units trap caught me a second time

- **Verified independently before committing** (`95b4c88`): `314 passed`, `ruff` clean, and `-m ""`
  / `-rsx` confirm nothing is skipped, deselected or xfailed. The α split, `stats.ALPHA_FAMILY` as
  the single home of the unadjusted 0.05, `Rule7Violation`, the McNemar veto and the `{"", ".."}`
  guard were each read in the source rather than taken from the report.
- **I re-ran the truncation sweep in percentage points and it appeared to refute the implementer**
  — zero misfires at `b_min = 7`, five at `b_min = 6`, the exact inverse of its claim. Redone in
  **proportions**, the units `format_floor_pp` actually computes in (`x / 0.001`, not `x * 100 /
  0.1`), it reproduces the implementer exactly: `b_min = 7` misfires at n = 5, 10, 20, 40; `b_min =
  6` never, to n = 2000. **Second time this coordination has produced a false refutation from the
  same cause.** The rule is now in both re-gate briefs: *verify in the units the code uses, not the
  units the document prints.* The code's own docstring names the same trap for the `floor(x*1000)`
  form — "a sweep run against the expression the code does not use is how this was missed the first
  time" — which is the same failure at a different layer.
- **`Path("..").name == ".."`.** The implementer rejected one third of finding P2-4 on this, and it
  is right: dropping `".."` from the guard would write `results/runs/..json`. Confirmed directly.
  Both re-gate briefs carry the correction so the `analyst` records it against its own premise.
- **Three note-side defects were routed to `data-scientist` rather than edited in place** by the
  implementer — correct ownership behaviour, and they are Part 1 of U9a rather than a separate unit,
  since the same agent must gate the code against whatever the note becomes.

### Pass 3: both gates fail the build, and my parallel dispatch collided outside the repo

- **Both gates returned `needs changes` independently, and they found the same major independently
  too** — `_mdd_clause` asserting *"the observed X pp is below that"* without checking. Filed as
  `M-ML-7` by statistics and `P3-2` by engineering, from different evidence (1,580 clause-printing
  tables vs. 5,525 else-branch tables at two pack sizes). Independent agreement is stronger evidence
  than either alone. I reproduced it myself before accepting: at n=20, b=13, c=5 the harness renders
  *"resolves differences of >=36.7 pp with 80% power … ; the observed 40.0 pp is below that."*
- **P3-1 is the first Pass-3 blocker and the worst defect this component has produced.** Two
  defaults compose: `scoreable.get(metric, True)` reads a missing declaration as scoreable, and
  `counts.get(metric, 0) > 0` reads missing data as a failed item — so an arm with **no data at
  all** is scored as losing every item, rendering `+100.0 pp … p=0.002`, while §4.3's tally, whose
  only job is to make dropped rows visible, prints `0 unscoreable in both`. Confirmed in the source.
  **S2's scorers are what will emit those mappings**, so the fix is a contract, not a patch.
- **Four of this component's defects now live in what the instrument *says*, not in what it
  computes** (CI orientation, Holm printed-not-applied, the "best case" caveat, and now the MDD
  clause) — plus `P3-3`'s *"widened by sqrt(DEFF)=1.00 for the declared clustering"* on the path
  every comparison currently takes, and `P3-4`'s unlabelled `--negative-control` report. Every one
  was found by **rendering output and reading it**, never by reading assertions.
- **A scratchpad collision I caused.** The two parallel gates shared a session-scoped temp
  directory, and one reviewer's mutation driver was **overwritten mid-pass by the other's file of
  the same name**, so it ran the wrong mutation list under its own invocation. It caught this,
  rebuilt a sandbox from `git archive 95b4c88`, and reproduced every survivor — but the recovery was
  its doing, not my dispatch's. **My serialize-on-shared-file rule was scoped to repo paths;** the
  sharing axis here was a temp path outside the tree, invisible to any diff-based check. Fix
  applied to the fix-round brief: scratch files must carry a unique suffix and no agent may assume
  a temp path is its own.
- **A reviewer recorded that its own earlier premise was wrong** — Pass 2's P2-4 claimed three
  redundant entries; only one was. That correction is now in the review rather than only in my
  ledger, which is where it belongs.

### U10 accepted — and the sweep found five more restatements than the finding named

- **The finding was one stale α sketch; the sweep withdrew six restatements.** The `z` literal
  `1.959963984540054` printed in full, the verdict string quoted as *the phrase to assert* in four
  places at a different capitalisation from the note's own rendered sentence, the judge gate's
  thresholds and both κ figures, §6 R-9's worked-case numbers — and **the note's own rule count**,
  which the plan gave as six while the note is at seven. That last one is the class in miniature: a
  count is a restatement, and it rots exactly like a constant. This is the third finding traced to
  plan-restates-note, so the fix being *citation* rather than *corrected numbers* is the point.
- **It corrected the reviewer's suggested split, and the correction checks out.** The engineering
  gate put test item 4 in S1; `packs.content_hash` does not exist at S1 — the shipped module has
  `PackRef` / `metrics_from_manifest` / `check_sampling_contract` only, which I verified directly.
  Item 7b moved the other way, into S1.
- **The table declares §4 authoritative where the two disagree.** That is the lesson from N-3 applied
  without being asked: a precedence rule pointing the wrong way is what turned a fixed finding back
  into a live regression at v1.3, and the new table would have been a second copy of the same
  hazard without it. Numbering was deliberately left unchanged so all three reviews' citations by
  number still resolve.
- **Two stale `queued` rows closed as `abandoned`** (U6e, U7c): both were folded into U8a/U8b when
  those units were re-scoped, and neither was ever dispatched. A `queued` row for work that already
  shipped is precisely what makes a resuming session re-spend it.
- **One item left deliberately, and flagged for the gate rather than decided:** κ = 0.21/0.83 stays
  in §2.1's data inventory and §6 R-4, attributed to `-ml` §6.1 — one attributed copy in the plan's
  own inventory versus zero copies. The `architect` declined to rule on its own document and handed
  it to the reviewer, which is the right instinct.

### U11 killed by a platform 500 — recovered by state, not by restart

- **Second platform kill in this coordination** (the first was a 429 during the Pass 2 gate). Same
  handling: a transient platform failure is **not a deficient result**, so nothing is re-briefed
  from scratch. I read the tree before writing the recovery brief — `git diff` showed **+428 lines
  across 6 files**, the suite at **329 passed / 2 failed**, and the two failures were the
  *deliberate RED tests for P3-3* that the agent's last line said it was writing. A clean,
  legible stopping point.
- **The difference between the two kills is worth recording.** The 429 hit before anything was
  written, so recovery cost nothing. This one hit with an uncommitted half-finished fix round in
  the tree — which is exactly the state in which the never-mutate-the-tree rule stops being
  hygiene and starts being the only thing preventing the loss. The recovery brief says so
  explicitly rather than assuming it.
- **The brief tells the successor to trust the tree over the brief.** My summary of what landed
  (`ItemResult.scored_outcome`, the `_NO_PAIRED_DATA` refusal path, `stats.mdd_clause` as a single
  home) is a starting point for it to verify, not an authority — I read a diff, I did not run the
  reasoning that produced it.
- **The interrupted run opened a finding of its own, `m-ML-8`:** the MDD stem was duplicated
  between `report.py` and `stats.py`, and M-ML-7's fix would have edited one copy — *scheduled
  drift*, named as such. It collapsed the copies while fixing the finding. Carried into U11b so it
  does not die with the transcript.
- **The plan moved under the interrupted agent** (v1.6 → v1.7, U10 landing mid-run). The recovery
  brief flags it explicitly, because a resumed agent's most dangerous inheritance is a quotation
  from a document version that no longer exists.

### LM Studio probed live (2026-09-03) — S3's prerequisite is met, and S2 has a design problem

Stakeholder confirmed LM Studio runs with **auto-load (JIT)**, so a request is all that is needed.
Verified directly rather than taken on report:

- **Reachable from WSL at `http://localhost:1234/v1`** — the same base URL `falkor-chat/config/`
  uses. **19 models** in the catalog, including every model that component names
  (`qwen/qwen3-4b-2507`, `mistralai/ministral-3-3b`, `text-embedding-qwen3-embedding-0.6b`).
- **Auto-load confirmed end to end.** A cold `POST /v1/chat/completions` against
  `mistralai/ministral-3-3b` at `temperature: 0` returned the expected content. **It took 21 s**,
  essentially all of it JIT load — a number S2's timeout design must accommodate, since the
  harness's first call to each arm pays it and no per-request timeout tuned to warm latency will
  survive it.
- **`/api/v0/models` (LM Studio's native API) carries the fingerprint metadata the REST v1 endpoint
  does not**: `state`, `quantization`, `arch`, `type`, `max_context_length`, `capabilities`. This is
  a better source for FR-7 host/model fields than anything §3.4.4 currently names.

**The finding that matters, and it is the plan's to answer, not mine:** `lms` is **not on PATH** on
this box. §3.4.4 and §6 R-1 both assume the `lms` CLI exists — `residentModelsAtStart` is specified
as `lms ps --json`, with `[]` on a clean box called out as the correct value. That command cannot
run here. The REST substitute is `/api/v0/models` filtered on `state != "not-loaded"`, which is
strictly richer, but it is a **contract change in the host-info design**, so it goes to `architect`
as part of S2's dispatch rather than being decided by an implementer mid-run. R-1's own claim (no
programmatic source for the app version and the KV-cache setting) is untouched by this and still
stands — if anything the missing CLI reinforces it.

**Net effect on scope:** S3's live-run prerequisite is met and needs nothing further from the
stakeholder. S2 gains one design question to settle before its runner unit is briefed.


### U11b accepted — the fix round closed 21 findings, and the recovery cost nothing

`d55f4d8`. The state-recovery brief worked: U11b resumed from disk rather than restarting, and
the platform 500 that killed U11 cost **no delivered work** — only the wall-clock of one dispatch.

**Verified by me, not accepted on report:** 353 passed / 353 collected, nothing skipped, deselected
or xfailed; `ruff check .` clean; `IncompleteItemRecord`, `scored_outcome`, `_NO_PAIRED_DATA`,
`negative_control` and `PackRef.seed` all present in the source. Two numeric claims re-derived
independently — `resolving_power` does refuse `design_effect` of 0.0/0.5/0.999 and accept 1.0
(n-ML-5), and the `|diff| == mdd80` boundary **is** reachable at k=2, which I swept to n=200 and
found at n = 90, 100, 120 (the round's own three) **plus 125, 150, 180**. Consistent with the
round's claim, which stopped its sweep at 120 — a wider sweep, not a contradiction.

**One thing I found while verifying that neither gate raised.** The comparison is strict, so exact
equality takes the *"is above that"* branch. The branch is right — `mdd_clause` claims the pack
resolves differences **`>=` mdd80**, so at equality the difference is resolvable and the
"not strictly dominant" wording is the correct one. But the sentence then prints *"the observed
10.0 pp is above that"* when it is exactly equal. That is a **published string §3.2e mandates
verbatim**, so it is the note's to change, not the code's: routed to `data-scientist` in U12b's
brief rather than fixed. The recurring shape of this coordination held one more time — the defect,
if it is one, is in what the instrument *says*.

**Carried forward into U12a/U12b as seeded checks rather than trusted claims:** the round's
*equivalent by construction* mutation survivor (m-ML-8's identical-copy form — an equivalent mutant
is the standard hiding place for a real gap); the `--negative-control` banner's claim that the mode
"cannot fail"; and the **one exposure U11b deliberately left open** — the Arms table is driven by
`run.aggregates` rather than the paired rows, so an arm declaring `BinaryMetric(successes=0, n=10)`
for a metric no item declared scoreable still prints `0/10 = 0.000`. The round argues that is the
*arm* misreporting rather than the reporter inferring, that the table is labelled descriptive, and
that the aggregates-vs-items cross-check is S2 scope because S2's scorer produces both. **Plausible,
and exactly the shape of deferral a fresh gate should judge rather than inherit** — so U12a judges
it.

**Also settled here, and recorded in `model-bench/AGENTS.md` as a contract on S2's scorers:**
`ItemResult.scored_outcome` is the only decider of whether a row has an outcome. Absent or `False`
in `scoreable` → no outcome, routed to the §4.3 tally. `True` → a `counts` entry is **mandatory**,
and its absence raises rather than reading as a zero. Every scorer must emit a count for each metric
it declares scoreable, and a manifest must declare `sampling.seed`. S2's dispatch inherits this.

**Dispatch note:** U12a and U12b run in parallel, but each was briefed to build its own sandbox
outside the repo (`git archive d55f4d8 | tar -x` into a uniquely-suffixed `/tmp` dir) and mutate
only there. Pass 3's scratchpad collision — two reviewers writing a mutation driver to the same
session-scoped temp path, one silently overwriting the other — was caused by my own parallel
dispatch, and my serialize-on-shared-file rule could not see it because the shared file was outside
the repo. Sandboxing per reviewer closes that axis without serializing the gates.

**The stakeholder question both gates were asked explicitly:** S1 has had three fix rounds and is
not converging monotonically — Pass 3 found *more* than Pass 2 (1 blocker + 6 majors against Pass
2's 1 new major), because its reviewers ran 86 mutations and read rendered output. Each Pass 4 gate
must state whether its residue **blocks building S2 on this core** or can ride as follow-ups. That
is the input to the decision I owe the human when U12a/U12b land.


### Pass 4: both gates approve, and they converge on the same sequencing without having talked

Neither gate saw the other's work. Both returned **approve with suggestions**, **no blocker**, and
both were asked the stakeholder question directly. They answered it the same way — *the residue does
not block S2* — and, more usefully, they independently named **the same shape of constraint**: land
specific fixes before real scored data exists, not before S2 is dispatched.

- `data-scientist` (M-ML-8): the conservative envelope **must land before the first
  stakeholder-facing comparison is published**, because it changes printed intervals and two reports
  over the same data must not disagree.
- `analyst` (P4-2, P4-4): gate **S3** on those two specifically — they are the nets that catch the
  first scorer's first mistake, and *"adding a net after the thing it protects has shipped is how
  all four passes began."*

Independent agreement on sequencing is much stronger evidence than either verdict alone, and it is
the reason this is a proceed rather than a fourth fix-and-regate cycle.

**Why the count stopped being the signal.** Pass 4's count is flat against Pass 3 (5 majors vs 6),
but the *character* changed and that is what decides it: Pass 3 held a blocker that rendered
`+100.0 pp, p=0.002` against an arm holding no data, plus four sentences that were false of the
numbers beside them. **Nothing in Pass 4 is a wrong number.** Every Pass 4 major is either an
unpinned string — a mutation that survives all 353 tests — or a missing refusal. `analyst` put the
trend correctly: Pass 3 found more than Pass 2 because the *method* changed (render-and-read plus
mutation campaigns), not because the code decayed; Pass 4 applied that same method to a wider
surface and found no blocker. By severity the statistics side is monotone across four passes:
1 blocker + 4 majors → 1 blocker + 1 major → 1 major → 1 major, each one a narrower layer of the
same corner (decision → calibration → quantification).

**Three Pass 4 findings I re-derived myself rather than accepting on report:**

- **P4-2 reproduced exactly.** In a sandbox from `d55f4d8`, `report.py:482`
  `holm_tested=step.tested` → `holm_tested=True` leaves **353 passed**. Pass 1's blocker verbatim,
  with no test net under it.
- **P4-4's overturning of U11b's deferral holds on both grounds.** `_DESCRIPTIVE_NOTE` reads *"Per-arm
  intervals are Wilson score intervals … descriptive, not the comparison instrument"* — it caveats
  the **interval**, and says nothing about the **rate**, which is the thing that misreports. And
  `RunResult` carries `items` **and** `aggregates` as required fields side by side, so the
  cross-check is **S1-local**; S2 owns the scorer contract, not the check. The deferral U11b argued
  for is wrong, and the evidence is structural rather than a matter of taste.
- **M-ML-8's headline case reproduced to the digit**, along with three other statistics figures
  (see the U11b section for the equality boundary and the floor contradiction).

**One sub-claim sent back rather than accepted.** The statistics gate states the bootstrap interval's
last digit moves with row order at a fixed seed. Two constructions of mine left it stable, and the
first provably **cannot** exhibit it (4 ones and 26 zeros row-shuffled is the same multiset, so the
percentiles cannot move). Rather than call it wrong — the failure mode that has already cost this
coordination twice — I asked for the exact reproduction. M-ML-8 does not depend on it; the note's
**published rationale** does, and an unreproducible rationale is what a later pass rediscovers as a
defect. `docs/plans/small-model-benchmarking-ml.md` v1.8 and its review are **held uncommitted**
until that answer lands.


### A stakeholder principle, and it retires a question I keep re-asking

Mid-run on U14, the stakeholder stated it plainly: **defects should not be carried into later
stages, because errors accumulate and multiply.** Standing principle, not a one-off call.

That supersedes the sequencing question I had been putting to them once per gate — *bank the residue
or keep gating?* — and it supersedes my own brief: U14 went out narrow, closing four findings, with
an explicit instruction to leave P4-1, P4-3 and P4-5 alone. I expanded it in place by
`SendMessage` rather than queuing a second unit, because all of it lands in `report.py` and
same-file serialization means it is one agent's work either way.

**U14 is now every major and every minor in both gates' `## Pass 4` sections**, nits at the
implementer's judgement. I told it to sequence the verdict-affecting fixes (P4-2, P4-3, m-ML-12,
M-ML-8) ahead of the CLI-surface ones (P4-1, P4-5) so a partial delivery is still coherent, and to
tell me where to cut rather than thin any individual fix — the one failure mode a scope expansion
mid-run actually invites.

**What this changes going forward.** The "approve with suggestions, residue rides as follow-ups"
disposition — which both Pass 4 gates recommended and which I was ready to act on — is **not
available in this project**. A gate verdict of *approve with suggestions* now means *close the
suggestions*, not *proceed and track them*. Brief future gates accordingly: their job is to find and
severity-rank, and the severity ranking drives **order within a fix unit**, not whether a finding
gets fixed at all. Four passes of evidence support the stakeholder here — P4-2 is Pass 1's blocker
returning **verbatim** after being fixed once, which is precisely what a deferred defect does.


### The architect corrected my brief, and the correction inverted the argument

I briefed U13 with the premise *"`lms` is not on PATH, so `residentModelsAtStart` cannot come from
`lms ps --json`."* The architect re-probed instead of accepting it, and **half of it was wrong**:
`command -v lms` does exit 1, but `/mnt/c/Users/mauri/.lmstudio/bin/lms.exe ps --json` runs fine and
returns `[]`. **I verified both myself**, along with the timing that actually decides it — **0.307 s**
for the CLI against **under 10 ms** for the HTTP call returning the same fact.

So the CLI is not unavailable; it is **bad**, on three measured axes (a globbed Windows path is a
host-layout accident rather than a contract, 175× slower for the same fact, and a second surface to
keep honest beside the HTTP one the adapter already targets). The conclusion I had reached is the
one the plan adopts, but **my stated reason for it would not have survived the first implementer who
typed `lms.exe`** — and an implementer discovering that a plan's premise is false has every reason
to doubt the conclusion too.

This is the *"coordinator brief is the least-reviewed input in the pipeline"* failure with a happy
ending: no gate reads briefs, and an isolated-context delegate has no cheap way to doubt a
coordinator's factual premise. What saved it was that the brief handed over the **raw probe evidence**
alongside the conclusion, which is what made re-probing natural rather than insubordinate. Keep doing
that — state the observation, not just the inference drawn from it.

### Two same-file hazards handled by pinning rather than serializing

Three agents are live and two of them read documents a third may be mid-edit. Rather than serialize
(which would cost a full dispatch each), every reader was pinned to a **commit**, not to `HEAD` and
not to the working tree: U15 reads the note at `git show aebb611:…`, U16 reads the plan at
`git show aebb611:…`.

**`HEAD` was not good enough, and U14 is why.** Its brief said *"read the plan as
`git show HEAD:docs/plans/small-model-benchmarking.md`"* — correct when written, when `HEAD` was
`27501c9` and the plan was v1.7. `HEAD` has since moved to `aebb611` and that same command now yields
**v1.8**, a different document. A relative ref in a brief is a **dangling reference the moment the
coordination continues**, which is exactly the class of bug the pinned-commit convention exists to
prevent. U14 was re-pinned by `SendMessage` to `aebb611` explicitly.

That message did double duty: v1.8's new **S1 done-condition 10** *specifies* the P4-4 behaviour my
own brief had told U14 to decide for itself (exclude-and-name in the existing
`INVALID RESULTS EXCLUDED` block — because raising reproduces P4-5's shape, the very defect U14 is
fixing two findings over, and per-metric suppression leaves a partially-trusted arm in the
comparison). Left alone, two units in flight would have shipped a plan and an implementation that
disagreed about the same check.


### The plan gate found what four code gates structurally could not

`ff499d5`. **Needs changes** — and the two blockers are both about **S3**, the stakeholder's stated
scope boundary for this pass, which no amount of S1 review could ever have surfaced.

**G3-1: the `embedder` arm cannot produce a storable fingerprint, so S3 is unbuildable as
specified.** `runtimeName`/`runtimeVersion` are `REQUIRED_NONEMPTY`; their only source is the chat
route's `runtime` object; §3.8.1 issues only `POST /api/v0/embeddings`, which carries none of it.
And v1.8's own §3.4.4a — the "no fallback, no default, refuse instead" rule I had just endorsed —
is what makes it fatal rather than merely awkward: `store()` refuses, correctly, and the run cannot
complete. **A correct rule applied to an incompletely-enumerated source set.** The gate's
generalisation is the finding worth keeping: *§3.4.4a is a source-of-truth section governing one of
four sources*, and both blockers are instances of that single gap rather than two coincidences.

**Note what this says about gate placement, because it is the transferable lesson.** Four passes of
S1 code review — 353 tests, hundreds of mutations, two independent reviewers per pass — could not
have found this. It is not in the code; it is in the specification of a stage not yet built, and it
only becomes visible when someone traces a *required field* back to the *call surface* that would
have to supply it. Reviewing the plan for S2 before implementing S2 is what caught it, and the cost
of catching it here rather than in S3 is one document revision instead of a stage rebuilt after a
live run fails at `store()`.

**Two findings landed inside a unit already executing, and were relayed rather than queued.** G3-6
and G3-7 are defects in **DC-10 itself** — the done-condition I had sent U14 two messages earlier as
authoritative:

- **G3-6** — DC-10's selector compares **two disjoint vocabularies**. `BinaryMetric.unit` is a
  denominator noun; `PackRef.analysisUnit` is a `pairingKey` component name (`packs.py:130`
  constrains it to `pairingKey[0]`). `metric.unit == pack.analysisUnit` is **never true**, so a
  cross-check written literally from DC-10 checks nothing and passes. I verified the two
  vocabularies are structurally distinct before relaying.
- **G3-7** — DC-10 counts `scored_outcome(metric) is not None`, but that method **raises** for the
  sibling malformation, producing an uncaught traceback at exit 1. **P4-5's shape, arriving through
  the specification written to fix P4-4** — the same defect class, in the fix for its neighbour, one
  document apart.

Both went to U14 by `SendMessage` within the same turn the gate landed. The rule that a finding
invalidating a still-running sibling's premise gets relayed immediately paid for itself twice here:
U14 was actively writing that check, and a selector that is never true is invisible in a green suite.

**A caution I put in the relay and would put in any like it:** the working tree is being edited by
U14 *while* the gate's line-number citations were taken from a pre-edit state, so I told it to
confirm the predicate against its own working copy rather than trusting a line number, and to check
which exception path actually renders before widening any `except`. Precise citations decay fastest
in exactly the situation where relaying them matters most.

**One claim of mine the gate checked and confirmed, with a correction attached:** the
`lmsCliCommit` → `residencySource` swap really is free (30 fields before and after; `results/` never
existed in any commit) — but the edit list was understated by a fourth site that would have shipped
silently wrong (`conftest.py:39` still declares residency elements as `{modelKey, sizeBytes}` against
§3.4.4a's `{id, state}`, and `REQUIRED_PRESENT` never checks element shape). A verified claim and an
incomplete one, in the same sentence.


### R-13 closed, and the two floors are derived rather than chosen

`460940c`. R-13 had been open across several plan revisions; v1.8 forced it by adding a second input
(the latency sample can be shorter than the item count) and the plan gate added a third (a minimum
surviving-sample floor). All three are answered in one new §11.

**Every figure I could check, I checked, and all of it holds exactly:**

- `0.28*25 == 7.000000000000001`, so `math.ceil` returns rank **8** where **7** is exact — the
  motivation for integer ceiling division is a real float defect, not a stylistic preference.
- `ceil(0.95*X) == X` for every `X <= 19`, with **20** the first divergence. So the identity floor's
  boundary is **derived**: below it, "p95" *is* the maximum, the number is sound and the **label** is
  false. That is this project's signature defect shape, found one document above the code.
- The level floor implies `X >= 11 of 12`, `36 of 38`, `81 of 85`.
- **The plan's own sketch `latency n = 34 of 38` prints no figure at all** (`r95 = 33`,
  `3300 < 3420`). A ruling that contradicts the sketch that prompted it is a ruling that bit.

**The units trap, third variant, and the cheapest one yet.** Checking the `round` half-to-even
finding, I first wrote `0.5*X` and found no tie at all — which would have read as a clean refutation.
The shipped expression is `int(round(pct/100 * (len-1)))`, and against **that** the claim reproduces
exactly: order statistic 3 of 4 (upper middle), 3 of 6 (lower middle). The generalisation is now
three-for-three: **read the shipped expression before re-deriving anything from it.** What made the
correct check obvious on the second attempt was that the delegate had named the exact expression and
the exact measured values — briefs should keep demanding that.

**A seam the ruling exposes rather than creates:** there are **two** `_percentile` implementations,
`results.py:541` and `stats.py:224`, with different signatures and different error behaviour. The
ruling mandating one shared implementation is therefore a real change with a real seam. Two homes for
one definition is the same drift this component has now hit for the note's strings, the plan's
restatements, and now an estimator.

**Four requirements routed to `architect` mid-revision**, the first of which retroactively constrains
a fix already in scope: **G3-5 is load-bearing for §11.5, not adjacent to it.** §11.5 depends on a
timed-out call yielding `latencyMs = None`; under `latencyMs = 120000` the mechanism **inverts** —
the tail biased *high* by a config constant while the published clause claims it is a lower bound.
So a finding filed as "an unspecified disposition" turns out to be one of exactly two admissible
values. Also routed: whole-timing-block nulling (G3-3), G3-4 as fix-anyway-not-blocker (the floors
hold either way at `Y >= 10`, but the cause counts would otherwise report a mechanism that did not
occur), and a **new `latencyMsMax` field** — a consequence, not a preference, because the identity
floor makes `latencyMsP95` `None` for *every* tool-caller run and the long-pole pack would otherwise
carry no tail figure anywhere in the record.

**One judgement call I made rather than escalating:** §11.7's rendering used illustrative millisecond
values, because LM Studio held no resident model and a warm sample needed a load. I ruled republish
with measured numbers. A method note publishing invented latencies — even labelled — is the same
shape as the defect this component exists to refuse, one document up; and one ~21 s JIT load is
exactly what the harness itself will pay. Cheap to fix, and expensive later to re-establish which
numbers were real.


### U14 accepted — and the measurement I authorised reversed a ruling mid-flight

`5878014`. Every Pass 4 finding across both gates, closed. **353 → 389 tests**, 389 collected equals
389 run, nothing skipped or xfailed, `ruff` clean. **48 behaviour-changing mutations, 48 killed** —
plus one deliberate no-op control included so the harness is shown capable of *reporting* a survivor
rather than only ever printing "killed". That control is the right instinct: a mutation harness that
has never produced a survivor is indistinguishable from one that cannot.

**The two I re-derived myself, both in a sandbox copy of the working tree:**

- **P4-2's mutation now dies.** `holm_tested=True` — the mutation that survived all 353 tests, and
  Pass 1's blocker verbatim — now fails
  `test_a_metric_past_the_holm_stop_never_carries_a_significance_claim`. Four passes to pin a defect
  that was fixed once and left unguarded.
- **DC-10's literal selector fails a test.** Mutating the corrected predicate back to
  `metric.unit == pack.analysisUnit` fails `test_no_wilson_interval_is_printed_over_a_turn_pooled_count`,
  so the corrected check is demonstrably **not vacuous** — the property G3-6 warned would be
  invisible in a green suite.

**Two deviations from DC-10, both reported rather than silently taken**, which is the behaviour the
brief asked for and the reason the spec defect surfaced as a finding instead of as a bug: the literal
selector was never implemented (verified disjoint: `unit_kind('guard-judge') == 'item'` against
`analysisUnit == 'itemId'`), and no catch was added to `_cmd_compare` because it would be
**unreachable** — the wide `except` is on `pack_ref_from_manifest`, not on `compare_report`, and the
implementer reproduced the traceback before fixing it. It cited this codebase's own P2-4 principle
back at me: an unreachable guard is worse than none.

**The signature defect, one more time, in the fix round's own work.** Its P4-4 fix left two
individually-true sentences contradicting each other — the block said the arms were *excluded* while
the line below still said *"fewer than two arms were selected … check `--models`"*, sending a scorer
author to their command line for a defect in their record. Found by printing the page. Six passes
now, and **not one of these has ever been found by reading assertions.**

### The §11.7 republish paid for itself twice over

I authorised spending a JIT load to replace §11.7's illustrative millisecond values with measured
ones, on the grounds that a method note publishing invented latencies is the same shape as the defect
the component exists to refuse. It returned two **reversals**, neither of which was the point of the
exercise:

**1. G3-3 resolves the opposite way, and the note reversed its own v1.9 ruling.** Measured:
LM-Studio-side TTFT **excludes** the JIT load — 3 485.6 ms of it that `stats` never sees, against a
warm gap of ±12 ms. So `ttftMs`, prefill and `tokensPerSecond` are **not** contaminated and must not
be nulled; only `latencyMs` is withheld, and the others keep their own denominator. **G3-3's defect
is real and its fix is the denominator, not the nulling.** I had relayed the nulling to `architect`
an hour earlier as a requirement; the reversal went out the moment it landed, before v1.9 of the plan
could bake it in.

**2. The note caught itself publishing a false string, in the act of documenting the rule against
false strings.** v1.9's slot 3 read *"a model load costs about 21 s"* — **my** figure, measured on
`ministral-3-3b`. This cold load measured **3.625 s**, same box, same call surface, different model.
Both real; the string false on most runs that render it. Slot 3 now names a magnitude, never a
figure. **A coordinator-supplied number propagated into a published string and had to be caught by a
delegate** — the least-reviewed-input failure again, and this time nothing in my brief flagged the
figure as provisional.

**Bonus: R-14's acknowledged residual is closable.** `latencyMs − (ttft + generation_time)` isolates
the load at **3 485.6 ms cold against ±12 ms warm — 461×**. §11.5.1 proposes withholding at a
**1 000 ms** gap, named as a starting value on two cold observations rather than a derived constant,
with the two quantities that would move it. Additive to the residency probe: the probe sees a reload
*before* an item, the gap sees one *inside* it.

### The fork U14 returned, and why it was not mine to answer

v1.8 §3.4 Rule 4's **closed-form percentile** is unimplemented. Implementing it **retires P3-5's
delivered contract** — `verdict`'s `bootstrap_seed`, `PackRef.seed`'s justification, the
`decided by: … (seed N)` line and four tests — because nothing would be decided by a resample. That
is a public signature S2 wires against *and* a prior gate's accepted deliverable, so U14 took the
reversible branch and returned the fork. Correct call.

The stakeholder principle answers **whether** (a residual seed-dependent printed bound is the same
defect as M-ML-8, smaller — the implementer measured it still moving between −27.1 and −33.3 pp at
`(4,5,3,0)`), so what remains is statistical and contractual, not a scope question: U18 rules on it.
Dispatched **fresh** rather than resuming Rule 4's author, whose context is at ~244k tokens — over
the threshold where continuing buys less than a self-contained brief costs.


### Plan v1.9: the general fix, and one residual that is genuinely blocked rather than deferred

`81a3ef7`. All 13 Pass 3 findings closed, plus R-13's four follow-ups and n-ML-9.

**The blockers were fixed at the general case, and the general case was bigger than the gate's.**
Pass 3 diagnosed §3.4.4a as *"a source-of-truth section governing one of four sources"*. v1.9 accepts
that and finds **five**, publishing the table (13 + 2 + 2 + 4 + 9 = **30**, arithmetic verified),
stating explicitly that the converse of the refusal rule is false, and pinning a ten-step capture
order — with the staleness trip-wire at step 6, argued as *the first instant its comparands exist and
the last instant before an item is consumed*. A reviewer's diagnosis extended by the author rather
than merely applied.

**G3-1 closes without anyone typing the four fields.** The required set becomes a function of
`callSurface` as well as `armKind` (`armProfile` ∈ {`model:chat`, `model:embeddings`,
`deterministic`}), and §3.4.1's forbidden-derivation rule generalises from a pairwise difference to
*union-of-others minus mine* — so `model:embeddings` **derives** as `model:chat` minus its four
fields. A hand-typed list is a fifth place to drift; a derivation is not.

**A deviation worth recording as precedent.** The gate proposed renumbering test 15b to 11d/12c;
the architect **kept the number and moved the test**, on the grounds that three reviews cite `15b`
and citation stability is a hard constraint here. It identified that the actual defect was a `live`
marker acquired **by adjacency** rather than by declaration — so placement plus a one-line pointer
fixes the defect while renumbering would have broken working citations to fix nothing. Correct, and
the general form is worth keeping: *when a fix and a constraint collide, check whether the fix is
addressing the real defect or its position.*

**One residual, disclosed rather than implied fixed.** Turn- and call-pooled `BinaryMetric`s fall
outside DC-10's selector, so **P4-4's defect stays printable for a pooled metric** — a number with no
interval. The architect argues closing it needs a scorer-side declaration of a pooled denominator's
provenance, which is unscoped S2 work. Under the stakeholder principle this is the distinction that
matters: **blocked on unbuilt work is not the same as deferred by choice**, and I have asked U19 to
rule on which it is rather than accepting the framing.

**New S1 work the plan created, queued as U20.** §3.4.2's edit list grew to four sites, and the
structural fix for the fourth is a **new element-shape assertion in S1 DC-1** — not the fixture edit
— because `REQUIRED_PRESENT` never checks element shape. I confirmed `tests/conftest.py:40` still
declares `{"modelKey": …, "sizeBytes": …}` against §3.4.4a's `{id, state}`, so this is specified and
unimplemented. A plan revision that lands new S1 done-conditions is easy to lose between stages;
it gets a ledger row rather than a mention.

**And every load-cost figure is out of the design's sizing.** §2.5 keeps 21.068 s as *one model's
dated measurement* with the 3.625 s counter-example beside it, and test 15's *"of the order of 21 s"*
became a magnitude assertion — which, as the architect put it, would have been false for the next
model. My own measurement, correctly demoted from a constant to an observation, two documents from
where I first stated it.


## PAUSED (superseded — see the section below) — 2026-09-03, at `4d99504`

Stakeholder called a pause. **Everything I verified is committed; nothing of mine is outstanding.**
This section is the resume point — read it before the ledger, then reconcile the ledger against
`git log` per step 1.

### State of the tree at the pause

- **Committed and verified:** S1 is green at `5878014` — **389 tests**, 389 collected equals 389 run,
  nothing skipped or xfailed, `ruff` clean, 48 mutations all killed. Plan at **v1.9** (`81a3ef7`),
  method note at **v1.10** (`69256a2`), both gates' Pass 4 reviews committed.
- **Uncommitted, and deliberately so:** `docs/plans/small-model-benchmarking-ml.md` shows **v1.11**
  with ~285 added lines. That is **U18 mid-write**, not my work left behind — the agent had not
  returned when the pause landed, so nothing in it has been declared finished by its author or
  verified by me. **Do not commit it on sight.** Either wait for U18's result, or, if the agent is
  gone, treat the file as a state-recovery input: read it, verify its claims independently, and only
  then commit it with an honest message saying it was recovered from disk rather than delivered.
- **Not mine, never touch:** `falkor-chat/server/**` belongs to a second, unrelated session that has
  been committing to this repo throughout (`salesperson-ui`). Two of its commits are interleaved with
  mine in `git log`; that is expected, not a problem to clean up.

### In flight when the pause landed

| Unit | Agent id | What it owes |
|---|---|---|
| **U18** | `ae4bd1d99239907a6` | Ruling on §3.4 Rule 4's unimplemented closed-form half: mandatory or not; what replaces P3-5's seed contract if so; whether the bound-by-bound reading of *"the wider of"* is right; whether any published string changes. Note v1.11 is its working state. |
| **U19** | `a84c263e5998ba953` | `## Pass 4` of the plan review (`docs/reviews/small-model-benchmarking.md`), re-gating plan v1.9. Must state explicitly **whether S2 can be dispatched from this plan**, and must rule on whether G3-6's pooled-metric residual is *blocked on unbuilt work* or *deferred by choice*. |

Both are addressable by `SendMessage` at those ids; a completion notification for either may still
arrive. **Neither has been verified, so neither may be reported as done.**

### Queued, not dispatched

- **U20** — the residency element-shape assertion. Plan v1.9 created this as a **new S1
  done-condition** (S1 DC-1) and it is specified but unimplemented: `tests/conftest.py:40` still
  declares `{"modelKey": …, "sizeBytes": …}` against §3.4.4a's `{id, state}`, and `REQUIRED_PRESENT`
  never checks element shape. I confirmed this directly.
- **The Rule 4 closed-form implementation**, if U18 rules it mandatory. Its cost is entirely in
  rewriting P3-5's delivered seed contract — a public signature S2 wires against.

### Where the work resumes

**S2 is not dispatched and must not be** until U19 says the plan can carry it. The stakeholder's
scope boundary for this pass is unchanged: **through S3, the first real end-to-end run**, then check
back. S3's external prerequisite is met — LM Studio is reachable, auto-load is on, and the fingerprint
source question that blocked it is settled at plan v1.9.

**The standing principle that governs whatever comes next:** defects are not carried into later
stages. *Approve with suggestions, residue rides as follow-ups* is not an available disposition —
severity ranks order within a fix unit, never whether a finding is fixed. A gate returning a residual
must be asked which kind it is, because "blocked on unbuilt work" and "deferred by choice" are
treated differently and only the first is acceptable.


## STOPPED CLEAN — 2026-09-03, at `bb0cacf` · **this is the resume point**

Supersedes the pause record above, which was written before both in-flight agents were killed by a
session rate limit. **Tree is clean, everything is committed, S1 is green (389 passed, `ruff` clean,
re-verified at the stop).** The only uncommitted files in the repo belong to the *other* session
(`falkor-chat/server/**`) and are not ours to touch.

### What happened to the two in-flight units

Both were killed by the session limit **after writing their deliverables but before returning them**.
Their final emitted lines were *"All checks complete. Writing Pass 4"* and *"Now I'll write the
ruling into the note"* — both **understated their own progress**: the documents were already on disk
and structurally complete. Recovered by state, committed with the recovery stated in the commit
message.

**The lesson, and it is the second time this coordination has hit it:** a killed agent's last words
describe what it was *about to do*, not what it had *done*. Assess the disk before believing the
transcript — and before re-dispatching anything, which would have thrown away two substantial
documents here.

### The two recovered documents — committed but NOT accepted

| Unit | Commit | State |
|---|---|---|
| **U18** — Rule 4 closed-form ruling | `a5f42f6` | note **v1.11**. Structurally complete; **headline claim verified by me**, rest unverified. |
| **U19** — plan gate Pass 4 | `bb0cacf` | review `## Pass 4`, verdict **needs changes** (3 blockers, 5 majors, 5 minors, 1 nit). **Nothing verified.** |

**Neither is an accepted deliverable.** Their authors never declared them finished, so the normal
"delivered → verified → accepted" chain is broken at the first link. A fresh session must verify
before acting, and must not treat the committed state as a gate having passed.

**The one thing I did verify, because it is decisive:** U18's ruling that Rule 4's closed form is
**binding**, on the grounds that the envelope's seed dependence reaches the **verdict** rather than
just the printed digits. At `(a=1, b=25, c=12, d=2)`, n=40, DEFF=1.2 I reproduced it directly against
the shipped `stats.conservative_envelope`: **86/64 across 150 seeds** and **92/58 across 150 row
permutations at one fixed seed**. The note reports 80/70 and 85/65; the difference is `B` (I ran
2 000), and the finding reproduces exactly in kind. **A seed decides whether the tool says
*distinguishable* or *not distinguishable*.** That is M-ML-8's defect class escalated from the digits
to the conclusion, and it settles the fork U14 returned: the closed form is not optional.

### Where a fresh session picks up

**S2 must not be dispatched.** The plan gate returned **needs changes** on v1.9, so the plan cannot
yet carry S2. That is the gating decision and it has not been satisfied.

Ordered, with dependencies:

1. **Verify the two recovered documents** — re-gate or read them critically. Everything below assumes
   they hold; if U19's blockers dissolve under scrutiny the order changes.
2. **Plan v1.10** (`architect`): close Pass 4's 3 blockers and 5 majors. Includes the finding I can
   independently corroborate — **the two shipped `_percentile` copies** (`results.py`, `stats.py`)
   both use `int(round(p/100*(X−1)))`, precisely the estimator note §11.2 **rejects**, and **neither
   appears in §3.4.2's edit table**. I confirmed both copies and that expression earlier in the
   session. This is the second instance of one pattern: *a plan revision specifying S1 work with an
   incomplete edit list.* Fix the pattern, not the instance.
3. **S1 fix unit** (`tdd-engineer`), which now carries three things and should be one dispatch since
   they share files: **U20**'s residency element-shape assertion (S1 DC-1, specified at plan v1.9,
   still unimplemented — `tests/conftest.py:40` declares `{modelKey, sizeBytes}` against §3.4.4a's
   `{id, state}`); the **single shared `_percentile`** per §11.2; and the **closed-form
   implementation** now that v1.11 makes it binding — which **retires P3-5's delivered seed
   contract** (`verdict`'s `bootstrap_seed`, `PackRef.seed`'s justification, the `decided by: … (seed
   N)` line and four tests). That last one is a public signature S2 wires against, so it needs the
   plan updated in step 2 first.
4. **Re-gate both**, then reconsider S2.

### Standing constraints a fresh session inherits

- **The stakeholder principle: defects are not carried into later stages.** *Approve with
  suggestions, residue rides as follow-ups* is **not** an available disposition. Severity ranks
  order within a fix unit, never whether a finding is fixed. When a gate reports a residual, ask
  which kind it is — **blocked on unbuilt work** is acceptable, **deferred by choice** is not. One
  such residual is open and unruled: G3-6's pooled-metric case, where P4-4's defect stays printable
  for turn- and call-pooled `BinaryMetric`s.
- **Scope boundary: through S3, the first real end-to-end run**, then check back. S3's external
  prerequisite is met — LM Studio reachable, auto-load on, fingerprint source settled.
- **A second session commits to this repo continuously.** Never stage, revert or tidy anything
  outside `model-bench/` and this coordination's documents.
- **Read the shipped expression before re-deriving anything from it.** Three separate confident wrong
  answers in this coordination came from re-deriving in the units a *document* prints rather than the
  units the *code* uses. Twice they were mine.


## RESUMED — 2026-09-06, from `d411ac7`

Fresh session. Reconciled the `STOPPED CLEAN` record above against the tree before acting: working
tree clean, `HEAD` at `d411ac7`, S1 re-run by teco → **389 passed in 5.76 s**. The record held in
every particular.

### Step 1 of the resume order — the two recovered documents are verified and now accepted

Both were committed but unaccepted, their authors having been killed before declaring them
finished. Verified here rather than re-dispatched:

- **U19 (plan gate `## Pass 4`) — complete, not truncated, and unusually well evidenced.** It carries
  its verdict, all fourteen findings, the Pass 3 disposition, what's solid, three open questions, the
  finding-ID note and **Appendix D** (D.1–D.4: a recomputed forbidden-set derivation, the LM Studio
  units citation, a re-probe of `GET /api/v0/models` across 19 models, and a shipped-S1 fact table).
  A killed document does not end with its appendix.
- **Its sharpest blocker, `plan-gate P4-1`, corroborated directly.** Plan lines **1168 and 1169 are
  adjacent rows of the same §3.6 table**: 1168 sources `ttftMs` from `stats.time_to_first_token`
  *directly*, 1169 computes prefill as `1000 × stats.time_to_first_token`. One table treats one field
  as ms and as seconds, one row apart. The finding is real at the plan's own text, independent of the
  LM Studio documentation the reviewer cites for the field being seconds.
- **`P4-2`'s second instance re-confirmed:** both `_percentile` copies still shipped, at
  `modelbench/results.py:573` and `modelbench/stats.py:296`.
- **U18 (note v1.11)** was already verified at the previous stop — the closed form is binding because
  the envelope's seed dependence reaches the *verdict*. Accepted on that record.

### Step 2 — dispatched, in parallel on disjoint files

**U22** (`architect`, fresh — the U17 architect is at 301k tokens and this work is self-contained)
carries plan v1.10: all fourteen findings, note v1.11's binding closed form and the seed contract it
retires, the `plan-gate`/`impl-gate` P4 citation split, and — explicitly — **the pattern behind
`P4-2` rather than its instance**, this being the second plan revision to specify S1 work with an
incomplete edit list.

**U21** (`data-scientist`, fresh) carries the three questions Pass 4 routed rather than decided: the
gap detector's right-censoring property against §11.5's two-producer exactness argument, P4-11's
threshold margin, and P4-13's §11.7 denominator on an embeddings run.

Each is told the other is live and that the other's document is not theirs to touch — the recurring
restatement finding is exactly what cross-editing produces here.

### Still true, and still binding

- **S2 must not be dispatched.** The plan gate is at *needs changes*; the plan cannot yet carry S2.
- **The S1 fix unit is step 3, not now.** It bundles U20's residency element-shape assertion, the
  single shared `_percentile`, and the closed-form implementation — that last one a public signature
  S2 wires against, so it waits on plan v1.10.
- **Defects do not ride to later stages**, and a disclosed residual must be named *blocked on unbuilt
  work* or *deferred by choice*. Both briefs carry it.
- **FalkorDB is unreachable this session** (`host.docker.internal:6379`). No CPG exists for
  `model-bench` anyway, and every unit here is offline work — but kaizen writes will fail.

### U21 delivered — 2026-09-06, note v1.12 (`fc2fcf6`)

All three routed questions changed the note; none resolved as a no-op.

- **Q1 — the exactness argument fails, and the failure is not marginal.** §11.5's *"every withheld
  call was slower than every timed call"* does not extend to §11.5.1's detector, which withholds on
  `unexplainedMs` — a **covariate** — never on the wall clock, so nothing orders a withheld call
  against a timed one. It needs only one timed call slower than the threshold, which at plan §2.2's
  ~1.3 s pack turns is **the ordinary case of the very pack that fires it**. The fix has the right
  shape: `censoringExact` becomes a **computed per-render flag** with §11.7 slot 3 selecting on it,
  so the claim is checkable at runtime rather than argued — the pattern this coordination has needed
  repeatedly. The gate's Open question 2 is closed.
- **Q2 — the 1 000 ms threshold survives; its second margin is withdrawn.** P4-11 was right: the two
  cold loads are n=1 each and differ in model, quantization *and* route, so they bound no load from
  below and *"~3.5× below the smallest cold load"* was unsupportable. Rebased on **asymmetric error
  costs** — a false positive is discrete and severe (three of them take a Y=38 run below §11.6's
  floor, so no latency summary at all), a false negative continuous and bounded by the threshold
  itself. The re-check is scheduled against a field the plan already stores, so it is not a deferral.
- **Q3 — suppression confirmed, condition corrected** to gate on the **call surface**, not the arm
  profile: a `deterministic` arm returns no `stats` either, so a profile-shaped condition misses it.

**§11.6 does not move** — the attained-level bound re-derives without the ordering assumption (it
needs only *subset*), so the floor, its 5-point constant and §11.5's table are untouched. Blast
radius on the plan is two asks, not a redesign.

**Two new plan asks, both non-blocking, relayed to the architect while still in flight** rather than
held — §11.9 **2b** (a withheld item's wall clock must stay readable somewhere other than
`latencyMs`; the fields `plan-gate P4-3` already adds may reconstruct it, so the architect may owe no
new field) and §11.9 **5** (`statsCoveredCount` is `None`, never `0`). 2b interacts directly with the
P4-3 closure U22 is writing now, which is why it could not wait.

Committed by explicit path — `docs/plans/small-model-benchmarking.md` is dirty with U22's in-flight
work and must not be swept into a commit.

**Not yet accepted:** v1.12 rides into the same `analyst` re-gate as plan v1.10.

### U22 delivered — 2026-09-06, plan v1.10 (`3e5dc50`)

**All fourteen findings closed. No residuals** — nothing deferred, nothing disclosed in place of a
fix. The stakeholder principle held without needing to be invoked.

The two structural closures worth naming:

- **P4-1 → the unit boundary.** §3.6 gains one place where the seconds→milliseconds conversion is
  written; `ChatResult` normalises at the transport boundary, raw `stats` becomes auditability-only
  and unreadable by any timing path, and `coldLoadSeconds` is named the sole seconds figure. The
  class of defect is closed, not the instance.
- **P4-2 → §7 rule 5, the pattern.** *An edit list over shipped code carries the command that
  enumerates its own sites, that command's counts at a named commit, and a done-condition that
  re-runs it and asserts the residual.* §4 S1e's four grep-pinned tables are that list; DC-12 is the
  assertion. This is the fix for **two** prior incomplete edit lists (v1.8's `residencySource`,
  v1.9's `_percentile`), not just the second.

**Every pinned count re-verified by teco against the tree**, since rule 5's whole value is
reproducibility: `lmsCliCommit` 3, `sizeBytes` 1, `armKind` **50 lines / 57 occurrences** with the
per-file breakdown exact, `FORBIDDEN_BY_ARM_KIND` 10, `ARM_KINDS` 2, `_percentile` 7. All reproduce.
**The 50/57 corrects the Pass 4 review's "59 occurrences"** and states the lines-versus-occurrences
distinction that makes it checkable — the reviewer's figure was loose, and rule 5 is precisely what
stops that.

**The handoff contract earned its keep.** My brief asserted that Rule 4's closed form retires
`paired_bootstrap(diffs, *, B, seed)`. It does not: the architect read the note rather than my
summary and found `paired_bootstrap` **keeps** its seed — only the paired *binary* path stops
resampling, the continuous path being untouched — so `sampling.seed` and `PackRef.seed` stay, and
§3.3 now states their object for the first time. *Never paraphrase an upstream artifact into a brief*
is the rule that caught this; it was my error, and the instruction to read the file is what contained
it.

One decision the note left open and the architect took: `DecidedBy`'s `"cluster-bootstrap"` token is
renamed **`"conservative-envelope"`** (27 lines), for the note's own stated reason with the sign
reversed.

### Two items routed back to the note — U23, in flight

1. **§4 S2 rule (iv-b)** applies §11.6's p50 gate to the three sibling medians with
   `X = statsCoveredCount`, `Y = latencyItemCount`. Asserted at v1.9 and accepted by the gate, but it
   is a plan-side *application* of a note rule to a figure group §11.6 does not enumerate — so it is
   `data-scientist`'s to confirm, not the gate's to keep accepting.
2. **Note §11.2 cites `results.py:541`**; the shipped copy is `:573`, which I verified. Cosmetic, but
   the plan's Table C pins the correct line and rule 5 makes the discrepancy visible.

**The Pass 5 re-gate is held until U23 settles item 1** — deliberately. If rule (iv-b) is wrong the
plan changes, and gating a plan that is about to move wastes a ~200k-token review. The wait is
minutes; the gate is not.

### U23 delivered — 2026-09-06, note v1.13 (`5197ce6`)

**Item 1 was not a rubber stamp: (iv-b) is confirmed by *correcting the note*, not by applying it.**
§11.4 sent `ttftMs` and prefill to §11.6's p50 gate while exempting `tokensPerSecond` as
diagnostic-only. But the three share **one** coverage number, so they print or refuse together by
construction — the exemption would have printed a `tokensPerSecond` median over exactly the subset
the gate had just judged too short. **§11.4's split is withdrawn; all three take the gate.** The plan
was right and the note was wrong.

**A defect class this coordination has demonstrated it can miss.** (iv-b) was asserted at plan v1.9
on an argument that **did not yet hold** — the transfer needs the missingness *direction*, unknown
for a `stats`-less item — and became sound only at note **v1.12**, when the attained-level bound was
re-derived as distribution-free (subset-hood only). **Passes 3 and 4 both accepted it.** A rule whose
justification postdates its assertion is invisible to a gate that reads the pair as of today, and
Pass 5's brief now carries an explicit sweep for the same shape.

**The confirmation carries one condition — co-presence.** One coverage number for three figures is
right only while the three exist on the same items: prefill *additionally* needs a usable
`usage.prompt_tokens`, so an item with `stats` but no token count puts prefill's true `X` below
`statsCoveredCount` and both the gate and slot 2's denominator overstate coverage for one figure of
three — **silently, in the direction that prints**. The note carries the assertion (§11.10 test 7);
whether the *plan* owes a rule of its own was left open and is routed to Pass 5.

Item 2 fixed and made durable: §11.2's `results.py:541` → `:573`, both copies now pinned **at
`5878014`**, so the reference cannot go stale again under §7 rule 5.

### U24 dispatched — the Pass 5 gate, and what turns on it

Three consecutive plan gates returned *needs changes* and S2 has been held throughout. Pass 5's
question is whether **v1.10 is finally implementable**, not whether it is perfect. The brief weights
it to four things: (1) are Pass 4's fourteen genuinely closed, compactly dispositioned; (2) **does
§7 rule 5 actually work** — the pinned counts all reproduce, so the gate's effort goes to whether the
four tables are *complete*, given Table A already documents a site the grep cannot find, which is
either a principled escape hatch or a reopening of the defect rule 5 exists to close; (3) the two
items carried here deliberately — co-presence, and plan §4 S2 rule **(i)** versus **(iv-a)**, which
are one word from contradicting; (4) the sweep for rules whose justification postdates their
assertion.

The brief states plainly that the verdict must not be softened to unblock S2 — an unfounded approval
costs more than a fifth revision.

### U24 delivered — 2026-09-07, plan gate Pass 5 (`b9964d1`) · **needs changes**

**v1.10's central claim holds: all fourteen Pass 4 findings are fixed, none unfixed.** But five carry
a residual, and **two of v1.10's own fixes introduced a new defect while closing the old one** — the
failure mode this coordination should now expect from a large single-revision closure.

**P5-2 is the finding that matters, and it vindicates weighting the brief at the mechanism rather
than the instance.** §7 rule 5 — adopted to stop a *third* incomplete edit list — **does not work as
stated.** It claims a table is "complete by construction"; it is not, twice: (a) sites carrying no
token are invisible to the grep — **verified here: 18 `arm_kind` (snake_case) lines that Table B's
`grep -rFn armKind` (camelCase) structurally cannot see**, including fixtures and the tests pinning
the required-field contract; (b) `armKind` **survives by design**, so there is no zero residual over
the token the table is about, and DC-12 checks two auxiliary tokens that zero out from the mapping
rename alone. The converse DC-12 needs — *residual zero ⇒ nothing missed* — is false. Stated as a
guarantee it would have been trusted by the next revision exactly as v1.8's and v1.9's lists were.
The fix is concrete and its commands are verified in Appendix E.1; Table A's `sizeBytes` row is
already the right pattern and becomes the rule.

**P5-1: a fix that reopened a closed finding.** The sentence written to close P4-5 applies §3.6's
*tool-calling* eligibility gate unscoped, so every `model:embeddings` arm exits `4` — refusing
`text-embedding-qwen3-embedding-0.6b`, the exact model §4 S3's done-condition 1 names as the first
real run. G3-1 re-entering through its own fix.

**No finding is blocked on unbuilt work.** The gate was explicit: every one is a plan edit available
today, most one sentence. That is why the verdict is *needs changes* rather than
*approve with suggestions*, and it means v1.11 should be a much smaller revision than v1.10 was.

**The dependency sweep the brief asked for found one instance** — v1.10 pairs itself to note v1.12
while the note is at v1.13 (P5-5) — plus **P5-6, the inverse of the same class**: a note claim
invalidated by a plan change made in the *same* revision (P4-7 merged timeout into
`latencyWithheldForNoResponse`, leaving `censoringExact`'s clause 1 unevaluable, so a timeout-only
run silently prints slot 3's weaker string). The class is now demonstrated in both directions.

### Routing — U25 and U26, parallel on disjoint files

**Three findings needed a method ruling before the plan can close them**, and are routed to
`data-scientist` rather than guessed: the **co-presence shape** (the note leans to a separate count
for prefill, the reviewer recommends the conservative single count — the architect is blocked on
this one), **whether `censoringExact` clause 1 survives** the widened no-response category, and
**whether `paired_cluster_bootstrap` is still needed at all** — Table D retires its only production
caller, and `paired_bootstrap`'s only production call site is inside it, so an architect working
Table D verbatim would be deleting a public statistical function on its own judgement.

`architect` is fresh (the v1.10 author closed at 293k tokens, and this work is self-contained). Its
brief sequences the three inbound rulings **last** and I relay them mid-run, the pattern that worked
for U21→U22.

### U25 delivered — 2026-09-07, note v1.14 (`ca69cb1`)

All three rulings changed the note; none was a plan-only instruction. Routing them rather than
letting the architect infer them was the right call on **all three**, and on one it was decisive.

- **P5-5 — the conservative single count**, not a separate `prefillCoveredCount`. An item with
  `stats` but no usable `promptTokens` leaves `statsCoveredCount` **and all three medians**. **The
  trap:** the reviewer's phrasing reads either way, and one reading is a defect — dropping the item
  from the count while keeping it in the `ttftMs`/`tokensPerSecond` medians prints a denominator that
  does not describe its own numerator. An architect inferring this had a coin-flip chance of encoding
  the bad half. Rule (iv)'s identity also becomes `≤`, since an excluded item was timed and sits on
  neither side of the old equality.
- **P5-6 — clause 1 survives, but the item needs a third state.** `withheldFor` becomes
  `load | timeout | no_response`; the **counter stays one**. The v1.10 merge was right for the counter
  and wrong for the item, and the reason is deeper than evaluability: a timeout is a **censored**
  observation, a call that failed at 40 ms is a **missing** one, and v1.12 had no false branch for the
  latter at all.
- **P5-3 — keep both functions.** The closed form retires the *binary* paired interval only; the chain
  lost its current caller, not its designed consumer (§3.2d's continuous verdicts — the embedder
  pack's MRR is a committed deliverable). The keep takes `sampling.seed`'s discriminator, so it is
  checkable rather than sentimental. **This is exactly what the routing existed to prevent:** an
  architect working Table D verbatim would have deleted a public statistical function on its own
  judgement.

**A live defect five review passes did not reach.** `stats._widen` clamps to `[-1.0, 1.0]` — verified
directly at `modelbench/stats.py:191`, its own docstring naming the difference of proportions it was
written for. That is **wrong for `sep_z`** (§5.2), whose per-query differences are z-score differences
and unbounded: an interval above 1 is silently clamped and the point estimate can land **outside its
own interval**. The verdict survives (exclusion of zero is decided by the lower bound); the printed
interval does not. It is invisible to a static gate because it only becomes wrong when §5.2's
comparison is wired — a **later stage** — which is precisely the shape the stakeholder principle
exists to catch. Fix is one argument, blocked on nothing, and **must land before §5.2 is wired**.

Relayed to the in-flight `architect` immediately; it was blocked on ruling 1 and the relay carries
the trap explicitly rather than the ruling alone.

### U26 delivered — 2026-09-07, plan v1.11 (`85a32e5`)

All ten Pass 5 findings closed, nothing carried, none named blocked on unbuilt work — and the
revision is **smaller than v1.10** (+540/−110 against +738/−134), the first time the trend has turned.

**The architect rejected half of the gate's own prescribed fix, and was right.** Pass 5's P5-2
prescribed two residual commands; one — `grep -rFn arm_kind … | grep -cF '"model"'` → 2 → 0 — is
**unsound**, because `armKind` keeps the value `"model"` by design, so `tests/conftest.py:148`'s
`arm_kind: str = "model"` survives a *faithful* edit. **Verified here: that line is a Python parameter
default, not a fingerprint field.** The prescribed residual would have failed on a *correct*
implementation. It was replaced with two that hold, and the lesson generalised into rule 5 itself —
***a residual that fails on a correct edit is a trap, not a check.*** A reviewer's prescription being
refused with evidence is the independent-review contract working in the direction people forget it
runs.

**Sixth instance, 2026-09-08, and it sharpens the rule rather than repeating it** *(plan-gate
P10-3)*. Table H's residuals 2 and 3 are the first **forward** application of §7 rule 5(b)'s third
form — the rule written to stop exactly this — and they are a trap anyway: they pin the locals
`lo`/`hi` while `-ml` Rule 4a, which the table explicitly declines to restate, names those
quantities `u_lo`/`u_hi`, so an implementer doing precisely what the table says reads **0** against
a stated target of **1**. **The forward application did not fail; an unstated precondition did.** A
third-form residual is only a check if the table **writes out the text it pins** — otherwise the
residual is the sole statement of a spelling nobody was told to use. Table C's `:159` row, which
Table H cites as its own precedent, does write it out; that is the difference. Routed to the
architect as U57, with the question of whether rule 5(b) should state the condition rather than
leave it implicit.

**Seventh instance, same day — and this one was *introduced by the fix for the sixth*** *(plan-gate
P11-1)*. P10-3's repair removed the introduced local from residuals 2 and 3 so they would survive an
implementer who varied the spelling. Sound in isolation. But **P10-2 landed in the same revision**
and requires `report.py` to reach `SUPPORT_DIFF_PROPORTIONS`, the residuals are scoped to the whole
`modelbench` package, and `-F` matches straight through a module qualifier — and `report.py` already
spells its constants `stats.LEVEL_CI95_LO` off a `from modelbench import stats` import, so the
natural renderer spelling reads **2** against a target of **1**. Verified in both halves, plus a
probe. **The v1.19 form was immune to this and vulnerable to the other**, which is the whole finding:
rule 5(b)'s two repairs are not interchangeable and the second has an unstated cost — **removing a
name widens what the command can accidentally match**, so it holds only with a scope narrow enough
to contain the widening. **The coordination lesson is separate and larger: two findings closed
correctly in one revision produced a defect from their *interaction*, which neither finding's own
re-check would have caught.** A re-gate must be briefed to check same-revision fixes against each
other, not only each against its own finding.

**v1.10's own count claims did not reproduce, and v1.11 caught it unprompted:** `ARM_KINDS` was pinned
at `fingerprint.py:135` and the membership test at `:161`; both are off by two (`:137`, `:162`) —
confirmed here. A *counts-at-a-named-commit* claim that does not reproduce, **in the very revision
that introduced rule 5**. Pass 5's Appendix E.1 was also off by one (14 `arm_kind` lines carrying no
`armKind`; it is **15** — 18 total, 3 carrying).

**All sixteen pinned counts across the five tables re-verified by teco against `5878014`. Every one
reproduces.** `armKind` 50/57 · `arm_kind` 18 · `REQUIRED_BY_SCHEMA` 22 · `EXPECTED_MODEL_SCHEMA_1` 3
· `FORBIDDEN_BY_ARM_KIND` 10 · `ARM_KINDS` 2 · `_percentile` 7 · `bootstrap_seed` 29 ·
`conservative_envelope` 8 · `cluster-bootstrap` 27 · `DecidedBy` 3 · `paired_cluster_bootstrap` 13 ·
`paired_bootstrap` 8 · `_widen` 7 · `Fingerprint(` 12 · `max(-1.0, point` 1.

**P5-1's fix is stated on both sides** (§3.4.4a *and* §3.6) deliberately, because a one-sided scope is
exactly how P4-5's fix reopened G3-1. The `_widen` clamp defect lands as **new §4 S1e Table E** — the
clamp required with **no default** (rejecting a silent-wrong-default), a test that reproduces the
defect, and **§4 S3 done-condition 2 gated on it**, the only deadline in S1e.

### U27 — the Pass 6 gate

Blocker trend across passes 3→4→5: **2 → 3 → 2**; majors **6 → 5 → 4**. Not converged, but every pass
has found real defects rather than churn, so the gate keeps earning its cost.

Pass 6's brief weights it to five things: the ten dispositions; **adjudicating the architect's refusal
of P5-2's second residual** (do not defer to either side); whether **rule 5's second formulation** is
now sound, given the counts are already verified so the effort goes to the rule; verifying — not
accepting — the sweep against the *fix-reopens-a-finding* shape, now seen in three consecutive
revisions; and **Table E**, plus a sweep for other shipped S1 code that is correct for its current
caller and wrong for a caller the plan commits to adding — the shape that hid `_widen` from five
static passes.

The brief says explicitly: do not soften the verdict to unblock S2, **and do not manufacture findings
to justify the pass** — if v1.11 is implementable, say so plainly, and say whether S2 may be
dispatched.

### U27 delivered — 2026-09-07, plan gate Pass 6 (`afca8e0`) · **needs changes, S2 still blocked**

**Converging.** Blockers across passes 3→6: **2, 3, 2, 1**. Majors: **6, 5, 4, 2**. All ten Pass 5
findings confirmed fixed *against the tree rather than the change list*, with `model-bench/`
byte-identical to `5878014`.

**Both of v1.11's adjudications upheld**, which settles them: the architect's rejection of Pass 5's
prescribed residual was correct, and **rule 5's second formulation is sound as stated**, Table B
satisfying it. What fails is its *application* — see P6-2/P6-3 below.

**P6-1 (blocker) — the item-5 sweep, and the class is larger than `_widen`.** The embedder pack
declares `verdictMetrics = ["mrr"]` and `headlineMetric = "mrr"`, but **no field on the record can
carry a per-item continuous value.** Verified here: `ItemResult.counts` is `Mapping[str, int]` and
`scored_outcome` returns `self.counts[metric] > 0` — a reciprocal rank of 0.5 is **unstorable** and
any positive count is `True`. The comparison loop is binary end-to-end (McNemar + Holm, every string
in `pp`); `separationZ` reaches no table because `named_metrics()` omits it. Two silent outcomes on a
**green** run: a booleanised MRR rendered as a McNemar `+X pp` verdict, or *"No verdict: no paired
data"* for the pack's only verdict metric. And the sharp edge: **Table E gates S3 DC-2 on the `_widen`
clamp, but the interval that clamp protects has no producer and no renderer.**

This is the second instance of the shape that hid `_widen` from five static passes — *shipped code
correct for its current caller and wrong for a caller the plan commits to adding*. It is invisible to
a gate that reads the plan and the tree as they stand, which is why the sweep had to be asked for
explicitly.

**P6-2 and P6-3 (majors) — v1.11 breaking its own new rule, internally.** Table B's residual
`{"model", "deterministic"}` → 0 fails on the implementation Table B's *own* `fingerprint.py:137` row
authorises, so DC-12 fails on a correct edit — **the exact trap the architect had just rejected Pass
5's residual for, reintroduced one table over.** And Table E's four "`_widen`-related tests" are
`verdict()` tests whose *names* contain "widen"; none calls `_widen`, while the three sites the edit
actually breaks are in no command. The rule is sound; the party that wrote it could not apply it
twice in the same revision. v1.12's brief therefore requires **every residual in all six tables run
against the tree** and shown to be non-zero now *and* zero after a faithful edit.

### Routing — U28 and U29, parallel on disjoint files

**One coordinator decision taken, and stated to both so it is not re-litigated:** P6-1's
**record-shape half lands in S1, not S2.** §4 S1e's own *free only now* argument applies —
`results/runs/` does not exist, so no stored record is invalidated and no migration is owed; deferring
means changing a record schema *after* records exist. That was the gate's open question 1, put to
teco, and it is a sequencing call within the agreed scope rather than a stakeholder decision.

The **method half** is `data-scientist`'s and is not guessed: what the continuous instrument needs per
item (type, domain, unscoreable representation, absent-versus-zero), whether `separationZ` shares the
carrier, and the gate's open question 2 — whether §3.2e owes a **fourth verdict string** for a pack
with no McNemar and an `n/a` floor, today's three all assuming a binary instrument with `pp` as the
unit. Relayed to the in-flight `architect` when it lands, the pattern now used three times.

### U28 delivered — 2026-09-07, note v1.15 (`0ad0e7a`)

**The carrier is a *second* per-item map — `measures: Mapping[str, float]` — never a widened
`counts`.** The reasoning is the part worth keeping: widening `counts` to `float` makes the
booleanisation **type-legal without making it wrong**, and puts a count that §4.2's denominators count
into the same key space as a measurement they must not. Field-level specifics:

- **No domain constraint on the carrier** — MRR is `[0,1]`, `sep_z` unbounded — because constraining
  it would be **the `_widen` clamp mistake one layer up**. Non-finite refused *at the carrier*: one
  `NaN` propagates through the mean and both percentiles and reaches the reader as a **rendered
  interval** rather than an error.
- **Absence stays on `scoreable`**, so `measures` is **not** `float | None`: an **MRR of `0.0` is a
  measurement** and must stay distinguishable from an unjudged query. A second home for absence is a
  second thing to keep in step.
- **A metric lives in `counts` or `measures`, never both** — the map *is* the declaration of which
  instrument decides it; a name in both is refused, not resolved by code order.
- **The one line that closes P6-1's silent failure:** `scored_outcome` **raises** on a `measures`
  metric instead of returning `value > 0`.

**It reverses a v1.11 decision, which is why routing it mattered.** `separationZ` is **not** on the
verdict path (`verdictMetrics = ["mrr"]`), so **Table E's `_widen` clamp is due with the `sep_z`
comparison, not as a precondition of S3** — v1.11 gates §4 S3 DC-2 on it and that gate must move.
This is exactly the deadline sentence Pass 6 flagged as protecting an interval with no producer and
no renderer; the gate saw the symptom, the note supplied the correction.

**And one ruling neither the gate nor the brief asked for**, which no architect could have guessed: a
**`verdictMetrics` family is homogeneous in kind, and `validate` refuses a mixed one.** Holm orders by
p-value; a continuous verdict has none, so a mixed family has **no ladder** and the correction
silently does not happen for one member. An all-continuous family with `k > 1` corrects **in the
interval** (Bonferroni, deliberately not Holm). The embedder is `k = 1`, so it binds nothing today and
forbids the silent case later.

Relayed to the in-flight `architect` — the fourth use of the mid-run relay, and the first that had to
carry a **reversal** rather than an addition.

### U29 delivered — 2026-09-07, plan v1.12 (`5b67416`)

All five Pass 6 findings closed; nothing carried, nothing *deferred by choice*, nothing blocked on
unbuilt work.

**P6-1 closed and split as teco ruled.** The record half is **§4 S1e Table F**, an **S1** edit —
`ItemResult.measures`, a `scored_value` sibling, and `scored_outcome` **raising** on a
`measures`-resident metric, which is the single line that turns *a booleanised MRR rendered as a
McNemar `+X pp` verdict* into a loud failure. The renderer half is stated end-to-end, and
`DistributionSummary` carries §5.2's median and p10 so **`sep_z` reaches a table at all** — it
previously reached none. **Table E's S3 deadline is withdrawn on the note's reversal**; S3 **DC-1** is
gated on Table F instead, and that gate is **self-enforcing**: no carrier, no storable result. A
deadline that enforces itself is strictly better than one a reviewer has to remember.

**P6-2 and P6-3 closed with the general lesson stated rather than the instance patched** — *a
command's output is a **superset**, never the site list.* Table E's four "`_widen` tests" are named as
**non-sites**: `def` lines that matched on a **test name**. That is the third distinct way rule 5 has
been mis-applied, and the first time the plan has generalised instead of patching.

**All twelve residuals across the six tables re-run against the tree today**, each confirmed non-zero
now **and** zero after a faithful edit — the specific demand the brief made after two revisions
shipped a residual that fails on a correct implementation. **teco re-verified the new ones:**
`frozenset(FORBIDDEN` 1 · `REQUIRED_BY_SCHEMA[1]["model"]` 3 · the key-set assertion 1 ·
`paired_cluster_bootstrap(` **5**, split 2 `stats.py` / 3 `test_stats.py` exactly as claimed ·
`separationRaw: float | None` 1 · `separationZ: float | None` 1. All reproduce.

**Raised rather than silently absorbed** — the behaviour §7 rule 3 exists to produce: `-ml` §3.2f
still publishes the pre-v1.11 *"cluster-bootstrap"* wording that §3.4 Rule 4 retires, and `stats.py`
renders §3.2f **verbatim** (confirmed here — the *"Not distinguishable at this sample size. The
cluster-bootstrap interval…"* string is in the shipped file). An implementer copying §3.2f leaves the
retired token and **Table D's residual never reaches zero**.

### U30 dispatched, and the Pass 7 gate is held behind it

Four items, all the note's: §3.2f's sweep (item 1, the trip hazard above); **the continuous-verdict
producer's signature**, which the plan calls but nobody has specified (item 2 — a gate would find it,
and it is better written than discovered); confirmation that the homogeneous-family rule's
enforcement point moves from `validate` to `compare_report` pass 1, since **a manifest carries no
records** and a manifest `kind` field is the inference §3.2d refuses (item 3); and which of
`sep_raw`'s figures print (item 4, presentational, working default already in place).

**Holding the gate is deliberate and is the same call that paid at Pass 5:** items 1 and 2 would both
become gate findings, and gating a pair about to move wastes a ~200k-token review to tell us what we
already know.

### U30 delivered — 2026-09-07, note v1.16 (`e290148`)

All four items changed the note, and two of them went beyond what was asked.

- **Item 1 — the sweep was six sites, not two.** The plan saw §3.2f's two variants; the same retired
  instrument name was also in §3.2e's B-1 precondition, §3.4's decision-rule branch and Rule 7's
  branch label. Two more found unasked, both the same defect one string over: **§3.2e verdict 3's
  interval name is a substitution the note had never published** — it lived only inside Rule 4's sweep
  list, so an implementer copying §3.2e, *the surface they copy*, never learned the word varies; and
  Rule 4 still read as though the `DecidedBy` rename were an open architect's call when plan v1.11
  had taken it. A document saying "recommended" about a decision both documents have made is how the
  next reviewer loses an hour.
- **Item 2 — new §3.4 Rule 8** specifies `continuous_verdict()`. Two things worth carrying:
  `ContinuousVerdict` is a **sibling type**, not a `Verdict` with six fields left empty (the
  "`None` until it is not" shape the note already refused for `alpha_step`); and **the multiplicity
  correction is made *unrepresentable* rather than guarded** — the function computes its own
  percentiles and exposes no percentile parameter, so a caller *cannot* render a `k = 3` family at
  2.5/97.5 by omission. Rule 4's `n != len(diffs)` lesson applied at the signature.
- **Item 3 — the enforcement *point* is confirmed, the *granularity* is corrected against the plan.**
  v1.12 excludes and names the offending members; the note rules the refusal is of **the whole
  family's verdicts**. `k` is pre-registered, so dropping the minority kind **shrinks `k` after
  results exist** and hands the survivors a *weaker* correction than the one declared — the fishing
  artefact pre-registration exists to prevent, arriving as a repair. Every number still prints,
  labelled `exploratory — no significance claim`.
- **Item 4 — `sep_raw` prints median, p10 and fraction above zero**, no mean for either quantity
  (the concrete reason `ContinuousMetric` is the wrong aggregate carrier), **and no difference between
  two models' `sep_raw` figures on any path** — it is scale-dependent per model, which is the entire
  reason `sep_z` exists, and the shared per-item carrier makes such a difference *computable*, so the
  prohibition has to be written rather than inferred.

### U32 dispatched — and its brief's unusual instruction is to stay small

v1.12 closed Pass 6 cleanly with twelve verified residuals; v1.13 absorbs **four specific deltas** and
nothing else. The brief says plainly that **a large v1.13 would be a defect in itself**, since the
Pass 7 gate is queued directly behind it. Only delta 2(a) — `paired_cluster_bootstrap`'s quantile
levels becoming parameters — needs a new home in the edit set under rule 5, with a command, counts at
`5878014` and a residual verified both ways.

**Watch for oscillation.** This round's note items came *from* plan v1.12, and the note's answers now
need plan absorption. That is normal convergence while the deltas shrink — v1.16's are four, specific,
and three are citations or prohibitions rather than mechanism. If a future round's deltas do not
shrink, the plan/note pair is oscillating rather than converging and that is the point to stop and
re-scope rather than dispatch again.

### U32 delivered — 2026-09-07, plan v1.13 (`fbe5741`) · **it stayed small**

**+300/−91**, against v1.12's +521, v1.11's +540 and v1.10's +738 — the revision-size trend is now
monotonically down, and ~75 of the added lines are one new table. The brief's unusual instruction
held.

- **Delta 1** — a mixed `verdictMetrics` family has **all** its verdicts refused. The good part is how
  it is asserted: DC-13(e) and test 11d pin the two **negatives** — both arms render,
  `INVALID RESULTS EXCLUDED` empty — because those are what fail if an implementer builds it as a
  DC-10 exclusion. Asserting the negative is what catches a plausible mis-build; asserting the
  positive would not have.
- **Delta 2** — quantile levels become required parameters, in **its own Table G** rather than folded
  into Table E, whose subject is a *clamp*. Residual `_percentile(means, 2.5)` → 1 → 0, **scoped to
  `means` precisely so `cluster_bootstrap`'s surviving `_percentile(rates, 2.5)` at `stats.py:292`
  cannot hold it above zero** — the residual-must-reach-zero discipline applied at the point where it
  is easy to get wrong. **teco re-verified:** `_percentile(means, 2.5)` 1 · `97.5` 2 ·
  `paired_bootstrap(` 5 · `paired_cluster_bootstrap(` 5.
- **Deltas 3 and 4** — Rule 8 cited rather than restated, and `sep_raw`'s prohibition written in the
  §11.7-slot-6 shape with both structural enforcements named.

**Five things touched beyond the four, all reported unprompted** — including a pre-existing
inconsistency v1.12 half-swept (the S1e preamble said "five tables" while the next paragraph said
"six"). Self-reporting the overreach is what makes "stay small" checkable rather than a hope.

### U33 — one new raise, and it exists only because delta 3 landed

Rule 8 takes **no `support`**, so `continuous_verdict()` — now the *only* caller on the verdict path,
since delta 3 stopped the loop calling `paired_cluster_bootstrap` directly — has nothing to forward as
Table E's **required-with-no-default** clamp. The architect classified it **bounded and non-blocking**
with a checkable reason (the embedder is the only continuous-verdict pack, `designEffect` 1.00 by
construction, and at scale 1.0 `clamp=None` and `(-1.0,1.0)` return identical intervals for `mrr`),
interim `clamp=None`, recommendation *Rule 8 gains `support`*.

Routed to `data-scientist` with the sharp question attached: v1.16 made a deliberate point that Rule
8's **negative** parameters are load-bearing — adding one cuts against that grain, so it must say why
`support` differs in kind, and what an *unbounded* support does given Table E exists precisely because
the clamp must not apply to `sep_z`.

### Pass 7 dispatched in parallel, not held — and it is asked the question that decides the next step

The deltas are shrinking (**four → one**), so the oscillation test set last round is passing and the
gate's other work is independent. Pass 7 is told the `support` item is in flight, to judge **only**
whether `clamp=None` is a safe interim, and not to spend effort on the fix.

Beyond the findings, the gate is asked two things directly: **may S2 be dispatched** — in those words
— and **is there remaining risk that static review can still reduce, or is the residual risk now the
kind only execution finds?** Two defects here were found by asking a specialist rather than by a
static pass (`_widen`'s clamp, the continuous carrier), both the same shape: *shipped code correct for
its current caller and wrong for a caller the plan commits to adding.* That judgement, more than any
single finding, decides whether a Pass 8 is worth its cost.

The brief names both failure modes explicitly: do not soften the verdict to unblock S2, **and do not
manufacture findings to justify the pass** — after six passes the pressure to produce a list is real,
and a gate that finds something because it is expected to is worse than no gate.

### U33 delivered — 2026-09-07, note v1.17 (`1fbdb6f`)

**The conclusion is endorsed and its premise replaced** — the most useful shape a ruling can take, and
not one the question invited.

The architect justified `clamp=None` on a **pack census**: the embedder is the only continuous-verdict
pack and its `designEffect` is 1.00 by construction. True today. But **nothing refuses a future pack**
declaring `design_effect > 1.0` on a bounded continuous verdict metric, and a census is exactly the
premise that goes stale **without anyone editing the sentence resting on it** — the class this
coordination has now spent four passes on.

The replacement is one line and strictly stronger: **`_widen` scales half-widths by
`sqrt(design_effect)`, so at 1.00 it returns its input unchanged and no clamp can bind — for *every*
metric, not just `mrr`**, an unwidened bound being a bootstrap percentile of per-unit differences that
already lies inside the difference's own support. Same conclusion, no census, nothing to go stale.

**Why `support` does not breach v1.16's "negative parameters are load-bearing" grain.** The four
refusals are **two kinds**, and `support` is a **third**: a category error (`ResolvingPower`,
`alpha_step`, McNemar *p*); a quantity **derivable** from what the function already holds (the
percentile levels); and a quantity that is needed and **irreducible** — `support` cannot be recovered
from `diffs`, since a sample of MRR differences in `[−0.3, 0.3]` is indistinguishable from
z-differences in the same range, and inferring a support from an observed range **is** the
silent-wrong default the clamp rule exists to refuse. The note grounds this in `design_effect`'s own
principle: *`_widen` can no more discover a metric's support than a resample can discover clustering
the declaration did not state.*

**And the shape is sharper than the plan proposed:** `continuous_verdict()` takes `support` and
**does not take `clamp`** — the difference-support derivation happens *inside*, because that
conversion is where a sign or an order gets transposed. So the loop **forwards
`ContinuousMetric.support` and derives nothing**, which is *less* work than v1.13 specifies. Plus a
fifth refusal: a `support` with `lo >= hi` **raises**, a degenerate support otherwise deriving
`(0, 0)` and pinning every bound to zero **silently, in the direction that prints**.

**Table E's required-with-no-default `clamp` still does real work** — `sep_z` is reported and not
verdicted, so its comparison calls the engine directly. Two callers, two surfaces; now written into
Rule 8 so nobody "simplifies" it away.

**Relayed to the in-flight Pass 7 gate**, which had been asked to judge the *old* premise — the fifth
mid-run relay, and the second to carry a correction rather than an addition. Two small plan-side
deltas follow (swap the justifying sentence; forward-don't-derive), to be absorbed after the gate
rather than racing it.

### Pass 7's first instance killed by a session rate limit — 2026-09-07

429, session limit, reset 12:50. **Checked the disk before believing the transcript**, the discipline
this coordination learned twice the hard way — and this time the transcript was accurate: the review
still ended at `## Pass 6`, `docs/` was clean, **nothing lost**. A platform failure, so re-dispatched
rather than re-scoped.

Two things folded into the re-dispatch rather than sent as a relay, because a fresh agent has no
transcript to relay into:

- The pair is **one revision out of step by design** — plan v1.13 was written against note v1.16, and
  v1.17 landed after. The two resulting plan deltas are named in the brief as **known, not findings,
  and explicitly not to be written up**; a gate that spends a section on something the coordinator
  already routed is wasted review.
- The gate is asked to judge the **safety classification on the note's replacement argument**, not on
  the census the plan still carries.

**One instruction added for the first time, and it should stay in every long-running brief here:**
*write findings to disk as you go rather than composing the whole document at the end.* Three agents
on this coordination have now been killed mid-run by session limits. Partial work on disk is
recoverable; partial work in a transcript is not — and twice the recovery only worked because the
agent happened to have written before dying.

**Timing note for future dispatches:** the session limit is real and recurring. Two long units in
parallel is affordable; three is what preceded this kill.

## Pass 7 — 2026-09-07 (`b6222c6`) · the convergence point

**1 blocker, 0 majors, 2 minors — the smallest set of seven passes, and the first in which no finding
is a defect *of* the mechanism.** Trend across passes 4→7: blockers **3, 2, 1, 1**; majors
**5, 4, 2, 0**; revision size **+738, +540, +521, +300**.

**The gate answered both questions it was asked, in the words they were asked in:**

> The plan **is implementable**. **S2 may not be dispatched yet**, and the gap is one small revision —
> not a redesign. … **The residual risk is now the kind only execution finds.** A Pass 8 is not worth
> a seventh full gate — a narrow re-check of v1.14 is.

**P7-1 (blocker) — `DistributionSummary` has no stored shape, and four shipped sites break.** Table F
retypes `separationRaw`/`separationZ` and puts them in `named_metrics()` without deciding the JSON
shape or naming what breaks. **Verified here, all four:** `results.py:359` and `:584` are bare-`else`
readers assuming `.mean`; `report.py:583` is a third; and `_decode`'s gate at `:385` admits only
`{"binary","continuous"}`, so a third shape **silently returns a raw dict** where a metric belongs —
the quiet one, and the worst of the four.

**Why rule 5 missed it is the durable part.** Table F's commands missed these sites for a **spelling**
reason: one greps the type *name* while the lines spell `"continuous"` as a string literal, and
another pins the variable `metric` while `results.py` uses `m`. The gate's diagnosis: **rule 5's
commands are *token*-based, and this defect class is *attribute*-based.** `grep -rn '\.mean'
modelbench` returns **exactly** the three bare-`else` readers — confirmed. A retype's site list is
found by *the attribute the new type lacks*, not by the type's name. This is the **fifth** instance of
*shipped code correct for its current caller and wrong for a caller the plan commits to adding*, and
the first with a mechanical detector.

**All four requested judgements came back sound**, which is why the pass is small rather than lenient:
the `clamp=None` classification **holds on the note's replacement argument** and is strictly stronger
than the census it replaced; Table E's required clamp **still does real work** on the exploratory
`sep_z` surface; Table G's `means` scoping is **principled** (every weaker form was run and cannot
reach zero) and it **belongs as its own table**; delta 1's negative assertions **do** catch the DC-10
mis-build, against all four plausible mis-builds.

**And the convergence answer, which is the one that changes what happens next.** There *was* remaining
static risk — P7-1 would have shipped. But the miss is explained and mechanically closable, and having
run the attribute procedure over **every caller the plan commits to adding**, only
`DistributionSummary` yields a finding. So: **v1.14 gets a narrow re-check, not a full Pass 8**, and
after it the work is implementation.

### U34 — intended as the last plan revision

Five items: P7-1's stored shape and site rows, the **attribute-based companion to rule 5** (judge the
shape, do not paste the grep), P7-2's symmetric residual that a *half-applied* edit cannot pass,
P7-3's label home, and note v1.17's two deltas — the premise swap and *forward-don't-derive*, both
already routed and deliberately excluded from Pass 7's findings.

The brief states the stakes plainly: **a large v1.14 would itself be evidence the convergence
judgement was wrong.** It also carries the write-as-you-go instruction now standard here after three
session-limit kills.

**After v1.14 clears its narrow re-check, the next dispatch is the S1 fix unit** — which has
accumulated: U20's residency element-shape assertion, the single shared `_percentile`, the closed-form
implementation, `_widen`'s conditional clamp, `withheldFor`'s three states, the `measures` carrier,
bootstrap quantile levels as parameters, and now `DistributionSummary`'s stored shape. It is one
coherent S1 diff and should be sized against the step-table rule before dispatch, not after.

## Gates deliberately skipped

The default is that nothing ships on its producer's word. Each exception is recorded here with its
reason, so a skipped gate is a decision on the record rather than an omission.

- **U48 (P7-1's fix), `c19f875`.** No behaviour change — `omit` is untouched and the only source
  edit is a comment; the change is test-strengthening plus two documentation corrections. The
  finding was *"these two narrowings pass green"* and the fix is *"now they do not"*, which is
  self-verifying: I reproduced both survivors failing, added a **third narrowing neither the gate
  nor the implementer proposed** (`armKind in ("deterministic", "robot")` — also fails), and
  re-ran P5-1's `null`-branch mutation to confirm replacing the old test lost no coverage. That is
  stronger evidence than a static read would produce. Secondary reason: the reviewer is at **214k
  tokens** after seven passes, and its remaining headroom is worth more on the `stats.py` diff.

## Working against an environment that kills runs

Seven rate-limit kills on this coordination. The two on 2026-09-08 landed in the same event and
separate cleanly, which is what makes the lesson usable rather than anecdotal: `analyst` had written
its deliverable to disk as a **skeleton with every section marked `_(pending)_`** before doing the
work and lost effectively nothing; `architect` held its measurements in context and lost all of
them. **So the mitigation is not to serialize dispatches or shorten them — it is to instruct
skeleton-to-disk-first, then fill each section to disk as it settles.** Every brief from here says
so. Standing companion rule, already paid for twice: **check the disk before believing the
transcript** — the `analyst`'s transcript ended mid-sentence on a read, while the disk held 27 lines
it had already written, and a coder killed mid-mutation can leave a deliberately broken source file.

**The rule survived contact and came back sharper** *(2026-09-08, U54's resumed run)*. The
`architect` **declined the literal instruction and was right to**: a plan is an approved, live
document that a gate reads and that I commit by explicit path, so a half-written v1.19 carrying
`_(pending)_` markers is a worse failure than a lost hour — the `analyst`'s case differs because it
was appending a *new pass* to its own review. What it substituted gets the same durability and more:
a **re-runnable measurement script**, covering all twenty-four residuals and every Table H
candidate, then six complete self-consistent edits each landing before the next began. That earned
its keep immediately — **the script caught its own bug**, reporting a false **0** for two residuals
it ran under `grep -rn` where the plan states `-rEn`, which transcription from context would have
carried into DC-12 as fact. So the rule generalises: **the durable artifact is the thing that
re-derives the numbers, not the prose holding them** — a skeleton where the deliverable is prose, a
script where it is measurement. Same instruction from here, stated as the property rather than the
form.

## A numbering collision I caused while warning against it

The impl review runs `## Pass 1`–`## Pass 9`, so U58's pass was always **Pass 10**. I briefed it as
**Pass 11** — carried across from the *plan* gate, whose sequence genuinely had reached 11 — in the
same hour I was telling U60 to prefix its findings `plan-gate P11-*` **precisely so two documents
would not collide on a pass number**. So I manufactured, from the other side, a second live instance
of the defect already open as a §7 rule 3 raise.

**The analyst surfaced it instead of working around it** — it followed the explicit instruction,
then wrote the gap into its scope section, its header pointer and a `Numbering:` note, and told me.
That is the right behaviour under a brief it has reason to think is wrong: comply, record, escalate;
do not silently correct a coordinator and do not block on it either. Renumbered to Pass 10 at my
request, with the three gap notes removed rather than reworded.

**The standing lesson is narrower than "be careful with numbers".** Two review documents on one
topic advance on **independent** counters, and a coordinator briefing both in the same session is
the single point where they get confused — no convention inside either document can catch it,
because each is locally consistent. So: **read the target document's own last heading before
naming the next one.** One `grep -n '^## Pass' <doc> | tail -1`, every time.

Landed at `5343ac6` — four edits, zero `Pass 11` references left, the anchor matched to the block's
existing convention, contiguous 1–10. **And the same agent volunteered the thing that mattered more
than the renumber**: `model-bench/` had gone from clean to carrying the coder's in-flight N4 work
while it was editing, so it told me to commit **only** the review document and warned that a broad
`git add` would sweep ungated source into a review-doc commit. It was right, and it is the second
time on this coordination that a delegate has protected a commit boundary I own rather than assuming
I would notice.

## Carried triggers

Not defects and not deferred work — conditions that change a decision if they occur. Each names the
document that must absorb it and when.

- **A clamp ruling that moves the support bound from the envelope's arms to its composed interval
  obliges four coupled plan edits, not one row** (`architect`, handoff at v1.18). Table E is the
  only table whose residuals pin an **exact source spelling** — `max(clamp[0], widened[0])` and
  `min(clamp[1], widened[1])`, both `stats.py`, both reading **1** on the live tree (re-checked).
  A faithful move rewrites that expression and sends both to **0**, which is precisely the
  third-form trap v1.17 removed. §7 rule 5(b) makes the response non-optional: **re-derive over
  the new spelling, never re-scope.** **Trigger: U51's ruling landing on "composed" rather than
  "arms."** Then `architect` does all four in one pass — the residual pair, Table E's three-state
  scoring table, and DC-12's Table E row — briefed with the note's conclusion. It has agreed to
  this and is standing by.
  **FIRED 2026-09-08** — the ruling is *composed* (`a707d09`, note v1.19 §3.4 Rule 4a). And it is
  **wider than four edits**: Rule 4a also retires the arm clamp to `clamp=None`, folds
  compose-and-clamp into **one private composer** (which is P8-5's fix, so P8-5 leaves U52's scope
  and joins this one), turns `bound_by` into a **three-token** closed set on a strict comparison,
  and changes the renderer's `zip(v.bound_by, (LEVEL_CI95_LO, LEVEL_CI95_HI))` branch so a
  `support bound` token never prints a `p=` clause. So the pass is: the residual pair, Table E's
  three-state scoring table, DC-12's Table E row, **the P8-1/P8-5 implementation spec itself**, and
  the line-pin clause below. **Held, not dispatched**, until U52 lands and is gated — Table E's
  residuals count *exact source text* in `stats.py`, and U52 is editing that file right now, so
  re-deriving them against a tree in motion is the one way to get this specific arithmetic wrong.
  **PREMISE CORRECTED 2026-09-08, before dispatch.** The bullet above assumed a faithful move
  rewrites the expression Table E pins. **It does not.** The pair pins `max(clamp[0], widened[0])`
  and `min(clamp[1], widened[1])` inside **`_widen`'s body** (`stats.py:261`); Rule 4a changes what
  `envelope_arms` *passes* at `:382`/`:387` (`clamp=(-1.0, 1.0)` → `clamp=None`) and leaves `_widen`
  verbatim. Re-run at `7f865e2`: **still 1 and 1**. So the third-form trap does not fire, Table E's
  pair is not invalidated, and the pass **shrinks** — its real work is a residual for each of Rule
  4a's four shipped-code edits, not a Table E re-derivation. **This is the line-pin lesson below
  confirming itself on the first occasion it could**: an exact-text residual pinning one function's
  body survived an edit elsewhere in the same file, where a line pin would have broken. The evidence
  for that clause is now measured rather than hypothetical.
- **Pending plan edit, not yet written: a third reason to prefer an exact-text residual over a
  line pin.** Derived by `architect` from the P8-3 near-miss and worth §7 rule 5's ink — *a
  residual stated as exact source text is robust to unrelated edits elsewhere in the same file in
  a way a line pin never is.* Evidence is on the record: P8-3's guard landed at `stats.py:228`,
  thirty-three lines above the expression Table E pins at `:261`, and moved neither count, where a
  line pin would have broken outright — and this plan has been bitten by line pins twice already
  (`test_results.py:507`, and the two v1.10 pinned wrong). **Deliberately not dispatched on its
  own:** `architect` is at 290k tokens, and this folds into the next plan revision at no extra
  cost. **Trigger: whichever comes first** — the clamp ruling's four-edit pass, or Table F landing
  and DC-12's sign-off. **FIRED 2026-09-08** — the clamp pass came first; it rides in that brief.
- **DC-12's end-of-round re-run of all eighteen residuals cannot be signed off until Table F
  lands** — Table F's three read **1** each today, correctly, being the one unstarted S1e table.
  *Blocked on unbuilt work*, the acceptable kind under the stakeholder's rule; not deferred.
## The gate was cut, and by whom

**2026-09-08.** Asked whether we were close to concluding, I answered no and put the real number
beside it: U60–U64 spent roughly **1.1M tokens across five units and shipped no implementation**.
They were the plan gate refining the specification of two edit-list tables. Every finding in them
was real — I verified them myself and several changed the outcome — which is exactly what made the
pattern hard to see: **the loop was regenerating, not converging.** Pass 12 found three majors in
the revision that closed Pass 11's majors. The seventh instance of the residual trap was introduced
by the fix for the sixth. A gate that finds fresh majors in each successive fix round is either
catching something genuinely hard or has become a machine for finding things, and from inside it I
could not tell which.

So I stopped guessing and asked. **The stakeholder cut the gate**: v1.23 ships ungated, and Table F
and Table H go straight to the implementer. The argument that decided it is that **a residual is
self-proving when run** — its whole purpose is to demonstrate an edit landed, and running it costs
one command. We had reached the point of *reviewing the specification of a check that is cheaper to
execute than to discuss*. A spec defect now surfaces as a grep that reads the wrong number, in the
implementer's hands, in seconds.

This is not the gate discipline being abandoned. The double gate earned its place on this
coordination repeatedly and the record above says so. It is the recognition that **the instrument
had outgrown the thing it measures**, and that the decision to keep paying for it was never mine to
take quietly by continuing. Recorded here because the next coordination will feel this same pull,
and the tell is precise and reusable: *findings stay real while fix rounds stop shrinking.*

- **§3.4.1's `iff` is transcribed twice with nothing tying them** — `validate()`'s deterministic
  branch and `to_dict`'s `omit` (impl review Pass 7, residual observation, explicitly not a
  finding). *Which* value a profile pins is a schema fact, so at schema 1 the duplication is cheap.
  **Trigger: a third transcription, or the `model-bench migrate` work of §3.4.3, whichever comes
  first** — at that point it is lifted into the schema rather than transcribed again. Absorbed by
  `docs/plans/small-model-benchmarking.md` when `migrate` is planned; raised to `architect` then,
  not now.
