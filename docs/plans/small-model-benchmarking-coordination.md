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
| U40 — Diff-scoped review of `8fc2341` + adjudicate the implementer's 6 reported items | `analyst` (fresh) | `ad216ed80e4e38da2` (**killed by a session rate limit**, wrote nothing, left no mutation in the tree — verified on disk; **resumed** 19:22 after the limit reset) | in-flight | `docs/reviews/small-model-benchmarking-impl.md` `## Pass 5` | self → — | — |
| U41 — Fix round on `8fc2341`: **P5-1** (callSurface three-state), **P5-2** (pin the `to_dict` decision), **P5-4** (a test that misses what it is named for) | `coder` (fresh) | `a18e03b6ec3876e7c` | in-flight | `fingerprint.py` + `test_fingerprint.py` | `analyst` re-check → — | — |
| U42 — Plan **v1.16**: **P5-3** (the rule-5(b) trap in the plan that wrote rule 5(b)), **P5-5** (Appendix A stale), **P5-6** (Table C's site row stale — blocks the next unit) | `architect` (fresh) | `a6e6e78f99baf4ead` | in-flight | `docs/plans/small-model-benchmarking.md` v1.16 | `analyst` re-check → — | — |
| U43 — **S1 impl 2 of 3**: Tables **C + D + E + G** (the `stats.py` signature round, C→G mandated) | `coder` | — | **queued** — needs U42's re-pinned Table C; serialized behind U41 on tree/suite sharing | `model-bench/` source + tests | `analyst` diff-scoped → — | — |
| U44 — **S1 impl 3 of 3**: Table **F** (the continuous carrier, `results.py`/`report.py`) | `coder` | — | **queued** | `model-bench/` source + tests | `analyst` diff-scoped → — | — |
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

