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
| U13 — Plan v1.8: host-info source, first-call JIT budget, P4-4 stage re-attribution | `architect` | `aee9ac26e4c41f8ea` | in-flight | `docs/plans/small-model-benchmarking.md` v1.8 | `analyst` → — | — |
| U14 — Fix unit: P4-2, P4-4, M-ML-8 (the three gating S3) | `tdd-engineer` | — | queued — blocked on note v1.8 settling | `model-bench/**` | re-gate (both) → — | — |
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
