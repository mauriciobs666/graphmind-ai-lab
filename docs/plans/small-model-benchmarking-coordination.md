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
| U3d — Pass 2 re-gate over plan v1.3 + note v1.3 + the amended requirements | `analyst` | `a0e3b74e34e1d4c40` (resumed) | in-flight | `docs/reviews/small-model-benchmarking.md` `## Pass 2` | — (is the gate) | — |
| U4 — S1 core (fingerprint, results, stats, report; no model calls) | `tdd-engineer` | — | queued (after U3d) | `modelbench/{fingerprint,results,stats,report,roles,cli}.py` + tests 1–6 | `analyst` + `data-scientist` (stats) → — | — |
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
