# Small-Model Catalog Sweep — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** — (M9)

Requirements: `model-bench/docs/requirements/small-model-catalog-sweep.md` (Status: Ready for
design, `tico`). Its own Decision log assigns execution ownership to `teco`: dispatch whoever runs
`./run.sh run`/`./run.sh compare`; the requirements document defines what the sweep must
accomplish, not how it is run.

**Key gap found before dispatch:** the current `compare` CLI/`report.py` only ever compares
**exactly two arms** (`report.py::_comparison_pair` takes `runs[0], runs[1]` unconditionally, even
when `--models` names more). FR-6/FR-7/FR-8/FR-9/FR-10/FR-12 need a genuinely new report shape (a
per-pack ranked table across all in-scope models, each with its own CI, plus an optional
reference-anchored Holm-Bonferroni family) — this is new code, not just repeated invocation of the
existing command. Useful primitives already exist and don't need re-deriving: `stats.wilson_interval`
(single-arm CI) and `stats.holm_steps` (generic Holm-Bonferroni over a p-value sequence).

Confirmed before dispatch: LM Studio is reachable (`GET /api/v0/models` responded, `qwen/qwen3-4b-2507`
shows `state: loaded`); `model-bench/host.json` exists (dated 2026-09-17, 2 days old — devops to judge
freshness against the `-ml` staleness trip-wire, not assumed current).

Two independent tracks, since sweep execution touches only `results/` data and the report feature
touches only `modelbench/report.py`/`cli.py`/`stats.py` — no file or fact overlap:

- **Track A (design → implement → gate) the report feature** needed for FR-6–FR-12.
- **Track B (execute)** the 70-run live sweep (FR-1–FR-5), which needs nothing from Track A —
  `./run.sh run` already exists and is stable.

Track B's output (stored runs under the shared session id) and Track A's output (the new report
capability) both feed the final compare/report/consolidated-document units.

**2026-09-19 — Track B held.** Stakeholder is using LM Studio for another purpose right now.
Starting Track A (design/implement/gate the report feature) only; U5 (and U6/U7, which need U5's
live data) stay `queued`, undispatched, until the stakeholder says LM Studio is free.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `ae8150d6702bb86b6` | delivered | `docs/plans/small-model-catalog-sweep.md` | `analyst` + `data-scientist` → — | 221.5k tok / 46 tools |
| U2 | `analyst` | `a5aa07976df6f4a5d` | gated | `docs/reviews/small-model-catalog-sweep.md` | `analyst` → needs changes (1 blocker, 2 major, 2 minor) | 154.5k tok / 40 tools |
| U2b | `data-scientist` | `acce2accdfd2c07ce` | gated | `docs/reviews/small-model-catalog-sweep-ml.md` | `data-scientist` → needs changes (Q1–Q4 resolved, corroborates U2's blocker) | 211.2k tok / 44 tools |
| U1r | `architect` (resume) | `ae8150d6702bb86b6` | delivered | plan v2, addresses U2+U2b | — | 322.9k tok / 25 tools |
| U2r | `analyst` (resume) | `a5aa07976df6f4a5d` | gated | Pass 2 on `docs/reviews/small-model-catalog-sweep.md` | `analyst` → **approve** | 208.6k tok / 9 tools |
| U2br | `data-scientist` (resume) | `acce2accdfd2c07ce` | gated | Pass 2 on `docs/reviews/small-model-catalog-sweep-ml.md` | `data-scientist` → **approve w/ 2 minor suggestions** | 249.5k tok / 5 tools |

**Plan v2 gated approved by both reviewers — proceeding to implementation.** Two non-blocking
suggestions from `data-scientist`'s Pass 2 (pre-registration-discipline test for the combined
ladder; pin `mean_bootstrap_interval`'s `levels` arg to unadjusted 95%) folded directly into Unit
A's brief rather than looping back to `architect` for a third revision — neither blocks A/B/C per
both reviewers.

| U1b | `tico` | `a76331f7598884859` | delivered | AC-3 wording reconciliation (committed `fc81cd3`) | — | 55.6k tok / 8 tools |
| U3a | `tdd-engineer` | `a24170310c31e20de` | gated | `stats.py`+`report.py` core (plan Unit A) | see U4a/U4a-ds | 400.6k tok / 167 tools |
| U4a | `analyst` (resume) | `a5aa07976df6f4a5d` | gated | code-gate, `-impl.md` new section | `analyst` → **approve w/ suggestions** (own lens clean; defers to U4a-ds's needs-changes as the binding verdict) | 295.1k tok / 39 tools |
| U3ar | `tdd-engineer` (resume) | `a24170310c31e20de` | accepted | Unit A fix: Q4 `n_units` + 2 minor suggestions | `analyst` + `data-scientist` → **both approve, explicit stopping signals — Unit A closed** | 465.4k tok / 53 tools |
| U4a-ds | `data-scientist` (resume) | `acce2accdfd2c07ce` | gated | code-gate, `-impl.md` new section | `data-scientist` → **needs changes** (Q4 `n_units` pooling defect, real & reproduced) | 311.7k tok / 33 tools |
| U3b | `coder` | `a04e75a1454a5ec5d` | delivered | `cli.py` wiring + README (plan Unit B, depends on U3a) | `analyst` (fresh, moderate) → in-flight | 173.2k tok / 55 tools |
| U4b | `analyst` (fresh) | `a79ed69e7e24cc7e1` | gated | code-gate, `-impl.md` new section | `analyst` → **approve w/ suggestions**, no blockers | 129.6k tok / 19 tools |
| U3br | `coder` (resume) | `a04e75a1454a5ec5d` | accepted | fold 2 minor suggestions (missing zero-arms test, README clause) — committed `4433e56` | — | 186.8k tok / 19 tools |
| U3c | `coder` | `a20f36e22226f0479` | delivered | `scripts/consolidate_sweep_reports.py` (plan Unit C, fixture-built, parallel to U3a) | `analyst` (light) → in-flight | 158.9k tok / 26 tools |
| U4c | `analyst` (fresh) | `ac2c3f299cd822b87` | accepted | `docs/reviews/small-model-catalog-sweep-impl.md` (own `-impl` doc, not a section of the plan review — analyst's own correct call per the closed role set) | `analyst` → **approve** (2 non-blocking: a `main()`-level test gap, HISTORY.md entry deferred) | 120.8k tok / 31 tools |
| U5 | `devops` | — | queued, **held** (LM Studio in use) | 71-run sweep under one session id, `results/` | — | — |
| U6 | TBD | — | queued | five per-pack reports via `rank` (needs U3a+U3b gated, U5 data) | `analyst`/`data-scientist` → — | — |
| U7 | TBD | — | queued | consolidated document via U3c's script (FR-9/FR-10) | `analyst` (+`qa-engineer` if it has walkthrough claims) → — | — |

Status legend: `queued` · `in-flight` · `delivered` · `gated` · `accepted` · `abandoned` · `paused`.

**2026-09-19 — Track A closed.** Units A (`stats.py`+`report.py`), B (`cli.py`+README wiring), and
C (`scripts/consolidate_sweep_reports.py`) are all implemented, independently gated (`analyst` +
`data-scientist` on A, `analyst` on B and C), and committed: `b605bed` (Unit A core),
`acb7e5e` (Unit C + its `-impl.md` gate section), `4433e56` (Unit B, folding both of `analyst`'s
minor suggestions). The `rank` CLI command and `scripts/consolidate_sweep_reports.py` now exist and
are ready for U6/U7 once U5's live data lands. One real, reviewer-found-and-fixed defect surfaced
and closed in this track (Q4 `n_units` pooling in `_rank_resolving_power_lines`, both reviewers
re-verified with explicit stopping signals) and one pre-existing, unrelated defect was logged to
`BACKLOG.md` rather than fixed (`stats.verdict()` polarity-blindness on guard-judge's two metrics).
Track A's own design docs (architect's plan v2, both plan-level reviews) were left uncommitted
during implementation and are being caught up now, in their own commit, as this checkpoint's
housekeeping — see the design-docs commit alongside this one.

Track B (U5 live sweep, U6 five reports, U7 consolidated document) remains **fully held,
undispatched** — still waiting on the stakeholder's explicit word that LM Studio is free for this
sweep's exclusive use.

**2026-09-19 — Scope grown to 20 models / 71 runs.** Stakeholder asked to add a third embedding
model, `granite-278m-multilingual`; confirmed downloaded and present in the LM Studio catalog
(`GET /api/v0/models`, catalog id `text-embedding-granite-embedding-278m-multilingual`) before
amending scope. `tico` (agent id `a3628d9fcdf158283`) amended
`docs/requirements/small-model-catalog-sweep.md` in place — Scope/FR-1/AC-1 counts 19/2/70 →
20/3/71, dated Decision log entry — diff independently re-verified by `teco` against the live
tree before commit (`92536a0`). No code or design impact: Track A's report code is generic over
model count, so nothing in Units A/B/C needs revisiting. U5's target run count updated above to
71; U6/U7 are pack-shaped and unaffected by the model-count change.
