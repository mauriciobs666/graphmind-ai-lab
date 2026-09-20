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
| U5 | `devops` | `aa7e668860b23a794` | delivered | 68-run sweep, session `catalog-sweep-2026-09-19`, `results/runs/` (committed `00f3b83`) | `teco` (independent re-verification, no specialist gate — data-collection execution, not design/code) → **confirmed**: file counts match exactly, 0 failures | 161.5k tok / 60 tools (across 3 resumes, ~10.4h wall incl. LM Studio runtime) |
| U6 | `qa-engineer` | `a7d1766672e6f8dca` | delivered | `docs/test-reports/small-model-catalog-sweep-report.md`, `footprints.json`, 5 reports (4 clean, 1 defective) | `teco` (independent re-verification: confirmed defect against raw run data + row counts on all 5) → **defect confirmed, real** | 192.8k tok / 51 tools |
| U6-fix | `tdd-engineer` | `a59c9478f8c3c4d1d` | delivered | `report.py` fix for n=0-vs-no-run conflation (Defect 1), TDD test, 5 regenerated `-02` reports | `teco` (independent: diff review, own mutation-test via `git stash`, own full-suite rerun — 1775 passed) → **confirmed correct**; formal gate below | 152.4k tok / 46 tools |
| U6-fix-gate | `analyst`+`data-scientist` | `abe64e775de8de18a` / `a8b97acd2b5cd9752` | gated | new sections in `docs/reviews/small-model-catalog-sweep-impl.md` | `analyst` → **approve w/ 2 minor suggestions** (own lens clean, didn't catch the below); `data-scientist` → **needs changes** (real, reproduced MAJOR: banner's "items attempted" count uses `len(r.items)`, wrong for any multi-metric pack — independently re-verified by `teco` against a real guard-judge run) | 145.5k / 44 tools + 135.3k / 37 tools |
| U6-fix2 | `tdd-engineer` (resume) | `a59c9478f8c3c4d1d` | in-flight | fix MAJOR (metric-scoped attempted-count) + 2 minors (untested `ContinuousMetric` path, untested multi-metric scoping) | — | — |
| U7 | TBD | — | queued, blocked on U6-fix | consolidated document via U3c's script (FR-9/FR-10) | `analyst` (+`qa-engineer` if it has walkthrough claims) → — | — |

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
model count, so nothing in Units A/B/C needs revisiting. U5's target run count was updated above to
71; U6/U7 are pack-shaped and unaffected by the model-count change.

**2026-09-19 — Scope grown again to 21 models / 72 runs.** Stakeholder asked to add a fourth
embedding model, `text-embedding-qwen3-embedding-4b`; confirmed present in the LM Studio catalog
(`GET /api/v0/models`: publisher `Qwen`, arch `qwen3`, quantization `Q4_K_M`, state `loaded` at
time of check — a read-only listing call only) before amending scope. Footprint check (4B params
at Q4_K_M, ~2.2–2.5 GB) clears the doc's existing footprint-at-quantization criterion cleanly —
same size class as several already-in-scope 4B chat/vlm models, no re-litigation of the criterion
needed. `tico` (agent id `a4c29d02ae1b639e6`, resumed for this task) amended
`docs/requirements/small-model-catalog-sweep.md` in place — Scope/FR-1/AC-1 counts 20/3/71 →
21/4/72, dated Decision log entry — and, self-flagged and fixed on request, a stale "19-model
list" cross-reference in the Out of scope section left over from the prior granite addition, now
corrected to "21-model list". Diff independently re-verified by `teco` against the live tree
before commit (`3248b69`); confirmed no other stale count references remain
(`grep -n "19-model\|20-model\|19 model\|20 model\|70 stored\|71 stored"`, the one remaining hit
is a historical Decision-log entry quoting the original 2026-09-18 question verbatim — correctly
left as-is). No code or design impact: same reasoning as the prior scope growth. U5's target run
count updated above to 72; U6/U7 remain pack-shaped and unaffected.

**2026-09-19 — Count correction: 18 chat/vlm models, not 17 (76 total runs, not 72).** `devops`
flagged a discrepancy present since the doc's very first commit (`56e2e02`): the Scope section's
bulleted Chat/VLM list has always held 18 models, while the summary line, FR-2, the acceptance
criteria, and the Problem-section's combo math all said 17 (68/70/72) — unrelated to either
embedding-count amendment above. `teco` independently re-verified the discrepancy directly against
the doc text and git history before escalating. `tico` (resumed, agent id `a786647d4686dccf8`)
investigated independently in parallel with the stakeholder being asked directly: three
independently-derived numbers in the doc (the original 70-combo arithmetic, a 2026-09-18
Decision-log "19-model list locked in" confirmation, and FR-2/AC's 17×4 math) all self-consistently
pointed to 17, with `mistralai_ministral-3-3b-instruct-2512` suspected as the accidental 18th
bullet. Stakeholder's ruling, direct and overriding that inference: **18 is correct** — both
Ministral catalog entries (`mistralai/ministral-3-3b` and `mistralai_ministral-3-3b-instruct-2512`)
are confirmed genuinely distinct Mistral releases. `tico` amended the doc accordingly (Scope/FR-2/
AC/Problem-section corrected to 18/72/76/76, plus two more stale numbers caught in the same pass —
FR-7's `C(17,2)=136`→`C(18,2)=153`, Out of scope's "21-model list"→"22-model list" — and a new dated
Decision log entry), independently re-verified by `teco` (arithmetic + a stale-reference grep sweep
— only historical/narrative log-entry text retains "17", correctly untouched) and committed
(`a912bcb`). U5 was `paused` for this question (relayed and resolved same day) and is now resumed
with the confirmed 22-model/76-run scope; U6/U7 targets update from 72→76 stored runs accordingly
but stay otherwise pack-shaped and unaffected.

**2026-09-19 — Track B released.** Stakeholder: "lets go the machine is all yours" — explicit,
unambiguous go-ahead that LM Studio is now free for this sweep's exclusive use, superseding the
standing hold. Confirmed LM Studio reachable (`GET /api/v0/models`) before dispatch. `host.json`
is dated 2026-09-17T18:19:13Z — now 2 days old; freshness against the plan's staleness trip-wire
(`docs/plans/small-model-benchmarking.md` §3.4, repo root) is folded into U5's brief as a
precondition check, since the requirements doc's Out-of-scope section assumes current attestation
going in rather than the sweep producing it. Dispatching U5 (the 72-run sweep) to `devops` now;
U6/U7 stay `queued` behind U5's data, unaffected by this release.

**2026-09-20 — Fourth scope correction: drop two unofficial 9B community finetunes.** Stakeholder
decision, direct: `qwen3.5-9b-uncensored-hauhaucs-aggressive` and
`qwen3.5-9b-claude-4.6-opus-uncensored-distilled` dropped from scope — `prism-ml/bonsai-27b` stays
in scope on its existing footprint-criterion clearance (aggressive Q1_0 quantization), but these two
are unofficial third-party finetunes (publishers `HauhauCS`, `LuffyTheFox`, non-standard `qwen35`
arch tag) judged unlikely to work reliably under this harness — a functional-risk call, not a
footprint one. Caught mid-run: `devops` had reached combo 7/72 (in flight, nothing stored yet) when
the instruction landed; it killed the loop cleanly rather than editing the plan file live under a
running read-loop, rebuilt the plan files down to 68/64 lines, and resumed on a 58-combo queue that
also skipped the 6 chat-role combos already OK'd. `tico` amended the doc accordingly (18/72/153/76/76
→ 16/64/120/68/68, 20 models/68 total runs), independently re-verified by `teco` (arithmetic +
stale-reference grep) and committed (`356a5bd`).

**2026-09-20 — U5 complete: 68/68 runs, 0 failures.** `devops` finished the corrected 68-run sweep
(all 16 confirmed chat/vlm models × 4 packs + 4 embedding models × the embedder pack), reported a
full close-out (session id, per-pack breakdown, host.json freshness action, artifact paths, and a
flagged-not-fixed observation that `bonsai-27b` is a severe latency outlier — 12-42 min per pack vs.
low-single-digits for the 3-4B models — worth a footnote in U6's report rather than a surprise).
`teco` independently re-verified before accepting: counted the actual stored JSON files by
`sessionId` and per-pack `modelKey`, confirming 72 files (68 real runs + 4 auto BM25 reference-arm
records the embedder pack always stores alongside an embedding run), all 16 chat/vlm models present
in each of the 4 chat-role packs, all 4 embedding models present, neither dropped 9b model present
anywhere. Committed the 68 run records as `00f3b83` (the 4 auxiliary BM25 records included in the
same commit, same file set). U5 marked `delivered`. **U6 and U7 are now unblocked** — proceeding to
dispatch both next.
