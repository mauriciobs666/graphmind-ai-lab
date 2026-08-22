# Workflow timers / scheduled wakeups — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-028

## Summary

Acceptance/black-box QA pass over K-028 (workflow timers / scheduled wakeups) as shipped, v3 of
`falkor-chat/docs/plans/workflow-timers.md`, against a real, running FastAPI process
(`falkorchat.app:app`) driven over REST — not a re-run of the offline unit/integration suite,
which is already extensive and independently re-verified by `analyst`'s diff-gate pass. Executed
2026-08-21 against the repo at its current commit (working tree as dispatched; no code changes
made by this pass).

**Verdict: PASS.** All twelve planned test items passed. No defects found. The feature behaves
exactly as `docs/plans/workflow-timers.md` v3 and `docs/DESIGN.md` §6.1–§6.3 describe: additive,
correctly composes with the shipped "not yet" pattern, correctly resolves the concurrent
human-vs-timer race to a single winner in both orderings, correctly avoids the cross-step marker
leakage the design's own self-review caught, respects `limit` under batch load, 503s cleanly
when disabled, and — the item this pass specifically existed to (re-)confirm given this
feature's own history — the mechanism actually works when driven end-to-end through the real
process, including the **automatic** periodic sweep tick, not just through static review or a
stubbed lifespan test.

`CPG: considered, not relevant — cpg_falkorchat was built 2026-08-17T00:40:42Z and the entire
K-028 delivery (services.py/executor.py/repository.py/api.py/app.py/config.py) postdates it;
every claim below is grounded in reading the live source and driving the live process directly.`

**Baseline.** Offline `pytest -q` → **1529 passed, 3 deselected** (matches the dispatch brief
exactly). This run wipes the shared `reference` graph at teardown (documented `AGENTS.md`
hazard) — re-seeded immediately with `bootstrap_schema.sh acme && seed_workflows.sh acme`,
re-verified `in sync` both before and after this pass via `verify_workflows.sh acme`. No shared
state (`acme`/`reference`'s two shipped defs) was left disturbed.

**Environment used for live driving**: a fresh, throwaway workspace `ws:qa028` (not `acme`), one
manually-started uvicorn process (`FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=1
FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S=5`, port 8010) plus a second, minimal disabled-engine
process for TP-011 (port 8011). Both stopped at the end of this pass. Five throwaway custom defs
were published (additively, new keys only) into the shared `reference` graph:
`qa028-timer-wait@v1`, `qa028-timer-human@v1`, `qa028-not-yet@v1`, `qa028-two-timers@v1`,
`qa028-batch@v1` — the same posture `seed_workflows.sh` already has for its own two defs, not a
new kind of shared-state mutation. No LLM/network call was exercised (every def here is
`kind:'process'`, `cmp`-only guards).

## Results

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-001 | Additive-only guarantee (`access-request@v1` unmodified) | **PASS** | Run `66fcb760…` drove `submit→route→approval→provision→activate` to `done`; two sweep calls mid-flight both returned `{"due":0,"resumed":[]}` — the run never appeared, matching pre-K-028 behavior byte-for-byte (7 `StepRun`s, same shape `DESIGN.md` §6.3 documents). |
| TP-002 | Publish rejects timer without escalation transition | **PASS** | `POST /workflow-defs` with `qa028-timer-bad` → `400 WorkflowDefSpecError`: *"step 'w1' declares config.waitForSeconds/waitUntil but has no outgoing transition guarded on {...} — this is what sweep_due_workflow_runs needs..."*; `GET /workflow-defs/qa028-timer-bad` → `404`, confirming nothing written. |
| TP-003 | Publish accepts correctly-shaped escalation transition | **PASS** | `qa028-timer-wait`/`qa028-timer-human` both published `201` and materialized `201`. |
| TP-004 | `wait` step timer happy path, manual sweep | **PASS** | Run `4c716308…`: immediate sweep → `{"due":0}`, still `waiting`; sweep after 3.2s → `{"due":1,"resumed":[{"runId":"4c716308…","status":"done"}]}`; final `ctx:{"timerFired":"w1"}`. |
| TP-005 | `human` step timer happy path, manual sweep | **PASS** | Run `0c3de09b…`: identical shape/outcome to TP-004 with `type:"human"` — proves the shared mechanism genuinely covers both step types. |
| TP-006 | Automatic periodic sweep tick fires in the real process | **PASS** | Run `87ffec03…`, due at t=+3s, **no manual sweep call made at all** — polled `GET` every 1s; run transitioned to `done` on its own at t=+7s (within the second 5s-interval tick), `ctx:{"timerFired":"w1"}` — the real `asyncio` periodic task, not the plan's own stubbed lifespan smoke test, was observed ticking. |
| TP-007 | "Not yet" pattern survives on a timer-bearing step | **PASS** | Run `ca9234c9…`: `POST .../input {"provisioned":false}` before due → run **re-parked** at `w1`, `ctx:{"provisioned":false}`, no `timerFired`; after due, sweep escalated it to `escalated` (not `activated`), final `ctx:{"provisioned":false,"timerFired":"w1"}` — exactly the composition v2 broke, now proven working end-to-end, not just documented. |
| TP-008 | Concurrent-resume guarantee, both orderings | **PASS** | **A (human wins, backgrounded near-simultaneous calls)**: run `0ef9048f…` → human input advanced to `activated` (`ctx:{"provisioned":true}`); the concurrent sweep call's own response showed `"raced":[{"runId":"0ef9048f…"}]`, `timerFired` never written, exactly one `StepRun` advance. **B (sweep wins, sequential)**: run `af95ebb7…` → sweep resumed via escalation (`ctx:{"timerFired":"w1"}`, terminal `escalated`); the subsequent human input got `409 WorkflowRunNotWaitingError`; exactly one advance in the step-run history both times. |
| TP-009 | Cross-step marker leakage does not occur | **PASS** | Run `85aa0ea4…`, def `qa028-two-timers` (A→B, each own timer/escalation): sweep resolved A's timeout → run parked at **B**, `ctx:{"timerFired":"A"}` — confirmed **not** already escalated (B's own guard needs `=="B"`); B's own due time later resolved it to `done2`, final `ctx:{"timerFired":"B"}`. The regression this test exists to catch did not reproduce. |
| TP-010 | Batch sweep evaluates multiple due runs; `limit` respected | **PASS** | Def `qa028-batch` used a past `waitUntil` so both runs were due at creation; `POST /workflow-runs/due {"limit":1}` → `{"checked":1,"due":1,"resumed":[{"runId":"d841b1aa…"}]}` while the second run (`7445e943…`) stayed `waiting`; a follow-up sweep cleared it. |
| TP-011 | Disabled-engine posture | **PASS** | Second process, `FALKORCHAT_WORKFLOW_ENABLED=0`: `POST /workflow-runs/due` → `503 WorkflowEngineDisabledError`; `POST /workflow-runs` on the same process → same `503`, same gate. Default-app-unaffected half already demonstrated by the offline baseline (1529 passing tests construct the default, sweep-free app). |
| TP-012 | Doc spot-check, `DESIGN.md` §6.1–§6.3 | **PASS** | Read in full during orientation: the `human`/`wait` bullets (§6.1) describe the shipped `waitForSeconds`/`waitUntil` mechanism and point at §6.2/§6.3; §6.2 has a dedicated "K-028 (shipped)" paragraph correctly framing the sweep as "one more `resume_run_with_ctx` submitter, not a new write path"; §6.3's K-025 handoff note is corrected in place (no stale "no scheduler" claim survives). Matches everything TP-004–TP-009 actually observed. |

## Defects

None found.

## Coverage & gaps

**Covered by this pass**: additive-only regression on the one real shipped `wait`/`human` def;
publish-time validation's positive and negative cases over REST; the timer happy path for both
step types via a genuinely manual sweep call; the automatic in-process periodic task actually
ticking in a real process (not a stub); the "not yet" composition fix (the actual defect class
v2 shipped with); the concurrent-resume single-winner guarantee in both orderings; the
step-scoped marker's cross-step leakage-prevention, proven, not just asserted by a unit test; the
batch/`limit` behavior; and the disabled-engine 503 gate.

**Deliberately not re-covered** (already offline-covered, independently re-verified by
`analyst`'s diff gate — see this plan's §7 "out of scope"): `_wait_due_at`'s pure-function edge
cases, `GRAPH.PROFILE` re-verification of `find_due_wait_candidates`, and the repository-level
CAS-contention test.

**Residual risk, named rather than tested** (matches the plan's own §8, restated for the
record): a true multi-process/multi-worker race was not exercised (only one uvicorn process was
running in this environment — the plan's own argument for why this is safe by construction, via
the shared CAS, was not independently stress-tested at that scale here); a def whose transitions
cycle back to a step *after* that step's own timer already fired once (the plan's named, accepted
narrow residual) was not constructed — narrower than TP-009's leakage case and explicitly
documented as accepted, not a gap in this pass; the batch fault-isolation case (a due run whose
drive raises one of `_drive_or_fault`'s four named faults, alongside a second due run that
still resolves) was not driven black-box — it is already covered offline
(`test_workflow_timers.py`) and constructing a faulting step black-box would have meant
authoring an unimplemented step type or similar synthetic failure with no added confidence over
the existing test; MCP-tool mirroring and multi-workspace sweeping were not tested because
neither was built (explicitly out of the plan's own scope, not a testing gap).

## Feedback & recommendations

- **No testability issues found.** The REST surface (`POST /workflow-runs/due`'s response shape
  — `checked`/`due`/`resumed`/`raced`/`faulted` — plus `GET /workflow-runs/{id}`'s `ctx`) gave
  everything this pass needed to assert on outcomes precisely, including which of two competing
  actors won a race, without reaching into Cypher.
- **Minor, non-blocking observation**: `config.waitUntil` accepts an absolute timestamp already
  in the past at publish time (no lower-bound check — confirmed while building TP-010's fixture,
  `qa028-batch`, deliberately, to get an immediately-due run for the batch test). This is
  consistent with the plan's own stated invariant text ("No upper bound... an arbitrarily-distant
  one costs nothing extra") and is arguably correct/useful behavior (an already-elapsed absolute
  deadline is a legitimate authoring case — the sweep just picks it up on its very next tick,
  which is the right behavior), not a defect. Flagging only so it's a documented, intentional
  reading rather than an unnoticed gap if a future reviewer wonders why there's no
  `waitUntil > now` publish-time check.
- **The automatic periodic task's default 30s interval makes an equivalent live check slow to
  reproduce by hand** (this pass used `FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S=5` specifically to
  make TP-006 provable in well under a minute). No action needed — the env var override already
  exists and is documented; noting it only because a future manual spot-check of "does the
  automatic tick work" should know to override the interval rather than waiting out a 30s+ real
  loop.
- **Suggested follow-up (not a defect, not blocking)**: none. The feature is in good shape to
  close per this pass; `teco`'s milestone closeout doc updates (`BACKLOG.md` flip, `HISTORY.md`
  entry) are the only remaining items this report is aware of, per the plan's §7.

## Artifacts

- Test plan: `falkor-chat/docs/test-plans/workflow-timers.md`
- Test report: `falkor-chat/docs/test-reports/workflow-timers-report.md` (this document)
