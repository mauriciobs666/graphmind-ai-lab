# Workflow timers / scheduled wakeups — Test Plan

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-028

## 1. Scope & objective

Acceptance/black-box QA pass over K-028 as **shipped** (v3 of `docs/plans/workflow-timers.md`,
plan-gated by `analyst` across three passes plus a post-implementation diff re-gate — verdict
*approve with suggestions*, `docs/reviews/workflow-timers.md`). The feature: a `wait`/`human`
step may declare `config.waitForSeconds`/`waitUntil`, and — since v3 — must then also declare an
escalation transition guarded on `{"kind":"cmp","path":"ctx.timerFired","op":"eq","value":"<own
key>"}`. A periodic sweep (`Services.sweep_due_workflow_runs`, exposed as `POST
/workflow-runs/due` and ticked automatically by an in-process `asyncio` task) resumes a due run
through the existing `resume_run_with_ctx` CAS, writing that step-scoped marker atomically.

The unit/integration layer is already extensive and independently re-verified (`analyst`'s diff
gate reran the suite, its own `GRAPH.PROFILE`, and its own mutation test rather than trusting the
implementer's report). This plan does **not** re-derive that coverage — no re-proving
`_wait_due_at`'s arithmetic, no re-running `GRAPH.PROFILE` on `find_due_wait_candidates`, no
re-litigating the CAS-contention proof at the repository layer. It complements that layer at
**acceptance altitude**: drive the real, running FastAPI process over REST, with real wall-clock
time for at least one item, to catch what only shows up when the mechanism is actually exercised
end-to-end — exactly the altitude at which this feature's own history shows static review can be
fooled (see §2, "why this matters more than usual" below).

**Why this matters more than usual for this item.** An earlier version of this mechanism (v2's
"unconditional fallback" invariant) was gated *approved* by two separate `analyst` passes, then
found — only when an agent actually tried to drive it end-to-end — to make the feature
completely non-functional (the step could never park in the first place). Static re-reading of
v3's design is not repeated here as a check on its own logic (that is `analyst`'s job and it has
already been done, twice, independently); this plan's job is to confirm the **running system**
actually behaves as v3 now claims, since that is exactly the altitude at which v2 was wrong.

## 2. References

- `falkor-chat/docs/BACKLOG.md` §K-028 — original requirements, SLA/escalation motivation, the
  "injected clock + CAS-contention" test-strategy sketch.
- `falkor-chat/docs/plans/workflow-timers.md` (v3) — full design, especially the v2→v3 revision
  note and §6 (test strategy already covered offline).
- `falkor-chat/docs/reviews/workflow-timers.md` — three plan-gate passes + post-implementation
  diff re-gate, verdict *approve with suggestions*.
- `falkor-chat/docs/DESIGN.md` §6.1–§6.3 — updated run-model narrative (spot-checked, §9 below).
- `falkor-chat/server/falkorchat/{services,executor,repository,api,app,config,proof_defs}.py` —
  read directly (source of truth; `cpg_falkorchat` is stale, see below).
- `falkor-chat/server/tests/{test_workflow_timers,test_executor_process,test_repository,
  test_process_flow}.py` — existing offline coverage, read to avoid duplicating it.

`CPG: considered, not relevant — cpg_falkorchat was built 2026-08-17T00:40:42Z; the entire K-028
delivery (services.py/executor.py/repository.py/api.py/app.py/config.py changes) postdates it.
Read the live source directly instead (as this plan's own citations show), which is the normal
mode for this pass regardless.`

## 3. Risk assessment & prioritization

| Risk | Why | Priority |
|---|---|---|
| A mechanism that "reads correct" but doesn't actually work end-to-end (the v2 precedent) | Proven failure mode for this exact feature | Highest |
| Additive-only guarantee breaks for existing, unmodified defs | Backward-compat is the item's own hard constraint | Highest |
| Concurrent human-signal vs. timer-sweep race double-resumes a run | Backlog's own named hard requirement | Highest |
| "Not yet" resume pattern stops composing with a timer on the same step | The exact defect class v2 shipped with | Highest |
| Cross-step marker leakage (stale `timerFired` escalates the wrong step) | Self-caught during design; worth an empirical, not just unit-level, check | High |
| The **automatic** periodic tick never actually fires in a real process | The plan's own lifespan test uses a stub — real end-to-end tick is unverified until now | High |
| Batch sweep doesn't evaluate multiple due runs / ignores `limit` | Stated operational requirement | Medium |
| Disabled-engine posture leaks sweep behavior, or a default run is affected | Safety-default requirement | Medium-High |
| Publish-time validation escape hatch (timer without escalation transition still publishes) | Would silently reintroduce the v2-class bug | High |
| Docs (`DESIGN.md` §6.1–§6.3) still describe the old "no scheduler" world | Reader-facing correctness, low functional risk | Low |

**Deliberately not tested here** (covered elsewhere, restated so omission reads as a choice):
`_wait_due_at` pure-function edge cases (offline, `test_workflow_timers.py`); the repository
query's `GRAPH.PROFILE` shape (offline, `test_repository.py`, re-verified independently by
`analyst`'s diff gate); a genuine multi-process/multi-worker race (out of scope for both the
plan and this environment — one uvicorn process is what's running); the MCP-mirror surface
(explicitly out of scope for the plan itself, not built); LLM-guard timer variants (Direction B,
rejected by the plan, not built).

## 4. Environment & data setup

- FalkorDB `falkordb-dev` (Docker), already running.
- Baseline offline suite confirmed green before touching anything: `pytest -q` → **1529 passed, 3
  deselected** (matches the dispatch brief exactly). This run wipes the shared `reference` graph
  at teardown (documented `AGENTS.md` hazard) — re-seeded immediately after with
  `./scripts/bootstrap_schema.sh acme && ./scripts/seed_workflows.sh acme`, re-verified with
  `./scripts/verify_workflows.sh acme` → `OK`.
- A fresh, throwaway workspace, `ws:qa028` (bootstrapped via `bootstrap_schema.sh qa028`), used
  for every live REST test below instead of `acme` — keeps this pass's run/def data fully
  isolated from `acme`'s. `access-request@v1`/`triage@v1` materialized into it via
  `seed_workflows.sh qa028` (create-only, additive; also touches the shared `reference` graph the
  same way `acme`'s own seeding already does — unavoidable, since `publish_def` has no per-graph
  seam, `AGENTS.md`'s own documented note).
- Server started manually (not `start_server.sh`, to control the sweep interval and avoid
  re-seeding `acme`): `FALKORCHAT_WS_ID=qa028 FALKORCHAT_ENABLE_AGENT=1
  FALKORCHAT_WORKFLOW_ENABLED=1 FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S=5
  FALKORCHAT_EMBEDDING_DIM=1024 FALKORCHAT_TRIGGER_DEF_KEY=triage
  FALKORCHAT_TRIGGER_DEF_VERSION=v1 FALKORCHAT_OPENCODE_CONFIG=$HOME/.config/opencode/opencode.json
  uvicorn falkorchat.app:app --host 127.0.0.1 --port 8010`. A 5s sweep interval (not the 30s
  default) is deliberate — it makes TP-006 (the automatic-tick item) provable in well under a
  minute of real wall-clock time without weakening what's being proven.
- No LLM call is exercised by any test below (every def used is `kind:'process'`, `cmp`-only
  guards) — `ModelGateway` construction is offline regardless, so LM Studio's actual availability
  is irrelevant to this pass.
- Custom throwaway defs for TP-004/006/007/008/009/010, published under keys that cannot collide
  with any shipped def (`qa028-timer-wait`, `qa028-timer-human`, `qa028-not-yet`,
  `qa028-two-timers`, `qa028-batch`), each `version: "v1"`. These additively land in the shared
  `reference` graph (new keys only) — same posture `seed_workflows.sh` already has for its own
  two defs, not a new kind of shared-state mutation.

## 5. Entry / exit criteria

**Entry**: offline baseline green, `reference`/`ws:acme` verified in sync, `ws:qa028` bootstrapped
and seeded, server up and `/health` 200.

**Exit**: every test item below resolved pass/fail/blocked with evidence in the test report; any
defect found is reproducible from written steps; `ws:acme`/`reference` left in the same verified
state they were found in (re-checked at teardown); the qa028 server process and workspace are
scratch and may be torn down without further obligation.

## 6. Test items

| ID | Title | Type | Priority |
|---|---|---|---|
| TP-001 | Additive-only guarantee: unmodified `access-request@v1` runs byte-identically | Regression/e2e | Highest |
| TP-002 | Publish-time validation: timer key without escalation transition is rejected (400) | Contract | High |
| TP-003 | Publish-time validation: timer + correctly-shaped escalation transition publishes cleanly | Contract | High |
| TP-004 | Core happy path — `wait` step + `waitForSeconds`, manual sweep before/after due | e2e | Highest |
| TP-005 | Core happy path — `human` step + `waitForSeconds`, manual sweep before/after due | e2e | High |
| TP-006 | The automatic periodic sweep actually ticks in the real running process | e2e | High |
| TP-007 | "Not yet" pattern survives on a timer-bearing step | e2e | Highest |
| TP-008 | Concurrent-resume guarantee — human vs. sweep race, both orderings | e2e | Highest |
| TP-009 | Cross-step marker leakage does not occur (two sequential timer steps) | e2e | High |
| TP-010 | Batch sweep evaluates multiple due runs; `limit` is respected | Functional | Medium |
| TP-011 | Disabled-engine posture: `/workflow-runs/due` 503s; default app unaffected | Functional | Medium-High |
| TP-012 | Doc spot-check: `DESIGN.md` §6.1–§6.3 matches shipped behavior | Exploratory/doc | Low |

### TP-001 — Additive-only guarantee
**Preconditions**: `access-request@v1` materialized in `ws:qa028`.
**Steps**: `POST /workflow-runs {defKey:access-request,version:v1,maxSteps:24,
ctx:{request:{role:"contractor"}}}`; drive through `submit`→`route`→`approval` (input
`{"decision":"approve"}`) →`provision` (input `{"provisioned": true}`) →`activate`; at each parked
state, call `POST /workflow-runs/due` and confirm it does **not** touch this run (it declares no
timer key).
**Expected**: run reaches `done`; every step/park matches the pre-K-028 `access-request` behavior
documented in `DESIGN.md` §6.3; the sweep never appears in this run's history.

### TP-002 — Publish rejects a timer without the escalation transition
**Steps**: `POST /workflow-defs` with a minimal one-`wait`-step def declaring
`config.waitForSeconds: 5` and **no** matching `ctx.timerFired` transition.
**Expected**: `400 WorkflowDefSpecError`, nothing written (`GET /workflow-defs/<key>` confirms
absence).

### TP-003 — Publish accepts the correctly-shaped escalation transition
**Steps**: same shape as TP-002 plus the canonical transition; publish; materialize into
`ws:qa028`.
**Expected**: `201`, def readable back with the timer config and escalation transition intact.

### TP-004 — `wait` step timer happy path (manual sweep)
**Steps**: publish/materialize `qa028-timer-wait@v1` (one `wait` step, `waitForSeconds: 3`, its
own escalation transition to a terminal step); start a run; confirm `GET /workflow-runs/{id}` →
`waiting`; immediately `POST /workflow-runs/due` → confirm run **still** `waiting`, unresumed;
sleep past the 3s due time (real wall clock); `POST /workflow-runs/due` again.
**Expected**: second sweep call reports this run in `resumed`; `GET /workflow-runs/{id}` → `done`
(or whatever the escalation branch's terminal is); `GET /workflow-runs/{id}/step-runs` shows the
escalation transition was taken, not the (absent) domain one.

### TP-005 — `human` step timer happy path (manual sweep)
Same shape as TP-004, `qa028-timer-human@v1`, step `type: "human"` instead of `"wait"` — proves
the mechanism is genuinely shared between both step types, not just `wait`.

### TP-006 — Automatic sweep tick fires in the real process
**Preconditions**: server running with `FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S=5`.
**Steps**: start a run against `qa028-timer-wait@v1` (or a fresh run of the same def) with
`waitForSeconds: 3`; park it; **do not** call `POST /workflow-runs/due` manually at all; poll
`GET /workflow-runs/{id}` every second.
**Expected**: within ~2 sweep-interval ticks (≤ 13s) the run transitions to its terminal state on
its own, with no manual sweep call — proving the `asyncio` periodic task (gated on
`WORKFLOW_ENABLED`, §3.6 of the plan) is actually wired and running in this process, not just
provable via the plan's own stubbed lifespan smoke test.

### TP-007 — "Not yet" pattern survives on a timer-bearing step
**Steps**: publish `qa028-not-yet@v1`: a `wait` step shaped like `proof_defs.py`'s `provision`
(accepts `{"provisioned": false}` as a no-op resume, `{"provisioned": true}` as the real advance),
**plus** `waitForSeconds: 5` and its escalation transition. Start a run, park it. Before 5s
elapses, `POST /workflow-runs/{id}/input {"provisioned": false}`.
**Expected**: run **re-parks** (still `waiting`, same step) — does **not** escalate. Then let the
due time pass and sweep (manual or automatic): run now resumes via the **escalation** branch.
Confirms the exact composition v2 broke.

### TP-008 — Concurrent-resume guarantee, both orderings
**Steps (ordering A — human first)**: park a run on a step with both a domain-signal transition
(`ctx.provisioned==true`) and a timer/escalation transition, due time already passed;
`POST /workflow-runs/{id}/input {"provisioned": true}` immediately followed by
`POST /workflow-runs/due`.
**Expected A**: run advances via the domain branch; the sweep call reports this run `raced`
(or absent from `resumed`); `ctx.timerFired` is never set; no duplicate `StepRun`s
(`GET .../step-runs` shows one advance past the wait step, not two).
**Steps (ordering B — sweep first)**: mirror with the sweep issued first, then the human input.
**Expected B**: run advances via escalation, `ctx.timerFired` set to the step's key; the
subsequent human input gets `409 WorkflowRunNotWaitingError`; still exactly one advance in the
step-run history.

### TP-009 — Cross-step marker leakage regression
**Steps**: publish `qa028-two-timers@v1`: two sequential `wait` steps, A then B, each with its
own short `waitForSeconds` and its own canonical `ctx.timerFired=="<own key>"` escalation
transition, A's escalation edge leading to B. Start a run, let A's due time pass, sweep (resolves
into B). Immediately check B's state.
**Expected**: B is `waiting` (parked), **not** already escalated — the stale `ctx.timerFired=="A"`
value must not satisfy B's own guard (`== "B"`). Then let B's own due time pass and sweep again —
confirm B *does* now escalate correctly on its own due time.

### TP-010 — Batch sweep + `limit`
**Steps**: publish `qa028-batch@v1` (one timer-bearing `wait` step); start two runs against it,
both due; `POST /workflow-runs/due {"limit": 1}`.
**Expected**: response's `checked`/`due` reflect both candidates were seen but only one is
resumed (or the `limit` caps candidates read — confirm which per the actual response shape); a
second sweep call with the same/no limit clears the remaining due run; both eventually resume.

### TP-011 — Disabled-engine posture
**Steps**: start a second, throwaway app process with `FALKORCHAT_WORKFLOW_ENABLED=0` (agent
enabled or not, irrelevant) and hit `POST /workflow-runs/due`.
**Expected**: `503` (`WorkflowEngineDisabledError` envelope). Additionally confirm via reading
`app.py` (already done in §2 orientation) that the default `create_app()`/pytest baseline path
never constructs a `sweep_task` — the offline suite's 1529-pass baseline (§4) already
demonstrates the default path is unaffected, so this item's live half only needs the 503 check,
not a second full server boot merely to prove a negative already covered by the baseline run.

### TP-012 — Doc spot-check
Read `DESIGN.md` §6.1 (`wait`/`human` bullets), §6.2 (the "K-028 (shipped)" paragraph), §6.3, and
confirm each statement matches what TP-004–TP-009 actually observed (mechanism, reserved key,
no new `WorkflowRun` property, sweep as one more `resume_run_with_ctx` submitter). Not a full doc
audit (that is `teco`'s closeout job) — a targeted check that nothing reads as stale "no
scheduler" framing.

## 7. Out of scope (restated)

- Re-running `_wait_due_at` unit cases, `GRAPH.PROFILE` re-verification, or the repository-level
  CAS-contention test — already offline-covered and independently re-verified by `analyst`'s diff
  gate.
- A true multi-process race (single uvicorn process is what's available here).
- The MCP-mirror surface and the LLM-guard Direction B variant — neither was built; not a gap in
  this pass, a gap in the plan's own stated scope.
- Full doc audit of `DESIGN.md`/`QUERIES.md`/`BACKLOG.md`/`HISTORY.md` beyond the §6.1–§6.3
  spot-check (TP-012) — `teco`'s milestone-closeout responsibility.
