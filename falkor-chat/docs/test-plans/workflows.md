# Workflows manual — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-022, K-024 (M3)

Verifies the walkthroughs and behavioral claims in `docs/manuals/workflows.md` (a brand-new,
never-verified `tico`-authored manual) against the running `falkor-chat` app. Per the
`qa-engineer` charter, the manual's own walkthroughs *are* the spec — each step below is a test
item and the "expected result" is exactly what the manual claims. This plan does **not** re-check
the manual's factual/architectural claims (e.g. its description of the graph model) — that is
`analyst`'s separate pass, routed by `tico`.

## Environment

- FalkorDB `falkordb-dev` (Docker), already running with `ws:acme` + `reference` pre-existing.
- Found `reference`/`ws:acme` workflow-def sync **broken** at pre-flight (`verify_workflows.sh
  acme` → both defs `MISSING` from `reference`, present in `ws:acme` — consistent with the known
  `test_queries.sh` teardown hazard documented in `AGENTS.md`). Fixed with `seed_workflows.sh acme`
  (create-only, safe) before testing; re-verified in sync. This is environment repair, not a
  manual defect, and is noted here for the record.
- `seed_demo.sh acme` run (idempotent) — demo `Agent assistant` ("Assistant"), `Channel
  demo-general`, `Thread demo-welcome`.
- Server started manually (not `start_server.sh`, to control ports/logs) with the same env vars
  that script would set: `FALKORCHAT_WS_ID=acme FALKORCHAT_ENABLE_AGENT=1
  FALKORCHAT_WORKFLOW_ENABLED=1 FALKORCHAT_AGENT_ID=assistant FALKORCHAT_AGENT_NAME=Assistant
  FALKORCHAT_TRIGGER_DEF_KEY=triage FALKORCHAT_TRIGGER_DEF_VERSION=v1
  FALKORCHAT_EMBEDDING_DIM=1024`, `uvicorn falkorchat.app:app` on `127.0.0.1:8000`.
- LM Studio reachable at `http://localhost:1234/v1` with `qwen/qwen3-4b-2507` (config default)
  loaded — **an LLM IS available** for this run, so Walkthrough 1's conversational path and the
  "resume without re-mention" FAQ claim are testable end-to-end, not just "run starts."

## Risk assessment / prioritization

Highest risk: the manual is brand new and has never touched the running system, so every literal
request/response shape it shows (field names, endpoint paths, payload keys) is unverified —
that's where a reader following it verbatim would get stuck first. Priority order:
1. The copy-pasteable API payloads in Walkthrough 4 (wrong field name = a reader's first request
   fails) and the endpoint paths throughout.
2. The state-machine claims (status values, `waiting`-only-advances-on-reply, error-not-hang) —
   these are safety/trust claims ("no timers", "clear error, not a hang").
3. The chat-triggered walkthrough (1) and run panel shapes (2) — needs the LLM to fully exercise,
   available here.
4. Cosmetic/UI-only claims (run cue, defs browser table) — verified via the REST responses that
   back them (no browser automation available), same equivalence approach as
   `docs/test-plans/web-api-coverage.md`.

**Explicitly out of scope:** the manual's architectural narrative (what a "definition" is, the
graph topology) — that's `analyst`'s factual-claims pass, not this black-box run. Also out of
scope: actual web-UI pixel rendering (no browser automation tool in this environment) — REST
responses are checked against what the UI's renderers would need, not rendered pixels.

## Test items

| ID | Title | Type | Priority |
|---|---|---|---|
| TP-01 | Walkthrough 1: `@mention` starts a `triage` run | e2e | High |
| TP-02 | Walkthrough 1: LLM asks a clarifying question, reply resumes without re-mention, run reaches `done` | e2e | High |
| TP-03 | Walkthrough 2 / run panel: `GET /workflow-runs/{id}`, `/step-runs`, `/trace` shapes match manual | functional | High |
| TP-04 | Walkthrough 3: `GET /workflow-defs` lists both seeded defs | functional | Medium |
| TP-05 | Walkthrough 3: `GET /workflow-defs/{key}/versions/{version}` structure matches manual's `access-request` flowchart | functional | High |
| TP-06 | Walkthrough 4 payload as literally printed in the manual (`defVersion` key) | contract | High |
| TP-07 | Walkthrough 4 end-to-end: `contractor` role parks at `approval`, then `provision`, then `done` | e2e | High |
| TP-08 | Walkthrough 4 second path: a role outside `["contractor","exec"]` skips `approval` | e2e | Medium |
| TP-09 | FAQ: a `waiting` run does not change state with no input, over an observable interval | functional | Medium |
| TP-10 | FAQ: an API-started process run is invisible to `GET /threads/{tid}/workflow-runs` | functional | Medium |
| TP-11 | Error-path claim: `POST /workflow-runs/{runId}/input` on a non-`waiting` (`done`) run returns a clear error, not a hang/no-op | functional | High |
| TP-12 | Walkthrough 4 rejection path: `approval` step submits `{"decision":"reject"}` → run reaches `rejected` | exploratory (spot-check) | Low |

### TP-01 — `@mention` starts a `triage` run
**Preconditions:** server up, `ws:acme` seeded, `demo-welcome` thread has no active run.
**Steps:** `POST /threads/demo-welcome/messages` `{"text": "@Assistant, can you help me figure out our current deploy process?", "mentions": ["assistant"]}`; poll `GET /threads/demo-welcome/workflow-runs`.
**Expected:** a new run appears with `defKey: "triage"`, status progresses from `running`.

### TP-02 — clarifying question → resume without re-mention → `done`
**Steps:** once run is `waiting`, post a plain reply to the same thread (no `@mention`) with an answer; poll the run until terminal.
**Expected:** manual's claim: "a reply in the same thread is enough... your next message is understood as a reply to it, as long as the run is still `waiting` on that thread." Run should progress and reach `done` (or a bounded number of clarifying rounds, per `intake.config.maxIterations`), with an answer posted into the thread.

### TP-03 — run panel shapes
**Steps:** `GET /workflow-runs/{id}`, `/step-runs`, `/trace` on the TP-01/02 run.
**Expected:** `status` one of `running/waiting/done/failed`; step-runs show each step's own status; if `failed`, reason surfaced (check where — manual doesn't specify field name, so confirm it's discoverable at all, e.g. via `ctx`/`error`).

### TP-04 — defs list
**Steps:** `GET /workflow-defs`.
**Expected:** both `triage@v1` and `access-request@v1` listed with name/version/kind.

### TP-05 — access-request structure
**Steps:** `GET /workflow-defs/access-request/versions/v1`.
**Expected:** steps `submit, route, approval, provision, activate, rejected` (manual's diagram calls the terminal step `activate`, note manual text elsewhere says "reaches `done`" — check for a wording mismatch between "activate" the step name and "done" the run status); transitions match the branch logic (`route` branches on role-needs-approval vs. otherwise; `approval` branches on approved/rejected).

### TP-06 — literal Walkthrough-4 payload
**Steps:** `POST /workflow-runs` with the exact body printed in the manual: `{"defKey": "access-request", "defVersion": "v1"}`.
**Expected (per manual):** starts a run. **Risk:** schema (`StartWorkflowRunIn`) declares the field as `version`, not `defVersion` — if so, this literal payload should fail validation (422), which would be a defect in the manual's copy-pasteable example.

### TP-07 — end-to-end contractor path
**Steps:** `POST /workflow-runs {"defKey":"access-request","version":"v1"}` → confirm parked at `submit`, `waiting`. `POST .../input {"input":{"request":{"role":"contractor"}}}` → confirm parked at `approval`. `POST .../input {"input":{"decision":"approve"}}` → confirm parked at `provision`. `POST .../input {"input":{"provisioned": true}}` → confirm `done`.
**Expected:** exactly the manual's narrated sequence.

### TP-08 — otherwise-branch path
**Steps:** start a second run, submit a role outside `["contractor","exec"]` (e.g. `"employee"`) at `submit`.
**Expected:** run skips `approval`, parks directly at `provision` (manual's "otherwise" edge).

### TP-09 — waiting run is inert
**Steps:** take a `waiting` run (from TP-07 mid-sequence or a fresh one), `GET` it, wait an observable interval (tens of seconds), `GET` again, submit nothing.
**Expected:** status and `atStepKey` unchanged.

### TP-10 — process run invisible in thread run list
**Steps:** `GET /threads/demo-welcome/workflow-runs` (the thread used in TP-01) after TP-07's run exists.
**Expected:** the `access-request` run from TP-07 does NOT appear in this list.

### TP-11 — input on non-waiting run errors clearly
**Steps:** on the TP-07 run once `done`, `POST /workflow-runs/{runId}/input {"input":{"anything":true}}`.
**Expected:** a real error response (4xx with a body), not 200/silent no-op, not a hang.

### TP-12 — rejection path (spot-check, time permitting)
**Steps:** a third run, role `contractor` → `approval` → submit `{"decision":"reject"}`.
**Expected:** run reaches terminal state matching the manual's `rejected` step.

## Entry/exit criteria
**Entry:** server reachable at `/health`, `ws:acme` workflow defs in sync (`verify_workflows.sh`).
**Exit:** all items executed to pass/fail/blocked with evidence (curl output); LLM-dependent items (TP-02) marked per actual LLM availability observed during the run, not assumed from this plan.

## Out of scope
- Browser/UI pixel-level verification (no automation tool available) — substituted with the REST
  responses the UI's renderers would consume, per the `web-api-coverage` plan's precedent.
- The manual's non-behavioral/architectural prose (routed to `analyst` separately by `tico`).
- Load/performance/concurrency testing of the workflow engine.
