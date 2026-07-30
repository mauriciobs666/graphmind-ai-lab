# Web API Coverage — Test Plan (K-036, Wave 5 / U10)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-036 (M3.5)

archived 2026-07-29 — K-036 delivered, M3.5 reached; see `docs/plans/web-api-coverage-coordination.md`

Black-box acceptance pass against `docs/requirements/web-api-coverage.md` (AC-1..AC-6, committed
scope FR-1..FR-10) and `docs/plans/web-api-coverage.md` §5.2, driving the running `falkor-chat`
app (server + seeded `ws:acme`) as it stands on disk (uncommitted working tree; Waves 1-4
delivered and independently reviewed by `analyst`, `docs/reviews/web-api-coverage-impl.md`
Pass 1-3, verdict on the delivered diff: approve with suggestions, zero blockers). This is Wave 5
(U10) of the plan's build sequence — the only unit not yet executed.

## Tooling constraint and equivalence statement

This session has no browser-automation tool (no Playwright/Puppeteer/DevTools MCP wired into this
environment) and no way to render/interact with a live page. Per the task's own allowance
("browser-only, or the closest black-box equivalent your tooling allows"), this plan substitutes:

1. **Direct REST driving** of the exact endpoints `falkor-chat/web/app.js` calls, in the same
   sequence, with the same payload shapes, that a person clicking through the UI would trigger
   (verified by reading `app.js`/`index.html` in full beforehand — see call sites cited per
   scenario below).
2. **Static confirmation of the rendering path** for each response — reading the specific
   `render*`/`open*` function in `app.js` that would consume each REST response, to confirm the
   browser would display exactly what the AC asks for. This mirrors, and does not re-litigate,
   what `analyst`'s three review passes already verified about the same code — it is not a fourth
   code review, only a targeted check that a given endpoint's actual live response shape matches
   what the already-reviewed renderer expects.
3. **Timing checks** (AC-2's 5s bar) via wall-clock `curl` round trips against `POLL_MS = 3000`,
   since a real browser's poll timer isn't available to observe directly.

This is a genuine gap against "browser-only" — flagged explicitly in the test report, not silently
substituted.

## Pre-flight

1. `docker ps` — confirm `falkordb-dev` is up.
2. `curl localhost:1234/v1/models` — confirm LM Studio is reachable (needed for Pass A's `triage`
   agent steps).
3. `./scripts/verify_workflows.sh acme` — check `reference`/`ws:acme` sync. If `MISSING`, run
   `./scripts/seed_workflows.sh acme` (create-only — safe). If `DIVERGE`, do **not** re-seed;
   record the drift (this is expected, pre-existing K-034-territory drift per the task brief) and
   proceed — a desynced def is in fact useful as AC-6's negative case.
4. Confirm no `uvicorn` already bound to `:8000`.

## Pass A — server started with default config (`FALKORCHAT_TRIGGER_DEF_KEY=triage`, `_VERSION=v1`)

Start: `./scripts/start_server.sh` (defaults; `FALKORCHAT_WORKFLOW_ENABLED=1`,
`FALKORCHAT_ENABLE_AGENT=1` per its own defaults).

### AC-1 — full M3 story, no terminal/curl/reload (demo path)

Drive, in order, the exact calls `app.js` issues for this story:
1. `GET /workflow-defs` (`loadDefsList`) — confirm `triage`/`access-request` both listed.
2. `GET /workflow-defs/triage/versions/v1` (`selectDef`) — confirm steps/transitions render
   (3 steps, 2 transitions per `scripts/seed_workflows.sh`).
3. `POST /threads/demo-welcome/messages` with `{"text": "@assistant <request>", "mentions":
   ["assistant"]}` (`postMessage`) — the `@mention` that starts a run.
4. `GET /threads/demo-welcome/workflow-runs` (`updateRunCue`, polled every `POLL_MS`) — confirm a
   new run appears, `defKey: "triage"`, and its status progression is observable across polls
   (`running` → `waiting` at the `intake` step).
5. `GET /workflow-runs/{id}` + `.../step-runs` (`refreshRunPanel`) — confirm `status: "waiting"`,
   `atStepKey: "intake"`.
6. Answer the parked step via a **plain chat reply** (`POST /threads/demo-welcome/messages`, no
   re-mention) — per plan §2.4, this resumes the run without a re-`@mention`.
7. Continue polling `GET /workflow-runs/{id}` until `status` reaches a terminal value
   (`done`/`failed`) — confirm the run reaches a terminal state within a bounded number of
   clarifying rounds (`intake.config.maxIterations: 4`).

Pass condition: every step above succeeds with no direct FalkorDB/Cypher/file intervention — only
the REST calls a browser session would make — and the response at each step is confirmed to be
exactly what the corresponding `render*` function needs (per the equivalence statement above).

### AC-3 — failure + readable reason (attempted in Pass A, deferred to Pass B if not convenient)

Per plan §7 risk #1 and the requirements' own framing, forcing a `failed` run needs either
step-budget exhaustion or an engine fault. Both demo defs' `human`/`wait` steps are explicitly
budget-exempt while parked (`executor.py` `_drive_loop` OUTCOME B — "No budget check here by
design," D-C), and neither def contains a re-loop (`OUTCOME C`) — so a **chat-triggered** run
cannot exhaust budget through any normal interaction with either def. If this holds on
inspection, AC-3 is deferred to Pass B, where `access-request`'s deterministic branch shape allows
a **directly-started** run (`POST /workflow-runs` with a deliberately small `maxSteps`) to
guarantee a budget failure in one step — noting explicitly that a directly-started run has no
`TRIGGERED_BY` edge and so cannot be reached via the thread cue, only via the same
`GET /workflow-runs/{id}` read the panel itself uses.

### AC-4 — thread participants, both kinds distinguishable

`GET /threads/demo-welcome/participants` (`openParticipants`) — confirm the demo channel's roster
returns both the seeded `User` (`u1`/"Demo User") and `Agent` (`assistant`/"Assistant"), and that
the response shape (`kind: "User"|"Agent"`) drives `openParticipants`'s `.badge`/"AI" rendering
for the `Agent` row only.

### AC-5 — default layout no busier than before this feature

Read `index.html`'s `<header>` and thread column, confirm (per plan §3.3 v3's AC-5 reading):
- The only always-visible additions are `#defs-btn` ("Workflow defs"), `#participants-toggle`
  ("Participants"), and `#readiness-badge` — each sharing CSS with an existing header element.
- None of `#defs-panel`, `#run-panel`, `#participants-row`, `#run-cue` render or fetch content
  without a click/expand/non-empty-poll-result — confirmed by reading the `display:none`
  defaults and each panel's open function.
- On a fresh load (`guard(loadChannels); guard(loadReadiness);` — the only two calls `app.js`
  fires unprompted), only `GET /channels` and `GET /workspaces/acme/readiness` fire — confirmed by
  reading the bottom of `app.js` (the "initial load" block) and cross-checked against actual
  server request logs during the session.

### AC-6 — readiness banner, synced vs. desynced

`GET /workspaces/acme/readiness` (`loadReadiness`) against the workspace as it stands (post
pre-flight): confirm the banner/panel correctly reports the synced def (`triage`) as in-sync and
names the desynced def (`access-request`, if pre-flight found it diverging) with a human-readable
`problems` string, matching `renderReadinessBadge`/`renderReadinessPanel`'s consumption of the
response shape.

## Restart

Stop the Pass-A server; restart with `FALKORCHAT_TRIGGER_DEF_KEY=access-request
FALKORCHAT_TRIGGER_DEF_VERSION=v1 ./scripts/start_server.sh`.

## Pass B — restarted, pointing at `access-request`/`v1`

### AC-2 — structured-input timing (≤5s, no reload)

1. `POST /threads/demo-welcome/messages` with an `@assistant` mention — chat-triggers an
   `access-request` run (now the process-kind def, since `TRIGGER_DEF_KEY` points at it).
2. Poll `GET /threads/demo-welcome/workflow-runs` / `GET /workflow-runs/{id}` until
   `status: "waiting"` at a **structured**-input step (`submit`, `fields: ["request"]`).
3. `POST /workflow-runs/{id}/input` with `{"input": {"request": {"role": "...", ...}}}` (the
   `renderWaitingForm`/`submitRunInput` shape) — record wall-clock time of the POST.
4. Poll `GET /workflow-runs/{id}` at the app's own cadence and record wall-clock time the response
   first reflects the state change (`atStepKey` advanced past `submit`, or `status` changed).
   Pass condition: elapsed ≤ 5s. (Plan §3.3.3 additionally re-polls immediately on submit success,
   so this should read well under the 3000ms worst case; confirm.)
5. Repeat once more at the `approval` step (`fields: ["decision"]`, `expects` constrained —
   confirms the `<select>` rendering path, `renderWaitingForm`'s `options` branch) for full FR-6
   coverage alongside the timing check.

### AC-3 — if not covered in Pass A

Force via `POST /workflow-runs` with `{"defKey": "access-request", "version": "v1", "maxSteps": 1,
"trace": true}`, then `POST /workflow-runs/{id}/input` with a `submit` payload whose `request.role`
is present (satisfies the `submit→route` guard) — this drives two consecutive `OUTCOME A` advances
(`submit→route`, `route→approval|provision`) in one loop iteration, which should exceed
`maxSteps: 1` and land the run in `status: "failed"` deterministically. Confirm
`GET /workflow-runs/{id}`'s `ctx` field parses to `{"error": "step budget exceeded"}` (per
`executor._fail_budget`), matching exactly what `renderRunFailure` reads (`JSON.parse(run.ctx
).error`).

## Restart back to default

`FALKORCHAT_TRIGGER_DEF_KEY=triage FALKORCHAT_TRIGGER_DEF_VERSION=v1` (i.e. no override —
`./scripts/start_server.sh` with no env override) — every other environment (pytest, future manual
checks) assumes this default per the task brief and `docs/plans/demo-environment-bringup.md`.

## Known, accepted items — not re-filed if observed

- **m6** — overlapping `refreshRunPanel` calls (poll tick vs. post-submit) can transiently render a
  stale panel state; self-heals within one ≤3s poll tick, no data loss.
- **m7** — an external same-step waiting→running→waiting round-trip between two poll ticks could
  leave the form stale; same self-healing property.
- Pre-existing `reference`/`access-request@v1` drift (K-034-territory), used here as AC-6's
  negative case rather than treated as a new defect.
- `wait`/`human` steps are signal-driven, not timer-driven (D-C, by design).
- `prompt`/`tool`/`message` step types raise `NotImplementedError` by design (D-E) — neither demo
  def uses them, not exercised here.

## Deliverable

Results, defects (if any, severity-ranked with repro steps), and a verdict recorded in
`docs/test-reports/web-api-coverage-report.md`.
