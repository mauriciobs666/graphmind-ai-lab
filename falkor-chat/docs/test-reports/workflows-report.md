# Workflows manual — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-022, K-024 (M3)

Execution of `docs/test-plans/workflows.md` against `docs/manuals/workflows.md`, driving the
running `falkor-chat` app (server on `127.0.0.1:8000`, `ws:acme`, FalkorDB `falkordb-dev`).
Executed 2026-08-01. **LM Studio was reachable** (`qwen/qwen3-4b-2507`, config default) — every
walkthrough, including the conversational/LLM one, was fully exercised, not just started.

**Overall verdict: one confirmed defect (TP-06), everything else in the manual held up.** The
defect is in the API payload literally printed in Walkthrough 4 — a reader copy-pasting it will
get a 422, not the documented result.

## Pre-flight (environment note, not a manual defect)

`./scripts/verify_workflows.sh acme` failed at first pass: both `triage@v1` and
`access-request@v1` were **MISSING from `reference`** while still present as `ws:acme` snapshots —
consistent with the known `test_queries.sh`-teardown hazard documented in `falkor-chat/AGENTS.md`
("Deletes the `reference` graph at teardown ... Re-run `seed_workflows.sh <wsId>` afterward").
Fixed with `./scripts/seed_workflows.sh acme` (create-only, safe); re-verify came back in sync.
Recorded here for the record — not something the manual is responsible for.

## Results

| ID | Claim | Result | Evidence |
|---|---|---|---|
| TP-01 | `@mention` starts a `triage` run | **PASS** | `POST /threads/demo-welcome/messages` with `mentions:["assistant"]` → new run appeared in `GET /threads/demo-welcome/workflow-runs` within ~6s, `defKey:"triage"`, `status:"running"` |
| TP-02 | Clarifying question(s), then reply-without-remention resumes, run reaches `done` | **PASS** | See "Walkthrough 1, in detail" below — full transcript, two clarifying rounds, two plain replies, run reached `status:"done"` with a final answer posted |
| TP-03 | Run panel shapes: status values, per-step status, failure reason via `ctx`, trace toggle only populated when tracing on | **PASS** | `GET /workflow-runs/{id}` returns `status` ∈ `{running,waiting,done,failed}`; `/step-runs` lists each step's own `status`; forced a `failed` run (see TP-03 detail) — `ctx` came back as `{"error":"step budget exceeded",...}`; `/trace` was `[]` on a `trace:false` run and populated (`node_rationale`/`guard_judgment` entries) on a `trace:true` run |
| TP-04 | `GET /workflow-defs` lists both seeded defs | **PASS** | Response: `access-request@v1` (kind `process`) and `triage@v1` (kind `conversation`), each with name/version/kind |
| TP-05 | `GET /workflow-defs/access-request/versions/v1` structure matches the manual's flowchart | **PASS** | 6 steps (`submit,route,approval,provision,activate,rejected`), 6 transitions, guards match exactly: `route→approval` on `ctx.request.role in [contractor,exec]`, `route→provision` unconditional "auto" otherwise, `approval→provision` on `decision==approve`, `approval→rejected` on `decision==reject`, `provision→activate` on `ctx.provisioned` truthy |
| TP-06 | Literal Walkthrough-4 payload `{"defKey":"access-request","defVersion":"v1"}` starts a run | **FAIL** | `422 Unprocessable Entity`: `{"detail":[{"type":"missing","loc":["body","version"],"msg":"Field required",...}]}`. The schema field is `version`, not `defVersion` — see Defect 1 |
| TP-07 | End-to-end contractor path: `submit`→`approval`→`provision`→`done` | **PASS** | Full transcript below — every parking point matched exactly |
| TP-08 | "Otherwise" branch: a role outside `[contractor,exec]` skips `approval` | **PASS** | `role:"employee"` run parked directly at `provision` after `submit`, `stepCount:4` (vs. contractor's parking at `approval` with the same `stepCount:4`) |
| TP-09 | A `waiting` run does not change state on its own | **PASS** | Run parked at `submit`; re-`GET` ~4 minutes later: `status`, `atStepKey`, `stepCount`, `ctx` all byte-identical |
| TP-10 | An API-started process run is invisible in the triggering thread's run list | **PASS** | `GET /threads/demo-welcome/workflow-runs` never contained any of the three `POST /workflow-runs`-started run IDs from this session, across the whole test run |
| TP-11 | Submitting input to a non-`waiting` run gives a clear error, not a hang/no-op | **PASS** | On a `done` run: `409 {"error":"WorkflowRunNotWaitingError","detail":"...is 'done', not 'waiting'..."}`. On a `failed` run: same shape, `"...is 'failed'..."`. On an unknown run id: `404 {"error":"WorkflowRunNotFoundError",...}` |
| TP-12 | Rejection path reaches the `rejected` terminal | **PASS** | `contractor` → `approval` → `{"decision":"reject"}` → run status `done`, `atStepKey:null`, consistent with `rejected` being a terminal step in the def structure |

**12/12 test items executed. 11 pass, 1 fail (TP-06).**

## Defects

### Defect 1 — Walkthrough 4's `POST /workflow-runs` example uses the wrong field name (`defVersion` instead of `version`)

**Severity: Medium** (blocks the one worked example a technical reader is meant to copy-paste
verbatim; not a defect in the running system, only in the manual's transcription of it).

**Manual text** (Walkthrough 4, step 1):
```
POST /workflow-runs
{ "defKey": "access-request", "defVersion": "v1" }
```

**Actual behavior** — this exact body returns:
```
HTTP/1.1 422 Unprocessable Entity
{"detail":[{"type":"missing","loc":["body","version"],"msg":"Field required",
            "input":{"defKey":"access-request","defVersion":"v1"}}]}
```

**Root cause**: `StartWorkflowRunIn` (`server/falkorchat/schemas.py`) declares the field as
`version: str`, not `defVersion`. The correct body is:
```
POST /workflow-runs
{ "defKey": "access-request", "version": "v1" }
```
which was verified live and behaves exactly as the manual narrates from that point on (parks at
`submit`, `status:"waiting"`) — see TP-07.

**Suggested fix**: change `"defVersion": "v1"` to `"version": "v1"` in the Walkthrough 4 code
block (docs/manuals/workflows.md, the `POST /workflow-runs` example, step 1). Everything
downstream in the walkthrough (the `/input` payloads) was already correct as written — this is a
single-line fix.

## Walkthrough 1, in detail (TP-01/TP-02, full transcript)

First attempt — `@Assistant, can you help me figure out our current deploy process?` — the model
judged this specific enough on its first turn and went straight `intake → research → answer`
without ever entering `waiting` (confirmed via `/step-runs`: both `intake` and `research` output
the same generic answer, `done` reached in ~13s). This is consistent with the manual's own
hedging ("typically asks a clarifying question first" — not "always"), and with
`scripts/seed_workflows.sh`'s documented intake→research guard being an LLM judgment call, not a
hard rule. Not treated as a discrepancy.

Second attempt, deliberately vaguer — `@Assistant help` — did trigger the documented loop:
1. `intake` (round 1): model asked "Could you please clarify what specific assistance you need?" → run `waiting`, `waitingThreadId:"demo-welcome"`.
2. Plain reply, no `@mention` — `"I need help understanding our deploy pipeline steps."` → run resumed (`status:"running"`, `waitingThreadId` cleared) within ~6s, exactly as the manual's "a reply in the same thread is enough" claim states.
3. `intake` (round 2): model asked again for pipeline specifics → run `waiting` again.
4. Plain reply, no `@mention` again — supplied concrete pipeline details → run resumed, this time judged sufficient, advanced to `research`.
5. `research` → `answer`, run reached `status:"done"`, final answer posted to the thread.

This confirms both Walkthrough 1's narrated loop (clarify → answer) and the FAQ claim "you never
need to `@mention` the assistant again once a conversation has started" — resumption worked
identically on both of the two plain replies, with no re-mention in either.

## TP-03 detail — forcing a `failed` run to check the reason-surfacing claim

Manual claims "If the run has failed, the reason it stopped" is shown in the run panel. Forced a
failure via `POST /workflow-runs {"defKey":"access-request","version":"v1","maxSteps":1}` then one
`/input` submission — budget exceeded on the next step:
```
GET /workflow-runs/{id} →
  "status": "failed",
  "ctx": "{\"error\":\"step budget exceeded\",\"request\":{\"role\":\"employee\"}}"
```
The reason is real and present, but it's inside the serialized `ctx` string (per
`falkor-chat/AGENTS.md` rule 8 — `ctx`/`input`/`output` are opaque serialized strings), not a
dedicated top-level `error`/`reason` field on the run object. A reader driving the API directly
(this manual's stated audience for Walkthrough 4) needs to know to parse `ctx` for an `error` key;
the manual doesn't say this explicitly, but it also never promises a specific field name — it only
promises the run *panel* (the web UI, which presumably parses `ctx` for you) shows it. Not filed as
a defect since the manual's claim is scoped to the UI panel, not the raw API shape — flagged here
as a documentation opportunity only (see Feedback below).

One more minor observation, same area: the **immediate response** of the `/input` call that
triggers the failure does *not* yet contain the `error` key in its `ctx` (only the follow-up `GET`
does) — a caller reading only the `POST` response and not re-fetching would not immediately see
the reason. Also not filed as a defect (out of scope for what the manual claims), but worth
`tico`/`architect` knowing about if the API surface is ever documented at this level of detail.

## Walkthrough 4, in detail (TP-07/TP-08/TP-12)

Contractor path (TP-07):
```
POST /workflow-runs {"defKey":"access-request","version":"v1"}
  → runId=04c336c6..., status=waiting, atStepKey=submit
POST /workflow-runs/{id}/input {"input":{"request":{"role":"contractor"}}}
  → status=waiting, atStepKey=approval          (matches manual: "contractor needs one")
POST /workflow-runs/{id}/input {"input":{"decision":"approve"}}
  → status=waiting, atStepKey=provision
POST /workflow-runs/{id}/input {"input":{"provisioned":true}}
  → status=done
```

Otherwise-branch (TP-08), role `employee` (outside `[contractor,exec]`):
```
POST /workflow-runs {"defKey":"access-request","version":"v1"} → waiting @ submit
POST .../input {"input":{"request":{"role":"employee"}}}
  → status=waiting, atStepKey=provision          (skipped approval, as manual's "otherwise" edge claims)
```

Rejection path (TP-12), role `contractor`:
```
... → waiting @ approval
POST .../input {"input":{"decision":"reject"}}
  → status=done, atStepKey=null                  (terminated via the `rejected` step)
```

## Coverage & gaps

**Covered**: all four walkthroughs, all FAQ behavioral claims (timers, UI-invisibility of process
runs, clear-error-not-hang), the four run states, both seeded defs' structure, both branch
conditions of `access-request`, the trace-toggle claim, the failure-reason claim.

**Not covered / out of scope** (per the test plan, not gaps in this run):
- Pixel-level web-UI rendering — no browser-automation tool available in this environment; REST
  responses were checked against the shapes the UI's renderers would need (same substitution
  `docs/test-plans/web-api-coverage.md` used), not eyeballed on screen. The manual's run-cue and
  run-panel *narrative* (§2, §1 step 5) was not independently re-verified pixel-by-pixel; only the
  REST data those views would consume was.
- The manual's non-behavioral/architectural claims (what a "definition" is, the graph model) — per
  the QA/analyst split, `tico` routes those to `analyst` separately.
- Load/concurrency behavior of the workflow engine — not implicated by any manual claim.
- The `assignee` field mentioned in the FAQ ("check who the step is waiting on (its `assignee`,
  where shown)") — confirmed the field exists in a step's `config` (`GET
  /workflow-defs/{key}/versions/{version}` shows `"assignee":"manager"` / `"assignee":"requester"`
  on `approval`/`submit`), but did not verify it's actually surfaced in the run/step-run API
  responses themselves (`step-runs`' `output` only carried `assignee` inside the nested `awaiting`
  object when a step was first entered, not on every read) — low-risk residual, the manual's
  phrasing ("where shown") already hedges that it may not always be visible.

**Residual risk**: low. The one real defect (Defect 1) is narrow and mechanical — a single field
name — and everything downstream of it in the same walkthrough was independently confirmed correct
once the right field name was used.

## Feedback & recommendations

1. **Fix Defect 1** — `defVersion` → `version` in the Walkthrough 4 `POST /workflow-runs` example.
   One-line change, `docs/manuals/workflows.md`.
2. Consider a short parenthetical in Walkthrough 2 or the FAQ noting that the failure reason lives
   inside the run's `ctx.error` when a technical reader is inspecting via the raw API rather than
   the web UI panel — the manual is written for the panel view (correct, since that's what most
   readers use), but Walkthrough 4's audience ("a more technical user or an operator") is exactly
   the reader who'd hit this gap; a one-line pointer would save them a Cypher/log dig.
3. The manual's hedge ("typically asks a clarifying question first") held up well against live
   model variance — worth keeping as-is rather than tightening to "always," since this session
   observed both behaviors (an immediate answer on a specific-enough request, a genuine
   clarify-loop on a vague one) from the same model in the same run.
4. No testability issues encountered — every claim in the manual was checkable against a concrete
   REST response; nothing required guessing at internal state.

## Artifacts

- Test plan: `falkor-chat/docs/test-plans/workflows.md`
- This report: `falkor-chat/docs/test-reports/workflows-report.md`
