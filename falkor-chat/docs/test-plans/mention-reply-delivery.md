# `mention-reply-delivery` — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-039 (M3.5)

## 1. Scope & objective

Acceptance-level pass for K-039's implicit-`post_message`-fallback fix, closing the gap the RCA and
implementation review deliberately left open: both are static/offline artifacts (root-cause
analysis + code review + unit/integration tests against a fake or `ws:test` LLM stub). **Nobody has
yet confirmed, live, against the running demo, that `@mention`-ing the assistant now actually
produces a chat message a user would see.** That is the acceptance criterion this bug was reported
against ("the demo is broken"), and it is the one thing this plan exists to check.

References:
- `falkor-chat/docs/reviews/mention-reply-delivery-rca.md` — root cause: `_run_agent_node` treated
  "model ended on plain text instead of calling `post_message`" as a normal success, discarding the
  answer with no `Message`/`PRODUCED` edge ever created.
- `falkor-chat/docs/reviews/mention-reply-delivery-impl.md` — independent review of the fix in
  `server/falkorchat/executor.py`'s `_run_agent_node`: approve with suggestions, no blockers; 647
  passed/1 deselected offline; `_drive_loop` SHA lock (`71055f756280`) confirmed untouched.
- `falkor-chat/docs/BACKLOG.md` K-039 — tracks scope item 1 (this fix, ✅ delivered 2026-07-31),
  item 2 (declined — full K-027 engine contract), item 3 (CI blind-spot follow-up, still open).
- Code under test: `falkor-chat/server/falkorchat/executor.py` (`_run_agent_node`, uncommitted
  working-tree diff — confirmed via `git status`/`git diff --stat`, not yet committed).

## 2. Risk assessment

The dominant risk is **not** "is the code wrong" — that's already been reviewed and unit-tested.
The risk this pass targets is: **does the fix actually engage when the real system runs against a
live, nondeterministic local LLM** (`qwen/qwen3-4b-2507` via LM Studio), the exact condition the RCA
reproduced the bug under.

- A live LLM's output is not deterministic turn to turn. The offline tests prove the mechanism is
  correct *given* a plain-text, no-tool-call model turn — they use scripted/fake LLM stubs to force
  that shape. They do not prove the live model still exhibits that shape today, nor that the
  executor's real dispatch path (real `Services`, real `ToolRegistry`, real FalkorDB write) behaves
  identically to the `ws:test` live integration test's fixture-backed version.
- **One successful live repro is weak evidence; one failed live repro is strong evidence.** A
  single pass could be luck (e.g., the model happens to call the tool correctly on that one
  attempt, side-stepping the fallback path entirely rather than exercising it). A single failure,
  by contrast, is a real, reportable defect — the whole point of this pass. Plan for **3
  independent live `@mention` trigger attempts**, not one, to have some confidence the fix
  generalizes across turns, not just that it worked once.
- Secondary risk: regression. The fix touches a hot path (`_run_agent_node`) shared by every
  `agent`-typed step. Confirm the full offline suite is still green and that an ordinary
  (non-workflow) message post/read is unaffected.
- Secondary risk: environment drift since the RCA/impl review sessions. Confirm FalkorDB, the M1
  server, and LM Studio are actually up before trusting any other result, and confirm the specific
  workflow snapshot (`triage@v1` in `ws:acme`) is actually what the executor will drive.

**What this pass explicitly does not attempt:**
- Re-deriving the root cause or re-reviewing the diff line-by-line — already done, referenced above.
- Statistically characterizing the local model's tool-call reliability rate (that is K-027's
  broader epic, `data-scientist`/`architect` territory) — 3 live attempts confirm the fallback
  engages in the running system, not a probability estimate.
- The M2/M1 exhaustion-only edge case flagged as impl-review finding M2 (loop exhausts on
  non-`post_message` tool calls only, never reaching the new branch) — out of scope; not the
  demo-blocking path and not something a live `@mention` repro can selectively trigger.
- Cleaning up or otherwise deciding the fate of the RCA's own pre-existing test artifacts in
  `demo-welcome` (`msgId ae8719305b5d4f3bb580b7e4c6d05253` / `runId
  00d95a27ac2a4dc8b74a86ed117b5c95`) — explicitly out of scope per the task brief; a new thread is
  used for this pass's own repro instead.

## 3. Environment & data setup

Pre-flight checks only — **no provisioning**; if any of these are down, the affected items are
blocked and reported, not fixed here.

1. `GET http://localhost:8000/health` — expect `{"status":"ok"}`.
2. `GET http://localhost:1234/v1/models` — expect `qwen/qwen3-4b-2507` listed.
3. `docker ps` — expect `falkordb-dev` up.
4. `./scripts/verify_workflows.sh acme` — sync check. A `reference`-graph `MISSING` result (known
   hazard: `test_queries.sh` wipes `reference` at teardown, `ws:<id>` snapshots survive) does
   **not** block this pass — `services.start_workflow_run`'s own docstring confirms it drives "a
   materialized def snapshot" (i.e. reads `ws:<id>`, never `reference`) on the `trigger_msg_id`
   path; the `reference` copy only backs the observability/diff endpoints. Recorded as a finding,
   not remediated (remediation would be a `devops`/demo-owner call, out of this pass's scope).
5. Test data: a **fresh thread** created via `POST /channels/demo-general/threads` (the
   `demo-general` channel, where the `assistant` Agent is already `MEMBER_OF`), so this pass's own
   repro messages/runs don't pile onto the RCA's pre-existing artifacts in `demo-welcome`.

## 4. Test items

| ID | Title | Priority | Type |
|---|---|---|---|
| TP-01 | Environment pre-flight (server, LM Studio, FalkorDB, workflow sync) | P0 | environment |
| TP-02 | Fresh thread creation in `demo-general` for isolated repro | P0 | setup |
| TP-03 | Live `@mention` trigger attempt #1 — run completes + reply visible | P0 | e2e/acceptance |
| TP-04 | Live `@mention` trigger attempt #2 — run completes + reply visible | P0 | e2e/acceptance |
| TP-05 | Live `@mention` trigger attempt #3 — run completes + reply visible | P0 | e2e/acceptance |
| TP-06 | Graph-level ground truth — `Message`/`PRODUCED` edge exists for each run | P0 | integration |
| TP-07 | Ordinary (non-mention) message post/read unaffected | P1 | regression |
| TP-08 | Full offline suite still green | P1 | regression |

### TP-01 — Environment pre-flight
**Preconditions:** none.
**Steps:** run the 4 checks in §3.
**Expected:** all four healthy/in-sync (or the `reference`-MISSING exception, which is non-blocking
per §3 item 4).
**Priority:** P0. **Type:** environment.

### TP-02 — Fresh thread creation
**Preconditions:** TP-01 passed.
**Steps:** `POST /channels/demo-general/threads {"title": "qa-mention-reply-delivery"}`.
**Expected:** `201`, a new `threadId` returned, distinct from `demo-welcome`.
**Priority:** P0. **Type:** setup.

### TP-03/04/05 — Live `@mention` trigger attempts (independent, same shape, different text)
**Preconditions:** TP-02's thread exists.
**Steps:**
1. `POST /threads/{tid}/messages {"text": "@assistant qa-mention-reply-delivery-N: <short question>",
   "mentions": ["assistant"]}` — the same request shape `web/app.js`'s `postMessage()` issues
   (`app.js:711-715`).
2. Poll `GET /threads/{tid}/workflow-runs` until the run reaches a terminal `status`.
3. `GET /threads/{tid}/messages?since=<pre-post timestamp>` — check for a new assistant-authored
   message.
**Expected:** `WorkflowRun.status == "done"`; a new `Message` (role/authorType indicating the
assistant) appears in the thread read — i.e., something a user would see in the chat window. Record
whether the reply came from a genuine tool call or the new implicit fallback (distinguishable via
the `StepRun` trace / `ERROR:` entries — the fallback path is exercised whenever the model ends a
turn on plain text with `post_message` granted and nothing posted yet).
**Priority:** P0. **Type:** e2e/acceptance.
**Note:** a failure on any one of the 3 attempts (reply still missing) is a reportable defect per
the task brief, not assumed to be the already-known limitation — it gets its own root-cause note in
the report (LLM output, whether the fallback fired, graph state).

### TP-06 — Graph-level ground truth
**Preconditions:** TP-03/04/05 executed.
**Steps:** for each run, direct Cypher against `ws:acme`:
```cypher
MATCH (r:WorkflowRun {runId:$runId})-[:HAS_STEP_RUN]->(sr:StepRun)-[:PRODUCED]->(m:Message)
RETURN sr.stepKey, sr.status, m.msgId, m.text
```
**Expected:** at least one `PRODUCED` edge per run linking to a real `Message` node, independent of
what the REST layer reports (defense against a read-path masking a write-path problem, mirroring
the RCA's own bisection method).
**Priority:** P0. **Type:** integration.

### TP-07 — Ordinary message post/read unaffected
**Preconditions:** TP-02's thread exists.
**Steps:** `POST /threads/{tid}/messages {"text": "no mention here"}` (no `mentions`), then
`GET /threads/{tid}/messages`.
**Expected:** message posts and reads back normally, no workflow triggered (no new `WorkflowRun`
appears for this message), confirming the fix didn't perturb the non-workflow path.
**Priority:** P1. **Type:** regression.

### TP-08 — Full offline suite
**Preconditions:** none (independent of the live checks).
**Steps:** `cd server && .venv/bin/python -m pytest -q`.
**Expected:** matches the impl review's baseline (647 passed, 1 deselected) or better — no new
failures.
**Priority:** P1. **Type:** regression.

## 5. Entry / exit criteria

**Entry:** TP-01 passes (or the `reference`-MISSING exception applies and is noted).

**Exit / verdict rule:**
- **Pass** — all of TP-03/04/05 show a visible reply + TP-06 confirms graph ground truth for all
  three, TP-07/TP-08 green. Verdict: "@mention-ing the assistant now reliably results in a visible
  chat reply."
- **Partial / at-risk** — 1-2 of 3 attempts succeed. Verdict: fallback engages but does not fully
  eliminate the failure mode on this model; report the failing attempt(s) as defects with full
  evidence per attempt.
- **Fail** — 0 of 3 attempts produce a visible reply. Verdict: the fix does not engage in the live
  system despite passing offline; escalate as a blocking defect, do not rubber-stamp the offline
  review's "approve" as sufficient.

## 6. Out of scope

See §2's explicit list. Additionally: no browser/UI automation (no such tool wired in this
environment) — driven via direct REST calls using the exact shape `web/app.js` sends, per the same
substitution the prior `web-api-coverage` test plan used and disclosed.
