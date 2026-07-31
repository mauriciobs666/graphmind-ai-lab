# `mention-reply-delivery` — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-039 (M3.5)

## Summary

Acceptance pass for K-039's implicit-`post_message`-fallback fix, executed live against the
running demo (FalkorDB in Docker, M1 server on `http://localhost:8000`, LM Studio on
`http://localhost:1234` serving `qwen/qwen3-4b-2507`, workspace `ws:acme`), per
`falkor-chat/docs/test-plans/mention-reply-delivery.md`. Tested against the **uncommitted
working-tree diff** in `falkor-chat/server/falkorchat/executor.py` (`git status`/`git diff --stat`
confirmed the fix is not yet committed — `HEAD` is `5e8a8c4`).

**Verdict: PASS.** `@mention`-ing the assistant now reliably results in a visible chat reply. All
3 independent live trigger attempts produced a real `Message` a user would see in the chat window,
confirmed both at the REST layer and at the graph layer (`PRODUCED` edges). One of the three runs
additionally exercised — and passed — the exact historical failure shape the RCA flagged as
worst-case ("every step after a human-wait resume fails to post"): after resuming, both the
re-run `intake` step and the terminal `answer` step posted successfully. No regressions: the full
offline suite stayed green (647 passed, 1 deselected, matching the impl review's baseline exactly)
and an ordinary non-mention message still posts/reads normally with no workflow ever triggered.

No defects found in the fix under test. One testability observation is recorded below (not a
defect) about thread-reuse confounding a "plain message never triggers a workflow" check when an
open `waiting` run already exists in that thread — this is pre-existing `WorkflowTrigger`
resume-on-next-message behavior, unrelated to K-039's scope.

## Results table

| ID | Result | Evidence |
|---|---|---|
| TP-01 | PASS | `GET /health` → `{"status":"ok"}`; `GET :1234/v1/models` lists `qwen/qwen3-4b-2507`; `docker ps` shows `falkordb-dev` up 42h; `verify_workflows.sh acme` → `FAIL` (`reference` MISSING both defs, `ws:acme` snapshot present both) — confirmed non-blocking, see Coverage note below. |
| TP-02 | PASS | `POST /channels/demo-general/threads` → `201 {"threadId":"4c7eb4368bee4b12a1ea85b4dc18d300", ...}` |
| TP-03 | PASS | Posted `msgId 467e982d76c947e3b48013327396dc7f` ("...capital of France?") → `WorkflowRun ae7b7a4a36754b63a30852bb9e43a7ce` reached `status:"done"`. `GET .../messages?since=...` returned 2 new assistant messages ("The capital of France is Paris.", `msgId`s `11c6f296...`, `5a9162b3...`). |
| TP-04 | PASS | Posted `msgId ce4975a51c3c4dff881b4cf0ea44e3fd` ("...10 times 7?") → run `58a3933a581f4fd6950fae10879ea641` `done`. New assistant messages `4a73faf1...`/`234a9c57...` ("...10 times 7 is 70."). |
| TP-05 | PASS | Posted `msgId 3a6a4d59ac7b4f9f913c2564a046f2b6` ("...color is the sky?") → run `9265582e1b8f4c5c994f9a2eb3c71908` parked `status:"waiting"` after `intake` (expected `waitsForHuman` design, matching the RCA's own historical account) — but `intake` **did post** a visible reply (`msgId 3cb535eb...`, "The sky is typically blue..."). Extended: a follow-up `@mention` (`msgId cc43efd1...`) resumed the same run; it then progressed through a second `intake`, `research`, and the terminal `answer` step, reaching `status:"done"` — **and the terminal `answer` step also posted** (`msgId fd2340cc...`), directly covering the RCA's worst historical failure shape ("every step after resume" previously failed to post). |
| TP-06 | PASS | Direct Cypher against `ws:acme` for all 3 runs confirmed `(StepRun)-[:PRODUCED]->(Message)` edges matching the REST-reported msgIds exactly (see per-run breakdowns in the Evidence detail below). `research` steps correctly show no `PRODUCED` edge (not granted `post_message`, expected). |
| TP-07 | PASS | First attempt (reused thread with an already-open `waiting` run) unexpectedly resumed that run instead of testing a clean no-workflow path — reclassified as an observation, not a result (see below). Repeated in a **fresh** thread (`b9984e3097c04aacb51f552970036768`): `POST` plain message → `GET /workflow-runs` → `[]` (no run triggered), `GET /messages` → only the one posted message, no assistant reply. |
| TP-08 | PASS | `cd server && .venv/bin/python -m pytest -q` → `647 passed, 1 deselected` — run once before live testing (baseline) and once after (regression check); identical both times, matching the impl review's documented count exactly. |

### Evidence detail — graph ground truth (TP-06)

```
run ae7b7a4a...: intake→done (msg 11c6f296...) | research→done (no msg) | answer→done (msg 5a9162b3...)
run 58a3933a...: intake→done (msg 4a73faf1...) | research→done (no msg) | answer→done (msg 234a9c57...)
run 9265582e...: intake→done (msg 3cb535eb...) | intake→done (msg 428b0082..., resumed by unrelated
                 plain message during the confounded TP-07 attempt) | intake→done (msg 570015b3...,
                 resumed by the deliberate follow-up mention) | research→done (no msg) |
                 answer→done (msg fd2340cc...)
```
(Queried via `docker exec falkordb-dev redis-cli GRAPH.QUERY ws:acme "MATCH (r:WorkflowRun
{runId:...})-[:HAS_STEP_RUN]->(sr:StepRun) OPTIONAL MATCH (sr)-[:PRODUCED]->(m:Message) RETURN
sr.stepKey, sr.status, m.msgId ORDER BY sr.startedAt"`.)

## Defects

None found. The fix behaves exactly as the RCA and impl review describe, and does so under live,
nondeterministic model conditions across 3 independent trigger attempts plus one resume-to-
completion extension — 4 successful post events total via the implicit-fallback-eligible path
(intake ×4 across the three runs' initial branches, answer ×2), none of which produced a
double-post, an unposted terminal answer, or a silently-discarded reply.

## Observation (not a defect)

**A `waiting` `WorkflowRun` resumes on the *next message in its thread*, regardless of whether that
message mentions the assistant.** During TP-07's first attempt, a plain non-mention message posted
into the same thread as TP-05's still-`waiting` run (`9265582e...`) silently resumed that run's
`intake` step instead of leaving the thread workflow-free — confirmed via the graph (a second
`intake` `StepRun` appeared, timestamped to the plain message, producing a new `Message`). This is
**pre-existing `WorkflowTrigger` "one handler per message" / resume design**, matches the RCA's own
historical corroboration ("the run parked for a human reply and later resumed"), and is **out of
K-039's scope** — not a regression introduced by this fix. It is, however, a genuine testability
gotcha: any future black-box check of "an ordinary message never triggers a workflow" must use a
thread with **no open `waiting` run**, or it silently tests resume behavior instead. TP-07 was
re-run in a fresh thread to get a clean result (see table above).

## Coverage & gaps

**Covered:**
- 3 independent live `@mention` trigger attempts against the real running server + real LM Studio
  model, each confirmed at both the REST layer (`GET /threads/{id}/messages`, the actual surface a
  user/UI reads) and the graph layer (`PRODUCED` edges — ground truth independent of any read-path
  bug, mirroring the RCA's own bisection method).
- The specific historical worst-case shape the RCA flagged ("every step after a human-wait resume
  fails to post") — directly exercised via TP-05's extension and found now working.
- Regression: full offline suite (647/1, unchanged) and the non-workflow message path (clean
  thread, TP-07 corrected run).
- The `reference`-graph-MISSING environment state flagged by `verify_workflows.sh acme` was
  investigated rather than assumed either way: `services.start_workflow_run`'s docstring and the
  live results both confirm the trigger path drives the `ws:acme` **snapshot**, never `reference` —
  so this pre-existing environment drift (caused by an earlier `test_queries.sh` teardown per
  `AGENTS.md`'s own documented hazard) did not and could not have affected this pass's results.
  Left un-remediated per the task's instruction not to provision beyond checking — re-seeding
  `reference` (if ever needed for the observability/diff endpoints) is a `devops`/demo-owner call.

**Gaps, deliberately out of scope (see plan §2):**
- The impl review's own M2 finding — the loop-exhausts-on-non-`post_message`-tool-calls edge case —
  was not exercised; it requires a model that only ever calls `graphrag_retrieve` and never ends on
  plain text, which isn't reliably inducible via black-box prompting and is explicitly a narrower,
  rarer shape than what this pass targets.
- No statistical characterization of the model's underlying tool-call reliability rate — that's
  K-027's broader epic (`data-scientist`/`architect` territory), not a 3-attempt acceptance check.
- No browser/UI automation tool was available in this environment; driven via direct REST calls
  using the exact request shape `web/app.js`'s `postMessage()` sends (verified by reading
  `web/app.js:711-715` beforehand) — the same substitution the prior `web-api-coverage` test plan
  used and disclosed. Gap noted, not silently substituted.
- The RCA's own pre-existing test artifacts in `demo-welcome` (`msgId
  ae8719305b5d4f3bb580b7e4c6d05253` / `runId 00d95a27ac2a4dc8b74a86ed117b5c95`) were left untouched,
  as instructed; this pass used two newly created threads in `demo-general`
  (`4c7eb4368bee4b12a1ea85b4dc18d300`, `b9984e3097c04aacb51f552970036768`) instead, each carrying its
  own small set of new test messages/runs that are likewise left in place for the demo owner's own
  disposal decision.

## Feedback & recommendations

1. **Ship it.** The fix closes the demo-blocking gap it was built for; nothing in this pass
   contradicts the impl review's "approve, no blockers" verdict — this pass corroborates it under
   the one condition that review couldn't itself exercise (a live, nondeterministic model).
2. **Consider the impl review's M1/M2 documentation suggestions** (docstring update on
   `_run_agent_node`, and calling out the tool-calls-only-exhaustion residual boundary) — still open,
   still low-cost, unaffected by this pass.
3. **The `waiting`-run resume-on-any-message behavior is worth a one-line callout** somewhere
   discoverable (`docs/DESIGN.md` §6 or the `trigger.py` docstring already partially covers "one
   handler per message" — an explicit "resume is not mention-gated" sentence would have saved this
   pass a wasted first TP-07 attempt). Not urgent, not K-039's scope, but a real, mildly surprising
   piece of system behavior for the next person testing this thread's resume path.
4. **K-039 item 3 (the CI blind-spot follow-up)** remains open per `BACKLOG.md` and is unaffected by
   this pass — still worth doing so a future regression on this exact path doesn't have to wait for
   a manual acceptance pass like this one to be caught.

## Artifacts left in the live demo (disclosed, not cleaned up)

- Thread `4c7eb4368bee4b12a1ea85b4dc18d300` ("qa-mention-reply-delivery") in channel
  `demo-general`: 3 triggered `WorkflowRun`s (`ae7b7a4a...`, `58a3933a...`, `9265582e...`), their
  messages, plus one plain non-mention message that incidentally resumed the third run.
- Thread `b9984e3097c04aacb51f552970036768` ("qa-mention-reply-delivery-clean") in channel
  `demo-general`: one plain message, no workflow run.
- Left in place per the same reasoning the RCA itself documented for its own artifacts — cleanup is
  the demo owner's explicit decision, not silently performed here.
