# `mention-reply-delivery` — RCA

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-039 (M3.5)

## 1. Symptom & impact

Reported symptom: in the web chat UI, `@mention`-ing the demo assistant visibly starts and runs
a workflow to completion (the run/step panel shows it), the LM Studio side confirms the LLM
returned a response, but **no reply ever appears in the chat thread**.

Impact: demo-blocking. Every `@mention` of the assistant in `ws:acme`'s demo channel currently
produces a workflow run that completes (`status: done`) without posting a single chat message —
live-reproduced below, freshly, in this session, not just inferred from historical data.

## 2. Reproduction & evidence

Environment used exactly as handed off: FalkorDB up in Docker, the M1 server already running
under `uvicorn --reload` on `http://localhost:8000` (confirmed via `/health` → `{"status":"ok"}`),
workspace `ws:acme`. Contrary to the brief's caveat, **LM Studio was reachable from this sandbox**
(`curl http://localhost:1234/v1/models` returned the model list, including
`qwen/qwen3-4b-2507`, the server's configured default `LLM_MODEL`), so the entire pipeline —
REST → trigger → executor → LLM → (attempted) chat write → REST read → web poll logic — was
verified **live**, not statically assumed.

Confirmed server env (`/proc/<uvicorn-pid>/environ`): `FALKORCHAT_WORKFLOW_ENABLED=1`,
`FALKORCHAT_ENABLE_AGENT=1`, `FALKORCHAT_WS_ID=acme`, `FALKORCHAT_EMBEDDING_DIM=1024` — **no**
`FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` override present, so `config.TRIGGER_DEF_KEY`/
`_VERSION` are at their defaults (`triage`/`v1`). This rules out the K-037 env-var-collision
hypothesis the brief flagged as a prime regression suspect: there is no override in effect, and
`./scripts/verify_workflows.sh acme` (re-checked) still reports both defs in sync. **K-037 is not
implicated.**

**Fresh, controlled repro (this session):**

```
POST /threads/demo-welcome/messages
{"text":"@assistant analyst-rca-live-repro: what is 2+2?","mentions":["assistant"]}
→ 201 {"msgId":"ae8719305b5d4f3bb580b7e4c6d05253", ...}
```

15 seconds later, direct Cypher against `ws:acme`:

```
MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(m:Message {msgId:'ae8719305b5d4f3bb580b7e4c6d05253'})
RETURN r.runId, r.status, r.startedAt, r.endedAt
→ runId 00d95a27ac2a4dc8b74a86ed117b5c95, status "done", defKey "triage", defVersion "v1",
  stepCount 3, maxSteps 12
```

```
MATCH (r:WorkflowRun {runId:'00d95a27ac2a4dc8b74a86ed117b5c95'})-[:HAS_STEP_RUN]->(sr:StepRun)
OPTIONAL MATCH (sr)-[:PRODUCED]->(m:Message)
RETURN sr.stepKey, sr.status, sr.output, m.msgId ORDER BY sr.startedAt
→ intake  done  "Assistant: 2 + 2 equals 4. 😊"   (m.msgId: null)
→ research done "2 + 2 equals 4."                 (m.msgId: null)
→ answer  done  "Assistant: 2 + 2 equals 4."       (m.msgId: null)
```

```
GET /threads/demo-welcome/messages?since=1785455961731
→ []
```

**Bisection result, per the brief's own (a)/(b)/(c) framing: (a) — the write path, confirmed.**
No `Message` node was ever created for this run; there is nothing for a read endpoint or the web
UI to miss. Zero `PRODUCED` edges exist from any of the 3 `StepRun`s. This is not a read-path
filter bug and not a frontend rendering bug.

**Corroborating historical evidence** (same failure mode, present before this session started —
i.e. not something this repro introduced): querying every `triage@v1` `WorkflowRun` in `ws:acme`
shows the identical pattern repeats. Run `6dea1ba3c5d543cebf5f5a578ad07073` (triggered by msg
`5eaa39307afe471986f8ad8f3cf590de`, "@assistant I need help understanding what falkor-chat is used
for.") *did* manage one successful `post_message` — but only on its **first** `intake` iteration
(before the run parked for a human reply and later resumed). Every step run after that resume
(a second `intake` execution, then `research`, then the terminal `answer` step carrying the actual
grounded final answer) again produced plain text with **zero** `PRODUCED` edges. The user-visible
effect: the thread shows one early clarifying question, then silence, even though the run itself
completed `done` and the `answer` node's `StepRun.output` contains a coherent final answer that was
never posted. Two other recent `@mention`s in the same thread
(`c64778ddc4914a9eaee39b6f2433d7ba` "please file another access request",
`6885e724a12345ea925ea47775824452` "one more access request please") also have no assistant reply
at all in the thread read.

**`web/app.js` code-path check** (static, since the confirmed root cause is upstream of the UI):
`pollMessages()` (`web/app.js:191`) and `refreshRunPanel()` (`web/app.js:410`) are separate fetch
loops against separate endpoints (`GET /threads/{id}/messages` vs.
`GET /workflow-runs/{id}` + `GET /workflow-runs/{id}/step-runs`), exactly as the brief suspected.
`refreshRunPanel` is why "the workflow steps can be viewed" is accurate — the run/step panel
correctly shows `intake`/`research`/`answer` all `done`. `pollMessages` correctly returns `[]`
because there is genuinely nothing new to return. **Neither loop is buggy; both are faithfully
reporting real, distinct backend state.** K-038 (the `refreshRunPanel` overlapping-poll-tick race)
is unrelated — it's a display race on an already-fetched run, not implicated here.

**Residual test artifact, disclosed for transparency.** The fresh repro above wrote a real
message (`ae8719305b5d4f3bb580b7e4c6d05253`, text prefixed `analyst-rca-live-repro:`) and a real
`WorkflowRun` (`00d95a27ac2a4dc8b74a86ed117b5c95`) into the live `ws:acme` demo thread
`demo-welcome`, exactly as the brief's own recommended reproduction approach requires (posting
through the real REST path the web UI uses). Both are left in place — an analyst does not mutate
live state beyond what reproduction itself required, and deleting them is a separate, explicit
act for whoever owns `ws:acme`'s demo data to decide on, not something to do silently here. They
are easy to identify and remove (`msgId ae8719305b5d4f3bb580b7e4c6d05253` / `runId
00d95a27ac2a4dc8b74a86ed117b5c95`) if the demo thread should be pristine before the next
walkthrough.

Ran the server's offline suite for a sanity check: `cd server && .venv/bin/python -m pytest -q` →
**642 passed, 1 deselected** (the 1 deselected is `tests/test_workflow_live.py`, `@pytest.mark.live`
— excluded by default via `addopts = -m "not live"`, needs LM Studio and is already documented in
`docs/BACKLOG.md` K-027 as "RED deterministically (2/2) on the AC-4 answer-post assertion — a
known, filed limitation (D12-B), not an unknown regression"). Not re-run here to avoid mutating
`ws:test`/`reference` state beyond what this investigation already touched; its documented status
is itself corroborating evidence for the same failure mode.

## 3. Causal chain

1. `POST /threads/{id}/messages` with `mentions:["assistant"]` → `api.py:164-165` schedules
   `WorkflowTrigger.maybe_trigger` as a background task (`trigger` is wired, not `responder`,
   because `config.WORKFLOW_ENABLED=1` — `app.py:266-282`).
2. `WorkflowTrigger.maybe_trigger` (`trigger.py:53-87`): no waiting run for the thread (rule 2), so
   rule 3 fires — `self._agent_id in mentions` and `self._def_key` (`"triage"`) is configured, so it
   calls `services.start_workflow_run(..., def_key="triage", version="v1", ...)`. The M2
   `AgentResponder` (rule 4, `responder.py`) is **held** and never reached — this is by design (the
   "one handler per message" guarantee), but it means every `@mention` today is committed to the
   workflow path, with no fallback if that path fails to post.
3. The executor drives `triage@v1`'s three `type:'agent'` steps (`executor.py:437-477`,
   `_execute_step`) — `intake` → `research` → `answer`, each via `_run_agent_node`
   (`executor.py:538-609`). Each node calls the LM Studio chat model
   (`qwen/qwen3-4b-2507`) with its `systemPrompt` and the granted tool (`post_message` for
   `intake`/`answer`, `graphrag_retrieve` for `research`) offered as a schema
   (`scripts/seed_workflows.sh`'s inline triage-literal `STEPS`).
4. **The model does not call the tool.** It returns plain, well-formed prose ("2 + 2 equals 4.")
   with no native `tool_calls`, no JSON-wrapped call, and no bare `name({...})` syntax —
   `llm._parse_chat_message`/`_parse_content_tool_calls`/`_parse_bare_call_syntax` (`llm.py:144-206,
   252+`) all correctly find nothing to parse, because there is genuinely nothing tool-call-shaped
   in the reply. `result.is_tool_call` is `False`, so `_run_agent_node` (`executor.py:589-592`)
   returns immediately with `output=result.text, emissions=[]` — no tool is ever dispatched.
5. Because `emissions` stays empty, `_buffer_emission` (`executor.py:675-683`) — which only
   appends a `msgId` when a **dispatched** `post_message` tool result carries a `"posted"` key —
   never runs. `_link_emissions` therefore has nothing to link: **no `StepRun -[:PRODUCED]->
   Message` edge, and critically, no `Message` node is created at all** (message creation itself
   only happens inside the `post_message` tool's dispatch, `tools.py:175+`, which is never
   invoked).
6. The engine has no contract enforcing "a node that is supposed to communicate must actually post
   something" — a plain-text final answer is a **perfectly valid** `StepResult` by the executor's
   own model (§6.1/§6.2 of `docs/DESIGN.md`: `agent` nodes end on any non-tool-call text,
   `on="done"`). The run therefore drives cleanly to completion (`status: done`), the guard on
   `intake→research` fires normally (the fuzzy LLM judge only looks at whether the request is
   understood, not whether anything was posted), and the run reaches its terminal `answer` step
   and finishes successfully — from the engine's point of view, nothing went wrong.
7. Net result: a workflow run that is indistinguishable from a healthy one in every place the UI
   surfaces run status (`GET /workflow-runs/{id}`, `GET /workflow-runs/{id}/step-runs`, the LM
   Studio server's own request/response log) — and zero chat messages.

**Hypotheses considered and ruled out:**
- *K-037 env-var collision (the brief's prime regression suspect)* — ruled out: no
  `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` override is set on the running server, and
  `verify_workflows.sh acme` reports both defs in sync. This bug is orthogonal to K-037; it would
  reproduce identically on a server started with none of K-037's affected variables ever touched.
- *Message written but not returned by the read endpoint (bucket b)* — ruled out directly:
  `GET /threads/.../messages?since=...` returns `[]`; there is no message to filter out, and a
  direct `MATCH (m:Message)` scan for the exact reply text also finds nothing new (only an
  unrelated older message from a pre-workflow-engine session happened to contain similar wording,
  confirming the model is echoing prior thread context, not that anything new was persisted).
- *Frontend rendering bug (bucket c)* — ruled out: `pollMessages()`/`refreshRunPanel()` are
  separate, independently-fetching code paths in `web/app.js`, and both are behaving correctly
  given what the backend actually returns. There is nothing wrong to render.
- *Run stuck/failed silently* — ruled out: `WorkflowRun.status` is `done`, not `failed` or
  `waiting`; `stepCount` (3) matches the def's 3 steps; no error surfaced anywhere.
- *LM Studio unreachable, blocking the investigation* — did not occur; the model server answered
  every request used in this repro.

## 4. Root cause

**Confirmed, live-reproduced.** The `type:'agent'` executor node
(`executor.py:_run_agent_node`) treats "the model returned plain text instead of calling its
granted `post_message` tool" as a normal, successful termination (`on="done"`), with no
engine-level guarantee that a node whose entire purpose is to communicate actually communicated
anything. Today's only defense against this is a **prompt-level instruction**
(`scripts/seed_workflows.sh`'s triage-literal `systemPrompt`s: "You MUST post your answer by
calling the `post_message` tool... an answer you write as plain text is discarded and the user
sees nothing") — and on the currently-configured local chat model
(`qwen/qwen3-4b-2507` via LM Studio), that instruction is not reliably followed: **3 of 3 steps
in this session's fresh repro, and every non-first-turn step in the sampled historical runs,
ended by emitting plain text instead of dispatching the tool.**

**This is not a new defect class.** It is the live, currently-blocking manifestation of an
already-identified and already-tracked gap: **"Defect C"**, documented at
`docs/BACKLOG.md` K-027 item 2 ("Terminal-node-must-post engine contract... Today's mitigation is
prompt-level and does not hold on a 4B") and its "Addendum from the K-025 QA pass" (which records
the same failure on a non-terminal `intake` node, and the `pytest -m live` test that is
"RED deterministically (2/2) on the AC-4 answer-post assertion"). What this RCA adds beyond the
existing K-027 write-up: **a fresh, direct, minimal repro against the exact server configuration
currently running the demo** (not a probe/eval harness), and confirmation that the failure now
affects **every single step** of the triage flow in a clean run (not just the previously-reported
"answer node ~2/8, then 0/3" rate) — i.e. today, on this box, with this model, the triage workflow
essentially never posts a reply.

- **Trigger** — none needed beyond "the demo is configured to run the `agent`-typed `triage@v1`
  workflow on `@mention`, backed by a small (4B) local model." This was already true before
  K-037/K-038; nothing recent changed it. The severity is inherent to the (workflow trigger) ×
  (small local model) combination, not to a specific recent commit.
- **Contributing factor 1** — `WorkflowTrigger` (`trigger.py`, by design, "one handler per
  message") makes the workflow path the **only** path once `FALKORCHAT_WORKFLOW_ENABLED=1`: there
  is no fallback to the older, more reliable M2 `AgentResponder` direct-answer path (visible in the
  thread history itself — the pre-workflow-era messages at `createdAt` ≈ `1783557xxx`/`1783595xxx`
  answered promptly and reliably) if the workflow's terminal node fails to post. A demo that wants
  reliable replies today has no graceful degradation available.
- **Contributing factor 2** — the `pytest -m live` characterization test for exactly this failure
  mode exists and is known-RED (K-027), but it is excluded from the default `pytest -q` run
  (`addopts = -m "not live"`), so **nothing in the normal CI-equivalent loop signals this is
  currently broken for a demo** — a green `642 passed` gives false confidence that the demo path
  works.

## 5. Suggested fix & prevention

This RCA does not implement a fix (that is `K-027` item 2's job, already scoped: "Terminal-node-
must-post engine contract... Needs an engine-level guarantee, not a prompt", owner `architect` for
the contract design → `coder`/`tdd-engineer`). Two things this RCA adds concretely for whoever
picks it up:

1. **Immediate, demo-scoped mitigation (small, does not need the full engine contract):** in
   `executor.py:_run_agent_node`, when a node's granted tools include `post_message` and the loop
   ends via the non-tool-call branch (`executor.py:589-592`) with a non-empty `result.text`, have
   the executor itself dispatch `post_message` with that text as a fallback — i.e. treat "granted
   the tool but didn't call it" as an implicit call, not a silent discard. This directly targets
   the two observed failure shapes (prose instead of a call; a call whose `mentions` arg gets
   rejected and the model "recovers" by dropping the tool) without waiting for the full
   "terminal-node-must-post" design. It should be built as a reproduction test first
   (`tdd-engineer`): construct a fake LLM stub returning plain text for a node granted
   `post_message`, assert a `Message` now exists with a `PRODUCED` edge, exactly mirroring this
   RCA's live repro (§2).
2. **Close the CI blind spot noted as contributing factor 2:** either promote (a subset of) the
   `pytest -m live` AC-4 assertion into the default-run suite behind a "LM Studio reachable" skip
   guard (so it runs whenever the model server happens to be up, as it was for this whole
   investigation), or add a `docs/HISTORY.md`/readiness-banner surfaced warning so "green pytest"
   is never read as "the demo's `@mention` path works." `services.workspace_readiness` (backing
   the `/workspaces/{ws}/readiness` route, `docs/BACKLOG.md` K-036) may be the right place to add
   an explicit "last N triage runs: N posted a reply / N did not" signal, since it already exists
   as a pre-demo check surface.
3. **Guardrail for this class of defect going forward:** any `type:'agent'` step whose config
   grants `post_message` (or any future "must-communicate" tool) should have an executor-level
   post-condition test asserting the node either dispatched that tool or the run recorded a
   traced, visible reason it didn't (not just a discarded `StepResult.output`) — the general shape
   K-027 item 2 is already scoped to build.
