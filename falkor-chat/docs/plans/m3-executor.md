# M3 — LLM-native workflow executor (K-022, reframed) — implementation plan

> **Status:** proposed (architect plan, 2026-07-10). Planning-only artifact — no code/DDL changed.
> **Designs to:** `docs/requirements/llm-native-workflows.md` (stakeholder-confirmed, "Ready for
> design"). **Builds on:** M3 Slice 1 (K-020 def model + K-021 snapshot materialization) —
> DELIVERED; `docs/plans/m3-workflow-engine.md` (Part A decomposition), `m3-workflow-engine-coordination.md`.
> **Baselines before this work:** `test_queries.sh` 193/193, pytest 196/196, both green. FalkorDB up.
> **Companion method note (see §10) — DELIVERED:** `docs/plans/m3-executor-ml.md` (data-scientist) —
> LLM-as-judge guard reliability + research-node retrieval grounding + tool-calling approach + safety
> defaults. Folded into this plan (§2.2/§4/§7/§10); consumed at Phase 2.
> **Review folded in:** `docs/reviews/m3-executor.md` (analyst, *approve w/ suggestions*) — M1–M4
> majors + M5–M8 minors closed in this revision (2026-07-10 plan-patch); D5 locked (research→answer
> unconditional). See the changelog at the end.

---

## 1. Goal & scope

Build the M3 **run/step-run executor** for **LLM-native workflows**: a workflow node is a
plain-language system prompt the model executes as a bounded agent within an author-set toolset
fence (FR-1/FR-6); transitions are natural-language conditions an LLM judges against run context
(FR-2); the author picks per node and per edge whether it is LLM-native or deterministic, and both
coexist in one workflow (FR-3). The run is triggered by a chat `@mention` (FR-7), the nodes can post
to the thread and retrieve via GraphRAG and call tools including MCP tools (FR-5), and — when the run
is a **debug instance** — every aspect of execution is traced for after-the-fact diagnosis (FR-4).
The delivered proof is the **conversational triage flow** (intake → research → answer, AC-1…AC-6).

**In scope (this feature):**
- The `WorkflowRun`/`StepRun` executor loop over a materialized snapshot (DESIGN §6.2).
- LLM-native step execution as a bounded, tool-scoped agent loop; fuzzy (LLM-judged) transition
  guards; deterministic step/guard **dispatch seam** so the two coexist.
- Suspend/resume driven by human chat replies (the intake loop, AC-2).
- Node capabilities: post message (reuse §4 write), GraphRAG retrieve (§6), call tool, **MCP-client**
  seam (scoped — see §4), human-handoff *capability* (present, not exercised — §4).
- Trigger linkage `TRIGGERED_BY` (FR-7/AC-1) and step-emission linkage (StepRun→Message), reconciled
  with the existing `AgentResponder` @mention path.
- Per-run tracing (debug vs lean) and the audit-trail records (FR-4/AC-5).
- Runaway-loop safety: run-level step budget + per-node iteration cap (§7).
- The triage proof def + seed + end-to-end run (AC-1…AC-4).

**Explicitly out of scope (deferred, do NOT build here):**
- The **business-process (deterministic) proof flow** — deferred by the requirements. The
  deterministic step/guard *evaluators* land only as a dispatch seam with the trivial cases; a full
  expression-formula evaluator is a later slice (§3, §6).
- **Human approval / hand-off in the flow** (FR-5d) — the tool/capability is wired but the triage
  proof does not exercise it (the intake *wait-for-reply* is a distinct mechanic — §2.4).
- **External MCP servers wired into the proof flow** — the MCP-**client** lands as an interface +
  registry + a stub-server integration test; connecting a real external MCP server is deferred (§4).
- Retrieval-quality *evaluation harness* itself (that is `graphrag-eval-ml.md` / K-026); this plan
  consumes its recommendations, it does not build the harness.

**Reconciliation with the M3 slice map.** This feature, as reframed, spans what the Part-A
decomposition split across **K-022** (executor), **K-023** (chat linkage: `TRIGGERED_BY` +
StepRun→Message), and the **conversational half of K-024** (the triage proof). K-024's
business-process proof and K-025 QA remain separate downstream items. The backlog entries should be
re-annotated to reflect that the LLM-native reframe pulls the trigger + conversational proof into
this executor deliverable (a doc task, flagged in §11).

---

## 2. Execution model

### 2.1 The run lifecycle & states

A `WorkflowRun` (DESIGN §6.2) has `status ∈ {running, waiting, done, failed}` (a **property**, not a
label — locked §1.2; the `status` index is already provisioned). The engine loop, per DESIGN §6.2:

```
start → [AT_STEP = start step]
loop:
  execute current step (deterministic handler OR LLM-native agent loop)  → step output + outcome `on`
  record StepRun (RAN→Step, HAS_STEP_RUN, NEXT from prev)                 [single GRAPH.QUERY]
  evaluate outgoing TRANSITION guards against run ctx, in `order`         → first firing transition
    OUTCOME A — a guard fires                                            → move AT_STEP to its `to` step; continue loop [single GRAPH.QUERY]
    OUTCOME B — no guard fires AND step.config.waitsForHuman == true      → status = waiting; return (resume on human reply)
    OUTCOME C — no guard fires AND NOT waitsForHuman:
        - step has outgoing transitions → RE-EXECUTE the same step (self-loop); continue loop
                                          (bounded by the step budget — never an unbounded park)
        - step has NO outgoing transitions (terminal) → status = done; return
  enforce step budget (§7): stepCount > maxSteps → status = failed; return
```

**The three loop outcomes are exhaustive and defined for every `type:'agent'` step** (closing the
M1 gap — there is no longer any "wait/intake node" inferred from graph shape). The suspend signal is
an **explicit per-step config flag `waitsForHuman: true`** (see §2.2 schema), not an inference:

- **A (advance)** — a guard fired; relink `AT_STEP` and continue.
- **B (suspend)** — no guard fired *and* the step declares `waitsForHuman: true` (only intake does in
  the triage flow). The run parks in `waiting`, to be resumed by a human reply (§2.4). A step that
  *cannot* be unblocked by a human must **never** set this flag — otherwise it parks forever.
- **C (re-loop or terminate)** — no guard fired and the step does **not** wait for a human: if it has
  outgoing transitions it **re-executes** (a legitimate self-loop, e.g. a research node whose guard
  isn't yet satisfied gathers more and retries), bounded by the step budget (§7); if it is terminal
  (no outgoing transitions) the run is `done`. This is the case that was previously undefined — a
  non-waiting node with an unsatisfied guard now re-loops against the budget rather than parking or
  silently hanging. Per **D5**, the triage research→answer transition is **unconditional** (§2.5/§8),
  so research advances the moment it produces findings and never relies on outcome C in the proof
  flow — but the rule is stated generally so the executor is deterministic for any def.

Each write that moves state (`record StepRun` + relink `AT_STEP`) is a **single `GRAPH.QUERY`** —
the AT_STEP relink (delete old edge, create new) plus the StepRun create + NEXT append must be atomic
per the locked HEAD/TAIL-style atomicity rule (AGENTS.md rule 4). The graph-dba gate authors the
verified Cypher; §3 specifies the shapes.

### 2.2 LLM-native node execution (FR-1/FR-6) — a bounded, tool-scoped agent loop

An LLM-native node is a new step type **`type: 'agent'`** whose opaque `config` string deserializes
(app-side only — never filtered in Cypher, rule 8) to:

```jsonc
{
  "mode": "llm",
  "systemPrompt": "You are the intake agent. Ask clarifying questions until you can state the user's request precisely…",
  "tools": ["graphrag_retrieve", "post_message"],   // the author-set fence (FR-6/AC-6)
  "permissions": { "post": true, "handoff": false }, // coarse allow-flags, optional
  "waitsForHuman": true,                              // explicit suspend signal (§2.1 outcome B) — intake only
  "maxIterations": 4                                  // per-node loop cap (§7; DS default 4)
}
```

The executor runs this as an agent loop (`executor.WorkflowExecutor._run_agent_node`):

1. Build the message list: the node `systemPrompt` + assembled **context** = run `ctx` (serialized
   state carried between steps) + recent thread messages (via `services.read_thread` / a since-read
   anchored on a per-run thread cursor stored in `ctx`) so the node sees the conversation and the
   prior nodes' outputs.
2. Call the LLM with **only the node's scoped tool schemas** (built from `config.tools` — this is the
   AC-6 enforcement point: an ungranted tool is never offered and, defensively, a tool call to an
   ungranted name is rejected by the dispatcher).
3. If the LLM returns a **tool call** → dispatch via the `ToolRegistry` (§4), append the result as a
   tool message, loop (bounded by `maxIterations`).
   **A tool that fails is a re-prompt, not a dead run** (K-022 U14 Defect B — amended 2026-07-15).
   The node's three argument-level failure modes are one rule: an **ungranted** tool name, a
   **malformed** call (missing required args), and a **dispatched tool raising a `ServiceError`**
   (the domain rejecting the model's arguments — e.g. a hallucinated `@mention` resolving to no
   member) all feed the error back to the model as an ordinary tool message and let it correct
   itself. Each costs one iteration, so the cycle is bounded by `maxIterations` exactly like a
   successful turn. The boundary is deliberate and narrow: **only `ServiceError`** is absorbed —
   an *engine* fault still propagates to the §2.1 M-1 fault net (`fail_run`, loudly), and
   `HumanHandoffSignal` is control flow raised *through* dispatch that must still reach the
   suspend path (§2.4). A blanket `except Exception` here would break both.
4. If the LLM returns a **final text / an explicit outcome** → that is the step's `output` and the
   emitted `on` outcome (default `"done"`; a node may emit a named outcome the guards branch on).
5. Every LLM prompt/response, tool call/result, and the iteration count are handed to the tracer
   (§5) — recorded only for a debug instance.

This honors FR-6 ("the model decides freely — which actions, in what order — within the fence"): the
executor does not script the action order; it offers the scoped tools and lets the model drive, with
the loop cap as the only bound.

> **LLM tool-calling is a new capability of the LLM seam.** Today `llm.LLM` is
> `complete(messages) -> str` (single-shot, used by `AgentResponder`). The executor needs
> function/tool-calling: extend the seam with `chat(messages, tools) -> ChatResult` (text-or-tool-calls),
> OpenAI-shaped, on `LMStudioLLM`. **Risk/verify (§9):** small local models (the default
> `qwen/qwen3-4b-2507`) are uneven at function-calling; the DS note (§10, delivered) judged this and
> recommends **native tool-calling primary + a wired JSON-structured-output fallback** (switch a node
> to structured-output if measured format-validity < ~90%) — see §10 Q3.

### 2.3 Deterministic nodes & the coexistence dispatch (FR-3)

The executor dispatches on `Step.type`:
- `type == 'agent'` → the LLM-native agent loop (§2.2).
- `type ∈ {prompt, tool, message, decision, human, wait}` → the **deterministic handler** —
  for this first cut a thin, honest seam: `message`/`prompt` post/complete via existing services;
  `wait`/`human` suspend; `decision`/`tool` land as documented stubs (`NotImplementedError` with a
  clear message) since the deterministic business-process proof is out of scope. The seam exists so a
  single workflow can mix LLM-native and deterministic nodes structurally, without building the full
  typed-step library now.

Transition guards dispatch the same way on the guard's serialized `kind` (§2.5).

### 2.4 Suspend / resume — the intake loop (AC-2)

AC-2 requires the intake node to post a clarifying question, **not advance**, and later advance when
it judges it has enough info. This is a **suspend-on-waiting-for-human-input** mechanic (distinct from
the FR-5d approval hand-off, which is deferred):

- Intake is the only step in the triage flow with `config.waitsForHuman: true` (§2.2). Its agent loop
  posts a clarifying question via `post_message`, then its single outgoing guard `"enough info to
  proceed?"` (LLM-judged, §2.5) is evaluated against `ctx` + thread.
- **Guard false + `waitsForHuman`** → §2.1 **outcome B** → the executor sets `status = waiting`,
  leaves `AT_STEP` on intake, and returns (releasing the background task). The run is now parked.
- **Resume trigger (M2 — one rule, mirrored in §6):** a subsequent **human** message
  (`role != 'assistant'`) **in the run's thread resumes the run with no re-`@mention` required**.
  Resume is checked **before** the `@mention` gate (see the ordered sequence in §6); the `@mention`
  requirement applies only to *starting* a run, never to answering an intake question. This is what
  keeps AC-2's natural conversational flow: the agent asks, the human answers in plain conversation,
  and the parked run picks the reply up. The message-post background path looks up a `waiting` run
  associated with the thread (via `TRIGGERED_BY`'s thread, or a denormalized `waitingThreadId` on the
  run for an index-anchored lookup) and re-enters the executor, which re-runs intake (now seeing the
  new reply in its context) and re-evaluates the guard.
- **Guard true** → §2.1 outcome A → transition to research → the loop continues.

Resume must be **loop-guarded exactly like the responder** (never resume on an agent-authored
message, `role != 'assistant'`) and must be single-flighted (a `running` run is not re-entered
concurrently — the `waiting → running` status flip is the guard; the graph-dba gate makes the flip a
guarded single-query CAS so two concurrent replies cannot both resume).

> **Only `waitsForHuman` steps can be resumed by a reply.** A `waiting` run always sits on a step
> that declared the flag, so a human reply is guaranteed to have something to unblock. Non-waiting
> nodes never enter `waiting` (§2.1 outcome C), so the resume path can never find a run parked on a
> step no human can advance.

### 2.5 Fuzzy guard evaluation (FR-2) & deterministic coexistence

`TRANSITION.guard` is already an opaque, set-on-create string (QUERIES §11.1) — **no schema change**.
Its content becomes a serialized JSON discriminator (parsed app-side only):

```jsonc
{ "kind": "llm",  "text": "the user has given enough information to research the request" }
""  /* empty = unconditional/default transition (fires whenever reached — lowest priority) */
```

Guard evaluation (`guards.evaluate_guard(guard, *, ctx, run, step_output, thread)`) dispatches on the
serialized `kind`, and in this cut resolves to exactly **two** live branches plus a raising seam:
- **Unconditional (empty guard `""`)** → fires whenever the transition is reached (lowest priority).
  This is how **deterministic** transitions are expressed in the triage flow — including the **D5**
  research→answer transition, which is unconditional (see below).
- `kind == 'llm'` → build a **judge prompt** (the guard `text` + the evidence contract below) and ask
  the LLM for a **structured boolean verdict + rationale**. The rationale is traced (FR-4). The judge
  prompt design, calibration, and the ambiguous-verdict policy are delivered by the DS note (§10) —
  the executor treats the judge as an injected callable so it is stub-testable offline.

  **The guard's evidence channel is two-tier** (DS note §Q1 extract-then-judge; implemented at
  `guards.evaluate_guard` + `app._build_llm_judge` — amended 2026-07-15, K-022 U14 **Defect A**,
  plan [`m3-guard-thread-context.md`](m3-guard-thread-context.md)). This section's earlier silence on
  *where the judge's evidence comes from* is precisely what let a broken seam ship:
  - **Primary — the compact `understanding`** (`{request, known, missing}`) the node emitted, read
    from `step_output` then `ctx`. Judging this compact state rather than a raw transcript is the
    DS's single highest-leverage reliability decision.
  - **Fallback — the last `RECENT_TURNS_N` (6) thread turns**, newest last, used **only when no
    `understanding` was emitted** (the DS *omit rule* — truthiness, not presence). Degraded but
    functional; it exists so a node emitting prose leaves the judge with evidence rather than `{}`.
    A judge handed `{}` every turn can only bias to suspend forever — that **was** Defect A.
  - **The turns cost no extra graph read.** `_run_agent_node` already reads its thread window; it
    rides out on `StepResult.thread` and `_select_transition` passes it through (m-C neutral). A
    `{"kind":"llm"}` guard on a **non-agent** step therefore sees `thread == []` and correctly
    degrades to the understanding-only path.
  - **The precedence lives in `guards`, not in the judge** — judges are dumb renderers of whichever
    tier they are handed, so every judge implementation inherits the rule instead of re-deriving it.
- **Any other `kind` (e.g. a would-be `expr`) → a pure `NotImplementedError` seam (M7).** The
  stakeholder reframed the §13 open question to "LLM, not an expression library," so we deliberately
  do **not** build an expression evaluator, and the triage proof exercises **no** `expr` guard (its
  deterministic transitions are the empty-string unconditional form, not `expr`). To avoid shipping an
  untested speculative branch, `expr`/unknown kinds dispatch straight to a clear `NotImplementedError`
  — the honest coexistence seam FR-3 asks for, with **zero dead code**. A full `expr` evaluator is a
  deferred slice; do not implement it here.
- Guards are evaluated in `TRANSITION.order`; the first firing guard wins (deterministic tie-break).

**D5 — research→answer is UNCONDITIONAL, not LLM-judged (locked by the user).** A fuzzy
research→answer guard could return `false` with **no human in the loop to unblock it** — a silent
park (§2.1 outcome B is impossible there since research is not `waitsForHuman`, so it would instead
re-loop against the step budget, §2.1 outcome C). Both are wasteful. The research node instead decides
sufficiency **internally** via its own abstention (§4/§10 Q2), and the research→answer transition is a
plain unconditional (empty) guard that fires as soon as research produces findings. Only the
**intake→research** guard is fuzzy/LLM-judged. `evaluate_guard` still supports the `llm` kind
generally (a future def may LLM-judge a *human-unblockable* gate), but the triage def uses it on
intake→research **only** (§8).

---

## 3. Graph / data-model changes (on top of DESIGN §6)

Most of the run/step-run DDL is **already provisioned** (verified in `bootstrap_schema.sh`):
`WorkflowRun.runId` index+UNIQUE, `WorkflowRun.status` index, `StepRun.stepRunId` index+UNIQUE,
`StepRun.status` index, `Step.stepUid`/`Step.key` indexes+constraint. The additions this feature
needs:

| Change | Where | DDL? | Notes |
|---|---|---|---|
| `Step.type` gains `'agent'` (LLM-native) | `services.STEP_TYPES` whitelist | no | `type` is opaque in-graph; one-line whitelist add. |
| `Step.config` carries the LLM-native bounds (systemPrompt/tools/permissions/waitsForHuman/maxIterations) as serialized JSON | write path unchanged | no | opaque string, parsed app-side only (rule 8). **`MAX_CONFIG_LEN` stays 8000 (M6, resolved)** — it already exists (`schemas.py:39,45,55`) and 8000 chars comfortably fits a node system prompt; **no bump, no RAM change**. |
| `TRANSITION.guard` carries `{kind,…}` JSON | write path unchanged | no | opaque, set-on-create; `_serialize_opaque` already JSON-encodes dicts. |
| `WorkflowRun` props: `trace` (debug flag), `maxSteps`, `stepCount`, `waitingThreadId` (denorm for resume lookup) | run write path | no (props additive) | `status` index already exists; `waitingThreadId` lookup can ride an existing anchor (see risk §9). |
| `WorkflowRun -[:TRIGGERED_BY]-> Message` | run start write | no (edge) | DESIGN §6.2; new edge type in the ws graph. |
| **`WorkflowRun -[:LAST_STEP_RUN]-> StepRun` tail pointer (M4)** | run start + each advance | no (edge) | **The `NEXT` anchor** — mirrors the locked `Thread` HEAD/TAIL pattern (DESIGN §5.2 / QUERIES §4). `record_step_and_advance` reads it to find the previous StepRun for the `NEXT` append and relinks it to the new StepRun in the **same** query → O(1) atomic advance, no chain-walk / label-scan. RAM: **one edge per run** (the tail moves, it does not accumulate) — a trivial additive line (rule 6); graph-dba confirms in the U1 RAM delta. graph-dba authors the actual Cypher at the U2 gate. |
| **StepRun → Message emission edge** | step emit write | no (edge) | DESIGN §6.2 names it `EMITTED`, but **`EMITTED` is already the K-013 `Message→Message` provenance edge** (QUERIES §10). **Disambiguate at the graph-dba gate** — recommend a distinct type (e.g. `PRODUCED`) for StepRun→Message to avoid the overload; final call is the gate's. |
| **`TraceEvent` node + `StepRun -[:TRACED]-> TraceEvent`** (debug-only) | tracer write | **yes** — `TraceEvent.traceId` index **then** UNIQUE constraint (index-before-constraint, §7.1) | New node type. RAM: **debug runs only** — call it out (§9). |

**RAM impact (rule 6).** Run/step-run nodes are the M3 per-workspace hot-growth line (execution
traces, DESIGN §3) — a run is `1 WorkflowRun + N StepRun + N-1 NEXT + a few edges`, bounded by step
count. `TraceEvent` is the larger line but is **gated to debug instances**: a debug run writes one
`TraceEvent` per LLM prompt/response, tool call/result, guard judgment, and retrieval — dozens per
run. Non-debug runs write **zero** trace nodes. Recommend: keep `TraceEvent.payload` a flat
serialized string (rule 8), cap its length at the write boundary, and note debug-run retention as an
open lever (§9). No change touches the M2 vector line.

**Single-`GRAPH.QUERY` atomicity (rule 4).** Every state-moving write is one query — the graph-dba
gate authors and PROFILE-verifies these shapes (QUERIES new **§12**), 1:1 with new `repository.py`
methods:

- **`start_run`** — CREATE `WorkflowRun {runId,defKey,defVersion,status:'running',startedAt,ctx,trace,maxSteps,stepCount:0,waitingThreadId}`, `OF_DEF`→snapshot, `AT_STEP`→start `Step`, `TRIGGERED_BY`→trigger `Message`. (No `LAST_STEP_RUN` yet — the tail edge is created by the first `record_step_and_advance`.) Backed by the `WorkflowRun.runId` UNIQUE constraint. Single query.
- **`record_step_and_advance`** — CREATE `StepRun` (constraint-backed), `RAN`→Step, `HAS_STEP_RUN`; **hang `NEXT` from the previous StepRun located via the `WorkflowRun -[:LAST_STEP_RUN]-> StepRun` tail pointer (M4)** — no chain-walk, no label scan; then **move the tail** (drop the old `LAST_STEP_RUN`, create a new one to the just-created StepRun), exactly like the `Thread` HEAD/TAIL relink; **relink** `AT_STEP` (drop old, create new to the `to` step); `SET run.stepCount = run.stepCount + 1`. The first advance in a run finds no `LAST_STEP_RUN` (so no `NEXT`) and just seeds the tail. All in **one query** (atomicity — rule 4). graph-dba PROFILEs it as edge-anchored at U2.
- **`suspend_run`** / **`resume_run`** — status flips `running↔waiting`, guarded (CAS on current status so concurrent replies can't double-resume). Single query.
- **`complete_run`** / **`fail_run`** — status → `done`/`failed`, clear `AT_STEP` (or leave for audit — gate decides).
- **`link_step_emission`** — StepRun→(`PRODUCED`)→Message. The chat message itself is posted via the existing guarded §4 write (`services.post_agent_answer`, single query); linking the emission is a **second** query (accepted two-step, like two-phase materialize — the message is the durable artifact; a missing link is a diagnosable, retry-able gap, not a torn thread). Note explicitly.
- **Reads:** `get_run`, `read_step_runs` (NEXT-ordered audit trail), `find_waiting_run_for_thread` (resume lookup), `read_trace` (debug reconstruction).

---

## 4. Node capabilities (FR-5)

Capabilities are exposed to LLM-native nodes as **tools** in a `ToolRegistry` (`tools.py`); a node's
`config.tools` list is the author-set fence (AC-6). Each tool is a typed callable the executor
dispatches when the LLM calls it, with the result traced (debug). Built-in tools for the first cut:

- **`post_message` (FR-5a)** — posts into the run's thread by reusing the existing guarded §4 write.
  Posts **as the workflow agent** (`services.post_agent_answer` with the agent actor, role derived
  `assistant`), and the executor links the emission StepRun→`PRODUCED`→Message (§3). The message is
  visible to participants as the run executes (FR-5a). Provenance seeds are attached when the post
  follows a retrieval (reuse the §10 EMITTED Message→Message provenance the responder already writes).
- **`graphrag_retrieve` (FR-5b)** — the LLM calls this with a **text** query, but
  `services.hybrid_search` takes a **query vector** `q_vec` (`services.py:381-383`), so the tool
  **depends on an injected `Embedder`** (`M5`): it embeds the query string (`embedder.embed(text)`,
  exactly as `responder.py:96-99` does) **then** calls `hybrid_search`. `ToolRegistry` wires the
  `Embedder` into this tool at construction (U9). **Precondition (AC-3):** the served app must run at
  the workspace's embedding dimension (`ws:acme` = 1024, per AGENTS.md) or the ANN silently returns
  nothing. Returns ranked seeds (msgId, text, score). **Relevance thresholding / seed count is
  delivered by the DS note** (§10, `m3-executor-ml.md` Q2): distance cutoff **τ≈0.5** (a calibration
  starting point, not a shipped constant), cap **5** / floor **1**, and **abstain** (findings =
  "no relevant context") when nothing passes τ — the research node applies this instead of the
  responder's raw-`k=10` anti-pattern.
- **`call_tool` / MCP-client (FR-5c)** — **scoped for the first cut**: land the tool-dispatch
  abstraction so *any* registered tool (built-in or MCP-exposed) is uniform to the node, and land an
  **MCP-client seam** (`tools.McpToolClient`) that can list+call tools on an external MCP server —
  **verified against a stub/in-memory MCP server in tests only**. Wiring a *real* external MCP server
  into the proof flow is **deferred** (the triage flow needs no external tool). This satisfies FR-5c
  as a present, tested capability without taking on external-integration surface now. (falkor-chat is
  today an MCP *server*, DESIGN §15; this adds the *client* direction — note the new dependency
  direction, no conflict with the server front door.)
- **`human_handoff` (FR-5d)** — the capability is registered (a tool that suspends the run pending a
  designated human's input) but **not granted to any triage node** — present, not exercised (per the
  requirements). Distinct from the intake wait-for-reply (§2.4), which is guard-driven suspend, not a
  handoff tool.

---

## 5. Tracing / serviceability (FR-4 / AC-5)

Tracing is **per-run-instance** and a primary value. `WorkflowRun.trace` (set at start — a debug
instance is started with `trace: true`) gates all trace writes. The executor holds an injected
`Tracer` with two implementations:

- **`NullTracer`** (non-debug) — records nothing; the run executes leanly. AC-5's negative half
  (a non-debug run records **no** rationale) is satisfied by construction.
- **`GraphTracer`** (debug) — writes a `TraceEvent` per recorded aspect via `StepRun -[:TRACED]->
  TraceEvent {traceId, seq, kind, at, payload}` (§3). `kind ∈ {node_rationale, guard_judgment,
  tool_call, tool_result, graphrag_retrieval, llm_prompt, llm_response, step_timing}`. `payload` is a
  flat serialized string (the prompt text, the tool args/result, the guard verdict + **why**, the
  retrieval seeds+scores, timing/status). `seq` orders events within a StepRun; the `NEXT` StepRun
  chain orders across steps — together they let a run be **fully reconstructed** after the fact
  (`repository.read_trace(runId)`), which is exactly FR-4's "reconstruct and diagnose."

The tracer is called at every seam already named in §2 (node loop, guard eval, tool dispatch,
retrieval). Because it is an injected interface, debug-vs-lean is one wiring choice and the executor
code carries no `if debug:` branches beyond obtaining the tracer.

**AC-5 test:** a debug run records `guard_judgment`/`node_rationale` events; the same flow started
non-debug records none (assert zero `TraceEvent` for that run).

---

## 6. Trigger (FR-7 / AC-1) & reconciliation with `AgentResponder`

Today: `api.post_message` schedules `_safe_respond` on `BackgroundTasks`, and `AgentResponder.
maybe_respond` posts a **direct** retrieval-grounded reply when the message @mentions the agent and
is not agent-authored (`responder.py`). The workflow trigger does **not** slot in as an independent
peer to `_safe_respond` — that would fire **both** on an `@mention` (double response, **M3**).
Instead the trigger **owns the responder fall-through**: it holds the `AgentResponder` and invokes it
only when no workflow handles the request, so **exactly one** of {workflow, direct reply} runs per
message.

**Trigger decision sequence (single ordered rule — M2/M3 reconciled; mirrors §2.4).**
`WorkflowTrigger.maybe_trigger(ctx, *, thread_id, msg_id, role, mentions)` runs, in order:

1. **loop-guard** — if `role == 'assistant'`, do nothing (never act on an agent-authored message).
2. **resume-if-waiting** — if a `waiting` run is associated with this thread (§2.4), **resume** it
   with this human reply (guarded CAS `waiting→running`). **No `@mention` required** — this is the
   step that keeps AC-2's natural flow (M2). Return; do **not** fall through to the responder.
3. **@mention-to-start** — else, if the message @mentions the agent **and** a triage workflow is
   configured for the workspace (`FALKORCHAT_TRIGGER_DEF_KEY`), **start** a `WorkflowRun` for that
   def's materialized snapshot, linked `TRIGGERED_BY` the triggering message (AC-1), and run the
   executor loop out-of-band. Return; do **not** also invoke the responder.
4. **fall-through** — else (no waiting run, and either no @mention or no workflow configured), invoke
   the held **`AgentResponder`** so M2 direct-reply behavior is preserved exactly. The responder
   keeps its own `@mention` guard, so a non-mention message that reaches step 4 is a no-op there.

**Wiring (M3 — exactly one background handler).** When `WORKFLOW_ENABLED`, `app._build_default_app`
registers **only** `_safe_run_workflow` and passes the `AgentResponder` **into** `WorkflowTrigger`
(for step 4). `_safe_respond` is **not** separately registered in that mode — so an `@mention` can
never trigger both a workflow and a direct reply. When `WORKFLOW_ENABLED` is off (the default), the
M2 wiring is **unchanged**: `_safe_respond` is registered as today and no trigger exists — keeping the
network-free pytest baseline and the M2 direct-responder path untouched. The executor runs on
`BackgroundTasks` (out-of-band, like the responder), so its LLM latency never blocks the poster — and
a `waiting` run simply returns, to be resumed by the next reply.

This keeps the trigger seam consistent with M2 (loop-guarded, out-of-band, failure-isolated) while
guaranteeing one handler per request, and lets workflow-mode and direct-reply-mode coexist behind
config.

---

## 7. Runaway-loop safety (the parked open question)

Every node and guard is an LLM call, so an unbounded loop is a real risk. Design decision — **two
nested bounds, no wall-clock/token budget** (the requirements explicitly set no cost budget):

1. **Run-level step budget** — `WorkflowRun.maxSteps` (**per-def default `12`, global hard ceiling
   `25`** — DS note Q4, §10). `record_step_and_advance` increments `stepCount`; when
   `stepCount > maxSteps` the executor transitions the run to `status = 'failed'` with a
   `TraceEvent`/`ctx` note ("step budget exceeded"). This is the **runaway-autonomy** guard — it bounds
   a run that *self-drives* (a `type:'agent'` node re-looping under §2.1 outcome C, an
   advance-per-step chain), and it is enforced on the **autonomous continue paths only** (outcome A
   advance, outcome C re-loop). It is deliberately **not** checked on the suspend path (see the intake
   note below).
2. **Per-node iteration cap** — `config.maxIterations` (**default `4`** for the tool-light proof
   nodes; the DS note keeps `6` only as an upper bound for a wider-fenced future node) bounds a single
   LLM-native node's tool-calling loop, so one node cannot spin forever calling tools. On exhaustion
   the node terminates with its best current text + a trace note (graceful — it does **not** hard-fail
   the run); only `maxSteps` exceeded fails the run.

**The intake (suspend/resume) loop is human-paced, bounded by the clarifying-round ceiling — NOT
`maxSteps`** (K-023, closing review m-1). Each suspend (§2.1 outcome B) records one StepRun and then
*stops*, resuming only when a human posts a reply; the loop **cannot self-drive**, so `maxSteps` — an
autonomy ceiling — is the wrong bound for it, and the executor deliberately does not check the budget
on the suspend path. A `waiting` run consumes **no autonomous budget** while parked (only executed
steps count, and a parked run executes none). The correct bound for the intake loop is the DS-note
**3-round clarifying ceiling** (`m3-executor-ml.md` Q1), carried in `ctx`. **Round-ceiling enforcement
is a follow-up** that needs the executor to write `ctx` back (the m-2 finding — today the executor
never mutates `ctx`); it is tracked with that ctx-write work, not forced into U11 (AC-2 closes via
thread re-read on resume, not ctx accumulation). Until it lands the intake loop is human-paced and
unbounded-by-`maxSteps` — acceptable for the proof.

**Not** a "never revisit a step" rule — the intake self-loop (AC-2, via §2.1 outcome B resume) and a
non-waiting node's re-execution (§2.1 outcome C) are *legitimate* cycles; the step budget is the
correct bound for the **autonomous** cycle (outcome C). Tracing (FR-4) aids diagnosis but, as the
requirements note, does not prevent loops — the budget does. Defaults come from the DS note (§10 Q4),
coupled to the judge/agent reliability gates — do not raise them to paper over a failing calibration.

---

## 8. Sequencing, file-by-file changes & interfaces

Phased so the tree stays buildable/green and each phase is reviewable. The component convention holds:
**graph-dba gate** (verified Cypher into `QUERIES.md` §12 + `test_queries.sh` assertions raising the
193 baseline, enumerated; `bootstrap_schema.sh` for the `TraceEvent` DDL; DESIGN §6/§13 updates) →
**tdd-engineer** (executor + services + tools + trigger). The DS note (§10) folds in at Phase 2.

**Phase 0 — graph-dba gate (data model + queries).**
- `bootstrap_schema.sh`: add `TraceEvent.traceId` index **then** UNIQUE (index-before-constraint).
  Confirm the resume lookup (`find_waiting_run_for_thread`) plans as an index scan — either via
  `TRIGGERED_BY`→Message→Thread traversal or a `WorkflowRun.waitingThreadId` value + the existing
  `status` index (gate PROFILEs and picks; may add a `waitingThreadId` index if needed — call out RAM).
- `QUERIES.md` §12: `start_run`, `record_step_and_advance`, `suspend/resume` (guarded CAS),
  `complete/fail_run`, `link_step_emission` (resolve the `EMITTED` overload → recommend `PRODUCED`),
  `get_run`, `read_step_runs`, `find_waiting_run_for_thread`, `read_trace`. Each live-verified +
  PROFILEd; each state-move is a single query.
- `test_queries.sh`: enumerated contract assertions (start creates the run subgraph; advance moves
  AT_STEP + appends NEXT + bumps stepCount atomically; suspend/resume CAS; step-budget fail;
  emission link; trace write/read; index-scan profiles). Gate pins the new count (193 → ~N).
- DESIGN §6.1/§6.2 (add `type:'agent'` + config shape incl. `waitsForHuman`, the guard `{kind}`
  discriminator, `trace`/`maxSteps`/`stepCount`, `TraceEvent`/`TRACED`, the emission edge name, and
  the **`LAST_STEP_RUN` tail edge**), and **§13** (mark the guard open question **resolved** —
  LLM-native + coexist, superseding "guard is an expression"). **M8 doc-sync in the same pass:** add
  `stepRunId` to the DESIGN §6.2 `StepRun` property list (it is indexed+constrained in
  `bootstrap_schema.sh`) and refresh the stale `test_queries.sh` "(126/126)" count in DESIGN §7.1 to
  the new pinned number.

**Phase 1 — executor core (deterministic, offline).**
- `repository.py`: run/step-run/trace methods (1:1 §12) + typed errors (`WorkflowRunNotFoundError`,
  `StepBudgetExceededError`).
- `executor.py` — `WorkflowExecutor(services, repo, *, llm, guard_judge, tool_registry, tracer,
  step_budget)`; the §2.1 loop with a **stub step handler + stub guard** so the whole engine
  (advance, NEXT trace, AT_STEP relink, suspend/resume, step budget, done/fail) is unit-testable with
  **no LLM/network** (deterministic transitions).
- `services.py`: `start_workflow_run(ctx, *, def_key, version, trigger_msg_id, trace)`,
  `resume_workflow_run(ctx, *, run_id, ...)`, run/step-run/trace reads; lock the **`start_key`
  contract** (slice-1 residual — the start step is the one with `start: True`; confirm and keep).
- pytest: executor units (drive intake-wait→resume→research→answer→done with stub handlers), audit
  trail, step-budget abort, suspend/resume CAS, tracing on/off (AC-5).

**Phase 2 — LLM-native execution + fuzzy guards (folds in the DS note §10).**
- `llm.py`: extend the seam with `chat(messages, tools) -> ChatResult` (tool-calls or text) on
  `LMStudioLLM`; keep `complete` for the responder.
- `guards.py`: `evaluate_guard` (LLM-judge via an injected judge callable + the deterministic seam);
  the judge prompt per the DS note.
- `executor._run_agent_node`: the §2.2 bounded tool-scoped agent loop; AC-6 scope enforcement.
- pytest: agent-node loop with a stub tool-calling LLM; guard-judge with a stub judge (verdict +
  rationale traced); AC-6 (ungranted tool rejected).

**Phase 3 — node capabilities + tools.**
- `tools.py`: `ToolRegistry`, built-in `post_message`/`graphrag_retrieve`/`human_handoff`, and
  `McpToolClient` (MCP-client seam) tested against a stub MCP server.
- Emission linking (StepRun→`PRODUCED`→Message) after the §4 post.
- pytest: each tool dispatches + traces; MCP-client against the stub; retrieve applies the DS
  threshold.

**Phase 4 — trigger + resume wiring.**
- `trigger.py` — `WorkflowTrigger.maybe_trigger` (§6); resume-before-start; fall through to
  `AgentResponder` when no workflow configured.
- `api.py`: `_safe_run_workflow` background handler (config-gated); `app._build_default_app` wires the
  trigger when enabled. Optional REST for run inspection (`GET /workflow-runs/{runId}`,
  `.../step-runs`, `.../trace`) via thin service pass-throughs (follows §14.4 pattern).
- `config.py`: `TRIGGER_DEF_KEY`, `WORKFLOW_ENABLED` flags (default off — network-free baseline
  preserved). `schemas.py`: run-start/inspect request models (size-bounded, rule 6).
- pytest: trigger starts a run + links `TRIGGERED_BY`; resume path; fall-through to responder.

**Phase 5 — triage proof flow (§9 below) + acceptance.**
- `scripts/seed_workflows.sh` (mirrors `seed_demo.sh`): publish the triage def, materialize into
  `ws:acme` (additive-only), register the trigger. Idempotent.
- One live-marked end-to-end test (real LM Studio, like the M2 responder smoke) exercising
  AC-1…AC-4; the rest of the acceptance is K-025 (qa-engineer).

**Layering stays locked:** `api.py`/`mcp.py` thin adapters → `services.py` → `repository.py` (Cypher
1:1 with `QUERIES.md`); tenant seam `config.get_context`. `executor.py`/`tools.py`/`guards.py`/
`trigger.py` are new domain modules the services layer orchestrates.

### The triage proof flow (AC-1…AC-4) mapped to the model

`kind: 'conversation'`, three `type:'agent'` steps + guards:

| Step (`type:'agent'`) | `config` flags | systemPrompt (plain language) | tools | outgoing guard |
|---|---|---|---|---|
| **intake** (start) | `waitsForHuman: true` | Ask one clarifying question at a time via `post_message`; **never pass `mentions`**; end each turn with ONLY the `{"understanding":{request,known,missing}}` JSON object (D8/S5 — the `understanding` primary path; see below). | `post_message` | `{kind:'llm', text:"the user has provided enough information to research their request"}` → **research**; false + `waitsForHuman` → **suspend/`waiting`** (§2.1 outcome B, AC-2) |
| **research** | — (no wait) | "Retrieve relevant context from the workspace and produce concise findings grounded only in what you retrieve; if nothing relevant is found, say so." | `graphrag_retrieve` | **`""` unconditional (D5)** → **answer** — fires as soon as findings exist; sufficiency is the node's own abstention (§4/§10 Q2), **not** an LLM guard (no human to unblock a suspend here) |
| **answer** | — (no wait) | **MUST** deliver the grounded answer by calling `post_message` (an answer emitted as plain text is discarded — Defect C); **never pass `mentions`**; cite the research findings used. | `post_message` | terminal (no outgoing transitions) → §2.1 outcome C → run `done` (AC-4) |

> **D8/S5 (intake) — `docs/plans/m3-guard-thread-context.md`.** The intake prompt asks the model to
> emit the compact `understanding` object as its final turn text so `guards._extract_understanding`
> reaches the DS **primary** extract-then-judge path (`m3-executor-ml.md` §Q1). Without it the guard
> runs forever on the degraded turns-only fallback. Shape mirrors `server/tests/eval/golden_guards.jsonl`.
>
> **Defect C (answer) — `m3-executor-coordination.md`.** Two measured 4B failure modes the answer
> prompt mitigates: (a) emitting the answer as final **text** instead of calling `post_message`; and
> (b) calling `post_message` with `mentions:["<displayName>"]` (the folded `"{displayName}: {text}"`
> thread context leaks the name), which §4 rejects — after which the 4B drops the tool and emits text,
> posting nothing. Both are prompt-only mitigations of an accepted D4 risk, not an engine guarantee.

- AC-1: @mention starts the run, `TRIGGERED_BY` the message. AC-2: intake posts a question and
  **suspends** (only intake sets `waitsForHuman`), resuming on the next human reply **without a
  re-`@mention`** (§2.4/§6) until the fuzzy intake→research guard fires. AC-3: research retrieves via
  GraphRAG over **seeded** conversation data (the seed script/QA create it — a test precondition, not
  this plan), applies the DS threshold/abstention, and advances **unconditionally** (D5). AC-4: answer
  posts a grounded reply, terminal → run `done`. No human approval (AC-4 simplified).

---

## 9. Risks, open questions & contradictions to flag

- **DESIGN §6.1/§13 reconciliation (not a locked-decision violation).** §6.1 says
  "`TRANSITION.guard` is an expression evaluated against run context" and §13 lists the
  expr-lib-vs-DSL open question. The stakeholder-confirmed requirements **reframe/resolve** this to
  LLM-judged guards (coexisting with a deterministic seam). This is authorized by the requirements,
  but it supersedes wording in DESIGN — **update §6.1/§13 in the same change** (do not silently leave
  the old wording). Rule 8 ("never filter inside serialized `config`/`guard`") is **respected** — the
  `{kind}` discriminator is parsed app-side, never in Cypher.
- **LLM function-calling reliability on the local 4B model** (§2.2) — the executor's agent loop and
  the tool-scope enforcement depend on the model emitting well-formed tool calls. Verify against LM
  Studio; if unreliable, fall back to JSON-structured-output prompting. **Resolved by the DS note
  (§10, delivered): native-primary + structured-output fallback, per-node format-validity gate.**
- **`EMITTED` edge overload** — DESIGN §6.2 names StepRun→Message `EMITTED`, but K-013 already uses
  `EMITTED` for Message→Message provenance (QUERIES §10). Recommend a distinct type (`PRODUCED`) for
  the StepRun sense; the graph-dba gate makes the final call and updates both DESIGN and QUERIES.
- **`TraceEvent` RAM & retention** (rule 6) — debug-only, but "trace all aspects" is verbose. Cap
  `payload` length; treat debug-run trace retention as an open lever (align with the §13 retention
  question). Non-debug runs write zero trace nodes — the default stays lean.
- **Resume lookup index** — `find_waiting_run_for_thread` must be index-anchored (not a
  `WorkflowRun` label scan). The gate PROFILEs the `TRIGGERED_BY`-traversal vs a denormalized
  `waitingThreadId` value; a new index is a small additive RAM line if needed.
- **Concurrent replies to a waiting run** — the `waiting → running` flip must be a guarded single-query
  CAS so two near-simultaneous human replies cannot both resume the same run (double execution).
- **Two-step emission link** (post message, then link StepRun→Message) is non-atomic across the two
  queries; accepted — the message is the durable artifact, a missing link is diagnosable/retry-able,
  not a torn thread. Documented.
- **Authz for global def publish / who may start a run** — inherits the M1 single-hardcoded-tenant
  seam (unauthenticated like everything else); ties to deferred K-016 auth. Non-blocking; note it.
- **Scope honesty** — this is a multi-slice feature (executor + trigger linkage + conversational
  proof). If the reviewer/orchestrator wants it delivered in smaller increments, Phases 0–3
  (executor + capabilities, offline-testable) form a green, self-contained first landing; Phases 4–5
  (trigger + proof) are the second. Flagged for the coordinator.

### Test strategy (altitudes)

- **Contract** (`test_queries.sh`, graph-dba gate): §12 run/step-run/trace queries — start subgraph,
  atomic advance (AT_STEP relink + NEXT + stepCount in one query), suspend/resume CAS, step-budget
  fail, emission link, trace write/read, and `Node By Index Scan` PROFILE assertions. Enumerated
  193 → ~N (gate pins).
- **Unit** (pytest, offline): executor loop with stub handlers/guards (advance, audit trail,
  suspend/resume, step budget, done/fail); agent-node loop with a stub tool-calling LLM; guard-judge
  with a stub judge (AC-2 verdict + rationale); AC-6 tool-scope rejection; AC-5 tracing on vs off
  (TraceEvent count); each tool dispatch + trace; MCP-client against a stub MCP server.
- **Integration** (pytest against `ws:test`): publish+materialize the triage def, start a run, drive
  it end-to-end with a stub LLM/judge; assert `TRIGGERED_BY`, the StepRun NEXT trace, and `done`.
- **Live** (behind a marker, like the M2 responder smoke): a real LM Studio triage run exercising
  AC-1…AC-4. Full acceptance is K-025 (qa-engineer).
- **Baselines stay green:** `test_queries.sh` at the new enumerated count; pytest ≥ 196; the
  default app import + baseline stay network-free (workflow trigger config-gated, like the M2 agent).

---

## 10. Data-scientist method note — `docs/plans/m3-executor-ml.md` (DELIVERED)

The ML-methodology core of this feature is answered by the **data-scientist** method note at
**`docs/plans/m3-executor-ml.md`** (delivered 2026-07-10; builds on `docs/plans/graphrag-eval-ml.md`).
It folds into the build at **Phase 2** and gates units U6–U8. **Do not re-derive the method here** —
the note is the source of truth for the numbers and prompt designs; this section only names the seams
that consume it and the decisions already folded into the plan above:

1. **Fuzzy-guard judge (FR-2/AC-2)** → note **Q1**: **extract-then-judge** (the intake node emits an
   `{request, known, missing}` `understanding` object; a *separate* minimal judge call scores the
   guard against it, not the raw transcript), output schema **`{decision, rationale}`**, a **calibrated
   golden set** (`golden_guards.jsonl`, κ ≥ 0.6 / false-advance ≤ 10% gate), **bias-to-suspend** on
   ambiguity for the intake guard, and a **3-round clarifying ceiling** in `ctx`. Consumed by
   `guards.evaluate_guard(judge=…)` (injected, stub-testable). Note Q1 also independently recommends
   the research→answer transition be **unconditional** — now locked as **D5** (§2.5/§8).
2. **Research-node grounding (FR-5b/AC-3)** → note **Q2**: distance-cutoff **τ≈0.5** (a calibration
   starting point), **cap 5 / floor 1**, **abstain** when nothing passes τ, grounding-strict +
   citation-forcing node prompt, and the **recall@5 (primary) / MRR / faithfulness** eval on the
   seeded corpus. Consumed by the `graphrag_retrieve` tool (§4, injected `Embedder` + threshold).
3. **Agent-loop tool-calling (§2.2 risk)** → note **Q3**: **native tool-calling primary, JSON
   structured-output as the wired fallback**, minimal 1–2-tool fences, name-against-granted-set +
   schema validation with a bounded re-prompt, switch a node to structured-output if format-validity
   < ~90%. Consumed by `llm.chat(messages, tools)` (U6) and `executor._run_agent_node` (U8).
4. **Runaway-safety defaults (§7)** → note **Q4**: **`maxSteps` = 12** (per-def default; global hard
   ceiling 25) and **`maxIterations` = 4** — already folded into §7 and the §2.2 config example below.

The seams stay explicit so these drop in without reshaping the executor: `guards.evaluate_guard` takes
an injected judge, `graphrag_retrieve` an injected retriever+threshold, and the LLM seam is swappable.
**Open decision surfaced by the note (already resolved):** OQ6 (research→answer judged vs
unconditional) → resolved **unconditional** by the user as **D5**.

---

## Ready to implement — summary

**Plan:** `falkor-chat/docs/plans/m3-executor.md`

**What it is.** The M3 LLM-native workflow executor (K-022, reframed to the stakeholder-confirmed
`llm-native-workflows` requirements): a run/step-run engine where an LLM-native node (`type:'agent'`)
runs as a bounded, author-tool-scoped agent loop, transitions are LLM-judged fuzzy guards
(deterministic step/guard **dispatch seam** kept so both coexist — FR-3), the intake loop
suspends/resumes on human replies (AC-2), nodes post/retrieve/call-tools (incl. an **MCP-client**
seam, scoped to a stubbed first cut) — FR-5, per-run debug tracing records **all aspects** into
`TraceEvent`s (FR-4/AC-5), @mention **triggers** a run via `TRIGGERED_BY` reconciled with the
existing `AgentResponder` (FR-7/AC-1), and a run-level step budget + per-node iteration cap bound
runaway loops (§7). Delivers the **triage proof flow** intake→research→answer (AC-1…AC-4).

**Key decisions:** `Step.type:'agent'` + opaque-JSON `config` for the LLM-native bounds incl. the
explicit **`waitsForHuman`** suspend flag (no DDL — rule 8 respected); guard is either the empty-string
**unconditional** form or `{kind:'llm'}` — the **`expr` kind is a pure `NotImplementedError` seam** (no
dead code, M7); new debug-only `TraceEvent`/`TRACED` (new DDL, index-before-constraint); **`WorkflowRun
-[:LAST_STEP_RUN]-> StepRun` tail pointer** anchors the atomic `NEXT` advance (M4, mirrors Thread HEAD/
TAIL); StepRun→Message emission as a **distinct `PRODUCED`** edge (avoids the K-013 `EMITTED` overload);
suspend/resume via a guarded `waiting↔running` CAS with resume **before** the @mention gate (M2);
**exactly one background handler** per request — the trigger owns the responder fall-through (M3);
**research→answer unconditional** (D5); DS-derived defaults **`maxSteps` 12 / `maxIterations` 4**;
**§13 guard open question resolved** (LLM, not an expr lib — reconcile DESIGN §6.1/§6.2/§13).

**Handoff order:** graph-dba gate (Phase 0/U1–U2: `TraceEvent` + `LAST_STEP_RUN` DDL, §12 queries incl.
edge-anchored `record_step_and_advance`, `test_queries.sh` 193→~N, DESIGN §6/§7.1/§13 incl. M8 sync) →
tdd-engineer/coder (Phases 1–5, per the coordination-doc units). The **data-scientist note**
`m3-executor-ml.md` is **delivered** (§10) and consumed at Phase 2.

**Decision/uncertainty flags:** (a) DESIGN §6.1/§6.2/§13 wording superseded by the reframe — update in
the same change (authorized D1, flagged). (b) LLM function-calling reliability on the local 4B — DS note
Q3 sets native-primary + structured-output fallback, verify against LM Studio. (c) `EMITTED` overload —
`PRODUCED` locked (D2), graph-dba updates DESIGN+QUERIES. (d) Multi-slice feature — Phases 0–3 (offline
executor, U0–U10) are a clean first landing; 4–5 (trigger + proof, U11–U15) the second. (e) The DS
judge-calibration gate (κ ≥ 0.6 / false-advance ≤ 10%) is the executor's reliability gate — if it fails,
the fuzzy intake guard should not ship live on the 4B (escalate to a stronger judge / more concrete
guard text), not be papered over by `maxSteps`. Baselines stay green; default app import stays
network-free (trigger config-gated).

---

## Changelog — 2026-07-10 plan-patch (closing the analyst review `docs/reviews/m3-executor.md`)

Targeted revision, no redesign. Sections touched and how each finding is resolved:

- **M1 (suspend semantics) — §2.1, §2.2, §2.4, §8.** Added an **explicit `waitsForHuman: true` config
  flag** as the suspend signal (replacing the undefined "wait/intake node"), and rewrote the §2.1 loop
  to define **all three outcomes** exhaustively: A advance (guard fires) / B suspend (`waitsForHuman`,
  human-unblockable only) / C re-loop-or-terminate (non-waiting: re-execute if it has outgoing
  transitions, bounded by the step budget; `done` if terminal). The previously-undefined "no guard +
  outgoing + not waiting" case is now a bounded self-loop, not an infinite park.
- **M2 (resume contradiction) — §2.4, §6.** Reconciled to one ordered rule, stated in both sections:
  `loop-guard → resume-if-waiting (no re-@mention) → @mention-to-start → fall-through`. Resume is
  checked **before** the @mention gate; @mention gates **start** only — restoring AC-2's natural flow.
- **M3 (double-response) — §6.** The trigger now **owns the responder fall-through**: when
  `WORKFLOW_ENABLED`, only `_safe_run_workflow` is registered and the `AgentResponder` is passed
  **into** `WorkflowTrigger` (step 4); `_safe_respond` is not separately registered — **exactly one
  handler per request**. Disabled mode = unchanged M2 wiring.
- **M4 (NEXT anchor) — §3.** Named the data-model decision: a **`WorkflowRun -[:LAST_STEP_RUN]->
  StepRun` tail pointer** (mirrors Thread HEAD/TAIL, DESIGN §5.2) that `record_step_and_advance` reads
  and relinks in the same query → **O(1) atomic advance**, no chain-walk/label-scan. RAM = one edge/run
  (tail moves, doesn't accumulate); graph-dba authors the Cypher + confirms RAM at U2.
- **D5 (research→answer unconditional, user-locked) — §2.5, §8, §10.** The research→answer transition
  is an **unconditional** (empty) guard — sufficiency is the research node's own abstention, not an
  LLM guard (no human to unblock a suspend there). Only intake→research is fuzzy/LLM-judged. §8 triage
  table and §2.5 updated; also partially covers M1 for research (but the general 3-outcome rule still
  stated).
- **DS note reconciled as DELIVERED — §10 (+ header, §2.2, §4, §7).** §10 now references
  `m3-executor-ml.md` as the delivered source of truth (extract-then-judge `{decision, rationale}` +
  calibrated golden set; τ≈0.5 / cap 5 / floor 1 / abstain; native tool-calling primary + structured-
  output fallback; `maxSteps` 12 / `maxIterations` 4) without restating the method. §7 defaults updated
  to 12/4; §2.2 config example updated.
- **Minors:** **M5** — §4 `graphrag_retrieve` now states it depends on an injected `Embedder` (embed
  text → `q_vec` → `hybrid_search`) + the `ws:acme` 1024-dim AC-3 precondition (→ U9). **M6** —
  `MAX_CONFIG_LEN` resolved in §3: **stays 8000, no bump, no RAM change** (→ U3). **M7** — §2.5 `expr`
  kind reduced to a **pure `NotImplementedError` seam** (no untested branch) (→ U7). **M8** — Phase 0
  bullet now folds the DESIGN §6.2 `stepRunId` add + §7.1 stale-count refresh into the graph-dba U1
  pass.

**Nothing left unclosed.** All four majors + D5 + the four minors are resolved in the doc. No new open
decision was created; the one live decision the DS note surfaced (OQ6 research→answer) is closed by D5.
