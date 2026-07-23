# M3 executor — Landing 2 (U11) wiring design-patch (K-022 / K-023)

> **Status:** proposed (architect design-patch, 2026-07-12). Planning-only — no code/DDL changed.
> **Extends:** `docs/archive/plans/m3-executor.md` (approved plan, §6 trigger / §7 safety / Phases 4–5) and
> `docs/archive/plans/m3-executor-coordination.md` ("Carried to Landing 2" + D1–D5). Closes the analyst
> impl-review items in `docs/archive/reviews/m3-executor-impl.md` (M-1, m-1, m-3, n-2) that route to U11.
> **Scope:** the U11 `@mention`→workflow trigger wiring + the seams the triage proof (U13/U14) and QA
> (U15) need. It does **not** design the tests or the def content themselves.
> **Baselines to hold:** `test_queries.sh` 241/241, pytest 283, default-app import network-free.

---

## ★ SURFACE-TO-USER DECISION — PRODUCED-link ordering for U11

**Recommendation: Option B (link emitted messages after `_record`). Firm.** Do **not** adopt Option A.

**The problem (verified in code).** In `executor._drive` the loop is *execute → select → record*
(`executor.py:289-324`): `_execute_step` runs the agent node — where `post_message` fires
(`tools.py:202-225`) — **before** `_record` (`executor.py:299`) calls `record_step_and_advance`
(QUERIES §12.2), which is the only place a `StepRun` node is created. So at `post_message` dispatch
time there is no `StepRun {stepRunId}` to `MATCH`, and `link_step_emission` (QUERIES §12.6) cannot
fire; the tool reads `run.get("stepRunId")` (`tools.py:219`), finds nothing, and returns
`linked:false`. In Landing 1 this path was never exercised (the offline stub never enters
`_run_agent_node` — `_execute_step` only does so when `self._llm is not None`, `executor.py:342`), so
it was correctly deferred here.

**Why Option B, not Option A.**

| Axis | **A — pre-create the StepRun before executing the agent node** | **B — buffer emissions, link after `_record`** ✅ |
|---|---|---|
| `record_step_and_advance` atomicity (rule 4, M4/D2 locked §12.2) | **Breaks it.** To have a StepRun *before* execution you must split the single atomic *record+advance* query into `create_step_run` (pre) + `advance` (post) — a new repository method, a new QUERIES entry, and a graph-dba re-PROFILE. The M4 tail-anchored one-query advance is a locked, live-verified invariant. | **Untouched.** §12.2 stays byte-for-byte; `link_step_emission` is already the accepted second, non-atomic query (§3/§9). |
| §2.1 A/B/C loop (D2: locked byte-for-byte in Landing 1) | **Materially disturbed** — the record/advance split changes where AT_STEP/NEXT/stepCount move relative to execution; the loop's ordering assumptions change. | **Preserved.** The A/B/C branch block (`executor.py:302-324`) is unchanged; we add one call next to the existing `_trace_step` line, *above* the branch dispatch. |
| Audit-trail / step-budget semantics | Pre-creating a StepRun before the outcome is known risks a recorded-but-un-advanced StepRun on an early failure (double-count / orphan-count). | No change — one StepRun per executed step, exactly as today. |
| Failure/retry | A crash between pre-create and advance leaves a StepRun with no AT_STEP move — a new torn state to reason about. | A crash between post and link leaves a message with no PRODUCED link — the **already-accepted** diagnosable/retry-able gap (§3/§9), never a torn thread. |
| Complexity / blast radius | New repo method + QUERIES §12 entry + graph-dba gate re-open + loop rewrite. | Executor-local + one tool tweak; zero graph/DDL/QUERIES change. |
| Precedent in the codebase | none | **Exactly the `_trace_step` pattern already in the loop:** trace events are collected during execution (`StepResult.trace`) and emitted *after* the StepRun exists, keyed to the real `stepRunId` (`executor.py:299-300, 521-544`). Emissions are the same shape of deferred, stepRun-keyed side effect. |

Option B reuses the two-step emission the plan already ratified, keeps every locked invariant, and
mirrors a pattern already in the loop. Option A buys nothing the proof flow needs while reopening the
graph-dba gate and the locked write.

**What the implementer changes for Option B (all in `executor.py` + one tweak in `tools.py`):**

1. `StepResult` (`executor.py:62-74`): add `emissions: list[str] = field(default_factory=list)` —
   the msgIds this node posted, collected during execution, drained after the StepRun exists (same
   lifecycle as the existing `trace` field; update the docstring to say so).
2. `_run_agent_node` / `_handle_tool_call` (`executor.py:346-430`): capture posted msgIds. `post_message`
   already returns `{"posted": msgId, "threadId": …}` as its tool-result string; in `_handle_tool_call`
   (`executor.py:410-430`), after `dispatch`, parse the returned JSON and, when it carries a `"posted"`
   key, append the msgId to a per-node `emissions` list that `_run_agent_node` threads into the returned
   `StepResult`. (This keeps the tool decoupled — the executor owns audit linking. If preferred, an
   explicit sink list on the `run` dict is an equivalent mechanism; pick one, do not do both.)
3. `_drive` (`executor.py:299-300`): immediately after `self._trace_step(...)`, add
   `self._link_emissions(ctx, rec["stepRunId"], result.emissions)`. Position is *above* the
   `if firing is not None:` branch, so the A/B/C block is unchanged.
4. New `_link_emissions(self, ctx, step_run_id, msg_ids)` helper: for each msgId call
   `self._services.link_step_emission(ctx, step_run_id=step_run_id, msg_id=msg_id)` (already exists,
   `services.py:628-641` → repository §12.6, idempotent `MERGE`). A `None` return is a logged, non-fatal
   gap (§3/§9) — never raise from here (a missing link must not fail a run).
5. `tools.PostMessageTool.run` (`tools.py:202-225`): drop the now-dead inline `run.get("stepRunId")`
   link block (`tools.py:216-224`) — the stepRunId is never present at execute time by design; return
   `{"posted": msgId, "threadId": …}` and let the executor link. `link_step_emission` in `services.py`
   stays (now called by the executor). Adjust the two `test_tools.py` link tests
   (`test_tools.py:129,146`) to the new contract; add an executor-level test asserting a real
   `StepRun-[:PRODUCED]->Message` edge after an integrated agent-node post (see Test strategy).

> This is the **only** item the coordinator must put to the user for a go/no-go before U11 coding.
> Everything below is design that follows from the approved plan and the review.

---

## Context recap (Landing 1 state, verified)

Offline executor + capabilities are delivered and analyst-approved. `executor.py` drives the §2.1
A/B/C loop with injected `llm`/`guard_judge`/`tool_registry`/`tracer`; `guards.evaluate_guard`
(`guards.py`) does empty→unconditional / `{kind:'llm'}`→extract-then-judge / else→`NotImplementedError`;
`tools.py` has the `ToolRegistry` + `post_message`/`graphrag_retrieve`/`human_handoff` + `McpToolClient`;
`services.start_workflow_run`/`resume_workflow_run` drive runs synchronously (`services.py:578-626`);
`repository` is 1:1 with QUERIES §12 including `find_waiting_run_for_thread` (`repository.py:1299`). There
is **no `trigger.py` yet**, and `_build_default_app` (`app.py:182-213`) wires only the M2 responder,
gated on `config.ENABLE_AGENT`.

---

## U11.1 — M-1: drive fault-handling (kill the zombie run)

**Where.** Wrap the `while True:` body of `_drive` (`executor.py:289-324`) in one `try/except` — DRY,
since both `run()` and `resume()` funnel through `_drive` (`executor.py:252,265`). Do not wrap the entry
points separately.

**Behavior.**
- `except HumanHandoffSignal as sig:` (import from `tools.py:296`) → **suspend**, not fail. Call
  `self._repo.suspend_run(ctx.ws, run_id=run_id, thread_id=run_ctx.get("threadId",""))` and return
  `"waiting"` — the handoff parks the run pending a human, reusing the intake suspend/resume mechanics
  (§2.4). It is a capability, not exercised by the triage proof (no node grants `human_handoff`), so this
  is defensive; catch it **before** the generic net (it is an `Exception` subclass). Catching it in the
  executor (not the trigger) is correct: only `_drive` holds `run_ctx`/`run_id` and the suspend query.
- `except Exception as exc:` → **fail, don't zombie.** Stamp `fail_run` with a diagnostic ctx note
  (reuse the `_fail_budget` shape, `executor.py:508-519`, with `note["error"] = f"unexpected: {exc!r}"`),
  then **re-raise** so `_safe_run_workflow`'s isolation logs the stack (mirrors `_safe_respond`,
  `api.py:47-67`). The run ends `failed` with `AT_STEP` cleared (§12.5) — resumable-never, but no longer
  a permanent `running` orphan.

**Cross-check m-3 / n-2 (fail loudly *earlier*, then let the net catch):**
- **n-2 (null guard):** in `evaluate_guard` (`guards.py:66-68`) treat `guard is None` like `""`
  (unconditional) — a one-line safety net (`if not guard: return GuardVerdict(True, "unconditional …")`).
  A null guard means "no condition", so firing-when-reached is the correct semantics; this removes one
  `NotImplementedError`→zombie vector at the source.
- **m-3 (`{kind:'llm'}` with `judge=None`):** in `evaluate_guard`, before invoking the judge, raise a
  **named** error when `judge is None` (e.g. `raise WorkflowConfigError("no guard_judge wired for an llm
  guard")`) instead of calling `None(...)` → bare `TypeError`. The M-1 net then converts it to a
  `fail_run` carrying that clear message. Same treatment lets the M7 `expr`/unknown `NotImplementedError`
  (`guards.py:80`) fail the run diagnosably rather than orphan it.

Net effect: every fault vector the review named (`expr`/unknown guard, null guard, missing judge, repo
error, handoff) now ends in a defined terminal (`failed` or `waiting`), never `running`-forever.

---

## U11.2 — m-1: suspend-path step budget (decision: amend §7 wording)

**Recommendation: amend plan §7, do not add a suspend-path `maxSteps` check.** The step budget is a
**runaway-autonomy** guard (§7 — "every node and guard is an LLM call, an unbounded loop is a real
risk"). The suspend path (OUTCOME B, `executor.py:309-314`) is **human-paced**: each suspend records one
StepRun and then *stops*, resuming only when a human posts a reply. It cannot self-drive, so `maxSteps`
(an autonomy ceiling) is the wrong bound. The correct bound for the intake loop is the DS-note **3-round
clarifying ceiling** (`m3-executor-ml.md` Q1), carried in `ctx`.

- **U11 (now):** amend §7 to state explicitly that the suspend/intake loop is bounded by the clarifying-
  round ceiling, **not** `maxSteps`, and that a `waiting` run consumes no autonomous budget. This removes
  the contradiction the review flagged (§7 currently says `maxSteps` "counts every StepRun including
  intake re-runs").
- **Round-ceiling enforcement is a follow-up (needs ctx-write).** Counting intake rounds requires the
  executor to write `ctx` back (today it never mutates `ctx` — the m-2 finding). That is out of U11's
  critical path for AC-2 (which closes via thread re-read, U11.3 below), so track it with the m-2
  ctx-write work rather than forcing it into U11. Note it in §7 as the intended intake bound.

---

## U11.3 — Agent-node thread-message context (HARD prerequisite for AC-2)

**Gap (verified).** `_assemble_messages` (`executor.py:432-446`) folds in only the node `systemPrompt` +
the serialized run `ctx`. Intake therefore cannot see the human's thread replies, so it can never judge
"enough info" — AC-2 cannot close without this. It must land **in U11**.

**Design.**
- In `_run_agent_node` (`executor.py:346`), before building messages, fetch the recent thread via the
  **service layer** (layering — the executor holds `self._services`, holds no Cypher):
  `thread_msgs = self._services.read_thread(ctx, thread_id=run_ctx.get("threadId"))`. `read_thread`
  (`services.py:415` → `repository.read_thread`, QUERIES §4, `repository.py:577-596`) is **thread-scoped**
  via `HEAD`/`NEXT*0..`, returns `role`/`text`/`authorId`/`displayName`/`authorType` in `createdAt`
  order. Because it is thread-scoped (not channel-scoped), the workspace-wide channel caveat (K-015,
  which affects `graphrag_retrieve`/`hybrid_search`) does **not** apply here.
- Convert `_assemble_messages` from a `@staticmethod` to take the fetched messages:
  `_assemble_messages(config, run_ctx, thread_msgs)` — emit the `systemPrompt` as `system`, then each
  thread message as a conversation turn (`role: "assistant"` when `role == "assistant"` else `"user"`,
  content = `f"{displayName or authorId}: {text}"` so the model sees who spoke), then keep the compact
  `CONTEXT` block for the prior nodes' serialized state. Guard `threadId` absent → skip the read (offline
  unit tests pass no thread), preserving the network-free stub path.
- **Cap** the folded window (e.g. last 20 messages, app-side slice of the returned list) so a long thread
  cannot blow the prompt — RAM/latency hygiene, not correctness.

**Why this closes AC-2 without ctx-write.** On resume, `_drive` re-executes intake, which **re-reads the
thread** and now sees the new human reply — the guard judge re-evaluates against fresh state. The intake
loop advances via thread re-read on each resume; no `ctx` accumulation is required for the core loop
(that is only needed for the round-ceiling, U11.2 follow-up).

---

## U11.4 — The trigger + resume wiring (§6)

### `trigger.py` (new module)

```python
class WorkflowTrigger:
    def __init__(self, services, *, agent_id, def_key, def_version,
                 responder=None, trace=False): ...

    def maybe_trigger(self, ctx, *, thread_id, msg_id, text, role, mentions):
        # §6 ordered rule — exactly one of {resume, start, responder} acts.
        if role == "assistant":                       # 1. loop-guard
            return None
        waiting = self._services.find_waiting_run_for_thread(ctx, thread_id=thread_id)
        if waiting is not None:                        # 2. resume-if-waiting (no re-@mention)
            return self._services.resume_workflow_run(ctx, run_id=waiting["runId"])
        if self._agent_id in (mentions or []) and self._def_key:   # 3. @mention-to-start
            return self._services.start_workflow_run(
                ctx, def_key=self._def_key, version=self._def_version,
                trigger_msg_id=msg_id, trace=self._trace)
        if self._responder is not None:                # 4. fall-through to M2 direct reply
            return self._responder.maybe_respond(
                ctx, thread_id=thread_id, msg_id=msg_id, text=text,
                role=role, channel_id=None, mentions=mentions)
        return None
```

Notes:
- **Signature adds `text`.** Plan §6 lists `maybe_trigger(ctx, *, thread_id, msg_id, role, mentions)`,
  but the step-4 responder fall-through needs `text` (`AgentResponder.maybe_respond` requires it,
  `responder.py:82-85`). The §6 signature was illustrative; `text` is required. `start_workflow_run` does
  **not** need `text` — it reads the trigger message from the graph by `trigger_msg_id`
  (`services.py:592`).
- **Resume is single-flight by construction.** `resume_workflow_run` → `executor.resume` CASes
  `waiting→running` (QUERIES §12.4); a concurrent loser returns `status=None` and simply no-ops. Whether
  the CAS wins or loses, step 2 **returns** — it never falls through to start/responder (a waiting run in
  this thread means the reply belongs to that run, per §2.4).
- **New thin service method required:** `Services.find_waiting_run_for_thread(ctx, *, thread_id)` →
  `repository.find_waiting_run_for_thread(ctx.ws, thread_id=thread_id)` (`repository.py:1299`, already
  exists; add the `ctx.ws`-scoped passthrough next to the other §12 reads, `services.py:643-659`).

### `api.py` — the background handler

Add `_safe_run_workflow`, mirroring `_safe_respond` (`api.py:47-67`):

```python
def _safe_run_workflow(trigger, ctx, posted):
    try:
        trigger.maybe_trigger(
            ctx, thread_id=posted["threadId"], msg_id=posted["msgId"],
            text=posted["text"], role=posted["role"],
            mentions=posted.get("mentions", []))
    except Exception:
        _log.exception("background workflow trigger failed (msgId=%s)", posted.get("msgId"))
```

`build_router` (`api.py:70-73`) gains a `trigger: Any | None = None` param. In the `post_message` route
(`api.py:113-132`): keep the embed task; then schedule **exactly one** responder-vs-trigger task:

```python
if embed_worker is not None:
    background.add_task(_safe_embed, embed_worker, ctx.ws, posted["msgId"], posted["text"])
if trigger is not None:
    background.add_task(_safe_run_workflow, trigger, ctx, posted)   # owns the responder fall-through
elif responder is not None:
    background.add_task(_safe_respond, responder, ctx, posted)      # M2 wiring, unchanged
```

This is the M3 "exactly one handler" guarantee — when a trigger is wired, `_safe_respond` is **not**
scheduled (the trigger holds the responder for step 4), so an `@mention` can never fire both.

### `app.py` — wiring (gated, default off)

- `create_app` (`app.py:89-97`) gains `trigger: object | None = None`, passed to `build_router`.
- `_build_default_app` (`app.py:182-213`): add a `config.WORKFLOW_ENABLED` branch (checked **after** the
  existing `ENABLE_AGENT` construction, since the trigger needs the responder + embedder + LLM). When on,
  build the execution stack and pass the trigger (not the responder) into `create_app`:

  ```python
  if config.WORKFLOW_ENABLED:
      from .executor import WorkflowExecutor, GraphTracer
      from .tools import build_builtin_registry
      from .trigger import WorkflowTrigger
      registry = build_builtin_registry(services, embedder, agent_id=config.AGENT_ID)
      judge = _build_llm_judge(LMStudioLLM())          # DS Q1 prompt → {decision, rationale}
      executor = WorkflowExecutor(
          services, repo, llm=LMStudioLLM(),
          guard_judge=judge, tool_registry=registry,
          tracer=GraphTracer(repo, id_gen=…, clock=…))
      services.set_executor(executor)                  # services.py:136 late-bind
      trigger = WorkflowTrigger(
          services, agent_id=config.AGENT_ID,
          def_key=config.TRIGGER_DEF_KEY, def_version=config.TRIGGER_DEF_VERSION,
          responder=responder)
      return create_app(services, trigger=trigger, embed_worker=worker)
  ```

  - **Production judge (new small helper `_build_llm_judge`).** The executor must **not** be built with
    `guard_judge=None` (that is the m-3 vector). Wire an LLM-backed callable matching the injected
    shape `(condition, *, understanding, ctx, step_output) -> {decision, rationale}` (`guards.py:51-54`):
    build the DS §Q1 judge prompt, call `llm.chat`/`complete`, parse the JSON verdict. Calibration
    (κ ≥ 0.6 / false-advance ≤ 10%) is a U14/U15 concern, but the wired judge must exist at U11.
  - `WORKFLOW_ENABLED` off (default) ⇒ this whole branch is skipped, `_build_default_app` returns exactly
    today's app — the network-free import + pytest baseline are untouched (the imports above are inside
    the branch, lazy).

### `config.py` — flags

Add next to `ENABLE_AGENT` (`config.py:59-72`), reusing `_env_flag`:

```python
WORKFLOW_ENABLED: bool = _env_flag("FALKORCHAT_WORKFLOW_ENABLED", default=False)
TRIGGER_DEF_KEY: str = os.environ.get("FALKORCHAT_TRIGGER_DEF_KEY", "triage")
TRIGGER_DEF_VERSION: str = os.environ.get("FALKORCHAT_TRIGGER_DEF_VERSION", "v1")
```

Document all three in `server/.env.example` and `scripts/start_server.sh` (the served run turns
`WORKFLOW_ENABLED` on and must keep `FALKORCHAT_EMBEDDING_DIM=1024` for `ws:acme`, the AC-3 precondition,
`app.py:193-195`).

---

## U13 — `scripts/seed_workflows.sh` (sanity-check only — do not author here)

**Shape is sound; one important deviation from `seed_demo.sh`.** `seed_demo.sh` writes plain node
`MERGE`s via raw `redis-cli GRAPH.QUERY` (`scripts/seed_demo.sh:55-56`). A workflow def is **not** a
trivial write — publish runs `_validate_def_spec` (start-key derivation, type whitelist, transition-
endpoint checks, opaque-JSON serialization, `services.py:427-512`). **Recommend `seed_workflows.sh` wrap
a tiny Python one-shot** that calls `services.publish_workflow_def(...)` (global `reference` graph) then
`services.materialize_def(ctx, key, version)` (into `ws:acme`), rather than reimplementing the opaque-
JSON encoding and start-key rule in bash Cypher. Both are idempotent by design: publish of an identical
spec is a no-op (stable-key serialization, `services.py:106-113`); materialize `MERGE`s the snapshot
(`services.py:514-535`). Additive-only — it touches neither chat nor demo data.

**Def shape (per §8 table) to confirm at author time:** `kind:'conversation'`, three `type:'agent'` steps
(`STEP_TYPES` already whitelists `'agent'`, `services.py:44-46`) — **intake** (`start:true`,
`config.waitsForHuman:true`, `tools:['post_message']`, `maxIterations:4`), **research**
(`tools:['graphrag_retrieve']`), **answer** (`tools:['post_message']`, terminal). Transitions:
intake→research guard `{"kind":"llm","text":"the user has provided enough information …"}`;
research→answer guard `""` (unconditional, **D5**); answer has no outgoing (terminal).

**Risks to flag:** (a) **run order** — `seed_workflows.sh` depends on `bootstrap_schema.sh` (reference +
`ws:acme` at dim 1024) **and** `seed_demo.sh` (the `assistant` Agent + a channel/thread to `@mention`);
sequence it after both. (b) `TRIGGER_DEF_KEY`/`_VERSION` config must match the seeded `key`/`version`
(`triage`/`v1`) or step 3 of the trigger never starts a run. (c) AC-3 needs seeded, embedded conversation
data for `graphrag_retrieve` to have anything to find — that corpus is a **test/QA precondition**
(U14/U15), not this script's job (plan §8 already notes this).

---

## U14 / U15 — test seams the implementation must expose

Design of the tests is out of scope; the implementation must leave these seams so U14 (live e2e
AC-1…AC-4) and U15 (qa-engineer AC-1…AC-6) can run:

- **A live marker.** Follow the existing live-test convention (real LM Studio, like the M2 responder
  smoke); the repo has no registered `live` marker yet — register one in `server/pyproject.toml`
  (`[tool.pytest.ini_options] markers = ["live: needs LM Studio + FalkorDB"]`) so the e2e is opt-in and
  the default `pytest` run stays green/offline. The live test drives: `@mention` → `TRIGGERED_BY` edge
  (AC-1); intake posts a question + parks `waiting` (AC-2 first half); a plain human reply (no re-mention)
  resumes and eventually fires intake→research (AC-2 second half); research retrieves + answer posts a
  grounded reply, run `done` (AC-3/AC-4). Assert `TRIGGERED_BY`, the `StepRun` `NEXT` trail, and terminal
  `status='done'`.
- **Observability for black-box QA (recommend pulling U12 forward).** AC-5 (a debug run traced, a lean run
  not) and AC-1/AC-4 are far easier to verify black-box if the run/step-run/trace reads are exposed over
  REST: `GET /workflow-runs/{runId}` + `/step-runs` + `/trace` as thin pass-throughs over the existing
  `services.get_workflow_run`/`read_workflow_step_runs`/`read_workflow_trace` (`services.py:643-659`).
  U12 is marked optional/low-priority in the coordination doc, but it is the AC-5 observability seam —
  recommend landing it in U11/U12 so QA does not have to reach into Cypher.
- **Deterministic integration seam (offline).** The trigger + executor are fully injectable (stub
  `services`, stub tool-calling `llm`, stub `judge`), so an offline integration test can publish+
  materialize the triage def into `ws:test`, post an `@mention`, and drive intake→research→answer with
  stubs — asserting `TRIGGERED_BY` + the NEXT trail + `done` without a live model (plan §9 test strategy).

---

## Sequencing (file-by-file, keeps the tree buildable)

1. **config + service passthrough** — `config.py` flags; `services.find_waiting_run_for_thread`. (No
   behavior change; unblocks the trigger.)
2. **executor robustness** — M-1 fault net (+ HumanHandoffSignal suspend), n-2 null-guard + m-3 named
   error in `guards.py`; the PRODUCED Option-B change (`StepResult.emissions`, `_link_emissions`,
   `_handle_tool_call` capture); thread-context assembly (`_run_agent_node`/`_assemble_messages`).
   `tools.PostMessageTool` inline-link removal. Update the two `test_tools.py` link tests; add executor
   tests (fault→failed, handoff→waiting, PRODUCED-after-post, thread-context folded).
3. **trigger + api + app** — `trigger.py`; `_safe_run_workflow` + `build_router(trigger=…)` +
   one-handler dispatch; `create_app(trigger=…)` + `_build_default_app` WORKFLOW_ENABLED branch +
   `_build_llm_judge`.
4. **plan/doc sync** — amend `m3-executor.md` §7 (suspend path human-paced, round-ceiling not `maxSteps`);
   add `node_note` to the QUERIES §12.10 / DESIGN §5 trace-kind enumeration (review n-1); `docs/HISTORY.md`
   + `docs/BACKLOG.md` (K-023) entries.
5. **U13 seed**, then **U14 live**, then **U15 QA** — per the coordination doc.

Layering stays locked (api/mcp → services → repository 1:1 with QUERIES; tenant seam `config.get_context`);
`executor`/`guards`/`tools`/`trigger` are domain modules the services layer orchestrates. No new Cypher,
no DDL, no graph-dba gate re-open.

## Test strategy (per unit, altitudes)

- **M-1 (unit, offline):** a stub handler/guard that raises → assert the run ends `failed` with the ctx
  note and `AT_STEP` cleared, and the exception re-raises; a granted `human_handoff` (test-only def) →
  assert `waiting`, not `failed`. n-2: a `None` guard drives as unconditional. m-3: an `{kind:'llm'}`
  guard with `guard_judge=None` raises the **named** config error (→ `failed`), not a bare `TypeError`.
- **PRODUCED (integration, `ws:test`):** drive an integrated agent-node run (stub tool-calling LLM that
  calls `post_message`) → assert exactly one `StepRun-[:PRODUCED]->Message` per posted message, keyed to
  the correct StepRun; a link-failure path (missing endpoint) leaves the message and logs, run still
  completes.
- **Thread context (unit):** stub `services.read_thread` returning a scripted thread → assert
  `_assemble_messages` folds the turns (role-mapped, capped) ahead of the CONTEXT block; missing
  `threadId` → no read, stub path unchanged.
- **Trigger (unit, mocked services):** the four §6 branches — loop-guard no-op; resume-when-waiting (and
  the concurrent-loser no-op); @mention-start; fall-through to responder — each asserted to take exactly
  one path. api: one background task scheduled (trigger XOR responder), never both.
- **Offline integration:** publish+materialize triage def in `ws:test`, post @mention, drive with stubs →
  `TRIGGERED_BY` + NEXT trail + `done`.
- **Baselines:** `test_queries.sh` unchanged at 241/241 (no graph change); pytest ≥ 283 and grows; default
  app import stays network-free (WORKFLOW_ENABLED off by default, imports lazy in the branch).

## Risks & open questions

- **The one go/no-go:** PRODUCED ordering — recommend **Option B** (above). Everything else follows the
  approved plan/review and needs no further user decision.
- **Production judge reliability (DS D4/Q3):** the wired `_build_llm_judge` on the local 4B is the AC-2
  risk; calibration is U14/U15. If the κ/false-advance gate fails, escalate the judge / concretize the
  guard text — do **not** paper over it with `maxSteps` (§7).
- **Intake round-ceiling (m-1 follow-up)** needs executor ctx-write (m-2); tracked with that work, not
  forced into U11. Until then the intake loop is human-paced/unbounded-by-`maxSteps` — acceptable for the
  proof, documented in §7.
- **`graphrag_retrieve` channel scoping stays workspace-wide** (K-015, review n-3) — unchanged here;
  thread-context assembly is thread-scoped (§4) and unaffected.
- **Authz** for run-start/def-publish inherits the M1 single-tenant unauthenticated seam (K-016) —
  non-blocking, noted.

---

## Ready to implement — summary

**Plan:** `falkor-chat/docs/archive/plans/m3-executor-landing2.md`

1. **Surface-to-user decision — PRODUCED-link ordering: adopt Option B** (buffer emissions, link after
   `_record`). It keeps the M4/D2-locked atomic `record_step_and_advance` (§12.2) and the §2.1 A/B/C loop
   byte-for-byte, reuses the already-accepted two-step emission, and mirrors the existing `_trace_step`
   deferral — Option A would split the locked atomic write and reopen the graph-dba gate for nothing the
   proof needs. Change is executor-local (`StepResult.emissions` + `_link_emissions` after `_trace_step`)
   plus dropping `PostMessageTool`'s dead inline link.
2. **M-1** — one `try/except` around `_drive`'s loop: `HumanHandoffSignal`→suspend, any other exception→
   `fail_run`+re-raise (no more `running` zombie); plus n-2 (null guard→unconditional) and m-3 (named
   error when an llm guard has no judge) so every fault reaches a defined terminal.
3. **m-1** — amend §7: the suspend/intake loop is human-paced, bounded by the DS 3-round ceiling (a
   ctx-write follow-up), not `maxSteps`. No suspend-path budget check.
4. **Thread context (hard AC-2 prereq)** — `_run_agent_node` reads the thread via `services.read_thread`
   (§4, thread-scoped, no K-015 caveat) and folds capped, role-mapped turns into `_assemble_messages`; AC-2
   closes via thread re-read on resume, no ctx accumulation.
5. **Trigger wiring** — new `trigger.py` (`maybe_trigger`, §6 ordered rule, `text` added for the responder
   fall-through); `services.find_waiting_run_for_thread` passthrough; `api._safe_run_workflow` + one-handler
   dispatch (trigger XOR responder); `app` WORKFLOW_ENABLED branch wiring executor+registry+judge+trigger;
   `config` flags — all default-off, baseline network-free.
6. **U13/U14/U15** — `seed_workflows.sh` should wrap a Python one-shot over the service layer (not raw
   Cypher), run after bootstrap+seed_demo, key/version matching the config; register a `live` pytest
   marker and pull the U12 run-inspection REST reads forward for AC-5 observability.

No graph/DDL/QUERIES change; no graph-dba gate re-open; `test_queries.sh` holds at 241/241.
