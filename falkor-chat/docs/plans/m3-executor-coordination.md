# M3 LLM-native executor — coordination & work breakdown
> Owner: teco (coordinator) · Status: **awaiting user review before implementation** · Last updated: 2026-07-10
>
> Companion to the design plan **`docs/plans/m3-executor.md`** (architect) and the requirements
> **`docs/requirements/llm-native-workflows.md`** (tico, stakeholder-confirmed). This doc slices the
> plan's 6 phases into small, independently-shippable units of work and routes each to an owner.

## Locked decisions (from the user, 2026-07-10)
- **D1 — DESIGN reconciliation authorized.** DESIGN §6.1/§6.2/§13 are updated in the same change to
  reflect LLM-judged guards + the `type:'agent'` node kind, superseding the old "guard is an
  expression" wording. §13 guard-language open question marked **resolved (LLM-native + coexist)**.
- **D2 — `PRODUCED` edge locked.** StepRun→Message emission uses a **distinct `PRODUCED`** type — do
  **not** overload K-013's `EMITTED` (Message→Message provenance, QUERIES §10). graph-dba updates both
  DESIGN and QUERIES accordingly.
- **D3 — Split into small definite units.** This backlog (below) is the split. First green landing =
  through **U10** (offline executor + capabilities, Phases 0–3); trigger + proof = **U11–U15**.
- **D4 — LLM function-calling reliability** accepted as a risk; the DS note (U0) sets safe defaults +
  a JSON-structured-output fallback. Proceed.
- **D5 — research→answer guard is UNCONDITIONAL** (user, 2026-07-10; DS note open-question 6). The
  research→answer transition is **not** LLM-judged — a suspend there has no human to unblock it and
  only burns the step budget. Only the **intake→research** guard is fuzzy/LLM-judged. Affects the
  proof-flow mapping (U13/U14) and the guard wiring (U7): `guards.evaluate_guard` still supports both
  kinds, but the triage def marks research→answer deterministic-unconditional.

## Routing & sequencing

Legend — Owner is the specialist the unit is delegated to; Dep = must-finish-before.
Two landings: **Landing 1 = U0–U10** (offline, suite green, no live LLM required to test);
**Landing 2 = U11–U15** (trigger + proof flow + acceptance).

| # | Unit of work (user-story framing) | Owner | Dep | Done-condition |
|---|---|---|---|---|
| **U0** | *Method note:* LLM-as-judge fuzzy-guard reliability, research-node GraphRAG grounding & AC-3 eval, function-calling reliability + safety defaults. | **data-scientist** | plan | `docs/plans/m3-executor-ml.md` written; the §10 questions answered; folds into Phase 2. |
| **R0** | *Review gate:* static review of `m3-executor.md` (grounding vs. real codebase, completeness, simpler alternatives). Complements the user's own review. | **analyst** | plan | Review at `docs/reviews/m3-executor.md` + verdict; findings routed to owners. |
| **U1** | As an operator, the graph carries the new workflow-execution schema and the DESIGN doc matches it (D1). | **graph-dba** | R0 | `bootstrap_schema.sh` adds `TraceEvent.traceId` index **then** UNIQUE (index-before-constraint), idempotent; DESIGN §6.1/§6.2/§13 reconciled per D1/D2; RAM delta called out. |
| **U2** | The verified query library §12 exists so every run state-move is one atomic, index-anchored query. | **graph-dba** | U1 | QUERIES §12 (`start_run`, `record_step_and_advance`, `suspend`/`resume` CAS, `complete`/`fail_run`, `link_step_emission` via **PRODUCED**, `get_run`, `read_step_runs`, `find_waiting_run_for_thread` index-anchored, `read_trace`) live-verified + PROFILEd; `test_queries.sh` enumerated assertions; baseline 193→N pinned & green. |
| **U3** | The repository exposes the §12 queries 1:1 with typed errors. | **tdd-engineer** | U2 | `repository.py` run/step-run/trace methods (1:1 §12) + `WorkflowRunNotFoundError`/`StepBudgetExceededError`; pytest green. |
| **U4** | As the engine, I walk a materialized workflow deterministically end-to-end with no network. | **tdd-engineer** | U3 | `executor.py` loop (advance, NEXT trace, AT_STEP relink, suspend/resume, step budget, done/fail) with **stub** handlers/guards; `start_key` contract locked; offline unit tests cover the loop + AC-5 trace on/off. |
| **U5** | The service layer starts/resumes runs and reads run/step/trace state. | **tdd-engineer** | U4 | `services.py` `start_workflow_run`/`resume_workflow_run` + reads; tenant seam respected; tests green. |
| **U6** | The LLM seam supports tool-calling chat (not just completion). | **coder** | U0, U5 | `LMStudioLLM.chat(messages, tools) -> ChatResult`; `complete` preserved; stub-LLM tests. |
| **U7** | A transition guard is judged in natural language by an LLM, with the deterministic path intact (FR-2/FR-3). | **tdd-engineer** | U0, U6 | `guards.py` `evaluate_guard` (injected judge + deterministic seam), judge prompt per U0; verdict+rationale traced; stub-judge tests (AC-2). |
| **U8** | An LLM-native node runs as a bounded, tool-scoped agent that can only use its granted tools (FR-1/FR-6/AC-6). | **tdd-engineer** | U6, U7 | `executor._run_agent_node` bounded loop; AC-6 scope enforcement; stub tool-calling-LLM tests incl. ungranted-tool rejection. |
| **U9** | A node can post to the thread and retrieve via GraphRAG, with emission linked (FR-5a/b). | **coder** | U8, U2 | `tools.py` `ToolRegistry` + `post_message`/`graphrag_retrieve`/`human_handoff`; StepRun→**PRODUCED**→Message link after the §4 write; per-tool dispatch+trace tests. |
| **U10** | A node can call an external MCP tool as an MCP **client** (FR-5c, new capability). | **coder** | U9 | `McpToolClient` seam tested against a **stub** MCP server; real external servers deferred (interface + stub only). |
| — | **▲ Landing 1 boundary — offline executor + capabilities, suite green ▲** | | | |
| **U11** | An `@mention` of the agent starts (or resumes) a workflow run, falling through to the M2 direct reply when no workflow is configured (FR-7/AC-1). | **coder** | U8, U9 | `trigger.py` `WorkflowTrigger.maybe_trigger` (resume-before-start; CAS on the waiting run); api background handler; `WORKFLOW_ENABLED`/`TRIGGER_DEF_KEY` config (**default off**, baseline stays network-free); tests. |
| **U12** | *(Optional)* An operator can inspect a run, its step-runs, and its trace over REST. | **coder** | U5, U11 | `GET /workflow-runs/{id}` + `/step-runs` + `/trace` thin pass-throughs; size-bounded request models; tests. Low priority. |
| **U13** | The triage proof workflow is seeded and triggerable in `ws:acme`. | **coder** (+ devops if env) | U11 | `scripts/seed_workflows.sh` publishes+materializes the triage def (additive-only), registers the trigger; idempotent. |
| **U14** | The intake→research→answer flow runs end-to-end against a live LLM (AC-1…AC-4). | **tdd-engineer** | U13 | One marker-gated live end-to-end test exercising AC-1…AC-4; asserts `TRIGGERED_BY`, StepRun NEXT trace, run `done`. |
| **U15** | *Acceptance:* the feature is QA-accepted against AC-1…AC-6 (this is K-025). | **qa-engineer** | U14 | Test plan + report under `docs/test-{plans,reports}/`; AC-1…AC-6 verified black-box; defects routed back to owners. |

## Notes / open threads carried from the plan
- **Runaway-loop safety** (requirements OQ) → resolved in-plan as a run-level step budget + per-node
  iteration cap (not no-revisit — intake self-loops legitimately). Enforced in U4, tested there.
- **`TraceEvent` RAM/retention** — debug-only; cap `payload` length; retention is a parked lever
  (ties to the §13 retention question). Called out in U1.
- **Authz for run start / def publish** inherits the M1 single-tenant unauthenticated seam (K-016
  deferred). Non-blocking; noted, not addressed here.
- **Human hand-off (FR-5d)** ships as a *capability* (the `human_handoff` tool + suspend/resume in
  U4/U9) but is **not exercised** by the triage proof flow — per requirements out-of-scope.

## Review outcome — R0 (analyst, `docs/reviews/m3-executor.md`)
**Verdict: approve with suggestions. No blockers** — plan claims verified against the real codebase.
Four **majors** to close in the plan *before the unit each governs starts* (localized gaps, not
redesigns → route to **architect** for a plan patch):
- **M1 → U4** — suspend semantics underspecified for `type:'agent'` steps; the "no guard fires + has
  outgoing transitions" case is undefined (a research node could park forever). Needs an explicit
  suspend signal (e.g. `waitsForHuman` config) + a defined rule for all three loop outcomes.
- **M2 → U11** — §2.4 (resume on any human reply) vs §6 (resume gated behind `@mention`) contradict;
  §6 would force a re-`@mention` on every intake answer, breaking AC-2. Reconcile to
  `loop-guard → resume-if-waiting → @mention-to-start → fall-through`.
- **M3 → U11** — double-response risk: `_safe_run_workflow` as a *peer* to `_safe_respond` fires both
  on an `@mention`. The trigger must own the responder fall-through — exactly one handler per request.
- **M4 → U2** — `record_step_and_advance` NEXT anchor unspecified; needs a `WorkflowRun→LAST_STEP_RUN`
  tail pointer (mirrors the locked Thread HEAD/TAIL pattern) or the atomic write is an O(n) scan. A
  data-model decision the plan must name — routes into the graph-dba gate.

Minors: M5 (`graphrag_retrieve` omits the embed step `hybrid_search` needs → U9), M6 (`MAX_CONFIG_LEN`
already 8000, answer the "bump?" question → U1), M7 (`expr` guard kind is dead/untested in this cut →
prune or defer, U7), M8 (DESIGN §6.2/§7.1 doc sync → U1).

## Status log
- 2026-07-10 — Requirements confirmed (tico) → architect plan delivered (`m3-executor.md`) → user
  review requested. D1–D4 locked. This split authored.
- 2026-07-10 — U0 (data-scientist method note `m3-executor-ml.md`) ✅ + R0 (analyst review) ✅
  delivered. D5 locked (research→answer unconditional, closing DS OQ6). Analyst = approve w/
  suggestions, 4 majors (M1–M4) routed above.
- 2026-07-10 — Architect **plan-patch applied** to `m3-executor.md`: M1 (three-outcome loop +
  `waitsForHuman` suspend signal), M2 (resume ordering reconciled), M3 (trigger owns fall-through,
  one handler/request), M4 (`WorkflowRun→LAST_STEP_RUN` tail pointer → O(1) atomic advance) all
  closed; D5 + minors M5–M8 folded in; DS note referenced as delivered. No new decisions created.
  **Plan is gap-free and ready for implementation. Held for user go on U1 (graph-dba gate).**
