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

## Cost datapoint (K-022 done-condition — vs. K-020/K-021 baseline ~100k tokens / 23 tool uses / ~45 min, ungated)
Running tally for this gated Landing-1 run (U1–U10 + analyst impl-review gate). Filled at close.

| Delegation | Owner | Units | ~Tokens | Tool uses | Wall time | Verdict/notes |
|---|---|---|---|---|---|---|
| D1 | graph-dba | U1+U2 | ~221k | 48 | ~47 min | Phase-0 gate done; suite 193→241/241; PRODUCED locked; no new index (resume rides `status` idx) |
| D2a | tdd-engineer (cut off) | — | ~86k | 5 | — | Session-limit abort before any edit; no changes made; re-dispatched fresh |
| D2 | tdd-engineer | U3+U4+U5 | ~214k | 49 | ~31 min | repository §12 (1:1), executor.py loop (A/B/C), services run methods; pytest 196→240; suite still 241 |
| D3 | coder | U6 | ~57k | 23 | ~5 min | `llm.chat(messages, tools)->ChatResult` dual-shape parse; `complete` unchanged; pytest 240→248 |
| D4a | tdd-engineer (cut off) | — | ~127k | 11 | — | Session-limit abort mid-read; no edits; re-dispatched fresh |
| D4 | tdd-engineer | U7+U8 | ~161k | 35 | ~133 min | `guards.py` extract-then-judge + `_run_agent_node` (AC-6 reject, bounded loop, graceful exhaustion); §2.1 loop intact; pytest 248→263 |
| D5 | coder | U9+U10 | ~184k | 42 | ~14 min | `tools.py` ToolRegistry + post_message/graphrag_retrieve(τ/cap/abstain)/human_handoff + PRODUCED link + McpToolClient (stub-tested); pytest 263→283 |
| Gate | analyst | impl review | ~149k | 25 | ~7 min | **approve-with-suggestions, 0 blocker / 1 major / 3 minor / 3 nit**; both deferrals ruled acceptable-for-Landing-1 |
| **TOTAL** | 6 deleg + gate (+2 cutoff retries) | U1–U10 + gate | **~1.20M** | **238** | **~4h productive** | vs K-020/21 baseline ~100k / 23 / ~45min **ungated & 2 units**; this = 10 units + independent gate |

## Landing-1 delegation grouping (efficiency, not ceremony)
Same-owner, same-phase units are handed as one brief to cut cold-start overhead (each brief still
points the owner at the plan/coordination doc **by path** to read directly — no paraphrase). The
final analyst gate reviews the whole Landing-1 diff. Grouping:
- **D1 = graph-dba** — Phase 0 gate (**U1+U2**): schema/DDL + DESIGN reconciliation + QUERIES §12.
- **D2 = tdd-engineer** — Phase 1 (**U3+U4+U5**): repository + executor loop + services (offline).
- **D3 = coder** — Phase 2a (**U6**): LLM `chat(messages, tools)` seam.
- **D4 = tdd-engineer** — Phase 2b (**U7+U8**): fuzzy guard + agent-node loop.
- **D5 = coder** — Phase 3 (**U9+U10**): tools/ToolRegistry + MCP-client seam.
- **Gate = analyst** — impl review of the Landing-1 diff → `docs/reviews/m3-executor-impl.md`.

The chain is inherently sequential (each unit builds on the prior's code); no parallelism available.

## Carried to Landing 2 (surfaced during Landing 1 — NOT Landing-1 blockers)
- **PRODUCED link ordering (from D5/U9).** The executor's drive loop creates the current step's
  `StepRun` in `record_step_and_advance` **after** `_execute_step` runs (where tools fire), so at
  `post_message` dispatch time no `stepRunId` exists yet and the live `StepRun-[:PRODUCED]->Message`
  link cannot fire in the integrated flow. `graphrag_retrieve`/`post_message` are correct + live-proven
  when a `stepRunId` is resolvable, and gracefully skip-with-`linked:false` otherwise (message is the
  durable artifact, §3/§9). **Landing-2 decision (U11 wiring):** either pre-mint+create the StepRun
  before executing an agent node, or link emitted messages after `_record`. The coder correctly did
  **not** mutate the locked U8 loop — routed here. Analyst gate should confirm this is acceptable to
  defer for Landing 1.
- **Agent-node thread-message context (from D4/U8).** `_run_agent_node` assembles run `ctx` only; full
  thread-message context assembly is deferred to the trigger/services seam (Landing 2, U11).

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
- 2026-07-12 — **User go received; Landing 1 (U1–U10) started.** Env verified: FalkorDB up;
  baselines green (`test_queries.sh` **193/193**, `pytest` **196**). Delegation grouping recorded
  above (D1–D5 by phase/owner). Analyst impl-review gate (K-022 done-condition) is non-negotiable.
  U11–U15 explicitly out of scope for this run. **D1 (graph-dba, Phase 0 U1+U2) dispatched.**
- 2026-07-12 — **D1 ✅ (graph-dba, U1+U2).** `bootstrap_schema.sh` adds `TraceEvent.traceId` index
  THEN UNIQUE; DESIGN §5.1/§5.2/§6.1/§6.2/§7.1/§13 reconciled (D1/D2/M6/M8; §5.1/§5.2 had stale
  `EMITTED` on StepRun→Message — corrected to `PRODUCED`). QUERIES §12 live-verified+PROFILEd:
  `start_run, record_step_and_advance, suspend_run, resume_run, complete_run, fail_run,
  link_step_emission, get_run, read_step_runs, find_waiting_run_for_thread, append_trace_event,
  read_trace` (last name added by gate for the tracer write path). M4 advance is O(1) tail-anchored
  (PROFILE-confirmed, no chain-walk). Resume lookup: **no new index** — `waitingThreadId` denorm rides
  the existing `status` index (`Node By Index Scan`). `test_queries.sh` **193→241/241**, +48
  assertions; **teco re-verified 241/241 green**. pytest untouched (196). RAM: run/step-run
  hot-growth line + debug-only TraceEvent line, no M2 vector-line change. No blockers.
  **D2 (tdd-engineer, Phase 1 U3+U4+U5) dispatched.**
- 2026-07-12 — D2 first attempt aborted mid-run on a provider session limit (5 tool uses, no edits
  made — verified via git status; D1's graph work intact). Re-dispatched fresh.
- 2026-07-12 — **D2 ✅ (tdd-engineer, U3+U4+U5).** `repository.py`: 12 §12 methods 1:1 +
  `WorkflowRunNotFoundError`/`StepBudgetExceededError`. `executor.py` (new): `WorkflowExecutor` with
  injected seams (`llm`/`guard_judge`/`tool_registry`/`tracer` — latter three used now, llm parked for
  U6), the §2.1 A/B/C loop, monotonic StepRun clock, `Tracer`/`NullTracer`/`GraphTracer`, AC-5
  trace-on/off. `services.py`: `start_workflow_run`/`resume_workflow_run` + reads, tenant-seam scoped,
  late-bound executor to break the ctor cycle. `start_key` = slice-1 `start:True` contract kept
  (resolved via the `START` snapshot edge, executor never re-derives). pytest **196→240**;
  **teco re-verified 240 green**; suite still 241; default app import network-free (verified). **One
  in-scope reconciliation** (flagged, not a deviation): B/C outcomes record a StepRun via
  advance-to-self so the audit trail is per-execution and the step budget counts every StepRun (§7
  reading) — lives in executor code, no graph/spec change. Logged for the analyst gate to see.
  **D3 (coder, Phase 2a U6 — LLM chat seam) dispatched.**
- 2026-07-12 — **D3 ✅ (coder, U6).** `llm.py`: `ChatResult`/`ToolCall` types + `LMStudioLLM.chat(
  messages, tools)->ChatResult` with Q3 dual-shape parsing (native `tool_calls` field primary;
  content-embedded-JSON fallback for the LM Studio/Qwen3 failure mode; plain text otherwise).
  `complete()` byte-for-byte unchanged (regression-tested); `LLM` Protocol gains `chat`. pytest
  **240→248** (+8, all green); suite still 241; network-free import verified. Name-against-granted-set
  + re-prompt correctly deferred to U8. **D4 (tdd-engineer, Phase 2b U7+U8 — guard + agent loop)
  dispatched.**
- 2026-07-12 — D4 first attempt aborted mid-read on a provider session limit (11 tool uses, no edits;
  verified pytest still 248, no `guards.py`). Re-dispatched fresh.
- 2026-07-12 — **D4 ✅ (tdd-engineer, U7+U8).** `guards.py` (new): `evaluate_guard` — `""`
  unconditional (lowest priority, never calls judge — D5 path); `{kind:'llm'}` = Q1 extract-then-judge
  (`_extract_understanding` → injected judge → `{decision,rationale}` with bias-to-suspend on
  ambiguity, traced); `expr`/unknown = `NotImplementedError` (M7). Wired into `_select_transition`.
  `executor._run_agent_node`: scoped-tool offering (ungranted never offered), bounded loop
  (`maxIterations` default 4), **AC-6 defensive rejection** of ungranted/malformed calls via re-prompt
  (never dispatched), graceful exhaustion (best text + trace note, does not fail run). §2.1 A/B/C loop
  **byte-for-byte unchanged** (all 9 D2 loop tests pass). pytest **248→263** (+15); **teco re-verified
  263 green**; suite 241; network-free import verified. **Flag for analyst gate:** agent-node
  thread-message context assembly deferred to the trigger/services seam (Landing 2, U11) — only run
  `ctx` is assembled now; not a locked-decision deviation, noted. **D5 (coder, Phase 3 U9+U10 —
  tools + MCP client) dispatched.**
- 2026-07-12 — **D5 ✅ (coder, U9+U10).** `tools.py` (new): `ToolRegistry` (`schema`/`dispatch` matching
  U8's calls); `post_message` (§4 write as agent → `services.link_step_emission` → **PRODUCED**),
  `graphrag_retrieve` (injected Embedder → `hybrid_search` → Q2 τ≈0.5 cutoff/cap 5/floor 1/**abstain**,
  all ctor-configurable), `human_handoff` (registered capability, raises `HumanHandoffSignal`, granted
  to no node). `McpToolClient` MCP-**client** seam (sync over a bg asyncio loop), MCP tools unify with
  built-ins through the same dispatch path — stub `FastMCP` in-memory tested; real servers deferred.
  `services.link_step_emission` passthrough added. **D2 honored** (PRODUCED, never EMITTED). pytest
  **263→283** (+20); **teco re-verified 283 green**; suite 241; network-free import + ruff clean.
  Two Landing-2 seams surfaced (see "Carried to Landing 2" above) — correctly routed, no locked-code
  mutation. **All Landing-1 implementation (U1–U10) complete. Dispatching the mandatory analyst
  impl-review gate → `docs/reviews/m3-executor-impl.md` (K-022 non-negotiable done-condition).**
- 2026-07-12 — **Analyst gate ✅ — verdict `approve-with-suggestions`, NO blockers** (review at
  `docs/reviews/m3-executor-impl.md`). Counts: 0 blocker / 1 major / 3 minor / 3 nit. Both carried
  deferrals ruled **acceptable for Landing 1**. The one **major M-1** (→ tdd-engineer, executor.py):
  `_drive` has no top-level `try/except`, so an unexpected mid-drive exception leaves the run stuck at
  `status='running'` — a permanent un-resumable zombie once live defs/tools run (not a Landing-1
  green-suite blocker; the offline path is deterministic). Analyst says fold M-1 into the U11 dispatch;
  since no blockers, no re-review cycle forced. **Landing 1 (U1–U10) satisfies the K-022 done-condition
  — DONE.** Cost datapoint finalized in the table above.
- 2026-07-12 — Briefly paused, then **user resumed: proceed to completion.**
- 2026-07-12 — **Doc rollup ✅ (delegated to coder — teco's Write/Edit is coordination-doc-only by
  hook, so BACKLOG/HISTORY go through an implementer).** `docs/HISTORY.md`: prepended the 2026-07-12
  Landing-1 entry. `docs/BACKLOG.md`: K-022 marked `🟡 Landing 1 ✅ delivered + analyst-approved` with a
  Delivered bullet; **M-1 + the two deferrals carried into K-023 (U11)** as inputs; top critical-path
  note updated (§13 resolved; next = Landing 2 → K-025 QA). **teco verified all edits by reading.**
- 2026-07-12 — ✅✅ **LANDING 1 (U1–U10) FULLY DONE** — implementation + mandatory analyst gate
  (approve-with-suggestions, 0 blockers) + documentation rollup all complete. Suites green: query
  **241/241**, pytest **283**. Nothing committed (per convention — commit on user request). FalkorDB
  left running. **Open follow-ups for Landing 2 (U11–U15, separate run): M-1 zombie-run guard,
  live PRODUCED-link ordering, agent-node thread context** — all logged in K-023 + the review doc.

## LANDING 2 (U11–U15) — trigger + triage proof + QA acceptance
- 2026-07-12 — **Landing 2 started.** Env re-verified green (FalkorDB up; query **241/241**, pytest
  **283**). Architect design-patch delivered → `docs/plans/m3-executor-landing2.md` covering the U11
  wiring + the three carried deferrals (M-1, agent-node thread context) + m-1.
- 2026-07-12 — **DECISION (user go): PRODUCED-link ordering = Option B** — buffer emissions during
  agent-node execution, link `StepRun-[:PRODUCED]->Message` after `_record` (mirrors the existing
  `_trace_step` pattern). Keeps `record_step_and_advance` (§12.2 / M4) byte-for-byte and the §2.1
  A/B/C loop untouched; zero graph/DDL/QUERIES change (suite holds 241/241); no graph-dba gate
  re-open. Option A (pre-mint StepRun) rejected — would split the locked atomic advance. Rationale:
  `m3-executor-landing2.md` §★.
- 2026-07-12 — **D6 (tdd-engineer, U11+U12) dispatched.** Trigger wiring + Option B emission linking
  + M-1 fault-handling + m-1 §7 amendment + agent-node thread context + optional REST run inspection.
  Analyst impl-review gate remains the non-negotiable K-022 done-condition.
- 2026-07-12 — **D6 ✅ (tdd-engineer, U11+U12).** teco re-verified: **pytest 283→312 (+29), green;
  query suite 241/241 unchanged** (zero graph/DDL/QUERIES change per Option B); default-app import
  network-free (`WORKFLOW_ENABLED` default off). Delivered:
  - **Option B** — `StepResult.emissions`; `_handle_tool_call`/`_buffer_emission` capture posted
    msgIds; `_link_emissions` after `_trace_step` (above the branch dispatch, §2.1 A/B/C loop block
    byte-for-byte unchanged); `PostMessageTool.run` drops the inline link → returns `{"posted",
    "threadId"}`. Integration test asserts a real `StepRun-[:PRODUCED]->Message` edge.
  - **M-1** — `_drive` is now a fault-net wrapper over extracted `_drive_loop`: `HumanHandoffSignal`
    →suspend; any other exception→`fail_run` + diagnostic ctx note, then re-raise. No more zombie
    `running` run. **m-3** — `guards.WorkflowConfigError` (named) when an `llm` guard has no judge.
    **n-2** — null guard treated as `""` unconditional.
  - **m-1** — plan `m3-executor.md` §7 amended: intake loop is human-paced, bounded by the DS 3-round
    ceiling, NOT `maxSteps`; no suspend-path budget check.
  - **Thread context (AC-2 prereq)** — `_read_thread_context` via `services.read_thread` (§4,
    thread-scoped, cap `THREAD_CONTEXT_WINDOW=20`); `_assemble_messages` folds role-mapped turns in.
  - **Trigger** — new `trigger.py` `WorkflowTrigger.maybe_trigger` (§6 ordered rule; signature gains
    `text` for fall-through); `services.find_waiting_run_for_thread`; `api._safe_run_workflow` +
    one-handler dispatch (trigger XOR responder); `app._build_default_app` `WORKFLOW_ENABLED` branch
    (wires executor+registry+production `_build_llm_judge`+trigger); `config` flags default off.
  - **U12** — `GET /workflow-runs/{id}` + `/step-runs` + `/trace` size-bounded pass-throughs.
  - New tests: `test_trigger.py` (7), `test_executor_produced.py` (2) + additions across
    executor/agent/guards/services/api/app; `test_tools.py` link tests updated to the new contract.

---

## ⏸️ PARKED 2026-07-13 — RESUME HERE

**State:** U11+U12 implementation DONE and verified green (**pytest 312**, **query 241/241**).
All work is **uncommitted** (per repo convention — commit on user request). Nothing is broken;
this is a clean stopping point *between* implementation and the mandatory review gate.

**Uncommitted working tree (from `git status`):**
- New: `docs/plans/m3-executor-landing2.md` (architect design-patch), `server/falkorchat/trigger.py`,
  `server/tests/test_executor_produced.py`, `server/tests/test_trigger.py`.
- Modified source: `server/falkorchat/{api,app,config,executor,guards,schemas,services,tools}.py`,
  `server/.env.example`; test files `test_{api,app,executor,executor_agent,guards,services,tools}.py`.
- Modified docs: `docs/plans/m3-executor.md` (§7 m-1 amendment), this coordination doc.
- **Note the source tree is `server/falkorchat/` (NOT `server/src/falkorchat/`).** venv: `server/.venv`.

**Verify-on-resume (env: FalkorDB must be up — `docker ps | grep falkor`):**
```
cd falkor-chat && ./scripts/test_queries.sh                 # expect 241/241
cd server && .venv/bin/python -m pytest -q                  # expect 312 passed
```

**Remaining Landing-2 work, in order:**
1. **▶ NEXT — Analyst impl-review gate (NON-NEGOTIABLE K-022 done-condition).** Dispatch `analyst`
   to statically review the U11+U12 diff (the uncommitted working tree above) against
   `docs/plans/m3-executor-landing2.md`, the plan §6/§7, and the review's closed items
   (M-1/m-1/m-3/n-2). Deliverable: `docs/reviews/m3-executor-landing2-impl.md` with a verdict.
   A "needs changes" loops back to tdd-engineer (`SendMessage` to agent `a7f82a9555f62278c` keeps
   context), then re-review. Confirm Option B, the byte-for-byte loop, and the M-1 net specifically.
2. **U13 (coder, +devops if env)** — `scripts/seed_workflows.sh`: publish + materialize the triage
   def (3 `type:'agent'` steps + guards per `m3-executor.md` §8 table) additive-only into `ws:acme`,
   idempotent; register the trigger. Architect flagged: wrap a Python one-shot over the **service
   layer** (real validation/start-key derivation), run after `bootstrap_schema` + `seed_demo`,
   key/version matching `TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` config. **Only here** should
   `WORKFLOW_ENABLED` be flipped on in `scripts/start_server.sh` (D6 left it off — flipping before a
   def is published makes @mention-to-start silently no-op).
3. **U14 (tdd-engineer)** — one `live`-marked e2e test exercising AC-1…AC-4 (real LM Studio, like the
   M2 responder smoke): assert `TRIGGERED_BY`, StepRun NEXT trace, run `done`. **Env dependency:**
   running it live needs LM Studio at :1234 + `FALKORCHAT_ENABLE_AGENT`/`WORKFLOW_ENABLED` on at the
   workspace embedding dim — surface to user if unreachable; the test is written marker-gated so the
   network-free baseline stays green regardless.
4. **U15 (qa-engineer)** — acceptance pass AC-1…AC-6, black-box; test plan + report under
   `docs/test-{plans,reports}/`. U12's REST reads give the observability hook. This is K-025.

**Doc rollup owed at close (coordinator-driven, via an implementer — teco's Write is docs/plans-only):**
`docs/HISTORY.md` (Landing-2 entry) + `docs/BACKLOG.md` (K-022 Landing 2 → done; K-023/K-024/K-025
status) once the gate + U13/U14/U15 land. **Carried nit n-1** (add `node_note` to the QUERIES §12.10 /
DESIGN §5 trace-kind enumeration) — fold into that rollup.

- 2026-07-15 — **Resumed from PARKED.** Env re-verified green (FalkorDB restarted; query **241/241**,
  pytest **312** — matches parked state, no drift). U11+U12 was committed as **514346b** ("parked
  pre-gate"). **Analyst impl-review gate ✅ dispatched on the 514346b diff** →
  `docs/reviews/m3-executor-landing2-impl.md`. **Verdict: approve-with-suggestions, 0 blocker / 0
  major / 3 minor / 3 nit.** All four mandated confirmations HOLD: Option B correct + §2.1 A/B/C loop
  byte-for-byte unchanged (`_link_emissions` at executor.py:342, above branch dispatch); M-1 fault net
  closes the Landing-1 zombie-run major; PRODUCED-not-EMITTED (live-test-asserted); one-handler
  trigger XOR responder (api.py:155-161, test-asserted). m-3/n-2/m-1/thread-context/U12 all confirmed
  closed; analyst re-ran pytest → 312. **The K-022 done-condition for U11+U12 is SATISFIED — no
  re-review cycle.** Minors → doc rollup at close: **m-A** (n-1: `node_note` missing from QUERIES
  §12.10 / DESIGN §5 trace-kind enum), **m-B** (HISTORY/BACKLOG entry owed at gate exit). **m-C**
  (every agent node does an unbounded `read_thread` then slices app-side — O(thread-length)/step) →
  non-blocking Landing-2-scale follow-up, routed to tdd-engineer/architect. **Proceeding to U13
  (coder — seed_workflows.sh).**

- 2026-07-15 — 💥 **CRASH + RECOVERY.** The machine crashed after U13's implementation landed but
  before it was logged/committed. **Nothing was lost.** Recovery performed by teco:
  - **Env rebuilt:** the `falkordb-dev` container did not survive; the **`falkordb-data` volume did**.
    Restarted via `./scripts/start_falkordb.sh -d`.
  - **No drift:** query suite **241/241**, pytest **312 passed** — both match the pre-crash parked
    numbers exactly. Working tree intact; no git stash/merge/rebase leftovers.
  - **U13 ✅ (coder) — recovered and live-verified by teco.** `scripts/seed_workflows.sh` (204 lines,
    untracked), `scripts/start_server.sh` (5→6 stages, `FALKORCHAT_WORKFLOW_ENABLED` default 1 with
    the def seeded *before* uvicorn), `AGENTS.md` script-table row. teco ran the script **twice**:
    publishes+materializes `triage@v1` (3 `type:'agent'` steps, 2 transitions — matches
    `m3-executor.md` §8 incl. the D5 unconditional research→answer guard), second run a clean
    `already present — no-op` on both def and snapshot. Wraps a Python one-shot over the **service
    layer** as the architect flagged. Done-condition met.
  - **U14 ❌ not started** — no artifacts on disk (no live-marked workflow e2e test; the
    `maybe_trigger` hits in `test_api.py`/`test_trigger.py` are D6's). The in-flight dispatch, if
    any, produced nothing. Resume = a clean U14 dispatch, no salvage needed.
  - **⚠️ New ops gotcha (surfaced during recovery, not previously documented):** `test_queries.sh`
    **deletes the global `reference` graph** at teardown, which wipes the published `triage@v1` def
    (the ws snapshot survives). Running the query suite therefore de-fangs `@mention`-to-start until
    `seed_workflows.sh` is re-run. `start_server.sh` self-heals this by seeding on every start.
    **→ fold into the doc rollup at close** (AGENTS.md script table / seed_workflows row).
  - **Env for U14 confirmed:** LM Studio reachable at `:1234` (HTTP 200) — the live e2e is unblocked.
  - **▶ NEXT: U14 (tdd-engineer)** — marker-gated live e2e over AC-1…AC-4. Then U15 (qa-engineer,
    K-025), then the doc rollup (HISTORY/BACKLOG + m-A/n-1 + m-B + this gotcha).

- 2026-07-15 — **U14 ⚠️ DELIVERED RED (tdd-engineer) — the live triage flow does NOT reach `done`.**
  `server/tests/test_workflow_live.py` (new, `live`-marked e2e over AC-1…AC-4) + `live` marker
  registered in `server/pyproject.toml` + AGENTS.md doc updates. **teco re-verified: default pytest
  **312 passed, 1 deselected** (baseline exactly held); gating is `addopts = -ra -m "not live"` —
  a real deselect, not just a reachability skip (LM Studio *is* up, so a skip alone would have let
  the live test run on every default `pytest`). Query suite untouched by design — **zero**
  graph/DDL/QUERIES change (confirmed: no `QUERIES.md`/`DESIGN.md`/`bootstrap_schema.sh` in the diff),
  so **241/241** stands from the recovery run. The test is **deliberately RED** — it pins two real
  defects; the engine was correctly **not** touched (constraint #3 honored).
  - **Defect A — the intake→research guard can NEVER fire (structural, not prompt calibration).**
    **teco verified in source:** `executor.py:566` passes `thread=None`; `guards.py` declares
    `thread: Any` (line 74) and **never uses it**; `_extract_understanding` (line 135) reads only
    `step_output`-as-JSON and `ctx["understanding"]`, falling back to `{}`. In the shipped flow both
    are guaranteed empty — the intake `systemPrompt` (plan §8, seeded verbatim) never asks for an
    `understanding` object, and the executor never writes `ctx` back (known **m-2**). The DS note's
    prescribed **RECENT-TURNS fallback (N=6) does not exist**; `_build_llm_judge` sends only
    CONDITION+UNDERSTANDING. Live evidence: judge rationale *"The user has not provided any
    information to research their request"* on the very turn the node said *"Thank you for providing
    all the details, Alice"* — the fingerprint of an empty understanding, not a rich transcript.
    Observed 4× `intake`, every judgment `False`. **The DS note's own risk #4 predicted this.**
    **⇒ AC-2b, AC-3, AC-4 are unreachable live. U15 cannot pass until this is fixed.**
  - **Defect B — a hallucinated `@mention` fails the whole run.** **teco verified in source:**
    `executor.py:478` dispatches with **no `try/except`**, so any tool error propagates to the M-1
    net → `fail_run`; `tools.py:208` forwards `mentions` **unvalidated** → `UnknownMemberError(['alice'])`.
    Cause: `_assemble_messages` folds thread turns in as `f"{speaker}: {text}"`, so the model sees
    `Alice: …` and mentions her. **Irony: the U11.3 thread-context feature added *for* AC-2 induces
    this.** Reproduced 2 of 3 runs. **Side effect — observability gap:** the failing step's trace is
    lost (`_trace_step` runs after `_record`, which never happens) exactly when it's most needed.
  - **Test-design calls (teco concurs):** throwaway **`ws:live`** at the **probed** embedding dim —
    `ws:test` is **dim 4**, where a real 1024-dim vector is silently accepted then drops out of ANN,
    so AC-3 would have "passed" while retrieving nothing (trap now documented). Test drives the real
    `seed_workflows.sh` rather than a copied def, so it tracks what ships. Left **RED, not `xfail`ed**
    — the red is the regression guard for the fix.
  - **⏸️ PAUSED FOR USER DECISION — Defects A+B are new work beyond U14/U15's scope.** Neither is the
    tdd-engineer's to fix under the locked-code constraint. Routing options put to the user; U15
    (qa-engineer) is **blocked on Defect A** and not dispatched.

- 2026-07-15 — **Defect-A design ✅ (architect)** → `docs/plans/m3-guard-thread-context.md`. Verdict:
  **structural defect (a declared seam no callee honors), localized fix** — ~40 lines, additive, **zero
  graph/DDL/QUERIES change, no graph-dba gate**, and `_drive_loop` **not edited at all** (the §2.1 lock
  satisfied by construction). Design: the thread turns `_run_agent_node` **already reads**
  (`executor.py:422`) ride out on a new `StepResult.thread` → `_select_transition` passes
  `thread=result.thread` (the one-line fix at :566) → `guards._recent_turns(thread, n=6)` with the DS
  precedence (**understanding primary; turns only when empty**) → `_build_llm_judge` renders the real
  §Q1 prompt. Having the guard read the thread itself was **rejected**: it would edit the locked loop's
  call line *and* double reads/step (m-C worse). Riding the existing read = **zero extra reads, m-C
  neutral**. Positions: **m-2 not fixed** (unnecessary — `step_output` is fresh in the same loop
  iteration, incl. across resume; fixing it would re-open the graph-dba gate for zero benefit);
  **m-C not fixed** (neutral by construction, inherits its eventual fix free).
  - **R-1 (highest risk, teco-verified):** `guards.py:132` substring-matches `_NEGATION_CUES`, which
    includes **`"more info"`** (line 39). A transcript-fed judge writes wordier rationales — *"…no more
    info is needed"* trips the cue → **forced suspend on a correct advance, looking IDENTICAL to Defect
    A**. The plan carries a rationale×expected contract table to drive minimal cue tightening.
  - **R-3 (sequencing):** the live test **cannot** go green on the Defect-A fix alone while Defect B
    reproduces — this fix is precisely what first drives the flow through B's unguarded `dispatch`.
    **Land B first.** (B is in flight with tdd-engineer.)
  - **OQ-2 — def-prompt change:** architect recommends it as a **genuine complement, landed SECOND**
    (after S1–S4 is live-verified alone): the seam fix alone leaves the shipped def permanently on the
    DS's **degraded fallback** path; the def prompt is what reaches the **primary** extract-then-judge
    path. Sequencing it second keeps "green for the right reason" attributable. Blast radius nil.
  - **OQ-4 — ⚠️ THE JUDGE IS WIRED LIVE UNCALIBRATED (teco-verified).** `server/tests/eval/
    golden_guards.jsonl` **does not exist** (nothing matches `golden*` in the repo). The DS note §272
    is explicit: *"Wire live only if **κ ≥ 0.6 AND false-advance ≤ 10%**"* and calls it "the executor's
    reliability gate". **Stakeholder decision — not the implementer's.**
  - **OQ-5 — ⚠️ NEW FINDING: the `answer` node never sees the research findings (teco-verified).**
    Findings live only in `StepRun.output`, which **nothing reads** (`executor.py` uses `result.output`
    only for the *current* step's guard :565, its own StepRun :589, trace :628 — never a *prior* step's);
    research posts nothing to the thread; `r.ctx` is only `SET` in `fail_run` (`repository.py:1217`),
    never on success (m-2). **⇒ AC-4 can pass STRUCTURALLY while being ungrounded in substance** — the
    live test asserts a `PRODUCED` reply, not its provenance. A real fix needs the m-2 ctx-write
    ⇒ **graph-dba gate**. Out of scope for the Defect-A patch; flagged so U15 judges AC-4 with eyes open.
  - **Review-gate posture (justified trim, stated explicitly):** no separate analyst *plan* review for
    this patch — it is a localized additive ~40-line fix whose every claim the architect verified in
    source and teco spot-verified independently (R-1, OQ-4, OQ-5 all confirmed). **The analyst
    impl-review gate remains NON-NEGOTIABLE**, and R-1 is exactly the class of silent failure it exists
    to catch.
  - **⏸️ PAUSED — OQ-4 + OQ-5 are stakeholder decisions put to the user.** Defect-A implementation not
    dispatched pending them (and pending Defect B landing first, per R-3). U15 still blocked.

## Locked decisions — round 2 (from the user, 2026-07-15)
- **D6 — OQ-4 = build the golden set (option a).** The DS reliability gate is **honored, not waived**:
  `server/tests/eval/golden_guards.jsonl` (20–30 hand-labeled cases) is built and the judge is wired
  live **only if κ ≥ 0.6 AND false-advance ≤ 10%** (DS note §272). Rationale: calibration is cheap
  relative to the credibility of every guard verdict, and the DS note's track record here is good — it
  **predicted Defect A** (its risk #4). Owner split: **data-scientist authors the golden set + the
  calibration protocol** (method/labeling = judgment); **tdd-engineer implements the eval harness** and
  runs it (DS is advisory, never implements). Calibration runs **after** the Defect-A fix — the judge
  prompt changes (gains the RECENT TURNS block), so calibrating the current judge would measure a
  prompt that is about to be replaced.
- **D7 — OQ-5 = accept for the proof, document the limitation (option b).** The `answer` node not
  seeing the research findings is a **scope** question, not a correctness trap. **AC-4 is
  STRUCTURAL-ONLY** for this cut — the live test asserts a `PRODUCED` reply, not its provenance. This
  must be **explicit in U15's brief and its test report**, and carried as a follow-up (needs the m-2
  ctx-write ⇒ graph-dba gate). The triage flow demos green while the middle step is, in substance,
  decorative — say so plainly rather than let AC-4 imply more than it proves.
- **D8 — OQ-2 = def-prompt change lands SECOND**, as the architect recommended: after the S1–S4 seam
  fix is live-verified **alone**. The seam fix alone leaves the shipped def on the DS's *degraded*
  fallback path; the def prompt reaches the *primary* extract-then-judge path. Sequencing it second is
  what keeps "green for the right reason" attributable to a single change.

- 2026-07-15 — **D6/D7/D8 locked; DS golden-set authoring dispatched** (parallel with Defect B, which
  is still in flight — the two share no files). Remaining chain, order forced by R-3: **Defect B →
  Defect A impl (S1–S4) → live-verify alone → def-prompt (D8) → eval harness + calibration run (D6) →
  analyst impl gate → U15 (AC-4 structural-only per D7) → doc rollup.**

- 2026-07-15 — **Defect B ✅ FIXED (tdd-engineer).** Reproduction tests first (7 new, each confirmed RED
  for the right reason before the fix). Localized: two `try/except`, **no new mechanism**.
  `executor._handle_tool_call:492` — `except ServiceError` around `dispatch` → trace + `return
  "error: …"`, the **exact existing convention** for ungranted/malformed calls; `tools.PostMessageTool.run`
  catches `UnknownMemberError` → an error string naming the **id-vs-display-name** confusion (a generic
  error is a blind retry; a specific one lets the model *fix* it). Both, deliberately: the executor catch
  is the survival property, the tool catch is the corrective signal. Pre-flight mention validation
  **rejected** — `services._validate_and_derive_role` already resolves mentions; validating in the tool
  would duplicate that graph read to prevent an error the write reports anyway.
  - **teco re-verified the safety-critical claims IN SOURCE (not taken on trust) — the class hierarchy
    makes the fix structurally safe:** `services.py:58` `ServiceError(Exception)` → `services.py:70`
    `UnknownMemberError(ServiceError)` is **caught**; `tools.py:308` **`HumanHandoffSignal(Exception)`
    is NOT a `ServiceError`**, so it cannot be swallowed and still reaches suspend; the catch at :492 is
    **narrow, never blanket** (its own comment at :497 says so); the **M-1 net at :306/:311 is intact**.
    All three proved by the agent's tests *through dispatch*, not asserted. pytest **312→319** (+7);
    query **241/241** (zero graph/DDL/QUERIES change); ruff clean; imports network-free. The `live` test
    stays deselected and **still RED — Defect A untouched, exactly as briefed.**
  - **Judgment calls (teco concurs):** dropped its own `tool_error` trace kind — it would force a QUERIES
    §12.10 enum change ⇒ re-opening the graph-dba gate; reused the established kind+marker convention
    (`tool_result` / `ERROR: …`) instead. **Trace loss fixed for this path** (node survives → `_record`
    runs → trace emitted; asserted in the graph), **deferred in general** (an engine fault still loses its
    trace — needs the locked `_record`/`_trace_step` ordering changed; correctly **not forced**). Docs:
    `m3-executor.md` §2.2 step 3 amended with the tool-error rule + its narrow boundary; `AGENTS.md`
    needed nothing (it never described tool-error semantics — correct call, not an omission).
  - **Rejected alternative (flagged for the record):** having `post_message` silently **drop** unknown
    mentions and post anyway — recovers in zero extra iterations but discards model intent and hides the
    bug. Available if we ever want that trade.
  - **Learnings inbox filed ✅** (teco confirmed): `materialize_snapshot` with `transitions=[]` raises
    `IndexError` at the caller — an **unguarded empty-`UNWIND` collapse**, a *second* instance of the
    quirk `AGENTS.md` documents only for the mentions write-block. Durable; for cobb to promote.
  - **R-3 now satisfied → Defect A implementation unblocked and dispatched (S1–S4).**

- 2026-07-15/16 — ⚠️ **Both parallel agents cut off by a provider session limit.** teco established the
  real on-disk state (did **not** trust the abort summaries):
  - **Defect A impl — PARTIAL, tree left RED.** The agent wrote its **Step-1 RED tests**
    (`test_guards.py`, incl. the R-1 cue contract) and was cut off **before implementing the fix**.
    `executor.py:596` still reads `thread=None` (line moved 566→596 by the Defect-B fix). **pytest is
    27 failed / 314 passed / 1 deselected.** This is **mid-TDD, not corruption** — the RED tests are
    correct-by-design and awaiting their fix. **Do not "repair" the baseline by deleting them.**
  - **DS golden set — NOTHING DURABLE.** No `server/tests/eval/` dir, no `golden_guards.jsonl`, no
    calibration doc. Its work existed only in context. Its last signal is a **real methodological
    finding worth preserving**: *"the FAR arm behaves very differently from κ … re-run under the
    judge's actual designed operating point — **bias-to-suspend** — since symmetric accuracy is the
    wrong model here."* That is exactly the kind of insight a fresh dispatch would lose.
  - **Recovery = resume both via SendMessage (context intact), not fresh dispatch.** Differs from the
    Landing-1 precedent (D2a/D4a were re-dispatched fresh) — **because those had made no edits**. Here
    one agent has half-written RED tests on disk and the other holds unsaved analysis; a cold agent
    would misread the RED tree as breakage and the DS insight would be lost outright.
  - **Coordination learning:** a session-limit abort is **not** uniformly "no edits made". Check the
    tree before choosing resume-vs-redispatch; the right call depends on what actually landed.

- 2026-07-16 — **D6 golden set ✅ (data-scientist, resumed).** Deliverables **teco-verified on disk**:
  `server/tests/eval/golden_guards.jsonl` (**26 cases**; schema richer than §272 — adds `tier`/`path`/
  `r1_probe`), `docs/plans/m3-guard-calibration.md` (the protocol), and `m3-executor-ml.md` §272/risk-#1
  **struck through + marked SUPERSEDED with bidirectional links** (teco read the row). No source touched.
  Separate doc rather than rewriting the note: *"a note that predicted Defect A shouldn't be silently
  rewritten — better to state the disagreement explicitly."* **teco concurs.**
  - **⚠️ THE APPROVED GATE WAS THE WRONG INSTRUMENT — D6's thresholds are superseded (needs user
    ratification).** The resumed FAR-vs-κ finding (preserved from the pre-cutoff context — it would have
    been lost in a cold restart) is **structural, and the killer argument is unanswerable: an
    always-suspend judge scores a PERFECT 0% FAR.** FAR is a function of specificity only; `_coerce_verdict`
    pins specificity near its ceiling **by design**, so at the designed operating point **FAR is nearly
    information-free and κ is covertly doing an advance-recall gate's job** — noisily, and contaminated by
    case-mix (same judge: E[κ] 0.70 at 11/10, 0.55 at 18/3). **The gate contradicted the design it was
    written to protect**: the note's own Q1 bias-to-suspend decision made the raters asymmetric *two
    sections before* it reached for a symmetric metric. **Instrument bug, not a reason to distrust the
    note — risk #1's intent (never wire an uncalibrated judge) is preserved intact.**
  - **Replacement gate (§4):** **false-advance ≤ 10% (screen) AND advance-recall ≥ 0.80**; κ demoted to a
    reported diagnostic-with-marginals. Advance-recall is a **safety** arm, not a usefulness one — a judge
    that won't advance burns the 3-round ceiling and then **force-advances with no judgment applied**,
    which is the exact harm FAR exists to prevent. **Gate failure ⇒ block the wiring, no override, no
    compensating with `maxSteps`. κ < 0.6 with both gates passing ⇒ do NOT block** (the behavioral change).
  - **Sample-size honesty (unprompted, and the mark of a trustworthy deliverable):** **26 cases CANNOT
    support a κ ≥ 0.6 claim** — observed κ=0.6 at N=21 has 95% CI ≈ **[0.24, 0.90]**; a true-0.85 judge
    fails 20% of the time; a judge exactly at true κ=0.60 passes **59% — a coin flip on the gate's own
    boundary**. The *kept* arm is no better: 0/10 perfect on FAR still yields **[0%, 27.8%]**; bounding
    true FAR ≤10% needs ~30 suspend cases at zero failures (**~50–60 total**). Replicates don't rescue it
    (repeated measures on the same inputs). **⇒ D6 is right but under-specified about the DIRECTION of
    inference: its *rejection* power is real (a true-0.5 judge fails advance-recall 97% of the time) —
    that IS the decision D6 needs — but PASS ⇏ CALIBRATED**, only "no blocker found at a sample size that
    could only have found a large one." Made **structural, not a matter of the reader's care**: §8 requires
    that caveat **verbatim next to the verdict line**.
  - **Case mix:** 21 gated (11 advance / 10 suspend ≈52/48) + **5 boundary cases reported-not-gated**
    (all 26 labels are the DS's own — boundary cases excluded from both gates precisely to avoid gating on
    one labeler's policy preferences). 15 understanding-fed / 7 turns-only, with `ca-01/tn-01` + `ca-02/tn-02`
    as deliberate **near-pairs carrying identical evidence through both paths — the delta IS the
    fallback-cost estimate for risk #4**. Load-bearing: `cs-04` (`missing:[]` but vacuous) vs `ca-04/05/08`
    (`missing` non-empty but immaterial) jointly detect **a judge reading `len(missing)` instead of assessing
    sufficiency — the one failure mode that passes both gates**. `tn-05` catches the judge crediting its own
    unanswered clarifying question.
  - **R-1 finding (teco-verified):** R-1 is **already owned deterministically** by `test_guards.py:279+`
    (pins both directions incl. the exact `"no more info is needed"` string). **The golden set is NOT the
    R-1 detector** — its distinct job is live incidence via a **coercion-flip rate** (raw vs. coerced
    decision — the only thing distinguishing R-1 from Defect A from outside). Honest limit stated:
    fixtures can't force rationale wording, so a **zero flip rate is weak evidence**.
  - **Schema finds:** §272's schema **cannot** be splatted into `evaluate_guard` — the real signature takes
    no `understanding`/`turns`, it **derives** them; the harness must synthesize `step_output`/`ctx`/`thread`
    (better — runs the real precedence + coercion). §104 and §272 **disagree on field names** — flagged and
    chosen, not silently picked. **`authorType` is `labels(author)` → a LIST, not a string** (plan prose says
    otherwise) — inbox-filed with the κ finding.
  - **Couldn't measure (stated):** the live judge (fix in flight — *every number is a simulated property of
    the metric, not a measurement of the judge*); self-preference (same 4B emitting **and** judging);
    inter-human agreement on the 5 boundary labels. **"Nobody should compute a headline accuracy over all
    26 — it would gate on my policy preferences."**
  - **⏸️ NEEDS USER RATIFICATION: D6's gate thresholds change** (κ≥0.6 ∧ FAR≤10% → **FAR≤10% ∧
    advance-recall≥0.80**), and "pass" is **weak evidence** at N=26 (~50–60 needed for a real bound).

## Locked decisions — round 3 (from the user, 2026-07-16)
- **D9 — the superseded gate is RATIFIED.** D6's thresholds are replaced by
  **`m3-guard-calibration.md` §4: false-advance ≤ 10% (screen) AND advance-recall ≥ 0.80**; κ is a
  **reported diagnostic**, not a gate — **κ < 0.6 with both arms passing does NOT block**. Gate failure
  ⇒ **block the wiring**, no override, no compensating with `maxSteps`. Rationale accepted as decisive:
  **an always-suspend judge scores a perfect 0% FAR**, so the old gate could be passed by a judge that
  never advances — it contradicted the bias-to-suspend design it was written to protect. Risk #1's
  intent (never wire an uncalibrated judge) is **unchanged**; this is an instrument fix, not a waiver.
- **D10 — this cut's calibration is a SCREEN, not a certification (accepted with eyes open).** At N=26,
  **pass ⇏ calibrated** — it means *"no blocker found at a sample size that could only have found a
  large one."* The gate's **rejection** power is what we're buying (a true-0.5 judge fails advance-recall
  97% of the time). The §8 verbatim caveat next to the verdict line is **mandatory and must survive to
  the U15 test report** — it is not editorial garnish.
- **D11 — golden-set expansion to ~50–60 cases is a FOLLOW-UP, not a U15 blocker.** Logged for the
  backlog (a real FAR ≤ 10% bound needs ~30 suspend cases at zero failures). Known limit carried with
  it: **all labels are one labeler's** — expansion should add a second labeler for the boundary tier,
  or it buys precision without buying independence.

- 2026-07-16 — **D9/D10/D11 locked.** Eval-harness implementation (tdd-engineer, against the DS §4/§7
  protocol) **deliberately NOT dispatched yet** — it must run against the **fixed** judge, and the
  Defect-A agent is still live in `guards.py`/`executor.py`/`app.py`. Sequencing holds: **Defect A
  live-verified alone → def-prompt (D8) → eval harness + calibration run (D9/D10) → analyst impl gate
  → U15 (AC-4 structural-only per D7) → doc rollup.**

- 2026-07-16 — **Defect A ✅ FIXED + live-verified alone (tdd-engineer, resumed). DEFECT A IS DEAD.**
  **teco re-verified:** pytest **348 passed, 1 deselected** (RED 27→0; baseline 319 **+29 pins**);
  `guards.py:136` `recent_turns = [] if understanding else _recent_turns(thread)` — the DS precedence
  (**understanding primary; turns only when empty**) exactly; `executor.py:607` `thread=result.thread`
  — the one-line seam fix; **query 241/241 with ZERO graph change** (teco confirmed `QUERIES.md`/
  `DESIGN.md`/`bootstrap_schema.sh`/`seed_workflows.sh` have **no diff** — D8 held, no graph-dba gate).
  Imports network-free (proved with `FALKORDB_HOST=10.255.255.1`). **Locked loop:** agent
  **AST-extracted `_drive_loop` and compared byte-for-byte to HEAD (2839b → 2839b)** — *not* inferred
  from hunk ranges; `record_step_and_advance` untouched; Defect-B's hunks undisturbed. ⚠️ **teco
  confirmed `executor.py` changed (+50/-9) but did NOT independently re-verify the `_drive_loop`
  byte-identity — the analyst impl gate must confirm it.**
  - **Live: the flow REACHES `done` and the judge reasons from real evidence.** Every run:
    `intake → intake → research → answer`, `status=done`, `stepCount=4`, **2 clarifying rounds** (inside
    the DS 3-round ceiling, not the `MAX_CLARIFY_ROUNDS=4` headroom). Judgment 1 `False` (correct on the
    vague opener); judgment 2 `True` — *"The user provided a clear timeline of the issue, including the
    specific version deployment, the error type, and that rolling back fixed it…"* — **cites the user's
    actual facts (v4.2, 502, rollback)**, i.e. plan §5 Step-4 criterion 2: reasoning from evidence, not
    being talked into yes. **This is green for the right reason**, on the DS's *degraded* turns-only
    fallback path, as designed for this cut.
  - **⚠️ NEW — DEFECT C: the `answer` node doesn't reliably call `post_message`. AC-4 passes ~1 in 3.**
    **The engineer's first `-m live` run PASSED — and it did not stop there.** It re-ran **7 more times:
    ~2 pass / 6 fail**. It nearly reported a false green and **caught itself**. Failure is always the
    same and always **downstream** of the fix: `AssertionError: the answer node never posted a reply
    (AC-4); posts came from: ['intake','intake','intake']`. The node produces a **good grounded answer
    as final text** but doesn't call the tool → no `PRODUCED` edge. **Not a Defect-A regression** (the
    guard fires identically in passing and failing runs); previously **invisible because the flow never
    reached `answer`**. **This is D4's accepted risk (4B function-calling reliability) materializing.**
    The test was **not** bent to force green.
  - **OQ-5 partially answered (inferred, not proven):** the answer node's text **is** well-grounded
    despite never seeing research findings — **it reads the thread turns directly**. OQ-5's *structural*
    concern stands (D7), but the observed symptom is **not** ungroundedness.
  - **R-1 bit — and demanded more than a tightening (teco-verified in source).** The cue table exposed
    **three** distinct modes, not one: (1) **negated cues** — `"no more info is needed"` tripped
    `more info`, `"nothing is unclear"` tripped `unclear` → fixed with a **polarity rule** (a cue
    preceded within **12 chars** by `no `/`not `/`nothing `/`never `/`n't ` **affirms**); the window is
    deliberately **too narrow to cross a clause boundary** (`"did not provide the version; more info is
    needed"` still suspends) — **erring narrow keeps failures on the safe over-suspend side**.
    (2) **`"no relevant"` embeds its own negator** — no window rule can resolve its polarity
    (`"no relevant details are missing"` vs `"no relevant information was provided"`) → **removed as
    unfixable-rather-than-tightenable**, documented in-place. (3) `"The user still needs to provide the
    version."` matched **nothing** → required a **new** cue (`"still need"`), which also replaces the
    coverage `no relevant` carried. The check itself + the DS bias-to-suspend policy are intact;
    `_coerce_verdict` untouched. **R-1 did NOT bite live** — the S4 prompt rule ("state only supporting
    evidence") kept the real rationale purely affirmative.
  - **Process notes worth keeping:** two of its own tests were wrong — **it fixed the tests, not the
    code**, and when T7/T8 passed on first write it **reverted S1 to prove they genuinely go RED** (they
    did — that is what makes them pins). The 9 signature failures in `test_executor.py`/`test_app.py`
    were **R-6 exactly as the plan predicted** (stub judges needing `recent_turns`) — signature updates
    only, **no assertions weakened**. **m-C pinned neutral** (T8: exactly one `read_thread`, not two).
  - **Docs (plan §10):** `m3-executor.md` §2.5 now states the **two-tier evidence contract** — *its
    silence is what let the seam ship*; `m3-executor-ml.md` §Q1 gained a dated implemented-at pointer.
  - **⏸️ Defect C needs triage before U15 — AC-4 currently passes ~1 in 3.** Options put to the user.

- 2026-07-16 — **D8/S5 + Defect-C attempt (coder) — DELIVERED AS TWO HONEST NEGATIVES.** *(Subagent
  carried a security flag; teco reviewed: destructive shared-state ops — `DETACH DELETE ws:acme` +
  `test_queries.sh` dropping `reference` — but **benign, disclosed, each followed by a re-seed**; teco
  verified graph state intact: only `ws:acme`+`reference` exist, both carry `triage@v1`; ws:live/livetest
  cleaned up. **Engine source 0-diff** on all four files — verified. Process note: coder doing unilateral
  graph surgery on the demo workspace should have routed through devops.)*
  - **Scope verified clean:** only `scripts/seed_workflows.sh` (prompt-only, no version bump, no config
    change) + `claude/coder/kaizen/inbox.md` in the tree; pytest **348** holds (no Python touched);
    query **241/241** (agent ran it, re-seeded after). *(Note: the agent's `m3-executor.md` §8 doc edit
    landed in the tree in time to be swept into teco's commit `aa8b813` — a benign concurrent-edit bleed.)*
  - **D8/S5 GOAL MET — the guard now runs on the PRIMARY path (proven).** Intake final text is a
    well-formed `{"understanding":{request,known,missing}}`; `_extract_understanding` parses it non-empty
    ⇒ `guards.py:136` runs the judge understanding-fed, `recent_turns=[]`. KEEP_WS trace confirms.
  - **⚠️ BUT S5 EMPIRICALLY *REGRESSES* THE FLOW — this is the critical finding.** The 4B fills
    `understanding.missing` with forensic demands (server logs, infra details) on **every** turn and never
    empties it; the **uncalibrated** judge reads `missing != []` and suspends. **Intake advancement
    (reaches `answer`), n=10 each: BEFORE (degraded turns path) 10/10 → AFTER (S5 primary path) 3/10.**
    On the degraded path the judge saw the raw human turns ("that's everything, please research now") and
    advanced; on the primary path it sees the 4B's structured missing-list and suspends. **This inverts
    the DS Q1 premise ("compact state beats transcript") for THIS 4B + an uncalibrated judge — OQ-4
    materializing on the primary path, exactly as the note warned "do not wire an uncalibrated judge."**
    Coder verdict (teco concurs): **S5 must NOT go live as-is.** Left in the tree as the mandated D8
    deliverable, uncommitted.
  - **⚠️ Defect-C PROMPT FIX DID NOT HOLD.** AC-4 posting: BEFORE 1/10 → AFTER 0/3 (small-n, but
    decisively not-reliable). Two measured mechanisms persist through the strengthened prompt: (a) answer
    node emits the answer as **plain text**, no tool call (D4's 4B risk); (b) it calls `post_message` with
    `mentions:["alice"]` (leaked from the folded `"{displayName}: {text}"` thread context) → §4 rejects →
    the 4B "recovers" by dropping the tool and emitting text. **Prompting a 4B does not hold this →
    needs an engine-level "terminal node must post" contract (architect; executor change, out of coder
    scope).**
  - **Immutable-def hazard CONFIRMED at source** (`repository.py:916` `_PUBLISH_CYPHER` `MERGE (st:Step) ON
    CREATE SET st.config` — existing config never updates; def + each ws snapshot go stale independently).
    Compounding trap: `test_queries.sh` drops `reference` but **not** `ws:acme`, so a naive re-seed leaves
    a def/snapshot **split-brain**. Coder filed it to its inbox. **→ doc rollup / AGENTS.md candidate.**
  - **OQ-5 corroborated LIVE:** the `answer` node asked the user for logs while `research` had already
    found the real root cause (pool 100→10) — the answer node cannot see research findings (m-2, D7).
  - **⏸️ STRATEGIC CROSSROADS — put to the user.** The executor *mechanism* is proven (Defect A dead,
    flow *can* reach `done`); the remaining blockers (judge over-weights `missing`; 4B won't reliably call
    the terminal tool) are **local-model-quality + engine-guarantee** issues, not executor bugs. What
    "done" means for the K-022 **proof** is now a scope decision, not another fix-by-fix delegation.

## Locked decisions — round 4 (from the user, 2026-07-16)
- **D12 — "C then B".** First a **one-shot capability probe** (C): put a **more capable model** behind
  the **unchanged, model-agnostic seams** and re-measure the two regressed numbers (intake advancement
  3/10 on the S5 primary path; AC-4 posting 0/3). Purpose: distinguish *"the design is sound, the 4B
  can't keep up"* from *"the design needs an engine crutch."* **Then B** regardless of the probe's
  outcome: **declare the executor mechanism proven and descope live-triage reliability** — U15 becomes a
  scoped acceptance (mechanism + AC-1/AC-5/AC-6 verified; AC-2b/AC-3/AC-4 recorded **model-gated,
  structurally-demonstrated**); the "terminal node must post" engine contract + judge calibration
  (D9/D10) + golden-set expansion (D11) become **K-023 follow-ups**, not U15 blockers. The probe is a
  **data point that informs how B is framed**, not a gate on reaching it.
- **Probe constraints:** config/env only — **no seam code change** (`guards`/`executor`/`tools`/`app`
  untouched), **no def-prompt change** beyond the S5 already in the tree (test the primary path as-is).
  If no more-capable model is available/loadable in LM Studio, that's a **blocker → surface to user**;
  do not download/install models. S5 stays **uncommitted** either way (it regresses on the 4B).

- 2026-07-16 — **D12 locked; capability probe dispatched (data-scientist).** Then → B framing (scoped
  U15) irrespective of result.
- 2026-07-18 — **Probe dispatch was lost when the user's machine crashed (nothing durable produced).
  RESUMED with a fresh data-scientist dispatch.** Env re-verified by teco before re-dispatch: FalkorDB
  restarted (`falkordb-data` volume survived); `reference` **still carries the S5 `triage@v1` def**
  (intake systemPrompt confirmed to contain the S5 `{"understanding":…}` JSON instruction — no def
  surgery needed, live re-seed is a safe republish); LM Studio up at :1234 with capable candidates
  available (`google/gemma-4-12b`, `qwen/qwen3.5-9b`, `openai/gpt-oss-20b`); network-free **pytest 348
  passed / 1 deselected** (no post-crash drift). `ws:acme` lost its def snapshot in the crash but the
  probe rides the live test's throwaway `ws:live`, so it's not needed. Probe brief: config/env-only
  model swap via `FALKORCHAT_LLM_MODEL`, S5 in tree, re-measure intake advancement (4B baseline 3/10)
  + AC-4 posting (4B baseline 0/3); deliverable `docs/plans/m3-capability-probe-ml.md`. Then B framing
  (scoped U15) regardless of result.

## Locked decisions — round 5 (from the user, 2026-07-18)

**Context — hardware downgrade breaks D12's premise.** The user's machine was downgraded **32GB → 16GB**
total RAM (repeated overload crashes; see `docs/plans/local-model-ram-budget-ml.md` +
`docs/plans/wsl2-memory-diagnostic.md`). D12's capability probe was specced to try a **more capable
(bigger)** model (`gemma-4-12b` / `qwen3.5-9b` / `gpt-oss-20b`) to see if it fixes the two regressed
numbers — **but none of those fit 16GB alongside FalkorDB + the co-resident embedder**, and even a passing
result would be unshippable. So the "would a *better* model fix it?" question D12's probe was built to
answer is now moot in the upward direction.

- **D13 — redirect D12's capability probe from an UPWARD (bigger-model) probe to a FITS-16GB model
  comparison for this task (user, Option A).** The probe now compares **`qwen/qwen3-4b-2507` (baseline)
  vs. Ministral 3 3B (2512, Apache 2.0)** — both fit the ~4–5GB chat-model budget (budget analysis:
  `local-model-ram-budget-ml.md` §1 + §7). Everything else about D12 is **unchanged and still binding**:
  **config/env-only** (`FALKORCHAT_LLM_MODEL` swap, **no `guards`/`executor`/`tools`/`app` seam change**,
  no def-prompt change beyond the S5 already in the tree), **S5 stays uncommitted**, and the probe
  re-measures the **same two numbers** — intake advancement on the S5 primary path (4B baseline **3/10**)
  and AC-4 terminal-post reliability (4B baseline **0/3**) — plus the guard-gate diagnostics
  (advance-recall ≥ 0.80 / false-advance ≤ 10% per `m3-guard-calibration.md` §4, reported not blocking
  here). **Then B regardless of result** (scoped U15) — D12's "then B" is untouched.
  - **Honest expectation set with the user:** the live blockers are a 4B being *too weak* to reliably
    call the terminal tool + an uncalibrated judge over-weighting `missing`. Ministral-3B is *smaller/
    weaker* than the 4B, so the probe is **most likely confirmatory** (Qwen3-4B is the best that fits;
    blockers persist ⇒ they are K-023 engine/calibration follow-ups, not a model choice). The user wants
    the Ministral-vs-Qwen number **on record** for this task regardless — that is the deliverable.
  - **⚠️ Prerequisite / likely blocker — Ministral not yet in LM Studio.** LM Studio currently exposes
    `gemma-4-12b` / `qwen3.5-9b` / `gpt-oss-20b` (per the 2026-07-18 env check), **not** Ministral 3 3B.
    D12's "do not download/install models" constraint stands, so the probe **cannot run until the user
    loads Ministral 3 3B (2512) into LM Studio**. If unavailable at run time → surface to user, do not
    self-provision. Owner split unchanged: **data-scientist** owns the probe method note + judgment
    (redirect `m3-capability-probe-ml.md`); it runs the measurement once the model is loadable; it never
    edits seams.
  - **Prior probe dispatch:** the 2026-07-18 fresh data-scientist probe (upward candidates) is
    **superseded by D13** — if it is still running from a prior session it should be stopped/ignored;
    its bigger-model measurements are moot under the RAM ceiling.

- 2026-07-18 — **D13 locked; probe redirected.** Coordination doc updated (this entry). Data-scientist
  dispatched to redirect `docs/plans/m3-capability-probe-ml.md` to the Qwen3-4B-vs-Ministral-3B fits-16GB
  comparison spec (ready to run once Ministral 3 3B is loaded in LM Studio). Then B framing (scoped U15)
  regardless of result. RESUME POINT is now: **load Ministral in LM Studio → run the redirected probe →
  B (scoped U15)**.

- 2026-07-19 — **Probe note ✅ DELIVERED + teco-verified on disk.** `docs/plans/m3-capability-probe-ml.md`
  (22KB, 8 sections + provenance, complete/untruncated) — the redirected Qwen3-4B-vs-Ministral-3B fits-16GB
  spec. ⚠️ **Recovery note:** the data-scientist dispatch reported *failed* on a session limit with an
  abort summary saying "no prior note exists, I'll author fresh" — but the tree shows the full on-brief file
  was written *before* the cutoff. Verified by content, not the summary (the Landing-1/2 crash lesson: check
  the tree). **No re-dispatch needed.** Key spec points: config/env-only `FALKORCHAT_LLM_MODEL` swap, S5 in
  tree, M1 intake-advancement (4B baseline 3/10, n=10) + M2 AC-4 terminal-post (4B baseline 0/3, conditional
  denominator); two structural cautions load-bearing — (1) M2's denominator is downstream of M1 so a weak
  intake starves the AC-4 sample, (2) a Ministral no-post may be a **parse artifact** (Mistral tool-call
  format) that must be classified before M2 is trusted; mandatory small-n honesty caveat travels with every
  number. **⛔ Run is gated on the user loading Ministral 3 3B (2512) into LM Studio** (`bartowski/
  mistralai_Ministral-3-3B-Instruct-2512-GGUF`, Q4_K_M ~2GB, ctx 8192, KV Q8; readiness check `curl -s
  localhost:1234/v1/models | grep -i ministral`) — LM Studio currently has only the bigger candidates.
  **RESUME POINT unchanged: user loads Ministral → dispatch the measurement run (data-scientist owns it) →
  B (scoped U15).**

- 2026-07-19 — **D13 probe RUN + COMPLETE (data-scientist). Ministral LOSES; Qwen3-4B stays.** User loaded
  Ministral 3 3B at **Q8_0** (fair-capability quant — teco+user call, folded into probe note §6 with the
  "Q4 reconfirm only if Ministral wins both" rule); teco brought FalkorDB up and verified both arms + embedder
  served. Results appended to `docs/plans/m3-capability-probe-ml.md` §"Results (run 2026-07-19)", teco-verified
  on disk (paired, same-session, n=10/arm, shipped wiring, S5 in tree):
  | Metric | Qwen3-4B (A) | Ministral-3B (B) |
  |---|---|---|
  | M1 intake advancement | **3/10** (re-measured = baseline exactly) | **0/10** |
  | M2 AC-4 terminal post | **2/3** (2 genuine posts, all §4-classified, 0 parse artifacts) | **not measurable (R1 — 0 reached `answer`)** |
  - **Verdict: no model swap — Qwen3-4B remains the shipped chat model.** The Q4_K_M reconfirm rule does
    **not** trigger (Ministral won neither number). Expected/confirmatory outcome per D13; the two live blockers
    are **K-023 engine/calibration work, not a model choice.**
  - **§4 classification (mandatory, done) — Ministral's 0/10 is NOT a clean capability loss, and it split into
    two findings that matter for K-023:**
    - **(K-023 finding 1 — NEW HARNESS DEFECT) the fuzzy-guard judge JSON parse is model-fragile.**
      `app._build_llm_judge` uses `complete()` + a **bare `json.loads`**; Ministral wraps its JSON in a
      ```` ```json ```` fence → **26/26 golden cases "unparseable judge output"** (one had a correct
      `decision:true` destroyed by the fence). Silently breaks **any** fence/prose-wrapping model. Fix =
      fence/prose-tolerant parse or structured output. **Does NOT affect the shipped Qwen path** (Qwen's JSON
      isn't fenced — parses fine), so it is not a B/U15 blocker; it is a K-023 robustness item.
    - **Even fence-fixed, Ministral would still lose:** fence-tolerant re-parse recovers its judge
      advance-recall to only **0.364** (vs Qwen **0.818**) — a genuine over-suspend weakness underneath the
      artifact.
    - **(K-023 finding 2) Ministral is actually BETTER on the terminal tool call** — native `post_message`
      3/3 in replay where Qwen emitted prose 3/3 (D4's 4B risk). Worth a targeted re-probe *only if* the judge
      is ever made model-robust. Does not change today's decision.
  - **Diagnostic that sharpens B:** Qwen's judge gate on clean golden inputs **passes both arms** (recall
    0.818, false-advance 0.067), so **Qwen's live 3/10 is a GENERATOR-half problem** (intake emits
    prose/forensic-`missing`), **not a judge problem** — informs how U15 frames the AC-2b/AC-3 model-gating.
  - **Env leftover (minor):** the run left throwaway `ws:live` + `ws:probe` graphs resident (RAM cost on the
    16GB box) — drop at convenience (`GRAPH.DELETE`, a mutating op → not teco's to run; devops/graph-dba or a
    manual `redis-cli`); not a blocker.

## ▶ RESUME HERE (2026-07-19) — probe done, into "B" (scoped U15)

D12/D13 "then B regardless" is now due. **B is user-locked in principle but its execution carries real
sub-steps and one genuine decision** — surfaced to the user, NOT auto-launched:
1. **DECISION owed — S5 disposition.** S5 is uncommitted and **regresses the shipped 4B** (intake advancement
   10/10 degraded-turns path → 3/10 S5 primary path). "B" ships Qwen, so S5 as-is makes the shipped flow worse.
   Options: **revert S5** (ship the 10/10 degraded turns path, guard on fallback), **keep S5 uncommitted/parked**
   (primary path exists for K-023 once the judge is model-robust + calibrated), or ship S5 accepting 3/10. User's call.
2. **Analyst impl-review gate (NON-NEGOTIABLE) still owed on the Defect-A/B fixes** — incl. the `_drive_loop`
   byte-identity teco flagged as not-independently-reverified. Must precede U15.
3. **U15 (qa-engineer)** — scoped acceptance per D12-B/D7: mechanism + AC-1/AC-5/AC-6 verified; AC-2b/AC-3/AC-4
   recorded model-gated/structurally-demonstrated; the D10 verbatim small-n caveat survives to the report.
4. **K-023 follow-ups** (NOT U15 blockers): judge-parse robustness (finding 1), terminal-node-must-post engine
   contract, judge calibration (D9/D10), golden-set expansion (D11), Ministral re-probe (finding 2).
5. **Doc rollup at close** (HISTORY/BACKLOG + carried nits m-A/n-1, m-B, the `test_queries.sh` drops-`reference`
   gotcha, the immutable-def hazard, and the two new K-023 findings).

## Locked decisions — round 6 (from the user, 2026-07-19)
- **D14 — S5 REVERTED (option a).** Ship the **degraded recent-turns fallback path** (intake advancement
  **10/10**), not the S5 primary extract-then-judge path (**3/10** on the shipped Qwen3-4B). The primary
  path is not reachable in this cut by design; it needs a **model-robust + calibrated judge** (K-023
  findings 1 + D9/D10) before it is worth re-landing. Rationale: "B" ships Qwen, so S5 as-is makes the
  shipped flow strictly worse; keeping it uncommitted would leave a permanently dirty tree and an
  ambiguous "what did we ship" story.
  - **⚠️ Scope nuance teco caught before dispatch — the uncommitted `seed_workflows.sh` diff bundled TWO
    independent changes, and a blanket `git checkout` would have overshot the user's decision.**
    (1) **S5** = the `intake` node's `{"understanding":{request,known,missing}}` JSON instruction — the
    thing that regressed 10/10→3/10. **Reverted.** (2) The **Defect-C prompt mitigations** — `intake`'s
    "deliver questions via `post_message`, never pass `mentions`" + the whole `answer`-node rewrite
    (mandate the terminal tool call; never pass `mentions`, since the folded `"{displayName}: {text}"`
    thread context leaks a display name into `mentions` → §4 rejects → the 4B drops the tool and posts
    nothing). These address a **different, separately-measured** failure mode, did not cause the intake
    regression, and are **KEPT**. Flagged to the user for override.

- 2026-07-19 — **D14 locked; two units dispatched in parallel** (no shared files):
  - **coder** — surgical S5-only revert of `scripts/seed_workflows.sh` (Defect-C mitigations retained,
    removal site commented so a future reader doesn't re-add S5) + a **live-graph parity report**: the
    `reference` graph still holds the **S5** `triage@v1`, and defs are effectively immutable on republish
    (`repository.py` `_PUBLISH_CYPHER` `MERGE (st:Step) ON CREATE SET st.config` — existing config never
    updates), so the script revert alone does **not** fix the live def. Coder investigates and reports the
    minimum safe procedure (version bump vs. delete-and-republish); **executing it is destructive
    shared-state work → user/coordinator call, not the implementer's.** Parity must be resolved before
    U15 runs live. Baseline to hold: pytest **348 passed / 1 deselected**; `test_queries.sh` deliberately
    NOT run (it drops `reference` and would worsen the split-brain).
  - **analyst** — the **non-negotiable impl-review gate**, target pinned to commit **`aa8b813`**
    (Defect-A guard seam + Defect-B tool-error survival). The `514346b` U11+U12 gate is already closed and
    is not re-reviewed. Five mandatory confirmations briefed: `_drive_loop` **byte-identity** (teco flagged
    it as not-independently-reverified), S1–S4 completeness incl. the DS understanding-primary precedence
    and the m-C-neutral "zero extra reads" claim, **R-1** negation-cue trap (`"more info"` substring →
    silent forced suspend that looks identical to Defect A), Defect-B catch **narrowness**
    (`HumanHandoffSignal` must not be swallowed; M-1 net intact), and zero graph/DDL/QUERIES change.
    Known-and-accepted items (m-2, m-C, OQ-5/D7, the K-023 judge/terminal-post items) explicitly fenced
    off so the gate doesn't re-litigate settled decisions. → `docs/reviews/m3-guard-thread-context-impl.md`.
  - **Still owed after these two: U15 (qa-engineer, scoped per D12-B/D7) → doc rollup.**

- 2026-07-19 — **S5 revert ✅ (coder) — teco-verified on disk, not taken on trust.** `grep` confirms
  `seed_workflows.sh` no longer instructs the `understanding` JSON (the only surviving `understanding`
  hits are the new do-not-re-add comment at lines 99/104); the **Defect-C mitigations are intact in both
  nodes** (intake :117-118, answer :156-159 — "deliver via `post_message`", "never pass `mentions`").
  Diff shrank +52/-4 → **+45/-3**. `bash -n` + `ast.parse` of the embedded heredoc clean; STEPS(3)/
  TRANSITIONS(2) untouched.
  - **⚠️ NEW ENVIRONMENT FINDING — the pytest baseline itself wipes `reference` (teco-verified in
    source).** `server/tests/conftest.py` `wf_repo` fixture runs `db.reference_graph(conn).query("MATCH
    (n) DETACH DELETE n")`. So it is **not only `test_queries.sh`** that drops the global `reference`
    graph — a plain `pytest` with the DB up destroys the published `triage@v1` while leaving the
    `ws:acme` snapshot intact: **the same silent split-brain**, from the command we treat as the routine
    baseline. `conftest` also `.delete()`s the whole `ws:test` graph per session. AGENTS.md's M1-server
    pytest bullet still advertises that suite as safe/network-free with no such warning → **doc-rollup
    item** (the coder flagged it; correctly did not expand scope).
  - **Baseline caveat — 348 was NOT observed, and the reason matters.** FalkorDB is **down**
    (teco-verified `docker ps` → 0 containers). Observed: **171 passed, 177 skipped, 1 deselected**;
    171+177+1 = **349 collected**, reconciling exactly with 348-passed+1-deselected — nothing lost,
    newly deselected, or failing. The coder **deliberately did not start the DB**, because doing so
    would have made the routine `pytest` a destructive act on `reference` (above). **teco concurs —
    that was the right call, and the restraint is the point:** the "green" pytest line with the DB down
    is not evidence the graph half ran. Filed to the coder's learnings inbox.
  - **AGENTS.md `seed_workflows.sh` row rewritten** (coder, in-scope — the revert invalidated the old
    "additive-only, idempotent" wording): "idempotent" = **create-only, not update**, with the
    `_PUBLISH_CYPHER` `ON CREATE SET` citation; independent def/snapshot staleness; both wipers named
    (`test_queries.sh` **and** the `wf_repo` fixture); and that landing a def edit needs an explicit act.
  - **Live-def parity — REPORTED, NOT EXECUTED (correct: destructive shared-state ⇒ user's call).**
    Recommendation **Option B, delete-and-republish `v1`** — keeps `key`/`version` matching
    `config.TRIGGER_DEF_KEY`/`_VERSION` with **zero code change**. **Option A (version bump) rejected as
    costlier than it looks:** `start_server.sh` neither forwards nor exports
    `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION`, so an env-only bump would seed `v2` nowhere and keep
    triggering `v1` — it needs `start_server.sh` + `.env.example` + a hardcoded echo changed, and leaves
    permanent dead-def clutter. **Option B's caveat is load-bearing:** `DETACH DELETE` on the Steps
    severs live runs' `AT_STEP`/`OF_DEF` (`record_step_and_advance` anchors on `AT_STEP`), making any
    suspended run permanently unadvanceable and **indistinguishable from a normal zero-row miss** ⇒ the
    read-only run-check must run first. **Sequencing rule: `pytest` → parity repair → verify → U15
    live**, never the reverse, or the wipe re-creates the split-brain and U15 measures a stale snapshot.
  - **Analyst gate re-dispatched** after the user stopped the first attempt (it had produced nothing —
    no review doc, tree untouched). Running against `aa8b813` with the corrected baseline expectation.
  - ⚠️ **teco process error, logged for honesty:** the baseline correction to the running analyst was
    sent via a **new `Agent` spawn instead of `SendMessage`**, starting a second, contextless analyst.
    Its output is to be **disregarded** — the gate of record is the re-dispatched one. No tree impact
    (analysts are read-only until their deliverable).

## Locked decisions — round 7 (from the user, 2026-07-19)
- **D15 — live-def parity = Option B (delete-and-republish `v1`), authorized for THIS dev environment.**
  User: *"for this env please do cleanup and recreate, no need to migrate anything."* ⇒ the Option-B
  run-breakage caveat is **explicitly accepted** — no `WorkflowRun` is worth preserving here, so the
  read-only run-check becomes informational (report what it finds) rather than a stop-gate. Option A
  (version bump) stays rejected per the coder's cost analysis (`start_server.sh` forwards/exports
  neither `FALKORCHAT_TRIGGER_DEF_KEY` nor `_VERSION`). **Scope note: this authorization is
  environment-specific — it is NOT a precedent for a shared/production graph, where severing
  `AT_STEP`/`OF_DEF` on live runs is a genuine data-loss event.**
- **Sequencing constraint (teco, load-bearing):** the repair is **held until the analyst gate returns**.
  The gate's brief has it run `pytest`, and `conftest.py:93` wipes `reference` whenever the DB is up —
  republishing first would let the gate silently re-create the split-brain on a graph that then *looks*
  repaired. Order is **`pytest` → parity repair → verify → U15**, never reversed.

- 2026-07-19 — **D15 locked; graph-dba dispatch PREPARED, deliberately not yet sent** (waiting on the
  gate per the sequencing constraint). Brief to cover: start FalkorDB (`./scripts/start_falkordb.sh -d`)
  → read-only inventory of `triage@v1` in `reference` + `ws:acme` (+ any `WorkflowRun`s, reported not
  gating per D15) → `DETACH DELETE` the def/snapshot + their `Step`s in both graphs → republish via the
  **reverted** `seed_workflows.sh` (must print `created`/`materialized`, **NOT** `already present —
  no-op`; a no-op means the delete missed ⇒ do not proceed to U15) → verify the live `intake` config
  contains **no** `understanding` instruction → drop the D13 leftovers `ws:live`/`ws:probe` (RAM on the
  16GB box). **After the repair, `pytest`/`test_queries.sh` must not run against this DB without a
  re-seed** — both wipe `reference`.

- 2026-07-19 — **✅ ANALYST IMPL GATE DELIVERED — `approve with suggestions`, 0 blocker / 2 major /
  3 minor / 3 nit** → `docs/reviews/m3-guard-thread-context-impl.md` (22.8KB, untracked; teco confirmed
  on disk). **The K-022 non-negotiable done-condition for the Defect-A/B commit `aa8b813` is SATISFIED**
  — no re-review cycle forced. All five mandatory confirmations affirmative:
  1. **`_drive_loop` byte-identity ESTABLISHED MECHANICALLY** (the item teco flagged as unverified):
     identical across `514346b`/`c3cc239`/`aa8b813` — SHA `71055f756280`, 2844 bytes, only its line
     offset moved 310→324. `_drive`/`_record` code-identical (docstring-only changes).
  2. **Defect-A S1–S4 complete** — `executor.py:607` passes `thread=result.thread`; DS
     understanding-primary precedence at `guards.py:136`; **exactly one `read_thread` per node**, so the
     m-C-neutral "zero extra reads" claim holds and is test-pinned.
  3. **R-1 addressed** — cue matching is now polarity-aware; `"no relevant"` dropped as unresolvable;
     both directions pinned in `test_guards.py`.
  4. **Defect-B catch provably narrow** — `HumanHandoffSignal` disjoint from `ServiceError` (MRO-verified);
     engine faults still reach the intact M-1 net.
  5. **Zero graph/DDL/QUERIES surface** in the diff.
  - **Two majors to close BEFORE U15** (teco spot-verified both in source — they hold):
    - **M-1 (doc drift → owner of the seed unit):** `m3-executor.md` §8 still documents the **D8/S5 intake
      + Defect-C answer** prompts that the **reverted** `seed_workflows.sh` no longer contains. QA would
      otherwise test against a spec matching **neither** the shipped script nor the live graph. Note this
      major is a **direct consequence of the D14 revert** — the revert was right, the spec just has to
      follow it.
    - **M-2 (code → coder):** `executor.py:503` `except ServiceError` also absorbs `UnknownActorError`
      and `ThreadNotFoundError` — **teco verified the hierarchy at `services.py:66/74`, both ARE
      `ServiceError` subclasses** — which are **not model-correctable**, and there is no log call. Net
      effect: a misconfigured agent id yields a run that reaches `done` having **posted nothing**, silently.
      The narrowness confirmation (#4) and this over-breadth finding are **both true**: the catch correctly
      excludes engine faults, but is still wider than the "bad argument the model can fix" rationale in
      its own comment (`executor.py:504-509`).
  - **Verification caveat carried forward (honest, not green-washed):** FalkorDB was **down**, so the gate
    observed **171 passed / 177 skipped / 1 deselected** (349 collected, consistent with 348+1). The
    affected files ran green (61 passed), but the **drive-level Defect-B pin
    `test_hallucinated_mention_does_not_fail_the_run` was SKIPPED** — the very test that proves a
    hallucinated mention no longer fails a run has **not been executed since the fix**. Must be covered
    by the U15 live pass or a DB-up rerun. The gate **correctly did not start FalkorDB** despite the
    stale "expect 348" brief (teco's correction never reached it — see the SendMessage slip above), so
    the feared unattended `reference` wipe **did not occur**.
  - **Revised order to U15:** close **M-2** (coder, code) + **M-1** (doc sync to the reverted script) →
    **then** the D15 parity repair (graph-dba) → verify → **U15**. Rationale: M-2's fix triggers a
    `pytest` run, which wipes `reference` — so the graph repair must come after it, per the standing
    sequencing constraint. The parity repair is therefore still **held**, now behind the fixes rather
    than behind the gate.

## Landing-2 cost datapoint (vs. Landing-1 ~1.20M tok / 238 tool uses / ~4h for U1–U10 + gate)
| Delegation | Owner | Units | ~Tokens | Tool uses | Wall time | Notes |
|---|---|---|---|---|---|---|
| Architect | architect | U11 design-patch | ~157k | 24 | ~8 min | Option B recommended (surfaced to user, approved); M-1/m-1/thread/trigger designed |
| D6 | tdd-engineer | U11+U12 | ~254k | 131 | ~43 min | pytest 283→312; M-1/m-3/n-2 closed; m-1 §7 amend; Option B; U12 REST |
| — | | remaining | — | — | — | analyst gate + U13 + U14 + U15 pending |

---

## Locked decisions — round 8

**D16 (2026-07-19) — tool-error split: propagate + log.** Closing the analyst gate's open
question 1. `UnknownActorError` / `ThreadNotFoundError` (and every future `ServiceError`
subclass) **propagate** to the M-1 fault net; only an explicit allowlist
(`UnknownMemberError`, `InvalidSearchQueryError`) is absorbed as a model re-prompt. Every
failed dispatch logs unconditionally (`_log.warning`), never tracer-gated. Rationale: these
are deployment misconfigurations, not model-correctable arguments — absorbing them produced a
run reaching `done` having posted nothing, which is precisely the AC-4 failure signature U15
exists to detect. Coordinator decision, matching the analyst's recommendation; written into
`m3-executor.md` §2.2.

## Gate majors closed (coder, aa1ba7ccf0004c81c)

- **M-2 (code) — closed.** `executor.py`: module constant `MODEL_CORRECTABLE_TOOL_ERRORS`
  (allowlist), unconditional `_log.warning`, `raise` for everything else; comment + docstring
  rewritten to state the split. Coder chose an **allowlist over a propagate-list** so the rule
  **fails closed** — a `ServiceError` subclass added later propagates by default instead of
  silently becoming a re-prompt. `ChannelNotFoundError` lands on the propagate side (its
  `channel_id` comes from deployment config, not model arguments) — correct under D16.
  Two new tests: `test_non_model_correctable_service_error_propagates_to_the_m1_net`
  (asserts escape on the *first* dispatch, so no `maxIterations` burn) and
  `test_a_model_correctable_tool_error_is_logged_even_without_a_tracer`.
- **M-1 (doc) — closed.** `m3-executor.md` §8 now describes what the working-tree
  `seed_workflows.sh` actually seeds: the S5 `understanding`-JSON instruction is **demoted to
  "Proposed, NOT seeded"** (rationale kept verbatim, D14 revert + 10/10→3/10 recorded, K-023
  re-landing gate named); Defect-C mitigations marked **RETAINED and seeded**, verified
  line-by-line; a **"Consequence for QA"** paragraph states that the shipped guard runs only on
  the degraded RECENT-TURNS tier, so a `guard_judgment` citing turn text is expected, not a
  defect. `scripts/seed_workflows.sh` untouched (settled by D14).

**Teco verification (not taken on the coder's word):** `git diff -U0` shows all five
`executor.py` hunks in imports / module constant / docstring / the except block — **none near
`_drive_loop`** (~line 324); lock intact. Independent `pytest -q` reproduces **173 passed, 177
skipped, 1 deselected** (= the gate's 171 + the 2 new tests). Doc edits confirmed present at
§2.2 (D16) and §8 (the demotion + QA consequence).

**Carried caveat (unchanged):** FalkorDB was down for both the gate and this unit, so the 177
graph-backed tests — including the Defect-B drive-level pin
`test_executor.py::test_hallucinated_mention_does_not_fail_the_run` — **still have not executed
since the Defect-A/B fix**. This must be covered by the DB-up run or U15.

**New finding (doc drift, for the rollup):** the `_drive_loop` lock is cited in the review and
plan as SHA `71055f756280` **+ 2844 bytes**. The SHA is real and reproducible; the byte count is
**wrong** — the extraction yielding that hash is **2860 bytes**. Anyone verifying the lock by
byte count would wrongly conclude it is broken. Fix the figure wherever the lock is quoted.

**Still open, in order:** D15 parity repair (graph-dba, authorized, now unblocked) → verify →
U15. Carried gate minors m-1 (negator window leaks across clause boundaries; residual failure
mode is *false-advance*, opposite of what the code comment claims), m-2 (`_recent_turns` slices
before filtering), m-3 (judge evidence tier invisible in the trace), nits n-1/n-2/n-3.

## Env-up verification run (teco, 2026-07-19)

FalkorDB started (`./scripts/start_falkordb.sh -d`, v4.18.11, ready in 1s). Full server suite:
**350 passed, 0 skipped, 1 deselected** — the first fully green graph-backed baseline since the
Defect-A/B fix. This **closes the carried caveat**: the drive-level pin
`tests/test_executor.py::test_hallucinated_mention_does_not_fail_the_run` **executed and passed**
(Defect B now proven at drive level, not just node level), as did both new D16 tests under a
real DB.

**Split-brain confirmed empirically (as predicted, not assumed).** The pytest run wiped
`reference` exactly as documented: `reference` is now **0 nodes** (no `WorkflowDef` at all),
while `ws:acme` retains `WorkflowDefSnapshot` x1 + `Step` x3 (plus Agent/Channel/Thread/User,
`Message` x15). This is the precise state the D15 repair must resolve — def gone, snapshot
stale and still the thing the executor drives.

Pre-existing graphs present: `ws:test`, `ws:probe`, `ws:live`, `cpg_falkorchat`,
`cpg_salesperson`. `ws:probe`/`ws:live` are the D15 cleanup targets.

## D15 parity repair — DONE (graph-dba adb3d1ab98d4d9063, 2026-07-19)

Executed and teco-verified. Before: `reference` 0 nodes; `ws:acme` held a stale
`WorkflowDefSnapshot` + 3 `Step`s whose `intake.config` **still carried the S5
`understanding`-JSON block** — i.e. the D14-reverted instruction was live in the graph the
executor drives. `WorkflowRun` count was 0, so nothing was broken by the delete.

Repair: `DETACH DELETE` of the snapshot + its Steps in `ws:acme` (4 nodes, 6 rels; Agent/
Channel/Thread/User/Message untouched), then `./scripts/seed_workflows.sh acme` → `reference
def … (created)` / `ws:acme snapshot … (materialized)` — **not** `already present — no-op`,
confirming the delete was complete. `ws:probe` and `ws:live` dropped.

**Teco-verified independently:** `GRAPH.LIST` = `cpg_falkorchat`, `cpg_salesperson`,
`reference`, `ws:acme`, `ws:test`. Both `reference` and `ws:acme`: `HAS_STEP` 3, `TRANSITION`
2, `intake.config CONTAINS 'understanding'` → **false**, `CONTAINS 'Never pass'` → **true**.
Split-brain resolved. Suite re-run green (350). Because that run re-wiped `reference`, teco
re-seeded last; final state verified with the def present.

**Numbering correction:** the "K-023 follow-ups" phrase used throughout this ledger is **wrong**
— K-023 was already taken (workflow↔chat linkage, delivered by U11). The follow-up bucket is
**K-027**, now filed in `docs/BACKLOG.md`. Read every earlier "K-023" in this doc as K-027.

**Also correct earlier in this ledger:** the `_drive_loop` byte figure appears as 2839 (line
~584) and 2844 (line ~917); both are wrong. The extraction that yields SHA `71055f756280` is
**2860 bytes**. Verify the lock by **SHA only** — the byte count has now been miscopied three
different ways.

**Status: U15 (qa-engineer acceptance) remains NOT RUN** — descoped per D12-B, tracked as K-025.
