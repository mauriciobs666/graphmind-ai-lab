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

## Landing-2 cost datapoint (vs. Landing-1 ~1.20M tok / 238 tool uses / ~4h for U1–U10 + gate)
| Delegation | Owner | Units | ~Tokens | Tool uses | Wall time | Notes |
|---|---|---|---|---|---|---|
| Architect | architect | U11 design-patch | ~157k | 24 | ~8 min | Option B recommended (surfaced to user, approved); M-1/m-1/thread/trigger designed |
| D6 | tdd-engineer | U11+U12 | ~254k | 131 | ~43 min | pytest 283→312; M-1/m-3/n-2 closed; m-1 §7 amend; Option B; U12 REST |
| — | | remaining | — | — | — | analyst gate + U13 + U14 + U15 pending |
