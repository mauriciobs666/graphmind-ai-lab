# Review — M3 LLM-native executor, Landing 1 (U1–U10) implementation

> Reviewer: analyst · Date: 2026-07-12 · Type: static code review (impl gate, K-022 done-condition)
> Baseline reviewed: the delivered Landing-1 diff on `main` (working tree) vs. `HEAD` (059b418) —
> `git diff` for tracked files + the three untracked new modules (`executor.py`, `guards.py`,
> `tools.py`) and their tests, read in full.
> Specs judged against: `docs/plans/m3-executor.md`, `docs/plans/m3-executor-ml.md`,
> `docs/plans/m3-executor-coordination.md` (units + D1–D5 + "Carried to Landing 2"), `AGENTS.md`.
> Scope boundary respected: **Landing 1 = Phases 0–3 only.** U11–U15 (trigger + proof + acceptance)
> are out of scope and are **not** faulted below.

## Verdict: **approve with suggestions** (no blockers)

The Landing-1 diff is a faithful, well-layered implementation of the approved plan. Contract adherence
(repository 1:1 with QUERIES §12), the §2.1 A/B/C loop, AC-6 defensive tool-scope enforcement, AC-5
tracing-by-construction, the D2 `PRODUCED`/`EMITTED` split, the M4 tail-anchored atomic advance, and the
M7 `expr` seam are all correct and evidence-backed. Both suites are green as claimed — I ran them:
**pytest 283 passed**, **`test_queries.sh` 241/241**. The two deliberately-deferred seams are correctly
routed to Landing 2 (ruling below). Findings are improvements and one operational-robustness gap, none
blocking.

Finding counts: **0 blocker · 1 major · 3 minor · 3 nit.**

---

## The two carried-forward deferrals — explicit ruling

**1. PRODUCED link ordering (from D5/U9) — ACCEPTABLE to defer to Landing 2.**
The `StepRun` is created inside `record_step_and_advance` *after* `_execute_step` runs, so at
`post_message` dispatch time no `stepRunId` is on `run` yet and the live `StepRun-[:PRODUCED]->Message`
link cannot fire in an integrated agent-node run. This is not a Landing-1 defect because:
- In Landing 1 the integrated executor never runs a live agent node — `_execute_step`
  (`executor.py:342`) only enters `_run_agent_node` when `self._llm is not None`, which is a Landing-2
  wiring; the offline engine tests drive the loop through the stub. So the missing-`stepRunId` path is
  never exercised by any Landing-1 run.
- The link is correctly implemented and tested at every layer that *does* run now:
  `repository.link_step_emission` (idempotent `MERGE`, QUERIES §12.6, live-asserted in
  `test_tools.py:246`), `services.link_step_emission`, and the tool's link-when-present /
  skip-with-`linked:false`-when-absent behavior (`test_tools.py:129,146`).
- The message is the durable artifact; a missing `PRODUCED` link is a diagnosable, retry-able gap, not
  a torn thread (§3/§9). The fix requires reordering the **locked** U8 loop (pre-mint the `StepRun`);
  the coder correctly did **not** mutate it and routed the decision to U11. Confirmed acceptable.

**2. Agent-node thread-message context (from D4/U8) — ACCEPTABLE to defer to Landing 2.**
`_assemble_messages` (`executor.py:432`) builds only the node `systemPrompt` + serialized run `ctx`;
recent thread messages are not folded in. No Landing-1 test needs them (stub LLM), and AC-2 — the
acceptance this unblocks — is U11–U15, out of scope. **Caveat for the coordinator:** this is a hard
prerequisite for AC-2 (intake must see the human's reply to judge "enough info"), so it must land *in*
U11, not slip further. Confirmed acceptable for Landing 1.

---

## Findings

### Major

**M-1 · An unexpected exception inside `_drive` leaves the run stuck in `running`, un-resumable.**
`executor.py:245-265` — `run()`/`resume()` call `_drive` with no `try/except`; the only status
transition on a fault is the step-budget `fail_run` (`executor.py:508`). Any other exception raised
mid-drive — `NotImplementedError` from an `expr`/unknown guard kind (`guards.py:80`), a `TypeError` if a
`{"kind":"llm"}` guard is reached with `guard_judge=None`, a repository/transport error, or a
`HumanHandoffSignal` raised by a granted `human_handoff` tool (`tools.py:339`, uncaught all the way up)
— propagates out and leaves the `WorkflowRun` at `status='running'` with `AT_STEP` still set. Because
`resume_run` CASes on `status='waiting'` (`repository.py`, §12.4), such a run can **never** be resumed
or cleaned up: it is a permanent zombie.
*Why it matters:* the state machine has no failed-transition for unexpected faults, so a single
malformed def (e.g. one `expr` guard) or a transient repo error orphans a run forever. In Landing 1 the
offline stub path is deterministic and the tests don't hit it, so it is not a green-suite blocker — but
it is a real operational-correctness gap the moment live defs/tools run.
*Suggested fix (owner: tdd-engineer, executor):* wrap the `_drive` body (or the `run`/`resume` entry
points) so an unexpected exception stamps `fail_run` with a diagnostic `ctx` note before re-raising,
mirroring the `_fail_budget` path; and decide the `HumanHandoffSignal` catch here vs. the U11 trigger.
Acceptable to close during the U11 background-handler wiring, but it should be tracked, not lost.

### Minor

**m-1 · Step budget is not enforced on OUTCOME B (suspend).** `executor.py:302-314` — the budget check
(`rec["stepCount"] > max_steps`) runs only on the continue paths (A at :304, C-reloop at :318), never
before `suspend_run`. Each intake suspend records a `StepRun` and bumps `stepCount`, so across many
suspend/resume rounds a run can accumulate `stepCount > maxSteps` and never fail. This is human-paced
(not an autonomous runaway) and the DS-note intake clarifying-round ceiling (3) — the *intended* intake
bound — is a Landing-2 concern, so the behavior is defensible. But it contradicts plan §7's wording that
`maxSteps` "counts every StepRun including intake re-runs" as a backstop.
*Suggested fix (owner: tdd-engineer / plan-clarify):* either check the budget on the suspend path too,
or amend §7 to state the intake loop is bounded by the (Landing-2) round ceiling, not `maxSteps`.

**m-2 · `StepRun.input` records the run's *initial* `ctx`, not the ctx at execution time.**
`executor.py:498-503` — `_record` passes `input=run["ctx"]`, the snapshot read once at drive start;
the evolving `run_ctx` dict is never re-serialized into a step's `input`. In Landing 1 `ctx` never
mutates, so every `StepRun.input` is identical and harmless — but once nodes write `ctx` (Landing 2)
the per-step audit `input` will be stale. Flag for the Landing-2 ctx-write work.

**m-3 · `evaluate_guard`'s `judge` parameter has no default and no None-guard on the call site.**
`guards.py:57-78` requires `judge`; `executor._select_transition` (`executor.py:476`) passes
`judge=self._guard_judge`, which is `None` when no judge is wired. A def whose transition carries a
`{"kind":"llm"}` guard, driven by an executor built without a `guard_judge`, calls `None(...)` →
`TypeError` → the M-1 zombie. Reachable only via misconfiguration, but cheap to fail loudly: raise a
clear "no judge wired for an llm guard" error rather than a bare `TypeError`. (Subsumed by M-1's
fault-handling, listed separately because the message should name the missing seam.)

### Nit

**n-1 · Undocumented trace `kind` value.** The agent loop emits `("node_note", …)` on iteration
exhaustion (`executor.py:405`), which is not in the `kind ∈ {…}` enumeration in QUERIES §12.10 / DESIGN
§5. Kinds are opaque in-graph (no schema impact), but the doc set should list `node_note` (and the code
uses `tool_call`/`tool_result`/`llm_prompt`/`llm_response` which *are* listed). Add it to the doc
enumeration or reuse `step_timing`.

**n-2 · Null-guard defensiveness.** `evaluate_guard`/`_select_transition` would raise
`NotImplementedError` on a `None` guard (`_load_obj(None)` → `{}` → `kind=None` → raise, `guards.py:80`).
Not reachable via the normal write path — `services._serialize_opaque(None)` → `""` (`services.py:109`)
— so today it is inert, but a hand-crafted or pre-existing transition with a null `guard` would crash a
drive (→ M-1). Treating `None` like `""` (unconditional) is a one-line safety net.

**n-3 · `graphrag_retrieve` channel scoping is workspace-wide.** `tools.py:251` defaults
`channel_id=None` (workspace-wide retrieval), consistent with the M2 responder's documented
carry-over — noting for continuity, not a defect (the thread→channel read isn't in QUERIES yet, K-015).

---

## What's solid (verified clean)

- **Repository ↔ QUERIES §12 is 1:1.** All 12 methods (`start_run`, `record_step_and_advance`,
  `suspend_run`, `resume_run`, `complete_run`, `fail_run`, `link_step_emission`, `get_run`,
  `read_step_runs`, `find_waiting_run_for_thread`, `append_trace_event`, `read_trace`) match the
  verified §12 Cypher verbatim; every write is a single `GRAPH.QUERY`; every query is parameterised;
  reads route via `ro_query`. No Cypher leaked into `executor.py`/`guards.py`/`tools.py`/`services.py`
  — layering (AGENTS.md) is respected, tools go through `services`, tenant seam via `ctx.ws`.
- **§2.1 A/B/C loop is correct and exhaustive** (`executor.py:289-324`). The D2-flagged
  "record-via-advance-to-self" for B/C is a clean reconciliation of §2.1's prose with §3's coupled
  `record_step_and_advance` query — relinking `AT_STEP` to self is a semantic no-op, preserving the
  per-execution audit trail and counting every `StepRun` against the budget. Budget correctly bounds
  the autonomous outcome-C self-loop (`test_step_budget_abort_fails_the_run`).
- **AC-6 is enforced defensively, not just by omission.** `_run_agent_node` offers only granted tool
  schemas *and* `_handle_tool_call` (`executor.py:410`) rejects an ungranted name and refuses a
  malformed (missing-required-arg) call — both a bounded re-prompt, never a dispatch. Asserted with a
  real "reject-then-never-dispatch" test (`test_executor_agent.py:134`) and a malformed-reprompt test
  (:154).
- **AC-5 holds by construction.** `_drive` selects `_NULL_TRACER` whenever `run["trace"]` is false
  regardless of the injected tracer (`executor.py:284`); `NullTracer.record` no-ops. Verified both ways
  against the live graph — debug run writes events, non-debug writes exactly zero
  (`test_executor.py:204,223`).
- **D2 `PRODUCED` vs `EMITTED` never conflated** — repository, services, QUERIES §12.6, and DESIGN
  §5.2/§6.2 all use the distinct `PRODUCED` type; DESIGN §215 explicitly notes both coexist. `MERGE`
  makes the link idempotent.
- **M4 tail pointer** (`record_step_and_advance`) is a single atomic query: `NEXT`-append from the
  `LAST_STEP_RUN` tail, move the tail, relink `AT_STEP`, bump `stepCount` — the `FOREACH(… IN CASE …)`
  idiom guards the optional prev-edge without collapsing the row; PROFILE-verified edge-anchored per
  QUERIES §12.2. **M7** `expr`/unknown guard is a pure `NotImplementedError` seam (`guards.py:80`),
  tested (`test_guards.py:127,135`), zero dead code.
- **U1 DESIGN reconciliation is complete** — §5.1/§5.2/§6.1/§6.2 corrected (stale StepRun→Message
  `EMITTED` → `PRODUCED`), §13 guard open-question marked resolved, `LAST_STEP_RUN`/`waitsForHuman`/
  `TraceEvent`/`TRACED` documented, §7.1 index table carries `TraceEvent` UNIQUE. `bootstrap_schema.sh`
  adds `TraceEvent.traceId` index **then** UNIQUE (index-before-constraint), idempotent, inert RAM
  until a debug run writes.
- **DS-method fidelity (U7/U9):** guards implement Q1 extract-then-judge (`_extract_understanding` →
  injected judge → `{decision,rationale}`) with bias-to-suspend on malformed/contradictory output
  (`guards.py:86-104`); `graphrag_retrieve` implements Q2 (τ cutoff on cosine distance, cap 5, abstain
  when nothing passes τ, all ctor-configurable) and avoids the responder's raw-k anti-pattern
  (`tools.py:277-293`). No method deviation spotted for routing to data-scientist.
- **Tests are substantive, not vacuous** (+87). They assert real graph state (audit-trail step order,
  `atStepKey`, `waitingThreadId`, trace-event kinds) against a live `ws:test`, and the agent-loop tests
  assert dispatch effects (`reg.dispatched == []` for ungranted, re-prompt content). Suspend/resume
  single-flight, budget abort, priority ordering, and the empty-vs-llm guard dispatch all have
  dedicated tests.

## Open questions (for the coordinator, not fixes)

- **M-1 timing:** close the drive fault-handling during U11 background-handler wiring, or as a small
  standalone patch now? It touches the reviewed loop only at the entry points, so it can land without
  reopening U8.
- **m-1:** is the intake loop intended to be bounded by `maxSteps` (needs the suspend-path budget
  check) or solely by the DS clarifying-round ceiling (a §7 wording amendment)? A one-line decision
  that belongs to the plan owner before U11.
