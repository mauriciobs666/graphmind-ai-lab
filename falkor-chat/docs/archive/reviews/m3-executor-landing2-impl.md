# Review — M3 LLM-native executor, Landing 2 (U11+U12) implementation

> Reviewer: analyst · Date: 2026-07-15 · Type: static impl-review gate (K-022/K-023 done-condition)
> Baseline reviewed: commit **`514346b`** ("K-022 Landing 2 U11+U12 — @mention workflow trigger
> wiring") vs. its parent `f71eaea`. All 22 files read in full (source diff + new `trigger.py` +
> all changed tests + the three doc changes).
> Specs judged against: `docs/archive/plans/m3-executor-landing2.md` (the design-patch this diff implements,
> incl. §★ Option-B ordering rationale), `docs/archive/plans/m3-executor.md` (§6 trigger / §7 budget / §2.1
> loop), `docs/archive/plans/m3-executor-coordination.md` (Landing-2 / D6 claims), `docs/archive/reviews/m3-executor-impl.md`
> (the Landing-1 gate whose M-1/m-1/m-3/n-2 route here), `AGENTS.md`.
> Evidence: I ran the suite — **pytest 312 passed** (1 warning) against live FalkorDB; the query
> suite is untouched by design (no graph/DDL/QUERIES change — verified: no `QUERIES.md`/`DESIGN.md`/
> `bootstrap_schema.sh` in the diff). I traced every integration seam to its definition.

## Verdict: **approve with suggestions** (0 blockers)

The diff is a faithful, well-layered implementation of the `m3-executor-landing2.md` design-patch.
Every item the brief asked me to confirm holds (explicit confirmations below). The prior Landing-1
major **M-1 is closed**, and its minors **m-1 / m-3 / n-2** are closed as designed. Findings are two
doc-sync gaps and a handful of accepted-tradeoff / footgun follow-ups — none blocks the gate.

Finding counts: **0 blocker · 0 major · 3 minor · 3 nit.**

---

## Brief's required confirmations — explicit rulings

**1. Option B correctness + §2.1 A/B/C loop logic unchanged — CONFIRMED.**
`executor.py:342` adds exactly one call — `self._link_emissions(ctx, rec["stepRunId"], result.emissions)`
— immediately after `_trace_step` and **above** the `if firing is not None:` branch dispatch, precisely
as the design-patch prescribed (item 3, §★). The A/B/C branch *logic* (`executor.py:344-369`) is
identical to Landing 1: OUTCOME A budget-then-advance, OUTCOME B suspend, OUTCOME C re-loop/terminal.
Emissions ride the same deferred, stepRun-keyed lifecycle as `trace`: buffered during execution
(`StepResult.emissions`, `executor.py:75`; captured in `_buffer_emission`, `executor.py:487-495`) and
drained only after `_record` created the StepRun. `record_step_and_advance` (§12.2) is byte-for-byte
untouched; zero graph/DDL/QUERIES change (verified). The `PostMessageTool` inline link block is
correctly removed — it now returns `{"posted": msgId, "threadId": …}` with no `linked` key
(`tools.py:216-222`). Backed by a live test asserting **exactly one** `StepRun-[:PRODUCED]->Message`
edge keyed to the `answer` step (`test_executor_produced.py:75-99`).

**2. M-1 fault net — CONFIRMED (the Landing-1 major is closed).**
`_drive` is now a thin fault-net wrapper over `_drive_loop` (`executor.py:283-308`):
`HumanHandoffSignal` → `suspend_run` + return `"waiting"` (caught first, before the generic net, since
it is an `Exception` subclass); any other `Exception` → `_fail_with_note(...)` (`fail_run` with a
`"unexpected: {exc!r}"` ctx note, `AT_STEP` cleared per §12.5) then **re-raise** so
`_safe_run_workflow`'s isolation logs the stack. No path leaves a `running` zombie. Both funnel entry
points (`run`/`resume`) go through `_drive`, so the wrap is DRY. Backed by
`test_unexpected_exception_fails_the_run_and_reraises` (:272), `test_llm_guard_without_judge_fails_the_run_with_named_error`
(:287), `test_human_handoff_signal_suspends_the_run` (:307).

**3. PRODUCED, not EMITTED — CONFIRMED (D2 respected).**
`_link_emissions` (`executor.py:642-658`) calls `services.link_step_emission` → `repository.link_step_emission`
(QUERIES §12.6), which is the `PRODUCED` edge. `EMITTED` (K-013) is never touched here. The live test
asserts the `PRODUCED` relationship type explicitly (`test_executor_produced.py:90`). A `None` link
return is logged and non-fatal — a missing audit link never fails a run whose message already stands
(`test_link_gap_does_not_fail_run:101`).

**4. One handler per request (trigger XOR responder) — CONFIRMED.**
`api.build_router` schedules **exactly one** background handler: `if trigger is not None: add_task(_safe_run_workflow…) elif responder is not None: add_task(_safe_respond…)` (`api.py:155-161`). The trigger *holds*
the responder for its §6 step-4 fall-through (`trigger.py:80-84`), so `_safe_respond` is never also
scheduled — an `@mention` cannot fire both a workflow and a direct reply. `_build_default_app` passes
`trigger=` (not `responder=`) into `create_app` when `WORKFLOW_ENABLED` (`app.py:239`). Backed by
`test_trigger_wired_schedules_trigger_not_responder` — asserts `trigger.calls == 1` **and**
`responder.calls == []` (`test_api.py:366-381`).

Also confirmed: **m-3** named `guards.WorkflowConfigError` for an `llm` guard with no judge
(`guards.py:43-51, 87-92`; tests `test_guards.py:68`, `test_executor.py:287`); **n-2** null guard
treated as `""`/unconditional (`guards.py:75`, `if not guard:`; test `test_guards.py:56`); **m-1** §7
amended in `m3-executor.md` (suspend/intake loop human-paced, bounded by the DS 3-round ceiling, not
`maxSteps`; the OUTCOME-B path deliberately skips the budget check — `executor.py:351-359`); **thread
context** via `services.read_thread` (§4, thread-scoped, capped `THREAD_CONTEXT_WINDOW=20`,
role-mapped fold in `_assemble_messages`, `executor.py:496-533`; tests `test_executor_agent.py:205,232`);
**U12** three RO pass-throughs with `MAX_ID_LEN`-bounded path params + 404-on-missing-run
(`api.py:243-271`).

---

## Findings

### Minor

**m-A · n-1 doc-sync is still open — `node_note` absent from the QUERIES §12.10 / DESIGN §5 trace-kind
enumeration.** `grep -n node_note docs/QUERIES.md docs/DESIGN.md` → no match, yet the executor emits
`("node_note", …)` on iteration exhaustion (`executor.py:451`). The `m3-executor-landing2.md`
sequencing item 4 explicitly listed "add `node_note` to the QUERIES §12.10 / DESIGN §5 trace-kind
enumeration (review n-1)" as part of this landing, and it was not done. Trace kinds are opaque in-graph
(no schema/RAM impact), so this is documentation drift, not a defect — but a debug-run reader hitting a
`node_note` event has no doc reference for it.
*Suggested fix (owner: doc owner — teco/architect):* add `node_note` to the §12.10 / DESIGN §5 kind
list, or fold it into the plan's remaining doc-sync work before the milestone closes.

**m-B · No HISTORY.md / BACKLOG.md (K-023) entry for a committed delivered change.** `HISTORY.md`'s
newest entry is the 2026-07-12 Landing 1; there is no Landing-2 line, and `AGENTS.md` states "append an
entry for every delivered change." The commit is self-described as "parked pre-gate," so deferring the
HISTORY/BACKLOG entry until this review passes is defensible — but it must not slip: the change is on
`main`.
*Suggested fix (owner: teco / doc owner):* add the HISTORY entry (and the K-023 BACKLOG status bump) at
gate exit, in the same change that closes m-A.

**m-C · Every agent node — not just intake — issues an unbounded full `read_thread` then slices
app-side.** `_read_thread_context` (`executor.py:511-533`) is called from `_run_agent_node` for *all*
`type:'agent'` nodes (research and answer included), and `services.read_thread` / `repository.read_thread`
(QUERIES §4) has **no `LIMIT`** — it reads the entire thread (`NEXT*0..`) and the cap
(`msgs[-THREAD_CONTEXT_WINDOW:]`) is applied only after the full result set is materialized. For the
triage proof (short threads) this is fine and the plan explicitly accepted "app-side slice of the
returned list," but as a general seam it is an O(thread-length) read per agent-node step — a scale
sharp edge once threads grow or a def has many agent nodes.
*Suggested fix (owner: tdd-engineer / architect follow-up, non-blocking):* when this moves past the
proof, bound the read at the query (a since/last-N thread read) rather than reading-all-then-slicing;
track with the K-015 / scale work, not this gate.

### Nit

**n-A · `WORKFLOW_ENABLED` silently no-ops when `ENABLE_AGENT` is off.** The `WORKFLOW_ENABLED` branch
(`app.py:223`) sits *after* the `if not config.ENABLE_AGENT: return create_app(services)` early return
(`app.py:202-203`), so setting `FALKORCHAT_WORKFLOW_ENABLED=1` alone yields the plain M2-less app with
no trigger and no warning. The dependency is documented in `.env.example` ("requires
FALKORCHAT_ENABLE_AGENT=1"), but a misconfigured operator gets silence, not a hint.
*Suggested fix (owner: tdd-engineer, optional):* log a one-line warning when `WORKFLOW_ENABLED` is truthy
but `ENABLE_AGENT` is falsy.

**n-B · "§2.1 A/B/C loop byte-for-byte unchanged" is true for logic, not literally.** The commit message
and D6 claim say byte-for-byte; the diff also expands the OUTCOME-B comment (`executor.py:352-355`,
documenting the m-1 no-budget-check decision) in addition to the sanctioned `_link_emissions` line. The
loop *logic* is unchanged and the comment is appropriate — flagging only so the claim's wording isn't
mistaken for "not one character moved." No action needed.

**n-C · Two consecutive `user`-role turns in the assembled prompt.** `_assemble_messages`
(`executor.py:517-533`) appends the thread turns (which typically end on a human = `user` turn) and then
always appends the `CONTEXT:` block as another `user` turn. OpenAI-compatible endpoints (LM Studio)
tolerate consecutive same-role turns, and the plan designed it this way, so this is a prompt-shape
observation, not a defect.

---

## What's solid (verified)

- **Option B is exactly the ratified two-step emission, mirroring `_trace_step`** — buffered during
  execution, drained after the StepRun exists, keyed to the real `stepRunId`; the locked
  `record_step_and_advance` atomic write and the §2.1 loop logic are untouched; zero graph change, so
  the 241/241 query suite is legitimately unaffected.
- **The M-1 fault net is complete and correctly ordered** — `HumanHandoffSignal` (suspend) caught
  before the generic net (fail+re-raise); `_fail_with_note` shared with `_fail_budget` so every faulted
  run terminates identically (`failed`, `AT_STEP` cleared, readable cause). m-3's `WorkflowConfigError`
  and the M7 `NotImplementedError` both reach the net.
- **The trigger's §6 ordered rule is exact and single-flight-safe** (`trigger.py:56-86`): loop-guard →
  resume-if-waiting (returns whether the CAS wins or loses — a concurrent loser no-ops, and both human
  replies are already in the thread the winner re-reads, so no content is lost) → @mention-to-start →
  responder fall-through. All four branches have dedicated tests, incl. resume-priority-when-also-mentioned
  (`test_trigger.py:88`) and start-def-missing fall-through (`test_trigger.py:137`).
- **AC-2 prereq wired correctly** — `start_workflow_run` denorms `{"threadId": …}` into the run ctx
  (`services.py:596-598`), so `_read_thread_context` resolves the thread on every drive/resume; AC-2
  closes via thread re-read on resume (no ctx accumulation needed), exactly as designed.
- **Layering held** — no Cypher in `executor`/`guards`/`trigger`; the trigger orchestrates the service
  layer only and holds no executor reference (late-bound via `services.set_executor`); U12 endpoints are
  thin `services` pass-throughs. Path params bounded at the boundary (`MAX_ID_LEN`).
- **The default-off / network-free baseline is preserved** — `WORKFLOW_ENABLED` defaults false, all M3
  imports are lazy inside the branch, `_build_llm_judge` construction is offline; pytest 312 passed with
  no live model.
- **Tests are substantive** (+29 net): a live PRODUCED-edge assertion, a link-gap non-fatal path, the
  fault→failed / handoff→waiting / no-judge→named-error trio, thread-fold + skip-when-no-threadId +
  emission-capture, all four trigger branches, and the trigger-XOR-responder dispatch.

## Open questions (for the coordinator, not fixes)

- **m-C bound-at-query** — is a since/last-N thread read wanted before the live proof (U14), or is the
  read-all-then-slice acceptable through the proof and deferred to the K-015 scale work? A one-line call
  for the plan owner; I recommend deferring (proof threads are short).
- **m-A/m-B timing** — close the two doc-sync gaps now (small standalone doc change) or fold them into
  the U13–U15 landing? Either is fine; they must land before the milestone closes.

---

*Routing:* executor/trigger follow-ups (m-C, n-A) → tdd-engineer; doc-sync (m-A n-1 enumeration, m-B
HISTORY/BACKLOG) → teco / doc owner. n-B/n-C are observations, no owner needed.
