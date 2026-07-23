# M3 — `kind:'process'` proof flow (K-024, remaining half) — implementation plan

> **Status:** **approved for implementation**, patch **v2.1 (2026-07-20)**. v2 answered the analyst
> plan gate (*request changes*: 2 blocker · 6 major · 10 minor · 5 nit); the **re-gate (v2) verdict
> was *approve with suggestions*** (U0 + U1 dispatched), and v2.1 closes its four new findings
> (M-7 · m-11 · m-12 · n-6). Both finding sets are answered in the **Gate response** section at the
> end of this file.
> Planning-only artifact — no code/DDL/doc changed by this plan itself.
> **Closes:** the open half of **K-024** (`docs/BACKLOG.md`) — the LLM-free business-process proof
> over `human` / `decision` / `wait` steps, i.e. the **DESIGN §6.3** "coordination is workflow, not a
> separate primitive" proof. Unblocks **K-025** (qa-engineer acceptance → M3 ✅).
> **Builds on:** `docs/archive/plans/m3-executor.md` (§2.1 loop, §2.3 deterministic-node seam, §2.4
> suspend/resume, §2.5 guard dispatch), delivered as K-022 Landings 1+2.
> **Coordination ledger:** `docs/archive/plans/m3-process-flow-coordination.md` (teco).
> **Explicitly NOT in scope:** **K-027** (live-triage reliability, carried minors m-1…m-3, nits
> n-1…n-3). Nothing from K-027 may be folded into a unit here.
> **Baselines to preserve and raise:** server pytest **350 passed / 0 skipped / 1 deselected**
> (network-free); `./scripts/test_queries.sh` **241/241**. See §7 for the honest, per-unit form of
> "preserve" — this patch **does** modify named existing fixtures (B-1/B-2).

---

## 0. Decisions

D-A…D-E were stakeholder calls, all now settled (user, 2026-07-19). **D-F, D-G, D-H are the three
coordinator open questions the gate raised (OQ-A/OQ-B/OQ-C), settled 2026-07-20** and written here
as decisions — they are **not** open and are not to be re-litigated. D-F refines the *mechanic*
inside D-B's chosen option; it does not reopen D-B.

### D-A · Deterministic guard language — ✅ settled: structured `cmp` comparator

A `decision` node branches on data. `guards.evaluate_guard` today resolves only `""` (unconditional)
and `{"kind":"llm"}`; every other `kind` raises `NotImplementedError` (the "M7 seam",
`guards.py:144`). `TRANSITION.guard` is an opaque set-on-create string — **no schema change is
needed for any option below**.

| # | Option | What a guard looks like | Pros | Cons |
|---|---|---|---|---|
| **A0** | **No guard language — match on `on`** | transition fires when `TRANSITION.on == StepResult.on` | zero new vocabulary; uses the two fields that exist and are *currently dead* (see Finding F-1) | pushes all branch logic into the step handler, which then needs its own config-driven comparison — the problem reappears one layer down; adds a second branching mechanism next to guards |
| **A1** ⭐ | **Structured comparator, data-only** | `{"kind":"cmp","path":"ctx.decision","op":"eq","value":"approve"}` + `{"kind":"all"\|"any"\|"not","of":[…]}` | **no parser, no `eval`, no dependency** — it is already-parsed JSON; every op is a whitelisted Python callable; trivially unit-testable; depth/width caps make DoS structural; readable in the graph | verbose to author by hand; not expressive. **The concrete expressiveness cliff (n-4): `cmp` compares a *path to a literal* only — there is no path-vs-path form**, so `ctx.approver != ctx.requester` is not expressible and never will be without a new `kind`. No arithmetic, no string ops beyond `contains` |
| **A2** | **Tiny whitelisted DSL string** | `{"kind":"expr","text":"ctx.decision == 'approve'"}` | pleasant to author | needs a hand-written tokenizer+parser (~200 lines to get right), which is a new attack/bug surface; every future op is a grammar change; reopens DESIGN §13 as "we built a language after all" |
| **A3** | **Expression library** (`simpleeval`, `json-logic`) | library-defined | expressive, off-the-shelf | a new runtime dependency on the guarded execution path; sandbox-escape history in this class of library; contradicts DESIGN §13's *resolved* "no expression library is built" |

**Settled: A1**, with the kind named **`cmp`** (not `expr`). Naming it `cmp` keeps `kind:'expr'`
reserved and still raising, so DESIGN §13's resolution ("no expression library") stays literally
true and the door to A2/A3 stays visibly shut. **When a guard is validated is part of this decision
(gate M-4): a `cmp`-family guard is validated *structurally at publish time*, not only at drive
time** — see §3.2 and §3.3.
**What it locks in:** the shape of deterministic branching for the life of the engine. Adding ops
later is additive (a dict entry); switching to A2/A3 later would be a def-content migration, and
published defs are immutable ⇒ a `key`/`version` bump for every def. That cost is real but small
today (one def).

### D-B · Human-input channel for a `process` flow — ✅ settled: REST

A parked `human` step must be advanceable by someone who is not in a chat thread. The existing
resume path (`trigger.py` step 2) is chat-message-driven.

| # | Option | Mechanics | Pros | Cons |
|---|---|---|---|---|
| **B1** ⭐ | **REST `POST /workflow-runs/{runId}/input`** (+ `POST /workflow-runs` to start) | body merges into the run `ctx`, and **the merged ctx is written by the resume CAS itself** (D-F) | reuses `waitsForHuman` + `suspend_run`/`resume_run` unchanged; input is **durable and auditable** in `ctx`; drives synchronously ⇒ deterministic offline tests; a UI/CLI/curl can drive it | needs **two new repository queries** ⇒ a graph-dba gate (see §5); a stale *read* before the merge is still possible (R-1) |
| **B2** | **Reuse the chat resume path** | the approver posts a message in a thread bound to the run | zero new API, zero new Cypher | forces a `Thread` + `Message` to exist for a *non-chat* process — it hollows out the very §6.3 claim this slice is meant to prove; input arrives as prose, so a deterministic guard has nothing structured to read (it would need an LLM to parse ⇒ violates "offline") |
| **B3** | **MCP tool only** | `submit_workflow_input` MCP tool | fits the agent-facing surface | MCP is the *agent* front door; a human approval belongs on the human front door. Also still needs the same service+repository work |

**Settled: B1.** B3 is a cheap follow-up once B1's service method exists (`mcp.py` is a thin
adapter — one tool, ~15 lines) and is listed as a non-blocking follow-up (OQ-2), not part of this
slice. Note B1 also covers **starting** a run without a chat message: `repository.start_run` today
*requires* a trigger `Message` to match (`repository.py:1067`, QUERIES §12.1) — see Finding F-2.

### D-C · `wait` step semantics — ✅ settled: external signal, not a timer

| # | Option | What it means | Verdict |
|---|---|---|---|
| **C1** ⭐ | **Wait for an external signal** | the step parks; an external actor posts the signal through the same D-B input endpoint; a deterministic guard on the signalled ctx key advances it | Zero new machinery — reuses the exact suspend/resume mechanic. Honest about what the system can do |
| **C2** | **Timer / deadline that fires by itself** | the run wakes at `wakeAt` | **This system has no scheduler.** `BackgroundTasks` are request-scoped; there is no periodic worker, no leader election, no due-run sweep. C2 needs a new long-lived component, a `WorkflowRun.wakeAt` **index** (new RAM line, rule 6), crash/duplicate-fire semantics, and ops. That is its own milestone item, not a corner of a proof flow |
| **C3** | **Deadline-as-data, checked on the next drive** | the step stamps a deadline; a guard compares it against `now` | Cheap, but it is a *lie by omission*: nothing fires, so the deadline only takes effect when something else pokes the run. It also makes guard evaluation clock-dependent, which costs the "fully deterministic tests" property |

**Settled: C1**, and **say plainly in DESIGN §6.1 that `wait` is signal-driven, not timer-driven,
because there is no scheduler** — with C2 filed as a new backlog item (proposed **K-028 — workflow
timers / scheduled wakeups**) rather than silently implied by the step name.
**Consequence to state plainly (gate m-7): under C1, `wait` is *mechanically identical* to `human`
to the engine** — same park (OUTCOME B on `waitsForHuman`), same publish invariant, same input
endpoint, same guard mechanism. The **only** difference is the `awaiting.kind` string in the step
output envelope (`"signal"` vs `"human"`), which exists so a client can render the right prompt.
This must be said in the DESIGN §6.1 edit and repeated in the K-025 handoff so QA does not read
`wait` as a distinct unimplemented mechanism.

### D-D · The proof def — ✅ settled: `access-request@v1` (renamed, recounted)

**`access-request@v1`**, kind `process` — a new-hire access-request flow. **Six steps, six
transitions**, exercising `human` ×2, `decision` ×3, `wait` ×1, plus **four** `cmp` ops
(`exists`, `in`, `eq`, `truthy`), a conditional-beats-unconditional ordering case, and a two-outcome
branch. Full spec in §4. Alternatives considered: a pure two-step approve/reject (too thin to prove
branching or `wait`), a purchase-order flow (identical shape, no extra proof value).

- **Recount (gate m-1):** the earlier summary said "seven transitions … three ops". §4 was and is
  correct — **six** transitions (#1–#6) and **four** ops. This summary is now the corrected one.
- **Key rename (gate m-2):** the def key is **`access-request`**, *not* `onboarding`.
  `tests/test_api.py:413`, `tests/test_services.py:731` and `test_repository.py` have published a
  fixture def keyed `onboarding` (version `"1"`) into `reference` for a long time. It would not
  collide on `(key, version)` and `wf_repo` wipes `reference` per test, so it is not a live bug —
  but `get_def(key, version=None)` sorts `version DESC` across both, and every grep for "the
  onboarding def" would return two unrelated things. The rename is applied **everywhere**: §4, §6,
  §7, `proof_defs.ACCESS_REQUEST_DEF`, `tests/test_process_flow.py`, and the seed-script section.

**Def identity: `access-request` / `v1` — brand new, additive. `triage@v1` is not touched.**

### D-E · Scope boundary vs. the full typed-step library — ✅ settled: E1

`m3-executor.md` §2.3 lists `prompt`, `tool`, `message` alongside `decision`/`human`/`wait`.

| # | Option | Verdict |
|---|---|---|
| **E1** ⭐ | Implement `human`/`decision`/`wait`; `prompt`/`tool`/`message` become a **documented raising seam** (`NotImplementedError`, reaching the M-1 fault net ⇒ `fail_run` with a named cause) | Closes gap 1 honestly, keeps the slice tight. Each deferred type has a genuine open design question (`prompt` needs an LLM ⇒ conflicts with "offline"; `tool` needs a permission fence outside an agent loop; `message` needs author/thread resolution for a run with no thread) |
| **E2** | Also implement `message` | Tempting (a "notify" terminal reads better than a `decision` terminal) but it needs a thread, which a non-chat process run does not have, and an author identity decision. Half a slice of design for cosmetics |
| **E3** | Implement all six | Speculative code with no consumer; the exact thing §2.3 warned against |

**Settled: E1.** Note the deliberate consequence: replacing today's silent no-op with a raise is a
**behaviour change** — see Finding F-3 (amended) and Risk R-3. **The blast radius on the existing
test estate is larger than the first draft measured** — B-1: a `type:'task'` fixture step is driven
by the real loop today. See §3.3 and U2.

### D-F · ctx write mechanic — ✅ settled (coordinator, 2026-07-20; gate OQ-A / M-1): **fold into the resume CAS**

**Decision.** There is **no `set_run_ctx`** in this slice. The ctx write rides inside the existing
resume CAS as a single new repository method:

```python
repo.resume_run_with_ctx(ws, run_id=..., ctx=...)   # QUERIES.md §12.13
```

```cypher
MATCH (r:WorkflowRun {runId: $runId})
WHERE r.status = 'waiting'
SET r.status = 'running', r.waitingThreadId = '', r.ctx = $ctx
RETURN r.runId AS runId, r.status AS status
```

i.e. `resume_run` (§12.4) **plus one `SET` term**. Zero rows ⇒ the run was not `waiting` ⇒ **409**,
and **nothing is written** — neither the flip nor the ctx.

**Why (the gate's argument, adopted verbatim).** With a split write, submitter B's `ctx` can land
between A's ctx write and A's CAS, so the drive that runs is A's while the data it reads is B's — a
**silent wrong branch**, not merely a lost input. Worse, a stale-read submitter could *erase* a key
an earlier step already branched on, leaving a run whose own `ctx` no longer explains its own trail.
Folded, only the CAS winner's ctx is ever written, so "which input advanced the run" and "which
input is in `ctx`" can never disagree — which is exactly the audit property §6.3 exists to prove.

**Consequences, all folded into this patch:**
- **§5 / U0's deliverable changes — it does not grow.** §12.13 `resume_run_with_ctx` *replaces* the
  proposed `set_run_ctx`. Still two new queries, same gate cost.
- **`executor.resume` grows one optional parameter**: `resume(ctx, *, run_id, run_ctx_json: str |
  None = None)` — dispatching to `repo.resume_run_with_ctx` when supplied, `repo.resume_run`
  otherwise (the existing trigger path is byte-identical in behaviour). `executor.resume` is at
  `executor.py:282`, i.e. **outside the SHA-locked `_drive_loop` region** (see §3.1). No behaviour
  change downstream: `resume` already re-reads the run with `get_run` *after* the CAS
  (`executor.py:295`), so the just-written ctx is what `_drive_loop` loads.
- **R-1 is rewritten** (§3.5, §8) to describe only the residual stale-merge-read window.
- Scope note: this is a mechanic refinement **inside** settled decision D-B, not a channel change.

### D-G · Error contract for a synchronous drive — ✅ settled (coordinator, 2026-07-20; gate OQ-B / M-3): **the endpoints catch the drive fault**

**Decision.** A fault *during the drive* (unimplemented step type, malformed `cmp` guard reaching
evaluation, step-budget exhaustion) is **caught by the endpoint's service method** and returned as a
success-shaped envelope describing a **correctly terminal run**:

```jsonc
POST /workflow-runs/{id}/input  →  200 {"runId":…, "status":"failed", "error":"NotImplementedError: step type 'tool' …"}
POST /workflow-runs             →  201 {"runId":…, "status":"failed", "error": …}
```

**Why.** The M-1 fault net (`executor.py:329`) has already `fail_run`-stamped the run before it
re-raises: the run *is* terminal and correct in the graph. A 500 traceback would misreport a
correctly-recorded terminal run as a server bug and would break exactly the audit property §6.3 is
meant to prove. The start route keeps **201** (the run resource *was* created; only its outcome is
`failed`) — the envelope is identical to the input route's.

**What the catch covers, precisely:** `NotImplementedError` and `guards.WorkflowConfigError`
**raised out of the drive call only**. Faults raised *before* anything is written (unknown def
snapshot, unknown run, rejected input) are **not** caught here — they keep their status codes below.

- **Budget exhaustion is NOT in the catch list (re-gate m-11).** It **never raises**: `_fail_budget`
  (`executor.py:663`) stamps `fail_run` and *returns* `"failed"` through OUTCOME A/C (`:370`,
  `:387`). There is no `StepBudgetExceededError` in the tree, and **U3 must not invent one** — that
  would convert a clean terminal return into a raise, i.e. an engine behaviour change inside a unit
  that must not make one. Budget exhaustion reaches the same envelope through the normal return path.
- **Re-read, and re-raise if it is not terminal (re-gate m-12).** After catching, the service
  re-reads the run via `get_run` and reports **the graph's status, never a guessed one**. If that
  status is **not** in `{failed, done, waiting}` — i.e. a fault escaped before `_drive`'s net stamped
  `fail_run`, or `fail_run` itself failed — the service **re-raises**, and a 500 is the correct
  answer: reporting a zombie `running` run as a 200/201 success envelope would be the worst possible
  outcome for the audit property §6.3 exists to prove. §7 U3.11 asserts the envelope's status came
  from `get_run`, and adds the zombie case.

**U3 additionally registers these handlers in `app._register_error_handlers`** (today it maps only
`WorkflowDefSpecError` → 400 and `WorkflowDefNotFoundError` → 404, `app.py:71–86`):

| Exception | Code | Rationale |
|---|---|---|
| `WorkflowRunNotFoundError` (`repository.py:64`, **already exists, unhandled**) | **404** | §3.4 step 1 promises a 404 that does not exist today |
| `WorkflowRunNotWaitingError` (**new**, `repository.py`) | **409** | input submitted to a non-parked run |
| `WorkflowInputRejectedError` (**new**, `repository.py`) | **400** | reserved key / undeclared key / disallowed value (D-H); nothing written |
| `guards.WorkflowConfigError` | **400** | after M-4 its dominant source is **publish-time** structural validation of a malformed guard — an authoring defect, same class as `WorkflowDefSpecError`. Drive-time occurrences never reach the handler: D-G's catch converts them to the failed envelope first |
| `WorkflowEngineDisabledError` (**new**, `services.py`, subclass of `RuntimeError`) | **503** | folds in the plan's own **OQ-1**: `_require_executor` (`services.py:570`) raises today; making it a *named `RuntimeError` subclass* maps it to 503 without a blanket `RuntimeError` handler that would mask real bugs, and keeps any existing `pytest.raises(RuntimeError)` green |

Envelope shape is the existing one: `{"error": type(exc).__name__, "detail": str(exc)}`.
**§7 U3 carries one test case per handler.**

### D-H · Input validation & the step budget — ✅ settled (coordinator, 2026-07-20; gate OQ-C / M-5): **validate before merge (option b) + an explicit def budget (option c)**

**The problem the gate found.** Every park is an advance-to-self `_record` that bumps `stepCount`
(`executor.py:363`), `maxSteps` defaults to `DEFAULT_STEP_BUDGET = 12` (`executor.py:58`), and §4.3's
happy path already costs 8. Four mistyped approvals therefore kill the run permanently — ordinary
human error, not a long process.

**Decision, part (b) — validate the submitted input against the parked step's declared expectations
*before* merging into ctx and resuming.** A value that can fire no guard becomes a **free 400** that
costs **no** step budget and writes nothing.

- **Where it lives: `services.submit_workflow_input` — the service layer, not only `schemas.py`.**
  Pydantic sees only HTTP callers; MCP (OQ-2) and direct service callers bypass the schema entirely,
  so the schema is a convenience bound, the service is the contract.
- **How the parked step is found: no new query.** `repo.get_run` already returns `atStepKey`,
  `defKey`, `defVersion` (§12.7); `repo.get_snapshot(ws, key, version)` (§11.5, already used by
  `_drive_loop` at `executor.py:334`) returns the steps with their `config`. Two existing RO reads.
- **What a step declares (the validation contract):**
  | step type | declares | accepted input keys | optional value check |
  |---|---|---|---|
  | `human` | `config.fields: [str]` | exactly those field names | `config.expects: {field: [allowed values]}` |
  | `wait` | `config.signal: str` | exactly `{signal}` | `config.expects` likewise |
- **Rules (each ⇒ `WorkflowInputRejectedError` → 400, nothing written, no step consumed):**
  1. a reserved key (`threadId`, `error`) is present — see M-2 in §3.4;
  2. an input key is not in the parked step's accepted set;
  3. `config.expects[field]` declares an allowed-value list and the submitted value is not in it;
  4. the merged ctx would exceed `MAX_CONFIG_LEN` once serialized (m-5: this bound is the
     *service's*, because pydantic cannot see the stored ctx it merges into).
- **Permissive fallback, deliberately:** if the parked step declares **neither** `fields` nor
  `signal`, rule 2 is **skipped** (any non-reserved key is accepted). This is what makes the
  invariant non-retroactive — no existing def or fixture can start failing because it never declared
  a field list.

**Decision, part (c) — state the budget explicitly.** `proof_defs.py` exports
`ACCESS_REQUEST_MAX_STEPS = 24` next to `ACCESS_REQUEST_DEF`, the seed/acceptance paths pass it as
the start body's `maxSteps`, and the def module docstring documents the arithmetic:

- happy path = **8** steps (§4.3) ⇒ **16 spare**;
- with rule 2/3 in force, an invalid `decision` value is a **free 400** ⇒ **mistakes of that class
  are unbounded and cost nothing**;
- the only remaining budget consumer is a *valid, declared* value that still fires no guard (e.g.
  `{"provisioned": false}` on the `wait` step, which `expects` deliberately does **not** constrain
  so that "not yet" is expressible). Budget 24 allows **16** such re-parks before `_fail_budget`.
- A caller that omits `maxSteps` gets the global default 12 ⇒ **4** re-parks. Documented, not
  silent.

*Not chosen:* raising `DEFAULT_STEP_BUDGET` globally (it would silently change `triage@v1`'s
runaway guard) or a per-def budget property (a def-schema change, out of scope; noted for K-028's
neighbourhood).

---

## 1. Goal & scope

**Goal.** Deliver an LLM-free, deterministic, offline `kind:'process'` workflow that publishes,
materializes, starts, parks on human input, branches on data, parks on an external signal, and runs
to a terminal `done` — proving DESIGN §6.3 with a real execution trace in the graph.

**In scope**
- A deterministic transition-guard kind (`cmp`) in `guards.py`, **plus a publish-time structural
  validator for it** (D-A, gate M-4).
- Real `human` / `decision` / `wait` step handlers in `executor._execute_step`; an explicit raising
  seam for `prompt` / `tool` / `message` (D-E); the `_select_transition`/`_trace_step` trace edit
  that makes a `cmp` verdict traceable (gate M-6).
- A run-start path that does **not** require a chat trigger message, and a human/external input
  path that validates (D-H), merges into the run `ctx` and resumes the run **in one CAS** (D-F).
- The `access-request@v1` proof def, seeded additively; an offline end-to-end acceptance test (D-D).
- The named existing-fixture edits the two new invariants require (B-1, B-2) — an explicit,
  budgeted part of U2, not incidental churn.
- The doc updates each of those invalidates (assigned per unit, §6).

**Out of scope (do NOT build here)**
- K-027 in its entirety (live-triage reliability, judge calibration, terminal-node contract).
- Any change to `triage@v1`'s content, to the LLM guard judge, or to the agent-node loop.
- A scheduler / timer wakeups (D-C ⇒ proposed K-028).
- `prompt` / `tool` / `message` handlers (D-E).
- A web UI for approvals. The REST endpoints are the surface; `web/` is untouched.
- Any change to `_drive_loop` (see §3.1 — the design is built specifically to avoid it).
- A `ctxVersion` compare-and-set for multi-approver races (R-1).

---

## 2. Context & findings (verified against the tree at `4f69a16`; re-verified at patch time)

### Verified state

- `executor._execute_step` (`server/falkorchat/executor.py:396`) dispatches `type == 'agent'` **with
  a wired LLM** to `_run_agent_node`; **everything else returns `StepResult(output="", on="done")`**.
- `guards.evaluate_guard` (`server/falkorchat/guards.py:97`): `""`/`None` → unconditional true;
  `{"kind":"llm"}` → injected judge; **anything else → `NotImplementedError`** (line 144).
- `_select_transition` (`executor.py:613`) sorts `(guard == "", order)` — conditional guards are
  evaluated before the unconditional default; first firing wins. It appends to `judgments` **only**
  when `parsed.get("kind") == "llm"` (`executor.py:636`), and `_trace_step` formats the payload as
  `f"{text} -> {verdict.decision}: {verdict.rationale}"` (`executor.py:706`) — see M-6/§3.2.
- `_drive_loop` (`executor.py:333–392`) OUTCOME B keys **only** on `config.get("waitsForHuman")`.
  It re-reads nothing mid-loop: `run_ctx` is loaded once at entry from `run["ctx"]`, and
  `executor.resume` re-reads the run via `repo.get_run` after the CAS ⇒ **a ctx written to the graph
  by/before the resume CAS is visible to the resumed drive.** This is the hinge the whole D-B/D-F
  design hangs on. *(The analyst independently traced every §4 def shape through the real loop and
  confirmed nothing spins, parks wrongly, or falls through to `done` unintentionally.)*
- `repository.suspend_run` / `resume_run` (`repository.py:1151`/`1169`) are guarded single-query CAS
  flips; `suspend_run` denorms `waitingThreadId`. `resume_run`'s `SET` is where D-F adds `r.ctx`.
- `services.start_workflow_run` (`services.py:578`) mints the run id + clock, resolves the trigger
  message's thread into `ctx = {"threadId": …}`, and passes `executor.step_budget` as `maxSteps`.
- `services._validate_def_spec` (`services.py:428`) already sees each step's `config` dict **and each
  transition's `guard`** (`services.py:504`) at publish time — publish-time invariants on both cost
  nothing structurally.
- `STEP_TYPES` (`services.py:44`) already whitelists all seven types; `WORKFLOW_KINDS` already
  includes `process`. **No whitelist change is needed.** Note `task` is **not** in `STEP_TYPES` —
  see F-3/B-1.
- `repo.get_snapshot` (`repository.py:1403`) returns `{name, kind, start_key, steps, transitions}`
  from `ws:{id}` — the read D-H's validator reuses.
- Test conventions: `server/tests/test_executor.py` drives the real engine against the live `ws:test`
  graph via the `wf_repo` fixture (`server/tests/conftest.py:85`, which wipes `reference` too), with
  stub judges and `id_gen`/`clock` injection. That is the pattern every new test here follows.

### Findings to carry into implementation

- **F-1 · `TRANSITION.on` and `StepResult.on` are vestigial.** Nothing reads either. The only
  consumers are `services.publish_workflow_def` writing `on` through (`services.py:504`) and the
  three `StepResult(..., on="done")` constructions (`executor.py:412,466,484`). DESIGN §6.1 describes
  `on` as "the event/outcome that fires it", which is **not true**. **And the same sentence's second
  half is also wrong (gate m-6):** it says "guards are evaluated in `TRANSITION.order`, first-firing
  wins", but the sort key is `(t["guard"] == "", t["order"])` (`executor.py:627`) — *conditional
  guards first*, `order` only as a tie-break **within** each class. U2's DESIGN §6.1 edit fixes
  **both halves**.
- **F-2 · A run cannot be started without a chat `Message`.** `repository.start_run` has
  `MATCH (trigger:Message {msgId: $triggerMsgId})` as a hard anchor; zero rows ⇒ `start_workflow_run`
  raises. A business process started from a form/API has no message. Hence the second start query in
  U0.
- **F-3 · (AMENDED, gate B-1) Two distinct silent-stub cases are load-bearing for the 350 baseline —
  the first draft named only one.**
  1. **`type:'agent'` with `llm=None`.** `test_executor.py` builds `agent` steps and constructs the
     executor with `llm=None`, relying on the fall-through stub to drive the loop offline. **This
     path must be preserved verbatim** and its docstring must say *why* it exists.
  2. **`type:'task'` — a non-`agent`, non-whitelisted type executed by the real loop.**
     `test_executor.py:345` declares `{"key":"end","type":"task","config":"{}"}` (written straight
     through `repo.materialize_snapshot`, bypassing publish validation) and
     `test_hallucinated_mention_does_not_fail_the_run` (`:352`) drives it, asserting `status ==
     "done"` and `trail == ["answer","end"]`. Under D-E's "unknown type ⇒ `NotImplementedError`",
     `_drive` `fail_run`s **and re-raises**, so the test sees an exception instead of `"done"` —
     **killing a Defect-B regression pin.**
     **Resolution (U2 scope, explicit):** change that fixture's `"type": "task"` → `"type": "agent"`.
     It is already driven with `llm=None`, so preserved stub case 1 covers it and the fixture's
     stated intent (`# terminal, non-agent → stub`, i.e. *a step the node loop does not run*) is
     preserved exactly. Update the comment to `# terminal, agent-without-LLM → stub`.
- **F-4 · Park/resume cycles consume the step budget.** Every park records a `StepRun`
  (advance-to-self) and bumps `stepCount`; `maxSteps` defaults to `executor.step_budget` = 12, and
  there is **no per-def budget** — `start_workflow_run` always passes the executor's global default.
  §4.3 costs 8 steps. **D-H settles the consequence**: invalid input is rejected for free (no step),
  and `access-request@v1` documents and passes `maxSteps = 24`.
- **F-5 · `find_waiting_run_for_thread` with an empty `threadId` would match parked process runs.**
  Process runs park with `waitingThreadId = ''` (there is no thread). No caller passes `''` today
  (`trigger.py` always has a real thread from a posted message), so this is latent, not live. U3 adds
  a one-line defensive short-circuit in the service (falsy `thread_id` → `None`) rather than changing
  Cypher.
- **F-6 · (NEW, gate M-2) The *reverse* of F-5 is live and reachable.** `_drive_loop`'s suspend
  passes `run_ctx.get("threadId","")` straight into `suspend_run(thread_id=…)`
  (`executor.py:379–381`), which denorms `waitingThreadId` (`repository.py:1151`); `trigger.py`
  step 2 (`trigger.py:76–79`) then resumes **any** waiting run whose `waitingThreadId` matches a
  posted message's thread — *before* it even looks at mentions. So an unguarded
  `POST /workflow-runs {"ctx":{"threadId":"<a real thread>"}}` would park a process run against that
  thread, and the next ordinary chat message there would silently drive it one step with no input
  and no guard data. **U3 rejects reserved keys on the *start* body's `ctx` too** — see §3.4.

---

## 3. Design & rationale

### 3.1 The central design choice: park-and-branch, so `_drive_loop` is never touched

The locked `_drive_loop` already implements *exactly* the mechanic a business process needs:

> execute step → evaluate outgoing guards → if none fires **and** the step declares
> `waitsForHuman` → `suspend_run` and return `waiting`.

So a `human` step is expressed as: **a step whose outgoing guards read data that is not in `ctx`
yet.** First pass: the guards are false (the key is missing) → the step parks. Input arrives → it is
merged into the run's `ctx` **by the resume CAS** (D-F) → `resume()` re-reads the run → the same step
re-executes → now the guard is true → it advances. No new outcome, no new state, no scheduler, no
change to the loop. *(Verified independently by the analyst against the real loop for every §4
shape.)*

That is why D-A (a guard that can read `ctx`) and D-B/D-F (a way to write `ctx`) are the *only* two
genuinely new capabilities in this slice. Everything else is assembly.

**`_drive_loop` lock — verification procedure for every unit's done-condition.** Verify by SHA
**only** (`71055f756280`); every byte count in the docs is wrong (miscopied three ways — see
`m3-executor-coordination.md`). Line-number-based extraction is brittle; use this
line-number-independent form, **re-verified by the architect at patch time (2026-07-20): prints
`71055f756280`**:

```bash
awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py \
  | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12
# => 71055f756280
```

If any unit finds it *must* change `_drive_loop`, that is a stop-and-escalate finding for teco with
written justification — not a silent edit.

> **What the lock does and does not cover — read this before U2/U3 (gate M-6).** The locked region is
> exactly `def _drive_loop` … up to the `# ── seams` marker (`executor.py:394`). This plan
> **deliberately edits `executor.py` in three places that are OUTSIDE that region**, and doing so is
> **not** a stop-and-escalate event:
> - `_execute_step` (`:396`) — after the marker (U2);
> - `_select_transition` (`:613`) and `_trace_step` (`:~700`) — well after the marker (U2, M-6);
> - `executor.resume` (`:282`) — **before** `_drive_loop` begins (U3, D-F).
>
> Re-run the SHA command after each of those edits: it must still print `71055f756280`. If it does,
> the lock is intact by construction. Do not freeze.

### 3.2 Deterministic guards — the `cmp` kind (D-A)

Guard grammar (all JSON, all already-parsed data — **no string parsing, no `eval`, no dependency**):

```jsonc
{"kind":"cmp","path":"ctx.decision","op":"eq","value":"approve"}
{"kind":"cmp","path":"ctx.request.role","op":"in","value":["contractor","exec"]}
{"kind":"cmp","path":"ctx.request.role","op":"exists"}
{"kind":"all","of":[ {…}, {…} ]}     // and
{"kind":"any","of":[ {…}, {…} ]}     // or
{"kind":"not","of":[ {…} ]}          // exactly one child
```

- **`path` roots (whitelist of exactly two):** `ctx.…` resolves into the run `ctx` dict (already
  passed to `evaluate_guard`); `output.…` resolves into the current step's `step_output` parsed as
  JSON (bare `output` = the raw string). Any other root ⇒ treated as **missing**. Traversal is dict
  key lookup only — no list indexing, no attribute access, no callables.
- **Ops (whitelist):** `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `in` (member ∈ list `value`),
  `contains` (list/str at `path` contains `value`), `exists`, `truthy`. Unknown op ⇒
  `WorkflowConfigError` (loud, named).
- **Missing path ⇒ `False` for every op**, including `exists` and `ne`. A missing value never fires a
  transition. This "bias to not-fire" mirrors the LLM guard's bias-to-suspend
  (`guards.py:150–168`): the safe direction is to park, because a parked run is unblockable and a
  wrongly-advanced run is not.
  > **Documented trap (gate m-10): this breaks De Morgan.** `{"op":"ne","path":p,"value":v}` and
  > `{"kind":"not","of":[{"op":"eq","path":p,"value":v}]}` **differ when `p` is missing**: `ne` →
  > `False` (bias to not-fire), `not(eq)` → `True` (`eq` is `False`, negated). This is intentional
  > — `not` negates a *verdict*, `ne` is a *comparison against a value that isn't there* — and it
  > **must** be stated in the `guards.py` module/function docstring and pinned by the contrasting
  > test pair in §7 U1 case 3b.
- **Type discipline:** no coercion. `lt/le/gt/ge` on non-comparable or mismatched types ⇒ `False`
  (never a `TypeError` that would fail the run). `eq`/`ne` use Python `==` on the JSON-native types.
- **Structural DoS caps (rule 6):** max nesting depth **5**, max total nodes **32**, max `of` width
  **8**; exceeding any ⇒ `WorkflowConfigError`. Guards are already capped at `MAX_CONFIG_LEN` (8000)
  at the API boundary.
- **`kind:'expr'` keeps raising `NotImplementedError`** — the "we did not build a language" seam
  stays visibly intact.

**Publish-time structural validation — `guards.validate_cmp(spec)` (gate M-4, authored in U1).**
The plan's own argument for the step invariant ("rejecting it at authoring time costs ~6 lines and
no runtime path") applies *more* strongly to guards: a typo'd `op` otherwise surfaces when a manager
clicks approve, killing a live run, instead of at seed time.

- `guards.validate_cmp(spec) -> None` raises `WorkflowConfigError` for: unknown `op`, `path` root not
  in `{ctx., output.}` (bare `output` allowed), missing `path`/`value` for ops that need them, `not`
  arity ≠ 1, and any depth/node/width cap breach. **One validator, two call sites** —
  `evaluate_guard` (runtime) and `services._validate_def_spec` (publish); no duplicated rules.
- **Call it only when `kind` ∈ `{cmp, all, any, not}`.** Fixture consequence to respect:
  `tests/test_services.py:726`'s `{"guard": {"expr": "x>0"}}` has **no `kind`** and must keep
  publishing successfully. `{"kind":"expr"}` is likewise untouched at publish (its `NotImplementedError`
  seam stays a *drive-time* seam).
- **`validate_cmp` takes an already-parsed dict.** Normalization of string-shaped guards happens at
  the `_validate_def_spec` call site (§3.3 normalization box, U2) — **U1's deliverable is unaffected
  by M-7**.

> **U2 ratifications of the three open items U1 left (O-1/O-2/O-3, coordination ledger).**
> Recorded here so no later unit re-derives them differently. All three are **as U1 built
> them** — U2 changed no `guards.py` behaviour, only called it.
> - **O-1 · unwhitelisted path root: strict at publish, total at drive.** §3.2 contradicted
>   itself (evaluation said "treat as missing ⇒ `False`", validation said
>   `WorkflowConfigError`). Both are right *for their call site*: an unresolvable path is an
>   **authoring defect** worth failing the publish, but at drive time it is only "a value that
>   is not there". One validator, one flag — `validate_cmp(spec)` checks paths,
>   `_evaluate_cmp` passes `check_paths=False`. **U2's publish call site uses the strict
>   (default) form**, so `services._validate_def_spec` rejects a bad root. Both halves are
>   pinned (`test_guards.py::test_an_undeclared_value_is_rejected_at_publish_but_total_at_
>   drive_time`).
> - **O-2 · `in` with a non-list literal: confirmed as-is — `False` at drive, NO publish
>   rule.** Two reasons, the second decisive: (a) §3.2 enumerates the validator's rules
>   exhaustively (unknown `op`, path root, missing `path`/`value`, `not` arity, caps) and a
>   type rule on the literal is not among them — inventing one at implementation time is
>   exactly the drift the O-list exists to prevent; (b) it would reject a shape that **works**:
>   `_in(left, value)` delegates to `_contains`, which accepts a **string** container, so
>   `{"op":"in","path":"ctx.tag","value":"abcdef"}` is a legitimate substring test. A rule
>   naïvely spelled "`value` must be a list" would forbid it. A genuinely dead literal (an int)
>   simply never fires — the same bias-to-not-fire every other missing/uncomparable case has.
> - **O-3 · bare `ctx` as a path: confirmed — not a value.** Rejected by `validate_cmp`,
>   `_MISSING` at drive. `ctx` is the whole run state, so `{"path":"ctx","op":"exists"}` would
>   be trivially true (and `truthy` would mean "the run has any state at all") — a guard that
>   looks like a data check and is actually a constant. Bare `output` **is** blessed (§3.2)
>   because it is one step's raw output string, i.e. an actual value.

**Trace rendering (gate M-6).** `GuardVerdict.rationale` is filled with a compact rendering
(`"ctx.decision eq 'approve' → true"`). To make that reach the debug trace, U2 edits
`_select_transition` (`executor.py:636`) so the judgment filter is
`parsed.get("kind") in {"llm","cmp","all","any","not"}`, and the label passed alongside the verdict
becomes `parsed.get("text") or guards.render_label(parsed)` — because a `cmp` guard has no `text`,
and the current `_trace_step` format (`f"{text} -> …"`, `executor.py:706`) would otherwise emit a
line beginning with a bare `" -> "`. `render_label` is a small pure helper in `guards.py` (U1),
reused by `rationale`. **These two functions are outside the SHA-locked region** — see the §3.1 box.

*Rejected:* putting the comparator in `executor._select_transition`. Guards belong in `guards.py`;
the executor is dispatch, not policy.

### 3.3 Typed step handlers (D-E)

`_execute_step` becomes an explicit dispatch table. Handlers are **pure and side-effect-free** —
their entire job is to produce an auditable `StepResult` describing what the step is/was waiting for.
All branching stays in the guards.

| `Step.type` | Handler | Behaviour |
|---|---|---|
| `agent` + wired LLM | `_run_agent_node` | **unchanged** |
| `agent`, no LLM | (fall-through stub) | **unchanged, deliberately** — F-3 case 1, the offline test affordance (and, after the B-1 fixture edit, F-3 case 2's `end` step). Documented in the docstring as such, not as an accident |
| `decision` | `_run_decision_node` | No side effect. Returns `StepResult(output=json({"node":{"step":<key>}}), on="done", trace=[("node_note","decision node — branching in guards")])`. **Envelope key is `node`, not `decision`** (gate n-1: `decision` would collide semantically with `ctx.decision`, the approval value in this very def). Its semantics live entirely in its outgoing guards; with **zero** outgoing transitions it is a terminal outcome node |
| `human` | `_run_human_node` | Returns `StepResult(output=json({"awaiting":{"kind":"human","prompt":cfg.prompt,"assignee":cfg.assignee,"fields":cfg.fields}}), on="done")`. **`on="done"`, not a new `"await"` value** (gate n-2: `on` is vestigial per F-1; inventing a value for a field this plan declares dead is inconsistent). The output lands on the `StepRun` ⇒ **`GET /workflow-runs/{id}/step-runs` tells a client exactly what the run is waiting for**, with no new query |
| `wait` | `_run_wait_node` | Same shape with `{"kind":"signal","signal":cfg.signal}`, `on="done"` (D-C — signal, not timer). **Mechanically identical to `human`** (m-7) — only this string differs |
| `prompt` / `tool` / `message` | — | `raise NotImplementedError(f"step type {t!r} is not implemented in this cut (typed-handler seam); see docs/archive/plans/m3-process-flow.md §D-E")` → M-1 fault net → `fail_run` → D-G envelope |
| unknown type | — | same `NotImplementedError` path |

**Publish-time invariant (new, in `services._validate_def_spec`):** a step of type `human` or `wait`
**must** declare `config.waitsForHuman: true`, else `WorkflowDefSpecError`. Without it such a step
self-loops against the budget until `fail_run` — a silent, expensive footgun.

> **Shape normalization — read before writing either invariant (re-gate M-7).**
> `_validate_def_spec` receives `config`/`guard` **heterogeneously typed**: serialization to opaque
> strings happens *after* it, in `publish_workflow_def` (`services.py:494/504`), so REST callers
> deliver **strings** (`schemas.py:48` `config: str | None`, `:58` `guard: str | None`;
> `test_api.py:418` is `"config": "{}"`) while service-layer callers deliver **dicts**
> (`test_services.py:719` `{"a": 1}`) or even non-JSON strings (`:720` `"raw-string"`).
> Both new invariants therefore **must normalize first**, in `_validate_def_spec` itself:
> - if the value is a `str`, `json.loads` it (tolerating failure); otherwise use it as-is;
> - **guard check:** a non-dict result ⇒ treated as *no declaration* ⇒ `validate_cmp` is not called
>   (so `{"expr":"x>0"}` and `"raw-string"` keep publishing unchanged);
> - **step check:** a `human`/`wait` step whose config does not normalize to a dict ⇒
>   **`WorkflowDefSpecError`** — a step that must declare `waitsForHuman` cannot carry an opaque
>   config.
>
> The two failure modes this forecloses, both easy to land silently: naive `.get()` on a str ⇒
> `AttributeError` ⇒ **500 on `POST /workflow-defs`**; a bare `isinstance(..., dict)` skip ⇒ **every
> REST-published def escapes both invariants**, which would gut M-4's "caught at authoring time"
> argument for the actual front door. `guards.validate_cmp` keeps taking an **already-parsed dict**
> — normalization is the call site's job, so U1's scope is untouched.
>
> **Where it runs, and the fixture debt it creates (gate B-2).**
> **Ordering:** this check runs **last** in `_validate_def_spec` — after the step-type whitelist, the
> key-uniqueness check, the start-count check and the dangling-endpoint checks. Running it last
> minimises masking: the existing `pytest.raises(WorkflowDefSpecError)` tests keep failing for their
> *own* reason.
> **Fixture edits (explicitly in U2's scope, budgeted, not incidental):**
> (a) declare `waitsForHuman` on the `human` steps in `tests/test_api.py:418` and
> `tests/test_services.py:719, 808, 822, 836, 851, 866`. **Mind the shape (M-7):**
> `test_api.py:418`'s config is a **string**, so the edit is `"config": "{}"` →
> `"config": '{"waitsForHuman": true}'`; the `test_services.py` ones are dicts (or absent) and take
> `"waitsForHuman": True` in a dict;
> (b) **tighten the five `pytest.raises(WorkflowDefSpecError)` tests** at `test_services.py:804,
> 818, 832, 846, 861` (**locate them by test name, not line** — n-6: these citations drifted once
> already) to assert on the message (`match="duplicate step key"`, `match="exactly one
> start step"`, `match="not a declared"`, …). They currently assert only the exception *type*, so a
> new check firing first would make them **pass vacuously** — a silent regression-net hole rather
> than a red test. Tightening them makes that class of failure impossible in future too.
> These edits are named test-by-test so the reviewer can confirm no *other* existing test was
> touched — see U2's done-condition (§6) and §7.

> **As built (U2, 2026-07-20) — two notes where §3.3 was underspecified.**
> - **Where the `waitsForHuman` check lives:** `services.WAITING_STEP_TYPES =
>   {"human","wait"}`, checked with the same **truthiness** the engine uses
>   (`config.get("waitsForHuman")`, `executor.py` OUTCOME B) rather than an `is True`
>   identity test — a publish invariant that accepted a value the engine would then ignore
>   (or vice versa) would be worse than no invariant.
> - **Normalization helper:** `services._normalize_opaque` — the deliberate inverse of the
>   existing `_serialize_opaque`, so the two shapes of the same field are handled by a
>   matched pair rather than an ad-hoc `json.loads` at each check.
> - **Handler signature:** the three handlers take `(step, config)` only. §3.3 calls them
>   "pure and side-effect-free", and giving them `ctx`/`run`/`run_ctx` they must not use would
>   have made that a comment instead of a fact. `config.fields` is read through a defensive
>   list-of-strings view (author-supplied data must not raise inside a pure handler).

**Deliberately *not* enforced (gate n-3, declined with reason):** "a `decision` step with outgoing
transitions must have an unconditional default or `waitsForHuman`". It is a real footgun (OUTCOME C
self-loops to budget exhaustion, `executor.py:384–388`) and `access-request@v1` avoids it (#3 is
unconditional) — but the symmetric invariant would retro-reject existing fixtures such as
`test_services.py:719`'s `review` `decision` step, whose only outgoing transition is guarded. That is
fixture churn beyond this slice's blast-radius budget for zero proof value. **Instead:** U2 adds it
as a *warning* bullet to `falkor-chat/AGENTS.md`'s executor-invariants list, and U5 files it with
K-028's neighbours as a proposed hardening.

### 3.4 Start + input path (D-B, D-F, D-G, D-H)

```
POST /workflow-runs                  {defKey, version, ctx?, trace?, maxSteps?}  -> 201 {runId, status, …}
POST /workflow-runs/{runId}/input    {input: {...}}                              -> 200 {runId, status, ctx}
GET  /workflow-runs/{runId}                                                      (exists)
GET  /workflow-runs/{runId}/step-runs                                            (exists — carries `awaiting`)
```

**`services.start_workflow_run(ctx, *, def_key, version, trigger_msg_id: str | None = None,
run_ctx: dict | None = None, trace=False, max_steps: int | None = None)`**
1. `trigger_msg_id` present ⇒ the existing path, **byte-identical in behaviour** (`repo.start_run`,
   `TRIGGERED_BY`, ctx `{"threadId": …}`).
2. `trigger_msg_id is None` ⇒ `repo.start_run_untriggered` (§12.12). The initial ctx is the caller's
   `run_ctx` (default `{}`).
3. **Reserved-key rule on the start ctx (gate M-2, NEW):** `threadId` and `error` are rejected
   (`WorkflowInputRejectedError` → 400, nothing started) — **in the service, not only in the pydantic
   schema**, because MCP/service callers bypass schemas. Rationale is F-6: a caller-set `threadId`
   would make `suspend_run` denorm a *real* thread, and `trigger.py` step 2 would then drive the
   process run on any ordinary chat message posted there. F-5's latent bug, live and reversed.
   *(Declined as additional belt: stamping a never-matchable sentinel `waitingThreadId`. It would
   require a matching filter on the trigger side and pollutes the denorm's meaning; the reserved-key
   rejection plus F-5's falsy short-circuit already makes `''` unmatchable from both ends.)*
4. Drive synchronously; on a drive fault apply **D-G** (catch → re-read → `{"status":"failed",
   "error":…}` at **201**).

**`services.submit_workflow_input(ctx, *, run_id, input)`:**
1. `repo.get_run` → `None` ⇒ `WorkflowRunNotFoundError` (**404**, handler added in U3 per D-G).
2. `run["status"] != "waiting"` ⇒ **new** `WorkflowRunNotWaitingError` (**409**). A run that is
   `running`/`done`/`failed` has nothing to unblock.
3. **Validate before touching anything (D-H):** reserved keys (`threadId`, `error`) ⇒ 400; then
   resolve the parked step (`run["atStepKey"]` + `repo.get_snapshot`) and check the submitted keys
   against `config.fields` / `config.signal` and `config.expects` ⇒ 400. **Nothing written, no step
   budget consumed.**
4. Merge `input` **flat** into the deserialized `ctx` (so guard paths read `ctx.decision`, not
   `ctx.input.decision`); enforce the **merged** serialized-size bound `MAX_CONFIG_LEN` here
   (gate m-5 — pydantic bounds the *input*, the service bounds the *merge*).
5. **One query (D-F):** `executor.resume(ctx, run_id=…, run_ctx_json=json(merged))` →
   `repo.resume_run_with_ctx` (§12.13). Zero rows (lost the CAS or no longer waiting) ⇒ **409**, and
   the ctx is **not** written. Otherwise the drive proceeds against exactly the ctx that won.
6. Return `{runId, status, ctx}` — or D-G's failed envelope at **200** if the drive faulted.

**Synchronously**, not on `BackgroundTasks`: a process drive is pure graph work with no LLM, so it
is fast and — crucially — **deterministically testable**. A future LLM-bearing process def would want
the background path; noted, not built. D-G is what makes synchronous driving safe at the HTTP
boundary.

*Rejected:* passing the input as an in-memory overlay to `executor.resume` to avoid the new query.
It would keep the input out of the graph, so the run's own `ctx` would not reflect the decision that
advanced it — destroying the audit property that is the entire point of §6.3 — and a second resume
would lose it.

*Rejected (D-F):* a separate `set_run_ctx` write. See D-F for the full argument; the short form is
that a split write admits a **silent wrong branch**, not merely a lost input.

### 3.5 Concurrency & atomicity posture

- **The ctx write and the resume CAS are one query** (D-F, §12.13). Consequences:
  - Only the **CAS winner's** ctx is ever written. "Which input advanced the run" and "which input
    is in `ctx`" can never disagree — the run's ctx always explains its own trail.
  - A loser gets **409** and its input is **not persisted** — a visible rejection the caller can
    retry, not a silent loss.
- **Residual window (R-1):** the *read* before the merge is still non-transactional. Two submitters
  can both `get_run`, both merge onto the same base ctx, and the CAS winner's merge may therefore
  omit a key the loser had intended to add. That is a **lost update on an unwritten input, reported
  as a 409 to its submitter** — not a wrong branch and not an erased key that a prior step branched
  on. Single-approver use case today; the real fix (a `ctxVersion` counter + CAS on it) is a
  follow-up, deliberately not built.
- Everything else reuses the existing per-query-atomic writes unchanged.

---

## 4. The proof def — `access-request@v1` (D-D)

`kind: 'process'`, key **`access-request`**, version `v1`. **New def — additive; `triage@v1`
untouched.** (Renamed from `onboarding` per gate m-2 — that key is taken by long-standing test
fixtures.)

### 4.1 Steps

| Step | `type` | `config` | Role |
|---|---|---|---|
| **submit** (start) | `human` | `{"waitsForHuman":true,"prompt":"File the access request","fields":["request"],"assignee":"requester"}` | Parks until the request is filed. **`fields` lists the accepted top-level input keys** (D-H rule 2); the submitted `request` is a nested object accepted whole — validation is on top-level keys only, there is no deep schema |
| **route** | `decision` | `{}` | Pure branch: privileged roles need approval, standard hires do not |
| **approval** | `human` | `{"waitsForHuman":true,"prompt":"Approve or reject this access request","fields":["decision"],"expects":{"decision":["approve","reject"]},"assignee":"manager"}` | Parks until a manager decides. **`expects` makes any other value a free 400** (D-H rule 3) — it can never burn budget |
| **provision** | `wait` | `{"waitsForHuman":true,"signal":"provisioned"}` | Parks until the provisioning system signals back. **No `expects`** — deliberately, so `{"provisioned": false}` ("not yet") stays expressible; it re-parks and costs one step |
| **activate** | `decision` | `{}` | **Terminal** — happy-path outcome (no outgoing ⇒ run `done`) |
| **rejected** | `decision` | `{}` | **Terminal** — rejected outcome (no outgoing ⇒ run `done`) |

**Declared budget (D-H part c): `maxSteps = 24`**, exported as `proof_defs.ACCESS_REQUEST_MAX_STEPS`
and passed by the seed/acceptance paths on the start body. Happy path = 8 ⇒ **16 spare re-parks**.
Values the def declares invalid cost **nothing** (free 400). A caller that omits `maxSteps` falls back
to the global default 12 ⇒ 4 spare re-parks — documented here, not discovered in production.

> A rejected request still ends the run `done`, not `failed`: the *process* completed; the *outcome*
> is the terminal step reached (readable from the `NEXT`-ordered step-run trail). `failed` remains
> reserved for engine faults and budget exhaustion.

### 4.2 Transitions — **six**

| # | from → to | `on` | `order` | guard |
|---|---|---|---|---|
| 1 | submit → route | `filed` | 0 | `{"kind":"cmp","path":"ctx.request.role","op":"exists"}` |
| 2 | route → approval | `needs_approval` | 0 | `{"kind":"cmp","path":"ctx.request.role","op":"in","value":["contractor","exec"]}` |
| 3 | route → provision | `auto` | 1 | `""` (unconditional default — fires only if #2 does not) |
| 4 | approval → provision | `approved` | 0 | `{"kind":"cmp","path":"ctx.decision","op":"eq","value":"approve"}` |
| 5 | approval → rejected | `rejected` | 1 | `{"kind":"cmp","path":"ctx.decision","op":"eq","value":"reject"}` |
| 6 | provision → activate | `provisioned` | 0 | `{"kind":"cmp","path":"ctx.provisioned","op":"truthy"}` |

Coverage: `human` ×2, `decision` ×3, `wait` ×1; **four** `cmp` ops — `exists` / `in` / `eq` /
`truthy`; conditional-beats-unconditional ordering (#2 vs #3, the existing `_select_transition`
rule); two-way branch (#4/#5) where *neither* firing ⇒ the step re-parks. With `expects` on
`approval` (D-H) a garbage `decision` value never reaches that re-park at all — it is rejected at the
boundary for free; the re-park remains the correct fallback for any value the def does not constrain.
(`on` values are descriptive labels only — F-1.)

### 4.3 The privileged-role happy path, step by step (also the acceptance-test script)

| # | Action | Drive | Step executed | Guard | Outcome | `stepCount` |
|---|---|---|---|---|---|---|
| 1 | `POST /workflow-runs {access-request,v1,maxSteps:24}` | 1 | submit | #1 false (no `ctx.request`) | park → `waiting` | 1 |
| 2 | `POST …/input {"request":{"role":"contractor","laptop":true}}` | 2 | submit | #1 true | advance → route | 2 |
| 3 | | 2 | route | #2 true (`contractor`) | advance → approval | 3 |
| 4 | | 2 | approval | #4/#5 false | park → `waiting` | 4 |
| 5 | `POST …/input {"decision":"approve"}` | 3 | approval | #4 true | advance → provision | 5 |
| 6 | | 3 | provision | #6 false | park → `waiting` | 6 |
| 7 | `POST …/input {"provisioned":true}` | 4 | provision | #6 true | advance → activate | 7 |
| 8 | | 4 | activate | no outgoing | `complete_run` → **`done`** | 8 |

8 steps of the declared 24-step budget (F-4/D-H). The standard-hire path (`role:"engineer"`) costs 6;
the rejected path 6. *(Arithmetic re-derived independently by the analyst.)*

### 4.4 Where the def lives — single source of truth

Put the spec in a new module **`server/falkorchat/proof_defs.py`**
(`ACCESS_REQUEST_DEF = {...}`, `ACCESS_REQUEST_MAX_STEPS = 24`), imported by **both**
`scripts/seed_workflows.sh`'s Python one-shot **and** the acceptance test. This is a direct response
to the K-022 U14 lesson: `test_workflow_live.py` had to shell out to the seed script precisely
because a copied def spec would drift. An importable constant gets the same no-drift property with no
subprocess and no network.

**Why proof/demo def *content* belongs in the shipped package:** it is **data the test must import**,
and the installed package is the only artifact both an offline test and a shell script can import
without a subprocess. It adds ~2 KB of constants and no runtime behaviour — `proof_defs` is imported
by nothing on the request path.

**Two def-source conventions in one script (gate m-9) — declined for this slice, with reason.**
`seed_workflows.sh` keeps `triage@v1` inline in its `<<'PY'` heredoc (~lines 85–190) while
`access-request@v1` comes from the import; the script loops over `[triage-spec-literal,
ACCESS_REQUEST_DEF]`. Moving the *published, live* triage literal into `proof_defs.py` during a slice
whose seeding path is already split-brain-prone buys no proof value and risks a byte-diff that
`MERGE … ON CREATE SET` would silently swallow (the re-seed prints `already present — no-op` while
the file and the graph disagree). **U5 files it as proposed backlog item K-029 — "converge seed def
sources into `proof_defs.py`"** so the two-convention state is recorded, not forgotten.

**Seeding / verification order (the immutability + split-brain rule, AGENTS.md; gate m-8):**

```
./scripts/bootstrap_schema.sh <ws>  →  ./scripts/seed_demo.sh <ws>  →  server pytest
  →  ./scripts/test_queries.sh      →  ./scripts/seed_workflows.sh <ws>
  →  verify (service-layer one-shot; add ./scripts/start_server.sh only for the REST check)
```

`pytest` (via the `wf_repo` fixture) and `test_queries.sh` both wipe `reference`; the `ws:<id>`
snapshot survives ⇒ **always re-seed last, then verify**. **Verification is via a service-layer
one-shot by default** (`get_def` / `list_snapshots` through the same venv the seed script uses) — the
first draft's `GET /workflow-defs/…` / `GET /workspaces/acme/snapshots` need uvicorn running, which
the order above did not include; run `start_server.sh` only if the REST surface is what is being
checked. Because `access-request@v1` is brand new there is no stale snapshot to split-brain against
on first landing — but from the second landing onward the rule bites, and any *edit* to the def
content requires a `key`/`version` bump (published defs are `MERGE … ON CREATE SET`, i.e.
**create-only**: re-seeding an edited def is a silent no-op).

**No new config var.** `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` stay `triage`/`v1` — the
process def is started over REST, not by `@mention`, so nothing in `config.py`, `.env.example`, or
`start_server.sh` changes. (The seed script may accept `FALKORCHAT_PROCESS_DEF_KEY`/`_VERSION`
locally, documented in its header only. This deliberately avoids the documented trap that
`start_server.sh` forwards neither var.)

---

## 5. Graph-dba gate — **REQUIRED** (small, additive, zero DDL)

**Two new queries — unchanged in count from the first draft; D-F *replaced* the second one.**
No new label, no new property on an existing hot node beyond a longer `ctx` string, no new index, no
DDL change ⇒ **RAM impact ≈ nil** (rule 6): `ctx` already exists on `WorkflowRun`; the input merge
grows it by tens of bytes per run, and process runs are rare compared to messages.
`bootstrap_schema.sh` is **not** touched. *(The analyst independently confirmed the required indexes
already exist: `WorkflowDefSnapshot.key` :117, `.version` :120, `WorkflowRun.runId` :123, and the
`WorkflowRun` uniqueness constraint :179.)*

**Numbering (gate n-5):** use **§12.12** and **§12.13** — plain numeric headings that continue the
existing `### 12.1 … ### 12.11` sequence. The first draft's "§12.1b" broke the convention;
§12.12 cross-references §12.1 instead.

1. **`start_run_untriggered`** → **QUERIES.md §12.12** — `start_run` (§12.1) minus the
   `MATCH (trigger:Message …)` anchor and the `CREATE (r)-[:TRIGGERED_BY]->(trigger)`; everything
   else identical, and the entry cross-references §12.1 as its parent.
   *Deliberately a second query, not an `OPTIONAL MATCH`+`FOREACH` conditional* — it follows the
   locked project doctrine of two separate self-contained write paths (the §4 first/subsequent
   precedent) and avoids the documented empty-row-collapse class of bug. The gate PROFILEs it
   (expect a single `Node By Index Scan` on `WorkflowDefSnapshot.key`) and confirms the zero-row
   contract (snapshot has no `START`).
2. **`resume_run_with_ctx`** → **QUERIES.md §12.13** (D-F; **replaces** the proposed `set_run_ctx`,
   which is not part of this slice at all) — `resume_run` (§12.4) with one extra `SET` term:
   ```cypher
   // $runId, $ctx (opaque serialized run state)
   MATCH (r:WorkflowRun {runId: $runId})
   WHERE r.status = 'waiting'
   SET r.status = 'running', r.waitingThreadId = '', r.ctx = $ctx
   RETURN r.runId AS runId, r.status AS status
   ```
   Gate confirms: index-anchored point lookup on `runId`; the `WHERE` makes a non-waiting run a clean
   **zero-row no-op with neither the flip nor the ctx written** (this is the single-flight property
   the whole D-F argument rests on — verify it, do not assume it); parameterised; cross-references
   §12.4 as its parent and notes that §12.4 remains in use for the chat/trigger resume path.

Gate deliverables: both queries live-verified + PROFILEd, written into `QUERIES.md` §12, plus
enumerated `test_queries.sh` assertions raising the pinned count from **241** to the gate's new
number (expect ~4–6 new assertions: untriggered start creates the run subgraph **without**
`TRIGGERED_BY`; missing-snapshot ⇒ zero rows; resume-with-ctx from `waiting` flips status **and**
writes ctx; resume-with-ctx against a `running` run ⇒ zero rows **and ctx unchanged**; index-scan
profiles). The gate pins the new count in `QUERIES.md`/`DESIGN §7.1` in the same pass.

---

## 6. Units — sequenced, independently reviewable

Each unit ends with: **its own suite green** (see the per-unit clause below — gate m-3), **`_drive_loop`
SHA re-verified `71055f756280`** (§3.1 command), and its named docs updated **in the same change**.

- **U0's suite clause:** `./scripts/test_queries.sh` green at the newly pinned count. U0 adds **no**
  pytest tests; the pytest baseline must be **unchanged** (still 350).
- **U1–U4's suite clause:** server pytest count **up** from the previous unit's number with **no
  existing test silently weakened** (see §7), and `test_queries.sh` green at U0's pinned count.

### U0 — graph-dba gate (queries only) · owner `graph-dba`
Per §5, **as amended by D-F**: `start_run_untriggered` (§12.12) + **`resume_run_with_ctx` (§12.13)**.
`set_run_ctx` is **not** part of this gate — D-F folded the ctx write into the resume CAS, so §12.13
*replaces* it rather than adding a third query.
**Done:** §12.12 + §12.13 in `QUERIES.md` with PROFILE notes and the RAM statement; the §12.13 entry
explicitly documents the **zero-row ⇒ nothing written (neither flip nor ctx)** contract;
`test_queries.sh` assertions added and the new count pinned; query suite green at the new count;
pytest still 350; explicit written confirmation that **no** `bootstrap_schema.sh` / index / DDL
change was needed (or, if the gate disagrees, that finding escalates to teco before U3 starts).
**Blocks:** U3. **Docs:** `QUERIES.md` §12; `DESIGN.md` §7.1 count refresh if the gate changes it.

### U1 — deterministic `cmp` guard · owner `tdd-engineer` (D-A)
`server/falkorchat/guards.py` only. Add `kind:'cmp'|'all'|'any'|'not'` per §3.2; add
**`validate_cmp(spec)`** (the publish-time structural validator, called by U2) and **`render_label`**
(used by `rationale` and by U2's trace edit); keep `expr` raising; keep the `llm` and unconditional
branches byte-identical in behaviour.
**Done:** `server/tests/test_guards.py` extended (see §7); no other module touched; `guards.py`
module docstring updated to describe three live branches + the seam, **and to state the De Morgan
asymmetry (m-10) explicitly**.
**Docs:** `DESIGN.md` §6.1 (guard discriminator now `""` | `{kind:'llm'}` | `{kind:'cmp'|all|any|not}`),
`DESIGN.md` §13 (the guard open question: note that the *deterministic* half is now a structured
comparator, explicitly **not** an expression language — the resolution stands).
**Independent of U0** — can run in parallel.

### U2 — typed step handlers + publish invariants + trace · owner `coder` (D-C, D-E; M-4, M-6, B-1, B-2)
**File scope (explicit — every file that will show in the diff):**
- `server/falkorchat/executor.py` — `_execute_step` dispatch + three handlers; **and
  `_select_transition` + `_trace_step`** for the cmp judgment/label (M-6). **`_drive_loop`
  untouched**; all three edit sites are outside the SHA-locked region (§3.1 box).
- `server/falkorchat/services.py` — `_validate_def_spec`: **the `config`/`guard` string-vs-dict
  normalization (M-7, §3.3 box) first**, then the `human`/`wait` ⇒ `waitsForHuman` invariant (running
  **last** among the spec checks, §3.3) **and** the `cmp`-family guard call to `guards.validate_cmp`
  (M-4; skip guards that do not normalize to a dict or carry no `kind`, so `{"expr":"x>0"}` and
  `"raw-string"` keep publishing).
- `server/tests/test_executor.py` — **one named fixture edit**: `:345` `"type":"task"` →
  `"type":"agent"` (B-1/F-3 case 2), comment updated.
- `server/tests/test_api.py` — **one named fixture edit**: `:418` `"config": "{}"` →
  `"config": '{"waitsForHuman": true}'` (**string-shaped**, M-7).
- `server/tests/test_services.py` — **named fixture edits** at `:719, 808, 822, 836, 851, 866` (add
  `"waitsForHuman": True`) **and** `match=` tightening on the five `pytest.raises` tests at
  `:804, 818, 832, 846, 861` — **locate by test name; these line numbers have drifted once** (B-2,
  n-6).
- `server/tests/test_executor_process.py` — **new** (§7).

**Done:** the §7 U2 behaviours pass; **the pytest count rises and the only pre-existing tests
modified are the ones named above** (see §7's honest done-condition); the `agent`-without-LLM stub
path is preserved (F-3) and its docstring now says *why*; `prompt`/`tool`/`message` raise with a
message naming this plan.
**Docs:** `DESIGN.md` §6.1 — step-type bullet list (which types the engine executes today), `wait` is
**signal-driven, not timer-driven, and mechanically identical to `human`** (D-C, m-7), and the F-1
correction to the `on` wording **in both halves** (m-6: `on` is descriptive; the sort key is
`(guard == "", order)`, i.e. conditional guards first with `order` as an intra-class tie-break);
`falkor-chat/AGENTS.md` — a new executor-invariants bullet ("a `human`/`wait` step must declare
`waitsForHuman`, enforced at publish; a `cmp`-family guard is structurally validated at publish; a
`decision` step has no side effect — its semantics are its outgoing guards; **a `decision` step whose
outgoing transitions are all conditional and which does not declare `waitsForHuman` self-loops to
budget exhaustion — not enforced, see n-3**; `prompt`/`tool`/`message` raise").
**Depends on:** U1 (hard now — `validate_cmp`/`render_label` are U1 deliverables; sequence U1 → U2).

### U3 — start-without-trigger + input endpoint + error map · owner `coder` (D-B, D-F, D-G, D-H; M-2)
`repository.py` (`start_run_untriggered`, **`resume_run_with_ctx`** — 1:1 with U0's §12.12/§12.13 —
plus new `WorkflowRunNotWaitingError` and `WorkflowInputRejectedError`), `executor.py`
(`resume(..., run_ctx_json=None)` dispatch — **outside the locked region**, §3.1), `services.py`
(`start_workflow_run(trigger_msg_id: str | None = None, run_ctx=None, max_steps=None)` dispatching to
the two repository paths — the §4 first/subsequent precedent; **reserved-key rule on the start ctx**
(M-2/F-6); `submit_workflow_input` per §3.4 including the **D-H validation** and the merged-ctx size
bound (m-5); the D-G drive-fault catch; the F-5 one-line short-circuit in
`find_waiting_run_for_thread`; `_require_executor` raises the new
`WorkflowEngineDisabledError(RuntimeError)`), `schemas.py` (`StartWorkflowRunIn`,
`SubmitWorkflowInputIn` — bounded: ≤32 input keys, key ≤ `MAX_KEY_LEN`, **serialized *input* ≤
`MAX_CONFIG_LEN`**, `maxSteps` 1…50), `api.py` (two routes), `app.py` (**five** handlers per D-G:
`WorkflowRunNotFoundError` → 404, `WorkflowRunNotWaitingError` → 409, `WorkflowInputRejectedError` →
400, `WorkflowConfigError` → 400, `WorkflowEngineDisabledError` → 503).
**Done:** `test_api.py` + `test_services.py` additions (§7), **one case per new handler**; the error
map is exactly the D-G table; no change to the existing `@mention` start path's behaviour (pinned by
a test).
**Depends on:** U0. **Docs:** `DESIGN.md` §14.4 REST-surface list (add the two routes **and the D-G
error map**); **`DESIGN.md` §6.2** — the run-`ctx` write posture: the resume CAS owns the ctx write
(D-F) and the residual stale-merge-read window (R-1) *(gate m-4: this doc had no owning unit)*.

### U4 — the proof def, seed & offline acceptance · owner `coder` (D-D)
`server/falkorchat/proof_defs.py` (`ACCESS_REQUEST_DEF`, `ACCESS_REQUEST_MAX_STEPS`),
`scripts/seed_workflows.sh` (loop over both defs; header/usage updated).
**Done:** new `server/tests/test_process_flow.py` — the §4.3 happy path plus the standard-hire and
rejected branches, driven **through the service/REST layer** against `ws:test`, fully offline
(no `live` marker); `./scripts/seed_workflows.sh acme` run per the §4.4 order and the snapshot
verified readable **via a service-layer one-shot** (`get_def("access-request")` / `list_snapshots`;
`start_server.sh` only if the REST surface is being checked — m-8); a re-run prints `already present
— no-op` for both defs.
**Depends on:** U1, U2, U3.
**Docs:** `falkor-chat/AGENTS.md` key-scripts table — the `seed_workflows.sh` row now seeds **two**
defs (`triage@v1` conversational + `access-request@v1` process), with the create-only/split-brain
warning extended to cover both.

### U5 — closeout · owner `teco` (integration)
**Done:** `DESIGN.md` §6.3 gains a pointer that the proof now exists and where it lives (def +
test), **repeating the m-7 statement that `wait` is signal-driven and mechanically identical to
`human`, as the K-025 handoff note**; `docs/BACKLOG.md` — **K-024 → ✅** with a one-paragraph
delivery note, **K-025 marked unblocked**, the milestone-map row updated, **K-028 (workflow
timers/scheduled wakeups) filed as 🔵 proposed** carrying D-C's reasoning, and **K-029 (converge seed
def sources into `proof_defs.py`; consider the symmetric `decision`-step publish invariant of n-3)
filed as 🔵 proposed**; `docs/HISTORY.md` — a dated entry (naming the R-3 behaviour change and the
named fixture edits); final full-suite run in the §4.4 order with the counts recorded; the
coordination ledger's Decisions table filled in with the settled D-A…D-H.

---

## 7. Test strategy

Altitudes, per unit. Everything below is **offline and deterministic** — no `live` marker, no LM
Studio, no network. Graph-backed tests use the existing `wf_repo`/`conn` fixtures against `ws:test`.

**U1 · unit (`test_guards.py`)** — behaviours to drive red→green, in order:
1. `cmp/eq` true and false on a `ctx` path; 2. dotted path into a nested dict (`ctx.request.role`);
3. missing path ⇒ `False` for each of `eq`, `ne`, `lt`, `contains`, `truthy`, and `exists`;
**3b. the De Morgan contrast pair (m-10):** on a *missing* path, `{"op":"ne"}` ⇒ `False` **while**
`{"kind":"not","of":[{"op":"eq",…}]}` ⇒ `True` — asserted side by side in one test so the asymmetry
is pinned, not discovered;
4. `exists` true on a present-but-falsy value (`0`, `""`, `false`) — the `exists`/`truthy`
distinction; 5. `in` with a list `value`; 6. `contains` on a list and on a string;
7. ordering comparisons on ints and on strings; 8. mismatched types on `lt` ⇒ `False`, **not** a
raised `TypeError`; 9. `output.` root reading the current step's JSON output; bare `output` reading
the raw string; 10. unknown root (`foo.bar`) ⇒ missing ⇒ `False`; 11. `all`/`any`/`not` including
empty `of` (define and assert: `all([]) → True`, `any([]) → False`, `not` with ≠1 child ⇒
`WorkflowConfigError`); 12. depth cap, node cap, width cap each ⇒ `WorkflowConfigError`;
13. unknown `op` ⇒ `WorkflowConfigError`; 14. `kind:'expr'` still ⇒ `NotImplementedError`;
15. `""` and `{"kind":"llm"}` behaviour unchanged (regression pins); 16. `rationale` is populated
and readable; **17. `validate_cmp` (M-4): accepts every guard shape used in §4.2; raises
`WorkflowConfigError` on unknown `op`, bad path root, `not` arity ≠ 1, and each cap breach — the same
rules as evaluation, one implementation; 18. `render_label` produces a non-empty, `text`-free label
for a `cmp` guard (the M-6 trace input).**

**U2 · unit/integration (`test_executor_process.py` + the named fixture edits)** —
1. a `human` step with a false guard parks the run (`status == 'waiting'`, `AT_STEP` still on it);
2. its `StepRun.output` carries the `awaiting` envelope with `prompt`/`fields`/`assignee`;
3. `wait` ditto with the `signal` envelope; 4. a `decision` step advances via the firing guard and
records an output with **the `node` envelope key (n-1)** and no side effect; 5. a `decision` step with
no outgoing transitions terminates the run `done`; 6. `prompt`/`tool`/`message` ⇒ the run ends
`failed` with a readable `ctx.error` and the exception re-raised **out of the executor** (the M-1 net
contract, mirroring `test_executor.py::test_llm_guard_without_judge_fails_the_run_with_named_error`;
the HTTP-level conversion to D-G's 200/201 envelope is tested in U3);
7. **regression pin:** `type:'agent'` with `llm=None` still returns the empty stub result (F-3);
8. `_validate_def_spec` rejects a `human`/`wait` step without `waitsForHuman`
(`WorkflowDefSpecError`, nothing written) and accepts one with it; **8b. ordering pin (B-2): a spec
that violates *both* the duplicate-key rule and the `waitsForHuman` rule raises with the
*duplicate-key* message — proving the new check runs last and cannot mask the older ones**;
9. a debug run records a `guard_judgment` trace line for a `cmp` guard whose payload **starts with
the guard label, not `" -> "`** (M-6);
**10. publish rejects a malformed `cmp` guard (unknown `op`) with `WorkflowConfigError`, nothing
written (M-4); 11. publish still accepts a `kind`-less guard (`{"expr":"x>0"}`) and a non-JSON
`"raw-string"` config unchanged (the `test_services.py:726`/`:720` fixture contract);
12. M-7 shape matrix — a `human` step published with a **string** config `'{"waitsForHuman":true}'`
is accepted (the REST shape); the same step with `"config": "{}"` (string) is **rejected**
`WorkflowDefSpecError`; a `cmp` guard delivered as a **JSON string** is validated exactly like the
dict form (unknown `op` ⇒ `WorkflowConfigError`) — i.e. the REST front door cannot escape either
invariant; a `human` step with an opaque non-JSON config ⇒ `WorkflowDefSpecError`.**

**U3 · integration (`test_services.py`, `test_api.py`)** —
1. `start_workflow_run(trigger_msg_id=None)` creates a run with **no** `TRIGGERED_BY` edge and an
`AT_STEP` on the start step; 2. the existing triggered path still creates `TRIGGERED_BY` (pin);
3. `submit_workflow_input` on a `waiting` run merges flat into `ctx`, persists it **in the same query
as the resume flip** (D-F — assert both the new `ctx` and `status` from one call), and drives;
4. on a `running`/`done` run ⇒ `WorkflowRunNotWaitingError` → **409**; 5. unknown run ⇒
`WorkflowRunNotFoundError` → **404** *(new handler)*; 6. reserved key (`threadId`, `error`) on
**`…/input`** ⇒ 400 and nothing written; **6b. reserved key on the *start* body's `ctx` ⇒ 400, no run
started, asserted at the *service* layer as well as through REST (M-2/F-6); 6c. the F-6 scenario
end-to-end: a process run started with a legitimate ctx parks with `waitingThreadId == ''` and a chat
message posted in an unrelated thread does not resume it;**
7. oversized input rejected by the schema (422) **and an oversized *merged* ctx rejected by the
service (400) — the m-5 layer split, one case each**;
8. `maxSteps` from the start body is honoured and bounded (1…50);
9. `find_waiting_run_for_thread(thread_id="")` ⇒ `None` without touching the graph (F-5);
10. REST round-trip: `POST /workflow-runs` → 201, `POST …/input` → 200 with the new status;
**11. D-G: a def whose next step is `type:'tool'` drives to a fault ⇒ the endpoint returns
200 (input) / 201 (start) with `{"status":"failed","error":…}`, **not** 500; the envelope's status is
asserted to come from the post-fault `get_run` (not a literal), and `get_run` confirms the run is
`failed` in the graph. **11b. m-12 zombie case:** with `get_run` stubbed/forced to report `running`
after the fault, the service **re-raises** (500) instead of returning a success envelope; 12. `WorkflowConfigError` (malformed guard at publish) → 400
*(new handler)*; 13. `WorkflowEngineDisabledError` (executor unwired) → 503 *(new handler, folds
OQ-1)*; 14. D-H: an undeclared input key and a value outside `config.expects` each ⇒ 400 with
`stepCount` **unchanged** — the "mistakes are free" property asserted, not assumed.**

**U4 · acceptance, offline (`test_process_flow.py`)** — the three §4.3 paths end-to-end through the
service layer, asserting at each stop: run `status`, the `AT_STEP` step key, the `awaiting` payload
on the newest `StepRun`, and — at the end — the full `NEXT`-ordered step-run trail (the audit proof),
the terminal step reached (`activate` vs `rejected`), `endedAt` stamped, `AT_STEP` cleared, and
`stepCount` matching the §4.3 table. Plus a **budget** case: a run started with `maxSteps=2` fails
with the step-budget note (F-4 made visible), **and a "typos are free" case: three rejected
submissions in a row leave `stepCount` untouched and the run still `waiting` (D-H).**

**Baseline movement — the honest done-condition (gate B-1/B-2 and §5.3 of the review).**
The first draft said "the existing 350 stay green"; that is **not** achievable and never was, because
the two new publish/dispatch invariants retro-invalidate named fixtures. The done-condition is:

> **350 modified-in-place (the named fixture edits in U2's file scope — `test_executor.py:345`;
> `test_api.py:418`; `test_services.py:719/804/808/818/822/832/836/846/851/861/866` — and *no
> others*) + N new. The total count rises, and no existing test was silently weakened.**
>
> *(Line numbers for the five `pytest.raises` tests are approximate — n-6; locate them by test name.
> The `test_executor.py:345`, `test_api.py:418` and `test_services.py:719/808/822/836/851/866`
> citations are exact.)*

Two concrete checks the reviewer can run, both cheap:
- `git diff --stat server/tests/` at the end of U2 must show exactly the files above, and
  `git diff server/tests/` must contain no deletion of an assertion;
- the five tightened `pytest.raises` tests must each carry a `match=` (B-2b), so a future invariant
  cannot make them vacuous again.

`395–405` remains a rough **estimate** of the final count, not a target to hit; `test_queries.sh`
241 → the U0-pinned number is a hard done-condition. If the pytest count does not rise, the unit is
not done.

---

## 8. Risks & open questions

| # | Risk | Severity | Mitigation / posture |
|---|---|---|---|
| **R-1** | **(REWRITTEN per D-F/M-1) Residual stale-read on the ctx *merge*.** The read that precedes the merge is not transactional: two submitters can merge onto the same base, and the CAS loser's input is not applied | Low (was Medium) | The write itself is atomic with the resume CAS (§12.13), so the winner's ctx is the ctx that drove the run — a **wrong branch is structurally impossible**, and no key an earlier step branched on can be erased. The loser is told (**409**), so nothing is lost silently. Documented in `submit_workflow_input`'s docstring and **DESIGN §6.2 (U3)**. The full fix (a `ctxVersion` counter + CAS on it) is a follow-up, deliberately not built |
| **R-2** | **Step budget vs. park/resume cycles** (F-4) | Low (was Medium) | D-H: invalid input is rejected **before** the merge and costs no step; `access-request@v1` declares `maxSteps = 24` with the arithmetic documented in §4.1; U3 exposes a bounded `maxSteps` on the start endpoint; U4 pins the §4.3 count (8) and adds both a budget-exhaustion test and a "typos are free" test |
| **R-3** | **Raising on `prompt`/`tool`/`message` is a behaviour change** — a def using them now `fail_run`s where it previously "succeeded" doing nothing | Low **in production**, **real in the test estate** | Only `triage@v1` exists in production and it is all `agent`. The *test* estate is where it bites: F-3 case 2's `type:'task'` fixture (B-1) — handled by a named, budgeted fixture edit in U2, not discovered mid-implementation. The old behaviour was a silent lie; failing loudly is the point (§2.3's stated intent). Called out in HISTORY |
| **R-4** | **Guard-language lock-in (D-A).** `cmp` is the deterministic branching vocabulary for the life of the engine; published defs are immutable, so a later switch costs a version bump per def | Medium | Adding ops is additive. Depth/width caps and the two-root path whitelist keep the surface closed. The concrete cliff (no path-vs-path form, n-4) is named in D-A's cons so it is chosen with eyes open |
| **R-5** | **`wait` without a scheduler may read as under-delivery** to QA at K-025 | Low | Stated explicitly in DESIGN §6.1 **and** §6.3's K-025 handoff note, including m-7's "mechanically identical to `human`", so K-025 tests the signal semantics that exist rather than a timer that does not |
| **R-6** | **Split-brain on re-seed** (`reference` wiped by pytest/`test_queries.sh`, `ws:<id>` not) | Medium (procedural) | §4.4 order is a done-condition of U4 and U5, not a footnote. First landing is clean (brand-new def); the discipline matters from landing two. Verification is a service-layer one-shot so it cannot silently require a server that is not running (m-8) |
| **R-7** | **`_drive_loop` pressure**, and its mirror: an implementer **freezing** at an edit near it | Medium | §3.1 states the SHA command (re-verified at patch time) and the explicit **box listing the three in-file edits that are outside the locked region** (`_execute_step`, `_select_transition`/`_trace_step`, `resume`). Any need to change `_drive_loop` itself is a stop-and-escalate to teco with justification |
| **R-8** | **Empty `waitingThreadId` collision** (F-5) **and its live reverse** (F-6/M-2) | Low (F-5, latent) / **Medium (F-6, reachable)** | F-5: one-line service short-circuit in U3 + a test. F-6: reserved-key rejection on the **start** ctx, enforced in the service (not only the schema) + tests 6b/6c |
| **R-9** | **(NEW) Fixture edits weaken the regression net.** U2 must touch 13 existing tests | Medium (procedural) | Every edit is named test-by-test in U2's file scope and §7; the five `pytest.raises` tests gain `match=` so they cannot go vacuous; the reviewer's two-command check (§7) makes any unlisted edit visible in the diff |

**Open questions for the coordinator:** *(the three that existed — OQ-A/OQ-B/OQ-C from the gate — are
now settled as **D-F/D-G/D-H**; the plan's own OQ-1 is folded into D-G's handler pass. Nothing below
blocks any unit.)*
- **OQ-2** — MCP parity (`submit_workflow_input` as an MCP tool, D-B/B3). Recommend a follow-up item,
  not this slice. Note that D-H's validation lives in the **service**, so an MCP adapter inherits it
  for free — that was part of why the service is the enforcement layer.

---

## Ready to implement — summary

**Plan:** `falkor-chat/docs/archive/plans/m3-process-flow.md` (this file, **patch v2**).

**Decisions (all settled — do not reopen):** D-A `cmp` comparator, validated at **publish** as well
as at drive time · D-B REST start + input · D-C `wait` = external signal (K-028 filed) · D-D
**`access-request@v1`** (6 steps / **6** transitions / **4** ops; key renamed off the `onboarding`
fixture collision) · D-E implement `human`/`decision`/`wait`, raise on `prompt`/`tool`/`message` ·
**D-F** the ctx write rides **inside** the resume CAS (`resume_run_with_ctx`; `set_run_ctx` dropped) ·
**D-G** a drive fault returns **200/201 + `{"status":"failed","error":…}`**, plus five registered
error handlers (404/409/400/400/503) · **D-H** input is validated against the parked step's declared
`fields`/`signal`/`expects` **before** the merge (invalid input is a free 400), and
`access-request@v1` declares `maxSteps = 24`.

**Unit sequence:** **U0** graph-dba gate (§12.12 `start_run_untriggered` + §12.13
`resume_run_with_ctx`; no DDL, no index, ~nil RAM) → **U1** `cmp` guard + `validate_cmp` +
`render_label` (`guards.py`, parallelizable with U0) → **U2** typed handlers + publish invariants +
cmp trace + the **named fixture edits** (`executor.py`, `services.py`, 3 test files) → **U3**
start-without-trigger + input endpoint + error map (`repository`/`executor.resume`/`services`/
`schemas`/`api`/`app`) → **U4** `access-request@v1` def + seed + offline acceptance → **U5** closeout
docs (BACKLOG K-024 ✅ / K-025 unblocked / K-028 + K-029 filed, HISTORY, DESIGN §6.3).

**graph-dba gate: YES, required before U3** — two queries, **unchanged in count** by the D-F fold.

**Two constraints added at re-gate, aimed straight at U2 and U3:** `_validate_def_spec` must
**normalize string-shaped `config`/`guard`** before applying either new invariant, or the REST front
door either 500s or silently escapes both (M-7, §3.3 box); and D-G must **not** invent a
budget-exhaustion exception (it never raises — m-11) while it **must** re-raise if the post-fault
re-read is not terminal (m-12).

**What an implementer must not get wrong:** `_drive_loop` is locked (SHA `71055f756280`, verify with
the §3.1 command — byte counts in the docs are all wrong), **but the three `executor.py` edits this
plan asks for are outside the locked region and are not an escalation trigger** (§3.1 box); the
baseline done-condition is "**350 modified-in-place, named edits only, + N new**", not "350 stay
green" (§7); every Cypher parameter is bound, never interpolated; and the seed order is always
**pytest → `test_queries.sh` → re-seed → verify** because both suites wipe `reference` but not
`ws:<id>`.

---

## Gate response — analyst review `docs/archive/reviews/m3-process-flow.md` (2026-07-19)

Every finding, what changed, and where. **Adopted: 2 blockers, 6 majors, 10 minors, 4 nits.
Declined with reason: 1 nit (n-3, partially — advisory retained) and 1 minor (m-9, declined with a
filed follow-up).**

| # | Finding | Response | Where |
|---|---|---|---|
| **B-1** | F-3 incomplete — `type:'task'` fixture is load-bearing | **Adopted in full.** F-3 rewritten to name **both** stub cases; the `test_executor.py:345` `task → agent` fixture edit (+ comment) is now explicit U2 scope; U2's done-condition restated | §2 F-3, §3.3, §6 U2, §7 |
| **B-2** | New publish invariant retro-invalidates def fixtures | **Adopted in full.** (a) `waitsForHuman: true` added to the 7 named fixtures; (b) the 5 `pytest.raises` tests gain `match=` so they cannot go vacuous; (c) baseline restated as "350 modified-in-place + N new"; (d) the invariant runs **last** in `_validate_def_spec`, with an ordering pin test (U2 case 8b) | §3.3 box, §6 U2, §7 U2/baseline |
| **M-1** | Fold ctx write into the resume CAS | **Settled as decision D-F (OQ-A).** `set_run_ctx` dropped entirely; `resume_run_with_ctx` (§12.13) *replaces* it — U0's deliverable is changed, not grown. `executor.resume` gains an optional `run_ctx_json`. R-1 rewritten to the residual stale-merge-read only | **D-F**, §3.4, §3.5, §5.2, §6 U0/U3, R-1 |
| **M-2** | Start body's `ctx?` has no reserved-key rule | **Adopted.** New finding **F-6** documents the live path (`executor.py:379` → `waitingThreadId` → `trigger.py:76`); the reserved-key rejection now applies to the **start** ctx too, enforced in `services.start_workflow_run` (not only pydantic); tests 6b/6c added. Sentinel-`waitingThreadId` belt **declined** (one-line reason in §3.4) | §2 F-6, §3.4, §6 U3, §7 U3, R-8 |
| **M-3** | Sync drive + re-raise ⇒ HTTP 500 | **Settled as decision D-G (OQ-B): 200/201 + `{"status":"failed","error":…}`.** Five handlers registered (`WorkflowRunNotFoundError` 404, `WorkflowRunNotWaitingError` 409, `WorkflowInputRejectedError` 400, `WorkflowConfigError` **400** — code decided and justified, `WorkflowEngineDisabledError` **503**, which folds the plan's OQ-1). One §7 test per handler | **D-G**, §3.4, §6 U3, §7 U3 cases 11–13 |
| **M-4** | Validate `cmp` guards at publish | **Adopted.** `guards.validate_cmp(spec)` authored in U1, called from `services._validate_def_spec` in U2 — one validator, two call sites. Fires **only** for `kind ∈ {cmp,all,any,not}`, so the `{"expr":"x>0"}` fixture keeps publishing (pinned by U2 case 11) | §3.2, §6 U1/U2, §7 U1.17 / U2.10–11 |
| **M-5** | Budget burned by invalid input | **Settled as decision D-H (OQ-C): analyst option (b) + (c).** Validation of the submitted input against the parked step's declared `fields`/`signal`/`expects` happens **before** the merge, in the **service layer** (MCP/service callers bypass schemas), using two existing RO reads (`get_run` + `get_snapshot`) — no new query. Invalid input = free 400, zero step cost. `access-request@v1` declares `maxSteps = 24` with the "N mistakes" arithmetic written out. Permissive fallback keeps it non-retroactive | **D-H**, §4.1, §3.4, §7 U3.14 / U4 |
| **M-6** | `cmp` trace needs `_select_transition`, unowned | **Adopted.** `_select_transition` + `_trace_step` added to U2's file scope; judgment filter widened to the cmp family; label = `parsed.get("text") or guards.render_label(parsed)` (no leading `" -> "`). **A dedicated box in §3.1 states that this, `_execute_step`, and `executor.resume` are all OUTSIDE the SHA-locked region** so no implementer freezes at the stop-and-escalate rule | §3.1 box, §3.2, §6 U2, §7 U2.9, R-7 |
| **m-1** | D-D recount (7/3 vs 6/4) | **Adopted.** D-D now reads "six steps, **six** transitions … **four** ops"; §4.2 unchanged (it was correct) | D-D, §4.2 |
| **m-2** | Def key `onboarding` collides | **Adopted.** Renamed to **`access-request@v1`** consistently: D-D, §4 (all subsections), §6 U4, §7, `proof_defs.ACCESS_REQUEST_DEF`, seed-script section, summary | D-D, §4, §6, §7 |
| **m-3** | Blanket done-condition impossible for U0 | **Adopted.** §6's clause is now per-unit: U0 = query suite only, pytest **unchanged at 350**; U1–U4 = pytest rises | §6 preamble, U0 |
| **m-4** | DESIGN §6.2 has no owning unit | **Adopted.** Assigned to **U3**, with the content specified (CAS owns the ctx write; residual R-1 window) | §6 U3 docs |
| **m-5** | ctx size cap can't live in the schema | **Adopted.** Layer split written down: pydantic bounds the **input**; `services.submit_workflow_input` bounds the **merged** ctx. One test each | §3.4 step 4, §6 U3, §7 U3.7 |
| **m-6** | F-1 incomplete — the ordering half is wrong too | **Adopted.** F-1 now names both halves; U2's DESIGN §6.1 edit must fix the sort-key sentence (`(guard == "", order)`, conditional-first, `order` as intra-class tie-break) | §2 F-1, §6 U2 docs |
| **m-7** | `wait` ≡ `human` under C1 | **Adopted.** Stated plainly in D-C, in §3.3's table, in U2's DESIGN §6.1 edit, and repeated in U5's §6.3 K-025 handoff note | D-C, §3.3, §6 U2/U5, R-5 |
| **m-8** | U4 verification needs a running server | **Adopted.** Verification is a **service-layer one-shot** by default (`get_def`/`list_snapshots`); `start_server.sh` is named explicitly and only for the REST check; the §4.4 order updated | §4.4, §6 U4, R-6 |
| **m-9** | Two def-source conventions in one script | **Declined for this slice, with reason + a filed follow-up.** Moving the live, published `triage@v1` literal during a split-brain-prone slice risks a byte-diff that `MERGE … ON CREATE SET` swallows silently, for zero proof value. A paragraph now says *why* triage stays put, and why proof def content belongs in the shipped package; **U5 files K-029** to converge | §4.4, §6 U5 |
| **m-10** | `ne`-missing vs `not(eq)`-missing breaks De Morgan | **Adopted.** Required in the `guards.py` docstring (U1 done-condition) and pinned by a dedicated **contrast pair** test (§7 U1 case 3b) | §3.2, §6 U1, §7 U1.3b |
| **n-1** | `decision` envelope key collides with `ctx.decision` | **Adopted.** Renamed to `node` | §3.3, §7 U2.4 |
| **n-2** | `on="await"` invents a value for a vestigial field | **Adopted.** `human`/`wait` return `on="done"` | §3.3 |
| **n-3** | Symmetric invariant for conditional-only `decision` steps | **Partially adopted / invariant declined.** Enforcing it would retro-reject `test_services.py:719`'s `review` step — fixture churn beyond this slice's budget. Instead: an explicit **warning bullet in `falkor-chat/AGENTS.md`** (U2) and a **K-029 line** proposing the hardening | §3.3 (declined-with-reason note), §6 U2/U5 |
| **n-4** | D-A cons should name the path-vs-path limit | **Adopted.** A1's cons now name it as the first cliff a reader hits (`ctx.approver != ctx.requester` is not expressible) | D-A table, R-4 |
| **n-5** | `§12.1b` breaks the numeric heading convention | **Adopted.** The two queries are **§12.12** and **§12.13**, cross-referencing §12.1 and §12.4 | §5, §6 U0 |

### Re-gate (v2) response — analyst "Re-gate (v2)" section, 2026-07-20 · verdict **approve with suggestions**

All four new findings adopted; none required a design change. U0/U1 were unblocked and dispatched
before this patch, and **neither is affected**: M-7's normalization lives at the
`_validate_def_spec` call site, so `guards.validate_cmp` still takes an already-parsed dict (U1's
deliverable is unchanged), and nothing here touches §12.12/§12.13 (U0).

| # | Finding | Response | Where |
|---|---|---|---|
| **M-7** | Publish invariants assume dict `config`/`guard`; REST delivers strings | **Adopted as written.** A **normalization box** now opens §3.3: `json.loads` string-shaped values first; a non-dict guard ⇒ *no declaration* (so `{"expr":"x>0"}` / `"raw-string"` keep publishing); a `human`/`wait` step whose config does not normalize to a dict ⇒ `WorkflowDefSpecError`. Both failure directions (500 on `AttributeError`; REST silently escaping both invariants) named explicitly. `validate_cmp` keeps its dict-only signature — **U1 untouched**. The `test_api.py:418` edit is restated in **string** form (`"config": "{}"` → `'{"waitsForHuman": true}'`). New §7 U2 case 12 is a four-way shape matrix pinning that the REST front door cannot escape either invariant | §3.3 box, §3.2 (last bullet), §6 U2 file scope, §7 U2.11–12 |
| **m-11** | D-G names a non-existent `StepBudgetExceededError` | **Adopted.** Removed from the catch list, and replaced by an explicit clause: budget exhaustion **never raises** (`_fail_budget` `executor.py:663` stamps `fail_run` and *returns* `"failed"` via OUTCOME A/C `:370`/`:387`), reaches the same envelope through the normal return path, and **U3 must not invent the exception** — that would be an engine behaviour change inside a unit that must not make one | D-G bullet 1 |
| **m-12** | D-G should re-raise when the post-fault re-read is not terminal | **Adopted.** New clause: re-read via `get_run`; if the status is not in `{failed, done, waiting}`, **re-raise — a 500 is correct there**, because reporting a zombie `running` run as a 200/201 success is the worst outcome for §6.3's audit property. §7 U3.11 now asserts the envelope's status came from `get_run`, and **new case 11b** pins the zombie path | D-G bullet 2, §7 U3.11/11b |
| **n-6** | Drifted line citations (inherited from v1) | **Adopted.** The five `pytest.raises` tests are now cited at `test_services.py:804/818/832/846/861` and the `{"expr":"x>0"}` fixture at `:726`, everywhere they appear (§3.2, §3.3 box, §6 U2, §7 baseline list) — each with an explicit **"locate by test name, not line"** instruction, since these have drifted once already. The exact citations (`test_executor.py:345`, `test_api.py:418`, `test_services.py:719/808/822/836/851/866`) are marked as exact | §3.2, §3.3 box, §6 U2, §7 |

**Also fixed while patching (not a gate finding):** §4.1's `submit` step declared
`fields: ["role","laptop"]` while §4.3 submitted `{"request": {...}}` — an internal inconsistency
that D-H's key-level validation would have turned into a bug. `fields` is now `["request"]`, with the
"top-level keys only, no deep schema" rule stated.

**Review §5's "could not verify statically" items** are unchanged in status and remain U0's job (both
queries' PROFILE plans and zero-row contracts — now including §12.13's *nothing-written* contract,
which D-F's whole argument rests on), plus the real suite counts and the RAM delta at integration.
