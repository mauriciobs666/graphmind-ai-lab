# M3 — `kind:'process'` proof flow (K-024, remaining half) — implementation plan

> **Status:** proposed (architect plan, 2026-07-19). Planning-only artifact — no code/DDL/doc
> changed by this plan itself.
> **Closes:** the open half of **K-024** (`docs/BACKLOG.md`) — the LLM-free business-process proof
> over `human` / `decision` / `wait` steps, i.e. the **DESIGN §6.3** "coordination is workflow, not a
> separate primitive" proof. Unblocks **K-025** (qa-engineer acceptance → M3 ✅).
> **Builds on:** `docs/plans/m3-executor.md` (§2.1 loop, §2.3 deterministic-node seam, §2.4
> suspend/resume, §2.5 guard dispatch), delivered as K-022 Landings 1+2.
> **Coordination ledger:** `docs/plans/m3-process-flow-coordination.md` (teco).
> **Explicitly NOT in scope:** **K-027** (live-triage reliability, carried minors m-1…m-3, nits
> n-1…n-3). Nothing from K-027 may be folded into a unit here.
> **Baselines to preserve and raise:** server pytest **350 passed / 0 skipped / 1 deselected**
> (network-free); `./scripts/test_queries.sh` **241/241**.

---

## 0. Decisions required (stakeholder calls — settle before U1 starts)

Five decisions belong to the stakeholder, not the architect. Each is presented with options, a
trade-off, and a recommendation. **The rest of this plan is designed against the recommended
option in every case, and every dependent section is flagged `[provisional: D-x]`.** If a decision
lands differently, the affected units change shape; the unit boundaries were drawn so that a
different choice on D-A/D-B/D-C rewrites *one* unit each rather than the whole plan.

### D-A · Deterministic guard language

A `decision` node branches on data. `guards.evaluate_guard` today resolves only `""` (unconditional)
and `{"kind":"llm"}`; every other `kind` raises `NotImplementedError` (the "M7 seam",
`guards.py:144`). `TRANSITION.guard` is an opaque set-on-create string — **no schema change is
needed for any option below**.

| # | Option | What a guard looks like | Pros | Cons |
|---|---|---|---|---|
| **A0** | **No guard language — match on `on`** | transition fires when `TRANSITION.on == StepResult.on` | zero new vocabulary; uses the two fields that exist and are *currently dead* (see Finding F-1) | pushes all branch logic into the step handler, which then needs its own config-driven comparison — the problem reappears one layer down; adds a second branching mechanism next to guards |
| **A1** ⭐ | **Structured comparator, data-only** | `{"kind":"cmp","path":"ctx.decision","op":"eq","value":"approve"}` + `{"kind":"all"\|"any"\|"not","of":[…]}` | **no parser, no `eval`, no dependency** — it is already-parsed JSON; every op is a whitelisted Python callable; trivially unit-testable; depth/width caps make DoS structural; readable in the graph | verbose to author by hand; not expressive (no arithmetic, no string ops beyond `contains`) |
| **A2** | **Tiny whitelisted DSL string** | `{"kind":"expr","text":"ctx.decision == 'approve'"}` | pleasant to author | needs a hand-written tokenizer+parser (~200 lines to get right), which is a new attack/bug surface; every future op is a grammar change; reopens DESIGN §13 as "we built a language after all" |
| **A3** | **Expression library** (`simpleeval`, `json-logic`) | library-defined | expressive, off-the-shelf | a new runtime dependency on the guarded execution path; sandbox-escape history in this class of library; contradicts DESIGN §13's *resolved* "no expression library is built" |

**Recommendation: A1**, with the kind named **`cmp`** (not `expr`). Naming it `cmp` keeps
`kind:'expr'` reserved and still raising, so DESIGN §13's resolution ("no expression library") stays
literally true and the door to A2/A3 stays visibly shut. A1 is the smallest thing that makes a
`decision` node real, and it is 100% deterministic and offline — exactly what K-024 asks for.
**What it locks in:** the shape of deterministic branching for the life of the engine. Adding ops
later is additive (a dict entry); switching to A2/A3 later would be a def-content migration, and
published defs are immutable ⇒ a `key`/`version` bump for every def. That cost is real but small
today (one def).

### D-B · Human-input channel for a `process` flow

A parked `human` step must be advanceable by someone who is not in a chat thread. The existing
resume path (`trigger.py` step 2) is chat-message-driven.

| # | Option | Mechanics | Pros | Cons |
|---|---|---|---|---|
| **B1** ⭐ | **REST `POST /workflow-runs/{runId}/input`** (+ `POST /workflow-runs` to start) | body merges into the run `ctx`, then `executor.resume` (existing CAS) | reuses `waitsForHuman` + `suspend_run`/`resume_run` unchanged; input is **durable and auditable** in `ctx`; drives synchronously ⇒ deterministic offline tests; a UI/CLI/curl can drive it | needs **two new repository queries** ⇒ a graph-dba gate (see §5); read-modify-write on `ctx` is last-write-wins |
| **B2** | **Reuse the chat resume path** | the approver posts a message in a thread bound to the run | zero new API, zero new Cypher | forces a `Thread` + `Message` to exist for a *non-chat* process — it hollows out the very §6.3 claim this slice is meant to prove; input arrives as prose, so a deterministic guard has nothing structured to read (it would need an LLM to parse ⇒ violates "offline") |
| **B3** | **MCP tool only** | `submit_workflow_input` MCP tool | fits the agent-facing surface | MCP is the *agent* front door; a human approval belongs on the human front door. Also still needs the same service+repository work |

**Recommendation: B1.** B3 is a cheap follow-up once B1's service method exists (`mcp.py` is a thin
adapter — one tool, ~15 lines) and is listed as a non-blocking follow-up, not part of this slice.
Note B1 also covers **starting** a run without a chat message: `repository.start_run` today
*requires* a trigger `Message` to match (`repository.py:1067`, QUERIES §12.1) — see Finding F-2.

### D-C · `wait` step semantics

| # | Option | What it means | Verdict |
|---|---|---|---|
| **C1** ⭐ | **Wait for an external signal** | the step parks; an external actor posts the signal through the same D-B input endpoint; a deterministic guard on the signalled ctx key advances it | Zero new machinery — reuses the exact suspend/resume mechanic. Honest about what the system can do |
| **C2** | **Timer / deadline that fires by itself** | the run wakes at `wakeAt` | **This system has no scheduler.** `BackgroundTasks` are request-scoped; there is no periodic worker, no leader election, no due-run sweep. C2 needs a new long-lived component, a `WorkflowRun.wakeAt` **index** (new RAM line, rule 6), crash/duplicate-fire semantics, and ops. That is its own milestone item, not a corner of a proof flow |
| **C3** | **Deadline-as-data, checked on the next drive** | the step stamps a deadline; a guard compares it against `now` | Cheap, but it is a *lie by omission*: nothing fires, so the deadline only takes effect when something else pokes the run. It also makes guard evaluation clock-dependent, which costs the "fully deterministic tests" property |

**Recommendation: C1**, and **say plainly in DESIGN §6.1 that `wait` is signal-driven, not
timer-driven, because there is no scheduler** — with C2 filed as a new backlog item (proposed
**K-028 — workflow timers / scheduled wakeups**) rather than silently implied by the step name.
C1 proves §6.3 completely: a process that parks on an external system and resumes on its callback
*is* coordination-as-workflow. A `now` term can be added to the D-A guard namespace later (additive)
if C3 is ever wanted.

### D-D · The proof def

**Recommendation:** `onboarding@v1`, kind `process` — a new-hire access-request flow. Six steps,
seven transitions, exercising `human` ×2, `decision` ×3, `wait` ×1, plus three `cmp` ops
(`exists`, `in`, `eq`), a conditional-beats-unconditional ordering case, and a two-outcome branch.
Full spec in §4. Alternatives considered: a pure two-step approve/reject (too thin to prove
branching or `wait`), a purchase-order flow (identical shape, no extra proof value).
**Def identity: `onboarding` / `v1` — brand new, additive. `triage@v1` is not touched.**

### D-E · Scope boundary vs. the full typed-step library

`m3-executor.md` §2.3 lists `prompt`, `tool`, `message` alongside `decision`/`human`/`wait`.

| # | Option | Verdict |
|---|---|---|
| **E1** ⭐ | Implement `human`/`decision`/`wait`; `prompt`/`tool`/`message` become a **documented raising seam** (`NotImplementedError`, reaching the M-1 fault net ⇒ `fail_run` with a named cause) | Closes gap 1 honestly, keeps the slice tight. Each deferred type has a genuine open design question (`prompt` needs an LLM ⇒ conflicts with "offline"; `tool` needs a permission fence outside an agent loop; `message` needs author/thread resolution for a run with no thread) |
| **E2** | Also implement `message` | Tempting (a "notify" terminal reads better than a `decision` terminal) but it needs a thread, which a non-chat process run does not have, and an author identity decision. Half a slice of design for cosmetics |
| **E3** | Implement all six | Speculative code with no consumer; the exact thing §2.3 warned against |

**Recommendation: E1.** Note the deliberate consequence: replacing today's silent no-op with a raise
is a **behaviour change** — see Finding F-3 and Risk R-3.

---

## 1. Goal & scope

**Goal.** Deliver an LLM-free, deterministic, offline `kind:'process'` workflow that publishes,
materializes, starts, parks on human input, branches on data, parks on an external signal, and runs
to a terminal `done` — proving DESIGN §6.3 with a real execution trace in the graph.

**In scope**
- A deterministic transition-guard kind (`cmp`) in `guards.py` [provisional: D-A].
- Real `human` / `decision` / `wait` step handlers in `executor._execute_step`; an explicit raising
  seam for `prompt` / `tool` / `message` [provisional: D-E].
- A run-start path that does **not** require a chat trigger message, and a human/external input
  path that merges into the run `ctx` and resumes the run [provisional: D-B].
- The `onboarding@v1` proof def, seeded additively; an offline end-to-end acceptance test
  [provisional: D-D].
- The doc updates each of those invalidates (assigned per unit, §6).

**Out of scope (do NOT build here)**
- K-027 in its entirety (live-triage reliability, judge calibration, terminal-node contract).
- Any change to `triage@v1`'s content, to the LLM guard judge, or to the agent-node loop.
- A scheduler / timer wakeups [D-C ⇒ proposed K-028].
- `prompt` / `tool` / `message` handlers [D-E].
- A web UI for approvals. The REST endpoints are the surface; `web/` is untouched.
- Any change to `_drive_loop` (see §3.1 — the design is built specifically to avoid it).

---

## 2. Context & findings (verified against the tree at `4f69a16`)

### Verified state

- `executor._execute_step` (`server/falkorchat/executor.py:396`) dispatches `type == 'agent'` **with
  a wired LLM** to `_run_agent_node`; **everything else returns `StepResult(output="", on="done")`**.
- `guards.evaluate_guard` (`server/falkorchat/guards.py:97`): `""`/`None` → unconditional true;
  `{"kind":"llm"}` → injected judge; **anything else → `NotImplementedError`** (line 144).
- `_select_transition` (`executor.py:613`) sorts `(guard == "", order)` — conditional guards are
  evaluated before the unconditional default; first firing wins.
- `_drive_loop` (`executor.py:333–392`) OUTCOME B keys **only** on `config.get("waitsForHuman")`.
  It re-reads nothing mid-loop: `run_ctx` is loaded once at entry from `run["ctx"]`, and
  `executor.resume` re-reads the run via `repo.get_run` after the CAS ⇒ **a ctx written to the graph
  before `resume()` is visible to the resumed drive.** This is the hinge the whole D-B design hangs on.
- `repository.suspend_run` / `resume_run` (`repository.py:1151`/`1169`) are guarded single-query CAS
  flips; `suspend_run` denorms `waitingThreadId`.
- `services.start_workflow_run` (`services.py:578`) mints the run id + clock, resolves the trigger
  message's thread into `ctx = {"threadId": …}`, and passes `executor.step_budget` as `maxSteps`.
- `services._validate_def_spec` (`services.py:427`) already sees each step's `config` dict at publish
  time — a publish-time invariant on `config` costs nothing structurally.
- `STEP_TYPES` (`services.py:46`) already whitelists all seven types; `WORKFLOW_KINDS` already
  includes `process`. **No whitelist change is needed.**
- Test conventions: `server/tests/test_executor.py` drives the real engine against the live `ws:test`
  graph via the `wf_repo` fixture (`server/tests/conftest.py:85`, which wipes `reference` too), with
  stub judges and `id_gen`/`clock` injection. That is the pattern every new test here follows.

### Findings to carry into implementation

- **F-1 · `TRANSITION.on` and `StepResult.on` are vestigial.** Nothing reads either. `_select_transition`
  ignores `on` entirely; every handler returns `on="done"`. DESIGN §6.1 describes `on` as "the
  event/outcome that fires it", which is **not true of the implementation**. This plan does not fix it
  (option A0 rejected), but §6's doc unit must correct the DESIGN wording to "`on` is a descriptive
  label on the transition; firing is decided by the guard alone" rather than leave a false statement
  standing.
- **F-2 · A run cannot be started without a chat `Message`.** `repository.start_run` has
  `MATCH (trigger:Message {msgId: $triggerMsgId})` as a hard anchor; zero rows ⇒ `start_workflow_run`
  raises. A business process started from a form/API has no message. Hence the second start query in
  U0.
- **F-3 · The current silent no-op is load-bearing for existing tests.** `test_executor.py` builds
  `type:'agent'` steps and constructs the executor with `llm=None`, relying on the fall-through stub
  to drive the loop offline. **The `agent`-without-LLM stub path must be preserved verbatim**; only
  the *other* types change behaviour. Getting this wrong breaks a large share of the 350 baseline.
- **F-4 · Park/resume cycles consume the step budget.** Every park records a `StepRun`
  (advance-to-self) and bumps `stepCount`; `maxSteps` defaults to `executor.step_budget` = 12, and
  there is **no per-def budget** — `start_workflow_run` always passes the executor's global default.
  The §4 def costs 8 steps (traced in §4.3), which fits, but a longer process would silently hit the
  budget and `fail_run`. U3 therefore accepts an optional bounded `maxSteps` on the start endpoint.
- **F-5 · `find_waiting_run_for_thread` with an empty `threadId` would match parked process runs.**
  Process runs park with `waitingThreadId = ''` (there is no thread). No caller passes `''` today
  (`trigger.py` always has a real thread from a posted message), so this is latent, not live. U3 adds
  a one-line defensive short-circuit in the service (falsy `thread_id` → `None`) rather than changing
  Cypher.

---

## 3. Design & rationale

### 3.1 The central design choice: park-and-branch, so `_drive_loop` is never touched

The locked `_drive_loop` already implements *exactly* the mechanic a business process needs:

> execute step → evaluate outgoing guards → if none fires **and** the step declares
> `waitsForHuman` → `suspend_run` and return `waiting`.

So a `human` step is expressed as: **a step whose outgoing guards read data that is not in `ctx`
yet.** First pass: the guards are false (the key is missing) → the step parks. Input arrives → it is
merged into the run's `ctx` in the graph → `resume()` re-reads the run → the same step re-executes →
now the guard is true → it advances. No new outcome, no new state, no scheduler, no change to the
loop.

That is why D-A (a guard that can read `ctx`) and D-B (a way to write `ctx`) are the *only* two
genuinely new capabilities in this slice. Everything else is assembly.

**`_drive_loop` lock — verification procedure for every unit's done-condition.** Verify by SHA
**only** (`71055f756280`); every byte count in the docs is wrong (miscopied three ways — see
`m3-executor-coordination.md`). Line-number-based extraction is brittle; use this
line-number-independent form, re-verified by the architect at plan time:

```bash
awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py \
  | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12
# => 71055f756280
```

If any unit finds it *must* change `_drive_loop`, that is a stop-and-escalate finding for teco with
written justification — not a silent edit.

### 3.2 Deterministic guards — the `cmp` kind [provisional: D-A]

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
  `WorkflowConfigError` (loud, named — reaches the M-1 fault net and `fail_run`s with a readable
  cause, exactly like the missing-judge case at `guards.py:126`).
- **Missing path ⇒ `False` for every op except `exists` (which returns `False`) and `ne`
  (also `False`).** A missing value never fires a transition. This "bias to not-fire" mirrors the LLM
  guard's bias-to-suspend: the safe direction is to park, because a parked run is unblockable and a
  wrongly-advanced run is not.
- **Type discipline:** no coercion. `lt/le/gt/ge` on non-comparable or mismatched types ⇒ `False`
  (never a `TypeError` that would fail the run). `eq`/`ne` use Python `==` on the JSON-native types.
- **Structural DoS caps (rule 6):** max nesting depth **5**, max total nodes **32**, max `of` width
  **8**; exceeding any ⇒ `WorkflowConfigError` at evaluation. Guards are already capped at
  `MAX_CONFIG_LEN` (8000) at the API boundary.
- **`GuardVerdict.rationale`** is filled with a compact rendering (`"ctx.decision eq 'approve' →
  true"`), so a debug run's `guard_judgment` trace records deterministic branches the same way it
  records LLM ones.
- **`kind:'expr'` keeps raising `NotImplementedError`** — the "we did not build a language" seam
  stays visibly intact.

*Rejected:* putting the comparator in `executor._select_transition`. Guards belong in `guards.py`;
the executor is dispatch, not policy.

### 3.3 Typed step handlers [provisional: D-E]

`_execute_step` becomes an explicit dispatch table. Handlers are **pure and side-effect-free** —
their entire job is to produce an auditable `StepResult` describing what the step is/was waiting for.
All branching stays in the guards.

| `Step.type` | Handler | Behaviour |
|---|---|---|
| `agent` + wired LLM | `_run_agent_node` | **unchanged** |
| `agent`, no LLM | (fall-through stub) | **unchanged, deliberately** — F-3, the offline test affordance. Documented in the docstring as such, not as an accident |
| `decision` | `_run_decision_node` | No side effect. Returns `StepResult(output=json({"decision":{"step":<key>}}), on="done", trace=[("node_note","decision node — branching in guards")])`. Its semantics live entirely in its outgoing guards; with **zero** outgoing transitions it is a terminal outcome node |
| `human` | `_run_human_node` | Returns `StepResult(output=json({"awaiting":{"kind":"human","prompt":cfg.prompt,"assignee":cfg.assignee,"fields":cfg.fields}}), on="await")`. The output lands on the `StepRun` ⇒ **`GET /workflow-runs/{id}/step-runs` tells a client exactly what the run is waiting for**, with no new query |
| `wait` | `_run_wait_node` | Same shape with `{"kind":"signal","signal":cfg.signal}` [provisional: D-C — signal, not timer] |
| `prompt` / `tool` / `message` | — | `raise NotImplementedError(f"step type {t!r} is not implemented in this cut (typed-handler seam); see docs/plans/m3-process-flow.md §D-E")` → M-1 fault net → `fail_run` with a readable cause |
| unknown type | — | same `NotImplementedError` path |

**Publish-time invariant (new, in `services._validate_def_spec`):** a step of type `human` or `wait`
**must** declare `config.waitsForHuman: true`, else `WorkflowDefSpecError`. Without it such a step
self-loops against the budget until `fail_run` — a silent, expensive footgun. Rejecting it at
authoring time costs ~6 lines and no runtime path. (Deliberately *not* enforced: "a `decision` step
must have ≥2 outgoing transitions" — the §4 def uses zero-outgoing `decision` nodes as terminals.)

### 3.4 Start + input path [provisional: D-B]

```
POST /workflow-runs                      {defKey, version, ctx?, trace?, maxSteps?}  -> 201 {runId, status, …}
POST /workflow-runs/{runId}/input        {input: {...}}                              -> 200 {runId, status, ctx}
GET  /workflow-runs/{runId}                                                          (exists)
GET  /workflow-runs/{runId}/step-runs                                                (exists — carries `awaiting`)
```

`services.submit_workflow_input(ctx, *, run_id, input)`:
1. `repo.get_run` → `None` ⇒ `WorkflowRunNotFoundError` (404).
2. `run["status"] != "waiting"` ⇒ **new** `WorkflowRunNotWaitingError` (409). A run that is
   `running`/`done`/`failed` has nothing to unblock.
3. Merge `input` **flat** into the deserialized `ctx` (so guard paths read `ctx.decision`, not
   `ctx.input.decision`). **Reserved keys** `threadId` and `error` are rejected (400) — they are
   engine-owned (`services.start_workflow_run` and `_fail_with_note`).
4. `repo.set_run_ctx(ws, run_id=…, ctx=…)` — guarded on `status = 'waiting'`; `None` ⇒ 409 (lost a race).
5. `executor.resume(ctx, run_id=…)` → status. `None` (CAS lost) ⇒ 409.
6. Return `{runId, status, ctx}`.

**Synchronously**, not on `BackgroundTasks`: a process drive is pure graph work with no LLM, so it
is fast and — crucially — **deterministically testable**. A future LLM-bearing process def would want
the background path; noted, not built.

*Rejected:* passing the input as an in-memory overlay to `executor.resume` to avoid the new query.
It would keep the input out of the graph, so the run's own `ctx` would not reflect the decision that
advanced it — destroying the audit property that is the entire point of §6.3 — and a second resume
would lose it.

### 3.5 Concurrency & atomicity posture

- `set_run_ctx` is a **read-modify-write across two queries** (get_run → set_run_ctx). Two concurrent
  submissions on the same parked run are **last-write-wins on `ctx`**, and only one wins the resume
  CAS. This is accepted and documented, not hidden: the parked-approval use case is single-approver,
  and the durable-correctness backstop is that the resume CAS is still single-flight (the run cannot
  be driven twice). A real multi-approver system needs a version counter on `WorkflowRun` and a
  compare-and-set on it — filed as a risk (R-1), not built.
- Everything else reuses the existing per-query-atomic writes unchanged.

---

## 4. The proof def — `onboarding@v1` [provisional: D-D]

`kind: 'process'`, key `onboarding`, version `v1`. **New def — additive; `triage@v1` untouched.**

### 4.1 Steps

| Step | `type` | `config` | Role |
|---|---|---|---|
| **submit** (start) | `human` | `{"waitsForHuman":true,"prompt":"File the access request","fields":["role","laptop"],"assignee":"requester"}` | Parks until the request is filed |
| **route** | `decision` | `{}` | Pure branch: privileged roles need approval, standard hires do not |
| **approval** | `human` | `{"waitsForHuman":true,"prompt":"Approve or reject this access request","fields":["decision"],"assignee":"manager"}` | Parks until a manager decides |
| **provision** | `wait` | `{"waitsForHuman":true,"signal":"provisioned"}` | Parks until the provisioning system signals back |
| **activate** | `decision` | `{}` | **Terminal** — happy-path outcome (no outgoing ⇒ run `done`) |
| **rejected** | `decision` | `{}` | **Terminal** — rejected outcome (no outgoing ⇒ run `done`) |

> A rejected request still ends the run `done`, not `failed`: the *process* completed; the *outcome*
> is the terminal step reached (readable from the `NEXT`-ordered step-run trail). `failed` remains
> reserved for engine faults and budget exhaustion.

### 4.2 Transitions

| # | from → to | `on` | `order` | guard |
|---|---|---|---|---|
| 1 | submit → route | `filed` | 0 | `{"kind":"cmp","path":"ctx.request.role","op":"exists"}` |
| 2 | route → approval | `needs_approval` | 0 | `{"kind":"cmp","path":"ctx.request.role","op":"in","value":["contractor","exec"]}` |
| 3 | route → provision | `auto` | 1 | `""` (unconditional default — fires only if #2 does not) |
| 4 | approval → provision | `approved` | 0 | `{"kind":"cmp","path":"ctx.decision","op":"eq","value":"approve"}` |
| 5 | approval → rejected | `rejected` | 1 | `{"kind":"cmp","path":"ctx.decision","op":"eq","value":"reject"}` |
| 6 | provision → activate | `provisioned` | 0 | `{"kind":"cmp","path":"ctx.provisioned","op":"truthy"}` |

Coverage: `human` ×2, `decision` ×3, `wait` ×1; `cmp` ops `exists` / `in` / `eq` / `truthy`;
conditional-beats-unconditional ordering (#2 vs #3, the existing `_select_transition` rule);
two-way branch (#4/#5) where *neither* firing ⇒ the step re-parks (a garbage `decision` value leaves
the approval parked, which is the correct, unblockable behaviour).

### 4.3 The privileged-role happy path, step by step (also the acceptance-test script)

| # | Action | Drive | Step executed | Guard | Outcome | `stepCount` |
|---|---|---|---|---|---|---|
| 1 | `POST /workflow-runs {onboarding,v1}` | 1 | submit | #1 false (no `ctx.request`) | park → `waiting` | 1 |
| 2 | `POST …/input {"request":{"role":"contractor","laptop":true}}` | 2 | submit | #1 true | advance → route | 2 |
| 3 | | 2 | route | #2 true (`contractor`) | advance → approval | 3 |
| 4 | | 2 | approval | #4/#5 false | park → `waiting` | 4 |
| 5 | `POST …/input {"decision":"approve"}` | 3 | approval | #4 true | advance → provision | 5 |
| 6 | | 3 | provision | #6 false | park → `waiting` | 6 |
| 7 | `POST …/input {"provisioned":true}` | 4 | provision | #6 true | advance → activate | 7 |
| 8 | | 4 | activate | no outgoing | `complete_run` → **`done`** | 8 |

8 of a 12-step budget (F-4). The standard-hire path (`role:"engineer"`) costs 6; the rejected path 6.

### 4.4 Where the def lives — single source of truth

Put the spec in a new module **`server/falkorchat/proof_defs.py`** (`ONBOARDING_DEF = {...}`),
imported by **both** `scripts/seed_workflows.sh`'s Python one-shot **and** the acceptance test. This
is a direct response to the K-022 U14 lesson: `test_workflow_live.py` had to shell out to the seed
script precisely because a copied def spec would drift. An importable constant gets the same
no-drift property with no subprocess and no network.

`scripts/seed_workflows.sh` is **extended** (not replaced): it loops over `[triage-spec,
ONBOARDING_DEF]`, publishing + materializing each, printing `created` / `already present — no-op`
per def as it does today. Additive and idempotent; `start_server.sh` needs no change.

**Seeding / verification order (the immutability + split-brain rule, AGENTS.md):**

```
./scripts/bootstrap_schema.sh <ws>   →  ./scripts/seed_demo.sh <ws>  →  server pytest
   →  ./scripts/test_queries.sh      →  ./scripts/seed_workflows.sh <ws>   →  verify
```

`pytest` (via the `wf_repo` fixture) and `test_queries.sh` both wipe `reference`; the `ws:<id>`
snapshot survives ⇒ **always re-seed last, then verify**. Because `onboarding@v1` is brand new there
is no stale snapshot to split-brain against on first landing — but from the second landing onward
the rule bites, and any *edit* to the def content requires a `key`/`version` bump (published defs are
`MERGE … ON CREATE SET`, i.e. **create-only**: re-seeding an edited def is a silent no-op).

**No new config var.** `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` stay `triage`/`v1` — the
process def is started over REST, not by `@mention`, so nothing in `config.py`, `.env.example`, or
`start_server.sh` changes. (The seed script may accept `FALKORCHAT_PROCESS_DEF_KEY`/`_VERSION`
locally, documented in its header only. This deliberately avoids the documented trap that
`start_server.sh` forwards neither var.)

---

## 5. Graph-dba gate — **REQUIRED** (small, additive, zero DDL)

Two new queries. **No new label, no new property on an existing hot node beyond a longer `ctx`
string, no new index, no DDL change ⇒ RAM impact ≈ nil** (rule 6): `ctx` already exists on
`WorkflowRun`; the input merge grows it by tens of bytes per run, and process runs are rare compared
to messages. `bootstrap_schema.sh` is **not** touched.

1. **`start_run_untriggered`** → **QUERIES.md §12.1b** — `start_run` (§12.1) minus the
   `MATCH (trigger:Message …)` anchor and the `TRIGGERED_BY` create; everything else identical.
   *Deliberately a second query, not an `OPTIONAL MATCH`+`FOREACH` conditional* — it follows the
   locked project doctrine of two separate self-contained write paths (the §4 first/subsequent
   precedent) and avoids the documented empty-row-collapse class of bug. The gate PROFILEs it
   (expect a single `Node By Index Scan` on `WorkflowDefSnapshot.key`) and confirms the zero-row
   contract (snapshot has no `START`).
2. **`set_run_ctx`** → **QUERIES.md §12.12** — `MATCH (r:WorkflowRun {runId:$runId}) WHERE r.status
   = 'waiting' SET r.ctx = $ctx RETURN …` (status-row shape, mirroring `suspend_run`). Gate confirms:
   index-anchored on `runId`; the guard makes a non-waiting run a zero-row no-op; parameterised.

Gate deliverables: both queries live-verified + PROFILEd, written into `QUERIES.md` §12, plus
enumerated `test_queries.sh` assertions raising the pinned count from **241** to the gate's new
number (expect ~4–6 new assertions: untriggered start creates the run subgraph **without**
`TRIGGERED_BY`; missing-snapshot ⇒ zero rows; ctx set while waiting; ctx set rejected while running;
index-scan profiles). The gate pins the new count in `QUERIES.md`/`DESIGN §7.1` in the same pass.

---

## 6. Units — sequenced, independently reviewable

Each unit ends with: **suites green** (server pytest count **up** from 350, `test_queries.sh` at its
pinned count), **`_drive_loop` SHA re-verified `71055f756280`** (§3.1 command), and its named docs
updated **in the same change**.

### U0 — graph-dba gate (queries only) · owner `graph-dba`
Per §5. **Done:** §12.1b + §12.12 in `QUERIES.md` with PROFILE notes and the RAM statement;
`test_queries.sh` assertions added and the new count pinned; suite green at the new count; explicit
written confirmation that **no** `bootstrap_schema.sh` / index / DDL change was needed (or, if the
gate disagrees, that finding escalates to teco before U3 starts). **Blocks:** U3.
**Docs:** `QUERIES.md` §12; `DESIGN.md` §7.1 count refresh if the gate changes it.

### U1 — deterministic `cmp` guard · owner `tdd-engineer` [provisional: D-A]
`server/falkorchat/guards.py` only. Add `kind:'cmp'|'all'|'any'|'not'` per §3.2; keep `expr`
raising; keep the `llm` and unconditional branches byte-identical in behaviour.
**Done:** `server/tests/test_guards.py` extended (see §7); no other module touched; `guards.py`
module docstring updated to describe three live branches + the seam.
**Docs:** `DESIGN.md` §6.1 (guard discriminator now `""` | `{kind:'llm'}` | `{kind:'cmp'|all|any|not}`),
`DESIGN.md` §13 (the guard open question: note that the *deterministic* half is now a structured
comparator, explicitly **not** an expression language — the resolution stands).
**Independent of U0** — can run in parallel.

### U2 — typed step handlers + publish invariant · owner `coder` [provisional: D-C, D-E]
`server/falkorchat/executor.py` (`_execute_step` dispatch + three handlers; **`_drive_loop`
untouched**) and `server/falkorchat/services.py` (`_validate_def_spec` human/wait ⇒ `waitsForHuman`).
**Done:** new `server/tests/test_executor_process.py` (§7); the existing 350 stay green — in
particular the `agent`-without-LLM stub path is preserved (F-3) and its docstring now says *why*;
`prompt`/`tool`/`message` raise with a message naming this plan.
**Docs:** `DESIGN.md` §6.1 step-type bullet list — which types the engine executes today, `wait` is
**signal-driven, not timer-driven** (D-C), and the F-1 correction to the `on` wording;
`falkor-chat/AGENTS.md` — a new executor-invariants bullet ("a `human`/`wait` step must declare
`waitsForHuman`, enforced at publish; a `decision` step has no side effect — its semantics are its
outgoing guards; `prompt`/`tool`/`message` raise").
**Depends on:** nothing hard (its tests are more expressive with U1 landed — sequence U1 → U2).

### U3 — start-without-trigger + input endpoint · owner `coder` [provisional: D-B]
`repository.py` (`start_run_untriggered`, `set_run_ctx` — 1:1 with U0's §12.1b/§12.12, plus
`WorkflowRunNotWaitingError`), `services.py` (`start_workflow_run(trigger_msg_id: str | None = None)`
dispatching to the two repository paths — the §4 first/subsequent precedent; `submit_workflow_input`
per §3.4; the F-5 one-line short-circuit in `find_waiting_run_for_thread`), `schemas.py`
(`StartWorkflowRunIn`, `SubmitWorkflowInputIn` — bounded: ≤32 input keys, key ≤ `MAX_KEY_LEN`,
serialized ctx ≤ `MAX_CONFIG_LEN`, `maxSteps` 1…50), `api.py` (two routes), `app.py`
(`WorkflowRunNotWaitingError` → 409 handler alongside the existing §11 handlers).
**Done:** `test_api.py` + `test_services.py` additions (§7); a `503`-free error map
(404 / 400 / 409); no change to the existing `@mention` start path's behaviour.
**Depends on:** U0. **Docs:** `DESIGN.md` §14.4 REST-surface list (add the two routes).

### U4 — the proof def, seed & offline acceptance · owner `coder` [provisional: D-D]
`server/falkorchat/proof_defs.py` (`ONBOARDING_DEF`), `scripts/seed_workflows.sh` (loop over both
defs; header/usage updated).
**Done:** new `server/tests/test_process_flow.py` — the §4.3 happy path plus the standard-hire and
rejected branches, driven **through the service/REST layer** against `ws:test`, fully offline
(no `live` marker); `./scripts/seed_workflows.sh acme` run per the §4.4 order and the snapshot
verified readable (`GET /workflow-defs/onboarding`, `GET /workspaces/acme/snapshots`); a re-run
prints `already present — no-op` for both defs.
**Depends on:** U1, U2, U3.
**Docs:** `falkor-chat/AGENTS.md` key-scripts table — the `seed_workflows.sh` row now seeds **two**
defs (triage@v1 conversational + onboarding@v1 process), with the create-only/split-brain warning
extended to cover both.

### U5 — closeout · owner `teco` (integration)
**Done:** `DESIGN.md` §6.3 gains a pointer that the proof now exists and where it lives (def +
test); `docs/BACKLOG.md` — **K-024 → ✅** with a one-paragraph delivery note, **K-025 marked
unblocked**, the milestone-map row updated, and **K-028 (workflow timers/scheduled wakeups) filed
as 🔵 proposed** carrying D-C's reasoning; `docs/HISTORY.md` — a dated entry; final full-suite run in
the §4.4 order with the counts recorded; the coordination ledger's Decisions table filled in with
the settled D-A…D-E.

---

## 7. Test strategy

Altitudes, per unit. Everything below is **offline and deterministic** — no `live` marker, no LM
Studio, no network. Graph-backed tests use the existing `wf_repo`/`conn` fixtures against `ws:test`.

**U1 · unit (`test_guards.py`)** — behaviours to drive red→green, in order:
1. `cmp/eq` true and false on a `ctx` path; 2. dotted path into a nested dict (`ctx.request.role`);
3. missing path ⇒ `False` for each of `eq`, `ne`, `lt`, `contains`, `truthy`, and `exists`;
4. `exists` true on a present-but-falsy value (`0`, `""`, `false`) — the `exists`/`truthy`
distinction; 5. `in` with a list `value`; 6. `contains` on a list and on a string;
7. ordering comparisons on ints and on strings; 8. mismatched types on `lt` ⇒ `False`, **not** a
raised `TypeError`; 9. `output.` root reading the current step's JSON output; bare `output` reading
the raw string; 10. unknown root (`foo.bar`) ⇒ missing ⇒ `False`; 11. `all`/`any`/`not` including
empty `of` (define and assert: `all([]) → True`, `any([]) → False`, `not` with ≠1 child ⇒
`WorkflowConfigError`); 12. depth cap, node cap, width cap each ⇒ `WorkflowConfigError`;
13. unknown `op` ⇒ `WorkflowConfigError`; 14. `kind:'expr'` still ⇒ `NotImplementedError`;
15. `""` and `{"kind":"llm"}` behaviour unchanged (regression pins); 16. `rationale` is populated
and readable.

**U2 · unit/integration (`test_executor_process.py`)** —
1. a `human` step with a false guard parks the run (`status == 'waiting'`, `AT_STEP` still on it);
2. its `StepRun.output` carries the `awaiting` envelope with `prompt`/`fields`/`assignee`;
3. `wait` ditto with the `signal` envelope; 4. a `decision` step advances via the firing guard and
records an output with no side effect; 5. a `decision` step with no outgoing transitions terminates
the run `done`; 6. `prompt`/`tool`/`message` ⇒ the run ends `failed` with a readable `ctx.error` and
the exception re-raised (the M-1 net contract, mirroring
`test_executor.py::test_llm_guard_without_judge_fails_the_run_with_named_error`);
7. **regression pin:** `type:'agent'` with `llm=None` still returns the empty stub result (F-3);
8. `_validate_def_spec` rejects a `human`/`wait` step without `waitsForHuman`
(`WorkflowDefSpecError`, nothing written) and accepts one with it;
9. a debug run records a `guard_judgment`-equivalent trace line for a `cmp` guard.

**U3 · integration (`test_services.py`, `test_api.py`)** —
1. `start_workflow_run(trigger_msg_id=None)` creates a run with **no** `TRIGGERED_BY` edge and an
`AT_STEP` on the start step; 2. the existing triggered path still creates `TRIGGERED_BY` (pin);
3. `submit_workflow_input` on a `waiting` run merges flat into `ctx`, persists it, and resumes;
4. on a `running`/`done` run ⇒ `WorkflowRunNotWaitingError` → 409; 5. unknown run ⇒ 404;
6. reserved key (`threadId`, `error`) ⇒ 400 and **nothing written**; 7. oversized input rejected by
the schema (422); 8. `maxSteps` from the start body is honoured and bounded;
9. `find_waiting_run_for_thread(thread_id="")` ⇒ `None` without touching the graph (F-5);
10. REST round-trip: `POST /workflow-runs` → 201, `POST …/input` → 200 with the new status.

**U4 · acceptance, offline (`test_process_flow.py`)** — the three §4.3 paths end-to-end through the
service layer, asserting at each stop: run `status`, the `AT_STEP` step key, the `awaiting` payload
on the newest `StepRun`, and — at the end — the full `NEXT`-ordered step-run trail (the audit proof),
the terminal step reached (`activate` vs `rejected`), `endedAt` stamped, `AT_STEP` cleared, and
`stepCount` matching the §4.3 table. Plus a **budget** case: a run started with `maxSteps=2` fails
with the step-budget note (F-4 made visible).

**Expected baseline movement:** 350 → roughly **395–405** passed, 0 skipped, 1 deselected.
`test_queries.sh` 241 → the U0-pinned number. Both are done-conditions, not estimates to be
rounded away: if the pytest count does not rise, the unit is not done.

---

## 8. Risks & open questions

| # | Risk | Severity | Mitigation / posture |
|---|---|---|---|
| **R-1** | **`ctx` read-modify-write is last-write-wins.** Two concurrent inputs to one parked run: one `ctx` write is lost, though only one resume wins the CAS | Medium (latent) | Documented in `services.submit_workflow_input`'s docstring and DESIGN §6.2. Single-approver use case today. The real fix (a `ctxVersion` counter + CAS on it) is a follow-up, deliberately not built |
| **R-2** | **Step budget vs. park/resume cycles** (F-4). A longer process silently `fail_run`s at `maxSteps` | Medium | U3 exposes a bounded `maxSteps` on the start endpoint; U4 pins the §4.3 count (8) and adds an explicit budget-exhaustion test so the failure mode is legible, not mysterious |
| **R-3** | **Raising on `prompt`/`tool`/`message` is a behaviour change** — a def using them now `fail_run`s where it previously "succeeded" doing nothing | Low | Only `triage@v1` exists and it is all `agent`. The old behaviour was a silent lie; failing loudly is the point (§2.3's stated intent). Called out in HISTORY |
| **R-4** | **Guard-language lock-in (D-A).** `cmp` is the deterministic branching vocabulary for the life of the engine; published defs are immutable, so a later switch costs a version bump per def | Medium | Adding ops is additive. Depth/width caps and the two-root path whitelist keep the surface closed. Escalated as decision D-A rather than settled quietly |
| **R-5** | **`wait` without a scheduler may read as under-delivery** to QA at K-025 | Low | Stated explicitly in DESIGN §6.1 and in the K-028 backlog item, so K-025 tests the signal semantics that exist rather than a timer that does not |
| **R-6** | **Split-brain on re-seed** (`reference` wiped by pytest/`test_queries.sh`, `ws:<id>` not) | Medium (procedural) | §4.4 order is a done-condition of U4 and U5, not a footnote. First landing is clean (brand-new def); the discipline matters from landing two |
| **R-7** | **`_drive_loop` pressure.** An implementer may find the parking design "obviously" easier with a loop tweak | Medium | §3.1 states the SHA-verification command; it is in every unit's done-condition. Any need to change it is a stop-and-escalate to teco with justification |
| **R-8** | **Empty `waitingThreadId` collision** (F-5) | Low (latent) | One-line service short-circuit in U3 + a test |

**Open questions for the coordinator (not blockers for U0/U1/U2):**
- **OQ-1** — should `POST /workflow-runs` be reachable when `FALKORCHAT_WORKFLOW_ENABLED` is off?
  The executor is only wired when it is on (`app._build_default_app`), so the route would 500 →
  `services._require_executor` already raises a `RuntimeError` with a clear message. Recommend
  mapping that to **503** in the same `app.py` handler pass (U3), rather than adding a config gate on
  the router. Flagged because it is a one-line judgement call inside U3.
- **OQ-2** — MCP parity (`submit_workflow_input` as an MCP tool, D-B/B3). Recommend a follow-up item,
  not this slice.

---

## Ready to implement — summary

**Plan:** `falkor-chat/docs/plans/m3-process-flow.md` (this file).

**Settle first (stakeholder):** D-A deterministic guard = structured `cmp` comparator (no `eval`, no
parser, no dependency); D-B human input = REST `POST /workflow-runs` + `POST
/workflow-runs/{id}/input` merging into run `ctx`; D-C `wait` = **external signal, not a timer**
(this system has no scheduler — file K-028); D-D proof def = `onboarding@v1` (6 steps: `human`×2,
`decision`×3, `wait`×1); D-E scope = implement `human`/`decision`/`wait`, raise on
`prompt`/`tool`/`message`.

**Unit sequence:** **U0** graph-dba gate (two additive queries, no DDL, no index, ~nil RAM) →
**U1** `cmp` guard (`guards.py`, parallelizable with U0) → **U2** typed step handlers + publish
invariant (`executor.py`, `services.py`) → **U3** start-without-trigger + input endpoint
(`repository`/`services`/`schemas`/`api`/`app`) → **U4** `onboarding@v1` def + seed + offline
acceptance → **U5** closeout docs (BACKLOG K-024 ✅ / K-025 unblocked / K-028 filed, HISTORY,
DESIGN §6.3).

**graph-dba gate: YES, required before U3** — but small: `start_run_untriggered` (§12.1b) and
`set_run_ctx` (§12.12). No new label, no new index, **no `bootstrap_schema.sh` change**, RAM impact
≈ nil.

**Risks the coordinator should know:** the design is deliberately built so **`_drive_loop` is never
touched** (verify SHA `71055f756280` with the line-number-independent command in §3.1 — every byte
count in the docs is wrong); making `prompt`/`tool`/`message` raise is an intentional behaviour
change (R-3); the run-`ctx` input merge is last-write-wins (R-1); park/resume cycles consume the
12-step budget and there is no per-def budget (R-2/F-4); and three real findings surfaced during
investigation — **`TRANSITION.on`/`StepResult.on` are vestigial and DESIGN describes them
inaccurately (F-1)**, **a run cannot currently be started without a chat `Message` (F-2)**, and
**the `agent`-without-LLM silent stub is load-bearing for the existing 350-test baseline and must be
preserved (F-3)**.
