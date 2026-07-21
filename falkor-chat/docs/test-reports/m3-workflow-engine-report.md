# Test report — M3 workflow engine (K-025 acceptance pass)

> **Plan:** [`docs/test-plans/m3-workflow-engine.md`](../test-plans/m3-workflow-engine.md) v1.0
> **Executed:** 2026-07-21 · **By:** `qa-engineer` · **Item:** K-025 (the un-run U15)
> **Under test:** commit **`98a3cc8`**, tree clean (no source file changed by this pass)
> **Environment:** FalkorDB `falkordb/falkordb:v4.18.11` (container `falkordb-dev`, already
> running) · LM Studio at `:1234` **reachable** (`qwen/qwen3-4b-2507` chat +
> `text-embedding-qwen3-embedding-0.6b`, live-probed dim **1024**) · `server/.venv`, Python 3.12 ·
> isolated workspace **`ws:qa`** (created, exercised, **deleted**).

---

## 1. Verdict

# ✅ **PASS with parked, model-gated limitations** ⇒ **M3 → ✅**

The M3 workflow engine does what it claims. Every deterministic behaviour on the acceptance
path — **publish → materialize → start → drive → park → resume → terminal**, plus chat linkage,
tracing and the per-node tool fence — was **executed black-box against the running system** and
behaved exactly as specified. All three documented paths of the `access-request@v1` process flow
reproduce the plan's §4.3 step-by-step table **exactly**, including step counts.

- **AC-1 · AC-5 · AC-6 — VERIFIED by execution.** No defects.
- **The whole `access-request@v1` process flow — VERIFIED by execution.** No defects.
- **AC-2b · AC-3 · AC-4 — recorded *model-gated, structurally demonstrated***, per decision
  **D12-B** / **D7**. All three were *observed working* in a live interactive run; AC-4's
  `PRODUCED` assertion then **failed deterministically (2/2)** in `pytest -m live` on the local 4B.
  That is the **already-filed K-027** limitation, **not a new defect and not an M3-green gate**.
- **Zero blocking defects.** Two non-blocking findings, filed as **K-031** (new) and appended to
  **K-027** (existing).

### 1.1 What is "executed" and what is "inferred"

Everything in the results tables below was **executed** and is backed by the command output or
graph read quoted beside it. Nothing in this report is inferred from reading code alone. Two items
are explicitly marked **not executed** (TP-063, TP-072) with the reason.

### 1.2 The D10 caveat

**No verdict line in this report is sourced from the guard calibration**
(`docs/plans/m3-guard-calibration.md`) — this pass ran no calibration arm, reports no
false-advance or advance-recall figure, and makes no claim about judge fitness. The D10 caveat is
therefore not attached to any line here; it remains binding for **K-027** item 3, which owns that
measurement.

---

## 2. Baselines (entry criteria) — MET

| Suite | Command | Result | Expected | |
|---|---|---|---|---|
| server unit/integration | `server/.venv/bin/python -m pytest -q` | **533 passed, 1 deselected** (7.66 s) | 533 / 1 deselected | ✅ |
| query suite | `./scripts/test_queries.sh` | **256/256 passed** | 256/256 | ✅ |
| live (opt-in) | `.venv/bin/python -m pytest -m live -q -s` | **1 failed, 533 deselected** | — | ⚠️ see DEF-K027-A |

Both baselines were **re-run after the pass** and are still green (**533 / 1 deselected**;
**256/256**), the working tree is unchanged apart from the two new documents, and `reference` was
left holding exactly the two production defs.

Sequencing followed the AGENTS.md rule throughout:
`pytest → test_queries.sh → seed_workflows.sh → verify → exercise`. The wipe trap is **real**
and was observed directly (TP-090).

---

## 3. Results

Legend: **PASS** · **FAIL** · **BLOCKED** · **N/E** (not executed) · **MG** (model-gated —
recorded, not a failure).

### 3.1 Baseline & seeding

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-000 | Green baselines at `98a3cc8` | **PASS** | `533 passed, 1 deselected`; `Results: 256/256 passed` |
| TP-001 | Re-seed after the wipes; both defs present | **PASS** | `seed_workflows.sh acme` → `reference def triage@v1 … (created)` + `access-request@v1 … (created)` — i.e. the suites had indeed wiped `reference`; `ws:qa` seeded fresh: `ws:qa snapshot access-request@v1 steps=6 transitions=6 (materialized)` |

### 3.2 Publish & materialize — FR-1/FR-3

| ID | Title | Result | Evidence (verbatim server response) |
|---|---|---|---|
| TP-010 | Def + snapshot readable black-box | **PASS** *(with finding F-1)* | `GET /workflow-defs` → both defs; `GET /workspaces/qa/snapshots` → both snapshots. Structure verified by graph read: `access-request@v1` = **6 steps / 6 transitions**, `START → submit`, guards exactly the plan §4.2 six. **The REST read returns metadata only** — see F-1 / K-031 |
| TP-011 | `human` without `waitsForHuman` rejected | **PASS** | `400 WorkflowDefSpecError: step 'a' of type 'human' must declare config.waitsForHuman: true — a parking step without it self-loops until the step budget fails the run`; post-check `GET` → **404 (absent)** |
| TP-011b | `wait` without `waitsForHuman` rejected | **PASS** | same error for `type 'wait'`; def absent |
| TP-012 | Typo'd `cmp` op rejected | **PASS** | `400 WorkflowConfigError: unknown guard op 'equalz' — allowed ops: contains, eq, exists, ge, gt, in, le, lt, ne, truthy`; def absent |
| TP-013 | Zero-transition def rejected **before** the half-write | **PASS** | `400 WorkflowDefSpecError: a def must declare at least one transition — a zero-transition publish partially writes the def and then fails (see O-6)…`; **def absent from `reference`** — the unrepairable-version hazard is closed |
| TP-014 | Start-step cardinality | **PASS** | 0 starts → `…found 0 ([])`; 2 starts → `…found 2 (['a', 'b'])`; both 400, both absent |
| TP-015 | REST string-typed `config`/`guard` still validated | **PASS** | Every case above was published **through REST**, where `config`/`guard` are typed `str` — the normalize-first rule holds; no escape hatch |
| TP-016 | Unwhitelisted `cmp` path root rejected at publish | **PASS** | `400 WorkflowConfigError: guard path root 'secrets' is not whitelisted (allowed: ctx, output)` |
| — | Dangling transition endpoint | **PASS** | `400 …transition to 'nowhere' is not a declared step key ['a']` |
| — | Duplicate step key | **PASS** | `400 …duplicate step key(s) ['a'] — step keys must be unique within a def` |
| TP-017 | Def immutability is create-only | **PASS** *(documented trap confirmed)* | Re-publishing `qa-imm@v1` with `name:"EDITED NAME"`, `kind:"conversation"` and edited step config returned **`201 {'stepCount': 2, 'transitionCount': 1}`** — a clean success — while the stored def is still `{'name': 'Original name', 'kind': 'process'}`. **A silent no-op that looks like a successful publish.** Exactly as AGENTS.md warns |

### 3.3 The `access-request@v1` process flow — DESIGN §6.3 / plan §4.3

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-020 | **Privileged happy path** (`contractor`) | **PASS** | start → `waiting @submit stepCount=1`; `{"request":{"role":"contractor","laptop":true}}` → `waiting @approval stepCount=4`; `{"decision":"approve"}` → `waiting @provision stepCount=6`; `{"provisioned":true}` → **`done`, stepCount=8, atStepKey=null**. Step-run trail, `NEXT`-ordered: `submit, submit, route, approval, approval, provision, provision, activate` — **byte-for-byte the plan §4.3 table, including the 8-of-24 budget** |
| TP-021 | **Standard-hire auto path** (`engineer`) | **PASS** | file → `waiting @provision stepCount=4` (**no approval step**); `{"provisioned":true}` → `done stepCount=6`. Trail: `submit, submit, route, provision, provision, activate` |
| TP-022 | **Rejected path** | **PASS** | `role:"exec"` → parks at `approval`; `{"decision":"reject"}` → `done stepCount=6`, terminal step **`rejected`**. Run is **`done`, not `failed`** — the process completed, the outcome is the terminal step |
| TP-023 | `wait` parks and does **not** self-advance (D-C) | **PASS** *(specified behaviour)* | At `provision`: `status=waiting stepCount=4`, envelope `{"awaiting":{"kind":"signal","signal":"provisioned"}}`. **After 25 s: `status=waiting atStep=provision stepCount=4` — unchanged.** Correct: signal-driven, no scheduler (K-028) |
| TP-024 | Re-park on a "not yet" signal | **PASS** | `{"provisioned": false}` → still `waiting @provision`, `stepCount 4 → 5` (one step spent, as designed); a later `{"provisioned": true}` → `done stepCount=7`. Trail shows `provision` three times |
| TP-025 | Conditional beats unconditional | **PASS** | `contractor`/`exec` take transition #2 (`needs_approval` → `approval`); `engineer` takes #3 (`auto` → `provision`). Guard trace: `ctx.request.role in ['contractor', 'exec'] → true` |
| TP-026 | Guard reads **nested** ctx | **PASS** | `ctx.request.role` resolved through the whole-object submission: `ctx.request.role exists → false` before filing, `→ true` after |

### 3.4 Input contract & error map

Every rejection was **free**: `stepCount` unchanged and the run still parked, verified before and
after each batch (`before: status=waiting atStep=submit stepCount=1` … `after: status=waiting
atStep=submit stepCount=1`).

| ID | Case | Result | Response |
|---|---|---|---|
| TP-030 | empty input | **PASS** | `400 WorkflowInputRejectedError: no input submitted — an empty input cannot advance a parked run and would consume a step of the run's budget` |
| TP-031 | undeclared key | **PASS** | `400 …key(s) ['nope'] are not declared by the parked step 'submit' (accepts ['request'])` |
| TP-032 | `expects` violation | **PASS** | `400 …value 'maybe' for 'decision' is not one of ['approve', 'reject'] declared by step 'approval'`; also case-sensitive: `'Approve'` rejected. `stepCount` stayed 4 |
| TP-033 | reserved ctx key | **PASS** | `400 …reserved key(s) ['threadId'] may not be set in the input — they are owned by the engine` |
| TP-034 | unknown run | **PASS** | `404 WorkflowRunNotFoundError: workflow run 'does-not-exist-xyz' not found in this workspace` |
| TP-035 | input to a terminal run | **PASS** | `409 WorkflowRunNotWaitingError: workflow run '…' is 'done', not 'waiting' — there is nothing to unblock` |
| TP-036 | start against an unknown def / wrong version | **PASS** | `404 WorkflowRunNotFoundError: cannot start run: snapshot 'nope'@'v9' has no START…` (same for `access-request@v99`) |

### 3.5 Tracing / serviceability — FR-4 / **AC-5** ✅ VERIFIED

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-040 | Debug run records the trail | **PASS** | Same 8-step flow with `trace:true` → **18 `TraceEvent`s**, kinds `{'node_rationale': 8, 'guard_judgment': 8, 'node_note': 2}`, `seq`-ordered within each step-run. Every guard evaluation carries its verdict **and its why**, e.g. `ctx.decision eq 'approve' -> False: ctx.decision eq 'approve' → false` then `-> True: … → true` after the input. The run is **fully reconstructable** from the trace alone |
| TP-041 | **Non-debug run records nothing** | **PASS** | Identical flow with `trace:false` → run `done stepCount=8`, `GET /workflow-runs/{id}/trace` → **0 events**. AC-5's negative half holds |
| TP-042 | Step-run trail reconstructable | **PASS** | `GET …/step-runs` returns one `NEXT`-ordered row per executed step with `status`, `startedAt/endedAt` and an `output` envelope that states *what the run is waiting for* (`{"awaiting":{"assignee":"manager","fields":["decision"],"kind":"human",…}}`) — a client can render the prompt with no extra query |

### 3.6 Node capability fence — FR-6 / **AC-6** ✅ VERIFIED

Driven through the **real** `WorkflowExecutor`, the **real** builtin `ToolRegistry`
(`post_message`, `graphrag_retrieve`, `human_handoff`) and the **real** `ws:qa` graph, with a
scripted LLM standing in for the model. The node granted **only** `post_message`.

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-050 | Only granted tools are **offered** | **PASS** | Offered set on every one of the 4 iterations: `['post_message']` — the two registered-but-ungranted tools were never offered |
| TP-051 | Ungranted call **defensively rejected** | **PASS** | Scripted call to `graphrag_retrieve` → `tool_call: REJECTED ungranted tool 'graphrag_retrieve' (AC-6)`; **no dispatch**, the node survived and continued to the next bounded iteration |
| TP-052 | Malformed granted call refused | **PASS** | `post_message` with no `text` → `tool_call: INVALID post_message: missing ['text']`; no dispatch. The corrected retry **did** dispatch for real: `tool_result: {"posted": "55e0e7f5…", "threadId": "demo-welcome"}` — proving the fence rejects without breaking the capability |

### 3.7 Trigger & chat linkage — FR-7 / **AC-1** ✅ VERIFIED

Served app on `ws:qa`, `FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=1`, dim 1024.

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-060 | `@mention` starts a run linked to the triggering message | **PASS** | Posted `@assistant I need help with a broken deploy on the billing service` → graph read: `(:WorkflowRun {runId:'28549782…', status:'waiting', defKey:'triage'})-[:TRIGGERED_BY]->(:Message {msgId:'24c7024b…'})`, and that message's text is the one posted. Run `ctx = {"threadId":"demo-welcome"}`, `waitingThreadId='demo-welcome'`, parked `atStepKey='intake'`. **Independently re-confirmed** by the live test's own AC-1 assertions, which passed |
| TP-061 | Exactly one handler per message | **PASS** | No direct-responder reply accompanied the workflow: the thread gained exactly one assistant message across the whole triage run, and it is the workflow's own `answer` post. Consistent with the wiring (`_safe_run_workflow` is the only registered handler when `WORKFLOW_ENABLED`) |
| TP-062 | `StepRun-[:PRODUCED]->Message` | **PASS** | `(:WorkflowRun {runId:'28549782…'})-[:HAS_STEP_RUN]->(:StepRun {stepKey:'answer'})-[:PRODUCED]->(:Message {msgId:'64d80b6f…', role:'assistant'})`. Structural only, per **D7** — the answer's *provenance* is not asserted |
| TP-063 | Loop guard (agent-authored message never re-triggers) | **N/E** | Not executable black-box: the REST tenancy seam always posts as `u1`, so an `assistant`-role message cannot be posted through the front door. Covered by the offline estate (`tests/test_trigger.py`), which is green |
| — | Resume **without** a re-`@mention` (trigger step 2) | **PASS** | A plain reply with **no mentions** resumed the parked run: `t+6s running @intake` → `t+12s running @answer stepCount=3` → `t+18s **done** stepCount=4`. The natural-conversation flow works |

### 3.8 Runaway safety & the typed-handler seam

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-070 | Step budget enforced **and recorded** | **PASS** *(nit F-2)* | `maxSteps:2` → after filing the request the run is `failed` with graph ctx `{"error":"step budget exceeded", …}`, `endedAt` set, trail `submit, submit, route` retained. Surfaced as a **200 envelope** `{"status":"failed"}`, **not** a 500 — D-G honoured. Nit: `stepCount=3` for `maxSteps=2` (checked after the step runs) |
| TP-071 | `prompt`/`tool`/`message` raise `NotImplementedError` **by design** | **PASS** *(specified, D-E)* | A def with a `prompt` start step publishes and materializes fine; starting it returns **`201`** with `{"status":"failed","error":"NotImplementedError: step type 'prompt' is not implemented in this cut (typed-handler seam); see docs/plans/m3-process-flow.md §D-E"}` and the run is terminal `failed` in the graph. The seam is loud, typed and correctly recorded |
| TP-072 | Default budget fall-back (12) | **N/E** | Opportunistic P3; the budget mechanism is proven by TP-070 and the default is documented in plan §4.1 |

### 3.9 Live triage — **model-gated** (AC-2b / AC-3 / AC-4)

Two independent live exercises were run. **Both** are reported; neither is cherry-picked.

**(a) Interactive black-box run on `ws:qa`** — `@mention` → intake parks → plain reply →
`intake → research → answer → done` in ~18 s, `stepCount=4`:

| AC | Observed | Recorded as |
|---|---|---|
| AC-2b (fuzzy guard advances) | The guard **fired** on the second intake turn and the run advanced to `research` | **MG — structurally demonstrated** |
| AC-3 (GraphRAG grounding) | The research node ran and **abstained**: step output `no relevant context found`. Correct behaviour on a near-empty `ws:qa` (the DS τ-abstain policy), but it is **not** proof of grounding | **MG — mechanism demonstrated, grounding not proven** |
| AC-4 (answer posted, run done) | Run reached **`done`**, and the `answer` step **did** post a real reply, `PRODUCED`-linked (TP-062) | **MG — structurally demonstrated (D7)** |
| AC-2a (intake posts a clarifying question) | **The first intake turn did not post.** Its step-run `output` is the literal string `post_message({"text": "I'm sorry to hear that you're experiencing a broken deploy…"})` — the model emitted the tool call as *prose*, so it was never dispatched and the question never reached the thread. The run still parked correctly | **K-027 territory** — see DEF-K027-B |

**(b) `pytest -m live -q -s`** (the shipped `tests/test_workflow_live.py`, own throwaway `ws:live`
at the probed dim) — **1 failed, 533 deselected**, run **twice, same failure both times**:

```
AssertionError: the answer node never posted a reply (AC-4); posts came from:
['intake', 'intake', 'intake', 'intake', 'intake', 'intake', 'intake', 'intake']
tests/test_workflow_live.py:354
```

Everything *before* that assertion passed: AC-1 (run started, `TRIGGERED_BY` correct), AC-2
(intake suspended on the vague opener; plain replies resumed it), AC-3 (`research` was visited),
the run reached **`done`**, the visit order was `intake … research … answer`, and the `StepRun`
`NEXT` chain was contiguous. Only the **AC-4 provenance-of-the-answer** assertion failed — the
`answer` node emitted prose instead of calling `post_message`, exactly the **Defect C** mechanism
K-027 item 2 exists for.

| ID | Title | Result |
|---|---|---|
| TP-080 | Live triage e2e | **MG / FAIL-attributed-to-K-027** — deterministic 2/2 on this box; **explicitly not an M3-green gate** (D12-B) |
| TP-081 | Guard evidence tier (D14 RECENT-TURNS) | **PASS (expected)** — the shipped judge runs on the degraded tier by design; nothing here is reported as a defect |

### 3.10 Operational hazards

| ID | Title | Result | Evidence |
|---|---|---|---|
| TP-090 | The `reference`-wipe trap is real | **PASS — warning confirmed accurate** | After `pytest` + `test_queries.sh`, `seed_workflows.sh acme` reported **`(created)`** for *both* `reference` defs (they were gone) while both `ws:acme` **snapshots** reported `already present — no-op`. That is the documented split-brain shape, observed live. Reproduced a second time at the end of the pass |
| TP-091 | Seed idempotence | **PASS** | A clean re-run prints `already present — no-op` for **all four** lines (2 defs × def+snapshot); step counts unchanged (`triage`=3, `access-request`=6) — no duplicates |

---

## 4. Acceptance-criteria roll-up

| AC | Status | Basis |
|---|---|---|
| **AC-1** trigger | ✅ **VERIFIED** | TP-060 (black-box `TRIGGERED_BY` graph read) + the live test's own AC-1 assertions |
| **AC-2** intake loop | **AC-2a mechanism ✅ VERIFIED** (park / resume-without-mention / re-park, TP-020…TP-024 + the live resume); **AC-2b MG** (fuzzy advance observed once live) | D12-B |
| **AC-3** retrieval | **MG — structurally demonstrated** (research node executes, applies the τ-abstain policy; grounding not proven on a near-empty workspace) | D12-B |
| **AC-4** answer / done | **MG — structural only (D7)**: observed complete in one live run (`done` + `PRODUCED` from `answer`); failed 2/2 in `pytest -m live` | D7 / D12-B / K-027 |
| **AC-5** tracing | ✅ **VERIFIED** — both halves (18 events debug, 0 non-debug) | TP-040/041/042 |
| **AC-6** per-node bounds | ✅ **VERIFIED** — offered-set fence *and* defensive dispatch rejection | TP-050/051/052 |
| **FR-1/FR-3** | ✅ verified | TP-010…TP-017, TP-071 |
| **FR-4** | ✅ verified | TP-040…TP-042 |
| **FR-5d** (human hand-off capability) | ✅ verified as a mechanism | the whole §3.3 park/resume estate |
| **FR-7** | ✅ verified | TP-060 |

---

## 5. Defects & findings

No **blocking** defect was found. Nothing found blocks M3 green.

### DEF-1 · minor · **No black-box way to read a def's or snapshot's structure** → filed as **K-031**

`GET /workflow-defs/{key}` and `GET /workspaces/{ws}/snapshots` return **metadata only**
(`{key, version, name, kind}` — `repository.get_def`, QUERIES §11.3). There is **no REST surface**
that returns a def's steps, transitions, guards, `startKey`, or a snapshot's materialized
structure. To verify TP-010 — *"is what I think is published actually published?"* — this pass had
to drop to raw Cypher.

That matters more here than it would elsewhere, because the component's most dangerous documented
trap is exactly this question: published defs are **create-only**, so a re-seed of an edited def is
a **silent no-op** (**confirmed in TP-017: an edited re-publish returned `201` while the stored def
kept its old name, kind and step config**), and `reference` vs `ws:{id}` go stale independently.
The operator has no first-class way to detect the split-brain the docs warn about.

- **Impact:** serviceability / operability. No runtime misbehaviour.
- **Repro:** publish `qa-imm@v1`; re-publish the same key/version with a different `name`;
  `GET /workflow-defs/qa-imm?version=v1` → `201` on the write, old `name` on the read; no endpoint
  reveals the step/guard divergence.
- **Suggested owner:** `architect` (read-surface shape) → `coder`.
- **Also folded into K-031 (nit):** the step budget **overshoots by one** — `maxSteps=2` produced
  `stepCount=3` before failing, because the budget is checked after a step executes. Harmless
  today; it makes `maxSteps` mean "at least N", which a future SLA/costing story would trip over.

### DEF-K027-A · known, filed · **`pytest -m live` is RED on this box (AC-4 answer post)**

`tests/test_workflow_live.py` fails deterministically (2/2 runs) at the AC-4 assertion: the
`answer` node emits a good, grounded answer as **prose with no tool call**, so no
`StepRun-[:PRODUCED]->Message` edge from `answer`. This is **K-027 item 2** verbatim
("terminal-node-must-post engine contract", Defect C) and is **explicitly not an M3-green gate**
(decision D12-B). Recorded here so the red live suite is not mistaken for an unknown regression.

- **Owner:** `architect` (engine-level terminal-node contract) → `coder`/`tdd-engineer`, per K-027.

### DEF-K027-B · new observation, **appended to K-027 (no new number)** · the failure is not terminal-node-specific, and has a second, cheap-to-fix shape

In the interactive run, the **intake** node — not a terminal node — emitted the literal text

```
post_message({"text": "I'm sorry to hear that you're experiencing a broken deploy on the
billing service. Could you please provide more details about the issue? …"})
```

as its step output. The model *did* intend a tool call; it just wrote it in **bare
function-call syntax**. `llm._parse_content_tool_calls` recovers embedded tool calls only from
**JSON** shapes (a `{"tool_calls": […]}` envelope or a JSON object wrapper), so this form is not
recovered and the clarifying question never reached the thread — while the run still parked
correctly and looked healthy from the outside.

Two things this adds to K-027 as written:
1. K-027 item 2 scopes the fix to the **terminal** node. The same failure occurs at a
   **non-terminal** node, where the user-visible symptom is worse (a parked run silently waiting
   on a question nobody was ever shown).
2. It is the **exact structural twin of K-027 item 1** (the judge's bare `json.loads` breaking on
   fenced JSON): a parse layer that is not tolerant of the shapes small local models actually
   emit. Widening `_parse_content_tool_calls` to recognise the bare `name({json})` form is a
   cheap, offline-testable mitigation that would have converted this run.

- **Owner:** `architect` → `coder`, inside **K-027** (items 1 + 2).

### Non-defects deliberately recorded as **specified behaviour**

| Observation | Why it is not a defect |
|---|---|
| A parked `wait` never advanced in 25 s | `wait` is **signal-driven, not timer-driven**, mechanically identical to `human` (D-C). No scheduler exists; timers are **K-028** |
| `prompt` step killed the run with `NotImplementedError` | The typed-handler seam, **D-E**. Correctly surfaced as a `{"status":"failed"}` envelope, not a 500 |
| A `guard_judgment` citing turn text | The shipped judge runs on the degraded **RECENT-TURNS** tier, **D14** |
| Re-publishing an edited def silently no-ops | Create-only `MERGE … ON CREATE SET`, documented in AGENTS.md. *Observability* of it is DEF-1 |
| `{"provisioned": false}` costs a step | Deliberate: `provision` declares no `expects` so "not yet" stays expressible; the 24-step budget has 16 spare re-parks |

---

## 6. Coverage & residual risk

**Covered:** publish validation (9 negative cases, all rejecting *before* any write), materialize,
start (triggered + untriggered), drive, park, resume, re-park, all three documented process paths
with exact step accounting, the full input error map (400/404/409 + free-rejection accounting),
budget exhaustion, the typed-handler seam, both halves of tracing, the tool fence (offer-side and
dispatch-side), `@mention` trigger + `TRIGGERED_BY` + `PRODUCED` linkage, resume-without-mention,
seed idempotence and the `reference`-wipe trap.

**Not covered / residual risk:**

| Gap | Risk carried |
|---|---|
| Concurrency: two simultaneous submitters racing the resume CAS | Covered only by the offline estate's zero-row contract tests; no load harness was built. Low — the CAS is a single query, and the 409-with-nothing-written path was verified single-threaded |
| Live-triage **reliability** across an n≥8 sample, judge calibration | **K-027**, by design (D12-B) |
| Real external MCP server as a node tool (FR-5c end-to-end) | Scoped to a stub client for this cut; the seam is exercised by `tests/test_mcp_client.py` |
| Multi-workspace / auth isolation of runs | M2.5 (K-016…K-018) |
| Performance & RAM of long-running or high-fan-out workflows | No new node type/index landed; not re-profiled |
| TP-063 (agent-authored message must not re-trigger) | Not reachable black-box through the single-tenant REST seam; offline coverage only |

---

## 7. Feedback & recommendations

1. **Ship a def/snapshot structure read** (K-031). The single most dangerous documented behaviour
   in this component — create-only publishes with independently-stale `reference` and `ws:{id}` —
   is invisible from the outside. A `GET /workflow-defs/{key}/versions/{v}?expand=steps` plus the
   same for a snapshot would make the trap *detectable* instead of merely *documented*, and would
   let an operator diff def against snapshot in one call. This pass had to use raw Cypher for it.
2. **Widen the tool-call parse before doing engine surgery** (K-027). DEF-K027-B suggests the
   cheapest first move on the terminal-post problem is a tolerant `_parse_content_tool_calls`, not
   a new engine contract. It is offline-testable with fixtures and would have rescued a real
   observed run. Do it *before* the engine-level guarantee, then re-measure.
3. **The process flow is the strongest artifact in M3.** The `access-request@v1` estate reproduces
   its plan's step-by-step table exactly, its error map is precise and its rejections are free.
   The `awaiting` envelope on `StepRun.output` is a genuinely good serviceability decision — a
   client renders the right prompt with no extra query. Worth keeping as the template for future
   process defs.
4. **The debug trace is excellent and under-advertised.** 18 events reconstructed an 8-step run
   including every guard's inputs, verdict and rationale. It deserves a short "how to debug a run"
   section in the README — it is the fastest path to diagnosing a parked run.
5. **Testability nit:** `AGENTS.md` documents the `reference`-wipe trap thoroughly, and the
   documentation is accurate — but every QA/dev run pays the cost manually. A
   `scripts/verify_workflows.sh <ws>` that asserts "both defs present in `reference` **and**
   snapshot-consistent in `ws:<id>`" would turn a documented discipline into a one-command check
   (it becomes trivial once DEF-1/K-031 lands).

---

## 8. Environment changes made by this pass

| Change | State at end |
|---|---|
| FalkorDB | Already running; **not** started or restarted by this pass. Untouched |
| LM Studio | Already running and reachable; not modified |
| **`ws:qa`** | Created (`bootstrap_schema.sh` @ dim 1024, `seed_demo.sh`, `seed_workflows.sh`), exercised, and **deleted** (`GRAPH.DELETE ws:qa` → `OK`). Absent from `GRAPH.LIST` |
| `reference` | **Additive-only.** Two throwaway defs (`qa-imm@v1`, `qa-notimpl@v1`) were published during TP-017/TP-071 and were **cleared by the closing `pytest` run**; `reference` now holds exactly `access-request@v1` + `triage@v1`, re-seeded and verified. No def or snapshot subgraph was ever deleted |
| `ws:acme` | Untouched apart from the idempotent re-seed no-ops; both snapshots intact |
| `ws:live` | Created and torn down by the live test itself (`KEEP_WS` unset). Absent from `GRAPH.LIST` |
| Working tree | No source file changed. Only this report, the test plan, and the `BACKLOG.md`/`HISTORY.md` updates. **Nothing committed** |
| Server process | A uvicorn instance was run on port **8100** against `ws:qa` for the black-box phase and **stopped** at teardown |
