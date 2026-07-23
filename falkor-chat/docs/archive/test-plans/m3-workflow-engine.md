# Test plan — M3 workflow engine (K-025 acceptance pass)

> **Version:** 1.0 · **Author:** `qa-engineer` · **Written:** 2026-07-21 (before execution)
> **Item:** `docs/BACKLOG.md` **K-025 — QA acceptance pass on M3 (Slice 5)** — the un-run U15.
> **Under test:** commit `98a3cc8` (tree clean), `falkor-chat/server` + `scripts/`.
> **Report:** `docs/archive/test-reports/m3-workflow-engine-report.md` (sibling artifact, written after).

---

## 1. Scope & objective

Verify, **black-box and by executing the running system**, that milestone M3 delivers the
workflow engine it claims: **publish → materialize → start → drive → park → resume → terminal**,
with chat linkage and tracing, across **both** proof flows:

| Flow | `kind` | Dependencies | Role in this pass |
|---|---|---|---|
| **`access-request@v1`** (K-024) | `process` | none — fully offline, no LM Studio | the **primary** subject: every deterministic behaviour is verified here |
| **`triage@v1`** (K-022 U13) | `conversation` | LM Studio (chat + embedding) at `:1234` | the **model-gated** subject: AC-1 trigger + structural demonstration only |

**Objective:** produce a defensible verdict that either takes **M3 → ✅** or names precisely
what blocks it.

### 1.1 References (sources of truth)

| Artifact | What this plan takes from it |
|---|---|
| `docs/archive/requirements/llm-native-workflows.md` | canonical **FR-1…FR-7 / AC-1…AC-6** text — every test item traces to an ID here |
| `docs/archive/plans/m3-executor.md` | §2 execution model · §4 node capabilities (AC-6 fence) · §5 tracing (AC-5) · §6 trigger (AC-1) · §7 runaway safety · §8.x triage-flow mapping |
| `docs/archive/plans/m3-process-flow.md` | §3.3 publish invariants · **§4** the `access-request@v1` def, its six transitions and its **three** expected paths (§4.3) |
| `docs/DESIGN.md` §6.1–§6.3 | step types, guard semantics, "coordination is workflow" |
| `docs/QUERIES.md` §12 | run/step-run/trace read contracts, used when reading the graph directly |
| `docs/BACKLOG.md` K-025 / K-027 / K-028 / K-029 / K-030 | the binding scope boundaries below |
| `AGENTS.md` | executor/workflow-def invariants; `reference`-wipe sequencing trap; def immutability |

### 1.2 What this pass may claim — **binding** (K-025 + decisions D12-B / D7 / D10 / D14 / D-C / D-E)

**Verified in full (executed):**
- **AC-1** (trigger → `WorkflowRun` linked to the triggering message)
- **AC-5** (tracing: debug records rationale, non-debug records none)
- **AC-6** (per-node tool fence)
- The **entire `access-request@v1` process flow** — all three §4.3 paths, park/resume, input
  validation, budget, error map, publish invariants.

**Recorded as "model-gated, structurally demonstrated" — NOT as failures:**
- **AC-2b** (fuzzy guard advances the intake loop), **AC-3** (GraphRAG-grounded findings),
  **AC-4** (answer posted + run done). Per **D7** AC-4 is **structural-only** — the live test
  asserts a `PRODUCED` reply, not its provenance.
- Live-triage **reliability** on the local 4B is descoped to **K-027**. A flaky terminal
  `post_message` or an uncalibrated fuzzy guard is a **known, filed limitation**, not a new defect.

**Specified behaviour — never reported as a defect:**
- `wait` is **signal-driven, not timer-driven**, and mechanically identical to `human` (**D-C**).
  A parked `wait` that never advances on its own is **correct**. Real timers are **K-028**.
- `prompt` / `tool` / `message` step types raise `NotImplementedError` by design (**D-E**).
- The shipped fuzzy guard runs on the degraded **RECENT-TURNS** tier (**D14**) — a
  `guard_judgment` citing turn text is expected.
- Published defs are **create-only** (`MERGE … ON CREATE SET`); a re-publish of an edited def is a
  silent no-op. Documented, tested for, not filed as a bug.

**Quotation rule (D10):** any verdict line sourced from the guard calibration must carry the
**D10 small-n caveat verbatim** (`docs/archive/plans/m3-guard-calibration.md` §8) immediately beside it.

### 1.3 Explicitly out of scope

| Not tested | Why |
|---|---|
| Judge calibration / false-advance / recall arms | **K-027** item 3; needs the golden-set harness, not an acceptance pass |
| Live-triage reliability across repeated runs (n≥8) | **K-027** items 2/5 (D12-B descope) |
| Timers / scheduled wakeups | **K-028** — no scheduler exists by decision D-C |
| Real external MCP server as a workflow node tool | plan §4 scoped it to a stub/in-memory client for the first cut |
| Auth / multi-tenancy | M2.5 (K-016…K-018), deferred |
| Performance / RAM of workflow nodes | no new node type or index landed in K-024; M1 §11 numbers stand |
| Web UI rendering of workflow runs | no UI surface shipped for runs in M3 |

---

## 2. Risk assessment (drives prioritisation)

| # | Risk | Likelihood | Impact | Priority |
|---|---|---|---|---|
| R1 | **Publish invariants leak** — a def with a `human` step missing `waitsForHuman`, a typo'd `cmp` op, or zero transitions gets published and parks/self-loops forever. The zero-transition case is *unrepairable* on that version (AGENTS.md). | med | **high** | P1 |
| R2 | **Park/resume mechanics wrong** — a `human`/`wait` step advances when it should park, or a resume applies to the wrong branch / double-resumes. Core of the K-024 design claim. | med | **high** | P1 |
| R3 | **Input validation not free** — a rejected input consumes step budget or half-writes ctx, so ordinary human mistakes exhaust a run. | med | high | P1 |
| R4 | **Tracing wrong in either direction** — a debug run records nothing (FR-4 serviceability lost) or a **non-debug run records rationale** (AC-5's negative half, a privacy/leanness violation). | low | high | P1 |
| R5 | **Trigger fires twice or not at all** — both a workflow and a direct reply on one `@mention`, or no run at all; or the run is not linked `TRIGGERED_BY`. | med | high | P1 |
| R6 | **Tool fence not enforced** — an ungranted tool is offered or dispatched (AC-6). | low | high | P1 |
| R7 | **Runaway loop** — budget not enforced, or budget exhaustion surfaces as a 500 rather than a recorded `failed` run. | low | high | P2 |
| R8 | **Operational traps bite** — the `reference`-wipe sequencing and def-immutability/split-brain hazards documented in AGENTS.md are real and easy to trip; a flow silently no-ops. | **high** | med | P2 |
| R9 | **Live triage** unreliable on the local 4B | **high** | low *(descoped to K-027)* | P3 |
| R10 | Baseline regression (suites not green at `98a3cc8`) | low | high | P1 |

Deliberately **not** covered: concurrency/CAS contention on resume beyond the single-writer path
(covered by the offline estate's CAS tests; a real race needs a load harness this pass does not
build), and long-run RAM behaviour.

---

## 3. Environment & data setup

- **FalkorDB** `falkordb/falkordb:v4.18.11` at `127.0.0.1:6379` (container `falkordb-dev`, existing
  `falkordb-data` volume — non-destructive).
- **LM Studio** at `:1234` — probe first. If unreachable, the live half is recorded
  **not executed / environment-gated**, and the pass does **not** fail for it.
- **Server venv:** `server/.venv` (`pip install -e '.[dev]'` already done).
- **Isolated workspace `ws:qa`** — created for this pass and **deleted at the end**:
  `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh qa` → `./scripts/seed_demo.sh qa` →
  `./scripts/seed_workflows.sh qa`. Dim **1024** matches the live embedding model
  (probed, not assumed).
- `reference` and `ws:acme` are **additive-only**. **Forbidden:** deleting any def or snapshot
  subgraph (breaks live runs via `OF_DEF`/`AT_STEP`); any tree-mutating git command; committing code.

**Mandatory sequencing** (AGENTS.md — both suites wipe the global `reference` graph):

```
server pytest  →  ./scripts/test_queries.sh  →  ./scripts/seed_workflows.sh <ws>  →  verify defs  →  exercise
```

Never the reverse: a workflow exercised against a wiped `reference` silently no-ops.

### 3.1 Entry criteria

- Tree clean at `98a3cc8`; FalkorDB reachable.
- **`server` pytest = 533 passed / 1 deselected** and **`test_queries.sh` = 256/256**. Any deviation
  is itself a finding and blocks the acceptance items.
- Both defs (`triage@v1`, `access-request@v1`) present in `reference` **and** snapshotted into the
  workspace under test, verified after the last suite run.

### 3.2 Exit criteria

- Every P1 item executed with recorded evidence, or explicitly marked **blocked** with the reason.
- Every failure reproducible from the written steps.
- Verdict + defect list + backlog/history updates written.

---

## 4. Test items

Priority **P1** = must run for a verdict · **P2** = run if green · **P3** = model-gated/opportunistic.
Types: `contract` (REST/API contract) · `e2e` · `integration` · `functional` · `exploratory` · `regression`.

### 4.0 Baseline

| ID | Title | Preconditions | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|---|
| **TP-000** | Green baselines at `98a3cc8` | FalkorDB up, tree clean | `pytest -q`; `./scripts/test_queries.sh` | **533 passed / 1 deselected**; **256/256** | P1 | regression |
| **TP-001** | Re-seed after the wipes; both defs present | TP-000 done | `seed_workflows.sh acme`; `seed_workflows.sh qa`; read defs + snapshots | both defs in `reference`; both snapshots in `ws:qa`; second run prints `already present — no-op` | P1 | integration |

### 4.1 Publish & materialize — FR-1/FR-3, R1

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| **TP-010** | Def + snapshot readable black-box | `GET /workflow-defs`, `GET /workflow-defs/access-request?version=v1`, `GET /workspaces/qa/snapshots` | def with **6 steps / 6 transitions**, `kind:'process'`, correct `startKey='submit'`; snapshot listed | P1 | contract |
| **TP-011** | `human`/`wait` without `waitsForHuman` rejected at publish | `POST /workflow-defs` with a `human` step lacking the flag | **400** `WorkflowDefSpecError`; nothing published | P1 | contract |
| **TP-012** | Typo'd `cmp` op rejected at publish | publish a transition guard `{"kind":"cmp","op":"equalz",…}` | **400** `WorkflowConfigError`/spec error; nothing published | P1 | contract |
| **TP-013** | Zero-transition def rejected **before** the half-write (K-024 U4b / O-6) | publish a def with `transitions: []` | **400**, and the def is **absent** from `reference` (no half-written, unrepairable version) | P1 | contract |
| **TP-014** | Start-step cardinality | publish with 0 start steps, then with 2 | **400** each | P2 | contract |
| **TP-015** | REST string-typed `config`/`guard` still validated | publish via REST where `config`/`guard` arrive as JSON **strings** with an invariant violation | **400** — the normalize-first rule holds; no escape hatch | P1 | contract |
| **TP-016** | Unwhitelisted `cmp` path root rejected at publish, total at drive | publish a guard on a non-whitelisted root | **400** at publish | P2 | contract |
| **TP-017** | Def immutability is create-only (documented behaviour) | re-publish the same `key/version` with changed content | no error **and no change** to the stored def — confirms the documented trap, not a defect | P2 | exploratory |

### 4.2 The `access-request@v1` process flow — DESIGN §6.3, plan §4.3, R2

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| **TP-020** | **Privileged happy path** (`role:"contractor"`) | `POST /workflow-runs {access-request,v1,maxSteps:24}` → park; `input {"request":{"role":"contractor",…}}`; `input {"decision":"approve"}`; `input {"provisioned":true}` | parks at **submit → approval → provision**; ends `status:"done"` at **`activate`**; **8** step-runs in `NEXT` order with keys `submit,submit,route,approval,approval,provision,provision,activate` | P1 | e2e |
| **TP-021** | **Standard-hire auto path** (`role:"engineer"`) | start; `input {"request":{"role":"engineer"}}`; `input {"provisioned":true}` | routes **submit→route→provision** (unconditional #3 beats nothing; #2 false), no approval step; `done` at `activate`; **6** step-runs | P1 | e2e |
| **TP-022** | **Rejected path** | start; file a `contractor` request; `input {"decision":"reject"}` | terminal step **`rejected`**, run `status:"done"` (not `failed`) — the *process* completed | P1 | e2e |
| **TP-023** | `wait` parks and does **not** self-advance (D-C, specified) | reach `provision`, wait ≥ 20 s, re-read the run | still `waiting`, `stepCount` unchanged, `awaiting.kind` = the wait signal — **correct**, not a defect | P1 | functional |
| **TP-024** | Re-park on a "not yet" signal | at `provision`, `input {"provisioned": false}` | run re-parks `waiting`; `stepCount` +1; a later `true` still advances | P2 | functional |
| **TP-025** | Conditional-beats-unconditional ordering | inspect the `route` step-run of TP-020 vs TP-021 | contractor takes #2 (`needs_approval`), engineer takes #3 (`auto`) | P1 | functional |
| **TP-026** | Guard reads nested ctx (`ctx.request.role`) | TP-020 evidence | transition #1 `exists` and #2 `in` fire off the nested submitted object | P2 | functional |

### 4.3 Input contract & error map — R3

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| **TP-030** | Empty input rejected free | `input {}` on a parked run | **400**; `stepCount` **unchanged**; run still `waiting` | P1 | contract |
| **TP-031** | Undeclared key rejected free | `input {"nope":1}` at `submit` (declares `fields:["request"]`) | **400**; no budget consumed | P1 | contract |
| **TP-032** | `expects` violation rejected free | `input {"decision":"maybe"}` at `approval` | **400**; no budget consumed; run still parked at `approval` | P1 | contract |
| **TP-033** | Reserved ctx key rejected | `input` naming a reserved key | **400** | P2 | contract |
| **TP-034** | Unknown run → 404 | `POST /workflow-runs/does-not-exist/input` | **404** | P1 | contract |
| **TP-035** | Input to a terminal run → 409 | submit input to the completed TP-020 run | **409**, nothing written | P1 | contract |
| **TP-036** | Start against an unknown def → 404 | `POST /workflow-runs` with a bogus key | **404**, nothing started | P2 | contract |

### 4.4 Tracing / serviceability — FR-4 / **AC-5**, R4

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| **TP-040** | Debug run records the trail | start with `trace:true`, drive to done, `GET /workflow-runs/{id}/trace` | non-empty, `seq`-ordered `TraceEvent`s with recognised `kind`s; guard evidence present per step | P1 | e2e |
| **TP-041** | **Non-debug run records nothing** (AC-5 negative half) | same flow, `trace:false` | trace read returns **zero** events; `TraceEvent` count for that run = 0 in the graph | P1 | e2e |
| **TP-042** | Step-run audit trail is reconstructable | `GET /workflow-runs/{id}/step-runs` for both | `NEXT`-ordered, one row per executed step, with status/outcome — the run can be reconstructed after the fact (FR-4) | P1 | contract |

### 4.5 Node capability fence — FR-6 / **AC-6**, R6

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| **TP-050** | Only granted tools are **offered** | drive an `agent` step whose `config.tools` grants exactly one tool, through the real executor + real graph with a scripted LLM that records what it was offered | exactly the granted schema(s) offered; nothing else | P1 | integration |
| **TP-051** | Ungranted call **defensively rejected** | scripted LLM calls a registered-but-ungranted tool | call is **not dispatched**; an error is fed back as a bounded re-prompt; run survives | P1 | integration |
| **TP-052** | Malformed granted call refused, not dispatched | scripted LLM calls a granted tool missing a required arg | refusal message back to the model; no dispatch; bounded by `maxIterations` | P2 | integration |

### 4.6 Trigger & chat linkage — FR-7 / **AC-1**, K-023, R5

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| **TP-060** | `@mention` starts a run **linked to the triggering message** (AC-1) | run the served app on `ws:qa` with `FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=1`; post `@assistant …` to a thread | a `WorkflowRun` exists with `-[:TRIGGERED_BY]->` **that** `Message`; run `ctx` carries the thread anchor | P1 | e2e |
| **TP-061** | Exactly one handler per message | same run | no *direct* responder reply in addition to the workflow (no double answer) | P1 | e2e |
| **TP-062** | `StepRun-[:PRODUCED]->Message` emission linkage | inspect the triggered run's step-runs | any message a step posts is linked `PRODUCED` from its StepRun (structural; **D7** — provenance of the answer is **not** asserted) | P2 | integration |
| **TP-063** | Loop guard: agent-authored message never re-triggers | inspect after TP-060 | no run triggered by an `assistant`-role message | P2 | functional |

### 4.7 Runaway safety & the typed-handler seam — §7, D-E, R7

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| **TP-070** | Step budget is enforced and **recorded** | start `access-request` with a deliberately small `maxSteps` and drive past it | run ends `status:"failed"` (budget), **not** a 500; the trail records how far it got | P1 | functional |
| **TP-071** | `prompt`/`tool`/`message` raise `NotImplementedError` **by design** (D-E) | publish a QA def with a `prompt` step; start it | the D-G envelope: HTTP **201/200** carrying `{"status":"failed", "error":…}`, run terminal `failed` in the graph — **specified**, not a defect | P1 | functional |
| **TP-072** | Default budget documented behaviour | start without `maxSteps` on a flow needing more | falls back to the global default (12); documented, observed | P3 | exploratory |

### 4.8 Live triage — model-gated (AC-2b/AC-3/AC-4), R9, **P3**

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| **TP-080** | Live triage e2e | `pytest -m live -s` (needs FalkorDB **and** LM Studio chat+embedding) | the run reaches a terminal state; AC-2b/AC-3/AC-4 recorded **model-gated, structurally demonstrated**. An unreliable terminal post is **K-027**, not a defect. If LM Studio is unreachable: **not executed / environment-gated**. | P3 | e2e |
| **TP-081** | Guard evidence tier is the degraded RECENT-TURNS tier (D14) | inspect a live run's `guard_judgment` payloads | payload citing turn text is **expected**, not a defect | P3 | exploratory |

### 4.9 Operational hazards — R8

| ID | Title | Steps | Expected | Pri | Type |
|---|---|---|---|---|---|
| **TP-090** | The `reference`-wipe trap is real and the documented order defuses it | observe def presence before/after a suite run, then after re-seed | after `pytest`/`test_queries.sh` the defs are gone/stale; the documented re-seed restores them — confirms the AGENTS.md warning is accurate and load-bearing | P2 | exploratory |
| **TP-091** | Seed idempotence | run `seed_workflows.sh qa` twice | second run: `already present — no-op` for **both** defs; no duplicate nodes | P2 | regression |

---

## 5. Traceability — AC → items

| AC | Items | Claim ceiling |
|---|---|---|
| **AC-1** trigger | TP-060, TP-061, TP-063 | **verified** |
| **AC-2** intake loop | TP-080 (live) + TP-020/023/024 for the *mechanism* (park/resume) | **AC-2a mechanism verified** on the process flow; **AC-2b fuzzy advance = model-gated** |
| **AC-3** retrieval | TP-080 | **model-gated, structurally demonstrated** |
| **AC-4** answer/done | TP-080, TP-062 | **structural-only (D7), model-gated** |
| **AC-5** tracing | TP-040, TP-041, TP-042 | **verified** |
| **AC-6** node bounds | TP-050, TP-051, TP-052 | **verified** |
| FR-1/FR-3 | TP-010…TP-017, TP-071 | verified |
| FR-4 | TP-040…TP-042 | verified |
| FR-5d (human handoff capability) | TP-020…TP-024 (the `human`/`wait` park+resume mechanism) | verified as a mechanism |
| FR-7 | TP-060 | verified |

## 6. Teardown

`ws:qa` is deleted at the end of the run (`GRAPH.DELETE ws:qa`). `reference` and `ws:acme` are left
exactly as found plus whatever the documented re-seed restores. No def or snapshot subgraph is
deleted. Nothing is committed by the execution phase; only the four K-025 documents are written.
