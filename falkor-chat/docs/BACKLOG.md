# Backlog — falkor-chat

> Forward-looking backlog for the `falkor-chat` component (formerly `kaizen/plan.md`; item IDs
> keep the `K-` prefix). Delivered work is logged in [`HISTORY.md`](./HISTORY.md); completed
> plan documents move to [`archive/`](./archive/).
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to HISTORY.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-21 (**K-025 delivered ✅ ⇒ MILESTONE M3 ✅** — the `qa-engineer`
> acceptance pass ran against commit `98a3cc8` on green baselines (server pytest **533 passed / 1
> deselected**, query suite **256/256**, both re-confirmed afterwards) and returned **PASS with
> parked, model-gated limitations, zero blocking defects**. **AC-1 / AC-5 / AC-6 verified by
> execution**; the entire `access-request@v1` process flow verified (all three §4.3 paths reproduce
> the plan's step table exactly); **AC-2b / AC-3 / AC-4 recorded model-gated, structurally
> demonstrated** per D12-B / D7 — observed working in a live interactive run, with `pytest -m live`
> red 2/2 on the AC-4 answer post, which is **K-027**, not a new defect. Two non-blocking findings:
> **K-031** (new — no black-box read of a def's/snapshot's structure, making the create-only
> split-brain undetectable; plus a one-step budget overshoot nit) and an addendum on **K-027** (the
> prose-tool-call failure is not terminal-node-specific and has a cheap parse-layer mitigation).
> Artifacts: `docs/archive/test-plans/m3-workflow-engine.md` + `docs/archive/test-reports/m3-workflow-engine-report.md`;
> see HISTORY.md 2026-07-21. Prior review 2026-07-21: **K-024 delivered ✅ + analyst-gated twice** — the LLM-free
> `kind:'process'` proof flow (`access-request@v1`) closes M3's last **build** item, so **K-025 (QA
> acceptance) is unblocked** and is now all that stands between the component and **M3 ✅**. Units
> U0–U5: deterministic `cmp` guards, typed `human`/`decision`/`wait` step handlers + two publish
> invariants, start-without-trigger + the human-input REST endpoint, the proof def + offline
> acceptance test, closeout. The central design claim held — **`_drive_loop` was never modified**
> (SHA `71055f756280` throughout). New baselines: server pytest **523 → 533 passed / 1 deselected**;
> query suite **241 → 256**. Gates: plan gate (request-changes → v2.1 approved), implementation gate
> U0–U4 and re-gate U4b (both *approve with suggestions*, no blockers) →
> `docs/archive/reviews/m3-process-flow.md`. Three items filed out of it: **K-028** (workflow timers),
> **K-029** (converge the seed def sources), **K-030** (allow zero-transition defs). See
> HISTORY.md 2026-07-21. Prior review 2026-07-19 (**K-022 Landing 2 delivered ✅ + analyst-gated** — trigger + triage proof
> flow (U11–U14), the Defect-A guard thread-context seam fix, Defect-B tool-error survival, the U13
> workflow seed; gate `approve with suggestions`, 0 blocker / 2 major (both closed) / 3 minor / 3 nit
> → `docs/archive/reviews/m3-guard-thread-context-impl.md`. New server baseline **pytest 350 passed, 0
> skipped**; query suite unchanged **241/241**. **U15 (qa-engineer acceptance = K-025) was NOT run** —
> per decision D12-B the executor *mechanism* is proven and live-triage *reliability* is descoped to
> the new **K-027**, which also carries the gate's minors/nits. K-023 (chat linkage) closed by the U11
> trigger + Option-B `PRODUCED` wiring; K-024 is half-delivered (conversational triage flow seeded and
> run; the LLM-free `kind:'process'` flow not built). See HISTORY.md 2026-07-19. Prior review
> 2026-07-12 — **K-022 amended:** analyst post-implementation review added to its
> owner chain and done-condition — the team's first fully-gated coordinated run; see the
> review-gate note on the item. Prior review 2026-07-09: **M3 — Workflow engine started: slice 1
> delivered ✅** — K-020 (def
> model in `reference`) + K-021 (snapshot materialization) landed via the teco-coordinated run,
> see HISTORY.md 2026-07-09; new baselines **pytest 196 / query suite 193/193**. Full M3
> decomposition (K-020…K-025) in `docs/archive/plans/m3-workflow-engine.md` Part A — canonical item text
> lives there; compact copies below. **K-022 Landing 1 (offline executor + capabilities, U1–U10)
> delivered ✅ + analyst-approved 2026-07-12** (§13 guard-language decision resolved; suites
> 241/283 green; see HISTORY.md). Next on the critical path: **Landing 2 — the trigger + triage
> proof flow (K-023/K-024) → K-025 QA** ⇒ M3 ✅. Prior: M2 GraphRAG complete ✅ 2026-07-08 (K-008 +
> K-013 + K-014 + K-015, QA-accepted); M2.5 hardening still deferred: K-016/K-017/K-018 + a
> channel-scoped retrieval read.) See the milestone map below.

## Milestone-to-green map (architect plan, 2026-07-05)

| Milestone | Reaches ✅ when | Items |
|---|---|---|
| **M1 — Chat core** ✅ | **Reached** — DoD closed: append path load-tested, hot reads PROFILEd (DESIGN §11.1/§11.2), request/response web UI de-staled | **K-011 + K-012** (delivered ✅) |
| **M2 — GraphRAG** ✅ | **Reached (2026-07-08)** — embeddings + vector index @1024 + hybrid retrieval + AI agent participant with `EMITTED` provenance, QA-accepted (K-015 PASS, zero defects) | **K-008 + K-013 + K-014 + K-015** (delivered ✅ → HISTORY.md) |
| **M3 — Workflows** ✅ | **Reached (2026-07-21)** — def model + snapshot + executor + chat linkage, proven by one conversational + one business-process flow, **QA-accepted**: K-025 verdict **PASS with parked, model-gated limitations**, zero blocking defects (`docs/archive/test-reports/m3-workflow-engine-report.md`) | **K-020 ✅ + K-021 ✅** (slice 1) + **K-022 ✅ + K-023 ✅** (2026-07-19, Landing 1 + 2) + **K-024 ✅** (2026-07-21 — **both** proof flows) + **K-025 ✅** (QA = U15, 2026-07-21) ⇒ **M3 ✅**. K-027 (live-triage reliability), K-028/K-029/K-030 (filed out of K-024) and K-031 (filed out of K-025) are follow-ups, **not** M3-green gates. |
| **M2.5 — Hardening** *(deferred)* | Real auth, transport-level agent path, real-time push | **K-016 → K-017, K-018** |

> ✅ **Scope decision — CONFIRMED (user, 2026-07-05).** "M2 green" = **functional GraphRAG** (the
> narrow §12 roadmap DoD: embeddings + vector index + hybrid retrieval + agent participant +
> `EMITTED`). Real auth and real-time push are **deferred to the M2.5 hardening track**
> (K-016/K-017/K-018) — rationale: "long road before production." This is safe because the AI
> participant is a **server-side responder** that posts as a configured Agent and needs no
> per-request auth to function, so auth never blocks M2 green.
>
> The identity source-of-truth axis that used to gate K-016 is now **decided** (2026-07-05, user):
> the `identity` graph is **authoritative (standalone)**, not an external-IdP projection — DESIGN §1.2.
> K-016 (deferred track — not on any M2 path) implements auth *per* that decision; no user input pending.

## Sequencing (critical path + parallelism)

```
Parallel wave 1 (start now):
  K-011 (M1 load/PROFILE)   ─ independent (harness/docs, read-only on data)
  K-012 (M1 web polish)     ─ independent (web/ only)
  devops LM-Studio spike ─▶ K-008 gate (graph-dba) ─▶ K-008 impl (tdd)
                                                            │
                                                            ▼
                                       K-013 (agent + EMITTED)  ◀─ needs K-008 + K-010 [done]
                                                            │
                                            ┌───────────────┴──▶ K-014 (web M2) ◀─ also needs K-012
                                            ▼
                                       K-015 (QA M2 pass) ◀─ needs K-008+K-013+K-014  ⇒ M2 ✅

M1 ✅ = K-011 + K-012 — ACHIEVED (both delivered 2026-07-06).
Deferred M2.5 (after M2-green): K-016 (auth) ─▶ K-017 (transport agent QA);  K-018 (real-time)
K-019 (doc sync) ─ rolls into the K-008 graph-dba gate (docs it already touches), or standalone anytime.
```

- **Critical path to M2 green:** devops spike → K-008 gate → K-008 impl → K-013 → K-014 → K-015.
- **Fully parallel with the K-008 chain:** K-011 (harness/docs) and K-012 (`web/`) — no shared files.
- **Suite discipline:** only the graph-dba gates in K-008 and K-013 touch `QUERIES.md` / `test_queries.sh`
  (raising the 126 baseline with enumerated assertions); K-011/K-012/K-014 are suite-neutral; K-015 is a QA overlay.

## Locked M2 stack decisions

> **M2 stack (embedding model/dim, agent LLM, runtime, VRAM, upgrade path) is locked in
> `docs/DESIGN.md` §1.3** (decided 2026-07-04). Implemented in K-008/K-013.

> `bootstrap_schema.sh` default is `EMBEDDING_DIM=1536` — **must** be run with `EMBEDDING_DIM=1024`
> for any new workspace from K-008 on. (`start_server.sh` guidance defaults to 1536 too — fold the
> 1024 note into both in the K-008 gate.)

## Active

> **Milestone M3 — Workflow engine, in progress.** Slice 1 (**K-020 + K-021**) delivered
> 2026-07-09; **K-022 (executor, both landings) + K-023 (chat linkage)** delivered 2026-07-19;
> **K-024 (both proof flows)** delivered 2026-07-21 → HISTORY.md. **All build items are done.**
> Remaining to M3 ✅: **K-025 (QA acceptance = U15) — unblocked, not yet run**. **K-027**
> (live-triage reliability) is a parallel follow-up track, explicitly *not* an M3-green gate per
> decision D12-B; so are **K-028/K-029/K-030**, filed out of the K-024 gates.
> Canonical item text + slice-1 implementation plan: `docs/archive/plans/m3-workflow-engine.md`
> (Part A = decomposition, Part B = slice 1). Compact copies below.

### — Milestone M3 (Workflow engine) — ✅ DELIVERED (K-020…K-025 → HISTORY.md 2026-07-21) —

> **K-020 — Workflow definition model in `reference`** and **K-021 — Snapshot materialization
> into `ws:{id}` on publish** — **delivered ✅ 2026-07-09 → HISTORY.md.** Suites raised to
> query 193/193, pytest 196. Slice-1 residuals to carry into K-022: lock the def-spec
> `start_key` contract (implemented as "exactly one step declares `start: True`"); the
> `-[:HAS_STEP]->` containment edge added at the gate (index-anchored def-scoped reads).

### K-022 — Run + StepRun executor core (Slice 2) (✅ delivered + analyst-gated — Landing 1 2026-07-12, Landing 2 2026-07-19 · **U15 acceptance not run**, descoped to K-025/K-027 per D12-B)

- **Delivered (Landing 2, U11–U14) ✅ 2026-07-19:** the `@mention` trigger (`trigger.py`, resume-before-
  start, one handler per request, `WORKFLOW_ENABLED` default off), Option-B `StepRun-[:PRODUCED]->Message`
  emission linking, the Landing-1 **M-1** zombie-run fault net, agent-node thread context, U12 REST run
  inspection, the U13 `seed_workflows.sh` triage seed, and the U14 live e2e — plus the two defects that
  landing it exposed: **Defect A** (the intake→research guard could never fire — `thread=None` at the
  seam; fixed at the seam, not in a prompt) and **Defect B** (a hallucinated `@mention` failed the whole
  run; tool errors now survivable, split per **D16**: propagate every non-allowlisted `ServiceError` to
  the M-1 net and log unconditionally). **D14** reverted the S5 `understanding`-JSON intake instruction
  (it regressed live intake advancement **10/10 → 3/10** on the shipped Qwen3-4B) while **retaining** the
  Defect-C prompt mitigations ⇒ the shipped guard runs only on the **degraded RECENT-TURNS tier**.
  Baselines: **pytest 283 → 350 passed, 0 skipped**; query suite **241/241** (zero graph/DDL/QUERIES
  surface in the whole landing). Analyst gate = **approve with suggestions, 0 blocker / 2 major / 3
  minor / 3 nit** (`docs/archive/reviews/m3-guard-thread-context-impl.md`); both majors closed before the commit.
  **Not accepted:** U15 was not run — mechanism proven, live-triage reliability descoped to **K-027**.
  See HISTORY.md 2026-07-19, `docs/archive/plans/m3-executor-landing2.md`, `docs/archive/plans/m3-guard-thread-context.md`,
  and the coordination log `docs/archive/plans/m3-executor-coordination.md`.

- **Delivered (Landing 1, U1–U10) ✅ 2026-07-12:** the offline LLM-native executor + node
  capabilities (Phases 0–3) — schema/DDL + DESIGN reconciliation + QUERIES §12, `executor.py`
  (§2.1 A/B/C loop) / `guards.py` / `tools.py`, repository/services wiring. Suites raised to
  **query 241/241, pytest 283**, both green. Analyst gate = **approve-with-suggestions, 0
  blockers** (1 major M-1 + 3 minor + 3 nit; two seams deferred to Landing 2). Reframed as an
  offline-first landing under `docs/archive/plans/m3-executor.md`; teco-coordinated
  graph-dba → tdd-engineer → coder with a mandatory analyst review gate — the team's first
  fully-gated run. See `docs/HISTORY.md` (2026-07-12), the review at
  `docs/archive/reviews/m3-executor-impl.md`, and the coordination log
  `docs/archive/plans/m3-executor-coordination.md`. Landing 2 (U11–U14) landed 2026-07-19 — see the bullet
  above; U15 remains open as K-025.

- **Owner:** **`architect`** design pass first — engine-loop semantics **+ resolve DESIGN §13
  guard expression language (expr lib vs minimal DSL in `Step.config`/`TRANSITION.guard`) — a
  genuine user decision point, surface before implementing** → **`graph-dba`** gate (run/step-run
  write/read queries; `WorkflowRun`/`StepRun` DDL already exists) → **`tdd-engineer`** →
  **`analyst` post-implementation review** (added 2026-07-12; see review-gate note below).
- **Inputs/prereqs:** K-021 ✅ (materialized snapshots to walk); the §13 decision. Plan doc:
  `docs/archive/plans/m3-executor.md`. Also lock the `start_key` contract here (slice-1 residual).
- **Scope (DESIGN §6.2):** `WorkflowRun {runId,defKey,defVersion,status,startedAt,ctx}` with
  `OF_DEF`/`AT_STEP`/`HAS_STEP_RUN`; `StepRun {stepRunId,stepKey,status,…,input,output}` with
  `RAN` + `NEXT` audit trail. Engine loop: read `AT_STEP` → evaluate `TRANSITION` guards against
  `ctx` → create next `StepRun` → execute → append `NEXT` → move `AT_STEP`.
- **Done-condition:** both suites green at the new enumerated gate baseline; a run walks a
  materialized def deterministically; guards evaluated per the §13 decision; audit trail complete;
  **analyst review of the delivered diff at `docs/archive/reviews/m3-executor-impl.md` with verdict
  approve / approve-with-suggestions** (a "needs changes" loops back to the implementer, then
  re-review — the gate is part of done, not optional).
- **Review-gate note (process addition 2026-07-12, not in the frozen plan text):** K-022 is
  deliberately the team's **first fully-gated coordinated run** — the K-020/K-021 run skipped
  independent code review ("left to the user") despite teco's review-by-default rule. The
  coordinator must treat the analyst gate as a non-negotiable done-condition, not a judgment
  call, and record the run's cost datapoint (tokens/time vs. the ~100k-token/45-min ungated
  slice-1 baseline) in the coordination doc so the gate's cost/benefit is finally measurable.
  Counterpart items: `claude/teco/kaizen/plan.md` K-003, `claude/analyst/kaizen/plan.md` K-001.
- **Risks/RAM (rule 6):** run/step-run nodes are the M3 per-workspace hot growth line (execution
  traces); `status` index already provisioned. Guard evaluation must be sandboxed/bounded
  (injection/DoS if an expr lib is chosen).
- **Test strategy:** gate contract assertions for run write/advance; pytest executor units with a
  stub step-executor; guard-evaluation tests.

### K-023 — Workflow ↔ chat linkage (Slice 3) (✅ delivered 2026-07-19 inside K-022 Landing 2 → HISTORY.md)

> **Closed by K-022 Landing 2 (U11).** `TRIGGERED_BY` + the trigger wiring shipped in `trigger.py`;
> the StepRun→Message sense was **disambiguated as `PRODUCED`** (locked decision D2 — K-013's
> `EMITTED` Message→Message provenance is untouched), delivered via **Option B** (buffer emissions,
> link after `_record`) so the §2.1 loop and `record_step_and_advance` stayed byte-for-byte. All three
> carried inputs below are closed: **M-1** (the `_drive` fault net), **PRODUCED-link ordering**
> (Option B), **agent-node thread context** (`_read_thread_context`, and the Defect-A seam fix that
> finally made the guard read it). Zero graph/DDL/QUERIES change. Original item text kept below for
> provenance.

- **Owner:** **`graph-dba`** gate (`TRIGGERED_BY` / StepRun-`EMITTED` writes/reads) →
  **`tdd-engineer`**/`coder`.
- **Inputs/carried from K-022 Landing 1** (this slice = U11, the trigger wiring; see
  `docs/archive/reviews/m3-executor-impl.md` findings + the coordination doc's "Carried to Landing 2"
  section):
  1. **M-1 (analyst major)** — add a top-level `try/except` in `executor._drive` that `fail_run`s
     on an unexpected exception (today a mid-drive fault leaves the run stuck at `status='running'`,
     un-resumable). Analyst-recommended to fold into the U11 background-handler wiring.
  2. **PRODUCED-link ordering** — the live `StepRun-[:PRODUCED]->Message` link needs the U11 wiring
     decision: either pre-mint the `StepRun` before executing an agent node, or link emitted
     messages after `_record`. The tool is correct + tested when a `stepRunId` is resolvable and
     skips-with-`linked:false` otherwise; the coder correctly did not mutate the locked U8 loop.
  3. **Agent-node thread context** — `_run_agent_node` assembles run `ctx` only today; folding in
     full thread-message context lands **in** U11 (a hard prerequisite for AC-2, must not slip
     further).
- **Scope (DESIGN §5.1/§6.2):** `(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)` (incl.
  materialize-on-first-use) and `(:StepRun)-[:EMITTED]->(:Message)` (step posts into a thread via
  the §4 write path). **Gate must disambiguate the `EMITTED` overload** — K-013's
  Message→Message provenance (QUERIES.md §10) vs this StepRun→Message sense — or confirm reuse.
- **Done-condition:** suites green; message triggers a run, step emits a message; linkage
  queryable both directions. **Risks:** edges negligible; the `EMITTED` overload is the
  modeling-clarity risk. **Test strategy:** gate assertions; pytest linkage + triggered-run tests.

### K-024 — Proof flows: one conversational + one business-process (Slice 4) (✅ delivered — conversational 2026-07-19, process flow 2026-07-21, analyst-gated twice → HISTORY.md)

> **Delivered ✅.** Both proof flows exist and run. **Conversational** — the `kind:'conversation'`
> **triage** def (`triage@v1`, intake→research→answer, three `type:'agent'` steps) is published +
> materialized by `scripts/seed_workflows.sh` (K-022 U13, idempotent, service-layer-driven) and runs
> end-to-end to `status='done'` against live LM Studio (`tests/test_workflow_live.py`, `live`-marked);
> its *reliability* remains model-gated, not proven — see K-027. **Business process** — the LLM-free
> `kind:'process'` def **`access-request@v1`** (six steps / six transitions, submit→route→approval→
> provision→activate\|rejected over `human`/`decision`/`wait`) is the DESIGN §6.3
> coordination-is-workflow proof, and it needed **no new primitive, no new run state and no
> scheduler**: a `human` step is a step whose outgoing guard reads a `ctx` key that does not exist
> yet, so the executor's existing "no transition fired" outcome parks it, and writing the key over
> REST makes the same guard fire on resume. Delivered across five units — **U0** two additive
> queries (`start_run_untriggered`, `resume_run_with_ctx`, graph-dba-gated and PROFILEd, no DDL),
> **U1** the deterministic `cmp` guard family (`all`/`any`/`not`, whitelisted ops + path roots, caps,
> no parser/`eval`/dependency, total at drive & strict at publish), **U2** typed
> `human`/`decision`/`wait` handlers + a `NotImplementedError` seam for `prompt`/`tool`/`message`
> (a deliberate behaviour change from the old silent no-op) + two publish invariants, **U3**
> start-without-trigger + `POST /workflow-runs/{id}/input` with a five-handler error map, **U4/U4b**
> the def in `server/falkorchat/proof_defs.py`, its seeding, and the fully offline acceptance test
> `server/tests/test_process_flow.py`, **U5** closeout. The central bet held: **`_drive_loop` was
> never modified** — SHA `71055f756280` before, during and after. Baselines: server pytest
> **523 → 533 passed / 1 deselected**, query suite **256/256**. Plan `docs/archive/plans/m3-process-flow.md`
> (v2.1); coordination `docs/archive/plans/m3-process-flow-coordination.md`; gates (plan, implementation
> U0–U4, re-gate U4b) in `docs/archive/reviews/m3-process-flow.md`. Follow-ups filed rather than folded in:
> **K-028**, **K-029**, **K-030**.

- **Owner:** **`coder`**/`tdd-engineer` + `scripts/seed_workflows.sh` (mirrors `seed_demo.sh`).
- **Scope:** the M3 DoD proof — publish two canonical defs, materialize into `ws:acme`
  (additive-only), run both to completion: **`kind:'conversation'`** (agent Q&A over
  `prompt`/`tool`/`message` steps, reuses the M2 responder, needs LM Studio behind
  `FALKORCHAT_ENABLE_AGENT`) and **`kind:'process'`** (onboarding/approval over
  `human`/`decision`/`wait` steps, LLM-free — the §6.3 coordination-is-workflow proof).
- **Done-condition:** both runs reach a terminal step; seed idempotent; documented run-through.
- **Test strategy:** one e2e run test per flow (conversational behind a live marker; process
  deterministic/offline).

### K-025 — QA acceptance pass on M3 (Slice 5) (✅ **DELIVERED 2026-07-21 — verdict PASS with parked, model-gated limitations ⇒ M3 ✅**)

> **Run and closed.** `qa-engineer` executed the pass against commit `98a3cc8` on green baselines
> (server pytest **533 passed / 1 deselected**; query suite **256/256**, both re-confirmed after the
> pass). Artifacts: test plan `docs/archive/test-plans/m3-workflow-engine.md` (v1.0, written before
> execution) + test report `docs/archive/test-reports/m3-workflow-engine-report.md`.
> **Verdict: PASS with parked, model-gated limitations ⇒ M3 → ✅.** Zero blocking defects.
> - **AC-1 / AC-5 / AC-6 — VERIFIED by execution.** `@mention` → `WorkflowRun-[:TRIGGERED_BY]->Message`
>   read back from the graph; a debug run recorded **18 `TraceEvent`s** (`node_rationale` ×8,
>   `guard_judgment` ×8, `node_note` ×2, each guard with its verdict *and* its why) against **0** for
>   the same flow non-debug; the AC-6 fence held on **both** sides — only the granted schema offered on
>   every iteration, and an ungranted call defensively rejected without dispatch.
> - **The whole `access-request@v1` process flow — VERIFIED.** All three §4.3 paths reproduce the
>   plan's step-by-step table **exactly**, step counts included (privileged 8, standard-hire 6,
>   rejected 6 — rejected ends `done`, not `failed`). Nine publish-invariant negatives all reject
>   **before any write** (the zero-transition half-write hazard is closed); the input error map
>   (400/404/409) is precise and every rejection is **free** (`stepCount` unchanged); budget
>   exhaustion and the `NotImplementedError` seam both surface as the D-G `{"status":"failed"}`
>   envelope, never a 500.
> - **AC-2b / AC-3 / AC-4 — recorded model-gated, structurally demonstrated** per **D12-B** / **D7**.
>   All three were observed working in a live interactive run (intake parked → plain reply resumed →
>   research → answer → `done`, with a real `PRODUCED`-linked answer post); `pytest -m live` then
>   failed **2/2** on the AC-4 answer-post assertion. That is **K-027**, not a new defect and not an
>   M3 gate.
> - **Specified behaviour confirmed, not filed as defects:** a parked `wait` unchanged after 25 s
>   (D-C); `prompt` → `NotImplementedError` (D-E); the RECENT-TURNS guard tier (D14); create-only
>   def publishes (an edited re-publish returned `201` while the stored def kept its old content).
> - **No verdict line in the report is sourced from the guard calibration**, so the D10 caveat is not
>   attached to any line there; it remains binding for K-027 item 3.
> - **Two non-blocking findings:** **K-031** (new — no black-box read of a def's/snapshot's structure,
>   plus the one-step budget overshoot nit) and an addendum appended to **K-027** (the prose-tool-call
>   failure is **not** terminal-node-specific and has a cheap parse-layer mitigation).
> - `ws:qa` created and deleted; `reference`/`ws:acme` additive-only; nothing committed by the pass.

> **(Original item text, for the record.)** K-024's delivery (2026-07-21) removed the
> last prerequisite: both proof flows now exist, so the pass can cover the conversational
> `triage@v1` *and* the LLM-free `access-request@v1` process flow. **Carry into the pass:** the
> process flow is fully offline (no LM Studio needed for that half); `wait` is **signal-driven, not
> timer-driven, and mechanically identical to `human`** — a parked `wait` that never advances on its
> own is specified behaviour, not a defect (DESIGN §6.1/§6.3, decision D-C; real timers are K-028);
> `prompt`/`tool`/`message` steps raise `NotImplementedError` by design (D-E).

> **This is the un-run U15.** K-022 Landing 2 closed without it (decision **D12-B**): the executor
> *mechanism* is proven, live-triage *reliability* is descoped to K-027. When it runs, its scope is
> **AC-1/AC-5/AC-6 verified** with **AC-2b/AC-3/AC-4 recorded model-gated and
> structurally-demonstrated** (per D7, AC-4 is **structural-only** — the live test asserts a `PRODUCED`
> reply, not its provenance, because the `answer` node cannot see the research findings). Two
> constraints travel with the pass: the **D10 small-n caveat must appear verbatim next to any verdict
> line** sourced from the guard calibration, and the shipped guard runs only on the **degraded
> RECENT-TURNS tier** (D14) — a `guard_judgment` citing turn text is expected, not a defect.
> ⚠️ Sequencing: `pytest` and `test_queries.sh` both wipe the global `reference` graph, so the order is
> **`pytest` → re-seed (`seed_workflows.sh`) → verify → live acceptance**, never the reverse.

- **Owner:** **`qa-engineer`**. Black-box publish → materialize → run → trace → chat linkage for
  both proof flows; versioned plan/report per repo convention (`docs/test-plans/`,
  `docs/test-reports/`); isolated `ws:qa` (create + delete), `reference`/`ws:acme` additive-only.
- **Done-condition:** PASS (or PASS-with-parked-defects) on green baselines ⇒ **M3 ✅**.

### K-027 — Live triage reliability + carried gate findings (🔵 proposed — the D12-B descope from K-022 Landing 2)

> **Numbering note:** the coordination log calls these "**K-023 follow-ups**", but K-023 is already
> taken (workflow ↔ chat linkage, now ✅). They are collected here as **K-027**, the next free number.

- **Why it exists:** K-022 Landing 2 proved the executor **mechanism** and stopped there (decision
  **D12-B**). What is *not* proven is that the live triage flow behaves reliably: the terminal
  `post_message` call is unreliable on the shipped 4B, and the fuzzy-guard judge is **uncalibrated**.
  Those are **local-model-quality + engine-guarantee** problems, not executor bugs — so they were
  descoped rather than fixed unit-by-unit. **Not an M3-green gate** (see K-024/K-025).
- **Owner:** **`architect`** for the terminal-node contract (an executor change) → **`coder`/
  `tdd-engineer`**; **`data-scientist`** owns calibration method + golden-set expansion (advisory,
  never implements) with **`tdd-engineer`** running the harness.
- **Scope:**
  1. **Judge-parse robustness** — `app._build_llm_judge` uses `complete()` + a **bare `json.loads`**.
     A model that wraps its JSON in a ```` ```json ```` fence breaks *every* verdict silently: in the
     D13 probe Ministral scored **26/26 "unparseable judge output"**, including one correct
     `decision:true` destroyed by the fence. The shipped Qwen path is unaffected (its JSON is
     unfenced), which is exactly why this can rot unnoticed. Fix = fence/prose-tolerant parsing or
     structured output.
  2. **Terminal-node-must-post engine contract** — the structural fix for **Defect C**. Today's
     mitigation is prompt-level and **does not hold on a 4B**: the `answer` node emits a good grounded
     answer as plain text with no tool call, so no `PRODUCED` edge (AC-4 measured ~2/8, then 0/3 after
     a strengthened prompt, then 2/3 in the probe replay). A second measured mechanism: the folded
     `"{displayName}: {text}"` thread context leaks a display name into `mentions` → the §4 write
     rejects → the model "recovers" by dropping the tool. Needs an engine-level guarantee, not a prompt.
  3. **Judge calibration (D9/D10)** — run the protocol in `docs/archive/plans/m3-guard-calibration.md` §4:
     **false-advance ≤ 10% (screen) AND advance-recall ≥ 0.80**; κ is a **reported diagnostic**, not a
     gate (an always-suspend judge scores a perfect 0% FAR, so the original κ-based gate could be
     passed by a judge that never advances). Gate failure ⇒ **block the wiring** — no override, no
     compensating with `maxSteps`. **D10 is binding:** at N=26 a pass means *"no blocker found at a
     sample size that could only have found a large one"* — the §8 verbatim caveat must travel with
     every verdict line. Diagnostic already on record: on clean golden inputs Qwen's judge passes both
     arms (recall 0.818, false-advance 0.067), so the live 3/10 is a **generator-half** problem, not a
     judge problem.
  4. **Golden-set expansion (D11)** — `server/tests/eval/golden_guards.jsonl` exists (**26 rows**,
     well-formed and labeled) but **nothing reads it yet** (gate nit **n-3**) — the file will rot
     unless this item consumes it. A real FAR ≤ 10% bound needs ~30 suspend cases at zero failures
     (≈50–60 total), and **all current labels are one labeler's** — expansion should add a second
     labeler for the boundary tier or it buys precision without independence.
  5. **Ministral re-probe (D13 finding 2)** — Ministral-3B is *better* at the terminal tool call
     (native `post_message` 3/3 in replay where Qwen emitted prose 3/3) but far worse at judging
     (fence-fixed advance-recall 0.364 vs Qwen 0.818). Worth re-probing **only if** the judge is made
     model-robust (item 1). Notes: `docs/archive/plans/m3-capability-probe-ml.md`,
     `docs/plans/local-model-ram-budget-ml.md` (the ~4–5GB fits-16GB chat-model budget).
  6. **Exit** — these outcomes are what would let the **K-025 / U15 acceptance** move AC-2b/AC-3/AC-4
     from *model-gated, structurally demonstrated* to *verified*. K-025 can run before this item; it
     just cannot claim more than D12-B allows.
- **Addendum from the K-025 QA pass (2026-07-21) — the failure is not terminal-node-specific, and
  has a second, cheaper shape.** In a live interactive run the **intake** node (non-terminal) emitted
  the literal text `post_message({"text": "…could you please provide more details…"})` as its step
  output. The model *did* intend a tool call — it wrote it in **bare function-call syntax**, which
  `llm._parse_content_tool_calls` does not recover (it handles only JSON shapes: a
  `{"tool_calls": […]}` envelope or a JSON-object wrapper). The clarifying question therefore never
  reached the thread while the run parked correctly and looked healthy from the outside — a worse
  user-visible symptom than the terminal case, because nobody was ever shown the question the run is
  waiting on. Two consequences for this item: **(a)** item 2's terminal-node scope is too narrow —
  the engine-level guarantee must cover any node whose contract is "post"; **(b)** this is the exact
  structural twin of **item 1** (a parse layer intolerant of the shapes small local models actually
  emit), so widening `_parse_content_tool_calls` to recognise `name({json})` is a cheap,
  offline-testable mitigation that would have converted the observed run — **do it before the engine
  surgery, then re-measure.** Also recorded: `pytest -m live` is **RED deterministically (2/2)** on
  the AC-4 answer-post assertion — a known, filed limitation (D12-B), not an unknown regression.
  Evidence: `docs/archive/test-reports/m3-workflow-engine-report.md` §3.9 / DEF-K027-A / DEF-K027-B.
- **Carried findings from the analyst gate** (`docs/archive/reviews/m3-guard-thread-context-impl.md`, minors +
  nits — recorded here so they cannot rot):
  - **m-1 · `guards.py` negator window leaks across clause boundaries.** The 12-char window misses
    e.g. `"The user did not say; more info is needed."`, and a missed contradiction is a
    **false advance** — the *dangerous* direction under DS Q1, and the **opposite** of what the code
    comment claims ("erring narrow keeps the failure on the safe over-suspend side"). Stakes are
    limited (the backstop only fires on an already-inconsistent verdict; the prompt rule is the primary
    defense). **At minimum, fix the comment;** the real fix is a same-clause requirement plus pinning
    the three missed rationales in `SUSPENDING_RATIONALES` (only the safe direction is pinned today).
  - **m-2 · `guards._recent_turns` slices before filtering** (`thread[-n:]` then skips malformed/empty
    rows), so malformed rows shrink the evidence window exactly when the judge is on its degraded tier.
    Fix = filter first, slice second.
  - **m-3 · the judge's evidence tier is invisible in the trace** — `_select_transition` traces
    `(transition, guard_text, verdict)` only, so nothing records whether a judgment ran on
    `understanding` (primary) or `recent_turns` (fallback). Calibration (item 3) needs results
    **stratified by tier**. Fix = return the tier on `GuardVerdict` and fold it into the
    `guard_judgment` payload — additive, no graph change.
  - **n-1 ·** function-local `import json as _json` in `app._render_judge_user` / `_build_llm_judge`;
    every other module imports it at the top. **n-2 ·** the judge-prompt cap loop re-joins the whole
    message on each eviction — **O(n²)** in turn count (irrelevant at N=6, but a test drives it with 50).
  - **Doc-drift · the `_drive_loop` byte-identity lock is quoted as SHA `71055f756280` + 2844 bytes.**
    The **SHA is correct and reproducible; the byte count is wrong** (the extraction yielding that hash
    is 2860 bytes; a third figure, 2839, appears in an earlier coordination entry). A future gate
    verifying the lock by byte count would wrongly report it broken. Correct the figure wherever the
    lock is quoted — or drop the byte count and verify by SHA only.
  - **m-A / n-1 (carried from the earlier `m3-executor-landing2-impl.md` gate) ·** `node_note` is
    missing from the trace-kind enumeration in `docs/QUERIES.md` §12.10 and `docs/DESIGN.md` §5,
    although the executor emits it.
- **Risks/RAM (rule 6):** none new — no node type, index, or vector dimension changes. The terminal-node
  contract touches the executor loop, whose §2.1 A/B/C block is byte-identity-locked: any change there
  is a deliberate, reviewed act, not a refactor.
- **Test strategy:** offline pins for the parse-robustness + negator fixes (fenced-JSON and
  clause-boundary cases as fixtures); a `live`-marked reliability run for the terminal-post contract
  with an explicit n and no cherry-picking; the calibration harness reading `golden_guards.jsonl` per
  the §4/§7 protocol, reporting both arms with the D10 caveat attached.

### K-028 — Workflow timers / scheduled wakeups (🔵 proposed — filed out of K-024, decision D-C)

> **Why it exists.** K-024 settled `wait` as **signal-driven, not timer-driven**, for a verifiable
> reason: **this system has no scheduler.** FastAPI `BackgroundTasks` are request-scoped, so nothing
> in the process outlives a request to wake a parked run at a future time. Rather than pretend
> otherwise, `wait` was implemented as mechanically identical to `human` — it parks and is released
> by an external signal on `POST /workflow-runs/{id}/input`; only the `awaiting.kind` string differs
> (DESIGN §6.1/§6.3). That is honest and complete for the proof flow, but it means an **SLA/escalation
> step ("if no approval in 48h, escalate") cannot be expressed today** without an external cron
> poking the endpoint.
- **Owner:** **`architect`** (the scheduling mechanism is an ops/architecture choice, not a step-type
  choice) → **`coder`**; **`devops`** for whatever process/timer surface it lands on;
  **`graph-dba`** only if a due-time index is added.
- **Scope sketch (to be designed, not decided here):** a durable due-time on a parked run
  (a `WorkflowRun.wakeAt` property + an index, or a separate timer node), plus *something that ticks*
  — an in-process scheduler, an external cron calling a `POST /workflow-runs/due` sweep, or a Redis
  keyspace-notification consumer. The sweep must reuse the **existing** resume CAS so a timer wakeup
  and a human signal cannot double-resume the same run.
- **Explicitly not in scope of K-024:** `wait`'s current semantics are correct and shipped; this item
  *adds* a release mechanism, it does not fix a defect. A parked `wait` that never advances on its own
  is specified behaviour.
- **Risks/RAM:** a due-time index on `WorkflowRun` is small but non-zero (AGENTS.md rule 6 — call it
  out at design time). The real risk is the scheduler becoming a second source of truth for run state.
- **Test strategy:** offline — an injected clock driving the sweep; a CAS-contention test proving a
  timer wakeup and a concurrent human submit resolve to exactly one resume.

### K-029 — Converge the seed def sources into `proof_defs.py` (+ the symmetric `decision` publish invariant) (🔵 proposed — filed out of K-024, open item O-5 / gate m-9 / nit n-3)

> **Why it exists.** The two seeded defs use **two different source conventions**, deliberately for
> the K-024 slice: `access-request@v1`'s spec is imported from `server/falkorchat/proof_defs.py`
> (so the seed script and the offline acceptance test provably cannot drift), while **`triage@v1`'s
> literal is still inline in `scripts/seed_workflows.sh`**. Moving `triage`'s def *during* K-024 was
> declined with a reason: published defs are **create-only** (`MERGE … ON CREATE SET`), so a byte-diff
> introduced while relocating a **live** def is silently swallowed — the re-seed prints
> `already present — no-op` while the old config keeps running, and `reference`/`ws:<id>` can go stale
> independently. That is a split-brain risk to take on its own, with its own verification, not as a
> rider on a feature slice.
- **Owner:** **`coder`**, with an explicit before/after equality check on the published def subgraph
  (not just "the script ran").
- **Scope:** (1) move `triage@v1`'s inline literal into `proof_defs.py` beside `ACCESS_REQUEST_DEF`,
  leaving `seed_workflows.sh` a pure driver over the service layer for **both** defs; (2) prove the
  move is byte-identical *in the graph*, which given create-only semantics means either verifying
  against a freshly published `reference` or bumping `triage`'s version in lockstep with
  `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` (note `start_server.sh` neither forwards nor exports
  those two vars today — a version bump also needs a script change); (3) fold the `n-A` warning
  (`ACCESS_REQUEST_DEF`'s key set **is** `publish_workflow_def`'s keyword signature) into whatever
  shape both defs end up sharing.
- **Also carries nit n-3 — the symmetric `decision` publish invariant.** K-024 enforces
  "a `human`/`wait` step must declare `config.waitsForHuman: true`" at publish, but **not** its
  mirror: **a `decision` step whose outgoing transitions are *all* conditional and which does not
  declare `waitsForHuman` self-loops until the step budget fails the run.** It is documented as a
  warning in `falkor-chat/AGENTS.md` and deliberately left unenforced because the symmetric check
  would **retro-reject existing test fixtures** (`server/tests/test_services.py`) — the same
  blast-radius problem B-2 caused in K-024, which is precisely why it belongs in an item that can
  budget for the fixture edits.
- **Risks:** touching a live published def is the risk; there is no new graph surface and no RAM cost.
- **Test strategy:** a test that both defs come from importable constants; a publish-equality check
  over the def subgraph; if n-3 is implemented, one ordering pin (it must run **last**, like the other
  three invariants) plus the fixture edits it forces.

### K-030 — Allow zero-transition (single-step) workflow defs; guard the `UNWIND` instead of rejecting (🔵 proposed — filed out of K-024 re-gate findings r-1/r-2)

> **Why it exists.** `repository._PUBLISH_CYPHER` ends in a bare `UNWIND $transitions AS tr …
> RETURN …`. With `$transitions = []` the row stream **collapses** — after the `WorkflowDef`, its
> `Step`s and the `START` edge have already been written — so the caller's `res.result_set[0]` raises
> `IndexError` on a **partially written** def. Because publish is `MERGE … ON CREATE SET`, retrying
> the corrected spec on the same `(key, version)` is a **silent no-op on the half-written def**: the
> version is permanently wrong and cannot be repaired by re-publishing. This is the same empty-`UNWIND`
> class that `AGENTS.md` documents as *guarded* for the §4 mention write-block; this path was not
> guarded. K-024 U4b **closed the reachable route** with a `_validate_def_spec` rule (running last)
> that rejects a transition-less spec **before any repository call** — prevention, not a nicer
> exception.
- **What is still open (re-gate r-1):** the fix is **publish-only**. `services.materialize_def` →
  `repository.materialize_snapshot` (`repository.py:1397`) **reuses the same query shape** and performs
  **no** spec validation, so a def poisoned before U4b — or any zero-transition subgraph read back by
  `read_def_subgraph`, which returns `transitions: []` rather than `None` — is still an unguarded
  `IndexError`/500 on materialize. Low likelihood (materialize is fed by publish, now guarded), but the
  guard is **asymmetric**, and the docs/tests currently imply it isn't:
  `server/tests/test_services.py:916` seeds a `FakeRepo` def with `"transitions": []` and asserts
  materialize **succeeds** — true of the fake, and exactly the shape the real query rejects.
- **Accepted limitation to remove (re-gate r-2):** the U4b rule also **rejects a legitimate shape** —
  a genuine single-step def. All four doc sites state the workaround ("a terminal outcome is a step
  with no *outgoing* transition, never a def with none") but none records it as **debt**. Without a
  `K-` number the next person needing a one-step def will either fight the rule or bypass validation.
- **The known cheap remedy:** guard the trailing `UNWIND` in `_PUBLISH_CYPHER` (and therefore
  `materialize_snapshot`, which reuses it) with the **§4 empty-`UNWIND` `CASE` pattern** this codebase
  already relies on and documents as load-bearing — `UNWIND (CASE WHEN $transitions = [] THEN [null]
  ELSE $transitions END) AS tr` with a `FOREACH` that never filters — then **relax**
  `_validate_def_spec`'s rule, and drop the `transitions=[]` mitigation comments in `proof_defs.py`
  and `tests/test_process_input.py`.
- **Owner:** **`graph-dba`** (the query change needs a gate + a re-PROFILE: the guard must not turn the
  index-anchored publish plan into a scan) → **`coder`** for the service-layer relaxation and the
  fixture/doc cleanup.
- **Risks/RAM:** none — no new node, index or property; a query-shape change only. The risk is plan
  regression, which is what the re-PROFILE is for.
- **Test strategy:** a publish and a materialize of a genuine single-step, zero-transition def, both
  asserted to succeed *and* to leave a complete subgraph (steps + `START` + the returned row); the
  existing ordering pins for the other publish invariants must stay green.

### K-031 — Def/snapshot **structure** read surface (make the create-only split-brain detectable) (🔵 proposed — filed out of the K-025 QA pass, DEF-1)

> **Why it exists.** `GET /workflow-defs/{key}` and `GET /workspaces/{ws}/snapshots` return
> **metadata only** (`{key, version, name, kind}` — `repository.get_def`, QUERIES §11.3). There is
> **no REST surface** that returns a def's steps, transitions, guards or `startKey`, nor a snapshot's
> materialized structure. The K-025 pass had to drop to raw Cypher to answer *"is what I think is
> published actually published?"* — which is precisely the component's most dangerous documented
> trap: published defs are **create-only**, so re-seeding an edited def is a **silent no-op**
> (QA confirmed: an edited re-publish of `qa-imm@v1` returned **`201`** while the stored def kept its
> old `name`, `kind` **and** step config), and `reference` (def) vs `ws:{id}` (snapshot) go stale
> **independently**. AGENTS.md documents the hazard thoroughly; nothing makes it **detectable**.
- **Owner:** **`architect`** (the read-surface shape — expand-on-demand vs. a dedicated subgraph
  endpoint vs. a diff endpoint) → **`coder`**. No `graph-dba` gate expected: `read_def_subgraph`
  already exists and is used by materialize; this is an adapter-layer exposure, not new Cypher.
- **Scope sketch (to be designed, not decided here):** a structure read for a def
  (`GET /workflow-defs/{key}/versions/{version}?expand=steps`, or a `/subgraph` sibling) and the same
  for a workspace snapshot, so an operator can **diff def against snapshot in one call** and see the
  split-brain the docs warn about. Optionally a `scripts/verify_workflows.sh <ws>` that asserts "both
  defs present in `reference` **and** snapshot-consistent in `ws:<id>`" — trivial once the read exists,
  and it turns a documented discipline into a one-command check (QA feedback item 5).
- **Also in scope (nit, same file surface):** the step budget **overshoots by one** — a run started
  with `maxSteps: 2` reached `stepCount: 3` before failing, because the budget is checked *after* a
  step executes. Harmless today, but it makes `maxSteps` mean "at least N", which a future
  SLA/costing story would trip over. Fix the check or document the off-by-one where `maxSteps` is
  specified.
- **Explicitly not in scope:** changing publish semantics. Create-only is a **decision**
  (`MERGE … ON CREATE SET`); converging the seed sources is **K-029** and allowing zero-transition
  defs is **K-030**. This item only makes the current semantics **observable**.
- **Risks/RAM:** none — read-only, no new node type, index or property. The only real risk is
  response size on a large def; bound it the way the other §12 RO reads are bounded.
- **Test strategy:** an offline contract test that a published def reads back with its exact steps,
  transitions and guards; a test that a def edited-and-re-published reads back **unchanged** (pinning
  the create-only semantics rather than hiding them); a def-vs-snapshot divergence fixture asserting
  the read makes the divergence visible.

> **K-011 + K-012 — delivered ✅ 2026-07-06 → milestone M1 — Chat core complete** (HISTORY.md).
> **K-008 + K-013 + K-014 + K-015 — delivered ✅ 2026-07-08 → milestone M2 — GraphRAG complete,
> QA-accepted** (HISTORY.md). Baselines: pytest 156 / query suite 149/149.

### — Milestone M2 (GraphRAG) — ✅ DELIVERED (K-008/K-013/K-014/K-015 → HISTORY.md 2026-07-08) —

### K-008 — GraphRAG retrieval core (✅ delivered 2026-07-08 → HISTORY.md — M2)

> **Re-scope:** the old K-008 bundled the web client and the AI participant. Those are split out —
> web request/response polish → **K-012** (M1), web agent-reply/`isMention` → **K-014** (M2), AI participant +
> `EMITTED` → **K-013**. K-008 is now purely the embedding pipeline + vector-index verification + hybrid
> retrieval read path, split at the graph-dba→tdd gate (mirrors the K-002/K-007 pattern).

- **Owner:** **`graph-dba`** gate (verify vector index @1024, live-verify + PROFILE §6, add `test_queries.sh`
  assertions) → **`tdd-engineer`** impl (embedding worker + repository/services wiring).
- **Inputs/prereqs:** locked M2 stack (Qwen3-Embedding-0.6B, `EMBEDDING_DIM=1024`); a **devops prerequisite spike** —
  verify LM Studio `/v1/embeddings` reachable from WSL2 and returns 1024-dim vectors (reuse the severino WSL2↔LM Studio
  path). K-011 not required (parallel). Note: the §6 vector DDL already exists in `bootstrap_schema.sh:171-177` —
  the work is "create workspaces @1024 + verify the ANN query plans," not new DDL.
- **Scope:**
  1. **graph-dba gate:** create a workspace `EMBEDDING_DIM=1024`; live-verify §6 ANN query + the embedding-set query;
     `GRAPH.PROFILE` the ANN query; add `test_queries.sh` assertions for §6 (ANN retrieval + `SET m.embedding`),
     pushing the suite past 126 (enumerate the new count). Fold the 1024 default note into `bootstrap_schema.sh` /
     `start_server.sh` guidance (default stays 1536 with the choose-before-creation comment, per K-007).
  2. **tdd impl:** async embedding worker → LM Studio `/v1/embeddings` (decoupled from the post path, DESIGN §9);
     `repository.set_embedding` (1:1 §6 set query); `repository.hybrid_search` (1:1 §6) + `services.hybrid_search`
     passing a **service-layer `timeout=` constant** on the `ro_query` (K-007 TIMEOUT posture, §10) — not per-call
     ad-hockery. LLM/embedding HTTP client injected/mockable.
- **Done-condition:** query suite green at the new gate baseline (≈126 → ~135, enumerated in the gate); pytest green
  with worker + repo/service tests; message posted → embedding lands out-of-band → hybrid search returns it ranked
  by cosine distance `ASC`. `Entity` expansion verified to no-op cleanly (no `Entity` nodes yet — see note).
- **Risks/RAM (rule 6):** **the dominant new RAM line** — the 1024-dim vector index is ~**12.5 KB/message ≈ 1.25 GB
  per 100k-msg workspace** (empirical §11). Call it out per workspace. `GRAPH.MEMORY USAGE` under-reports vector
  memory (§11 caveat) — size from `INFO memory` deltas. Keep LM Studio latency off the write path (async worker).
- **Test strategy:** repository tests against isolated `ws:test` @1024 with a stub embedder (deterministic vectors)
  for ranking assertions; one live check against real LM Studio behind a marker; PROFILE assertion in `test_queries.sh`.
- **NOTE — `Entity` extraction is OUT OF SCOPE for M2.** No entity-extraction pipeline exists; the §6
  `MENTIONS→Entity` expansion is an `OPTIONAL MATCH` that no-ops cleanly, so M2 GraphRAG = vector-ANN + thread-scope
  without it. Entity extraction is parked (M3-adjacent, see Parking lot).

### K-013 — AI `Agent` participant with `EMITTED` provenance (✅ delivered 2026-07-08 → HISTORY.md — M2)

- **Owner:** **`graph-dba`** gate (author + verify the `EMITTED` provenance write + any read surfacing it; add
  `test_queries.sh` assertions) → **`tdd-engineer`** (the responder service). `cobb` consult only if later exposed as an MCP tool.
- **Inputs/prereqs:** K-008 (hybrid retrieval) + K-010 (namespace-unique member ids — real `Agent` identity wired
  without shadowing) + `ensure_agent` v2 (§7, live). LM Studio `/v1/chat/completions` (Qwen3-4B-Instruct-2507) reachable.
- **Scope:** a server-side responder that, on a triggering message (agent `@mention` / new question in a channel the
  agent belongs to), runs K-008 hybrid retrieval, calls the LLM with retrieved context, and **posts the answer as the
  `Agent`** (role `assistant`, via the existing §4 write path — K-007 agent authorship is in) with a **new `EMITTED`
  edge** from the answer message to its provenance (seed messages / retrieval context). graph-dba defines `EMITTED`'s shape.
- **Done-condition:** query suite green at the new gate baseline; pytest green with responder tests (LLM + embedder
  mocked); live check — a user question in a seeded channel yields an agent-authored answer reading `role:"assistant"`
  on all read surfaces (K-007 invariant) with a queryable `EMITTED` provenance edge.
- **Risks/RAM (rule 6):** one `EMITTED` edge + one answer `Message` (with its own embedding once K-008 embeds it) per
  answer — **negligible vs. the K-008 vector line**; count the new relationship type. LLM latency/failure must not
  corrupt the thread — the LLM call precedes the guarded §4 write; failure = no post. **Trigger must exclude
  agent-authored messages** (no self-answer feedback loop).
- **Test strategy:** unit — responder with mocked retrieval + mocked LLM (deterministic answer); contract — the
  `EMITTED` write in `test_queries.sh`; one live smoke behind a marker.

### K-014 — Web M2: render agent replies + reader `isMention` highlighting (✅ delivered 2026-07-08 → HISTORY.md — M2)

- **Owner:** `coder` (same web-JS-no-harness justification as K-012).
- **Inputs/prereqs:** K-012 (polling base) + K-013 (agents actually posting). Uses the since-read `isMention` flag (§9,
  already server-side).
- **Scope:** render agent-authored (`role:assistant`) messages distinctly; restore reader `isMention` highlighting via
  the since-read flag (the K-005 "dead highlight" is alive once polling drives the UI); surface agent answers as they
  arrive via the K-012 poll loop. **Fold-in from K-012:** polled (`?since=`) message rows currently carry `authorId`
  but no `displayName` (a `coder` left a code comment in `web/app.js`) — resolving it needs a small server change to
  include `displayName` on since-read rows; it belongs to this K-014 web-M2 pass.
- **Done-condition:** manual checklist — an agent answer appears in the polling web UI styled as assistant; a message
  mentioning the reader is highlighted. Suites untouched (110 / 126/126).
- **Risks/RAM:** none (client-side).
- **Test strategy:** manual smoke against a running server with the K-013 responder live.

### K-015 — QA acceptance pass on M2 GraphRAG (✅ delivered 2026-07-08 → HISTORY.md — M2 · PASS, zero defects)

- **Owner:** `qa-engineer`.
- **Inputs/prereqs:** K-008 + K-013 + K-014 landed.
- **Scope:** black-box acceptance pass on the GraphRAG loop — embedding lands out-of-band, hybrid retrieval ranks
  correctly, the agent participant answers with provenance, the web UI renders it. Versioned test plan + report per repo
  convention (`docs/test-plans/`, `docs/test-reports/`). **Explicitly notes** the still-deferred transport-level
  agent-actor path (carries the K-007 QA carry-over forward to K-017) since auth isn't in yet.
- **Done-condition:** `docs/archive/test-plans/m2-graphrag.md` + `docs/archive/test-reports/m2-graphrag-report.md`; PASS (or
  PASS-with-parked-defects) on green baselines; isolated `ws:qa` (create + delete), `ws:acme`/`reference` untouched.
- **Risks/RAM:** none (no code under test changed); budget the transient `ws:qa` @1024 vector index.
- **Test strategy:** the pass itself; drives REST + MCP + the running responder.

> **K-019 — Documentation-inconsistency sweep — delivered ✅ 2026-07-05** (doc-only; moved to
> HISTORY.md). Reconciled stale test counts (110 / 126/126) in README/DESIGN, closed the §13
> embedding "still open" drift (now points to the §1.3 decision), and aligned §14.1/README
> real-time wording to M2.5. Counts sourced from a live suite run.

### — Deferred M2.5 hardening track (auth + real-time; not on any M2-green path) —

### K-016 — Real auth/tenancy replacing the hardcoded `get_context` seam (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design pass — designs the auth mechanism *per* the authoritative-identity decision, now
  resolved: the `identity` graph is authoritative/standalone, DESIGN §1.2) → **`tdd-engineer`** (implement the resolved `get_context`).
- **Inputs/prereqs:** the identity source-of-truth is **decided** (identity graph authoritative/standalone; DESIGN §1.2) —
  K-016 no longer needs the user for that axis; it implements per that decision. Localized by design — only
  `config.get_context` changes (`config.py:43`); everything below already parameterized on `ws`/`actor`.
- **Scope:** token → (user, workspace claim) resolution replacing hardcoded `ws=acme/user=u1`; wire the `identity`
  graph per the §1.2 authoritative-identity decision; keep or replace MCP's `frm`-ignoring rule with authenticated agent identity.
- **Done-condition:** `get_context` resolves a real principal from a credential; multi-tenant isolation test; pytest green.
- **Risks/RAM:** `identity` graph nodes (small). First real trust boundary — MCP endpoint is currently unauthenticated (§15.3).
- **Test strategy:** service/api tests with injected auth contexts; a cross-tenant isolation test.

### K-017 — Transport-level agent-actor path (K-007 QA carry-over) (🔵 proposed — M2.5, deferred · depends on K-016)

- **Owner:** `qa-engineer` (+ small `tdd-engineer`/`coder` fold-in if MCP must express an authenticated agent actor).
- **Scope:** with auth able to express an *agent* principal, drive an external agent authoring over MCP/REST (the M1
  hardcoded seam couldn't) and verify authorship/role/provenance end-to-end.
- **Done-condition:** the K-007 QA carry-over closed — a report showing an externally-authenticated agent authoring
  first-class over the transport.
- **Risks/RAM:** none new. **Test strategy:** black-box over MCP with an agent credential.

### K-018 — Real-time push (Redis Pub/Sub → WebSocket/SSE) (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design: Pub/Sub fan-out topology; resolve the DESIGN §13 Bolt-vs-RESP gateway question
  here since it touches the transport) → **`coder`/`tdd-engineer`**.
- **Inputs/prereqs:** K-012/K-014 web client (swap polling → push).
- **Scope:** Redis Pub/Sub on message write → WebSocket/SSE endpoint on the same FastAPI process (§14.1: "slots onto
  the same service layer, no schema change") → web client subscribes instead of polling.
- **Done-condition:** a posted message appears in another client without a poll; graceful fallback to polling.
- **Risks/RAM:** no graph RAM; Pub/Sub is transient. Publish *after* the guarded §4 write commits, never inside it (atomicity rule).
- **Test strategy:** integration test of publish-on-write + a WebSocket client receiving it.

### — M2.5-quality track (retrieval evaluation; parallel to M2.5 hardening, off the M3 critical path) —

### K-026 — GraphRAG retrieval + generation evaluation harness (🔵 proposed — M2.5-quality)

- **Owner:** **`data-scientist`** method note ✅ (`docs/plans/graphrag-eval-ml.md`) → **`coder`/`tdd-engineer`**
  (harness + golden-set fixture) → **`graph-dba`** only if a retrieval query change is later measured through it.
- **Inputs/prereqs:** M2 GraphRAG ✅ (K-008/K-013). A representative corpus — build a seeded **`ws:eval`**
  (step 0; `seed_demo.sh` is too thin). Local LM Studio for the (optional) judged generation layer.
- **Scope:** (1) 30–50 **paraphrased**, human-verified `query→relevant_msgId` golden pairs
  (`server/tests/eval/golden_retrieval.jsonl`); (2) retrieval eval over `hybrid_search` — **recall@10** (primary),
  recall@5, MRR — **establishing the vector-only @1024 baseline**; (3) thin **LLM-as-judge** faithfulness +
  answer-relevance layer over ~15–20 Q&A, **calibrated against ~10 human labels before its numbers are trusted**;
  (4) a metrics report the K-025-style QA pass can read. Behind a live marker; network-free baseline stays green.
- **Done-condition:** baseline recall@10/recall@5/MRR recorded; harness re-runnable; judge–human agreement reported;
  golden set asserts no verbatim self-retrieval; both suites green.
- **Why now:** it's the **prerequisite baseline** for un-parking Entity extraction, hybrid fusion, a seed-relevance
  threshold, or any embedding-model swap — today those would ship unmeasured. Also unblocks two cheap tracked
  quality fixes: a **seed-distance cutoff** (drop distractor seeds) and resolving the **grounding-permissive system
  prompt** — each measurable against this baseline.
- **Risks/RAM:** transient `ws:eval` @1024 vector index (budget per K-008's ~12.5 KB/msg line); no production RAM.
  Corpus representativeness + local-judge validity are the methodology risks (see the method note).
- **Test strategy:** deterministic retrieval metrics (no judge) as the core; calibrated judged layer as an overlay;
  golden-set fixture versioned and test-only (leakage guard).

## Recommended plan docs (author when each item is picked up — not yet created)

| Path | Scope |
|---|---|
| `docs/archive/test-plans/m3-workflow-engine.md` + `docs/archive/test-reports/m3-workflow-engine-report.md` | **Created ✅ 2026-07-21** — the K-025 M3 acceptance pass: risk-based plan (written before execution) + report (verdict **PASS with parked, model-gated limitations** ⇒ M3 ✅). |
| `docs/archive/plans/m3-workflow-engine.md` | **Created ✅ 2026-07-09** — M3 decomposition (Part A, K-020…K-025) + slice-1 plan (Part B). Coordination log: `m3-workflow-engine-coordination.md`. |
| `docs/archive/plans/m3-executor.md` | **Created ✅ 2026-07-10** — K-022: run/step-run executor + the §13 guard-language decision. §8 is the seeded triage def (kept in sync with the reverted `seed_workflows.sh`); §2.2 carries the D16 tool-error rule. Coordination log: `m3-executor-coordination.md`. |
| `docs/archive/plans/m3-executor-landing2.md` | **Created ✅ 2026-07-12** — K-022 Landing 2 design patch: U11 trigger wiring, Option-B `PRODUCED` linking, the M-1 fault net. |
| `docs/archive/plans/m3-process-flow.md` | **Created ✅ 2026-07-19, v2.1 approved 2026-07-20** — K-024's second half: the LLM-free `kind:'process'` proof flow (park-and-branch, the `cmp` guard family, typed step handlers, start-without-trigger + the input endpoint, the `access-request@v1` def in §4). Coordination log: `m3-process-flow-coordination.md`; gates: `docs/archive/reviews/m3-process-flow.md`. |
| `docs/archive/plans/m3-guard-thread-context.md` | **Created ✅ 2026-07-15** — the Defect-A design (guard thread-context seam; ~40 lines, zero graph change, `_drive_loop` untouched by construction). |
| `docs/archive/plans/m3-guard-calibration.md` | **Created ✅ 2026-07-16** — K-027 item 3: the judge-calibration protocol (D9 gate = false-advance ≤ 10% ∧ advance-recall ≥ 0.80; D10 small-n caveat mandatory). |
| `docs/archive/plans/m3-capability-probe-ml.md` | **Created ✅ 2026-07-19** — the D13 fits-16GB Qwen3-4B-vs-Ministral-3B comparison + its run results (no model swap). |
| `docs/plans/local-model-ram-budget-ml.md` | **Created ✅ 2026-07-18** — local-model RAM budget for the downgraded 16GB host (what fits alongside FalkorDB + the co-resident embedder). |
| `docs/archive/plans/m2-graphrag.md` | K-008 re-scoped: embedding worker + vector-index-@1024 verification + hybrid retrieval read path. |
| `docs/archive/plans/m2-agent-participant.md` | K-013: `EMITTED` provenance edge + LLM responder posting as the `Agent`. |
| `docs/plans/m1-hardening-loadtest.md` | K-011: append-path load harness + hot-read PROFILE targets + per-workspace RAM budget. |
| `docs/plans/m2-auth-tenancy.md` | K-016 (deferred): real auth replacing `get_context`, per the §1.2 identity-authoritative decision. |
| `docs/plans/m2-realtime.md` | K-018 (deferred): Pub/Sub → WebSocket/SSE, resolving §13 Bolt-vs-RESP. |
| `docs/plans/graphrag-eval-ml.md` | **Created ✅ 2026-07-10** — K-026 (M2.5-quality): retrieval + generation eval harness (golden set, recall@k/MRR, calibrated LLM-as-judge faithfulness). |

## Parking lot / ideas

- **`Entity` extraction pipeline** (M3-adjacent) — build the `MENTIONS→Entity` corpus so the §6 hybrid query's entity
  expansion becomes live (today it's an `OPTIONAL MATCH` no-op). Enables entity-anchored GraphRAG; watch the `Entity`
  supernode risk (DESIGN §5.4).
- Verify the K-009 GitHub Action goes green on first push (path-filtered `.github/workflows/falkor-chat.yml`; FalkorDB
  service container). Note the CI baseline echoes in its comments (75/92) predate K-007/K-010's 110/126 — the suites
  themselves are the source of truth. (K-019 fixes the README/DESIGN body numbers; the CI comments are separate.)
- File upstream FalkorDB issues (K-007 OQ6, recommended to the user): `GRAPH.MEMORY USAGE` under-reports vector-index
  memory; one-shot instant-timeout anomaly after a long override run.
- Per-endpoint response schemas (QA, recommended three times now): full-thread / since-reads / search each carry a
  different field subset (all documented/intentional) — a declared schema per endpoint would make the contract testable
  and stop accretion.
- DESIGN §13 remaining open questions — resolve as their milestones arrive: workflow guard expression language (M3),
  real auth (K-016), message/embedding retention, cross-workspace analytics, Bolt vs RESP
  for the gateway (K-018).
- **WSL2 memory cap for the 16GB host** (parked, not applied per user 2026-07-18) — WSL2 runs uncapped at its 8GB
  default (50% of the 16GB host) with `autoMemoryReclaim` off, overcommitting host RAM alongside Windows-side LM Studio;
  likely root cause of the recent memory-overload crashes. Parked fix: set `memory=6GB` + `swap=4GB` +
  `autoMemoryReclaim=gradual` in `C:\Users\mauri\.wslconfig` (keep `networkingMode=mirrored`), then `wsl --shutdown`.
  Full diagnostic + apply procedure: `docs/plans/wsl2-memory-diagnostic.md`. Un-park (apply) if the crashes recur —
  verdict was confirmed-by-defaults, not reproduced live (FalkorDB was down during the diagnostic).
