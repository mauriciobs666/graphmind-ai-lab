# Backlog — falkor-chat

> Forward-looking backlog for the `falkor-chat` component (formerly `kaizen/plan.md`; item IDs
> keep the `K-` prefix). Delivered work is logged in [`HISTORY.md`](./HISTORY.md); completed
> plan documents stay in place and are marked `Status: archived` (root `AGENTS.md`);
> `archive/` holds frozen documents from the previous convention.
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
| **M3 — Workflows** ✅ | **Reached (2026-07-21)** — def model + snapshot + executor + chat linkage, proven by one conversational + one business-process flow, **QA-accepted**: K-025 verdict **PASS with parked, model-gated limitations**, zero blocking defects (`docs/archive/test-reports/m3-workflow-engine-report.md`) | **K-020 ✅ + K-021 ✅** (slice 1) + **K-022 ✅ + K-023 ✅** (2026-07-19, Landing 1 + 2) + **K-024 ✅** (2026-07-21 — **both** proof flows) + **K-025 ✅** (QA = U15, 2026-07-21) ⇒ **M3 ✅**. K-027 (live-triage reliability), K-028/K-029/K-030 (filed out of K-024), **K-031 ✅** (filed out of K-025; delivered 2026-07-24 — def/snapshot structure read surface) and K-032 (CPG-style data-dependence overlay for publish-time static analysis) are follow-ups, **not** M3-green gates. |
| **M2.5 — Hardening** *(deferred)* | Real auth, transport-level agent path, real-time push | **K-016 → K-017, K-018** |
| **M3.5 — Web API Coverage** ✅ | **Reached (2026-07-29)** — FR-1..FR-10/AC-1..AC-6 wired into `web/` (defs viewer, inline run cue + detail panel, structured-input resume, participants list, ready-to-demo banner), **QA-accepted**: K-036 verdict **PASS with parked/non-blocking limitations**, zero blocking defects (`docs/test-reports/web-api-coverage-report.md`) | **K-036 ✅** (5 waves, 2026-07-28→2026-07-29) ⇒ **M3.5 ✅**. K-037 (`TRIGGER_DEF_KEY` graft bug + banner cosmetic) and K-038 (`refreshRunPanel` overlapping-poll-tick race) are follow-ups, **not** M3.5-green gates. |

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

### K-027 — Live triage reliability + carried gate findings (🟡 in-progress — the D12-B descope from K-022 Landing 2)

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
  1. ✅ **Judge-parse robustness** (**delivered 2026-07-24, slice A** — see
     [`HISTORY.md`](./HISTORY.md)) — `app._build_llm_judge` used `complete()` + a **bare
     `json.loads`**. A model that wraps its JSON in a ```` ```json ```` fence broke *every* verdict
     silently: in the D13 probe Ministral scored **26/26 "unparseable judge output"**, including one
     correct `decision:true` destroyed by the fence. The shipped Qwen path is unaffected (its JSON is
     unfenced), which is exactly why this could rot unnoticed. **Fixed** by parsing the reply with
     `llm.extract_own_line_json_object(…, require_key="decision")`. **The first draft of this slice
     used the permissive `llm.extract_json_object` and claimed "tolerance runs in the safe direction
     only"; the analyst gate proved that claim false** (blocker **B-1**) — being order-blind, that
     extractor lifts a *quoted* verdict straight out of prose, so *"I would answer
     `{"decision": true, …}` but they did not, so I answer false"* **advanced** a guard the old bare
     `json.loads` correctly suspended, and `guards._coerce_verdict` cannot catch it because the
     quoted rationale reads clean. The seam is now split: `extract_json_object` stays permissive and
     keeps the tool-call path (where the agent loop re-validates name + schema);
     `extract_own_line_json_object` is conservative and takes the judge — it accepts a reply that is
     entirely one JSON object (bare or fenced), or **exactly one** `decision`-carrying object that
     owns its lines, and returns None otherwise. **The accurate statement of the property**: prose
     containing no object, an object quoted mid-sentence, and two disagreeing candidates all resolve
     to `decision=False`; tolerance is applied to how a verdict is *wrapped*, never to whether one
     was *asserted*. Residual, accepted (pinned as a characterisation test, gate **N-2**, and named
     in `app._build_llm_judge`'s docstring): a hypothetical or schema-echo verdict on its own line,
     with no second object to disambiguate it, is still read and still **advances**. Note the rationale string for a JSON-but-not-object reply
     changed from `"non-object judge output"` to `"unparseable judge output"` (same verdict,
     different trace text). Structured output remains an option if a future model still misbehaves.
     **Item 3 (calibration) is unaffected and still open** — this fixes the transport of a verdict,
     not its quality, and its false-advance metric (D9) is only meaningful against this settled
     parse.
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
     model-robust (item 1) — **that precondition is now met (slice A, 2026-07-24), so this is
     unblocked**; the re-probe should run against the widened parse layer, not the old one. Notes:
     `docs/archive/plans/m3-capability-probe-ml.md`,
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
  the engine-level guarantee must cover any node whose contract is "post" — **still open, and now
  the *only* remaining part of (a): the parse-layer half is delivered** (below); **(b)** ✅
  **delivered 2026-07-24, slice A** — this is the exact structural twin of **item 1** (a parse layer
  intolerant of the shapes small local models actually emit), so `_parse_content_tool_calls` now
  recovers bare `name({json})` / `name()` call syntax as a second probe *after* the JSON probe
  (`llm._parse_bare_call_syntax`; `_parse_chat_message`'s native-over-content precedence unchanged).
  Recognition is deliberately narrow, and the analyst gate tightened it twice after finding the
  first draft's rule materially wider than its docstring: three rules, **all** enforced — the call
  opens a line (indentation and a space before the paren allowed), its argument is empty or a JSON
  **object**, and **from the first accepted call onward the message holds nothing but calls and
  whitespace** (no prose *between* two calls, nothing after the last one). That third rule is the
  fix: without it a call the model was merely *quoting* inside a fenced example fired and dispatched
  a real thread write, while `text=None` discarded the model's actual answer. Gate **M-1** anchored
  it on the *last* call only, which gate **N-1** re-opened — `executor._run_agent_node` dispatches
  **every** returned call, so *"I considered handing off:\n```\nhuman_handoff()\n```\nBut instead I
  will ask:\npost_message({…})"* dispatched both, and the whole M-1 family came back whenever the
  model ended on a genuine call (the common shape). The current rule is the N-1 form. Inline
  mid-sentence calls, trailing prose on the call's line **or any later line**, prose *between* two
  call-shaped expressions, and keyword/positional arguments
  stay text, because a dispatched call the model never made is worse than the miss. Identical
  repeated calls collapse to one dispatch (gate **m-5**). **Residuals, accepted** (position alone
  cannot separate either from an intended call; both pinned as characterisation tests, and closing
  them needs the granted tool names as a recognition filter — remedy 3 of **K-035**): a message
  whose final line happens to be an illustrative call still fires, and a *contiguous catalogue* of
  own-line calls with no prose between them still fires. This **would have converted the observed
  shape as reconstructed (de-wrapped)** — the shape *as recorded* in the report is line-wrapped and
  a raw newline inside the JSON string still leaves it unparseable (bullet 4 below), and the
  stronger "would have converted the observed run" is *not* established
  and should not be relied on (gate **m-4**): the only surviving record is the report's
  line-wrapped, `…`-elided quotation, the run's real `StepRun.output` is no longer in any live graph
  (checked: zero matches across `ws:acme`, `ws:test`, `reference`), and bullet 4 below records that a
  raw newline inside the JSON string argument would have left it unparseable. Item 5's re-probe
  should not treat the claim as a proven precondition. The **engine-level guarantee (item 2)
  is still owed, and item 5's re-probe should now re-measure against this parse layer.** Also
  recorded: `pytest -m live` is **RED deterministically (2/2)** on
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
- **Discovered during slice A, deliberately not fixed** (candidate follow-ups, small):
  - **Loose-JSON shadowing of a bare call → promoted to its own item, K-035 (gate M-2).**
    It stays deferred, but it is no longer a bullet: the failure manufactures a call named after a
    **user-supplied value**, which is not a "small" shape even at low likelihood.
  - **Namespaced call names** (`functions.post_message({…})`) are not recognised — the identifier
    pattern is deliberately un-dotted. No evidence any local model emits this shape here.
    (Re-verified at the gate: still accurate.)
  - ~~**Fences with a non-`json` language tag** (```` ```tool_code ````) are not stripped~~ —
    **withdrawn at the gate (m-3): true of the tag text, but the stated consequence does not hold, so
    the bullet sent the next author at a non-issue.** The backticks *are* stripped; the leftover tag
    line is harmless to both consumers. Re-verified directly: ```` ```tool_code\npost_message({"text":
    "hi"})\n``` ```` → **recovered as a call**, and ```` ```tool_code\n{"decision": true,…}\n``` ````
    → **recovered as a verdict**. Nothing to fix here.
  - **Raw newlines inside a JSON string argument** make the argument unparseable ⇒ the call stays
    text. Tolerating them would mean a lenient JSON reader, not a parse-order change. (Re-verified at
    the gate: still accurate — and it is why the "would have converted the observed run" claim above
    could not be substantiated.)
- **Test strategy:** offline pins for the parse-robustness + negator fixes (fenced-JSON and
  clause-boundary cases as fixtures) — **the parse half is delivered** (30 pins in
  `server/tests/test_llm.py` + `test_app.py`; 19 in the first draft, 11 added at the analyst gate,
  **19 of the 30 red before their fix**). **17 of the 30 now guard the false-positive direction**,
  and the gate's own lesson is recorded here for the next author: the first draft's 6 negative pins
  were *all single-line*, which is precisely why three multi-line false positives shipped
  undetected — **a negative pin corpus for a line-anchored rule must include multi-line shapes.**
  The negator (m-1) half is not delivered; a `live`-marked reliability run for the terminal-post
  contract with an explicit n and no cherry-picking; the calibration harness reading
  `golden_guards.jsonl` per the §4/§7 protocol, reporting both arms with the D10 caveat attached.

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

### K-031 — Def/snapshot **structure** read surface (make the create-only split-brain detectable) (✅ **delivered 2026-07-24** — plan v2 + analyst re-gate → HISTORY.md)

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

**Delivered 2026-07-24** — plan `docs/plans/workflow-def-structure-read.md` **v2** (analyst re-gate:
*approve with suggestions*, all 15 round-1 findings closed), review
`docs/reviews/workflow-def-structure-read.md`. Three read-only REST routes
(`GET /workflow-defs/{key}/versions/{version}`, `GET /workspaces/{ws}/snapshots/{key}/versions/{version}`,
and `…/diff`) over the **existing** `_READ_META_CYPHER`/`_READ_TRANSITIONS_CYPHER` constants — **zero new or
modified Cypher**, `test_queries.sh` unchanged at **256/256** (the plan's no-new-Cypher tripwire) — plus
`scripts/verify_workflows.sh <wsId>`, the read-only one-command form of the re-seed discipline.
- **`maxSteps` off-by-one → DOCUMENTED, not fixed** (binding stakeholder decision). `executor.py:410`/`:427`
  are untouched, the `_drive_loop` SHA lock (`71055f756280`) is intact, `tests/test_executor.py:158` keeps its
  assertion. The semantics ("a tripwire checked *after* each recorded step ⇒ at most `maxSteps + 1`; not
  checked on the park or terminal paths") are now stated at DESIGN §6, QUERIES §12.5 + the two `$maxSteps`
  comments, `schemas.py` and `AGENTS.md`'s executor-invariants block. The real fix is filed as **K-033**.
- **Cross-reference: this read surface is the *detection* mechanism for K-034**, not its fix. K-031 makes the
  current publish semantics **observable** and deliberately changes nothing about them; the additive-`MERGE`
  finding and the **thirteen** doc assertions it falsifies are **K-034's** (its table below carries all
  thirteen — three were added at the K-031 implementation gate so its done-condition covers them).
- **Multi-`START` finding, verified live (V-1):** two `START` edges on one root make QUERIES §11.2's one-row
  collapse fail — the meta query returns **one row per distinct start key** (falkordb/falkordb:v4.18.11), so
  `_read_subgraph`'s `result_set[0]` is arbitrary. The new `repository._read_structure` reads **all** rows and
  surfaces `startKeys`; `verify_workflows.sh` treats a `startKeys` list as a failure. Recorded in QUERIES §11.2/§11.5,
  and **pinned in the suite** by a fake-graph unit test replaying V-1's two-row `result_set` — no publish, no
  Cypher, no FalkorDB, so no coupling to K-034's publish semantics (gate finding M-1).
- **Live state (R-1):** `verify_workflows.sh acme` reports **both** `triage@v1` and `access-request@v1`
  **in sync** with one start key each — no live divergence found, nothing repaired.

### K-032 — Materialize the workflow def's **data-dependence overlay** (CPG-style READS/WRITES) for publish-time static analysis (🔵 proposed — from a design conversation, 2026-07-22)

> **The framing (Code Property Graph lens).** The def graph is already a control-flow graph:
> `(:Step)-[:TRANSITION {guard, order}]->(:Step)` is a guarded CFG, `HAS_STEP` is one-level AST
> containment, and `(:StepRun)-[:NEXT]->` is an executed CFG path. **What's missing is the
> data-dependence layer (DDG):** which `ctx`/`output` keys each step *reads* (via its `cmp`/`llm`
> guards and, for a `decision`/`human`, the keys it branches on) and which it *writes* (a `human`/
> `wait` step's `config.expects`, a step's declared outputs). That information is **not missing —
> it's trapped inside the opaque `guard`/`config` strings**, and `services._validate_def_spec` +
> `guards.validate_cmp` already walk the `cmp` guard tree at publish, so ~90% of the extraction pass
> exists and is currently thrown away. Materialize it as real edges and three otherwise-impossible
> checks become one-hop Cypher.
- **Why it's worth doing (the payoff).** Publish-time (not live-run) detection of:
  1. **Dangling read** — a guard reads a `ctx`/`output` key no upstream step writes. This is exactly
     the **un-enforced n-3 hazard** AGENTS.md documents (a `decision` step with all-conditional
     outgoing transitions and no `waitsForHuman` **self-loops to budget exhaustion**) — today a live
     discovery, turned into a `WorkflowDefSpecError` at seed time. Overlaps K-029's symmetric-invariant
     proposal; this is the graph-shaped way to get there.
  2. **Unreachable step / dead branch** — plain CFG reachability from `START`.
  3. **Change-impact / blast radius** — "I changed `submit`'s output shape; which downstream guards
     read it?" This matters **specifically because published defs are create-only + immutable**
     (K-031): a def edit costs a version bump + snapshot republish + a `reference`↔`ws:{id}`
     split-brain risk, so knowing the blast radius *before* the bump has real value here.
- **Hard constraints (fall out of locked decisions — non-negotiable in any plan):**
  - **Derive at publish, never parse in Cypher.** Rule 8 (`ctx`/`config`/`guard` opaque, never
    filtered in Cypher) holds *iff* publish is treated as a compile step — `joern-parse` builds
    overlays once, queries traverse edges; same contract. Extraction runs app-side in the publish
    validator; only the resulting edges hit the graph.
  - **Overlay edges built inside `_PUBLISH_CYPHER` and `materialize_snapshot`, same query.** A
    separately-written overlay on a `MERGE … ON CREATE SET` def is a **new split-brain axis** on top
    of the `reference`-vs-snapshot one — and per the K-030 note the materialize path still `IndexError`s
    after a partial write, so the overlay must ride the existing atomic publish, not a follow-up write.
  - **Static-only, on the def — never on `StepRun`.** The def graph is tens of nodes (overlay edges
    are single-digit multiples → RAM non-issue, the inverse of a repo CPG where AST/CFG/REACHING_DEF
    fan-out dominates). The run graph is thousands of nodes and RAM-bound; "why did *this run* branch
    here" is a join through `RAN`, not a second copy of the layer.
  - **Honest `READS_UNKNOWN` for what can't be derived statically** — an `agent`/`llm`-guard node
    whose reads aren't extractable gets an explicit marker (Joern marks indirect calls the same way).
    A **feature**: it says precisely which parts of a flow are analyzable vs. trust-the-model. Do
    **not** attempt a probabilistic DDG — an unsound dependence edge produces confident-wrong impact
    answers, worse than none.
- **Owner:** **`graph-dba`** gates the FalkorDB model first (the overlay labels/edge types, whether a
  `CtxKey`-style node is per-def or shared, indexes) → **`architect`** designs the publish-time
  extraction pass + the three validations → **`coder`**/**`tdd-engineer`**. A CPG-model design note
  would land at `falkor-chat/docs/plans/<slug>-graph.md` (graph-dba convention).
- **Scope sketch (to be designed, not decided here):** first slice = extract read/write sets from
  `cmp` guard paths (`ctx.`/`output.` roots) + `config.expects` at publish → materialize
  `(:Step)-[:READS]->` / `(:Step)-[:WRITES]->` a key node → add the **dangling-read** and
  **unreachable-step** publish validations (closes n-3 the graph way). `llm`-guard reads and the
  change-impact query are follow-on slices. No DDL beyond one node label + its index; no rule-8
  violation; no run-side cost.
- **Relationship to neighbours:** complements **K-031** (that exposes def *structure* for reading;
  this *analyzes* it), and overlaps **K-029**'s symmetric-`decision` invariant (K-029 proposes the
  rule; K-032 proposes the graph mechanism that could enforce it). Not an M3-green gate — M3 is ✅.
- **Risks/RAM:** negligible on the def side (see the static-only constraint). Real risk is *scope
  creep* toward a general expression/data-flow engine — the `expr` seam stays a `NotImplementedError`;
  this rides the existing closed `cmp` family only.
- **Test strategy:** offline contract tests — a published def reads back with the exact READS/WRITES
  overlay for its guards/`expects`; a def with a guard reading an unwritten key is **rejected at
  publish**; an unreachable step is **rejected at publish**; a step whose reads can't be derived
  carries `READS_UNKNOWN` rather than silently claiming zero reads.

### K-033 — Make `maxSteps` an exact cap (`>` → `>=` in `_drive_loop`) (🔵 proposed — filed out of K-031, stakeholder decision OQ-1 "document now, fix later", 2026-07-24)

> **Why it exists.** `maxSteps` does not mean what its name says. `executor._drive_loop` records a
> step and *then* checks `rec["stepCount"] > max_steps` — at `executor.py:410` (OUTCOME A, a guard
> fired) and `:427` (OUTCOME C, a legitimate self-loop). With `maxSteps: 2`: step 1 → `1 > 2` false;
> step 2 → `2 > 2` false; **step 3 runs**, `3 > 2` → fail. So the budget means *"at least N, then one
> more"*, and a run executes at most **`maxSteps + 1`** steps. Confirmed by reading and **pinned by a
> passing test** — `tests/test_executor.py:158`, `assert len(trail) == 4  # maxSteps=3 → the 4th
> advance trips the guard`. Harmless for the two proof defs (8/6/6 steps against `maxSteps: 24`), but
> it makes `maxSteps` unusable as an SLA or a cost budget, which is exactly what a caller reaches for
> it for. K-031 shipped the **documentation** of the real semantics at six sites (DESIGN §6,
> QUERIES §12.5 + the two `$maxSteps` comments, `schemas.py`, `AGENTS.md`'s executor-invariants
> block) per the binding stakeholder decision; this item is the **fix**.
- **The change is two characters, in two places** — `>` → `>=` at `executor.py:410` and `:427`.
  Everything else about this item is ceremony, and the ceremony is the reason it was deferred out of
  an observability slice rather than the difficulty of the edit.
- **Both sites are *inside* the SHA-locked `_drive_loop`** (`71055f756280`,
  `docs/archive/plans/m3-process-flow.md` §3.1). Landing it therefore costs:
  - a lock break + **re-lock ceremony**: recompute the SHA, then re-quote it in `falkor-chat/AGENTS.md`,
    `docs/BACKLOG.md` (×2), `docs/HISTORY.md` (×2) — grep `71055f756280`, the line numbers drift;
  - **frozen archive documents that must not be rewritten** — `docs/archive/plans/m3-process-flow.md`
    (×4), `docs/archive/reviews/m3-process-flow.md` (×5),
    `docs/archive/plans/m3-executor-coordination.md` (×3), `m3-process-flow-coordination.md`. These
    are *historical records* asserting the SHA was unchanged throughout K-024. The re-lock has to be
    expressed as *"as of K-033 the lock is `<new>`; archived records quote the pre-K-033 value"* —
    i.e. the lock stops being a single grep-able constant. **Decide that framing before editing.**
  - **test edits**: `tests/test_executor.py:142-158` (the pinned count 4 → 3, and its explanatory
    comment), plus a sweep of `tests/test_process_flow.py`'s step accounting and the
    `access-request@v1` `maxSteps: 24` headroom;
  - **behavioural blast radius**: every existing run's effective budget shrinks by one.
- **Bundling is a PREFERENCE, not a precondition — and its premise is UNVERIFIED.** It would be
  pleasant to land this alongside the next item that legitimately breaks the `_drive_loop` lock,
  plausibly **K-027 item 2** (the terminal-node-must-post engine contract): one re-lock ceremony,
  two fixes. But *"K-027 item 2 must break the lock anyway"* is an **assumption, never established**
  — the lock covers `_drive_loop` **only**, and `_execute_step`, `_select_transition`, `_trace_step`
  and `resume` sit **outside** it (`AGENTS.md`, executor-invariants block), so a terminal-post
  guarantee might well be implementable at one of those seams. K-027 is itself **🟡 in-progress**
  — slice A delivered 2026-07-24; items 2–5, **item 2 included**, remain open and unscheduled.
  **K-033 therefore does not depend on K-027**: if no such item arrives, K-033 breaks
  the lock on its own — the ceremony is the cost either way, and waiting only leaves the honest
  `maxSteps + 1` prose sitting in six documents indefinitely, which is the cheap half of a permanent
  divergence.
- **Also decide (part of the item, not a separate one):** whether the *park* path (OUTCOME B,
  `executor.py:415-421`) and the terminal path stay deliberately unchecked. They are unchecked today
  by design — a parked run cannot self-drive — and K-031's documentation says so explicitly, so
  changing that is a second semantic decision, not a consequence of `>=`.
- **Owner:** **`tdd-engineer`** (a behaviour change pinned by an existing passing test is the
  test-first shape: flip the assertion red, then the operator) — with an **`architect`** call first
  **only** on the re-lock framing above, if the coordinator wants the archive-document question
  settled before code moves. No `graph-dba` gate: no Cypher, no DDL, no index.
- **Risks/RAM:** **none** — no new node type, label, property, index or vector dimension; no query
  changes. The risk is purely behavioural (every run's budget shrinks by one) and procedural (the
  lock/archive framing).
- **Test strategy:** flip `tests/test_executor.py:142-158` to assert exactly `maxSteps` advances
  (3, not 4) and watch it go red before the fix; add the boundary case at `maxSteps = 1`; assert the
  park path is still **not** budget-checked (a parked run at `stepCount == maxSteps` must stay
  `waiting`, not fail); re-run `tests/test_process_flow.py` and the `access-request@v1` acceptance
  flow to confirm the `maxSteps: 24` headroom still covers it.

### K-034 — Create-only re-publish is *additive*, not a silent no-op — duplicate `TRANSITION`/`START` edges, and the thirteen doc sites that say otherwise (🔵 proposed — discovered by `architect` while designing K-031, confirmed by `analyst` at the K-031 plan gate, 2026-07-24)

> **Why it exists.** The whole component is documented on the claim that re-publishing the same
> `key@version` is a **structural no-op** — "immutability per version comes for free from `MERGE`".
> That claim is **false**, and the shipped docs, docstrings and `AGENTS.md` all assert it.
> `repository._PUBLISH_CYPHER` (`server/falkorchat/repository.py:937-956`) is `MERGE … ON CREATE SET`,
> which is create-only on **properties** but whose `MERGE` **patterns still create structure**: a new
> step key mints a `Step` + `HAS_STEP`; a changed `to`/`on`/`order` mints a **parallel `TRANSITION`**
> edge beside the old one (the MERGE key is `{on, order}` *plus* the `(from)→(to)` endpoints, so a
> retargeted transition is a *different* edge, not an update); a changed start step mints a **second
> `START`** edge. Only `guard`, `d.name`, `d.kind` and `st.type`/`st.config` are genuinely create-only.
> `materialize_snapshot` (`repository.py:1540-1566`) formats the **same constant** at `:1555` against
> the workspace graph, so the snapshot side is additive too — and the snapshot is what the executor
> drives. **Full evidence, do not re-derive it:**
> [`docs/reviews/workflow-def-structure-read.md`](reviews/workflow-def-structure-read.md) **finding
> B-2** (and its closure in the re-gate section, R2 · B-2); formally handed off to this item by
> [`docs/plans/workflow-def-structure-read.md`](plans/workflow-def-structure-read.md) **§0.2** and
> **§1.2**.
- **Two live consequences — this is a defect, not a curiosity.**
  1. **Nondeterministic branching.** `executor._select_transition` orders by
     `sorted(transitions, key=lambda t: (t["guard"] == "", t["order"]))`
     (`server/falkorchat/executor.py:758`) and takes the first firing guard. Two `TRANSITION` edges
     with the same `on`/`order` but different `to` **sort equal**; Python's sort is stable, so the
     branch a live run takes is whatever edge order FalkorDB happened to return. Silent, and it can
     differ between two runs of the same def.
  2. **Run start breaks outright.** `repository.start_run` (`repository.py:1158-1195`) is
     `MATCH (snap:WorkflowDefSnapshot …)-[:START]->(start:Step) … CREATE (r:WorkflowRun {runId: …})`.
     Two `START` edges ⇒ two rows ⇒ the `CREATE` executes twice against
     `UNIQUE NODE WorkflowRun PROPERTIES 1 runId` (`scripts/bootstrap_schema.sh:180`) — the start
     errors. `start_run_untriggered` (`:1197-1239`, the REST/`kind:'process'` start path) carries the
     identical shape, so **both** start paths are affected, `access-request@v1` included.
  Both are reachable by the operation `AGENTS.md` actively instructs ("re-run `seed_workflows.sh`
  after `test_queries.sh` or a pytest run") combined with a def edit.
- **Blast radius — thirteen shipped assertions are falsified.** Verified against the tree on
  2026-07-24 (line numbers drift; grep the quoted text):

  | # | Site | The falsified claim |
  |---|---|---|
  | 1 | `docs/QUERIES.md:782-784` (§11 preamble) | "re-running the same `key@version` is a structural no-op (0 nodes/rels created). Immutability per version comes for free from `MERGE`" |
  | 2 | `docs/QUERIES.md:829` (§11.1 footnote) | "run 2 → 0 created (idempotent), same row" |
  | 3 | `docs/QUERIES.md:947-948` (§11.4 footnote) | "Snapshots are **immutable** per `(workspace, key, version)`; re-materialize is a no-op" |
  | 4 | `docs/DESIGN.md:544` (§ write paths table) | "Immutable per version; bump version, never mutate in place" |
  | 5 | `docs/DESIGN.md:102` (topology diagram) | "Canonical WorkflowDef templates (versioned, immutable)" |
  | 6 | `docs/DESIGN.md:144` + `:146-149` | "(immutable, versioned)" / "immutable once published" / "the snapshot is immutable so it never drifts" |
  | 7 | `server/falkorchat/repository.py:1065` (`publish_def` docstring) | "re-publishing the same `key@version` is a structural no-op (immutability per version)" |
  | 8 | `server/falkorchat/repository.py:1551-1552` (`materialize_snapshot` docstring) | "Re-materialize is a no-op." |
  | 9 | `server/falkorchat/services.py:808-809` (`materialize_def` docstring) | "Idempotent (the workspace MERGE no-ops on re-materialize)." |
  | 10 | `falkor-chat/AGENTS.md:149` (`seed_workflows.sh` row) | "⚠️ Published defs are effectively IMMUTABLE … re-running changes nothing live" |
  | 11 | `docs/requirements/agent-import.md:81` | "Published workflow defs are effectively **immutable** (`MERGE … ON CREATE SET`); a re-import of a changed def cannot update in place" — **the dangerous direction**: a changed re-import does not *fail to update*, it **additively mutates**. **Load-bearing** — it is the stated collision with that document's **FR-2** idempotence requirement, so an architect designing agent-import off it would design against the wrong hazard. Correct it *and* re-check FR-2's collision handling. |
  | 12 | `docs/requirements/workflow-dependence-overlay.md:21`, `:44`, `:54`, `:160` | "published defs are effectively immutable" / "create-only + immutable (K-031): a def edit costs a version bump" / "before I edit an (immutable) published def" / "by immutable, create-only published defs" — motivational framing, same class as the weaker sites below (`:44` and `:160` found while adding this row; the K-031 report named `:21`/`:54` only) |
  | 13 | `docs/BACKLOG.md:670` (K-032's "why it exists" premise) | "This matters **specifically because published defs are create-only + immutable** (K-031)" — K-032's own premise, so correcting it may shift that item's motivation |

  Sites 11–13 were surfaced by K-031's R-8 classification sweep (11 and 12 reported by `coder`, 13
  by `analyst` at the K-031 implementation gate, `docs/reviews/k031-structure-read-impl.md` **m-3**).
  Deliberately **left uncorrected in place** — correcting them is this item's deliverable, forbidden
  to K-031 by its plan §1.2, and would pre-commit wording before this item decides what publish
  semantics become.

  **Five further sites found while verifying the first ten** (weaker framings of the same claim, cheap to
  fix in the same pass): `docs/QUERIES.md:744` ("versioned, immutable `WorkflowDef` templates"),
  the §11.1/§11.4 headings `docs/QUERIES.md:799` and `:918` ("— idempotent"),
  `server/falkorchat/repository.py:925` (§11 block comment, "versioned, immutable"), `:933`
  ("Publish/materialize share the same idempotent MERGE shape") and `:1549` (the same phrase in
  `materialize_snapshot`'s docstring, one line above site 8).
  **Do *not* "correct" `docs/QUERIES.md:826-827`** — *"The `TRANSITION` MERGE-key is
  `(from, on, order, to)` so distinct outcomes/orders between the same two steps are distinct edges"*
  is **true**, and is the mechanism; it already contradicts the preamble two paragraphs above it.
  Suggested replacement framing throughout: *"create-only on **properties**, additive on
  **structure** — a re-publish never updates, but it does add."*
- **It undermines K-029's core premise — re-read K-029 before picking it up.** K-029 ("converge the
  seed def sources") declines the `triage@v1` move-during-K-024 on the belief that "a byte-diff
  introduced while relocating a **live** def is **silently swallowed**". Under this finding a typo'd
  step key or a retargeted transition is **additive**, not swallowed — which makes K-029's proposed
  relocation *more* dangerous than K-029 currently believes, and promotes its "before/after equality
  check on the published def subgraph" acceptance criterion from a nicety to the item's load-bearing
  safeguard. K-029's own text is deliberately left unedited by this filing; correcting it is part of
  K-034's deliverable (or K-029's own re-scoping, whichever lands first).
- **Owner:** **`architect`** first — changing publish semantics is a **decision**, not a bug fix, and
  the candidate remedies differ in kind: `ON MATCH SET` (mutable versions — reopens a locked
  decision), a pre-publish structural-equality check that **rejects** a differing re-publish
  (preserves immutability, needs an error contract + a repair story), or an explicit
  delete-and-republish affordance. → **`graph-dba`** gate **if and only if** `_PUBLISH_CYPHER`
  changes (a query-shape change needs a re-`GRAPH.PROFILE` and must keep every `MERGE` backed by a
  uniqueness constraint — `AGENTS.md` schema conventions) → **`coder`**/**`tdd-engineer`** for the
  implementation and the eighteen doc/docstring corrections.
- **Scope:** (1) decide and implement the publish/materialize semantics (the `architect` pass above);
  (2) correct the thirteen falsified sites + the five weaker ones listed here, and fix K-029's premise
  paragraph; (3) whatever detection K-031 has by then shipped becomes the acceptance instrument — do
  not build a second one.
- **Explicitly not in scope:** K-031's read surface (it makes this state **detectable** and
  deliberately neither fixes nor documents it — see its §1.2); K-030's empty-`UNWIND` publish hazard
  (same query, different defect — sequence the two so the second one to land re-PROFILEs); repairing
  whatever live `reference`/`ws:acme` divergence already exists (a destructive shared-state op,
  stakeholder-gated, see K-031 §6 R-1).
- **Risks/RAM:** no new node type, label, property, index or vector dimension ⇒ **zero graph RAM
  impact** (rule 6). The risk is entirely in the publish query: a change there needs a re-`GRAPH.PROFILE`
  to confirm the plan stays index-anchored on `Step.stepUid`/`WorkflowDef {key,version}` and does not
  degrade to a scan (rule 3), and `./scripts/test_queries.sh` must stay green. A reject-on-difference
  remedy also changes a **live REST contract** (`POST /workflow-defs` today returns `201` for an
  edited re-publish), so it needs an API-contract decision, not just a validator.
- **Test strategy:** the missing tests are exactly the ones K-031 was forbidden to write. Offline
  contract tests that (a) a re-publish with a **changed transition `to`** leaves the def with the
  *intended* transition count — not two edges; (b) a re-publish with a **changed start step** leaves
  exactly one `START`; (c) a re-publish with a **new step key** behaves per the chosen decision;
  (d) the same three on `materialize_snapshot` (shared constant ⇒ shared defect); (e) an executor
  test pinning that a def with duplicate outgoing transitions is either impossible or deterministic —
  today `_select_transition`'s tie is unpinned in either direction. Plus a grep-verified done-condition
  over the doc surface: from `falkor-chat/`, `grep -rn -i "immutab\|no-op" docs/ server/falkorchat/ AGENTS.md`
  classifies every hit as corrected / correct-as-is / unrelated.

### K-035 — An argument key named `name`/`action`/`tool` shadows the bare call's own name (🔵 proposed — filed out of the K-027 slice A analyst gate, finding M-2, 2026-07-24)

> **Why it exists.** In `llm._parse_content_tool_calls` the **JSON probe runs before** the bare-call
> probe, and `_normalize_tool_call` maps `name` / `action` / `tool` loosely. So for a bare call whose
> *argument object* happens to carry one of those keys, the argument is mistaken for the call
> envelope and the real call name is never seen:
>
> | model emits | parsed as |
> |---|---|
> | `create_user({"name": "bob"})` | `ToolCall(name='bob', arguments={})` |
> | `run_tool({"action": "delete"})` | `ToolCall(name='delete', arguments={})` |
> | `x({"tool": "y", "args": {"a": 1}})` | `ToolCall(name='y', arguments={"a": 1})` |
>
> Reproduced by `analyst` at the K-027 slice A gate; the three rows above are verbatim from
> [`docs/reviews/k027-parse-robustness.md`](reviews/k027-parse-robustness.md) **M-2**.
- **Not currently reachable, and that is the whole risk.** No registered tool declares such a
  parameter — `post_message` takes `text`/`mentions`, `graphrag_retrieve` takes `query`,
  `human_handoff` takes `reason` (`server/falkorchat/tools.py`). The premise was verified at the
  gate and **deferring the fix was judged correct** for a slice whose point was not to widen
  precedence. It is filed as its own item because the *shape* of the failure is bad, not because the
  likelihood is high: it manufactures a tool call **named after a user-supplied value**, so it fails
  **silently and misleadingly** — `executor._handle_tool_call`'s AC-6 check rejects `'bob'` as an
  ungranted tool, burns a re-prompt iteration, and tells the model its own argument is not a tool.
  That trace is close to undebuggable, and a bullet under "candidate follow-ups, small" would not
  have surfaced at the moment someone registers `create_user(name)`.
- **Tripwire in place (do not remove).** `server/falkorchat/llm.py` carries a comment at the
  probe-order site naming this item, because that is where the next author will be looking. **Read
  this item before registering any tool with a `name`, `action` or `tool` parameter.**
- **Owner:** **`architect`** to pick the remedy (it is a precedence decision, not a bug fix), then
  **`tdd-engineer`**. Candidates, cheapest first: (1) in `_normalize_tool_call`, skip the loose
  `name`/`action`/`tool` mapping when the surrounding content also matches `_BARE_CALL_OPEN` — a
  partial hardening available today; (2) run the bare-call probe **first**, which reorders the
  content fallback and needs its own regression pass over the JSON shapes; (3) pass the granted tool
  names down as a **recognition filter** — this also closes most of M-1's residual and the
  `Summary ({"a": 1})` class, but it is a real layering decision (`llm.py`'s note that name
  validation belongs to the agent loop is deliberate, and worth keeping unless consciously reversed).
  Open question 3 of the K-027 gate review is exactly this choice.
- **Risks/RAM (rule 6):** none — parse layer only, no node type, index, property or vector dimension.
  No Cypher, no schema, no script.
- **Test strategy:** offline pins in `server/tests/test_llm.py` driving the public `llm.chat(...)`
  seam (never the private probes). Pin all three rows in the table above to the **call's own**
  identifier; pin that a genuine `{"name": …, "arguments": …}` envelope with no surrounding call
  expression still parses as a call (the shape this must not regress); and pin the negative
  direction — an argument object carrying `name` must not resurrect a call the M-1 final-content rule
  rejected.

### K-036 — Web API Coverage — drive the M3 agent/workflow story + a workspace ready-to-demo check from `web/` (✅ delivered 2026-07-29 — verdict PASS with parked/non-blocking limitations ⇒ M3.5 ✅; plan `docs/plans/web-api-coverage.md` v3, analyst-approved Pass 3)

> **Why it exists.** M3 delivered the def/executor/chat-linkage engine end to end (K-020…K-025 ✅)
> and K-031 added the def/snapshot structure+diff read surface, but none of it is reachable from
> `falkor-chat/web/` — there is no in-thread run cue, no run detail panel (status/steps/trace/
> failure), no structured-input form for a parked step, no thread participants list, and no
> workspace ready-to-demo check. `docs/requirements/web-api-coverage.md` (FR-1..FR-14,
> AC-1..AC-6, committed scope FR-1..FR-10/AC-1..AC-6) captures the ask; the plan cross-checked it
> against the current REST surface (`server/falkorchat/api.py`) and found most of the committed
> path is pure UI wiring — **FR-2** (inline run cue) and **FR-8** (thread participants) are the
> genuine server-side read gaps, and **FR-10** (ready-to-demo) needs one new aggregation endpoint
> composing existing service methods. Relates to **K-018** (real-time push — explicitly NOT pulled
> forward; FR-4's 5s freshness bar is met by polling, §3.2) and **K-031** (FR-10 reuses
> `diff_def_snapshot`/the structure reads K-031 shipped).
> **Id/milestone assigned by the plan itself** (`docs/plans/web-api-coverage.md` header — "K-036
> by inspection, next free id after K-035"; milestone M3.5 recommended, not gating M3 or M4).
- **Owner:** `architect` (plan authored + analyst-gated, Pass 3 approved) → per-unit: `graph-dba`
  (U1, queries) → `coder`/`tdd-engineer` (U2/U4/U5, backend) → `frontend-engineer` (U3/U6/U7/U8/U9,
  web UI) → `qa-engineer` (U10, black-box AC pass). Ten units across 5 waves — see the plan §4 for
  the full dependency graph.
- **Scope (committed):** FR-1..FR-10/AC-1..AC-6 — defs viewer, inline run cue, run detail panel
  (status/steps/trace/failure/structured-input resume), thread participants list, ready-to-demo
  banner. **Out of scope:** FR-11..FR-14 (publish/materialize UI, deep snapshot browsing, explicit
  "start a run" UI, health/get-message UI), end-user UX polish, MCP parity, chat layout/theming
  migration, auth, in-browser def authoring, and pulling K-018 forward.
- **New server-side surface (plan §3.1):** `GET /threads/{id}/workflow-runs` (FR-2, backed by new
  query `find_runs_for_thread`, QUERIES.md §12.14), `GET /threads/{id}/participants` (FR-8, backed
  by new query "List thread participants", QUERIES.md §2), `GET /workspaces/{ws}/readiness`
  (FR-10, no new Cypher — pure service-layer composition).
- **Risks/RAM (rule 6):** one new index, `WorkflowRun.startedAt` (K-036/U1) — `WorkflowRun`
  cardinality is tiny per workspace, RAM cost negligible. No new label, edge type, or vector
  dimension; no row in DESIGN §1's decision register reopened (plan §3.4).
- **Test strategy:** `docs/plans/web-api-coverage.md` §5 — server: repository/service/API
  three-layer tests per new unit + `./scripts/test_queries.sh`; web: no new JS harness (thin
  client, server owns the logic), except the FR-2 "most relevant run" tie-break, which must be a
  dependency-free pure function with bare-`node` assertions (U6); acceptance: `qa-engineer`
  two-pass black-box session (U10, plan §5.2) driving the real seeded demo workspace, since
  `TRIGGER_DEF_KEY` is a single process-wide env var and AC-1 (`triage`) / AC-2 (`access-request`)
  need different server configs.
- **Progress:** **Wave 1 (U1 graph-dba, U2 readiness endpoint, U3 defs-viewer UI) delivered
  2026-07-28** (`3d2234c`). U1: both queries authored, `GRAPH.PROFILE`-verified, and landed
  (QUERIES.md §2 "List thread participants" + §12.14 `find_runs_for_thread`;
  `scripts/test_queries.sh` 256/256 → **276/276**); U1 also surfaced and documented a
  previously-unknown FalkorDB planner quirk (a `WHERE` predicate on one pattern variable can pull
  the label-scan anchor onto that variable's label even when a much smaller label sits elsewhere
  in the same pattern — `claude/graph-dba/falkordb-quirks.md`, "Query tuning"). U2:
  `GET /workspaces/{ws}/readiness` (FR-10/AC-6). U3: the defs-viewer web UI.
  **Wave 2 (U4 `GET /threads/{id}/workflow-runs`, U5 `GET /threads/{id}/participants`) delivered
  2026-07-29** — both wire repository/service/API layers on the two Wave-1 queries, no new
  Cypher (HISTORY.md 2026-07-29 "K-036 U4+U5 (Wave 2)"). 27 new tests (10 repository + 7 service +
  10 API); full suite **641 passed, 1 deselected** (up from the 614 baseline);
  `./scripts/test_queries.sh` unaffected, still **276/276**.
  **Waves 3-4 (U6 run cue + run detail panel, U7 participants toggle, U8 readiness banner, U9
  waiting-form/structured-input/failure display) delivered 2026-07-29** — all web-only, no
  server-side change (HISTORY.md 2026-07-29 "K-036 U6+U7+U8+U9 (Waves 3-4)"). U6's "most relevant
  run" tie-break shipped as the required dependency-free pure function
  (`web/run-select.js`/`web/tests/run-select.test.js`, 12/12 `node`-run assertions passing, §5.2
  finding m3). Manually verified end to end against a live server (no headless-browser driver
  available in this sandbox, so API-level rather than click-level): `@mention` → cue → panel →
  parked-step form → submit → re-poll → terminal `done`; a rejected submit's 400 toast path; a
  forced budget-exhaustion `failed` run's reason display; the participants chip row (both kinds);
  and the readiness banner's not-ready path, which surfaced a **genuine, pre-existing** `reference`
  `access-request@v1` drift (2 `START` edges, diverges from the `ws:acme` snapshot) in this dev
  environment — unrelated to this change, not touched, flagged as K-034-territory cleanup.
  Also fixed in passing: a latent CSS scoping bug (`.badge` was `.msg`-only, didn't style the
  reused agent badge inside a participant chip) and a missing `#run-panel` join to the shared
  overlay `display:none`/positioning rule.
  **Wave 5 (U10, qa-engineer two-pass black-box AC-1..AC-6 session, plan §5.2) delivered
  2026-07-29 — verdict PASS with parked/non-blocking limitations.** All six ACs (AC-1..AC-6)
  satisfied against the delivered `web/` + `server/` code; no defect found in K-036's own diff.
  Session tooling constraint: no browser-automation tool was available, so the pass drove the
  exact REST calls `web/app.js` makes, cross-checked against a direct reading of the render logic
  that consumes each response (documented as a substitution, not treated as a real browser
  session). Two non-blocking findings, both filed as follow-ups rather than reopening this item:
  a **major** operational finding that the plan's own sanctioned Pass-B restart
  (`FALKORCHAT_TRIGGER_DEF_KEY=access-request`) causes `scripts/start_server.sh`'s unconditional
  `seed_workflows.sh` re-seed to silently graft `triage`'s steps onto `access-request@v1` in the
  `reference` graph — **K-037**; a minor cosmetic finding that `start_server.sh`'s startup banner
  hardcodes "triage def triage@v1" regardless of an active env override, wiring itself unaffected
  — folded into K-037. A testability gap (not a defect) was also recorded: forcing a
  *chat-triggered, thread-linked* budget-exhaustion run for AC-3 isn't achievable with either
  seeded demo def (parked steps are budget-exempt by design and neither def contains a self-loop);
  AC-3 was validated at the REST/rendering-contract level via a directly-started run instead. This
  mirrors K-025/M3's own close (QA PASS with parked, model-gated limitations, follow-ups filed
  rather than blocking). Also carried out of the Wave 3+4 analyst re-review gate (Pass 3, findings
  m6/m7): `refreshRunPanel` (`web/app.js`) has no mutex against overlapping poll-tick/submit-
  response invocations, so a stale response can transiently overwrite a fresher one (m6), and an
  external same-step waiting→running→waiting round-trip between two poll ticks could leave the
  form stale (m7) — both self-heal within one `POLL_MS` (≤3s) tick, no data loss, filed as
  **K-038** rather than chased at gate time. **All five waves now delivered; K-036 done.**
  Artifacts: `docs/test-plans/web-api-coverage.md` (v1, written before execution),
  `docs/test-reports/web-api-coverage-report.md` (verdict + findings), all three
  `docs/reviews/web-api-coverage-impl.md` passes (Pass 1 Wave 1+2 gate, Pass 2 Wave 3+4 gate ⛔
  needs-changes → fix, Pass 3 re-review ✅), `docs/plans/web-api-coverage-coordination.md` (teco's
  full run ledger, all 5 waves + both review gates).

### K-037 — `FALKORCHAT_TRIGGER_DEF_KEY` override during a restart grafts `triage`'s steps onto `access-request@v1` in `reference` (🔵 proposed — filed out of K-036's Wave 5 QA pass, Finding 1 (major) + Finding 2 (minor), 2026-07-29)

> **Why it exists.** `docs/test-reports/web-api-coverage-report.md` Finding 1 (major, confirmed
> reproducible): the plan's own sanctioned Pass-B demo/QA procedure — restart with
> `FALKORCHAT_TRIGGER_DEF_KEY=access-request FALKORCHAT_TRIGGER_DEF_VERSION=v1
> ./scripts/start_server.sh` — silently and irreversibly corrupts the `reference` graph.
> `start_server.sh` unconditionally re-runs `seed_workflows.sh` on every start (stage 5/6,
> including a restart). `seed_workflows.sh` reads the **same** `FALKORCHAT_TRIGGER_DEF_KEY`/
> `_VERSION` pair to identify its **first, inline `triage`-literal** def entry
> (`DEF_KEY="${FALKORCHAT_TRIGGER_DEF_KEY:-triage}"`). With the override active it publishes the
> triage-shaped step literal (`intake`/`research`/`answer`) **under the key `access-request@v1`**:
> the `WorkflowDef` node itself reports "already present — no-op" (it already existed), but the
> per-step `MERGE` underneath is keyed by `stepUid`, so `"access-request:v1:intake"` (never seen
> before) gets created and `HAS_STEP`/`START`-linked into the *existing*, unrelated
> `access-request@v1` def. Confirmed via Cypher against `reference`: **9** `Step`s (the correct 6
> plus 3 spurious `triage` ones) and **2** `START` edges (`submit` — correct — and `intake` —
> spurious) where there should be 6/1. Every restart with this override further corrupts
> `reference`, additively and irreversibly (publish/materialize are append-only). No impact on any
> run exercised so far — the executor drives off the `ws:acme` snapshot, and the spurious steps
> are unreachable dead nodes — but the corruption is real, silent, and compounds on repeat use of a
> procedure the plan itself endorses (§7 risk #1) and that this session's own QA pass had to use.
- **Distinct from K-034.** Same *symptom* class (duplicate `START`/`TRANSITION` edges from a
  create-only re-publish; the readiness endpoint's own message cites K-034 verbatim when it
  detects this). But the **trigger mechanism is different**: K-034 is about re-publishing an
  *edited* version of the *same* def; this is a **generic env-var name collision** between an
  operational override meant for the chat-trigger wiring (`config.TRIGGER_DEF_KEY`) and
  `seed_workflows.sh`'s reuse of that same variable name for its unrelated inline `triage`-literal
  def's identity — two different defs, sharing one env var, for two different purposes.
- **Owner:** **`devops`**/**`coder`** — this is repo automation-script territory (`scripts/
  seed_workflows.sh`), not application code. Candidate fix: give the triage-literal's key/version
  their own dedicated env var, independent of `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION`, so
  overriding the trigger for a demo/QA session can never feed back into what `seed_workflows.sh`
  publishes under a different def's key. Route to **`graph-dba`** only for the `ws:acme` /
  `reference` cleanup hand (deleting and republishing the now-doubly-contaminated
  `access-request@v1` def+snapshot subgraphs is a destructive shared-state op on live data, the
  same class of stakeholder-gated cleanup K-031 §6 R-1 already declined to do inline) — the script
  fix itself needs no graph-dba gate (no query-shape or DDL change).
- **Also fold in (same script, same owner, trivial to land together) — Finding 2 (minor,
  cosmetic).** `scripts/start_server.sh:136` prints a hardcoded `"Workflow:  enabled=… (triage def
  triage@v1)"` banner regardless of an active `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` override —
  confirmed the actual wiring is correct (`/proc/<pid>/environ` + a live functional test), only
  the printed text is stale. One-line fix: interpolate the actual configured key/version into the
  banner instead of the literal.
- **Risks/RAM:** none — no new node type, label, property, index or vector dimension; the fix is a
  script-level identity/naming change, not a schema or query-shape change.
- **Test strategy:** offline — a script-level check (or a `verify_workflows.sh`-style assertion)
  that seeding with `FALKORCHAT_TRIGGER_DEF_KEY=access-request` set does **not** add any `Step`/
  `START` edge to `access-request@v1` beyond its canonical 6/1 shape; a manual restart-with-override
  smoke test against a throwaway workspace, reusing the exact repro sequence from the QA report.
  The `ws:acme`/`reference` cleanup (if done) needs its own before/after Cypher verification
  (step/`START`-edge counts back to 6/1), same discipline as K-034's acceptance instrument.

### K-038 — `refreshRunPanel` has no mutex against overlapping poll-tick/submit-response invocations (🔵 proposed — filed out of K-036's Wave 3+4 analyst re-review gate, `docs/reviews/web-api-coverage-impl.md` Pass 3, findings m6/m7, 2026-07-29)

> **Why it exists.** Pass 3 fixed M1 (the destructive every-tick `renderWaitingForm` rebuild) and,
> while deep-tracing the fix per the task's own request, found two narrower, non-blocking races in
> the same run-panel poll/submit machinery — recorded there rather than chased at gate time, and
> filed here as this item per the K-036 close-out plan.
- **m6 — unordered concurrent `refreshRunPanel` calls; a stale response can transiently overwrite
  a fresher one.** `refreshRunPanel` (`web/app.js`) is not mutex-protected against overlapping
  invocations, and both `startRunPolling`'s periodic tick and `submitRunInput`'s post-submit call
  invoke it independently. If a tick's `refreshRunPanel` is still in flight
  (`Promise.all([GET run, GET step-runs])` awaiting) at the moment a submit resolves and triggers
  its own `refreshRunPanel`, both fetches race against the same `runId`, and whichever's
  `Promise.all` resolves *last* wins the render — regardless of which is fresher. A stale
  (pre-submit) response landing after the fresh one briefly shows an already-superseded step.
- **m7 — the same-step-key rebuild guard can't distinguish "still the same wait" from "revisited
  after an externally-driven round-trip this session's poll never observed."** The
  `state.runWaitingKey` guard (M1's fix) only resets when *this session's own* render observes a
  non-`waiting` status. If some other actor completes a full `waiting → running → waiting`
  round-trip on the *same* `atStepKey` entirely between two of this session's poll ticks, the key
  still matches on the next tick and the box stays hidden behind the early-return guard even
  though it's a genuinely new visit to that step.
- **Non-blocking, self-healing — carry this framing accurately.** Both require timing conditions
  well outside normal single-operator, human-typing-speed form use (a sub-`POLL_MS`, i.e. <3s,
  window for m6; an external actor plus a round-trip faster than one `POLL_MS` for m7), and both
  self-heal within one more `POLL_MS` (≤3s) tick — the next poll fetches true current state again
  and forces a correct rebuild either way. Neither loses user-entered input (unlike M1); neither is
  a permanent inconsistency.
- **Owner:** **`frontend-engineer`**, `web/app.js`.
- **Scope sketch (to be designed, not decided here):** m6 — a request-sequence token (stamp each
  `refreshRunPanel` call with an incrementing counter, ignore a response whose stamp is behind the
  latest issued) would close it; m7 — needs a server-supplied, monotonically-changing identifier
  per wait-instance (not just `atStepKey`) to distinguish "unchanged wait" from "revisited wait",
  which is a server-side surface change, not JS-only.
- **Risks/RAM:** none — web-only, no server/graph surface touched by m6; m7's full fix would need a
  new/changed field on the run-detail response, scoped by whoever picks this up, not decided here.
- **Test strategy:** a DOM-stub harness (the same shape Pass 2's fix verification used) driving two
  overlapping `refreshRunPanel` promises resolving out of order, asserting the later-issued one
  always wins regardless of resolution order (m6); an injected-clock or mocked-response test
  simulating an external round-trip between two ticks, asserting the panel still rebuilds (m7).

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
  and stop accretion. **Status update (K-031, 2026-07-24):** the three new §11 structure/diff routes **do** declare
  `response_model=` (`WorkflowDefStructureOut` / `WorkflowDiffOut`, `schemas.py`) with exact-key-set contract tests; every
  pre-existing route still does not. That is a **deliberate** non-retrofit — FastAPI's `response_model` *filters*
  undeclared fields, so a wrong model silently drops a field the web client reads — and it leaves the repo with a **mixed
  convention**. This entry stays open; the new routes are the worked precedent for the eventual retrofit, not the retrofit.
- **Opportunistic nit — re-slug the K-031 implementation review** (recorded, **not** scheduled work).
  It is filed under the slug `k031-structure-read-impl`, while the rest of its family — the plan and
  the plan review — uses `workflow-def-structure-read`. The filename grammar's family rule (*the same
  slug across several kinds **is** the family; a downstream document inventing a new slug is a
  defect*) is therefore broken by one member. Correcting it to `workflow-def-structure-read-impl`
  costs **4 occurrences across 3 files** (this backlog, the component change log, and the M3
  follow-ups coordination plan). Fold it into a change that already opens all three; it does not earn
  a change of its own, and renames in this repo are forward-only by ruling.
- DESIGN §13 remaining open questions — resolve as their milestones arrive: workflow guard expression language (M3),
  real auth (K-016), message/embedding retention, cross-workspace analytics, Bolt vs RESP
  for the gateway (K-018).
- **WSL2 memory cap for the 16GB host** (parked, not applied per user 2026-07-18) — WSL2 runs uncapped at its 8GB
  default (50% of the 16GB host) with `autoMemoryReclaim` off, overcommitting host RAM alongside Windows-side LM Studio;
  likely root cause of the recent memory-overload crashes. Parked fix: set `memory=6GB` + `swap=4GB` +
  `autoMemoryReclaim=gradual` in `C:\Users\mauri\.wslconfig` (keep `networkingMode=mirrored`), then `wsl --shutdown`.
  Full diagnostic + apply procedure: `docs/plans/wsl2-memory-diagnostic.md`. Un-park (apply) if the crashes recur —
  verdict was confirmed-by-defaults, not reproduced live (FalkorDB was down during the diagnostic).
