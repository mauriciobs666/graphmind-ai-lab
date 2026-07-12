# Backlog — falkor-chat

> Forward-looking backlog for the `falkor-chat` component (formerly `kaizen/plan.md`; item IDs
> keep the `K-` prefix). Delivered work is logged in [`HISTORY.md`](./HISTORY.md); completed
> plan documents move to [`archive/`](./archive/).
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to HISTORY.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-12 (**K-022 amended:** analyst post-implementation review added to its
> owner chain and done-condition — the team's first fully-gated coordinated run; see the
> review-gate note on the item. Prior review 2026-07-09: **M3 — Workflow engine started: slice 1
> delivered ✅** — K-020 (def
> model in `reference`) + K-021 (snapshot materialization) landed via the teco-coordinated run,
> see HISTORY.md 2026-07-09; new baselines **pytest 196 / query suite 193/193**. Full M3
> decomposition (K-020…K-025) in `docs/plans/m3-workflow-engine.md` Part A — canonical item text
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
| **M3 — Workflows** 🟡 | Def model + snapshot + executor + chat linkage, proven by one conversational + one business-process flow, QA-accepted | **K-020 ✅ + K-021 ✅** (slice 1, 2026-07-09) → **K-022 → K-023 → K-024 → K-025** |
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
> 2026-07-09 → HISTORY.md. The remaining M3 chain is essentially linear:
> **K-022 (executor) → K-023 (chat linkage) → K-024 (proof flows) → K-025 (QA) ⇒ M3 ✅.**
> Canonical item text + slice-1 implementation plan: `docs/plans/m3-workflow-engine.md`
> (Part A = decomposition, Part B = slice 1). Compact copies below.

### — Milestone M3 (Workflow engine) — slice 1 ✅, K-022…K-025 queued —

> **K-020 — Workflow definition model in `reference`** and **K-021 — Snapshot materialization
> into `ws:{id}` on publish** — **delivered ✅ 2026-07-09 → HISTORY.md.** Suites raised to
> query 193/193, pytest 196. Slice-1 residuals to carry into K-022: lock the def-spec
> `start_key` contract (implemented as "exactly one step declares `start: True`"); the
> `-[:HAS_STEP]->` containment edge added at the gate (index-anchored def-scoped reads).

### K-022 — Run + StepRun executor core (Slice 2) (🟡 Landing 1 ✅ delivered + analyst-approved 2026-07-12 — Landing 2 (trigger+proof, U11–U15) remaining)

- **Delivered (Landing 1, U1–U10) ✅ 2026-07-12:** the offline LLM-native executor + node
  capabilities (Phases 0–3) — schema/DDL + DESIGN reconciliation + QUERIES §12, `executor.py`
  (§2.1 A/B/C loop) / `guards.py` / `tools.py`, repository/services wiring. Suites raised to
  **query 241/241, pytest 283**, both green. Analyst gate = **approve-with-suggestions, 0
  blockers** (1 major M-1 + 3 minor + 3 nit; two seams deferred to Landing 2). Reframed as an
  offline-first landing under `docs/plans/m3-executor.md`; teco-coordinated
  graph-dba → tdd-engineer → coder with a mandatory analyst review gate — the team's first
  fully-gated run. See `docs/HISTORY.md` (2026-07-12), the review at
  `docs/reviews/m3-executor-impl.md`, and the coordination log
  `docs/plans/m3-executor-coordination.md`. **Remaining: Landing 2 (trigger + triage proof,
  U11–U15)** — carried into K-023 below.

- **Owner:** **`architect`** design pass first — engine-loop semantics **+ resolve DESIGN §13
  guard expression language (expr lib vs minimal DSL in `Step.config`/`TRANSITION.guard`) — a
  genuine user decision point, surface before implementing** → **`graph-dba`** gate (run/step-run
  write/read queries; `WorkflowRun`/`StepRun` DDL already exists) → **`tdd-engineer`** →
  **`analyst` post-implementation review** (added 2026-07-12; see review-gate note below).
- **Inputs/prereqs:** K-021 ✅ (materialized snapshots to walk); the §13 decision. Plan doc:
  `docs/plans/m3-executor.md`. Also lock the `start_key` contract here (slice-1 residual).
- **Scope (DESIGN §6.2):** `WorkflowRun {runId,defKey,defVersion,status,startedAt,ctx}` with
  `OF_DEF`/`AT_STEP`/`HAS_STEP_RUN`; `StepRun {stepRunId,stepKey,status,…,input,output}` with
  `RAN` + `NEXT` audit trail. Engine loop: read `AT_STEP` → evaluate `TRANSITION` guards against
  `ctx` → create next `StepRun` → execute → append `NEXT` → move `AT_STEP`.
- **Done-condition:** both suites green at the new enumerated gate baseline; a run walks a
  materialized def deterministically; guards evaluated per the §13 decision; audit trail complete;
  **analyst review of the delivered diff at `docs/reviews/m3-executor-impl.md` with verdict
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

### K-023 — Workflow ↔ chat linkage (Slice 3) (🔵 proposed — needs K-022)

- **Owner:** **`graph-dba`** gate (`TRIGGERED_BY` / StepRun-`EMITTED` writes/reads) →
  **`tdd-engineer`**/`coder`.
- **Inputs/carried from K-022 Landing 1** (this slice = U11, the trigger wiring; see
  `docs/reviews/m3-executor-impl.md` findings + the coordination doc's "Carried to Landing 2"
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

### K-024 — Proof flows: one conversational + one business-process (Slice 4) (🔵 proposed — needs K-023)

- **Owner:** **`coder`**/`tdd-engineer` + `scripts/seed_workflows.sh` (mirrors `seed_demo.sh`).
- **Scope:** the M3 DoD proof — publish two canonical defs, materialize into `ws:acme`
  (additive-only), run both to completion: **`kind:'conversation'`** (agent Q&A over
  `prompt`/`tool`/`message` steps, reuses the M2 responder, needs LM Studio behind
  `FALKORCHAT_ENABLE_AGENT`) and **`kind:'process'`** (onboarding/approval over
  `human`/`decision`/`wait` steps, LLM-free — the §6.3 coordination-is-workflow proof).
- **Done-condition:** both runs reach a terminal step; seed idempotent; documented run-through.
- **Test strategy:** one e2e run test per flow (conversational behind a live marker; process
  deterministic/offline).

### K-025 — QA acceptance pass on M3 (Slice 5) (🔵 proposed — needs K-020…K-024)

- **Owner:** **`qa-engineer`**. Black-box publish → materialize → run → trace → chat linkage for
  both proof flows; versioned plan/report per repo convention (`docs/test-plans/`,
  `docs/test-reports/`); isolated `ws:qa` (create + delete), `reference`/`ws:acme` additive-only.
- **Done-condition:** PASS (or PASS-with-parked-defects) on green baselines ⇒ **M3 ✅**.

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
| `docs/plans/m3-workflow-engine.md` | **Created ✅ 2026-07-09** — M3 decomposition (Part A, K-020…K-025) + slice-1 plan (Part B). Coordination log: `m3-workflow-engine-coordination.md`. |
| `docs/plans/m3-executor.md` | K-022: run/step-run executor + the §13 guard-language decision (author at pickup). |
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
