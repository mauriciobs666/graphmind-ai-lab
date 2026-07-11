# M3 — Workflow engine · Coordination doc (teco)

> Coordination/work-breakdown doc for the start of milestone **M3 — Workflow engine**
> (`falkor-chat`). Owned by teco (coordinator). Tracks delegation, sequencing, and integration.
> Companion to the architect plan at `docs/plans/m3-workflow-engine.md` (authored under step 1).
> Last updated: 2026-07-09.

## Goal & definition of done (this engagement)

Start M3 per DESIGN §12.4. Two deliverables in order:

1. **Decompose all of M3** into sequenced falkor-chat backlog items (K-020 onward), matching the
   existing item format (owner, inputs/prereqs, scope, done-condition, risks/RAM, test strategy),
   **plus** an implementation plan for **slice 1 only**. Architect deliverable, plan doc at
   `docs/plans/m3-workflow-engine.md`, handed onward by path.
2. **Execute slice 1 only:** the workflow **definition model in the `reference` graph** +
   **snapshot materialization into a workspace graph on publish** (immutable, versioned).
   Shape per component convention: **graph-dba gate** (model + verified Cypher + `test_queries.sh`
   assertions raising the 149 baseline, new count enumerated) → **tdd-engineer** implementation
   (repository/service layer per the plan).
   - **Out of scope for slice 1 (later slices):** the run/step-run executor and chat linkage.
3. Acceptance verification: route to **qa-engineer** if slice 1 warrants; at minimum both suites
   green: `./scripts/test_queries.sh` (baseline 149/149) and `server/ pytest` (baseline 156).

## Grounding findings (read from repo, 2026-07-09)

- **M3 model is fully specified in DESIGN:** §6.1 definition (`WorkflowDef {key,version,name,kind}`
  `-[:START]->(:Step {key,type,config})`, `(:Step)-[:TRANSITION {on,guard,order}]->(:Step)`),
  §6.2 run/step-run (later slices), §4 the materialize-on-publish decision (Approach B: immutable
  versioned snapshot copied into `ws:{id}` under `WorkflowDefSnapshot`), §3 rule 3 (edges can't
  cross graphs — the reason defs are materialized), §6.3 (coordination = workflow, not a separate
  primitive).
- **Schema scaffolding partly pre-laid:** `scripts/bootstrap_schema.sh` already creates
  `WorkflowDef.key` + `WorkflowDef.version` indexes and the composite UNIQUE `{key,version}`,
  `Step.key` index (reference graph); and `WorkflowDefSnapshot.key/version` (+UNIQUE) and `StepRun`
  indexes/constraint (workspace graph). Slice 1 must **reconcile with this existing DDL**, not
  assume greenfield.
- **Layering (locked, M1):** `api.py`/`mcp.py` thin adapters → `services.py` → `repository.py`
  (Cypher 1:1 with `QUERIES.md`); tenant seam `config.get_context`. New workflow queries land in
  `QUERIES.md` as the graph-dba verified-query deliverable (mirrors `m2-groundwork-queries.md`).
- **Component convention:** graph-dba gate (verified Cypher + `test_queries.sh` assertions raising
  the suite baseline, enumerated) → tdd-engineer impl. Pattern used by K-002/K-007/K-008/K-013.
- **Baselines before changes:** query suite 149/149; pytest 156. FalkorDB running (`falkordb-dev`,
  6379/3000). LM Studio NOT required for slice 1.
- **Hard rules:** FalkorDB OpenCypher (no APOC/GDS); `config`/`ctx`/`input`/`output` are serialized
  strings — never filter inside them; every `MERGE` backed by a uniqueness constraint; additive
  writes only to `reference`/`ws:acme` (no destructive ops on shared graphs; isolated test
  workspaces only).

## Decision points (return to user, do not guess)

- **(a)** §13 guard expression language (expr lib vs minimal DSL in `Step.config`) — only if slice
  1's `Step.config` shape forces the choice early. Expectation: slice 1 stores `config` as an
  opaque serialized string (guards are evaluated at *run* time, a later slice), so it likely does
  **not** force the choice — but confirm via the architect and flag if it does.
- **(b)** Any change to served tenant `ws:acme` beyond additive schema.
- **(c)** Scope creep beyond slice 1.

## Work breakdown & status

| # | Unit | Owner | Depends on | Status |
|---|------|-------|-----------|--------|
| 1 | Decompose all M3 → backlog items K-020+ · plan slice 1 | architect | — | ✅ done |
| 2a | Slice 1 graph-dba gate (model + verified Cypher + suite assertions) | graph-dba | plan (1) | ✅ done — 193/193 |
| 2b | Slice 1 impl (repository/service layer + REST) | tdd-engineer | 2a | ✅ done — pytest 196 |
| 3 | Acceptance verification | teco (suites) / qa deferred to K-025 | 2b | ✅ suites green; QA parked at K-025 |

## Architect deliverable (step 1) — DONE 2026-07-09

Plan at `docs/plans/m3-workflow-engine.md`. Decomposition **K-020…K-025**:
K-020 def model in `reference` → K-021 snapshot materialization into `ws:{id}` (**= Slice 1**) →
K-022 run/step-run executor + guard eval (**§13 guard-language decision forced HERE, not slice 1**) →
K-023 chat linkage (`TRIGGERED_BY` / StepRun-`EMITTED`) → K-024 two proof flows (conversational +
process, the DoD proof) → K-025 QA acceptance ⇒ M3 ✅. Spine essentially linear.

**Decision points — all resolved, no user pause needed:**
- (a) §13 guard language: **NOT forced by slice 1** — `Step.config`/`TRANSITION.guard` stored as
  opaque serialized strings, never filtered inside; guard *evaluation* is K-022. Confirmed by
  architect (plan F7/§B8). Decision deferred to K-022 (returns to user then).
- (b) `ws:acme`: slice-1 change is **additive-only** (new `Step` index + UNIQUE constraint via
  idempotent bootstrap). Within the allowance.
- Open, non-blocking: authz for global `reference` def publish (ties to deferred K-016 auth) — noted, not gating.

Key design decision (architect): synthetic `Step.stepUid = "{defKey}:{version}:{stepKey}"` as the
`MERGE`-backing identity (index + UNIQUE in both graphs), because `Step.key` is unique only within a
def. `Step.key` stays index-only. Expected suite target **149 → ~160 (K-020) → ~172 (K-021)** — gate pins exact.

## Delegation log

- **2026-07-09 — architect delegated & DONE** (step 1): plan at `docs/plans/m3-workflow-engine.md`.
- **2026-07-09 — graph-dba delegated & DONE** (step 2a, consolidated K-020+K-021 gate):
  **`test_queries.sh` 149 → 193/193** (teco-verified). Added: `Step.stepUid` index+UNIQUE in both
  graphs + workspace `Step.key` index (`bootstrap_schema.sh`); `QUERIES.md` §11 (11.1 publish,
  11.2a/b read-def, 11.3 list/get def, 11.4 materialize, 11.5 read-snapshot, 11.6 list/get snapshot),
  all live-verified + PROFILEd; DESIGN §6.1/§7.1/§7.2 updated. **Model addition (justified, in-scope):**
  added `(:WorkflowDef|:WorkflowDefSnapshot)-[:HAS_STEP]->(:Step)` containment edge — the plan's
  `STARTS WITH stepUid` scoping PROFILEd as a label scan; `HAS_STEP` gives index-anchored O(steps-in-def)
  reads. §11.2a returns `{name,kind,startKey,steps:[{key,type,config}]}`, §11.2b returns
  `{transitions:[{from,to,on,guard,order}]}` — 1:1 with §11.4 materialize params. Handoff notes: spec
  validation must run before publish/materialize; reference-wiping pytest fixture needed (F8);
  materialize is two-phase non-atomic (retry-safe via idempotent MERGE).
- **2026-07-09 — tdd-engineer delegated & DONE** (step 2b, consolidated K-020+K-021 impl):
  **pytest 156 → 196/196** (teco-verified), `test_queries.sh` still 193/193 (untouched). Added
  `db.reference_graph`; repository `publish_def`/`read_def_subgraph`/`get_def`/`list_defs` +
  `materialize_snapshot`/`get_snapshot`/`list_snapshots` (1:1 with QUERIES.md §11) + typed errors
  `WorkflowDefNotFoundError`/`WorkflowDefSpecError`; services `publish_workflow_def` (spec validation
  before any write) + two-phase `materialize_def` + pass-throughs; size-bounded `schemas.py` models;
  thin REST surface in `api.py`/`app.py` (included, not deferred); reference-wiping `wf_repo` fixture.
  Structural-parity + idempotency proven at repo level.

## Final state (2026-07-09) — Slice 1 DELIVERED

- **Suites (teco-verified):** `test_queries.sh` **193/193** (was 149, +44); pytest **196** (was 156, +40).
- **Slice 1 done-condition met:** workflow definition model canonical in `reference`; publish →
  materialize immutable versioned snapshot into a workspace graph; both suites green. Executor
  (K-022), chat linkage (K-023), proof flows (K-024) NOT built — later slices, as scoped.

### Follow-ups (not blocking; carry into K-022+)
- **`start_key` contract (from tdd-engineer):** the plan's `publish_workflow_def` signature had no
  `start_key`; the implementer resolved it by having exactly one step declare `start: True` (that
  step's key becomes the repo `start_key`), validated in `_validate_def_spec`. Reasonable
  self-contained choice — **confirm/lock this def-spec contract when K-022 consumes it.** (Not one of
  the user decision points; fully reversible, one-line change if a top-level `start_key` field is
  preferred instead.)
- **§13 guard expression language:** the genuine user decision (expr lib vs minimal DSL in
  `Step.config`/`TRANSITION.guard`), forced at **K-022** (run-time guard evaluation). Return to the
  user at K-022's architect design pass.
- **Authz for global `reference` def publish** (any tenant mutates the shared graph): additive +
  immutable-per-version so safe now; ties to deferred K-016 auth.
- **QA acceptance pass:** parked at **K-025** per the architect plan (after the executor lands),
  covering publish → materialize → run → step-run → chat linkage for both proof flows.
- **Backlog transcription:** the K-020…K-025 items drafted in `docs/plans/m3-workflow-engine.md`
  Part A still need transcribing into `docs/BACKLOG.md` (with K-020+K-021 marked delivered) — a doc
  task not performed in this engagement.
</content>
</invoke>
