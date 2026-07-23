# M3 — Workflow engine: decomposition + Slice 1 implementation plan

> **Status:** proposed (architect plan, 2026-07-09). Planning-only artifact — no code/DDL changed.
> **Scope:** Part A decomposes **all of M3** into sequenced backlog items K-020…K-025 (to be
> transcribed into `docs/BACKLOG.md` later — this doc is the draft, not an edit of `plan.md`).
> Part B is the **full implementation plan for Slice 1 only** (K-020 + K-021: the definition model
> in `reference` + snapshot materialization into `ws:{id}`). Later slices are Part-A items only.
> **Baselines before this work:** query suite **149/149**, pytest **156**, both green. FalkorDB up.
> **M3 DoD (DESIGN §12 roadmap item 4 / §12.4):** "Definition model in `reference`, snapshot
> materialization, run/step-run executor, chat linkage; both a conversational flow and a
> business-process flow as proof."

---

## 0. Context & findings (read before implementing)

Investigated: `docs/DESIGN.md` §3/§4/§6/§7/§9, `docs/QUERIES.md`, `scripts/bootstrap_schema.sh`,
`scripts/test_queries.sh`, `server/falkorchat/{db,config,repository,services}.py`, `tests/conftest.py`.
Live-probed the publish query shape against the running FalkorDB edge build (throwaway `_probe`
graph, since deleted).

**F1 — Existing workflow DDL is partially pre-provisioned (do NOT assume greenfield).**
`bootstrap_schema.sh` already creates, unprompted by any slice:
- **reference graph:** `WorkflowDef.key` index, `WorkflowDef.version` index, composite
  `WorkflowDef UNIQUE {key, version}`, `Step.key` index (script lines 42–58). **No `Step`
  constraint** (deliberate, DESIGN §7.2: "traversal anchor for materialization").
- **workspace graph:** `WorkflowDefSnapshot.key`/`.version` indexes + composite
  `WorkflowDefSnapshot UNIQUE {key, version}`; `WorkflowRun.runId` index+constraint +
  `WorkflowRun.status` index; `StepRun.stepRunId` index+constraint + `StepRun.status` index
  (lines 95–155). **No `Step` index or constraint at all in the workspace graph.**

  `WorkflowRun`/`StepRun` DDL is *ahead of its slice* (belongs to K-022) — Slice 1 leaves it
  untouched. `test_queries.sh` already asserts the reference `WorkflowDef` composite constraint
  (3 assertions, lines 600–609); nothing asserts the workspace snapshot constraint or any `Step`.

**F2 — The DDL gap Slice 1 must fill.** Materialized steps land in the workspace graph as `:Step`
nodes, and def authoring writes `:Step` nodes into `reference`. A `Step`'s `key` is unique only
*within a def* — never globally — so it cannot back a `MERGE`. Neither graph currently has a
`Step` uniqueness constraint, so **every `MERGE (:Step …)` Slice 1 needs is currently
unconstrained** (violates the locked rule "every `MERGE` must be backed by a uniqueness
constraint", AGENTS.md). Slice 1's design closes this with a synthetic composite identity
`Step.stepUid = "{defKey}:{version}:{stepKey}"` (workspace- and reference-unique) plus a new
index + UNIQUE constraint on it, in **both** graphs. `Step.key` stays index-only (§7.2 rationale
intact — it's the display/traversal anchor, not an identity).

**F3 — The reference-graph seam is new in the app layer.** `repository.py` only ever selects a
*workspace* graph (`db.workspace_graph(conn, ws)` → `ws:{ws}`); `db.py` has no `reference` helper.
Def authoring/reading operates on the **global `reference` graph**, and materialization
**spans two graphs** (read `reference`, write `ws:{id}`). This is the central new seam:
add `db.reference_graph(conn)` and a repository split between reference-scoped methods (no `ws`
arg) and workspace-scoped methods.

**F4 — Materialization is inherently a two-phase app-layer op (edges can't cross graphs,
DESIGN §3 rule 3 / §4).** A single `GRAPH.QUERY` runs against one graph key. So materialize =
(1) read the full def subgraph from `reference` → (2) write the snapshot subgraph into `ws:{id}`,
parameterized with the serialized steps/transitions. This is *why* Approach B (materialize) exists
and is unavoidable — flag it, don't fight it.

**F5 — The publish/materialize query shape is live-verified working on this build.** A single
`MERGE (WorkflowDef)…WITH…UNWIND $steps MERGE (Step)…WITH…MATCH(start) MERGE(START)…WITH…UNWIND
$transitions MATCH(from) MATCH(to) MERGE(TRANSITION {on,order})` query created the subgraph
(3 nodes, 2 rels) and **re-ran idempotently (0 new nodes/rels)**. Multi-UNWIND with intermediate
`MATCH` and relationship-`MERGE`-with-properties are all confirmed on `falkordb/falkordb:edge`.
The materialization query is the same shape scoped to the workspace graph.

**F6 — `length(path)` in `ORDER BY` is unsupported (live-verified again here; matches the
QUERIES.md §4 note).** A def-read cannot order steps by traversal depth. The def-read returns the
subgraph (steps + transitions as an edge list) and the app/executor reconstructs order from
`TRANSITION.order`/topology — do **not** try to order the walk by path length in Cypher.

**F7 — `config` (Step) and `guard`/`on` (TRANSITION) are flat serialized strings (rule 8).** Slice 1
stores them **opaquely** and never queries inside them. Guards are *evaluated at run time* — a later
slice (K-022). **Therefore Slice 1 does NOT force the DESIGN §13 guard-expression-language decision**
(expr lib vs minimal DSL). Confirmed explicitly (see §Open Questions). If, while building, the gate
finds `config`/`guard` must carry structure that Slice 1 queries filter on, stop and escalate — but
the design here needs none.

**F8 — Test isolation for `reference`.** `tests/conftest.py` wipes only `ws:test` node data between
tests (`MATCH (n) DETACH DELETE n`) and never touches `reference`; `test_queries.sh` treats
`reference` as throwaway (deletes it at teardown). Reference-graph pytest tests will otherwise
accumulate/collide — Slice 1 needs a `reference`-wiping fixture (or per-test unique def keys).

---

## PART A — M3 backlog decomposition (K-020 … K-025)

Six items. Slice 1 = **K-020 + K-021** (planned in full in Part B). K-022…K-025 are scoped here
as backlog items only (no implementation detail), matching the existing K-002/K-007/K-008/K-013
format: owner · inputs/prereqs · scope · done-condition · risks/RAM (rule 6) · test strategy.

### Sequencing (critical path + parallelism)

```
Slice 1 (definition + materialization) — the M3 foundation
  K-020 (def model in reference)  ── graph-dba gate ─▶ tdd
        │  (defs must exist before they can be materialized)
        ▼
  K-021 (materialize snapshot → ws) ── graph-dba gate ─▶ tdd   ⇒ Slice 1 done

Slice 2+ (executor & proof) — each builds on the prior; largely linear
  K-022 (run/step-run executor + guard eval) ── architect (§13 guard decision → USER)
                                                 ─▶ graph-dba gate ─▶ tdd   ◀─ needs K-021
        ▼
  K-023 (chat linkage: TRIGGERED_BY + StepRun EMITTED) ── graph-dba gate ─▶ tdd  ◀─ needs K-022
        ▼
  K-024 (proof flows: 1 conversation + 1 process, end-to-end) ── coder/tdd + seed script ◀─ needs K-023
        ▼
  K-025 (QA acceptance pass on M3) ── qa-engineer   ⇒ M3 ✅
```

- **Critical path:** K-020 → K-021 → K-022 → K-023 → K-024 → K-025 (essentially linear — each
  slice consumes the prior slice's structure).
- **Parallelism is limited.** The only genuinely-parallel work: K-025's *test-plan authoring*
  (qa-engineer) can begin once K-024's shape is known; a workflow web/MCP surface (not in the DoD)
  could parallel K-022+ if prioritized — out of scope here.
- **Suite discipline:** only the **graph-dba gates** in K-020, K-021, K-022, K-023 touch
  `QUERIES.md` + `test_queries.sh` (+ `bootstrap_schema.sh` where DDL changes), each raising the
  baseline with enumerated assertions. K-024 is a seed+wiring item (may add a live-marked test);
  K-025 is a QA overlay (its own `docs/test-plans/` + `docs/test-reports/`).
- **§13 decision point:** the workflow guard-expression-language choice is forced at **K-022**
  (run-time guard evaluation), *not* in Slice 1. K-022 opens with an architect design pass that
  surfaces it to the user (expr lib vs minimal DSL in `Step.config`/`TRANSITION.guard`).

### K-020 — Workflow definition model in `reference` (Slice 1a)

- **Owner:** **`graph-dba`** gate (data model reconciliation + `Step.stepUid` DDL + author/read/
  list/get def queries into `QUERIES.md`, live-verified + `GRAPH.PROFILE`ed; `test_queries.sh`
  assertions) → **`tdd-engineer`** (repository reference-graph methods + services `publish_workflow_def`
  / def-read surface).
- **Inputs/prereqs:** none beyond green baselines (149/156). No LM Studio. Reconciles with the
  **existing** reference DDL (F1) — adds only `Step.stepUid`.
- **Scope:** the canonical §6.1 definition model authored into the `reference` graph:
  `(:WorkflowDef {key,version,name,kind})-[:START]->(:Step {stepUid,key,type,config})`,
  `(:Step)-[:TRANSITION {on,guard,order}]->(:Step)`. Publish (author a def version), read a def
  subgraph, list defs, get def by key (latest / specific version). See Part B for the full plan.
- **Done-condition:** query suite green at the new gate baseline (149 → ~160, enumerated in Part B
  §B7); pytest green with reference-graph repository/service tests; a def published to `reference`
  reads back as a complete subgraph; composite `{key,version}` + `Step.stepUid` constraints block
  duplicates; re-publish of a version is an idempotent no-op.
- **Risks/RAM (rule 6):** `reference` is global, read-mostly, tiny — a def is a handful of nodes;
  `Step.stepUid` adds one short string + one range-index entry per step. Negligible RAM, but a
  **new global-write path** (any tenant publishing mutates the shared `reference` graph) — additive
  and immutable-per-version, but authz for global publish is an open question (ties to K-016).
- **Test strategy:** `test_queries.sh` reference-section assertions (publish subgraph, read, list,
  get, stepUid constraint, idempotent re-publish, index-scan profiles); pytest repository tests
  against `reference` with a reference-wiping fixture (F8); service tests for spec validation
  (start step exists, transition endpoints exist, `kind`/`type` whitelists, config/guard
  serialization).

### K-021 — Snapshot materialization into `ws:{id}` on publish (Slice 1b)

- **Owner:** **`graph-dba`** gate (workspace `Step.stepUid` DDL + materialize/read-snapshot/
  list-snapshots queries into `QUERIES.md`, live-verified + profiled; `test_queries.sh` assertions)
  → **`tdd-engineer`** (workspace-write repository methods + `services.materialize_def` two-phase
  orchestration).
- **Inputs/prereqs:** **K-020** (defs must exist in `reference` to materialize). Reconciles with the
  existing workspace `WorkflowDefSnapshot` DDL (F1) — adds only `Step.stepUid`/`Step.key`.
- **Scope:** copy a `defKey@version` subgraph from `reference` into `ws:{id}` under
  `(:WorkflowDefSnapshot {key,version,name,kind})` with **local** `START`/`Step`/`TRANSITION`
  (real in-graph edges, DESIGN §4 Approach B), immutable + versioned. Read a snapshot subgraph;
  list/get snapshots. Two-phase (F4): read reference → write workspace. See Part B.
- **Done-condition:** query suite green at the new gate baseline (~160 → ~172, enumerated in Part B
  §B7); pytest green with workspace materialization tests; materializing `defKey@v` produces a
  local snapshot subgraph structurally identical to the reference def; composite `{key,version}` +
  workspace `Step.stepUid` constraints block duplicates; re-materialize is an idempotent no-op.
- **Risks/RAM (rule 6):** one snapshot subgraph per `(workspace, defKey, version)` — bounded by
  step count (dozens, not thousands). Trivial vs. the M2 vector line. **Not** the hot path.
  Materialization spanning two graphs is **not atomic across the graph boundary** (F4) — if the
  workspace write fails after the reference read, nothing is torn (reference is untouched; the
  workspace MERGE is idempotent, so a retry completes it). Note explicitly.
- **Test strategy:** `test_queries.sh` workspace-section assertions (materialize subgraph, snapshot
  composite constraint, workspace stepUid constraint, read/list snapshot, idempotent re-materialize,
  index-scan profiles); pytest repository tests against `ws:test`; a service test that publishes to
  `reference` then materializes into `ws:test` and asserts structural parity (F8 fixture wipes both).

### K-022 — Run + StepRun executor core (Slice 2) — LATER

- **Owner:** **`architect`** (design pass: the engine loop semantics + **resolve DESIGN §13 guard
  expression language** — a genuine user decision point, expr lib vs minimal DSL) → **`graph-dba`**
  gate (run/step-run write/read queries; the `WorkflowRun`/`StepRun` DDL already exists, F1) →
  **`tdd-engineer`**.
- **Inputs/prereqs:** **K-021** (materialized snapshots to walk). The §13 guard decision (surface
  to the user before implementing evaluation).
- **Scope (per DESIGN §6.2):** `(:WorkflowRun {runId,defKey,defVersion,status,startedAt,ctx})
  -[:OF_DEF]->(:WorkflowDefSnapshot)`, `-[:AT_STEP]->(:Step)`, `-[:HAS_STEP_RUN]->(:StepRun)`;
  `(:StepRun {stepRunId,stepKey,status,startedAt,endedAt,input,output})-[:RAN]->(:Step)`,
  `(:StepRun)-[:NEXT]->(:StepRun)` audit trail. The engine loop: read `AT_STEP` → evaluate outgoing
  `TRANSITION` guards against `ctx` → create next `StepRun` → execute → append `NEXT` → move
  `AT_STEP`. `status`/`ctx`/`input`/`output` per the locked §1.2 conventions (property status; flat
  serialized strings).
- **Done-condition:** query suite green at the new gate baseline; pytest green; a run walks a
  materialized def deterministically; guards evaluated per the §13 decision; audit trail complete.
- **Risks/RAM (rule 6):** run/step-run nodes grow with execution volume (execution traces, DESIGN
  §3) — the per-workspace hot growth line for M3; index `status` is already provisioned. Guard
  evaluation must be sandboxed/bounded (injection/DoS if an expr lib is chosen).
- **Test strategy:** graph-dba contract assertions for the run write/advance queries; pytest
  executor unit tests with a stub step-executor (deterministic transitions); guard-evaluation tests.

### K-023 — Workflow ↔ chat linkage (Slice 3) — LATER

- **Owner:** **`graph-dba`** gate (`TRIGGERED_BY` / `StepRun`-`EMITTED` writes/reads) →
  **`tdd-engineer`**/`coder`.
- **Inputs/prereqs:** **K-022** (runs + step-runs exist to link).
- **Scope (DESIGN §5.1/§6.2, all within the ws graph):** `(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)`
  (a message starts/associates a run — including on-first-use materialization) and
  `(:StepRun)-[:EMITTED]->(:Message)` (a step posts a chat message). Note `EMITTED` already exists as
  a `Message→Message` provenance edge (K-013, QUERIES.md §10) — this is the **StepRun→Message** sense
  from §6.2; the gate must disambiguate the two `EMITTED` uses (or confirm reuse) explicitly.
- **Done-condition:** query suite green; a chat message can trigger a workflow run and a step can
  emit a message into a thread via the §4 write path; linkage queryable both directions.
- **Risks/RAM:** a few edges per run — negligible. The `EMITTED` overload (§10 vs §6.2) is a
  modeling-clarity risk — resolve in the gate.
- **Test strategy:** graph-dba contract assertions; pytest linkage tests; message-triggered-run
  service test.

### K-024 — Proof flows: one conversational + one business-process (Slice 4) — LATER

- **Owner:** **`coder`**/`tdd-engineer` (+ a `scripts/seed_workflows.sh` seed, mirroring
  `seed_demo.sh`).
- **Inputs/prereqs:** **K-023** (full engine + linkage).
- **Scope:** the M3 DoD proof. Author + publish two canonical defs and drive both end-to-end:
  - **`kind:'conversation'`** — an agent Q&A / triage flow over `prompt`/`tool`/`message` steps
    (LLM answer path; reuses the M2 responder). Maps the conversational world.
  - **`kind:'process'`** — a business process (e.g. onboarding/approval) over
    `human`/`decision`/`wait` steps. Maps the business world. (DESIGN §6.3: coordination *is*
    workflow — this is the `kind:'process'` proof.)
  Both defs published to `reference`, materialized into `ws:acme`, run to completion.
- **Done-condition:** both flows demonstrably execute (a run reaches a terminal step for each);
  seed script idempotent; documented run-through. **This item is where the two proof flows required
  by the M3 DoD are proven.**
- **Risks/RAM:** the conversational flow needs LM Studio (reuses M2 gating,
  `FALKORCHAT_ENABLE_AGENT`); the process flow is LLM-free. Additive seed to `ws:acme` — flag any
  non-additive change.
- **Test strategy:** an end-to-end run test per flow (conversational behind a live marker like the
  M2 responder smoke; process flow fully deterministic/offline).

### K-025 — QA acceptance pass on M3 (Slice 5) — LATER

- **Owner:** **`qa-engineer`**.
- **Inputs/prereqs:** K-020…K-024 landed.
- **Scope:** black-box acceptance pass on the workflow engine — publish → materialize → run →
  step-run trace → chat linkage — for both proof flows. Versioned test plan + report per repo
  convention (`docs/archive/test-plans/m3-workflow-engine.md`, `docs/test-reports/…-report.md`), isolated
  `ws:qa` (create + delete), `reference`/`ws:acme` untouched (or additive-only).
- **Done-condition:** PASS (or PASS-with-parked-defects) on green baselines ⇒ **M3 ✅**.
- **Risks/RAM:** transient `ws:qa`; no code under test changed. **Test strategy:** the pass itself
  (drives REST/MCP + the executor).

### Recommended plan docs (author when picked up)

| Path | Scope |
|---|---|
| `docs/archive/plans/m3-workflow-engine.md` (this doc) | K-020 + K-021 full plan; K-022–K-025 as items. |
| `docs/plans/m3-workflow-def-queries.md` | graph-dba verified-query deliverable for K-020+K-021 (mirror `m2-groundwork-queries.md`). |
| `docs/archive/plans/m3-executor.md` | K-022: run/step-run executor + the §13 guard-language decision. |

---

## PART B — Slice 1 implementation plan (K-020 + K-021)

### B1. Goal & scope

**Goal:** stand up the workflow **definition model** as canonical, versioned, immutable defs in the
`reference` graph, and **materialize** a def version into a workspace graph as an immutable local
snapshot subgraph (DESIGN §4 Approach B / §6.1) — with the repository→services surface, verified
queries, and DDL to support it.

**In scope:** the §6.1 definition graph in `reference`; publish/read/list/get def; the
`WorkflowDefSnapshot` materialization into `ws:{id}` with local `START`/`Step`/`TRANSITION`;
read/list/get snapshot; the `Step.stepUid` DDL reconciliation in both graphs; a thin REST surface
(optional final step).

**Out of scope (later slices — do NOT build):** `WorkflowRun`/`StepRun` and the executor loop
(K-022); guard *evaluation* and the §13 decision (K-022); `TRIGGERED_BY`/`StepRun`-`EMITTED` chat
linkage (K-023); the two proof flows (K-024); MCP tools for workflows (fold in when agents drive
workflows, K-023/K-024). No change to `WorkflowRun`/`StepRun` DDL (already present, F1).

### B2. Data model (the design decision)

**Definition (canonical, `reference` graph):**
```
(:WorkflowDef {key, version, name, kind})            // kind ∈ {'conversation','process'}
(:WorkflowDef)-[:START]->(:Step)
(:Step {stepUid, key, type, config})                 // type ∈ {prompt,tool,decision,human,message,wait}
(:Step)-[:TRANSITION {on, guard, order}]->(:Step)
```

**Snapshot (materialized, `ws:{id}` graph) — structurally identical, local edges:**
```
(:WorkflowDefSnapshot {key, version, name, kind})
(:WorkflowDefSnapshot)-[:START]->(:Step)
(:Step {stepUid, key, type, config})
(:Step)-[:TRANSITION {on, guard, order}]->(:Step)
```

**The key design decision — synthetic `Step.stepUid`.** `Step.key` is unique only within a def, so
it cannot back a `MERGE` (F2). Slice 1 adds `stepUid = "{defKey}:{version}:{stepKey}"` — globally
unique within each graph — as the MERGE-backing identity, with a range index + UNIQUE constraint in
**both** graphs. `Step.key`, `type`, `config` remain as-is; `Step.key` keeps its index (display/
traversal; §7.2 rationale intact — it was never meant to be a global identity).

*Why this over the alternatives:*
- **vs. guarded plain `CREATE` (no Step constraint), gating on snapshot non-existence** — avoids a
  new constraint but hits the FOREACH-can't-relink wall (transitions/START need to re-`MATCH`
  created steps, which a `FOREACH` body can't do), forcing either an app-layer existence check with
  a race window or an unindexed label-scan `MATCH`. Rejected: fights the repo's "every MERGE
  backed by a constraint" + "index the anchor" discipline.
- **vs. composite `Step UNIQUE {key, <discriminator>}`** — needs an extra discriminator property on
  every Step anyway; single-property `stepUid` is simpler and gives the later executor (K-022) a
  clean indexed point-lookup (`MATCH (:Step {stepUid: "{defKey}:{v}:{stepKey}"})`) to resolve
  `AT_STEP`/`RAN` targets.

**config/guard are opaque strings (F7).** The caller serializes `Step.config` and `TRANSITION.guard`
to strings; Slice 1 stores and returns them verbatim and never filters inside them. Guard
*evaluation* is K-022. **Slice 1 does not force the §13 decision** (confirmed, §Open Questions).

**Publish/materialize are idempotent single-graph MERGE queries (F5, live-verified).** Immutability
per version means re-publish/re-materialize of the same `key@version` is a structural no-op — MERGE
gives this for free.

### B3. DDL reconciliation (`scripts/bootstrap_schema.sh`) — additive only

The graph-dba gate adds, index-before-constraint (§7.1 rule):

**`bootstrap_reference()`** — after the existing `Step.key` index (line 51):
```
[index]      Step.stepUid          →  CREATE INDEX FOR (n:Step) ON (n.stepUid)
[constraint] Step unique {stepUid}  →  GRAPH.CONSTRAINT CREATE reference UNIQUE NODE Step PROPERTIES 1 stepUid
```

**`bootstrap_workspace()`** — new (the workspace graph has no `Step` DDL today, F1):
```
[index]      Step.key              →  CREATE INDEX FOR (n:Step) ON (n.key)
[index]      Step.stepUid          →  CREATE INDEX FOR (n:Step) ON (n.stepUid)
[constraint] Step unique {stepUid}  →  GRAPH.CONSTRAINT CREATE ws:{id} UNIQUE NODE Step PROPERTIES 1 stepUid
```

- Update DESIGN §7.1 (workspace table: add `Step | key, stepUid | UNIQUE 1 (stepUid)`) and §7.2
  (reference table: add the `Step.stepUid` UNIQUE row; keep the `Step.key` "traversal anchor,
  no constraint" row) **in the same change** (repo doc rule).
- **Additive & idempotent.** No existing data mutated; `ws:acme` gains the new `Step` index/
  constraint on its next (idempotent) bootstrap run. `WorkflowDef`/`WorkflowDefSnapshot`/
  `WorkflowRun`/`StepRun` DDL is unchanged. This is within the "additive schema to `ws:acme`"
  allowance — **flag any non-additive change** (there is none here).
- The pytest `conftest` and `test_queries.sh` both bootstrap via this script, so the new DDL flows
  into `ws:test` + `reference` automatically.

### B4. Query shapes (→ `docs/QUERIES.md`, new §11 "Workflow definitions & snapshots")

The graph-dba gate authors these into `QUERIES.md`, each live-verified + `GRAPH.PROFILE`d, then
mirrors them into `test_queries.sh`. Shapes below are the *design* (verified against `_probe`);
the gate owns the canonical bodies.

**B4.1 Publish a def (reference; idempotent) — verified F5.**
```cypher
// $key,$version,$name,$kind,$startKey, $steps=[{key,type,config}], $transitions=[{from,to,on,guard,order}]
MERGE (d:WorkflowDef {key: $key, version: $version})
  ON CREATE SET d.name = $name, d.kind = $kind
WITH d
UNWIND $steps AS s
  MERGE (st:Step {stepUid: $key + ':' + $version + ':' + s.key})
    ON CREATE SET st.key = s.key, st.type = s.type, st.config = s.config
WITH d
MATCH (start:Step {stepUid: $key + ':' + $version + ':' + $startKey})
MERGE (d)-[:START]->(start)
WITH d
UNWIND $transitions AS tr
  MATCH (from:Step {stepUid: $key + ':' + $version + ':' + tr.from})
  MATCH (to:Step   {stepUid: $key + ':' + $version + ':' + tr.to})
  MERGE (from)-[rel:TRANSITION {on: tr.on, order: tr.order}]->(to)
    ON CREATE SET rel.guard = tr.guard
RETURN d.key AS key, d.version AS version
```
*`TRANSITION` MERGE-key is `(from, on, order, to)` so distinct outcomes/orders are distinct edges;
`guard` is set-on-create (may be empty, never a match key). Every node MERGE is backed by a UNIQUE
constraint (`WorkflowDef {key,version}`, `Step {stepUid}`).*

**B4.2 Read a def subgraph (reference) — respects F6 (no `length(path)` ordering).**
```cypher
MATCH (d:WorkflowDef {key: $key, version: $version})
OPTIONAL MATCH (d)-[:START]->(start:Step)
OPTIONAL MATCH (from:Step)-[tr:TRANSITION]->(to:Step)
  WHERE from.stepUid STARTS WITH $key + ':' + $version + ':'
RETURN d.key, d.version, d.name, d.kind, start.key AS startKey,
       collect(DISTINCT {uid: from.stepUid, key: from.key, type: from.type,
                         to: to.key, on: tr.on, guard: tr.guard, order: tr.order}) AS edges
```
*Returns the def + its edge list; the app reconstructs order/topology (F6). The gate decides the
exact projection shape — likely two focused reads (one for steps, one for transitions) if the
combined projection is awkward; anchors on `WorkflowDef.key` index, and the transition scope predicate
uses the `Step.stepUid` index (`STARTS WITH` on an indexed string prefix — gate to confirm it plans
as an index scan, else scope via a `HAS_STEP`/`START`+`TRANSITION*` walk).*

**B4.3 List defs / get def by key (reference).**
```cypher
// list: newest-version-per-key or all; anchor on WorkflowDef.key index
MATCH (d:WorkflowDef) WHERE d.key > ''
RETURN d.key, d.version, d.name, d.kind ORDER BY d.key, d.version DESC LIMIT $limit
// get latest version for a key
MATCH (d:WorkflowDef {key: $key}) RETURN d ORDER BY d.version DESC LIMIT 1
```

**B4.4 Materialize a snapshot (workspace; idempotent) — same shape as B4.1, `WorkflowDefSnapshot`
label, scoped to `ws:{id}`.** The def structure comes as parameters from B4.2's read (F4 two-phase).
```cypher
MERGE (snap:WorkflowDefSnapshot {key: $key, version: $version})
  ON CREATE SET snap.name = $name, snap.kind = $kind
WITH snap
UNWIND $steps AS s
  MERGE (st:Step {stepUid: $key + ':' + $version + ':' + s.key})
    ON CREATE SET st.key = s.key, st.type = s.type, st.config = s.config
WITH snap
MATCH (start:Step {stepUid: $key + ':' + $version + ':' + $startKey})
MERGE (snap)-[:START]->(start)
WITH snap
UNWIND $transitions AS tr
  MATCH (from:Step {stepUid: $key + ':' + $version + ':' + tr.from})
  MATCH (to:Step   {stepUid: $key + ':' + $version + ':' + tr.to})
  MERGE (from)-[rel:TRANSITION {on: tr.on, order: tr.order}]->(to)
    ON CREATE SET rel.guard = tr.guard
RETURN snap.key AS key, snap.version AS version
```

**B4.5 Read / list snapshots (workspace).** Mirror B4.2/B4.3 with `WorkflowDefSnapshot` +
`ws:{id}` scope; anchor on `WorkflowDefSnapshot.key` index.

### B5. Repository → services surface (`server/falkorchat/`)

**`db.py` — new reference-graph seam (F3):**
```python
def reference_graph(db: FalkorDB | LazyFalkorDB) -> Graph:
    """Select the global `reference` graph (canonical WorkflowDef templates)."""
    return db.select_graph("reference")
```

**`repository.py` — new methods (reference-scoped take no `ws`; workspace-scoped take `ws`):**
| Method | Graph | Query | Notes |
|---|---|---|---|
| `publish_def(*, key, version, name, kind, start_key, steps, transitions)` | `reference` (write) | B4.1 | returns `{key, version}` |
| `get_def(*, key, version=None)` | `reference` (ro) | B4.2/B4.3 | latest if `version` None; `None` if absent |
| `list_defs(*, limit=50)` | `reference` (ro) | B4.3 | |
| `read_def_subgraph(*, key, version)` | `reference` (ro) | B4.2 | `{name,kind,start_key,steps,transitions}` — the materialize input |
| `materialize_snapshot(ws, *, key, version, name, kind, start_key, steps, transitions)` | `ws:{id}` (write) | B4.4 | idempotent |
| `get_snapshot(ws, *, key, version)` | `ws:{id}` (ro) | B4.5 | `None` if absent |
| `list_snapshots(ws, *, limit=50)` | `ws:{id}` (ro) | B4.5 | |

Add a `_reference()` helper (peer of `_graph(ws)`): `return db.reference_graph(self._conn)`.

**`services.py` — new methods:**
| Method | Does |
|---|---|
| `publish_workflow_def(ctx, *, key, version, name, kind, steps, transitions) -> dict` | **Validates the spec** (kind ∈ {conversation,process}; each step `type` ∈ {prompt,tool,decision,human,message,wait}; exactly one start step / `start_key` resolves to a step; every transition `from`/`to` references a declared step key; step keys unique within the def), **serializes** `config`/`guard` to strings, calls `repo.publish_def`. Raises typed errors (e.g. `WorkflowDefSpecError`) on invalid specs — nothing written. |
| `materialize_def(ctx, *, key, version) -> dict` | **Two-phase (F4):** `repo.read_def_subgraph` from `reference` → `repo.materialize_snapshot` into `ctx.ws`. Raises `WorkflowDefNotFoundError` if the def version is absent in `reference`. Idempotent (re-materialize no-ops). |
| `get_workflow_def` / `list_workflow_defs` / `get_snapshot` / `list_snapshots` | thin pass-throughs. |

New exception types in `repository.py` (re-exported by `services`, mirroring `MemberIdCollisionError`):
`WorkflowDefNotFoundError`, `WorkflowDefSpecError`.

**Tenant/graph seam (F3).** Def authoring/reading is **global** — it uses `db.reference_graph`, not
`workspace_graph(ctx.ws)`, and its repository methods **omit `ws`**. Only materialization consumes
`ctx.ws` (the target workspace). `CallContext` is unchanged. `config.get_context` is unchanged.

**Thin API surface (optional final step, follows §14.4 pattern; MCP deferred to K-023/K-024):**
`POST /workflow-defs` → `publish_workflow_def`; `GET /workflow-defs[?limit=]` → `list_workflow_defs`;
`GET /workflow-defs/{key}[?version=]` → `get_workflow_def`;
`POST /workflow-defs/{key}/versions/{version}/materialize` → `materialize_def`;
`GET /workspaces/{ws}/snapshots` → `list_snapshots`. Add Pydantic request models to `schemas.py`
(size-bounded per §14.4: step/transition list caps, `config`/`guard` string caps → RAM rule 6).

### B6. Sequencing (Slice 1 build order)

1. **K-020 graph-dba gate:** reconcile + add `Step.stepUid` DDL (reference) in `bootstrap_schema.sh`;
   author B4.1–B4.3 into `QUERIES.md` §11 (live-verify + PROFILE); add `test_queries.sh`
   reference-section assertions (enumerated, 149 → ~160). Update DESIGN §7.2 + §6. **Gate: suite
   green at ~160.**
2. **K-020 tdd:** `db.reference_graph`; repository `publish_def`/`get_def`/`list_defs`/
   `read_def_subgraph` + exceptions; `services.publish_workflow_def` (with spec validation) +
   def-read pass-throughs. Reference-wiping pytest fixture (F8). **Gate: pytest green.**
3. **K-021 graph-dba gate:** add workspace `Step.key`/`Step.stepUid` DDL; author B4.4–B4.5 into
   `QUERIES.md` §11 (live-verify + PROFILE); add `test_queries.sh` workspace-section assertions
   (~160 → ~172). Update DESIGN §7.1. **Gate: suite green at ~172.**
4. **K-021 tdd:** repository `materialize_snapshot`/`get_snapshot`/`list_snapshots`;
   `services.materialize_def` (two-phase orchestration); structural-parity service test
   (publish → materialize → compare). **Gate: pytest green.**
5. **(optional) thin REST surface** + `schemas.py` models + `api` tests (TestClient contract).

Each step keeps both suites green before the next. Steps 1–2 (K-020) land independently of 3–4
(K-021), but K-021 depends on K-020 (materialization consumes published defs).

### B7. Test strategy & expected suite count

**`test_queries.sh` (the gate baseline — enumerate to hit; final number pinned by the gate).**
Current **149**. Candidate new assertions:

- *K-020 (reference, ~+11 → ~160):* publish authors full subgraph (steps created / START edge /
  TRANSITION edges); `Step.stepUid` constraint blocks duplicate; read-def returns the subgraph;
  get-def latest vs specific version; list-defs; **idempotent re-publish creates 0 nodes**;
  PROFILE: read-def anchors on `WorkflowDef.key` (no label scan); PROFILE: step lookup anchors on
  `Step.stepUid`.
- *K-021 (workspace, ~+12 → ~172):* materialize creates the local snapshot subgraph
  (snapshot / local steps / local START / local TRANSITION); workspace `WorkflowDefSnapshot`
  composite constraint blocks dup `{key,version}` (not asserted today); workspace `Step.stepUid`
  constraint blocks dup; read-snapshot returns the subgraph; get/list snapshot; **idempotent
  re-materialize creates 0 nodes**; PROFILE: snapshot read anchors on `WorkflowDefSnapshot.key`;
  PROFILE: local step lookup anchors on `Step.stepUid`.

**Expected new `test_queries.sh` count: ≈ 172/172** (149 → ~160 after K-020 → ~172 after K-021).
This is the enumerated *target*; the graph-dba gate pins the exact number when authoring the
assertions and records it in the item's done-condition (per the K-008 "≈126 → ~135, enumerated in
the gate" precedent).

**pytest (baseline 156).** Repository integration tests against `reference` (K-020) + `ws:test`
(K-021) with the F8 reference-wiping fixture; service tests for spec validation (invalid kind/type,
missing start, dangling transition endpoint, duplicate step key → raises, nothing written) and the
two-phase materialize (def-not-found → raises; publish→materialize→structural parity; idempotent
re-materialize). Optional API contract tests via `TestClient`. Net new count is the tdd-engineer's
to report; the baseline must stay green.

**Altitudes:** contract (queries in `test_queries.sh`) · integration (repository against live
`reference`/`ws:test`) · unit (service validation with a fake repo) · optional contract (API
TestClient). No live LLM needed (Slice 1 is definition + materialization only).

### B8. Risks & open questions

- **§13 guard expression language — NOT forced by Slice 1 (confirmed, F7).** `Step.config` and
  `TRANSITION.guard` are stored as opaque serialized strings and never queried inside; guards are
  *evaluated* at run time (K-022). The decision (expr lib vs minimal DSL) is surfaced to the user at
  K-022's architect design pass, not here. *If the implementer finds Slice 1 must query inside
  `config`/`guard` (it shouldn't per this design), stop and escalate rather than committing to a
  guard language.*
- **`ws:acme` change is additive-only (constraint (b) satisfied).** Slice 1 adds a `Step` index +
  UNIQUE constraint to the workspace graph (idempotent bootstrap) and, if K-024's proof flows later
  run, materialized snapshot nodes — all additive. **No non-additive change to `ws:acme` in Slice 1.**
- **Global `reference` writes.** Publishing a def mutates the shared, replicated `reference` graph
  (additive, immutable-per-version). With the M1 hardcoded seam this is unauthenticated like
  everything else. **Open question (non-blocking for Slice 1):** authz for global def publish — who
  may publish/supersede a canonical def? Ties to K-016 auth; note it, don't gate on it.
- **Cross-graph materialization is not atomic across the boundary (F4).** Read `reference` → write
  `ws:{id}` are two operations. A failure between them leaves `reference` untouched and the
  workspace MERGE idempotent, so a retry completes cleanly — no torn state. Documented, accepted.
- **`STARTS WITH` on `stepUid` for scoping transition reads (B4.2)** — the gate must PROFILE this;
  if it degrades to a label scan, scope transitions via a `(:WorkflowDef)-[:START]->()-[:TRANSITION*]`
  walk or a `HAS_STEP` edge instead. A live-verify item for the gate, not a design blocker.
- **Reference test isolation (F8).** Reference-graph pytest tests need a wiping/unique-key fixture,
  else they collide across tests — the tdd-engineer must add it.

---

## Ready to implement — summary

**Plan:** `falkor-chat/docs/archive/plans/m3-workflow-engine.md` (repo-relative)

**Part A (M3 decomposition):** K-020 def model in `reference` → K-021 snapshot materialization into
`ws:{id}` (**Slice 1**) → K-022 run/step-run executor + guard eval (§13 decision here) → K-023 chat
linkage (`TRIGGERED_BY`/StepRun `EMITTED`) → K-024 two proof flows (conversational + process,
the DoD proof) → K-025 QA acceptance ⇒ M3 ✅. Spine is essentially linear (each slice consumes the
prior); only K-025 plan-authoring and a non-DoD workflow UI could parallel.

**Slice 1 handoff:**
- **graph-dba gate:** add `Step.stepUid` index+UNIQUE to `bootstrap_schema.sh` (reference *and*
  workspace; workspace also gets `Step.key`); author publish/read/list/get-def (K-020) +
  materialize/read/list-snapshot (K-021) queries into `QUERIES.md` §11, live-verified + PROFILEd;
  add enumerated `test_queries.sh` assertions raising **149 → ~160 (K-020) → ~172 (K-021)**; update
  DESIGN §6/§7.1/§7.2. Core query shapes are live-verified (F5) and idempotent.
- **tdd-engineer:** `db.reference_graph` seam; reference-scoped repo methods (`publish_def`,
  `get_def`, `list_defs`, `read_def_subgraph`) + workspace-scoped (`materialize_snapshot`,
  `get_snapshot`, `list_snapshots`); services `publish_workflow_def` (spec validation) +
  `materialize_def` (two-phase); typed errors; reference-wiping pytest fixture; optional thin REST
  surface. Keep pytest (156) green.

**Expected new `test_queries.sh` count:** **≈ 172/172** (enumerated target; gate pins the exact
number). pytest stays green above 156.

**Decision-point flags:** (a) §13 guard language — **NOT forced by Slice 1** (opaque config/guard;
decided at K-022). (b) `ws:acme` — **additive-only** in Slice 1 (new `Step` DDL; idempotent). Open
(non-blocking): authz for global `reference` def publish (ties to K-016).
