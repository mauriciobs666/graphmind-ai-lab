# falkor-chat — Design Document & Blueprint

> **Philosophy:** FalkorDB graph for *everything* — reference data, workflow definitions
> and runs, chat history, and user/workspace information. No second store for the primary
> domain. One engine, one query language (OpenCypher), one operational model (Redis).

**Status:** Draft v0.3 — thread-scoped model, DayBucket removed
**Date:** 2026-06-06
**Owner:** the repo maintainer

---

## 1. Decisions locked in

This section is the **single authoritative decision register**. §1.1 holds the top-level axes;
§1.2 the detailed locked design decisions; §1.3 the decided-but-pending M2 stack. Each row is the
authoritative *statement* of a decision; where the body already explains the mechanics, the row
links there and does not re-explain. `AGENTS.md` carries only a terse pointer index back here.

### 1.1 Top-level axes

| Axis | Decision | Consequence |
|---|---|---|
| **Chat type** | **Hybrid** — humans chat in channels *and* an AI participant answers from the graph (GraphRAG) | Messages have a `role` (`user`/`assistant`, derived from the author's label); AI is a first-class `Agent` author; retrieval = vector + traversal |
| **Tenancy** | **Per-workspace/team graph** — the graph boundary is a workspace, many users share it | One named graph per workspace; each workspace must fit one shard's RAM; workspaces distribute across a Redis Cluster |
| **Workflows** | **General workflow engine** — one definition model serves both conversational/agent flows and business processes | Definitions (templates) are reusable + versioned; runs are per-workspace execution traces |

### 1.2 Locked design decisions (detailed register)

> Each row is the authoritative statement of the decision; the "Detailed in" column links to the
> body section that explains the mechanics (or to QUERIES.md where the mechanics are canonical).
> Do not re-explain the mechanics here.

| Decision | Rationale / consequence | Detailed in |
|---|---|---|
| **Single store** — FalkorDB holds all domain data; no secondary store | Project philosophy: one engine, one query language, one ops model | Philosophy header, §2 |
| **Thread-scoped `NEXT` linked list** | Users read threads, not channel feeds; O(1) append; Thread stays sparse | §5.2 |
| **No DayBucket** *(rejected alternative)* | Designed for channel-wide ordering; dropped when the thread-scoped model was chosen | §5.2 |
| **`Thread` owns `HEAD` + `TAIL` pointers** | Thread stays sparse — exactly 2 edges regardless of message count | §5.2 |
| **`Message.role` inline property; values `user`/`assistant` derived server-side** from the author label (`User→user`, `Agent→assistant`), never trusted from the caller | Filter by role without traversing `POSTED_BY`; agents author first-class (K-007) | §5.1, §5.2 |
| **`coalesce(u.userId, a.agentId)` for member identity** (two indexed `OPTIONAL MATCH` + `coalesce`) | `User` has `userId`, `Agent` has `agentId` — both are members; anchored lookup avoids the `OR`-scan | QUERIES §2 |
| **Vector indexes via DDL**, not a procedure | `db.idx.vector.createNodeIndex` is not registered on this build | §2, §7.1 |
| **Index before constraint, always** | `GRAPH.CONSTRAINT CREATE` requires a pre-existing range index | §2, §7.1 |
| **`Message.embedding` inline as `vecf32`** | Single-query vector + traversal hybrid retrieval | §5.2, §7.3 |
| **Vector score is cosine *distance*** (0 = identical) → `ORDER BY score ASC` | Most-similar-first ranking | §7.1, §8 |
| **`status` as a property, not a label** | Avoids re-labeling churn on state changes; index it for "all running" reads | §6.2 |
| **`ctx` / `input` / `output` are flat/serialised strings** | FalkorDB stores scalars + scalar lists only — no nested maps; never query inside them | §6.2 |
| **`Message.threadId` denormalized inline, unindexed** | Nav metadata for §9.2/§5 rows; HEAD/NEXT walk stays canonical; unindexed saves RAM/write cost (K-007) | §5.1 |
| **Guarded-CREATE write paths** (`FOREACH`+`CASE` per path) with an always-returned status row; **no MERGE on `Message`** | Retry replay is a no-op (`dupMsg`); first-post race refused (`hadHead`); uniqueness constraint is the backstop (K-007) | §5.3, §9 |
| **Composite `(createdAt, msgId)` keyset cursor** (`ReadCursor.lastReadAt`/`lastReadMsgId`) | Timestamp alone is not a total order — same-ms ties skipped rows; cursor reads are lossless (K-007) | QUERIES §9.1/§9.3 |
| **Member ids are namespace-unique across `User`/`Agent`**; `ensure_user`/`ensure_agent` are v2 guarded-CREATE queries returning `(created, existed, collided)`; cross-label collision refuses (`MemberIdCollisionError`) | A shadow node with the other label's id eclipses it in every `coalesce` lookup (K-010) | QUERIES §2/§7 |
| **Identity source of truth — the `identity` graph is authoritative (standalone)**, not an external-IdP projection | Self-contained system; the `identity` graph owns user identity + auth principals; per-workspace `User` nodes are membership projections of it; steers K-016 auth | §3, §14.3 |

### 1.3 M2 stack (decided 2026-07-04, pending implementation)

> User-approved M2 stack. Locked here; implemented in K-008/K-013 (see docs/BACKLOG.md). Numbers
> detailed in §11 (RAM) and §12 (M2 roadmap). **K-042 (Landing 1) note:** the rows below are the
> *shipped defaults*, not env vars — they are now `config/models.json`'s per-kind `defaults` +
> the shared `config/opencode.example.json`'s provider entries. Model choice is a config-file
> edit + restart (§14.8), never a code/env change. `EMBEDDING_DIM` remains an env var
> (`FALKORCHAT_EMBEDDING_DIM`) — it is DDL-time/write-path input, not a model choice (§14.8).

| Component | Decision | Rationale | Detailed in |
|---|---|---|---|
| Embedding model | **Qwen3-Embedding-0.6B** (GGUF, Q8_0) | Best small-model MTEB quality; 100+ languages (PT-BR + EN); ~0.6 GB resident | §11, §12 |
| Vector dimension | **`EMBEDDING_DIM=1024`** (MRL 512/256 later) | Native dim; ~12.5 KB/message with HNSW — the §11 RAM line | §11 |
| Agent LLM | **Qwen3-4B-Instruct-2507** Q4_K_M (non-thinking) | RAG answering, not CoT; low latency; `-Thinking-2507` a drop-in for M3 | §12 |
| Runtime | **LM Studio** on the Windows host (OpenAI-compatible), reached from WSL2 (mirrored networking → localhost) | Reuses the severino path; zero new moving parts; Ollama fallback | §10, §12 |
| VRAM budget | **6 GB dedicated** (RTX 4050) — embedder + 4B LLM co-resident | Do not plan around shared-RAM spill | §11 |
| Upgrade path | **`qwen3-embedding:4b`** — same family, same 1024-dim MRL | Re-embed only; no schema change | §12 |

---

## 2. The hard FalkorDB constraints this design is shaped around

These are not style choices — they are engine facts that the topology must respect.

1. **In-memory, RAM-bound.** The whole graph + its sparse adjacency matrices + indexes live in RAM. **Memory is the binding constraint**, sized per graph.
2. **A single graph lives entirely on one shard.** FalkorDB does *not* split one graph across shards (no Neo4j-Fabric equivalent). We scale by spreading *many* workspace graphs across cluster shards — a natural fit for per-workspace tenancy. **A single workspace can never outgrow one node's RAM.**
3. **Relationships cannot cross graphs.** An edge can only connect two nodes in the *same* named graph. This is the single most important fact for the reference-data / workflow-definition split (see §4 and §6). Cross-graph references are carried as **properties (keys), resolved at query time**, or by **materializing a copy** of the shared subgraph into the workspace graph.
4. **OpenCypher subset, not Neo4j.** No APOC, no GDS, no Fabric. Algorithms are built-in `algo.*` procedures; full-text/vector are `db.idx.*` procedures; profiling is the `GRAPH.PROFILE` *command*, not a `PROFILE` keyword prefix.
5. **Supernodes are dense matrix rows.** Less catastrophic than pointer-chasing engines (traversal is matrix algebra), but a Channel with millions of `HAS_MESSAGE` edges is still a dense row that costs RAM and compute. We avoid it with a linked-list + time-bucket pattern (§5).

> **Live-verified** on this deployment (now pinned to `v4.18.11`, module `41811`, Redis 8.6.3;
> originally probed on edge/main, re-verified 2026-07-09 via the full query suite):
> `vectorset` module also loaded. The findings below in §7 reflect confirmed behavior — not docs
> assumptions. Details: cross-graph edges confirmed silent (no error, MATCH returns 0); constraint
> requires an existing range index first; vector indexes use DDL syntax, not a procedure call;
> `db.idx.vector.createNodeIndex` is **not registered** in this build. See §7 for what's indexed and
> why; the executable DDL is `scripts/bootstrap_schema.sh`.

---

## 3. Graph topology (multi-graph layout)

We use **four classes of named graph**. Each is an independent Redis key; edges stay within a class.

```
┌─────────────────────────────────────────────────────────────────────┐
│ identity                 (1 graph, global)                            │
│   Global user identity, auth principals, cross-workspace membership   │
│   Read-mostly. Replicated. Small.                                     │
├─────────────────────────────────────────────────────────────────────┤
│ reference                (1 graph, global, read-mostly)               │
│   Domain reference data / ontology / catalogs                         │
│   Canonical WorkflowDef templates (topology-immutable, K-034)         │
│   Tool registry, prompt templates                                     │
│   Replicated; served via GRAPH.RO_QUERY                               │
├─────────────────────────────────────────────────────────────────────┤
│ ws:{workspaceId}         (N graphs, one per workspace)  ← hot path    │
│   Workspace-local Users (membership projection of identity)           │
│   Channels, Threads, Messages (chat history)                          │
│   WorkflowRun + StepRun execution traces                              │
│   Chunks/Documents + embeddings (GraphRAG corpus)                     │
│   Extracted Entities + mentions                                       │
│   Materialized copy of the WorkflowDef versions this ws uses          │
├─────────────────────────────────────────────────────────────────────┤
│ (optional) analytics:{...}  rollups / cross-workspace aggregates      │
└─────────────────────────────────────────────────────────────────────┘
```

**Why per-workspace graphs (not one mega-graph with a `workspaceId` property):**
- **Blast radius:** corruption, a runaway query, or a delete is scoped to one workspace.
- **Sharding:** each `ws:{id}` graph hashes to a cluster slot → workspaces spread across shards automatically. A mega-graph would pin *all* tenants to one shard's RAM.
- **Speed:** every traversal starts already scoped — no `WHERE n.workspaceId = $w` filter threaded through every query, no shared dense matrices.
- **Lifecycle:** archive/export/delete a workspace = one `GRAPH.DELETE`.

**The cost:** cross-workspace queries (admin analytics) need fan-out across graphs at the
app layer or a dedicated `analytics` rollup graph. Accepted — cross-workspace reads are rare
and not latency-critical.

**Naming conventions** (project-wide):
- Labels: `PascalCase` — `User`, `Channel`, `Message`, `WorkflowRun`
- Relationship types: `UPPER_SNAKE` — `POSTED_BY`, `REPLY_TO`, `AT_STEP`
- Properties: `camelCase` — `userId`, `createdAt`, `embedding`
- Graph keys: `ws:{workspaceId}`, `reference`, `identity`

---

## 4. The cross-graph problem & the definition/instance split

Because **edges can't cross graphs**, a `WorkflowRun` in `ws:acme` cannot have a real
relationship to a `WorkflowDef` that lives in `reference`. Two ways to bridge:

| Approach | How | When |
|---|---|---|
| **A. Property reference** | Run stores `defKey` + `defVersion` as properties; app resolves the def by querying `reference` | Cheap, always correct, but no traversal across the boundary |
| **B. Materialize (chosen for defs)** | On workflow *publish*, copy the def subgraph (topology-immutable per version, K-034) into each workspace graph that uses it | Real edges → runs traverse their own steps locally; def graphs stay small and are duplicated cheaply |

**Decision:** canonical definitions live in `reference` (single source of truth, versioned).
When a workspace first uses `defKey@v`, we **materialize that version's step subgraph into
`ws:{id}`** under a `WorkflowDefSnapshot`. Runs then have real, local edges to their steps —
fast, self-contained. **The enforced guarantee (K-034):** topology (steps, transitions, start)
is immutable per version — a re-publish/re-materialize whose topology differs from what's
stored is rejected (`409 WorkflowDefConflictError`) before any write, so the snapshot never
silently drifts from the def it was materialized from. Properties (`name`, `kind`, step
`config`, transition `guard`) are create-only — a differing resubmit of those stays a silent
no-op, unchanged. Same pattern applies to any *small, shared, read-mostly* reference subgraph a
hot traversal needs to walk (e.g. an ontology fragment).

> Large reference catalogs that are only *looked up* (not traversed from workspace nodes) stay
> in `reference` and are reached by property key — no materialization.

---

## 5. Chat model (hybrid: humans + AI)

### 5.1 Arrow notation

```
// Membership
(:User)-[:MEMBER_OF {role, joinedAt}]->(:Channel)
(:Agent)-[:MEMBER_OF {role:'assistant'}]->(:Channel)   // AI is a first-class member

// Channel → Thread → Message hierarchy
(:Channel)-[:HAS_THREAD]->(:Thread {threadId, title, createdAt, updatedAt})
(:Thread)-[:HEAD]->(:Message)                          // first message in thread (set once)
(:Thread)-[:TAIL]->(:Message)                          // last message (updated on each append)
(:Message {msgId, text, role, createdAt, threadId})-[:NEXT]->(:Message)  // thread-scoped linked list

// Authorship & replies
(:Message)-[:POSTED_BY]->(:User)                       // human author
(:Message)-[:POSTED_BY]->(:Agent)                      // AI author
(:Message)-[:REPLY_TO]->(:Message)                     // explicit quote/reply (optional)

// GraphRAG corpus
(:Document {documentId})-[:HAS_CHUNK]->(:Chunk {chunkId, text, embedding: vecf32})
(:Chunk)-[:DERIVED_FROM]->(:Message)
(:Entity {entityId, name, type})<-[:MENTIONS]-(:Message)
(:Chunk)-[:ABOUT]->(:Entity)

// Workflow ↔ chat linkage (all within ws graph)
(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)
(:StepRun)-[:PRODUCED]->(:Message)                     // step-emitted chat message (D2 — NOT the K-013 EMITTED)
```

**Key properties:**
- `Message.role` — `'user'` | `'assistant'` (fast filter without traversing `POSTED_BY`).
  **Derived server-side from the author's node label** (`User → user`, `Agent → assistant`),
  never trusted from the caller (K-007).
- `Message.threadId` — **denormalized, deliberately unindexed** navigation metadata (K-007):
  lets §9.2/§5 result rows point back to their thread without a traversal; §9.1's HEAD/NEXT
  walk stays the canonical thread read. `null` on pre-K-007 rows until the one-off backfill
  (`scripts/backfill_thread_ids.sh`, QUERIES.md §4.x) runs.
- `Thread.updatedAt` — bumped on every new message; drives "recent threads" listing
- `Message.embedding` — inline `vecf32`; no separate node needed

### 5.2 Why these choices (traversal cost)

- **Thread-scoped `NEXT` linked list.** Reading a thread is a bounded `NEXT*` walk from
  `Thread HEAD`. Thread stays permanently **sparse** — always exactly 2 edges (HEAD and TAIL)
  regardless of message count. Append is O(1): link new message to current TAIL, move TAIL
  pointer, all in one atomic query.
- **No direct `Channel→Message` edges.** Channel fan-out is bounded by thread count, not message
  count — eliminating the Channel supernode risk entirely. Channel-level time queries use the
  `Message.createdAt` and `Thread.updatedAt` range indexes, not edge traversal.
- **`Message.embedding` inline as `vecf32`** rather than a separate `Embedding` node: a single
  query seeds with `db.idx.vector.queryNodes` then traverses `REPLY_TO` / `MENTIONS` for precise
  context — hybrid retrieval in one round trip.
- **`Message.role` as a property, not derived from `POSTED_BY` label**: lets the app filter by
  role (`WHERE m.role = 'assistant'`) without an extra hop.
- **AI as `Agent` author, not a magic flag**: assistant messages share one timeline with human
  messages; a workflow step's `PRODUCED` edge (§6.2, D2) and an answer's `EMITTED` seed provenance
  (§9 / K-013, QUERIES §10) make an AI message's origin explicit and auditable.

### 5.3 Thread append (write path)

Two cases — both must be a single `GRAPH.QUERY` (atomic):

- **First message in a thread** (no `HEAD`/`TAIL` yet) — create the message, link
  `Thread -[:HEAD]-> m` and `Thread -[:TAIL]-> m`, and attach `(m)-[:POSTED_BY]->(author)`.
- **Subsequent messages** — match the current `TAIL`, link `prev -[:NEXT]-> m`, move `TAIL` to
  `m` (delete the old `TAIL` edge, create the new one), and attach `(m)-[:POSTED_BY]->(author)`.

Both bump `Thread.updatedAt`. The service picks the variant by checking whether the thread
already has a `HEAD` (§14 keeps this dispatch inside `post_message`), then re-dispatches on the
v2 **status row** each write returns (K-007): a lost first-post race (`hadHead`) retries as
subsequent, a TAIL-less subsequent retries as first, and a replayed `msgId` (`dupMsg`) is
idempotent success — see the §9 table and QUERIES.md §4. The re-dispatch loop is bounded at 4
attempts — a tripwire, not a real retry budget: ping-pong between the two paths is impossible by
contract (a headed thread always has a TAIL), so hitting the bound means the invariant broke.
`createdAt` comes from the service's lock-guarded monotonic per-process clock
(`max(clock, last+1)`), never the raw wall clock — same-millisecond ties across writes on one
process are impossible at the source, which is what makes the §9 `(createdAt, msgId)` composite
order well-defined.

> **Canonical Cypher: `docs/QUERIES.md` §4.** The exact, live-verified queries live there and
> nowhere else — this section describes their *shape* only, so the two never drift. Every message
> must carry `(m)-[:POSTED_BY]->(author)`: the canonical thread-read path requires that edge, so a
> message written without it is invisible to thread reads.

### 5.4 Supernode watch

`Channel` (via `HAS_THREAD`) and popular `Entity` nodes are the remaining risks. Channel fan-out
is threads-per-channel — manageable. Entity fan-out (`MENTIONS`) grows with corpus size; mitigate
by capping entity extraction per message and partitioning `MENTIONS` by relationship type if
needed. Re-evaluate with `GRAPH.PROFILE` once real data lands.

---

## 6. Workflow engine model (general)

A definition is a directed graph of steps; a run is an execution trace that walks it.

### 6.1 Definition (canonical in `reference`, materialized into `ws:{id}`)

```
(:WorkflowDef {key, version, name, kind})         // kind: 'conversation' | 'process'
(:WorkflowDef)-[:HAS_STEP]->(:Step)                // index-anchored containment (all steps of a def)
(:WorkflowDef)-[:START]->(:Step)                   // the entry step
(:Step {stepUid, key, type, config})               // type: agent|prompt|tool|decision|human|message|wait
(:Step)-[:TRANSITION {on, guard, order}]->(:Step)  // edge-labeled state machine
```

> **`Step.stepUid` is the MERGE-backing identity** (M3 Slice 1 / K-020). A step `key` is unique only
> *within a def*, so it can't back a `MERGE`; every Step carries a synthetic
> `stepUid = "{defKey}:{version}:{stepKey}"` (globally unique within each graph) with an index +
> `UNIQUE` constraint in both `reference` and `ws:{id}` (§7.1/§7.2). `key`/`type`/`config` are the
> display/behaviour props. **`HAS_STEP`** is the def→step containment edge: without it the only
> def→step link is `START`, so reading "all steps of a def" would label-scan every `Step` in the graph
> — and the `stepUid`-prefix `STARTS WITH` alternative live-profiles as a label scan on this build.
> `HAS_STEP` keeps step/transition reads anchored on the def's index (`Node By Index Scan`, verified).
> Canonical publish/materialize/read Cypher: **`QUERIES.md` §11**.

`type` unifies conversational and business flows. **What the engine executes today (K-024 U2):**
- **`agent` → LLM-native node** (M3 executor / K-022): the `config` string carries a plain-language
  `systemPrompt` + an author-set tool fence + bounds; the model runs as a bounded, tool-scoped agent
  loop. This is the type the triage proof uses. **`agent` with no LLM wired returns an empty stub
  result** — deliberate, and load-bearing for the offline test estate.
- `human` / `decision` / `wait` → business processes, **implemented and proven** by the
  `access-request@v1` proof flow (K-024, §6.3):
  - **`human`** — parks the run awaiting an assignee. `config.waitsForHuman: true` is **mandatory**
    and enforced at publish; the `StepRun.output` carries an `awaiting` envelope
    (`prompt`/`fields`/`assignee`), so a client learns what the run is waiting for from
    `GET /workflow-runs/{id}/step-runs` with no new query.
  - **`decision`** — **no side effect at all**: its semantics are entirely its outgoing guards. With
    no outgoing transition it is a terminal outcome node (the run ends `done`).
    > ⚠️ **Not enforced (residual).** A `decision` step whose outgoing transitions are *all*
    > conditional, and which does not declare `waitsForHuman`, self-loops to budget exhaustion if
    > none ever fires — there is no symmetric check forcing either an unconditional default arm or
    > a park declaration. The equivalent check for `human`/`wait` steps *is* enforced (above); doing
    > the same here would retro-reject existing fixtures, so it is a deliberate gap, not an oversight
    > (K-029).
  - **`wait`** — **signal-driven, not timer-driven.** This system has **no scheduler** (decision
    D-C); a `wait` step parks exactly as `human` does and is released by an **external signal**
    delivered through `POST /workflow-runs/{id}/input`. Mechanically it *is* `human` to the engine —
    only the `awaiting.kind` string differs, and it carries the same mandatory `waitsForHuman: true`.
    Real timers/scheduled wakeups are backlog **K-028**, not a gap in this model.
- `prompt` / `message` / `tool` → agent-adjacent flows (LLM call, post a message, invoke a tool) —
  **not implemented**: `executor._execute_step` raises `NotImplementedError` naming
  `docs/archive/plans/m3-process-flow.md` (the documented typed-handler seam, decision D-E). A deliberate
  behaviour change from the pre-K-024 silent no-op, so an unimplemented type fails a run loudly
  instead of "succeeding" having done nothing.

`Step.config` and `TRANSITION.guard` are **opaque serialized strings** parsed app-side only (rule 8) —
`type:'agent'` needs **no DDL** (whitelist-only add). The `agent` config deserializes to
`{mode, systemPrompt, tools[], permissions{}, waitsForHuman, maxIterations}`; `waitsForHuman:true` is the
explicit suspend signal for a node that parks awaiting a human reply (only intake, in the triage flow).

`config.requiredTools: [<tool name>, ...]` (K-027 item 2, `docs/plans/must-post-engine-contract.md`)
is `waitsForHuman`'s sibling in the same opaque config, and the same authoring convention: an
`agent`-typed step names a subset of its own `config.tools` that must be successfully dispatched at
least once before the node's turn ends. Absent/empty ⇒ no obligation (every def shipped before this
item, unchanged). Enforced inside `_run_agent_node` at both of its own exit points — never inside
`_drive_loop` — with a violation trace-and-continuing rather than failing or parking the run: an
unconditional `_log.warning`, plus a `must_post_violation` trace entry on debug/traced runs only.
Validated at publish by a fourth invariant in `services._validate_def_spec`, mirroring the
`waitsForHuman` check's own shape (list-of-strings, `agent`-only, and a subset of `config.tools`).

**`TRANSITION.guard` is LLM-native *and* deterministic (K-022 + K-024, supersedes the old
"expression" wording — §13 resolved).** The guard string is one of:
- **empty `""`** — an unconditional/default transition (**lowest** priority; fires whenever reached);
- a `{kind:'llm', text}` discriminator — judged in natural language against run context (a structured
  boolean verdict + traced rationale);
- a **`cmp`-family** discriminator (K-024) — `{kind:'cmp', path, op, value}` plus the combinators
  `all` / `any` / `not`. A closed comparator, **not** an expression language: whitelisted ops, two
  whitelisted path roots (`ctx.` / `output.`), depth/width/node caps, no parser and no `eval`.
  It is **total at drive** (a missing path ⇒ `False`, never a raised error) and **strict at
  publish** (a typo'd `op` or an unwhitelisted path root is a `WorkflowConfigError` at seed time, not
  a run that parks forever). This is what makes an LLM-free `kind:'process'` flow possible.

A would-be **`expr`** kind is still a deliberate `NotImplementedError` seam — no expression library is
built here (the deterministic family is deliberately named `cmp` to keep that literally true).

**Two corrections to earlier wording (K-024 finding F-1):**
1. **`on` is descriptive only.** `TRANSITION.on` and `StepResult.on` are **vestigial** — nothing in
   the engine reads either. `on` is a human-readable outcome label, *not* "the event that fires the
   transition"; the guard alone decides. This is exactly the fact `docs/plans/must-post-engine-contract.md`
   leans on to justify a must-post violation as trace-and-continue rather than a new `StepResult.on`
   outcome — a design that changed `on` conditionally on violation would be inventing meaning for a
   field the engine has never read.
2. **Guards are not evaluated in plain `TRANSITION.order`.** The sort key is
   `(guard == "", order)` — i.e. **conditional guards first**, with `order` as a tie-break *within*
   each class, then first-firing wins. That is what makes "conditional beats unconditional" work and
   an empty guard a true default arm.

One model, both worlds. Runtime evaluation & the run/step-run schema: §6.2; verified Cypher:
**`QUERIES.md` §12**.

### 6.2 Run (per-workspace, real local edges to the materialized def)

```
(:WorkflowRun {runId, defKey, defVersion, status, startedAt, endedAt,
               ctx, trace, maxSteps, stepCount, waitingThreadId})   // status ∈ running|waiting|done|failed
(:WorkflowRun)-[:OF_DEF]->(:WorkflowDefSnapshot {key, version})   // local, materialized
(:WorkflowRun)-[:AT_STEP]->(:Step)                               // current position (cleared on terminal)
(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)                       // the @mention that started it (FR-7)
(:WorkflowRun)-[:HAS_STEP_RUN]->(:StepRun)                       // membership
(:WorkflowRun)-[:LAST_STEP_RUN]->(:StepRun)                      // TAIL pointer → the NEXT anchor (M4)
(:StepRun {stepRunId, stepKey, status, startedAt, endedAt, input, output})
(:StepRun)-[:RAN]->(:Step)                                       // which def step
(:StepRun)-[:NEXT]->(:StepRun)                                   // execution order (audit trail)
(:StepRun)-[:PRODUCED]->(:Message)                               // step-emitted chat message (D2)
(:StepRun)-[:TRACED]->(:TraceEvent)                              // debug-only trace record (FR-4)
(:TraceEvent {traceId, seq, kind, at, payload})                  // debug runs only; payload = flat string
```

> `ctx` (on `WorkflowRun`), `input`/`output` (on `StepRun`) and `payload` (on `TraceEvent`) are **flat,
> serialised strings**, not nested maps — FalkorDB stores only scalars and scalar lists. Queries never
> filter *inside* them (see §1.2).

**Run-model additions (M3 executor, K-022):**
- **`trace`/`maxSteps`/`stepCount`** on `WorkflowRun` — the debug-instance flag (§5 gates all trace
  writes), the run-level step budget (§7, DS default 12 — a **tripwire**, see the note below), and
  the executed-step counter the atomic advance bumps. **`waitingThreadId`** denorms the parked run's thread so resume is an index-anchored
  lookup (rides the existing `status` index — no new index; QUERIES §12.9). `endedAt` stamps terminal.
- **`LAST_STEP_RUN` — the tail pointer (M4).** Mirrors the locked `Thread` HEAD/TAIL pattern (§5.2):
  `record_step_and_advance` reads it to find the previous `StepRun`, hangs `NEXT`, and moves the tail —
  all in one query → **O(1) atomic advance, no chain-walk / label scan**. One edge per run (the tail
  moves, it does not accumulate).
- **`PRODUCED`, not `EMITTED` (D2, locked).** StepRun→Message emission is a **distinct** edge type —
  `EMITTED` is already the K-013 **Message→Message** provenance edge (§9/QUERIES §10); overloading it
  would conflate "cited that seed" with "produced that message."
- **`TraceEvent` + `TRACED` (FR-4) — debug-only.** A debug run writes one `TraceEvent` per LLM
  prompt/response, tool call/result, guard judgment, and retrieval (dozens per run); a non-debug run
  writes **zero**. New node type → new DDL: `TraceEvent.traceId` index **then** UNIQUE (§7.1).
- **`stepRunId`** is the `StepRun`'s stable identity (indexed + UNIQUE, §7.1).
- **Snapshot/`Step` deletion blast radius (graph-dba, verified 2026-08-18).** A
  `WorkflowDefSnapshot`+`Step` subgraph can be `DETACH DELETE`d (e.g. to force a
  re-materialize) without corrupting a live/completed `WorkflowRun`'s executed-step
  history: `stepKey` is copied onto `StepRun` at write time (`record_step_and_advance`,
  `repository.py`) rather than read live from `Step`, and `HAS_STEP_RUN`/`LAST_STEP_RUN`/
  `NEXT`/`PRODUCED` never touch `Step`. The one edge that **is** severed by deleting the
  `Step` nodes above is `RAN` (`StepRun`→`Step`, this section) — today that's a write-only
  pointer (created at advance time, not read by any shipped query), so the practical blast
  radius is "live position (`AT_STEP`) + `OF_DEF` back-reference + the `RAN` pointer,"
  never the audit trail's own readable data. If a future query starts traversing `RAN`,
  re-check this note before relying on it across a snapshot deletion.

> `ctx`/`input`/`output`/`payload` are opaque; the executor (de)serializes app-side.

**Who writes the run `ctx` (K-024 D-F).** Every write of `WorkflowRun.ctx` that carries human or
external input rides **inside the resume CAS itself** — `resume_run_with_ctx` (QUERIES §12.13) is
`resume_run` (§12.4) plus one `SET` term, one query. There is deliberately **no** standalone
"set the ctx" write. With a split write, submitter B's ctx can land between A's ctx write and A's
CAS, so the drive that runs is A's while the data it reads is B's — a **silent wrong branch**, and
worse, a stale submitter could erase a key an earlier step already branched on, leaving a run whose
own `ctx` no longer explains its own trail. Folded, only the CAS **winner's** ctx is ever written:
"which input advanced the run" and "which input is in `ctx`" can never disagree, which is the audit
property §6.3 exists to prove. A loser gets a visible 409 with **nothing written** — neither the
status flip nor the ctx — never a silent loss.

> **Residual window, stated plainly (R-1).** The *read* before the merge is still
> non-transactional: two submitters can both read the same base ctx, merge onto it, and the
> winner's merge may omit a key the loser intended to add. That is a lost update **on an unwritten
> input, reported as a 409 to its submitter** — not a wrong branch and not an erased key a prior
> step branched on. Single-approver use today; the real fix (a `ctxVersion` counter CAS) is a
> deliberate follow-up, not built.

The engine loop: read `AT_STEP` → execute the step (LLM-native agent loop or deterministic handler) →
evaluate outgoing `TRANSITION` guards against `ctx` (first-firing wins) → `record_step_and_advance`
(create the `StepRun`, append `NEXT` via the tail, move `AT_STEP`) — **or** suspend to `waiting` if the
step declares `waitsForHuman` and no guard fired, **or** terminate (`done`/`failed`). Runaway loops are
bounded by `maxSteps` (run budget) + per-node `maxIterations` (§7). The whole walk is local to the
workspace graph (fast, isolated, fully auditable). **Verified Cypher: `QUERIES.md` §12** — this section
is the *why*, not a query copy.

> **`_drive_loop` is SHA-locked** (`71055f756280`, see `docs/archive/plans/m3-process-flow.md`
> §3.1) — do not edit its body without deliberately re-opening that lock. `_execute_step`,
> `_select_transition`, `_trace_step`, and `resume` sit **outside** the lock and may be changed
> freely. **Recompute the lock line-number-independently** — a `sed` line-range breaks the moment
> anything shifts elsewhere in the file, even with the locked body untouched — by bounding the
> extraction on the `def`/seam-comment markers instead:
> `awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12`
> (verified reproducing `71055f756280` on this commit).

> **What `maxSteps` actually means (K-031, documented — not changed).** `maxSteps` is a **runaway
> tripwire checked *after* each recorded step**, not a hard cap: a run executes at most
> **`maxSteps + 1`** steps before failing with `"step budget exceeded"`. The check runs only on the
> two driving outcomes — a guard fired (OUTCOME A, `executor.py:410`) and a legitimate self-loop
> (OUTCOME C, `:427`), both `rec["stepCount"] > max_steps`. It is **deliberately not applied on the
> park path** (OUTCOME B — a parked run cannot self-drive; see the comment at `executor.py:415-421`)
> **or on the terminal path**. Treat it as a safety bound, not an SLA or a cost budget. Making it an
> exact cap (`>` → `>=`) lands inside the SHA-locked `_drive_loop` and is filed as proposed **K-033**.

> **`status` as a property, not a label**, so a run's state changes in place without
> re-labeling churn; index it for "all running workflows" and the `waiting`-run resume lookup (§12.9).
> Suspend/resume are guarded single-query CAS flips (`running↔waiting`) so concurrent replies can't
> double-resume (QUERIES §12.3/§12.4).

### 6.3 Coordination is workflow, not a separate primitive

Agent/team coordination (task lifecycle, "room state") is modelled as an M3 `WorkflowDef` of
`kind:'process'` over `Step` + `TRANSITION` + `StepRun` — **not** a flat `Task` node or a
presence field. This avoids a parallel model that would later need migrating into the engine
(single-store philosophy). Full rationale/ADR: `docs/archive/plans/m1-chat-mcp.md` Appendix B.

**The proof now exists (K-024, 2026-07-21).** `access-request@v1` — an LLM-free `kind:'process'`
def of six steps and six transitions over `human` / `decision` / `wait` — runs end to end with **no
LLM and no network**:

| Artifact | Where |
|---|---|
| The def (the single source both seed and test read) | `server/falkorchat/proof_defs.py` — `ACCESS_REQUEST_DEF` |
| Offline acceptance test (all three §4.3 paths) | `server/tests/test_process_flow.py` |
| Seeding into `reference` + `ws:{id}` | `scripts/seed_workflows.sh` (second def) |
| Design + traced paths | `docs/archive/plans/m3-process-flow.md` |

The design claim it settles: **a business process needs no new primitive, no new run state and no
scheduler.** A `human` step is just a step whose outgoing guard reads a `ctx` key that does not exist
yet — the executor's existing "no transition fired" outcome parks it; writing the key
(`POST /workflow-runs/{id}/input`) makes the same guard fire on resume. The executor's drive loop was
**not modified** to support any of it.

> **Handoff note for K-025 (QA acceptance) — repeated from §6.1 because it is the single most
> misreadable thing here:** `wait` is **signal-driven, not timer-driven, and mechanically identical
> to `human`** — only the `awaiting.kind` string differs. There is no scheduler in this system
> (decision D-C); a parked `wait` is released by an external signal on the input endpoint, never by
> elapsed time. Timers/scheduled wakeups are backlog **K-028**. A `wait` step that never advances on
> its own is the specified behaviour, not a defect.

---

## 7. Indexes, constraints & vector search

### 7.1 Per workspace graph `ws:{id}`

> **Executable DDL is `scripts/bootstrap_schema.sh` — the single source of truth**, asserted by
> `test_queries.sh` (256/256). This section describes *what* is indexed/constrained and *why*, not
> the runnable statements, so the two can't drift (the same discipline §5.3/§8 apply to queries).
> `bootstrap_schema.sh <wsId>` is idempotent; `EMBEDDING_DIM` (default `1536`) sets the vector
> dimension per workspace.

**Critical ordering rules (live-verified):**
1. `GRAPH.CONSTRAINT CREATE` requires an existing range index on the same property — always index
   first. The script emits every index before its constraint for this reason.
2. Composite constraints (`PROPERTIES 2 …`) are supported and **live-verified** on this build —
   the script creates them and `test_queries.sh` asserts they block duplicate `key+version`.
3. Constraint creation returns `PENDING` → becomes `OPERATIONAL` asynchronously. Verify with
   `CALL db.constraints()`.

**Range indexes backing a uniqueness constraint** — one per entity anchor (`{label}Id`), plus the
composite-keyed `WorkflowDefSnapshot`:

| Label | Indexed property(ies) | Constraint |
|---|---|---|
| `User` | `userId` | UNIQUE 1 |
| `Agent` | `agentId` | UNIQUE 1 |
| `Channel` | `channelId` | UNIQUE 1 |
| `Thread` | `threadId` | UNIQUE 1 |
| `Message` | `msgId` | UNIQUE 1 |
| `Document` | `documentId` | UNIQUE 1 |
| `Chunk` | `chunkId` | UNIQUE 1 |
| `Entity` | `entityId` | UNIQUE 1 |
| `WorkflowRun` | `runId` | UNIQUE 1 |
| `StepRun` | `stepRunId` | UNIQUE 1 |
| `TraceEvent` | `traceId` | UNIQUE 1 (debug-only nodes; M3 K-022) |
| `ReadCursor` | `cursorId` | UNIQUE 1 |
| `WorkflowDefSnapshot` | `key`, `version` (two indexes) | UNIQUE 2 (composite) |
| `Step` | `key`, `stepUid` (two indexes) | UNIQUE 1 (`stepUid`); `key` index-only (§6.1) |

**Hot-filter indexes (no constraint)** — support scans/ordering, not identity:

| Label | Property | Serves |
|---|---|---|
| `Thread` | `updatedAt` | recent-threads listing |
| `Message` | `createdAt` | time-range / keyset reads (§9) |
| `WorkflowRun` | `status` | "all running workflows" |
| `StepRun` | `status` | step-state filters |

> `Message.threadId` is **deliberately unindexed** (§5.1) — nav metadata, not an anchor.

**Full-text index (RediSearch):** `Message.text`, via `db.idx.fulltext.createNodeIndex('Message',
'text')` — backs §5's keyword search.

**Vector indexes:** `Message.embedding` and `Chunk.embedding`, created via **DDL**
(`CREATE VECTOR INDEX … OPTIONS {dimension, similarityFunction:'cosine'}`).
- ⚠️ `db.idx.vector.createNodeIndex` is **not** a registered procedure on this build (live-verified)
  — the DDL form is mandatory (§1.2).
- Dimension **must** match the embedding model exactly (`EMBEDDING_DIM`; e.g. `1536` for
  `text-embedding-ada-002`) and is fixed per workspace at bootstrap.
- Vectors stored as `vecf32`; **score is cosine distance** (`0` = identical, lower = more similar).
  Write `SET n.embedding = vecf32([...])`; read `CALL db.idx.vector.queryNodes('Message','embedding',
  $k, vecf32($vec)) YIELD node, score` → `ORDER BY score ASC`. Canonical read: `QUERIES.md` §6/§8.
- Vector indexes are usually the biggest per-workspace RAM line (`dim × 4 bytes × #vectors`, §10/§11).

### 7.2 `reference` graph

Same ordering rule (index first, constraint second); executable DDL lives in
`bootstrap_schema.sh` alongside §7.1.

| Label | Indexed property(ies) | Constraint |
|---|---|---|
| `WorkflowDef` | `key`, `version` (two indexes) | UNIQUE 2 (composite) |
| `Entity` | `entityId` | UNIQUE 1 |
| `Step` | `key`, `stepUid` (two indexes) | UNIQUE 1 (`stepUid`) — the MERGE identity (§6.1); `key` index-only (display/traversal anchor) |

**Rule:** index the *anchor* of a traversal (the start node you look up), not every hop. Always
confirm the index is actually used with `GRAPH.PROFILE` — an index that isn't hit is just RAM.

### 7.3 Which vector store

Two vector engines are present on this box. **Use FalkorDB's in-graph vector index**
(`db.idx.vector.queryNodes`) so a single query fuses similarity + traversal — the whole point of
GraphRAG here. The standalone **Redis Vector Sets (`vectorset`)** module is *not* traversable;
reserve it only for an out-of-graph, high-throughput ANN index if one is ever needed.

---

## 8. Hybrid retrieval (the GraphRAG read path)

The AI participant answers a question in a channel by combining semantic recall with structured
traversal — one read-only query, routable to a replica.

> **Canonical Cypher: `docs/QUERIES.md` §6.** This section describes the read path's *shape*
> only, so the two never drift.

- **Vector** finds *what's semantically relevant*; **traversal** pulls *precise, explainable
  neighbors* (same thread, shared entities, prior workflow steps). Either alone is weaker.
- **Score is cosine distance** (live-verified: identical vectors → score `0`). Order `ASC` to
  rank most-similar first.
- Served via `GRAPH.RO_QUERY` → can hit read replicas (mind replica lag for just-posted
  messages; route "include my last message" reads to the primary).

---

## 9. Write paths

| Operation | Pattern | Notes |
|---|---|---|
| Post message | Guarded `CREATE` inside `FOREACH`+`CASE` per path + relink `Thread TAIL → NEXT` (QUERIES.md §4 v2) | Two separate self-guarding variants (first vs subsequent — see §5.3), never a conditional MERGE of the two paths. Always returns a **status row**; **retry-idempotent via the `dupMsg` status** (the old "idempotent via unique constraint" claim was falsified — a replayed MERGE re-ran the relink clauses and corrupted the chain, K-007 evidence). The `Message.msgId` uniqueness constraint stays as the concurrency backstop (rollback verified all-or-nothing). O(1) append. |
| Create channel / thread | plain `CREATE` (server-minted uuid ids) | **Non-idempotent** — a retried create mints a new id; the uniqueness constraints backstop. A MERGE on a fresh uuid could never match (K-007 fold-in). |
| Backfill / import | `UNWIND $rows AS row …` in chunks, or `falkordb-py` bulk loader | Never one giant CREATE — bound transaction memory; size batches (writes ignore TIMEOUT — §10) |
| Embed messages | async worker: compute embedding → `SET m.embedding = vecf32($v)` | Decouple embedding latency from the post path |
| Advance workflow | create `StepRun`, append `NEXT`, move `AT_STEP` | All local to `ws:{id}`; fully transactional within the graph |
| Publish workflow def | write to `reference`; materialize snapshot into consuming `ws:{id}` graphs | Topology-immutable per version (rejected `409` on a differing re-publish, K-034); properties stay create-only. Bump version to change either. |

**Rule:** every `MERGE` is backed by a uniqueness constraint, or it's a duplicate-node bug
waiting for concurrency. (The §4 v2 message writes contain no MERGE at all — guarded CREATE
with the constraint as backstop.)

---

## 10. Architecture & operations

```
            ┌────────────┐      RESP / Bolt      ┌──────────────────────────┐
   clients →│  App / API │ ───────────────────── │  FalkorDB (Redis 8.x)     │
            │  (gateway) │   GRAPH.QUERY (RW) →   │  PRIMARY  ┌─ ws:acme      │
            └────────────┘   GRAPH.RO_QUERY (RO)  │           ├─ ws:globex    │
                  │                               │           ├─ reference    │
                  │ embeddings / LLM              │           └─ identity     │
            ┌─────▼──────┐                        │  REPLICAS (RO_QUERY, RAG) │
            │ LLM + embed│                        └──────────────────────────┘
            │  workers   │     scale out: Redis Cluster — workspace graphs
            └────────────┘     distributed across shards by key hash slot
```

- **Client SDK:** `falkordb-py` **pinned 1.6.x** —
  `db = FalkorDB(host, port)` → `g = db.select_graph(f"ws:{wid}")` →
  `g.query(cypher, params={...})` / `g.ro_query(...)`. **Always parameterize** (`params=`),
  never string-concatenate user input into Cypher.
- **Memory sizing first.** Estimate per workspace: nodes + relationships + properties +
  per-relationship-type matrices + full-text + **vector indexes (often the biggest line:
  `dim × 4 bytes × #vectors`)**. A workspace graph must fit one shard's RAM with headroom.
  Watch `GRAPH.MEMORY USAGE`, `GRAPH.INFO`, Redis `INFO memory`. Set `maxmemory` deliberately
  and **do not evict** the graph's own keys.
- **Persistence:** RDB snapshots + AOF; choose AOF fsync policy per RPO. Restart replays into RAM.
- **HA / scale reads:** primary takes writes; read replicas serve `GRAPH.RO_QUERY` (RAG reads).
  Async replication → eventual consistency; watch replica lag. Sentinel for failover.
- **Scale tenants:** Redis Cluster spreads `ws:{id}` graphs across shards by hash slot. Each
  graph stays whole on one shard. Rebalance by moving slots; isolate hot workspaces onto
  dedicated shards if needed.
- **Tuning:** `GRAPH.CONFIG` — `THREAD_COUNT` (size to cores), `QUERY_MEM_CAPACITY`,
  `MAX_QUEUED_QUERIES`, `TIMEOUT_DEFAULT`/`TIMEOUT_MAX`, `CACHE_SIZE`.
- **TIMEOUT posture (K-007, live-probed).** Keep the legacy single-knob `TIMEOUT=1000` as the
  deployment default — right for chat CRUD and verified to fire (enforcement is
  batch-granular; slightly-over reads can slip through). Future GraphRAG/§6/§8 hybrid reads
  and long thread walks pass a **per-query client override**
  (`g.ro_query(q, params=…, timeout=…)`, e.g. 5000–10000 ms; pass-through verified, uncapped
  while `TIMEOUT_MAX=0`) — expose it as a service-layer constant, not per-call ad-hockery.
  **Writes ignore TIMEOUT entirely on this build** — a write runs to completion regardless of
  clause or default; bounded batches (≤ a few hundred `UNWIND` rows) and the existing API
  input caps are the only write-path protection. If ops later wants a hard ceiling on client
  overrides, switch to `TIMEOUT_DEFAULT`/`TIMEOUT_MAX` (>0) — mutually exclusive with the
  legacy `TIMEOUT` knob; change deliberately, in one step. Caveat noted once (not
  reproduced): an instant-timeout anomaly right after a long override run — edge-build timer
  bookkeeping noise; re-check on upgrades (upstream filing recommended, OQ6).
- **Observability:** `GRAPH.SLOWLOG` for slow queries, `GRAPH.PROFILE` for plans, Redis metrics.
- **Security:** Redis ACLs scoping `GRAPH.*` per principal (ideally per workspace key pattern),
  TLS in transit, network isolation; secrets outside the data.

---

## 11. Capacity — empirical line at 1024 dims (K-007)

Measured live (`falkordb/falkordb:edge`, 4096 realistic messages bulk-loaded into a
1024-dim-indexed scratch workspace: `msgId/text/role/createdAt/threadId` + inline `vecf32`
embedding + `POSTED_BY` + full `NEXT` chain + HEAD/TAIL; `INFO memory` delta):

| Component | Bytes/message |
|---|---|
| raw `vecf32` embedding (1024 × 4 B) | 4,096 |
| node + attrs (text ~50 chars, ids, role, `createdAt`, `threadId`) + edges (`NEXT`, `POSTED_BY`) | ~1,900 |
| HNSW vector index + range-index entries + allocator overhead | ~6,400 |
| **Total observed** | **12,387 ≈ 12.4 KB** |

- Rule of thumb: **~12.5 KB/message at 1024 dims ≈ 1.25 GB per 100k-message workspace**
  (vs ~17–18 KB extrapolated at 1536 — the dim cut saves roughly a third). The bootstrap
  default stays 1536 (chosen per workspace); set `EMBEDDING_DIM=1024` for the decided model
  (§1.3) **before** workspace creation — vector index dimension is fixed at creation.
- `threadId` cost: one short string, ~50–60 B/message, no index — noise (<0.5%) against the
  12.4 KB line. `ReadCursor.lastReadMsgId`: one string per (member, thread) cursor — negligible.
- Ingestion datapoint: ~1,178 msg/s with 256-row `UNWIND` batches incl. embeddings, single
  client — bulk batches of 100–500 rows sit comfortably inside the write-path safety envelope
  (writes are unkillable by TIMEOUT, §10 — keep batches bounded).
- **Measurement caveat (K-007, upstream filing recommended):** `GRAPH.MEMORY USAGE` reported
  `indices_sz_mb: 0` while the HNSW index demonstrably held 4096 vectors — on this edge build
  it **under-reports vector-index memory**. Size workspaces from **`INFO memory` deltas**, not
  `GRAPH.MEMORY USAGE`, until fixed upstream.

> Action: re-measure on a pilot workspace with the real embedding model, and back into a
> per-workspace RAM budget + a shard:workspace packing ratio before scaling out. *(Chat-core
> floor + the budget/packing table are now measured — §11.1–§11.2, K-011; still re-measure with
> the real embedding model at M2 before scaling out.)*

### 11.1 M1 append-path load test + hot-read PROFILE closeout (K-011)

Measured live on `falkordb/falkordb:edge` through the **M1 REST service path** — 16 concurrent
posters, 3,000 messages, one channel / 16 threads, each `POST /threads/{id}/messages` a full
`services.post_message` round trip (actor + mention validation, role derivation, §4 v2 guarded
write). This is the **live request path**, not the K-007 bulk-`UNWIND` ingestion datapoint
(§11, ~1,178 msg/s single-client batched). Harness: `scripts/load_test.sh` →
`scripts/load_append.py`.

| Metric | Value |
|---|---|
| Sustained append throughput | **~614 msg/s** (16 clients, single graph) |
| Append latency p50 / p90 / p99 | **24.4 / 30.6 / 40.7 ms** (max 146 ms) |
| Errors | 0 / 3,000 |

Throughput is **graph-write-bound** (FalkorDB serialises writes per graph key), so the
per-thread fan-out only removes first-post/TAIL race dispatch from the latency sample — a single
busy channel lands the same ceiling. Each post is ~4 round trips (`thread_exists` +
`resolve_member_kinds` + `thread_has_head` + the write), so this is a conservative service-layer
figure, not raw Cypher throughput.

**Hot-read plans — all four hit an index-backed anchor; none degraded to a `NodeByLabelScan`**
(`GRAPH.PROFILE` on the loaded `ws:load` graph, raw plans archived by the harness). Re-profile on
engine upgrades:

| Hot read | Anchor op | Verdict |
|---|---|---|
| §4 thread read | `Node By Index Scan \| (t:Thread)` → HEAD/NEXT walk | index ✓ |
| §9.1 since-read (thread) | `Node By Index Scan \| (t:Thread)`; keyset predicate folds into a `Filter` on the walk | index ✓ |
| §9.2 since-read (ws-wide) | `Node By Index Scan \| (m:Message)` on `createdAt`; composite `OR` folds into the scan, **no residual Filter** | index ✓ |
| §5 full-text search | `ProcedureCall` (`db.idx.fulltext.queryNodes`, RediSearch full-text index) | index ✓ |

Confirms the AGENTS.md standing note (Formulation-A composite keyset still plans as a bare
`Node By Index Scan` with no residual Filter on this build) and the §9.2 plan claim — no
graph-dba escalation.

### 11.2 Per-workspace RAM budget & shard packing (K-011, `INFO memory` deltas)

**Chat-core floor (M1, no embeddings) — measured `INFO memory` `used_memory` delta:** 3,000
messages added **3,173,056 B → ~1.06 KB/message** (node + `text`/ids/`role`/`createdAt`/`threadId`
attrs + `NEXT`/`POSTED_BY` edges + `createdAt` range index + `msgId` constraint index + full-text
index entry) → **~101 MB per 100k-message workspace**. That sits *below* the ~1.9 KB K-007
node-line estimate, confirming that at 1024 dims the embedding (4 KB) + HNSW/range overhead
(~6.4 KB) dominate — **~85% of the 12.4 KB/message total is vector, not chat.**

**Per-workspace RAM budget line (per 100k messages):**

| Profile | Per message | Per 100k-msg workspace |
|---|---|---|
| M1 chat-core (no embeddings) — *measured (K-011)* | ~1.06 KB | **~101 MB** |
| M2 with 1024-dim embeddings (§11 K-007) | ~12.4 KB | **~1.25 GB** |
| M2 with 1536-dim embeddings (§11 K-007) | ~17–18 KB | **~1.7 GB** |

**Shard:workspace packing ratio** = (shard `maxmemory`) ÷ (per-workspace RAM × 1.3 headroom for
writes / RDB fork / index build; no eviction of graph keys, §10). Worked example on a 32 GB shard
with `maxmemory` ≈ 22 GB:

| Workspace profile (100k msgs) | Fits per 22 GB shard |
|---|---|
| chat-core only (~101 MB) | **~170 workspaces** |
| 1024-dim embedded (~1.25 GB) | **~13 workspaces** |
| 1536-dim embedded (~1.7 GB) | **~10 workspaces** |

Size real deployments from the **embedded** row (M2 is the target); the chat-core floor is the
M1 reality and the lower bound. `GRAPH.MEMORY USAGE` still reported all-zero
`indices_sz_mb`/`total_graph_sz_mb` for the loaded `ws:load` graph (the K-007 caveat holds even
with **no** vectors present) — budget from `INFO memory` deltas, never `GRAPH.MEMORY USAGE`.

---

## 12. Roadmap

1. **M0 — Stand up the engine.** ✅ FalkorDB running (`falkordb/falkordb:edge`, Redis 8.2.2, module `999999`) via Docker. Live probes confirmed: cross-graph edge behavior, vector DDL syntax, index-before-constraint ordering, `algo.*` procedure set, `vecf32` storage and `db.idx.vector.queryNodes` query surface.
2. **M1 — Chat core.** ✅ Users/Channels/Threads/Messages, thread-scoped `NEXT` + `HEAD`/`TAIL` append path, full-text index, basic read windows. **Application layer:** FastAPI REST server over a service/repository split, single hardcoded tenant, minimal web UI — full design in §14. **Plus an MCP (Streamable-HTTP) agent front door on the same service layer — §15 (K-002).** Full stack (repository → services → MCP + REST + full-text `search`, plus the static `web/` UI, all mounted in `app.py`) is built and green (110 tests). The append-path load-test + hot-read `GRAPH.PROFILE` DoD is now **closed — see §11.1/§11.2** (~614 msg/s, all four hot reads index-backed, per-workspace RAM budget). The web request/response path was also de-staled — incremental `?since=` polling, inline non-blocking errors, clickable search results (K-012). M1 chat core is complete.
3. **M2 — GraphRAG.** ✅ Embedding workers, in-graph vector index @1024, hybrid retrieval query (§8), AI `Agent` participant posting answers with `EMITTED` provenance — **QA-accepted (K-015), M2 done.** Delivered: every posted message embedded out-of-band via an async `EmbeddingWorker` → LM Studio `/v1/embeddings` (Qwen3-Embedding-0.6B, 1024-dim); `repository.hybrid_search` (§6, cosine-distance ASC, dormant Entity no-op); `AgentResponder` — an `@mention` of the configured agent triggers retrieval-grounded LLM answering (Qwen3-4B-Instruct via LM Studio) posted as the `Agent` (`role:"assistant"`, derived) with an `EMITTED` provenance edge (`QUERIES.md` §10, score+rank), loop-guarded and failure-isolated; K-014 web renders assistant replies + reader `isMention`. Served tenant `ws:acme` runs at `EMBEDDING_DIM=1024` (`start_server.sh` gates the live loop on `FALKORCHAT_ENABLE_AGENT`). Baselines: pytest 156 / query suite 149/149. **Groundwork (K-007) had landed earlier:** agent authorship (role derived from the author label), self-guarding v2 write paths (status-row contract, retry-idempotent via `dupMsg`, first-post race refused), `Message.threadId` denorm + backfill script, composite `(createdAt, msgId)` keyset cursors (tie-safe reads), TIMEOUT posture (§10), empirical 1024-dim RAM line (§11). **Deferred to M2.5** (not on the M2-green path): real auth/tenancy (K-016), transport-level externally-authenticated agent actor (K-017, the K-007 QA carry-over), real-time push (K-018); and a channel-scoped retrieval read (responder currently workspace-wide).
4. **M3 — Workflow engine.** Definition model in `reference`, snapshot materialization, run/step-run executor, chat linkage; both a conversational flow and a business-process flow as proof.
5. **M4 — Scale & ops.** Redis Cluster, replicas for RO reads, Sentinel, ACL/TLS, backup/restore drill, per-workspace memory budgeting + shard packing.

---

## 13. Open questions

- ~~**Workflow guard expression language** — reuse an existing expr lib or define a minimal DSL stored in `Step.config`?~~ **RESOLVED (M3 executor, K-022): LLM-native + coexist.** A `TRANSITION.guard` is either the empty-string unconditional/default form or a `{kind:'llm', text}` discriminator judged in natural language against run context (§6.1/§6.2); deterministic transitions use the empty form. **No expression library is built** — a would-be `expr` kind is a `NotImplementedError` seam (zero dead code). Rule 8 respected: the `{kind}` discriminator is parsed app-side, never filtered in Cypher.
  **Amended (K-024, decision D-A): the deterministic half is `cmp`, not `expr`.** The LLM-free
  `kind:'process'` proof flow needed a guard that could branch on run state without a model, so a
  **closed structured comparator** was added: `{kind:'cmp', path, op, value}` + `all`/`any`/`not`,
  with whitelisted ops, two whitelisted path roots (`ctx.`/`output.`), depth/width/node caps, total
  at drive and strict at publish (§6.1). It is deliberately **named `cmp` and not `expr`** so this
  resolution stays literally true: there is still no parser, no `eval`, no expression library and no
  new dependency, and `kind:'expr'` still raises `NotImplementedError`.
- **Retention** — do old messages/embeddings age out (and how does that interact with the always-in-RAM constraint)? (→ decide on K-011 load-test data; evicting cold embeddings is the cheapest lever — ~10 KB of the 12.5 KB/msg is vector + index.)
- **Cross-workspace analytics** — app-layer fan-out vs. a dedicated `analytics` rollup graph. (Cost accepted §4; mechanism open, no milestone yet.)
- **Real-time gateway transport** — for the M2.5 push path, Bolt (port `65535`, confirmed in `GRAPH.CONFIG`) vs. RESP/WebSocket. The M1 app *driver* is settled (RESP via `falkordb-py`); this is only the push-gateway choice. (→ K-018.)
- **Pre-production config review:** live config defaults noted — `THREAD_COUNT 4`, `OMP_THREAD_COUNT 4`, `CACHE_SIZE 25`, `MAX_QUEUED_QUERIES 25`, `QUERY_MEM_CAPACITY 0` (unlimited), `ASYNC_DELETE 1`. Review before production (TIMEOUT 1000ms already reviewed & kept — K-007, §10).

---

## 14. M1 application architecture (client/server)

§10 sketches the *operational* topology (app ⇄ FalkorDB). This section pins the *application*
code architecture for **M1 — Chat core**: what the client and server are, the transport between
them, and the internal layering.

### 14.1 Scope decisions locked for M1

| Axis | Decision | Rationale |
|---|---|---|
| **Transport** | **REST/JSON over FastAPI** | The only M1 client is a browser, which speaks HTTP natively — no gRPC-Web bridge tax. Free OpenAPI console to exercise the API. M2.5 real-time adds native WebSocket/SSE on the same server. |
| **Client** | **Minimal web UI** (channels list + thread view) | Smallest end-to-end path that exercises the full stack visually. |
| **Real-time** | **Deferred to M2.5** | M1 is request/response; the UI re-fetches a thread window after posting. The push path (Redis Pub/Sub → WebSocket) slots onto the same service layer in M2.5 with no schema change. |
| **Auth / tenancy** | **Single hardcoded tenant** — `ws=acme`, `user=u1` | Keeps M1 focused on the chat data path. Injected at one seam (see §14.3) so real auth replaces it without touching services/repo. |

> Transport was deliberately re-evaluated away from gRPC: gRPC's wins (polyglot typed contracts,
> native streaming, service-to-service perf) are all unused when the sole client is a browser, and
> gRPC-Web can't do client/bidi streaming in browsers anyway — WebSocket/SSE is the stronger M2.5
> real-time path. REST keeps the layers below the router transport-agnostic, so a gRPC servicer or
> a service-to-service hop can still be bolted onto the same `Service` later if a non-browser
> consumer ever appears.

### 14.2 Layering

```
┌─ Browser (minimal web UI) ─┐                ┌─ Python server (FastAPI, one process) ───────┐
│ channels | thread view     │   REST/JSON    │ api.py      router (thin: HTTP ⇄ Service)    │
│ post / read / search       │ ─────────────▶ │   ▲  CallContext dep = {ws:acme, actor:u1}  │
└────────────────────────────┘                │ services.py  domain logic, append dispatch   │
                                              │ repository.py  Cypher ⇄ QUERIES.md (RO|RW)   │
                                              │ db.py        falkordb-py conn, select_graph  │
                                              │ modelconfig.py  the model-resolution seam    │
                                              │   (K-042, §14.8) — every LLM/embedding call  │
                                              │ transport.py    ← one HTTP transport, §14.8  │
                                              └────────────────────────────────────────────┬─┘
                                                                                            ▼  FalkorDB / LLM providers
```

- **`repository.py` is the only place Cypher lives.** Each method maps 1:1 to a verified query in
  `QUERIES.md`, always parameterised (`params=`), `ro_query` for reads / `query` for writes,
  `select_graph(f"ws:{id}")` for scoping.
- **`services.py` owns the invariants** the write-path rules describe: choosing the first-vs-subsequent
  append variant, id generation, `Thread.updatedAt` bumps, setting `role`/`POSTED_BY`.
- **`api.py` is the only layer that changes** if the transport is ever revisited.
- **`modelconfig.py`/`transport.py` (K-042, §14.8) are the model-resolution seam.** Every LLM/
  embedding consumer (`responder.py`, `executor.py`, `guards.py`, `embedding.py`, `tools.py`) holds
  a `ModelGateway` and resolves per call — never constructs `llm.py`'s/`embedding.py`'s
  OpenAI-compatible clients directly (FR-4).

### 14.3 The auth/tenancy seam

The hardcoded scope lives in **one FastAPI dependency**, not scattered through the code:

```python
# config.py
WS_ID = "acme"
USER_ID = "u1"

# api.py
def get_context() -> CallContext:        # the seam
    return CallContext(ws=WS_ID, actor=USER_ID)
```

Services and the repository already take `ws` / `actor` as parameters, so when auth lands
(token → user + workspace claim, or the `identity` graph as source of truth) **only `get_context`
changes** — everything below is untouched.

### 14.4 REST surface → service → verified query

| Endpoint | Service method | `QUERIES.md` |
|---|---|---|
| `GET /health` | `ping` | liveness probe (trivial `RO_QUERY RETURN 1`; 503 when FalkorDB is down) |
| `POST /channels` | `create_channel` | §3 create a channel |
| `GET /channels[?limit=]` | `list_channels` | §3 list channels in a workspace |
| `POST /channels/{cid}/threads` | `create_thread` | §3 create a thread |
| `GET /channels/{cid}/threads[?limit=]` | `list_threads` | §3 list recent threads in a channel |
| `POST /threads/{tid}/messages` | `post_message` | §4 first message / subsequent message |
| `GET /threads/{tid}/messages[?since=&limit=]` | `read_thread` / `read_messages` | §4 full thread; with `since`/`limit` → §9.1 window as a pure read (`since` defaults to 0 — the browser never touches cursors) |
| `GET /messages/{mid}` | `get_message` | §4 get a single message |
| `GET /search?q=` | `search_messages` | §5 full-text keyword search |

**Workflow-run drive surface (M3 / K-024 U3)** — the non-chat front door for a `kind:'process'`
run. (The §11 def-authoring routes `POST/GET /workflow-defs…`, the §11 **def/snapshot structure
reads** `GET /workflow-defs/{key}/versions/{version}` and
`GET /workspaces/{ws}/snapshots/{key}/versions/{version}` plus their `…/diff` sibling —
QUERIES.md §11.2 / §11.5 — and the §12 inspection routes
`GET /workflow-runs/{id}[/step-runs|/trace]` are also mounted; they are read/publish paths and are
described at their own sections.)

**The structure/diff reads (K-031), four operator-facing facts.** They answer *"is what I think is
published actually published"*, *"is the workspace running the same thing"*, and *"have `reference`
and `ws:{id}` gone stale independently"* without dropping to raw Cypher. (1) Both structure reads
are **whole-object and unpaginated** — there is no `?limit=`, deliberately: a truncated subgraph is
a *wrong* answer, not a partial one (an operator who gets 50 of 60 steps concludes ten are
missing). They are bounded upstream by the publish-time caps (`MAX_STEPS`, `MAX_TRANSITIONS`,
`MAX_CONFIG_LEN`), matching the unpaginated §12 run reads; service-layer publishers bypass Pydantic,
so those caps are not universal — an accepted, documented residual. (2) The **diff** is bounded
instead by preview truncation (`MAX_DIFF_PREVIEW = 200`): its response is O(differences), never
O(def). (3) **The snapshot is what the executor drives** (`executor._drive_loop` → `get_snapshot`),
so `snapshot` is the operational truth and `def` (`reference`) is the intended truth. (4) The diff
is **version-qualified** — it answers "same version, different content", never "wrong version"; to
detect a stale *version*, compare `GET /workflow-defs` against `GET /workspaces/{ws}/snapshots`
first, or run `./scripts/verify_workflows.sh <wsId>`, which checks both seeded defs at their
expected versions in one command. There is no `latest` alias on the structure route: an operator
investigating a version mismatch must name the version.

> The publish/materialize receipt counts what was **submitted**; the structure read counts what is
> **stored**. **A divergence between the two is a signal, not an endpoint bug** — see K-034.

`config`/`guard` come back **verbatim** as opaque strings (rule 8) — never parsed, re-serialized or
pretty-printed, so a whitespace-only divergence is still visible. The two structure bodies are the
same shape apart from `source` (so `jq`-diffing them by hand works) — **but that parity is the 200
body only**: the def route 404s through `WorkflowDefNotFoundError` (`{"error": …}`) while the
snapshot route raises a plain `HTTPException` (`{"detail": …}`), each mirroring its sibling
non-structure route's established style. A client must not assume one error shape. These three routes are the only
ones in the surface that declare a `response_model`; the rest are deliberately not retrofitted
(FastAPI's `response_model` *filters* undeclared fields), leaving a mixed convention recorded on the
standing response-schema backlog entry.

| Endpoint | Service method | `QUERIES.md` |
|---|---|---|
| `POST /workflow-runs` | `start_workflow_run` (`trigger_msg_id=None`) | §12.12 start a run from a snapshot with **no** chat trigger `Message` — → **201** |
| `POST /workflow-runs/{runId}/input` | `submit_workflow_input` | §12.7 + §11.5 (validate) then §12.13 merge-into-ctx **and** resume in one CAS — → **200** |

Both routes drive the run **synchronously**, not on `BackgroundTasks`: a process drive is pure
graph work with no LLM, so it is fast and — the deciding property — deterministically testable.
An LLM-bearing process def would want the background path; noted, not built.

**Bounds, and which layer owns which (K-024 m-5).** Pydantic bounds only what it can see: the
*submitted* dict (≤32 keys, key ≤ 200 chars, serialized ≤ 8000) and `maxSteps` (1…50). The
**merged** ctx bound, the reserved-key rule and the parked-step declaration check live in
`services.py`, because MCP tools and direct service callers never reach a pydantic model.

**Reserved run-ctx keys — `threadId` and `error` — are rejected on both routes, in the service.**
`threadId` is the resume denorm anchor: a caller-set one would park a process run against a live
chat thread, and the trigger's step 2 would then advance it on the next ordinary human message
there — no input, no guard data (K-024 M-2/F-6). A process run parks with `waitingThreadId = ''`,
and the thread lookup short-circuits on an empty thread id from the other end (F-5).

**Error map (D-G).**

| Condition | Code |
|---|---|
| unknown run id | **404** `WorkflowRunNotFoundError` |
| run is not parked, or lost the resume CAS (nothing written) | **409** `WorkflowRunNotWaitingError` |
| reserved key / undeclared key / value outside `config.expects` / oversized merged ctx | **400** `WorkflowInputRejectedError` |
| structurally malformed `cmp` guard (dominant source: publish-time validation) | **400** `WorkflowConfigError` |
| workflow engine not wired into this deployment | **503** `WorkflowEngineDisabledError` |
| **a fault *during* the drive** (unimplemented step type, malformed guard reaching evaluation) | **201/200** carrying `{"status":"failed","error":…}` |

That last row is the deliberate one: the executor's fault net has already stamped the run
`failed`, so the run *is* terminal and correct in the graph — a 500 traceback would misreport a
correctly-recorded terminal run as a server bug.

**The failed envelope reports graph truth, whole.** Its `status` **and** its `ctx` both come from
the *same* post-fault `get_run` re-read — never a re-read status beside the caller's submitted
input. So the `ctx` a caller gets back on the fault path is the engine's own state including the
diagnostic note `fail_run` stamped, not the merge that was attempted; on the clean path the two
are the same value by construction (the CAS wrote exactly that merge). Reporting one field from
the graph and the other from what the caller hoped happened would half-apply the rule, and the two
could disagree in exactly the situation where a reader most needs them consistent. If the re-read
status is anything other than `failed`/`done`/`waiting` the service **re-raises**, because
reporting a still-`running` zombie as success would be the worst outcome available.
**Step-budget exhaustion is not a fault**: it returns `"failed"` through the normal path and
reaches the same envelope without raising.

Request bodies are size-bounded at the Pydantic boundary (`schemas.py`: text ≤ 8000 chars,
name/title ≤ 200, mentions ≤ 50) — message text lands in graph RAM *and* the full-text index,
so the transport caps it (RAM rule 6). List `limit`s are `Query`-bounded (1–200; thread window
1–1000).

The **two append variants** (§5.3) stay hidden inside `post_message`: the service checks whether
the thread already has a `HEAD`/`TAIL` and dispatches the correct single-`GRAPH.QUERY` write. The
API only ever sees "post a message."

### 14.5 Layout (as built, M1)

```
falkor-chat/
├── server/
│   ├── falkorchat/{config,db,repository,services,schemas,api,mcp,app}.py
│   ├── tests/{test_repository,test_services,test_services_live,test_mcp,test_api,test_app}.py
│   ├── pyproject.toml          # fastapi, uvicorn, falkordb, mcp, pytest, httpx
│   └── .venv/                  # python3 -m venv (no uv on the box)
└── web/{index.html, app.js}    # fetch() against REST; channels | threads | messages + search
```

`mcp.py` is the second front door — see §15. `app.py` mounts both on one process, and also
serves `web/` as static files at `/` (mounted **last**, since `/` is a catch-all that must sit
behind the REST routes and the `/mcp` mount). Serving the UI from the same process means there is
no CORS seam. The mount is skipped gracefully if the `web/` directory is absent.

### 14.6 TDD build order

Bottom-up, red → green per unit, reusing the isolated-`ws:test`-graph approach `test_queries.sh`
already uses:

0. **Prerequisite (graph-dba):** ✅ done — the `list_channels` query gap (K-001) landed in
   `QUERIES.md` §3 + `test_queries.sh` (baseline 64/64 → 67/67). The `list_channels` repository
   method can now be built.
1. **`repository`** — integration tests against an isolated `ws:test` graph, one method at a time.
2. **`services`** — append-variant dispatch, id-gen, `updatedAt` bumps (fake repo + a few live checks).
3. **`api`** — FastAPI `TestClient` request/response contract tests. ✅ done — incl.
   `GET /search?q=` (full-text, `search_messages` → `QUERIES.md` §5).
4. **`web`** — ✅ done — minimal `web/{index.html,app.js}` (channels · threads · messages · search),
   served as static files by `app.py`; the mount seam is unit-tested, the UI itself verified
   manually against a running server.

> When this code lands, update `AGENTS.md` (key scripts/commands, working-context rules) and the
> README repo-layout/roadmap in the same change, per the repo's documentation rule.

### 14.7 Testing hazards specific to `server/`

Four gotchas that a green `pytest` run does not surface, distinct from the `test_queries.sh`
teardown hazard already documented at the `AGENTS.md` "Key scripts" table:

- **`pytest -q` is destructive to the global `reference` graph too — a different mechanism than
  `test_queries.sh`'s teardown wipe.** The `wf_repo` fixture (`tests/conftest.py`) runs
  `MATCH (n) DETACH DELETE n` on `reference` at fixture **setup**, once per workflow test, to
  isolate it from earlier tests. Because the wipe never runs at teardown, a finished pytest
  session leaves the *last* workflow test's own published defs sitting in `reference` — so
  `already present — no-op` after a pytest run may be reporting a **test's** publish, not a real
  seed, while each `ws:<id>` snapshot still holds whatever it held before. Re-run
  `scripts/seed_workflows.sh <wsId>` after any pytest run, exactly as after `test_queries.sh`.
- **A green exit code is not evidence the graph-backed half of the suite ran.** With FalkorDB
  unreachable, `conftest._falkordb_reachable()` turns the whole integration suite into
  `pytest.skip` rather than failures, so the run still exits 0 with roughly half the tests
  silently skipped. Always read `N passed, M skipped`, never just the absence of failures.
- **`ruff check .` is clean but is not a wired gate.** `pyproject.toml` configures ruff and ships
  it as a dev dependency, but no script or hook runs it — a clean manual run is evidence of that
  one run only. The real gates here are `pytest` and (coordinator-run) `scripts/test_queries.sh`.
- **`ws:test`'s vector indexes are dim 4** (`conftest.TEST_EMBEDDING_DIM`), fixed at bootstrap and
  unrelated to the served workspaces' real dimension (1024/1536). Never point a real-embedder test
  at it: a wrong-dimension `vecf32` write is silently accepted (§2/§7.1) and then drops out of ANN
  — the write "succeeds" and retrieval finds nothing, with no error anywhere in the chain.

**Verifying a claimed test count safely:** `pytest --collect-only -q` reports the suite's test
count with no FalkorDB connection and no writes — the correct way to check a plan's or review's
"N tests" baseline claim without triggering either the `wf_repo` setup-time wipe or
`test_queries.sh`'s teardown wipe above.

- **A wired agent now requires two config files (K-042).** `FALKORCHAT_ENABLE_AGENT=1` or
  `FALKORCHAT_WORKFLOW_ENABLED=1` builds a `ModelGateway.from_env()`, which reads
  `FALKORCHAT_OPENCODE_CONFIG` (no product default) and `FALKORCHAT_MODEL_CONFIG` (defaults to
  `config/models.json`). `tests/conftest.py`'s `_model_config_env` autouse fixture points both at
  `tests/data/` fixtures for every test — the suite must pass on a machine with **no**
  `~/.config/opencode/opencode.json` (verified: `HOME=<empty dir> pytest -q` is green). A test that
  needs a different value must override both the env var **and** the `falkorchat.config` module
  attribute (`monkeypatch.setattr`) — `config.py` resolves its env vars once at *import* time
  (FR-15, no reload path), so a bare `monkeypatch.setenv` alone never reaches
  `ModelGateway.from_env()` once the module is already imported.

**QA/acceptance-testing gotchas, black-box-observed (distinct from the pytest hazards above):**

- **A `verify_workflows.sh` FAIL for `reference` (def MISSING) does not, by itself, block a live
  `@mention`-triggered workflow run.** `start_workflow_run`'s trigger/execute path never reads
  `db.reference_graph` — only the observability/diff endpoints (`get_workflow_def_structure`,
  `diff_def_snapshot`) do. Three independent `@mention` triggers all started and completed
  `triage@v1` runs against `ws:acme` while `reference` was MISSING throughout. Check which code
  path actually reads `reference` before treating a `verify_workflows.sh FAIL` as an environment
  blocker for a *behavioral* test.
- **A `WorkflowRun` parked `waiting` (`waitsForHuman`) resumes on the *next message posted to its
  thread*, whether or not that message `@mention`s the assistant.** A plain, non-mention message
  into a thread with an open `waiting` run silently resumes it. Only a fresh thread with **no**
  open run correctly exercises "an ordinary message never triggers a workflow" — reusing a thread
  from an earlier test item in the same pass will confound this check.
- **`POST /workflow-runs/{id}/input`'s own response does not carry the `error` reason when that
  submission is what causes the run to fail** — only a follow-up `GET /workflow-runs/{id}` does
  (the reason lands in that run's `ctx`, not in the triggering call's response body). A caller that
  inspects only the `/input` response on a fault sees `status:"failed"` with no explanation.
- **MCP `send_message` never schedules the responder/workflow trigger — only the REST
  `POST /threads/{id}/messages` route does.** `api.py`'s REST handler is the only place
  `background.add_task(_safe_run_workflow/_safe_respond, ...)` is scheduled (via FastAPI's
  `BackgroundTasks`); `mcp.py`'s `send_message` tool has no such scheduling. A message posted via a
  real MCP client produces zero reply and zero `WorkflowRun`, confirmed live. Any black-box check of
  "does `@mention` produce a reply" must specify REST vs. MCP — they are not equivalent front doors.
- **`ModelGateway`/`modelconfig.py` requires an explicit `options.baseURL` for every provider —
  there is no implicit per-npm-package default** (unlike OpenCode's own `@ai-sdk/openai`, which
  has one). An example/fixture `opencode.json` that omits `baseURL` on an `openai`-kind entry
  parses fine but fails to *resolve* (`ModelConfigError: ... no options.baseURL ...`). Any example
  or fixture file authored for this seam should be re-**resolved** once via `ModelGateway.resolve`,
  not just parsed, before being called documented or shipped.

### 14.8 The model-resolution seam (K-042 Landing 1 + Landing 2)

**The FR-4 rule, in one sentence:** every LLM/embedding consumer holds a `ModelGateway` and asks
it for a client; a directly-injected client (the pre-K-042 `llm=`/`embedder=` constructor kwargs
every consumer still accepts) is sugar `__init__` wraps into a `StaticModelGateway` — dependency
injection for tests, never a configuration route. There is exactly one internal path from "a
kind + an optional requested ref" to a working client, and zero consumers read an endpoint or
model id from `config.py` or any file directly. Enforced, not aspirational: an AST check in
`test_modelconfig.py` fails the suite if any module outside `modelconfig.py`/`tests/` constructs
`llm.OpenAICompatibleLLM`/`embedding.OpenAICompatibleEmbedder` directly.

```
                     +-----------------------------------------------+
  opencode.json -->  | modelconfig.py                                |
  (pristine,         |   ProviderCatalog  <- parse + {env:}/{file:}   |
   shared)           |   Overlay          <- defaults . models        |
                     |   ModelGateway     <- resolve(kind, requested,  |
  models.json  -->   |                          ws, overrides)         |
  (falkor-chat       |                    <- .llm(...) / .embedder(...)|
   overlay)          +-------+-----------------------------------------+
                             | ResolvedModel(ref, base_url, model, key, timeout, params)
                             v
                     +------------------------------------+
                     | transport.make_http_transport()    |  timeout + headers + loud errors
                     +-------+----------------------------+
                             v
        OpenAICompatibleLLM / OpenAICompatibleEmbedder
                             ^
      +--------------+-------+--------+------------------+----------------+
   responder      judge            executor         embedding worker   retrieval tool
   (kind=agent)   (kind=guard)     (kind=step)      (kind=embedding)   (kind=embedding)
```

**Four closed consumer kinds** (`agent`, `step`, `embedding`, `guard`) — adding a fifth means
adding its own override property, or it silently escapes FR-17's future hard cap (routed to
`tico`, `docs/plans/llm-provider-config.md` §9.3). Five binding sites resolve through them:
`AgentResponder.maybe_respond` (`agent` + `embedding`), the executor's `_run_agent_node` (`step`),
`EmbeddingWorker.embed_message` (`embedding`), `GraphragRetrieveTool.run` (`embedding`), and the
llm-kind guard judge (`guard`). Resolution is **per call**, not at construction — the workspace
override (Landing 2) is then a function of `ws` with no signature changes.

**Two hand-edited files** feed the gateway (`FALKORCHAT_OPENCODE_CONFIG` — a pristine, unmodified
OpenCode `opencode.json`, providers only, no product default; `FALKORCHAT_MODEL_CONFIG` — falkor-
chat's own overlay, defaults to the shipped `config/models.json`), read once at wiring time
(`ModelGateway.from_env()` — no reload path). `config.assert_no_legacy_model_env()` refuses to
start if any of the four legacy per-provider/per-model env vars (`config.LEGACY_MODEL_ENV_VARS`)
is still set.

**The `/v1` normalization rule (AC-1).** LM Studio's `baseURL` convention omits `/v1`, and a
missing `/v1` is not an HTTP error — it is a `200` carrying an error envelope (a string on one
wrong-prefix path, an object on the right one, on the *same* server), so falkor-chat must
normalize rather than probe. Three ordered steps: **validate** (`scheme in {http,https}` and a
non-empty `netloc`, or reject at load naming the provider/file/value), **strip** every trailing
`/`, then **normalize** (append `/v1` only when the path is now empty; otherwise use it verbatim).
An overlay `providers.<id>.baseURL` override wins outright over both the file and the rule — used
exactly as declared, never auto-suffixed — and one INFO line per provider at startup names the
declared `baseURL`, the resolved API base, and which of {the rule, the override, verbatim}
produced it.

**The `guard` kind's workspace carrier (§4.10 of the plan).** Three of the four kinds carry `ws`
to their resolution point via `ctx`; the llm-guard judge does not — `guards.evaluate_guard`'s
`ctx` is the *run* ctx dict, not a `CallContext`, and `_select_transition` has no `CallContext`
either. `executor._drive` stamps `run["ws"] = ctx.ws` outside the SHA-locked `_drive_loop`
boundary (a fresh per-drive dict, never shared, never stored on `self`); `evaluate_guard` forwards
`run=` to the judge only when the judge advertises `accepts_run = True` (the production
`app._LlmGuardJudge`, not the closure it used to be) and forwards `model=` only when the guard
itself declared one (`{"kind":"llm","text":…,"model":…}`) — both zero-churn conditional kwargs, so
every stub judge in the test suite is called exactly as before.

**Roles + ordered fallback chains (Landing 2, FR-7/FR-18).** A ref with no `/` now resolves as a
**role name** — looked up in the overlay's `roles` map and expanded to an ordered, settings-applied
chain of `provider/model` refs, rather than being rejected. A role name must not itself contain
`/`, and a chain element that resolves to another role is rejected at **load** time, not first use
— the role namespace can never accidentally nest. `ModelGateway.llm()`/`.embedder()` build a
`FallbackClient` over the chain's resolved clients: `.chat(...)`/`.complete(...)` try element 0,
then element 1, … on a `ProviderCallError` (a transport-layer `TimeoutError` already converts to
one, closing B-2), and raise naming **every** model tried only if all fail. `FallbackClient` holds
no mutable "last used" state (`__slots__` makes that structural, M-5) — the answering model and
whether it came from a fallback travel on the `ChatResult` return value itself: `.model` (the
answering ref) and `.fallback` (`True` iff a later element answered, `None` — never `False` — for
a one-element chain or a direct non-role ref).

**The resolved-model trace (Landing 2, FR-8).** `StepRun` gains three durable properties —
`resolvedModel`, `modelSource` (`workspace`/`step`/`default` — the precedence rung that won,
Landing 2) and `modelFallback` (nullable bool, orthogonal to `modelSource` — a workspace override
can itself resolve to a role with its own fallback) — written by the same atomic
`record_step_and_advance` every run already calls, and surfaced on `GET
/workflow-runs/{id}/step-runs`. This is never a `TraceEvent`: those are debug-only by construction
(a non-debug run writes zero), so an audit-relevant field placed there would silently vanish on
precisely the runs nobody thought to flag for debugging — see `docs/plans/llm-provider-config-graph.md`
§1.2 for the full rejection rationale. An agent node that loops and answers on more than one model
records the **last** iteration's three fields together, never a mix of iterations.

**The workspace override + precedence, and the guard-kind hard cap (Landing 2, FR-16/FR-17,
closes B-1).** A per-workspace `WorkspaceConfig` singleton (one MERGE-backed node per `ws:{id}`)
carries an optional per-kind override, read once per drive/responder call and stamped onto
`run["modelOverrides"]` — never re-read per resolution. `ModelGateway.resolve()` now implements
the real, first-match-wins precedence: **workspace → the consumer's own requested choice → the
per-kind default.** The workspace rung is a **hard cap**: when present it wins outright, even over
an explicit `requested=`, for **all four consumer kinds — `guard` included**. `guard` is the kind
this section already flagged above as lacking a `CallContext`-borne `ws`; the `run["ws"]` carrier
documented there is exactly what makes the workspace override reachable at the guard-judge
resolution point too, closing finding B-1 (Landing 2's plan review had found the naive fix would
otherwise have to reopen the SHA-locked `_drive_loop`).

**Publish-time rejection (Landing 2, FR-9).** `publish_workflow_def` now runs
`_check_models_resolvable` **immediately before `self._repo.publish_def(...)`, after
`_check_no_structural_conflict`** (K-034's topology-conflict check): every step's `config.model`
and every `{"kind":"llm"}` guard's `model` is resolved through the gateway (no `ws=`/`overrides=` —
a global publish is never gated by per-workspace state), and an unresolvable model or role fails
the publish with a **400** naming the offending step key (or transition endpoints) and the
identifier — nothing is written. A def that fails **both** checks (bad topology **and** an
unresolvable model) returns K-034's **409**, not this check's 400, since the ordering runs the
structural check first. A `Services` built without a gateway skips the pass, but logs a WARNING
naming the def and its unchecked identifiers if it declares any model/role, so the skip is never
silently invisible.

**The embedding-dimension guard (Landing 2, FR-19).** Before the first embed write for a
`(workspace, label)` pair, `EmbeddingWorker` compares the resolved embedding model's *declared*
dimension against the workspace's *introspected* vector-index dimension (`Repository
.read_index_dimension`, cached per `(ws, label)` for the process lifetime — never caching a
failure) and raises `EmbeddingDimensionError` **before** calling the embedder on a mismatch: no
vector is written, no inference is wasted. This closes a real silent-failure mode — a wrong-
dimension `vecf32` write is accepted at `SET` with no engine-level error, then simply drops out of
ANN, so retrieval quietly finds nothing with no error anywhere in the chain.

Full design + rationale: `docs/plans/llm-provider-config.md` §3–§4, §7; graph design:
`docs/plans/llm-provider-config-graph.md`; requirements: `docs/requirements/llm-provider-config.md`.

---

## 15. MCP transport (K-002) — the agent front door

M1 exposes a second, additive transport for AI agents: **MCP over Streamable-HTTP**, mounted on
the *same* FastAPI process and calling the *same* `services.py` as the REST router. Full spec and
rationale: `docs/archive/plans/m1-chat-mcp.md`. Two capabilities were folded into M1 to support it:
participant **@mentions** (`MENTIONS_MEMBER` edge) and per-member **read-cursors** (`ReadCursor`).

### 15.1 Shape

```
browser ── REST/JSON ──┐
                       ├─▶ services.py ─▶ repository.py ─▶ FalkorDB
agents  ── MCP/HTTP ───┘   (all invariants here; both front doors call the SAME methods)
```

`mcp.py` is a thin adapter (peer of `api.py`), no business logic. `app.py`'s `create_app()`
builds one `Services`, `mcp.configure(services)`, then:

```python
mcp_app = mcp.streamable_http_app()
app = FastAPI(lifespan=mcp_app.router.lifespan_context)  # MUST forward, or session mgr never inits
app.include_router(api.build_router(services))
app.mount("/mcp", mcp_app)                                # agents connect at /mcp
```

> **Lifespan gotcha (python-sdk #1367):** forward the MCP app's lifespan to FastAPI or the
> Streamable-HTTP session manager is never started (requests 500 with "task group not
> initialized"). On this `mcp` build the lifespan is `mcp_app.router.lifespan_context`, and the
> handler's own path is set to `/` (`mcp.settings.streamable_http_path = "/"`) so mounting under
> `/mcp` yields a clean `/mcp` endpoint rather than `/mcp/mcp`. The app's lifespan also runs
> `services.ensure_actor()` so the configured actor node exists before the first write (the §4
> write paths anchor on the author node — QUERIES.md §4 zero-rows note).
>
> **Trailing-slash gotcha (QA DEF-1, fixed):** Starlette's Mount serves the sub-app only under
> `/mcp/`; a bare `POST /mcp` was 405 and MCP clients don't auto-append the slash. `create_app`
> adds an ASGI path-alias middleware rewriting `/mcp` → `/mcp/` so both spellings work.

### 15.2 Tools → service → query

| MCP tool | Service method | Query |
|---|---|---|
| `send_message(body, re, mentions=[], frm=None)` | `post_message` | §4 first/subsequent (+ mentions) |
| `read_messages(re?, since?, limit, advance=True)` | `read_messages` | §9.1 (thread) / §9.2 (room-wide) |
| `create_thread(channel_id, title)` | `create_thread` | §3 create a thread |
| `create_channel(name)` | `create_channel` | §3 create a channel |
| `list_channels(limit=50)` | `list_channels` | §3 list channels in a workspace |
| `list_threads(channel_id, limit=50)` | `list_threads` | §3 list recent threads in a channel |
| `search_messages(query, limit=50)` | `search_messages` | §5 full-text keyword search |

- **Actor identity (Q#1):** MCP ignores any client-supplied `frm`; every call is attributed to the
  `get_context()` actor (§14.3). M1's actor is the single configured `User` (role `user`).
- **`read_messages` is RW when it advances a cursor.** Explicit `since` → pure read; otherwise the
  per-thread cursor is read and (unless `since` given) advanced to the newest `createdAt` actually
  delivered — never the server clock, which would permanently skip rows a `limit` truncated (an
  empty page advances nothing). Rows are chronological with reader-mentions carried by the
  `isMention` flag (see `QUERIES.md` §9 ordering note). Room-wide reads (no `re`) default `since`
  to epoch 0 and never advance (no room cursor in M1, Q#3).
- **REST mention parity:** `POST /threads/{tid}/messages` also accepts an optional `mentions[]`.

### 15.3 Client connection contract

Streamable-HTTP; a consuming agent points at the URL (no subprocess):

```json
{ "mcpServers": { "falkor-chat": { "type": "streamable-http", "url": "http://localhost:8000/mcp" } } }
```

Unauthenticated in M1 — bind to localhost / a trusted network only. Run:
`cd server && .venv/bin/uvicorn falkorchat.app:app` (bootstrap `ws:acme` first).
