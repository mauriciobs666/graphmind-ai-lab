# falkor-chat — Design Document & Blueprint

> **Philosophy:** FalkorDB graph for *everything* — reference data, workflow definitions
> and runs, chat history, and user/workspace information. No second store for the primary
> domain. One engine, one query language (OpenCypher), one operational model (Redis).

**Status:** Draft v0.3 — thread-scoped model, DayBucket removed
**Date:** 2026-06-06
**Owner:** mauriciobs@gmail.com

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

> User-approved M2 stack. Locked here; implemented in K-008/K-013 (see kaizen/plan.md). Numbers
> detailed in §11 (RAM) and §12 (M2 roadmap).

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
│   Canonical WorkflowDef templates (versioned, immutable)              │
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
| **B. Materialize (chosen for defs)** | On workflow *publish*, copy the def subgraph (immutable, versioned) into each workspace graph that uses it | Real edges → runs traverse their own steps locally; def graphs stay small and are duplicated cheaply |

**Decision:** canonical definitions live in `reference` (single source of truth, versioned,
immutable once published). When a workspace first uses `defKey@v`, we **materialize that
version's step subgraph into `ws:{id}`** under a `WorkflowDefSnapshot`. Runs then have real,
local edges to their steps — fast, self-contained, and the snapshot is immutable so it never
drifts. Same pattern applies to any *small, shared, read-mostly* reference subgraph a hot
traversal needs to walk (e.g. an ontology fragment).

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
(:StepRun)-[:EMITTED]->(:Message)
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
  messages; `EMITTED` provenance back to `StepRun` is explicit and auditable.

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
idempotent success — see the §9 table and QUERIES.md §4.

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
(:Step {stepUid, key, type, config})               // type: prompt|tool|decision|human|message|wait
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

`type` unifies conversational and business flows:
- `prompt` / `message` / `tool` → agent flows (LLM call, post a message, invoke a tool)
- `human` / `decision` / `wait` → business processes (assignee task, branch, SLA/timer)

`TRANSITION.guard` is an expression evaluated against run context; `on` is the event/outcome
that fires it. One model, both worlds.

### 6.2 Run (per-workspace, real local edges to the materialized def)

```
(:WorkflowRun {runId, defKey, defVersion, status, startedAt, ctx})
(:WorkflowRun)-[:OF_DEF]->(:WorkflowDefSnapshot {key, version})   // local, materialized
(:WorkflowRun)-[:AT_STEP]->(:Step)                               // current position
(:WorkflowRun)-[:HAS_STEP_RUN]->(:StepRun)
(:StepRun {stepKey, status, startedAt, endedAt, input, output})
(:StepRun)-[:RAN]->(:Step)                                       // which def step
(:StepRun)-[:NEXT]->(:StepRun)                                   // execution order (audit trail)
(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)                       // chat linkage
(:StepRun)-[:EMITTED]->(:Message)
```

> `ctx` (on `WorkflowRun`) and `input`/`output` (on `StepRun`) are **flat, serialised strings**,
> not nested maps — FalkorDB stores only scalars and scalar lists. Queries never filter *inside*
> them (see §1.2).

The engine loop: read `AT_STEP` → evaluate outgoing `TRANSITION` guards against `ctx` →
create the next `StepRun` → execute (LLM/tool/human) → append to the `NEXT` trace → move
`AT_STEP`. The whole walk is local to the workspace graph (fast, isolated, fully auditable).

> **`status` as a property, not a label**, so a run's state changes in place without
> re-labeling churn; index it for "all running workflows" queries.

### 6.3 Coordination is workflow, not a separate primitive

Agent/team coordination (task lifecycle, "room state") is modelled as an M3 `WorkflowDef` of
`kind:'process'` over `Step` + `TRANSITION` + `StepRun` — **not** a flat `Task` node or a
presence field. This avoids a parallel model that would later need migrating into the engine
(single-store philosophy). Full rationale/ADR: `docs/plans/m1-chat-mcp.md` Appendix B.

---

## 7. Indexes, constraints & vector search

### 7.1 Per workspace graph `ws:{id}`

> **Executable DDL is `scripts/bootstrap_schema.sh` — the single source of truth**, asserted by
> `test_queries.sh` (126/126). This section describes *what* is indexed/constrained and *why*, not
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
| Publish workflow def | write to `reference`; materialize snapshot into consuming `ws:{id}` graphs | Immutable per version; bump version, never mutate in place |

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

- **Workflow guard expression language** — reuse an existing expr lib or define a minimal DSL stored in `Step.config`? (→ M3, decided with the engine.)
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
                                              └────────────────────────────────────────────┬─┘
                                                                                            ▼  FalkorDB
```

- **`repository.py` is the only place Cypher lives.** Each method maps 1:1 to a verified query in
  `QUERIES.md`, always parameterised (`params=`), `ro_query` for reads / `query` for writes,
  `select_graph(f"ws:{id}")` for scoping.
- **`services.py` owns the invariants** the write-path rules describe: choosing the first-vs-subsequent
  append variant, id generation, `Thread.updatedAt` bumps, setting `role`/`POSTED_BY`.
- **`api.py` is the only layer that changes** if the transport is ever revisited.

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

---

## 15. MCP transport (K-002) — the agent front door

M1 exposes a second, additive transport for AI agents: **MCP over Streamable-HTTP**, mounted on
the *same* FastAPI process and calling the *same* `services.py` as the REST router. Full spec and
rationale: `docs/plans/m1-chat-mcp.md`. Two capabilities were folded into M1 to support it:
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
