# falkor-chat — Design Document & Blueprint

> **Philosophy:** FalkorDB graph for *everything* — reference data, workflow definitions
> and runs, chat history, and user/workspace information. No second store for the primary
> domain. One engine, one query language (OpenCypher), one operational model (Redis).

**Status:** Draft v0.3 — thread-scoped model, DayBucket removed
**Date:** 2026-06-06
**Owner:** mauriciobs@gmail.com

---

## 1. Decisions locked in

| Axis | Decision | Consequence |
|---|---|---|
| **Chat type** | **Hybrid** — humans chat in channels *and* an AI participant answers from the graph (GraphRAG) | Messages have a `role` (human/assistant/system); AI is a first-class `Agent` author; retrieval = vector + traversal |
| **Tenancy** | **Per-workspace/team graph** — the graph boundary is a workspace, many users share it | One named graph per workspace; each workspace must fit one shard's RAM; workspaces distribute across a Redis Cluster |
| **Workflows** | **General workflow engine** — one definition model serves both conversational/agent flows and business processes | Definitions (templates) are reusable + versioned; runs are per-workspace execution traces |

---

## 2. The hard FalkorDB constraints this design is shaped around

These are not style choices — they are engine facts that the topology must respect.

1. **In-memory, RAM-bound.** The whole graph + its sparse adjacency matrices + indexes live in RAM. **Memory is the binding constraint**, sized per graph.
2. **A single graph lives entirely on one shard.** FalkorDB does *not* split one graph across shards (no Neo4j-Fabric equivalent). We scale by spreading *many* workspace graphs across cluster shards — a natural fit for per-workspace tenancy. **A single workspace can never outgrow one node's RAM.**
3. **Relationships cannot cross graphs.** An edge can only connect two nodes in the *same* named graph. This is the single most important fact for the reference-data / workflow-definition split (see §4 and §6). Cross-graph references are carried as **properties (keys), resolved at query time**, or by **materializing a copy** of the shared subgraph into the workspace graph.
4. **OpenCypher subset, not Neo4j.** No APOC, no GDS, no Fabric. Algorithms are built-in `algo.*` procedures; full-text/vector are `db.idx.*` procedures; profiling is the `GRAPH.PROFILE` *command*, not a `PROFILE` keyword prefix.
5. **Supernodes are dense matrix rows.** Less catastrophic than pointer-chasing engines (traversal is matrix algebra), but a Channel with millions of `HAS_MESSAGE` edges is still a dense row that costs RAM and compute. We avoid it with a linked-list + time-bucket pattern (§5).

> **Live-verified** on this deployment: Redis 8.2.2, FalkorDB module `999999` (edge/main),
> `vectorset` module also loaded. The findings below in §7 reflect confirmed behavior — not docs
> assumptions. Details: cross-graph edges confirmed silent (no error, MATCH returns 0); constraint
> requires an existing range index first; vector indexes use DDL syntax, not a procedure call;
> `db.idx.vector.createNodeIndex` is **not registered** in this build. See §7 for corrected commands.

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
(:Message {msgId, text, role, createdAt})-[:NEXT]->(:Message)  // thread-scoped linked list

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
- `Message.role` — `'human'` | `'assistant'` | `'system'` (fast filter without traversing `POSTED_BY`)
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
already has a `HEAD` (§14 keeps this dispatch inside `post_message`).

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
(:WorkflowDef)-[:START]->(:Step)
(:Step {key, type, config})                        // type: prompt|tool|decision|human|message|wait
(:Step)-[:TRANSITION {on, guard, order}]->(:Step)  // edge-labeled state machine
```

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

The engine loop: read `AT_STEP` → evaluate outgoing `TRANSITION` guards against `ctx` →
create the next `StepRun` → execute (LLM/tool/human) → append to the `NEXT` trace → move
`AT_STEP`. The whole walk is local to the workspace graph (fast, isolated, fully auditable).

> **`status` as a property, not a label**, so a run's state changes in place without
> re-labeling churn; index it for "all running workflows" queries.

---

## 7. Indexes, constraints & vector search

### 7.1 Per workspace graph `ws:{id}`

**Critical ordering rules (live-verified):**
1. `GRAPH.CONSTRAINT CREATE` requires an existing range index on the same property — always index first.
2. Composite constraints (`PROPERTIES 2 …`) are supported and **live-verified** on this build — `bootstrap_schema.sh` creates them and `test_queries.sh` asserts they block duplicate `key+version`.

```
-- ── Step 1: Range indexes (must exist BEFORE constraints) ──────────────────

-- Identity anchors (back the uniqueness constraints below)
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:User)                ON (n.userId)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:Channel)             ON (n.channelId)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:Thread)              ON (n.threadId)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:Message)             ON (n.msgId)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:Agent)               ON (n.agentId)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:Document)            ON (n.documentId)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:Chunk)               ON (n.chunkId)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:Entity)              ON (n.entityId)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:WorkflowDefSnapshot) ON (n.key)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:WorkflowDefSnapshot) ON (n.version)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:WorkflowRun)         ON (n.runId)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:StepRun)             ON (n.stepRunId)"

-- Hot filter indexes (no constraint needed)
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:Thread)      ON (n.updatedAt)"   -- recent threads
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:Message)     ON (n.createdAt)"   -- time-range reads
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:WorkflowRun) ON (n.status)"
GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:StepRun)     ON (n.status)"

-- ── Step 2: Uniqueness constraints ─────────────────────────────────────────
-- Response is "PENDING" → becomes OPERATIONAL asynchronously.
-- Verify with: GRAPH.QUERY ws:acme "CALL db.constraints()"

GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE User                PROPERTIES 1 userId
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE Channel             PROPERTIES 1 channelId
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE Thread              PROPERTIES 1 threadId
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE Message             PROPERTIES 1 msgId
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE Agent               PROPERTIES 1 agentId
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE Document            PROPERTIES 1 documentId
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE Chunk               PROPERTIES 1 chunkId
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE Entity              PROPERTIES 1 entityId
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE WorkflowRun         PROPERTIES 1 runId
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE StepRun             PROPERTIES 1 stepRunId
-- composite constraint — PROPERTIES 2 syntax live-verified on this build:
GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE WorkflowDefSnapshot PROPERTIES 2 key version

-- ── Step 3: Full-text index (RediSearch) ───────────────────────────────────
GRAPH.QUERY ws:acme "CALL db.idx.fulltext.createNodeIndex('Message', 'text')"

-- ── Step 4: Vector indexes ─────────────────────────────────────────────────
-- ⚠️  db.idx.vector.createNodeIndex is NOT a registered procedure (live-verified).
-- Use DDL. Dimension must match the embedding model exactly (e.g. 1536 for text-embedding-ada-002).
GRAPH.QUERY ws:acme "CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding) OPTIONS {dimension:1536, similarityFunction:'cosine'}"
GRAPH.QUERY ws:acme "CREATE VECTOR INDEX FOR (n:Chunk)   ON (n.embedding) OPTIONS {dimension:1536, similarityFunction:'cosine'}"

-- Vectors stored as vecf32; score is cosine distance (0 = identical, lower = more similar):
--   WRITE: SET n.embedding = vecf32([0.1, 0.2, ...])
--   READ:  CALL db.idx.vector.queryNodes('Message','embedding', $k, vecf32($vec))
--          YIELD node, score  →  ORDER BY score ASC
```

### 7.2 `reference` graph

```
-- Same ordering rule: index first, constraint second
GRAPH.QUERY reference "CREATE INDEX FOR (n:WorkflowDef) ON (n.key)"
GRAPH.QUERY reference "CREATE INDEX FOR (n:WorkflowDef) ON (n.version)"
GRAPH.QUERY reference "CREATE INDEX FOR (n:Entity)      ON (n.entityId)"

GRAPH.CONSTRAINT CREATE reference UNIQUE NODE WorkflowDef PROPERTIES 2 key version
GRAPH.CONSTRAINT CREATE reference UNIQUE NODE Entity      PROPERTIES 1 entityId
```

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
traversal — one read-only query, routable to a replica:

```cypher
// $qVec = vecf32 of query embedding, $k = neighbors to retrieve, $chId = channel scope
// score = cosine distance (0 = identical, lower = more similar) → ORDER BY score ASC
CALL db.idx.vector.queryNodes('Message', 'embedding', $k, $qVec)
YIELD node AS seed, score
// scope to the target channel via Thread
MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed)
MATCH (c:Channel {channelId:$chId})-[:HAS_THREAD]->(t)
// expand to related messages that mention the same entities
OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message)
WITH seed, score, collect(DISTINCT related)[..5] AS expanded
RETURN seed.text AS hit, score, [m IN expanded | m.text] AS context
ORDER BY score ASC
```

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
| Post message | `MERGE (m:Message {msgId:$id})` + relink `Thread TAIL → NEXT` | Idempotent via unique constraint; O(1) append; two Cypher variants (first vs subsequent message — see §5.3) |
| Backfill / import | `UNWIND $rows AS row …` in chunks, or `falkordb-py` bulk loader | Never one giant CREATE — bound transaction memory; size batches |
| Embed messages | async worker: compute embedding → `SET m.embedding = vecf32($v)` | Decouple embedding latency from the post path |
| Advance workflow | create `StepRun`, append `NEXT`, move `AT_STEP` | All local to `ws:{id}`; fully transactional within the graph |
| Publish workflow def | write to `reference`; materialize snapshot into consuming `ws:{id}` graphs | Immutable per version; bump version, never mutate in place |

**Rule:** every `MERGE` is backed by a uniqueness constraint, or it's a duplicate-node bug
waiting for concurrency.

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
- **Observability:** `GRAPH.SLOWLOG` for slow queries, `GRAPH.PROFILE` for plans, Redis metrics.
- **Security:** Redis ACLs scoping `GRAPH.*` per principal (ideally per workspace key pattern),
  TLS in transit, network isolation; secrets outside the data.

---

## 11. Capacity sketch (fill in with real numbers)

Per workspace, rough RAM order-of-magnitude:

```
nodes        ≈ (#messages + #chunks + #entities + #runs + #stepruns) × ~payload bytes
rel matrices ≈ Σ per-type sparse-matrix overhead (sparse → small unless dense nodes)
vector index ≈ embeddingDim × 4 bytes × (#messages + #chunks)        ← usually dominant
fulltext     ≈ RediSearch index over Message.text
+ Redis overhead + ~30% headroom
```

> Action: instrument a pilot workspace, measure `GRAPH.MEMORY USAGE`, and back into a
> per-workspace RAM budget + a shard:workspace packing ratio before scaling out.

---

## 12. Roadmap

1. **M0 — Stand up the engine.** ✅ FalkorDB running (`falkordb/falkordb:edge`, Redis 8.2.2, module `999999`) via Docker. Live probes confirmed: cross-graph edge behavior, vector DDL syntax, index-before-constraint ordering, `algo.*` procedure set, `vecf32` storage and `db.idx.vector.queryNodes` query surface.
2. **M1 — Chat core.** Users/Channels/Threads/Messages, thread-scoped `NEXT` + `HEAD`/`TAIL` append path, full-text index, basic read windows. Load test the append path; `GRAPH.PROFILE` the hot reads. **Application layer:** FastAPI REST server over a service/repository split, single hardcoded tenant, minimal web UI — full design in §14. **Plus an MCP (Streamable-HTTP) agent front door on the same service layer — §15 (K-002).** Full stack (repository → services → MCP + REST + full-text `search`, plus the static `web/` UI, all mounted in `app.py`) is built and green (68 tests). M1 chat core is code-complete.
3. **M2 — GraphRAG.** Embedding workers, in-graph vector index, hybrid retrieval query (§8), AI `Agent` participant posting answers with `EMITTED` provenance.
4. **M3 — Workflow engine.** Definition model in `reference`, snapshot materialization, run/step-run executor, chat linkage; both a conversational flow and a business-process flow as proof.
5. **M4 — Scale & ops.** Redis Cluster, replicas for RO reads, Sentinel, ACL/TLS, backup/restore drill, per-workspace memory budgeting + shard packing.

---

## 13. Open questions

- **Embedding model & dimension** (fixes vector index size and RAM line).
- **Workflow guard expression language** — reuse an existing expr lib or define a minimal DSL stored in `Step.config`?
- **Identity source of truth** — is `identity` graph authoritative, or a projection of an external IdP?
- **Retention** — do old messages/embeddings age out (and how does that interact with the always-in-RAM constraint)?
- **Cross-workspace analytics** — app-layer fan-out vs. a dedicated `analytics` rollup graph.
- **Bolt vs. RESP** for the app gateway — Bolt port is `65535` (confirmed in `GRAPH.CONFIG`); decide whether to use it or RESP with `falkordb-py`.
- **Live config defaults noted:** `THREAD_COUNT 4`, `OMP_THREAD_COUNT 4`, `TIMEOUT 1000ms`, `CACHE_SIZE 25`, `MAX_QUEUED_QUERIES 25`, `QUERY_MEM_CAPACITY 0` (unlimited), `ASYNC_DELETE 1`. Review before production.

---

## 14. M1 application architecture (client/server)

§10 sketches the *operational* topology (app ⇄ FalkorDB). This section pins the *application*
code architecture for **M1 — Chat core**: what the client and server are, the transport between
them, and the internal layering.

### 14.1 Scope decisions locked for M1

| Axis | Decision | Rationale |
|---|---|---|
| **Transport** | **REST/JSON over FastAPI** | The only M1 client is a browser, which speaks HTTP natively — no gRPC-Web bridge tax. Free OpenAPI console to exercise the API. M2 real-time adds native WebSocket/SSE on the same server. |
| **Client** | **Minimal web UI** (channels list + thread view) | Smallest end-to-end path that exercises the full stack visually. |
| **Real-time** | **Deferred to M2** | M1 is request/response; the UI re-fetches a thread window after posting. The push path (Redis Pub/Sub → WebSocket) slots onto the same service layer in M2 with no schema change. |
| **Auth / tenancy** | **Single hardcoded tenant** — `ws=acme`, `user=u1` | Keeps M1 focused on the chat data path. Injected at one seam (see §14.3) so real auth replaces it without touching services/repo. |

> Transport was deliberately re-evaluated away from gRPC: gRPC's wins (polyglot typed contracts,
> native streaming, service-to-service perf) are all unused when the sole client is a browser, and
> gRPC-Web can't do client/bidi streaming in browsers anyway — WebSocket/SSE is the stronger M2
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
| `POST /channels` | `create_channel` | §3 create a channel |
| `GET /channels` | `list_channels` | §3 list channels in a workspace |
| `POST /channels/{cid}/threads` | `create_thread` | §3 create a thread |
| `GET /channels/{cid}/threads` | `list_threads` | §3 list recent threads in a channel |
| `POST /threads/{tid}/messages` | `post_message` | §4 first message / subsequent message |
| `GET /threads/{tid}/messages[?after=]` | `read_thread` | §4 read full thread / read thread window |
| `GET /threads/{tid}/messages/{mid}` | `get_message` | §4 get a single message |
| `GET /search?q=` | `search_messages` | §5 full-text keyword search |

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
