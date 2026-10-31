# falkor-chat — Design

> **Status:** active · **Owner:** `architect` · **Tracks:** —

> **Philosophy:** FalkorDB graph for *everything* — reference data, workflow definitions
> and runs, chat history, and user/workspace information. No second store for the primary
> domain. One engine, one query language (OpenCypher), one operational model (Redis).

> **How to read this document.** It is part of falkor-chat's live spec and states the graph and
> the workflow engine **as they are now**: topology, schema, indexes and constraints, executor
> semantics, write paths and capacity. Change history lives in `falkor-chat/docs/HISTORY.md`;
> proposed but unbuilt work lives in `falkor-chat/docs/BACKLOG.md`, and a `K-` number appears
> here only when it points at an open item there. The server process (REST/MCP surface,
> auth/tenancy seam, configuration, model resolution) is `falkor-chat/docs/SERVER.md`; the exact
> text of every query is `falkor-chat/docs/QUERIES.md`; the executable DDL is
> `scripts/bootstrap_schema.sh`.

---

## 1. Decisions locked in

This section is the **single authoritative decision register**. §1.1 holds the top-level axes;
§1.2 the detailed locked design decisions; §1.3 the model stack. Each row is the
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
| **`Message.role` inline property; values `user`/`assistant` derived server-side** from the author label (`User→user`, `Agent→assistant`), never trusted from the caller | Filter by role without traversing `POSTED_BY`; agents author first-class | §5.1, §5.2 |
| **`coalesce(u.userId, a.agentId)` for member identity** (two indexed `OPTIONAL MATCH` + `coalesce`) | `User` has `userId`, `Agent` has `agentId` — both are members; anchored lookup avoids the `OR`-scan | QUERIES §2 |
| **Vector indexes via DDL**, not a procedure | `db.idx.vector.createNodeIndex` is not registered on this build | §2, §7.1 |
| **Index before constraint, always** | `GRAPH.CONSTRAINT CREATE` requires a pre-existing range index | §2, §7.1 |
| **`Message.embedding` inline as `vecf32`** | Single-query vector + traversal hybrid retrieval | §5.2, §7.3 |
| **Vector score is cosine *distance*** (0 = identical) → `ORDER BY score ASC` | Most-similar-first ranking | §7.1, §8 |
| **`status` as a property, not a label** | Avoids re-labeling churn on state changes; index it for "all running" reads | §6.2 |
| **`ctx` / `input` / `output` are flat/serialised strings** | FalkorDB stores scalars + scalar lists only — no nested maps; never query inside them | §6.2 |
| **`Message.threadId` denormalized inline, unindexed** | Nav metadata for QUERIES §9.2/§5 rows; HEAD/NEXT walk stays canonical; unindexed saves RAM/write cost | §5.1 |
| **Guarded-CREATE write paths** (`FOREACH`+`CASE` per path) with an always-returned status row; **no MERGE on `Message`** | Retry replay is a no-op (`dupMsg`); first-post race refused (`hadHead`); uniqueness constraint is the backstop | §5.3, §9 |
| **Composite `(createdAt, msgId)` keyset cursor** (`ReadCursor.lastReadAt`/`lastReadMsgId`) | Timestamp alone is not a total order — same-ms ties skipped rows; cursor reads are lossless | QUERIES §9.1/§9.3 |
| **Member ids are namespace-unique across `User`/`Agent`**; `ensure_user`/`ensure_agent` are guarded-CREATE queries returning `(created, existed, collided)`; cross-label collision refuses (`MemberIdCollisionError`) | A shadow node with the other label's id eclipses it in every `coalesce` lookup | QUERIES §2/§7 |
| **Identity source of truth — the `identity` graph is authoritative (standalone)**, not an external-IdP projection | Self-contained system; the `identity` graph owns user identity + auth principals; per-workspace `User` nodes are membership projections of it; steers the open **K-016** auth work | §3, `SERVER.md` §1.3 |

### 1.3 Model stack

> The rows below are the *shipped defaults*, not env vars: they live in `config/models.json`'s
> per-kind `defaults` (kinds `agent`, `step`, `guard`, `embedding`, `extraction`) plus the shared
> `config/opencode.example.json`'s provider entries, so **changing a model is a config-file edit +
> restart**, never a code or env change. How a model reference resolves (roles, fallback chains,
> the per-workspace override) is `SERVER.md` §1.8. The one exception is `EMBEDDING_DIM`
> (`FALKORCHAT_EMBEDDING_DIM`), which stays an env var because it is DDL-time/write-path input,
> not a model choice — and it is fixed at workspace creation (§11). RAM numbers: §11.

| Component | Decision | Rationale | Detailed in |
|---|---|---|---|
| Embedding model | **Qwen3-Embedding-0.6B** (GGUF, Q8_0) | Best small-model MTEB quality; 100+ languages (PT-BR + EN); ~0.6 GB resident | §11 |
| Vector dimension | **`EMBEDDING_DIM=1024`** | Native dim; ~12.4 KB/message with HNSW — the §11 RAM line | §11 |
| Default LLM (every chat kind) | **Qwen3-4B-Instruct-2507** Q4_K_M (non-thinking) | RAG answering, not CoT; low latency | `config/models.json` |
| Storefront agent LLM | **Ministral-3-3B**, pinned in the `salesperson` def's `assistant` step `config.model` (`proof_defs.py`); `query_graph_data` inherits the calling step's model (§6.1) | Fits the storefront demo's single-model GPU budget | §6.1 |
| Runtime | **LM Studio** on the Windows host (OpenAI-compatible), reached from WSL2 (mirrored networking → localhost) | Zero new moving parts | §10 |
| VRAM budget | **6 GB dedicated** (RTX 4050) — embedder + one chat LLM co-resident | Do not plan around shared-RAM spill | §11 |
| Upgrade path | **`qwen3-embedding:4b`** — same family, same 1024-dim MRL | Re-embed only; no schema change | §11 |

---

## 2. The hard FalkorDB constraints this design is shaped around

These are not style choices — they are engine facts that the topology must respect.

1. **In-memory, RAM-bound.** The whole graph + its sparse adjacency matrices + indexes live in RAM. **Memory is the binding constraint**, sized per graph.
2. **A single graph lives entirely on one shard.** FalkorDB does *not* split one graph across shards (no Neo4j-Fabric equivalent). We scale by spreading *many* workspace graphs across cluster shards — a natural fit for per-workspace tenancy. **A single workspace can never outgrow one node's RAM.**
3. **Relationships cannot cross graphs.** An edge can only connect two nodes in the *same* named graph. This is the single most important fact for the reference-data / workflow-definition split (see §4 and §6). Cross-graph references are carried as **properties (keys), resolved at query time**, or by **materializing a copy** of the shared subgraph into the workspace graph.
4. **OpenCypher subset, not Neo4j.** No APOC, no GDS, no Fabric. Algorithms are built-in `algo.*` procedures; full-text/vector are `db.idx.*` procedures; profiling is the `GRAPH.PROFILE` *command*, not a `PROFILE` keyword prefix.
5. **Supernodes are dense matrix rows.** Less catastrophic than pointer-chasing engines (traversal is matrix algebra), but a Channel with millions of `HAS_MESSAGE` edges is still a dense row that costs RAM and compute. We avoid it with the thread-scoped linked-list pattern (§5) — the channel-wide time-bucket
   alternative was considered and rejected (§1.2, "No DayBucket").

> **Live-verified on this deployment** — pinned `v4.18.11`, module `41811`, Redis 8.6.3, with the
> `vectorset` module also loaded. Four behaviours were probed directly rather than taken from the
> docs: a cross-graph edge fails **silently** (no error; the `MATCH` simply returns 0 rows); a
> constraint requires an existing range index first; a vector index needs DDL syntax, not a
> procedure call; and `db.idx.vector.createNodeIndex` is **not registered** on this build. §7 is
> what's indexed and why; the executable DDL is `scripts/bootstrap_schema.sh`.

---

## 3. Graph topology (multi-graph layout)

We use **four classes of named graph**. Each is an independent Redis key; edges stay within a class.

```
┌───────────────────────────────────────────────────────────────────────┐
│ identity                 (1 graph, global)                            │
│   Global user identity, auth principals, cross-workspace membership   │
│   Read-mostly. Replicated. Small.                                     │
├───────────────────────────────────────────────────────────────────────┤
│ reference                (1 graph, global, read-mostly)               │
│   Domain reference data / ontology / catalogs                         │
│   Canonical WorkflowDef templates (topology-immutable)                │
│   Product catalog (storefront demo; seed-script-written only)         │
│   Replicated; served via GRAPH.RO_QUERY                               │
├───────────────────────────────────────────────────────────────────────┤
│ ws:{workspaceId}         (N graphs, one per workspace)  ← hot path    │
│   Workspace-local Users (membership projection of identity)           │
│   Channels, Threads, Messages (chat history)                          │
│   WorkflowRun + StepRun execution traces                              │
│   Chunks/Documents + embeddings (GraphRAG corpus)                     │
│   Extracted Entities + fusion edges                                   │
│   Materialized copy of the WorkflowDef versions this ws uses          │
│   Storefront Customers, Carts, Orders; WorkspaceConfig singleton      │
├───────────────────────────────────────────────────────────────────────┤
│ (optional) analytics:{...}  rollups / cross-workspace aggregates      │
└───────────────────────────────────────────────────────────────────────┘
```

**Built today:** `reference` and `ws:{workspaceId}` (`server/falkorchat/db.py` selects only
those two key shapes). The `identity` graph is the decided home for user identity (§1.2) but holds
no data and has no DDL yet — the server runs one hardcoded tenant behind the auth seam, and real
auth is open work (**K-016**). `analytics:{...}` is optional and unbuilt (§13).

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
| **B. Materialize (chosen for defs)** | An explicit materialize call copies one published version's def subgraph (topology-immutable per version) into one workspace graph | Real edges → runs traverse their own steps locally; def graphs stay small and are duplicated cheaply |

**Decision:** canonical definitions live in `reference` (single source of truth, versioned).
Publishing writes `reference` only. A version reaches a workspace through an **explicit
materialize call** (`POST /workflow-defs/{key}/versions/{version}/materialize`, `SERVER.md` §1.4),
which copies that version's step subgraph into `ws:{id}` under a `WorkflowDefSnapshot`; a
version never published is refused (`404`). Nothing materializes implicitly: starting a run
against a version not materialized in that workspace raises `WorkflowRunNotFoundError` and
writes nothing (REST start: `404`; an `@mention` start logs it). Runs then have real, local edges to their
steps — fast, self-contained. **The enforced guarantee:** topology (steps, transitions, start)
is immutable per version — a re-publish/re-materialize whose topology differs from what's
stored is rejected (`409 WorkflowDefConflictError`) before any write, so the snapshot never
silently drifts from the def it was materialized from. Properties (`name`, `kind`, step
`config`, transition `guard`) are create-only — a differing resubmit of those is a silent
no-op. Same pattern applies to any *small, shared, read-mostly* reference subgraph a
hot traversal needs to walk (e.g. an ontology fragment).

> Large reference catalogs that are only *looked up* (not traversed from workspace nodes) stay
> in `reference` and are reached by property key — no materialization.

---

## 5. Chat model (hybrid: humans + AI)

### 5.1 Arrow notation

Everything below lives in `ws:{workspaceId}` unless marked `reference`.

```
// Members and membership
(:User  {userId, displayName, email})                  // a workspace member (projection of identity)
(:Agent {agentId, name, model, createdAt})             // an AI member, a first-class author
(:User)-[:MEMBER_OF {role, joinedAt}]->(:Channel)
(:Agent)-[:MEMBER_OF {role:'assistant', joinedAt}]->(:Channel)

// Channel → Thread → Message hierarchy
(:Channel {channelId, name, participantId, createdAt})-[:HAS_THREAD]->(:Thread {threadId, title, createdAt, updatedAt})
  // participantId: NULLable storefront provenance marker, written ONLY by
  // `ensure_participant` (QUERIES.md §18.1). It is what makes a channel a reset
  // target: both storefront resets match on `ch.participantId = u.userId`, and a
  // channel without it (every non-storefront channel) is structurally unreachable.
(:Thread)-[:HEAD]->(:Message)                          // first message in thread (set once)
(:Thread)-[:TAIL]->(:Message)                          // last message (moved on each append)
(:Message {msgId, text, role, createdAt, threadId, embedding: vecf32, toolsUsed})
(:Message)-[:NEXT]->(:Message)                         // thread-scoped linked list

// Authorship, mentions, provenance, read state
(:Message)-[:POSTED_BY]->(:User|:Agent)                // exactly one author
(:Message)-[:MENTIONS_MEMBER]->(:User|:Agent)          // resolved @mentions, written with the message
(:Message)-[:EMITTED {score, rank}]->(:Message|:Chunk) // an AI answer's retrieval seeds
(:User|:Agent)-[:HAS_CURSOR]->(:ReadCursor {cursorId, memberId, threadId, lastReadAt, lastReadMsgId})

// Storefront participant (a User with a token, its own channel and thread)
(:User {userId, displayName, tokenHash, language, channelId, threadId, joinedAt})

// GraphRAG corpus
(:Document {documentId, title, text, sourceFormat, sourceKind, status, pendingJobs, createdAt,
            currentVersion, supersededAt, supersededBy, textNormalizedHash})
  // status ∈ processing|ready|failed; sourceKind ∈ document|agent, derived from the
  // ingesting member's label, never trusted from the caller
(:Document)-[:INGESTED_BY]->(:User|:Agent)
(:Document)-[:HAS_CHUNK]->(:Chunk {chunkId, text, seq, documentId, documentCurrent, embedding: vecf32})
(:Document)-[:SUPERSEDES {matchId, status, confidence, technique, createdAt,
                          decidedAt, decidedBy, resuggestCount, lastResuggestedAt}]->(:Document)
  // new → old version; status ∈ pending|confirmed|rejected
(:DocumentDeletion {documentId, deletedBy, deletedAt})  // standalone audit node of a hard delete
(:Entity {entityId, name, nameNormalized, type, createdAt})
(:Chunk)-[:ABOUT]->(:Entity)
(:Entity)-[:RELATES_TO {label, sourceChunkId, sourceDocumentId, createdAt}]->(:Entity)
  // LLM-extracted fact; label = free-text predicate, never its own rel type; never deduplicated
(:Entity)-[:SAME_AS {matchId, status, confidence, technique, createdAt,
                      decidedAt, decidedBy, resuggestCount, lastResuggestedAt}]->(:Entity)
  // fusion match, status ∈ pending|confirmed|rejected; direction is a write-time convention
  // (new→existing), not a semantic claim — SAME_AS-anchored reads that need
  // direction-agnosticism match endpoints undirected (QUERIES.md §14.6)

// Read by queries but written by no current code path (dormant schema)
(:Message)-[:MENTIONS]->(:Entity)                      // hybrid_search's co-occurrence expansion
(:Message)-[:REPLY_TO]->(:Message)                     // get_message's quoted-message lookup

// Workflow ↔ chat linkage (§6.2)
(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)
(:StepRun)-[:PRODUCED]->(:Message)                     // step-emitted chat message — NOT the Message→Message EMITTED

// Workspace configuration singleton
(:WorkspaceConfig {workspaceConfigId:'default', agentModelOverride, guardModelOverride,
                   embeddingModelOverride, responderModelOverride,
                   modelOverrideUpdatedAt, modelOverrideUpdatedBy})

// Storefront commerce (customerId = the participant's userId)
(:Customer {customerId, createdAt, name, deliveryAddress, profileUpdatedAt})
(:Customer)-[:HAS_CART]->(:Cart {customerId, createdAt, updatedAt})          // at most one per customer
(:Cart)-[:HAS_ITEM]->(:CartItem {customerId, productId, quantity, addedAt, updatedAt})
(:Customer)-[:PLACED]->(:Order {orderId, status, placedAt, updatedAt})
  // status ∈ placed|fulfilled|delivered|cancelled
(:Order)-[:HAS_LINE]->(:OrderLine {productId, name, unitPrice, quantity, lineTotal})  // price snapshot

// reference graph: product catalog (written only by scripts/seed_catalog.sh)
(:Product {productId, name, nameNormalized, category, categoryNormalized, price})
```

**Key properties:**
- `Message.role` — `'user'` | `'assistant'` (fast filter without traversing `POSTED_BY`).
  **Derived server-side from the author's node label** (`User → user`, `Agent → assistant`),
  never trusted from the caller.
- `Message.threadId` — **denormalized, deliberately unindexed** navigation metadata: lets QUERIES
  §9.2/§5 result rows point back to their thread without a traversal; QUERIES §9.1's HEAD/NEXT walk stays the
  canonical thread read. It may be `null` on a message written before the property existed;
  `scripts/backfill_thread_ids.sh` stamps such rows.
- `Message.toolsUsed` — the tools (other than `post_message`) the producing workflow step
  successfully dispatched, stamped
  when the step's `PRODUCED` edge is linked (§6.2). An audit property only: no prompt ever reads it.
- `Thread.updatedAt` — bumped on every new message; drives "recent threads" listing
- `Message.embedding` / `Chunk.embedding` — inline `vecf32`, written out-of-band (§9); no separate
  node
- `Entity.nameNormalized` — case-folded, whitespace-collapsed `name`, computed app-side by the
  shared `extraction.normalize_name` helper. Backs the exact-tier fusion lookup with a real `=`
  comparison, decoupled from RediSearch tokenizer/stemmer behavior (§7.1).
- `Document.currentVersion` / `Chunk.documentCurrent` — the version-lifecycle axis: `true` at
  creation, flipped `false` (with `supersededAt`/`supersededBy` on the `Document`) when a
  `SUPERSEDES` edge to it is confirmed. Default chunk search returns current chunks only (§8).
- `Document.textNormalizedHash` — a hash of the case-folded, whitespace-collapsed full text; the
  exact-identity key of automatic supersede (§9). A content fingerprint, not an identity, so it
  carries no constraint.
- `Document.lshBand0..7` — indexed (§7.1) and read by update detection's candidate shortlist, but
  **no write path sets them**, so that shortlist signal never matches a stored document; only the
  title-fuzzy signal shortlists a candidate today.
- `WorkspaceConfig` — one node per workspace. An absent override property means "no override";
  `''` is never written for "none".
- `Product.productId` is a deterministic slug of the product name, so a re-seed reproduces
  identical ids. `CartItem` and `OrderLine` carry the `productId` as a property key, not an edge:
  the catalog lives in `reference` and edges cannot cross graphs (§2, §4).

### 5.2 Why these choices (traversal cost)

- **Thread-scoped `NEXT` linked list.** Reading a thread is a bounded `NEXT*` walk from
  `Thread HEAD`. Thread stays permanently **sparse** — always exactly 2 edges (HEAD and TAIL)
  regardless of message count. Append is O(1): link new message to current TAIL, move TAIL
  pointer, all in one atomic query.
- **No direct `Channel→Message` edges.** Channel fan-out is bounded by thread count, not message
  count — eliminating the Channel supernode risk entirely. Channel-level time queries use the
  `Message.createdAt` and `Thread.updatedAt` range indexes, not edge traversal.
- **`Message.embedding` inline as `vecf32`** rather than a separate `Embedding` node: a single
  query seeds with `db.idx.vector.queryNodes` then traverses from the seed (thread scope, entity
  co-occurrence) — hybrid retrieval in one round trip (§8).
- **`Message.role` as a property, not derived from `POSTED_BY` label**: lets the app filter by
  role (`WHERE m.role = 'assistant'`) without an extra hop.
- **AI as `Agent` author, not a magic flag**: assistant messages share one timeline with human
  messages; a workflow step's `PRODUCED` edge (§6.2) and an answer's `EMITTED` seed provenance
  (§9, QUERIES §10) make an AI message's origin explicit and auditable.

### 5.3 Thread append (write path)

Two cases — both must be a single `GRAPH.QUERY` (atomic):

- **First message in a thread** (no `HEAD`/`TAIL` yet) — create the message, link
  `Thread -[:HEAD]-> m` and `Thread -[:TAIL]-> m`, and attach `(m)-[:POSTED_BY]->(author)`.
- **Subsequent messages** — match the current `TAIL`, link `prev -[:NEXT]-> m`, move `TAIL` to
  `m` (delete the old `TAIL` edge, create the new one), and attach `(m)-[:POSTED_BY]->(author)`.

Both bump `Thread.updatedAt`. The service picks the variant by checking whether the thread
already has a `HEAD` (`SERVER.md` §1.4 keeps this dispatch inside `post_message`), then re-dispatches on the
**status row** each write returns: a lost first-post race (`hadHead`) retries as subsequent, a
TAIL-less subsequent retries as first, and a replayed `msgId` (`dupMsg`) is idempotent success —
see the §9 table and QUERIES.md §4. The re-dispatch loop is bounded at 4 attempts — a tripwire,
not a real retry budget: ping-pong between the two paths is impossible by contract (a headed
thread always has a TAIL), so hitting the bound means the invariant broke. `createdAt` comes from
the service's lock-guarded monotonic per-process clock (`max(clock, last+1)`), never the raw wall
clock — same-millisecond ties across writes on one process are impossible at the source, which is
what makes the QUERIES §9 `(createdAt, msgId)` composite order well-defined.

> **Canonical Cypher: `docs/QUERIES.md` §4.** The exact, live-verified queries live there and
> nowhere else — this section describes their *shape* only, so the two never drift. Every message
> must carry `(m)-[:POSTED_BY]->(author)`: the canonical thread-read path requires that edge, so a
> message written without it is invisible to thread reads.

### 5.4 Supernode watch

`Channel` (via `HAS_THREAD`) and popular `Entity` nodes are the remaining risks. Channel fan-out
is threads-per-channel — manageable. Entity fan-out runs through `ABOUT` (Chunk→Entity),
`RELATES_TO` and `SAME_AS`. Every mention is a new `Entity`, so `ABOUT`/`RELATES_TO` degree per
entity stays small; the hub is the **oldest** `Entity` of a popular `(nameNormalized, type)`, which
every later exact match links to with a `confirmed` `SAME_AS` (§9), so its inbound `SAME_AS` degree
grows with corpus size. Extraction is capped per **chunk** at 20 entities and 20 relationships (stub
repair can add up to two entities per kept relationship). Re-evaluate with `GRAPH.PROFILE` once real
data lands.

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

> **`Step.stepUid` is the MERGE-backing identity.** A step `key` is unique only *within a def*, so
> it can't back a `MERGE`; every Step carries a synthetic
> `stepUid = "{defKey}:{version}:{stepKey}"` (globally unique within each graph) with an index +
> `UNIQUE` constraint in both `reference` and `ws:{id}` (§7.1/§7.2). `key`/`type`/`config` are the
> display/behaviour props. **`HAS_STEP`** is the def→step containment edge: without it the only
> def→step link is `START`, so reading "all steps of a def" would label-scan every `Step` in the graph
> — and the `stepUid`-prefix `STARTS WITH` alternative live-profiles as a label scan on this build.
> `HAS_STEP` keeps step/transition reads anchored on the def's index (`Node By Index Scan`, verified).
> Canonical publish/materialize/read Cypher: **`QUERIES.md` §11**.

`type` unifies conversational and business flows. **What the engine executes**
(`executor._execute_step`):
- **`agent` → LLM-native node** (`_run_agent_node`, below). **`agent` with no chat LLM wired
  returns an empty stub result** — deliberate, and load-bearing for the offline test estate.
- `human` / `decision` / `wait` → business-process nodes, pure and side-effect-free, proven by
  the `access-request@v1` proof flow (§6.3):
  - **`human`** — parks the run awaiting an assignee. `config.waitsForHuman: true` is **mandatory**
    and enforced at publish; the `StepRun.output` carries an `awaiting` envelope
    (`{kind:'human', prompt, assignee, fields}`), so a client learns what the run is waiting for
    from `GET /workflow-runs/{id}/step-runs` with no new query. **Optionally timer-releasable** —
    see the `wait` bullet below for the shared mechanism, identical for both step types.
  - **`decision`** — **no side effect at all**: its output is `{node:{step}}` and its semantics
    are entirely its outgoing guards. With no outgoing transition it is a terminal outcome node
    (the run ends `done`).
    > ⚠️ **Not enforced (residual, K-029).** A `decision` step whose outgoing transitions are *all*
    > conditional, and which does not declare `waitsForHuman`, self-loops to budget exhaustion if
    > none ever fires — there is no symmetric check forcing either an unconditional default arm or
    > a park declaration. The equivalent check for `human`/`wait` steps *is* enforced (above); doing
    > the same here would retro-reject existing fixtures, so it is a deliberate gap.
  - **`wait`** — **signal-driven by default, optionally timer-releasable as well.** A `wait`/`human`
    step parks awaiting an **external signal** delivered through `POST /workflow-runs/{id}/input` —
    mechanically `wait` *is* `human` to the engine, only the `awaiting` envelope differs
    (`{kind:'signal', signal}`), and both carry the mandatory `waitsForHuman: true`. A step may
    **additionally** declare `config.waitForSeconds` (relative) or `config.waitUntil` (absolute
    epoch-ms); a periodic in-process sweep (`Services.sweep_due_workflow_runs`, also exposed as
    `POST /workflow-runs/due`) then resumes it once due, exactly as an external signal would. A
    step that declares neither key parks forever — valid, specified behaviour. Mechanism and
    residual limits: §6.2; full design: `docs/plans/workflow-timers.md`.
- `prompt` / `message` / `tool`, and any unknown type → **not implemented**: `_execute_step`
  raises `NotImplementedError` naming the typed-handler seam, which fails the run loudly (§6.2's
  fault net) rather than "succeeding" having done nothing.

`Step.config` and `TRANSITION.guard` are **opaque serialized strings** parsed app-side only (rule 8) —
a new step type needs **no DDL**. The `agent` config deserializes to
`{systemPrompt, tools[], model, maxIterations, waitsForHuman, requiredTools[]}` (other keys are
carried, not read by the engine); `waitsForHuman:true` is the explicit suspend signal for a node
that parks awaiting a human reply.

**The `agent` node loop.** One node execution is one bounded tool-calling conversation:
- **Model.** The `step`-kind chat client is resolved **once per node execution** from
  `config.model` (else the kind default; a workspace override caps either — `SERVER.md` §1.8). The
  answering model, its resolution rung and whether a fallback answered are recorded on the
  `StepRun` (`resolvedModel`/`modelSource`/`modelFallback`, last answering call wins). Tools
  dispatched by the node receive the node's own `config.model` as `run.stepModel`, so a tool that
  makes its own `step`-kind call (`query_graph_data`) resolves exactly as the node did.
- **Prompt.** `systemPrompt`, then the last `THREAD_CONTEXT_WINDOW` (20) messages of the run's
  thread as `"{speaker}: {text}"` turns (role from `Message.role`; speaker = display name, else
  author id, else `member` — or, when the executor is built with `human_speaker_label`, that fixed
  label for every non-assistant turn while assistant turns keep their own name; the storefront
  passes `"Customer"`, so a customer never appears under their display name), then a `CONTEXT:`
  turn carrying the run `ctx` as JSON. Consecutive same-role turns are merged, so the list always
  alternates. The list is assembled once per node execution and resent whole on every tool-loop
  `chat` call: the `system` message (present when `systemPrompt` is non-empty) leads every call, while the `CONTEXT` block (merged
  into the last user turn when the thread ends on one) is never re-appended, so each iteration's
  tool-call and tool-result turns land after it. A constraint that must hold on every reply
  belongs in `systemPrompt`, not only in run `ctx`.
- **Tool fence.** Only `config.tools` is offered. A call naming an ungranted tool, or missing a
  required argument, is refused back to the model, never dispatched. A dispatched tool failing with
  a model-correctable `ServiceError` (`UnknownMemberError`, `InvalidSearchQueryError`) is also handed
  back as an error; every other `ServiceError` and every engine fault propagate and fail the run.
  Each refusal costs one iteration.
- **Write-target grounding.** `add_to_cart` and `remove_from_cart` are held (not dispatched; a
  `{held:true, reason}` result goes back to the model) unless their normalized `productId` was
  returned, in this same node execution, by a successful call to a grounding tool —
  `lookup_product_fact`, `filter_products` or `query_graph_data` for an add, those plus
  `view_cart` for a remove. An ID from an earlier turn is never grounded: earlier turns' tool
  results are not in the prompt.
- **Same-turn repeat.** A grounded write whose resolved arguments (normalized `productId` plus the
  `quantity` the tool honours — `add_to_cart` treats an omitted quantity as 1, `remove_from_cart`
  keeps it as "whole line") equal an already-successful call in this node execution is held too.
- **Ending.** A plain-text answer ends the node with that text as `output`. If the node was granted
  `post_message`, posted nothing and answered in text, the engine dispatches `post_message` with
  that text itself. Reaching `maxIterations` (default 4) ends the node with its last text — it
  never fails the run; only `maxSteps` does (§6.2).
- **Signals, never failures.** On every run the engine logs a WARNING (and, on a debug run, writes
  a trace entry) for: a `requiredTools` entry never successfully dispatched (`must_post_violation`);
  a fact-bearing answer (a price-like number) from a node that dispatched none of its granted domain
  tools (`possible_fabrication`); and each held write (`off_turn_write_held`,
  `same_turn_write_held`). None of them changes the node's `output` or the run's path.

`config.requiredTools: [<tool name>, ...]` (`docs/plans/must-post-engine-contract.md`) is the
must-post contract: a subset of the step's own `config.tools` that must be successfully dispatched
at least once before the node's turn ends. Absent/empty ⇒ no obligation; a violation is the
signal above, never a park or a failure.

**Publish-time invariants** (`services._validate_def_spec`; any violation rejects the publish
with nothing written): `kind` ∈ `conversation`|`process`; every step `type` is known; step keys
are unique; exactly one step is marked start; every transition endpoint is a declared step; a
`human`/`wait` step declares `waitsForHuman: true`; `requiredTools` is a list of strings, only on
an `agent` step, and a subset of its `tools`; every `cmp` guard is structurally valid; the def
has at least one transition. Timer keys: only a `human`/`wait` step may declare
`config.waitForSeconds` or `config.waitUntil`, and at most one of the two; `waitForSeconds` is a
positive number of at most 15 552 000 (180 days); `waitUntil` is a positive epoch-ms number; and a
step declaring either has an outgoing transition whose guard is
`{kind:'cmp', path:'ctx.timerFired', op:'eq', value:'<that step's own key>'}` (extra fields
allowed; the marker is §6.2's). A step or `llm` guard naming a model or role that does not
resolve is rejected at publish as well (`SERVER.md` §1.8). Only `cmp`-family guards are validated
at publish: a guard of any other or missing `kind`, or a string that is not a JSON object,
publishes and then fails the run when a drive evaluates it.

**`TRANSITION.guard` is LLM-native *and* deterministic.** The guard string is one of:
- **empty `""`** — an unconditional/default transition (**lowest** priority; fires whenever reached);
- a `{kind:'llm', text, model?}` discriminator — judged in natural language against run context (a
  structured boolean verdict + traced rationale); an optional `model` is resolved as the `guard`
  kind;
- a **`cmp`-family** discriminator — `{kind:'cmp', path, op, value}` plus the combinators
  `all` / `any` / `not`. A closed comparator, **not** an expression language: whitelisted ops, two
  whitelisted path roots (`ctx.` / `output.`), depth/width/node caps, no parser and no `eval`.
  It is **total at drive** (a missing path ⇒ `False`, never a raised error) and **strict at
  publish** (a typo'd `op` or an unwhitelisted path root is a `WorkflowConfigError` at seed time, not
  a run that parks forever). This is what makes an LLM-free `kind:'process'` flow possible.

A would-be **`expr`** kind is a deliberate `NotImplementedError` seam — no expression library is
built here (the deterministic family is deliberately named `cmp` to keep that literally true).

**Two guard facts that read counter-intuitively:**
1. **`on` is descriptive only.** `TRANSITION.on` and `StepResult.on` are **vestigial** — nothing in
   the engine reads either. `on` is a human-readable outcome label, *not* "the event that fires the
   transition"; the guard alone decides. This is what lets
   `docs/plans/must-post-engine-contract.md` treat a must-post violation as trace-and-continue
   rather than a new `StepResult.on` outcome — a design that changed `on` conditionally on violation
   would be inventing meaning for a field the engine has never read.
2. **Guards are not evaluated in plain `TRANSITION.order`.** The sort key is
   `(guard == "", order, to)` — i.e. **conditional guards first**, with `order` (then the target
   step key) as a tie-break *within* each class, then first-firing wins. That is what makes "conditional beats unconditional" work and
   an empty guard a true default arm.

One model, both worlds. Runtime evaluation & the run/step-run schema: §6.2; verified Cypher:
**`QUERIES.md` §12**.

### 6.2 Run (per-workspace, real local edges to the materialized def)

```
(:WorkflowRun {runId, defKey, defVersion, status, startedAt, endedAt,
               ctx, trace, maxSteps, stepCount, waitingThreadId})   // status ∈ running|waiting|done|failed
(:WorkflowRun)-[:OF_DEF]->(:WorkflowDefSnapshot {key, version})   // local, materialized
(:WorkflowRun)-[:AT_STEP]->(:Step)                               // current position (cleared on terminal)
(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)                       // the @mention that started it (absent on a REST-started run)
(:WorkflowRun)-[:HAS_STEP_RUN]->(:StepRun)                       // membership
(:WorkflowRun)-[:LAST_STEP_RUN]->(:StepRun)                      // TAIL pointer → the NEXT anchor
(:StepRun {stepRunId, stepKey, status, startedAt, endedAt, input, output,
           resolvedModel, modelSource, modelFallback})
(:StepRun)-[:RAN]->(:Step)                                       // which def step
(:StepRun)-[:NEXT]->(:StepRun)                                   // execution order (audit trail)
(:StepRun)-[:PRODUCED]->(:Message)                               // step-emitted chat message
(:StepRun)-[:TRACED]->(:TraceEvent)                              // debug-only trace record
(:TraceEvent {traceId, seq, kind, at, payload})                  // debug runs only; payload = flat string
```

> `ctx` (on `WorkflowRun`), `input`/`output` (on `StepRun`) and `payload` (on `TraceEvent`) are **flat,
> serialised strings**, not nested maps — FalkorDB stores only scalars and scalar lists. Queries never
> filter *inside* them (see §1.2). The executor (de)serializes app-side.

**Run-model notes:**
- **`trace`/`maxSteps`/`stepCount`** on `WorkflowRun` — the debug-instance flag (gates all trace
  writes), the run-level step budget (DS default 12 — a **tripwire**, see the note below), and
  the executed-step counter the atomic advance bumps. **`waitingThreadId`** denorms the parked run's
  thread so resume is an index-anchored lookup (rides the existing `status` index — no new index;
  QUERIES §12.9). `endedAt` stamps terminal.
- **`LAST_STEP_RUN` — the tail pointer.** Mirrors the locked `Thread` HEAD/TAIL pattern (§5.2):
  `record_step_and_advance` reads it to find the previous `StepRun`, hangs `NEXT`, and moves the tail —
  all in one query → **O(1) atomic advance, no chain-walk / label scan**. One edge per run (the tail
  moves, it does not accumulate).
- **`PRODUCED`, not `EMITTED`.** StepRun→Message emission is a **distinct** edge type —
  `EMITTED` is the **Message→Message** provenance edge (§9/QUERIES §10); overloading it
  would conflate "cited that seed" with "produced that message."
- **`TraceEvent` + `TRACED` — debug-only.** A debug run writes one `TraceEvent` per LLM
  prompt/response, tool call/result, guard judgment, and retrieval (dozens per run); a non-debug run
  writes **zero**. `TraceEvent.traceId` is indexed **then** UNIQUE-constrained (§7.1).
- **`stepRunId`** is the `StepRun`'s stable identity (indexed + UNIQUE, §7.1).
- **`resolvedModel`/`modelSource`/`modelFallback`** on `StepRun` record which model answered an
  `agent` node and how it was resolved (§6.1); `null` for non-LLM steps. The workspace's
  `WorkspaceConfig` overrides are read once per drive and apply to every resolution in it.
- **Snapshot/`Step` deletion blast radius** (live-verified). A
  `WorkflowDefSnapshot`+`Step` subgraph can be `DETACH DELETE`d (e.g. to force a
  re-materialize) without corrupting a live/completed `WorkflowRun`'s executed-step
  history: `stepKey` is copied onto `StepRun` at write time (`record_step_and_advance`,
  `repository.py`) rather than read live from `Step`, and `HAS_STEP_RUN`/`LAST_STEP_RUN`/
  `NEXT`/`PRODUCED` never touch `Step`. The one edge that **is** severed by deleting the
  `Step` nodes above is `RAN` (`StepRun`→`Step`, this section) — a write-only
  pointer (created at advance time, not read by any shipped query), so the practical blast
  radius is "live position (`AT_STEP`) + `OF_DEF` back-reference + the `RAN` pointer,"
  never the audit trail's own readable data. A query that starts traversing `RAN` voids this
  note.

**Who writes the run `ctx`.** Every write of `WorkflowRun.ctx` that carries human or
external input rides **inside the resume CAS itself** — `resume_run_with_ctx` (QUERIES §12.13) is
`resume_run` (QUERIES §12.4) plus one `SET` term, one query. There is deliberately **no** standalone
"set the ctx" write. With a split write, submitter B's ctx can land between A's ctx write and A's
CAS, so the drive that runs is A's while the data it reads is B's — a **silent wrong branch**, and
worse, a stale submitter could erase a key an earlier step already branched on, leaving a run whose
own `ctx` no longer explains its own trail. Folded, only the CAS **winner's** ctx is ever written:
"which input advanced the run" and "which input is in `ctx`" can never disagree, which is the audit
property §6.3 exists to prove. A loser gets a visible 409 with **nothing written** — neither the
status flip nor the ctx — never a silent loss.

> **Residual window, stated plainly (R-1).** The *read* before the merge is
> non-transactional: two submitters can both read the same base ctx, merge onto it, and the
> winner's merge may omit a key the loser intended to add. That is a lost update **on an unwritten
> input, reported as a 409 to its submitter** — not a wrong branch and not an erased key a prior
> step branched on. Acceptable for single-approver use; the fix (a `ctxVersion` counter CAS) is
> unbuilt.

**Timer release is one more `resume_run_with_ctx` submitter, not a new write path.**
`Services.sweep_due_workflow_runs` resumes a genuinely-due `wait`/`human` run through the same CAS,
merging an engine-reserved `ctx.timerFired = "<parked step's own key>"` marker (in
`RESERVED_CTX_KEYS` — no human/API caller can set or spoof it) so the run's own
publish-time-required escalation guard fires deterministically on that resume, and only then. It
inherits R-1's exact shape (one more racing submitter, same single-winner CAS) and adds no new
residual class. The sweep writes no new `WorkflowRun` property (there is deliberately no `wakeAt`) and keeps
no scheduler state of its own: dueness is derived fresh, every call, from `StepRun.startedAt` (via the `LAST_STEP_RUN` tail
pointer above) and `Step.config`. Full argument: `docs/plans/workflow-timers.md` §8.

> **Design rule for any automated resume caller** (the sweep is the first). Such a caller
> carries two independent risk classes: a **CAS race** on concurrent resumers (closed generically
> by the ctx-write-inside-CAS discipline above) and a **re-park loop** — the automated caller
> landing on a step whose only advance path is a guard the automation itself cannot satisfy. The
> second class has no generic mitigation; each new automated caller needs its own
> guard-satisfiability argument. An unconditional fallback transition is not one: it fires on
> every arrival, not only on an automated resume, so the step would never park. The timer sweep's
> argument is the `ctx.timerFired` marker guard above, which only the sweep can satisfy.

The engine loop: read `AT_STEP` → execute the step (LLM-native agent loop or deterministic handler) →
evaluate outgoing `TRANSITION` guards against `ctx` (first-firing wins) → `record_step_and_advance`
(create the `StepRun`, append `NEXT` via the tail, move `AT_STEP`) — **or** suspend to `waiting` if the
step declares `waitsForHuman` and no guard fired, **or** re-run the same step if no guard fired, it
has outgoing transitions and it declares no `waitsForHuman` (any step type; each re-run records a
`StepRun` and counts against `maxSteps`), **or** terminate (`done` when the step has no outgoing
transition, `failed` on a fault or an exhausted budget). Runaway loops are
bounded by `maxSteps` (run budget) + per-node `maxIterations` (§6.1). The whole walk is local to the
workspace graph (fast, isolated, fully auditable). **Fault net (`_drive`):** a `HumanHandoffSignal`
raised through a tool suspends the run to `waiting`; any other exception fails the run
(`fail_run`, `AT_STEP` cleared, a diagnostic note in `ctx`) and re-raises, so a drive fault never
leaves a `running` orphan. **Verified Cypher: `QUERIES.md` §12** — this section
is the *why*, not a query copy.

> **`_drive_loop` is byte-locked.** Its body is not edited without deliberately reopening the
> lock; `_drive`, `_execute_step`, `_select_transition`, `_trace_step`, `_link_emissions` and
> `resume` sit **outside** it and may change freely. The lock value at the pinned tree is
> **`6cd35d650b5d`**, computed line-number-independently by bounding the extraction on the
> `def`/seam-comment markers:
> `awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12`.
> Before extending the lock for a new requirement, check whether the needed data is already
> written atomically elsewhere in the run's history — the timer sweep derives dueness from
> `StepRun.startedAt` instead of writing a `WorkflowRun.wakeAt` inside the lock.

> **What `maxSteps` actually means.** `maxSteps` is a **runaway tripwire checked *after* each
> recorded step**, not a hard cap: a run executes at most **`maxSteps + 1`** steps before failing
> with `"step budget exceeded"`. The check (`rec["stepCount"] > max_steps`) runs only on the two
> driving outcomes in `_drive_loop` — a guard fired (OUTCOME A) and a legitimate self-loop
> (OUTCOME C). It is **deliberately not applied on the park path** (OUTCOME B — a parked run
> cannot self-drive) **or on the terminal path**. Treat it as a safety bound, not an SLA or a cost
> budget. Making it an exact cap (`>` → `>=`) lands inside the locked `_drive_loop` and is open as
> **K-033**.

> **`status` as a property, not a label**, so a run's state changes in place without
> re-labeling churn; index it for "all running workflows" and the `waiting`-run resume lookup (QUERIES §12.9).
> Suspend/resume are guarded single-query CAS flips (`running↔waiting`) so concurrent replies can't
> double-resume (QUERIES §12.3/§12.4).

### 6.3 Coordination is workflow, not a separate primitive

Agent/team coordination (task lifecycle, "room state") is modelled as a `WorkflowDef` of
`kind:'process'` over `Step` + `TRANSITION` + `StepRun` — **not** a flat `Task` node or a
presence field. This avoids a parallel model that would later need migrating into the engine
(single-store philosophy). Full rationale/ADR: `docs/archive/plans/m1-chat-mcp.md` Appendix B.

**The proof exists.** `access-request@v1` — an LLM-free `kind:'process'` def of six steps and six
transitions over `human` / `decision` / `wait` — runs end to end with **no LLM and no network**:

| Artifact | Where |
|---|---|
| The def (the single source both seed and test read) | `server/falkorchat/proof_defs.py` — `ACCESS_REQUEST_DEF` |
| Offline acceptance test (all three paths of `m3-process-flow.md` §4.3) | `server/tests/test_process_flow.py` |
| Seeding into `reference` + `ws:{id}` | `scripts/seed_workflows.sh` (second def) |
| Design + traced paths | `docs/archive/plans/m3-process-flow.md` |

The design claim it settles: **a business process needs no new primitive, no new run state and no
scheduler.** A `human` step is just a step whose outgoing guard reads a `ctx` key that does not exist
yet — the executor's existing "no transition fired" outcome parks it; writing the key
(`POST /workflow-runs/{id}/input`) makes the same guard fire on resume. The executor's drive loop has
**no process-specific code**.

---

## 7. Indexes, constraints & vector search

### 7.1 Per workspace graph `ws:{id}`

> **Executable DDL is `scripts/bootstrap_schema.sh` — the single source of truth**, asserted by
> `test_queries.sh`. This section describes *what* is indexed/constrained and *why*, not
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
composite-keyed `WorkflowDefSnapshot` and `CartItem`:

| Label | Indexed property(ies) | Constraint |
|---|---|---|
| `User` | `userId` | UNIQUE 1 |
| `Agent` | `agentId` | UNIQUE 1 |
| `Channel` | `channelId` | UNIQUE 1 |
| `Thread` | `threadId` | UNIQUE 1 |
| `Message` | `msgId` | UNIQUE 1 |
| `Document` | `documentId` | UNIQUE 1 |
| `DocumentDeletion` | `documentId` | UNIQUE 1 (one audit node per deleted document) |
| `Chunk` | `chunkId` | UNIQUE 1 |
| `Entity` | `entityId` | UNIQUE 1 |
| `WorkflowRun` | `runId` | UNIQUE 1 |
| `StepRun` | `stepRunId` | UNIQUE 1 |
| `TraceEvent` | `traceId` | UNIQUE 1 (debug-only nodes) |
| `ReadCursor` | `cursorId` | UNIQUE 1 |
| `WorkflowDefSnapshot` | `key`, `version` (two indexes) | UNIQUE 2 (composite) |
| `Step` | `key`, `stepUid` (two indexes) | UNIQUE 1 (`stepUid`); `key` index-only (§6.1) |
| `WorkspaceConfig` | `workspaceConfigId` | UNIQUE 1 (backs the singleton's `MERGE`) |
| `Customer` | `customerId` | UNIQUE 1 |
| `Cart` | `customerId` | UNIQUE 1 (one cart per customer) |
| `CartItem` | `customerId`, `productId` (two indexes) | UNIQUE 2 (composite — one line per product per cart) |
| `Order` | `orderId` | UNIQUE 1 |

`OrderLine` has no index or constraint: it is only ever reached from its `Order`.

**Relationship-scoped indexes** — same index-before-constraint ordering rule, on an edge property
rather than a node property:

| Relationship | Indexed property | Constraint |
|---|---|---|
| `SAME_AS` | `matchId` | UNIQUE 1 |
| `SAME_AS` | `status` | — (hot-filter, no constraint) |
| `SUPERSEDES` | `matchId` | UNIQUE 1 |
| `SUPERSEDES` | `status` | — (hot-filter, no constraint) |

> Every `SAME_AS`- or `SUPERSEDES`-anchored query matches its endpoints **unlabeled** (`(a)`/`(b)`, never
> `(a:Entity)`) — live-verified: a bare label on either endpoint forces a full `Node By Label Scan`
> even though the relationship-property scan alone is fully selective
> (`docs/plans/document-ingestion-graph.md` §1.4; `claude/graph-dba/falkordb-quirks.md`).

**Hot-filter indexes (no constraint)** — support scans/ordering, not identity:

| Label | Property | Serves |
|---|---|---|
| `Thread` | `updatedAt` | recent-threads listing |
| `Message` | `createdAt` | time-range / keyset reads (QUERIES §9) |
| `WorkflowRun` | `status` | "all running workflows"; the parked-run resume lookup |
| `WorkflowRun` | `startedAt` | runs-for-thread listing (index-backed only with the query's `r.startedAt >= 0` predicate, QUERIES §12.14) |
| `StepRun` | `status` | step-state filters |
| `Entity` | `nameNormalized` | exact-tier fusion `=` lookup; distinct real entities can share `(nameNormalized, type)` before fusion runs |
| `Document` | `currentVersion` | current-version filters (many documents share `true`) |
| `Document` | `textNormalizedHash` | automatic-supersede exact lookup (a fingerprint, so no constraint) |
| `Document` | `lshBand0` … `lshBand7` | update-detection shortlist (an 8-way `OR`, one `Node By Index Scan`); no write path sets these today (§5.1) |

> `Message.threadId` is **deliberately unindexed** (§5.1) — nav metadata, not an anchor.

> `Channel.participantId` is **deliberately unindexed and deliberately un-constrained** (§5.1) —
> the storefront provenance marker, and a load-bearing predicate in a *destructive* query
> (`QUERIES.md` §18.4/§18.5). **No index:** it is only ever a filter on a `ch` already bound by a
> `MEMBER_OF` traversal from an index-anchored `u`, so there is nothing for an index to anchor.
> **No UNIQUE:** rejected **on scope, not on safety** — a `UNIQUE 1` here was built and
> live-verified to work (FalkorDB exempts both an absent property and an explicit `null`, so no
> legacy channel would need backfilling), but `ensure_participant` already prevents a second
> channel per participant, §18.4's row-multiplication makes a duplicate marker fail loudly anyway,
> and adding it costs a `bootstrap_schema.sh` change plus a DDL pass over every existing workspace
> for no measured gain. **Reversal condition:** add it if a second writer of the marker is ever
> introduced, or if §18.4's raise stops being an acceptable failure mode
> (`docs/plans/salesperson-ui-graph.md` §9).

**Full-text index (RediSearch):** `Message.text`, `Entity.name`, `Document.title`, via
`db.idx.fulltext.createNodeIndex('Message', 'text')` /
`db.idx.fulltext.createNodeIndex('Entity', 'name')` /
`db.idx.fulltext.createNodeIndex('Document', 'title')` — backing message keyword search, the
suggested-tier entity-fusion lookup, and update detection's title-fuzzy shortlist respectively.
`Document.text` deliberately has no full-text index: at the 500,000-character document ceiling it
would cost several times the raw text in RAM.

**Vector indexes:** `Message.embedding` and `Chunk.embedding`, created via **DDL**
(`CREATE VECTOR INDEX … OPTIONS {dimension, similarityFunction:'cosine'}`).
- ⚠️ `db.idx.vector.createNodeIndex` is **not** a registered procedure on this build (live-verified)
  — the DDL form is mandatory (§1.2).
- Dimension **must** match the embedding model exactly (`EMBEDDING_DIM`; `1024` for this
  system's model, §1.3 — the script's own default of `1536` is wrong for every workspace here) and
  is fixed per workspace at bootstrap. A wrong-length `vecf32` write is accepted silently and the
  node then drops out of ANN results, so the embedding worker checks the vector's length against
  the index dimension before writing.
- Vectors stored as `vecf32`; **score is cosine distance** (`0` = identical, lower = more similar).
  Write `SET n.embedding = vecf32([...])`; read `CALL db.idx.vector.queryNodes('Message','embedding',
  $k, vecf32($vec)) YIELD node, score` → `ORDER BY score ASC`. Canonical read: `QUERIES.md` §6/§8.
- Vector indexes are usually the biggest per-workspace RAM line (`dim × 4 bytes × #vectors`, §10/§11).

### 7.2 `reference` graph

Same ordering rule (index first, constraint second); executable DDL is the same
`bootstrap_schema.sh`, which always bootstraps `reference` (DDL only, never data) whatever
workspaces it is given.

| Label | Indexed property(ies) | Constraint |
|---|---|---|
| `WorkflowDef` | `key`, `version` (two indexes) | UNIQUE 2 (composite) |
| `Entity` | `entityId` | UNIQUE 1 |
| `Step` | `key`, `stepUid` (two indexes) | UNIQUE 1 (`stepUid`) — the MERGE identity (§6.1); `key` index-only (display/traversal anchor) |
| `Product` | `productId` | UNIQUE 1 — backs the catalog seed's `MERGE` |
| `Product` | `nameNormalized`, `categoryNormalized`, `price` | — (exact-name lookup, case-insensitive category equality, price-range scan anchor) |

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
traversal — read-only queries, routable to a replica.

> **Canonical Cypher: `docs/QUERIES.md` §6 (messages) and §14.3 (chunks).** This section describes
> the read paths' *shape* only, so the two never drift.

- **Message hybrid read** — an ANN seed over `Message.embedding`, then traversal: the seed's
  thread (`HEAD|NEXT*`), optionally scoped to one channel, plus an entity co-occurrence expansion
  (`MENTIONS` → `Entity` ← `MENTIONS`) that returns nothing today because no write path creates
  `MENTIONS` (§5.1).
- **Chunk read** — an ANN seed over `Chunk.embedding` with no scope traversal, filtered after the
  ANN step to `documentCurrent = true` (a superseded version's chunks never surface by default).
  The filter runs on the `k` ANN rows, so a caller wanting `limit` results over-fetches via `k`.
- **Vector** finds *what's semantically relevant*; **traversal** pulls *precise, explainable
  neighbors*. An AI answer records its seeds as `EMITTED {score, rank}` edges to the `Message` or
  `Chunk` it used (§5.1).
- **Score is cosine distance** (live-verified: identical vectors → score `0`). Order `ASC` to
  rank most-similar first. ANN is approximate: a read may return fewer than `k` rows.
- Served via `GRAPH.RO_QUERY` with the service-layer `RAG_QUERY_TIMEOUT_MS` (5000 ms) override
  (§10) → can hit read replicas (mind replica lag for just-posted messages; route "include my
  last message" reads to the primary).

---

## 9. Write paths

| Operation | Pattern | Notes |
|---|---|---|
| Post message | Guarded `CREATE` inside `FOREACH`+`CASE` per path + relink `Thread TAIL → NEXT` (QUERIES.md §4) | Two separate self-guarding variants (first vs subsequent — see §5.3), never a conditional MERGE of the two paths. `MENTIONS_MEMBER` edges (and, for an AI answer, `EMITTED` seeds) ride the same query. Always returns a **status row**; retry-idempotency comes from that `dupMsg` status, **not** from the uniqueness constraint — a replayed MERGE re-runs the relink clauses and corrupts the chain. The `Message.msgId` uniqueness constraint stays as the concurrency backstop (rollback verified all-or-nothing). O(1) append. |
| Create channel / thread | plain `CREATE` (server-minted uuid ids) | **Non-idempotent** — a retried create mints a new id; the uniqueness constraints backstop. A MERGE on a fresh uuid could never match. |
| Ensure member / storefront participant | guarded `CREATE` with a status row | Member ids are unique across `User`/`Agent`; a participant is created with its own `Channel` (`participantId` set), `Thread` and both `MEMBER_OF` edges in one query. |
| Advance read cursor | `MERGE` on `HAS_CURSOR`→`ReadCursor {cursorId}` + composite monotonic guard | Never moves backward on a stale or tie-smaller replay. |
| Backfill / import | `UNWIND $rows AS row …` in chunks, or `falkordb-py` bulk loader | Never one giant CREATE — bound transaction memory; size batches (writes ignore TIMEOUT — §10) |
| Embed message / chunk | background job: compute embedding → `SET n.embedding = vecf32($v)` | Decoupled from the post/ingest path; a failure is logged, never surfaced to the caller. |
| Ingest document | one guarded `CREATE` of `Document` + `INGESTED_BY` + every `Chunk`/`HAS_CHUNK` (chunks split app-side) | Non-idempotent like channel creation. If a current `Document` has the same `textNormalizedHash`, the same query supersedes it (`SUPERSEDES` `confirmed`, technique `exact_normalized_text_hash`). |
| Document background jobs | per chunk, one embed job and one extract job, scheduled independently | `pendingJobs` is set to the job count before any job starts; each job decrements it once with its **final** outcome. `status` becomes `failed` at the first failed job, `ready` when the count reaches 0 with none failed; both flips guard on `processing`, so the first terminal write wins. A job retries a transient provider error (connection kinds; HTTP 408, 429, 5xx; known LM Studio error text) up to 4 attempts with 2 s / 8 s / 30 s nominal back-off, and stops retrying once its document is no longer `processing`. |
| Update detection | background job after a non-auto-superseded ingest | Shortlists current documents (LSH bands, title full-text fuzzy), scores each by shingled Jaccard in Python, and records a `pending` `SUPERSEDES` suggestion for the best candidate at or above 0.1. A rejected suggestion that recurs is reopened (`resuggestCount` +1), never duplicated. |
| Confirm / reject a supersede | `SET` on the edge by `matchId` | Confirming sets the old `Document`'s `currentVersion` and its chunks' `documentCurrent` to `false` in the same query. |
| Delete document | `DETACH DELETE` of the `Document` and its chunks + `CREATE` of a `DocumentDeletion` audit node, one query | A missing document is a no-op with no audit node. |
| Extract entities | plain `CREATE` per extracted `Entity`, `ABOUT`, `RELATES_TO` | Every mention becomes a new `Entity`; fusion only links. An exact `(nameNormalized, type)` match is linked `SAME_AS` `confirmed` in the creating query; otherwise a full-text fuzzy candidate yields a `pending` `SAME_AS` (reopened, never duplicated, like supersede suggestions). Entities are never merged. |
| Advance workflow | create `StepRun`, append `NEXT`, move `AT_STEP` | All local to `ws:{id}`; fully transactional within the graph (§6.2) |
| Publish workflow def | write to `reference` only | Topology-immutable per version (a differing re-publish is rejected `409`); properties stay create-only. Bump version to change either. |
| Materialize workflow def | copy one published version into one `ws:{id}` as a `WorkflowDefSnapshot` (explicit call, §4) | Same topology `409` against the existing snapshot; an unpublished version is `404`. A run can start only against a materialized version. |
| Write model overrides | constraint-backed `MERGE` of the `WorkspaceConfig` singleton + `SET` of all four overrides | A `null` clears an override. |
| Cart writes | constraint-backed `MERGE` of `Customer`, `Cart`, `CartItem`; quantity increment/adjust in place | A line adjusted to ≤ 0 is deleted. |
| Place order | guarded `CREATE` of `Order` (+ `OrderLine` price snapshots) on a caller-minted `orderId` | `orderId` is the idempotency key; the cart's items are deleted in the same query. |
| Order status | guarded `SET` | `placed → fulfilled → delivered`; `placed → cancelled`; any other transition is a no-op. |
| Customer profile | constraint-backed `MERGE` + per-field `coalesce` `SET` | A partial update never clears an omitted field. |
| Reset storefront participant | one query deleting the participant channel's threads, messages, runs, step-runs, traces, cursors, the participant's `Customer` (saved profile: `name`, `deliveryAddress`), cart, cart items, orders and `OrderLine`s, then a fresh `Thread` that `User.threadId` is re-pointed to | Scoped by `ch.participantId = u.userId` on a `User` with a `tokenHash`; anything else is unreachable. |

**Rule:** every `MERGE` is backed by a uniqueness constraint, or it's a duplicate-node bug
waiting for concurrency. (The QUERIES §4 message writes contain no MERGE at all — guarded CREATE
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
- **TIMEOUT posture (live-probed).** Keep the legacy single-knob `TIMEOUT=1000` as the
  deployment default — right for chat CRUD and verified to fire (enforcement is
  batch-granular; slightly-over reads can slip through). GraphRAG reads (§8) pass a
  **per-query client override** (`g.ro_query(q, params=…, timeout=…)`; pass-through verified,
  uncapped while `TIMEOUT_MAX=0`), held as one service-layer constant (`RAG_QUERY_TIMEOUT_MS`,
  5000 ms) rather than chosen per call.
  **Writes ignore TIMEOUT entirely on this build** — a write runs to completion regardless of
  clause or default; bounded batches (≤ a few hundred `UNWIND` rows) and the existing API
  input caps are the only write-path protection. If ops later wants a hard ceiling on client
  overrides, switch to `TIMEOUT_DEFAULT`/`TIMEOUT_MAX` (>0) — mutually exclusive with the
  legacy `TIMEOUT` knob; change deliberately, in one step. One unreproduced anomaly is on
  record: an instant timeout right after a long override run; re-check on upgrades.
- **Observability:** `GRAPH.SLOWLOG` for slow queries, `GRAPH.PROFILE` for plans, Redis metrics.
- **Security:** Redis ACLs scoping `GRAPH.*` per principal (ideally per workspace key pattern),
  TLS in transit, network isolation; secrets outside the data.

---

## 11. Capacity — the numbers that shape the design

> **Full measurements: `docs/test-reports/capacity-report.md`** — the per-message RAM breakdown,
> the append-path load test, the hot-read `GRAPH.PROFILE` results, and the per-workspace RAM
> budget + shard-packing tables. This section carries only what the *design* turns on.

Three numbers, all measured live on this deployment:

| Number | Value | What it constrains |
|---|---|---|
| Per-message RAM at 1024 dims | **~12.4 KB** → ~1.25 GB per 100k-message workspace | Per-workspace sizing; **~85% of it is vector + HNSW, not chat** — the chat-core floor alone is ~1.06 KB/message |
| Sustained append throughput | **~614 msg/s**, p99 40.7 ms (16 clients, one graph) | The ceiling is **per-graph-key write serialisation**, so per-thread fan-out does not raise it |
| Vector-index dimension | Fixed **at workspace creation**, never after | `EMBEDDING_DIM` must match the embedding model *before* bootstrap (§7.1) |

Two consequences the rest of this document depends on:

- **Size workspaces from `INFO memory` deltas, never `GRAPH.MEMORY USAGE`.** On this build the
  latter under-reports vector-index memory — it reported `indices_sz_mb: 0` while the HNSW index
  demonstrably held 4,096 vectors. Upstream filing recommended.
- **The dim choice is the biggest lever available.** Cutting 1536 → 1024 saves roughly a third of
  per-message RAM; evicting cold embeddings is the next-cheapest lever, since ~10 KB of the
  12.4 KB is vector + index (retention is an open question, §13).

---

## 12. Roadmap

Open work and milestones: `falkor-chat/docs/BACKLOG.md`; delivered milestones: `falkor-chat/docs/HISTORY.md`.

---

## 13. Open questions

**Resolved — these stand as design:**
- **Workflow guard language** — guards are the empty-string default form, `{kind:'llm', text}`, or
  the closed `cmp` comparator family (§6.1). No expression library, no parser, no `eval`;
  `kind:'expr'` raises `NotImplementedError`.
- **App driver** — the server reaches FalkorDB over RESP via `falkordb-py` (§10).
- **TIMEOUT** — the legacy `TIMEOUT=1000` single knob stays the deployment default, with a
  per-query override for GraphRAG reads (§10).

**Open** — tracked in `falkor-chat/docs/BACKLOG.md`: message/embedding retention under the
always-in-RAM constraint (§11), cross-workspace analytics (app-layer fan-out vs. an `analytics`
rollup graph, §3), the real-time push gateway's transport (Bolt vs. RESP/WebSocket, **K-018**),
and a pre-production review of the live `GRAPH.CONFIG` defaults (`THREAD_COUNT 4`,
`OMP_THREAD_COUNT 4`, `CACHE_SIZE 25`, `MAX_QUEUED_QUERIES 25`, `QUERY_MEM_CAPACITY 0`
(unlimited), `ASYNC_DELETE 1`).

---

## 14. The server application → `SERVER.md`

The server process — internal layering, the auth/tenancy seam, the REST and MCP front doors, the
`server/` layout, the `server/` testing hazards, and the model-resolution seam — is
**`falkor-chat/docs/SERVER.md`**. This document stops at the graph and the workflow engine.

Citations of `DESIGN §14.x`/`§15.x` (in source docstrings and elsewhere) resolve by this map:

| Cited as | Section |
|---|---|
| `DESIGN §14` — application architecture | `SERVER.md` §1 |
| `DESIGN §14.1` Scope decisions | `SERVER.md` §1.1 |
| `DESIGN §14.2` Layering | `SERVER.md` §1.2 |
| `DESIGN §14.3` Auth/tenancy seam | `SERVER.md` §1.3 |
| `DESIGN §14.4` REST surface | `SERVER.md` §1.4 |
| `DESIGN §14.5` Layout | `SERVER.md` §1.5 |
| `DESIGN §14.6` TDD build order | `SERVER.md` §1.6 |
| `DESIGN §14.7` Testing hazards | `SERVER.md` §1.7 |
| `DESIGN §14.8` Model-resolution seam | `SERVER.md` §1.8 |
| `DESIGN §15` — MCP transport | `SERVER.md` §2 |
| `DESIGN §15.1` Shape | `SERVER.md` §2.1 |
| `DESIGN §15.2` Tools → service → query | `SERVER.md` §2.2 |
| `DESIGN §15.3` Client connection contract | `SERVER.md` §2.3 |
