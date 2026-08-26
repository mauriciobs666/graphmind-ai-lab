# falkor-chat — Design Document & Blueprint

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.0

> **Philosophy:** FalkorDB graph for *everything* — reference data, workflow definitions
> and runs, chat history, and user/workspace information. No second store for the primary
> domain. One engine, one query language (OpenCypher), one operational model (Redis).

> **How to read this document.** It states the system **as it is now**. *When* something changed
> and *which* backlog item delivered it live in `falkor-chat/docs/HISTORY.md`; what is proposed
> but unbuilt lives in `falkor-chat/docs/BACKLOG.md`. A `K-` number appears here **only** when it
> points at work that is not yet done — a delivered item's number is history, not design.
>
> *Revision note — 2026-08-24 (v0.3 → v1.0).* Whole document restated in present tense:
> delivered-ticket annotations, superseded-wording notes and duplicated handoff notes removed, no
> technical content dropped. §14–§15 (the server application + MCP transport) moved out to
> `SERVER.md`, leaving §14 as the redirect table; §11's measurements moved to
> `docs/test-reports/capacity-report.md`, leaving the three numbers the design turns on. Every
> `K-` number that survives points at **open** work (K-016/K-017/K-018, K-029, K-033). Two stale
> claims corrected in passing: §1.3 was headed "pending implementation" for a stack that shipped
> with M2, and §2's supernode note still prescribed a "time-bucket pattern" that §1.2 records as
> a rejected alternative.

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
| **`Message.role` inline property; values `user`/`assistant` derived server-side** from the author label (`User→user`, `Agent→assistant`), never trusted from the caller | Filter by role without traversing `POSTED_BY`; agents author first-class | §5.1, §5.2 |
| **`coalesce(u.userId, a.agentId)` for member identity** (two indexed `OPTIONAL MATCH` + `coalesce`) | `User` has `userId`, `Agent` has `agentId` — both are members; anchored lookup avoids the `OR`-scan | QUERIES §2 |
| **Vector indexes via DDL**, not a procedure | `db.idx.vector.createNodeIndex` is not registered on this build | §2, §7.1 |
| **Index before constraint, always** | `GRAPH.CONSTRAINT CREATE` requires a pre-existing range index | §2, §7.1 |
| **`Message.embedding` inline as `vecf32`** | Single-query vector + traversal hybrid retrieval | §5.2, §7.3 |
| **Vector score is cosine *distance*** (0 = identical) → `ORDER BY score ASC` | Most-similar-first ranking | §7.1, §8 |
| **`status` as a property, not a label** | Avoids re-labeling churn on state changes; index it for "all running" reads | §6.2 |
| **`ctx` / `input` / `output` are flat/serialised strings** | FalkorDB stores scalars + scalar lists only — no nested maps; never query inside them | §6.2 |
| **`Message.threadId` denormalized inline, unindexed** | Nav metadata for §9.2/§5 rows; HEAD/NEXT walk stays canonical; unindexed saves RAM/write cost | §5.1 |
| **Guarded-CREATE write paths** (`FOREACH`+`CASE` per path) with an always-returned status row; **no MERGE on `Message`** | Retry replay is a no-op (`dupMsg`); first-post race refused (`hadHead`); uniqueness constraint is the backstop | §5.3, §9 |
| **Composite `(createdAt, msgId)` keyset cursor** (`ReadCursor.lastReadAt`/`lastReadMsgId`) | Timestamp alone is not a total order — same-ms ties skipped rows; cursor reads are lossless | QUERIES §9.1/§9.3 |
| **Member ids are namespace-unique across `User`/`Agent`**; `ensure_user`/`ensure_agent` are v2 guarded-CREATE queries returning `(created, existed, collided)`; cross-label collision refuses (`MemberIdCollisionError`) | A shadow node with the other label's id eclipses it in every `coalesce` lookup | QUERIES §2/§7 |
| **Identity source of truth — the `identity` graph is authoritative (standalone)**, not an external-IdP projection | Self-contained system; the `identity` graph owns user identity + auth principals; per-workspace `User` nodes are membership projections of it; steers the open **K-016** auth work | §3, `SERVER.md` §1.3 |

### 1.3 Model stack (shipped)

> User-approved and **shipped** (M2, decided 2026-07-04). The rows below are the *shipped
> defaults*, not env vars: they live in `config/models.json`'s per-kind `defaults` plus the shared
> `config/opencode.example.json`'s provider entries, so **changing a model is a config-file edit +
> restart** (`SERVER.md` §1.8), never a code or env change. The one exception is `EMBEDDING_DIM`
> (`FALKORCHAT_EMBEDDING_DIM`), which stays an env var because it is DDL-time/write-path input,
> not a model choice — and it is fixed at workspace creation (§11). RAM numbers: §11.

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
│   Tool registry, prompt templates                                     │
│   Replicated; served via GRAPH.RO_QUERY                               │
├───────────────────────────────────────────────────────────────────────┤
│ ws:{workspaceId}         (N graphs, one per workspace)  ← hot path    │
│   Workspace-local Users (membership projection of identity)           │
│   Channels, Threads, Messages (chat history)                          │
│   WorkflowRun + StepRun execution traces                              │
│   Chunks/Documents + embeddings (GraphRAG corpus)                     │
│   Extracted Entities + mentions                                       │
│   Materialized copy of the WorkflowDef versions this ws uses          │
├───────────────────────────────────────────────────────────────────────┤
│ (optional) analytics:{...}  rollups / cross-workspace aggregates      │
└───────────────────────────────────────────────────────────────────────┘
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
| **B. Materialize (chosen for defs)** | On workflow *publish*, copy the def subgraph (topology-immutable per version) into each workspace graph that uses it | Real edges → runs traverse their own steps locally; def graphs stay small and are duplicated cheaply |

**Decision:** canonical definitions live in `reference` (single source of truth, versioned).
When a workspace first uses `defKey@v`, we **materialize that version's step subgraph into
`ws:{id}`** under a `WorkflowDefSnapshot`. Runs then have real, local edges to their steps —
fast, self-contained. **The enforced guarantee:** topology (steps, transitions, start)
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
(:Entity)-[:RELATES_TO {label, sourceChunkId, sourceDocumentId, createdAt}]->(:Entity)
  // LLM-extracted fact; label = free-text predicate, never its own rel type; never deduplicated
(:Entity)-[:SAME_AS {matchId, status, confidence, technique, createdAt,
                      decidedAt, decidedBy, resuggestCount, lastResuggestedAt}]->(:Entity)
  // fusion match; direction is a write-time convention (new→existing), not a semantic
  // claim — SAME_AS-anchored reads that need direction-agnosticism match endpoints
  // undirected (QUERIES.md §14.6)

// Workflow ↔ chat linkage (all within ws graph)
(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)
(:StepRun)-[:PRODUCED]->(:Message)                     // step-emitted chat message — NOT the Message→Message EMITTED
```

**Key properties:**
- `Message.role` — `'user'` | `'assistant'` (fast filter without traversing `POSTED_BY`).
  **Derived server-side from the author's node label** (`User → user`, `Agent → assistant`),
  never trusted from the caller.
- `Message.threadId` — **denormalized, deliberately unindexed** navigation metadata: lets §9.2/§5
  result rows point back to their thread without a traversal; §9.1's HEAD/NEXT walk stays the
  canonical thread read. `null` on any row predating the one-off backfill
  (`scripts/backfill_thread_ids.sh`, QUERIES.md §4.x).
- `Thread.updatedAt` — bumped on every new message; drives "recent threads" listing
- `Message.embedding` — inline `vecf32`; no separate node needed
- `Entity.nameNormalized` — case-folded, whitespace-collapsed `name`, computed app-side by the
  same normalization helper extraction stub-repair uses. Backs the FR-8 exact-tier fusion lookup
  with a real `=` comparison, decoupled from RediSearch tokenizer/stemmer behavior (§7.1).

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
what makes the §9 `(createdAt, msgId)` composite order well-defined.

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

> **`Step.stepUid` is the MERGE-backing identity.** A step `key` is unique only *within a def*, so
> it can't back a `MERGE`; every Step carries a synthetic
> `stepUid = "{defKey}:{version}:{stepKey}"` (globally unique within each graph) with an index +
> `UNIQUE` constraint in both `reference` and `ws:{id}` (§7.1/§7.2). `key`/`type`/`config` are the
> display/behaviour props. **`HAS_STEP`** is the def→step containment edge: without it the only
> def→step link is `START`, so reading "all steps of a def" would label-scan every `Step` in the graph
> — and the `stepUid`-prefix `STARTS WITH` alternative live-profiles as a label scan on this build.
> `HAS_STEP` keeps step/transition reads anchored on the def's index (`Node By Index Scan`, verified).
> Canonical publish/materialize/read Cypher: **`QUERIES.md` §11**.

`type` unifies conversational and business flows. **What the engine executes:**
- **`agent` → LLM-native node**: the `config` string carries a plain-language `systemPrompt` + an
  author-set tool fence + bounds; the model runs as a bounded, tool-scoped agent loop. This is the
  type the triage proof uses. **`agent` with no LLM wired returns an empty stub result** —
  deliberate, and load-bearing for the offline test estate.
- `human` / `decision` / `wait` → business processes, **implemented and proven** by the
  `access-request@v1` proof flow (§6.3):
  - **`human`** — parks the run awaiting an assignee. `config.waitsForHuman: true` is **mandatory**
    and enforced at publish; the `StepRun.output` carries an `awaiting` envelope
    (`prompt`/`fields`/`assignee`), so a client learns what the run is waiting for from
    `GET /workflow-runs/{id}/step-runs` with no new query. **Optionally timer-releasable** — see
    the `wait` bullet below for the shared mechanism, identical for both step types.
  - **`decision`** — **no side effect at all**: its semantics are entirely its outgoing guards. With
    no outgoing transition it is a terminal outcome node (the run ends `done`).
    > ⚠️ **Not enforced (residual, K-029).** A `decision` step whose outgoing transitions are *all*
    > conditional, and which does not declare `waitsForHuman`, self-loops to budget exhaustion if
    > none ever fires — there is no symmetric check forcing either an unconditional default arm or
    > a park declaration. The equivalent check for `human`/`wait` steps *is* enforced (above); doing
    > the same here would retro-reject existing fixtures, so it is a deliberate gap, not an oversight.
  - **`wait`** — **signal-driven by default, optionally timer-releasable as well.** A `wait`/`human`
    step parks awaiting an **external signal** delivered through `POST /workflow-runs/{id}/input` —
    mechanically `wait` *is* `human` to the engine, only the `awaiting.kind` string differs, and
    both carry the mandatory `waitsForHuman: true`. A step may **additionally** declare
    `config.waitForSeconds` (relative) or `config.waitUntil` (absolute epoch-ms); a periodic
    in-process sweep (`Services.sweep_due_workflow_runs`, also exposed as
    `POST /workflow-runs/due`) then resumes it once due, exactly as an external signal would. A
    step that declares neither key parks forever — that is valid, specified behaviour, not a bug.
    Mechanism and residual limits: §6.2; full design: `docs/plans/workflow-timers.md`.
- `prompt` / `message` / `tool` → agent-adjacent flows (LLM call, post a message, invoke a tool) —
  **not implemented**: `executor._execute_step` raises `NotImplementedError` naming
  `docs/archive/plans/m3-process-flow.md` (the documented typed-handler seam, decision D-E). An
  unimplemented type fails a run loudly rather than "succeeding" having done nothing.

`Step.config` and `TRANSITION.guard` are **opaque serialized strings** parsed app-side only (rule 8) —
`type:'agent'` needs **no DDL** (whitelist-only add). The `agent` config deserializes to
`{mode, systemPrompt, tools[], permissions{}, waitsForHuman, maxIterations}`; `waitsForHuman:true` is the
explicit suspend signal for a node that parks awaiting a human reply (only intake, in the triage flow).

`config.requiredTools: [<tool name>, ...]` (`docs/plans/must-post-engine-contract.md`) is
`waitsForHuman`'s sibling in the same opaque config, and the same authoring convention: an
`agent`-typed step names a subset of its own `config.tools` that must be successfully dispatched at
least once before the node's turn ends. Absent/empty ⇒ no obligation. Enforced inside
`_run_agent_node` at both of its own exit points — never inside `_drive_loop` — with a violation
trace-and-continuing rather than failing or parking the run: an unconditional `_log.warning`, plus a
`must_post_violation` trace entry on debug/traced runs only. Validated at publish by a fourth
invariant in `services._validate_def_spec`, mirroring the `waitsForHuman` check's own shape
(list-of-strings, `agent`-only, and a subset of `config.tools`).

**`TRANSITION.guard` is LLM-native *and* deterministic.** The guard string is one of:
- **empty `""`** — an unconditional/default transition (**lowest** priority; fires whenever reached);
- a `{kind:'llm', text}` discriminator — judged in natural language against run context (a structured
  boolean verdict + traced rationale);
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
(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)                       // the @mention that started it
(:WorkflowRun)-[:HAS_STEP_RUN]->(:StepRun)                       // membership
(:WorkflowRun)-[:LAST_STEP_RUN]->(:StepRun)                      // TAIL pointer → the NEXT anchor
(:StepRun {stepRunId, stepKey, status, startedAt, endedAt, input, output})
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
- **`PRODUCED`, not `EMITTED` (D2, locked).** StepRun→Message emission is a **distinct** edge type —
  `EMITTED` is the **Message→Message** provenance edge (§9/QUERIES §10); overloading it
  would conflate "cited that seed" with "produced that message."
- **`TraceEvent` + `TRACED` — debug-only.** A debug run writes one `TraceEvent` per LLM
  prompt/response, tool call/result, guard judgment, and retrieval (dozens per run); a non-debug run
  writes **zero**. New node type → new DDL: `TraceEvent.traceId` index **then** UNIQUE (§7.1).
- **`stepRunId`** is the `StepRun`'s stable identity (indexed + UNIQUE, §7.1).
- **Snapshot/`Step` deletion blast radius** (verified 2026-08-18). A
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

**Who writes the run `ctx`.** Every write of `WorkflowRun.ctx` that carries human or
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

**Timer release is one more `resume_run_with_ctx` submitter, not a new write path.**
`Services.sweep_due_workflow_runs` resumes a genuinely-due `wait`/`human` run through the same CAS,
merging an engine-reserved `ctx.timerFired = "<parked step's own key>"` marker (in
`RESERVED_CTX_KEYS` — no human/API caller can set or spoof it) so the run's own
publish-time-required escalation guard fires deterministically on that resume, and only then. It
inherits R-1's exact shape (one more racing submitter, same single-winner CAS) and adds no new
residual class. The sweep writes no new `WorkflowRun` property (there is deliberately no `wakeAt`) and keeps
no scheduler state of its own: dueness is derived fresh, every call, from `StepRun.startedAt` (via the `LAST_STEP_RUN` tail
pointer above) and `Step.config`. Full argument: `docs/plans/workflow-timers.md` §8.

> **Design-review note for any future automated resume caller** (the sweep is the first; a
> future scheduler-driven actor would be another). Such a caller carries two independent risk
> classes, easy to conflate: a **CAS race** on concurrent resumers (closed generically by the
> ctx-write-inside-CAS discipline above) and a **re-park loop** — the automated caller landing on
> a step whose only advance path is a guard the automation itself cannot satisfy, with no human
> present to eventually supply what's missing. The second class has no generic mitigation; each
> new automated caller needs its own guard-satisfiability argument. K-028's first attempt (an
> unconditional-fallback transition) closed the re-park risk but broke `evaluate_guard`'s ordinary
> first-arrival/"not yet" behavior, since that guard fires unconditionally whenever reached, not
> only on a genuine resume — replaced by the `ctx.timerFired` marker-guard approach documented
> above, which is conditional on the automated caller specifically. `docs/plans/workflow-timers.md`'s
> v2→v3 revision note has the full trace.

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
> (verified reproducing `71055f756280` on this commit). **Before extending the lock for a new
> requirement, check whether the needed data is already written atomically elsewhere in the run's
> history first** — K-028 needed a per-run due-time and got it from `StepRun.startedAt` (already
> atomically written by the unmodified suspend path) instead of adding a `WorkflowRun.wakeAt`
> write inside this lock; see the "Timer release…" paragraph above.

> **What `maxSteps` actually means.** `maxSteps` is a **runaway tripwire checked *after* each
> recorded step**, not a hard cap: a run executes at most **`maxSteps + 1`** steps before failing
> with `"step budget exceeded"`. The check runs only on the two driving outcomes — a guard fired
> (OUTCOME A, `executor.py:410`) and a legitimate self-loop (OUTCOME C, `:427`), both
> `rec["stepCount"] > max_steps`. It is **deliberately not applied on the park path** (OUTCOME B —
> a parked run cannot self-drive; see the comment at `executor.py:415-421`) **or on the terminal
> path**. Treat it as a safety bound, not an SLA or a cost budget. Making it an exact cap
> (`>` → `>=`) lands inside the SHA-locked `_drive_loop` and is filed as proposed **K-033**.

> **`status` as a property, not a label**, so a run's state changes in place without
> re-labeling churn; index it for "all running workflows" and the `waiting`-run resume lookup (§12.9).
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
| Offline acceptance test (all three §4.3 paths) | `server/tests/test_process_flow.py` |
| Seeding into `reference` + `ws:{id}` | `scripts/seed_workflows.sh` (second def) |
| Design + traced paths | `docs/archive/plans/m3-process-flow.md` |

The design claim it settles: **a business process needs no new primitive, no new run state and no
scheduler.** A `human` step is just a step whose outgoing guard reads a `ctx` key that does not exist
yet — the executor's existing "no transition fired" outcome parks it; writing the key
(`POST /workflow-runs/{id}/input`) makes the same guard fire on resume. The executor's drive loop is
**not modified** to support any of it.

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
| `TraceEvent` | `traceId` | UNIQUE 1 (debug-only nodes) |
| `ReadCursor` | `cursorId` | UNIQUE 1 |
| `WorkflowDefSnapshot` | `key`, `version` (two indexes) | UNIQUE 2 (composite) |
| `Step` | `key`, `stepUid` (two indexes) | UNIQUE 1 (`stepUid`); `key` index-only (§6.1) |

**Relationship-scoped indexes** — same index-before-constraint ordering rule, on an edge property
rather than a node property (`SAME_AS`, K-050):

| Relationship | Indexed property | Constraint |
|---|---|---|
| `SAME_AS` | `matchId` | UNIQUE 1 |
| `SAME_AS` | `status` | — (hot-filter, no constraint) |

> Every `SAME_AS`-anchored query matches its endpoints **unlabeled** (`(a)`/`(b)`, never
> `(a:Entity)`) — live-verified: a bare label on either endpoint forces a full `Node By Label Scan`
> even though the relationship-property scan alone is fully selective
> (`docs/plans/document-ingestion-graph.md` §1.4; `claude/graph-dba/falkordb-quirks.md`).

**Hot-filter indexes (no constraint)** — support scans/ordering, not identity:

| Label | Property | Serves |
|---|---|---|
| `Thread` | `updatedAt` | recent-threads listing |
| `Message` | `createdAt` | time-range / keyset reads (§9) |
| `WorkflowRun` | `status` | "all running workflows" |
| `StepRun` | `status` | step-state filters |
| `Entity` | `nameNormalized` | FR-8 exact-tier fusion `=` lookup; distinct real entities can share `(nameNormalized, type)` before fusion runs |

> `Message.threadId` is **deliberately unindexed** (§5.1) — nav metadata, not an anchor.

**Full-text index (RediSearch):** `Message.text`, `Entity.name`, via
`db.idx.fulltext.createNodeIndex('Message', 'text')` /
`db.idx.fulltext.createNodeIndex('Entity', 'name')` — backs §5's keyword search and the FR-9
suggested-tier fusion lookup.

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
| Post message | Guarded `CREATE` inside `FOREACH`+`CASE` per path + relink `Thread TAIL → NEXT` (QUERIES.md §4 v2) | Two separate self-guarding variants (first vs subsequent — see §5.3), never a conditional MERGE of the two paths. Always returns a **status row**; retry-idempotency comes from that `dupMsg` status, **not** from the uniqueness constraint — a replayed MERGE re-runs the relink clauses and corrupts the chain. The `Message.msgId` uniqueness constraint stays as the concurrency backstop (rollback verified all-or-nothing). O(1) append. |
| Create channel / thread | plain `CREATE` (server-minted uuid ids) | **Non-idempotent** — a retried create mints a new id; the uniqueness constraints backstop. A MERGE on a fresh uuid could never match. |
| Backfill / import | `UNWIND $rows AS row …` in chunks, or `falkordb-py` bulk loader | Never one giant CREATE — bound transaction memory; size batches (writes ignore TIMEOUT — §10) |
| Embed messages | async worker: compute embedding → `SET m.embedding = vecf32($v)` | Decouple embedding latency from the post path |
| Advance workflow | create `StepRun`, append `NEXT`, move `AT_STEP` | All local to `ws:{id}`; fully transactional within the graph |
| Publish workflow def | write to `reference`; materialize snapshot into consuming `ws:{id}` graphs | Topology-immutable per version (a differing re-publish is rejected `409`); properties stay create-only. Bump version to change either. |

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
- **TIMEOUT posture (live-probed).** Keep the legacy single-knob `TIMEOUT=1000` as the
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
  bookkeeping noise; re-check on upgrades (upstream filing recommended).
- **Observability:** `GRAPH.SLOWLOG` for slow queries, `GRAPH.PROFILE` for plans, Redis metrics.
- **Security:** Redis ACLs scoping `GRAPH.*` per principal (ideally per workspace key pattern),
  TLS in transit, network isolation; secrets outside the data.

---

## 11. Capacity — the numbers that shape the design

> **Full measurements: `docs/test-reports/capacity-report.md`** — the per-message RAM breakdown,
> the M1 append-path load test, the hot-read `GRAPH.PROFILE` closeout, and the per-workspace RAM
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
  12.4 KB is vector + index (§13, retention).

---

## 12. Roadmap

1. **M0 — Stand up the engine.** ✅ FalkorDB running (`falkordb/falkordb:edge`, Redis 8.2.2, module `999999`) via Docker. Live probes confirmed: cross-graph edge behavior, vector DDL syntax, index-before-constraint ordering, `algo.*` procedure set, `vecf32` storage and `db.idx.vector.queryNodes` query surface.
2. **M1 — Chat core.** ✅ Users/Channels/Threads/Messages, thread-scoped `NEXT` + `HEAD`/`TAIL` append path, full-text index, basic read windows. **Application layer:** FastAPI REST server over a service/repository split, single hardcoded tenant, minimal web UI, plus an MCP (Streamable-HTTP) agent front door on the same service layer — full design in `SERVER.md`. Full stack (repository → services → MCP + REST + full-text `search`, plus the static `web/` UI, all mounted in `app.py`) is built and green (110 tests). The append-path load-test + hot-read `GRAPH.PROFILE` DoD is **closed** (~614 msg/s, all four hot reads index-backed, per-workspace RAM budget — §11 and `docs/test-reports/capacity-report.md`). The web request/response path carries incremental `?since=` polling, inline non-blocking errors and clickable search results. M1 chat core is complete.
3. **M2 — GraphRAG.** ✅ Embedding workers, in-graph vector index @1024, hybrid retrieval query (§8), AI `Agent` participant posting answers with `EMITTED` provenance — **QA-accepted, M2 done.** Delivered: every posted message embedded out-of-band via an async `EmbeddingWorker` → LM Studio `/v1/embeddings` (Qwen3-Embedding-0.6B, 1024-dim); `repository.hybrid_search` (§6, cosine-distance ASC, dormant Entity no-op); `AgentResponder` — an `@mention` of the configured agent triggers retrieval-grounded LLM answering (Qwen3-4B-Instruct via LM Studio) posted as the `Agent` (`role:"assistant"`, derived) with an `EMITTED` provenance edge (`QUERIES.md` §10, score+rank), loop-guarded and failure-isolated; the web UI renders assistant replies + reader `isMention`. Served tenant `ws:acme` runs at `EMBEDDING_DIM=1024` (`start_server.sh` gates the live loop on `FALKORCHAT_ENABLE_AGENT`). Baselines: pytest 156 / query suite 149/149. **Groundwork landed earlier:** agent authorship (role derived from the author label), self-guarding v2 write paths (status-row contract, retry-idempotent via `dupMsg`, first-post race refused), `Message.threadId` denorm + backfill script, composite `(createdAt, msgId)` keyset cursors (tie-safe reads), TIMEOUT posture (§10), empirical 1024-dim RAM line (§11). **Deferred to M2.5** (not on the M2-green path, all still open in `BACKLOG.md`): real auth/tenancy (**K-016**), a transport-level externally-authenticated agent actor (**K-017**), real-time push (**K-018**), and a channel-scoped retrieval read (the responder is workspace-wide today).
4. **M3 — Workflow engine.** ✅ Definition model in `reference`, snapshot materialization, run/step-run executor, chat linkage. Both proof flows shipped: the conversational `triage@v1` and the LLM-free `kind:'process'` `access-request@v1` (§6.3), the latter proving a business process needs no new primitive, no new run state and no scheduler. QA-accepted 2026-07-21.
5. **M4 — LLM provider & model configuration.** ✅ Two hand-edited config files behind one internal resolution seam (`ModelGateway`), per-consumer model choice, roles + ordered fallback chains, a per-workspace override that acts as a hard cap across all four consumer kinds, publish-time rejection of unresolvable models, and a durable resolved-model trace on `StepRun`. Delivered 2026-08-11 — `SERVER.md` §1.8.
6. **M5 — Document ingestion & entity fusion.** 🟡 In progress (**K-050**). Document → chunk → extract → fuse, populating the `Document`/`Chunk`/`Entity` shapes §5.1/§7.1 have carried dormant since M2, served as both chat grounding and a standalone knowledge base.
7. **Unscheduled — scale & ops.** Redis Cluster, replicas for RO reads, Sentinel, ACL/TLS, backup/restore drill, per-workspace memory budgeting + shard packing (§10, §11). Deliberately *not* numbered: this was the old "M4" before that number was reassigned to LLM provider configuration, and it has no milestone today.

> **Milestone status is authoritative in `docs/BACKLOG.md`**, not here — this list is the shape of
> the work, and a status marker on it can go stale between milestone closes.

---

## 13. Open questions

- ~~**Workflow guard expression language.**~~ **Resolved — see §6.1.** Guards are the empty-string default form, `{kind:'llm', text}`, or the closed `cmp` comparator family. No expression library, no parser, no `eval`; `kind:'expr'` raises `NotImplementedError`.
- **Retention** — do old messages/embeddings age out (and how does that interact with the always-in-RAM constraint)? (Evicting cold embeddings is the cheapest lever — ~10 KB of the 12.4 KB/msg is vector + index, §11.)
- **Cross-workspace analytics** — app-layer fan-out vs. a dedicated `analytics` rollup graph. (Cost accepted §4; mechanism open, no milestone yet.)
- **Real-time gateway transport** — for the M2.5 push path, Bolt (port `65535`, confirmed in `GRAPH.CONFIG`) vs. RESP/WebSocket. The M1 app *driver* is settled (RESP via `falkordb-py`); this is only the push-gateway choice. (→ **K-018**.)
- **Pre-production config review:** live config defaults noted — `THREAD_COUNT 4`, `OMP_THREAD_COUNT 4`, `CACHE_SIZE 25`, `MAX_QUEUED_QUERIES 25`, `QUERY_MEM_CAPACITY 0` (unlimited), `ASYNC_DELETE 1`. Review before production (TIMEOUT 1000 ms already reviewed & kept — §10).

---

## 14. The server application → `SERVER.md`

The server process — internal layering, the auth/tenancy seam, the REST and MCP front doors, the
`server/` layout, the `server/` testing hazards, and the model-resolution seam — is
**`falkor-chat/docs/SERVER.md`**. This document stops at the graph.

**Redirect table for citations written before 2026-08-24.** Section numbers map straight across:

| Was | Now |
|---|---|
| `DESIGN.md` §14 — M1 application architecture | `SERVER.md` §1 — Application architecture |
| `DESIGN.md` §14.1 Scope decisions | `SERVER.md` §1.1 |
| `DESIGN.md` §14.2 Layering | `SERVER.md` §1.2 |
| `DESIGN.md` §14.3 Auth/tenancy seam | `SERVER.md` §1.3 |
| `DESIGN.md` §14.4 REST surface | `SERVER.md` §1.4 |
| `DESIGN.md` §14.5 Layout | `SERVER.md` §1.5 |
| `DESIGN.md` §14.6 TDD build order | `SERVER.md` §1.6 |
| `DESIGN.md` §14.7 Testing hazards | `SERVER.md` §1.7 |
| `DESIGN.md` §14.8 Model-resolution seam | `SERVER.md` §1.8 |
| `DESIGN.md` §15 — MCP transport | `SERVER.md` §2 — MCP transport |
| `DESIGN.md` §15.1 Shape | `SERVER.md` §2.1 |
| `DESIGN.md` §15.2 Tools → service → query | `SERVER.md` §2.2 |
| `DESIGN.md` §15.3 Client connection contract | `SERVER.md` §2.3 |

> Historical records — `HISTORY.md`, `docs/reviews/*`, closed plans, and the agents' own
> `kaizen/history.md` files — were **not** rewritten: a dated record should say what was true when
> it was written. This table is how those citations resolve.
