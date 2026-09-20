# falkor-chat — Canonical Query Library

Verified against `falkordb/falkordb:v4.18.11` (Redis 8.6.3, module `41811`) — full suite green
**387/387, 2026-08-28** (`./scripts/test_queries.sh`; 346/346 immediately before the K-053 §16
cart/order gate added 41 assertions; 343/343 immediately before U15b's `categoryNormalized`
case-insensitivity fix added 3 assertions; 320/320 immediately before the K-052 §15
product-catalog gate; 282/282, 2026-07-31, before that; 276/276 before the K-039 §12.15 gate;
256/256 before that, before the K-036 §2/§12.14 gate; 241/241 before that, before the K-024
§12.12/§12.13 gate).

All queries use **parameters** — never interpolate user input into Cypher strings.
In `falkordb-py`: `g.query(cypher, params={"key": value})`.

Notation: `$param` = query parameter supplied by the caller.

---

## 1. Workspace setup

Bootstrap is handled by `scripts/bootstrap_schema.sh`. The app calls it (or its
equivalent) once when a new workspace is created. All subsequent queries assume
indexes and constraints are OPERATIONAL.

---

## 2. Users & membership

### Create or update a user (global identity graph)
```cypher
MERGE (u:User {userId: $userId})
ON CREATE SET u.displayName = $displayName,
              u.email       = $email,
              u.createdAt   = $createdAt
ON MATCH  SET u.displayName = $displayName
RETURN u
```
*Graph: `identity`*

### Add user to workspace — guarded ensure (v2, DEF-1 fix)

> **Locked rule: member ids are namespace-unique across `User`/`Agent`.** Every
> `coalesce(u, a)` lookup (role derivation, `POSTED_BY` author resolution, mentions,
> cursors) assumes one id resolves to one member. A `User` created with an id already
> held by an `Agent` (or vice versa) silently eclipses the other node everywhere —
> so both ensures refuse a cross-label collision instead of writing.

```cypher
OPTIONAL MATCH (u:User  {userId:  $userId})
OPTIONAL MATCH (a:Agent {agentId: $userId})
WITH u, a, (u IS NULL AND a IS NULL) AS ok
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (:User {userId: $userId, displayName: $displayName, email: $email})
)
RETURN ok            AS created,
       u IS NOT NULL AS existed,
       a IS NOT NULL AS collided
```
*Graph: `ws:{id}` — keeps a workspace-local copy; only the fields needed for chat.*

**Status-row contract** (exactly one row, always — there is no anchor `MATCH`, so the
query can never zero-row):

| `created` | `existed` | `collided` | Meaning | Caller action |
|---|---|---|---|---|
| `true` | `false` | `false` | fresh node written | success |
| `false` | `true` | `false` | id already a `User` — nothing written | idempotent success (matches the old `MERGE … ON CREATE`-only behavior: re-ensure never updates properties) |
| `false` | `false` | `true` | id held by an `Agent` — **nothing written** | refuse (member-id collision error) |
| `false` | `true` | `true` | pre-guard shadow state: both labels hold the id | alarm — corrupted namespace, manual repair |

Notes (live-verified on this build):
- The write is a **guarded `CREATE` inside `FOREACH`** — `MERGE` inside `FOREACH` is not
  standard OpenCypher, so idempotency comes from the status logic (the `existed` path is
  a structural no-op), not from `MERGE`. The `User.userId` uniqueness constraint stays as
  the same-label concurrency backstop: two racing fresh ensures → one wins, the loser gets
  a constraint violation and retries into `existed=true`.
- **Residual cross-label race window (documented, not closed):** the engine has no
  cross-label constraint, so two *concurrent* `ensure_user`/`ensure_agent` calls with the
  same id can each pass their check and both write — landing in the
  `existed AND collided` alarm state on the next ensure. The window is one
  query-execution wide.
- Both existence checks profile as `Node By Index Scan` (`User.userId` + `Agent.agentId`);
  no label scans.

### Add user to channel
```cypher
MATCH (u:User    {userId:    $userId})
MATCH (c:Channel {channelId: $channelId})
MERGE (u)-[r:MEMBER_OF]->(c)
ON CREATE SET r.role = $role, r.joinedAt = $joinedAt
RETURN r
```

### List channel members
```cypher
MATCH (u)-[:MEMBER_OF]->(c:Channel {channelId: $channelId})
RETURN coalesce(u.userId, u.agentId) AS memberId,
       u.displayName                 AS displayName,
       labels(u)                     AS type
ORDER BY u.displayName
```
*Returns both `User` and `Agent` members. `coalesce(u.userId, u.agentId)` gives a single
stable identifier regardless of node type; `labels(u)` lets the caller distinguish.*

### List thread participants (K-036 — web-api-coverage FR-8)

```cypher
// $threadId — a thread's participants = its parent channel's roster (design decision:
// docs/plans/web-api-coverage.md §2.3 — MEMBER_OF is modeled only at Channel granularity;
// there is no Thread-level membership edge, and this is a UI/visibility choice, not a
// technically-derived "who can be @mentioned here" set — see that section's "Known,
// accepted gap")
MATCH (c:Channel)-[:HAS_THREAD]->(t:Thread {threadId: $threadId})
MATCH (u)-[:MEMBER_OF]->(c)
RETURN coalesce(u.userId, u.agentId) AS memberId,
       u.displayName                 AS displayName,
       labels(u)                     AS type
ORDER BY u.displayName
```

*Extends "List channel members" (above) by one leading hop: `Thread` ← `HAS_THREAD` (backward)
← `Channel` → `MEMBER_OF` (forward) → member. Same `coalesce`/`labels()` shape, and the same
pre-existing gap that query already has: `Agent` nodes carry `.name`, not `.displayName` —
`displayName` comes back `null` for an `Agent` row (`labels(u)` still correctly reads
`["Agent"]`, which is all the caller needs to derive `kind`). Not a new gap introduced here;
not fixed here either — out of this query's scope.*

**`GRAPH.PROFILE` (2026-07-28, v4.18.11 / module `41811`, isolated `ws:gdbtest`, 3-member
roster — 2 `User` + 1 `Agent`)** — anchors on `Node By Index Scan | (t:Thread)`
(`Thread.threadId`), then two forward `Conditional Traverse` hops (`HAS_THREAD` backward to
`Channel`, `MEMBER_OF` forward to each member) — **no label scan anywhere**:

```
Results | Records produced: 3
    Sort | Records produced: 3
        Project | Records produced: 3
            Conditional Traverse | Records produced: 3
                Node By Index Scan | (t:Thread) | Records produced: 1
```

Also verified: a thread whose channel has zero members returns an empty result (not an error)
— the demo-seed-timing edge case the plan's §5.2 calls out. No new index/constraint/RAM —
reuses `Thread.threadId`, `Channel.channelId` is not even needed as an anchor (the traversal
starts from `t`), and `MEMBER_OF` carries no index of its own (traversal-only, per the
existing "List channel members" query).

### Member-kind lookup (author/mention validation + role derivation)
```cypher
UNWIND $ids AS id
OPTIONAL MATCH (u:User  {userId:  id})
OPTIONAL MATCH (a:Agent {agentId: id})
RETURN id, CASE WHEN coalesce(u, a) IS NULL THEN null
                ELSE labels(coalesce(u, a))[0] END AS kind
```
*One round trip resolves a batch of ids to `'User'`, `'Agent'`, or `null` (unknown). Both legs
are `Node By Index Scan`s; `labels(…)[0]` subscripting is live-verified on this build. The
service maps the kind to a message role — `User → 'user'`, `Agent → 'assistant'`,
`null → reject before writing` (the mapping itself is service-side, not stored here).*

---

## 3. Channels & threads

### Create a channel
```cypher
CREATE (c:Channel {channelId: $channelId, name: $name, createdAt: $createdAt})
RETURN c
```
*Plain `CREATE` (K-007 fold-in): `channelId` is a server-minted uuid, so a `MERGE` could never
match — it was a CREATE wearing a MERGE costume. The `Channel.channelId` uniqueness constraint
stays as the backstop. Consequence: **creates are non-idempotent** — a retried create mints a
new id (unlike message posts, which are retry-idempotent via `dupMsg`).*

### List channels in a workspace
```cypher
MATCH (c:Channel)
WHERE c.channelId > ''
RETURN c.channelId AS channelId,
       c.name      AS name,
       c.createdAt AS createdAt
ORDER BY c.createdAt DESC
LIMIT $limit
```
*Anchors on the `Channel.channelId` range index — the always-present predicate
`c.channelId > ''` (every `channelId` is a non-empty string) turns the listing into a
`Node By Index Scan`, not a `NodeByLabelScan`. Ordered by `createdAt` (channel **creation**
time, newest first), which is free once the scan is index-backed. True activity-recency
(most-recent message/thread per channel) would require expanding `HAS_THREAD` to
`Thread.updatedAt` for every channel — the Channel-level edge traversal §5.2 of `DESIGN.md`
deliberately avoids — so it is intentionally **not** used here. Route via `GRAPH.RO_QUERY`.*

### Create a thread
```cypher
MATCH (c:Channel {channelId: $channelId})
CREATE (t:Thread {threadId: $threadId, title: $title,
                  createdAt: $createdAt, updatedAt: $createdAt})
CREATE (c)-[:HAS_THREAD]->(t)
RETURN t
```
*Plain `CREATE` like the channel create (server-minted id, constraint backstop,
non-idempotent). Zero rows back = the channel anchor missed and **nothing was written** — the
repository raises (tripwire; the service pre-validates the channel).*

### List recent threads in a channel
```cypher
MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t:Thread)
RETURN t.threadId, t.title, t.updatedAt
ORDER BY t.updatedAt DESC
LIMIT $limit
```
*Uses the `Thread.updatedAt` range index — no edge scan.*

---

## 4. Messages

### Post the first message in a thread (v2 — self-guarding)
*Use when the thread has no HEAD yet (first message ever).*
```cypher
MATCH (t:Thread {threadId: $threadId})
OPTIONAL MATCH (t)-[:HEAD]->(h)
OPTIONAL MATCH (dup:Message {msgId: $msgId})
OPTIONAL MATCH (ua:User  {userId:  $authorId})
OPTIONAL MATCH (aa:Agent {agentId: $authorId})
WITH t, h, dup, coalesce(ua, aa) AS author
UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid
OPTIONAL MATCH (mu:User  {userId:  mid})
OPTIONAL MATCH (ma:Agent {agentId: mid})
WITH t, h, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems
WITH t, h, dup, author, mems,
     (h IS NULL AND dup IS NULL AND author IS NOT NULL) AS ok
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (m:Message {msgId: $msgId, text: $text, role: $role,
                     createdAt: $createdAt, threadId: $threadId})
  CREATE (t)-[:HEAD]->(m)
  CREATE (t)-[:TAIL]->(m)
  CREATE (m)-[:POSTED_BY]->(author)
  SET t.updatedAt = $createdAt
  FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))
)
RETURN ok                 AS written,
       h    IS NOT NULL   AS hadHead,
       dup  IS NOT NULL   AS dupMsg,
       author IS NOT NULL AS authorFound
```

### Post a subsequent message in a thread (v2 — self-guarding)
*Use for every message after the first.*
```cypher
MATCH (t:Thread {threadId: $threadId})-[tailRel:TAIL]->(prev:Message)
OPTIONAL MATCH (dup:Message {msgId: $msgId})
OPTIONAL MATCH (ua:User  {userId:  $authorId})
OPTIONAL MATCH (aa:Agent {agentId: $authorId})
WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author
UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid
OPTIONAL MATCH (mu:User  {userId:  mid})
OPTIONAL MATCH (ma:Agent {agentId: mid})
WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems
WITH t, tailRel, prev, dup, author, mems,
     (dup IS NULL AND author IS NOT NULL) AS ok
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (m:Message {msgId: $msgId, text: $text, role: $role,
                     createdAt: $createdAt, threadId: $threadId})
  CREATE (prev)-[:NEXT]->(m)
  DELETE tailRel
  CREATE (t)-[:TAIL]->(m)
  CREATE (m)-[:POSTED_BY]->(author)
  SET t.updatedAt = $createdAt
  FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))
)
RETURN ok                 AS written,
       false              AS hadHead,
       dup  IS NOT NULL   AS dupMsg,
       author IS NOT NULL AS authorFound
```

### Status-row contract (service dispatch)

Both v2 paths **always return a status row** when their anchor matches, so zero rows now
unambiguously means "anchor missing":

| Result | Meaning | Service action |
|---|---|---|
| zero rows | thread missing (first) / no TAIL yet (subsequent) | first-path: 404; subsequent: retry as first-path |
| `written=true` | committed | success |
| `dupMsg=true` | msgId already exists | **idempotent success** (retry replay) |
| `hadHead=true` (first path) | lost the first-post race | re-dispatch as subsequent |
| `authorFound=false` | unknown member | 4xx, nothing written |

### v2 notes (live-verified — must survive future edits)

- **Why v2 exists.** The old §4 paths `MERGE`d the message: a retry replay matched the existing
  node and re-ran the unconditional `CREATE`/`DELETE` clauses — NEXT self-loops, duplicated
  `POSTED_BY`, corrupted TAIL. And two racing first-posts both saw "no HEAD" and produced two
  chains. v2 guards each path *inside its single `GRAPH.QUERY`* via
  `FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | …)` — a guard on each path, never a
  conditional merge of the two paths (locked decision). `MERGE` on the message is replaced by a
  guarded `CREATE`; the `Message.msgId` uniqueness constraint stays as the concurrency backstop
  (constraint-violation rollback is verified all-or-nothing).
- **Author resolution is label-specific** (two indexed `OPTIONAL MATCH`es + `coalesce`) — the old
  label-less `MATCH (author {userId: $authorId})` profiled as an `All Node Scan` **and** silently
  no-opped Agent authors (Agents carry `agentId`). Agents can now author messages.
- **The `UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END)` guard is
  load-bearing for the write itself** — a bare empty `UNWIND` collapses the row stream *before*
  the `FOREACH` and the whole write silently no-ops. `$mentions = []` is a verified no-op.
- **`DELETE` inside `FOREACH`** (the TAIL relink) and **nested `FOREACH`** (mentions) are
  live-verified on this build.
- **`dupMsg` trusts the msgId.** msgIds are server-minted (`uuid4().hex`, never client-supplied),
  so a duplicate can only be a replay of our own write — payload equality is not re-checked. If
  msgIds ever become client-supplied, add a payload checksum before honoring `dupMsg` as
  idempotent success.
- **`threadId` is written inline and is deliberately unindexed** — navigation/display metadata
  for §9.2/§5 results; §9.1 remains the canonical thread walk. Skipping the index saves
  per-workspace RAM and write cost.

### Participant mentions — how the mention block works (live-verified)

Both write paths carry a `$mentions` parameter: a flat list of member ids, each a `userId`
**or** an `agentId`. The mention resolution runs *before* the guard; the nested `FOREACH`
creates `(:Message)-[:MENTIONS_MEMBER]->(member)` edges *inside* it — atomically, in the
**same** `GRAPH.QUERY` as the HEAD/TAIL write (the atomicity rule).

- **`MENTIONS_MEMBER` is distinct from `MENTIONS`.** `MENTIONS`→`Entity` is GraphRAG co-occurrence
  (§6). Participant mentions use `MENTIONS_MEMBER`. Do not conflate them.
- **Empty list is a true no-op.** `$mentions = []` → the `CASE` yields `[null]`, both `OPTIONAL
  MATCH`es miss, `collect(DISTINCT …)` drops the null → `[]`, the nested `FOREACH` creates
  nothing — and the guard above keeps the write itself alive.
- **Index-anchored member resolution.** Each id is resolved by two `Node By Index Scan`s
  (`User.userId`, `Agent.agentId`) — *not* `WHERE u.userId = mid OR u.agentId = mid`, which
  `GRAPH.PROFILE` shows degrading to an `All Node Scan`.
- **Dedup + unknown-skip are free.** `collect(DISTINCT …)` collapses duplicate ids to one edge and
  drops ids that resolve to no member (`collect` ignores nulls). `['u3','u3','a7','nope']` → 2
  edges `[u3, a7]`, one result row. Validating that unknown mentions are an error vs. silently
  dropped is a **service-layer** decision; the query itself skips them.

### Post a reply (REPLY_TO inside the guarded FOREACH — live-verified)

Anchor the quoted message alongside the other anchors and create the edge **inside** the guard
(shown for the subsequent path; the first path is analogous):

```cypher
MATCH (t:Thread {threadId: $threadId})-[tailRel:TAIL]->(prev:Message)
MATCH (quoted:Message {msgId: $quotedMsgId})
OPTIONAL MATCH (dup:Message {msgId: $msgId})
…same WITH chain, carrying quoted…
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (m:Message { … })
  …
  CREATE (m)-[:REPLY_TO]->(quoted)
  …
)
RETURN …same status row…
```

*Verified by `test_queries.sh` (§4 m4). Not yet exposed by the server API — fold into
`repository.py` when reply support lands.*

### Read a full thread (in order)
```cypher
MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message)
MATCH (first)-[:NEXT*0..]->(m:Message)
MATCH (m)-[:POSTED_BY]->(author)
RETURN m.msgId, m.text, m.role, m.createdAt,
       author.userId, author.displayName, labels(author) AS authorType,
       coalesce(m.toolsUsed, []) AS toolsUsed
ORDER BY m.createdAt
```
*`length(path)` is not supported in ORDER BY on this build — use `m.createdAt`
(indexed) instead. Bounded by thread length; paginate with cursor for long threads.*

*`toolsUsed` (K-056, `docs/reviews/salesperson-tool-reliability-ml.md` §4.3) is the
tool-use breadcrumb source: `link_step_emission` (§12.6) stamps it with the producing
step's own non-`post_message` dispatched tool names; `coalesce(…, [])` degrades a
pre-K-056 row (no property at all) and a step that dispatched nothing to the same empty
list. `executor._assemble_messages` reads it back to tag a tool-backed `assistant` turn
when this thread is replayed into a later turn's prompt — the fix for the diagnosed
skip-and-fabricate mechanism (small local model, no visible tool-use precedent by turn
3-4 of an extended conversation).*

### Read a thread window (cursor-based pagination)
```cypher
// Page forward from a known message
MATCH (cursor:Message {msgId: $afterMsgId})-[:NEXT*1..]->(m:Message)
MATCH (t:Thread {threadId: $threadId})-[:HEAD|NEXT*0..]->(cursor)
MATCH (m)-[:POSTED_BY]->(author)
RETURN m.msgId, m.text, m.role, m.createdAt,
       author.userId, labels(author) AS authorType
ORDER BY m.createdAt
LIMIT $limit
```

### Get a single message
```cypher
MATCH (m:Message {msgId: $msgId})
OPTIONAL MATCH (m)-[:POSTED_BY]->(author)
OPTIONAL MATCH (m)-[:REPLY_TO]->(quoted:Message)
RETURN m, author, m.threadId AS threadId, quoted.msgId AS quotedId
```

### 4.x Backfill `threadId` (one-off, verified idempotent)

The v2 write paths stamp `threadId` inline; messages written before K-007 lack it. Run once per
existing workspace after deploying v2 (`scripts/backfill_thread_ids.sh`). Per-thread variant
(batchable — run per threadId to bound query time; **writes cannot be killed by TIMEOUT** on
this build, so bound the work yourself):

```cypher
MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message)
MATCH (first)-[:NEXT*0..]->(m:Message)
WHERE m.threadId IS NULL
SET m.threadId = t.threadId
RETURN count(m) AS backfilled
```

Workspace-wide variant: drop the `{threadId: $threadId}` filter
(`MATCH (t:Thread)-[:HEAD]->(first:Message) …`).

*Idempotent — a second run returns `0`. **Orphan caveat:** the walk anchors on `HEAD`, so a
message unreachable from a HEAD (residue of the pre-v2 defects) is not backfilled — acceptable,
since such messages are already invisible to thread reads. Until backfilled, old rows return
`threadId: null` in §9.2/§5 — clients must tolerate null.*

---

## 5. Full-text search

### Keyword search over message text (within a workspace)
```cypher
CALL db.idx.fulltext.queryNodes('Message', $query)
YIELD node AS m, score
MATCH (t:Thread)-[:HEAD|NEXT*0..]->(m)
MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)
RETURN m.msgId, m.threadId AS threadId, m.text, m.createdAt, score
ORDER BY score DESC
LIMIT $limit
```
*Scoped to a channel via traversal. Omit the channel MATCH to search workspace-wide.
`m.threadId` (denormalized, K-007) lets clients jump from a hit to its thread without a
traversal — `null` on pre-K-007 rows until the §4.x backfill runs.*

---

## 6. GraphRAG hybrid retrieval

The AI participant answers a question using vector similarity + graph traversal
in a single read-only query. Route via `GRAPH.RO_QUERY` to a replica if available.

```cypher
// $qVec     = vecf32 of the query embedding (same dim as the vector index)
// $k        = number of ANN neighbors to seed from (e.g. 10)
// $channelId = scope results to a specific channel (omit MATCH on c to go workspace-wide)
// score     = cosine distance: 0 = identical, lower = more similar → ORDER BY score ASC

CALL db.idx.vector.queryNodes('Message', 'embedding', $k, $qVec)
YIELD node AS seed, score
MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed)
MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)
OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message)
WITH seed, score, collect(DISTINCT related)[..5] AS expanded
RETURN seed.msgId, seed.text, seed.role, score,
       [m IN expanded | m.text] AS relatedContext
ORDER BY score ASC
LIMIT $limit
```

### Set a message embedding (async, after posting)
```cypher
MATCH (m:Message {msgId: $msgId})
SET m.embedding = vecf32($embedding)
```
*Run from the embedding worker after the message is posted. Decoupled from the
write path — the message is readable before the embedding lands.*

### 6.1 Introspect a workspace's vector-index dimension (K-042 Landing 2, FR-19)

Design: `docs/plans/llm-provider-config-graph.md` §3.2 (edge behaviours), §3.4 (error
surface). `EmbeddingWorker.embed_message` (`server/falkorchat/embedding.py`) runs this
**before** calling the embedder at all — a mismatch raises `EmbeddingDimensionError`
with no HTTP call made and no vector written. `Repository.read_index_dimension`
(`server/falkorchat/repository.py`).

```cypher
// [verified] GRAPH.RO_QUERY ws:{id} — replica-routable, zero write risk, never
// creates the graph as a side effect (see the "graph key absent" row below).
// $label = the label about to be written (only 'Message' is written anywhere in
//          this codebase today — 'Chunk' is bootstrapped DDL, never populated);
// $prop  = 'embedding' (parameterised, though this codebase only ever uses one).
CALL db.indexes() YIELD label, types, options
WHERE label = $label AND types[$prop] = ['VECTOR']
RETURN options[$prop].dimension AS dim
```

Four distinct edge behaviours, tabulated in `-graph.md` §3.2 — all four collapse to
two outcomes at the repository layer (a real `int`, or `None` for every "no vector
index to compare against" case):

| Situation | Raw result | `read_index_dimension` returns |
|---|---|---|
| Vector index exists (even with zero vectors written — dimension is index metadata, not data) | one row, `dim` = the configured int | the int |
| Label has only non-vector (e.g. `RANGE`) indexes | one row, `dim` = `NULL` | `None` |
| Label unknown to this graph | zero rows | `None` |
| Graph key `ws:{id}` does not exist at all | FalkorDB raises `ERR Invalid graph operation on empty key` | `None` (caught) |

The last row is why the guard's own error handling cannot be a bare `if not rows`
— an un-bootstrapped workspace *errors*, it does not return an empty result. The
read is routed via `ro_query` precisely so this probe can never implicitly create
the graph as a side effect (the same "empty key" `ResponseError` behaviour
`services._read_or_absent` already relies on for the reference/snapshot cold-graph
probe).

**Caching** (`-graph.md` §3.3 layer 2): `EmbeddingWorker` caches the returned
dimension per `(ws, label)` for the process lifetime — the index dimension
provably cannot change in place (§3.1: only drop+recreate does, an out-of-band
admin action) — but **never caches a `None`**, since an un-bootstrapped workspace
can become bootstrapped without a process restart.

---

## 7. Agents

### Register an AI agent in a workspace — guarded ensure (v2, DEF-1 fix)

Mirror of the §2 guarded user ensure — same locked rule (**member ids are
namespace-unique across `User`/`Agent`**), same status-row contract with the labels
swapped: `existed` = id already an `Agent` (idempotent success), `collided` = id held
by a `User` (**nothing written** — refuse). Exactly one row, always. See §2 for the
full contract table, the FOREACH/CREATE idempotency note, and the residual
cross-label race window.

```cypher
OPTIONAL MATCH (a:Agent {agentId: $agentId})
OPTIONAL MATCH (u:User  {userId:  $agentId})
WITH a, u, (a IS NULL AND u IS NULL) AS ok
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (:Agent {agentId: $agentId, name: $name, model: $model, createdAt: $createdAt})
)
RETURN ok            AS created,
       a IS NOT NULL AS existed,
       u IS NOT NULL AS collided
```
*Both existence checks are `Node By Index Scan`s; the `Agent.agentId` uniqueness
constraint remains the same-label concurrency backstop.*

### Add agent to a channel
```cypher
MATCH (a:Agent   {agentId:   $agentId})
MATCH (c:Channel {channelId: $channelId})
MERGE (a)-[r:MEMBER_OF]->(c)
ON CREATE SET r.role = 'assistant', r.joinedAt = $joinedAt
RETURN r
```

---

## 8. Diagnostics

### List all graphs
```
GRAPH.LIST
```

### Show indexes for a graph
```cypher
CALL db.indexes()
YIELD label, properties, types, entitytype, status
RETURN label, properties, types, entitytype, status
ORDER BY entitytype, label
```

### Show constraints for a graph
```cypher
CALL db.constraints()
YIELD type, label, properties, status
RETURN type, label, properties, status
ORDER BY label
```

### Profile a query (spot label scans, cartesian products)
```
GRAPH.PROFILE ws:acme "MATCH (m:Message {msgId:'abc'}) RETURN m"
```
*Run in redis-cli. Look for NodeByLabelScan (bad) vs NodeIndexSeek (good).*

### Slow query log
```
GRAPH.SLOWLOG ws:acme
```

---

## 9. Read-cursors & since-reads (MCP `read_messages`)

Per-agent read state for the MCP `read_messages` tool. A `ReadCursor` node holds a per-*(member,
thread)* `(lastReadAt, lastReadMsgId)` composite pair; `read_messages` returns messages after
that point, where the base is either an explicit `since` timestamp (pure read, plain `>`) or the
member's cursor pair (composite keyset). Mentions of the reader are **flagged** (`isMention`),
and results are **chronological**.

> **Ordering is the pagination invariant.** Since-reads must return the *earliest* messages first
> so that a `LIMIT`-truncated page is a contiguous prefix and the cursor can advance to the last
> delivered row without skipping anything. The original mention-first sort
> (`ORDER BY isMention DESC, …`) broke this: with more unread rows than `limit`, a late mention
> crowded out earlier messages that the cursor then jumped past — silent message loss. Clients
> that want mentions surfaced first should sort by the `isMention` flag locally.
>
> **The deterministic total order is `(createdAt, msgId)` (K-007).** A timestamp alone is not a
> total order: two messages in the same millisecond made the old plain-`>` paging skip the tied
> sibling at a page boundary — permanently (live-reproduced). Both since-reads now
> `ORDER BY m.createdAt, m.msgId`, and the cursor advances to the newest **returned**
> `(createdAt, msgId)` pair — never the server clock. Tie-break is *lexical* msgId: within one
> millisecond, delivery order is id order, not arrival order — acceptable across writers (the
> service's monotonic clock removes same-process ties). If human-facing tie order ever matters,
> mint k-sortable ids (UUIDv7/ULID) — never re-sort pages.

Schema (bootstrap): `ReadCursor.cursorId` **range index + uniqueness constraint** (index before
constraint). `cursorId = "{memberId}:{threadId}"` — deterministic, so `MERGE` is safe and unique.
`lastReadMsgId` is a plain property — **no schema/bootstrap change** was needed for K-007.

**Reader-match convention.** The reader may be a `User` (`userId`) or an `Agent` (`agentId`), so the
mention-flag and cursor queries match `me.userId = $meId OR me.agentId = $meId`. This `OR` is
acceptable **only** because `me`/`mem` is already bound by a traversal or the indexed
`cursorId`/`agentId` anchor — it is never the scan anchor (contrast the write-path resolution
above, where the same `OR` would force an `All Node Scan`). Author id is returned with
`coalesce(author.userId, author.agentId)` for the same User-or-Agent reason.

### 9.1 Read a thread since a cursor/timestamp (thread-scoped) — v2

Two predicate forms over one body. **Cursor-driven reads** use the composite keyset
(formulation A — never skips or re-delivers, even across millisecond ties); **explicit-`since`
reads** keep the plain `>` and may re-deliver or skip *within that exact millisecond* (OQ3
contract — agents that need lossless catch-up use cursor mode).

```cypher
MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message)
MATCH (first)-[:NEXT*0..]->(m:Message)
WHERE m.createdAt > $since
   OR (m.createdAt = $since AND m.msgId > $sinceMsgId)
-- plain-`>` form (explicit since): WHERE m.createdAt > $since
MATCH (m)-[:POSTED_BY]->(author)
OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me)
       WHERE me.userId = $meId OR me.agentId = $meId
WITH m, author, count(me) > 0 AS isMention
RETURN m.msgId, m.text, m.role, m.createdAt,
       coalesce(author.userId, author.agentId) AS authorId,
       labels(author) AS authorType, isMention, m.threadId AS threadId,
       author.displayName AS displayName
ORDER BY m.createdAt, m.msgId
LIMIT $limit
```
*Anchored on the `Thread.threadId` index; walks the thread's `NEXT` chain. Chronological in the
`(createdAt, msgId)` total order (see the §9 ordering note) — **both** forms use the same ORDER
BY. `$sinceMsgId` defaults to `''` when only a timestamp is known (empty string sorts before
every id). Formulation A mirrors the ORDER BY 1:1; the fallback **formulation B**
(`WHERE m.createdAt >= $since AND (m.createdAt > $since OR m.msgId > $sinceMsgId)`) plans
identically today — re-profile on engine upgrades (edge build, moving target).*

### 9.2 Read workspace-wide since a timestamp (room-wide, no thread) — v2
```cypher
MATCH (m:Message)
WHERE m.createdAt > $since                        // Node By Index Scan on Message.createdAt
   OR (m.createdAt = $since AND m.msgId > $sinceMsgId)
-- plain-`>` form (explicit since): WHERE m.createdAt > $since
MATCH (m)-[:POSTED_BY]->(author)
OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me)
       WHERE me.userId = $meId OR me.agentId = $meId
WITH m, author, count(me) > 0 AS isMention
RETURN m.msgId, m.text, m.role, m.createdAt,
       coalesce(author.userId, author.agentId) AS authorId,
       labels(author) AS authorType, isMention, m.threadId AS threadId,
       author.displayName AS displayName
ORDER BY m.createdAt, m.msgId
LIMIT $limit
```
*`GRAPH.PROFILE`-confirmed: **both** predicate forms plan as a bare
`Node By Index Scan | (m:Message)` on `Message.createdAt` with no residual `Filter` op (the OR
folds into the scan). `m.threadId` rides along as navigation metadata — `null` on pre-K-007 rows
until the §4.x backfill runs; clients must tolerate null. `author.displayName` rides along too
(K-014): the polling web client renders it in place of the raw `authorId` and tolerates `null`
(members seeded without a display name). Both since-reads carry the same column set. **TIMEOUT risk:** the live default is
1000 ms; on a large workspace keep `$limit` modest and consider a bounded `$since` window. No
room-wide cursor in M1 — this variant requires an explicit `$since` (service defaults it to
`0`/epoch, plain `>`).*

### 9.3 Advance a read-cursor (RW — only in `advance` mode) — v2 composite
```cypher
MATCH (mem) WHERE mem.userId = $meId OR mem.agentId = $meId
MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId: $cursorId})
ON CREATE SET rc.memberId = $meId, rc.threadId = $threadId
WITH rc, ($now > coalesce(rc.lastReadAt, 0)
      OR ($now = coalesce(rc.lastReadAt, 0)
          AND $nowMsgId > coalesce(rc.lastReadMsgId, ''))) AS adv
SET rc.lastReadAt    = CASE WHEN adv THEN $now      ELSE rc.lastReadAt    END,
    rc.lastReadMsgId = CASE WHEN adv THEN $nowMsgId ELSE rc.lastReadMsgId END
RETURN rc.lastReadAt, rc.lastReadMsgId
```
*`cursorId = "{meId}:{threadId}"`; `MERGE` backed by the `ReadCursor.cursorId` uniqueness
constraint. The composite monotonic guard is computed **once** in the `WITH` so both `SET`s see
the pre-write state. All five scenarios live-verified: create `(2000,'k2')` → tie-larger
`(2000,'k3')` advances → tie-smaller `(2000,'k2')` refused (stale replay) → backward
`(1500,'k9')` refused → forward `(3000,'k4')` advances. `coalesce(rc.lastReadMsgId, '')` covers
pre-K-007 cursors — **no cursor backfill needed**. The service owns `($now, $nowMsgId)` — the
newest `(createdAt, msgId)` pair it actually delivered, never the server clock and never
client-supplied — and may short-circuit before writing (an empty page advances nothing). When
the member node doesn't exist the anchor `MATCH` yields no rows and the query is a **no-op
returning no row** — callers must not index into an empty result (the repository returns
`None`). This is a write — cannot route to a replica; use only when `advance=true`.*

> **Member-match caveat here.** §9.3's opening `MATCH (mem) WHERE mem.userId=$meId OR mem.agentId
> =$meId` *is* an anchor, so on paper it risks an `All Node Scan`. It is acceptable in practice
> because the write is a single-member point operation gated by the subsequent `MERGE` on the
> indexed `cursorId`; the candidate set is one node. If a large-workspace `GRAPH.PROFILE` ever
> shows this hurting, split into two label-specific `OPTIONAL MATCH`es + `coalesce` (the write-path
> pattern) before the `MERGE`.

### 9.4 Read a cursor (to compute the composite `since` when not supplied)
```cypher
MATCH (rc:ReadCursor {cursorId: $cursorId})
RETURN rc.lastReadAt, rc.lastReadMsgId
```
*Single `Node By Index Scan` point-lookup on `ReadCursor.cursorId`, returning the
`(lastReadAt, lastReadMsgId)` pair. When no cursor exists, this returns no row and the service
uses the epoch base `(0, '')`. A pre-K-007 cursor has no `lastReadMsgId` — the pair reads back
`(ts, null)` and the service maps it to `''`.*

---

## 10. Agent answer provenance — `EMITTED` (K-013, generalized K-050 M5 Stage 5)

The server-side AI responder posts its answer **as the `Agent`** (role derived `assistant`, K-007)
via the §4 write path, and records the retrieval seeds (§6 hybrid search, merged with the §14.3
`Chunk` ANN pool app-side, K-050 M5 Stage 5 FR-2) that grounded the answer as
**`(answer:Message)-[:EMITTED {score, rank}]->(seed)`** provenance edges — `seed` is a `Message` or
a `Chunk`.

**Edge shape (locked, live-verified; generalized to a `Chunk` seed at K-050 M5 Stage 5).**

- **Direction / endpoints:** `(:Message)-[:EMITTED]->(:Message|:Chunk)`, answer → seed. The answer
  is the subject; each cited seed is the object. Same convention as `REPLY_TO` (subject message
  points to the referenced message). The hot query — "given an agent answer, what did it cite?" —
  anchors on the answer's `msgId` (`Node By Index Scan`) and expands `EMITTED` outward to ≤k seeds;
  the reverse ("which answers cited this seed?") is the same edge type traversed inbound.
- **Properties:** `score` (the cosine distance of that seed at answer time; 0 = identical) and
  `rank` (0-based position in the ranked seed list). Both are point-in-time snapshots — retrieval is
  non-deterministic as the graph grows, so the score/rank at answer time is the provenance value.
- **No index / no constraint.** Endpoints are `Message`/`Chunk` nodes already carrying their own
  range index + uniqueness constraint (`msgId`/`chunkId`); the provenance read anchors there.
  FalkorDB traverses the typed `EMITTED` edge from the anchored answer via its adjacency matrix — no
  relationship-property index is needed. Uniqueness is guaranteed structurally (see idempotency
  below), so no relationship constraint (which would also need a supporting index) is created.
- **`EMITTED` is a third, distinct edge type.** `MENTIONS_MEMBER`→`User`/`Agent` (participants, §4)
  and `MENTIONS`→`Entity` (GraphRAG co-occurrence, §6) are unrelated — do not conflate any of them.
- **Seed-id resolution: bare-id `coalesce`, not a namespaced id scheme
  (`document-ingestion-graph.md` §3.1).** `$seedIds` resolves against both `Message.msgId` and
  `Chunk.chunkId` directly via the same two-label `OPTIONAL MATCH` + `coalesce` idiom already used
  for author/mention resolution below — no `kind` field, no `msg:<uuid>`/`chunk:<uuid>` prefix
  scheme. Both id spaces are disjoint server-minted `uuid4` generators (the same "astronomically
  negligible" collision argument already accepted for `User.userId`/`Agent.agentId` sharing one
  namespace, §2), so a bare id is enough.

**Atomicity (locked):** the `EMITTED` edges are written **inside the same `GRAPH.QUERY` as the
answer's §4 write**, inside the guarded `FOREACH`, exactly like `MENTIONS_MEMBER`. This makes
"message + provenance" one all-or-nothing unit gated by the same status guard: a `dupMsg` retry
replay (`ok=false`) skips the whole `FOREACH`, so provenance is written **exactly once** — no torn
"answer with no provenance" state, no separate idempotency mechanism. (Contrast: a follow-up
`link_emitted` write would double-write on retry unless separately made idempotent, and could tear
on a crash between the two queries.)

### 10.1 Post an agent answer with provenance (subsequent path — self-guarding)

*The realistic path: the agent answers into a thread that already has the triggering message, so a
HEAD exists and the answer is always a subsequent-path write. The first-path variant is analogous —
same seed block folded into the §4 first-message write; verified.*

```cypher
// $seedIds = ranked list of seed ids (Message.msgId or Chunk.chunkId) from the merged
//            §6/§14.3 retrieval (order = rank)
// $scoreBy = { <seedId>: <cosine distance> }  — per-seed ANN score at answer time
// $rankBy  = { <seedId>: <0-based rank> }
// $mentions, $authorId (= the agentId), $msgId, $text, $role='assistant', etc. as in §4
MATCH (t:Thread {threadId: $threadId})-[tailRel:TAIL]->(prev:Message)
OPTIONAL MATCH (dup:Message {msgId: $msgId})
OPTIONAL MATCH (ua:User  {userId:  $authorId})
OPTIONAL MATCH (aa:Agent {agentId: $authorId})
WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author
UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid
OPTIONAL MATCH (mu:User  {userId:  mid})
OPTIONAL MATCH (ma:Agent {agentId: mid})
WITH t, tailRel, prev, dup, author, collect(DISTINCT coalesce(mu, ma)) AS mems
UNWIND (CASE WHEN $seedIds = [] THEN [null] ELSE $seedIds END) AS sid
OPTIONAL MATCH (sm:Message {msgId: sid})
OPTIONAL MATCH (sc:Chunk {chunkId: sid})
WITH t, tailRel, prev, dup, author, mems, collect(DISTINCT coalesce(sm, sc)) AS seeds
WITH t, tailRel, prev, dup, author, mems, seeds,
     (dup IS NULL AND author IS NOT NULL) AS ok
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (m:Message {msgId: $msgId, text: $text, role: $role,
                     createdAt: $createdAt, threadId: $threadId})
  CREATE (prev)-[:NEXT]->(m)
  DELETE tailRel
  CREATE (t)-[:TAIL]->(m)
  CREATE (m)-[:POSTED_BY]->(author)
  SET t.updatedAt = $createdAt
  FOREACH (mem  IN mems  | CREATE (m)-[:MENTIONS_MEMBER]->(mem))
  FOREACH (seed IN seeds | CREATE (m)-[:EMITTED {
    score: $scoreBy[coalesce(seed.msgId, seed.chunkId)],
    rank:  $rankBy[coalesce(seed.msgId, seed.chunkId)]}]->(seed))
)
RETURN ok                 AS written,
       false              AS hadHead,
       dup  IS NOT NULL   AS dupMsg,
       author IS NOT NULL AS authorFound
```

*This is the §4 subsequent write with **one added block**: a second guarded `UNWIND` resolves the
seed ids to bound `Message`/`Chunk` nodes (`collect(DISTINCT coalesce(sm, sc))` — dedups, drops
unknown seeds like the mention block), and a nested `FOREACH` creates the `EMITTED` edges inside the
guard. Status-row contract is identical to §4 (zero rows = no TAIL → retry as first-path;
`dupMsg=true` = idempotent replay). `$seedIds = []` is a **verified no-op** — the `CASE` guard keeps
the write itself alive (a bare `UNWIND []` would collapse the row stream before the `FOREACH`), so
this same query serves non-provenance writes too.*

**Live-verified build quirks that shape this query:**
- **A map-projection cannot be a `CREATE` endpoint** (`CREATE (m)-[:EMITTED]->(rec.node)` errors:
  *"Invalid input '.'"*). The endpoint must be a **bound node variable**, so seeds are collected as
  nodes (`collect(DISTINCT coalesce(sm, sc))`) and per-edge props are pulled from **map parameters
  keyed by `coalesce(seed.msgId, seed.chunkId)`** — exactly one of the two properties is non-null
  per seed node, so `coalesce` always resolves the right key regardless of seed kind. Dynamic
  map-param indexing by a node property is verified working on this build.
- **Two sequential guarded `UNWIND`s** (mentions, then seeds) each collapse via `collect` before the
  next expands — no row multiplication. Verified, including with the seed block resolving against
  two labels (`gdba_probe_emitted`: one `Message` seed and one `Chunk` seed in a single guarded
  write — both edges created correctly, no row multiplication, no interference with the mention
  block's own `UNWIND`).

### 10.2 Read an answer's provenance (forward — the hot path)

```cypher
MATCH (a:Message {msgId: $msgId})-[e:EMITTED]->(s)
OPTIONAL MATCH (s)<-[:HAS_CHUNK]-(d:Document)
RETURN labels(s)[0]              AS seedKind,       // 'Message' | 'Chunk'
       coalesce(s.msgId, s.chunkId) AS seedId,
       s.text                    AS text,
       s.role                    AS role,           // null when seedKind = 'Chunk'
       d.documentId              AS documentId,     // null when seedKind = 'Message'
       d.title                   AS documentTitle,
       e.score, e.rank
ORDER BY e.rank
```
*`s` is deliberately **unlabeled** in the pattern — it's discovered by traversal from the
index-anchored `a`, not independently searched, so there's no `Node By Label Scan` risk (a plain
traversal endpoint, the existing precedent this codebase already uses via `labels(coalesce(a,b))[0]`
for author/mention resolution). The `OPTIONAL MATCH (s)<-[:HAS_CHUNK]-(d:Document)` only ever
matches when `s` is a `Chunk` (a `Message` has no incoming `HAS_CHUNK` edge) — this is what satisfies
AC-5's "traverses back through `(:Chunk)<-[:HAS_CHUNK]-(:Document)`" requirement, folded into the
same read, one extra hop.*

*`GRAPH.PROFILE` (`gdba_probe_emitted2`): `Node By Index Scan | (a:Message)` →
`Conditional Traverse (a)-[:EMITTED]->(s)` → `Optional Conditional Traverse (d)` — clean, no label
scan, matches the pre-generalization profile shape with one added hop. Data correctness verified: a
`Message`-seeded row returns `seedKind:'Message'`, `role:'user'`, `documentId:null`; a `Chunk`-seeded
row returns `seedKind:'Chunk'`, `role:null`, `documentId`/`documentTitle` correctly populated from
the extra hop. Ordered by `rank` (ascending = most influential seed first). Route via
`GRAPH.RO_QUERY`.*

### 10.3 Read which answers cited a seed (reverse)

```cypher
OPTIONAL MATCH (sm:Message {msgId: $seedId})
OPTIONAL MATCH (sc:Chunk {chunkId: $seedId})
WITH coalesce(sm, sc) AS s
MATCH (a:Message)-[e:EMITTED]->(s)
RETURN a.msgId, a.role, a.createdAt, e.score, e.rank
ORDER BY a.createdAt DESC
```
*Resolves the anchor via the same two-label `coalesce` as the write (§10.1) and forward-read
(§10.2) — bare id, not a namespaced scheme, same locked-convention argument as §10's seed-resolution
note. An answer is always a `Message`, so the row shape (`{answerMsgId, role, createdAt, score,
rank}`) is unchanged from before generalization.*

*`GRAPH.PROFILE` (`gdba_probe_1103`), including the specific planner-trap question this needed
checking: whether `a:Message`'s label triggers a `Node By Label Scan` here, now that `a` is the
**unbound** endpoint (traversed *to*, not *from*) and `s` is the one resolved by the earlier
`OPTIONAL MATCH` pair. Populated 1,000 candidate `Message` answer nodes all citing the same `Chunk`
seed (the worst case for a label-scan risk on `a`) plus one citing a `Message` seed. Both profiled
identically clean: `Node By Index Scan (sm:Message)` / `Node By Index Scan (sc:Chunk)` (exactly one
produces a row per call) → `Conditional Traverse` from the resolved `s` outward to `a` → `Project` →
`Sort` — **no `Node By Label Scan` at any point**: `s` already supplies the index anchor one step
earlier, so `a`'s label costs nothing (the opposite shape from a genuine two-unbound-endpoint
label-scan trap). Data correctness confirmed: the `Chunk`-seed call returned all 1,000 citing
answers correctly ordered `createdAt DESC`; the `Message`-seed call returned exactly the one correct
citing answer. Route via `GRAPH.RO_QUERY`.*

---

## 11. Workflow definitions & snapshots (M3 Slice 1 — K-020 / K-021)

The workflow **definition model** lives canonically in the global **`reference`** graph as versioned
`WorkflowDef` templates; publishing a def version and **materializing** it into a workspace
graph (`ws:{id}`) as a local `WorkflowDefSnapshot` subgraph are the two write paths here. DESIGN §6.1;
the executor (`WorkflowRun`/`StepRun`, K-022) and chat linkage (K-023) are **out of scope** for this
slice — these queries only build and read the definition/snapshot structure.

**Model (both graphs — structurally identical, only the root label differs):**

```
// reference graph                        // ws:{id} graph
(:WorkflowDef {key,version,name,kind})    (:WorkflowDefSnapshot {key,version,name,kind})
     -[:HAS_STEP]->(:Step)                     -[:HAS_STEP]->(:Step)
     -[:START]->(:Step)                        -[:START]->(:Step)
(:Step {stepUid,key,type,config})         (:Step {stepUid,key,type,config})
(:Step)-[:TRANSITION {on,guard,order}]->(:Step)
```

**The `Step.stepUid` identity (locked, K-020).** A step's `key` is unique only *within a def*, so it
can never back a `MERGE` (AGENTS.md: "every `MERGE` must be backed by a uniqueness constraint"). Every
Step therefore carries a synthetic **`stepUid = "{defKey}:{version}:{stepKey}"`** — globally unique
within each graph — with a range index + `UNIQUE` constraint in **both** graphs (`bootstrap_schema.sh`).
`stepUid` is the MERGE key; `key`/`type`/`config` are set on create. `Step.key` keeps its own index
(display/traversal anchor) but **no** constraint.

**`HAS_STEP` — the containment edge (locked, K-020; the §B8 resolution).** `(:WorkflowDef|
:WorkflowDefSnapshot)-[:HAS_STEP]->(:Step)` gives every def/snapshot an **index-anchored** handle on
*all* its steps. It exists because the plan's original scoping candidate — filtering transitions by
`from.stepUid STARTS WITH ($key + ':' + $version + ':')` — **live-profiles as a `Node By Label Scan`
+ `Filter` on this build, not an index range scan** (verified: a `STARTS WITH` on the indexed prefix
does not plan as an index scan; and the `(:WorkflowDef)-[:START]->()-[:TRANSITION*]` walk alternative
is worse — a `Cartesian Product` + `Semi Apply` that *still* label-scans `Step` and silently misses
steps unreachable from `START`). Without `HAS_STEP` there is **no** def→step edge except `START`, so
reading "all steps of a def" would scan every `Step` of every def in the graph. With `HAS_STEP`, both
the step read and the transition read anchor on `Node By Index Scan | (d:WorkflowDef)` /
`(snap:WorkflowDefSnapshot)` and traverse outward — O(steps-in-this-def), verified below.

**`config` and `guard` are opaque serialized strings (rule 8 / DESIGN §1.2).** They are stored and
returned **verbatim** and **never** filtered inside. Guard *evaluation* is run-time (K-022); Slice 1
does not force the §13 guard-language decision.

**Create-only on properties, topology-enforced by `services` (K-034).** Publish and materialize are
single-graph `MERGE` queries — a re-publish/re-materialize never updates a stored `name`/`kind`/step
`config`/transition `guard`. It is **not** additive on **structure** as of K-034:
`services.publish_workflow_def`/`materialize_def` reject a re-publish/re-materialize whose step set,
transition set, or start key differs from what's stored (`409 WorkflowDefConflictError`), before any
write. `Repository.publish_def`/`materialize_snapshot` themselves remain thin, non-validating
primitives — the guarantee is enforced one layer up. Because the query has two sequential `UNWIND`
blocks (steps, then transitions), the naive shape **row-multiplies** the final `RETURN` (steps ×
transitions rows);
each block is collapsed back to one row with an aggregation (`WITH d, count(st) AS stepCount` …) so the
contract returns **exactly one status row**. `MATCH (start …)`/`MATCH (from …)`/`MATCH (to …)` inside
the write resolve MERGE-created steps by their indexed `stepUid` — the spec validation in `services`
(start step exists; every transition endpoint is a declared step key) runs *before* the write, so
these matches always resolve for a valid spec.

> **Two-phase materialization is inherently non-atomic across the graph boundary (DESIGN §3/§4).**
> `materialize_def` reads the def subgraph from `reference` (§11.2) then writes the snapshot into
> `ws:{id}` (§11.4) — two separate `GRAPH.QUERY` calls (edges can't cross graphs). A failure between
> them leaves `reference` untouched and the workspace `MERGE` idempotent, so a retry completes cleanly
> — no torn state. Accepted, documented.

### 11.1 Publish a def version (reference — idempotent)

```cypher
// $key,$version,$name,$kind,$startKey
// $steps       = [ {key, type, config}, … ]                       config = opaque string
// $transitions = [ {from, to, on, guard, order}, … ]              guard  = opaque string
MERGE (d:WorkflowDef {key: $key, version: $version})
  ON CREATE SET d.name = $name, d.kind = $kind
WITH d
UNWIND $steps AS s
  MERGE (st:Step {stepUid: $key + ':' + $version + ':' + s.key})
    ON CREATE SET st.key = s.key, st.type = s.type, st.config = s.config
  MERGE (d)-[:HAS_STEP]->(st)
WITH d, count(st) AS stepCount
MATCH (start:Step {stepUid: $key + ':' + $version + ':' + $startKey})
MERGE (d)-[:START]->(start)
WITH d, stepCount
UNWIND $transitions AS tr
  MATCH (from:Step {stepUid: $key + ':' + $version + ':' + tr.from})
  MATCH (to:Step   {stepUid: $key + ':' + $version + ':' + tr.to})
  MERGE (from)-[rel:TRANSITION {on: tr.on, order: tr.order}]->(to)
    ON CREATE SET rel.guard = tr.guard
WITH d, stepCount, count(rel) AS transitionCount
RETURN d.key AS key, d.version AS version, stepCount, transitionCount
```

*Every node `MERGE` is backed by a `UNIQUE` constraint (`WorkflowDef {key,version}`, `Step {stepUid}`).
The `TRANSITION` MERGE-key is `(from, on, order, to)` so distinct outcomes/orders between the same two
steps are distinct edges; `guard` is set-on-create only (may be empty, never a match key). Live-verified
on `_probe`: run 1 → 5 nodes / 9 rels (4 `HAS_STEP` + 1 `START` + 4 `TRANSITION`) / 32 props, returns
one row `{key, version, stepCount:4, transitionCount:4}`; run 2 (same content) → 0 created (idempotent),
same row. A re-publish with a **different** topology (§11 preamble) is rejected at the service layer
before reaching this query — see K-034.*

> **⚠️ `$transitions = []` poisons the version — this query is deliberately *un*guarded.** Unlike the
> §4 mention block (which wraps its list in `UNWIND (CASE WHEN $x = [] THEN [null] ELSE $x END)`), the
> trailing `UNWIND $transitions` here collapses the row stream to zero rows **after** the `WorkflowDef`,
> its `Step`s and the `START` edge have already been MERGEd. The `RETURN` then yields nothing, so
> `repository.publish_def` / `materialize_snapshot` index `result_set[0]` → `IndexError` (a 500, not a
> named error) on a **half-written** def — and because publish is `MERGE … ON CREATE SET`, re-publishing
> the corrected spec on the same `(key, version)` is a silent no-op: the version is permanently wrong and
> unrepairable without deleting the subgraph. The guard is therefore **at the service layer**, not here:
> `services._validate_def_spec` rejects a zero-transition spec with `WorkflowDefSpecError` → 400, nothing
> written (K-024 U4b, O-6). Model a terminal outcome as a step with **no outgoing transition**, never as
> a def with no transitions. Any *new* caller reaching `_PUBLISH_CYPHER` without going through
> `services.publish_workflow_def` must re-do that validation itself — including `materialize_snapshot`
> (`repository.py:1397`), which reuses this same query shape unguarded (see **K-030**,
> `docs/BACKLOG.md`, for the proposed Cypher-level `CASE`-guard fix that would close this gap for both
> callers). *(Verified 2026-07-24; discovered 2026-07-20 while writing `tests/test_executor_process.py`
> — every pre-existing publish test carried ≥1 transition, so no test reached the real query with an
> empty list.)* A second, independent service-layer guard sits beside this one — K-034's topology-
> conflict gate (`services._check_no_structural_conflict`, 409) — see the preamble above.

### 11.2 Read a def subgraph (reference — the materialize input, F6-safe)

Two focused, index-anchored reads (no `length(path)` ordering — unsupported on this build, F6; the app
reconstructs step order from `TRANSITION.order`/topology).

**11.2a Meta + steps:**
```cypher
MATCH (d:WorkflowDef {key: $key, version: $version})
OPTIONAL MATCH (d)-[:START]->(start:Step)
OPTIONAL MATCH (d)-[:HAS_STEP]->(s:Step)
RETURN d.name AS name, d.kind AS kind, start.key AS startKey,
       collect(DISTINCT {key: s.key, type: s.type, config: s.config}) AS steps
```
**11.2b Transitions:**
```cypher
MATCH (d:WorkflowDef {key: $key, version: $version})-[:HAS_STEP]->(from:Step)-[tr:TRANSITION]->(to:Step)
RETURN collect({from: from.key, to: to.key, on: tr.on, guard: tr.guard, order: tr.order}) AS transitions
```
*Both anchor on `Node By Index Scan | (d:WorkflowDef)` and traverse `HAS_STEP` outward — verified no
`Node By Label Scan`. `start.key` is a grouping key, not an engine-level constant: the one-row
collapse is a cardinality **premise**, see the callout below. A def with no transitions returns
`transitions: []` (11.2b yields zero rows → the app treats absence as empty). Route via
`GRAPH.RO_QUERY`.*

> **⚠️ The one-row collapse is CONDITIONAL — it holds only while the root has exactly one `START`
> edge (K-031).** `start.key` is a **non-aggregated grouping key** beside `collect(DISTINCT …)`, so
> "constant across the fan-out" above is a *premise*, not a guarantee. **Verified live on
> falkordb/falkordb:v4.18.11 (K-031 V-1, snapshot side, throwaway `ws:k031probe`):** two `START`
> edges on one root ⇒ **11.2a returns two rows**, one per distinct `startKey`, each carrying the
> full `steps` collection. Consumers that take `result_set[0]` therefore pick an **arbitrary** start
> key. `repository._read_subgraph` (the materialize + executor input) still does exactly that —
> unchanged, because it is on locked paths; the K-031 observability reader
> `repository._read_structure` consumes **all** rows for precisely this reason.
>
> Two consequences worth stating plainly. First, `_read_structure` returns `start_keys: list[str]`
> where this section documents a scalar `startKey` — a **deliberate, documented shape divergence**
> from the "`repository.py` is a 1:1 mirror of QUERIES.md" rule (DESIGN §14.2); the query text is
> untouched, only the Python row handling above it. The service layer renames it to
> `startKey`/`startKeys` at the REST boundary. Second, **how a root acquires a second `START` edge
> is not this section's subject — it is K-034's** (a re-publish is create-only on *properties* but
> additive on *structure*).

### 11.3 List defs / get a def (reference)

```cypher
// list all defs (every version), newest version first within each key
// $limit
MATCH (d:WorkflowDef) WHERE d.key > ''
RETURN d.key AS key, d.version AS version, d.name AS name, d.kind AS kind
ORDER BY d.key, d.version DESC
LIMIT $limit
```
```cypher
// get the latest version for a key
MATCH (d:WorkflowDef {key: $key})
RETURN d.key AS key, d.version AS version, d.name AS name, d.kind AS kind
ORDER BY d.version DESC
LIMIT 1
```
```cypher
// get a specific version (point lookup on the composite key)
MATCH (d:WorkflowDef {key: $key, version: $version})
RETURN d.key AS key, d.version AS version, d.name AS name, d.kind AS kind
```
*All three anchor on `Node By Index Scan | (d:WorkflowDef)` (the `key` index; `WHERE d.key > ''` makes
the list an index scan rather than a label scan). `version` is a string — order is lexicographic; the
caller uses zero-padded or monotonic version strings if numeric ordering matters. Route via
`GRAPH.RO_QUERY`.*

### 11.4 Materialize a snapshot (workspace — idempotent)

Same shape as §11.1 with the `WorkflowDefSnapshot` root label, run against `ws:{id}`. The
`$name/$kind/$startKey/$steps/$transitions` parameters come from the §11.2 read of `reference`
(two-phase, see the note above).

```cypher
MERGE (snap:WorkflowDefSnapshot {key: $key, version: $version})
  ON CREATE SET snap.name = $name, snap.kind = $kind
WITH snap
UNWIND $steps AS s
  MERGE (st:Step {stepUid: $key + ':' + $version + ':' + s.key})
    ON CREATE SET st.key = s.key, st.type = s.type, st.config = s.config
  MERGE (snap)-[:HAS_STEP]->(st)
WITH snap, count(st) AS stepCount
MATCH (start:Step {stepUid: $key + ':' + $version + ':' + $startKey})
MERGE (snap)-[:START]->(start)
WITH snap, stepCount
UNWIND $transitions AS tr
  MATCH (from:Step {stepUid: $key + ':' + $version + ':' + tr.from})
  MATCH (to:Step   {stepUid: $key + ':' + $version + ':' + tr.to})
  MERGE (from)-[rel:TRANSITION {on: tr.on, order: tr.order}]->(to)
    ON CREATE SET rel.guard = tr.guard
WITH snap, stepCount, count(rel) AS transitionCount
RETURN snap.key AS key, snap.version AS version, stepCount, transitionCount
```

*Node MERGEs backed by `WorkflowDefSnapshot {key,version}` + `Step {stepUid}` constraints (both
workspace-local). Produces a snapshot subgraph **structurally identical** to the reference def.
Live-verified idempotent (run 2, same content → 0 created). Re-materialize with unchanged topology is
a no-op on write (properties are always create-only). A re-materialize whose topology differs from the
stored snapshot is rejected (`409 WorkflowDefConflictError`) before this query runs — K-034.*

### 11.5 Read a snapshot subgraph (workspace)

Mirror of §11.2 with the `WorkflowDefSnapshot` root; anchors on `Node By Index Scan |
(snap:WorkflowDefSnapshot)`.

```cypher
// meta + steps
MATCH (snap:WorkflowDefSnapshot {key: $key, version: $version})
OPTIONAL MATCH (snap)-[:START]->(start:Step)
OPTIONAL MATCH (snap)-[:HAS_STEP]->(s:Step)
RETURN snap.name AS name, snap.kind AS kind, start.key AS startKey,
       collect(DISTINCT {key: s.key, type: s.type, config: s.config}) AS steps
```
```cypher
// transitions
MATCH (snap:WorkflowDefSnapshot {key: $key, version: $version})-[:HAS_STEP]->(from:Step)-[tr:TRANSITION]->(to:Step)
RETURN collect({from: from.key, to: to.key, on: tr.on, guard: tr.guard, order: tr.order}) AS transitions
```

> Same conditional one-row collapse as §11.2 — this is the label the K-031 V-1 probe actually ran
> against: **two `START` edges ⇒ two meta rows**. See the §11.2 note.

### 11.6 List / get snapshots (workspace)

```cypher
// list snapshots in a workspace
// $limit
MATCH (snap:WorkflowDefSnapshot) WHERE snap.key > ''
RETURN snap.key AS key, snap.version AS version, snap.name AS name, snap.kind AS kind
ORDER BY snap.key, snap.version DESC
LIMIT $limit
```
```cypher
// get a specific snapshot (point lookup on the composite key)
MATCH (snap:WorkflowDefSnapshot {key: $key, version: $version})
RETURN snap.key AS key, snap.version AS version, snap.name AS name, snap.kind AS kind
```
*Index-anchored on `WorkflowDefSnapshot.key`. Route via `GRAPH.RO_QUERY`.*

**Live-verified build quirks that shape §11:**
- **`STARTS WITH` on an indexed string prefix does NOT plan as an index scan** on this build — it
  profiles as `Node By Label Scan` + `Filter`. Scope def/snapshot subgraph reads via the `HAS_STEP`
  containment edge (index-anchored), never a `stepUid` prefix filter. (New — folds into the KB.)
- **`STARTS WITH` with a concatenated prefix needs explicit parentheses:** `x STARTS WITH ($a + ':' +
  $b)` — without them the parser mis-associates (`STARTS WITH` binds tighter than `+`) and errors
  *"Type mismatch: expected Boolean but was String"*. Moot here (we use `HAS_STEP`) but noted.
- **Sequential `UNWIND` blocks row-multiply the final `RETURN`** unless each is collapsed with an
  aggregation (`WITH d, count(st) AS stepCount`). Verified: the collapsed form returns one clean row.

---

## 12. Workflow execution — runs, step-runs & traces (M3 executor — K-022)

The **executor** walks a materialized `WorkflowDefSnapshot` (§11) as a `WorkflowRun` that records each
executed step as a `StepRun`. All of these live **workspace-local** in `ws:{id}` (the snapshot, the run,
the trace are one connected subgraph — no cross-graph edge). DESIGN §6.2. These queries are the 1:1
contract for `repository.py` (U3) — **method name = query name** below. Every state-move is a **single
`GRAPH.QUERY`** (atomicity, rule 4); every read anchors on an index (PROFILE-verified, no label scan).

**Model (ws:{id}) — additive to §11's snapshot:**

```
(:WorkflowRun {runId, defKey, defVersion, status, startedAt, endedAt,
               ctx, trace, maxSteps, stepCount, waitingThreadId})
(:WorkflowRun)-[:OF_DEF]->(:WorkflowDefSnapshot)        // which materialized def (§11)
(:WorkflowRun)-[:AT_STEP]->(:Step)                      // current position (cleared on terminal)
(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)              // the @mention that started it (FR-7/AC-1)
(:WorkflowRun)-[:HAS_STEP_RUN]->(:StepRun)              // membership (all step-runs of a run)
(:WorkflowRun)-[:LAST_STEP_RUN]->(:StepRun)             // TAIL pointer — the NEXT anchor (M4)
(:StepRun {stepRunId, stepKey, status, startedAt, endedAt, input, output})
(:StepRun)-[:RAN]->(:Step)                              // which def step this run-step executed
(:StepRun)-[:NEXT]->(:StepRun)                          // execution order (audit trail)
(:StepRun)-[:PRODUCED]->(:Message)                      // step-emitted chat message (D2 — NOT EMITTED)
(:StepRun)-[:TRACED]->(:TraceEvent)                     // debug-only trace record (FR-4)
(:TraceEvent {traceId, seq, kind, at, payload})         // debug runs only; payload = flat string
```

**Locked shape decisions (this gate):**

- **`PRODUCED`, not `EMITTED` (D2, locked).** StepRun→Message emission is a **distinct** edge type.
  `EMITTED` is already the K-013 **Message→Message** provenance edge (§10) — reusing it would conflate
  "this answer cited that seed" with "this step produced that message." `PRODUCED` is
  `(:StepRun)-[:PRODUCED]->(:Message)`; endpoints carry their own `stepRunId`/`msgId` unique index, so
  no relationship index/constraint is needed (same reasoning as §10's `EMITTED`).
- **`LAST_STEP_RUN` tail pointer anchors the atomic advance (M4).** Mirrors the `Thread` HEAD/TAIL
  pattern (§4/§5.2): `record_step_and_advance` reads the tail to find the previous `StepRun`, hangs
  `NEXT` from it, and moves the tail — all in the **same** query. **No chain-walk, no label scan** —
  the previous step-run is reached by one `Optional Conditional Traverse` of the tail edge (O(1)).
- **`ctx`/`input`/`output`/`payload` are opaque flat strings (rule 8)** — stored/returned verbatim,
  never filtered inside. The executor (de)serializes app-side.
- **`waitingThreadId` denorm rides the `WorkflowRun.status` index — no new index.** See §12.9.
- **`runId`/`stepRunId`/`traceId` are server-minted** → plain guarded `CREATE`, with the `UNIQUE`
  constraint as the concurrency backstop (the §3/§4 channel/thread pattern). `link_step_emission` and
  `append_trace_event` use the endpoints' existing indexes.

**Status-move contract (all state-move queries).** Each state-move `MATCH`es its anchor(s), so
**zero rows = the anchor missed** (run gone / wrong current state): the service treats zero rows as
"CAS did not apply" (suspend/resume) or "run not found" (advance/complete/fail → `WorkflowRunNotFound`).
A returned row = the move committed.

### 12.1 `start_run` — begin a run at the snapshot's START step

```cypher
// $runId,$defKey,$defVersion,$startedAt,$triggerMsgId server-minted / caller-supplied
// $ctx = opaque serialized state. On this chat path the service seeds the trigger
//        message's thread as the §2.4 resume denorm anchor, MERGED with any caller
//        `run_ctx` — {"threadId": …} alone when the caller supplies none, otherwise
//        {"threadId": …, …caller keys}. Reserved keys (threadId/error/timerFired) are
//        rejected service-side before this write ever runs, same rule as §12.12.
// $trace = bool (debug instance?);
// $maxSteps = run-level step budget (DS default 12, §7). A tripwire checked AFTER each
//             recorded step, not a hard cap: a run executes at most maxSteps + 1 steps
//             (see the §12.5 note).
MATCH (snap:WorkflowDefSnapshot {key: $defKey, version: $defVersion})-[:START]->(start:Step)
MATCH (trigger:Message {msgId: $triggerMsgId})
CREATE (r:WorkflowRun {runId: $runId, defKey: $defKey, defVersion: $defVersion,
                       status: 'running', startedAt: $startedAt, ctx: $ctx,
                       trace: $trace, maxSteps: $maxSteps, stepCount: 0,
                       waitingThreadId: ''})
CREATE (r)-[:OF_DEF]->(snap)
CREATE (r)-[:AT_STEP]->(start)
CREATE (r)-[:TRIGGERED_BY]->(trigger)
RETURN r.runId AS runId, start.key AS startKey, r.status AS status, r.stepCount AS stepCount
```
*Both anchors are `Node By Index Scan` (`WorkflowDefSnapshot.key`, `Message.msgId`) — PROFILE-verified.
No `LAST_STEP_RUN` yet; the first `record_step_and_advance` seeds the tail. `waitingThreadId` starts
`''` (set only while parked, §12.4). Backed by the `WorkflowRun.runId` UNIQUE constraint. Zero rows =
snapshot has no START, or the trigger message is missing.*

### 12.2 `record_step_and_advance` — the M4 tail-anchored atomic advance

The engine's hot write: one query records the just-executed step as a `StepRun`, appends it to the
`NEXT` audit trail via the tail pointer, moves the tail, relinks `AT_STEP` to the transition's `to`
step, and bumps `stepCount`. **All atomic (rule 4).**

```cypher
// $runId; $stepRunId (server-minted); $stepStatus (e.g. 'done'); $startedAt,$endedAt;
// $input,$output (opaque strings); $toStepUid = the destination Step's stepUid (executor
// resolves it app-side from the firing transition = "{defKey}:{version}:{toKey}");
// $resolvedModel,$modelSource,$modelFallback (K-042 Landing 2, FR-8) — NULL for a
// non-LLM step (decision/human/wait, the offline agent-without-LLM stub)
MATCH (r:WorkflowRun {runId: $runId})-[atRel:AT_STEP]->(cur:Step)
MATCH (to:Step {stepUid: $toStepUid})
OPTIONAL MATCH (r)-[lastRel:LAST_STEP_RUN]->(prevSR:StepRun)
CREATE (sr:StepRun {stepRunId: $stepRunId, stepKey: cur.key, status: $stepStatus,
                    startedAt: $startedAt, endedAt: $endedAt,
                    input: $input, output: $output,
                    resolvedModel: $resolvedModel,
                    modelSource: $modelSource,
                    modelFallback: $modelFallback})
CREATE (r)-[:HAS_STEP_RUN]->(sr)
CREATE (sr)-[:RAN]->(cur)
FOREACH (p  IN CASE WHEN prevSR  IS NULL THEN [] ELSE [prevSR]  END | CREATE (p)-[:NEXT]->(sr))
FOREACH (lr IN CASE WHEN lastRel IS NULL THEN [] ELSE [lastRel] END | DELETE lr)
CREATE (r)-[:LAST_STEP_RUN]->(sr)
DELETE atRel
CREATE (r)-[:AT_STEP]->(to)
SET r.stepCount = r.stepCount + 1
RETURN r.stepCount AS stepCount, sr.stepRunId AS stepRunId, cur.key AS ranStepKey
```

*Anchors: `Node By Index Scan | (r:WorkflowRun)` + `(to:Step)`; the previous step-run is found by a
single `Optional Conditional Traverse` of `LAST_STEP_RUN` (**no chain-walk / no label scan** —
PROFILE-verified). The **first** advance finds no `LAST_STEP_RUN` (both `FOREACH`s no-op) and just
seeds the tail + `AT_STEP` relink. Every later advance: `NEXT` from the old tail, drop the old tail
edge, create the new tail, drop the old `AT_STEP`, create the new. Verified: advance 1 → `stepCount=1`,
exactly one `AT_STEP`, tail = the new SR, zero `NEXT`; advance 2 → `stepCount=2`, `NEXT` old→new, exactly
one tail, two `HAS_STEP_RUN`, `RAN` edges to the correct def steps. Zero rows = run missing, no `AT_STEP`
(already terminal), or `$toStepUid` not a step in this workspace.*

> **The `FOREACH (x IN CASE WHEN n IS NULL THEN [] ELSE [n] END | …)` idiom** is the verified way to
> act on an optionally-present node/edge without collapsing the row (quirks KB) — used here twice (NEXT
> append; tail-edge delete). `DELETE` inside `FOREACH` and top-level `DELETE atRel` + re-`CREATE` of the
> same edge type are both live-verified on this build.

**K-042 Landing 2 (FR-8, `docs/plans/llm-provider-config-graph.md` §1.4).**
`resolvedModel`/`modelSource`/`modelFallback` ride the same `CREATE` as three additional,
nullable `StepRun` properties, written by the executor's `_run_agent_node`/`_record` for
an LLM-executing step and passed as `NULL` for every non-LLM step. **[verified]** A `NULL`
parameter **omits** the property entirely — `CREATE (s:StepRun {..., resolvedModel: $rm})`
with `rm=NULL` reports one fewer `Properties set` than a non-null case and `keys(s)` does
not include `resolvedModel`. One query shape serves LLM and non-LLM steps alike: no
branching, no extra bytes on a `decision`/`human`/`wait` StepRun. `modelSource ∈
{'workspace', 'step', 'default'}` names the precedence rung that won (`'workspace'` is
L2-3, not yet reachable); `modelFallback` is `true` only when `resolvedModel` is not the
first model in the chain the winning rung named, and is **omitted** (never written `false`)
on the common, non-fallback path — same "nullable, absent by default" contract as the other
two. No backfill: a `StepRun` written before Landing 2 reads back `NULL` for all three,
permanently — a historical run's model is genuinely unknown, never re-derived from today's
config.

### 12.3 `suspend_run` — guarded CAS `running → waiting`

```cypher
// $runId, $threadId (the run's thread — denormed so §12.9 can find it index-anchored)
MATCH (r:WorkflowRun {runId: $runId})
WHERE r.status = 'running'
SET r.status = 'waiting', r.waitingThreadId = $threadId
RETURN r.runId AS runId, r.status AS status
```
*A **compare-and-set**: the flip commits only if the run is currently `running`. A second suspend (or a
suspend of a non-running run) matches the node but fails the `WHERE` → **zero rows**, nothing written.*

### 12.4 `resume_run` — guarded CAS `waiting → running` (single-flight)

```cypher
// $runId
MATCH (r:WorkflowRun {runId: $runId})
WHERE r.status = 'waiting'
SET r.status = 'running', r.waitingThreadId = ''
RETURN r.runId AS runId, r.status AS status
```
*The single-flight guard for concurrent human replies (§2.4/§6): two near-simultaneous replies both
try to resume, but per-query atomicity means only the one that observes `status = 'waiting'` flips it;
the loser sees `running` → `WHERE` fails → **zero rows** → does not re-enter the executor. **Verified:**
first resume returns the row, an immediate second returns zero rows. Clears `waitingThreadId` so the run
is no longer discoverable as parked.*

### 12.5 `complete_run` / `fail_run` — terminal states (clear `AT_STEP`)

```cypher
// complete_run — $runId, $endedAt
MATCH (r:WorkflowRun {runId: $runId})
OPTIONAL MATCH (r)-[atRel:AT_STEP]->()
DELETE atRel
SET r.status = 'done', r.endedAt = $endedAt
RETURN r.runId AS runId, r.status AS status
```
```cypher
// fail_run — $runId, $endedAt, $ctx (executor stamps a note, e.g. "step budget exceeded", §7)
MATCH (r:WorkflowRun {runId: $runId})
OPTIONAL MATCH (r)-[atRel:AT_STEP]->()
DELETE atRel
SET r.status = 'failed', r.endedAt = $endedAt, r.ctx = $ctx
RETURN r.runId AS runId, r.status AS status
```
*`AT_STEP` ("current position") is cleared on terminal states — the audit trail is preserved by the
`HAS_STEP_RUN` set + `NEXT` chain + `LAST_STEP_RUN` (the *last executed* step is `LAST_STEP_RUN`-[:RAN]->).
`DELETE` of a **null** `OPTIONAL MATCH`ed edge is a verified no-op (re-completing a run that already has
no `AT_STEP` does not error). **Step-budget fail (§7):** the executor compares `stepCount` (returned by
§12.2) to `maxSteps` app-side; on `stepCount > maxSteps` it calls `fail_run` — verified `failed` +
`AT_STEP` cleared + `StepRun`s retained. Zero rows = run not found (→ `WorkflowRunNotFoundError`).*

> **`maxSteps` is a tripwire checked *after* each recorded step, not a hard cap (K-031).** Because
> the comparison is `stepCount > maxSteps`, a run executes at most **`maxSteps + 1`** steps before
> failing with `"step budget exceeded"`. The check runs only on the two driving outcomes — a guard
> fired (OUTCOME A, `executor.py:410`) and a legitimate self-loop (OUTCOME C, `:427`) — and is
> **deliberately not applied on the park path** (OUTCOME B; a parked run cannot self-drive) **or on
> the terminal path**. Treat it as a safety bound, not an SLA or a cost budget. Making it exact
> (`>` → `>=`) lands inside the SHA-locked `_drive_loop`; filed as proposed **K-033**.

### 12.6 `link_step_emission` — `StepRun -[:PRODUCED]-> Message` (D2)

```cypher
// $stepRunId, $msgId, $toolsUsed — run AFTER the §4 chat write that created the message (two-step, accepted)
MATCH (sr:StepRun {stepRunId: $stepRunId})
MATCH (m:Message  {msgId: $msgId})
MERGE (sr)-[:PRODUCED]->(m)
SET m.toolsUsed = $toolsUsed
RETURN sr.stepRunId AS stepRunId, m.msgId AS msgId
```
*Both endpoints anchor on their `UNIQUE` index (`stepRunId`, `msgId`); `MERGE` on the relationship makes
the link **idempotent** (a retry after a crash between the post and the link re-links exactly once — no
duplicate `PRODUCED`, verified). This is the **second** query of the deliberately two-step emission (post
the message via the guarded §4 write, then link) — the message is the durable artifact; a missing link is
a diagnosable, retry-able gap, not a torn thread (§3/§9). **Distinct from `EMITTED`** (§10) — verified a
`PRODUCED` write adds zero `EMITTED` edges.*

*`$toolsUsed` (K-056, `docs/reviews/salesperson-tool-reliability-ml.md` §4.3) is the producing step's
own non-`post_message` dispatched tool names (`executor.StepResult.toolsUsed`, empty list when the step
dispatched none) — the tool-use breadcrumb source `read_thread` (§4) surfaces on the next turn's replay.
Live-verified: `SET`/round-trip of a list-of-strings property works on this build (`GRAPH.QUERY … SET
n.toolsUsed = ['a','b'] RETURN n.toolsUsed` → `[a, b]`; `coalesce(missing, [])` → `[]`). Always set,
even when empty — an explicit write, not a reliance on the read-side `coalesce` fallback alone.*

### 12.7 `get_run` — read a run's state

```cypher
// $runId
MATCH (r:WorkflowRun {runId: $runId})
OPTIONAL MATCH (r)-[:AT_STEP]->(cur:Step)
OPTIONAL MATCH (r)-[:OF_DEF]->(snap:WorkflowDefSnapshot)
RETURN r.runId AS runId, r.status AS status, r.stepCount AS stepCount, r.maxSteps AS maxSteps,
       r.trace AS trace, r.ctx AS ctx, r.startedAt AS startedAt, r.endedAt AS endedAt,
       r.waitingThreadId AS waitingThreadId,
       cur.key AS atStepKey, snap.key AS defKey, snap.version AS defVersion
```
*Point lookup on `WorkflowRun.runId`. `atStepKey` is `null` for terminal runs (§12.5). Route via
`GRAPH.RO_QUERY`.*

### 12.8 `read_step_runs` — the NEXT-ordered audit trail

```cypher
// $runId
MATCH (r:WorkflowRun {runId: $runId})-[:HAS_STEP_RUN]->(sr:StepRun)
OPTIONAL MATCH (pv:StepRun)-[:NEXT]->(sr)
WITH sr, pv WHERE pv IS NULL                    // the head = the one StepRun with no NEXT predecessor
MATCH (sr)-[:NEXT*0..]->(x:StepRun)
RETURN x.stepRunId AS stepRunId, x.stepKey AS stepKey, x.status AS status,
       x.startedAt AS startedAt, x.endedAt AS endedAt, x.input AS input, x.output AS output,
       x.resolvedModel AS resolvedModel, x.modelSource AS modelSource,
       x.modelFallback AS modelFallback
ORDER BY x.startedAt
```
*Anchors on `Node By Index Scan | (r:WorkflowRun)`, finds the chain head via **`OPTIONAL MATCH` +
`IS NULL`** (never the broken `exists()`-in-pattern check — quirks KB), then walks `NEXT*0..`. Ordered by
the executor's monotonic `startedAt` (same lock-guarded clock as messages — ties impossible at source),
which coincides with `NEXT` order. Route via `GRAPH.RO_QUERY`.*

**K-042 Landing 2 (FR-8, `docs/plans/llm-provider-config-graph.md` §1.7).** The three new
columns project `null` for a `StepRun` that never had them written — a non-LLM step, or a
pre-Landing-2 row; both read back identically (no backfill). This is the read surface behind
`GET /workflow-runs/{id}/step-runs`, which serializes the returned dicts verbatim — no
`schemas.py` pydantic model gates this route, so the three keys reach the client with no
further code change once the repository projects them.

### 12.9 `find_waiting_run_for_thread` — the resume lookup (index-anchored)

```cypher
// $threadId — resume a parked run when a human replies in its thread (§2.4/§6)
MATCH (r:WorkflowRun {status: 'waiting'})
WHERE r.waitingThreadId = $threadId
RETURN r.runId AS runId, r.status AS status
LIMIT 1
```
*Anchors on `Node By Index Scan | (r:WorkflowRun)` via the **existing `WorkflowRun.status` index**
(point lookup on value `'waiting'`), then a residual `Filter` on the denormed `waitingThreadId`.
**No new index** — the `waiting` set is tiny (at most a handful of parked conversations per workspace;
the value-index visits only `waiting` nodes, never the accumulating `done` runs), so the residual filter
is trivial. **Decision (this gate):** the `TRIGGERED_BY`→`Message` traversal alternative also
index-anchors on `status` but adds a `Conditional Traverse` and depends on the trigger edge surviving —
the denorm is simpler and self-contained. RAM: **zero new index**; `waitingThreadId` is one short
string property per run. Route via `GRAPH.RO_QUERY`. (A thread holds at most one `waiting` run at a time
— `LIMIT 1` is belt-and-suspenders.)*

### 12.10 `append_trace_event` — write one debug trace record (FR-4)

```cypher
// $stepRunId, $traceId (server-minted), $seq (order within the StepRun), $kind, $at, $payload
// Called ONLY when the run is a debug instance (WorkflowRun.trace = true) — the GraphTracer;
// the NullTracer (non-debug) issues no query, so a lean run writes zero TraceEvent nodes.
MATCH (sr:StepRun {stepRunId: $stepRunId})
CREATE (te:TraceEvent {traceId: $traceId, seq: $seq, kind: $kind, at: $at, payload: $payload})
CREATE (sr)-[:TRACED]->(te)
RETURN te.traceId AS traceId
```
*`kind ∈ {node_rationale, guard_judgment, tool_call, tool_result, graphrag_retrieval, llm_prompt,
llm_response, step_timing}` (DESIGN §5, app-enforced — opaque in-graph). `payload` is a flat serialized
string, length-capped at the write boundary (rule 6). Backed by the `TraceEvent.traceId` UNIQUE
constraint.*

### 12.11 `read_trace` — reconstruct a run's execution (debug)

```cypher
// $runId
MATCH (r:WorkflowRun {runId: $runId})-[:HAS_STEP_RUN]->(sr:StepRun)-[:TRACED]->(te:TraceEvent)
RETURN sr.stepRunId AS stepRunId, sr.stepKey AS stepKey,
       te.traceId AS traceId, te.seq AS seq, te.kind AS kind, te.at AS at, te.payload AS payload
ORDER BY sr.startedAt, te.seq
```
*Anchors on `Node By Index Scan | (r:WorkflowRun)`, traverses to step-runs then their trace events,
ordered by `(StepRun.startedAt, TraceEvent.seq)` — the full cross-step reconstruction (FR-4). A
non-debug run has zero `TRACED` edges → empty result (AC-5's negative half, by construction). Route via
`GRAPH.RO_QUERY`.*

### 12.12 `start_run_untriggered` — begin a run with **no** chat trigger message

Parent: **§12.1** (`start_run`). Identical, **minus** the `MATCH (trigger:Message …)` anchor and the
`CREATE (r)-[:TRIGGERED_BY]->(trigger)` edge. A `kind:'process'` run (K-024) is started from
REST/API — there is no `Message`, no `Thread`, and therefore no trigger to link. Deliberately a
**second, self-contained write path** rather than an `OPTIONAL MATCH` + `FOREACH` conditional inside
§12.1: same doctrine as the §4 first/subsequent message paths, and it sidesteps the
empty-row-collapse class of bug entirely.

```cypher
// $runId,$defKey,$defVersion,$startedAt server-minted / caller-supplied
// $ctx = opaque serialized state ("{}" or the caller's initial run ctx — reserved keys
//        threadId/error/timerFired are rejected service-side, see plan §3.4 M-2. The
//        SAME check now guards the §12.1 chat path too, hoisted ahead of both branches.)
// $trace = bool; $maxSteps = run-level step budget (a process def declares its own, e.g. 24).
//                A tripwire checked AFTER each step ⇒ at most maxSteps + 1 (§12.5 note).
MATCH (snap:WorkflowDefSnapshot {key: $defKey, version: $defVersion})-[:START]->(start:Step)
CREATE (r:WorkflowRun {runId: $runId, defKey: $defKey, defVersion: $defVersion,
                       status: 'running', startedAt: $startedAt, ctx: $ctx,
                       trace: $trace, maxSteps: $maxSteps, stepCount: 0,
                       waitingThreadId: ''})
CREATE (r)-[:OF_DEF]->(snap)
CREATE (r)-[:AT_STEP]->(start)
RETURN r.runId AS runId, start.key AS startKey, r.status AS status, r.stepCount AS stepCount
```

*Single anchor ⇒ **zero rows = the snapshot has no `START`** (or the `(key, version)` pair does not
exist), and **nothing is written** — verified: the response carries no `Nodes created` and no
`WorkflowRun` is left behind. Backed by the `WorkflowRun.runId` UNIQUE constraint (server-minted id ⇒
plain `CREATE`, constraint as the concurrency backstop). `waitingThreadId` starts `''` and stays `''`
for a process run — it has no thread, which is exactly why §12.9's thread lookup must never be called
with an empty `threadId` (plan F-5/F-6).*

**`GRAPH.PROFILE` (2026-07-20, v4.18.11 / module `41811`, `ws:test`)** — one `Node By Index Scan` on
`WorkflowDefSnapshot.key`, **no label scan**:

```
Results | Records produced: 1
    Project | Records produced: 1
        Create | Records produced: 1
            Create | Records produced: 1
                Create | Records produced: 1
                    Conditional Traverse | (snap)->(start:Step) | Records produced: 1
                        Node By Index Scan | (snap:WorkflowDefSnapshot) | Records produced: 1
```

### 12.13 `resume_run_with_ctx` — guarded CAS `waiting → running` **that also writes `ctx`**

Parent: **§12.4** (`resume_run`), which **remains in use unchanged** for the chat/trigger resume path
(`trigger.py`, where no ctx is submitted). This variant is §12.4 **plus one `SET` term** and is the
human/signal-input path for a `process` run (decision **D-F**): the submitted input, already merged
into the run ctx service-side, rides **inside** the CAS so the write and the flip cannot be split.

```cypher
// $runId; $ctx = the FULL merged run ctx (opaque serialized string, rule 8) — the service
// reads the current ctx, merges the validated input flat into it, and passes the result
MATCH (r:WorkflowRun {runId: $runId})
WHERE r.status = 'waiting'
SET r.status = 'running', r.waitingThreadId = '', r.ctx = $ctx
RETURN r.runId AS runId, r.status AS status
```

**Zero-row contract (live-verified — this is what D-F rests on, do not assume it):** a run that is not
`waiting` matches the node but fails the `WHERE` ⇒ **zero rows and NOTHING is written — neither the
status flip nor the ctx**. Verified by replaying the CAS with a marker ctx (`{"decision":"LOSER"}`)
against an already-`running` run: zero rows returned and `r.ctx` still holds the winner's value. So
only the CAS **winner's** ctx is ever persisted — "which input advanced the run" and "which input is in
`ctx`" can never disagree. A missing `runId` is likewise zero rows (the service distinguishes the two
cases with a prior `get_run`: `None` ⇒ 404, present-but-not-waiting ⇒ 409). The loser's input is
**rejected, never silently lost**.

**`GRAPH.PROFILE` (2026-07-20)** — point lookup on `WorkflowRun.runId`, **no label scan**; the
`status` predicate is folded **into** the index scan (no residual `Filter` operator), so a
non-matching run produces zero records at the scan itself:

```
# winner (run is waiting)                     # loser (run already running)
Results | Records produced: 1                 Results | Records produced: 0
    Project | Records produced: 1                 Project | Records produced: 0
        Update | Records produced: 1                 Update | Records produced: 0
            Node By Index Scan | (r:WorkflowRun)        Node By Index Scan | (r:WorkflowRun)
              | Records produced: 1                       | Records produced: 0
```
*Anchoring confirmed on `runId`, not `status`: with five other `waiting` runs in the graph the scan
still produced exactly 1 record. (`WorkflowRun` carries RANGE indexes on both `runId` and `status`.)*

**No DDL, no new index, ≈ zero RAM (rule 6).** Both queries reuse what `bootstrap_schema.sh` already
creates in every workspace: `WorkflowDefSnapshot.key` (:117), `.version` (:120), `WorkflowRun.runId`
(:123) and the `WorkflowRun` UNIQUE `{runId}` constraint (:179). No new label, no new property, no new
index — **`bootstrap_schema.sh` is not touched**. The only RAM delta is a longer `ctx` string on
`WorkflowRun` (tens of bytes of merged human input per run), on a node type that is rare compared to
`Message`; §12.12 in fact stores *less* than §12.1 (one fewer relationship — no `TRIGGERED_BY`).

**Live-verified build quirks that shape §12** (all confirmed on `falkordb:v4.18.11`, module `41811`,
against an isolated `ws:gdbtest`):
- **The tail-pointer advance (§12.2) plans edge-anchored** — the previous `StepRun` is found by
  `Optional Conditional Traverse` of `LAST_STEP_RUN`, not a scan of the `NEXT` chain. This is what makes
  the atomic advance O(1) regardless of trail length (M4).
- **`DELETE` of a null `OPTIONAL MATCH`ed relationship is a no-op** (§12.5) — no guard needed to
  re-complete an already-terminal run.
- **Guarded CAS via `WHERE` on the current status value** (§12.3/§12.4) gives single-flight
  suspend/resume without a lock — per-query atomicity serializes the read-modify-write.
- **`waitingThreadId` on `WorkflowRun` rides the `status` index** (§12.9) — a value-point index scan on
  `status:'waiting'` + a residual property filter, no dedicated `waitingThreadId` index.

### 12.14 `find_runs_for_thread` — every run this thread has ever had (K-036 — web-api-coverage FR-2)

```cypher
// $threadId, $limit
MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(m:Message)
WHERE r.startedAt >= 0 AND m.threadId = $threadId
RETURN r.runId AS runId, r.status AS status, r.defKey AS defKey,
       r.defVersion AS defVersion, r.startedAt AS startedAt, r.endedAt AS endedAt
ORDER BY r.startedAt DESC
LIMIT $limit
```

*The `r.startedAt >= 0` conjunct is functionally a no-op — `startedAt` is always a non-negative
epoch-ms timestamp (§12.1/§12.12) — but it is **load-bearing for the query plan**, not decoration.
See the PROFILE findings below.*

**PROFILE finding — a genuinely new, previously-undocumented planner fact (verified 2026-07-28,
v4.18.11 / module `41811`, isolated `ws:gdbtest`, 3 `WorkflowRun` vs up to 20,003 `Message`).**
The plan's originally-proposed query shape (`docs/plans/web-api-coverage.md` §3.1a — no predicate
on `r`, only `WHERE m.threadId = $threadId`) does **not** anchor on
`Node By Label Scan | (r:WorkflowRun)` as that section expected. It anchors on
`Node By Label Scan | (m:Message)` instead — scanning **every `Message` in the workspace**, not
just the thread's, before filtering the unindexed `threadId` property and traversing back to `r`:

```
Conditional Traverse | (r)->(m) | Records produced: 3
    Filter | Records produced: 3
        Node By Label Scan | (m:Message) | Records produced: 20003
```

This holds regardless of `MATCH` clause shape — tested single-`MATCH`, split two-`MATCH`
(`MATCH (r:WorkflowRun) MATCH (r)-[:TRIGGERED_BY]->(m:Message) WHERE …`), and reversed direction
(`MATCH (m:Message)<-[:TRIGGERED_BY]-(r:WorkflowRun) WHERE …`) — all four anchor on `m`. The
mechanism: **a `WHERE` predicate on a pattern variable pulls the label-scan anchor onto that
variable's label, even when a much smaller, filter-free label sits elsewhere in the same
pattern** — relative cardinality does not decide the anchor here, "which variable carries a
`WHERE` predicate" does. Confirmed the inverse too: the identical pattern with **no** `WHERE` at
all correctly anchors on the smaller `WorkflowRun` label
(`Node By Label Scan | (r:WorkflowRun) | Records produced: 3`). Promoted to the general quirks KB
(`claude/graph-dba/falkordb-quirks.md`, "Query tuning" — this is an engine fact, not specific to
this schema).

**The plan's own proposed fallback — a bare `WorkflowRun.startedAt` range index, added with no
query change — is confirmed a no-op**, exactly as the plan's v2 caveat (§3.1a) warned might
happen: adding the index alone did not move the anchor off `Message` at all (identical profile,
label scan still on `m`, index unused).

**What actually redirects the anchor: a second, functionally-vacuous predicate on `r` —
`WHERE r.startedAt >= 0`.** Verified in three configurations:
- No index + the predicate → `Node By Label Scan | (r:WorkflowRun)` (small-label scan, 3
  records, not the 20,003-record `Message` scan) → `Conditional Traverse` → `Filter` on
  `threadId`.
- `WorkflowRun.startedAt` range index present + the predicate →
  **`Node By Index Scan | (r:WorkflowRun)`** — genuinely used, not a no-op once paired with the
  predicate.
- Predicate order in `WHERE` doesn't matter (`m.threadId = … AND r.startedAt >= 0` profiles
  identically to the reverse order).

**Decision (this gate): ship both** — the `WHERE r.startedAt >= 0` predicate (does the real work:
moves the anchor off the workspace-wide `Message` scan) **and** the `WorkflowRun.startedAt` range
index (upgrades that small-label scan to a small-label **index** scan, keeping this query
consistent with this file's own "every read anchors on an index, no label scan" convention,
line 1007). Full PROFILE with the index in place:

```
Results | Records produced: 3
    Limit | Records produced: 3
        Sort | Records produced: 3
            Project | Records produced: 3
                Filter | Records produced: 3
                    Conditional Traverse | (r)->(m:Message) | Records produced: 3
                        Node By Index Scan | (r:WorkflowRun) | Records produced: 3
```

**RAM (rule 6): one new range index, `WorkflowRun.startedAt`, added to `bootstrap_schema.sh`.**
`WorkflowRun` cardinality is tiny per workspace (same argument §12.9 already accepted for the
`status` index) — cost is a few bytes per run, negligible next to `Message`-scale RAM.

**General engine fact, not project-specific:** any FalkorDB two-hop pattern where one side is a
small, unfiltered label and the other is a much larger label filtered by an **unindexed**
property in `WHERE` can hit this same anchor trap — the fix (an extra, even-if-vacuous predicate
on the variable you want as anchor) generalizes. See `claude/graph-dba/falkordb-quirks.md` for the
schema-independent write-up.

### 12.15 `read_recent_post_success` — last-N post-success sample for the `@mention` def (K-039 item 3)

```cypher
// $defKey, $defVersion, $limit
MATCH (r:WorkflowRun)
WHERE r.startedAt >= 0
  AND r.defKey = $defKey AND r.defVersion = $defVersion
  AND r.status IN ['done', 'failed']
WITH r ORDER BY r.startedAt DESC LIMIT $limit
OPTIONAL MATCH (r)-[:HAS_STEP_RUN]->(:StepRun)-[:PRODUCED]->(m:Message)
WITH r, count(m) AS producedCount
RETURN count(r) AS sampleSize,
       sum(CASE WHEN producedCount > 0 THEN 1 ELSE 0 END) AS postedCount
```

Feeds the K-036 readiness route's new `postSuccess` field (`docs/plans/mention-reply-delivery.md`
§3.3): of the last `$limit` **terminal** (`done`/`failed`) runs of the `@mention`-triggered def
(`config.TRIGGER_DEF_KEY`@`config.TRIGGER_DEF_VERSION`, currently `triage`@`v1`), how many produced
at least one reply (`StepRun -[:PRODUCED]-> Message`, D2, §12.6). `waiting`/`running` runs are
excluded from the sample — they haven't reached a verdict yet. Verified live against `ws:test`
(synthetic fixture) and read-only against `ws:acme` (real production data), pinned build
`v4.18.11`/module `41811`.

**Result types (Python side, `falkordb-py` 1.6.x) — live-verified, both the zero-row and
non-empty case:** `sampleSize` is a clean `int` (from `count(r)`); `postedCount` is always a
**Python `float`** (`0.0`, `1.0`, …) — `sum()` over this `CASE` expression **never returns
`NULL`/`None`** in either case. Confirmed via `g.query(...).result_set`:

```
row: [3, 1.0]              # non-empty sample (3 terminal runs, 1 posted)
zero-case row: [0, 0.0]     # unknown defKey/defVersion — "no data," not an exception
```

The repository layer must cast `postedCount` to `int` before returning it — left as a `float`, the
JSON response carries `"postedCount": 1.0` and the readiness banner would render `"1.0/2 replied"`.

**Which index the query actually lands on — not just "an index scan":** `r.defKey`/`r.defVersion`
carry no index (none needed — `WorkflowRun` cardinality is tiny per workspace, §RAM below), so this
query's index-anchor situation is **materially different from §12.14's** (there, the load-bearing
`WHERE r.startedAt >= 0` conjunct exists purely to pull the label-scan anchor off a *different*
pattern variable, `m:Message`; here every filter is on `r` itself and two of them — `startedAt` and
`status` — are independently indexed).

PROFILE against `ws:test` (dedicated fixture, defKey `post_success_probe`, 3 terminal + 1 waiting
run):

```
Results | Records produced: 1
    Aggregate | Records produced: 1
        Aggregate | Records produced: 3
            Optional Conditional Traverse | (r)->(m:Message) | Records produced: 3
                Limit | Records produced: 3
                    Sort | Records produced: 3
                        Project | Records produced: 3
                            Filter | Records produced: 3
                                Node By Index Scan | (r:WorkflowRun) | Records produced: 3
```

No `Node By Label Scan` anywhere (AGENTS.md rule 3) — but the interesting fact is *what the single
`Node By Index Scan` step already excludes before `Filter` ever runs*. Isolated by testing three
variants of the same fixture:

- Query with only `r.startedAt >= 0` (drop the `status` predicate): `Node By Index Scan` alone
  produces all 4 fixture rows (the `waiting` run included) — `startedAt` anchors on its own.
- Query with only `r.status IN [...]` (drop the `startedAt` predicate): `Node By Index Scan` alone
  produces exactly the 3 terminal rows (the `waiting` run excluded) — `status` anchors on its own
  too.
- Both predicates dropped (only the unindexed `defKey`/`defVersion` left): the plan falls back to
  **`Node By Label Scan | (r:WorkflowRun)`** — confirmed neither `defKey` nor `defVersion` can
  anchor anything by themselves.
- **The deciding test:** with the full query (both predicates present) plus one extra probe row
  (`status:'done'`, `startedAt:-5`, `defKey` matching), the `Node By Index Scan` step still
  produces only 3 — the probe row is excluded **at the index-scan step itself**, before `Filter`
  ever runs. If the engine had picked only one of the two indexes as anchor and pushed the other
  predicate into `Filter`, the probe row (which fails only the `startedAt` predicate) would have
  survived to the `Filter` step and been visible in its input count. It wasn't.

**New planner fact, previously undocumented (verified 2026-07-31, v4.18.11/module `41811`):** when
two independently-indexed properties on the same label both appear as `AND`-ed `WHERE` predicates
(one a numeric range, one a `status IN [...]` list), FalkorDB's planner does not pick one as "the"
anchor and filter the other — it folds **both indexed predicates into the single `Node By Index
Scan` step**, and only the genuinely unindexed predicates (`defKey`, `defVersion` here) surface as
a separate `Filter` operator above it. So the honest answer to "which index" is **both
`WorkflowRun.startedAt` and `WorkflowRun.status`, combined** — not an either/or choice, and not
"just an index scan." Cross-checked read-only against `ws:acme`'s real data (11 `WorkflowRun`s,
mixed `triage@v1`/`access-request@v1`): `Node By Index Scan` produced exactly 7 (every `done`/
`failed` row across *both* defs — the `status` half of the compound scan, `startedAt` doesn't
additionally reduce there since every real timestamp is positive), then `Filter` narrowed to the
2 `triage@v1` rows (`sampleSize=2, postedCount=1` — the RCA's own corroborating "1/2 degraded"
case, §7 finding 4). Promoted to the general quirks KB (`claude/graph-dba/falkordb-quirks.md`,
"Query tuning") — this is an engine fact, not specific to this schema.

**RAM (rule 6): none.** No new index, no new label, no new property — the query reuses the
existing `WorkflowRun.status` and `WorkflowRun.startedAt` indexes (`bootstrap_schema.sh:145-156`),
confirmed live (`CALL db.indexes()` on both `ws:test` and `ws:acme` returns exactly
`[runId, status, startedAt]` for `WorkflowRun`, no `defKey`). `WorkflowRun` cardinality is tiny per
workspace (same argument §12.9/§12.14 already accepted), so an unindexed residual `Filter` on
`defKey`/`defVersion` costs nothing worth adding an index for.

### 12.16 `find_due_wait_candidates` — the K-028 sweep's read half (index-anchored)

```cypher
// $limit — every parked `wait`/`human` candidate, due-agnostic (K-028)
// v3: RETURN gained `s.key AS stepKey` (additive projection off the already-bound `s`,
// no new traversal) -- the sweep needs it to write the step-scoped `ctx.timerFired`
// marker and to detect a candidate that moved to a DIFFERENT waiting step between the
// scan and the sweep's per-candidate act (docs/plans/workflow-timers.md §3.4/§3.5 step 5.1).
MATCH (r:WorkflowRun {status: 'waiting'})-[:AT_STEP]->(s:Step)
OPTIONAL MATCH (r)-[:LAST_STEP_RUN]->(sr:StepRun)
RETURN r.runId AS runId, s.key AS stepKey, s.type AS stepType, s.config AS stepConfig,
       sr.startedAt AS parkedAt
LIMIT $limit
```

Feeds `Services.sweep_due_workflow_runs` (`docs/plans/workflow-timers.md` §3.4/§3.5): returns
**every** `waiting` run, regardless of whether it is actually due or even whether its parked step
declares a timer key at all — dueness is derived app-side (`services._wait_due_at`) from
`stepConfig`/`parkedAt`, never filtered in Cypher (rule 8). Anchors on the **existing**
`WorkflowRun.status` value index — the same anchor §12.9 already uses, on the same "waiting set is
tiny" cardinality argument — then two `Conditional Traverse`s (`AT_STEP`, `LAST_STEP_RUN`), the
identical traversal shape `get_run` (§12.7) already uses for the same two edges off an
already-bound node.

**PROFILE finding (live-verified against `ws:test`, one seeded `WorkflowRun {status:'waiting'}`
with an `AT_STEP`/`LAST_STEP_RUN` pair) — re-confirmed after the v3 `stepKey` RETURN-clause
addition, not assumed unchanged:**

```
Results | Records produced: 1, Execution time: 0.000219 ms
    Limit | Records produced: 1, Execution time: 0.000254 ms
        Project | Records produced: 1, Execution time: 0.003554 ms
            Optional Conditional Traverse | (r)->(sr:StepRun) | Records produced: 1, Execution time: 0.080543 ms
                Conditional Traverse | (r)->(s:Step) | Records produced: 1, Execution time: 0.109142 ms
                    Node By Index Scan | (r:WorkflowRun) | Records produced: 1, Execution time: 0.015080 ms
```

`Node By Index Scan | (r:WorkflowRun)` — exactly as expected, no `Node By Label Scan` anywhere
(rule 3), identical shape to the pre-v3 profile (the `s.key` addition is a free projection off the
already-bound `s`, confirmed to add no traversal). The probe fixture was created and torn down
directly against `ws:test` via `redis-cli` (not through the pytest suite), matching the shape
`repository.find_due_wait_candidates` returns.

**RAM (rule 6): zero new indexes, zero new node/relationship types.** The new bytes on the graph,
v3: (1) `config.waitForSeconds`/`waitUntil` on a `wait`/`human` step's already-existing,
already-opaque `config` string (plan §3.4); (2) the reserved `ctx.timerFired: "<stepKey>"` marker
`Services.sweep_due_workflow_runs` writes into `WorkflowRun.ctx` on a genuine timer-triggered
resume, via the existing `resume_run_with_ctx` (§12.13) — a short string, only ever written on the
subset of runs a timer actually fires for. This query itself reuses the `WorkflowRun.status` index
and the `AT_STEP`/`LAST_STEP_RUN` edges unchanged; `s.key` costs nothing extra to project since
`Step.key` is already a stored property read by every other `Step`-touching query in this file.

---

## 13. Workspace configuration — model overrides (K-042 Landing 2, FR-16/FR-17)

A different topic from §12 above: this is workspace-level **configuration**, not an
execution/trace record — a distinct topic gets its own top-level section rather than a strained
§12.N slot. Design: `docs/plans/llm-provider-config-graph.md` §2 (schema §2.2, alternatives §2.3,
write §2.4, read §2.5, placement/timing §2.6); resolver-facing contract §6.1/§6.3. DDL: §4 of the
same document, `scripts/bootstrap_schema.sh` (`bootstrap_workspace()`, index-before-constraint).

A singleton `(:WorkspaceConfig {workspaceConfigId: 'default'})` node per workspace, one nullable
scalar property per consumer kind (`agentModelOverride`, `guardModelOverride`,
`embeddingModelOverride`, `responderModelOverride`) plus two provenance fields. No edges — a
workspace-scoped singleton, not a traversal participant, so it adds no adjacency matrix (rule 7 is
moot here: there is nothing to filter by `workspaceId`, the graph key already *is* the scope).

**Property-name crosswalk, load-bearing and easy to get backwards:** `-graph.md` §8.4 names its
four properties after the **workflow node type** ("agent") vs. the **chat responder class**
("responder") — which does **not** match this server's own `kind` strings 1:1.
`agentModelOverride` governs the executor's `kind="step"` consumer (the workflow's `type:'agent'`
node); `responderModelOverride` governs `kind="agent"` (`AgentResponder`, the `@mention`
responder). `guardModelOverride`/`embeddingModelOverride` match their kinds by name. The crosswalk
lives once, in code, at `server/falkorchat/modelconfig.py`'s `_KIND_TO_OVERRIDE_KEY` — this
section documents the graph's own property names only.

### 13.1 `write_model_overrides` — set/clear the per-kind overrides

```cypher
// [verified] MERGE on the singleton, backed by the WorkspaceConfig.workspaceConfigId
// UNIQUE constraint (§4 below / bootstrap_schema.sh). GRAPH.QUERY.
// $agent/$guard/$embedding/$responder may each be a "<provider>/<model-id>" ref, a
// role name, or NULL (= leave unset / CLEAR an existing override at that kind).
MERGE (c:WorkspaceConfig {workspaceConfigId: 'default'})
SET c.agentModelOverride     = $agent,
    c.guardModelOverride     = $guard,
    c.embeddingModelOverride = $embedding,
    c.responderModelOverride = $responder,
    c.modelOverrideUpdatedAt = $at,
    c.modelOverrideUpdatedBy = $by
RETURN c.agentModelOverride AS agent, c.guardModelOverride AS guard,
       c.embeddingModelOverride AS embedding, c.responderModelOverride AS responder
```

**[verified] `NULL` in `SET` clears the property** (unlike `CREATE`, where a `NULL` param merely
omits the key) — confirmed live against a throwaway probe workspace (`ws:u9probe`, torn down after):
a first write set `agentModelOverride`/`guardModelOverride` only (`Properties set: 5` — the two
overrides + the two provenance fields + `workspaceConfigId` from the `MERGE`/`CREATE`; the two
`NULL` overrides cost nothing), `keys(c)` confirmed `guardModelOverride`/`embeddingModelOverride`/
`responderModelOverride` absent. A **second** write then set `embeddingModelOverride` and passed
`guardModelOverride = NULL` — `Properties removed: 3`, and the follow-up read's `keys(c)` no longer
included `guardModelOverride` at all: the previously-set kind was genuinely cleared, not merely set
to an empty value. `MATCH (c:WorkspaceConfig) RETURN count(c)` stayed `1` across both writes — the
constraint-backed `MERGE` never duplicates the node.

**Never write `''`** — an empty string is a *value* ("a model literally named empty"), never "no
override"; that representation is `NULL`/property-absent alone (§2.4's own explicit warning). Not
enforced in Cypher or in `repository.write_model_overrides` — a caller discipline, same as the
"nullable, absent by default" contract §12.2's `resolvedModel`/`modelSource`/`modelFallback`
already rely on.

### 13.2 `read_model_overrides` — the per-kind overrides, one code path for "unset"

```cypher
// [verified] Node By Index Scan | (c:WorkspaceConfig) — 0.008ms (PROFILE, below).
// GRAPH.RO_QUERY (replica-safe).
MATCH (c:WorkspaceConfig {workspaceConfigId: 'default'})
RETURN c.agentModelOverride     AS agentModel,
       c.guardModelOverride     AS guardModel,
       c.embeddingModelOverride AS embeddingModel,
       c.responderModelOverride AS responderModel
```

**[verified] zero rows and an all-`NULL` row are the same answer — "no override at any kind" —
and `repository.read_model_overrides` returns the identical all-`None` dict for both, one code
path, no branching on which case fired.** Confirmed live: before any write, the read returned zero
rows (every existing workspace's starting state); after a write that set only two of the four
kinds, the read returned exactly one row with the other two columns `NULL`. Either way the
resolver's precedence (`modelconfig.ModelGateway.resolve()`, §6.1: "the graph never enforces
precedence — precedence is entirely the resolver's, workspace → own choice → per-kind default")
sees "no override" and falls through to the next rung.

PROFILE (live, `ws:u9probe`):

```
Results | Records produced: 1, Execution time: 0.000160 ms
    Project | Records produced: 1, Execution time: 0.002256 ms
        Node By Index Scan | (c:WorkspaceConfig) | Records produced: 1, Execution time: 0.008047 ms
```

No label scan (rule 3) — anchors on the `WorkspaceConfig.workspaceConfigId` index (`bootstrap_schema.sh`,
§4 below), the same index the write's `MERGE` uses to find its match.

### 13.3 DDL

```bash
# ── workspace-level configuration singleton (K-042 / FR-16), bootstrap_workspace() ──
echo "[index] WorkspaceConfig.workspaceConfigId"
gquery "$g" "CREATE INDEX FOR (n:WorkspaceConfig) ON (n.workspaceConfigId)"

# … alongside the other constraints, after all indexes:
echo "[constraint] WorkspaceConfig unique {workspaceConfigId}"
gconstraint "$g" UNIQUE NODE WorkspaceConfig PROPERTIES 1 workspaceConfigId
```

**[verified]** on the same probe: the index creates (`Indices created: 1`), the constraint attaches
with no error (index-before-constraint — the standing ordering rule), and the write/read above both
behave exactly as documented against it.

**RAM (rule 6): effectively zero.** One node per workspace, six scalar properties, one range index,
one constraint over a single row — below `GRAPH.MEMORY USAGE`'s 1 MB resolution (`-graph.md` §5).
No new relationship type, no new adjacency matrix. `reference` and `identity` are untouched — this
is workspace-scoped state, per DESIGN §1/§3's single-store philosophy.

---

## 14. Documents & Chunks (K-050 M5 Stage 1)

Design: `docs/plans/document-ingestion.md` (architect, §3.2/§2.4/§3.5/§3.6, Stage 1) and
`docs/plans/document-ingestion-graph.md` §2.1/§2.4 (graph-dba — the exact Cypher below, already
live-verified there against a throwaway probe graph). `Document`/`Chunk` node indexes+constraints
and the `Chunk.embedding` vector index are bootstrapped DDL since M2 (`scripts/bootstrap_schema.sh`)
— no schema change for this stage; `Chunk.seq`/`Chunk.documentId` are plain unindexed properties
(read-time ordering / navigation metadata, the same posture as `Message.threadId`).

`Document.text` stores the full original source verbatim (FR-12/AC-9); `Chunk`s are pre-split
app-side (`falkorchat.chunking.split_into_chunks`) and ride in as a `$chunks` list-of-maps
parameter. `Document.sourceKind` is derived server-side from which label resolved the ingesting
actor (`'document'` for `User`, `'agent'` for `Agent`) — never trusted from the caller, the same
posture as `Message.role`. No `dup` guard: per §2.4's deliberate non-idempotent-on-retry posture
(mirrors the channel/thread-creation precedent), a retried call mints a second `Document`.

### 14.1 `create_document` — Document + Chunks + HAS_CHUNK, one guarded write

```cypher
// $documentId/$chunkIds server-minted uuids. $chunks = [{chunkId, text, seq}, ...].
OPTIONAL MATCH (u:User  {userId:  $ingestedBy})
OPTIONAL MATCH (a:Agent {agentId: $ingestedBy})
WITH u, a, coalesce(u, a) AS ingestor, (coalesce(u, a) IS NOT NULL) AS ok
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (d:Document {
    documentId: $documentId, title: $title, text: $text,
    sourceFormat: $sourceFormat,
    sourceKind: CASE WHEN u IS NOT NULL THEN 'document' ELSE 'agent' END,
    status: 'processing', pendingJobs: 0, createdAt: $createdAt
  })
  CREATE (d)-[:INGESTED_BY]->(ingestor)
  FOREACH (ch IN $chunks |
    CREATE (d)-[:HAS_CHUNK]->(:Chunk {
      chunkId: ch.chunkId, text: ch.text, seq: ch.seq, documentId: $documentId
    })
  )
)
RETURN ok AS written, ingestor IS NOT NULL AS ingestorFound
```

**[verified]** (`document-ingestion-graph.md` §2.1, `gdba_probe_docs`): a 3-chunk document with a
known `User` actor writes 1 `Document` + 3 `Chunk` nodes, 1 `INGESTED_BY` + 3 `HAS_CHUNK` edges, all
`seq`/`text`/`chunkId` correctly populated from the `$chunks` list-of-maps parameter. Unknown-actor
case writes nothing (`written=false, ingestorFound=false`) — same "unknown actor ⇒ silent no-op,
guarded by a status row" contract §4's message write paths use, so the service can distinguish it
from a genuine fault and raise `UnknownActorError` rather than reporting false success.

### 14.1a `Document.status` background-completion tracking (K-051)

`create_document` (§14.1) seeds `pendingJobs: 0`, but never itself writes `'ready'`/`'failed'` —
nothing did, before K-051 (`docs/test-reports/document-ingestion-report.md` Defect 1: a fully-
processed document read back `status: "processing"` forever). These two queries, both on
`repository.py`'s `Repository` (internal seam — `background._schedule_chunk_processing` and
`background._safe_embed_chunk`/`_safe_extract` are the only callers, never a transport directly),
are what closes that gap.

**`start_document_progress` — initialize the outstanding-job counter**
```cypher
MATCH (d:Document {documentId: $documentId})
SET d.pendingJobs = $totalJobs
WITH d
FOREACH (_ IN CASE WHEN $totalJobs <= 0 AND d.status = 'processing'
  THEN [1] ELSE [] END | SET d.status = 'ready')
```
*Run from `background._schedule_chunk_processing`, synchronously, before scheduling any per-chunk
job — `$totalJobs` is the number of `_safe_embed_chunk`/`_safe_extract` calls about to be scheduled
(one per chunk per wired worker), so `report_document_job_done` below always has a real count to
decrement against, never a race with the jobs themselves. `$totalJobs <= 0` (a zero-chunk document)
flips `status` straight to `'ready'` in the same query instead of leaving it parked forever waiting
for a completion report that will never come.*

**`report_document_job_done` — one job's outcome, first-terminal-write-wins**
```cypher
MATCH (d:Document {documentId: $documentId})
SET d.pendingJobs = coalesce(d.pendingJobs, 0) - 1
WITH d
FOREACH (_ IN CASE WHEN NOT $success AND d.status = 'processing'
  THEN [1] ELSE [] END | SET d.status = 'failed')
FOREACH (_ IN CASE WHEN $success AND d.pendingJobs <= 0 AND d.status = 'processing'
  THEN [1] ELSE [] END | SET d.status = 'ready')
```
*Run from `background._safe_embed_chunk`/`_safe_extract` after every chunk-level embed/extract job,
success or failure (via their shared `_report_document_job` helper) — decrements the counter, then
flips `status` to `'failed'` the instant any single job fails (regardless of how many others are
still outstanding — one failed chunk fails the whole document), or to `'ready'` once every scheduled
job has succeeded and none has failed. Both `FOREACH` guards check `status = 'processing'` first, so
the first terminal write wins: a late success after a failure, or a stray failure after the document
already reached `'ready'`, both no-op rather than flipping status back. One `GRAPH.QUERY` per call —
decrement + both guarded flips in the same round trip, so no read-then-write race exists across
separate calls either.*

**[verified]** (`tests/test_repository.py`, live against `falkordb-dev`): all-jobs-succeed reaches
`'ready'` only once the counter is fully drained, not before; a single failure reaches `'failed'`
immediately even with other jobs still outstanding; a late success after a failure does not revert
`'failed'`; a late failure after `'ready'` does not revert it either; `total_jobs <= 0` reaches
`'ready'` immediately and, same guard, does not resurrect an already-`'failed'` document.

### 14.2 `get_document` — read back the verbatim text + ingesting actor

```cypher
MATCH (d:Document {documentId: $documentId})
OPTIONAL MATCH (d)-[:INGESTED_BY]->(actor)
RETURN d.documentId AS documentId, d.title AS title, d.text AS text,
       d.sourceFormat AS sourceFormat, d.sourceKind AS sourceKind,
       d.status AS status, d.createdAt AS createdAt,
       labels(actor)[0] AS ingestedByKind,
       coalesce(actor.userId, actor.agentId) AS ingestedById
```

Same shape as `document-ingestion-graph.md` §2.1's `get_document` note, projected as explicit
properties (this repository's own convention throughout — e.g. `get_message` §4 — rather than
returning a whole node, which the graph note's own illustrative Cypher does but no other query in
this library relies on). `AC-9`: `text` round-trips byte-identical to the ingested input.

### 14.3 Chunk embeddings + `search_chunks` (K-050 M5 Stage 2, FR-3)

Mirrors §6's `Message` embedding pattern exactly, `Chunk` in place of `Message` — same
decoupled-from-the-write-path posture (a chunk is readable before its embedding lands, the same
eventually-consistent posture already used for a posted message) and the same §6.1 FR-19
pre-flight dimension guard, gated per-label: a `Chunk` embed consults only `Chunk`'s vector index,
never `Message`'s, and vice versa. **No new DDL** — `Chunk.embedding`'s vector index has existed
since M2 (§2.1 above), dormant until this stage populates it.

**Set a chunk's embedding (async, after ingestion)**
```cypher
MATCH (c:Chunk {chunkId: $chunkId})
SET c.embedding = vecf32($embedding)
```
*Run from `EmbeddingWorker.embed_chunk` right after `ingest_document` returns — same
decoupled-from-the-write-path posture as §6's message embed.*

**`search_chunks` — the FR-3 standalone-KB-search read**
```cypher
CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qVec))
YIELD node AS seed, score
RETURN seed.chunkId, seed.text, seed.documentId, seed.seq, score
ORDER BY score ASC
LIMIT $limit
```
No scope traversal (a `Chunk` has no Thread/Channel to scope by, unlike §6's Message-seeded
`hybrid_search`) and no Entity co-occurrence expansion: `(:Chunk)-[:ABOUT]->(:Entity)` is dormant
until extraction lands (Stage 3), so an `OPTIONAL MATCH` on it here would just no-op today —
deferred rather than added speculatively; wire it in when Stage 3 makes it meaningful.
`documentId` rides denormalized on `Chunk` itself (plan §3.2), so a search hit reports its source
document with no extra hop. `score` is cosine distance (0 = identical) — rows come back already
`ORDER BY score ASC` (most similar first), same convention as §6; do not re-sort.

`Repository.set_chunk_embedding`/`Repository.search_chunks`
(`server/falkorchat/repository.py`); `EmbeddingWorker.embed_chunk`
(`server/falkorchat/embedding.py`).

**Chat-grounding merge (K-050 M5 Stage 5, FR-2).** `Services.hybrid_search` calls this query and
§6's `hybrid_search` with the same `$qVec`/`$k`, then merges the two ranked result lists **app-side**
by `score` ascending and truncates to `limit` — there is no combined-ANN Cypher shape for this (the
`EMITTED` generalization above, §10, is the only graph-side change FR-2 needed). `channel_id` scopes
only the `Message` pool (§6); a `Chunk` seed has no channel/thread to scope by, so a channel-scoped
call still searches the full `Chunk` ANN index unscoped. Each merged item is tagged app-side
`seedKind: 'Message'|'Chunk'` so `AgentResponder.maybe_respond` can resolve the right id generically
before it flows into §10.1's `$seedIds`. Tie-break: `Message` pool concatenated first, `list.sort`
stable, so an exactly-equal score keeps `Message` ahead of `Chunk`. The two RO queries run
sequentially (this codebase's existing convention — no other RO-query pair here fires concurrently),
roughly doubling worst-case retrieval latency; re-measure once real ingestion volume exists.

**`list_document_chunks` — internal seam, not a public query**

```cypher
MATCH (d:Document {documentId: $documentId})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.chunkId, c.text
ORDER BY c.seq
```

Not part of the MCP/REST surface (§14.4) — `api.py`/`mcp.py` call it right after
`services.ingest_document` returns, to fetch the just-created chunks and schedule each one for
background embedding (`_safe_embed_chunk`). Kept out of `ingest_document`'s own return value
deliberately: that receipt stays at the documented `{documentId, chunkCount, status}` shape
(§14.4) rather than echoing up to ~500,000 characters of chunk text back into the response body.
Always traversed from an already-anchored `Document`, never independently scanned — no new index
needed, same posture as `ABOUT`/`RELATES_TO` (`document-ingestion-graph.md` §2.2).

### 14.4 MCP / REST surface (Stages 1-2 slice of plan §3.5, bulk row added Stage 6a)

| MCP tool | REST | Service method |
|---|---|---|
| `ingest_document(text, title=None, source_format="text", source_label=None)` | `POST /documents` | `services.ingest_document` |
| `ingest_documents(items: list[dict])` | `POST /documents/batch` | `services.ingest_documents` |
| `get_document(document_id)` | `GET /documents/{id}` | `services.get_document` |
| `search_documents(query, limit=20)` | `GET /documents/search?q=` | `services.search_documents` |

`ingest_document` stamps the ingesting actor from `get_context()` (FR-4) — MCP ignores any
client-supplied actor, exactly the existing `send_message` posture. `MAX_DOCUMENT_CHARS = 500_000`
is enforced in `services.ingest_document` itself (`DocumentTooLargeError`, maps to 400), not only at
the REST pydantic boundary — an MCP caller has no schema layer, so this is the one place both
transports are bound by the same cap.

`search_documents` embeds `query` through the injected `ModelGateway` (mirrors
`GraphragRetrieveTool`/`AgentResponder`'s own text→vector step) then calls `search_chunks` (§14.3)
above. Raises `SearchNotAvailableError` (maps to REST 503, mirroring `WorkflowEngineDisabledError`)
when no `ModelGateway` is wired into this deployment — a configuration gap, not a caller mistake.
`GET /documents/search` is registered **before** `GET /documents/{document_id}` in `api.py` —
Starlette matches routes in registration order, and the dynamic path would otherwise swallow the
literal `search` segment as a `document_id`.

**`ingest_documents` (FR-11 bulk ingestion, K-050 M5 Stage 6a, plan §3.6).** Loops
`services.ingest_document` per item and returns **one receipt per item**, in the same order as
the input — no new Cypher, no batch-aware fusion logic: each item is written through the exact
same §14.1 `create_document` write path, and cross-document fusion (AC-8) falls out naturally
once each item's independent background extraction runs, because §1.5/§1.7's fuzzy/exact-match
lookups always read the graph's *current* state (including sibling documents from the same
batch, once their entities land), never a batch-local view. Capped at `MAX_BATCH_SIZE = 20`
(`BatchTooLargeError`, maps to REST 400) — enforced in the service, not only at the REST
`IngestDocumentsIn` pydantic boundary, mirroring `MAX_DOCUMENT_CHARS`'s posture above. **Per-item
failure isolation:** one bad item (empty text, oversized text, unknown actor) does not abort the
batch — it comes back as that item's own `{"status": "error", "error": ..., "errorType": ...}`
receipt, and chunk embedding/extraction is scheduled only for the items that actually succeeded
(`services.py`'s `ingest_documents` docstring has the full reasoning). `POST /documents/batch` is
registered **before** `GET /documents/{document_id}` in `api.py`, same static-before-dynamic
reason as `/documents/search` above. **A malformed item** (missing/non-string `text`, or a
non-dict entry — only reachable via MCP, since `IngestDocumentIn` guarantees `text: str` at the
REST boundary) is isolated the same way, not a bare `KeyError`/`TypeError` escaping the batch
(Pass 6 review BLOCKER fix) — it comes back as `{"status": "error", "errorType":
"MalformedItemError", ...}`.

`ingest_document`'s and `ingest_documents`' identical per-chunk embed+extract scheduling block —
previously duplicated inline between `api.py` and `mcp.py` — is now one shared helper,
`background._schedule_chunk_processing(schedule, ws, document_id, chunks, *, embed_worker,
ingestion_pipeline)`, parameterized over each transport's own scheduling primitive
(`BackgroundTasks.add_task` for REST, `mcp._schedule` for MCP) so a third call site (the batch
routes above) didn't triple the duplication.

**MCP thread fan-out compounds by up to 20x on the batch path (Pass 6 review MAJOR).** REST's
`BackgroundTasks` scheduling is unaffected (cheap, bounded via anyio's worker-pool limiter), but
MCP's `ingest_documents` calls the shared helper once per successfully-ingested item in a plain
sequential loop, so `_default_schedule`'s already-flagged per-document fan-out (`mcp.py`'s own
docstring, Stage 3: ~1,000-1,200 raw OS threads for a max-size document) now multiplies by up to
`MAX_BATCH_SIZE = 20` — **~23,000 threads** for a max-size batch, sequentially, synchronously,
inside one tool call, before it returns. `MAX_BATCH_SIZE` bounds the multiplier but does not
shrink the per-call number; every per-thread failure is still isolated (`_default_schedule`'s
try/except). Not re-mitigated in Stage 6a beyond this documentation — a bounded thread pool is the
real fix, deferred (see `mcp.py`'s `_default_schedule` docstring for the full reasoning) pending a
coordinator scope decision.

### 14.5 Entities & `RELATES_TO` (K-050 M5 Stage 3, FR-7a)

Design: `docs/plans/document-ingestion.md` §3.3 (architect, extraction recommendation),
`docs/plans/document-ingestion-ml.md` §3 (data-scientist, extraction technique/prompt/schema —
adopted as-is) and `docs/plans/document-ingestion-graph.md` §2.2/§2.3 (graph-dba, the exact Cypher
below, already live-verified there). One LLM call per chunk (`falkorchat.extraction.extract`)
produces a `{entities: [{name, type}], relationships: [{subject, predicate, object}]}` payload —
parsed via the same fence-tolerant `llm.extract_own_line_json_object` the K-027 guard judge uses,
then independently schema-validated app-side (the parser's `require_key` alone does not reject a
malformed top-level shape — ML note F1). `falkorchat.ingestion.IngestionPipeline` writes the result:
a fresh `Entity` per mention (including any subject/object stub-repaired by `extraction.py`) plus
its `ABOUT` edge, and one `RELATES_TO` edge per extracted relationship. **No fusion yet** (plan
§3.1/§3.3) — every mention is a brand-new node, even a repeat of one already in the graph; that's a
later stage, not a defect here.

**`create_entity` — always a NEW node**
```cypher
// $nameNormalized = case-folded + whitespace-collapsed $name, computed app-side by the
// SAME normalization helper (`extraction.normalize_name`) extraction's own subject/object
// stub-repair uses — one shared function, not two independently-written normalizers.
CREATE (e:Entity {
  entityId: $entityId, name: $name, nameNormalized: $nameNormalized,
  type: $type, createdAt: $createdAt
})
RETURN e.entityId AS entityId
```
Never looks up or reuses an existing entity — fusion (a future stage) never blocks creation, only
decides *linking* after the fact. Plain `CREATE`, no guard-and-status-row contract: both the
`Entity` id and every edge below are freshly minted moments apart by *this same pipeline run*, so a
`MATCH` miss downstream would indicate a real bug, not routine caller input (same posture as
`create_document`'s `HAS_CHUNK` writes, §14.1).

**`link_chunk_about_entity` — the dormant `ABOUT` edge, now populated**
```cypher
MATCH (c:Chunk {chunkId: $chunkId})
MATCH (e:Entity {entityId: $entityId})
CREATE (c)-[:ABOUT]->(e)
```
Scaffolded since M2 (`docs/DESIGN.md` §5.1) and never written until this stage. Plain `CREATE`,
never `MERGE` — a chunk is extracted exactly once per ingestion, so a duplicate `(chunk, entity)`
pair shouldn't occur under normal operation; if it ever did, an extra co-occurrence edge is
harmless (no properties to conflict), the same never-deduplicated posture `RELATES_TO` adopts below.

**`create_entity_relationship` — the `RELATES_TO` fact edge**
```cypher
MATCH (subj:Entity {entityId: $subjectId})
MATCH (obj:Entity  {entityId: $objectId})
CREATE (subj)-[:RELATES_TO {
  label: $label, sourceChunkId: $sourceChunkId,
  sourceDocumentId: $sourceDocumentId, createdAt: $createdAt
}]->(obj)
```
`label` is the LLM-extracted predicate, free text — **not** its own Cypher relationship type (plan
§3.3/§3.1: an unbounded, LLM-controlled set of relationship types is a real risk this avoids, the
same "opaque string, parsed app-side" convention `Step.config`/`TRANSITION.guard` already use).
**Never deduplicated**: every extracted fact is independent provenance and is written even if an
identical edge already exists between the same two entities (FR-6 — conflicting or repeated facts
are always kept, never merged or overwritten). `sourceChunkId`/`sourceDocumentId` are what makes
AC-10's traceability provable: every fact traces back to the exact chunk (and, transitively via
`HAS_CHUNK`, the document) that produced it.

No DDL for `ABOUT` or `RELATES_TO` — both are always traversed *from* an already-anchored
`Chunk`/`Entity`, never independently scanned by their own properties (the same `EMITTED`/
`MENTIONS_MEMBER` precedent, §4/§10).

**`Entity.name` full-text index + `Entity.nameNormalized` — new DDL, inert until Stage 4**
```cypher
CALL db.idx.fulltext.createNodeIndex('Entity', 'name')
CREATE INDEX FOR (n:Entity) ON (n.nameNormalized)
```
`nameNormalized` is written starting this stage (`create_entity`, above) even though a future fusion
stage is the first reader — this avoids a backfill migration once that stage lands. No uniqueness
constraint: distinct real entities can and do share a normalized name+type before fusion runs. Two
separate mechanisms, deliberately: the fulltext index exists for a future fuzzy-suggestion tier;
`nameNormalized`'s plain RANGE index exists for a future deterministic exact-match tier, kept off
RediSearch's tokenizer/stemmer behavior entirely. Neither is queried yet — bootstrapped-but-dormant,
the same posture the original `Chunk` scaffolding demonstrated (§14 intro).

`Repository.create_entity`/`link_chunk_about_entity`/`create_entity_relationship`
(`server/falkorchat/repository.py`); `falkorchat.extraction.extract`, `falkorchat.ingestion.
IngestionPipeline.extract_chunk` (new modules); `background._safe_extract` (mirrors
`_safe_embed_chunk`'s failure isolation — an extraction failure for one chunk never corrupts the
`Document` or blocks sibling chunks). Wired the same way as Stage 2's chunk-embed scheduling:
`ingest_document` on both MCP and REST schedules one `_safe_extract` per chunk, alongside (never
instead of, never chained to) that chunk's own `_safe_embed_chunk` schedule. `config/models.json`
gains a fifth `ModelGateway` kind, `extraction` (`document-ingestion-graph.md` §4 — "zero graph
cost," resolved through the existing per-kind config, no new resolution mechanism). No `test_queries.sh`
additions for this stage, same precedent Stage 2 set: the new Cypher shapes are direct analogues of
already-verified patterns in this library, covered by the pytest integration suite instead.

**Resource note (Pass 3 code-gate MAJOR 2, `docs/reviews/document-ingestion-impl.md`).** On the MCP
transport, "scheduled independently, alongside" means a **second** raw `threading.Thread` per chunk
(`mcp.py`'s `_default_schedule` — see its docstring): a max-size document (`MAX_DOCUMENT_CHARS =
500_000`, ~1,000-char chunks) now spawns on the order of **~1,000-1,200 threads** per `ingest_document`
call, up from Stage 2's ~500-600 — doubled, not incidental. `_default_schedule` now catches a
thread-creation failure itself (`RuntimeError: can't start new thread`, a real ceiling once thread
count nears a `ulimit -u`/cgroup limit) and logs+continues rather than letting it escape the tool
handler mid-loop — fixed this stage, not deferred. The fuller fix (a bounded thread pool instead of
one-thread-per-(chunk × job)) is still deferred to Stage 6, same disposition as Stage 2's original
finding. REST is unaffected: `BackgroundTasks.add_task` is a cheap list append, not a thread spawn,
so no equivalent amplification or failure mode exists on that transport.

Stages 5–6 (chat-grounding, batch hardening) add the rest of `document-ingestion.md`'s MCP/REST
table; not built here.

### 14.6 Entity fusion — `SAME_AS` (K-050 M5 Stage 4, FR-6/7/8/9/10)

Design: `docs/plans/document-ingestion.md` §3.4 ("Concurrency note" — why the exact tier is one
atomic query, not a processing-order guarantee) and `docs/plans/document-ingestion-graph.md`
§1.5-§1.8 (graph-dba, the exact Cypher below, already live-verified there — this section is a
copy-and-cite against `server/falkorchat/repository.py`, not new verification work). Every fusion
decision — auto or suggested — is a property-bearing `SAME_AS` edge, never a physical node merge
(FalkorDB has no APOC-style refactor procedure, and this codebase avoids destructive graph surgery
elsewhere too):

```
(:Entity)-[:SAME_AS {
  matchId, status, confidence, technique,
  createdAt, decidedAt, decidedBy,
  resuggestCount, lastResuggestedAt
}]->(:Entity)
```

`status ∈ {pending, confirmed, rejected}`. Write direction is a convention, not a semantic claim —
always `(newlyExtractedEntity)-[:SAME_AS]->(existingCandidateEntity)`; reads that need to be
direction-agnostic use an undirected pattern, reads that always know the direction (because they
wrote it) use a directed one. **Every query below matches its `SAME_AS` endpoints UNLABELED
(`(a)`/`(b)`, never `(a:Entity)`)** — a bare label on either endpoint of a relationship-index-
anchored query forces a full `Node By Label Scan` on this build even though the relationship-
property scan alone is fully selective (`document-ingestion-graph.md` §1.4,
`claude/graph-dba/falkordb-quirks.md`) — the one exception is `create_or_reopen_match`'s two
`MATCH` clauses that resolve `a`/`b` by `entityId` in the first place (real per-node predicates,
not a bystander label next to a relationship filter, so the trap doesn't apply there).

**`create_entity_with_auto_match` — FR-8 exact tier, folded into entity creation itself**

The plan-gate review's BLOCKER fix (`document-ingestion.md` §3.4): the original design ran the
exact-tier candidate lookup, the entity `CREATE`, and the conditional auto-link as three
independent round trips — two entities extracted around the same wall-clock time could each miss
the other's not-yet-committed sibling, silently defeating FR-8's "no confirmation required"
guarantee. Closed by folding all three into one atomic `GRAPH.QUERY`; FalkorDB/Redis serializes
command execution, so two concurrent calls against the same `(nameNormalized, type)` can never
interleave.

```cypher
// $matchId is server-minted; consumed only if a candidate is actually found.
// Oldest-first tie-break when multiple pre-existing entities share
// (nameNormalized, type) without ever having been fused.
OPTIONAL MATCH (candidate:Entity {nameNormalized: $nameNormalized, type: $type})
WITH candidate
ORDER BY candidate.createdAt ASC
LIMIT 1
CREATE (e:Entity {
  entityId: $entityId, name: $name, nameNormalized: $nameNormalized,
  type: $type, createdAt: $createdAt
})
WITH e, candidate
FOREACH (_ IN CASE WHEN candidate IS NOT NULL THEN [1] ELSE [] END |
  CREATE (e)-[:SAME_AS {
    matchId: $matchId, status: 'confirmed', confidence: 1.0,
    technique: 'exact_normalized_name_type', createdAt: $createdAt,
    decidedAt: $createdAt, decidedBy: 'system',
    resuggestCount: 0, lastResuggestedAt: null
  }]->(candidate)
)
RETURN e.entityId AS entityId,
       candidate IS NOT NULL AS exactMatched,
       candidate.entityId AS candidateEntityId,
       CASE WHEN candidate IS NOT NULL THEN $matchId ELSE null END AS matchId
```

**[verified]** (`document-ingestion-graph.md` §1.8, `gdba_probe_atomic`, re-verified live against
`repository.py`'s shipped shape during the Stage 4 code gate, `docs/reviews/document-ingestion-
impl.md` Pass 4): a brand-new `(nameNormalized, type)` pair reports `exactMatched=false` on the
first call (never self-matches its own `CREATE`); a second call with the same pair correctly
reports `exactMatched=true` against the first call's entity; a third call with two eligible
pre-existing candidates picks the **older** one. A concurrency regression test
(`test_create_entity_with_auto_match_concurrent_calls_produce_exactly_one_edge`, real threads, a
`threading.Barrier`, separate connections) proves the fix closes the race: exactly one `SAME_AS`
edge, never zero, never duplicated. No reopen branch — a brand-new `Entity`, `CREATE`d fresh inside
this same query, cannot structurally already carry a `SAME_AS` edge to reopen.

**`find_fuzzy_candidates` — FR-9 suggested tier, a genuinely separate read**

```cypher
// $fuzzyQuery is built app-side (falkorchat.fusion._fuzzy_query), one RediSearch
// 1-edit fuzzy term per name token: '%acme%'.
CALL db.idx.fulltext.queryNodes('Entity', $fuzzyQuery) YIELD node AS candidate, score
WHERE candidate.type = $type
RETURN candidate.entityId AS entityId, candidate.name AS name,
       candidate.type AS type, score
ORDER BY score DESC
LIMIT $limit
```

Unaffected by the exact tier's concurrency fix — a missed/duplicated fuzzy suggestion under
concurrent timing still lands in the reviewed `pending` queue either way, never silently defeating
a zero-review guarantee the way the exact tier's race did. The `type` filter is a routine
implementation choice (avoids a fuzzy name hit surfacing a nonsensical cross-type suggestion), not
plan-mandated. `falkorchat.fusion.find_fuzzy_candidates` builds `$fuzzyQuery` and calls this method;
`IngestionPipeline.fuse_entity` excludes the just-created entity's own id from the results before
classifying — **live-confirmed necessary**: a same-connection write is synchronously visible to the
next RediSearch fulltext query on this build, so an entity's own fuzzy lookup against its own name
can and does return itself as a hit.

**`create_or_reopen_match` — the guarded find-or-create-or-reopen write (OQ-3)**

Called for the suggested tier only (`status='pending'`) — the exact tier no longer calls this,
folded into `create_entity_with_auto_match` above.

```cypher
// $newEntityId, $candidateEntityId = the two Entity.entityId values (write
// direction new -> existing; reads elsewhere don't rely on this direction).
// $matchId = server-minted uuid, consumed only if a fresh edge is created.
MATCH (a:Entity {entityId: $newEntityId})
MATCH (b:Entity {entityId: $candidateEntityId})
OPTIONAL MATCH (a)-[existing:SAME_AS]-(b)
WITH a, b, existing,
     (existing IS NULL) AS isNew,
     (existing IS NOT NULL AND existing.status = 'rejected') AS reopen
FOREACH (_ IN CASE WHEN isNew THEN [1] ELSE [] END |
  CREATE (a)-[:SAME_AS {
    matchId: $matchId, status: $status, confidence: $confidence,
    technique: $technique, createdAt: $createdAt,
    decidedAt: CASE WHEN $status = 'confirmed' THEN $createdAt ELSE null END,
    decidedBy: CASE WHEN $status = 'confirmed' THEN 'system' ELSE null END,
    resuggestCount: 0, lastResuggestedAt: null
  }]->(b)
)
FOREACH (_ IN CASE WHEN reopen THEN [1] ELSE [] END |
  SET existing.status = 'pending',
      existing.resuggestCount = coalesce(existing.resuggestCount, 0) + 1,
      existing.lastResuggestedAt = $createdAt
)
RETURN isNew AS created, reopen AS reopened,
       coalesce(existing.matchId, $matchId) AS matchId,
       coalesce(existing.status, $status)   AS status
```

**Why not a bare `MERGE (a)-[:SAME_AS]->(b)`**: `SAME_AS` is semantically symmetric, but a
direction-fixed `MERGE` only matches an edge written in *that* direction — a suggestion already
written as `(b)-[:SAME_AS]->(a)` on an earlier, opposite-order ingestion would be invisible to it
and get duplicated. The `OPTIONAL MATCH ... undirected` lookup sidesteps that by construction.

**[verified]** (`document-ingestion-graph.md` §1.6, `gdba_probe_ingestion2`): call 1 on a fresh
pair → `created=true`; call 2 (same pair) → `created=false, reopened=false` (idempotent no-op);
manually reject, then re-derive the same pair → `created=false, reopened=true`, `resuggestCount`
bumps on the **original** `matchId` (no duplicate edge); querying from the reversed argument order
finds the same edge.

**`confirm_match` / `reject_match` / `recheck_match` — FR-10, OQ-3's manual reopen**

```cypher
// confirm_match(match_id, decided_by, decided_at)
MATCH (a)-[r:SAME_AS {matchId: $matchId}]->(b)
SET r.status = 'confirmed', r.decidedAt = $decidedAt, r.decidedBy = $decidedBy
RETURN r.matchId AS matchId, r.status AS status,
       a.entityId AS entityA, b.entityId AS entityB
```

```cypher
// reject_match(match_id, decided_by, decided_at) — never deletes the edge, the
// rejected record is what makes OQ-3 answerable with no second mechanism.
MATCH (a)-[r:SAME_AS {matchId: $matchId}]->(b)
SET r.status = 'rejected', r.decidedAt = $decidedAt, r.decidedBy = $decidedBy
RETURN r.matchId AS matchId, r.status AS status,
       a.entityId AS entityA, b.entityId AS entityB
```

```cypher
// recheck_match(match_id, at) — rejected -> pending only; a no-op otherwise
// (including "no such matchId" — the WHERE guard can't tell the two apart,
// and both are equally "nothing to transition" to the caller).
MATCH (a)-[r:SAME_AS {matchId: $matchId}]->(b)
WHERE r.status = 'rejected'
SET r.status = 'pending',
    r.resuggestCount = coalesce(r.resuggestCount, 0) + 1,
    r.lastResuggestedAt = $at
RETURN r.matchId AS matchId, r.status AS status,
       a.entityId AS entityA, b.entityId AS entityB
```

`decidedBy` is a real `User`/`Agent` id on this path, never `'system'` — an audit trail can always
tell an automatic decision (`create_entity_with_auto_match`) from a human/agent one.

**`list_pending_matches` / `list_matches` — OQ-2's review surface**

```cypher
// list_pending_matches(limit) — directed, matches the canonical write direction.
MATCH (a)-[r:SAME_AS {status: 'pending'}]->(b)
RETURN r.matchId AS matchId,
       a.entityId AS entityA, a.name AS nameA,
       b.entityId AS entityB, b.name AS nameB,
       r.confidence AS confidence, r.technique AS technique, r.createdAt AS createdAt
ORDER BY r.createdAt
LIMIT $limit
```

`list_matches(status=None, limit)` is the plan-gate review's MAJOR-finding fix — the auto-merged
tier (`status='confirmed', decidedBy='system'`) had no discovery surface at all before this.
**Two separate query strings, not a `WHERE $status IS NULL OR r.status = $status` null-guard** —
live-verified that idiom silently discards the `SAME_AS.status` index even when `$status` is bound
to a real value (`document-ingestion-graph.md` §1.7, `claude/graph-dba/falkordb-quirks.md`):

```cypher
// list_matches(status=<value>, limit) — filtered branch, same shape as
// list_pending_matches with status parameterized instead of a literal.
MATCH (a)-[r:SAME_AS {status: $status}]->(b)
RETURN r.matchId AS matchId,
       a.entityId AS entityA, a.name AS nameA,
       b.entityId AS entityB, b.name AS nameB,
       r.status AS status, r.confidence AS confidence, r.technique AS technique,
       r.createdAt AS createdAt
ORDER BY r.createdAt
LIMIT $limit
```

```cypher
// list_matches(status=None, limit) — unfiltered branch. A full scan is
// genuinely unavoidable here (no relationship-type-only scan operator exists
// on this build) — reasonable for an infrequent admin/audit call, not a hot
// path; $limit bounds the result size, not the scan cost.
MATCH (a)-[r:SAME_AS]->(b)
RETURN r.matchId AS matchId,
       a.entityId AS entityA, a.name AS nameA,
       b.entityId AS entityB, b.name AS nameB,
       r.status AS status, r.confidence AS confidence, r.technique AS technique,
       r.createdAt AS createdAt
ORDER BY r.createdAt
LIMIT $limit
```

**DDL — `SAME_AS` relationship-scoped indexes + constraint (index-before-constraint)**
```cypher
CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.matchId)
CREATE INDEX FOR ()-[r:SAME_AS]-() ON (r.status)
GRAPH.CONSTRAINT CREATE <graph> UNIQUE RELATIONSHIP SAME_AS PROPERTIES 1 matchId
```
Relationship-property indexing is a proven, first-class capability on this build — live-verified
(`document-ingestion-graph.md` §1.1): `db.indexes()` reports `RELATIONSHIP`-scoped indexes exactly
like node ones, going `PENDING → OPERATIONAL` the same way, and a property-filtered `SAME_AS` query
profiles as `Edge By Index Scan`. No new `Entity`-side DDL for this stage — `Entity.name`
(fulltext) and `Entity.nameNormalized` (RANGE) already landed in Stage 3 (§14.5), dormant until
this stage's reads.

`Repository.create_entity_with_auto_match`/`find_fuzzy_candidates`/`create_or_reopen_match`/
`confirm_match`/`reject_match`/`recheck_match`/`list_pending_matches`/`list_matches`
(`server/falkorchat/repository.py`); `falkorchat.fusion.find_fuzzy_candidates`/`classify_fuzzy`
(new module); `falkorchat.ingestion.IngestionPipeline.fuse_entity`; `background._safe_fuse`
(per-ENTITY failure isolation, one level finer than `_safe_extract`'s per-chunk isolation — called
inline from `extract_chunk`'s own loop, never scheduled separately by `api.py`/`mcp.py`, since a
fuzzy lookup can only run once the entity it's for already exists). MCP tools `list_pending_matches`/
`list_matches`/`confirm_match`/`reject_match`/`recheck_match`, mirrored as REST routes
`GET /matches/pending`, `GET /matches?status=&limit=`, `POST /matches/{id}/confirm`,
`POST /matches/{id}/reject`, `POST /matches/{id}/recheck` (`docs/plans/document-ingestion.md` §3.5).

Stage 6 (batch hardening) is the rest of `document-ingestion.md`'s MCP/REST table; not built here.

---

## 15. Product catalog — structured lookup (K-052 M6)

Design: `docs/plans/workflow-catalog-lookup.md` §3.1/§3.5 (architect). A fixed-shape, seed-script-
write-only electronics catalog in `reference` (FR-6: global, not scoped per workspace — `Product`
is looked up by property key, never traversed from a workspace node, so it never needs
per-workspace materialization, unlike `WorkflowDef`). Backs two fixed query shapes for the
`salesperson` demo agent's tools: an exact-name fact lookup (FR-1) and a category/price-range
filter (FR-2/FR-3) — not a free-text search (no full-text index is added; AC-4's wording tolerance
is handled by the calling LLM's own argument extraction, not by fuzzy matching in the database).

**Schema:** `(:Product {productId, name, nameNormalized, category, categoryNormalized, price})`.
`nameNormalized`/`categoryNormalized` are both computed with `extraction.normalize_name`
(`server/falkorchat/extraction.py:67-78`, "the ONE shared normalization helper" — reused, not
reimplemented, the same tier-mechanics precedent already established for `Entity.nameNormalized`,
§14.5) — a real `=` comparison, decoupled from RediSearch tokenizer/stemmer behavior and from any
runtime `toLower()` call.

**DDL — index-then-constraint (`scripts/bootstrap_schema.sh`'s `bootstrap_reference`)**
```cypher
CREATE INDEX FOR (n:Product) ON (n.productId)
CREATE INDEX FOR (n:Product) ON (n.nameNormalized)
CREATE INDEX FOR (n:Product) ON (n.categoryNormalized)
CREATE INDEX FOR (n:Product) ON (n.price)
GRAPH.CONSTRAINT CREATE reference UNIQUE NODE Product PROPERTIES 1 productId
```
`productId` is the standard `{label}Id` identity anchor (range index + UNIQUE constraint, index
first — §7.1 rule 1). `nameNormalized`, `categoryNormalized`, `price` are plain range indexes with
no constraint: two products could in principle share a normalized name in a different category (no
constraint assumes otherwise), and categoryNormalized/price are hot-filter-only. The raw `category`
property carries no index of its own — same convention as `name`/`nameNormalized`: only the
normalized property is indexed/matched on; the raw property is returned in results, never filtered
on directly.

### 15.1 `lookup_product` — exact-name fact lookup (FR-1/AC-1/AC-4)

```cypher
// $nameNormalized = case-folded + whitespace-collapsed product name, computed by the
// caller (services.lookup_product) via extraction.normalize_name — same convention as
// $nameNormalized elsewhere in this file (§14.5's create_entity/find_fuzzy_candidates).
MATCH (p:Product {nameNormalized: $nameNormalized})
RETURN p.productId AS productId, p.name AS name,
       p.category AS category, p.price AS price
LIMIT 1
```
Anchors on the `Product.nameNormalized` range index — a single `Node By Index Scan`, no label
scan. Zero rows back means "no such product" (AC-3 abstention); the service maps that to
`{"found": false}`, never a fabricated row. `LIMIT 1` is belt-and-suspenders: `nameNormalized`
carries no uniqueness constraint (see DDL note above), so a future data quirk that produced two
same-named products in different categories still returns exactly one row rather than erroring.

**`productId` in the projection (K-053 M6).** K-052's original shape projected
`{name, category, price}` only — a pure fact lookup, with no cart concept yet to key on an id.
K-053 added `p.productId` so `services.add_cart_item` could resolve a customer-named product to
the `productId` a `CartItem` is keyed on (`docs/plans/workflow-cart-and-totals-graph.md` §2);
additive and backward-compatible, since the `Product` node always carried the property and no
caller reads the row positionally. `LookupProductFactTool` returns the row verbatim, so the slug
reaches the model's context — the precedent §15.2's own widening was judged against.
*Transcribed here on 2026-09-03, and the gap it closes is checkable rather than approximate:*
this block was written at **`14891c9`** (K-052, 2026-08-28 09:53) and the code widened at
**`bcd2dcc`** (K-053, 2026-08-28 20:21) — ten hours later the same day — so §15.1 documented a
projection `Repository.lookup_product` had already stopped using, until `8aaeca3` six days on.
`scripts/test_queries.sh` reported **408/408 throughout**, and could not have done otherwise:
its `LOOKUP` constant and its abstention header were both written at `14891c9` and stayed
consistent **with each other**, which is the one state the gate cannot see — narrow either alone
and it drops to 407/408. That is the lesson, not the six days: a transcription gate goes green on
a wrong transcription, so the doc block and the shell constant have to be checked against the
code, never against each other.

### 15.2 `filter_products` — category/price-range filter (FR-2/FR-3)

```cypher
// $categoryNormalized, $minPrice, $maxPrice are each optional (NULL when the caller
// omits that filter) — the WHERE clause is parameterized, never string-interpolated
// (rule 1). $limit bounds the result size (services.py's DEFAULT_FILTER_LIMIT).
// $categoryNormalized = case-folded + whitespace-collapsed category, computed by
// Repository.filter_products via extraction.normalize_name from the caller's raw
// `category` — same convention as $nameNormalized (§15.1) and Entity.nameNormalized
// (§14.5): a precomputed normalized property + exact match, not a runtime toLower().
MATCH (p:Product)
WHERE p.categoryNormalized = coalesce($categoryNormalized, p.categoryNormalized)
  AND p.price >= coalesce($minPrice, -1.0)
  AND p.price <= coalesce($maxPrice, 1000000000.0)
RETURN p.productId AS productId, p.name AS name,
       p.category AS category, p.price AS price
ORDER BY p.price ASC
LIMIT $limit
```

**Deviation from the plan's illustrative Cypher (§3.5), and why.** The plan's illustrative form
wraps every predicate in a `$param IS NULL OR prop = $param` null-guard. Live-profiled on this
build (`GRAPH.PROFILE`, throwaway probe graph, 20-row fixture) before finalizing: that shape
degrades to a full `Node By Label Scan` in *every* call, including calls where a filter carries a
real, selective value — the exact same live-verified quirk already documented for this codebase's
`list_matches` (§14.6-adjacent note above, `claude/graph-dba/falkordb-quirks.md` "Query tuning" —
"`$param IS NULL OR prop = $param` … defeats an otherwise-available index — even when `$param` is
bound to a real, selective value"). That prior case branched into two separate query strings
(filtered / unfiltered) as the fix — the established convention for a single optional predicate.
`filter_products` has **three independent optional predicates**, so the same branching approach
would require up to 8 query strings; instead:
- `category`'s optional-equality is expressed as
  `p.categoryNormalized = coalesce($categoryNormalized, p.categoryNormalized)` — a self-referential
  coalesce that is tautologically true when `$categoryNormalized` is `NULL` (every seeded `Product`
  always has a non-null `categoryNormalized`). This does **not** itself anchor an index scan
  (verified — same degradation as the null-guard form when it's the only predicate), but it doesn't
  need to, because:
- `minPrice`/`maxPrice` use **literal numeric sentinels** via `coalesce($minPrice, -1.0)` /
  `coalesce($maxPrice, 1000000000.0)` — effectively ±infinity for a consumer-electronics price
  domain (no seeded product is ever priced at or below `$0`, or at/above `$1,000,000,000`). Unlike
  the `IS NULL OR` form, a plain range comparison against a **literal** bound (even a
  parameter-supplied one) is something the planner can still use to anchor an index scan — verified:
  every combination of supplied/omitted `category`/`minPrice`/`maxPrice` (including the
  fully-omitted "list everything" call) profiles as `Node By Index Scan | (p:Product)` on
  `Product.price`, with `categoryNormalized`'s coalesce riding along as a cheap residual `Filter`
  when supplied. This is strictly better than the plan's illustrative shape even for the "list
  everything" case, which the plan explicitly accepted as an unavoidable full scan.
- Result correctness confirmed identical to the null-guard form across every filter combination
  (unfiltered; category-only; price-range-only; category+price; nonexistent category → zero rows)
  before adopting this shape — same rows, same order, just index-anchored instead of a label scan.

An all-omitted call (`$categoryNormalized`/`$minPrice`/`$maxPrice` all `NULL`) lists the whole
catalog, bounded by `$limit` — acceptable at this catalog's demo scale (~15 rows) and, per the
profiling above, index-anchored anyway. Zero rows back means "nothing matches" (AC-3 abstention);
the service maps that to `{"items": [], "finding": "no matching products found"}`. Ordered by price
ascending — a stable, predictable order for a listing shape, cheap once index-backed.

**Case/whitespace-insensitive `category` (live-discovered fix, 2026-08-28).** The original shape
compared the raw `$category` argument against `p.category` with an exact, case-sensitive `=`. A
live LLM-driven run lowercased a category when calling the `filter_products` tool (e.g. `"audio"`
for the seeded `"Audio"`), and the exact-case comparison silently returned zero rows — re-asking
with correct casing worked, so the def/tool/service wiring was sound; only this comparison was
wrong. Fix: `Repository.filter_products` normalizes the caller's raw `category` with
`extraction.normalize_name` (same helper as `nameNormalized`, §15.1) into `$categoryNormalized`,
compared against a new precomputed `Product.categoryNormalized` property — **not** a runtime
`toLower(p.category) = toLower($category)` in the `WHERE` clause. Two reasons: (1) this codebase's
existing convention for case/whitespace-insensitive lookup is a precomputed normalized property +
exact match, already established for `Entity.nameNormalized` (§14.5) and `Product.nameNormalized`
(§15.1) — a second, ad-hoc `toLower()` comparison would be a second, differently-implemented
normalization tier; (2) it costs nothing extra here: `category`/`categoryNormalized` was never this
query's scan anchor (`Product.price` is, via the literal-sentinel bounds above) — a runtime
`toLower()` on a non-anchor residual `Filter` predicate would not have reintroduced the label-scan
regression documented above either, but the normalized-property form is free of that risk by
construction and stays consistent with the rest of this file. Live-`GRAPH.PROFILE`-verified after
the change: every filter combination, including a differently-cased `category` argument, still
anchors on `Node By Index Scan | (p:Product)` via `Product.price` — the regression this section
already documents once was not reintroduced.

**`productId` in the projection (`salesperson-ui` S7c, 2026-09-03).** The `RETURN` above
originally projected `{name, category, price}` only. **This is §15.1's K-053 widening applied on
the list side** — the same additive change, for the same reason, with the same consequence for the
model's context (§15.1's note above): the `Product` node always carried the property, it just
wasn't selected, and the scan anchor (`Product.price`) is untouched, so none of the
`GRAPH.PROFILE` findings above change. What it buys: the storefront catalog route
(`docs/plans/salesperson-ui.md` §5.2 `GET /shop/api/catalog`, §4.7's image manifest) is keyed on
`productId`, and without it `Storefront._catalog_rows` had to resolve each listed row's id with a
second `lookup_product` point read — a `1 + n` plus a silent-drop branch for a row that failed to
re-resolve, both deleted by this projection. `FilterProductsTool` returns these rows verbatim to
the model, exactly as `LookupProductFactTool` has since K-053, so the two sibling catalog tools
now put the same kind of thing in front of the model rather than one of them a new kind.

**Fully implemented (K-052 M6, plan §4 steps 3-9):** `Repository.lookup_product`/
`filter_products`, `Services.lookup_product`/`filter_products`,
`LookupProductFactTool`/`FilterProductsTool`, seed/verify data
(`scripts/seed_catalog.sh`/`verify_catalog.sh`), and the `salesperson@v1` demo
`WorkflowDef` (`falkorchat.proof_defs.SALESPERSON_DEF`,
`scripts/seed_salesperson.sh`/`verify_salesperson.sh`).

---

## 16. Cart / Order — durable cart, deterministic totals, order lifecycle (K-053 M6)

Design: `docs/plans/workflow-cart-and-totals.md` §3.2/§3.3/§3.4 (architect) and
`docs/plans/workflow-cart-and-totals-graph.md` (graph-dba, Version 2 — every query below is
`[verified]` there against the live pinned instance; this section transcribes, it does not
re-derive). Workspace-scoped (`ws:{workspaceId}`), unlike §15's global catalog — a cart/order
belongs to one customer inside one workspace, never shared across workspaces.

**Schema:**
```
(:Customer {customerId, createdAt})
(:Customer)-[:HAS_CART]->(:Cart {customerId, createdAt, updatedAt})
(:Cart)-[:HAS_ITEM]->(:CartItem {customerId, productId, quantity, addedAt, updatedAt})
(:Customer)-[:PLACED]->(:Order {orderId, status, placedAt, updatedAt})
(:Order)-[:HAS_LINE]->(:OrderLine {productId, name, unitPrice, quantity, lineTotal})
```
`customerId = User.userId` (graph note §1.2) — the workspace-local `Customer` anchor is a
distinct node from `User` (keeps `ensure_user`'s create-once contract untouched; `Customer` is
freely mutable) but reuses the same member-id namespace. `Cart`/`CartItem` are keyed on natural
keys (`customerId`, and the composite `(customerId, productId)`) — no synthetic `cartId`/
`cartItemId`, mirroring the `WorkflowDefSnapshot` natural-key precedent (DESIGN §7.1). **`Cart`/
`CartItem` carry no price** — FR-3 requires the cart's displayed price to always be the catalog's
*current* price (`reference`), never a stored, staleable snapshot; §16.9 below is the two-graph
read this forces. **`Order`/`OrderLine` carry a full frozen snapshot** (`name`, `unitPrice`,
`quantity`, `lineTotal`) — AC-6's requirement, since `place_order` computes and persists it once,
atomically, and nothing ever re-derives it from a live catalog read again. `Order.total` is
deliberately **not stored** — computed as `sum(OrderLine.lineTotal)` on read (§16.8): per-order
line count is tiny, so this avoids a write-time-only derived field that could drift from its own
source lines. `status ∈ {'placed', 'fulfilled', 'delivered', 'cancelled'}` is a **property**, not
a label, mirroring `WorkflowRun.status`/`StepRun.status` (DESIGN §1.2). `OrderLine` carries no
independent identity/index/constraint — a pure value-object always reached via `HAS_LINE`.

**DDL — index-then-constraint (`scripts/bootstrap_schema.sh`'s `bootstrap_workspace`)**
```cypher
CREATE INDEX FOR (n:Customer) ON (n.customerId)
CREATE INDEX FOR (n:Cart) ON (n.customerId)
CREATE INDEX FOR (n:CartItem) ON (n.customerId)
CREATE INDEX FOR (n:CartItem) ON (n.productId)
CREATE INDEX FOR (n:Order) ON (n.orderId)
GRAPH.CONSTRAINT CREATE ws:<id> UNIQUE NODE Customer PROPERTIES 1 customerId
GRAPH.CONSTRAINT CREATE ws:<id> UNIQUE NODE Cart PROPERTIES 1 customerId
GRAPH.CONSTRAINT CREATE ws:<id> UNIQUE NODE CartItem PROPERTIES 2 customerId productId
GRAPH.CONSTRAINT CREATE ws:<id> UNIQUE NODE Order PROPERTIES 1 orderId
```
No index/constraint on `OrderLine` or `Order.status` — `OrderLine` has no independent lookup path
(always reached via `HAS_LINE`), and no AC needs to scan/list orders *by* status across a
workspace at this cardinality (orders per customer — never a supernode). A `GRAPH.PROFILE` on the
lifecycle CAS queries (§16.9) shows a residual `Filter` above the `Node By Index Scan | (o:Order)`
for this reason — functionally identical to a folded index scan (zero rows on a mismatch either
way), just not folded into the scan itself (graph note §3.4).

### 16.1 `ensure_customer` — get-or-create the `Customer` anchor (idempotent)

```cypher
// $customerId, $now
MERGE (c:Customer {customerId: $customerId})
ON CREATE SET c.createdAt = $now
RETURN c.customerId AS customerId, c.createdAt AS createdAt
```
`ON CREATE` only — a re-call leaves `createdAt` at the first call's value. Backed by the
`Customer.customerId` uniqueness constraint (every `MERGE` must be constraint-backed, rule 8).

### 16.2 `ensure_cart` — get-or-create the `Cart` anchor (idempotent)

```cypher
// $customerId, $now — call after ensure_customer
MATCH (cust:Customer {customerId: $customerId})
MERGE (cust)-[:HAS_CART]->(cart:Cart {customerId: $customerId})
ON CREATE SET cart.createdAt = $now, cart.updatedAt = $now
RETURN cart.customerId AS customerId, cart.createdAt AS createdAt
```
The `MERGE` spans the whole `(Customer)-[:HAS_CART]->(Cart)` pattern — safe here (not the K-034
changed-endpoint gotcha) because `Cart`'s matched property (`customerId`) is fully specified and
constraint-backed on both ends. Zero rows (no such `Customer`) means the caller didn't call
`ensure_customer` first.

### 16.3 `add_to_cart` — MERGE-and-increment a cart line (idempotent-by-design)

```cypher
// $customerId, $productId, $qty, $now
MATCH (cart:Cart {customerId: $customerId})
MERGE (cart)-[:HAS_ITEM]->(item:CartItem {customerId: $customerId, productId: $productId})
ON CREATE SET item.quantity = $qty, item.addedAt = $now, item.updatedAt = $now
ON MATCH  SET item.quantity = item.quantity + $qty, item.updatedAt = $now
SET cart.updatedAt = $now
RETURN item.productId AS productId, item.quantity AS quantity
```
**"Add" means increment, not replace** — two calls with the same `productId` sum their `qty`,
matching the FR-1 shopping-cart mental model. Zero rows means no `Cart` exists yet for
`customerId` — the caller runs `ensure_customer` + `ensure_cart` first (`services.add_cart_item`'s
own ordering; the closed MAJOR finding from `docs/reviews/workflow-cart-and-totals.md`).

### 16.4 `adjust_cart_item` — guarded decrement-or-remove, single query

```cypher
// $customerId, $productId, $qty (positive = amount to remove), $now
MATCH (cart:Cart {customerId: $customerId})-[:HAS_ITEM]->(item:CartItem {customerId: $customerId, productId: $productId})
WITH cart, item, (item.quantity - $qty) AS newQty
FOREACH (_ IN CASE WHEN newQty > 0  THEN [1] ELSE [] END | SET item.quantity = newQty, item.updatedAt = $now)
FOREACH (_ IN CASE WHEN newQty <= 0 THEN [1] ELSE [] END | DETACH DELETE item)
SET cart.updatedAt = $now
RETURN newQty AS quantity, (newQty <= 0) AS removed
```
The same `FOREACH (_ IN CASE WHEN … THEN [1] ELSE [] END | …)` guarded-write idiom §4/§14.6 use
for a single-query check-then-act, race-free by construction (FalkorDB serializes writes per
graph). Decrementing to a positive remainder updates in place; decrementing to zero or below
deletes the `CartItem` node and its edge in the same call. Zero rows means no such line (already
removed, or never added) — a no-op the caller reports as such, not an error.

### 16.5 `read_cart` — current lines, oldest-added first

```cypher
// $customerId
MATCH (cart:Cart {customerId: $customerId})-[:HAS_ITEM]->(item:CartItem)
RETURN item.productId AS productId, item.quantity AS quantity, item.addedAt AS addedAt
ORDER BY item.addedAt
```
Prices are resolved separately (§16.9 — the catalog lives in `reference`, a different graph).
Zero rows is a legitimate "empty cart" (or no cart yet) — both look the same by design; nothing
here needs to distinguish "never touched" from "emptied."

### 16.6 `clear_cart` — delete every line under a customer's cart

```cypher
// $customerId
MATCH (cart:Cart {customerId: $customerId})-[:HAS_ITEM]->(item:CartItem)
DETACH DELETE item
```
Used standalone (FR-1) and inside `place_order` (§16.7) as part of checkout. A `MATCH` against
zero items is a plain no-op — no `UNWIND`-collapse risk (that quirk is about an empty **list
parameter**, a different mechanism).

### 16.7 `place_order` — snapshot + clear cart, one atomic query

```cypher
// $customerId, $orderId, $now, $lines: [{productId, name, unitPrice, quantity, lineTotal}, ...]
MATCH (cust:Customer {customerId: $customerId})
OPTIONAL MATCH (dup:Order {orderId: $orderId})
WITH cust, (dup IS NULL) AS created
FOREACH (_ IN CASE WHEN created THEN [1] ELSE [] END |
  CREATE (cust)-[:PLACED]->(:Order {orderId: $orderId, status: 'placed',
                                     placedAt: $now, updatedAt: $now})
)
WITH cust, created
UNWIND (CASE WHEN $lines = [] THEN [null] ELSE $lines END) AS line
OPTIONAL MATCH (o:Order {orderId: $orderId})
FOREACH (_ IN CASE WHEN created AND line IS NOT NULL THEN [1] ELSE [] END |
  CREATE (o)-[:HAS_LINE]->(:OrderLine {productId: line.productId, name: line.name,
                                        unitPrice: line.unitPrice, quantity: line.quantity,
                                        lineTotal: line.lineTotal})
)
WITH cust, created, count(CASE WHEN line IS NOT NULL THEN 1 END) AS lineCount
OPTIONAL MATCH (cart:Cart {customerId: cust.customerId})-[:HAS_ITEM]->(item:CartItem)
FOREACH (_ IN CASE WHEN created THEN [1] ELSE [] END | DETACH DELETE item)
RETURN created, lineCount
```
`$lines` is pre-resolved **app-side** before this query runs (`services._priced_cart_lines`): the
current name/price per line is read from `reference` and `lineTotal = unitPrice × quantity` is
computed via `pricing.compute_line_total` — no arithmetic happens in Cypher (FR-4/FR-8). Keyed on
a caller-minted `orderId` (the idempotency key): a **guarded `CREATE`, not a `MERGE`** — mirrors
`Message`'s guarded-`CREATE` idiom (§4/§9), since placing an order is a one-time durable-record
creation event, not a find-or-update concept like `Cart`/`Customer`. A retried call with the same
`orderId` is a true no-op (`created=false`), never a duplicate order — live-verified end to end:
first call → `created=true, lineCount=2`, the `Order` + 2 `OrderLine`s exist with the right
snapshotted values and the cart's `CartItem`s are gone; an immediate retry with the same `orderId`
→ `created=false`, still exactly one `Order` and 2 `OrderLine`s. The cart's items are cleared
**only** on the call that actually creates the order, so a replay never re-clears an
already-emptied cart. The `Cart` node itself is left in place, empty, ready for the next shopping
session (matches `Thread` staying in place after every message).

### 16.8 `get_order` — status + snapshot + computed total

```cypher
// $orderId
MATCH (o:Order {orderId: $orderId})
OPTIONAL MATCH (o)-[:HAS_LINE]->(l:OrderLine)
RETURN o.orderId AS orderId, o.status AS status, o.placedAt AS placedAt, o.updatedAt AS updatedAt,
       collect({productId: l.productId, name: l.name, unitPrice: l.unitPrice,
                 quantity: l.quantity, lineTotal: l.lineTotal}) AS lines,
       sum(l.lineTotal) AS total
```
A `collect()` and a `sum()` together over the same fan-out, safe here (unlike the general "constant
scalar beside an aggregate fan-out" caveat) because `orderId` is unique by constraint — exactly one
`o` value for the whole fan-out, never more than one row per distinct group. `total` is never
stored (§3 above). `None` when no such `Order`. A zero-line order (not reachable via the intended
tool path, but a valid direct repository call) yields one all-`null` placeholder entry from
`collect()`, not `[]` — filtered client-side (`Repository.get_order`) rather than reshaping this
`[verified]` query.

### 16.9 The two-graph computation (FR-4/FR-8) — why a live cart/order total can't be one query

`CartItem`/`Order` live in `ws:{workspaceId}`; the catalog (§15) lives in `reference`. There is no
single Cypher query that can join a cart line to its current catalog price — not a missing edge,
but a structural fact: **`GRAPH.QUERY` operates on one named graph per call**, full stop (DESIGN §2
constraint #3, "relationships cannot cross graphs"). The mechanism is necessarily: (1) **read**
`ws:{workspaceId}` — §16.5's `read_cart` → `[{productId, quantity}, ...]`; (2) **read** `reference`
— `Repository.lookup_products_by_id` (§15's `Product` schema, batched `UNWIND $productIds`);
(3) **compute** (app/tool code, not Cypher, not an LLM call) — `pricing.compute_line_total`, plain
Python. This is exactly the shape §16.7's `place_order` already uses: the app performs steps 1–3
*before* calling the write, then passes the finished `$lines` in. Whatever step type or tool needs
"compute the total" cannot be a single graph traversal here — it is at minimum two
`GRAPH.RO_QUERY` calls plus app-side arithmetic, a property of the topology, not an implementation
shortcut.

### 16.10 Order lifecycle — guarded CAS, one transition per call (FR-6/FR-7, AC-7/AC-8)

Same compare-and-set idiom as `resume_run`/`suspend_run` (§12.3/§12.4): the write commits only if
the order is currently in the expected prior state; a stale/duplicate/out-of-order transition
attempt matches zero rows and writes nothing.
```cypher
// fulfill: $orderId, $now
MATCH (o:Order {orderId: $orderId})
WHERE o.status = 'placed'
SET o.status = 'fulfilled', o.updatedAt = $now
RETURN o.orderId AS orderId, o.status AS status
```
```cypher
// deliver: $orderId, $now
MATCH (o:Order {orderId: $orderId})
WHERE o.status = 'fulfilled'
SET o.status = 'delivered', o.updatedAt = $now
RETURN o.orderId AS orderId, o.status AS status
```
```cypher
// cancel: $orderId, $now — only reachable from 'placed', enforcing AC-8
// ("cannot cancel once fulfilled")
MATCH (o:Order {orderId: $orderId})
WHERE o.status = 'placed'
SET o.status = 'cancelled', o.updatedAt = $now
RETURN o.orderId AS orderId, o.status AS status
```
Live-verified: `placed→fulfilled` succeeds; a `cancel` attempt on a `fulfilled` order returns zero
rows and the status stays unchanged (AC-8 holds); `fulfilled→delivered` succeeds. These three
queries drive the `order-fulfillment@v1` process def's lifecycle (`falkorchat.proof_defs.
ORDER_FULFILLMENT_DEF`) via `services.advance_order` — called from whichever caller accepts the
operator's fulfillment input **alongside** `submit_workflow_input`, the same "two-step, accepted"
pairing §12.6's `link_step_emission` uses (the def's own `human`/`decision` steps have no side
effect on `Order` — DESIGN §6.1 — so advancing the run and advancing the order are always two
separate graph writes from one request, never one query doing both).

**Fully implemented (K-053 M6, plan §4 steps 2-9):** `pricing.compute_line_total`,
`Repository.ensure_customer`/`ensure_cart`/`add_to_cart`/`adjust_cart_item`/`read_cart`/
`clear_cart`/`place_order`/`get_order`/`fulfill_order`/`deliver_order`/`cancel_order`,
`Services.add_cart_item`/`get_cart`/`remove_cart_item`/`clear_cart`/`place_order`/
`get_order_status`/`advance_order`, `ViewCartTool`/`AddToCartTool`/`RemoveFromCartTool`/
`ClearCartTool`/`PlaceOrderTool`, the `salesperson@v2` version bump and the
`order-fulfillment@v1` process def (`falkorchat.proof_defs.ORDER_FULFILLMENT_DEF`), and seed/
verify data (`scripts/seed_salesperson.sh`/`verify_salesperson.sh`, extended to publish/check
both defs).

## 17. Durable customer profile (K-054 M6)

Design: `docs/plans/workflow-durable-profile.md` §3 (architect) and `docs/plans/
workflow-durable-profile-graph.md` (graph-dba, Version 2 — both queries below are `[verified]`
there against the live pinned instance, twice — v1 and the v2 `coalesce()` fix; this section
transcribes, it does not re-derive). Workspace-scoped (`ws:{workspaceId}`), same anchor §16 uses.

**Schema — two more nullable properties on the existing `Customer` node, no new label:**
```
(:Customer {customerId, createdAt,
             name, deliveryAddress, profileUpdatedAt})   // name/deliveryAddress: NULLable
```
`name`/`deliveryAddress` are independently nullable (a customer may give one before the other).
No new index, no new constraint — `Customer.customerId`'s existing uniqueness constraint (§16's
DDL) already backs the `MERGE` below. **This is not a second identity mechanism** — it reuses,
unchanged, the same `Customer` anchor §16's cart/order queries write to; both capabilities'
writes land on one node, never two competing ones (graph note §0/§1).

### 17.1 `upsert_profile` — `MERGE` + per-field `coalesce()`-guarded `SET`, update-in-place

```cypher
// $customerId, $name, $deliveryAddress, $now
MERGE (c:Customer {customerId: $customerId})
ON CREATE SET c.createdAt = $now
SET c.name             = coalesce($name, c.name),
    c.deliveryAddress = coalesce($deliveryAddress, c.deliveryAddress),
    c.profileUpdatedAt = $now
RETURN c.customerId AS customerId, c.name AS name, c.deliveryAddress AS deliveryAddress
```
**Deliberately not** the `ensure_user`/`ensure_customer` "guarded create, never updates" shape
(§16.1) — FR-3 requires the opposite: a later-supplied name/address updates the stored profile,
never frozen after the first write. **Also deliberately not** an unconditional `SET` (the shape
`write_model_overrides`, §13.1, uses) — that was v1's BLOCKER, caught at plan-gate: `SaveProfileTool`'s
arguments are genuinely optional, so a partial update (the customer only gives an updated address)
passes `$name = NULL` meaning "not provided," and an unconditional `SET` would silently null the
already-stored name, defeating AC-2. The fix is `coalesce($field, c.field)` per data field, so a
`NULL` argument means "leave this field as it is," never "clear it." `profileUpdatedAt` alone stays
an unconditional `SET` — every call that reaches this query touched the profile, partial or full.
**`NULL` has no way to *clear* a field through this query** — nothing in FR-1..FR-4 asks for
clearing a previously-captured value, only updating one; a caller must never pass `''` to mean "no
value" (an empty string is a value, not "not provided").

Live-verified in both partial-update directions (graph note §0/§3, `ws_cartprobe_profile_check2`,
disposable, deleted after): full write (`name='Alice', deliveryAddress='123 Main St'`) → partial
call with `$name=NULL, $deliveryAddress='456 New Ave'` → `name='Alice'` (**preserved**),
`deliveryAddress='456 New Ave'` (updated) → partial call with `$name='Bob', $deliveryAddress=NULL`
→ `name='Bob'` (updated), `deliveryAddress='456 New Ave'` (**preserved**). `count(Customer) = 1`
throughout both partial calls — no duplicate node from either `MERGE`.

### 17.2 `read_profile` — the one code path for "unset"

```cypher
// $customerId
MATCH (c:Customer {customerId: $customerId})
RETURN c.name AS name, c.deliveryAddress AS deliveryAddress, c.profileUpdatedAt AS profileUpdatedAt
```
Zero rows ⇒ no `Customer` node at all yet for this id (nobody has ever interacted with this
workspace as this customer — cart, order, or profile). A row with `name`/`deliveryAddress` both
`NULL` ⇒ the customer exists (e.g. from cart activity) but has never given a name/address. Both
cases mean "ask them" to the calling agent — the distinction is preserved here (never collapsed to
one shape), per the note's own §4, even though nothing in this requirements doc currently needs
"brand-new" told apart from "returning, no profile yet."

**Fully implemented (K-054 M6, plan §4 steps 2-3):** `Repository.upsert_profile`/`get_profile`,
`Services.get_profile`/`save_profile`, `GetProfileTool`/`SaveProfileTool`, the `salesperson@v3`
version bump (`falkorchat.proof_defs.SALESPERSON_DEF`), and seed/verify data
(`scripts/seed_salesperson.sh`/`verify_salesperson.sh`, default version bumped to `v3`).

---

## 18. Storefront participants & resets (salesperson-ui S4)

Design: `docs/plans/salesperson-ui.md` §4.3/§4.6/§4.8 (architect) and
`docs/plans/salesperson-ui-graph.md` **v1.2** (graph-dba — every query below is live-verified
there, three review passes, §11 is its verification log). **This section transcribes that note
verbatim; it does not re-derive it.** Workspace-scoped (`ws:{workspaceId}`); `reference` is never
referenced by any query here, which is structural — `GRAPH.QUERY` operates on one named graph per
call — not a convention.

**Cypher comments in this section are `//`.** `--` is **not** a comment on this build: it fails to
parse (`Invalid input 'G': expected '>' or '('`).

**Schema — one new nullable, unindexed property on the existing `Channel` node, no new label:**
```
(:Channel {channelId, name, createdAt,
            participantId})    // NULLable; written ONLY by §18.1 ensure_participant
```
No DDL: `bootstrap_schema.sh` is untouched and every existing workspace stays valid. A `UNIQUE`
constraint on it was considered and declined on scope (§18.4 makes a duplicate marker fail loudly
anyway); no new index either — the property is only ever a `Filter` on a `ch` already bound by a
`MEMBER_OF` traversal from an index-anchored `u`.

### 18.0 The two guards — the whole safety argument

Both resets carry two independent guards, and **neither may be relaxed for readability**:

| # | Guard | Where | What it stops |
|---|---|---|---|
| **G1** | `WHERE u.tokenHash IS NOT NULL` | the anchor, both resets | any non-participant `User` being used as a reset root — `config.USER_ID`'s lifespan node, `seed_demo.sh`'s `u1`, an operator account, an `Agent` id, an unknown id |
| **G2** | `WHERE ch.participantId = u.userId` | the channel hop, both resets | any `Channel` **not minted by this participant's own join** — even one the participant is a genuine `MEMBER_OF` |

**G2 is a provenance check, not an id check, and that distinction is the design.** The obvious
formulation — `ch.channelId = u.channelId` against the `User.channelId` denormalization — is a
*coincidence* check: it says two ids happen to be equal, and cannot tell "this channel was minted
for this participant" from "this participant's `channelId` field happens to name a pre-existing
channel". `Channel.participantId` is written **only** by §18.1. `seed_demo.sh`'s `demo-general` —
and every channel created by any other code path, seed script or human — has no `participantId` at
all, and `null = anything` evaluates to `null`, never `true`. Such a channel is **structurally
unreachable** as a delete target regardless of what any `User` property says, who is a member of
it, or how carelessly the repository method is called. `Repository.create_channel` writes a fixed
three-property map with no caller-controlled extras, so the marker cannot be forged, and nothing
in the codebase deletes a `MEMBER_OF` edge, so a participant's own channel cannot be detached
either.

**The governing rule is a *scoping* rule, not a label list** (`salesperson-ui.md` §4.8):

> Every `Channel`, `Thread` and `Message` **not reachable from a participant `User`** survives
> both resets — `demo-general`/`demo-welcome` and every message in them included. A participant
> `User` is one carrying `tokenHash`.

**A survivor assertion written by label cannot enforce it**, because victims and survivors share
the labels `Channel`, `Thread` and `Message`. Live-reproduced with both guards ablated
(`reset_participant('u2')`, where `u2` is a non-participant whose own `channelId` names
`demo-general`): `demo-welcome`, its three messages and two read-cursors are destroyed and a
replacement thread minted in their place, while the `Channel` and `Thread` label counts are
**unchanged** (3 → 3 each) and every §4.8 survivor label is still present. Only a *positive*
assertion naming `demo-general`/`demo-welcome` catches it, and it is the point of the whole
section. **Each reset has its own such assertion and they are not interchangeable** — an ablation
of one query leaves the other query's test green, so cite the one that matches the query you are
changing. Stripping G1 **and** G2 reddens exactly five tests either way, all in
`server/tests/test_repository.py` (live-verified, both directions):

| Ablated query | Tests that go red |
|---|---|
| §18.4 `reset_participant` — the `u2` ablation described above | `test_reset_participant_is_a_no_op_for_non_participants[u2]` and `[u1]`, `…_of_a_cross_member_leaves_the_demo_channel_whole`, `…_keeps_a_cursor_on_a_surviving_thread`, `…_with_a_mismatched_marker_is_a_total_no_op` |
| §18.5 `reset_all_participants` | `test_reset_all_leaves_the_non_participant_subgraph_intact`, `…_deletes_every_participant_subgraph`, `…_keeps_exactly_the_survivor_labels`, `…_reports_unscoped_participants_and_leaves_them_whole`, `…_is_idempotent_and_returns_an_all_zeros_row_when_clean` |

**Two completeness dependencies this scoping inherits.** Neither is visible from the Cypher alone:

1. **`WorkflowRun` completeness assumes the storefront deployment leaves `api.build_router`
   unmounted** (`create_app(dev_surface=False)`, `salesperson-ui.md` §4.9). Runs are reached only
   through `TRIGGERED_BY` from a thread message — the storefront's only run-creation path
   (`trigger.maybe_trigger` → `start_workflow_run`). A run started through `api.py`'s untriggered
   path carries **no `TRIGGERED_BY` edge**, so it and its `StepRun`s/`TraceEvent`s are unreachable
   by both resets. **A deployment that re-mounts the dev surface alongside participants has
   re-opened an under-delete here.**
2. **Message completeness assumes the `HEAD`/`NEXT`/`TAIL` chain invariant.** Both resets reach
   messages by the structural `HEAD` → `NEXT*0..` walk, never by `WHERE m.threadId = $tid`
   (`Message.threadId` is a deliberately unindexed denormalization, `null` on any pre-backfill
   row, and a `WHERE` on it would drag the plan onto a full `Message` label scan). §4's write
   paths link the chain inside the same guarded `FOREACH` as the `CREATE`, so an off-chain message
   is unproducible through the platform — but a hand-planted one **survives both resets**
   (verified). A future writer that creates a `Message` before linking it breaks reset
   completeness, not just thread reads.

### 18.1 `ensure_participant` — provisioning, idempotent, one atomic write

```cypher
// $participantId $displayName $tokenHash $language $channelId $threadId
// $threadTitle $agentId $now
OPTIONAL MATCH (existing:User  {userId:  $participantId})
OPTIONAL MATCH (clash:Agent    {agentId: $participantId})
OPTIONAL MATCH (agent:Agent    {agentId: $agentId})
WITH existing, clash, agent,
     (existing IS NULL AND clash IS NULL AND agent IS NOT NULL) AS doCreate
FOREACH (_ IN CASE WHEN doCreate THEN [1] ELSE [] END |
  CREATE (u:User {userId:      $participantId,
                  displayName: $displayName,
                  tokenHash:   $tokenHash,
                  language:    $language,
                  channelId:   $channelId,
                  threadId:    $threadId,
                  joinedAt:    $now})
  CREATE (c:Channel {channelId:     $channelId, name: $displayName,
                     participantId: $participantId, createdAt: $now})
  CREATE (t:Thread  {threadId:  $threadId,  title: $threadTitle,
                     createdAt: $now, updatedAt: $now})
  CREATE (c)-[:HAS_THREAD]->(t)
  CREATE (u)-[:MEMBER_OF     {role: 'member',    joinedAt: $now}]->(c)
  CREATE (agent)-[:MEMBER_OF {role: 'assistant', joinedAt: $now}]->(c)
)
RETURN doCreate                            AS created,
       existing IS NOT NULL                AS existed,
       existing.tokenHash IS NOT NULL      AS existedParticipant,
       clash    IS NOT NULL                AS collided,
       agent    IS NULL                    AS agentMissing,
       CASE WHEN doCreate THEN $channelId ELSE existing.channelId END AS channelId,
       CASE WHEN doCreate THEN $threadId  ELSE existing.threadId  END AS threadId,
       CASE WHEN doCreate THEN $language  ELSE existing.language  END AS language
```
Shape follows §2's `ensure_user` **guarded-`CREATE`-inside-`FOREACH`** idiom, extended three ways:
the whole join is **one query** (a partial join would leave an orphan `Channel` in a graph the
presenter roster reads, and would create the unscoped shape §18.5 can only skip); the demo `Agent`
is a **third precondition**; and the status row carries the resolved
`channelId`/`threadId`/`language` so a replay is a read-through (restart survival,
`salesperson-ui.md` §4.3).

**Status-row contract** — exactly one row, always (three `OPTIONAL MATCH`es, no anchor `MATCH`, so
it can never zero-row):

| `created` | `existed` | `existedParticipant` | `collided` | `agentMissing` | Meaning | Caller action |
|---|---|---|---|---|---|---|
| `true` | `false` | `false` | `false` | `false` | fresh participant written (3 nodes, 3 rels) | success |
| `false` | `true` | **`true`** | `false` | `false` | id already a participant — nothing written; the row returns the **stored** ids/language | idempotent success (restart-survival read-through) |
| `false` | `true` | **`false`** | `false` | `false` | id exists but is **not** a participant (no `tokenHash`) — nothing written; ids/language all `NULL` | **refuse: member-id collision**, exactly like the `collided` row. Without `existedParticipant` a caller following this table builds a participant record out of three `NULL`s and a token the graph never stored |
| `false` | `false` | `false` | `true` | `false` | id held by an `Agent` — nothing written | refuse: member-id collision (§2's locked namespace rule) |
| `false` | `false` | `false` | `false` | `true` | the demo `Agent` is absent — nothing written | refuse: `503`, name `seed_demo.sh` |

`Repository.ensure_participant` raises `MemberIdCollisionError` for both collision rows and returns
the status row otherwise; `agentMissing` is returned rather than raised because "the demo isn't
seeded" is a deployment-readiness answer, not a namespace violation.

**Not a token-rotation path.** A replay carrying a *fresh* `$tokenHash` returns `existed=true` and
the new hash is **not** written — correct for an idempotent ensure (§2's re-ensure never updates
properties), and the reason `set_participant_record` (§18.3) exists.

Notes: multi-`CREATE` inside `FOREACH` with cross-clause variable binding works on this build, and
an outer-bound variable (`agent`) is a legal `CREATE` relationship endpoint inside the `FOREACH`
body (contrast the known quirk that a map-projection or list-subscript expression is **not**).
Idempotency comes from the **status logic**, not from `MERGE` — `MERGE` inside `FOREACH` is not
standard OpenCypher; the `User.userId` UNIQUE constraint stays the same-label concurrency backstop
and §2's residual cross-label race applies unchanged.

### 18.2 `add_channel_member` — the label-agnostic membership write

```cypher
// $memberId $channelId $role $joinedAt
MATCH (mem) WHERE mem.userId = $memberId OR mem.agentId = $memberId
MATCH (c:Channel {channelId: $channelId})
MERGE (mem)-[r:MEMBER_OF]->(c)
  ON CREATE SET r.role = $role, r.joinedAt = $joinedAt
RETURN r.role AS role
```
§2's "Add user to channel" generalized to either member kind via the same
`userId OR agentId` anchor `advance_cursor` (§9.3) already uses, so one query covers the `User`
and `Agent` sides (`seed_demo.sh` needs both). Idempotent: a re-add keeps the original
`role`/`joinedAt`. Zero rows ⇒ one of the two anchors is missing and **nothing was written** — the
caller must not read that as success.

**Membership is not provenance.** This query cannot make a channel a participant's own: it never
touches `participantId`, so a participant added to `demo-general` here is a genuine member of a
channel that is still structurally unreachable by either reset (§18.0).

### 18.3 The participant record — read, update, roster

```cypher
// get_participant_record — $participantId
MATCH (u:User {userId: $participantId})
WHERE u.tokenHash IS NOT NULL                       // G1
RETURN u.userId AS participantId, u.displayName AS displayName,
       u.tokenHash AS tokenHash, u.channelId AS channelId,
       u.threadId AS threadId, u.language AS language, u.joinedAt AS joinedAt
```
Zero rows ⇒ unknown id **or** a `User` that is not a participant. The two collapse deliberately:
both mean "not a participant" to a token check, and an already-reset-away participant is
indistinguishable from an id that never existed. This read is what makes the **graph** the
authoritative registry — the storefront's in-process map is a read-through cache, so a server
restart is invisible to participants (`salesperson-ui.md` §4.3).

```cypher
// set_participant_record — $participantId $displayName $tokenHash $language
MATCH (u:User {userId: $participantId})
WHERE u.tokenHash IS NOT NULL                       // G1
SET u.displayName = coalesce($displayName, u.displayName),
    u.tokenHash   = coalesce($tokenHash,   u.tokenHash),
    u.language    = coalesce($language,    u.language)
RETURN u.userId AS participantId, u.displayName AS displayName,
       u.tokenHash AS tokenHash, u.channelId AS channelId,
       u.threadId AS threadId, u.language AS language, u.joinedAt AS joinedAt
```
Post-join updates only (a re-chosen language after "reset mine", a rotated token). Per-field
`coalesce()`, the same caller contract as §17.1: an omitted field (`NULL`) means "not provided,
leave unchanged", **never** "clear it" — a caller must never pass `''` to mean "no value".

**G1 here is a guard, not a filter.** Without it this query would stamp a `tokenHash` onto
`seed_demo.sh`'s `u1` and so mint a "participant" whose channel carries no `participantId` —
exactly the unscoped shape §18.5 can only skip, and one nothing else in the shipped code can
create.

**Neither `channelId` nor `threadId` is settable here**, for the same reason. Together they are the
server-resolved scope denorm the storefront reads from (`salesperson-ui.md` §4.3), and both are
decided **in-query**: the channel a participant owns is decided once, by §18.1, together with the
marker G2 reads; the thread is decided by §18.1 at join and re-decided by §18.4 at every "reset
mine", which mints the replacement thread and repoints `u.threadId` in the same atomic write. A
settable `$threadId` would be the one lever in §18 able to point one participant at *another*
participant's thread — G1 proves the target is a participant, but this query cannot prove the
thread lies inside that participant's own channel. If a caller ever needs to repoint it, the
containment check belongs in the `MATCH` alongside G1/G2
(`OPTIONAL MATCH (:Channel {channelId: u.channelId})-[:HAS_THREAD]->(t:Thread {threadId: $threadId})`,
setting only when `$threadId IS NULL OR t IS NOT NULL`), never in the calling Python.

```cypher
// list_participants (the presenter roster) — $limit
MATCH (u:User) WHERE u.tokenHash IS NOT NULL        // G1
RETURN u.userId AS participantId, u.displayName AS displayName,
       u.channelId AS channelId, u.threadId AS threadId,
       u.language AS language, u.joinedAt AS joinedAt
ORDER BY u.joinedAt ASC, u.userId ASC LIMIT $limit
```
`config.USER_ID`'s lifespan-created `User` and `seed_demo.sh`'s `u1` never appear. No `tokenHash`
in the projection — the roster is presenter-facing.

**Do NOT add a `u.userId > ''` conjunct** to "upgrade" the `Node By Label Scan` to an index scan.
It matches every row, so it has **no selectivity** — both forms are O(|`User`|) and visit every
record (measured: 52 records either way at 50 participants, sub-millisecond, with overlapping
spreads in both directions across runs). And it is **wrong**: `42 > ''` evaluates to `null` on this
build (as do `42.5 > ''`, `true > ''`, `null > ''`), so the conjunct silently drops any participant
whose `userId` is not a string — live-reproduced, and in `reset_all` that participant survived the
reset entirely while the status row reported success. The house `WHERE c.channelId > ''` pattern
(§3) is sound where the property is reliably a non-empty string; it is not worth taking here, and
in a **destructive** query it is actively dangerous. One bare predicate in all three places means
there is no second form for a future editor to copy into a reset.

### 18.4 `reset_participant` — "reset mine"

```cypher
// $participantId $newThreadId $threadTitle $now
MATCH (u:User {userId: $participantId})
WHERE u.tokenHash IS NOT NULL                       // G1
OPTIONAL MATCH (u)-[:MEMBER_OF]->(ch:Channel)
  WHERE ch.participantId = u.userId                 // G2
OPTIONAL MATCH (ch)-[:HAS_THREAD]->(t:Thread)
OPTIONAL MATCH (t)-[:HEAD]->(h:Message)-[:NEXT*0..]->(m:Message)
WITH u, ch, collect(DISTINCT t) AS threads, collect(DISTINCT m) AS msgs

UNWIND (CASE WHEN msgs = [] THEN [null] ELSE msgs END) AS mm
OPTIONAL MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(mm)
WITH u, ch, threads, msgs, collect(DISTINCT r) AS runs

UNWIND (CASE WHEN runs = [] THEN [null] ELSE runs END) AS rr
OPTIONAL MATCH (rr)-[:HAS_STEP_RUN]->(sr:StepRun)
OPTIONAL MATCH (sr)-[:TRACED]->(te:TraceEvent)
WITH u, ch, threads, msgs, runs,
     collect(DISTINCT sr) AS steps, collect(DISTINCT te) AS traces,
     [x IN threads | x.threadId] AS threadIds

OPTIONAL MATCH (rc:ReadCursor) WHERE rc.threadId IN threadIds
WITH u, ch, threads, msgs, runs, steps, traces, threadIds,
     collect(DISTINCT rc) AS tcur

OPTIONAL MATCH (u)-[:HAS_CURSOR]->(own:ReadCursor)
WITH u, ch, threads, msgs, runs, steps, traces, threadIds, tcur,
     collect(DISTINCT own) AS allOwn

UNWIND (CASE WHEN allOwn = [] THEN [null] ELSE allOwn END) AS oc
OPTIONAL MATCH (liveT:Thread) WHERE oc IS NOT NULL AND liveT.threadId = oc.threadId
WITH u, ch, threads, msgs, runs, steps, traces, tcur,
     collect(DISTINCT CASE WHEN oc IS NOT NULL
                            AND (oc.threadId IN threadIds OR liveT IS NULL)
                           THEN oc END) AS ocur
WITH u, ch, threads, msgs, runs, steps, traces,
     tcur + [x IN ocur WHERE NOT x IN tcur] AS cursors

OPTIONAL MATCH (cust:Customer {customerId: $participantId})
OPTIONAL MATCH (cust)-[:HAS_CART]->(cart:Cart)
OPTIONAL MATCH (cart)-[:HAS_ITEM]->(item:CartItem)
WITH u, ch, threads, msgs, runs, steps, traces, cursors,
     collect(DISTINCT cust) AS custs, collect(DISTINCT cart) AS carts,
     collect(DISTINCT item) AS items

OPTIONAL MATCH (:Customer {customerId: $participantId})-[:PLACED]->(o:Order)
OPTIONAL MATCH (o)-[:HAS_LINE]->(ol:OrderLine)
WITH u, ch, threads, msgs, runs, steps, traces, cursors, custs, carts, items,
     collect(DISTINCT o) AS orders, collect(DISTINCT ol) AS lines

FOREACH (c IN CASE WHEN ch IS NULL THEN [] ELSE [ch] END |
  CREATE (c)-[:HAS_THREAD]->(:Thread {threadId:  $newThreadId,
                                      title:     $threadTitle,
                                      createdAt: $now, updatedAt: $now})
)
SET u.threadId = CASE WHEN ch IS NULL THEN u.threadId ELSE $newThreadId END

WITH u, ch,
     CASE WHEN ch IS NULL THEN []
          ELSE threads + msgs + runs + steps + traces + cursors
                       + custs + carts + items + orders + lines END AS victims,
     size(threads) AS threadCount, size(msgs)    AS messageCount,
     size(runs)    AS runCount,    size(steps)   AS stepRunCount,
     size(traces)  AS traceCount,  size(cursors) AS cursorCount,
     size(orders)  AS orderCount,  size(items)   AS cartItemCount
FOREACH (v IN victims | DETACH DELETE v)
WITH ch, u, victims, ch IS NOT NULL AS scoped,
     threadCount, messageCount, runCount, stepRunCount, traceCount,
     cursorCount, orderCount, cartItemCount
RETURN scoped, u.threadId AS threadId, size(victims) AS deletedCount,
       CASE WHEN scoped THEN threadCount   ELSE 0 END AS threadCount,
       CASE WHEN scoped THEN messageCount  ELSE 0 END AS messageCount,
       CASE WHEN scoped THEN runCount      ELSE 0 END AS runCount,
       CASE WHEN scoped THEN stepRunCount  ELSE 0 END AS stepRunCount,
       CASE WHEN scoped THEN traceCount    ELSE 0 END AS traceCount,
       CASE WHEN scoped THEN cursorCount   ELSE 0 END AS cursorCount,
       CASE WHEN scoped THEN orderCount    ELSE 0 END AS orderCount,
       CASE WHEN scoped THEN cartItemCount ELSE 0 END AS cartItemCount
```
One atomic query: collect the victim set **structurally**, mint the replacement `Thread`, repoint
`User.threadId`, then delete. The participant's `User` and `Channel` **survive**, so their token
stays valid (`salesperson-ui.md` §4.8 — "reset mine" keeps the identity); their `Customer`, `Cart`,
`CartItem`s, `Order`s and `OrderLine`s do not.

**Every write is gated on `ch IS NOT NULL`**, so a participant whose own channel does not resolve
is a total no-op rather than a half-reset.

**Status row — three outcomes the caller must tell apart:**

| Outcome | Meaning | Required behaviour |
|---|---|---|
| **zero rows** | G1 rejected the id — not a participant, or already deleted (indistinguishable) | the route's existing not-a-participant handling (`404`/`401`) |
| **`scoped=false`** | G2 found no owned channel. Guaranteed no-op: empty victim list, no re-mint, `User.threadId` untouched, every per-class count forced to `0` so the row cannot read as a partial success | **`409 Conflict`** with a machine-readable code (e.g. `"unscoped_participant"`), **never `200`** — nothing was reset and nothing will be until the graph is repaired |
| **`scoped=true`** | the reset ran; `deletedCount` is the authoritative field | `200` |

**Two `Channel`s carrying the same marker make this query raise, permanently — and that is a
fail-safe.** `WITH u, ch, collect(…)` groups by `ch`, so a duplicate marker yields two rows and the
re-mint `FOREACH` fires twice with one `$newThreadId`, aborting on `Thread.threadId`'s UNIQUE
constraint with **nothing written** (verified: old thread, messages and orders all intact, new
thread absent). Propagate it as a `5xx`; **do not retry** — a retry re-raises forever and the graph
needs repair. §18.5 still collects such a participant, because it never re-mints.

**The delete is thread-scoped, not author-scoped** (`salesperson-ui.md` §4.8, locked). The
`Agent`-authored reply living inside the participant's thread **is** deleted; the identical reply
in another participant's thread is not. An author-scoped sweep would orphan those replies against a
deleted `Thread` and, the moment the `Agent` is the author, cross participant boundaries.

**`ReadCursor` is collected from two sources, and the second is narrower here than in §18.5.**
There is no `Thread`→`ReadCursor` edge (`(member)-[:HAS_CURSOR]->(:ReadCursor {cursorId:
"{memberId}:{threadId}"})`), so cursors for the threads being deleted *now* are found by a
`ReadCursor` label scan filtered on `rc.threadId IN threadIds` — the one label scan in the plan
(102 records / 0.02 ms at 50 participants), chosen for completeness over a structural walk from
channel members, which would miss a cursor whose owner has left the channel. On top of it,
`(u)-[:HAS_CURSOR]->(own)` sweeps the participant's own cursors — **but only those whose thread is
being deleted now, or no longer exists.** Here the `User` survives, so a cursor it holds on a
*surviving* thread (a cross-member participant's cursor on `demo-welcome`) is live read-state for a
membership this operation was never asked to touch, and deleting it would silently reset their read
position in a channel outside this reset's scope. §18.5 sweeps wide for the opposite reason.

**One product consequence for the service layer, not a change here:** deleting the `Customer` also
deletes the profile name the join wrote, while `User.displayName` survives — so after "reset mine"
the profile panel shows an em-dash for a participant who typed their name at join. The service
wrapper should re-call `services.save_profile(ctx, name=<User.displayName>)` immediately after the
reset (no new Cypher).

### 18.5 `reset_all_participants` — "reset everyone"

```cypher
MATCH (u:User)
WHERE u.tokenHash IS NOT NULL                       // G1
OPTIONAL MATCH (u)-[:MEMBER_OF]->(ch:Channel)
  WHERE ch.participantId = u.userId                 // G2
OPTIONAL MATCH (ch)-[:HAS_THREAD]->(t:Thread)
OPTIONAL MATCH (t)-[:HEAD]->(h:Message)-[:NEXT*0..]->(m:Message)
WITH collect(DISTINCT CASE WHEN ch IS NOT NULL THEN u END)        AS users,
     collect(DISTINCT ch)                                          AS channels,
     collect(DISTINCT t)                                           AS threads,
     collect(DISTINCT m)                                           AS msgs,
     collect(DISTINCT CASE WHEN ch IS NOT NULL THEN u.userId END)  AS pids,
     collect(DISTINCT CASE WHEN ch IS NULL THEN u.userId END)      AS unscopedIds

UNWIND (CASE WHEN msgs = [] THEN [null] ELSE msgs END) AS mm
OPTIONAL MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(mm)
WITH users, channels, threads, msgs, pids, unscopedIds, collect(DISTINCT r) AS runs

UNWIND (CASE WHEN runs = [] THEN [null] ELSE runs END) AS rr
OPTIONAL MATCH (rr)-[:HAS_STEP_RUN]->(sr:StepRun)
OPTIONAL MATCH (sr)-[:TRACED]->(te:TraceEvent)
WITH users, channels, threads, msgs, pids, unscopedIds, runs,
     collect(DISTINCT sr) AS steps, collect(DISTINCT te) AS traces,
     [x IN threads | x.threadId] AS threadIds

OPTIONAL MATCH (rc:ReadCursor) WHERE rc.threadId IN threadIds
WITH users, channels, threads, msgs, pids, unscopedIds, runs, steps, traces,
     collect(DISTINCT rc) AS tcur

UNWIND (CASE WHEN users = [] THEN [null] ELSE users END) AS uu
OPTIONAL MATCH (uu)-[:HAS_CURSOR]->(own:ReadCursor)
WITH users, channels, threads, msgs, pids, unscopedIds, runs, steps, traces,
     tcur, collect(DISTINCT own) AS ocur
WITH users, channels, threads, msgs, pids, unscopedIds, runs, steps, traces,
     tcur + [x IN ocur WHERE NOT x IN tcur] AS cursors

OPTIONAL MATCH (cust:Customer) WHERE cust.customerId IN pids
OPTIONAL MATCH (cust)-[:HAS_CART]->(cart:Cart)
OPTIONAL MATCH (cart)-[:HAS_ITEM]->(item:CartItem)
WITH users, channels, threads, msgs, pids, unscopedIds, runs, steps, traces, cursors,
     collect(DISTINCT cust) AS custs, collect(DISTINCT cart) AS carts,
     collect(DISTINCT item) AS items

OPTIONAL MATCH (c2:Customer)-[:PLACED]->(o:Order) WHERE c2.customerId IN pids
OPTIONAL MATCH (o)-[:HAS_LINE]->(ol:OrderLine)
WITH users, channels, threads, msgs, unscopedIds, runs, steps, traces, cursors,
     custs, carts, items, collect(DISTINCT o) AS orders,
     collect(DISTINCT ol) AS lines

WITH users + channels + threads + msgs + runs + steps + traces + cursors
           + custs + carts + items + orders + lines AS victims,
     unscopedIds, size(unscopedIds) AS unscopedCount,
     size(users)   AS userCount,     size(channels) AS channelCount,
     size(threads) AS threadCount,   size(msgs)     AS messageCount,
     size(runs)    AS runCount,      size(steps)    AS stepRunCount,
     size(traces)  AS traceCount,    size(cursors)  AS cursorCount,
     size(custs)   AS customerCount, size(orders)   AS orderCount
FOREACH (v IN victims | DETACH DELETE v)
RETURN userCount, channelCount, threadCount, messageCount, runCount,
       stepRunCount, traceCount, cursorCount, customerCount, orderCount,
       unscopedCount, unscopedIds
```
Identical scoping, widened to every participant `User`, and additionally deleting the `User` and
`Channel` nodes themselves — so **every participant token is invalidated** and every client is
bounced to the join screen. No thread is re-minted; §18.1 mints a fresh subgraph on the next join.

**Always exactly one status row**, including on a clean graph: the global `collect()` over an empty
match produces one all-zeros row, which gives the caller an unambiguous "nothing to do" instead of
an empty result to index into.

**The anchor is the bare `tokenHash IS NOT NULL`** — see §18.3 for why the `u.userId > ''` conjunct
is withdrawn on correctness, not just on measurement.

**An unscoped participant is skipped whole, never beheaded.** `users` and `pids` are gated on
`ch IS NOT NULL`; the skipped participants are returned as **`unscopedCount` + `unscopedIds`** and
left **intact and collectable** (`User` + `Channel` + `Thread` + messages + run subtree + commerce
all still there). Deleting such a `User` — it passes G1 — while G2 leaves its channel unmatched
would strand the whole transcript with no anchor any future reset could reach: one participant's
conversation left visible in a graph the presenter has just told a room is clean, reported as
success. That is the worse failure, so this branch degrades gracefully and **counts** instead.

**Response contract for the two anomaly signals** (they are a contract, not a logging preference):

| Signal | Required behaviour |
|---|---|
| `unscopedCount > 0` | **`200` with `incomplete: true` and `unresolved: unscopedIds`** — the reset did everything it could, so it is not an error, but the response must not read as clean. `unscopedIds` names exactly whose state is still live |
| `unscopedCount == 0` | `200`, no `incomplete` flag |

**Reversal trigger.** The unscoped branch is **unreachable on a healthy graph** — §18.1 writes the
`User`, `Channel`, marker and `MEMBER_OF` in one atomic query, `create_channel` cannot set the
marker, and nothing in the codebase deletes a `MEMBER_OF` edge. If anything ever introduces a way
to detach a participant from their own channel (a `MEMBER_OF`-deleting query, a channel-transfer
feature, an admin tool that rewrites `participantId`), this branch stops being dead and the
trade-off above must be re-made: a reachable "keeps a valid token" path is a materially different
proposition from an unreachable one.

**The `HAS_CURSOR` sweep here is wide — no liveness filter, unlike §18.4's.** The `User` is
deleted, so any cursor it owns that survived would be **unowned**: a real orphan, including one on
a surviving non-participant thread. Verified: after `reset_all`,
`MATCH (rc:ReadCursor) WHERE NOT ()-[:HAS_CURSOR]->(rc) RETURN count(rc)` is `0`.
**Collapse the `ReadCursor` `collect` with its own `WITH` *before* the `users` `UNWIND`** — the
first cut left the scan un-collapsed, the 50-row `UNWIND` multiplied it to ~5 000 rows and
`reset_all` cost ~690 ms instead of ~240 ms. That is the "collapse each block before the next
`UNWIND` expands" rule, and it bites at scale only.

**The `ReadCursor` orphan class, and the one residual deliberately left open.** `advance_cursor`
(§9.3) `MERGE`s on the **member**, not the thread, so a turn racing a reset mints a `ReadCursor`
naming a thread that no longer exists — it is the *only* in-flight write that orphans anything
(`post_message`, `record_step_and_advance` and `append_trace_event` are all anchored on nodes the
reset deleted, so they match zero rows and create nothing). Both resets' structural `HAS_CURSOR`
sweep collects the **participant-owned** case. The **`Agent`-owned** case — `assistant:th-A`, the
demo agent's cursor for a deleted participant thread — escapes both, because the `Agent` is a
survivor and sweeping *its* cursors by owner would reach `assistant:demo-welcome`. A complete sweep
exists and parses (`OPTIONAL MATCH (t:Thread {threadId: rc.threadId}) WITH rc, t WHERE t IS NULL`)
but widens the contract from "everything reachable from a participant `User`" to "every dangling
cursor in the workspace" — it also collects a *non-participant's* dangling cursor — so it is
**declined** here and recorded for a future deliberate garbage-collection job. The residual is
bounded (at most one per `(Agent, deleted thread)`), is produced only by the race the quiesce
exists to prevent, and no read path is affected (§9.4 point-looks-up a cursor by `cursorId`, and a
stale one simply never matches a live thread again).

### 18.6 Keep/delete inventory — documentation, **not** the mechanism

The mechanism is §18.0's two guards. This table lets a reader check that the guards produce the
intended outcome; it is **not** what a test should assert on (§18.0's ablation is why). Every row
was observed live.

| Label | Reset mine | Reset everyone | How it is reached / why |
|---|---|---|---|
| `User` (participant, **scoped**) | **keeps** | **deletes** | the reset root itself; deleting it invalidates the token |
| `User` (participant, **unscoped**) | keeps (total no-op) | **keeps**, counted in `unscopedCount`/`unscopedIds` | skipped whole rather than beheaded |
| `User` (non-participant) | keeps | **keeps** | G1 — no `tokenHash` |
| `Channel` (participant's own) | **keeps** | **deletes** | G2 |
| `Channel` (any other) | keeps | **keeps** | G2 — no `participantId` |
| `Thread` (in a participant channel) | **deletes**, one re-minted | **deletes** | `(ch)-[:HAS_THREAD]->` |
| `Thread` (any other) | keeps | **keeps** | unreachable from a participant `User` |
| `Message` (in a participant thread) | **deletes** | **deletes** | `HEAD` → `NEXT*0..` — incl. `Agent`-authored replies (thread-scoped, not author-scoped) |
| `Message` (any other) | keeps | **keeps** | unreachable |
| `Message` **off-chain** (has `threadId`, not in the `HEAD`/`NEXT` chain) | keeps | **keeps** | unreachable by the structural walk; unproducible through §4's write paths (§18.0 dependency 2) |
| `ReadCursor` (for a thread being deleted now, any member) | **deletes** | **deletes** | `rc.threadId IN threadIds`; orphaned by the thread delete |
| `ReadCursor` (participant-owned, naming an *already*-deleted thread) | **deletes** | **deletes** | the structural `(u)-[:HAS_CURSOR]->` sweep |
| `ReadCursor` (participant-owned, on a **surviving non-participant** thread) | **keeps** | **deletes** | the one deliberate asymmetry — §18.4/§18.5 carry the reasoning |
| `ReadCursor` (`Agent`-owned, naming an already-deleted thread) | keeps | **keeps** | documented residual, §18.5 |
| `WorkflowRun` / `StepRun` / `TraceEvent` | **deletes** | **deletes** | `TRIGGERED_BY` → `HAS_STEP_RUN` → `TRACED` (subject to §18.0 dependency 1) |
| `Customer` / `Cart` / `CartItem` | **deletes** | **deletes** | `customerId = participantId`, exact by construction |
| `Order` / `OrderLine` | **deletes** | **deletes** | `(Customer)-[:PLACED]->` → `HAS_LINE` |
| **`WorkspaceConfig`** | **keeps** | **keeps** | never matched by either query. Taking it would silently undo K-056's Ministral re-point — the single most expensive mistake available here. Positively asserted alive after both resets |
| `Document` / `Chunk` / `Entity` | **keeps** | **keeps** | survivors of *both*; never matched |
| `WorkflowDefSnapshot` / `Step` | **keeps** | **keeps** | never matched — `verify_salesperson.sh <ws>` exits 0 after `reset_all` |
| `Agent` | **keeps** | **keeps** | never matched; its `MEMBER_OF` edges into deleted channels go with `DETACH DELETE`, the node does not |
| `Product` | n/a | n/a | lives in `reference`, a different graph — structurally unreachable |

Boundary edges, verified: a surviving `Chunk` that `DERIVED_FROM` a deleted participant `Message`
keeps its `Document` and loses only that edge; a surviving `Message` with an `EMITTED` edge into a
deleted one stays intact with its other `EMITTED` edges untouched.

### 18.7 Atomicity and the failure boundary

Each reset is **one `GRAPH.QUERY`**, therefore atomic — live-proved: a reset made to violate the
`Thread.threadId` UNIQUE constraint mid-query raised and wrote **nothing** (node count identical
before and after, `User.threadId` unchanged). There is no observable half-reset state.

**The failure boundary is client-side, and it is not "nothing changed".** The module's
`TIMEOUT` argument applies to **reads only**, so a slow `reset_all` is never truncated
server-side; the real bound is `FALKORDB_SOCKET_TIMEOUT` (default **10 s**, `config.py` →
`db.py`). At ~240 ms for 50 participants that is 40× headroom, but if a reset ever crosses it
**the client raises `TimeoutError` while the server commits the delete**. A client-side timeout on
a reset therefore means *unknown*, **never** "nothing changed": re-read state and report from the
graph.

**Atomicity is per query, not per turn**, so the quiesce is application-level: "reset mine" cancels
that participant's queued turn and waits (bounded) for one in flight; "reset everyone" stops intake
(`409`), drains, then runs the single query. **The order is not interchangeable** — deleting first
and draining after produces turns that consume an LLM call and write nothing.

### 18.8 `get_customer_current_order`

```cypher
// $customerId
MATCH (cust:Customer {customerId: $customerId})-[:PLACED]->(o:Order)
WITH o ORDER BY o.placedAt DESC, o.orderId DESC LIMIT 1
OPTIONAL MATCH (o)-[:HAS_LINE]->(l:OrderLine)
RETURN o.orderId AS orderId, o.status AS status,
       o.placedAt AS placedAt, o.updatedAt AS updatedAt,
       collect({productId: l.productId, name: l.name, unitPrice: l.unitPrice,
                quantity: l.quantity, lineTotal: l.lineTotal}) AS lines,
       sum(l.lineTotal) AS total
```
**"Current" is the most recently *placed* order, whatever its status** — not "the most recent
non-terminal order". A demo participant walks `placed → fulfilled → delivered` and the whole point
is to *see* the status change; a non-terminal filter would make the order card vanish at the moment
the demo is trying to show it. A subsequent `place_order` supersedes the card.

- **Return shape deliberately mirrors §16.8's `get_order`**, so the repository's existing
  row-shaping applies unchanged — including its two quirks: a **zero-line order** yields one
  all-`NULL` placeholder entry from `collect()` rather than `[]` (filter client-side, as
  `Repository.get_order` already does), and `sum()` returns a **float** (`0.0` on an empty
  aggregation, never `NULL`) — a `float`-vs-`int` mismatch to expect wherever this feeds a response
  model.
- Zero rows means **either** no `Customer` **or** a `Customer` with no orders. That collapse is
  deliberate: `GET /shop/api/state` renders "no order" identically for both. §17.2 preserves the
  distinction for the profile if it is ever needed.
- `collect()` and `sum()` over one fan-out are safe here for the same reason §16.8 gives: `LIMIT 1`
  guarantees exactly one `o` for the whole aggregation.
- **Tie-break.** Two orders sharing a `placedAt` millisecond resolve by `orderId DESC` — arbitrary
  but *stable across polls*, which is what matters for a card that repaints every 2 s. One customer
  placing two orders inside one millisecond is not reachable through `place_order` anyway.
- Anchors on `Customer.customerId`'s existing index (`Node By Index Scan`, 1 record). No new index.

### 18.9 `order_belongs_to_customer`

```cypher
// $customerId, $orderId
OPTIONAL MATCH (cust:Customer {customerId: $customerId})-[:PLACED]->(o:Order {orderId: $orderId})
RETURN o IS NOT NULL AS owned, o.status AS status
```
Always exactly one row — `owned` is never `NULL`, so the caller never has to distinguish "no row"
from "not owned". Verified across all five cases: own order → `[true, 'placed']`; another
participant's order, a non-participant's order, an unknown order and an unknown customer → 
`[false, NULL]` in every case. `status` rides along free, letting the caller decide `404` (not
theirs / no such order) before the CAS without a second round trip.

**This ownership check is not optional.** `services.advance_order`'s CAS is keyed on `orderId`
alone, so without this gate any participant who learned another's `orderId` could cancel their
order. The storefront never exposes an `orderId` in a request body, so this is defence in depth —
but it is the only thing standing between the two.

**Implemented (salesperson-ui S4):** `Repository.ensure_participant`/`add_channel_member`/
`get_participant_record`/`set_participant_record`/`list_participants`/`reset_participant`/
`reset_all_participants`/`get_customer_current_order`/`order_belongs_to_customer`, plus the two
thin service wrappers `Services.get_current_order`/`order_belongs_to_customer` (so `storefront.py`
never holds Cypher). Integration tests: `server/tests/test_repository.py` §18.1–§18.6.
