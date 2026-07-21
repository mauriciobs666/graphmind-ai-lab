# falkor-chat — Canonical Query Library

Verified against `falkordb/falkordb:v4.18.11` (Redis 8.6.3, module `41811`) — full suite green
**256/256, 2026-07-20** (`./scripts/test_queries.sh`; 241/241 before the K-024 §12.12/§12.13 gate).

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
       author.userId, author.displayName, labels(author) AS authorType
ORDER BY m.createdAt
```
*`length(path)` is not supported in ORDER BY on this build — use `m.createdAt`
(indexed) instead. Bounded by thread length; paginate with cursor for long threads.*

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

## 10. Agent answer provenance — `EMITTED` (K-013)

The server-side AI responder posts its answer **as the `Agent`** (role derived `assistant`, K-007)
via the §4 write path, and records the retrieval seeds (§6 hybrid search) that grounded the answer
as **`(answer:Message)-[:EMITTED {score, rank}]->(seed:Message)`** provenance edges.

**Edge shape (locked, live-verified).**

- **Direction / endpoints:** `(:Message)-[:EMITTED]->(:Message)`, answer → seed. The answer is the
  subject; each cited seed is the object. Same convention as `REPLY_TO` (subject message points to
  the referenced message). The hot query — "given an agent answer, what did it cite?" — anchors on
  the answer's `msgId` (`Node By Index Scan`) and expands `EMITTED` outward to ≤k seeds; the reverse
  ("which answers cited this seed?") is the same edge type traversed inbound.
- **Properties:** `score` (the §6 cosine distance of that seed at answer time; 0 = identical) and
  `rank` (0-based position in the ranked seed list). Both are point-in-time snapshots — retrieval is
  non-deterministic as the graph grows, so the score/rank at answer time is the provenance value.
- **No index / no constraint.** Endpoints are `Message` nodes already carrying the `msgId` range
  index + uniqueness constraint; the provenance read anchors there. FalkorDB traverses the typed
  `EMITTED` edge from the anchored answer via its adjacency matrix — no relationship-property index
  is needed. Uniqueness is guaranteed structurally (see idempotency below), so no relationship
  constraint (which would also need a supporting index) is created.
- **`EMITTED` is a third, distinct edge type.** `MENTIONS_MEMBER`→`User`/`Agent` (participants, §4)
  and `MENTIONS`→`Entity` (GraphRAG co-occurrence, §6) are unrelated — do not conflate any of them.

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
// $seedIds = ranked list of seed msgIds from §6 hybrid retrieval (order = rank)
// $scoreBy = { <seedMsgId>: <cosine distance> }  — per-seed ANN score at answer time
// $rankBy  = { <seedMsgId>: <0-based rank> }
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
OPTIONAL MATCH (s:Message {msgId: sid})
WITH t, tailRel, prev, dup, author, mems, collect(DISTINCT s) AS seeds
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
  FOREACH (seed IN seeds | CREATE (m)-[:EMITTED {score: $scoreBy[seed.msgId],
                                                 rank:  $rankBy[seed.msgId]}]->(seed))
)
RETURN ok                 AS written,
       false              AS hadHead,
       dup  IS NOT NULL   AS dupMsg,
       author IS NOT NULL AS authorFound
```

*This is the §4 subsequent write with **one added block**: a second guarded `UNWIND` resolves the
seed msgIds to bound `Message` nodes (`collect(DISTINCT s)` — dedups, drops unknown seeds like the
mention block), and a nested `FOREACH` creates the `EMITTED` edges inside the guard. Status-row
contract is identical to §4 (zero rows = no TAIL → retry as first-path; `dupMsg=true` = idempotent
replay). `$seedIds = []` is a **verified no-op** — the `CASE` guard keeps the write itself alive
(a bare `UNWIND []` would collapse the row stream before the `FOREACH`), so this same query serves
non-provenance writes too.*

**Live-verified build quirks that shape this query:**
- **A map-projection cannot be a `CREATE` endpoint** (`CREATE (m)-[:EMITTED]->(rec.node)` errors:
  *"Invalid input '.'"*). The endpoint must be a **bound node variable**, so seeds are collected as
  nodes (`collect(DISTINCT s)`) and per-edge props are pulled from **map parameters keyed by the
  node's own `msgId`** (`$scoreBy[seed.msgId]`, `$rankBy[seed.msgId]`) — dynamic map-param
  indexing by a node property is verified working on this build.
- **Two sequential guarded `UNWIND`s** (mentions, then seeds) each collapse via `collect` before the
  next expands — no row multiplication. Verified.

### 10.2 Read an answer's provenance (forward — the hot path)

```cypher
MATCH (a:Message {msgId: $msgId})-[e:EMITTED]->(s:Message)
RETURN s.msgId, s.text, s.role, e.score, e.rank
ORDER BY e.rank
```
*`GRAPH.PROFILE`: `Node By Index Scan | (a:Message)` → `Conditional Traverse (a)-[:EMITTED]->(s)` —
index-anchored, no label scan. Ordered by `rank` (ascending = most influential seed first). Route
via `GRAPH.RO_QUERY`.*

### 10.3 Read which answers cited a seed (reverse)

```cypher
MATCH (a:Message)-[e:EMITTED]->(s:Message {msgId: $seedMsgId})
RETURN a.msgId, a.role, a.createdAt, e.score, e.rank
ORDER BY a.createdAt DESC
```
*Anchored on the seed's `msgId` index; traverses `EMITTED` inbound. Answers what a given message
grounded — useful for "impact"/attribution views. Route via `GRAPH.RO_QUERY`.*

---

## 11. Workflow definitions & snapshots (M3 Slice 1 — K-020 / K-021)

The workflow **definition model** lives canonically in the global **`reference`** graph as versioned,
immutable `WorkflowDef` templates; publishing a def version and **materializing** it into a workspace
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

**Idempotency & the collapse idiom (live-verified).** Publish and materialize are single-graph `MERGE`
queries — re-running the same `key@version` is a structural no-op (0 nodes/rels created). Immutability
per version comes for free from `MERGE`. Because the query has two sequential `UNWIND` blocks (steps,
then transitions), the naive shape **row-multiplies** the final `RETURN` (steps × transitions rows);
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
one row `{key, version, stepCount:4, transitionCount:4}`; run 2 → 0 created (idempotent), same row.*

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
`Node By Label Scan`. `collect(DISTINCT …)` after the `OPTIONAL MATCH` fan-out collapses to one row;
`start.key` is constant across the fan-out so the grouping is well-defined. A def with no transitions
returns `transitions: []` (11.2b yields zero rows → the app treats absence as empty). Route via
`GRAPH.RO_QUERY`.*

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
Live-verified idempotent (run 2 → 0 created). Snapshots are immutable per `(workspace, key, version)`;
re-materialize is a no-op.*

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
// $ctx = opaque serialized state ("{}" at start); $trace = bool (debug instance?);
// $maxSteps = run-level step budget (DS default 12, §7)
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
// resolves it app-side from the firing transition = "{defKey}:{version}:{toKey}")
MATCH (r:WorkflowRun {runId: $runId})-[atRel:AT_STEP]->(cur:Step)
MATCH (to:Step {stepUid: $toStepUid})
OPTIONAL MATCH (r)-[lastRel:LAST_STEP_RUN]->(prevSR:StepRun)
CREATE (sr:StepRun {stepRunId: $stepRunId, stepKey: cur.key, status: $stepStatus,
                    startedAt: $startedAt, endedAt: $endedAt,
                    input: $input, output: $output})
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

### 12.6 `link_step_emission` — `StepRun -[:PRODUCED]-> Message` (D2)

```cypher
// $stepRunId, $msgId — run AFTER the §4 chat write that created the message (two-step, accepted)
MATCH (sr:StepRun {stepRunId: $stepRunId})
MATCH (m:Message  {msgId: $msgId})
MERGE (sr)-[:PRODUCED]->(m)
RETURN sr.stepRunId AS stepRunId, m.msgId AS msgId
```
*Both endpoints anchor on their `UNIQUE` index (`stepRunId`, `msgId`); `MERGE` on the relationship makes
the link **idempotent** (a retry after a crash between the post and the link re-links exactly once — no
duplicate `PRODUCED`, verified). This is the **second** query of the deliberately two-step emission (post
the message via the guarded §4 write, then link) — the message is the durable artifact; a missing link is
a diagnosable, retry-able gap, not a torn thread (§3/§9). **Distinct from `EMITTED`** (§10) — verified a
`PRODUCED` write adds zero `EMITTED` edges.*

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
       x.startedAt AS startedAt, x.endedAt AS endedAt, x.input AS input, x.output AS output
ORDER BY x.startedAt
```
*Anchors on `Node By Index Scan | (r:WorkflowRun)`, finds the chain head via **`OPTIONAL MATCH` +
`IS NULL`** (never the broken `exists()`-in-pattern check — quirks KB), then walks `NEXT*0..`. Ordered by
the executor's monotonic `startedAt` (same lock-guarded clock as messages — ties impossible at source),
which coincides with `NEXT` order. Route via `GRAPH.RO_QUERY`.*

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
//        threadId/error are rejected service-side, see plan §3.4 M-2)
// $trace = bool; $maxSteps = run-level step budget (a process def declares its own, e.g. 24)
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
