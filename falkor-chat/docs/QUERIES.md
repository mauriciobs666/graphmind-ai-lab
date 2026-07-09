# falkor-chat — Canonical Query Library

Verified against `falkordb/falkordb:edge` (Redis 8.2.2, module `999999`).

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
