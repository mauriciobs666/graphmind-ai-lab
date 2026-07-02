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

### Add user to workspace (project into workspace graph)
```cypher
MERGE (u:User {userId: $userId})
ON CREATE SET u.displayName = $displayName, u.email = $email
RETURN u
```
*Graph: `ws:{id}` — keeps a workspace-local copy; only the fields needed for chat.*

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

---

## 3. Channels & threads

### Create a channel
```cypher
MERGE (c:Channel {channelId: $channelId})
ON CREATE SET c.name = $name, c.createdAt = $createdAt
RETURN c
```

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
MERGE (t:Thread {threadId: $threadId})
ON CREATE SET t.title     = $title,
              t.createdAt = $createdAt,
              t.updatedAt = $createdAt
MERGE (c)-[:HAS_THREAD]->(t)
RETURN t
```

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

### Post the first message in a thread
*Use when the thread has no HEAD yet (first message ever).*
```cypher
MATCH (t:Thread {threadId: $threadId})
MATCH (author {userId: $authorId})            // works for User or Agent node
MERGE (m:Message {msgId: $msgId})
ON CREATE SET m.text      = $text,
              m.role      = $role,
              m.createdAt = $createdAt
CREATE (t)-[:HEAD]->(m)
CREATE (t)-[:TAIL]->(m)
CREATE (m)-[:POSTED_BY]->(author)
SET t.updatedAt = $createdAt
WITH m
UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid
OPTIONAL MATCH (u:User  {userId:  mid})
OPTIONAL MATCH (a:Agent {agentId: mid})
WITH m, collect(DISTINCT coalesce(u, a)) AS mems
FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))
RETURN m
```

> **Note for developer:** call this only when creating the first message.
> Check for a HEAD beforehand with `OPTIONAL MATCH (t)-[:HEAD]->(h) RETURN h IS
> NOT NULL` — **not** `exists((t)-[:HEAD]->())`, which returns `true` even when
> the edge is absent on this build (see AGENTS.md live-verified facts). Two
> separate write paths are cleaner than a conditional MERGE for HEAD/TAIL.
>
> **Zero rows back = nothing written.** Both write paths anchor on `MATCH`
> (thread, author, TAIL); if any anchor misses, the whole query no-ops and
> returns no rows while the transport still reports success. Callers must treat
> an empty result as an error (the repository raises), and the service layer
> validates the author exists before writing.

### Post a subsequent message in a thread
*Use for every message after the first.*
```cypher
MATCH (t:Thread {threadId: $threadId})-[tailRel:TAIL]->(prev:Message)
MATCH (author {userId: $authorId})
MERGE (m:Message {msgId: $msgId})
ON CREATE SET m.text      = $text,
              m.role      = $role,
              m.createdAt = $createdAt
CREATE (prev)-[:NEXT]->(m)
DELETE tailRel
CREATE (t)-[:TAIL]->(m)
CREATE (m)-[:POSTED_BY]->(author)
SET t.updatedAt = $createdAt
WITH m
UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid
OPTIONAL MATCH (u:User  {userId:  mid})
OPTIONAL MATCH (a:Agent {agentId: mid})
WITH m, collect(DISTINCT coalesce(u, a)) AS mems
FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))
RETURN m
```

### Participant mentions — how the block above works (live-verified)

Both write paths carry a `$mentions` parameter: a flat list of member ids, each a `userId`
**or** an `agentId`. The trailing block turns each id into a `(:Message)-[:MENTIONS_MEMBER]->(member)`
edge, atomically, inside the **same** `GRAPH.QUERY` as the HEAD/TAIL write (the atomicity rule).

- **`MENTIONS_MEMBER` is distinct from `MENTIONS`.** `MENTIONS`→`Entity` is GraphRAG co-occurrence
  (§6). Participant mentions use `MENTIONS_MEMBER`. Do not conflate them.
- **Empty list is a true no-op.** `$mentions = []` → the `CASE` yields `[null]`, both `OPTIONAL
  MATCH`es miss, `collect(DISTINCT …)` drops the null → `[]`, `FOREACH` creates nothing. Verified
  byte-identical in effect to a plain post (no `MENTIONS_MEMBER` edge). The `CASE` guard is
  **required**: a bare `UNWIND []` collapses the row stream and `RETURN m` comes back empty.
- **Index-anchored member resolution.** Each id is resolved by two `Node By Index Scan`s
  (`User.userId`, `Agent.agentId`) — *not* `WHERE u.userId = mid OR u.agentId = mid`, which
  `GRAPH.PROFILE` shows degrading to an `All Node Scan`.
- **Dedup + unknown-skip are free.** `collect(DISTINCT …)` collapses duplicate ids to one edge and
  drops ids that resolve to no member (`collect` ignores nulls). `['u3','u3','a7','nope']` → 2
  edges `[u3, a7]`, one result row. Validating that unknown mentions are an error vs. silently
  dropped is a **service-layer** decision (§6 of the plan); the query itself skips them.

### Post a reply (with explicit REPLY_TO)
*Add to either write-path above:*
```cypher
// after creating (m), also link to the quoted message:
MATCH (quoted:Message {msgId: $quotedMsgId})
CREATE (m)-[:REPLY_TO]->(quoted)
```

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
RETURN m, author, quoted.msgId AS quotedId
```

---

## 5. Full-text search

### Keyword search over message text (within a workspace)
```cypher
CALL db.idx.fulltext.queryNodes('Message', $query)
YIELD node AS m, score
MATCH (t:Thread)-[:HEAD|NEXT*0..]->(m)
MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)
RETURN m.msgId, m.text, m.createdAt, score
ORDER BY score DESC
LIMIT $limit
```
*Scoped to a channel via traversal. Omit the channel MATCH to search workspace-wide.*

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

### Register an AI agent in a workspace
```cypher
MERGE (a:Agent {agentId: $agentId})
ON CREATE SET a.name      = $name,
              a.model     = $model,
              a.createdAt = $createdAt
RETURN a
```

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
thread)* `lastReadAt` timestamp; `read_messages` returns messages with `createdAt > since`, where
`since` is either supplied explicitly (pure read) or taken from the member's cursor. Mentions of
the reader are **flagged** (`isMention`), and results are **chronological**.

> **Ordering is the pagination invariant.** Since-reads must return the *earliest* messages first
> so that a `LIMIT`-truncated page is a contiguous prefix and the cursor can advance to the last
> delivered `createdAt` without skipping anything. The original mention-first sort
> (`ORDER BY isMention DESC, …`) broke this: with more unread rows than `limit`, a late mention
> crowded out earlier messages that the cursor then jumped past — silent message loss. Clients
> that want mentions surfaced first should sort by the `isMention` flag locally. Likewise the
> service advances the cursor to the newest **returned** `createdAt`, never the server clock.

Schema (bootstrap): `ReadCursor.cursorId` **range index + uniqueness constraint** (index before
constraint). `cursorId = "{memberId}:{threadId}"` — deterministic, so `MERGE` is safe and unique.

**Reader-match convention.** The reader may be a `User` (`userId`) or an `Agent` (`agentId`), so the
mention-flag and cursor queries match `me.userId = $meId OR me.agentId = $meId`. This `OR` is
acceptable **only** because `me`/`mem` is already bound by a traversal or the indexed
`cursorId`/`agentId` anchor — it is never the scan anchor (contrast the write-path resolution
above, where the same `OR` would force an `All Node Scan`). Author id is returned with
`coalesce(author.userId, author.agentId)` for the same User-or-Agent reason.

### 9.1 Read a thread since a cursor/timestamp (thread-scoped)
```cypher
MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message)
MATCH (first)-[:NEXT*0..]->(m:Message)
WHERE m.createdAt > $since                       // $since = cursor.lastReadAt, or 0 for "all"
MATCH (m)-[:POSTED_BY]->(author)
OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me)
       WHERE me.userId = $meId OR me.agentId = $meId
WITH m, author, count(me) > 0 AS isMention
RETURN m.msgId, m.text, m.role, m.createdAt,
       coalesce(author.userId, author.agentId) AS authorId,
       labels(author) AS authorType, isMention
ORDER BY m.createdAt
LIMIT $limit
```
*Anchored on the `Thread.threadId` index; walks the thread's `NEXT` chain. Chronological (see the
§9 ordering note); mentions of the reader carry `isMention = true`.*

### 9.2 Read workspace-wide since a timestamp (room-wide, no thread)
```cypher
MATCH (m:Message)
WHERE m.createdAt > $since                        // Node By Index Scan on Message.createdAt
MATCH (m)-[:POSTED_BY]->(author)
OPTIONAL MATCH (m)-[:MENTIONS_MEMBER]->(me)
       WHERE me.userId = $meId OR me.agentId = $meId
WITH m, author, count(me) > 0 AS isMention
RETURN m.msgId, m.text, m.role, m.createdAt,
       coalesce(author.userId, author.agentId) AS authorId,
       labels(author) AS authorType, isMention
ORDER BY m.createdAt
LIMIT $limit
```
*`GRAPH.PROFILE`-confirmed `Node By Index Scan | (m:Message)` on `Message.createdAt` (not a label
scan). **TIMEOUT risk:** the live default is 1000 ms; on a large workspace keep `$limit` modest and
consider a bounded `$since` window. No room-wide cursor in M1 — this variant requires an explicit
`$since` (service defaults it to `0`/epoch).*

### 9.3 Advance a read-cursor (RW — only in `advance` mode)
```cypher
MATCH (mem) WHERE mem.userId = $meId OR mem.agentId = $meId
MERGE (mem)-[:HAS_CURSOR]->(rc:ReadCursor {cursorId: $cursorId})
ON CREATE SET rc.memberId = $meId, rc.threadId = $threadId
SET rc.lastReadAt = CASE WHEN $now > coalesce(rc.lastReadAt, 0) THEN $now
                        ELSE rc.lastReadAt END    // monotonic — never moves backward
RETURN rc.lastReadAt
```
*`cursorId = "{meId}:{threadId}"`; `MERGE` backed by the `ReadCursor.cursorId` uniqueness
constraint. The `CASE`/`coalesce` monotonic guard is **live-verified** on this build (advance 300 →
stale 200 stays 300 → 400). The service owns `$now` — the newest `createdAt` it actually delivered,
never the server clock and never client-supplied — and may short-circuit before writing (an empty
page advances nothing). When the member node doesn't exist the anchor `MATCH` yields no rows and
the query is a **no-op returning no row** — callers must not index into an empty result (the
repository returns `None`). This is a write — cannot route to a replica; use only when
`advance=true`.*

> **Member-match caveat here.** §9.3's opening `MATCH (mem) WHERE mem.userId=$meId OR mem.agentId
> =$meId` *is* an anchor, so on paper it risks an `All Node Scan`. It is acceptable in practice
> because the write is a single-member point operation gated by the subsequent `MERGE` on the
> indexed `cursorId`; the candidate set is one node. If a large-workspace `GRAPH.PROFILE` ever
> shows this hurting, split into two label-specific `OPTIONAL MATCH`es + `coalesce` (the write-path
> pattern) before the `MERGE`.

### 9.4 Read a cursor (to compute `since` when not supplied)
```cypher
MATCH (rc:ReadCursor {cursorId: $cursorId})
RETURN rc.lastReadAt
```
*Single `Node By Index Scan` point-lookup on `ReadCursor.cursorId`. When no cursor exists, this
returns no row and the service treats `since` as `0`/epoch.*
