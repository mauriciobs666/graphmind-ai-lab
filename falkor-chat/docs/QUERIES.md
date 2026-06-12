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
RETURN m
```

> **Note for developer:** call this only when creating the first message.
> Check `exists((t)-[:HEAD]->())` beforehand, or use separate code paths for
> thread creation vs. reply. Two separate write paths are cleaner than a
> conditional MERGE for HEAD/TAIL.

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
RETURN m
```

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
