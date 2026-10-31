# falkor-chat — Canonical Query Library

Verified against `falkordb/falkordb:v4.18.11` (module `41811`) by `./scripts/test_queries.sh`.
The Cypher here is the text `server/falkorchat/repository.py` sends (the agent-generated
read path is `querygen.py`, §15.3); where a section shows a library query no server path runs,
it says so.

All queries use **parameters** — never interpolate user input into Cypher strings.
In `falkordb-py`: `g.query(cypher, params={"key": value})`.

Notation: `$param` = query parameter supplied by the caller.

---

## 1. Workspace setup

Schema bootstrap and workspace creation: `DESIGN.md` §7.1. Every query below assumes the indexes
and constraints are OPERATIONAL.

---

## 2. Users & membership

### Create or update a user (global identity graph)

*Library query: no server path runs it — the server never writes the `identity` graph; the
workspace-local copy below is what chat reads.*
```cypher
MERGE (u:User {userId: $userId})
ON CREATE SET u.displayName = $displayName,
              u.email       = $email,
              u.createdAt   = $createdAt
ON MATCH  SET u.displayName = $displayName
RETURN u
```
*Graph: `identity`*

### Add user to workspace — guarded ensure (`ensure_user`)

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
| `false` | `true` | `false` | id already a `User` — nothing written | idempotent success (re-ensure never updates properties) |
| `false` | `false` | `true` | id held by an `Agent` — **nothing written** | refuse (member-id collision error) |
| `false` | `true` | `true` | both labels hold the id | alarm — corrupted namespace, manual repair |

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

*Suite-only (`test_queries.sh`); `scripts/seed_demo.sh` writes a role-less variant (both
`MEMBER_OF` edges in one statement, no `role`/`joinedAt`). The server's own membership write is §18.2
`add_channel_member`.*
```cypher
MATCH (u:User    {userId:    $userId})
MATCH (c:Channel {channelId: $channelId})
MERGE (u)-[r:MEMBER_OF]->(c)
ON CREATE SET r.role = $role, r.joinedAt = $joinedAt
RETURN r
```

### List channel members

*Library query (suite-verified); the server reads the roster per thread (next query).*
```cypher
MATCH (u)-[:MEMBER_OF]->(c:Channel {channelId: $channelId})
RETURN coalesce(u.userId, u.agentId) AS memberId,
       u.displayName                 AS displayName,
       labels(u)                     AS type
ORDER BY u.displayName
```
*Returns both `User` and `Agent` members. `coalesce(u.userId, u.agentId)` gives a single
stable identifier regardless of node type; `labels(u)` lets the caller distinguish.*

### List thread participants (`list_thread_participants`)

```cypher
// $threadId — a thread's participants = its parent channel's roster. MEMBER_OF exists only
// at Channel granularity (no Thread-level membership edge); this is a visibility roster, not
// a "who can be @mentioned here" set.
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
`["Agent"]`, which is all the caller needs to derive `kind`).*

**`GRAPH.PROFILE`** (3-member roster — 2 `User` + 1 `Agent`) — anchors on `Node By Index Scan | (t:Thread)`
(`Thread.threadId`), then two forward `Conditional Traverse` hops (`HAS_THREAD` backward to
`Channel`, `MEMBER_OF` forward to each member) — **no label scan anywhere**:

```
Results | Records produced: 3
    Sort | Records produced: 3
        Project | Records produced: 3
            Conditional Traverse | Records produced: 3
                Node By Index Scan | (t:Thread) | Records produced: 1
```

A thread whose channel has zero members returns an empty result, not an error. Uses only the
`Thread.threadId` index; `MEMBER_OF` carries no index of its own (traversal-only).

### Member-kind lookup (author/mention validation + role derivation) (`resolve_member_kinds`)
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
*Plain `CREATE` (`create_channel`): `channelId` is a server-minted uuid, so a `MERGE` could
never match. The `Channel.channelId` uniqueness constraint
stays as the backstop. Consequence: **creates are non-idempotent** — a retried create mints a
new id (unlike message posts, which are retry-idempotent via `dupMsg`).*

### List channels in a workspace (`list_channels`)
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

### Create a thread (`create_thread`)
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

### List recent threads in a channel (`list_threads`)
```cypher
MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t:Thread)
RETURN t.threadId AS threadId, t.title AS title, t.updatedAt AS updatedAt
ORDER BY t.updatedAt DESC
LIMIT $limit
```
*Anchored on `Channel.channelId`; sorted on `Thread.updatedAt`, which every message write
bumps (§4).*

### Existence checks (`channel_exists`, `thread_exists`)
```cypher
// channel_exists
OPTIONAL MATCH (c:Channel {channelId: $channelId}) RETURN c IS NOT NULL
```
```cypher
// thread_exists
OPTIONAL MATCH (t:Thread {threadId: $threadId}) RETURN t IS NOT NULL
```
*One row always (the `OPTIONAL MATCH` never zero-rows), read-only, index point lookups. The
service pre-validates with these before a write whose missing anchor would otherwise no-op.*

---

## 4. Messages

### Post the first message in a thread (`post_first_message` — self-guarding)
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

### Post a subsequent message in a thread (`post_subsequent_message` — self-guarding)
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

Both paths **always return a status row** when their anchor matches, so zero rows
unambiguously means "anchor missing":

| Result | Meaning | Service action |
|---|---|---|
| zero rows | thread missing (first) / no TAIL yet (subsequent) | first-path: 404; subsequent: retry as first-path |
| `written=true` | committed | success |
| `dupMsg=true` | msgId already exists | **idempotent success** (retry replay) |
| `hadHead=true` (first path) | lost the first-post race | re-dispatch as subsequent |
| `authorFound=false` | unknown member | 4xx, nothing written |

### Write-path invariants (live-verified)

- **Why a guard, not `MERGE`.** A `MERGE`d message lets a retry replay match the existing node
  and re-run the unconditional `CREATE`/`DELETE` clauses — NEXT self-loops, duplicated
  `POSTED_BY`, corrupted TAIL — and lets two racing first-posts both see "no HEAD" and build two
  chains. Each path therefore guards its write *inside its single `GRAPH.QUERY`* via
  `FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END | …)` — a guard on each path, never a
  conditional merge of the two paths (locked decision). The message is a guarded `CREATE`; the
  `Message.msgId` uniqueness constraint is the concurrency backstop (constraint-violation
  rollback is verified all-or-nothing).
- **Author resolution is label-specific** (two indexed `OPTIONAL MATCH`es + `coalesce`). A
  label-less `MATCH (author {userId: $authorId})` profiles as an `All Node Scan` **and** misses
  Agent authors (Agents carry `agentId`). Users and Agents both author messages.
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

*Library query, suite-verified. No server write path creates `REPLY_TO`; the server
only reads it (`get_message`, below).*

### Read a full thread (in order) (`read_thread`)
```cypher
MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message)
MATCH (first)-[:NEXT*0..]->(m:Message)
MATCH (m)-[:POSTED_BY]->(author)
RETURN m.msgId AS msgId, m.text AS text, m.role AS role, m.createdAt AS createdAt,
       coalesce(author.userId, author.agentId) AS authorId,
       author.displayName AS displayName, labels(author) AS authorType,
       coalesce(m.toolsUsed, []) AS toolsUsed
ORDER BY m.createdAt
```
*`length(path)` is not supported in ORDER BY on this build — use `m.createdAt`
(indexed) instead. Bounded by thread length; paginate with cursor for long threads.*

*`toolsUsed` is the tool-use breadcrumb: `link_step_emission` (§12.6) stamps it with the
producing step's own non-`post_message` dispatched tool names; `coalesce(…, [])` reads a message
that was never stamped (no property) and a step that dispatched nothing as the same empty list.
`executor._assemble_messages` reads it back to tag a tool-backed `assistant` turn when the
thread is replayed into a later turn's prompt, so a small model sees its own tool-use
precedent.*

### Read a thread window (cursor-based pagination)

*Library query (suite-verified); the server pages with the §9.1 since-read instead.*
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

### Get a single message (`get_message`)
```cypher
MATCH (m:Message {msgId: $msgId})
OPTIONAL MATCH (m)-[:POSTED_BY]->(author)
OPTIONAL MATCH (m)-[:REPLY_TO]->(quoted:Message)
RETURN m.msgId, m.text, m.role, m.createdAt,
       coalesce(author.userId, author.agentId) AS authorId,
       labels(author) AS authorType, m.threadId AS threadId,
       quoted.msgId AS quotedId
```
*Zero rows → the repository returns `None`.*

### Has the thread a first message? (`thread_has_head`)
```cypher
MATCH (t:Thread {threadId: $threadId})
OPTIONAL MATCH (t)-[:HEAD]->(h)
RETURN h IS NOT NULL AS hasHead
```
*Picks the first-vs-subsequent write path. `exists((t)-[:HEAD]->())` returns true even with no
HEAD edge on this build and `count{ }` is unsupported, so the `OPTIONAL MATCH` + `IS NOT NULL`
form is the reliable existence check.*

### 4.x Backfill `threadId` (one-off, verified idempotent)

The write paths stamp `threadId` inline; older messages may lack it.
`scripts/backfill_thread_ids.sh` runs this once per workspace. Per-thread variant
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
message unreachable from a HEAD is not backfilled — acceptable, since such messages are already invisible to thread
reads. Until backfilled, such rows return
`threadId: null` in §9.2/§5 — clients must tolerate null.*

---

## 5. Full-text search

### Keyword search over message text (within a workspace) (`search_messages`)
```cypher
CALL db.idx.fulltext.queryNodes('Message', $query)
YIELD node AS m, score
RETURN m.msgId AS msgId, m.threadId AS threadId, m.text AS text,
       m.createdAt AS createdAt, score
ORDER BY score DESC
LIMIT $limit
```
*Workspace-wide — the server's search surface. A channel-scoped library variant inserts
`MATCH (t:Thread)-[:HEAD|NEXT*0..]->(m) MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)`
after the `YIELD`. `m.threadId` (denormalized) lets clients jump from a hit to its thread
without a traversal — `null` on a row the §4.x backfill has not reached.*

---

## 6. GraphRAG hybrid retrieval

The AI participant answers a question using vector similarity + graph traversal
in a single read-only query (`hybrid_search`, `GRAPH.RO_QUERY`).

```cypher
// $qVec      = the query embedding as a float list (same dim as the vector index)
// $k         = number of ANN neighbors to seed from (default 10)
// $channelId = present only in the channel-scoped variant; the workspace-wide
//              variant drops the `MATCH (c:Channel …)` line
// score      = cosine distance: 0 = identical, lower = more similar → ORDER BY score ASC

CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec))
YIELD node AS seed, score
MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed)
MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)
OPTIONAL MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message)
WITH seed, score, collect(DISTINCT related)[..5] AS expanded
RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, score,
       [m IN expanded | m.text] AS relatedContext
ORDER BY score ASC
LIMIT $limit
```

*Rows come back most-similar first — do not re-sort. ANN recall is approximate: kNN may return
fewer than `$k` seeds. Nothing writes `(:Message)-[:MENTIONS]->(:Entity)`, so `relatedContext`
is `[]` and the expansion no-ops. The service may pass a per-query `timeout` (ms). Document
chunks are searched separately (§14.3).*

### Set a message embedding (async, after posting)
```cypher
MATCH (m:Message {msgId: $msgId})
SET m.embedding = vecf32($embedding)
```
*`set_embedding`, run from the embedding worker after the message is posted — the message is
readable before the embedding lands. The repository checks `len($embedding)` against the
expected dimension **before** the `SET` and raises `EmbeddingDimensionError` on a mismatch: a
wrong-dimension `vecf32` write is accepted by the engine but drops the node out of the ANN
index for good. Returns whether a property was set (no matching message → `False`).*

### 6.1 Introspect a workspace's vector-index dimension (`read_index_dimension`)

`EmbeddingWorker` (`server/falkorchat/embedding.py`) runs this for the label it is about to
write (`Message` or `Chunk`) **before** calling the embedder at all — a mismatch raises
`EmbeddingDimensionError` with no HTTP call made and no vector written.

```cypher
// [verified] GRAPH.RO_QUERY ws:{id} — replica-routable, zero write risk, never
// creates the graph as a side effect (see the "graph key absent" row below).
// $label = the label about to be written ('Message' or 'Chunk');
// $prop  = 'embedding' (the only vector property).
CALL db.indexes() YIELD label, types, options
WHERE label = $label AND types[$prop] = ['VECTOR']
RETURN options[$prop].dimension AS dim
```

Four edge behaviours collapse to two outcomes at the repository layer (a real `int`, or `None` for every "no vector
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

**Caching:** `EmbeddingWorker` caches the returned
dimension per `(ws, label)` for the process lifetime — the index dimension
cannot change in place (only an out-of-band drop+recreate changes it) — but **never caches a `None`**, since an un-bootstrapped workspace
can become bootstrapped without a process restart.

---

## 7. Agents

### Register an AI agent in a workspace — guarded ensure (`ensure_agent`)

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

*Suite-only (`test_queries.sh`); `scripts/seed_demo.sh` writes a role-less variant (see "Add user to
channel"). The server uses §18.2 `add_channel_member`.*
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
*Run in redis-cli. Look for `Node By Label Scan` / `All Node Scan` (bad) vs `Node By Index Scan`
(good).*

### Slow query log
```
GRAPH.SLOWLOG ws:acme
```

### Liveness probe (`ping`)
```cypher
RETURN 1
```
*`GRAPH.RO_QUERY` against the workspace graph; healthy when the single value is `1`.*

---

## 9. Read-cursors & since-reads (MCP `read_messages`)

Per-agent read state for the MCP `read_messages` tool. A `ReadCursor` node holds a per-*(member,
thread)* `(lastReadAt, lastReadMsgId)` composite pair; `read_messages` returns messages after
that point, where the base is either an explicit `since` timestamp (pure read, plain `>`) or the
member's cursor pair (composite keyset). Mentions of the reader are **flagged** (`isMention`),
and results are **chronological**.

> **Ordering is the pagination invariant.** Since-reads must return the *earliest* messages first
> so that a `LIMIT`-truncated page is a contiguous prefix and the cursor can advance to the last
> delivered row without skipping anything. A mention-first sort (`ORDER BY isMention DESC, …`)
> breaks this: with more unread rows than `limit`, a late mention crowds out earlier messages
> that the cursor then jumps past — silent message loss. Clients that want mentions surfaced
> first sort by the `isMention` flag locally.
>
> **The deterministic total order is `(createdAt, msgId)`.** A timestamp alone is not a total
> order: with plain-`>` paging, two messages in the same millisecond let the tied sibling be
> skipped at a page boundary — permanently (live-reproduced). Both since-reads
> `ORDER BY m.createdAt, m.msgId`, and the cursor advances to the newest **returned**
> `(createdAt, msgId)` pair — never the server clock. Tie-break is *lexical* msgId: within one
> millisecond, delivery order is id order, not arrival order — acceptable across writers (the
> service's monotonic clock removes same-process ties). If human-facing tie order ever matters,
> mint k-sortable ids (UUIDv7/ULID) — never re-sort pages.

Schema (bootstrap): `ReadCursor.cursorId` **range index + uniqueness constraint** (index before
constraint). `cursorId = "{memberId}:{threadId}"` — deterministic, so `MERGE` is safe and unique.
`lastReadMsgId` is a plain, unindexed property.

**Reader-match convention.** The reader may be a `User` (`userId`) or an `Agent` (`agentId`), so the
mention-flag and cursor queries match `me.userId = $meId OR me.agentId = $meId`. This `OR` is
acceptable **only** because `me`/`mem` is already bound by a traversal or the indexed
`cursorId`/`agentId` anchor — it is never the scan anchor (contrast the write-path resolution
above, where the same `OR` would force an `All Node Scan`). Author id is returned with
`coalesce(author.userId, author.agentId)` for the same User-or-Agent reason.

### 9.1 Read a thread since a cursor/timestamp (thread-scoped) (`read_thread_since`)

Two predicate forms over one body. **Cursor-driven reads** use the composite keyset
(formulation A — never skips or re-delivers, even across millisecond ties); **explicit-`since`
reads** keep the plain `>` and may re-deliver or skip *within that exact millisecond*, by contract —
agents that need lossless catch-up use cursor mode.

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
identically on this build — re-profile on engine upgrades. The repository picks the form by
`since_msg_id`: `None` → plain `>`, a string (possibly `''`) → composite keyset.*

### 9.2 Read workspace-wide since a timestamp (room-wide, no thread) (`read_ws_since`)
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
folds into the scan). `m.threadId` rides along as navigation metadata — `null` on a row the §4.x
backfill has not reached; clients must tolerate null. `author.displayName` rides along too: the
polling web client renders it in place of the raw `authorId` and tolerates `null`
(members seeded without a display name). Both since-reads carry the same column set. **TIMEOUT risk:** the live default is
1000 ms; on a large workspace keep `$limit` modest and consider a bounded `$since` window. No
room-wide cursor exists — this variant requires an explicit `$since` (service defaults it to
`0`/epoch, plain `>`).*

### 9.3 Advance a read-cursor (RW — only in `advance` mode) (`advance_cursor`)
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
a cursor with no `lastReadMsgId` — **no cursor backfill needed**. The service owns `($now, $nowMsgId)` — the
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

### 9.4 Read a cursor (to compute the composite `since` when not supplied) (`get_cursor`)
```cypher
MATCH (rc:ReadCursor {cursorId: $cursorId})
RETURN rc.lastReadAt, rc.lastReadMsgId
```
*Single `Node By Index Scan` point-lookup on `ReadCursor.cursorId`, returning the
`(lastReadAt, lastReadMsgId)` pair. When no cursor exists, this returns no row and the service
uses the epoch base `(0, '')`. A cursor with no `lastReadMsgId` reads back `(ts, null)` and the service
maps it to `''`.*

---

## 10. Agent answer provenance — `EMITTED`

The server-side AI responder posts its answer **as the `Agent`** (role derived `assistant`)
via the §4 write path, and records the retrieval seeds (§6 hybrid search, merged with the §14.3
`Chunk` ANN pool app-side) that grounded the answer as
**`(answer:Message)-[:EMITTED {score, rank}]->(seed)`** provenance edges — `seed` is a `Message` or
a `Chunk`.

**Edge shape (locked, live-verified).**

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
- **Seed-id resolution: bare-id `coalesce`, not a namespaced id scheme.** `$seedIds` resolves against both `Message.msgId` and
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

*`post_agent_answer`. The realistic path: the agent answers into a thread that already has the
triggering message, so a HEAD exists and the answer is a subsequent-path write.
`post_agent_answer_first` is the fallback for a headless thread: the §4 first-message write with
the same seed block folded in (same `hadHead` guard and status row).*

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
  next expands — no row multiplication. Verified with one `Message` seed and one `Chunk` seed in a
  single guarded write: both edges created, no interference with the mention block's own
  `UNWIND`.

### 10.2 Read an answer's provenance (forward — the hot path) (`read_provenance`)

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
matches when `s` is a `Chunk` (a `Message` has no incoming `HAS_CHUNK` edge), so a chunk seed
carries its document in the same read, one extra hop.*

*`GRAPH.PROFILE`: `Node By Index Scan | (a:Message)` →
`Conditional Traverse (a)-[:EMITTED]->(s)` → `Optional Conditional Traverse (d)` — no label
scan. Data correctness verified: a
`Message`-seeded row returns `seedKind:'Message'`, `role:'user'`, `documentId:null`; a `Chunk`-seeded
row returns `seedKind:'Chunk'`, `role:null`, `documentId`/`documentTitle` correctly populated from
the extra hop. Ordered by `rank` (ascending = most influential seed first). Route via
`GRAPH.RO_QUERY`.*

### 10.3 Read which answers cited a seed (reverse) (`read_citing_answers`)

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
note. An answer is always a `Message`, so the row shape is `{answerMsgId, role, createdAt, score,
rank}`.*

*`GRAPH.PROFILE`, including the planner-trap question: whether `a:Message`'s label triggers a `Node By Label Scan` here, now that `a` is the
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

## 11. Workflow definitions & snapshots

The workflow **definition model** lives canonically in the global **`reference`** graph as versioned
`WorkflowDef` templates; publishing a def version and **materializing** it into a workspace
graph (`ws:{id}`) as a local `WorkflowDefSnapshot` subgraph are the two write paths here (DESIGN
§6.1). These queries only build and read the definition/snapshot structure; the executor's run
queries are §12.

**Model (both graphs — structurally identical, only the root label differs):**

```
// reference graph                        // ws:{id} graph
(:WorkflowDef {key,version,name,kind})    (:WorkflowDefSnapshot {key,version,name,kind})
     -[:HAS_STEP]->(:Step)                     -[:HAS_STEP]->(:Step)
     -[:START]->(:Step)                        -[:START]->(:Step)
(:Step {stepUid,key,type,config})         (:Step {stepUid,key,type,config})
(:Step)-[:TRANSITION {on,guard,order}]->(:Step)
```

**The `Step.stepUid` identity (locked).** A step's `key` is unique only *within a def*, so it
can never back a `MERGE` (AGENTS.md: "every `MERGE` must be backed by a uniqueness constraint"). Every
Step therefore carries a synthetic **`stepUid = "{defKey}:{version}:{stepKey}"`** — globally unique
within each graph — with a range index + `UNIQUE` constraint in **both** graphs (`bootstrap_schema.sh`).
`stepUid` is the MERGE key; `key`/`type`/`config` are set on create. `Step.key` keeps its own index
(display/traversal anchor) but **no** constraint.

**`HAS_STEP` — the containment edge (locked).** `(:WorkflowDef|
:WorkflowDefSnapshot)-[:HAS_STEP]->(:Step)` gives every def/snapshot an **index-anchored** handle on
*all* its steps. It exists because the alternative scoping filter —
`from.stepUid STARTS WITH ($key + ':' + $version + ':')` — **live-profiles as a `Node By Label Scan`
+ `Filter` on this build, not an index range scan** (verified: a `STARTS WITH` on the indexed prefix
does not plan as an index scan; and the `(:WorkflowDef)-[:START]->()-[:TRANSITION*]` walk alternative
is worse — a `Cartesian Product` + `Semi Apply` that *still* label-scans `Step` and silently misses
steps unreachable from `START`). Without `HAS_STEP` there is **no** def→step edge except `START`, so
reading "all steps of a def" would scan every `Step` of every def in the graph. With `HAS_STEP`, both
the step read and the transition read anchor on `Node By Index Scan | (d:WorkflowDef)` /
`(snap:WorkflowDefSnapshot)` and traverse outward — O(steps-in-this-def), verified below.

**`config` and `guard` are opaque serialized strings (rule 8 / DESIGN §1.2).** They are stored and
returned **verbatim** and **never** filtered inside. Guard *evaluation* is run-time, in the
executor (DESIGN §6.1).

**Create-only on properties, topology-enforced by `services`.** Publish and materialize are
single-graph `MERGE` queries — a re-publish/re-materialize never updates a stored `name`/`kind`/step
`config`/transition `guard`. The queries themselves are additive on **structure**;
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
> — no torn state.

### 11.1 Publish a def version (reference — idempotent) (`publish_def`)

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
steps are distinct edges; `guard` is set-on-create only (may be empty, never a match key). The query
text is one constant, `Repository._PUBLISH_CYPHER`, formatted with the root label. Live-verified:
run 1 → 5 nodes / 9 rels (4 `HAS_STEP` + 1 `START` + 4 `TRANSITION`) / 32 props, returns
one row `{key, version, stepCount:4, transitionCount:4}`; run 2 (same content) → 0 created (idempotent),
same row. A re-publish with a **different** topology (§11 preamble) is rejected at the service layer
before reaching this query.*

> **⚠️ `$transitions = []` poisons the version — this query is deliberately *un*guarded.** Unlike the
> §4 mention block (which wraps its list in `UNWIND (CASE WHEN $x = [] THEN [null] ELSE $x END)`), the
> trailing `UNWIND $transitions` here collapses the row stream to zero rows **after** the `WorkflowDef`,
> its `Step`s and the `START` edge have already been MERGEd. The `RETURN` then yields nothing, so
> `repository.publish_def` / `materialize_snapshot` index `result_set[0]` → `IndexError` (a 500, not a
> named error) on a **half-written** def — and because publish is `MERGE … ON CREATE SET`, re-publishing
> the corrected spec on the same `(key, version)` is a silent no-op: the version is permanently wrong and
> unrepairable without deleting the subgraph. The guard is therefore **at the service layer**, not here:
> `services._validate_def_spec` rejects a zero-transition spec with `WorkflowDefSpecError` → 400, nothing
> written. Model a terminal outcome as a step with **no outgoing transition**, never as
> a def with no transitions. Any *new* caller reaching `_PUBLISH_CYPHER` without going through
> `services.publish_workflow_def` must re-do that validation itself — including `materialize_snapshot`,
> which reuses this same query unguarded (open: **K-030**, `docs/BACKLOG.md`, a Cypher-level
> `CASE`-guard fix for both callers). A second, independent service-layer guard sits beside this one —
> the topology-conflict gate (`services._check_no_structural_conflict`, 409) — see the preamble above.

### 11.2 Read a def subgraph (reference — the materialize input) (`read_def_subgraph`, `read_def_structure`)

Two focused, index-anchored reads (`_READ_META_CYPHER`, `_READ_TRANSITIONS_CYPHER`, formatted with the
root label). No `length(path)` ordering — unsupported on this build; the app reconstructs step order
from `TRANSITION.order`/topology.

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
> edge.** `start.key` is a **non-aggregated grouping key** beside `collect(DISTINCT …)`, so
> "constant across the fan-out" above is a *premise*, not a guarantee. **Verified live:** two `START`
> edges on one root ⇒ **11.2a returns two rows**, one per distinct `startKey`, each carrying the
> full `steps` collection. Consumers that take `result_set[0]` therefore pick an **arbitrary** start
> key. `_read_subgraph` (`read_def_subgraph`/`get_snapshot` — the materialize and executor input)
> does exactly that; the observability reader `_read_structure` (`read_def_structure`/
> `read_snapshot_structure`, behind the structure/diff routes) consumes **all** rows.
>
> `_read_structure` therefore returns `start_keys: list[str]` (deduplicated, in row order) and steps
> deduplicated by `key` across rows, where this section documents a scalar `startKey` — the query
> text is the same, only the Python row handling differs. The service layer renames it to
> `startKey`/`startKeys` at the REST boundary. `_read_structure` also returns `None`, rather than
> raising, when the graph key itself does not exist (`ResponseError` "empty key"). A second `START`
> edge can only come from a direct `publish_def`/`materialize_snapshot` call with a different start
> key, since both queries are additive on structure (§11 preamble).

### 11.3 List defs / get a def (reference) (`list_defs`, `get_def`)

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

### 11.4 Materialize a snapshot (workspace — idempotent) (`materialize_snapshot`)

The §11.1 constant `_PUBLISH_CYPHER` formatted with the `WorkflowDefSnapshot` root label (exact text
below), run against `ws:{id}`. The `$name/$kind/$startKey/$steps/$transitions` parameters come from the
§11.2 read of `reference` (two-phase, see the note above).

```cypher
MERGE (d:WorkflowDefSnapshot {key: $key, version: $version})
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

*Node MERGEs backed by `WorkflowDefSnapshot {key,version}` + `Step {stepUid}` constraints (both
workspace-local). Produces a snapshot subgraph **structurally identical** to the reference def.
Live-verified idempotent (run 2, same content → 0 created). Re-materialize with unchanged topology is
a no-op on write (properties are always create-only). A re-materialize whose topology differs from the
stored snapshot is rejected (`409 WorkflowDefConflictError`) before this query runs.*

### 11.5 Read a snapshot subgraph (workspace) (`get_snapshot`, `read_snapshot_structure`)

The §11.2 constants formatted with the `WorkflowDefSnapshot` root label (exact text below); anchors on
`Node By Index Scan | (d:WorkflowDefSnapshot)`.

```cypher
// meta + steps
MATCH (d:WorkflowDefSnapshot {key: $key, version: $version})
OPTIONAL MATCH (d)-[:START]->(start:Step)
OPTIONAL MATCH (d)-[:HAS_STEP]->(s:Step)
RETURN d.name AS name, d.kind AS kind, start.key AS startKey,
       collect(DISTINCT {key: s.key, type: s.type, config: s.config}) AS steps
```
```cypher
// transitions
MATCH (d:WorkflowDefSnapshot {key: $key, version: $version})-[:HAS_STEP]->(from:Step)-[tr:TRANSITION]->(to:Step)
RETURN collect({from: from.key, to: to.key, on: tr.on, guard: tr.guard, order: tr.order}) AS transitions
```

> Same conditional one-row collapse as §11.2 (live-verified on this label): **two `START` edges ⇒
> two meta rows**. See the §11.2 note.

### 11.6 List / get snapshots (workspace) (`list_snapshots`)

```cypher
// list snapshots in a workspace
// $limit
MATCH (snap:WorkflowDefSnapshot) WHERE snap.key > ''
RETURN snap.key AS key, snap.version AS version, snap.name AS name, snap.kind AS kind
ORDER BY snap.key, snap.version DESC
LIMIT $limit
```
```cypher
// get a specific snapshot (point lookup on the composite key) — library query; the server
// reads a snapshot through §11.5 instead
MATCH (snap:WorkflowDefSnapshot {key: $key, version: $version})
RETURN snap.key AS key, snap.version AS version, snap.name AS name, snap.kind AS kind
```
*Index-anchored on `WorkflowDefSnapshot.key`. Route via `GRAPH.RO_QUERY`.*

**Live-verified build quirks that shape §11:**
- **`STARTS WITH` on an indexed string prefix does NOT plan as an index scan** on this build — it
  profiles as `Node By Label Scan` + `Filter`. Scope def/snapshot subgraph reads via the `HAS_STEP`
  containment edge (index-anchored), never a `stepUid` prefix filter.
- **`STARTS WITH` with a concatenated prefix needs explicit parentheses:** `x STARTS WITH ($a + ':' +
  $b)` — without them the parser mis-associates (`STARTS WITH` binds tighter than `+`) and errors
  *"Type mismatch: expected Boolean but was String"*. Moot here (we use `HAS_STEP`) but noted.
- **Sequential `UNWIND` blocks row-multiply the final `RETURN`** unless each is collapsed with an
  aggregation (`WITH d, count(st) AS stepCount`). Verified: the collapsed form returns one clean row.

---

## 12. Workflow execution — runs, step-runs & traces

The **executor** walks a materialized `WorkflowDefSnapshot` (§11) as a `WorkflowRun` that records each
executed step as a `StepRun`. All of these live **workspace-local** in `ws:{id}` (the snapshot, the run,
the trace are one connected subgraph — no cross-graph edge). DESIGN §6.2. These queries are the 1:1
contract for `repository.py` — **method name = query name** below. Every state-move is a **single
`GRAPH.QUERY`** (atomicity, rule 4); every read anchors on an index (PROFILE-verified, no label scan).

**Model (ws:{id}) — additive to §11's snapshot:**

```
(:WorkflowRun {runId, defKey, defVersion, status, startedAt, endedAt,
               ctx, trace, maxSteps, stepCount, waitingThreadId})
(:WorkflowRun)-[:OF_DEF]->(:WorkflowDefSnapshot)        // which materialized def (§11)
(:WorkflowRun)-[:AT_STEP]->(:Step)                      // current position (cleared on terminal)
(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)              // the @mention that started it (chat runs only)
(:WorkflowRun)-[:HAS_STEP_RUN]->(:StepRun)              // membership (all step-runs of a run)
(:WorkflowRun)-[:LAST_STEP_RUN]->(:StepRun)             // TAIL pointer — the NEXT anchor
(:StepRun {stepRunId, stepKey, status, startedAt, endedAt, input, output,
           resolvedModel, modelSource, modelFallback})   // model fields: LLM steps only
(:StepRun)-[:RAN]->(:Step)                              // which def step this run-step executed
(:StepRun)-[:NEXT]->(:StepRun)                          // execution order (audit trail)
(:StepRun)-[:PRODUCED]->(:Message)                      // step-emitted chat message (NOT EMITTED)
(:StepRun)-[:TRACED]->(:TraceEvent)                     // debug-only trace record
(:TraceEvent {traceId, seq, kind, at, payload})         // debug runs only; payload = flat string
```

**Locked shape decisions:**

- **`PRODUCED`, not `EMITTED` (locked).** StepRun→Message emission is a **distinct** edge type.
  `EMITTED` is the answer→seed provenance edge (§10) — reusing it would conflate
  "this answer cited that seed" with "this step produced that message." `PRODUCED` is
  `(:StepRun)-[:PRODUCED]->(:Message)`; endpoints carry their own `stepRunId`/`msgId` unique index, so
  no relationship index/constraint is needed (same reasoning as §10's `EMITTED`).
- **`LAST_STEP_RUN` tail pointer anchors the atomic advance.** Mirrors the `Thread` HEAD/TAIL
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
//        message's thread as the resume anchor, MERGED with any caller
//        `run_ctx` — {"threadId": …} alone when the caller supplies none, otherwise
//        {"threadId": …, …caller keys}. Reserved keys (threadId/error/timerFired) are
//        rejected service-side before this write ever runs, same rule as §12.12.
// $trace = bool (debug instance?);
// $maxSteps = run-level step budget (default 12; semantics: DESIGN.md §6.2).
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

### 12.2 `record_step_and_advance` — the tail-anchored atomic advance

The engine's hot write: one query records the just-executed step as a `StepRun`, appends it to the
`NEXT` audit trail via the tail pointer, moves the tail, relinks `AT_STEP` to the transition's `to`
step, and bumps `stepCount`. **All atomic (rule 4).**

```cypher
// $runId; $stepRunId (server-minted); $stepStatus (e.g. 'done'); $startedAt,$endedAt;
// $input,$output (opaque strings); $toStepUid = the destination Step's stepUid (executor
// resolves it app-side from the firing transition = "{defKey}:{version}:{toKey}");
// $resolvedModel,$modelSource,$modelFallback — NULL for a non-LLM step
// (decision/human/wait, the offline agent-without-LLM stub)
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

**Model provenance.** `resolvedModel`/`modelSource`/`modelFallback` ride the same `CREATE` as
three nullable `StepRun` properties, written by the executor's `_run_agent_node`/`_record` for
an LLM-executing step and passed as `NULL` for every non-LLM step. **[verified]** A `NULL`
parameter **omits** the property entirely — `CREATE (s:StepRun {..., resolvedModel: $rm})`
with `rm=NULL` reports one fewer `Properties set` than a non-null case and `keys(s)` does
not include `resolvedModel`. One query shape serves LLM and non-LLM steps alike: no
branching, no extra bytes on a `decision`/`human`/`wait` StepRun. `modelSource ∈
{'workspace', 'step', 'default'}` names the precedence rung that won (a workspace override,
the step's own model, or the kind default); `modelFallback` is `true` only when `resolvedModel` is not the
first model in the chain the winning rung named, and is **omitted** (never written `false`)
on the common, non-fallback path — same "nullable, absent by default" contract as the other
two. A `StepRun` without these properties reads back `NULL` for all three — its model is
unknown and is never re-derived from today's config.

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
*The single-flight guard for concurrent human replies (DESIGN §6): two near-simultaneous replies both
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
// fail_run — $runId, $endedAt, $ctx (executor stamps a note, e.g. "step budget exceeded")
MATCH (r:WorkflowRun {runId: $runId})
OPTIONAL MATCH (r)-[atRel:AT_STEP]->()
DELETE atRel
SET r.status = 'failed', r.endedAt = $endedAt, r.ctx = $ctx
RETURN r.runId AS runId, r.status AS status
```
*`AT_STEP` ("current position") is cleared on terminal states — the audit trail is preserved by the
`HAS_STEP_RUN` set + `NEXT` chain + `LAST_STEP_RUN` (the *last executed* step is `LAST_STEP_RUN`-[:RAN]->).
`DELETE` of a **null** `OPTIONAL MATCH`ed edge is a verified no-op (re-completing a run that already has
no `AT_STEP` does not error). **Step-budget fail:** the executor compares `stepCount` (returned by
§12.2) to `maxSteps` app-side; on `stepCount > maxSteps` it calls `fail_run` — verified `failed` +
`AT_STEP` cleared + `StepRun`s retained. Zero rows = run not found (→ `WorkflowRunNotFoundError`).*

> What `maxSteps` bounds, and on which executor paths it is checked: `DESIGN.md` §6.2.

### 12.6 `link_step_emission` — `StepRun -[:PRODUCED]-> Message`

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
a diagnosable, retry-able gap, not a torn thread. **Distinct from `EMITTED`** (§10) — verified a
`PRODUCED` write adds zero `EMITTED` edges.*

*`$toolsUsed` is the producing step's
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

The three model columns project `null` for a `StepRun` that never had them written (a non-LLM
step). This is the read surface behind `GET /workflow-runs/{id}/step-runs` (`SERVER.md`).

### 12.9 `find_waiting_run_for_thread` — the resume lookup (index-anchored)

```cypher
// $threadId — resume a parked run when a human replies in its thread (DESIGN §6)
MATCH (r:WorkflowRun {status: 'waiting'})
WHERE r.waitingThreadId = $threadId
RETURN r.runId AS runId, r.status AS status
LIMIT 1
```
*Anchors on `Node By Index Scan | (r:WorkflowRun)` via the **existing `WorkflowRun.status` index**
(point lookup on value `'waiting'`), then a residual `Filter` on the denormed `waitingThreadId`.
**No new index** — the `waiting` set is tiny (at most a handful of parked conversations per workspace;
the value-index visits only `waiting` nodes, never the accumulating `done` runs), so the residual filter
is trivial. The `TRIGGERED_BY`→`Message` traversal alternative also
index-anchors on `status` but adds a `Conditional Traverse` and depends on the trigger edge surviving —
the denorm is simpler and self-contained. RAM: **zero new index**; `waitingThreadId` is one short
string property per run. Route via `GRAPH.RO_QUERY`. (A thread holds at most one `waiting` run at a time
— `LIMIT 1` is belt-and-suspenders.)*

### 12.10 `append_trace_event` — write one debug trace record

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
ordered by `(StepRun.startedAt, TraceEvent.seq)` — the full cross-step reconstruction. A
non-debug run has zero `TRACED` edges → empty result, by construction. Route via
`GRAPH.RO_QUERY`.*

### 12.12 `start_run_untriggered` — begin a run with **no** chat trigger message

Parent: **§12.1** (`start_run`). Identical, **minus** the `MATCH (trigger:Message …)` anchor and the
`CREATE (r)-[:TRIGGERED_BY]->(trigger)` edge. A `kind:'process'` run is started from
REST/API — there is no `Message`, no `Thread`, and therefore no trigger to link. Deliberately a
**second, self-contained write path** rather than an `OPTIONAL MATCH` + `FOREACH` conditional inside
§12.1: same doctrine as the §4 first/subsequent message paths, and it sidesteps the
empty-row-collapse class of bug entirely.

```cypher
// $runId,$defKey,$defVersion,$startedAt server-minted / caller-supplied
// $ctx = opaque serialized state ("{}" or the caller's initial run ctx — reserved keys
//        threadId/error/timerFired are rejected service-side, the same check that guards
//        the §12.1 chat path)
// $trace = bool; $maxSteps = run-level step budget (a process def declares its own, e.g. 24;
//                semantics: DESIGN.md §6.2).
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
with an empty `threadId`.*

**`GRAPH.PROFILE`** — one `Node By Index Scan` on
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
human/signal-input path for a `process` run: the submitted input, already merged
into the run ctx service-side, rides **inside** the CAS so the write and the flip cannot be split.

```cypher
// $runId; $ctx = the FULL merged run ctx (opaque serialized string, rule 8) — the service
// reads the current ctx, merges the validated input flat into it, and passes the result
MATCH (r:WorkflowRun {runId: $runId})
WHERE r.status = 'waiting'
SET r.status = 'running', r.waitingThreadId = '', r.ctx = $ctx
RETURN r.runId AS runId, r.status AS status
```

**Zero-row contract (live-verified — the input path rests on it):** a run that is not
`waiting` matches the node but fails the `WHERE` ⇒ **zero rows and NOTHING is written — neither the
status flip nor the ctx**. Verified by replaying the CAS with a marker ctx (`{"decision":"LOSER"}`)
against an already-`running` run: zero rows returned and `r.ctx` still holds the winner's value. So
only the CAS **winner's** ctx is ever persisted — "which input advanced the run" and "which input is in
`ctx`" can never disagree. A missing `runId` is likewise zero rows (the service distinguishes the two
cases with a prior `get_run`: `None` ⇒ 404, present-but-not-waiting ⇒ 409). The loser's input is
**rejected, never silently lost**.

**`GRAPH.PROFILE`** — point lookup on `WorkflowRun.runId`, **no label scan**; the
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

**Indexes (rule 6).** Both queries use only indexes `bootstrap_schema.sh` creates in every
workspace: `WorkflowDefSnapshot.key`/`.version`, `WorkflowRun.runId` and the `WorkflowRun` UNIQUE
`{runId}` constraint. The RAM cost is the merged `ctx` string; §12.12 stores one relationship fewer
than §12.1 (no `TRIGGERED_BY`).

**Live-verified build quirks that shape §12** (`falkordb:v4.18.11`, module `41811`):
- **The tail-pointer advance (§12.2) plans edge-anchored** — the previous `StepRun` is found by
  `Optional Conditional Traverse` of `LAST_STEP_RUN`, not a scan of the `NEXT` chain. This is what makes
  the atomic advance O(1) regardless of trail length.
- **`DELETE` of a null `OPTIONAL MATCH`ed relationship is a no-op** (§12.5) — no guard needed to
  re-complete an already-terminal run.
- **Guarded CAS via `WHERE` on the current status value** (§12.3/§12.4) gives single-flight
  suspend/resume without a lock — per-query atomicity serializes the read-modify-write.
- **`waitingThreadId` on `WorkflowRun` rides the `status` index** (§12.9) — a value-point index scan on
  `status:'waiting'` + a residual property filter, no dedicated `waitingThreadId` index.

### 12.14 `find_runs_for_thread` — every run this thread has ever had

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
epoch-ms timestamp (§12.1/§12.12) — but it is **load-bearing for the query plan**.*

**Why the vacuous predicate (live-verified, 3 `WorkflowRun` vs 20,003 `Message`).** With only
`WHERE m.threadId = $threadId`, the planner anchors on `Node By Label Scan | (m:Message)` — every
`Message` in the workspace — whatever the `MATCH` shape (single, split, or reversed direction):

```
Conditional Traverse | (r)->(m) | Records produced: 3
    Filter | Records produced: 3
        Node By Label Scan | (m:Message) | Records produced: 20003
```

A `WHERE` predicate on a pattern variable pulls the scan anchor onto that variable's label, even
when a much smaller, filter-free label sits elsewhere in the pattern (an engine fact, recorded in
`claude/graph-dba/falkordb-quirks.md`). A `WorkflowRun.startedAt` index alone does not move the
anchor. The predicate on `r` does; with the index present it becomes an index scan, and predicate
order inside `WHERE` does not matter:

```
Results | Records produced: 3
    Limit | Records produced: 3
        Sort | Records produced: 3
            Project | Records produced: 3
                Filter | Records produced: 3
                    Conditional Traverse | (r)->(m:Message) | Records produced: 3
                        Node By Index Scan | (r:WorkflowRun) | Records produced: 3
```

**Index (rule 6):** `WorkflowRun.startedAt` range index (`bootstrap_schema.sh`). `WorkflowRun`
cardinality is tiny per workspace, so its cost is negligible.

### 12.15 `read_recent_post_success` — last-N post-success sample for the `@mention` def

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

Feeds the readiness route's `postSuccess` field (`SERVER.md`): of the last `$limit` **terminal**
(`done`/`failed`) runs of the `@mention`-triggered def (`config.TRIGGER_DEF_KEY`@
`config.TRIGGER_DEF_VERSION`, default `triage`@`v1`), how many produced at least one reply
(`StepRun -[:PRODUCED]-> Message`, §12.6). `waiting`/`running` runs are excluded — they have no
verdict yet.

**Result types (`falkordb-py` 1.6.x, live-verified):** `sampleSize` is an `int`; `postedCount` is
always a **`float`** (`0.0`, `1.0`, …) and never `None`, including on an empty sample
(`[0, 0.0]`). The repository casts `postedCount` to `int`, so the response carries `1`, not `1.0`.

**Plan:** no label scan.

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

Both indexed predicates (`startedAt` range and `status IN [...]`) fold into the single
`Node By Index Scan` — a row failing either one is excluded at the scan itself — and only the
unindexed `defKey`/`defVersion` surface in `Filter`. With neither indexed predicate, the plan falls
back to `Node By Label Scan | (r:WorkflowRun)` (an engine fact, recorded in
`claude/graph-dba/falkordb-quirks.md`).

**Indexes (rule 6):** reuses `WorkflowRun.status` and `WorkflowRun.startedAt`; `defKey`/`defVersion`
stay unindexed because `WorkflowRun` cardinality is tiny per workspace.

### 12.16 `find_due_wait_candidates` — the timer sweep's read half (index-anchored)

```cypher
// $limit — every parked `wait`/`human` candidate, due-agnostic
// stepKey lets the sweep write the step-scoped `ctx.timerFired` marker and detect a candidate
// that has since parked on a different waiting step
MATCH (r:WorkflowRun {status: 'waiting'})-[:AT_STEP]->(s:Step)
OPTIONAL MATCH (r)-[:LAST_STEP_RUN]->(sr:StepRun)
RETURN r.runId AS runId, s.key AS stepKey, s.type AS stepType, s.config AS stepConfig,
       sr.startedAt AS parkedAt
LIMIT $limit
```

Feeds `Services.sweep_due_workflow_runs`: returns **every** `waiting` run, whether or not it is due
and whether or not its parked step declares a timer key — dueness is derived app-side
(`services._wait_due_at`) from `stepConfig`/`parkedAt`, never filtered in Cypher (rule 8). Anchors
on the `WorkflowRun.status` value index (the same "waiting set is tiny" argument as §12.9), then two
`Conditional Traverse`s (`AT_STEP`, `LAST_STEP_RUN`), the traversal shape `get_run` (§12.7) uses:

```
Results | Records produced: 1
    Limit | Records produced: 1
        Project | Records produced: 1
            Optional Conditional Traverse | (r)->(sr:StepRun) | Records produced: 1
                Conditional Traverse | (r)->(s:Step) | Records produced: 1
                    Node By Index Scan | (r:WorkflowRun) | Records produced: 1
```

**RAM (rule 6): no new index or node/relationship type.** The timer keys
(`config.waitForSeconds`/`waitUntil`) live inside a step's opaque `config` string; on a
timer-triggered resume the sweep writes the reserved `ctx.timerFired: "<stepKey>"` marker into
`WorkflowRun.ctx` via `resume_run_with_ctx` (§12.13).

---

## 13. Workspace configuration — model overrides

Workspace-level **configuration**, not an execution record. A singleton
`(:WorkspaceConfig {workspaceConfigId: 'default'})` node per workspace carries one nullable scalar
property per consumer kind (`agentModelOverride`, `guardModelOverride`, `embeddingModelOverride`,
`responderModelOverride`) plus two provenance fields (`modelOverrideUpdatedAt`/`By`). No edges —
it adds no adjacency matrix, and the graph key already *is* the workspace scope.

**Property-name crosswalk, load-bearing and easy to get backwards:** the properties are named after
the **workflow node type** ("agent") vs. the **chat responder class** ("responder"), which does
**not** match the server's `kind` strings 1:1. `agentModelOverride` governs the executor's
`kind="step"` consumer (the workflow's `type:'agent'` node); `responderModelOverride` governs
`kind="agent"` (`AgentResponder`, the `@mention` responder). `guardModelOverride`/
`embeddingModelOverride` match their kinds by name. The crosswalk lives once, in code:
`modelconfig.py`'s `_KIND_TO_OVERRIDE_KEY`. Precedence is entirely the resolver's
(`modelconfig.ModelGateway.resolve()`, `SERVER.md`); the graph only stores the values.

### 13.1 `write_model_overrides` — set/clear the per-kind overrides

```cypher
// [verified] MERGE on the singleton, backed by the WorkspaceConfig.workspaceConfigId
// UNIQUE constraint (bootstrap_schema.sh). GRAPH.QUERY.
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
omits the key): a write passing `guardModelOverride = NULL` over a previously-set value removes the
property — `keys(c)` no longer includes it. Every call writes all four kinds, so a caller that wants
to keep a kind must pass its current value. The constraint-backed `MERGE` never duplicates the node.

**Never write `''`** — an empty string is a *value* ("a model literally named empty"), never "no
override"; "no override" is `NULL`/property-absent alone. Not enforced in Cypher or in
`write_model_overrides` — a caller discipline.

### 13.2 `read_model_overrides` — the per-kind overrides, one code path for "unset"

```cypher
// [verified] Node By Index Scan | (c:WorkspaceConfig). GRAPH.RO_QUERY (replica-safe).
MATCH (c:WorkspaceConfig {workspaceConfigId: 'default'})
RETURN c.agentModelOverride     AS agentModel,
       c.guardModelOverride     AS guardModel,
       c.embeddingModelOverride AS embeddingModel,
       c.responderModelOverride AS responderModel
```

**[verified] zero rows and an all-`NULL` row are the same answer — "no override at any kind"** —
and `read_model_overrides` returns the identical all-`None` dict for both. A workspace never written
returns zero rows; one with two kinds set returns one row with the other two `NULL`.

```
Results | Records produced: 1
    Project | Records produced: 1
        Node By Index Scan | (c:WorkspaceConfig) | Records produced: 1
```

No label scan (rule 3) — the same `WorkspaceConfig.workspaceConfigId` index the write's `MERGE` uses.

### 13.3 DDL

`WorkspaceConfig` schema: `DESIGN.md` §5.1; its index and `UNIQUE` constraint: `DESIGN.md` §7.1. **RAM (rule 6): effectively zero** — one node per workspace, six scalar
properties, no relationship type.

---

## 14. Documents & Chunks

Schema: `DESIGN.md` §5.1; indexes, constraints and the `Chunk.embedding` vector index: `DESIGN.md`
§7.1. `Chunk.seq`/`Chunk.documentId` are plain unindexed properties (read-time ordering / navigation metadata, the same posture as
`Message.threadId`).

`Document.text` stores the full original source verbatim; `Chunk`s are pre-split app-side
(`falkorchat.chunking.split_into_chunks`) and ride in as a `$chunks` list-of-maps parameter
(`[{chunkId, text, seq}, …]`). `Document.sourceKind` is derived server-side from which label
resolved the ingesting actor (`'document'` for `User`, `'agent'` for `Agent`) — never trusted from
the caller, the same posture as `Message.role`. There is no `dup` guard: like channel/thread
creation, a retried call mints a second `Document` (the §14.9 auto tier then supersedes the earlier
copy when its normalized text is identical).

**Versioning flags** (`currentVersion`/`documentCurrent`, the version-lifecycle axis: `DESIGN.md`
§5.1). The flips are written by §14.8 confirm and the §14.9 auto tier; chunk search (§14.3) and the
update-detection lookups (§14.9) read current versions only.

### 14.1 `create_document` — Document + Chunks + HAS_CHUNK, one guarded write

*The base ingestion write. The server's ingest path runs the §14.9 superset,
`create_document_with_auto_supersede`, which embeds this same create; `create_document` itself has
no server caller.*

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
    status: 'processing', pendingJobs: 0, createdAt: $createdAt,
    currentVersion: true
  })
  CREATE (d)-[:INGESTED_BY]->(ingestor)
  FOREACH (ch IN $chunks |
    CREATE (d)-[:HAS_CHUNK]->(:Chunk {
      chunkId: ch.chunkId, text: ch.text, seq: ch.seq, documentId: $documentId,
      documentCurrent: true
    })
  )
)
RETURN ok AS written, ingestor IS NOT NULL AS ingestorFound
```

**[verified]** A 3-chunk document with a known `User` actor writes 1 `Document` + 3 `Chunk` nodes,
1 `INGESTED_BY` + 3 `HAS_CHUNK` edges, all `seq`/`text`/`chunkId` populated from `$chunks`. An
unknown actor writes nothing (`written=false, ingestorFound=false`) — the same "unknown actor ⇒
no-op, reported by a status row" contract as §4's message writes, so the service raises
`UnknownActorError` rather than reporting false success.

### 14.1a `Document.status` background-completion tracking

`Document.status` starts `'processing'` and reaches `'ready'` or `'failed'` through these two
writes. Both are internal: `background._schedule_chunk_processing` and
`background._safe_embed_chunk`/`_safe_extract` are the only callers, never a transport.

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
(one per chunk per wired worker), so `report_document_job_done` always has a real count to
decrement against. `$totalJobs <= 0` (a zero-chunk document) flips `status` straight to `'ready'`
in the same query.*

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
*Run after every chunk-level embed/extract job, success or failure — decrements the counter, then
flips `status` to `'failed'` the instant any single job fails (one failed chunk fails the whole
document), or to `'ready'` once every scheduled job has succeeded. Both guards check
`status = 'processing'` first, so the first terminal write wins: a late success after a failure, or
a stray failure after `'ready'`, no-ops. Decrement and both flips run in one `GRAPH.QUERY`, so no
read-then-write race exists across calls.*

**[verified]** (`tests/test_repository.py`, live): all-jobs-succeed reaches `'ready'` only once the
counter is drained; a single failure reaches `'failed'` immediately with other jobs outstanding;
neither terminal state reverts; `total_jobs <= 0` reaches `'ready'` immediately and does not
resurrect a `'failed'` document.

A reported failure is **post-retry**: `IngestionPipeline.extract_chunk` and
`EmbeddingWorker.embed_chunk` retry a transient provider error in place, with bounded backoff
(`retry.DEFAULT_PROVIDER_RETRY` / `retry.EMBED_CHUNK_RETRY`), so `success=False` is the job's final
outcome. A retrying job stays outstanding in `pendingJobs` until it finishes.

**`chunk_document_is_processing` — the retry liveness gate (read-only)**
```cypher
MATCH (c:Chunk {chunkId: $chunkId})<-[:HAS_CHUNK]-(d:Document)
RETURN d.status = 'processing'
```
*Run (`ro_query`) by `retry.call_with_retry`'s `keep_going` callback, from `extract_chunk` and
`embed_chunk`, only on the failure path: once before and once after each backoff sleep. `False`
(including no row: chunk or document deleted) stops the retry and re-raises the last transient
error, so a document a sibling job already failed, or a deleted one, costs no further attempts.
EXPLAIN: `Node By Index Scan (c:Chunk)` → `Conditional Traverse (c)->(d:Document)` → `Project`.*

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

Explicit properties, never a whole node (this library's convention throughout). `text` round-trips
byte-identical to the ingested input. Zero rows → `None`.

### 14.3 Chunk embeddings + `search_chunks`

Mirrors §6's `Message` embedding pattern, `Chunk` in place of `Message` — a chunk is readable before
its embedding lands, and the same §6.1 pre-flight dimension guard applies per label: a `Chunk` embed
consults only `Chunk`'s vector index, never `Message`'s.

**`set_chunk_embedding` (async, after ingestion)**
```cypher
MATCH (c:Chunk {chunkId: $chunkId})
SET c.embedding = vecf32($embedding)
```
*Run from `EmbeddingWorker.embed_chunk`, scheduled per chunk after ingestion.*

**`search_chunks` — the standalone knowledge-base search read**
```cypher
CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qVec))
YIELD node AS seed, score
WHERE seed.documentCurrent = true
RETURN seed.chunkId AS chunkId, seed.text AS text,
       seed.documentId AS documentId, seed.seq AS seq, score
ORDER BY score ASC
LIMIT $limit
```
*Only chunks of a current document version are returned — a superseded document's chunks stay in
the vector index but are filtered after the ANN step, so a search can return fewer than `$k` rows.
No scope traversal (a `Chunk` has no Thread/Channel) and no Entity expansion. `documentId` rides
denormalized on `Chunk`, so a hit reports its source document with no extra hop. `score` is cosine
distance, most similar first — do not re-sort. The service may pass a per-query `timeout` (ms).*

**Chat-grounding merge.** `Services.hybrid_search` calls this query and §6's `hybrid_search` with
the same `$qVec`/`$k`, then merges the two ranked lists **app-side** by `score` ascending and
truncates to `limit` — there is no combined-ANN Cypher shape. `channel_id` scopes only the
`Message` pool (§6); a channel-scoped call still searches the full `Chunk` index. Each merged item
is tagged app-side `seedKind: 'Message'|'Chunk'` so `AgentResponder.maybe_respond` can pass the
right id into §10.1's `$seedIds`. Tie-break: the `Message` pool is concatenated first and the sort
is stable, so an exactly-equal score keeps `Message` ahead of `Chunk`. The two read-only queries run
sequentially.

**`list_document_chunks` — internal seam, not a public query**

```cypher
MATCH (d:Document {documentId: $documentId})-[:HAS_CHUNK]->(c:Chunk)
RETURN c.chunkId AS chunkId, c.text AS text
ORDER BY c.seq
```

Called by the REST and MCP ingest handlers right after the ingest service returns, to fetch the
just-created chunks and schedule each one's background embed/extract jobs — the ingest receipt
itself does not echo chunk text. Always traversed from an anchored `Document`, never scanned.

### 14.4 MCP / REST surface

The document routes and tools, and the query each one runs: `SERVER.md` §1.4 (its `QUERIES.md` column).

### 14.5 Entities & `RELATES_TO`

One LLM call per chunk (`falkorchat.extraction.extract`) produces a
`{entities: [{name, type}], relationships: [{subject, predicate, object}]}` payload, parsed by the
fence-tolerant `llm.extract_own_line_json_object` and then schema-validated app-side.
`falkorchat.ingestion.IngestionPipeline.extract_chunk` writes the result: a fresh `Entity` per
mention (including any subject/object stub-repaired by `extraction.py`) plus its `ABOUT` edge, and
one `RELATES_TO` edge per extracted relationship. Every mention is a new node; fusion links them
afterwards with `SAME_AS` edges (§14.6), never by merging nodes.

**`create_entity` — always a NEW node**
```cypher
// $nameNormalized = case-folded + whitespace-collapsed $name, computed app-side by
// `extraction.normalize_name` — the same helper extraction's stub-repair uses.
CREATE (e:Entity {
  entityId: $entityId, name: $name, nameNormalized: $nameNormalized,
  type: $type, createdAt: $createdAt
})
RETURN e.entityId AS entityId
```
*The base entity write. The ingestion pipeline runs its §14.6 superset,
`create_entity_with_auto_match`, which embeds this same `CREATE`; `create_entity` itself has no
server caller.* Plain `CREATE`, no status row: the `Entity` id and every edge below are minted
moments apart by the same pipeline run, so a downstream `MATCH` miss would be a bug, not caller
input.

**`link_chunk_about_entity` — the `ABOUT` edge**
```cypher
MATCH (c:Chunk {chunkId: $chunkId})
MATCH (e:Entity {entityId: $entityId})
CREATE (c)-[:ABOUT]->(e)
```
Plain `CREATE`, never `MERGE` — a chunk is extracted once per ingestion; a duplicate
`(chunk, entity)` pair would be a harmless extra co-occurrence edge (no properties to conflict).

**`create_entity_relationship` — the `RELATES_TO` fact edge**
```cypher
MATCH (subj:Entity {entityId: $subjectId})
MATCH (obj:Entity  {entityId: $objectId})
CREATE (subj)-[:RELATES_TO {
  label: $label, sourceChunkId: $sourceChunkId,
  sourceDocumentId: $sourceDocumentId, createdAt: $createdAt
}]->(obj)
```
`label` is the LLM-extracted predicate, free text — **not** its own relationship type (an
unbounded, LLM-controlled set of relationship types is the risk this avoids; the same "opaque
string, parsed app-side" convention as `Step.config`/`TRANSITION.guard`). **Never deduplicated**:
every extracted fact is independent provenance and is written even if an identical edge exists;
conflicting or repeated facts are kept. `sourceChunkId`/`sourceDocumentId` trace every fact to the
chunk (and, via `HAS_CHUNK`, the document) that produced it.

`ABOUT` and `RELATES_TO` carry no index — both are always traversed *from* an anchored
`Chunk`/`Entity`. `Entity.name` (full-text) and `Entity.nameNormalized` (RANGE, no uniqueness —
distinct real entities can share a normalized name+type) back the two §14.6 fusion tiers:
full-text for the fuzzy suggestion tier, the RANGE index for the exact tier, kept off RediSearch's
tokenizer/stemmer entirely.

`background._safe_extract` isolates failure per chunk (an extraction failure never corrupts the
`Document` or blocks sibling chunks) and is scheduled per chunk next to, never chained to, that
chunk's `_safe_embed_chunk`. Extraction resolves its model as `ModelGateway` kind `extraction`.

### 14.6 Entity fusion — `SAME_AS`

Every fusion decision — auto or suggested — is a property-bearing `SAME_AS` edge, never a physical
node merge (FalkorDB has no APOC-style refactor procedure, and this codebase avoids destructive
graph surgery):

```
(:Entity)-[:SAME_AS {
  matchId, status, confidence, technique,
  createdAt, decidedAt, decidedBy,
  resuggestCount, lastResuggestedAt
}]->(:Entity)
```

`status ∈ {pending, confirmed, rejected}`. Write direction is a convention, not a semantic claim —
always `(newlyExtractedEntity)-[:SAME_AS]->(existingCandidateEntity)`; reads that need to be
direction-agnostic use an undirected pattern, reads that know the direction use a directed one.
**Every query below matches its `SAME_AS` endpoints UNLABELED (`(a)`/`(b)`, never `(a:Entity)`)**
— a bare label on either endpoint of a relationship-index-anchored query forces a full
`Node By Label Scan` on this build even though the relationship-property scan alone is fully
selective (`claude/graph-dba/falkordb-quirks.md`). The exception is `create_or_reopen_match`'s two
`MATCH` clauses that resolve `a`/`b` by `entityId` (real per-node predicates).

**`create_entity_with_auto_match` — the exact tier, folded into entity creation**

The exact-tier lookup, the entity `CREATE` and the conditional auto-link run as one atomic
`GRAPH.QUERY`: as separate round trips, two entities extracted at about the same time could each
miss the other's not-yet-committed sibling. FalkorDB/Redis serializes command execution, so two
concurrent calls on the same `(nameNormalized, type)` cannot interleave.

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

**[verified]** A new `(nameNormalized, type)` pair reports `exactMatched=false` (never self-matches
its own `CREATE`); a second call with the same pair reports `exactMatched=true` against the first;
with two eligible candidates the **older** wins. A concurrency test (real threads, a
`threading.Barrier`, separate connections) yields exactly one `SAME_AS` edge, never zero or two. No
reopen branch — a freshly created `Entity` cannot already carry a `SAME_AS` edge.

**`find_fuzzy_candidates` — the suggested tier, a separate read**

```cypher
// $fuzzyQuery is built app-side (falkorchat.fusion._fuzzy_query), one RediSearch
// 1-edit fuzzy term per name token, tokenized the way this index tokenizes Entity.name:
// 'model-bench FX1' -> '%model% %bench% %FX1%', 'A1–A15' -> '%A1–A15%'.
CALL db.idx.fulltext.queryNodes('Entity', $fuzzyQuery) YIELD node AS candidate, score
WHERE candidate.type = $type
RETURN candidate.entityId AS entityId, candidate.name AS name,
       candidate.type AS type, score
ORDER BY score DESC
LIMIT $limit
```

A missed or duplicated fuzzy suggestion under concurrent timing still lands in the reviewed
`pending` queue, so this read needs no atomic folding. The `type` filter keeps a fuzzy name hit from
surfacing a cross-type suggestion. `IngestionPipeline.fuse_entity` excludes the just-created
entity's own id before classifying — **live-confirmed necessary**: a same-connection write is
synchronously visible to the next full-text query, so an entity's own lookup returns itself.

Between word characters, RediSearch's query parser rejects every ASCII punctuation and control
character except `_` (`RediSearch: Syntax error`). So `_fuzzy_query` builds terms the way this index
tokenizes `Entity.name` (measured live, one character at a time):
- it splits on ASCII space, tab and ASCII punctuation other than `_` and `\`;
- it deletes `\` and the other ASCII control characters;
- it keeps everything else inside a term, as the index does, including non-ASCII punctuation such
  as `–`, `→` and `∅`.

Each term is therefore one the index can hold for the same name; stripping (`%modelbench%`) or
splitting on every non-word character (`%A1% %A15%`) would produce terms it never holds. The builder
drops underscore-only tokens (a lone `%_%` is a syntax error); a name left with no token makes no
call. An emoji inside a word is a known miss under every query shape. Symbol-only names and
`\`-escaped punctuation have open issues (**K-076**). The `Document` title lookup (§14.9) strips per
token instead.

**`create_or_reopen_match` — the guarded find-or-create-or-reopen write**

Called for the suggested tier only (`status='pending'`); the exact tier is
`create_entity_with_auto_match` above.

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

**Why not a bare `MERGE (a)-[:SAME_AS]->(b)`**: a direction-fixed `MERGE` only matches an edge
written in *that* direction — a suggestion written as `(b)-[:SAME_AS]->(a)` on an earlier,
opposite-order ingestion would be duplicated. The undirected `OPTIONAL MATCH` avoids that.

**[verified]** Call 1 on a fresh pair → `created=true`; call 2 (same pair) → `created=false,
reopened=false`; reject, then re-derive the pair → `created=false, reopened=true`, `resuggestCount`
bumps on the **original** `matchId` (no duplicate edge); the reversed argument order finds the same
edge.

**`confirm_match` / `reject_match` / `recheck_match` — review decisions and manual reopen**

```cypher
// confirm_match(match_id, decided_by, decided_at)
MATCH (a)-[r:SAME_AS {matchId: $matchId}]->(b)
SET r.status = 'confirmed', r.decidedAt = $decidedAt, r.decidedBy = $decidedBy
RETURN r.matchId AS matchId, r.status AS status,
       a.entityId AS entityA, b.entityId AS entityB
```

```cypher
// reject_match(match_id, decided_by, decided_at) — never deletes the edge; the
// rejected record is what makes reopen-on-corroboration possible.
MATCH (a)-[r:SAME_AS {matchId: $matchId}]->(b)
SET r.status = 'rejected', r.decidedAt = $decidedAt, r.decidedBy = $decidedBy
RETURN r.matchId AS matchId, r.status AS status,
       a.entityId AS entityA, b.entityId AS entityB
```

```cypher
// recheck_match(match_id, at) — rejected -> pending only; a no-op otherwise
// (including "no such matchId" — the WHERE guard can't tell the two apart).
MATCH (a)-[r:SAME_AS {matchId: $matchId}]->(b)
WHERE r.status = 'rejected'
SET r.status = 'pending',
    r.resuggestCount = coalesce(r.resuggestCount, 0) + 1,
    r.lastResuggestedAt = $at
RETURN r.matchId AS matchId, r.status AS status,
       a.entityId AS entityA, b.entityId AS entityB
```

Zero rows → `None`. `decidedBy` is a real `User`/`Agent` id on this path, never `'system'`, so an
audit trail can always tell an automatic decision from a human/agent one.

**`list_pending_matches` / `list_matches` — the review surface**

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

`list_matches(status=None, limit)` is the only discovery surface for the auto-merged tier
(`status='confirmed', decidedBy='system'`). **Two separate query strings, not a
`WHERE $status IS NULL OR r.status = $status` null-guard** — live-verified that idiom discards the
`SAME_AS.status` index even when `$status` is bound (`claude/graph-dba/falkordb-quirks.md`):

```cypher
// list_matches(status=<value>, limit) — filtered branch.
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
// unavoidable (no relationship-type-only scan operator on this build) — fine for
// an infrequent admin/audit call; $limit bounds the result size, not the scan cost.
MATCH (a)-[r:SAME_AS]->(b)
RETURN r.matchId AS matchId,
       a.entityId AS entityA, a.name AS nameA,
       b.entityId AS entityB, b.name AS nameB,
       r.status AS status, r.confidence AS confidence, r.technique AS technique,
       r.createdAt AS createdAt
ORDER BY r.createdAt
LIMIT $limit
```

**Indexes:** `DESIGN.md` §7.1 (relationship-scoped). A property-filtered `SAME_AS` query profiles
as `Edge By Index Scan`.

`background._safe_fuse` isolates failure per entity and runs inline from `extract_chunk`'s loop,
since a fuzzy lookup needs its entity to exist first.

### 14.7 `delete_document` / `get_document_deletion` / `list_documents`

**`delete_document` — hard delete + audit record, one write**
```cypher
MATCH (d:Document {documentId: $documentId})
WITH d, d.documentId AS did
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
WITH d, did, collect(c) AS chunks
FOREACH (ch IN chunks | DETACH DELETE ch)
DETACH DELETE d
CREATE (:DocumentDeletion {documentId: did, deletedBy: $deletedBy, deletedAt: $deletedAt})
RETURN did AS documentId
```
*`documentId` is captured into a plain scalar via `WITH` **before** the delete, since nothing can
reference the node afterwards; the audit node is created off that scalar. A row → deleted and
audited (`True`); zero rows → the document was already absent, nothing written, no audit node
(`False`). `Entity`/`RELATES_TO` are untouched — only the chunks' own `ABOUT` edges go with them.
`EMITTED` edges citing a deleted chunk, and any `SUPERSEDES` edge to or from the document, go with
the same `DETACH`.*

**`get_document_deletion` — the audit record**
```cypher
MATCH (dd:DocumentDeletion {documentId: $documentId})
RETURN dd.documentId AS documentId, dd.deletedBy AS deletedBy, dd.deletedAt AS deletedAt
```
*Keyed by the deleted `documentId` value (indexed). Zero rows → `None`, whether the document still
exists or never existed.*

**`list_documents` — summaries, oldest first**
```cypher
// current_only=True (default)
MATCH (d:Document {currentVersion: true})
OPTIONAL MATCH (d)-[:INGESTED_BY]->(actor)
RETURN d.documentId AS documentId, d.title AS title,
       d.sourceFormat AS sourceFormat, d.sourceKind AS sourceKind,
       d.status AS status, d.currentVersion AS currentVersion,
       d.createdAt AS createdAt,
       labels(actor)[0] AS ingestedByKind,
       coalesce(actor.userId, actor.agentId) AS ingestedById
ORDER BY d.createdAt
LIMIT $limit
```
*`current_only=False` is the same text with `MATCH (d:Document)`. **Two query strings, not a
null-guarded `WHERE`** — the same index-preserving idiom as `list_matches` (§14.6): a
`$currentOnly IS NULL OR d.currentVersion = true` guard drops the `Document.currentVersion` index
even when bound.*

### 14.8 `SUPERSEDES` — document versions, review decisions, history

```
(:Document)-[:SUPERSEDES {
  matchId, status, confidence, technique,
  createdAt, decidedAt, decidedBy,
  resuggestCount, lastResuggestedAt
}]->(:Document)                       // new -> old
```

The document-granularity mirror of §14.6's `SAME_AS`: same properties, same
`pending`/`confirmed`/`rejected` lifecycle, same write direction convention (new → old), and the
same **unlabeled-endpoint** discipline on every `SUPERSEDES`-anchored query (a bare `:Document`
label beside the relationship-index filter forces a label scan). Unlike `SAME_AS`, **confirming has
a graph side effect**: the old document flips `currentVersion: false` and its chunks
`documentCurrent: false`. `SUPERSEDES.matchId`/`.status` are relationship-indexed
(`bootstrap_schema.sh`); a `matchId` lookup profiles as `Edge By Index Scan`.

**`create_or_reopen_supersede_suggestion` — find-or-create-or-reopen**
```cypher
MATCH (a:Document {documentId: $newDocumentId})
MATCH (b:Document {documentId: $candidateDocumentId})
OPTIONAL MATCH (a)-[existing:SUPERSEDES]-(b)
WITH a, b, existing,
     (existing IS NULL) AS isNew,
     (existing IS NOT NULL AND existing.status = 'rejected') AS reopen
FOREACH (_ IN CASE WHEN isNew THEN [1] ELSE [] END |
  CREATE (a)-[:SUPERSEDES {
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
*`create_or_reopen_match` (§14.6) at `Document` granularity, with the same undirected lookup and
the same reopen semantics: a reopened edge keeps its original `matchId`. Called by the
suggested-tier detector (§14.9) with `status='pending'`, `technique='shingled_jaccard_overlap'` and
the Jaccard ratio as `confidence`.*

**`confirm_document_update` — confirm, and supersede the old version**
```cypher
MATCH (a)-[r:SUPERSEDES {matchId: $matchId}]->(b)
SET r.status = 'confirmed', r.decidedAt = $decidedAt, r.decidedBy = $decidedBy
SET b.currentVersion = false, b.supersededAt = $decidedAt,
    b.supersededBy = $decidedBy
WITH a, b, r
OPTIONAL MATCH (b)-[:HAS_CHUNK]->(c:Chunk)
WITH a, b, r, collect(c) AS chunks
FOREACH (ch IN chunks | SET ch.documentCurrent = false)
RETURN r.matchId AS matchId, r.status AS status,
       a.documentId AS documentA, b.documentId AS documentB
```
*One `GRAPH.QUERY`. Leaves a confirmed suggestion indistinguishable from an auto-supersede (§14.9)
except for `decidedBy`, which here is a real `User`/`Agent` id. No status guard: it confirms the
edge whatever its current status.
`documentA` = the new document, `documentB` = the superseded one. Zero rows → `None`.*

**`reject_document_update` / `recheck_document_update`**
```cypher
// reject — never deletes the edge; both documents stay current and searchable
MATCH (a)-[r:SUPERSEDES {matchId: $matchId}]->(b)
SET r.status = 'rejected', r.decidedAt = $decidedAt, r.decidedBy = $decidedBy
RETURN r.matchId AS matchId, r.status AS status,
       a.documentId AS documentA, b.documentId AS documentB
```
```cypher
// recheck — rejected -> pending only; a no-op (zero rows) for any other status or no such matchId
MATCH (a)-[r:SUPERSEDES {matchId: $matchId}]->(b)
WHERE r.status = 'rejected'
SET r.status = 'pending',
    r.resuggestCount = coalesce(r.resuggestCount, 0) + 1,
    r.lastResuggestedAt = $at
RETURN r.matchId AS matchId, r.status AS status,
       a.documentId AS documentA, b.documentId AS documentB
```

**`list_pending_document_updates` / `list_document_updates`**
```cypher
// list_pending_document_updates(limit)
MATCH (a)-[r:SUPERSEDES {status: 'pending'}]->(b)
RETURN r.matchId AS matchId,
       a.documentId AS documentA, a.title AS titleA,
       b.documentId AS documentB, b.title AS titleB,
       r.confidence AS confidence, r.technique AS technique,
       r.createdAt AS createdAt
ORDER BY r.createdAt
LIMIT $limit
```
```cypher
// list_document_updates(status=<value>, limit); status=None drops `{status: $status}`
// (two query strings, the §14.6 list_matches idiom; the unfiltered branch is a full scan)
MATCH (a)-[r:SUPERSEDES {status: $status}]->(b)
RETURN r.matchId AS matchId,
       a.documentId AS documentA, a.title AS titleA,
       b.documentId AS documentB, b.title AS titleB,
       r.status AS status, r.confidence AS confidence,
       r.technique AS technique, r.createdAt AS createdAt
ORDER BY r.createdAt
LIMIT $limit
```
*`list_document_updates(status='confirmed')` is how the auto-superseded tier (`decidedBy='system'`)
is discovered.*

**`get_document_history` — the confirmed version chain, oldest to newest**
```cypher
// 1. the anchor itself (zero rows -> [])
MATCH (d:Document {documentId: $documentId})
RETURN d.documentId AS documentId, d.title AS title,
       d.createdAt AS createdAt, d.currentVersion AS currentVersion,
       d.supersededAt AS supersededAt, d.supersededBy AS supersededBy
```
```cypher
// 2. one hop older, repeated from each result until no row
MATCH (a:Document {documentId: $documentId})
MATCH (a)-[r:SUPERSEDES {status: 'confirmed'}]->(b)
RETURN b.documentId AS documentId, b.title AS title,
       b.createdAt AS createdAt, b.currentVersion AS currentVersion,
       b.supersededAt AS supersededAt, b.supersededBy AS supersededBy
ORDER BY r.createdAt ASC LIMIT 1
```
```cypher
// 3. one hop newer, repeated likewise
MATCH (b:Document {documentId: $documentId})
MATCH (a)-[r:SUPERSEDES {status: 'confirmed'}]->(b)
RETURN a.documentId AS documentId, a.title AS title,
       a.createdAt AS createdAt, a.currentVersion AS currentVersion,
       a.supersededAt AS supersededAt, a.supersededBy AS supersededBy
ORDER BY r.createdAt ASC LIMIT 1
```
*Works from any version's id. Only `confirmed` edges count — a `pending`/`rejected` suggestion is
not history. Walked one hop at a time with single-hop, `documentId`-anchored queries rather than a
variable-length `[:SUPERSEDES*]` with a relationship-property filter (whose plan on this build is
unverified); chains are short. A `seen` set guards against a cycle; ties resolve to the oldest edge.*

### 14.9 Update detection — `create_document_with_auto_supersede` and `find_update_shortlist`

**`create_document_with_auto_supersede` — the ingest write, with the exact (auto) tier folded in**

The server's document create. One atomic `GRAPH.QUERY` resolves the ingestor, looks up the oldest
current document with the same normalized-text hash, creates the new document (§14.1's shape plus
`textNormalizedHash`), and — only when the ingestor resolved and a candidate exists — supersedes the
candidate with a `system`-confirmed `SUPERSEDES` edge. Folding lookup and create into one query
closes the race where two concurrent ingests of the same content each miss the other.

The query starts with one of two ingestor-resolution prefixes — two literal texts, because a single
text with a possibly-`NULL` `$producedBy` plans `Node By Label Scan` + `Filter` when the parameter
is `NULL`:

```cypher
// default: the caller's actor ($ingestedBy), User or Agent
OPTIONAL MATCH (u:User  {userId:  $ingestedBy})
OPTIONAL MATCH (a:Agent {agentId: $ingestedBy})
WITH coalesce(u, a) AS ingestor, (coalesce(u, a) IS NOT NULL) AS ok,
     (u IS NOT NULL) AS ingestorIsUser
```
```cypher
// produced_by given: only that Agent; $ingestedBy is not sent
OPTIONAL MATCH (pa:Agent {agentId: $producedBy})
WITH pa AS ingestor, (pa IS NOT NULL) AS ok, false AS ingestorIsUser
```

followed by the shared body:

```cypher
OPTIONAL MATCH (candidate:Document {
  textNormalizedHash: $textNormalizedHash, currentVersion: true
})
WITH ingestor, ok, ingestorIsUser, candidate
ORDER BY candidate.createdAt ASC
LIMIT 1
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (d:Document {
    documentId: $documentId, title: $title, text: $text,
    sourceFormat: $sourceFormat,
    sourceKind: CASE WHEN ingestorIsUser THEN 'document' ELSE 'agent' END,
    status: 'processing', pendingJobs: 0, createdAt: $createdAt,
    currentVersion: true, textNormalizedHash: $textNormalizedHash
  })
  CREATE (d)-[:INGESTED_BY]->(ingestor)
  FOREACH (ch IN $chunks |
    CREATE (d)-[:HAS_CHUNK]->(:Chunk {
      chunkId: ch.chunkId, text: ch.text, seq: ch.seq, documentId: $documentId,
      documentCurrent: true
    })
  )
)
WITH ok, candidate
OPTIONAL MATCH (d:Document {documentId: $documentId})
WITH ok, d, candidate, (ok AND candidate IS NOT NULL) AS doSupersede
FOREACH (_ IN CASE WHEN doSupersede THEN [1] ELSE [] END |
  SET candidate.currentVersion = false, candidate.supersededAt = $createdAt,
      candidate.supersededBy = 'system'
)
WITH ok, d, candidate, doSupersede
OPTIONAL MATCH (candidate)-[:HAS_CHUNK]->(c:Chunk)
WITH ok, d, candidate, doSupersede, collect(c) AS candidateChunks
FOREACH (ch IN CASE WHEN doSupersede THEN candidateChunks ELSE [] END |
  SET ch.documentCurrent = false
)
WITH ok, d, candidate, doSupersede
FOREACH (_ IN CASE WHEN doSupersede THEN [1] ELSE [] END |
  CREATE (d)-[:SUPERSEDES {
    matchId: $matchId, status: 'confirmed', confidence: 1.0,
    technique: 'exact_normalized_text_hash', createdAt: $createdAt,
    decidedAt: $createdAt, decidedBy: 'system',
    resuggestCount: 0, lastResuggestedAt: null
  }]->(candidate)
)
RETURN ok AS ingestorFound, d.documentId AS documentId,
       doSupersede AS autoSuperseded,
       CASE WHEN doSupersede THEN candidate.documentId ELSE null END
         AS supersededDocumentId,
       CASE WHEN doSupersede THEN $matchId ELSE null END AS matchId
```
*The candidate binds before the new `Document` is created, so a document is never its own
candidate. A node created inside `FOREACH` is out of scope after it, hence the indexed re-`MATCH` of
`d` by `documentId`. Every conditional step is guarded on `doSupersede`, so an unresolvable actor
writes nothing and leaves every existing document untouched (`ingestorFound=false` → the service
raises `UnknownActorError`). A resolved `produced_by` always yields `sourceKind: 'agent'`. The
repository adds `chunkCount = len(chunks)` app-side.*

**`find_update_shortlist` — the suggested tier's candidate read**

Run by `background._safe_detect_update`, scheduled after an ingest the exact tier did not
auto-supersede, and only when an embed worker or extraction pipeline is wired (`SERVER.md` §1.12). It
computes MinHash LSH bands of the new document's normalized text
(`update_detection.lsh_bands`, 8 bands) and a title, then unions two read-only lookups, both limited
to current versions and de-duplicated by `documentId` in Python:

```cypher
// band signal — one OR term per band, generated for i in 0..len(bands)-1
MATCH (d:Document {currentVersion: true})
WHERE d.lshBand0 = $band0 OR d.lshBand1 = $band1 OR … OR d.lshBand7 = $band7
RETURN d.documentId AS documentId, d.title AS title, d.text AS text
LIMIT $limit
```
```cypher
// title signal — skipped when $fuzzyQuery would be empty
CALL db.idx.fulltext.queryNodes('Document', $fuzzyQuery)
YIELD node AS d
WHERE d.currentVersion = true
RETURN d.documentId AS documentId, d.title AS title, d.text AS text
LIMIT $limit
```
*`$limit` defaults to 5 per lookup. `$fuzzyQuery` is one `%token%` term per whitespace token of the
title, each token stripped of every non-word character (`_escape_fuzzy_token`); a token left empty
or underscore-only is dropped (a lone `%_%` is a syntax error), and no call is made when none is
left. Title-tokenization questions are open as **K-064**. The band `OR` across the indexed
`Document.lshBand0..7` properties is live-verified to plan as a single `Node By Index Scan`.*

**No server path writes `Document.lshBand0..7`.** The bands are computed only for this read, so the
band lookup matches no document and the title lookup is the only signal that produces candidates.

The detector then scores each candidate (excluding the document itself) by exact shingled Jaccard
in Python and, for the best one at or above the noise floor of `0.1`, records a `pending`
suggestion with `create_or_reopen_supersede_suggestion` (§14.8).

---

## 15. Product catalog — structured lookup

A fixed-shape, seed-script-write-only electronics catalog in the global `reference` graph —
`Product` is looked up by property key, never traversed from a workspace node, so it needs no
per-workspace materialization. It backs the `salesperson` agent's tools: an exact-name fact lookup
(§15.1), a category/price-range filter (§15.2), and the constrained generated read (§15.3). There
is no full-text index on `Product`; wording tolerance is the calling model's argument extraction,
not fuzzy matching in the database.

**Schema:** `DESIGN.md` §5.1; **indexes:** `DESIGN.md` §7.2. The `$nameNormalized`/
`$categoryNormalized` parameters are computed with `extraction.normalize_name`, the helper behind
`Entity.nameNormalized` (§14.5), so the lookups are a real `=` comparison, never a runtime
`toLower()`. The raw `name`/`category` are returned, never filtered on.

### 15.1 `lookup_product` — exact-name fact lookup

```cypher
// $nameNormalized = case-folded + whitespace-collapsed product name, computed by the
// caller (services.lookup_product) via extraction.normalize_name.
MATCH (p:Product {nameNormalized: $nameNormalized})
RETURN p.productId AS productId, p.name AS name,
       p.category AS category, p.price AS price
LIMIT 1
```
Anchors on the `Product.nameNormalized` range index — a single `Node By Index Scan`. Zero rows means
"no such product"; the service returns `None`, never a fabricated row, and `LookupProductFactTool`
answers `{"found": false, "hint": …, "catalog": [{productId, name}, …]}` — every product (up to
`CATALOG_INDEX_LIMIT`, 50), sorted by name, ID and name only, no price (`Services.catalog_index`,
built from §15.2's query with no filters) — so a model that looked up a translated name can pick the
real product's `productId`. `LIMIT 1` guards against two same-named products (`nameNormalized`
carries no uniqueness constraint).

`productId` is the key a `CartItem` is keyed on (§16). `LookupProductFactTool` returns the row
verbatim, so the id reaches the model's context; the cart tools take that `productId` and the
service resolves it with `lookup_products_by_id` (§16.9), not by name through this query.

### 15.2 `filter_products` — category/price-range filter

```cypher
// $categoryNormalized, $minPrice, $maxPrice are each optional (NULL when the caller
// omits that filter). $limit bounds the result size (services.py's DEFAULT_FILTER_LIMIT).
// $categoryNormalized = extraction.normalize_name(category), computed by
// Repository.filter_products. A category that normalizes to "" (empty/whitespace-only)
// is sent as NULL — coalesce reads only NULL as "no filter"; "" would match nothing.
MATCH (p:Product)
WHERE p.categoryNormalized = coalesce($categoryNormalized, p.categoryNormalized)
  AND p.price >= coalesce($minPrice, -1.0)
  AND p.price <= coalesce($maxPrice, 1000000000.0)
RETURN p.productId AS productId, p.name AS name,
       p.category AS category, p.price AS price
ORDER BY p.price ASC
LIMIT $limit
```

**Why `coalesce`, not `$param IS NULL OR prop = $param`.** The null-guard form degrades to a full
`Node By Label Scan` on every call, even with a selective bound value
(`claude/graph-dba/falkordb-quirks.md`); with three independent optional predicates, branching into
separate query strings (the §14.6 `list_matches` fix) would need up to eight. Instead:
- `category`'s optional equality is a self-referential `coalesce`, tautologically true when
  `$categoryNormalized` is `NULL` (every `Product` has a non-null `categoryNormalized`). It does not
  anchor an index scan on its own and does not need to, because
- `minPrice`/`maxPrice` use **literal numeric sentinels** (effectively ±infinity for this price
  domain), and a range comparison against them anchors the scan: every combination of
  supplied/omitted filters, including the list-everything call and a differently-cased category,
  profiles as `Node By Index Scan | (p:Product)` on `Product.price`, with the category comparison
  as a residual `Filter`.

Results are identical to the null-guard form across every filter combination. An all-omitted call
lists the whole catalog, bounded by `$limit`, ordered by price ascending.

**Blank means omitted.** `Repository.filter_products` maps a `category` that normalizes to `""` to
`NULL`; `FilterProductsTool` (`_coerce_price_bound`) maps a blank/whitespace-only `minPrice`/
`maxPrice` string to `None` and a numeric string (`"50"`) to that `float`; any other string is passed
through and would match nothing. Zero rows means "nothing matches", and the **tool** builds the
result: when a `category` was given, it first checks whether that category exists by re-running
this query with that `category` only and `$limit = 1` (`Services.has_category`). An unknown category
returns `{"items": [], "finding": "no catalog category has that name", "hint": …, "catalog": […]}`
(the same price-free list as a §15.1 miss); every other empty result is
`{"items": [], "finding": "no matching products found"}`. `Services.catalog_categories` is the
distinct stored `category` values of this query run unfiltered (no extra Cypher). `FilterProductsTool`
returns the rows verbatim, including `productId`; the storefront catalog route keys on it too.

### 15.3 `query_graph_data` — the compiled read-only query (`querygen.compile`)

The agent tool `query_graph_data` never sends model-written Cypher. The model's structured
completion is validated into a `QueryRequest` and compiled by `querygen.compile` into one of the
shapes below; `Repository.run_readonly_query` executes it with `GRAPH.RO_QUERY` and returns rows
keyed by column name. The tool's result contract is in `SERVER.md`.

**Datasets** (`querygen.DATASET_REGISTRY`, an allowlist, not schema introspection):

| Dataset | Graph | Label → queryable properties |
|---|---|---|
| `catalog` | `reference` | `Product`: `name`, `nameNormalized`, `category`, `price` |
| `knowledge_base` | `ws:{id}` of the caller | `Entity`: `entityId`, `name`, `nameNormalized`, `type` · `Document`: `documentId`, `title`, `sourceFormat` · `Chunk`: `chunkId`, `text`, `seq`, `documentId` |

**Compiled shapes** (`$p0…$p3` are the filter values, `$limit` the row cap):

```cypher
// default
MATCH (v:Label) [WHERE v.prop1 OP $p0 AND … AND v.prop4 OP $p3]
RETURN [DISTINCT] <returns>
[ORDER BY v.orderProp ASC|DESC]
LIMIT $limit
```
```cypher
// no aggregate in <returns>, and order_by is not one of <returns>
MATCH (v:Label) [WHERE …]
WITH DISTINCT <r1> AS c0, …, <rN> AS c(N-1), v.orderProp AS cN
ORDER BY cN ASC|DESC
LIMIT $limit
RETURN c0 AS `<r1>`, …, c(N-1) AS `<rN>`
```
*`DISTINCT` applies only when no `returns` entry is an aggregate, so a `count(...)` counts raw rows.
The second shape exists because `RETURN DISTINCT … ORDER BY <column not returned>` de-duplicates
before sorting and keeps an arbitrary sort key per group — a wrong superlative answer on this build
(`claude/graph-dba/falkordb-quirks.md`); de-duplicating on the full tuple, then re-aliasing, keeps
the "column key = original expression text" contract of every shape.*

**Guards** (Layer 1, `querygen`; each failure raises `ValueError`/`ValidationError`, and the tool
abstains):
- exactly one `MATCH (var:Label)`, no relationship traversal; `var` matches `[a-z][a-z0-9]{0,7}`;
  `Label` and every property must be in the dataset's allowlist (exact match, no coercion);
- at most 4 filters, each `var.property OP $pN` with `OP ∈ {=, <>, <, <=, >, >=}`; values are
  always bound parameters, never spliced;
- `returns`: 1–6 entries, each `var.property` or `count|avg|min|max(var[.property])`, no
  duplicates (the engine rejects two same-named columns at execution); `order_by` is a bare
  `var.property`; `limit` 1–50, default 20;
- a string value compared with `=`/`<>` on a `*Normalized` property is normalized with
  `extraction.normalize_name`; a string value on an `int`/`float` property is parsed as that type or
  rejected. A `bool` value on a non-`bool` property is neither coerced nor rejected and matches
  nothing.
- the compiler's templates contain no `CREATE`, `MERGE`, `SET`, `DELETE`, `REMOVE`, `DROP`,
  `FOREACH` or `CALL`.

**Before compile** the tool normalizes the completion: repeated JSON keys merge
(`merge_duplicate_keys` — identical values collapse, list values union, conflicting scalars
reject); an absent `returns` defaults to `["<var>.name"]` unless the request is a single-entity
lookup (an `=` filter on an identity property); and `ensure_row_identity` prepends the label's
first identity property (`Product.name`, `Entity.name`, `Document.title`, `Chunk.chunkId`) to a
per-row attribute projection that has none — not to aggregates, facet enumerations
(`Product.category`, `Entity.type`, `Document.sourceFormat`) or single-entity lookups.

**Layer 2:** `run_readonly_query` always uses `GRAPH.RO_QUERY`, which the engine refuses to run if
the text is a write, with `timeout = DEFAULT_QUERY_TIMEOUT_MS` (2500 ms) — a safety margin, since
this build enforces read `TIMEOUT` at batch granularity. Two shape classes that compile but fail at
execution are open as **K-090**.

---

## 16. Cart / Order — durable cart, deterministic totals, order lifecycle

Every query below is live-verified on the pinned build. Workspace-scoped (`ws:{workspaceId}`), unlike §15's global catalog — a cart/order
belongs to one customer inside one workspace, never shared across workspaces.

**Schema:** `DESIGN.md` §5.1. `customerId = User.userId` — the workspace-local `Customer` anchor is a
distinct node from `User` (keeps `ensure_user`'s create-once contract untouched; `Customer` is
freely mutable) but reuses the same member-id namespace. `Cart`/`CartItem` are keyed on natural
keys (`customerId`, and the composite `(customerId, productId)`) — no synthetic `cartId`/
`cartItemId`, mirroring the `WorkflowDefSnapshot` natural-key precedent (DESIGN §7.1). **`Cart`/
`CartItem` carry no price** — the cart's displayed price is always the catalog's
*current* price (`reference`), never a stored, staleable snapshot; §16.9 below is the two-graph
read this forces. **`Order`/`OrderLine` carry a full frozen snapshot** (`name`, `unitPrice`,
`quantity`, `lineTotal`): `place_order` computes and persists it once,
atomically, and nothing ever re-derives it from a live catalog read again. `Order.total` is
deliberately **not stored** — computed as `sum(OrderLine.lineTotal)` on read (§16.8): per-order
line count is tiny, so this avoids a write-time-only derived field that could drift from its own
source lines. `OrderLine` is a pure value-object always reached via `HAS_LINE`.

**Indexes:** `DESIGN.md` §7.1. `Order.status` is unindexed, so the lifecycle CAS queries (§16.10)
profile with a residual `Filter` above `Node By Index Scan | (o:Order)`, functionally identical (zero rows on a mismatch).

### 16.1 `ensure_customer` — get-or-create the `Customer` anchor (idempotent)

```cypher
// $customerId, $now
MERGE (c:Customer {customerId: $customerId})
ON CREATE SET c.createdAt = $now
RETURN c.customerId AS customerId, c.createdAt AS createdAt
```
`ON CREATE` only — a re-call leaves `createdAt` at the first call's value. Backed by the
`Customer.customerId` uniqueness constraint (every `MERGE` is constraint-backed).

### 16.2 `ensure_cart` — get-or-create the `Cart` anchor (idempotent)

```cypher
// $customerId, $now — call after ensure_customer
MATCH (cust:Customer {customerId: $customerId})
MERGE (cust)-[:HAS_CART]->(cart:Cart {customerId: $customerId})
ON CREATE SET cart.createdAt = $now, cart.updatedAt = $now
RETURN cart.customerId AS customerId, cart.createdAt AS createdAt
```
The `MERGE` spans the whole `(Customer)-[:HAS_CART]->(Cart)` pattern — safe here because `Cart`'s matched property (`customerId`) is fully specified and
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
**"Add" means increment, not replace** — two calls with the same `productId` sum their `qty`.
Zero rows means no `Cart` exists yet for `customerId` — `services.add_cart_item` runs
`ensure_customer` + `ensure_cart` first.

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
Used by the clear-cart tool; `place_order` (§16.7) clears the cart itself, inside its own
query. A `MATCH` against
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
computed via `pricing.compute_line_total` — no arithmetic happens in Cypher. Keyed on
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
`collect()`, not `[]` — `Repository.get_order` filters it out, so `lines` is `[]` there.

### 16.9 The two-graph computation — why a live cart/order total can't be one query

`CartItem`/`Order` live in `ws:{workspaceId}`; the catalog (§15) lives in `reference`. There is no
single Cypher query that can join a cart line to its current catalog price — not a missing edge,
but a structural fact: **`GRAPH.QUERY` operates on one named graph per call**, full stop (DESIGN §2
constraint #3, "relationships cannot cross graphs"). The mechanism is necessarily: (1) **read**
`ws:{workspaceId}` — §16.5's `read_cart` → `[{productId, quantity}, ...]`; (2) **read** `reference`
— `Repository.lookup_products_by_id`:

```cypher
// $productIds — an empty list short-circuits app-side with no query
UNWIND $productIds AS pid
MATCH (p:Product {productId: pid})
RETURN p.productId AS productId, p.name AS name, p.price AS price
```
an unknown `productId` simply yields no row; then
(3) **compute** (app/tool code, not Cypher, not an LLM call) — `pricing.compute_line_total`, plain
Python. This is exactly the shape §16.7's `place_order` already uses: the app performs steps 1–3
*before* calling the write, then passes the finished `$lines` in. Whatever step type or tool needs
"compute the total" cannot be a single graph traversal here — it is at minimum two
`GRAPH.RO_QUERY` calls plus app-side arithmetic, a property of the topology, not an implementation
shortcut.

### 16.10 Order lifecycle — guarded CAS, one transition per call (`fulfill_order`, `deliver_order`, `cancel_order`)

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
// cancel: $orderId, $now — only reachable from 'placed' ("cannot cancel once fulfilled")
MATCH (o:Order {orderId: $orderId})
WHERE o.status = 'placed'
SET o.status = 'cancelled', o.updatedAt = $now
RETURN o.orderId AS orderId, o.status AS status
```
Live-verified: `placed→fulfilled` succeeds; a `cancel` attempt on a `fulfilled` order returns zero
rows and the status stays unchanged; `fulfilled→delivered` succeeds. These three
queries drive the `order-fulfillment@v1` process def's lifecycle (`falkorchat.proof_defs.
ORDER_FULFILLMENT_DEF`) via `services.advance_order` — called from whichever caller accepts the
operator's fulfillment input **alongside** `submit_workflow_input`, the same "two-step, accepted"
pairing §12.6's `link_step_emission` uses (the def's own `human`/`decision` steps have no side
effect on `Order` — DESIGN §6.1 — so advancing the run and advancing the order are always two
separate graph writes from one request, never one query doing both).

---

## 17. Durable customer profile

Both queries below are live-verified on the pinned build. Workspace-scoped (`ws:{workspaceId}`), same anchor §16 uses.

**Schema:** `DESIGN.md` §5.1 — the profile is `name`, `deliveryAddress` and `profileUpdatedAt` on the
existing `Customer` node, no new label. `name`/`deliveryAddress` are independently nullable (a
customer may give one before the other). `Customer.customerId`'s uniqueness constraint
(`DESIGN.md` §7.1) backs the `MERGE` below. **This is not a second identity mechanism** — the profile lives on the same `Customer` node
§16's cart/order queries write to.

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
(§16.1): a later-supplied name/address updates the stored profile. **Also deliberately not** an
unconditional `SET` (the §13.1 `write_model_overrides` shape): `SaveProfileTool`'s arguments are
optional, so a partial update (only a new address) passes `$name = NULL` meaning "not provided,"
and an unconditional `SET` would null the stored name. Hence `coalesce($field, c.field)` per data
field, so a
`NULL` argument means "leave this field as it is," never "clear it." `profileUpdatedAt` alone stays
an unconditional `SET` — every call that reaches this query touched the profile, partial or full.
**No value can be *cleared* through this query**, only updated; a caller must never pass `''` to
mean "no value" (an empty string is a value, not "not provided").

Live-verified in both partial-update directions: full write (`name='Alice', deliveryAddress='123 Main St'`) → partial
call with `$name=NULL, $deliveryAddress='456 New Ave'` → `name='Alice'` (**preserved**),
`deliveryAddress='456 New Ave'` (updated) → partial call with `$name='Bob', $deliveryAddress=NULL`
→ `name='Bob'` (updated), `deliveryAddress='456 New Ave'` (**preserved**). `count(Customer) = 1`
throughout both partial calls — no duplicate node from either `MERGE`.

### 17.2 `get_profile` — the one code path for "unset"

```cypher
// $customerId
MATCH (c:Customer {customerId: $customerId})
RETURN c.name AS name, c.deliveryAddress AS deliveryAddress, c.profileUpdatedAt AS profileUpdatedAt
```
Zero rows (`None`) ⇒ no `Customer` node at all yet for this id (nobody has ever interacted with this
workspace as this customer — cart, order, or profile). A row with `name`/`deliveryAddress` both
`NULL` ⇒ the customer exists (e.g. from cart activity) but has never given a name/address. Both
cases mean "ask them" to the calling agent; the distinction is kept (never collapsed to one
shape).

---

## 18. Storefront participants & resets

Every query below is live-verified on the pinned build, and the repository holds each as a
multi-line constant identical to the block shown (`Repository._ENSURE_PARTICIPANT_CYPHER`,
`_RESET_PARTICIPANT_CYPHER`, `_RESET_ALL_PARTICIPANTS_CYPHER`, `_CURRENT_ORDER_CYPHER`,
`_ORDER_OWNERSHIP_CYPHER`). Workspace-scoped (`ws:{workspaceId}`); `reference` is never
referenced by any query here, which is structural — `GRAPH.QUERY` operates on one named graph per
call — not a convention.

**Cypher comments in this section are `//`.** `--` is **not** a comment on this build: it fails to
parse (`Invalid input 'G': expected '>' or '('`).

**Schema — one new nullable, unindexed property on the existing `Channel` node, no new label:**
```
(:Channel {channelId, name, createdAt,
            participantId})    // NULLable; written ONLY by §18.1 ensure_participant
```
No index or constraint on it: a `UNIQUE` constraint is declined on scope (§18.4 makes a duplicate
marker fail loudly anyway), and no index is needed either — the property is only ever a `Filter` on a `ch` already bound by a
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

**The governing rule is a *scoping* rule, not a label list:**

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
   unmounted** (`create_app(dev_surface=False)`, `SERVER.md`). Runs are reached only
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
`channelId`/`threadId`/`language` so a replay is a read-through (restart survival).

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
restart is invisible to participants.

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
server-resolved scope denorm the storefront reads from, and both are
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
stays valid ("reset mine" keeps the identity); their `Customer`, `Cart`,
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

**The delete is thread-scoped, not author-scoped** (locked). The
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

**Profile after the reset:** deleting the `Customer` also deletes the profile name the join wrote,
while `User.displayName` survives; the storefront re-saves it with `services.save_profile(ctx,
name=<User.displayName>)` (§17.1) right after the reset.

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
**The `ReadCursor` `collect` is collapsed with its own `WITH` *before* the `users` `UNWIND`** —
left un-collapsed, the 50-row `UNWIND` multiplies the scan to ~5 000 rows and `reset_all` costs
~690 ms instead of ~240 ms. That is the "collapse each block before the next `UNWIND` expands"
rule, and it bites at scale only.

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
| **`WorkspaceConfig`** | **keeps** | **keeps** | never matched by either query. Taking it would silently drop the workspace's model overrides (§13) — the single most expensive mistake available here. Positively asserted alive after both resets |
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
