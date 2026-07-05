# K-007 — M2 groundwork: verified query changes (graph-dba deliverable)

**Date:** 2026-07-04 · **Author:** graph-dba
**Verified against:** `falkordb/falkordb:edge` (module `999999`), Redis 8.2.2, `falkordb-py` 1.6.1,
scratch graph `ws:k007scratch` bootstrapped at `EMBEDDING_DIM=1024` (deleted after verification).

Every query in this document was **executed against the live instance**; evidence excerpts are
pasted verbatim. This document proposes changes to `QUERIES.md` §4/§5/§9 and supporting service
behavior — it changes no canonical doc itself. Fixing `DESIGN.md`/`QUERIES.md` and the server
code is the architect/coder step that follows.

---

## Item 1 — Agent authorship (defect: agents cannot post)

### Defect, live-confirmed twice

1. The current §4 author anchor `MATCH (author {userId: $authorId})` is **label-less**: it
   profiles as `All Node Scan` (whole graph), and an `Agent` (which carries `agentId`, not
   `userId`) matches nothing — the whole write silently no-ops (zero rows, transport success).
2. Repro during item-3 setup: a §4 subsequent-write with `authorId='a1'` (Agent) wrote nothing —
   message `L3` absent, TAIL not moved:

   ```
   legacy rows: [['L1', None], ['L2', None]]        # L3 never written
   ```

### Fix (folded into the v2 write paths below)

Author resolution uses the locked label-specific member-resolution pattern — two indexed
`OPTIONAL MATCH`es + `coalesce`:

```cypher
OPTIONAL MATCH (ua:User  {userId:  $authorId})
OPTIONAL MATCH (aa:Agent {agentId: $authorId})
WITH …, coalesce(ua, aa) AS author
```

`GRAPH.PROFILE` (full excerpt under item 2): both legs are `Node By Index Scan` — no
`All Node Scan` anywhere in either v2 path.

### Role derivation lookup (service-side, decision confirmed: `user` / `assistant`)

`services.post_message` derives `role` from the author's kind instead of trusting the caller.
Verified lookup (`labels(…)[0]` subscripting works on this build):

```cypher
UNWIND $ids AS id
OPTIONAL MATCH (u:User  {userId:  id})
OPTIONAL MATCH (a:Agent {agentId: id})
RETURN id, CASE WHEN coalesce(u, a) IS NULL THEN null
                ELSE labels(coalesce(u, a))[0] END AS kind
```

```
[['u1', 'User'], ['a1', 'Agent'], ['ghost', None]]
```

Mapping: `User → 'user'`, `Agent → 'assistant'`, `None → reject before writing`.
(`DESIGN.md` §5.1 says `human|assistant|system`; code uses `user`. Confirmed direction: keep
`user` as-built, add `assistant`. Doc correction = architect.)

---

## Item 2 — Retry idempotency + first-post race (write paths v2)

### Defect A, reproduced: retrying a §4 subsequent-write corrupts the thread

The §4 `MERGE (m:Message {msgId:…})` matches the existing node on replay, then the unconditional
`CREATE`/`DELETE` clauses run again. Replay of the exact same write (client-timeout → retry):

```
replay of m2 returned rows: [['m2']]     (transport reported success)
NEXT self-loops: [['m2', 1]]             # (m2)-[:NEXT]->(m2)
NEXT edges total: 2                      # was 1
POSTED_BY from m2: 2                     # duplicated
```

This falsifies the DESIGN §9 table claim "idempotent via unique constraint" (doc fix =
architect; confirmed as the defect being addressed here).

### Defect B, reproduced: two first-posts → two HEADs

The HEAD check and the first-write are separate queries, so two racing clients both see "no
HEAD" and both run the first-path:

```
t2 HEAD edges: 2   TAIL edges: 2
thread-read from t2 now returns: [[['mA', 'mB']]]   # two independent chains
```

### v2 design

Two **separate** write paths are kept (locked decision). Each becomes self-guarding inside its
single `GRAPH.QUERY` via the `FOREACH (… IN CASE WHEN ok THEN [1] ELSE [] END | …)` idiom — a
*guard on each path*, not a conditional merge of the two paths (compliance confirmed by user).
`MERGE` on the message is replaced by a guarded `CREATE`; the `Message.msgId` uniqueness
constraint remains the concurrency backstop (rollback verified below). Both paths **always
return a status row**, so zero rows now unambiguously means "thread anchor missing".

#### §4 v2 — post the first message in a thread

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

#### §4 v2 — post a subsequent message

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

Notes that must survive into QUERIES.md:

- The `UNWIND (CASE WHEN $mentions = [] THEN [null] …)` guard is **still mandatory** — a bare
  empty `UNWIND` collapses the row stream *before* the FOREACH and the whole write silently
  no-ops (the known live-verified fact, now load-bearing for the write itself).
- `DELETE` inside `FOREACH` (the TAIL relink) and nested `FOREACH` (mentions) are live-verified
  on this build.
- The reply add-on (`MATCH (quoted:Message {msgId:$quotedMsgId}) CREATE (m)-[:REPLY_TO]->(quoted)`)
  must move **inside the guarded FOREACH** when folded in (rely on the same live-verified
  "reference a FOREACH-created node in later clauses of the same body" fact); it was not
  separately re-verified in this pass — flag for the coder to test.

#### Status-row contract (service dispatch)

| Result | Meaning | Service action |
|---|---|---|
| zero rows | thread missing (first) / no TAIL yet (subsequent) | first-path: 404; subsequent: retry as first-path |
| `written=true` | committed | success |
| `dupMsg=true` | msgId already exists | **idempotent success** (retry replay) |
| `hadHead=true` (first path) | lost the first-post race | re-dispatch as subsequent |
| `authorFound=false` | unknown member | 4xx, nothing written |

### Verification evidence

Happy path (User first, **Agent subsequent** — item 1 fixed; mentions dedup/unknown-skip intact):

```
first  status row: [[True, False, False, True]]
subseq status row: [[True, False, False, True]]     (author = Agent a1)
chain: [['n1', 'user', 't1'], ['n2', 'assistant', 't1']]
mentions: [['n1', 'a1'], ['n2', 'u1']]              ('u1','u1','nope' -> one edge to u1
POSTED_BY n2 -> : [[['Agent'], 'a1']]
```

Defect A closed — exact replay is a structural no-op:

```
replay status row: [[False, False, True, True]]     # written=false, dupMsg=true
graph (nodes,edges) before=(7, 9) after=(7, 9) unchanged=True
NEXT self-loops: 0
replay of first-path status row: [[False, True, True, True]]
```

Defect B closed — first-post on a headed thread refuses; unknown author refuses:

```
late first-post:  [[False, True, False, True]]   nX exists? 0   HEAD,TAIL = (1, 1)
unknown author:   [[False, False, False, False]] nY exists? 0
```

**Concurrency hammer** — 16 Python threads (half User, half Agent authors, distinct msgIds) race
first-post on a fresh thread, each using the dispatch contract above:

```
threads=16 ok=16 errors=[]
paths taken: first=1 subseq=15
t3 HEAD=1 TAIL=1 messages=16 chain-walk=16 NEXT=15 bad-POSTED_BY-count=0
HAMMER PASS: exactly one HEAD/TAIL, contiguous chain of 16
```

**Constraint-violation rollback is all-or-nothing** (backstop for anything the guards miss). A
single query that `SET`s a property then `CREATE`s a duplicate `msgId`:

```
query raised: unique constraint violation on node of type Message
t1.updatedAt before=2000 after=2000 rolled_back=True
n1 node count (must stay 1): 1
```

### GRAPH.PROFILE — all anchors indexed, no All Node Scan

First path (excerpt; subsequent path is identical in anchor structure, with
`Conditional Traverse | (t)-[tailRel:TAIL]->(prev:Message)` above the Thread scan):

```
Optional Conditional Traverse | (t)->(h)
    Node By Index Scan | (t:Thread)
Optional / Node By Index Scan | (dup:Message)
Optional / Node By Index Scan | (ua:User)
Optional / Node By Index Scan | (aa:Agent)
Optional / Node By Index Scan | (mu:User)      # mention resolution
Optional / Node By Index Scan | (ma:Agent)
Foreach
    Update / Create ×4 (m, HEAD, TAIL, POSTED_BY, SET t.updatedAt)
    Foreach / Create (MENTIONS_MEMBER)
```

Cost note: v2 adds three point index lookups (`h`, `dup`, second author leg) per post — all
O(log n) `Node By Index Scan`s; negligible vs. the old `All Node Scan` it removes.

---

## Item 3 — `threadId` denormalized on Message

- Both v2 write paths above already write `threadId: $threadId` inline.
- **No index on `Message.threadId`** (deliberate): it is navigation/display metadata for §9.2
  and §5 results; §9.1 remains the canonical thread walk. Skipping the index saves per-workspace
  RAM and write cost (rule 6 costing under item 6).

### Backfill (one-off, verified idempotent)

Per-thread (batchable — run per threadId to bound query time):

```cypher
MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message)
MATCH (first)-[:NEXT*0..]->(m:Message)
WHERE m.threadId IS NULL
SET m.threadId = t.threadId
RETURN count(m) AS backfilled
```

Workspace-wide variant: drop the `{threadId: $threadId}` filter (`MATCH (t:Thread)-[:HEAD]->…`).
Evidence:

```
backfill t2 run 1: [[2]]
backfill t2 run 2 (idempotent): [[0]]
after: [['L1', 't2'], ['L2', 't2']]
workspace-wide run: [[0]]
```

Orphan caveat: the walk anchors on `HEAD`, so a message not reachable from a HEAD (e.g. residue
of the item-2 defects) is not backfilled — acceptable, since such messages are already invisible
to thread reads.

### §9.2 / §5 gain `m.threadId` in RETURN — verified + profiled

§9.2 with `m.threadId` added (rows show it populated across threads); the anchor is still the
`createdAt` index:

```
Node By Index Scan | (m:Message)        # createdAt range — not a label scan
```

§5 full-text with `m.threadId` added:

```
search 'legacy': [['L1', 't2', 'legacy 1', 1000, 4.0], ['L2', 't2', 'legacy 2', 2000, 4.0]]
```

---

## Item 4 — millisecond `createdAt` ties (page-boundary skip)

### Defect, reproduced

Four messages `k1..k4` at `createdAt` 1000, 2000, **2000**, 3000. Plain `>` cursor paging with
`LIMIT 2` (cursor advanced to newest returned `createdAt`, per the §9 contract):

```
page 1 (limit 2): [['k1', 1000], ['k2', 2000]]
page 2 (since=2000): [['k4', 3000]]
delivered: ['k1', 'k2', 'k4']   SKIPPED: ['k3']     # silent permanent loss
```

### Fix — deterministic order + composite keyset

Two layers (per approved plan):

**(a) Service:** monotonic server clock — `now = max(clock_ms(), last_issued + 1)` — makes ties
impossible at the single-process M2 source. (Service code, not a query; not testable here.)

**(b) Query correctness backstop** — `ORDER BY m.createdAt, m.msgId` + composite predicate.
§9.1 keyset form (§9.2 identical minus the thread-walk anchor, plus `POSTED_BY`/mention block
unchanged):

```cypher
MATCH (t:Thread {threadId: $threadId})-[:HEAD]->(first:Message)
MATCH (first)-[:NEXT*0..]->(m:Message)
WHERE m.createdAt > $since
   OR (m.createdAt = $since AND m.msgId > $sinceMsgId)
…
ORDER BY m.createdAt, m.msgId
LIMIT $limit
```

`$sinceMsgId` defaults to `''` (empty string sorts before every id) when only a timestamp is
known. Evidence — same fixture, no skip:

```
page 1: [['k1', 1000], ['k2', 2000]]
page 2 (since=2000, sinceMsgId='k2'): [['k3', 2000], ['k4', 3000]]
delivered: ['k1', 'k2', 'k3', 'k4']   skipped: []
```

### PROFILE — the composite predicate keeps the index anchor

Both formulations were profiled on the index-anchored §9.2 shape; **both** plan as a bare
`Node By Index Scan | (m:Message)` with no residual `Filter` op, and both return the correct
tied rows:

```
-- A: WHERE m.createdAt > $since OR (m.createdAt = $since AND m.msgId > $sinceMsgId)
Limit / Sort / Project / Node By Index Scan | (m:Message)
-- B: WHERE m.createdAt >= $since AND (m.createdAt > $since OR m.msgId > $sinceMsgId)
Limit / Sort / Project / Node By Index Scan | (m:Message)
```

**Recommendation: formulation A** (mirrors the ORDER BY semantics 1:1). B is the documented
fallback if a future engine version stops folding the OR into the index scan — re-profile on
engine upgrades (edge build, moving target).

### §9.3 v2 — composite monotonic cursor advance (verified)

`ReadCursor` gains `lastReadMsgId` (plain property — **no schema/bootstrap change**; the
`cursorId` index + constraint already cover the MERGE). The monotonic guard becomes composite,
computed **once** in a `WITH` so both `SET`s see the pre-write state:

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

All five scenarios verified live:

```
create           (2000,'k2'): [[2000, 'k2']]
tie, larger id   (2000,'k3'): [[2000, 'k3']]
tie, smaller id  (2000,'k2'): [[2000, 'k3']]     # stale replay refused
backward         (1500,'k9'): [[2000, 'k3']]     # never moves backward
forward          (3000,'k4'): [[3000, 'k4']]
```

Service continues to own `$now`/`$nowMsgId` = the newest **returned** `(createdAt, msgId)` pair.
Explicit `?since=` REST reads keep plain `>` timestamp semantics (document: an explicit `since`
may re-deliver or skip within that exact millisecond; cursor-driven reads never do).

Tie-break caveat to document: `msgId` order is lexical, so within one millisecond the delivery
order is id order, not arrival order. With the monotonic service clock (layer a) ties only occur
across *writers*, and any deterministic total order is acceptable there. If human-facing
ordering within ties ever matters, mint k-sortable ids (UUIDv7/ULID) — do not re-sort pages.

---

## Item 5 — TIMEOUT review (live-probed)

Config on this box: `TIMEOUT=1000`, `TIMEOUT_DEFAULT=0`, `TIMEOUT_MAX=0` (legacy single-knob
mode). Probes on the live instance:

| Probe | Result |
|---|---|
| read, ~420 ms internal, no clause | completes (under the 1000 ms default) |
| read, ~1161 ms internal, no clause | **completed** — enforcement is batch-granular, slightly-over queries can slip through |
| read, ≥1.4 s, no clause | `Query timed out` (default fires) |
| read + `TIMEOUT 10` | `Query timed out` |
| read + `TIMEOUT 15000` (> default) | completes (4.0 s) — override is **uncapped** while `TIMEOUT_MAX=0` |
| **write** (1319 ms, 3 M nodes) + `TIMEOUT 100` | **TIMEOUT ignored — write ran to completion** |
| `falkordb-py` `g.ro_query(q, timeout=10)` / `timeout=5000` | raises `Query timed out` / completes — client pass-through confirmed |

Anomaly noted once (not reproduced): immediately after a long override run, an identical query
returned `Query timed out` in ~0 ms; subsequent runs behaved normally. Treat as edge-build timer
bookkeeping noise; re-check on upgrades.

**Recommendation:**

1. Keep `TIMEOUT=1000` as the deployment default — right for chat CRUD, and now verified to fire.
2. GraphRAG/§6/§8 hybrid reads and long thread walks: pass a per-query override from the client
   (`g.ro_query(q, params=…, timeout=…)`, e.g. 5000–10000 ms) instead of raising the global
   default. Expose it as a service-layer constant, not per-call ad-hockery.
3. If ops later wants a hard ceiling on client overrides, switch the deployment to
   `TIMEOUT_DEFAULT`/`TIMEOUT_MAX` (>0) — but note module docs say the legacy `TIMEOUT` and the
   new pair are mutually exclusive; change deliberately, in one step.
4. **Writes cannot be killed by TIMEOUT on this build.** The only protection for the write path
   is bounding work per query — keep `UNWIND` ingestion batches modest (≤ a few hundred rows,
   see item 6) and never accept unbounded client-supplied lists (input caps already exist in the
   API; keep them).

---

## Item 6 — RAM costing at 1024 dims (empirical)

Method: fresh re-bootstrapped `ws:k007scratch` (1024-dim vector indexes), bulk-loaded **4096**
messages (realistic shape: `msgId/text/role/createdAt/threadId` + inline 1024-dim `vecf32`
embedding + `POSTED_BY` + full `NEXT` chain + HEAD/TAIL), measured Redis `used_memory` and
`GRAPH.MEMORY USAGE` before/after. (Plan said ~1k; raised to 4k because `GRAPH.MEMORY USAGE`
reports whole MB — 4k gives usable resolution.)

```
BASELINE used_memory=812,945,336 B    graph total=0 MB   indices=0 MB
loaded 4096 messages in 3.5s (1178 msg/s), batches of 256
chain walk count: 4096
kNN top-5 via vector index: works (cosine distances ~0.92)
AFTER    used_memory=863,683,304 B    graph total=16 MB  indices=0 MB
DELTA    used_memory = 50,737,968 B   ->  12,387 B/message
```

### Per-message line at 1024 dims (replaces the DESIGN §11 1536-dim figures)

| Component | Bytes/message |
|---|---|
| raw `vecf32` embedding (1024 × 4 B) | 4,096 |
| node + attrs (text ~50 chars, ids, role, `createdAt`, **`threadId`**) + edges (`NEXT`, `POSTED_BY`) | ~1,900 (within the 16 MB attrs accounting) |
| HNSW vector index + range-index entries + allocator overhead | ~6,400 |
| **Total observed** | **~12.4 KB** |

- Rule of thumb: **~12.5 KB/message at 1024 dims ≈ 1.25 GB per 100k-message workspace**
  (vs ~17–18 KB extrapolated at 1536 — the dim cut saves roughly a third).
- **`threadId` cost:** one short string property, ~50–60 B/message, no index — noise (<0.5%)
  against the 12.4 KB line. `ReadCursor.lastReadMsgId`: one string per (member, thread) cursor —
  negligible.
- **Measurement caveat:** `GRAPH.MEMORY USAGE` reported `indices_sz_mb: 0` while the HNSW index
  demonstrably held 4096 vectors (kNN worked) and the allocator grew ~34 MB beyond the accounted
  16 MB of attributes. On this edge build the command **under-reports vector-index memory** — size
  workspaces from `INFO memory` deltas, not `GRAPH.MEMORY USAGE`, until this is fixed upstream.
- Ingestion throughput datapoint: ~1,178 msg/s with 256-row `UNWIND` batches incl. embeddings,
  single client — bulk batches of 100–500 rows are comfortably inside the write path's safety
  envelope (cf. item 5, writes are unkillable — keep batches bounded).

---

## Fold-in design notes (server code — for the architect/coder)

1. **`db.connect()` binds `config.FALKORDB_*` at import time.** Default the connection params to
   `None` and resolve inside the call so tests/deploys can repoint without re-import.
2. **`create_channel`/`create_thread` use `MERGE` on freshly-minted uuids** — misleading: it can
   never match, so it is a `CREATE` wearing a MERGE costume. Make them `CREATE` (constraints
   still backstop). Consequence to document: create endpoints are non-idempotent — a retried
   create mints a new id.
3. `services.post_message` changes: author-kind lookup (item 1) → derive `role`; dispatch on the
   v2 status row (contract table in item 2); monotonic clock (item 4a); cursor advance passes
   `(now, nowMsgId)` (item 4).
4. Repository: v2 queries return a status row on *every* anchored execution — the "empty result
   ⇒ raise" rule now applies only to the thread anchor (see contract table).

## Locked-decision compliance

- **Two write paths, never a conditional MERGE** — preserved; the FOREACH+CASE construct guards
  each path independently (user-confirmed compliant).
- **Single-`GRAPH.QUERY` atomicity for HEAD/TAIL** — preserved; the guard moved the relink
  *inside* the same query, and constraint-violation rollback is verified all-or-nothing.
- **Every MERGE backed by a uniqueness constraint** — v2 write paths contain no MERGE (guarded
  CREATE + constraint backstop); §9.3's MERGE keeps its `cursorId` constraint.
- **Label-specific member resolution** — now used for authors too (closes the last
  `All Node Scan` in the write path).

## Open questions for the architect

1. **DESIGN §9 "idempotent via MERGE" claim** — falsified (item 2 evidence); rewrite around the
   status-row contract. **DESIGN §5.1 role values** — align to `user`/`assistant` as confirmed.
2. Should `dupMsg=true` verify payload equality (same text/author) before reporting idempotent
   success, or is msgId trust sufficient? (Current recommendation: trust msgId — it is
   server-mintable; if msgIds ever become client-supplied, add payload checksum.)
3. REST surface for explicit `?since=` reads: document plain-`>` semantics, or extend the API to
   accept the composite `(since, sinceMsgId)` pair the MCP cursor path uses?
4. `REPLY_TO` inside the guarded FOREACH was reasoned, not live-verified — coder must add a test
   when folding into `repository.py`.
5. TIMEOUT strategy sign-off: per-query client overrides (recommended) vs. switching the
   deployment to `TIMEOUT_DEFAULT`/`TIMEOUT_MAX`. Note the write-path exemption (item 5, probe
   table) in DESIGN's ops section.
6. Upstream: consider filing the `GRAPH.MEMORY USAGE` vector-index under-reporting (item 6) and
   the one-shot instant-timeout anomaly (item 5) against FalkorDB — both observed on the edge
   build (module `999999`).
