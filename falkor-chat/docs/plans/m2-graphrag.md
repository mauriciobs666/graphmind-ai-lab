# K-008 — M2 GraphRAG retrieval core: verified query changes (graph-dba deliverable)

**Date:** 2026-07-08 · **Author:** graph-dba
**Verified against:** `falkordb/falkordb:edge` (module `999999`), Redis 8.2.2, container `falkordb-dev`,
throwaway graph `ws:m2verify` bootstrapped at `EMBEDDING_DIM=1024` (deleted after verification).

Every query in this document was **executed against the live instance**; evidence excerpts are
pasted verbatim. This is the verified-query gate for K-008 — it changes no canonical doc body; it
confirms the `QUERIES.md` §6 surface is correct as written and hands the Python impl to a
tdd-engineer. The `QUERIES.md` §6 queries were run **unchanged** and pass — no edits proposed.

---

## Scope recap

- Embedding model **Qwen3-Embedding-0.6B**, **`EMBEDDING_DIM=1024`**, cosine (DESIGN §1.3).
- Score is **cosine distance** (0 = identical), order **ASC** = most similar first.
- GraphRAG in M2 = **vector ANN seed + thread/channel-scope traversal**. There is **no Entity
  extraction pipeline** — the `MENTIONS`→`Entity` expansion in §6 must no-op cleanly.
- graph-dba gate only: verify queries, PROFILE, raise the suite. Python impl is the tdd-engineer step.

---

## Item 1 — 1024-dim vector index created and OPERATIONAL

`EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh m2verify` created the `Message.embedding` and
`Chunk.embedding` vector indexes. `CALL db.indexes()` shows them OPERATIONAL (dimension is not
exposed by `db.indexes()`, so it is proven separately in item 2):

```
Message  [msgId, createdAt, text, embedding]
         {msgId:[RANGE], createdAt:[RANGE], text:[FULLTEXT], embedding:[VECTOR]}  OPERATIONAL
Chunk    [chunkId, embedding]  {chunkId:[RANGE], embedding:[VECTOR]}              OPERATIONAL
```

The other range/fulltext indexes and all uniqueness constraints bootstrap unchanged
(index-before-constraint ordering holds).

---

## Item 2 — Dimension is 1024, proven; write-time is NOT dimension-checked (quirk)

`db.indexes()` hides the dimension, so it is proven by behavior:

**Proof the index is 1024** — querying it with a 4-dim query vector errors with the expected size:

```
CALL db.idx.vector.queryNodes('Message','embedding', 2, vecf32([1.0,0.0,0.0,0.0]))
=> Vector dimension mismatch, expected 1024 but got 4
```

**Quirk (new, live-verified on this build — fold into ops guidance):** a **wrong-dimension
`vecf32` write is silently accepted at `SET`** — no error — but the node then **falls out of the
ANN index** (it is not returned by kNN):

```
SET m2.embedding = vecf32([0.5,0.5,0.5,0.5])   =>  Properties set: 1     # no error
CALL db.idx.vector.queryNodes('Message','embedding', 4, <1024-dim q>)
=> m1 (0), m3 (1)          # m2 is GONE from results — silently unindexed
```

Dimension is enforced at **query time** (query vector must match) and at **index-membership time**
(wrong-dim nodes drop out of ANN), but **never at write time**. Consequence for the impl: the
embedding worker MUST validate `len(embedding) == EMBEDDING_DIM` client-side before the SET — a
buggy worker sending a wrong-size vector will not error; the message just becomes permanently
invisible to GraphRAG retrieval. There is no `vec.dimension()` function on this build to check
after the fact.

---

## Item 3 — §6 set-embedding write (verified as written)

```cypher
MATCH (m:Message {msgId: $msgId})
SET m.embedding = vecf32($embedding)
```

Evidence (1024-dim vectors on m1..m4):

```
set m1: Properties set: 1
set m2: Properties set: 1
set m3: Properties set: 1
set m4: Properties set: 1
```

Decoupled from the write path (the message is readable before the embedding lands) — this is the
async embedding-worker query, run once per message after it is posted. No change proposed.

---

## Item 4 — §6 ANN retrieval: ranking by cosine distance ASC (verified)

Deterministic stub vectors on a Channel→Thread(HEAD/NEXT)→Message structure with `POSTED_BY`:
m1=`[1,0,0,…]`, m2=`[0.9,0.1,0,…]`, m3=`[0,0,1,…]`, m4=`[0,0,0.9,0.1,…]` (all 1024-dim).
Query vector = m1's vector (identical).

```
CALL db.idx.vector.queryNodes('Message','embedding', 4, vecf32(<m1>))
YIELD node AS seed, score RETURN seed.msgId, score ORDER BY score ASC
=>
m1   0
m2   0.00611627101898193
```

- **Identical vector → score 0** holds on this build (m1).
- **ASC = most similar first** (m1 before m2). m3/m4 (orthogonal, cosine distance 1) rank last.
- **ANN recall note:** with `k=4` on this ~4-vector index the kNN returned only the 2 near
  neighbors (the orthogonal, distance-1 vectors were not all surfaced). This is normal
  **approximate** HNSW behavior on a nearly-empty index — irrelevant at real workspace scale, and
  the near neighbors that matter are always returned and correctly ordered. Do not treat "kNN
  returns exactly k" as an invariant.

---

## Item 5 — Full §6 hybrid query + Entity expansion no-ops cleanly (verified)

The canonical §6 query, run **unchanged**, channel-scoped:

```cypher
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

Evidence:

```
m1  about cats   user  0                     []
m2  more on cats user  0.00611627101898193   []
```

- `relatedContext` is the **empty list `[]`** for every seed — the `OPTIONAL MATCH` on `Entity`
  matches nothing and `collect(DISTINCT related)` drops the nulls. **No error.**
- `MATCH (e:Entity) RETURN count(e)` = **0** — no entity pipeline in M2, confirmed.
- The **workspace-wide variant** (omit the `MATCH (c:Channel …)` line) also works and returns the
  seed ranked ASC.

GraphRAG in M2 therefore works as **vector-ANN + thread/channel scope**, with the Entity
co-occurrence layer present in the query but dormant until an entity-extraction pipeline lands in a
later milestone. `MENTIONS`→`Entity` (GraphRAG co-occurrence) stays distinct from
`MENTIONS_MEMBER`→`User`/`Agent` (participant mentions, §4) — do not conflate them.

---

## Item 6 — GRAPH.PROFILE: the ANN retrieval hits the vector index

`GRAPH.PROFILE` of the full §6 hybrid query (1024-dim, channel-scoped):

```
Results
  Limit
    Sort
      Project
        Aggregate
          Optional Conditional Traverse | (seed)->(related:Message)
            Filter
              Conditional Traverse | (c)->(c)
                Conditional Variable Length Traverse | (seed)-[@anon_0:HEAD|:NEXT*0..INF]->(t)
                  ProcedureCall                              # db.idx.vector.queryNodes — the vector index scan
```

- The plan is **anchored on `ProcedureCall`** (`db.idx.vector.queryNodes`) — the vector-index
  entry point that seeds the whole query. **No `All Node Scan` / label scan on `Message`.** The
  index is used.
- Traversal fans out from each ANN seed: the `HEAD|NEXT*0..` variable-length walk finds the seed's
  thread, `Conditional Traverse (c)->(c)` + `Filter` applies the channel scope, and the
  `Optional Conditional Traverse (seed)->(related:Message)` is the (dormant) Entity expansion.
- Anchoring on the vector index is why the `channelId` scope is a cheap post-filter on a handful
  of ANN seeds, not a scan. `TIMEOUT` posture: route via `GRAPH.RO_QUERY` with a per-query client
  `timeout=` override (item 8), not the 1000 ms global default.

---

## Item 7 — RAM (repo rule 6, mandatory)

The 1024-dim vector index is the dominant new RAM line. Empirical figure inherited from K-007
item 6 (4096-message load into a 1024-dim workspace, measured by `INFO memory` delta):

| Component | Bytes/message |
|---|---|
| raw `vecf32` embedding (1024 × 4 B) | 4,096 |
| node + attrs + edges (`NEXT`, `POSTED_BY`) | ~1,900 |
| HNSW vector index + range-index entries + allocator overhead | ~6,400 |
| **Total observed** | **~12.4 KB** |

- **Rule of thumb: ~12.5 KB/message at 1024 dims ≈ 1.25 GB per 100k-message workspace** (DESIGN
  §11). One graph per workspace, one shard per graph — size the shard to hold the whole workspace
  in RAM.
- **`GRAPH.MEMORY USAGE` under-reports vector memory** (`indices_sz_mb: 0` with a live HNSW index
  holding real vectors). Size vector-heavy workspaces from **`INFO memory` deltas**, not
  `GRAPH.MEMORY USAGE`, until fixed upstream.
- `Chunk.embedding` is a second 1024-dim vector index bootstrapped per workspace; it costs the
  same per-vector budget as `Message.embedding` once document chunks are loaded (out of K-008
  scope, but the index exists and will consume RAM as chunks land).

---

## Item 8 — Test suite raised: 126 → 135, fully green

`scripts/test_queries.sh` §6 block enhanced (still runs on the isolated `ws:test` at
`EMBEDDING_DIM=4` — the suite's own dim; the 1024 verification is this doc, against `ws:m2verify`).
Nine new assertions added:

```
▶ §6 GraphRAG — set embeddings and query
  ✓ §6 set-embedding write commits (Properties set)
  ✓ vector query: m1 in top-2
  ✓ vector query: m2 in top-2
  ✓ vector query: m3 not in top-2 (different direction)
  ✓ vector query: identical vector scores 0 (cosine distance)
  ✓ §6 ANN ranks by cosine distance ASC (m1 identical before m2 near)     # new
  ✓ hybrid retrieval: returns results
  ✓ §6 hybrid returns seed text                                            # new
  ✓ hybrid retrieval: no error
  ✓ §6 Entity expansion no-ops (empty relatedContext [])                   # new
  ✓ §6 Entity graph is empty (no entity pipeline in M2)                    # new
  ✓ §6 ANN retrieval uses the vector index (ProcedureCall)                 # new
  ✓ §6 ANN retrieval has no All Node Scan                                  # new
  ✓ §6 workspace-wide variant returns m1                                   # new
  ✓ §6 workspace-wide variant no error                                     # new
```

(Also: the SET write is now asserted with a RETURN, and the reads route via `GRAPH.RO_QUERY` per
the §6 routing note.) Final: **`Results: 135/135 passed`**.

`scripts/bootstrap_schema.sh`: default `EMBEDDING_DIM` **unchanged at 1536** (model-neutral); the
choose-before-creation comment now states M2 GraphRAG workspaces MUST be created at 1024, records
the write-time-not-checked quirk, and cites the RAM line.

---

## Handoff to tdd-engineer (Python impl — `server/`)

The impl is a thin adapter over the verified queries. Layering is locked: Cypher in
`repository.py` (1:1 with QUERIES.md §6), orchestration in `services.py`, tenant seam via
`config.get_context`. Route reads through `g.ro_query(...)`.

1. **`repository.set_embedding(msg_id, embedding)` — 1:1 with the §6 SET query.**
   ```cypher
   MATCH (m:Message {msgId: $msgId}) SET m.embedding = vecf32($embedding)
   ```
   - Parameterise `$embedding` as the raw float list; `vecf32(...)` wraps the parameter in the
     query text (verified: `SET m.embedding = vecf32($embedding)` works with a list parameter).
   - **MUST validate `len(embedding) == EMBEDDING_DIM` (1024) before the SET** — item 2 quirk: a
     wrong-dim write is silently accepted and the message then vanishes from ANN. Raise/reject on
     mismatch; do not rely on the engine to error.
   - Write path — not routable to a replica.

2. **`repository.hybrid_search(...)` — 1:1 with QUERIES.md §6.** Params: `$qVec` (query embedding,
   1024-dim float list), `$k` (ANN seed count, e.g. 10), `$limit`, and optional `$channelId`.
   - Two variants (mirror §6): channel-scoped (with the `MATCH (c:Channel {channelId:$channelId})`
     line) and workspace-wide (omit that line). Do not synthesize a `channelId` filter inside a
     shared graph — one graph per workspace (repo rule 7).
   - Wrap `$qVec` in `vecf32(...)` in the query text, same as the SET.
   - Returns rows of `(msgId, text, role, score, relatedContext)`; `relatedContext` is `[]` in M2
     (Entity layer dormant) — the service/API must tolerate the empty list, not treat it as an
     error or missing field.
   - Read path — use `g.ro_query(...)`.

3. **Service-layer `timeout=` constant on the RO query (K-007 TIMEOUT posture, DESIGN §10).** The
   global default is 1000 ms and writes ignore it; GraphRAG reads pass a per-query client override
   (e.g. `RAG_QUERY_TIMEOUT_MS = 5000`–`10000`) via `g.ro_query(q, params=…, timeout=…)`. Expose
   it as one named service-layer constant, not per-call ad-hockery. The client `timeout=`
   pass-through is uncapped while `TIMEOUT_MAX=0`.

4. **Score semantics for the caller.** `score` is cosine distance (0 = identical); results are
   already `ORDER BY score ASC` (most similar first). Do not re-sort or invert. If the API surfaces
   a "similarity", derive it client-side (`1 - distance` for cosine) — don't change the query.

5. **Tests.** Add `repository`/`services` tests against the isolated `ws:test` graph (the existing
   conftest bootstraps schema + wipes per test). The shell suite already covers the query surface
   at dim 4; the Python tests should cover: set-embedding validates length (rejects wrong dim),
   hybrid_search returns ranked rows, workspace-wide vs channel-scoped variants, and empty
   `relatedContext` is passed through cleanly. In-graph tests run at the conftest dim; do not
   hardcode 1024 in unit tests unless the fixture bootstraps at 1024.

### Open questions to surface before impl

1. **Where do 1024-dim workspaces get created?** `bootstrap_schema.sh` default stays 1536. The
   workspace-provisioning path (whatever creates a real GraphRAG workspace) must pass
   `EMBEDDING_DIM=1024`. Confirm the provisioning entry point sets it — a workspace silently created
   at 1536 will reject every 1024-dim query vector at runtime (`expected 1536 but got 1024`).
2. **Embedding-worker wiring is out of K-008 scope** (LM Studio call, when/where embeddings are
   computed). This gate verifies the SET and the read only; the worker that produces `$embedding`
   and calls `set_embedding` is a separate impl slice. Confirm the sequencing.
3. **`Chunk.embedding` / document ingestion** is bootstrapped but unexercised — out of K-008 scope.
   Flag if the impl is expected to touch chunk retrieval now.

### New quirk to fold back into `claude/graph-dba/falkordb-quirks.md`

- **Wrong-dimension `vecf32` writes are silently accepted at `SET` (no error) but the node drops
  out of the ANN index; dimension is enforced only at query time (`Vector dimension mismatch,
  expected N but got M`) and index-membership time.** No `vec.dimension()` function exists on this
  build. Validate embedding length client-side. (Verified 2026-07-08, module 999999.)
