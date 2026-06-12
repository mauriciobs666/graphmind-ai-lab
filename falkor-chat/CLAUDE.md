# falkor-chat — Claude working context

## Project in one sentence

A hybrid chat system (humans + AI) where **FalkorDB is the single store for everything**:
chat history, workspace data, reference data, workflow definitions and execution traces.

---

## Decisions locked in — do not reopen without strong cause

| Decision | Rationale |
|---|---|
| FalkorDB for all domain data, no secondary store | philosophy of the project |
| One graph per workspace (`ws:{id}`) | blast-radius isolation, natural cluster sharding |
| Thread-scoped `NEXT` linked list | users read threads, not channel feeds |
| No DayBucket | was designed for channel-wide ordering; dropped when thread-scoped was chosen |
| `Thread` owns `HEAD` and `TAIL` pointers | Thread stays sparse (2 edges) regardless of message count |
| `Message.role` as inline property | filter by role without traversing `POSTED_BY` |
| `coalesce(u.userId, u.agentId)` for member identity | `User` has `userId`, `Agent` has `agentId` — both are channel members |
| Vector indexes via DDL, not a procedure | `db.idx.vector.createNodeIndex` is not registered in this build |
| Index before constraint, always | `GRAPH.CONSTRAINT CREATE` requires a pre-existing range index |
| `Message.embedding` stored inline as `vecf32` | enables single-query vector + traversal hybrid retrieval |
| Vector score is cosine **distance** (0 = identical) | sort `ORDER BY score ASC` for most-similar first |
| `status` as property, not label | avoids re-labeling churn on state changes; index it |
| `ctx`, `input`, `output` must be flat/serialised | FalkorDB stores scalars and scalar lists only — no nested maps |

---

## Live-verified FalkorDB facts (falkordb/falkordb:edge, Redis 8.2.2, module 999999)

These diverge from general docs or Neo4j assumptions — treat them as ground truth:

- **Vector index creation** is DDL:
  ```sql
  CREATE VECTOR INDEX FOR (n:Label) ON (n.prop)
  OPTIONS {dimension: N, similarityFunction: 'cosine'}
  ```
- **`db.idx.vector.createNodeIndex` is not a registered procedure** on this build.
- **Constraint ordering:** `CREATE INDEX FOR …` must run before `GRAPH.CONSTRAINT CREATE` on the same property, or you get `"missing supporting exact-match index"`.
- **Composite constraints** (`PROPERTIES 2 key version`) are supported and operational — verified.
- **Cross-graph edges** silently no-op — no error, MATCH returns 0 rows. There is no error to catch.
- **`(:User | :Agent)` union label syntax** in Cypher patterns — not verified on this build; use `coalesce()` or label-agnostic traversal instead.
- **`length(path)` in ORDER BY** — not supported. Use a property (e.g. `m.createdAt`) instead.
- **Fulltext index** via `CALL db.idx.fulltext.createNodeIndex('Label', 'prop')` — confirmed.
- **`algo.*` procedures confirmed:** `BFS`, `WCC`, `pageRank`, `SPpaths`, `SSpaths`, `MSF`, `betweenness`, `labelPropagation`.
- **`GRAPH.RO_QUERY`** routes to replicas — use for all read-only queries (RAG retrieval, thread reads).
- **Bolt port** is `65535` per `GRAPH.CONFIG`.
- **Default `TIMEOUT` is 1000ms** — may fire on GraphRAG queries over large workspaces; review before M2.

---

## Graph topology

```
identity          — global user identity, auth (read-mostly, replicated)
reference         — WorkflowDef templates, ontology, tool registry (read-mostly, replicated)
ws:{workspaceId}  — per-workspace hot path: chat, embeddings, workflow runs
```

Edges cannot cross graphs. Cross-graph references use property keys or materialized snapshots.

---

## Schema conventions

- Labels: `PascalCase` — `User`, `Channel`, `Thread`, `Message`, `Agent`
- Relationship types: `UPPER_SNAKE` — `POSTED_BY`, `REPLY_TO`, `HAS_THREAD`, `NEXT`
- Properties: `camelCase` — `userId`, `createdAt`, `embedding`
- Graph keys: `ws:{workspaceId}`, `reference`, `identity`
- Every entity node has a stable `{label}Id` property, a range index, and a uniqueness constraint
- Every `MERGE` must be backed by a uniqueness constraint — no exceptions

---

## Message write paths (two variants — keep them separate)

**First message in a thread** (no HEAD/TAIL yet):
```cypher
MATCH (t:Thread {threadId:$threadId})
MATCH (author {userId:$authorId})
MERGE (m:Message {msgId:$msgId})
  ON CREATE SET m.text=$text, m.role=$role, m.createdAt=$createdAt
CREATE (t)-[:HEAD]->(m), (t)-[:TAIL]->(m), (m)-[:POSTED_BY]->(author)
SET t.updatedAt = $createdAt
```

**Subsequent messages** (move TAIL forward — atomic, single query):
```cypher
MATCH (t:Thread {threadId:$threadId})-[tailRel:TAIL]->(prev:Message)
MATCH (author {userId:$authorId})
MERGE (m:Message {msgId:$msgId})
  ON CREATE SET m.text=$text, m.role=$role, m.createdAt=$createdAt
CREATE (prev)-[:NEXT]->(m), (t)-[:TAIL]->(m), (m)-[:POSTED_BY]->(author)
DELETE tailRel
SET t.updatedAt = $createdAt
```

---

## Key scripts

| Script | Purpose |
|---|---|
| `./scripts/start_falkordb.sh` | Start FalkorDB in Docker (foreground). Data in `falkordb-data` volume. |
| `./scripts/bootstrap_schema.sh <wsId> …` | Create all indexes + constraints for `reference` + workspace(s). Idempotent. |
| `./scripts/test_queries.sh` | 64-assertion end-to-end test suite against the live instance. Must pass before any schema change is committed. |

Bootstrap takes an optional `EMBEDDING_DIM` env var (default `1536`). Set it to match the
embedding model before creating a workspace.

---

## Key documents

| File | Contents |
|---|---|
| `docs/DESIGN.md` | Full blueprint: graph topology, data model, indexes, ops, roadmap |
| `docs/QUERIES.md` | Canonical query library — all verified against the live instance |

---

## Rules for future work

1. **Always parameterise Cypher.** Never interpolate variables into query strings.
2. **Verify dialect before assuming.** This is FalkorDB OpenCypher, not Neo4j. No APOC, no GDS, no `PROFILE` keyword prefix. Check `CALL dbms.procedures()` when unsure.
3. **Profile before tuning.** Use `GRAPH.PROFILE` to confirm an index is actually hit before declaring a query fast. Look for `Node By Index Scan`, not `NodeByLabelScan`.
4. **All writes that touch HEAD/TAIL must be a single `GRAPH.QUERY`** — atomicity is per-query.
5. **Test suite must stay green.** Run `./scripts/test_queries.sh` after any schema or query change. 64/64 is the baseline.
6. **RAM is the binding constraint.** Any new node type, index, or vector dimension affects per-workspace RAM. Call it out.
7. **One graph per workspace.** Never add a `workspaceId` property to filter inside a shared graph.
8. **`ctx`, `input`, `output` on workflow nodes are serialised strings.** Do not design queries that filter inside them.
