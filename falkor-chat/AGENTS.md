# falkor-chat — agent working context

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
- **Empty `UNWIND` collapses the row stream.** `WITH m UNWIND [] AS x …` drops `m` and a trailing `RETURN m` comes back empty even though earlier writes committed. The mention write-block guards this with `UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid` + a `FOREACH` that never filters — verified regression-safe (see `QUERIES.md` §4 mentions note).
- **`FOREACH (x IN CASE … | CREATE …)`** is the idiom for conditional writes without dropping rows — confirmed on this build.
- **`exists((n)-[:REL]->())` in a pattern returns `true` even when the edge is absent** (broken on this build), and `count{ … }` subquery syntax is unsupported. For existence checks use `OPTIONAL MATCH (n)-[:REL]->(x) RETURN x IS NOT NULL` instead (used by `repository.thread_has_head`/`thread_exists`).
- **Member resolution must be label-specific.** `WHERE n.userId = $x OR n.agentId = $x` as a *scan anchor* profiles as an `All Node Scan`; two `OPTIONAL MATCH (u:User {userId:mid}) / (a:Agent {agentId:mid})` + `coalesce(u,a)` gives two `Node By Index Scan`s. The `OR` form is fine only when `n` is already bound by a traversal/indexed anchor (mention-flag and cursor reads).

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

- Labels: `PascalCase` — `User`, `Channel`, `Thread`, `Message`, `Agent`, `ReadCursor`
- Relationship types: `UPPER_SNAKE` — `POSTED_BY`, `REPLY_TO`, `HAS_THREAD`, `NEXT`, `MENTIONS_MEMBER`, `HAS_CURSOR`
- Properties: `camelCase` — `userId`, `createdAt`, `embedding`
- Graph keys: `ws:{workspaceId}`, `reference`, `identity`
- Every entity node has a stable `{label}Id` property, a range index, and a uniqueness constraint
- Every `MERGE` must be backed by a uniqueness constraint — no exceptions

---

## Message write paths (two variants — keep them separate)

The exact, verified Cypher lives in **one place — `docs/QUERIES.md` §4** (single source of
truth). Do not copy query bodies here or into `DESIGN.md`; link to QUERIES.md instead — the
duplication is what lets the copies drift. The invariants that govern those queries:

- **Two separate write paths, never a conditional MERGE:** *first message in a thread*
  (creates `HEAD` + `TAIL`) vs. *subsequent message* (moves `TAIL` forward via `NEXT`). The
  service picks the variant by checking whether the thread already has a `HEAD`.
- **Each write is a single `GRAPH.QUERY`** — atomicity is per-query; the HEAD/TAIL relink must
  not be split across queries.
- **Every message records its author** with `(m)-[:POSTED_BY]->(author)`. The canonical
  thread-read path (`QUERIES.md` §4) *requires* that edge — a message written without it is
  invisible to thread reads.
- **Participant mentions ride inside the same write query.** Both write paths carry a `$mentions`
  list and append the mention block (`QUERIES.md` §4) that writes `(m)-[:MENTIONS_MEMBER]->(member)`
  edges atomically — never a follow-up query (atomicity rule). `MENTIONS_MEMBER` (participants) is
  **distinct from** `MENTIONS`→`Entity` (GraphRAG co-occurrence, §6) — do not conflate them.
  `$mentions = []` is a verified no-op.
- **Every `MERGE` is backed by a uniqueness constraint** (`Message.msgId`; `ReadCursor.cursorId`).
- **A write that returns zero rows wrote nothing.** The §4 queries anchor on `MATCH` (thread,
  author, TAIL); a missing anchor no-ops the whole query with no error. The repository raises on
  an empty result, the service validates the author is a known member before writing, and
  `create_app`'s lifespan runs `services.ensure_actor()` so the configured actor node exists
  before the first write.
- **Since-reads (§9.1/§9.2) are chronological; cursors advance to what was delivered.** Reader
  mentions are carried by the `isMention` flag, never a mention-first sort — a resorted page +
  `LIMIT` breaks the contiguous-prefix invariant and the cursor (advanced to the newest *returned*
  `createdAt`, never the server clock) would skip messages permanently.

---

## Key scripts

| Script | Purpose |
|---|---|
| `./scripts/start_falkordb.sh` | Start FalkorDB in Docker (foreground; `-d`/`--detach` for headless). Data in `falkordb-data` volume. |
| `./scripts/bootstrap_schema.sh <wsId> …` | Create all indexes + constraints for `reference` + workspace(s). Idempotent. |
| `./scripts/test_queries.sh` | 92-assertion end-to-end test suite against the live instance. Must pass before any schema change is committed. |

Bootstrap takes an optional `EMBEDDING_DIM` env var (default `1536`). Set it to match the
embedding model before creating a workspace.

### M1 server (`server/`)

The M1 app (FastAPI REST + MCP Streamable-HTTP + static web UI on one process) lives in `server/`
(and `web/`). No `uv` on the box — use a `venv`.

```bash
cd server
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # first time
.venv/bin/python -m pytest -q                                # 75 passed (needs FalkorDB up)
.venv/bin/uvicorn falkorchat.app:app                         # web UI + REST under /, MCP at /mcp
```

- **Layering (locked):** `api.py` (REST) and `mcp.py` (MCP) are thin adapters over `services.py`;
  all Cypher lives in `repository.py` (1:1 with `QUERIES.md`); the tenant seam is `config.get_context`.
- **Front doors on one process:** `app.py` mounts REST + MCP, and serves the repo-root `web/`
  (`index.html` + `app.js`) as static files at `/`. The static mount is registered **last** — `/`
  is a catch-all that must sit behind the REST routes and the `/mcp` mount. Same-origin ⇒ no CORS.
- **Full-text search:** `GET /search?q=` → `services.search_messages` → `repository.search_messages`
  (`QUERIES.md` §5, workspace-wide — the channel-scoping MATCH is omitted).
- Repository/services tests run against the isolated `ws:test` graph (same approach as
  `test_queries.sh`); the `conftest` fixture bootstraps schema + wipes node data per test.
- MCP is tested in-memory (`mcp.call_tool` / `list_tools`) — no HTTP server needed.

---

## Key documents

| File | Contents |
|---|---|
| `docs/DESIGN.md` | Full blueprint: graph topology, data model, indexes, ops, roadmap, §14–§15 M1 app + MCP |
| `docs/QUERIES.md` | Canonical query library — all verified against the live instance |
| `docs/plans/m1-chat-mcp.md` | K-002 plan: MCP transport + mentions + read-cursors |

---

## Rules for future work

1. **Always parameterise Cypher.** Never interpolate variables into query strings.
2. **Verify dialect before assuming.** This is FalkorDB OpenCypher, not Neo4j. No APOC, no GDS, no `PROFILE` keyword prefix. Check `CALL dbms.procedures()` when unsure.
3. **Profile before tuning.** Use `GRAPH.PROFILE` to confirm an index is actually hit before declaring a query fast. Look for `Node By Index Scan`, not `NodeByLabelScan`.
4. **All writes that touch HEAD/TAIL must be a single `GRAPH.QUERY`** — atomicity is per-query.
5. **Test suite must stay green.** Run `./scripts/test_queries.sh` after any schema or query change. 92/92 is the baseline.
6. **RAM is the binding constraint.** Any new node type, index, or vector dimension affects per-workspace RAM. Call it out.
7. **One graph per workspace.** Never add a `workspaceId` property to filter inside a shared graph.
8. **`ctx`, `input`, `output` on workflow nodes are serialised strings.** Do not design queries that filter inside them.
