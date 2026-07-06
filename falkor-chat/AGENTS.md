# falkor-chat — agent working context

## Project in one sentence

A hybrid chat system (humans + AI) where **FalkorDB is the single store for everything**:
chat history, workspace data, reference data, workflow definitions and execution traces.

---

## Decisions locked in — do not reopen without strong cause

> Rationale lives once, in `docs/DESIGN.md` §1 (the authoritative register). This is the quick
> do-not-reopen index — follow the link for the *why*.

| Decision | Home |
|---|---|
| FalkorDB is the single store (no secondary store) | DESIGN §1.2 → §2 |
| One graph per workspace (`ws:{id}`) | DESIGN §1.1 (Tenancy) / §3 |
| Thread-scoped `NEXT` linked list | DESIGN §1.2 → §5.2 |
| No DayBucket | DESIGN §1.2 |
| `Thread` owns `HEAD`+`TAIL` pointers | DESIGN §1.2 → §5.2 |
| `Message.role` inline + derived server-side | DESIGN §1.2 → §5.1 |
| `coalesce` member identity | DESIGN §1.2 → QUERIES §2 |
| Vector indexes via DDL, not a procedure | DESIGN §1.2 → §7.1 |
| Index before constraint, always | DESIGN §1.2 → §7.1 |
| `Message.embedding` inline `vecf32` | DESIGN §1.2 → §5.2 |
| Vector score is cosine distance (ASC) | DESIGN §1.2 → §8 |
| `status` as property, not label | DESIGN §1.2 → §6.2 |
| Flat `ctx`/`input`/`output` | DESIGN §1.2 → §6.2 |
| `Message.threadId` denorm, unindexed | DESIGN §1.2 → §5.1 |
| Guarded-CREATE write paths + status row | DESIGN §1.2 → §5.3/§9 |
| Composite `(createdAt, msgId)` keyset cursor | DESIGN §1.2 → QUERIES §9 |
| Member ids namespace-unique across `User`/`Agent` | DESIGN §1.2 → QUERIES §2/§7 |
| Identity graph is authoritative (standalone) | DESIGN §1.2 → §3 |

---

## Live-verified FalkorDB facts (falkordb/falkordb:edge, Redis 8.2.2, module 999999)

General engine/dialect quirks verified against this build (vector index DDL, index-before-constraint
ordering, the `exists()` pattern bug, empty-`UNWIND` row collapse, `TIMEOUT` behavior, `OR`-as-scan-anchor,
etc.) now live in the `graph-dba` agent's knowledge base, **`claude/graph-dba/falkordb-quirks.md`** —
check there first. What's below is specific to this project's schema/queries:

- **`repository.thread_has_head`/`thread_exists`** exist specifically to route around graph-dba's
  `exists()`-pattern-bug finding — they use `OPTIONAL MATCH (n)-[:REL]->(x) RETURN x IS NOT NULL`,
  never a pattern-`exists()` check.
- **The mention write-block's empty-`UNWIND` guard is load-bearing for the write itself**, not just
  the mentions (see `QUERIES.md` §4 mentions note): `UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE
  $mentions END) AS mid` + a `FOREACH` that never filters. A bare `UNWIND []` would collapse the whole
  row stream before that `FOREACH`, silently dropping the message write, not just the
  `MENTIONS_MEMBER` edges.
- **Member resolution (`userId`/`agentId`) is the concrete case of graph-dba's `OR`-scan-anchor
  quirk** — two `OPTIONAL MATCH (u:User {userId:mid})` / `(a:Agent {agentId:mid})` + `coalesce(u,a)`
  for anchored lookups (`labels(coalesce(u,a))[0]` gives the member kind). The `OR` form is fine only
  in mention-flag and cursor reads, where `n` is already bound by a traversal/indexed anchor.
- **Formulation-A composite keyset predicate** (`m.createdAt > $since OR (m.createdAt = $since
  AND m.msgId > $sinceMsgId)`) still plans as a bare `Node By Index Scan` on `Message.createdAt`
  with no residual Filter — **re-profile on engine upgrades** (edge build; formulation B in
  QUERIES.md §9.1 is the documented fallback).
- **`TIMEOUT` default (1000ms) was reviewed for M2 (K-007) and kept as the deployment default** —
  writes ignore it entirely regardless (graph-dba finding); GraphRAG reads pass a per-query client
  `timeout=` override instead (DESIGN §10 posture).

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
duplication is what lets the copies drift. The invariants that govern those queries (v2, K-007):

- **Two separate write paths, never a conditional MERGE:** *first message in a thread*
  (creates `HEAD` + `TAIL`) vs. *subsequent message* (moves `TAIL` forward via `NEXT`). Each is
  **self-guarding**: the write happens inside a `FOREACH (… IN CASE WHEN ok THEN [1] ELSE []
  END | …)` guard *per path* — a guarded `CREATE`, **no MERGE on Message** (the constraint
  stays as the concurrency backstop). The service picks the initial variant by checking for a
  `HEAD`, then dispatches on the returned status row.
- **Status-row contract:** both paths always return `(written, hadHead, dupMsg, authorFound)`
  when their anchor matches. **Zero rows = the anchor missed only** (first: thread missing →
  404; subsequent: no TAIL → retry as first). `dupMsg=true` = **idempotent success** (a retry
  replay of our own server-minted msgId — trusted without a payload check; add a checksum if
  msgIds ever become client-supplied). `hadHead=true` = lost the first-post race → re-dispatch
  as subsequent. `authorFound=false` = unknown member, nothing written. The dispatch loop is
  bounded at 4 attempts (tripwire — ping-pong is impossible by contract).
- **Each write is a single `GRAPH.QUERY`** — atomicity is per-query; the HEAD/TAIL relink must
  not be split across queries.
- **`role` is derived, never trusted:** the service resolves the author's label
  (`User → user`, `Agent → assistant`) via the §2 member-kind lookup — Agents author
  first-class. Author resolution in the write is label-specific (two indexed `OPTIONAL
  MATCH`es + `coalesce`), closing the old `All Node Scan`/silent-Agent-no-op defect.
- **Every message records its author** with `(m)-[:POSTED_BY]->(author)`. The canonical
  thread-read path (`QUERIES.md` §4) *requires* that edge — a message written without it is
  invisible to thread reads.
- **Participant mentions ride inside the same write query.** Mention resolution runs before the
  guard; the nested `FOREACH` creates `(m)-[:MENTIONS_MEMBER]->(member)` edges *inside* it —
  never a follow-up query (atomicity rule). The empty-`UNWIND` `CASE` guard is now
  **load-bearing for the write itself** (a bare `UNWIND []` collapses the stream before the
  FOREACH). `MENTIONS_MEMBER` (participants) is **distinct from** `MENTIONS`→`Entity` (GraphRAG
  co-occurrence, §6) — do not conflate them. `$mentions = []` is a verified no-op.
- **Every `MERGE` is backed by a uniqueness constraint** (`ReadCursor.cursorId`; `ensure_user`/
  `ensure_agent`). Channel/thread creates are plain `CREATE` (server-minted ids —
  **non-idempotent**, a retried create mints a new id).
- **The service owns the timestamps:** message `createdAt` comes from a lock-guarded monotonic
  per-process clock (`max(clock, last+1)`) — same-ms ties are impossible at the source.
- **Since-reads (§9.1/§9.2) are chronological in the `(createdAt, msgId)` total order; cursors
  advance to what was delivered.** Reader mentions are carried by the `isMention` flag, never a
  mention-first sort — a resorted page + `LIMIT` breaks the contiguous-prefix invariant.
  Cursor-driven reads use the composite keyset + the composite `ReadCursor` pair (advanced to
  the newest *returned* `(createdAt, msgId)`, never the server clock) and never skip or
  re-deliver, even across millisecond ties. Explicit-`since` reads keep plain `>` semantics
  (may re-deliver/skip within that exact millisecond — documented, OQ3).

---

## Key scripts

| Script | Purpose |
|---|---|
| `./scripts/start_falkordb.sh` | Start FalkorDB in Docker (foreground; `-d`/`--detach` for headless). Data in `falkordb-data` volume. |
| `./scripts/bootstrap_schema.sh <wsId> …` | Create all indexes + constraints for `reference` + workspace(s). Idempotent. |
| `./scripts/test_queries.sh` | 126-assertion end-to-end test suite against the live instance. Must pass before any schema change is committed. |
| `./scripts/backfill_thread_ids.sh <wsId> …` | One-off: stamp `Message.threadId` on pre-K-007 messages (QUERIES.md §4.x). Idempotent; run once per existing workspace after deploying the v2 write paths. |

Bootstrap takes an optional `EMBEDDING_DIM` env var (default `1536`). Set it to match the
embedding model before creating a workspace.

### M1 server (`server/`)

The M1 app (FastAPI REST + MCP Streamable-HTTP + static web UI on one process) lives in `server/`
(and `web/`). No `uv` on the box — use a `venv`.

```bash
cd server
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # first time
.venv/bin/python -m pytest -q                                # 110 passed (needs FalkorDB up)
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
| `docs/plans/m2-groundwork.md` · `docs/plans/m2-groundwork-queries.md` | K-007 plan + graph-dba verified-query deliverable: v2 write paths, keyset cursors, threadId denorm, TIMEOUT/RAM findings |

---

## Rules for future work

1. **Always parameterise Cypher.** Never interpolate variables into query strings.
2. **Verify dialect before assuming.** This is FalkorDB OpenCypher, not Neo4j. No APOC, no GDS, no `PROFILE` keyword prefix. Check `CALL dbms.procedures()` when unsure.
3. **Profile before tuning.** Use `GRAPH.PROFILE` to confirm an index is actually hit before declaring a query fast. Look for `Node By Index Scan`, not `NodeByLabelScan`.
4. **All writes that touch HEAD/TAIL must be a single `GRAPH.QUERY`** — atomicity is per-query.
5. **Test suite must stay green.** Run `./scripts/test_queries.sh` after any schema or query change. 126/126 is the baseline.
6. **RAM is the binding constraint.** Any new node type, index, or vector dimension affects per-workspace RAM. Call it out.
7. **One graph per workspace.** Never add a `workspaceId` property to filter inside a shared graph.
8. **`ctx`, `input`, `output` on workflow nodes are serialised strings.** Do not design queries that filter inside them.
