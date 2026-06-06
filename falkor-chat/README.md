# falkor-chat

A hybrid chat system where **FalkorDB is the single store for everything** — chat history,
user and workspace data, reference data, workflow definitions and their execution traces.

Human users chat in channels. An AI participant answers from the graph via GraphRAG: vector
similarity search fused with multi-hop traversal in a single query. Workflows — both
conversational agent flows and business processes — are modelled as graph state machines and
executed inside the same graph.

---

## Core design decisions

| Axis | Choice |
|---|---|
| **Store** | FalkorDB for all domain data — no secondary store |
| **Tenancy** | One named graph per workspace (`ws:{id}`); global shared graphs for `reference` and `identity` |
| **Chat** | Hybrid — humans + AI agent on the same message timeline |
| **Retrieval** | In-graph vector index + traversal (GraphRAG hybrid) |
| **Workflows** | Graph state machine — one model for agent flows and business processes |

→ Full rationale, data model, Cypher patterns, index strategy, and ops blueprint: [`docs/DESIGN.md`](docs/DESIGN.md)

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- `redis-cli` — for local inspection (`sudo apt install redis-tools` / `brew install redis`)
- Python 3.11+ with `falkordb-py` 1.6.x (coming in M1)

---

## Dev environment

### 1 — Start FalkorDB

The script runs FalkorDB in the **foreground** so you see the logs. Open a dedicated terminal for it:

```bash
./start_falkordb.sh
```

| What | Where |
|---|---|
| FalkorDB / Redis | `localhost:6379` |
| Web console (browser UI) | `http://localhost:3000` |
| Data volume | `falkordb-data` (persists across restarts) |

Stop with **Ctrl+C**. Data survives in the Docker volume.

Need it in the **background** instead?
```bash
docker run --name falkordb-dev -p 6379:6379 -p 3000:3000 -v falkordb-data:/data --rm -d falkordb/falkordb:edge
docker stop falkordb-dev   # to stop
```

Wipe all data and start fresh:
```bash
docker volume rm falkordb-data
```

Override default ports:
```bash
FALKORDB_PORT=6380 FALKORDB_WEB_PORT=3001 ./start_falkordb.sh
```

### 2 — Verify the instance

In a second terminal, confirm everything is up:

```bash
redis-cli ping                 # → PONG
redis-cli MODULE LIST          # → graph (ver 999999)  +  vectorset
redis-cli GRAPH.LIST           # → (empty on a fresh volume)
```

### 3 — Bootstrap a workspace

Creates all indexes, constraints, full-text, and vector indexes for the `reference` graph
and one or more workspace graphs. Safe to re-run (idempotent):

```bash
./scripts/bootstrap_schema.sh <workspaceId> [<workspaceId> ...]

# examples:
./scripts/bootstrap_schema.sh myworkspace
./scripts/bootstrap_schema.sh acme globex          # multiple workspaces at once
EMBEDDING_DIM=3072 ./scripts/bootstrap_schema.sh acme   # override embedding dimension
```

Verify the result:
```bash
redis-cli GRAPH.QUERY ws:myworkspace "CALL db.indexes()"
redis-cli GRAPH.QUERY ws:myworkspace "CALL db.constraints()"
```

### 4 — Run the query test suite

Exercises every canonical query (write paths, read paths, full-text, vector ANN, hybrid
retrieval, agents, index usage) against the live instance. Uses an isolated `ws:test` graph
wiped before and after:

```bash
./scripts/test_queries.sh
```

Expected output: `64/64 passed`.

### 5 — Browse the graph (optional)

Open the FalkorDB web console at **http://localhost:3000** to explore graphs visually,
run ad-hoc Cypher, and inspect the schema.

---

## Multi-graph topology

```
identity          — global users, auth principals, cross-workspace membership
reference         — workflow definitions (versioned), ontology, tool registry
ws:{workspaceId}  — per-workspace: chat, embeddings, workflow runs  ← hot path
```

Edges are contained within a graph; cross-graph links are property keys resolved at the app
layer, or immutable snapshots materialized into the workspace graph (see §4 of the design doc).

---

## Key verified behaviours (live probes on `falkordb/falkordb:edge`, Redis 8.2.2)

- **Vector index** uses DDL, not a procedure call:
  ```sql
  CREATE VECTOR INDEX FOR (n:Message) ON (n.embedding)
  OPTIONS {dimension:1536, similarityFunction:'cosine'}
  ```
- **Uniqueness constraints** require an existing range index on the same property first:
  ```bash
  GRAPH.QUERY ws:acme "CREATE INDEX FOR (n:User) ON (n.userId)"
  GRAPH.CONSTRAINT CREATE ws:acme UNIQUE NODE User PROPERTIES 1 userId
  ```
- **Vector query** score is cosine *distance* — `0` = identical, lower = more similar. Sort `ASC`.
- **Cross-graph edges** silently do nothing — no error, MATCH returns 0 rows. The materialization
  pattern is intentional (see §4 of DESIGN.md).
- **`algo.*` procedures** available: `BFS`, `WCC`, `pageRank`, `SPpaths`, `SSpaths`, `MSF`,
  `betweenness`, `labelPropagation`.

---

## Roadmap

| Milestone | Status | Scope |
|---|---|---|
| **M0** — Engine up | ✅ | FalkorDB running, live-probed, design locked, schema + queries verified (64/64) |
| **M1** — Chat core | — | Python layer: users, channels, threads, thread-scoped message append, full-text search |
| **M2** — GraphRAG | — | Embeddings, vector index, AI agent participant, hybrid retrieval |
| **M3** — Workflows | — | Def → snapshot → run/step executor, chat linkage |
| **M4** — Scale & ops | — | Redis Cluster, replicas, ACL/TLS, memory budgeting |

---

## Repository layout

```
falkor-chat/
├── docs/
│   ├── DESIGN.md          # full blueprint (data model, Cypher, ops)
│   └── QUERIES.md         # canonical query library — verified against live instance
├── scripts/
│   ├── bootstrap_schema.sh  # create indexes + constraints for any workspace
│   └── test_queries.sh      # end-to-end query test suite (64 assertions)
├── start_falkordb.sh      # spin up FalkorDB in Docker
└── README.md
```
