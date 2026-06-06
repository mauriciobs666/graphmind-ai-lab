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

## Start FalkorDB locally

```bash
./start_falkordb.sh
```

| What | Where |
|---|---|
| FalkorDB / Redis | `localhost:6379` |
| Web console (browser UI) | `http://localhost:3000` |
| Data volume | `falkordb-data` (persists across restarts) |

Override defaults with env vars:

```bash
FALKORDB_PORT=6380 FALKORDB_WEB_PORT=3001 ./start_falkordb.sh
```

Stop with **Ctrl+C**. Data survives in the Docker volume. To wipe it completely:

```bash
docker volume rm falkordb-data
```

### Verify the instance

```bash
redis-cli ping                          # → PONG
redis-cli MODULE LIST                   # → graph (ver 999999) + vectorset
redis-cli GRAPH.LIST                    # → (empty on a fresh volume)
redis-cli GRAPH.CONFIG GET "*"          # → runtime tuning knobs
```

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
| **M0** — Engine up | ✅ | FalkorDB running, live-probed, design locked |
| **M1** — Chat core | — | Users, Channels, Messages, `DayBucket` append, full-text index |
| **M2** — GraphRAG | — | Embeddings, vector index, AI agent participant, hybrid retrieval |
| **M3** — Workflows | — | Def → snapshot → run/step executor, chat linkage |
| **M4** — Scale & ops | — | Redis Cluster, replicas, ACL/TLS, memory budgeting |

---

## Repository layout

```
falkor-chat/
├── docs/
│   └── DESIGN.md          # full blueprint (data model, Cypher, ops)
├── start_falkordb.sh      # spin up FalkorDB in Docker
└── README.md
```
