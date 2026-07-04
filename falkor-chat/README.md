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
- Python 3.12+ for the M1 server (`server/` — FastAPI + `falkordb-py` 1.6.x + `mcp`)

---

## Dev environment

### 1 — Start FalkorDB

The script runs FalkorDB in the **foreground** so you see the logs. Open a dedicated terminal for it:

```bash
./scripts/start_falkordb.sh
```

| What | Where |
|---|---|
| FalkorDB / Redis | `localhost:6379` |
| Web console (browser UI) | `http://localhost:3000` |
| Data volume | `falkordb-data` (persists across restarts) |

Stop with **Ctrl+C**. Data survives in the Docker volume.

Need it in the **background** instead? Use the `-d` / `--detach` flag:
```bash
./scripts/start_falkordb.sh -d     # run headless
docker stop falkordb-dev           # to stop
```

Wipe all data and start fresh:
```bash
docker volume rm falkordb-data
```

Override default ports:
```bash
FALKORDB_PORT=6380 FALKORDB_WEB_PORT=3001 ./scripts/start_falkordb.sh
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

Expected output: `92/92 passed`.

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
| **M0** — Engine up | ✅ | FalkorDB running, live-probed, design locked, schema + queries verified (92/92) |
| **M1** — Chat core | 🟡 | FastAPI REST server (router → service → repository over `falkordb-py`) **+ MCP (Streamable-HTTP) agent front door** on the same service layer; single hardcoded tenant; users, channels, threads, thread-scoped append, @mentions, read-cursors, full-text search, and a minimal static web UI — all on one process (75 tests). Code-complete; hardening/real-time deferred to M2. See [DESIGN.md §14–§15](docs/DESIGN.md#14-m1-application-architecture-clientserver) |
| **M2** — GraphRAG | — | Embeddings, vector index, AI agent participant, hybrid retrieval |
| **M3** — Workflows | — | Def → snapshot → run/step executor, chat linkage |
| **M4** — Scale & ops | — | Redis Cluster, replicas, ACL/TLS, memory budgeting |

---

## Repository layout

```
falkor-chat/
├── docs/
│   ├── DESIGN.md          # full blueprint (data model, Cypher, ops, §14 M1 app architecture)
│   └── QUERIES.md         # canonical query library — verified against live instance
├── kaizen/
│   ├── plan.md            # forward-looking backlog (active items + parking lot)
│   └── history.md         # dated change log
├── scripts/
│   ├── bootstrap_schema.sh  # create indexes + constraints for any workspace
│   ├── start_falkordb.sh    # spin up FalkorDB in Docker
│   └── test_queries.sh      # end-to-end query test suite (92 assertions)
├── server/                  # M1 app: FastAPI REST + MCP on one process
│   ├── falkorchat/{config,db,repository,services,schemas,api,mcp,app}.py
│   ├── tests/               # pytest — repository/services (live), MCP, REST, app-mount
│   └── pyproject.toml       # fastapi, uvicorn, falkordb, mcp, pytest, httpx
├── web/                     # minimal browser client (index.html + app.js) served by app.py
└── README.md
```

---

## Run the M1 server (REST + MCP)

The server hosts the browser REST API and the MCP agent front door on one process.

```bash
cd server
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # first time
./../scripts/bootstrap_schema.sh acme                        # schema for the default tenant (ws:acme)
.venv/bin/uvicorn falkorchat.app:app                         # web UI + REST under /, MCP at /mcp
```

Then open **http://localhost:8000/** for the minimal web client (channels · threads · messages ·
search). It talks to the REST API on the same origin (no CORS). The browser and MCP front doors
share one `services.py`.

Run the server test suite (needs FalkorDB up; uses an isolated `ws:test` graph):

```bash
cd server && .venv/bin/python -m pytest -q      # 75 passed
```

Agents connect to MCP at `http://localhost:8000/mcp` (`type: streamable-http`; the trailing-slash
spelling `/mcp/` works too). The endpoint is unauthenticated in M1 — bind to localhost / a trusted
network only. Tools: `send_message`, `read_messages`, `create_thread`, `create_channel`,
`list_channels`, `list_threads`, `search_messages` (see [DESIGN.md §15](docs/DESIGN.md#15-mcp-transport-k-002--the-agent-front-door)).
