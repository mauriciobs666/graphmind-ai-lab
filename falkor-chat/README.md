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
- Python 3.12+ for the server (`server/` — FastAPI + `falkordb-py` 1.6.x + `mcp`)
- **For the AI agent (M2 GraphRAG) only:** an OpenAI-compatible LLM endpoint — e.g.
  [LM Studio](https://lmstudio.ai/) on `:1234` — serving an **embedding** model
  (`text-embedding-qwen3-embedding-0.6b`, 1024-dim) **and** a **chat** model
  (`qwen/qwen3-4b-2507`). Not needed if you run with `FALKORCHAT_ENABLE_AGENT=0` (chat core only).

---

## Quick start

**A — Full GraphRAG, with the AI agent.** Start LM Studio (both models above, server on `:1234`),
then from `falkor-chat/`:

```bash
./scripts/start_server.sh
```

One command does it all: starts FalkorDB (detached) → creates the `server/.venv` → bootstraps
`ws:acme` **at `EMBEDDING_DIM=1024`** → seeds the `assistant` agent + a `#general` channel /
`Welcome` thread → launches uvicorn with the AI responder enabled. Then open
**http://localhost:8000/**, go to `#general` → `Welcome`, and post `@assistant <your question>` —
a graph-grounded reply (with `EMITTED` provenance) appears within a few seconds.

**B — Chat core only, no LM Studio / GPU.** Same UI/REST/MCP on `:8000`, minus the AI responder:

```bash
FALKORCHAT_ENABLE_AGENT=0 ./scripts/start_server.sh
```

Stop with **Ctrl+C** (FalkorDB keeps running; `docker stop falkordb-dev` to stop it too).
Prefer to wire things up step by step, or run just FalkorDB? See **Dev environment** below.

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

**One-command alternative — Docker Compose.** `compose.yaml` runs FalkorDB **and** the
M1 server (REST + MCP + web UI) together:

```bash
docker compose up --build     # FalkorDB :6379/:3000 + server at http://localhost:8000
docker compose down           # stop — data persists in falkordb-data; never `down -v`
```

The FalkorDB service uses the same image, ports, and `falkordb-data` volume as the script,
so pick one at a time: stop `falkordb-dev` before `compose up` (they share the ports *and*
the volume). The server container connects via `FALKORDB_HOST=falkordb` and healthchecks
on `GET /health`.

### 2 — Verify the instance

In a second terminal, confirm everything is up:

```bash
redis-cli ping                 # → PONG
redis-cli MODULE LIST          # → graph (ver 41811 = v4.18.11)  +  vectorset
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
EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh acme   # M2 GraphRAG dim (Qwen3-Embedding)
```

> **For the AI agent (M2), bootstrap at `EMBEDDING_DIM=1024`** to match the embedding model. The
> dimension is fixed at index-creation time and the running app must use the same value
> (`FALKORCHAT_EMBEDDING_DIM`) — a wrong-dim vector is silently accepted then dropped from the ANN
> index. `start_server.sh` handles this for you (defaults to 1024).

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

Expected output: all assertions pass.

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

## Key verified behaviours (live probes on `falkordb/falkordb:edge`; re-verified 2026-07-09 on the pinned `v4.18.11` via the full query suite)

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
| **M0** — Engine up | ✅ | FalkorDB running, live-probed, design locked, schema + queries verified |
| **M1** — Chat core | ✅ | FastAPI REST server (router → service → repository over `falkordb-py`) **+ MCP (Streamable-HTTP) agent front door** on the same service layer; single hardcoded tenant; users, channels, threads, thread-scoped append, @mentions, read-cursors, full-text search, and a minimal static web UI — all on one process. DoD closed: append path load-tested + hot reads `GRAPH.PROFILE`d (§11.1), web request/response de-staled (K-012). Hardening/real-time (auth, push) deferred to M2.5. See [DESIGN.md §14–§15](docs/DESIGN.md#14-m1-application-architecture-clientserver) |
| **M2** — GraphRAG | ✅ | Every message embedded out-of-band (async worker → LM Studio, 1024-dim); in-graph vector index @1024 + hybrid retrieval (`hybrid_search`, cosine-ASC); AI `Agent` participant — `@mention` triggers a retrieval-grounded LLM answer posted as the agent (`role:"assistant"`) with an `EMITTED` provenance edge; web renders assistant replies + reader `isMention`. QA-accepted (K-015, PASS). Served via `start_server.sh` (gated on `FALKORCHAT_ENABLE_AGENT`, `EMBEDDING_DIM=1024`). Auth + real-time deferred to M2.5 |
| **M3** — Workflows | — | Def → snapshot → run/step executor, chat linkage |
| **M4** — Scale & ops | — | Redis Cluster, replicas, ACL/TLS, memory budgeting |

---

## Repository layout

```
falkor-chat/
├── docs/
│   ├── DESIGN.md          # full blueprint (data model, Cypher, ops, §14 M1 app architecture)
│   ├── QUERIES.md         # canonical query library — verified against live instance
│   ├── BACKLOG.md         # forward-looking backlog (K-numbered items + parking lot)
│   ├── HISTORY.md         # dated change log
│   ├── requirements/      # product requirements (tico)
│   ├── plans/             # ACTIVE plans, method notes, coordination logs (architect/ds/teco)
│   ├── reviews/           # plan/methodology reviews (analyst/ds)
│   ├── test-plans/        # ACTIVE test plans (qa-engineer)
│   ├── test-reports/      # ACTIVE test reports (qa-engineer)
│   └── archive/           # frozen docs of closed milestones (same subdir names)
├── scripts/
│   ├── bootstrap_schema.sh  # create indexes + constraints for any workspace
│   ├── start_falkordb.sh    # spin up FalkorDB in Docker
│   ├── start_server.sh      # one-shot: FalkorDB + venv + bootstrap@1024 + seed + uvicorn
│   ├── seed_demo.sh         # register the AI agent + a demo channel/thread (idempotent)
│   ├── test_queries.sh      # end-to-end query test suite
│   ├── load_test.sh         # append-path load harness + hot-read PROFILE (M1 DoD)
│   └── backfill_thread_ids.sh # one-off: stamp Message.threadId on pre-K-007 messages
├── server/                  # app: FastAPI REST + MCP + AI responder on one process
│   ├── falkorchat/{config,db,repository,services,schemas,api,mcp,app}.py
│   ├── falkorchat/{embedding,llm,responder}.py   # M2: embed worker, LLM client, agent responder
│   ├── tests/               # pytest — repository/services (live), MCP, REST, responder, app-mount
│   └── pyproject.toml       # fastapi, uvicorn, falkordb, mcp, pytest, httpx
├── web/                     # minimal browser client (index.html + app.js) served by app.py
├── Dockerfile               # server image (uvicorn, non-root, /health healthcheck)
├── compose.yaml             # FalkorDB + server dev stack (falkordb-data volume, external)
└── README.md
```

---

## Run the server (REST + MCP + AI agent)

The server hosts the browser REST API, the MCP agent front door, and the M2 AI responder on one
process. For the fast path use `./scripts/start_server.sh` (see **Quick start** above). The manual
steps, for when you want to run pieces yourself:

```bash
cd server
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'          # first time
EMBEDDING_DIM=1024 ./../scripts/bootstrap_schema.sh acme            # schema for the default tenant (ws:acme), M2 dim
FALKORCHAT_EMBEDDING_DIM=1024 .venv/bin/uvicorn falkorchat.app:app  # web UI + REST under /, MCP at /mcp
```

Then open **http://localhost:8000/** for the minimal web client (channels · threads · messages ·
search). It talks to the REST API on the same origin (no CORS). The browser and MCP front doors
share one `services.py`.

The AI responder is **off by default** so imports stay network-free; set `FALKORCHAT_ENABLE_AGENT=1`
(and have LM Studio up) to wire the live embedder + LLM + `@mention` responder — or just use
`start_server.sh`, which enables it and seeds the demo agent. `FALKORCHAT_EMBEDDING_DIM` **must**
match the workspace's vector index (1024 for `ws:acme`).

Run the server test suite (needs FalkorDB up; uses an isolated `ws:test` graph; offline — no LM Studio):

```bash
cd server && .venv/bin/python -m pytest -q      # needs FalkorDB up
```

Agents connect to MCP at `http://localhost:8000/mcp` (`type: streamable-http`; the trailing-slash
spelling `/mcp/` works too). The endpoint is unauthenticated in M1 — bind to localhost / a trusted
network only. Tools: `send_message`, `read_messages`, `create_thread`, `create_channel`,
`list_channels`, `list_threads`, `search_messages` (see [DESIGN.md §15](docs/DESIGN.md#15-mcp-transport-k-002--the-agent-front-door)).
