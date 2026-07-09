# Demo environment bring-up — reusable runbook

**Component:** `falkor-chat/` · **Owner altitude:** devops (execution) + qa-engineer (acceptance)
**Goal:** stand up the falkor-chat demo (web UI + REST + MCP + live AI responder) from scratch, verified green.
**Reuse:** this is a repeatable checklist — run it top to bottom any time you need the demo. Idempotent by design; safe to re-run.

> **Fast path:** `cd falkor-chat && ./scripts/start_server.sh` does stages 1–5 below in one shot.
> The rest of this doc is the reusable breakdown: prerequisites, what each stage does, verification, teardown, and troubleshooting — for when the one-shot fails or you need to run a stage by hand.

---

## 0. Prerequisites (once per machine — NOT installed by the scripts)

| Need | Check | Notes |
|---|---|---|
| Docker running | `docker info` | FalkorDB runs as `falkordb/falkordb:edge`, container `falkordb-dev`. |
| `redis-cli` on PATH | `redis-cli --version` | `start_server.sh` polls it for readiness. |
| Python 3 + venv | `python3 -m venv --help` | venv created under `server/.venv`. |
| LM Studio at `:1234` | `curl -s localhost:1234/v1/models` | **Only if the AI loop is on** (default). Serves embedder + LLM. |

**LM Studio / WSL note:** LM Studio runs on **Windows**; reach it from WSL2 via mirrored networking (`localhost`) or the gateway-IP fallback. If the AI loop isn't reachable, either fix networking or run without AI (`FALKORCHAT_ENABLE_AGENT=0` — UI/REST only).

**Models LM Studio must serve** (defaults, from `server/.env.example`):
- Embedder: `text-embedding-qwen3-embedding-0.6b` → **dim 1024** (must match the workspace index)
- LLM: `qwen/qwen3-4b-2507`

---

## 1. One-shot bring-up (default path)

```bash
cd falkor-chat
./scripts/start_server.sh          # Ctrl+C to stop the server; FalkorDB keeps running
```

Runs five stages, each idempotent:

| Stage | Action | Script |
|---|---|---|
| 1/5 | Start FalkorDB detached (skips if `falkordb-dev` already running); wait for `PONG` | `start_falkordb.sh -d` |
| 2/5 | Create `server/.venv` if absent; `pip install -e '.[dev]'` | (inline) |
| 3/5 | Bootstrap schema (indexes + constraints) for `ws:acme` at `EMBEDDING_DIM` | `bootstrap_schema.sh acme` |
| 4/5 | Seed AI agent + demo channel/thread + `MEMBER_OF` edges (fixed ids → MERGE) | `seed_demo.sh acme` |
| 5/5 | Export AI env vars, launch uvicorn on `:8000` (reload) | (inline) |

**Endpoints when up:**
- Web UI → http://localhost:8000/
- REST + `/search` → http://localhost:8000/
- MCP (Streamable-HTTP) → http://localhost:8000/mcp
- FalkorDB web console → http://localhost:3000

---

## 2. Manual equivalent (run a stage by hand)

From `falkor-chat/`:

```bash
# 1. FalkorDB (ports 6379 + 3000; data in named volume falkordb-data)
./scripts/start_falkordb.sh -d

# 2. venv + deps
cd server && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' && cd ..

# 3. schema — EMBEDDING_DIM MUST match the embedding model (1024 for Qwen3-Embedding-0.6B)
EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh acme

# 4. seed agent + demo channel/thread (idempotent — fixed ids + MERGE)
./scripts/seed_demo.sh acme

# 5. serve — export the AI env vars first (see server/.env.example), then:
cd server && \
  FALKORCHAT_EMBEDDING_DIM=1024 FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_AGENT_ID=assistant \
  .venv/bin/uvicorn falkorchat.app:app --reload
```

---

## 3. Configuration knobs (env vars)

| Var | Default | Purpose |
|---|---|---|
| `FALKORCHAT_WS_ID` | `acme` | workspace id → graph key `ws:<id>` |
| `FALKORCHAT_USER_ID` | `u1` | must match `config.USER_ID` |
| `FALKORDB_HOST` / `FALKORDB_PORT` | `127.0.0.1` / `6379` | FalkorDB connection |
| `EMBEDDING_DIM` | `1024` | **must match the embedding model AND the workspace's vector index** |
| `FALKORCHAT_ENABLE_AGENT` | `1` | `0` = serve UI/REST without the AI loop (no LM Studio needed) |
| `FALKORCHAT_AGENT_ID` / `_NAME` | `assistant` / `Assistant` | **must match the seeded agent** or `@mention` won't trigger |
| `UVICORN_ARGS` | `--reload` | passthrough to uvicorn |

Full reference: `server/.env.example`.

---

## 4. Two gotchas that break the demo silently

1. **Embedding-dim mismatch.** A wrong-dim `vecf32` write is *silently accepted* by FalkorDB, then drops out of the ANN index — retrieval quietly returns nothing. Keep `FALKORCHAT_EMBEDDING_DIM` == the model's dim (1024) == the dim `ws:acme` was bootstrapped at. The one-shot keeps these aligned; hand-runs must not.
2. **Agent-id mismatch.** `@mention`-ing only triggers a reply when the mentioned id equals `FALKORCHAT_AGENT_ID` **and** that agent was registered by `seed_demo.sh`. Change one, change both.

---

## 5. Verification (definition of done)

| Check | Command / action | Expected |
|---|---|---|
| Query suite green | `./scripts/test_queries.sh` | `149/149 passed` (verified 2026-07-09; AGENTS.md still cites the older 126 baseline) |
| Server tests green | `cd server && .venv/bin/python -m pytest -q` | `156 passed` (verified 2026-07-09; needs FalkorDB up) |
| UI loads | open http://localhost:8000/ | demo channel/thread visible |
| AI loop works | post a message `@assistant …` in the UI | agent replies (role `assistant`), grounded in retrieval |

For an acceptance-level pass (the `@mention` → grounded-reply loop end to end), route to **qa-engineer**.

---

## 6. Teardown

```bash
# stop the server: Ctrl+C in its terminal

# stop FalkorDB (data survives in the volume)
docker stop falkordb-dev

# full reset — DESTROYS all graph data (approval-gated for the devops agent)
docker rm -f falkordb-dev
docker volume rm falkordb-data
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FalkorDB did not respond after 30s` | Docker down / port 6379 taken | `docker info`; free the port or set `FALKORDB_PORT` |
| Agent never replies to `@mention` | AI loop off, LM Studio unreachable, or agent-id mismatch | check `FALKORCHAT_ENABLE_AGENT=1`, `curl localhost:1234/v1/models`, confirm `FALKORCHAT_AGENT_ID` matches seed |
| Replies ungrounded / retrieval empty | embedding-dim mismatch (§4.1) | rebuild workspace at correct dim; re-seed |
| `pip install` fails | no network / stale venv | recreate `server/.venv`, retry |
| Schema errors on bootstrap | wrong dialect assumptions | this is FalkorDB OpenCypher — no APOC/GDS; see `falkor-chat/AGENTS.md` |

---

## 8. Delegation map (if run as a coordinated task)

| Work | Owner |
|---|---|
| Execute bring-up, unblock env/Docker/venv/LM-Studio issues | **devops** (destructive ops approval-gated) |
| Acceptance pass on the `@mention` loop; write test report | **qa-engineer** |
| FalkorDB schema / dim / Cypher questions | **graph-dba** |
| App behavior changes uncovered during bring-up | **coder** / **tdd-engineer** (from an architect plan) |

**Prereq decision the user owns:** is LM Studio reachable from this box? If not → run AI-off (`FALKORCHAT_ENABLE_AGENT=0`) or fix WSL↔Windows networking first.
