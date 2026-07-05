# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-05 (K-007 delivered ✅ — M2 groundwork landed, baselines now
> pytest 98 / query suite 115/115; K-008 unblocked and is the next action. QA acceptance
> pass on K-007: PASS with two low-severity defects — QA DEF-1 parked as a K-008
> prerequisite below, QA DEF-2 in the parking lot)

## Locked M2 stack decisions (2026-07-04, user-approved)

| Axis | Decision | Rationale |
|---|---|---|
| Embedding model | **Qwen3-Embedding-0.6B** (GGUF, Q8_0) | Best small-model MTEB quality; 100+ languages (PT-BR + EN chat); ~0.6 GB resident |
| **`EMBEDDING_DIM=1024`** | Native dim; MRL allows 512/256 truncation later | ~4 KB/message vector (`vecf32`), ~8 KB with HNSW overhead — RAM rule 6 line |
| Agent LLM | **Qwen3-4B-Instruct-2507** Q4_K_M (non-thinking) | RAG answering, not CoT; low latency; KV-cache headroom; `-Thinking-2507` is a drop-in swap if M3 needs it |
| Runtime | **LM Studio** on Windows host, OpenAI-compatible `/v1/embeddings` + `/v1/chat/completions`, reached from WSL2 (mirrored networking → localhost) | Existing severino path; zero new moving parts. Ollama is the fallback if headless/always-on friction bites |
| VRAM budget | 6 GB dedicated (RTX 4050); embedder + 4B LLM co-resident | Do not plan around shared-RAM spill |
| Upgrade path | `qwen3-embedding:4b` — same family, same 1024-dim MRL | Re-embed only; no schema change |

> `bootstrap_schema.sh` default is `EMBEDDING_DIM=1536` — **must** be run with `EMBEDDING_DIM=1024`
> for any new workspace from K-008 on.

## Active

### K-008 — GraphRAG proper (🔵 proposed — **unblocked**, K-007 delivered 2026-07-05)

DESIGN §12 M2 core, on the locked stack above:

- Embedding worker calling LM Studio `/v1/embeddings` (Qwen3-Embedding-0.6B, 1024d);
  `Message.embedding` inline `vecf32`.
- Vector index at 1024 dims; hybrid retrieval query (DESIGN §8 / QUERIES.md §6).
- AI `Agent` participant (Qwen3-4B-Instruct-2507 via `/v1/chat/completions`) posting answers
  with `EMITTED` provenance — K-007 agent authorship is in (role derived from the author label).
- Web-client staleness story: polling/since-reads (adopt the K-006 `?since=&limit=` window),
  `isMention` rendering, clickable search results (K-007 `threadId` denorm is in), replace
  `alert()` errors.
- GraphRAG reads pass a per-query client `timeout=` override (service-layer constant, not
  per-call ad-hockery) — the K-007 TIMEOUT posture (DESIGN §10).
- **Prerequisite — QA DEF-1 (2026-07-05, low now / hazard once agents get identities):**
  add a cross-label member-id uniqueness guard before wiring real agent identities —
  `ensure_user`/`ensure_agent` should refuse an id already held by the other label (or lock a
  "member ids are namespace-unique across `User`/`Agent`" rule + validation). Today a
  configured actor colliding with an existing `agentId` silently MERGEs a shadow `User` that
  eclipses the Agent in **every** `coalesce(u, a)` lookup (role derivation, `POSTED_BY`,
  mentions) — silent misattribution, reproduced live. See
  `docs/test-reports/k007-m2-groundwork-report.md` §3.

## Parking lot / ideas

- **QA DEF-2 (2026-07-05, low — ops/diagnosability):** with FalkorDB unreachable,
  `uvicorn falkorchat.app:app` hangs silently at import (no log line, port never binds; ≥90s
  observed) — `db.connect()`'s `FalkorDB()` issues an eager command with no socket/connect
  timeout, and the module-level `app = create_app()` triggers it at import. Fix: pass
  `socket_connect_timeout`/`socket_timeout` in `db.connect()` and/or defer the first
  connection to lifespan with a clear startup error. Compose is shielded by
  `depends_on: service_healthy`; the README bare-`uvicorn` dev path is not. See
  `docs/test-reports/k007-m2-groundwork-report.md` §3.
- Verify the K-009 GitHub Action goes green on first push (path-filtered
  `.github/workflows/falkor-chat.yml`; FalkorDB service container). Note the CI baseline
  echoes in its comments (75/92) predate K-007's 98/115 — the suites themselves are the
  source of truth.
- File upstream FalkorDB issues (K-007 OQ6, recommended to the user): `GRAPH.MEMORY USAGE`
  under-reports vector-index memory; one-shot instant-timeout anomaly after a long override run.
- DESIGN §13 remaining open questions — resolve as their milestones arrive: workflow guard
  expression language (M3), `identity` source of truth + real auth (replaces the M1
  hardcoded-tenant seam, §14.3), message/embedding retention, cross-workspace analytics,
  Bolt vs RESP for the gateway.
