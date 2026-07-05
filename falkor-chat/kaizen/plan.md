# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-05 (K-007 delivered ✅ — M2 groundwork landed, baselines now
> pytest 98 / query suite 115/115; K-008 unblocked and is the next action)

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

## Parking lot / ideas

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
