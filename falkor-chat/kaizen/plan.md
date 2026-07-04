# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-04 (M2 kickoff — embedding stack locked; K-009 delivered ✅;
> K-007 pending relaunch; K-008 blocked on K-007)

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

## ⏸ Session resume state (2026-07-04 — K-009 committed; next: K-007 relaunch)

- **K-009 delivered, verified, and committed** (see history.md entry; the interrupted session's
  earlier checkpoint claimed the commit prematurely — the resume pass re-verified everything and
  made the actual commit). Includes the **critical `falkordb-data` persistence fix**: the volume
  was mounted at `/data` but the image writes to `/var/lib/falkordb/data`, so **no graph data
  ever survived a container stop until today**. Fixed in both `scripts/start_falkordb.sh` and the
  new `compose.yaml`; persistence live-proven twice (post-fix bootstrap, and `ws:acme` present via
  `GRAPH.LIST` after a full stop/start on resume). `ws:acme` schema re-bootstrapped (12 indexes).
  Resume verification: pins install cleanly, ruff clean, server suite 75 passed, query suite
  92/92. CI workflow exists but **has never run** — first push to GitHub will tell. Not pushed.
  FalkorDB left **running** (`falkordb-dev`) for the K-007 relaunch; `ws:k007scratch` residue from
  the aborted run is still present.
- **K-007 graph-dba run was aborted** — coordination error: devops swapped the FalkorDB
  container mid-verification (fixing the persistence bug), pulling the scratch graph out from
  under graph-dba. **No deliverable produced** — `docs/plans/m2-groundwork-queries.md` does not
  exist yet. **Next action on resume: relaunch graph-dba with the K-007 brief** (below) on the
  now-stable environment, serialized — nothing else touches FalkorDB during its run.
- K-007 brief essentials for the relaunch: design + live-verify (scratch graph `ws:k007scratch`,
  delete after) the six items in the K-007 scope below; deliver
  `docs/plans/m2-groundwork-queries.md`; respect all locked decisions (two write paths / no
  conditional MERGE, single-`GRAPH.QUERY` atomicity, label-specific member resolution, MERGE ⇒
  uniqueness constraint, param'd Cypher, RAM rule 6); flag — never silently violate — any
  locked-decision conflict. Then: **architect** (`docs/plans/m2-groundwork.md`) → **tdd-engineer**.

## Active

### K-007 — M2 groundwork: concurrent-writer correctness + read completeness (🟡 pending relaunch)

Prerequisites for letting AI agents write. Owner chain: **graph-dba** (query/schema design,
live-verified) → **architect** (`docs/plans/m2-groundwork.md`) → **tdd-engineer** (implement) →
suites green (server pytest 75-baseline, `./scripts/test_queries.sh` 92/92-baseline).

- **Agent authorship write path** — §4 author MATCH is `{userId:…}`-only and services hardcodes
  `role="user"`; needs label-specific author resolution (member-resolution rule) + role from
  actor type so an `Agent` can author messages.
- **Write retry idempotency + first-post race** — same-`msgId` retry duplicates NEXT/TAIL edges
  (live-verified); two concurrent first posts → two HEADs. Guard/dispatch must respect the locked
  "two write paths, never a conditional MERGE" and single-`GRAPH.QUERY` atomicity decisions.
- **`threadId` in room-wide reads (§9.2)** — likely denormalised `threadId` on `Message`
  (schema change; RAM rule 6 costing required; QA addendum A2).
- **Millisecond `createdAt` ties** — unstable order + cursor page-boundary skip; becomes real
  with concurrent agent writers.
- **`TIMEOUT 1000ms` review** — before GraphRAG queries land (DESIGN §12 note).
- **Per-workspace RAM line re-costed at 1024 dims.**
- Fold in if touched: `db.connect()` import-time config bind; misleading `MERGE` on
  freshly-minted uuids in `create_channel`/`create_thread`.

### K-008 — GraphRAG proper (🔵 proposed — blocked on K-007)

DESIGN §12 M2 core, on the locked stack above:

- Embedding worker calling LM Studio `/v1/embeddings` (Qwen3-Embedding-0.6B, 1024d);
  `Message.embedding` inline `vecf32`.
- Vector index at 1024 dims; hybrid retrieval query (DESIGN §8 / QUERIES.md §6).
- AI `Agent` participant (Qwen3-4B-Instruct-2507 via `/v1/chat/completions`) posting answers
  with `EMITTED` provenance — depends on K-007 agent authorship.
- Web-client staleness story: polling/since-reads (adopt the K-006 `?since=&limit=` window),
  `isMention` rendering, clickable search results (needs K-007 `threadId` denorm), replace
  `alert()` errors.

## Parking lot / ideas

- Verify the K-009 GitHub Action goes green on first push (path-filtered
  `.github/workflows/falkor-chat.yml`; FalkorDB service container).
- `db.connect()` binds `config.FALKORDB_*` at import time (test-seam nit — fold into K-007).
- `MERGE` on freshly-generated uuids in `create_channel`/`create_thread` (fold into K-007).
- DESIGN §13 remaining open questions — resolve as their milestones arrive: workflow guard
  expression language (M3), `identity` source of truth + real auth (replaces the M1
  hardcoded-tenant seam, §14.3), message/embedding retention, cross-workspace analytics,
  Bolt vs RESP for the gateway.
