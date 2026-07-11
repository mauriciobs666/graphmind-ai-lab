# Test Report — K-015 M2 GraphRAG loop (acceptance / DoD gate)

> **Author:** qa-engineer · **Date:** 2026-07-08
> **Plan:** [`../test-plans/m2-graphrag.md`](../test-plans/m2-graphrag.md)
> **Build under test:** working tree at HEAD `62c7222` (untracked/modified M2 server modules
> present: `embedding.py`, `responder.py`, `llm.py`, `services.py`, `repository.py`, `test_*`).
> FalkorDB `falkordb-dev` @ `:6379` (`falkordb/falkordb:edge`, Redis 8.2.2, module `999999`, up
> since 20:56Z). LM Studio @ `:1234` (embed `text-embedding-qwen3-embedding-0.6b` 1024-dim, chat
> `qwen/qwen3-4b-2507`). Live app via `./scripts/start_server.sh` on `:8000` (agent loop on, dim
> 1024, tenant `ws:acme`). Isolated fixture: `ws:qa` @1024 (deleted after).

## 1. Summary

**Verdict: PASS. M2 GraphRAG meets its Definition of Done.** No defects.

Every DoD claim was verified from the outside in against the running system, on green baselines
(pytest **156 passed**, query suite **149/149**):

- **Out-of-band embeddings @1024:** `ws:acme`'s vector index is confirmed 1024-dim (a 4-dim probe
  errors `expected 1024 but got 4`) and the served app runs at `FALKORCHAT_EMBEDDING_DIM=1024`. A
  posted message is readable in the same tick; its embedding lands ~1s later via the background
  worker and the message becomes ANN-retrievable. All 5 human-posted seed messages carried
  embeddings — the corpus grows.
- **Hybrid retrieval ranks correctly:** a cooking query over a mixed cooking/space/biology corpus
  returned the two cooking messages first (cosine distance 0.368, 0.484) ahead of space (0.742,
  0.771) and biology (0.786), scores strictly **ascending**; the `MENTIONS→Entity` expansion
  returned `[]` on every row with the Entity graph empty (0 nodes) — no error.
- **AI agent participant:** `@assistant` in the seeded thread produced an Agent-authored answer in
  ~4s that reads `role:"assistant"` / `authorType:["Agent"]` on **all** read surfaces (full-thread
  read, `?since=0`, `GET /messages/{id}`, `/search`, MCP `read_messages`) — the K-007 authorship
  invariant holds. The answer carried a queryable **`EMITTED`** provenance edge to each retrieved
  seed with `score` (cosine distance) + 0-based contiguous `rank` in ascending-score order. The
  **loop guard** held (exactly one Agent message, stable over an 8s watch — no self-triggered
  runaway) and **failure isolation** is proven by dedicated unit tests (LLM/embedder failure ⇒
  nothing posted; no torn thread).
- **Web rendering seam:** the exact endpoints the UI consumes return the shapes it needs — a
  reader `@mention` flips `isMention` on precisely the mentioned row; since-read rows carry
  `displayName`; channel/thread/message navigation lists are correct.
- **MCP path:** all 7 tools are discoverable and post/read/search round-trip the same service
  layer, showing the agent answer with `role:"assistant"`.

The four documented carry-overs (workspace-wide channel scope, `ensure_agent` no displayName,
unsurfaced reverse provenance, deferred transport agent-auth) were confirmed as **documented
design state, not defects** — see §4. One is directly observable: the trigger message appears as
its own rank-0 provenance seed (score ≈ 0), exactly as predicted.

## 2. Results

| ID | Item | Result | Evidence |
|---|---|---|---|
| TP-000 | Baselines green | ✅ PASS | `pytest -q` → **156 passed** (4.66s); `./scripts/test_queries.sh` → **149/149 passed** |
| TP-001 | App is actually at dim 1024 | ✅ PASS | `ws:acme` vector probe with 4-dim vec → `Vector dimension mismatch, expected 1024 but got 4`; boot log `Dim: 1024`; `config.EMBEDDING_DIM=1024`, `ENABLE_AGENT` on |
| TP-002 | Message readable immediately (pre-embedding) | ✅ PASS | `POST /threads/demo-welcome/messages` → 201 `role:"user"`; immediate full-thread `GET` returns it (`displayName:"Demo User"`, `authorType:["User"]`) |
| TP-003 | Embedding lands out-of-band; corpus grows | ✅ PASS | `m.embedding IS NOT NULL` true after ~1s; `db.idx.vector.queryNodes` on its own vector returns the msgId |
| TP-004 | Every posted message embedded | ✅ PASS | after seeding: `count(Message)=5`, `embedded=5` |
| TP-005 | Hybrid retrieval ranks by cosine distance ASC | ✅ PASS | cooking query → pasta `0.36784`, pizza `0.48435`, JWST `0.74155`, rockets `0.77140`, mitochondria `0.78618`; ascending, cooking cluster top-2 |
| TP-006 | Entity expansion no-ops to `[]` | ✅ PASS | every hybrid_search row `relatedContext == []`; `MATCH (e:Entity) RETURN count(e)` = **0**; no error |
| TP-007 | Workspace-wide variant (`channel_id=None`) | ✅ PASS | responder's real path (`channel_id=None`) returned 5 ranked rows, no error |
| TP-010 | `@assistant` triggers an agent answer | ✅ PASS | trigger `0f3cf47…` → Agent-authored answer `b3ffc55…` (`role:"assistant"`) in ~4s; trigger readable immediately |
| TP-011 | Agent answer `role:"assistant"` on ALL surfaces | ✅ PASS | full-thread + `?since=0` + `GET /messages/{id}` + MCP `read_messages` all show `role:"assistant"`, `authorType:["Agent"]`, `displayName:"Assistant"`; `/search` finds it. Answer grounded in the pasta seed |
| TP-012 | `EMITTED` provenance written + queryable | ✅ PASS | 5 `EMITTED` edges via raw read and `repository.read_provenance`; each carries `score` + 0-based contiguous `rank`, scores ascending (`0.0, 0.327, 0.478, 0.657, 0.712`) |
| TP-013 | Provenance atomic; no torn write | ✅ PASS | thread chain walk = 7 = total messages; full-thread read = 7 rows chronological (6 user + 1 assistant), no dup (NEXT doesn't follow EMITTED); `count(EMITTED)=5`, only on the answer |
| TP-014 | Loop guard: assistant msg doesn't trigger | ✅ PASS | exactly **1** Agent-authored message, unchanged after an 8s watch — no runaway |
| TP-015 | Failure isolation (LLM/embedder fail ⇒ no post) | ✅ PASS | `tests/test_responder.py` **7 passed** incl. `test_llm_failure_posts_nothing` / `test_embedder_failure_posts_nothing` (`post_calls == []` — no torn thread) + loop-guard + self-embed-no-retrigger |
| TP-020 | Reader `@mention` flips `isMention` | ✅ PASS | `mentions:["u1"]` post → `?since=0` as `u1`: exactly that row `isMention=true`, all others false |
| TP-021 | Since-read rows carry `displayName` | ✅ PASS | all `?since=0` rows carry `displayName` (`"Demo User"`, `"Assistant"`) |
| TP-022 | UI navigation endpoints | ✅ PASS | `GET /channels` → `demo-general`; `/channels/demo-general/threads` → `demo-welcome`; `/threads/{id}/messages` correct shapes |
| TP-023 | `/search` finds the agent answer | ✅ PASS | `/search` returns the answer msgId with its `threadId` (also `search_messages('pasta')` → 3 hits) |
| TP-030 | MCP tools drive the same service layer | ✅ PASS | `list_tools` → all 7 tools; MCP `read_messages` shows the `assistant` answer; `search_messages('pasta')`=3; `send_message` posts `role:"user"` as `u1` |
| TP-040 | Isolated repo-level checks on `ws:qa` @1024 | ✅ PASS | wrong-dim (4) `set_embedding` raises `EmbeddingDimensionError`; correct 1024-dim commits + ANN self-retrieval at score 0, `relatedContext=[]` |

## 3. Defects

**None.** No behavior diverged from the acceptance criteria; every P1 item passed with observed
evidence.

## 4. Coverage, carry-overs & residual risk

### Covered
The full M2 loop end-to-end through the running app: REST post → out-of-band embed → 1024-dim ANN
retrieval → LLM answer → Agent-authored post with `EMITTED` provenance → read-back on every
surface (REST full/since/point/search + MCP). Ranking correctness on a semantically-labelled
corpus, the Entity dormant path, the loop guard, failure isolation, the reader-mention flag, the
`displayName` carry, and the wrong-dim rejection at true 1024.

### Documented carry-overs (confirmed as design state, NOT defects)
- **Channel scoping is workspace-wide for the wired responder.** `api._safe_respond` passes
  `channel_id=None` (a thread→channel read isn't in `QUERIES.md` yet). **Observed consequence:**
  the trigger message embeds before retrieval runs, so it surfaced as its own **rank-0 provenance
  seed** (`score` ≈ `-1.19e-07`, i.e. ~0 / identical). This is exactly the predicted behavior.
  *(Minor observation, not a bug: the identical-vector distance came back as a tiny negative float
  `-1.19e-07` rather than exactly 0 — cosine-distance floating-point noise on this build; harmless
  for ASC ranking. The shell suite asserts `= 0` at dim 4, where it holds exactly.)*
- **`repository.ensure_agent` does not persist `displayName`.** Agents registered purely via
  `ensure_agent` would show their id. In this run the agent was seeded via `seed_demo.sh`'s guarded
  create, so it correctly showed `displayName:"Assistant"`.
- **`read_citing_answers` (reverse provenance) is not surfaced in any API/UI route.** The
  `EMITTED` edge and the repository method support it; no endpoint exercises it. Forward provenance
  (`read_provenance`) is likewise not on a public route yet — it was verified at the repository
  seam.
- **Transport-level externally-authenticated agent actor is deferred (M2.5 / K-017).** There is no
  per-client agent auth; the AI participant is a **server-side responder** wired in `app.py` and
  gated by `FALKORCHAT_ENABLE_AGENT`. Both front doors attribute every call to the single
  configured actor via `config.get_context` (MCP ignores any client-supplied `from`). **You cannot
  drive an externally-authenticated agent over the transport** — this is the K-007 QA carry-over,
  carried forward unchanged. The agent authors first-class *internally* (role derived `assistant`),
  which is the M2 scope.

### Residual risk (low)
- **Retrieval quality is model-dependent.** Ranking was correct on a clean, well-separated corpus;
  real-world semantic overlap and larger corpora aren't exercised here. ANN recall is approximate
  (K-008 item 4) — "returns exactly k" is not an invariant, and was not treated as one.
- **RAM (repo rule 6, qualitative).** No re-measurement; the K-008/§11 budget stands: ~12.5 KB per
  message at 1024 dims (dominant line = the HNSW `Message.embedding` index), `EMITTED` a rounding
  error on top (a few edges only on answer messages). `ws:acme` at this small scale is far under
  budget. Size vector-heavy workspaces from `INFO memory` deltas, not `GRAPH.MEMORY USAGE`.
- **LLM/embedder availability.** The loop hard-depends on LM Studio; failure is isolated (nothing
  posted, background task logs+swallows) but the agent simply doesn't answer when the backend is
  down — acceptable and by design.

## 5. Feedback & recommendations (non-blocking)

- **Surface provenance on a route.** `read_provenance` / `read_citing_answers` are verified at the
  repository seam but invisible to any client. A `GET /messages/{id}/provenance` (forward) would
  let the UI show "why the AI said this" — the data is already there. (Follow-up, not M2 scope.)
- **Close the workspace-wide-scope carry-over when the thread→channel read lands** so the responder
  can pass a real `channel_id` and the trigger stops self-citing at rank 0. Track with the
  QUERIES.md thread→channel addition.
- **Consider persisting `displayName` in `ensure_agent`** for parity with the seed script, so
  agents created at runtime render with a name rather than their id.
- **`identical vector → 0` on this build** returned `-1.19e-07` at dim 1024. Worth a one-line note
  in `claude/graph-dba/falkordb-quirks.md` if any consumer ever asserts `score == 0` at 1024 dims
  (ASC ordering is unaffected).

## 6. Environment hygiene

- Transient `ws:qa` created at `EMBEDDING_DIM=1024` for TP-040 and **deleted** (`GRAPH.DELETE
  ws:qa` → `OK`). Final `GRAPH.LIST` = `ws:acme, reference, ws:test`.
- `reference` and `ws:test` are throwaway fixtures the baseline suites wipe at both start and end
  (`test_queries.sh` lines 7/71-72/674-675). The mandated TP-000 baseline deleted them at teardown
  **by its own design**; acceptance activity never touched them, and both were **rebuilt** via
  `bootstrap_schema.sh test` at the end so the graph list matches the pre-session state.
- `ws:acme` node data was wiped (`DETACH DELETE`, schema/indexes preserved) and reseeded via
  `start_server.sh`/`seed_demo.sh` for a clean acceptance run — permitted (non-precious demo
  tenant). It now holds the demo seed plus this run's acceptance messages.
- The live server (`uvicorn falkorchat.app:app --port 8000`, pid 28019) was left running for the
  reporter; stop with `docker`-independent `kill 28019` or Ctrl+C in its terminal (FalkorDB keeps
  running).
