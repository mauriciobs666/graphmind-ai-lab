# Test Plan — K-015 M2 GraphRAG loop (acceptance / DoD gate)

> **Author:** qa-engineer · **Date:** 2026-07-08 · **Status:** v1 (pre-execution)
> **Feature under test:** M2 GraphRAG loop — out-of-band embeddings + 1024-dim vector index +
> hybrid retrieval + AI agent participant + `EMITTED` provenance. Built across K-008 (retrieval
> core), K-013 (agent participant + provenance), K-014 (web rendering).
> **References:** `docs/plans/m2-graphrag.md` (K-008) · `docs/plans/m2-agent-participant.md` (K-013)
> · `docs/QUERIES.md` §6 (hybrid retrieval), §10 (EMITTED provenance) · `docs/DESIGN.md` §9/§10/§11
> · `server/falkorchat/{embedding,responder,services,repository,api,mcp,app,config}.py`
> **Prior passes:** `k007-m2-groundwork-report.md` (v2 write paths, agent authorship, cursors) —
> this gate builds on that green foundation and closes M2.

## 1. Scope & objective

Verify, **from the outside in on the running system**, that the M2 GraphRAG loop meets its
Definition of Done and is safe to flip **M2 → done**. The DoD, restated as concrete acceptance
criteria:

1. **Embeddings are out-of-band.** Every posted message is readable immediately; its
   `Message.embedding` (1024-dim) lands shortly after via the async `EmbeddingWorker` → LM Studio,
   growing the retrievable corpus. The served app runs at dim **1024** (matches `ws:acme`'s vector
   index) so embeddings are ANN-retrievable, not silently dropped.
2. **Hybrid retrieval ranks correctly.** A semantically-related query returns the expected seed
   messages ranked by cosine distance **ASC** (most similar first); the `MENTIONS→Entity`
   expansion no-ops to `[]` cleanly (no Entity nodes in M2).
3. **AI agent participant works end-to-end.** `@assistant <question>` in a channel triggers the
   server-side `AgentResponder`: embed trigger → hybrid retrieve → LLM → post the answer **as the
   Agent** with `role:"assistant"` (derived, not trusted) on **all** read surfaces, plus a
   queryable **`EMITTED`** provenance edge (with `score`/`rank`) to the retrieved seeds. The loop
   guard holds (assistant-authored message never triggers a response); LLM/embedder failure ⇒
   nothing posted (no torn thread).
4. **Web rendering.** Assistant replies are distinguishable (role/`assistant`) and a reader
   `@mention` flips `isMention`, verified through the exact endpoints the UI consumes
   (`/channels`, `/channels/{id}/threads`, `/threads/{id}/messages`, `?since=`, `/search`);
   since-read rows carry `displayName`.
5. **MCP path.** The agent-facing MCP tools (`send_message`/`read_messages`/`search_messages`/…)
   drive the same service layer and behave identically.

**Verdict target:** PASS / PASS-with-parked-defects / FAIL, with reproducible evidence per item.

## 2. Environment & data setup

- **FalkorDB:** container `falkordb-dev`, `localhost:6379`, `falkordb/falkordb:edge` (Redis 8.2.2,
  module `999999`).
- **Served tenant:** `ws:acme`, vector index confirmed at **`EMBEDDING_DIM=1024`** (probe: a 4-dim
  query vector errors `expected 1024 but got 4`). Non-precious demo tenant — node data may be
  wiped/reseeded for a clean run; **schema/indexes preserved** (`DETACH DELETE`, not `GRAPH.DELETE`).
- **LM Studio:** `http://localhost:1234/v1` — embedding `text-embedding-qwen3-embedding-0.6b`
  (1024-dim), chat `qwen/qwen3-4b-2507`. Live.
- **App under test:** `./scripts/start_server.sh` (exports `FALKORCHAT_ENABLE_AGENT=1`,
  `FALKORCHAT_EMBEDDING_DIM=1024`; runs `seed_demo.sh` → registers agent `assistant` + channel
  `demo-general` / thread `demo-welcome` + `MEMBER_OF` edges; uvicorn on `:8000`, REST at `/`, MCP
  at `/mcp`, web UI at `/`).
- **Isolated fixture (optional):** transient `ws:qa` at `EMBEDDING_DIM=1024` for direct
  repository/engine-level checks; **deleted at end** (`GRAPH.DELETE ws:qa`). `reference` and
  `ws:test` left to their suite-managed lifecycle (both are throwaway fixtures the baseline
  suites wipe/rebuild; not touched by acceptance activity).
- **Actor:** `u1` (hardcoded M1 tenant); agent `assistant`.

## 3. Risk assessment (drives prioritization)

| # | Risk | Likelihood | Impact | Priority |
|---|---|---|---|---|
| R1 | Wrong-dim embedding silently drops a message out of ANN (K-008 item-2 quirk) — app not actually at 1024 | Low (guarded 3×) | High (silent corpus loss) | **P1** |
| R2 | Agent answer reads a wrong `role` on some surface (K-007 invariant regression) | Low | High (breaks UI trust + contract) | **P1** |
| R3 | Retrieval ranks wrong / Entity expansion errors instead of `[]` | Low | High (bad grounding) | **P1** |
| R4 | `EMITTED` provenance missing / wrong score-rank / torn on failure | Med | High (provenance is a DoD claim) | **P1** |
| R5 | Loop guard fails — agent answers its own answer (runaway) | Low | High (cost/spam runaway) | **P1** |
| R6 | Embedding not truly out-of-band — post blocks on / fails with embedder | Low | Med | P2 |
| R7 | Web endpoints don't carry `isMention`/`displayName`/role for the UI | Low | Med | P2 |
| R8 | MCP path diverges from REST behavior | Low | Med | P2 |
| R9 | Failure isolation: LLM/embedder failure leaves a torn thread | Low | High | P2 |

## 4. Test items

Priority: **P1** = DoD-blocking; **P2** = important; **P3** = documentation/edge.
Type: F=functional, I=integration, C=contract, E=e2e, X=exploratory, N=non-functional.

| ID | Title | Type | Pri | Preconditions | Steps | Expected |
|---|---|---|---|---|---|---|
| TP-000 | Baselines green | I | P1 | venv built, FalkorDB up | `pytest -q`; `./scripts/test_queries.sh` | pytest **156 passed**; shell **149/149** |
| TP-001 | App is actually at dim 1024 | C | P1 | server up | Probe `ws:acme` vector index with a 4-dim vector; confirm `FALKORCHAT_EMBEDDING_DIM=1024` in server env | `expected 1024 but got 4`; app config dim = 1024 |
| TP-002 | Message readable immediately (pre-embedding) | F/E | P1 | server up, thread exists | `POST /threads/{id}/messages`; immediately `GET /threads/{id}/messages` | 201; message present in read within the same tick, before its embedding lands |
| TP-003 | Embedding lands out-of-band; corpus grows | I/E | P1 | TP-002 posted | Poll `Message.embedding IS NOT NULL` for the msgId (direct read); confirm it becomes ANN-retrievable | embedding non-null shortly after; msg returned by `hybrid_search` on its own vector |
| TP-004 | Every posted message is embedded (corpus growth) | I | P1 | several msgs posted | Count `Message` vs `Message WHERE embedding IS NOT NULL` after settle | all human-posted messages carry a 1024-dim embedding |
| TP-005 | Hybrid retrieval ranks by cosine distance ASC | F | P1 | known corpus seeded + embedded | `services.hybrid_search` (or via responder retrieval) with a query near one seed cluster | expected seed(s) returned first; scores ascending; identical/near vector ranks top |
| TP-006 | Entity expansion no-ops to `[]` | F/C | P1 | corpus seeded | Inspect `relatedContext` on every hybrid_search row; `MATCH (e:Entity) RETURN count(e)` | every `relatedContext == []`; Entity count = 0; no error |
| TP-007 | Workspace-wide variant (`channel_id=None`) works | F | P2 | corpus seeded | hybrid_search with no channel scope (responder's real path) | rows returned, ranked ASC, no error |
| TP-010 | `@assistant` triggers an agent answer | E | P1 | server up, agent seeded | `POST` a message mentioning `assistant` into `demo-welcome`; poll thread | a new message authored by the Agent appears; original readable immediately |
| TP-011 | Agent answer reads `role:"assistant"` on ALL surfaces | C | P1 | TP-010 produced an answer | Read the answer via full-thread read, `?since=0`, `/search`, `GET /messages/{id}`, MCP `read_messages` | `role:"assistant"` + Agent authorship everywhere (K-007 invariant) |
| TP-012 | `EMITTED` provenance is written + queryable | I | P1 | TP-010 answer with seeds | Read `(answer)-[:EMITTED]->(seed)` with `score`/`rank`; use `repository.read_provenance` | ≥1 EMITTED edge; each carries `score` (cosine distance) + 0-based `rank`; ordered |
| TP-013 | Provenance rides atomically (no torn write) | I | P1 | TP-012 | Confirm answer message + its EMITTED edges committed together; thread read integrity (no dup, NEXT doesn't follow EMITTED) | answer present with provenance; thread chain intact |
| TP-014 | Loop guard: assistant message doesn't trigger a response | E | P1 | server up | Observe that the agent's own answer (role assistant, may `@`-mention) does not spawn another answer; assert no unbounded growth | exactly one answer per human trigger; no runaway |
| TP-015 | Failure isolation (LLM/embedder failure ⇒ nothing posted) | I | P2 | responder unit + logs | Assert from `test_responder`/logs that a raise before the write posts nothing (no torn thread); note if reachable black-box | failure short-circuits before the guarded write |
| TP-020 | Reader `@mention` flips `isMention` | F/C | P2 | server up | Post a message mentioning `u1`; read via `?since=0` as `u1` | that row `isMention=true`; others false |
| TP-021 | Since-read rows carry `displayName` | C | P2 | messages exist | `GET /threads/{id}/messages?since=0` | each row has a `displayName` field (agent/user) |
| TP-022 | UI navigation endpoints | C | P2 | seeded | `GET /channels`, `GET /channels/{id}/threads`, `GET /threads/{id}/messages` | channel/thread/message lists shape correct for the web client |
| TP-023 | `/search` finds the agent answer | F | P2 | answer exists | `GET /search?q=<token from answer>` | answer msg found with `threadId` |
| TP-030 | MCP tools drive the same service layer | I/C | P2 | server configured | In-memory `mcp.list_tools()` + `call_tool(send_message/read_messages/search_messages)` against `ws:acme` (or `ws:test`) | tools listed; post/read/search behave identically to REST |
| TP-040 | Isolated repo-level checks on `ws:qa` @1024 | I | P3 | `ws:qa` bootstrapped | Direct `set_embedding` wrong-dim rejection; ANN round-trip at true 1024 | wrong-dim raises `EmbeddingDimensionError`; correct dim retrievable |

## 5. Entry / exit criteria

**Entry:** FalkorDB up; LM Studio serving both models; `server/.venv` built; baselines (TP-000)
green.

**Exit (M2 → done) — all must hold:**
- TP-000 through TP-014 (all P1) **PASS**.
- P2 items PASS or downgraded to explicitly-parked, documented defects with severity.
- Every reported defect is reproducible from the steps recorded.
- `ws:qa` deleted; `reference`/`ws:test` left to their suite lifecycle.

## 6. Explicitly out of scope (documented carry-overs, NOT defects)

These are known M2-green design choices / deferrals. Documented, not filed as bugs unless found
actually broken:

- **Channel scoping is workspace-wide for the wired responder** — `api._safe_respond` passes
  `channel_id=None` (a thread→channel read isn't in QUERIES.md yet). Consequence: the trigger
  message embeds before retrieval runs, so it can surface as its own rank-0 provenance seed —
  **expected**, not a defect.
- **`repository.ensure_agent` doesn't persist `displayName`** (only `seed_demo.sh`'s guarded
  create does) — agents registered purely via `ensure_agent` show their id as displayName.
- **`read_citing_answers` (reverse provenance) is not surfaced in any API/UI route** yet — the
  edge supports it; no endpoint exercises it.
- **Transport-level externally-authenticated agent actor is deferred (M2.5 / K-017)** — there is
  no per-client agent auth; the agent is a **server-side responder**. You cannot drive an
  externally-authenticated agent over the transport. MCP/REST attribute to the single configured
  actor (`config.get_context`). This is the K-007 QA carry-over, carried forward.
- **Non-functional depth** (load/latency/RAM profiling of the retrieval path) is out of scope for
  this acceptance gate — RAM sizing was closed in K-008 item 7 / DESIGN §11 (~12.5 KB/msg @1024).
  A qualitative RAM note is included, not a re-measurement.
