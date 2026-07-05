# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-05 (K-010 delivered ✅ — QA DEF-1 and DEF-2 closed; baselines now
> pytest 110 / query suite 126/126. **Road-to-green planning pass added (architect): K-011..K-018
> sequence both M1 and M2 to ✅. Scope confirmed by user 2026-07-05 — "M2 green = functional
> GraphRAG"; auth + real-time deferred to the M2.5 hardening track. Doc-inconsistency sweep added
> K-019 (stale test counts in README/DESIGN; embedding model still listed §13-open though locked;
> §12/§14.1 scope wording).** See the milestone map below.)

## Milestone-to-green map (architect plan, 2026-07-05)

| Milestone | Reaches ✅ when | Items |
|---|---|---|
| **M1 — Chat core** | Its own un-finished DoD is closed: append-path load-tested, hot reads PROFILEd, request/response web UI de-staled | **K-011 + K-012** |
| **M2 — GraphRAG** | Functional GraphRAG loop: embeddings + vector index @1024 + hybrid retrieval + AI agent participant with `EMITTED` provenance, QA-accepted | **K-008 (re-scoped) + K-013 + K-014 + K-015** |
| **M2.5 — Hardening** *(deferred)* | Real auth, transport-level agent path, real-time push | **K-016 → K-017, K-018** |

> ✅ **Scope decision — CONFIRMED (user, 2026-07-05).** "M2 green" = **functional GraphRAG** (the
> narrow §12 roadmap DoD: embeddings + vector index + hybrid retrieval + agent participant +
> `EMITTED`). Real auth and real-time push are **deferred to the M2.5 hardening track**
> (K-016/K-017/K-018) — rationale: "long road before production." This is safe because the AI
> participant is a **server-side responder** that posts as a configured Agent and needs no
> per-request auth to function, so auth never blocks M2 green.
>
> One decision still **open, but it only gates K-016** (in the deferred track — not on any M2 path):
> DESIGN §13 identity source of truth — is the `identity` graph authoritative, or a projection of
> an external IdP? Resolve when K-016 is picked up.

## Sequencing (critical path + parallelism)

```
Parallel wave 1 (start now):
  K-011 (M1 load/PROFILE)   ─ independent (harness/docs, read-only on data)
  K-012 (M1 web polish)     ─ independent (web/ only)
  devops LM-Studio spike ─▶ K-008 gate (graph-dba) ─▶ K-008 impl (tdd)
                                                            │
                                                            ▼
                                       K-013 (agent + EMITTED)  ◀─ needs K-008 + K-010 [done]
                                                            │
                                            ┌───────────────┴──▶ K-014 (web M2) ◀─ also needs K-012
                                            ▼
                                       K-015 (QA M2 pass) ◀─ needs K-008+K-013+K-014  ⇒ M2 ✅

M1 ✅ = K-011 + K-012.
Deferred M2.5 (after M2-green): K-016 (auth) ─▶ K-017 (transport agent QA);  K-018 (real-time)
K-019 (doc sync) ─ rolls into the K-008 graph-dba gate (docs it already touches), or standalone anytime.
```

- **Critical path to M2 green:** devops spike → K-008 gate → K-008 impl → K-013 → K-014 → K-015.
- **Fully parallel with the K-008 chain:** K-011 (harness/docs) and K-012 (`web/`) — no shared files.
- **Suite discipline:** only the graph-dba gates in K-008 and K-013 touch `QUERIES.md` / `test_queries.sh`
  (raising the 126 baseline with enumerated assertions); K-011/K-012/K-014 are suite-neutral; K-015 is a QA overlay.

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
> for any new workspace from K-008 on. (`start_server.sh` guidance defaults to 1536 too — fold the
> 1024 note into both in the K-008 gate.)

## Active

### — Milestone M1 (closeout) —

### K-011 — M1 DoD closeout: load-test the append path + `GRAPH.PROFILE` the hot reads (🔵 proposed — M1)

- **Owner:** `devops` (build the load harness + capture RAM/throughput) with a **`graph-dba`** sub-pass
  (author/interpret `GRAPH.PROFILE`). Not `tdd-engineer` — this is a measurement/harness task, not a feature.
- **Inputs/prereqs:** running FalkorDB + M1 server; K-007 empirical RAM line (DESIGN §11) as baseline. Independent of K-008/K-012.
- **Scope:** (a) repeatable load harness driving the **service-layer append path** through REST (concurrent
  posters; p50/p99 append latency + sustained msg/s) — not the K-007 bulk-`UNWIND` datapoint; (b) `GRAPH.PROFILE`
  the hot reads — §4 thread read, §9.1/§9.2 since-reads, §5 search — confirm each hits `Node By Index Scan`, not
  `NodeByLabelScan`; (c) fold findings into DESIGN §10/§11 (per-workspace RAM budget + shard:workspace packing ratio).
- **Done-condition:** committed harness (`scripts/` or `docs/`) + a results section in DESIGN §11; PROFILE output for
  all four hot reads showing index scans; a documented per-workspace RAM budget line. Suite stays 126/126, pytest 110.
- **Risks/RAM (rule 6):** read-only measurement — **zero new RAM cost**. Writes ignore `TIMEOUT` (§10) → harness must
  bound its own batch sizes. Surfaces the DESIGN §13 retention question as a data-backed follow-up.
- **Test strategy:** the harness *is* the test; assert throughput/latency thresholds + no PROFILE degrades to a label
  scan. Idempotent against an isolated `ws:load` graph (create + delete), never `ws:acme`.

### K-012 — Web request/response UX polish (pulled out of K-008) (🔵 proposed — M1)

- **Owner:** `coder` (justified: `web/app.js` is vanilla JS with **no test harness** — K-005 established web JS is
  verified manually; strict TDD is a poor fit, so `coder` over `tdd-engineer`).
- **Inputs/prereqs:** `GET /threads/{tid}/messages?since=&limit=` (K-006) + `threadId` denorm (K-007) — both shipped.
  No server change required.
- **Scope:** (a) adopt `?since=&limit=` polling for the open thread (replace the full re-fetch-after-post with an
  incremental window); (b) replace `alert()` errors (`app.js:153,195`) with inline, non-blocking rendering; (c)
  clickable search results → open the message's thread (uses the `threadId` now on search rows). **Excludes** agent-reply
  rendering / `isMention` highlighting — those need the AI participant → K-014.
- **Done-condition:** manual verification checklist in the PR (poll updates a thread without full reload; failed post
  shows inline error; search-result click opens its thread). pytest/query suite untouched — 110 / 126/126 hold.
- **Risks/RAM:** none (client-side). Keep the poll windowed (`limit`) + `since`-anchored so it never walks the full
  `NEXT*` chain past the 1000ms `TIMEOUT`.
- **Test strategy:** manual smoke against a running server (documented steps); no automated web harness (accepted, K-005 precedent).

### — Milestone M2 (GraphRAG) —

### K-008 — GraphRAG retrieval core (🔵 proposed — **fully unblocked** — M2 · RE-SCOPED 2026-07-05)

> **Re-scope:** the old K-008 bundled the web client and the AI participant. Those are split out —
> web request/response polish → **K-012** (M1), web agent-reply/`isMention` → **K-014** (M2), AI participant +
> `EMITTED` → **K-013**. K-008 is now purely the embedding pipeline + vector-index verification + hybrid
> retrieval read path, split at the graph-dba→tdd gate (mirrors the K-002/K-007 pattern).

- **Owner:** **`graph-dba`** gate (verify vector index @1024, live-verify + PROFILE §6, add `test_queries.sh`
  assertions) → **`tdd-engineer`** impl (embedding worker + repository/services wiring).
- **Inputs/prereqs:** locked M2 stack (Qwen3-Embedding-0.6B, `EMBEDDING_DIM=1024`); a **devops prerequisite spike** —
  verify LM Studio `/v1/embeddings` reachable from WSL2 and returns 1024-dim vectors (reuse the severino WSL2↔LM Studio
  path). K-011 not required (parallel). Note: the §6 vector DDL already exists in `bootstrap_schema.sh:171-177` —
  the work is "create workspaces @1024 + verify the ANN query plans," not new DDL.
- **Scope:**
  1. **graph-dba gate:** create a workspace `EMBEDDING_DIM=1024`; live-verify §6 ANN query + the embedding-set query;
     `GRAPH.PROFILE` the ANN query; add `test_queries.sh` assertions for §6 (ANN retrieval + `SET m.embedding`),
     pushing the suite past 126 (enumerate the new count). Fold the 1024 default note into `bootstrap_schema.sh` /
     `start_server.sh` guidance (default stays 1536 with the choose-before-creation comment, per K-007).
  2. **tdd impl:** async embedding worker → LM Studio `/v1/embeddings` (decoupled from the post path, DESIGN §9);
     `repository.set_embedding` (1:1 §6 set query); `repository.hybrid_search` (1:1 §6) + `services.hybrid_search`
     passing a **service-layer `timeout=` constant** on the `ro_query` (K-007 TIMEOUT posture, §10) — not per-call
     ad-hockery. LLM/embedding HTTP client injected/mockable.
- **Done-condition:** query suite green at the new gate baseline (≈126 → ~135, enumerated in the gate); pytest green
  with worker + repo/service tests; message posted → embedding lands out-of-band → hybrid search returns it ranked
  by cosine distance `ASC`. `Entity` expansion verified to no-op cleanly (no `Entity` nodes yet — see note).
- **Risks/RAM (rule 6):** **the dominant new RAM line** — the 1024-dim vector index is ~**12.5 KB/message ≈ 1.25 GB
  per 100k-msg workspace** (empirical §11). Call it out per workspace. `GRAPH.MEMORY USAGE` under-reports vector
  memory (§11 caveat) — size from `INFO memory` deltas. Keep LM Studio latency off the write path (async worker).
- **Test strategy:** repository tests against isolated `ws:test` @1024 with a stub embedder (deterministic vectors)
  for ranking assertions; one live check against real LM Studio behind a marker; PROFILE assertion in `test_queries.sh`.
- **NOTE — `Entity` extraction is OUT OF SCOPE for M2.** No entity-extraction pipeline exists; the §6
  `MENTIONS→Entity` expansion is an `OPTIONAL MATCH` that no-ops cleanly, so M2 GraphRAG = vector-ANN + thread-scope
  without it. Entity extraction is parked (M3-adjacent, see Parking lot).

### K-013 — AI `Agent` participant with `EMITTED` provenance (🔵 proposed — M2)

- **Owner:** **`graph-dba`** gate (author + verify the `EMITTED` provenance write + any read surfacing it; add
  `test_queries.sh` assertions) → **`tdd-engineer`** (the responder service). `cobb` consult only if later exposed as an MCP tool.
- **Inputs/prereqs:** K-008 (hybrid retrieval) + K-010 (namespace-unique member ids — real `Agent` identity wired
  without shadowing) + `ensure_agent` v2 (§7, live). LM Studio `/v1/chat/completions` (Qwen3-4B-Instruct-2507) reachable.
- **Scope:** a server-side responder that, on a triggering message (agent `@mention` / new question in a channel the
  agent belongs to), runs K-008 hybrid retrieval, calls the LLM with retrieved context, and **posts the answer as the
  `Agent`** (role `assistant`, via the existing §4 write path — K-007 agent authorship is in) with a **new `EMITTED`
  edge** from the answer message to its provenance (seed messages / retrieval context). graph-dba defines `EMITTED`'s shape.
- **Done-condition:** query suite green at the new gate baseline; pytest green with responder tests (LLM + embedder
  mocked); live check — a user question in a seeded channel yields an agent-authored answer reading `role:"assistant"`
  on all read surfaces (K-007 invariant) with a queryable `EMITTED` provenance edge.
- **Risks/RAM (rule 6):** one `EMITTED` edge + one answer `Message` (with its own embedding once K-008 embeds it) per
  answer — **negligible vs. the K-008 vector line**; count the new relationship type. LLM latency/failure must not
  corrupt the thread — the LLM call precedes the guarded §4 write; failure = no post. **Trigger must exclude
  agent-authored messages** (no self-answer feedback loop).
- **Test strategy:** unit — responder with mocked retrieval + mocked LLM (deterministic answer); contract — the
  `EMITTED` write in `test_queries.sh`; one live smoke behind a marker.

### K-014 — Web M2: render agent replies + reader `isMention` highlighting (🔵 proposed — M2)

- **Owner:** `coder` (same web-JS-no-harness justification as K-012).
- **Inputs/prereqs:** K-012 (polling base) + K-013 (agents actually posting). Uses the since-read `isMention` flag (§9,
  already server-side).
- **Scope:** render agent-authored (`role:assistant`) messages distinctly; restore reader `isMention` highlighting via
  the since-read flag (the K-005 "dead highlight" is alive once polling drives the UI); surface agent answers as they
  arrive via the K-012 poll loop.
- **Done-condition:** manual checklist — an agent answer appears in the polling web UI styled as assistant; a message
  mentioning the reader is highlighted. Suites untouched (110 / 126/126).
- **Risks/RAM:** none (client-side).
- **Test strategy:** manual smoke against a running server with the K-013 responder live.

### K-015 — QA acceptance pass on M2 GraphRAG (🔵 proposed — M2 · the gate that flips M2 → ✅)

- **Owner:** `qa-engineer`.
- **Inputs/prereqs:** K-008 + K-013 + K-014 landed.
- **Scope:** black-box acceptance pass on the GraphRAG loop — embedding lands out-of-band, hybrid retrieval ranks
  correctly, the agent participant answers with provenance, the web UI renders it. Versioned test plan + report per repo
  convention (`docs/test-plans/`, `docs/test-reports/`). **Explicitly notes** the still-deferred transport-level
  agent-actor path (carries the K-007 QA carry-over forward to K-017) since auth isn't in yet.
- **Done-condition:** `docs/test-plans/m2-graphrag.md` + `docs/test-reports/m2-graphrag-report.md`; PASS (or
  PASS-with-parked-defects) on green baselines; isolated `ws:qa` (create + delete), `ws:acme`/`reference` untouched.
- **Risks/RAM:** none (no code under test changed); budget the transient `ws:qa` @1024 vector index.
- **Test strategy:** the pass itself; drives REST + MCP + the running responder.

### — Doc hygiene (cross-cutting) —

### K-019 — Documentation-inconsistency sweep (🔵 proposed — doc-only)

- **Owner:** `coder` (doc-only; no test discipline needed). Preferably **folded into the K-008 graph-dba gate**, which
  already edits `bootstrap_schema.sh` / `start_server.sh` / DESIGN §11 — one coherent doc-sync instead of a drive-by.
- **Inputs/prereqs:** none. Source of truth for counts = `AGENTS.md` (pytest 110 / query suite 126/126) and the live
  suites (`./scripts/test_queries.sh`, `pytest -q`) — re-run to confirm the current numbers before editing.
- **Scope (verified 2026-07-05, file:line):**
  1. **Stale test counts** — reconcile to current 110 / 126/126: `README.md:122` (`115/115`→`126/126`),
     `README.md:189` (`115 assertions`→126), `README.md:169` (`75 tests`→110), `README.md:220` (`98 passed`→110),
     `DESIGN.md:459` (`70 tests`→110). Leave `README.md:168` M0 `92/92` only if re-labelled as the historical M0
     figure; otherwise update.
  2. **Locked-decision drift** — the embedding model + dim is decided (Qwen3-Embedding-0.6B, `EMBEDDING_DIM=1024`,
     user-approved 2026-07-04) but DESIGN still lists it open: remove/rewrite `DESIGN.md:468` (§13 "Embedding model &
     dimension" open question) and `DESIGN.md:439` ("embedding model still open, §13") to point at the locked decision.
     Keep the `EMBEDDING_DIM` *default* at 1536 in scripts (intentional — chosen per workspace; the 1024 fold-in is
     K-008's job) but make the docs say the model is chosen.
  3. **Scope wording** — reconcile DESIGN §14.1 ("real-time + auth deferred *to M2*") with the §12 M2 DoD and the now
     confirmed "M2 green = functional GraphRAG; auth/real-time → M2.5" decision, so the two sections agree.
- **Done-condition:** every count in README/DESIGN matches the live suites; no doc still lists the embedding model as
  an open question; §12/§14.1 scope wording is internally consistent. Grep sweep clean; no code/query change → suites
  untouched (110 / 126/126).
- **Risks/RAM:** none (doc-only).
- **Test strategy:** re-run both suites to source the true counts before editing; final grep for the old numbers
  (`115`, `98`, `75`, `70`) and the "still open" embedding phrasing returns nothing stale.

### — Deferred M2.5 hardening track (auth + real-time; not on any M2-green path) —

### K-016 — Real auth/tenancy replacing the hardcoded `get_context` seam (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design pass — resolves DESIGN §13 "identity source of truth": is the `identity` graph
  authoritative, or a projection of an external IdP?) → **`tdd-engineer`** (implement the resolved `get_context`).
- **Inputs/prereqs:** the DESIGN §13 identity decision (**needs the user** when picked up). Localized by design — only
  `config.get_context` changes (`config.py:43`); everything below already parameterized on `ws`/`actor`.
- **Scope:** token → (user, workspace claim) resolution replacing hardcoded `ws=acme/user=u1`; wire the `identity`
  graph per the §13 decision; keep or replace MCP's `frm`-ignoring rule with authenticated agent identity.
- **Done-condition:** `get_context` resolves a real principal from a credential; multi-tenant isolation test; pytest green.
- **Risks/RAM:** `identity` graph nodes (small). First real trust boundary — MCP endpoint is currently unauthenticated (§15.3).
- **Test strategy:** service/api tests with injected auth contexts; a cross-tenant isolation test.

### K-017 — Transport-level agent-actor path (K-007 QA carry-over) (🔵 proposed — M2.5, deferred · depends on K-016)

- **Owner:** `qa-engineer` (+ small `tdd-engineer`/`coder` fold-in if MCP must express an authenticated agent actor).
- **Scope:** with auth able to express an *agent* principal, drive an external agent authoring over MCP/REST (the M1
  hardcoded seam couldn't) and verify authorship/role/provenance end-to-end.
- **Done-condition:** the K-007 QA carry-over closed — a report showing an externally-authenticated agent authoring
  first-class over the transport.
- **Risks/RAM:** none new. **Test strategy:** black-box over MCP with an agent credential.

### K-018 — Real-time push (Redis Pub/Sub → WebSocket/SSE) (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design: Pub/Sub fan-out topology; resolve the DESIGN §13 Bolt-vs-RESP gateway question
  here since it touches the transport) → **`coder`/`tdd-engineer`**.
- **Inputs/prereqs:** K-012/K-014 web client (swap polling → push).
- **Scope:** Redis Pub/Sub on message write → WebSocket/SSE endpoint on the same FastAPI process (§14.1: "slots onto
  the same service layer, no schema change") → web client subscribes instead of polling.
- **Done-condition:** a posted message appears in another client without a poll; graceful fallback to polling.
- **Risks/RAM:** no graph RAM; Pub/Sub is transient. Publish *after* the guarded §4 write commits, never inside it (atomicity rule).
- **Test strategy:** integration test of publish-on-write + a WebSocket client receiving it.

## Recommended plan docs (author when each item is picked up — not yet created)

| Path | Scope |
|---|---|
| `docs/plans/m2-graphrag.md` | K-008 re-scoped: embedding worker + vector-index-@1024 verification + hybrid retrieval read path. |
| `docs/plans/m2-agent-participant.md` | K-013: `EMITTED` provenance edge + LLM responder posting as the `Agent`. |
| `docs/plans/m1-hardening-loadtest.md` | K-011: append-path load harness + hot-read PROFILE targets + per-workspace RAM budget. |
| `docs/plans/m2-auth-tenancy.md` | K-016 (deferred): real auth replacing `get_context`, resolving §13 identity source of truth. |
| `docs/plans/m2-realtime.md` | K-018 (deferred): Pub/Sub → WebSocket/SSE, resolving §13 Bolt-vs-RESP. |

## Parking lot / ideas

- **`Entity` extraction pipeline** (M3-adjacent) — build the `MENTIONS→Entity` corpus so the §6 hybrid query's entity
  expansion becomes live (today it's an `OPTIONAL MATCH` no-op). Enables entity-anchored GraphRAG; watch the `Entity`
  supernode risk (DESIGN §5.4).
- Verify the K-009 GitHub Action goes green on first push (path-filtered `.github/workflows/falkor-chat.yml`; FalkorDB
  service container). Note the CI baseline echoes in its comments (75/92) predate K-007/K-010's 110/126 — the suites
  themselves are the source of truth. (K-019 fixes the README/DESIGN body numbers; the CI comments are separate.)
- File upstream FalkorDB issues (K-007 OQ6, recommended to the user): `GRAPH.MEMORY USAGE` under-reports vector-index
  memory; one-shot instant-timeout anomaly after a long override run.
- Per-endpoint response schemas (QA, recommended three times now): full-thread / since-reads / search each carry a
  different field subset (all documented/intentional) — a declared schema per endpoint would make the contract testable
  and stop accretion.
- DESIGN §13 remaining open questions — resolve as their milestones arrive: workflow guard expression language (M3),
  `identity` source of truth + real auth (K-016), message/embedding retention, cross-workspace analytics, Bolt vs RESP
  for the gateway (K-018).
