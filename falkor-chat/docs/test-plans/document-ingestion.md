# Document ingestion & entity fusion — Test Plan

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-050 (M5)

Milestone-closing acceptance pass for K-050 (M5 — ingestion pipeline & entity fusion). All six
implementation stages are delivered, diff-gated by `analyst`, and committed (most recently Stage
6a, `a099f9b`, bulk `ingest_documents`). This is unit U30 of
`docs/plans/document-ingestion-coordination.md` — Stage 6c, the last unit before M5 closes. Verifies
`docs/requirements/document-ingestion.md`'s AC-1..AC-10 against the running system, at the altitude
`docs/plans/document-ingestion.md` §5's test-strategy table names for each, extending past the
1751-passed/4-deselected offline suite that already covers this feature's unit/integration layer.

## References

- Requirements: `docs/requirements/document-ingestion.md` (FR-1..FR-14, AC-1..AC-10 verbatim, §
  "Acceptance criteria").
- Plan: `docs/plans/document-ingestion.md` — §3 (design & rationale, all fusion-tier/retrieval
  mechanics), §5 (test strategy — the altitude for each AC), §4 (stage-by-stage "Done" criteria).
- Graph note: `docs/plans/document-ingestion-graph.md` (exact Cypher, live-verified).
- Coordination ledger: `docs/plans/document-ingestion-coordination.md` (U1-U29 — every stage's
  implementation + gate history; read in full before this unit per the dispatch brief).
- Query library: `docs/QUERIES.md` §14 (Documents & Chunks), §14.4 (MCP/REST surface + bulk),
  §14.5 (extraction), §14.6 (fusion — exact/fuzzy/confirm/reject/recheck/list).
- Precedent for this document's own shape: `docs/test-plans/workflows.md` /
  `docs/test-reports/workflows-report.md` (K-015/K-022/K-024 milestone-closing acceptance pass).

## CPG

Considered, not relevant for this pass's own strategy: `cpg_falkorchat` is stale (built
2026-08-17, now well over a dozen commits behind, including this entire feature per the dispatch
brief) — not leaned on for structural claims; every claim in this plan is grounded in reading
`server/falkorchat/*.py`/`docs/QUERIES.md` directly and in live-driving the running server.

## Risk assessment / prioritization

This feature's own design already separates two axes that are easy to conflate (plan §3.1): fact
provenance (FR-6, never deduplicated) and entity identity fusion (FR-7..FR-10, three confidence
tiers). The highest residual risk at milestone close is not "does the mechanism work" — six stages
of diff-gated, independently-verified offline tests already prove the mechanics repeatedly
(1751 passed/4 deselected) — it is **whether the system behaves the same way when driven through
the real front doors** (REST/MCP, over the wire, against the real running server and a real local
LLM) as it does inside the offline suite's fakes/fixtures, and whether the plan's own "Done"
promises (a caller-visible completion signal, DESIGN.md staying in sync) actually shipped. Priority
order:

1. **Cross-transport / full-fidelity surface claims (AC-6, AC-9)** — the dispatch brief's own
   named priority: these are pure read/write correctness claims with no LLM dependency, so a live
   failure here would be a real, high-confidence regression, not extraction noise.
2. **The two "no confirmation" / "always both kept" guarantees (AC-1, AC-2)** — FR-8's exact-tier
   auto-merge and FR-6's never-overwrite guarantee are the two zero-human-review paths in this
   design; a live miss here is the highest-impact defect this pass could find.
3. **The new bulk-API-altitude AC-8 coverage** — Stage 6a added `ingest_documents` but the
   dispatch brief is explicit that this pass owes AC-8 *real* bulk-endpoint coverage (not just the
   pipeline-internal `extract_chunk` calls Stages 1-5 tested), through both REST and MCP.
4. **AC-10 (traceability) and AC-5 (chat-grounding provenance)** — both have solid existing
   coverage (offline + a live-marked e2e for AC-5); this pass re-confirms live rather than
   discovering from scratch.
5. **AC-3/AC-4/AC-7 (suggested tier + confirm/reject/recheck)** — mechanically proven offline and
   independently gated at Stage 4 (including a live-reproduced concurrency fix); this pass attempts
   genuine live derivation where a real LLM can be steered toward it, and falls back to citing the
   existing gated coverage where forcing a *specific* fusion outcome out of a real, non-deterministic
   local model is not a reliable acceptance-test technique (see "Deliberate scope calls" below).
6. **Milestone done-condition bookkeeping** — the K-050 done-condition names `DESIGN.md` §5.1/§7
   staying in sync as part of what M5 closing requires; checked as part of this pass, not assumed.

**Deliberate scope calls (what this pass does NOT try to force live, and why):**

- **AC-7's automatic re-open path (OQ-3 path 1)** is not forced through real extraction. Triggering
  it precisely requires a *later* ingestion's fuzzy candidate lookup to land on the **exact same
  entity-id pair** an earlier, now-rejected `SAME_AS` edge already connects — but every extraction
  always mints a **brand-new** `Entity` node (`create_entity`/`create_entity_with_auto_match`,
  §14.5/§14.6), so a third ingestion can never reproduce the first two ingestions' exact pair; it
  can only create new pairs against the same candidates. This is not a testability gap in the
  *system* — the offline suite proves the mechanism directly at the repository layer
  (`create_or_reopen_match`'s reopen branch, live-verified per `document-ingestion-graph.md` §1.6
  and re-confirmed at Stage 4's gate) — it is a live-acceptance-technique limit: manufacturing the
  precise precondition through a real, temperature-0-but-still-not-perfectly-deterministic local
  model would be theater, not a stronger acceptance signal than the existing repository-level proof.
  This pass instead re-confirms the mechanism is still present in the shipped code and cites the
  existing gated test as the altitude the plan itself specifies (§5: "repository/service
  integration"). The **manual** re-open path (`recheck_match`, OQ-3 path 2) has no such
  determinism problem and **is** driven live end-to-end.
- **AC-2/AC-3's exact fusion-tier outcome** depends on a real local LLM extracting the *same*
  normalized name + type from near-identical input twice (AC-2) or a *fuzzy-but-not-exact* variant
  (AC-3) — attempted live (model config is `temperature: 0` per `config/models.json`, improving but
  not guaranteeing determinism), with the actual observed outcome reported honestly either way, and
  the deterministic mechanism itself corroborated against the existing, independently-gated
  repository-level tests (`test_extract_chunk_exact_match_...`,
  `test_create_entity_with_auto_match_concurrent_calls_produce_exactly_one_edge`, etc.) regardless
  of what the live LLM happens to produce on this run.
- **A pending-match fixture for AC-4/AC-7's manual paths**: if live fuzzy-tier extraction does not
  land a pending `SAME_AS` edge on this run (see above), a `SAME_AS{status:'pending'}` edge is
  written directly via `redis-cli GRAPH.QUERY` between two pre-existing `ws:acme` entities as
  **disclosed test-data setup**, not a substitute for testing the write path — `confirm_match`/
  `reject_match`/`recheck_match`/`list_pending_matches`/`list_matches` are then still driven for
  real over REST/MCP against that fixture, which is exactly what those test items are checking
  (the API/service contract, not the fuzzy-candidate-generation mechanism, which is a different,
  already-covered concern).
- **AC-5** is re-verified by re-running the existing live-marked e2e
  (`server/tests/test_ac5_chat_grounding_live.py`, `pytest -m live`) against the current tree,
  rather than re-deriving a new live drive through the full `@mention`-triggered workflow path —
  that test's own docstring records a deliberate scope choice (bypass the M3 workflow-trigger
  machinery, exercise `Services`/`Repository`/`AgentResponder` directly) that exactly matches the
  plan's own AC-5 test-strategy altitude ("responder integration ... + a live-marked e2e"); driving
  the *workflow-triggered* `@mention` path live as well would exercise the `triage` workflow's own
  retrieval step, a different code path from `AgentResponder.maybe_respond` and out of this
  feature's scope.
- **AC-10's assertion surface is direct Cypher, not a documented read endpoint** — there is no
  MCP/REST "list a document's entities" tool (`search_documents` returns ranked chunks only, no
  `ABOUT`/`RELATES_TO` expansion, §14.4). The **write** is driven live through the real server; the
  **read** that proves traceability is a direct graph read, matching the plan's own "repository/
  service integration" altitude for this AC.

## Environment

- FalkorDB `falkordb-dev` (Docker, already running), `ws:acme` pre-existing and pre-seeded with 3
  `Document`s / 50 `Chunk`s / 523 `Entity`s / 371 `RELATES_TO` edges / 0 `SAME_AS` edges from
  earlier stages' own testing (per U18's checkpoint) — a real, non-empty corpus, not a clean-room
  workspace.
- Baseline established first (both green, matching the dispatch brief's stated counts):
  `.venv/bin/python -m pytest -q` → **1751 passed, 4 deselected**; `./scripts/test_queries.sh` →
  **320/320**. Both wipe the shared `reference` graph at teardown (documented `AGENTS.md` hazard);
  restored via `bootstrap_schema.sh acme` (needed first — a bare `seed_workflows.sh` after
  `test_queries.sh`'s `GRAPH.DELETE` teardown fails "Invalid graph operation on empty key" per the
  same documented gotcha U9 hit) → `seed_workflows.sh acme` → `verify_workflows.sh acme` confirms
  in sync, both before and after this pass's live driving.
- Server started per `falkor-chat/AGENTS.md`'s `./scripts/start_server.sh` (REST + MCP +
  AI responder + workflow engine on one uvicorn process, `ws:acme`, port 8000).
- **LM Studio is reachable** (`http://localhost:1234/v1/models` returns a live model list), so
  every LLM-dependent path (extraction, embedding, AC-5's live test) is exercised for real, not
  skipped. **Environment note, not a `falkor-chat` code defect**: the shared, pristine
  `~/.config/opencode/opencode.json` this repo's `start_server.sh` uses by default has a stale
  gateway IP (`192.168.0.69`) baked into the `lmstudio` provider's `baseURL`, refused from this
  WSL2 box — confirmed via the server's own background-thread log (`ProviderCallError: ...
  connection failed: [Errno 111] Connection refused`), consistent with a kaizen entry already filed
  during Stage 3's `data-scientist` checkpoint (U18) for the same stale IP. Restarted the server
  with `FALKORCHAT_OPENCODE_CONFIG` pointed at a corrected local copy (`baseURL:
  http://localhost:1234`, otherwise identical) so this pass's live LLM-dependent items are not
  blocked by an unrelated, already-known environment quirk. `~/.config/opencode/opencode.json`
  itself was **not** modified (shared, out-of-repo, other tools/agents depend on it).
- A real MCP Streamable-HTTP client (`mcp.client.streamable_http`, the same SDK the server itself
  depends on, `mcp>=1.28,<1.29`) drives MCP tool calls over the wire against `http://localhost:8000/mcp`
  — not an in-process FastMCP call the way `server/tests/test_mcp.py` exercises it — so AC-6's
  "MCP write, any-transport read" is proven cross-process, matching how K-041's own live QA pass
  (`kiro/docs/test-reports/kiro-demo-agent-report.md`) found a real cross-transport gap that no
  in-process test had ever exercised.

## Test items

| ID | AC | Title | Altitude | Priority |
|---|---|---|---|---|
| TP-01 | AC-9 | Multi-chunk document round-trips byte-identical via REST | e2e | High |
| TP-02 | AC-6 | MCP `ingest_document` write → REST `get_document` read finds it | integration | High |
| TP-03 | AC-6 | REST `ingest_document` write → MCP `get_document` read finds it | integration | High |
| TP-04 | AC-10 | Entities + relationship extracted, traceable to source chunk/document | repository/service integration (live write, direct read) | High |
| TP-05 | AC-1 | Two conflicting facts about the same subject both survive, neither overwritten | repository/service integration (live write, direct read) | High |
| TP-06 | AC-2 | Identical content ingested twice auto-merges with no pending step | repository/service integration (live attempt) + offline corroboration | Medium |
| TP-07 | AC-3 | A fuzzy-but-not-exact name variant lands as a pending suggestion, stays unlinked | repository/service integration (live attempt) + offline corroboration | Medium |
| TP-08 | AC-4 | `confirm_match`/`reject_match` transition status + stamp audit fields | contract (REST, live) | High |
| TP-09 | AC-7 | `recheck_match` reopens a rejected match on demand; auto-reopen mechanism re-confirmed | contract (REST, live) + repository-level corroboration for the automatic path | High |
| TP-10 | AC-8 | Bulk `ingest_documents` (MCP) fuses two documents' shared entity, cross-document | e2e (real bulk API, MCP transport) | High |
| TP-11 | AC-5 | Chat-grounded answer's provenance resolves to source chunk/document | e2e (live-marked pytest, real LLM+embedder) | High |
| TP-12 | — | Milestone done-condition: `DESIGN.md` §5.1/§7 reflect the shipped schema | doc audit | Medium |

### TP-01 — AC-9 full document retention round-trip
**Preconditions:** server up.
**Steps:** `POST /documents` with an ~8KB multi-paragraph synthetic text (forces multiple chunks
at the 1000-char/150-overlap default); `GET /documents/{id}`.
**Expected:** `text` field byte-identical to the input; `chunkCount` > 1.

### TP-02 — AC-6 cross-transport, MCP write → REST read
**Steps:** MCP tool call `ingest_document` (real Streamable-HTTP client) with distinctive marker
text; `GET /documents/{id}` over REST.
**Expected:** REST read returns the exact text/title the MCP call wrote.

### TP-03 — AC-6 cross-transport, REST write → MCP read
**Steps:** `POST /documents` over REST with distinctive marker text; MCP tool call `get_document`.
**Expected:** MCP read returns the exact text/title the REST call wrote.

### TP-04 — AC-10 entity/relationship traceability
**Steps:** `POST /documents` with a clear "X acquired Y"-shaped sentence naming two distinctive
entities; wait for background extraction; direct Cypher read: `Entity` nodes for both names,
`RELATES_TO` edge between them, `sourceChunkId`/`sourceDocumentId` on the edge resolving back to
the ingested chunk/document.
**Expected:** both entities exist, the relationship edge exists and carries correct provenance
properties.

### TP-05 — AC-1 conflicting facts both survive
**Steps:** `POST /documents` twice, each naming the same distinctive subject with a directly
conflicting numeric fact (e.g. two different headcounts); wait for background extraction; direct
Cypher read for `RELATES_TO` edges sourced from each document's chunk.
**Expected:** both facts present as separate edges, distinct `sourceChunkId`/`createdAt`, neither
edge missing/overwritten by the other.

### TP-06 — AC-2 auto-merge, no confirmation needed
**Steps:** `POST /documents` twice with **identical** text naming one distinctive entity; wait for
background extraction+fusion; `GET /matches?status=confirmed`.
**Expected (live, if the model extracts identically twice):** a `SAME_AS{status:'confirmed',
decidedBy:'system', confidence:1.0}` edge links the two entities, written at ingestion time with no
pending step. **Corroborating altitude regardless of the live outcome:** the offline, Stage-4-gated
`test_extract_chunk_exact_match_skips_the_fuzzy_lookup_entirely` and the concurrency regression
test prove the deterministic mechanism directly.

### TP-07 — AC-3 suggested match stays unlinked until confirmed
**Steps:** `POST /documents` twice naming the same entity with a fuzzy-but-not-exact name variant
(e.g. "X Manufacturing" vs. "X Manufacturing Inc"); wait for background extraction+fusion;
`GET /matches/pending`.
**Expected (live, if the model's naming lands a fuzzy-not-exact pair):** a `SAME_AS{status:'pending'}`
edge; the two entities are not treated as fused by any confirmed-only read. **Corroborating
altitude regardless of live outcome:** `test_extract_chunk_fuzzy_candidate_writes_a_pending_match_
for_the_top_hit` (offline, Stage-4-gated).

### TP-08 — AC-4 confirm/reject transitions + audit fields
**Steps:** on a real pending match (from TP-07, or a disclosed direct-Cypher fixture if TP-07 did
not land one — see "Deliberate scope calls"): `POST /matches/{id}/confirm`; on a second pending
match: `POST /matches/{id}/reject`.
**Expected:** confirm → `status:'confirmed'`, `decidedBy` = the real actor id (never `'system'` on
this path), `decidedAt` set; reject → `status:'rejected'`, same audit-field shape.

### TP-09 — AC-7 rejection is reversible
**Steps:** `POST /matches/{id}/recheck` on the match rejected in TP-08.
**Expected:** `status` flips back to `'pending'`, `resuggestCount` increments. **Automatic
re-open path (OQ-3 path 1):** not forced live (see "Deliberate scope calls"); re-confirmed present
in the shipped `create_or_reopen_match` Cypher and its existing, Stage-4-gated reopen test.

### TP-10 — AC-8 bulk cross-document fusion via the real batch API
**Steps:** MCP tool call `ingest_documents` with two items whose text identically names one
distinctive entity; wait for background extraction+fusion; direct Cypher / `GET /matches` read.
**Expected:** two receipts (one per item, `MAX_BATCH_SIZE` semantics per §14.4), each document's
chunk yields its own `Entity`, and a `SAME_AS` edge links the two — proving fusion works against
the real bulk endpoint + MCP transport, not just the pipeline-internal `extract_chunk` calls the
offline `test_ac8_...`/`test_batch_ingest_...` tests already cover.

### TP-11 — AC-5 chat-grounding provenance (live)
**Steps:** `cd server && .venv/bin/python -m pytest -m live -s tests/test_ac5_chat_grounding_live.py`.
**Expected:** passes against the current tree (last known-passing at Stage 5's gate, before Stage
6a's changes) — ingests a document, `@mention`s the agent with a question it answers, asserts the
answer's `EMITTED` provenance resolves to the source chunk/document via `read_provenance`/
`read_citing_answers`.

### TP-12 — Milestone done-condition: DESIGN.md currency
**Steps:** read `docs/DESIGN.md` §5.1 (arrow notation) and §7.1 (index/constraint table); compare
against the shipped schema (`docs/QUERIES.md` §14, `scripts/bootstrap_schema.sh`).
**Expected:** §5.1 shows `RELATES_TO` and `SAME_AS` (with their properties) alongside the existing
`ABOUT`/`HAS_CHUNK` edges, and `Entity.nameNormalized`; §7.1's index table includes
`Entity.nameNormalized` (RANGE), `Entity.name` (fulltext), and `SAME_AS.matchId`/`SAME_AS.status`
(relationship-scoped indexes + the `matchId` uniqueness constraint). A gap here is a doc-currency
finding, not an AC failure — reported as such, not fixed by this pass (out of `qa-engineer`'s lane
per the dispatch brief; flagged back to the coordinator).

## Entry/exit criteria

**Entry:** FalkorDB reachable; offline `pytest -q` and `./scripts/test_queries.sh` both green on
the current tree (established fresh, not assumed from the dispatch brief's stated counts); server
reachable at `/health`; `ws:acme`'s `reference` workflow-def sync verified
(`verify_workflows.sh acme`).
**Exit:** all twelve items executed to pass/fail/blocked/inconclusive with evidence; every
LLM-dependent item's actual live outcome reported honestly (an inconclusive live fusion-tier
outcome is not a defect by itself — see "Deliberate scope calls" — but is reported as observed,
not silently upgraded to a pass on the strength of the offline corroboration alone); `ws:acme`'s
`reference` sync re-verified after this pass's own suite re-runs.

## Out of scope

- Re-deriving unit-level coverage the offline suite (1751 passed/4 deselected, independently
  gated stage-by-stage per the coordination ledger) already proves — this pass extends past it,
  not under it.
- The MCP per-chunk background-thread fan-out's *scaling* characteristic (documented, accepted,
  non-blocking per the dispatch brief) — not re-litigated; only reported if actually observed
  destabilizing this pass's own live driving (it was not — see the report).
- OQ-1's fusion-default methodology (exact normalized-name+type, no ML confidence score) — a
  confirmed v1 design decision (`data-scientist` checkpoint, U18), not re-opened here.
- Web UI (`web/`) — no UI work is in scope for this feature (plan §1, "Out of scope").
- Load/concurrency/performance testing beyond what the existing concurrency regression test
  already covers offline.
