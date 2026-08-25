# Document ingestion & entity fusion — Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-050 (M5)

Execution of `docs/test-plans/document-ingestion.md` — the K-050 (M5) milestone-closing
acceptance pass, unit U30 of `docs/plans/document-ingestion-coordination.md`. Executed
2026-08-25 against commit `a099f9b` (Stage 6a, bulk `ingest_documents`), the tip of `main` at
dispatch time. FalkorDB `falkordb-dev`, `ws:acme` (pre-existing corpus + this pass's own test
data). LM Studio was reachable and used for real throughout — every AC that depends on real
extraction/embedding/chat completion was driven with a live local model, not mocked.

**CPG: considered, not relevant — `cpg_falkorchat` is stale (built 2026-08-17, well over a dozen
commits behind, including this entire feature per the dispatch brief); this pass's own strategy
and every finding below are grounded in reading `server/falkorchat/*.py`/`docs/QUERIES.md`
directly and in live-driving the running server, not in querying the graph.**

## Overall verdict: PASS-with-parked-defects

All ten acceptance criteria (AC-1..AC-10) hold, live-verified against the running server through
real REST and MCP calls (not just the offline suite). Two findings are parked as non-blocking
defects for a fast follow, neither of which falsifies any AC: **Document.status never reaches a
terminal state** (Defect 1, Medium) and **`docs/DESIGN.md` §5.1/§7.1 are stale against the shipped
schema** (Defect 2, Medium — a milestone done-condition gap, not a runtime defect). Baselines are
green both before and after this pass's live driving: offline `pytest -q` **1751 passed, 4
deselected** (unchanged), `./scripts/test_queries.sh` **320/320** (unchanged), the live-marked
AC-5 e2e test **1 passed**.

## What was actually driven live vs. verified by reading/re-running

**Driven live, for real, against the running server (not just re-running existing tests):**
AC-9 (REST), AC-6 both directions (a genuine out-of-process MCP Streamable-HTTP client against
`/mcp`, plus REST), AC-10 (REST write + direct graph read), AC-1 (REST write + direct graph
read), AC-2 (REST write, real LLM extraction, observed live auto-merge), AC-3 (REST write, real
LLM extraction, observed live pending suggestion), AC-4 (REST confirm/reject against real
matches), AC-7's manual path (REST recheck against a real rejected match), AC-8 (a real MCP
`ingest_documents` batch call, real cross-document fusion observed).

**Verified by re-running an existing test, not re-derived from scratch:** AC-5 (re-ran
`server/tests/test_ac5_chat_grounding_live.py`, `pytest -m live` — passed against the current
tree); the full offline suite and query suite as green baselines before and after.

**Verified by citing existing, independently-gated coverage rather than forcing a live outcome:**
AC-7's automatic re-open path (OQ-3 path 1) — see "Deliberate scope calls" in the test plan for
why this is a live-acceptance-technique limit, not a testability gap in the system.

## Results

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP-01 | AC-9 | **PASS** | `POST /documents` with an 8,366-char, 12-chunk synthetic doc → `GET /documents/{id}` returned `text` byte-identical (`got["text"] == orig` → `True`), `sourceFormat`/`title` correct |
| TP-02 | AC-6 | **PASS** | MCP `ingest_document` (real Streamable-HTTP client) wrote doc `b4906bc7...`; `GET /documents/b4906bc7...` over REST returned the exact text/title the MCP call wrote |
| TP-03 | AC-6 | **PASS** | `POST /documents` over REST wrote doc `693a10f9...`; MCP `get_document` returned the exact text/title the REST call wrote |
| TP-04 | AC-10 | **PASS** | `POST /documents {"text":"Griffin Aerospace acquired Solstice Robotics in 2024."}` → `Entity(Griffin Aerospace, Organization)`, `Entity(Solstice Robotics, Organization)`, one `RELATES_TO{label:"acquired"}` edge between them, `sourceChunkId`/`sourceDocumentId` both resolving to the ingested chunk/document (direct Cypher read) |
| TP-05 | AC-1 | **PASS** | Two docs, "Meridian Analytics has 40 employees." / "...400 employees." → two separate `RELATES_TO{label:"has"}` edges from the same `Meridian Analytics` entity, distinct `sourceDocumentId`/`sourceChunkId`/`createdAt` (`93ec271c.../6767e176...` @ `1787695691745` and `c0b17028.../18acf5e4...` @ `1787696147647`), neither missing |
| TP-06 | AC-2 | **PASS (live)** | "Northbridge Systems provides cloud infrastructure services." ingested twice → two `Entity{nameNormalized:"northbridge systems"}` nodes → `SAME_AS{status:"confirmed", confidence:1.0, technique:"exact_normalized_name_type", decidedBy:"system"}` written at ingestion, no pending step. A second, unplanned confirmation of the same mechanism landed too: "Meridian Analytics" (from TP-05's two docs) also auto-merged (`matchId 151523482c...`) |
| TP-07 | AC-3 | **PASS (live)** | "Talbridge Manufacturing..." / "Talbridge Manufacturing Inc..." → `nameNormalized` **not** exact (`talbridge manufacturing` vs. `talbridge manufacturing inc`) → `SAME_AS{status:"pending", confidence:16, technique:"fuzzy_fulltext"}`; `GET /matches/pending` listed it, `GET /matches?status=confirmed` did not |
| TP-08 | AC-4 | **PASS** | `POST /matches/9d665cc9.../confirm` → `{status:"confirmed", ...}`, graph read confirms `decidedBy:"u1"` (real actor, never `"system"` on this path), `decidedAt` stamped; `POST /matches/ee2bb47d.../reject` → `{status:"rejected", ...}`, same audit-field shape |
| TP-09 | AC-7 | **PASS** | `POST /matches/ee2bb47d.../recheck` → `{status:"pending", ...}`; graph read confirms `resuggestCount:1`, `lastResuggestedAt` stamped. Automatic re-open (OQ-3 path 1) not forced live (see plan's "Deliberate scope calls"); re-confirmed present in the shipped `create_or_reopen_match` Cypher (`repository.py`, matches `document-ingestion-graph.md` §1.6 verbatim) and its Stage-4-gated reopen test, unchanged since that gate |
| TP-10 | AC-8 | **PASS (live)** | MCP `ingest_documents` (one call, two items, both "Pinnacle Freight operates a nationwide trucking network.") → two receipts, two documents (`c1809522.../a4fdfee2...`), each with its **own** `Entity(Pinnacle Freight)` traced through its own `Document-[:HAS_CHUNK]->Chunk-[:ABOUT]->Entity` chain → `SAME_AS{status:"confirmed", decidedBy:"system"}` linking them, cross-document, through the real bulk endpoint + MCP transport (not just the pipeline-internal `extract_chunk` calls the offline `test_ac8_.../test_batch_ingest_...` tests already cover) |
| TP-11 | AC-5 | **PASS** | `pytest -m live -s tests/test_ac5_chat_grounding_live.py` → `1 passed in 2.08s` against the current tree (post Stage 6a) |
| TP-12 | done-condition | **FINDING, not pass/fail** | `docs/DESIGN.md` §5.1/§7.1 confirmed stale — see Defect 2 |

**12/12 test items executed. 11 pass, 1 finding (doc-currency, not an AC).**

## Defects

### Defect 1 — `Document.status` never reaches a terminal state; permanently stuck at `'processing'`

**Severity: Medium.** Not a blocker — no AC names a `Document.status` value, and every AC this
pass checked holds regardless. But it breaks the **one** caller-visible completion signal this
feature's own plan promises, and it is trivially reproducible on any document, however long ago
ingested and however completely processed.

**Steps to reproduce:**
1. `POST /documents {"text": "Griffin Aerospace acquired Solstice Robotics in 2024."}` →
   `{"documentId": "ad0605b8...", "status": "processing"}`.
2. Wait for background extraction to finish (confirmed complete — `Entity`/`RELATES_TO` fully
   present, see TP-04).
3. `GET /documents/ad0605b8...` → `"status": "processing"` — **still**, indefinitely.

**Expected:** per `docs/plans/document-ingestion.md` §4 Stage 1 ("`Document.status` starts
`'processing'`; nothing in this stage flips it to `'ready'`" — explicitly deferred to "a later
stage's background pipeline, plan §3.6") and §5's own AC-8 test-strategy row ("background-
completion-aware (poll `Document.status` or run the pipeline synchronously in the test)"), a
document whose background pipeline has finished should report a terminal status.

**Actual:** `grep -rn "'ready'\|status.*'failed'" server/falkorchat/*.py` finds no code path that
ever writes anything but `'processing'` to `Document.status` — confirmed by `analyst`'s own Stage
1 gate (`docs/reviews/document-ingestion-impl.md:97`, "`grep`-verified no `'ready'` status flip
exists outside doc comments describing a *future* stage" — correct and expected **at that time**,
since Stage 3 was named as the stage that would build it). No stage from 3 through 6a ever built
it: `create_document`'s Cypher (`repository.py:1029`) hardcodes `status: 'processing'` at
creation, and no other write touches `Document.status` anywhere in the current tree. Reproduced
live on both this pass's freshly-created documents and the three pre-existing `ws:acme` documents
from earlier stages' own testing (weeks old, fully processed — `523` pre-existing entities before
this pass even started — all three still read `status: "processing"`).

**Root cause:** the plan's Stage 3 file list never actually named the code change (only its Stage
1 "Done" text and §5's "additional, non-AC-mapped" bullet describe the intended behavior); no
later stage's diff picked it up either, and no gate flagged it as a gap because the offline test
suite's own AC-8 coverage (`test_batch_ingest_two_documents_mentioning_the_same_entity_fuse`) works
around the missing signal by completing extraction *synchronously in the test*, never actually
polling `status`. This pass is the first to drive the real, asynchronous background pipeline
end-to-end and observe that the promised signal is silent.

**Suggested fix:** have `IngestionPipeline`'s per-chunk background completion (or a simple
per-document counter/completion check) `SET d.status = 'ready'` once every chunk's
extraction+embedding has run, and `'failed'`/`'partial'` on isolated failure, exactly as
Stage 3's own §5 "Background-job failure isolation" bullet already specified. Not fixed by this
pass (implementation, out of `qa-engineer`'s lane) — recommend routing to `coder`/`tdd-engineer`
as a fast follow; does not block M5 closing since no AC depends on it.

### Defect 2 — `docs/DESIGN.md` §5.1/§7.1 do not reflect the shipped M5 schema

**Severity: Medium (documentation currency, not a runtime defect).** The K-050 done-condition
(`docs/plans/document-ingestion-coordination.md`) requires "DESIGN §5.1/§7 ... updated in the same
changes" for M5 to close; this pass's brief explicitly asked this to be checked rather than
assumed. It is stale.

**§5.1 (arrow notation, `docs/DESIGN.md:199-207`)** still shows only the pre-M5 dormant shapes:
```
(:Document {documentId})-[:HAS_CHUNK]->(:Chunk {chunkId, text, embedding: vecf32})
(:Chunk)-[:DERIVED_FROM]->(:Message)
(:Entity {entityId, name, type})<-[:MENTIONS]-(:Message)
(:Chunk)-[:ABOUT]->(:Entity)
```
Missing entirely: `(:Entity)-[:RELATES_TO {label, sourceChunkId, sourceDocumentId, createdAt}]->
(:Entity)` (FR-6, live-verified this pass — TP-04/TP-05) and `(:Entity)-[:SAME_AS {matchId,
status, confidence, technique, createdAt, decidedAt, decidedBy, resuggestCount,
lastResuggestedAt}]->(:Entity)` (FR-7..FR-10, live-verified this pass — TP-06/TP-07/TP-08/TP-09).
`Entity.nameNormalized` is also not mentioned anywhere in §5.1's "Key properties" list.

**§7.1 (index/constraint table, `docs/DESIGN.md:527-559`)** lists `Entity.entityId` (the identity
constraint) but is missing every M5-added index: `Entity.nameNormalized` (RANGE, Stage 3),
`Entity.name` (RediSearch fulltext, Stage 3 — the "Full-text index" bullet at line 558 names only
`Message.text`), and both `SAME_AS`-relationship-scoped indexes plus its `matchId` uniqueness
constraint (Stage 4) — all four confirmed live and in active use by this pass's own fusion-tier
tests (TP-06/TP-07).

**Both sections are accurate and current in `docs/QUERIES.md` §14/§14.5/§14.6** and
`scripts/bootstrap_schema.sh` (the actual DDL) — this is specifically a `DESIGN.md` currency gap,
not a missing capability or an undocumented-anywhere gap.

**Not fixed by this pass** — per the dispatch brief, `DESIGN.md` staleness routes back to the
coordinator/`architect` rather than a QA-authored edit; the fix is substantive (new arrow-notation
blocks + table rows), not the "trivial factual correction squarely in `qa-engineer`'s lane"
exception. Flagging here as the done-condition check the brief asked for.

## Environment notes (not `falkor-chat` code defects — recorded for the record)

- **Stale LM Studio gateway IP in the shared `~/.config/opencode/opencode.json`.** The pristine,
  out-of-repo config `start_server.sh` uses by default has `"baseURL": "http://192.168.0.69:1234"`
  for the `lmstudio` provider — refused from this WSL2 box (`ProviderCallError: ... connection
  failed: [Errno 111] Connection refused`), while `http://localhost:1234` is reachable. This
  matches a kaizen entry already filed during Stage 3's `data-scientist` checkpoint (U18) for the
  same stale IP, still unfixed as of this pass. Worked around by restarting the server with
  `FALKORCHAT_OPENCODE_CONFIG` pointed at a corrected local copy (`baseURL:
  http://localhost:1234`, otherwise identical); the shared file itself was **not** modified.
- **LM Studio concurrent extract+embed thrashing, reproduced repeatedly this pass, not just read
  about.** Firing two-or-more concurrent background chunk-processing jobs (the normal shape of
  this app's own per-chunk scheduling, `background._schedule_chunk_processing`) against this
  LM Studio instance intermittently produced `HTTP 400 {"error":"Model is unloaded."}`,
  `HTTP 500 Internal Server Error`, and embedding `TimeoutError`s — `text-embedding-qwen3-
  embedding-0.6b` and `qwen/qwen3-4b-2507` appear to compete for load slots under concurrent
  requests. This is the same class of instability `data-scientist`'s U18 checkpoint already filed
  ("LM Studio concurrent-model-swap thrashing under combined embed+extract background load") — not
  a new finding, but this pass reproduces it directly (not just derives it from the math the way
  the dispatch brief's MCP-thread-fan-out caveat anticipated) and confirms it is severe enough to
  require 2-3 retries on 3 of this pass's live test items (AC-2's Northbridge pair needed a retry,
  AC-3's Talbridge pair needed a retry, AC-8's batch needed two retries before landing cleanly).
  `_safe_extract`/`_safe_embed_chunk`'s try/except-log-never-raise discipline correctly isolated
  every one of these failures (no `Document`/sibling-chunk corruption observed, matching the
  design intent) — the isolation mechanism itself is not in question, only the underlying local
  model server's stability under this app's concurrency shape. No server crash, no thread
  exhaustion, and no MCP-thread-fan-out-specific instability was observed at the batch sizes this
  pass used (2-item batches, well under `MAX_BATCH_SIZE=20`) — the already-accepted ~23,000-thread
  scaling concern named in the dispatch brief was not re-triggered or newly implicated.
- **`uvicorn --reload` (this repo's `start_server.sh` default) restarts the server on any file
  write inside the `falkor-chat/` tree it watches — including this pass's own test-plan/test-
  report documents under `falkor-chat/docs/`** — killing every in-flight background
  extraction/embedding daemon thread. Discovered mid-pass (writing the test plan mid-run silently
  dropped several in-flight jobs); worked around for the rest of this pass by restarting with
  `UVICORN_ARGS="--timeout-keep-alive 5"` (any non-empty override defeats
  `start_server.sh`'s `UVICORN_ARGS="${UVICORN_ARGS:---reload}"` default — an **empty**
  `UVICORN_ARGS=""` override does not, since bash's `:-` treats empty and unset the same way).
  Worth a kaizen note for any future QA/live-driving pass against this server: don't edit files
  under `falkor-chat/` while background jobs are in flight unless `--reload` is disabled first.

## Coverage & gaps

**Covered**: all ten acceptance criteria, both directions of cross-transport (MCP→REST,
REST→MCP), both new-to-Stage-6a bulk-API-altitude claims for AC-8 (real MCP `ingest_documents`
call, cross-document fusion), the full confirm/reject/recheck contract with audit-field
verification, a live re-confirmation of AC-5's chat-grounding path, and a milestone
done-condition check (`DESIGN.md` currency) the brief asked for explicitly.

**Not covered / deliberately out of scope this pass** (see the test plan's "Deliberate scope
calls" for the reasoning on each):
- AC-7's *automatic* re-open path (OQ-3 path 1) was not forced through real live extraction —
  corroborated via the existing, Stage-4-gated repository-level test instead. This is a technique
  limit (manufacturing the exact entity-id-pair precondition through a real, non-deterministic
  local model is not a reliable acceptance-test method), not a gap in the shipped mechanism.
- No MCP/REST "list a document's entities" read surface exists (`search_documents` returns ranked
  chunks only) — AC-10's read-side assertion is direct Cypher, matching the plan's own "repository/
  service integration" altitude for that AC; the write is still driven through the real server.
- Re-deriving unit-level coverage the offline suite (1751 passed/4 deselected, independently
  gated stage-by-stage) already proves.
- Load/concurrency/performance testing beyond the existing gated concurrency regression test.
- The MCP per-chunk background-thread fan-out's scaling characteristic (documented, accepted,
  non-blocking) — not re-litigated; this pass's batch sizes (2 items) never approached it.
- OQ-1's fusion-default methodology — a confirmed v1 design decision, not re-opened.
- Web UI — out of scope for this feature (plan §1).

**Residual risk: low.** Both parked defects are non-blocking for every AC. Defect 1 (status stuck)
means a real integration relying on `Document.status` to know when ingestion finished has no
working signal today — a real but bounded risk since FR-5's actual consumers (agents reading via
MCP/REST) can still poll `get_document`/`search_documents` for the content itself. Defect 2 means
a reader of `DESIGN.md` alone would not learn the fusion schema exists — bounded by `QUERIES.md`
already being complete and correct.

## Feedback & recommendations

1. **Fix Defect 1** — wire `Document.status` to `'ready'`/`'failed'` at background-pipeline
   completion, exactly as the plan's own Stage 1/Stage 3 text already specifies. Cheap (a
   completion-aware `SET`, mirroring `_safe_*`'s existing per-chunk isolation) and closes a gap
   that has silently persisted across all six stages because the offline test suite's own
   AC-8 coverage never needed to poll for it (it completes synchronously in-test).
2. **Fix Defect 2** — bring `DESIGN.md` §5.1/§7.1 current with `RELATES_TO`/`SAME_AS`/
   `nameNormalized`/the new indexes, mirroring what `QUERIES.md` §14 already documents correctly.
   Routing back to the coordinator per this pass's brief rather than editing `DESIGN.md` directly.
3. **Testability win worth keeping**: this feature's dual-transport (REST/MCP) design made
   cross-transport acceptance testing (AC-6) straightforward with a real out-of-process client —
   no bespoke test harness needed beyond the standard `mcp` SDK's `streamablehttp_client`.
4. **Environment robustness, not this feature's bug**: the LM Studio concurrent-model-swap
   thrashing (see "Environment notes") is real and reproducible on demand under this app's own
   normal per-chunk background concurrency, not just a math-derived scaling concern — worth
   escalating the existing kaizen entry's priority now that a live QA pass has hit it directly and
   needed retries to get clean signal, even though it did not corrupt any data or destabilize the
   server itself.
5. **No testability issues found in the shipped API surface** — every AC had a concrete,
   checkable REST/MCP response or (for AC-1/AC-10, by design) a direct graph read the plan itself
   specifies at that altitude; no claim required guessing at internal state.

## Milestone done-condition assessment

Per `docs/plans/document-ingestion-coordination.md`'s stated M5 done-condition ("all six
implementation stages delivered and analyst-gated, `qa-engineer` acceptance PASS (or
PASS-with-parked-defects) on green baselines, DESIGN §5.1/§7 and this component's docs updated in
the same changes ⇒ M5 ✅"):

- Six stages delivered + analyst-gated: **✅** (coordination ledger, U9-U29).
- `qa-engineer` acceptance: **✅ PASS-with-parked-defects** (this report).
- Green baselines: **✅** (offline suite 1751/4 deselected, query suite 320/320, both before and
  after this pass).
- `DESIGN.md` §5.1/§7 updated: **❌ not done** — Defect 2, above. This is the one open item against
  the done-condition as stated; flagged to the coordinator rather than resolved here.

## Artifacts

- Test plan: `falkor-chat/docs/test-plans/document-ingestion.md`
- This report: `falkor-chat/docs/test-reports/document-ingestion-report.md`
