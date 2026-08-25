# Ingestion Pipeline & Entity Fusion — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-050 (M5, proposed)

Coordinating the design → implementation → QA flow for the ingestion pipeline & entity fusion
feature, kicked off from `falkor-chat/docs/requirements/document-ingestion.md` (Status: Ready for
design, confirmed by the stakeholder 2026-08-22).

## Context

- Requirements: `falkor-chat/docs/requirements/document-ingestion.md` — FR-1..FR-14, AC-1..AC-10,
  OQ-1..OQ-3 open (explicitly non-blocking for handoff).
- Dormant schema already scaffolded for this feature: `(:Document)-[:HAS_CHUNK]->(:Chunk)`,
  `(:Chunk)-[:ABOUT]->(:Entity)`, `Chunk.embedding` vector index — `docs/DESIGN.md` §5.1,
  `scripts/bootstrap_schema.sh`, never populated until now.
- No existing plan/backlog item for this feature. Next free backlog id: **K-050**. Next milestone
  slot: **M5** (M4 closed 2026-08-11).
- CPG freshness (checked 2026-08-22, per `skills/cpg-analysis/references/freshness.md`):
  `cpg_falkorchat`, built `2026-08-17T00:40:42Z`, scratch-copy build (`sourceCommit` null,
  `sourcePath=/tmp/cpg-src/falkor-chat-server` → real counterpart `falkor-chat/server`) — **stale**:
  6 commits touched `falkor-chat/server` since build (K-028 timers, K-027 items 3-5, K-046/K-047).
  Flagged to each dispatched specialist rather than trusted silently.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a6134db58779dae0b` | accepted | `docs/plans/document-ingestion.md` | `analyst` → **approve** (Pass 2) | 214k tok / 26 tools |
| U2 | `graph-dba` | `af233df4cee5889c0` | accepted | `docs/plans/document-ingestion-graph.md` | `analyst` → **approve** (Pass 2) | 242k tok / 43 tools |
| U3 | `data-scientist` | `a7118bfb884d3eccf` | accepted | `docs/plans/document-ingestion-ml.md` | `analyst` → **approve** (Pass 2) | 111k tok / 18 tools |
| U1b | `architect` | `a6134db58779dae0b` | accepted | reconciled `document-ingestion.md` §0/§2.5/§3.4/§4/§5/§6/§7 | (verified by teco) | 277k tok / 29 tools |
| U4 | `analyst` | `afe09c0e1480da147` | gated | `docs/reviews/document-ingestion.md` | plan gate → **needs changes** | 158k tok / 30 tools |
| U5 | `architect` | `aebd2b8e92466e5fc` | accepted | fix `document-ingestion.md` (blocker + major #1/#3b + minors) | `analyst` → **approve** (Pass 2) | 132k tok / 25 tools |
| U6 | `data-scientist` | `a7118bfb884d3eccf` (resumed) | accepted | fix `document-ingestion-ml.md` (major #2 terminology) | `analyst` → **approve** (Pass 2) | 123k tok / 17 tools |
| U7 | `graph-dba` | `af233df4cee5889c0` (resumed) | accepted | fix `document-ingestion-graph.md` (blocker mechanism + major #3a + minor #3) | `analyst` → **approve** (Pass 2) | 310k tok / 35 tools |
| U8 | `analyst` | `a22557e8972eec926` | accepted | `docs/reviews/document-ingestion.md` Pass 2 | plan re-gate → **approve** | 127k tok / 10 tools |
| U9 | `coder` | `a1d788e3e8868b48a` | accepted | Stage 1: chunking + Document/Chunk write path | `analyst` (U10) → **approve w/ suggestions** | 284k tok / 137 tools |
| U10 | `analyst` | `a1f9d0b8d1f75e6cc` | accepted | `docs/reviews/document-ingestion-impl.md` | code gate → **approve with suggestions** | 126k tok / 45 tools |
| U11 | `coder` | `a1d788e3e8868b48a` (resumed) | accepted | fix: reject empty/whitespace-only `ingest_document` text | (verified by teco) | 303k tok / 26 tools |
| U12 | `coder` | `a5db435e79c550371` | accepted | Stage 2: chunk embeddings + standalone search (FR-3) | `analyst` (U13) → **approve** | 344k tok / 19 tools |
| U13 | `analyst` | `a1854598c0cc66618` | accepted | `docs/reviews/document-ingestion-impl.md` Pass 2 | code gate → **approve** | 141k tok / 52 tools |
| U14 | `coder` | `aa3b282bce11b9f3b` | accepted | Stage 3: extraction (FR-7a) | `analyst` (U15/U17) → **approve** (re-gate) | 318k tok / 161 tools |
| U15 | `analyst` | `a451eafaf56d89c64` | accepted | `docs/reviews/document-ingestion-impl.md` Pass 3 | code gate → **approve w/ suggestions** | 166k tok / 66 tools |
| U16 | `coder` | `aa3b282bce11b9f3b` (resumed) | accepted | fix U15's 2 MAJORs (cap/stub-repair ordering; MCP schedule try/except + doc the doubled fan-out) | `analyst` (U17) → **approve** | 357k tok / 43 tools |
| U17 | `analyst` | `a451eafaf56d89c64` (resumed) | accepted | Pass 3 re-gate of U16's fixes | code re-gate → **approve** | 194k tok / 31 tools |
| U18 | `data-scientist` | `a454015041210268b` | accepted | `docs/reviews/document-ingestion-ml.md` — checkpoint | advisory, non-blocking — no gate (verified by teco) | 228k tok / 149 tools |
| U19 | `coder` | `a7d65e37255f8b2ca` | accepted | Stage 4: fusion (FR-6/7/8/9/10, OQ-1/2/3) | `analyst` (U20, Pass 4) → **approve w/ suggestions** | 359k tok / 171 tools |
| U20 | `analyst` | `a40241fa3792bc3cf` | accepted | `docs/reviews/document-ingestion-impl.md` Pass 4 | code gate → **approve w/ suggestions** | 151k tok / 70 tools |
| U21 | `coder` | `a7d65e37255f8b2ca` (resumed) | accepted | fix U20's 2 MINORs (QUERIES.md §14.6; AC-8 pipeline-altitude test) | (verified by teco) | 399k tok / 26 tools |
| U22 | `coder` | `a230563f056d3d9de` | accepted | Stage 5: chat-grounding integration (FR-2) | `analyst` (U23, Pass 5) → **needs changes** (fixed U24) | 230k tok / 123 tools |
| U23 | `analyst` | `a21a439e02926453a` | accepted | `docs/reviews/document-ingestion-impl.md` Pass 5 | code gate → **needs changes** | 138k tok / 52 tools |
| U24 | `coder` | `a230563f056d3d9de` (resumed) | accepted | fix U23's BLOCKER (tools.py `GraphragRetrieveTool` KeyError on Chunk seed) + 2 MINORs (eval-harness same pattern; AC-5 live-e2e gap) + NIT (tie-break/latency doc note) | `analyst` (Pass 5 re-gate, U25) → **approve** | 319k tok / 59 tools |
| U25 | `analyst` | `a21a439e02926453a` (resumed) | accepted | Pass 5 re-gate of U24's fixes | code re-gate → **approve** | 178k tok / 29 tools |
| U26 | `coder` | `aa1d16116dbc910fe` | delivered | Stage 6a: `ingest_documents` (MCP+REST+service), shared `_schedule_chunk_processing`, batch AC-8 test, QUERIES.md §14.4 | `analyst` (Pass 6) → — | 193k tok / 108 tools |
| U27 | `analyst` | `a1eaeac5da1a10786` | accepted | `docs/reviews/document-ingestion-impl.md` Pass 6 | code gate → **needs changes** (1 BLOCKER, 1 MAJOR, 1 MINOR) | 153k tok / 55 tools |
| U28 | `coder` | `aa1d16116dbc910fe` (resumed) | delivered | fix Pass 6 BLOCKER (malformed-item isolation) + MAJOR (fan-out doc note) + MINOR (HISTORY.md) | `analyst` (Pass 6 re-gate) → — | 228k tok / 136 tools |
| U29 | `analyst` | `a1eaeac5da1a10786` (resumed) | queued | Pass 6 re-gate of U28's fixes | code re-gate → — | — |

U1 verified: plan reads through all FR-1..FR-14 with a coherent staged sequence (6 stages),
correctly reconciles the dormant `Document`/`Chunk`/`Entity` schema, resolves OQ-2/OQ-3, proposes
an explicitly-flagged OQ-1 default, and delegates the two ML/graph-modeling axes cleanly. Also
verified the `BACKLOG.md` diff (M5 row + K-050 item) — format matches the K-042/M4 precedent,
correctly marked 🟡 in-progress, correctly cites both not-yet-authored follow-on notes as such.
Not yet committed — will commit alongside U2/U3 once the design set is complete, one coherent
design-phase commit.

## Notes from delivered units

- **U3 (`document-ingestion-ml.md`) verified**: agrees with the plan's LLM-extraction and
  deterministic-exact-match-auto-merge defaults, with concrete refinements (closed 7-value entity-
  type taxonomy, app-side schema validation, a stub-entity repair rule for dangling relationship
  references) and one genuine bug find (F1: `llm.extract_own_line_json_object`'s `require_key`
  check only fires on its "prose amid text" branch, not the "whole reply is one JSON object" branch
  extraction will actually hit — `extraction.py` must validate shape itself). Recommends **against**
  embedding-based matching for v1 at any tier (cost + no calibration data), with a scoped v2
  evaluation design recorded for later. **Flags a follow-up gate, not yet in the plan's stage
  table**: once stage 3 (extraction) is live, `data-scientist` should do a quick qualitative read of
  20-30 real extraction outputs before stage 4 (fusion) is trusted with what extraction produced —
  fold this in as a stage-3→stage-4 checkpoint when sequencing implementation units, not a formal
  gate with its own ledger row unless it surfaces something.

- **U2 (`document-ingestion-graph.md`) verified**: all Cypher live-verified against throwaway probe
  graphs on the real `falkordb-dev` instance (never `reference`/`ws:*`), a genuinely new write shape
  proven (`FOREACH` over a parameter list-of-maps), a real planner trap found and documented (a bare
  label on a `SAME_AS`-anchored query endpoint forces a full label scan — worked around in every
  query in the note), settles (b)/(c)/(d) fully, and **diverges from the plan on (a)**: recommends a
  plain `(:Entity)-[:SAME_AS {...}]->(:Entity)` edge over the plan's `MatchSuggestion` node, with the
  plan's own explicitly-flagged-unverified blocker (does this build support indexed relationship
  properties) now live-verified **yes** — RAM measured as a wash between the two shapes (not assumed),
  decided instead on hop count for the eventual "expand to fused siblings" read and write-path fit.
  This is exactly the kind of call the plan deferred to graph-dba (§0(a): "graph-dba's -graph note
  makes the final call"), so it's expected, not a defect — but it leaves `document-ingestion.md`
  §3.4/§4 (stage 4)/§7 textually describing the superseded `MatchSuggestion` node design. Dispatched
  U1b (resuming the same architect agent) to reconcile the plan's prose to the final `SAME_AS` edge
  schema before the U4 plan gate, rather than let `analyst` file it as a findable inconsistency and
  cost a round trip.

- **U4 (plan gate) verdict: needs changes.** 1 blocker + 4 major + 4 minor findings
  (`docs/reviews/document-ingestion.md`). Blocker: FR-8's exact-tier auto-merge has an unguarded
  check-then-act race across concurrent extraction (candidate lookup + entity/edge creation are
  separate `GRAPH.QUERY` round trips, no atomic guard) — can silently defeat "no confirmation
  required" for the exact pair it's supposed to catch, on the one fusion action with zero human
  review. Majors: a stale §3.7 reference never reconciled to graph-dba's actual `EMITTED` resolution
  design; the ML note's 6 leftover `MatchSuggestion` references post-schema-reconciliation; no
  audit/discovery surface for auto-confirmed matches + the ML note's stage-3→4 qualitative-review
  checkpoint missing from the plan's own stage table. Minors: missing `_safe_extract`/`_safe_fuse`-
  style file-list entries, an unexamined `MAX_DOCUMENT_CHARS`×`MAX_BATCH_SIZE` compounding ceiling,
  an unwritten/unverified §10.3 reverse-read generalization, `test_provenance.py` not named in Stage
  5's file list. Grounding, FR/AC coverage, and the schema-reconciliation process itself were called
  out as strong. Routed to owners below (U5/U6/U7), then a re-gate (U8).

- **Process note (self-correction):** U5 should have been a `SendMessage` resume of the original
  `architect` agent (`a6134db58779dae0b`, still the ledger's "owner of record" for this plan) rather
  than a fresh `Agent` dispatch — caught after the fact, not before. The brief was fully
  self-contained (per the standing brief-isolation rule) so this shouldn't affect correctness, only
  efficiency (a fresh spawn re-reads context the original agent already had loaded). Not repeating
  for U6, which correctly resumed `a7118bfb884d3eccf`.

- **U6 verified**: exhaustive sweep confirms zero remaining `MatchSuggestion`/`CANDIDATE_A`/
  `CANDIDATE_B` references in `document-ingestion-ml.md`. Flagged a nuance to check once U5 lands:
  the stage-3→4 checkpoint should read as "a gate on trusting stage 4's *input*," not "stage 4
  cannot start until this review lands" — the plan's own precedent already allows fusion code to
  exist/be tested against synthetic extraction output before the checkpoint runs. **Check this when
  reviewing U5's wording.**

- **U5 verified**: chose blocker fix (a) — atomic combined query — grounded in actually reading
  `background.py` before deciding (confirmed no serialization point exists for option (b), would
  need new machinery). Clean, precise handoff spec for `graph-dba`'s `create_entity_with_auto_match`
  (inputs, 3-step atomic behavior, the correctness-critical MATCH-before-CREATE ordering guarantee
  flagged as needing live verification, and an explicit simplification — no reopen-branch needed —
  for graph-dba to confirm rather than blindly copy). §3.7 rewritten correctly (bare-id `coalesce`,
  both original options marked rejected). Checkpoint section wording verified against U6's
  preservation request — reads "advisory, not a hard gate... Stage 4 implementation can proceed in
  parallel," exactly as asked. All three minors applied. Dispatching U7 (`graph-dba`) now with the
  atomic-query spec.

- **U7 verified**: the atomic `create_entity_with_auto_match` query is exactly what the blocker
  needed, and the ordering guarantee was genuinely verified two ways (a self-match-ruling-out
  behavioral test + a `GRAPH.PROFILE` structural proof), not assumed — matches the review's own
  standard for this codebase. The reopen-branch simplification was confirmed, not just copied.
  `list_matches` caught a real second trap along the way (an `IS NULL OR` filter silently defeats
  the relationship index even when bound to a real value) and fixed it by branching at the
  repository layer — a genuinely new finding, added to `falkordb-quirks.md`. §10.3 reverse-read
  completed and live-verified, including checking (not assuming) it doesn't hit the §1.4 label-scan
  trap given its different shape. All three items closed to the same rigor as the original note.
  Dispatching U8 (`analyst` re-gate) now. **Process note (self-correction, second occurrence):**
  U8 should also have been a `SendMessage` resume of the original analyst (`afe09c0e1480da147`)
  rather than a fresh `Agent` dispatch — same lapse as U5, caught after the fact again. Brief was
  self-contained; flagging as a pattern worth a kaizen entry, not just a one-off.

- **U9 verified independently, not just on report**: read the `chunking.py`/`repository.py` diffs
  directly (Cypher matches `document-ingestion-graph.md` §2.1 verbatim), re-ran the new/related
  tests (35 passed) and the **full** offline suite myself (1563 passed / 3 deselected — matches
  reported), and the query suite (320/320 — matches reported, unchanged as expected since no new
  DDL entered it). Both runs wiped the shared `reference` graph at teardown as documented
  (`falkor-chat/AGENTS.md`); re-seeding a bare `seed_workflows.sh` after `test_queries.sh`
  specifically **failed** ("Invalid graph operation on empty key" — `GRAPH.DELETE` removes the key
  entirely, not just its contents) until `bootstrap_schema.sh acme` ran first to recreate the graph
  key; `verify_workflows.sh` confirms in-sync after the full bootstrap→seed sequence. Worth a kaizen
  note for future coordination sessions hitting the same teardown gotcha back-to-back.
  Mutation-testing was real (5 deliberate breaks, each caught by the right test, all reverted via
  targeted `Edit` — including recovering cleanly from an accidental `git checkout` that wiped
  in-progress edits mid-mutation-test). Three named deviations from the plan's exact wording are all
  reasonable, routine judgment calls, not scope changes. Dispatching U10 (`analyst` diff-scoped
  re-gate) now.

- **U10 verdict: approve with suggestions.** One MAJOR, real, live-verified finding: `ingest_document`
  silently accepts empty/whitespace-only text (only `MAX_DOCUMENT_CHARS` is checked), creating a
  `Document` node with zero `Chunk`s permanently stuck at `status:'processing'` — contradicts
  `chunking.py`'s own docstring claim that the service rejects this upstream, and is reachable via
  MCP with no schema layer at all. Not a blocker (doesn't corrupt data, doesn't violate a tested AC)
  but cheap to fix and would otherwise let later stages (extraction/fusion) waste work on empty
  documents. Everything else — Cypher fidelity, the three self-reported deviations, scope discipline,
  test quality (own independent mutation-test spot check performed), conventions fit — confirmed
  solid. Deciding to fix now rather than defer to Stage 6 (cheap, and avoids compounding into later
  stages) — dispatching U11 (resuming `coder`).

- **U12 verified independently, not just on report**: run was interrupted mid-task by a platform
  session-limit error (not a deficient result — resumed the same agent via `SendMessage` per
  standing practice rather than re-dispatching cold; its work was already substantially on disk
  and it confirmed nothing was mid-sentence before finishing verification). Read the full
  `embedding.py`/`repository.py`/`services.py`/`api.py`/`mcp.py`/`app.py` diffs directly:
  `embed_chunk`/`_resolve_and_embed` correctly factor out the FR-19 dimension guard while keeping
  `Message`/`Chunk` gated independently on their own index only; `search_chunks` mirrors
  `hybrid_search`'s ANN shape with no scope traversal/Entity expansion (correctly deferred, `ABOUT`
  is dormant until Stage 3); the `/documents/search` route is registered before
  `/documents/{document_id}` with a correct explanation (Starlette registration-order matching —
  a real bug the agent caught and fixed during the build, not left for QA). Re-ran the full offline
  suite myself (1597 passed / 3 deselected, +31 over Stage 1's 1566 — matches reported) and the
  query suite (320/320, unchanged as expected — no new DDL), re-seeding `reference` afterward
  (`bootstrap_schema.sh acme` → `seed_demo.sh acme` → `seed_workflows.sh acme` →
  `verify_workflows.sh acme`: OK). Read `docs/HISTORY.md`'s new entry and `docs/QUERIES.md` §14.3/
  §14.4 in full — both complete and accurate, not truncated by the earlier interruption. Two
  reasonable, narrowly-scoped deviations from the brief (both self-disclosed): `search_chunks`
  omits Entity co-occurrence expansion (deferred, not speculative); added an internal
  `list_document_chunks` seam (repository + service) rather than changing `ingest_document`'s
  documented response shape. Dispatching U13 (`analyst` diff-scoped code gate) now.

- **U13 verdict: approve.** No blockers, no majors. One MINOR (real, not a correctness bug):
  `mcp.py`'s `ingest_document` now spawns one raw daemon thread per chunk synchronously in the tool
  handler before returning — up to ~500 threads for a max-size document, a ~500x amplification of an
  already-accepted "1 thread per MCP call" trade-off, undocumented at the new scale. REST's
  `BackgroundTasks`-based equivalent doesn't have this amplification. Explicitly flagged non-blocking
  (lab-scale, self-contained, consistent with existing accepted posture) — **deferring to Stage 6
  (batch hardening)** rather than patching now, since this is exactly the concern that stage exists
  to address and the review itself offers cheap/thorough options as Stage 6 implementer's call, not
  an immediate fix. One NIT (MCP `search_documents` has no `limit` cap, matching pre-existing
  `search_messages` precedent) — no action needed, not a regression. All mutation-test claims
  spot-checked against actual test assertions (not just labels) and confirmed real.

- **U14 verified independently, not just on report**: read `extraction.py`/`ingestion.py` in full
  and the `repository.py`/`background.py`/`api.py`/`mcp.py`/`app.py` diffs directly. Confirmed:
  `normalize_name` is genuinely the one shared helper both `extraction.py`'s stub-repair and
  `ingestion.py`'s `create_entity` call use; the closed 7-value type enum coerces (not rejects) an
  out-of-enum/missing `type` to `Other`; `create_entity_relationship` is a plain `CREATE`, never a
  guarded `MERGE` (FR-6 never-deduplicated); `_safe_extract` mirrors `_safe_embed_chunk`'s
  try/except-log-never-raise shape exactly; no fusion/matching/lookup-existing-entity logic is
  present anywhere (correct Stage 3/4 boundary — every path is unconditional create); extraction and
  embedding are scheduled as two independent background calls per chunk on both transports, neither
  chained to the other; `config/models.json`'s new `extraction` kind resolves generically through
  `ModelGateway.resolve()`/`default_for(kind)` with no closed-KINDS check blocking it (confirmed by
  reading `modelconfig.py` directly — the implementer's scope note about deferring
  `WorkspaceConfig.extractionModelOverride` wiring is correct: not named in Stage 3's file list, and
  `_workspace_override_ref` degrades to "no override" for a kind absent from `_KIND_TO_OVERRIDE_KEY`,
  not an error). Re-ran the full offline suite myself (1647 passed / 3 deselected, +50 over Stage
  2's 1597 — matches reported) and the query suite (320/320, unchanged as expected — no new DDL
  entered that suite), re-seeding `reference` afterward (`bootstrap_schema.sh acme` →
  `seed_demo.sh acme` → `seed_workflows.sh acme` → `verify_workflows.sh acme`: OK). Read
  `docs/HISTORY.md`'s new entry and `docs/QUERIES.md` §14.5 in full — both complete and accurate.
  Confirmed a genuinely new, live-verified FalkorDB quirk the implementer found and captured as a
  `KaizenEntry` in `kaizen_team` (producer `coder`): `MATCH ()-[:REL]->() RETURN count(*)`
  under-counts parallel edges between the same node pair (returns 1 for 2 identical edges) —
  `count(r)` with a bound relationship variable correctly returns 2; the never-deduplicated
  `RELATES_TO` tests use the correct `count(r)` form. One scope note self-disclosed by the
  implementer (workspace-override wiring for the new `extraction` kind, not built) verified as a
  reasonable, correctly-scoped omission, not a gap. **Compounding note carried into U15's brief**:
  Stage 3 adds a second per-chunk `threading.Thread` spawn on the MCP path (`_safe_extract`,
  alongside Stage 2's `_safe_embed_chunk`) — doubling U13's already-deferred MINOR (up to ~500 →
  ~1000 threads for a max-size document on MCP). Still judging this as correctly deferred to Stage 6
  rather than an immediate fix (same reasoning as U13: lab-scale, self-contained, exactly the class
  of concern that stage exists to address) — asked U15 for an independent second opinion on whether
  the compounding changes that severity/urgency judgment, not just a note that it exists. Dispatching
  U15 (`analyst` diff-scoped code gate, Pass 3) now.

- **U15 verdict: approve with suggestions.** Two real MAJORs, both cheap to fix, neither a blocker:
  (1) the entity-cap/stub-repair truncation order silently drops a relationship fact when a chunk's
  raw entity list is already at the 20-cap and a relationship references a not-yet-listed name — the
  exact failure stub-repair exists to prevent (FR-6), live-reproduced by the reviewer both via a
  direct `extract()` call and end-to-end through `IngestionPipeline`, no existing test covers this
  interaction; (2) confirmed and independently judged my carried-forward compounding question — the
  reviewer's own verdict is that the doubling (now ~1,000-1,200 MCP threads for a max-size document)
  is not just "more of the same accepted trade-off" but introduces a genuinely new, currently-uncaught
  failure mode (`_schedule()`'s bare `threading.Thread(...).start()` has no try/except, so a
  thread-creation failure under resource pressure now propagates unhandled out of the tool call) —
  and recommends the cheapest of three fixes (a try/except wrap) now, deferring the fuller
  batching/pooling redesign to Stage 6 same as before. Everything else — Cypher fidelity,
  shared-normalizer discipline, enum coercion, stage-boundary discipline, scheduling independence, all
  6 mutation-test claims, docs accuracy, suite counts (1647/3 deselected, 320/320 query suite) — spot-
  checked and confirmed clean. Deciding to fix both MAJORs now (same precedent as Stage 1's U10→U11):
  cheap, correctness-critical for #1 (FR-6 is a load-bearing guarantee), and the reviewer's own
  judgment on #2 is that it shouldn't keep sliding to Stage 6 unexamined a second time — resuming the
  same `coder` agent (`aa3b282bce11b9f3b`) via `SendMessage` as U16, scoped to the gate's own cheapest
  suggested fixes (fix #1's ordering bug + a new test; fix #2's try/except wrap + doc the doubled
  count), explicitly NOT building the heavier Stage-6-deferred batching/pooling redesign.

- **U16 verified independently, not just on report**: read both diffs (`extraction.py`, `mcp.py`) in
  full — the entity cap now runs before `_repair_stub_entities` (not after), and `_default_schedule`'s
  `threading.Thread(...).start()` is wrapped in the same try/except-log-never-raise shape every other
  `_safe_*` isolation point in this codebase uses, with the doubled fan-out documented explicitly in
  both the docstring and a new `QUERIES.md` §14.5 "Resource note." Re-ran the full offline suite
  myself (1650 passed / 3 deselected, +3 over the pre-fix 1647) and the query suite (320/320
  unchanged), re-seeding `reference` afterward (full bootstrap→seed→verify sequence, confirmed OK).
  Read both new regression tests (`test_extract_stub_repair_is_not_truncated_away_by_the_entity_cap`,
  `test_default_schedule_swallows_a_thread_start_failure_and_logs` +
  `test_default_schedule_still_runs_the_job_on_the_happy_path`) and confirmed they genuinely exercise
  the fixed behavior, not just restate the mutation's own description. Dispatching U17 (resuming
  `analyst`, `a451eafaf56d89c64`) for the re-gate now — asking the reviewer to independently
  re-verify rather than accept the coordinator's own check, per standing practice.

- **U17 verdict: approve.** Both Pass 3 MAJORs confirmed genuinely closed — the reviewer re-ran their
  own original reproduction scripts against the fixed code (both now behave correctly), mutation-spot-
  checked both new regression tests by reconstructing the pre-fix behavior in memory and confirming
  each fails against it, confirmed no file outside the two fixes + their tests + the `QUERIES.md` note
  changed (no drive-by scope creep), and re-ran both suites themselves with matching counts (1650/3
  deselected, 320/320). One pre-existing documentation nit noted as still open, not urgent, no action
  needed: `modelconfig.KINDS`'s docstring says "four closed consumer kinds," now stale in spirit since
  `extraction` is a fifth defined kind — `KINDS` itself is a test-parametrization/documentation
  constant only, never a runtime validation gate, so this doesn't block anything. **Stage 3 is done:
  implementation + diff-scoped gate + fix + re-gate, all independently verified. Committing.**

- **U18 (checkpoint) verified independently, not just on report.** Read the delivered review
  (`docs/reviews/document-ingestion-ml.md`) in full. Independently re-queried `ws:acme` directly
  (`mcp__cypher__query`) rather than trusting the reported figures: `Entity` count 523 (matches),
  `RELATES_TO` count 371 via `count(r)` on a bound relationship variable (matches), and recomputed
  the headline 30% type-inconsistency statistic from scratch with my own Cypher (84 repeat-mention
  normalized names, 25 with >1 distinct `type`, 29.76% ≈ 30% — exact match) plus spot-checked one
  concrete example cited (`falkordb` → 9 mentions, `[Product, Organization]` — exact match). Both
  filed `kaizen_team` entries (LM Studio concurrent-model-swap thrashing under combined embed+
  extract background load; the stale `192.168.0.69:1234` gateway IP) confirmed genuinely present,
  correctly attributed to `data-scientist`. Confirmed the environment was left clean: no `uvicorn`
  process running, port 8000 unreachable, matching the review's own "stopped, no orphaned process"
  claim — the `ws:acme` graph data (README's 523 entities/371 edges) was left in place as disclosed,
  intentionally, as a usable starting corpus for Stage 4. Disclosed method deviation (Apollo 11
  sample run via direct `extraction.extract()` calls after live concurrent-background-task ingestion
  destabilized the local LM Studio instance) is judged reasonable and adequately flagged — same
  code path under review, decoupled only from the background-scheduling machinery, not from the
  extraction logic itself; the instability that forced it is exactly the same MCP/background
  thread-fan-out class of concern already tracked into Stage 6, now with a second data point.
  **Advisory-only, no gate, but the finding is real and load-bearing for Stage 4 planning**: FR-8's
  exact-match auto-merge tier will under-perform naive same-name-match intuition on real content
  from day one (30% of repeat mentions carry a type conflict, not a rare edge case) — this is now
  on record before Stage 4 design work starts, exactly what the checkpoint existed to surface. Two
  concrete, cheap follow-ups recommended (widen stub-repair to a same-call substring/containment
  match; flag the type-inconsistency rate to whoever tunes Stage 4's exact-match tier) — both
  explicitly scoped as Stage 4 design-time considerations, not built here, not blocking.

- **U19-U21 (Stage 4: fusion) verified independently, not just on report.** Read the diffs directly
  against `document-ingestion.md` §4/§3.4 and `document-ingestion-graph.md` §1.5-§1.8 rather than
  trusting `coder`'s summary: `create_entity_with_auto_match`, `create_or_reopen_match`,
  `confirm_match`/`reject_match`/`recheck_match`/`list_pending_matches`/`list_matches` all match the
  graph note's exact, live-verified Cypher (unlabeled `SAME_AS` endpoints throughout, `list_matches`'
  two-query-string fix, not a null-guard). Ran the default `pytest -q` suite myself (1711 passed on
  first delivery, matching the claim) and read the concurrency regression test (real threads +
  `threading.Barrier` + separate connections, not sequential calls dressed up as concurrent) and the
  `kaizen_team` entry it produced (a genuine new FalkorDB quirk: an undirected relationship pattern
  with an inline property filter silently degrades to directed — confirmed genuinely filed, correctly
  attributed). Dispatched `analyst` for the mandatory diff-scoped re-gate (Pass 4, U20, distinct from
  the design-phase plan gate) rather than accepting on `coder`'s word alone — independently
  re-verified Cypher fidelity, the self-match-filter necessity (live-probed, not assumed), the
  concurrency test (re-run 5×), one mutation-test spot-check, and the test-count arithmetic itself.
  Verdict: **approve with suggestions** — two MINORs (no `docs/QUERIES.md` §14.6 despite the diff's
  own code comments already citing it; AC-8's named pipeline-altitude test-strategy scenario not
  literally exercised, only proven at the repository-primitive altitude). Both routed back to the
  same `coder` agent (U21) rather than accepted as deferred follow-ups, since documentation is part
  of done and AC-8 is a named Stage 4 "Done" criterion. Re-verified U21's fix directly: read the new
  `docs/QUERIES.md` §14.6 section end-to-end (matches shipped `repository.py` exactly), read the new
  `test_ac8_two_documents_mentioning_the_same_entity_fuse_via_extract_chunk` test (correctly avoids
  the undirected+property-filter quirk via a directed, unlabeled probe), re-ran the full suite myself
  (1712 passed — the one new test, exactly as claimed), and confirmed a second, narrower kaizen entry
  (`REFINES` the first: the quirk trigger is any relationship-property predicate — inline or `WHERE`
  — on an undirected pattern, not just inline maps) was filed as a plain new node, not an
  unauthorized edge-write shape. **Stage 4 is done: implementation + diff-scoped gate + fix, all
  independently verified. Committing.**
- **U22-U25 (Stage 5: chat-grounding integration, FR-2) verified independently.** Dispatched
  `coder` for the generalized `EMITTED` write/read (per graph-dba's §3.1-§3.4, essentially
  verbatim) and an app-side `Message`+`Chunk` ANN merge in `services.hybrid_search` (no
  combined-ANN Cypher shape exists for this — a deliberate, documented implementer design
  choice, not a spec gap). Re-verified U22 directly before gating: read every diff hunk against
  the graph note's exact Cypher, ran the full offline suite myself (1723 passed, up from 1712),
  and independently reproduced one of the two implementer-claimed mutation tests. Dispatched
  `analyst` for the mandatory diff-scoped gate (Pass 5, U23) — verdict **needs changes**: a
  genuine BLOCKER (`services.hybrid_search`'s new merged row shape broke `GraphragRetrieveTool`,
  a shipped, workflow-granted tool at `tools.py:309-312` still doing unconditional `r["msgId"]`
  — `KeyError` on any `Chunk`-seeded hit; independently reproduced by both me and the analyst
  before/after the fix), plus 2 MINORs (the same fragile pattern in the eval harness;
  AC-5's plan-specified live-marked e2e variant absent with no note) and a NIT (undocumented
  merge tie-break/sequential-ANN-latency). Routed all four back to the same `coder` agent (U24)
  rather than deferred — the BLOCKER is a real regression in already-shipped functionality
  (`Chunk`s have existed since Stage 2), not a Stage 5 scope question. U24 fixed the BLOCKER
  (id resolved generically + `documentId` surfaced, mirroring `responder.py`), resolved MINOR 1
  (eval harness now filters to `Message`-shaped rows with a documented rationale) and MINOR 2
  outright rather than deferring it (`test_ac5_chat_grounding_live.py`, a new `pytest.mark.live`
  test mirroring `test_workflow_live.py`'s gating discipline), and folded in the NIT's doc note.
  Independently re-verified U24 myself: read every changed diff, re-ran the offline suite
  (1725 passed), **ran the new live-marked test myself** (LM Studio reachable — `1 passed in
  5.50s`), and reproduced the BLOCKER's fix via my own mutation test (reverted the id-resolution
  branch, confirmed the exact original `KeyError: 'msgId'` crash across all three relevant
  tests, restored). Dispatched the same `analyst` agent for the Pass 5 re-gate (U25) rather than
  accepting my own check as sufficient — it independently re-verified everything from scratch
  (including running the live test itself and re-grepping every `hybrid_search`/`search_chunks`
  call site across the codebase to confirm no other consumer was missed) and returned **approve**,
  with one non-blocking design note (the tool's new `documentId` is an opaque uuid, no
  `documentTitle` — reasonable for a scoped bug fix, flagged as a possible follow-up, not
  required). **Stage 5 is done: implementation + diff-scoped gate + BLOCKER/MINOR fixes +
  re-gate, all independently verified twice over (coordinator + analyst). Committing.**

## Design phase — ✅ complete, committed `30366f4`

Design phase (U1-U8) is done: three-document design set (main plan + graph note + ML note)
plan-gated by `analyst` — Pass 1 needs changes (1 blocker, 4 major, 4 minor), all routed to owners
and fixed, Pass 2 **approve**. Committed as one design-phase change: `30366f4` (plan, graph note,
ML note, review, `BACKLOG.md` K-050/M5 entries, `claude/graph-dba/falkordb-quirks.md` additions,
this coordination doc).

## Plan — implementation phase

1. ~~**U1-U8 — design phase**~~ ✅ complete, see above.
2. ~~**Stages 1-5**~~ ✅ complete, diff-gated, committed — see the ledger (U9-U25) and "Notes from
   delivered units" above.
3. **Stage 6 — in progress.** `teco`'s pre-dispatch orientation (2026-08-25) found a real scope gap,
   not just a test gap: **FR-11 bulk `ingest_documents` was never implemented in Stages 1-5** —
   `grep -rn "def ingest_documents" server/falkorchat/*.py` returns nothing, and neither `mcp.py`
   nor `api.py` register a plural/bulk route; only the singular `ingest_document` (Stage 1) exists.
   Plan §3.6 already specifies the design ("loops the single-document path per item, returning one
   receipt per item... no special batch-aware fusion logic is needed"), so this is implementation
   against an already-gated spec, not a design question. Stage 6 is therefore split:
   - **6a (U26, `coder`)** — implement `ingest_documents` (MCP tool + REST route + service method),
     looping `services.ingest_document` per item exactly per §3.6, mirroring the existing
     `ingest_document` background-scheduling block in both `api.py` and `mcp.py` (currently
     duplicated inline in each — factor into a shared helper rather than tripling the duplication).
     Confirm batch semantics under a real multi-document fixture with cross-document entity overlap
     (the AC-8 scenario, this time through the actual bulk endpoint — `test_ac8_...` in
     `test_ingestion.py` already proves the *fusion* wiring at pipeline altitude via direct
     `extract_chunk` calls, but not the batch API surface itself, background scheduling, or
     `Document.status` per item in a batch).
   - **6b (`analyst`)** — diff-scoped re-gate (Pass 6), same pattern as Stages 1-5.
   - **6c (`qa-engineer`)** — full acceptance pass against AC-1..AC-10, versioned test plan + report
     (`docs/test-plans/document-ingestion.md` + `-report.md`), mirroring the K-015/K-025/K-036
     pattern. Gates M5 ✅.
   - **CPG freshness (checked 2026-08-25):** `cpg_falkorchat`, built `2026-08-17T00:40:42Z`,
     scratch-copy (no `sourceCommit`) — **stale**: 13 commits have touched `falkor-chat/server`
     since build, including all of Stages 1-5 of this very feature. Flagged to 6a/6b rather than
     trusted silently; `coder`/`analyst` should read the actual files, not lean on `cpg-analysis`
     for structural claims about the new ingestion code.
