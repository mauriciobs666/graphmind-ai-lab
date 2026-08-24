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
| U12 | `coder` | `a5db435e79c550371` | delivered | Stage 2: chunk embeddings + standalone search (FR-3) | `analyst` (U13) → — | 344k tok / 19 tools |
| U13 | `analyst` | `a1854598c0cc66618` | accepted | `docs/reviews/document-ingestion-impl.md` Pass 2 | code gate → **approve** | 141k tok / 52 tools |

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

## Design phase — ✅ complete, committed `30366f4`

Design phase (U1-U8) is done: three-document design set (main plan + graph note + ML note)
plan-gated by `analyst` — Pass 1 needs changes (1 blocker, 4 major, 4 minor), all routed to owners
and fixed, Pass 2 **approve**. Committed as one design-phase change: `30366f4` (plan, graph note,
ML note, review, `BACKLOG.md` K-050/M5 entries, `claude/graph-dba/falkordb-quirks.md` additions,
this coordination doc).

## Plan — implementation phase (not yet dispatched)

1. ~~**U1-U8 — design phase**~~ ✅ complete, see above.
2. **Implementation** — sized to the plan's own 6-stage step table (`document-ingestion.md` §4):
   Stage 1 (chunking + Document/Chunk write path), Stage 2 (chunk embeddings + standalone search),
   Stage 3 (extraction), **checkpoint** (advisory `data-scientist` qualitative review of real
   extraction output, non-blocking), Stage 4 (fusion — the atomic `create_entity_with_auto_match` +
   fuzzy/suggested tier + audit surface), Stage 5 (chat-grounding integration, touches
   `test_provenance.py`), Stage 6 (batch hardening + QA acceptance). Likely one unit per stage or a
   small adjacent-stage cluster, per the step-table sizing rule — not yet dispatched, pending a
   stakeholder decision on how much to build now (see report to user).
3. **Diff-scoped re-gate** — `analyst`, after implementation (a second, code-level gate distinct
   from the design-phase plan gate above).
4. **QA acceptance pass** — `qa-engineer`, against AC-1..AC-10 (`docs/test-plans/document-ingestion.md`
   + `-report.md`).
