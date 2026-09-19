# K-030 Track 2 — AC-2 golden-set regression gate, Stage 8 Phase 2

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-030 (Stage 8 Phase 2)

## Scope & objective

Execute the 29 design-only rows of the 45-pair AC-2 golden set that `data-scientist` designed and
partially piloted in Stage 8 Phase 1 (`claude/docs/plans/agent-knowledge-base-strategy-ml.md`,
"Stage 8 Phase 1" section), then pool the result with Phase 1's 16 executed rows to (a) report the
full-set recall@5/MRR with a Wilson-interval CI, (b) confirm or re-derive the 0.42 cosine-distance
score floor `skills/agent-kb-retrieval/SKILL.md` currently ships, and (c) assess whether the one
duplicate-chunk crowding-out defect Phase 1 found (family h39/R3) is systematic or isolated, using
the 3 stratum-(e) families Phase 1 never ran (R4/h14, R5/h22, R6/h40).

This is **not** a from-scratch test design: the golden set itself (45 rows, every query text and
expected `documentId`, five stratification axes) is `data-scientist`'s already-reviewed
deliverable (`analyst` review: `claude/docs/reviews/agent-knowledge-base-strategy4-stage8-phase1.md`,
approve with suggestions, no blockers). This plan covers only the execution method and judgment
criteria for the remaining 29 rows — reproducing the design table here would duplicate, not extend,
an artifact already gated.

## References

- `claude/docs/plans/agent-knowledge-base-strategy-ml.md` — Recommendation 4 (design) and "Stage 8
  Phase 1" (the full 45-row table, pilot execution, floor derivation).
- `skills/agent-kb-retrieval/SKILL.md` — the exact query-prefix template (verbatim), `limit=5`, and
  the current 0.42 floor with its own explicit "Phase 2 should confirm or re-derive it" instruction.
- `claude/docs/reviews/agent-knowledge-base-strategy4-stage8-phase1.md` — `analyst`'s Phase 1
  review, incl. its independent live reproduction of the R3/h39 crowding-out finding.
- `claude/docs/plans/agent-knowledge-base-strategy4-coordination.md` — coordination ledger (U5/U5a
  = Phase 1 design+pilot+review; this unit is U6).

## Risk assessment (execution-specific; design-level risks already carried in the ml.md doc)

- **Floor-margin risk, carried forward from Phase 1**: the 0.42 floor was calibrated on only 4
  negative queries and 14 found positives, margin 0.037 — thin enough that a single Phase 2
  data point could flip the confirm/re-derive call. Treated as the primary open question this
  gate exists to answer, not a nice-to-have.
- **Prose/narrative retrieval risk**: Phase 1's one clean miss (C1) was prose-tagged; most of the
  29 remaining rows are `b-prose`-tagged (per the design's own stratification), making this run the
  first real test of whether that was a one-off or a pattern.
- **Crowding-out risk**: only 4 of 7 stratum-(e) families are pilot-tested; the 3 remaining
  (R4/h14, R5/h22, R6/h40) are the whole evidence base for judging whether h39's crowding-out is
  systematic.
- **Out of scope for this gate**: re-deriving the query-instruction prefix itself (settled,
  Recommendation 1), re-opening top-K (settled at 5, no evidence against it in Phase 1), the
  `familyId` sibling-pull design question (already answered by the set-recall stratum's existence,
  not reopened here), and any change to `search_documents`'s server-side behavior (out of this
  gate's authority — `qa-engineer` verifies, does not modify the retrieval pipeline).

## Test items

Each item ID is `AC2-<design-table row ID>`, tracing 1:1 to the golden-set design table's own row
IDs in ml.md's "Stage 8 Phase 1" section (e.g. `AC2-R4` = design row R4). All 29 are **functional /
contract** tests of the `search_documents` retrieval contract under the documented calling
convention. Preconditions, steps, and expected result are the same shape for every item, stated
once:

- **Preconditions**: `ws:agent-team` holds the post-Stage-6-migration corpus (327/332 claims
  `ready`, confirmed unchanged since Phase 1 — no migration activity between Phase 1 and this run);
  `skills/agent-kb-retrieval/SKILL.md`'s prefix template and `limit=5` are current.
- **Steps**: (1) build the query string by substituting the row's situation text into the exact
  fenced prefix template; (2) call `search_documents(query=<prefixed>, limit=5)` against
  `ws:agent-team`; (3) record every returned hit's `documentId` and `score` (not just the winning
  one); (4) judge raw hit/miss (expected `documentId` present anywhere in the 5 returned, regardless
  of score) and floor-applied hit/miss (present **and** `score <= 0.42`) separately.
- **Expected result**: the row's fixed expected `documentId`(s) (single-answer rows) or sibling set
  (stratum-(e) rows R4/R5/R6) appear in the top 5, at a score that survives the floor. A negative
  row (N5/N6) expects **no** hit under 0.42 — i.e. the query's best returned score should exceed the
  floor.
- **Priority**: all 29 are P1 (this is the formal AC-2 gate; every row was sized into the design's
  ~40-45-pair budget deliberately, no row is disposable).
- **Type**: contract (single-answer rows, `AC2-F4`…`AC2-L2`), exploratory/completeness (stratum-(e)
  rows `AC2-R4`/`AC2-R5`/`AC2-R6`, scored by set-recall not recall@5), negative/rejection
  (`AC2-N5`/`AC2-N6`).

**Row roster (29 items):** `AC2-R4, AC2-R5, AC2-R6` (review-techniques.md, stratum e) ·
`AC2-F4…AC2-F9` (falkordb-quirks.md) · `AC2-C2…AC2-C5` (coordination-techniques.md) · `AC2-Q2`
(qa-testing-techniques.md) · `AC2-O2` (ops-quirks.md) · `AC2-P1, AC2-P2`
(plan-authoring-techniques.md) · `AC2-S1` (statistical-method-techniques.md) · `AC2-E1`
(frontend-quirks.md) · `AC2-T1` (test-design-techniques.md) · `AC2-X1`
(estimator-test-fixtures.md) · `AC2-G1, AC2-G2` (guard-testing-techniques.md) · `AC2-D1, AC2-D2`
(falkordb-reference.md) · `AC2-L1, AC2-L2` (lm-studio-model-notes.md) · `AC2-N5, AC2-N6` (negative
stratum).

## Environment & data setup

Live `ws:agent-team` workspace via the already-shipped `search_documents`/`get_document` MCP tools
(`mcp__falkor-chat-agent-team__*`). No data setup needed — the corpus is Stage 6's finished
migration, read-only for this gate (no writes, no destructive ops). No mock, no stub: this is a
black-box contract test against the real embedding pipeline and the real stored corpus.

## Entry / exit criteria

**Entry**: Stage 6 migration closed (confirmed, coordination ledger), Stage 8 Phase 1 accepted
(confirmed, U5/U5a), `SKILL.md`'s prefix template unchanged since Phase 1 calibration (confirmed by
reading the file before this run).

**Exit**: all 29 rows executed with recorded score data (not just pass/fail); pooled recall@5/MRR/CI
computed over all 45; floor confirmed or re-derived with documented method and updated in
`SKILL.md` if re-derived; crowding-out systematic-vs-isolated judgment stated with evidence;
prose-vs-code pattern assessed. A kill mid-run still leaves raw score data landed (see report,
"Execution — raw scores").

## Out of scope

Re-running the 16 already-piloted Phase 1 rows (would not be genuine out-of-sample evidence);
authoring new golden-set rows; changing `search_documents`'s server implementation; testing the
write side (`ingest_document`) or attribution (`produced_by`) — unrelated to AC-2's retrieval
contract.
