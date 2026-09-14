# Document ingestion — update & delete — Implementation Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M5 follow-on)

**Note on filename:** dispatched to me as `document-ingestion2.md` (no role suffix). Root
`AGENTS.md`'s doc-family convention reserves the bare slug for the *plan* review and requires an
`-impl` suffix for a review of code claiming to deliver that plan — this is the latter, and the
coordination (`docs/plans/document-ingestion2-coordination.md`) already anticipates one
diff-scoped gate per stage (A/B/C/D), which is exactly what the `-impl` slug's "later `## Pass N`"
mechanism is for. Filed here as `document-ingestion2-impl.md`, Pass 1 = Stage A; please update the
coordination ledger's path accordingly.

## Pass 1 — Stage A (commit `4a6186b`)

**Scope.** Diff-scoped review of commit `4a6186b` ("document-ingestion2 Stage A —
delete/list/deletion-audit") against `falkor-chat/docs/plans/document-ingestion2.md` §4 Stage A,
read alongside §3.5 (delete design + the live-verified Cypher shape), §3.7 (`list_documents`),
§3.8 (surface table), §2.1 (precedents to mirror), and §5's AC-6/AC-8 rows. Baseline: `git diff
cf027c5 4a6186b -- falkor-chat/`. The plan's own Cypher/index shapes are treated as already
live-verified by `graph-dba` (two passes, plan §0) — not re-verified here; this pass checks the
application-layer half (correctness, edge cases, test coverage, transport parity, blast radius).
Per the brief, the concurrent, uncommitted `storefront_api.py`/`test_storefront_api.py`/
`HISTORY.md` state (a different session's in-flight `salesperson-ui` work plus a small
`DocumentNotFoundError`-registration hunk left for this coordination) is explicitly out of scope
and not flagged below.

**Verdict: approve.**

**CPG: not applicable — the plan's own §2 scoped this out as new-code design read directly rather
than an impact-analysis question, and that holds unchanged for this diff-scoped implementation
review; no CPG for `falkor-chat` was consulted or needed.**

### What I verified

- Read the plan in full, plus the coordination doc's Notes section for the storefront_api.py
  context.
- Read every hunk of `git diff cf027c5 4a6186b -- falkor-chat/` (11 files, both source and tests),
  not just the stat summary.
- Diffed `repository.py`'s `delete_document` query, `test_queries.sh`'s new `DELETE_DOCUMENT`
  variable, and the plan's §3.5 verbatim block — all three are byte-identical in Cypher shape (same
  clause order, same variable names).
- Ran the full suite live: `cd falkor-chat/server && .venv/bin/python -m pytest -q` against the
  confirmed-up FalkorDB instance (`redis-cli -p 6379 ping` → `PONG`) — **2680 passed, 14 deselected
  (the `-m live` subset), 0 failed**, in the shared, actively-edited tree (the concurrent
  salesperson-ui session's in-flight `storefront_api.py`/`test_storefront_api.py` changes are
  present and green, consistent with the coordination doc's claim).
- Did not run `./scripts/test_queries.sh` — it `GRAPH.DELETE`s the shared `reference` graph at
  teardown (`falkor-chat/AGENTS.md`), and re-seeding requires touching workspace state a concurrent
  session may depend on; read its new `§14.7` section directly instead (confirmed to exercise
  delete + the audit node + the no-op-on-repeat case).
- Grepped the diff for every later-stage token (`SUPERSEDES`, `update_detection`, `lshBand`,
  `minhash`, `textNormalizedHash`, `jaccard`, `autoSuperseded`,
  `create_document_with_auto_supersede`) — zero hits outside doc comments explaining *why* nothing
  supersedes anything yet. `search_chunks`/`search_documents` are untouched (Stage C is a no-op
  here, correctly).

### Findings

None at blocker or major severity. Two minor/nit observations, neither actionable as a defect:

**Minor — `list_documents`'s two query strings are near-duplicates of each other, differing only
in one `WHERE`/label-pattern clause** (`repository.py`'s `list_documents`, the `current_only`
branch vs. the else branch). This is the plan's own prescribed idiom (§3.7, "two separate query
strings, not a null-guarded `WHERE`" — the live-verified, index-preserving pattern `list_matches`
already established), so it is not a defect; flagging only so a future edit to one branch's
`RETURN` projection doesn't drift from the other unnoticed (both currently list all nine same
columns in the same order).

**Nit — the MCP `test_delete_document_tool_errors_for_unknown_document_id` test asserts only
`pytest.raises(Exception)`**, not the specific `DocumentNotFoundError`/error-code shape. This
matches the existing convention for the analogous `confirm_match`/`reject_match` MCP error-path
tests in this same file (checked: none of them narrow past `Exception` either), so it's consistent
with the codebase, not a regression — noting it only because a tighter assertion would catch a
future accidental swap to a different exception type.

### What's solid

- **Cypher fidelity.** The delete query in `repository.py`, the DDL in `bootstrap_schema.sh`, and
  the new `test_queries.sh` section all reproduce the plan's exact live-verified shapes — no
  rewritten "equivalent" anywhere I could find.
- **AC-6 is proven at the right altitude.** `test_delete_document_removes_document_and_its_chunks`
  and `test_delete_document_route_removes_content`/`_excludes_it_from_list` assert node counts and
  cross-cutting absence from `list_documents`/`GET /documents/{id}`, not just that
  `get_document` returns `None` — this proves "genuinely gone," not "merely filtered," exactly what
  AC-6's wording (plan §5) demands.
- **AC-8 is proven precisely as the plan's own test-strategy row asks**:
  `test_get_document_deletion_returns_audit_record_after_delete` (repository) and
  `test_get_document_deletion_route_after_delete` (API) both assert the audit record survives even
  though the primary `get_document`/`GET /documents/{id}` lookup 404s on the same id.
  `test_delete_document_unknown_document_id_writes_no_audit_node` closes the inverse case (no
  audit node for a no-op delete).
- **Open Question 1 regression test present and correctly scoped**:
  `test_delete_document_leaves_entities_and_relates_to_untouched` is exactly the "regression test
  for the design decision itself" the plan's §5 additional-coverage list calls for, not just
  design-note prose.
- **Transport parity is real, not assumed.** Every new surface
  (`delete_document`/`list_documents`/`get_document_deletion`) has an independent test in both
  `test_api.py` and `test_mcp.py`, and `MAX_ID_LEN` is reused verbatim on both new path params
  (`api.py:260,267`), matching the existing `match_id`/`run_id` precedent exactly (`Path(...,
  min_length=1, max_length=MAX_ID_LEN)`).
- **`app.py`'s 404-mapping edit is minimal and correct** — `DocumentNotFoundError` added to the
  existing tuple with no reordering or removal of `ChannelNotFoundError`/`ThreadNotFoundError`/
  `MatchNotFoundError`; `DocumentNotFoundError` itself mirrors `MatchNotFoundError`'s shape exactly
  (bare `ServiceError` subclass, no custom `__init__`).
- **Index-before-constraint ordering preserved** in `bootstrap_schema.sh`
  (`Document.currentVersion`/`DocumentDeletion.documentId` indexes both appear before their
  corresponding constraint block).
- **Scope discipline.** Nothing from Stage B (`SUPERSEDES` DDL/confirm-reject-recheck/history),
  Stage C (search filtering), or Stage D (update-detection) leaked into this commit — confirmed by
  grep across the full diff, not just by reading the stat summary.

### Open questions

- Confirm the coordination ledger's "Gate → verdict" cell for Stage A should point at
  `document-ingestion2-impl.md` (Pass 1) rather than the bare-slug path it was dispatched against —
  flagged above, not re-litigated here.

## Pass 2 — Stage B (commit `aa1c9be`)

**Scope.** Diff-scoped review of commit `aa1c9be` ("document-ingestion2 Stage B — SUPERSEDES DDL,
update lifecycle, history") against `falkor-chat/docs/plans/document-ingestion2.md` §4 Stage B, read
alongside §2.1 (the `SAME_AS`/`create_or_reopen_match` precedent and the unlabeled-endpoint planner
trap), §3.2 (edge shape/direction/status), §3.4 (confirm/reject/recheck/reopen semantics), §3.7
(`get_document_history`), §3.8 (surface table), and §5 (test strategy). Baseline: `git show aa1c9be`
(the commit's own diff — 11 files, matches the stat in the commit message exactly); cross-checked
against `git diff 4a6186b aa1c9be -- falkor-chat/` and found that range pulls in unrelated intervening
commits (e.g. an `AGENTS.md` hunk from the S11 demo-bring-up work) not part of `aa1c9be` itself, so
`git show aa1c9be` was used as the sole basis for what is and isn't in scope here. Also read
`falkor-chat/docs/plans/document-ingestion2-coordination.md` in full (Sequencing, Ledger, Notes) for
the storefront_api.py/S10 exclusion context and the Stage-B-fix history.

No overlap with Pass 1's two minor/nit findings — this is a disjoint set of files/behaviors (the
`list_documents` two-query-string pattern isn't touched here; the MCP `pytest.raises(Exception)`
convention recurs at line-item level below but is the same already-accepted convention, not
re-litigated).

**Verdict: approve.**

**CPG: not applicable — the plan's own §2 scoped this out as new-code design read directly rather
than an impact-analysis question, and that holds unchanged for this diff-scoped implementation
review; no CPG for `falkor-chat` was consulted or needed.**

### What I verified

- Read every hunk of `git show aa1c9be` across all 11 files (`repository.py`, `services.py`,
  `api.py`, `mcp.py`, `app.py`, `bootstrap_schema.sh`, `test_queries.sh`, and the four test files),
  not just the stat summary.
- Traced every new `SUPERSEDES`-anchored Cypher query in `repository.py`
  (`create_or_reopen_supersede_suggestion`, `confirm_document_update`, `reject_document_update`,
  `recheck_document_update`, `list_pending_document_updates`, `list_document_updates`,
  `get_document_history`'s two per-hop traversal queries) — every one matches its endpoints
  unlabeled where the query is anchored on the relationship's own indexed properties
  (`matchId`/`status`); the two places a `Document` label *does* appear
  (`create_or_reopen_supersede_suggestion`'s two initial `MATCH`es, `get_document_history`'s
  per-hop anchor `MATCH`) are anchored on `Document.documentId`'s own unique index, not on
  `SUPERSEDES`, which is the same pattern `create_or_reopen_match` itself uses for its `Entity`
  endpoints (verified by direct comparison, below).
- Diffed `create_or_reopen_supersede_suggestion` against `create_or_reopen_match`
  (`repository.py:1885-1937`) line by line: structurally byte-identical (`Document`/`SUPERSEDES` in
  place of `Entity`/`SAME_AS`, `new_document_id`/`candidate_document_id` in place of
  `new_entity_id`/`candidate_entity_id`) — the "verbatim mirror" claim in the commit message and
  docstring holds exactly, not just in spirit.
- Ran the scoped suite live: `cd falkor-chat/server && .venv/bin/python -m pytest -q
  tests/test_repository.py tests/test_services.py tests/test_api.py tests/test_mcp.py` against the
  confirmed-up FalkorDB instance (`redis-cli -p 6379 ping` → `PONG`) — **721 passed, 0 failed**.
- Counted added test functions per file against the commit message's "52 new tests" claim:
  `test_repository.py` +25, `test_services.py` +11, `test_api.py` +10, `test_mcp.py` +6 = **52** —
  exact match, not an approximation.
- Grepped the full commit diff for every Stage C/D token
  (`documentCurrent`, `search_chunks`, `search_documents`, `update_detection`, `lshBand`, `minhash`,
  `textNormalizedHash`, `jaccard`, `autoSuperseded`, `create_document_with_auto_supersede`) — the
  only hits are: pre-existing, unmodified `search_documents` inside the MCP tool-discovery list
  (Stage C didn't touch it), `documentCurrent` only inside `confirm_document_update`'s bulk-flip
  (in-scope for Stage B per plan §3.4's confirm-supersedes-the-same-way-as-auto-supersede design),
  and `create_document_with_auto_supersede`/`autoSuperseded` appearing only in prose (docstrings/
  comments explaining *why* this primitive is being built now for Stage D's future benefit) — no
  Stage D code exists in this diff.
- Verified the deliberate `storefront_api.py`/`config.py`/`storefront.py` exclusion against the
  live working tree: `git diff` on those files shows real, substantial uncommitted S10
  (presenter-login/reset-all) work, and a small uncommitted `storefront_api.py` hunk that does add
  `DocumentNotFoundError`/`DocumentUpdateNotFoundError` to a classification map — consistent with
  the commit message's claim, not fictional.
- Confirmed index-before-constraint ordering in `bootstrap_schema.sh` holds file-wide, not just
  in the local diff hunk (`grep`'d every `[index]`/`[constraint]` echo line in `bootstrap_workspace`
  — all indexes at lines 116-259 precede all constraints at 263-326; `SUPERSEDES`'s two indexes and
  one constraint fall correctly on each side).

### Findings

**Minor — `test_queries.sh`'s "pending suggestion excluded from history" assertion doesn't
exercise what it claims** (`falkor-chat/scripts/test_queries.sh:150-151`). The fixture has `sd3`
supersede `sd4` (edge direction `sd3 -> sd4`, reopened to `status='pending'` by the preceding
reopen-on-corroboration test), but the assertion runs `OLDER_STEP` — which looks for an *outgoing*
`SUPERSEDES` edge from the given id — anchored on `documentId='sd4'`. `sd4` has no outgoing edge at
all (it's only ever a supersede *target* in this fixture), so the query returns zero rows
regardless of whether the real `sd3->sd4` edge is `pending`, `confirmed`, or doesn't exist — the
assertion is a tautology, not a check of the status filter. The equivalent check anchored on
`documentId='sd3'` (which *does* have an outgoing edge, with the wrong status) would actually
exercise the exclusion. Low stakes: the real regression coverage for this exact behavior exists and
is correct at the pytest level
(`test_get_document_history_excludes_pending_and_rejected_suggestions`, `test_repository.py`,
verified by reading it — it seeds a pending and a rejected edge both targeting `d1` and asserts
`get_document_history("d1") == ["d1"]`, which does exercise the traversal's status filter
correctly). Suggested fix: change the `CYPHER documentId='sd4'` call at
`test_queries.sh:150` to `documentId='sd3'` so the assertion tests the direction that actually has
a non-confirmed edge to exclude.

**Nit — the MCP `test_confirm_document_update_tool_errors_for_unknown_match_id` test asserts only
`pytest.raises(Exception)`**, not the specific `DocumentUpdateNotFoundError` (`test_mcp.py`). Same
disposition as Pass 1's identical nit against the pre-existing `confirm_match`/`reject_match` MCP
tests: consistent with the established convention in this file (checked: none of the analogous
`SAME_AS` MCP error-path tests narrow past `Exception` either), not a regression.

### Plan-sequencing judgment call — verified sound

The diff implements `create_or_reopen_supersede_suggestion` in Stage B even though the plan's own
§4 file list nominally places it under Stage D (because that's where the detection pipeline first
*calls* it). I checked this against three things and it holds:

1. **The plan's own signature match.** Stage D's file inventory (plan §4) specifies
   `create_or_reopen_supersede_suggestion(ws, *, new_document_id, candidate_document_id, match_id,
   status, confidence, technique, created_at) -> dict` — and that is exactly the signature
   implemented in this diff (`repository.py`), verified by direct comparison. Stage D will consume
   it unchanged; there is no parameter Stage D's background job would need that this signature
   lacks (candidate-generation/shortlisting is entirely Stage D's own concern, upstream of this
   call).
2. **The plan's own verification precedent.** Plan §0's second `graph-dba` live-verification pass
   already exercised `create_or_reopen_supersede_suggestion` at `Document` granularity end-to-end
   (create → no-op-on-re-derive → reject → reopen) as part of validating the *plan's design*, before
   any implementation stage ran — meaning the plan itself already treated this primitive as
   detection-independent, which is the same reasoning this diff's docstring gives for building it in
   Stage B.
3. **Stage B's stated "Done" condition is genuinely unsatisfiable without it.** Plan §4 Stage B says
   the confirm/reject/recheck/list plumbing must be "buildable and testable with directly-injected
   `matchId`s... before any detection code exists" — and the reopen-on-corroboration behavior (an
   explicit plan §5 test-strategy row) requires *re-deriving* a suggestion for the same pair after a
   reject, which needs the same find-or-reopen write path Stage D would eventually use. Building a
   separate, Stage-B-only, throwaway "raw fixture edge" writer instead would have meant re-writing
   (and independently re-verifying) the same guarded find-or-create-or-reopen logic twice.

No parameter or behavior of `create_or_reopen_supersede_suggestion` would need to change once Stage
D's detection pipeline actually calls it — this is a legitimate, low-risk pull-forward, not scope
creep.

### What's solid

- **`SUPERSEDES` planner-trap discipline holds everywhere it needs to.** Every query anchored on
  the relationship's own indexed properties (`matchId`, `status`) matches both endpoints unlabeled;
  the two places a `Document` label appears are anchored on `Document.documentId`'s own index, the
  identical pattern `create_or_reopen_match` already uses — confirmed by direct line-by-line
  comparison, not just by re-reading the plan's prose.
- **`get_document_history` is correctly confirmed-edges-only, direction-correct, and works from any
  version in the chain.** `test_get_document_history_works_from_a_middle_version_not_just_the_tip`
  and `..._from_the_oldest_version` both assert the identical oldest→newest ordering regardless of
  which version's id is queried; `..._excludes_pending_and_rejected_suggestions` is a real,
  correctly-targeted regression test (see Findings above for the one place the *shell-script* analog
  of this check is weaker).
- **Confirm/reject/recheck/reopen full lifecycle parity with `SAME_AS` is real, not asserted.**
  `confirm_document_update` flips `currentVersion`/`supersededAt`/`supersededBy` and bulk-flips
  chunks in one `GRAPH.QUERY`; `reject_document_update` changes nothing on either `Document`/`Chunk`
  (asserted directly, not just inferred from "no code touches it"); the reopen-on-corroboration
  sequence (reject → re-derive) reopens to `pending` (never `confirmed`), bumps `resuggestCount` on
  the **original** `matchId`, and creates no duplicate edge — asserted at both the repository level
  (`test_create_or_reopen_supersede_suggestion_reopens_a_rejected_edge`) and the `test_queries.sh`
  integration level.
- **`decided_by=ctx.actor`, never `'system'`, verified at the service layer**
  (`test_confirm_document_update_stamps_the_calling_actor_never_system` asserts the exact call
  tuple reaching the fake repo), correctly contrasted in both the code comments and this review
  against Stage D's future `'system'`-stamped auto tier.
- **`DocumentUpdateNotFoundError` is a genuinely distinct class**, correctly added to `app.py`'s
  404-mapped tuple alongside (not replacing) `DocumentNotFoundError`/`MatchNotFoundError`/
  `ChannelNotFoundError`/`ThreadNotFoundError`.
- **REST/MCP transport parity is complete and independently tested** for all six new surfaces plus
  `get_document_history` — every one has its own test in both `test_api.py` and `test_mcp.py`,
  matching the existing `confirm_match`/`reject_match`-style pattern exactly, including the 404/
  no-op edge cases.
- **Route ordering introduces no new ambiguity.** `/documents/{document_id}/history` is two
  segments past the existing one-segment `/documents/{document_id}`; `/document-updates/pending`
  (literal, one segment) and `/document-updates/{match_id}/confirm|reject|recheck` (two segments)
  never collide regardless of registration order, since no `GET /document-updates/{match_id}` route
  exists at all.
- **Scope discipline is real, confirmed by grep across the whole commit diff**, not just by reading
  the stat summary — zero Stage C (`search_chunks`/`search_documents` filtering) or Stage D
  (`update_detection`, `lshBand`, `minhash`, `textNormalizedHash`, `jaccard`, `autoSuperseded`,
  `create_document_with_auto_supersede` as actual code) leakage, beyond the one deliberate,
  well-reasoned pull-forward addressed above.
- **The deliberate `storefront_api.py`/`config.py`/`storefront.py`/`HISTORY.md` exclusion is
  accurately described** — checked against the live uncommitted working-tree diff, which does show
  the described concurrent S10 work and the small, verified-correct, deliberately-withheld
  classification-map hunk.
- **52 new tests, exact count verified**, all green in a scoped live run (721 passed, 0 failed).

### Open questions

None beyond Pass 1's still-open ledger-path item (unrelated to this pass).

## Pass 3 — Stage C (commit `6443365`)

**Scope.** Diff-scoped review of commit `6443365` ("document-ingestion2 Stage C — default-search
current-chunk filtering") against `falkor-chat/docs/plans/document-ingestion2.md` §3.3/§4 Stage C.
Baseline: `git show 6443365` (repository.py's `search_chunks` gains `WHERE seed.documentCurrent =
true` post-`YIELD`; services.py's `search_documents` gains `SEARCH_DOCUMENTS_OVERFETCH = 3`;
`test_queries.sh` §14.9 plus a real fix to a Pass 2-flagged tautology; new tests in
`test_graphrag.py`/`test_api.py`/`test_services.py`). Per the brief, `docs/reviews/document-
ingestion2-rca.md` (the ANN-recall-flakiness RCA, already independently spot-checked by the
dispatching coordinator) is accepted as settled and **not re-litigated** — this pass judges the
Stage C mechanism itself (filter placement, over-fetch math, test coverage, blast radius), which
the RCA did not examine.

**Verdict: needs changes.** Two blockers below are both live/verified, not hypothetical.

**CPG: used `cpg_falkorchat`** (`cpg/.cpg-artifacts/MANIFEST.txt`, built at `b795f4c`, predates this
commit but not `search_chunks` itself, added 2026-08-24) — ran a call-graph query for every `CALL`
node named `search_chunks` to independently corroborate Blocker 2's "exactly two production
callers, only one over-fetches" claim beyond grep; it agreed exactly (see Finding 2).

### What I verified

- Read `git show 6443365` in full (all 7 files, not just the stat), plus the plan's §3.3 and §4
  Stage C sections directly (not the brief's paraphrase).
- Read the accepted RCA (`document-ingestion2-rca.md`) in full and the coordination ledger's Stage
  C notes — confirmed the RCA's independent spot-check happened (coordinator re-ran the exact repro
  and got the identical `4 failed, 739 passed`) before treating it as settled.
- Traced `Repository.search_chunks`'s current source (`repository.py:1209-1261`) — the `WHERE`
  clause is correctly post-`YIELD`, before `ORDER BY`/`LIMIT`, matching the plan's §3.3 shape
  verbatim; compared against `hybrid_search` (`:826-866`) for idiom consistency.
- Grepped every call site of `repository.search_chunks` in the whole codebase (`grep -n
  "\.search_chunks(" server/falkorchat/*.py`) — exactly two, `services.py:1096`
  (`Services.hybrid_search`) and `services.py:1300` (`Services.search_documents`) — then
  cross-checked via `cpg_falkorchat`'s `CALL` nodes named `search_chunks`, which independently
  confirmed the same two production call sites (plus 4 test-file calls), agreeing on file/shape.
- Ran `bash -n falkor-chat/scripts/test_queries.sh` (syntax-only, per the brief — not executed live
  against the shared `reference` graph) — clean. Read the new §14.9 section and the Pass 2
  tautology fix line-by-line: `sd3` (the fixed anchor) does have an outgoing `pending`-status
  `SUPERSEDES` edge in the fixture at that point (verified against the surrounding `sd3`/`sd4` setup
  and the reopen-on-corroboration step above it) — the fix is genuine, not a second tautology.
- Ran read-only (`GRAPH.RO_QUERY`) probes against the live, shared FalkorDB instance to check
  Finding 1's "pre-existing chunk with `documentCurrent` unset" scenario against real data, not a
  synthetic one — see Finding 1 for the exact queries/output. No writes issued.
- Compared `repository.py:1013` (pre-plan `create_document`, per the plan's own §2.1 inventory:
  `Document{documentId,title,text,sourceFormat,sourceKind,status,pendingJobs,createdAt}` +
  `HAS_CHUNK → Chunk{chunkId,text,seq,documentId}` — no version fields) against the current
  `create_document`, which only gained `currentVersion`/`documentCurrent` in Stage A (`4a6186b`,
  2026-09-11) — established the exact commit boundary before which any ingested `Document`/`Chunk`
  has neither property at all.
- Checked `services.search_documents`'s over-fetch math against its API-layer `limit` bounds
  (`api.py:229`, `Query(20, ge=1, le=200)`) — `k` ranges 3..600, no known `db.idx.vector.queryNodes`
  max-`k` ceiling in `claude/graph-dba/falkordb-quirks.md`; not a concern.

### Findings

**Blocker — every `Document`/`Chunk` ingested before Stage A landed has no `currentVersion`/
`documentCurrent` property at all, and the new `WHERE seed.documentCurrent = true` filter silently
excludes all of them, forever, from `search_chunks`/`search_documents` — already live, not
hypothetical.** `create_document` only started setting these two properties in Stage A (`4a6186b`,
2026‑09‑11); the original K‑050 M5 feature (landed 2026‑08‑23/24) and at least one later seed script
(`scripts/seed_nlq_eval_corpus.py`, commit `c88688b`, 2026‑08‑29) ingested real documents through the
old `create_document` shape 13+ days before Stage A. Verified live, read-only, against the shared
instance:
```
GRAPH.RO_QUERY ws:acme     "MATCH (d:Document) RETURN count(d), count(d.currentVersion)"   → 29, 0
GRAPH.RO_QUERY ws:nlq-eval "MATCH (d:Document) RETURN count(d), count(d.currentVersion)"   → 12, 0
# unfiltered ANN on ws:acme's real Chunk data: 20 rows. With the new WHERE added: 0 rows.
```
A missing property makes `seed.documentCurrent = true` evaluate to `NULL` (falsy), so `GET
/documents/search`, the `search_documents` MCP tool, and `Services.search_documents` now return
**zero results for every query** against `ws:acme` and `ws:nlq-eval` — a full, silent regression of
FR-3 in exactly the two workspaces that had real content before this commit. No test in the diff
exercises this axis: every fixture (`_seed_document`, `client.post("/documents", ...)`, the raw
`_mark_document_superseded[_raw]` helpers) goes through the *current* `create_document`, which
always sets the property, so a chunk with the property **absent** (not `false`) is structurally
untested. Suggested fix: a backfill migration for every already-populated workspace (mirrors this
codebase's own precedent for exactly this shape, `scripts/backfill_thread_ids.sh` — see
`falkor-chat/AGENTS.md`'s table entry) — `MATCH (d:Document) WHERE d.currentVersion IS NULL SET
d.currentVersion = true` / `MATCH (c:Chunk) WHERE c.documentCurrent IS NULL SET c.documentCurrent =
true` — run against `ws:acme` and `ws:nlq-eval` at minimum, plus a regression test that creates a
`Chunk` via a raw fixture write that never sets `documentCurrent` and asserts it is still found
(covering the "pre-migration data" axis directly, not a list of already-current/already-superseded
shapes). Alternative: make the filter itself null-tolerant (`WHERE seed.documentCurrent <> false`)
as a no-migration fix, but that is a deliberate semantic choice (treat "never touched by this
feature" as "current") that should be stated, not silently substituted for the plan's own "plain
equality" wording.

**Blocker — the over-fetch fix was applied to only one of `repository.search_chunks`'s two
production callers, leaving the agent chat-grounding retrieval path exposed to the exact under-fill
defect this diff exists to prevent.** `services.py` has exactly two callers of
`self._repo.search_chunks(...)`: `Services.search_documents` (`:1300`, fixed — `k = limit *
SEARCH_DOCUMENTS_OVERFETCH`) and `Services.hybrid_search` (`:1096`, **unfixed** — still `k=k,
limit=limit`, unchanged since before this commit). `Services.hybrid_search` is the merge behind
`GraphragRetrieveTool`/`AgentResponder`'s chat-grounding retrieval (`tools.py:358`,
`responder.py:104`) — its `chunk_hits` half now runs through the same filtered `search_chunks`, so
whenever superseded chunks rank ahead of current ones in the ANN pool, the merged grounding result
can under-fill on `Chunk`-sourced hits exactly as `search_documents` could before this diff's own
fix (plan §3.3's stated rationale — "once a post-filter can discard rows... must over-fetch" —
applies verbatim here too). Verified via two independent methods: `grep -n "\.search_chunks("
server/falkorchat/*.py` (2 hits) and a `cpg_falkorchat` `CALL`-node query for `search_chunks`,
which independently confirmed the same two production sites (plus test-only calls) and showed both
used `k=limit`/`k=k` in the pre-Stage-C snapshot — consistent with the diff touching only one.
Suggested fix: apply the same `SEARCH_DOCUMENTS_OVERFETCH`-style multiplier (or a shared
constant/helper, since the ratio is now duplicated reasoning) to `Services.hybrid_search`'s
`search_chunks` call, and add a hard-negative test mirroring
`test_search_documents_overfetch_prevents_under_fill_when_superseded_chunks_rank_first` but through
`hybrid_search`'s merge path.

**Minor — no evidence of mutation testing on the new mechanism.** The coordination ledger names
mutation testing for the concurrent Stage B-fix unit ("confirmed via mutation test") but the Stage
C row and commit message carry no such note. Note for calibration: a mutation pass on the touched
lines (the `WHERE` clause, the `SEARCH_DOCUMENTS_OVERFETCH` constant/multiplication) would **not**
have caught either blocker above — Blocker 1 is a live-data-state issue, and Blocker 2 is an
omitted call site, both invisible to mutating code that exists. Flagging only because the brief
asked, not because it would have prevented what's actually wrong here.

**Nit — `services.search_documents`'s docstring analogy is slightly overstated.** It says over-
fetching is "the same idiom `hybrid_search` already uses for its own scope filtering," but
`repository.hybrid_search`'s `OPTIONAL MATCH`-based scope join never discards a seed row (it's a
left join, not an exclusion filter), so it never had an under-fill risk to over-fetch against in
the first place — unlike `search_chunks`'s new `WHERE`. Not consequential on its own, but worth
correcting alongside Blocker 2's fix so the next reader doesn't infer `hybrid_search` was already
handling this correctly.

### What's solid

- **Filter placement and semantics are exactly per plan where data has the property.**
  `repository.py`'s `WHERE seed.documentCurrent = true` sits post-`YIELD`, pre-`LIMIT`, a plain
  equality (not `exists()`) — `GRAPH.PROFILE` in the accepted RCA already confirmed `Filter` runs
  strictly after `ProcedureCall` with no label scan, and this pass's own live probe against
  `ws:acme` confirms the same shape end-to-end on real data (20 unfiltered vs. 0 filtered, for the
  reason in Blocker 1 — not a planner defect).
- **The over-fetch idea itself is sound and well-tested where it was applied.**
  `SEARCH_DOCUMENTS_OVERFETCH = 3`'s math is verified correct in `test_services.py`'s call-tuple
  assertions and in the genuinely deterministic hard-negative test
  (`test_search_documents_overfetch_prevents_under_fill_when_superseded_chunks_rank_first`, via
  `_RankedChunkRepo`) — a real unit test of the `k`-vs-`limit` interaction, not just a smoke test.
- **The dropped live-E2E over-fetch attempt is honestly documented**, with a specific, checkable
  reason (`test_api.py`'s comment: small-corpus ANN recall isn't a simple function of `k` on this
  build) rather than silently skipped.
- **The Pass 2 tautology fix is a real fix, verified by tracing the fixture.** `sd3`'s outgoing
  `SUPERSEDES` edge is genuinely `pending` at the point `OLDER_STEP` now anchors on it, so the
  assertion exercises the status filter it claims to.
- **AC-5 parity (direct lookup unaffected by search filtering) is asserted at all three altitudes**
  (`test_queries.sh` §14.9, `test_graphrag.py`, `test_api.py`) — consistent, not just repeated.
- **Scope/RCA discipline.** The commit correctly carries no code change in response to the accepted
  RCA, and this pass independently re-confirms that conclusion still holds for the flakiness
  question — nothing here reopens it.

### Open questions

- Whether the Blocker 1 backfill should block Stage D's dispatch (Stage D adds more document
  ingestion, compounding the affected surface if left open) or land as a fast, narrow follow-up unit
  before Stage D — the caller's/`teco`'s call on sequencing, not diagnosed here.
- Whether any other already-populated workspace beyond `ws:acme`/`ws:nlq-eval` exists outside this
  FalkorDB instance (e.g., a different environment) and needs the same backfill — out of this pass's
  reach (checked only the one live instance available here).

## Pass 4 — Stage C-testinfra (uncommitted, "vector-index churn" fix)

**Scope.** Diff-scoped review of the currently uncommitted working-tree diff fixing the
test-infrastructure defect root-caused in `document-ingestion2-rca.md` (already independently
spot-checked by the coordinator per the coordination ledger) — **not** a re-review of the RCA
itself, and not Stage C's application code (Pass 3, above, `needs changes` for unrelated reasons,
out of scope here). Touched: `falkor-chat/server/tests/conftest.py` (new
`rebuild_vector_indexes`/`fresh_vector_index`), `test_api.py`, `test_graphrag.py`, `test_mcp.py`,
`test_responder.py`, `test_tools.py` (fixture applied to every real-ANN test), and the new
`test_vector_index_churn_guard.py`. Baseline: `git diff -- <those six files>` plus the new file's
full contents read directly, not any prior summary.

**Verdict: approve.**

**CPG: used `cpg_falkorchat`** (`cpg/.cpg-artifacts/MANIFEST.txt`, includes `tests/` per its
`--verify-prefix`) — ran a call-graph query for every `CALL` node named `search_chunks`/
`hybrid_search`/`set_chunk_embedding`/`set_embedding`, joined to its enclosing `METHOD` for
`FILENAME` (`CALL.FILENAME` is empty — a documented trap, `skills/joern-cpg/references/
cpg-model.md:138`) to independently corroborate the audit's completeness beyond grep — it surfaced
two production callers in `tests/eval/` that a `tests/*.py`-scoped grep alone would have missed
(see Finding 1/What I verified).

### What I verified

- Read every hunk of `git diff -- conftest.py test_api.py test_graphrag.py test_mcp.py
  test_responder.py test_tools.py` and the full new `test_vector_index_churn_guard.py`, not a
  summary of them.
- **Audit completeness (brief item 1).** Grepped `tests/*.py` for
  `queryNodes|search_chunks|search_documents|hybrid_search|vecf32` — confirms `test_repository.py`
  has zero embedding-related tokens at all, and `test_services.py`'s only hits are through its
  module-local `FakeRepo`/`SpyRepo`-style doubles (`class FakeRepo`, `def search_chunks`/`def
  hybrid_search` on the fake itself), never a real `conn`/`repo` fixture — confirmed by reading the
  file's fixtures directly, not inferring from the grep. Then cross-checked with `cpg_falkorchat`
  (query above): the only test-file callers of `search_chunks`/`hybrid_search` are
  `test_graphrag.py` (11 hits, all covered by the module's new `pytestmark`),
  `test_services.py` (7 hits, all inside `FakeRepo`-backed unit tests, corroborating the grep), and
  two hits in `tests/eval/` my plain `tests/*.py` grep glob had missed entirely:
  `test_judge_live.py` (`pytest.mark.live`, deselected by default — confirmed via its own
  `pytestmark = pytest.mark.live` line) and `test_retrieval_eval.py::_aggregate_metrics` (runs
  unconditionally, but against `ws:eval`, whose own `tests/eval/conftest.py` docstring states the
  fixture is deliberately "probe-only... never writes" — read directly, confirms this module
  contributes zero embedding-write churn to any index, `ws:test`'s or its own). Also grepped
  `test_background.py` (the other file the tool-independent grep flagged): its one `hybrid_search`
  reference is a raising stub asserting the method is *never* reached, not a live call. No gap.
- **Rebuild correctness (item 2).** `rebuild_vector_indexes`'s `CREATE VECTOR INDEX` statement is
  character-for-character the same shape as `scripts/bootstrap_schema.sh:358,361`
  (`dimension:${dim}, similarityFunction:'cosine'`), parameterized by `TEST_EMBEDDING_DIM`
  (matches `conftest.py`'s existing constant, used by `_schema` too) — same index, same params, not
  a look-alike. Order relative to `conn`'s node wipe is safe regardless of fixture-resolution order
  between `repo`/`fresh_vector_index` in a test signature: `conn`'s body already runs `MATCH (n)
  DETACH DELETE n` before returning, and drop+recreate of an index touches no node data, so there
  is never a real ordering hazard here.
- **Determinism (item 3).** Ran `tests/test_vector_index_churn_guard.py` alone four times
  (`-v` once, plain three more) — `2 passed` every time, ~0.35s each. Read both tests: test 1 pins
  the phenomenon with an assertion that k=4 recall is *reliable* at low churn (10/200 cumulative)
  and *broken* at 300, plus k=50 staying reliable throughout the same churn — not tautological,
  since a broken/no-op churn simulation would make the low-churn assertions fail too, not just the
  high-churn one. Test 2 is the actual fix-guard: asserts a churn state that just broke k=4, then
  calls `rebuild_vector_indexes` and asserts recall recovers — this is exactly the "monotonic-in-
  churn regression test" the RCA's §5 asked for, not a fixed-k bump.
- **RCA repro + full suite (item 4).** Ran `pytest -q tests/test_repository.py tests/test_services.py
  tests/test_api.py tests/test_mcp.py tests/test_graphrag.py` twice — **743 passed, 0 failed**,
  both times, against the live, shared FalkorDB instance (`redis-cli -p 6379 ping` → `PONG`). Ran
  the full suite (`pytest -q`) — **2744 passed, 14 deselected, 0 failed**, 30.93s wall time. To
  sanity-check the runtime-delta claim, stashed the six touched test files (moving the new guard
  file out of `tests/` first so it doesn't fail to import `conftest.rebuild_vector_indexes`) and
  re-ran the full suite on the unmodified baseline: **6 failed** (the churn phenomenon reproducing
  live, as expected — 3 in `test_graphrag.py`, 1 in `test_responder.py`, 2 in `test_tools.py`, more
  than the RCA's original 4 because the suite has grown since), **30.88s**. Popped the stash to
  restore the working tree exactly. Delta is ~0.05s over the whole suite, not the ~1-1.5s the
  delegate's report apparently claimed (Finding 1, minor) — but the direction is right (negligible,
  not concerning) and no module shows a runtime outlier in `--durations=15` on the ANN-heavy
  modules (`test_graphrag.py`'s worst case: 0.29s one-time fixture setup, not per-test).
- **Untouched-file guarantee (item 5).** `git diff --stat -- falkorchat/repository.py
  falkorchat/services.py falkorchat/storefront_api.py tests/test_storefront_api.py
  falkorchat/storefront.py falkorchat/config.py` — empty output, confirmed clean against the
  concurrent salesperson-ui coordination's files.
- **Fixture granularity (item 6).** Function-scoped-per-ANN-test (module-wide `pytestmark` in
  `test_graphrag.py`, since every test there is a live-ANN integration test — confirmed by reading
  all 19 test signatures, all take `repo`/`conn`) is more conservative than the RCA's own suggested
  module-or-function boundary, at negligible measured cost (above). No test class or indirect-
  fixture-import path found in any of the six touched files or their siblings that bypasses it —
  the CPG cross-check (item 1) was the check most likely to surface a missed indirect caller, and
  it didn't.
- **Mutation-test claim (item 7).** Independently reproduced the underlying claim myself rather
  than merely judging its plausibility: stashing the fix (above) and re-running the RCA's exact
  5-file repro command reproduced **the identical `4 failed, 739 passed`, same 4 test names**, byte
  for byte against the RCA's own reported output — this is direct evidence the fix (not some
  unrelated suite change) is what closes those 4 failures, which is what a "revert the fix, watch
  it fail the same way" mutation test is meant to show. Did not additionally edit
  `rebuild_vector_indexes` itself to a no-op, since this independent stash-based reproduction
  already settles the claim without needing to.

### Findings

**Minor — the coordination ledger's reported runtime-delta figure (~1-1.5s) doesn't match what
this pass measured (~0.05s over the full 2744-test suite).** Not consequential — the actual number
is smaller, not larger, than claimed, so it isn't hiding a regression — but worth a one-line
correction in the ledger so a future reader doesn't budget suite-runtime headroom against a
overstated number. See "What I verified" above for the stash-based before/after measurement.

**Nit — the `("Message", "Chunk")` label tuple is duplicated** between `conftest.rebuild_vector_indexes`
and `test_vector_index_churn_guard.py`'s `probe_conn` fixture (which only creates, never drops,
since its graph starts empty). Two lines, low stakes, but a third label added to one and not the
other would silently under-cover. Suggested: export the tuple as a small constant next to
`TEST_EMBEDDING_DIM` in `conftest.py` and import it in the guard-test file.

### What's solid

- **The audit is complete**, verified two independent ways (grep + CPG call-graph), including a
  gap the grep-only method would have missed (`tests/eval/`) that turned out not to be a real gap
  once read — `test_judge_live.py` is `live`-marked/deselected, `test_retrieval_eval.py` runs
  against `ws:eval`, which the codebase's own `tests/eval/conftest.py` already documents as
  deliberately never written to by tests.
- **The rebuild helper is a genuine fix, not a no-op or partial reset** — same DDL shape as
  `bootstrap_schema.sh`, at the right dimension, and directly proven to restore recall after heavy
  churn by `test_vector_index_churn_guard.py`'s second test (run live, passed deterministically
  across four runs).
- **The regression guard is a real, non-tautological pin of the phenomenon**, not just of one
  fixed-k assertion — it separately asserts low-churn reliability, high-churn failure, and
  large-k reliability throughout, so a broken churn simulation (not just a broken fix) would also
  be caught.
- **All target ANN-touching modules are covered, nothing is over-broad**: `test_services.py`/
  `test_repository.py` are correctly left alone (no live ANN in either), confirmed independently of
  the delegate's own claim.
- **The fix is genuinely isolated** — `git diff --stat` confirms zero overlap with the concurrent
  salesperson-ui coordination's files, and the working tree was left exactly as found after this
  pass's own stash-based before/after measurement (`git stash pop`, verified via `git status`
  immediately after).
- **The mutation-test claim holds up** — independently reproduced (not merely judged plausible)
  via a direct stash-and-rerun that matches the RCA's own byte-for-byte failure signature.

### Open questions

- `ws:eval`'s own vector index (`tests/eval/test_retrieval_eval.py`) is architecturally insulated
  from *this* defect (the suite never writes to it), but nothing found here rules out the same
  HNSW-churn-degradation mechanism affecting it over its own lifetime from periodic corpus reseeds
  (`scripts/seed_eval_corpus.sh`, run outside pytest) — not diagnosed here, and out of this fix's
  scope, but worth a `graph-dba`/`data-scientist` note if `test_retrieval_eval.py`'s baseline ever
  starts drifting unexplainably.

## Pass 5 — Stage C-fix (uncommitted, both Pass 3 blockers)

**Scope.** Diff-scoped review of the currently uncommitted working-tree diff fixing Pass 3's two
blockers — Blocker 1 (pre-Stage-A `Document`/`Chunk` rows have no `currentVersion`/
`documentCurrent` property, silently zeroing search results) and Blocker 2 (`Services.
hybrid_search`'s `search_chunks` call was exposed to the same under-fill defect `search_documents`
was already fixed for). Touched: `server/falkorchat/services.py`, `server/tests/test_graphrag.py`,
`server/tests/test_services.py`, plus new `scripts/backfill_document_current.sh`. Baseline: `git
diff`/`git status --short` in `falkor-chat/` (repository.py confirmed untouched, matching the
coordination ledger's own file-list claim). The coordinator's independent live re-probe of
`ws:acme`/`ws:nlq-eval` backfill counts (29/29, 87/87, 12/12, 12/12) is accepted as already
corroborated and not re-run here, per the brief.

**Verdict: approve with suggestions.** No blockers; one minor finding (a docstring accuracy issue
that traces back to this review's own Pass 3 nit, not an implementer error).

**CPG: considered, not relevant** — the diff modifies arguments to two already-known, already-
enumerated call sites (Pass 3 exhaustively confirmed the caller set for `search_chunks`, corroborated
independently via `cpg_falkorchat`'s `CALL`-node index at the time); it adds no new caller, no new
symbol, and no new edge for a call-graph query to find. Direct reading (repository.py's Cypher,
QUERIES.md §6) was the faster and more precise tool for this diff's one CPG-shaped question
(finding 1, below).

### What I verified

- Read the full diff of all three changed files plus the new script's full contents directly (not
  a summary).
- **Blocker 1 — script vs. precedent.** Diffed `scripts/backfill_document_current.sh` against
  `scripts/backfill_thread_ids.sh` side by side: identical shape (header block, env vars, `PING`
  check, per-workspace loop, `GRAPH.QUERY`/`sed -n '2p'` count extraction, idempotency framing).
  Cypher matches Pass 3's exact suggested fix (`WHERE …IS NULL SET … = true`, two independent
  statements, no shared `MATCH`) and only touches rows where the property is genuinely absent —
  confirmed by reading, not inferring.
  - Ran the shipped script live against the throwaway `ws:test` workspace as part of the new
    `test_graphrag.py` tests (below) — this is the same live-execution path the new tests exercise,
    not a separate manual run.
  - Accepted the coordinator's independent `ws:acme`/`ws:nlq-eval` count re-probe per the brief;
    did not re-run it.
- **Blocker 1 — test coverage of the "property absent" axis.** Read
  `_strip_document_current`/`_run_document_current_backfill`/all four new tests in
  `test_graphrag.py` in full: `_strip_document_current` does a raw `REMOVE` (not `SET … = false`),
  genuinely simulating "never touched by Stage A," distinct from `_mark_document_superseded`'s
  "touched, then explicitly superseded" shape. `_run_document_current_backfill` invokes the real
  shipped script as a subprocess (mirrors `conftest.py`'s own `bootstrap_schema.sh` invocation
  pattern) rather than a duplicated inline Cypher string, so these tests pin the shipped script
  itself. Four tests cover: baseline exclusion (absent ⇒ invisible), post-backfill discoverability,
  non-reopening of a genuinely-superseded document (asserted via direct property reads, with a
  documented and correct reason for not using `search_chunks` for that one — small-corpus ANN
  recall isn't reliable enough to pin an exact-membership claim across two documents sharing one
  tiny index, per the accepted RCA), and idempotency (second run touches nothing, verified via a
  workspace-wide `count()` after `conn`'s per-test `DETACH DELETE n` wipe, not just the one seeded
  row). Ran `tests/test_graphrag.py tests/test_services.py`: **288 passed**, live, against the
  shared FalkorDB instance.
- **Blocker 2 — fix placement and math.** `services.py`'s `hybrid_search` now passes `k = k *
  SEARCH_DOCUMENTS_OVERFETCH` to `chunk_hits`'s `search_chunks` call, `msg_hits`'s
  `repo.hybrid_search` call left unchanged (`k=k`) — the right call site, matching Pass 3's Finding
  2 exactly. Confirmed the base variable is correct for this call site: `hybrid_search` has two
  independently-tunable `k`/`limit` params (unlike `search_documents`, which only has `limit`), so
  multiplying `k` (ANN candidate depth) rather than `limit` (final cap, left untouched on both the
  `search_chunks` call and the outer `merged[:limit]` truncation) preserves the existing
  independent-tunability contract — not a behavior change beyond fixing the under-fill.
- **Same-revision-fixes interaction check.** Read both fixes together for any interaction: they
  touch the same file but different call sites and different downstream consumers (`search_chunks`
  called with `limit=limit` internally in both cases — unchanged by this diff, only `k` changed), so
  there is no double-counting — increasing `k` only deepens the pre-filter ANN candidate pool, it
  does not change how many rows either call site or the outer merge ultimately returns. Checked
  `self._k`'s callers (`tools.py:358` `GraphragRetrieveTool`, `responder.py:104`
  `AgentResponder`) — both pass a small constructor-time constant (`DEFAULT_RETRIEVE_K`, not an
  open-ended user-supplied value like `search_documents`'s REST `limit` (`ge=1, le=200`)), so the
  `k * 3` scaling here has no pathological-size exposure.
- **Test-update genuineness.** Ran the two pre-existing `hybrid_search` call-tuple tests
  (`test_hybrid_search_applies_rag_timeout_constant`,
  `test_hybrid_search_forwards_channel_scope_to_message_pool_only`) and the new hard-negative
  (`test_hybrid_search_overfetch_prevents_under_fill_when_superseded_chunks_rank_first`) — all
  green against the current diff. Checked the updated assertions do the arithmetic themselves
  (`10 * SEARCH_DOCUMENTS_OVERFETCH`, `3 * SEARCH_DOCUMENTS_OVERFETCH`) rather than loosening to a
  range or removing the check — genuine updates, not weakened ones.
- **Mutation-test verification — reproduced independently, not just judged plausible.** Copied
  `services.py` and `scripts/backfill_document_current.sh` to `/tmp` first (md5-recorded), then:
  - Mutated the backfill script's two `IS NULL` → `IS NOT NULL` (sed, both occurrences) and ran
    `pytest tests/test_graphrag.py -k "backfill or pre_migration or idempotent"`: **3 failed, 1
    passed** — `test_search_chunks_finds_pre_migration_chunk_after_documentCurrent_backfill`,
    `test_document_current_backfill_never_reopens_a_genuinely_superseded_document`,
    `test_document_current_backfill_is_idempotent` all failed (the baseline-exclusion test, which
    doesn't call the script, stayed green as expected). Matches the delegate's claimed "3 tests"
    exactly. Restored from the `/tmp` copy; md5 confirmed byte-identical to pre-mutation.
  - Mutated `hybrid_search`'s `chunk_hits` call back to unmultiplied `k=k` and ran `pytest
    tests/test_services.py -k hybrid_search`: **3 failed, 5 passed** —
    `test_hybrid_search_applies_rag_timeout_constant`,
    `test_hybrid_search_forwards_channel_scope_to_message_pool_only`,
    `test_hybrid_search_overfetch_prevents_under_fill_when_superseded_chunks_rank_first` all
    failed. Matches the delegate's claimed "3 tests" exactly. Restored from the `/tmp` copy; md5
    confirmed byte-identical.
  - Re-ran `tests/test_graphrag.py tests/test_services.py` after both restores: **288 passed**,
    confirming the working tree was left exactly as found.
- **Docstring nit (Pass 3) correction — reasoning check against the actual Cypher (brief item).**
  Read `repository.hybrid_search`'s full Cypher (`repository.py:826-874`) and its canonical copy
  (`docs/QUERIES.md` §6, lines 447-463) side by side against the new docstring text in both
  `services.search_documents` and `services.hybrid_search`. Result in Finding 1 below — the
  specific mechanism claimed does not hold as stated, though I did not find evidence the underlying
  conclusion (no over-fetch needed for `msg_hits`) is wrong for the two production-relevant cases.

### Findings

**Minor — the corrected docstring's claim that `repository.hybrid_search`'s "scope join" is an
`OPTIONAL MATCH`, and therefore "never discards a seed row," is not accurate against the actual
Cypher.** `repository.py:850-863` (= `docs/QUERIES.md` §6 verbatim) has exactly one `OPTIONAL
MATCH` — the `(seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Message)` co-occurrence
expansion that feeds `relatedContext`. The actual "scope join" is two plain, required `MATCH`
clauses: `MATCH (t:Thread)-[:HEAD|NEXT*0..]->(seed)` (always) and, when `channel_id` is given,
`MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)` — both of which *can* discard a
seed row, structurally the same "post-YIELD, pre-`LIMIT` exclusion" shape as `search_chunks`'s new
`WHERE`, not a left join. This claim originates verbatim in this review's own Pass 3 nit ("the
`OPTIONAL MATCH`-based scope join never discards a seed row… it's a left join, not an exclusion
filter") — the implementer faithfully transcribed Pass 3's suggested correction rather than
introducing the error independently, so this is on me, not the delegate. That said, I found no
evidence the *conclusion* is currently wrong: the channel-scoped case is caller-requested, intended
narrowing (not an unpredictable staleness exclusion analogous to Blocker 2), and the unscoped
`Thread` traversal is guaranteed to match for any message inserted via this codebase's own
self-guarding HEAD/NEXT write paths — the one case where it would legitimately discard a live ANN
candidate is an orphaned message "unreachable from a HEAD," which `scripts/backfill_thread_ids.sh`'s
own header comment documents as residue of pre-v2 write defects, not a live, systemic gap like
Blocker 1's 100%-of-pre-Stage-A-data finding. Suggested fix: replace "that join is an `OPTIONAL
MATCH` (never discards a seed row)" with the accurate mechanism — e.g., "that join's `Thread`/
`Channel` matches are required, not optional, but (a) the `Thread` match holds for any message
reachable via this codebase's self-guarding HEAD/NEXT write paths, and (b) a channel-scoped match
narrowing to fewer results is the caller's own intended scope, not a staleness exclusion — so
neither currently has Blocker 2's 'silently drops rows the caller didn't ask to exclude' shape." If
a future change makes messages orphan-prone again, or the docstring's confident wording is relied
on to skip over-fetching `msg_hits` without re-checking, this claim would then be the thing that's
wrong — flagging now while it's cheap to fix in the same diff that authored it.

### What's solid

- **Blocker 1's script is a faithful, idempotent mirror of an established precedent**
  (`backfill_thread_ids.sh`), touches only rows with a genuinely absent property (never reopens an
  explicitly-superseded row — proven by a dedicated test, not just claimed), and its live effect
  was independently re-probed by the coordinator against both real affected workspaces before this
  gate.
- **Blocker 1's regression tests genuinely close the coverage gap Pass 3 found** — a raw `REMOVE`
  fixture (not a `SET … = false` stand-in) exercises the "property absent" axis that every prior
  fixture missed, and the tests run the real shipped script as a subprocess rather than a
  duplicated Cypher string, so a regression in the script itself is what they'd catch.
- **Blocker 2's fix lands on the correct call site, with the correct base variable, and no
  interaction hazard with Blocker 1's fix** — verified by reading `self._k`'s two production
  callers (bounded constructor-time constants, not open-ended user input) and by tracing that only
  `k` (candidate depth) changed, never `limit` (response cap).
- **Both mutation-test claims reproduced independently, byte-for-byte matching the delegate's
  reported "3 tests" each**, with the working tree restored and md5-verified clean afterward.
- **The two updated call-tuple assertions are genuine, arithmetic-checking updates**, not loosened
  ones — confirmed by reading and by the second mutation (removing the multiplier) failing exactly
  those two tests plus the new hard-negative.

### Open questions

- Whether the orphaned-message edge case surfaced while checking Finding 1 (a message unreachable
  from any `Thread` HEAD, per `backfill_thread_ids.sh`'s own documented caveat, would silently drop
  out of `hybrid_search`'s `msg_hits` pool exactly like a pre-migration chunk did before Blocker 1's
  fix) is live in any current workspace — not checked here, out of this diff's scope, and distinct
  from both of Pass 3's blockers (neither of which concerned the `Message` pool).

## Pass 6 — Stage D (uncommitted, "update-detection" FR-2/AC-1/AC-2)

**Scope.** Diff-scoped review of the currently uncommitted working-tree diff implementing Stage D
(document-ingestion2 §4 Stage D) against `falkor-chat/docs/plans/document-ingestion2.md` §3.4/§4
Stage D and `falkor-chat/docs/plans/document-ingestion2-ml.md` §3/§4.1. Touched: new
`server/falkorchat/update_detection.py` + `server/tests/test_update_detection.py`; changed
`repository.py` (`create_document_with_auto_supersede`, `find_update_shortlist`), `services.py`
(`ingest_document`'s new flow), `background.py` (`_safe_detect_update`/
`_schedule_update_detection`), `api.py`/`mcp.py`/`app.py` (transport wiring), `bootstrap_schema.sh`
(the `lshBand0..7`/`textNormalizedHash` indexes + `Document.title` fulltext), and the corresponding
test files. Baseline: `git diff` in `falkor-chat/` scoped to exactly the file list in the dispatch
brief (confirmed empty diff on every other tracked/untracked path via `git status --short`).
`docs/plans/document-ingestion2-coordination.md`'s Stage D entry (the dispatching coordinator's own
independent spot-check — full-suite re-run, concurrency-probe re-run, direct Cypher read) is
accepted as already corroborated per the brief; re-verified two of its specific claims directly
below rather than re-running the whole thing blind.

**Verdict: approve with suggestions.** No blockers. Four minor findings, none load-bearing on
correctness — the core mechanism (atomic auto-supersede, soft-failure background detection,
transport wiring, pure-function detection math) is sound and well-tested; the findings are gaps in
test/verification coverage and one plan-mandated deliverable (`test_queries.sh`) left for a
follow-up.

**CPG: considered, not relevant** — matches the plan's own §2 scoping (new-code design, read
directly) and Pass 1/2's disposition for the same feature; also `cpg_falkorchat` predates this
uncommitted diff entirely (unlike Pass 3/4, which had a pre-existing graph to query against). Direct
reading of `update_detection.py`, the two new/changed `repository.py` methods, and `EXPLAIN`
probes against the live `ws:test` graph (below) were the right tools for this diff's actual
questions.

### What I verified

- Read the plan (§0, §2.1, §3.4, §4 Stage D, §5, §7), the ML note in full (§3/§4.1/§4.2), and Pass
  1–5 of this file (Pass 3/Pass 5 for calibration on scrutiny level, per the brief).
- Read every hunk of `git diff` across all 13 target files (both new files in full, not excerpts).
- **The atomic auto-tier write** (`repository.create_document_with_auto_supersede`,
  `repository.py:1714-1911`): traced the Cypher clause by clause. (a) The `OPTIONAL MATCH` for
  `candidate` is bound in the same first `WITH`/`FOREACH` block that `CREATE`s the new `Document`,
  strictly before that `CREATE` — the new document can never match itself. (b) The guard is
  `(ok AND candidate IS NOT NULL) AS doSupersede`, not `candidate IS NOT NULL` alone — read
  directly, matching the coordination ledger's own independent read. (c) Two genuinely separate
  `FOREACH`-guarded blocks (one `SET`s the candidate + stamps `supersededAt`/`supersededBy`, a
  second later block `CREATE`s the `SUPERSEDES` edge), sharing intermediate `WITH`s — not one
  `FOREACH` mixing `SET`+`CREATE`, the form the plan explicitly says is wrong even though
  equivalent-looking. (d) Chunks are bulk-flipped via their own `OPTIONAL MATCH`+`collect`+`FOREACH`
  step between the two guarded blocks (a `FOREACH` body can't itself contain a `MATCH`), verified
  correct via `test_create_document_with_auto_supersede_flips_old_document_and_its_chunks`.
- **Mutation-tested the compound guard myself**, independent of the coordinator's read-only check:
  copied `repository.py` to `/tmp` (md5-recorded), changed `(ok AND candidate IS NOT NULL)` to
  `(candidate IS NOT NULL)`, ran `pytest tests/test_repository.py -k auto_supersede` — **1 failed,
  8 passed**, the failure landing exactly on
  `test_create_document_with_auto_supersede_unknown_actor_leaves_hash_colliding_candidate_untouched`
  (as a FalkorDB `ResponseError: Failed to create relationship; endpoint was not found`, since the
  mutated guard lets `doSupersede` go true with `d` never created — a different failure *shape*
  than a clean assertion failure, but still a hard failure any CI run would catch). Restored from
  the `/tmp` copy; md5 confirmed byte-identical. Re-ran the full scoped suite after restore — clean
  (below).
- **`find_update_shortlist`'s index-anchored `OR`** (`repository.py:1913-1968`): the query is
  `MATCH (d:Document {currentVersion: true}) WHERE d.lshBand0 = $band0 OR ... OR d.lshBand7 =
  $band7 ...` — functionally scoped correctly (never actor-scoped, `currentVersion`-only per ML note
  §4.1, confirmed both by reading and by
  `test_find_update_shortlist_excludes_noncurrent_documents`). See Finding 1 below on the one gap
  this surfaced: this literal shape (inline `{currentVersion: true}` on the `MATCH` pattern) is not
  byte-identical to what `graph-dba`'s second live-verification pass profiled (`MATCH (d:Document)
  WHERE d.lshBand0 = $band0 OR ...`, no `currentVersion` predicate) or to the shape
  `claude/graph-dba/falkordb-quirks.md`'s dated 2026-09-11 entry documents.
- **The unlabeled-endpoint planner-trap discipline**: grepped the full diff for `SUPERSEDES` —
  exactly one occurrence outside prose/comments, the `CREATE` inside
  `create_document_with_auto_supersede` (no `MATCH` of an existing `SUPERSEDES` edge anywhere in
  this diff). `find_update_shortlist` never touches `SUPERSEDES` at all (`Document`-only). This
  confirms the coordination ledger's "the discipline literally doesn't apply to this new code"
  claim directly, not by re-trusting it.
- **The background job's soft-failure discipline** (`background._safe_detect_update`,
  `background.py:226-296`): read the function in full — no `_report_document_job` call anywhere in
  it or on any path it reaches, no write to `Document.status`/`pendingJobs`. `test_background.py`
  independently pins this with three dedicated failure-injection tests
  (`get_document`/`find_update_shortlist`/`create_or_reopen_supersede_suggestion` each raising),
  each asserting the exception is logged and swallowed, never propagated — genuine regression
  coverage of the docstring's claim, not just the claim itself.
- **AC-1/AC-2 test coverage, at the right altitude**: `test_api.py::wired_real_ingestion` and
  `test_mcp.py`'s real-`repo` `_configure(..., detection_repo=repo)` path both run AC-1
  (`test_ingest_document_byte_identical_modulo_whitespace_auto_supersedes[_tool]`) and AC-2
  (`test_ingest_document_plausible_edit_produces_pending_suggestion...`) against a real FalkorDB
  connection (`conn` fixture), not `FakeRepo` — confirmed by reading the fixtures, not assuming from
  the name. These would fail if the real Cypher/wiring were broken, unlike a mock-level test.
  `test_services.py`'s parallel `FakeRepo`-based tests are a legitimate additional unit-level pin
  (this file's own documented "review-safe subset" posture, `falkor-chat/AGENTS.md`), not a
  substitute for the above.
- **Wiring completeness**: read `api.py`'s two ingest routes and `mcp.py`'s two tools — both gate
  `_schedule_update_detection` on `if not receipt.get("autoSuperseded")`/`if not
  receipt.get("autoSuperseded")`, a genuinely separate scheduling call from chunk
  processing, matching the plan exactly. See Finding 2 below for what a regression here would (and
  wouldn't) be caught by.
- **The known v1 limitation** (ML note §4.1/§7): read `update_detection.py`/`find_update_shortlist`
  in full for any undiscussed heuristic that might silently close this gap (e.g. a text-only
  fallback ignoring title, or a wider candidate net) — found none; the two-signal (LSH-band + title-
  fuzzy) shortlist is exactly as designed, the gap stands as an accepted, unaddressed limitation.
- **Mutation-test claims** (coordination ledger, delegate's reported 3 mutations): independently
  reproduced one (the compound-guard weakening, above) rather than just judging it plausible. Did
  not independently reproduce the other two (the two-round-trip-race split, the shortlist-scoping
  bug) — the coordinator's own independent live re-run of the concurrency probe
  (`test_create_document_with_auto_supersede_concurrent_calls_produce_exactly_one_edge`, passing)
  and this pass's own reading of `find_update_shortlist`'s scoping (confirmed correct, above, via
  both source and `test_find_update_shortlist_excludes_noncurrent_documents`) are accepted as
  sufficient corroboration for those two without re-running the mutations myself.
- **Full suite**: ran the scoped set (`test_repository.py`, `test_services.py`,
  `test_background.py`, `test_api.py`, `test_mcp.py`, `test_update_detection.py`) after restoring
  the mutation — **805 passed**, live, against the shared FalkorDB instance (`redis-cli -p 6379
  ping` → `PONG`). Did not re-run the full 2812-test suite (accepted the coordinator's own
  independent re-run per the brief).
- **`EXPLAIN` probes against `ws:test`** (which already carries every Stage D index, from the
  delegate's own test runs) to check Finding 1: `MATCH (d:Document {currentVersion: true}) WHERE
  d.lshBand0 = 'a' OR ... OR d.lshBand7 = 'h' ...` plans as `Limit → Project → Node By Index Scan |
  (d:Document)` — no `Filter`, no `Node By Label Scan`. As a control, the same query with a
  genuinely unindexed property instead of `currentVersion` correctly plans `Filter → Node By Label
  Scan`, confirming `EXPLAIN` does surface a residual `Filter`/label-scan when one is actually
  present, so its absence here is meaningful. `PROFILE` (which would show real `Records produced`
  per operator, the only way to know *which* index actually anchored the scan) isn't available
  through the `cypher` MCP tool — see Finding 1 for why this doesn't fully close the question.

### Findings

**Minor — `find_update_shortlist`'s live Cypher (`repository.py:1913-1968`) is not the literal
shape `graph-dba` profiled, and the difference is exactly the kind this build's planner is known to
be sensitive to.** The plan's §0/§7 and `claude/graph-dba/falkordb-quirks.md`'s dated 2026-09-11
entry both live-verify `MATCH (d:Document) WHERE d.lshBand0 = $band0 OR ... OR d.lshBand7 =
$band7 ...` — no `currentVersion` predicate anywhere in the profiled shape. The shipped query adds
`{currentVersion: true}` as an inline pattern-property on the same `MATCH`, to satisfy the ML
note's "never compare against an already-superseded document" requirement (§4.1) — functionally
correct (confirmed by `test_find_update_shortlist_excludes_noncurrent_documents`) but a materially
different query for the planner, not the byte-identical string that earned the "single `Node By
Index Scan`, no label scan" guarantee. My own `EXPLAIN` check (What I verified, above) is
consistent with the combination still folding cleanly into one index scan — matching the pattern
`falkordb-quirks.md` separately documents for "two independently-indexed predicates fold into ONE
scan, both evaluated inside it" (lines 689-706) and "a guarded-CAS `WHERE` on a second indexed
property folds into the scan" (lines 679-687) — but `EXPLAIN` alone cannot show *which* index
actually anchors the scan or real `Records produced` counts the way `PROFILE` with planted probe
rows can, and this specific combination (a pattern-property equality plus an 8-way cross-property
`OR`) was never the literal shape live-verified. Suggested fix: a short `graph-dba` follow-up —
`GRAPH.PROFILE` the exact shipped query against a disposable probe workspace with planted rows (a
document matching only on `currentVersion` but no band, a document matching one band but
`currentVersion: false`, per the existing `ws:docprobe2` methodology) to confirm `Records produced`
tracks true selectivity rather than falling back to a `currentVersion`-anchored near-label-scan —
then fold the result into `falkordb-quirks.md` as a dated addendum to the existing entry, the same
way the entry itself was added.

**Minor — the `autoSuperseded` gate that skips scheduling `_safe_detect_update` after a synchronous
auto-supersede (`api.py`/`mcp.py`, `if not receipt.get("autoSuperseded")`) has no direct regression
test, and the existing AC-1 end-to-end tests cannot catch its accidental removal.** Confirmed by
tracing what would happen if the guard were deleted: in every AC-1 test's fixture, the only
document sharing content with the just-auto-superseded pair is the old version itself, which
`find_update_shortlist` already excludes via its own `currentVersion` scope — so an errantly-
scheduled detection job would run, find no candidates, and write nothing, leaving every current
assertion green. This is exactly the "harmless but real" shape the dispatch brief asked to check
for, and the guard is genuinely present (confirmed by reading), but a future accidental removal
would silently reintroduce wasted background work (and, in a workspace with an unrelated
lexically-similar document also present, a spurious pending suggestion against a document that just
got auto-decided) with no test failure to flag it. Suggested fix: one spy-based unit test in
`test_api.py`/`test_mcp.py` (or `test_background.py`, monkeypatching `_schedule_update_detection`)
asserting it is *not* called when `ingest_document` returns `autoSuperseded: True`, mirroring the
existing spy pattern already used for the chunk-processing scheduler
(`test_schedule_chunk_processing_with_no_repo_on_either_worker_does_not_raise`'s general shape).

**Minor — Stage D adds two genuinely new Cypher shapes (the atomic auto-supersede write, the
LSH-band shortlist lookup) but `scripts/test_queries.sh` is untouched by this diff**, despite the
plan's own §5 test-strategy explicitly naming it: "every new Cypher shape... raises the enumerated
baseline — exact queries are `graph-dba`'s to author/verify, not enumerated here." Every prior
stage's gate (Pass 1's delete query, Pass 2's `SUPERSEDES` confirm/reject/reopen queries, Pass 3/5's
search filter) landed its `test_queries.sh` section in the same commit; this stage's repository/
service/API/MCP coverage is thorough on its own (confirmed above), so this is not a functional gap,
but it is a plan-mandated deliverable left open. Suggested fix: a fast `graph-dba` follow-up unit
(or fold into the Finding 1 follow-up above, since both need the same disposable-probe-workspace
methodology) adding the auto-supersede write and the shortlist `OR` lookup to `test_queries.sh`
before Stage E closes.

**Nit — AC-1's own test-strategy wording ("search returns only the new version") is proven by
combining two independently-tested mechanisms, never by one test exercising both through the real
auto-supersede write path.** `test_create_document_with_auto_supersede_flips_old_document_and_its_
chunks` (Stage D, this diff) proves `Chunk.documentCurrent` flips correctly on a real
auto-supersede; `test_search_documents_excludes_a_superseded_documents_chunks` (pre-existing, Stage
C/Pass 3-5) proves `search_chunks`'s filter honors `documentCurrent` — but that test flips the
property via `_mark_document_superseded_raw` (a raw fixture write), not by calling
`create_document_with_auto_supersede`/`POST /documents` twice. Both halves are solid and share the
exact same property, so the combined risk is low — flagging only because the AC-1 row's literal
wording asks for the combination and no single test currently proves it end to end. Suggested fix:
extend `test_ingest_document_byte_identical_modulo_whitespace_auto_supersedes` (or its MCP twin)
with one more assertion — a `client.get("/documents/search", params={"q": ...})` call (with a real
embedded chunk, mirroring `search_client`'s setup) confirming the old document's chunk is excluded.

### What's solid

- **The atomic write is exactly the live-verified shape**, confirmed by direct clause-by-clause
  reading and by independently reproducing the one mutation this pass had time to run myself (the
  compound-guard weakening) — caught, and the restore was verified byte-identical via md5.
- **The soft-failure discipline is real, not just documented.** `_safe_detect_update` touches
  nothing but `repo.get_document`/`find_update_shortlist`/`create_or_reopen_supersede_suggestion`,
  and three dedicated failure-injection tests independently pin that a failure at any of those three
  steps is logged and swallowed, never touching `Document.status`/`pendingJobs`.
- **AC-1/AC-2 both have genuine end-to-end coverage against a real FalkorDB connection**, on both
  REST and MCP, not mocked at a level that would pass with broken wiring — confirmed by reading the
  fixtures, not the test names.
- **The known v1 candidate-generation gap is left exactly as designed**, no undiscussed heuristic
  papering over it — confirmed by reading `find_update_shortlist`/`update_detection.py` in full.
- **The unlabeled-`SUPERSEDES`-endpoint discipline needed no new application here** — verified
  directly (grep + read), not re-trusted from the coordination ledger's own claim.
- **Pure-function coverage in `test_update_detection.py` mirrors the ML note's own named cases**
  (empty text, whitespace-only variance, case/whitespace-identical pairs, the shared-boilerplate
  hard negative) plus the MinHash/LSH additions `graph-dba`'s redesign introduced — genuinely
  thorough, not a token smoke test.
- **Transport wiring is symmetric and correctly gated** — both `api.py` routes and both `mcp.py`
  tools schedule `_safe_detect_update` via the identical `if not
  receipt.get("autoSuperseded")`/`_schedule_update_detection` pattern, sourcing `repo` through the
  same `getattr(embed_worker, "repo", None) or getattr(ingestion_pipeline, "repo", None)` idiom
  `_schedule_chunk_processing` already established.
- **The `ingest_documents` malformed-item defense (the docstring's self-flagged fix)** is a genuine,
  correctly-scoped defensive improvement — validated `isinstance` checks before dispatch, widened
  `except` clause, and the batch-loop restructuring needed to let update-detection scheduling run
  independently of whether `embed_worker`/`ingestion_pipeline` are wired — read directly, not just
  from the docstring's own claim.

### Open questions

- Whether Finding 1's `PROFILE`-with-planted-probe-rows follow-up should block Stage E's dispatch
  (a genuine planner-behavior question on the suggested tier's cost characteristics) or land as a
  fast, narrow `graph-dba` unit alongside the Finding 3 `test_queries.sh` follow-up — both need the
  same disposable-workspace methodology, so bundling them is likely the efficient sequencing, but
  that's `teco`'s call.
- Whether the `_UPDATE_DETECTION_NOISE_FLOOR = 0.1` constant (`background.py`) should be documented
  anywhere more discoverable than its own module comment — it's exactly the "implementer-tunable,
  not load-bearing" posture the plan explicitly sanctions (§3.2/§5), so not a defect, just flagging
  in case `qa-engineer`'s Stage E test plan wants to name it explicitly as a known tuning knob when
  writing the "plausible-but-not-identical" acceptance scenario.

## Pass 7 — Stage D-fix (uncommitted, Pass 6 Findings 2+3), 2026-09-13

**Scope.** Diff-scoped re-check of the currently uncommitted working-tree diff closing Pass 6's
Finding 2 (no regression test for the `autoSuperseded` scheduling-skip guard) and Finding 3 (no
`test_queries.sh` coverage of Stage D's two new Cypher shapes). Touched: `server/tests/test_api.py`
(one new test, `test_ingest_document_auto_supersede_skips_scheduling_update_detection`) and
`scripts/test_queries.sh` (new `§14.10` section). Pass 6's Finding 1 (`find_update_shortlist`'s
live shape vs. the profiled shape) was closed separately by a `graph-dba` diagnostic consult,
already committed (`19ad080`) — not re-examined here. Baseline: `git diff -- falkor-chat/server/
tests/test_api.py falkor-chat/scripts/test_queries.sh`. This is a re-check of two already-reported
findings, not a fresh full review — see the disposition lines below rather than re-argued analysis.

**Verdict: approve.** Both findings closed; no new findings.

**CPG: considered, not relevant** — same disposition as Pass 5/6 for this feature: the diff adds
test-only assertions against two already-enumerated symbols/shapes (Pass 6 already traced
`create_document_with_auto_supersede`/`find_update_shortlist` clause by clause and confirmed the
`_schedule_update_detection` import/call site), no new caller, no new symbol. Direct reading plus
a live `test_queries.sh` run were the right tools for this diff's actual questions.

### Finding 2 — disposition: fixed

Verified `test_ingest_document_auto_supersede_skips_scheduling_update_detection`
(`server/tests/test_api.py:1038-1082`) genuinely exercises both branches of the
`if not receipt.get("autoSuperseded"):` guard (`api.py:190`), not just the happy path:

- **Wiring is correct.** `api.py:17-21` imports `_schedule_update_detection` by name
  (`from .background import ... _schedule_update_detection`) into its own module namespace, and the
  route body calls the unqualified name — so `monkeypatch.setattr(api_mod,
  "_schedule_update_detection", ...)` patches exactly the reference `ingest_document` resolves at
  call time (the standard "patch where used" shape, confirmed by reading both files, not assumed).
- **Args-indexing is correct against the real signature.** `background._schedule_update_detection(
  schedule, repo, ws, document_id)` (`background.py:295-296`) is called from `api.py:191-193` as
  `_schedule_update_detection(background.add_task, repo, ctx.ws, receipt["documentId"])` — so
  `args[3]` in the spy lambda is `document_id`, matching what the test asserts against
  (`scheduled_for.append(args[3])`).
- **Both branches are genuinely exercised.** The first ingest (nothing to collide with) asserts
  `scheduled_for == [old_id]`, proving the spy itself fires on the non-auto-superseded path before
  the interesting assertion; the second, byte-identical-modulo-whitespace ingest asserts
  `autoSuperseded is True` and then `scheduled_for == [old_id]` again (unchanged) plus `new_id not
  in scheduled_for` — proving the skip branch.
- **Mutation-tested myself, independent of the delegate's own claim.** Copied `falkorchat/api.py`
  to `/tmp` (md5-recorded), replaced the guard at line 190 with `if True:` (scoped to the single-
  document route only, not the `/documents/batch` guard at line 230 — a different, untouched code
  path), and re-ran the new test: **1 failed**, exactly at the second `assert scheduled_for ==
  [old_id]` (`AssertionError: … Left contains one more item: '<new_id>'`) — the guard's removal is
  caught, not silently absorbed. Restored `api.py` from the `/tmp` copy; `md5sum` confirmed
  byte-identical to the pre-mutation file before and after. Re-ran the un-mutated test afterward —
  green (`1 passed`).

### Finding 3 — disposition: fixed

Verified the new `§14.10` section (`scripts/test_queries.sh:1816-1907`) covers both Stage D Cypher
shapes named in the plan (§3.4), with assertions that would fail on a real regression, not just
smoke checks:

- **`create_document_with_auto_supersede`'s atomic write.** The section's `CREATE_DOC_WITH_AUTO_
  SUPERSEDE` variable is byte-identical to the shipped query (`repository.py:1779-1831`, diffed
  clause by clause) and the assertions exercise the full atomic behavior in one call: the old
  document flips `currentVersion=false` + gets `supersededAt`/`supersededBy` stamped, its `Chunk`
  bulk-flips `documentCurrent=false`, the new document is current, and the `SUPERSEDES{status:
  'confirmed', decidedBy:'system', confidence:1.0}` edge is created — six independent property/edge
  assertions, not one "it returned 200"-style check. A separate probe (`asd3`, fresh content with
  no real candidate) asserts `autoSuperseded=false` and zero `SUPERSEDES` edges, closing the
  self-match axis the query's `OPTIONAL MATCH`-before-`CREATE` ordering is meant to guarantee.
- **`find_update_shortlist`'s two signals.** Both `FIND_SHORTLIST_BANDS` and
  `FIND_SHORTLIST_TITLE` are byte-identical to the shipped queries (`repository.py:1881-1901`). The
  band-OR fixture (`lsd1`/`lsd2` sharing `lshBand3`, differing only in `currentVersion`) and the
  title-fuzzy fixture (`lsd3`/`lsd4` both fuzzy-matching `%Quarterly%`, differing only in
  `currentVersion`) are genuinely discriminating, not tautological — each asserts the current
  match is included AND the non-current match sharing the exact same signal is excluded, so a
  regression that dropped the `currentVersion` scoping from either query would fail the
  corresponding `assert_not_contains`, not just leave a `assert_contains` vacuously true. `lsd5`
  (matches neither signal) is a third, independent negative control. The band-OR shape is also
  checked with `assert_index_scan` (`GRAPH.PROFILE`) — live-confirmed `Node By Index Scan`, no
  label scan — closing the "is this still the live-verified shape" half of Pass 6 Finding 1's
  concern for the band query specifically (the title-fuzzy query, a `CALL db.idx.fulltext.
  queryNodes` + post-`YIELD` filter, isn't a scan-anchoring question the same way).

### What I verified beyond the two findings

- Ran `./scripts/test_queries.sh` live against the up FalkorDB instance: **458/459 passed**. The
  one failure is `§14.8 SUPERSEDES.matchId lookup … uses the relationship index, no label scan`
  (`Edge By Index Scan` vs. the helper's `Node By Index Scan` string) — the pre-existing,
  already-logged `assert_index_scan` helper gap named in the brief, unrelated to this diff and not
  re-flagged as new. All 20 `§14.10` assertions passed.
- Ran the full Python suite: `cd falkor-chat/server && .venv/bin/python -m pytest -q` — **2813
  passed, 14 deselected, 0 failed** (2813 = Pass 6's 2812-and-accepted-unrun baseline + this pass's
  one new test).
- Confirmed no other file changed: `git status --short` shows exactly the two target files as
  modified in this unit (plus unrelated concurrent working-tree state called out in the brief as
  out of scope).

### Operational note (not a finding)

`test_queries.sh`'s teardown deletes the shared `reference` graph, as documented
(`falkor-chat/AGENTS.md`'s script table); this run did so. `reference` currently holds a sparse
leftover (1 `WorkflowDef` + 4 `Step` nodes, not a full re-seed) — most likely residual from a
different concurrent session's own test activity in the few seconds since this run's teardown, not
something this pass's diff or verification steps left behind deliberately. Re-seeding
(`bootstrap_schema.sh` → `seed_demo.sh` → `seed_workflows.sh`, per the same table) is out of this
review's scope and is the coordinator's/next session's call, same as any other run of this script.

### Open questions

None beyond Pass 6's still-open items (Finding 1's `graph-dba` `PROFILE`-with-planted-probe-rows
follow-up, already dispatched separately per the brief and not re-litigated here).

## Pass 8 — Stage F (uncommitted, QA Defect 1 fix), 2026-09-13

**Scope.** Diff-scoped review of the currently uncommitted working-tree diff fixing
`document-ingestion2-report.md`'s Defect 1 (`find_update_shortlist`'s title-fuzzy RediSearch query
crashed — silently, inside `_safe_detect_update`'s never-raise isolation — on any `Document.title`
containing a common RediSearch metacharacter), dispatched as Stage F (`document-ingestion2-
coordination.md`, dispatch `a4426e33398d5d973`). Touched: `server/falkorchat/repository.py` (new
`_escape_fuzzy_token` helper, wired into `find_update_shortlist`'s title-fuzzy branch),
`server/tests/test_repository.py` (two new tests), `scripts/test_queries.sh` (new `§14.10`
assertions), `docs/HISTORY.md` (new dated entry). Baseline: `git diff` on exactly these four files
(`git status --porcelain` confirms no other tracked file in scope changed). This is a focused
bug-fix review, not a full feature re-review — scaled to the diff's actual size (one new 4-line
helper, one call-site change, two tests, one script section, one history entry).

**Verdict: approve with suggestions.** No blockers, no majors. One minor (a new code branch this
fix introduces has no dedicated test) and two nits below.

**CPG: considered, not relevant** — the diff changes the internals of one already-enumerated
function (`find_update_shortlist`, exhaustively traced clause-by-clause in Pass 6) and adds one new
private helper with a single caller inside the same file; no new symbol, caller, or edge a
call-graph query would need to find. Direct reading plus live execution (pytest, mutation testing,
`test_queries.sh`) were the right tools for this diff's actual questions.

### What I verified

- Read every hunk of `git diff` across all four target files directly, not a summary.
- **Simulated `_escape_fuzzy_token` against all seven QA characterization-table titles** (`"Report -
  Final"`, `"Spec: v2"`, `'Q3 "Draft" Notes'`, `"Notes [2024]"`, `"A|B test"`, `"Report (Draft)"`,
  `"Plain Clean Title"`) plus the empty-token case: every one now strips to a fuzzy term made up
  only of plain word characters (`%Report% %Final%`, `%Spec% %v2%`, `%Q3% %Draft% %Notes%`,
  `%Notes% %2024%`, `%AB% %test%`, `%Report% %Draft%`, `%Plain% %Clean% %Title%`) — no metacharacter
  survives into any `%...%` term, and the standalone `-` token in `"Report - Final"` strips to `""`
  and is correctly dropped by the `if escaped` filter rather than producing a stray `%%` term. This
  confirms the fix closes Defect 1 for the QA report's own evidence, not just plausibly.
- **Confirmed `fusion._fuzzy_query` (`fusion.py:34-42`) is byte-unchanged** (`git diff` on the file
  is empty) — the deliberate, QA-report-scoped exclusion holds; see Finding 2 (nit) below for the
  one-line asymmetry note the brief asked for.
- **Read both new `test_repository.py` tests in full.**
  `test_find_update_shortlist_title_with_redisearch_metacharacters_does_not_raise` seeds a document
  via `_document_with_lsh` with unrelated filler text/real LSH bands, then calls
  `find_update_shortlist` with a deliberately non-matching `bands=["zzzzzzzzzzzzzzzz"] * 8` — so the
  only signal that can produce a match is the title-fuzzy branch, and the test asserts the seeded
  document **is** found. This is a genuine "still matches" proof, not merely "doesn't raise": the
  band signal is intentionally defeated, isolating the assertion to the fixed code path.
  `test_find_update_shortlist_survives_the_qa_characterization_table` parametrizes over the exact
  seven titles and asserts only `isinstance(candidates, list)` (no seeded document to match against)
  — correctly scoped to "doesn't raise," complementing rather than duplicating the first test's
  "still matches" proof.
- **Ran the new/adjacent tests live**: `pytest tests/test_repository.py -k "fuzzy or
  characterization or shortlist"` — **15 passed** (against FalkorDB, `redis-cli -p 6379 ping` →
  `PONG`).
- **Mutation-tested the fix myself**, independent of the delegate's own claim. Copied
  `repository.py` to `/tmp` (`md5sum`-recorded), reverted just the escaping change (`fuzzy_query =
  " ".join(f"%{tok}%" for tok in title.split())`, dropping `_escape_fuzzy_token` from the call site
  only — the helper itself, and every other line, left untouched), and re-ran the same test
  selection: **7 failed, 8 passed** — every one of the seven metacharacter-bearing cases failed with
  `redis.exceptions.ResponseError: RediSearch: Syntax error at offset 10 near Report` (the identical
  live crash signature the QA report itself reproduced), while `Plain Clean Title` and every
  pre-existing shortlist test stayed green, confirming the mutation is isolated to the defect path
  and the tests fail for the right reason. Restored from the `/tmp` copy; `md5sum` confirmed
  byte-identical before and after. Re-ran the un-mutated selection afterward — **15 passed** again.
- **Ran the full suites live.** `pytest -q` (full): **2821 passed, 14 deselected, 0 failed** —
  matches `HISTORY.md`'s claimed figure exactly. `./scripts/test_queries.sh`: **460/461 passed**;
  the one failure is `§14.8 SUPERSEDES.matchId lookup … uses the relationship index, no label scan`
  (`Edge By Index Scan` vs. the helper's hardcoded `Node By Index Scan` string) — the pre-existing,
  already-logged `assert_index_scan` helper gap named in the brief, not a regression; all 20 new
  `§14.10` assertions (including the two escaping-specific ones) passed. Both figures match
  `HISTORY.md`'s claimed "was 2813/14" → "2821/14" and "was 458/459" → "460/461" deltas exactly.
- **Re-seeded `reference` after `test_queries.sh`'s documented teardown wipe**
  (`falkor-chat/AGENTS.md`'s script table): `bootstrap_schema.sh acme` → `seed_demo.sh acme` →
  `seed_workflows.sh acme`, then `./scripts/verify_workflows.sh acme` → `RESULT: OK — 2 defs in sync
  between reference and ws:acme`.
- **Read `docs/HISTORY.md`'s new entry against two neighboring entries** (the 2026-09-11
  `salesperson-ui S10` entry and the file's header convention) — same dated-`##`-header,
  **What:**/**Verified:** bulleted-bold-lead-in shape; the claimed test figures independently
  re-verified above, not taken on report.
- Confirmed no other tracked file changed: `git status --porcelain` on the four target paths shows
  exactly those four, no side effects from this pass's own mutation-test/backfill activity (working
  tree restored byte-identical, `reference` re-seeded and verified `OK`).

### Findings

**Minor — the fix introduces a new "title present but every token strips to empty" skip branch
(`repository.py`, the `else: fuzzy_query = ""` / `if fuzzy_query:` restructuring) that no test
exercises.** All seven QA characterization-table titles have at least one alphanumeric token, so
none of them hit this branch — the closest, `"Report - Final"`, only has *one* token strip to `""`
(the standalone `-`), leaving `%Report% %Final%` non-empty. A title that is pure punctuation (e.g.
`"---"` or `"(...)"`) would make `fuzzy_query` stay `""` even though `title` itself was truthy,
silently skipping the title-fuzzy lookup entirely — correct, fail-safe behavior per the docstring's
own stated reasoning ("a degenerate query"), but currently unverified by any test, unlike the
already-well-covered "title empty" and "title has real content" axes. Suggested fix: one small
`test_repository.py` case — seed a document, call `find_update_shortlist` with a pure-punctuation
`title` (e.g. `"---"`), assert it returns without raising (mirroring
`test_find_update_shortlist_empty_title_skips_title_lookup`'s existing shape for the analogous
empty-title case).

**Nit — the shared root cause is now visibly asymmetric between the two call sites, as the QA report
itself flagged as a scoped-out follow-up, not an oversight here.** `fusion._fuzzy_query`
(`fusion.py:34-42`) still builds its `%token%` terms from raw, unescaped entity-name tokens — the
identical crash shape Defect 1 fixed for `find_update_shortlist`, just for a different
`RediSearch.queryNodes('Entity', ...)` call, currently lower-risk only because LLM-extracted entity
names rarely carry punctuation (this diff's own docstring says so directly). Correctly out of scope
per the QA report's explicit deferral — not re-flagging as a defect, just confirming the asymmetry
is real and worth keeping on the backlog so both call sites eventually share one escaping helper
rather than diverging further.

**Nit — stripping (not escaping) merges some distinct words into one token, changing match
semantics for one QA-table case without breaking it.** `"A|B test"` strips to tokens `["AB",
"test"]`, not two independent fuzzy terms — `%AB%` will fuzzy-match differently than a title
genuinely containing the word "AB" would suggest (e.g. it now also 1-edit-fuzzy-matches "AB" as a
whole rather than "A" and "B" separately). Not a defect: the docstring's own justification for
stripping over backslash-escaping (surviving characters must be plain word characters for `%...%`
fuzzy matching to behave sensibly) is internally consistent, and no QA-table case regresses from OK
to CRASH or vice versa. Worth a one-line note only because a future reader diffing "why does `A|B`
title-fuzzy-match differently than `A B` would" might otherwise assume a bug rather than a
documented trade-off.

### What's solid

- **The fix genuinely closes Defect 1** — independently simulated and live-tested against all seven
  QA characterization-table titles, every one now produces a syntactically valid RediSearch query,
  confirmed by both the new `test_repository.py` cases and the new `test_queries.sh` §14.10
  assertions passing live.
- **At least one test proves a real match survives the escaping**, not just "doesn't raise" —
  `test_find_update_shortlist_title_with_redisearch_metacharacters_does_not_raise` deliberately
  defeats the band signal so only the title-fuzzy path can produce its asserted match.
- **The mutation test reproduces the exact live crash signature** the QA report itself hit
  (`RediSearch: Syntax error at offset 10 near Report`), on exactly the seven metacharacter-bearing
  cases and no others — independently confirmed here, not just judged plausible from the delegate's
  own claim.
- **`test_queries.sh` §14.10's new assertions are genuinely discriminating**, not smoke checks —
  one asserts no `Syntax error` substring appears, a second independently asserts the seeded
  document (`lsd6`) is actually found, so a regression that silently degraded into "never matches
  anything" would be caught by the second assertion even if the first stayed green.
- **`fusion._fuzzy_query` is confirmed byte-unchanged**, matching the QA report's explicit scoping
  of this fix to `find_update_shortlist` only.
- **`docs/HISTORY.md`'s new entry follows the file's established convention** and its quantitative
  claims (pytest 2821/14, `test_queries.sh` 460/461, both deltas) independently re-verified against
  live runs, not taken on report.
- **The working tree was left exactly as found** — this pass's own mutation-test backup/restore is
  `md5sum`-verified byte-identical, and the `reference` graph wiped by `test_queries.sh`'s documented
  teardown was re-seeded and re-verified `OK` before finishing.

### Open questions

None beyond the pre-existing, already-tracked items: the shared `fusion._fuzzy_query` gap (Finding 2
above, already on the QA report's own deferred list) and Pass 6's still-open Finding 1 follow-up
(unrelated to this diff).
