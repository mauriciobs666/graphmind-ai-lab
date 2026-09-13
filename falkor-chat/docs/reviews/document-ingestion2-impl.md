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
