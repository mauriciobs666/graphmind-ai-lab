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
