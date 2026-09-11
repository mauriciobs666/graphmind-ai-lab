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
