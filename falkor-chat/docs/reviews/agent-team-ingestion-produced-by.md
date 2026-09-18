# `produced_by` document-ingestion attribution — implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (M6)

## Scope & verdict

Diff-scoped review of `git diff -- falkor-chat/server/` (12 files: `falkorchat/{repository,
services,app,mcp,schemas,api,storefront_api}.py` and their five sibling test files) against the
implementation spec at `falkor-chat/docs/plans/agent-team-ingestion-graph.md` (read in full, all
535 lines). Scope is the diff on disk and its conformance to that spec — not the spec's own
soundness, which is out of scope per the brief (already reviewed twice at the parent-plan level).

**Verdict: approve.**

**CPG:** considered, not relevant — `cpg_falkorchat` predates `document-ingestion2`'s ship (per the
design note's own §0 grounding note and `falkor-chat/AGENTS.md`'s CPG-staleness convention), so it
cannot answer questions about code this recent; every finding below is from a direct read of the
diff and the surrounding source, plus one live suite run and one live-graph read/repair (below).

## Findings

No blockers, no majors. Two low-stakes findings below; everything else in the diff conforms
exactly to the design note.

### Minor — the two new query-text constants sit at module top, not beside their method

`repository.py:190-199` places `_INGESTOR_RESOLVE_BY_ACTOR`/`_INGESTOR_RESOLVE_BY_PRODUCER` at
module scope, immediately before `class Repository:` (line 202) — 1,558 lines above
`create_document_with_auto_supersede`, the only method that references them (`repository.py:1833`).
The design note's own instruction was "place beside the method, same file"
(`agent-team-ingestion-graph.md` §2.2), and this file already has a precedent for exactly that
shape: `_SINCE_PLAIN`/`_SINCE_KEYSET` (`repository.py:928-932`) sit directly above
`read_thread_since`, the one method that uses them. The new constants instead landed next to
`_escape_fuzzy_token` — an unrelated, coincidentally-adjacent top-of-file function — purely because
that happened to be the first `Edit` insertion point before the class body. A reader who reaches
`resolve = _INGESTOR_RESOLVE_BY_PRODUCER if …` at line 1833 has no clue what either constant
contains without a file-wide search.

Not a correctness issue — verified by running the targeted (166) and full (2,844) suites, both
green — purely a navigability/fit regression against this file's own established convention.
**Suggested fix:** move both constants (and their explanatory comment) to sit directly above
`create_document_with_auto_supersede`'s `def` line, mirroring `_SINCE_PLAIN`/`_SINCE_KEYSET`'s
placement.

### Nit — a bolded span splits across a docstring line break

`mcp.py:337-338`: `` Returns **one\n    receipt per item**, `` — the `**…**` markdown-bold span
opens on one line and closes on the next. Harmless (still parses as bold in any renderer that
reflows the docstring), but reads oddly in a raw view. Rewrap so the bolded phrase doesn't straddle
the line break.

## What's solid

- **§2 (repository query)** — the diff is a byte-for-byte match to the design note's shown method
  body: the two module constants, the `resolve`/`params` branching, and the two textual changes
  inside the previously-fixed query body (`WITH u, a, ingestor, ok, candidate` →
  `WITH ingestor, ok, ingestorIsUser, candidate`; the `sourceKind` `CASE`) are exactly as specified,
  confirmed by direct diff read (`repository.py:1817-1859`).
- **§3 (`AgentNotFoundError` + `Services.ingest_document`)** — exception shape mirrors
  `DocumentNotFoundError` exactly; the `if produced_by is not None: raise AgentNotFoundError(...)`
  guard sits ahead of the pre-existing `UnknownActorError` raise, structurally guaranteeing no
  silent fallback (confirmed by tracing `create_document_with_auto_supersede`'s producer branch,
  which never references `$ingestedBy`).
- **§4 (per-item `produced_by` in `ingest_documents`)** — the added
  `isinstance(doc.get("produced_by"), (str, type(None)))` check, `resolved_actor` bookkeeping in
  `FakeRepo`, and the batch-loop threading all match §4 exactly.
- **§5 (MCP/REST surface)** — `mcp.py`'s new parameter, `schemas.py`'s `producedBy` field
  (reusing `MAX_NAME_LEN`, not inventing a new constant), and `api.py`'s threading into both
  `/documents` and `/documents/batch` are unchanged from the design note's shown diffs.
- **§7 test coverage** — every named sibling test exists at the claimed location with a sensible
  name: repository (`test_create_document_with_auto_supersede_produced_by_resolves_agent_not_actor`,
  `..._missing_agent_nothing_written`), services (success, `AgentNotFoundError`-not-
  `UnknownActorError`, per-item independence, malformed-item), API (404 with
  `{"error": "AgentNotFoundError"}`, batch per-item), MCP (round-trip, unresolvable, batch
  per-item, malformed-item). Nothing in §7's list is missing or partial.
- **Regression-proof integrity** — independently re-diffed (not trusted from the brief):
  `test_repository.py`'s `_auto_supersede` block (lines ~1487-1735, the pre-existing
  `create_document_with_auto_supersede` tests) shows zero changes beyond the helper's new
  `produced_by=None` kwarg; `test_services.py`'s
  `test_ingest_document_unknown_actor_raises_instead_of_silent_write` and
  `test_ingest_document_known_agent_actor_source_kind_agent` are untouched; `test_api.py`'s
  `:269`/`:274` region is untouched. Confirmed via `git diff` hunk boundaries, not the brief's
  say-so.
- **Storefront classification (finding 4 in the brief)** — confirmed correct by
  `grep -n "ingest_document" falkor-chat/server/falkorchat/storefront_api.py`: zero matches. No
  storefront route reaches document ingestion, so `AgentNotFoundError`'s addition to
  `SERVICE_ERRORS_UNREACHABLE` with that stated reason is accurate, and the two bumped literal
  counts (12→13) were independently recomputed by enumerating `ServiceError.__subclasses__()` at
  runtime — 13, matching.
- **Mutation-testing sanity checks** — (a) traced by hand: deleting the `AgentNotFoundError` raise
  falls through to the pre-existing `raise UnknownActorError(ctx.actor)`, which is still an
  `Exception` — so `test_ingest_document_produced_by_unresolvable_agent_errors`'s
  `pytest.raises(Exception)` (mirroring the file's pre-existing
  `test_ingest_document_unknown_actor_errors` convention) would *not* catch the mutant, while the
  `test_services.py` (`pytest.raises(AgentNotFoundError)` specifically) and `test_api.py` (asserts
  404 + `"AgentNotFoundError"`, and `UnknownActorError` isn't in `app.py`'s `not_found` tuple so it
  would 400 instead) tests would — exactly the claimed 2-of-3 mechanism, not a coverage gap. (b) the
  `kaizen_team` entry (`entryId a1c4e6b2…`, dated 2026-09-18, authored `coder`) matches the claimed
  reconstruction trap almost verbatim, and `claude/graph-dba/falkordb-quirks.md`'s new entry
  (also dated 2026-09-18) independently corroborates the underlying index-scan-vs-label-scan quirk
  the design note's §2.2/§2.3 and this mutant both rest on. (c) not independently re-derived beyond
  reading the call sites — no red flags found.
- **Full offline suite**: re-ran independently (not taken on the brief's word) — `2844 passed, 14
  deselected, 0 failed`, matching the reported count exactly. Also ran the `produced_by`/
  `agent_not_found`/`storefront_api` subset alone (166 passed) as a faster, targeted check.

## Open questions

None — the diff conforms to the design note in every section checked, and the one drift found
(constant placement) is cosmetic.

## Appendix — live-graph side effect from this review, repaired

Running the full offline suite (`pytest -q`, no `-m live`) exercises `tests/conftest.py`'s
`wf_repo` fixture, which wipes the global `reference` graph's node data on the workspaces this box
happened to have live (`falkor-chat/AGENTS.md`'s documented "default pytest run wipes `reference`"
hazard). This left `ws:demo`/`ws:acme` failing `./scripts/verify_workflows.sh`,
`./scripts/verify_salesperson.sh`, and `./scripts/verify_catalog.sh` (all defs/catalog rows
present in the workspace snapshots, absent from `reference`). Repaired in place, in the documented
order, before finishing this review:

```
EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh demo   && ./scripts/seed_demo.sh demo
EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh acme   && ./scripts/seed_demo.sh acme
./scripts/seed_catalog.sh
./scripts/seed_workflows.sh demo   && ./scripts/seed_workflows.sh acme
./scripts/seed_salesperson.sh demo && ./scripts/seed_salesperson.sh acme
```

Re-verified: `verify_workflows.sh demo`/`acme`, `verify_salesperson.sh demo`/`acme`, and
`verify_catalog.sh` all now report `RESULT: OK`. No `ws:qa-*`/`ws:probe-*` throwaway workspace was
touched (out of scope for this repair — they don't depend on `reference`'s workflow defs the same
way). Noting this for whoever next runs the full offline suite against this shared box: the
brief's own full-suite rerun (2,844/14/0, matching) almost certainly caused and silently left the
same wipe — this review's repair likely also fixes that prior state, but that wasn't verified
before this run since there's no "reference was already broken as of commit X" marker to check
against.
