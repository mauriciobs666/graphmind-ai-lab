# `Document.status` Reaches a Terminal State — Implementation Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-051

**Scope.** Code review of `tdd-engineer`'s uncommitted fix for K-051 (`Document.status` never
reaches a terminal state — `docs/BACKLOG.md` K-051, `docs/test-reports/document-ingestion-report.md`
Defect 1), coordinated at `docs/plans/document-status-terminal-coordination.md`. Reviewed against
the working tree as of this session. Files in scope: `server/falkorchat/repository.py`
(`create_document`, `start_document_progress`, `report_document_job_done`, ~1026-1120 only —
the concurrent, unrelated K-005 hunk touching `_read_structure` at ~1795-1820 was identified via
`git diff` and excluded), `server/falkorchat/background.py`, `server/falkorchat/embedding.py`,
`server/falkorchat/ingestion.py`, and the 12 new/updated tests across `tests/test_repository.py`,
`tests/test_background.py`, `tests/test_api.py`.

**CPG:** considered, not relevant — `cpg_falkorchat` is flagged stale in the coordination doc
(built before 14 commits touching `falkor-chat/server`, including all of K-050 Stage 6a). The
diff under review is small and self-contained (4 production files, ~150 changed lines), and every
call-site question the brief raised (other callers of `_safe_embed_chunk`/`create_document`, the
`.repo` wiring in `app.py`) was answered more reliably with direct `grep`/`Read` against the live
tree than a graph that predates this feature's own most recent stage.

**Verdict: approve with suggestions.** The completion-counting mechanism is correct, race-safe,
and directly verified against a live FalkorDB instance (not just read). Failure-isolation
discipline is preserved. No blockers. Two minor test/robustness gaps and one documentation-currency
regression (introduced by this diff, not pre-existing) are worth fixing before/shortly after this
lands.

## Findings

### MAJOR — `QUERIES.md` §14 no longer mirrors the shipped Cypher; two new queries are undocumented

`docs/QUERIES.md` §14 is this repo's declared "canonical query library... source of truth for
queries," and §14 specifically follows that convention exhaustively for exactly this feature area
— it documents even the internal, non-public `list_document_chunks` seam in its own subsection.
This diff changes `create_document`'s literal (`repository.py:1029`, `pendingJobs: 0` added) and
adds two brand-new query-bearing methods, `start_document_progress`/`report_document_job_done`
(`repository.py:1069-1119`) — none of which appear in `QUERIES.md` §14.1 or anywhere else in the
file. §14.1's code block there still shows the pre-fix literal verbatim (no `pendingJobs`).

The coordination doc's documentation-impact note ("No DESIGN.md/QUERIES.md impact — no new
node/index/query, a status-field write path only") is factually incorrect on the query-count claim
— two new `GRAPH.QUERY` calls were added, which is exactly what §14's per-query documentation
convention exists to track, independent of whether a new node/index was involved.

**Suggested fix:** add a `14.1a`-style subsection (or extend §14.1) with both new Cypher blocks and
a short note on the first-terminal-write-wins guard, matching the format of every other entry in
§14 (code block + `[verified]`/mechanics note).

### MINOR — the `total_jobs == 0` edge case is a silent no-op relying entirely on an external invariant, untested at any level

`_schedule_chunk_processing` (`background.py:130-190`) skips `start_document_progress` entirely
when `total_jobs == 0` (its own docstring: "a no-op ... when nothing is being scheduled at all
(`total_jobs == 0`, e.g. a document with zero chunks)"). Since `report_document_job_done` is then
never called either, such a document stays parked at `'processing'` forever — the exact symptom
K-051 exists to fix.

Verified this is currently unreachable: `repository.create_document`'s only production caller is
`Services.ingest_document` (`services.py:1091`, confirmed via `grep -rn "\.create_document("`),
which raises `EmptyDocumentError` before any empty/whitespace text reaches chunking, and
`chunking.split_into_chunks` (`chunking.py:30-31`) returns `[]` only for empty/whitespace input —
so a non-empty document can never produce zero chunks today. But this safety is an *invariant
enforced two layers away*, not asserted or defended locally, and nothing — not even a repository-
or background-level unit test bypassing the service guard — exercises `total_jobs == 0` to prove
the no-op is intentional rather than an oversight. A future direct caller of
`repository.create_document`/`_schedule_chunk_processing` (a new transport, a test helper, a
scripted backfill) that supplies an empty chunk list reproduces K-051's original defect silently.

**Suggested fix:** either have `_schedule_chunk_processing`/`start_document_progress` flip
`status` straight to `'ready'` when `total_jobs == 0` (closing the gap regardless of the caller),
or at minimum add a test pinning the current no-op behavior so a change to the upstream invariant
is caught here too.

### MINOR — no test proves `_report_document_job` itself swallows a raising `repo.report_document_job_done`

Every other `_safe_*` wrapper in `background.py` has a dedicated "swallows failure, never raises"
test (`test_safe_embed_chunk_swallows_failure_logs_error_never_raises`,
`test_safe_extract_swallows_failure_logs_error_never_raises`,
`test_safe_fuse_swallows_failure_logs_error_never_raises`). `_report_document_job`
(`background.py:38-64`) is a new failure-isolation boundary of exactly that kind — its own
docstring states "it must never raise into the caller's scheduling mechanism either" — but no test
in the 12 new ones exercises `repo.report_document_job_done` itself raising (as opposed to the
underlying `embed_chunk`/`extract_chunk` job raising, which is well covered). The code is correct
by inspection (`except Exception`, log, no re-raise), but the module's own stated convention goes
unverified for this specific path.

**Suggested fix:** add e.g. `test_report_document_job_swallows_a_raising_repo` — a fake repo whose
`report_document_job_done` raises, asserting `_safe_embed_chunk`/`_safe_extract` still complete
without propagating.

## What's solid

- **Completion-counting mechanism is correct and live-verified, not just read.** Ran the new
  repository-level tests directly against the live `falkordb-dev` instance
  (`test_start_document_progress_then_report_all_done_reaches_ready`,
  `test_report_document_job_done_one_failure_marks_failed_even_with_others_pending`,
  `test_report_document_job_done_late_success_does_not_revert_failed`,
  `test_report_document_job_done_late_failure_does_not_revert_ready`) — all pass. The FOREACH-chained
  guard logic really does implement "first terminal write wins": a failure flips to `'failed'`
  regardless of outstanding count, and both directions of late-arrival (success-after-failed,
  failure-after-ready) are exercised and correctly no-op. Each `report_document_job_done` call is
  one atomic `GRAPH.QUERY` round trip (decrement + both guarded flips in one query) — no read-then-
  write race across separate calls exists anywhere in the new code.
- **Failure-isolation discipline preserved.** `_report_document_job` mirrors the module's existing
  `_safe_*` convention (broad `except Exception`, log via `_log.exception`, never re-raise); by
  inspection it cannot propagate into `BackgroundTasks`/thread scheduling.
- **`.repo` property design is safe in production.** Both `EmbeddingWorker.__init__` and
  `IngestionPipeline.__init__` take `repo` as a required positional parameter, and `app.py`'s two
  construction sites (`_build_default_app`, the MCP-mount path) always pass the same `Repository`
  instance to both. Confirmed via `grep -n "EmbeddingWorker(\|IngestionPipeline("
  falkorchat/app.py` — there is no production path where a real worker/pipeline lacks `.repo`; the
  silent-skip is exercised only by pre-K-051 test fakes, exactly as documented.
  `getattr(worker_or_pipeline, "repo", None)` also safely handles fakes that never define `.repo`
  at all (not just ones that set it to `None`), confirmed against `RecordingWorker`/
  `RecordingIngestionPipeline` in `test_api.py`.
- **No regression from the `_safe_embed_chunk` signature change.** `grep -rn "_safe_embed_chunk"`
  across `falkorchat/` and `tests/` confirms the only caller is `background._schedule_chunk_processing`
  (updated in the same diff) plus `test_background.py` (updated); `api.py`/`mcp.py` call
  `_schedule_chunk_processing` only, never `_safe_embed_chunk` directly, as their own docstrings
  claim.
- **Integration tests are deterministic, not flaky-by-construction.** `test_ingest_document_background_completion_reaches_ready`
  and `test_ingest_document_background_embed_failure_reaches_failed` wire the *real*
  `EmbeddingWorker`/`IngestionPipeline` (stub embedder/LLM, no network) against the real `ws:test`
  repo, relying on Starlette's `TestClient` running `BackgroundTasks` synchronously before the
  response returns (an existing, already-relied-upon behavior in this test file, not a new
  assumption). The forced-failure test induces a real `EmbeddingDimensionError` via a genuinely
  mismatched embedder output dimension — a realistic production failure mode, not a contrived
  monkeypatch. Ran both directly: pass.
- Targeted re-run (`pytest -q -k "start_document_progress or report_document_job_done or
  ingest_document_background"`): 6 passed. Full-suite count independently corroborated by the
  coordinator (1766 passed, 4 deselected) is consistent with what this diff should add.

## Open questions

- Should the `QUERIES.md` §14 gap (MAJOR finding above) block closing K-051, or land as a fast
  follow-up documentation pass? Given the item's own "Risks/RAM: none" framing and that the
  mechanism itself is sound, I'd lean toward not re-opening `tdd-engineer`'s unit for it, but the
  coordination doc's "no QUERIES.md impact" premise should be corrected regardless of who does the
  write.

## 2026-08-25 — Pass 2

**Scope.** Re-gated `tdd-engineer`'s follow-up diff against `repository.py`, `background.py`,
`QUERIES.md`, and the corresponding test files, same scope filter as Pass 1 (K-005/`_read_structure`
hunks excluded, confirmed still present but untouched by this pass's changes). Ran the K-051 test
slice directly (21 passed) and the full offline suite myself: **1773 passed, 4 deselected** —
matches the coordinator's independently-reported count exactly.

**Verdict: approve.**

- **MAJOR (`QUERIES.md` §14 currency) — fixed.** New §14.1a documents both `start_document_progress`
  and `report_document_job_done` verbatim, in §14's existing format (code block + prose + `[verified]`
  note); §14.1's `create_document` literal now includes `pendingJobs: 0`, byte-matching
  `repository.py:1029`. Diff-checked line-for-line against the shipped Cypher — no drift left.
- **MINOR (zero-chunk `total_jobs == 0` edge case) — fixed.** `_schedule_chunk_processing` now calls
  `start_document_progress` unconditionally (the `if total_jobs:` guard from Pass 1 is gone), and
  `start_document_progress` itself flips `status` straight to `'ready'` when `total_jobs <= 0`,
  guarded on `status = 'processing'` (same first-terminal-write-wins posture, verified it also
  doesn't resurrect an already-`'failed'` document). No longer relies on the external
  `EmptyDocumentError` invariant at all. Covered at both altitudes: repository-level
  (`test_start_document_progress_zero_total_jobs_flips_straight_to_ready`,
  `..._does_not_revert_an_already_failed_document`) and background-level
  (`test_schedule_chunk_processing_with_zero_chunks_still_initializes_progress_to_zero`) — the
  latter also newly pins the `total_jobs` arithmetic itself
  (`test_schedule_chunk_processing_initializes_progress_with_total_jobs`, 2 chunks × 2 wired workers
  = 4), closing a soft gap Pass 1 noted but didn't formally flag. Ran all of these live against
  `falkordb-dev`: pass.
- **MINOR (untested `_report_document_job` swallow behavior) — fixed.** New `_RaisingProgressRepo`
  fake plus `test_safe_embed_chunk_swallows_a_raising_repo_logs_error_never_raises` and the
  `_safe_extract` sibling, matching the exact pattern of every other `_safe_*` "swallows failure,
  never raises" test in the module (asserts the underlying job still ran, exactly one ERROR record,
  the right message, `exc_info` populated). Ran directly: pass.
