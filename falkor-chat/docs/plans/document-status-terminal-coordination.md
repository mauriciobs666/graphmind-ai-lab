# `Document.status` reaches a terminal state — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-051 (follow-up, not milestone-gated)

Coordinating the fix for K-051 — `Document.status` never leaves `'processing'` — filed as a
Medium-severity, non-blocking defect out of `qa-engineer`'s K-050/M5 acceptance pass
(`docs/test-reports/document-ingestion-report.md`, Defect 1).

## Context

- Backlog item: `docs/BACKLOG.md` K-051. Owner named there: `tdd-engineer` (bug fix, clear
  behavior contract).
- Root cause (already diagnosed by the acceptance pass — no RCA needed): no code path in
  `server/falkorchat/*.py` ever writes `Document.status = 'ready'`/`'failed'`.
  `repository.create_document` (`repository.py:1029`) hardcodes `status: 'processing'` at
  creation and nothing downstream ever flips it. The plan's own Stage 1/Stage 3 text and §5's
  AC-8 test-strategy row name `Document.status` as the caller's completion signal, but no stage
  3-6a diff ever built the write.
- Relevant background-pipeline seam: `server/falkorchat/background.py` — `_safe_embed_chunk`
  (Stage 2) and `_safe_extract` (Stage 3) each run out-of-band, per chunk, independently
  scheduled side by side by `_schedule_chunk_processing` (Stage 6a) from both transports
  (`api.py`, `mcp.py`). Neither currently reports completion back up to the `Document`. Both
  already swallow+log exceptions (never raise) — same failure-isolation discipline the `'failed'`
  path must not break.
- Suggested fix shape (from the report, not binding — `tdd-engineer`'s call to make): a
  per-document completion counter or check-remaining-work query, flipped to `'ready'` once every
  chunk's embed **and** extract have both completed (success or isolated failure), `'failed'`
  when a background step's isolation catches a real failure rather than silently logging it only.
- Test strategy (from the backlog item, binding as the acceptance bar, mechanism still
  `tdd-engineer`'s call): an integration test that ingests a document, drives its background
  processing to completion (mirroring how the Stage 6a AC-8 batch test does this), and polls
  `get_document` until `status` leaves `'processing'`; a separate test forcing a background
  failure and asserting `status: 'failed'`.
- CPG freshness (checked 2026-08-25, `skills/cpg-analysis/references/freshness.md`):
  `cpg_falkorchat`, built `2026-08-17T00:40:42Z` (scratch-copy build, `sourceCommit` null,
  `sourcePath=/tmp/cpg-src/falkor-chat-server` → real counterpart `falkor-chat/server`) —
  **stale**: 14 commits touched `falkor-chat/server` since build (includes all of K-050 Stage
  6a). Flagged to the dispatched specialist rather than trusted silently; this fix is small
  enough that neither unit is expected to need it, but the flag rides the brief regardless.
- Documentation impact: `docs/HISTORY.md` gets a dated entry on close; K-051's row is removed
  from `docs/BACKLOG.md` (delivered items aren't kept there, not even as an index row) in the
  same change. **Correction (caught by `analyst`'s Pass 1 review, U2):** this Context section
  originally claimed no `QUERIES.md` impact on the strength of the backlog item's Risks/RAM line
  ("no new node/index" — true, but that line never spoke to the query library itself). Wrong —
  `start_document_progress`/`report_document_job_done` are two new queries needing a §14 entry,
  and `create_document`'s §14.1 literal needed the `pendingJobs: 0` addition. Both fixed in U3.
  No `DESIGN.md` impact (no new node/index).

**Known concurrent-tree wrinkle:** a separate in-flight session is running
`workflow-diff-absent-key` (K-005) concurrently against this same working tree
(`falkor-chat/docs/plans/workflow-diff-absent-key-coordination.md`), touching
`falkor-chat/server/falkorchat/repository.py`'s `_read_structure` (~line 1717) and its two
callers — a different region of the same file than this unit's `create_document`/
`start_document_progress`/`report_document_job_done` edit site (~1026-1120). No line-range
collision, but `git diff`/`git add <path>` on `repository.py` currently shows **both**
sessions' uncommitted changes mixed together. Independently re-verified solo (`.venv/bin/python
-m pytest -q` from `falkor-chat/server`, no concurrent session running): **1766 passed, 4
deselected** — matches `tdd-engineer`'s reported count.

**Resolved at step 5 (integration):** `repository.py`'s and `test_repository.py`'s diffs turned
out hunk-disjoint from K-005's (`repository.py`: this unit's hunks at lines 1026 and 1066,
K-005's at 1809; `test_repository.py`: this unit's hunk at line 805, K-005's at 14/1699/1782 —
confirmed via `git diff | grep '^@@'` on each file before touching anything). Extracted this
unit's hunks into standalone patch files and staged with `git apply --cached`, leaving K-005's
hunks untouched and still unstaged in the working tree for that session to commit on its own
schedule. No `git add -p`, `stash`, or working-tree mutation needed.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `tdd-engineer` | `a257db0be7fb00bdf` | gated | fix in `repository.py`/`background.py`/`embedding.py`/`ingestion.py` + 12 new tests | `analyst` (U2) → **approve with suggestions** | 212k tok / 111 tools |
| U2 | `analyst` | `a425d5a959bcc0668` | accepted | `docs/reviews/document-status-terminal.md` | — | 120k tok / 54 tools |
| U3 | `tdd-engineer` | `a257db0be7fb00bdf` (resumed) | accepted | `QUERIES.md` §14/§14.1a sync + zero-chunk fix + swallow-tests, 7 new tests | `analyst` (U4) → **approve** | 267k tok / 51 tools |
| U4 | `analyst` | `a425d5a959bcc0668` (resumed) | accepted | `docs/reviews/document-status-terminal.md` Pass 2 | re-gate → **approve** | 147k tok / 10 tools |

## Close-out

Integration commit lands `repository.py`/`test_repository.py`'s K-051 hunks only (see resolved
wrinkle above), `background.py`/`embedding.py`/`ingestion.py`/`test_background.py`/`test_api.py`
in full, `QUERIES.md` §14.1/§14.1a, this coordination doc, and the review doc. `docs/HISTORY.md`
gets a dated K-051 entry; K-051's row is removed from `docs/BACKLOG.md` in the same change.
Offline suite independently re-verified solo, twice: 1766/4 after U1, 1773/4 after U3.

**Commit-mechanics defect, caught after the fact.** The hunk isolation itself was correct — I
staged only this unit's `repository.py`/`test_repository.py` hunks via `git apply --cached`,
verified with `git diff --cached | grep '^@@'` before committing. But the actual `git commit -m
"..." -- <paths>` call listed `repository.py`/`test_repository.py` among its explicit pathspecs,
and `git commit <pathspec>` re-stages the **current working-tree** content for every matching
path before committing — silently overriding the careful partial-hunk staging. The resulting
commit (`d41da78`) therefore also carries the concurrent K-005 session's `_read_structure` fix +
4 tests (confirmed via `git show HEAD -- repository.py | grep _read_structure`, unexpectedly
present). Impact assessed as **benign, not corrective-action-worthy on the code itself**: that
K-005 content was already independently `analyst`-reviewed and accepted per
`workflow-diff-absent-key-coordination.md`'s own ledger, and that coordination's own notes had
already named "K-051 lands first, re-diff and commit the remaining hunk" as one of its two
anticipated resolution paths — this is that path, just landed inside K-051's commit rather than
a clean follow-up commit of its own. Not corrected via `reset`/rebase (destructive, against
standing guardrails, and the content is legitimate); flagged transparently here, in the other
coordination's own Notes section, and to the user. Kaizen entry filed
(`git commit -- <pathspec>` re-stages from the working tree, not the index — a durable gotcha
for any future partial-hunk integration).
