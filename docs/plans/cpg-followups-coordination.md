# CPG backlog follow-ups — coordination log

> **Status:** archived · **Owner:** `teco` · **Tracks:** C-308, C-312, C-314, C-315, C-316, C-318,
> C-319, C-321 (post-M2/M3 follow-ups — not milestone-gating)

Working the concrete, ready `docs/BACKLOG.md` follow-up items for the CPG / `cpg-analysis` /
`cpg/mcp` component. Two items explicitly excluded from this round:

- **C-310** (OpenCode + Kiro MCP wiring) — open-ended feasibility question ("reversal trigger: an
  upstream server that can be filtered down to one read-only tool"), not a scoped fix. Left for a
  dedicated round if/when it becomes a priority.
- **C-323** — backlog itself says *"deliberately deferred — do not schedule."* Left alone.

## Units — Wave 1 (parallel, file-disjoint)

| Unit | Item(s) | Owner | Files | Done-condition |
|---|---|---|---|---|
| U1 | C-308 | `graph-dba` | `skills/cpg-analysis/references/impact-analysis.md` | Bounded transitive upward call-closure query added, live-verified against a loaded CPG graph, no naive-composition 0-row regression |
| U2 | C-312 | `graph-dba` | `skills/joern-cpg/SKILL.md` | Post-load `FILENAME`-prefix verification step documented/added to the pipeline |
| U3 | C-314 + C-315 | `coder` | `cpg/mcp/server.py` (+ tests) | Map cells render without leaking the client's Python type; booleans render as real JSON/Cypher-literal booleans, not `True`/`False`; offline suite green |
| U4 | C-318 + C-321 (core) | `tdd-engineer` | `cpg/mcp/tests/test_server.py` | New pin: `mcp.instructions` non-empty and ≤2000 chars; scratch-graph name in the live suite derived from `uuid4().hex[:8]` instead of `os.getpid()` |
| U5 | C-321 (deferred sub-items) | `devops` | `cpg/mcp/docker-run.sh`, `cpg/mcp/build.sh`, `cpg/mcp/image-tag.sh` | Autobuild path sets `CPG_MCP_NO_PULL=1` on the `--runtime-only` build + one-line network-fallback message; `image-tag.sh` walk excludes `.pytest_cache`; file-mode/symlink/missing-dir/failed-`find` edge cases from review §17–§18 addressed or explicitly accepted with a comment |
| U6 | C-319 | `cobb` | `skills/agent-standards/claude-code.md` §MCP | `enabledMcpjsonServers` approval-scoping (session-cwd-keyed vs. git-root-keyed discovery) documented |
| U7 | C-316 close-out | `qa-engineer` | `docs/BACKLOG.md` only | See note below — **no edit to the archived plan** |

### Note on U7 / C-316

`docs/plans/cpg-query-access.md` is `Status: archived`. Per the doc-reference convention, an
archived document takes no substantive edits — only a header-pointer edit. C-316's original text
("correct the probe in the plan") predates that convention. The corrected probe is **already**
recorded verbatim in the C-316 backlog entry itself, so the practical fix is: leave the archived
plan untouched, and have `qa-engineer` confirm the corrected probe is accurate (re-run it against
a live CPG) and close C-316 in `docs/BACKLOG.md` with a note that the archived plan is
intentionally left as-is (freeze rule), pointing future readers at the backlog entry as authority.

**Shared-file guard:** U1–U6 do **not** touch `docs/BACKLOG.md` or `docs/HISTORY.md` — both are
shared files every unit would otherwise want to edit in parallel. Each unit reports its suggested
backlog-closing note in its own final report; `teco` applies one consolidated update in Wave 3
after both review gates pass. U7 is the one exception — its whole job *is* the C-316 backlog entry.

## Wave 2 — review gates

- **`analyst`** reviews U3 + U4 + U5 (code/script changes: `server.py`, `test_server.py`,
  `docker-run.sh`/`build.sh`/`image-tag.sh`) in one consolidated pass once Wave 1 lands.
- **`cobb`** reviews U1 + U2 + U6 (skill-content changes) in one consolidated pass — skill
  authoring/standards is its own domain, mirrors the C-303/C-307 precedent.
- U7 is a documentation-only closure with no code/design stakes — reviewed by nobody per the
  "trivial, low-risk" exception; flagged explicitly here rather than silently skipped.

## Wave 3 — closeout

One consolidated pass (owner TBD at the time, whichever Wave-1/2 agent is best positioned) applies
the `docs/BACKLOG.md` ✅ flips (C-308, C-312, C-314, C-315, C-318, C-319, C-321 core + sub-items)
and matching `docs/HISTORY.md` entries, using each unit's reported closing note. This is the only
point at which those two shared files are touched by the delegated units.

## Log

- 2026-08-09 — coordination doc opened, Wave 1 dispatched (U1–U7, parallel).
- 2026-08-09 — Wave 1 complete, all 7 units delivered. `falkordb-dev` was found down mid-round
  (no container at all) — `teco` started it; the persisted data volume still had `cpg_falkorchat`
  loaded, so U1 (C-308) and U2 (C-312) resumed and completed real live verification instead of
  shipping with a documented gap, and U7 (C-316) likewise resumed and closed with real evidence.
  U2's first pass had misdiagnosed the FalkorDB-down state as a structural per-subagent sandbox
  connectivity gap; it self-corrected and logged a same-day retraction in its own kaizen inbox once
  the resume proved reachability was fine. U1's own live verification caught and fixed a real bug
  in its first-draft query (a self-recursion filter that silently dropped a legitimate same-named
  caller). Offline `cpg/mcp` suite independently re-run by `teco`: 66 passed / 7 deselected, green.
  Wave 2 dispatched: `analyst` reviewing U3+U4+U5 (code/scripts) at
  `docs/reviews/cpg-followups-impl.md`; `cobb` reviewing U1+U2+U6 (skill/doc content), same path or
  a split sibling if there's a write collision.
- 2026-08-09 — Wave 2 complete. `analyst`'s Pass 1 found a real Major on U3 (C-314/C-315):
  `render_cell()`'s dict/bool fix only handled the top-level cell value, not nested structures —
  `IS_EXTERNAL` nested inside a returned map (C-315's own motivating example) still leaked. Routed
  back to the U3 implementer, who added a recursive `_normalize_for_repr()` helper; `analyst`'s Pass
  2 independently re-verified (including edge cases beyond the new test: apostrophe-in-key, 10-level
  nesting, tuples, nested `set`) and returned **approve**. `cobb`'s review hit a real path collision
  with `analyst`'s (both wrote `docs/reviews/cpg-followups-impl.md` concurrently) — recovered at
  `docs/reviews/cpg-followups-skills-impl.md` with cross-pointers in both files. `cobb` returned
  **approve** for U1/U2 and a self-caught Major on its own U6 text (an unsupported causal claim
  about `$CLAUDE_PROJECT_DIR`); applied its own verified-correct rewrite directly per the trivial/
  low-risk exception rather than looping a second review.
- 2026-08-09 — Wave 3 closeout: `docs/BACKLOG.md` items C-308, C-312, C-314, C-315, C-318, C-319,
  C-321 flipped to ✅ (C-316 was already closed directly by its own unit against the archived-plan
  freeze rule); one `docs/HISTORY.md` entry added for the round (one factual slip in a fabricated
  test-count breakdown caught by `teco` and corrected); a stale present-tense bug description in
  `cpg/mcp/README.md` updated to reflect the C-321 fix. `teco` independently re-ran the offline
  `cpg/mcp` suite as the final check: 67 passed, 7 deselected, green. All deliverables committed by
  `teco` as the integrator. Round closed — C-310 and C-323 remain intentionally out of scope (see
  the top of this document).
