# Review — Workflow def/snapshot re-publish semantics implementation (K-034)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-034

## Scope & verdict

Post-implementation code review of the uncommitted diff implementing
`docs/plans/workflow-republish-semantics.md` (Status: active, "approve with suggestions" at the
pre-implementation gate — `docs/reviews/workflow-republish-semantics.md`, Minor 1 / Minor 2). No
identity is attached to the diff (the session that produced it crashed before commit); this review
treats the working tree as the deliverable and the `## 2026-08-01 — K-034` entry at the top of
`docs/HISTORY.md` as the implementer's own narrative, verified rather than trusted.

Reviewed: `server/falkorchat/{repository,services,app,api,executor}.py`, all four touched test
files, and the doc-sync set (`docs/{BACKLOG,DESIGN,QUERIES}.md`,
`docs/requirements/{agent-import,workflow-dependence-overlay}.md`, `falkor-chat/AGENTS.md`,
`docs/HISTORY.md`'s own new entry). Every load-bearing claim below was independently checked
against the live file — line-by-line diff reading, not the HISTORY narrative's word.

**Verification actually run (not just read):**
- `pytest -q` in `server/` (FalkorDB up via the already-running `falkordb-dev` container) →
  **691 passed, 1 deselected** — matches the HISTORY entry's claimed count exactly, and matches the
  prior baseline (**658** passed, confirmed from the previous `docs/HISTORY.md` entry) plus the
  claimed **+33** new tests (12 `_structural_diffs` + 7 publish-gate + 7 materialize-gate + 2
  executor tie-break + 1 repository contract-boundary + 3 API E2E + 1 app-wiring — sums to 33
  exactly, counted from the actual test function names, not taken on faith).
- `./scripts/test_queries.sh` → **282/282 passed** — matches, and independently confirms
  `_PUBLISH_CYPHER` is byte-identical (query-suite-invariant claim).
- `ruff check` on the five touched source files → clean. On the four touched test files → **one**
  `E501` (line too long) in `test_executor.py` — see Nit below.
- Traced `_diff_structures`' actual path grammar (`services.py`) against `_structural_diffs`' filter
  predicate and confirmed the topology/property boundary is drawn correctly (transition identity
  `(from, to, on, order)` in the diff comparator matches `_PUBLISH_CYPHER`'s own `MERGE` key exactly
  — read the live Cypher, not the plan's quote of it).
- Confirmed the `_select_transition` diff touches only that method (single hunk,
  `executor.py:786+`) and never `_drive_loop`'s body, so the HISTORY entry's "SHA-lock unchanged"
  claim is trivially true and the quoted hash (`71055f756280`) matches `docs/DESIGN.md:410`
  verbatim.
- Grepped `-i "immutab"` across `docs/`, `server/falkorchat/`, `AGENTS.md` (excluding archive,
  reviews, and the BACKLOG's own before/after correction table) and confirmed every live hit
  reflects the corrected two-part framing — no site was missed, no site was over-corrected.
- Read the `test_diff_reports_divergence_after_the_documented_reseed_trap` test in full and
  confirmed it is unmodified and still asserts `201` — the review's Minor 1 canary genuinely stays
  green, not merely claimed to.

**Verdict: approve with suggestions.** No blockers. The implementation is a faithful, well-tested
realization of the plan — every wiring snippet in plan §3.3 matches the shipped code close to
verbatim, both plan-review Minor findings are demonstrably addressed, and the topology/property
boundary is correctly drawn against the real `_PUBLISH_CYPHER` MERGE-key semantics. One Minor
(backlog hygiene) and two Nits below.

---

## Findings

### Minor — `docs/BACKLOG.md`'s K-034 entry header is not flipped to delivered

`docs/BACKLOG.md:786` still reads `### K-034 — … (🔵 proposed — discovered by `architect` …)` after
this diff, even though the item is now fully implemented, tested, and doc-synced end to end. Compare
the sibling precedent this same diff leaves untouched: K-031's own header
(`docs/BACKLOG.md:604`) reads `(✅ **delivered 2026-07-24** — plan v2 + analyst re-gate →
HISTORY.md)`. The diff *does* correct K-034's downstream premise paragraphs (the K-029 relocation
note, the K-032 "why it exists" bullet) and the doc-sweep table's individual rows, but never touches
K-034's own status badge or its `- **Owner:**`/`- **Scope:**` bullets to reflect that the decision
was made and shipped. A reader scanning the backlog for open items would still see K-034 flagged
`🔵 proposed`, with no pointer to this delivered change or to `docs/HISTORY.md`.

This is not a code defect — nothing here affects behavior — but it is a real doc-sync gap inside a
change whose own stated deliverable is "correct every doc/docstring site" (plan §1, in-scope item
4), and the M3 milestone summary (`docs/BACKLOG.md:60`) that lists K-031's delivery does not
mention K-034 either.

**Suggested improvement:** flip K-034's header to `✅ **delivered 2026-08-01** — → HISTORY.md`
(mirroring K-031's exact phrasing), same commit as the rest of the doc-sync sweep.

---

## Nits

- **`test_executor.py:224` — one `ruff` E501** (101 > 100-char limit from `pyproject.toml:38`):
  `@pytest.mark.parametrize("transitions", [TIEBREAK_TRANSITIONS, list(reversed(TIEBREAK_TRANSITIONS))])`.
  Cosmetic — `falkor-chat/docs/DESIGN.md` §14.7 documents that ruff is not a wired gate, so this
  does not block anything, but it is a one-line fix (wrap the parametrize list) if the file is
  touched again.
- **`WorkflowDefConflictError`'s message always points at `GET /workflow-defs/{key}/versions/{version}`**
  regardless of `resource` (`services.py::_check_no_structural_conflict`) — for a `materialize_def`
  conflict (the "workspace snapshot" resource), that route shows the *reference* def's structure,
  not the drifted workspace snapshot itself; the more directly actionable pointer for that case is
  the existing `/workspaces/{ws}/snapshots/{key}/versions/{version}/diff` route the plan's own §7
  risk 1 names. This is **inherited verbatim from the plan's own §3.2 code snippet**, not an
  implementation deviation — flagging only as a possible cheap follow-up polish, not something to
  hold this change for.

---

## What's solid

- **Topology/property boundary is correct, verified against the real Cypher, not the plan's quote of
  it.** Read `_PUBLISH_CYPHER` directly (`repository.py`): the `TRANSITION` `MERGE` key is
  `{on, order}` plus the `(from)->(to)` pattern endpoints — exactly the 4-tuple
  `_diff_structures`' `_identity()` helper already used, and exactly what `_structural_diffs`
  classifies as structural (bare `transitions[...]` presence rows) versus property
  (`.guard`-suffixed rows). Traced the filter's string-suffix matching against every path shape
  `_diff_structures` can emit (`meta.*`, `steps[<key>]`/`steps[<key>].<type|config>`,
  `transitions[...]`/`transitions[...].guard`) and found no case where the filter mis-classifies —
  the 12 new `_structural_diffs` unit tests (11 parametrized one-class-at-a-time + 1 bundled-diff
  case) cover exactly this grammar.
- **Both plan-review Minor findings genuinely addressed, not just asserted.** Minor 1
  (`test_diff_reports_divergence_after_the_documented_reseed_trap` staying green as a canary) —
  confirmed by reading the test: it is byte-for-byte unmodified and still asserts `201`, and the
  HISTORY entry correctly attributes *why* (`_wipe_reference` makes `existing_raw is None`, so the
  gate never fires). Minor 2 (broadening the TOCTOU framing beyond publish-only) — confirmed in both
  `WorkflowDefConflictError`'s docstring (`repository.py`) and `materialize_def`'s docstring
  (`services.py`), both stating the identical residual-race shape applies to the workspace-snapshot
  side, matching the review's exact suggested wording.
- **`WorkflowDefConflictError` → 409 mapping is a correct, faithful mirror of the
  `WorkflowRunNotWaitingError` precedent** (`app.py`) — same handler shape, same envelope
  (`{"error": type(exc).__name__, "detail": str(exc)}`), same "state conflict, nothing written"
  409 semantics documented in `WorkflowRunNotWaitingError`'s own docstring.
- **The `_select_transition` tie-break is genuinely additive-only and genuinely outside the SHA
  lock.** The diff is a single hunk entirely inside `_select_transition`; `_drive_loop`'s own lines
  are untouched, so the "hash unchanged" claim needs no separate AST-hash tool to verify — the diff
  itself proves it. `docs/DESIGN.md:412` independently confirms `_select_transition` is documented
  as outside the lock.
- **Test coverage is proportional and symmetric.** Both `publish_workflow_def` and
  `materialize_def` get the identical 7-case matrix (no-existing / identical / property-only /
  new-step / retargeted-transition / moved-start / removed-transition), at the service (FakeRepo),
  repository (live FalkorDB, contract-boundary pin), and API (live FalkorDB, E2E structure-unchanged
  assertion) layers — the API-layer tests in particular assert the *stored* structure is byte-equal
  before/after the rejected write, which is the strongest evidence available that the gate actually
  runs before the real `_PUBLISH_CYPHER`, not just that it raises.
- **The residual concurrent-first-publish race is accurately still described as residual**, not
  newly claimed as closed, in both the plan-mirroring docstring and the HISTORY entry — matches the
  plan's §7 risk 2 exactly, correctly broadened to both call sites per Minor 2 without overclaiming
  a fix.
- **Doc-sync sweep is accurate and complete** against a systematic re-grep, not spot-checked in
  isolation — every live (non-archived, non-HISTORY, non-review) hit for "immutab"/"no-op" reflects
  the corrected framing, including sites the plan only asked to "consider" touching (e.g. the
  `agent-import.md` FR-2b re-check, which the implementer resolved inline with visible reasoning
  rather than silently dropping).
- **`Repository.publish_def`/`materialize_snapshot` and `_PUBLISH_CYPHER` are genuinely untouched**
  — confirmed by diff (docstring/comment-only changes in `repository.py`'s write methods), which
  independently supports the "no `graph-dba` gate needed" claim and explains why
  `test_queries.sh`'s count is unaffected.

## Open questions

None that block landing. The Minor (BACKLOG status flip) is cheap enough to fold into the same
commit as the rest of this diff; it does not need a loop back to `architect` or `tdd-engineer`.
