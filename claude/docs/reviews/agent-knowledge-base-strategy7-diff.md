# Agent knowledge-base strategy — item 2 hybrid fusion, U4 diff-level review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

## Scope & verdict

Diff-scoped gate on U4 of `claude/docs/plans/agent-knowledge-base-strategy7-coordination.md`:
`coder`'s commit `af535ecb` (`git show af535ecb`, 9 files, +779/-65), implementing
`claude/docs/plans/agent-knowledge-base-strategy7-impl.md` (already reviewed and
approved-with-suggestions across two passes, `claude/docs/reviews/agent-knowledge-base-strategy7-impl.md`).
Read the plan in full and diffed every changed production file
(`services.py`, `repository.py`, `mcp.py`, `bootstrap_schema.sh`, `DESIGN.md`, `HISTORY.md`,
`QUERIES.md`, `SKILL.md`) against it line by line, then read every new/changed test body in
`test_services.py` directly (not just names/docstrings). Ran the file's test suite
(`.venv/bin/python -m pytest tests/test_services.py`, 288 passed, no live DB needed for this file)
and `ruff check` on the five touched Python files (clean). Did not re-verify the live index or
re-run the full monorepo suite — both already independently confirmed by `teco` per this unit's
brief, not re-derived here. Deliberately did not re-check U1/U2/U3's own settled decisions (RRF
formula, `k=60`, the 0.43/rank-≤2 gate values, the strip-marker design) — this gate is about
whether the diff matches the plan and what only a running test can catch, not a second plan
review.

**Verdict: approve with suggestions.** The diff matches the plan faithfully — no unflagged
deviation found anywhere across `services.py`/`repository.py`/`mcp.py`/`bootstrap_schema.sh`, and
every documentation edit (`DESIGN.md`, `QUERIES.md`, `HISTORY.md`, `SKILL.md`, `mcp.py`'s
docstring) accurately describes the code as it actually landed, verified word-for-word against
the diff, not just skimmed. I confirmed two mutation escapes the plan's/coder's own mutation
tables don't cover (below) — both real coverage gaps on a load-bearing, plan-flagged subtlety
(§3.6's `score`-semantics distinction), neither a live defect in the shipped code, which I
verified is correct on both points by reading `_fuse_chunk_hits_rrf` directly.

**CPG: considered, not relevant.** U3's plan review already ran and independently re-verified the
one call-graph question that matters (`cpg_falkorchat`, `search_chunks`/`search_documents` callers)
and the coordination doc's own freshness note confirms the one commit since then touches neither
`repository.py`/`services.py`/`mcp.py`. This diff adds no new caller of either method, so there is
no new call-graph question for a CPG to answer at this gate.

## Findings

### Major — the `score`-semantics distinction §3.6 calls out as its own key finding has zero direct test coverage, and two real mutations slip through undetected

The plan (§3.6, repeated in `_fuse_chunk_hits_rrf`'s docstring and `mcp.py`'s tool docstring)
states, as a deliberate, load-bearing API contract: `score is None` iff the chunk was absent from
`vector_hits` — **not** a synonym for "admitted via the lexical gate." A chunk present in
`vector_hits` with a floor-failing score (e.g. `0.6`), admitted only because it also passes the
lexical rank gate, must still report that real `0.6`, not `None`.

I confirmed by direct mutation (`falkor-chat/server/falkorchat/services.py`,
`_fuse_chunk_hits_rrf`'s output-row construction) that no test in `test_services.py` pins this:

1. Changed `row["score"] = vector_score.get(chunk_id)` to only populate `score` when the vector
   half of the gate itself passes (`vs if vs <= vector_floor else None`) — i.e. exactly the bug
   §3.6 warns against (silently collapsing "admitted via lexical" into `score is None`). Ran
   `pytest tests/test_services.py -k "fuse_chunk_hits_rrf or search_documents"` — **18/18 still
   pass.** Reverted; suite green again (verified `git diff` is empty after revert).
2. Changed the vector-floor comparison from `vs <= vector_floor` to `vs < vector_floor` (excludes
   the exact boundary `score == 0.43` from admitting). Same 18/18 still pass. Reverted, confirmed
   clean.

Neither mutation is in the plan's/coder's own named mutation table (`RRF_K`, sort direction,
never-rejecting gate, `partition`/`rpartition`, `tail or query`) — both are exactly the kind of
"boundary condition in the admissibility gate" this gate's brief asked me to look for. The
counterpart boundary (`lexicalRank == LEXICAL_ADMISSIBILITY_RANK` exactly, i.e. rank 2) **is**
caught, incidentally, by `test_fuse_chunk_hits_rrf_tie_break_is_deterministic_by_chunk_id` (its
`Y` candidate sits at exactly lexical rank 2) — confirmed by mutating `lr <= lexical_rank_gate` to
`lr < lexical_rank_gate`, which does fail that test. The vector side has no equivalent incidental
cover.

**Suggested fix — two new tests in `falkor-chat/server/tests/test_services.py`'s §5.2 block**
(after `test_fuse_chunk_hits_rrf_rejected_when_both_signals_fail_their_half_of_the_gate`, same
style):
- A case where a chunk is present in `vector_hits` at exactly `score ==
  VECTOR_ADMISSIBILITY_FLOOR` with no lexical presence → assert it is admitted and its output
  `score` equals the floor value exactly (pins the `<=`, not `<`).
- A case where a chunk is present in **both** `vector_hits` (floor-failing score, e.g.
  `VECTOR_ADMISSIBILITY_FLOOR + 0.17`) and `lexical_hits` (rank 1, so the OR's lexical half
  admits it) → assert the output row's `score` equals that real floor-failing value, not `None`.
  This is the one shape the coverage gap above hides, and is the exact scenario §3.6's prose
  names by example.

This is a coverage gap, not a shipped defect — I read the actual admissibility/score-assignment
code (`services.py`'s `admissible()` and the output-row construction inside `_fuse_chunk_hits_rrf`)
and it is correct on both points today (`<=` for the floor; `vector_score.get(chunk_id)` applied
unconditionally, independent of which gate half admitted). The risk is a future edit silently
regressing exactly the subtlety the plan itself flagged as easy to get wrong, with nothing to
catch it.

## What's solid

- `services.py`/`repository.py`/`mcp.py`/`bootstrap_schema.sh` match the plan's Step 1-6 code
  samples essentially verbatim — constants, docstrings, the `depth = max(HYBRID_OVERFETCH_K,
  limit)` fix, the `_strip_query_instruction_prefix` first-occurrence design, the
  `try/except ResponseError` mirroring `search_messages` — no unflagged deviation found anywhere.
- The two plan-review-driven fixes (the genuine two-marker fixture; the depth-scaling formula) are
  both actually in the diff and both have dedicated, correctly-targeted tests
  (`test_strip_query_instruction_prefix_keeps_first_occurrence_only`;
  `test_search_documents_overfetch_depth_scales_up_for_a_large_limit` +
  `test_search_documents_large_limit_is_not_capped_by_the_hybrid_overfetch_floor`).
- Every documentation edit is accurate against the landed code, checked directly: `DESIGN.md`'s
  `Document.title` "title-fuzzy" cross-reference is a real, pre-existing term from
  `document-ingestion2-ml.md` (not an invented gloss); `QUERIES.md` §14.3a's Cypher/formula/table
  match the code exactly; `SKILL.md`'s step-4 revision correctly distinguishes "no longer required"
  from "the value's derivation history stays intact"; `mcp.py`'s docstring correctly states the new
  `rrfScore`/`vectorRank`/`lexicalRank` fields and the narrowed `score` semantics.
- Reported test-count claims verified exactly: `git show af535ecb -- .../test_services.py | grep
  -c '^+def test_'` → 19, `'^-def test_'` → 1, matching the ledger's "19 new, 1 renamed-away."
  `api.py` confirmed untouched, as the plan specified (no change needed).
- `ruff check` clean on all five touched Python files; `test_services.py`'s 288 tests all pass with
  no live dependency.

## Open questions

None — the one substantive gap found (Major, above) has a concrete, self-contained fix that does
not require a design decision from `architect`/`data-scientist`; it's a test-coverage addition
`coder` (or whoever picks up this finding) can make directly against the existing, correct code.

---

Ledger note for `teco`: recommend setting U4's row `Gate → verdict` cell to
`` `analyst` (`ad05017f8d25ad27e`) → **approve with suggestions**
(`claude/docs/reviews/agent-knowledge-base-strategy7-diff.md`). No blocker; one Major (two
`_fuse_chunk_hits_rrf` `score`-semantics boundary cases — exact vector-floor equality, and
present-in-both-signals-admitted-via-lexical — have no direct test and two confirmed mutation
escapes; two named test additions suggested, not a design change). `` — I have not edited the
coordination doc myself (outside this review document's own remit).
