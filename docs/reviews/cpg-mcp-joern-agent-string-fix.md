# cypher-mcp joern-agent stale-string fix — Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** —

## Scope & verdict

Static review of an already-applied, uncommitted fix to `cypher-mcp/server.py` (two user-facing
error-message strings) and `cypher-mcp/tests/test_server.py` (two matching assertions), closing the
stale-agent-name finding originally logged in `claude/analyst/kaizen/inbox.md:133` and echoed in
`docs/reviews/cpg-getting-started.md` (M1). Reviewed via `git diff`, direct reads of both files,
a run of the non-live suite, and a grep sweep for remaining live occurrences of the stale string.

**Verdict: approve with suggestions.** No blockers. The fix is correct, the two assertions it
touches now match the new source strings verbatim, the non-live suite is green (52 passed, 8
`live` tests deselected), and the sweep confirms no other live `.py` source carries the stale
`"joern agent"` string. Two nits below are worth a follow-up but don't block landing this.

## Findings

### Minor

**m1 — Phrasing drifts from the codebase's established `graph-dba` convention.**
Every other live reference to this agent's CPG-build responsibility phrases it as
`` `graph-dba`'s job `` / "Ask `graph-dba` to build one" / "graph-dba won't do it silently" —
see `docs/manuals/cpg-getting-started.md:78,174`, `skills/cpg-analysis/SKILL.md:61`
(`` **`graph-dba`'s** job ``), and `claude/README.md`. None of these append the word "agent" to
the name — because "graph-dba" alone is unambiguous (unlike bare "joern", which needed
disambiguating from the Joern *tool*, hence the original "the joern agent's job"). The new
`server.py:327` string, "...is the graph-dba agent's job...", is not wrong or ambiguous, but it's
the one place in the codebase that spells it this way instead of the established
"`graph-dba`'s job" form. Purely cosmetic — optional tightening, not a defect: e.g.
`"...is graph-dba's job (joern-cpg pipeline)..."` would match house style more closely.

**m2 — Test function name still says "routes_to_joern".**
`test_error_missing_graph_lists_loaded_graphs_and_routes_to_joern` (test_server.py:366) keeps its
old name even though its assertion no longer mentions joern. Harmless (pytest doesn't care), but
it's now a stale label a future reader has to mentally correct. Consider renaming to something
like `test_error_missing_graph_lists_loaded_graphs_and_routes_to_graph_dba` in the same pass, or
as a quick follow-up.

**m3 — The second changed string (`explain_error`'s RO_QUERY branch) has no dedicated assertion
at all**, before or after this fix. `test_error_write_attempt_names_the_read_only_mode` (line
386-393) and its live counterpart `test_live_write_is_rejected_server_side` (line 530-538) only
assert `out.startswith("This tool is read-only (GRAPH.RO_QUERY).")` — neither checks the
"...the joern-cpg pipeline (graph-dba), or redis-cli for ad-hoc writes." tail, so the wording
change to that string is exercised by the suite but not verified by any assertion. This is a
pre-existing coverage gap (the old "the joern pipeline..." text wasn't asserted either), not a
regression introduced by this diff — flagging it only because a reviewer might otherwise assume
the two "matching assertion updates" described in the task cover both changed strings; they
cover only the `graph_not_found_message` one.

## What checks out

- **Correctness of the new wording.** `graph-dba` is confirmed the current, sole owner of CPG
  build/load via the `joern-cpg` skill (`claude/graph-dba/graph-dba.md` frontmatter: "Also drives
  Joern, on demand..."; `skills/joern-cpg/SKILL.md`: "Primarily driven by `graph-dba`"). No
  `joern` agent exists in `claude/README.md`'s roster. Both edited strings now name the right
  agent.
- **Completeness.** `grep -rn "joern agent" --include="*.py" .` returns only the two intentional
  occurrences left in `cypher-mcp/tests/test_server.py` line 366 (the test *function name*, not a
  string literal under test — see m2) — zero remaining stale string literals in live `.py`
  source. Two other files still contain the exact phrase and are correctly out of scope, as the
  task anticipated:
  - `docs/plans/cpg-query-access.md:627` — `Status: archived` in its header; only a header-pointer
    edit is permitted on an archived doc per `AGENTS.md`'s lifecycle rules. Confirmed archived
    via direct read of its header.
  - `docs/archive/test-reports/cpg-query-access-report.md` — lives under `docs/archive/`, which
    is read-only history; nothing is ever un-archived into it or edited in place.
  - Also noted but *not* named in the task's scope list: `docs/reviews/cpg-getting-started.md`
    (Status: `active`, not archived) quotes the stale string twice (lines 42, 46) as **live-verified
    evidence** of the defect this fix closes. Confirmed this is correctly left alone too — it's an
    audit-trail quote of what the tool said at review time, not a live claim about current
    behavior; rewriting a review's evidence to match the post-fix state would be revisionist. No
    action needed here, but if the team wants closure recorded, that's a `tico`/owner call for a
    dated addendum, not something this diff should touch.
- **Test correctness.** Both updated assertions (`test_server.py:376`, `:519`) use
  `assert "graph-dba agent" in out`, a substring match against `graph-dba agent's job` in the new
  source string — verified character-for-character against `server.py:327`. No typo risk (e.g.
  "graph-dba's agent" would not match, and isn't what's in either file). Confirmed the second
  RO_QUERY-branch test (`test_error_write_attempt_names_the_read_only_mode`, line 386-393) needed
  no update — its assertion only checks the `startswith` prefix, which is unchanged; the changed
  tail text isn't asserted at all (see m3).
- **Regression risk.** Ran the full non-live suite (`.venv/bin/python -m pytest tests/test_server.py
  -k "not live"`): **52 passed, 8 deselected**, zero failures. No other test, doc, or source file
  outside the two archived/out-of-scope docs above quotes either exact string, so nothing else
  depends on the old wording.

## Recommendation

Approve as-is. m1/m2/m3 are optional polish — safe to land now and fold into a later pass (or
skip) at the owner's discretion.
