# `workflow-diff-absent-key` — review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-005 (M5)

## Scope & verdict

Reviewed the uncommitted, on-disk K-005 fix (`tdd-engineer`, U1 in
`falkor-chat/docs/plans/workflow-diff-absent-key-coordination.md`): the `try`/`except
ResponseError` added to `Repository._read_structure`
(`falkor-chat/server/falkorchat/repository.py:1809-1817`), its docstring update, the four new
tests in `falkor-chat/server/tests/test_repository.py`, the `falkor-chat/AGENTS.md`
`test_queries.sh` row edit, and the new 2026-08-25 `falkor-chat/docs/HISTORY.md` entry — against
the coordination doc's stated goal and fix scope. Scope was deliberately narrowed per the brief:
the concurrent K-051 hunks in `repository.py` (~1026-1120) and `test_repository.py` (§14
Document.status block) were read only to distinguish them from the K-005 hunk, never reviewed.
Verified by reading the whole `_read_structure` function (not just the diff hunk), tracing both
callers (`read_def_structure`, `read_snapshot_structure`) up through `Services.diff_def_snapshot`
and the `/diff` REST route (`api.py:474-489`, which has no `_read_or_absent` wrapper of its own),
confirming `_read_subgraph` (`repository.py:1746-1767`) is untouched, running the four new tests
live, and a temporary mutation (removed the `except`, confirmed the two "returns None" tests fail
while the reraise test still passes, then restored the exact original text and re-diffed to
confirm a net-zero change — see Appendix).

**Verdict: needs changes** — one Major: the coordination doc's fix item 2 has two named parts
(the `test_queries.sh` row and the `verify_workflows.sh` row); only the first landed, and
`HISTORY.md`'s new entry claims both did.

**CPG:** considered, not relevant — the coordination doc itself flags `cpg_falkorchat` as stale
for this unit (built 2026-08-17, 14 commits since including feature milestones) and pins the
affected functions by file:line instead; I followed that guidance and read the live files
directly rather than querying the graph.

## Findings

### Major — `HISTORY.md`'s K-005 entry claims a doc fix that wasn't made

`falkor-chat/docs/HISTORY.md`'s new entry says: *"its `verify_workflows.sh` row notes this
false-negative case existed and is fixed."* This is false. `git diff -- falkor-chat/AGENTS.md`
shows only the `test_queries.sh` row (line 71) was edited; the `verify_workflows.sh` row (line 76)
is byte-identical to before — still just "Read-only check that `reference` and `ws:<id>` agree...
Exit `0` = in sync, `1` = missing/divergent," with no mention of the fixed false-negative-on-full-
delete case. This is exactly the "check it against the actual diff, don't just check it reads
plausibly" trap: the sentence is plausible prose but doesn't match `git diff`.

This also means the coordination doc's own fix item 2 (`workflow-diff-absent-key-coordination.md`
lines 24-29, "the `verify_workflows.sh` row should note the false-negative-on-full-delete case so
an operator doesn't mistake it for real data loss") is only half-delivered — the operationally
important half, since `verify_workflows.sh`'s own `read()` wrapper is the thing whose exit code an
operator actually reads after `test_queries.sh`, and that row still gives no hint that a `1`/
"missing" reading there used to conflate "actually missing" with "reference key fully deleted, now
fixed elsewhere."

**Suggested fix:** add a note to the `verify_workflows.sh` row (`falkor-chat/AGENTS.md:76`) — e.g.
append something like *"Before K-005 (fixed 2026-08-25), a full `reference` `GRAPH.DELETE` (as
`test_queries.sh`'s teardown performs) made this report the `ws:<id>` snapshot itself missing too;
that false negative is fixed — a report after this date reflects real state."* Then correct
`HISTORY.md`'s claim to match (or drop the "verify_workflows.sh row" clause from `HISTORY.md`
until the row is actually edited).

## What's solid

- **Correctness (check 1).** Read the whole function, not just the hunk: `_read_structure` reaches
  its second `ro_query` (`_READ_TRANSITIONS_CYPHER`) only after `meta.result_set` is confirmed
  non-empty, which can only happen if the first query didn't raise — so in the target scenario
  (a fully `GRAPH.DELETE`d key, checked once per call) the early return is always reached before
  the second call. The only residual gap is a same-invocation race — the key being deleted by a
  *different* process between the two `ro_query` calls inside one `_read_structure` call — which
  would still raise uncaught from the second query. This is a genuine, if narrow, gap; I'm not
  raising it as a finding because it's a different failure class (concurrent mutation mid-read,
  not the "already fully deleted" scenario this fix targets) and the coordination doc scopes the
  fix to the latter.
- **Over-catching (check 2).** The `except` re-raises anything whose message doesn't contain
  "empty key" — verified live: `test_read_structure_reraises_response_errors_that_are_not_empty_key`
  feeds a `ResponseError("RediSearch: Syntax error at offset 6")` through the fake graph and
  asserts it propagates (`pytest.raises(ResponseError, match="Syntax error")`), passing against
  the real fix.
- **Untouched invariant (check 4).** `_read_subgraph` (`repository.py:1746-1767`, feeds
  `materialize_def` and the SHA-locked executor path) has no `try`/`except` — confirmed via `Read`,
  not just the diff — matching its own "must not move" docstring.
- **Test quality (check 3).** `test_read_snapshot_structure_none_when_graph_key_fully_deleted` is a
  genuine live reproduction: `graph.query("RETURN 1")` to materialize a real key, then
  `graph.delete()` (the actual `GRAPH.DELETE` op), on a throwaway `ws:k005probe` key — matching
  `falkor-chat/AGENTS.md`'s "probing shared graph state without mutating it" guidance, not the
  shared `ws:test`/`reference`. Mutation-tested it myself (see Appendix): removing the `except`
  makes the two "returns None" tests fail with the real `ResponseError` while the reraise test
  still passes, confirming the fix — not incidental test structure — is what's gating green.
- **Blast-radius tracing.** Confirmed `Services.diff_def_snapshot` (`services.py:1748`) reads
  `read_def_structure` before `read_snapshot_structure`, so the pre-fix bug really did short-circuit
  the snapshot read entirely, exactly as the coordination doc describes — and that the `/diff` REST
  route (`api.py:489`) calls `diff_def_snapshot` with **no** `_read_or_absent` wrapper of its own
  (unlike `check_demo_readiness`), so pre-fix it would have 500'd on a fully-deleted `reference`
  rather than returning the documented 200; the fix repairs that route too, not just
  `verify_workflows.sh`/`check_demo_readiness`.
- **`test_queries.sh` row accuracy (check 5, first half).** The `falkor-chat/AGENTS.md` edit
  correctly reflects that `GRAPH.DELETE` destroys indexes/constraints too, not just data, and
  that the full `bootstrap_schema.sh` → `seed_demo.sh` → `seed_workflows.sh` sequence is needed —
  this matches the coordination doc's fix item 2 and the actual behavior of `GRAPH.DELETE`.
- All four K-005 tests pass in isolation and the full offline suite is green apart from one
  pre-existing, out-of-scope K-051 failure (`test_start_document_progress_zero_total_jobs_flips_
  straight_to_ready`) that belongs to the concurrent session, not this diff.

## Open questions

- Should the `verify_workflows.sh` row fix land as a follow-up to this same unit (cheapest, since
  the context is fresh) or get its own tracked item? Coordination doc names it as part of the same
  fix item 2, so I'd default to finishing it here rather than opening a new ticket, but that's
  `teco`'s call given the concurrent-tree wrinkle already complicates this file's commit.

## Appendix — mutation test transcript

```
$ cd falkor-chat/server && .venv/bin/python -m pytest -q -k \
    "read_structure_none_when or read_snapshot_structure_none_when_graph_key or reraises_response_errors"
....                                                                     [100%]
4 passed, 1766 deselected

# Temporarily removed the `try`/`except ResponseError` in _read_structure (Edit), re-ran:
2 failed, 2 passed
FAILED test_read_snapshot_structure_none_when_graph_key_fully_deleted
  redis.exceptions.ResponseError: Invalid graph operation on empty key
FAILED test_read_structure_none_when_graph_key_fully_deleted
  redis.exceptions.ResponseError: ERR Invalid graph operation on empty key
# reraise test still passed, as expected (no except at all still propagates ResponseError)

# Restored the exact original text via Edit; re-ran:
....                                                                     [100%]
4 passed, 1768 deselected
# git diff -- falkor-chat/server/falkorchat/repository.py confirmed only the original K-005 +
# K-051 hunks remain — no net change from the mutation round-trip.
```
