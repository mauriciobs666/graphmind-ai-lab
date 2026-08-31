# `salesperson-tool-reliability2` — implementation review, K-061 fix

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-061 (post-M6)

## Scope & verdict

Reviewed commit `381c9fc` ("fix(falkor-chat): K-061 — same-turn add_to_cart self-duplicate dedup
guard") against `docs/BACKLOG.md`'s `### K-061` entry (owner/fix-shape/test-strategy) and the
diagnosis it's based on, `docs/reviews/salesperson-tool-reliability-ml.md` §12. In scope:
`server/falkorchat/executor.py`, `server/tests/test_executor_agent.py`, `docs/HISTORY.md` (the
three files the commit touches). This is a new topic slug per root `AGENTS.md` collision rule 2 —
K-061 is a fresh defect against the same `salesperson@v5` capability the original
`salesperson-tool-reliability` coordination shipped, not a revision of that closed coordination's
own `-impl`/`-impl2`/`-impl3` reviews.

**Verdict: approve with suggestions.** The guard is correctly placed, correctly keyed, correctly
scoped, and its headline reproduction claim is independently verified (below) — I reverted
`executor.py` to its pre-fix state in an isolated copy and confirmed the new
`test_same_turn_exact_repeat_of_own_successful_write_is_held` genuinely goes RED (both calls
dispatch, cart lands at quantity 2) without the fix, and GREEN with it. No blocker. Two findings
below are real gaps worth closing — one is a verified, reproducible mutation-testing hole in the
"must not poison the dedup set" guarantee; the other is a documentation-process gap (`BACKLOG.md`
left stale, and HISTORY's "resolved" header overclaims against K-061's own filed test strategy).

CPG: not applicable — no CPG is loaded for `falkor-chat` server code (per the task brief; only
`cpg_salesperson`, the UI, exists per `skills/cpg-analysis`).

## Findings

### MAJOR 1 — The "must not poison the dedup set" guarantee is structurally correct but has zero test coverage on the branch that actually matters

The code is right: `dispatched_writes.add(dispatch_key)` (`executor.py:1041-1042`) sits strictly
after the `try/except ServiceError` block, so a dispatch that raises a model-correctable
`ServiceError` returns (`executor.py:1035-1036`) before ever reaching the `.add()` line — a failed
dispatch cannot poison the set, by construction.

But I verified (copy-aside mutation, not on the live tree) that **no existing test would catch a
regression here**: I patched a scratch copy so a failed dispatch also seeds `dispatched_writes`
(mirroring exactly the bug BACKLOG's "must not poison the dedup set with a held/rejected call"
requirement warns against), then ran the full `test_executor_agent.py` file — **all 69 tests
passed**, including all three new K-061 tests and `test_same_turn_dedup_does_not_hold_a_call_that_
never_succeeded`. That test's name promises this exact guarantee, but it only exercises the
K-058-off-turn-held path, which returns *before* `dispatch_key` is even computed
(`executor.py:970-982`) — it never reaches the dispatch/seed code at all, so it cannot detect a
seed-on-attempt mutation. A future refactor that moves the `.add()` earlier (e.g. "simplifying" by
seeding right after computing `dispatch_key`) would ship silently: a customer whose `add_to_cart`
fails on a transient dispatch error and who is then correctly retried with identical arguments
would be wrongly told "already succeeded earlier this turn" — never actually added.

**Suggested fix:** add a test where `StubRegistry` raises a `MODEL_CORRECTABLE_TOOL_ERRORS`
member on the first `add_to_cart` call and returns success on an identical-argument retry; assert
the retry dispatches (not held) and `reg.dispatched` contains it.

### MAJOR 2 — `BACKLOG.md`'s K-061 entry is stale, and HISTORY's "resolved" header overclaims against K-061's own filed test strategy

The commit touches only `executor.py`, the test file, and `HISTORY.md` — `docs/BACKLOG.md` is
unchanged. Its `### K-061` entry (`docs/BACKLOG.md:98`) still reads "🟡 in-progress ... fix now
being designed", which is now false (a fix has shipped), and per root `AGENTS.md` ("`BACKLOG.md`
is forward-looking only — a delivered item does not stay in it, not even as an index row") this
entry should have been resolved out of `BACKLOG.md` in the same change.

More substantively: K-061's own filed **Test strategy** (`docs/BACKLOG.md:143-146`) has three
parts — reproduction test (done, verified above), offline suite green (done, verified below), and
**"live regression at the same n≈20-30 ... to confirm the rate actually drops"**. `HISTORY.md`'s
own entry admits this third part "was not run here — out of scope for this unit test-first fix,
left for whoever runs the next live verification pass" — yet the entry's own header states flatly
"K-061 resolved". Calling K-058's own confirmatory step optional in body text while the section
title claims full resolution is inconsistent, and because `BACKLOG.md` was never touched, there is
now **no forward-looking, trackable record** of the still-outstanding live-regression obligation —
it exists only as a footnote inside a change-log entry, which per this repo's own documentation
convention is not where anyone looks for open work.

**Suggested fix:** either (a) retitle the `HISTORY.md` header to something like "K-061 fix shipped
(unit-verified); live regression confirmation still open" and leave a slimmed-down `BACKLOG.md`
K-061 entry (or a new follow-up item) tracking just the live-regression step, or (b) if the live
regression is judged non-blocking enough to fully close K-061 without it, say so explicitly and
remove the K-061 entry from `BACKLOG.md` per convention. Either is fine; leaving both docs as-is
(stale BACKLOG + an overclaiming HISTORY header) is not.

### MINOR 1 — `test_same_turn_dedup_does_not_hold_a_call_that_never_succeeded`'s name promises more than it tests

Same underlying fact as MAJOR 1, framed as a naming/scope issue rather than a coverage gap: the
test's docstring comment says "a call that was itself HELD by K-058 (off-turn) must not be treated
as 'already successfully dispatched'" — accurate for what it tests — but the test's *function
name* ("a call that never succeeded") reads as covering the broader "must not poison" requirement
BACKLOG states, including a dispatch that reaches `self._tools.dispatch` and fails. Once MAJOR 1's
suggested test is added, consider renaming this one to
`test_same_turn_dedup_does_not_hold_an_off_turn_held_call` (or similar) so the two together
actually match their names to what they cover.

## What's solid

- **Placement is correct and verified by reading, not just by docstring claim**: the K-061 block
  sits inside the same `if target_arg:` branch, strictly after the K-058 off-turn check returns,
  so K-061 only ever evaluates a call that already passed K-058 — confirmed at
  `executor.py:968-1004`.
- **Dedup key genuinely includes the full argument set**: `dispatch_key = (call.name,
  _dumps(call.arguments))` with `_dumps` using `sort_keys=True` (`executor.py:282-285`) for stable
  ordering; `test_same_turn_different_args_for_same_target_still_dispatches_both` exercises two
  different-`quantity` calls for the same product and both dispatch — confirmed by reading and by
  running the test.
- **Reproduction test genuinely fails without the fix.** Verified independently: copied
  `executor.py` aside, checked out `381c9fc^`'s version into the copy, ran
  `test_same_turn_exact_repeat_of_own_successful_write_is_held` against it via a `PYTHONPATH`
  override (the editable install's own finder otherwise always wins) — it failed exactly as
  claimed, both calls dispatched, cart doubled. Ran the same test against the real fixed code —
  green. No `git stash`/`git checkout <path>` was used on the live tree for this.
- **Observability signal is genuinely distinct**: `_note_same_turn_write_held`
  (`same_turn_write_held` trace kind, its own `_log.warning`) is a separate method from
  `_note_off_turn_write_held` (`off_turn_write_held`), and the two are asserted by disjoint
  assertions in the tests (`executor.py:1103-1163`).
- **Blast radius fully accounted for**: `_handle_tool_call` has exactly two call sites in the
  whole file (`executor.py:855`, `:881`), both updated; it's a private method with no callers
  outside `executor.py` (`grep -rn _handle_tool_call` across `server/`).
- **Scope discipline held**: `_WRITE_TARGET_ARG` is unchanged (`add_to_cart`/`remove_from_cart`
  only) — `place_order`/`clear_cart` untouched, matching K-059's own still-`🔵 proposed` status
  (confirmed current in `BACKLOG.md:43`). K-058's held-reason string block (the text K-062 is
  about) is byte-unchanged in the diff — no drift into K-062's scope.
- **HISTORY.md's suite figures check out**: `pytest --collect-only -q` on the current tree reports
  "2305/2319 tests collected (14 deselected)" — matches "2305 passed, 14 deselected" exactly. The
  three new test names in `HISTORY.md` match the diff verbatim.

## Open questions

- Should the live n≈20-30 regression (K-061's own filed test-strategy item 3) be scheduled now, or
  is unit-test-first + mutation-test verification judged sufficient to call this "resolved" given
  the fix's narrow, structurally-obvious mechanism? This is a judgment call for whoever owns
  `BACKLOG.md`'s next pass (`teco`/the fix's own owner), not something this review can settle —
  flagged as MAJOR 2 above because the current state (silently skipped, untracked) isn't a
  deliberate answer to that question either way.

## Pass 2 (2026-08-31)

**MAJOR 1 / MAJOR 2 / MINOR 1 disposition:** all closed by U4 (same coordination), independently
re-verified byte-identical by `teco` and committed as `381fdb8`; MAJOR 2's documentation gap was
handled by `teco` directly, not this delegate. Not re-litigated here. This pass reviews an
unrelated, later diff against the same K-061 topic: the live n≈25 regression pass (U5,
`ml.md` §15.2) found a **second, narrower loophole** in the same guard's own keying — this pass
covers `tdd-engineer`'s (U6) fix for it. Diff (uncommitted): `server/falkorchat/executor.py`
(new `_DEDUP_ARG_RESOLVERS` table + `_resolve_dedup_arguments`, wired into the K-061
`dispatch_key` computation), `server/tests/test_executor_agent.py` (two new tests),
`docs/HISTORY.md` (one new entry).

**Verdict: approve.** No blockers, no majors. One minor worth closing before or shortly after
commit.

### MINOR 1 — `_DEDUP_ARG_RESOLVERS`'s "no entry ⇒ pass through raw" default has no guardrail against a future write tool silently reintroducing this exact bug class

`_resolve_dedup_arguments` (`executor.py:364-369`) falls through to raw `arguments` for any tool
name with no table entry — correct today because `_WRITE_TARGET_ARG` (`executor.py:321-324`) has
exactly two members, and `remove_from_cart`'s absence from the table is a deliberate, verified
choice (`RemoveFromCartTool.run`, `tools.py:646-654`, passes `quantity` through as `None`
unchanged — no wrapper default to mirror). But nothing ties the two tables together: if a future
change adds a third tool to `_WRITE_TARGET_ARG` whose own `run()` wrapper applies a hidden
post-guard default (exactly K-061's original mechanism), the new tool falls through to "raw
arguments," silently reintroducing the same bug class, with no test or comment anywhere flagging
that `_DEDUP_ARG_RESOLVERS` needs a matching entry. The current code comment
(`executor.py:333-358`) explains *why* the existing two tools are handled the way they are, but
doesn't warn a future editor of `_WRITE_TARGET_ARG` to check back here.

**Suggested fix:** a one-line comment on `_WRITE_TARGET_ARG` itself ("adding a write tool here?
check whether its own `run()` wrapper applies a post-guard argument default, and if so add a
`_DEDUP_ARG_RESOLVERS` entry mirroring it — K-061's original bug, `ml.md` §15.2") is enough; a
test isn't practical since there's no third tool yet to test against.

### Verified independently

- **Resolver mirrors `AddToCartTool.run` exactly**: `arguments.get("quantity") or 1`
  (`tools.py:589`) byte-for-byte matches `_resolve_add_to_cart_dedup_args`
  (`executor.py:327-334`). **`remove_from_cart`'s omission is correct**: `RemoveFromCartTool.run`
  passes `arguments.get("quantity")` through unchanged (`tools.py:650`), and `services
  .remove_cart_item`'s `quantity=None` branch reads the current line and removes it whole
  (`services.py:2696-2726`) — confirmed by reading, not the delegate's claim alone.
- **Both new tests are real, targeted assertions**, not smoke tests: `reg.dispatched` (the actual
  tool-dispatch record, `StubRegistry.dispatch`, `test_executor_agent.py:83-85`) is asserted
  exactly, not just "no exception" — the first new test asserts only the omitted-quantity call
  reaches dispatch; the second asserts both `remove_from_cart` calls do.
- **Mutation-tested both directions myself** (not just re-running the delegate's claim), via `cp`
  backup (not `git stash`, per this coordination's own U2 lesson), restored `md5sum`-identical
  after each: (1) reverted `dispatch_key` to the raw pre-fix `_dumps(call.arguments)` — the new
  `test_same_turn_omitted_and_explicit_default_quantity_are_the_same_key` failed for exactly the
  predicted reason (both calls dispatched), the other 5 same-turn tests stayed green; (2) added a
  wrong `remove_from_cart` resolver entry that collapsed omitted quantity to `1` — the new
  `test_same_turn_remove_omitted_quantity_and_explicit_quantity_are_different_keys` failed for
  exactly the predicted reason (one call wrongly held). Both mutations independently confirm the
  tests catch the specific regressions they claim to.
- **The carve-out survives, and is still meaningfully distinct**:
  `test_same_turn_different_args_for_same_target_still_dispatches_both`
  (`test_executor_agent.py:1831-1863`, quantities 1 and 2, both nonzero) is unaffected by the
  resolver (`1 or 1 = 1`, `2 or 1 = 2` — no collapse), stays green, and tests a genuinely different
  case from the two new tests (a real customer-intended quantity change vs. the same intended
  quantity expressed two syntactically different ways).
- **Scope discipline holds**: the K-061 guard's docstring (`executor.py:978-996`) is updated
  accurately to describe resolved-argument keying and still names the carve-out correctly; no
  other guard (K-058, K-059) or unrelated code is touched.
- **`HISTORY.md`'s claims all check out against source**: the pre-fix line citations
  (`executor.py:989`, `:944-947`) match `381c9fc`'s committed tree exactly; the `services
  .remove_cart_item` "whole line" claim and the wrapper-default mechanism description are both
  accurate (see above).
- **Full offline suite reproduced exactly**: `server/.venv/bin/python -m pytest -q` →
  **2309 passed, 14 deselected**, matching the delegate's report. Shared state
  (`reference`/`ws:acme`) re-verified `OK` after re-seeding
  (`bootstrap_schema.sh acme` → `seed_demo.sh` → `seed_workflows.sh` → `seed_catalog.sh` →
  `seed_salesperson.sh acme`, all three `verify_*.sh` reports `OK`).

CPG: not applicable — no CPG is loaded for `falkor-chat` server code (unchanged from Pass 1).
