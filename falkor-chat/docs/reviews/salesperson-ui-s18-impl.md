# salesperson-ui — S18 implementation review (DEF-3 dead-turn-latch fix)

> **Status:** active · **Owner:** `analyst` · **Tracks:** DEF-3 (M<n> TBD)

## Scope & verdict

Reviewed the implementation of `U-DEF3-fix` (`tdd-engineer`, agent id `ab6861fe4dd72fc0f`) against
the already-gated design at `docs/plans/salesperson-ui.md` §4.14 / step S18 (design review:
`falkor-chat/docs/reviews/salesperson-ui-s18-design.md`, approve with suggestions). Baseline: the
uncommitted diff `git diff -- falkor-chat/server/falkorchat/storefront.py
falkor-chat/server/tests/test_storefront.py`. This is a **conformance gate** on the implementation,
not a re-litigation of the design — findings below are about whether the code and tests match what
§4.14 specified and whether they are correctly tested, not about the design's own merits.

**Verdict: approve.** No blockers, no majors. Two minors, both about the sixth test case's fidelity
to its own framing rather than any functional gap.

**CPG:** considered, not relevant — `cpg_falkorchat` exists in this FalkorDB instance, but this
review is a line-by-line diff comparison against an already-traced, already-gated design plus
direct test execution/mutation; a CPG's call-graph answers would not have shortened either of those
and I did not query it.

## Findings

### Minor — case 6 ("lifecycle... extended") is a new, narrower test, not literally an extension

§4.14's test-first done-condition, item 6, says "the existing lifecycle test extended to the new
failure shape." The implementer instead added a **new, sibling** test
(`test_the_dead_turn_latchs_lifecycle_holds_for_a_swallowed_envelope_too`,
`falkor-chat/server/tests/test_storefront.py:1229`) rather than editing
`test_the_dead_turn_latchs_lifecycle_set_postable_and_cleared_by_the_next_enqueue` (`:986`) in
place. The new test is also narrower: it proves only *set-by-failed-turn → cleared-by-next-post*,
while the original additionally exercises the `409`-refusal-while-latched and the
release-without-reaching-`enqueue_turn` sub-cases (Pass 20's named case).

This is defensible rather than a real gap — those two omitted sub-cases are pure
`reserve_turn`/`release_turn` mechanics that don't depend on *how* the latch got set, so re-running
them under a second trigger fake would be duplication, not new coverage — but it is a looser reading
of "extended" than the design's wording suggests, and it means the *specific* claim "lifecycle holds
for a swallowed envelope too" is proven only for the clear-on-next-post half, not the full lifecycle
the original test covers.

**Suggested improvement:** none required to unblock; if this file is touched again, consider a
one-line note in the new test's docstring making explicit that the 409/release sub-cases are
deliberately not re-run here (they're shape-independent), so a future reader doesn't read the
narrower test as a full lifecycle-parity claim.

### Nit — `_FailOnceViaEnvelope` is defined inline, unlike its module-level siblings

`_RecordingTrigger`, `_FailOnceTrigger`, and the new `_EnvelopeTrigger` are all module-level fakes;
`_FailOnceViaEnvelope` (`test_storefront.py:1237-1246`) is instead defined inside the test function
body. Purely cosmetic — no functional effect, and it's used exactly once — but it breaks the file's
own established pattern of promoting reusable `WorkflowTrigger` fakes to module scope.

**Suggested improvement:** take or leave; promote it alongside `_EnvelopeTrigger` if a future test
needs the same "envelope on turn N, `None` after" shape.

## Verification detail (independently re-derived, not inherited from the implementer's or the
coordinating session's reports)

- **Diff matches §4.14's prescribed code verbatim.** Compared `git diff -- storefront.py` line by
  line against the design's own code block (§4.14, "The change, concretely"): the `result =` capture,
  the `isinstance(result, dict) and result.get("status") == "failed"` guard, the `_log.error(...)`
  call (exact format string and four `%s` args, exact order — `participant_id`, `posted.get("msgId")`,
  `result.get("runId")`, `result.get("error")`), and the `_mark_turn_failed(participant_id)` call are
  all byte-for-byte what the design specifies, placed exactly where specified (inside the existing
  `try`, right after the `maybe_trigger` call).
- **Docstrings match §4.14's wording guidance.** `_mark_turn_failed`'s docstring now says "Called
  from either of `_run_turn`'s two failure-marking places" and "never from the request thread" —
  the design's own two requirements ("say it is now called from either that block or the post-call
  check", "both worker-thread-only"), both satisfied. `_run_turn`'s docstring now says "from one of
  two places" and cites §4.14/DEF-3 — exactly the design's own prescribed wording. Neither docstring
  overstates or understates what the code does.
- **Zero-touch claim confirmed independently**, not just accepted: `git diff --stat -- services.py
  executor.py api.py trigger.py` (repo root) returns empty, and `git status --short falkor-chat/`
  shows only `storefront.py` and `test_storefront.py` changed. `TERMINAL_OR_PARKED_STATUSES`
  (`services.py:127`) is still `frozenset({"failed", "done", "waiting"})`, matching the closed set
  the design's safety argument depends on — no drift since the design gate.
- **Case 1+2 combination is faithful, not a shortcut.** Compared against the file's own precedent:
  the existing raise-path test
  (`test_a_turn_whose_trigger_raises_is_isolated_and_still_clears_the_gate`, `:861-916`) *also*
  combines the isolation assertions (future/turn_in_flight/turn_state/lastTurn) and the logged-record
  assertions (count/level/exc_info/message content) into one test function. The new reproduction test
  mirrors this exact structure 1:1, including the "isolation holds, then the negative half of the
  logging discipline" ordering. Combining does not lose independent verifiability of either half —
  both are still four/five separate `assert` statements that fail independently and point at the
  specific broken invariant; it only avoids re-running the same enqueue/drain setup twice, which is
  what the file already does for the analogous existing test.
- **Responder-shape control (`post_agent_answer`'s envelope) verified against current source, not
  assumed stable since the design gate.** Re-read `services.py:975-989` directly:
  `post_agent_answer` returns exactly `{"msgId", "threadId", "authorId", "text", "role", "createdAt",
  "mentions", "seeds"}` — no `status` key — matching both the design's claim and the test's fixture
  dict.
- **Full suite reproduced independently, both scopes.** `falkor-chat/server/.venv/bin/python -m
  pytest -q tests/test_storefront.py` → 105 passed. Full suite from `falkor-chat/server`:
  `.venv/bin/python -m pytest -q` → 2830 passed, 14 deselected — matches both the implementer's and
  the coordinating session's reported figures exactly.
- **Two independent mutations, chosen outside both the implementer's table (delete-check,
  narrow-to-`"error"`-key, truthiness-instead-of-status) and the coordinating session's own
  (remove-`isinstance`-guard)**, both restored byte-identical afterward (confirmed via `diff` and a
  clean re-run of `tests/test_storefront.py`, 105/105):
  1. **Broadened the status check** to `result.get("status") in ("failed", "waiting")` (a plausible
     off-by-one on the closed three-value status set). **Killed cleanly** — 1 failure,
     `test_a_turn_whose_trigger_returns_a_successful_envelope_does_not_latch` (case 3, the positive
     control), correctly catching the over-broad match.
  2. **Swapped `and` for `or`** between the `isinstance` and `status` checks (a plausible operator
     slip). **Killed broadly** — 4 failures, including a crash (`AttributeError: 'NoneType' object
     has no attribute 'get'`) on the ordinary no-op-trigger path used by most ordinary-turn tests,
     plus both new positive-control tests and the new lifecycle test. Confirms the `and` is load-
     bearing in two independent ways (guards both the type and, transitively, the `None`-return case)
     and that the test suite would catch either failure mode on its own.
- **Log message content is sufficient for incident debugging** — carries `participantId`, `msgId`,
  `runId`, and the `error` string verbatim from the swallowed envelope, at `ERROR` (matching the
  existing `except` path's level, so an operator's `ERROR`-grep in the storefront logger finds both
  shapes of a died turn, exactly the design's stated intent). No `exc_info`/traceback is possible
  here (nothing raised), which the docstring and design both call out explicitly — not an oversight.

## What's solid

- The implementation is a verbatim, line-for-line match of the design's prescribed diff — no
  deviation, well-intentioned or otherwise, anywhere in the production code change.
- Six test cases cover exactly the check's decision boundary the design specified: the reproduction,
  the negative half of the logging discipline, both false-positive guards (successful envelope,
  responder fall-through), the independently-found resume-path gap, and lifecycle continuity — no
  case is a shape enumeration standing in for a coverage probe, and the two "must not latch" controls
  exhaust the check's actual boundary (`status == "failed"` vs. every other producible value vs. no
  `status` key at all).
- Mutation coverage is real, not decorative: three implementer mutations, one coordinator mutation,
  and two of my own (five distinct mutants total across the `isinstance`/`status`/`and` surface) all
  killed cleanly, with two of the five (the coordinator's guard-removal and my `or`-swap) killed via
  a crash on the file's own most-used ordinary-turn fake rather than a targeted assertion — strong
  evidence the guard is load-bearing production logic, not defensive dead code.
- The existing raise-path regression test is untouched (diff confirms no edits to it), and the full
  2830-test suite stays green.

## Open questions

None — no finding here needs a decision; both are logged as optional follow-ups, not blockers.
