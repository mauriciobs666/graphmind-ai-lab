# salesperson-ui — S18 design review (DEF-3 dead-turn-latch fix)

> **Status:** active · **Owner:** `analyst` · **Tracks:** DEF-3 (M<n> TBD)

## Scope & verdict

Reviewed `docs/plans/salesperson-ui.md`'s v1.40 amendment (`git diff` against the prior commit):
new §4.14 ("The dead-turn latch's swallowed-envelope gap (DEF-3)") and new step-table row **S18**,
plus the header/§5.0/§5.1/§5.2/§6.1 cross-reference updates that accompany it. Baseline for the
defect itself: `docs/test-reports/salesperson-ui-report.md`'s DEF-3 section. This is a **design
gate**, not a code review — nothing has been implemented yet; the deliverable under review is the
plan text and the `_run_turn` diff it prescribes for `tdd-engineer`'s `U-DEF3-fix`. Every mechanism
claim below was re-derived independently by reading `trigger.py`, `services.py`, `executor.py`,
`storefront.py`, `api.py` and the existing `test_storefront.py` fixtures directly — not inherited
from the plan's own prose or the dispatching session's summary.

**Verdict: approve with suggestions.** No blockers, no majors. One minor (a stale, unswept
cross-reference elsewhere in the same document) and one nit. `U-DEF3-fix` may be dispatched against
this design as-is.

**CPG:** considered, not relevant — `cpg_falkorchat` exists in this FalkorDB instance, but this
review traced four hand-picked functions across four files for return-shape and exception-flow
correctness; a CPG's call-graph/data-flow answers would not have shortened that trace (the
functions are already named and small), and I did not query it.

## Findings

### Minor — §9 "Ready to implement" summary was not swept for S18 (or S17 before it)

`docs/plans/salesperson-ui.md` §9 ("Ready to implement", lines ~2867–2881) still reads "21 steps
(S0–S16...)" and its "Dispatch order" line ends `... S12d → S15 → S16` — S18 is entirely absent, and
so is S17 (the i18n-chrome-scope step added at v1.39, before this amendment). §4.14's own closing
sentence claims "§5.0 and §5.1 swept to carry it; §5.2's ... paragraph and §6.1 gain one
sentence/bullet each" — an accurate, narrower claim that does not cover §9, and indeed §9 was left
stale by v1.39 too, so this is a pre-existing gap this amendment had the opportunity to close but
didn't. It is purely a summary section (`§5.0`'s file map and `§5.1`'s step table, which an
implementer actually dispatches from, are both correctly updated), so it does not block `U-DEF3-fix`.

**Suggested improvement:** fold into S18's own done-condition or, more efficiently, batch as a
one-line fix the next time any step touches §9 (S16's docs close-out pass is the natural next
touchpoint, since S16 already gates on "S18 and every other defect-fix unit" landing first) — update
the step count, the `S0–S16` range, and the dispatch-order chain to include S17 and S18 in one edit
rather than two.

### Nit — the swallowed-envelope check's `"failed"` literal is coupled to `TERMINAL_OR_PARKED_STATUSES`'s current membership, with no cross-reference

`services.py:127` defines `TERMINAL_OR_PARKED_STATUSES = frozenset({"failed", "done", "waiting"})`
— the closed set that makes `result.get("status") == "failed"` safe today (verified: `_drive`/
`_drive_loop`/`_fail_budget` never return any other status string, so `"failed"` is unambiguous, and
this is the exact reasoning §4.14 gives). If that frozenset ever grows a new terminal status that
also represents a fault (as opposed to today's exhaustive `done`/`waiting`/`failed`), `_run_turn`'s
check would silently need updating too, and nothing ties the two together.

**Suggested improvement:** not worth a runtime coupling for a three-string enum that has been stable
across the whole plan's history — but §4.14 or the `_run_turn` code comment could note the
dependency in one clause ("mirrors `TERMINAL_OR_PARKED_STATUSES`'s closed set") so a future change
to that frozenset is grep-discoverable from the check site. Take or leave.

## Verification detail (claims re-derived independently)

- **(a) Three-branch return shape** — confirmed against `trigger.py:53-93`. Branch 1 (loop-guard)
  returns `None`, not data-producing. Branches 2/3/4 return, respectively,
  `services.resume_workflow_run`'s dict, `services.start_workflow_run`'s dict, and
  `responder.maybe_respond`'s dict (== `post_agent_answer`'s dict on the fall-through path this
  demo actually exercises).
- **(b) `post_agent_answer` has no `status` key** — confirmed at `services.py:985-989`: returns
  exactly `{"msgId", "threadId", "authorId", "text", "role", "createdAt", "mentions", "seeds"}`.
- **(c) A successful envelope's `status` is never `"failed"`** — confirmed: `executor.py`'s `_drive`/
  `_drive_loop` return only `"waiting"` (:585, :642), `"done"` (:652), or `"failed"` (`_fail_budget`,
  :1375); `services.py:127`'s `TERMINAL_OR_PARKED_STATUSES` closes the set at exactly these three
  strings, and `_drive_or_fault` only ever reports a status drawn from that same re-read (:2546).
  No other status string is producible by any path the check can see.
- **(d) Zero changes to `services.py`/`executor.py`/`api.py`, REST/sweep contracts unaffected** —
  confirmed by file scope (S18's row names only `storefront.py`) and by tracing each named
  caller: `api.py`'s two REST routes (`start_workflow_run`/`submit_workflow_input`) and
  `services.sweep_due_workflow_runs` (:2567-2740) all still drive through the unmodified
  `_drive_or_fault`, and `resume_workflow_run` (:2554-2565) still calls `executor.resume` directly
  with no wrapper — none of these call sites reference `storefront.py` or `_run_turn`.
- **Second finding (resume-path budget exhaustion)** — confirmed real and independent of the main
  gap: `services.resume_workflow_run` never routes through `_drive_or_fault`, so when
  `executor.resume` internally hits `_fail_budget` (a normal return, "step budget exceeded",
  `executor.py:1369-1375`, never a raise), the dict `{"runId": ..., "status": "failed"}` — no
  `error` key — returns cleanly to `trigger.py` step 2 and, before this fix, past `_run_turn`
  unexamined. The single `isinstance(result, dict) and result.get("status") == "failed"` check
  closes this too, since it does not key off the `error` field's presence.
- **`_run_turn`'s before-context** — `storefront.py:1377-1398` matches the plan's quoted "before"
  code verbatim (the `maybe_trigger(...)` call's return value is fully unused), so the proposed
  diff's context lines apply cleanly.
- **Docstring claims** — `_mark_turn_failed`'s current docstring (`storefront.py:956-960`, "Called
  only from `_run_turn`'s own failure-isolation block ... never from the request thread") is exactly
  the text §4.14 says needs to change to "either that block or the post-call check" — confirmed by
  direct read, not assumed.
- **Test-plan grounding** — `_RecordingTrigger`/`_FailOnceTrigger` and
  `test_a_turn_whose_trigger_raises_is_isolated_and_still_clears_the_gate` exist exactly as cited
  (`test_storefront.py:759-908`, `966-`). Checked every other `maybe_trigger` fake in the file
  (`_Blocking`, `_Holding`, `_BlockingRecorder` at lines 1100/1135/1347) — each implicitly returns
  `None`, so the new post-call check introduces no regression risk against the existing suite.
  The six proposed cases correctly separate the reproduction (case 1), the negative/log-shape proof
  (case 2), two false-positive guards (cases 3/4 — a successful envelope and a responder dict, which
  matter because the check's whole safety argument rests on those two shapes never colliding with
  `"failed"`), the second gap's own proof that the check keys on `status` and not `error`'s presence
  (case 5, deliberately omitting the `error` key), and lifecycle continuity (case 6). No case is
  redundant with another; nothing in the six-case list is a shape enumeration masquerading as a
  coverage probe — the two "must not fire" controls exhaust the check's own decision boundary
  (`status == "failed"` vs. every other value the closed status set can produce, plus "no `status`
  key at all"), not an arbitrary sample of trigger shapes.

## What's solid

- The root-cause trace is accurate down to the line: `_drive`'s fault net always re-raises, but
  `_drive_or_fault` (used only by the start/submit-input paths) converts four named exception types
  into a normal return — and `resume_workflow_run`'s deliberate bypass of `_drive_or_fault` is why
  the *exception* path stayed safe on resume while the *budget-exhaustion* return path did not. This
  is a genuinely subtle distinction and the plan gets it right.
- The rejected alternative (option (b), widening `_drive_or_fault`) is argued on real, checked
  evidence — the REST error-map comment (`api.py`'s "Error map" block) and the sweep's `faulted`
  bucketing (`services.py:2713-2727`) both genuinely depend on today's swallow-and-return contract,
  so re-opening that method for one caller's benefit would be the wrong trade.
- Proportionality is right: one method, one new `if`, two docstring updates, no new graph read, no
  signature change. The fix reuses data `_run_turn` already receives rather than adding a second
  graph round-trip, and the plan explicitly argues why the re-read alternative would need the exact
  same guard anyway.
- The six-case test-first done-condition is well-scoped to the check's actual false-positive
  surface rather than an arbitrary list of trigger shapes, and correctly preserves the existing
  raise-path regression test unchanged.

## Open questions

None — the design is self-contained and the one minor finding does not need a decision, only a
future edit.
