# `must-post-engine-contract` — implementation review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-027 (M3.5)

## Scope & verdict

Diff-scoped, post-implementation re-gate of Units U5 (`coder`, `executor.py` +
`test_executor_agent.py`) and U6 (`coder`, `services.py` + `test_services.py` +
`seed_workflows.sh`), implementing `falkor-chat/docs/plans/must-post-engine-contract.md`
(Version 2, already gated at `falkor-chat/docs/reviews/must-post-engine-contract.md`, verdict
*approve with suggestions*, all four findings folded into v2). Baseline: `git diff` of the current
working tree against `HEAD` for exactly the five files named in the brief — nothing else is
touched, nothing is committed. This is not a second design review; the plan's soundness was
already judged. I checked: (1) diff-vs-plan conformance at the exact insertion points named, (2)
the `post_message`-via-`emissions` vs. generic-tool-via-`satisfied` split, (3) both
`_run_agent_node` exit points actually calling the check, (4) the publish-time invariant's three
sub-checks and exception type, (5) test quality including the supplementary test's soundness, (6)
both units' own claims (test counts, mutation-testing result, SHA-lock reconfirmation), (7)
backward compatibility by tracing the no-`requiredTools` path, (8) scope leak into files/functions
the plan ruled out.

**Verdict: approve.** No blockers, no majors. Every insertion point, every logic split, and every
compatibility claim I checked matched the plan and the live code exactly. One minor and one nit
below, both cosmetic/informational, not blocking.

**CPG: considered, not relevant — the target diff is uncommitted working-tree state, and
`cpg_falkorchat`'s only build (`CpgBuildInfo.BUILT_AT = 2026-08-17T00:40:42Z`, `sourceCommit`
null, confirmed by direct query) necessarily predates it — a call-graph query against that graph
would answer for the *pre-change* code, not the diff under review, so it cannot verify the new
call sites (`_missing_required_tools`, `_note_must_post_violation`, the `satisfied` threading)
this review exists to check. Direct reading of every diff hunk (below) is the correct verification
method for an uncommitted diff, and is what I did instead.**

## Findings

### Minor

**m1 — `required`'s initialization sits a few lines earlier than the plan's own placement
guidance, with zero behavioral effect.** §9 step 1 says to initialize `required` "alongside the
existing `trace`/`emissions`/`last_text` initializations, before the iteration loop"
(`executor.py:699-707`); the actual diff initializes it right after `max_iter`
(`executor.py:692-697`), a few lines earlier, still unconditionally before the loop and still with
identical semantics (it's read only inside `_missing_required_tools` calls after the loop). Purely
cosmetic — no logic depends on the exact statement order between these two blocks, and grouping it
with the `granted_set` computation it depends on (`& granted_set`) is arguably a *better* locality
choice than the plan's own suggestion. **Suggested action:** none required; noted only so the
implementer/reviewer trail records the (harmless) deviation from the plan's literal wording.

### Nit

**n1 — `satisfied` is threaded through the K-039 implicit-dispatch call site
(`executor.py:751-754`) even though that call always dispatches the literal tool name
`post_message`, whose satisfaction `_missing_required_tools` reads exclusively off `emissions`, never
`satisfied`.** The `satisfied.add("post_message")` this produces (inside `_handle_tool_call`'s
success path) is therefore currently inert for this contract's own logic — a fact the plan's own
§3.2 rationale anticipated and accepted for uniformity ("both call sites... unchanged otherwise").
Not a defect; flagging only because a future reader tracing why `satisfied` is threaded through a
call site whose only possible tool name never reads it might wonder if something was missed. No
action needed — this is the plan's own designed shape, correctly implemented.

## Verification detail (what was checked, and how)

- **Insertion points match exactly.** `executor.py`: `_missing_required_tools` lands in the
  "value objects" helper section near `_str_list`/`_dumps` (`:253-275`), as specified. The
  must-post check is called at both of `_run_agent_node`'s own return points — immediately after
  the K-039 implicit-dispatch attempt inside the non-tool-call branch (`:756-758`) and immediately
  before the `maxIterations`-exhaustion `return` (`:779-781`) — confirmed by reading both call
  sites directly, not taking either report's word. `services.py`: the new invariant's loop
  (`:960-989`) lands textually **between** the existing `waitsForHuman` loop (ending `:958`) and
  the `for tr in transitions: ... validate_cmp(...)` loop (starting `:991`) — read directly, exact
  match to plan §3.4/§9's "second of four" placement.
- **The `post_message`-via-`emissions` / generic-tool-via-`satisfied` split is implemented exactly
  as designed**, verified by reading `_missing_required_tools`'s body (`executor.py:253-275`): the
  one `if name == "post_message"` branch checks `emissions`; every other name checks `satisfied`
  membership. No path collapses the two. `_handle_tool_call` adds `call.name` to `satisfied` only
  at the single success path that survives AC-6 rejection, malformed-arg rejection, and an absorbed
  `ServiceError` (`:855-856`, right after `trace.append(("tool_result", ...))`) — the same place the
  plan names.
- **Publish-time invariant's three sub-checks**, read in the order they execute
  (`services.py:966-989`): (a) list-of-strings — `isinstance(required, list)` +
  `all(isinstance(t, str) ...)`; (b) `type == "agent"`; (c) `⊆ config.tools` via
  `set(required) - granted`. All three raise `WorkflowDefSpecError` with a message in the
  `waitsForHuman` check's own style, and none of the four `_publish` test paths I traced (below)
  writes anything to `FakeRepo` before raising.
- **Backward compatibility traced, not trusted.** `config.get("requiredTools", [])` on a def with
  no such key returns `[]`; `_str_list([])` returns `[]`; `set() & granted_set` is `{}`;
  `_missing_required_tools` iterates zero names and returns `{}`; the `if missing:` guard at both
  call sites is false; nothing is logged, nothing is appended to `trace`, and both
  `StepResult(...)` constructions (`:759-763`, `:783-786`) are byte-identical to what they were
  before this diff — confirmed by diffing the actual constructor calls, which show only the
  five-line insertion of the check itself, no change to any `StepResult` field or value. `_str_list`
  also degrades gracefully on a present-but-`None`/non-list `requiredTools` (`isinstance` guard,
  returns `[]`, never raises) — the drive-time posture the plan cites holds under inspection.
- **Test-count and mutation-testing claims re-verified, not trusted.**
  `git diff tests/test_executor_agent.py | grep -c '^+def test_'` → 9;
  `git diff tests/test_services.py | grep -c '^+def test_'` → 4 — both match the coordination
  ledger's claimed counts (U5: "9 tests, added 1 beyond plan's 8"; U6: tests 9-12) exactly.
  The supplementary test's mutation-testing rationale was independently re-derived, not taken on
  faith: if the main dispatch loop's `_handle_tool_call` call site regressed to not thread
  `satisfied` (e.g. passed a throwaway local set instead of the shared one), every one of the
  plan's original 8 tests would still pass — test 1 uses `post_message`, whose check bypasses
  `satisfied` via `emissions`; tests 3/5 use `notify_owner` but only assert it's *never*
  dispatched (so `satisfied` never mattering either way). Only a test that dispatches a
  **non-`post_message`** tool successfully via the **main** loop and asserts *no* violation would
  catch that regression — exactly what
  `test_compliant_node_dispatching_a_non_post_message_required_tool_leaves_no_violation_trace` does
  (`notify_owner` dispatched via `result.tool_calls`, `requiredTools=["notify_owner"]`, asserts no
  `must_post_violation`). The claim holds; the addition is real and non-redundant, not merely
  present.
- **SHA-lock reconfirmed independently, not re-derived from either unit's or `teco`'s report.** Ran
  the documented recipe myself from the repo root:
  `awk '/^    def _drive_loop/{f=1} /^    # ── seams/{f=0} f' server/falkorchat/executor.py | sed
  -e :a -e '/^\n*$/{$d;N;};/\n$/ba' | sha256sum | cut -c1-12` → `71055f756280`, matching
  `falkor-chat/AGENTS.md`'s locked value and all three prior confirmations.
- **Full offline suite run independently**: `cd falkor-chat/server && .venv/bin/python -m pytest
  -q` → `1064 passed, 2 deselected` (the two `-m live` tests, correctly deselected by default), one
  pre-existing, unrelated `StarletteDeprecationWarning`. Green, confirmed by execution, not by
  trusting either unit's report. (Per the brief, this run wiped the `reference` graph at teardown
  as documented in `falkor-chat/AGENTS.md`'s testing-hazards note — expected, not a defect, and
  `teco` owns re-seeding separately.)
- **No scope leak.** `git status --porcelain=v1 falkor-chat/` shows exactly the five files named in
  the brief as modified (plus pre-existing untracked plan/review/coordination docs from the earlier
  design-gate phase, not part of this diff). `seed_workflows.sh`'s change is confined to the
  `intake`/`answer` step configs in the `triage@v1` literal — `research` (which grants only
  `graphrag_retrieve`) is untouched, and `access-request@v1` is untouched, matching plan §8's exact
  scope. `StepResult`'s dataclass shape, `_drive_loop`, `_select_transition`, `_trace_step`, and
  `resume` show zero diff lines anywhere in `executor.py`'s changes — confirmed by grep against the
  diff, not by assuming the plan's "explicitly out of scope" list was honored.
- **`falkor-chat/docs/BACKLOG.md`/`HISTORY.md`/`DESIGN.md` are unmodified** — this is *not* a
  finding: the coordination doc (`docs/plans/must-post-engine-contract-coordination.md`, U7 row)
  explicitly sequences the doc updates (§9 step 4 of the plan) as a separate, still-queued unit
  after U5+U6, so their absence from this diff is by design, not an omission by either coder unit
  under review here.
- **Trace-gating consistency spot-checked.** `_note_must_post_violation` appends to `trace`
  unconditionally (no `run["trace"]` check inline), which at first read looks like it contradicts
  §3.3's "when tracing is on (debug runs only)" framing for the trace-entry half. Tracing the
  actual gate: every other `trace.append` call in this file (e.g. `("tool_result", ...)` at
  `:855`) is equally unconditional — the real debug-run gate is centralized once, in `_drive_loop`
  (`tracer = self._tracer if run["trace"] else _NULL_TRACER`, `:466`), before `_trace_step` ever
  forwards the list to a real tracer. The new code follows the file's existing pattern exactly; it
  is not a bug.

## What's solid

- **Every insertion point named in the plan is exactly where the diff put it** — `services.py`'s
  new invariant genuinely lands between `waitsForHuman` and `validate_cmp`, not "somewhere in the
  LAST block"; both `_run_agent_node` exit points genuinely call the check.
- **The `post_message`/`satisfied` split is correct and the one case it exists to catch
  (`PostMessageTool.run`'s no-thread-bound decline) is exercised by a dedicated test** — test 4's
  registry double reproduces that exact non-raising-error-string shape.
- **Backward compatibility is real, not asserted** — traced end-to-end through `_str_list`,
  the `&` intersection, and both unchanged `StepResult(...)` constructions.
- **The publish-time invariant mirrors `waitsForHuman`'s style and exception type precisely**, and
  its four new tests correctly isolate each of the three sub-checks from the other two (e.g. the
  non-agent-step test deliberately pre-satisfies the `⊆ config.tools` check so the failure can only
  come from the `type == "agent"` check).
- **The supplementary test is a genuine, verified improvement**, not a report inflating its own
  value — its stated mutation-testing rationale reproduces under independent re-derivation.
- **Both units' self-reported claims held up under re-verification** — test counts, the SHA-lock,
  and the "offline suite green" claim all checked out by rerunning them myself rather than trusting
  the reports.

## Open questions

None. The two remaining plan-level open questions (M1's scope-boundary acceptability, the `ws:acme`
rollout decision) are stakeholder-level and were already resolved outside this diff-scoped
gate — the coordination ledger's U9 row records the user's decision ("leave as-is, tracked
follow-up, not blocking"). Nothing here needs further input before this diff can land.
