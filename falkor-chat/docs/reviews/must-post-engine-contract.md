# `must-post-engine-contract` — plan review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-027 (M3.5)

## Scope & verdict

Static, pre-implementation review of `falkor-chat/docs/plans/must-post-engine-contract.md`
(architect, v1) against K-027 item 2 + its K-025-QA addendum (`falkor-chat/docs/BACKLOG.md`),
`docs/reviews/mention-reply-delivery-rca.md` §5, `falkor-chat/docs/DESIGN.md` §6, `falkor-chat/AGENTS.md`'s
executor-invariants block, and the actual current code of `server/falkorchat/executor.py`,
`server/falkorchat/services.py`, `server/falkorchat/tools.py`, `scripts/seed_workflows.sh` and
`scripts/bootstrap_schema.sh`. Every citation the plan makes to a file/line/function/test name was
independently re-read against the live tree (not taken on the plan's word); the CPG blast-radius
claim was independently re-run, not just re-stated. I did not evaluate implementation code (none
exists yet) or run the offline suite (nothing to run). This is the design gate; a diff-scoped
re-gate belongs at `docs/reviews/must-post-engine-contract-impl.md` once code lands.

**Verdict: approve with suggestions.** No blockers. The mechanism is sound, proportionate, well
precedented, and every grounding claim I checked held up byte-for-byte. Two majors and a couple of
minors below are worth resolving — mostly by naming a limitation explicitly rather than by
redesigning anything — before or during implementation.

**CPG: used `cpg_falkorchat` — independently re-ran the plan's own call-graph query
(`(caller:METHOD)-[:CONTAINS]->(call:CALL {NAME:"_handle_tool_call"})` and the equivalent for
`_run_agent_node`) rather than trusting its stated result, and separately queried the freshness
marker (`CpgBuildInfo.BUILT_AT = 2026-08-17T00:40:42Z`, `sourceCommit` null) in the same
investigation step. Both call-graph claims reproduced exactly: `_handle_tool_call` has exactly one
caller, `_run_agent_node`, with 2 call sites; `_run_agent_node` has exactly one production caller,
`_execute_step` (all 34 other callers are `tests/test_executor_agent.py`). Freshness: `git log
--oneline --since=2026-08-16 -- falkor-chat/server` returns zero commits, and the last actual commit
to that tree (`2026-08-16T22:01:50-03:00` = `2026-08-17T01:01:50Z`) predates the graph's build
timestamp — the graph is current, no rebuild needed.**

## Findings

### Major

**M1 — The "visible, diagnosable reason" promise (§1 goal, §3.3) doesn't hold for the actual
demo/production case; the plan doesn't say so.** §3.3's chosen design writes the violation to two
places: an unconditional `_log.warning` (process log only) and a `("must_post_violation", …)` trace
entry gated on `run["trace"]` (debug runs only). For a non-debug run — which is every real `@mention`
in the shipped demo, since nothing in `trigger.py`/`app.py` starts a debug run by default — **nothing
is written onto the run or step-run at all**, and the trace panel the web UI ships for exactly this
purpose (`web/app.js:461-466`) renders the literal string `"No trace events (not a debug run)."` A
demo presenter (or the `ws:acme` shared-box audience the RCA and §8/§11 explicitly name) watching a
run finish `done` with no reply sees **exactly what they see today** — the plan makes the failure
diagnosable to whoever has server log access, not visible to whoever is watching the run. The plan's
own §1 goal states "the run records a visible, diagnosable reason it didn't [dispatch]" — for the
common case, the run records nothing; a log line is emitted by the process, which is a different
claim. This isn't a design flaw (§2's `output`-shape argument for why the violation can't ride on
`StepResult.output` is correct and I didn't find a cheap alternative that doesn't reopen that
constraint or add a new persisted property against rule 6), but the gap should be **named**, not
left implicit — right now §11 doesn't mention it, and a reader could reasonably conclude from §3.3's
"visible marker" language that this closes the RCA's user-facing symptom. It only closes the
engineer-diagnosability half.
**Suggested fix:** add a bullet to §11 stating plainly that for a non-debug run the only signal is
the process log, and that this plan does not change what a demo presenter or the web UI shows — the
demo-visible symptom (`ws:acme`'s "done, no reply") is unchanged by this plan even once the rollout
caveat in §8 is resolved. If that's an acceptable scope boundary (which is defensible — recovery
was explicitly rejected in §3.3 for good reasons), say so explicitly so nobody downstream conflates
"K-027 item 2 delivered" with "the demo is now fixed."

**M2 — Two real drive-time behaviors the plan itself designs and reasons about are absent from the
10-test list (§10).**
1. The `& granted_set` intersection in `_missing_required_tools`'s input (§3.2) is explicit,
   load-bearing defense-in-depth against a hand-crafted graph write that bypasses the publish-time
   check — but no test in the list constructs a `requiredTools` entry that names a tool absent from
   `config.tools` at drive time and asserts it's silently dropped from `required` (no violation, no
   crash). Given the plan spends a full paragraph justifying this behavior, it should be pinned, not
   left to accident.
2. K-039's implicit-dispatch attempt (`executor.py:678`) is gated on `result.text` being truthy —
   `if "post_message" in granted_set and result.text and not emissions:`. A `post_message`-required
   node whose model ends the turn with **empty** text never even attempts the implicit dispatch, so
   the new required-tools check is the *sole* defense on that path — a materially different code
   path from test 4 (`…_whose_own_implicit_dispatch_declines_still_logs_a_violation`, where dispatch
   is attempted and declines). None of tests 1-2, 4 exercises an empty-text ending; none should be
   assumed to cover it by proxy.
**Suggested fix:** add two tests to §10 (11 and 12, or fold into the existing numbering) for these
two cases before implementation starts, so the offline-suite-green claim in §10's closing paragraph
actually covers what §3.2 designs.

### Minor

**m1 — "Both existing exit points" is one exit point narrower than `_run_agent_node`'s full
obligation surface.** `_execute_step`'s no-wired-LLM stub branch (`executor.py:517-518`,
`if not self._models.has_chat(): return StepResult(output="", on="done")`) returns *before*
`_run_agent_node` is ever called — a `requiredTools`-declaring node driven with no chat LLM
configured (`ModelGateway.has_chat() == False`, independent of the `FALKORCHAT_ENABLE_AGENT` flag
gate) silently skips the entire contract: no log, no trace, nothing. This is narrow in practice
(nothing meaningful happens on any agent node without an LLM, and no shipped def or existing test
combines `requiredTools` with a no-LLM executor), but "both existing exit points" (§3.2/§9) reads as
exhaustive when it isn't quite. **Suggested fix:** one sentence in §3.2 or §7 naming this third,
earlier return as out of scope (deliberately, matching the stub's own documented "deliberate, not a
fall-through accident" rationale at `_execute_step`'s docstring), so a future reader doesn't have to
rediscover it.

**m2 — Publish-invariant insertion order within the "deliberately LAST" block is unspecified.**
`services._validate_def_spec`'s docstring states "Running them last is load-bearing: an older check
must keep failing for its own reason, so a new invariant can never mask... a pre-existing one" —
good discipline the codebase already follows for the `waitsForHuman` → `validate_cmp` →
zero-transitions sequence (`services.py:944-982`). §9 step 2 says to add the fourth invariant "in
the same section as the existing `waitsForHuman` loop" but doesn't say before/after the
zero-transitions check. In practice the four checks are independent (none can currently mask
another, since they inspect disjoint failure surfaces), so this doesn't change behavior — but the
section's own stated rule is exactly the kind of thing worth being explicit about in a plan that
otherwise mirrors this codebase's conventions carefully everywhere else. **Suggested fix:** one line
specifying where the new loop lands relative to the other three.

## What's solid

- **Grounding is excellent.** Every line-range citation I checked (`_run_agent_node` at 590,
  `_handle_tool_call` at 734, the K-039 fallback at 677-713, the maxIterations-exhaustion path at
  724-732, `_trace_step`'s verbatim forward at 978-982, `_link_emissions`'s warning at 1005-1009,
  `services.py`'s `waitsForHuman` loop at 944-954, `PostMessageTool.run`'s no-thread-bound and
  `UnknownMemberError`-swallowing shapes at `tools.py:210-238`, `_str_list`'s existing defensive
  coercion, `TraceEvent`'s `traceId`-only indexing in `bootstrap_schema.sh`) matched the current
  source exactly, including the recomputed `_drive_loop` SHA-lock (`71055f756280`, reproduced
  independently with the documented `awk`/`sha256sum` recipe).
- **The mechanism is proportionate.** One new opaque config key, one small pure helper, two call
  sites threading one new parameter, one publish-time invariant mirroring an existing one almost
  verbatim. No new node/index/property, no `_drive_loop`/`_select_transition`/`_trace_step` change —
  verified true by reading the actual code, not just the plan's assertion.
- **Rejected alternatives (§3.3) are genuinely reasoned**, not waved away: "fail the run" is
  correctly identified as regressing K-039's own reason for existing; "park" correctly has no signal
  to wait for; "corrective retry" is deferred with a concrete, falsifiable reason (the RCA's own
  evidence that the shipped model already ignores an equivalent system-prompt instruction) and left
  explicitly composable on top rather than foreclosed.
- **§4's relationship to K-039 is genuinely non-duplicative** — orthogonal trigger conditions
  (unconditional grant vs. opt-in declaration), and it correctly identifies K-039's real residual gap
  (a declining implicit dispatch that produces zero signal today) as something only this design's
  `emissions`-based check closes.
- **The `emissions`-vs-`satisfied` split for `post_message` (§3.2)** is the right call and is backed
  by a concretely verified fact: `PostMessageTool.run` really does return an error string without
  raising in two cases (no thread bound; `UnknownMemberError` caught and converted internally) —
  confirmed by reading `tools.py:210-238` directly.
- **§8's K-034 rollout caveat is transparent and correctly sourced** (verified against
  `docs/plans/workflow-republish-semantics.md` §0/§1 and the cited `test_api.py` test, which exists
  under that exact name) — it declines to opportunistically bundle a `triage@v1` version bump, for a
  stated, sound reason (K-037's coupling).

## Open questions

- **M1's scope boundary is a call for the plan's stakeholder, not this review.** Is "diagnosable to
  an engineer with log access" an acceptable interim bar for K-027 item 2, with actual demo-visible
  delivery reliability left to K-039 (shipped) + K-027 items 3-5 (calibration, still open)? Or should
  this plan (or a fast follow-up) also make the violation visible in the run's own queryable state
  regardless of debug-run status? The plan doesn't ask this question of itself; I'm surfacing it
  rather than answering it on the plan's behalf.
- **Is the §11 rollout caveat (wipe-and-reseed vs. version bump for `ws:acme`) something `teco`/the
  user should decide before this lands**, given the plan explicitly declines to resolve it? Combined
  with M1 above, the practical effect of merging this plan without a rollout decision is that
  `ws:acme` keeps behaving exactly as the RCA described, indefinitely, until someone separately acts
  on §8.
