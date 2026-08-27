# `_assemble_messages` role-alternation fix — plan review (K-048)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-048 (M2.5-adjacent, post-M3 follow-up)

## Scope & verdict

Reviewed: `docs/plans/assemble-messages-alternation.md` (the `architect` plan, U1 in
`docs/plans/assemble-messages-alternation-coordination.md`) against the backlog entry it fixes
(`docs/BACKLOG.md` § K-048) and the current, unmodified production code
(`server/falkorchat/executor.py`, `_assemble_messages` at line 912, its call site in
`_run_agent_node` at line 703, and the SHA-locked `_drive_loop` at lines 451–512). This is the
plan-gate review (pre-implementation); a second, diff-scoped review follows `tdd-engineer`'s
implementation per the coordination doc. No code was run or written; every factual claim below
was independently re-derived (grep, direct reads, the SHA-lock recipe, the CPG) rather than taken
from the plan's prose.

**Verdict: approve with suggestions.** The root-cause framing is accurate, the chosen remedy is
sound and its own worked traces are correct, the blast-radius and SHA-lock claims all verify, and
the test strategy is adequate. Two minor documentation gaps and one low-cost test-strategy
upgrade are worth folding in before or during implementation; none blocks handoff to
`tdd-engineer`.

**CPG:** used `cpg_falkorchat` — independently re-ran the plan's own call-site query
(`MATCH (call:CALL) WHERE call.METHOD_FULL_NAME CONTAINS '_assemble_messages' RETURN
call.METHOD_FULL_NAME, call.CODE, call.LINE_NUMBER`) and got the identical single row the plan
cites (`WorkflowExecutor._assemble_messages`, call at line 703) — the impact surface is confirmed
closed by a source I ran myself, not only by the plan's stated result. Per the coordination doc's
stated freshness (built 2026-08-26T22:27:22Z; one commit since, `da10d57`/K-049, touches
`api.py`/`schemas.py`/`services.py`/tests only, not `executor.py`), the graph is current for this
review.

## Findings

### Minor — the docstring drift the task flagged is real, and the plan's own snippets make it easy to lose

`server/falkorchat/executor.py:917-921`'s existing docstring on `_assemble_messages` ("Build the
node's opening message list… Thread turns come pre-capped by `_read_thread_context`; an empty
list (offline stub path) leaves only system+CONTEXT.") says nothing about coalescing and will be
factually incomplete once the fix lands — it should describe the no-two-consecutive-same-role
invariant `_append_turn` now enforces. The plan's own §3 replacement code block for
`_assemble_messages` omits the docstring entirely (mirroring the elision in its §2 "the function
today" quote), so an implementer who copies the snippet literally will silently drop it rather
than update it — a real risk given how documentation-heavy this file's other methods are (see the
extensive docstrings on `_run_agent_node`, `_handle_tool_call`, `_note_must_post_violation`
alongside it).

**Suggested improvement:** add one sentence to the plan's §3 (or leave it for `tdd-engineer` to
apply as an explicit step) instructing that the existing docstring be *extended*, not replaced:
note that thread-turn/CONTEXT appends now route through `_append_turn`, which merges into the
previous message instead of adding a new one when that would break user/assistant alternation
(K-048), and that this makes the function's output a no-op on the already-alternating case (the
same guarantee `_append_turn`'s own docstring already documents in detail — it just isn't
mirrored one level up).

### Minor — §1's "already alternation-safe" claim for the tool-call loop points to §3, which never substantiates it

§1 (Out of scope) says the tool-call loop body in `_run_agent_node` is untouched because "its own
appends are already alternation-safe — see §3," but §3 ("Design & rationale") only discusses
`_assemble_messages`'s three traced examples and never mentions the tool-call loop at all. I
independently verified the underlying claim is true: `_run_agent_node`'s loop (`executor.py:
767-775`) always appends exactly one `_assistant_turn` (role `assistant`) followed by one or more
`role: "tool"` messages per iteration — the crash template's own error text ("...must alternate
user and assistant roles **except for tool calls and results**") exempts this shape, and no two
consecutive `assistant` appends are structurally possible since a tool-role message always
intervenes. So the claim holds, but the plan misdirects the reader to a section that doesn't
contain the promised evidence.

**Suggested improvement:** either fix the cross-reference (drop "— see §3" or point it at a new
short paragraph), or add 2-3 sentences to §3 stating the reasoning above so the "out of scope"
claim is self-contained.

### Minor — the "sibling shape" test is marked recommended, not mandatory, despite already being latent in an existing fixture

§4 step 4 and §6 both correctly flag the sibling shape (two-plus consecutive same-role thread
turns before `CONTEXT`) as "inferred, not live-verified" and file it as a cheap, optional test.
That characterization slightly understates how present the shape already is in this suite:
`test_executor_agent.py:454-460`'s `_thread_rows(n)` fixture generates `n` **consecutive
`role: "user"` rows**, and it is already used with `n=8` and `n=3`
(`test_agent_node_carries_its_thread_window_out_on_the_step_result`,
`test_the_thread_window_also_rides_out_when_max_iterations_are_exhausted`) to exercise
`_run_agent_node` today — those tests just never inspect `llm.calls[0]["messages"]`, so they
wouldn't catch a coalescing regression. The shape isn't merely a plausible-but-unconfirmed
production scenario; it's the literal fixture shape the suite already builds for other purposes.
Given the fixture support is already there, step 4's test costs one assertion block, not new
fixture work.

**Suggested improvement:** promote §4 step 4 from "recommended" to a required step in the TDD
sequence (or note explicitly why it stays optional despite the fixture already existing) — this
is a low-cost, low-risk upgrade to the test strategy, not a correctness concern (the algorithmic
guarantee already covers the shape regardless of whether a test pins it).

## What's solid

- **Root-cause framing matches the backlog exactly** — the unconditional trailing `user`-role
  `CONTEXT` append after role-mapped thread turns, and the structural reasons `intake`'s first
  call and every `research`→`answer` handoff hit it (`research` never posts, so `answer` always
  sees a `user`-terminated thread). Confirmed against the live code and the backlog's own
  evidence trail (`docs/plans/ministral-reprobe-ml.md` §4.2, `docs/reviews/ministral-reprobe.md`).
- **All three worked traces in §3 check out.** I re-derived each independently against
  `_append_turn`'s stated logic (`if messages and messages[-1]["role"] == role: merge else:
  append`): thread-ending-in-`assistant` produces zero merges and the identical 4-message,
  4-role shape as today (byte-for-byte no-op, as claimed); thread-ending-in-`user` merges exactly
  once into the 2-message `[system, user]` shape claimed; the two-consecutive-`user` sibling case
  merges twice into the single coalesced `user` message claimed. All three match the plan's
  stated outputs exactly.
- **Blast-radius claims verify.** `grep -n 'messages\[' server/falkorchat/executor.py` and
  `grep -n 'len(.*messages' server/tests/test_executor_agent.py` both return zero hits, exactly
  as the plan states — no index- or count-dependent code exists anywhere the returned list is
  consumed. I additionally traced every existing test that drives `_run_agent_node` with
  thread rows (`test_agent_node_folds_thread_messages_into_prompt`,
  `test_agent_node_carries_its_thread_window_out_on_the_step_result`, and its
  `maxIterations`-exhaustion sibling) and confirmed none would break under the new coalescing:
  the one test that does inspect `llm.calls[0]["messages"]` content
  (`test_agent_node_folds_thread_messages_into_prompt`) uses a thread ending in `assistant` — the
  proven no-op path.
- **SHA-lock claim independently reproduced.** Re-ran the exact awk/sed/sha256sum recipe from
  `docs/DESIGN.md` §9 against the current tree myself (not copied from the plan) and got the same
  `71055f756280`, with the `def _drive_loop`/`# ── seams` markers landing at lines 451/512 —
  `_assemble_messages` (912) and `_run_agent_node` (618, its call to `_assemble_messages` at 703)
  sit hundreds of lines outside that span.
- **The candidate comparison in §3 is honest and correctly reasoned** — candidate 1 (CONTEXT-tail
  only) is rejected for leaving the sibling shape exposed at no implementation savings; candidate
  2 (unconditional full collapse) is rejected for touching the already-safe path for no benefit.
  The chosen general helper is provably a no-op on an already-alternating sequence (the merge
  guard `messages[-1]["role"] == role` can only fire when it wasn't going to alternate anyway),
  which is the right proportionality call.
- **Live-Ministral deferral is the right call, not merely cost-driven.** The offline shape-pinning
  tests (§4 steps 1-2) prove an *algorithmic* invariant — no two consecutive same-role turns can
  occur in the output — which is a strictly stronger guarantee than any one live template replay
  could offer: any strict-alternation validator by construction accepts a sequence with that
  property. Reusing the existing live triage test against Qwen (§4 step 6) is adequate for its
  actually-stated purpose (model *behavior* quality on the merged content, not the alternation
  crash itself), and both structural trigger shapes (`intake`'s first call, `research`→`answer`)
  are exercised by that one test (confirmed by reading its AC-1…AC-4 assertions directly).
- **Merge-separator and sibling-shape open questions are correctly scoped as low-risk, not
  blocking** — I agree with the plan's own categorization in §6. The `"\n\n"` separator affects
  only prompt readability/formatting for the model, not correctness, and is trivially revisited;
  the sibling shape is closed algorithmically by the same helper regardless of whether it is ever
  live-verified (see the previous point on the offline tests' strength).
- No security/performance surface — pure in-process string/list assembly, no new I/O, no
  injection vector, and the signature stays unchanged (`@staticmethod`, same params/return
  shape), so the one call site needs no changes beyond what the plan already shows.

## Open questions

None that block proceeding to `tdd-engineer`. The three findings above are folding-in-passing
suggestions the implementer or a plan revision can pick up; none requires the caller's input to
resolve.

## Pass 2 (2026-08-26) — diff-scoped review of `tdd-engineer`'s implementation

**Scope:** the uncommitted working-tree diff (not yet a commit) — `git diff -- \
server/falkorchat/executor.py server/tests/test_executor_agent.py docs/HISTORY.md \
docs/BACKLOG.md`, read directly rather than taken from the coordinator's or `tdd-engineer`'s
description. I re-ran the full offline suite myself (`.venv/bin/python -m pytest -q`), re-ran the
SHA-lock recipe against the modified tree, and reconstructed the pre-fix behavior from `git show
HEAD:falkor-chat/server/falkorchat/executor.py` in a scratch script to independently confirm the
two crash/sibling tests are not tautological — I did not re-run the live pass (not required, and
it needs a running LM Studio this sandbox doesn't have).

**Verdict: approve.**

### Disposition of Pass 1 findings

1. Docstring drift (minor) — **fixed.** `executor.py`'s `_assemble_messages` docstring now states
   every `user`/`assistant` append routes through `_append_turn`, describing both the
   no-op-when-already-alternating guarantee and the merge-on-collision behavior (diff hunk at
   `_assemble_messages`, confirmed by direct read of the new docstring text).
2. §1→§3 cross-reference nit (minor) — **not fixed, correctly so.** Plan prose only, no code
   implication; left alone per the coordinator's note.
3. Sibling-shape test promoted to mandatory (minor) — **fixed.**
   `test_assemble_messages_coalesces_consecutive_same_role_thread_turns` now exists
   unconditionally (not gated as "recommended"), reuses `_thread_rows(2)` exactly as suggested.

### New findings

None. The diff is a clean, minimal implementation of the approved plan with no unreviewed
surface: `_append_turn` is byte-identical to the plan's §3 text (including its docstring), placed
immediately before `_assistant_turn` as specified, and `_assemble_messages` routes both the
thread-turn loop and the trailing `CONTEXT` append through it, with nothing else in the file
touched (confirmed by the diff's hunk boundaries — no changes anywhere near `_run_agent_node`'s
tool-call loop, lines 767/773, which stays on raw `.append()` calls as scoped).

### Checks against the coordinator's four questions

1. **Diff matches Pass 1's predicted traces:** yes. `_append_turn` sits directly above
   `_assistant_turn`; both append sites in `_assemble_messages` route through it; the docstring
   is extended, not left stale.
2. **New tests are genuine, not tautological.** Reconstructed the pre-fix `_assemble_messages`
   from `git show HEAD:...` in a standalone script and ran the same inputs the new tests use:
   the crash-shape test's input (`_thread_rows(1)`) produces pre-fix roles
   `["system","user","user"]` (test asserts `["system","user"]` — genuinely fails pre-fix); the
   sibling-shape test's input (`_thread_rows(2)`) produces pre-fix roles
   `["system","user","user","user"]` (same assertion — genuinely fails pre-fix); the
   characterization test's input reproduces `["system","user","assistant","user"]` pre-fix with
   an unmerged, byte-exact `CONTEXT:\n{...}` tail — passes pre-fix as a true characterization,
   not a red test. All three assert exactly what the plan's §3 traces predicted.
3. **`tdd-engineer`'s reported numbers are internally consistent.** I ran
   `.venv/bin/python -m pytest -q` myself on the current tree: **1785 passed, 4 deselected** —
   matches the report exactly, and matches the claimed 1782-baseline-plus-3-new-tests arithmetic
   (verified the 3 new tests independently via `-k test_assemble_messages`: 3 passed). Re-ran the
   `docs/DESIGN.md` §9 SHA-lock recipe against the modified tree myself: `71055f756280`,
   unchanged, matching the report and my own Pass 1 baseline. The described mutation (disabling
   the merge branch) is equivalent in effect to the pre-fix code I reconstructed independently
   for the two sequences that matter, so the reported "fails for the right reason, characterization
   still passes" is directionally corroborated by a source I generated myself, not merely
   asserted. I did not reproduce the live pass (no LM Studio available here) — its 1-passed claim
   is taken as reported, consistent with the plan's own scoping of that step as a one-time,
   non-standing check.
4. **`HISTORY.md`/`BACKLOG.md` are accurate.** The new `HISTORY.md` entry cites both
   `docs/plans/assemble-messages-alternation.md` and this review by path, states the verdict
   (approve with suggestions) and which suggestions were folded in, and its suite/SHA-lock figures
   match what I independently reproduced — no fabricated numbers found. `BACKLOG.md`'s K-048
   section is removed in full (not left as an index row), matching the
   delivered-item-doesn't-stay-in-BACKLOG convention; nothing else in the deferred section below
   it was disturbed.
