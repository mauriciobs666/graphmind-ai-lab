# `salesperson` K-058 dispatch-time write guard — implementation review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-058 (post-M6)

## Scope & verdict

Diff-scoped review of commit `67752add9fd1fb98991cba48f53458ed67a3aea6` ("feat(falkor-chat):
K-058 — dispatch-time guard holds off-turn write-mutating tool calls"), touching
`server/falkorchat/executor.py` (+114) and `server/tests/test_executor_agent.py` (+240); the
same commit's `docs/HISTORY.md`/`docs/BACKLOG.md` edits are read for context only, per the
brief (their accuracy is `teco`'s check, not this review's). Baseline: `docs/reviews/
salesperson-tool-reliability-ml.md` §9.2 (confirmed repro)/§9.4 (the candidate spec this diff
claims to implement) and `docs/reviews/salesperson-tool-reliability-impl.md` (precedent for
what NOT to do on this same defect family — the ruled-out dedup-by-signature approach and the
K-056 breadcrumb-imitation mistake). Not in scope: K-057/Track A (`proof_defs.py`, a separate,
concurrently in-flight unit — see the operational note in the Appendix); any new live-model
regression run (the commit's own n=20 live check is taken as reported evidence, cross-checked
for mechanism plausibility, not independently re-run here — no live LM Studio access from this
review).

**Verdict: approve.** No blocker. The guard is a faithful, correctly-scoped implementation of
§9.4's spec — dispatch-time text-presence check against the current turn's own text, not the
explicitly-ruled-out cross-turn dedup-by-signature — the fail-open path is genuinely confined to
the offline-stub code path (verified, not assumed), the scope boundary (`add_to_cart`/
`remove_from_cart` in, `place_order`/`clear_cart`/`save_profile` out) holds up against the real
tool schemas, and the new observability signal is a legitimate, non-overengineered distinction
from `_note_possible_fabrication`. Tests assert real dispatch behavior (`reg.dispatched`), not
just trace/log lines. I ran the full offline suite against the exact commit, isolated from a
concurrent unit's uncommitted changes (see Appendix): **2296 passed, 14 deselected** — matches
the commit message's claim exactly. Two MINOR findings and one residual-risk note below; none
block.

**CPG:** considered, not relevant — `cpg_falkorchat` was confirmed stale at coordination-dispatch
time (built 2026-08-26, many relevant commits since) and this is a diff-scoped correctness/test
review of new, small, non-structural code (one new module-level dict, one new pure function, one
new observability method, two call-site signature changes with only two call sites total,
grep-verified) — no structural-impact question here a CPG would answer that direct reading
didn't already settle.

## Findings

### MINOR 1 — commit message/`HISTORY.md` claim "7 new unit tests"; the diff adds 6

**Evidence:** `git diff 67752ad^..67752ad -- server/tests/test_executor_agent.py` adds exactly
six `def test_` functions (`test_off_turn_write_is_held_not_dispatched_on_confirmed_repro_shape`,
`test_legitimate_repeat_mentioned_in_turn_text_still_dispatches`,
`test_target_mention_matching_is_case_and_whitespace_insensitive`,
`test_off_turn_remove_from_cart_is_also_held`,
`test_write_tool_with_no_resolved_target_argument_is_unaffected`,
`test_no_thread_context_available_does_not_hold_write_calls`) — confirmed via
`grep -c '^+def test_'` on the diff and via `pytest --collect-only`. `HISTORY.md`'s 2026-08-30
K-058 entry (and the commit message) say "7 new unit tests" and enumerate exactly six behaviors
in the same sentence — an off-by-one in the count, not a missing behavior.

**Why it matters:** cosmetic — the six tests genuinely cover the six named behaviors (verified
below), so nothing is under-tested. Flagged because the brief's own premise repeated the "7"
figure; worth a one-word correction wherever this entry is next touched so the number doesn't
propagate.

**Suggested improvement:** `HISTORY.md`'s K-058 entry: "7 new unit tests" → "6 new unit tests" (a
trivial doc fix in the next unit that touches this entry; not worth a dedicated edit today).

### MINOR 2 — the guard's turn-text surface includes the model's own not-yet-persisted reply text, which is a self-reported (not independently verified) grounding signal

**Evidence:** `_run_agent_node`, `executor.py:896-899`: `turn_text = f"{trigger_text}
{last_text}"`, where `last_text` is updated from `result.text` on *every* iteration that carries
any text (`executor.py:814-815`), including an iteration that emits both text and a tool call in
the same turn. So the guard's "is the target mentioned in this turn's own text" check trusts
whatever the model itself just said in this same turn, not only the customer's own trigger
message. This is exactly what §9.4 specifies ("the current turn's own trigger/reply text") — not
a deviation — but it is a real, if narrow, residual gap: a model that pads a sentence naming the
off-turn product before firing the duplicate call would satisfy the check trivially.

**Why it matters:** low — the confirmed §9.2/§9.4 repro mechanism is a call with *no* accompanying
narration ("the model's second turn does one correct, newly-requested write, then spontaneously
repeats an unrelated, already-completed one" — ml note §9.2, no text mentioned), so this gap is
theoretical against the evidence gathered so far, and it is inherited from the approved spec, not
introduced by this implementation. Distinct from K-056's breadcrumb-imitation failure (that one
fed a *fabricated* tag back into *future* turns' replayed history; this one only affects
*this-turn* self-grounding and is never persisted — see What's Solid).

**Suggested improvement:** no action needed now; worth a one-line note in `_target_mentioned_
in_turn_text`'s docstring (or `_WRITE_TARGET_ARG`'s comment) naming this as a known, accepted
limitation of the current mechanism, so a future incident report on a model that "talks its way
past" the guard isn't mistaken for a fresh defect.

## Residual risk, not a finding — `place_order` duplicate-dispatch is unaddressed by this guard, by design, and should stay visible

`place_order`/`clear_cart` take zero arguments (`tools.py:672-679` /`:706-717`, verified directly
against the real schemas, not just the test stubs) — there is no resolved target argument for
`_WRITE_TARGET_ARG` to check, so this guard structurally cannot cover a duplicate `place_order`
re-fire. The ml note's own §9.3 already named this: `place_order`'s idempotency guard mints a
fresh `order_id` per call, so it does not protect against two independently-decided dispatches.
§9's `place-order-retrigger` condition found 0/4 at a sample too small to support any claim. This
diff's exclusion is correct and faithful to the spec (not an oversight — the code comment at
`executor.py:42-46` states it plainly), but now that K-058 reads "resolved" in `HISTORY.md`, this
specific named gap is one step further from visibility. Not a blocker for this diff; worth the
coordination keeping it on `docs/BACKLOG.md` rather than letting it quietly close with K-058.

## What's solid

- **Correctly targets §9.4's spec, not the ruled-out dedup-by-signature fix.** The guard checks
  text-presence of *this turn's own* trigger/reply text against the call's resolved target
  (`_target_mentioned_in_turn_text`, `executor.py:53-63`) — it holds nothing based on whether the
  same `(tool, args)` pair fired earlier in the run. `test_legitimate_repeat_mentioned_in_turn_
  text_still_dispatches` explicitly pins the case the ruled-out approach would have wrongly
  blocked (a genuine later repeat, named in its own turn's text) and confirms it still dispatches.
- **No K-056-style breadcrumb-imitation risk.** I traced whether the `{"held": true, ...}` tool
  response could leak into a *future* turn's replayed history the way the reverted K-056 tag did:
  it cannot — this JSON only ever enters the current node execution's own `messages` list
  (`_handle_tool_call`'s return value, appended as a `role: "tool"` entry), which is discarded
  once the node returns; `_read_thread_context`/`_assemble_messages` only ever replay persisted
  `Message` nodes, created solely via `post_message` dispatch, never via an intermediate tool-loop
  turn. A held call is invisible to every later turn's context by construction.
- **Fail-open path verified confined to the offline-stub code path, not a live gap.**
  `_read_thread_context` (`executor.py:1101-1105`) returns `[]` only when `threadId` is absent
  from `run_ctx` or `self._services is None`; `services.py:2004-2036` seeds `threadId` into
  `run_ctx` unconditionally for a `conversation`-kind def's `start_workflow_run`, which
  `salesperson` is — so a live conversation always has `thread_msgs` non-empty (at minimum, the
  triggering message itself) and never hits `_target_mentioned_in_turn_text`'s `return True`
  branch. Confirmed by reading the actual `read_thread` Cypher (`repository.py:701-720`, `m.text
  AS text` — the exact key `.get("text", "")` reads) rather than assuming field-name agreement.
- **Scope boundary verified against real schemas, not narrative.** `add_to_cart`/
  `remove_from_cart` both declare a single string `productName` (grep-confirmed in `tools.py`);
  `place_order`/`clear_cart` declare zero parameters; `save_profile` declares two independently
  optional free-form strings (`name`, `deliveryAddress`) with no single resolved target — the
  `_WRITE_TARGET_ARG` map and its in-code comment match the real tool surface exactly.
- **Only two call sites, both correctly updated** (grep-confirmed: `_handle_tool_call` is called
  from exactly the K-039 implicit-`post_message` fallback and the main tool-call loop, both now
  passing `step` and `turn_text=`), and no test calls the private method directly — all six new
  tests exercise it through `_run_agent_node`, matching the file's existing convention.
- **Tests pin dispatch behavior, not logging.** Every new test asserts `reg.dispatched` (what
  actually reached the tool registry), not merely the trace/log signal — I ran them in isolation
  (`pytest -k "..."`, 7 collected including one pre-existing false match on my own keyword filter,
  6 of which are this diff's) and they pass. `_note_off_turn_write_held` is a genuinely distinct
  signal from `_note_possible_fabrication` — the two name different failure classes (no grounding
  tool ran at all, vs. a real write about to run ungrounded) with different remediation
  implications, matching the same non-conflation discipline the ml note applies throughout.
- **Live regression evidence is mechanistically consistent, not just asserted.** n=20 exact-§9.2-
  shape conversations, 0/20 inflated carts, 1/20 a live re-attempt caught by the guard: a 5%
  single-attempt observation sits inside the wide but overlapping Wilson CIs the ml note's own
  §9.2 (1/24 pooled add-conditions, 4.2%) and §8.4 (3/10 conversation-level, 30%) established for
  this same underlying model tendency — the guard doesn't change the model's propensity to
  attempt the off-turn call, only whether it dispatches, so a low observed attempt rate with zero
  resulting inflation is exactly the shape this design predicts.

## Open questions

- Should `place_order`'s still-open duplicate-dispatch risk (see Residual risk above) get its own
  `docs/BACKLOG.md` line now, rather than relying on `docs/reviews/salesperson-tool-reliability-
  ml.md` §9.3 as its only record? Not this review's call — routes to whoever owns the K-058
  coordination close-out.

## Appendix — operational note: a shared-instance data-contamination incident, caused and fixed during this review, not a finding about the diff

Verifying the commit's own claimed offline-suite count required running the default (destructive)
`pytest` suite against the live `falkordb-dev` instance (`falkor-chat/AGENTS.md`'s documented
`reference`-wipe hazard). Doing so from the main working tree collided with a second, concurrent,
uncommitted in-flight unit (K-057/Track A, `server/falkorchat/proof_defs.py` bumped to `v5`
locally, not yet committed) via `./scripts/seed_salesperson.sh`'s **hardcoded default**
`SALESPERSON_DEF_VERSION="${FALKORCHAT_SALESPERSON_DEF_VERSION:-v4}"` (the script's own header
comment had already been updated for `v5` by the concurrent unit; its executable default was
not) — my restore-sequence re-seed published `salesperson@v4` into `reference` with `v5`'s
content (the K-057 `systemPrompt` addition), silently mislabeled. Detected via direct Cypher
(`s.config CONTAINS 'Do not tell the customer a filter'` returning `true` under the `v4` label),
root-caused by tracing the editable-install package resolution (`server/.venv`'s
`__editable__.falkorchat...pth` hardcodes an absolute path to the main tree's `falkorchat/`
package — a `git worktree` at the reviewed commit only isolates imports when Python is invoked
with **cwd inside the worktree's own `server/` directory**, since a local `./falkorchat` package
directory shadows the editable finder's absolute mapping; invoking from the repo root does not
shadow it). Fixed by `DETACH DELETE`-ing the mislabeled `WorkflowDef{key:'salesperson',
version:'v4'}` subgraph (`Step` nodes are version-scoped via `stepUid`, safe to delete without
touching `v1`/`v2`/`v2.1`/`v3`/`v5`) and re-publishing from a cwd-isolated worktree checkout of
the reviewed commit. `ws:acme`'s own `v4` snapshot was never touched by the pytest wipe (only
`reference` is wiped) and was independently confirmed clean throughout. Final state re-verified
`OK` on all three of `verify_workflows.sh acme` / `verify_catalog.sh` / `verify_salesperson.sh
acme` before finishing. No file under git version control was touched by this review; the
worktree was removed (`git worktree remove --force`) after use. Recorded here for the
coordination's own transparency, not as a finding against the K-058 diff itself.
