# `model-bench` S5 — tool-caller pack, part 1: code gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (S5)

## Scope & verdict

Reviewed the full S5 diff, `git diff e3f7220..HEAD -- model-bench/` (17 files, +4278/-31, six
commits U139-U143 per `docs/plans/small-model-benchmarking-coordination.md`), against
`docs/plans/small-model-benchmarking-s5-spec.md` (the governing contract) and
`docs/plans/small-model-benchmarking-ml.md` §4.2/§4.3/§11.4 (the counting/denominator rules the
spec defers to). This covers the tool-caller-shop-assistant pack, `modelbench/scoring/toolcalls.py`,
`modelbench/results.py`'s three `ToolCallAggregates` extensions, `report.py`'s three new renderers,
`runner.py`'s `_load_conversation_scorer`, and their tests. It does not re-litigate any finding
already recorded as closed in the U139-U143 ledger rows; it focuses on cross-unit integration,
denominator exactness, and a targeted mutation probe distinct from the five already run.

**Verdict: needs changes.** Two real, reproduced defects in `modelbench/scoring/toolcalls.py`'s
`score_conversations` assembly (Step 5) survive all five coordinator mutation probes and the full
suite; both are silent — no crash, wrong number in the shipped report. One further gap (an entire
required `-ml` §4.2(d) reporting facet, computed but never wired to anything printable) needs an
explicit scope decision, not necessarily code in this stage. Nothing here touches the plan's own
Steps 0-4 (already covered exhaustively by U139-U141's own review + mutation history, re-verified
below only at the seams).

**Suite/lint, reproduced myself**: `.venv/bin/python -m pytest -q` → `1597 passed, 3 deselected`;
`.venv/bin/ruff check .` → clean. Both match the ledger's claim exactly.

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(confirmed via the brief and consistent with the S5 spec's own §1 CPG line); this is a code-level
task in a component with no CPG.

## Findings

### Blocker — a restraint turn that never replied (`timed-out`/`cap-hit`) scores as a `restraint` *success*, contradicting the same turn's own `fail` verdict everywhere else

`_score_one_conversation` (`modelbench/scoring/toolcalls.py:757-764`) computes the standalone
`restraint` `BinaryMetric`'s success flag from `restraint(dispatched_count)` alone — dispatch count
only, no disposition check — while the *same* turn's `turn_clean` entry (feeding
`cleanThroughTurnH`/the hazard curve) is correctly gated on `disposition == "replied"` (the U142
fix): `turn_clean.append(clean and t_turn.turnDisposition == "replied")`. The two consumers of one
turn disagree. Reproduced directly:

```
script = make_script("A-01", 1)  # one restraint turn (toolRequired: False)
turn = make_turn(disposition="timed-out", dispatches=(), final_reply=None)
items, aggregates = score_conversations([(script, make_trace("A-01", (turn,)), ())], [], pack=make_pack(h=1))
# items[0].outcome == "fail"                       <- headline: this turn failed
# aggregates.restraint == BinaryMetric(successes=1, n=1, ...)   <- restraint rate: 100%, a "success"
```

`-ml` §4.3 rule 4's table states, in terms this module quotes verbatim in its own docstring
(`toolcalls.py:112-118`): *"timed-out ... outcome fail, never n_a ... the only non-completion that
earns it."* A turn the harness never got a reply from cannot simultaneously be a restraint
*success* — the model may well have been about to call a spurious tool, and the harness has no
observation either way (the same laundering the note spends a full paragraph forbidding at §4.3
rule 4's closing clause: *"a model that hangs out-scores one that answers wrongly"*). This is
directly reader-visible: `restraint` reaches `report.py`'s generic "## Arms" table via
`named_metrics()` unchanged (§2.5's own carve-out only excludes `hazard`/`funnelCounts`/
`perTurnPosition`, not `restraint`), so a model whose restraint turns mostly time out would print a
misleadingly high restraint rate in the shipped comparison report.

No existing test (all five coordinator probes, U139-U143, plus the U142/U143-added
`test_score_conversations_restraint_turn_that_never_replied_is_still_not_clean` parametrized over
`cap-hit`/`timed-out`) checks `aggregates.restraint` on this fixture — that test only asserts
`items[0].outcome`/`.counts`, never the standalone rate. This is exactly the "two independently
fixed defects combined" seam the brief asked for: U142's disposition-gate fix landed on
`turn_clean` only, not on the sibling `restraintSuccesses` tally three lines above it in the same
function.

**Fix**: gate `tally.restraintSuccesses` the same way `turn_clean` already is —
`clean = restraint(dispatched_count) and t_turn.turnDisposition == "replied"` — and add a test
pairing a restraint turn with each of `cap-hit`/`timed-out` asserting `aggregates.restraint.successes
== 0` (mirroring the existing parametrized headline test, in the same file).

### Major — `iterationSummary`'s `yCalls` nets out the one non-returning call per failing turn, reproducing the exact "netting" `-ml` §11.4 rules out by name

`_iteration_summary_dict` (`toolcalls.py:883-898`) computes `"yCalls": sum(iterations for _, iterations in observations)`
— i.e. `Σ` completed-call counts only. `-ml` §4.2(f) point 4 names this figure's job explicitly
("the cost question — how many model calls did this run actually make") and binds it to §11.4's own
formula: *"`a_i = callCount_i + [D(t_i) ∈ {timed-out, no-response, server-rejected}]` per item"* —
completed calls **plus one** for the turn's own non-returning attempt, since a turn's loop stops on
the first call that raises and therefore has *at most one* uncounted attempt. The shipped `yCalls`
omits that `+1` term entirely. Reproduced directly (a turn that completed 2 calls then timed out on
its 3rd — a real, reachable shape: `convo.py`'s `_drive_turn` keeps `chat_results` from completed
iterations even when a later iteration in the same turn raises, `convo.py:722-741`):

```
turn = make_turn(disposition="timed-out", iterations=2, final_reply=None)
# aggregates.iterationSummary["yCalls"] == 2   (shipped)
# -ml §11.4's own formula:            == 2 + 1 == 3
```

§11.4 dedicates a long, explicit subsection to rejecting exactly this shape of undercount ("the
netting is real, it is not intended, and it does not survive" — three numbered consequences,
including "a netted base cannot refuse" and "the invariant the report is written against
inverts"). The existing test for this field
(`test_score_conversations_populates_iteration_summary_restricted_and_unrestricted`,
`test_scoring_toolcalls.py:1240-1286`) bakes the gap in rather than catching it: its one
non-`replied`/`cap-hit` fixture is a `no-response` turn constructed with `iterations=0`
(`make_turn(disposition="no-response", iterations=0, ...)`), so `+0` and `+1` are indistinguishable
in that fixture — the test cannot tell a correct implementation from this one. A run where every
turn fails outright (`Y_calls == 0`) would print an unrestricted "calls per turn" of exactly `0.00`,
which §11.4 calls out by name as the reachable, wrong-denominator failure mode this design exists to
prevent (*"`Y_calls == 0` is reachable ... §11.6's integer gate ... is vacuously true"* — the same
principle applies one level down to this report-only sentence, even without a refusal gate of its
own).

**Fix**: `yCalls = sum(iterations) + count of turns whose disposition is in {timed-out, no-response,
server-rejected}` (mirroring §11.4's own `a_i` formula, using this module's own
`_UNRUNNABLE_DISPOSITIONS ∪ {"timed-out"}`, i.e. every disposition outside `{replied, cap-hit}` —
which is exactly `ITERATION_SUMMARY_EXCLUDED`, already declared in this file). Add a test with a
non-zero-`iterations` `timed-out`/`no-response` turn (the exact fixture reproduced above) asserting
`yCalls` includes the `+1`.

### Major — `-ml` §4.2(d)'s required failure decomposition (`omitted_required` / `wrong_value` / `boundary_unit`) is computed and unit-tested but never reaches `FunnelCounts`, `ToolCallAggregates`, or any renderer

`argument_correctness` (`toolcalls.py:276-312`) returns a full `ArgumentCorrectness` with
`omittedRequired`/`wrongValue`/`boundaryUnit` tuples, each correctly populated and tested in
isolation (Step 3). But `_score_one_conversation` (`toolcalls.py:788-795`) reads only
`correctness.allCorrect` to decide `argsCorrectSuccesses`/`args_all_correct` — the three failure
tuples are discarded on every call, every turn, every conversation. Confirmed by exhaustive grep:
`omittedRequired`/`wrongValue`/`boundaryUnit` appear nowhere outside `toolcalls.py`'s own
`ArgumentCorrectness` definition — no `FunnelCounts` field, no `ToolCallAggregates` field, no
`report.py` renderer, no test asserting any of the three ever reaches a stored or printed value.

`-ml` §4.2(d) is explicit that the headline (`all_args_correct`) is not the whole requirement: *"The
requirement's 'split three ways' is not three disjoint buckets on equal footing... Failure
decomposition, per argument not per call: `omitted_required` / `wrong_value` / `boundary_unit`...
Report it as `wrong_value: 12, of which boundary/unit: 7`."* As shipped, the tool-caller report can
never print that line — the whole reason §3.3's `boundaryRule` schema extension and the
`_boundary_confused` classifier exist (to let a reader see how much of a wrong-value miss is a
boundary/unit confusion rather than a genuine miss) has no path to the page.

This is arguably a **spec gap** rather than a pure implementation miss: the S5 spec's own §3.1
`results.py` code block allocates no field for this decomposition, and §4.4's file/layout table
names exactly three renderers (funnel, per-turn-position, hazard) plus the `I(t)` sentence and E5 —
never a fourth for this. So the pure function was built to spec (§3.4's own bullet: "one function
per §4.2 letter... each returning the numerator/denominator pair(s) it owns"), but nothing in the
spec's own downstream steps ever asks for the return value to be wired anywhere. Flagging as major
rather than blocker because of that ambiguity, but it needs an explicit resolution before stage
close — either wire it now (a `FunnelCounts`-adjacent tally + a short renderer line, the same shape
as the funnel's other "-> k/n" annotations) or record the deferral explicitly (with a reversal
trigger, per this component's own honesty convention) rather than leaving a silently unfulfilled
`-ml` requirement.

### Minor — `runner.py`'s module docstring is stale about both scorer seams

`runner.py:1-36`'s own module docstring (predating S5, last touched at S2/S3) still reads *"Inert
today: `_load_item_scorer`/`_load_conversation_scorer` raise `NotImplementedError` unconditionally
(no scorer ships before S3)"* and *"nothing here calls a real pack's scorer."* Both scorers are now
real (`_load_item_scorer` since S3, `_load_conversation_scorer` as of this stage,
`runner.py:245-262`), and `_drive_conversations` now calls a real `ConversationScorer` for a real
`tool-caller` pack end to end (the Step 6 integration test proves it). Not a functional defect, but
exactly the kind of drifted context this component's own documentation convention (`AGENTS.md`)
asks to be rewritten rather than left stale — a future reader of this module's own docstring would
be told something false about the very code the docstring sits above. **Fix**: strike the "Inert
today" sentence from gap 3's writeup and the closing "nothing here calls a real pack's scorer"
sentence; a one-line note that both seams are now live is enough.

### Observation, not a finding — `_structural_n`'s constant-ceiling approximation is a disclosed, self-limiting simplification

`report.py:684-706`'s `_structural_n` returns `len(run.items)` (a flat 12) rather than the real
pack's per-position structural n (12/8/4 by script length, per §4.4 item 2's own illustration). The
function's docstring states this plainly as its own decision, within the spec's own explicit
deferral ("this document's own decision... the implementer's own judgement at Step 5" — actually
§4.4 item 2 leaves the exact wiring open), and correctly argues the direction of the error is safe
(can only over-state, never under-state, the true structural n, so the `_LOW_N_CAVEAT` still fires
whenever warranted). No fix required; noted for completeness since it is a literal deviation from
the plan's own illustrated numbers, and future per-shape script-length data (once S6 lands
`conversations.jsonl`) would let it be exact rather than an honest upper bound.

## What's solid

- **Design fidelity is otherwise excellent.** The one-`ItemResult`-per-conversation shape (§2.6),
  `FunnelCounts`/`HazardPoint`'s round-trip encode/decode ordering (`"censored"` checked before
  `"turnIndex"`), `named_metrics()`'s exclusion of `hazard`/`funnelCounts`/`perTurnPosition`, and
  the three renderers' placement (funnel before `## Arms`, per-position and hazard after) all match
  the spec exactly, confirmed by direct reading rather than the ledger's prose.
- **The hazard curve's no-cross-arm-diff prohibition is enforced structurally** (`report.py:806-859`
  reads only one run's own `HazardPoint` per cell — no expression could syntactically become
  `a_rate - b_rate`), and is additionally tested
  (`test_render_hazard_never_computes_a_cross_arm_difference`).
- **`turn_disposition_scores`/rule 4's mapping, the two discriminating pairs (`cap-hit`-empty-
  dispatch, `timed-out`-vs-`no-response`), `clean_through_turn`'s third state, and
  `hazard_points`'s censoring bookkeeping** are all correct on direct reading and well covered —
  the mutation-probe pattern the ledger describes (five gaps found, five fixed, one per unit) held
  up under my own independent re-read of Steps 0/1/3/4.
- **`_load_conversation_scorer` is a byte-for-byte structural match** to `_load_item_scorer`, as
  claimed.
- **Test quality is generally high** — fixtures are named after the exact rule they pin, the
  parametrized `cap-hit`/`timed-out` restraint test is exactly the right shape of test (it just
  checks the wrong attribute for the gap this review found), and the Step 6 integration test
  asserts on actual rendered markdown content rather than "did not raise."

## Open questions

- **The §4.2(d) failure-decomposition gap (finding 3) needs an owner decision**: wire it into S5
  now, or record an explicit, reversal-triggered deferral to a follow-up unit or S6. Either is
  defensible; leaving it unresolved and undocumented is not.
- Whether the two coordinator-facing gaps above (findings 1-2) should reopen U142/U143's own units
  or land as a fresh fix-round unit is a `teco` sequencing call, not this review's.
