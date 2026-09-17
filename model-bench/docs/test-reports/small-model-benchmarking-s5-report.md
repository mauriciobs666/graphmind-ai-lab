# `model-bench` S5 — `tool-caller` pack, part 1: acceptance test report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** — (S5)

## Summary

Executed the plan at `model-bench/docs/test-plans/small-model-benchmarking-s5.md` against
`model-bench/` at its current tree state (S5 implementation code-complete, U139-U147 all accepted,
`analyst`'s code gate closed — `docs/plans/small-model-benchmarking-coordination.md` rows U139-U148;
`model-bench/docs/reviews/small-model-benchmarking-s5.md`). All work was read-only against the repo:
every adversarial/scenario script lives in the session scratch directory and only *reads*
`packs/tool-caller-shop-assistant/`'s already-committed files; no repo file, pack data, or shared
service was mutated.

**Verdict: PASS, with one defect found and one testability gap noted — neither blocks stage
close.** The pack's environment/tooling, the scorer's arithmetic, the report renderers, and the
cross-unit seams all behave correctly under fresh, independently-constructed scenarios that go
beyond the shipped test suite. One genuine defect was found in `tools/sim.py`'s dispatch-totality
contract (TD-1, below) — real, reproducible, but not reachable through the actual driving code path
today, so it does not gate this stage; it is a robustness gap in a *stated* contract worth fixing
before that guarantee is relied on elsewhere (e.g. if a future pack or a fuzzing harness calls
`dispatch` more directly than `convo.py` does).

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(confirmed by `teco`'s brief, consistent with the S5 spec's own CPG line); this is a code-level task
in a component with no CPG.

**Suite/lint, reproduced myself:**

```
$ .venv/bin/python -m pytest -q
1606 passed, 3 deselected in 7.39s

$ .venv/bin/ruff check .
All checks passed!
```

Matches the coordination ledger's last-claimed counts (U147: `1606 passed, 3 deselected`; ruff
clean) exactly.

## Results table

| ID | Result | Evidence |
|---|---|---|
| TP-001 | Pass | `.venv/bin/python -m pytest -q` → `1606 passed, 3 deselected` (reproduced above). |
| TP-002 | Pass | `.venv/bin/ruff check .` → `All checks passed!`. |
| TP-003 | Pass | `./run.sh validate --pack packs/tool-caller-shop-assistant` → exit `4`; stderr is exactly one line: `tool-caller-shop-assistant: cannot read data.conversations 'conversations.jsonl' for the row-count identity ([Errno 2] No such file or directory: ...)`. Confirmed via `1>/tmp/out.txt 2>/tmp/err.txt` that stdout is empty and stderr carries only this one line. |
| TP-004 | Pass | Called `_call_surface_problems`, `_tool_module_problems`, `_tool_import_problems`, `_prompt_problems`, `_scorer_problems`, `_answerability_stamp_problems` directly against `load_pack(Path("packs/tool-caller-shop-assistant"))`; each returned `[]`. Only `_sampling_problems` returned the one expected row-count-identity message. This confirms the AST import allowlist, the `tools.module` path check, `environment.requires`'s callSurface derivation, the `prompt` block's `historyReplay`/`maxIterationsPerTurn` scoping, and the (tool-caller-scoped-out) scorer-resolution check all pass clean on the real, on-disk artifact — not merely in `test_packs.py`'s in-memory fixtures. |
| TP-005 | Pass | 16 adversarial argument-shaped `dispatch()` calls beyond E1's own generated set (non-mapping `arguments`; float/negative/huge-int/nested-object/stringified-number/bool `quantity`; unicode/nested-object/empty-string `productName`; wrong-typed/nested `minPrice`/`category`; an unknown tool name with garbage arguments) — zero raises, every case returned a `dict`/error/abstain shape. Full transcript: scratch `probe_sim_final_output.txt`. |
| TP-006 | **Fail — defect TD-1** | `dispatch(["not_a_real_tool"], {})` and `dispatch({"x": 1}, {})` both raise `TypeError: unhashable type`, violating `tools/sim.py`'s own stated "dispatch never raises on anything the model can produce" contract and its own docstring's explicit caveat that the sim "must not assume" a well-typed name always reaches it. See Defects below. |
| TP-007 | Pass | Over-removing (100 of 3) reports `removed: True, quantityRemoved: 3` (the whole line, not an error); two sequential `place_order` calls on one environment produce `order-1`/`order-2`, cart clears between them. |
| TP-008 | Pass | `_load_conversation_scorer(load_pack(Path("packs/tool-caller-shop-assistant")))` returns `modelbench.scoring.toolcalls` itself (`is` identity confirmed), with `score_conversations` present — the real manifest's `"scorer": "toolcalls"` resolves through the real seam with zero mocking. |
| TP-009 | Pass | Built 2 arms x up to 5 conversations each (`A-01` boundary/unit-confused `maxPrice`, `A-02` omitted-required `productName`, `B-01` a case-folding non-defect confirming `_canon_str`'s deliberate case-insensitivity, `B-02` a `timed-out` restraint turn, `C-01` a tool-dispatch-censored conversation only in the "bad" arm) against the pack's real `tools/schemas.json`. `FunnelCounts` for the "bad" arm: `argsOmittedRequired=1, argsWrongValue=1, argsBoundaryUnit=1` — hand-verified: only `A-01`'s `maxPrice=4999` (vs. expected `49.99`, matching the schema's own `confusedWith` value) is a real wrong-value/boundary-unit hit; `A-02`'s missing `productName` is the one `omittedRequired` hit; `B-01`'s case-folded name is correctly *not* flagged, confirming `_scalar_equal`'s canonicalization is intentional, not a scorer bug. |
| TP-010 | Pass | Full rendered report read in `rendered_report.md` (scratch). Funnel table, `I(t)`/`Y_calls/Y` sentence, `## Arms`, per-turn-position table, and hazard curve all render, are legible, and are mutually consistent — cross-checked below. |
| TP-011 | Pass | `aggregates.restraint` on the "bad" arm: `successes=2, n=3` — the `timed-out` restraint turn (`B-02` turn 0) correctly excluded from `successes` (hand-traced: 3 restraint turns total, 2 replied cleanly, 1 timed out) — confirms U145's fix holds on a fixture none of U139-U147's own tests built. |
| TP-012 | Pass | Rendered `Y_calls / Y` line: "bad" arm `8/7 = 1.14`, "good" arm `7/6 = 1.17`. Hand-computed independently: `sum(iterations)` = `turnsDriven` in both fixtures (every turn has `iterations=1`), `+1` for each arm's one `timed-out` turn (`B-02` is shared by both arms) → bad `7+1=8` over `Y=7`; good `6+1=7` over `Y=6`. Matches exactly. |
| TP-013 | Pass | `model-bench/AGENTS.md`'s "Current state" section still reads "Stage S4 is closed..." and "What S5+ owes...`_load_conversation_scorer` (`runner.py`) still raises `NotImplementedError` unconditionally" — stale relative to the code (expected; doc sync is a separate, later step, not a defect per the task brief). |

## Defects

### TD-1 (Minor — no live-path impact today, but a real contract violation) — `ShopEnvironment.dispatch` raises `TypeError` on a non-hashable tool name, contradicting its own documented dispatch-totality contract

**Severity rationale**: the module's own docstring (`packs/tool-caller-shop-assistant/tools/sim.py`
lines 10-21) states as a load-bearing rule: *"`dispatch(name, arguments)` never raises on anything
the model can produce... an unknown tool name that still reached `dispatch` (should not happen
given `drive` only dispatches a name it parsed off the model's own `tool_calls`, but the sim must
not assume it)."* That second clause is an explicit self-imposed requirement to tolerate a
malformed name, not just a malformed argument set. As shipped, it does not.

**Steps to reproduce**:
```python
import sys
sys.path.insert(0, "packs/tool-caller-shop-assistant/tools")
import sim
env = sim.build_environment()
env.dispatch(["not_a_real_tool"], {})   # -> TypeError: unhashable type: 'list'
env.dispatch({"x": 1}, {})              # -> TypeError: unhashable type: 'dict'
```

**Root cause** (`tools/sim.py:111-119`, `ShopEnvironment.dispatch`):
```python
def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any:
    args = dict(arguments) if isinstance(arguments, Mapping) else {}
    safe_name = name if isinstance(name, str) else str(name)
    handler = self._handlers.get(name)   # <- BUG: looks up the raw `name`, not `safe_name`
    ...
```
`safe_name` is computed specifically to coerce a non-`str` name to something safe (and is used
correctly for the returned `{"error": "unknown-tool", "name": safe_name}` payload and the stored
`DispatchRecord`), but the dict lookup on the line above it uses the raw `name` — so a non-hashable
`name` (a `list`/`dict`, the shapes a malformed or adversarially-fuzzed tool call could plausibly
carry) raises before `safe_name` is ever consulted. `dispatch(None, {})` and `dispatch(42, {})`
happen to work only because `None`/`int` are hashable and simply miss every key in `_handlers`.

**Expected vs. actual**: expected — `dispatch` returns `{"error": "unknown-tool", "name": "..."}`
for any input, coerced to string, matching the existing `test_dispatch_unknown_tool_name_returns_an_error_rather_than_raising`
test's own contract for a string name that isn't in the catalog. Actual — `TypeError` propagates
out of `dispatch` for a `list`/`dict` name.

**Live-path impact**: none today. `convo.py`'s `_tool_call_name` (`convo.py:387-393`) already
guarantees `name` is always a real, non-empty `str` before `env.dispatch(name, arguments)` is ever
called (`return name if isinstance(name, str) and name else None`, with the caller `continue`-ing
past any call whose name fails that check) — confirmed by direct reading of `convo.py:744-756`.
So the real driving loop (`_drive_conversations` → `drive` → `dispatch`) can never present this
shape. This is why it survived E1 (which only varies `dispatch`'s **argument** shapes, never the
tool name field) and every mutation round to date — none of them targeted this seam, because
nothing in the driving code path exercises it.

**Fix**: one-line — use `safe_name` for the lookup too: `handler = self._handlers.get(safe_name)`.
Trivial, local, no test currently relies on the buggy behavior. Recommended before this contract is
relied on by a future fuzzing harness or a second pack's tool module copied from this one as a
template — deferring implementation to `coder`/`tdd-engineer` per this agent's own guardrails.

## Coverage & gaps

**Covered by this pass, beyond what the unit suite and code gate already established:**
- The real on-disk pack's seven `validate_pack` axes, six of which have zero prior "against a real
  directory" coverage (`test_packs.py`'s own tests build in-memory/fixture packs under
  `tests/fixtures/packs/`, never this real tree).
- `tools/sim.py`'s dispatch-totality contract under a tool-*name* type adversarial axis E1 does not
  cover at all.
- A rendered, two-arm, five-conversation-per-arm report built against the pack's **real**
  `tools/schemas.json` content (including its actual `boundaryRule.confusedWith` values), which no
  shipped test does — every shipped fixture either uses a synthetic schema (Steps 3-4) or a
  single-conversation/single-defect shape (Step 5's E2-E4, Step 6's two-conversation integration
  test).
- `_load_conversation_scorer`'s real resolution against the real manifest with zero monkeypatching
  (the shipped integration test monkeypatches this specific seam to speed up the test; TP-008 closes
  the gap that leaves).
- Independent arithmetic verification of U145's two fixes (restraint-disposition gating, the
  `yCalls` `+1` term) on a fixture none of U139-U147's own tests built.

**Not covered, and why that's fine for this stage:**
- Anything requiring `conversations.jsonl`, the real 12 scripts, or a live LM Studio call — all
  structurally impossible before S6 lands its own artifacts, per the spec's own scope line.
- The real catalog/prose content's domain fidelity (product names, prices) — S6/S7 authoring
  concern, not a testable claim against S5's own done-conditions.
- Re-deriving Wilson/McNemar/Holm statistical correctness — unchanged in kind by S5, already
  `data-scientist`-reviewed elsewhere.

**Residual risk**: low. The one defect found (TD-1) has no path through the real driving code
today and is cheap to fix. The testability gap it points at — E1's adversarial generator varies
only argument values, never the tool-name field's type — is worth closing alongside the fix so a
future regression in this exact seam is caught automatically rather than requiring another
by-hand acceptance pass.

## Feedback & recommendations

1. **Fix TD-1** (`tools/sim.py:114`, `self._handlers.get(name)` → `self._handlers.get(safe_name)`)
   and add a regression test to `tests/test_tools_sim.py` alongside the existing
   `test_dispatch_unknown_tool_name_returns_an_error_rather_than_raising` test, parametrized over a
   `list`/`dict` name. Low effort, no design decision involved — a good `coder` one-off or folded
   into whichever unit next touches `tools/sim.py`.
2. **Consider widening E1's adversarial generator** (`tests/test_tools_sim.py`'s
   `_adversarial_argument_dicts`) with one additional axis: the tool *name* itself, not just its
   arguments — a small, generic addition (cross every declared schema name's own adversarial-value
   sibling: `None`, a list, a dict) that would have caught TD-1 automatically and guards the same
   seam for any future tool module built from this one as a template.
3. No other testability issues surfaced. Fixture-building for the `tool-caller` role (`Conversation`/
   `Turn`/`ConversationTrace`/`TurnTrace`/`DispatchRecord`) is straightforward and well-factored
   (`tests/test_scoring_toolcalls.py`'s `make_turn`/`make_trace`/`make_script` helpers were directly
   reusable in spirit for this pass's own independent fixtures).

## Artifacts

- Test plan: `model-bench/docs/test-plans/small-model-benchmarking-s5.md`
- This report: `model-bench/docs/test-reports/small-model-benchmarking-s5-report.md`
- Scratch evidence (not part of the repo): adversarial probe scripts and their output, and the
  full rendered two-arm comparison report, under this session's scratch directory
  (`probe_sim_final.py`/`probe_sim_final_output.txt`, `probe_report.py`/`rendered_report.md`,
  `probe_validate.py`).
