# `model-bench` S5 — `tool-caller` pack, part 1: acceptance test plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (S5)

## 1. Scope & objective

Independent, black-box/execution-based acceptance pass for stage S5
(`docs/plans/small-model-benchmarking.md` §4 S5, governed by
`docs/plans/small-model-benchmarking-s5-spec.md`), which built
`packs/tool-caller-shop-assistant/` and `modelbench/scoring/toolcalls.py` (the `tool-caller` role's
environment and scorer, part 1). Implementation is code-complete across seven internal units
(U139-U143, plus the U144 code-gate's two fix rounds U145-U147); `analyst`'s code gate
(`model-bench/docs/reviews/small-model-benchmarking-s5.md`, verdict **needs changes**, all four
findings independently closed per the coordination ledger) is done. This is the final gate before
S5's stage close, and it verifies behavior a mutation-tested unit suite and a static review cannot:
does the real on-disk pack actually validate as far as it should; does `tools/sim.py` hold its own
stated dispatch-totality contract under inputs the shipped property test does not try; does a real
rendered report read correctly to a human; and do the units integrate correctly end to end on a
scenario nobody has built before.

**In scope:**
- Reproducing the full suite/lint as an independent baseline.
- Driving `validate_pack`/`validate` against the real, on-disk `tool-caller-shop-assistant`
  directory (not a test fixture).
- Exercising `tools/sim.py`'s storefront tools directly with adversarial inputs of my own,
  distinct from the shipped E1 property test's generated set.
- Visually inspecting a real rendered comparison report (funnel, per-turn-position, hazard,
  `I(t)`/`Y_calls/Y`, and the arg-decomposition annotation) built from a hand-constructed,
  multi-conversation, multi-defect scenario using the pack's **real** `tools/schemas.json`
  (including its real `boundaryRule` values) — not the synthetic single-purpose fixtures the unit
  suite already uses.
- Confirming the `argument_correctness` → `_Tally` → `FunnelCounts` → `_render_funnel` chain and
  the `_load_conversation_scorer` → `runner.py` seam hold together on that same scenario.
- Confirming `model-bench/AGENTS.md`'s "Current state" section does not yet claim S5 closed.

**Out of scope** (per the S5 spec's own §1/§5 framing, and the task brief):
- Any live LM Studio call against this pack — S5's own Done-when list has none; the first live
  `tool-caller` run is S6's (spec §1, §5).
- `conversations.jsonl`, `PROVENANCE.md`, the ~20 labelled prose-detector replies, and any
  `validate --strict` pass — all S6's own artifacts/obligations (spec §1, §2.6, §2.7).
- Re-litigating `analyst`'s already-closed code-gate findings or re-running the coordinator's own
  mutation probes (U139-U147) — both already done, independently, per the coordination ledger.
- Statistical-method validity of the Wilson/McNemar/Holm/bootstrap machinery — `-ml`'s and
  `data-scientist`'s territory, unchanged by S5 except for the three new renderers, which this plan
  does check for *rendering* correctness (not re-deriving the underlying statistics).

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(confirmed by `teco`'s brief); this is a code-level acceptance task in a component with no CPG
built, so "considered, not relevant" applies rather than "not applicable".

## 2. References

- `docs/plans/small-model-benchmarking.md` §4 S5 (the stage's own scope, part of the larger
  8-stage plan) — the contract under test.
- `docs/plans/small-model-benchmarking-s5-spec.md` (implementation spec, including the 2026-09-14
  correction adopting review Finding 3 option (a)) — the file-level design the code claims to
  satisfy.
- `model-bench/docs/reviews/small-model-benchmarking-s5.md` (`analyst`'s code gate, verdict needs
  changes; all four findings closed per U145-U147) — the static-review half this pass does not
  repeat.
- `docs/plans/small-model-benchmarking-coordination.md`, rows U139-U148 — the unit-by-unit delivery
  and mutation-probe trail, cited rather than re-derived.
- Code under test: `packs/tool-caller-shop-assistant/{pack.json,catalog.json,tools/sim.py,
  tools/schemas.json,prompts/system.md}`, `modelbench/scoring/toolcalls.py`, `modelbench/results.py`
  (`FunnelCounts`/`HazardPoint`/`ToolCallAggregates`), `modelbench/report.py` (the three new
  renderers), `modelbench/runner.py` (`_load_conversation_scorer`).
- `model-bench/AGENTS.md` (CLI conventions, hard rules, the S5+ "what's owed" framing).

## 3. Risk assessment

- **Highest risk: the rendered report is read by a human, and an assertion like `"label" in
  line` can pass while the output is confusing or wrong-looking.** All five coordinator mutation
  rounds and the code gate worked from the shipped test fixtures; none built a *fresh*,
  multi-conversation scenario against the pack's real schema content and eyeballed the output.
  Mitigated by building an independent scenario (two arms, five conversations each, deliberately
  combining an omitted-required defect, a wrong-value defect, and a boundary/unit-confused defect
  — using the real `confusedWith` values from `tools/schemas.json`, not a synthetic test schema —
  plus a restraint-timeout turn and a tool-dispatch-censored conversation) and reading the full
  rendered markdown end to end, hand-verifying every number against the fixture I built.
- **Second risk: `tools/sim.py`'s own stated "dispatch never raises on anything the model can
  produce" contract is checked by E1 only over argument *values*, never over the tool *name*
  field's type** — a gap the spec's own docstring calls out as a case the sim "must not assume"
  won't happen, yet no test drives it. Mitigated by adversarial `dispatch()` calls with
  non-string/unhashable names, in addition to the argument-shaped adversarial inputs E1 already
  covers.
- **Third risk: `validate_pack`'s real on-disk run is expected to fail** (S6's
  `conversations.jsonl` does not exist yet) — the risk is not "does it fail" but "does it fail for
  *only* the expected reason," i.e., every other one of `validate_pack`'s seven independent axes
  passes clean on the real artifact. Mitigated by calling each of the seven problem-check functions
  directly against the loaded real pack and asserting each returns `[]` except the row-count
  identity check.
- **Deliberately not tested**: statistical correctness of Wilson/McNemar bounds (already
  `data-scientist`-reviewed machinery, unchanged in kind by S5); the real catalog/prose content's
  domain fidelity (S6's authoring, not a testable claim yet); anything requiring
  `conversations.jsonl` (structurally impossible before S6).

## 4. Test items

| ID | Title | Preconditions | Steps | Expected result | Priority | Type |
|---|---|---|---|---|---|---|
| TP-001 | Full suite baseline | `.venv` set up | `.venv/bin/python -m pytest -q` | Matches the ledger's last-claimed count, 0 failures | High | Functional |
| TP-002 | Lint baseline | same | `.venv/bin/ruff check .` | Clean | High | Functional |
| TP-003 | `validate` CLI against the real on-disk pack | pack tree present, no `conversations.jsonl` | `./run.sh validate --pack packs/tool-caller-shop-assistant` | Exit code 4; exactly one printed problem, naming the missing `data.conversations` row-count-identity file | High | Acceptance/CLI |
| TP-004 | Each of `validate_pack`'s other six axes passes clean on the real pack | same | Call `_call_surface_problems`, `_tool_module_problems`, `_tool_import_problems`, `_prompt_problems`, `_scorer_problems`, `_answerability_stamp_problems` directly against the loaded real `Pack` | Each returns `[]` | High | Integration |
| TP-005 | `dispatch()` totality under argument-shaped adversarial inputs beyond E1 | real `build_environment()` | Call `dispatch` with float/negative/huge/bool/nested-object quantities, unicode/nested/empty product names, wrong-typed price filters, over-removal, non-mapping `arguments` | Zero raises; every case returns a `dict`/str "abstain"/"error" shape per §3.3's contract | High | Functional/exploratory |
| TP-006 | `dispatch()` totality under a non-string/unhashable tool `name` | real `build_environment()` | Call `dispatch(["x"], {})`, `dispatch({"x":1}, {})`, `dispatch(None, {})`, `dispatch(42, {})` | Zero raises (per the module's own stated contract) | High | Functional/exploratory |
| TP-007 | Ordinary storefront behavior sanity (place order twice, over-remove) | real `build_environment()` | Add/remove/place across two independent orders | `orderId` increments, cart clears on order, over-removal reports `removed: True` with the full available quantity | Medium | Functional |
| TP-008 | `_load_conversation_scorer` resolves the real pack manifest to the real module (no monkeypatch) | real `Pack` loaded via `load_pack` | `_load_conversation_scorer(pack)` | Returns `modelbench.scoring.toolcalls` itself (`is` identity), with `score_conversations` present | High | Integration |
| TP-009 | Multi-conversation, multi-defect scenario end to end through the real scorer + real schemas | hand-built `Conversation`/`ConversationTrace` fixtures (5 scripts/arm) using `tools/schemas.json`'s real `boundaryRule` | `score_conversations(...)` for two arms with deliberately different quality; render via `compare_report` | `FunnelCounts` (`argsOmittedRequired`/`argsWrongValue`/`argsBoundaryUnit`), `restraint`, hazard risk-set/censoring all hand-verifiable against the fixture's own construction | High | Integration/e2e |
| TP-010 | Human-eye read of the rendered report | output of TP-009 | Read the full markdown: funnel table, `I(t)`/`Y_calls/Y` sentence, `## Arms`, per-turn-position table, hazard curve | All four sections render, are internally consistent with each other and with the fixture, and are legible (no truncated/misaligned/misleading line) | High | Acceptance |
| TP-011 | `restraint` correctly excludes a turn that never replied (U145 regression check, on an independently-built fixture) | TP-009's fixture includes a `timed-out` restraint turn | Inspect `aggregates.restraint` | `successes` excludes the timed-out turn (verified arithmetically against the fixture) | Medium | Regression |
| TP-012 | `yCalls` `+1` term correct on an independently-built fixture (U145 regression check) | TP-009's fixture | Inspect the rendered `Y_calls / Y` line | Value equals `sum(iterations) + count(excluded-disposition turns)` computed by hand | Medium | Regression |
| TP-013 | `model-bench/AGENTS.md` doc-sync sanity | — | Read "Current state" | Still states S4 closed / "what S5+ owes" — no premature S5-closed claim | Low | Documentation |

## 5. Environment & data setup

- `model-bench/` as working directory throughout (repo has no root pytest config).
- `.venv` from `./setup.sh`, already present; no install/mutation performed.
- No LM Studio connection used or required (nothing in scope needs it).
- All adversarial/scenario scripts are written to the session scratch directory, not the repo, and
  read only `packs/tool-caller-shop-assistant/`'s already-committed files (`schemas.json`,
  `catalog.json`) — no repo file is written to or mutated by this pass.
- No shared/live state (FalkorDB, other components) touched.

## 6. Entry/exit criteria

**Entry**: implementation code-complete (confirmed: U139-U147 all accepted per the coordination
ledger); `analyst`'s code gate closed.

**Exit**: every test item above executed with recorded evidence; a clear verdict (pass / fail /
pass-with-notes) per item and overall; any defect written up reproducibly in the test report.

## 7. Out of scope (restated)

Live LM Studio calls; `conversations.jsonl`/S6 content; re-running U139-U147's own mutation probes;
statistical-method derivation review. See §1 for the full list and rationale.
