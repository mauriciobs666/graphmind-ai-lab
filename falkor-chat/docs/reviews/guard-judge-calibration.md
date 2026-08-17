# Guard-judge calibration harness — code review (K-027 item 3)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-027 item 3 (M3.5)

## Scope & verdict

Diff-scoped review of the uncommitted guard-judge calibration harness in `falkor-chat/server/`:

- `server/tests/eval/guard_calibration.py` (327 lines) — library: fixture assembly, `RecordingJudge`,
  `run_case`, and the metric functions (G1/G2/κ/coercion-flip/flip-rate/materiality-probe/per-path).
- `server/tests/eval/test_guard_calibration.py` (401 lines, 24 tests) — offline unit tests.
- `server/tests/eval/test_guard_calibration_live.py` (456 lines) — the live k=3×26-case run and report
  writer.
- `config/models.json` — the one-line `"lmstudio/qwen/qwen3-4b-2507": {"temperature": 0}` addition.

Baseline: `docs/archive/plans/m3-guard-calibration.md` (frozen protocol, §4/§5) and
`docs/plans/guard-judge-calibration-ml.md` (current-code addendum, §4's `run` construction). This is
the **implementation** gate — the plan-level review already happened via the `-ml.md` note. Out of
scope: statistical/methodology soundness of the gate math itself (a separate `data-scientist` unit),
and the report's own narrative framing beyond verifying its numbers are correctly computed.

Verified live (not just read): ran the offline suite (24/24 green), ran the full default `pytest -q`
baseline (1088 passed, 3 deselected, no live-marker leakage), performed the two requested mutations
myself (G1 per-call → per-case-majority; the path-assertion made a no-op) and confirmed both go red
then green again on revert, and hand-recomputed G1/G2/FAR_all/κ/boundary-confusion/per-path-accuracy/
materiality-probe/flip-rate/coercion-flip-rate from the report's own 26-row per-case table, independent
of reading the metric functions' source.

**Verdict: approve with suggestions.** No blocker. The G1/G2 counting rules are correct, the
`ModelGateway` construction deviation is sound and traces to a real precedent, the live suite is
correctly gated, and every number in the shipped report reproduces by hand from its own per-case
table. Findings below are minor conformance/quality items, none of which put the "wire" verdict in
`docs/test-reports/guard-judge-calibration-2026-08-17.md` in doubt.

**CPG:** considered, not relevant — `cpg_falkorchat` exists and is same-day fresh
(`BUILT_AT=2026-08-17T00:40:42Z`, no `SOURCE_COMMIT` since the parse root `/tmp/cpg-src/falkor-chat-server`
wasn't a git working tree at build time), but it does not contain the files under review (`MATCH (f:FILE)
WHERE f.NAME CONTAINS 'guard_calibration'` → 0 rows) — these are new, uncommitted files created after
whatever snapshot fed the build. This review's central risks (metric-formula correctness, a spec
counting rule, a construction-path deviation) are call-count/data-flow questions best answered by
direct reading, live test execution, and independent hand-recomputation against real output — all of
which I did — rather than by static call-graph structure a CPG would expose; a rebuild wouldn't have
changed the method here even if it existed.

## Findings

### Major

None.

### Minor

1. **Per-case table drops two columns the archived spec asks for.**
   `docs/archive/plans/m3-guard-calibration.md` §5.3 specifies the report's full per-case table
   carries "id, tier, path, expected, the k raw decisions, **raw rationales**, final decisions,
   **coercion_flip**." `_build_report`'s table (`test_guard_calibration_live.py:441-453`) renders id,
   tier, path, expected, raw decisions, final decisions, and only `r.final_rationale` — no raw
   rationale column and no explicit per-row `coercion_flip` boolean. Impact is smaller than the gap
   looks: reading `_coerce_verdict` (`guards.py:466-484`), `final_rationale` equals the raw rationale
   verbatim in the "not-True-decision" branch, and it *embeds* the raw rationale in the contradicts
   branch — the only case with genuine loss is a non-mapping raw output, where `raw_rationale` is `""`
   anyway (`guard_calibration.py:179-180`), so there's nothing to show. A reader can also infer
   `coercion_flip` per row by eye-diffing the raw/final columns. Still, this is what the protocol asked
   for and it's cheap to add — `ReplicateResult` already carries `raw_rationale`, and
   `raw_decision is True and not r.final_decision` is a one-line per-replicate `coercion_flip`. Suggest
   `tdd-engineer` add both columns before this table format gets reused for a re-run (a future run that
   does see coercion flips would want the raw text without cross-referencing `raw`/`final` columns by
   hand).

2. **Dead imports and lint noise** (`ruff check` run locally against the three files; ruff is not a
   wired gate per `falkor-chat/AGENTS.md` §14.7, so this is cosmetic, not a gate failure):
   `test_guard_calibration.py` imports `RecordingJudge` and `path_taken` from `guard_calibration` but
   never uses either (`test_guard_calibration.py:30,42`) — both are exercised indirectly through
   `run_case`/`assert_path_matches`, so the imports are just stale. `test_guard_calibration_live.py`
   imports `datetime` and `timezone` from `datetime` (`:51`) and uses neither — only `date` is used.
   Both files also have an unsorted import block (`I001`) and `test_guard_calibration.py` has two
   lines over the 100-col limit (`:386-387`). None affect behavior; a quick `ruff check --fix` (plus
   manually dropping the two genuinely-unused names, since `--fix` won't know they're dead until the
   `F401`s are re-run) would clear all of it.

### Nit

3. **`WrongPathRow` class in `test_run_case_raises_when_a_live_judge_answer_silently_drove_the_wrong_path`
   is unused** (`test_guard_calibration.py:192-193`) — declared, never instantiated or referenced; the
   test uses a plain `dict(TURNS_ROW)` instead. Harmless, but it reads as a leftover from an earlier
   draft of the test. Fine to delete on next touch.

## What's solid

- **G1/G2 counting rules are correct and match the archived spec exactly**, verified three ways:
  (a) direct reading of `false_advance_rate` (per-call, `guards_calibration.py:201-213`) and
  `advance_recall` (per-case majority, `:229-237`); (b) offline tests
  `test_g1_false_advance_rate_is_per_call_not_per_case_majority` /
  `test_g2_advance_recall_is_per_case_majority` pin the distinction with a case designed to expose a
  swapped rule; (c) I personally re-ran the swap-mutation described in the brief (per-call → per-case
  majority for G1) and watched two tests go red (`n_calls`/`advances` assertions), then reverted and
  confirmed 24/24 green again, with the file byte-identical to the pre-mutation copy afterward.
- **The path-assertion mutation-test claim is also confirmed independently.** Turning
  `assert_path_matches`'s body into a no-op reliably fails three tests
  (`test_assert_path_matches_raises_when_turns_row_silently_used_understanding`,
  the understanding-side mirror, and `test_run_case_raises_when_a_live_judge_answer_silently_drove_the_wrong_path`),
  reverting restores 24/24 green.
- **Every number in the shipped report reproduces by hand from its own 26-row table** — G1 (0/30),
  G2 (9/11 = 81.8%), FAR_all (3/45 = 6.7%), κ (0.811, with po/pe/marginals all matching a from-scratch
  2×2-confusion-matrix derivation), boundary confusion (tp=0 fp=1 tn=4 fn=0, driven entirely by `tn-07`
  advancing on a boundary case), per-path accuracy (understanding 13/15, turns 6/6), materiality-probe
  ("passed" — correctly, since `cs-04`'s majority is suspend, so the control never satisfies the
  bloc-failure precondition regardless of the three probes), flip-rate (0/26, every case unanimous
  across replicates), and coercion-flip-rate (0/78 overall, 0/15 on the 5 `r1_probe` cases — confirmed
  the fixture really has exactly 5 `r1_probe: true` rows × k=3). No formula produced a number that
  happened to match by coincidence — I derived each independently from the table before comparing.
- **The `ModelGateway` construction deviation is real, sound, and traces to a verifiable precedent.**
  `ModelGateway.from_env()` (`modelconfig.py:625-647`) is exactly `catalog =
  ProviderCatalog.load(opencode_path); overlay = Overlay.load(overlay_path); cls(catalog, overlay,
  ...)` — the live harness's `models` fixture (`test_guard_calibration_live.py:143-179`) does the
  identical two-call construction against the real files, differing only in skipping
  `_config.assert_no_legacy_model_env()` (irrelevant here) and reading `DEFAULT_MODEL_CONFIG_PATH`
  directly instead of through the env-redirectable `_config.MODEL_CONFIG_PATH` — which is the whole
  point, since `server/tests/conftest.py:116-136`'s autouse `_model_config_env` fixture monkeypatches
  both `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` for every test unconditionally, and
  `.from_env()` would silently resolve against the offline dim-4 test fixture instead of the real
  `config/models.json`/`qwen3-4b-2507`. The cited precedent, `test_golden_set_integrity.py`'s "D7
  mechanism 1" (`:103-113`), is real and does exactly this pattern (`Overlay.load(DEFAULT_MODEL_CONFIG_PATH)`,
  bypassing `.from_env()`) — the live harness's extension to also build a `ProviderCatalog` the same
  way is a necessary and faithful generalization (the integrity test only needed `overlay.default_for(...)`,
  never an actual LLM call, so it didn't need the catalog half). The module-docstring's claim that
  `_AMBIENT_OPENCODE_CONFIG` is "captured at MODULE IMPORT time... before any per-test fixture...
  executes" is correct: pytest collects and imports test modules before running any fixture, autouse
  or not, so the module-level `os.environ.get(...)` at line 84 genuinely sees the pre-monkeypatch
  value.
- **Live-suite gating is correct and leak-free.** `pytest.mark.live` correctly deselects exactly 3
  tests (`test_guard_calibration_live.py`, `test_judge_live.py`, `test_workflow_live.py`) under the
  project's `addopts = -m "not live"`; a bare `.venv/bin/python -m pytest -q` from `server/` ran clean
  (1088 passed, 3 deselected, no network activity, no LM Studio dependency touched) and did not exercise
  this harness at all.
- **`config/models.json`'s one-line change is exactly what was asked and nothing else** —
  `git diff --stat` confirms it's the only tracked-file change in this diff, and the added entry
  (`"lmstudio/qwen/qwen3-4b-2507": {"temperature": 0}`) sits correctly under the existing `models` map
  alongside the pre-existing embedding-dim entry, matching `modelconfig._resolve_element`'s expected
  shape (confirmed both by reading the resolver and by the live report's own provenance header showing
  `temperature: 0` actually reached the resolved request params).
- **Fixture-to-call assembly matches archived §5.1's table precisely, including the ml-note's `run`
  addendum** — `build_call` produces the envelope form for understanding-path rows, a genuinely
  non-parsing prose string plus an `understanding`-key-free `ctx` for turns-path rows (F5's three-part
  condition), and `run = {"ws": "ws:golden-eval", "modelOverrides": {}}` rather than the archived
  document's literal `run = {}` — correctly citing why (`{}` still resolves identically today, but
  through a branch production doesn't take). The "assert the evidence path was taken" requirement is
  enforced per replicate, not once per case (`run_case` builds a fresh `RecordingJudge` and calls
  `assert_path_matches` inside the `k` loop), which is the stricter and correct reading of "not
  optional."
- **`RecordingJudge`'s `accepts_run` propagation is correct** and specifically tested
  (`test_run_addendum_forwards_run_only_when_judge_accepts_it`) against a legacy stub judge lacking the
  K-042 capability flag — `run=` is never forwarded unconditionally, matching `guards.evaluate_guard`'s
  own `getattr(judge, "accepts_run", False)` gate.

## Open questions

None — the brief's specific concerns (G1/G2 counting, the mutation-test claim, the `ModelGateway`
deviation, live-suite leakage) all resolved cleanly on independent verification, and no blocking
correctness issue surfaced in the metric logic. The one item worth a maintainer decision rather than a
pure fix is whether to backfill the per-case table's missing `raw_rationale`/`coercion_flip` columns
now (cheap, harmless) or defer until a future run actually produces a non-zero coercion-flip rate
where the gap would matter in practice — either is reasonable; I don't have a strong preference.
