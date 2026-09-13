# `model-bench` S4 — acceptance test report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (S4)

## Summary

Independent, black-box acceptance pass for S4 (`guard-judge-understanding` +
`nlq-structured-query`), executed 2026-09-13 against this branch's working tree (`analyst`'s code
gate, `docs/reviews/small-model-benchmarking-s4.md`, verdict approve, is the last commit ahead of
this pass: `080e395`). Test plan: `docs/test-plans/small-model-benchmarking-s4.md`. All six of
S4's done-conditions were driven live — a fresh LM Studio run against a model neither pack had
seen before (`qwen2.5-3b-instruct`), plus standalone probes hitting the shipped pack modules
directly — rather than re-read from the implementers' own stored artifacts.

**Overall verdict: ACCEPT. All 7 test items pass (TP-000..TP-006). Zero defects found.** S4 is
behaviorally sound against its own stated done-conditions; nothing here should hold up stage close.

**CPG:** considered, not relevant — confirmed live this session (`mcp__cypher__query` against
`GRAPHS`): no `cpg_model-bench` graph exists among the 27 graphs loaded on this FalkorDB instance.
This is code-level acceptance work in a component with no CPG built, so "considered, not relevant"
applies rather than "not applicable".

## Results table

| ID | Result | Evidence |
|---|---|---|
| TP-000 | **PASS** | `.venv/bin/python -m pytest -q` → `1 failed, 1450 passed, 3 deselected`. The one failure is exactly the named, expected S5 tripwire (`tests/test_convo.py::test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5`) — identical count and identical failure to `analyst`'s own gate. |
| TP-001 | **PASS** | Fresh, independent live runs against `qwen2.5-3b-instruct` (never before used against either pack): `./run.sh run --pack nlq-structured-query --model qwen2.5-3b-instruct` → `stored: results/runs/nlq-structured-query-qwen2.5-3b-instruct-2026-09-13T22:40:27Z.json` (51s, exit 0, 40/40 items). `./run.sh run --pack guard-judge-understanding --model qwen2.5-3b-instruct` → `stored: results/runs/guard-judge-understanding-qwen2.5-3b-instruct-2026-09-13T22:41:23Z.json` (65s, exit 0, 85/85 items). |
| TP-002 | **PASS** | `./run.sh validate --pack packs/guard-judge-understanding` → `guard-judge-understanding 1.0.0 (guard-judge): valid`, exit 0. `./run.sh validate --pack packs/nlq-structured-query` → `nlq-structured-query 1.2.0 (nlq-generator): valid`, exit 0. `--strict` on both packs raised the identical, named, pre-existing `NotImplementedError` ("validate --strict: semantics are not decided anywhere in the plan... deliberately deferred") — matches `model-bench/AGENTS.md`'s documented deferral exactly, not a defect. |
| TP-003 | **PASS** | Standalone script re-derived every item's `answerable` flag by calling the shipped `packs/nlq-structured-query/tools/exec.py::compile_and_execute` against each item's own `reference_specs.json` entry (the exact algorithm `refresh_golden.py::run_stamp_answerability` uses), read-only, no write. Output: `Derived split: 34 answerable / 6 unanswerable (total 40)`; `Mismatches: NONE`; `Items missing 'answerable' key entirely: NONE`; unanswerable set `['nlq-34', 'nlq-35', 'nlq-36', 'nlq-37', 'nlq-38', 'nlq-39']` — exactly the corrected 34/6 split. **Independently re-confirmed live via the actual CLI**: `./run.sh` was not used for this (no CLI verb wraps it); instead `scripts/refresh_golden.py --pack packs/nlq-structured-query --stamp-answerability` was run directly and printed `answerable=34, unanswerable=6` — identical to both the standalone re-derivation and the committed file; `git status --porcelain packs/nlq-structured-query/` showed **zero diff** afterward (the write reproduced byte-identical content, confirming determinism, not merely that a run happened). |
| TP-004 | **PASS** | Standalone probe drove `packs/nlq-structured-query/tools/exec.py` and `modelbench/scoring/extraction.score_pair` directly with three hand-crafted cases: (a) `filters` at length 5 (>4) → `MalformedSpecError: matches[0].filters has 5 entries, max 4`; (b) a structurally valid spec against label `"Widget"` (unregistered) → `SchemaViolationError: label 'Widget' is not registered for this dataset`; (c) a fully valid spec executed against the real catalog data (`Wireless Mouse Pro`, true price 29.99) scored against a deliberately wrong expected value (999.99) → `score_pair` returned `correct=False, reason="29.99 does not match expected 999.99"`, and a sanity check against the true value (29.99) returned `correct=True` (proves the harness isn't defaulting to always-False). Three distinct, non-conflated outcomes observed directly. |
| TP-005 | **PASS** | Same probe: `score_pair(expected={"type":"set","values":["Marlowe Robotics","Acme Corp"]}, shape="conflicting-facts", tool_result={items containing Marlowe Robotics, Acme Corp, AND Globex})` → `correct=True, reason="all expected conflicting values present"` (subset containment). The **identical** input scored under `shape="filter-list"` (negative control) → `correct=False, reason="expected ['acme corp', 'marlowe robotics'], got ['acme corp', 'globex', 'marlowe robotics']"` (exact set equality). Confirms the containment exception is shape-scoped to `conflicting-facts` specifically, not a blanket relaxation. |
| TP-006 | **PASS** | Fresh `./run.sh compare --pack guard-judge-understanding --models qwen2.5-3b-instruct,qwen/qwen3-4b-2507` (new arm from TP-001's run, paired against the existing stored `qwen/qwen3-4b-2507` run) → `reports/guard-judge-understanding-20260913-01.md`, read in full by eye. `falseAdvanceRate` and `falseSuspendRate` render as two `###` subsections under one `## Verdicts` section, immediately followed by `_This pack declares no headline metric: its verdict metrics are co-equal and are printed side by side, in the manifest's declared order, with no summary line above them and no arithmetic combining them_`. Every occurrence of "85" in the rendered file is a denominator/composition note — `"paired n: 40 of 85 items"`, `"45 unscoreable in both"`, `"paired n: 30 of 85 items"`, `"55 unscoreable in both"` — never a pooled accuracy, count, or rate computed across all 85 items; no line anywhere reads as an aggregate headline. Six exploratory metrics (`advanceRecall`, the boundary tier, four path-split diagnostics) print separately, each explicitly labelled `— no significance claim`. |

## Defects

None found.

## Coverage & gaps

**Covered, live, this pass:** all six of S4's plan-stated done-conditions, each with fresh evidence
independent of the implementers' own stored runs (a model neither pack had been run against
before; a read-only re-derivation of the answerability stamp that does not trust the committed
file; hand-crafted adversarial specs driven straight at the shipped executor rather than only the
shipped unit tests). The pre-existing suite (1450 tests) was re-run as a baseline and matches
`analyst`'s own count exactly.

**Not covered, deliberately (see test plan §3/§7):** the statistical machinery behind the printed
Wilson/McNemar/Holm numbers (already `-ml`/`data-scientist`-owned, unchanged by S4 — `report.py`
has zero diff in the reviewed range); model-quality/accuracy itself (not a done-condition — this
tool has no pass/fail gate by design); the two already-logged, already-not-blocking minor findings
from `analyst`'s gate (`luckyPassCount` et al. unrendered in `report.py`; the `order_by` sort's
missing `None`-key guard in `tools/exec.py`) — re-read, not re-tested, since they were not disputed
and are already tracked as `docs/BACKLOG.md` follow-ups per that gate's own recommendation.

**Residual risk:** low. The one live-run-only path this pass did not separately re-exercise is
`--stamp-answerability`'s and `--check-tables-shape`'s behavior under a version-gate *refusal*
(i.e., re-running against an *unchanged*-since-last-stamp `packVersion`) — this pass's own live
`--stamp-answerability` run happened to land on a version where the gate was open (see note below)
rather than closed, so the refusal path itself was not observed firing, only inferred from reading
`_pack_version_gate`'s code. This is a gap in this pass's own live coverage, not a suspected defect
— the gate's logic was read directly and is simple (`recorded_version == current_version`).

## Feedback & recommendations

- **Operational note, not a defect:** `run` refuses with a plain-text warning
  ("LM Studio changed since you last attested host.json...") whenever `host.json`'s attestation
  goes stale relative to LM Studio's live-observed runtime fields, and requires `attest` to be
  re-run (non-interactively, via repeated `--set`) before any live run proceeds. This fired during
  this pass and was resolved in ~10 seconds; worth knowing ahead of time for the next acceptance
  pass on this component so it isn't mistaken for a blocker.
- **Observation, not a defect:** running `scripts/refresh_golden.py --pack packs/nlq-structured-query
  --stamp-answerability` against the pack's current state (packVersion `1.2.0`, `PROVENANCE.md`
  recording `1.1.0`) was **not** refused by `_pack_version_gate` — the versions already differed
  (the pack's `packVersion` had already been bumped past `PROVENANCE.md`'s last-recorded value by
  the implementation units), so the write path executed and reproduced `items.jsonl` byte-for-byte.
  This is correct behavior, not a defect — flagged here only because it means TP-003's live-CLI
  confirmation was a genuine write path (verified idempotent via `git status --porcelain` showing
  no diff afterward), not a no-op dry run, worth knowing for anyone re-running this exact command
  against a tracked pack file in the future.
- **Suggestion:** the two `analyst`-logged minors (unrendered `luckyPassCount`/malformed/schema
  counts in `report.py`; the `order_by` `None`-key sort guard) are both still open in
  `docs/BACKLOG.md` per that gate's recommendation — no new information from this pass changes
  either's priority; restating only to confirm this pass did not surface a reason to escalate
  them.

## Artifacts produced by this pass

- `results/runs/nlq-structured-query-qwen2.5-3b-instruct-2026-09-13T22:40:27Z.json`
- `results/runs/guard-judge-understanding-qwen2.5-3b-instruct-2026-09-13T22:41:23Z.json`
- `reports/nlq-structured-query-20260913-01.md`
- `reports/guard-judge-understanding-20260913-01.md`

All four are new, untracked files in the working tree (same convention as the implementers' own
committed `-20260911-*` artifacts) — left uncommitted for the user/`teco` to decide whether to
land them alongside this report.
