# `model-bench` S4 — `guard-judge-understanding` / `nlq-structured-query` acceptance test plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (S4)

## 1. Scope & objective

Independent, black-box acceptance pass for stage S4 (`docs/plans/small-model-benchmarking.md` §4
S4, as corrected by `docs/plans/small-model-benchmarking-s4-spec.md`), which built the
`guard-judge-understanding` and `nlq-structured-query` packs. Implementation and `analyst`'s
cumulative code gate (`docs/reviews/small-model-benchmarking-s4.md`, verdict **approve**) are both
already closed; this is the last gate before S4's stage close, and it judges S4's **six stated
done-conditions** behaviorally, by driving the real CLI and the real pack modules myself, rather
than by re-reading the diff or the already-approved review.

**In scope:** exactly the six done-conditions quoted in the task brief (verbatim from the plan's
S4 section) — end-to-end execution of both packs against a model; the answerability stamp; the
executor's three-way validation split; the `conflicting-facts` subset-containment exception;
`validate` passing on both packs; and a human-eye read of the guard-judge report's no-headline/
no-pooled-figure shape.

**Out of scope:** re-litigating `analyst`'s code-level findings (naming, type shapes, docstring
accuracy) — that gate already passed and is not re-run here; the two logged minor
follow-ups (unrendered `luckyPassCount` et al. in `report.py`, the `order_by` `None`-key sort
guard) — tracked in `docs/BACKLOG.md`, not blocking, not re-verified; `--strict`'s
`NotImplementedError` — a named, deliberate deferral (runner-spec §9), unrelated to S4; S5+ roles
and scorers; statistical-method validity of the Wilson/McNemar/Holm machinery — that is `-ml`'s
and `data-scientist`'s territory, already reviewed, and unchanged by S4 (`report.py` has zero diff
in this range per the code gate).

**CPG:** considered, not relevant — no `cpg_model-bench` graph is loaded on this FalkorDB instance
(checked live this session via `mcp__cypher__query` against the graph name); this is a code-level
acceptance task in a component with no CPG built, so "considered, not relevant" applies rather
than "not applicable".

## 2. References

- `docs/plans/small-model-benchmarking.md` §4 S4 (the two done-conditions, verbatim) — the
  contract under test.
- `docs/plans/small-model-benchmarking-s4-spec.md` (implementation spec, as corrected 2026-09-11 to
  the 34/6 answerability framing) — the file-level design the code claims to satisfy.
- `docs/reviews/small-model-benchmarking-s4.md` (`analyst`'s cumulative code gate, verdict approve)
  — the static-review half this pass does not repeat.
- `docs/reviews/nlq-conflicting-facts-answerability-ml.md` (the 34/6 methodology ruling).
- Code under test: `packs/guard-judge-understanding/`, `packs/nlq-structured-query/`,
  `modelbench/scoring/classification.py`, `modelbench/scoring/extraction.py`,
  `packs/nlq-structured-query/tools/exec.py`, `scripts/refresh_golden.py`'s
  `run_stamp_answerability`.
- `model-bench/AGENTS.md` (CLI conventions, the `--strict` deferral, the S5 tripwire).

## 3. Risk assessment

- **Highest risk: done-condition 6 (no headline/no pooled figure) is a negative claim about
  rendered prose** — a grep can miss a differently-worded aggregate, and this is explicitly the
  done-condition the task brief flags as needing a human read. Mitigated by reading the full
  rendered markdown of a **freshly generated** comparison report end to end, not just the
  already-stored one, and reasoning about every appearance of the pack's own item count (85)
  rather than pattern-matching one string.
- **Second risk: done-condition 2 (answerability stamping) and done-condition 3 (three-way
  validation split) are the two done-conditions closest to the stage's own stated defect this
  document exists to keep out of the denominator** — the accuracy figure is worthless if the
  stamp or the validation split is wrong. Mitigated by re-deriving the stamp independently (a
  fresh script driving the shipped `tools/exec.py`, not trusting `items.jsonl`'s committed
  values) and by constructing three hand-crafted specs (one per failure class) driven straight at
  the shipped executor.
- **Lower risk: done-conditions 1 and 5 (end-to-end run, `validate` passing)** — already
  demonstrated by the implementers' own stored artifacts and re-confirmed by `analyst`'s gate.
  Still re-run live here (a fresh model, fresh timestamps) rather than accepted on the stored
  artifacts' word alone, per this pass's own independence requirement — but not treated as
  high-risk, since two independent parties already observed them pass.
- **Deliberately not tested:** the statistical correctness of the Wilson/McNemar/Holm numbers
  themselves (owned by `-ml`/`data-scientist`, unchanged by S4); performance/latency of the
  scorers; `guard-judge`'s/`nlq-generator`'s actual model *accuracy* (not a done-condition — this
  stage has no pass/fail gate on model quality, per the component's own stated non-features).

## 4. Test items

| ID | Title | Preconditions | Steps | Expected result | Priority | Type |
|---|---|---|---|---|---|---|
| TP-000 | Baseline suite is green (sanity gate before new evidence) | `model-bench/` venv installed | `.venv/bin/python -m pytest -q` from `model-bench/` | Exactly one pre-existing, named, unrelated failure (`test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5`, the S5 tripwire); every other test passes | High | functional (regression baseline) |
| TP-001 | Both packs run end to end against one model (DC1) | LM Studio reachable; `host.json` fresh | `./run.sh run --pack nlq-structured-query --model <key>` and `./run.sh run --pack guard-judge-understanding --model <key>`, against a model **not** already used in a stored run | Both commands exit 0 and write a new `results/runs/*.json` record for the full item set (40 / 85 items respectively) | High | e2e |
| TP-002 | `validate` passes on both packs (DC5) | none | `./run.sh validate --pack packs/guard-judge-understanding` and `...--pack packs/nlq-structured-query` (plain, then `--strict` as a documented-deferral check) | Plain `validate` exits 0, prints `valid`, for both packs; `--strict` raises the named, pre-existing `NotImplementedError` (deliberate deferral, not a defect) on both | High | functional/contract |
| TP-003 | Every `nlq-structured-query` item is stamped `answerable: true\|false`, derived by running its reference spec through the real executor (DC2) | `reference_specs.json`, `tables.json`, `schema.json` present | Independent script re-derives each item's answerability by calling the shipped `tools/exec.py::compile_and_execute` against each item's own reference spec, exactly mirroring `run_stamp_answerability`'s rule, and diffs the result against the committed `items.jsonl` | Zero mismatches; every item carries the key; split is 34 `true` / 6 `false` (`nlq-34..nlq-39`), matching the corrected methodology ruling | High | functional/integration |
| TP-004 | The executor's validation half distinguishes a malformed spec, a schema violation, and a wrong-but-well-formed answer into three separate counts (DC3) | none | Drive `tools/exec.py` directly with (a) a structurally invalid spec (`filters` at length 5), (b) a structurally valid spec against an unregistered label, (c) a fully valid spec whose executed answer is scored against a deliberately wrong expected value via `extraction.score_pair` | (a) raises `MalformedSpecError`; (b) raises `SchemaViolationError`; (c) executes cleanly and `score_pair` returns `correct=False` — three distinguishable outcomes, none conflated | High | functional |
| TP-005 | `conflicting-facts` items score by subset containment, not set equality (DC4) | none | Call `extraction.score_pair` with `shape="conflicting-facts"` and an `expected` set that is a strict subset of `actual`; repeat the identical input with `shape="filter-list"` (negative control) | `conflicting-facts` call returns `correct=True` (containment); `filter-list` call returns `correct=False` (exact equality still enforced) — proves the exception is shape-scoped, not a global relaxation | High | functional |
| TP-006 | The guard-judge report shows both class-conditional verdict metrics side by side, with no headline number and no pooled 85-item figure anywhere (DC6) | a freshly rendered comparison report exists | Generate a **new** `compare` report from a live run against a fresh model (TP-001's run), then read the full rendered markdown end to end by eye | `falseAdvanceRate`/`falseSuspendRate` render side by side under one `## Verdicts` section; the "no headline metric" line is present; every appearance of "85" is a composition/denominator note (e.g. "40 of 85 items", "45 unscoreable in both"), never a pooled accuracy or score computed over all 85 items | High | acceptance (manual read) |

## 5. Environment & data setup

- Working directory `model-bench/` (component-local venv, `.venv/bin/python`).
- LM Studio reachable at `http://localhost:1234` (confirmed live this session); `host.json`
  re-attested before any live `run` (LM Studio's own staleness trip-wire fires otherwise — an
  operational check, not a defect).
- A model not already present in `results/runs/` is used for the fresh TP-001/TP-006 runs, so the
  evidence is independently produced rather than re-reading an implementer-produced artifact.
- TP-003/TP-004/TP-005 run against the pack's own committed `tables.json`/`schema.json`/
  `reference_specs.json` — read-only, no pack file is written by these probes.

## 6. Entry / exit criteria

**Entry:** `analyst`'s S4 code gate has landed at verdict approve (confirmed:
`docs/reviews/small-model-benchmarking-s4.md`); the component installs and the baseline suite is
runnable (TP-000).

**Exit:** all six test items (TP-001..TP-006) pass with recorded evidence: command run, output
observed. A blocked or failed item downgrades the overall verdict — S4 does not close on this
pass's word alone if TP-006 (the human-eye read) finds a wording that reads as a headline or a
pooled figure, or if any of TP-003/TP-004/TP-005's independent re-derivation disagrees with the
shipped artifacts.

## 7. Out of scope (restated)

No new automated test files are authored into the component's own `tests/` tree by this pass — the
six done-conditions are already covered by the shipped `tests/test_tools_exec.py`/
`tests/test_scoring_extraction.py`/`tests/test_scoring_classification.py` (confirmed non-tautological
by `analyst`'s gate); this pass's job is to verify the *system*, live, from the outside, not to add
to unit coverage `tdd-engineer`/`coder` already own. No performance, security, or non-functional
angle is assessed — none carries material risk for a human-started, offline-by-default benchmarking
CLI with no attack surface beyond a local LM Studio call.
