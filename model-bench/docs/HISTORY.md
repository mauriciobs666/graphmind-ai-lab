# Change History — model-bench

> Dated log of actual changes to the `model-bench` component. Most recent first.

## 2026-09-03 — S1: fingerprint, results, stats, report, CLI (no model calls)

**What:** Built the harness core per stage S1 of `docs/plans/small-model-benchmarking.md` §4 —
everything that decides whether a number may be printed, and nothing that produces one. No model
calls, no network, no LM Studio, no pack loader: the whole S1 suite runs offline.

- `modelbench/fingerprint.py` — `Fingerprint` (frozen, `armKind`-discriminated), `FieldSpec`,
  `FieldProblem`, `REQUIRED_BY_SCHEMA` (`{schemaVersion: {armKind: {field: spec}}}`) and
  `FORBIDDEN_BY_ARM_KIND`. Fields are held in a **mapping, not dataclass attributes**, because a
  dataclass with `None` defaults collapses *absent* into *null* — the two states plan §3.4.2 exists
  to separate. `validate()` returns problems and never raises; the `deterministic` arm kind
  forbids every model field, so `{"modelKey": "bm25"}` fails loudly on write (plan §3.4.1, gate B-3).
- `modelbench/results.py` — `ItemResult`, `RunResult`, `InvalidRecord`, `BENCH_SCHEMA_VERSION = 1`,
  a **closed union** of five typed aggregate dataclasses, `store()` (raises, no bypass parameter),
  `load_history()` (returns `(valid, invalid)`, re-validating each record against **its own**
  `benchSchemaVersion`), `rebuild_index()` and `models_with_stored_results()`.
- `modelbench/stats.py` — implements `docs/plans/small-model-benchmarking-ml.md` §3.4's six binding
  rules and nothing else: `wilson_interval` (`z` keyword-only, defaulting to the pinned
  `_Z_95 = 1.959963984540054`), `mcnemar_exact`, `mover_d_interval`, `paired_bootstrap`,
  `cluster_bootstrap`, `PairedOutcomes` (duplicate-unit guard in `__post_init__`, so it holds on
  every construction route), `resolving_power`/`ResolvingPower`, `min_detectable_difference`
  (exact bisection over the McNemar rejection region, ceilinged to the printed precision, and
  taking `n_effective: float` so a raw `int` count raises `TypeError`), `observable_floor`,
  `design_effect`/`effective_n`/`width_inflation`, `verdict()` and `holm_thresholds`.
- `modelbench/report.py` — `compare_report()`: the excluded-invalid block (AC-2), the pack
  version/content-hash banners (AC-3), the `SCHEMA VERSIONS IN THIS COMPARISON` line (§3.4.3), the
  comparison-kind line (§3.7), per-arm Wilson intervals labelled *descriptive, not the comparison
  instrument*, the resolving-power line, the three verdict strings (AC-4), Holm–Bonferroni for a
  k>1 family, and the marginal-overlap diagnostic with its footnote.
- `modelbench/packs.py` — `PackRef`, `PackMetrics`, `metrics_from_manifest`,
  `check_sampling_contract`, `pack_ref_from_manifest`. **Not** S2's pack loader: no content hash,
  no AST import walk, no data-file row-count identity. `PackRef` extends Appendix A's five fields
  with `pairingKey` and `analysisUnit`, without which §3.3's analysis-unit resolution has no source.
- `modelbench/roles.py` — FR-21's five roles and `-ml` §3.3's unit-kind column.
- `modelbench/cli.py` + `modelbench/__main__.py` — `compare` (with `--negative-control`),
  `index rebuild`, `models --tested`; §3.6a's closed exit-code set. `attest`, `validate` and `run`
  are S2's and their absence is asserted by a test.
- `run.sh` — the S0 guard block deleted, as S0's own entry said S1 would.

**Two decisions taken here that the plan does not state, both additive and both flagged to
`architect`:**

- **`RunResult` gains `designEffect: float` and `basis`.** §5 test 12b requires `runner` to *set*
  `basis`, and `-ml` §3.4 Rule 4 decides which instrument may decide from it — but the plan's
  `RunResult` shape carries neither, and a report cannot recompute either after the fact. Without
  them S1 done-condition 5b is unsatisfiable. The degradation is fail-safe: any arm not
  `by-construction` drops the comparison to `assumed`, which moves the decision off McNemar.
- **`FieldProblem.reason` gains `"unknown"`** beside Appendix A's four, for a discriminator this
  build cannot interpret — an unrecognized `armKind`, or a `benchSchemaVersion` from the future.
  Forcing either into `absent`/`empty` would mislabel it.

**One defect found and fixed by reading the rendered output rather than the assertions:** when arm
B won, `verdict()` re-oriented the difference to the winner (`+66.7 pp`) but left the confidence
interval in A-minus-B orientation (`[-86.2, -29.9]`) — a positive effect printed beside a wholly
negative interval. Nothing raised; it is a plausible-looking, internally contradictory line, which
is the exact failure mode a measuring instrument must not have. The non-significant strings now
keep the signed A-minus-B difference for the same reason.

**Verification:** `.venv/bin/python -m pytest -q` from `model-bench/` → **233 passed**, exit 0
(0 failed / 0 skipped; the `live` marker deselects nothing at S1 because no live test exists yet).
`.venv/bin/ruff check .` → `All checks passed!`. `./run.sh --help` and `./run.sh models --tested`
both exit 0. Every done-condition test was mutation-tested; the load-bearing one is S1
done-condition 5(c), where pairing on the conversation id instead of the pack-declared
`analysisUnit` is caught by the captured-argument assertion independently of the raise.

## 2026-09-02 — S0: component skeleton

**What:** Created the `model-bench/` component per stage S0 of
`docs/plans/small-model-benchmarking.md` §4 — packaging, scripts, docs skeleton and an empty
package/suite. No harness code: S0's done-condition is deliberately an empty test suite, so that
S1–S8 land against a tree that already builds and lints.

- `pyproject.toml` — `requires-python = ">=3.12"`, **no runtime dependencies** (plan §3.2, a hard
  design constraint), dev extras `pytest>=9.1,<10` + `ruff>=0.14,<0.15`, ruff `select = ["E","F","W","I"]`
  / `line-length = 100` (mcp-monitor's shape), pytest `testpaths = ["tests"]` plus falkor-chat's
  live-test convention verbatim: `addopts = '-ra -m "not live"'` and a `live` marker.
- `setup.sh` — adapted from `mcp-monitor/setup.sh`: idempotent, `--recreate`, resolves paths from the
  script's own location, ends with an import smoke test.
- `run.sh` — the mcp-monitor shape (venv check, then `exec .venv/bin/python -m modelbench "$@"`) with
  an **S0 guard**: `modelbench/__main__.py` does not exist until S1, so the script reports that in
  words and exits 1 rather than `exec`-ing into a `No module named` traceback. S1 deletes the guard.
- `.gitignore` — `.venv/`, `host.json` (the operator-attested fingerprint fields, plan §3.4),
  `results/transcripts/` (raw model output: large, and not needed for any comparison, plan §3.5).
- `README.md` — what the tool is, and the three non-features stated up front: no CI/scheduler, no
  pass/fail gate, no leaderboard or cross-role aggregate.
- `AGENTS.md` — working context: current state, the hard rules (zero runtime deps, FR-23 standalone,
  no cross-role aggregate), the `live` marker, the attested fingerprint fields, and the note that
  an empty suite exits 5.
- `docs/{BACKLOG.md,HISTORY.md}` plus empty `requirements/ plans/ reviews/ test-plans/ test-reports/`
  held by `.gitkeep` files. `BACKLOG.md` is seeded with the two items plan §7 carries forward.
- `modelbench/__init__.py` (`__version__`) and `tests/test_package.py` — one install smoke test,
  asserting `modelbench.__version__` equals the installed distribution's metadata version. The plan
  called for an empty suite at S0, but pytest exits 5 (`EXIT_NOTESTSCOLLECTED`) when nothing is
  collected, so "runs and passes with zero tests collected" cannot return 0 (plan gate finding m1).
  Resolved with this one real test rather than by configuring the exit code away: a permanent
  "no tests ran is fine" setting would still be in place at S5 and would hide a collection
  breakage. The assertion is not filler — that version string is what stamps `benchVersion` into
  every run record (plan §3.4), so a skew between `pyproject.toml` and `__init__.py` fails here.
- Root `AGENTS.md` — a `model-bench/` bullet in **Structure** and a row in **Component docs**. The
  feature's requirements and plan stay at the repo root, where they were written (plan §4 S0).

**One defect found and fixed by reading the rendered output rather than the assertions:** when arm
B won, `verdict()` re-oriented the difference to the winner (`+66.7 pp`) but left the confidence
interval in A-minus-B orientation (`[-86.2, -29.9]`) — a positive effect printed beside a wholly
negative interval. Nothing raised; it is a plausible-looking, internally contradictory line, which
is the exact failure mode a measuring instrument must not have. The non-significant strings now
keep the signed A-minus-B difference for the same reason.

**Verification:** `model-bench/setup.sh` → venv created with Python 3.12.3, `model-bench[dev]`
installed (pytest 9.1.1, ruff 0.14.14), smoke import printed `model-bench 0.1.0`; re-run to confirm
idempotence. `.venv/bin/python -m pytest -q` from `model-bench/` → `1 passed in 0.01s`, exit 0
(0 failed / 0 skipped / 0 deselected). `.venv/bin/ruff check .` → `All checks passed!`.
`./run.sh --help` → the S0 guard's message, exit 1. Note that the test command must be run with
`model-bench/` as the working directory: the repo has no root pytest configuration, so from the repo
root pytest ignores this component's `testpaths` and walks the whole monorepo (measured: 9 collected,
8 collection errors, exit 2).
