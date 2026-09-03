# Change History — model-bench

> Dated log of actual changes to the `model-bench` component. Most recent first.

## 2026-09-03 — S1 gate remediation: both blockers, all ten majors, and Rule 7

**What:** Fixed the findings of the two independent S1 gates —
`docs/reviews/small-model-benchmarking-impl.md` (`analyst`: 1 blocker, 6 majors, 7 minors, 4 nits)
and `docs/reviews/small-model-benchmarking-ml.md` (`data-scientist`: 1 blocker, 4 majors, 5 minors,
3 nits) — against plan v1.5 and method note v1.5. Test-first throughout; every fix was
mutation-tested and the reviewer's **ten surviving mutations are now all killed**.

**The two blockers.**

- **B-ML-1 — the clustered decision path did not cluster.** `verdict()`'s substitute for McNemar
  was `paired_bootstrap` over the *rows* of the paired table: an i.i.d. resample of observations
  the declared design effect says are correlated, so the interval was identical at DEFF 2, 4 and 7
  and *narrower* than the MOVER-D it replaced. It changed the instrument's name, not its interval.
  New primitive `paired_cluster_bootstrap` inflates the percentile half-widths about the point
  estimate by `sqrt(design_effect)` — the Kish variance ratio is exactly the quantity that converts
  (`-ml` §3.4 Rule 5). **This is the note's "smallest honest version", taken deliberately:** the
  structurally right fix resamples clusters of paired differences, and `PairedOutcomes` carries one
  row per analysis unit with no grouping, which could only come from a pack declaring
  `replicatesPerScript > 1` — something Rule 6 makes a validation error while only the one-level
  `cluster_bootstrap` exists. Building it now would have had no data to consume and no seam to
  reach it.
- **B-1 / M-ML-2 — Holm–Bonferroni was printed and never applied.** `report.py` called `verdict()`
  without `alpha_step`, so every metric was decided at plain Bonferroni α/k, and `holm_thresholds`
  had no step-down stop. `compare_report` now runs **two passes** — Holm is a property of the
  family, so no verdict can be decided until every p-value exists — and `holm_thresholds` is
  replaced by `holm_steps`, returning a `HolmStep` per member with its rank, threshold, `tested`
  and `rejected`. `verdict()` gained `holm_tested`, which is the stop.

**Rule 7 (`-ml` v1.5 §3.4), enforced in `verdict()` rather than left to a test.** No verdict path
returns `distinguishable` when `|diff|` is below `resolving.observable_floor`. Three decisions in
it, each with a reason:

- **It demotes and says so; it does not raise.** The note's contrast is code-versus-test, not
  raise-versus-demote, and a raise would be unreachable in practice: the √DEFF-widened bootstrap
  and McNemar's exact rejection region are different instruments that do not align by construction
  (measured — at DEFF 2 on the `(34, 6, 0, 0)` table the widened interval still excludes zero while
  15.0 pp sits below the 30.0 pp floor). The demotion renders the contradiction it resolved, which
  surfaces the defect more loudly than a traceback the report never prints.
- **It compares against the exact float, never `format_floor_pp`'s truncation** — otherwise the
  invariant inherits the presentation layer's rounding and can fire, or fail to fire, by 0.05 pp.
- **The converse is not asserted.** `|diff| >= floor` does not imply distinguishable; §3.2c's row 4
  `(20, 8, 2, 10)` is the counterexample already in the suite — 15.0 pp exactly on the α=0.05
  floor, p = 7/64, not distinguishable.

It never fires on the McNemar branch: `test_the_mcnemar_path_satisfies_rule_7_by_construction`
checks every `(b, c)` split at n ∈ {12, 20, 30, 38, 40, 48, 85} and α ∈ {0.05, 0.025}. That
asymmetry is what makes it a detector rather than a formality.

**The floor's rounding direction, per the adjudication: the floor truncates, the MDD ceilings.**
`stats.format_floor_pp` is the one place the direction lives, and it is where the report and the
verdict strings both print from — the tests assert **through the formatter**, because re-rounding
inside a test (`round(observable_floor(...) * 100, 1)`) asserts the presentation layer's arithmetic
against itself. `ResolvingPower.observable_floor` stays exact, so Rule 7's guard is not weakened.
Truncation is guarded (`math.floor(x / precision + 1e-12)`), mirroring the MDD's `- 1e-12`, and
**the guard is load-bearing rather than defensive**: `7/40 = 0.175` is `174.99999999999997` bins in
IEEE doubles, so naive truncation prints `17.4` for the α=0.025, n=38–40 row the note publishes as
**17.5**. Corrected cells: 15.8→15.7 (n=38), 7.1→7.0 (n=85), 46.7→46.6 (n=15, α=0.025); 58.3 and
23.3 were already truncations.

**The other majors.**

- `load_history` validated *after* the pack filter, so a record whose `packId` was blanked or
  deleted on disk landed in **neither** returned list — the comparison quietly lost an arm (M-1).
  The filter now drops only a record that *says* it belongs to another pack; it also applies to an
  unknown schema, whose `packId` is readable, and stays **off** `unparseable`, which cannot declare
  one (m-1).
- `RunResult.designEffect`/`basis` lost their dataclass defaults (M-2, m-ML-3, plan v1.5 §3.5). The
  legacy fallback stays in `from_dict`, where it is a reader's §3.4.3 compatibility rule.
- `BinaryMetric` gained a required `unit`, and the Arms table prints a Wilson interval only over
  the analysis unit (M-ML-3). §4.4's first mandatory consequence is verbatim *"Never print a Wilson
  interval over a turn-pooled count"*, and a turn-pooled 142/320 was printing ±5 pp where the
  honest bound is ~48.7 pp. The count is never suppressed; only the precision claim is.
- `_paired_rows` returns a `PairedRows` tally and every verdict prints it — the `asymmetry` count
  §4.3's paired corollary requires, plus rows present in one arm only and unscoreable in both
  (M-5, M-ML-4). It is printed even when nothing was dropped, because otherwise a reader cannot
  tell a shrunken `n` from a full one.
- `min_detectable_difference` raises `UnattainablePower` below `b_min(alpha)` units instead of
  converging on its bisection bracket and returning `1.0`; `ResolvingPower.mdd80` is then `None`
  and the line reads *"No difference is resolvable…"* (M-ML-1). The delivered build printed
  *"resolves differences of >=100.0 pp with 80% power"* where power is identically **zero**.
- A comparison with fewer than two arms has its own reason, and `--models` naming a key with no
  stored run exits **2** rather than silently rendering a one-arm report (M-6).
- The basis/design-effect propagation is now tested at report level (M-3), and prints the **weaker
  of the two actual bases** rather than collapsing to `assumed` — false provenance in the one
  sentence whose job is auditability (m-ML-4). The decision rule is unchanged.
- `REQUIRED_BY_SCHEMA` and `FORBIDDEN_BY_ARM_KIND` are pinned against **independently transcribed
  literals**, by name and by tier (M-4). Parametrizing over them meant deleting an entry deleted
  its test case rather than failing one.

**Minors and nits:** the unpaired label distinguishes a content-hash divergence from a version one
(m-2); `_unit_ids` is called by `_paired_rows` instead of being dead code the docstring names
(m-3); `--role` and the index's `latencyMsP95` gained tests (m-4); `PackRef.contentHash` is
`str | None` so "not yet computed" is expressible (m-5); `compare` filters by the manifest's
`packId`, not the directory name (m-6); `store()` refuses a `runId` that is not a bare filename
(m-7); the tautological assertion is gone (n-1); `Fingerprint` copies its mapping behind a
`MappingProxyType` and hashes its **values** (n-2); an absent `aggregates` block is reported as
`unparseable` rather than repaired into an empty one (n-3); the conditionality clause names the
pack's own sample noun (n-4, m-ML-5); the `-ml` §3.2c fixtures are republished at 10 dp and
asserted at the mandated **1e-9 on the proportion**, with the docstring's margin claim corrected
from four orders to three (m-ML-1); `test_z_95_matches_the_inverse_normal_cdf` records that the
pinned literal is one ULP from `NormalDist().inv_cdf(0.975)` and must not be tightened to `==`
(n-ML-3).

**One finding declined, with its reason.** n-ML-1 asked for the floor and the MDD to share a
denominator (`observable_floor` divides by the unfloored `n_effective`; `min_detectable_difference`
floors first). Unifying them would make one of the two anti-conservative: Rule 3's principle is to
round each printed bound in the direction that keeps its own claim true, and the two claims point
opposite ways — a **larger** MDD is the safe error, a **smaller** floor is. The asymmetry is now
documented at `observable_floor`, which is the one line the finding asked for.

**Two defects found by reading rendered output, not assertions** — the same discipline that caught
the CI-orientation bug at S1. The "Best case — assumes the candidate wins every…" caveat was still
printing where no MDD exists, qualifying a figure that is not on the page; and the clustered label
was appended to two of the five verdict strings rather than all of them, so a reader seeing only a
demoted verdict was never told which instrument produced it.

**Verification, from `model-bench/`:** `.venv/bin/python -m pytest -q` → **296 passed** in 2.12s,
exit 0 (0 failed, 0 skipped, 0 deselected — the `live` marker still deselects nothing because no
live test exists until S2). `.venv/bin/ruff check .` → `All checks passed!`. **34 source mutations
against a scratch copy — 34 killed, 0 survivors**, including all ten the `analyst` gate reported as
surviving and 24 new ones aimed at this change's own fixes. Two of the new ones initially survived,
both because a test asserted a passthrough field instead of the behaviour it gates; both tests were
rewritten onto cases where the mutation changes a verdict.

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
