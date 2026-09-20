# Small-Model Catalog Sweep — Unit C Implementation Review (`consolidate_sweep_reports.py`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M9)

**Filename note:** the dispatching brief asked for this to land as a new section inside the
existing `docs/reviews/small-model-catalog-sweep.md` (the plan review). I deviated: that document
is the review of the **plan** (an `architect` artifact); this is a review of **code** claiming to
implement one unit of it, which this repo's own filename convention (`AGENTS.md`'s closed role
set, `-impl`) and this component's own precedent
(`docs/reviews/small-model-benchmarking-impl.md`, plus a dozen more across the monorepo —
`git ls-files '*-impl.md'`) route to a separate `-impl` document on the same slug, never grown
into the plan review. Cross-referencing rather than merging.

**Scope note (added for Unit A's code gate below):** this `-impl` document now covers two units'
code gates under one file, per the coordinating brief's explicit instruction to add a new section
here rather than open a second `-impl` document — sections 1–5 above are Unit C's
(`scripts/consolidate_sweep_reports.py`, verdict approve); the "Unit A" section below is a
separate code gate, on a different diff, reviewed independently.

---

## 1. Scope & verdict

Reviewed `model-bench/scripts/consolidate_sweep_reports.py` and
`model-bench/tests/test_consolidate_sweep_reports.py` (new, 16 tests) — Unit C of
`docs/plans/small-model-catalog-sweep.md` **Version 2** (§3.5, §4/§5 Unit C) — against that plan,
`docs/requirements/small-model-catalog-sweep.md` FR-9/FR-10, and `scripts/refresh_golden.py`'s
existing house style (this component's own precedent for a "light touch," non-mutation-tested
utility script). Explicitly a light-touch, static/fixture-based review per the brief and per the
plan's own §4 Unit C note — no live `rank` output exists yet (Units A/B in-flight), so end-to-end
integration is out of scope here and expected as a follow-up once they land.

**Verdict: approve.**

**CPG:** considered, not relevant — queried `GRAPHS` directly: `cpg_falkorchat`, `kaizen_team`,
`reference`, `ws:*` are loaded; no `cpg_model-bench` graph exists. A ~270-line script plus its test
file were read in full; no call-graph tooling was needed beyond direct reading.

## 2. Findings

### 2.1 [MINOR] The unknown-pack refusal is verified at the function level only, not at the CLI (`main()`) level

The plan/brief's required refusal path — a marker naming a pack this script has no role mapping
for — is asserted only against `build_consolidated_document` directly
(`tests/test_consolidate_sweep_reports.py:280-291`, a hand-built `RankMarker`), never through
`main()`. I independently drove it through the CLI to confirm the wiring itself is correct:

```
$ report.md carries: <!-- rank-report: pack=some-future-pack metric=x top=model-a value=0.5 ci=[0.4,0.6] -->
exit code: 1
out exists: False
```

This passes today, but nothing pins it — a future change to `main()`'s try/except ordering
(e.g. reordering the `--out` mkdir before the `build_consolidated_document` call) could silently
reintroduce a partial-write bug with no test catching it. **Suggested fix:** add
`test_main_exits_1_on_unknown_pack_marker`, mirroring the existing
`test_main_exits_1_when_no_reports_carry_any_marker` shape, asserting exit code 1, the pack name in
stderr, and `not out_path.exists()`.

### 2.2 [NIT] `docs/HISTORY.md` carries no Unit C entry yet

The plan's own Documentation section (§4, "whoever lands each unit updates... `docs/HISTORY.md` (a
dated entry per unit landed)") applies to this unit; `docs/HISTORY.md` has no
`consolidate_sweep_reports.py`/Unit C entry today. The coordination ledger
(`docs/plans/small-model-catalog-sweep-coordination.md:56`) marks U3c `delivered`, so this may be
intentionally deferred to a coordinated close alongside Units A/B rather than dropped — flagging so
`teco` doesn't lose track of it before this ledger row gates closed. Not a code defect, not
gating.

## 3. Verification performed

- **Marker-parsing contract vs. plan §3.5**, including the guard-judge two-marker case: read
  `_MARKER_RE`, `parse_markers`, `load_markers` (`scripts/consolidate_sweep_reports.py:64-125`) and
  the matching fixtures (`tests/test_consolidate_sweep_reports.py:34-127`) — confirmed the
  `five_reports` fixture yields exactly 6 markers across 5 files, guard-judge contributing 2 kept
  adjacent in file order, matching `docs/plans/small-model-catalog-sweep.md:499-519` exactly.
  `.venv/bin/python -m pytest -q tests/test_consolidate_sweep_reports.py` → **16 passed**.

- **No-cross-pack-arithmetic invariant (FR-9), independently verified, not taken on trust:**
  - Read the whole script for numeric parsing: `grep -n "float(\|int(\|Decimal\|eval("
    scripts/consolidate_sweep_reports.py` matches nothing in executable code — every hit is in a
    docstring/comment describing the invariant. Every `RankMarker` field is typed `str` and only
    ever f-string-interpolated (`_index_table`, `:154-165`), never parsed.
  - Rather than trust the coder's reported mutation (temporarily edited and reverted during their
    own session, which I cannot re-inspect after the fact), I constructed an **independent
    equivalent mutant without touching the repo file**: imported the real module into a scratch
    script and monkeypatched `_index_table` in-memory to inject a genuine cross-pack sum
    (`sum(float(m.value) for m in markers)`) into the rendered document, then called the actual
    test function `test_assembler_never_combines_two_packs_values` against the mutated module.
    Result: the real test **failed** with `AssertionError: consolidated document contains a
    cross-pack sum ('1.0779999999999998')...` — confirming the guard is real, not decorative.
    Reverting the monkeypatch (i.e., the shipped code) passes clean. See Appendix for the full
    scratch script.

- **Error paths:** `main()`'s missing-`--reports`-file check (`:246-252`, exit 2) and the
  zero-marker/unrecognized-pack refusal (`ConsolidateSweepReportsError` → exit 1, `:256-260`) are
  both real (read) and both tested at the CLI level for the missing-file and zero-marker cases
  (`test_main_exits_2_on_a_missing_report_file`, `test_main_exits_1_when_no_reports_carry_any_
  marker`). The unrecognized-pack case is real (I drove it through `main()` myself, §2.1) but only
  pinned at the function level, not the CLI level — finding 2.1.

- **`_ROLE_BY_PACK_ID` design choice:** cross-checked its five entries against each pack's own
  `pack.json` `"role"` field directly (`grep -n '"role"' packs/*/pack.json`) — verbatim match on
  all five (`embedder`, `guard-judge`, `nlq-generator`, `tool-caller`, `chat-responder`), and
  against `modelbench/roles.py`'s canonical `ROLES` tuple (same five, FR-21's closed set — "there
  is no sixth, and no default"). Confirmed no existing `packId -> role` constant already exists
  elsewhere in `modelbench/` for this to duplicate (`packs.py` only ever reads `role` off a loaded
  `pack.json`, never exposes a bare mapping). The reasoning holds: this is a genuinely new,
  complete, closed-set constant, not a re-derivable duplicate, and matches this component's own
  "name a closed set explicitly" convention (`AGENTS.md`).

- **Suite/lint, run myself:**
  `.venv/bin/python -m pytest -q tests/test_consolidate_sweep_reports.py` → 16 passed.
  `.venv/bin/ruff check scripts/consolidate_sweep_reports.py tests/test_consolidate_sweep_reports.py`
  → **All checks passed!** (zero findings in Unit C's own two files).

- **The reported `ruff` finding's attribution, checked rather than assumed:** `git diff --stat`
  shows only `modelbench/stats.py` and `tests/test_stats.py` as modified (`M`); Unit C's own two
  files are new/untracked (`??`). Running `ruff check tests/test_stats.py` directly reproduces 5
  `E501` (line-too-long) findings, all inside test bodies/docstrings added by the concurrent Unit A
  work (`tests/test_stats.py:969-1031`), none touching anything Unit C wrote. Attribution confirmed
  correct.

## 4. What's solid

- Zero cross-pack arithmetic, verified two independent ways (static read + live mutation test),
  matching FR-9's letter and purpose exactly — this is the property the unit exists for, and it
  holds.
- Explicit `--reports` paths (never glob-discovered), matching the plan's stated rationale
  (no silent stale/wrong-session pickup) and consistent with `refresh_golden.py`'s own
  explicit-input philosophy.
- The refuse-rather-than-silently-partial discipline is real: no output file is written on either
  error path (confirmed for the missing-file, zero-marker, **and** unknown-pack cases — the last
  one my own addition to the check, §2.1).
- Docstrings do real work (contract citations back to plan section numbers, explicit "why not
  X" notes) without narrating obvious code — matches this component's established documentation
  density (`refresh_golden.py`).
- Test fixtures explicitly exercise the variable-marker-count shape the plan calls out as
  easy to get wrong (six markers/five files, guard-judge's two), including the
  role-count-vs-marker-count distinction (five roles from six markers).

## 5. Open questions

- None blocking. §2.2 (the pending `HISTORY.md` entry) is for `teco` to resolve at whatever point
  this ledger row is judged fully closed — possibly batched with Units A/B rather than per-unit.

## Appendix — independent mutation-test script

```python
import sys
from pathlib import Path

sys.path.insert(0, "/home/mauricio/prg/graphmind-ai-lab/model-bench/scripts")
sys.path.insert(0, "/home/mauricio/prg/graphmind-ai-lab/model-bench/tests")
import consolidate_sweep_reports as csr
import test_consolidate_sweep_reports as t

_orig_index_table = csr._index_table
def _mutated_index_table(markers, *, out_dir):
    base = _orig_index_table(markers, out_dir=out_dir)
    total = sum(float(m.value) for m in markers)
    return base + f"\n\n<!-- injected: {total} -->"
csr._index_table = _mutated_index_table

try:
    t.test_assembler_never_combines_two_packs_values(Path("/tmp"))
    print("TEST PASSED (mutation NOT caught) -- unexpected")
except AssertionError as e:
    print("TEST FAILED as expected (mutation caught):")
    print(e)
```

Output: `TEST FAILED as expected (mutation caught): consolidated document contains a cross-pack sum
('1.0779999999999998') that no single marker's own value could have produced — the assembler
combined two packs' numbers`

No repository file was edited to produce this — the mutation was applied to the imported module
object in-process only.

---

# Unit A Implementation Review — `stats.py` / `report.py` core (code gate)

**Scope.** Reviewed the uncommitted working-tree diff (`git diff`, not the implementer's
self-report) of `modelbench/stats.py`, `modelbench/report.py`,
`modelbench/scoring/classification.py` (docstring only), `tests/test_stats.py`,
`tests/test_report.py` — Unit A of `docs/plans/small-model-catalog-sweep.md` **Version 2** — against
that plan, my own two-pass plan review (`docs/reviews/small-model-catalog-sweep.md`, Pass 1 + Pass
2), and `data-scientist`'s two-pass methodology review (`docs/reviews/small-model-catalog-sweep-ml.md`,
including its Pass 2 findings Pass2-1/Pass2-2, issued after the plan's approval and *before* this
code existed — so this is the first check of whether the implementation actually picked them up).
Units B (`cli.py`) and C (`scripts/consolidate_sweep_reports.py`, reviewed above) are out of scope
and confirmed untouched by this diff (§Verification, item 6).

**Verdict: approve with suggestions** (general code-correctness/design-fit gate — statistical
soundness is `data-scientist`'s parallel remit, not mine; see the note at the end of this section
for how the two verdicts combine). Two minor findings and one test-coverage suggestion, none
blocking.

**CPG:** considered, not relevant — same as above; no `cpg_model-bench` graph exists, and this diff
(two source files + one docstring line + two test files, ~1150 lines total) needed no call-graph
tooling beyond direct reading.

## Findings

### [MINOR] `rank_report` doesn't mirror `compare_report`'s adjacent `headlineMetric ∈ verdictMetrics` guard

`compare_report` opens with two guard lines back-to-back (`report.py:1402-1406`):
`check_sampling_contract(pack)`, then `if pack.metrics.headlineMetric is not None and (... not in
pack.metrics.verdictMetrics): raise PackConfigError(...)`. `rank_report` copies the first line
(`check_sampling_contract(pack)`, confirmed present) but not the second, immediately-adjacent one —
confirmed by reading `rank_report`'s full body: no `headlineMetric`/`PackConfigError` check appears
anywhere in it.

The implementer's stated reasoning (already enforced at pack-load time, not in the plan's §5 test
list) doesn't fully hold: pack-load-time enforcement (`packs.py:151-156`,
`pack_ref_from_manifest`/`_parse_pack_metrics`) is a guarantee only for the `./run.sh rank` CLI's
own manifest-loading path — not for `rank_report` as a general function. Every test fixture that
exercises it, including the ones this very diff adds, constructs `PackRef`/`PackMetrics` directly
(`tests/conftest.py:127-140`'s `guard_pack`, `tests/test_report.py:3113-3120`'s `_rank_pack`),
bypassing that validation entirely — which is exactly why `compare_report` carries its own,
independent check on the same field despite the same load-time enforcement existing. The risk is
low-probability (only a hand-built `PackRef` with divergent headline/verdict fields, not a real CLI
call, can trigger it) but the asymmetry with `rank_report`'s own sibling function, on the identical
field, is real and the fix is a direct 4-line copy.

**Route: `coder`/`tdd-engineer` (whoever lands a follow-up), low priority.** Suggested: add the same
check immediately after `rank_report`'s `check_sampling_contract(pack)` call, and one test
mirroring `compare_report`'s own coverage of it (a `PackRef` with `headlineMetric` not in
`verdictMetrics` passed to `rank_report` raises `PackConfigError`).

### [FLAG for `data-scientist`, not an `analyst` finding] Q4's resolving-power sentence recomputes a fresh `ResolvingPower`, not a reuse of the pack's static published figure

Confirmed the mechanism exactly as the brief described, so this is reported for clarity and
precision, not because I judge its soundness — that's `data-scientist`'s call, reviewing the same
diff in parallel. `_rank_resolving_power_lines` (`report.py`, new):

```python
ns = [agg.n for r in runs for m in family for agg in [_metric_aggregate(r, m)] if agg is not None]
n_units = max(ns)
published = stats.resolving_power(
    n_units, unit_kind=unit_kind, design_effect=1.0, basis="by-construction",
    alpha_family=pack.metrics.alpha_family, alpha_mdd=pack.metrics.alpha_mdd,
)
hypothetical = stats.resolving_power(
    n_units, unit_kind=unit_kind, design_effect=1.0, basis="by-construction",
    alpha_family=pack.metrics.alpha_family, alpha_mdd=pack.metrics.alpha_family / k,
)
```

Two things worth `data-scientist` ruling on explicitly, neither of which I judge here: (1)
`design_effect=1.0`/`basis="by-construction"` are **hard-coded**, not drawn from any in-scope run's
own recorded `designEffect`/`basis` fields — meaning the "published" sentence rendered here is a
**fresh recomputation** under an always-optimistic assumption, not a literal reuse of whichever
static figure `-ml` §7.1–§7.4 already publishes for that pack (which may reflect a different,
non-1.0 design effect in a real run). Plan §3.4 Q4's text says "the pack's own already-published
single-comparison resolving power... via the existing, unmodified `resolving_power_line`" — I
confirm `resolving_power_line` itself is byte-for-byte unmodified (`git diff` shows no touch to its
body), but the `ResolvingPower` object it's handed here is newly computed, not the static one. (2)
`n_units = max(ns)` takes the **largest** aggregate `.n` across every in-scope run for the family's
metrics — an optimistic choice (the best-case sample size actually observed in this report) rather
than, say, the pack's own canonical/minimum n. Neither choice is obviously wrong, but both are real,
non-trivial judgment calls a reader could misread as "the pack's published number," and I flag them
precisely so `data-scientist` can rule on them without re-deriving the mechanism from the diff.

### [MINOR, test-coverage suggestion] No fixture exercises Rule 7's floor at `rank_report`'s new candidate-axis scale

I independently confirmed Rule 7's floor enforcement is untouched by reading the actual code, not by
inference: `verdict()`'s floor logic (`below_floor = abs(diff) < resolving.observable_floor`, the
`Rule7Violation` raise, `floor_demoted`) shares no line with the `correction_k`/precondition-3 change
(`git diff` shows the only touched lines in `verdict()` are the signature and precondition 3 itself),
and `ResolvingPower.observable_floor` is built (`stats.py:941`) from `alpha_family`/`n_eff` alone —
no `k`/`family`/`correction_k` involvement at any point. This is confirmed, not merely plausible.

What's *not* covered by a test: the two new `test_verdict_correction_k_*` tests (`test_stats.py`)
exercise the precondition mechanics alone, never combined with a floor-demoted or `Rule7Violation`
scenario; and `test_report.py`'s only floor test (`test_a_floor_demotion_is_named_in_both_the_prose_
and_the_decision_column`, pre-existing, unchanged by this diff) exercises `compare_report`'s two-arm
path, never `rank_report`'s new family renderer at a non-default `correction_k`. Since I've verified
by reading that the mechanisms are structurally independent, this isn't a live defect — but it's the
one place a future edit that accidentally threads `correction_k` into the floor computation would go
uncaught. **Route: `tdd-engineer`, cheap.** Suggested: one `rank_report` family fixture (any
single-metric pack, 17+ candidates or an engineered diff/n pair below that pack's floor) asserting
the decision cell reads `"not distinguishable — below the observable floor"` at the candidate-axis
`correction_k`, not just at the metric-axis one `compare_report`'s existing test already covers.

## Verification performed

- **The `correction_k` blocker fix, re-derived against the actual diff, not the implementer's
  report.** `git diff -- modelbench/stats.py`: `verdict()` gains `correction_k: int | None = None`;
  precondition 3 becomes `k = correction_k if correction_k is not None else len(family)`, checked
  against `resolving.alpha_mdd == resolving.alpha_family / k` — exactly the mechanism I specified in
  Pass 2 and `data-scientist` specified independently. Confirmed `metric_name in family` is
  untouched (unrelated line), confirmed the default (`None`) reproduces `len(family)` exactly via
  `test_verdict_correction_k_defaults_to_len_family_and_matches_the_omitted_path`
  (`tests/test_stats.py:567-584`, asserts `v_omitted.text == v_explicit.text`), and confirmed
  decoupling via `test_verdict_correction_k_decouples_the_divisor_from_the_familys_own_length`
  (`:587-607`, `correction_k=16` accepted against a 1-member `family`, membership check still
  independently enforced). Confirmed the one pre-existing call site
  (`compare_report`'s homogeneous-binary path) was **not** touched to pass `correction_k` — `git
  diff` shows no change near it, so it keeps relying on the default, exactly as designed. Confirmed
  Rule 7's floor logic (`stats.py:836-844`, `:941`, `verdict()`'s `below_floor`/`Rule7Violation`
  block) shares no line with this change — see the test-coverage finding above for the one gap in
  *proving* this by test rather than by reading.

- **The combined guard-judge ladder (Q3), re-derived from the actual `rank_report` body, not the
  implementer's claim of a "mutation test."** `rank_report`'s new code builds `combined_p_values`/
  `combined_cells` by iterating `for metric in family: for cand in candidates:`, appending
  `stats.mcnemar_exact(b_, c_)` **unconditionally** for every (metric, candidate) pair, then calls
  `stats.holm_steps(combined_p_values, alpha=pack.metrics.alpha_family)` **exactly once** —
  confirmed by reading the diff directly, not by trusting the report. Independently confirmed the
  regression test actually pins this:
  `test_rank_report_guard_judge_reference_family_uses_one_combined_ladder`
  (`tests/test_report.py:3396-3422`) monkeypatches `stats.holm_steps` with a call-recording spy and
  asserts `len(calls) == 1` and `len(calls[0]) == 4` (2 metrics × 2 candidates, at the fixture's
  small N) — this is a genuine mutation-catching test: reverting the implementation to two
  independent per-metric `holm_steps` calls (the plan's originally-rejected default) would make
  `len(calls) == 2` and fail it. I did not need to hand-apply that mutation to confirm this, since
  the assertion's failure mode is unambiguous from reading it directly against the single-call
  implementation.

- **The Pass2-1 pre-registration-discipline fix, re-derived.** `data-scientist`'s Pass 2 finding
  (`docs/reviews/small-model-catalog-sweep-ml.md:480-497`, "Pass2-1") required that a candidate
  missing paired data for one guard-judge metric still consume a Holm rank (`k` must not shrink from
  32/4 to 31/3). Confirmed present: the diff's comment at the append site names it explicitly
  (`# -ml review Pass2-1: k is fixed by pre-registration...`), and the mechanism is real —
  `combined_p_values.append(stats.mcnemar_exact(b_, c_))` executes unconditionally even when
  `outcomes.n_units == 0` (an empty intersection yields `b_=c_=0`, `mcnemar_exact(0,0)=1.0`, still
  appended). Confirmed by a dedicated test, not just the mechanism:
  `test_rank_report_guard_judge_family_k_does_not_shrink_when_a_candidate_has_no_paired_data`
  (`tests/test_report.py:3425-3462`) builds a candidate with disjoint item ids for `falseSuspendRate`
  (zero paired intersection with the reference), asserts `"alpha_mdd = 0.05/4"` still appears (k
  stays 4, not 3) and that candidate's row in the `falseSuspendRate` family table reads `"no verdict
  — no paired data"` rather than vanishing or silently renumbering the ladder. This is exactly the
  test `data-scientist`'s Pass 2 asked for (their own suggested "test 10a").

- **`data-scientist`'s Pass2-2 finding also picked up, unprompted.** Their suggestion (pin
  `mean_bootstrap_interval`'s `levels` at the `rank_report` call site to the plain, unadjusted 95%
  levels, never the family-corrected ones) is implemented: `_render_one_rank_table`'s continuous
  branch calls `stats.mean_bootstrap_interval(..., levels=(stats.LEVEL_CI95_LO,
  stats.LEVEL_CI95_HI), ...)` explicitly — confirmed by reading the call site directly.

- **Suite and lint, run myself, from `model-bench/`:**
  `.venv/bin/python -m pytest -q` → **1757 passed, 3 deselected** (matches the expected count
  exactly). `.venv/bin/ruff check .` → **All checks passed!**

- **Units B/C confirmed untouched by this diff, checked via `git`, not assumed.**
  `git diff --stat -- modelbench/cli.py` → empty (no changes). `git status --porcelain --
  modelbench/cli.py scripts/consolidate_sweep_reports.py` → empty (`cli.py` unmodified,
  `consolidate_sweep_reports.py` not newly created here — it's already committed as Unit C,
  `acb7e5e`, per `git log`). `git diff --stat` for the whole tree confirms only the five files named
  in the brief are touched by this diff.

## What's solid

- Both statistically load-bearing fixes from the two-pass review chain (the `correction_k` blocker,
  the Q3 combined ladder) are implemented exactly as specified, with tests that would genuinely catch
  a regression to the rejected alternative (two independent ladders) or the original blocker
  (`ValueError` at real sweep scale) — not just tests that exercise the code path.
- The implementer independently read and applied `data-scientist`'s **Pass 2** findings
  (Pass2-1, Pass2-2), issued after the plan's approval — these weren't in the plan text itself, so
  picking them up means the actual review chain was followed end-to-end, not just the plan document.
- `mean_bootstrap_interval` shares its resample engine with `paired_bootstrap` via the extracted
  `_bootstrap_means` helper (confirmed: `paired_bootstrap`'s own body now calls it too, not a
  parallel reimplementation) — exactly the "one copy of the engine, two different clamps" design
  both reviews specified.
- `_polarity_corrected`'s docstring works out the sign convention by hand for both polarity cases,
  and `test_polarity_corrected_reads_a_lower_false_advance_rate_candidate_as_better` pins both
  directions with a hand-computed example rather than asserting against the implementation's own
  output — a real behavioral test, not a tautology.
- Doc density and citation discipline match this file's established style throughout (plan section
  numbers, `-ml` section numbers, and now review-pass identifiers like "Pass2-1" cited precisely).

## Open questions

- ~~The Q4 resolving-power mechanism (flagged above) needs `data-scientist`'s explicit ruling~~ —
  **ruled while this document was being written**: `data-scientist`'s own section below (added
  concurrently, "Unit A — statistical validity") reproduces a real defect in exactly the mechanism I
  flagged — `n_units = max(...)` pools guard-judge's two metrics' differing item counts (40 vs. 30)
  into one `n_units`, silently applying the better metric's resolving power to both, understating the
  true MDD by 7.7pp for `falseSuspendRate`. Their verdict: **needs changes**. This does not change my
  own verdict above (a general code-correctness gate correctly deferred a statistical-soundness
  question rather than guessing at it), but it does mean the combined Unit A gate — my section's
  "approve" plus `data-scientist`'s "needs changes" — is **not** clean overall; see their §Unit A.5
  for the required fix before this unit is complete.
- Whether the two minor findings above (the `headlineMetric` guard, the Rule-7-at-scale fixture) are
  worth a follow-up commit now or batched with Unit B's landing, and how `data-scientist`'s required
  fix (§Unit A.5) sequences against them, is `teco`'s call.

## Verification round 2 — 2026-09-19 (`analyst`)

Second code-gate pass on the same diff, after `tdd-engineer`'s fix round. Verified against the real
`git diff`, not against the dispatching report of what was fixed.

**1. `headlineMetric ∈ verdictMetrics` guard (my suggestion 1) — landed, correct, mutation
independently reproduced.** `rank_report` (`report.py:1292-1301`) now carries the identical 4-line
shape as `compare_report`'s own check (`:1421-1425`), immediately after `check_sampling_contract
(pack)`, with a comment citing this review by name. New test
`test_rank_report_refuses_a_headline_outside_the_verdict_family` (`tests/test_report.py:3576-3591`)
hand-builds a `PackRef` bypassing pack-load-time enforcement, exactly as I specified. **I did not
take the reported mutation on trust** — reproduced it myself without touching the working tree: copied
`modelbench/` to a scratch location outside the repo, removed the guard in that copy only, loaded the
mutated `report.py` under an isolated module name (`importlib.util.spec_from_file_location`, so its
own `from modelbench import stats` etc. still resolve to the real, unmutated package — only the guard
itself is mutated), and called `rank_report` with the test's own fixture. Result: no exception raised
— confirms the real test's `pytest.raises(PackConfigError)` would fail with "DID NOT RAISE" against
the mutant, exactly as reported. `git status`/`git diff` on the real repo confirm zero footprint from
this exercise.

**2. Rule 7 floor at the new candidate-axis `correction_k` scale (my suggestion 2) — landed, correct,
mutation independently reproduced.** `test_rank_report_reference_family_floor_demotion_fires_at_the_
candidate_axis_correction_k` (`tests/test_report.py:3594-3627`) reuses `stats.py`'s own worked
floor-demotion fixture shape (`(32, 8, 0, 0)` at `designEffect=2.0`, the same numbers as
`test_rule_7_is_what_catches_the_case_the_widened_interval_still_misses` in `test_stats.py`), with
two candidates against one reference so `correction_k=2`, not the trivial `k=1` a single-candidate
fixture would exercise. Reproduced the reported mutation the same way as above (isolated scratch
copy, no working-tree edits): changed `_render_reference_family`'s
`design_effect=max(reference_run.designEffect, candidate.designEffect)`
(`report.py:1205`) to a hardcoded `design_effect=1.0` and re-ran the equivalent scenario directly.
Result: the `cand-floor` row's decision flips from `"not distinguishable — below the observable
floor"` to plain `"distinguishable"` — confirmed byte-for-byte against the mutant's own rendered row
(`| cand-floor | -20.0 pp | [-34.8, -7.1] pp | 0.0500 | distinguishable |`). Matches the reported
mutation exactly.

**3. `data-scientist`'s `n_units` fix — sanity-checked from the general-correctness/interaction
lens, as asked; statistical soundness is their call, not re-litigated here.** Read the actual
restructuring (`report.py:1223-1277`, `_rank_resolving_power_lines`; `:1399-1408`, the call site):
the function gained a `metric` parameter and now computes `ns`/`n_units` from that single metric's
own aggregates only (`[agg.n for r in runs for agg in [_metric_aggregate(r, metric)] if agg is not
None]`) — the cross-metric `max`/`min` pooling is gone entirely, not merely narrowed. The call site
moved from a single post-loop call to inside `for metric in members:`, immediately after
`_render_one_rank_table` (which still owns the FR-11 caveat and the `<!-- rank-report: ... -->`
marker internally, unchanged) and before the optional `_render_reference_family` block for that same
metric. Checked specifically for the interaction risk the brief named:
- **Ordering/placement:** each metric's own resolving-power sentence pair now prints directly below
  that metric's own table+caveat, before that metric's own reference family (if any) — a strictly
  more localized, more correct placement than the old single block after every table, not a
  regression.
- **`k` (the compound family size) is unaffected by the restructuring** — still computed from
  `len(family) * (n - 1)` outside any per-run/per-metric loop, so guard-judge's two calls both still
  report `k=4` (only `n_units` now differs between them) — confirmed via the new
  `test_rank_report_resolving_power_sentence_uses_each_metrics_own_n_not_pooled`
  (`tests/test_report.py:3541-3573`) asserting `"that family of 4 tests"` appears correctly in
  **both** metrics' sections with each one's own, different MDD figure.
- **No shared/global state**: `_rank_resolving_power_lines` remains a pure function of its arguments:
  calling it twice (once per guard-judge metric) instead of once has no accumulator or mutable
  default to interact across calls.
- **The `<!-- rank-report: ... -->` marker Unit C depends on is unaffected** — it's emitted inside
  `_render_one_rank_table`, whose own body and call site are untouched by this round's diff; Unit C's
  parser reads every matching comment line regardless of what surrounds it (confirmed in the Unit C
  review above), so relocating the resolving-power block has no bearing on that contract.
- **The new figures match `-ml` §7.3 exactly, spot-checked independently**: grepped
  `docs/plans/small-model-benchmarking-ml.md:3214-3215` myself — `falseAdvanceRate` n=40, floor
  15.0pp, MDD80 21.9pp; `falseSuspendRate` n=30, floor 20.0pp, MDD80 28.7pp — matching
  `test_rank_report_resolving_power_sentence_uses_each_metrics_own_n_not_pooled`'s own assertions
  (`>=21.9 pp`/`>=28.7 pp` published, `>=24.6 pp`/`>=32.3 pp` hypothetical at k=4) and
  `data-scientist`'s own reproduction numbers cited in their §Unit A.5. I am not re-ruling on
  whether `n_units = max(ns)` **within** one metric (across models in the ranked table) is itself the
  right choice — that's unchanged from before this round and is `data-scientist`'s methodology
  question, not a new interaction defect from the restructuring.

No new interaction defects found. Both of my own suggestions are correctly and completely resolved,
with mutation claims independently reproduced rather than trusted.

**Suite and lint, re-run myself, from `model-bench/`:**
`.venv/bin/python -m pytest -q` → **1760 passed, 3 deselected** (up from 1757 — the three new tests
above account for the delta exactly). `.venv/bin/ruff check .` → **All checks passed!**
`git diff --stat -- modelbench/cli.py` and `git status --porcelain -- modelbench/cli.py scripts/
consolidate_sweep_reports.py` → both empty again — Units B/C still untouched.

**Final verdict, this round: approve. From my lens (general code-correctness, design-fit, and the
specific structural interaction the brief asked me to check), Unit A's code gate is closed — I have
no further findings and no open items of my own.** This is an explicit stopping signal for my part of
the gate, not a provisional one. The one thing outside my lens that still gates the unit as a whole is
`data-scientist`'s own independent re-verification of the `n_units` fix's statistical correctness (in
progress in parallel, per the coordinator's message) — I checked the fix's mechanics, loop
interaction, and the figures it now prints against `-ml`'s own published numbers, and found all of
that sound, but the final statistical sign-off is theirs to give, not mine.

---

## Unit A — statistical validity (`data-scientist`)

Reviewed the uncommitted working-tree diff of `modelbench/stats.py`, `modelbench/report.py`,
`modelbench/scoring/classification.py`, `tests/test_stats.py`, `tests/test_report.py` — Unit A of
`docs/plans/small-model-catalog-sweep.md` Version 2, implementing the shared blocker and Q1/Q3/Q4
from `docs/reviews/small-model-catalog-sweep-ml.md` (Pass 1 and Pass 2). Focus is statistical
faithfulness to what those reviews specified, verified against the real code and by independent
reproduction, not against the implementer's own report of what they did. `analyst`'s parallel
general code-gate on the same diff is not duplicated here.

**Verdict: needs changes.** One real, reproduced defect (§Unit A.5 below) in the Q4 resolving-power
sentence's `n_units` construction, which will print a silently-wrong, overly-optimistic number for
the guard-judge pack specifically — the one pack this defect can fire on, and one of the four packs
this sweep actually targets with FR-8's family. Everything else checked (the blocker fix, Q1, Q3,
FR-11 figures) is correctly and faithfully implemented, confirmed by reading the real code, not by
trusting either review's own changelog.

### Unit A.1 — `correction_k` (the shared blocker), confirmed faithful

`stats.py`'s diff adds exactly the specified signature —
`correction_k: int | None = None`, keyword-only, and precondition 3 rewritten as
`k = correction_k if correction_k is not None else len(family)` checked against
`resolving.alpha_mdd == resolving.alpha_family / k` (`stats.py:1219-1275`, reading the actual diff,
not the docstring). `metric_name in family` and Rule 7's floor enforcement are untouched — one copy
of the floor check, as required. `tests/test_stats.py:567-610` pins both the default-reproduces-
today's-behavior path and the `correction_k=16` decoupled path explicitly, matching the "assert
both the passing and the omitted path" obligation from my Pass 1 review verbatim.

### Unit A.2 — Q1 (`mean_bootstrap_interval`), confirmed faithful, and its `levels` cannot regress by construction

- **Clamps directly to the metric's own support, never a difference-support conversion.** Read
  `mean_bootstrap_interval`'s body (`stats.py`): `if support is None: return lo, hi; return
  max(support[0], lo), min(support[1], hi)` — a direct clamp on the *result*, no `(lo - hi, hi -
  lo)` conversion anywhere in this function. `tests/test_stats.py:975-992` proves the direction
  empirically, not just by code inspection: a support of `(0.45, 0.55)` narrower than the raw
  resample range clamps to exactly `(0.45, 0.55)`, where a difference-support conversion would
  instead have clamped to `(-0.1, 0.1)` — a visibly different, wrong pair the test explicitly rules
  out.
- **Shares the resample engine via `_bootstrap_means`, not a second implementation.**
  `paired_bootstrap`'s own resample loop is now `means = _bootstrap_means(diffs, B=B, seed=seed)`
  and `mean_bootstrap_interval` calls the same helper. `tests/test_stats.py:945-972` is a genuinely
  strong test here, not a coincidence-prone one: it monkeypatches `_bootstrap_means` with a spy and
  asserts `mean_bootstrap_interval` actually calls it with the caller's own `(data, B, seed)` —
  ruling out the failure mode the test's own docstring names (two different resample loops landing
  on the same percentile bound by coincidence). I re-ran this test in isolation
  (`.venv/bin/python -m pytest tests/test_stats.py -k mean_bootstrap_interval -q`) — passes.
- **`levels` cannot regress to `continuous_verdict`'s family-adjusted values by construction, not
  just by convention** — confirmed exactly as the brief asks me to verify, not merely to accept.
  `rank_report`'s own public signature (`report.py:1269-1276`) has **no `levels` parameter at
  all** — there is no path from any external caller (CLI, another `report.py` function, a test) down
  to `mean_bootstrap_interval`'s `levels` argument except the one hardcoded literal inside
  `_render_one_rank_table`: `levels=(stats.LEVEL_CI95_LO, stats.LEVEL_CI95_HI)` (`report.py:1123`).
  Changing this value requires editing that literal, not passing a different argument through an
  exposed seam — this is the "by construction" property the brief specifically asked me to
  distinguish from "by convention," and it holds.

### Unit A.3 — Q3 (combined ladder), confirmed faithful

- `_render_reference_family`/`rank_report`'s new code builds `combined_p_values`/`combined_cells`
  by iterating `for metric in family: for cand in candidates: ...`, then calls
  `stats.holm_steps(combined_p_values, alpha=pack.metrics.alpha_family)` **exactly once**
  (`report.py:1358-1377`) — never once per metric. `correction_k = len(combined_p_values)` is
  computed from the same flattened list and threaded into every `stats.verdict()` call inside
  `_render_reference_family` (`report.py:1192,1211`).
- **Composition with the blocker fix, verified concretely, not just assumed to typecheck.** For
  guard-judge, `correction_k = len(family) * len(candidates) = 2·(N-1)`; for every other pack,
  `= 1·(N-1) = N-1`. `resolving.alpha_mdd = pack.metrics.alpha_family / correction_k` is computed
  fresh per candidate row (`report.py:1192`) and passed straight into `resolving_power`, so
  precondition 3 is satisfied by construction for whichever magnitude applies — I don't have to
  take this on faith: `tests/test_stats.py:587-610`'s `correction_k=16` case and
  `tests/test_report.py:3396-3422`'s guard-judge fixture (asserting `len(calls) == 1`,
  `len(calls[0]) == 4`, and the literal string `alpha_mdd = 0.05/4` in the rendered output) pin both
  magnitudes directly, and I re-ran both (`pytest -k "correction_k or combined_ladder" -q`) —
  passed.
- **Pre-registration discipline for a missing-data candidate (my own Pass 2-1 suggestion) — landed,
  not merely acknowledged.** `report.py:1364-1372`'s comment cites "review Pass2-1" by name and
  unconditionally appends `stats.mcnemar_exact(b_, c_)` for every `(metric, candidate)` pair
  regardless of whether `outcomes.n_units == 0`, mirroring `compare_report`'s own existing
  empty-intersection handling. `tests/test_report.py` carries a dedicated test for this
  (`test_rank_report_guard_judge_family_k_does_not_shrink_when_a_candidate_has_no_paired_data`,
  `:3425` on) — the suggestion from my Pass 2 review was acted on, not just noted.

### Unit A.4 — FR-11 figures, confirmed faithful

`_RANK_CAVEATS["guard-judge-understanding"]` and `["tool-caller-shop-assistant"]`
(`report.py:1012-1031`) carry the exact concrete figures cited in my Pass 1 review: guard-judge
"floor 15.0/20.0 pp, MDD80 21.9/28.7 pp for falseAdvanceRate/falseSuspendRate at the two-member
alpha_mdd=0.025"; tool-caller "floor 50.0 pp, MDD80 57.8 pp" — not just the qualitative shape. These
strings are rendered via `_rank_caveat_lines`, called once per ranked table
(`_render_one_rank_table`, `report.py:1095`), so guard-judge's two tables each carry the full,
correct two-metric caveat above them — confirmed by reading the call site, and consistent with
`tests/test_report.py`'s fixtures asserting the caveat block's literal text.

### Unit A.5 — Q4, the real judgment call: `n_units = max(...)` overstates power for a two-metric family — reproduced, not just argued

This is the item the brief specifically flagged as the implementer's own non-trivial resolution and
asked me to rule on directly. **My ruling: not statistically defensible as implemented — it
overstates what the design can prove, and I reproduced the exact failure it will ship.**

**The code.** `_rank_resolving_power_lines` (`report.py:1223-1266`) is called **once per pack**
(`report.py:1389`, outside the `for metric in members` loop), and pools **both** of guard-judge's
verdict metrics into a single `n_units`:

```python
ns = [agg.n for r in runs for m in family for agg in [_metric_aggregate(r, m)] if agg is not None]
n_units = max(ns)
```

For guard-judge, `family = ["falseAdvanceRate", "falseSuspendRate"]`, whose real, already-published
`-ml` §7.3 item counts are **40** and **30** respectively (the same asymmetry `_RANK_CAVEATS`'s own
guard-judge string states two lines above this one — and matching the actual pack, not a fixture
artifact). `max(40, 30) = 40` is used for **both** the "published" sentence (which claims to
reproduce the pack's own already-published `-ml` §7.1–§7.4 figure) and the "hypothetical" FR-8
family-cost sentence — silently applying `falseAdvanceRate`'s own better (n=40, less-constrained)
resolving power to a report section that also, two tables away, publishes `falseSuspendRate`'s own
worse (n=30) figure.

**Reproduced against the real pack shape, not a symmetric test fixture.** I built three guard-judge
runs with the pack's actual asymmetric item counts (40 `clear_suspend` items scoring
`falseAdvanceRate`, 30 `clear_advance` items scoring `falseSuspendRate` — `-ml` §7.3's own split)
and called `rank_report` directly, no monkeypatching:

```
This pack resolves differences of >=21.9 pp with 80% power at n=40 effective items (40 units,
design effect 1.00, by-construction, alpha=0.025). Differences below 15.0 pp cannot reach
significance at any observed outcome, at any Holm step (alpha <= 0.05). ...

If this pack's optional reference-anchored family (FR-8) were run ... that family of 4 tests would
resolve differences of >=24.6 pp with 80% power (alpha_mdd = 0.05/4).
```

Both numbers are `falseAdvanceRate`'s own (n=40) figures. The report never states
`falseSuspendRate`'s own, worse figures in this sentence pair — confirmed independently with
`stats.resolving_power` called directly at `n=30, alpha_mdd=0.05/4`: **floor 20.0 pp, MDD80 32.3
pp**, not 15.0/24.6. The hypothetical sentence specifically understates the true cost of the
compound family by **7.7 percentage points** on the MDD — a reader who trusts this sentence's
number for `falseSuspendRate`'s half of the family would believe an 8-point-smaller difference is
detectable than the design actually supports.

**Why the test suite didn't catch this.** `_guard_judge_arm` (`tests/test_report.py:3380`, the
helper every guard-judge fixture in the suite uses, including
`test_rank_report_no_reference_resolving_power_sentence_doubles_k_for_guard_judge`) takes a single
`total: int = 40` shared by both metrics — every fixture in the suite gives `falseAdvanceRate` and
`falseSuspendRate` the **same** n, so `max(ns) == min(ns)` always in every test that exists today,
and the discrepancy is invisible to the suite. This is not a fixture bug to fix in isolation —
it's a sign the code path the fixture exercises was never driven with the pack's own real shape.

**Why this specific implementation is the wrong direction, not merely imprecise.** Both the floor
and the MDD get *worse* (larger) as `n` shrinks (`-ml` §3.4 Rule 3's whole point: the denominator
that keeps a bound's own claim true). Picking `max(ns)` — the *larger*, easier n — is the
*optimistic* direction, compounding onto a sentence pair whose MDD clause is already labelled "Best
case" for a different reason (strict-dominance). An honesty statement about "what this sample size
can and cannot prove" (the exact framing AC-6 and my own Pass 1 §6 recommendation both use) needs
the *conservative* bound, not the favorable one — the entire resolving-power apparatus in `-ml`
exists specifically to prevent a report from stating power it does not have.

**The fix is not "use `min(ns)` instead."** A single pooled sentence — even at the conservative
`n`, which would at least not overstate `falseSuspendRate`'s own power — would then *understate*
`falseAdvanceRate`'s genuinely better figure, which is equally a misrepresentation of "the pack's
own already-published" result (`-ml` §7.3 publishes **two** rows for exactly this reason). The
correct construction mirrors what `_rank_caveat_lines`/`_render_one_rank_table` already do for the
FR-11 caveat: **print the published+hypothetical sentence pair once per metric in `family`**
(looping the way `_render_one_rank_table` already loops over `members`), each using that metric's
own `n`. For a single-headline-metric pack this produces exactly what exists today (one sentence
pair, since `family` has one member); only guard-judge's report changes shape, gaining a second,
distinct sentence pair under its second ranked table — which is also where a reader would expect
to find it, immediately below the table whose own caveat already states that metric's own figures.

**Severity: major, not a blocker for Units B/C, but must be fixed before this ships for the real
sweep** — guard-judge is one of the four packs FR-8 actually targets, its asymmetric 40/30 split is
already documented (`-ml` §7.3, and this very diff's own `_RANK_CAVEATS` string two lines above the
bug), and the defect fires on the very first render of a real guard-judge ranked report without
`--reference` — not a contrived edge case. This is the "wrong-but-plausible-looking number" failure
mode this component's own convention names as never to ship silently.

**Required before Unit A is complete:**
1. Restructure `_rank_resolving_power_lines` to compute and print one published+hypothetical
   sentence pair per member of `family` (not pooled via `max`/`min` across metrics), called from
   inside or parallel to `rank_report`'s existing `for metric in members` loop.
2. Add a guard-judge fixture with the pack's real asymmetric n (40/30, not `_guard_judge_arm`'s
   uniform `total`) to `tests/test_report.py`, asserting each metric's own sentence pair states its
   own figures (falseAdvanceRate: floor 15.0/MDD 21.9-at-k=4-recomputed; falseSuspendRate: floor
   20.0/MDD 32.3) — this is the test that would have caught the defect and must survive it.

### Unit A.6 — suite and lint

```
$ .venv/bin/python -m pytest -q
1757 passed, 3 deselected in 7.49s

$ .venv/bin/ruff check .
All checks passed!
```

Both clean. Note this does **not** cover Unit A.5's defect — no existing test exercises an
asymmetric-n guard-judge fixture, so nothing red signals it; the suite passing is expected and does
not contradict the finding above.

### Unit A.7 — 2026-09-19 re-verification: Unit A.5 fix, confirmed correct

Re-derived independently against the real diff, not the implementer's report.

1. **`n_units` is genuinely metric-specific now, confirmed by reading the code.**
   `_rank_resolving_power_lines` gained a required `metric: str` parameter (`report.py:1223-1224`)
   and its `ns` comprehension now reads `_metric_aggregate(r, metric)` for the one `metric` passed
   in, never looping `for m in family` — the cross-metric pooling is gone, not merely relabeled
   (`report.py:1254`). The call site moved inside `rank_report`'s existing `for metric in members:`
   loop (`report.py:1399-1402`), immediately after `_render_one_rank_table` for that same metric —
   exactly mirroring where the FR-11 caveat already sits, as I asked for.

   I reproduced the real pack shape again (not trusting the new fixture in isolation): built three
   guard-judge runs with the actual 40/30 asymmetric split and called `rank_report` directly.
   Extracted figures, independently re-derived, **match exactly** what I computed in Unit A.5 and
   what the new fixture asserts:

   | metric | n | floor | published MDD80 | hypothetical MDD80 (k=4) |
   |---|---|---|---|---|
   | `falseAdvanceRate` | 40 | 15.0 pp | 21.9 pp | 24.6 pp |
   | `falseSuspendRate` | 30 | 20.0 pp | 28.7 pp | 32.3 pp |

   No cross-contamination — `falseSuspendRate`'s section contains none of `falseAdvanceRate`'s
   numbers and vice versa, confirmed by string search over the actual rendered markdown, not by
   reading the assertions alone.

2. **`k` is unchanged — still the full compound family size, confirmed.** `_rank_resolving_power_lines`
   still computes `k = len(family) * (n - 1)` from the full `family` parameter (still passed in
   full, not narrowed to `[metric]`) — `report.py:1243-1250`'s own docstring states this explicitly
   and the code matches it. Both metrics' hypothetical sentences read `alpha_mdd = 0.05/4` and
   `"that family of 4 tests"` in my reproduction above — the correction axis (candidate × metric,
   `2·(N-1)=4`) is untouched; only the per-metric `n_units` feeding each metric's own two
   `resolving_power` calls changed, exactly as scoped.

3. **Mutation check, run myself, not taken on the implementer's word.** Monkeypatched
   `modelbench.report._rank_resolving_power_lines` in-process (no repository file touched) back to
   the original pooled-`max(ns)`-across-`family` implementation, then ran the new test
   (`test_rank_report_resolving_power_sentence_uses_each_metrics_own_n_not_pooled`) against the
   mutated module: **it failed**, exactly where expected — the negative assertion
   `assert "n=40 effective items" not in suspend_section` is what catches it, since the pooled
   implementation leaks `falseAdvanceRate`'s n=40 into the `falseSuspendRate` section. Reverting the
   monkeypatch (i.e. the shipped fix) passes clean.

4. **Suite and lint, re-run.**

   ```
   $ .venv/bin/python -m pytest -q
   1760 passed, 3 deselected in 7.69s

   $ .venv/bin/ruff check .
   All checks passed!
   ```

   (1760, not 1757 — the three new tests: the asymmetric-n regression fixture and the two
   `analyst`-suggested tests visible elsewhere in this document's Unit A findings.)

**Verdict, restated for this round: approve.** The Unit A.5 defect is fixed, correctly and at the
right granularity (per-metric, not merely re-pooled in the conservative direction), the fix
composes cleanly with `k`/`correction_k` (unaffected, confirmed), and the regression test is
genuinely load-bearing (mutation-caught, not decorative). Combined with Unit A.1–A.4's findings
(already confirmed correct in the first round and unaffected by this fix), I consider **Unit A
closed from the statistical-validity lens** — no further re-check needed from me unless a future
change touches `stats.py`'s `verdict`/`resolving_power`/`mean_bootstrap_interval` contracts, the
combined-ladder construction, or the per-metric resolving-power sentence again.

---

# Unit B Implementation Review — CLI wiring (`modelbench/cli.py`) (`analyst`)

**Scope.** Reviewed the uncommitted working-tree diff of `modelbench/cli.py`, `tests/test_cli.py`,
`README.md`, `docs/HISTORY.md` (U183) — Unit B of `docs/plans/small-model-catalog-sweep.md`
**Version 2** §3.3/§4/§5 — against that plan's exact spec for the `rank` subcommand (`--build_parser`
registration, `_select_rank_arms`, `_rank_report_path`, `_cmd_rank`'s structure). Moderate-depth
gate, matching the plan's own framing of this unit as "fully specified, mechanical" reuse of Unit
A's already-gated `rank_report()` — not the mutation-testing-grade bar Unit A/the `correction_k`
change got. `modelbench/stats.py`, `modelbench/report.py`, `scripts/consolidate_sweep_reports.py`
are out of scope and confirmed untouched by this diff (see Verification).

**Verdict: approve with suggestions.** No blockers. Structural mirroring against `compare` holds,
both judgment calls the brief asked me to scrutinize are correctly resolved, and the 13 new tests
exercise real CLI behavior (arg parsing, exit codes, real file writes) rather than mocks. Two minor
findings and one nit, none gating.

**CPG:** considered, not relevant — `GRAPHS` lists `cpg_falkorchat`, `kaizen_team`, `reference`,
`ws:*`; no `cpg_model-bench` graph exists, and this diff (~320 changed lines across four files)
needed no call-graph tooling beyond direct reading.

## Findings

### [MINOR] `rank` has no test for the "pack exists but zero in-scope arms" path

`rank_report` handles zero stored runs gracefully by design (renders `"No in-scope model has a
stored, consistent result for <metric> in this report."` per table, exit `0`) — I confirmed this
directly, not by inference, by driving `main(["rank", "--pack", "guard-judge-understanding",
"--root", <empty root>])` against a real empty `results/runs/` directory: exit `0`, a report is
written, no exception. So this is **not a live defect** — the behavior is correct and matches the
"no score-driven exit code" philosophy `compare` already follows. But the brief's own point 3 (walk
every failure/edge path against the closed exit-code set) names exactly this path, and nothing in
the 13 new tests pins it: `test_rank_session_restricts_the_arm_set_to_that_session` only checks a
*non-empty* filtered set, and there is no `--session` that matches nothing, nor a "no stored runs at
all" fixture. **Suggested fix:** one cheap test, e.g. `test_rank_with_no_stored_runs_still_exits_
zero`, asserting `code == 0` and a report is written — cheap insurance against a future change to
`_cmd_rank` or `rank_report`'s zero-run branch regressing to a crash or a wrong exit code silently.

### [MINOR] README's `rank` paragraph doesn't document the unknown-`--reference`-key exit code

`README.md`'s `rank` paragraph (lines 119–129) documents `--footprints`' malformed-file exit `2`
explicitly, but says nothing about what happens when `--reference` names a model key with no stored
run — a real, tested (`test_rank_exits_two_naming_an_unknown_reference_model`), user-reachable
failure mode with the same exit code (`2`) as the footprints case one sentence away. A reader of
just this paragraph (as opposed to the closed exit-code table two paragraphs below, which only
states the *generic* "`2` bad arguments/usage" without naming this specific case) would not know an
unstored `--reference` key fails this way rather than, say, silently rendering the table without a
family. **Suggested fix:** one clause after the `--reference` sentence, e.g. "naming a model key
with no stored run for this pack is a usage error, exit `2`, and nothing is written" — matching the
footprints sentence's own level of detail immediately after it.

### [NIT] `docs/HISTORY.md`'s U183 entry doesn't name the `headlineMetric ∉ verdictMetrics` exit-4 case explicitly in its own prose

The entry's last sentence lists the exit-4 cases tested ("an unknown pack and a `headlineMetric ∉
verdictMetrics` manifest each exiting `4`") — this is accurate, so this is not a defect, just a
note that the phrasing groups two different failure origins (an unloadable manifest vs. a
structurally-invalid-but-loadable one) under one clause. Not worth a rewrite on its own; flagging
only because a future reader diffing this entry against `_cmd_rank`'s two separate `except` blocks
might wonder if they're the same code path (they're not — one is `pack_ref_from_manifest`'s own
raise, the other is `rank_report`'s new guard covered by the round-2 Unit A fix above). No action
needed.

## Verification performed

- **Structural mirroring against `compare`, read side by side.** `_build_parser`: `rank`'s
  subparser is registered with the same `with_root` wrapper, `--pack` required identically to
  `compare`'s, `--out` present identically; the new `rank` choice appears in `--help`'s subcommand
  set, pinned by the updated `test_validate_and_run_are_now_recognized_commands`
  (`{compare,rank,index,models,attest,validate,run}`). `_cmd_rank`'s flow (manifest check → pack
  load → `load_history` → select arms → render in a `try`/`except` → write to `reports/` and
  stdout) matches `_cmd_compare`'s shape exactly, with two justified, commented divergences: an
  extra `_load_footprints` step (no `compare` equivalent — `compare` has no footprints concept) and
  a second `except ValueError` clause after `except PackConfigError` (covering `rank_report`'s own
  reference-not-found/`DuplicateModelInReport` raises, which `compare_report` has no equivalent
  of). Neither divergence is surprising given what each command actually does.

- **The exit-2-for-unknown-`--reference` judgment call, checked against `README.md`'s closed exit-
  code set.** `README.md`'s own table: `2` "bad arguments/usage," `4` "an invalid pack... an unmet
  `environment.requires`... a `callSurface` the model's catalog type contradicts... (after `run`'s
  artifacts are already written) a `tool-caller` conversation censored by a tool-dispatch failure."
  Exit 4's category is entirely about the **pack manifest/environment** being wrong; naming a
  `--reference` value that happens not to be stored is a property of the **argument the user typed**,
  not the pack — squarely exit 2's "bad arguments/usage," not exit 4's category. Confirmed the
  implementation matches: `_cmd_rank`'s `except ValueError` maps `rank_report`'s reference-not-found
  raise to `EXIT_USAGE` (`2`), and `test_rank_exits_two_naming_an_unknown_reference_model` pins it
  end to end (exit `2`, `"nope"` in stderr, nothing written to `reports/`) — I re-ran this test in
  isolation (`.venv/bin/python -m pytest tests/test_cli.py -k rank_exits_two_naming -q`) and drove
  the CLI myself with a fresh empty root to confirm no report file appears on disk after a `2` exit.

- **The dedup-parity judgment call, verified against the actual code, not the docstring's claim.**
  `_select_arms`'s `--models` path: `by_key = {r.modelKey: r for r in candidates}` (line 209),
  last-value-wins by Python dict-comprehension semantics. `_select_rank_arms`: `by_key =
  {r.modelKey: r for r in candidates}` (line 240) — byte-identical shape. `load_history`
  (`modelbench/results.py:1014`) iterates `sorted(directory.glob("*.json"))`, i.e. ascending by
  filename, so both functions' "last occurrence wins" resolves to the alphabetically-last (and, by
  this codebase's `runId`/filename convention, newest-timestamped) stored file for a given
  `modelKey`. No inconsistency between `compare`'s and `rank`'s dedup semantics. Also ran
  `test_rank_dedupes_a_repeated_model_key_keeping_the_newest_stored_run` in isolation — passes, and
  its assertions (`30/40` present, `10/40` absent) genuinely exercise which record survives, not
  merely that dedup happened.

- **Exit codes walked against every path `_cmd_rank` can take**, cross-checked one by one against
  `README.md`'s closed set: missing manifest file → `4` (`test_rank_an_unknown_pack_exits_four`);
  `pack_ref_from_manifest` raising (bad/inconsistent manifest, incl. `headlineMetric ∉
  verdictMetrics` at load time) → `4` (`test_rank_a_headline_outside_the_verdict_family_exits_four`);
  malformed/non-object `--footprints` file → `2`, both cases tested
  (`test_rank_exits_two_on_an_unparseable_footprints_file`,
  `test_rank_exits_two_when_the_footprints_file_is_not_a_json_object`); unknown `--reference` → `2`
  (above); normal render, any ranking → `0`, including the zero-arms case I drove by hand (finding
  above — correct behavior, undertested). No path lands on an ad-hoc code outside `{0, 2, 4}` — `3`
  (LM Studio) and `5` (fingerprint) are not reachable from `rank` at all, correctly, since `rank`
  never touches LM Studio or `host.json`.

- **Tests read in full (13 new), confirmed real rather than mocked.** Every new test drives
  `main([...])` (the actual CLI entry point) against a real `workspace` fixture with real files
  under a real `tmp_path`-backed root, asserts real process exit codes, real stdout content
  (`capsys`), and real files on disk (`(workspace / "reports").glob("*.md")`) — no
  `unittest.mock`/monkeypatch anywhere in the new test bodies. The footprints tests write a real
  JSON file and assert the rendered markdown cell; the dedup test mutates a real stored JSON
  record's `modelKey` field on disk rather than constructing an in-memory double. This is the same
  house style the rest of `test_cli.py` already uses.

- **Scope claim, verified via `git`, not assumed.** `git diff --stat -- modelbench/stats.py
  modelbench/report.py scripts/consolidate_sweep_reports.py` → empty; `git status --porcelain` on
  the same three paths → empty. Confirms Unit B's diff genuinely touches only `cli.py`,
  `test_cli.py`, `README.md`, `docs/HISTORY.md`, matching the coder's stated scope.

- **`README.md`/`docs/HISTORY.md` read for accuracy against actual behavior.** Both are accurate in
  substance (verified above); the two gaps found (README's exit-2 omission, HISTORY's grouped
  phrasing) are precision nits, not overstatement/understatement of what `rank` does — neither
  claims a capability the code lacks or omits one the code has.

- **Suite and lint, run myself, from `model-bench/`:**
  `.venv/bin/python -m pytest -q` → **1773 passed, 3 deselected** (exact match to the brief's
  expectation; `1760 + 13` new). `.venv/bin/ruff check .` → **All checks passed!**

## What's solid

- Structural mirroring against `compare` is genuine, not superficial — same registration pattern,
  same flow shape, same never-overwrites same-day-sequence filename discipline
  (`-rank-` infix keeps its own counter, confirmed by
  `test_a_same_day_rank_rerun_does_not_overwrite_the_earlier_one_or_collide_with_compare`).
- Both judgment calls the brief flagged for independent scrutiny (exit-2 for an unstored
  `--reference`, dedup-keeps-newest parity with `compare`) are correctly resolved and match the
  documented closed exit-code set and `_select_arms`'s own precedent exactly.
- Test quality is real: exit codes, file-writing, and rendered content are asserted against the
  actual CLI entry point and actual files on disk, not mocks — the dedup test in particular mutates
  a real stored record's `modelKey` rather than fabricating a convenient double.
- `docs/HISTORY.md`'s U183 entry and `README.md`'s new `rank` documentation are substantively
  accurate against the real diff — no overclaim, no silent omission of a real behavior.
- Scope discipline held: `stats.py`, `report.py`, and `scripts/consolidate_sweep_reports.py` are
  genuinely untouched, confirmed via `git`, not taken on the coder's word.

## Open questions

- None blocking. Whether the two minor findings (the zero-arms test, the README exit-2 clause) are
  worth a follow-up commit now or batched with a later documentation pass is `teco`'s call.

---

# Defect fix code gate — `nlq`/`rank` n=0-vs-no-aggregate naming (`analyst`, 2026-09-20)

**Scope.** Reviewed the uncommitted working-tree diff (`git diff HEAD -- modelbench/report.py
tests/test_report.py`) that fixes Defect 1 from `docs/test-reports/small-model-catalog-sweep-report.md`:
`rank_report`'s ranked table silently dropped `qwen/qwen3-4b-thinking-2507` and
`stable-code-instruct-3b` from `nlq-structured-query`'s table with no mention anywhere, because a
run's real, internally-consistent `n=0` aggregate (100% item parse-failure) rendered identically to
a run declaring no aggregate at all. `tdd-engineer`'s fix adds `_zero_n_arms`/`_rank_zero_n_lines`
and a one-line change to `_metric_value`'s `ContinuousMetric` branch. Verified against the real diff
and the real code around it (`_metric_aggregate`, `_aggregate_item_mismatches`,
`_render_one_rank_table`, `BinaryMetric.rate`), not against the fix's own docstring claims, and
against `AGENTS.md`'s honesty-rule invariants (the exclude-and-name rule, "an item's outcome is
declared, never inferred"). Also spot-checked all five regenerated `*-20260920-02.md` reports.

**Verdict: approve with suggestions.** The root-cause diagnosis is correct, the fix is minimal and
correctly composes with the pre-existing `INVALID RESULTS EXCLUDED` (AC-2) banner (provably, not
just plausibly, mutually exclusive with it), and the pinning test is a genuine mutation-catcher for
the exact reported scenario. Two minor, non-blocking test-coverage gaps, found by mutation, not
inference.

**CPG:** considered, not relevant — no `cpg_model-bench` graph exists (`GRAPHS` unchanged from prior
sections' checks); a ~60-line diff across two files needed no call-graph tooling.

## Findings

### [MINOR] The `ContinuousMetric` half of the fix is untested — reverting it passes the whole suite

`_metric_value`'s new guard (`report.py:146-147`, `return agg.mean if agg.n else None`) is necessary
for symmetry with `_zero_n_arms` (which treats `BinaryMetric`/`ContinuousMetric` identically at
`report.py:165`) — without it, a `ContinuousMetric` with `n=0` would be *named* as excluded by the
new banner **and still rendered** in the ranked table with a meaningless mean, which is the exact
contradiction this fix exists to prevent. But no test exercises it: I copied `modelbench/` to a
scratch location, reverted this one line to the pre-fix `return agg.mean`, and ran the full
`tests/test_report.py` against the mutant — **156 passed**, zero failures (see Appendix). No shipped
scorer currently constructs a `ContinuousMetric(n=0, ...)` either (`retrieval.py:476-486` returns
`mrr=None` outright when `scored` is empty, never a zero-`n` `ContinuousMetric`), so this is
currently dead code protecting against a shape no pack produces today — correct to add defensively,
but its only current guarantee is "I read it and it's right," not "the suite would catch me being
wrong." **Suggested fix:** one test mirroring the existing binary one, built directly
(`ContinuousMetric(name=..., mean=0.0, n=0, support=(0.0, 1.0))` in a hand-built aggregate, the same
way `test_rank_report_ranks_a_continuous_headline_metric_by_mean` builds its fixtures) asserting the
model is named in the banner and absent from the ranked rows — this closes the metric-type axis the
current single test doesn't cover.

### [MINOR] The per-metric scoping claim (a two-metric family, n=0 on only one member) is asserted in the docstring but has no test

`_rank_zero_n_lines`'s docstring (`report.py:174-175`) explicitly claims: "a run can carry a real
aggregate for a sibling verdict metric while declaring `n=0` on this one" — the guard-judge shape
Defect 1's own root-cause writeup distinguishes from the single-metric `nlq` case. I verified this
claim holds by building the scenario directly (a `guard_pack` two-metric family, one run with a real
`falseAdvanceRate` aggregate and `n=0` `falseSuspendRate`) and calling `rank_report` — the banner
correctly appears only under `### falseSuspendRate`, the row renders normally under
`### falseAdvanceRate`, exactly as claimed (see Appendix for the script and output). But nothing in
`tests/test_report.py` pins this — the one new test uses `_rank_pack()`'s single-headline-metric
shape throughout. **Suggested fix:** extend (or add alongside) the existing guard-judge no-headline
fixture pattern (`test_rank_report_with_no_headline_renders_two_independently_sorted_tables`) with
one candidate declaring `n=0` on exactly one of the two verdict metrics, asserting the banner is
scoped to that metric's own `###` section and absent from the sibling's.

Together, these two gaps are a coverage probe over the same two axes Defect 1's own root-cause
section named (metric type: binary vs. continuous; family shape: single vs. multi-member) — the
shipped test covers exactly one cell of that 2×2, the cell the defect happened to surface in.

### [NIT] `docs/HISTORY.md` carries no entry for this fix yet

Consistent with the same nit already raised for Unit C above (§2.2) — likely deferred to a
coordinated close rather than dropped, not gating.

## Verification performed

- **Root cause, re-derived against the real code, not the fix's docstring.** Confirmed
  `BinaryMetric.rate` (`results.py:110-112`, `self.successes / self.n if self.n else None`) already
  returned `None` for `n=0` **before this diff** — untouched by it — so the true gap was never in
  `BinaryMetric.rate` itself but in `_metric_value`/`_rank_rows` treating that `None` identically to
  "no aggregate declared at all." This matches the QA report's own root-cause section
  (`docs/test-reports/small-model-catalog-sweep-report.md`, Defect 1) precisely — confirmed by
  reading, not assumed from its prose.
- **Banner/mismatch mutual exclusion, proved structurally, not just observed.** `rank_report`
  (`report.py:1362-1364`) filters `runs` via `_aggregate_item_mismatches` — which flags a run if
  *any* of its verdict-metric aggregates disagrees with its own items — **before** any run reaches
  `_render_one_rank_table`/`_zero_n_arms`. A run reaching the new banner has therefore already
  cleared that check for every verdict metric, so the same run can never simultaneously trigger
  `INVALID RESULTS EXCLUDED` and `EXCLUDED — n=0` on the same or a sibling metric. Also confirmed the
  `n=0` case genuinely clears the mismatch check by construction: `_aggregate_item_mismatches`
  (`report.py:434-473`) counts items whose `scored_outcome`/`scored_value` is not `None`; every item
  in the zero-n case is declared unscoreable for that metric, so `counted == 0 == metric.n` — no
  disagreement, exactly as both the fix's docstring and the QA report claim.
- **The pinning test, run and read.** `.venv/bin/python -m pytest -q tests/test_report.py -k
  "zero_n or rank_report"` → 19 passed. Full suite: `.venv/bin/python -m pytest -q` → **1775 passed,
  3 deselected** (1774 on the pre-diff tree, confirmed by `git stash`/re-run — the delta is exactly
  this diff's one new test). `.venv/bin/ruff check .` → **All checks passed!**
- **Two coverage gaps found by mutation, not inference** — see Findings above; scripts and output in
  the Appendix.
- **Five regenerated reports spot-checked.** `diff`ed all five `*-20260920-01.md` against
  `*-20260920-02.md`: the four clean packs (`embedder-graphrag-retrieval`,
  `guard-judge-understanding`, `tool-caller-shop-assistant`, `chat-responder-grounded-answers`) are
  **byte-identical** — the fix introduces zero false positives on this sweep's real data.
  `nlq-structured-query-rank-20260920-02.md` gains exactly the expected banner, naming both
  Defect-1 models with the correct item count (`40 item(s) attempted, n=0 scored`), positioned
  between the pack's own restated caveat and the ranked table; the two models still separately
  appear as `no verdict — no paired data` in the reference-anchored family table lower down,
  consistent with (not duplicating) the new banner.
- **Marker-comment (`<!-- rank-report: ... -->`) placement unaffected, confirmed by reading.** The
  new banner is inserted before the table (`report.py:1140`); the marker comment is emitted after
  the table (`report.py:1185-1189`), so Unit C's consolidator — which matches that comment
  line-by-line regardless of surrounding content — is untouched by this diff.

## What's solid

- Root-cause diagnosis is correct and precisely scoped: the fix touches exactly the two places the
  conflation lived (`_metric_value`'s `ContinuousMetric` branch, and the missing naming step in
  `_render_one_rank_table`), not the presentation code around them.
- The shipped test is a real mutation-catcher for the reported scenario: it distinguishes "declares
  a real `n=0` aggregate" (named, excluded) from "declares no aggregate at all" (silently absent)
  using both cases in one fixture, matching the exact confusion Defect 1 reported.
- Zero regressions on real sweep data — four of five regenerated reports are byte-identical to their
  pre-fix versions, and the fifth changes in exactly the way the QA report asked for.
- The new banner cannot collide with the pre-existing `INVALID RESULTS EXCLUDED` banner by
  construction (verified structurally, not just by absence of an observed collision), and is scoped
  correctly per-metric for a multi-metric family (verified by direct construction, §Findings above).

## Open questions

- None blocking. Whether the two coverage-gap suggestions above land now or batch with a later test
  hardening pass is `teco`'s call.

## Appendix — mutation checks

**`ContinuousMetric` guard, reverted in a scratch copy (no repo file touched):**

```
$ cp -r modelbench tests <scratch>/
$ python3 -c "... replace 'return agg.mean if agg.n else None' with 'return agg.mean' in <scratch>/modelbench/report.py"
$ PYTHONPATH=<scratch> .venv/bin/python -m pytest <scratch>/tests/test_report.py -q
156 passed in 0.65s
```

**Per-metric scoping in a two-metric family, verified directly (no repo file touched):**

```python
pack = guard_pack(headline=None, verdicts=("falseAdvanceRate", "falseSuspendRate"))
# model-x: real aggregate for falseAdvanceRate (n=40), n=0 for falseSuspendRate
x = run("model-x", items=advance_items + suspend_items, aggregates=agg_x, ...)
md = rank_report([x], pack=pack)
```

Output:
```
EXCLUDED in advance section: False
EXCLUDED in suspend section: True
model-x ranked row in advance: True
model-x ranked row in suspend: False
```

Matches the docstring's claim exactly: the banner and the row-exclusion are both scoped to
`falseSuspendRate`'s own section only.

## Verification round 2 — 2026-09-20 (`analyst`)

Re-verified `tdd-engineer`'s follow-up diff (`git diff HEAD -- modelbench/report.py
tests/test_report.py`, new `_attempted_for_metric`), which addresses both my minors below in the
same pass that fixed `data-scientist`'s MAJOR (the `len(r.items)` overcount — see their section
below; not re-argued here). Re-checked by re-deriving, not by trusting either the dispatch message
or the new tests' own assertions.

**Correction to my own prior methodology, disclosed rather than buried.** My original Appendix
above (`PYTHONPATH=<scratch> pytest ...`) reported "156 passed" for the reverted `ContinuousMetric`
guard and I read that as "untested." Re-running the same technique on this pass, I found it
**silently imports the real, installed `modelbench.report`, not the scratch mutant** — this
component installs as an editable package, so `PYTHONPATH` prepending does not override it, and
every one of my `PYTHONPATH`-scratch mutation checks in the original section was therefore
verifying the *unmutated* code, not the mutant. My qualitative conclusion (no test existed) was
still correct at the time — because no test existed, full stop — but I should not have presented a
broken-harness "156 passed, zero failures" as evidence for it. Redone correctly this round via
`importlib.util.spec_from_file_location` (the technique my own round-1 Unit A section already used
successfully, and which I should have reused instead of reintroducing `PYTHONPATH`): loading the
mutant `report.py` as an isolated module while letting its own `from modelbench import stats`/etc.
resolve normally confirms the revert now genuinely **crashes** —
`stats.mean_bootstrap_interval` raises `ValueError: ... needs at least two values` when a zero-`n`
continuous arm reaches the bootstrap call unguarded — matching `tdd-engineer`'s own reported
mutation exactly. Flagging this so nobody treats my original Appendix's "156 passed" line as
verified; the new fix and new test are confirmed correct by this round's redone check, not by that
one.

**1. Minor #1 (`ContinuousMetric` zero-n, untested) — closed, mutation-confirmed with a working
harness this time.** `test_rank_report_names_a_zero_n_continuous_aggregate_not_silently_drops_it`
exercises exactly the guard: I reproduced the crash above independently, and separately confirmed
the real (unmutated) code passes the same fixture cleanly. No further gap.

**2. Minor #2 (per-metric scoping for a multi-metric family, unpinned) — closed, and the new test
does more than I asked for.** `test_rank_report_zero_n_banner_reads_the_metrics_own_item_count_not_the_runs_whole_list`
covers the scoping claim I asked for (banner in `falseAdvanceRate`'s section only, normal ranking
under `falseSuspendRate`) **and** the attempted-count correctness `data-scientist` separately
required — reproduced via the same isolated-module technique: mutating `_attempted_for_metric` back
to `len(run.items)` makes the banner read "70 item(s)" where the test asserts "70 item(s)" is
**not** present, and the assertion catches it. Confirmed.

**3. A third gap, found independently before reading `data-scientist`'s own round-2 section below —
their finding, not a new one of mine, but worth recording that two reviewers reproduced it
separately.** `_attempted_for_metric`'s "declares nothing at all" fallback
(`metric in it.scoreable or not it.scoreable`) cannot distinguish which metric a **fully empty**
(`scoreable == {}`) item belonged to when a run's pack has more than one verdict metric and items
from more than one of them are all empty on the same run — I constructed exactly this (70 items,
half nominally `falseAdvanceRate`'s, half `falseSuspendRate`'s, all `scoreable={}`) against the real
`_attempted_for_metric` and got 70/70, not the true 40/30 split. `data-scientist`'s own section
below traces this to a concrete, real trigger (`classification.py:239-248`'s `result is None`
no-response path, shared by every scorer in the package) and confirms it hasn't shipped wrong in any
of the 17 stored guard-judge records today. I have nothing to add to their analysis or severity call
— citing it here only so a reader of my section doesn't read "both my minors closed" as "the whole
diff is clean."

**Stopping signal, scoped precisely, same shape as `data-scientist`'s own below.** For my own two
minors — the untested `ContinuousMetric` branch and the unpinned multi-metric scoping — both are
now correctly fixed and correctly tested; I have no further findings there and consider that part of
the gate closed. This is **not** a stopping signal for `_attempted_for_metric` as a whole: the
cross-metric empty-scoreable ambiguity `data-scientist` found (and I independently reproduced) is
real, unresolved, and — per their assessment, which I share — should not be treated as closed. My
own suite/lint re-run: `.venv/bin/python -m pytest -q` → **1778 passed, 3 deselected**;
`.venv/bin/ruff check .` → **All checks passed!**

---

# Defect 1 fix — `rank_report`'s n=0 exclusion banner (`data-scientist`, methodology gate) — 2026-09-20

**Scope.** Reviewed the uncommitted working-tree diff (`git diff HEAD -- modelbench/report.py
tests/test_report.py`) that fixes Defect 1 from
`docs/test-reports/small-model-catalog-sweep-report.md` — the ranked table silently dropping
`qwen/qwen3-4b-thinking-2507`/`stable-code-instruct-3b` on `nlq-structured-query` (both 100%
parse-failure, real `n=0` aggregate). Read the diff's new `_zero_n_arms`/`_rank_zero_n_lines`, the
one-line `_metric_value` change, `_render_one_rank_table`'s call site, `docs/plans/
small-model-benchmarking-ml.md` (`-ml`, this file's authoritative stats source) for anything
bearing on zero-`n` handling, and `AGENTS.md`'s "Load-bearing invariants"/five-honesty-rules
section. Independent of `analyst`'s parallel general-correctness pass on the same diff.

**Verdict: needs changes.** The exclusion-and-naming *design* is correct and is the right fix for
Defect 1 as reported — but the banner's own "N item(s) attempted" figure is wrong for any pack
whose per-run `items` list spans more than one verdict metric, which I reproduced against
guard-judge's real production data shape, not a hypothetical. Guard-judge is one of only two
multi-verdict-metric packs this sweep runs, and the defect fires the moment guard-judge's own
scorer ever produces a real `n=0` (it doesn't in the current live sweep, only because its
classifier has a parse-failure fallback the nlq extractor lacks — a data accident, not a structural
guarantee).

## Findings

### [MAJOR] `_rank_zero_n_lines`'s "item(s) attempted" count uses the run's whole item list, not the metric's own — wrong for any multi-metric-per-run pack, reproduced against guard-judge

`_rank_zero_n_lines` (`modelbench/report.py`) renders:

```python
lines.append(f"> - `{r.modelKey}` — {len(r.items)} item(s) attempted, n=0 scored")
```

`len(r.items)` is the run's **entire** stored item list — every item across every metric that run's
pack scores — not the count of items that made any declaration (`scoreable[metric]` present, true
or false) about the one `metric` this banner section is for. For `nlq-structured-query` (a
single-verdict-metric pack: `_rank_pack`'s default `verdicts=("layer1ExactMatchRate",)`), every
item in `r.items` does declare that one metric, so `len(r.items) == 40` happens to be correct for
the exact case Defect 1 reported. It is not correct in general, and the function's own docstring
already reasons about the case where it isn't: *"a run can carry a real aggregate for a sibling
verdict metric while declaring `n=0` on this one"* — i.e., the author explicitly considered the
guard-judge shape when scoping *which table the banner appears in*, but the attempted-item count
inside that banner was left unscoped to the metric.

**Confirmed against real production data, not just a constructed fixture.** Every stored
guard-judge run record declares each item's `scoreable` map with exactly one metric key — never
both:

```
$ python3 -c "... json.load(open('results/runs/guard-judge-understanding-qwen_qwen3-4b-2507-...json'))..."
distinct scoreable key sets: {('falseSuspendRate',), ('falseAdvanceRate',), ('falseAdvanceRateBoundary',)}
```

85 total items on that record, split across the two verdict metrics (plus the exploratory boundary
metric) — never 85 items each individually scoreable for both `falseAdvanceRate` and
`falseSuspendRate`. `ItemResult.scored_outcome`'s own three-state contract (`results.py:390-429`)
gates on `self.scoreable.get(metric, False)` per metric — the codebase already has the concept of
"did this item make a declaration for this specific metric," and `_rank_zero_n_lines` doesn't use
it.

**Reproduced end to end**, not argued from reading alone: built a guard-judge pack (`verdicts=
("falseAdvanceRate", "falseSuspendRate")`, mirroring `_guard_judge_arm`'s asymmetric-item shape)
with one candidate whose 40 `falseAdvanceRate` items are all `scoreable=False` (parse failure, real
`n=0`) and whose 30 `falseSuspendRate` items are normal, real data (no monkeypatching, no
repository file touched — only in-memory fixtures via the real `rank_report`). Result, the rendered
`falseAdvanceRate` section:

```
> **EXCLUDED — n=0 for `falseAdvanceRate`**
>
> Ran, and the stored record is internally consistent, but the declared aggregate honestly reports
> zero scoreable observations for this metric — excluded from the ranking below, never ranked
> "worst" and never silently absent either:
> - `zero-advance-cand` — 70 item(s) attempted, n=0 scored
```

**70, not 40.** The banner counts all 70 items on the run record (40 `falseAdvanceRate` +
30 `falseSuspendRate`), overstating by 75% how many items were actually presented to the model for
the metric this section is about. The correct figure — confirmed by re-deriving it as
`sum(1 for it in r.items if metric in it.scoreable)` — is 40, matching what the pack's own FR-11
caveat two lines above already states for this exact metric ("floor 15.0/20.0 pp... for
falseAdvanceRate/falseSuspendRate," `-ml` §7.3's published 40/30 split) — so the banner would ship
a number that visibly disagrees with the report's own caveat text sitting directly above it.

**Why this is major, not cosmetic.** This component's own established standard — the exact phrase
used earlier in this document for the `n_units = max(...)` defect (Unit A.5) — is that a
"wrong-but-plausible-looking number" is never to ship silently. This is precisely that shape: 70 is
plausible (it's a real count that exists somewhere in the record), attached with confident,
specific-sounding language ("N item(s) attempted"), and wrong for the metric the sentence is
about. It doesn't corrupt the ranking or any statistical test — the *exclusion* itself is correctly
scoped to `agg.n == 0` on the metric-specific aggregate, verified below — but it corrupts the one
piece of free-text context a reader has for judging *how much data was actually thrown away*,
which is the entire reason Defect 1 asked for a named banner instead of a silent drop.

**Route: `coder`/`tdd-engineer`, cheap.** Suggested fix: `attempted = sum(1 for it in r.items if
metric in it.scoreable)`, replacing `len(r.items)`. Suggested test: a guard-judge fixture (reusing
`_guard_judge_arm`'s asymmetric-shape convention) with one candidate whose `falseAdvanceRate` items
are all `scoreable=False` and whose `falseSuspendRate` items are real, asserting the banner in the
`falseAdvanceRate` section reads the metric-specific attempted count (40, not 70) — this is the
test that would have caught the defect, and the existing single-metric `nlq` fixture provably
cannot (`len(r.items)` and the correct count coincide there by construction).

## Answers to the brief's four questions

**1. Is excluding the `n=0` aggregate from the ranked table (vs. rendering 0/0, imputing, or
excluding some other way) statistically correct?** Yes, and it's the only defensible choice. A
`BinaryMetric.rate`/`ContinuousMetric.mean` and a `wilson_interval` are both undefined at `n=0` —
there is no rate to sort by and no interval to print, so "render it as 0/0" is not a real
alternative, it's a different bug (a `0.000` rate cell reading as a legitimate, worst-observed
score rather than "undefined"). Imputing a value would violate the "declared, never inferred"
honesty rule directly (`AGENTS.md`'s load-bearing invariants) — there is no principled value to
impute for a scorer that produced zero scoreable observations. Excluding-and-naming is exactly the
pattern this file already uses for the *other* case where real data can't be ranked
(`_aggregate_item_mismatches` → `INVALID RESULTS EXCLUDED` (AC-2)) — this fix is that same
pattern applied to a second cause of unrankability, which is the right level of consistency to
hold. It also composes correctly with the pre-existing "absent, not worst" rule for a run with no
aggregate at all: both states return `None` from `_metric_value` and are excluded from `_rank_rows`
identically, and the new banner is what tells the two apart for a reader — the ranking mechanics
did not need to change, and didn't.

**2. Does naming the excluded model create a risk of misreading it as a real 0% rate?** The
banner's own wording is fine as a proposition — "declared aggregate honestly reports zero
scoreable observations for this metric" and "n=0 scored" are both explicit that the denominator
itself is zero, not that a rate of `0/N` was observed, and the phrase "never ranked 'worst'"
directly forecloses the reading a rate-focused skim might reach. The one place this answer flips is
the finding above: a wrong, inflated "item(s) attempted" count is its own, different misreading
risk — not "mistaken for a 0% rate" but "mistaken for a worse failure than actually occurred" (a
guard-judge reader could conclude the model failed all 70 of its guard-judge items, when it
produced 30 perfectly normal `falseSuspendRate` results one table below). Fix the count and this
question's answer is clean; as shipped, it is not.

**3. Interaction with the five honesty rules?** The exclusion mechanism itself reads cleanly off
`agg.n == 0` (a declared fact on the stored aggregate) — compliant with "an item's outcome for a
metric is declared, never inferred," and it introduces no new Wilson-interval call, so the
analysis-unit rule is untouched (excluded arms never reach `stats.wilson_interval` at all, at any
unit). The attempted-count defect above is a **soft** violation of the same declared-not-inferred
spirit one level up the stack: the report presents a number about "this metric" that isn't actually
read from any metric-scoped source, it's inferred (incorrectly) from an unrelated whole-run
count. It's not the same class of violation the rule is written to prevent (no `ItemResult` state
is misread), but the practical effect — a report asserting a specific-sounding fact about a metric
that isn't true of that metric — is the failure mode the rule exists to prevent one level up.

**4. Does the per-metric scoping hold up for guard-judge's two-class-conditional-metric shape?**
The *routing* — which table gets the banner, and whether the excluded candidate still renders
normally in its sibling metric's table — holds up correctly, verified by direct reproduction (see
above and Verification): the banner fires only in `falseAdvanceRate`'s own section, the
`falseSuspendRate` section carries no banner and lists `zero-advance-cand` as an ordinary ranked
row with its own real `12/30` data, and the family/Holm-ladder machinery (untouched by this diff)
is unaffected. What does **not** hold up is the attempted-count figure inside the banner, which is
the Finding above. I also checked `guard-judge-understanding-rank-20260920-02.md` (the brief's
suggested spot-check): no model has a real `n=0` in that report today (guard-judge's classifier has
a parse-failure fallback, per the QA report's own root-cause note), so the live sweep's rendered
reports do not currently exhibit either the fix's benefit or its defect — this is a latent-but-real
bug in delivered code, not something already shipped wrong to a stakeholder.

## Verification performed

- Read the full diff (`git diff HEAD -- modelbench/report.py tests/test_report.py`) and the
  surrounding, unchanged code: `_metric_aggregate`, `_rank_rows`, `_render_one_rank_table`,
  `_rank_resolving_power_lines`, `ItemResult.scored_outcome`/`scored_value` (`results.py:390-447`).
- Read `AGENTS.md`'s "Load-bearing invariants" section in full (the five honesty rules,
  `-ml`-implements-`stats.py` rule, Wilson-over-analysis-unit rule) and grepped `docs/plans/
  small-model-benchmarking-ml.md` for `n=0`/zero-observation handling — found no section
  prescribing report-layer treatment of a real `n=0` aggregate specifically (this is `report.py`'s
  own presentation-layer judgment call, not something `-ml` already rules on), so I evaluated it
  against the honesty rules and this file's own established exclude-and-name precedent instead.
- **Independent reproduction, no repository file touched, in-memory fixtures only**: built a
  guard-judge two-metric pack and a candidate with a real `n=0` aggregate on `falseAdvanceRate`
  (all 40 items `scoreable=False`) and a real, normal aggregate on `falseSuspendRate` (30 items),
  called `rank_report` directly with `reference=` set to exercise the family path too. Confirmed:
  (a) the `falseAdvanceRate` section carries the exclusion banner naming the candidate; (b) the
  `falseSuspendRate` section carries no banner and ranks the candidate normally with its real
  `12/30` data; (c) the banner's attempted-item count reads 70, not the correct 40 — the Finding
  above. Full script and output are in this session's scratchpad, reproducible from
  `guard_pack`/`item`/`run`/`ClassificationAggregates`/`BinaryMetric` (`tests/conftest.py`,
  `modelbench/results.py`) without touching any repository file.
- **Checked the real production data shape**, not assumed: loaded a real stored guard-judge run
  record (`results/runs/guard-judge-understanding-qwen_qwen3-4b-2507-*.json`) and confirmed every
  item's `scoreable` dict carries exactly one metric key — never both verdict metrics on the same
  item — which is what makes `len(r.items)` structurally wrong as a per-metric attempted count for
  this pack, not merely wrong for a contrived fixture.
- **Confirmed the resolving-power sentence (`_rank_resolving_power_lines`) is not newly affected by
  this diff.** Its `ns` list is built from `_metric_aggregate` (not `_metric_value`) and already
  included a zero-`n` aggregate's `0` before this change — `git diff` shows no touch to that
  function. Since it takes `max(ns)`, a lone zero-`n` arm cannot lower the reported `n_units` unless
  every arm for that metric is zero-`n`, an edge case this diff neither introduces nor worsens. Not
  raised as a finding against this diff; noting only that a future all-arms-zero-`n` scenario
  (not observed in any live data) is untested territory for `resolving_power`'s own zero-`n_units`
  behavior, unrelated to Defect 1.
- Confirmed the new test (`test_rank_report_names_a_model_with_a_declared_zero_n_aggregate_not_
  silently_drops_it`) uses `_rank_pack`'s single-metric shape only, so it structurally cannot catch
  the Finding above — consistent with why the defect shipped in the diff without a red test.
- `.venv/bin/python -m pytest -q` and `.venv/bin/ruff check .` were not re-run by me for this pass
  (no source file was touched by this review); `analyst`'s parallel general-correctness gate on the
  same diff covers suite/lint.

## What's solid

- The core Defect 1 fix — excluding a real `n=0` aggregate from ranking while naming it, rather
  than conflating it with "never attempted" — is the statistically correct choice, matches this
  file's own established exclude-and-name precedent (`INVALID RESULTS EXCLUDED`), and is exactly
  scoped per metric (not per run), which is the right granularity for a pack like guard-judge.
  Confirmed correct in both directions: correctly fires when it should, correctly stays silent for
  a run's other, unaffected metric.
- The wording distinguishing "ran but scored nothing" from a real rate is clear and does the honesty
  work it needs to.
- The gap found here is narrow and cheap to close — one expression inside one function, one new
  fixture reusing an existing helper's asymmetric-shape convention.

## Open questions

- None beyond the required fix above. Whether it lands as a follow-up to this same diff or a
  separate commit is `teco`'s call; it should land before this fix is considered complete for any
  pack beyond `nlq-structured-query`, since guard-judge is squarely in scope for this sweep and the
  defect is real, reproduced, and currently invisible to the test suite.

## Verification round 2 — 2026-09-20

Second pass, on `tdd-engineer`'s follow-up diff (`git diff -- modelbench/report.py
tests/test_report.py`, new `_attempted_for_metric`), after being told my own suggested one-liner
(`metric in it.scoreable`) was deliberately not applied.

**1. The rejection of my literal one-liner is correct — verified against the real data myself, not
taken on the report.** Pulled `results/runs/nlq-structured-query-stable-code-instruct-3b-
2026-09-20T01:18:34Z.json` directly: every one of its 40 parse-failure items carries `scoreable:
{}` — the key is genuinely absent, not `{metric: False}`. My suggested `metric in it.scoreable`
would read every one of those as "not attempted," so the fixed banner would have shipped "0
item(s) attempted" for the exact live Defect-1 rows this whole fix exists to name correctly. Good
catch; my original suggestion was wrong on the real data shape, and `_attempted_for_metric`'s
"declares `metric`, or declares nothing at all" predicate correctly fixes that case — confirmed by
re-running the pack against the real fixture shape, 40 attempted, as expected.

**2. The asymmetric guard-judge case (only my finding's own reproduction) is now correct.**
Rebuilt my Verification-round-1 fixture (one candidate, real `n=0` on `falseAdvanceRate` via
40 `scoreable=False` items, real normal data on `falseSuspendRate`) against the new code: banner
now reads "40 item(s) attempted," not 70; the `falseSuspendRate` section is unaffected. Matches
`tdd-engineer`'s own new regression test
(`test_rank_report_zero_n_banner_reads_the_metrics_own_item_count_not_the_runs_whole_list`), which
I independently re-derive rather than trust.

**3. A real, reproduced gap remains — the "declares nothing at all" fallback is ambiguous, not
just permissive, once *two* verdict metrics both have empty-scoreable items on the same run.**
`_attempted_for_metric`'s predicate (`metric in it.scoreable or not it.scoreable`) treats every
truly-empty item as belonging to *every* metric being asked about, because a `scoreable == {}`
item's *original* tier/metric is not recoverable from the stored record — and this is not a
theoretical gap: `classification.py`'s own `score_item` (guard-judge's real scorer) emits
`scoreable={}` whenever `result is None` (`:239-248`, a timeout or no-response item), **before**
the tier→metric mapping (`_METRIC_BY_TIER`) is ever consulted — the same "the call never happened"
shape every scorer in this package uses (`extraction.py`, `retrieval.py`, `grounding.py` all emit
an identical `scoreable={}` on their own no-response path). So a guard-judge run in which items
from *both* tiers time out — a total outage, not a hypothetical — produces exactly the
cross-metric ambiguity my finding was about, just triggered by "no response" instead of "response
received but unparseable."

**Reproduced directly**, no repository file touched: built a guard-judge run with 40
`falseAdvanceRate`-tier items and 30 `falseSuspendRate`-tier items, **all 70** with
`scoreable={}`/`counts={}` (mirroring `classification.py:239-248`'s own `ItemResult` shape for a
`result is None` item) and a real `n=0` aggregate on both metrics. Result:

```
falseAdvanceRate banner:  `total-outage-cand` — 70 item(s) attempted, n=0 scored
falseSuspendRate banner:  `total-outage-cand` — 70 item(s) attempted, n=0 scored
```

Both read 70. The true split is 40/30 (the pack's own published `-ml` §7.3 figures, restated in
this exact report's own FR-11 caveat two lines above each banner) — this is the identical
70-instead-of-(40,30) overcount my original finding reported, now reachable via a different,
equally real trigger. `report.py` cannot recover the true split from the stored record: a
`result is None` `ItemResult` carries no `detail`/tier information (`classification.py:241-248`
passes no `detail` kwarg on that branch), so which of the two tiers a fully-empty item belonged to
is genuinely not present in the data `_attempted_for_metric` has to work with — this isn't a
missed lookup, the information doesn't exist in the record.

**Checked whether this has already shipped wrong**: no. Scanned every stored guard-judge run file
in the current sweep (`results/runs/guard-judge-understanding-*.json`) for any item with
`scoreable == {}` — none exist in any of the 17 stored guard-judge records. The gap is real and
reachable but has not yet produced a wrong number in any of the five rendered reports this sweep
has shipped.

**Severity and recommendation.** Narrower than the original defect — it requires a total or
partial *outage* (no response at all) hitting items from more than one verdict metric in the same
run, on a pack that has more than one verdict metric sharing one run's items (today, only
guard-judge). Not a blocker for the current five reports (confirmed absent from all live data
above). But it is the same failure class this whole fix exists to close, reachable through a
documented, general code path every scorer in the package shares, and the honest fix is not "guess
a split" (that would itself be an inferred number this file's own "declared, never inferred" rule
forbids) — it's to stop presenting a single, specific-looking attempted count when the record
cannot support one. Suggested shape: when a run's ambiguous (`scoreable == {}}`) item count is
shared across more than one of the pack's verdict metrics, the banner should say so explicitly —
e.g. "N item(s) declared `{metric}`, plus M further item(s) that recorded no response for any
metric in this run and cannot be attributed to `{metric}` specifically from the stored record" —
rather than folding the ambiguous M into a single confident number. Route:
`coder`/`tdd-engineer`, via `teco`.

**Stopping signal, scoped precisely**: for the scenario Defect 1 actually reported (a response
that arrives and fails to parse) and for the asymmetric single-metric-zero-n case (my own original
reproduction), this fix is correct and complete, and I have no further findings there — that part
of the gate is closed. It is **not** a full stopping signal for `_attempted_for_metric` as a
general-purpose function: the total-outage/cross-metric-ambiguity scenario above is real,
reproduced, and unresolved. Given it's unreachable in today's live data, whether it blocks this
diff or is tracked as a fast-follow is `teco`'s sequencing call, not mine — but it should not be
treated as closed.
