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
