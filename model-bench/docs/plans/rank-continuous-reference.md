# `rank --reference` on a continuous verdict metric — implementation plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** —

## 1. Goal & scope

Fix D-1 (`docs/test-reports/small-model-benchmarking-manual-additions-report.md`): `./run.sh rank
--pack embedder-graphrag-retrieval --reference bm25` crashes with an uncaught `MetricKindError`
because `rank_report`'s reference-anchored-family builder (`modelbench/report.py`, function
`rank_report`, lines 1445-1483) unconditionally treats every verdict metric as boolean. This plan
extends `rank_report`'s FR-8 reference-anchored family (plan §3.2.2) to a continuous verdict
metric, building directly on `data-scientist`'s method note (`docs/plans/
rank-continuous-reference-ml.md`, hereafter "`-ml`" in this document) — every statistical formula,
constant and rendering shape below is that note's, cited by section; this plan does not re-derive
them.

**In scope:** the `stats.continuous_verdict` signature change (`-ml` §3.1); the combined-divisor
formula for a continuous reference family (`-ml` §3.2); a mixed-kind "refuse whole" guard for
`rank_report`'s reference path, **extended to a genuine cross-arm kind-disagreement guard** the
`-ml` note left open (§5 below resolves both of its open questions); the continuous
reference-family rendering shape (`-ml` §3.4); the two adjacent defects the note requires fixed in
the same unit — `_rank_resolving_power_lines`' kind-blindness (`-ml` §3.5) and
`ContinuousVerdict.text`'s hardcoded `"95% CI"` (`-ml` §3.6); and the regression guarantee that
`guard-judge-understanding`'s existing binary reference-family rendering is byte-for-byte
unchanged.

**Out of scope** (per `-ml` §5's own scope boundary, restated): a continuous-metric
power-preview sentence (no such concept is designed anywhere, and none is needed to fix D-1); a
`k >= 1` defensive guard on `correction_k` (`-ml` §3.1's "minor, optional" item — this plan elects
**not** to add it; see §3.6 below for the reasoning); any change to `compare_report`, which already
handles a continuous/mixed family correctly on the two-arm path and is touched only as read-only
precedent.

**CPG:** considered, not relevant — `GRAPHS` on the FalkorDB instance available to this session
lists no `cpg_model-bench` (only `cpg_falkorchat` exists among loaded graphs); this is a code-level
task in a component with no loaded CPG, so `cpg-analysis` has nothing to query and the plan is
built from direct file reads instead.

## 2. Context & findings

All line numbers below were read directly from the current tree, not carried over from the `-ml`
note (which cites some of the same lines — reconfirmed independently where it matters to this
plan's mechanics).

### 2.1 The two functions this unit adds parameters to / restructures

- **`stats.continuous_verdict`** (`modelbench/stats.py:1651-1769`). Today: `k = len(family)`
  (line 1719), no way for a caller to substitute a larger, independent axis. `stats.verdict`
  already has the parameter this needs (`stats.py:1216-1301`, `correction_k: int | None = None`,
  `k = correction_k if correction_k is not None else len(family)` at line 1270) — the pattern to
  mirror exactly (`-ml` §3.1).
- **`stats.ContinuousVerdict.text`** — the two f-string templates inside `continuous_verdict`
  (`stats.py:1740-1753`) hardcode the literal `"95% CI"` in both the `distinguishable` and
  not-distinguishable branches, never reading the dataclass's own `alpha_used: float` field
  (declared `stats.py:1647`, computed `stats.py:1721`).
- **`modelbench.report.rank_report`** (`report.py:1355-1486`). The reference-family construction
  block (lines 1445-1472) unconditionally loops `for metric in family: for cand in candidates:`
  calling `_paired_rows` (the boolean-only accessor) and `stats.mcnemar_exact` — this is D-1's
  exact crash site the moment `metric` resolves to a `ContinuousMetric` aggregate, because
  `_paired_rows` calls `item.scored_outcome(metric)`, which raises `MetricKindError` for a metric
  whose real values live in `measures`, not `counts` (`results.py`; the same exception
  `_aggregate_item_mismatches` already catches at `report.py:483`, but nothing catches it here).
  `_render_reference_family` (`report.py:1246-1295`) is called once per `metric in members`
  (`report.py:1474-1483`), filtering that one shared combined ladder back to `metric`'s own rows —
  never rebuilding a second, per-metric ladder.
- **`_rank_resolving_power_lines`** (`report.py:1298-1352`) runs unconditionally, once per member
  of `family`, for every pack with `len(runs) >= 2` — gated on neither `--reference` nor metric
  kind. Confirmed live and wrong today for `embedder-graphrag-retrieval`'s continuous `mrr`
  (`-ml` §2.3, spot-checked by `teco` against `reports/embedder-graphrag-retrieval-
  rank-20260920-03.md`).

### 2.2 The precedent this plan mirrors, already built and tested on the two-arm path

`compare_report` (`report.py:1488-1930`) already does, for **two arms**, almost everything this
plan needs for **N arms** (a reference plus every candidate):

- `_metric_kind(a: RunResult, b: RunResult, name: str) -> str` (`report.py:229-242`): resolves a
  metric's kind from whichever of the two arms declares an aggregate for it, preferring `a`'s own
  declaration, falling back to `b`'s only when `a` carries none. Used at `report.py:1672`:
  `kinds = {metric: _metric_kind(a, b, metric) for metric in family}`.
- The **mixed-kind refuse-whole guard** (`report.py:1664-1676`, rendered at `1681-1706` via
  `_MIXED_FAMILY_MEMBER`, `report.py:769-774`, and the family-wise section at `1854-1855` via
  `_MIXED_FAMILY_CORRECTION`, `report.py:779-783`): `mixed_kinds = len(resolved_kinds) > 1`; no
  member of a mixed family is verdicted, and — notably, confirmed by reading
  `test_a_mixed_kind_family_is_refused_whole_not_pruned_to_the_majority_kind`
  (`tests/test_report.py:1741-1764`) — **each metric's own per-candidate tally still renders**
  (`_paired_rows`/`_paired_diffs` called per that metric's own resolved kind), because under
  `compare_report`'s two-arm mixed case every *individual* metric's kind is still cleanly
  resolved; only the *combination* across metrics is mixed.
- The **all-continuous combined-family rendering** already exists on the two-arm path
  (`report.py:1707-1755`, `_CONTINUOUS_FAMILY_CORRECTION` at `report.py:788-793`, rendered at
  `1856-1857`): `stats.continuous_verdict(diffs_row.diffs, metric_name=metric, family=family, ...)`
  is called **with the whole family**, never `family=[metric]` — pinned by
  `test_continuous_verdict_receives_the_whole_family_not_just_the_metric`
  (`tests/test_report.py:1828+`) after a real mutation-testing miss (U67) collapsed it once.
  `continuous_verdict`'s own `k = len(family)` (today, no `correction_k`) is exactly what this
  plan's new parameter must not disturb at this call site — `compare_report`'s call passes no
  `correction_k`, so it must keep defaulting to `len(family)` unchanged.
- **`_paired_diffs`** (`report.py:335-403`): the continuous sibling of `_paired_rows`, one
  difference per analysis unit via `item.scored_value(metric)`, same `-ml` §4.3 tally shape
  (`PairedDiffs`, `report.py:318-332`).
- **Orientation, confirmed by reading `stats.mover_d_interval` directly**
  (`stats.py:126-146`): `diff = p1 - p2` where `p1` is the **first** argument arm's own rate — i.e.
  `stats.verdict`'s printed `diff`/`ci` (and, by the identical construction, `mover_d_interval`'s
  own binary-table analogue `(b-c)/n`) is **first-arm-minus-second-arm**. `continuous_verdict`'s
  `diff = sum(diffs)/len(diffs)` over `diffs` built by `_paired_diffs(a, b, ...)` is the same
  orientation: `a`-minus-`b`. `rank_report`'s existing binary reference table calls
  `_paired_rows(reference_run, cand, metric, pack)` (`report.py:1455`, reference first) and then
  `_polarity_corrected` (`report.py:1226-1243`) to reorient the printed sign to "positive =
  candidate is better" — a pure sign-flip helper that reads only `metric`/`diff`/`ci`, with no
  kind-specific logic at all. **This means `_polarity_corrected` is directly reusable, unchanged,
  for the new continuous table**, calling `_paired_diffs(reference_run, cand, metric, pack)` in
  the identical reference-first argument order. No new polarity function is needed — confirmed by
  reading `mover_d_interval`'s arithmetic directly rather than assuming the analogy holds.

### 2.3 Pack shapes that ground the regression requirement and the "unreached today" claims

- `packs/embedder-graphrag-retrieval/pack.json`: `verdictMetrics: ["mrr"]`, `headlineMetric:
  "mrr"` — single-metric, all-continuous family; `members == family`.
- `packs/guard-judge-understanding/pack.json`: `verdictMetrics: ["falseAdvanceRate",
  "falseSuspendRate"]`, `headlineMetric: null` — two-metric, all-binary family; `members ==
  family`. **This is the pack the coordination ledger's regression constraint names** (`docs/plans/
  rank-continuous-reference-coordination.md`): its rendered rank report must be byte-for-byte
  identical before and after this change.
- No pack shipped today mixes binary and continuous `verdictMetrics`, and no pack's arms disagree
  with each other on a metric's kind — both refusal paths this plan adds are defensive, mirroring
  `compare_report`'s own mixed-kind guard's history ("equally unreached at the time it was
  written," `-ml` §3.3).

## 3. Design & rationale

### 3.1 `stats.continuous_verdict` gains `correction_k` — adopted verbatim from `-ml` §3.1

```python
def continuous_verdict(
    diffs: Sequence[float],
    *,
    metric_name: str,
    family: Sequence[str],
    alpha_family: float,
    unit_kind: str,
    design_effect: float,
    basis: Basis,
    B: int,
    seed: int,
    support: tuple[float, float] | None,
    correction_k: int | None = None,   # NEW
    a_label: str = "A",
    b_label: str = "B",
) -> ContinuousVerdict: ...
```

Inside the function, replace:

```python
k = len(family)
```

with:

```python
k = correction_k if correction_k is not None else len(family)
```

`alpha_used = alpha_family / k` (unchanged expression, now reading the new `k`). No other line in
the function's body changes. `family`'s only remaining job stays the `metric_name in family`
membership check (`stats.py:1707-1710`), untouched. `None` (the default) reproduces
`compare_report`'s one existing call site (§2.2 above) exactly, unchanged — verified by reading
that call site directly: it passes no `correction_k` today and must not need to.

**No `k >= 1` guard is added** (`-ml` §3.1's "minor, optional" item). Decision: leave both
`verdict()` and `continuous_verdict()` unguarded. Every real call site in this plan constructs `k =
len(family) * len(candidates)` where `family` is non-empty (a pack with no verdict metrics has
nothing to rank) and `candidates` is checked truthy before any reference-family code runs
(`rank_report`'s existing `if reference_run is not None and candidates:` guard, `report.py:1478`,
preserved unchanged) — so `k <= 0` is unreachable from any code this plan adds, matching `-ml`
§3.1's own finding that `verdict()`'s identical parameter has shipped unguarded with no reported
problem. Adding a guard to one function and not the other would be the exact inconsistency `-ml`
warns against; adding it to both is a genuine, separable follow-up with no bearing on D-1.

### 3.2 `ContinuousVerdict.text`'s coverage label — adopted verbatim from `-ml` §3.6

Immediately after `alpha_used = alpha_family / k` inside `continuous_verdict`, add:

```python
coverage_label = f"{100 * (1 - alpha_used):g}% CI"
```

and replace the two literal occurrences of `"95% CI"` inside the `text` f-strings
(`stats.py:1740-1753`) with `{coverage_label}`. At `k = 1` this reproduces `"95% CI"` byte-for-byte
(every pack shipped today); at `k = 4` it produces `"98.75% CI"` — the exact worked example `-ml`
§2.4/§3.6 gives.

### 3.3 The combined divisor — adopted verbatim from `-ml` §3.2

`k = len(family) * len(candidates)` for an all-continuous reference family — identical formula and
justification to the binary path's `correction_k = len(combined_p_values) = len(family) *
len(candidates)` (`report.py:1472`, confirmed unchanged in this plan). Computed directly (no
per-metric ladder, no `holm_steps` call — continuous has no p-value to step against, `-ml` §2.2).

### 3.4 Resolving question (a): does `_metric_kind`'s two-arg shape extend to N arms?

**No — not by the pairwise-anchored-at-the-reference pattern `-ml` §3.3 step 1 itself sketches,
and this plan does not use that pattern.** Tracing `_metric_kind`'s own logic
(`report.py:229-242`) against what an N-arm check needs to catch exposes a real gap in the note's
own suggested pseudocode, not just a shape question:

`_metric_kind(a, b, name)` returns `a`'s own declared kind whenever `a` has one, falling back to
`b` **only when `a` declares nothing at all** for that metric. Calling it as `_metric_kind
(reference_run, cand, metric)` for each candidate in turn (the note's literal sketch) therefore
returns the *same* value for every candidate whenever `reference_run` itself declares an aggregate
for that metric — it never even inspects `cand`'s own declared kind in that case. A genuine
disagreement between two *candidates* (or between a candidate and a reference that has already
declared its own kind) would be silently invisible to this pattern, precisely because it is
anchored at one arm whose own declaration always wins. This is worse than "does the shape extend
cleanly" — it is a pattern that answers a different, narrower question ("does the reference agree
with each candidate it can see") than the one needed ("do all arms — reference and every
candidate, pairwise — agree").

**Resolution: a small N-ary function, `_resolve_reference_kinds`, that inspects every arm's own
declared aggregate directly** — sharing, not duplicating, the isinstance check `_metric_kind`
already has (this codebase's own "two copies of a formula is one copy and one bug" convention,
`AGENTS.md`). Extract the shared predicate first:

```python
def _aggregate_kind(
    metric: BinaryMetric | ContinuousMetric | DistributionSummary | None,
) -> str:
    """"binary" or "continuous" for one already-resolved aggregate (or none declared, which
    resolves "binary" — `_metric_kind`'s own pre-Table-F default). The one isinstance check both
    `_metric_kind` and `_resolve_reference_kinds` need, so it lives in exactly one place."""
    return "continuous" if isinstance(metric, (ContinuousMetric, DistributionSummary)) else "binary"
```

Rewrite `_metric_kind` (behavior-preserving — this is a pure internal refactor, not a change in
what it returns for any input) to call it:

```python
def _metric_kind(a: RunResult, b: RunResult, name: str) -> str:
    return _aggregate_kind(_metric_aggregate(a, name) or _metric_aggregate(b, name))
```

Then the new N-ary resolver, placed near `_metric_kind` in `report.py`:

```python
def _resolve_reference_kinds(
    reference_run: RunResult, candidates: Sequence[RunResult], family: Sequence[str]
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Each family member's kind, resolved across every arm the reference-anchored family
    touches — reference and every candidate — not just a reference-anchored pair at a time.

    Deliberately does NOT delegate to `_metric_kind(a, b, name)` pairwise, anchored at
    `reference_run`: `_metric_kind` always prefers its first argument's own declaration when it
    has one, so anchoring every pairwise call at `reference_run` would silently miss two
    *candidates* disagreeing with each other whenever `reference_run` itself declares any
    aggregate for that metric — exactly the failure this function exists to catch.

    Returns `(kinds, per_arm)`. `kinds[metric]` is the resolved kind ("binary"/"continuous"),
    present only when every arm that declares an aggregate for `metric` agrees (falling back to
    "binary" when no arm declares one at all, mirroring `_metric_kind`'s own default). A `metric`
    absent from `kinds` is a genuine cross-arm disagreement; `per_arm[metric]` always carries
    every arm that declared an aggregate for it, keyed by `modelKey`, for the disagreement banner
    to name them regardless of whether `metric` resolved cleanly.
    """
    kinds: dict[str, str] = {}
    per_arm: dict[str, dict[str, str]] = {}
    for metric in family:
        declared: dict[str, str] = {}
        for run in (reference_run, *candidates):
            agg = _metric_aggregate(run, metric)
            if agg is not None:
                declared[run.modelKey] = _aggregate_kind(agg)
        per_arm[metric] = declared
        distinct = set(declared.values())
        if len(distinct) <= 1:
            kinds[metric] = next(iter(distinct), "binary")
    return kinds, per_arm
```

This is the shape decision `-ml` §5 left open: **a small N-ary wrapper, not a reuse of
`_metric_kind`'s two-arg shape**, because the two-arg shape's own preference order is the wrong
tool for detecting disagreement among more than two arms — it was built to pick a kind when one
arm might not declare a given metric at all (two-arm comparison), not to audit agreement among
many.

### 3.5 Resolving question (b): how should a genuine cross-arm kind disagreement be rendered?

> **Revision (2026-09-20, `tdd-engineer`, folding in `data-scientist`'s methodology review of this
> plan, `docs/reviews/rank-continuous-reference-ml.md` §3):** this section's rationale is rewritten
> in place — the plan is still `Status: active`, not yet executed against at the time of this
> revision — to fix an overstated argument the review's finding 3 identified. The *behavior* this
> section specifies (whole-family refusal, with a distinct, louder banner) is unchanged and was
> never in question; only the *justification* below moves.

`-ml` §5 frames this as a choice between two options — silent refuse-whole (mirroring the ordinary
pre-registered mixed-kind case) or a sharper, visible integrity banner (mirroring the version/hash/
schema banners, `report.py:1396-1420`) — and leans toward the latter without deciding.

**Decision: both, but on two different axes, not as a single either/or.** The two cases are
different kinds of uncertainty, and they are refused whole for different reasons.

An ordinary **pre-registered** mixed family (e.g. one binary and one continuous verdict metric, by
design) has every member's kind already resolved and known — the problem is purely structural:
there is no designed procedure that spans a Holm-stepped p-value ladder and a bootstrap-interval-
widening family in one combined correction, because no such procedure exists anywhere in this
codebase. Refusing that family whole (`-ml` §3.3: "`k` is `len(verdictMetrics)` and pre-registered,
so dropping the minority kind would shrink `k` after the results exist and under-correct the
survivors") is a scoping decision about an undesigned mechanism — a real, load-bearing reason, but
a mechanical/multiplicity one, and `_MIXED_FAMILY_MEMBER`'s calm "exploratory" tone is correct for
this foreseen, legitimate pack shape.

A genuine **cross-arm kind disagreement** is not the same kind of problem, and does not need the
same kind of argument to justify the same whole-family refusal. Its binding constraint is that the
metric's own type cannot be established at all, independent of any multiplicity concern:
`correction_k = len(family) * len(candidates)` (§3.2's formula) is a pure count, computed from
`family`'s and `candidates`' sizes, and does **not** mechanically depend on any metric's kind
resolving cleanly — nothing forces `k` to shrink if only the disagreeing metric's own cells were
refused while a sibling, cleanly-resolved metric's ladder still ran at the full, correctly-sized
`k` (this codebase already has the mechanism for exactly that shape: `rank_report`'s existing
binary combined ladder keeps a zero-paired-data `(candidate, metric)` cell *in* the combined
`p_values` list via `mcnemar_exact(0, 0) = 1.0`, consuming a Holm rank without ever being decided).
So a disagreement is refused whole for a **data-integrity reason, primary**: a metric's aggregate
kind is supposed to be a type fact, and a metric that resolves to a different type depending on
which arm answers it is a self-contradictory record — the same class of problem as a pack-version
or schema-version mismatch (`report.py:1396-1420`, "visible, never silent, never a reason to drop
a record"), neither of which invokes a multiplicity argument either. Once one metric's type
declaration is shown unreliable for some arm, that arm's *other* declared aggregates are owed the
same distrust, which is why the whole reference-anchored family is refused rather than only the
disagreeing metric's own cells — a conservative response to epistemic uncertainty about the data,
not a forced consequence of `k`-shrinkage mechanics. (`k` staying fixed across the family for the
clean-metric case is a real, secondary property this plan's design already delivers — see §3.7's
rejected alternative — not the forcing reason for the disagreement case's own refusal.)

So this plan prints a distinct, loud, named banner for a disagreement (§4 step 5 below) in addition
to (not instead of) refusing the whole reference family — reconciling both of `-ml`'s options rather
than picking one exclusively, with each case's own refusal resting on the argument that actually
holds for it.

### 3.6 The continuous reference-family rendering shape — adopted from `-ml` §3.4, with the polarity/orientation and formatting decisions resolved in §2.2 above

Columns: `| candidate | diff | CI | decision |` (one row per candidate; the caption states the
actual correction, so no per-row threshold column exists, mirroring `-ml` §3.4's reasoning that
nothing steps here). Diff/CI print as **plain decimals** (`+.3f`, matching
`continuous_verdict.text`'s own convention and `_render_one_rank_table`'s continuous branch,
`report.py:1201`), never through `_pp()` — `_pp` multiplies by 100 for a *rate of an event*, and
`mrr` is a ratio with no percentage-point semantics; applying `_pp` to it would silently mislabel
the unit, unlike the binary table where `_pp` is correct.

### 3.7 Rejected alternative: dropping the disagreeing metric and rendering the rest

Considered and rejected: letting a disagreeing metric's row alone refuse while the family's
*other*, cleanly-resolved metrics still build a combined ladder/correction over the shrunken
remainder. Rejected because it reproduces exactly the "k shrinks after the results exist" failure
the mixed-kind guard was built to prevent (§3.5 above) — the combined `k` must be fixed by
pre-registration, and letting one metric's disagreement silently resize it for the others is a
worse, more silent version of the very bug this whole plan closes.

### 3.8 Rejected alternative: mirroring `compare_report`'s per-metric tally under refusal

`compare_report`'s own mixed-kind refusal still renders a per-candidate paired-n tally for each
metric (§2.2 above), because each metric's own kind is individually resolved even when the
*combination* is mixed. This plan does **not** mirror that for `rank_report`'s refusal paths
(neither the ordinary mixed case nor the disagreement case): for the disagreement case there is no
resolvable kind to pick `_paired_rows` vs. `_paired_diffs` by, so no tally can be built safely; for
consistency, this plan renders no tally in the ordinary-mixed sub-case either, matching `-ml`
§3.3's own literal instruction ("skip both the binary-ladder and continuous-family code below
entirely — no partial table for either kind"). Both refusal shapes are unreached by any pack
shipped today (§2.3), so this is a scope-tightening choice for defensive code, not a regression
against anything tested.

## 4. Step-by-step implementation

Sequenced so the tree stays buildable at each step; steps 1-2 are pure `stats.py` additions with
no caller yet, steps 3-6 are `report.py` additions with no behavior change until step 7 wires them
in, step 7 is the one behavior-changing step, step 8 is the two adjacent-defect fixes.

### Step 1 — `modelbench/stats.py`: `continuous_verdict` gains `correction_k`

File: `modelbench/stats.py`, function `continuous_verdict` (currently `stats.py:1651-1769`).

- Add `correction_k: int | None = None` to the signature, positioned exactly where `-ml` §3.1
  places it (after `support`, before `a_label`).
- Replace `k = len(family)` with `k = correction_k if correction_k is not None else len(family)`.
- Update the docstring: add a paragraph mirroring `verdict()`'s own `correction_k` paragraph
  (`stats.py:1230-1238`), citing this plan's §3.1.

Done when: `continuous_verdict(..., correction_k=None)` and the omitted-keyword form produce
identical `ContinuousVerdict` objects on any fixture, and `compare_report`'s existing continuous
call site (which passes no `correction_k`) is unaffected — verified by the existing suite staying
green with no other file touched yet.

### Step 2 — `modelbench/stats.py`: `ContinuousVerdict.text`'s coverage label

Same function. Immediately after computing `alpha_used = alpha_family / k`, add:

```python
coverage_label = f"{100 * (1 - alpha_used):g}% CI"
```

Replace both literal `"95% CI"` occurrences in the `text` f-strings (`stats.py:1740-1753`) with
`{coverage_label}`.

Done when: every existing test asserting `"95% CI"` in `ContinuousVerdict.text` still passes at
`k=1` (unchanged output), and a new test at `k>1` (§5, test 3) asserts the correct percentage.

### Step 3 — `modelbench/report.py`: extract `_aggregate_kind`, refactor `_metric_kind`

Near `_metric_kind` (`report.py:229-242`):

```python
def _aggregate_kind(
    metric: BinaryMetric | ContinuousMetric | DistributionSummary | None,
) -> str:
    return "continuous" if isinstance(metric, (ContinuousMetric, DistributionSummary)) else "binary"


def _metric_kind(a: RunResult, b: RunResult, name: str) -> str:
    return _aggregate_kind(_metric_aggregate(a, name) or _metric_aggregate(b, name))
```

Pure refactor — no caller of `_metric_kind` (only `compare_report:1672`) changes behavior. Done
when: the full existing suite is green with no assertion changes.

### Step 4 — `modelbench/report.py`: `_resolve_reference_kinds`

Add the function exactly as specified in §3.4 above, placed near `_metric_kind`/`_aggregate_kind`.
Not yet called from anywhere — a standalone, directly-testable addition.

Done when: the new unit tests in §5 (tests 5-8) pass against this function called directly, with
no `rank_report` change yet.

### Step 5 — `modelbench/report.py`: new rendering functions and constants

Add, near the existing `_MIXED_FAMILY_MEMBER`/`_render_reference_family` (`report.py:765-802`,
`1246-1295`):

```python
#: Ordinary pre-registered mixed-kind refusal for `rank_report`'s reference-anchored family — the
#: sibling of `_MIXED_FAMILY_MEMBER` (report.py:769), reworded for "no candidate's row is
#: verdicted against the reference" rather than "no member of the two-arm comparison is verdicted".
_MIXED_REFERENCE_FAMILY_MEMBER = (
    "**{kind} metric — no reference-anchored verdict.** This pack's pre-registered "
    "verdict-metric family mixes binary and continuous members, so the optional "
    "reference-anchored family (FR-8) has no single Holm ladder or single bootstrap correction "
    "it can apply across the whole family — mirroring `compare_report`'s own two-arm "
    "mixed-family refusal. No candidate's row for `{metric}` is verdicted against `{reference}`."
)

#: A genuine cross-arm kind DISAGREEMENT (not a pre-registered mixed family) — sharper framing
#: than `_MIXED_REFERENCE_FAMILY_MEMBER`, printed for the metric(s) named in the banner above it.
_REFERENCE_KIND_DISAGREEMENT_MEMBER = (
    "**kind disagreement — no reference-anchored verdict.** Arms in this ranking do not agree "
    "on `{metric}`'s aggregate kind (see the banner above); no candidate's row for this metric "
    "is verdicted against `{reference}`."
)

#: A metric that resolved CLEANLY but is refused anyway because a SIBLING metric in the same
#: pre-registered family has a cross-arm disagreement (§3.5/§4 step 7) — this metric's own block
#: must not say "mixes binary and continuous members" (`_MIXED_REFERENCE_FAMILY_MEMBER`), because
#: for THIS metric that is not true; the reason it is refused lives one metric over.
_REFERENCE_FAMILY_SIBLING_DISAGREEMENT_MEMBER = (
    "**no reference-anchored verdict.** A sibling metric in this pack's pre-registered "
    "verdict-metric family has a cross-arm kind disagreement (see the banner above); the whole "
    "optional reference-anchored family (FR-8) is refused because `k` is fixed across the whole "
    "family and cannot be honestly re-sized once results already exist. No candidate's row for "
    "`{metric}` is verdicted against `{reference}`."
)

#: Printed once, before the per-metric loop, when `_resolve_reference_kinds` finds a genuine
#: cross-arm disagreement — mirrors PACK VERSION MISMATCH / SCHEMA VERSIONS IN THIS COMPARISON
#: (report.py:1396-1420): visible, never silent, never a reason to drop a record.
_REFERENCE_KIND_DISAGREEMENT_EXPLANATION = (
    "This is a data-integrity problem, not a foreseen pre-registration choice: a metric's "
    "aggregate kind is supposed to be a type fact, never something that varies by which arm "
    "answers it. The whole optional reference-anchored family (FR-8) is refused for the "
    "metric(s) named above, for the same mechanical reason a pre-registered mixed-kind family "
    "is refused whole: `k` is fixed across the family and cannot be honestly re-sized once "
    "results already exist."
)


def _render_reference_kind_disagreement_banner(
    disagreements: Sequence[str], per_arm: Mapping[str, Mapping[str, str]]
) -> list[str]:
    lines = ["> **REFERENCE-FAMILY KIND DISAGREEMENT**", ">"]
    for metric in disagreements:
        detail = ", ".join(f"`{model}`: {kind}" for model, kind in per_arm[metric].items())
        lines.append(f"> - `{metric}` — {detail}")
    lines += ["> ", "> " + _REFERENCE_KIND_DISAGREEMENT_EXPLANATION, ""]
    return lines


def _render_reference_family_refused(
    metric: str, reference_run: RunResult, text: str
) -> list[str]:
    return [
        f"#### Reference-anchored family — {metric} vs `{reference_run.modelKey}`", "",
        text, "",
    ]


def _render_reference_family_continuous(
    *,
    metric: str,
    pack: PackRef,
    reference_run: RunResult,
    candidates: Sequence[RunResult],
    correction_k: int,
    unit_kind: str,
) -> list[str]:
    """The continuous sibling of `_render_reference_family` (report.py:1246). Every candidate's
    interval is decided independently at the same fixed `alpha_used` (`-ml` §3.4) — no Holm
    ladder, no per-row threshold, no "not tested" state."""
    alpha_used = pack.metrics.alpha_family / correction_k
    lines = [
        f"#### Reference-anchored family — {metric} vs `{reference_run.modelKey}`", "",
        "_exploratory — no significance claim outside this family_", "",
        f"_`alpha_family={pack.metrics.alpha_family:g}, k={correction_k}, "
        f"alpha_used={alpha_used:g} ({100 * (1 - alpha_used):g}% CI)`_", "",
        "| candidate | diff | CI | decision |",
        "|---|---|---|---|",
    ]
    for candidate in candidates:
        diffs_row = _paired_diffs(reference_run, candidate, metric, pack)
        if not diffs_row.diffs:
            lines.append(f"| {candidate.modelKey} | — | — | no verdict — no paired data |")
            continue
        if len(diffs_row.diffs) == 1:
            lines.append(f"| {candidate.modelKey} | — | — | no verdict — one paired unit |")
            continue
        support_metric = (
            _metric_aggregate(reference_run, metric) or _metric_aggregate(candidate, metric)
        )
        support = support_metric.support if support_metric is not None else None
        cv = stats.continuous_verdict(
            diffs_row.diffs, metric_name=metric, family=[metric],
            alpha_family=pack.metrics.alpha_family, unit_kind=unit_kind,
            design_effect=max(reference_run.designEffect, candidate.designEffect),
            basis=min(
                (reference_run.basis, candidate.basis), key=_BASIS_STRENGTH.__getitem__
            ),
            B=_BOOTSTRAP_B, seed=pack.seed, support=support, correction_k=correction_k,
            a_label=reference_run.modelKey, b_label=candidate.modelKey,
        )
        diff, ci = _polarity_corrected(metric, cv.diff, cv.ci)
        decision = "distinguishable" if cv.distinguishable else "not distinguishable"
        lines.append(
            f"| {candidate.modelKey} | {diff:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {decision} |"
        )
    lines.append("")
    return lines
```

Note: `support_metric.support` requires `ContinuousMetric`/`DistributionSummary` to expose a
`.support` field — confirm against `results.py`'s dataclass definitions before writing the test
fixtures (the existing `compare_report` continuous branch, `report.py:1739`, already reads
`support_metric.support` the same way — this is read-only precedent, not a new assumption).

Not yet called from `rank_report` — directly testable in isolation first (§5 tests 9-16, once
step 7 wires them in; the rendering helpers have no meaningful behavior to assert before that
wiring exists, unlike step 4's standalone `_resolve_reference_kinds`).

### Step 6 — `modelbench/report.py`: gate `_rank_resolving_power_lines` on kind

File: `report.py`, function `_rank_resolving_power_lines` (currently `report.py:1298-1352`).
Restructure the top of the function from:

```python
    n = len(runs)
    k = len(family) * (n - 1)
    if k <= 0:
        return []
    unit_kind = unit_kind_for_role(pack.role)
    ns = [agg.n for r in runs for agg in [_metric_aggregate(r, metric)] if agg is not None]
    if not ns:
        return []
    n_units = max(ns)
```

to:

```python
    n = len(runs)
    k = len(family) * (n - 1)
    if k <= 0:
        return []
    unit_kind = unit_kind_for_role(pack.role)
    aggs = [_metric_aggregate(r, metric) for r in runs]
    first_agg = next((a for a in aggs if a is not None), None)
    if first_agg is None:
        return []
    if _aggregate_kind(first_agg) == "continuous":
        return []
    ns = [a.n for a in aggs if a is not None]
    n_units = max(ns)
```

The rest of the function is unchanged (`ns` still computed identically, just from `aggs` rather
than a fresh generator expression over the same values). Done when: this returns `[]` for
`embedder-graphrag-retrieval`'s `mrr` with or without `--reference`, and every existing
binary-pack test for this function is unaffected (§5 tests 17-18).

### Step 7 — `modelbench/report.py`: wire the dispatch into `rank_report`

File: `report.py`, function `rank_report` (`report.py:1355-1486`). Replace the combined-family
construction block (currently lines 1445-1472) with:

```python
    combined_p_values: list[float] = []
    combined_cells: list[tuple[str, str]] = []
    combined_outcomes: dict[tuple[str, str], stats.PairedOutcomes] = {}
    candidates: list[RunResult] = []
    combined_steps: list[stats.HolmStep] = []
    correction_k = 0
    reference_kinds: dict[str, str] = {}
    reference_kind_details: dict[str, dict[str, str]] = {}
    reference_kind_disagreements: list[str] = []
    reference_family_refused = False
    if reference_run is not None:
        candidates = [r for r in runs if r.modelKey != reference_run.modelKey]
        if candidates:
            reference_kinds, reference_kind_details = _resolve_reference_kinds(
                reference_run, candidates, family
            )
            reference_kind_disagreements = [m for m in family if m not in reference_kinds]
            resolved_kinds = set(reference_kinds.values())
            reference_family_refused = (
                bool(reference_kind_disagreements) or len(resolved_kinds) > 1
            )
            if reference_kind_disagreements:
                lines += _render_reference_kind_disagreement_banner(
                    reference_kind_disagreements, reference_kind_details
                )
            if not reference_family_refused and resolved_kinds == {"binary"}:
                # --- unchanged existing binary combined-ladder construction ---
                for metric in family:
                    for cand in candidates:
                        paired = _paired_rows(reference_run, cand, metric, pack)
                        outcomes = stats.PairedOutcomes.from_units(
                            unit_kind, list(zip(paired.unit_ids, paired.a_ok, paired.b_ok))
                        )
                        combined_outcomes[(cand.modelKey, metric)] = outcomes
                        _a, b_, c_, _d = outcomes.table
                        combined_p_values.append(stats.mcnemar_exact(b_, c_))
                        combined_cells.append((cand.modelKey, metric))
                combined_steps = stats.holm_steps(
                    combined_p_values, alpha=pack.metrics.alpha_family
                )
                correction_k = len(combined_p_values)
            elif not reference_family_refused:  # resolved_kinds == {"continuous"}
                correction_k = len(family) * len(candidates)
```

and replace the per-`metric in members` reference-family call (currently lines 1478-1483) with:

```python
        if reference_run is not None and candidates:
            if reference_family_refused:
                if metric in reference_kind_disagreements:
                    text = _REFERENCE_KIND_DISAGREEMENT_MEMBER.format(
                        metric=metric, reference=reference_run.modelKey
                    )
                elif reference_kind_disagreements:
                    # This metric itself resolved cleanly; a SIBLING metric's disagreement is
                    # what forced the whole family refused (§3.5) — `_MIXED_REFERENCE_FAMILY_
                    # MEMBER`'s "mixes binary and continuous members" wording would misstate the
                    # reason for THIS metric's own refusal.
                    text = _REFERENCE_FAMILY_SIBLING_DISAGREEMENT_MEMBER.format(
                        metric=metric, reference=reference_run.modelKey
                    )
                else:
                    text = _MIXED_REFERENCE_FAMILY_MEMBER.format(
                        kind=reference_kinds[metric], metric=metric,
                        reference=reference_run.modelKey,
                    )
                lines += _render_reference_family_refused(metric, reference_run, text)
            elif reference_kinds[metric] == "binary":
                lines += _render_reference_family(
                    metric=metric, pack=pack, reference_run=reference_run,
                    candidates=candidates, cells=combined_cells, steps=combined_steps,
                    outcomes=combined_outcomes, correction_k=correction_k, unit_kind=unit_kind,
                )
            else:
                lines += _render_reference_family_continuous(
                    metric=metric, pack=pack, reference_run=reference_run,
                    candidates=candidates, correction_k=correction_k, unit_kind=unit_kind,
                )
```

Everything above this block in `rank_report` (the version/hash/schema banners, the
`family`/`members`/`unit_kind`/`reference_run` resolution) is unchanged. `_render_one_rank_table`
and `_rank_resolving_power_lines` calls inside the `for metric in members` loop are unchanged
except for step 6's internal gating.

Done when: `test_rank_report_guard_judge_reference_family_uses_one_combined_ladder` and
`test_rank_report_guard_judge_family_k_does_not_shrink_when_a_candidate_has_no_paired_data`
(`tests/test_report.py:3558-3624`, both exercise `guard-judge-understanding`, an all-binary
family) pass unchanged, and `./run.sh rank --pack embedder-graphrag-retrieval --reference bm25`
against the real stored corpus exits 0.

### Step 8 — verify the two adjacent-defect fixes land inside this same unit

Steps 2 and 6 above already are the two fixes `-ml` §3.5/§3.6 require in this unit, not
separately. No additional code beyond steps 2 and 6 is needed for them — this step is a checkpoint,
not new work: confirm both are covered by the tests in §5 (tests 3, 17-18) before calling the unit
done.

### Step 9 — manual callout

`docs/manuals/small-model-benchmarking.md` documents D-1's workaround (`compare` instead of
`rank --reference`) per the coordination ledger's own note. Once this unit is green, revisit that
callout — either remove it or add a "fixed as of `<HISTORY.md entry>`" pointer. This is `tico`'s
edit, not this plan's implementer's; flagging it here only so `teco`'s U4 (`docs/BACKLOG.md`
removal + `docs/HISTORY.md` entry) does not drop it.

## 5. Test strategy

Extends, not replaces, `-ml` §4's eight test obligations (cited here as "`-ml` test N"). New tests
below use this repo's existing fixture helpers — `_embedder_pack`, `_mrr_item`,
`RetrievalAggregates`, `ContinuousMetric`, `run(..., role="embedder", call_surface="embeddings",
...)`, `embeddings_fields`/`deterministic_fields` (all in `tests/test_report.py`, already used by
the existing continuous-`compare_report` tests at `tests/test_report.py:1570-1701`) for
`report.py` tests, and the existing `_outcomes`/`_rp` helpers plus `continuous_verdict`'s own
fixture shape (`tests/test_stats.py:2508-2882`) for `stats.py` tests. `_rank_pack`/`_rank_arm`
(`tests/test_report.py:3113-3145`) are the binary rank-report fixture helpers; add a parallel
`_rank_mrr_arm(model_key, values, ...)` returning a `run(..., role="embedder",
call_surface="embeddings", ...)` built from `_mrr_item`/`RetrievalAggregates`, for the new
continuous rank fixtures.

### `stats.py` unit tests (new)

1. **`test_continuous_verdict_correction_k_defaults_to_len_family_and_matches_the_omitted_path`**
   — mirrors `test_verdict_correction_k_defaults_to_len_family_and_matches_the_omitted_path`
   (`tests/test_stats.py:567-584`) exactly: `correction_k=None` and the omitted keyword produce
   identical `ContinuousVerdict.text` on a `len(family) == 1` fixture (`-ml` test 1).
2. **`test_continuous_verdict_correction_k_decouples_the_divisor_from_the_familys_own_length`** —
   mirrors `test_verdict_correction_k_decouples_the_divisor_from_the_familys_own_length`
   (`tests/test_stats.py:587-607`): `correction_k=16` on a `family=["mrr"]` fixture produces
   `alpha_used = alpha_family / 16`, a materially wider interval than `correction_k=None` on
   identical `diffs`, and flips `distinguishable` from `True` to `False` on a fixture engineered
   to be significant only at the uncorrected level (`-ml` test 2).
3. **`test_continuous_verdict_text_reports_the_correct_coverage_at_k_greater_than_1`** — direct
   regression test for §3.2/§3.6: build with `family=["mrr", "sep_z"]` (so `k=2`,
   `alpha_used=0.025`) and assert `"97.5% CI"` appears in `.text`, never `"95% CI"` (`-ml` test 3).
4. **`test_continuous_verdict_correction_k_boundary_value_does_not_silently_round`** — `k` large
   enough that `alpha_family/(2k)` is a small but exact `Fraction` (e.g. `k=100`); assert `.text`'s
   printed coverage and `.alpha_used` are numerically exact against the closed-form expression, not
   merely "close" (`-ml` test 4; this repo's real scale tops out at `k=36`, so this is a boundary
   check, not a realistic-scale one).

### `report.py` unit tests — `_resolve_reference_kinds` and `_aggregate_kind` (new)

5. **`test_resolve_reference_kinds_agrees_when_every_arm_declares_the_same_kind`** — reference plus
   two candidates, all declaring `ContinuousMetric` for `mrr`: `kinds == {"mrr": "continuous"}`,
   no disagreement.
6. **`test_resolve_reference_kinds_falls_back_to_binary_when_no_arm_declares_an_aggregate`** —
   mirrors `_metric_kind`'s own pre-Table-F default.
7. **`test_resolve_reference_kinds_catches_two_candidates_disagreeing_with_each_other`** —
   reference declares no aggregate for the metric at all (or the metric is absent from its
   `named_metrics()`), one candidate declares `ContinuousMetric`, another declares `BinaryMetric`
   for the same metric name: `metric not in kinds`, and `per_arm[metric]` names both candidates'
   modelKeys with their distinct kinds. **This is the exact case a pairwise-anchored-at-reference
   check (the note's literal §3.3 step 1 sketch) would miss** — the test's docstring should say so,
   since it is this plan's own justification for not implementing that sketch literally (§3.4
   above).
8. **`test_resolve_reference_kinds_catches_a_candidate_disagreeing_with_a_declaring_reference`** —
   reference declares `ContinuousMetric`, one candidate declares `BinaryMetric` for the same
   metric: also a disagreement, confirming the fix is not narrower than the case in test 7.

### `report.py` integration tests — the continuous reference family (new)

9. **`test_rank_report_reference_family_continuous_renders_k_and_four_state_decisions`** — the
   direct regression test for D-1 itself, mirroring `-ml` test 5:
   `_embedder_pack()`-shaped fixture, reference plus >= 3 candidates spanning all four decision
   states — `distinguishable`, `not distinguishable`, `no verdict — no paired data` (disjoint item
   ids vs. the reference), `no verdict — one paired unit` (exactly one shared item id) — asserts
   `correction_k = len(family) * len(candidates)` appears in the caption
   (`` `alpha_family=0.05, k=<n>, alpha_used=... `` ), the table has exactly the four decision
   strings used above and never `"not tested"`/`"below the observable floor"` (binary-only
   vocabulary), and no `MetricKindError`/crash.
10. **`test_rank_report_reference_family_continuous_caption_alpha_used_matches_each_rows_cv`** —
    reads the caption's own `alpha_used` number back out and asserts it equals
    `pack.metrics.alpha_family / correction_k` exactly, closing the "two independent computations
    of the same trivial formula" seam noted in §3.6/step 5 above (a drift-detection pin, not a
    correctness requirement on its own).
11. **`test_rank_report_reference_family_continuous_diff_and_ci_print_as_plain_decimals_not_pp`**
    — asserts `" pp"` never appears inside the continuous family table's own section, mirroring
    `test_a_continuous_verdict_member_is_routed_through_continuous_verdict_not_booleanised`'s own
    `" pp" not in section` assertion (`tests/test_report.py:1604`) on the two-arm path.
12. **`test_rank_report_reference_family_continuous_polarity_matches_the_binary_convention`** —
    construct a fixture where the candidate has a **higher** `mrr` than the reference; assert the
    printed `diff` is **positive** (candidate-better reads positive, exactly as the binary table's
    `_polarity_corrected` convention already guarantees) — the direct pin for §2.2's
    orientation-reuse argument, since nothing else in this plan tests it in isolation.

### `report.py` integration tests — mixed/disagreement refusal (new)

13. **`test_rank_report_reference_family_mixed_pack_refuses_whole_no_partial_table`** — mirrors
    `-ml` test 6, extended: a synthetic pack with one binary and one continuous verdict metric,
    `--reference` given, `>= 2` candidates. Assert: both metrics' reference-family sections render
    `_MIXED_REFERENCE_FAMILY_MEMBER`'s text, no candidate table (no `|---|` row) appears under
    either, and — the extension beyond `-ml` test 6 — the `#### Reference-anchored family` header
    still appears (so a reader knows FR-8 was requested and explicitly refused, not silently
    skipped).
14. **`test_rank_report_reference_family_kind_disagreement_prints_the_named_banner_and_refuses`**
    — a fixture where the reference declares no aggregate for a single-metric pack's own verdict
    metric and two candidates disagree with each other (mirroring unit test 7's fixture, driven
    through the full `rank_report`, not `_resolve_reference_kinds` directly). Assert: the `>
    **REFERENCE-FAMILY KIND DISAGREEMENT**` banner appears once, before any `#### Reference-
    anchored family` header, naming the metric and both candidates' distinct kinds; the metric's
    own reference-family section renders `_REFERENCE_KIND_DISAGREEMENT_MEMBER`'s text, not
    `_MIXED_REFERENCE_FAMILY_MEMBER`'s; no candidate table renders.
15. **`test_rank_report_reference_family_disagreement_banner_never_fires_for_ordinary_mixed`** —
    negative pin: the ordinary mixed-family fixture from test 13 must **not** print the
    `REFERENCE-FAMILY KIND DISAGREEMENT` banner (guards against the two refusal paths being
    conflated in the implementation).
16. **`test_rank_report_reference_family_sibling_disagreement_gets_its_own_message_not_mixed_text`**
    — a three-metric family (one binary, one continuous, one disagreeing) with `--reference`
    given: the disagreeing metric's block uses `_REFERENCE_KIND_DISAGREEMENT_MEMBER`; the *other
    two*, cleanly-resolved metrics' blocks use `_REFERENCE_FAMILY_SIBLING_DISAGREEMENT_MEMBER`,
    **never** `_MIXED_REFERENCE_FAMILY_MEMBER`'s "mixes binary and continuous members" wording —
    which would misstate the reason for a metric that did not itself disagree with anything (§3.5,
    step 7's three-way message selection).

### `report.py` unit tests — the `_rank_resolving_power_lines` fix (new)

17. **`test_rank_resolving_power_lines_returns_empty_for_a_continuous_metric_with_no_reference`**
    — direct regression test for §2.3/`-ml` test 7: `_embedder_pack()`-shaped `runs` with `>= 2`
    arms, `reference` not given at all (this defect fires independent of `--reference`); assert
    `_rank_resolving_power_lines(...) == []`, and — driven through the full `rank_report` — assert
    `"resolves differences of"` / `"80% power"` / `"pp"` never appear anywhere in the `### mrr`
    section.
18. **`test_rank_resolving_power_lines_still_fires_for_a_binary_metric_unchanged`** — negative pin
    using the existing `_rank_pack()`/`guard_pack()` binary fixtures: confirms step 6's
    restructuring changed no binary-path output, run against
    `test_rank_report_no_reference_resolving_power_sentence_is_alongside_the_pack_line` and
    `test_rank_report_no_reference_resolving_power_sentence_doubles_k_for_guard_judge`
    (`tests/test_report.py:3659-3687`) staying green unmodified — no new test needed beyond
    confirming those two still pass, listed here for completeness of the done-condition.

### Regression pin — the non-negotiable constraint

19. **`test_rank_report_guard_judge_binary_reference_family_is_byte_for_byte_unchanged`** — the
    direct pin for the coordination ledger's stated constraint (`-ml` test 8, restated as its own
    named test rather than folded into an existing one, since it is the ledger's explicit
    non-negotiable rather than an ordinary regression). Render `rank_report` for
    `guard-judge-understanding` with a `reference` argument and >= 2 candidates **before** this
    unit's code lands (capture the string in a fixture/golden constant from the current tree) and
    assert the **identical string** renders after — not "contains the same substrings," the full
    `==`. Combine with `test_rank_report_guard_judge_reference_family_uses_one_combined_ladder`
    and `test_rank_report_guard_judge_family_k_does_not_shrink_when_a_candidate_has_no_paired_data`
    (already-passing tests 3558-3624) as the three tests that jointly satisfy the constraint —
    run all three, not just the byte-for-byte one, since the existing two also assert *mechanism*
    (one combined `holm_steps` call, `k` not shrinking) that a byte-for-byte string diff alone
    would not localize on failure.

### Acceptance threshold (unchanged from `-ml` §4, restated)

All 19 tests above green, plus a live re-run of `./run.sh rank --pack embedder-graphrag-retrieval
--reference bm25` against the real stored corpus: exits 0, renders a reference-family table whose
`k`, decision states, and CI coverage label are manually verifiable against this plan's formulas —
the same live-verification bar `docs/test-reports/small-model-benchmarking-manual-additions-
report.md` originally applied to find D-1.

## 6. Risks & open questions

- **Both of `-ml` §5's open questions are resolved above** (§3.4, §3.5) — not left for the
  implementer. §3.4's resolution also corrects a gap in the note's own suggested pseudocode (the
  pairwise-anchored-at-reference pattern under-detects disagreement); flagging this explicitly so
  a reviewer checks the reasoning in §3.4 rather than assuming the note's sketch was directly
  implementable.
- **The disagreement-detection code path (§3.4/§3.5, steps 4 and 7) is unreached by any pack
  shipped today** (§2.3) — same status `compare_report`'s own mixed-kind guard had when written.
  Correct, defensive, and tested at the unit level (tests 5-8, 14-15), but there is no live pack
  to re-verify it against in the acceptance threshold's live re-run — the live check only exercises
  the all-continuous happy path (`embedder-graphrag-retrieval`) and the all-binary regression
  (`guard-judge-understanding`).
- **`support_metric.support` field access** (step 5's `_render_reference_family_continuous`) is
  read-only precedent copied from `compare_report`'s existing continuous branch
  (`report.py:1739`) — not independently re-verified against `results.py`'s dataclass definitions
  in this plan; the implementer should confirm the field name once while writing step 5, since a
  typo here would surface only as an `AttributeError` at render time, not at import time.
- **Rollback:** every change is additive or behind a new dispatch branch except the two `stats.py`
  edits (steps 1-2) and the `_metric_kind` refactor (step 3), all three of which are proven
  behavior-preserving by the existing suite staying green before any new test is added. If step 7's
  wiring needs to be reverted independently of steps 1-6, the `rank_report` diff is isolated to the
  two blocks named in step 7 and reverts cleanly without touching `stats.py`.
- **Performance:** no change to bootstrap resample counts or Holm ladder sizes; `_resolve_
  reference_kinds` is `O(len(family) * len(candidates))` dictionary lookups, negligible next to the
  `B=10000` bootstrap resamples already dominating this code path.
