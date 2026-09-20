"""The markdown comparison — where AC-2, AC-3 and AC-4 become visible output.

Design: `docs/plans/small-model-benchmarking.md` §3.4.3, §3.5, §3.7 and §3.9; the statistics and
every string they carry are `docs/plans/small-model-benchmarking-ml.md`'s (§3.2e's three verdicts,
§7.1's resolving-power template, §7.2's rendered line).

Three refusals here are **missing functions, not guarded ones** (§3.5) — a rule you cannot express
is a rule you cannot break under deadline pressure:

* there is no code path that synthesises a headline when `headlineMetric` is `null`;
* there is no path that pools a per-class table into one accuracy figure, because
  `ClassificationAggregates` has no field to hold one;
* there is no parameter through which a caller could choose the analysis unit — it is resolved from
  `PackRef.analysisUnit`, which §3.3 fixes by rule as `pairingKey[0]`.

That last one is the whole of gate finding N-1. `PairedOutcomes.from_units` raising on a repeated
unit id is a **backstop**: it fires only if the id handed to it is the *cluster* key, and 48
conversation ids drawn from 12 scripts are all unique. What closes it is `_unit_ids` below, which
`_paired_rows` calls for every row it builds — the resolution has exactly one home and no
parameter reaches it.
"""

from __future__ import annotations

import functools
from typing import Mapping, NamedTuple, Sequence

from modelbench import stats
from modelbench.packs import PackConfigError, PackRef, check_sampling_contract
from modelbench.results import (
    BinaryMetric,
    ContinuousMetric,
    DistributionSummary,
    IncompleteItemRecord,
    InvalidRecord,
    ItemResult,
    MetricKindError,
    RunResult,
    ToolCallAggregates,
)
from modelbench.roles import unit_kind as unit_kind_for_role

_DESCRIPTIVE_NOTE = (
    "_Per-arm intervals are Wilson score intervals over the arm's own items: **descriptive, not "
    "the comparison instrument**. The comparison is the paired difference below (`-ml` §3.2)._"
)

_OVERLAP_FOOTNOTE = (
    "_The marginal-overlap line is a **diagnostic**, never the verdict: at this lab's sample sizes "
    "two marginal intervals overlapping is a far stronger condition than their difference covering "
    "zero, and the literal rule cannot fire at all at n<=40 with a baseline >=0.90 (`-ml` §3.1)._"
)

_POOLED_FOOTNOTE = (
    "_A count whose denominator is not the analysis unit prints **without an interval**: `-ml` "
    "§4.4's first mandatory consequence is *\"Never print a Wilson interval over a turn-pooled "
    "count\"*, because the turns of one conversation are not independent observations and the "
    "resulting interval is understated several-fold. The honest bound is a one-level cluster "
    "bootstrap over the conversations (`stats.cluster_bootstrap`, Rule 6), which needs the "
    "per-unit observations a stored aggregate does not carry — S2's runner does._"
)

#: `-ml` §7.2's fourth sentence says "the 12 **scripts**" for a conversation-unit pack: the sample
#: is the set of scripts, one conversation each. Everything else names its own unit.
_SAMPLE_NOUN = {"conversation": "scripts", "item": "items", "query": "queries"}

#: How much a basis is worth, weakest first. A comparison takes the weaker of its two arms, and the
#: degradation is one-directional: any arm whose determinism probe did not run and agree drops the
#: whole comparison, never the other way round (plan §5 test 12b, `-ml` §3.4 Rule 4).
_BASIS_STRENGTH = {"assumed": 0, "measured": 1, "by-construction": 2}

#: The resample count `continuous_verdict()` is called with — this loop's one home for the number
#: rather than a literal at the call site, matching `stats.cluster_bootstrap`'s own default and
#: the `-ml` note's own rendered examples (`B=10000`).
_BOOTSTRAP_B = 10_000


def _pp(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}"


def _sentence(clause: str) -> str:
    """A clause promoted to its own sentence. Not `str.capitalize`, which lower-cases the rest —
    it would print "at any holm step" in the one line whose job is to be quotable."""
    return clause[0].upper() + clause[1:] + "."


def _fp(run: RunResult, name: str, default: str = "") -> str:
    return str(run.fingerprint.get(name, default))


def _unit_ids(items: Sequence[ItemResult], pack: PackRef) -> list[str]:
    """Resolve each row's analysis-unit id from the pack. **No call site chooses this** (§3.3)."""
    index = pack.analysisUnitIndex
    return [item.pairingKey[index] for item in items]


def _metric_aggregate(
    run: RunResult, name: str
) -> BinaryMetric | ContinuousMetric | DistributionSummary | None:
    """`name`'s own aggregate on `run`, or `None` when this arm declares none for it (§3.3 (iv))."""
    return next((m for m in run.aggregates.named_metrics() if m.name == name), None)


class DuplicateModelInReport(ValueError):
    """A `modelKey` appears more than once in `rank_report`'s `runs` (plan §3.1).

    Mirrors `PairedOutcomes.from_units`'s duplicate-unit-id guard: a backstop, not the mechanism.
    Deduplication is the caller's job (Unit B's `_select_rank_arms`, which keeps the newest-stored
    run per `modelKey` — the same one-liner `_select_arms`'s `--models` path already uses).
    """


#: Verdict metrics whose raw rate is a rate of an UNDESIRED event (plan §2.4) — declared explicitly
#: because no manifest field carries polarity. Scoped to `rank_report`'s ranking/family code only;
#: it does not change `stats.verdict()`'s own text, which stays polarity-blind (plan §2.4's
#: flagged, unfixed defect in already-shipped, heavily-tested machinery). Confirmed complete
#: against every pack.json shipped today (`data-scientist`'s review §4). **If
#: `falseAdvanceRateBoundary` (currently exploratory-only, `classification.py::_METRIC_BY_TIER`) is
#: ever promoted into a pack's `verdictMetrics`, this set must gain it in the same change** — same
#: undesired-event polarity, confirmed by the same review.
_LOWER_IS_BETTER: frozenset[str] = frozenset({"falseAdvanceRate", "falseSuspendRate"})


def _metric_value(run: RunResult, metric: str) -> float | None:
    """The one sortable number for `metric` on `run`'s own aggregate (plan §3.1.1):
    `BinaryMetric.rate` or `ContinuousMetric.mean`. `None` when `run` declares no aggregate for
    `metric` at all — excluded from ranking, never ranked last: a model with no data for this pack
    is absent, not "worst". Also `None` when `run` **does** declare an aggregate but its own `n`
    is honestly zero — a mean or rate over zero observations is not a number this module can sort,
    and `_zero_n_arms` is what tells that case apart from "no aggregate at all" so it can be named
    rather than silently conflated with it (catalog-sweep Defect 1).

    Raises `PackConfigError` for a `DistributionSummary` — no shipped pack's headline/
    `verdictMetrics` member resolves to one today, and silently picking median over p10 would be a
    guess this module refuses elsewhere.
    """
    agg = _metric_aggregate(run, metric)
    if agg is None:
        return None
    if isinstance(agg, DistributionSummary):
        raise PackConfigError(
            f"{metric!r} resolves to a DistributionSummary on run {run.runId!r}; rank_report has "
            "no sortable single number for a median+p10 aggregate (plan §3.1.1)"
        )
    if isinstance(agg, ContinuousMetric):
        return agg.mean if agg.n else None
    return agg.rate


def _zero_n_arms(runs: Sequence[RunResult], metric: str) -> list[RunResult]:
    """Runs that declare a real, own aggregate for `metric` whose `n` is honestly zero — ran, and
    produced an internally-consistent record (it clears `_aggregate_item_mismatches`, since a
    declared `n=0` agrees with zero items counting as scored), but every item failed to produce a
    scoreable outcome for this metric.

    Distinct from a run that declares **no aggregate at all** for `metric`: `_metric_value` returns
    `None` for both, and both are excluded from `_rank_rows`, but only this case ran and is owed a
    name in the report — the other stays silently absent, which is `_metric_value`'s own correct
    rule for a model that never attempted this pack (catalog-sweep Defect 1's root cause: the two
    were rendered identically, with neither named)."""
    out = []
    for r in runs:
        agg = _metric_aggregate(r, metric)
        if isinstance(agg, (BinaryMetric, ContinuousMetric)) and agg.n == 0:
            out.append(r)
    return out


def _attempted_for_metric(run: RunResult, metric: str) -> int:
    """How many of `run`'s own items belong to `metric`'s item set — the banner's "N item(s)
    attempted" figure (methodology-gate MAJOR, `docs/reviews/small-model-catalog-sweep-impl.md`).

    **Not `len(run.items)`**: that is the run's whole item list across *every* metric its pack
    scores, and a multi-verdict-metric pack (guard-judge: `falseAdvanceRate`/`falseSuspendRate`)
    partitions `items` disjointly between them — `len(run.items)` overcounts by every sibling
    metric's own items (reproduced against real guard-judge run records: each item's `scoreable`
    carries exactly one metric key, never both).

    **Also not the literal `metric in it.scoreable`** the review's own suggested one-liner reads:
    checked against the real, live `nlq-structured-query` parse-failure records this fix exists
    for (`results/runs/nlq-structured-query-stable-code-instruct-3b-*.json`), a fully-failed
    item's `scoreable` is `{}` — the key is *absent*, not `False` — because the extraction scorer
    never reaches the point of declaring scoreability when parsing itself fails. The literal
    one-liner reads that shape as "0 items attempted", regressing the exact scenario Defect 1
    reported (it would have shipped `nlq`'s real defect-1 rows reading "0 item(s) attempted"
    instead of the correct 40).

    The predicate an item satisfies to belong to `metric`'s set: it either **declares** `metric`
    (present in `scoreable`, `True` or `False` — `_guard_judge_arm`'s asymmetric-fixture shape) or
    it **declares nothing at all** (`scoreable == {}` — a single-verdict-metric pack's own
    total-failure shape, where there is no sibling metric to have been declared instead). An item
    that positively declares a *different* metric (guard-judge's sibling-metric items, and its own
    exploratory `falseAdvanceRateBoundary` items) is excluded either way — it was never part of
    this metric's item set to begin with."""
    return sum(1 for it in run.items if metric in it.scoreable or not it.scoreable)


def _rank_zero_n_lines(runs: Sequence[RunResult], metric: str) -> list[str]:
    """The exclusion banner for `_zero_n_arms(runs, metric)` — same shape as `rank_report`'s own
    `INVALID RESULTS EXCLUDED` (AC-2) block, for the same reason: an excluded arm is named, never
    merely absent, so a reader cannot mistake "ran, scored nothing" for "never attempted this
    pack." Scoped to one metric's own section, because the zero-`n` state is per metric — a run can
    carry a real aggregate for a sibling verdict metric while declaring `n=0` on this one."""
    zero_n = _zero_n_arms(runs, metric)
    if not zero_n:
        return []
    lines = [
        f"> **EXCLUDED — n=0 for `{metric}`**",
        ">",
        "> Ran, and the stored record is internally consistent, but the declared aggregate "
        "honestly reports zero scoreable observations for this metric — excluded from the "
        "ranking below, never ranked \"worst\" and never silently absent either:",
    ]
    for r in zero_n:
        attempted = _attempted_for_metric(r, metric)
        lines.append(f"> - `{r.modelKey}` — {attempted} item(s) attempted, n=0 scored")
    lines.append("")
    return lines


def _better(metric: str, value: float, other: float) -> bool:
    """True if `value` ranks at or above `other` on `metric`, honoring `_LOWER_IS_BETTER`."""
    if metric in _LOWER_IS_BETTER:
        return value <= other
    return value >= other


def _metric_kind(a: RunResult, b: RunResult, name: str) -> str:
    """`"binary"` or `"continuous"`, resolved from a family member's own arm aggregate type — never
    guessed from which per-item map it lives in (§3.3 (iv)): a member whose resolved aggregate is a
    `BinaryMetric` is binary, a `ContinuousMetric` or `DistributionSummary` is continuous. DC-10's
    cross-check (`_aggregate_item_mismatches`) has already reconciled an arm's own aggregate against
    its own items for any arm reaching this point, so the aggregate type is a type fact rather than
    a guess.

    Prefers `a`'s declaration, falling back to `b`'s when `a` carries none for this metric; a metric
    neither arm declares an aggregate for resolves `"binary"`, matching this loop's pre-Table-F
    assumption for the one case Table F's own carrier gives no aggregate to ask.
    """
    metric = _metric_aggregate(a, name) or _metric_aggregate(b, name)
    return "continuous" if isinstance(metric, (ContinuousMetric, DistributionSummary)) else "binary"


class PairedRows(NamedTuple):
    """The paired intersection, plus the tally of everything it had to leave out (`-ml` §4.3).

    The paired `n` printed inside a verdict string shrinks honestly, so the statistic is never
    laundered — but a reader cannot see *that* it shrank, or which arm caused it, unless the
    excluded rows are counted and printed too (§4.3 rule 2, and its paired corollary's
    **`asymmetry`** count). This is also the only place a violated `H <= min(script length)` would
    surface at S1.
    """

    unit_ids: list[str]
    a_ok: list[bool]
    b_ok: list[bool]
    considered: int
    only_in_a: int
    only_in_b: int
    asymmetry_a: int
    asymmetry_b: int
    unscoreable_both: int


def _paired_rows(a: RunResult, b: RunResult, metric: str, pack: PackRef) -> PairedRows:
    """Items scored by *both* arms, in A's order (`-ml` §4.3's paired-n intersection).

    An item whose precondition was not met in either arm is dropped from the pair rather than
    counted as a failure — a precondition failure must never be laundered into the numerator. That
    is `-ml` §10's risk R2, rated **high**: a model that collapses early otherwise scores *better*
    on every conditional count downstream.

    **Which state a row is in is `ItemResult.scored_outcome`'s call, and nothing is inferred here**
    (review P3-1). This function once carried two defaults of its own — a missing scoreability
    declaration read as scoreable, a missing count read as a failure — which combined into a
    verdict of `+100.0 pp, p=0.002` against an arm holding no data whatsoever.
    """
    by_key = {item.pairingKey: item for item in b.items}
    a_units = _unit_ids(a.items, pack)
    unit_ids: list[str] = []
    a_ok: list[bool] = []
    b_ok: list[bool] = []
    only_in_a = asymmetry_a = asymmetry_b = unscoreable_both = 0
    for item, unit_id in zip(a.items, a_units):
        other = by_key.get(item.pairingKey)
        if other is None:
            only_in_a += 1
            continue
        a_outcome = item.scored_outcome(metric)
        b_outcome = other.scored_outcome(metric)
        if a_outcome is None or b_outcome is None:
            if b_outcome is None and a_outcome is not None:
                asymmetry_a += 1
            elif a_outcome is None and b_outcome is not None:
                asymmetry_b += 1
            else:
                unscoreable_both += 1
            continue
        unit_ids.append(unit_id)
        a_ok.append(a_outcome)
        b_ok.append(b_outcome)
    a_keys = {item.pairingKey for item in a.items}
    only_in_b = sum(1 for item in b.items if item.pairingKey not in a_keys)
    return PairedRows(
        unit_ids=unit_ids,
        a_ok=a_ok,
        b_ok=b_ok,
        considered=len(a_keys | {item.pairingKey for item in b.items}),
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        asymmetry_a=asymmetry_a,
        asymmetry_b=asymmetry_b,
        unscoreable_both=unscoreable_both,
    )


class PairedDiffs(NamedTuple):
    """The continuous sibling of `PairedRows` (§4 S1e Table F, `-ml` §3.2d): one difference per
    analysis unit rather than one boolean pair per item, plus the same `-ml` §4.3 tally
    `PairedRows` carries. `_pairing_tally` reads only those six shared fields and does not care
    which producer built them.
    """

    unit_ids: list[str]
    diffs: list[float]
    considered: int
    only_in_a: int
    only_in_b: int
    asymmetry_a: int
    asymmetry_b: int
    unscoreable_both: int


def _paired_diffs(a: RunResult, b: RunResult, metric: str, pack: PackRef) -> PairedDiffs:
    """One difference per **analysis unit**, never per observation (`-ml` §3.2d).

    The binary join above is at item-`pairingKey` granularity because a `verdictMetrics` item
    already *is* its unit; this one joins at the analysis-unit id itself — every item in each arm
    is read, and a unit's items are folded into one value by averaging: the identity when a unit is
    one item (the embedder's case, unit ≡ query ≡ item), the mean §3.2d asks for when it is not.

    A unit present in only one arm's items is excluded and counted `only_in_{a,b}`; a unit present
    in both but scoreable in only one is excluded and counted into the `-ml` §4.3 asymmetry tally
    — a silent drop here is that tally's laundering arriving on the continuous path.
    """

    def unit_values(run: RunResult, units: list[str]) -> dict[str, list[float]]:
        acc: dict[str, list[float]] = {}
        for item, unit_id in zip(run.items, units):
            value = item.scored_value(metric)
            if value is not None:
                acc.setdefault(unit_id, []).append(value)
        return acc

    a_units = _unit_ids(a.items, pack)
    b_units = _unit_ids(b.items, pack)
    a_present, b_present = set(a_units), set(b_units)
    a_values = unit_values(a, a_units)
    b_values = unit_values(b, b_units)

    # Deterministic walk order: a `set`'s own iteration is hash-randomized per process, and
    # `diffs`' order is what the seeded bootstrap resamples by index (`rng.choice`) — a dict's
    # insertion order is not affected by that randomization.
    order: dict[str, None] = {}
    for unit_id in a_units:
        order.setdefault(unit_id, None)
    for unit_id in b_units:
        order.setdefault(unit_id, None)

    unit_ids: list[str] = []
    diffs: list[float] = []
    only_in_a = only_in_b = asymmetry_a = asymmetry_b = unscoreable_both = 0
    for unit_id in order:
        in_a, in_b = unit_id in a_present, unit_id in b_present
        if in_a and not in_b:
            only_in_a += 1
            continue
        if in_b and not in_a:
            only_in_b += 1
            continue
        a_vals, b_vals = a_values.get(unit_id), b_values.get(unit_id)
        if a_vals is None or b_vals is None:
            if a_vals is not None:
                asymmetry_a += 1
            elif b_vals is not None:
                asymmetry_b += 1
            else:
                unscoreable_both += 1
            continue
        unit_ids.append(unit_id)
        diffs.append(sum(a_vals) / len(a_vals) - sum(b_vals) / len(b_vals))

    return PairedDiffs(
        unit_ids=unit_ids,
        diffs=diffs,
        considered=len(order),
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        asymmetry_a=asymmetry_a,
        asymmetry_b=asymmetry_b,
        unscoreable_both=unscoreable_both,
    )


class AggregateMismatch(NamedTuple):
    """One verdict-family rate an arm declares that its own items do not support.

    `detail` is the diagnosis, already phrased for the excluded block: either the two denominators
    that disagree, or the item whose own record could not be read at all (plan-gate G3-7).
    """

    metric: str
    detail: str


def _aggregate_item_mismatches(run: RunResult, pack: PackRef) -> list[AggregateMismatch]:
    """Plan v1.8 §4 S1 done-condition 10 — the `aggregates`-versus-`items` cross-check (P4-4).

    For each `BinaryMetric` in the pack's pre-registered `verdictMetrics` family whose denominator
    is the analysis unit, `metric.n` must equal the number of the arm's items for which
    `scored_outcome` yields an outcome.

    **The selector is `metric.unit == unit_kind_for_role(pack.role)`, not
    `metric.unit == pack.analysisUnit`** (plan-gate G3-6, which found DC-10's prose naming the
    latter). Those are two disjoint vocabularies: `BinaryMetric.unit` is a *denominator noun*
    (`item` / `conversation` / `query` / `turn` / `call`) while `PackRef.analysisUnit` is a
    *`pairingKey` component name*, fixed by `packs.py` to `pairingKey[0]` — `itemId` where the
    unit noun is `item`. The literal predicate is never true, so a check written from it selects
    nothing and passes everything, invisibly. This is the predicate `report.py` already uses one
    function over, for `-ml` §4.4's Wilson-interval suppression.

    The reproduction is an arm declaring
    `BinaryMetric(m, successes=0, n=10)` for a metric **no item declares scoreable**, which
    rendered `0/10 = 0.000` — a claim that ten items were scored — in the same document as
    *"No verdict: no paired data"*.

    **Scoreability, never the score.** An arm that declares ten items scoreable and gets none of
    them right is a consistent record and `0/10` is then a real measurement; a check written
    against `successes` would throw away exactly the arm the tool exists to report on.

    Scoped to the verdict family, because that is the set the comparison actually reads: an
    exploratory metric carries no significance claim, so a disagreement there is not grounds for
    discarding an arm whose pre-registered metrics are sound. The pooled counts are out for the
    same reason `-ml` §4.4 suppresses their intervals — their denominator is not the analysis unit,
    so there is no per-item count to compare them against.

    The check is **S1's whole**: `RunResult` carries `items` and `aggregates` as required fields
    side by side, so it needs nothing S2 produces. S2 owes only the *contract* — that a scorer
    derives its aggregates from the same items in one pass — which is what makes this failure
    unreachable rather than merely reported (plan v1.8 §4 S2).

    **A third arithmetic, for a continuous member** (§4 S1e Table F, plan-gate P6-1). A
    `ContinuousMetric` or `DistributionSummary` widens the same check: `metric.n` must equal the
    number of the arm's items for which `scored_value(metric.name) is not None`, with no unit
    filter — neither carries a denominator noun to compare. **The same check is where a kind
    disagreement surfaces**: a member whose aggregate is continuous while its per-item values live
    in `counts`, or the reverse, is a mismatch of exactly this class, because the arm's aggregate
    path and its per-item path disagree about what was measured. Excluded and named, never
    reconciled — `scored_value` raises `IncompleteItemRecord` on the sibling malformation for the
    same reason `scored_outcome` does, and a `MetricKindError` from either reader is caught here
    too, so neither escapes as a traceback.
    """
    unit = unit_kind_for_role(pack.role)
    found: list[AggregateMismatch] = []
    for metric in run.aggregates.named_metrics():
        continuous = isinstance(metric, (ContinuousMetric, DistributionSummary))
        if not continuous and (not isinstance(metric, BinaryMetric) or metric.unit != unit):
            continue
        if metric.name not in pack.metrics.verdictMetrics:
            continue
        counted = 0
        unreadable: str | None = None
        for it in run.items:
            try:
                seen = (
                    it.scored_value(metric.name) is not None
                    if continuous
                    else it.scored_outcome(metric.name) is not None
                )
                if seen:
                    counted += 1
            except (IncompleteItemRecord, MetricKindError):
                # **A mismatch, never a raise** (plan-gate G3-7). `scored_outcome`/`scored_value`
                # refuse the sibling malformation — a metric declared scoreable with no entry in
                # the map its instrument owns, or an instrument disagreement between the aggregate
                # and the item — and that refusal is right at its own seam, but letting it out of
                # *this* function turns an inconsistent record into a traceback at exit 1, outside
                # §3.6a's closed exit-code set: the exact response this check exists to avoid,
                # arriving through the check's own implementation. The arm is excluded and named,
                # like every other disagreement between an arm's two paths. (`load_history` also
                # quarantines such a record on read — review P4-5 — so the two nets sit at
                # different seams and this one is what makes `compare_report` total rather than
                # trusting its caller.)
                what = "measure" if continuous else "count"
                unreadable = f"item {it.itemId!r} declares it scoreable and records no {what}"
                break
        if unreadable is not None:
            found.append(AggregateMismatch(metric.name, unreadable))
        elif counted != metric.n:
            found.append(
                AggregateMismatch(metric.name, f"declared n={metric.n}, counted {counted}")
            )
    return found


def _pairing_tally(
    rows: PairedRows | PairedDiffs, unit_plural: str, a_label: str, b_label: str
) -> str:
    """§4.3 rule 2's `n/a` tally, printed beside the rate it shaped — always, including when it is
    all zeros, because otherwise a reader cannot tell a shrunken `n` from a full one."""
    return (
        f"- paired n: {len(rows.unit_ids)} of {rows.considered} {unit_plural} "
        f"(`asymmetry`: {rows.asymmetry_a} scoreable for {a_label} only, "
        f"{rows.asymmetry_b} scoreable for {b_label} only; "
        f"{rows.unscoreable_both} unscoreable in both; "
        f"{rows.only_in_a} present in {a_label} only, {rows.only_in_b} in {b_label} only) — §4.3"
    )


def resolving_power_line(rp: stats.ResolvingPower, pack: PackRef) -> str:
    """`-ml` §7.1's template, rendered. Four sentences, all mandatory.

    None of them is derivable from a bare `n` — which is the whole of gate B-1's fix: 48 turns, 48
    conversations and 12 scripts would otherwise print the same sentence and mean three different
    things. The exact string this produces for the tool-caller pack is `-ml` §7.2's, and the suite
    asserts against it.
    """
    unit_plural = f"{rp.unit_kind}s"
    sample_noun = _SAMPLE_NOUN.get(rp.unit_kind, unit_plural)
    if rp.mdd80 is None:
        # Below b_min(alpha_mdd) effective units no effect size attains the power, so the MDD
        # sentence would print a figure the instrument cannot deliver (M-ML-1). The floor sentence
        # is printed either way: it takes the other alpha and can still be attainable when the MDD
        # is not, and where it is not, `floor_clause` says so without quoting a >100 pp threshold.
        sentences = [
            stats.unattainable_clause(rp, unit_plural),
            _sentence(stats.floor_clause(rp)),
        ]
    else:
        sentences = [
            # The stem has one home, in `stats`, because M-ML-7's fix edits it and a second copy
            # here is a scheduled drift (review m-ML-8) — as `provenance`, `floor_clause` and
            # `unattainable_clause` already are.
            f"{stats.mdd_clause(rp, unit_plural)}.",
            _sentence(stats.floor_clause(rp)),
        ]
    # The power model is strict dominance, which is the most favourable case, so the figure is a
    # lower bound and must carry its label. Below n_eff = 20 the 2:1 discordance mix reaches 80%
    # power at NO effect size, and there the label stops being a caveat and starts being the
    # finding (`-ml` §7.1).
    if rp.mdd80 is None:
        # The label qualifies an MDD figure that is not printed; the replacement sentence above
        # already says power is zero at every difference.
        best_case = None
    else:
        best_case = (
            f"Best case — assumes the candidate wins every {rp.unit_kind} the models differ on"
        )
    if best_case is not None and rp.n_effective < 20:
        best_case += (
            f"; if it loses one for every two it wins, {rp.power:.0%} power is not reached at any "
            "effect size at this n."
        )
    elif best_case is not None:
        best_case += "."
    if best_case is not None:
        sentences.append(best_case)
    # `-ml` §4.5.1(ii) publishes this clause for the tool-caller pack, where the sample is a set of
    # written scripts. The claim is right for every unit kind; the noun is not, and `_SAMPLE_NOUN`
    # was already two lines up (n-4, m-ML-5).
    sentences.append(
        f"Inference is conditional on the {rp.n_units} {sample_noun} in {pack.label}; "
        f"generalization to unwritten {sample_noun} is not certified by any interval in this "
        "report."
    )
    return " ".join(sentences)


#: What replaces a verdict when the paired intersection is empty (review P3-1). The two ways in
#: are a scorer that emitted no data for the metric and an arm that could not score a single item;
#: neither is an outcome, and the tally printed under it says which one happened.
_NO_PAIRED_DATA = (
    "**No verdict: no paired data.** No {unit} is scoreable for `{metric}` in both arms, so there "
    "is no paired table, no interval and no verdict. An arm carrying no data for a metric is not "
    "an arm that failed every {unit} of it (`-ml` §4.3)."
)

#: Only the metric's own section has a tally under it; the headline repeats the refusal without
#: pointing at a table that is not beside it.
_NO_PAIRED_DATA_TALLY = " The tally below says where the rows went."


def _decided_by_line(v: stats.Verdict) -> str:
    """The `- decided by:` bullet (`-ml` v1.11 §3.4 Rule 4, §4 S1e Tables D and H).

    On the envelope path it names **which arm bound each bound**, with the exact bootstrap arm
    carrying the level it was taken at. That audit is what the retired seed parenthetical's place
    is owed: naming one arm of a two-arm interval, as the retired token did, is M-ML-8's error
    one layer over, and it was wrong on the 12.6-16.5% of tables where MOVER-D binds both bounds.

    A third token, `"support bound"`, names a bound the `√DEFF` widening pushed past the
    parameter space — no arm produced it, so it carries no `p=` clause, and it renders instead
    with the support's own boundary value: `support bound (-1)` on the lower bound, `(1)` on the
    upper (`-ml` §3.4 Rule 4a).
    """
    if v.bound_by is None:
        return f"- decided by: {v.decided_by}"
    bounds = []
    for arm, level, boundary in zip(
        v.bound_by,
        (stats.LEVEL_CI95_LO, stats.LEVEL_CI95_HI),
        stats.SUPPORT_DIFF_PROPORTIONS,
        strict=True,
    ):
        if arm == "MOVER-D":
            bounds.append(arm)
        elif arm == "support bound":
            bounds.append(f"{arm} ({boundary:g})")
        else:
            bounds.append(f"{arm}, p={float(level):g}")
    lower, upper = bounds
    return f"- decided by: conservative envelope (lower bound: {lower}; upper bound: {upper})"


def _decision(v: stats.Verdict | None, step: stats.HolmStep) -> str:
    """What the family table says happened, so no reader has to re-derive it from a threshold."""
    if v is None:
        return "no verdict — no paired data"
    if not step.tested:
        return "not tested (Holm stops here)"
    if v.distinguishable:
        return "distinguishable"
    if v.floor_demoted:
        return "not distinguishable — below the observable floor"
    return "not distinguishable"


def _comparison_kind(a: RunResult, b: RunResult) -> str:
    """§3.7 — the report says which kind of comparison it is doing, never silently mixing them."""
    # Two labels, not one: the banner above says "same declared version, different bytes", and a
    # single "different pack version" line contradicts it in the same report (review m-2).
    if _fp(a, "packVersion") != _fp(b, "packVersion"):
        return "unpaired (different pack version)"
    if _fp(a, "packContentHash") != _fp(b, "packContentHash"):
        return "unpaired (same pack version, different content hash)"
    if a.sessionId is not None and a.sessionId == b.sessionId:
        return "paired, same session"
    return "paired, cross-session"


def _arm_names(runs: Sequence[RunResult]) -> dict[str, str]:
    """A short, **distinguishing** name per arm, keyed by `runId` (review P4-10).

    Plan §5 test 19a — *"the highest-value single test in the harness"* — is two independent runs
    of one model, and on that comparison the report was unreadable: two identical `arm` cells, a
    §4.3 tally reading *"0 scoreable for qwen/qwen3-4b-2507 only, 0 scoreable for
    qwen/qwen3-4b-2507 only"*, and a significant verdict that would have said *"X is better than
    X"*. Neither `runId` nor `sessionId` was printed anywhere.

    The suffix is added **only where the model key is ambiguous**, so an ordinary comparison keeps
    the bare key and grows no noise. Within an ambiguous group it must actually distinguish:
    `sessionId` is preferred because §5 test 19a's two runs are two *sessions*, but two runs of one
    model inside one session share it, and printing it twice would leave the arms exactly as
    identical as the bare key did — so that case falls back to `runId`, which is unique by
    construction (it is the record's own filename, `results.store`).
    """
    names: dict[str, str] = {}
    for run in runs:
        # Ambiguity is counted over **distinct records**, not over arm slots. `--negative-control`
        # puts the same stored record in both arms deliberately, and there a suffix would
        # disambiguate nothing while implying there were two records to tell apart — the banner
        # above already says there is one.
        group = [r for r in runs if r.modelKey == run.modelKey]
        if len({r.runId for r in group}) == 1:
            names[run.runId] = run.modelKey
            continue
        sessions = [r.sessionId for r in group]
        by_session = run.sessionId is not None and sessions.count(run.sessionId) == 1
        marker = f"session {run.sessionId}" if by_session else f"run {run.runId}"
        names[run.runId] = f"{run.modelKey} ({marker})"
    return names


def _arm_label(run: RunResult, name: str) -> str:
    if run.armKind == "deterministic":
        return f"{name} — reference arm (deterministic given pack version)"
    return name


#: Why no verdict is computed, keyed by cause. One explanation for two causes let a one-arm
#: comparison assert a deterministic-arm reason that is untrue (review M-6).
_NO_VERDICT_REASON = {
    "too-few-arms": (
        "_None: fewer than two arms were selected, so there is nothing to compare. Check "
        "`--models` and `--session` against `model-bench models --tested`; a comparison needs two "
        "stored runs for this pack._"
    ),
    "both-deterministic": (
        "_None: no verdict is computed between two deterministic arms — a deterministic arm is "
        "reproducible from its pack version and arm parameters, so a difference between two of "
        "them is a pack change, not a finding (§3.4.1)._"
    ),
}


def _excluded_reason(count: int) -> str:
    """Why there is nothing to compare when the cross-check took the arms (S1 done-condition 10)."""
    plural = "arm was" if count == 1 else "arms were"
    return (
        f"_None: fewer than two arms remain — {count} {plural} excluded above because their "
        "stored aggregates disagree with their own items, so what is left cannot be compared. "
        "That is a defect in how those records were written, not a scoring outcome: the arm's "
        "per-item and aggregate paths report different denominators for a pre-registered verdict "
        "metric, and neither can be trusted while they disagree (S1 done-condition 10)._"
    )


def _comparison_pair(
    runs: Sequence[RunResult],
) -> tuple[RunResult, RunResult] | str:
    """Two arms to compare, or the **reason** there are none (§3.4.1)."""
    if len(runs) < 2:
        return "too-few-arms"
    a, b = runs[0], runs[1]
    if a.armKind == "deterministic" and b.armKind == "deterministic":
        return "both-deterministic"
    return a, b


#: The `--negative-control` mode's banner (review P3-4). The mode puts **two copies of one stored
#: record** in the two arms, so `b = c = 0` is arithmetic, not a measurement, and the report it
#: writes is durable and filed next to real comparisons under a filename that differs only in its
#: sequence number. Without this it reads as a validated null — the one output a tool whose value
#: claim is *"it refuses to report a number it cannot stand behind"* cannot afford (`-ml` §9,
#: plan §3.9(5)). It is the first thing in the document because it changes how everything below
#: it is read.
_NEGATIVE_CONTROL_BANNER = (
    "> **NEGATIVE CONTROL (WIRING SMOKE CHECK)** — both arms are the *same stored record*, so "
    "`b = c = 0 by construction` and this comparison **cannot fail**. It proves the mode is "
    "wired; it says nothing about whether the harness is sound. The real negative control is two "
    "**independent** runs of the same model and is an acceptance step, not this (`-ml` §9, "
    "plan §5 test 19a)."
)

#: What the mode says when it had nothing to duplicate (review P4-1). The banner above describes
#: *"the same stored record"*, and with no record selected there is no such subject: emitting it
#: anyway produced a durable report opening with **cannot fail** and stating ten lines below, in
#: the same document, that fewer than two arms were selected — P3-4's own failure mode (an artifact
#: asserting something untrue of itself) re-entered through the case P3-4's fix did not cover.
#:
#: The banner is **replaced, never merely suppressed.** This document is filed beside real
#: comparisons under a filename that differs only in its sequence number, so a reader who cannot
#: see that the mode was requested and did not run would read it as an ordinary empty comparison
#: and conclude nothing about the wiring — when what actually happened is that the check the
#: operator asked for never executed.
_NEGATIVE_CONTROL_UNAVAILABLE = (
    "> **NEGATIVE CONTROL REQUESTED, NOT RUN** — the mode puts two copies of *one stored record* "
    "in the two arms, and fewer than two arms reached this comparison, so there was nothing to "
    "duplicate and no wiring was exercised. This document is **not** a negative control and says "
    "nothing about whether the mode works; the reason no arms reached it is below (`-ml` §9, "
    "plan §3.9(5))."
)

#: §3.3 (iv) — printed in place of a verdict for every member of a family whose resolved kinds
#: mixed: no member of a mixed family is verdicted, whichever kind it resolved to, and nothing is
#: excluded — a mixed family is a pack-authoring defect with a one-line fix, never a fault in the
#: arms' own numbers.
_MIXED_FAMILY_MEMBER = (
    "**{kind} metric — no verdict.** This pack's pre-registered verdict-metric family mixes "
    "binary and continuous members, so Holm has no p-value ordering to rank it by and the "
    "correction cannot be taken honestly for a subset of it. No member of the family is "
    "verdicted; every one of the family's numbers prints as exploratory instead (§3.3 (iv))."
)

#: The `### Family-wise error control` block's replacement line for a refused family (decision
#: (3), §3.3 (iv), plan-gate P8-2) — every column that block would otherwise print is a ladder
#: artefact, so nothing in it is true when no ladder ran.
_MIXED_FAMILY_CORRECTION = (
    "This family mixes binary and continuous verdict metrics (named above), so no Holm ladder "
    "ran and no correction was applied to any member — the whole family's verdicts are refused "
    "rather than corrected at a weaker, ad-hoc threshold (§3.3 (iv))."
)

#: The same block's replacement line for an all-continuous family with `k > 1` — the other
#: condition decision (3) names, taking its correction in each metric's own interval rather than
#: in a ladder (`-ml` §3.3, §3.4 Rule 8).
_CONTINUOUS_FAMILY_CORRECTION = (
    "All {k} pre-registered verdict metrics are continuous, so Holm has no p-value ordering to "
    "rank them by; the family-wise correction is taken in each metric's own interval instead of "
    "in a ladder, at the family-adjusted levels `-ml` §3.4 Rule 8 derives from `alpha_family` and "
    "`k` (§3.3)."
)

#: Continuous sibling of `_NO_PAIRED_DATA`, for the one case that message would state falsely:
#: exactly one paired unit exists, so paired data is not absent, but `continuous_verdict` refuses
#: a one-unit interval as a point masquerading as a measurement (`-ml` §3.4 Rule 8, refusal 4).
_ONE_PAIRED_UNIT = (
    "**No verdict: one paired {unit}.** Exactly one {unit} is scoreable for `{metric}` in both "
    "arms, and a one-unit interval is a point: it would report a CI of zero width as though it "
    "were a measurement, so no verdict is computed (`-ml` §3.4 Rule 8)."
)


#: `-ml` §4.4 item 2's own literal text — printed verbatim beside any position whose OBSERVED
#: risk-set size is below 10, never derived or paraphrased at each call site.
_LOW_N_CAVEAT = "descriptive at this n — no significance claim"


def _structural_n(run: RunResult) -> int:
    """The per-turn-position table's "structural" ceiling (§4.4 item 2): how many conversations
    this run scored in total, i.e. what every position's `n` would be had nothing ever been
    censored by a real (model- or tool-channel) failure OR run out of turns early.

    **Computed, not hardcoded — and deliberately the coarser of two honest readings, stated as
    this module's own decision** (S5 spec §4.4 item 2 leaves the exact wiring to the implementer).
    The real pack's own illustration prints a structural n that SHRINKS by position (12/8/4)
    purely from its scripts' own length distribution (shape A/B/C), independent of any run's
    censoring. That per-shape breakdown is not recoverable here: `report.py` is handed a `PackRef`
    (Appendix A's identity-plus-sampling-contract carrier), never a loaded `Pack` with its scripts'
    turn counts, and `RunResult`/`ToolCallAggregates` stores no per-script length either — only the
    POST-censoring `hazard`/`perTurnPosition` tuples, whose own `.censored` field conflates a
    genuine failure with a script simply ending, by design (`hazard_points`'s own docstring).
    `len(run.items)` — one `ItemResult` per scored conversation, unconditionally (§2.6) — is the
    one ceiling always available and always correct: it can only ever OVER-state the true
    per-position structural n (never under), so the `_LOW_N_CAVEAT` this table exists to attach
    still fires whenever the observed n genuinely warrants it.
    """
    return len(run.items)


def _render_funnel(run: RunResult, arm_label: str) -> list[str]:
    """`-ml` §4.3 rule 3's illustrated funnel table (plan `ml.md:2098-2111`), one per arm, plus
    §4.4 item 4's `I(t)`/`Y_calls/Y` distinctness sentence appended to the same block (this
    module's own wiring call, §4.4 item 4's own text — no `LatencyBlock` section exists in this
    file to piggyback on at all: `run.latency` is always `None` for a `tool-caller` run, since
    every conversation-level `ItemResult.timing` is `None` by construction, §2.6). `[]` when this
    run carries no `funnelCounts` at all (every metric-only fixture predating S5 Step 6, and any
    non-`tool-caller` run) — never a table of zeros.

    The restraint line's own "-> k/n" annotation is computed here, at render time, by
    cross-referencing `ToolCallAggregates.restraint` — never a second, derivable copy stored on
    `FunnelCounts` (§7 rule 4).

    2026-09-14 correction (review `small-model-benchmarking-s5.md` Finding 3, option (a)): two
    more lines, `-ml` §4.2(d)'s per-argument failure decomposition
    (`argsOmittedRequired`/`argsWrongValue`, with `argsBoundaryUnit` folded into the second line's
    own "-> of which boundary/unit: N" annotation), nested under "dispatched calls" and denominated
    by cross-referencing `ToolCallAggregates.funnel`'s own `allArgsCorrect` entry — same discipline
    as the restraint annotation above, never a second stored copy.
    """
    fc = run.aggregates.funnelCounts if isinstance(run.aggregates, ToolCallAggregates) else None
    if fc is None:
        return []
    restraint = run.aggregates.restraint
    restraint_note = (
        f"   -> restraint rate {restraint.successes}/{restraint.n}" if restraint is not None else ""
    )
    # 2026-09-14 correction (review Finding 3, option (a)): the denominator for the two new
    # decomposition lines is `allArgsCorrect`'s own `n` — never a second, derivable copy stored on
    # `FunnelCounts` itself (§7 rule 4).
    args_correct_n = next((m.n for m in run.aggregates.funnel if m.name == "allArgsCorrect"), 0)
    lines = [
        f"### Funnel — {arm_label}",
        "",
        "```",
        f"turns driven                    {fc.turnsDriven}",
        f"  unrunnable (model channel)    {fc.unrunnableModelChannel}   "
        "-> no-response / server-rejected",
        f"    turns scored after unrunnable  {fc.turnsScoredAfterUnrunnable}",
        f"  unrunnable (tool channel)     {fc.unrunnableToolChannel}   "
        "-> dispatch raised; conversation censored at t",
        f"  R(t) = 0 (restraint turns)    {fc.restraintTurns}{restraint_note}",
        f"  R(t) >= 1                     {fc.requiredCallTurns}",
        f"    native call emitted         {fc.nativeCallEmitted}   "
        f"-> (a)+(b) partition over {fc.requiredCallTurns}",
        f"    prose pseudo-call           {fc.prosePseudoCall}",
        f"    no attempt                  {fc.noAttempt}",
        f"  turns with >=1 call           {fc.turnsWithAnyCall}   -> (c), (e), (f) denominators",
        f"  dispatched calls              {fc.dispatchedCalls}   "
        "-> (d) denominator (calls, not turns)",
        f"    args omitted required       {fc.argsOmittedRequired}   "
        f"-> per-argument split of (d)'s {args_correct_n} calls w/ correct tool",
        f"    args wrong value            {fc.argsWrongValue}   "
        f"-> of which boundary/unit: {fc.argsBoundaryUnit}",
        f"  fact-bearing returns          {fc.factBearingReturns}   -> (g) denominator",
        f"  unscoreable returns           {fc.unscoreableReturns}",
        "```",
        "",
    ]
    detector = (
        run.aggregates.prosePseudoCallDetector
        if isinstance(run.aggregates, ToolCallAggregates)
        else None
    )
    if detector is not None:
        lines.append(
            f"- prose-pseudo-call detector: precision {detector['precision']:.3f}, "
            f"recall {detector['recall']:.3f} (n={detector['n']} calibration replies)"
        )
    else:
        lines.append(
            "- prose-pseudo-call detector: no calibration corpus declared — "
            "precision/recall unmeasured"
        )
    summary = (
        run.aggregates.iterationSummary if isinstance(run.aggregates, ToolCallAggregates) else None
    )
    if summary is not None and summary.get("n"):
        mean_prefix = ">= " if summary["meanCensored"] else ""
        p95_prefix = ">= " if summary["p95Censored"] else ""
        y = summary["y"]
        y_calls = summary["yCalls"]
        unrestricted_rate = y_calls / y if y else 0.0
        lines += [
            f"- `I(t)` (iterations/turn, replied+cap-hit only, n={summary['n']}): "
            f"mean {mean_prefix}{summary['mean']:.2f}, p95 {p95_prefix}{summary['p95']:.2f}",
            f"- `Y_calls / Y` (unrestricted, every turn driven): "
            f"{y_calls}/{y} = {unrestricted_rate:.2f}",
            "- Different statistics, both printed: `Y_calls / Y` pools every driven turn "
            "including ones that never completed, while `I(t)`'s mean/p95 count only "
            "replied/cap-hit turns (`-ml` §4.2(f)/§11.4) — they will differ whenever a turn did "
            "not complete, and neither substitutes for the other.",
            "",
        ]
    return lines


def _render_per_turn_position(pack: PackRef, runs: Sequence[RunResult]) -> list[str]:
    """`-ml` §4.4's per-position table, one column per arm. `n` is the OBSERVED count
    (`TurnPositionRate.metric.n`, already censoring-aware — `scoring.toolcalls.per_turn_position`'s
    own docstring); the STRUCTURAL n (`_structural_n`, above) prints beside it. Every position
    whose observed n is below 10 is marked `_LOW_N_CAVEAT`, printed verbatim. `[]` when no run in
    `runs` carries any `perTurnPosition` data at all (every non-`tool-caller` report, and every
    `tool-caller` fixture that predates this data existing)."""
    if not any(
        isinstance(r.aggregates, ToolCallAggregates) and r.aggregates.perTurnPosition for r in runs
    ):
        return []
    arm_names = _arm_names(runs)
    max_position = max(
        (
            point.turnIndex
            for r in runs
            if isinstance(r.aggregates, ToolCallAggregates)
            for point in r.aggregates.perTurnPosition
        ),
        default=-1,
    )
    lines = [
        "## Per-turn position",
        "",
        "| position | "
        + " | ".join(f"{arm_names[r.runId]} (observed k/n, structural n)" for r in runs)
        + " |",
        "|---|" + "---|" * len(runs),
    ]
    for t in range(max_position + 1):
        cells = []
        for r in runs:
            points = (
                r.aggregates.perTurnPosition if isinstance(r.aggregates, ToolCallAggregates) else ()
            )
            point = next((p for p in points if p.turnIndex == t), None)
            if point is None:
                cells.append("—")
                continue
            cell = f"{point.metric.successes}/{point.metric.n} (structural {_structural_n(r)})"
            if point.metric.n < 10:
                cell += f" — {_LOW_N_CAVEAT}"
            cells.append(cell)
        lines.append(f"| t={t} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _render_hazard(runs: Sequence[RunResult]) -> list[str]:
    """`-ml` §4.3 rule 5's hazard curve, every arm's own column side by side, each with its own
    `c_t` column reading `HazardPoint.censored` directly — never a recomputed rate. `[]` when no
    run in `runs` carries any `hazard` data at all.

    **Load-bearing prohibition, honoured structurally rather than by discipline**: this function
    computes nothing across arms — every cell is read off exactly one run's own `HazardPoint` at
    exactly one position, so there is no expression anywhere in this function's body that could
    even syntactically become `a_rate - b_rate`. The two curves are conditioned on different,
    arm-specific risk sets after censoring (`-ml` §4.3 rule 5's own closing clause), so a cross-arm
    difference would compare two different populations under one number.
    """
    if not any(isinstance(r.aggregates, ToolCallAggregates) and r.aggregates.hazard for r in runs):
        return []
    arm_names = _arm_names(runs)
    max_position = max(
        (
            point.turnIndex
            for r in runs
            if isinstance(r.aggregates, ToolCallAggregates)
            for point in r.aggregates.hazard
        ),
        default=-1,
    )
    lines = [
        "## Hazard (time-to-first-failure)",
        "",
        "| position | " + " | ".join(f"{arm_names[r.runId]} (f_t, r_t, c_t)" for r in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for t in range(max_position + 1):
        cells = []
        for r in runs:
            points = r.aggregates.hazard if isinstance(r.aggregates, ToolCallAggregates) else ()
            point = next((p for p in points if p.turnIndex == t), None)
            cells.append(
                "—"
                if point is None
                else f"f={point.metric.successes}, r={point.metric.n}, c={point.censored}"
            )
        lines.append(f"| t={t} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _render_role_caveat(pack: PackRef) -> list[str]:
    """Plan §3.8.5/S7 Done-when: `chat-responder`'s deterministic layer never measures reply
    *quality* — only grounding-by-containment, format compliance, and latency. Stated once, in
    words, so a reader does not mistake `groundingRate` for a quality score. `[]` for every
    other role (structural self-gate, `_render_funnel`'s own pattern)."""
    if pack.role != "chat-responder":
        return []
    return [
        "> **Reply quality is not measured by this pack.** `groundingRate` is a deterministic "
        "containment check against the retrieved context, never a judgement of how good, "
        "helpful, or well-written a reply is (FR-21a — the judged-quality layer is deferred, "
        "`docs/BACKLOG.md`).",
        "",
    ]


def _render_speed(runs: Sequence[RunResult], arm_names: Mapping[str, str]) -> list[str]:
    """FR-11's headline block, printed once this component-wide: `RunResult.latency` has
    existed since S2 and this is its first renderer (S7 spec §2.5). `[]` when every run's
    `latency is None` (a `deterministic` arm, or a role that never times — none exist today, but
    the guard costs nothing). Never a new capture: prints exactly `LatencyBlock`'s own thirteen
    fields, nothing FR-11 asks for that this class does not already carry (cold-load time, peak
    RAM — both out of scope, S7 spec §2.5)."""
    present = [r for r in runs if r.latency is not None]
    if not present:
        return []
    lines = [
        "## Speed", "",
        "| arm | p50 | p95/max | timed/n | withheld (load/no-resp) | TTFT median | "
        "prefill ms/1k | tok/s median (diagnostic) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in present:
        lat = r.latency
        p95_or_max = (
            f"{lat.latencyMsP95:.0f}" if lat.latencyMsP95 is not None
            else (f"{lat.latencyMsMax:.0f} (max)" if lat.latencyMsMax is not None else "—")
        )
        ttft = (
            f"{lat.ttftMsMedian:.0f}" if lat.ttftMsMedian is not None
            else "— (insufficient coverage)"
        )
        prefill = f"{lat.prefillMsPer1kMedian:.1f}" if lat.prefillMsPer1kMedian is not None else "—"
        tps = f"{lat.tokensPerSecondMedian:.1f}" if lat.tokensPerSecondMedian is not None else "—"
        p50 = (
            f"{lat.latencyMsP50:.0f}" if lat.latencyMsP50 is not None
            else "— (insufficient coverage)"
        )
        lines.append(
            f"| {_arm_label(r, arm_names[r.runId])} | {p50} | {p95_or_max} | "
            f"{lat.latencyTimedCount}/{lat.latencyItemCount} | "
            f"{lat.latencyWithheldForLoad}/{lat.latencyWithheldForNoResponse} | {ttft} | "
            f"{prefill} | {tps} |"
        )
    lines += [
        "", "*Descriptive only — decode tokens/sec is a diagnostic, never a comparison "
        "instrument (FR-11).*", "",
    ]
    return lines


#: FR-11's restated ceiling/adequacy caveat, per pack (plan §2.3/§3.1, sourced from `-ml`, cited by
#: section — matching this module's own citation discipline). `chat-responder`'s own caveat is
#: `_render_role_caveat`'s existing string, reused verbatim rather than duplicated here.
_RANK_CAVEATS: dict[str, str] = {
    "embedder-graphrag-retrieval": (
        "recall@10 = 37/38 at this pack's own item set: only 1 item is available to win, and "
        "McNemar needs 6 — this ranking can detect a materially worse embedder but cannot certify "
        "a better one (`-ml` §7.4)."
    ),
    "guard-judge-understanding": (
        "Two co-equal class-conditional error rates, no single headline: floor 15.0/20.0 pp, "
        "MDD80 21.9/28.7 pp for falseAdvanceRate/falseSuspendRate at the two-member "
        "alpha_mdd=0.025 (`-ml` §7.3)."
    ),
    "nlq-structured-query": (
        "The true denominator is 34, not 40 — 6 items are structurally unanswerable and excluded "
        "(`-ml` §7.2, v1.25 note)."
    ),
    "tool-caller-shop-assistant": (
        "The analysis unit is scripts, n=12 (3 shapes x 4 scripts): floor 50.0 pp, MDD80 57.8 pp "
        "at this n (`-ml` §7.2/§4.5)."
    ),
}


def _rank_caveat_lines(pack: PackRef) -> list[str]:
    """FR-11's restated caveat, block-quoted above a ranked table (plan §3.1). `chat-responder`'s
    is `_render_role_caveat`'s own string, reused rather than duplicated — "two copies of a string
    is one copy and one drift" cuts here exactly as it does for `stats.py`'s formula strings."""
    if pack.role == "chat-responder":
        return _render_role_caveat(pack)
    caveat = _RANK_CAVEATS.get(pack.packId)
    if caveat is None:
        return []
    return [f"> {caveat}", ""]


#: `mean_bootstrap_interval`'s own descriptive caveat (plan §3.1, `-ml` review §3) — stronger than
#: `_DESCRIPTIVE_NOTE`: a single arm's own mean CI is not itself a basis for a verdict, even by
#: eyeball comparison against another arm's interval. A distinct footnote so the two can't merge.
_MEAN_DESCRIPTIVE_NOTE = (
    "_This interval describes this model's own mean; it is not a comparison, and two such "
    "intervals overlapping or not overlapping is not itself a basis for a verdict — see FR-8's "
    "optional reference-anchored family for an actual test, when one was run._"
)


def _rank_item_values(run: RunResult, metric: str) -> list[float]:
    """A continuous member's own per-item values, for `stats.mean_bootstrap_interval`."""
    return [v for it in run.items for v in [it.scored_value(metric)] if v is not None]


def _rank_latency_cell(run: RunResult) -> str:
    """The ranked table's latency column: `RunResult.latency.latencyMsP95`, `—` when the run
    carries no latency at all or no p95 within it — matching `_render_speed`'s own per-field guard
    (plan §3.1) rather than inventing a new convention."""
    if run.latency is None or run.latency.latencyMsP95 is None:
        return "—"
    return f"{run.latency.latencyMsP95:.0f}"


def _rank_rows(runs: Sequence[RunResult], metric: str) -> list[tuple[RunResult, float]]:
    """Every run with a value for `metric`, sorted best-first via `_better` (plan §3.1.1) — a run
    with no aggregate for `metric` is excluded, never ranked last."""
    rows = [(r, v) for r in runs for v in [_metric_value(r, metric)] if v is not None]

    def _cmp(x: tuple[RunResult, float], y: tuple[RunResult, float]) -> int:
        if x[1] == y[1]:
            return 0
        return -1 if _better(metric, x[1], y[1]) else 1

    rows.sort(key=functools.cmp_to_key(_cmp))
    return rows


def _render_one_rank_table(
    runs: Sequence[RunResult],
    *,
    metric: str,
    pack: PackRef,
    footprints: Mapping[str, str] | None,
) -> list[str]:
    """One ranked table for `metric` (plan §3.1): rank | model | k/n (or n) | rate (or mean) |
    95% CI | latency p95 | footprint, plus the `<!-- rank-report: ... -->` marker Unit C depends
    on (plan §3.5)."""
    lines: list[str] = [f"### {metric}", ""]
    lines += _rank_caveat_lines(pack)
    lines += _rank_zero_n_lines(runs, metric)
    rows = _rank_rows(runs, metric)
    if not rows:
        lines += [
            f"_No in-scope model has a stored, consistent result for `{metric}` in this report._",
            "",
        ]
        return lines

    continuous = isinstance(_metric_aggregate(rows[0][0], metric), ContinuousMetric)
    if continuous:
        lines += [
            "| rank | model | n | mean | 95% CI | latency p95 | footprint |",
            "|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| rank | model | k/n | rate | 95% Wilson | latency p95 | footprint |",
            "|---|---|---|---|---|---|---|",
        ]

    rendered: list[tuple[RunResult, float, float, float]] = []
    for r, value in rows:
        agg = _metric_aggregate(r, metric)
        if isinstance(agg, ContinuousMetric):
            values = _rank_item_values(r, metric)
            lo, hi = stats.mean_bootstrap_interval(
                values, B=_BOOTSTRAP_B, seed=pack.seed,
                levels=(stats.LEVEL_CI95_LO, stats.LEVEL_CI95_HI), support=agg.support,
            )
            count_cell = f"n={agg.n}"
            value_cell, ci_cell = f"{agg.mean:.4f}", f"[{lo:.4f}, {hi:.4f}]"
        else:
            lo, hi = stats.wilson_interval(agg.successes, agg.n)
            count_cell = f"{agg.successes}/{agg.n}"
            value_cell, ci_cell = f"{agg.rate:.3f}", f"[{lo:.3f}, {hi:.3f}]"
        latency_cell = _rank_latency_cell(r)
        footprint_cell = (footprints or {}).get(r.modelKey, "—")
        rendered.append((r, value, lo, hi))
        lines.append(
            f"| {len(rendered)} | {r.modelKey} | {count_cell} | {value_cell} | {ci_cell} | "
            f"{latency_cell} | {footprint_cell} |"
        )

    lines.append("")
    lines += [_MEAN_DESCRIPTIVE_NOTE if continuous else _DESCRIPTIVE_NOTE, ""]

    top_run, top_value, top_lo, top_hi = rendered[0]
    lines.append(
        f"<!-- rank-report: pack={pack.packId} metric={metric} top={top_run.modelKey} "
        f"value={top_value:.4f} ci=[{top_lo:.4f},{top_hi:.4f}] -->"
    )
    lines.append("")
    return lines


def _polarity_corrected(
    metric: str, diff: float, ci: tuple[float, float]
) -> tuple[float, tuple[float, float]]:
    """Flips `diff`/`ci` — both oriented reference-minus-candidate on the raw "ok" rate, exactly
    as `stats.verdict` computes them — so a positive number always reads "candidate is better"
    (plan §3.2.2), honoring `_LOWER_IS_BETTER`'s polarity rather than `stats.verdict`'s own
    polarity-blind `diff >= 0` (plan §2.4). Deliberately does not touch `Verdict.text`.

    For an ordinary higher-is-better metric, `scored_outcome`'s `True` means success, so a `b`-unit
    (reference ok, candidate not) is a reference win: `diff >= 0` means the *reference* is ahead,
    the opposite of what this table wants — negate both. For a `_LOWER_IS_BETTER` metric,
    `scored_outcome`'s `True` means the undesired event occurred, so a `b`-unit (reference "ok" —
    i.e. reference had the bad event, candidate did not) is a *candidate* win: `diff >= 0` already
    means the candidate is ahead, and no flip is needed.
    """
    if metric in _LOWER_IS_BETTER:
        return diff, ci
    return -diff, (-ci[1], -ci[0])


def _render_reference_family(
    *,
    metric: str,
    pack: PackRef,
    reference_run: RunResult,
    candidates: Sequence[RunResult],
    cells: Sequence[tuple[str, str]],
    steps: Sequence[stats.HolmStep],
    outcomes: Mapping[tuple[str, str], stats.PairedOutcomes],
    correction_k: int,
    unit_kind: str,
) -> list[str]:
    """FR-8's optional reference-anchored family (plan §3.2.2), filtered back to `metric`'s own
    rows from the one combined (candidate x metric) Holm ladder `rank_report` already ran over the
    whole family — never a second, per-metric ladder."""
    lines = [
        f"#### Reference-anchored family — {metric} vs `{reference_run.modelKey}`", "",
        "_exploratory — no significance claim outside this family_", "",
        "| candidate | diff | 95% CI | Holm-adjusted threshold | decision |",
        "|---|---|---|---|---|",
    ]
    alpha_mdd = pack.metrics.alpha_family / correction_k
    for candidate in candidates:
        idx = cells.index((candidate.modelKey, metric))
        step = steps[idx]
        outcome = outcomes[(candidate.modelKey, metric)]
        if outcome.n_units == 0:
            lines.append(
                f"| {candidate.modelKey} | — | — | {step.threshold:.4f} | "
                f"{_decision(None, step)} |"
            )
            continue
        rp = stats.resolving_power(
            outcome.n_units, unit_kind=unit_kind,
            design_effect=max(reference_run.designEffect, candidate.designEffect),
            basis=min((reference_run.basis, candidate.basis), key=_BASIS_STRENGTH.__getitem__),
            alpha_family=pack.metrics.alpha_family, alpha_mdd=alpha_mdd,
        )
        v = stats.verdict(
            outcome, resolving=rp, metric_name=metric, family=[metric],
            correction_k=correction_k, a_label=reference_run.modelKey, b_label=candidate.modelKey,
            alpha_step=step.threshold, holm_tested=step.tested,
        )
        diff, ci = _polarity_corrected(metric, v.diff, v.ci)
        lines.append(
            f"| {candidate.modelKey} | {'+' if diff >= 0 else ''}{_pp(diff)} pp | "
            f"[{_pp(ci[0])}, {_pp(ci[1])}] pp | {step.threshold:.4f} | {_decision(v, step)} |"
        )
    lines.append("")
    return lines


def _rank_resolving_power_lines(
    runs: Sequence[RunResult], pack: PackRef, family: Sequence[str], metric: str
) -> list[str]:
    """FR-7's "recomputed for the model count actually in this report" sentence (plan §3.4 Q4),
    printed alongside — never instead of — the pack's own already-published single-comparison
    `resolving_power_line` (unchanged), **once per member of `family`** (code-gate finding, Unit
    A.5, `docs/reviews/small-model-catalog-sweep-impl.md`) — mirroring how `_rank_caveat_lines`/
    `_render_one_rank_table` already loop per ranked table, immediately below the table whose own
    FR-11 caveat already states that metric's own figures.

    **`n_units` is `metric`'s own item count, never pooled with a sibling metric's via `max`/`min`
    across `family`.** Guard-judge's two verdict metrics have different, already-published item
    counts (`-ml` §7.3: 40 for `falseAdvanceRate`, 30 for `falseSuspendRate`) — pooling them (the
    original, defective implementation took `max` across both) silently applied whichever metric
    happened to have the larger, less-constrained n to the other metric's own sentence too, always
    in the optimistic direction (both the floor and the MDD get *worse*, i.e. larger, as n shrinks
    — `-ml` §3.4 Rule 3). Neither `max` nor `min` pooled into one sentence is correct: each metric
    publishes its own row in `-ml` §7.3 for exactly this reason, and a single number in either
    direction misrepresents whichever metric it doesn't match.

    `k` still reflects Q3's resolution — the full **compound** family size (`len(family) * (n-1)`,
    `2*(N-1)` for guard-judge) — since the hypothetical sentence describes what the one joint,
    reference-anchored family (not a per-metric one) would cost; only the `n_units` feeding each
    metric's own two `resolving_power` calls is metric-specific. Names no specific anchor model —
    naming "the top-ranked model" would read as a post-hoc, choose-after-you-see-the-data
    pre-registration (plan §3.4 Q4)."""
    n = len(runs)
    k = len(family) * (n - 1)
    if k <= 0:
        return []
    unit_kind = unit_kind_for_role(pack.role)
    ns = [agg.n for r in runs for agg in [_metric_aggregate(r, metric)] if agg is not None]
    if not ns:
        return []
    n_units = max(ns)
    published = stats.resolving_power(
        n_units, unit_kind=unit_kind, design_effect=1.0, basis="by-construction",
        alpha_family=pack.metrics.alpha_family, alpha_mdd=pack.metrics.alpha_mdd,
    )
    hypothetical = stats.resolving_power(
        n_units, unit_kind=unit_kind, design_effect=1.0, basis="by-construction",
        alpha_family=pack.metrics.alpha_family, alpha_mdd=pack.metrics.alpha_family / k,
    )
    joint = " jointly across both verdict metrics," if len(family) > 1 else ""
    cost = (
        "no difference at any effect size" if hypothetical.mdd80 is None
        else f">={_pp(hypothetical.mdd80)} pp"
    )
    sentence = (
        f"If this pack's optional reference-anchored family (FR-8) were run — any one of the "
        f"{n} models here designated as the reference, the other {n - 1} compared against it"
        f"{joint} — that family of {k} tests would resolve differences of {cost} with 80% power "
        f"(alpha_mdd = {pack.metrics.alpha_family:g}/{k})."
    )
    return [resolving_power_line(published, pack), "", sentence, ""]


def rank_report(
    runs: Sequence[RunResult],
    *,
    pack: PackRef,
    invalid: Sequence[InvalidRecord] = (),
    reference: str | None = None,
    footprints: Mapping[str, str] | None = None,
) -> str:
    """FR-6/FR-7/FR-11/FR-12: one ranked table per pack (two, for a pack with no headline metric —
    one per `verdictMetrics` member), covering every in-scope model with a stored, consistent
    result — never a pairwise matrix. `reference`, when given, additionally renders FR-8's optional
    reference-anchored Holm-Bonferroni family (plan §3.2)."""
    check_sampling_contract(pack)
    # Mirrors `compare_report`'s own guard on the identical field (`analyst` code-gate suggestion,
    # `docs/reviews/small-model-catalog-sweep-impl.md`) — pack-load-time enforcement
    # (`packs.metrics_from_manifest`) only protects the real CLI manifest-loading path; a
    # hand-built `PackRef`/`PackMetrics`, which every test fixture (including this function's own)
    # constructs directly, bypasses it entirely.
    if pack.metrics.headlineMetric is not None and (
        pack.metrics.headlineMetric not in pack.metrics.verdictMetrics
    ):
        raise PackConfigError("headlineMetric is not a member of verdictMetrics")

    seen: set[str] = set()
    for r in runs:
        if r.modelKey in seen:
            raise DuplicateModelInReport(
                f"modelKey {r.modelKey!r} appears more than once in rank_report's runs; "
                "deduplication is the caller's job, not rank_report's (plan §3.1)"
            )
        seen.add(r.modelKey)

    lines: list[str] = [f"# Ranked comparison — {pack.label} ({pack.role})", ""]

    # S1 done-condition 10, reused: an arm whose stored aggregates disagree with its own items is
    # excluded from the ranking, not repaired and not partly trusted — the same net `compare_report`
    # already casts.
    inconsistent = [(r, _aggregate_item_mismatches(r, pack)) for r in runs]
    excluded = [(r, m) for r, m in inconsistent if m]
    runs = [r for r, m in inconsistent if not m]

    versions = {_fp(r, "packVersion") for r in runs}
    hashes = {_fp(r, "packContentHash") for r in runs}
    if len(versions) > 1:
        lines += [
            "> **PACK VERSION MISMATCH** — these runs span pack versions "
            + ", ".join(sorted(versions))
            + ". They are not measuring the same thing; the ranking below is rendered anyway.",
            "",
        ]
    if len(hashes) > 1:
        lines += [
            "> **PACK CONTENT HASH MISMATCH** — same declared version, different bytes: "
            + ", ".join(sorted(h[:8] for h in hashes))
            + ". A declared version can be forgotten; a hash cannot (§3.3).",
            "",
        ]
    schemas = sorted({r.fingerprint.benchSchemaVersion for r in runs})
    if len(schemas) > 1:
        lines += [
            "> **SCHEMA VERSIONS IN THIS COMPARISON** — "
            + ", ".join(str(s) for s in schemas)
            + ". Each record was validated against the contract it was written under; a schema "
            "difference is visible, never silent, and never a reason to drop a record (§3.4.3).",
            "",
        ]
    if invalid or excluded:
        lines += ["> **INVALID RESULTS EXCLUDED** (AC-2)", ">"]
        for record in invalid:
            detail = ", ".join(f"`{p.field}` ({p.reason})" for p in record.problems)
            suffix = f": {detail}" if detail else ""
            lines.append(f"> - `{record.runId or record.path.name}` — {record.reason}{suffix}")
        for run_, mismatches in excluded:
            detail = ", ".join(f"`{m.metric}` ({m.detail})" for m in mismatches)
            lines.append(f"> - `{run_.runId}` — aggregates disagree with items: {detail}")
        lines.append("")

    family = list(pack.metrics.verdictMetrics)
    unit_kind = unit_kind_for_role(pack.role)
    members = [pack.metrics.headlineMetric] if pack.metrics.headlineMetric is not None else family

    reference_run: RunResult | None = None
    if reference is not None:
        reference_run = next((r for r in runs if r.modelKey == reference), None)
        if reference_run is None:
            raise ValueError(
                f"reference model {reference!r} has no stored, consistent run for this pack"
                + (" and session" if runs else "")
            )

    combined_p_values: list[float] = []
    combined_cells: list[tuple[str, str]] = []
    combined_outcomes: dict[tuple[str, str], stats.PairedOutcomes] = {}
    candidates: list[RunResult] = []
    combined_steps: list[stats.HolmStep] = []
    correction_k = 0
    if reference_run is not None:
        candidates = [r for r in runs if r.modelKey != reference_run.modelKey]
        for metric in family:
            for cand in candidates:
                paired = _paired_rows(reference_run, cand, metric, pack)
                outcomes = stats.PairedOutcomes.from_units(
                    unit_kind, list(zip(paired.unit_ids, paired.a_ok, paired.b_ok))
                )
                combined_outcomes[(cand.modelKey, metric)] = outcomes
                _a, b_, c_, _d = outcomes.table
                # `-ml` review Pass2-1: `k` is fixed by pre-registration, never by how much data
                # arrived — a candidate with no paired data for one metric still consumes a Holm
                # rank, via the same `mcnemar_exact(0, 0) = 1.0` empty-intersection handling
                # `compare_report`'s own homogeneous-binary path already relies on (its own
                # unconditional `p_values.append(stats.mcnemar_exact(table_b, table_c))` below),
                # reused rather than reinvented.
                combined_p_values.append(stats.mcnemar_exact(b_, c_))
                combined_cells.append((cand.modelKey, metric))
        # ONE combined call over (metric x candidate), never one call per metric — the plan's
        # originally-rejected per-metric default under-corrects the family-wise error rate (§3.2.2).
        combined_steps = stats.holm_steps(combined_p_values, alpha=pack.metrics.alpha_family)
        correction_k = len(combined_p_values)

    for metric in members:
        lines += _render_one_rank_table(runs, metric=metric, pack=pack, footprints=footprints)
        if len(runs) >= 2:
            lines += _rank_resolving_power_lines(runs, pack, family, metric)
        if reference_run is not None and candidates:
            lines += _render_reference_family(
                metric=metric, pack=pack, reference_run=reference_run, candidates=candidates,
                cells=combined_cells, steps=combined_steps, outcomes=combined_outcomes,
                correction_k=correction_k, unit_kind=unit_kind,
            )

    return "\n".join(lines) + "\n"


def compare_report(
    runs: Sequence[RunResult],
    *,
    pack: PackRef,
    invalid: Sequence[InvalidRecord] = (),
    negative_control: bool = False,
) -> str:
    """Render the markdown comparison for one pack. Never ranks across roles or packs."""
    check_sampling_contract(pack)
    if pack.metrics.headlineMetric is not None and (
        pack.metrics.headlineMetric not in pack.metrics.verdictMetrics
    ):
        raise PackConfigError("headlineMetric is not a member of verdictMetrics")

    lines: list[str] = [f"# Comparison — {pack.label} ({pack.role})", ""]
    lines += _render_role_caveat(pack)

    # S1 done-condition 10 — an arm whose stored aggregates disagree with its own items is
    # **excluded from the comparison, not repaired and not partly trusted**, and named below. The
    # partition happens before any banner or table reads `runs`, because an excluded arm is not in
    # the comparison at all: it must not colour the version/hash/schema banners either. If this
    # leaves fewer than two arms, `_comparison_pair` renders the no-comparison case it already has.
    inconsistent = [(r, _aggregate_item_mismatches(r, pack)) for r in runs]
    excluded = [(r, m) for r, m in inconsistent if m]
    runs = [r for r, m in inconsistent if not m]
    arm_names = _arm_names(runs)

    # The mode's banner is decided **after** the arms are known, not before (review P4-1): it
    # describes "the same stored record", and whether there is such a record is exactly what
    # `_select_arms` and the cross-check above have just settled.
    if negative_control:
        lines += [
            _NEGATIVE_CONTROL_BANNER if len(runs) >= 2 else _NEGATIVE_CONTROL_UNAVAILABLE,
            "",
        ]

    # --- banners: never silent, and never a reason to drop a record -----------------------------
    versions = {_fp(r, "packVersion") for r in runs}
    hashes = {_fp(r, "packContentHash") for r in runs}
    if len(versions) > 1:
        lines += [
            "> **PACK VERSION MISMATCH** — these runs span pack versions "
            + ", ".join(sorted(versions))
            + ". "
            "They are not measuring the same thing; the comparison below is rendered anyway and is "
            "labelled unpaired (AC-3).",
            "",
        ]
    if len(hashes) > 1:
        lines += [
            "> **PACK CONTENT HASH MISMATCH** — same declared version, different bytes: "
            + ", ".join(sorted(h[:8] for h in hashes))
            + ". A declared version can be forgotten; a hash cannot (§3.3).",
            "",
        ]
    schemas = sorted({r.fingerprint.benchSchemaVersion for r in runs})
    if len(schemas) > 1:
        lines += [
            "> **SCHEMA VERSIONS IN THIS COMPARISON** — "
            + ", ".join(str(s) for s in schemas)
            + ". Each record was validated against the contract it was written under; a schema "
            "difference is visible, never silent, and never a reason to drop a record (§3.4.3).",
            "",
        ]
    if invalid or excluded:
        lines += ["> **INVALID RESULTS EXCLUDED** (AC-2)", ">"]
        for record in invalid:
            # An `unparseable` record carries no problems at all — nothing about the file was
            # legible, so there are no fields to name (`results.load_history`). The detail used
            # to fall back to `record.reason`, which is already the first half of this line, so
            # every such record printed as "unparseable: unparseable". The colon introduces the
            # fields that failed; with none to introduce, the reason stands on its own.
            detail = ", ".join(f"`{p.field}` ({p.reason})" for p in record.problems)
            suffix = f": {detail}" if detail else ""
            lines.append(f"> - `{record.runId or record.path.name}` — {record.reason}{suffix}")
        for run, mismatches in excluded:
            # The same block, deliberately: exclude-and-name is AC-2's own mechanism, already built
            # and already read as "this record did not enter the comparison, and here is why". Both
            # counts are printed because the *direction* of the disagreement is the diagnosis a
            # scorer author needs, and neither number alone carries it.
            detail = ", ".join(f"`{m.metric}` ({m.detail})" for m in mismatches)
            lines.append(
                f"> - `{run.runId}` — aggregates disagree with items: {detail}"
            )
        lines.append("")

    # --- the funnel table (`-ml` §4.3 rule 3): opens the report, before any metric table --------
    for r in runs:
        lines += _render_funnel(r, _arm_label(r, arm_names[r.runId]))

    # --- per-arm descriptive table --------------------------------------------------------------
    lines += ["## Arms", "", "| arm | metric | k/n | rate | 95% Wilson |", "|---|---|---|---|---|"]
    pooled_seen = False
    for run in runs:
        for metric in run.aggregates.named_metrics():
            if isinstance(metric, BinaryMetric) and not metric.n:
                # **Rendered, never silently dropped** (review P4-6). `and metric.n` was doing two
                # jobs — suppressing the row *and* keeping `successes / n` away from a zero
                # denominator — and a reader could not tell a dropped row from an arm that never
                # declared the metric. `0/0` with no rate and no interval distinguishes them and
                # is true; the division is simply not reached. Same call `_POOLED_FOOTNOTE` makes
                # one column over: the count is never suppressed, only the precision claim is.
                lines.append(
                    f"| {_arm_label(run, arm_names[run.runId])} | {metric.name} | 0/0 | — | "
                    "— (no observations) |"
                )
            elif isinstance(metric, BinaryMetric):
                if metric.unit == unit_kind_for_role(pack.role):
                    lo, hi = stats.wilson_interval(metric.successes, metric.n)
                    interval = f"[{lo:.3f}, {hi:.3f}]"
                else:
                    # `-ml` §4.4: a turn- or call-pooled count's observations are not independent,
                    # so a Wilson interval over them is fiction — measured at 4.5x too narrow on a
                    # representative funnel count (review M-ML-3). The count itself is never
                    # suppressed; only the precision claim is.
                    pooled_seen = True
                    interval = f"— (n is {metric.unit}s; not the analysis unit)"
                lines.append(
                    f"| {_arm_label(run, arm_names[run.runId])} | {metric.name} | "
                    f"{metric.successes}/{metric.n} | "
                    f"{metric.successes / metric.n:.3f} | {interval} |"
                )
            elif isinstance(metric, DistributionSummary):
                # A `DistributionSummary` renders its median and p10 and no interval — never
                # `.mean`, which it does not carry (§4 S1e Table F, plan-gate P6-1(a) item (f)).
                lines.append(
                    f"| {_arm_label(run, arm_names[run.runId])} | {metric.name} | n={metric.n} | "
                    f"p50 {metric.median:.4f}, p10 {metric.p10:.4f} | — |"
                )
            else:
                lines.append(
                    f"| {_arm_label(run, arm_names[run.runId])} | {metric.name} | n={metric.n} | "
                    f"{metric.mean:.4f} | — |"
                )
    lines += ["", _DESCRIPTIVE_NOTE, ""]
    if pooled_seen:
        lines += [_POOLED_FOOTNOTE, ""]

    # --- speed (S7 spec §2.5/§3.5): FR-11's headline latency block, one more descriptive block
    # before the verdict machinery, after the Arms table and its footnotes -----------------------
    lines += _render_speed(runs, arm_names)

    # --- the per-turn-position table and the hazard curve (`-ml` §4.4/§4.3 rule 5), both after
    # the generic Arms table (§4.4 items 2-3) -----------------------------------------------------
    lines += _render_per_turn_position(pack, runs)
    lines += _render_hazard(runs)

    pair = _comparison_pair(runs)
    if isinstance(pair, str):
        # **Which** too-few-arms this is changes the remedy, and the two are not interchangeable
        # (the same class M-6 split `_NO_VERDICT_REASON` for). Arms removed by the cross-check were
        # *selected* and then thrown away, so pointing the reader at `--models`/`--session` sends a
        # scorer author to their command line when the defect is in their record.
        reason = (
            _excluded_reason(len(excluded))
            if pair == "too-few-arms" and excluded
            else _NO_VERDICT_REASON[pair]
        )
        lines += ["## Verdicts", "", reason, ""]
        return "\n".join(lines) + "\n"

    a, b = pair
    lines += ["## Verdicts", "", f"Comparison kind: **{_comparison_kind(a, b)}** (§3.7).", ""]

    family = list(pack.metrics.verdictMetrics)
    unit_kind = unit_kind_for_role(pack.role)
    # A basis is only as strong as its weakest arm, and the degradation is fail-safe: any arm whose
    # determinism probe did not run and agree drops the whole comparison to "assumed", which via
    # `-ml` §3.4 Rule 4 moves the decision off McNemar (plan §5 test 12b).
    design_effect = max(a.designEffect, b.designEffect)
    # The **weaker of the two actual bases**, not a collapse to `"assumed"`. The decision rule is
    # unchanged either way — only `by-construction` lets McNemar decide — but printing `assumed`
    # for two genuinely *measured* design effects is false provenance in the one sentence whose
    # entire job is auditability (`-ml` §7.1, review m-ML-4).
    basis = min((a.basis, b.basis), key=_BASIS_STRENGTH.__getitem__)

    # §3.3 (iv) — a family member's kind is a type fact, resolved from its own arm aggregate, never
    # guessed from which per-item map it lives in: DC-10's cross-check above has already reconciled
    # the two for any arm reaching this point. A mixed family is refused *whole* — no member is
    # verdicted, nothing is excluded — because `k` is `len(verdictMetrics)` and pre-registered, so
    # dropping the minority kind would shrink `k` after the results exist and under-correct the
    # survivors. An all-continuous family with `k > 1` takes its correction in each metric's own
    # interval rather than in a ladder; a homogeneous binary family is the unchanged two-pass Holm
    # flow below.
    kinds = {metric: _metric_kind(a, b, metric) for metric in family}
    resolved_kinds = set(kinds.values())
    mixed_kinds = len(resolved_kinds) > 1
    continuous_family = resolved_kinds == {"continuous"}

    computed: list[
        tuple[str, stats.Verdict | stats.ContinuousVerdict | None, stats.HolmStep | None]
    ] = []

    if mixed_kinds:
        for metric in family:
            kind = kinds[metric]
            if kind == "binary":
                tally = _pairing_tally(
                    _paired_rows(a, b, metric, pack),
                    f"{unit_kind}s",
                    arm_names[a.runId],
                    arm_names[b.runId],
                )
            else:
                tally = _pairing_tally(
                    _paired_diffs(a, b, metric, pack),
                    f"{unit_kind}s",
                    arm_names[a.runId],
                    arm_names[b.runId],
                )
            computed.append((metric, None, None))
            lines += [
                f"### {metric}",
                "",
                _MIXED_FAMILY_MEMBER.format(kind=kind),
                "",
                tally,
                "",
            ]
    elif continuous_family:
        # §3.2d's continuous branch — one difference per analysis unit, handed to Rule 8's
        # producer. No `holm_steps`, no `mcnemar_exact`, no `resolving_power`: none of the three
        # exists on this path (`-ml` §3.4 Rule 8's four refused parameters).
        for metric in family:
            diffs_row = _paired_diffs(a, b, metric, pack)
            tally = _pairing_tally(
                diffs_row, f"{unit_kind}s", arm_names[a.runId], arm_names[b.runId]
            )
            if not diffs_row.diffs:
                computed.append((metric, None, None))
                lines += [
                    f"### {metric}",
                    "",
                    _NO_PAIRED_DATA.format(unit=unit_kind, metric=metric) + _NO_PAIRED_DATA_TALLY,
                    "",
                    tally,
                    "",
                ]
                continue
            if len(diffs_row.diffs) == 1:
                computed.append((metric, None, None))
                lines += [
                    f"### {metric}",
                    "",
                    _ONE_PAIRED_UNIT.format(unit=unit_kind, metric=metric),
                    "",
                    tally,
                    "",
                ]
                continue
            support_metric = _metric_aggregate(a, metric) or _metric_aggregate(b, metric)
            support = support_metric.support if support_metric is not None else None
            cv = stats.continuous_verdict(
                diffs_row.diffs,
                metric_name=metric,
                family=family,
                alpha_family=pack.metrics.alpha_family,
                unit_kind=unit_kind,
                design_effect=design_effect,
                basis=basis,
                B=_BOOTSTRAP_B,
                seed=pack.seed,
                support=support,
                a_label=arm_names[a.runId],
                b_label=arm_names[b.runId],
            )
            computed.append((metric, cv, None))
            lines += [f"### {metric}", "", cv.text, "", tally, ""]
    else:
        # --- the unchanged homogeneous-binary two-pass flow (§4 S1, gate B-1) --------------------
        # Two passes, because Holm is a property of the **family**: the step a metric is tested at
        # depends on every other member's p-value, so no verdict can be decided until all of them
        # exist. The delivered build ran one pass, decided every metric at the plain Bonferroni
        # `resolving.alpha`, and then printed a Holm table beside verdicts that had not used it —
        # `stats.verdict`'s `alpha_step` was built for exactly this and was passed by nothing (B-1).
        tables: list[tuple[str, stats.PairedOutcomes, stats.ResolvingPower | None]] = []
        p_values: list[float] = []
        tallies: list[str] = []
        for metric in family:
            rows = _paired_rows(a, b, metric, pack)
            outcomes = stats.PairedOutcomes.from_units(
                unit_kind, list(zip(rows.unit_ids, rows.a_ok, rows.b_ok))
            )
            tallies.append(
                _pairing_tally(rows, f"{unit_kind}s", arm_names[a.runId], arm_names[b.runId])
            )
            # An empty intersection has no resolving power to describe — `n_effective` of zero is
            # not a small sample, it is no sample — so the metric gets no verdict rather than a
            # figure computed from nothing (review P3-1). It stays in the family: *k* is fixed by
            # pre-registration, not by how much data arrived.
            rp = (
                stats.resolving_power(
                    outcomes.n_units,
                    unit_kind=unit_kind,
                    design_effect=design_effect,
                    basis=basis,
                    # The two αs come from the pack's pre-registered family, which is the only
                    # thing that fixes *k*. They are different numbers whenever k > 1, and each
                    # bound takes the one that keeps its own sentence true (`-ml` v1.6 §7.1,
                    # review M-ML-6).
                    alpha_family=pack.metrics.alpha_family,
                    alpha_mdd=pack.metrics.alpha_mdd,
                )
                if outcomes.n_units
                else None
            )
            tables.append((metric, outcomes, rp))
            _a, table_b, table_c, _d = outcomes.table
            p_values.append(stats.mcnemar_exact(table_b, table_c))

        steps = stats.holm_steps(p_values, alpha=pack.metrics.alpha_family)

        # `strict=True`: a Holm ladder shorter than the family would otherwise truncate the loop
        # and a pre-registered verdict metric would vanish from the report — indistinguishable, to
        # a reader, from one that was never pre-registered (review P2-3).
        for (metric, outcomes, rp), step, tally in zip(tables, steps, tallies, strict=True):
            if rp is None:
                computed.append((metric, None, step))
                lines += [
                    f"### {metric}",
                    "",
                    _NO_PAIRED_DATA.format(unit=unit_kind, metric=metric) + _NO_PAIRED_DATA_TALLY,
                    "",
                    tally,
                    "",
                ]
                continue
            v = stats.verdict(
                outcomes,
                resolving=rp,
                metric_name=metric,
                family=family,
                a_label=arm_names[a.runId],
                b_label=arm_names[b.runId],
                alpha_step=step.threshold,
                holm_tested=step.tested,
            )
            computed.append((metric, v, step))
            lines += [
                f"### {metric}",
                "",
                v.text,
                "",
                tally,
                f"- marginal Wilson intervals overlap: {'yes' if v.marginal_overlap else 'no'}",
                # **No seed parenthetical, and the audit that replaces it** (`-ml` v1.11 §3.4 Rule
                # 4, §4 S1e Table D). Nothing on the paired binary path resamples any more, so
                # there is no seed to quote; what the bullet owes instead is *which arm bound each
                # bound*, which is cheap and deterministic once the resample is gone and makes
                # Rule 4's mixture legible to a reader instead of inferable only from the code.
                # Naming one arm of a two-arm interval was M-ML-8's error one layer over, and it
                # was wrong on the 12.6-16.5% of tables where MOVER-D binds both bounds.
                _decided_by_line(v),
                "",
                resolving_power_line(rp, pack),
                "",
            ]

    if len(family) > 1:
        # Two co-equal verdicts at alpha=0.05 each carry a ~9.75% chance of at least one false
        # "better" under the null, which is the fishing artefact pre-registration exists to prevent
        # (§3.3, `-ml` §3.3). Family-wise control is mandatory, not optional — but the ladder is
        # only one of the three ways this section's claim can be made honestly (§3.3 (iv), decision
        # (3)): a refused family had no ladder to run, and an all-continuous family takes its
        # correction in the interval instead, so only the homogeneous-binary case below renders one.
        lines += ["### Family-wise error control", ""]
        if mixed_kinds:
            lines += [_MIXED_FAMILY_CORRECTION, ""]
        elif continuous_family:
            lines += [_CONTINUOUS_FAMILY_CORRECTION.format(k=len(family)), ""]
        else:
            # The `decision` column is not decoration: a threshold alone is only interpretable
            # under a step-down the table does not show, so a reader comparing p against it can
            # reach the opposite conclusion from the verdict three paragraphs above (B-1, M-ML-2).
            lines += [
                f"Holm–Bonferroni across the {len(family)} pre-registered verdict metrics, "
                f"applied: the smallest p is tested at alpha/{len(family)}, the next at "
                f"alpha/{len(family) - 1}, and the first non-rejection stops the procedure. Every "
                f"**MDD** above is computed at the family-adjusted alpha="
                f"{pack.metrics.alpha_mdd:g}; every **observable floor** is computed at the "
                f"unadjusted alpha={pack.metrics.alpha_family:g}, the loosest step a member can "
                "face, because that is the only alpha at which the floor's own sentence is true "
                "(§7.1).",
                "",
                "| metric | McNemar p | Holm-adjusted threshold | decision |",
                "|---|---|---|---|",
            ]
            for metric, v, step in computed:
                # A member with no paired table has no p-value to print. `mcnemar_exact(0, 0)`
                # returns 1.0 and would render as `1.000`, which reads as a test that was run and
                # found nothing — so the cell says what actually happened instead (review P3-1).
                p_cell = "—" if v is None else f"{v.mcnemar_p:.3f}"
                lines.append(
                    f"| {metric} | {p_cell} | {step.threshold:.4f} | {_decision(v, step)} |"
                )
            lines.append("")

    # --- presentation: a headline exists only if the pack declared one ---------------------------
    if pack.metrics.headlineMetric is not None:
        if mixed_kinds:
            # §3.3 (iv) — `_NO_PAIRED_DATA` is false here either way: there *is* paired data, and
            # there is no verdict, which is the state that message cannot express truthfully.
            headline_text = "exploratory — no significance claim"
        else:
            headline = next(v for m, v, _ in computed if m == pack.metrics.headlineMetric)
            headline_text = (
                _NO_PAIRED_DATA.format(unit=unit_kind, metric=pack.metrics.headlineMetric)
                if headline is None
                else headline.text
            )
        lines += [f"**Headline ({pack.metrics.headlineMetric}):** {headline_text}", ""]
    else:
        # No summary line above the verdicts, and no arithmetic combining them (§3.3(i)). The
        # metrics stand side by side in the manifest's declared order, which is how they were
        # rendered above.
        lines += [
            "_This pack declares no headline metric: its verdict metrics are co-equal and are "
            "printed side by side, in the manifest's declared order, with no summary line above "
            "them and no arithmetic combining them (§3.3)._",
            "",
        ]

    exploratory = [
        m
        for run in (a, b)
        for m in run.aggregates.named_metrics()
        # §3.3 (iv), decision (1) — a refused family's own members are `in family` and must widen
        # into this section too; a member with no paired data must not, and stays `_NO_PAIRED_DATA`
        # in its own `### <metric>` block above rather than migrating here.
        if m.name not in family or mixed_kinds
    ]
    if exploratory:
        lines += ["### Exploratory metrics", ""]
        seen: set[str] = set()
        for metric in exploratory:
            if metric.name in seen:
                continue
            seen.add(metric.name)
            lines.append(f"- `{metric.name}` — exploratory — no significance claim")
        lines.append("")

    lines += [_OVERLAP_FOOTNOTE, ""]
    return "\n".join(lines) + "\n"
