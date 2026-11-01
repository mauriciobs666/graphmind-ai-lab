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
from dataclasses import dataclass
from typing import Any, Literal, Mapping, NamedTuple, Sequence

from modelbench import decision_stats, decisions, stats
from modelbench.packs import (
    Pack,
    PackConfigError,
    PackRef,
    check_sampling_contract,
    model_block_problem,
)
from modelbench.results import (
    BinaryMetric,
    ContinuousMetric,
    DistributionSummary,
    ExtractionAggregates,
    IncompleteItemRecord,
    InvalidRecord,
    ItemResult,
    MetricKindError,
    RunResult,
    ToolCallAggregates,
)
from modelbench.roles import unit_kind as unit_kind_for_role
from modelbench.scoring.classification import PER_KIND_VERDICT_METRICS

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


#: A candidate replicate's load classification (typed-decision-models plan2 §3.6).
ReplicateKind = Literal["load", "same-load", "assumed"]


@dataclass(frozen=True)
class ReplicateGroup:
    """One candidate arm's replicates on a decision-point `compare` (plan2 §3.6), built by
    `cli._replicate_groups`: every run of the arm's `armKey` in the session, oldest `startedAt`
    first, all of one `candidateContentHash`.

    - `kind[i]`: `assumed` (`basis == "assumed"`, note §Q6 "Load rule"), `load` (the group's first
      non-assumed run, or one whose decision model was not resident at start) or `same-load`;
    - `loadCount`: the `load` replicates;
    - `k`: v1.3 §3.5's reuse count K, computed once for the **earliest** run and used by every
      replicate, so a pair of replicates counts once (note §Q6); `earlierCandidates` are the K
      `candidateContentHash` values it counts, sorted."""

    runs: tuple[RunResult, ...]
    kind: tuple[ReplicateKind, ...]
    loadCount: int
    k: int
    earlierCandidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.kind) != len(self.runs) or not self.runs:
            raise ValueError("a replicate group has one kind per run, and at least one run")
        if self.loadCount != self.kind.count("load") or self.k != len(self.earlierCandidates):
            raise ValueError("a replicate group's loadCount and k must count its kinds and hashes")


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
#: A decision-point guard pack's four per-kind rates are its `verdictMetrics` (architect ruling on
#: impl review P12-m2), so they join here in that change, by `classification`'s own constant.
_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"falseAdvanceRate", "falseSuspendRate", *PER_KIND_VERDICT_METRICS}
)


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


def _aggregate_kind(
    metric: BinaryMetric | ContinuousMetric | DistributionSummary | None,
) -> str:
    """`"binary"` or `"continuous"` for one already-resolved aggregate (or none declared, which
    resolves `"binary"` — `_metric_kind`'s own pre-Table-F default). The one isinstance check both
    `_metric_kind` and `_resolve_reference_kinds` need, so it lives in exactly one place
    (`docs/plans/rank-continuous-reference.md` §3.4)."""
    return "continuous" if isinstance(metric, (ContinuousMetric, DistributionSummary)) else "binary"


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
    return _aggregate_kind(_metric_aggregate(a, name) or _metric_aggregate(b, name))


def _resolve_reference_kinds(
    reference_run: RunResult, candidates: Sequence[RunResult], family: Sequence[str]
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Each family member's kind, resolved across every arm the reference-anchored family
    touches — reference and every candidate — not just a reference-anchored pair at a time
    (`docs/plans/rank-continuous-reference.md` §3.4).

    Deliberately does NOT delegate to `_metric_kind(a, b, name)` pairwise, anchored at
    `reference_run`: `_metric_kind` always prefers its first argument's own declaration when it
    has one, so anchoring every pairwise call at `reference_run` would silently miss two
    *candidates* disagreeing with each other whenever `reference_run` itself declares any
    aggregate for that metric — exactly the failure this function exists to catch.

    Returns `(kinds, per_arm)`. `kinds[metric]` is the resolved kind (`"binary"`/`"continuous"`),
    present only when every arm that declares an aggregate for `metric` agrees (falling back to
    `"binary"` when no arm declares one at all, mirroring `_metric_kind`'s own default). A `metric`
    absent from `kinds` is a genuine cross-arm disagreement; `per_arm[metric]` always carries
    every arm that declared an aggregate for it, keyed by `modelKey`, for the disagreement banner
    to name them regardless of whether `metric` resolved cleanly.
    """
    kinds: dict[str, str] = {}
    per_arm: dict[str, dict[str, str]] = {}
    for metric in family:
        declared: dict[str, str] = {}
        for candidate_run in (reference_run, *candidates):
            agg = _metric_aggregate(candidate_run, metric)
            if agg is not None:
                declared[candidate_run.modelKey] = _aggregate_kind(agg)
        per_arm[metric] = declared
        distinct = set(declared.values())
        if len(distinct) <= 1:
            kinds[metric] = next(iter(distinct), "binary")
    return kinds, per_arm


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

#: Ordinary pre-registered mixed-kind refusal for `rank_report`'s reference-anchored family — the
#: sibling of `_MIXED_FAMILY_MEMBER` (report.py above), reworded for "no candidate's row is
#: verdicted against the reference" rather than "no member of the two-arm comparison is verdicted"
#: (`docs/plans/rank-continuous-reference.md` §4 step 5).
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
#: pre-registered family has a cross-arm disagreement (plan §3.5/§4 step 7) — this metric's own
#: block must not say "mixes binary and continuous members" (`_MIXED_REFERENCE_FAMILY_MEMBER`),
#: because for THIS metric that is not true; the reason it is refused lives one metric over.
_REFERENCE_FAMILY_SIBLING_DISAGREEMENT_MEMBER = (
    "**no reference-anchored verdict.** A sibling metric in this pack's pre-registered "
    "verdict-metric family has a cross-arm kind disagreement (see the banner above); the whole "
    "optional reference-anchored family (FR-8) is refused because `k` is fixed across the whole "
    "family and cannot be honestly re-sized once results already exist. No candidate's row for "
    "`{metric}` is verdicted against `{reference}`."
)

#: Printed once, before the per-metric loop, when `_resolve_reference_kinds` finds a genuine
#: cross-arm disagreement — mirrors PACK VERSION MISMATCH / SCHEMA VERSIONS IN THIS COMPARISON
#: (`rank_report`, above): visible, never silent, never a reason to drop a record.
#:
#: **The primary reason a disagreement refuses the whole family is a data-integrity failure, not
#: a multiplicity/k-shrinkage argument** (`docs/reviews/rank-continuous-reference-ml.md` §3,
#: `data-scientist`'s methodology finding on the plan this note builds on): a metric's aggregate
#: kind is supposed to be a type fact, and a metric that resolves to a different type depending
#: on which arm answers it is a self-contradictory record — the same class of problem as the
#: pack/version/schema mismatch banners above, none of which invoke a multiplicity argument
#: either. Refusing the whole family (rather than only the disagreeing metric's own cells) is a
#: conservative response to that epistemic uncertainty: once one metric's type declaration is
#: shown unreliable for some arm, the same arm's other declared aggregates are owed the same
#: distrust. `k` staying fixed across the family is a real, secondary property of the fix (this
#: plan already keeps the same `k` correctly-sized for the clean-metric case), not the forcing
#: reason for refusing the whole family in the first place.
_REFERENCE_KIND_DISAGREEMENT_EXPLANATION = (
    "This is a data-integrity problem, not a foreseen pre-registration choice: a metric's "
    "aggregate kind is supposed to be a type fact, never something that varies by which arm "
    "answers it. A metric that resolves to a different type depending on which arm answers it is "
    "a self-contradictory record, in the same class as a pack/version/schema mismatch — and once "
    "one metric's type declaration is shown unreliable for some arm, the same arm's other "
    "declared aggregates are owed the same distrust. The whole optional reference-anchored "
    "family (FR-8) is refused for the metric(s) named above, not just the disagreeing metric's "
    "own cells."
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
    metric: str, reference_run: RunResult, text: str, tallies: Sequence[str] = ()
) -> list[str]:
    """The refused-family block for one metric. `tallies` (m1,
    `docs/reviews/rank-continuous-reference.md`) carries one `_pairing_tally` line per candidate
    for the ordinary pre-registered-mixed case only — each metric's own kind is still cleanly
    resolved there, only the *combination* is mixed, mirroring `compare_report`'s own precedent
    (report.py above, its mixed-kind branch). The disagreement case passes none: no resolvable
    kind exists to build a tally by."""
    lines = [
        f"#### Reference-anchored family — {metric} vs `{reference_run.modelKey}`", "",
        text, "",
    ]
    if tallies:
        lines.extend(tallies)
        lines.append("")
    return lines


def _render_reference_family_continuous(
    *,
    metric: str,
    pack: PackRef,
    reference_run: RunResult,
    candidates: Sequence[RunResult],
    correction_k: int,
    unit_kind: str,
) -> list[str]:
    """The continuous sibling of `_render_reference_family` (report.py above). Every candidate's
    interval is decided independently at the same fixed `alpha_used` (`-ml` §3.4) — no Holm
    ladder, no per-row threshold, no "not tested" state.

    **M1** (`docs/reviews/rank-continuous-reference.md`) — `alpha_used` is computed via
    `stats.alpha_used`, the same one-line helper `stats.continuous_verdict` itself calls, rather
    than a second, independent `alpha_family / correction_k` here: one formula, one home, so the
    caption and `cv.alpha_used` can never drift apart. The caption still prints before any `cv`
    exists (the correction is fixed once per table, before any candidate's own interval is
    computed), so this calls the helper directly rather than reading `cv.alpha_used` off some
    candidate's result.
    """
    used_alpha = stats.alpha_used(pack.metrics.alpha_family, correction_k)
    lines = [
        f"#### Reference-anchored family — {metric} vs `{reference_run.modelKey}`", "",
        "_exploratory — no significance claim outside this family_", "",
        f"_`alpha_family={pack.metrics.alpha_family:g}, k={correction_k}, "
        f"alpha_used={used_alpha:g} ({100 * (1 - used_alpha):g}% CI)`_", "",
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


#: Every scalar count an aggregate kind carries beside its rates — never a rate, never in
#: `named_metrics()`, so the Arms/rank tables never reach them — keyed by `Aggregates.kind`, in
#: the dataclass's own field order. A kind absent here carries none (`retrieval`, `toolcalls`,
#: whose structural counts have `_render_funnel` instead). Bound to `results.py`'s dataclasses by
#: `tests/test_report.py` (every `int` field but `ClassificationAggregates.n`, the run's item
#: count, which is a denominator rather than a count of anything that happened to an item).
_SCORER_COUNTS: dict[str, tuple[str, ...]] = {
    "classification": ("parseFailures",),
    "extraction": (
        "parseFailures", "malformedSpecCount", "schemaViolationCount", "luckyPassCount",
    ),
    "grounding": ("parseFailures",),
}

#: One clause per counter, printed under `## Scorer counts`. Wording follows the field's own
#: definition in `results.py` / the producing scorer's `aggregate()` docstring.
_SCORER_COUNT_GLOSS: dict[str, str] = {
    "parseFailures": "replies the scorer could not parse",
    "malformedSpecCount": "specs that failed structural validation (Layer A)",
    "schemaViolationCount": "specs that failed schema-bound validation (Layer B)",
    "luckyPassCount": (
        "items stamped `answerable: false` whose exploratory spec nonetheless scored correct — "
        "each is **also** counted in its shape's `exactMatchBy*` k, additive, never an "
        "alternative"
    ),
}

_LUCKY_PASS_FOOTNOTE = (
    "_† This shape's k **includes lucky passes** (`luckyPassCount`, `scoring/extraction.py` "
    "`aggregate()`): items stamped `answerable: false` whose exploratory spec nonetheless matched "
    "the expected answer. Each is counted in the shape's successes as well — additive, never an "
    "alternative — so a marked k/n is not a count of genuine answers._"
)

_LUCKY_PASS_DISAGREEMENT_FOOTNOTE = (
    "_‡ The arm's stored `luckyPassCount` disagrees with the lucky passes its own items place on a "
    "shape ({arms}) — an inconsistent record, since `aggregate()` derives one from the other. "
    "Every shape row of that arm is marked: the † counts follow the items, and where the stored "
    "count is higher the remainder cannot be placed, so any of them may include one._"
)


#: The per-shape metrics a lucky pass is counted in (`scoring/extraction.py` names each
#: `f"exactMatchBy{shape}"`). `ExtractionAggregates.byShape` also carries
#: `unanswerableAbstainRate`, which a lucky item scores too — but as an abstention, not as the
#: lucky pass, so it is never marked.
_LUCKY_PASS_METRIC_PREFIX = "exactMatchBy"


def _lucky_shape_metrics(agg: ExtractionAggregates) -> set[str]:
    return {m.name for m in agg.byShape if m.name.startswith(_LUCKY_PASS_METRIC_PREFIX)}


def _lucky_by_shape(run: RunResult) -> tuple[dict[str, int], int]:
    """Where an extraction run's lucky passes sit: `({exactMatchBy* name: lucky passes in its
    k}, stored luckyPassCount)`. `({}, 0)` for every other aggregate kind.

    `aggregate()` stores one run-level scalar, but its docstring's obligation is per shape rate —
    any rendering of `luckyPassCount` "must not let the shape rate stand alone unlabelled". The
    attribution reads only what the scorer **declared** on each item: `detail["luckyPass"] is
    True` and a `True` outcome for that `exactMatchBy*` metric via `ItemResult.scored_outcome`
    (the one reader of an item's outcome). Nothing is inferred; a record that cannot be read for
    a metric attributes nothing to it. When the placed total and the stored count disagree, in
    **either** direction, the record is inconsistent (`aggregate()` derives the count from the
    same flags in one pass): `_lucky_mark` then marks every shape row of the arm `‡` rather than
    none — fail-safe where the stored count is higher, since the remainder cannot be placed — and
    the footnote prints both numbers rather than silently preferring one."""
    agg = run.aggregates
    if not isinstance(agg, ExtractionAggregates):
        return {}, 0
    lucky_items = [it for it in run.items if it.detail.get("luckyPass") is True]
    by_shape: dict[str, int] = {}
    for name in sorted(_lucky_shape_metrics(agg)):
        count = 0
        for it in lucky_items:
            try:
                if it.scored_outcome(name) is True:
                    count += 1
            except (IncompleteItemRecord, MetricKindError):
                continue
        if count:
            by_shape[name] = count
    return by_shape, agg.luckyPassCount


def _lucky_mark(run: RunResult, metric: str) -> str:
    """The suffix a shape row's k/n cell carries: `" † (N lucky)"` when N of its successes are
    lucky passes, `" ‡"` when the arm's stored `luckyPassCount` disagrees with the lucky passes
    its items place, `""` otherwise — and
    always `""` for any metric but an `exactMatchBy*` one, so every other row renders
    byte-for-byte as before."""
    agg = run.aggregates
    if not isinstance(agg, ExtractionAggregates) or metric not in _lucky_shape_metrics(agg):
        return ""
    by_shape, stored = _lucky_by_shape(run)
    mark = f" † ({by_shape[metric]} lucky)" if metric in by_shape else ""
    if stored != sum(by_shape.values()):
        mark += " ‡"
    return mark


def _lucky_footnotes(cells: Sequence[tuple[str, RunResult, str]]) -> list[str]:
    """The legend for whichever lucky-pass markers a table actually printed — none, one, or both.
    `cells` is `(arm label, run, mark)` per rendered binary row; the `‡` legend names each marked
    arm with its stored count and its items' placed count, both printed, neither preferred."""
    lines: list[str] = []
    if any("†" in mark for _, _, mark in cells):
        lines += [_LUCKY_PASS_FOOTNOTE, ""]
    disagreeing: dict[str, str] = {}
    for label, run, mark in cells:
        if "‡" in mark and label not in disagreeing:
            by_shape, stored = _lucky_by_shape(run)
            disagreeing[label] = (
                f"{label}: stored `luckyPassCount` {stored}, items place {sum(by_shape.values())}"
            )
    if disagreeing:
        arms = "; ".join(disagreeing.values())
        lines += [_LUCKY_PASS_DISAGREEMENT_FOOTNOTE.format(arms=arms), ""]
    return lines


def _render_scorer_counts(labelled: Sequence[tuple[str, RunResult]]) -> list[str]:
    """Every arm's `_SCORER_COUNTS` fields, verbatim by name, one line per arm, plus a legend for
    the fields shown. `[]` when every such count in the report is zero — a run with nothing to
    report under these fields renders exactly as it did before they were printed at all (and an
    all-zero block for a role whose scorer never populates the field, `grounding`'s
    `parseFailures`, would read as a measurement). Once any arm has a non-zero count, every arm's
    counts print, zeros included, so arms can be read side by side."""
    rows = [
        (label, [(name, getattr(r.aggregates, name)) for name in _SCORER_COUNTS[r.aggregates.kind]])
        for label, r in labelled
        if r.aggregates.kind in _SCORER_COUNTS
    ]
    if not any(value for _, counts in rows for _, value in counts):
        return []
    lines = [
        "## Scorer counts",
        "",
        "_Scalar counts the scorer records beside its rates — never a rate, never in a verdict._",
        "",
    ]
    for label, counts in rows:
        lines.append(
            f"- {label}: " + " · ".join(f"`{name}` {value}" for name, value in counts)
        )
    lines.append("")
    shown = list(dict.fromkeys(name for _, counts in rows for name, _ in counts))
    legend = "; ".join(f"`{name}`: {_SCORER_COUNT_GLOSS[name]}" for name in shown)
    lines += [f"_{legend}._", ""]
    return lines


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
    lucky_cells: list[tuple[str, RunResult, str]] = []
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
            lucky = _lucky_mark(r, metric)
            lucky_cells.append((r.modelKey, r, lucky))
            count_cell = f"{agg.successes}/{agg.n}{lucky}"
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
    lines += _lucky_footnotes(lucky_cells)

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
    aggs = [_metric_aggregate(r, metric) for r in runs]
    first_agg = next((a for a in aggs if a is not None), None)
    if first_agg is None:
        return []
    if _aggregate_kind(first_agg) == "continuous":
        # `-ml` §3.5 — this sentence's whole vocabulary (McNemar/Wilson-shaped: "resolves
        # differences of >=X pp", "80% power") is undefined for a continuous metric, exactly as
        # `compare_report`'s own continuous branch never calls `resolving_power` for the same
        # reason (`stats.continuous_verdict`'s own docstring, `stats.py:1675-1677`). Printing it
        # anyway is a confidently-stated, methodologically meaningless figure, live and wrong for
        # `embedder-graphrag-retrieval`'s `mrr` before this fix.
        return []
    ns = [a.n for a in aggs if a is not None]
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
    reference_kinds: dict[str, str] = {}
    reference_kind_details: dict[str, dict[str, str]] = {}
    reference_kind_disagreements: list[str] = []
    reference_family_refused = False
    if reference_run is not None:
        candidates = [r for r in runs if r.modelKey != reference_run.modelKey]
        if candidates:
            # §3.3 (iv) / plan §3.4-§3.5 — a family member's kind is a type fact, resolved across
            # every arm the reference-anchored family touches (reference and every candidate), not
            # merely a reference-anchored pair at a time: a metric absent from `reference_kinds`
            # is a genuine cross-arm disagreement, distinct from an ordinary pre-registered mixed
            # family (where every member resolves cleanly and only the combination is mixed).
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
                # --- unchanged existing binary combined-ladder construction --------------------
                for metric in family:
                    for cand in candidates:
                        paired = _paired_rows(reference_run, cand, metric, pack)
                        outcomes = stats.PairedOutcomes.from_units(
                            unit_kind, list(zip(paired.unit_ids, paired.a_ok, paired.b_ok))
                        )
                        combined_outcomes[(cand.modelKey, metric)] = outcomes
                        _a, b_, c_, _d = outcomes.table
                        # `-ml` review Pass2-1: `k` is fixed by pre-registration, never by how
                        # much data arrived — a candidate with no paired data for one metric still
                        # consumes a Holm rank, via the same `mcnemar_exact(0, 0) = 1.0`
                        # empty-intersection handling `compare_report`'s own homogeneous-binary
                        # path already relies on (its own unconditional
                        # `p_values.append(stats.mcnemar_exact(table_b, table_c))` below), reused
                        # rather than reinvented.
                        combined_p_values.append(stats.mcnemar_exact(b_, c_))
                        combined_cells.append((cand.modelKey, metric))
                # ONE combined call over (metric x candidate), never one call per metric — the
                # plan's originally-rejected per-metric default under-corrects the family-wise
                # error rate (§3.2.2).
                combined_steps = stats.holm_steps(
                    combined_p_values, alpha=pack.metrics.alpha_family
                )
                correction_k = len(combined_p_values)
            elif not reference_family_refused and resolved_kinds == {"continuous"}:
                # m2 (`docs/reviews/rank-continuous-reference.md`) — written explicitly rather
                # than as a bare `elif`/implicit `else`: an empty `family` (a malformed pack, or a
                # test fixture bug) resolves `resolved_kinds == set()`, which is neither
                # `{"binary"}` nor `{"continuous"}`, and must not silently fall into this branch
                # and set `correction_k = 0` under a comment claiming the continuous case.
                correction_k = len(family) * len(candidates)

    for metric in members:
        lines += _render_one_rank_table(runs, metric=metric, pack=pack, footprints=footprints)
        if len(runs) >= 2:
            lines += _rank_resolving_power_lines(runs, pack, family, metric)
        if reference_run is not None and candidates:
            if reference_family_refused:
                if metric in reference_kind_disagreements:
                    text = _REFERENCE_KIND_DISAGREEMENT_MEMBER.format(
                        metric=metric, reference=reference_run.modelKey
                    )
                    tallies: tuple[str, ...] = ()
                elif reference_kind_disagreements:
                    # This metric itself resolved cleanly; a SIBLING metric's disagreement is what
                    # forced the whole family refused (§3.5) — `_MIXED_REFERENCE_FAMILY_MEMBER`'s
                    # "mixes binary and continuous members" wording would misstate the reason for
                    # THIS metric's own refusal.
                    text = _REFERENCE_FAMILY_SIBLING_DISAGREEMENT_MEMBER.format(
                        metric=metric, reference=reference_run.modelKey
                    )
                    tallies = ()
                else:
                    # m1 (`docs/reviews/rank-continuous-reference.md`) — the ordinary
                    # pre-registered-mixed case (not a disagreement): this metric's own kind is
                    # still individually resolved cleanly, only the family's *combination* is
                    # mixed, so its own per-candidate `_pairing_tally` diagnostic still renders,
                    # mirroring `compare_report`'s existing mixed-kind precedent.
                    text = _MIXED_REFERENCE_FAMILY_MEMBER.format(
                        kind=reference_kinds[metric], metric=metric,
                        reference=reference_run.modelKey,
                    )
                    paired_or_diffs = (
                        _paired_rows if reference_kinds[metric] == "binary" else _paired_diffs
                    )
                    tallies = tuple(
                        _pairing_tally(
                            paired_or_diffs(reference_run, cand, metric, pack),
                            f"{unit_kind}s", reference_run.modelKey, cand.modelKey,
                        )
                        for cand in candidates
                    )
                lines += _render_reference_family_refused(
                    metric, reference_run, text, tallies=tallies
                )
            elif reference_kinds[metric] == "binary":
                lines += _render_reference_family(
                    metric=metric, pack=pack, reference_run=reference_run, candidates=candidates,
                    cells=combined_cells, steps=combined_steps, outcomes=combined_outcomes,
                    correction_k=correction_k, unit_kind=unit_kind,
                )
            else:
                lines += _render_reference_family_continuous(
                    metric=metric, pack=pack, reference_run=reference_run, candidates=candidates,
                    correction_k=correction_k, unit_kind=unit_kind,
                )

    # Once, after every ranked table: these are per-model, not per-metric. In-scope models only —
    # an excluded one is named in the banner above and is not in the report.
    lines += _render_scorer_counts([(r.modelKey, r) for r in runs])
    return "\n".join(lines) + "\n"


# --- the decision-point bar section (typed-decision-models v1.3 §3.5–§3.6; plan2 §3.6; plan3 §3.6)
#
# One section, printed on a decision-point pack's `compare`, that **states** the requirement's bar
# and never acts on it: the exit code is unchanged and nothing is written outside `reports/`
# (v1.3 §3.6). Every figure is `decision_stats`' (the method note's); this code composes and
# prints, and decides nothing beyond the per-replicate state rule below (plan2 §3.6 "Bar states").

_LABELER_CAVEAT = (
    "Labels: {labeler}, inputs generated by {generator}, unreviewed (FR-5). Results measure "
    "agreement with this labeler on synthetic inputs, not verified correctness."
)
_BAR_NOTE = (
    "_This section states the requirement's bar and never acts on it: no exit code depends on "
    "it, and adopting a candidate stays a human edit (v1.3 §3.6). Each bar is judged on its "
    "measured value, its interval printed beside it (FR-11a)._"
)
_EXPLORATORY = "exploratory — no significance claim"
#: Per-replicate states this section adds beside `decision_stats`' own (plan2 §3.6, §3.2).
_TIMEOUT_REASON = (
    f"not assessable (chat timeout ≠ {decisions.DECISION_POINT_CHAT_TIMEOUT_SECONDS} s)"
)
_NO_INCUMBENT = "not assessable (no incumbent run in this comparison)"
_INCUMBENT_INCONSISTENT = "not assessable (the incumbent run's aggregates disagree with its items)"
_REPLICATE_INCONSISTENT = "not assessable (this replicate's aggregates disagree with its items)"
_SIDE_CHECK_PREFIX = "sideCheck"
_DIFFERENT_CONTENT = "not assessable (arms ran on different pack content"


def _pack_content_reason(
    dp: DecisionPointPack,
    incumbent: RunResult | None,
    replicates: Mapping[str, ReplicateGroup],
) -> str | None:
    """The bar's own pack-content guard (impl review P18-M1). Candidates reach the report only as
    `replicates`, so the `runs` banners (`PACK CONTENT HASH MISMATCH`) never see them; and the bar
    reads strata and counts from the pack on disk, a third content. Any disagreement among the
    incumbent's, every replicate's and the loaded pack's hash makes every bar not assessable."""
    runs = [r for g in replicates.values() for r in g.runs]
    if incumbent is not None:
        runs.append(incumbent)
    hashes = {r.fingerprint.get("packContentHash") for r in runs} | {dp.contentHash}
    if len(hashes) == 1:
        return None
    shown = ", ".join(
        sorted("none recorded" if h is None else str(h)[:8] for h in hashes)
    )
    return f"{_DIFFERENT_CONTENT}: {shown})"


@dataclass(frozen=True)
class DecisionPointPack:
    """What the bar section reads from the pack itself (v1.3 §3.5): the labeler and the input
    generator the caveat names (plan4 §3.9), the incumbent, every item's `stratum` by `itemId`
    (`None` for an item with none), which `pair_items` takes so the bars' drop counts stay
    eval-only (impl review P16-n2), each `eval` item's `genRunId` by `itemId` (plan4 §3.9, the
    design effect's grouping), and the pack's `contentHash`, which every arm's recorded
    `packContentHash` must equal for a bar to be stated (impl review P18-M1)."""

    labeler: str
    generator: str
    incumbent: str | None
    strata: Mapping[str, str | None]
    gen_run_ids: Mapping[str, str]
    contentHash: str

    @classmethod
    def from_pack(cls, pack: Pack) -> "DecisionPointPack":
        """Raises `PackConfigError` when the `labels.labeler` or `labels.generator` the caveat
        names lacks a non-empty string `provider` or `model` (`packs.model_block_problem`), or an
        `eval` item lacks a non-empty string `genRunId` — each a pack `validate_pack` refuses, the
        blocks by this same predicate (FR-5, plan4 §3.9; impl review P25-m1)."""
        labels = pack.manifest.get("labels")
        labeler = _model_name(pack, labels, "labeler")
        generator = _model_name(pack, labels, "generator")
        decision_point = pack.manifest.get("decisionPoint")
        incumbent = (
            decision_point.get("incumbent") if isinstance(decision_point, Mapping) else None
        )
        rows = list(pack.iter_items())
        gen_run_ids: dict[str, str] = {}
        for row in rows:
            if row.get("stratum") != "eval":
                continue
            gen_run_id = row.get("genRunId")
            if not (isinstance(gen_run_id, str) and gen_run_id):
                raise PackConfigError(
                    f"{pack.packId}: eval item {row.get('itemId')!r} has no genRunId for the "
                    "design effect's grouping (plan4 §3.9)"
                )
            gen_run_ids[str(row.get("itemId"))] = gen_run_id
        return cls(
            labeler=labeler,
            generator=generator,
            incumbent=incumbent if isinstance(incumbent, str) else None,
            strata={str(row.get("itemId")): row.get("stratum") for row in rows},
            gen_run_ids=gen_run_ids,
            contentHash=pack.contentHash,
        )

    @property
    def eval_count(self) -> int:
        return sum(1 for s in self.strata.values() if s == "eval")

    @property
    def side_check_count(self) -> int:
        return sum(1 for s in self.strata.values() if s == "side-check")


def _model_name(pack: Pack, labels: Any, block: str) -> str:
    """`provider/model` from the pack's `labels.<block>`, for the caveat (FR-5, plan4 §3.9),
    checked by `validate_pack`'s own predicate (`packs.model_block_problem`)."""
    problem = model_block_problem(labels, block)
    if problem is not None:
        raise PackConfigError(f"{pack.packId}: {problem}")
    return f"{labels[block]['provider']}/{labels[block]['model']}"


def _rate_cell(k: int, n: int) -> str:
    """`k/n [lo, hi]` with a Wilson interval, or `0/0 [—]` when there is no denominator."""
    if not n:
        return "0/0 [—]"
    lo, hi = stats.wilson_interval(k, n)
    return f"{k}/{n} [{lo:.3f}, {hi:.3f}]"


def _count_cell(run: RunResult, name: str) -> str:
    """One exploratory count from the run's own aggregate, `k/n`, or why there is none."""
    metric = _metric_aggregate(run, name)
    if isinstance(metric, BinaryMetric):
        return f"{name} {metric.successes}/{metric.n}"
    return f"{name} — (not recorded on this surface)"


def _drops_text(drops: Mapping[tuple[str, str], int]) -> str:
    if not drops:
        return "none dropped"
    parts = [f"{arm} {reason} {count}" for (arm, reason), count in sorted(drops.items())]
    return "dropped: " + ", ".join(parts)


def _holds_clause(line: decision_stats.ParityLine, k: int) -> str:
    """Note §Q3.3's `holds (…)` form, and §Q3.5's reuse-adjusted limit when K ≥ 1."""
    if line.mdd is None:
        return (
            f"holds (no worsening is detectable at n = {line.n_effective}: fewer than 5 items)"
        )
    psi = "—" if line.psi_hat is None else f"{line.psi_hat:.2f}"
    clause = f"could only detect ≥ {_pp(line.mdd)} pp at n = {line.n_effective}, ψ̂ = {psi}"
    if k >= 1 and line.mdd_reuse is not None:
        clause += f"; reuse-adjusted at K = {k}: ≥ {_pp(line.mdd_reuse)} pp"
    return f"holds ({clause})"


def _design_effect_text(design_effect: float) -> str:
    """Note §Q6 "Print" beside each FR-1 cell and FR-10 class: `DE <x.xx>`."""
    return f"DE {design_effect:.2f}"


def _family_design_effect(
    outcomes: Mapping[str, Sequence[bool]],
    ids: Mapping[str, Sequence[str]],
    gen_run_ids: Mapping[str, str],
) -> decision_stats.FamilyDesignEffect:
    """Note §Q6 "Design effect" for one family (plan4 §3.9): each key's per-item outcomes,
    grouped by each item's `genRunId` through the ids parallel to them; an empty key stays an
    empty cell."""
    cells: dict[str, dict[str, list[bool]]] = {}
    for key, values in outcomes.items():
        by_run = cells.setdefault(key, {})
        for outcome, item_id in zip(values, ids[key], strict=True):
            by_run.setdefault(gen_run_ids[item_id], []).append(outcome)
    return decision_stats.family_design_effect(cells)


def _family_design_effect_line(family: decision_stats.FamilyDesignEffect, unit: str) -> str:
    """Note §Q6 "Print": the family's line, `unit` being `cells` (FR-1) or `classes` (FR-10),
    singular at `k = 1` (`1 cell`, `1 class`)."""
    source = decision_stats.DESIGN_EFFECT_SOURCE
    if family.icc is None:
        return f"design effect: {source} not estimable ({family.reason}) — DE 1.00"
    k = family.contributing
    noun = _SINGULAR_UNIT[unit] if k == 1 else unit
    return f"design effect: {source} {family.icc:+.3f} pooled over {k} {noun}"


_SINGULAR_UNIT = {"cells": "cell", "classes": "class"}


def _parity_text(
    metric: str, line: decision_stats.ParityLine, *, k: int, state: str, design_effect: float
) -> str:
    """Note §Q3.4's printed line, with the cell's design effect (plan4 §3.9). `state` is the
    replicate's: on `met`/`not met` the line ends with the cell's own reading; on a `not
    assessable (…)` replicate it ends with that state, never with a `holds` the replicate cannot
    claim (plan2 §3.6)."""
    if line.n:
        diff = f"{(line.e1 - line.e2) / line.n * 100:+.1f} pp"
        upper = f"{line.U * 100:.1f}" if line.U is not None else "—"
    else:
        diff, upper = "—", "—"
    figures = (
        f"{metric}: candidate {_rate_cell(line.e1, line.n)} vs incumbent "
        f"{_rate_cell(line.e2, line.n)}; diff {diff}; b={line.b}, c={line.c}, "
        f"p_worse={line.p_worse:.4f}; worse by at most {upper} pp (one-sided 95%); "
        f"{_design_effect_text(design_effect)}"
    )
    if state not in (decision_stats.MET, decision_stats.NOT_MET):
        return f"{figures} — {state}"
    if line.state == "worse":
        return f"{figures} — not met: candidate worse"
    return f"{figures} — {_holds_clause(line, k)}"


def _speedup_text(paired: decision_stats.PairedItems, *, seed: int) -> tuple[str, float | None]:
    """Note §Q4.1's figure, `R = Σx/Σy` with its paired bootstrap interval, and the measured
    `R` (or `None` when none can be computed, and why)."""
    x = [row[1] for row in paired.latency]
    y = [row[2] for row in paired.latency]
    try:
        ratio = decision_stats.speedup_ratio(
            x,
            y,
            seed=seed,
            B=decision_stats.BOOTSTRAP_B,
            levels=(stats.LEVEL_CI95_LO, stats.LEVEL_CI95_HI),
        )
    except ValueError as exc:
        return f"FR-2 speed-up: no ratio over {len(x)} paired eval item(s) ({exc})", None
    text = (
        f"FR-2 speed-up: {ratio.point:.2f} [{ratio.lo:.2f}, {ratio.hi:.2f}] — the incumbent's "
        f"mean {sum(x) / len(x):.0f} ms over the candidate's {sum(y) / len(y):.0f} ms, "
        f"{len(x)} paired eval items (paired percentile bootstrap 95%, B={ratio.B}, "
        f"seed {ratio.seed}); the bar is ≥ {decisions.FR2_MIN_SPEEDUP:g} on the measured ratio"
    )
    return text, ratio.point


def _fr1_state(paired: decision_stats.PairedItems, lines: Mapping[str, decision_stats.ParityLine]):
    """Plan2 §3.6 "Bar states" with note §Q3.1/§Q4.3: a paired scored eval n below
    `MIN_EVAL_ITEMS` is `not assessable (scored n < 200)` whatever the cells say; otherwise any
    `worse` cell is `not met`, else `met`."""
    if paired.scored_n < decisions.MIN_EVAL_ITEMS:
        return decision_stats.SCORED_N_BELOW_MIN
    if any(line.state == "worse" for line in lines.values()):
        return decision_stats.NOT_MET
    return decision_stats.MET


def _fr2_state(paired: decision_stats.PairedItems, ratio: float | None) -> str:
    if paired.scored_n < decisions.MIN_EVAL_ITEMS:
        return decision_stats.SCORED_N_BELOW_MIN
    if ratio is None:
        return "not assessable (no speed-up ratio)"
    met = decisions.bar_holds(ratio, decisions.FR2_MIN_SPEEDUP, direction="at-least")
    return decision_stats.MET if met else decision_stats.NOT_MET


class _ReplicateBars(NamedTuple):
    """One replicate's block and its two bar states: FR-1 and FR-2 on a speed-up point, FR-10's
    per-class agreement and mean latency on a new-ability point (plan2 §3.6)."""

    lines: list[str]
    quality: str
    latency: str


#: Each bar kind's two bars, as the per-replicate and overall lines name them (plan2 §3.6).
_BAR_NAMES: Mapping[str, tuple[str, str]] = {
    "speed-up": ("FR-1 parity", "FR-2 speed-up"),
    "new-ability": ("FR-10 per-class agreement", "FR-10 mean latency"),
}


def _replicate_head(index: int, run: RunResult, kind: str, counts: Sequence[str]) -> list[str]:
    """A replicate's heading, its same-load line and its exploratory counts (plan2 §3.6; plan3
    §3.6: `overLimit k/n` beside the others, `0/n` included)."""
    suffix = ", not counted as a load" if kind == "same-load" else ""
    lines = [f"#### replicate {index} — run `{run.runId}`, basis {run.basis}{suffix}", ""]
    if kind == "same-load":
        lines.append(
            f"- replicate {index}: same load as an earlier run — not counted as a load"
        )
    lines.append("- " + " · ".join(_count_cell(run, name) for name in counts))
    return lines


def _render_replicate(
    index: int,
    run: RunResult,
    kind: str,
    *,
    group: ReplicateGroup,
    incumbent: RunResult | None,
    blocked: str | None,
    dp: DecisionPointPack,
    pack: PackRef,
) -> _ReplicateBars:
    """Plan2 §3.6's per-replicate block (with plan3 §3.6's `overLimit` prints). `blocked` is a
    comparison-wide `not assessable (…)` state (no incumbent, chat timeout ≠ 180 s, arms on
    different pack content — which, like an inconsistent arm, leaves nothing to pair)."""
    lines = _replicate_head(
        index, run, kind, ("handoffShare", "decisionCallFailures", "overLimit")
    )
    state = decision_stats.LOAD_CHANGED if kind == "assumed" else blocked
    if _aggregate_item_mismatches(run, pack):
        state = _REPLICATE_INCONSISTENT
    unpairable = (_REPLICATE_INCONSISTENT, _INCUMBENT_INCONSISTENT)
    different_content = blocked is not None and blocked.startswith(_DIFFERENT_CONTENT)
    if incumbent is None or state in unpairable or blocked in unpairable or different_content:
        reason = state or blocked or _NO_INCUMBENT
        lines += [f"- FR-1 (replicate {index}): {reason}", f"- FR-2 (replicate {index}): {reason}"]
        return _ReplicateBars(lines + [""], reason, reason)
    try:
        paired = decision_stats.pair_items(
            incumbent.items, run.items, role=pack.role, strata=dp.strata
        )
    except ValueError as exc:
        reason = f"not assessable (cannot pair with the incumbent: {exc})"
        lines += [f"- FR-1 (replicate {index}): {reason}", f"- FR-2 (replicate {index}): {reason}"]
        return _ReplicateBars(lines + [""], reason, reason)

    cells = PER_KIND_VERDICT_METRICS  # the guard's four FR-1 cells; `pair_items` seeds each
    # Plan4 §3.9: one family over the four cells' per-pair discordance (`cand ≠ inc`); the
    # run-level `designEffect` stays 1.0 for `compare_report`.
    family = _family_design_effect(
        {cell: [cand != inc for cand, inc in paired.outcomes[cell]] for cell in cells},
        paired.outcome_ids,
        dp.gen_run_ids,
    )
    effects = family.effects
    parity = {
        cell: decision_stats.parity_cell(
            paired.outcomes[cell],
            design_effect=effects[cell],
            k=group.k,
            alpha=decision_stats.FR1_ALPHA,
        )
        for cell in cells
    }
    ratio_text, ratio = _speedup_text(paired, seed=pack.seed)
    fr1 = state or _fr1_state(paired, parity)
    fr2 = state or _fr2_state(paired, ratio)
    lines.append(
        f"- scored eval n: {paired.scored_n} of the pack's {dp.eval_count} eval items, beside "
        f"FR-11's {decisions.MIN_EVAL_ITEMS}; {_drops_text(paired.drops)}"
    )
    lines.append(
        "- FR-1 parity, per cell (one-sided exact McNemar, α = "
        f"{decision_stats.FR1_ALPHA:g}, unadjusted; `p_worse` decides, `U` is descriptive):"
    )
    lines.append(f"  - {_family_design_effect_line(family, 'cells')}")
    for cell in cells:
        line = parity[cell]
        minimum = decision_stats.GUARD_CELL_METHOD_MIN
        thin = " — below it: printed, never refused" if line.n < minimum else ""
        lines += [
            f"  - {_parity_text(cell, line, k=group.k, state=fr1, design_effect=effects[cell])}",
            f"    - scored n {line.n} beside method minimum {minimum}{thin}; "
            f"overLimit {paired.over_limit[cell]}/{line.n}",
        ]
    lines += [
        f"- FR-1 (replicate {index}): {fr1}",
        f"- {ratio_text} — {fr2}",
    ]
    pooled = [m for m in paired.outcomes if m not in cells and not m.startswith(_SIDE_CHECK_PREFIX)]
    for metric in pooled:
        pairs = paired.outcomes[metric]
        cand, inc = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        lines.append(
            f"- {metric} (pooled): candidate {_rate_cell(cand, len(pairs))} vs incumbent "
            f"{_rate_cell(inc, len(pairs))} — {_EXPLORATORY}"
        )
    lines += [
        "",
        f"Side check — {dp.side_check_count} human-curated items (does not affect pass/fail)",
        "",
    ]
    side = [m for m in paired.outcomes if m.startswith(_SIDE_CHECK_PREFIX)]
    for metric in side:
        pairs = paired.outcomes[metric]
        cand, inc = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        lines.append(
            f"- {metric}: candidate {_rate_cell(cand, len(pairs))} vs incumbent "
            f"{_rate_cell(inc, len(pairs))} — {_EXPLORATORY}"
        )
    lines += [f"- side check: {_drops_text(paired.side_check_drops)}", ""]
    return _ReplicateBars(lines, fr1, fr2)


def _final_answer(item: ItemResult) -> tuple[tuple[str, bool], ...] | None:
    """An item's final answer as its declared outcomes (an answer is never stored, AGENTS
    invariant; the same item's events differ iff its answers do), or `None` with no answer."""
    declared = sorted(m for m in item.scoreable if item.scoreable[m])
    if not declared:
        return None
    return tuple((m, item.scored_outcome(m)) for m in declared)


def _disagreement_line(first: tuple[int, RunResult], other: tuple[int, RunResult]) -> str:
    """Note §Q6's diagnostic: one replicate against the first, on final answer and `decidedBy`,
    each `k/n` with Wilson over the items with a final answer in both; failed calls and
    one-sided withholds are counted beside it (plan2 §3.6)."""
    (i, a), (j, b) = first, other
    by_id = {item.itemId: item for item in a.items}
    answer = decided = n = failed = one_sided = 0
    for item in b.items:
        mate = by_id.get(item.itemId)
        if mate is None:
            continue
        if mate.detail.get("failedCall") or item.detail.get("failedCall"):
            failed += 1
        withheld = [x.timing is not None and x.timing.withheldFor is not None for x in (mate, item)]
        if withheld[0] != withheld[1]:
            one_sided += 1
        left, right = _final_answer(mate), _final_answer(item)
        if left is None or right is None:
            continue
        n += 1
        answer += left != right
        decided += mate.detail.get("decidedBy") != item.detail.get("decidedBy")
    return (
        f"- replicate {j} vs replicate {i}: final answer differs on {_rate_cell(answer, n)}, "
        f"decidedBy on {_rate_cell(decided, n)}; beside it, {failed} item(s) with a failed call "
        f"in either and {one_sided} withheld in one replicate only"
    )


# --- the new-ability side: FR-10 (v1.3 §3.6; plan2 §3.6; plan3 §3.6; note §Q4.2, §Q5, §Q6) ----

#: Note §Q1.3: an FR-10 class's design size, printed beside `FR10_CLASS_METHOD_MIN` (plan2 §3.6
#: "method minimum 52 (design 60)").
_FR10_CLASS_DESIGN_N = 60
_FR10_INTERVAL = "per-class 95% Wilson, not simultaneous"
#: A class with no scored eval item has no measured value, so its bar cannot be judged (FR-11a).
#: It never counts as met — FR-10 needs every class (note §Q5.1) — and a `not met` class still
#: decides the replicate, because the conjunction is then false whatever the empty class holds.
_EMPTY_CLASS = "not assessable (no scored eval item in this class)"


def _fr10_class_interval(k: int, n: int, *, design_effect: float) -> tuple[float, float]:
    """Note §Q5.1's interval: Wilson 95% at the design effect, `k/DE` of `n/DE` (note §Q6), the
    class's own per-class value (plan4 §3.9), never the run's `designEffect`."""
    return stats.wilson_interval(k / design_effect, n / design_effect)


def _fr10_class_state(agreed: Sequence[bool]) -> str:
    """One class's own reading, on its measured `k/n` alone (note §Q5.1, FR-11a)."""
    if not agreed:
        return _EMPTY_CLASS
    met = decisions.bar_holds(
        sum(agreed) / len(agreed), decisions.FR10_PER_ANSWER_AGREEMENT, direction="at-least"
    )
    return decision_stats.MET if met else decision_stats.NOT_MET


def _fr10_quality_state(
    agreement: decision_stats.AgreementItems, class_states: Mapping[str, str]
) -> str:
    """Plan2 §3.6 "Bar states" with note §Q4.3 and §Q5.1: a scored eval n below
    `MIN_EVAL_ITEMS` is `not assessable (scored n < 200)` whatever the classes say; otherwise
    any class `not met` is `not met`; otherwise a class with no scored item leaves it not
    assessable; otherwise — every class met — `met`."""
    if agreement.scored_n < decisions.MIN_EVAL_ITEMS:
        return decision_stats.SCORED_N_BELOW_MIN
    if decision_stats.NOT_MET in class_states.values():
        return decision_stats.NOT_MET
    empty = [metric for metric, state in class_states.items() if state == _EMPTY_CLASS]
    if empty:
        return f"not assessable (no scored eval item in {', '.join(empty)})"
    return decision_stats.MET


def _fr10_latency(
    agreement: decision_stats.AgreementItems, *, seed: int
) -> tuple[str, float | None]:
    """Note §Q4.2's figure, `ȳ` over the scored items in `itemId` order with its bootstrap
    interval, and the measured `ȳ` (or `None` when none can be computed, and why)."""
    y = [ms for _, ms in agreement.latency]
    try:
        mean = decision_stats.mean_latency(
            y,
            seed=seed,
            B=decision_stats.BOOTSTRAP_B,
            levels=(stats.LEVEL_CI95_LO, stats.LEVEL_CI95_HI),
        )
    except ValueError as exc:
        return f"FR-10 mean latency: no mean over {len(y)} scored eval item(s) ({exc})", None
    text = (
        f"FR-10 mean latency: {mean.point:.0f} ms [{mean.lo:.0f}, {mean.hi:.0f}] over {len(y)} "
        f"scored eval items (percentile bootstrap 95%, B={mean.B}, seed {mean.seed}); the bar "
        f"is ≤ {decisions.FR10_MEAN_LATENCY_MS_MAX:g} ms on the measured mean"
    )
    return text, mean.point


def _fr10_latency_state(agreement: decision_stats.AgreementItems, mean: float | None) -> str:
    if agreement.scored_n < decisions.MIN_EVAL_ITEMS:
        return decision_stats.SCORED_N_BELOW_MIN
    if mean is None:
        return "not assessable (no mean latency)"
    met = decisions.bar_holds(mean, decisions.FR10_MEAN_LATENCY_MS_MAX, direction="at-most")
    return decision_stats.MET if met else decision_stats.NOT_MET


def _render_fr10_replicate(
    index: int,
    run: RunResult,
    kind: str,
    *,
    blocked: str | None,
    dp: DecisionPointPack,
    pack: PackRef,
) -> _ReplicateBars:
    """Plan2 §3.6's per-replicate block on a new-ability point: the per-class FR-10 lines with
    their Wilson intervals and class minimum, `overLimit` per class (plan3 §3.6), and the
    mean-latency line (note §Q4.2). A single arm: nothing pairs, and `blocked` is a
    comparison-wide state (chat timeout ≠ 180 s, arms on different pack content)."""
    lines = _replicate_head(
        index,
        run,
        kind,
        ("handoffShare", "decisionCallFailures", "handoffCallFailures", "overLimit"),
    )
    quality_name, latency_name = _BAR_NAMES["new-ability"]
    state = decision_stats.LOAD_CHANGED if kind == "assumed" else blocked
    if _aggregate_item_mismatches(run, pack):
        state = _REPLICATE_INCONSISTENT
    different_content = blocked is not None and blocked.startswith(_DIFFERENT_CONTENT)
    agreement: decision_stats.AgreementItems | None = None
    if state != _REPLICATE_INCONSISTENT and not different_content:
        try:
            agreement = decision_stats.agreement_items(
                run.items, role=pack.role, strata=dp.strata
            )
        except ValueError as exc:
            state = f"not assessable (cannot read this replicate's items: {exc})"
    if agreement is None:
        reason = state or blocked or _REPLICATE_INCONSISTENT
        lines += [
            f"- {quality_name} (replicate {index}): {reason}",
            f"- {latency_name} (replicate {index}): {reason}",
        ]
        return _ReplicateBars(lines + [""], reason, reason)

    noun = _SAMPLE_NOUN[unit_kind_for_role(pack.role)]
    class_states = {m: _fr10_class_state(agreed) for m, agreed in agreement.outcomes.items()}
    quality = state or _fr10_quality_state(agreement, class_states)
    latency_text, mean = _fr10_latency(agreement, seed=pack.seed)
    latency = state or _fr10_latency_state(agreement, mean)
    lines += [
        f"- scored eval n: {agreement.scored_n} of the pack's {dp.eval_count} eval {noun}, "
        f"beside FR-11's {decisions.MIN_EVAL_ITEMS}; {_drops_text(agreement.drops)}",
        f"- {quality_name}, within each true answer (each class ≥ "
        f"{decisions.FR10_PER_ANSWER_AGREEMENT:g} on its measured value; {_FR10_INTERVAL}):",
    ]
    # Plan4 §3.9: one family over every class's per-item agreement, empty classes included.
    family = _family_design_effect(agreement.outcomes, agreement.outcome_ids, dp.gen_run_ids)
    lines.append(f"  - {_family_design_effect_line(family, 'classes')}")
    minimum = decision_stats.FR10_CLASS_METHOD_MIN
    # On a replicate whose state is not its classes' (assumed, blocked, scored n < 200), each class
    # line ends with that state, never with a `met` the replicate cannot claim.
    own = state is None and agreement.scored_n >= decisions.MIN_EVAL_ITEMS
    for metric, agreed in agreement.outcomes.items():
        k, n = sum(agreed), len(agreed)
        effect = family.effects[metric]
        if n:
            lo, hi = _fr10_class_interval(k, n, design_effect=effect)
            figures = f"{metric}: {k}/{n} {noun} [{lo:.3f}, {hi:.3f}]"
        else:
            figures = f"{metric}: 0/0 {noun} [—]"
        figures += f"; {_design_effect_text(effect)}"
        thin = " — below it: printed, never refused" if n < minimum else ""
        lines += [
            f"  - {figures} — {class_states[metric] if own else quality}",
            f"    - scored n {n} beside method minimum {minimum} (design "
            f"{_FR10_CLASS_DESIGN_N}){thin}; overLimit {agreement.over_limit[metric]}/{n}",
        ]
    lines += [
        f"- {quality_name} (replicate {index}): {quality}",
        f"- {latency_text} — {latency}",
        "",
    ]
    return _ReplicateBars(lines, quality, latency)


class _AnswerAgreement(NamedTuple):
    """Two new-ability replicates compared item by item (note §Q6's diagnostic; the
    data-scientist's ruling on plan2 §3.6's disagreement line, ledgered in
    `docs/plans/typed-decision-models-coordination.md`)."""

    n: int
    answer_differs: int
    decided_by_differs: int
    no_label_in_both: int
    failures: int
    one_sided_withholds: int
    two_sided_withholds: int


def _answer_agreement(
    first: Sequence[ItemResult], other: Sequence[ItemResult]
) -> _AnswerAgreement:
    """Over the `itemId`s in both replicates, in this order:

    1. withheld (`withheldFor`) on either side → excluded, counted one- or two-sided;
    2. a failed call (`detail.failedCall`) on either side → excluded, counted as a failure. Never
       decided on `answer is None`: message type's fail-closed answer is also `None`;
    3. otherwise in `n`, differing iff the final answers (`detail.answer`) differ. `None` is a
       final answer — no label — so `None` vs `None` agrees, counted as no label in both;
       `decidedBy` is compared over the same `n`."""
    by_id = {item.itemId: item for item in first}
    n = answer = decided = no_label = failures = one_sided = two_sided = 0
    for item in other:
        mate = by_id.get(item.itemId)
        if mate is None:
            continue
        withheld = [x.timing is not None and x.timing.withheldFor is not None for x in (mate, item)]
        if all(withheld):
            two_sided += 1
            continue
        if any(withheld):
            one_sided += 1
            continue
        if mate.detail.get("failedCall") or item.detail.get("failedCall"):
            failures += 1
            continue
        n += 1
        left, right = mate.detail.get("answer"), item.detail.get("answer")
        answer += left != right
        no_label += left is None and right is None
        decided += mate.detail.get("decidedBy") != item.detail.get("decidedBy")
    return _AnswerAgreement(n, answer, decided, no_label, failures, one_sided, two_sided)


def _answer_disagreement_line(
    first: tuple[int, RunResult], other: tuple[int, RunResult], *, role: str
) -> str:
    """Note §Q6's diagnostic on a new-ability point: each `k/n` with Wilson, `no label in both`
    beside it on a choice point, and the excluded items counted."""
    (i, a), (j, b) = first, other
    got = _answer_agreement(a.items, b.items)
    no_label = (
        f"; no label in both: {got.no_label_in_both}"
        if decisions.ANSWER_SPACE_BY_ROLE[role]["type"] == "choice"
        else ""
    )
    return (
        f"- replicate {j} vs replicate {i}: final answer differs on "
        f"{_rate_cell(got.answer_differs, got.n)}, decidedBy on "
        f"{_rate_cell(got.decided_by_differs, got.n)}{no_label}; beside it, failures: "
        f"{got.failures}, one-sided withholds: {got.one_sided_withholds}, two-sided withholds: "
        f"{got.two_sided_withholds}"
    )


def _render_candidate_arm(
    key: str,
    group: ReplicateGroup,
    *,
    bar_kind: str,
    incumbent: RunResult | None,
    blocked: str | None,
    dp: DecisionPointPack,
    pack: PackRef,
    invalid_count: int,
) -> list[str]:
    first = group.runs[0]
    hashes = ", ".join(h[:8] for h in group.earlierCandidates) or "none"
    # K reads valid records only; an invalid one may still have seen this evaluation set, so K is
    # then a lower bound (impl review P18-n1).
    partial = (
        f" (valid records only; {invalid_count} invalid excluded)" if invalid_count else ""
    )
    lines = [
        f"### Candidate `{key}`",
        "",
        f"- fit manifest: {first.fingerprint.get('fitManifestSize')} inputs "
        f"(training/calibration/cut-off); 0 overlap with this pack's {len(dp.strata)} items "
        "(checked at run time, sha256 over canonical input fields)",
        f"- Candidates already run on this evaluation set: {group.k} ({hashes}){partial}",
        f"- fit outcome: {first.fingerprint.get('fitOutcome')} (quality target "
        f"{first.fingerprint.get('fitQualityTarget')})",
        "",
    ]
    quality: list[str] = []
    latency: list[str] = []
    inconsistent: list[int] = []
    for index, (run, kind) in enumerate(zip(group.runs, group.kind, strict=True), 1):
        if bar_kind == "speed-up":
            bars = _render_replicate(
                index, run, kind, group=group, incumbent=incumbent, blocked=blocked, dp=dp,
                pack=pack,
            )
        else:
            bars = _render_fr10_replicate(index, run, kind, blocked=blocked, dp=dp, pack=pack)
        lines += bars.lines
        quality.append(bars.quality)
        latency.append(bars.latency)
        if bars.quality == _REPLICATE_INCONSISTENT:
            inconsistent.append(index)

    lines += ["#### Across the replicates", ""]
    counted = [
        (i, run)
        for i, (run, kind) in enumerate(zip(group.runs, group.kind, strict=True), 1)
        if kind != "assumed"
    ]
    if len(counted) < 2:
        lines.append(
            f"- replicate disagreement: {len(counted)} non-assumed replicate(s), nothing to compare"
        )
    for other in counted[1:]:
        if bar_kind == "speed-up":
            lines.append(_disagreement_line(counted[0], other))
        else:
            lines.append(_answer_disagreement_line(counted[0], other, role=pack.role))
    if inconsistent:
        shown = ", ".join(str(i) for i in inconsistent)
        overall_1 = overall_2 = (
            f"not assessable (replicate {shown}: aggregates disagree with items)"
        )
    else:
        overall_1 = decision_stats.combine_replicates(quality, group.loadCount)
        overall_2 = decision_stats.combine_replicates(latency, group.loadCount)
    over = f"over {len(group.runs)} replicate(s) ({group.loadCount} load(s))"
    name_1, name_2 = _BAR_NAMES[bar_kind]
    lines += [f"- **{name_1}, {over}: {overall_1}**", f"- **{name_2}, {over}: {overall_2}**", ""]
    return lines


def _comparison_block(
    pack: PackRef,
    dp: DecisionPointPack,
    *,
    bar_kind: str | None,
    incumbent: RunResult | None,
    replicates: Mapping[str, ReplicateGroup],
) -> str | None:
    """The comparison-wide `not assessable (…)` state every bar of every replicate takes, or
    `None`. Decided before the bar kind's own section, so FR-10 gets it as FR-1/FR-2 do (impl
    review Pass 18). The incumbent rules are the speed-up kind's alone: FR-10 is single-arm (note
    §Q4.2, §Q5.1; plan2 §3.6 "New-ability points pair nothing"), and a new-ability chat arm is
    U7's reference run, whose results no bar reads. Its timeout and pack content still block,
    which can only withhold a verdict (impl review P23-m2)."""
    timeouts = {r.requestTimeoutSeconds for g in replicates.values() for r in g.runs}
    if incumbent is not None:
        timeouts.add(incumbent.requestTimeoutSeconds)
    blocked = None
    if bar_kind == "speed-up":
        if incumbent is None:
            blocked = _NO_INCUMBENT
        elif _aggregate_item_mismatches(incumbent, pack):
            blocked = _INCUMBENT_INCONSISTENT
    if any(t != decisions.DECISION_POINT_CHAT_TIMEOUT_SECONDS for t in timeouts):
        blocked = _TIMEOUT_REASON
    # Last, so it wins: pairing items across two pack contents states nothing (P18-M1).
    return _pack_content_reason(dp, incumbent, replicates) or blocked


def _render_decision_bar(
    pack: PackRef,
    dp: DecisionPointPack,
    *,
    incumbent: RunResult | None,
    replicates: Mapping[str, ReplicateGroup],
    invalid_count: int = 0,
) -> list[str]:
    """The bar section (v1.3 §3.6, plan2 §3.6): FR-1 and FR-2 on a speed-up point, FR-10 on a
    new-ability one. `incumbent` is the comparison's one chat arm — the incumbent on a speed-up
    point, U7's hand-off-LLM reference run on a new-ability one. `invalid_count` is the pack's
    stored records excluded as invalid, which K cannot read."""
    lines = ["## Requirement bar", "", _BAR_NOTE, ""]
    bar_kind = decisions.BAR_KIND_BY_ROLE.get(pack.role)
    blocked = _comparison_block(
        pack, dp, bar_kind=bar_kind, incumbent=incumbent, replicates=replicates
    )
    if bar_kind not in _BAR_NAMES:
        return lines + [f"Role `{pack.role}` has no requirement bar.", ""]
    if bar_kind == "new-ability":
        reference = (
            f"`{incumbent.armKey}` — run `{incumbent.runId}`" if incumbent is not None else "none"
        )
        lines += [
            f"Reference chat arm: {reference} (FR-10 is single-arm: no incumbent, and no bar "
            "reads this run's results; its timeout and pack content still apply).",
            "",
        ]
    elif incumbent is not None:
        lines += [f"Incumbent: `{incumbent.armKey}` — run `{incumbent.runId}`.", ""]
    else:
        lines += [f"Incumbent: none in this comparison (the pack names `{dp.incumbent}`).", ""]
    if not replicates:
        return lines + ["No candidate arm was named, so there is no bar to state.", ""]
    for key, group in replicates.items():
        lines += _render_candidate_arm(
            key,
            group,
            bar_kind=bar_kind,
            incumbent=incumbent,
            blocked=blocked,
            dp=dp,
            pack=pack,
            invalid_count=invalid_count,
        )
    return lines


def compare_report(
    runs: Sequence[RunResult],
    *,
    pack: PackRef,
    invalid: Sequence[InvalidRecord] = (),
    negative_control: bool = False,
    decision_point: DecisionPointPack | None = None,
    replicates: Mapping[str, ReplicateGroup] | None = None,
) -> str:
    """Render the markdown comparison for one pack. Never ranks across roles or packs.

    On a decision-point pack (`decision_point` given) the labeler caveat is the first result
    line and the bar section follows it (v1.3 §3.5–§3.6). `runs` is then the chat arm alone, or
    empty, so the generic tables below render their fewer-than-two-arms case; candidates arrive
    only as `replicates` (plan2 §3.6)."""
    check_sampling_contract(pack)
    if pack.metrics.headlineMetric is not None and (
        pack.metrics.headlineMetric not in pack.metrics.verdictMetrics
    ):
        raise PackConfigError("headlineMetric is not a member of verdictMetrics")
    if replicates and decision_point is None:
        raise ValueError("replicates are a decision-point comparison's, and need its pack")

    lines: list[str] = [f"# Comparison — {pack.label} ({pack.role})", ""]
    if decision_point is not None:
        lines += [_LABELER_CAVEAT.format(
            labeler=decision_point.labeler, generator=decision_point.generator
        ), ""]
        lines += _render_decision_bar(
            pack,
            decision_point,
            incumbent=runs[0] if runs else None,
            replicates=replicates or {},
            invalid_count=len(invalid),
        )
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
    lucky_cells: list[tuple[str, RunResult, str]] = []
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
                lucky = _lucky_mark(run, metric.name)
                lucky_cells.append((_arm_label(run, arm_names[run.runId]), run, lucky))
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
                    f"{metric.successes}/{metric.n}{lucky} | "
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
    lines += _lucky_footnotes(lucky_cells)
    lines += _render_scorer_counts(
        [(_arm_label(r, arm_names[r.runId]), r) for r in runs]
    )

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
