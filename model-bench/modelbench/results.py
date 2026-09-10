"""Run records: the shapes, and the two enforcement points that make a bad record visible.

Design: `docs/plans/small-model-benchmarking.md` §3.4.5 and §3.5.

* **Write refuses** — `store()` validates the fingerprint and raises. There is no "save anyway"
  flag, and its absence is asserted by the suite against the API surface rather than trusted to a
  comment.
* **Read quarantines** — `load_history()` re-validates every record against **its own**
  `benchSchemaVersion` and returns `(valid, invalid)`. AC-2's real test surface is the *read* side:
  a hand-edited record must be excluded there, not merely rejected on write.

The aggregates are a **closed union of typed dataclasses**, not `dict[str, Any]`. That is what
makes §3.5's structural refusals structural: there is no `overall` field on
`ClassificationAggregates` to hold a pooled 85-item guard accuracy, and no blended percentage field
on `ToolCallAggregates`. A rule you cannot express is a rule you cannot break under deadline
pressure — with an untyped mapping the report would render whatever a pack put in it and the
enforcement would be back to convention.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from modelbench.fingerprint import FieldProblem, Fingerprint
from modelbench.stats import LEVEL_P50, LEVEL_P95, Basis, percentile

#: Plan §3.4.3 — a separate integer, never derived from `benchVersion` and never bumped by a
#: release. It increments only when the required-field set or the on-disk record shape changes in a
#: way a *reader* must branch on. A bump is a deliberate act: a new `REQUIRED_BY_SCHEMA` entry, a
#: `HISTORY.md` line, and a decision about whether a migration is needed.
BENCH_SCHEMA_VERSION: int = 1

Outcome = Literal["pass", "fail", "n_a", "parse_failure"]

# `Basis` is **imported** above, never re-declared here: `RunResult.basis` and
# `stats.resolving_power`'s `basis` parameter are one vocabulary (`-ml` §7.1), and a second
# `Literal` on an import edge this module already crosses is a copy waiting to drift. Python
# enforces neither copy at runtime, so dropping a member from one of the two was invisible in
# both directions and left the whole suite green (impl review Pass 16, P16-1) — the same reason
# `percentile` is imported rather than re-implemented here.


class InvalidFingerprint(ValueError):
    """Raised by `store()`. A result whose environment is not fully recorded is not a result."""


class IncompleteItemRecord(ValueError):
    """An item declares a metric scoreable and records no count for it (review P3-1)."""


class MetricKindError(ValueError):
    """A metric whose instrument is ambiguous (§4 S1e Table F, `-ml` v1.15 §3.2d).

    Raised when a name is present in **both** `ItemResult.counts` and `ItemResult.measures`, and
    when `scored_outcome`/`scored_value` is asked for a metric that lives in the other map. The
    map a name lives in *is* the declaration of which instrument decides it, so an ambiguity is
    refused rather than resolved by code order.
    """


class NonFiniteMeasure(ValueError):
    """A `measures` value that is not finite, refused at construction (`-ml` v1.15 §3.2d).

    One NaN or infinity propagating through a mean and both percentiles would arrive as a
    rendered interval rather than as an error.
    """


# --- metric values ---------------------------------------------------------------------------
# Every rate prints as `k/n = p̂ [lo, hi]` — never a bare percentage, never without its denominator
# (`-ml` §3.2a). Carrying the numerator and denominator in the type is what makes that possible.


@dataclass(frozen=True)
class BinaryMetric:
    """A count and the unit its denominator is in — the second half is not optional.

    `-ml` §4.4's first mandatory consequence is verbatim *"Never print a Wilson interval over a
    turn-pooled count"*, and `report.py` could not honour it because a `BinaryMetric` carried no
    denominator unit: a per-conversation rate and a turn-pooled one were the same type, so the Arms
    table rendered a Wilson interval over both (review M-ML-3). `unit` is what tells them apart —
    `"conversation"`, `"item"`, `"query"` for an analysis-unit rate, `"turn"` or `"call"` for a
    pooled one (§4.2's denominators).

    **No default**, for the reason Rule 2 gives about `design_effect`: the value a forgetful caller
    would want is the analysis unit, and that is exactly the value that licenses the interval.
    """

    name: str
    successes: int
    n: int
    unit: str

    @property
    def rate(self) -> float | None:
        return self.successes / self.n if self.n else None


@dataclass(frozen=True)
class ContinuousMetric:
    name: str
    mean: float
    n: int
    #: The metric's own support, required with no default — `BinaryMetric.unit`'s discipline for
    #: the same reason: `report.py` is generic over packs and cannot know that `mrr` is `[0, 1]`
    #: and `sep_z` unbounded, so the scorer that produced the figure states it (§4 S1e Table F).
    support: tuple[float, float] | None


@dataclass(frozen=True)
class DistributionSummary:
    """`-ml` §5.2's median-and-p10 publication for a per-item continuous figure — neither a mean,
    so `ContinuousMetric` cannot carry them and a bare `float | None` carries neither (§4 S1e
    Table F, plan-gate P6-1)."""

    name: str
    median: float
    p10: float
    n: int
    unit: str
    support: tuple[float, float] | None


MetricValue = BinaryMetric | ContinuousMetric | DistributionSummary


@dataclass(frozen=True)
class TurnPositionRate:
    """One column of `-ml` §4.4's per-position table: n is **conversations**, never turns.

    Its `metric.unit` is therefore the analysis unit, which is what makes a per-position interval
    printable at all."""

    turnIndex: int
    metric: BinaryMetric


# --- per-item records ------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemResult:
    """One scored unit of work.

    `pairingKey` carries the components the pack's `sampling.pairingKey` names, outermost first
    (§3.3). The analysis-unit id is `pairingKey[pack.analysisUnitIndex]` — resolved from the pack,
    never chosen by a caller and with no parameter through which one could.

    `scoreable` records, per conditional count, whether its precondition was met (`-ml` §4.3), so a
    precondition failure can never be laundered into the numerator or silently out of the
    denominator.
    """

    itemId: str
    pairingKey: tuple[str, ...]
    outcome: Outcome
    scoreable: Mapping[str, bool]
    counts: Mapping[str, int]
    latencyMs: float | None
    #: The per-item **continuous** values (`mrr`, `separationRaw`, `separationZ`) — a second map,
    #: never a widened `counts`: widening makes the booleanisation type-legal without making it
    #: wrong, and puts a count that §4.2's denominators count into the same key space as a
    #: measurement they must not (§4 S1e Table F, `-ml` v1.15 §3.2d). Finite floats; the carrier
    #: constrains no domain. **Not** `float | None`: absence stays `scoreable`'s job, so a
    #: measurement of `0.0` stays distinguishable from an unjudged query.
    measures: Mapping[str, float] = field(default_factory=dict)
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Instrument selection is **total**: a metric name lives in `counts` or in `measures`,
        never both, and every `measures` value is finite (§4 S1e Table F, `-ml` v1.15 §3.2d)."""
        overlap = sorted(set(self.counts) & set(self.measures))
        if overlap:
            raise MetricKindError(
                f"item {self.itemId!r} declares {overlap!r} in both `counts` and `measures`; "
                "a metric's instrument has exactly one home"
            )
        for name, value in self.measures.items():
            if not math.isfinite(value):
                raise NonFiniteMeasure(
                    f"item {self.itemId!r} metric {name!r} is not finite: {value!r}"
                )

    def scored_outcome(self, metric: str) -> bool | None:
        """This item's outcome for `metric` — `None` when it carries none (review P3-1).

        **The three states are declared, never inferred**, and this is the one place that decides
        which of them a record is in:

        * `metric` absent from `scoreable` — the item makes no statement about it, so there is no
          outcome. It was read as *scoreable* by default, and the count default then scored it a
          **loss**: an arm carrying no data at all for a metric rendered *"cand is better than
          incumbent … +100.0 pp … p=0.002"* while the §4.3 tally reported `0 unscoreable in both`.
          Absence is not a declaration, in either map.
        * `scoreable[metric] is False` — a declared precondition failure, `None`, and `-ml` §4.3's
          paired corollary counts it as an `asymmetry` finding about the arm that could not
          produce it.
        * `scoreable[metric] is True` — the arm says it scored this item, so the count must be
          there. **A metric declared scoreable and left out of `counts` is refused**, never read
          as a zero: publishing a failure the scorer never observed is §4.3's laundering pointed
          the other way, and the shape of the two maps cannot distinguish it from a scorer that
          simply dropped the key. **S2's scorers must emit a `counts` entry for every metric they
          declare scoreable** — that is what makes this a contract rather than a default.

        **A metric that lives in `measures` raises `MetricKindError` here instead of returning
        `counts[metric] > 0`** (§4 S1e Table F, `-ml` v1.15 §3.2d): that metric's instrument is
        continuous, so its outcome has no boolean to return — the caller wants `scored_value`.
        This is what turns a `measures`-resident metric's booleanisation into a loud failure
        instead of a silent `+100.0 pp` verdict for a metric nobody scored that way.
        """
        if not self.scoreable.get(metric, False):
            return None
        if metric in self.measures:
            raise MetricKindError(
                f"item {self.itemId!r} metric {metric!r} lives in `measures` (a continuous "
                "measurement) and has no boolean outcome; call `scored_value` instead"
            )
        if metric not in self.counts:
            raise IncompleteItemRecord(
                f"item {self.itemId!r} declares {metric!r} scoreable and records no count for it; "
                "an absent count is not a zero, and a scored item must carry its score (-ml §4.3)"
            )
        return self.counts[metric] > 0

    def scored_value(self, metric: str) -> float | None:
        """`scored_outcome`'s sibling over `measures` — the **same three states** (§4 S1e Table F,
        `-ml` v1.15 §3.2d): `metric` absent from `scoreable` is `None`; a declared precondition
        failure (`scoreable[metric] is False`) is `None`, counted into `-ml` §4.3's asymmetry
        tally; `scoreable[metric] is True` with no entry in `measures` is refused
        (`IncompleteItemRecord`), never read as `0.0` — a query nobody judged must not become a
        query that retrieved nothing.
        """
        if not self.scoreable.get(metric, False):
            return None
        if metric not in self.measures:
            raise IncompleteItemRecord(
                f"item {self.itemId!r} declares {metric!r} scoreable and records no measure for "
                "it; an absent measure is not a zero, and a scored item must carry its score "
                "(-ml §4.3)"
            )
        return self.measures[metric]

    def to_dict(self) -> dict[str, Any]:
        return {
            "itemId": self.itemId,
            "pairingKey": list(self.pairingKey),
            "outcome": self.outcome,
            "scoreable": dict(self.scoreable),
            "counts": dict(self.counts),
            "latencyMs": self.latencyMs,
            "measures": dict(self.measures),
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ItemResult":
        return cls(
            itemId=d["itemId"],
            pairingKey=tuple(d["pairingKey"]),
            outcome=d["outcome"],
            scoreable=dict(d.get("scoreable", {})),
            counts=dict(d.get("counts", {})),
            latencyMs=d.get("latencyMs"),
            # A reader's compatibility rule under §3.4.3, not a constructor default: a record
            # written before `measures` existed reads as carrying none (§4 S1e Table F).
            measures=dict(d.get("measures", {})),
            detail=dict(d.get("detail", {})),
        )


# --- aggregates: a closed union, one per role ------------------------------------------------


@dataclass(frozen=True)
class RetrievalAggregates:
    kind: Literal["retrieval"] = "retrieval"
    recallAtK: tuple[BinaryMetric, ...] = ()
    mrr: ContinuousMetric | None = None
    precisionAt1: BinaryMetric | None = None
    #: `-ml` §5.2 publishes a median and a p10 for `sep_raw`/`sep_z`, neither a mean — a
    #: `DistributionSummary`, not the bare `float | None` these carried before (§4 S1e Table F).
    separationRaw: DistributionSummary | None = None
    separationZ: DistributionSummary | None = None

    def named_metrics(self) -> tuple[MetricValue, ...]:
        found = [*self.recallAtK]
        if self.precisionAt1 is not None:
            found.append(self.precisionAt1)
        if self.mrr is not None:
            found.append(self.mrr)
        # This one addition is what makes `sep_z` reach a table at all — today it reaches none,
        # whatever the scorer computes (§4 S1e Table F).
        if self.separationRaw is not None:
            found.append(self.separationRaw)
        if self.separationZ is not None:
            found.append(self.separationZ)
        return tuple(found)


@dataclass(frozen=True)
class ToolCallAggregates:
    """No blended "tool-calling accuracy" field exists here, deliberately (§3.5, AC-1)."""

    kind: Literal["toolcalls"] = "toolcalls"
    cleanThroughTurn: BinaryMetric | None = None
    perTurnPosition: tuple[TurnPositionRate, ...] = ()
    funnel: tuple[BinaryMetric, ...] = ()
    restraint: BinaryMetric | None = None
    hazard: tuple[BinaryMetric, ...] = ()

    def named_metrics(self) -> tuple[MetricValue, ...]:
        found = []
        if self.cleanThroughTurn is not None:
            found.append(self.cleanThroughTurn)
        if self.restraint is not None:
            found.append(self.restraint)
        return tuple([*found, *self.funnel, *self.hazard])


@dataclass(frozen=True)
class ClassificationAggregates:
    """Per-class rates only. There is no pooled-accuracy field to print (§3.8.2)."""

    perClass: tuple[BinaryMetric, ...] = ()
    parseFailures: int = 0
    n: int = 0
    kind: Literal["classification"] = "classification"

    def named_metrics(self) -> tuple[MetricValue, ...]:
        return self.perClass


@dataclass(frozen=True)
class ExtractionAggregates:
    kind: Literal["extraction"] = "extraction"
    exactMatch: BinaryMetric | None = None
    byShape: tuple[BinaryMetric, ...] = ()
    parseFailures: int = 0

    def named_metrics(self) -> tuple[MetricValue, ...]:
        found = [self.exactMatch] if self.exactMatch is not None else []
        return tuple([*found, *self.byShape])


@dataclass(frozen=True)
class GroundingAggregates:
    kind: Literal["grounding"] = "grounding"
    checklistPass: BinaryMetric | None = None
    perCheck: tuple[BinaryMetric, ...] = ()
    parseFailures: int = 0

    def named_metrics(self) -> tuple[MetricValue, ...]:
        found = [self.checklistPass] if self.checklistPass is not None else []
        return tuple([*found, *self.perCheck])


Aggregates = (
    RetrievalAggregates
    | ToolCallAggregates
    | ClassificationAggregates
    | ExtractionAggregates
    | GroundingAggregates
)

_AGGREGATE_BY_KIND: dict[str, type] = {
    "retrieval": RetrievalAggregates,
    "toolcalls": ToolCallAggregates,
    "classification": ClassificationAggregates,
    "extraction": ExtractionAggregates,
    "grounding": GroundingAggregates,
}


# --- the run record --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """One model (or one deterministic arm) × one pack × the pack's declared sampling.

    `designEffect` and `basis` are recorded per run because `-ml` §3.4 Rule 4 decides *which
    instrument may decide* from them, and a report cannot recompute either after the fact: the
    basis comes from whether the determinism probe ran and agreed (plan §5 test 12b), which only
    the runner knows. `basis` is fail-safe — a probe that did not run yields `"assumed"`, which
    moves the decision off McNemar and onto the cluster bootstrap.
    """

    runId: str
    sessionId: str | None
    role: str
    armKind: str
    fingerprint: Fingerprint
    items: tuple[ItemResult, ...]
    aggregates: Aggregates
    #: Both are **required, with no dataclass default** (plan v1.5 §3.5, review M-2 / m-ML-3).
    #: `designEffect = 1.0` is the anti-conservative value, so a default here rebuilds gate B-1's
    #: "default by omission" at the seam S2's runner constructs — and it makes DC-5's clause
    #: "report.py refuses to render one when the required input is absent" true only vacuously,
    #: because with a default the input can never *be* absent. The legacy fallback lives in
    #: `from_dict`, where it means "a record written before these fields existed" (§3.4.3).
    designEffect: float
    basis: Basis

    def to_dict(self) -> dict[str, Any]:
        return {
            "runId": self.runId,
            "sessionId": self.sessionId,
            "role": self.role,
            "armKind": self.armKind,
            "fingerprint": self.fingerprint.to_dict(),
            "items": [i.to_dict() for i in self.items],
            "aggregates": _aggregates_to_dict(self.aggregates),
            "designEffect": self.designEffect,
            "basis": self.basis,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunResult":
        return cls(
            runId=d["runId"],
            sessionId=d.get("sessionId"),
            role=d["role"],
            armKind=d["armKind"],
            fingerprint=Fingerprint.from_dict(d["fingerprint"]),
            items=tuple(ItemResult.from_dict(i) for i in d.get("items", [])),
            # No `.get` default: fabricating an empty `ClassificationAggregates` for a record
            # that has no aggregates block repairs the one absence this module exists to report
            # (review n-3). A `KeyError` here reaches `load_history` as `unparseable` when the
            # record declares a schema this build knows, and as `unknown_schema` when it declares
            # a later one — that call is the reader's, made from the envelope, and not this
            # function's to anticipate (review P14-4).
            aggregates=_aggregates_from_dict(d["aggregates"]),
            designEffect=d.get("designEffect", 1.0),
            basis=d.get("basis", "assumed"),
        )

    @property
    def modelKey(self) -> str:
        """The literal LM Studio id for a model arm; the arm id for a deterministic one."""
        if self.armKind == "deterministic":
            return str(self.fingerprint.get("armId", ""))
        return str(self.fingerprint.get("modelKey", ""))


@dataclass(frozen=True)
class InvalidRecord:
    """A stored record that may not enter a comparison, and why (AC-2)."""

    path: Path
    runId: str | None
    benchSchemaVersion: int | None
    problems: list[FieldProblem]
    reason: Literal["field", "unknown_schema", "unparseable"]


# --- (de)serialization -----------------------------------------------------------------------


def _metric_to_dict(m: MetricValue) -> dict[str, Any]:
    if isinstance(m, BinaryMetric):
        return {
            "type": "binary", "name": m.name, "successes": m.successes, "n": m.n, "unit": m.unit,
        }
    if isinstance(m, DistributionSummary):
        # A third value of the same "type" discriminator "binary" and "continuous" already
        # carry, never a "continuous" with extra keys — a reader that took one for the other
        # would read a median as a mean (§4 S1e Table F, plan-gate P7-1).
        return {
            "type": "distribution",
            "name": m.name,
            "median": m.median,
            "p10": m.p10,
            "n": m.n,
            "unit": m.unit,
            "support": list(m.support) if m.support is not None else None,
        }
    return {
        "type": "continuous",
        "name": m.name,
        "mean": m.mean,
        "n": m.n,
        "support": list(m.support) if m.support is not None else None,
    }


def _decode_support(value: Any) -> tuple[float, float] | None:
    return tuple(value) if value is not None else None


def _decode_binary(d: Mapping[str, Any]) -> BinaryMetric:
    # No `.get` fallback: a stored count whose denominator unit is unknown is exactly the
    # record §4.4 says must not be given an interval, and guessing one restores the defect.
    return BinaryMetric(name=d["name"], successes=d["successes"], n=d["n"], unit=d["unit"])


def _decode_continuous(d: Mapping[str, Any]) -> ContinuousMetric:
    # `support` has no `.get` fallback either, for the same reason (§4 S1e Table F): it is not
    # recomputed by a reader, so a stored metric dict with no `support` key raises rather than
    # defaulting.
    return ContinuousMetric(
        name=d["name"], mean=d["mean"], n=d["n"], support=_decode_support(d["support"])
    )


def _decode_distribution(d: Mapping[str, Any]) -> DistributionSummary:
    return DistributionSummary(
        name=d["name"],
        median=d["median"],
        p10=d["p10"],
        n=d["n"],
        unit=d["unit"],
        support=_decode_support(d["support"]),
    )


#: The tag set's **one home** (§4 S1e Table F, plan-gate P7-1): `_metric_from_dict` dispatches on
#: it and `_decode` gates on it, so the two functions cannot disagree about how many metric types
#: exist (§7 rule 4).
_METRIC_DECODERS: Mapping[str, Callable[[Mapping[str, Any]], MetricValue]] = {
    "binary": _decode_binary,
    "continuous": _decode_continuous,
    "distribution": _decode_distribution,
}


def _metric_from_dict(d: Mapping[str, Any]) -> MetricValue:
    try:
        decoder = _METRIC_DECODERS[d["type"]]
    except KeyError:
        # An unrecognised tag raises rather than falling through as a raw `dict` — a dict tagged
        # with a `"type"` this build does not know is a record written by a build that knew a
        # metric type this one does not (§4 S1e Table F, plan-gate P7-1). Raising is all this
        # function decides: whether that record reads as a later build's or as a damaged one is
        # `load_history`'s call, taken from the envelope before the body is decoded, exactly as
        # for a `KeyError` in `from_dict` (review P14-4).
        raise ValueError(f"unrecognised metric type {d.get('type')!r}") from None
    return decoder(d)


def _encode(value: Any) -> Any:
    if isinstance(value, (BinaryMetric, ContinuousMetric, DistributionSummary)):
        return _metric_to_dict(value)
    if isinstance(value, TurnPositionRate):
        return {"turnIndex": value.turnIndex, "metric": _metric_to_dict(value.metric)}
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode(v) for v in value)
    if isinstance(value, dict) and "type" in value:
        return _metric_from_dict(value)
    if isinstance(value, dict) and "turnIndex" in value:
        return TurnPositionRate(
            turnIndex=value["turnIndex"], metric=_metric_from_dict(value["metric"])
        )
    return value


def _aggregates_to_dict(agg: Aggregates) -> dict[str, Any]:
    return {k: _encode(v) for k, v in vars(agg).items()}


def _aggregates_from_dict(d: Mapping[str, Any]) -> Aggregates:
    cls = _AGGREGATE_BY_KIND[d["kind"]]
    return cls(**{k: _decode(v) for k, v in d.items() if k != "kind"})


# --- storage ---------------------------------------------------------------------------------


def runs_dir(root: Path) -> Path:
    return Path(root) / "results" / "runs"


def store(run: RunResult, root: Path) -> Path:
    """Write one run record. Raises `InvalidFingerprint` on any field problem.

    There is deliberately no `force`/`allow_invalid` parameter (§3.4.5 point 1): the absence of a
    bypass is the guarantee, so it is a property of the signature rather than of a docstring.
    """
    problems = run.fingerprint.validate()
    if problems:
        detail = ", ".join(f"{p.field} ({p.reason})" for p in problems)
        raise InvalidFingerprint(f"run {run.runId} has an incomplete fingerprint: {detail}")
    # Plan §3.5 specifies `modelSlug` sanitisation precisely because real model keys contain `/`
    # (`qwen/qwen3-4b-2507`), and the slugging is S2's runner. Until then an unslugged id raised a
    # bare `FileNotFoundError` from `pathlib` — loud, but not a named reason — and a segment that
    # happened to name an existing directory would have written outside `runs/` (review m-7).
    # `{"", ".."}`, not `{"", ".", ".."}`: `Path(".").name` is `""`, so `"."` is already caught
    # by the first clause — but **`Path("..").name` is `".."`**, so `".."` is not, and dropping it
    # would write `results/runs/..json`. Review P2-4 called two-thirds of the set unreachable;
    # measured here, one-third is (see `test_store_refuses_an_empty_run_id`). An unreachable guard
    # reads as a case someone thought about, which is worse than no guard at all — so the one that
    # is unreachable goes and the two that are not stay.
    if run.runId != Path(run.runId).name or run.runId in {"", ".."}:
        raise ValueError(
            f"runId {run.runId!r} is not a bare filename; a record is written to "
            "results/runs/<runId>.json, so the id must already carry plan §3.5's slug"
        )
    target = runs_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{run.runId}.json"
    path.write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _item_problems(run: RunResult) -> list[FieldProblem]:
    """The read-time half of `scored_outcome`'s contract (review P4-5).

    `scored_outcome` **raises** for an item that declares a metric scoreable and records no count,
    and that refusal is right: an absent count is not a zero. But `store()` accepts such a record
    (its validation is fingerprint-only) and `load_history` accepted it too, so the refusal landed
    at *report* time — from `report.py`, whose only CLI caller catches `PackConfigError` and
    nothing else. Measured end-to-end: one bad item in one of three otherwise-valid records aborted
    `compare` with an uncaught traceback, **exit 1** (outside §3.6a's closed `0/2/3/4/5`), **no
    report written**, and the valid arms lost with it.

    AC-2's mechanism is *excluded on read **and named***, so the record is quarantined here like
    any other field failure, with the item and the metric in the field path — which is also the
    half the exception message could not supply, since an `ItemResult` does not know its run.

    **The truthiness test is `scored_outcome`'s own**, not `is True`: that function reads
    `scoreable.get(metric, False)`, so a truthy non-bool declaration demands a count there and must
    demand one here, or the two disagree about which records are readable.
    """
    found: list[FieldProblem] = []
    for item in run.items:
        for metric, declared in item.scoreable.items():
            if declared and metric not in item.counts:
                found.append(
                    FieldProblem(field=f"items[{item.itemId}].counts.{metric}", reason="absent")
                )
    return found


def load_history(root: Path, *, packId: str) -> tuple[list[RunResult], list[InvalidRecord]]:
    """Read every stored run for **one** pack, re-validating each against its own schema.

    There is no API to load across packs, and that is how FR-20 (no cross-role aggregate) is
    enforced structurally rather than by convention (§3.5).

    Each record is classified from its **envelope** — `runId`, and the fingerprint's `packId` and
    `benchSchemaVersion` — *before* its body is decoded, so a record left by a later build is
    diagnosed by the schema it declares rather than by whether this build happens to understand
    its aggregates. Decoding first made the pack filter and `unknown_schema` below reachable only
    for the one future record whose shape had not changed, which is the future record that would
    not have needed the version bump (review P14-4).
    """
    from modelbench.fingerprint import REQUIRED_BY_SCHEMA

    valid: list[RunResult] = []
    invalid: list[InvalidRecord] = []
    directory = runs_dir(root)
    if not directory.is_dir():
        return valid, invalid

    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            # The **envelope** — the record's identity, and the two fields that decide whether
            # this build may read the body at all — is parsed first and *separately from the
            # body*, because the body is the half a later build changes. `BENCH_SCHEMA_VERSION`'s
            # docstring says the integer increments when the on-disk shape changes in a way a
            # reader must branch on, so the ordinary future record is one this build cannot
            # decode — and decoding it first put it on `unparseable` before either branch below
            # ran, which is P14-4's consequence with no drift of `_AGGREGATE_BY_KIND` required.
            # It is read through `Fingerprint.from_dict` rather than off `raw` so that how a
            # stored fingerprint is read keeps one home.
            runId = raw["runId"]
            fingerprint = Fingerprint.from_dict(raw["fingerprint"])
        except Exception:
            # A truncated or hand-mangled file cannot declare its pack, so it is surfaced rather
            # than silently skipped: an unreadable record is a finding, not an absence. This is
            # now the *only* record that answers `None` for both fields, and that is the claim
            # the two values make: nothing about this file was legible, not even its name.
            invalid.append(
                InvalidRecord(
                    path=path,
                    runId=None,
                    benchSchemaVersion=None,
                    problems=[],
                    reason="unparseable",
                )
            )
            continue

        # The pack filter may only drop a record that **says** it belongs to another pack. A
        # `packId` that is absent, null or blank is a required-field failure, and skipping it here
        # put the record in neither returned list — the comparison quietly lost an arm and the
        # report said nothing, against AC-2's "excluded on read *and named*" (review M-1).
        #
        # The filter applies to an unknown schema too, and — since the envelope is parsed on its
        # own — to a record whose body this build cannot decode at all: in both cases the
        # `packId` is right there and readable, so another pack's future record is not surfaced
        # as this pack's exclusion (review m-1). It stays **off** the `unparseable` above,
        # because a file that could not even yield an envelope genuinely cannot declare its pack.
        schema = fingerprint.benchSchemaVersion
        declared = fingerprint.get("packId")
        if isinstance(declared, str) and declared and declared != packId:
            continue
        if not isinstance(schema, int) or isinstance(schema, bool) or (
            schema not in REQUIRED_BY_SCHEMA
        ):
            invalid.append(
                InvalidRecord(
                    path=path,
                    runId=runId,
                    # `not isinstance(schema, bool)` as well, because `True` is an `int` and
                    # this field is typed `int | None`: a quarantined bool would otherwise be
                    # reported as the schema version `True` (review P2-2).
                    benchSchemaVersion=(
                        schema
                        if isinstance(schema, int) and not isinstance(schema, bool)
                        else None
                    ),
                    problems=[FieldProblem(field="benchSchemaVersion", reason="unknown")],
                    reason="unknown_schema",
                )
            )
            continue

        try:
            run = RunResult.from_dict(raw)
        except Exception:
            # A record that claims a schema this build **does** know and still will not decode is
            # not a record from the future: it is damaged, or its writer changed the shape without
            # bumping the version. `unknown_schema` would launder that into a tooling-version
            # excuse, so it stays `unparseable` — but its envelope was legible, so AC-2's line
            # names the run and the schema it claimed instead of a bare filename.
            invalid.append(
                InvalidRecord(
                    path=path,
                    runId=runId,
                    benchSchemaVersion=schema,
                    problems=[],
                    reason="unparseable",
                )
            )
            continue

        # Both halves are `field` failures and share one exclusion path: a record whose
        # environment is incomplete and one whose items are internally self-contradictory are
        # equally unreadable, and AC-2's block already says which fields failed (review P4-5).
        problems = run.fingerprint.validate() + _item_problems(run)
        if problems:
            invalid.append(
                InvalidRecord(
                    path=path,
                    runId=run.runId,
                    benchSchemaVersion=schema,
                    problems=problems,
                    reason="field",
                )
            )
        else:
            valid.append(run)
    return valid, invalid


INDEX_COLUMNS = (
    "runId",
    "date",
    "role",
    "packId",
    "packVersion",
    "packContentHash8",
    "modelKey",
    "quantization",
    "armKind",
    "n",
    "headlineMetrics",
    "latencyMsP50",
    "latencyMsP95",
    "valid",
)


def _metric_cell(m: MetricValue) -> str:
    if isinstance(m, BinaryMetric):
        return f"{m.name}={m.successes}/{m.n}"
    if isinstance(m, DistributionSummary):
        # The `p50` label is not decoration: the cell's continuous form is otherwise a bare
        # number, and a median printed like a mean is §3.5's defect one column over. `p10` is
        # not in this cell — the index is a per-run locator and the Arms table is where a
        # distribution prints (§4 S1e Table F).
        return f"{m.name}=p50 {m.median:.4f}"
    return f"{m.name}={m.mean:.4f}"


def _index_row(run: RunResult, valid: bool) -> dict[str, Any]:
    latencies = [i.latencyMs for i in run.items if i.latencyMs is not None]
    metrics = "; ".join(_metric_cell(m) for m in run.aggregates.named_metrics())
    return {
        "runId": run.runId,
        "date": str(run.fingerprint.get("startedAt", ""))[:10],
        "role": run.role,
        "packId": run.fingerprint.get("packId", ""),
        "packVersion": run.fingerprint.get("packVersion", ""),
        "packContentHash8": str(run.fingerprint.get("packContentHash", ""))[:8],
        "modelKey": run.modelKey,
        "quantization": run.fingerprint.get("quantization", ""),
        "armKind": run.armKind,
        "n": len(run.items),
        "headlineMetrics": metrics,
        # `-ml` §11.10(3): `stats.percentile` is the ONLY percentile in the package and this
        # module imports *that object* — a second private copy is what let this very row report
        # `latencyMsP95` at the 50th percentile and stay green (review M27). The emptiness test
        # stays here rather than inside `percentile`, which raises: whether a latency figure
        # exists at all is a decision about the run, and on a deterministic arm there are no
        # timings to take a percentile of. **Both cells still bypass `-ml` §11's two floors**, and
        # closing that is §4 S2's — every latency cell is copied from the run's own `LatencyBlock`
        # (§3.5), which does not exist until the runner builds it.
        "latencyMsP50": percentile(latencies, level=LEVEL_P50) if latencies else None,
        "latencyMsP95": percentile(latencies, level=LEVEL_P95) if latencies else None,
        "valid": "yes" if valid else "no",
    }


def rebuild_index(root: Path) -> Path:
    """Regenerate `results/index.csv` from `results/runs/`.

    Derived and fully regenerable, so it is never a second source of truth to keep honest (§3.5).
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir(root).glob("*.json")) if runs_dir(root).is_dir() else []:
        try:
            run = RunResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        rows.append(_index_row(run, valid=not run.fingerprint.validate()))
    out = Path(root) / "results" / "index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return out


def models_with_stored_results(
    root: Path, *, packId: str | None = None, role: str | None = None
) -> list[str]:
    """FR-17a — models with stored results. Filters to `armKind == "model"` (§3.4.1), so a BM25
    reference arm can never be offered as a reference *model*."""
    seen: dict[str, None] = {}
    for path in sorted(runs_dir(root).glob("*.json")) if runs_dir(root).is_dir() else []:
        try:
            run = RunResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if run.armKind != "model":
            continue
        if packId is not None and run.fingerprint.get("packId") != packId:
            continue
        if role is not None and run.role != role:
            continue
        seen.setdefault(run.modelKey, None)
    return list(seen)
