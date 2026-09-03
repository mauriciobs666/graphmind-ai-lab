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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from modelbench.fingerprint import FieldProblem, Fingerprint

#: Plan §3.4.3 — a separate integer, never derived from `benchVersion` and never bumped by a
#: release. It increments only when the required-field set or the on-disk record shape changes in a
#: way a *reader* must branch on. A bump is a deliberate act: a new `REQUIRED_BY_SCHEMA` entry, a
#: `HISTORY.md` line, and a decision about whether a migration is needed.
BENCH_SCHEMA_VERSION: int = 1

Outcome = Literal["pass", "fail", "n_a", "parse_failure"]
Basis = Literal["by-construction", "measured", "assumed"]


class InvalidFingerprint(ValueError):
    """Raised by `store()`. A result whose environment is not fully recorded is not a result."""


class IncompleteItemRecord(ValueError):
    """An item declares a metric scoreable and records no count for it (review P3-1)."""


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


MetricValue = BinaryMetric | ContinuousMetric


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
    detail: Mapping[str, Any] = field(default_factory=dict)

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
        """
        if not self.scoreable.get(metric, False):
            return None
        if metric not in self.counts:
            raise IncompleteItemRecord(
                f"item {self.itemId!r} declares {metric!r} scoreable and records no count for it; "
                "an absent count is not a zero, and a scored item must carry its score (-ml §4.3)"
            )
        return self.counts[metric] > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "itemId": self.itemId,
            "pairingKey": list(self.pairingKey),
            "outcome": self.outcome,
            "scoreable": dict(self.scoreable),
            "counts": dict(self.counts),
            "latencyMs": self.latencyMs,
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
            detail=dict(d.get("detail", {})),
        )


# --- aggregates: a closed union, one per role ------------------------------------------------


@dataclass(frozen=True)
class RetrievalAggregates:
    kind: Literal["retrieval"] = "retrieval"
    recallAtK: tuple[BinaryMetric, ...] = ()
    mrr: ContinuousMetric | None = None
    precisionAt1: BinaryMetric | None = None
    separationRaw: float | None = None
    separationZ: float | None = None

    def named_metrics(self) -> tuple[MetricValue, ...]:
        found = [*self.recallAtK]
        if self.precisionAt1 is not None:
            found.append(self.precisionAt1)
        if self.mrr is not None:
            found.append(self.mrr)
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
            # (review n-3). A `KeyError` here surfaces as `unparseable` in `load_history`.
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
    return {"type": "continuous", "name": m.name, "mean": m.mean, "n": m.n}


def _metric_from_dict(d: Mapping[str, Any]) -> MetricValue:
    if d["type"] == "binary":
        # No `.get` fallback: a stored count whose denominator unit is unknown is exactly the
        # record §4.4 says must not be given an interval, and guessing one restores the defect.
        return BinaryMetric(
            name=d["name"], successes=d["successes"], n=d["n"], unit=d["unit"]
        )
    return ContinuousMetric(name=d["name"], mean=d["mean"], n=d["n"])


def _encode(value: Any) -> Any:
    if isinstance(value, (BinaryMetric, ContinuousMetric)):
        return _metric_to_dict(value)
    if isinstance(value, TurnPositionRate):
        return {"turnIndex": value.turnIndex, "metric": _metric_to_dict(value.metric)}
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode(v) for v in value)
    if isinstance(value, dict) and value.get("type") in {"binary", "continuous"}:
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


def load_history(root: Path, *, packId: str) -> tuple[list[RunResult], list[InvalidRecord]]:
    """Read every stored run for **one** pack, re-validating each against its own schema.

    There is no API to load across packs, and that is how FR-20 (no cross-role aggregate) is
    enforced structurally rather than by convention (§3.5).
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
            run = RunResult.from_dict(raw)
        except Exception:
            # A truncated or hand-mangled file cannot declare its pack, so it is surfaced rather
            # than silently skipped: an unreadable record is a finding, not an absence.
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
        # The filter now applies to an unknown schema too, whose `packId` is right there and
        # readable, so another pack's future-schema record is no longer surfaced as this pack's
        # exclusion — but it stays **off** `unparseable` above, because a truncated file genuinely
        # cannot declare its pack (review m-1).
        schema = run.fingerprint.benchSchemaVersion
        declared = run.fingerprint.get("packId")
        if isinstance(declared, str) and declared and declared != packId:
            continue
        if not isinstance(schema, int) or isinstance(schema, bool) or (
            schema not in REQUIRED_BY_SCHEMA
        ):
            invalid.append(
                InvalidRecord(
                    path=path,
                    runId=run.runId,
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
        problems = run.fingerprint.validate()
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


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _index_row(run: RunResult, valid: bool) -> dict[str, Any]:
    latencies = [i.latencyMs for i in run.items if i.latencyMs is not None]
    metrics = "; ".join(
        f"{m.name}={m.successes}/{m.n}" if isinstance(m, BinaryMetric) else f"{m.name}={m.mean:.4f}"
        for m in run.aggregates.named_metrics()
    )
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
        "latencyMsP50": _percentile(latencies, 50),
        "latencyMsP95": _percentile(latencies, 95),
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
