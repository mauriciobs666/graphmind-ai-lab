"""The fit's calls: residency, abort record, dispositions, the call collector and the call
summaries (`docs/plans/typed-decision-models4.md` §3.11, §4.4 U6f.8a, §5 row U6f.8a;
`docs/plans/typed-decision-models-fit-ml.md` §4).

The collector makes one decision call per (item, text) and one hand-off call per item, the client
injected, and returns the shared `ScoredCall`s both candidates' ladders read:

- `OverLimit` is an `over-limit` call (its `wallClockMs` is the elapsed time);
- any other client failure aborts with one `AbortRecord` file and nothing else, unless the call is
  exactly a `Disposition`'s `(itemId, callKind, textId, failureKind, status)`: then it is a `failed`
  call (answer `None`, elapsed time from the injected clock) and the run continues;
- a hand-off parse failure is an `ok` call carrying the parser's fallback verdict, counted;
- no retry.

`ScoredCall.answer` of an `ok` call: a predicate decision's `p`; a choice decision's tuple
`(choice, confidence)`; a hand-off's `parse_*_handoff` verdict (a bool for the guard and join-in,
the label or `None` for message type).

Stdlib + this package only (FR-8). `fit.py` (U6f.8b) calls `check_start`, `collect_calls`,
`summarize_calls` and `check_end`; `exclusion.py` (U6f.8c) builds `Disposition`s.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from decisionmodels.fitstats import CALL_KINDS, ScoredCall
from decisionmodels.handoff import (
    HANDOFF_CHAT_TIMEOUT_SECONDS,
    HANDOFF_MAX_TOKENS,
    HANDOFF_TEMPERATURE,
    handoff_messages,
    parse_guard_handoff,
    parse_join_in_handoff,
    parse_message_type_handoff,
)
from decisionmodels.lmstudio import (
    ChatResult,
    DecisionResult,
    LMStudioCallFailed,
    LMStudioCallTimeout,
    LMStudioError,
    OverLimit,
    ResidentModel,
    failure_kind,
)
from decisionmodels.points import ROLES, load_point
from decisionmodels.questions import Question

ABORT_SCHEMA = "decision-models.fit-abort/1"
FAILURE_KINDS = ("timeout", "connection", "http", "malformed")
WILSON_Z = 1.96

_RESIDENCY_READ_ERRORS = (LMStudioError, TimeoutError)


class Client(Protocol):
    """The slice of `lmstudio.LMStudio` the fit's calls use."""

    def residency(self) -> list[ResidentModel]: ...

    def decide(
        self, input: str, questions: Sequence[Mapping[str, Any]], *, model: str, timeout_s: float
    ) -> DecisionResult | OverLimit: ...

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> ChatResult: ...


class ResidencyRefused(RuntimeError):
    """A residency check refused the fit: `phase` is `"start"` or `"end"`. Nothing is written."""

    def __init__(self, phase: str, message: str) -> None:
        super().__init__(message)
        self.phase = phase


class ExclusionDidNotFire(RuntimeError):
    """A `Disposition`'s call did not fail in this run (it returned `ok` or over-limit, or was
    never made: a ghost item or text): the disposition's premise (the failure is the item's
    property) is falsified. Raised at that call, or after the
    loops for a call never made; nothing is written (fit note §4, "The exclusion must fire")."""

    def __init__(self, disposition: Disposition, outcome: str) -> None:
        super().__init__(
            f"exclusion of {disposition.callKind} call for item {disposition.itemId!r} "
            f"(text {disposition.textId!r}) did not fire: the call was {outcome}"
        )
        self.disposition = disposition
        self.outcome = outcome


class FitAborted(RuntimeError):
    """A call failed outside every disposition: `record` was written to `path`, nothing else."""

    def __init__(self, record: AbortRecord, path: Path) -> None:
        super().__init__(
            f"fit aborted: {record.callKind} call for item {record.itemId!r} failed "
            f"({record.failureKind}); record {path.name}"
        )
        self.record = record
        self.path = path


@dataclass(frozen=True)
class FitInputs:
    """The run's inputs, recorded in every abort record (§3.11)."""

    role: str
    datasetVersion: str  # noqa: N815
    questionsSha256: str  # noqa: N815
    handoffSha256: str  # noqa: N815
    decisionModel: str  # noqa: N815
    handoffModel: str  # noqa: N815
    decisionTimeoutSeconds: float  # noqa: N815


@dataclass(frozen=True)
class AbortRecord:
    """The one file an aborted fit writes (§3.11). No item text, no end snapshot."""

    schema: str
    runId: str  # noqa: N815
    itemId: str  # noqa: N815
    callKind: str  # noqa: N815
    textId: str | None  # noqa: N815
    failureKind: str  # noqa: N815
    status: int | None
    elapsedSeconds: float  # noqa: N815
    inputs: FitInputs
    residencyStart: tuple[ResidentModel, ...]  # noqa: N815
    residencyNow: tuple[ResidentModel, ...] | None  # noqa: N815

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Disposition:
    """A recorded item disposition (§3.11): the exemption is exactly `key`, `status` included."""

    itemId: str  # noqa: N815
    callKind: str  # noqa: N815
    textId: str | None  # noqa: N815
    failureKind: str  # noqa: N815
    status: int | None
    reason: str
    recordFiles: tuple[str, str]  # noqa: N815

    def __post_init__(self) -> None:
        if self.callKind not in CALL_KINDS:
            raise ValueError(f"Disposition.callKind must be one of {CALL_KINDS}")
        if self.failureKind not in FAILURE_KINDS:
            raise ValueError(f"Disposition.failureKind must be one of {FAILURE_KINDS}")
        if (self.textId is None) != (self.callKind == "handoff"):
            raise ValueError("Disposition.textId is None iff callKind is handoff")

    @property
    def key(self) -> tuple[str, str, str | None, str, int | None]:
        return (self.itemId, self.callKind, self.textId, self.failureKind, self.status)


@dataclass(frozen=True)
class CalibrationItem:
    """One calibration item: the decision call's `input` and the hand-off call's rendering."""

    itemId: str  # noqa: N815
    decisionInput: str  # noqa: N815
    handoffInput: str  # noqa: N815


@dataclass(frozen=True)
class CollectedCalls:
    """The collector's result: the shared `ScoredCall`s and the hand-off parse-failure count."""

    calls: tuple[ScoredCall, ...]
    parseFailures: int  # noqa: N815
    #: `(itemId, textId)` -> `max(probabilities)` of each `ok` choice decision call (for
    #: `TextScore.maxProbAuc`); absent for every other call, so empty on a predicate point, and
    #: absent for an `ok` choice call whose `probabilities` list is empty (no maximum exists).
    maxProbabilities: Mapping[tuple[str, str], float]  # noqa: N815


@dataclass(frozen=True)
class CallSummary:
    """One call kind's report fields (§3.11 "Fit report"; fit note §4); never gating."""

    callKind: str  # noqa: N815
    n: int
    k: int
    dispositionItemIds: tuple[str, ...]  # noqa: N815
    medianSeconds: float  # noqa: N815
    maxSeconds: float  # noqa: N815
    wilsonUpper95: float  # noqa: N815
    halfBudgetCount: int  # noqa: N815
    parseFailures: int = field(default=0)  # noqa: N815


# --- residency ---


def _read_residency(client: Client, phase: str) -> tuple[ResidentModel, ...]:
    try:
        return tuple(client.residency())
    except _RESIDENCY_READ_ERRORS as exc:
        raise ResidencyRefused(
            phase, f"residency read at {phase} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _require_both(
    snapshot: tuple[ResidentModel, ...], models: Iterable[str], phase: str
) -> tuple[ResidentModel, ...]:
    resident = {row.id for row in snapshot}
    absent = sorted(m for m in set(models) if m not in resident)
    if absent:
        raise ResidencyRefused(phase, f"model(s) not resident at {phase}: {absent}")
    return snapshot


def check_start(client: Client, inputs: FitInputs) -> tuple[ResidentModel, ...]:
    """The start snapshot, or `ResidencyRefused` if the read raises or a model is absent."""
    snapshot = _read_residency(client, "start")
    return _require_both(snapshot, (inputs.decisionModel, inputs.handoffModel), "start")


def check_end(client: Client, inputs: FitInputs) -> tuple[ResidentModel, ...]:
    """A fresh end read, or `ResidencyRefused` if it raises or a model is gone. Writes nothing."""
    snapshot = _read_residency(client, "end")
    return _require_both(snapshot, (inputs.decisionModel, inputs.handoffModel), "end")


# --- abort ---


def write_abort_record(
    record: AbortRecord, fits_dir: Path, *, now: Callable[[], datetime] | None = None
) -> Path:
    """`fits/abort-<UTC ts>-<runId>.json`, created exclusively; the directory is made if absent."""
    stamp = (now() if now is not None else datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    fits_dir.mkdir(parents=True, exist_ok=True)
    path = fits_dir / f"abort-{stamp}-{record.runId}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record.to_json(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


# --- the collector ---


def _handoff_verdict(role: str, content: str) -> tuple[Any, bool]:
    if role == "guard-judge":
        return parse_guard_handoff(content)
    if role == "join-in-judge":
        return parse_join_in_handoff(content)
    return parse_message_type_handoff(content, load_point(role).expected)


def _decision_answer(result: DecisionResult, kind: str) -> Any:
    answer = result.answers[0]
    if kind == "predicate":
        return answer.probability
    return (answer.choice, answer.confidence)


def collect_calls(
    client: Client,
    *,
    inputs: FitInputs,
    items: Sequence[CalibrationItem],
    texts: Sequence[Question],
    residency_start: tuple[ResidentModel, ...],
    fits_dir: Path,
    dispositions: Iterable[Disposition] = (),
    clock: Callable[[], float] | None = None,
    now: Callable[[], datetime] | None = None,
) -> CollectedCalls:
    """Every call of the fit, in item order (hand-off, then each text), or `FitAborted`.

    `clock` (default `time.monotonic`) times a failed call; a success is timed by the client's own
    `wallClockMs`. `texts` are the questions under test: each is sent alone, its `id` is the
    `textId`.
    """
    if inputs.role not in ROLES:
        raise ValueError(f"unknown role {inputs.role!r}; expected one of {ROLES}")
    if clock is None:
        clock = time.monotonic
    kind = load_point(inputs.role).kind
    dispositions = list(dispositions)
    exempt = {d.key for d in dispositions}
    by_call = {d.key[:3]: d for d in dispositions}
    calls: list[ScoredCall] = []
    max_probabilities: dict[tuple[str, str], float] = {}
    fired: set[tuple[str, str, str | None]] = set()
    parse_failures = 0

    def succeeded(item_id: str, call_kind: str, text_id: str | None, outcome: str) -> None:
        excluded = by_call.get((item_id, call_kind, text_id))
        if excluded is not None:
            raise ExclusionDidNotFire(excluded, outcome)

    def failed(
        exc: LMStudioError, item_id: str, call_kind: str, text_id: str | None, started: float
    ) -> None:
        elapsed = clock() - started
        fkind = failure_kind(exc)
        status = exc.status if isinstance(exc, LMStudioCallFailed) else None
        if (item_id, call_kind, text_id, fkind, status) in exempt:
            fired.add((item_id, call_kind, text_id))
            calls.append(ScoredCall(item_id, call_kind, text_id, "failed", None, elapsed))
            return
        try:
            residency_now: tuple[ResidentModel, ...] | None = tuple(client.residency())
        except Exception:  # the abort record must be written whatever the read does
            residency_now = None
        record = AbortRecord(
            schema=ABORT_SCHEMA,
            runId=uuid.uuid4().hex,
            itemId=item_id,
            callKind=call_kind,
            textId=text_id,
            failureKind=fkind,
            status=status,
            elapsedSeconds=elapsed,
            inputs=inputs,
            residencyStart=residency_start,
            residencyNow=residency_now,
        )
        raise FitAborted(record, write_abort_record(record, fits_dir, now=now)) from exc

    for item in items:
        started = clock()
        try:
            chat = client.chat(
                handoff_messages(inputs.role, item.handoffInput),
                model=inputs.handoffModel,
                temperature=HANDOFF_TEMPERATURE,
                max_tokens=HANDOFF_MAX_TOKENS,
                timeout_s=HANDOFF_CHAT_TIMEOUT_SECONDS,
            )
        except (LMStudioCallTimeout, LMStudioCallFailed) as exc:
            failed(exc, item.itemId, "handoff", None, started)
        else:
            succeeded(item.itemId, "handoff", None, "ok")
            verdict, parse_failure = _handoff_verdict(inputs.role, chat.content)
            parse_failures += parse_failure
            calls.append(
                ScoredCall(item.itemId, "handoff", None, "ok", verdict, chat.wallClockMs / 1000)
            )
        for text in texts:
            started = clock()
            try:
                result = client.decide(
                    item.decisionInput,
                    [text.spec],
                    model=inputs.decisionModel,
                    timeout_s=inputs.decisionTimeoutSeconds,
                )
            except (LMStudioCallTimeout, LMStudioCallFailed) as exc:
                failed(exc, item.itemId, "decision", text.id, started)
            else:
                succeeded(
                    item.itemId,
                    "decision",
                    text.id,
                    "over-limit" if isinstance(result, OverLimit) else "ok",
                )
                if isinstance(result, OverLimit):
                    calls.append(
                        ScoredCall(
                            item.itemId,
                            "decision",
                            text.id,
                            "over-limit",
                            None,
                            result.wallClockMs / 1000,
                        )
                    )
                else:
                    if kind == "choice" and result.answers[0].probabilities:
                        max_probabilities[(item.itemId, text.id)] = max(
                            probability for _, probability in result.answers[0].probabilities
                        )
                    calls.append(
                        ScoredCall(
                            item.itemId,
                            "decision",
                            text.id,
                            "ok",
                            _decision_answer(result, kind),
                            result.wallClockMs / 1000,
                        )
                    )
    for call_key, unfired in by_call.items():
        if call_key not in fired:  # a call that never ran cannot have failed (fit note §4)
            raise ExclusionDidNotFire(unfired, "never made")
    return CollectedCalls(tuple(calls), parse_failures, max_probabilities)


# --- summaries ---


def wilson_upper(k: int, n: int, z: float = WILSON_Z) -> float:
    """The Wilson 95% upper bound for `k` of `n` (fit note §4)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"wilson_upper needs 0 <= k <= n and n > 0, got k={k}, n={n}")
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + spread) / (1 + z * z / n))


def summarize_calls(
    calls: Iterable[ScoredCall],
    *,
    decision_timeout_seconds: float,
    parse_failures: int = 0,
) -> dict[str, CallSummary]:
    """One `CallSummary` per call kind present: count `n`, median and max wall clock, `k` (calls
    scored `failed`, i.e. under a disposition) with their item ids, the Wilson bound and the count
    of calls above half the call's timeout budget (the decision timeout; 180 s for hand-offs).
    `parse_failures` (the collector's count) is attached to the hand-off summary."""
    budgets = {"decision": decision_timeout_seconds, "handoff": HANDOFF_CHAT_TIMEOUT_SECONDS}
    by_kind: dict[str, list[ScoredCall]] = {}
    for call in calls:
        by_kind.setdefault(call.callKind, []).append(call)
    summaries: dict[str, CallSummary] = {}
    for call_kind in CALL_KINDS:
        group = by_kind.get(call_kind)
        if not group:
            continue
        times = [c.elapsedSeconds for c in group]
        dispositioned = [c for c in group if c.outcome == "failed"]
        summaries[call_kind] = CallSummary(
            callKind=call_kind,
            n=len(group),
            k=len(dispositioned),
            dispositionItemIds=tuple(sorted({c.itemId for c in dispositioned})),
            medianSeconds=statistics.median(times),
            maxSeconds=max(times),
            wilsonUpper95=wilson_upper(len(dispositioned), len(group)),
            halfBudgetCount=sum(1 for t in times if t > budgets[call_kind] / 2),
            parseFailures=parse_failures if call_kind == "handoff" else 0,
        )
    return summaries
