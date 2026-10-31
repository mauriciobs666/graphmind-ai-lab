"""Typed decision points: the pure half (typed-decision-models plan v1.3 §3.3, §3.5, §3.6, §4 U1c,
as extended by plan2 v1.1 §3.2, §3.3 and plan3 v1.2 §3.2, §3.3).

Everything here is a pure function or a pinned table, so the same behaviour can be transcribed
into falkor-chat and decision-models and held equal by golden vectors rather than a shared library
(FR-23 forbids the dependency, v1.3 §3.2.2). `apply_policy` is pinned by
`tests/fixtures/decision-contract/apply-policy.jsonl`, `input_hash`/`INPUT_FIELDS_BY_ROLE` by
`input-hash.jsonl`, and `is_over_limit`/`OVER_LIMIT_MARKER` by `over-limit.jsonl`; those files
are model-bench's canonical copies.

The requirement constants are requirement values, cited by FR. A pack cannot lower them: they
are code, not pack data (v1.3 §3.5).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

from modelbench.lmstudio import DecisionAnswer

# --- requirement constants (v1.3 §3.5; plan2 §3.2) ---------------------------------------------

#: FR-11: the fewest scored `eval`-stratum items a bar claim may rest on.
MIN_EVAL_ITEMS = 200
#: FR-2: the incumbent's mean latency over the candidate's, at least.
FR2_MIN_SPEEDUP = 2.0
#: FR-10: agreement within each true answer, at least.
FR10_PER_ANSWER_AGREEMENT = 0.90
#: FR-10: mean decision latency, at most, in milliseconds.
FR10_MEAN_LATENCY_MS_MAX = 350
#: Plan2 §3.2 (note §Q4.3): the one chat-call timeout a decision-point run may carry a verdict at.
DECISION_POINT_CHAT_TIMEOUT_SECONDS = 180

# --- pinned tables ------------------------------------------------------------------------------

#: v1.3 §3.5: a decision point's bar kind comes from its role, never from the pack. `guard-judge`
#: only until U4 adds `join-in-judge` and `message-type-classifier` (both `new-ability`).
BAR_KIND_BY_ROLE: Mapping[str, str] = MappingProxyType({"guard-judge": "speed-up"})

#: Plan2 §3.3: what a failed **hand-off chat call** does to the item. `withhold` keeps today's
#: behaviour; `fail-closed` keeps the item scored as the fail-closed answer. Keyed exactly like
#: `BAR_KIND_BY_ROLE` (U4 adds the two `fail-closed` rows); U2 is the first consumer.
HANDOFF_FAILURE_BY_ROLE: Mapping[str, str] = MappingProxyType({"guard-judge": "withhold"})

#: v1.3 §3.5: the item fields `input_hash` covers, per model-bench role. `guard-judge` only until
#: U4 adds the other two rows.
INPUT_FIELDS_BY_ROLE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"guard-judge": ("condition", "understanding", "turns", "path")}
)

#: Plan2 §3.2 / note §Q2.0: the rung a fit reached. Every outcome is evaluated (FR-4).
FIT_OUTCOMES: frozenset[str] = frozenset(
    {
        "within-margin",
        "below-margin",
        "fit-infeasible: ranking",
        "fit-infeasible: agreement",
        "fit-infeasible: latency",
    }
)

#: Plan2 §3.2 / note §Q2.1–§Q2.2: the quality target a fit used, closed per bar kind.
QUALITY_TARGETS_BY_BAR_KIND: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "speed-up": frozenset({"D_c<=0", "mcnemar-nonreject"}),
        "new-ability": frozenset(
            {"T=0.94", "T=0.93", "T=0.92", "T=0.91", "T=0.90", "max-min-agreement"}
        ),
    }
)

#: The policy kind each question type takes (v1.3 §3.3).
_POLICY_KIND_BY_QUESTION_TYPE: Mapping[str, str] = MappingProxyType(
    {"predicate": "predicate-band", "choice": "choice-confidence"}
)

# --- apply_policy (v1.3 §3.3) -------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyDecision:
    """A policy's verdict on one decision-model answer: `handoff=True` with `answer=None`, or the
    model's own answer — `True`/`False` for a predicate, the chosen value for a choice."""

    handoff: bool
    answer: bool | str | None

    def __post_init__(self) -> None:
        if (self.answer is None) != self.handoff:
            raise ValueError(
                f"PolicyDecision: answer is None iff handoff (handoff={self.handoff!r}, "
                f"answer={self.answer!r})"
            )


def _policy_bounds_problem(policy: Mapping[str, Any]) -> str | None:
    """The policy's own bounds — `0 ≤ l ≤ h ≤ 1` for a band, `0 ≤ c ≤ 1` for a confidence floor —
    or `None`. The one home of the rule: `apply_policy` and `Candidate.load` both call it."""
    if policy.get("kind") == "predicate-band":
        low, high = policy.get("noBelow"), policy.get("yesAtOrAbove")
        if not (_probability(low) and _probability(high) and low <= high):
            return f"policy band must have 0 <= noBelow <= yesAtOrAbove <= 1, got {low!r}, {high!r}"
        return None
    floor = policy.get("minConfidence")
    if not _probability(floor):
        return f"policy.minConfidence must be in [0, 1], got {floor!r}"
    return None


def apply_policy(policy: Mapping[str, Any], answer: DecisionAnswer) -> PolicyDecision:
    """v1.3 §3.3. The policy's bounds are checked here too, not left to `Candidate.load`: on a
    band with `l > h` the yes-first and no-first branch orders disagree, so a transcribed copy
    could silently differ (impl review Pass 2, P2-m2). With `l ≤ h` enforced first, the order is
    immaterial — `p < l` implies `p < h`, and `p ≥ h` implies `p ≥ l`.

    - `predicate-band {yesAtOrAbove: h, noBelow: l}`: *yes* iff `p ≥ h`, *no* iff `p < l`,
      hand-off iff `l ≤ p < h`. `l == h` is an empty band: never a hand-off.
    - `choice-confidence {minConfidence: c}`: the choice iff `confidence ≥ c`, else hand-off.
      `c == 0` never hands off.
    """
    kind = policy.get("kind")
    if kind in ("predicate-band", "choice-confidence"):
        problem = _policy_bounds_problem(policy)
        if problem is not None:
            raise ValueError(f"apply_policy: {problem}")
    if kind == "predicate-band" and answer.type == "predicate":
        p = answer.probability
        if p is None:
            raise ValueError("apply_policy: a predicate answer with no probability")
        if p >= policy["yesAtOrAbove"]:
            return PolicyDecision(handoff=False, answer=True)
        if p < policy["noBelow"]:
            return PolicyDecision(handoff=False, answer=False)
        return PolicyDecision(handoff=True, answer=None)
    if kind == "choice-confidence" and answer.type == "choice":
        if answer.confidence is None or answer.choice is None:
            raise ValueError("apply_policy: a choice answer with no choice or confidence")
        if answer.confidence >= policy["minConfidence"]:
            return PolicyDecision(handoff=False, answer=answer.choice)
        return PolicyDecision(handoff=True, answer=None)
    raise ValueError(f"apply_policy: a {kind!r} policy cannot decide a {answer.type!r} answer")


# --- over-limit recognition (plan3 §3.3) -------------------------------------------------------

#: Plan3 §3.3: the text inside LM Studio's 500 body when one question's input exceeds the decision
#: model's physical batch (`ws:agent-team` a1d388f4…). One LM Studio version's wording (R-16): a
#: change turns over-limit into an ordinary decision failure, which fails safe.
OVER_LIMIT_MARKER = "too large to process"


def is_over_limit(status: int | None, body: str | None) -> bool:
    """Plan3 §3.3: a decision call is over-limit iff its status is `500` **and** its error body
    contains `OVER_LIMIT_MARKER`, case-sensitively. Any other 500 stays a failure; `body=None` (a
    non-HTTP raise) never is. Pinned by `tests/fixtures/decision-contract/over-limit.jsonl`."""
    return status == 500 and body is not None and OVER_LIMIT_MARKER in body


# --- input_hash, eval_set_hash (v1.3 §3.5) ------------------------------------------------------


def _input_fields(role: str) -> tuple[str, ...]:
    try:
        return INPUT_FIELDS_BY_ROLE[role]
    except KeyError:
        raise ValueError(f"no input fields are defined for role {role!r}") from None


def input_hash(item: Mapping[str, Any], role: str) -> str:
    """The leakage key: sha256 over the canonical JSON of the item's input fields only (v1.3
    §3.5) — keys sorted at every level, no whitespace, non-ASCII kept raw. A field the item lacks
    is omitted, never written as `null`."""
    fields = {k: item[k] for k in _input_fields(role) if k in item}
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def eval_set_hash(items: Iterable[Mapping[str, Any]], role: str) -> str:
    """FR-4: the evaluation set's identity is its content, not the pack version (v1.3 §3.5).

    sha256 over the sorted lines `input_hash(item) + ":" + json.dumps(item["expected"],
    sort_keys=True)` of the `eval`-stratum items, each line ending in `\\n`. Item ids, order and
    the `side-check` stratum do not enter it; a changed label does. An item with no `stratum` is
    refused — the stratum is never defaulted."""
    lines: list[str] = []
    for item in items:
        if "stratum" not in item:
            raise ValueError(f"item {item.get('itemId')!r} has no stratum")
        if item["stratum"] == "eval":
            label = json.dumps(item["expected"], sort_keys=True)
            lines.append(input_hash(item, role) + ":" + label)
    text = "".join(line + "\n" for line in sorted(lines))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- bar_holds (v1.3 §3.6, FR-11a) --------------------------------------------------------------

BarDirection = Literal["at-least", "at-most"]


def _finite_real(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def bar_holds(measured: float, threshold: float, *, direction: BarDirection) -> bool:
    """FR-11a: whether a requirement bar holds, judged on the **measured value** alone. The
    interval is printed beside it by the report and never reaches this function. `direction` has
    no default: which side of a bar is the good one is the bar's, never a call site's guess."""
    if direction not in ("at-least", "at-most"):
        raise ValueError(f"bar_holds: unknown direction {direction!r}")
    if not (_finite_real(measured) and _finite_real(threshold)):
        raise ValueError(f"bar_holds: not a finite measurement: {measured!r} vs {threshold!r}")
    return measured >= threshold if direction == "at-least" else measured <= threshold


# --- Candidate (v1.3 §3.3; plan2 §3.2) ----------------------------------------------------------


class CandidateError(ValueError):
    """A candidate file that cannot be measured as written. The CLI's exit-4 refusal."""


#: Used with `fullmatch` only: a `$` anchor also matches before a trailing `\n` (P2-n1).
_CANDIDATE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROVENANCE_KEYS = ("builtBy", "datasetVersion", "sourceSha")


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _probability(value: Any) -> bool:
    return _finite_real(value) and 0.0 <= value <= 1.0


def _one_of(value: Any, allowed: Iterable[str] | Mapping[str, Any]) -> bool:
    """Closed-set membership for a JSON value: a list or object is simply not a member, never a
    `TypeError` from hashing it."""
    return isinstance(value, str) and value in allowed


@dataclass(frozen=True)
class Candidate:
    """`candidates/<candidateId>.json`, validated (v1.3 §3.3; plan2 §3.2). `contentHash` is the
    sha256 of the file's bytes, so any in-place edit is a new candidate for the reuse count."""

    candidateId: str
    role: str
    decisionModel: str
    question: Mapping[str, Any]
    policy: Mapping[str, Any]
    handoffModel: str | None
    decisionTimeoutSeconds: float
    fit: Mapping[str, Any]
    fitManifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    contentHash: str

    @classmethod
    def load(cls, root: Path | str, candidate_id: str) -> Candidate:
        """Read `<root>/candidates/<candidate_id>.json` and refuse, as `CandidateError`, anything
        that is not a measurable candidate: a missing or ill-typed field, a `score` question (its
        answer shape is unprobed, so `decide` would refuse every item after the POST), a policy
        outside its bounds or not matching the question, a policy that can hand off with no
        hand-off model, a fit outcome or quality target outside its closed set."""
        if (
            not isinstance(candidate_id, str)
            or not _CANDIDATE_ID_RE.fullmatch(candidate_id)
            or candidate_id in (".", "..")
        ):
            raise CandidateError(f"candidateId {candidate_id!r} is not a plain file name")
        path = Path(root) / "candidates" / f"{candidate_id}.json"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CandidateError(f"cannot read candidate {path}: {exc}") from exc
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise CandidateError(f"{path}: not valid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise CandidateError(f"{path}: not a JSON object")
        problem = _candidate_problem(body, candidate_id)
        if problem is not None:
            raise CandidateError(f"{path}: {problem}")
        return cls(
            candidateId=body["candidateId"],
            role=body["role"],
            decisionModel=body["decisionModel"],
            question=MappingProxyType(body["question"]),
            policy=MappingProxyType(body["policy"]),
            handoffModel=body["handoffModel"],
            decisionTimeoutSeconds=float(body["decisionTimeoutSeconds"]),
            fit=MappingProxyType(body["fit"]),
            fitManifest=MappingProxyType(body["fitManifest"]),
            provenance=MappingProxyType(body["provenance"]),
            contentHash=hashlib.sha256(raw).hexdigest(),
        )


_REQUIRED_KEYS = (
    "candidateId",
    "role",
    "decisionModel",
    "question",
    "policy",
    "handoffModel",
    "decisionTimeoutSeconds",
    "fit",
    "fitManifest",
    "provenance",
)


def _candidate_problem(body: Mapping[str, Any], candidate_id: str) -> str | None:
    """The first reason `body` is not a measurable candidate, or `None`."""
    for key in _REQUIRED_KEYS:
        if key not in body:
            return f"missing required field {key!r}"
    if body["candidateId"] != candidate_id:
        return f"candidateId {body['candidateId']!r} does not match its file name {candidate_id!r}"
    role = body["role"]
    if not _one_of(role, BAR_KIND_BY_ROLE):
        return f"role {role!r} is not a decision point (known: {sorted(BAR_KIND_BY_ROLE)})"
    if not _nonempty_str(body["decisionModel"]):
        return "decisionModel must be a non-empty string"
    handoff = body["handoffModel"]
    if handoff is not None and not _nonempty_str(handoff):
        return "handoffModel must be a non-empty string or null"
    timeout = body["decisionTimeoutSeconds"]
    if not (_finite_real(timeout) and timeout > 0):
        return f"decisionTimeoutSeconds must be a finite number > 0, got {timeout!r}"
    return (
        _question_problem(body["question"])
        or _policy_problem(body["policy"], body["question"]["type"], handoff)
        or _fit_problem(body["fit"], BAR_KIND_BY_ROLE[role])
        or _fit_manifest_problem(body["fitManifest"])
        or _provenance_problem(body["provenance"])
    )


def _question_problem(question: Any) -> str | None:
    if not isinstance(question, Mapping):
        return "question must be an object"
    kind = question.get("type")
    if kind == "score":
        return (
            "question.type 'score' is refused: no probe has recorded a score answer's shape, so "
            "decide() would refuse every item after the call"
        )
    if not _one_of(kind, _POLICY_KIND_BY_QUESTION_TYPE):
        return f"question.type must be predicate or choice, got {kind!r}"
    if not _nonempty_str(question.get("name")):
        return "question.name must be a non-empty string"
    if not isinstance(question.get("instructions"), str):
        return "question.instructions must be a string"
    if kind == "predicate":
        return "question.choices is not allowed on a predicate" if "choices" in question else None
    choices = question.get("choices")
    if not isinstance(choices, list) or not choices:
        return "question.choices must be a non-empty list"
    values: list[str] = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, Mapping) or not _nonempty_str(choice.get("value")):
            return f"question.choices[{index}] must be a {{value, instructions}} object"
        if not isinstance(choice.get("instructions"), str):
            return f"question.choices[{index}].instructions must be a string"
        values.append(choice["value"])
    if len(values) != len(set(values)):
        return f"question.choices values repeat: {values}"
    return None


def _policy_problem(policy: Any, question_type: str, handoff: str | None) -> str | None:
    """Band bounds `0 ≤ l ≤ h ≤ 1` (or `0 ≤ c ≤ 1`), the kind the question takes, and plan3
    §3.2's one-way hand-off-model rule: a policy that can hand off (a non-empty band, or
    `minConfidence > 0`) needs a hand-off model. The converse is **not** a rule — an empty band
    with a hand-off model is an overflow-only composite, whose only hand-offs are over-limit
    items, and the runtime's own shape."""
    if not isinstance(policy, Mapping):
        return "policy must be an object"
    kind = policy.get("kind")
    expected_kind = _POLICY_KIND_BY_QUESTION_TYPE[question_type]
    if kind != expected_kind:
        return f"policy.kind {kind!r} does not fit a {question_type} question ({expected_kind!r})"
    problem = _policy_bounds_problem(policy)
    if problem is not None:
        return problem
    if kind == "predicate-band":
        can_hand_off = policy["noBelow"] != policy["yesAtOrAbove"]
    else:
        can_hand_off = policy["minConfidence"] > 0
    if can_hand_off and handoff is None:
        return "a policy that can hand off needs a hand-off model (handoffModel is null)"
    return None


def _fit_problem(fit: Any, bar_kind: str) -> str | None:
    if not isinstance(fit, Mapping):
        return "fit must be an object"
    if not _one_of(fit.get("outcome"), FIT_OUTCOMES):
        return f"fit.outcome {fit.get('outcome')!r} is not one of {sorted(FIT_OUTCOMES)}"
    targets = QUALITY_TARGETS_BY_BAR_KIND[bar_kind]
    if not _one_of(fit.get("qualityTarget"), targets):
        return (
            f"fit.qualityTarget {fit.get('qualityTarget')!r} is not a {bar_kind} target "
            f"({sorted(targets)})"
        )
    return None


def _fit_manifest_problem(manifest: Any) -> str | None:
    if not isinstance(manifest, Mapping):
        return "fitManifest must be an object"
    if not _nonempty_str(manifest.get("file")):
        return "fitManifest.file must be a non-empty string"
    if not isinstance(manifest.get("sha256"), str) or not _SHA256_RE.fullmatch(manifest["sha256"]):
        return "fitManifest.sha256 must be 64 lowercase hex characters"
    size = manifest.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return f"fitManifest.size must be a non-negative integer, got {size!r}"
    purposes = manifest.get("purposes")
    if not isinstance(purposes, list) or not all(_nonempty_str(p) for p in purposes):
        return "fitManifest.purposes must be a list of strings"
    return None


def _provenance_problem(provenance: Any) -> str | None:
    if not isinstance(provenance, Mapping):
        return "provenance must be an object"
    for key in _PROVENANCE_KEYS:
        if not _nonempty_str(provenance.get(key)):
            return f"provenance.{key} must be a non-empty string"
    return None
