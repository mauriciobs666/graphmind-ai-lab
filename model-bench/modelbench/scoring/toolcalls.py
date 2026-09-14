"""`tool-caller` scorer (`ConversationScorer`-shaped, structural — `retrieval.py`'s/
`classification.py`'s own precedent).

Design: `docs/plans/small-model-benchmarking-s5-spec.md` §3.4, §5 Steps 0/3/4/5. Step 0 landed the
two `-ml` §4.2(f)/§4.3.1 item 8 scoring constants and the cross-module union+disjointness assertion
below. Steps 3-4 landed this module's per-turn pure functions — one per FR-8 letter (a)-(g) plus
`restraint`, plus `clean_through_turn`/`hazard_points` (`-ml` §4.3 rule 5's three consumers) and
`outcome_vectors_differ` (the determinism probe, plan item 12b). **Step 5 lands
`score_conversations` itself** (the `ConversationScorer` assembly, §2.6/§2.3/§3.4's own bullet) —
the only function in this module that needs a loaded pack or a real `Conversation` script;
everything above it is tested against hand-built `TurnTrace`/`ConversationTrace` fixtures alone,
and `score_conversations`'s own tests add hand-built `Conversation`/`Turn` scripts and a small
duck-typed pack stand-in (no on-disk pack tree, matching `retrieval.py`'s offline-stub convention).

**`Turn.expect`'s exact schema is `score_conversations`'s own synthesis, stated rather than left
implicit** (§7's honesty convention): `conversations.jsonl` does not exist yet (S6's), so no prior
unit pins one. The fields read here — `toolRequired: bool`, `tool: str` (the one required tool
name), `args: Mapping[str, Any]` (its expected arguments), `finalReplyMustContain`/
`finalReplyMustNotContain: Sequence[str]` — mirror `tests/test_convo.py`'s own hand-authored
`expect` blocks and the plan's own `conversations.jsonl` row example (`:2188-2192`), extended with
the symmetric `finalReplyMustNotContain` that `reply_matches_tool` already accepts. `argChecks`
(the plan's own literal) is deliberately not read here — S5 spec §3.3 relocated the boundary/unit
rule onto the pack's own `tools/schemas.json` (`boundaryRule`), so a scripted `argChecks` block, if
S6 authors one, is inert data `score_conversations` tolerates and never reads, exactly as
`assemble` already tolerates it (`convo.py`, `test_convo.py`'s own
`test_assemble_tolerates_the_plans_own_conversation_row_literal`). One per-turn axis is a stated,
deliberate gap: `stopping_when_done`'s own
`continued_after_satisfied` flag is approximated here as "the turn was spurious or duplicated a
call" — a real but incomplete proxy for "kept calling after `R(t)` was already satisfied" (a model
re-issuing a second, differently-argued but still-required call is neither caught) — cheap to widen
once S6's real scripts show whether it matters, and called out rather than silently narrowed.

**Two design decisions this module makes on its own, stated rather than left implicit** (S5 spec
§7's own honesty convention):

1. **`turn_disposition_scores` takes `disposition` alone**, not the `has_dispatch` parameter the S5
   spec's own signature sketch names. `-ml` §4.3 rule 4's table maps the SCORED-vs-`unrunnable`
   split onto the mechanism alone (`replied`/`cap-hit`/`timed-out` score, `no-response`/
   `server-rejected` don't) — no `E(t)` condition changes which side of that split a disposition
   lands on, only what a *scored* turn goes on to score once it is on the scored side, which is
   every other function below's own job. Threading an unused parameter through the one function
   rule 4 lives in would misstate the table it exists to consult.
2. **`clean_through_turn`/`hazard_points` take a caller-supplied `turn_clean: Sequence[bool]`
   (per conversation, one entry per `TurnTrace`) rather than deriving "did this turn fail" from the
   trace alone.** A turn's pass/fail verdict against the FR-8 (a)-(g) letters needs the script's own
   `Turn.expect` ground truth, which a bare `ConversationTrace` does not carry (`Conversation` does)
   — `score_conversations` (Step 5) is the one place that ever holds both together, so it is the
   one place that can build `turn_clean` by combining this module's own per-letter functions across
   a script's turns. What `clean_through_turn`/`hazard_points` own is `-ml` §4.3 rule 5's own
   addition on top of that verdict: the censoring carve-out for an `unrunnable` turn, which needs
   only the trace's own `turnDisposition`s and `h`/position, never `expect`.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from modelbench import convo
from modelbench.convo import Conversation, ConversationTrace, Turn, TurnDisposition
from modelbench.packs import Pack
from modelbench.results import (
    BinaryMetric,
    FunnelCounts,
    HazardPoint,
    ItemResult,
    ItemTiming,
    ToolCallAggregates,
    TurnPositionRate,
)
from modelbench.stats import LEVEL_P95, percentile

#: `-ml` §4.2(f)/§4.3.1 item 8, transcribed literally (plan `:5838-5844`): the population of the
#: `I(t)` mean/p95 summary. A **scoring** vocabulary, so it sits here beside the scorer rather than
#: in `convo`, whose `TURN_DISPOSITIONS` is a **mechanism** vocabulary. Written out literally —
#: never derived from `ITERATION_SUMMARY_EXCLUDED` or from `convo.TURN_DISPOSITIONS` — because a
#: derived complement would make the assertion below a tautology.
ITERATION_SUMMARY_DISPOSITIONS: frozenset[str] = frozenset({"replied", "cap-hit"})

#: The complement `-ml` §4.2(f) excludes from that summary, transcribed literally for the same
#: reason `ITERATION_SUMMARY_DISPOSITIONS` is.
ITERATION_SUMMARY_EXCLUDED: frozenset[str] = frozenset(
    {"timed-out", "no-response", "server-rejected"}
)

#: The cross-module union (plan `:5796-5799`, S5 spec §4 S5 *Done when* item (4)): binds two
#: independently authored declarations across two modules, so a sixth mechanism belongs to neither
#: set and reddens before it can be silently included or dropped, and so a mechanism placed in both
#: (an incomplete reclassification) reddens on the disjointness half rather than passing because the
#: union still happens to come out whole.
assert ITERATION_SUMMARY_DISPOSITIONS | ITERATION_SUMMARY_EXCLUDED == convo.TURN_DISPOSITIONS, (
    "ITERATION_SUMMARY_DISPOSITIONS | ITERATION_SUMMARY_EXCLUDED must equal "
    "convo.TURN_DISPOSITIONS exactly — a mechanism outside both sets is a coverage gap"
)
assert not (ITERATION_SUMMARY_DISPOSITIONS & ITERATION_SUMMARY_EXCLUDED), (
    "ITERATION_SUMMARY_DISPOSITIONS and ITERATION_SUMMARY_EXCLUDED must be disjoint — a mechanism "
    "in both is an incomplete reclassification, not a valid scoring vocabulary"
)

#: `-ml` §4.3 rule 4's own two-way split: which mechanisms carry no observation of the model at all
#: (the harness cannot tell an argument-shaped raise from a broken pack payload) and so are OUT of
#: every §4.2 scoring denominator, versus every other mechanism, which is IN wherever its own
#: `R(t)`/`E(t)` condition admits it (`turn_disposition_scores` below is the one place this is
#: consulted — §4.3.1 item 1).
_UNRUNNABLE_DISPOSITIONS: frozenset[str] = frozenset({"no-response", "server-rejected"})


def turn_disposition_scores(disposition: TurnDisposition) -> Literal["scored", "unrunnable"]:
    """`-ml` §4.3 rule 4's table, consulted from exactly one place. `replied`, `cap-hit` and
    `timed-out` are all `"scored"` — a turn on this side still has its OWN `R(t)`/`E(t)` condition
    to satisfy before any individual §4.2 count admits it (item 4.3.1(ii): a disposition does not
    by itself decide what scores a turn), and `timed-out` scores `fail` wherever it lands, never
    `n_a`. `no-response`/`server-rejected` are `"unrunnable"` — out of every scoring denominator,
    unconditionally."""
    return "unrunnable" if disposition in _UNRUNNABLE_DISPOSITIONS else "scored"


# --- (a)+(b) — emission form -----------------------------------------------------------------


def emission_form(
    required: bool, dispatched_count: int, prose_detected: bool
) -> Literal["native", "prose_pseudo_call", "no_attempt"]:
    """`-ml` §4.2(a)+(b)'s three-way, mutually exclusive, exhaustive partition over turns with
    `|R(t)| >= 1` — collapsing FR-8(a) ("called a tool when required") and FR-8(b) ("native form
    rather than prose") into one denominator so neither needs an intent inference.

    `required` states this turn genuinely has `|R(t)| >= 1`; a restraint turn (`R(t) = ∅`) is
    `restraint`'s to score, never this function's, and calling this on one is a caller bug this
    function refuses rather than silently misclassifying."""
    if not required:
        raise ValueError(
            "emission_form: called on a turn with R(t) = ∅ — that is restraint's denominator, "
            "not (a)+(b)'s (-ml §4.2)"
        )
    if dispatched_count >= 1:
        return "native"
    if prose_detected:
        return "prose_pseudo_call"
    return "no_attempt"


# --- the prose-pseudo-call detector and its calibration (§2.7) ---------------------------------

_PROSE_CALL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # A function-call-shaped fragment: `add_to_cart("Pad"` / `filter_products({...`.
    re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*[\"'{]"),
    # A JSON-tool-call-shaped key emitted as plain text rather than a native `tool_calls` entry.
    re.compile(r'"tool_call"\s*:', re.IGNORECASE),
    # An explicit first-person announcement of calling/invoking a tool.
    re.compile(r"\bI(?:'m| am) (?:calling|going to call|invoking)\b", re.IGNORECASE),
)


def detect_prose_pseudo_call(reply_text: str | None) -> bool:
    """The heuristic half of `-ml` §4.2(a)+(b)'s partition: did a reply that dispatched NO native
    call still describe a call in prose? A pure textual heuristic — never scored against ground
    truth here, that is `prose_detector_precision_recall`'s job against a labelled corpus.
    `None`/`""` never match: there is no text to read a pseudo-call out of."""
    if not reply_text:
        return False
    return any(pattern.search(reply_text) for pattern in _PROSE_CALL_PATTERNS)


def prose_detector_precision_recall(
    labelled: Sequence[tuple[str, bool]],
) -> tuple[float, float] | None:
    """`-ml` §4.2(a)+(b)/§2.7: `detect_prose_pseudo_call`'s own precision/recall against a pack's
    human-labelled `(replyText, isActuallyAPseudoCall)` corpus. `None` when no corpus is available
    at all (S6's own ~20-reply corpus; this module's tests use small hand-built lists) — an
    uncalibrated detector's number is not a measurement (`-ml` §4.2). At a zero denominator
    (no positive prediction, or no positive label) precision/recall follow the standard convention
    of `0.0`, never a `ZeroDivisionError` — that is a different, per-figure absence from the
    corpus-level `None` above."""
    if not labelled:
        return None
    true_positive = sum(
        1 for text, is_call in labelled if is_call and detect_prose_pseudo_call(text)
    )
    predicted_positive = sum(1 for text, _ in labelled if detect_prose_pseudo_call(text))
    actual_positive = sum(1 for _, is_call in labelled if is_call)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    return precision, recall


# --- (c) — right tool chosen --------------------------------------------------------------------


def right_tool_chosen(required_names: Collection[str], dispatched_names: Collection[str]) -> bool:
    """`-ml` §4.2(c): coverage of the REQUIRED tool names only — an extra call alongside the right
    one is (e)'s business, not (c)'s, so keeping it out of this predicate is what stops one failure
    being counted twice. Denominator (`|R(t)| >= 1` and `|E(t)| >= 1` — arguments/tool identity are
    undefined when nothing was called) is the caller's own filter; an empty argument here means the
    caller reached this function outside that denominator, refused rather than silently scored."""
    if not required_names:
        raise ValueError(
            "right_tool_chosen: `required_names` is empty (-ml §4.2(c)'s denominator)"
        )
    if not dispatched_names:
        raise ValueError(
            "right_tool_chosen: `dispatched_names` is empty (-ml §4.2(c)'s denominator)"
        )
    return set(required_names).issubset(set(dispatched_names))


# --- (d) — argument correctness -----------------------------------------------------------------

#: `nlq_scoring._scalar_equal`'s own tolerance, reused verbatim per `-ml` §4.2(d)'s explicit
#: instruction ("reuse `nlq_scoring._scalar_equal`'s discipline") — the same copy
#: `modelbench/scoring/extraction.py` already carries; a third module reusing the same rule is what
#: makes copying it, rather than importing a private helper cross-module, the right call here too.
_NUMERIC_EPSILON = 0.01

_WHITESPACE_RE = re.compile(r"\s+")


def _canon_str(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value).strip().casefold())


def _scalar_equal(expected: Any, actual: Any) -> bool:
    """Numeric epsilon for two numbers, canonical string equality otherwise, never coerced across
    the two; `bool` compared by identity, checked before the numeric branch since `bool` is an
    `int` subclass (verbatim port of `nlq_scoring._scalar_equal`'s discipline)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= _NUMERIC_EPSILON + 1e-9
    if isinstance(expected, (int, float)) or isinstance(actual, (int, float)):
        return False
    return _canon_str(expected) == _canon_str(actual)


def properties_for_tool(schemas: Sequence[Mapping[str, Any]], tool_name: str) -> Mapping[str, Any]:
    """One tool's own JSON-Schema `parameters.properties` block, looked up by name off a pack's
    `tools/schemas.json` shape (S5 spec §3.3: a list of `{"type": "function", "function": {"name":
    ..., "parameters": {"properties": {...}}}}` entries). `{}` when `tool_name` is not found —
    `argument_correctness` then finds no `boundaryRule` for any argument and degrades to a plain
    `wrong_value` classification, rather than this helper raising over a schema a caller may not
    carry for every call it checks."""
    for entry in schemas:
        function = entry.get("function", {}) if isinstance(entry, Mapping) else {}
        if not isinstance(function, Mapping) or function.get("name") != tool_name:
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
        return properties if isinstance(properties, Mapping) else {}
    return {}


@dataclass(frozen=True)
class ArgumentCorrectness:
    """One dispatched call's per-argument classification (`-ml` §4.2(d)). `wrongValue` and
    `boundaryUnit` are **not disjoint** — `boundaryUnit` is a named SUBSET of `wrongValue`, never a
    sibling bucket: an argument name confused across a declared boundary/unit rule appears in both
    tuples, so a reader can print `wrong_value: 12, of which boundary/unit: 7` directly off the two
    lengths."""

    allCorrect: bool
    omittedRequired: tuple[str, ...]
    wrongValue: tuple[str, ...]
    boundaryUnit: tuple[str, ...]


def _boundary_confused(actual_value: Any, boundary_rule: Mapping[str, Any] | None) -> bool:
    if not boundary_rule:
        return False
    confused_with = boundary_rule.get("confusedWith", ())
    return any(_scalar_equal(actual_value, alternate) for alternate in confused_with)


def argument_correctness(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
) -> ArgumentCorrectness:
    """`-ml` §4.2(d), over ONE dispatched call whose tool name is already known correct (the
    caller's own denominator filter — arguments of a wrong-tool call are meaningless). `expected`
    maps each required argument name to its correct value; `actual` is the call's own dispatched
    arguments; `schema` is that tool's own `properties_for_tool(...)` result, consulted only for a
    `boundaryRule` on an argument already found `wrong_value` — never a regex, and never consulted
    for an argument that matched.

    An argument absent from `expected` is not checked at all: `-ml` §4.2(d) scores REQUIRED
    arguments, and a tool with none (e.g. `view_cart`) scores `allCorrect=True` by construction,
    with all three failure tuples empty."""
    schema = schema or {}
    omitted: list[str] = []
    wrong: list[str] = []
    boundary: list[str] = []
    for name, expected_value in expected.items():
        if name not in actual:
            omitted.append(name)
            continue
        if _scalar_equal(expected_value, actual[name]):
            continue
        wrong.append(name)
        prop = schema.get(name, {})
        boundary_rule = prop.get("boundaryRule") if isinstance(prop, Mapping) else None
        if _boundary_confused(actual[name], boundary_rule):
            boundary.append(name)
    return ArgumentCorrectness(
        allCorrect=not omitted and not wrong,
        omittedRequired=tuple(omitted),
        wrongValue=tuple(wrong),
        boundaryUnit=tuple(boundary),
    )


# --- (e) — spurious and duplicate calls -----------------------------------------------------


@dataclass(frozen=True)
class SpuriousAndDuplicate:
    """`-ml` §4.2(e)'s two counts, the duplicate one already split into its own named breakdown
    (within-turn vs. cross-turn) rather than pooled — K-061's same-turn `add_to_cart` and the
    ministral turn-2 re-issue of turn 1 are different defects and a pooled rate would have hidden
    the turn-specific one."""

    spurious: bool
    duplicateWithinTurn: bool
    duplicateCrossTurn: bool


def _call_signature(
    name: str, arguments: Mapping[str, Any]
) -> tuple[str, tuple[tuple[str, Any], ...]]:
    return (name, tuple(sorted(arguments.items())))


def spurious_and_duplicate(
    required_names: Collection[str],
    dispatched_calls: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    prior_completed_calls: Collection[tuple[str, Mapping[str, Any]]] = (),
) -> SpuriousAndDuplicate:
    """`-ml` §4.2(e), over one turn's `E(t)` (`dispatched_calls`, `(name, parsedArguments)` pairs in
    emission order). `spurious` — `E(t)` contains a call to a tool not in `R(t)` (`required_names`).
    `duplicateWithinTurn` — `E(t)` repeats an already-emitted call within this same turn.
    `duplicateCrossTurn` — `E(t)` re-issues a call already completed in a PRIOR turn
    (`prior_completed_calls`, the same `(name, arguments)` shape). Both breakdowns key on the exact
    `(name, arguments)` pair, never name alone — a second, differently-argued call to the same tool
    is not what either K-061 or the ministral defect named."""
    required = set(required_names)
    spurious = any(name not in required for name, _ in dispatched_calls)
    prior_signatures = {_call_signature(name, args) for name, args in prior_completed_calls}
    seen_this_turn: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    duplicate_within = False
    duplicate_cross = False
    for name, arguments in dispatched_calls:
        signature = _call_signature(name, arguments)
        if signature in prior_signatures:
            duplicate_cross = True
        elif signature in seen_this_turn:
            duplicate_within = True
        seen_this_turn.add(signature)
    return SpuriousAndDuplicate(
        spurious=spurious, duplicateWithinTurn=duplicate_within, duplicateCrossTurn=duplicate_cross
    )


# --- (f) — stopping when done, and the two iteration figures beside it -------------------------


def stopping_when_done(
    disposition: TurnDisposition,
    dispatched_count: int,
    *,
    continued_after_satisfied: bool,
) -> bool | None:
    """`-ml` §4.2(f)'s `stopping_when_done`. `None` when this turn is OUTSIDE the denominator:
    `dispatched_count == 0` (`|E(t)| = 0` — includes the cap-hit-with-empty-dispatch-trace
    discriminating case, `-ml` §4.3.1 item 4) or `turn_disposition_scores(disposition) ==
    "unrunnable"`. Otherwise `False` for every non-`replied` disposition on the scored side
    (`cap-hit` — it never stopped on its own; `timed-out` — outcome `fail`, never `n_a`, per rule
    4), and for `replied` turns, `True` iff no call was dispatched after `R(t)` was already fully
    satisfied."""
    if dispatched_count == 0:
        return None
    if turn_disposition_scores(disposition) == "unrunnable":
        return None
    if disposition != "replied":
        return False
    return not continued_after_satisfied


def _p95_rank(x: int) -> int:
    """The exact integer rank `stats.percentile` computes internally for `LEVEL_P95`
    (`-ml` §11.2.1's own expression, reproduced here — never re-derived — only to answer `-ml`
    §4.2(f)'s censoring QUESTION, "is `r <= X - c`?", never to recompute the p95's VALUE itself,
    which stays `stats.percentile`'s alone per §11.2.2's one-implementation rule)."""
    return max(1, min(x, -(-LEVEL_P95.numerator * x // LEVEL_P95.denominator)))


@dataclass(frozen=True)
class IterationSummary:
    """`-ml` §4.2(f)'s two censored statistics over `ITERATION_SUMMARY_DISPOSITIONS` alone (never
    a rate; the report divides). `mean`/`p95` are `None` iff `n == 0`. `meanCensored` is `True`
    whenever `capHitCount > 0` (the mean must print `>= value`); `p95Censored` is `True` whenever
    the p95's own rank falls at or above `n - capHitCount` (the p95 must print `>= value`) — both
    booleans, never a baked-in rendering, since presentation is `report.py`'s (rule 1)."""

    n: int
    capHitCount: int
    mean: float | None
    meanCensored: bool
    p95: float | None
    p95Censored: bool


def iteration_summary(observations: Sequence[tuple[TurnDisposition, int]]) -> IterationSummary:
    """`-ml` §4.2(f)'s mean/p95 of `I(t)`, over EVERY turn driven (any disposition) —
    this function itself filters to `ITERATION_SUMMARY_DISPOSITIONS`, so a caller cannot filter it
    two different ways at two different call sites. `observations` pairs each turn's own
    `turnDisposition` with its `iterations` count."""
    included = [
        (disposition, iterations)
        for disposition, iterations in observations
        if disposition in ITERATION_SUMMARY_DISPOSITIONS
    ]
    n = len(included)
    if n == 0:
        return IterationSummary(
            n=0, capHitCount=0, mean=None, meanCensored=False, p95=None, p95Censored=False
        )
    values = [iterations for _, iterations in included]
    cap_hit_count = sum(1 for disposition, _ in included if disposition == "cap-hit")
    rank = _p95_rank(n)
    return IterationSummary(
        n=n,
        capHitCount=cap_hit_count,
        mean=statistics.fmean(values),
        meanCensored=cap_hit_count > 0,
        p95=percentile(values, level=LEVEL_P95),
        p95Censored=rank > n - cap_hit_count,
    )


# --- (g) — final reply matches what the tool returned -------------------------------------------


def reply_matches_tool(
    reply_text: str | None, must_contain: Sequence[str], must_not_contain: Sequence[str]
) -> bool:
    """`-ml` §4.2(g): deterministic normalized-substring containment, never a judge. `reply_text`
    is canonicalized once (casefold, whitespace-collapsed) and every `must_contain` value must
    appear in it while no `must_not_contain` value may. `reply_text is None` (a non-`replied` turn)
    scores `False` outright — there is no reply to check containment against."""
    if reply_text is None:
        return False
    canon_reply = _canon_str(reply_text)
    return all(_canon_str(value) in canon_reply for value in must_contain) and not any(
        _canon_str(value) in canon_reply for value in must_not_contain
    )


# --- restraint -----------------------------------------------------------------------------------


def restraint(dispatched_count: int) -> bool:
    """`-ml` §4.2's own added count, over a turn with `R(t) = ∅` (the caller's own filter): `True`
    iff the model dispatched nothing. Without this count a model that calls tools indiscriminately
    scores perfectly on (a) and abstention turns contribute nothing."""
    return dispatched_count == 0


# --- `-ml` §4.3 rule 5's three consumers: the headline's third state, and the hazard curve -------


def clean_through_turn(
    trace: ConversationTrace,
    turn_clean: Sequence[bool],
    *,
    h: int,
) -> Literal["clean", "failed", "n_a"]:
    """`-ml` §4.3 rule 5 / plan §4.6's headline third state. `turn_clean[i]` is the caller's own
    per-turn clean/failed verdict for `trace.turns[i]` — every FR-8 (a)-(g) check this module
    exposes, folded together against the script's own `expect`, is `score_conversations`'s job
    (Step 5), never this function's; `clean_through_turn` decides only what rule 5 adds on top: the
    censoring carve-out.

    A trace shorter than `h` (`len(trace.turns) < h`) means a tool-channel `ToolDispatchFailed`
    censored the conversation before turn `h` was ever reached (rule 5's tool-channel exception —
    those later turns are never driven at all): `"n_a"`. Otherwise, if any of the first `h` turns
    carries a model-channel `unrunnable` disposition (`no-response`/`server-rejected`): `"n_a"`
    too. Failing both, `"clean"` iff every one of the first `h` turns' `turn_clean` entry is `True`,
    else `"failed"`.
    """
    if len(turn_clean) != len(trace.turns):
        raise ValueError(
            "clean_through_turn: turn_clean must carry exactly one entry per trace.turns "
            f"({len(turn_clean)} given for {len(trace.turns)} turns)"
        )
    if len(trace.turns) < h:
        return "n_a"
    considered = trace.turns[:h]
    if any(turn_disposition_scores(turn.turnDisposition) == "unrunnable" for turn in considered):
        return "n_a"
    return "clean" if all(turn_clean[:h]) else "failed"


def hazard_points(
    traces: Sequence[ConversationTrace],
    turn_clean: Sequence[Sequence[bool]],
    *,
    h: int | None = None,
) -> tuple[HazardPoint, ...]:
    """`-ml` §4.3 rule 5's other two consumers (§4.3.1 item 11): the per-turn survival curve,
    conditioned on the SAME "clean through `t-1`" predicate `clean_through_turn` scores at one
    fixed `H`, computed here at every position instead. `turn_clean[i][t]` is the caller's own
    per-turn verdict for `traces[i].turns[t]` (`score_conversations`'s job, Step 5) — this function
    decides only the survival bookkeeping.

    At each position `t`: `r_t` (`HazardPoint.metric.n`) is the count of conversations still "at
    risk" entering `t` (clean through `t-1`, never previously censored) whose turn `t` ALSO exists
    and is not itself an `unrunnable` mechanism — an unrunnable turn carries no observation, so it
    is out of `r_t` at its own position too, exactly as rule 4 puts it out of every OTHER scoring
    denominator. `c_t` (`.censored`) is, of those entering `t`, how many are newly removed there —
    by an `unrunnable` disposition, or by the trace simply ending before `t` (tool-channel
    censoring: those turns are never driven at all). `f_t` (`.metric.successes`) is, of the `r_t`
    that DO have an observation at `t`, how many are `turn_clean[i][t] is False`. A conversation
    leaves the risk set for every later position the moment it is censored OR fails — this is a
    time-to-FIRST-failure curve, not a per-position independent rate.

    `h` bounds how many positions are computed (`0 .. h-1`); `None` computes every position up to
    the longest trace given.
    """
    if len(turn_clean) != len(traces):
        raise ValueError(
            "hazard_points: turn_clean must carry one entry per trace "
            f"({len(turn_clean)} given for {len(traces)} traces)"
        )
    for trace, clean in zip(traces, turn_clean, strict=True):
        if len(clean) != len(trace.turns):
            raise ValueError(
                f"hazard_points: turn_clean for {trace.scriptId!r} must match its own turn count "
                f"({len(clean)} given for {len(trace.turns)} turns)"
            )
    max_len = max((len(t.turns) for t in traces), default=0)
    limit = max_len if h is None else min(h, max_len)
    at_risk = list(range(len(traces)))
    points: list[HazardPoint] = []
    for t in range(limit):
        censored_now: list[int] = []
        observable: list[int] = []
        for i in at_risk:
            trace = traces[i]
            if t >= len(trace.turns) or (
                turn_disposition_scores(trace.turns[t].turnDisposition) == "unrunnable"
            ):
                censored_now.append(i)
            else:
                observable.append(i)
        failures_now = [i for i in observable if not turn_clean[i][t]]
        still_at_risk = [i for i in observable if turn_clean[i][t]]
        points.append(
            HazardPoint(
                turnIndex=t,
                metric=BinaryMetric(
                    name="hazard",
                    successes=len(failures_now),
                    n=len(observable),
                    unit="conversation",
                ),
                censored=len(censored_now),
            )
        )
        at_risk = still_at_risk
    return tuple(points)


# --- the determinism probe's own pure function (plan item 12b) ---------------------------------


def _turn_signature(turn: Any) -> tuple[Any, ...]:
    return (
        turn.turnDisposition,
        turn.finalReplyText,
        tuple((d.name, d.parsedArguments) for d in turn.dispatches),
    )


def outcome_vectors_differ(a: ConversationTrace, b: ConversationTrace) -> tuple[int, ...]:
    """S5 spec §3.4's determinism-probe comparator: per turn, compares
    `(turnDisposition, finalReplyText, tuple((d.name, d.parsedArguments) for d in dispatches))` —
    disposition and reply text because scoring rules 4/5 branch on them, dispatched
    `(name, parsedArguments)` pairs because FR-8(c)/(d) score them. Deliberately NOT `wallClockMs`
    (always differs), not `DispatchRecord.timestamp`/`.returnValue` (derivable from a deterministic
    sim's `parsedArguments`), not raw `messagesSent`/tool-call ids (transport-level, can differ with
    no behavioural difference).

    Returns the tuple of turn indices where the two traces differ — `()` when identical. A trace
    shorter than the other is itself a difference: every position beyond the shorter one counts as
    differing, rather than raising, since a length mismatch is exactly the kind of non-identical
    outcome this probe exists to report."""
    length = max(len(a.turns), len(b.turns))
    differing: list[int] = []
    for i in range(length):
        sig_a = _turn_signature(a.turns[i]) if i < len(a.turns) else None
        sig_b = _turn_signature(b.turns[i]) if i < len(b.turns) else None
        if sig_a != sig_b:
            differing.append(i)
    return tuple(differing)


# ==================================================================================================
# Step 5 — `score_conversations`, the `ConversationScorer` assembly (S5 spec §2.3/§2.6/§3.4)
# ==================================================================================================


class _Tally:
    """One run's `FunnelCounts`/`.funnel` inputs, accumulated in a single pass over every scored
    conversation's turns (never a second pass — `score_conversations` builds `turn_clean` and these
    counters together, since both read the same per-turn FR-8 verdict)."""

    def __init__(self) -> None:
        self.turnsDriven = 0
        self.unrunnableModelChannel = 0
        self.unrunnableToolChannel = 0
        self.turnsScoredAfterUnrunnable = 0
        self.restraintTurns = 0
        self.restraintSuccesses = 0
        self.requiredCallTurns = 0
        self.nativeCallEmitted = 0
        self.prosePseudoCall = 0
        self.noAttempt = 0
        self.turnsWithAnyCall = 0
        self.dispatchedCalls = 0
        self.factBearingReturns = 0
        self.unscoreableReturns = 0
        self.capHitScored = 0
        self.rightToolSuccesses = 0
        self.argsCorrectSuccesses = 0
        self.argsCorrectTotal = 0
        self.spuriousCount = 0
        self.duplicateWithinCount = 0
        self.duplicateCrossCount = 0
        self.stoppingSuccesses = 0
        self.stoppingTotal = 0
        self.replyMatchSuccesses = 0

    def funnel_counts(self) -> FunnelCounts:
        return FunnelCounts(
            turnsDriven=self.turnsDriven,
            unrunnableModelChannel=self.unrunnableModelChannel,
            unrunnableToolChannel=self.unrunnableToolChannel,
            turnsScoredAfterUnrunnable=self.turnsScoredAfterUnrunnable,
            restraintTurns=self.restraintTurns,
            requiredCallTurns=self.requiredCallTurns,
            nativeCallEmitted=self.nativeCallEmitted,
            prosePseudoCall=self.prosePseudoCall,
            noAttempt=self.noAttempt,
            turnsWithAnyCall=self.turnsWithAnyCall,
            dispatchedCalls=self.dispatchedCalls,
            factBearingReturns=self.factBearingReturns,
            unscoreableReturns=self.unscoreableReturns,
        )

    def funnel_metrics(self) -> tuple[BinaryMetric, ...]:
        """The FR-8 (a)-(g) RATE metrics only (S5 spec §2.5) — pooled over every scored
        conversation's turns/calls, never per-item. Excludes the funnel-head's own structural
        counts (`funnel_counts()`, above) and `restraint` (`ToolCallAggregates`'s own field)."""
        return (
            BinaryMetric(
                name="iterationCapHitRate",
                successes=self.capHitScored,
                n=self.turnsDriven - self.unrunnableModelChannel,
                unit="turn",
            ),
            BinaryMetric(
                name="native", successes=self.nativeCallEmitted, n=self.requiredCallTurns,
                unit="turn",
            ),
            BinaryMetric(
                name="rightToolChosen", successes=self.rightToolSuccesses,
                n=self.turnsWithAnyCall, unit="turn",
            ),
            BinaryMetric(
                name="allArgsCorrect", successes=self.argsCorrectSuccesses,
                n=self.argsCorrectTotal, unit="call",
            ),
            BinaryMetric(
                name="spurious", successes=self.spuriousCount, n=self.turnsWithAnyCall,
                unit="turn",
            ),
            BinaryMetric(
                name="duplicateWithinTurn", successes=self.duplicateWithinCount,
                n=self.turnsWithAnyCall, unit="turn",
            ),
            BinaryMetric(
                name="duplicateCrossTurn", successes=self.duplicateCrossCount,
                n=self.turnsWithAnyCall, unit="turn",
            ),
            BinaryMetric(
                name="stoppingWhenDone", successes=self.stoppingSuccesses, n=self.stoppingTotal,
                unit="turn",
            ),
            BinaryMetric(
                name="replyMatchesTool", successes=self.replyMatchSuccesses,
                n=self.factBearingReturns, unit="turn",
            ),
        )


def _required_tool_names(turn: Turn) -> set[str]:
    return {turn.expect["tool"]} if turn.expect.get("toolRequired") else set()


def _score_one_conversation(
    script: Conversation,
    trace: ConversationTrace,
    *,
    schemas: Sequence[Mapping[str, Any]],
    tally: _Tally,
) -> list[bool]:
    """One conversation's own pass: folds every driven turn's FR-8 verdict into `tally` (mutated in
    place) and returns `turn_clean`, aligned one-to-one with `trace.turns` — `clean_through_turn`/
    `hazard_points`'s own required shape (§4 Step 4)."""
    turn_clean: list[bool] = []
    seen_unrunnable = False
    prior_calls: list[tuple[str, Mapping[str, Any]]] = []

    if len(trace.turns) < len(script.turns):
        tally.unrunnableToolChannel += 1

    for i, t_turn in enumerate(trace.turns):
        s_turn = script.turns[i]
        tally.turnsDriven += 1
        mechanism = turn_disposition_scores(t_turn.turnDisposition)

        if mechanism == "unrunnable":
            tally.unrunnableModelChannel += 1
            seen_unrunnable = True
            turn_clean.append(True)  # moot: excluded by mechanism before `turn_clean` is read
            prior_calls.extend((d.name, d.parsedArguments) for d in t_turn.dispatches)
            continue

        if seen_unrunnable:
            tally.turnsScoredAfterUnrunnable += 1
        if t_turn.turnDisposition == "cap-hit":
            tally.capHitScored += 1

        required_names = _required_tool_names(s_turn)
        dispatched_count = len(t_turn.dispatches)

        if not required_names:
            tally.restraintTurns += 1
            clean = restraint(dispatched_count)
            if clean:
                tally.restraintSuccesses += 1
            turn_clean.append(clean and t_turn.turnDisposition == "replied")
            prior_calls.extend((d.name, d.parsedArguments) for d in t_turn.dispatches)
            continue

        tally.requiredCallTurns += 1
        if dispatched_count == 0:
            prose_detected = detect_prose_pseudo_call(t_turn.finalReplyText)
            form = emission_form(required=True, dispatched_count=0, prose_detected=prose_detected)
            if form == "prose_pseudo_call":
                tally.prosePseudoCall += 1
            else:
                tally.noAttempt += 1
            turn_clean.append(False)
            continue

        tally.turnsWithAnyCall += 1
        tally.nativeCallEmitted += 1
        tally.dispatchedCalls += dispatched_count
        dispatched_names = {d.name for d in t_turn.dispatches}
        right = right_tool_chosen(required_names, dispatched_names)
        if right:
            tally.rightToolSuccesses += 1

        expected_args = s_turn.expect.get("args", {})
        matching_calls = [d for d in t_turn.dispatches if d.name in required_names]
        args_all_correct = bool(matching_calls)
        for call in matching_calls:
            tally.argsCorrectTotal += 1
            schema = properties_for_tool(schemas, call.name)
            correctness = argument_correctness(expected_args, call.parsedArguments, schema=schema)
            if correctness.allCorrect:
                tally.argsCorrectSuccesses += 1
            else:
                args_all_correct = False

        calls_this_turn = [(d.name, d.parsedArguments) for d in t_turn.dispatches]
        spurious_dup = spurious_and_duplicate(
            required_names, calls_this_turn, prior_completed_calls=prior_calls
        )
        if spurious_dup.spurious:
            tally.spuriousCount += 1
        if spurious_dup.duplicateWithinTurn:
            tally.duplicateWithinCount += 1
        if spurious_dup.duplicateCrossTurn:
            tally.duplicateCrossCount += 1

        tally.stoppingTotal += 1
        continued = (
            spurious_dup.spurious
            or spurious_dup.duplicateWithinTurn
            or spurious_dup.duplicateCrossTurn
        )
        stops_cleanly = stopping_when_done(
            t_turn.turnDisposition, dispatched_count, continued_after_satisfied=continued
        )
        if stops_cleanly:
            tally.stoppingSuccesses += 1

        must_contain = s_turn.expect.get("finalReplyMustContain", [])
        must_not_contain = s_turn.expect.get("finalReplyMustNotContain", [])
        reply_ok = True
        if must_contain or must_not_contain:
            tally.factBearingReturns += 1
            reply_ok = reply_matches_tool(t_turn.finalReplyText, must_contain, must_not_contain)
            if reply_ok:
                tally.replyMatchSuccesses += 1
        else:
            tally.unscoreableReturns += 1

        clean = (
            t_turn.turnDisposition == "replied"
            and right
            and args_all_correct
            and not (
                spurious_dup.spurious
                or spurious_dup.duplicateWithinTurn
                or spurious_dup.duplicateCrossTurn
            )
            and reply_ok
        )
        turn_clean.append(clean)
        prior_calls.extend(calls_this_turn)

    return turn_clean


def _determinism_probe(
    scored: Sequence[tuple[Conversation, ConversationTrace, Sequence[ItemTiming]]],
    probes: Sequence[tuple[Conversation, ConversationTrace, Sequence[ItemTiming]]],
    *,
    pack: Pack,
) -> Mapping[str, Any]:
    """Plan `:2262-2264`'s exact shape: `{scriptIds, ran, identical, differingTurns}`. `ran` mirrors
    `runner._drive_conversations`'s own definition (every declared probe script produced a trace);
    `identical` is `ran` AND every probe's `outcome_vectors_differ` against its same-`scriptId`
    counterpart in `scored` is empty — `False`, never vacuously `True`, when a probe is missing its
    counterpart entirely (a pack/runner defect, not a legitimate "identical")."""
    declared_ids = tuple(pack.manifest.get("sampling", {}).get("determinismProbeScripts", ()))
    scored_by_id = {script.scriptId: trace for script, trace, _ in scored}
    ran = len(probes) == len(declared_ids)

    differing: list[dict[str, Any]] = []
    all_matched = True
    for probe_script, probe_trace, _ in probes:
        counterpart = scored_by_id.get(probe_script.scriptId)
        if counterpart is None:
            all_matched = False
            continue
        diff = outcome_vectors_differ(probe_trace, counterpart)
        if diff:
            differing.append({"scriptId": probe_script.scriptId, "turns": list(diff)})

    identical = ran and all_matched and not differing
    return {
        "scriptIds": list(declared_ids),
        "ran": ran,
        "identical": identical,
        "differingTurns": differing,
    }


def score_conversations(
    scored: Sequence[tuple[Conversation, ConversationTrace, Sequence[ItemTiming]]],
    probes: Sequence[tuple[Conversation, ConversationTrace, Sequence[ItemTiming]]],
    *,
    pack: Pack,
) -> tuple[tuple[ItemResult, ...], ToolCallAggregates]:
    """The `ConversationScorer` Protocol method (S5 spec §3.4's own bullet): assembles every
    function above into the per-conversation `ItemResult`s and the run's `ToolCallAggregates`.

    `items` carries EXACTLY ONE `ItemResult` per `scored` conversation (§2.6's own load-bearing
    ruling, traced against `_paired_rows`/`_aggregate_item_mismatches`/`PairedOutcomes` — never one
    per turn). `pairingKey = (scriptId, replicate, H - 1)`, `itemId = scriptId`. `probes` are
    diagnostic only (plan `:2248`: "excluded from every denominator") and contribute nothing to
    `items`/`FunnelCounts`/`hazard` — their only consumer is `determinismProbe`, below.
    """
    h = pack.manifest["metrics"]["cleanThroughTurnH"]["H"]
    schemas = pack.prompt_config().toolSchemas

    tally = _Tally()
    traces: list[ConversationTrace] = []
    turn_clean_by_conversation: list[list[bool]] = []
    items: list[ItemResult] = []

    for script, trace, _timings in scored:
        turn_clean = _score_one_conversation(script, trace, schemas=schemas, tally=tally)
        traces.append(trace)
        turn_clean_by_conversation.append(turn_clean)

        state = clean_through_turn(trace, turn_clean, h=h)
        pairing_key = (script.scriptId, str(script.replicate), str(h - 1))
        if state == "clean":
            outcome, scoreable, counts = "pass", {"cleanThroughTurnH": True}, {
                "cleanThroughTurnH": 1
            }
        elif state == "failed":
            outcome, scoreable, counts = "fail", {"cleanThroughTurnH": True}, {
                "cleanThroughTurnH": 0
            }
        else:
            outcome, scoreable, counts = "unrunnable", {"cleanThroughTurnH": False}, {}
        items.append(
            ItemResult(
                itemId=script.scriptId,
                pairingKey=pairing_key,
                outcome=outcome,  # type: ignore[arg-type]
                scoreable=scoreable,
                counts=counts,
                timing=None,
            )
        )

    # The hazard/per-position curves run the FULL trace length (`h=None`) — a report artifact
    # distinct from `cleanThroughTurnH`'s own H-bounded headline verdict above (S5 spec §3.4:
    # `hazard_points`'s `h` "bounds how many positions are computed"; the headline's own H is a
    # different cut, already applied via `clean_through_turn` alone).
    hazard = hazard_points(traces, turn_clean_by_conversation, h=None)
    per_turn_position = tuple(
        TurnPositionRate(turnIndex=point.turnIndex, metric=point.metric) for point in hazard
    )

    clean_count = sum(1 for it in items if it.outcome == "pass")
    scoreable_count = sum(1 for it in items if it.scoreable.get("cleanThroughTurnH") is True)
    clean_through_turn_metric = (
        BinaryMetric(
            name="cleanThroughTurnH", successes=clean_count, n=scoreable_count,
            unit="conversation",
        )
        if scoreable_count
        else None
    )

    restraint_metric = (
        BinaryMetric(
            name="restraint", successes=tally.restraintSuccesses, n=tally.restraintTurns,
            unit="turn",
        )
        if tally.restraintTurns
        else None
    )

    aggregates = ToolCallAggregates(
        cleanThroughTurn=clean_through_turn_metric,
        perTurnPosition=per_turn_position,
        funnel=tally.funnel_metrics(),
        funnelCounts=tally.funnel_counts(),
        restraint=restraint_metric,
        hazard=hazard,
        determinismProbe=_determinism_probe(scored, probes, pack=pack),
    )
    return tuple(items), aggregates
