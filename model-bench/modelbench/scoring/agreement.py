"""The new-ability decision points' contract half — `join-in-judge` and `message-type-classifier`
(typed-decision-models plan v1.3 §3.2, §3.2.1, §3.6; plan2 §3.1; method note §Q1.5).

Pure functions, transcribed into falkor-chat (U9) and decision-models (U5.2b) and held equal by
golden vectors, never a shared library (FR-23, v1.3 §3.2.2):

* `render_join_in_input` — `render-join-in.jsonl`;
* `render_message_type_input` — `render-message-type.jsonl`;
* `parse_join_in_handoff` — `handoff-parse-join-in.jsonl`;
* `parse_message_type_handoff` — `handoff-parse-message-type.jsonl`;
* `per_class_agreement_metrics` — the FR-10 agreement metric names a new-ability pack declares as
  its `verdictMetrics` (`packs.validate_pack` refuses any other set).

The vectors' expected values were written by hand from the plans' canonical text, never produced
by this module. falkor-chat has no join-in or message-type renderer or parse yet (U9 builds them
against the same vectors), so there is no production code to generate them from.

**The scorer** (typed-decision-models U4.2; v1.3 §3.6; plan2 §3.3; plan3 §3.3): the new-ability
roles' `ItemScorer` — a module, like `classification`. Each `eval` item declares one metric, its
true answer's class (`agreeWhenYes`/`agreeWhenNo`, `agree[<label>]`), counted `1` when the final
answer equals the label. A kept failed call (`failed_call="decision"` or `"handoff"`) is scored on
the role's fail-closed answer (`FAIL_CLOSED_ANSWER_BY_ROLE`), never on `outcome`; an over-limit
item handed off is scored on the hand-off LLM's answer. The hand-off chat prompt is pack data
(`prompts/join-in-handoff.md`, `prompts/message-type-handoff.md`, the pack's `systemPrompt`).
The FR-10 bar's input — note §Q4.3's drops over the candidate run — is
`decision_stats.agreement_items`, not this module's aggregate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from modelbench.decisions import ANSWER_SPACE_BY_ROLE, BAR_KIND_BY_ROLE
from modelbench.lmstudio import ChatResult
from modelbench.packs import Pack
from modelbench.results import BinaryMetric, ClassificationAggregates, ItemResult, ItemTiming
from modelbench.runner import ChatMessage
from modelbench.scoring.classification import (
    _CALL_FAILURE_WITHHELD,
    _DECIDED_BY,
    _FAILED_CALLS,
    _REASONS,
    _normalize_turns,
    extract_own_line_json_object,
)

#: v1.3 §3.2.1: the join-in new message is cut to 400 chars (prior turns are cut by
#: `_normalize_turns`, `guards._recent_turns`' rule).
NEW_MESSAGE_MAX_CHARS = 400
#: Note §Q1.5: the message-type `priorTurn` text is cut to 400 chars.
PRIOR_TURN_MAX_CHARS = 400
#: v1.3 §3.2.1 / note §Q1.5: the classified message's text is cut to 2000 chars.
MESSAGE_TEXT_MAX_CHARS = 2000

#: v1.3 §3.2: the hand-off reply's verdict key, per new-ability role.
JOIN_IN_HANDOFF_KEY = "reply"
MESSAGE_TYPE_HANDOFF_KEY = "type"

#: v1.3 §3.6: a predicate point's two FR-10 agreement metrics, by true answer.
PREDICATE_AGREEMENT_METRICS: Mapping[bool, str] = {True: "agreeWhenYes", False: "agreeWhenNo"}


def _speaker(row: Mapping[str, Any]) -> str:
    """`guards._recent_turns`' speaker rule (note §Q1.5: join-in uses the same rule)."""
    return str(row.get("displayName") or row.get("authorId") or "member")


def render_join_in_input(
    agent_name: str, prior_turns: Sequence[Mapping[str, Any]], new_message: Mapping[str, Any]
) -> str:
    """v1.3 §3.2.1, verbatim: `AGENT: {agent_name}\\n\\nRECENT TURNS (context only):\\n`
    `{speaker}: {text}\\n…\\n\\nNEW MESSAGE:\\n{speaker}: {text}`.

    `prior_turns` are thread rows strictly before the new message, normalized by
    `guards._recent_turns`' rule (filter first, keep the last 6, each cut to 400 chars); the block
    is omitted when none survive. The new message takes the same speaker rule, its text cut to
    `NEW_MESSAGE_MAX_CHARS`; it is not filtered, so a missing or non-string text raises, as in
    `render_message_type_input` (impl review P21-n3)."""
    blocks = [f"AGENT: {agent_name}"]
    turns = _normalize_turns(prior_turns)
    if turns:
        lines = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in turns)
        blocks.append(f"RECENT TURNS (context only):\n{lines}")
    text = new_message["text"][:NEW_MESSAGE_MAX_CHARS]
    blocks.append(f"NEW MESSAGE:\n{_speaker(new_message)}: {text}")
    return "\n\n".join(blocks)


def render_message_type_input(text: str, prior_turn: Mapping[str, Any] | None) -> str:
    """Plan2 §3.1 / note §Q1.5, verbatim:
    `PREVIOUS MESSAGE:\\n{speaker}: {priorTurn.text[:400]}\\n\\nMESSAGE:\\n{text[:2000]}`; the first
    block is omitted when `prior_turn` is `None`. `prior_turn` is `{speaker, text}`, its speaker
    already chosen by `guards._recent_turns`' rule (it may be the agent)."""
    message = f"MESSAGE:\n{text[:MESSAGE_TEXT_MAX_CHARS]}"
    if prior_turn is None:
        return message
    previous = f"{prior_turn['speaker']}: {prior_turn['text'][:PRIOR_TURN_MAX_CHARS]}"
    return f"PREVIOUS MESSAGE:\n{previous}\n\n{message}"


def parse_join_in_handoff(content: Any) -> tuple[bool, bool]:
    """v1.3 §3.2's join-in hand-off parse: `(reply, parse_failure)`.
    `extract_own_line_json_object(require_key="reply")`; only a real bool `True` replies, and
    anything else — a `"true"` string, `1`, no object — is no reply (FR-12's fail-closed side)."""
    parsed = extract_own_line_json_object(content, require_key=JOIN_IN_HANDOFF_KEY)
    if parsed is None:
        return False, True
    return parsed.get(JOIN_IN_HANDOFF_KEY) is True, False


def parse_message_type_handoff(content: Any, labels: Sequence[str]) -> tuple[str | None, bool]:
    """v1.3 §3.2's message-type hand-off parse: `(label, parse_failure)`.
    `extract_own_line_json_object(require_key="type")`, and the value must be one of `labels`
    exactly (case included); otherwise there is no label (`None`). `labels` is the pack's answer
    space, required: the parse never guesses it."""
    parsed = extract_own_line_json_object(content, require_key=MESSAGE_TYPE_HANDOFF_KEY)
    if parsed is None:
        return None, True
    value = parsed.get(MESSAGE_TYPE_HANDOFF_KEY)
    return (value if isinstance(value, str) and value in labels else None), False


def per_class_agreement_metrics(answer_space: Mapping[str, Any]) -> tuple[str, ...]:
    """v1.3 §3.6: the FR-10 per-class agreement metrics of an answer space — a predicate's
    `agreeWhenYes`/`agreeWhenNo`, a choice's `agree[<label>]` per label, in label order. A
    malformed answer space raises `ValueError`."""
    kind = answer_space.get("type")
    if kind == "predicate":
        return (PREDICATE_AGREEMENT_METRICS[True], PREDICATE_AGREEMENT_METRICS[False])
    labels = answer_space.get("labels")
    if kind == "choice" and isinstance(labels, (list, tuple)) and labels:
        return tuple(f"agree[{label}]" for label in labels)
    raise ValueError(f"not a predicate or choice answer space: {dict(answer_space)!r}")


# --- the scorer (typed-decision-models U4.2) ----------------------------------------------------

#: Plan2 §3.3 / note §Q4.3: the answer a new-ability role's runtime acts on when a call fails — no
#: reply on join-in, no label on message type (a disagreement for every true class). Keyed exactly
#: like `BAR_KIND_BY_ROLE`'s `new-ability` rows.
FAIL_CLOSED_ANSWER_BY_ROLE: Mapping[str, bool | None] = MappingProxyType(
    {"join-in-judge": False, "message-type-classifier": None}
)


def _answer_space(pack: Pack) -> Mapping[str, Any]:
    """The pack's answer space: its role's (`ANSWER_SPACE_BY_ROLE`), which `validate_pack` holds
    the declared `decisionPoint.answer` equal to. Only a new-ability role has one here."""
    if BAR_KIND_BY_ROLE.get(pack.role) != "new-ability":
        raise ValueError(f"agreement scores new-ability roles only, not {pack.role!r}")
    return ANSWER_SPACE_BY_ROLE[pack.role]


def _is_answer(value: Any, answer_space: Mapping[str, Any]) -> bool:
    """A value inside the answer space: a real bool on a predicate, one of the labels on a
    choice."""
    if answer_space["type"] == "predicate":
        return isinstance(value, bool)
    return isinstance(value, str) and value in answer_space["labels"]


def class_metric(item_input: Mapping[str, Any], *, pack: Pack) -> str:
    """The one FR-10 metric an item declares: its true answer's class (v1.3 §3.6). Only the `eval`
    stratum exists on a new-ability pack (FR-15), and its `expected` must be inside the answer
    space. `validate_pack` refuses both before a run, so a raise here is a run of an unvalidated
    pack."""
    space = _answer_space(pack)
    item_id = item_input.get("itemId")
    if item_input.get("stratum") != "eval":
        raise ValueError(
            f"item {item_id!r}: stratum {item_input.get('stratum')!r} is not 'eval' (a new-ability "
            "pack has no side check, FR-15)"
        )
    expected = item_input.get("expected")
    if not _is_answer(expected, space):
        raise ValueError(f"item {item_id!r}: expected {expected!r} is outside the answer space")
    if space["type"] == "predicate":
        return PREDICATE_AGREEMENT_METRICS[expected]
    return f"agree[{expected}]"


def decision_input(item_input: Mapping[str, Any], *, pack: Pack) -> str:
    """The `ItemScorer.decision_input` hook: the role's §3.2.1 renderer over the item's input
    fields (`INPUT_FIELDS_BY_ROLE`, every one present on a valid pack)."""
    _answer_space(pack)
    if pack.role == "join-in-judge":
        return render_join_in_input(
            item_input["agentName"], item_input["priorTurns"], item_input["newMessage"]
        )
    return render_message_type_input(item_input["text"], item_input["priorTurn"])


def build_messages(item_input: Mapping[str, Any], *, pack: Pack) -> list[ChatMessage]:
    """The hand-off chat call (v1.3 §3.2): the pack's hand-off prompt as the system message
    (`prompts/<point>-handoff.md`, the manifest's `systemPrompt`), and the same rendered input the
    decision model sees as the user message. Never reads `expected`."""
    return [
        {"role": "system", "content": pack.prompt_config().systemPrompt},
        {"role": "user", "content": decision_input(item_input, pack=pack)},
    ]


def answer_from_chat(result: ChatResult, *, pack: Pack) -> tuple[bool | str | None, str]:
    """The role's hand-off parse (v1.3 §3.2), as `(answer, outcome)`. An unparseable reply is
    `"parse_failure"` with the parse's own fallback — no reply on join-in, no label on message
    type; a parsed reply is `"pass"`, its label `None` when outside the answer space."""
    space = _answer_space(pack)
    content = result.message.get("content")
    if space["type"] == "predicate":
        answer, failed = parse_join_in_handoff(content)
    else:
        answer, failed = parse_message_type_handoff(content, space["labels"])
    return answer, "parse_failure" if failed else "pass"


def _decision_detail(
    answer: Any,
    *,
    decided_by: str | None,
    probability: float | None,
    failed_call: str | None,
    reason: str | None,
    input_tokens: int | None,
    withheld_for: str | None,
) -> dict[str, Any]:
    """The decision-surface `detail` keys, `classification`'s names, or `{}` on a chat-arm item
    (every keyword `None`). Every state the plans cannot produce raises `ValueError`:

    - a new ability never withholds a failed call (`HANDOFF_FAILURE_BY_ROLE`: `fail-closed`), so a
      `withheldFor` of `"timeout"`/`"no_response"` never reaches here;
    - `failed_call` (`"decision"` or `"handoff"`) has no answer, `decided_by` or `probability`
      (coordination ruling (b)); a decision failure has no `input_tokens`; a hand-off failure
      carries its decision call's, `None` when over-limit (plan3 §3.3: no decision call returned);
    - `decided_by="decision-model"` has a `probability` and no `reason`;
    - `decided_by="handoff"` has a `probability` after a policy hand-off, and neither it nor
      `input_tokens` when over-limit;
    - with neither, no other keyword."""
    if decided_by is not None and decided_by not in _DECIDED_BY:
        raise ValueError(
            f"score_answer: decided_by {decided_by!r} is not one of {sorted(_DECIDED_BY)}"
        )
    if failed_call is not None and failed_call not in _FAILED_CALLS:
        raise ValueError(
            f"score_answer: failed_call {failed_call!r} is not one of {sorted(_FAILED_CALLS)}"
        )
    if reason is not None and reason not in _REASONS:
        raise ValueError(f"score_answer: reason {reason!r} is not one of {sorted(_REASONS)}")
    if withheld_for in _CALL_FAILURE_WITHHELD:
        raise ValueError(
            f"score_answer: withheldFor {withheld_for!r} — a new ability keeps a failed call "
            "(HANDOFF_FAILURE_BY_ROLE: fail-closed), it never withholds one"
        )
    over_limit = reason is not None
    if failed_call is not None:
        if answer is not None:
            raise ValueError("score_answer: a failed call has no answer; it is scored fail-closed")
        if decided_by is not None or probability is not None:
            raise ValueError("score_answer: a failed call has no decided_by or probability")
        if input_tokens is not None and (failed_call == "decision" or over_limit):
            raise ValueError("score_answer: input_tokens needs a decision call that returned")
    elif decided_by == "decision-model":
        if over_limit:
            raise ValueError("score_answer: an over-limit item is decided_by 'handoff' only")
        if probability is None:
            raise ValueError("score_answer: a decided item has the decision call's probability")
    elif decided_by == "handoff":
        if over_limit and (probability is not None or input_tokens is not None):
            raise ValueError("score_answer: an over-limit hand-off has no probability or tokens")
        if not over_limit and probability is None:
            raise ValueError("score_answer: a policy hand-off has the decision call's probability")
    elif probability is not None or over_limit or input_tokens is not None:
        raise ValueError(
            "score_answer: probability, reason and input_tokens need a decided_by or failed_call"
        )
    detail: dict[str, Any] = {}
    if decided_by is not None:
        detail["decidedBy"] = decided_by
    if failed_call is not None:
        detail["failedCall"] = failed_call
    if detail:
        detail["probability"] = probability
        detail["inputTokens"] = input_tokens
    if reason is not None:
        detail["reason"] = reason
    return detail


def score_answer(
    item_input: Mapping[str, Any],
    answer: bool | str | None,
    outcome: str,
    timing: ItemTiming,
    *,
    pack: Pack,
    decided_by: str | None,
    probability: float | None,
    failed_call: str | None,
    reason: str | None,
    input_tokens: int | None,
) -> ItemResult:
    """One item's FR-10 record, wherever its answer came from (v1.3 §3.3 "Scorer seam").

    The item declares its class metric (`class_metric`), scoreable, counted `1` iff the **final
    answer** equals `expected`. The final answer is `answer`, or — on a kept failed call
    (`failed_call` set, `answer=None`) — the role's `FAIL_CLOSED_ANSWER_BY_ROLE` answer (plan2
    §3.3). `outcome` is carried through and never read: a failed item's `"fail"` is not its score
    (coordination ruling (b)). An over-limit hand-off (`decided_by="handoff"`,
    `reason="over-limit"`) is scored on the hand-off LLM's answer, never fail-closed (plan3 §4
    U4). `answer` is the role's answer type: a real bool on join-in; a label on message type, or
    `None` for a reply with no label. `detail` carries `answer` (the final answer, which the
    replicate disagreement line reads on message type) and `_decision_detail`'s keys. The five
    keywords have no default."""
    space = _answer_space(pack)
    metric = class_metric(item_input, pack=pack)
    detail = _decision_detail(
        answer,
        decided_by=decided_by,
        probability=probability,
        failed_call=failed_call,
        reason=reason,
        input_tokens=input_tokens,
        withheld_for=timing.withheldFor,
    )
    if failed_call is not None:
        final = FAIL_CLOSED_ANSWER_BY_ROLE[pack.role]
    elif _is_answer(answer, space) or (answer is None and space["type"] == "choice"):
        final = answer
    else:
        raise ValueError(f"score_answer: answer {answer!r} is outside the answer space")
    item_id = item_input["itemId"]
    return ItemResult(
        itemId=item_id,
        pairingKey=(item_id,),
        outcome=outcome,
        scoreable={metric: True},
        counts={metric: int(final == item_input["expected"])},
        timing=timing,
        detail={"answer": final, **detail},
    )


def score_item(
    item_input: Mapping[str, Any], result: ChatResult | None, timing: ItemTiming, *, pack: Pack,
) -> ItemResult:
    """One chat-arm item (the hand-off LLM run alone, U7's reference arm): `answer_from_chat`, then
    `score_answer` with every decision keyword `None`. `result is None` — a chat call that never
    answered — is withheld as on every chat arm (`classification.score_item`'s rule): outcome
    `"fail"`/`"unrunnable"` by `timing.withheldFor`, `scoreable={}`. Only a decision surface's kept
    hand-off failure is scored fail-closed."""
    if result is None:
        item_id = item_input["itemId"]
        return ItemResult(
            itemId=item_id,
            pairingKey=(item_id,),
            outcome="fail" if timing.withheldFor == "timeout" else "unrunnable",
            scoreable={},
            counts={},
            timing=timing,
        )
    answer, outcome = answer_from_chat(result, pack=pack)
    return score_answer(
        item_input,
        answer,
        outcome,
        timing,
        pack=pack,
        decided_by=None,
        probability=None,
        failed_call=None,
        reason=None,
        input_tokens=None,
    )


def aggregate(items: Sequence[ItemResult], *, pack: Pack) -> ClassificationAggregates:
    """`perClass` is the answer space's per-class agreement metrics, in its label order, every one
    present (`n=0` when no item declared it): `successes` the agreeing items, `n` every item that
    declared the class — kept failures (fail-closed) and load-withheld items included, as the
    guard's aggregate includes them. The bar's drops are `decision_stats.agreement_items`'.
    `parseFailures` counts `"parse_failure"` outcomes; `n` is every item. The runner appends the
    decision-surface counts to `perClass` (`runner._with_decision_call_metrics`)."""
    names = per_class_agreement_metrics(_answer_space(pack))
    per_class = tuple(
        BinaryMetric(
            name=name,
            successes=sum(it.counts[name] for it in items if it.scoreable.get(name)),
            n=sum(1 for it in items if it.scoreable.get(name)),
            unit="item",
        )
        for name in names
    )
    parse_failures = sum(1 for it in items if it.outcome == "parse_failure")
    return ClassificationAggregates(perClass=per_class, parseFailures=parse_failures, n=len(items))
