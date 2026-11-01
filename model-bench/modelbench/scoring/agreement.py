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

This module is not yet a scorer: `score_item`/`aggregate` and the decision hooks
(`decision_input`, `answer_from_chat`, `score_answer`) are typed-decision-models U4.2's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from modelbench.scoring.classification import _normalize_turns, extract_own_line_json_object

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
