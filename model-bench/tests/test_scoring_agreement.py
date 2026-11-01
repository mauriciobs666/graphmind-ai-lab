"""`modelbench/scoring/agreement.py`'s contract half — typed-decision-models U4.1 (plan v1.3 §3.2,
§3.2.1, §3.6, §5 U4; plan2 §3.1, §4.1 U4; method note §Q1.5).

The four vector files under `tests/fixtures/decision-contract/` are model-bench's canonical copies
(v1.3 §3.2.2): `render-join-in.jsonl`, `render-message-type.jsonl`,
`handoff-parse-join-in.jsonl`, `handoff-parse-message-type.jsonl`. Their `expected` values were
written by hand, case by case, from the plans' canonical text (v1.3 §3.2.1's join-in template,
note §Q1.5's message-type template, v1.3 §3.2's parse column), never produced by this module.
falkor-chat has no production join-in or message-type renderer or parse to generate them from; U9
builds those against these files. decision-models byte-copies them (U5.2b).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modelbench.decisions import MESSAGE_TYPE_LABELS
from modelbench.scoring import agreement

_CONTRACT = Path(__file__).parent / "fixtures" / "decision-contract"


def _rows(name: str) -> list[dict[str, Any]]:
    with (_CONTRACT / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


_RENDER_JOIN_IN = _rows("render-join-in.jsonl")
_RENDER_MESSAGE_TYPE = _rows("render-message-type.jsonl")
_PARSE_JOIN_IN = _rows("handoff-parse-join-in.jsonl")
_PARSE_MESSAGE_TYPE = _rows("handoff-parse-message-type.jsonl")


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row["case"] for row in rows]


@pytest.mark.parametrize(
    ("name", "rows"),
    [
        ("render-join-in.jsonl", _RENDER_JOIN_IN),
        ("render-message-type.jsonl", _RENDER_MESSAGE_TYPE),
        ("handoff-parse-join-in.jsonl", _PARSE_JOIN_IN),
        ("handoff-parse-message-type.jsonl", _PARSE_MESSAGE_TYPE),
    ],
)
def test_every_vector_line_is_a_parametrized_row(name: str, rows: list[dict[str, Any]]):
    """The loader drops no row — counted from the raw bytes, not the loader (impl review P14-n1)."""
    lines = [line for line in (_CONTRACT / name).read_bytes().splitlines() if line.strip()]
    assert len(rows) == len(lines)
    assert len(set(_ids(rows))) == len(rows), "case names repeat"


# --- the renderers ------------------------------------------------------------------------------


@pytest.mark.parametrize("row", _RENDER_JOIN_IN, ids=_ids(_RENDER_JOIN_IN))
def test_render_join_in_input_matches_render_join_in_jsonl(row: dict[str, Any]):
    rendered = agreement.render_join_in_input(
        row["agentName"], row["priorTurns"], row["newMessage"]
    )
    assert rendered == row["expected"]


@pytest.mark.parametrize("row", _RENDER_MESSAGE_TYPE, ids=_ids(_RENDER_MESSAGE_TYPE))
def test_render_message_type_input_matches_render_message_type_jsonl(row: dict[str, Any]):
    assert agreement.render_message_type_input(row["text"], row["priorTurn"]) == row["expected"]


def _cases_with(rows: list[dict[str, Any]], *needles: str) -> list[str]:
    return [row["case"] for row in rows if all(n in row["case"] for n in needles)]


def test_the_join_in_vectors_cover_the_plans_cases():
    """v1.3 §3.2.1: the omitted block, `_recent_turns`' filter-then-last-6 rule, the speaker rule,
    and both 400 cuts, each at its boundary."""
    for needles in [
        ("no prior turns", "omitted"),
        ("filter first", "last 6"),
        ("speaker rule",),
        ("speaker rule on the new message", "authorId"),
        ("prior turn cut", "400", "401"),
        ("new message cut", "400 chars kept whole"),
        ("new message cut", "401 chars lose the last"),
    ]:
        assert _cases_with(_RENDER_JOIN_IN, *needles), needles


def test_the_message_type_vectors_cover_plan2s_cases():
    """Plan2 §4.1 U4: `priorTurn` null, an agent speaker, and the 400/2000 cuts at each side of
    the boundary."""
    assert any(row["priorTurn"] is None for row in _RENDER_MESSAGE_TYPE)
    assert any(
        row["priorTurn"] is not None and row["priorTurn"]["speaker"] == "salesperson"
        for row in _RENDER_MESSAGE_TYPE
    )
    for needles in [
        ("priorTurn cut", "400 chars kept whole"),
        ("priorTurn cut", "401 chars lose the last"),
        ("message cut", "2000 chars kept whole"),
        ("message cut", "2001 chars lose the last"),
    ]:
        assert _cases_with(_RENDER_MESSAGE_TYPE, *needles), needles


@pytest.mark.parametrize("text", [None, 5])
def test_both_renderers_raise_on_a_non_string_message_text(text: Any):
    """No plan coerces the rendered message's text, and decision-models' schema forces a non-empty
    string, so a malformed item fails loudly in both renderers alike rather than rendering `""`
    (impl review P21-n3). Prior turns differ: `_recent_turns`' filter drops them."""
    with pytest.raises(TypeError):
        agreement.render_join_in_input("Pastel", [], {"displayName": "Ana", "text": text})
    with pytest.raises(TypeError):
        agreement.render_message_type_input(text, None)


def test_the_cut_constants_are_the_plans():
    """v1.3 §3.2.1 (new message 400) and note §Q1.5 (prior turn 400, message 2000)."""
    assert (
        agreement.NEW_MESSAGE_MAX_CHARS,
        agreement.PRIOR_TURN_MAX_CHARS,
        agreement.MESSAGE_TEXT_MAX_CHARS,
    ) == (400, 400, 2000)


# --- the hand-off parses ------------------------------------------------------------------------


@pytest.mark.parametrize("row", _PARSE_JOIN_IN, ids=_ids(_PARSE_JOIN_IN))
def test_parse_join_in_handoff_matches_handoff_parse_join_in_jsonl(row: dict[str, Any]):
    expected = row["expected"]
    assert agreement.parse_join_in_handoff(row["reply"]) == (
        expected["reply"], expected["parseFailure"]
    )


@pytest.mark.parametrize("row", _PARSE_MESSAGE_TYPE, ids=_ids(_PARSE_MESSAGE_TYPE))
def test_parse_message_type_handoff_matches_handoff_parse_message_type_jsonl(row: dict[str, Any]):
    """The label set is FR-10's five (`decisions.MESSAGE_TYPE_LABELS`), as a message-type pack's
    answer space must declare (`ANSWER_SPACE_BY_ROLE`)."""
    expected = row["expected"]
    assert agreement.parse_message_type_handoff(row["reply"], MESSAGE_TYPE_LABELS) == (
        expected["type"], expected["parseFailure"]
    )


def test_every_label_has_an_accepting_row_and_an_out_of_space_label_has_a_refusing_one():
    """A label dropped from the accepted set, or one outside it accepted, reddens a row."""
    accepted = {row["expected"]["type"] for row in _PARSE_MESSAGE_TYPE}
    assert set(MESSAGE_TYPE_LABELS) <= accepted
    assert any(
        not row["expected"]["parseFailure"] and row["expected"]["type"] is None
        and json.loads(row["reply"]).get("type") not in MESSAGE_TYPE_LABELS
        and isinstance(json.loads(row["reply"]).get("type"), str)
        for row in _PARSE_MESSAGE_TYPE
        if row["reply"] and row["reply"].startswith("{") and row["reply"].endswith("}")
    )


def test_both_parses_have_a_strict_true_and_a_non_bool_row():
    """v1.3 §3.2: join-in replies only on a real bool `True`; the parse rows include a string
    `"true"` and an integer `1` that are not replies."""
    by_reply = {row["reply"]: row["expected"]["reply"] for row in _PARSE_JOIN_IN}
    assert by_reply['{"reply": true}'] is True
    assert by_reply['{"reply": "true"}'] is False
    assert any(r.startswith('{"reply": 1') for r in by_reply if r)


def test_the_message_type_parse_takes_the_packs_labels_not_a_built_in_set():
    reply = '{"type": "smalltalk"}'
    assert agreement.parse_message_type_handoff(reply, MESSAGE_TYPE_LABELS) == (None, False)
    assert agreement.parse_message_type_handoff(reply, ("smalltalk",)) == ("smalltalk", False)


# --- per-class agreement metric names (v1.3 §3.6) -----------------------------------------------


def test_a_predicate_answer_space_has_the_yes_and_no_agreement_metrics():
    assert agreement.per_class_agreement_metrics({"type": "predicate"}) == (
        "agreeWhenYes", "agreeWhenNo",
    )


def test_a_choice_answer_space_has_one_agreement_metric_per_label_in_label_order():
    space = {"type": "choice", "labels": list(MESSAGE_TYPE_LABELS)}
    assert agreement.per_class_agreement_metrics(space) == (
        "agree[order]", "agree[question]", "agree[complaint]", "agree[escalation]", "agree[other]",
    )


@pytest.mark.parametrize(
    "space",
    [{}, {"type": "score"}, {"type": "choice"}, {"type": "choice", "labels": []},
     {"type": "choice", "labels": "order"}],
)
def test_a_malformed_answer_space_has_no_agreement_metrics(space):
    with pytest.raises(ValueError):
        agreement.per_class_agreement_metrics(space)
