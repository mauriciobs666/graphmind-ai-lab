"""`decisionmodels.handoff` (plan4 §4.4 U6f.3a; §3.11): every row of the three byte-copies of
model-bench's `handoff-parse-*.jsonl`, plus the parse's own edges. Rows carry the raw reply."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from decisionmodels.handoff import (
    extract_own_line_json_object,
    parse_guard_handoff,
    parse_join_in_handoff,
    parse_message_type_handoff,
)

_DIR = Path(__file__).resolve().parent / "fixtures" / "decision-contract"
_LABELS = ("order", "question", "complaint", "escalation", "other")


def _rows(name: str) -> list[dict[str, Any]]:
    path = _DIR / f"handoff-parse-{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


_GUARD = _rows("guard")
_JOIN_IN = _rows("join-in")
_MESSAGE_TYPE = _rows("message-type")


def test_the_copies_have_every_golden_row() -> None:
    assert (len(_GUARD), len(_JOIN_IN), len(_MESSAGE_TYPE)) == (99, 22, 26)


@pytest.mark.parametrize("row", _GUARD, ids=[r["case"] for r in _GUARD])
def test_guard_rows(row: dict[str, Any]) -> None:
    assert parse_guard_handoff(row["reply"]) == (
        row["expected"]["advance"],
        row["expected"]["parseFailure"],
    )


@pytest.mark.parametrize("row", _JOIN_IN, ids=[r["case"] for r in _JOIN_IN])
def test_join_in_rows(row: dict[str, Any]) -> None:
    assert parse_join_in_handoff(row["reply"]) == (
        row["expected"]["reply"],
        row["expected"]["parseFailure"],
    )


@pytest.mark.parametrize("row", _MESSAGE_TYPE, ids=[r["case"] for r in _MESSAGE_TYPE])
def test_message_type_rows(row: dict[str, Any]) -> None:
    assert parse_message_type_handoff(row["reply"], _LABELS) == (
        row["expected"]["type"],
        row["expected"]["parseFailure"],
    )


def test_message_type_label_is_judged_against_the_labels_passed() -> None:
    assert parse_message_type_handoff('{"type": "order"}', ("other",)) == (None, False)


def test_extract_returns_the_object_not_a_verdict() -> None:
    assert extract_own_line_json_object('x\n  {"a": {"b": "}"}}\n') == {"a": {"b": "}"}}
    assert extract_own_line_json_object('x\n{"a": 1}\n', require_key="b") is None


def test_one_line_fenced_object_is_unwrapped() -> None:
    assert parse_join_in_handoff('```json{"reply": true}```') == (True, False)


def test_object_ending_a_sentence_is_quoted_not_asserted() -> None:
    assert parse_join_in_handoff('I would say {"reply": true}') == (False, True)


def test_non_string_content_is_a_parse_failure_whatever_it_prints_as() -> None:
    class Printed:
        def __str__(self) -> str:
            return '{"reply": true}'

    assert parse_join_in_handoff(Printed()) == (False, True)


def test_nested_own_line_object_is_not_a_second_match() -> None:
    reply = 'ok\n{"reply": true, "meta":\n {"reply": false}\n}\n'
    assert parse_join_in_handoff(reply) == (True, False)


def test_upper_case_json_fence_is_unwrapped() -> None:
    assert parse_join_in_handoff('```JSON{"reply": true}```') == (True, False)


def test_escaped_quote_before_a_brace_does_not_end_the_string() -> None:
    assert parse_join_in_handoff('x\n{"reply": true, "s": "\\"}"}\n') == (True, False)
