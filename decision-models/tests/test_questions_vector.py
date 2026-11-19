"""U6f.4b: `decisionmodels.questions.question_problem` matches every row of the hand-written
`fixtures/question-rules.jsonl` (`docs/plans/typed-decision-models4.md` §3.11, §5 row U6f.4b).
The vector is this component's own (not a model-bench copy); `test_questions_differential.py`
holds its verdicts equal to the bench's real `_question_problem`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decisionmodels.questions import question_problem

VECTOR = Path(__file__).parent / "fixtures" / "question-rules.jsonl"
ROWS = [json.loads(line) for line in VECTOR.read_text(encoding="utf-8").splitlines()[1:]]
_IDS = lambda r: r["case"]  # noqa: E731


def test_vector_has_both_verdicts_and_a_list_valued_type():
    assert len(ROWS) == 32
    assert {r["valid"] for r in ROWS} == {True, False}
    assert any(
        isinstance(r["question"], dict) and isinstance(r["question"].get("type"), list)
        for r in ROWS
    )


def agrees(row):
    return (question_problem(row["question"]) is None) is row["valid"]


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
def test_row_verdict(row):
    assert agrees(row)


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
def test_planted_wrong_verdict_reddens(row):
    assert not agrees({**row, "valid": not row["valid"]})
