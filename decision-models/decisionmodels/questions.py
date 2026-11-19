"""The fit's question texts: loader, answer-space rule and the question-rule copy
(`docs/plans/typed-decision-models4.md` §3.11 "Question texts", §5 row U6f.4b).

`fit/<role>/questions.json` is `{"questions": [{"id", "control", "spec"}]}`: at most 5 questions,
exactly one `control`, ids unique, every `spec` valid under the question rules and the point's
answer space. `question_problem` is a copy of `model-bench/modelbench/decisions.py:
_question_problem` (FR-8 bars the import); `tests/test_questions_vector.py` pins it with a
hand-written vector and `tests/test_questions_differential.py` holds it equal to the bench's real
function. The answer-space rule is this loader's own: a predicate point takes only `predicate`
questions, a choice point only `choice` questions whose values equal the point's `expected`
vocabulary as a set.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decisionmodels.points import Point, load_point

#: The fit inputs' data directory, beside `points/` (plan4 §3.11).
FIT_DIR = Path(__file__).resolve().parents[1] / "fit"

#: A fit has at most this many question texts.
MAX_QUESTIONS = 5

#: The point kinds are the question types the bench's `_POLICY_KIND_BY_QUESTION_TYPE` accepts.
_QUESTION_TYPES = ("predicate", "choice")


class QuestionsRefused(ValueError):
    """A question file the fit cannot use; `rule` names the broken rule."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(message)
        self.rule = rule


@dataclass(frozen=True)
class Question:
    """One fit question: its id, whether it is the control, its spec as sent in `questions[]`."""

    id: str
    control: bool
    spec: Mapping[str, Any]


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _one_of(value: Any, allowed: Iterable[str]) -> bool:
    """Closed-set membership for a JSON value: a list or object is not a member, never a
    `TypeError`."""
    return isinstance(value, str) and value in allowed


def question_problem(question: Any) -> str | None:
    """The first problem with a question spec, or `None`: a copy of the bench's `_question_problem`
    (the verdict is what is pinned; the wording is free)."""
    if not isinstance(question, Mapping):
        return "question must be an object"
    kind = question.get("type")
    if kind == "score":
        return "question.type 'score' is refused"
    if not _one_of(kind, _QUESTION_TYPES):
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


def answer_space_problem(point: Point, question: Mapping[str, Any]) -> str | None:
    """The first way a rule-valid `question` falls outside `point`'s answer space, or `None`."""
    if question["type"] != point.kind:
        return f"a {point.kind} point takes only {point.kind} questions, got {question['type']!r}"
    if point.kind == "choice":
        values = {choice["value"] for choice in question["choices"]}
        if values != set(point.expected):
            return f"choice values {sorted(values)} differ from expected {sorted(point.expected)}"
    return None


def parse_questions(role: str, document: Any) -> tuple[Question, ...]:
    """The questions of a parsed `questions.json` for `role`, or `QuestionsRefused`."""
    point = load_point(role)
    entries = document.get("questions") if isinstance(document, Mapping) else None
    if not isinstance(entries, list):
        raise QuestionsRefused("shape", 'questions.json must be {"questions": [...]}')
    if not 1 <= len(entries) <= MAX_QUESTIONS:
        raise QuestionsRefused(
            "count", f"between 1 and {MAX_QUESTIONS} questions are required, got {len(entries)}"
        )
    questions: list[Question] = []
    for index, entry in enumerate(entries):
        where = f"questions[{index}]"
        if not isinstance(entry, Mapping) or set(entry) != {"id", "control", "spec"}:
            raise QuestionsRefused("shape", f'{where} must be exactly {{"id", "control", "spec"}}')
        if not _nonempty_str(entry["id"]):
            raise QuestionsRefused("shape", f"{where}.id must be a non-empty string")
        if not isinstance(entry["control"], bool):
            raise QuestionsRefused("shape", f"{where}.control must be a boolean")
        questions.append(Question(entry["id"], entry["control"], entry["spec"]))
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise QuestionsRefused("duplicate-id", f"question ids repeat: {ids}")
    controls = sum(q.control for q in questions)
    if controls != 1:
        raise QuestionsRefused(
            "control", f"exactly one control question is required, got {controls}"
        )
    for question in questions:
        problem = question_problem(question.spec)
        if problem is None:
            problem = answer_space_problem(point, question.spec)
            rule = "answer-space"
        else:
            rule = "question-rule"
        if problem is not None:
            raise QuestionsRefused(rule, f"question {question.id!r}: {problem}")
    return tuple(questions)


def load_questions(role: str, path: Path | None = None) -> tuple[Question, ...]:
    """`fit/<role>/questions.json` (or `path`), parsed and checked against `role`'s point."""
    source = path if path is not None else FIT_DIR / role / "questions.json"
    return parse_questions(role, json.loads(source.read_text(encoding="utf-8")))
