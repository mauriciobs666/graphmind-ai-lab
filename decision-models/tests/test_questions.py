"""U6f.4b: the question loader's refusals and answer-space rule
(`docs/plans/typed-decision-models4.md` §3.11, §5 row U6f.4b). The per-spec rules are pinned by
`test_questions_vector.py`."""

from __future__ import annotations

import copy
import json

import pytest

from decisionmodels.points import load_point
from decisionmodels.questions import QuestionsRefused, load_questions, parse_questions

GUARD = "guard-judge"
CHOICE = "message-type-classifier"


def predicate(qid, control=False):
    return {
        "id": qid,
        "control": control,
        "spec": {"type": "predicate", "name": "advance", "instructions": "ok?"},
    }


def choice(qid, values, control=False):
    spec = {
        "type": "choice",
        "name": "kind",
        "instructions": "which?",
        "choices": [{"value": v, "instructions": ""} for v in values],
    }
    return {"id": qid, "control": control, "spec": spec}


def doc(*entries):
    return {"questions": list(entries)}


def refusal(role, document):
    with pytest.raises(QuestionsRefused) as caught:
        parse_questions(role, document)
    return caught.value.rule


def test_valid_predicate_file_loads_in_order():
    questions = parse_questions(GUARD, doc(predicate("a", True), predicate("b")))
    assert [(q.id, q.control) for q in questions] == [("a", True), ("b", False)]


def test_five_questions_load():
    entries = [predicate("q0", True)] + [predicate(f"q{i}") for i in range(1, 5)]
    assert len(parse_questions(GUARD, doc(*entries))) == 5


def test_six_texts_refused():
    entries = [predicate("q0", True)] + [predicate(f"q{i}") for i in range(1, 6)]
    assert refusal(GUARD, doc(*entries)) == "count"


def test_no_questions_refused():
    assert refusal(GUARD, doc()) == "count"


def test_no_control_refused():
    assert refusal(GUARD, doc(predicate("a"), predicate("b"))) == "control"


def test_two_controls_refused():
    assert refusal(GUARD, doc(predicate("a", True), predicate("b", True))) == "control"


def test_duplicate_id_refused():
    assert refusal(GUARD, doc(predicate("a", True), predicate("a"))) == "duplicate-id"


def test_choice_question_on_a_predicate_point_refused():
    assert refusal(GUARD, doc(choice("a", ["true", "false"], True))) == "answer-space"


def test_predicate_question_on_a_choice_point_refused():
    assert refusal(CHOICE, doc(predicate("a", True))) == "answer-space"


def test_choice_values_equal_expected_as_a_set_in_any_order():
    expected = list(load_point(CHOICE).expected)
    questions = parse_questions(CHOICE, doc(choice("a", expected[::-1], True)))
    assert len(questions) == 1


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(["order", "question", "complaint", "escalation"], id="missing"),
        pytest.param(
            ["order", "question", "complaint", "escalation", "other", "extra"], id="extra"
        ),
        pytest.param(["order", "question", "complaint", "escalation", "Other"], id="renamed"),
    ],
)
def test_choice_values_differing_from_expected_refused(values):
    assert refusal(CHOICE, doc(choice("a", values, True))) == "answer-space"


def test_a_spec_the_question_rules_refuse_is_refused_before_the_answer_space():
    bad = predicate("a", True)
    bad["spec"]["type"] = "score"
    assert refusal(GUARD, doc(bad)) == "question-rule"


@pytest.mark.parametrize(
    "document",
    [
        pytest.param([], id="not-an-object"),
        pytest.param({}, id="no-questions-key"),
        pytest.param({"questions": {}}, id="questions-not-list"),
        pytest.param(doc("a"), id="entry-not-object"),
        pytest.param(doc({"id": "a", "control": True}), id="entry-missing-spec"),
        pytest.param(doc({**predicate("a", True), "extra": 1}), id="entry-extra-key"),
        pytest.param(doc({**predicate("a", True), "id": ""}), id="empty-id"),
        pytest.param(doc({**predicate("a", True), "id": 1}), id="non-string-id"),
        pytest.param(doc({**predicate("a", True), "control": 1}), id="control-not-bool"),
    ],
)
def test_malformed_files_refused_as_shape(document):
    assert refusal(GUARD, document) == "shape"


def test_unknown_role_is_a_value_error():
    with pytest.raises(ValueError):
        parse_questions("nope", doc(predicate("a", True)))


def test_parse_does_not_mutate_its_input():
    document = doc(predicate("a", True))
    before = copy.deepcopy(document)
    parse_questions(GUARD, document)
    assert document == before


def test_load_questions_reads_a_file(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(doc(predicate("a", True))), encoding="utf-8")
    assert load_questions(GUARD, path)[0].id == "a"
