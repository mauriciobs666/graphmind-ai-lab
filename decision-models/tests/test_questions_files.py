"""U6f.4c: the shipped `fit/<role>/questions.json` files load through the loader and carry no
forbidden fragment and no side-check shingle (`docs/plans/typed-decision-models4.md` §3.11
"Question texts", §5 row U6f.4c)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_fr8_isolation import FORBIDDEN_FRAGMENTS
from test_points import _shingles, _side_check_shingles

from decisionmodels.questions import FIT_DIR, QuestionsRefused, load_questions

ROLES = ("guard-judge", "join-in-judge", "message-type-classifier")


def _forbidden_in(fit_dir: Path) -> list[tuple[str, str]]:
    return [
        (str(path.relative_to(fit_dir)), name)
        for path in sorted(fit_dir.rglob("questions.json"))
        for name, pattern in FORBIDDEN_FRAGMENTS.items()
        if pattern.search(path.read_text(encoding="utf-8").lower())
    ]


def _shared_with(side: set[tuple[str, ...]], fit_dir: Path) -> list[str]:
    return [
        str(path.relative_to(fit_dir))
        for path in sorted(fit_dir.rglob("questions.json"))
        if _shingles(path.read_text(encoding="utf-8")) & side
    ]


def _copy_fit(tmp_path: Path) -> Path:
    fit = tmp_path / "fit"
    for role in ROLES:
        (fit / role).mkdir(parents=True)
        (fit / role / "questions.json").write_bytes(
            (FIT_DIR / role / "questions.json").read_bytes()
        )
    return fit


@pytest.mark.parametrize("role", ROLES)
def test_each_questions_file_loads_for_its_point(role: str) -> None:
    questions = load_questions(role)
    assert 1 <= len(questions) <= 5
    assert sum(q.control for q in questions) == 1


@pytest.mark.parametrize("role", ROLES)
def test_a_questions_file_is_refused_for_another_points_answer_space(role: str) -> None:
    other = "message-type-classifier" if role != "message-type-classifier" else "guard-judge"
    with pytest.raises(QuestionsRefused) as refused:
        load_questions(other, FIT_DIR / role / "questions.json")
    assert refused.value.rule == "answer-space"


def test_no_questions_file_carries_a_forbidden_fragment() -> None:
    assert _forbidden_in(FIT_DIR) == []


@pytest.mark.parametrize("fragment", sorted(FORBIDDEN_FRAGMENTS))
def test_a_planted_forbidden_fragment_in_questions_is_refused(
    fragment: str, tmp_path: Path
) -> None:
    fit = _copy_fit(tmp_path)
    assert _forbidden_in(fit) == []
    target = fit / ROLES[0] / "questions.json"
    document = json.loads(target.read_text(encoding="utf-8"))
    document["questions"][0]["spec"]["instructions"] += f" {fragment.upper()}"
    target.write_text(json.dumps(document), encoding="utf-8")
    assert _forbidden_in(fit) == [(f"{ROLES[0]}/questions.json", fragment)]


def test_no_questions_text_shares_an_eight_token_span_with_a_side_check_item() -> None:
    assert _shared_with(_side_check_shingles(), FIT_DIR) == []


def test_a_planted_side_check_shingle_in_questions_is_refused(tmp_path: Path) -> None:
    side = _side_check_shingles()
    assert side
    fit = _copy_fit(tmp_path)
    assert _shared_with(side, fit) == []
    planted = " ".join(sorted(side)[0])
    target = fit / ROLES[1] / "questions.json"
    document = json.loads(target.read_text(encoding="utf-8"))
    document["questions"][1]["spec"]["instructions"] += f" {planted}"
    target.write_text(json.dumps(document), encoding="utf-8")
    assert _shared_with(side, fit) == [f"{ROLES[1]}/questions.json"]
