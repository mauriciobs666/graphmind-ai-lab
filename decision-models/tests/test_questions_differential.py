"""U6f.4b: every row of `fixtures/question-rules.jsonl` gets the same verdict from the bench's real
`modelbench.decisions._question_problem` as the vector says (`docs/plans/typed-decision-models4.md`
§3.11). The only test module that imports `modelbench`; skips only when `model-bench/` is absent.
An edit to the bench's rule, `_one_of`, `_nonempty_str` or `_POLICY_KIND_BY_QUESTION_TYPE` that
changes a verdict reddens; a reformat does not."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from decisionmodels.questions import question_problem

_MODEL_BENCH = Path(__file__).resolve().parents[2] / "model-bench"
_VECTOR = Path(__file__).parent / "fixtures" / "question-rules.jsonl"
_ROWS = [json.loads(line) for line in _VECTOR.read_text(encoding="utf-8").splitlines()[1:]]

pytestmark = pytest.mark.skipif(not _MODEL_BENCH.is_dir(), reason="model-bench/ absent")


@pytest.fixture(scope="module")
def bench_problem():
    sys.path.insert(0, str(_MODEL_BENCH))
    try:
        from modelbench.decisions import _question_problem
    finally:
        sys.path.remove(str(_MODEL_BENCH))
    return _question_problem


@pytest.mark.parametrize("row", _ROWS, ids=lambda r: r["case"])
def test_vector_verdict_equals_bench(bench_problem, row):
    assert (bench_problem(row["question"]) is None) is row["valid"]
    assert (bench_problem(row["question"]) is None) is (question_problem(row["question"]) is None)
