"""The 20-case transcribed agreement fixture (`docs/plans/small-model-benchmarking.md` §5 item 11,
`docs/plans/small-model-benchmarking-s3-spec.md` §8 Step 1, item 2).

Reads only `tests/fixtures/metrics_agreement.json` — never `falkor-chat/`, matching plan §3.1 point
2(b)'s discipline for this exact fixture ("the default suite still passes with `falkor-chat/`
renamed away"). Every case in the fixture is asserted, including the two `ValueError` cases — a
test that skips or xfails any case is a failing test (plan §5 item 11): the case count is the
guarantee, so this parametrizes over every entry the fixture holds rather than a fixed literal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelbench.scoring import retrieval

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "metrics_agreement.json"


def _load_cases() -> list[dict]:
    data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    return data["cases"]


_CASES = _load_cases()


@pytest.mark.parametrize("case", _CASES, ids=[f"{c['function']}::{c['case']}" for c in _CASES])
def test_agrees_with_the_transcribed_case(case: dict) -> None:
    fn = getattr(retrieval, case["function"])
    args = dict(case["args"])

    if "expectedError" in case:
        error_type = {"ValueError": ValueError}[case["expectedError"]]
        with pytest.raises(error_type):
            fn(**args)
        return

    result = fn(**args)
    assert result == pytest.approx(case["expected"])


def test_the_fixture_holds_exactly_twenty_cases() -> None:
    """The case count is the guarantee (plan §5 item 11) — a fixture that silently lost a case
    would still pass every remaining one."""
    assert len(_CASES) == 20


def test_every_case_names_a_real_retrieval_function() -> None:
    for case in _CASES:
        assert hasattr(retrieval, case["function"]), (
            f"fixture case {case['case']!r} names {case['function']!r}, which "
            "modelbench.scoring.retrieval does not define"
        )
