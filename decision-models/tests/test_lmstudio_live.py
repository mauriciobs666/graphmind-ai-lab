"""`pytest -m live`: the client against a real LM Studio (deselected by default, human-run).

`DM_LMSTUDIO_URL` is the server (default `http://localhost:1234`). Needs `laya` served there.
"""

from __future__ import annotations

import os

import pytest

from decisionmodels.lmstudio import LMStudio, OverLimit

pytestmark = pytest.mark.live

_BASE = os.environ.get("DM_LMSTUDIO_URL", "http://localhost:1234")
_QUESTION = {"type": "predicate", "name": "advance", "instructions": "Is this a greeting?"}


def test_live_decide_answers_a_predicate() -> None:
    result = LMStudio(_BASE).decide("Hello there!", [_QUESTION], model="laya", timeout_s=60.0)
    assert not isinstance(result, OverLimit)
    assert 0.0 <= result.answers[0].probability <= 1.0  # type: ignore[operator]


def test_live_over_limit_body_is_recognised() -> None:
    """An input far over laya's 512-token physical batch is a hard 500, not a truncation
    (plan3 §3.3; the method-change tripwire of U7's confirm step)."""
    long_input = "the quick brown fox jumps over the lazy dog " * 400
    result = LMStudio(_BASE).decide(long_input, [_QUESTION], model="laya", timeout_s=60.0)
    assert isinstance(result, OverLimit)
