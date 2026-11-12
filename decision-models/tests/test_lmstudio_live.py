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


def test_live_models_lists_a_model_with_a_state() -> None:
    """plan4 §3.11's unverified premise: `GET /api/v0/models` lists a served model, `laya`, with
    a `state`. Loads nothing; the raw row is read, so a catalog that lacks the row fails here
    rather than being read as 'not resident'."""
    import json
    import urllib.request

    with urllib.request.urlopen(f"{_BASE}/api/v0/models", timeout=10.0) as resp:
        rows = json.loads(resp.read())["data"]
    row = next((r for r in rows if r.get("id") == "laya"), None)
    assert row is not None, [r.get("id") for r in rows]
    assert isinstance(row["state"], str) and row["state"]
    resident = [m.id for m in LMStudio(_BASE).residency()]
    assert ("laya" in resident) == (row["state"] != "not-loaded")
